"""인식 영상 업로드, Object Storage 저장, DB 기록을 조정한다."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import UUID, uuid4

from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.errors import (
    EmptyVideoUploadError,
    IdempotencyConflictError,
    InvalidIdempotencyKeyError,
    UnsupportedVideoMimeTypeError,
    UnsupportedVideoUploadModeError,
    VideoTooLargeError,
)
from src.backend.recognition.ports import (
    ObjectStorage,
    VideoAssetRecord,
    VideoAssetSaveResult,
    VideoUploadRepository,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_VIDEO_UPLOAD_BYTES = 64 * 1024 * 1024
TEMPORARY_VIDEO_RETENTION = timedelta(hours=24)

SUPPORTED_VIDEO_MIME_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class VideoUploadResult:
    """영상 업로드 처리 결과."""

    asset: VideoAssetRecord
    duplicate: bool


class VideoUploadService:
    """영상 검증부터 Object Storage와 DB 기록까지 조정한다."""

    def __init__(
        self,
        *,
        repository: VideoUploadRepository,
        object_storage: ObjectStorage,
        max_upload_bytes: int = DEFAULT_MAX_VIDEO_UPLOAD_BYTES,
        clock: Callable[[], datetime] = _utc_now,
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes는 양수여야 합니다.")

        self._repository = repository
        self._object_storage = object_storage
        self._max_upload_bytes = max_upload_bytes
        self._clock = clock
        self._uuid_factory = uuid_factory

    async def upload(
        self,
        *,
        user_id: int,
        storage_uuid: UUID,
        idempotency_key: str,
        data: bytes,
        content_type: str,
        mode: RecognitionMode,
    ) -> VideoUploadResult:
        """영상을 검증하고 private Object Storage와 DB에 기록한다."""

        canonical_idempotency_key = self._validate_idempotency_key(idempotency_key)

        mime_type, extension = self._validate_video(
            data=data,
            content_type=content_type,
            mode=mode,
        )

        checksum = sha256(data).hexdigest()

        existing_asset = await self._repository.find_video_asset_by_idempotency_key(
            user_id=user_id,
            idempotency_key=canonical_idempotency_key,
        )

        if existing_asset is not None:
            self._ensure_same_request(
                existing_asset,
                mime_type=mime_type,
                size_bytes=len(data),
                checksum=checksum,
            )

            return VideoUploadResult(
                asset=existing_asset,
                duplicate=True,
            )

        now = self._clock()

        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock은 timezone-aware datetime을 반환해야 합니다.")

        object_key = self._build_object_key(
            storage_uuid=storage_uuid,
            created_at=now,
            extension=extension,
        )

        retention_until = now + TEMPORARY_VIDEO_RETENTION

        await self._object_storage.ensure_bucket()

        await self._object_storage.put(
            object_key=object_key,
            data=data,
            content_type=mime_type,
            checksum=checksum,
        )

        try:
            save_result = await self._repository.create_or_get_video_asset(
                user_id=user_id,
                idempotency_key=canonical_idempotency_key,
                object_key=object_key,
                original_mime_type=mime_type,
                size_bytes=len(data),
                checksum=checksum,
                storage_purpose="TEMPORARY_INFERENCE",
                consent_version=None,
                retention_until=retention_until,
            )
        except BaseException:
            await self._delete_uploaded_object_safely(object_key)
            raise

        return await self._finish_save(
            save_result=save_result,
            uploaded_object_key=object_key,
            mime_type=mime_type,
            size_bytes=len(data),
            checksum=checksum,
        )

    def _validate_video(
        self,
        *,
        data: bytes,
        content_type: str,
        mode: RecognitionMode,
    ) -> tuple[str, str]:
        if mode is not RecognitionMode.CLOSED:
            raise UnsupportedVideoUploadModeError(
                "현재 영상 업로드는 CLOSED mode만 지원합니다."
            )

        if not data:
            raise EmptyVideoUploadError("빈 영상 파일은 업로드할 수 없습니다.")

        if len(data) > self._max_upload_bytes:
            raise VideoTooLargeError("영상 파일 크기가 허용 범위를 초과했습니다.")

        mime_type = content_type.split(";", maxsplit=1)[0].strip().lower()

        extension = SUPPORTED_VIDEO_MIME_TYPES.get(mime_type)

        if extension is None:
            raise UnsupportedVideoMimeTypeError(
                "지원하는 영상 형식은 video/mp4와 video/webm입니다."
            )

        return (
            mime_type,
            extension,
        )

    @staticmethod
    def _validate_idempotency_key(
        value: str,
    ) -> str:
        try:
            parsed = UUID(value.strip())
        except (
            AttributeError,
            ValueError,
        ) as exc:
            raise InvalidIdempotencyKeyError(
                "Idempotency-Key는 UUID 형식이어야 합니다."
            ) from exc

        return str(parsed)

    def _build_object_key(
        self,
        *,
        storage_uuid: UUID,
        created_at: datetime,
        extension: str,
    ) -> str:
        object_uuid = self._uuid_factory()

        return (
            f"{storage_uuid}/{created_at:%Y}/{created_at:%m}/{object_uuid}{extension}"
        )

    @staticmethod
    def _ensure_same_request(
        asset: VideoAssetRecord,
        *,
        mime_type: str,
        size_bytes: int,
        checksum: str,
    ) -> None:
        if (
            asset.original_mime_type != mime_type
            or asset.size_bytes != size_bytes
            or asset.checksum != checksum
        ):
            raise IdempotencyConflictError(
                "같은 Idempotency-Key를 서로 다른 영상 요청에 재사용할 수 없습니다."
            )

    async def _finish_save(
        self,
        *,
        save_result: VideoAssetSaveResult,
        uploaded_object_key: str,
        mime_type: str,
        size_bytes: int,
        checksum: str,
    ) -> VideoUploadResult:
        if save_result.created:
            return VideoUploadResult(
                asset=save_result.asset,
                duplicate=False,
            )

        # 동시에 같은 Idempotency-Key 요청이 먼저 DB에 저장된 경우,
        # 이번 요청이 올린 별도 Object Storage 객체는 제거한다.
        if save_result.asset.object_key != uploaded_object_key:
            await self._delete_uploaded_object_safely(uploaded_object_key)

        self._ensure_same_request(
            save_result.asset,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
        )

        return VideoUploadResult(
            asset=save_result.asset,
            duplicate=True,
        )

    async def _delete_uploaded_object_safely(
        self,
        object_key: str,
    ) -> None:
        try:
            await self._object_storage.delete(object_key)
        except Exception as exc:
            logger.error(
                "영상 업로드 보상 삭제 실패: error_type=%s",
                type(exc).__name__,
            )
