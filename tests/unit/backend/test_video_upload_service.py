"""비동기 영상 업로드 서비스의 검증과 보상 처리를 테스트한다."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.errors import (
    EmptyVideoUploadError,
    IdempotencyConflictError,
    InvalidIdempotencyKeyError,
    UnsupportedVideoMimeTypeError,
    UnsupportedVideoUploadModeError,
    VideoTooLargeError,
)
from src.backend.recognition.ports import VideoAssetRecord, VideoAssetSaveResult
from src.backend.recognition.video_upload_service import VideoUploadService

FIXED_NOW = datetime(
    2026,
    8,
    27,
    15,
    0,
    tzinfo=UTC,
)

STORAGE_UUID = UUID("11111111-2222-4333-8444-555555555555")

OBJECT_UUID = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")

IDEMPOTENCY_KEY = "12345678-1234-4234-8234-123456789abc"


class FakeObjectStorage:
    def __init__(self) -> None:
        self.ensure_bucket_calls = 0
        self.put_calls: list[dict[str, object]] = []
        self.deleted_keys: list[str] = []

        self.put_error: BaseException | None = None
        self.delete_error: Exception | None = None

    async def ensure_bucket(self) -> None:
        self.ensure_bucket_calls += 1

    async def put(
        self,
        *,
        object_key: str,
        data: bytes,
        content_type: str,
        checksum: str,
    ) -> None:
        if self.put_error is not None:
            raise self.put_error

        self.put_calls.append(
            {
                "object_key": object_key,
                "data": data,
                "content_type": content_type,
                "checksum": checksum,
            }
        )

    async def get(
        self,
        object_key: str,
    ) -> bytes:
        raise NotImplementedError

    async def exists(
        self,
        object_key: str,
    ) -> bool:
        raise NotImplementedError

    async def delete(
        self,
        object_key: str,
    ) -> None:
        self.deleted_keys.append(object_key)

        if self.delete_error is not None:
            raise self.delete_error


class FakeVideoUploadRepository:
    def __init__(self) -> None:
        self.existing_asset: VideoAssetRecord | None = None
        self.save_result: VideoAssetSaveResult | None = None
        self.save_error: BaseException | None = None

        self.find_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []

    async def find_video_asset_by_idempotency_key(
        self,
        *,
        user_id: int,
        idempotency_key: str,
    ) -> VideoAssetRecord | None:
        self.find_calls.append(
            {
                "user_id": user_id,
                "idempotency_key": idempotency_key,
            }
        )

        return self.existing_asset

    async def create_or_get_video_asset(
        self,
        *,
        user_id: int,
        idempotency_key: str,
        object_key: str,
        original_mime_type: str,
        size_bytes: int,
        checksum: str,
        storage_purpose: str,
        consent_version: str | None,
        retention_until: datetime | None,
    ) -> VideoAssetSaveResult:
        self.create_calls.append(
            {
                "user_id": user_id,
                "idempotency_key": idempotency_key,
                "object_key": object_key,
                "original_mime_type": original_mime_type,
                "size_bytes": size_bytes,
                "checksum": checksum,
                "storage_purpose": storage_purpose,
                "consent_version": consent_version,
                "retention_until": retention_until,
            }
        )

        if self.save_error is not None:
            raise self.save_error

        if self.save_result is None:
            raise AssertionError("save_result가 설정되지 않았습니다.")

        return self.save_result


def make_asset(
    *,
    object_key: str,
    mime_type: str,
    data: bytes,
    checksum: str,
) -> VideoAssetRecord:
    return VideoAssetRecord(
        video_id=10,
        utterance_id=20,
        user_id=1,
        idempotency_key=IDEMPOTENCY_KEY,
        object_key=object_key,
        original_mime_type=mime_type,
        size_bytes=len(data),
        checksum=checksum,
        storage_status="UPLOADED",
        storage_purpose="TEMPORARY_INFERENCE",
        created_at=FIXED_NOW,
        retention_until=None,
    )


def make_service(
    repository: FakeVideoUploadRepository,
    object_storage: FakeObjectStorage,
    *,
    max_upload_bytes: int = 1024,
) -> VideoUploadService:
    return VideoUploadService(
        repository=repository,
        object_storage=object_storage,
        max_upload_bytes=max_upload_bytes,
        clock=lambda: FIXED_NOW,
        uuid_factory=lambda: OBJECT_UUID,
    )


async def test_upload_stores_object_and_metadata():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    data = b"example-video"

    expected_object_key = f"{STORAGE_UUID}/2026/08/{OBJECT_UUID}.webm"

    expected_checksum = (
        "0059a143a40dc3c01b769d079e00acd29ec94efaa718b1359390e6e305eaee1f"
    )

    asset = make_asset(
        object_key=expected_object_key,
        mime_type="video/webm",
        data=data,
        checksum=expected_checksum,
    )

    repository.save_result = VideoAssetSaveResult(
        asset=asset,
        created=True,
    )

    service = make_service(
        repository,
        storage,
    )

    result = await service.upload(
        user_id=1,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=data,
        content_type="video/webm",
        mode=RecognitionMode.CLOSED,
    )

    assert result.duplicate is False
    assert result.asset == asset

    assert storage.ensure_bucket_calls == 1
    assert len(storage.put_calls) == 1

    put_call = storage.put_calls[0]

    assert put_call["object_key"] == expected_object_key

    assert put_call["data"] == data

    assert put_call["content_type"] == "video/webm"

    assert put_call["checksum"] == expected_checksum

    assert len(repository.create_calls) == 1

    create_call = repository.create_calls[0]

    assert create_call["object_key"] == expected_object_key

    assert create_call["storage_purpose"] == "TEMPORARY_INFERENCE"

    assert create_call["consent_version"] is None

    retention_until = create_call["retention_until"]

    assert retention_until is not None

    assert (retention_until - FIXED_NOW).total_seconds() == 24 * 60 * 60


async def test_existing_identical_request_does_not_upload_again():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    data = b"same-video"

    checksum = "658b8b098537a8c88834f50bda20293eec7bf11829555630a323eba6b5d50402"

    existing_asset = make_asset(
        object_key=(f"{STORAGE_UUID}/2026/08/existing.webm"),
        mime_type="video/webm",
        data=data,
        checksum=checksum,
    )

    repository.existing_asset = existing_asset

    service = make_service(
        repository,
        storage,
    )

    result = await service.upload(
        user_id=1,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=data,
        content_type="video/webm",
        mode=RecognitionMode.CLOSED,
    )

    assert result.duplicate is True
    assert result.asset == existing_asset

    assert storage.ensure_bucket_calls == 0

    assert storage.put_calls == []
    assert repository.create_calls == []


async def test_same_idempotency_key_with_different_file_is_rejected():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    existing_data = b"first-video"

    existing_asset = make_asset(
        object_key=(f"{STORAGE_UUID}/2026/08/existing.mp4"),
        mime_type="video/mp4",
        data=existing_data,
        checksum="0" * 64,
    )

    repository.existing_asset = existing_asset

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(IdempotencyConflictError):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"different-video",
            content_type="video/mp4",
            mode=RecognitionMode.CLOSED,
        )

    assert storage.ensure_bucket_calls == 0

    assert storage.put_calls == []


async def test_racing_duplicate_removes_loser_object():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    data = b"race-video"

    checksum = "0833e43a40d82d14af8b3afea618bbdd71322f741e933bf733ac30f06925073e"

    winning_asset = make_asset(
        object_key=(f"{STORAGE_UUID}/2026/08/winner.webm"),
        mime_type="video/webm",
        data=data,
        checksum=checksum,
    )

    repository.save_result = VideoAssetSaveResult(
        asset=winning_asset,
        created=False,
    )

    service = make_service(
        repository,
        storage,
    )

    result = await service.upload(
        user_id=1,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=data,
        content_type="video/webm",
        mode=RecognitionMode.CLOSED,
    )

    uploaded_key = f"{STORAGE_UUID}/2026/08/{OBJECT_UUID}.webm"

    assert result.duplicate is True
    assert result.asset == winning_asset

    assert storage.deleted_keys == [uploaded_key]


async def test_database_failure_removes_uploaded_object():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    repository.save_error = RuntimeError("database failed")

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(
        RuntimeError,
        match="database failed",
    ):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type="video/mp4",
            mode=RecognitionMode.CLOSED,
        )

    expected_object_key = f"{STORAGE_UUID}/2026/08/{OBJECT_UUID}.mp4"

    assert storage.deleted_keys == [expected_object_key]


async def test_object_storage_failure_does_not_create_database_row():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    storage.put_error = RuntimeError("storage failed")

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(
        RuntimeError,
        match="storage failed",
    ):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type="video/mp4",
            mode=RecognitionMode.CLOSED,
        )

    assert repository.create_calls == []


async def test_cleanup_failure_does_not_hide_database_failure():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    repository.save_error = RuntimeError("database failed")

    storage.delete_error = RuntimeError("delete failed")

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(
        RuntimeError,
        match="database failed",
    ):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type="video/mp4",
            mode=RecognitionMode.CLOSED,
        )


@pytest.mark.parametrize(
    "content_type",
    [
        "text/plain",
        "application/octet-stream",
        "image/jpeg",
        "",
    ],
)
async def test_unsupported_mime_type_is_rejected(
    content_type,
):
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(UnsupportedVideoMimeTypeError):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type=content_type,
            mode=RecognitionMode.CLOSED,
        )

    assert storage.put_calls == []


async def test_content_type_parameters_are_normalized():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    data = b"video"

    expected_key = f"{STORAGE_UUID}/2026/08/{OBJECT_UUID}.webm"

    asset = make_asset(
        object_key=expected_key,
        mime_type="video/webm",
        data=data,
        checksum="0" * 64,
    )

    repository.save_result = VideoAssetSaveResult(
        asset=asset,
        created=True,
    )

    service = make_service(
        repository,
        storage,
    )

    result = await service.upload(
        user_id=1,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=data,
        content_type=" Video/WebM; codecs=vp9 ",
        mode=RecognitionMode.CLOSED,
    )

    assert result.duplicate is False

    assert storage.put_calls[0]["content_type"] == "video/webm"


async def test_empty_file_is_rejected():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(EmptyVideoUploadError):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"",
            content_type="video/mp4",
            mode=RecognitionMode.CLOSED,
        )

    assert storage.put_calls == []


async def test_oversized_file_is_rejected():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    service = make_service(
        repository,
        storage,
        max_upload_bytes=4,
    )

    with pytest.raises(VideoTooLargeError):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"12345",
            content_type="video/mp4",
            mode=RecognitionMode.CLOSED,
        )

    assert storage.put_calls == []


@pytest.mark.parametrize(
    "idempotency_key",
    [
        "",
        "not-a-uuid",
        "1234",
    ],
)
async def test_invalid_idempotency_key_is_rejected(
    idempotency_key,
):
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(InvalidIdempotencyKeyError):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=idempotency_key,
            data=b"video",
            content_type="video/mp4",
            mode=RecognitionMode.CLOSED,
        )

    assert repository.find_calls == []
    assert storage.put_calls == []


async def test_idempotency_key_is_canonicalized():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    data = b"video"

    expected_key = f"{STORAGE_UUID}/2026/08/{OBJECT_UUID}.mp4"

    asset = make_asset(
        object_key=expected_key,
        mime_type="video/mp4",
        data=data,
        checksum="0" * 64,
    )

    repository.save_result = VideoAssetSaveResult(
        asset=asset,
        created=True,
    )

    service = make_service(
        repository,
        storage,
    )

    await service.upload(
        user_id=1,
        storage_uuid=STORAGE_UUID,
        idempotency_key=("12345678-1234-4234-8234-123456789ABC"),
        data=data,
        content_type="video/mp4",
        mode=RecognitionMode.CLOSED,
    )

    assert repository.find_calls[0]["idempotency_key"] == IDEMPOTENCY_KEY


async def test_open_mode_upload_is_rejected():
    repository = FakeVideoUploadRepository()
    storage = FakeObjectStorage()

    service = make_service(
        repository,
        storage,
    )

    with pytest.raises(UnsupportedVideoUploadModeError):
        await service.upload(
            user_id=1,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type="video/mp4",
            mode=RecognitionMode.OPEN,
        )

    assert storage.put_calls == []
