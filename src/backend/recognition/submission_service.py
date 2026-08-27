"""영상 저장과 비동기 추론 Job 등록을 조정하는 서비스."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.ports import (
    InferenceJobQueue,
    InferenceJobRecord,
    VideoAssetRecord,
)
from src.backend.recognition.video_upload_service import VideoUploadService


@dataclass(frozen=True, slots=True)
class RecognitionSubmissionResult:
    """영상 저장 및 추론 Job 등록 결과."""

    asset: VideoAssetRecord
    job: InferenceJobRecord
    duplicate: bool


class RecognitionSubmissionService:
    """영상 저장 후 실제 Redis 추론 Queue에 Job을 등록한다."""

    def __init__(
        self,
        *,
        video_upload_service: VideoUploadService,
        inference_job_queue: InferenceJobQueue,
        clock: Callable[[], datetime] | None = None,
        uuid_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._video_upload_service = video_upload_service
        self._inference_job_queue = inference_job_queue
        self._clock = clock or _utc_now
        self._uuid_factory = uuid_factory

    async def submit(
        self,
        *,
        user_id: int,
        storage_uuid: UUID,
        idempotency_key: str,
        data: bytes,
        content_type: str,
        mode: RecognitionMode,
    ) -> RecognitionSubmissionResult:
        """영상을 저장하고 해당 영상의 추론 Job을 한 번만 enqueue한다."""

        upload_result = await self._video_upload_service.upload(
            user_id=user_id,
            storage_uuid=storage_uuid,
            idempotency_key=idempotency_key,
            data=data,
            content_type=content_type,
            mode=mode,
        )

        queued_at = self._clock()

        if queued_at.tzinfo is None or queued_at.utcoffset() is None:
            raise ValueError(
                "추론 Job 생성 시간은 timezone-aware datetime이어야 합니다."
            )

        enqueue_result = await self._inference_job_queue.enqueue_or_get(
            job_id=str(self._uuid_factory()),
            utterance_id=upload_result.asset.utterance_id,
            video_id=upload_result.asset.video_id,
            object_key=upload_result.asset.object_key,
            mode=mode,
            created_at=queued_at,
        )

        return RecognitionSubmissionResult(
            asset=upload_result.asset,
            job=enqueue_result.job,
            duplicate=(upload_result.duplicate or not enqueue_result.created),
        )


def _utc_now() -> datetime:
    """현재 UTC 시간을 timezone-aware datetime으로 반환한다."""

    return datetime.now(UTC)
