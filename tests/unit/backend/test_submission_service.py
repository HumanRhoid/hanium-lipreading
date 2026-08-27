"""영상 저장과 Redis 추론 Job 등록 조정 서비스 테스트."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.ports import (
    InferenceJobEnqueueResult,
    InferenceJobRecord,
    VideoAssetRecord,
)
from src.backend.recognition.submission_service import RecognitionSubmissionService
from src.backend.recognition.video_upload_service import VideoUploadResult

STORAGE_UUID = UUID("11111111-2222-4333-8444-555555555555")

JOB_UUID = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")

IDEMPOTENCY_KEY = "12345678-1234-4234-8234-123456789abc"

QUEUED_AT = datetime(
    2026,
    8,
    27,
    16,
    30,
    tzinfo=UTC,
)


def make_asset() -> VideoAssetRecord:
    return VideoAssetRecord(
        video_id=45,
        utterance_id=123,
        user_id=7,
        idempotency_key=IDEMPOTENCY_KEY,
        object_key=(
            f"{STORAGE_UUID}/2026/08/bbbbbbbb-cccc-4ddd-8eee-ffffffffffff.webm"
        ),
        original_mime_type="video/webm",
        size_bytes=5,
        checksum="a" * 64,
        storage_status="UPLOADED",
        storage_purpose="TEMPORARY_INFERENCE",
        created_at=QUEUED_AT,
        retention_until=None,
    )


def make_job(
    *,
    job_id: str = str(JOB_UUID),
) -> InferenceJobRecord:
    asset = make_asset()

    return InferenceJobRecord(
        job_id=job_id,
        utterance_id=asset.utterance_id,
        video_id=asset.video_id,
        object_key=asset.object_key,
        mode=RecognitionMode.CLOSED,
        status="QUEUED",
        created_at=QUEUED_AT,
        updated_at=QUEUED_AT,
        error_code=None,
    )


class FakeVideoUploadService:
    def __init__(
        self,
        *,
        result: VideoUploadResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "user_id": user_id,
                "storage_uuid": storage_uuid,
                "idempotency_key": idempotency_key,
                "data": data,
                "content_type": content_type,
                "mode": mode,
            }
        )

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError("영상 업로드 결과가 설정되지 않았습니다.")

        return self.result


class FakeInferenceJobQueue:
    def __init__(
        self,
        *,
        result: InferenceJobEnqueueResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def enqueue_or_get(
        self,
        *,
        job_id: str,
        utterance_id: int,
        video_id: int,
        object_key: str,
        mode: RecognitionMode,
        created_at: datetime,
    ) -> InferenceJobEnqueueResult:
        self.calls.append(
            {
                "job_id": job_id,
                "utterance_id": utterance_id,
                "video_id": video_id,
                "object_key": object_key,
                "mode": mode,
                "created_at": created_at,
            }
        )

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError("추론 Job enqueue 결과가 설정되지 않았습니다.")

        return self.result

    async def get_job(
        self,
        *,
        job_id: str,
    ) -> InferenceJobRecord | None:
        del job_id

        return None


async def test_submit_uploads_video_and_enqueues_new_job():
    asset = make_asset()
    job = make_job()

    upload_service = FakeVideoUploadService(
        result=VideoUploadResult(
            asset=asset,
            duplicate=False,
        )
    )

    job_queue = FakeInferenceJobQueue(
        result=InferenceJobEnqueueResult(
            job=job,
            created=True,
        )
    )

    service = RecognitionSubmissionService(
        video_upload_service=upload_service,
        inference_job_queue=job_queue,
        clock=lambda: QUEUED_AT,
        uuid_factory=lambda: JOB_UUID,
    )

    result = await service.submit(
        user_id=7,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=b"video",
        content_type="video/webm",
        mode=RecognitionMode.CLOSED,
    )

    assert result.asset == asset
    assert result.job == job
    assert result.duplicate is False

    assert len(upload_service.calls) == 1
    assert len(job_queue.calls) == 1

    upload_call = upload_service.calls[0]

    assert upload_call["user_id"] == 7
    assert upload_call["storage_uuid"] == STORAGE_UUID
    assert upload_call["idempotency_key"] == IDEMPOTENCY_KEY
    assert upload_call["data"] == b"video"
    assert upload_call["content_type"] == "video/webm"
    assert upload_call["mode"] is RecognitionMode.CLOSED

    queue_call = job_queue.calls[0]

    assert queue_call["job_id"] == str(JOB_UUID)
    assert queue_call["utterance_id"] == asset.utterance_id
    assert queue_call["video_id"] == asset.video_id
    assert queue_call["object_key"] == asset.object_key
    assert queue_call["mode"] is RecognitionMode.CLOSED
    assert queue_call["created_at"] == QUEUED_AT


async def test_submit_returns_existing_job_for_duplicate_video():
    asset = make_asset()
    existing_job = make_job()

    upload_service = FakeVideoUploadService(
        result=VideoUploadResult(
            asset=asset,
            duplicate=True,
        )
    )

    job_queue = FakeInferenceJobQueue(
        result=InferenceJobEnqueueResult(
            job=existing_job,
            created=False,
        )
    )

    service = RecognitionSubmissionService(
        video_upload_service=upload_service,
        inference_job_queue=job_queue,
        clock=lambda: QUEUED_AT,
        uuid_factory=lambda: JOB_UUID,
    )

    result = await service.submit(
        user_id=7,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=b"video",
        content_type="video/webm",
        mode=RecognitionMode.CLOSED,
    )

    assert result.asset == asset
    assert result.job == existing_job
    assert result.duplicate is True


async def test_submit_can_enqueue_after_previous_redis_failure():
    asset = make_asset()
    new_job = make_job()

    upload_service = FakeVideoUploadService(
        result=VideoUploadResult(
            asset=asset,
            duplicate=True,
        )
    )

    job_queue = FakeInferenceJobQueue(
        result=InferenceJobEnqueueResult(
            job=new_job,
            created=True,
        )
    )

    service = RecognitionSubmissionService(
        video_upload_service=upload_service,
        inference_job_queue=job_queue,
        clock=lambda: QUEUED_AT,
        uuid_factory=lambda: JOB_UUID,
    )

    result = await service.submit(
        user_id=7,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=b"video",
        content_type="video/webm",
        mode=RecognitionMode.CLOSED,
    )

    assert result.asset == asset
    assert result.job == new_job

    assert result.duplicate is True

    assert len(job_queue.calls) == 1


async def test_submit_marks_duplicate_when_queue_already_has_job():
    asset = make_asset()
    existing_job = make_job()

    upload_service = FakeVideoUploadService(
        result=VideoUploadResult(
            asset=asset,
            duplicate=False,
        )
    )

    job_queue = FakeInferenceJobQueue(
        result=InferenceJobEnqueueResult(
            job=existing_job,
            created=False,
        )
    )

    service = RecognitionSubmissionService(
        video_upload_service=upload_service,
        inference_job_queue=job_queue,
        clock=lambda: QUEUED_AT,
        uuid_factory=lambda: JOB_UUID,
    )

    result = await service.submit(
        user_id=7,
        storage_uuid=STORAGE_UUID,
        idempotency_key=IDEMPOTENCY_KEY,
        data=b"video",
        content_type="video/webm",
        mode=RecognitionMode.CLOSED,
    )

    assert result.duplicate is True
    assert result.job == existing_job


async def test_upload_failure_does_not_enqueue_job():
    upload_error = RuntimeError("object storage failed")

    upload_service = FakeVideoUploadService(error=upload_error)

    job_queue = FakeInferenceJobQueue(
        result=InferenceJobEnqueueResult(
            job=make_job(),
            created=True,
        )
    )

    service = RecognitionSubmissionService(
        video_upload_service=upload_service,
        inference_job_queue=job_queue,
        clock=lambda: QUEUED_AT,
        uuid_factory=lambda: JOB_UUID,
    )

    with pytest.raises(
        RuntimeError,
        match="object storage failed",
    ):
        await service.submit(
            user_id=7,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type="video/webm",
            mode=RecognitionMode.CLOSED,
        )

    assert job_queue.calls == []


async def test_queue_failure_is_propagated_after_video_storage():
    asset = make_asset()

    upload_service = FakeVideoUploadService(
        result=VideoUploadResult(
            asset=asset,
            duplicate=False,
        )
    )

    queue_error = RuntimeError("redis unavailable")

    job_queue = FakeInferenceJobQueue(error=queue_error)

    service = RecognitionSubmissionService(
        video_upload_service=upload_service,
        inference_job_queue=job_queue,
        clock=lambda: QUEUED_AT,
        uuid_factory=lambda: JOB_UUID,
    )

    with pytest.raises(
        RuntimeError,
        match="redis unavailable",
    ):
        await service.submit(
            user_id=7,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type="video/webm",
            mode=RecognitionMode.CLOSED,
        )

    assert len(upload_service.calls) == 1
    assert len(job_queue.calls) == 1


async def test_naive_queue_time_is_rejected_before_enqueue():
    asset = make_asset()

    upload_service = FakeVideoUploadService(
        result=VideoUploadResult(
            asset=asset,
            duplicate=False,
        )
    )

    job_queue = FakeInferenceJobQueue(
        result=InferenceJobEnqueueResult(
            job=make_job(),
            created=True,
        )
    )

    naive_time = datetime(
        2026,
        8,
        27,
        16,
        30,
    )

    service = RecognitionSubmissionService(
        video_upload_service=upload_service,
        inference_job_queue=job_queue,
        clock=lambda: naive_time,
        uuid_factory=lambda: JOB_UUID,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        await service.submit(
            user_id=7,
            storage_uuid=STORAGE_UUID,
            idempotency_key=IDEMPOTENCY_KEY,
            data=b"video",
            content_type="video/webm",
            mode=RecognitionMode.CLOSED,
        )

    assert len(upload_service.calls) == 1
    assert job_queue.calls == []
