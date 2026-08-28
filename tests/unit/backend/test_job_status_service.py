"""로그인 사용자의 추론 Job 상태 조회 서비스를 테스트한다."""

from datetime import UTC, datetime

import pytest

from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.job_status_service import (
    InferenceJobStatusService,
)
from src.backend.recognition.ports import (
    InferenceJobRecord,
    InferenceResultRecord,
    VideoAssetRecord,
)

JOB_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

CREATED_AT = datetime(
    2026,
    8,
    27,
    15,
    0,
    tzinfo=UTC,
)


def make_job(
    *,
    video_id: int = 45,
) -> InferenceJobRecord:
    return InferenceJobRecord(
        job_id=JOB_ID,
        utterance_id=123,
        video_id=video_id,
        object_key="storage-user/2026/08/video.webm",
        mode=RecognitionMode.CLOSED,
        status="QUEUED",
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
        error_code=None,
    )


def make_asset(
    *,
    video_id: int = 45,
    user_id: int = 7,
) -> VideoAssetRecord:
    return VideoAssetRecord(
        video_id=video_id,
        utterance_id=123,
        user_id=user_id,
        idempotency_key=("11111111-1111-4111-8111-111111111111"),
        object_key="storage-user/2026/08/video.webm",
        original_mime_type="video/webm",
        size_bytes=12345,
        checksum="a" * 64,
        storage_status="UPLOADED",
        storage_purpose="TEMPORARY_INFERENCE",
        created_at=CREATED_AT,
        retention_until=None,
    )


class FakeInferenceJobQueue:
    def __init__(
        self,
        *,
        job: InferenceJobRecord | None,
    ) -> None:
        self.job = job
        self.requested_job_ids: list[str] = []

    async def get_job(
        self,
        *,
        job_id: str,
    ) -> InferenceJobRecord | None:
        self.requested_job_ids.append(job_id)

        return self.job


class FakeVideoUploadRepository:
    def __init__(
        self,
        *,
        asset: VideoAssetRecord | None,
        result: InferenceResultRecord | None = None,
    ) -> None:
        self.asset = asset
        self.result = result
        self.requested_video_ids: list[int] = []
        self.requested_utterance_ids: list[int] = []

    async def find_video_asset_by_id(
        self,
        *,
        video_id: int,
    ) -> VideoAssetRecord | None:
        self.requested_video_ids.append(video_id)

        return self.asset

    async def get_inference_result(
        self,
        *,
        utterance_id: int,
    ) -> InferenceResultRecord | None:
        self.requested_utterance_ids.append(utterance_id)

        return self.result


async def test_get_for_user_returns_owned_job():
    job = make_job()

    queue = FakeInferenceJobQueue(
        job=job,
    )

    repository = FakeVideoUploadRepository(
        asset=make_asset(
            user_id=7,
        )
    )

    service = InferenceJobStatusService(
        repository=repository,
        inference_job_queue=queue,
    )

    result = await service.get_for_user(
        user_id=7,
        job_id=JOB_ID,
    )

    assert result is not None
    assert result.job == job
    assert result.result is None

    assert queue.requested_job_ids == [JOB_ID]

    assert repository.requested_video_ids == [job.video_id]
    assert repository.requested_utterance_ids == []


async def test_get_for_user_returns_result_for_succeeded_job():
    job = make_job()
    job = InferenceJobRecord(
        job_id=job.job_id,
        utterance_id=job.utterance_id,
        video_id=job.video_id,
        object_key=job.object_key,
        mode=job.mode,
        status="SUCCEEDED",
        created_at=job.created_at,
        updated_at=job.updated_at,
        error_code=None,
    )
    inference_result = InferenceResultRecord(
        utterance_id=job.utterance_id,
        text="물 주세요",
        phrase_code="REQUEST_WATER",
        confidence=0.91,
        model_version="fake-v1",
        created_at=CREATED_AT,
    )
    queue = FakeInferenceJobQueue(job=job)
    repository = FakeVideoUploadRepository(
        asset=make_asset(),
        result=inference_result,
    )
    service = InferenceJobStatusService(
        repository=repository,
        inference_job_queue=queue,
    )

    result = await service.get_for_user(
        user_id=7,
        job_id=JOB_ID,
    )

    assert result is not None
    assert result.job == job
    assert result.result == inference_result
    assert repository.requested_utterance_ids == [job.utterance_id]


async def test_get_for_user_returns_none_for_unknown_job():
    queue = FakeInferenceJobQueue(
        job=None,
    )

    repository = FakeVideoUploadRepository(
        asset=make_asset(),
    )

    service = InferenceJobStatusService(
        repository=repository,
        inference_job_queue=queue,
    )

    result = await service.get_for_user(
        user_id=7,
        job_id=JOB_ID,
    )

    assert result is None

    assert repository.requested_video_ids == []


async def test_get_for_user_returns_none_when_video_metadata_is_missing():
    job = make_job()

    queue = FakeInferenceJobQueue(
        job=job,
    )

    repository = FakeVideoUploadRepository(
        asset=None,
    )

    service = InferenceJobStatusService(
        repository=repository,
        inference_job_queue=queue,
    )

    result = await service.get_for_user(
        user_id=7,
        job_id=JOB_ID,
    )

    assert result is None

    assert repository.requested_video_ids == [job.video_id]


async def test_get_for_user_hides_another_users_job():
    job = make_job()

    queue = FakeInferenceJobQueue(
        job=job,
    )

    repository = FakeVideoUploadRepository(
        asset=make_asset(
            user_id=99,
        )
    )

    service = InferenceJobStatusService(
        repository=repository,
        inference_job_queue=queue,
    )

    result = await service.get_for_user(
        user_id=7,
        job_id=JOB_ID,
    )

    assert result is None


async def test_get_for_user_rejects_nonpositive_user_id():
    queue = FakeInferenceJobQueue(
        job=make_job(),
    )

    repository = FakeVideoUploadRepository(
        asset=make_asset(),
    )

    service = InferenceJobStatusService(
        repository=repository,
        inference_job_queue=queue,
    )

    with pytest.raises(
        ValueError,
        match="user_id는 양의 정수",
    ):
        await service.get_for_user(
            user_id=0,
            job_id=JOB_ID,
        )

    assert queue.requested_job_ids == []
    assert repository.requested_video_ids == []
