"""HTTP부터 PostgreSQL, MinIO, Redis까지 영상 업로드 파이프라인을 검증한다."""

import os
from types import SimpleNamespace
from urllib.parse import urlparse, urlsplit
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from sqlalchemy import func, select

from src.backend.auth.adapters.repository import User
from src.backend.core.config import Settings
from src.backend.recognition.adapters.object_storage import S3ObjectStorage
from src.backend.recognition.adapters.redis_job_queue import (
    INFERENCE_JOB_STREAM,
    RedisInferenceJobQueue,
)
from src.backend.recognition.adapters.repository import (
    SQLAlchemyRecognitionRepository,
    Utterance,
    VideoAsset,
)
from src.backend.recognition.job_status_api import router as job_status_router
from src.backend.recognition.job_status_service import InferenceJobStatusService
from src.backend.recognition.submission_service import (
    RecognitionSubmissionService,
)
from src.backend.recognition.upload_api import router as upload_router
from src.backend.recognition.video_upload_service import (
    VideoUploadService,
)

pytestmark = pytest.mark.integration

SESSION_TOKEN = "integration-test-session-token"


def _build_pipeline_settings(
    postgres_url: str,
) -> Settings:
    """로컬 MinIO와 테스트 전용 Redis만 통합 테스트 대상으로 허용한다."""

    if os.getenv("RUN_OBJECT_STORAGE_INTEGRATION_TESTS") != "1":
        pytest.skip(
            "RUN_OBJECT_STORAGE_INTEGRATION_TESTS=1이 없어 "
            "MinIO 통합 테스트를 건너뜁니다"
        )

    if os.getenv("RUN_REDIS_INTEGRATION_TESTS") != "1":
        pytest.skip(
            "RUN_REDIS_INTEGRATION_TESTS=1이 없어 Redis 통합 테스트를 건너뜁니다"
        )

    endpoint_url = os.getenv(
        "TEST_OBJECT_STORAGE_ENDPOINT_URL",
        "http://localhost:9000",
    )

    parsed_endpoint = urlparse(endpoint_url)

    if parsed_endpoint.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise pytest.UsageError(
            "Object Storage 통합 테스트는 loopback endpoint만 허용합니다"
        )

    redis_url = os.getenv(
        "TEST_REDIS_URL",
        "redis://localhost:6380/0",
    )

    parsed_redis = urlsplit(redis_url)

    if parsed_redis.scheme not in {
        "redis",
        "rediss",
    }:
        raise pytest.UsageError(
            "TEST_REDIS_URL은 redis:// 또는 rediss:// URL이어야 합니다"
        )

    if parsed_redis.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise pytest.UsageError("Redis 통합 테스트는 loopback host만 허용합니다")

    if parsed_redis.port != 6380:
        raise pytest.UsageError(
            "Redis 통합 테스트는 테스트 전용 localhost:6380만 허용합니다"
        )

    return Settings(
        _env_file=None,
        app_env="test",
        database_url=postgres_url,
        redis_url=redis_url,
        object_storage_endpoint_url=endpoint_url,
        object_storage_access_key=os.getenv(
            "TEST_OBJECT_STORAGE_ACCESS_KEY",
            "minioadmin",
        ),
        object_storage_secret_key=os.getenv(
            "TEST_OBJECT_STORAGE_SECRET_KEY",
            "minioadmin123",
        ),
        object_storage_bucket=os.getenv(
            "TEST_OBJECT_STORAGE_BUCKET",
            "recognition-videos",
        ),
        object_storage_region=os.getenv(
            "TEST_OBJECT_STORAGE_REGION",
            "us-east-1",
        ),
    )


async def _create_test_user(
    postgres_session_factory,
) -> tuple[int, UUID]:
    """실제 PostgreSQL에 영상 소유 사용자를 생성한다."""

    user = User(
        username=f"pipeline-{uuid4().hex[:16]}",
        password_hash="integration-test-password-hash",
        display_name="파이프라인 통합 테스트 사용자",
    )

    async with postgres_session_factory.begin() as db_session:
        db_session.add(user)

        await db_session.flush()

        return (
            user.user_id,
            user.storage_uuid,
        )


class FakeAuthService:
    """인증 계층 자체가 아닌 업로드 이후 파이프라인을 검증하기 위한 대역."""

    def __init__(
        self,
        *,
        user_id: int,
        storage_uuid: UUID,
    ) -> None:
        self._user_id = user_id
        self._storage_uuid = storage_uuid

    async def get_current_user(
        self,
        session_token: str,
    ):
        if session_token != SESSION_TOKEN:
            raise AssertionError("예상하지 않은 테스트 세션 토큰입니다.")

        return SimpleNamespace(
            user_id=self._user_id,
            storage_uuid=self._storage_uuid,
        )


async def _post_video(
    app: FastAPI,
    *,
    idempotency_key: str,
    data: bytes,
):
    """실제 업로드 HTTP endpoint에 multipart 요청을 보낸다."""

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/api/v1/recognition/videos",
            headers={
                "X-Session-Token": SESSION_TOKEN,
                "Idempotency-Key": idempotency_key,
            },
            files={
                "file": (
                    "clip.webm",
                    data,
                    "video/webm",
                )
            },
            data={
                "mode": "CLOSED",
            },
        )


async def _get_job_status(
    app: FastAPI,
    *,
    job_id: str,
):
    """실제 추론 Job 상태 조회 HTTP endpoint를 호출한다."""

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(
            f"/api/v1/inference-jobs/{job_id}",
            headers={
                "X-Session-Token": SESSION_TOKEN,
            },
        )


async def test_http_upload_persists_and_enqueues_across_real_dependencies(
    postgres_url,
    postgres_session_factory,
):
    settings = _build_pipeline_settings(postgres_url)

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    object_storage = S3ObjectStorage(settings)

    video_upload_service = VideoUploadService(
        repository=repository,
        object_storage=object_storage,
        max_upload_bytes=(settings.max_video_upload_bytes),
    )

    job_queue = RedisInferenceJobQueue(settings)

    submission_service = RecognitionSubmissionService(
        video_upload_service=video_upload_service,
        inference_job_queue=job_queue,
    )

    job_status_service = InferenceJobStatusService(
        repository=repository,
        inference_job_queue=job_queue,
    )

    (
        user_id,
        storage_uuid,
    ) = await _create_test_user(postgres_session_factory)

    auth_service = FakeAuthService(
        user_id=user_id,
        storage_uuid=storage_uuid,
    )

    app = FastAPI()

    app.include_router(upload_router)
    app.include_router(job_status_router)

    app.state.settings = settings
    app.state.auth_service = auth_service
    app.state.submission_service = submission_service
    app.state.inference_job_status_service = job_status_service

    redis_client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )

    data = b"integration-pipeline-webm-video-bytes"
    idempotency_key = str(uuid4())

    uploaded_object_key: str | None = None

    await redis_client.ping()
    await redis_client.flushdb()

    try:
        first_response = await _post_video(
            app,
            idempotency_key=idempotency_key,
            data=data,
        )

        assert first_response.status_code == 202

        first_payload = first_response.json()

        assert first_payload["status"] == "QUEUED"
        assert first_payload["duplicate"] is False

        assert isinstance(
            first_payload["utterance_id"],
            int,
        )

        assert isinstance(
            first_payload["video_id"],
            int,
        )

        job_id = first_payload["job_id"]

        UUID(job_id)

        found_asset = await repository.find_video_asset_by_idempotency_key(
            user_id=user_id,
            idempotency_key=idempotency_key,
        )

        assert found_asset is not None

        uploaded_object_key = found_asset.object_key

        assert first_payload["video_id"] == found_asset.video_id

        assert first_payload["utterance_id"] == found_asset.utterance_id

        assert found_asset.user_id == user_id

        assert found_asset.original_mime_type == "video/webm"

        assert found_asset.storage_status == "UPLOADED"

        assert found_asset.storage_purpose == "TEMPORARY_INFERENCE"

        assert found_asset.retention_until is not None

        assert uploaded_object_key.startswith(f"{storage_uuid}/")

        assert uploaded_object_key.endswith(".webm")

        assert "recognition-videos/" not in uploaded_object_key

        assert await object_storage.exists(uploaded_object_key) is True

        stored_data = await object_storage.get(uploaded_object_key)

        assert stored_data == data

        raw_job = await redis_client.hgetall(f"inference:job:{job_id}")

        assert raw_job["job_id"] == job_id

        assert raw_job["video_id"] == str(found_asset.video_id)

        assert raw_job["utterance_id"] == str(found_asset.utterance_id)

        assert raw_job["object_key"] == uploaded_object_key

        assert raw_job["mode"] == "CLOSED"

        assert raw_job["status"] == "QUEUED"

        mapped_job_id = await redis_client.get(
            f"inference:video:{found_asset.video_id}:job"
        )

        assert mapped_job_id == job_id

        stream_entries = await redis_client.xrange(INFERENCE_JOB_STREAM)

        assert len(stream_entries) == 1

        (
            _stream_entry_id,
            stream_payload,
        ) = stream_entries[0]

        assert stream_payload["job_id"] == job_id

        assert stream_payload["video_id"] == str(found_asset.video_id)

        assert stream_payload["utterance_id"] == str(found_asset.utterance_id)

        assert stream_payload["object_key"] == uploaded_object_key

        assert stream_payload["mode"] == "CLOSED"

        status_response = await _get_job_status(
            app,
            job_id=job_id,
        )

        assert status_response.status_code == 200

        assert status_response.json() == {
            "job_id": job_id,
            "utterance_id": found_asset.utterance_id,
            "video_id": found_asset.video_id,
            "status": "QUEUED",
            "error_code": None,
        }

        (
            other_user_id,
            other_storage_uuid,
        ) = await _create_test_user(postgres_session_factory)

        app.state.auth_service = FakeAuthService(
            user_id=other_user_id,
            storage_uuid=other_storage_uuid,
        )

        unowned_response = await _get_job_status(
            app,
            job_id=job_id,
        )

        assert unowned_response.status_code == 404

        assert unowned_response.json() == {"detail": "추론 Job을 찾을 수 없습니다."}

        app.state.auth_service = auth_service

        second_response = await _post_video(
            app,
            idempotency_key=idempotency_key,
            data=data,
        )

        assert second_response.status_code == 200

        second_payload = second_response.json()

        assert second_payload["status"] == "QUEUED"

        assert second_payload["duplicate"] is True

        assert second_payload["job_id"] == job_id

        assert second_payload["video_id"] == found_asset.video_id

        assert second_payload["utterance_id"] == found_asset.utterance_id

        assert await redis_client.xlen(INFERENCE_JOB_STREAM) == 1

        async with postgres_session_factory() as db_session:
            video_count = await db_session.scalar(
                select(func.count())
                .select_from(VideoAsset)
                .where(VideoAsset.user_id == user_id)
            )

            utterance_count = await db_session.scalar(
                select(func.count())
                .select_from(Utterance)
                .where(Utterance.user_id == user_id)
            )

            stored_video = await db_session.get(
                VideoAsset,
                found_asset.video_id,
            )

            stored_utterance = await db_session.get(
                Utterance,
                found_asset.utterance_id,
            )

        assert video_count == 1
        assert utterance_count == 1

        assert stored_video is not None
        assert stored_utterance is not None

        assert stored_video.utterance_id == stored_utterance.utt_id

        assert stored_utterance.user_id == user_id

        assert stored_utterance.raw_text is None

    finally:
        if uploaded_object_key is not None and await object_storage.exists(
            uploaded_object_key
        ):
            await object_storage.delete(uploaded_object_key)

        await redis_client.flushdb()

        await redis_client.aclose()

        await job_queue.close()
