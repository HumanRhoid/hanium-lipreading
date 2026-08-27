"""실제 PostgreSQL과 MinIO를 함께 사용하는 영상 업로드 통합 테스트."""

import os
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from src.backend.auth.adapters.repository import User
from src.backend.core.config import Settings
from src.backend.recognition.adapters.object_storage import S3ObjectStorage
from src.backend.recognition.adapters.repository import (
    SQLAlchemyRecognitionRepository,
    Utterance,
    VideoAsset,
)
from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.video_upload_service import VideoUploadService

pytestmark = pytest.mark.integration


def _build_object_storage_settings(
    postgres_url: str,
) -> Settings:
    """로컬 MinIO만 실제 Object Storage 통합 테스트 대상으로 허용한다."""

    if os.getenv("RUN_OBJECT_STORAGE_INTEGRATION_TESTS") != "1":
        pytest.skip(
            "RUN_OBJECT_STORAGE_INTEGRATION_TESTS=1이 없어 "
            "MinIO 통합 테스트를 건너뜁니다"
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

    return Settings(
        _env_file=None,
        app_env="test",
        database_url=postgres_url,
        object_storage_endpoint_url=(endpoint_url),
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
) -> tuple[int, object]:
    """영상 upload FK와 저장 경로에 사용할 실제 사용자를 생성한다."""

    user = User(
        username=(f"upload-{uuid4().hex[:16]}"),
        password_hash=("integration-test-password-hash"),
        display_name="업로드 통합 테스트 사용자",
    )

    async with postgres_session_factory.begin() as db_session:
        db_session.add(user)

        await db_session.flush()

        return (
            user.user_id,
            user.storage_uuid,
        )


async def test_video_upload_persists_to_postgresql_and_minio(
    postgres_url,
    postgres_session_factory,
):
    settings = _build_object_storage_settings(postgres_url)

    repository = SQLAlchemyRecognitionRepository(postgres_session_factory)

    object_storage = S3ObjectStorage(settings)

    service = VideoUploadService(
        repository=repository,
        object_storage=object_storage,
        max_upload_bytes=(settings.max_video_upload_bytes),
    )

    (
        user_id,
        storage_uuid,
    ) = await _create_test_user(postgres_session_factory)

    data = b"integration-test-webm-video-bytes"

    idempotency_key = str(uuid4())

    uploaded_object_key = None

    try:
        first_result = await service.upload(
            user_id=user_id,
            storage_uuid=storage_uuid,
            idempotency_key=idempotency_key,
            data=data,
            content_type="video/webm",
            mode=RecognitionMode.CLOSED,
        )

        uploaded_object_key = first_result.asset.object_key

        assert first_result.duplicate is False

        assert first_result.asset.user_id == user_id

        assert first_result.asset.idempotency_key == idempotency_key

        assert first_result.asset.original_mime_type == "video/webm"

        assert first_result.asset.size_bytes == len(data)

        assert first_result.asset.storage_status == "UPLOADED"

        assert first_result.asset.storage_purpose == "TEMPORARY_INFERENCE"

        assert first_result.asset.retention_until is not None

        assert uploaded_object_key.startswith(f"{storage_uuid}/")

        assert uploaded_object_key.endswith(".webm")

        assert "recognition-videos/" not in uploaded_object_key

        assert await object_storage.exists(uploaded_object_key) is True

        stored_data = await object_storage.get(uploaded_object_key)

        assert stored_data == data

        found_asset = await repository.find_video_asset_by_idempotency_key(
            user_id=user_id,
            idempotency_key=idempotency_key,
        )

        assert found_asset is not None

        assert found_asset.video_id == first_result.asset.video_id

        assert found_asset.utterance_id == first_result.asset.utterance_id

        assert found_asset.object_key == uploaded_object_key

        second_result = await service.upload(
            user_id=user_id,
            storage_uuid=storage_uuid,
            idempotency_key=idempotency_key,
            data=data,
            content_type="video/webm",
            mode=RecognitionMode.CLOSED,
        )

        assert second_result.duplicate is True

        assert second_result.asset.video_id == first_result.asset.video_id

        assert second_result.asset.utterance_id == first_result.asset.utterance_id

        assert second_result.asset.object_key == first_result.asset.object_key

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
                first_result.asset.video_id,
            )

            stored_utterance = await db_session.get(
                Utterance,
                first_result.asset.utterance_id,
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
