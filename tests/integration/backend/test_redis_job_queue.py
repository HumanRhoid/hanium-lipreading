"""실제 Redis를 사용하는 비동기 추론 Job queue 통합 테스트."""

import os
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from src.backend.core.config import Settings
from src.backend.recognition.adapters.redis_job_queue import (
    INFERENCE_JOB_STREAM,
    RedisInferenceJobQueue,
)
from src.backend.recognition.domain import RecognitionMode

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"


def _get_test_redis_url() -> str:
    """실수로 개발 Redis를 삭제하지 않도록 테스트 Redis만 허용한다."""

    if os.getenv("RUN_REDIS_INTEGRATION_TESTS") != "1":
        pytest.skip(
            "RUN_REDIS_INTEGRATION_TESTS=1이 없어 Redis 통합 테스트를 건너뜁니다"
        )

    redis_url = os.getenv(
        "TEST_REDIS_URL",
        "redis://localhost:6380/0",
    )

    parsed = urlsplit(redis_url)

    if parsed.scheme not in {
        "redis",
        "rediss",
    }:
        raise pytest.UsageError(
            "TEST_REDIS_URL은 redis:// 또는 rediss:// URL이어야 합니다"
        )

    if parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise pytest.UsageError("Redis 통합 테스트는 loopback host만 허용합니다")

    if parsed.port != 6380:
        raise pytest.UsageError(
            "Redis 통합 테스트는 테스트 전용 localhost:6380만 허용합니다"
        )

    return redis_url


@pytest.fixture
async def redis_test_context():
    """각 테스트마다 test Redis DB를 비우고 queue를 제공한다."""

    redis_url = _get_test_redis_url()

    verification_client = Redis.from_url(
        redis_url,
        decode_responses=True,
    )

    await verification_client.ping()
    await verification_client.flushdb()

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
        redis_url=redis_url,
    )

    queue = RedisInferenceJobQueue(settings)

    try:
        yield (
            queue,
            verification_client,
        )
    finally:
        await queue.close()

        await verification_client.flushdb()
        await verification_client.aclose()


async def test_redis_job_queue_ping(
    redis_test_context,
):
    queue, _ = redis_test_context

    assert await queue.ping() is True


async def test_enqueue_persists_job_hash_and_stream_entry(
    redis_test_context,
):
    (
        queue,
        redis_client,
    ) = redis_test_context

    job_id = str(uuid4())

    created_at = datetime(
        2026,
        8,
        27,
        16,
        30,
        tzinfo=UTC,
    )

    result = await queue.enqueue_or_get(
        job_id=job_id,
        utterance_id=123,
        video_id=45,
        object_key=(
            "11111111-2222-4333-8444-555555555555/"
            "2026/08/"
            "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee.webm"
        ),
        mode=RecognitionMode.CLOSED,
        created_at=created_at,
    )

    assert result.created is True

    assert result.job.job_id == job_id

    assert result.job.utterance_id == 123

    assert result.job.video_id == 45

    assert result.job.mode is RecognitionMode.CLOSED

    assert result.job.status == "QUEUED"

    assert result.job.created_at == created_at

    assert result.job.updated_at == created_at

    assert result.job.error_code is None

    raw_job = await redis_client.hgetall(f"inference:job:{job_id}")

    assert raw_job["job_id"] == job_id

    assert raw_job["utterance_id"] == "123"

    assert raw_job["video_id"] == "45"

    assert raw_job["mode"] == "CLOSED"

    assert raw_job["status"] == "QUEUED"

    assert raw_job["error_code"] == ""

    mapped_job_id = await redis_client.get("inference:video:45:job")

    assert mapped_job_id == job_id

    stream_entries = await redis_client.xrange(INFERENCE_JOB_STREAM)

    assert len(stream_entries) == 1

    (
        _stream_entry_id,
        stream_payload,
    ) = stream_entries[0]

    assert stream_payload["job_id"] == job_id

    assert stream_payload["utterance_id"] == "123"

    assert stream_payload["video_id"] == "45"

    assert stream_payload["mode"] == "CLOSED"


async def test_get_job_restores_saved_job(
    redis_test_context,
):
    queue, _ = redis_test_context

    job_id = str(uuid4())

    created_at = datetime(
        2026,
        8,
        27,
        16,
        35,
        tzinfo=UTC,
    )

    await queue.enqueue_or_get(
        job_id=job_id,
        utterance_id=201,
        video_id=101,
        object_key=("storage-user/2026/08/clip.mp4"),
        mode=RecognitionMode.CLOSED,
        created_at=created_at,
    )

    found = await queue.get_job(job_id=job_id)

    assert found is not None

    assert found.job_id == job_id

    assert found.utterance_id == 201

    assert found.video_id == 101

    assert found.object_key == "storage-user/2026/08/clip.mp4"

    assert found.status == "QUEUED"


async def test_get_unknown_job_returns_none(
    redis_test_context,
):
    queue, _ = redis_test_context

    result = await queue.get_job(job_id=str(uuid4()))

    assert result is None


async def test_same_video_is_not_enqueued_twice(
    redis_test_context,
):
    (
        queue,
        redis_client,
    ) = redis_test_context

    first_job_id = str(uuid4())

    second_job_id = str(uuid4())

    created_at = datetime(
        2026,
        8,
        27,
        16,
        40,
        tzinfo=UTC,
    )

    first = await queue.enqueue_or_get(
        job_id=first_job_id,
        utterance_id=300,
        video_id=200,
        object_key=("storage-user/2026/08/same-video.webm"),
        mode=RecognitionMode.CLOSED,
        created_at=created_at,
    )

    second = await queue.enqueue_or_get(
        job_id=second_job_id,
        utterance_id=300,
        video_id=200,
        object_key=("storage-user/2026/08/same-video.webm"),
        mode=RecognitionMode.CLOSED,
        created_at=created_at,
    )

    assert first.created is True

    assert second.created is False

    assert second.job.job_id == first_job_id

    assert await redis_client.xlen(INFERENCE_JOB_STREAM) == 1

    assert await redis_client.exists(f"inference:job:{second_job_id}") == 0


async def test_same_video_rejects_different_job_metadata(
    redis_test_context,
):
    (
        queue,
        redis_client,
    ) = redis_test_context

    created_at = datetime(
        2026,
        8,
        27,
        16,
        45,
        tzinfo=UTC,
    )

    await queue.enqueue_or_get(
        job_id=str(uuid4()),
        utterance_id=400,
        video_id=300,
        object_key=("storage-user/2026/08/original.webm"),
        mode=RecognitionMode.CLOSED,
        created_at=created_at,
    )

    with pytest.raises(
        RuntimeError,
        match="메타데이터",
    ):
        await queue.enqueue_or_get(
            job_id=str(uuid4()),
            utterance_id=401,
            video_id=300,
            object_key=("storage-user/2026/08/different.webm"),
            mode=RecognitionMode.CLOSED,
            created_at=created_at,
        )

    assert await redis_client.xlen(INFERENCE_JOB_STREAM) == 1


async def test_job_id_cannot_be_reused_for_different_video(
    redis_test_context,
):
    (
        queue,
        redis_client,
    ) = redis_test_context

    job_id = str(uuid4())

    created_at = datetime(
        2026,
        8,
        27,
        16,
        50,
        tzinfo=UTC,
    )

    await queue.enqueue_or_get(
        job_id=job_id,
        utterance_id=500,
        video_id=400,
        object_key=("storage-user/2026/08/first.webm"),
        mode=RecognitionMode.CLOSED,
        created_at=created_at,
    )

    with pytest.raises(
        RuntimeError,
        match="job_id",
    ):
        await queue.enqueue_or_get(
            job_id=job_id,
            utterance_id=501,
            video_id=401,
            object_key=("storage-user/2026/08/second.webm"),
            mode=RecognitionMode.CLOSED,
            created_at=created_at,
        )

    assert await redis_client.xlen(INFERENCE_JOB_STREAM) == 1

    assert await redis_client.exists("inference:video:401:job") == 0
