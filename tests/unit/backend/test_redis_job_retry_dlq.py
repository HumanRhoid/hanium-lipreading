"""Redis inference retry / DLQ policy tests."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.backend.recognition.adapters.redis_job_queue import (
    DEFAULT_INFERENCE_JOB_MAX_RETRIES,
    INFERENCE_JOB_DEAD_LETTER_STREAM,
    INFERENCE_JOB_STREAM,
    RedisInferenceJobQueue,
)


pytestmark = pytest.mark.asyncio

NOW = datetime(
    2026,
    9,
    3,
    17,
    30,
    tzinfo=UTC,
)


def raw_job(
    *,
    job_id: str,
    status: str,
    error_code: str,
    retry_count: int,
):
    return {
        "job_id": job_id,
        "utterance_id": "101",
        "video_id": "202",
        "object_key": "private/video.webm",
        "mode": "CLOSED",
        "status": status,
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
        "error_code": error_code,
        "retry_count": str(retry_count),
    }


class FakeRedis:
    def __init__(
        self,
        *,
        eval_result,
        raw_job_value,
    ) -> None:
        self.eval_result = eval_result
        self.raw_job_value = raw_job_value
        self.eval_calls = []

    async def eval(
        self,
        script,
        numkeys,
        *args,
    ):
        self.eval_calls.append(
            (
                script,
                numkeys,
                args,
            )
        )

        return self.eval_result

    async def hgetall(
        self,
        key: str,
    ):
        return dict(
            self.raw_job_value
        )


def make_queue(
    *,
    redis: FakeRedis,
    max_retries: int = 3,
):
    queue = object.__new__(
        RedisInferenceJobQueue
    )

    queue._redis = redis
    queue._max_retries = max_retries
    queue._terminal_ttl_seconds = 86400

    return queue


async def test_retry_policy_is_three_additional_attempts():
    assert DEFAULT_INFERENCE_JOB_MAX_RETRIES == 3


async def test_failure_requeues_before_retry_limit():
    job_id = str(uuid4())

    redis = FakeRedis(
        eval_result=[
            "RETRY",
            "1",
        ],
        raw_job_value=raw_job(
            job_id=job_id,
            status="QUEUED",
            error_code="INFERENCE_FAILED",
            retry_count=1,
        ),
    )

    queue = make_queue(
        redis=redis,
    )

    result = await queue.mark_failed(
        job_id=job_id,
        error_code="INFERENCE_FAILED",
        updated_at=NOW,
    )

    assert result.status == "QUEUED"

    script, numkeys, args = (
        redis.eval_calls[0]
    )

    assert numkeys == 3

    assert args[1] == INFERENCE_JOB_STREAM
    assert (
        args[2]
        == INFERENCE_JOB_DEAD_LETTER_STREAM
    )

    assert args[5] == 3
    assert args[6] == 86400

    assert (
        "retry_count < max_retries"
        in script
    )

    assert '"status", "QUEUED"' in script


async def test_fourth_failure_goes_to_dead_letter():
    job_id = str(uuid4())

    redis = FakeRedis(
        eval_result=[
            "DLQ",
            "3",
        ],
        raw_job_value=raw_job(
            job_id=job_id,
            status="FAILED",
            error_code="INFERENCE_FAILED",
            retry_count=3,
        ),
    )

    queue = make_queue(
        redis=redis,
    )

    result = await queue.mark_failed(
        job_id=job_id,
        error_code="INFERENCE_FAILED",
        updated_at=NOW,
    )

    assert result.status == "FAILED"

    script, _numkeys, _args = (
        redis.eval_calls[0]
    )

    assert (
        '"status", "FAILED"'
        in script
    )

    assert (
        '"retry_count", tostring(retry_count)'
        in script
    )

    assert '"failed_at", ARGV[1]' in script

    # Final failure receives the same 24h terminal TTL.
    assert script.count('"EXPIRE"') == 2


async def test_retry_stream_preserves_existing_job_contract():
    job_id = str(uuid4())

    redis = FakeRedis(
        eval_result=[
            "RETRY",
            "2",
        ],
        raw_job_value=raw_job(
            job_id=job_id,
            status="QUEUED",
            error_code="RESULT_PERSISTENCE_FAILED",
            retry_count=2,
        ),
    )

    queue = make_queue(
        redis=redis,
    )

    await queue.mark_failed(
        job_id=job_id,
        error_code="RESULT_PERSISTENCE_FAILED",
        updated_at=NOW,
    )

    script = redis.eval_calls[0][0]

    # Keep the existing inference stream payload contract.
    for field in (
        '"job_id"',
        '"utterance_id"',
        '"video_id"',
        '"object_key"',
        '"mode"',
        '"created_at"',
    ):
        assert field in script

    # Do not add user identity or binary data.
    assert '"user_id"' not in script
    assert '"video_data"' not in script
    assert '"frame"' not in script
