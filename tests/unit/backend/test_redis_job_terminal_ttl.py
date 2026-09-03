"""Redis terminal inference Job TTL tests."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.backend.core.config import Settings
from src.backend.recognition.adapters.redis_job_queue import (
    RedisInferenceJobQueue,
)

pytestmark = pytest.mark.asyncio

UPDATED_AT = datetime(
    2026,
    9,
    3,
    9,
    0,
    tzinfo=UTC,
)


def raw_job(
    *,
    job_id: str,
    status: str,
    error_code: str = "",
):
    return {
        "job_id": job_id,
        "utterance_id": "101",
        "video_id": "201",
        "object_key": "private/input.webm",
        "mode": "CLOSED",
        "status": status,
        "created_at": UPDATED_AT.isoformat(),
        "updated_at": UPDATED_AT.isoformat(),
        "error_code": error_code,
    }


class FakeRedis:
    def __init__(self, *, job_id: str) -> None:
        self.job_id = job_id

        self.hashes = {
            f"inference:job:{job_id}": raw_job(
                job_id=job_id,
                status="PROCESSING",
            )
        }

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

        job_key = args[0]
        target_status = args[1]
        updated_at = args[2]
        error_code = args[3]

        job = self.hashes[job_key]

        job["status"] = target_status
        job["updated_at"] = updated_at
        job["error_code"] = error_code

        return target_status

    async def hgetall(
        self,
        key,
    ):
        return dict(
            self.hashes.get(key, {})
        )


def make_queue(
    *,
    job_id: str,
    ttl: int = 86400,
):
    settings = Settings(
        inference_job_terminal_ttl_seconds=ttl,
    )

    queue = RedisInferenceJobQueue(settings)

    fake = FakeRedis(job_id=job_id)

    # No network connection is made by Redis.from_url().
    queue._redis = fake

    return queue, fake


async def test_succeeded_transition_passes_24h_ttl_to_lua():
    job_id = str(uuid4())

    queue, redis = make_queue(
        job_id=job_id,
    )

    job = await queue.mark_succeeded(
        job_id=job_id,
        updated_at=UPDATED_AT,
    )

    assert job.status == "SUCCEEDED"

    assert len(redis.eval_calls) == 1

    script, numkeys, args = redis.eval_calls[0]

    assert numkeys == 1
    assert args[0] == f"inference:job:{job_id}"
    assert args[1] == "SUCCEEDED"
    assert args[4] == 86400
    assert args[5] == "inference:video:"
    assert args[6] == ":job"
    assert args[7] == job_id

    # Lua and the Python helper must construct the exact same
    # video -> job mapping key.
    assert (
        RedisInferenceJobQueue._video_job_key(201)
        == "inference:video:201:job"
    )

    assert (
        "ARGV[5] .. video_id .. ARGV[6]"
        in script
    )

    # The Lua script must expire both:
    # 1) inference:job:{job_id}
    # 2) inference:video:{video_id}
    #
    # Do not couple this test to source-code indentation.
    assert script.count('"EXPIRE"') == 2


async def test_failed_transition_uses_configurable_terminal_ttl():
    """??? ?? ? FAILED/DLQ ???? ?? TTL? ????."""

    job_id = str(uuid4())

    class FailureRedis:
        def __init__(self) -> None:
            self.eval_calls: list[tuple[object, ...]] = []

        async def eval(
            self,
            *args,
        ):
            self.eval_calls.append(args)

            # ?? _HANDLE_FAILURE_SCRIPT? ?? ?? ?? ??.
            return [
                "DLQ",
                "0",
            ]

        async def hgetall(
            self,
            key: str,
        ):
            assert key == f"inference:job:{job_id}"

            return {
                "job_id": job_id,
                "utterance_id": "101",
                "video_id": "201",
                "object_key": "private/video.webm",
                "mode": "CLOSED",
                "status": "FAILED",
                "created_at": UPDATED_AT.isoformat(),
                "updated_at": UPDATED_AT.isoformat(),
                "error_code": "MODEL_INFERENCE_FAILED",
            }

    redis = FailureRedis()

    queue = object.__new__(
        RedisInferenceJobQueue
    )

    queue._redis = redis

    # ? ???? "??? ?"? ???
    # ?? ?? ? terminal TTL ??? ????.
    queue._max_retries = 0
    queue._terminal_ttl_seconds = 12345

    job = await queue.mark_failed(
        job_id=job_id,
        error_code="MODEL_INFERENCE_FAILED",
        updated_at=UPDATED_AT,
    )

    assert job.status == "FAILED"
    assert (
        job.error_code
        == "MODEL_INFERENCE_FAILED"
    )

    assert len(redis.eval_calls) == 1

    (
        script,
        numkeys,
        job_key,
        inference_stream,
        dead_letter_stream,
        updated_at_text,
        error_code,
        max_retries,
        ttl_seconds,
        video_key_prefix,
        video_key_suffix,
        canonical_job_id,
    ) = redis.eval_calls[0]

    assert numkeys == 3

    assert (
        job_key
        == f"inference:job:{job_id}"
    )

    assert (
        inference_stream
        == "stream:inference:jobs"
    )

    assert (
        dead_letter_stream
        == "stream:inference:dead-letter"
    )

    assert (
        updated_at_text
        == UPDATED_AT.isoformat()
    )

    assert (
        error_code
        == "MODEL_INFERENCE_FAILED"
    )

    # ??? ?? ??? ?? ???.
    assert max_retries == 0

    # ??: ??? terminal TTL? Lua? ????.
    assert ttl_seconds == 12345

    assert (
        video_key_prefix
        == "inference:video:"
    )

    assert video_key_suffix == ":job"
    assert canonical_job_id == job_id

    # ?? DLQ? ? job hash? video->job mapping
    # ? ? terminal TTL ????? ??.
    assert script.count('"EXPIRE"') == 2

    assert (
        "ARGV[5]"
        in script
    )

    assert (
        "ARGV[6]"
        in script
    )


async def test_default_terminal_ttl_is_24_hours():
    settings = Settings()

    assert (
        settings.inference_job_terminal_ttl_seconds
        == 24 * 60 * 60
    )
