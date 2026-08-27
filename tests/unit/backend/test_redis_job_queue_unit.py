"""Redis 추론 Job queue의 Redis 비의존 로직을 단위 테스트한다."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from redis.exceptions import ResponseError

from src.backend.recognition.adapters import (
    redis_job_queue as redis_job_queue_module,
)
from src.backend.recognition.adapters.redis_job_queue import (
    RedisInferenceJobQueue,
)
from src.backend.recognition.domain import RecognitionMode

CREATED_AT = datetime(
    2026,
    8,
    27,
    17,
    30,
    tzinfo=UTC,
)


def _raw_job(
    *,
    job_id: str | None = None,
    utterance_id: str = "123",
    video_id: str = "45",
    object_key: str = "storage-user/2026/08/video.webm",
    mode: str = "CLOSED",
    status: str = "QUEUED",
    created_at: str | None = None,
    updated_at: str | None = None,
    error_code: str = "",
) -> dict[str, str]:
    actual_job_id = job_id or str(uuid4())
    actual_created_at = created_at or CREATED_AT.isoformat()
    actual_updated_at = updated_at or CREATED_AT.isoformat()

    return {
        "job_id": actual_job_id,
        "utterance_id": utterance_id,
        "video_id": video_id,
        "object_key": object_key,
        "mode": mode,
        "status": status,
        "created_at": actual_created_at,
        "updated_at": actual_updated_at,
        "error_code": error_code,
    }


class FakeRedis:
    def __init__(
        self,
        *,
        eval_result: object = None,
        eval_error: Exception | None = None,
        hashes: dict[str, dict[str, str]] | None = None,
        ping_result: bool = True,
    ) -> None:
        self.eval_result = eval_result
        self.eval_error = eval_error
        self.hashes = hashes or {}
        self.ping_result = ping_result

        self.eval_calls: list[tuple[object, ...]] = []
        self.hgetall_calls: list[str] = []
        self.closed = False

    async def ping(self):
        return self.ping_result

    async def aclose(self):
        self.closed = True

    async def eval(
        self,
        *args,
    ):
        self.eval_calls.append(args)

        if self.eval_error is not None:
            raise self.eval_error

        return self.eval_result

    async def hgetall(
        self,
        key: str,
    ):
        self.hgetall_calls.append(key)

        return self.hashes.get(
            key,
            {},
        )


def _queue_with_fake_redis(
    redis: FakeRedis,
) -> RedisInferenceJobQueue:
    queue = object.__new__(RedisInferenceJobQueue)

    queue._redis = redis

    return queue


async def test_constructor_ping_and_close(
    monkeypatch,
):
    redis = FakeRedis(
        ping_result=True,
    )

    captured: dict[str, object] = {}

    def fake_from_url(
        url: str,
        *,
        decode_responses: bool,
    ):
        captured["url"] = url
        captured["decode_responses"] = decode_responses

        return redis

    monkeypatch.setattr(
        redis_job_queue_module.Redis,
        "from_url",
        fake_from_url,
    )

    settings = SimpleNamespace(
        redis_url="redis://localhost:6380/0",
    )

    queue = RedisInferenceJobQueue(settings)

    assert captured == {
        "url": "redis://localhost:6380/0",
        "decode_responses": True,
    }

    assert await queue.ping() is True

    await queue.close()

    assert redis.closed is True


async def test_enqueue_creates_new_job():
    job_uuid = uuid4()

    redis = FakeRedis(
        eval_result=[
            str(job_uuid),
            "1",
        ]
    )

    queue = _queue_with_fake_redis(redis)

    result = await queue.enqueue_or_get(
        job_id=f" {str(job_uuid).upper()} ",
        utterance_id=123,
        video_id=45,
        object_key=("  storage-user/2026/08/video.webm  "),
        mode=RecognitionMode.CLOSED,
        created_at=CREATED_AT,
    )

    assert result.created is True

    assert result.job.job_id == str(job_uuid)

    assert result.job.utterance_id == 123
    assert result.job.video_id == 45

    assert result.job.object_key == "storage-user/2026/08/video.webm"

    assert result.job.mode is RecognitionMode.CLOSED

    assert result.job.status == "QUEUED"

    assert result.job.created_at == CREATED_AT

    assert result.job.updated_at == CREATED_AT

    assert result.job.error_code is None

    assert len(redis.eval_calls) == 1


@pytest.mark.parametrize(
    (
        "job_id",
        "utterance_id",
        "video_id",
        "object_key",
        "created_at",
        "message",
    ),
    [
        (
            "not-a-uuid",
            123,
            45,
            "video.webm",
            CREATED_AT,
            "UUID",
        ),
        (
            str(uuid4()),
            0,
            45,
            "video.webm",
            CREATED_AT,
            "utterance_id",
        ),
        (
            str(uuid4()),
            123,
            0,
            "video.webm",
            CREATED_AT,
            "video_id",
        ),
        (
            str(uuid4()),
            123,
            45,
            "   ",
            CREATED_AT,
            "object_key",
        ),
        (
            str(uuid4()),
            123,
            45,
            "video.webm",
            datetime(
                2026,
                8,
                27,
                17,
                30,
            ),
            "timezone-aware",
        ),
    ],
)
async def test_enqueue_rejects_invalid_request(
    job_id,
    utterance_id,
    video_id,
    object_key,
    created_at,
    message,
):
    redis = FakeRedis()

    queue = _queue_with_fake_redis(redis)

    with pytest.raises(
        ValueError,
        match=message,
    ):
        await queue.enqueue_or_get(
            job_id=job_id,
            utterance_id=utterance_id,
            video_id=video_id,
            object_key=object_key,
            mode=RecognitionMode.CLOSED,
            created_at=created_at,
        )

    assert redis.eval_calls == []


async def test_enqueue_translates_reused_job_id_error():
    redis = FakeRedis(eval_error=ResponseError("JOB_ID_ALREADY_EXISTS"))

    queue = _queue_with_fake_redis(redis)

    with pytest.raises(
        RuntimeError,
        match="job_id",
    ):
        await queue.enqueue_or_get(
            job_id=str(uuid4()),
            utterance_id=123,
            video_id=45,
            object_key="video.webm",
            mode=RecognitionMode.CLOSED,
            created_at=CREATED_AT,
        )


async def test_enqueue_reraises_unexpected_redis_response_error():
    error = ResponseError("OTHER_REDIS_ERROR")

    redis = FakeRedis(
        eval_error=error,
    )

    queue = _queue_with_fake_redis(redis)

    with pytest.raises(
        ResponseError,
        match="OTHER_REDIS_ERROR",
    ):
        await queue.enqueue_or_get(
            job_id=str(uuid4()),
            utterance_id=123,
            video_id=45,
            object_key="video.webm",
            mode=RecognitionMode.CLOSED,
            created_at=CREATED_AT,
        )


async def test_enqueue_rejects_created_job_id_mismatch():
    requested_job_id = str(uuid4())

    redis = FakeRedis(
        eval_result=[
            str(uuid4()),
            "1",
        ]
    )

    queue = _queue_with_fake_redis(redis)

    with pytest.raises(
        RuntimeError,
        match="식별자가 요청과 일치",
    ):
        await queue.enqueue_or_get(
            job_id=requested_job_id,
            utterance_id=123,
            video_id=45,
            object_key="video.webm",
            mode=RecognitionMode.CLOSED,
            created_at=CREATED_AT,
        )


async def test_enqueue_returns_existing_job_for_same_video():
    existing_job_id = str(uuid4())

    raw_job = _raw_job(
        job_id=existing_job_id,
    )

    redis = FakeRedis(
        eval_result=[
            existing_job_id,
            "0",
        ],
        hashes={(f"inference:job:{existing_job_id}"): raw_job},
    )

    queue = _queue_with_fake_redis(redis)

    result = await queue.enqueue_or_get(
        job_id=str(uuid4()),
        utterance_id=123,
        video_id=45,
        object_key=("storage-user/2026/08/video.webm"),
        mode=RecognitionMode.CLOSED,
        created_at=CREATED_AT,
    )

    assert result.created is False

    assert result.job.job_id == existing_job_id

    assert result.job.utterance_id == 123
    assert result.job.video_id == 45


async def test_enqueue_fails_when_existing_job_is_missing():
    existing_job_id = str(uuid4())

    redis = FakeRedis(
        eval_result=[
            existing_job_id,
            "0",
        ],
    )

    queue = _queue_with_fake_redis(redis)

    with pytest.raises(
        RuntimeError,
        match="기존 추론 Job 상태",
    ):
        await queue.enqueue_or_get(
            job_id=str(uuid4()),
            utterance_id=123,
            video_id=45,
            object_key=("storage-user/2026/08/video.webm"),
            mode=RecognitionMode.CLOSED,
            created_at=CREATED_AT,
        )


async def test_enqueue_rejects_existing_job_with_different_metadata():
    existing_job_id = str(uuid4())

    redis = FakeRedis(
        eval_result=[
            existing_job_id,
            "0",
        ],
        hashes={
            (f"inference:job:{existing_job_id}"): _raw_job(
                job_id=existing_job_id,
                object_key=("storage-user/2026/08/different.webm"),
            )
        },
    )

    queue = _queue_with_fake_redis(redis)

    with pytest.raises(
        RuntimeError,
        match="메타데이터",
    ):
        await queue.enqueue_or_get(
            job_id=str(uuid4()),
            utterance_id=123,
            video_id=45,
            object_key=("storage-user/2026/08/video.webm"),
            mode=RecognitionMode.CLOSED,
            created_at=CREATED_AT,
        )


async def test_get_unknown_job_returns_none():
    redis = FakeRedis()

    queue = _queue_with_fake_redis(redis)

    job_id = str(uuid4())

    result = await queue.get_job(job_id=job_id)

    assert result is None

    assert redis.hgetall_calls == [f"inference:job:{job_id}"]


async def test_get_job_restores_failed_job():
    job_id = str(uuid4())

    redis = FakeRedis(
        hashes={
            f"inference:job:{job_id}": (
                _raw_job(
                    job_id=job_id,
                    status="FAILED",
                    error_code="MODEL_FAILED",
                )
            )
        }
    )

    queue = _queue_with_fake_redis(redis)

    result = await queue.get_job(job_id=job_id)

    assert result is not None

    assert result.job_id == job_id
    assert result.status == "FAILED"

    assert result.error_code == "MODEL_FAILED"


@pytest.mark.parametrize(
    "raw_result",
    [
        None,
        [],
        ["only-one"],
        ["one", "two", "three"],
        "not-a-list",
    ],
)
def test_parse_enqueue_result_rejects_invalid_shape(
    raw_result,
):
    with pytest.raises(
        RuntimeError,
        match="enqueue 결과 형식",
    ):
        RedisInferenceJobQueue._parse_enqueue_result(raw_result)


def test_parse_enqueue_result_rejects_invalid_created_flag():
    with pytest.raises(
        RuntimeError,
        match="생성 여부",
    ):
        RedisInferenceJobQueue._parse_enqueue_result(
            [
                str(uuid4()),
                "unexpected",
            ]
        )


@pytest.mark.parametrize(
    (
        "created_flag",
        "expected_created",
    ),
    [
        (
            "0",
            False,
        ),
        (
            "1",
            True,
        ),
    ],
)
def test_parse_enqueue_result_accepts_valid_flags(
    created_flag,
    expected_created,
):
    job_id = str(uuid4())

    result = RedisInferenceJobQueue._parse_enqueue_result(
        [
            job_id,
            created_flag,
        ]
    )

    assert result == (
        job_id,
        expected_created,
    )


def test_parse_datetime_rejects_invalid_text():
    with pytest.raises(
        RuntimeError,
        match="시간 형식",
    ):
        RedisInferenceJobQueue._parse_datetime("not-a-datetime")


def test_parse_datetime_rejects_naive_datetime():
    with pytest.raises(
        RuntimeError,
        match="timezone",
    ):
        RedisInferenceJobQueue._parse_datetime("2026-08-27T17:30:00")


def test_parse_datetime_restores_aware_datetime():
    result = RedisInferenceJobQueue._parse_datetime(CREATED_AT.isoformat())

    assert result == CREATED_AT


def test_parse_job_rejects_missing_required_field():
    raw_job = _raw_job()

    del raw_job["video_id"]

    queue = _queue_with_fake_redis(FakeRedis())

    with pytest.raises(
        RuntimeError,
        match="불완전",
    ):
        queue._parse_job(raw_job)


def test_parse_job_rejects_invalid_status():
    raw_job = _raw_job(
        status="UNKNOWN",
    )

    queue = _queue_with_fake_redis(FakeRedis())

    with pytest.raises(
        RuntimeError,
        match="status",
    ):
        queue._parse_job(raw_job)


@pytest.mark.parametrize(
    (
        "field_name",
        "value",
    ),
    [
        (
            "utterance_id",
            "not-an-int",
        ),
        (
            "video_id",
            "not-an-int",
        ),
        (
            "mode",
            "UNKNOWN",
        ),
    ],
)
def test_parse_job_rejects_invalid_value_format(
    field_name,
    value,
):
    raw_job = _raw_job()

    raw_job[field_name] = value

    queue = _queue_with_fake_redis(FakeRedis())

    with pytest.raises(
        RuntimeError,
        match="데이터 형식",
    ):
        queue._parse_job(raw_job)


@pytest.mark.parametrize(
    "field_name",
    [
        "utterance_id",
        "video_id",
    ],
)
def test_parse_job_rejects_nonpositive_database_id(
    field_name,
):
    raw_job = _raw_job()

    raw_job[field_name] = "0"

    queue = _queue_with_fake_redis(FakeRedis())

    with pytest.raises(
        RuntimeError,
        match=field_name,
    ):
        queue._parse_job(raw_job)


def test_parse_job_rejects_invalid_stored_job_id():
    raw_job = _raw_job(
        job_id="not-a-uuid",
    )

    queue = _queue_with_fake_redis(FakeRedis())

    with pytest.raises(
        RuntimeError,
        match="Job ID 형식",
    ):
        queue._parse_job(raw_job)


def test_parse_job_rejects_blank_object_key():
    raw_job = _raw_job(
        object_key="   ",
    )

    queue = _queue_with_fake_redis(FakeRedis())

    with pytest.raises(
        RuntimeError,
        match="object_key",
    ):
        queue._parse_job(raw_job)


def test_job_key_helpers():
    job_id = str(uuid4())

    assert RedisInferenceJobQueue._job_key(job_id) == f"inference:job:{job_id}"

    assert RedisInferenceJobQueue._video_job_key(45) == "inference:video:45:job"
