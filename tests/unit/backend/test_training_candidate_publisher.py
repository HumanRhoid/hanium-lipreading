"""Training candidate Redis outbox tests."""

from datetime import UTC, datetime

import pytest

from src.backend.recognition.training_candidate_publisher import (
    TRAINING_CANDIDATE_STREAM,
    TRAINING_CANDIDATE_STREAM_FIELDS,
    TrainingCandidatePublisher,
)
from src.backend.recognition.training_candidate_types import (
    TrainingCandidateRecord,
)


pytestmark = pytest.mark.asyncio

NOW = datetime(
    2026,
    9,
    3,
    11,
    0,
    tzinfo=UTC,
)


def candidate() -> TrainingCandidateRecord:
    return TrainingCandidateRecord(
        sample_id=(
            "12345678-1234-4234-8234-123456789012"
        ),
        user_id=11,
        utterance_id=22,
        video_id=33,
        model_version="bundle-v1",
        predicted_phrase_code="REQUEST_HELP",
        confidence=0.91,
        status="UNLABELED",
        created_at=NOW,
        updated_at=NOW,
        published_at=None,
    )


class FakeRepository:
    def __init__(self) -> None:
        self.candidates = (candidate(),)
        self.marked: list[tuple[str, datetime]] = []
        self.reset_calls = 0

    async def list_unpublished_training_candidates(
        self,
        *,
        limit: int,
    ):
        return self.candidates[:limit]

    async def mark_training_candidate_published(
        self,
        *,
        sample_id: str,
        published_at: datetime,
    ):
        self.marked.append(
            (
                sample_id,
                published_at,
            )
        )

        return True

    async def reset_training_candidate_publications(
        self,
    ):
        self.reset_calls += 1
        return len(self.candidates)


class FakeRedis:
    def __init__(
        self,
        *,
        stream_exists: bool = True,
    ) -> None:
        self.stream_exists = stream_exists
        self.eval_calls = []
        self.deleted = []

    async def exists(
        self,
        key: str,
    ):
        assert key == TRAINING_CANDIDATE_STREAM
        return int(self.stream_exists)

    async def delete(
        self,
        key: str,
    ):
        self.deleted.append(key)
        return 1

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

        return "1700000000000-0"


def make_publisher(
    repository: FakeRepository,
    redis: FakeRedis,
):
    publisher = object.__new__(
        TrainingCandidatePublisher
    )

    publisher._repository = repository
    publisher._redis = redis
    publisher._clock = lambda: NOW

    return publisher


async def test_stream_schema_is_exact_old_contract():
    assert TRAINING_CANDIDATE_STREAM_FIELDS == (
        "sample_id",
        "user_id",
        "utterance_id",
        "video_id",
        "model_version",
        "predicted_phrase_code",
        "confidence",
        "created_at",
    )

    assert "object_key" not in (
        TRAINING_CANDIDATE_STREAM_FIELDS
    )

    assert "status" not in (
        TRAINING_CANDIDATE_STREAM_FIELDS
    )


async def test_publish_pending_marks_durable_row_after_redis():
    repository = FakeRepository()
    redis = FakeRedis()

    publisher = make_publisher(
        repository,
        redis,
    )

    count = await publisher.publish_pending_once()

    assert count == 1
    assert len(redis.eval_calls) == 1

    _script, numkeys, args = redis.eval_calls[0]

    assert numkeys == 2
    assert args[1] == TRAINING_CANDIDATE_STREAM

    # Lua keys are followed by exactly the 8 contract values.
    values = args[2:]

    assert values == (
        candidate().sample_id,
        "11",
        "22",
        "33",
        "bundle-v1",
        "REQUEST_HELP",
        "0.91",
        NOW.isoformat(),
    )

    assert repository.marked == [
        (
            candidate().sample_id,
            NOW,
        )
    ]


async def test_missing_redis_stream_resets_pg_projection():
    repository = FakeRepository()

    redis = FakeRedis(
        stream_exists=False
    )

    publisher = make_publisher(
        repository,
        redis,
    )

    reset_count = (
        await publisher.restore_projection_if_needed()
    )

    assert reset_count == 1
    assert repository.reset_calls == 1
    assert len(redis.deleted) == 1


async def test_existing_stream_does_not_reset_pg():
    repository = FakeRepository()
    redis = FakeRedis(stream_exists=True)

    publisher = make_publisher(
        repository,
        redis,
    )

    reset_count = (
        await publisher.restore_projection_if_needed()
    )

    assert reset_count == 0
    assert repository.reset_calls == 0
    assert redis.deleted == []
