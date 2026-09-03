"""Redis personalization projection tests."""

from datetime import UTC, datetime

import pytest

from src.backend.recognition.personalization import (
    RedisPersonalizationStore,
)
from src.backend.recognition.personalization_types import (
    PhraseUsageRecord,
)

pytestmark = pytest.mark.asyncio

NOW = datetime(
    2026,
    9,
    3,
    10,
    0,
    tzinfo=UTC,
)


def stat(
    phrase_code: str,
    usage_count: int,
) -> PhraseUsageRecord:
    return PhraseUsageRecord(
        user_id=1,
        phrase_code=phrase_code,
        usage_count=usage_count,
        accepted_count=0,
        corrected_count=0,
        last_used_at=NOW,
        updated_at=NOW,
    )


class FakeRepository:
    def __init__(self) -> None:
        self.stats = (
            stat("REQUEST_HELP", 5),
            stat("PAIN_GENERAL", 2),
        )

        self.calls: list[int] = []

    async def list_phrase_usage_stats(
        self,
        *,
        user_id: int,
    ):
        self.calls.append(user_id)
        return self.stats


class FakeRedis:
    def __init__(self) -> None:
        self.sorted_sets: dict[
            str,
            dict[str, float],
        ] = {}

        self.deleted: list[str] = []

    async def delete(
        self,
        key: str,
    ):
        self.deleted.append(key)
        self.sorted_sets.pop(key, None)
        return 1

    async def zadd(
        self,
        key: str,
        mapping: dict[str, float],
    ):
        self.sorted_sets[key] = dict(mapping)
        return len(mapping)

    async def zrevrange(
        self,
        key: str,
        start: int,
        end: int,
        *,
        withscores: bool,
    ):
        assert withscores is True

        values = sorted(
            self.sorted_sets.get(
                key,
                {},
            ).items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        if end >= 0:
            values = values[start : end + 1]
        else:
            values = values[start:]

        return values


def make_store(
    repository: FakeRepository,
    redis: FakeRedis,
):
    store = object.__new__(
        RedisPersonalizationStore
    )

    store._repository = repository
    store._redis = redis

    return store


async def test_sync_uses_required_sorted_set_key_and_usage_score():
    repository = FakeRepository()
    redis = FakeRedis()

    store = make_store(
        repository,
        redis,
    )

    await store.sync_from_postgres(
        user_id=1
    )

    assert (
        redis.sorted_sets[
            "personalization:1:phrases"
        ]
        == {
            "REQUEST_HELP": 5.0,
            "PAIN_GENERAL": 2.0,
        }
    )


async def test_get_top_phrases_ranks_by_usage_count():
    repository = FakeRepository()
    redis = FakeRedis()

    store = make_store(
        repository,
        redis,
    )

    result = await store.get_top_phrases(
        user_id=1,
        limit=2,
    )

    assert [
        item.phrase_code
        for item in result
    ] == [
        "REQUEST_HELP",
        "PAIN_GENERAL",
    ]

    assert [
        item.score
        for item in result
    ] == [
        5.0,
        2.0,
    ]


async def test_redis_loss_is_rebuilt_from_postgres():
    repository = FakeRepository()
    redis = FakeRedis()

    store = make_store(
        repository,
        redis,
    )

    await store.get_top_phrases(
        user_id=1
    )

    redis.sorted_sets.clear()

    restored = await store.get_top_phrases(
        user_id=1
    )

    assert restored

    assert (
        "personalization:1:phrases"
        in redis.sorted_sets
    )

    assert repository.calls == [
        1,
        1,
    ]


async def test_empty_postgres_removes_stale_redis_members():
    repository = FakeRepository()
    redis = FakeRedis()

    redis.sorted_sets[
        "personalization:1:phrases"
    ] = {
        "STALE": 999.0,
    }

    repository.stats = ()

    store = make_store(
        repository,
        redis,
    )

    result = await store.get_top_phrases(
        user_id=1
    )

    assert result == ()

    assert (
        "personalization:1:phrases"
        not in redis.sorted_sets
    )
