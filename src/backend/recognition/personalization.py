"""Redis Sorted Set projection for phrase personalization."""

from dataclasses import dataclass
from typing import Final

from redis.asyncio import Redis

from src.backend.core.config import Settings
from src.backend.recognition.personalization_types import (
    PhraseUsageRecord,
)

PERSONALIZATION_KEY_PREFIX: Final = "personalization:"
PERSONALIZATION_KEY_SUFFIX: Final = ":phrases"


@dataclass(frozen=True, slots=True)
class PersonalizedPhrase:
    """One phrase ranked by the simple usage-count score."""

    phrase_code: str
    score: float


class RedisPersonalizationStore:
    """Synchronize PostgreSQL phrase usage into Redis Sorted Sets."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository,
    ) -> None:
        self._repository = repository

        self._redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def close(self) -> None:
        await self._redis.aclose()

    @staticmethod
    def key_for_user(
        user_id: int,
    ) -> str:
        if user_id < 1:
            raise ValueError(
                "user_id must be a positive integer."
            )

        return (
            f"{PERSONALIZATION_KEY_PREFIX}"
            f"{user_id}"
            f"{PERSONALIZATION_KEY_SUFFIX}"
        )

    async def sync_from_postgres(
        self,
        *,
        user_id: int,
    ) -> tuple[PhraseUsageRecord, ...]:
        """Rebuild one Redis personalization set from PostgreSQL."""

        stats = (
            await self._repository.list_phrase_usage_stats(
                user_id=user_id
            )
        )

        key = self.key_for_user(user_id)

        # PostgreSQL is authoritative. Delete stale Redis members
        # first; a later call can always rebuild after Redis loss.
        await self._redis.delete(key)

        if stats:
            await self._redis.zadd(
                key,
                {
                    stat.phrase_code: float(
                        stat.usage_count
                    )
                    for stat in stats
                },
            )

        return stats

    async def get_top_phrases(
        self,
        *,
        user_id: int,
        limit: int = 10,
    ) -> tuple[PersonalizedPhrase, ...]:
        """Return ranked phrases after synchronizing the PG source."""

        if limit < 1 or limit > 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        # There is no durable Redis write path in the current
        # Worker contract. Synchronizing here guarantees Redis
        # never becomes a stale source of truth and also restores
        # the set after Redis is flushed.
        await self.sync_from_postgres(
            user_id=user_id
        )

        key = self.key_for_user(user_id)

        rows = await self._redis.zrevrange(
            key,
            0,
            limit - 1,
            withscores=True,
        )

        return tuple(
            PersonalizedPhrase(
                phrase_code=str(phrase_code),
                score=float(score),
            )
            for phrase_code, score in rows
        )
