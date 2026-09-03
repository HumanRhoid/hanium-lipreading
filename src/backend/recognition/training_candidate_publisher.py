"""Durable PostgreSQL -> Redis training-candidate outbox."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Final

from redis.asyncio import Redis

from src.backend.core.config import Settings
from src.backend.recognition.training_candidate_types import (
    TrainingCandidateRecord,
)

logger = logging.getLogger(__name__)

TRAINING_CANDIDATE_STREAM: Final = (
    "stream:training:candidates"
)

TRAINING_CANDIDATE_PUBLISH_MARKER_KEY: Final = (
    "training:candidates:published"
)

TRAINING_CANDIDATE_STREAM_FIELDS: Final = (
    "sample_id",
    "user_id",
    "utterance_id",
    "video_id",
    "model_version",
    "predicted_phrase_code",
    "confidence",
    "created_at",
)

DEFAULT_TRAINING_CANDIDATE_BATCH_SIZE: Final = 100
DEFAULT_TRAINING_CANDIDATE_INTERVAL_SECONDS: Final = 5.0


_PUBLISH_SCRIPT: Final = """
local existing = redis.call(
    "HGET",
    KEYS[1],
    ARGV[1]
)

if existing then
    return existing
end

local stream_id = redis.call(
    "XADD",
    KEYS[2],
    "*",
    "sample_id", ARGV[1],
    "user_id", ARGV[2],
    "utterance_id", ARGV[3],
    "video_id", ARGV[4],
    "model_version", ARGV[5],
    "predicted_phrase_code", ARGV[6],
    "confidence", ARGV[7],
    "created_at", ARGV[8]
)

redis.call(
    "HSET",
    KEYS[1],
    ARGV[1],
    stream_id
)

return stream_id
"""


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TrainingCandidatePublisher:
    """Publish durable PG candidates to the Redis queue."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository,
        clock: Clock = _utc_now,
    ) -> None:
        self._repository = repository
        self._clock = clock

        self._redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def close(self) -> None:
        await self._redis.aclose()

    async def restore_projection_if_needed(
        self,
    ) -> int:
        """Make PG candidates publishable again after Redis loss."""

        stream_exists = await self._redis.exists(
            TRAINING_CANDIDATE_STREAM
        )

        if stream_exists:
            return 0

        # If the stream disappeared, the Redis publication
        # markers are no longer trustworthy either.
        await self._redis.delete(
            TRAINING_CANDIDATE_PUBLISH_MARKER_KEY
        )

        return (
            await self._repository
            .reset_training_candidate_publications()
        )

    async def publish_pending_once(
        self,
        *,
        limit: int = (
            DEFAULT_TRAINING_CANDIDATE_BATCH_SIZE
        ),
    ) -> int:
        """Publish one bounded outbox batch."""

        candidates = (
            await self._repository
            .list_unpublished_training_candidates(
                limit=limit
            )
        )

        published_count = 0

        for candidate in candidates:
            try:
                await self._publish_candidate(
                    candidate
                )

                marked = await self._repository.mark_training_candidate_published(
                    sample_id=candidate.sample_id,
                    published_at=self._clock(),
                )

                if marked:
                    published_count += 1

            except asyncio.CancelledError:
                raise

            except Exception:
                # sample_id is an internal random identifier.
                # Do not log object_key, user name or binary data.
                logger.exception(
                    "Training candidate publication failed: "
                    "sample_id=%s",
                    candidate.sample_id,
                )

                # A Redis outage will affect the whole batch.
                # Leave PG rows unpublished for the next retry.
                break

        return published_count

    async def _publish_candidate(
        self,
        candidate: TrainingCandidateRecord,
    ) -> str:
        result = await self._redis.eval(
            _PUBLISH_SCRIPT,
            2,
            TRAINING_CANDIDATE_PUBLISH_MARKER_KEY,
            TRAINING_CANDIDATE_STREAM,
            candidate.sample_id,
            str(candidate.user_id),
            str(candidate.utterance_id),
            str(candidate.video_id),
            candidate.model_version or "",
            candidate.predicted_phrase_code or "",
            (
                str(candidate.confidence)
                if candidate.confidence is not None
                else ""
            ),
            candidate.created_at.isoformat(),
        )

        if not isinstance(result, str) or not result:
            raise RuntimeError(
                "Redis training candidate publication "
                "returned an invalid stream id."
            )

        return result


class TrainingCandidatePublisherRunner:
    """Retry candidate outbox delivery without blocking inference."""

    def __init__(
        self,
        *,
        publisher: TrainingCandidatePublisher,
        interval_seconds: float = (
            DEFAULT_TRAINING_CANDIDATE_INTERVAL_SECONDS
        ),
        batch_size: int = (
            DEFAULT_TRAINING_CANDIDATE_BATCH_SIZE
        ),
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(
                "interval_seconds must be positive."
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        self._publisher = publisher
        self._interval_seconds = interval_seconds
        self._batch_size = batch_size
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if (
            self._task is not None
            and not self._task.done()
        ):
            return

        self._task = asyncio.create_task(
            self._run(),
            name="training-candidate-publisher",
        )

    async def close(self) -> None:
        task = self._task
        self._task = None

        if task is None:
            return

        task.cancel()

        with suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        while True:
            try:
                await (
                    self._publisher
                    .restore_projection_if_needed()
                )

                while True:
                    published = (
                        await self._publisher
                        .publish_pending_once(
                            limit=self._batch_size
                        )
                    )

                    if published < self._batch_size:
                        break

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "Training candidate publisher "
                    "iteration failed."
                )

            await asyncio.sleep(
                self._interval_seconds
            )
