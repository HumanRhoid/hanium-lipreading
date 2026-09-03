"""Automatic cleanup for temporary inference videos."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime

from src.backend.recognition.ports import ObjectStorage
from src.backend.recognition.video_policy_service import (
    VideoPolicyService,
)

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]

DEFAULT_VIDEO_RETENTION_CLEANUP_INTERVAL_SECONDS = 60.0
DEFAULT_VIDEO_RETENTION_CLEANUP_BATCH_SIZE = 100


def _utc_now() -> datetime:
    return datetime.now(UTC)


class VideoRetentionCleanupService:
    """Delete temporary videos after inference or retention expiry."""

    def __init__(
        self,
        *,
        repository,
        object_storage: ObjectStorage,
        clock: Clock = _utc_now,
    ) -> None:
        self._repository = repository
        self._clock = clock

        self._video_policy_service = VideoPolicyService(
            repository=repository,
            object_storage=object_storage,
            clock=clock,
        )

    async def cleanup_once(
        self,
        *,
        limit: int = DEFAULT_VIDEO_RETENTION_CLEANUP_BATCH_SIZE,
    ) -> int:
        """Process one bounded cleanup batch."""

        candidates = (
            await self._repository.list_video_assets_due_for_cleanup(
                now=self._clock(),
                limit=limit,
            )
        )

        deleted_count = 0

        for asset in candidates:
            try:
                deleted = (
                    await self._video_policy_service.delete_video(
                        user_id=asset.user_id,
                        video_id=asset.video_id,
                    )
                )

                if deleted:
                    deleted_count += 1

            except asyncio.CancelledError:
                raise

            except Exception:
                # Do not log object_key or video binary.
                logger.exception(
                    "Automatic video cleanup failed "
                    "for video_id=%s user_id=%s",
                    asset.video_id,
                    asset.user_id,
                )

        return deleted_count


class VideoRetentionCleanupRunner:
    """Run temporary-video cleanup periodically in the API process."""

    def __init__(
        self,
        *,
        service: VideoRetentionCleanupService,
        interval_seconds: float = (
            DEFAULT_VIDEO_RETENTION_CLEANUP_INTERVAL_SECONDS
        ),
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(
                "interval_seconds must be positive."
            )

        self._service = service
        self._interval_seconds = interval_seconds
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the cleanup loop exactly once."""

        if (
            self._task is not None
            and not self._task.done()
        ):
            return

        self._task = asyncio.create_task(
            self._run(),
            name="video-retention-cleanup",
        )

    async def close(self) -> None:
        """Stop the cleanup loop without leaking a task."""

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
                await self._service.cleanup_once()

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "Automatic video retention cleanup "
                    "batch failed."
                )

            await asyncio.sleep(
                self._interval_seconds
            )
