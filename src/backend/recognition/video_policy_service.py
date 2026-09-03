"""Training-consent and private video retention operations."""

from collections.abc import Callable
from datetime import UTC, datetime

from src.backend.recognition.ports import ObjectStorage
from src.backend.recognition.video_policy_types import (
    UserConsentRecord,
    VideoPolicyAssetRecord,
)


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class VideoNotFoundError(Exception):
    """The requested video does not exist for this user."""


class VideoPolicyService:
    """Coordinate PostgreSQL video metadata and Object Storage."""

    def __init__(
        self,
        *,
        repository,
        object_storage: ObjectStorage,
        clock: Clock = _utc_now,
    ) -> None:
        self._repository = repository
        self._object_storage = object_storage
        self._clock = clock

    async def get_consent(
        self,
        *,
        user_id: int,
    ) -> UserConsentRecord | None:
        return await self._repository.get_user_consent(
            user_id=user_id
        )

    async def update_consent(
        self,
        *,
        user_id: int,
        model_training_consent: bool,
        consent_version: str,
    ) -> UserConsentRecord:
        consent = await self._repository.upsert_user_consent(
            user_id=user_id,
            model_training_consent=(
                model_training_consent
            ),
            consent_version=consent_version,
        )

        # Withdrawal takes effect immediately for policy
        # decisions. Existing training-purpose objects are
        # removed with the durable DELETE_PENDING flow.
        if not model_training_consent:
            await self.delete_all_videos(
                user_id=user_id,
                storage_purpose="MODEL_TRAINING",
            )

        return consent

    async def list_videos(
        self,
        *,
        user_id: int,
    ) -> tuple[VideoPolicyAssetRecord, ...]:
        return await self._repository.list_user_video_assets(
            user_id=user_id,
            include_deleted=True,
        )

    async def delete_video(
        self,
        *,
        user_id: int,
        video_id: int,
    ) -> bool:
        asset = (
            await self._repository.mark_video_delete_pending(
                user_id=user_id,
                video_id=video_id,
            )
        )

        if asset is None:
            raise VideoNotFoundError

        if asset.storage_status == "DELETED":
            return False

        # S3-compatible delete is idempotent. If this succeeds
        # but the final DB update fails, retrying safely deletes
        # the same key again and can finalize the metadata.
        await self._object_storage.delete(
            asset.object_key
        )

        finalized = await self._repository.mark_video_deleted(
            user_id=user_id,
            video_id=video_id,
            deleted_at=self._clock(),
        )

        if finalized is None:
            raise RuntimeError(
                "Video disappeared while finalizing deletion."
            )

        return True

    async def delete_all_videos(
        self,
        *,
        user_id: int,
        storage_purpose: str | None = None,
    ) -> int:
        assets = (
            await self._repository.list_user_video_assets(
                user_id=user_id,
                include_deleted=False,
            )
        )

        deleted_count = 0

        for asset in assets:
            if (
                storage_purpose is not None
                and asset.storage_purpose
                != storage_purpose
            ):
                continue

            deleted = await self.delete_video(
                user_id=user_id,
                video_id=asset.video_id,
            )

            if deleted:
                deleted_count += 1

        return deleted_count
