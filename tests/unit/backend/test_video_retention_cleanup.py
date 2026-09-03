"""Automatic temporary-video retention cleanup tests."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from src.backend.recognition.video_policy_types import (
    VideoPolicyAssetRecord,
)
from src.backend.recognition.video_retention_cleanup import (
    VideoRetentionCleanupService,
)

pytestmark = pytest.mark.asyncio

NOW = datetime(
    2026,
    9,
    3,
    8,
    0,
    tzinfo=UTC,
)


def make_asset(
    *,
    video_id: int,
    user_id: int = 1,
    storage_status: str = "READY",
    storage_purpose: str = "TEMPORARY_INFERENCE",
) -> VideoPolicyAssetRecord:
    return VideoPolicyAssetRecord(
        video_id=video_id,
        user_id=user_id,
        utterance_id=video_id + 100,
        object_key=f"private/{video_id}.webm",
        original_mime_type="video/webm",
        normalized_mime_type=None,
        codec=None,
        width=None,
        height=None,
        fps=None,
        duration_ms=None,
        size_bytes=1000,
        checksum="a" * 64,
        storage_status=storage_status,
        storage_purpose=storage_purpose,
        consent_version=None,
        created_at=NOW,
        retention_until=None,
        deleted_at=None,
    )


class FakeStorage:
    def __init__(self) -> None:
        self.deleted_keys: list[str] = []
        self.fail_keys: set[str] = set()

    async def delete(
        self,
        object_key: str,
    ) -> None:
        self.deleted_keys.append(object_key)

        if object_key in self.fail_keys:
            raise RuntimeError("storage failure")


class FakeRepository:
    def __init__(
        self,
        candidates: list[VideoPolicyAssetRecord],
    ) -> None:
        self.assets = {
            asset.video_id: asset
            for asset in candidates
        }

        self.last_now: datetime | None = None
        self.last_limit: int | None = None

    async def list_video_assets_due_for_cleanup(
        self,
        *,
        now: datetime,
        limit: int,
    ):
        self.last_now = now
        self.last_limit = limit

        return tuple(
            list(self.assets.values())[:limit]
        )

    async def mark_video_delete_pending(
        self,
        *,
        user_id: int,
        video_id: int,
    ):
        asset = self.assets.get(video_id)

        if (
            asset is None
            or asset.user_id != user_id
        ):
            return None

        if asset.storage_status != "DELETED":
            asset = replace(
                asset,
                storage_status="DELETE_PENDING",
            )

            self.assets[video_id] = asset

        return asset

    async def mark_video_deleted(
        self,
        *,
        user_id: int,
        video_id: int,
        deleted_at: datetime,
    ):
        asset = self.assets.get(video_id)

        if (
            asset is None
            or asset.user_id != user_id
        ):
            return None

        asset = replace(
            asset,
            storage_status="DELETED",
            deleted_at=deleted_at,
        )

        self.assets[video_id] = asset

        return asset


async def test_cleanup_once_deletes_all_candidates():
    first = make_asset(video_id=10)
    second = make_asset(video_id=11)

    repository = FakeRepository(
        [first, second]
    )

    storage = FakeStorage()

    service = VideoRetentionCleanupService(
        repository=repository,
        object_storage=storage,
        clock=lambda: NOW,
    )

    deleted_count = await service.cleanup_once()

    assert deleted_count == 2

    assert storage.deleted_keys == [
        first.object_key,
        second.object_key,
    ]

    assert (
        repository.assets[10].storage_status
        == "DELETED"
    )

    assert (
        repository.assets[11].storage_status
        == "DELETED"
    )

    assert repository.last_now == NOW
    assert repository.last_limit == 100


async def test_one_storage_failure_does_not_block_other_cleanup():
    first = make_asset(video_id=10)
    second = make_asset(video_id=11)

    repository = FakeRepository(
        [first, second]
    )

    storage = FakeStorage()

    storage.fail_keys.add(
        first.object_key
    )

    service = VideoRetentionCleanupService(
        repository=repository,
        object_storage=storage,
        clock=lambda: NOW,
    )

    deleted_count = await service.cleanup_once()

    assert deleted_count == 1

    assert (
        repository.assets[10].storage_status
        == "DELETE_PENDING"
    )

    assert (
        repository.assets[10].deleted_at
        is None
    )

    assert (
        repository.assets[11].storage_status
        == "DELETED"
    )


async def test_cleanup_respects_batch_limit():
    assets = [
        make_asset(video_id=index)
        for index in range(1, 6)
    ]

    repository = FakeRepository(assets)

    storage = FakeStorage()

    service = VideoRetentionCleanupService(
        repository=repository,
        object_storage=storage,
        clock=lambda: NOW,
    )

    deleted_count = await service.cleanup_once(
        limit=2
    )

    assert deleted_count == 2
    assert len(storage.deleted_keys) == 2
