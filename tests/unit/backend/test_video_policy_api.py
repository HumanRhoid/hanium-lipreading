"""Current-user consent and video-management API tests."""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.auth.service import InvalidSessionError
from src.backend.recognition.video_policy_api import router
from src.backend.recognition.video_policy_types import (
    UserConsentRecord,
    VideoPolicyAssetRecord,
)

pytestmark = pytest.mark.asyncio

NOW = datetime(
    2026,
    9,
    3,
    7,
    30,
    tzinfo=UTC,
)


def make_asset(
    *,
    video_id: int = 10,
    user_id: int = 1,
    purpose: str = "TEMPORARY_INFERENCE",
    status: str = "READY",
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
        size_bytes=1234,
        checksum="a" * 64,
        storage_status=status,
        storage_purpose=purpose,
        consent_version=(
            "2026-09-v1"
            if purpose == "MODEL_TRAINING"
            else None
        ),
        created_at=NOW,
        retention_until=None,
        deleted_at=(
            NOW
            if status == "DELETED"
            else None
        ),
    )


class FakeAuthService:
    def __init__(
        self,
        *,
        user_id: int = 1,
        invalid: bool = False,
    ) -> None:
        self.user_id = user_id
        self.invalid = invalid

    async def get_current_user(
        self,
        session_token: str,
    ):
        if self.invalid:
            raise InvalidSessionError

        return SimpleNamespace(
            user_id=self.user_id
        )


class FakeObjectStorage:
    def __init__(self) -> None:
        self.deleted_keys: list[str] = []
        self.delete_error: Exception | None = None

    async def delete(
        self,
        object_key: str,
    ) -> None:
        self.deleted_keys.append(object_key)

        if self.delete_error is not None:
            raise self.delete_error


class FakeRepository:
    def __init__(self) -> None:
        self.consent: UserConsentRecord | None = None

        self.assets: dict[
            int,
            VideoPolicyAssetRecord,
        ] = {}

    async def get_user_consent(
        self,
        *,
        user_id: int,
    ):
        if (
            self.consent is not None
            and self.consent.user_id == user_id
        ):
            return self.consent

        return None

    async def upsert_user_consent(
        self,
        *,
        user_id: int,
        model_training_consent: bool,
        consent_version: str,
    ):
        self.consent = UserConsentRecord(
            user_id=user_id,
            model_training_consent=(
                model_training_consent
            ),
            consent_version=consent_version,
            created_at=NOW,
            updated_at=NOW,
        )

        return self.consent

    async def list_user_video_assets(
        self,
        *,
        user_id: int,
        include_deleted: bool = True,
    ):
        assets = [
            asset
            for asset in self.assets.values()
            if asset.user_id == user_id
        ]

        if not include_deleted:
            assets = [
                asset
                for asset in assets
                if asset.storage_status != "DELETED"
            ]

        return tuple(assets)

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


def make_app(
    *,
    repository: FakeRepository | None = None,
    storage: FakeObjectStorage | None = None,
    auth_service: FakeAuthService | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    app.state.repository = (
        repository
        if repository is not None
        else FakeRepository()
    )

    app.state.object_storage = (
        storage
        if storage is not None
        else FakeObjectStorage()
    )

    app.state.auth_service = (
        auth_service
        if auth_service is not None
        else FakeAuthService()
    )

    return app


async def request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    json=None,
    token: str | None = "session-token",
):
    headers = {}

    if token is not None:
        headers["X-Session-Token"] = token

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(
            method,
            path,
            headers=headers,
            json=json,
        )


async def test_get_consent_defaults_to_false():
    response = await request(
        make_app(),
        "GET",
        "/api/v1/users/me/consents",
    )

    assert response.status_code == 200

    assert response.json() == {
        "model_training_consent": False,
        "consent_version": None,
        "updated_at": None,
    }


async def test_missing_session_token_returns_401():
    response = await request(
        make_app(),
        "GET",
        "/api/v1/users/me/consents",
        token=None,
    )

    assert response.status_code == 401


async def test_patch_consent_persists_enabled_state():
    repository = FakeRepository()

    response = await request(
        make_app(repository=repository),
        "PATCH",
        "/api/v1/users/me/consents",
        json={
            "model_training_consent": True,
            "consent_version": "2026-09-v1",
        },
    )

    assert response.status_code == 200

    assert (
        repository.consent
        is not None
    )

    assert (
        repository.consent.model_training_consent
        is True
    )


async def test_withdrawal_deletes_training_videos_only():
    repository = FakeRepository()

    training = make_asset(
        video_id=10,
        purpose="MODEL_TRAINING",
    )

    temporary = make_asset(
        video_id=11,
        purpose="TEMPORARY_INFERENCE",
    )

    repository.assets = {
        10: training,
        11: temporary,
    }

    storage = FakeObjectStorage()

    response = await request(
        make_app(
            repository=repository,
            storage=storage,
        ),
        "PATCH",
        "/api/v1/users/me/consents",
        json={
            "model_training_consent": False,
            "consent_version": "2026-09-v1",
        },
    )

    assert response.status_code == 200

    assert storage.deleted_keys == [
        training.object_key
    ]

    assert (
        repository.assets[10].storage_status
        == "DELETED"
    )

    assert (
        repository.assets[11].storage_status
        == "READY"
    )


async def test_video_list_never_exposes_object_key():
    repository = FakeRepository()

    repository.assets[10] = make_asset()

    response = await request(
        make_app(repository=repository),
        "GET",
        "/api/v1/users/me/videos",
    )

    assert response.status_code == 200

    payload = response.json()

    assert len(payload["videos"]) == 1

    video = payload["videos"][0]

    assert video["video_id"] == 10

    assert "object_key" not in video
    assert "idempotency_key" not in video


async def test_delete_video_deletes_object_then_metadata():
    repository = FakeRepository()

    asset = make_asset(video_id=10)

    repository.assets[10] = asset

    storage = FakeObjectStorage()

    response = await request(
        make_app(
            repository=repository,
            storage=storage,
        ),
        "DELETE",
        "/api/v1/users/me/videos/10",
    )

    assert response.status_code == 204

    assert storage.deleted_keys == [
        asset.object_key
    ]

    assert (
        repository.assets[10].storage_status
        == "DELETED"
    )

    assert (
        repository.assets[10].deleted_at
        is not None
    )


async def test_unowned_video_returns_404():
    repository = FakeRepository()

    repository.assets[10] = make_asset(
        video_id=10,
        user_id=2,
    )

    response = await request(
        make_app(repository=repository),
        "DELETE",
        "/api/v1/users/me/videos/10",
    )

    assert response.status_code == 404


async def test_storage_failure_leaves_delete_pending():
    repository = FakeRepository()

    repository.assets[10] = make_asset(
        video_id=10
    )

    storage = FakeObjectStorage()
    storage.delete_error = RuntimeError(
        "storage unavailable"
    )

    response = await request(
        make_app(
            repository=repository,
            storage=storage,
        ),
        "DELETE",
        "/api/v1/users/me/videos/10",
    )

    assert response.status_code == 500

    assert (
        repository.assets[10].storage_status
        == "DELETE_PENDING"
    )

    assert repository.assets[10].deleted_at is None


async def test_delete_all_only_deletes_owned_active_videos():
    repository = FakeRepository()

    repository.assets = {
        10: make_asset(video_id=10),
        11: make_asset(video_id=11),
        12: make_asset(
            video_id=12,
            user_id=2,
        ),
        13: make_asset(
            video_id=13,
            status="DELETED",
        ),
    }

    storage = FakeObjectStorage()

    response = await request(
        make_app(
            repository=repository,
            storage=storage,
        ),
        "DELETE",
        "/api/v1/users/me/videos",
    )

    assert response.status_code == 200

    assert response.json() == {
        "deleted_count": 2
    }

    assert (
        repository.assets[10].storage_status
        == "DELETED"
    )

    assert (
        repository.assets[11].storage_status
        == "DELETED"
    )

    assert (
        repository.assets[12].storage_status
        != "DELETED"
    )
