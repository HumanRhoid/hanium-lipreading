"""S3 호환 Object Storage adapter 단위 테스트."""

from hashlib import sha256
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.backend.core.config import Settings
from src.backend.recognition.adapters.object_storage import S3ObjectStorage


def make_settings() -> Settings:
    return Settings(
        object_storage_endpoint_url="http://localhost:9000",
        object_storage_access_key="test-access",
        object_storage_secret_key="test-secret",
        object_storage_bucket="recognition-videos",
        object_storage_region="us-east-1",
    )


def make_storage():
    client = MagicMock()

    patcher = patch(
        "src.backend.recognition.adapters.object_storage.boto3.client",
        return_value=client,
    )

    patcher.start()

    storage = S3ObjectStorage(make_settings())

    return storage, client, patcher


@pytest.mark.asyncio
async def test_ensure_bucket_checks_private_bucket():
    storage, client, patcher = make_storage()

    try:
        await storage.ensure_bucket()

        client.head_bucket.assert_called_once_with(Bucket="recognition-videos")
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_put_stores_object_with_checksum_metadata():
    storage, client, patcher = make_storage()

    payload = b"video-data"
    checksum = sha256(payload).hexdigest()

    try:
        await storage.put(
            object_key=("recognition-videos/1/2026/08/utterance-1.mp4"),
            data=payload,
            content_type="video/mp4",
            checksum=checksum,
        )

        client.put_object.assert_called_once_with(
            Bucket="recognition-videos",
            Key=("recognition-videos/1/2026/08/utterance-1.mp4"),
            Body=payload,
            ContentType="video/mp4",
            Metadata={
                "sha256": checksum,
            },
        )
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_get_returns_object_bytes_and_closes_body():
    storage, client, patcher = make_storage()

    body = MagicMock()
    body.read.return_value = b"stored-video"

    client.get_object.return_value = {
        "Body": body,
    }

    try:
        result = await storage.get("recognition-videos/1/video.mp4")

        assert result == b"stored-video"

        client.get_object.assert_called_once_with(
            Bucket="recognition-videos",
            Key="recognition-videos/1/video.mp4",
        )

        body.close.assert_called_once()
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_exists_returns_true_when_object_exists():
    storage, client, patcher = make_storage()

    try:
        result = await storage.exists("recognition-videos/1/video.mp4")

        assert result is True

        client.head_object.assert_called_once_with(
            Bucket="recognition-videos",
            Key="recognition-videos/1/video.mp4",
        )
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_exists_returns_false_for_missing_object():
    storage, client, patcher = make_storage()

    client.head_object.side_effect = ClientError(
        {
            "Error": {
                "Code": "404",
                "Message": "Not Found",
            }
        },
        "HeadObject",
    )

    try:
        result = await storage.exists("recognition-videos/1/missing.mp4")

        assert result is False
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_delete_removes_object():
    storage, client, patcher = make_storage()

    try:
        await storage.delete("recognition-videos/1/video.mp4")

        client.delete_object.assert_called_once_with(
            Bucket="recognition-videos",
            Key="recognition-videos/1/video.mp4",
        )
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_put_rejects_invalid_checksum():
    storage, client, patcher = make_storage()

    try:
        with pytest.raises(
            ValueError,
            match="SHA-256",
        ):
            await storage.put(
                object_key="recognition-videos/1/video.mp4",
                data=b"video",
                content_type="video/mp4",
                checksum="invalid",
            )

        client.put_object.assert_not_called()
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_put_rejects_empty_payload():
    storage, client, patcher = make_storage()

    try:
        with pytest.raises(
            ValueError,
            match="빈 객체",
        ):
            await storage.put(
                object_key="recognition-videos/1/video.mp4",
                data=b"",
                content_type="video/mp4",
                checksum=sha256(b"").hexdigest(),
            )

        client.put_object.assert_not_called()
    finally:
        patcher.stop()


@pytest.mark.parametrize(
    "object_key",
    [
        "",
        "/recognition-videos/video.mp4",
        "../video.mp4",
        "recognition-videos/../video.mp4",
        "recognition-videos//video.mp4",
    ],
)
@pytest.mark.asyncio
async def test_object_key_rejects_unsafe_paths(
    object_key: str,
):
    storage, client, patcher = make_storage()

    try:
        with pytest.raises(ValueError):
            await storage.exists(object_key)

        client.head_object.assert_not_called()
    finally:
        patcher.stop()
