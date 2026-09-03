"""User consent and private video policy DTOs."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class UserConsentRecord:
    """Persisted model-training consent state."""

    user_id: int
    model_training_consent: bool
    consent_version: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class VideoPolicyAssetRecord:
    """Internal video metadata used by the video policy service."""

    video_id: int
    user_id: int
    utterance_id: int
    object_key: str
    original_mime_type: str
    normalized_mime_type: str | None
    codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    duration_ms: int | None
    size_bytes: int
    checksum: str
    storage_status: str
    storage_purpose: str
    consent_version: str | None
    created_at: datetime
    retention_until: datetime | None
    deleted_at: datetime | None
