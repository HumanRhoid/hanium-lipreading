"""Durable training-candidate records."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TrainingCandidateRecord:
    """PostgreSQL source-of-truth training candidate."""

    sample_id: str
    user_id: int
    utterance_id: int
    video_id: int
    model_version: str | None
    predicted_phrase_code: str | None
    confidence: float | None
    status: str
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None
