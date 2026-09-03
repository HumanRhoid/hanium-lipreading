"""Personalization data-transfer records."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PhraseUsageRecord:
    """PostgreSQL source-of-truth phrase usage statistics."""

    user_id: int
    phrase_code: str
    usage_count: int
    accepted_count: int
    corrected_count: int
    last_used_at: datetime | None
    updated_at: datetime
