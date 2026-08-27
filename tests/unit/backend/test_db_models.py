"""ERD 명세를 실행 가능한 SQLAlchemy metadata로 검증한다."""

from sqlalchemy import CheckConstraint, UniqueConstraint

from src.backend.auth.adapters.repository import User
from src.backend.core.database import Base
from src.backend.recognition.adapters.repository import (
    Phrase,
    PhraseUsageStat,
    RecognitionSession,
    UserConsent,
    Utterance,
    VideoAsset,
)


def constraint_names(table, constraint_type):
    return {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, constraint_type)
    }


def test_metadata_contains_expected_tables():
    assert set(Base.metadata.tables) == {
        "session",
        "phrase",
        "utterance",
        "users",
        "login_session",
        "video_asset",
        "user_consent",
        "phrase_usage_stat",
    }


def test_user_matches_account_contract():
    table = User.__table__

    assert set(table.columns.keys()) == {
        "user_id",
        "username",
        "password_hash",
        "display_name",
        "created_at",
    }

    assert table.c.user_id.primary_key is True
    assert table.c.username.nullable is False
    assert table.c.password_hash.nullable is False
    assert table.c.display_name.nullable is False

    assert "uq_users_username" in constraint_names(
        table,
        UniqueConstraint,
    )


def test_session_matches_erd_contract():
    table = RecognitionSession.__table__

    assert set(table.columns.keys()) == {
        "session_id",
        "mode",
        "started_at",
        "ended_at",
    }

    assert table.c.session_id.primary_key is True
    assert table.c.ended_at.nullable is True

    assert "ck_session_mode" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "ck_session_ended_after_started" in constraint_names(
        table,
        CheckConstraint,
    )

    assert {index.name for index in table.indexes} == {
        "ix_session_started_at",
    }


def test_phrase_uses_stable_unique_code():
    table = Phrase.__table__

    assert set(table.columns.keys()) == {
        "phrase_id",
        "phrase_code",
        "phrase_text",
        "category",
    }

    assert "uq_phrase_phrase_code" in constraint_names(
        table,
        UniqueConstraint,
    )

    assert "ck_phrase_category" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "ck_phrase_text_not_blank" in constraint_names(
        table,
        CheckConstraint,
    )


def test_utterance_supports_async_recognition_contract():
    table = Utterance.__table__

    assert set(table.columns.keys()) == {
        "utt_id",
        "user_id",
        "session_id",
        "phrase_id",
        "raw_text",
        "corrected_text",
        "confidence",
        "model_version",
        "created_at",
    }

    assert table.c.utt_id.primary_key is True

    # 기존 WebSocket 방식과 새 비동기 업로드 방식을 함께 지원한다.
    assert table.c.user_id.nullable is True
    assert table.c.session_id.nullable is True

    # 추론 전에 utterance를 먼저 만들 수 있어야 한다.
    assert table.c.raw_text.nullable is True

    assert table.c.phrase_id.nullable is True
    assert table.c.corrected_text.nullable is True
    assert table.c.confidence.nullable is True
    assert table.c.model_version.nullable is True

    assert "ck_utterance_confidence_range" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "ck_utterance_corrected_text_not_blank" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "ck_utterance_raw_text_not_blank" in constraint_names(
        table,
        CheckConstraint,
    )

    foreign_keys = {
        foreign_key.parent.name: foreign_key for foreign_key in table.foreign_keys
    }

    assert foreign_keys["user_id"].target_fullname == "users.user_id"
    assert foreign_keys["user_id"].ondelete == "CASCADE"

    assert foreign_keys["session_id"].target_fullname == "session.session_id"
    assert foreign_keys["session_id"].ondelete == "CASCADE"

    assert foreign_keys["phrase_id"].target_fullname == "phrase.phrase_id"
    assert foreign_keys["phrase_id"].ondelete == "SET NULL"


def test_utterance_has_expected_indexes():
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in Utterance.__table__.indexes
    }

    assert indexes == {
        "ix_utterance_phrase_id": ("phrase_id",),
        "ix_utterance_session_created_at": (
            "session_id",
            "created_at",
        ),
        "ix_utterance_user_created_at": (
            "user_id",
            "created_at",
        ),
    }


def test_video_asset_matches_storage_contract():
    table = VideoAsset.__table__

    assert set(table.columns.keys()) == {
        "video_id",
        "user_id",
        "utterance_id",
        "idempotency_key",
        "object_key",
        "original_mime_type",
        "normalized_mime_type",
        "codec",
        "width",
        "height",
        "fps",
        "duration_ms",
        "size_bytes",
        "checksum",
        "storage_status",
        "storage_purpose",
        "consent_version",
        "created_at",
        "retention_until",
        "deleted_at",
    }

    assert table.c.video_id.primary_key is True
    assert table.c.user_id.nullable is False
    assert table.c.utterance_id.nullable is False
    assert table.c.idempotency_key.nullable is False
    assert table.c.object_key.nullable is False
    assert table.c.original_mime_type.nullable is False
    assert table.c.size_bytes.nullable is False
    assert table.c.checksum.nullable is False
    assert table.c.storage_status.nullable is False
    assert table.c.storage_purpose.nullable is False

    assert "ck_video_asset_storage_status" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "ck_video_asset_storage_purpose" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "uq_video_asset_user_id_idempotency_key" in constraint_names(
        table,
        UniqueConstraint,
    )

    foreign_keys = {
        foreign_key.parent.name: foreign_key for foreign_key in table.foreign_keys
    }

    assert foreign_keys["user_id"].target_fullname == "users.user_id"
    assert foreign_keys["user_id"].ondelete == "CASCADE"

    assert foreign_keys["utterance_id"].target_fullname == "utterance.utt_id"
    assert foreign_keys["utterance_id"].ondelete == "CASCADE"


def test_video_asset_has_expected_indexes():
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in VideoAsset.__table__.indexes
    }

    assert indexes == {
        "ix_video_asset_retention_until": ("retention_until",),
        "ix_video_asset_user_created_at": (
            "user_id",
            "created_at",
        ),
    }


def test_user_consent_matches_storage_contract():
    table = UserConsent.__table__

    assert set(table.columns.keys()) == {
        "user_id",
        "model_training_consent",
        "consent_version",
        "created_at",
        "updated_at",
    }

    assert table.c.user_id.primary_key is True
    assert table.c.model_training_consent.nullable is False
    assert table.c.consent_version.nullable is False
    assert table.c.created_at.nullable is False
    assert table.c.updated_at.nullable is False

    foreign_keys = {
        foreign_key.parent.name: foreign_key for foreign_key in table.foreign_keys
    }

    assert foreign_keys["user_id"].target_fullname == "users.user_id"
    assert foreign_keys["user_id"].ondelete == "CASCADE"


def test_phrase_usage_stat_matches_personalization_contract():
    table = PhraseUsageStat.__table__

    assert set(table.columns.keys()) == {
        "user_id",
        "phrase_code",
        "usage_count",
        "accepted_count",
        "corrected_count",
        "last_used_at",
        "updated_at",
    }

    primary_key_columns = {column.name for column in table.primary_key.columns}

    assert primary_key_columns == {
        "user_id",
        "phrase_code",
    }

    assert table.c.usage_count.nullable is False
    assert table.c.accepted_count.nullable is False
    assert table.c.corrected_count.nullable is False
    assert table.c.last_used_at.nullable is True
    assert table.c.updated_at.nullable is False

    assert "ck_phrase_usage_stat_usage_count_nonnegative" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "ck_phrase_usage_stat_accepted_count_nonnegative" in constraint_names(
        table,
        CheckConstraint,
    )

    assert "ck_phrase_usage_stat_corrected_count_nonnegative" in constraint_names(
        table,
        CheckConstraint,
    )

    foreign_keys = {
        foreign_key.parent.name: foreign_key for foreign_key in table.foreign_keys
    }

    assert foreign_keys["user_id"].target_fullname == "users.user_id"
    assert foreign_keys["user_id"].ondelete == "CASCADE"

    assert foreign_keys["phrase_code"].target_fullname == "phrase.phrase_code"
    assert foreign_keys["phrase_code"].ondelete == "CASCADE"
