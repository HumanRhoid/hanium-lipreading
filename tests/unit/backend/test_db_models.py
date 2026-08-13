"""ERD 명세를 실행 가능한 SQLAlchemy metadata로 검증한다."""

from sqlalchemy import CheckConstraint, UniqueConstraint

from src.backend.auth.adapters import repository as auth_repository  # noqa: F401
from src.backend.core.database import Base
from src.backend.recognition.adapters.repository import (
    Phrase,
    RecognitionSession,
    Utterance,
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
    }


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
    assert "ck_session_mode" in constraint_names(table, CheckConstraint)
    assert "ck_session_ended_after_started" in constraint_names(
        table,
        CheckConstraint,
    )
    assert {index.name for index in table.indexes} == {"ix_session_started_at"}


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


def test_utterance_matches_storage_and_deletion_contract():
    table = Utterance.__table__

    assert set(table.columns.keys()) == {
        "utt_id",
        "session_id",
        "phrase_id",
        "raw_text",
        "corrected_text",
        "confidence",
        "created_at",
    }
    assert table.c.phrase_id.nullable is True
    assert "ck_utterance_confidence_range" in constraint_names(
        table,
        CheckConstraint,
    )

    foreign_keys = {
        foreign_key.parent.name: foreign_key for foreign_key in table.foreign_keys
    }
    assert foreign_keys["session_id"].ondelete == "CASCADE"
    assert foreign_keys["phrase_id"].ondelete == "SET NULL"


def test_utterance_has_session_created_at_index():
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in Utterance.__table__.indexes
    }

    assert indexes["ix_utterance_session_created_at"] == (
        "session_id",
        "created_at",
    )
    assert indexes["ix_utterance_phrase_id"] == ("phrase_id",)
