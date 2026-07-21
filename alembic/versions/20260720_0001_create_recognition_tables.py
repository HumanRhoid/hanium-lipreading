"""인식 세션, 문구와 발화 테이블 생성.

Revision ID: 20260720_0001
Revises:
Create Date: 2026-07-20
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260720_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """ERD 명세의 세 테이블과 제약조건을 생성한다."""

    op.create_table(
        "session",
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.Identity(),
            nullable=False,
        ),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "ended_at IS NULL OR ended_at >= started_at",
            name="ended_after_started",
        ),
        sa.CheckConstraint(
            "mode IN ('CLOSED', 'OPEN')",
            name="mode",
        ),
        sa.PrimaryKeyConstraint("session_id", name="pk_session"),
    )
    op.create_table(
        "phrase",
        sa.Column(
            "phrase_id",
            sa.Integer(),
            sa.Identity(),
            nullable=False,
        ),
        sa.Column("phrase_code", sa.String(length=64), nullable=False),
        sa.Column("phrase_text", sa.String(length=100), nullable=False),
        sa.Column("category", sa.String(length=16), nullable=False),
        sa.CheckConstraint(
            "category IN ('PAIN', 'REQUEST', 'REPLY', 'ETC')",
            name="category",
        ),
        sa.CheckConstraint(
            "phrase_text ~ '[^[:space:]]'",
            name="text_not_blank",
        ),
        sa.PrimaryKeyConstraint("phrase_id", name="pk_phrase"),
        sa.UniqueConstraint("phrase_code", name="uq_phrase_phrase_code"),
    )
    op.create_table(
        "utterance",
        sa.Column(
            "utt_id",
            sa.Integer(),
            sa.Identity(),
            nullable=False,
        ),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("phrase_id", sa.Integer(), nullable=True),
        sa.Column("raw_text", sa.String(length=200), nullable=False),
        sa.Column("corrected_text", sa.String(length=200), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        sa.CheckConstraint(
            "corrected_text IS NULL OR corrected_text ~ '[^[:space:]]'",
            name="corrected_text_not_blank",
        ),
        sa.CheckConstraint(
            "raw_text ~ '[^[:space:]]'",
            name="raw_text_not_blank",
        ),
        sa.ForeignKeyConstraint(
            ["phrase_id"],
            ["phrase.phrase_id"],
            name="fk_utterance_phrase_id_phrase",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["session.session_id"],
            name="fk_utterance_session_id_session",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("utt_id", name="pk_utterance"),
    )
    op.create_index(
        "ix_session_started_at",
        "session",
        ["started_at"],
        unique=False,
    )
    op.create_index(
        "ix_utterance_phrase_id",
        "utterance",
        ["phrase_id"],
        unique=False,
    )
    op.create_index(
        "ix_utterance_session_created_at",
        "utterance",
        ["session_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """의존 관계의 역순으로 테이블을 제거한다."""

    op.drop_index("ix_utterance_session_created_at", table_name="utterance")
    op.drop_index("ix_utterance_phrase_id", table_name="utterance")
    op.drop_table("utterance")
    op.drop_table("phrase")
    op.drop_index("ix_session_started_at", table_name="session")
    op.drop_table("session")
