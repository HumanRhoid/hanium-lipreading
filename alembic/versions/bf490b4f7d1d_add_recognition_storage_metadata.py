"""add recognition storage metadata

Revision ID: bf490b4f7d1d
Revises: abc56ec8eb17
Create Date: 2026-08-26

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "bf490b4f7d1d"
down_revision: str | Sequence[str] | None = "abc56ec8eb17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """비동기 영상 인식 및 개인화에 필요한 DB 구조를 추가한다."""

    # 새 비동기 영상 업로드에서는 추론 전에 utterance를 먼저 생성한다.
    op.add_column(
        "utterance",
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=True,
        ),
    )
    op.add_column(
        "utterance",
        sa.Column(
            "model_version",
            sa.String(length=100),
            nullable=True,
        ),
    )

    op.alter_column(
        "utterance",
        "session_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.alter_column(
        "utterance",
        "raw_text",
        existing_type=sa.String(length=200),
        nullable=True,
    )

    op.create_foreign_key(
        "fk_utterance_user_id_users",
        "utterance",
        "users",
        ["user_id"],
        ["user_id"],
        ondelete="CASCADE",
    )

    op.create_index(
        "ix_utterance_user_created_at",
        "utterance",
        ["user_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "video_asset",
        sa.Column(
            "video_id",
            sa.Integer(),
            sa.Identity(),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "utterance_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "idempotency_key",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "object_key",
            sa.String(length=512),
            nullable=False,
        ),
        sa.Column(
            "original_mime_type",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "normalized_mime_type",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "codec",
            sa.String(length=50),
            nullable=True,
        ),
        sa.Column(
            "width",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "height",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "fps",
            sa.Numeric(precision=7, scale=3),
            nullable=True,
        ),
        sa.Column(
            "duration_ms",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "size_bytes",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "checksum",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "storage_status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "storage_purpose",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "consent_version",
            sa.String(length=50),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "retention_until",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            ("storage_purpose IN ('TEMPORARY_INFERENCE', 'MODEL_TRAINING')"),
            name="storage_purpose",
        ),
        sa.CheckConstraint(
            (
                "storage_status IN "
                "('UPLOADED', 'NORMALIZING', 'READY', "
                "'DELETE_PENDING', 'DELETED', 'FAILED')"
            ),
            name="storage_status",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="fk_video_asset_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["utterance_id"],
            ["utterance.utt_id"],
            name="fk_video_asset_utterance_id_utterance",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "video_id",
            name="pk_video_asset",
        ),
        sa.UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_video_asset_user_id_idempotency_key",
        ),
    )

    op.create_index(
        "ix_video_asset_retention_until",
        "video_asset",
        ["retention_until"],
        unique=False,
    )

    op.create_index(
        "ix_video_asset_user_created_at",
        "video_asset",
        ["user_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "user_consent",
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "model_training_consent",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "consent_version",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="fk_user_consent_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "user_id",
            name="pk_user_consent",
        ),
    )

    op.create_table(
        "phrase_usage_stat",
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "phrase_code",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "usage_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "accepted_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "corrected_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "accepted_count >= 0",
            name="accepted_count_nonnegative",
        ),
        sa.CheckConstraint(
            "corrected_count >= 0",
            name="corrected_count_nonnegative",
        ),
        sa.CheckConstraint(
            "usage_count >= 0",
            name="usage_count_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="fk_phrase_usage_stat_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["phrase_code"],
            ["phrase.phrase_code"],
            name="fk_phrase_usage_stat_phrase_code_phrase",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "user_id",
            "phrase_code",
            name="pk_phrase_usage_stat",
        ),
    )


def downgrade() -> None:
    """비동기 영상 인식용 DB 확장을 제거한다."""

    op.drop_table("phrase_usage_stat")
    op.drop_table("user_consent")

    op.drop_index(
        "ix_video_asset_user_created_at",
        table_name="video_asset",
    )
    op.drop_index(
        "ix_video_asset_retention_until",
        table_name="video_asset",
    )
    op.drop_table("video_asset")

    op.drop_index(
        "ix_utterance_user_created_at",
        table_name="utterance",
    )
    op.drop_constraint(
        "fk_utterance_user_id_users",
        "utterance",
        type_="foreignkey",
    )

    op.alter_column(
        "utterance",
        "raw_text",
        existing_type=sa.String(length=200),
        nullable=False,
    )
    op.alter_column(
        "utterance",
        "session_id",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.drop_column(
        "utterance",
        "model_version",
    )
    op.drop_column(
        "utterance",
        "user_id",
    )
