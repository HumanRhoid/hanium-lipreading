"""add training candidate outbox

Revision ID: 6d31c8e4a7f2
Revises: 0012fb49dc84
Create Date: 2026-09-03
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op


revision: str = "6d31c8e4a7f2"
down_revision: str | None = "0012fb49dc84"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_candidate",
        sa.Column(
            "sample_id",
            sa.String(length=36),
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
            "video_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "model_version",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "predicted_phrase_code",
            sa.String(length=64),
            nullable=True,
        ),
        sa.Column(
            "confidence",
            sa.Numeric(precision=4, scale=3),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'UNLABELED'"),
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
        sa.Column(
            "published_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR "
            "(confidence >= 0 AND confidence <= 1)",
            name=op.f(
                "ck_training_candidate_confidence_range"
            ),
        ),
        sa.CheckConstraint(
            "status IN ("
            "'UNLABELED', "
            "'PSEUDO_LABELED', "
            "'REVIEW_PENDING', "
            "'APPROVED', "
            "'REJECTED'"
            ")",
            name=op.f(
                "ck_training_candidate_status"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name=op.f(
                "fk_training_candidate_user_id_users"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["utterance_id"],
            ["utterance.utt_id"],
            name=op.f(
                "fk_training_candidate_utterance_id_utterance"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["video_asset.video_id"],
            name=op.f(
                "fk_training_candidate_video_id_video_asset"
            ),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "sample_id",
            name=op.f(
                "pk_training_candidate"
            ),
        ),
        sa.UniqueConstraint(
            "utterance_id",
            name="uq_training_candidate_utterance_id",
        ),
        sa.UniqueConstraint(
            "video_id",
            name="uq_training_candidate_video_id",
        ),
    )

    op.create_index(
        "ix_training_candidate_user_status",
        "training_candidate",
        ["user_id", "status"],
        unique=False,
    )

    op.create_index(
        "ix_training_candidate_published_at",
        "training_candidate",
        ["published_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_candidate_published_at",
        table_name="training_candidate",
    )

    op.drop_index(
        "ix_training_candidate_user_status",
        table_name="training_candidate",
    )

    op.drop_table(
        "training_candidate"
    )
