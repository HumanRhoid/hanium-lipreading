"""add user storage uuid

Revision ID: 507d7bb35761
Revises: bf490b4f7d1d
Create Date: 2026-08-27

"""

from collections.abc import Sequence
from uuid import uuid4

import sqlalchemy as sa

from alembic import op

revision: str = "507d7bb35761"
down_revision: str | Sequence[str] | None = "bf490b4f7d1d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Object Storage 내부 경로에 사용할 사용자 UUID를 추가한다."""

    op.add_column(
        "users",
        sa.Column(
            "storage_uuid",
            sa.Uuid(),
            nullable=True,
        ),
    )

    connection = op.get_bind()

    user_ids = connection.execute(
        sa.text(
            """
            SELECT user_id
            FROM users
            WHERE storage_uuid IS NULL
            """
        )
    ).scalars()

    for user_id in user_ids:
        connection.execute(
            sa.text(
                """
                UPDATE users
                SET storage_uuid = :storage_uuid
                WHERE user_id = :user_id
                """
            ),
            {
                "storage_uuid": str(uuid4()),
                "user_id": user_id,
            },
        )

    op.alter_column(
        "users",
        "storage_uuid",
        existing_type=sa.Uuid(),
        nullable=False,
    )

    op.create_unique_constraint(
        "uq_users_storage_uuid",
        "users",
        ["storage_uuid"],
    )


def downgrade() -> None:
    """사용자 Object Storage UUID를 제거한다."""

    op.drop_constraint(
        "uq_users_storage_uuid",
        "users",
        type_="unique",
    )

    op.drop_column(
        "users",
        "storage_uuid",
    )
