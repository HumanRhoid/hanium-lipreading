"""generalize user account fields

Revision ID: abc56ec8eb17
Revises: b8066cc1edc8
Create Date: 2026-08-26

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "abc56ec8eb17"
down_revision: Union[str, Sequence[str], None] = "b8066cc1edc8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """의료진 기준 users 구조를 일반 사용자 기준으로 변경한다."""

    op.alter_column(
        "users",
        "name",
        new_column_name="display_name",
        existing_type=sa.String(length=50),
        existing_nullable=False,
    )

    op.drop_column("users", "hospital")
    op.drop_column("users", "ward")


def downgrade() -> None:
    """일반 사용자 구조를 기존 의료진 사용자 구조로 되돌린다."""

    op.add_column(
        "users",
        sa.Column(
            "hospital",
            sa.String(length=100),
            nullable=False,
            server_default="",
        ),
    )

    op.add_column(
        "users",
        sa.Column(
            "ward",
            sa.String(length=100),
            nullable=True,
        ),
    )

    op.alter_column(
        "users",
        "display_name",
        new_column_name="name",
        existing_type=sa.String(length=50),
        existing_nullable=False,
    )

    op.alter_column(
        "users",
        "hospital",
        existing_type=sa.String(length=100),
        existing_nullable=False,
        server_default=None,
    )
