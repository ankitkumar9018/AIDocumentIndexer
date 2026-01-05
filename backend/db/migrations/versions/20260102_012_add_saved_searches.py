"""Add saved_searches column to user_preferences table.

Revision ID: 20260102_012
Revises: 20260102_011
Create Date: 2026-01-02
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "20260102_012"
down_revision = "20260102_011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add saved_searches JSON column to user_preferences."""
    op.add_column(
        "user_preferences",
        sa.Column("saved_searches", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Remove saved_searches column."""
    op.drop_column("user_preferences", "saved_searches")
