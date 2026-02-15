"""Add answer to query history and schema_annotations to connections.

Revision ID: 034
Revises: 033
Create Date: 2026-02-15
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "034_add_query_answer_and_annotations"
down_revision = "033_add_moodboard_canvas_data"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "database_query_history",
        sa.Column("answer", sa.Text(), nullable=True),
    )
    op.add_column(
        "external_database_connections",
        sa.Column("schema_annotations", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("external_database_connections", "schema_annotations")
    op.drop_column("database_query_history", "answer")
