"""Add question_embedding column to text_to_sql_examples

Revision ID: 035_add_example_embedding
Revises: 034_add_query_answer_and_annotations
Create Date: 2026-02-15
"""

from alembic import op
import sqlalchemy as sa

revision = "035_add_example_embedding"
down_revision = "034_add_query_answer_and_annotations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "text_to_sql_examples",
        sa.Column("question_embedding", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("text_to_sql_examples", "question_embedding")
