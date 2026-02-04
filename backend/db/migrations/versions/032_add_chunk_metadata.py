"""Add chunk_metadata column to chunks table.

Revision ID: 032
Revises: 031
Create Date: 2026-02-03
"""
from alembic import op
import sqlalchemy as sa


revision = "032_add_chunk_metadata"
down_revision = "031_add_has_embedding"
branch_labels = None
depends_on = None


def upgrade():
    """Add chunk_metadata column to chunks table."""
    # Check if column already exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col["name"] for col in inspector.get_columns("chunks")]

    if "chunk_metadata" not in columns:
        op.add_column(
            "chunks",
            sa.Column("chunk_metadata", sa.JSON(), nullable=True),
        )


def downgrade():
    """Remove chunk_metadata column from chunks table."""
    op.drop_column("chunks", "chunk_metadata")
