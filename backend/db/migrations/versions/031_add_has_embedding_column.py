"""
Add has_embedding boolean column to chunks table.

This replaces storing actual embeddings in SQLite with a simple boolean flag
to track whether the embedding is stored in the vector database (ChromaDB).

Benefits:
- Saves significant storage (1 boolean vs 768+ floats per chunk)
- Still allows UI to show embedding progress
- Keeps actual vector data only in ChromaDB

Revision ID: 031
Revises: 030
Create Date: 2026-02-01
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '031_add_has_embedding'
down_revision = '030_add_reports_research_moodboard'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add has_embedding column to chunks table."""
    # Add has_embedding column with default False
    op.add_column(
        'chunks',
        sa.Column(
            'has_embedding',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('0'),
            comment='True if embedding is stored in vector database'
        )
    )

    # Create index for faster queries
    op.create_index(
        'ix_chunks_has_embedding',
        'chunks',
        ['has_embedding'],
        unique=False
    )

    # Backfill: Set has_embedding=True for chunks that have embeddings
    # This handles existing data where embedding column was populated
    op.execute("""
        UPDATE chunks
        SET has_embedding = 1
        WHERE embedding IS NOT NULL
    """)


def downgrade() -> None:
    """Remove has_embedding column."""
    op.drop_index('ix_chunks_has_embedding', table_name='chunks')
    op.drop_column('chunks', 'has_embedding')
