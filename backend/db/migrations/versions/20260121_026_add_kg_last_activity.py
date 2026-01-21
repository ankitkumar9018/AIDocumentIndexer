"""Add last_activity_at to kg_extraction_jobs table

Revision ID: 026_add_kg_last_activity
Revises: 025_add_kg_extraction_provider
Create Date: 2026-01-21

This migration adds the last_activity_at field to kg_extraction_jobs table
to track heartbeat/progress and detect stuck jobs.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '026_add_kg_last_activity'
down_revision = '025_add_kg_extraction_provider'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add last_activity_at column to kg_extraction_jobs."""
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('kg_extraction_jobs', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    """Remove last_activity_at column from kg_extraction_jobs."""
    with op.batch_alter_table('kg_extraction_jobs', schema=None) as batch_op:
        batch_op.drop_column('last_activity_at')
