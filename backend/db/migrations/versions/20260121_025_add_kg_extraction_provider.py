"""Add provider_id to kg_extraction_jobs table

Revision ID: 025_add_kg_extraction_provider
Revises: 024_add_kg_extraction_status
Create Date: 2026-01-21

This migration adds the provider_id field to kg_extraction_jobs table
to allow users to specify which LLM provider to use for entity extraction.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '025_add_kg_extraction_provider'
down_revision = '024_add_kg_extraction_status'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add provider_id column to kg_extraction_jobs."""
    # Use batch mode for SQLite compatibility
    with op.batch_alter_table('kg_extraction_jobs', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('provider_id', sa.String(36), nullable=True)
        )


def downgrade() -> None:
    """Remove provider_id column from kg_extraction_jobs."""
    with op.batch_alter_table('kg_extraction_jobs', schema=None) as batch_op:
        batch_op.drop_column('provider_id')
