"""Add OCR metrics table

Revision ID: 20251230_004
Revises: 003_ocr_settings
Create Date: 2025-12-30 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20251230_004'
down_revision: Union[str, None] = '003_ocr_settings'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add OCR metrics table for performance tracking."""

    # Check database dialect
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    # Create OCR metrics table
    op.create_table(
        'ocr_metrics',
        # UUIDMixin columns
        sa.Column('id',
                 postgresql.UUID(as_uuid=True) if dialect_name == 'postgresql' else sa.CHAR(36),
                 primary_key=True),

        # TimestampMixin columns
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),

        # Provider info
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('variant', sa.String(50), nullable=True),
        sa.Column('language', sa.String(10), nullable=False),

        # Document context
        sa.Column('document_id',
                 postgresql.UUID(as_uuid=True) if dialect_name == 'postgresql' else sa.CHAR(36),
                 sa.ForeignKey('documents.id', ondelete='SET NULL'),
                 nullable=True),
        sa.Column('user_id',
                 postgresql.UUID(as_uuid=True) if dialect_name == 'postgresql' else sa.CHAR(36),
                 sa.ForeignKey('users.id', ondelete='SET NULL'),
                 nullable=True),

        # Performance metrics
        sa.Column('processing_time_ms', sa.Integer(), nullable=False),
        sa.Column('page_count', sa.Integer(), server_default='1', nullable=False),
        sa.Column('character_count', sa.Integer(), nullable=True),

        # Quality indicators
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('success', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('fallback_used', sa.Boolean(), server_default='false', nullable=False),

        # Cost estimation
        sa.Column('cost_usd', sa.Float(), nullable=True),

        # Additional data
        sa.Column('extra_data',
                 postgresql.JSONB() if dialect_name == 'postgresql' else sa.Text(),
                 nullable=True),
    )

    # Create indexes for efficient querying
    op.create_index('idx_ocr_metrics_provider', 'ocr_metrics', ['provider'])
    op.create_index('idx_ocr_metrics_document', 'ocr_metrics', ['document_id'])
    op.create_index('idx_ocr_metrics_user', 'ocr_metrics', ['user_id'])
    op.create_index('idx_ocr_metrics_created', 'ocr_metrics', ['created_at'])
    op.create_index('idx_ocr_metrics_provider_date', 'ocr_metrics', ['provider', 'created_at'])
    op.create_index('idx_ocr_metrics_success', 'ocr_metrics', ['success', 'created_at'])


def downgrade() -> None:
    """Remove OCR metrics table."""
    op.drop_index('idx_ocr_metrics_success', table_name='ocr_metrics')
    op.drop_index('idx_ocr_metrics_provider_date', table_name='ocr_metrics')
    op.drop_index('idx_ocr_metrics_created', table_name='ocr_metrics')
    op.drop_index('idx_ocr_metrics_user', table_name='ocr_metrics')
    op.drop_index('idx_ocr_metrics_document', table_name='ocr_metrics')
    op.drop_index('idx_ocr_metrics_provider', table_name='ocr_metrics')
    op.drop_table('ocr_metrics')
