"""Add upload_jobs table for persistent upload tracking

Revision ID: 20260102_009
Revises: 20260102_008
Create Date: 2026-01-02

This migration adds the upload_jobs table to persist file upload status
in the database instead of in-memory storage. This ensures upload status
survives server restarts.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260102_009'
down_revision = '20260102_008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create upload_jobs table
    op.create_table(
        'upload_jobs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('file_hash', sa.String(64), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='queued'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_step', sa.String(100), nullable=False, server_default='Queued'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('collection', sa.String(200), nullable=True),
        sa.Column('access_tier', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('enable_ocr', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('enable_image_analysis', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('auto_generate_tags', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('chunk_count', sa.Integer(), nullable=True),
        sa.Column('word_count', sa.Integer(), nullable=True),
        sa.Column('document_id', sa.String(36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='SET NULL'),
    )

    # Create indexes
    op.create_index('idx_upload_jobs_status', 'upload_jobs', ['status'])
    op.create_index('idx_upload_jobs_file_hash', 'upload_jobs', ['file_hash'])
    op.create_index('idx_upload_jobs_status_created', 'upload_jobs', ['status', 'created_at'])
    op.create_index('idx_upload_jobs_document_id', 'upload_jobs', ['document_id'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_upload_jobs_document_id', table_name='upload_jobs')
    op.drop_index('idx_upload_jobs_status_created', table_name='upload_jobs')
    op.drop_index('idx_upload_jobs_file_hash', table_name='upload_jobs')
    op.drop_index('idx_upload_jobs_status', table_name='upload_jobs')

    # Drop table
    op.drop_table('upload_jobs')
