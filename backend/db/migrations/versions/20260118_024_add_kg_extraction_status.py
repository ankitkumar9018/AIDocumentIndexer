"""Add knowledge graph extraction status and job tracking

Revision ID: 024_add_kg_extraction_status
Revises: 023_add_file_watcher_tables
Create Date: 2026-01-18

Adds:
- kg_extraction_status, kg_extracted_at, kg_entity_count, kg_relation_count to documents table
- kg_extraction_jobs table for background job tracking
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers
revision = '024_add_kg_extraction_status'
down_revision = '023_add_file_watcher_tables'
branch_labels = None
depends_on = None


def get_uuid_type():
    """Return appropriate UUID type based on database dialect."""
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        from sqlalchemy.dialects.postgresql import UUID
        return UUID(as_uuid=True)
    else:
        return sa.CHAR(36)


def get_json_type():
    """Return appropriate JSON type based on database dialect."""
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        from sqlalchemy.dialects.postgresql import JSONB
        return JSONB()
    else:
        return sa.Text()


def upgrade():
    uuid_type = get_uuid_type()
    json_type = get_json_type()

    # Add KG extraction status columns to documents table
    op.add_column(
        'documents',
        sa.Column('kg_extraction_status', sa.String(20), nullable=True, default='pending')
    )
    op.add_column(
        'documents',
        sa.Column('kg_extracted_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        'documents',
        sa.Column('kg_entity_count', sa.Integer(), nullable=False, server_default='0')
    )
    op.add_column(
        'documents',
        sa.Column('kg_relation_count', sa.Integer(), nullable=False, server_default='0')
    )

    # Create index on kg_extraction_status for efficient filtering
    op.create_index('ix_documents_kg_extraction_status', 'documents', ['kg_extraction_status'])

    # Create kg_extraction_jobs table
    op.create_table(
        'kg_extraction_jobs',
        sa.Column('id', uuid_type, primary_key=True),
        sa.Column('organization_id', uuid_type,
                  sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('user_id', uuid_type,
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='queued'),
        sa.Column('total_documents', sa.Integer(), nullable=False, default=0),
        sa.Column('processed_documents', sa.Integer(), nullable=False, default=0),
        sa.Column('failed_documents', sa.Integer(), nullable=False, default=0),
        sa.Column('total_entities', sa.Integer(), nullable=False, default=0),
        sa.Column('total_relations', sa.Integer(), nullable=False, default=0),
        sa.Column('current_document_id', uuid_type,
                  sa.ForeignKey('documents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('current_document_name', sa.String(500), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('avg_doc_processing_time', sa.Float(), nullable=True),
        sa.Column('error_log', json_type, nullable=True),
        sa.Column('only_new_documents', sa.Boolean(), nullable=False, default=True),
        sa.Column('document_ids', sa.Text(), nullable=True),  # Store as JSON string for SQLite compatibility
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(),
                  onupdate=sa.func.now(), nullable=False),
    )

    # Create indexes for kg_extraction_jobs
    op.create_index('ix_kg_extraction_jobs_organization_id', 'kg_extraction_jobs', ['organization_id'])
    op.create_index('ix_kg_extraction_jobs_user_id', 'kg_extraction_jobs', ['user_id'])
    op.create_index('ix_kg_extraction_jobs_status', 'kg_extraction_jobs', ['status'])


def downgrade():
    # Drop kg_extraction_jobs indexes and table
    op.drop_index('ix_kg_extraction_jobs_status', table_name='kg_extraction_jobs')
    op.drop_index('ix_kg_extraction_jobs_user_id', table_name='kg_extraction_jobs')
    op.drop_index('ix_kg_extraction_jobs_organization_id', table_name='kg_extraction_jobs')
    op.drop_table('kg_extraction_jobs')

    # Remove documents columns
    op.drop_index('ix_documents_kg_extraction_status', table_name='documents')
    op.drop_column('documents', 'kg_relation_count')
    op.drop_column('documents', 'kg_entity_count')
    op.drop_column('documents', 'kg_extracted_at')
    op.drop_column('documents', 'kg_extraction_status')
