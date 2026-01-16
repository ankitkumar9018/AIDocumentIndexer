"""Add file watcher tables

Revision ID: 023_add_file_watcher_tables
Revises: 022
Create Date: 2026-01-12

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '023_add_file_watcher_tables'
down_revision = '022'
branch_labels = None
depends_on = None


def upgrade():
    # Create file_watcher_config table
    op.create_table(
        'file_watcher_config',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('enabled', sa.Boolean(), nullable=False, default=False),
        sa.Column('auto_start', sa.Boolean(), nullable=False, default=False),
        sa.Column('batch_size', sa.Integer(), nullable=False, default=10),
        sa.Column('poll_interval_seconds', sa.Integer(), nullable=False, default=5),
        sa.Column('max_retries', sa.Integer(), nullable=False, default=3),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create watched_directories table
    op.create_table(
        'watched_directories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('path', sa.String(1024), nullable=False),
        sa.Column('recursive', sa.Boolean(), nullable=False, default=True),
        sa.Column('auto_process', sa.Boolean(), nullable=False, default=True),
        sa.Column('access_tier_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('access_tiers.id', ondelete='SET NULL'), nullable=True),
        sa.Column('collection', sa.String(255), nullable=True),
        sa.Column('folder_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('folders.id', ondelete='SET NULL'), nullable=True),
        sa.Column('enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('files_processed', sa.Integer(), nullable=False, default=0),
        sa.Column('last_scan_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_event_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create file_watcher_events table
    op.create_table(
        'file_watcher_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('watch_dir_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('watched_directories.id', ondelete='CASCADE'), nullable=False),
        sa.Column('event_type', sa.String(20), nullable=False),
        sa.Column('file_path', sa.String(2048), nullable=False),
        sa.Column('file_name', sa.String(512), nullable=False),
        sa.Column('file_extension', sa.String(32), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False, default=0),
        sa.Column('file_hash', sa.String(64), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('processing_error', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, default=0),
        sa.Column('detected_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='SET NULL'), nullable=True),
    )

    # Create indexes
    op.create_index('ix_watched_directories_path', 'watched_directories', ['path'])
    op.create_index('ix_watched_directories_organization_id', 'watched_directories', ['organization_id'])
    op.create_index('ix_watched_directories_enabled', 'watched_directories', ['enabled'])

    op.create_index('ix_file_watcher_events_watch_dir_id', 'file_watcher_events', ['watch_dir_id'])
    op.create_index('ix_file_watcher_events_status', 'file_watcher_events', ['status'])
    op.create_index('ix_file_watcher_events_file_hash', 'file_watcher_events', ['file_hash'])
    op.create_index('ix_file_watcher_events_detected_at', 'file_watcher_events', ['detected_at'])


def downgrade():
    # Drop indexes
    op.drop_index('ix_file_watcher_events_detected_at', table_name='file_watcher_events')
    op.drop_index('ix_file_watcher_events_file_hash', table_name='file_watcher_events')
    op.drop_index('ix_file_watcher_events_status', table_name='file_watcher_events')
    op.drop_index('ix_file_watcher_events_watch_dir_id', table_name='file_watcher_events')

    op.drop_index('ix_watched_directories_enabled', table_name='watched_directories')
    op.drop_index('ix_watched_directories_organization_id', table_name='watched_directories')
    op.drop_index('ix_watched_directories_path', table_name='watched_directories')

    # Drop tables
    op.drop_table('file_watcher_events')
    op.drop_table('watched_directories')
    op.drop_table('file_watcher_config')
