"""Add user_preferences table for UI settings persistence

Revision ID: 20260102_011
Revises: 20260102_010
Create Date: 2026-01-02

This migration adds the user_preferences table to persist user UI settings
such as theme, view mode, default filters, and recent items.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260102_011'
down_revision = '20260102_010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create user_preferences table
    op.create_table(
        'user_preferences',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False),

        # UI Theme
        sa.Column('theme', sa.String(20), nullable=False, server_default='system'),

        # Document List View
        sa.Column('documents_view_mode', sa.String(20), nullable=False, server_default='grid'),
        sa.Column('documents_sort_by', sa.String(30), nullable=False, server_default='created_at'),
        sa.Column('documents_sort_order', sa.String(4), nullable=False, server_default='desc'),
        sa.Column('documents_page_size', sa.Integer(), nullable=False, server_default='20'),

        # Default Filters
        sa.Column('default_collection', sa.String(200), nullable=True),
        sa.Column('default_folder_id', sa.String(36), sa.ForeignKey('folders.id', ondelete='SET NULL'), nullable=True),

        # Search Preferences
        sa.Column('search_include_content', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('search_results_per_page', sa.Integer(), nullable=False, server_default='10'),

        # Chat/RAG Preferences
        sa.Column('chat_show_sources', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('chat_expand_sources', sa.Boolean(), nullable=False, server_default='false'),

        # Sidebar State
        sa.Column('sidebar_collapsed', sa.Boolean(), nullable=False, server_default='false'),

        # Recent Items (stored as JSON array)
        sa.Column('recent_documents', sa.Text(), nullable=True),
        sa.Column('recent_searches', sa.Text(), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )

    # Create index on user_id
    op.create_index('idx_user_preferences_user', 'user_preferences', ['user_id'])


def downgrade() -> None:
    # Drop index
    op.drop_index('idx_user_preferences_user', table_name='user_preferences')

    # Drop table
    op.drop_table('user_preferences')
