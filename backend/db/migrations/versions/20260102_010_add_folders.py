"""Add folders table for hierarchical document organization

Revision ID: 20260102_010
Revises: 20260102_009
Create Date: 2026-01-02

This migration adds the folders table for hierarchical folder structure
and adds folder_id to the documents table for folder assignment.

Features:
- Materialized path pattern for efficient subtree queries
- Permission inheritance from parent folders
- Folder ownership tracking
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '20260102_010'
down_revision = '20260102_009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create folders table
    op.create_table(
        'folders',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('path', sa.String(2000), nullable=False),
        sa.Column('parent_folder_id', sa.String(36), nullable=True),
        sa.Column('depth', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('access_tier_id', sa.String(36), nullable=False),
        sa.Column('inherit_permissions', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_by_id', sa.String(36), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('color', sa.String(7), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        # Foreign keys
        sa.ForeignKeyConstraint(['parent_folder_id'], ['folders.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['access_tier_id'], ['access_tiers.id']),
        sa.ForeignKeyConstraint(['created_by_id'], ['users.id'], ondelete='SET NULL'),
    )

    # Create indexes for folders table
    op.create_index('idx_folders_path', 'folders', ['path'])
    op.create_index('idx_folders_parent', 'folders', ['parent_folder_id'])
    op.create_index('idx_folders_access_tier', 'folders', ['access_tier_id'])
    op.create_index('idx_folders_created_by', 'folders', ['created_by_id'])
    op.create_index('idx_folders_parent_name', 'folders', ['parent_folder_id', 'name'])

    # Add folder_id column to documents table
    op.add_column('documents', sa.Column('folder_id', sa.String(36), nullable=True))

    # Create foreign key constraint for documents.folder_id
    op.create_foreign_key(
        'fk_documents_folder_id',
        'documents',
        'folders',
        ['folder_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # Create index for folder_id on documents
    op.create_index('idx_documents_folder_id', 'documents', ['folder_id'])


def downgrade() -> None:
    # Drop index on documents.folder_id
    op.drop_index('idx_documents_folder_id', table_name='documents')

    # Drop foreign key constraint
    op.drop_constraint('fk_documents_folder_id', 'documents', type_='foreignkey')

    # Drop folder_id column from documents
    op.drop_column('documents', 'folder_id')

    # Drop indexes on folders table
    op.drop_index('idx_folders_parent_name', table_name='folders')
    op.drop_index('idx_folders_created_by', table_name='folders')
    op.drop_index('idx_folders_access_tier', table_name='folders')
    op.drop_index('idx_folders_parent', table_name='folders')
    op.drop_index('idx_folders_path', table_name='folders')

    # Drop folders table
    op.drop_table('folders')
