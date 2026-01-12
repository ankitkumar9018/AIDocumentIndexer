"""Add private document support and fix user-document cascade

Revision ID: 022_private_docs
Revises: 021_org_isolation
Create Date: 2026-01-10

Adds:
- is_private column to documents table
- Updates user-document relationship for cascade delete of private documents
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '022'
down_revision = '021'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_private column to documents table
    op.add_column(
        'documents',
        sa.Column('is_private', sa.Boolean(), nullable=False, server_default='false')
    )

    # Create index for efficient filtering
    op.create_index('ix_documents_is_private', 'documents', ['is_private'])

    # Create composite index for private document queries (owner + private flag)
    op.create_index(
        'ix_documents_uploaded_by_private',
        'documents',
        ['uploaded_by_id', 'is_private']
    )


def downgrade() -> None:
    # Remove indexes first
    op.drop_index('ix_documents_uploaded_by_private', table_name='documents')
    op.drop_index('ix_documents_is_private', table_name='documents')

    # Remove the column
    op.drop_column('documents', 'is_private')
