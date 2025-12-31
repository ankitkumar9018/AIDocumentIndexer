"""Add performance indexes for large-scale document operations

Revision ID: 20251230_006
Revises: 20251230_005
Create Date: 2025-12-30

These indexes optimize:
1. Document tag filtering for collection queries
2. Chunk retrieval by document ID
3. Access tier filtering in vector search
4. Document status filtering for processing queue
5. Search by filename and original filename
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20251230_006"
down_revision: Union[str, None] = "20251230_005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add performance indexes."""
    # Get the dialect name to handle SQLite vs PostgreSQL differences
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Document indexes
    if dialect == "postgresql":
        # GIN index for JSON array tags (PostgreSQL only)
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_tags_gin ON documents USING GIN (tags)"
        )
    else:
        # SQLite - create a regular index on tags (JSON stored as text)
        op.create_index(
            "idx_document_tags",
            "documents",
            ["tags"],
            if_not_exists=True,
        )

    # Document status index for processing queue queries
    op.create_index(
        "idx_document_processing_status",
        "documents",
        ["processing_status"],
        if_not_exists=True,
    )

    # Document collection index for filtered queries
    op.create_index(
        "idx_document_collection",
        "documents",
        ["collection"],
        if_not_exists=True,
    )

    # Document filename search indexes
    op.create_index(
        "idx_document_filename",
        "documents",
        ["filename"],
        if_not_exists=True,
    )

    op.create_index(
        "idx_document_original_filename",
        "documents",
        ["original_filename"],
        if_not_exists=True,
    )

    # Document user and created_at for listing
    op.create_index(
        "idx_document_user_created",
        "documents",
        ["user_id", "created_at"],
        if_not_exists=True,
    )

    # Chunk indexes
    op.create_index(
        "idx_chunk_document_id",
        "chunks",
        ["document_id"],
        if_not_exists=True,
    )

    # Chunk access tier for filtered vector search
    op.create_index(
        "idx_chunk_access_tier",
        "chunks",
        ["access_tier_id"],
        if_not_exists=True,
    )

    # Chunk page number for page-level queries
    op.create_index(
        "idx_chunk_page_number",
        "chunks",
        ["page_number"],
        if_not_exists=True,
    )

    # Composite index for chunk retrieval with ordering
    op.create_index(
        "idx_chunk_document_order",
        "chunks",
        ["document_id", "chunk_index"],
        if_not_exists=True,
    )

    # Chat session indexes
    op.create_index(
        "idx_chat_session_user",
        "chat_sessions",
        ["user_id"],
        if_not_exists=True,
    )

    op.create_index(
        "idx_chat_session_updated",
        "chat_sessions",
        ["updated_at"],
        if_not_exists=True,
    )

    # Chat message indexes
    op.create_index(
        "idx_chat_message_session",
        "chat_messages",
        ["session_id"],
        if_not_exists=True,
    )

    op.create_index(
        "idx_chat_message_created",
        "chat_messages",
        ["created_at"],
        if_not_exists=True,
    )

    # Audit log indexes for compliance queries
    op.create_index(
        "idx_audit_user_action",
        "audit_logs",
        ["user_id", "action"],
        if_not_exists=True,
    )

    op.create_index(
        "idx_audit_resource",
        "audit_logs",
        ["resource_type", "resource_id"],
        if_not_exists=True,
    )

    op.create_index(
        "idx_audit_created",
        "audit_logs",
        ["created_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    """Remove performance indexes."""
    # Document indexes
    op.drop_index("idx_document_tags_gin", table_name="documents", if_exists=True)
    op.drop_index("idx_document_tags", table_name="documents", if_exists=True)
    op.drop_index("idx_document_processing_status", table_name="documents", if_exists=True)
    op.drop_index("idx_document_collection", table_name="documents", if_exists=True)
    op.drop_index("idx_document_filename", table_name="documents", if_exists=True)
    op.drop_index("idx_document_original_filename", table_name="documents", if_exists=True)
    op.drop_index("idx_document_user_created", table_name="documents", if_exists=True)

    # Chunk indexes
    op.drop_index("idx_chunk_document_id", table_name="chunks", if_exists=True)
    op.drop_index("idx_chunk_access_tier", table_name="chunks", if_exists=True)
    op.drop_index("idx_chunk_page_number", table_name="chunks", if_exists=True)
    op.drop_index("idx_chunk_document_order", table_name="chunks", if_exists=True)

    # Chat session indexes
    op.drop_index("idx_chat_session_user", table_name="chat_sessions", if_exists=True)
    op.drop_index("idx_chat_session_updated", table_name="chat_sessions", if_exists=True)

    # Chat message indexes
    op.drop_index("idx_chat_message_session", table_name="chat_messages", if_exists=True)
    op.drop_index("idx_chat_message_created", table_name="chat_messages", if_exists=True)

    # Audit log indexes
    op.drop_index("idx_audit_user_action", table_name="audit_logs", if_exists=True)
    op.drop_index("idx_audit_resource", table_name="audit_logs", if_exists=True)
    op.drop_index("idx_audit_created", table_name="audit_logs", if_exists=True)
