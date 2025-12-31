"""Add chat_feedback table for storing user feedback on chat responses

Revision ID: 20251230_007
Revises: 20251230_006
Create Date: 2025-12-30

Stores user feedback (like/dislike ratings and comments) on chat responses.
Links to agent trajectories for agent mode responses, enabling the prompt
optimization system to learn from user preferences.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20251230_007"
down_revision: Union[str, None] = "20251230_006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create chat_feedback table."""
    # Get the dialect name to handle SQLite vs PostgreSQL differences
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Create the chat_feedback table
    if dialect == "postgresql":
        # PostgreSQL uses UUID type
        op.create_table(
            "chat_feedback",
            sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("message_id", sa.String(36), nullable=False, index=True),
            sa.Column("session_id", sa.dialects.postgresql.UUID(as_uuid=True),
                      sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True),
            sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True),
                      sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("rating", sa.Integer, nullable=False),
            sa.Column("comment", sa.Text, nullable=True),
            sa.Column("mode", sa.String(20), nullable=True),
            sa.Column("trajectory_id", sa.dialects.postgresql.UUID(as_uuid=True),
                      sa.ForeignKey("agent_trajectories.id", ondelete="SET NULL"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                      onupdate=sa.func.now(), nullable=False),
        )
    else:
        # SQLite uses CHAR(36) for UUIDs
        op.create_table(
            "chat_feedback",
            sa.Column("id", sa.CHAR(36), primary_key=True),
            sa.Column("message_id", sa.String(36), nullable=False, index=True),
            sa.Column("session_id", sa.CHAR(36),
                      sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True),
            sa.Column("user_id", sa.CHAR(36),
                      sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("rating", sa.Integer, nullable=False),
            sa.Column("comment", sa.Text, nullable=True),
            sa.Column("mode", sa.String(20), nullable=True),
            sa.Column("trajectory_id", sa.CHAR(36),
                      sa.ForeignKey("agent_trajectories.id", ondelete="SET NULL"), nullable=True),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        )

    # Create indexes
    op.create_index("idx_chat_feedback_message", "chat_feedback", ["message_id"])
    op.create_index("idx_chat_feedback_user", "chat_feedback", ["user_id"])
    op.create_index("idx_chat_feedback_rating", "chat_feedback", ["rating"])
    op.create_index("idx_chat_feedback_created", "chat_feedback", ["created_at"])


def downgrade() -> None:
    """Drop chat_feedback table."""
    op.drop_index("idx_chat_feedback_created", table_name="chat_feedback")
    op.drop_index("idx_chat_feedback_rating", table_name="chat_feedback")
    op.drop_index("idx_chat_feedback_user", table_name="chat_feedback")
    op.drop_index("idx_chat_feedback_message", table_name="chat_feedback")
    op.drop_table("chat_feedback")
