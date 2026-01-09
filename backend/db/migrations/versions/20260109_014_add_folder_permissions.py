"""Add folder permissions and tags for per-user folder access control

Revision ID: 20260109_014
Revises: 20260108_013
Create Date: 2026-01-09

This migration adds:
- folders.tags: JSON column for folder tagging/categorization (array stored as JSON)
- users.use_folder_permissions_only: Boolean to enable restrictive folder access
- folder_permissions table: Per-user folder permission grants
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "20260109_014"
down_revision = "20260108_013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add folder permissions system."""

    # Get the database dialect to handle PostgreSQL vs SQLite differences
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Add tags column to folders table
    # Use JSON for cross-database compatibility (works with both PostgreSQL and SQLite)
    op.add_column(
        "folders",
        sa.Column(
            "tags",
            sa.JSON(),
            nullable=True,
            server_default="[]",
        ),
    )

    # Add use_folder_permissions_only to users table
    # When True, user ONLY sees explicitly granted folders (ignores tier)
    # When False (default), user sees tier-based folders + granted folders
    op.add_column(
        "users",
        sa.Column(
            "use_folder_permissions_only",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )

    # Create folder_permissions table for per-user folder access
    op.create_table(
        "folder_permissions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("folder_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column(
            "permission_level",
            sa.String(20),
            nullable=False,
            server_default="view",
        ),  # view, edit, manage
        sa.Column("granted_by_id", sa.String(36), nullable=True),
        sa.Column(
            "inherit_to_children",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        # Foreign keys
        sa.ForeignKeyConstraint(
            ["folder_id"], ["folders.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["granted_by_id"], ["users.id"], ondelete="SET NULL"
        ),
    )

    # Create indexes for folder_permissions
    op.create_index(
        "ix_folder_permissions_folder_id",
        "folder_permissions",
        ["folder_id"],
    )
    op.create_index(
        "ix_folder_permissions_user_id",
        "folder_permissions",
        ["user_id"],
    )

    # Unique constraint: one permission per user-folder pair
    op.create_unique_constraint(
        "uq_folder_permissions_folder_user",
        "folder_permissions",
        ["folder_id", "user_id"],
    )

    # Index on tags for filtering
    # Use GIN index for PostgreSQL (efficient JSON containment queries)
    # Skip for SQLite as it doesn't support GIN indexes
    if dialect == "postgresql":
        op.create_index(
            "ix_folders_tags",
            "folders",
            ["tags"],
            postgresql_using="gin",
        )


def downgrade() -> None:
    """Remove folder permissions system."""

    # Get dialect for conditional operations
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Drop indexes (only if they exist for the dialect)
    if dialect == "postgresql":
        op.drop_index("ix_folders_tags", table_name="folders")
    op.drop_constraint(
        "uq_folder_permissions_folder_user",
        "folder_permissions",
        type_="unique",
    )
    op.drop_index("ix_folder_permissions_user_id", table_name="folder_permissions")
    op.drop_index("ix_folder_permissions_folder_id", table_name="folder_permissions")

    # Drop folder_permissions table
    op.drop_table("folder_permissions")

    # Drop columns
    op.drop_column("users", "use_folder_permissions_only")
    op.drop_column("folders", "tags")
