"""Add language fields to Entity and EntityMention tables for cross-language entity linking.

Revision ID: 20260108_013
Revises: 20260102_012
Create Date: 2026-01-08

This migration adds:
- Entity: entity_language, canonical_name, language_variants (for cross-language linking)
- EntityMention: mention_language, mention_script (for tracking mention context)
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "20260108_013"
down_revision = "20260102_012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add language fields to Entity and EntityMention tables."""

    # Add columns to entities table (without index=True to avoid duplicate indexes)
    op.add_column(
        "entities",
        sa.Column("entity_language", sa.String(10), nullable=True),
    )
    op.add_column(
        "entities",
        sa.Column("canonical_name", sa.String(500), nullable=True),
    )
    op.add_column(
        "entities",
        sa.Column("language_variants", sa.JSON(), nullable=True),
    )

    # Add columns to entity_mentions table
    op.add_column(
        "entity_mentions",
        sa.Column("mention_language", sa.String(10), nullable=True),
    )
    op.add_column(
        "entity_mentions",
        sa.Column("mention_script", sa.String(20), nullable=True),
    )

    # Create indexes for language fields (explicitly to control naming)
    op.create_index(
        "ix_entities_entity_language",
        "entities",
        ["entity_language"],
        unique=False,
    )
    op.create_index(
        "ix_entities_canonical_name",
        "entities",
        ["canonical_name"],
        unique=False,
    )


def downgrade() -> None:
    """Remove language fields from Entity and EntityMention tables."""

    # Drop indexes
    op.drop_index("ix_entities_canonical_name", table_name="entities")
    op.drop_index("ix_entities_entity_language", table_name="entities")

    # Drop columns from entity_mentions
    op.drop_column("entity_mentions", "mention_script")
    op.drop_column("entity_mentions", "mention_language")

    # Drop columns from entities
    op.drop_column("entities", "language_variants")
    op.drop_column("entities", "canonical_name")
    op.drop_column("entities", "entity_language")
