"""Add agent publishing fields

Revision ID: 027_add_agent_publishing
Revises: 026_add_kg_last_activity
Create Date: 2026-01-22

Adds publishing fields to agent_definitions table for embeddable agents:
- is_published: Whether agent is publicly available
- embed_token: Unique token for embedding
- publish_config: JSON config (allowed_domains, rate_limit, branding)
- agent_mode: voice, chat, or hybrid
- tts_config: Text-to-speech configuration

Also adds VOICE_AGENT and CHAT_AGENT to workflow node types.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '027_add_agent_publishing'
down_revision = '026_add_kg_last_activity'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add publishing fields to agent_definitions using batch mode for SQLite
    with op.batch_alter_table('agent_definitions', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('is_published', sa.Boolean(), nullable=False, server_default='0')
        )
        batch_op.add_column(
            sa.Column('embed_token', sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column('publish_config', sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column('agent_mode', sa.String(20), nullable=True)
        )
        batch_op.add_column(
            sa.Column('tts_config', sa.JSON(), nullable=True)
        )

    # Create indexes for publishing
    with op.batch_alter_table('agent_definitions', schema=None) as batch_op:
        batch_op.create_index(
            'idx_agent_definitions_published',
            ['is_published']
        )
        batch_op.create_index(
            'idx_agent_definitions_embed_token',
            ['embed_token'],
            unique=True
        )


def downgrade() -> None:
    # Drop indexes and columns using batch mode for SQLite
    with op.batch_alter_table('agent_definitions', schema=None) as batch_op:
        batch_op.drop_index('idx_agent_definitions_embed_token')
        batch_op.drop_index('idx_agent_definitions_published')
        batch_op.drop_column('tts_config')
        batch_op.drop_column('agent_mode')
        batch_op.drop_column('publish_config')
        batch_op.drop_column('embed_token')
        batch_op.drop_column('is_published')
