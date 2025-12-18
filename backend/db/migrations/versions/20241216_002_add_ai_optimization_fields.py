"""Add AI optimization fields for semantic caching and hierarchical chunking

Revision ID: 002_ai_optimization
Revises: 001_initial
Create Date: 2024-12-16

Adds:
- ResponseCache: query_embedding (vector), query_text
- CacheSettings: enable_semantic_cache, semantic_similarity_threshold, max_semantic_cache_entries
- Chunk: is_summary, chunk_level, parent_chunk_id

All features are optional and disabled by default.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_ai_optimization'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add AI optimization columns."""

    # Check if we're using PostgreSQL (pgvector) or SQLite
    bind = op.get_bind()
    is_postgres = bind.dialect.name == 'postgresql'

    # ==========================================================================
    # ResponseCache table: Add semantic caching columns
    # ==========================================================================

    # Check if response_cache table exists (it might be created in a later migration)
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if 'response_cache' in tables:
        existing_columns = [c['name'] for c in inspector.get_columns('response_cache')]

        # Add query_embedding column for semantic similarity matching
        if 'query_embedding' not in existing_columns:
            if is_postgres:
                op.execute('ALTER TABLE response_cache ADD COLUMN query_embedding vector(1536)')
                # Create index for fast similarity search
                op.execute('''
                    CREATE INDEX IF NOT EXISTS idx_response_cache_embedding
                    ON response_cache
                    USING ivfflat (query_embedding vector_cosine_ops)
                    WITH (lists = 100)
                ''')
            else:
                # SQLite fallback: store as TEXT (JSON serialized)
                op.add_column('response_cache', sa.Column('query_embedding', sa.Text(), nullable=True))

        # Add query_text column for debugging/display
        if 'query_text' not in existing_columns:
            op.add_column('response_cache', sa.Column('query_text', sa.Text(), nullable=True))

    # ==========================================================================
    # CacheSettings table: Add semantic cache settings
    # ==========================================================================

    if 'cache_settings' in tables:
        existing_columns = [c['name'] for c in inspector.get_columns('cache_settings')]

        if 'enable_semantic_cache' not in existing_columns:
            op.add_column('cache_settings', sa.Column(
                'enable_semantic_cache', sa.Boolean(), nullable=False, server_default='false'
            ))

        if 'semantic_similarity_threshold' not in existing_columns:
            op.add_column('cache_settings', sa.Column(
                'semantic_similarity_threshold', sa.Float(), nullable=False, server_default='0.95'
            ))

        if 'max_semantic_cache_entries' not in existing_columns:
            op.add_column('cache_settings', sa.Column(
                'max_semantic_cache_entries', sa.Integer(), nullable=False, server_default='10000'
            ))

    # ==========================================================================
    # Chunk table: Add hierarchical chunking columns
    # ==========================================================================

    if 'chunks' in tables:
        existing_columns = [c['name'] for c in inspector.get_columns('chunks')]

        # is_summary: True for document/section summaries (for large doc optimization)
        if 'is_summary' not in existing_columns:
            op.add_column('chunks', sa.Column(
                'is_summary', sa.Boolean(), nullable=False, server_default='false'
            ))
            op.create_index('idx_chunks_is_summary', 'chunks', ['is_summary'])

        # chunk_level: Hierarchy level (0=detail, 1=section summary, 2=document summary)
        if 'chunk_level' not in existing_columns:
            op.add_column('chunks', sa.Column(
                'chunk_level', sa.Integer(), nullable=False, server_default='0'
            ))

        # parent_chunk_id: Reference to parent chunk in hierarchy
        if 'parent_chunk_id' not in existing_columns:
            if is_postgres:
                op.add_column('chunks', sa.Column(
                    'parent_chunk_id',
                    postgresql.UUID(as_uuid=True),
                    sa.ForeignKey('chunks.id', ondelete='SET NULL'),
                    nullable=True
                ))
            else:
                op.add_column('chunks', sa.Column(
                    'parent_chunk_id', sa.String(36), nullable=True
                ))

            # Index for fast parent-child lookups
            op.create_index('idx_chunks_parent', 'chunks', ['parent_chunk_id'])


def downgrade() -> None:
    """Remove AI optimization columns."""

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # Remove chunk columns
    if 'chunks' in tables:
        op.drop_index('idx_chunks_parent', table_name='chunks')
        op.drop_index('idx_chunks_is_summary', table_name='chunks')
        op.drop_column('chunks', 'parent_chunk_id')
        op.drop_column('chunks', 'chunk_level')
        op.drop_column('chunks', 'is_summary')

    # Remove cache_settings columns
    if 'cache_settings' in tables:
        op.drop_column('cache_settings', 'max_semantic_cache_entries')
        op.drop_column('cache_settings', 'semantic_similarity_threshold')
        op.drop_column('cache_settings', 'enable_semantic_cache')

    # Remove response_cache columns
    if 'response_cache' in tables:
        is_postgres = bind.dialect.name == 'postgresql'
        if is_postgres:
            op.execute('DROP INDEX IF EXISTS idx_response_cache_embedding')
        op.drop_column('response_cache', 'query_text')
        op.drop_column('response_cache', 'query_embedding')
