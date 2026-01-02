"""Add HNSW vector index and FTS GIN index for million-document scale

Revision ID: 20260102_008
Revises: 20251230_007
Create Date: 2026-01-02

CRITICAL PERFORMANCE INDEXES:
1. HNSW index for pgvector - 100x speedup (O(n) → O(log n)) for vector search
2. GIN index for full-text search - 10-20x speedup for keyword/BM25 queries
3. Composite indexes for access-tier filtered searches

These indexes enable the system to scale from thousands to millions of documents.
HNSW (Hierarchical Navigable Small World) provides sub-linear search complexity.

For 10M+ vectors, consider switching to IVFFlat for faster index build time
(slightly lower recall but much faster to build).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260102_008"
down_revision: Union[str, None] = "20251230_007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add critical performance indexes for million-document scale."""
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        # =============================================================================
        # CRITICAL: HNSW Index for Vector Similarity Search
        # =============================================================================
        # This is the single most important index for RAG performance.
        # Without it, every vector search is O(n) - full table scan.
        # With HNSW, search becomes O(log n).
        #
        # Parameters:
        # - m=16: Number of bi-directional links per node (default 16, higher = better recall, more memory)
        # - ef_construction=200: Size of dynamic candidate list during index construction (higher = better recall, slower build)
        #
        # For 10M+ vectors, consider IVFFlat instead (faster build, slightly lower recall):
        # CREATE INDEX idx_chunks_embedding_ivf ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=200)
        """)

        # Also create HNSW index for scraped content embeddings
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scraped_content_embedding_hnsw
            ON scraped_content USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=200)
        """)

        # HNSW index for entity embeddings (GraphRAG)
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entities_embedding_hnsw
            ON entities USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=200)
        """)

        # HNSW index for response cache query embeddings (semantic caching)
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_response_cache_embedding_hnsw
            ON response_cache USING hnsw (query_embedding vector_cosine_ops)
            WITH (m=16, ef_construction=100)
        """)

        # =============================================================================
        # CRITICAL: GIN Index for Full-Text Search (BM25/Keyword)
        # =============================================================================
        # Without this, every keyword search does a full table scan.
        # GIN index enables efficient full-text search with to_tsvector.
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_content_fts
            ON chunks USING GIN (to_tsvector('english', content))
        """)

        # FTS index for scraped content
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scraped_content_fts
            ON scraped_content USING GIN (to_tsvector('english', content))
        """)

        # =============================================================================
        # Composite Indexes for Filtered Searches
        # =============================================================================
        # These optimize the common case of searching with access tier filters

        # Composite index for access-tier filtered searches (most common query pattern)
        # Note: We can't INCLUDE embedding in a btree index, so we use a partial covering approach
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_access_created
            ON chunks (access_tier_id, created_at DESC)
        """)

        # Composite index for document-scoped searches with ordering
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_doc_chunk_index
            ON chunks (document_id, chunk_index)
        """)

        # Index for hierarchical chunk queries (summaries first, then details)
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_level_summary
            ON chunks (chunk_level, is_summary)
        """)

        # Index for document metadata searches (enhanced RAG)
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_enhanced_metadata
            ON documents USING GIN (enhanced_metadata)
            WHERE enhanced_metadata IS NOT NULL
        """)

        print("PostgreSQL: Created HNSW, GIN FTS, and composite indexes")

    else:
        # SQLite: Create basic indexes (no HNSW or GIN support)
        # These won't provide the same performance benefits but maintain compatibility

        # Basic index on chunk content (limited FTS capability)
        op.create_index(
            "idx_chunks_content_basic",
            "chunks",
            ["content"],
            if_not_exists=True,
        )

        # Composite indexes work in SQLite
        op.create_index(
            "idx_chunks_access_created",
            "chunks",
            ["access_tier_id", "created_at"],
            if_not_exists=True,
        )

        op.create_index(
            "idx_chunks_doc_chunk_index",
            "chunks",
            ["document_id", "chunk_index"],
            if_not_exists=True,
        )

        op.create_index(
            "idx_chunks_level_summary",
            "chunks",
            ["chunk_level", "is_summary"],
            if_not_exists=True,
        )

        print("SQLite: Created basic indexes (HNSW and GIN not supported)")


def downgrade() -> None:
    """Remove the performance indexes."""
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        # Drop HNSW indexes
        op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_hnsw")
        op.execute("DROP INDEX IF EXISTS idx_scraped_content_embedding_hnsw")
        op.execute("DROP INDEX IF EXISTS idx_entities_embedding_hnsw")
        op.execute("DROP INDEX IF EXISTS idx_response_cache_embedding_hnsw")

        # Drop FTS indexes
        op.execute("DROP INDEX IF EXISTS idx_chunks_content_fts")
        op.execute("DROP INDEX IF EXISTS idx_scraped_content_fts")

        # Drop composite indexes
        op.execute("DROP INDEX IF EXISTS idx_chunks_access_created")
        op.execute("DROP INDEX IF EXISTS idx_chunks_doc_chunk_index")
        op.execute("DROP INDEX IF EXISTS idx_chunks_level_summary")
        op.execute("DROP INDEX IF EXISTS idx_documents_enhanced_metadata")

    else:
        # SQLite
        op.drop_index("idx_chunks_content_basic", table_name="chunks", if_exists=True)
        op.drop_index("idx_chunks_access_created", table_name="chunks", if_exists=True)
        op.drop_index("idx_chunks_doc_chunk_index", table_name="chunks", if_exists=True)
        op.drop_index("idx_chunks_level_summary", table_name="chunks", if_exists=True)
