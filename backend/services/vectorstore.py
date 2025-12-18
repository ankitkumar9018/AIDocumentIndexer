"""
AIDocumentIndexer - Vector Store Service
=========================================

Provides vector storage and similarity search using PostgreSQL + pgvector.
Supports hybrid search (vector + keyword) with access tier filtering.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import structlog
from sqlalchemy import select, text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context, get_async_session_factory
from backend.db.models import Chunk, Document, AccessTier, HAS_PGVECTOR

logger = structlog.get_logger(__name__)

# Cross-encoder reranking support
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    CrossEncoder = None


# =============================================================================
# Types
# =============================================================================

class SearchType(str, Enum):
    """Search type for retrieval."""
    VECTOR = "vector"           # Pure vector similarity
    KEYWORD = "keyword"         # Full-text search
    HYBRID = "hybrid"           # Combined vector + keyword


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Document info
    document_title: Optional[str] = None
    document_filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    # Search settings
    default_top_k: int = 10
    similarity_threshold: float = 0.4  # Lower threshold for OCR'd documents
    search_type: SearchType = SearchType.HYBRID

    # Hybrid search weights
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    # Re-ranking with cross-encoder
    enable_reranking: bool = True  # Enabled by default for better accuracy
    rerank_top_k: int = 20  # Fetch more candidates for reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, accurate reranker


# =============================================================================
# Vector Store Service
# =============================================================================

class VectorStore:
    """
    Vector storage and retrieval service using PostgreSQL + pgvector.

    Features:
    - Vector similarity search (cosine, L2, inner product)
    - Hybrid search with full-text keywords
    - Access tier filtering (RLS-compatible)
    - Batch operations for efficiency
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize vector store.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or VectorStoreConfig()
        self._has_pgvector = HAS_PGVECTOR
        self._reranker = None

        if not self._has_pgvector:
            logger.warning("pgvector not available, vector search will be limited")

        # Initialize reranker if enabled and available
        if self.config.enable_reranking and HAS_CROSS_ENCODER:
            try:
                self._reranker = CrossEncoder(self.config.rerank_model)
                logger.info("Initialized cross-encoder reranker", model=self.config.rerank_model)
            except Exception as e:
                logger.warning("Failed to initialize reranker, disabling", error=str(e))
                self._reranker = None
        elif self.config.enable_reranking:
            logger.warning("Reranking enabled but sentence-transformers not installed")

    # =========================================================================
    # Storage Operations
    # =========================================================================

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        access_tier_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[str]:
        """
        Add chunks with embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'content', 'embedding', and metadata
            document_id: ID of the parent document
            access_tier_id: Access tier for permission filtering
            session: Optional existing database session

        Returns:
            List of created chunk IDs
        """
        chunk_ids = []

        async def _add_chunks(db: AsyncSession):
            for i, chunk_data in enumerate(chunks):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=uuid.UUID(document_id),
                    access_tier_id=uuid.UUID(access_tier_id),
                    content=chunk_data["content"],
                    content_hash=chunk_data.get("content_hash", ""),
                    embedding=chunk_data.get("embedding"),
                    chunk_index=chunk_data.get("chunk_index", i),
                    page_number=chunk_data.get("page_number"),
                    section_title=chunk_data.get("section_title"),
                    token_count=chunk_data.get("token_count"),
                    char_count=chunk_data.get("char_count", len(chunk_data["content"])),
                )
                db.add(chunk)
                chunk_ids.append(str(chunk.id))

            await db.flush()

        if session:
            await _add_chunks(session)
        else:
            async with async_session_context() as db:
                await _add_chunks(db)

        logger.info(
            "Added chunks to vector store",
            document_id=document_id,
            chunk_count=len(chunk_ids),
        )

        return chunk_ids

    async def delete_document_chunks(
        self,
        document_id: str,
        session: Optional[AsyncSession] = None,
    ) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID
            session: Optional existing database session

        Returns:
            Number of chunks deleted
        """
        async def _delete(db: AsyncSession) -> int:
            result = await db.execute(
                select(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
            )
            chunks = result.scalars().all()
            count = len(chunks)

            for chunk in chunks:
                await db.delete(chunk)

            return count

        if session:
            count = await _delete(session)
        else:
            async with async_session_context() as db:
                count = await _delete(db)

        logger.info("Deleted document chunks", document_id=document_id, count=count)
        return count

    async def update_chunk_embedding(
        self,
        chunk_id: str,
        embedding: List[float],
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Update embedding for a specific chunk.

        Args:
            chunk_id: Chunk ID
            embedding: New embedding vector
            session: Optional existing database session

        Returns:
            True if updated successfully
        """
        async def _update(db: AsyncSession) -> bool:
            result = await db.execute(
                select(Chunk).where(Chunk.id == uuid.UUID(chunk_id))
            )
            chunk = result.scalar_one_or_none()

            if chunk:
                chunk.embedding = embedding
                return True
            return False

        if session:
            return await _update(session)
        else:
            async with async_session_context() as db:
                return await _update(db)

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            similarity_threshold: Minimum similarity score
            session: Optional existing database session

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k
        threshold = similarity_threshold or self.config.similarity_threshold

        if not self._has_pgvector:
            logger.warning("pgvector not available, returning empty results")
            return []

        async def _search(db: AsyncSession) -> List[SearchResult]:
            # Build the query using pgvector's cosine similarity
            # 1 - (embedding <=> query) gives similarity (higher is better)
            similarity_expr = 1 - Chunk.embedding.cosine_distance(query_embedding)

            # Base query with similarity
            query = (
                select(
                    Chunk,
                    Document,
                    similarity_expr.label("similarity"),
                )
                .join(Document, Chunk.document_id == Document.id)
                .join(AccessTier, Chunk.access_tier_id == AccessTier.id)
                .where(AccessTier.level <= access_tier_level)
                .where(Chunk.embedding.isnot(None))
            )

            # Filter by document IDs if provided
            if document_ids:
                doc_uuids = [uuid.UUID(d) for d in document_ids]
                query = query.where(Document.id.in_(doc_uuids))

            # Apply similarity threshold and ordering
            query = (
                query
                .where(similarity_expr >= threshold)
                .order_by(similarity_expr.desc())
                .limit(top_k)
            )

            result = await db.execute(query)
            rows = result.all()

            results = []
            for chunk, doc, similarity in rows:
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=float(similarity),
                    metadata={
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                    },
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                ))

            return results

        if session:
            return await _search(session)
        else:
            async with async_session_context() as db:
                return await _search(db)

    async def keyword_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Perform full-text keyword search.

        Args:
            query: Search query string
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            session: Optional existing database session

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k

        async def _search(db: AsyncSession) -> List[SearchResult]:
            # PostgreSQL full-text search
            # Using to_tsvector and to_tsquery for ranking
            ts_query = func.plainto_tsquery('english', query)
            ts_rank = func.ts_rank(
                func.to_tsvector('english', Chunk.content),
                ts_query
            )

            base_query = (
                select(
                    Chunk,
                    Document,
                    ts_rank.label("rank"),
                )
                .join(Document, Chunk.document_id == Document.id)
                .join(AccessTier, Chunk.access_tier_id == AccessTier.id)
                .where(AccessTier.level <= access_tier_level)
                .where(
                    func.to_tsvector('english', Chunk.content).op('@@')(ts_query)
                )
            )

            # Filter by document IDs if provided
            if document_ids:
                doc_uuids = [uuid.UUID(d) for d in document_ids]
                base_query = base_query.where(Document.id.in_(doc_uuids))

            # Order by rank
            base_query = (
                base_query
                .order_by(ts_rank.desc())
                .limit(top_k)
            )

            result = await db.execute(base_query)
            rows = result.all()

            results = []
            for chunk, doc, rank in rows:
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=float(rank),
                    metadata={
                        "chunk_index": chunk.chunk_index,
                        "search_type": "keyword",
                    },
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                ))

            return results

        if session:
            return await _search(session)
        else:
            async with async_session_context() as db:
                return await _search(db)

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity and keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to combine results.

        Args:
            query: Search query string
            query_embedding: Query embedding vector
            top_k: Number of final results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            vector_weight: Weight for vector results (0-1)
            keyword_weight: Weight for keyword results (0-1)
            session: Optional existing database session

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k
        vec_weight = vector_weight or self.config.vector_weight
        kw_weight = keyword_weight or self.config.keyword_weight

        # Get more results from each method for better fusion
        fetch_k = self.config.rerank_top_k if self.config.enable_reranking else top_k * 2

        # Run both searches
        vector_results = await self.similarity_search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
            session=session,
        )

        keyword_results = await self.keyword_search(
            query=query,
            top_k=fetch_k,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
            session=session,
        )

        # Reciprocal Rank Fusion
        # RRF(d) = Σ 1/(k + rank(d)) where k is a constant (typically 60)
        k = 60
        scores: Dict[str, Tuple[float, SearchResult]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            rrf_score = vec_weight * (1.0 / (k + rank + 1))
            if result.chunk_id in scores:
                scores[result.chunk_id] = (
                    scores[result.chunk_id][0] + rrf_score,
                    result,
                )
            else:
                scores[result.chunk_id] = (rrf_score, result)

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            rrf_score = kw_weight * (1.0 / (k + rank + 1))
            if result.chunk_id in scores:
                scores[result.chunk_id] = (
                    scores[result.chunk_id][0] + rrf_score,
                    result,
                )
            else:
                scores[result.chunk_id] = (rrf_score, result)

        # Sort by combined score and return top_k
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        # Update scores in results
        final_results = []
        for score, result in sorted_results:
            result.score = score
            result.metadata["search_type"] = "hybrid"
            final_results.append(result)

        logger.debug(
            "Hybrid search completed",
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            final_count=len(final_results),
        )

        # Apply reranking if enabled and available
        if self.config.enable_reranking and self._reranker is not None and query:
            final_results = self._rerank_results(query, final_results, top_k)

        return final_results

    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder model.

        Cross-encoders provide more accurate relevance scores by jointly
        encoding the query and document, at the cost of being slower.

        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        if not results or self._reranker is None:
            return results

        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [(query, result.content) for result in results]

            # Get reranking scores
            scores = self._reranker.predict(pairs)

            # Pair results with their rerank scores
            scored_results = list(zip(results, scores))

            # Sort by rerank score (higher is better)
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Update scores and return top_k
            reranked = []
            for result, score in scored_results[:top_k]:
                result.score = float(score)  # Use rerank score
                result.metadata["reranked"] = True
                result.metadata["rerank_score"] = float(score)
                reranked.append(result)

            logger.debug(
                "Reranked results",
                original_count=len(results),
                reranked_count=len(reranked),
            )

            return reranked

        except Exception as e:
            logger.warning("Reranking failed, returning original results", error=str(e))
            return results[:top_k]

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        search_type: Optional[SearchType] = None,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Unified search interface.

        Args:
            query: Search query string
            query_embedding: Optional query embedding (required for vector/hybrid)
            search_type: Type of search to perform
            top_k: Number of results
            access_tier_level: Maximum access tier level
            document_ids: Optional document filter
            session: Optional database session

        Returns:
            List of SearchResult objects
        """
        search_type = search_type or self.config.search_type

        if search_type == SearchType.VECTOR:
            if not query_embedding:
                raise ValueError("query_embedding required for vector search")
            return await self.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
                session=session,
            )

        elif search_type == SearchType.KEYWORD:
            return await self.keyword_search(
                query=query,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
                session=session,
            )

        else:  # HYBRID
            if not query_embedding:
                # Fall back to keyword search if no embedding
                logger.warning("No embedding provided for hybrid search, using keyword only")
                return await self.keyword_search(
                    query=query,
                    top_k=top_k,
                    access_tier_level=access_tier_level,
                    document_ids=document_ids,
                    session=session,
                )

            return await self.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                access_tier_level=access_tier_level,
                document_ids=document_ids,
                session=session,
            )

    # =========================================================================
    # Utility Operations
    # =========================================================================

    async def get_chunk_by_id(
        self,
        chunk_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        async def _get(db: AsyncSession) -> Optional[SearchResult]:
            result = await db.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.id == uuid.UUID(chunk_id))
            )
            row = result.first()

            if row:
                chunk, doc = row
                return SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=1.0,
                    metadata={"chunk_index": chunk.chunk_index},
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                )
            return None

        if session:
            return await _get(session)
        else:
            async with async_session_context() as db:
                return await _get(db)

    async def get_document_chunks(
        self,
        document_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """Get all chunks for a document."""
        async def _get(db: AsyncSession) -> List[SearchResult]:
            result = await db.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.document_id == uuid.UUID(document_id))
                .order_by(Chunk.chunk_index)
            )
            rows = result.all()

            results = []
            for chunk, doc in rows:
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=1.0,
                    metadata={"chunk_index": chunk.chunk_index},
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                ))

            return results

        if session:
            return await _get(session)
        else:
            async with async_session_context() as db:
                return await _get(db)

    async def get_stats(
        self,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """Get vector store statistics."""
        async def _stats(db: AsyncSession) -> Dict[str, Any]:
            # Total chunks
            chunk_count = await db.scalar(select(func.count(Chunk.id)))

            # Chunks with embeddings
            embedded_count = await db.scalar(
                select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
            )

            # Total documents
            doc_count = await db.scalar(select(func.count(Document.id)))

            return {
                "total_chunks": chunk_count or 0,
                "embedded_chunks": embedded_count or 0,
                "total_documents": doc_count or 0,
                "embedding_coverage": (
                    (embedded_count / chunk_count * 100) if chunk_count else 0
                ),
                "has_pgvector": self._has_pgvector,
            }

        if session:
            return await _stats(session)
        else:
            async with async_session_context() as db:
                return await _stats(db)


# =============================================================================
# Factory Function
# =============================================================================

_vector_store: Optional[VectorStore] = None


def get_vector_store(
    config: Optional[VectorStoreConfig] = None,
    backend: Optional[str] = None,
) -> VectorStore:
    """
    Get or create vector store instance.

    Supports multiple backends:
    - "pgvector" (default): PostgreSQL + pgvector (recommended for production)
    - "chroma": ChromaDB (local, no server required, good for development)
    - "auto": Auto-detect based on DATABASE_URL and configuration

    Args:
        config: Optional configuration
        backend: Backend type ("pgvector", "chroma", "auto", or None for default)

    Returns:
        VectorStore instance (either VectorStore or ChromaVectorStore)
    """
    import os
    global _vector_store

    # Determine backend
    if backend is None:
        backend = os.getenv("VECTOR_STORE_BACKEND", "auto")

    if backend == "auto":
        # Auto-detect: use ChromaDB if SQLite or explicitly configured
        database_url = os.getenv("DATABASE_URL", "")
        if "sqlite" in database_url or not HAS_PGVECTOR:
            backend = "chroma"
        else:
            backend = "pgvector"

    if backend == "chroma":
        # Use ChromaDB local vector store
        from backend.services.vectorstore_local import (
            get_chroma_vector_store,
            ChromaVectorStore,
        )
        return get_chroma_vector_store(config=config)

    # Default: PostgreSQL + pgvector
    if _vector_store is None or config is not None:
        _vector_store = VectorStore(config=config)

    return _vector_store


def get_vector_store_backend() -> str:
    """
    Get the current vector store backend type.

    Returns:
        "pgvector" or "chroma"
    """
    import os
    backend = os.getenv("VECTOR_STORE_BACKEND", "auto")

    if backend == "auto":
        database_url = os.getenv("DATABASE_URL", "")
        if "sqlite" in database_url or not HAS_PGVECTOR:
            return "chroma"
        return "pgvector"

    return backend
