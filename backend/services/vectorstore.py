"""
AIDocumentIndexer - Vector Store Service
=========================================

Provides vector storage and similarity search using PostgreSQL + pgvector.
Supports hybrid search (vector + keyword) with access tier filtering.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import structlog
from sqlalchemy import select, text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context, get_async_session_factory
from backend.db.models import Chunk, Document, AccessTier, HAS_PGVECTOR, ProcessingStatus

logger = structlog.get_logger(__name__)


# =============================================================================
# Query Parser for Search Operators
# =============================================================================

def parse_search_query(query: str) -> Tuple[str, bool]:
    """
    Parse a search query with AND/OR/NOT operators into PostgreSQL tsquery format.

    Supports:
    - AND: term1 AND term2 → term1 & term2
    - OR: term1 OR term2 → term1 | term2
    - NOT: NOT term → !term
    - Phrases: "exact phrase" → preserved with <->
    - Parentheses: (term1 OR term2) AND term3

    Args:
        query: User search query with optional operators

    Returns:
        Tuple of (tsquery_string, has_operators)
        - tsquery_string: PostgreSQL-compatible tsquery
        - has_operators: True if explicit operators were found

    Examples:
        "python AND machine learning" → ("python & machine & learning", True)
        "python OR java" → ("python | java", True)
        "python NOT java" → ("python & !java", True)
        "machine learning" → ("machine & learning", False)
        '"exact phrase"' → ("exact <-> phrase", True)
    """
    if not query or not query.strip():
        return ("", False)

    original_query = query.strip()

    # Check if query has explicit operators
    has_operators = bool(re.search(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE))

    # Extract quoted phrases first and replace with placeholders
    phrases = []
    phrase_pattern = r'"([^"]+)"'

    def replace_phrase(match):
        phrase = match.group(1).strip()
        phrases.append(phrase)
        return f"__PHRASE_{len(phrases) - 1}__"

    query = re.sub(phrase_pattern, replace_phrase, query)

    # If no explicit operators, convert to simple tsquery (plainto_tsquery behavior)
    if not has_operators and not phrases:
        # Just return the original - let PostgreSQL handle it with plainto_tsquery
        return (original_query, False)

    # Normalize operators to uppercase for consistent processing
    query = re.sub(r'\bAND\b', ' & ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bOR\b', ' | ', query, flags=re.IGNORECASE)
    query = re.sub(r'\bNOT\s+', ' !', query, flags=re.IGNORECASE)

    # Handle parentheses - keep them for grouping
    # Convert spaces between words (not operators) to &

    # Split by operators and parentheses while keeping them
    tokens = re.split(r'(\s*[&|!()]\s*)', query)

    result_parts = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Keep operators and parentheses as-is
        if token in ['&', '|', '!', '(', ')']:
            result_parts.append(token)
        elif token.startswith('__PHRASE_') and token.endswith('__'):
            # Restore phrase with phrase search operator
            idx = int(token.replace('__PHRASE_', '').replace('__', ''))
            phrase_words = phrases[idx].split()
            if len(phrase_words) > 1:
                # Use <-> for phrase matching (adjacent words)
                phrase_tsquery = ' <-> '.join(phrase_words)
                result_parts.append(f"({phrase_tsquery})")
            else:
                result_parts.append(phrase_words[0])
        else:
            # Regular term - might contain multiple words
            words = token.split()
            if len(words) > 1:
                # Multiple words without operator - AND them together
                result_parts.append('(' + ' & '.join(words) + ')')
            elif len(words) == 1:
                result_parts.append(words[0])

    # Join result
    result = ' '.join(result_parts)

    # Clean up multiple spaces and fix operator spacing
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*&\s*', ' & ', result)
    result = re.sub(r'\s*\|\s*', ' | ', result)
    result = re.sub(r'\s*!\s*', ' !', result)

    # Remove trailing/leading operators
    result = re.sub(r'^[\s&|]+', '', result)
    result = re.sub(r'[\s&|]+$', '', result)

    # If result is empty after processing, use original
    if not result.strip():
        return (original_query, False)

    return (result.strip(), True)


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
    score: float  # RRF score for ranking in hybrid search
    similarity_score: float = 0.0  # Original vector similarity (0-1) for display/confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Document info
    document_title: Optional[str] = None
    document_filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    collection: Optional[str] = None  # Collection/tag for document grouping

    # Enhanced metadata (from LLM analysis)
    enhanced_summary: Optional[str] = None
    enhanced_keywords: Optional[List[str]] = None

    # Context expansion (surrounding chunks for better context)
    prev_chunk_snippet: Optional[str] = None  # Preview of previous chunk
    next_chunk_snippet: Optional[str] = None  # Preview of next chunk
    chunk_index: Optional[int] = None  # Position in document for navigation


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    # Search settings
    default_top_k: int = 10
    similarity_threshold: float = 0.55  # Balanced threshold (0.4 was too permissive, 0.7 too strict for OCR)
    search_type: SearchType = SearchType.HYBRID

    # Context expansion (surrounding chunks)
    enable_context_expansion: bool = True  # Include prev/next chunk snippets
    context_snippet_length: int = 200  # Max characters for context snippets

    # Hybrid search weights
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    # Re-ranking with cross-encoder
    enable_reranking: bool = True  # Enabled by default for better accuracy
    rerank_top_k: int = 20  # Fetch more candidates for reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, accurate reranker

    # Enhanced metadata search (from LLM document analysis)
    use_enhanced_search: bool = True  # Search summaries, keywords, hypothetical questions
    enhanced_weight: float = 0.3  # Weight for enhanced metadata matches in RRF


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

        # ColBERT reranker (lazy-loaded when setting is enabled)
        self._colbert_reranker = None

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
            # Build all chunk objects first for batch insertion
            chunk_objects = []
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
                chunk_objects.append(chunk)
                chunk_ids.append(str(chunk.id))

            # Batch add all chunks at once (2-3x faster than individual adds)
            db.add_all(chunk_objects)
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
            from sqlalchemy import delete
            # Use bulk delete for better performance (avoids loading all objects)
            stmt = delete(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
            result = await db.execute(stmt)
            return result.rowcount

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
                # Exclude soft-deleted documents (status=FAILED with error="Deleted by user")
                .where(
                    ~(
                        (Document.processing_status == ProcessingStatus.FAILED)
                        & (Document.processing_error == "Deleted by user")
                    )
                )
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
                sim_float = float(similarity)
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=sim_float,
                    similarity_score=sim_float,  # Store original similarity
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
        Perform full-text keyword search with operator support.

        Supports search operators:
        - AND: term1 AND term2 (both must match)
        - OR: term1 OR term2 (either must match)
        - NOT: NOT term (exclude matches)
        - Phrases: "exact phrase" (words must appear adjacent)
        - Parentheses: (term1 OR term2) AND term3 (grouping)

        Args:
            query: Search query string (may contain operators)
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            session: Optional existing database session

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k

        # Parse query for operators (AND, OR, NOT, phrases)
        parsed_query, has_operators = parse_search_query(query)

        async def _search(db: AsyncSession) -> List[SearchResult]:
            # PostgreSQL full-text search
            # Use to_tsquery for operator support, plainto_tsquery for simple queries
            if has_operators:
                # Use to_tsquery for advanced queries with operators
                ts_query = func.to_tsquery('english', parsed_query)
            else:
                # Use plainto_tsquery for simple queries (handles stemming better)
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
                # Exclude soft-deleted documents (status=FAILED with error="Deleted by user")
                .where(
                    ~(
                        (Document.processing_status == ProcessingStatus.FAILED)
                        & (Document.processing_error == "Deleted by user")
                    )
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
                # Normalize rank to 0-1 range for display (keyword ranks are typically 1-N)
                # Use a reasonable default similarity for keyword matches
                normalized_score = min(1.0, 1.0 / (float(rank) + 1)) if rank > 0 else 0.5
                results.append(SearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.document_id),
                    content=chunk.content,
                    score=float(rank),
                    similarity_score=normalized_score,  # Keyword match score for display
                    metadata={
                        "chunk_index": chunk.chunk_index,
                        "search_type": "keyword",
                        "used_operators": has_operators,
                        "parsed_query": parsed_query if has_operators else None,
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

    async def enhanced_metadata_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Search documents using enhanced metadata (summaries, keywords, questions).

        This searches the LLM-extracted metadata stored in Document.enhanced_metadata
        including summaries, keywords, topics, and hypothetical questions.

        Args:
            query: Search query string
            top_k: Number of results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            session: Optional existing database session

        Returns:
            List of SearchResult objects (one per matching document)
        """
        top_k = top_k or self.config.default_top_k

        async def _search(db: AsyncSession) -> List[SearchResult]:
            # Get documents with enhanced metadata
            # Only include completed documents (excludes soft-deleted which are FAILED)
            base_query = (
                select(Document)
                .where(Document.enhanced_metadata.isnot(None))
                .where(Document.processing_status == ProcessingStatus.COMPLETED)
            )

            # Filter by document IDs if provided
            if document_ids:
                doc_uuids = [uuid.UUID(d) for d in document_ids]
                base_query = base_query.where(Document.id.in_(doc_uuids))

            result = await db.execute(base_query)
            documents = result.scalars().all()

            if not documents:
                return []

            # Score each document based on query match to enhanced metadata
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            scored_docs = []

            for doc in documents:
                metadata = doc.enhanced_metadata or {}
                score = 0.0

                # Search in summary
                summary_short = (metadata.get("summary_short") or "").lower()
                summary_detailed = (metadata.get("summary_detailed") or "").lower()
                if query_lower in summary_short:
                    score += 0.5
                if query_lower in summary_detailed:
                    score += 0.3

                # Search in keywords
                keywords = [k.lower() for k in metadata.get("keywords", [])]
                keyword_matches = sum(1 for term in query_terms if any(term in k for k in keywords))
                score += keyword_matches * 0.2

                # Search in topics
                topics = [t.lower() for t in metadata.get("topics", [])]
                topic_matches = sum(1 for term in query_terms if any(term in t for t in topics))
                score += topic_matches * 0.15

                # Search in hypothetical questions
                questions = [q.lower() for q in metadata.get("hypothetical_questions", [])]
                for question in questions:
                    # Check for query term overlap with questions
                    question_terms = set(question.split())
                    overlap = len(query_terms & question_terms)
                    if overlap > 0:
                        score += overlap * 0.1

                # Search in entities
                entities = metadata.get("entities", {})
                for entity_type, entity_list in entities.items():
                    entity_values = [e.lower() for e in entity_list]
                    entity_matches = sum(1 for term in query_terms if any(term in e for e in entity_values))
                    score += entity_matches * 0.1

                if score > 0:
                    scored_docs.append((doc, score, metadata))

            # Sort by score and take top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = scored_docs[:top_k]

            # Convert to SearchResult (using first chunk as representative)
            results = []
            for doc, score, metadata in top_docs:
                # Get first chunk for content preview
                chunk_result = await db.execute(
                    select(Chunk)
                    .where(Chunk.document_id == doc.id)
                    .order_by(Chunk.chunk_index)
                    .limit(1)
                )
                first_chunk = chunk_result.scalar_one_or_none()

                summary = metadata.get("summary_detailed") or metadata.get("summary_short") or ""
                content = summary if summary else (first_chunk.content if first_chunk else "")

                results.append(SearchResult(
                    chunk_id=str(first_chunk.id) if first_chunk else "",
                    document_id=str(doc.id),
                    content=content,
                    score=score,
                    similarity_score=min(score, 1.0),  # Metadata match score for display
                    metadata={
                        "search_type": "enhanced",
                        "topics": metadata.get("topics", []),
                        "document_type": metadata.get("document_type"),
                    },
                    document_title=doc.title or doc.filename,
                    document_filename=doc.filename,
                    enhanced_summary=metadata.get("summary_short"),
                    enhanced_keywords=metadata.get("keywords", []),
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
        use_enhanced: Optional[bool] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity, keyword matching,
        and optionally enhanced metadata search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from all sources.

        Args:
            query: Search query string
            query_embedding: Query embedding vector
            top_k: Number of final results to return
            access_tier_level: Maximum access tier level for filtering
            document_ids: Optional list of document IDs to search within
            vector_weight: Weight for vector results (0-1)
            keyword_weight: Weight for keyword results (0-1)
            use_enhanced: Whether to include enhanced metadata search
            session: Optional existing database session

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.config.default_top_k
        vec_weight = vector_weight or self.config.vector_weight
        kw_weight = keyword_weight or self.config.keyword_weight
        enhanced_weight = self.config.enhanced_weight
        use_enhanced = use_enhanced if use_enhanced is not None else self.config.use_enhanced_search

        # Get more results from each method for better fusion
        fetch_k = self.config.rerank_top_k if self.config.enable_reranking else top_k * 2

        # Run searches in parallel conceptually (await each)
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

        # Optionally search enhanced metadata
        enhanced_results = []
        if use_enhanced:
            enhanced_results = await self.enhanced_metadata_search(
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

        # Process vector results - preserve original similarity scores
        for rank, result in enumerate(vector_results):
            rrf_score = vec_weight * (1.0 / (k + rank + 1))
            original_similarity = result.similarity_score  # Preserve before RRF overwrites
            if result.chunk_id in scores:
                existing_rrf, existing_result = scores[result.chunk_id]
                # Keep max similarity from vector search
                existing_result.similarity_score = max(existing_result.similarity_score, original_similarity)
                scores[result.chunk_id] = (existing_rrf + rrf_score, existing_result)
            else:
                # Ensure similarity_score is preserved (already set from similarity_search)
                scores[result.chunk_id] = (rrf_score, result)

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            rrf_score = kw_weight * (1.0 / (k + rank + 1))
            if result.chunk_id in scores:
                existing_rrf, existing_result = scores[result.chunk_id]
                scores[result.chunk_id] = (existing_rrf + rrf_score, existing_result)
            else:
                # Keyword results don't have similarity score - leave as 0
                scores[result.chunk_id] = (rrf_score, result)

        # Process enhanced metadata results
        # Enhanced results are document-level, so we boost all chunks from matching docs
        enhanced_doc_scores: Dict[str, float] = {}
        for rank, result in enumerate(enhanced_results):
            rrf_score = enhanced_weight * (1.0 / (k + rank + 1))
            enhanced_doc_scores[result.document_id] = rrf_score
            # Also add the enhanced result itself if it has a chunk
            if result.chunk_id:
                if result.chunk_id in scores:
                    existing_score, existing_result = scores[result.chunk_id]
                    # Merge enhanced metadata into existing result
                    existing_result.enhanced_summary = result.enhanced_summary
                    existing_result.enhanced_keywords = result.enhanced_keywords
                    scores[result.chunk_id] = (existing_score + rrf_score, existing_result)
                else:
                    scores[result.chunk_id] = (rrf_score, result)

        # Boost existing chunk scores for documents with enhanced metadata matches
        for chunk_id, (current_score, result) in list(scores.items()):
            if result.document_id in enhanced_doc_scores:
                boost = enhanced_doc_scores[result.document_id] * 0.5  # 50% of doc-level score
                scores[chunk_id] = (current_score + boost, result)

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
            if use_enhanced and result.document_id in enhanced_doc_scores:
                result.metadata["enhanced_boost"] = True
            final_results.append(result)

        logger.debug(
            "Hybrid search completed",
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            enhanced_count=len(enhanced_results),
            final_count=len(final_results),
        )

        # Apply reranking if enabled and available
        if self.config.enable_reranking and query:
            final_results = await self._rerank_results_async(query, final_results, top_k)

        # Expand context with surrounding chunks
        if self.config.enable_context_expansion:
            final_results = await self.expand_context(final_results, session=session)

        return final_results

    async def _rerank_results_async(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank search results using ColBERT or cross-encoder based on settings.

        Checks rag.use_colbert_reranker setting to decide which reranker to use.
        ColBERT provides 10-20% better precision with similar speed.

        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return results

        # Check if ColBERT reranking is enabled
        try:
            from backend.services.settings import get_settings_service
            settings = get_settings_service()
            use_colbert = await settings.get_setting("rag.use_colbert_reranker")

            if use_colbert:
                return await self._rerank_with_colbert(query, results, top_k)
        except Exception as e:
            logger.debug("Could not check ColBERT setting, using cross-encoder", error=str(e))

        # Fall back to cross-encoder
        return self._rerank_results(query, results, top_k)

    async def _rerank_with_colbert(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank results using ColBERT reranker.

        ColBERT uses late interaction (MaxSim) scoring for better precision.
        """
        try:
            # Lazy-load ColBERT reranker
            if self._colbert_reranker is None:
                from backend.services.colbert_reranker import ColBERTReranker
                self._colbert_reranker = ColBERTReranker()
                logger.info("Initialized ColBERT reranker")

            # Prepare documents for ColBERT (expects list of dicts)
            documents = [
                {
                    "content": result.content,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in results
            ]

            # Rerank using ColBERT
            reranked_results = await self._colbert_reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )

            # Build lookup for original results by chunk_id
            result_lookup = {r.chunk_id: r for r in results}

            # Map reranked results back to SearchResult objects
            reranked = []
            for rr in reranked_results:
                if rr.chunk_id in result_lookup:
                    result = result_lookup[rr.chunk_id]
                    result.score = rr.rerank_score
                    result.metadata["reranked"] = True
                    result.metadata["rerank_score"] = rr.rerank_score
                    result.metadata["reranker"] = "colbert"
                    reranked.append(result)

            logger.debug(
                "ColBERT reranked results",
                original_count=len(results),
                reranked_count=len(reranked),
            )

            return reranked

        except Exception as e:
            logger.warning("ColBERT reranking failed, falling back to cross-encoder", error=str(e))
            return self._rerank_results(query, results, top_k)

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
                result.metadata["reranker"] = "cross-encoder"
                reranked.append(result)

            logger.debug(
                "Cross-encoder reranked results",
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
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
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
            vector_weight: Dynamic weight for vector results (0-1), overrides config
            keyword_weight: Dynamic weight for keyword results (0-1), overrides config

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
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )

    # =========================================================================
    # Context Expansion
    # =========================================================================

    async def expand_context(
        self,
        results: List[SearchResult],
        session: Optional[AsyncSession] = None,
    ) -> List[SearchResult]:
        """
        Expand search results with surrounding chunk context.

        For each result, fetches snippets from the previous and next chunks
        in the same document to provide better context for the LLM.

        Args:
            results: List of search results to expand
            session: Optional existing database session

        Returns:
            The same results with prev_chunk_snippet and next_chunk_snippet populated
        """
        if not self.config.enable_context_expansion or not results:
            return results

        async def _expand(db: AsyncSession) -> List[SearchResult]:
            # Group results by document for efficient querying
            doc_chunks: Dict[str, List[Tuple[int, SearchResult]]] = {}
            for result in results:
                chunk_index = result.metadata.get("chunk_index")
                if chunk_index is not None:
                    if result.document_id not in doc_chunks:
                        doc_chunks[result.document_id] = []
                    doc_chunks[result.document_id].append((chunk_index, result))
                    result.chunk_index = chunk_index

            # For each document, fetch surrounding chunks
            snippet_length = self.config.context_snippet_length

            for doc_id, chunks in doc_chunks.items():
                # Get all relevant chunk indices
                indices = set()
                for chunk_index, _ in chunks:
                    if chunk_index > 0:
                        indices.add(chunk_index - 1)  # Previous
                    indices.add(chunk_index + 1)  # Next

                if not indices:
                    continue

                # Fetch surrounding chunks in one query
                try:
                    surrounding = await db.execute(
                        select(Chunk.chunk_index, Chunk.content)
                        .where(Chunk.document_id == uuid.UUID(doc_id))
                        .where(Chunk.chunk_index.in_(list(indices)))
                    )
                    surrounding_map = {row[0]: row[1] for row in surrounding.fetchall()}

                    # Populate prev/next snippets
                    for chunk_index, result in chunks:
                        # Previous chunk snippet
                        if chunk_index > 0 and (chunk_index - 1) in surrounding_map:
                            prev_content = surrounding_map[chunk_index - 1]
                            # Take last snippet_length characters
                            if len(prev_content) > snippet_length:
                                result.prev_chunk_snippet = "..." + prev_content[-snippet_length:]
                            else:
                                result.prev_chunk_snippet = prev_content

                        # Next chunk snippet
                        if (chunk_index + 1) in surrounding_map:
                            next_content = surrounding_map[chunk_index + 1]
                            # Take first snippet_length characters
                            if len(next_content) > snippet_length:
                                result.next_chunk_snippet = next_content[:snippet_length] + "..."
                            else:
                                result.next_chunk_snippet = next_content

                except Exception as e:
                    logger.warning(
                        "Failed to fetch surrounding chunks",
                        document_id=doc_id,
                        error=str(e),
                    )

            return results

        if session:
            return await _expand(session)
        else:
            async with async_session_context() as db:
                return await _expand(db)

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
                    similarity_score=1.0,  # Direct document access = 100% match
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
