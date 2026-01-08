"""
AIDocumentIndexer - Smart Document Pre-Filter
==============================================

Intelligently pre-filters documents before vector search to improve
performance on large collections (10k-100k+ documents).

Multi-Stage Filtering Pipeline:
1. Metadata filter: Use enhanced_metadata (keywords, topics, entities)
2. Summary similarity: Quick embedding match on document summaries
3. Recency bias: Recent documents get priority for time-sensitive queries
4. Entity matching: Match entities in query to document entities

This allows efficient RAG on large collections without searching all chunks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import structlog
import re

from sqlalchemy import select, desc, func, or_, and_
from sqlalchemy.orm import selectinload

from backend.db.database import async_session_context
from backend.db.models import Document as DBDocument

logger = structlog.get_logger(__name__)


class FilterStrategy(str, Enum):
    """Available pre-filtering strategies."""
    NONE = "none"                    # No pre-filtering (search all)
    METADATA = "metadata"            # Use keywords/topics from enhanced_metadata
    SUMMARY = "summary"              # Use document summaries for quick similarity
    HYBRID = "hybrid"                # Combine metadata + summary
    ADAPTIVE = "adaptive"            # Auto-select based on collection size


@dataclass
class SmartFilterConfig:
    """Configuration for smart pre-filtering."""
    # Thresholds for triggering pre-filtering
    min_docs_for_filter: int = 500       # Don't pre-filter if fewer docs
    max_docs_for_full_search: int = 5000 # Force pre-filter above this

    # Stage 1: Metadata filtering
    metadata_max_candidates: int = 1000  # Max docs from metadata stage
    keyword_match_boost: float = 2.0     # Boost for exact keyword matches
    topic_match_boost: float = 1.5       # Boost for topic matches

    # Stage 2: Summary similarity
    summary_top_k: int = 500             # Docs to keep after summary stage
    summary_min_similarity: float = 0.3  # Minimum summary similarity

    # Recency bias
    recency_enabled: bool = True         # Enable recency boosting
    recency_days: int = 30               # Recent window for boost
    recency_boost: float = 1.2           # Multiplier for recent docs

    # Entity matching
    entity_match_enabled: bool = True    # Match query entities to doc entities
    entity_match_boost: float = 1.5      # Boost for entity matches

    # Caching
    cache_ttl_seconds: int = 300         # Cache results for 5 minutes


@dataclass
class FilteredDocument:
    """Document that passed pre-filtering with score."""
    document_id: str
    score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)
    keywords_matched: List[str] = field(default_factory=list)
    topics_matched: List[str] = field(default_factory=list)
    entities_matched: List[str] = field(default_factory=list)


@dataclass
class FilterResult:
    """Result of pre-filtering operation."""
    document_ids: List[str]
    total_docs_scanned: int
    docs_after_filter: int
    filter_time_ms: float
    strategy_used: FilterStrategy
    stage_stats: Dict[str, Any] = field(default_factory=dict)


class SmartDocumentFilter:
    """
    Intelligently pre-filter documents before vector search.

    This service reduces the search space for large collections by:
    1. Using document metadata (keywords, topics) for fast initial filtering
    2. Using document summaries for quick semantic similarity
    3. Applying recency bias for time-sensitive queries
    4. Matching entities between query and documents
    """

    def __init__(
        self,
        config: Optional[SmartFilterConfig] = None,
        embedding_service: Optional[Any] = None,
    ):
        """
        Initialize smart filter.

        Args:
            config: Filter configuration
            embedding_service: Embedding service for summary similarity
        """
        self.config = config or SmartFilterConfig()
        self.embedding_service = embedding_service
        self._cache: Dict[str, Tuple[FilterResult, datetime]] = {}

        logger.info(
            "Initialized SmartDocumentFilter",
            min_docs_for_filter=self.config.min_docs_for_filter,
            max_docs_for_full_search=self.config.max_docs_for_full_search,
        )

    async def should_prefilter(
        self,
        collection_filter: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[bool, int]:
        """
        Check if pre-filtering should be applied based on collection size.

        Returns:
            Tuple of (should_filter, total_doc_count)
        """
        try:
            async with async_session_context() as db:
                # Build count query based on filters
                count_query = select(func.count(DBDocument.id))

                if document_ids is not None:
                    # Already filtered by folder
                    return len(document_ids) > self.config.min_docs_for_filter, len(document_ids)

                if collection_filter:
                    if collection_filter == "(Untagged)":
                        count_query = count_query.where(
                            or_(DBDocument.tags.is_(None), DBDocument.tags == [])
                        )
                    else:
                        from sqlalchemy import cast, String, literal
                        safe_filter = collection_filter.replace("\\", "\\\\")
                        safe_filter = safe_filter.replace("%", "\\%")
                        safe_filter = safe_filter.replace("_", "\\_")
                        safe_filter = safe_filter.replace('"', '\\"')
                        pattern = f'%"{safe_filter}"%'
                        count_query = count_query.where(
                            cast(DBDocument.tags, String).like(literal(pattern))
                        )

                result = await db.execute(count_query)
                total_count = result.scalar() or 0

                should_filter = total_count > self.config.min_docs_for_filter

                logger.info(
                    "Pre-filter decision",
                    total_docs=total_count,
                    should_filter=should_filter,
                    threshold=self.config.min_docs_for_filter,
                )

                return should_filter, total_count

        except Exception as e:
            logger.warning("Failed to check pre-filter requirement", error=str(e))
            return False, 0

    async def filter_documents(
        self,
        query: str,
        max_candidates: int = 1000,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        document_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> FilterResult:
        """
        Multi-stage filtering to reduce search space.

        Args:
            query: Search query
            max_candidates: Maximum documents to return
            collection_filter: Filter by collection
            access_tier: User's access tier
            document_ids: Optional pre-filtered document IDs
            query_embedding: Pre-computed query embedding for summary similarity

        Returns:
            FilterResult with filtered document IDs
        """
        import time
        start_time = time.time()

        # Check cache first
        cache_key = f"{query}:{collection_filter}:{max_candidates}"
        cached = self._get_cached_result(cache_key)
        if cached:
            return cached

        stage_stats = {}

        try:
            # Stage 1: Metadata filtering (keywords, topics, entities)
            metadata_candidates = await self._metadata_filter(
                query=query,
                max_candidates=self.config.metadata_max_candidates,
                collection_filter=collection_filter,
                access_tier=access_tier,
                document_ids=document_ids,
            )
            stage_stats["metadata_stage"] = {
                "input_count": "all" if document_ids is None else len(document_ids),
                "output_count": len(metadata_candidates),
            }

            # If metadata stage returned few results, skip further stages
            if len(metadata_candidates) <= max_candidates:
                doc_ids = [d.document_id for d in metadata_candidates]
                result = FilterResult(
                    document_ids=doc_ids,
                    total_docs_scanned=stage_stats["metadata_stage"]["input_count"],
                    docs_after_filter=len(doc_ids),
                    filter_time_ms=(time.time() - start_time) * 1000,
                    strategy_used=FilterStrategy.METADATA,
                    stage_stats=stage_stats,
                )
                self._cache_result(cache_key, result)
                return result

            # Stage 2: Summary similarity (if embedding service available)
            if self.embedding_service and query_embedding:
                summary_candidates = await self._summary_filter(
                    query_embedding=query_embedding,
                    candidates=metadata_candidates,
                    top_k=self.config.summary_top_k,
                )
                stage_stats["summary_stage"] = {
                    "input_count": len(metadata_candidates),
                    "output_count": len(summary_candidates),
                }
                candidates = summary_candidates
            else:
                candidates = metadata_candidates

            # Stage 3: Apply recency bias
            if self.config.recency_enabled:
                candidates = await self._apply_recency_bias(candidates)
                stage_stats["recency_applied"] = True

            # Sort by score and take top candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            final_candidates = candidates[:max_candidates]

            doc_ids = [d.document_id for d in final_candidates]

            result = FilterResult(
                document_ids=doc_ids,
                total_docs_scanned=stage_stats["metadata_stage"]["input_count"],
                docs_after_filter=len(doc_ids),
                filter_time_ms=(time.time() - start_time) * 1000,
                strategy_used=FilterStrategy.HYBRID,
                stage_stats=stage_stats,
            )

            self._cache_result(cache_key, result)

            logger.info(
                "Smart pre-filtering completed",
                query_length=len(query),
                final_candidates=len(doc_ids),
                filter_time_ms=result.filter_time_ms,
                stages=list(stage_stats.keys()),
            )

            return result

        except Exception as e:
            logger.error("Smart filtering failed", error=str(e), exc_info=True)
            # Fall back to no filtering
            return FilterResult(
                document_ids=document_ids or [],
                total_docs_scanned=len(document_ids) if document_ids else 0,
                docs_after_filter=len(document_ids) if document_ids else 0,
                filter_time_ms=(time.time() - start_time) * 1000,
                strategy_used=FilterStrategy.NONE,
                stage_stats={"error": str(e)},
            )

    async def _metadata_filter(
        self,
        query: str,
        max_candidates: int,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        document_ids: Optional[List[str]] = None,
    ) -> List[FilteredDocument]:
        """
        Stage 1: Filter documents based on metadata keywords and topics.

        Uses enhanced_metadata stored in the document for fast filtering
        without needing vector operations.
        """
        # Extract query keywords (simple tokenization)
        query_keywords = self._extract_keywords(query)

        candidates: List[FilteredDocument] = []

        async with async_session_context() as db:
            # Build base query
            stmt = select(DBDocument).where(DBDocument.is_soft_deleted == False)

            # Apply access tier filter
            stmt = stmt.where(DBDocument.access_tier <= access_tier)

            # Apply collection filter
            if collection_filter:
                if collection_filter == "(Untagged)":
                    stmt = stmt.where(
                        or_(DBDocument.tags.is_(None), DBDocument.tags == [])
                    )
                else:
                    from sqlalchemy import cast, String, literal
                    safe_filter = collection_filter.replace("\\", "\\\\")
                    safe_filter = safe_filter.replace("%", "\\%")
                    safe_filter = safe_filter.replace("_", "\\_")
                    safe_filter = safe_filter.replace('"', '\\"')
                    pattern = f'%"{safe_filter}"%'
                    stmt = stmt.where(
                        cast(DBDocument.tags, String).like(literal(pattern))
                    )

            # Apply document ID filter
            if document_ids:
                stmt = stmt.where(DBDocument.id.in_(document_ids))

            # Order by updated_at for recency
            stmt = stmt.order_by(desc(DBDocument.updated_at))

            # Limit to avoid memory issues
            stmt = stmt.limit(max_candidates * 3)  # Get more for scoring

            result = await db.execute(stmt)
            docs = result.scalars().all()

            for doc in docs:
                score = 0.0
                match_reasons = []
                keywords_matched = []
                topics_matched = []
                entities_matched = []

                # Score based on enhanced_metadata
                metadata = doc.enhanced_metadata or {}

                # Check keywords
                doc_keywords = metadata.get("keywords", [])
                if isinstance(doc_keywords, list):
                    for kw in doc_keywords:
                        if isinstance(kw, str):
                            kw_lower = kw.lower()
                            for qkw in query_keywords:
                                if qkw in kw_lower or kw_lower in qkw:
                                    score += self.config.keyword_match_boost
                                    keywords_matched.append(kw)
                                    match_reasons.append(f"keyword:{kw}")

                # Check topics
                doc_topics = metadata.get("topics", [])
                if isinstance(doc_topics, list):
                    for topic in doc_topics:
                        if isinstance(topic, str):
                            topic_lower = topic.lower()
                            for qkw in query_keywords:
                                if qkw in topic_lower:
                                    score += self.config.topic_match_boost
                                    topics_matched.append(topic)
                                    match_reasons.append(f"topic:{topic}")

                # Check entities (if enabled)
                if self.config.entity_match_enabled:
                    doc_entities = metadata.get("entities", [])
                    if isinstance(doc_entities, list):
                        for entity in doc_entities:
                            if isinstance(entity, str):
                                entity_lower = entity.lower()
                                for qkw in query_keywords:
                                    if qkw in entity_lower or entity_lower in qkw:
                                        score += self.config.entity_match_boost
                                        entities_matched.append(entity)
                                        match_reasons.append(f"entity:{entity}")

                # Also score on document title/filename
                doc_name = (doc.original_filename or doc.title or "").lower()
                for qkw in query_keywords:
                    if qkw in doc_name:
                        score += 1.0
                        match_reasons.append(f"title_match:{qkw}")

                # Only include if there's some match
                if score > 0 or not query_keywords:  # Include all if no keywords extracted
                    candidates.append(FilteredDocument(
                        document_id=str(doc.id),
                        score=score,
                        match_reasons=match_reasons,
                        keywords_matched=keywords_matched,
                        topics_matched=topics_matched,
                        entities_matched=entities_matched,
                    ))

        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "Metadata filter completed",
            total_docs=len(docs) if docs else 0,
            candidates_with_matches=len(candidates),
            query_keywords=query_keywords[:10],
        )

        return candidates[:max_candidates]

    async def _summary_filter(
        self,
        query_embedding: List[float],
        candidates: List[FilteredDocument],
        top_k: int,
    ) -> List[FilteredDocument]:
        """
        Stage 2: Filter based on document summary similarity.

        Uses pre-computed summary embeddings for quick similarity check.
        """
        if not candidates:
            return []

        # Get document summaries and their embeddings
        doc_ids = [c.document_id for c in candidates]

        async with async_session_context() as db:
            stmt = select(DBDocument).where(DBDocument.id.in_(doc_ids))
            result = await db.execute(stmt)
            docs = {str(d.id): d for d in result.scalars().all()}

        scored_candidates = []
        for candidate in candidates:
            doc = docs.get(candidate.document_id)
            if not doc:
                continue

            # Check if document has summary embedding
            metadata = doc.enhanced_metadata or {}
            summary_embedding = metadata.get("summary_embedding")

            if summary_embedding and isinstance(summary_embedding, list):
                # Compute cosine similarity
                similarity = self._cosine_similarity(query_embedding, summary_embedding)

                if similarity >= self.config.summary_min_similarity:
                    # Combine metadata score with summary similarity
                    candidate.score = candidate.score * 0.4 + similarity * 0.6
                    scored_candidates.append(candidate)
            else:
                # Keep candidates without summary embedding but lower score
                candidate.score *= 0.5
                scored_candidates.append(candidate)

        # Sort by combined score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "Summary filter completed",
            input_count=len(candidates),
            output_count=len(scored_candidates[:top_k]),
        )

        return scored_candidates[:top_k]

    async def _apply_recency_bias(
        self,
        candidates: List[FilteredDocument],
    ) -> List[FilteredDocument]:
        """
        Apply recency bias to boost recently updated documents.
        """
        if not candidates:
            return candidates

        doc_ids = [c.document_id for c in candidates]

        async with async_session_context() as db:
            stmt = select(DBDocument.id, DBDocument.updated_at).where(
                DBDocument.id.in_(doc_ids)
            )
            result = await db.execute(stmt)
            update_times = {str(row[0]): row[1] for row in result.fetchall()}

        cutoff_date = datetime.utcnow() - timedelta(days=self.config.recency_days)

        for candidate in candidates:
            updated_at = update_times.get(candidate.document_id)
            if updated_at and updated_at > cutoff_date:
                candidate.score *= self.config.recency_boost
                candidate.match_reasons.append("recent_update")

        return candidates

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Simple keyword extraction - remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "and",
            "but", "if", "or", "because", "until", "while", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "it",
            "its", "my", "your", "his", "her", "our", "their", "i", "you",
            "he", "she", "we", "they", "me", "him", "us", "them",
        }

        # Tokenize and clean
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _get_cached_result(self, cache_key: str) -> Optional[FilterResult]:
        """Get cached filter result if still valid."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.config.cache_ttl_seconds):
                return result
            else:
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: FilterResult):
        """Cache filter result."""
        # Limit cache size
        if len(self._cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )[:50]
            for key in oldest_keys:
                del self._cache[key]

        self._cache[cache_key] = (result, datetime.utcnow())

    def clear_cache(self):
        """Clear the filter cache."""
        self._cache.clear()
        logger.info("Smart filter cache cleared")


# Singleton instance
_smart_filter: Optional[SmartDocumentFilter] = None


def get_smart_filter(
    config: Optional[SmartFilterConfig] = None,
    embedding_service: Optional[Any] = None,
) -> SmartDocumentFilter:
    """Get or create the smart filter singleton."""
    global _smart_filter

    if _smart_filter is None or config is not None:
        _smart_filter = SmartDocumentFilter(
            config=config,
            embedding_service=embedding_service,
        )

    return _smart_filter
