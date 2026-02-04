"""
AIDocumentIndexer - LightRAG Retriever
========================================

Implements LightRAG-style dual-level retrieval for 10x token reduction.

LightRAG Architecture (EMNLP 2025):
- Dual-level retrieval: Low-level (specific entities) + High-level (concepts)
- 10x token reduction vs GraphRAG
- 65-80% cost savings for 1,500+ documents

Key Concepts:
1. Low-level retrieval: Specific entities, facts, data points
2. High-level retrieval: Concepts, themes, relationships
3. Fusion: Combine both levels for comprehensive answers

Research:
- LightRAG: Simple and Fast Retrieval-Augmented Generation (EMNLP 2025)
- Better than GraphRAG for most queries at fraction of cost
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from backend.core.config import settings
from backend.core.performance import gather_with_concurrency, LRUCache

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class RetrievalLevel(str, Enum):
    """Retrieval levels for LightRAG."""
    LOW = "low"       # Specific entities, facts
    HIGH = "high"     # Concepts, themes
    HYBRID = "hybrid" # Both levels combined


@dataclass
class LightRAGConfig:
    """Configuration for LightRAG retriever."""
    # Low-level (entity) retrieval
    low_level_top_k: int = 20
    entity_similarity_threshold: float = 0.6

    # High-level (concept) retrieval
    high_level_top_k: int = 10
    concept_similarity_threshold: float = 0.5

    # Fusion settings
    fusion_method: str = "rrf"  # "rrf" or "weighted"
    low_level_weight: float = 0.6
    high_level_weight: float = 0.4
    rrf_k: int = 60  # RRF constant

    # Final output
    final_top_k: int = 10

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    # Concurrency
    max_concurrent: int = 10


@dataclass(slots=True)
class LightRAGResult:
    """Result from LightRAG retrieval."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    level: RetrievalLevel
    entity_matches: List[str] = field(default_factory=list)
    concept_matches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Document metadata for proper citation
    document_filename: Optional[str] = None
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


@dataclass(slots=True)
class EntityMatch:
    """Entity match for low-level retrieval."""
    entity_id: str
    entity_name: str
    entity_type: str
    score: float
    chunk_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ConceptMatch:
    """Concept match for high-level retrieval."""
    concept: str
    description: str
    score: float
    related_entities: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)


# =============================================================================
# LightRAG Retriever
# =============================================================================

class LightRAGRetriever:
    """
    LightRAG-style dual-level retriever.

    Implements the key insight from LightRAG: separate retrieval into
    low-level (specific entities/facts) and high-level (concepts/themes),
    then fuse results for comprehensive answers.

    Benefits:
    - 10x token reduction vs GraphRAG
    - Better handling of both specific and abstract queries
    - 65-80% cost savings for large document collections

    Usage:
        retriever = LightRAGRetriever(vectorstore, kg_service)
        await retriever.initialize()

        results = await retriever.retrieve(
            query="What is the revenue of Apple?",
            level=RetrievalLevel.HYBRID,
        )
    """

    def __init__(
        self,
        vectorstore,
        knowledge_graph_service=None,
        config: Optional[LightRAGConfig] = None,
    ):
        """
        Initialize LightRAG retriever.

        Args:
            vectorstore: Vector store for semantic search
            knowledge_graph_service: KG service for entity/concept retrieval
            config: Configuration options
        """
        self.vectorstore = vectorstore
        self.kg_service = knowledge_graph_service
        self.config = config or LightRAGConfig()

        self._entity_cache = LRUCache[List[EntityMatch]](capacity=1000)
        self._concept_cache = LRUCache[List[ConceptMatch]](capacity=500)
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the retriever."""
        if self._initialized:
            return True

        try:
            # Initialize KG service if provided
            if self.kg_service:
                logger.info("LightRAG retriever initialized with knowledge graph")
            else:
                logger.info("LightRAG retriever initialized (vector-only mode)")

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize LightRAG retriever", error=str(e))
            return False

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        level: RetrievalLevel = RetrievalLevel.HYBRID,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        access_tier_level: int = 100,
        organization_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[LightRAGResult]:
        """
        Perform LightRAG dual-level retrieval.

        Args:
            query: Search query
            query_embedding: Pre-computed query embedding
            level: Retrieval level (low, high, or hybrid)
            top_k: Number of results to return
            document_ids: Filter to specific documents
            access_tier_level: Access tier filter

        Returns:
            List of LightRAGResult objects
        """
        if not await self.initialize():
            logger.warning("LightRAG not initialized, falling back to vector search")
            return await self._vector_only_search(
                query, query_embedding, top_k or self.config.final_top_k,
                organization_id, is_superadmin
            )

        import time
        start_time = time.time()

        final_top_k = top_k or self.config.final_top_k

        logger.info(
            "Starting LightRAG retrieval",
            query_length=len(query),
            level=level.value,
            top_k=final_top_k,
        )

        # Parallel retrieval based on level
        if level == RetrievalLevel.LOW:
            results = await self._low_level_retrieve(
                query, query_embedding, final_top_k, document_ids, access_tier_level,
                organization_id, is_superadmin
            )
        elif level == RetrievalLevel.HIGH:
            results = await self._high_level_retrieve(
                query, query_embedding, final_top_k, document_ids, access_tier_level,
                organization_id, is_superadmin
            )
        else:  # HYBRID
            # Run both levels in parallel
            low_task = self._low_level_retrieve(
                query, query_embedding,
                self.config.low_level_top_k, document_ids, access_tier_level,
                organization_id, is_superadmin
            )
            high_task = self._high_level_retrieve(
                query, query_embedding,
                self.config.high_level_top_k, document_ids, access_tier_level,
                organization_id, is_superadmin
            )

            low_results, high_results = await asyncio.gather(low_task, high_task)

            # Fuse results
            results = self._fuse_results(low_results, high_results, final_top_k)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "LightRAG retrieval complete",
            level=level.value,
            results=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return results

    async def _low_level_retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        top_k: int,
        document_ids: Optional[List[str]],
        access_tier_level: int,
        organization_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[LightRAGResult]:
        """
        Low-level retrieval: Focus on specific entities and facts.

        Strategy:
        1. Extract entities from query
        2. Find matching entities in knowledge graph
        3. Retrieve chunks containing those entities
        4. Score by entity relevance + semantic similarity
        """
        results = []
        entity_matches = []

        # Step 1: Extract entities from query using KG service
        if self.kg_service:
            try:
                query_entities = await self._extract_query_entities(query)

                # Step 2: Find matching entities in KG
                for entity_name in query_entities:
                    matches = await self._find_entity_matches(
                        entity_name, access_tier_level
                    )
                    entity_matches.extend(matches)

            except Exception as e:
                logger.debug("Entity extraction failed, using vector only", error=str(e))

        # Step 3: Get chunks for matched entities
        matched_chunk_ids = set()
        entity_by_chunk: Dict[str, List[str]] = {}

        for match in entity_matches:
            for chunk_id in match.chunk_ids:
                matched_chunk_ids.add(chunk_id)
                if chunk_id not in entity_by_chunk:
                    entity_by_chunk[chunk_id] = []
                entity_by_chunk[chunk_id].append(match.entity_name)

        # Step 4: Vector search with entity boost
        from backend.services.vectorstore import SearchType

        vector_results = await self.vectorstore.search(
            query=query,
            query_embedding=query_embedding,
            search_type=SearchType.HYBRID,
            top_k=top_k * 2,  # Get more for filtering
            access_tier_level=access_tier_level,
            document_ids=document_ids,
            organization_id=organization_id,
            is_superadmin=is_superadmin,
        )

        # Step 5: Score and combine
        for vr in vector_results:
            entity_boost = 0.0
            matched_entities = entity_by_chunk.get(vr.chunk_id, [])

            if matched_entities:
                # Boost score for entity matches
                entity_boost = 0.3 * len(matched_entities)

            final_score = vr.score + entity_boost

            results.append(LightRAGResult(
                chunk_id=vr.chunk_id,
                document_id=vr.document_id,
                content=vr.content,
                score=final_score,
                level=RetrievalLevel.LOW,
                entity_matches=matched_entities,
                metadata={
                    "original_score": vr.score,
                    "entity_boost": entity_boost,
                    **vr.metadata,
                },
                # Include document metadata for proper citations
                document_filename=getattr(vr, 'document_filename', None) or vr.metadata.get('document_filename'),
                document_title=getattr(vr, 'document_title', None) or vr.metadata.get('document_title'),
                page_number=getattr(vr, 'page_number', None) or vr.metadata.get('page_number'),
                section_title=getattr(vr, 'section_title', None) or vr.metadata.get('section_title'),
            ))

        # Sort by score and take top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _high_level_retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        top_k: int,
        document_ids: Optional[List[str]],
        access_tier_level: int,
        organization_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[LightRAGResult]:
        """
        High-level retrieval: Focus on concepts and themes.

        Strategy:
        1. Extract concepts/themes from query
        2. Find related concepts in document summaries
        3. Retrieve chunks from conceptually relevant documents
        4. Score by concept relevance + semantic similarity
        """
        results = []
        concept_matches = []

        # Step 1: Extract concepts from query
        query_concepts = await self._extract_query_concepts(query)

        # Step 2: Find matching concepts using document metadata
        if query_concepts:
            concept_matches = await self._find_concept_matches(
                query_concepts, access_tier_level
            )

        # Step 3: Get prioritized document IDs from concept matches
        concept_doc_ids = set()
        concept_by_doc: Dict[str, List[str]] = {}

        for match in concept_matches:
            for chunk_id in match.chunk_ids:
                # Extract document ID from chunk ID if needed
                doc_id = self._extract_doc_id_from_chunk(chunk_id)
                if doc_id:
                    concept_doc_ids.add(doc_id)
                    if doc_id not in concept_by_doc:
                        concept_by_doc[doc_id] = []
                    concept_by_doc[doc_id].append(match.concept)

        # Step 4: Vector search with concept boost
        from backend.services.vectorstore import SearchType

        # If we have concept-matched docs, prioritize them
        search_doc_ids = document_ids
        if concept_doc_ids and not document_ids:
            search_doc_ids = list(concept_doc_ids)[:50]  # Top 50 concept-matched docs

        vector_results = await self.vectorstore.search(
            query=query,
            query_embedding=query_embedding,
            search_type=SearchType.HYBRID,
            top_k=top_k * 2,
            access_tier_level=access_tier_level,
            document_ids=search_doc_ids,
            organization_id=organization_id,
            is_superadmin=is_superadmin,
        )

        # Step 5: Score and combine
        for vr in vector_results:
            concept_boost = 0.0
            matched_concepts = concept_by_doc.get(vr.document_id, [])

            if matched_concepts:
                concept_boost = 0.25 * len(matched_concepts)

            final_score = vr.score + concept_boost

            results.append(LightRAGResult(
                chunk_id=vr.chunk_id,
                document_id=vr.document_id,
                content=vr.content,
                score=final_score,
                level=RetrievalLevel.HIGH,
                concept_matches=matched_concepts,
                metadata={
                    "original_score": vr.score,
                    "concept_boost": concept_boost,
                    **vr.metadata,
                },
                # Include document metadata for proper citations
                document_filename=getattr(vr, 'document_filename', None) or vr.metadata.get('document_filename'),
                document_title=getattr(vr, 'document_title', None) or vr.metadata.get('document_title'),
                page_number=getattr(vr, 'page_number', None) or vr.metadata.get('page_number'),
                section_title=getattr(vr, 'section_title', None) or vr.metadata.get('section_title'),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _fuse_results(
        self,
        low_results: List[LightRAGResult],
        high_results: List[LightRAGResult],
        top_k: int,
    ) -> List[LightRAGResult]:
        """
        Fuse low-level and high-level results.

        Uses Reciprocal Rank Fusion (RRF) for robust combination.
        """
        if self.config.fusion_method == "rrf":
            return self._rrf_fusion(low_results, high_results, top_k)
        else:
            return self._weighted_fusion(low_results, high_results, top_k)

    def _rrf_fusion(
        self,
        low_results: List[LightRAGResult],
        high_results: List[LightRAGResult],
        top_k: int,
    ) -> List[LightRAGResult]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each result list
        where k is a constant (default 60) to dampen high-ranking items.
        """
        k = self.config.rrf_k

        # Build score map
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, LightRAGResult] = {}

        # Process low-level results
        for rank, result in enumerate(low_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += self.config.low_level_weight / (k + rank + 1)

            if chunk_id not in result_map:
                result_map[chunk_id] = result
            else:
                # Merge entity matches
                result_map[chunk_id].entity_matches.extend(result.entity_matches)

        # Process high-level results
        for rank, result in enumerate(high_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += self.config.high_level_weight / (k + rank + 1)

            if chunk_id not in result_map:
                result_map[chunk_id] = result
            else:
                # Merge concept matches
                result_map[chunk_id].concept_matches.extend(result.concept_matches)

        # Sort by RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build final results
        final_results = []
        for chunk_id, rrf_score in sorted_chunks[:top_k]:
            result = result_map[chunk_id]
            result.score = rrf_score
            result.level = RetrievalLevel.HYBRID
            result.metadata["rrf_score"] = rrf_score
            final_results.append(result)

        return final_results

    def _weighted_fusion(
        self,
        low_results: List[LightRAGResult],
        high_results: List[LightRAGResult],
        top_k: int,
    ) -> List[LightRAGResult]:
        """Simple weighted score fusion."""
        # Normalize scores
        def normalize(results: List[LightRAGResult]) -> Dict[str, float]:
            if not results:
                return {}
            max_score = max(r.score for r in results)
            min_score = min(r.score for r in results)
            if max_score == min_score:
                return {r.chunk_id: 1.0 for r in results}
            return {
                r.chunk_id: (r.score - min_score) / (max_score - min_score)
                for r in results
            }

        low_norm = normalize(low_results)
        high_norm = normalize(high_results)

        # Combine scores
        all_chunks = set(low_norm.keys()) | set(high_norm.keys())
        result_map = {}

        for r in low_results + high_results:
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        combined_scores = {}
        for chunk_id in all_chunks:
            low_score = low_norm.get(chunk_id, 0.0) * self.config.low_level_weight
            high_score = high_norm.get(chunk_id, 0.0) * self.config.high_level_weight
            combined_scores[chunk_id] = low_score + high_score

        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        final_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            result = result_map[chunk_id]
            result.score = score
            result.level = RetrievalLevel.HYBRID
            final_results.append(result)

        return final_results

    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entity names from query using KG service."""
        if not self.kg_service:
            return []

        # Check cache
        cache_key = hashlib.md5(f"entities:{query}".encode()).hexdigest()[:16]
        cached = await self._entity_cache.get(cache_key)
        if cached:
            return [e.entity_name for e in cached]

        try:
            # Use KG service entity extraction
            from backend.db.database import async_session_context
            async with async_session_context() as db:
                entities = await self.kg_service.extract_entities(query, db)
                entity_names = [e.name for e in entities]
                return entity_names
        except Exception as e:
            logger.debug("Query entity extraction failed", error=str(e))
            return []

    async def _find_entity_matches(
        self,
        entity_name: str,
        access_tier_level: int,
    ) -> List[EntityMatch]:
        """Find entities matching the given name in the knowledge graph."""
        if not self.kg_service:
            return []

        try:
            from backend.db.database import async_session_context
            from backend.db.models import Entity, EntityMention
            from sqlalchemy import select

            async with async_session_context() as db:
                # Search for matching entities
                stmt = (
                    select(Entity)
                    .where(Entity.name.ilike(f"%{entity_name}%"))
                    .limit(10)
                )
                result = await db.execute(stmt)
                entities = result.scalars().all()

                matches = []
                for entity in entities:
                    # Get chunk IDs from mentions
                    mention_stmt = (
                        select(EntityMention.chunk_id)
                        .where(EntityMention.entity_id == entity.id)
                        .limit(20)
                    )
                    mention_result = await db.execute(mention_stmt)
                    chunk_ids = [str(m) for m in mention_result.scalars().all() if m]

                    matches.append(EntityMatch(
                        entity_id=str(entity.id),
                        entity_name=entity.name,
                        entity_type=entity.entity_type.value if entity.entity_type else "unknown",
                        score=1.0 if entity.name.lower() == entity_name.lower() else 0.8,
                        chunk_ids=chunk_ids,
                    ))

                return matches

        except Exception as e:
            logger.debug("Entity match search failed", error=str(e))
            return []

    async def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract high-level concepts from query."""
        # Simple keyword-based concept extraction
        # In production, could use LLM for better concept extraction

        import re

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "how",
            "why", "when", "where", "who", "which", "do", "does", "did",
            "can", "could", "would", "should", "about", "for", "to", "of",
            "in", "on", "at", "by", "from", "with", "and", "or", "but",
        }

        words = re.findall(r'\b\w+\b', query.lower())
        concepts = [w for w in words if w not in stop_words and len(w) > 3]

        return concepts

    async def _find_concept_matches(
        self,
        concepts: List[str],
        access_tier_level: int,
    ) -> List[ConceptMatch]:
        """Find documents/chunks matching high-level concepts."""
        matches = []

        try:
            from backend.db.database import async_session_context
            from backend.db.models import Document as DBDocument
            from sqlalchemy import select, or_

            async with async_session_context() as db:
                for concept in concepts[:5]:  # Limit to top 5 concepts
                    # Search in document keywords and topics
                    # Join with AccessTier to filter by level
                    from backend.db.models import AccessTier
                    stmt = (
                        select(DBDocument)
                        .join(AccessTier, DBDocument.access_tier_id == AccessTier.id)
                        .where(AccessTier.level <= access_tier_level)
                        .where(
                            or_(
                                DBDocument.title.ilike(f"%{concept}%"),
                                DBDocument.original_filename.ilike(f"%{concept}%"),
                            )
                        )
                        .limit(10)
                    )
                    result = await db.execute(stmt)
                    docs = result.scalars().all()

                    if docs:
                        chunk_ids = []
                        for doc in docs:
                            # Get first few chunk IDs from each doc
                            from backend.db.models import Chunk
                            chunk_stmt = (
                                select(Chunk.id)
                                .where(Chunk.document_id == doc.id)
                                .limit(5)
                            )
                            chunk_result = await db.execute(chunk_stmt)
                            chunk_ids.extend([str(c) for c in chunk_result.scalars().all()])

                        matches.append(ConceptMatch(
                            concept=concept,
                            description=f"Documents related to '{concept}'",
                            score=0.8,
                            related_entities=[],
                            chunk_ids=chunk_ids,
                        ))

                return matches

        except Exception as e:
            logger.debug("Concept match search failed", error=str(e))
            return []

    def _extract_doc_id_from_chunk(self, chunk_id: str) -> Optional[str]:
        """Extract document ID from chunk ID if encoded."""
        # Chunk IDs are typically UUIDs, document info stored in chunk metadata
        # For now, return None - actual extraction depends on chunk ID format
        return None

    async def _vector_only_search(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        top_k: int,
        organization_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[LightRAGResult]:
        """Fallback to vector-only search."""
        from backend.services.vectorstore import SearchType

        results = await self.vectorstore.search(
            query=query,
            query_embedding=query_embedding,
            search_type=SearchType.HYBRID,
            top_k=top_k,
            organization_id=organization_id,
            is_superadmin=is_superadmin,
        )

        return [
            LightRAGResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                level=RetrievalLevel.HYBRID,
                metadata=r.metadata,
                # Include document metadata for proper citations
                document_filename=getattr(r, 'document_filename', None) or r.metadata.get('document_filename'),
                document_title=getattr(r, 'document_title', None) or r.metadata.get('document_title'),
                page_number=getattr(r, 'page_number', None) or r.metadata.get('page_number'),
                section_title=getattr(r, 'section_title', None) or r.metadata.get('section_title'),
            )
            for r in results
        ]


# =============================================================================
# Singleton Management
# =============================================================================

_lightrag_retriever: Optional[LightRAGRetriever] = None
_retriever_lock = asyncio.Lock()


async def get_lightrag_retriever(
    vectorstore=None,
    knowledge_graph_service=None,
    config: Optional[LightRAGConfig] = None,
) -> LightRAGRetriever:
    """
    Get or create LightRAG retriever singleton.

    Args:
        vectorstore: Vector store service (required on first call)
        knowledge_graph_service: Optional KG service for entity retrieval
        config: Optional configuration

    Returns:
        LightRAGRetriever instance
    """
    global _lightrag_retriever

    if _lightrag_retriever is not None:
        return _lightrag_retriever

    async with _retriever_lock:
        if _lightrag_retriever is not None:
            return _lightrag_retriever

        if vectorstore is None:
            from backend.services.vectorstore import get_vector_store
            vectorstore = get_vector_store()

        _lightrag_retriever = LightRAGRetriever(
            vectorstore=vectorstore,
            knowledge_graph_service=knowledge_graph_service,
            config=config,
        )

        return _lightrag_retriever


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LightRAGConfig",
    "LightRAGResult",
    "LightRAGRetriever",
    "RetrievalLevel",
    "EntityMatch",
    "ConceptMatch",
    "get_lightrag_retriever",
]
