"""
AIDocumentIndexer - Retrieval Strategies
=========================================

Advanced retrieval strategies for improved RAG quality.

Strategies:
- Standard: Direct chunk retrieval (current behavior)
- Hierarchical: Two-stage retrieval (documents first, then chunks)
- TwoStage: Fast ANN retrieval (stage 1) + precise reranking (stage 2)

Two-Stage Retrieval:
Industry-standard approach for scale. Stage 1 uses fast ANN (HNSW) to get
150-200 candidates in <50ms. Stage 2 uses ColBERT or cross-encoder reranking
for precise scoring.

Research shows two-stage retrieval can improve precision by 15-30% over
single-stage approaches.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import asyncio
import structlog

from backend.services.vectorstore import VectorStore, SearchResult, SearchType

logger = structlog.get_logger(__name__)


# Cached settings
_two_stage_enabled: Optional[bool] = None
_stage1_candidates: Optional[int] = None


async def _get_two_stage_settings() -> Tuple[bool, int]:
    """Get two-stage retrieval settings from database."""
    global _two_stage_enabled, _stage1_candidates

    if _two_stage_enabled is not None and _stage1_candidates is not None:
        return _two_stage_enabled, _stage1_candidates

    try:
        from backend.services.settings import get_settings_service

        settings = get_settings_service()
        enabled = await settings.get_setting("rag.two_stage_retrieval_enabled")
        candidates = await settings.get_setting("rag.stage1_candidates")

        _two_stage_enabled = enabled if enabled is not None else False
        _stage1_candidates = candidates if candidates else 150

        return _two_stage_enabled, _stage1_candidates
    except Exception as e:
        logger.debug("Could not load two-stage settings, using defaults", error=str(e))
        return False, 150


def invalidate_two_stage_settings():
    """Invalidate cached settings."""
    global _two_stage_enabled, _stage1_candidates
    _two_stage_enabled = None
    _stage1_candidates = None


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    STANDARD = "standard"        # Direct chunk retrieval
    HIERARCHICAL = "hierarchical"  # Document-first, then chunks
    TWO_STAGE = "two_stage"      # Fast ANN + precise reranking


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical retrieval."""
    doc_limit: int = 10           # Max documents to consider in stage 1
    chunks_per_doc: int = 3       # Chunks to retrieve per document in stage 2
    final_top_k: int = 10         # Final number of results after reranking
    min_doc_score: float = 0.3    # Minimum document-level score
    diversity_boost: float = 0.1  # Boost for results from different documents


class HierarchicalRetriever:
    """
    Two-stage hierarchical retrieval for better document diversity.

    Stage 1: Retrieve top documents based on aggregate chunk relevance
    Stage 2: Within top documents, get the best chunks
    Stage 3: Cross-document rerank for final results

    Benefits:
    - Ensures coverage across multiple relevant documents
    - Reduces redundancy from same-document chunks
    - Better for queries spanning multiple topics
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        config: Optional[HierarchicalConfig] = None,
    ):
        """
        Initialize hierarchical retriever.

        Args:
            vectorstore: Vector store service for searches
            config: Configuration for hierarchical retrieval
        """
        self.vectorstore = vectorstore
        self.config = config or HierarchicalConfig()

        logger.info(
            "Initialized HierarchicalRetriever",
            doc_limit=self.config.doc_limit,
            chunks_per_doc=self.config.chunks_per_doc,
        )

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 10,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform hierarchical retrieval.

        Args:
            query: Search query
            query_embedding: Pre-computed query embedding
            search_type: Type of search
            top_k: Final number of results
            access_tier_level: Access tier filter
            document_ids: Optional document filter
            vector_weight: Dynamic vector weight
            keyword_weight: Dynamic keyword weight

        Returns:
            List of SearchResult objects with diverse document coverage
        """
        logger.info(
            "Starting hierarchical retrieval",
            query_length=len(query),
            top_k=top_k,
        )

        # Stage 1: Get initial broad results to identify top documents
        # Fetch more candidates than usual to analyze document distribution
        broad_top_k = max(
            top_k * 3,
            self.config.doc_limit * self.config.chunks_per_doc,
        )

        initial_results = await self.vectorstore.search(
            query=query,
            query_embedding=query_embedding,
            search_type=search_type,
            top_k=broad_top_k,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        if not initial_results:
            logger.debug("No results from initial retrieval")
            return []

        # Stage 2: Rank documents by aggregate chunk scores
        doc_scores: Dict[str, float] = {}
        doc_chunks: Dict[str, List[SearchResult]] = {}

        for result in initial_results:
            doc_id = result.document_id

            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                doc_chunks[doc_id] = []

            # Aggregate scores with diminishing returns for additional chunks
            # First chunk counts full, subsequent chunks count less
            chunk_position = len(doc_chunks[doc_id])
            diminishing_factor = 1.0 / (1.0 + chunk_position * 0.3)
            doc_scores[doc_id] += result.score * diminishing_factor
            doc_chunks[doc_id].append(result)

        # Sort documents by aggregate score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Take top documents
        top_docs = sorted_docs[:self.config.doc_limit]

        logger.debug(
            "Stage 1 complete: identified top documents",
            total_docs=len(doc_scores),
            top_docs=len(top_docs),
            top_doc_scores=[s for _, s in top_docs[:3]],
        )

        # Stage 3: Select best chunks from top documents with diversity
        final_results: List[SearchResult] = []
        doc_count_in_results: Dict[str, int] = {}

        # Interleave results from different documents for diversity
        max_rounds = self.config.chunks_per_doc

        for round_idx in range(max_rounds):
            for doc_id, _ in top_docs:
                if len(final_results) >= top_k:
                    break

                chunks = doc_chunks.get(doc_id, [])
                if round_idx < len(chunks):
                    chunk = chunks[round_idx]

                    # Apply diversity boost - chunks from less-represented docs score higher
                    doc_count = doc_count_in_results.get(doc_id, 0)
                    if doc_count > 0:
                        # Slightly penalize repeated documents
                        chunk.score *= (1.0 - self.config.diversity_boost * doc_count)

                    final_results.append(chunk)
                    doc_count_in_results[doc_id] = doc_count + 1

            if len(final_results) >= top_k:
                break

        # Final sort by score
        final_results.sort(key=lambda r: r.score, reverse=True)
        final_results = final_results[:top_k]

        # Count unique documents in final results
        unique_docs = len(set(r.document_id for r in final_results))

        logger.info(
            "Hierarchical retrieval complete",
            total_results=len(final_results),
            unique_documents=unique_docs,
            avg_chunks_per_doc=len(final_results) / max(unique_docs, 1),
        )

        return final_results


@dataclass
class TwoStageConfig:
    """Configuration for two-stage retrieval."""
    stage1_candidates: int = 150  # Fast ANN retrieval candidates
    final_top_k: int = 10         # Final results after reranking
    use_colbert: bool = True      # Use ColBERT for stage 2 (else cross-encoder)
    use_hybrid_stage1: bool = True  # Use hybrid search in stage 1


class TwoStageRetriever:
    """
    Two-stage retrieval for production-scale RAG.

    Architecture:
    Stage 1: Fast ANN search using HNSW index
        - Retrieves 150-200 candidates in <50ms
        - Uses vector similarity or hybrid search
        - Optimized for recall (get all potentially relevant docs)

    Stage 2: Precise reranking
        - Uses ColBERT or cross-encoder for accurate scoring
        - 100ms-1s for 150 documents
        - Optimized for precision

    Benefits:
    - 5-10x faster than full cross-encoder search
    - 15-30% better precision than single-stage
    - Scales to millions of documents
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        config: Optional[TwoStageConfig] = None,
    ):
        """
        Initialize two-stage retriever.

        Args:
            vectorstore: Vector store service
            config: Two-stage configuration
        """
        self.vectorstore = vectorstore
        self.config = config or TwoStageConfig()
        self._reranker = None

        logger.info(
            "Initialized TwoStageRetriever",
            stage1_candidates=self.config.stage1_candidates,
            use_colbert=self.config.use_colbert,
        )

    def _get_reranker(self):
        """Lazy initialization of reranker."""
        if self._reranker is None:
            if self.config.use_colbert:
                try:
                    from backend.services.colbert_reranker import get_colbert_reranker
                    self._reranker = get_colbert_reranker()
                except Exception as e:
                    logger.warning("ColBERT reranker not available", error=str(e))
                    # Fall back to cross-encoder from vectorstore
                    self._reranker = None
        return self._reranker

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 10,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform two-stage retrieval.

        Args:
            query: Search query
            query_embedding: Pre-computed query embedding
            search_type: Type of search for stage 1
            top_k: Final number of results
            access_tier_level: Access tier filter
            document_ids: Optional document filter
            vector_weight: Vector weight for hybrid search
            keyword_weight: Keyword weight for hybrid search

        Returns:
            List of SearchResult objects
        """
        import time
        start_time = time.time()

        logger.info(
            "Starting two-stage retrieval",
            query_length=len(query),
            stage1_candidates=self.config.stage1_candidates,
            top_k=top_k,
        )

        # Stage 1: Fast ANN retrieval
        stage1_start = time.time()

        # Determine search type for stage 1
        stage1_search_type = search_type if self.config.use_hybrid_stage1 else SearchType.VECTOR

        stage1_results = await self.vectorstore.search(
            query=query,
            query_embedding=query_embedding,
            search_type=stage1_search_type,
            top_k=self.config.stage1_candidates,
            access_tier_level=access_tier_level,
            document_ids=document_ids,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        stage1_time = (time.time() - stage1_start) * 1000

        if not stage1_results:
            logger.debug("No results from stage 1 retrieval")
            return []

        logger.debug(
            "Stage 1 complete",
            candidates=len(stage1_results),
            time_ms=round(stage1_time, 2),
        )

        # Stage 2: Precise reranking
        stage2_start = time.time()

        reranker = self._get_reranker()

        if reranker is not None:
            # Use ColBERT reranking
            documents = [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                    "document_title": r.document_title,
                    "document_filename": r.document_filename,
                    "page_number": r.page_number,
                    "section_title": r.section_title,
                    "similarity_score": r.similarity_score,
                }
                for r in stage1_results
            ]

            reranked = await reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )

            # Convert back to SearchResult
            final_results = [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=r.rerank_score,
                    similarity_score=r.original_score,
                    metadata={**r.metadata, "two_stage": True, "reranked": True},
                    document_title=documents[i].get("document_title") if i < len(documents) else None,
                    document_filename=documents[i].get("document_filename") if i < len(documents) else None,
                    page_number=documents[i].get("page_number") if i < len(documents) else None,
                    section_title=documents[i].get("section_title") if i < len(documents) else None,
                )
                for i, r in enumerate(reranked[:top_k])
            ]
        else:
            # Fallback: use vectorstore's built-in reranking or just take top_k
            final_results = stage1_results[:top_k]
            for r in final_results:
                r.metadata["two_stage"] = True
                r.metadata["reranked"] = False

        stage2_time = (time.time() - stage2_start) * 1000
        total_time = (time.time() - start_time) * 1000

        logger.info(
            "Two-stage retrieval complete",
            stage1_time_ms=round(stage1_time, 2),
            stage2_time_ms=round(stage2_time, 2),
            total_time_ms=round(total_time, 2),
            final_results=len(final_results),
        )

        return final_results


class RetrieverFactory:
    """Factory for creating retrieval strategy instances."""

    @staticmethod
    def create(
        strategy: RetrievalStrategy,
        vectorstore: VectorStore,
        config: Optional[Dict[str, Any]] = None,
    ) -> Union[VectorStore, HierarchicalRetriever, TwoStageRetriever]:
        """
        Create a retriever for the specified strategy.

        Args:
            strategy: Retrieval strategy to use
            vectorstore: Vector store service
            config: Optional strategy-specific configuration

        Returns:
            Retriever instance
        """
        if strategy == RetrievalStrategy.HIERARCHICAL:
            hier_config = HierarchicalConfig(
                doc_limit=config.get("doc_limit", 10) if config else 10,
                chunks_per_doc=config.get("chunks_per_doc", 3) if config else 3,
                final_top_k=config.get("final_top_k", 10) if config else 10,
            )
            return HierarchicalRetriever(vectorstore, hier_config)

        if strategy == RetrievalStrategy.TWO_STAGE:
            two_stage_config = TwoStageConfig(
                stage1_candidates=config.get("stage1_candidates", 150) if config else 150,
                final_top_k=config.get("final_top_k", 10) if config else 10,
                use_colbert=config.get("use_colbert", True) if config else True,
                use_hybrid_stage1=config.get("use_hybrid_stage1", True) if config else True,
            )
            return TwoStageRetriever(vectorstore, two_stage_config)

        # Standard strategy just uses vectorstore directly
        return vectorstore


# Convenience function
def get_hierarchical_retriever(
    vectorstore: VectorStore,
    doc_limit: int = 10,
    chunks_per_doc: int = 3,
) -> HierarchicalRetriever:
    """
    Get a hierarchical retriever instance.

    Args:
        vectorstore: Vector store service
        doc_limit: Maximum documents in stage 1
        chunks_per_doc: Chunks per document in stage 2

    Returns:
        HierarchicalRetriever instance
    """
    config = HierarchicalConfig(
        doc_limit=doc_limit,
        chunks_per_doc=chunks_per_doc,
    )
    return HierarchicalRetriever(vectorstore, config)
