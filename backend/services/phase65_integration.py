"""
AIDocumentIndexer - Phase 65 Integration Module
================================================

Wires all Phase 65 services into the existing pipeline:
- Binary Quantization for 32x memory reduction
- GPU Acceleration for 8-20x search speedup
- Learning-to-Rank for search quality
- Spell Correction for query preprocessing
- Semantic Cache for query caching
- Streaming Citations for real-time citation matching

This module provides a unified interface for Phase 65 features
that can be easily integrated into the RAG pipeline.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)

# Import Phase 65 services
from backend.services.binary_quantization import (
    BinaryQuantizer,
    BinaryQuantizationConfig,
    BinarySearchResult,
    get_binary_quantizer,
)
from backend.services.gpu_acceleration import (
    GPUVectorSearch,
    GPUSearchConfig,
    GPUSearchResult,
    get_gpu_vector_search,
    check_gpu_availability,
)
from backend.services.learning_to_rank import (
    LTRRanker,
    LTRConfig,
    LTRFeatureExtractor,
    RankedResult,
    get_ltr_ranker,
)
from backend.services.spell_correction import (
    SpellCorrector,
    SpellCorrectionConfig,
    CorrectionResult,
    get_spell_corrector,
)
from backend.services.semantic_cache import (
    SemanticQueryCache,
    SemanticCacheConfig,
    CacheStats,
    get_semantic_cache,
)
from backend.services.streaming_citations import (
    StreamingCitationMatcher,
    CitationConfig,
    CitationStyle,
    EnrichedToken,
    StreamingCitationResult,
    get_citation_matcher,
    enrich_streaming_response,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Phase65Config:
    """Configuration for Phase 65 features."""
    # Binary Quantization
    enable_binary_quantization: bool = False
    binary_rerank_factor: int = 10
    binary_similarity_threshold: float = 0.8

    # GPU Acceleration
    enable_gpu_acceleration: bool = True  # Auto-fallback to CPU if unavailable
    gpu_index_type: str = "ivf"  # flat, ivf, hnsw
    gpu_use_float16: bool = True

    # Learning-to-Rank
    enable_ltr: bool = True
    ltr_min_training_samples: int = 100

    # Spell Correction
    enable_spell_correction: bool = True
    spell_max_edit_distance: int = 2

    # Semantic Cache
    # Disabled: stale cache entries return wrong answers when retrieval logic changes.
    # The cache stores FULL responses and doesn't invalidate when documents/models change.
    enable_semantic_cache: bool = False
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000

    # Streaming Citations
    enable_streaming_citations: bool = True
    citation_style: CitationStyle = CitationStyle.NUMBERED
    citation_min_confidence: float = 0.5


# =============================================================================
# Phase 65 Pipeline
# =============================================================================

class Phase65Pipeline:
    """
    Unified Phase 65 pipeline integrating all advanced features.

    Usage:
        pipeline = Phase65Pipeline()

        # Pre-retrieval: spell correction + cache check
        query, cache_hit = await pipeline.preprocess_query(query, embedding)
        if cache_hit:
            return cache_hit

        # Retrieval (use existing vectorstore or GPU-accelerated)
        results = await pipeline.search(query, embedding, top_k=10)

        # Post-retrieval: LTR reranking
        results = await pipeline.rerank(query, results)

        # Generation: streaming with citations
        async for token in pipeline.stream_with_citations(llm_stream, sources):
            yield token
    """

    def __init__(self, config: Optional[Phase65Config] = None):
        self.config = config or Phase65Config()

        # Initialize services lazily
        self._spell_corrector: Optional[SpellCorrector] = None
        self._semantic_cache: Optional[SemanticQueryCache] = None
        self._ltr_ranker: Optional[LTRRanker] = None
        self._gpu_search: Optional[GPUVectorSearch] = None
        self._binary_quantizer: Optional[BinaryQuantizer] = None

        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize all Phase 65 services.

        Returns:
            Dict with initialization status for each service
        """
        status = {}

        # Spell Correction
        if self.config.enable_spell_correction:
            try:
                self._spell_corrector = get_spell_corrector(
                    SpellCorrectionConfig(
                        max_edit_distance=self.config.spell_max_edit_distance,
                    )
                )
                status["spell_correction"] = "enabled"
            except Exception as e:
                logger.warning("Spell correction init failed", error=str(e))
                status["spell_correction"] = f"failed: {e}"

        # Semantic Cache
        if self.config.enable_semantic_cache:
            try:
                self._semantic_cache = get_semantic_cache(
                    SemanticCacheConfig(
                        similarity_threshold=self.config.cache_similarity_threshold,
                        default_ttl_seconds=self.config.cache_ttl_seconds,
                        max_entries=self.config.cache_max_entries,
                    )
                )
                status["semantic_cache"] = "enabled"
            except Exception as e:
                logger.warning("Semantic cache init failed", error=str(e))
                status["semantic_cache"] = f"failed: {e}"

        # Learning-to-Rank
        if self.config.enable_ltr:
            try:
                self._ltr_ranker = get_ltr_ranker(
                    LTRConfig(
                        min_training_samples=self.config.ltr_min_training_samples,
                    )
                )
                status["ltr"] = "enabled"
                status["ltr_trained"] = self._ltr_ranker._is_trained
            except Exception as e:
                logger.warning("LTR init failed", error=str(e))
                status["ltr"] = f"failed: {e}"

        # GPU Acceleration
        if self.config.enable_gpu_acceleration:
            try:
                gpu_info = check_gpu_availability()
                if gpu_info.get("faiss_available"):
                    self._gpu_search = get_gpu_vector_search(
                        GPUSearchConfig(
                            use_float16=self.config.gpu_use_float16,
                        )
                    )
                    status["gpu_acceleration"] = f"enabled ({gpu_info.get('recommended_backend')})"
                else:
                    status["gpu_acceleration"] = "unavailable (faiss not installed)"
            except Exception as e:
                logger.warning("GPU acceleration init failed", error=str(e))
                status["gpu_acceleration"] = f"failed: {e}"

        # Binary Quantization
        if self.config.enable_binary_quantization:
            try:
                self._binary_quantizer = get_binary_quantizer(
                    BinaryQuantizationConfig(
                        rerank_factor=self.config.binary_rerank_factor,
                    )
                )
                status["binary_quantization"] = "enabled"
            except Exception as e:
                logger.warning("Binary quantization init failed", error=str(e))
                status["binary_quantization"] = f"failed: {e}"

        self._initialized = True
        logger.info("Phase 65 pipeline initialized", **status)

        return status

    # =========================================================================
    # Pre-Retrieval: Query Preprocessing
    # =========================================================================

    async def preprocess_query(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        model_key: Optional[str] = None,
    ) -> Tuple[str, Optional[Any]]:
        """
        Preprocess query with spell correction and cache check.

        Args:
            query: Original query string
            embedding: Optional query embedding for semantic cache
            model_key: Optional model+intelligence key for cache namespacing.
                       Different models/intelligence levels get separate cache entries.

        Returns:
            Tuple of (processed_query, cached_result_or_None)
        """
        processed_query = query

        # Spell correction
        if self._spell_corrector and self.config.enable_spell_correction:
            correction = await self._spell_corrector.correct(query)
            if correction.is_corrected:
                logger.debug(
                    "Query spell-corrected",
                    original=query,
                    corrected=correction.corrected,
                    corrections=correction.corrections,
                )
                processed_query = correction.corrected

        # Semantic cache check — namespace by model to avoid cross-model cache hits
        cached_result = None
        if self._semantic_cache and self.config.enable_semantic_cache:
            cache_query = f"{model_key}::{processed_query}" if model_key else processed_query
            cached_result = await self._semantic_cache.get(cache_query, embedding)
            if cached_result:
                logger.debug("Semantic cache hit", query=processed_query[:50], model_key=model_key)

        return processed_query, cached_result

    async def cache_result(
        self,
        query: str,
        result: Any,
        embedding: Optional[List[float]] = None,
        model_key: Optional[str] = None,
    ) -> None:
        """Cache a query result for future semantic matching."""
        if self._semantic_cache and self.config.enable_semantic_cache:
            # Namespace by model to prevent cross-model cache pollution
            cache_query = f"{model_key}::{query}" if model_key else query
            await self._semantic_cache.set(
                cache_query,
                result,
                embedding,
                self.config.cache_ttl_seconds,
            )

    # =========================================================================
    # Retrieval: GPU-Accelerated Search
    # =========================================================================

    async def build_gpu_index(
        self,
        embeddings: List[List[float]],
    ) -> Dict[str, Any]:
        """
        Build GPU index for fast search.

        Args:
            embeddings: Corpus embeddings

        Returns:
            Index statistics
        """
        if not self._gpu_search:
            return {"status": "gpu_search_not_enabled"}

        return await self._gpu_search.build_index(embeddings)

    async def gpu_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[GPUSearchResult]:
        """
        Perform GPU-accelerated vector search.

        Args:
            query_embedding: Query embedding
            top_k: Number of results

        Returns:
            Search results
        """
        if not self._gpu_search:
            raise ValueError("GPU search not initialized")

        return await self._gpu_search.search(query_embedding, top_k)

    async def binary_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[BinarySearchResult]:
        """
        Perform binary quantized search (32x memory reduction).

        Args:
            query_embedding: Query embedding
            top_k: Number of results

        Returns:
            Search results
        """
        if not self._binary_quantizer:
            raise ValueError("Binary quantizer not initialized")

        return await self._binary_quantizer.search_binary(
            query_embedding,
            top_k=top_k,
        )

    # =========================================================================
    # Post-Retrieval: Learning-to-Rank
    # =========================================================================

    async def rerank_with_ltr(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[RankedResult]:
        """
        Rerank candidates using Learning-to-Rank model.

        Args:
            query: Search query
            candidates: List of candidate results with metadata

        Returns:
            Reranked results with LTR scores
        """
        if not self._ltr_ranker or not self.config.enable_ltr:
            # Return candidates unchanged with default ranking
            return [
                RankedResult(
                    doc_id=c.get("chunk_id", str(i)),
                    original_score=c.get("score", 0.0),
                    ltr_score=c.get("score", 0.0),
                    features={},
                    rank=i,
                )
                for i, c in enumerate(candidates)
            ]

        return await self._ltr_ranker.rerank(query, candidates)

    async def record_click_feedback(
        self,
        query: str,
        doc_id: str,
        rank: int,
        clicked: bool = True,
        dwell_time_seconds: float = 0.0,
    ) -> None:
        """
        Record user click feedback for LTR training.

        Args:
            query: Search query
            doc_id: Clicked document ID
            rank: Original rank position
            clicked: Whether clicked
            dwell_time_seconds: Time spent on result
        """
        if self._ltr_ranker and self.config.enable_ltr:
            await self._ltr_ranker.record_feedback(
                query=query,
                doc_id=doc_id,
                rank=rank,
                clicked=clicked,
                dwell_time_seconds=dwell_time_seconds,
            )

    async def train_ltr_model(self, force: bool = False) -> Dict[str, Any]:
        """Train LTR model on collected feedback."""
        if not self._ltr_ranker:
            return {"status": "ltr_not_enabled"}

        return await self._ltr_ranker.train(force=force)

    # =========================================================================
    # Generation: Streaming with Citations
    # =========================================================================

    async def stream_with_citations(
        self,
        token_stream: AsyncGenerator[str, None],
        sources: List[Dict[str, Any]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Wrap LLM token stream with real-time citation matching.

        Args:
            token_stream: Async generator of tokens from LLM
            sources: Source documents used for retrieval

        Yields:
            Enriched tokens with citation information
        """
        if not self.config.enable_streaming_citations:
            # Pass through without citation matching
            async for token in token_stream:
                yield {"token": token, "citations": [], "is_complete": False}
            yield {"token": "", "citations": [], "is_complete": True}
            return

        config = CitationConfig(
            style=self.config.citation_style,
            min_confidence=self.config.citation_min_confidence,
        )

        async for chunk in enrich_streaming_response(
            token_stream,
            sources,
            include_footer=True,
            citation_style=self.config.citation_style,
        ):
            yield chunk

    # =========================================================================
    # Statistics & Monitoring
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all Phase 65 services."""
        stats = {
            "initialized": self._initialized,
            "config": {
                "binary_quantization": self.config.enable_binary_quantization,
                "gpu_acceleration": self.config.enable_gpu_acceleration,
                "ltr": self.config.enable_ltr,
                "spell_correction": self.config.enable_spell_correction,
                "semantic_cache": self.config.enable_semantic_cache,
                "streaming_citations": self.config.enable_streaming_citations,
            },
        }

        if self._semantic_cache:
            cache_stats = self._semantic_cache.get_stats()
            stats["semantic_cache"] = {
                "entries": cache_stats.total_entries,
                "hit_rate": cache_stats.hit_rate,
                "semantic_hits": cache_stats.semantic_hits,
            }

        if self._ltr_ranker:
            ltr_stats = self._ltr_ranker.get_stats()
            stats["ltr"] = {
                "trained": ltr_stats["is_trained"],
                "feedback_samples": ltr_stats["n_feedback_samples"],
            }

        if self._spell_corrector:
            spell_stats = self._spell_corrector.get_stats()
            stats["spell_correction"] = {
                "vocab_size": spell_stats["vocabulary_size"],
            }

        if self._gpu_search:
            gpu_stats = self._gpu_search.get_stats()
            stats["gpu_search"] = {
                "backend": gpu_stats["backend"],
                "index_size": gpu_stats["index_size"],
            }

        return stats


# =============================================================================
# Singleton
# =============================================================================

_phase65_pipeline: Optional[Phase65Pipeline] = None


async def get_phase65_pipeline(
    config: Optional[Phase65Config] = None,
) -> Phase65Pipeline:
    """Get or create Phase 65 pipeline singleton."""
    global _phase65_pipeline

    if _phase65_pipeline is None or config is not None:
        _phase65_pipeline = Phase65Pipeline(config)
        await _phase65_pipeline.initialize()

    return _phase65_pipeline


def get_phase65_pipeline_sync(
    config: Optional[Phase65Config] = None,
) -> Phase65Pipeline:
    """Get Phase 65 pipeline synchronously (without initialization)."""
    global _phase65_pipeline

    if _phase65_pipeline is None or config is not None:
        _phase65_pipeline = Phase65Pipeline(config)

    return _phase65_pipeline


# =============================================================================
# Convenience Functions for RAG Integration
# =============================================================================

async def preprocess_query(
    query: str,
    embedding: Optional[List[float]] = None,
) -> Tuple[str, Optional[Any]]:
    """Convenience function for query preprocessing."""
    pipeline = await get_phase65_pipeline()
    return await pipeline.preprocess_query(query, embedding)


async def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[RankedResult]:
    """Convenience function for LTR reranking."""
    pipeline = await get_phase65_pipeline()
    return await pipeline.rerank_with_ltr(query, candidates)


async def cache_query_result(
    query: str,
    result: Any,
    embedding: Optional[List[float]] = None,
) -> None:
    """Convenience function for caching results."""
    pipeline = await get_phase65_pipeline()
    await pipeline.cache_result(query, result, embedding)
