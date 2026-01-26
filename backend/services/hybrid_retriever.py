"""
AIDocumentIndexer - Hybrid Retriever
======================================

Unified hybrid retrieval combining Dense + Sparse + Graph + ColBERT.

This is the main retriever that orchestrates all retrieval strategies
and fuses results using Reciprocal Rank Fusion (RRF).

Retrieval Strategies Combined:
1. Dense Vector Search: Semantic similarity (OpenAI/local embeddings)
2. Sparse BM25 Search: Keyword/lexical matching
3. ColBERT PLAID: Late interaction for precise matching (optional)
4. Knowledge Graph: Entity-aware retrieval (optional)
5. Contextual: Context-enhanced embeddings (optional)

Fusion Methods:
- Reciprocal Rank Fusion (RRF): Robust, position-based fusion
- Weighted Linear: Score-based weighted combination
- Learned Fusion: ML-based optimal weighting (future)

Research:
- RRF: "Reciprocal Rank Fusion outperforms Condorcet and individual
  rank learning methods" (Cormack et al., 2009)
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings
from backend.core.performance import gather_with_concurrency

logger = structlog.get_logger(__name__)

# Phase 55: Import audit logging for fallback events
try:
    from backend.services.audit import audit_service_fallback, audit_service_error
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

class FusionMethod(str, Enum):
    """Methods for fusing multiple result sets."""
    RRF = "rrf"                    # Reciprocal Rank Fusion
    WEIGHTED = "weighted"          # Weighted linear combination
    INTERLEAVED = "interleaved"    # Round-robin interleaving
    CASCADED = "cascaded"          # Stage-wise filtering


class RetrievalSource(str, Enum):
    """Sources of retrieval results."""
    DENSE = "dense"           # Dense vector search
    SPARSE = "sparse"         # BM25/keyword search
    COLBERT = "colbert"       # ColBERT PLAID
    GRAPH = "graph"           # Knowledge graph
    CONTEXTUAL = "contextual" # Contextual embeddings
    WARP = "warp"             # WARP engine (3x faster than ColBERT)
    COLPALI = "colpali"       # ColPali visual document retrieval
    LIGHTRAG = "lightrag"     # LightRAG dual-level retrieval (Phase 58)
    RAPTOR = "raptor"         # RAPTOR tree-organized retrieval (Phase 58)


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    # Enable/disable sources
    enable_dense: bool = True
    enable_sparse: bool = True
    enable_colbert: bool = False
    enable_graph: bool = False
    enable_contextual: bool = False
    enable_warp: bool = False      # WARP engine (Phase 51)
    enable_colpali: bool = False   # ColPali visual retrieval (Phase 51)
    enable_lightrag: bool = False  # LightRAG dual-level retrieval (Phase 58)
    enable_raptor: bool = False    # RAPTOR tree-organized retrieval (Phase 58)

    # Source weights (for weighted fusion)
    dense_weight: float = 0.4
    sparse_weight: float = 0.3
    colbert_weight: float = 0.2
    graph_weight: float = 0.1
    contextual_weight: float = 0.0
    warp_weight: float = 0.0
    colpali_weight: float = 0.0
    lightrag_weight: float = 0.2   # Higher weight - effective for entity queries
    raptor_weight: float = 0.15    # Moderate weight - good for hierarchical docs

    # RRF settings
    rrf_k: int = 60  # Dampening constant

    # Retrieval settings
    candidates_per_source: int = 50  # Candidates from each source
    final_top_k: int = 10
    fusion_method: FusionMethod = FusionMethod.RRF

    # Reranking
    enable_reranking: bool = True
    rerank_top_k: int = 20  # Candidates to rerank

    # Performance
    max_concurrent: int = 5
    timeout_seconds: float = 10.0


@dataclass(slots=True)
class HybridResult:
    """Result from hybrid retrieval."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    sources: List[RetrievalSource] = field(default_factory=list)
    source_scores: Dict[str, float] = field(default_factory=dict)
    source_ranks: Dict[str, int] = field(default_factory=dict)
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Document info
    document_title: Optional[str] = None
    document_filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


@dataclass
class RetrievalMetrics:
    """Metrics from hybrid retrieval."""
    total_time_ms: float
    source_times: Dict[str, float]
    fusion_time_ms: float
    rerank_time_ms: float
    candidates_per_source: Dict[str, int]
    final_count: int
    sources_used: List[str]


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================

def reciprocal_rank_fusion(
    result_lists: List[Tuple[RetrievalSource, List[Any]]],
    k: int = 60,
    id_extractor=lambda x: x.chunk_id if hasattr(x, 'chunk_id') else x.get('chunk_id'),
    score_extractor=lambda x: x.score if hasattr(x, 'score') else x.get('score', 0.0),
    weights: Optional[Dict[RetrievalSource, float]] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Reciprocal Rank Fusion algorithm.

    RRF score = sum(weight[source] / (k + rank)) for each result list

    Benefits:
    - Robust to score scale differences between sources
    - No need for score normalization
    - Works well with incomplete rankings
    - Simple and effective

    Args:
        result_lists: List of (source, results) tuples
        k: Dampening constant (default 60, from original paper)
        id_extractor: Function to extract ID from result
        score_extractor: Function to extract score from result
        weights: Optional source weights

    Returns:
        List of (id, rrf_score, metadata) tuples sorted by score
    """
    rrf_scores: Dict[str, float] = {}
    result_info: Dict[str, Dict[str, Any]] = {}
    source_ranks: Dict[str, Dict[str, int]] = {}
    source_scores: Dict[str, Dict[str, float]] = {}

    default_weight = 1.0 / len(result_lists) if result_lists else 1.0

    for source, results in result_lists:
        source_name = source.value if isinstance(source, RetrievalSource) else str(source)
        weight = (weights or {}).get(source, default_weight)

        for rank, result in enumerate(results):
            item_id = id_extractor(result)
            if not item_id:
                continue

            # Calculate RRF contribution
            rrf_contribution = weight / (k + rank + 1)
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + rrf_contribution

            # Track source information
            if item_id not in source_ranks:
                source_ranks[item_id] = {}
                source_scores[item_id] = {}

            source_ranks[item_id][source_name] = rank + 1
            source_scores[item_id][source_name] = score_extractor(result)

            # Store result info (from first occurrence)
            if item_id not in result_info:
                result_info[item_id] = {
                    "result": result,
                    "sources": [source],
                }
            else:
                result_info[item_id]["sources"].append(source)

    # Sort by RRF score
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Build output with metadata
    output = []
    for item_id, rrf_score in sorted_results:
        info = result_info.get(item_id, {})
        metadata = {
            "rrf_score": rrf_score,
            "sources": [s.value if isinstance(s, RetrievalSource) else s
                       for s in info.get("sources", [])],
            "source_ranks": source_ranks.get(item_id, {}),
            "source_scores": source_scores.get(item_id, {}),
            "result": info.get("result"),
        }
        output.append((item_id, rrf_score, metadata))

    return output


# =============================================================================
# Hybrid Retriever
# =============================================================================

class HybridRetriever:
    """
    Unified hybrid retriever combining multiple retrieval strategies.

    Orchestrates:
    - Dense vector search (semantic)
    - Sparse BM25 search (lexical)
    - ColBERT PLAID (late interaction)
    - Knowledge graph (entity-aware)
    - Contextual embeddings (context-enhanced)

    All results are fused using RRF for robust combination.

    Usage:
        retriever = HybridRetriever(
            vectorstore=vectorstore,
            config=HybridConfig(enable_colbert=True),
        )

        results = await retriever.retrieve(
            query="What is Apple's revenue?",
            top_k=10,
        )
    """

    def __init__(
        self,
        vectorstore,
        colbert_retriever=None,
        knowledge_graph_service=None,
        contextual_service=None,
        reranker=None,
        warp_retriever=None,
        colpali_retriever=None,
        lightrag_retriever=None,
        raptor_retriever=None,
        config: Optional[HybridConfig] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vectorstore: Main vector store (supports dense + sparse)
            colbert_retriever: Optional ColBERT PLAID retriever
            knowledge_graph_service: Optional KG service
            contextual_service: Optional contextual embedding service
            reranker: Optional reranker for final stage
            warp_retriever: Optional WARP retriever (3x faster than ColBERT)
            colpali_retriever: Optional ColPali retriever (visual documents)
            lightrag_retriever: Optional LightRAG retriever (dual-level)
            raptor_retriever: Optional RAPTOR retriever (tree-organized)
            config: Configuration options
        """
        self.vectorstore = vectorstore
        self.colbert_retriever = colbert_retriever
        self.kg_service = knowledge_graph_service
        self.contextual_service = contextual_service
        self.reranker = reranker
        self.warp_retriever = warp_retriever
        self.colpali_retriever = colpali_retriever
        self.lightrag_retriever = lightrag_retriever
        self.raptor_retriever = raptor_retriever
        self.config = config or HybridConfig()

        # Apply config settings
        self._update_config_from_settings()

        self._initialized = False

        logger.info(
            "Initialized HybridRetriever",
            sources=self._get_enabled_sources(),
            fusion=self.config.fusion_method.value,
        )

    def _update_config_from_settings(self):
        """Update config from global settings."""
        if settings.ENABLE_COLBERT_RETRIEVAL:
            self.config.enable_colbert = True
        if settings.ENABLE_CONTEXTUAL_RETRIEVAL:
            self.config.enable_contextual = True
        # Phase 51: WARP and ColPali settings
        if getattr(settings, 'ENABLE_WARP', False):
            self.config.enable_warp = True
            self.config.warp_weight = 0.25  # Higher weight - WARP is high quality
        if getattr(settings, 'ENABLE_COLPALI', False):
            self.config.enable_colpali = True
            self.config.colpali_weight = 0.15  # Lower weight - specialized for visual
        # Phase 58: LightRAG and RAPTOR settings
        if getattr(settings, 'ENABLE_LIGHTRAG', False):
            self.config.enable_lightrag = True
            self.config.lightrag_weight = 0.2  # Good for entity-based queries
        if getattr(settings, 'ENABLE_RAPTOR', False):
            self.config.enable_raptor = True
            self.config.raptor_weight = 0.15  # Good for hierarchical docs

    def _get_enabled_sources(self) -> List[str]:
        """Get list of enabled retrieval sources."""
        sources = []
        if self.config.enable_dense:
            sources.append("dense")
        if self.config.enable_sparse:
            sources.append("sparse")
        if self.config.enable_colbert and self.colbert_retriever:
            sources.append("colbert")
        if self.config.enable_graph and self.kg_service:
            sources.append("graph")
        if self.config.enable_contextual and self.contextual_service:
            sources.append("contextual")
        # Phase 51: Add WARP and ColPali
        if self.config.enable_warp and self.warp_retriever:
            sources.append("warp")
        if self.config.enable_colpali and self.colpali_retriever:
            sources.append("colpali")
        # Phase 58: Add LightRAG and RAPTOR
        if self.config.enable_lightrag and self.lightrag_retriever:
            sources.append("lightrag")
        if self.config.enable_raptor and self.raptor_retriever:
            sources.append("raptor")
        return sources

    async def initialize(self) -> bool:
        """Initialize all retrieval components."""
        if self._initialized:
            return True

        try:
            # Initialize ColBERT if enabled
            if self.config.enable_colbert and self.colbert_retriever:
                await self.colbert_retriever.initialize()

            # Initialize contextual service if enabled
            if self.config.enable_contextual and self.contextual_service:
                await self.contextual_service.initialize()

            # Phase 51: Initialize WARP if enabled
            if self.config.enable_warp and self.warp_retriever:
                if hasattr(self.warp_retriever, 'initialize'):
                    await self.warp_retriever.initialize()
                logger.info("WARP retriever initialized")

            # Phase 51: Initialize ColPali if enabled
            if self.config.enable_colpali and self.colpali_retriever:
                if hasattr(self.colpali_retriever, 'initialize'):
                    await self.colpali_retriever.initialize()
                logger.info("ColPali retriever initialized")

            # Phase 58: Initialize LightRAG if enabled
            if self.config.enable_lightrag and self.lightrag_retriever:
                if hasattr(self.lightrag_retriever, 'initialize'):
                    await self.lightrag_retriever.initialize()
                logger.info("LightRAG retriever initialized")

            # Phase 58: Initialize RAPTOR if enabled
            if self.config.enable_raptor and self.raptor_retriever:
                if hasattr(self.raptor_retriever, 'initialize'):
                    await self.raptor_retriever.initialize()
                logger.info("RAPTOR retriever initialized")

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize hybrid retriever", error=str(e))
            return False

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        access_tier_level: int = 100,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> Tuple[List[HybridResult], RetrievalMetrics]:
        """
        Perform hybrid retrieval across all enabled sources.

        Args:
            query: Search query
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            document_ids: Filter to specific documents
            access_tier_level: Access tier filter
            vector_weight: Override dense weight
            keyword_weight: Override sparse weight

        Returns:
            Tuple of (results, metrics)
        """
        await self.initialize()

        start_time = time.time()
        final_top_k = top_k or self.config.final_top_k

        logger.info(
            "Starting hybrid retrieval",
            query_length=len(query),
            sources=self._get_enabled_sources(),
            top_k=final_top_k,
        )

        # Build retrieval tasks
        tasks = []
        task_sources = []
        source_times = {}

        # Dense vector search
        if self.config.enable_dense:
            tasks.append(self._dense_search(
                query, query_embedding,
                self.config.candidates_per_source,
                document_ids, access_tier_level,
            ))
            task_sources.append(RetrievalSource.DENSE)

        # Sparse BM25 search
        if self.config.enable_sparse:
            tasks.append(self._sparse_search(
                query,
                self.config.candidates_per_source,
                document_ids, access_tier_level,
            ))
            task_sources.append(RetrievalSource.SPARSE)

        # ColBERT search
        if self.config.enable_colbert and self.colbert_retriever:
            tasks.append(self._colbert_search(
                query,
                self.config.candidates_per_source,
                document_ids,
            ))
            task_sources.append(RetrievalSource.COLBERT)

        # Graph search
        if self.config.enable_graph and self.kg_service:
            tasks.append(self._graph_search(
                query,
                self.config.candidates_per_source,
                access_tier_level,
            ))
            task_sources.append(RetrievalSource.GRAPH)

        # Phase 51: WARP search (3x faster than ColBERT)
        if self.config.enable_warp and self.warp_retriever:
            tasks.append(self._warp_search(
                query,
                self.config.candidates_per_source,
                document_ids,
            ))
            task_sources.append(RetrievalSource.WARP)

        # Phase 51: ColPali visual document search
        if self.config.enable_colpali and self.colpali_retriever:
            tasks.append(self._colpali_search(
                query,
                self.config.candidates_per_source,
                document_ids,
            ))
            task_sources.append(RetrievalSource.COLPALI)

        # Phase 58: LightRAG dual-level retrieval
        if self.config.enable_lightrag and self.lightrag_retriever:
            tasks.append(self._lightrag_search(
                query,
                self.config.candidates_per_source,
                document_ids,
            ))
            task_sources.append(RetrievalSource.LIGHTRAG)

        # Phase 58: RAPTOR tree-organized retrieval
        if self.config.enable_raptor and self.raptor_retriever:
            tasks.append(self._raptor_search(
                query,
                self.config.candidates_per_source,
                document_ids,
            ))
            task_sources.append(RetrievalSource.RAPTOR)

        # Execute all searches in parallel
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Hybrid retrieval timed out", timeout=self.config.timeout_seconds)
            results = [[] for _ in tasks]

        retrieval_time = (time.time() - start_time) * 1000

        # Process results
        result_lists = []
        candidates_per_source = {}

        for source, result in zip(task_sources, results):
            if isinstance(result, Exception):
                logger.warning(f"Retrieval from {source.value} failed", error=str(result))
                result = []

            result_lists.append((source, result))
            candidates_per_source[source.value] = len(result)
            source_times[source.value] = 0.0  # Individual times not tracked

        # Fuse results
        fusion_start = time.time()

        if self.config.fusion_method == FusionMethod.RRF:
            fused = self._rrf_fuse(result_lists, final_top_k * 2)
        elif self.config.fusion_method == FusionMethod.WEIGHTED:
            fused = self._weighted_fuse(result_lists, final_top_k * 2)
        else:
            fused = self._interleaved_fuse(result_lists, final_top_k * 2)

        fusion_time = (time.time() - fusion_start) * 1000

        # Optional reranking
        rerank_start = time.time()
        rerank_time = 0.0

        if self.config.enable_reranking and self.reranker and len(fused) > final_top_k:
            try:
                fused = await self._rerank(query, fused[:self.config.rerank_top_k])
                rerank_time = (time.time() - rerank_start) * 1000
            except Exception as e:
                logger.warning("Reranking failed", error=str(e))

        # Take final top_k
        final_results = fused[:final_top_k]

        total_time = (time.time() - start_time) * 1000

        metrics = RetrievalMetrics(
            total_time_ms=total_time,
            source_times=source_times,
            fusion_time_ms=fusion_time,
            rerank_time_ms=rerank_time,
            candidates_per_source=candidates_per_source,
            final_count=len(final_results),
            sources_used=[s.value for s in task_sources],
        )

        logger.info(
            "Hybrid retrieval complete",
            total_time_ms=round(total_time, 2),
            fusion_time_ms=round(fusion_time, 2),
            rerank_time_ms=round(rerank_time, 2),
            results=len(final_results),
        )

        return final_results, metrics

    async def _dense_search(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        top_k: int,
        document_ids: Optional[List[str]],
        access_tier_level: int,
    ) -> List[Any]:
        """Dense vector search."""
        from backend.services.vectorstore import SearchType

        return await self.vectorstore.search(
            query=query,
            query_embedding=query_embedding,
            search_type=SearchType.VECTOR,
            top_k=top_k,
            document_ids=document_ids,
            access_tier_level=access_tier_level,
        )

    async def _sparse_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
        access_tier_level: int,
    ) -> List[Any]:
        """Sparse BM25/keyword search."""
        from backend.services.vectorstore import SearchType

        return await self.vectorstore.search(
            query=query,
            search_type=SearchType.KEYWORD,
            top_k=top_k,
            document_ids=document_ids,
            access_tier_level=access_tier_level,
        )

    async def _colbert_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
    ) -> List[Any]:
        """ColBERT PLAID search."""
        if not self.colbert_retriever:
            return []

        return await self.colbert_retriever.search(
            query=query,
            top_k=top_k,
        )

    async def _graph_search(
        self,
        query: str,
        top_k: int,
        access_tier_level: int,
    ) -> List[Any]:
        """Knowledge graph search."""
        if not self.kg_service:
            return []

        try:
            from backend.db.database import async_session_context
            async with async_session_context() as db:
                results = await self.kg_service.search(
                    query=query,
                    db=db,
                    top_k=top_k,
                )
                return results
        except Exception as e:
            logger.debug("Graph search failed", error=str(e))
            return []

    async def _warp_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
    ) -> List[Any]:
        """
        WARP engine search (Phase 51).

        WARP provides 3x speedup over ColBERT PLAID with:
        - Dynamic similarity imputation with WARP SELECT
        - Implicit decompression avoiding costly vector reconstruction
        - 41x faster than XTR reference implementation
        """
        if not self.warp_retriever:
            return []

        try:
            results = await self.warp_retriever.search(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            )
            return results
        except Exception as e:
            logger.warning("WARP search failed, falling back", error=str(e))

            # Phase 55: Log fallback to audit system
            if AUDIT_AVAILABLE:
                try:
                    if self.colbert_retriever:
                        await audit_service_fallback(
                            service_type="retrieval",
                            primary_provider="warp",
                            fallback_provider="colbert",
                            error_message=str(e),
                            context={"query_length": len(query), "top_k": top_k},
                        )
                    else:
                        await audit_service_error(
                            service_type="retrieval",
                            provider="warp",
                            error_message=str(e),
                            context={"query_length": len(query), "top_k": top_k, "no_fallback": True},
                        )
                except Exception:
                    pass  # Don't let audit logging break retrieval

            # Fallback to ColBERT if available
            if self.colbert_retriever:
                try:
                    return await self._colbert_search(query, top_k, document_ids)
                except Exception:
                    pass
            return []

    async def _colpali_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
    ) -> List[Any]:
        """
        ColPali visual document search (Phase 51).

        ColPali provides visual document retrieval using:
        - Vision Language Model embeddings
        - ColBERT-style late interaction for visual features
        - Eliminates need for OCR in many cases
        - Best for charts, infographics, scanned documents
        """
        if not self.colpali_retriever:
            return []

        try:
            results = await self.colpali_retriever.search(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            )
            return results
        except Exception as e:
            logger.warning("ColPali search failed", error=str(e))

            # Phase 55: Log error to audit system (no fallback available for visual search)
            if AUDIT_AVAILABLE:
                try:
                    await audit_service_error(
                        service_type="retrieval",
                        provider="colpali",
                        error_message=str(e),
                        context={"query_length": len(query), "top_k": top_k, "type": "visual_search"},
                    )
                except Exception:
                    pass  # Don't let audit logging break retrieval

            return []

    async def _lightrag_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
    ) -> List[Any]:
        """
        LightRAG dual-level retrieval (Phase 58).

        LightRAG provides:
        - Low-level: Entity-specific chunk retrieval
        - High-level: Abstract/summary-based retrieval
        - 10x token reduction through selective context
        - Best for entity-based and factual queries
        """
        if not self.lightrag_retriever:
            return []

        try:
            results = await self.lightrag_retriever.retrieve(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            )
            return results
        except Exception as e:
            logger.warning("LightRAG search failed", error=str(e))

            # Phase 55: Log error to audit system
            if AUDIT_AVAILABLE:
                try:
                    await audit_service_error(
                        service_type="retrieval",
                        provider="lightrag",
                        error_message=str(e),
                        context={"query_length": len(query), "top_k": top_k},
                    )
                except Exception:
                    pass  # Don't let audit logging break retrieval

            return []

    async def _raptor_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
    ) -> List[Any]:
        """
        RAPTOR tree-organized retrieval (Phase 58).

        RAPTOR provides:
        - Hierarchical tree of document summaries
        - Top-down traversal for overview queries
        - Bottom-up for detailed queries
        - Best for multi-hop reasoning and hierarchical documents
        """
        if not self.raptor_retriever:
            return []

        try:
            results = await self.raptor_retriever.retrieve(
                query=query,
                top_k=top_k,
                document_id=document_ids[0] if document_ids else None,
            )
            return results
        except Exception as e:
            logger.warning("RAPTOR search failed", error=str(e))

            # Phase 55: Log error to audit system
            if AUDIT_AVAILABLE:
                try:
                    await audit_service_error(
                        service_type="retrieval",
                        provider="raptor",
                        error_message=str(e),
                        context={"query_length": len(query), "top_k": top_k},
                    )
                except Exception:
                    pass  # Don't let audit logging break retrieval

            return []

    def _rrf_fuse(
        self,
        result_lists: List[Tuple[RetrievalSource, List[Any]]],
        top_k: int,
    ) -> List[HybridResult]:
        """Fuse results using RRF."""
        weights = {
            RetrievalSource.DENSE: self.config.dense_weight,
            RetrievalSource.SPARSE: self.config.sparse_weight,
            RetrievalSource.COLBERT: self.config.colbert_weight,
            RetrievalSource.GRAPH: self.config.graph_weight,
            RetrievalSource.CONTEXTUAL: self.config.contextual_weight,
            # Phase 51: Add WARP and ColPali weights
            RetrievalSource.WARP: self.config.warp_weight,
            RetrievalSource.COLPALI: self.config.colpali_weight,
            # Phase 58: Add LightRAG and RAPTOR weights
            RetrievalSource.LIGHTRAG: self.config.lightrag_weight,
            RetrievalSource.RAPTOR: self.config.raptor_weight,
        }

        fused = reciprocal_rank_fusion(
            result_lists,
            k=self.config.rrf_k,
            weights=weights,
        )

        # Convert to HybridResult
        return [
            self._to_hybrid_result(item_id, score, metadata)
            for item_id, score, metadata in fused[:top_k]
        ]

    def _weighted_fuse(
        self,
        result_lists: List[Tuple[RetrievalSource, List[Any]]],
        top_k: int,
    ) -> List[HybridResult]:
        """Fuse results using weighted linear combination."""
        # Normalize scores per source
        normalized: Dict[str, Dict[str, float]] = {}
        result_map: Dict[str, Any] = {}
        source_info: Dict[str, List[RetrievalSource]] = {}

        for source, results in result_lists:
            if not results:
                continue

            scores = [r.score if hasattr(r, 'score') else r.get('score', 0.0)
                     for r in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            range_score = max_score - min_score if max_score != min_score else 1.0

            weight = getattr(self.config, f"{source.value}_weight", 0.25)

            for r in results:
                chunk_id = r.chunk_id if hasattr(r, 'chunk_id') else r.get('chunk_id')
                score = r.score if hasattr(r, 'score') else r.get('score', 0.0)

                if not chunk_id:
                    continue

                # Normalize score to [0, 1]
                norm_score = (score - min_score) / range_score if range_score else 1.0

                if chunk_id not in normalized:
                    normalized[chunk_id] = {}
                    source_info[chunk_id] = []

                normalized[chunk_id][source.value] = norm_score * weight
                source_info[chunk_id].append(source)

                if chunk_id not in result_map:
                    result_map[chunk_id] = r

        # Combine scores
        combined_scores = {
            chunk_id: sum(scores.values())
            for chunk_id, scores in normalized.items()
        }

        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            result = result_map.get(chunk_id)
            if result:
                results.append(self._to_hybrid_result(
                    chunk_id, score,
                    {
                        "result": result,
                        "sources": source_info.get(chunk_id, []),
                        "source_scores": normalized.get(chunk_id, {}),
                    }
                ))

        return results

    def _interleaved_fuse(
        self,
        result_lists: List[Tuple[RetrievalSource, List[Any]]],
        top_k: int,
    ) -> List[HybridResult]:
        """Round-robin interleaved fusion."""
        seen = set()
        results = []
        max_len = max(len(r) for _, r in result_lists) if result_lists else 0

        for i in range(max_len):
            for source, source_results in result_lists:
                if i >= len(source_results):
                    continue

                r = source_results[i]
                chunk_id = r.chunk_id if hasattr(r, 'chunk_id') else r.get('chunk_id')

                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    results.append(self._to_hybrid_result(
                        chunk_id,
                        1.0 / (i + 1),  # Score based on position
                        {"result": r, "sources": [source]},
                    ))

                    if len(results) >= top_k:
                        return results

        return results

    def _to_hybrid_result(
        self,
        chunk_id: str,
        score: float,
        metadata: Dict[str, Any],
    ) -> HybridResult:
        """Convert fusion result to HybridResult."""
        result = metadata.get("result")

        # Extract content and document info
        content = ""
        document_id = ""
        document_title = None
        document_filename = None
        page_number = None
        section_title = None

        if result:
            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict):
                content = result.get('content', '')

            if hasattr(result, 'document_id'):
                document_id = result.document_id
            elif isinstance(result, dict):
                document_id = result.get('document_id', '')

            if hasattr(result, 'document_title'):
                document_title = result.document_title
            if hasattr(result, 'document_filename'):
                document_filename = result.document_filename
            if hasattr(result, 'page_number'):
                page_number = result.page_number
            if hasattr(result, 'section_title'):
                section_title = result.section_title

        sources = [
            s if isinstance(s, RetrievalSource) else RetrievalSource(s)
            for s in metadata.get("sources", [])
        ]

        return HybridResult(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            score=score,
            sources=sources,
            source_scores=metadata.get("source_scores", {}),
            source_ranks=metadata.get("source_ranks", {}),
            metadata={"fusion_method": self.config.fusion_method.value},
            document_title=document_title,
            document_filename=document_filename,
            page_number=page_number,
            section_title=section_title,
        )

    async def _rerank(
        self,
        query: str,
        results: List[HybridResult],
    ) -> List[HybridResult]:
        """Rerank results using cross-encoder or ColBERT."""
        if not self.reranker:
            return results

        documents = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "content": r.content,
                "score": r.score,
            }
            for r in results
        ]

        reranked = await self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=len(results),
        )

        # Update scores
        rerank_map = {r.chunk_id: r.rerank_score for r in reranked}

        for result in results:
            if result.chunk_id in rerank_map:
                result.rerank_score = rerank_map[result.chunk_id]

        # Sort by rerank score
        results.sort(
            key=lambda r: r.rerank_score if r.rerank_score else r.score,
            reverse=True,
        )

        return results


# =============================================================================
# Factory and Singleton
# =============================================================================

_hybrid_retriever: Optional[HybridRetriever] = None
_retriever_lock = asyncio.Lock()


async def get_hybrid_retriever(
    vectorstore=None,
    colbert_retriever=None,
    knowledge_graph_service=None,
    contextual_service=None,
    reranker=None,
    warp_retriever=None,
    colpali_retriever=None,
    lightrag_retriever=None,
    raptor_retriever=None,
    config: Optional[HybridConfig] = None,
) -> HybridRetriever:
    """
    Get or create hybrid retriever singleton with all available retrievers.

    Phase 62: Auto-initializes all enabled retrievers with fallback handling.

    Args:
        vectorstore: Vector store (required on first call)
        colbert_retriever: Optional ColBERT retriever
        knowledge_graph_service: Optional KG service
        contextual_service: Optional contextual service
        reranker: Optional reranker
        warp_retriever: Optional WARP retriever (Phase 51)
        colpali_retriever: Optional ColPali retriever (Phase 51)
        lightrag_retriever: Optional LightRAG retriever (Phase 58)
        raptor_retriever: Optional RAPTOR retriever (Phase 58)
        config: Optional configuration

    Returns:
        HybridRetriever instance
    """
    global _hybrid_retriever

    if _hybrid_retriever is not None:
        return _hybrid_retriever

    async with _retriever_lock:
        if _hybrid_retriever is not None:
            return _hybrid_retriever

        if vectorstore is None:
            from backend.services.vectorstore import get_vector_store
            vectorstore = get_vector_store()

        # Phase 62: Auto-initialize ColBERT if enabled
        if colbert_retriever is None and getattr(settings, 'ENABLE_COLBERT_RETRIEVAL', False):
            try:
                from backend.services.colbert_retriever import ColBERTRetriever
                colbert_retriever = ColBERTRetriever()
                await colbert_retriever.initialize()
                logger.info("ColBERT retriever initialized")
            except ImportError:
                logger.debug("ColBERT retriever not available - ragatouille not installed")
            except Exception as e:
                logger.warning("Failed to initialize ColBERT retriever", error=str(e))

        # Phase 51: Auto-initialize WARP if enabled
        if warp_retriever is None and getattr(settings, 'ENABLE_WARP', False):
            try:
                from backend.services.warp_retriever import get_warp_retriever
                warp_retriever = await get_warp_retriever()
                logger.info("WARP retriever initialized")
            except ImportError:
                logger.debug("WARP retriever not available")
            except Exception as e:
                logger.warning("Failed to initialize WARP retriever", error=str(e))

        # Phase 51: Auto-initialize ColPali if enabled
        if colpali_retriever is None and getattr(settings, 'ENABLE_COLPALI', False):
            try:
                from backend.services.colpali_retriever import get_colpali_retriever
                colpali_retriever = await get_colpali_retriever()
                logger.info("ColPali retriever initialized")
            except ImportError:
                logger.debug("ColPali retriever not available - colpali-engine not installed")
            except Exception as e:
                logger.warning("Failed to initialize ColPali retriever", error=str(e))

        # Phase 62: Auto-initialize LightRAG if enabled
        if lightrag_retriever is None and getattr(settings, 'ENABLE_LIGHTRAG', False):
            try:
                from backend.services.lightrag_retriever import LightRAGRetriever
                from backend.services.knowledge_graph import get_kg_service
                kg_service = await get_kg_service()
                lightrag_retriever = LightRAGRetriever(vectorstore, kg_service)
                await lightrag_retriever.initialize()
                logger.info("LightRAG retriever initialized")
            except ImportError:
                logger.debug("LightRAG retriever not available")
            except Exception as e:
                logger.warning("Failed to initialize LightRAG retriever", error=str(e))

        # Phase 62: Auto-initialize RAPTOR if enabled
        if raptor_retriever is None and getattr(settings, 'ENABLE_RAPTOR', False):
            try:
                from backend.services.raptor_retriever import RAPTORRetriever
                raptor_retriever = RAPTORRetriever(vectorstore)
                await raptor_retriever.initialize()
                logger.info("RAPTOR retriever initialized")
            except ImportError:
                logger.debug("RAPTOR retriever not available")
            except Exception as e:
                logger.warning("Failed to initialize RAPTOR retriever", error=str(e))

        # Phase 62: Auto-initialize contextual embeddings if enabled
        if contextual_service is None and getattr(settings, 'ENABLE_CONTEXTUAL_EMBEDDINGS', False):
            try:
                from backend.services.contextual_embeddings import ContextualEmbeddingService
                contextual_service = ContextualEmbeddingService()
                await contextual_service.initialize()
                logger.info("Contextual embedding service initialized")
            except ImportError:
                logger.debug("Contextual embedding service not available")
            except Exception as e:
                logger.warning("Failed to initialize contextual service", error=str(e))

        # Phase 62: Auto-initialize tiered reranker if enabled
        if reranker is None and getattr(settings, 'ENABLE_TIERED_RERANKING', False):
            try:
                from backend.services.tiered_reranking import get_tiered_reranker
                reranker = await get_tiered_reranker()
                logger.info("Tiered reranker initialized")
            except ImportError:
                logger.debug("Tiered reranker not available")
            except Exception as e:
                logger.warning("Failed to initialize reranker", error=str(e))

        _hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            colbert_retriever=colbert_retriever,
            knowledge_graph_service=knowledge_graph_service,
            contextual_service=contextual_service,
            reranker=reranker,
            warp_retriever=warp_retriever,
            colpali_retriever=colpali_retriever,
            lightrag_retriever=lightrag_retriever,
            raptor_retriever=raptor_retriever,
            config=config,
        )

        return _hybrid_retriever


def create_hybrid_retriever(
    vectorstore,
    config: Optional[HybridConfig] = None,
    colbert_retriever=None,
    knowledge_graph_service=None,
    contextual_service=None,
    reranker=None,
    warp_retriever=None,
    colpali_retriever=None,
    lightrag_retriever=None,
    raptor_retriever=None,
) -> HybridRetriever:
    """
    Create a new hybrid retriever instance (not singleton).

    Args:
        vectorstore: Vector store service
        config: Configuration options
        colbert_retriever: Optional ColBERT retriever
        knowledge_graph_service: Optional KG service
        contextual_service: Optional contextual service
        reranker: Optional reranker
        warp_retriever: Optional WARP retriever (Phase 51)
        colpali_retriever: Optional ColPali retriever (Phase 51)
        lightrag_retriever: Optional LightRAG retriever (Phase 58)
        raptor_retriever: Optional RAPTOR retriever (Phase 58)

    Returns:
        New HybridRetriever instance
    """
    return HybridRetriever(
        vectorstore=vectorstore,
        colbert_retriever=colbert_retriever,
        knowledge_graph_service=knowledge_graph_service,
        contextual_service=contextual_service,
        reranker=reranker,
        warp_retriever=warp_retriever,
        colpali_retriever=colpali_retriever,
        lightrag_retriever=lightrag_retriever,
        raptor_retriever=raptor_retriever,
        config=config,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "HybridConfig",
    "FusionMethod",
    "RetrievalSource",
    # Results
    "HybridResult",
    "RetrievalMetrics",
    # Core
    "HybridRetriever",
    "reciprocal_rank_fusion",
    # Factory
    "get_hybrid_retriever",
    "create_hybrid_retriever",
]
