"""
AIDocumentIndexer - RAG Service
================================

Retrieval-Augmented Generation service using LangChain.
Provides hybrid search (vector + keyword) and conversational RAG chains.

NOTE: This module is being refactored into backend/services/rag_module/
Data models, config, and prompts have been extracted. RAGService remains here
for now during incremental migration.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import hashlib
import io
import json
import os
import traceback
import structlog

# Token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangChain chains (correct import paths for v0.3+)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangChain memory (Phase 87 audit: still valid in 0.3.x;
# future migration to langgraph state management recommended)
from langchain.memory import ConversationBufferWindowMemory

# Vector store and retrieval
from langchain_community.vectorstores import PGVector, FAISS

from backend.services.llm import (
    LLMFactory,
    LLMConfig,
    EnhancedLLMFactory,
    LLMConfigManager,
    LLMConfigResult,
    LLMUsageTracker,
)
from backend.services.embeddings import EmbeddingService
from backend.services.vectorstore import VectorStore, get_vector_store, SearchResult, SearchType
from backend.services.session_memory import get_session_memory_manager
from backend.db.database import async_session_context
from backend.db.models import Document as DBDocument
from sqlalchemy import select, or_
from backend.services.query_expander import (
    QueryExpander,
    QueryExpansionConfig,
    get_query_expander,
)
from backend.services.rag_verifier import (
    RAGVerifier,
    VerifierConfig,
    VerificationLevel,
    VerificationResult,
)
from backend.services.query_classifier import (
    QueryClassifier,
    QueryClassification,
    QueryIntent,
    get_query_classifier,
)
from backend.services.hyde import HyDEExpander, get_hyde_expander
from backend.services.corrective_rag import CorrectiveRAG, get_corrective_rag, CRAGResult
from backend.services.retrieval_strategies import (
    HierarchicalRetriever,
    HierarchicalConfig,
    get_hierarchical_retriever,
    TwoStageRetriever,
    TwoStageConfig,
)
from backend.services.smart_filter import (
    SmartDocumentFilter,
    SmartFilterConfig,
    get_smart_filter,
    FilterResult,
)
from backend.services.self_rag import (
    SelfRAG,
    SelfRAGResult,
    get_self_rag,
)
# Phase 66: Adaptive Router for intelligent query routing
from backend.services.adaptive_router import (
    AdaptiveRouter,
    RoutingDecision,
    RetrievalStrategy,
    QueryComplexity,
    get_adaptive_router,
)
# Phase 66: Advanced RAG utilities (RAG-Fusion, context compression, etc.)
from backend.services.advanced_rag_utils import (
    RAGFusion,
    ContextCompressor,
    ContextReorderer,
    StepBackPrompter,
    apply_rag_fusion,
    compress_context,
    reorder_for_attention,
    apply_stepback_prompting,
    FusionResult,
    CompressedContext,
    StepBackResult,
)
# Phase 66: LazyGraphRAG for cost-efficient knowledge graph retrieval (99% cost reduction)
from backend.services.lazy_graphrag import (
    LazyGraphRAGService,
    LazyGraphRAGConfig,
    LazyGraphContext,
    get_lazy_graphrag_service,
    lazy_graph_retrieve,
)
# Phase 58: HybridRetriever for LightRAG + RAPTOR + WARP + ColPali fusion
from backend.services.hybrid_retriever import (
    HybridRetriever,
    HybridConfig,
    HybridResult,
    get_hybrid_retriever,
)
# Phase 59: Multilingual cross-lingual search
try:
    from backend.services.multilingual_search import (
        CrossLingualRetriever,
        MultilingualEmbeddingService,
        MultilingualSearchConfig,
        Language,
        get_cross_lingual_retriever,
    )
    MULTILINGUAL_SEARCH_AVAILABLE = True
except ImportError:
    MULTILINGUAL_SEARCH_AVAILABLE = False
    CrossLingualRetriever = None
    Language = None

# Phase 65: Advanced enhancements (spell correction, semantic cache, LTR, streaming citations)
try:
    from backend.services.phase65_integration import (
        Phase65Pipeline,
        Phase65Config,
        get_phase65_pipeline,
        get_phase65_pipeline_sync,
        preprocess_query as phase65_preprocess_query,
        rerank_results as phase65_rerank_results,
        cache_query_result as phase65_cache_result,
    )
    PHASE65_AVAILABLE = True
except ImportError:
    PHASE65_AVAILABLE = False
    Phase65Pipeline = None
    Phase65Config = None

# Note: KnowledgeGraphService is imported lazily inside _enhance_with_knowledge_graph
# to avoid circular imports (knowledge_graph -> llm -> services.__init__ -> rag)

# Import from new modular structure
from backend.services.rag_module.models import Source, RAGResponse, StreamChunk
from backend.services.rag_module.config import RAGConfig
from backend.services.rag_module.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    CONVERSATIONAL_RAG_TEMPLATE,
    LANGUAGE_NAMES,
    get_language_instruction as _get_language_instruction,
    parse_suggested_questions as _parse_suggested_questions,
    # Phase 14: Enhanced prompt selection
    get_template_for_intent as _get_template_for_intent,
    get_system_prompt_for_model as _get_system_prompt_for_model,
    get_template_for_model as _get_template_for_model,
    is_tiny_model as _is_tiny_model,
    is_llama_model as _is_llama_model,
    is_llama_small as _is_llama_small,
    is_llama_weak as _is_llama_weak,
    get_recommended_temperature as _get_recommended_temperature,
    # Phase 15: Model-specific optimizations and sampling config
    get_sampling_config as _get_sampling_config,
    get_adaptive_sampling_config as _get_adaptive_sampling_config,
    optimize_chunk_count_for_model as _optimize_chunk_count_for_model,
    is_qwen_model as _is_qwen_model,
    is_phi_model as _is_phi_model,
    is_gemma_model as _is_gemma_model,
    is_deepseek_model as _is_deepseek_model,
    SMALL_MODEL_SYSTEM_PROMPT,
    TINY_MODEL_SYSTEM_PROMPT,
    TINY_MODEL_TEMPLATE,
    LLAMA_SMALL_SYSTEM_PROMPT,
    LLAMA_SMALL_TEMPLATE,
    LLAMA_WEAK_SYSTEM_PROMPT,
    QWEN_SMALL_SYSTEM_PROMPT,
    QWEN_SMALL_TEMPLATE,
    PHI_SMALL_SYSTEM_PROMPT,
    PHI_SMALL_TEMPLATE,
    GEMMA_SMALL_SYSTEM_PROMPT,
    GEMMA_SMALL_TEMPLATE,
    DEEPSEEK_SMALL_SYSTEM_PROMPT,
    DEEPSEEK_SMALL_TEMPLATE,
)

# Phase 70: Resilience patterns for LLM calls
from backend.services.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    retry_with_backoff,
    RetryConfig,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Phase 82: Anthropic Prompt Caching Helper
# =============================================================================

def _make_system_message(content: str, model_name: Optional[str] = None) -> SystemMessage:
    """
    Create a SystemMessage with Anthropic prompt caching when applicable.

    Anthropic's prompt caching caches the system prompt prefix, saving 50-60%
    on repeated calls with the same system prompt. The `cache_control` block
    marks the system message as cacheable for up to 5 minutes.

    Args:
        content: System prompt text
        model_name: Model name to detect Anthropic models

    Returns:
        SystemMessage with cache_control if using Anthropic
    """
    is_anthropic = model_name and any(
        k in model_name.lower() for k in ("claude", "anthropic")
    )
    if is_anthropic and len(content) > 100:
        return SystemMessage(
            content=content,
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        )
    return SystemMessage(content=content)


# =============================================================================
# Phase 70: Resilient LLM Call Wrapper
# =============================================================================

# Circuit breakers for different LLM providers (prevent cascading failures)
_llm_circuit_breakers: Dict[str, CircuitBreaker] = {}
_llm_circuit_breaker_lock = asyncio.Lock()


async def _get_llm_circuit_breaker(provider: str) -> CircuitBreaker:
    """Get or create a circuit breaker for an LLM provider."""
    async with _llm_circuit_breaker_lock:
        if provider not in _llm_circuit_breakers:
            # Read from settings service (database-persisted, admin-configurable)
            # with env var fallback for startup/bootstrap scenarios
            try:
                from backend.services.settings import get_settings_service
                settings_svc = get_settings_service()
                failure_threshold = await settings_svc.get_setting("llm.circuit_breaker_threshold")
                recovery_timeout = await settings_svc.get_setting("llm.circuit_breaker_recovery")
                call_timeout = await settings_svc.get_setting("llm.call_timeout")
            except Exception:
                failure_threshold = None
                recovery_timeout = None
                call_timeout = None

            config = CircuitBreakerConfig(
                failure_threshold=int(failure_threshold or os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "5")),
                recovery_timeout=float(recovery_timeout or os.getenv("LLM_CIRCUIT_BREAKER_RECOVERY", "60.0")),
                success_threshold=2,
                timeout=float(call_timeout or os.getenv("LLM_CALL_TIMEOUT", "120.0")),
            )
            _llm_circuit_breakers[provider] = CircuitBreaker(f"llm_{provider}", config)
        return _llm_circuit_breakers[provider]


async def resilient_llm_invoke(
    llm: Any,
    messages: List[Any],
    provider: str = "default",
    **kwargs,
) -> Any:
    """
    Invoke LLM with circuit breaker and retry patterns.

    Provides:
    - Circuit breaker: Prevents cascading failures when LLM service is down
    - Retry with exponential backoff: Handles transient failures
    - Timeout handling: Prevents hanging on slow responses

    Args:
        llm: The LangChain LLM instance
        messages: Messages to send to the LLM
        provider: Provider name for circuit breaker isolation
        **kwargs: Additional arguments for LLM invocation

    Returns:
        LLM response

    Raises:
        CircuitBreakerOpen: If provider circuit is open
        Exception: If all retries fail
    """
    cb = await _get_llm_circuit_breaker(provider)

    # Define the actual LLM call
    async def _invoke():
        return await llm.ainvoke(messages, **kwargs)

    # Configure retry for transient failures
    # Read max_retries from settings service with env var fallback
    try:
        from backend.services.settings import get_settings_service
        settings_svc = get_settings_service()
        max_retries_setting = await settings_svc.get_setting("llm.max_retries")
    except Exception:
        max_retries_setting = None

    retry_config = RetryConfig(
        max_retries=int(max_retries_setting or os.getenv("LLM_MAX_RETRIES", "3")),
        base_delay=float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0")),
        max_delay=float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0")),
        jitter=True,
    )

    # Apply circuit breaker with retry inside
    async def _invoke_with_retry():
        return await retry_with_backoff(_invoke, config=retry_config)

    return await cb.call(_invoke_with_retry)


def get_llm_circuit_breaker_stats() -> Dict[str, Any]:
    """Get stats for all LLM circuit breakers."""
    return {
        name: cb.get_stats()
        for name, cb in _llm_circuit_breakers.items()
    }


async def resilient_llm_stream(
    llm: Any,
    messages: List[Any],
    provider: str = "default",
    **kwargs,
) -> Any:
    """
    Create LLM stream with circuit breaker protection.

    For streaming, we check the circuit breaker before starting the stream.
    If the circuit is open, we fail fast. Otherwise, we track the success/failure
    of the stream initiation.

    Args:
        llm: The LangChain LLM instance
        messages: Messages to send to the LLM
        provider: Provider name for circuit breaker isolation
        **kwargs: Additional arguments for LLM streaming

    Returns:
        Async generator of stream chunks

    Raises:
        CircuitBreakerOpen: If provider circuit is open
    """
    cb = await _get_llm_circuit_breaker(provider)

    # Check circuit state before starting stream
    if cb.state.value == "open":
        raise CircuitBreakerOpen(
            f"Circuit breaker '{cb.name}' is open. "
            f"Service unavailable, will retry after {cb.config.recovery_timeout}s"
        )

    try:
        # Create the stream - this is where connection issues typically occur
        stream = llm.astream(messages, **kwargs)

        # Wrap the stream to track success on first chunk
        first_chunk = True

        async def tracked_stream():
            nonlocal first_chunk
            try:
                async for chunk in stream:
                    if first_chunk:
                        # First chunk received = success
                        await cb._record_success()
                        first_chunk = False  # Only record once
                    yield chunk
            except Exception as e:
                # Stream failed mid-way
                await cb._record_failure(e)
                raise

        return tracked_stream()

    except Exception as e:
        # Failed to create stream (connection error)
        await cb._record_failure(e)
        raise


# NOTE: Source, RAGResponse, StreamChunk, RAGConfig are now imported from
# backend.services.rag_module. The following class definitions have been removed
# and replaced with imports at the top of this file.

# NOTE: RAG_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, CONVERSATIONAL_RAG_TEMPLATE,
# LANGUAGE_NAMES, _get_language_instruction, and _parse_suggested_questions are
# now imported from backend.services.rag_module.prompts


# =============================================================================
# Search Result Cache (reduces redundant vectorstore queries by 40-60%)
# =============================================================================

class SearchResultCache:
    """
    TTL cache for vector search results.

    Caches identical query + collection + access_tier combinations to avoid
    redundant vectorstore queries. Useful for repeated/paginated queries.
    """

    def __init__(self, ttl_seconds: int = None, max_size: int = 100):
        """
        Initialize search result cache.

        Args:
            ttl_seconds: Cache TTL in seconds (default: 5 minutes from env)
            max_size: Maximum number of cached entries
        """
        # Read from settings service (synchronous default), with env var fallback
        if ttl_seconds is None:
            try:
                from backend.services.settings import get_settings_service
                ttl_seconds = get_settings_service().get_default_value("search.query_cache_ttl_seconds")
            except Exception:
                pass
        self._ttl_seconds = ttl_seconds or int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "300"))
        self._max_size = max_size
        self._cache: Dict[str, Tuple[List[Any], datetime]] = {}

    def _make_key(self, query: str, collection: Optional[str], access_tier: int, top_k: int) -> str:
        """Create cache key from query parameters (SHA-256 for collision resistance)."""
        key_data = f"{query}|{collection or ''}|{access_tier}|{top_k}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        query: str,
        collection: Optional[str],
        access_tier: int,
        top_k: int,
    ) -> Optional[List[Any]]:
        """Get cached results if available and not expired."""
        key = self._make_key(query, collection, access_tier, top_k)

        if key in self._cache:
            results, cached_at = self._cache[key]
            if datetime.now() - cached_at < timedelta(seconds=self._ttl_seconds):
                logger.debug("Search cache hit", query_hash=key[:8])
                return results
            else:
                # Expired, remove it
                del self._cache[key]

        return None

    def set(
        self,
        query: str,
        collection: Optional[str],
        access_tier: int,
        top_k: int,
        results: List[Any],
    ) -> None:
        """Cache search results."""
        # Don't cache empty results - they might be due to temporary issues
        if not results:
            logger.debug("Skipping cache for empty results", query_preview=query[:50])
            return

        # Evict oldest entries if at capacity
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        key = self._make_key(query, collection, access_tier, top_k)
        self._cache[key] = (results, datetime.now())
        logger.debug("Search cache set", query_hash=key[:8], result_count=len(results))

    def invalidate(self, collection: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            collection: If provided, only invalidate entries for this collection.
                       If None, invalidate all entries.

        Returns:
            Number of entries invalidated.
        """
        if collection is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        # Remove entries matching collection (requires re-computing keys, so just clear all)
        count = len(self._cache)
        self._cache.clear()
        return count


# Global search result cache instance
_search_cache = SearchResultCache()


def get_search_cache() -> SearchResultCache:
    """Get the global search result cache."""
    return _search_cache


# =============================================================================
# RAG Service
# =============================================================================

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service.

    Features:
    - Hybrid search (vector + keyword)
    - Conversational memory
    - Source citation
    - Streaming responses
    - Multi-provider LLM support
    - Database-driven LLM configuration
    - Per-session LLM override support
    - Usage tracking and cost estimation
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        vector_store: Optional[Any] = None,
        use_custom_vectorstore: bool = True,
        use_db_config: bool = True,  # Enable database-driven LLM configuration
        track_usage: bool = True,  # Enable usage tracking
    ):
        """
        Initialize RAG service.

        Args:
            config: RAG configuration
            llm_config: LLM configuration with API keys
            vector_store: Pre-initialized vector store (optional)
            use_custom_vectorstore: Use our custom VectorStore service (default: True)
            use_db_config: Use database-driven LLM configuration (default: True)
            track_usage: Enable LLM usage tracking (default: True)
        """
        self.config = config or RAGConfig()
        self.llm_config = llm_config or LLMConfig.from_env()
        self.use_db_config = use_db_config
        self.track_usage = track_usage

        # Initialize components lazily
        self._llm = None
        self._embeddings = None
        self._vector_store = vector_store
        self._custom_vectorstore: Optional[VectorStore] = None
        self._use_custom_vectorstore = use_custom_vectorstore

        # Use centralized session memory manager with LRU eviction (prevents memory leaks)
        self._session_memory = get_session_memory_manager(
            max_sessions=1000,
            memory_window_k=self.config.memory_window,
            cleanup_stale_after_hours=24.0,
        )

        # Cache for session-specific LLM instances (TTL + max size to prevent unbounded growth)
        self._session_llm_cache: Dict[str, Any] = {}
        self._session_config_cache: Dict[str, LLMConfigResult] = {}
        self._session_cache_times: Dict[str, float] = {}
        self._SESSION_CACHE_TTL = 3600  # 1 hour
        self._SESSION_CACHE_MAX_SIZE = 200

        # Initialize custom vector store if enabled
        if use_custom_vectorstore and vector_store is None:
            self._custom_vectorstore = get_vector_store()

        # Search result cache (reduces redundant vectorstore queries)
        self._search_cache = get_search_cache()

        # Initialize query expander if enabled
        self._query_expander: Optional[QueryExpander] = None
        if self.config.enable_query_expansion:
            expansion_config = QueryExpansionConfig(
                enabled=True,
                expansion_count=self.config.query_expansion_count,
                model=self.config.query_expansion_model,
            )
            self._query_expander = QueryExpander(expansion_config)

        # Initialize verifier if enabled (lazy initialization to use shared embedding service)
        self._verifier: Optional[RAGVerifier] = None
        self._verifier_config: Optional[VerifierConfig] = None
        if self.config.enable_verification:
            level_map = {
                "none": VerificationLevel.NONE,
                "quick": VerificationLevel.QUICK,
                "standard": VerificationLevel.STANDARD,
                "thorough": VerificationLevel.THOROUGH,
            }
            self._verifier_config = VerifierConfig(
                level=level_map.get(self.config.verification_level, VerificationLevel.QUICK),
            )

        # Initialize query classifier for dynamic hybrid search weighting
        self._query_classifier: Optional[QueryClassifier] = None
        if self.config.enable_dynamic_weighting:
            self._query_classifier = get_query_classifier()

        # Initialize HyDE expander for short/abstract queries
        self._hyde_expander: Optional[HyDEExpander] = None
        if self.config.enable_hyde:
            self._hyde_expander = get_hyde_expander()

        # Initialize CRAG for low-confidence result correction
        self._crag: Optional[CorrectiveRAG] = None
        if self.config.enable_crag:
            self._crag = get_corrective_rag()

        # Initialize hierarchical retriever for document-diverse retrieval
        self._hierarchical_retriever: Optional[HierarchicalRetriever] = None
        if self.config.enable_hierarchical_retrieval and self._custom_vectorstore:
            self._hierarchical_retriever = get_hierarchical_retriever(
                vectorstore=self._custom_vectorstore,
                doc_limit=self.config.hierarchical_doc_limit,
                chunks_per_doc=self.config.hierarchical_chunks_per_doc,
            )

        # Initialize two-stage retriever for ColBERT reranking at scale
        self._two_stage_retriever: Optional[TwoStageRetriever] = None
        if self.config.enable_two_stage_retrieval and self._custom_vectorstore:
            two_stage_config = TwoStageConfig(
                stage1_candidates=self.config.two_stage_candidates,
                final_top_k=self.config.top_k,
                use_colbert=self.config.use_colbert_reranker,
                use_hybrid_stage1=self.config.use_hybrid_search,
            )
            self._two_stage_retriever = TwoStageRetriever(
                vectorstore=self._custom_vectorstore,
                config=two_stage_config,
            )

        # Knowledge graph is initialized lazily per-request (requires db session)
        self._enable_knowledge_graph = self.config.enable_knowledge_graph
        self._knowledge_graph_max_hops = self.config.knowledge_graph_max_hops

        # Initialize smart document pre-filter for large collections
        self._smart_filter: Optional[SmartDocumentFilter] = None
        if self.config.enable_smart_filter:
            smart_filter_config = SmartFilterConfig(
                min_docs_for_filter=self.config.smart_filter_min_docs,
                max_docs_for_full_search=self.config.smart_filter_min_docs * 10,  # 10x threshold
                metadata_max_candidates=self.config.smart_filter_max_candidates,
                summary_top_k=self.config.smart_filter_max_candidates // 2,
            )
            self._smart_filter = get_smart_filter(config=smart_filter_config)

        # Initialize Self-RAG for response verification (hallucination detection)
        self._self_rag: Optional[SelfRAG] = None
        if self.config.enable_self_rag:
            self._self_rag = get_self_rag(
                min_supported_ratio=self.config.self_rag_min_supported_ratio,
                enable_regeneration=self.config.self_rag_enable_regeneration,
            )

        # Phase 58: Initialize HybridRetriever for LightRAG + RAPTOR fusion
        # HybridRetriever is used when LightRAG or RAPTOR is enabled
        from backend.core.config import settings as core_settings
        self._hybrid_retriever: Optional[HybridRetriever] = None
        self._use_hybrid_retriever = (
            getattr(core_settings, 'ENABLE_LIGHTRAG', False) or
            getattr(core_settings, 'ENABLE_RAPTOR', False) or
            getattr(core_settings, 'ENABLE_WARP', False) or
            getattr(core_settings, 'ENABLE_COLPALI', False)
        )
        # HybridRetriever is initialized lazily in _retrieve_with_custom_store
        # to avoid circular dependencies and allow async initialization

        # Phase 65: Initialize unified pipeline for spell correction, semantic cache, LTR, citations
        # Pipeline is initialized lazily (async) on first use, but we prepare config here
        self._phase65_pipeline: Optional["Phase65Pipeline"] = None
        self._phase65_enabled = PHASE65_AVAILABLE and getattr(
            core_settings, 'ENABLE_PHASE65', True
        )
        if self._phase65_enabled and PHASE65_AVAILABLE:
            # Initialize synchronously (without calling async initialize)
            # Full initialization happens on first query
            self._phase65_pipeline = get_phase65_pipeline_sync()

        # Phase 66: Initialize Adaptive Router for intelligent query routing
        # Routes queries to optimal strategies (DIRECT, HYBRID, TWO_STAGE, AGENTIC, GRAPH)
        self._adaptive_router: Optional[AdaptiveRouter] = None
        self._enable_adaptive_routing = getattr(core_settings, 'ENABLE_ADAPTIVE_ROUTING', True)
        if self._enable_adaptive_routing:
            self._adaptive_router = get_adaptive_router(
                query_classifier=self._query_classifier,
            )

        # Phase 66: Advanced RAG utilities
        self._rag_fusion: Optional[RAGFusion] = None
        self._context_compressor: Optional[ContextCompressor] = None
        self._stepback_prompter: Optional[StepBackPrompter] = None
        self._enable_rag_fusion = getattr(core_settings, 'ENABLE_RAG_FUSION', True)
        self._enable_context_compression = getattr(core_settings, 'ENABLE_CONTEXT_COMPRESSION', True)
        self._enable_stepback_prompting = getattr(core_settings, 'ENABLE_STEPBACK_PROMPTING', True)
        self._enable_context_reordering = getattr(core_settings, 'ENABLE_CONTEXT_REORDERING', True)

        # Phase 66: LazyGraphRAG for cost-efficient knowledge graph retrieval
        # Uses query-time community summarization instead of index-time (99% cost reduction)
        self._lazy_graphrag: Optional[LazyGraphRAGService] = None
        self._enable_lazy_graphrag = getattr(core_settings, 'ENABLE_LAZY_GRAPHRAG', True)
        self._last_lazy_graphrag_context: Optional[LazyGraphContext] = None

        logger.info(
            "Initialized RAG service",
            chat_provider=self.config.chat_provider,
            embedding_provider=self.config.embedding_provider,
            top_k=self.config.top_k,
            use_custom_vectorstore=use_custom_vectorstore,
            use_db_config=use_db_config,
            track_usage=track_usage,
            query_expansion=self.config.enable_query_expansion,
            verification=self.config.enable_verification,
            verification_level=self.config.verification_level,
            dynamic_weighting=self.config.enable_dynamic_weighting,
            hyde=self.config.enable_hyde,
            crag=self.config.enable_crag,
            two_stage_retrieval=self.config.enable_two_stage_retrieval,
            hierarchical_retrieval=self.config.enable_hierarchical_retrieval,
            semantic_dedup=self.config.enable_semantic_dedup,
            knowledge_graph=self.config.enable_knowledge_graph,
            smart_filter=self.config.enable_smart_filter,
            self_rag=self.config.enable_self_rag,
            hybrid_retriever=self._use_hybrid_retriever,
        )

    async def get_runtime_settings(self) -> Dict[str, Any]:
        """
        Load RAG settings from the database at runtime.

        Returns settings dict with keys:
        - top_k: Number of documents to retrieve
        - rerank_results: Whether to rerank results
        - query_expansion_count: Number of query expansions
        - similarity_threshold: Minimum similarity score
        """
        from backend.services.settings import get_settings_service

        settings_service = get_settings_service()
        settings = await settings_service.get_all_settings()

        return {
            "top_k": settings.get("rag.top_k", self.config.top_k),
            "rerank_results": settings.get("rag.rerank_results", self.config.rerank_results),
            "query_expansion_count": settings.get("rag.query_expansion_count", self.config.query_expansion_count),
            "similarity_threshold": settings.get("rag.similarity_threshold", self.config.similarity_threshold),
            "query_decomposition_enabled": settings.get("rag.query_decomposition_enabled", False),
            "decomposition_min_words": settings.get("rag.decomposition_min_words", 10),
        }

    @property
    def llm(self):
        """Get or create default LLM instance (without session override)."""
        if self._llm is None:
            self._llm = LLMFactory.get_chat_model(
                provider=self.config.chat_provider,
                model=self.config.chat_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_response_tokens,
            )
        return self._llm

    async def get_llm_for_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[Any, Optional[LLMConfigResult]]:
        """
        Get LLM instance with session-specific configuration.

        Uses database-driven configuration with priority:
        1. Per-session override
        2. Operation-specific config (rag)
        3. Default provider
        4. Environment variables

        Args:
            session_id: Chat session ID for per-session override
            user_id: User ID for tracking

        Returns:
            Tuple of (LLM instance, LLMConfigResult or None)
        """
        if not self.use_db_config:
            return self.llm, None

        # Check cache first (with TTL validation)
        import time as _cache_time
        cache_key = session_id or "_default"
        now = _cache_time.time()

        if cache_key in self._session_llm_cache:
            cached_at = self._session_cache_times.get(cache_key, 0)
            if (now - cached_at) < self._SESSION_CACHE_TTL:
                return self._session_llm_cache[cache_key], self._session_config_cache.get(cache_key)
            else:
                # TTL expired — evict stale entry
                self._session_llm_cache.pop(cache_key, None)
                self._session_config_cache.pop(cache_key, None)
                self._session_cache_times.pop(cache_key, None)

        try:
            llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="rag",
                session_id=session_id,
                user_id=user_id,
            )

            # Evict oldest entries if cache is full
            if len(self._session_llm_cache) >= self._SESSION_CACHE_MAX_SIZE:
                oldest_key = min(self._session_cache_times, key=self._session_cache_times.get)
                self._session_llm_cache.pop(oldest_key, None)
                self._session_config_cache.pop(oldest_key, None)
                self._session_cache_times.pop(oldest_key, None)

            # Cache the result with timestamp
            self._session_llm_cache[cache_key] = llm
            self._session_config_cache[cache_key] = config
            self._session_cache_times[cache_key] = now

            logger.debug(
                "Got LLM for session",
                session_id=session_id,
                provider=config.provider_type,
                model=config.model,
                source=config.source,
            )

            return llm, config

        except Exception as e:
            logger.warning(
                "Failed to get session LLM config, using default",
                session_id=session_id,
                error=str(e),
            )
            return self.llm, None

    def clear_session_llm_cache(self, session_id: Optional[str] = None):
        """Clear cached LLM for a session (or all sessions)."""
        if session_id:
            self._session_llm_cache.pop(session_id, None)
            self._session_config_cache.pop(session_id, None)
            self._session_cache_times.pop(session_id, None)
        else:
            self._session_llm_cache.clear()
            self._session_config_cache.clear()
            self._session_cache_times.clear()

    @property
    def embeddings(self):
        """Get or create embeddings instance."""
        if self._embeddings is None:
            self._embeddings = EmbeddingService(
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
                config=self.llm_config,
            )
        return self._embeddings

    @property
    def verifier(self) -> Optional[RAGVerifier]:
        """Get or create verifier instance with proper embedding service."""
        if self._verifier_config is not None and self._verifier is None:
            # Lazy initialize verifier with shared embedding service
            self._verifier = RAGVerifier(
                config=self._verifier_config,
                embedding_service=self.embeddings,  # Pass the configured embedding service
            )
        return self._verifier

    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create conversation memory for session using centralized manager."""
        return self._session_memory.get_memory(session_id)

    def clear_memory(self, session_id: str):
        """Clear conversation memory for session."""
        self._session_memory.clear_memory(session_id)

    def _count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken for accurate counting.

        Args:
            text: Text to count tokens for
            model: Model name for encoding selection (defaults to cl100k_base)

        Returns:
            Number of tokens
        """
        if not HAS_TIKTOKEN or not text:
            # Fallback to rough estimation: ~4 characters per token
            return len(text) // 4

        try:
            # Map model names to encoding
            # cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
            # o200k_base is used by gpt-4o
            model = model or self.config.chat_model or "gpt-4"

            if "gpt-4o" in model or "o1" in model:
                encoding = tiktoken.get_encoding("o200k_base")
            elif "gpt-4" in model or "gpt-3.5" in model or "text-embedding" in model:
                encoding = tiktoken.get_encoding("cl100k_base")
            elif "claude" in model.lower():
                # Anthropic models don't have tiktoken support, use estimation
                # Claude uses ~3.5 chars per token on average
                return int(len(text) / 3.5)
            else:
                # Default to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except Exception as e:
            logger.debug("Token counting failed, using estimation", error=str(e))
            return len(text) // 4

    async def search(
        self,
        query: str,
        limit: int = 5,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search documents without generating an LLM response.

        This method is useful for agent workflows that need raw document
        chunks without the full RAG query processing.

        Args:
            query: Search query
            limit: Maximum number of results to return
            collection_filter: Filter by collection
            access_tier: User's access tier for RLS
            organization_id: Organization ID for multi-tenant isolation
            user_id: User ID for private document access
            is_superadmin: Whether user is a superadmin (can access all private docs)

        Returns:
            List of document dicts with content, source, and score
        """
        logger.info(
            "Searching documents",
            query_length=len(query),
            limit=limit,
            collection_filter=collection_filter,
            organization_id=organization_id,
        )

        # Use _retrieve to get document chunks
        retrieved_docs = await self._retrieve(
            query=query,
            collection_filter=collection_filter,
            access_tier=access_tier,
            top_k=limit,
            organization_id=organization_id,
            user_id=user_id,
            is_superadmin=is_superadmin,
        )

        # Format results for agent consumption
        results = []
        for doc, score in retrieved_docs:
            # Get document name with proper fallback chain
            # Custom vectorstore sets "document_name", others may use "document_filename" or "source"
            doc_name = (
                doc.metadata.get("document_name") or
                doc.metadata.get("document_filename") or
                doc.metadata.get("source") or
                "Unknown Document"
            )
            # Use similarity_score (0-1) for display, fallback to RRF score if not available
            # RRF scores are typically 0.01-0.02, so we need to use the original similarity
            display_score = doc.metadata.get("similarity_score", score)

            results.append({
                "content": doc.page_content,
                "document_name": doc_name,
                "source": doc.metadata.get("source") or doc_name,
                "chunk_id": doc.metadata.get("chunk_id"),
                "score": display_score,
                "metadata": doc.metadata,
            })

        logger.info(
            "Document search completed",
            result_count=len(results),
            query=query[:50],
        )

        return results

    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
        include_collection_context: bool = True,
        additional_context: Optional[str] = None,
        top_k: Optional[int] = None,  # Override documents to retrieve per query
        folder_id: Optional[str] = None,  # Folder-scoped query
        include_subfolders: bool = True,  # Include subfolders in folder query
        language: str = "en",  # Language for response
        enhance_query: Optional[bool] = None,  # Per-query override for query enhancement (expansion + HyDE)
        organization_id: Optional[str] = None,  # Organization ID for multi-tenant isolation
        is_superadmin: bool = False,  # Whether user is a superadmin
        use_multilingual_search: bool = False,  # Phase 59: Enable cross-lingual search
    ) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            question: User's question
            session_id: Session ID for conversation memory
            collection_filter: Filter by document collection
            access_tier: User's access tier for RLS
            user_id: User ID for usage tracking and private document access
            include_collection_context: Whether to include collection tags in LLM context
            additional_context: Additional context to include (e.g., from temp documents)
            top_k: Optional override for number of documents to retrieve
            folder_id: Optional folder ID to scope query to specific folder
            include_subfolders: Whether to include documents from subfolders
            language: Language code for response (en, de, es, fr, etc.)
            enhance_query: Per-query override for query enhancement (expansion + HyDE).
                           None = use admin default, True = enable, False = disable.
            organization_id: Organization ID for multi-tenant isolation.
            is_superadmin: Whether user is a superadmin (can access all private docs).

        Returns:
            RAGResponse with answer and sources
        """
        import time
        start_time = time.time()

        # Load runtime settings from database
        runtime_settings = await self.get_runtime_settings()
        if top_k is None:
            top_k = runtime_settings.get("top_k", self.config.top_k)

        # Batch-load all RAG settings in a single DB query to avoid 30+ individual lookups
        from backend.services.settings import get_settings_service
        settings_svc = get_settings_service()
        try:
            _all_settings = await settings_svc.get_all_settings()
        except Exception:
            _all_settings = {}

        def _s(key: str, default=None):
            """Read a setting from the preloaded batch (zero DB cost)."""
            val = _all_settings.get(key)
            return val if val is not None else default

        # PHASE 12: Enhanced debug logging for RAG search troubleshooting
        # Get vectorstore stats to help diagnose "no results" issues
        vectorstore_stats = None
        try:
            if self._custom_vectorstore is not None:
                vectorstore_stats = await self._custom_vectorstore.get_stats()
        except Exception as e:
            logger.debug("Could not get vectorstore stats", error=str(e))

        logger.info(
            "Processing RAG query",
            question_length=len(question),
            question_preview=question[:100],
            session_id=session_id,
            collection_filter=collection_filter,
            top_k=top_k,
            language=language,
            folder_id=folder_id,
            organization_id=organization_id,
            user_id=user_id,
            is_superadmin=is_superadmin,
            vectorstore_stats=vectorstore_stats,
        )

        # =============================================================================
        # Phase 65: Query Preprocessing (Spell Correction + Semantic Cache)
        # =============================================================================
        original_question = question  # Preserve original for logging
        phase65_cache_hit = None
        query_embedding_for_cache = None

        if self._phase65_enabled and self._phase65_pipeline:
            try:
                # Initialize pipeline if not already done (lazy async init)
                if not self._phase65_pipeline._initialized:
                    await self._phase65_pipeline.initialize()

                # Get query embedding for semantic cache (reuse later for retrieval)
                embedding_service = self.embeddings
                if embedding_service:
                    try:
                        query_embedding_for_cache = await embedding_service.embed_query(question)
                    except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
                        logger.debug("Could not get query embedding for cache", error=str(e), error_type=type(e).__name__)

                # Preprocess: spell correction + semantic cache check
                question, phase65_cache_hit = await self._phase65_pipeline.preprocess_query(
                    question, query_embedding_for_cache
                )

                if question != original_question:
                    logger.info(
                        "Phase 65: Query spell-corrected",
                        original=original_question[:50],
                        corrected=question[:50],
                    )

                if phase65_cache_hit is not None:
                    logger.info("Phase 65: Semantic cache hit - returning cached result")
                    # Cache hit returns the full RAGResponse
                    if isinstance(phase65_cache_hit, RAGResponse):
                        phase65_cache_hit.metadata = phase65_cache_hit.metadata or {}
                        phase65_cache_hit.metadata["phase65_cache_hit"] = True
                        return phase65_cache_hit
                    # If it's a dict, reconstruct RAGResponse
                    elif isinstance(phase65_cache_hit, dict):
                        return RAGResponse(
                            content=phase65_cache_hit.get("content", ""),
                            sources=phase65_cache_hit.get("sources", []),
                            query=original_question,
                            model=phase65_cache_hit.get("model", "cached"),
                            confidence_score=phase65_cache_hit.get("confidence_score"),
                            confidence_level=phase65_cache_hit.get("confidence_level"),
                            metadata={"phase65_cache_hit": True},
                        )

            except (ValueError, RuntimeError, ConnectionError, TimeoutError, KeyError, TypeError) as e:
                logger.warning(
                    "Phase 65 preprocessing failed, continuing with original query",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                question = original_question  # Restore original on error

        # PHASE 14: Query classification for adaptive retrieval and prompt selection
        # Classify query intent to optimize retrieval settings and prompt templates
        query_classification: Optional[QueryClassification] = None
        try:
            classifier = get_query_classifier()
            query_classification = classifier.classify(question)

            # Apply classification to retrieval settings
            if query_classification:
                # Override top_k if classification suggests different value
                if query_classification.suggested_top_k and top_k is None:
                    top_k = query_classification.suggested_top_k

                logger.info(
                    "Query classified for adaptive retrieval",
                    intent=query_classification.intent.value,
                    confidence=query_classification.confidence,
                    use_mmr=query_classification.use_mmr,
                    use_cot=query_classification.use_cot,
                    use_kg_enhancement=query_classification.use_kg_enhancement,
                    suggested_top_k=query_classification.suggested_top_k,
                    prompt_template=query_classification.prompt_template,
                )
        except (ValueError, RuntimeError, AttributeError, KeyError) as e:
            logger.warning("Query classification failed, using defaults", error=str(e), error_type=type(e).__name__)
            query_classification = None

        # =============================================================================
        # Phase 66: Adaptive Query Routing
        # =============================================================================
        # Routes queries to optimal retrieval strategies based on complexity analysis
        routing_decision: Optional[RoutingDecision] = None
        use_rag_fusion_for_query = False
        use_stepback_for_query = False
        use_context_compression = False
        use_context_reordering = self._enable_context_reordering

        if self._adaptive_router and self._enable_adaptive_routing:
            try:
                # Get adaptive routing settings from database
                adaptive_routing_enabled = _s("rag.adaptive_routing_enabled")
                if adaptive_routing_enabled is None:
                    adaptive_routing_enabled = True  # Default enabled

                if adaptive_routing_enabled:
                    routing_decision = await self._adaptive_router.route(
                        query=question,
                        context={
                            "session_id": session_id,
                            "collection_filter": collection_filter,
                            "query_classification": query_classification.intent.value if query_classification else None,
                            "max_sources": top_k,
                        },
                    )

                    # Apply routing decision to retrieval settings
                    if routing_decision:
                        # Override top_k if router suggests different value
                        if top_k is None or routing_decision.top_k > top_k:
                            top_k = routing_decision.top_k

                        # Apply advanced RAG techniques based on routing decision
                        use_rag_fusion_for_query = (
                            routing_decision.use_rag_fusion and
                            self._enable_rag_fusion and
                            _s("rag.rag_fusion_enabled") != False
                        )
                        use_stepback_for_query = (
                            routing_decision.use_step_back and
                            self._enable_stepback_prompting and
                            _s("rag.stepback_prompting_enabled") != False
                        )
                        use_context_compression = (
                            self._enable_context_compression and
                            _s("rag.context_compression_enabled") != False
                        )

                        # Log routing decision
                        logger.info(
                            "Phase 66: Adaptive routing decision",
                            complexity=routing_decision.complexity.value,
                            strategy=routing_decision.strategy.value,
                            top_k=routing_decision.top_k,
                            use_reranking=routing_decision.use_reranking,
                            use_hyde=routing_decision.use_hyde,
                            use_rag_fusion=use_rag_fusion_for_query,
                            use_stepback=use_stepback_for_query,
                            confidence=routing_decision.confidence,
                            reasoning=routing_decision.reasoning[:100] if routing_decision.reasoning else None,
                        )

            except (ValueError, RuntimeError, TimeoutError, ConnectionError, AttributeError) as e:
                logger.warning("Phase 66: Adaptive routing failed, using defaults", error=str(e), error_type=type(e).__name__)
                routing_decision = None

        # Fallback: when adaptive routing is off or unavailable, check individual feature settings
        if routing_decision is None:
            try:
                use_rag_fusion_for_query = (
                    self._enable_rag_fusion
                    and _s("rag.rag_fusion_enabled") != False
                )
                use_stepback_for_query = (
                    self._enable_stepback_prompting
                    and _s("rag.stepback_prompting_enabled") != False
                )
                use_context_compression = (
                    self._enable_context_compression
                    and _s("rag.context_compression_enabled") != False
                )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError, OSError) as e:
                logger.debug(
                    "Feature settings lookup failed, keeping defaults",
                    error=str(e),
                    error_type=type(e).__name__,
                    operation="feature_settings_fallback",
                    query=question[:100],
                    user_id=user_id,
                )

        # Phase 62: Tree of Thoughts for complex analytical queries
        # Use runtime settings for hot-reload (no server restart needed)
        tot_enabled = _s("rag.tree_of_thoughts_enabled", False)
        if (
            tot_enabled
            and query_classification
            and query_classification.intent in [QueryIntent.ANALYTICAL, QueryIntent.MULTI_HOP]
        ):
            try:
                from backend.services.tree_of_thoughts import TreeOfThoughts, ToTConfig
                tot_max_depth = _s("rag.tot_max_depth", 3)
                tot_branching = _s("rag.tot_branching_factor", 3)
                tot = TreeOfThoughts(ToTConfig(
                    max_depth=tot_max_depth,
                    branching_factor=tot_branching,
                    search_strategy='beam',
                ))
                tot_result = await tot.solve(problem=question, context=context)
                if tot_result and tot_result.confidence > 0.7:
                    logger.info("ToT produced high-confidence answer", confidence=tot_result.confidence)
                    # ToT provides reasoning path, continue with RAG for citation support
            except (ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Tree of Thoughts failed, continuing with standard RAG", error=str(e), error_type=type(e).__name__)

        # Check if this is an aggregation query (e.g., "total spending by Company A")
        # Route to specialized handler for numerical extraction and calculation
        aggregation_enabled = runtime_settings.get("aggregation_query_enabled", True)
        if aggregation_enabled:
            from backend.services.structured_extraction import get_structured_extractor
            extractor = get_structured_extractor()
            if extractor.is_aggregation_query(question):
                logger.info(
                    "Detected aggregation query - routing to specialized handler",
                    question=question[:50],
                )
                return await self.handle_aggregation_query(
                    question=question,
                    session_id=session_id,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    user_id=user_id,
                    folder_id=folder_id,
                    include_subfolders=include_subfolders,
                    language=language,
                    organization_id=organization_id,
                    is_superadmin=is_superadmin,
                )

        # Check if query decomposition is enabled
        decomposition_enabled = runtime_settings.get("query_decomposition_enabled", False)
        decomposed_query = None

        if decomposition_enabled:
            try:
                from backend.services.query_decomposer import get_query_decomposer
                decomposer = get_query_decomposer()
                decomposed_query = await decomposer.decompose(question)

                if len(decomposed_query.sub_queries) > 1:
                    logger.info(
                        "Query decomposed",
                        original=question,
                        sub_queries=decomposed_query.sub_queries,
                        query_type=decomposed_query.query_type,
                    )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Query decomposition failed", error=str(e), error_type=type(e).__name__)
                decomposed_query = None

        # Get LLM for this session (with database-driven config)
        llm, llm_config = await self.get_llm_for_session(
            session_id=session_id,
            user_id=user_id,
        )

        # PHASE 15: Dynamic chunk capping based on model capabilities
        # Research shows tiny models struggle with long context even if they support it
        # Cap chunks to prevent context overload while maintaining scalability
        if top_k is not None:
            original_top_k = top_k
            top_k = _optimize_chunk_count_for_model(
                intent_top_k=top_k,
                model_name=llm_config.model if llm_config else None,
            )
            if top_k != original_top_k:
                logger.info(
                    "Dynamic chunk capping applied for model capability",
                    original_top_k=original_top_k,
                    optimized_top_k=top_k,
                    model=llm_config.model if llm_config else None,
                )

        # Get folder-scoped document IDs if folder_id is specified
        folder_document_ids = None
        if folder_id:
            from backend.services.folder_service import get_folder_service
            folder_service = get_folder_service()
            folder_document_ids = await folder_service.get_folder_document_ids(
                folder_id=folder_id,
                include_subfolders=include_subfolders,
                user_tier_level=access_tier,
            )
            logger.debug(
                "Folder-scoped query",
                folder_id=folder_id,
                document_count=len(folder_document_ids),
            )
            # If no documents in folder, return empty response early
            if not folder_document_ids:
                logger.info("No documents found in specified folder")
                return RAGResponse(
                    content="No documents found in the specified folder.",
                    sources=[],
                    confidence_score=0.0,
                    confidence_level="low",
                )

        # =============================================================================
        # Phase 66: RAG-Fusion (Multi-Query with Reciprocal Rank Fusion)
        # =============================================================================
        rag_fusion_result: Optional[FusionResult] = None
        if use_rag_fusion_for_query and self._custom_vectorstore:
            try:
                logger.info("Phase 66: Applying RAG-Fusion with multi-query retrieval")

                # Initialize RAG-Fusion if needed
                if self._rag_fusion is None:
                    num_variations = _s("rag.rag_fusion_variations", 4)
                    self._rag_fusion = RAGFusion(
                        num_variations=num_variations,
                        rrf_k=60,
                        include_original=True,
                    )

                # Generate query variations and fuse results
                rag_fusion_result = await self._rag_fusion.fuse_results(
                    query=question,
                    retriever=self._custom_vectorstore,
                    session=None,  # Not needed for our vectorstore
                    top_k=top_k,
                    llm=llm,
                )

                logger.info(
                    "Phase 66: RAG-Fusion complete",
                    query_variations=len(rag_fusion_result.query_variations),
                    fused_results=len(rag_fusion_result.fused_results),
                )

            except (ValueError, RuntimeError, TimeoutError, ConnectionError, TypeError, KeyError) as e:
                logger.warning(
                    "Phase 66: RAG-Fusion failed, falling back to standard retrieval",
                    error=str(e),
                    error_type=type(e).__name__,
                    operation="rag_fusion",
                    query=question[:100],
                    user_id=user_id,
                    session_id=session_id,
                )
                rag_fusion_result = None

        # =============================================================================
        # Phase 66: Step-Back Prompting (Abstract reasoning for complex queries)
        # =============================================================================
        stepback_result: Optional[StepBackResult] = None
        stepback_context = ""
        if use_stepback_for_query and self._custom_vectorstore:
            try:
                logger.info("Phase 66: Applying Step-Back prompting for complex query")

                # Initialize step-back prompter if needed
                if self._stepback_prompter is None:
                    max_background = _s("rag.stepback_max_background", 3)
                    self._stepback_prompter = StepBackPrompter(max_background_chunks=max_background)

                stepback_result = await self._stepback_prompter.retrieve_with_stepback(
                    query=question,
                    retriever=self._custom_vectorstore,
                    session=None,
                    llm=llm,
                    top_k=top_k,
                )

                # Use combined context (background + specific) as additional context
                stepback_context = stepback_result.combined_context

                logger.info(
                    "Phase 66: Step-Back prompting complete",
                    stepback_query=stepback_result.stepback_query[:50] if stepback_result.stepback_query else None,
                    background_length=len(stepback_result.background_context),
                )

            except (ValueError, RuntimeError, TimeoutError, ConnectionError, TypeError, KeyError) as e:
                logger.warning(
                    "Phase 66: Step-Back prompting failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    operation="stepback_prompting",
                    query=question[:100],
                    user_id=user_id,
                    session_id=session_id,
                )
                stepback_result = None

        # Phase 95L: Conversation-aware retrieval
        # Enrich query with recent conversation context for better embeddings
        enriched_question = question
        if session_id:
            try:
                _conv_memory = self._get_memory(session_id)
                _conv_history = _conv_memory.load_memory_variables({}).get("chat_history", [])
                if _conv_history:
                    # Use last 2-3 user messages for context
                    recent_user_msgs = [
                        msg.content for msg in _conv_history[-6:]  # Last 3 turns (user+assistant pairs)
                        if hasattr(msg, 'content') and isinstance(msg, HumanMessage)
                    ][-3:]  # Take last 3 user messages
                    if recent_user_msgs:
                        conversation_context = " ".join(recent_user_msgs[-3:])
                        # Prepend a compact version of recent context
                        enriched_question = f"Context: {conversation_context[:200]}. Question: {question}"
                        logger.debug("Conversation-aware retrieval",
                            original_query=question[:100],
                            enriched_query_length=len(enriched_question),
                            context_messages=len(recent_user_msgs))
            except Exception as e:
                logger.debug("Conversation-aware retrieval failed, using original query", error=str(e))
                enriched_question = question

        # Retrieve relevant documents
        # Phase 66: If RAG-Fusion was used, convert its results to retrieved_docs format
        if rag_fusion_result and rag_fusion_result.fused_results:
            # Convert FusionResult to (Document, score) format
            retrieved_docs = []
            for result in rag_fusion_result.fused_results[:top_k]:
                # SearchResult has content, chunk_id, metadata, similarity_score
                score = getattr(result, 'similarity_score', 0.5)
                # Create a Document-like object
                from langchain_core.documents import Document as LCDocument
                doc = LCDocument(
                    page_content=getattr(result, 'content', str(result)),
                    metadata={
                        "chunk_id": getattr(result, 'chunk_id', ''),
                        "document_name": getattr(result, 'document_title', ''),
                        "document_filename": getattr(result, 'document_filename', ''),
                        "collection": getattr(result, 'collection', ''),
                        **(getattr(result, 'metadata', {}) or {}),
                    },
                )
                retrieved_docs.append((doc, score))

            logger.info(
                "Phase 66: Using RAG-Fusion results",
                count=len(retrieved_docs),
                query_variations=len(rag_fusion_result.query_variations),
            )
        # If query was decomposed, retrieve for each sub-query and merge
        elif decomposed_query and len(decomposed_query.sub_queries) > 1:
            all_retrieved_docs = []
            seen_chunk_ids = set()

            for sub_query in decomposed_query.sub_queries:
                sub_docs = await self._retrieve(
                    sub_query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=max(top_k // len(decomposed_query.sub_queries), 3),
                    document_ids=folder_document_ids,
                    enhance_query=enhance_query,
                    organization_id=organization_id,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                    use_multilingual_search=use_multilingual_search,
                    target_language=language,
                )
                # Deduplicate by chunk_id
                for doc in sub_docs:
                    chunk_id = doc.metadata.get("chunk_id")
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_retrieved_docs.append(doc)

            retrieved_docs = all_retrieved_docs[:top_k]
            logger.debug(
                "Retrieved documents for decomposed query",
                sub_query_count=len(decomposed_query.sub_queries),
                total_docs=len(retrieved_docs),
            )
        else:
            # PHASE 14: Pass query classification for adaptive retrieval (MMR, KG enhancement)
            # Phase 95L: Use enriched_question for retrieval (conversation-aware), keep original question for LLM prompt
            retrieved_docs = await self._retrieve(
                enriched_question,
                collection_filter=collection_filter,
                access_tier=access_tier,
                top_k=top_k,
                document_ids=folder_document_ids,  # Filter to folder documents
                enhance_query=enhance_query,
                organization_id=organization_id,
                user_id=user_id,
                is_superadmin=is_superadmin,
                query_classification=query_classification,
                use_multilingual_search=use_multilingual_search,
                target_language=language,
            )
            # PHASE 12: Enhanced logging for RAG search troubleshooting
            if not retrieved_docs:
                logger.warning(
                    "RAG retrieval returned NO documents - user may see 'no info' response",
                    question=question[:100],
                    collection_filter=collection_filter,
                    folder_document_ids_count=len(folder_document_ids) if folder_document_ids else 0,
                    organization_id=organization_id,
                    user_id=user_id,
                )
            else:
                logger.info(
                    "RAG retrieval successful",
                    count=len(retrieved_docs),
                    top_doc_score=retrieved_docs[0][1] if retrieved_docs else None,
                    top_doc_name=retrieved_docs[0][0].metadata.get("document_name", "unknown")[:50] if retrieved_docs else None,
                )

        # =============================================================================
        # Phase 66: Context Reordering (Lost-in-the-Middle Mitigation)
        # =============================================================================
        if use_context_reordering and retrieved_docs and len(retrieved_docs) > 2:
            try:
                reorder_strategy = _s("rag.context_reorder_strategy", "sandwich")

                # Create wrapper objects with similarity_score for reordering
                class ScoreWrapper:
                    def __init__(self, doc_tuple):
                        self.doc, self.similarity_score = doc_tuple
                        self.original_tuple = doc_tuple

                wrapped = [ScoreWrapper(dt) for dt in retrieved_docs]
                reordered = reorder_for_attention(wrapped, strategy=reorder_strategy)
                retrieved_docs = [w.original_tuple for w in reordered]

                logger.debug(
                    "Phase 66: Context reordered for better attention",
                    strategy=reorder_strategy,
                    count=len(retrieved_docs),
                )
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning("Phase 66: Context reordering failed", error=str(e), error_type=type(e).__name__)

        # =============================================================================
        # Phase 65: Learning-to-Rank Reranking
        # =============================================================================
        if self._phase65_enabled and self._phase65_pipeline and retrieved_docs:
            try:
                # Check if LTR is enabled via settings
                ltr_enabled = _s("search.ltr_enabled", False)
                if ltr_enabled is None:
                    ltr_enabled = True  # Default to enabled if setting not found

                if ltr_enabled:
                    # Convert retrieved docs to candidate format for LTR
                    candidates = []
                    for doc, score in retrieved_docs:
                        candidates.append({
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            "content": doc.page_content,
                            "score": score,
                            "document_title": doc.metadata.get("document_name", ""),
                            "document_filename": doc.metadata.get("document_filename", ""),
                            "collection": doc.metadata.get("collection", ""),
                            "metadata": doc.metadata,
                        })

                    # Apply LTR reranking
                    ltr_results = await self._phase65_pipeline.rerank_with_ltr(question, candidates)

                    if ltr_results:
                        # Rebuild retrieved_docs with LTR ordering
                        # Create a map from chunk_id to original (doc, score) tuples
                        doc_map = {
                            doc.metadata.get("chunk_id", ""): (doc, score)
                            for doc, score in retrieved_docs
                        }

                        # Reorder based on LTR results
                        reranked_docs = []
                        for ltr_result in ltr_results:
                            if ltr_result.doc_id in doc_map:
                                doc, _ = doc_map[ltr_result.doc_id]
                                # Use LTR score as new score
                                reranked_docs.append((doc, ltr_result.ltr_score))

                        if reranked_docs:
                            retrieved_docs = reranked_docs
                            logger.info(
                                "Phase 65: LTR reranking applied",
                                original_count=len(candidates),
                                reranked_count=len(reranked_docs),
                                top_ltr_score=reranked_docs[0][1] if reranked_docs else None,
                            )

            except (ValueError, RuntimeError, TimeoutError, ConnectionError, KeyError) as e:
                logger.warning("Phase 65 LTR reranking failed, using original order", error=str(e), error_type=type(e).__name__)

        # Verify retrieved documents if enabled
        verification_result = None
        if self.verifier is not None:
            try:
                verification_result = await self.verifier.verify(
                    query=question,
                    retrieved_docs=retrieved_docs,
                )
                # Filter to only relevant documents
                retrieved_docs = self.verifier.filter_by_relevance(
                    retrieved_docs, verification_result
                )
                logger.debug(
                    "Verification complete",
                    confidence=verification_result.confidence_score,
                    relevant=verification_result.num_relevant,
                    filtered=verification_result.num_filtered,
                )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Verification failed, continuing without filtering", error=str(e), error_type=type(e).__name__)
                verification_result = None

        # Phase 95K: Compute freshness config for _format_context (async settings lookup)
        freshness_config = None
        try:
            freshness_enabled = _s("rag.content_freshness_enabled")
            if freshness_enabled:
                freshness_config = {
                    "enabled": True,
                    "decay_days": int(_s("rag.freshness_decay_days", 180)),
                    "boost_factor": float(_s("rag.freshness_boost_factor", 1.05)),
                    "penalty_factor": float(_s("rag.freshness_penalty_factor", 0.95)),
                }
        except Exception as e:
            logger.debug("Freshness config lookup failed", error=str(e))

        # Format context from retrieved documents
        context, sources = self._format_context(retrieved_docs, include_collection_context, freshness_config=freshness_config)
        logger.debug("Formatted context", context_length=len(context), sources_count=len(sources))

        # Phase 79: Graph-O1 enhanced reasoning (beam search over knowledge graph)
        from backend.core.config import settings as _go1_settings
        if getattr(_go1_settings, 'KG_ENABLED', False):
            try:
                graph_o1_enabled = _s("rag.graph_o1_enabled", False)
                if graph_o1_enabled:
                    from backend.services.graph_o1 import reason_over_graph
                    go1_result = await reason_over_graph(
                        query=question,
                        context=context[:2000],
                    )
                    if go1_result and go1_result.confidence > 0.5 and go1_result.evidence:
                        go1_context = "\n[Graph-O1 Reasoning]\n"
                        go1_context += f"Confidence: {go1_result.confidence:.2f}\n"
                        go1_context += "\n".join(go1_result.evidence[:5])
                        context = context + "\n" + go1_context
                        logger.info(
                            "Graph-O1 reasoning added to context",
                            confidence=go1_result.confidence,
                            evidence_count=len(go1_result.evidence),
                            hops=go1_result.hops_used,
                        )
            except Exception as e:
                logger.debug("Graph-O1 reasoning skipped", error=str(e))

        # PHASE 2: Context sufficiency check (reduces hallucinations by 25-40%)
        # Check if retrieved contexts are sufficient to answer the query
        context_sufficiency_result = None
        try:
            from backend.services.context_sufficiency import get_context_sufficiency_checker
            sufficiency_checker = get_context_sufficiency_checker()

            # Extract context texts and metadata for sufficiency check
            context_texts = [s.full_content or s.snippet for s in sources]
            context_metadata = [
                {"document_name": s.document_name, "chunk_id": s.chunk_id}
                for s in sources
            ]

            context_sufficiency_result = await sufficiency_checker.check_sufficiency(
                query=question,
                contexts=context_texts,
                context_metadata=context_metadata,
            )

            logger.info(
                "Context sufficiency check complete",
                is_sufficient=context_sufficiency_result.is_sufficient,
                coverage_score=context_sufficiency_result.coverage_score,
                has_conflicts=context_sufficiency_result.has_conflicts,
                confidence_level=context_sufficiency_result.confidence_level,
                missing_aspects=context_sufficiency_result.missing_aspects[:3] if context_sufficiency_result.missing_aspects else [],
            )
        except (ValueError, RuntimeError, TimeoutError, ConnectionError, TypeError, AttributeError) as e:
            logger.warning(
                "Context sufficiency check failed, continuing without",
                error=str(e),
                error_type=type(e).__name__,
                operation="context_sufficiency_check",
                query=question[:100],
                user_id=user_id,
                sources_count=len(sources) if sources else 0,
            )

        # Add additional context (e.g., from temporary documents)
        if additional_context:
            context = f"--- Uploaded Documents ---\n{additional_context}\n\n--- Library Documents ---\n{context}"
            logger.debug("Added additional context", additional_context_length=len(additional_context))

        # =============================================================================
        # Phase 66: Add Step-Back Context (Background knowledge for complex queries)
        # =============================================================================
        if stepback_context and stepback_result:
            # Prepend background context from step-back prompting
            context = f"--- Background Knowledge ---\n{stepback_context}\n\n--- Specific Information ---\n{context}"
            logger.debug(
                "Phase 66: Added step-back background context",
                background_length=len(stepback_result.background_context),
                stepback_query=stepback_result.stepback_query[:50] if stepback_result.stepback_query else None,
            )

        # =============================================================================
        # Context Compression Pipeline (consolidates Phase 66, 79, 63)
        # =============================================================================
        context = await self._compress_context(
            context=context,
            question=question,
            use_context_compression=use_context_compression,
            settings_getter=_s,
            llm=llm,
        )

        # =============================================================================
        # Phase 63: Advanced Sufficiency Checker (ICLR 2025)
        # =============================================================================
        sufficiency_enabled = _s("rag.sufficiency_checker_enabled", False)
        if sufficiency_enabled:
            try:
                from backend.services.sufficiency_checker import SufficiencyChecker, SufficiencyLevel
                adv_checker = SufficiencyChecker()
                sufficiency = await adv_checker.check_sufficiency(
                    query=question, 
                    retrieved_chunks=[{"content": context}] 
)
                if sufficiency.level == SufficiencyLevel.SUFFICIENT:
                    logger.info(
                        "Advanced sufficiency check: context is sufficient",
                        confidence=sufficiency.confidence,
                        reasoning=sufficiency.reasoning[:100] if sufficiency.reasoning else None,
                    )
                elif sufficiency.level == SufficiencyLevel.PARTIAL:
                    logger.info(
                        "Advanced sufficiency check: context is partial",
                        missing_info=sufficiency.missing_information[:3] if sufficiency.missing_information else [],
                    )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Advanced sufficiency checker failed", error=str(e), error_type=type(e).__name__)

        # =============================================================================
        # PHASE 51: RLM Integration for Large Context Queries
        # =============================================================================
        # Check if context exceeds RLM threshold (default 100K tokens)
        # RLM excels at processing 10M+ token contexts with O(log N) complexity
        from backend.core.config import settings
        rlm_enabled = getattr(settings, 'ENABLE_RLM', True)
        rlm_threshold = getattr(settings, 'RLM_THRESHOLD_TOKENS', 100000)

        # Estimate context tokens (rough estimate: 4 chars per token)
        estimated_context_tokens = len(context) // 4

        use_rlm = (
            rlm_enabled
            and estimated_context_tokens > rlm_threshold
            and not getattr(runtime_settings, 'disable_rlm', False)
        )

        if use_rlm:
            logger.info(
                "Context exceeds RLM threshold - using Recursive Language Model",
                estimated_tokens=estimated_context_tokens,
                threshold=rlm_threshold,
            )
            try:
                from backend.services.recursive_lm import RecursiveLMService, RLMConfig

                # Configure RLM
                rlm_config = RLMConfig(
                    root_model=getattr(settings, 'RLM_ROOT_MODEL', 'gpt-4o'),
                    recursive_model=getattr(settings, 'RLM_RECURSIVE_MODEL', 'gpt-4o-mini'),
                    max_iterations=getattr(settings, 'RLM_MAX_ITERATIONS', 20),
                    timeout_seconds=getattr(settings, 'RLM_TIMEOUT_SECONDS', 120.0),
                    log_trajectory=getattr(settings, 'RLM_LOG_TRAJECTORY', False),
                )

                # Process with RLM
                rlm_service = RecursiveLMService(config=rlm_config)
                rlm_result = await rlm_service.process(
                    query=question,
                    context=context,
                )

                if rlm_result.success:
                    processing_time_ms = rlm_result.execution_time_ms

                    # Derive confidence from convergence speed and reasoning depth
                    max_iters = rlm_config.max_iterations or 20
                    convergence_ratio = 1.0 - (rlm_result.iterations / max_iters) if max_iters > 0 else 0.5
                    has_reasoning = len(rlm_result.reasoning_steps) > 0
                    rlm_confidence = min(0.95, 0.6 + (convergence_ratio * 0.3) + (0.05 if has_reasoning else 0.0))
                    rlm_confidence_level = "high" if rlm_confidence >= 0.8 else ("medium" if rlm_confidence >= 0.6 else "low")

                    # Return RLM response
                    return RAGResponse(
                        content=rlm_result.answer,
                        sources=sources,
                        query=question,
                        model=f"RLM({rlm_config.root_model})",
                        confidence_score=rlm_confidence,
                        confidence_level=rlm_confidence_level,
                        processing_time_ms=processing_time_ms,
                        suggested_questions=[],
                        context_used=min(len(context), 2000),  # Preview only
                        rlm_metadata={
                            "iterations": rlm_result.iterations,
                            "tokens_processed": rlm_result.tokens_processed,
                            "sandbox_type": rlm_result.sandbox_type,
                            "reasoning_steps": rlm_result.reasoning_steps[:3] if rlm_result.reasoning_steps else [],
                        },
                    )
                else:
                    logger.warning(
                        "RLM processing failed, falling back to standard RAG",
                        error=rlm_result.error,
                    )
                    # Continue with standard RAG below

            except (ValueError, RuntimeError, TimeoutError, ConnectionError, ImportError, MemoryError) as e:
                logger.warning(
                    "RLM integration error, falling back to standard RAG",
                    error=str(e),
                    error_type=type(e).__name__,
                    operation="rlm_integration",
                    query=question[:100],
                    user_id=user_id,
                    estimated_context_tokens=estimated_context_tokens,
                )
                # Continue with standard RAG below

        # Smart Model Routing: swap to cheaper/premium model based on query complexity
        try:
            from backend.services.smart_model_router import route_query_to_model
            _route = await route_query_to_model(
                question=question,
                query_classification=query_classification,
                context_length=len(context),
                num_documents=len(retrieved_docs) if retrieved_docs else 0,
                current_provider=llm_config.provider_type if llm_config else None,
                settings_getter=_s,
            )
            if _route.model and _route.provider:
                from backend.services.llm import LLMFactory
                routed_llm = LLMFactory.get_chat_model(
                    provider=_route.provider,
                    model=_route.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_response_tokens,
                )
                llm = routed_llm
                if llm_config:
                    llm_config.model = _route.model
                    llm_config.provider_type = _route.provider
                logger.info(
                    "Smart routing: swapped model",
                    tier=_route.tier.value,
                    new_model=_route.model,
                    reason=_route.reason,
                )
        except Exception as e:
            logger.debug("Smart model routing skipped", error=str(e))

        # Build prompt with language instruction
        # Support "auto" mode: respond in the same language as the question
        auto_detect = (language == "auto")
        effective_language = "en" if auto_detect else language
        language_instruction = _get_language_instruction(effective_language, auto_detect=auto_detect)

        # PHASE 14/15: Adaptive system prompt and template selection + sampling config
        # Select system prompt based on model size (tiny/small models need more explicit instructions)
        model_name = llm_config.model if llm_config else None
        system_prompt = _get_system_prompt_for_model(model_name)
        is_tiny = _is_tiny_model(model_name) if model_name else False
        is_llama = _is_llama_model(model_name) if model_name else False
        is_llama_sm = _is_llama_small(model_name) if model_name else False

        # Get research-backed sampling configuration (temperature, top_p, top_k, repeat_penalty)
        # Intent-based scaling: factual queries get lower temp, exploratory slightly higher
        # PHASE 15: Model-based + intent-based temperature optimization
        # This provides optimal temperature by default, but user can override via session config
        query_intent = query_classification.intent.value if query_classification else None
        sampling_config = _get_adaptive_sampling_config(model_name, query_intent)
        recommended_temp = sampling_config["temperature"]

        # Check if user has manually overridden temperature via session config
        # Use explicit flag if available, otherwise fall back to value comparison
        has_manual_override = (
            llm_config
            and (
                # Prefer explicit flag over value comparison
                getattr(llm_config, 'temperature_manual_override', False)
                or (
                    # Fall back to value check (but any value counts as override if explicitly set)
                    hasattr(llm_config, 'temperature')
                    and llm_config.temperature is not None
                    and getattr(llm_config, 'temperature_explicitly_set', False)
                )
            )
        )

        if has_manual_override:
            # User has set manual temperature - respect their choice but log it
            logger.info(
                "Using manual temperature override (Phase 15 optimization available)",
                manual_temp=llm_config.temperature,
                optimized_temp=recommended_temp,
                model=model_name,
                intent=query_intent,
            )
            # Keep the manual override in sampling_config
            sampling_config["temperature"] = llm_config.temperature
        else:
            # Use Phase 15 optimized temperature
            logger.debug(
                "Using Phase 15 optimized temperature",
                optimized_temp=recommended_temp,
                model=model_name,
                intent=query_intent,
            )

        # For tiny models (<3B), always use the structured template
        # This provides ultra-explicit format that reduces hallucination
        if is_tiny:
            # Llama 3.2 1B/3B uses step-by-step template, others use fixed-format
            prompt_template = _get_template_for_model(model_name)
            logger.info(
                "Using optimized template for small model",
                model=model_name,
                is_llama=is_llama,
                recommended_temperature=recommended_temp,
            )
        # Small Llama models (7B-8B) benefit from step-by-step reasoning
        elif is_llama_sm:
            prompt_template = LLAMA_SMALL_TEMPLATE
            logger.info(
                "Using Llama small model template with step-by-step reasoning",
                model=model_name,
                recommended_temperature=recommended_temp,
            )
        # For larger models, select prompt template based on query classification
        elif query_classification:
            use_cot = query_classification.use_cot
            intent_str = query_classification.intent.value if query_classification.intent else "factual"
            prompt_template = _get_template_for_intent(intent_str, use_cot=use_cot)
            logger.debug(
                "Selected adaptive prompt template",
                intent=intent_str,
                use_cot=use_cot,
                template_type=query_classification.prompt_template,
            )
        else:
            # Default template
            prompt_template = _get_template_for_model(model_name)

        if language_instruction:
            system_prompt = f"{system_prompt}\n{language_instruction}"

        # Phase 93: DSPy compiled prompt injection
        # When DSPy inference is enabled, override system prompt with compiled instructions
        # and inject few-shot demonstrations from the optimized module
        dspy_demos_messages = []
        dspy_inference_enabled = _s("rag.dspy_inference_enabled", False)
        if dspy_inference_enabled:
            try:
                dspy_instructions, dspy_demos = await self._load_dspy_compiled_state("rag_answer")
                if dspy_instructions:
                    # Prepend compiled instructions to system prompt
                    system_prompt = f"{dspy_instructions}\n\n{system_prompt}"
                    logger.info("DSPy compiled instructions injected into system prompt")
                if dspy_demos:
                    # Build few-shot demo messages (max 4 to avoid token overflow)
                    for demo in dspy_demos[:4]:
                        demo_q = demo.get("question", demo.get("query", ""))
                        demo_a = demo.get("answer", "")
                        if demo_q and demo_a:
                            demo_ctx = demo.get("context", "")
                            if demo_ctx:
                                dspy_demos_messages.append(
                                    HumanMessage(content=f"Context: {demo_ctx[:500]}\nQuestion: {demo_q}")
                                )
                            else:
                                dspy_demos_messages.append(HumanMessage(content=demo_q))
                            dspy_demos_messages.append(AIMessage(content=demo_a))
                    if dspy_demos_messages:
                        logger.info("DSPy few-shot demos injected", num_demos=len(dspy_demos_messages) // 2)
            except Exception as e:
                logger.debug("DSPy inference injection skipped", error=str(e))

        from backend.core.config import settings as app_settings

        if session_id:
            # Use conversational prompt with history
            memory = self._get_memory(session_id)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

            # PHASE 38: Apply context compression for long conversations
            compressed_context = ""
            if getattr(app_settings, "ENABLE_CONTEXT_COMPRESSION", False) and len(chat_history) > 10:
                try:
                    from backend.services.context_compression import get_context_compressor

                    compressor = await get_context_compressor()

                    # Convert chat_history to list of dicts
                    history_dicts = []
                    for msg in chat_history:
                        role = "user" if hasattr(msg, 'type') and msg.type == "human" else "assistant"
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        history_dicts.append({"role": role, "content": content})

                    compression_result = await compressor.compress_conversation(
                        messages=history_dicts,
                        session_id=session_id,
                    )

                    if compression_result.compressed_context:
                        compressed_context = compression_result.compressed_context
                        # Use compressed context instead of full history
                        chat_history = []  # Clear - we'll use compressed_context instead
                        logger.info(
                            "Applied context compression",
                            original_tokens=compression_result.original_tokens,
                            compressed_tokens=compression_result.compressed_tokens,
                            compression_ratio=compression_result.compression_ratio,
                        )
                except (ValueError, RuntimeError, TimeoutError, ConnectionError, ImportError) as e:
                    logger.warning(
                        "Context compression failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        operation="context_compression",
                        query=question[:100],
                        user_id=user_id,
                        session_id=session_id,
                        history_length=len(chat_history),
                    )

            # Build messages with optional compressed context
            # Phase 82: Use _make_system_message for Anthropic prompt caching
            if compressed_context:
                messages = [
                    _make_system_message(f"{system_prompt}\n\nPrevious Conversation Summary:\n{compressed_context}", model_name),
                    *dspy_demos_messages,  # Phase 93: DSPy few-shot demos
                    HumanMessage(content=f"{CONVERSATIONAL_RAG_TEMPLATE}\n\nQuestion: {question}".replace("{context}", context)),
                ]
            else:
                messages = [
                    _make_system_message(system_prompt, model_name),
                    *dspy_demos_messages,  # Phase 93: DSPy few-shot demos
                    *chat_history,
                    HumanMessage(content=f"{CONVERSATIONAL_RAG_TEMPLATE}\n\nQuestion: {question}".replace("{context}", context)),
                ]
        else:
            # Single-turn query with adaptive template
            messages = [
                _make_system_message(system_prompt, model_name),
                *dspy_demos_messages,  # Phase 93: DSPy few-shot demos
                HumanMessage(content=prompt_template.format(context=context, question=question)),
            ]

        # Generate response with research-backed sampling parameters
        # Apply temperature, top_p, top_k, repeat_penalty based on model characteristics
        # Research shows this dramatically reduces hallucinations in small models

        # PHASE 42: GenerativeCache - Check cache before LLM generation
        cache_hit = False
        cached_response = None
        if getattr(app_settings, "ENABLE_GENERATIVE_CACHE", False):
            try:
                from backend.services.generative_cache import get_generative_cache, ContentType

                gen_cache = await get_generative_cache()

                # Map query intent to cache content type
                cache_content_type = ContentType.DEFAULT
                if query_classification:
                    intent_map = {
                        "factual": ContentType.FACTUAL,
                        "analytical": ContentType.ANALYTICAL,
                        "creative": ContentType.CREATIVE,
                        "code": ContentType.CODE,
                        "conversational": ContentType.CONVERSATIONAL,
                    }
                    cache_content_type = intent_map.get(
                        query_classification.intent.value if query_classification.intent else "default",
                        ContentType.DEFAULT,
                    )

                cache_result = await gen_cache.get(
                    query=question,
                    context=context[:2000] if context else None,  # Use truncated context for cache key
                    content_type=cache_content_type,
                )

                if cache_result.hit:
                    cache_hit = True
                    cached_response = cache_result.response
                    logger.info(
                        "GenerativeCache hit",
                        tier=cache_result.tier.value,
                        similarity=cache_result.similarity,
                        lookup_time_ms=cache_result.lookup_time_ms,
                    )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError, KeyError) as e:
                logger.warning("GenerativeCache lookup failed", error=str(e), error_type=type(e).__name__)

        # If cache hit, return cached response directly
        if cache_hit and cached_response:
            processing_time_ms = (time.time() - start_time) * 1000

            # Parse suggested questions from cached response
            content, suggested_questions = _parse_suggested_questions(cached_response)

            # Get model name from config or fallback
            model_name_result = llm_config.model if llm_config else (self.config.chat_model or "default")

            return RAGResponse(
                content=content,
                sources=sources,
                query=question,
                model=f"{model_name_result} (cached)",
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                processing_time_ms=processing_time_ms,
                suggested_questions=suggested_questions,
                context_used=len(context) if context else 0,
                cache_hit=True,
            )

        # Speculative RAG: parallel draft generation (if enabled and enough documents)
        speculative_enabled = _s("rag.speculative_rag_enabled", False)
        if speculative_enabled and retrieved_docs and len(retrieved_docs) >= 4:
            try:
                from backend.services.speculative_rag import speculative_rag_generate
                from backend.services.smart_model_router import DEFAULT_TIER_MODELS, QueryTier

                # Use a cheaper model as drafter
                drafter_provider = llm_config.provider_type if llm_config else "openai"
                drafter_model_name = DEFAULT_TIER_MODELS.get(QueryTier.SIMPLE, {}).get(drafter_provider)
                drafter_llm = llm  # Fallback: same model
                if drafter_model_name:
                    try:
                        from backend.services.llm import LLMFactory
                        drafter_llm = LLMFactory.get_chat_model(
                            provider=drafter_provider,
                            model=drafter_model_name,
                            temperature=0.3,
                            max_tokens=self.config.max_response_tokens,
                        )
                    except Exception:
                        drafter_llm = llm

                num_drafts = int(_s("rag.speculative_rag_num_drafts", 3))
                spec_result = await speculative_rag_generate(
                    query=question,
                    documents=retrieved_docs,
                    drafter_llm=drafter_llm,
                    verifier_llm=llm,
                    full_context=context,
                    num_drafts=num_drafts,
                )

                if spec_result and spec_result.answer:
                    raw_content = spec_result.answer
                    logger.info(
                        "Speculative RAG produced answer",
                        drafts=spec_result.total_drafts,
                        selected=spec_result.selected_draft,
                    )
                    # Skip standard LLM invocation below
                    # Jump to post-processing (answer refinement, caching, etc.)
                    # We set a flag so the standard generation block is skipped
                    _speculative_answer = True
                else:
                    _speculative_answer = False
            except Exception as e:
                logger.debug("Speculative RAG failed, falling back to standard", error=str(e))
                _speculative_answer = False
        else:
            _speculative_answer = False

        # PHASE 15 OPTIONAL ENHANCEMENTS: Advanced optimizations
        from backend.services.rag_module.advanced_optimizations import (
            should_use_json_mode,
            invoke_with_json_mode,
            invoke_with_multi_sampling,
            get_telemetry,
        )
        from backend.services.rag_module.prompts import is_tiny_model

        if _speculative_answer:
            # Speculative RAG already produced raw_content above; skip standard LLM call
            use_json_mode = False
            use_multi_sampling = False
        else:
            # Determine which advanced optimizations to apply
            use_json_mode = should_use_json_mode(model_name, question)
            use_multi_sampling = is_tiny_model(model_name) and not use_json_mode  # Don't combine both

        if not _speculative_answer:
            try:
                # Try to apply sampling config if LLM supports it (most modern LLMs do)
                invoke_kwargs = {
                    "temperature": sampling_config["temperature"],
                }
                if sampling_config.get("top_p") is not None:
                    invoke_kwargs["top_p"] = sampling_config["top_p"]
                if sampling_config.get("top_k") is not None:
                    invoke_kwargs["top_k"] = sampling_config["top_k"]
                if sampling_config.get("repeat_penalty") is not None and sampling_config["repeat_penalty"] != 1.0:
                    invoke_kwargs["repeat_penalty"] = sampling_config["repeat_penalty"]

                # Apply advanced optimizations if appropriate
                if use_json_mode:
                    # Qwen model with structured query - use JSON mode
                    logger.info(
                        "Using JSON mode for structured output",
                        model=model_name,
                        query_preview=question[:50]
                    )
                    raw_content = await invoke_with_json_mode(
                        llm, messages, model_name, **invoke_kwargs
                    )
                elif use_multi_sampling:
                    # Tiny model - use multi-sampling for quality
                    logger.info(
                        "Using multi-sampling for tiny model",
                        model=model_name,
                        num_samples=3
                    )
                    raw_content = await invoke_with_multi_sampling(
                        llm, messages, model_name, num_samples=3, **invoke_kwargs
                    )
                else:
                    # Standard invocation with Phase 70 resilience patterns
                    provider_name = llm_config.provider_type if llm_config else "default"
                    try:
                        response = await resilient_llm_invoke(
                            llm, messages, provider=provider_name, **invoke_kwargs
                        )
                        raw_content = response.content if hasattr(response, 'content') else str(response)
                    except CircuitBreakerOpen as e:
                        logger.error("LLM circuit breaker open", provider=provider_name, error=str(e))
                        raise
                    except (ValueError, RuntimeError, TimeoutError, ConnectionError, asyncio.TimeoutError) as e:
                        logger.warning("Resilient LLM invoke failed, using direct call", error=str(e), error_type=type(e).__name__)
                        response = await llm.ainvoke(messages, **invoke_kwargs)
                        raw_content = response.content if hasattr(response, 'content') else str(response)

            except TypeError:
                # Fallback if LLM doesn't support these parameters
                logger.warning(
                    "LLM does not support sampling parameters, using defaults",
                    model=model_name,
                )
                # Phase 70: Use resilient invocation even for fallback
                provider_name = llm_config.provider_type if llm_config else "default"
                try:
                    response = await resilient_llm_invoke(llm, messages, provider=provider_name)
                    raw_content = response.content if hasattr(response, 'content') else str(response)
                except CircuitBreakerOpen:
                    raise
                except (ValueError, RuntimeError, TimeoutError, ConnectionError, asyncio.TimeoutError):
                    response = await llm.ainvoke(messages)
                    raw_content = response.content if hasattr(response, 'content') else str(response)

        # Phase 62: Answer Refinement
        # Use runtime settings for hot-reload (no server restart needed)
        refiner_enabled = _s("rag.answer_refiner_enabled", False)
        if refiner_enabled and raw_content:
            try:
                from backend.services.answer_refiner import AnswerRefiner, RefinerConfig
                refiner_strategy = _s("rag.answer_refiner_strategy", "self_refine")
                refiner_max_iter = _s("rag.answer_refiner_max_iterations", 2)
                refiner = AnswerRefiner(RefinerConfig(
                    strategy=refiner_strategy,
                    max_iterations=refiner_max_iter,
                ))
                refine_result = await refiner.refine(
                    query=question,
                    answer=raw_content,
                    context=context if context else "",
                )
                if refine_result.improvement_score > 0.1:
                    raw_content = refine_result.refined_answer
                    logger.info(
                        "Answer refined",
                        improvement=refine_result.improvement_score,
                        iterations=refine_result.iterations,
                        strategy=refiner_strategy,
                    )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Answer refinement failed", error=str(e), error_type=type(e).__name__)

        # Parse suggested questions from the response
        content, suggested_questions = _parse_suggested_questions(raw_content)

        # PHASE 42: Cache the generated response for future use
        if getattr(app_settings, "ENABLE_GENERATIVE_CACHE", False) and not cache_hit:
            try:
                from backend.services.generative_cache import get_generative_cache, ContentType

                gen_cache = await get_generative_cache()

                # Map query intent to cache content type
                cache_content_type = ContentType.DEFAULT
                if query_classification:
                    intent_map = {
                        "factual": ContentType.FACTUAL,
                        "analytical": ContentType.ANALYTICAL,
                        "creative": ContentType.CREATIVE,
                        "code": ContentType.CODE,
                        "conversational": ContentType.CONVERSATIONAL,
                    }
                    cache_content_type = intent_map.get(
                        query_classification.intent.value if query_classification.intent else "default",
                        ContentType.DEFAULT,
                    )

                await gen_cache.set(
                    query=question,
                    response=raw_content,  # Cache raw content including suggested questions
                    context=context[:2000] if context else None,
                    content_type=cache_content_type,
                    metadata={
                        "model": model_name,
                        "collection": collection_name,
                        "session_id": session_id,
                    },
                )
                logger.debug("Response cached in GenerativeCache")
            except Exception as e:
                logger.warning("Failed to cache response", error=str(e))

        processing_time_ms = (time.time() - start_time) * 1000

        # Track usage if enabled and we have config
        if self.track_usage and llm_config:
            # Accurate token counting using tiktoken
            input_text = RAG_SYSTEM_PROMPT + context + question
            input_tokens = self._count_tokens(input_text, llm_config.model)
            output_tokens = self._count_tokens(content, llm_config.model)

            await LLMUsageTracker.log_usage(
                provider_type=llm_config.provider_type,
                model=llm_config.model,
                operation_type="rag",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_id=llm_config.provider_id,
                user_id=user_id,
                session_id=session_id,
                duration_ms=int(processing_time_ms),
                success=True,
            )

        # Update memory if using session
        if session_id:
            memory = self._get_memory(session_id)
            memory.save_context(
                {"input": question},
                {"output": content}
            )

        # Get model name from config or fallback
        model_name = llm_config.model if llm_config else (self.config.chat_model or "default")

        # Extract confidence from verification
        confidence_score = verification_result.confidence_score if verification_result else None
        confidence_level = verification_result.confidence_level if verification_result else None

        # Generate confidence warning for UI display
        confidence_warning = ""
        crag_result = None
        if confidence_score is not None:
            if confidence_score < 0.2:
                confidence_warning = "Very low confidence - the retrieved documents may not be relevant. Consider rephrasing your question."
            elif confidence_score < 0.4:
                confidence_warning = "Low confidence - some retrieved documents may not fully address your question."
            elif confidence_score < self.config.crag_confidence_threshold:
                confidence_warning = "Moderate confidence - results may be incomplete. Try adding more specific terms."

        # Apply CRAG for low-confidence results (auto-refine query)
        if (
            self._crag is not None
            and confidence_score is not None
            and confidence_score < self.config.crag_confidence_threshold
            and len(sources) > 0
        ):
            try:
                # Convert sources to SearchResult format for CRAG
                from backend.services.vectorstore import SearchResult as VSSearchResult
                search_results = [
                    VSSearchResult(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        content=s.full_content,
                        score=s.relevance_score,
                        similarity_score=s.similarity_score,
                        document_title=s.document_name,
                        document_filename=s.document_name,
                        collection=s.collection,
                        page_number=s.page_number,
                        metadata=s.metadata,
                    )
                    for s in sources
                ]

                crag_result = await self._crag.process(
                    query=question,
                    search_results=search_results,
                    llm=llm,
                )

                if crag_result.action_taken == "refined_query" and crag_result.refined_query:
                    # Re-search with the refined query
                    logger.info(
                        "CRAG refined query - re-searching",
                        original_query=question[:50],
                        refined_query=crag_result.refined_query[:50],
                    )

                    # Re-search with refined query using _retrieve
                    new_retrieved_docs = await self._retrieve(
                        query=crag_result.refined_query,
                        collection_filter=collection_filter,
                        access_tier=access_tier,
                        top_k=top_k,
                        document_ids=document_ids,  # Maintain folder filtering if applicable
                        organization_id=organization_id,
                        user_id=user_id,
                        is_superadmin=is_superadmin,
                    )

                    if new_retrieved_docs and len(new_retrieved_docs) > 0:
                        # Convert to Source objects using _format_context
                        _, new_sources = self._format_context(new_retrieved_docs, include_collection_context)

                        if new_sources:
                            # Calculate new confidence from similarity scores
                            new_confidence = sum(s.similarity_score or s.relevance_score for s in new_sources) / len(new_sources)
                            if new_confidence > confidence_score:
                                # Use new results - also update context for response
                                sources = new_sources
                                confidence_score = new_confidence
                                # Regenerate context with new sources
                                context, _ = self._format_context(new_retrieved_docs, include_collection_context)
                                logger.info(
                                    "CRAG re-search improved results",
                                    new_source_count=len(sources),
                                    new_confidence=confidence_score,
                                )

                elif crag_result.action_taken == "filtered":
                    logger.info(
                        "CRAG filtered results",
                        original_count=len(sources),
                        filtered_count=len(crag_result.filtered_results),
                        new_confidence=crag_result.confidence,
                    )

                # Update confidence based on CRAG evaluation
                if crag_result.confidence > confidence_score:
                    confidence_score = crag_result.confidence
                    if confidence_score >= 0.7:
                        confidence_level = "high"
                        confidence_warning = ""
                    elif confidence_score >= 0.5:
                        confidence_level = "medium"
                        confidence_warning = ""

            except (ValueError, RuntimeError, TimeoutError, ConnectionError, TypeError, KeyError) as e:
                logger.warning(
                    "CRAG processing failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    operation="corrective_rag",
                    query=question[:100],
                    user_id=user_id,
                    confidence_score=confidence_score,
                )

        # Self-RAG: Verify response against sources to detect hallucinations
        self_rag_result: Optional[SelfRAGResult] = None
        if self._self_rag and content and sources:
            try:
                # Convert sources to SearchResult format for Self-RAG
                search_results_for_selfrag = [
                    SearchResult(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        content=s.full_content,
                        score=s.relevance_score,
                        similarity_score=s.similarity_score,
                        document_title=s.document_name,
                        document_filename=s.document_name,
                        collection=s.collection,
                        page_number=s.page_number,
                        metadata=s.metadata,
                    )
                    for s in sources
                ]

                self_rag_result = await self._self_rag.verify_response(
                    response=content,
                    sources=search_results_for_selfrag,
                    query=question,
                    llm=llm,
                )

                logger.info(
                    "Self-RAG verification complete",
                    overall_confidence=self_rag_result.overall_confidence,
                    supported_ratio=self_rag_result.supported_claim_ratio,
                    hallucinations=self_rag_result.hallucination_count,
                    needs_regeneration=self_rag_result.needs_regeneration,
                )

                # Update confidence based on Self-RAG verification
                if self_rag_result.overall_confidence < confidence_score:
                    confidence_score = self_rag_result.overall_confidence
                    if confidence_score < 0.5:
                        confidence_level = "low"
                        confidence_warning = f"Response may contain unverified claims. {self_rag_result.hallucination_count} potential hallucination(s) detected."
                    elif confidence_score < 0.7:
                        confidence_level = "medium"
                        if self_rag_result.hallucination_count > 0:
                            confidence_warning = f"Some claims could not be fully verified against sources."

            except (ValueError, RuntimeError, TimeoutError, ConnectionError, TypeError, KeyError) as e:
                logger.warning(
                    "Self-RAG verification failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    operation="self_rag_verification",
                    query=question[:100],
                    user_id=user_id,
                    sources_count=len(sources),
                    response_length=len(content) if content else 0,
                )

        # PHASE 15 OPTIONAL ENHANCEMENTS: Record telemetry
        telemetry = get_telemetry()
        telemetry.record_query(
            model_name=model_name,
            query=question,
            response=content,
            has_phase15=True,  # Phase 15 optimizations are always applied in this codebase
            used_json_mode=use_json_mode,
            used_multi_sampling=use_multi_sampling,
        )

        # =============================================================================
        # Phase 65: Cache the result in semantic cache for future similar queries
        # =============================================================================
        if self._phase65_enabled and self._phase65_pipeline and not phase65_cache_hit:
            try:
                # Cache the full response for semantic matching
                cache_data = {
                    "content": content,
                    "sources": [
                        {
                            "document_id": s.document_id,
                            "document_name": s.document_name,
                            "chunk_id": s.chunk_id,
                            "snippet": s.snippet,
                            "relevance_score": s.relevance_score,
                            "collection": s.collection,
                        }
                        for s in sources
                    ] if sources else [],
                    "model": model_name,
                    "confidence_score": confidence_score,
                    "confidence_level": confidence_level,
                }
                await self._phase65_pipeline.cache_result(
                    original_question,  # Use original (pre-spell-corrected) for cache key
                    cache_data,
                    query_embedding_for_cache,
                )
                logger.debug("Phase 65: Cached result in semantic cache")
            except Exception as e:
                logger.debug("Phase 65: Failed to cache result", error=str(e))

        # Phase 95J: Compute hallucination and confidence scores
        hallucination_score = await self._compute_hallucination_score(content, context)
        confidence_score_computed = await self._compute_confidence_score(
            sources,
            [s.similarity_score for s in sources],
            len(retrieved_docs)
        )

        response = RAGResponse(
            content=content,
            sources=sources if self.config.include_sources else [],
            query=question,
            model=model_name,
            processing_time_ms=processing_time_ms,
            metadata={
                "num_sources": len(sources),
                "session_id": session_id,
                "provider": llm_config.provider_type if llm_config else self.config.chat_provider,
                "config_source": llm_config.source if llm_config else "static",
            },
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            verification_result=verification_result,
            suggested_questions=suggested_questions,
            confidence_warning=confidence_warning,
            crag_result=crag_result,
            context_sufficiency=context_sufficiency_result,
        )
        response.metadata["hallucination_score"] = hallucination_score
        response.metadata["confidence_score_raw"] = confidence_score_computed
        return response

    async def query_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
        include_collection_context: bool = True,
        top_k: Optional[int] = None,  # Override documents to retrieve per query
        folder_id: Optional[str] = None,  # Folder to scope query to
        include_subfolders: bool = True,  # Include subfolders in folder-scoped query
        language: str = "en",  # Language for response
        enhance_query: Optional[bool] = None,  # Per-query override for query enhancement (expansion + HyDE)
        organization_id: Optional[str] = None,  # Organization ID for multi-tenant isolation
        is_superadmin: bool = False,  # Whether user is a superadmin
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Query RAG with streaming response.

        Args:
            question: User's question
            session_id: Session ID for memory
            collection_filter: Collection filter
            access_tier: User's access tier
            user_id: User ID for usage tracking and private document access
            include_collection_context: Whether to include collection tags in LLM context
            top_k: Optional override for number of documents to retrieve
            folder_id: Optional folder ID to restrict search to
            include_subfolders: When folder_id is set, include documents in subfolders
            language: Language code for response (en, de, es, fr, etc.)
            enhance_query: Per-query override for query enhancement (expansion + HyDE).
                           None = use config default, True = enable, False = disable.
            organization_id: Organization ID for multi-tenant isolation.
            is_superadmin: Whether user is a superadmin (can access all private docs).

        Yields:
            StreamChunk objects with response parts
        """
        import time
        start_time = time.time()

        # Load runtime settings from database if top_k not specified
        if top_k is None:
            runtime_settings = await self.get_runtime_settings()
            top_k = runtime_settings.get("top_k", self.config.top_k)

        logger.info(
            "Processing streaming RAG query",
            question_length=len(question),
            session_id=session_id,
            top_k=top_k,
        )

        # Get LLM for this session (with database-driven config)
        llm, llm_config = await self.get_llm_for_session(
            session_id=session_id,
            user_id=user_id,
        )

        # Resolve folder_id to document IDs if provided
        document_ids = None
        if folder_id:
            from backend.services.folder_service import get_folder_service
            folder_service = get_folder_service()
            document_ids = await folder_service.get_folder_document_ids(
                folder_id=folder_id,
                include_subfolders=include_subfolders,
                user_tier_level=access_tier,
            )
            if not document_ids:
                logger.info(
                    "No documents found in folder for streaming query",
                    folder_id=folder_id,
                    include_subfolders=include_subfolders,
                )
                yield StreamChunk(
                    type="error",
                    data="No documents found in the selected folder.",
                )
                return

        # Retrieve documents
        retrieved_docs = await self._retrieve(
            question,
            collection_filter=collection_filter,
            access_tier=access_tier,
            top_k=top_k,
            document_ids=document_ids,
            enhance_query=enhance_query,
            organization_id=organization_id,
            user_id=user_id,
            is_superadmin=is_superadmin,
        )

        context, sources = self._format_context(retrieved_docs, include_collection_context)

        # Build prompt with language instruction
        # Support "auto" mode: respond in the same language as the question
        auto_detect = (language == "auto")
        effective_language = "en" if auto_detect else language
        language_instruction = _get_language_instruction(effective_language, auto_detect=auto_detect)

        # PHASE 15: Apply model-specific prompts and sampling for streaming too
        model_name = llm_config.model if llm_config else None
        system_prompt = _get_system_prompt_for_model(model_name)
        if language_instruction:
            system_prompt = f"{system_prompt}\n{language_instruction}"

        # Get research-backed sampling configuration (temperature, top_p, top_k, repeat_penalty)
        # Note: streaming doesn't have query_classification, so intent will be None (uses base config)
        # PHASE 15: Model-based temperature optimization for streaming
        sampling_config = _get_adaptive_sampling_config(model_name, query_intent=None)

        # Check if user has manually overridden temperature via session config
        # Use explicit flag if available, otherwise fall back to value comparison
        has_manual_override = (
            llm_config
            and (
                # Prefer explicit flag over value comparison
                getattr(llm_config, 'temperature_manual_override', False)
                or (
                    # Fall back to value check (but any value counts as override if explicitly set)
                    hasattr(llm_config, 'temperature')
                    and llm_config.temperature is not None
                    and getattr(llm_config, 'temperature_explicitly_set', False)
                )
            )
        )

        if has_manual_override:
            # User has set manual temperature - respect their choice
            logger.info(
                "Streaming: Using manual temperature override",
                manual_temp=llm_config.temperature,
                optimized_temp=sampling_config["temperature"],
                model=model_name,
            )
            sampling_config["temperature"] = llm_config.temperature
        else:
            # Use Phase 15 optimized temperature
            logger.debug(
                "Streaming: Using Phase 15 optimized temperature",
                optimized_temp=sampling_config["temperature"],
                model=model_name,
            )

        # Select template based on model
        prompt_template = _get_template_for_model(model_name)

        if session_id:
            memory = self._get_memory(session_id)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

            messages = [
                _make_system_message(system_prompt, model_name),
                *chat_history,
                HumanMessage(content=f"{CONVERSATIONAL_RAG_TEMPLATE}\n\nQuestion: {question}".replace("{context}", context)),
            ]
        else:
            messages = [
                _make_system_message(system_prompt, model_name),
                HumanMessage(content=prompt_template.format(context=context, question=question)),
            ]

        # Stream response with sampling configuration - use StringIO for efficient string accumulation (O(n) vs O(n²))
        response_buffer = io.StringIO()
        try:
            # Apply sampling config if supported
            stream_kwargs = {
                "temperature": sampling_config["temperature"],
            }
            if sampling_config.get("top_p") is not None:
                stream_kwargs["top_p"] = sampling_config["top_p"]
            if sampling_config.get("top_k") is not None:
                stream_kwargs["top_k"] = sampling_config["top_k"]
            if sampling_config.get("repeat_penalty") is not None and sampling_config["repeat_penalty"] != 1.0:
                stream_kwargs["repeat_penalty"] = sampling_config["repeat_penalty"]

            # Phase 70: Use resilient streaming with circuit breaker
            provider_name = llm_config.provider_type if llm_config else "default"
            try:
                stream = await resilient_llm_stream(llm, messages, provider=provider_name, **stream_kwargs)
            except CircuitBreakerOpen as e:
                logger.error("LLM circuit breaker open for streaming", provider=provider_name, error=str(e))
                yield StreamChunk(type="error", data=f"Service temporarily unavailable: {str(e)}")
                return
            except TypeError:
                # Fallback if LLM doesn't support these parameters
                logger.warning(
                    "LLM does not support sampling parameters in streaming, using defaults",
                    model=model_name,
                )
                try:
                    stream = await resilient_llm_stream(llm, messages, provider=provider_name)
                except CircuitBreakerOpen as e:
                    yield StreamChunk(type="error", data=f"Service temporarily unavailable: {str(e)}")
                    return

            # =============================================================================
            # Phase 65: Streaming with Citations (optional)
            # =============================================================================
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            streaming_citations_enabled = await settings_svc.get_setting("rag.streaming_citations_enabled") or False

            if streaming_citations_enabled and self._phase65_enabled and self._phase65_pipeline and sources:
                try:
                    # Convert sources to dict format for citation matcher
                    source_dicts = [
                        {
                            "chunk_id": s.chunk_id,
                            "id": s.chunk_id,
                            "content": s.full_content or s.snippet,
                            "document_title": s.document_name,
                            "title": s.document_name,
                            "document_filename": s.document_name,
                        }
                        for s in sources
                    ]

                    # Create async generator from LLM stream
                    async def token_generator():
                        async for chunk in stream:
                            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                            if token:
                                yield token

                    # Wrap with citation matcher
                    async for enriched_chunk in self._phase65_pipeline.stream_with_citations(
                        token_generator(),
                        source_dicts,
                    ):
                        token = enriched_chunk.get("token", "")
                        citations = enriched_chunk.get("citations", [])
                        is_complete = enriched_chunk.get("is_complete", False)

                        if token:
                            response_buffer.write(token)
                            # Include citation info in chunk data
                            chunk_data = {
                                "content": token,
                                "citations": citations,
                                "is_cited": enriched_chunk.get("is_cited", False),
                            }
                            yield StreamChunk(type="content_with_citations", data=chunk_data)

                        if is_complete:
                            # Send citation footer if available
                            footer = enriched_chunk.get("footer")
                            if footer:
                                yield StreamChunk(type="citation_footer", data=footer)

                            summary = enriched_chunk.get("summary")
                            if summary:
                                yield StreamChunk(type="citation_summary", data=summary)

                    logger.info("Phase 65: Streaming with citations complete")

                except Exception as e:
                    logger.warning("Phase 65: Streaming citations failed, falling back to plain streaming", error=str(e))
                    # Fallback to plain streaming
                    async for chunk in stream:
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        if content:
                            response_buffer.write(content)
                            yield StreamChunk(type="content", data=content)
            else:
                # Standard streaming without citations
                async for chunk in stream:
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if content:
                        response_buffer.write(content)
                        yield StreamChunk(type="content", data=content)

            # Send sources after content
            if self.config.include_sources and sources:
                yield StreamChunk(type="sources", data=[
                    {
                        "document_id": s.document_id,
                        "document_name": s.document_name,
                        "page_number": s.page_number,
                        "relevance_score": s.relevance_score,
                        "similarity_score": s.similarity_score,  # Original vector similarity (0-1)
                        "snippet": s.snippet,
                        "full_content": s.full_content,
                        "collection": s.collection,
                    }
                    for s in sources
                ])

            # Get the full response from buffer
            full_response = response_buffer.getvalue()

            # Parse and yield suggested questions
            _, suggested_questions = _parse_suggested_questions(full_response)
            if suggested_questions:
                yield StreamChunk(type="suggestions", data=suggested_questions)

            # Update memory
            if session_id:
                memory = self._get_memory(session_id)
                memory.save_context(
                    {"input": question},
                    {"output": full_response}
                )

            # Track usage if enabled and we have config
            if self.track_usage and llm_config:
                processing_time_ms = (time.time() - start_time) * 1000
                input_text = RAG_SYSTEM_PROMPT + context + question
                # Accurate token counting using tiktoken
                input_tokens = self._count_tokens(input_text, llm_config.model)
                output_tokens = self._count_tokens(full_response, llm_config.model)

                await LLMUsageTracker.log_usage(
                    provider_type=llm_config.provider_type,
                    model=llm_config.model,
                    operation_type="rag",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider_id=llm_config.provider_id,
                    user_id=user_id,
                    session_id=session_id,
                    duration_ms=int(processing_time_ms),
                    success=True,
                )

            yield StreamChunk(type="done", data=None)

        except Exception as e:
            # Outermost catch-all for streaming: log full structured context
            logger.error(
                "Streaming error",
                error=str(e),
                error_type=type(e).__name__,
                operation="query_stream",
                query=question[:100],
                user_id=user_id,
                session_id=session_id,
                collection_filter=collection_filter,
                exc_info=True,
            )
            # Track failed usage
            if self.track_usage and llm_config:
                processing_time_ms = (time.time() - start_time) * 1000
                await LLMUsageTracker.log_usage(
                    provider_type=llm_config.provider_type,
                    model=llm_config.model,
                    operation_type="rag",
                    input_tokens=0,
                    output_tokens=0,
                    provider_id=llm_config.provider_id,
                    user_id=user_id,
                    session_id=session_id,
                    duration_ms=int(processing_time_ms),
                    success=False,
                    error_message=str(e),
                )
            yield StreamChunk(type="error", data=str(e))

    async def handle_aggregation_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        include_subfolders: bool = True,
        language: str = "en",
        organization_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> RAGResponse:
        """
        Handle queries requiring numerical aggregation (e.g., "What is the total spending by Company A in last 4 months?").

        Uses structured extraction to accurately extract and aggregate numerical values across documents.

        Args:
            question: User's aggregation query
            session_id: Session ID for conversation memory
            collection_filter: Filter by document collection
            access_tier: User's access tier for RLS
            user_id: User ID for usage tracking and private document access
            folder_id: Optional folder ID to scope query to
            include_subfolders: Whether to include documents from subfolders
            language: Language code for response
            organization_id: Organization ID for multi-tenant isolation
            is_superadmin: Whether user is a superadmin (can access all private docs)

        Returns:
            RAGResponse with aggregated answer and sources
        """
        import time
        start_time = time.time()

        from backend.services.structured_extraction import (
            get_structured_extractor,
            AggregationResult,
        )

        logger.info(
            "Processing aggregation query",
            question_length=len(question),
            session_id=session_id,
            collection_filter=collection_filter,
        )

        # Get LLM for this session
        llm, llm_config = await self.get_llm_for_session(
            session_id=session_id,
            user_id=user_id,
        )

        # Get folder-scoped document IDs if specified
        folder_document_ids = None
        if folder_id:
            from backend.services.folder_service import get_folder_service
            folder_service = get_folder_service()
            folder_document_ids = await folder_service.get_folder_document_ids(
                folder_id=folder_id,
                include_subfolders=include_subfolders,
                user_tier_level=access_tier,
            )
            if not folder_document_ids:
                return RAGResponse(
                    content="No documents found in the specified folder.",
                    sources=[],
                    query=question,
                    model=llm_config.model if llm_config else "default",
                    confidence_score=0.0,
                    confidence_level="low",
                )

        # Retrieve MORE documents for aggregation (need comprehensive data)
        # Use top_k of 20-30 for aggregation queries to gather more data points
        aggregation_top_k = 25
        retrieved_docs = await self._retrieve(
            question,
            collection_filter=collection_filter,
            access_tier=access_tier,
            top_k=aggregation_top_k,
            document_ids=folder_document_ids,
            enhance_query=True,  # Always enhance for aggregation
            organization_id=organization_id,
            user_id=user_id,
            is_superadmin=is_superadmin,
        )

        if not retrieved_docs:
            return RAGResponse(
                content="No relevant documents found to answer your aggregation query.",
                sources=[],
                query=question,
                model=llm_config.model if llm_config else "default",
                confidence_score=0.0,
                confidence_level="low",
            )

        # Convert retrieved docs to format expected by structured extractor
        chunks = []
        for doc, score in retrieved_docs:
            chunks.append({
                "content": doc.page_content,
                "chunk_id": doc.metadata.get("chunk_id"),
                "metadata": {
                    "document_name": doc.metadata.get("document_name", "Unknown"),
                    "document_id": doc.metadata.get("document_id"),
                    "page_number": doc.metadata.get("page_number"),
                    "similarity_score": score,
                },
            })

        # Use structured extractor for numerical extraction and aggregation
        extractor = get_structured_extractor()
        extractor._llm = llm  # Use the session's LLM

        try:
            # Extract and aggregate numerical values
            aggregation_result = await extractor.extract_and_aggregate(
                query=question,
                chunks=chunks,
            )

            # Generate natural language response
            response_text = await extractor.generate_aggregation_response(
                query=question,
                result=aggregation_result,
            )

            # Format sources from retrieved docs
            _, sources = self._format_context(retrieved_docs, include_collection_context=True)

            processing_time_ms = (time.time() - start_time) * 1000

            # Calculate confidence from aggregation result
            confidence_score = aggregation_result.confidence_score
            confidence_level = "high" if confidence_score >= 0.7 else "medium" if confidence_score >= 0.4 else "low"

            # Generate confidence warning
            confidence_warning = ""
            if aggregation_result.warnings:
                confidence_warning = " ".join(aggregation_result.warnings)

            # Track usage
            if self.track_usage and llm_config:
                input_tokens = self._count_tokens(question + str(chunks), llm_config.model)
                output_tokens = self._count_tokens(response_text, llm_config.model)

                await LLMUsageTracker.log_usage(
                    provider_type=llm_config.provider_type,
                    model=llm_config.model,
                    operation_type="aggregation_rag",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider_id=llm_config.provider_id,
                    user_id=user_id,
                    session_id=session_id,
                    duration_ms=int(processing_time_ms),
                    success=True,
                )

            # Update session memory
            if session_id:
                memory = self._get_memory(session_id)
                memory.save_context(
                    {"input": question},
                    {"output": response_text}
                )

            return RAGResponse(
                content=response_text,
                sources=sources if self.config.include_sources else [],
                query=question,
                model=llm_config.model if llm_config else "default",
                processing_time_ms=processing_time_ms,
                metadata={
                    "aggregation_type": aggregation_result.calculation_method,
                    "data_points": aggregation_result.count,
                    "total_value": aggregation_result.total,
                    "unit": aggregation_result.unit,
                    "breakdown_by_category": aggregation_result.breakdown_by_category,
                    "breakdown_by_period": aggregation_result.breakdown_by_period,
                    "breakdown_by_entity": aggregation_result.breakdown_by_entity,
                    "is_aggregation_query": True,
                },
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                confidence_warning=confidence_warning,
            )

        except Exception as e:
            logger.error("Aggregation query failed", error=str(e))
            # Fallback to standard RAG query
            logger.info("Falling back to standard RAG for aggregation query")
            return await self.query(
                question=question,
                session_id=session_id,
                collection_filter=collection_filter,
                access_tier=access_tier,
                user_id=user_id,
                folder_id=folder_id,
                include_subfolders=include_subfolders,
                language=language,
            )

    async def _compress_context(
        self,
        context: str,
        question: str,
        use_context_compression: bool,
        settings_getter,
        llm=None,
    ) -> str:
        """
        Unified context compression pipeline.

        Applies compression methods in priority order (only one will fire):
        1. Phase 66 ContextCompressor (LLM-based, for >4K context)
        2. Phase 79 AttentionRAG (attention-scoring, for >4K context)
        3. Phase 63 TTT Compression (for ultra-long >100K context)

        Returns the (possibly compressed) context string.
        """
        _s = settings_getter

        # Phase 66: LLM-based context compression
        if use_context_compression and len(context) > 4000:
            try:
                if _s("rag.context_compression_enabled") != False:
                    if self._context_compressor is None:
                        target_tokens = _s("rag.context_compression_target_tokens", 2000)
                        self._context_compressor = ContextCompressor(
                            target_tokens=target_tokens,
                            use_llm_compression=True,
                        )
                    compressed = await self._context_compressor.compress(
                        query=question,
                        contexts=[context],
                        llm=llm if _s("rag.context_compression_use_llm") else None,
                    )
                    if compressed.compression_ratio < 0.9:
                        logger.info(
                            "Context compression applied (Phase 66)",
                            original_tokens=compressed.original_tokens,
                            compressed_tokens=compressed.compressed_tokens,
                            ratio=compressed.compression_ratio,
                        )
                        context = compressed.compressed_context
            except (ValueError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.warning("Context compression failed (Phase 66)", error=str(e), error_type=type(e).__name__)

        # Phase 79: AttentionRAG compression (only if still large)
        if len(context) > 4000:
            try:
                if _s("rag.attention_rag_enabled"):
                    from backend.services.attention_rag import compress_context_with_attention, AttentionCompressionMode
                    mode_str = _s("rag.attention_rag_mode", "moderate")
                    mode_map = {
                        "light": AttentionCompressionMode.LIGHT,
                        "moderate": AttentionCompressionMode.MODERATE,
                        "aggressive": AttentionCompressionMode.AGGRESSIVE,
                    }
                    mode = mode_map.get(mode_str, AttentionCompressionMode.MODERATE)
                    compressed_text = await compress_context_with_attention(
                        query=question, context=context, mode=mode,
                    )
                    if compressed_text and len(compressed_text) < len(context) * 0.9:
                        logger.info(
                            "AttentionRAG compression applied (Phase 79)",
                            original_len=len(context),
                            compressed_len=len(compressed_text),
                            ratio=len(compressed_text) / len(context),
                        )
                        context = compressed_text
            except (ValueError, RuntimeError, TimeoutError, ConnectionError, ImportError) as e:
                logger.warning("AttentionRAG compression failed (Phase 79)", error=str(e), error_type=type(e).__name__)

        # Phase 63: TTT compression for ultra-long contexts
        from backend.core.config import settings as app_settings
        max_context_len = getattr(app_settings, 'MAX_CONTEXT_LENGTH', 100000)
        if _s("rag.ttt_compression_enabled", False) and len(context) > max_context_len:
            try:
                from backend.services.ttt_compression import TTTCompressionService
                compressor = TTTCompressionService()
                original_len = len(context)
                compressed = await compressor.compress(context, target_ratio=0.5)
                context = compressed.compressed_text if hasattr(compressed, 'compressed_text') else str(compressed)
                logger.info(
                    "TTT compression applied (Phase 63)",
                    original_len=original_len,
                    compressed_len=len(context),
                    ratio=len(context) / original_len,
                )
            except (ValueError, RuntimeError, TimeoutError, ConnectionError, MemoryError, ImportError) as e:
                logger.warning("TTT compression failed (Phase 63)", error=str(e), error_type=type(e).__name__)

        return context

    async def _retrieve(
        self,
        query: str,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        enhance_query: Optional[bool] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
        query_classification: Optional[QueryClassification] = None,
        use_multilingual_search: bool = False,
        target_language: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for query.

        Args:
            query: Search query
            collection_filter: Filter by collection
            access_tier: User's access tier
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter to (for folder-scoped queries)
            enhance_query: Per-query override for query enhancement (expansion + HyDE).
                           None = use config default, True = enable, False = disable.
            organization_id: Optional organization ID for multi-tenant isolation
            user_id: Optional user ID for private document access
            is_superadmin: Whether user is a superadmin (can access all private docs)
            query_classification: Optional query classification for adaptive retrieval
                                  (MMR diversity, KG enhancement, etc.)

        Returns:
            List of (document, score) tuples
        """
        top_k = top_k or self.config.top_k

        # Check cache first (for identical queries)
        cached_results = self._search_cache.get(query, collection_filter, access_tier, top_k)
        if cached_results is not None:
            logger.debug(
                "Using cached retrieval results",
                query_length=len(query),
                result_count=len(cached_results),
            )
            return cached_results

        # Phase 59: Multilingual cross-lingual search
        # Enables search across language barriers using multilingual embeddings
        if use_multilingual_search and MULTILINGUAL_SEARCH_AVAILABLE:
            try:
                logger.info(
                    "Using multilingual cross-lingual search",
                    query_preview=query[:50],
                    target_language=target_language,
                )

                # Get LLM for translations (optional)
                llm = None
                try:
                    llm, _ = await self.get_llm_for_session()
                except Exception:
                    pass  # Translation is optional

                # Create or get cross-lingual retriever
                ml_retriever = get_cross_lingual_retriever(
                    embedding_service=self._embedding_service,
                    vectorstore=self._custom_vectorstore,
                    llm=llm,
                )

                # Convert target_language to Language enum if provided
                user_lang = None
                if target_language and Language:
                    for lang in Language:
                        if lang.value == target_language:
                            user_lang = lang
                            break

                # Perform cross-lingual search
                async with async_session_context() as session:
                    ml_results = await ml_retriever.search(
                        query=query,
                        session=session,
                        top_k=top_k,
                        translate_results=True,
                        user_language=user_lang,
                    )

                # Convert CrossLingualSearchResult to (Document, score) tuples
                results = []
                for r in ml_results:
                    doc = Document(
                        page_content=r.translated_content or r.content,
                        metadata={
                            "document_id": r.document_id,
                            "chunk_id": r.chunk_id,
                            "source_language": r.source_language.value if r.source_language else "unknown",
                            "original_content": r.content if r.translated_content else None,
                            **r.metadata,
                        },
                    )
                    results.append((doc, r.similarity_score))

                logger.info(
                    "Multilingual search complete",
                    result_count=len(results),
                    languages_found=list(set(
                        r.source_language.value for r in ml_results if r.source_language
                    )),
                )

                # Cache results
                self._search_cache.set(query, collection_filter, access_tier, top_k, results)
                return results

            except Exception as e:
                logger.warning(
                    "Multilingual search failed, falling back to standard retrieval",
                    error=str(e),
                )

        logger.debug(
            "Starting retrieval",
            query_length=len(query),
            top_k=top_k,
            has_custom_vectorstore=self._custom_vectorstore is not None,
            enhance_query=enhance_query,
        )

        # Determine if query enhancement is enabled
        # Per-query override takes precedence, otherwise use config defaults
        use_hyde = enhance_query if enhance_query is not None else (self._hyde_expander is not None)
        use_expansion = enhance_query if enhance_query is not None else (self._query_expander is not None)

        # Build list of queries to search
        queries_to_search = [query]
        query_word_count = len(query.split())

        # Use HyDE for short/abstract queries (< 5 words by default)
        # HyDE generates a hypothetical document that might contain the answer
        hyde_query = None
        if (
            use_hyde
            and self._hyde_expander is not None
            and query_word_count < self.config.hyde_min_query_words
            and query_word_count >= 2  # Skip very short queries
        ):
            try:
                llm, _ = await self.get_llm_for_session()
                hyde_result = await self._hyde_expander.expand(query, llm)
                if hyde_result.hypothetical_document and hyde_result.hypothetical_document != query:
                    hyde_query = hyde_result.hypothetical_document
                    queries_to_search.append(hyde_query)
                    logger.info(
                        "HyDE expansion applied",
                        original_query=query,
                        hypothetical_preview=hyde_query[:100],
                    )
            except Exception as e:
                logger.warning("HyDE expansion failed, using original query", error=str(e))

        # Expand query if enabled (generates paraphrased versions for better recall)
        if use_expansion and self._query_expander is not None and self._query_expander.should_expand(query):
            try:
                expansion_result = await self._query_expander.expand_query(query)
                if expansion_result.expanded_queries:
                    queries_to_search.extend(expansion_result.expanded_queries)
                    logger.debug(
                        "Query expanded",
                        original=query,
                        expanded_count=len(expansion_result.expanded_queries),
                    )
            except Exception as e:
                logger.warning("Query expansion failed, using original query", error=str(e))

        # Phase 60: KG-based query expansion with related entities
        from backend.core.config import settings as app_settings
        if (
            getattr(app_settings, 'KG_ENABLED', True)
            and getattr(app_settings, 'KG_QUERY_EXPANSION_ENABLED', True)
        ):
            try:
                from backend.services.knowledge_graph import get_kg_service
                kg_service = await get_kg_service()

                # Extract entities from query
                entities, _ = await kg_service.extract_entities_from_text(
                    text=query,
                    use_fast_extraction=True,
                )

                if entities:
                    # Get related entity names for query expansion
                    max_entities = getattr(app_settings, 'KG_EXPANSION_MAX_ENTITIES', 5)
                    entity_names = [e.name for e in entities[:max_entities]]

                    # Create expanded query with entity names
                    kg_expansion = f"{query} {' '.join(entity_names)}"
                    if kg_expansion not in queries_to_search:
                        queries_to_search.append(kg_expansion)
                        logger.debug(
                            "KG-based query expansion applied",
                            original=query,
                            added_entities=entity_names,
                        )
            except Exception as e:
                logger.debug("KG query expansion skipped", error=str(e))

        # Collect results from all queries in PARALLEL for better performance
        all_results: List[Tuple[Document, float]] = []
        seen_chunk_ids: set = set()

        # Create search coroutines for parallel execution
        async def search_single_query(search_query: str) -> List[Tuple[Document, float]]:
            """Execute search for a single query."""
            if self._custom_vectorstore is not None:
                return await self._retrieve_with_custom_store(
                    query=search_query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
                    document_ids=document_ids,
                    organization_id=organization_id,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                    query_classification=query_classification,  # PHASE 14: Pass classification
                )
            elif self._vector_store is not None:
                return await self._retrieve_with_langchain_store(
                    query=search_query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
                    document_ids=document_ids,
                )
            else:
                return []

        # Execute all searches in parallel
        if len(queries_to_search) > 1:
            logger.info(
                "Parallel retrieval starting",
                num_queries=len(queries_to_search),
                has_hyde=hyde_query is not None,
                has_expansion=len(queries_to_search) > (2 if hyde_query else 1),
            )
            search_results = await asyncio.gather(
                *[search_single_query(q) for q in queries_to_search],
                return_exceptions=True,
            )

            # Process results from all queries
            for i, results in enumerate(search_results):
                if isinstance(results, Exception):
                    logger.warning(
                        "Query search failed",
                        query_index=i,
                        error=str(results),
                    )
                    continue

                # Deduplicate results by chunk_id
                for doc, score in results:
                    chunk_id = doc.metadata.get("chunk_id", id(doc))
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_results.append((doc, score))
        else:
            # Single query - no parallel execution needed
            logger.info(
                "RAG retrieval starting",
                has_custom_vectorstore=self._custom_vectorstore is not None,
                vectorstore_type=type(self._custom_vectorstore).__name__ if self._custom_vectorstore else None,
                collection_filter=collection_filter,
                has_folder_filter=document_ids is not None,
                organization_id=organization_id,
            )
            if self._custom_vectorstore is not None:
                results = await self._retrieve_with_custom_store(
                    query=query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
                    document_ids=document_ids,
                    organization_id=organization_id,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                    query_classification=query_classification,  # PHASE 14: Pass classification
                )
            elif self._vector_store is not None:
                results = await self._retrieve_with_langchain_store(
                    query=query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
                    document_ids=document_ids,
                )
            else:
                logger.warning("No vector store configured, returning mock results")
                return self._mock_retrieve(query, top_k)

            # Deduplicate results by chunk_id
            for doc, score in results:
                chunk_id = doc.metadata.get("chunk_id", id(doc))
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_results.append((doc, score))

        # Sort by score (descending) and limit to top_k
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Apply semantic deduplication if enabled (removes near-duplicate content)
        if self.config.enable_semantic_dedup and len(all_results) > 1 and len(queries_to_search) > 1:
            all_results = self._semantic_dedupe(all_results, self.config.semantic_dedup_threshold)

        if len(queries_to_search) > 1:
            logger.debug(
                "Query expansion retrieval complete",
                queries_searched=len(queries_to_search),
                unique_results=len(all_results),
                returning=min(top_k, len(all_results)),
            )

        # PHASE 43: Apply tiered reranking if enabled
        # Multi-stage pipeline: ColBERT -> Cross-Encoder -> (optional) LLM
        from backend.core.config import settings as app_settings
        if getattr(app_settings, "ENABLE_TIERED_RERANKING", False) and len(all_results) > 3:
            try:
                from backend.services.tiered_reranking import get_tiered_reranker

                reranker = await get_tiered_reranker()
                rerank_result = await reranker.rerank(
                    query=query,
                    candidates=all_results,
                    top_k=top_k,
                )

                if rerank_result.reranked_documents:
                    all_results = rerank_result.reranked_documents
                    logger.info(
                        "Tiered reranking applied",
                        original_count=len(all_results),
                        stages_used=rerank_result.stages_used,
                        latency_ms=rerank_result.latency_ms,
                    )
            except Exception as e:
                logger.warning("Tiered reranking failed, using original order", error=str(e))

        final_results = all_results[:top_k]

        # Cache the results for future identical queries
        self._search_cache.set(query, collection_filter, access_tier, top_k, final_results)

        return final_results

    async def _retrieve_with_custom_store(
        self,
        query: str,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
        query_classification: Optional[QueryClassification] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using our custom VectorStore service.

        Args:
            query: Search query
            collection_filter: Filter by collection (or "(Untagged)" for untagged docs)
            access_tier: User's access tier
            top_k: Number of results
            document_ids: Optional list of document IDs to filter to (for folder-scoped queries)
            organization_id: Optional organization ID for multi-tenant isolation
            user_id: Optional user ID for private document access
            is_superadmin: Whether user is a superadmin (can access all private docs)
            query_classification: Optional query classification for adaptive retrieval
                                  (MMR diversity, KG enhancement, etc.)

        Returns:
            List of (Document, score) tuples
        """
        try:
            # PHASE 12: Verbose logging for RAG debugging
            logger.info(
                "RAG _retrieve_with_custom_store ENTRY",
                query_preview=query[:80],
                collection_filter=collection_filter,
                access_tier=access_tier,
                top_k=top_k,
                document_ids_count=len(document_ids) if document_ids else 0,
                organization_id=organization_id,
                user_id=user_id,
                is_superadmin=is_superadmin,
            )

            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)

            # PHASE 12: Log embedding generation success
            logger.info(
                "Query embedding generated",
                embedding_dims=len(query_embedding) if query_embedding else 0,
                embedding_sample=query_embedding[:3] if query_embedding else None,
            )

            # Determine search type
            search_type = SearchType.HYBRID if self.config.use_hybrid_search else SearchType.VECTOR

            # PHASE 14: Use passed query classification or classify if not provided
            # Extract dynamic weights for hybrid search
            vector_weight = None
            keyword_weight = None

            # If classification was passed from caller, use it
            if query_classification is not None:
                vector_weight = query_classification.vector_weight
                keyword_weight = query_classification.keyword_weight
                logger.debug(
                    "Using passed query classification for adaptive retrieval",
                    intent=query_classification.intent.value,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    use_mmr=query_classification.use_mmr,
                    use_kg=query_classification.use_kg_enhancement,
                )
            # Otherwise classify locally (for direct calls to this method)
            elif self._query_classifier is not None and search_type == SearchType.HYBRID:
                query_classification = self._query_classifier.classify(query)
                vector_weight = query_classification.vector_weight
                keyword_weight = query_classification.keyword_weight
                logger.debug(
                    "Query classified locally for dynamic weighting",
                    query=query[:50],
                    intent=query_classification.intent.value,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    confidence=query_classification.confidence,
                )

            # Get document IDs matching the collection filter
            # Start with folder-filtered document IDs if provided
            filtered_document_ids = document_ids  # From folder filtering
            collection_document_ids = None

            if collection_filter:
                async with async_session_context() as db:
                    if collection_filter == "(Untagged)":
                        # Filter for untagged documents
                        query_stmt = select(DBDocument.id).where(
                            or_(
                                DBDocument.tags.is_(None),
                                DBDocument.tags == [],
                            )
                        )
                    else:
                        # Filter for documents with matching tag
                        # SQLite stores JSON arrays as text, so we use LIKE pattern
                        # SECURITY FIX: Escape special LIKE characters to prevent injection
                        from sqlalchemy import cast, String, literal

                        # Escape LIKE special characters: %, _, \
                        safe_filter = collection_filter.replace("\\", "\\\\")
                        safe_filter = safe_filter.replace("%", "\\%")
                        safe_filter = safe_filter.replace("_", "\\_")
                        # Also escape quotes to prevent breaking out of the pattern
                        safe_filter = safe_filter.replace('"', '\\"')

                        # e.g., tags = '["German", "Marketing"]' LIKE '%"German"%'
                        pattern = f'%"{safe_filter}"%'

                        # Cast tags column to String to bypass StringArrayType processing
                        query_stmt = select(DBDocument.id).where(
                            cast(DBDocument.tags, String).like(literal(pattern))
                        )
                    result = await db.execute(query_stmt)
                    collection_document_ids = [str(row[0]) for row in result.fetchall()]
                    logger.info(
                        "Collection filter results",
                        document_ids=collection_document_ids[:5] if collection_document_ids else [],
                        count=len(collection_document_ids),
                    )

                    if not collection_document_ids:
                        logger.debug(
                            "No documents match collection filter",
                            collection_filter=collection_filter,
                        )
                        return []

            # Merge folder and collection filters (intersection if both exist)
            if filtered_document_ids is not None and collection_document_ids is not None:
                # Both folder and collection filters - take intersection
                document_ids = list(set(filtered_document_ids) & set(collection_document_ids))
                if not document_ids:
                    logger.debug(
                        "No documents match both folder and collection filters",
                        folder_doc_count=len(filtered_document_ids),
                        collection_doc_count=len(collection_document_ids),
                    )
                    return []
                logger.info(
                    "Merged folder and collection filters",
                    folder_count=len(filtered_document_ids),
                    collection_count=len(collection_document_ids),
                    merged_count=len(document_ids),
                )
            elif filtered_document_ids is not None:
                # Only folder filter
                document_ids = filtered_document_ids
            elif collection_document_ids is not None:
                # Only collection filter
                document_ids = collection_document_ids
            else:
                # No filters - this is "All Documents" mode
                document_ids = None

            # Apply smart pre-filtering for large collections (when no filter = "All Documents")
            # This reduces search space from potentially 10k-100k docs to a manageable subset
            if document_ids is None and self._smart_filter is not None:
                should_filter, total_docs = await self._smart_filter.should_prefilter(
                    collection_filter=collection_filter,
                    document_ids=document_ids,
                )
                if should_filter:
                    logger.info(
                        "Applying smart pre-filtering for large collection",
                        total_docs=total_docs,
                        query_preview=query[:50],
                    )
                    filter_result = await self._smart_filter.filter_documents(
                        query=query,
                        max_candidates=self.config.smart_filter_max_candidates,
                        collection_filter=collection_filter,
                        access_tier=access_tier,
                        document_ids=document_ids,
                        query_embedding=query_embedding,
                    )
                    if filter_result.document_ids:
                        document_ids = filter_result.document_ids
                        logger.info(
                            "Smart pre-filter reduced search space",
                            original_docs=total_docs,
                            filtered_docs=len(document_ids),
                            filter_time_ms=filter_result.filter_time_ms,
                            strategy=filter_result.strategy_used.value,
                        )
                    else:
                        logger.warning(
                            "Smart pre-filter returned no candidates, using full search"
                        )

            # Perform search - use hybrid (LightRAG/RAPTOR), two-stage (ColBERT), or hierarchical retrieval
            # Priority: HybridRetriever > Two-stage > Hierarchical > Standard
            logger.info(
                "Calling vectorstore search",
                query_length=len(query),
                has_embedding=query_embedding is not None and len(query_embedding) > 0,
                search_type=search_type.value,
                document_ids_count=len(document_ids) if document_ids else 0,
                hybrid=self._use_hybrid_retriever,
                two_stage=self._two_stage_retriever is not None,
                hierarchical=self._hierarchical_retriever is not None,
            )

            # Phase 58: Use HybridRetriever for LightRAG + RAPTOR + WARP + ColPali fusion
            if self._use_hybrid_retriever:
                try:
                    # Lazy initialization of HybridRetriever
                    if self._hybrid_retriever is None:
                        self._hybrid_retriever = await get_hybrid_retriever(
                            vectorstore=self._custom_vectorstore,
                        )

                    hybrid_results, metrics = await self._hybrid_retriever.retrieve(
                        query=query,
                        query_embedding=query_embedding,
                        top_k=top_k,
                        document_ids=document_ids,
                        access_tier_level=access_tier,
                        vector_weight=vector_weight,
                        keyword_weight=keyword_weight,
                    )

                    # Convert HybridResult to SearchResult format
                    from backend.services.vectorstore import SearchResult
                    results = []
                    for hr in hybrid_results:
                        results.append(SearchResult(
                            chunk_id=hr.chunk_id,
                            document_id=hr.document_id,
                            content=hr.content,
                            score=hr.score,
                            similarity_score=hr.rerank_score or hr.score,
                            document_title=hr.document_title,
                            document_filename=hr.document_filename,
                            page_number=hr.page_number,
                            section_title=hr.section_title,
                            metadata={
                                **hr.metadata,
                                "retrieval_sources": [s.value for s in hr.sources],
                                "source_scores": hr.source_scores,
                            },
                        ))

                    logger.info(
                        "HybridRetriever search complete",
                        total_results=len(results),
                        sources_used=metrics.sources_used,
                        total_time_ms=round(metrics.total_time_ms, 2),
                    )
                except Exception as e:
                    logger.warning(
                        "HybridRetriever failed, falling back to standard search",
                        error=str(e),
                    )
                    # Fall through to standard search
                    results = None

                if results is not None:
                    pass  # Use hybrid results
                elif self._two_stage_retriever is not None:
                    # Fallback to two-stage
                    results = await self._two_stage_retriever.retrieve(
                        query=query,
                        query_embedding=query_embedding,
                        search_type=search_type,
                        top_k=top_k,
                        access_tier_level=access_tier,
                        document_ids=document_ids,
                        vector_weight=vector_weight,
                        keyword_weight=keyword_weight,
                    )
                else:
                    # Fallback to standard
                    results = await self._custom_vectorstore.search(
                        query=query,
                        query_embedding=query_embedding,
                        search_type=search_type,
                        top_k=top_k,
                        access_tier_level=access_tier,
                        document_ids=document_ids,
                        vector_weight=vector_weight,
                        keyword_weight=keyword_weight,
                        organization_id=organization_id,
                        user_id=user_id,
                        is_superadmin=is_superadmin,
                    )
            elif self._two_stage_retriever is not None:
                # Use two-stage retrieval with ColBERT reranking for better precision
                results = await self._two_stage_retriever.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    search_type=search_type,
                    top_k=top_k,
                    access_tier_level=access_tier,
                    document_ids=document_ids,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                )
            elif self._hierarchical_retriever is not None:
                # Use hierarchical retrieval for better document diversity
                results = await self._hierarchical_retriever.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    search_type=search_type,
                    top_k=top_k,
                    access_tier_level=access_tier,
                    document_ids=document_ids,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                )
            else:
                # Standard retrieval
                results = await self._custom_vectorstore.search(
                    query=query,
                    query_embedding=query_embedding,
                    search_type=search_type,
                    top_k=top_k,
                    access_tier_level=access_tier,
                    document_ids=document_ids,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    organization_id=organization_id,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                )

            # Convert SearchResult to LangChain Document format
            langchain_results = []
            logger.info(
                "Vector search returned results",
                total_results=len(results),
                first_result_doc_id=results[0].document_id if results else None,
            )
            # Note: The vectorstore already applies similarity threshold filtering
            # during the search phase. The scores here are RRF fusion scores for
            # hybrid search, which are on a different scale (typically 0.01-0.02).
            # We should NOT filter again here as it would incorrectly discard results.
            for result in results:
                doc = Document(
                    page_content=result.content,
                    metadata={
                        "document_id": result.document_id,
                        "document_name": result.document_filename or result.document_title or f"Document {result.document_id[:8]}",
                        "chunk_id": result.chunk_id,
                        "collection": result.collection,
                        "page_number": result.page_number,
                        "section_title": result.section_title,
                        "similarity_score": result.similarity_score,  # Original vector similarity (0-1)
                        # Context expansion
                        "prev_chunk_snippet": result.prev_chunk_snippet,
                        "next_chunk_snippet": result.next_chunk_snippet,
                        "chunk_index": result.chunk_index,
                        **result.metadata,
                    },
                )
                langchain_results.append((doc, result.score))

            logger.info(
                "Retrieved documents with custom store",
                num_results=len(langchain_results),
                search_type=search_type.value,
                query_intent=query_classification.intent.value if query_classification else None,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )

            # PHASE 14: Conditionally enhance with knowledge graph based on query classification
            # If classification suggests KG would help (entity queries, relationship queries)
            # or if KG is always enabled, apply enhancement
            should_use_kg = self._enable_knowledge_graph
            if query_classification is not None:
                # Use KG if classification indicates it would help OR if globally enabled
                should_use_kg = should_use_kg or query_classification.use_kg_enhancement
                if query_classification.use_kg_enhancement:
                    logger.debug(
                        "Query classification recommends KG enhancement",
                        intent=query_classification.intent.value,
                    )

            if should_use_kg:
                # PHASE 15: First, rerank existing results based on entity overlap
                langchain_results = await self._rerank_with_graph_augmentation(
                    query=query,
                    results=langchain_results,
                    organization_id=organization_id,
                )

                # Then, add additional KG-enhanced chunks
                langchain_results = await self._enhance_with_knowledge_graph(
                    query=query,
                    existing_results=langchain_results,
                    top_k=top_k,
                    organization_id=organization_id,
                    access_tier_level=access_tier,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                )

            # Phase 66: LazyGraphRAG for efficient community-level context
            # Provides high-level community summaries alongside chunk-level retrieval
            lazy_graphrag_context: Optional[LazyGraphContext] = None
            if self._enable_lazy_graphrag and should_use_kg:
                langchain_results, lazy_graphrag_context = await self._enhance_with_lazy_graphrag(
                    query=query,
                    existing_results=langchain_results,
                    top_k=top_k,
                    organization_id=organization_id,
                    collection_filter=collection_filter,
                )
                # Store context for later use in prompt building
                if lazy_graphrag_context:
                    self._last_lazy_graphrag_context = lazy_graphrag_context

            # PHASE 14: Apply MMR for diversity if query classification recommends it
            # MMR helps with comparison queries, summary queries where diverse sources are valuable
            if (
                query_classification is not None
                and query_classification.use_mmr
                and len(langchain_results) > top_k
            ):
                diversity_weight = query_classification.diversity_weight
                langchain_results = self._apply_mmr_to_results(
                    results=langchain_results,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    lambda_param=1.0 - diversity_weight,  # Convert diversity to lambda (1=pure relevance)
                )
                logger.info(
                    "Applied MMR for result diversity",
                    diversity_weight=diversity_weight,
                    original_count=len(langchain_results),
                    final_count=min(top_k, len(langchain_results)),
                )

            return langchain_results

        except Exception as e:
            logger.error("Custom vectorstore retrieval failed", error=str(e), exc_info=True)
            return []

    def _apply_mmr_to_results(
        self,
        results: List[Tuple[Document, float]],
        query_embedding: List[float],
        top_k: int,
        lambda_param: float = 0.7,
    ) -> List[Tuple[Document, float]]:
        """
        Apply Maximal Marginal Relevance (MMR) to diversify search results.

        MMR balances relevance to the query with diversity among selected results.
        Formula: MMR = λ * Relevance(doc, query) - (1-λ) * max(Similarity(doc, selected_docs))

        This is useful for:
        - Comparison queries: Need diverse sources to compare
        - Summary queries: Need broad coverage across documents
        - List queries: Need comprehensive enumeration from different sources

        Args:
            results: List of (Document, score) tuples
            query_embedding: Query embedding vector for relevance calculation
            top_k: Number of results to return
            lambda_param: Balance parameter (0=max diversity, 1=max relevance)

        Returns:
            Diversified list of (Document, score) tuples
        """
        if not results or len(results) <= top_k:
            return results

        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        # Try to get embeddings from document metadata
        doc_embeddings = []
        for doc, score in results:
            emb = doc.metadata.get("embedding")
            doc_embeddings.append(emb)

        # If no embeddings available, fall back to simple truncation
        if all(e is None for e in doc_embeddings):
            logger.debug("No embeddings in results for MMR, using original order")
            return results[:top_k]

        # Calculate relevance scores (similarity to query)
        relevance_scores = []
        for i, (doc, score) in enumerate(results):
            if doc_embeddings[i] is not None:
                rel = cosine_similarity(query_embedding, doc_embeddings[i])
            else:
                # Use the original score as fallback
                rel = score
            relevance_scores.append(rel)

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(results)))

        while len(selected_indices) < top_k and remaining_indices:
            best_idx = None
            best_mmr = float("-inf")

            for idx in remaining_indices:
                # Relevance term
                relevance = relevance_scores[idx]

                # Diversity term: max similarity to already selected docs
                if selected_indices and doc_embeddings[idx] is not None:
                    max_sim = 0.0
                    for sel_idx in selected_indices:
                        if doc_embeddings[sel_idx] is not None:
                            sim = cosine_similarity(doc_embeddings[idx], doc_embeddings[sel_idx])
                            max_sim = max(max_sim, sim)
                else:
                    max_sim = 0.0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        # Return selected results in MMR order
        return [results[i] for i in selected_indices]

    async def _retrieve_with_langchain_store(
        self,
        query: str,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using LangChain vector store.

        Args:
            query: Search query
            collection_filter: Filter by collection
            access_tier: User's access tier
            top_k: Number of results
            document_ids: Optional list of document IDs to restrict search to (from folder filtering)

        Returns:
            List of (Document, score) tuples
        """
        # Build filter for RLS
        filter_dict = {"access_tier": {"$lte": access_tier}}
        if collection_filter:
            filter_dict["collection"] = collection_filter
        if document_ids:
            filter_dict["document_id"] = {"$in": document_ids}

        try:
            # Vector similarity search
            results = await self._vector_store.asimilarity_search_with_score(
                query,
                k=top_k,
                filter=filter_dict,
            )

            # Filter by similarity threshold
            results = [
                (doc, score) for doc, score in results
                if score >= self.config.similarity_threshold
            ]

            logger.debug(
                "Retrieved documents with LangChain store",
                num_results=len(results),
                query_length=len(query),
                document_ids_filter=len(document_ids) if document_ids else 0,
            )

            return results

        except Exception as e:
            logger.error("LangChain retrieval failed", error=str(e))
            return []

    def _mock_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Return mock results for development without vector store."""
        mock_docs = [
            Document(
                page_content="The Q4 strategy focuses on digital transformation and customer experience improvements. Key initiatives include cloud migration, AI integration, and process automation.",
                metadata={
                    "document_id": "550e8400-e29b-41d4-a716-446655440001",
                    "document_name": "Q4 Strategy Presentation.pptx",
                    "chunk_id": "chunk-001",
                    "page_number": 5,
                    "access_tier": 50,
                },
            ),
            Document(
                page_content="Market analysis shows significant growth opportunities in the enterprise segment. Data-driven decision making is essential for modern marketing strategies.",
                metadata={
                    "document_id": "550e8400-e29b-41d4-a716-446655440002",
                    "document_name": "Marketing Report 2024.pdf",
                    "chunk_id": "chunk-002",
                    "page_number": 12,
                    "access_tier": 30,
                },
            ),
            Document(
                page_content="Previous stadium activations included experiential marketing booths, fan engagement zones, and digital interactive displays. These generated 150% increase in brand awareness.",
                metadata={
                    "document_id": "550e8400-e29b-41d4-a716-446655440003",
                    "document_name": "Event Case Studies.pptx",
                    "chunk_id": "chunk-003",
                    "page_number": 8,
                    "access_tier": 20,
                },
            ),
        ]

        return [(doc, 0.85 - i * 0.05) for i, doc in enumerate(mock_docs[:top_k])]

    async def _rerank_with_graph_augmentation(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        organization_id: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank chunks by boosting those with high entity overlap with query.

        Scoring formula:
        - Base score: vector similarity (unchanged)
        - Entity overlap: +0.2 per matching entity
        - Relationship bonus: +0.3 if entities are connected in graph

        This improves relevance scoring for entity-centric queries by:
        1. Extracting entities from the query
        2. Finding entities mentioned in each chunk
        3. Computing entity overlap and relationship scores
        4. Boosting chunk scores accordingly

        Args:
            query: Search query
            results: List of (document, score) tuples from vector search
            organization_id: Optional organization ID for multi-tenant isolation

        Returns:
            Reranked list of (document, augmented_score) tuples
        """
        if not results:
            return results

        try:
            # Import lazily to avoid circular import
            from backend.services.knowledge_graph import (
                get_knowledge_graph_service,
                EntityType,
            )

            async with async_session_context() as db:
                kg_service = await get_knowledge_graph_service(db)

                # Extract entities from query
                query_entities = await kg_service.find_entities_by_query(
                    query=query,
                    entity_types=None,
                    limit=20,  # Get top entities from query
                    query_language=None,
                    organization_id=organization_id,
                )
                query_entity_ids = {e.id for e in query_entities}

                if not query_entity_ids:
                    # No entities in query, return original results
                    return results

                # Process each chunk and compute augmented scores
                augmented_results = []
                for doc, base_score in results:
                    # Get chunk ID from metadata
                    chunk_id_str = doc.metadata.get("chunk_id")
                    if not chunk_id_str:
                        # No chunk ID, keep original score
                        augmented_results.append((doc, base_score))
                        continue

                    try:
                        chunk_id = uuid.UUID(chunk_id_str)

                        # Find entities mentioned in this chunk
                        chunk_entities_result = await db.execute(
                            select(Entity)
                            .join(EntityMention, EntityMention.entity_id == Entity.id)
                            .where(EntityMention.chunk_id == chunk_id)
                        )
                        chunk_entities = list(chunk_entities_result.scalars().all())
                        chunk_entity_ids = {e.id for e in chunk_entities}

                        # Compute entity overlap
                        overlap_count = len(query_entity_ids & chunk_entity_ids)
                        overlap_score = overlap_count * 0.2

                        # Compute relationship bonus (check if entities are connected)
                        relationship_bonus = 0.0
                        if overlap_count > 0:
                            # Check for relationships between query entities and chunk entities
                            for query_entity_id in query_entity_ids:
                                for chunk_entity_id in chunk_entity_ids:
                                    if query_entity_id != chunk_entity_id:
                                        # Check if these entities have a relationship
                                        relation_result = await db.execute(
                                            select(EntityRelation).where(
                                                or_(
                                                    and_(
                                                        EntityRelation.source_entity_id == query_entity_id,
                                                        EntityRelation.target_entity_id == chunk_entity_id,
                                                    ),
                                                    and_(
                                                        EntityRelation.source_entity_id == chunk_entity_id,
                                                        EntityRelation.target_entity_id == query_entity_id,
                                                    ),
                                                )
                                            ).limit(1)
                                        )
                                        if relation_result.scalar_one_or_none():
                                            relationship_bonus = 0.3
                                            break  # Only count once per chunk
                                    if relationship_bonus > 0:
                                        break

                        # Boost chunk score
                        augmented_score = base_score + overlap_score + relationship_bonus

                        # Update metadata to track augmentation
                        doc.metadata["graph_augmented"] = True
                        doc.metadata["entity_overlap_count"] = overlap_count
                        doc.metadata["has_relationship_bonus"] = relationship_bonus > 0

                        augmented_results.append((doc, augmented_score))

                    except Exception as e:
                        logger.warning(
                            "Failed to augment chunk with graph data",
                            chunk_id=chunk_id_str,
                            error=str(e)
                        )
                        # Keep original score if augmentation fails
                        augmented_results.append((doc, base_score))

                # Re-sort by augmented score (descending)
                augmented_results.sort(key=lambda x: x[1], reverse=True)

                logger.debug(
                    "Graph-augmented reranking complete",
                    query_entities=len(query_entity_ids),
                    chunks_processed=len(results),
                    chunks_boosted=sum(1 for doc, _ in augmented_results if doc.metadata.get("graph_augmented")),
                )

                return augmented_results

        except Exception as e:
            logger.warning(
                "Graph-augmented reranking failed, returning original results",
                error=str(e),
            )
            return results

    async def _enhance_with_knowledge_graph(
        self,
        query: str,
        existing_results: List[Tuple[Document, float]],
        top_k: int = 5,
        organization_id: Optional[str] = None,
        access_tier_level: int = 100,
        user_id: Optional[str] = None,
        is_superadmin: bool = False,
    ) -> List[Tuple[Document, float]]:
        """
        Enhance retrieval results with knowledge graph context.

        Uses the knowledge graph to find entities mentioned in the query,
        traverse relationships to discover related entities, and retrieve
        chunks that contain these entities but might not have high
        vector similarity to the query.

        This improves recall for entity-centric queries like:
        - "What did Company X do in 2024?"
        - "Who worked with Person Y on Project Z?"
        - "What are the relationships between A and B?"

        Args:
            query: Search query
            existing_results: Results from vector search
            top_k: Maximum additional chunks to add from KG
            organization_id: Optional organization ID for multi-tenant isolation
            access_tier_level: Maximum access tier level for filtering
            user_id: Optional user ID for private document access
            is_superadmin: Whether the user is a superadmin

        Returns:
            Merged results with KG-enhanced chunks (original + KG chunks)
        """
        if not self._enable_knowledge_graph:
            return existing_results

        try:
            # Import lazily to avoid circular import
            from backend.services.knowledge_graph import (
                get_knowledge_graph_service,
                GraphRAGContext,
            )

            async with async_session_context() as db:
                kg_service = await get_knowledge_graph_service(db)

                # Get graph-enhanced context with proper filtering
                graph_context: GraphRAGContext = await kg_service.graph_search(
                    query=query,
                    max_hops=self._knowledge_graph_max_hops,
                    top_k=top_k,
                    organization_id=organization_id,
                    access_tier_level=access_tier_level,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                )

                if not graph_context.chunks:
                    logger.debug(
                        "No additional chunks from knowledge graph",
                        query=query[:50],
                    )
                    return existing_results

                # Get existing chunk IDs to avoid duplicates
                existing_chunk_ids = set()
                for doc, _ in existing_results:
                    chunk_id = doc.metadata.get("chunk_id")
                    if chunk_id:
                        existing_chunk_ids.add(str(chunk_id))

                # Convert KG chunks to LangChain Document format
                kg_results: List[Tuple[Document, float]] = []
                for chunk in graph_context.chunks:
                    if str(chunk.id) in existing_chunk_ids:
                        continue  # Skip duplicates

                    # Build metadata from chunk attributes (Chunk model doesn't have metadata dict)
                    doc = Document(
                        page_content=chunk.content,
                        metadata={
                            "document_id": str(chunk.document_id),
                            "document_name": f"Document {str(chunk.document_id)[:8]}",
                            "chunk_id": str(chunk.id),
                            "page_number": chunk.page_number,
                            "section_title": chunk.section_title,
                            "chunk_index": chunk.chunk_index,
                            "source": "knowledge_graph",  # Mark as KG-enhanced
                            "kg_entities": [e.name for e in graph_context.entities[:5]],  # Top entities
                        },
                    )

                    # KG chunks get a boosted score but lower than top vector results
                    # This ensures they're included but don't dominate
                    base_score = 0.5 if not existing_results else min(r[1] for r in existing_results) * 0.9
                    kg_results.append((doc, base_score))

                # Merge results: keep all original + add KG results up to limit
                merged = list(existing_results)
                space_for_kg = max(0, top_k - len(merged))
                merged.extend(kg_results[:space_for_kg])

                logger.info(
                    "Knowledge graph enhancement complete",
                    original_count=len(existing_results),
                    kg_chunks_found=len(graph_context.chunks),
                    kg_chunks_added=min(len(kg_results), space_for_kg),
                    entities_found=len(graph_context.entities),
                    relations_found=len(graph_context.relations),
                )

                return merged

        except Exception as e:
            logger.warning(
                "Knowledge graph enhancement failed, returning original results",
                error=str(e),
            )
            return existing_results

    async def _enhance_with_lazy_graphrag(
        self,
        query: str,
        existing_results: List[Tuple[Document, float]],
        top_k: int = 5,
        organization_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
    ) -> Tuple[List[Tuple[Document, float]], Optional[LazyGraphContext]]:
        """
        Enhance retrieval with LazyGraphRAG for cost-efficient knowledge graph context.

        LazyGraphRAG performs query-time community summarization instead of index-time,
        achieving 99% cost reduction compared to standard GraphRAG while maintaining
        similar or better answer quality.

        Key features:
        - Identifies query-relevant entities
        - Discovers their communities on-demand
        - Summarizes only relevant communities (lazy evaluation)
        - Returns community context for LLM generation

        Args:
            query: Search query
            existing_results: Results from vector search
            top_k: Maximum chunks to consider
            organization_id: Optional organization ID for multi-tenant isolation
            collection_filter: Optional collection filter

        Returns:
            Tuple of (enhanced results, LazyGraphContext with community summaries)
        """
        if not self._enable_lazy_graphrag:
            return existing_results, None

        try:
            from backend.db.database import async_session_context

            async with async_session_context() as db:
                # Initialize LazyGraphRAG service if needed
                if self._lazy_graphrag is None:
                    self._lazy_graphrag = get_lazy_graphrag_service()

                # Retrieve lazy graph context
                lazy_context: LazyGraphContext = await lazy_graph_retrieve(
                    query=query,
                    session=db,
                    collection_filter=collection_filter,
                    organization_id=organization_id,
                )

                logger.info(
                    "LazyGraphRAG enhancement complete",
                    query=query[:50],
                    entities=lazy_context.total_entities,
                    relations=lazy_context.total_relations,
                    communities_summarized=len(lazy_context.community_summaries),
                    cost_saved_pct=f"{lazy_context.cost_saved_percentage:.1f}%",
                )

                # LazyGraphRAG provides context for generation, not additional chunks
                # The community summaries and entity context enhance the LLM prompt
                return existing_results, lazy_context

        except Exception as e:
            logger.warning(
                "LazyGraphRAG enhancement failed, continuing without it",
                error=str(e),
            )
            return existing_results, None

    def _semantic_dedupe(
        self,
        results: List[Tuple[Document, float]],
        threshold: float = 0.95,
    ) -> List[Tuple[Document, float]]:
        """
        Remove semantically duplicate chunks from results.

        Uses MinHash (locality-sensitive hashing) for efficient approximate
        Jaccard similarity. This reduces O(n²) pairwise comparisons to O(n)
        by using fixed-size hash signatures.

        For small result sets (<50), falls back to exact Jaccard for accuracy.

        Args:
            results: List of (document, score) tuples, already sorted by score desc
            threshold: Similarity threshold (0-1), above which chunks are considered duplicates

        Returns:
            Deduplicated list of results
        """
        if len(results) <= 1:
            return results

        def get_ngrams(text: str, n: int = 3) -> set:
            """Get character n-grams from text."""
            text = text.lower().replace(" ", "")
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        # For small result sets, use exact Jaccard (more accurate)
        if len(results) < 50:
            return self._semantic_dedupe_exact(results, threshold, get_ngrams)

        # For larger result sets, use MinHash approximation (O(n) vs O(n²))
        return self._semantic_dedupe_minhash(results, threshold, get_ngrams)

    def _semantic_dedupe_exact(
        self,
        results: List[Tuple[Document, float]],
        threshold: float,
        get_ngrams,
    ) -> List[Tuple[Document, float]]:
        """Exact Jaccard deduplication for small result sets."""

        def jaccard_similarity(set1: set, set2: set) -> float:
            """Calculate Jaccard similarity between two sets."""
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0

        deduped = []
        seen_ngrams: List[set] = []

        for doc, score in results:
            content = doc.page_content
            ngrams = get_ngrams(content)

            # Check similarity against already-included results
            is_duplicate = False
            for seen in seen_ngrams:
                if jaccard_similarity(ngrams, seen) > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append((doc, score))
                seen_ngrams.append(ngrams)

        if len(deduped) < len(results):
            logger.debug(
                "Semantic deduplication removed duplicates (exact)",
                original_count=len(results),
                deduped_count=len(deduped),
                removed=len(results) - len(deduped),
            )

        return deduped

    def _semantic_dedupe_minhash(
        self,
        results: List[Tuple[Document, float]],
        threshold: float,
        get_ngrams,
        num_hashes: int = 128,
    ) -> List[Tuple[Document, float]]:
        """
        MinHash-based deduplication for large result sets.

        Uses locality-sensitive hashing to approximate Jaccard similarity
        in O(n) time instead of O(n²).

        Args:
            results: List of (document, score) tuples
            threshold: Similarity threshold for deduplication
            get_ngrams: Function to extract n-grams from text
            num_hashes: Number of hash functions (more = more accurate but slower)
        """
        import random

        # Generate hash function parameters (a*x + b mod p)
        # Use a large prime and fixed seed for reproducibility
        random.seed(42)
        MAX_HASH = (1 << 32) - 1
        PRIME = 4294967311  # Large prime > MAX_HASH

        hash_params = [
            (random.randint(1, MAX_HASH), random.randint(0, MAX_HASH))
            for _ in range(num_hashes)
        ]

        def compute_minhash(ngrams: set) -> List[int]:
            """Compute MinHash signature for a set of n-grams."""
            if not ngrams:
                return [MAX_HASH] * num_hashes

            signature = []
            for a, b in hash_params:
                min_hash = MAX_HASH
                for ngram in ngrams:
                    # Hash the n-gram using polynomial rolling hash
                    h = hash(ngram) & MAX_HASH
                    # Apply hash function
                    hash_val = ((a * h + b) % PRIME) & MAX_HASH
                    min_hash = min(min_hash, hash_val)
                signature.append(min_hash)
            return signature

        def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
            """Estimate Jaccard similarity from MinHash signatures."""
            if not sig1 or not sig2:
                return 0.0
            matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
            return matches / len(sig1)

        deduped = []
        seen_signatures: List[List[int]] = []

        for doc, score in results:
            content = doc.page_content
            ngrams = get_ngrams(content)
            signature = compute_minhash(ngrams)

            # Check similarity against already-included results
            is_duplicate = False
            for seen in seen_signatures:
                if minhash_similarity(signature, seen) > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append((doc, score))
                seen_signatures.append(signature)

        if len(deduped) < len(results):
            logger.debug(
                "Semantic deduplication removed duplicates (minhash)",
                original_count=len(results),
                deduped_count=len(deduped),
                removed=len(results) - len(deduped),
            )

        return deduped

    # -------------------------------------------------------------------------
    # Phase 95J: Hallucination + Confidence Scoring
    # -------------------------------------------------------------------------

    async def _compute_hallucination_score(self, answer: str, context: str) -> float:
        """
        Score 0.0 (fully grounded) to 1.0 (hallucinated) using reranker similarity.
        Compares the answer text against the source context.
        """
        try:
            if not answer or not context:
                return 0.5
            # Use reranker to check how well the answer aligns with context
            if hasattr(self, '_reranker') and self._reranker:
                score = await self._rerank_single(answer, context)
                return round(1.0 - score, 3)
            # Fallback: simple word overlap check
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            if not answer_words:
                return 0.5
            overlap = len(answer_words & context_words) / len(answer_words)
            return round(1.0 - min(overlap, 1.0), 3)
        except Exception as e:
            logger.warning("Hallucination scoring failed", error=str(e))
            return 0.5

    async def _compute_confidence_score(self, sources: list, rerank_scores: list, retrieval_count: int) -> float:
        """
        Multi-signal confidence from retrieval quality.
        Returns 0.0-1.0 score.
        """
        try:
            signals = []
            # Signal 1: source count (5+ sources = max confidence)
            if sources:
                signals.append(min(len(sources) / 5.0, 1.0))
            # Signal 2: average reranker score
            if rerank_scores:
                avg_rerank = sum(rerank_scores) / len(rerank_scores)
                signals.append(min(avg_rerank, 1.0))
            # Signal 3: retrieval density
            if retrieval_count > 0:
                signals.append(min(retrieval_count / 10.0, 1.0))
            if not signals:
                return 0.5
            return round(sum(signals) / len(signals), 3)
        except Exception as e:
            logger.warning("Confidence scoring failed", error=str(e))
            return 0.5

    def _format_context(
        self,
        retrieved_docs: List[Tuple[Document, float]],
        include_collection_context: bool = True,
        freshness_config: Optional[dict] = None,
    ) -> Tuple[str, List[Source]]:
        """
        Format retrieved documents into context string and sources list.

        Args:
            retrieved_docs: List of (document, score) tuples
            include_collection_context: Whether to include collection tags in LLM context
            freshness_config: Optional dict with freshness scoring settings
                (keys: enabled, decay_days, boost_factor, penalty_factor)

        Returns:
            Tuple of (context_string, sources_list)
        """
        if not retrieved_docs:
            return "No relevant documents found.", []

        # Phase 95K: Content freshness scoring
        if freshness_config and freshness_config.get("enabled"):
            try:
                decay_days = int(freshness_config.get("decay_days", 180))
                boost = float(freshness_config.get("boost_factor", 1.05))
                penalty = float(freshness_config.get("penalty_factor", 0.95))
                now = datetime.now()
                adjusted_docs = []
                for doc, score in retrieved_docs:
                    updated = doc.metadata.get("updated_at") or doc.metadata.get("created_at")
                    if updated:
                        try:
                            if isinstance(updated, str):
                                updated = datetime.fromisoformat(updated.replace("Z", "+00:00")).replace(tzinfo=None)
                            age_days = (now - updated).days
                            if age_days <= 30:
                                score *= boost
                            elif age_days > decay_days:
                                score *= penalty
                        except (ValueError, TypeError):
                            pass
                    adjusted_docs.append((doc, score))
                retrieved_docs = sorted(adjusted_docs, key=lambda x: x[1], reverse=True)
            except Exception as e:
                logger.debug("Freshness scoring skipped", error=str(e))

        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(retrieved_docs, 1):
            metadata = doc.metadata or {}

            # Build context entry - check document_name first (set by custom vectorstore)
            doc_name = metadata.get("document_name") or metadata.get("document_filename") or f"Document {metadata.get('document_id', 'unknown')[:8]}"
            collection = metadata.get("collection")

            # Include collection info if enabled and available
            collection_info = ""
            if include_collection_context and collection:
                collection_info = f" [Collection: {collection}]"

            page_info = ""
            if metadata.get("page_number"):
                page_info = f" (Page {metadata['page_number']})"
            elif metadata.get("slide_number"):
                page_info = f" (Slide {metadata['slide_number']})"

            context_parts.append(
                f"[Source {i}: {doc_name}{collection_info}{page_info}]\n{doc.page_content}\n"
            )

            # Build source citation (always include collection for UI display)
            # Get original similarity score from metadata (0-1), fallback to score
            similarity = metadata.get("similarity_score", score)
            source = Source(
                document_id=metadata.get("document_id", f"doc-{i}"),
                document_name=doc_name,
                chunk_id=metadata.get("chunk_id", f"chunk-{i}"),
                collection=collection,
                page_number=metadata.get("page_number"),
                slide_number=metadata.get("slide_number"),
                relevance_score=score,  # RRF score for ranking
                similarity_score=similarity,  # Original vector similarity (0-1) for display
                snippet=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                full_content=doc.page_content,  # Full content for source viewer modal
                metadata=metadata,
                # Context expansion (surrounding chunks)
                prev_chunk_snippet=metadata.get("prev_chunk_snippet"),
                next_chunk_snippet=metadata.get("next_chunk_snippet"),
                chunk_index=metadata.get("chunk_index"),
            )
            sources.append(source)

        context = "\n".join(context_parts)

        # Phase 66: Add LazyGraphRAG community context if available
        # Community summaries provide high-level thematic context beyond individual chunks
        if self._last_lazy_graphrag_context and self._last_lazy_graphrag_context.community_summaries:
            lazy_context = self._last_lazy_graphrag_context
            community_context_parts = [
                "\n[Knowledge Graph Community Context]",
                f"Entities identified: {lazy_context.total_entities}",
                f"Relationships mapped: {lazy_context.total_relations}",
                ""
            ]

            for i, summary in enumerate(lazy_context.community_summaries[:3], 1):
                community_context_parts.append(f"Community {i}: {summary}")

            if lazy_context.entity_context:
                community_context_parts.append(f"\n{lazy_context.entity_context}")

            context = context + "\n" + "\n".join(community_context_parts)

            # Clear the context after use to avoid stale data
            self._last_lazy_graphrag_context = None

            logger.debug(
                "Added LazyGraphRAG community context to prompt",
                communities=len(lazy_context.community_summaries),
                entities=lazy_context.total_entities,
            )

        return context, sources

    def set_vector_store(self, vector_store: Any):
        """Set vector store for retrieval."""
        self._vector_store = vector_store

    # -------------------------------------------------------------------------
    # Phase 93: DSPy Compiled Prompt Loading
    # -------------------------------------------------------------------------

    async def _load_dspy_compiled_state(
        self, signature_name: str = "rag_answer"
    ) -> tuple:
        """
        Load the latest deployed DSPy compiled state from the database.

        Returns:
            Tuple of (compiled_instructions: str | None, compiled_demos: list | None)
        """
        try:
            from backend.db.models import DSPyOptimizationJob
            from sqlalchemy import select

            async with async_session_context() as db:
                result = await db.execute(
                    select(DSPyOptimizationJob)
                    .where(
                        DSPyOptimizationJob.signature_name == signature_name,
                        DSPyOptimizationJob.status.in_(["deployed", "completed"]),
                        DSPyOptimizationJob.compiled_state.isnot(None),
                    )
                    .order_by(DSPyOptimizationJob.created_at.desc())
                    .limit(1)
                )
                job = result.scalar_one_or_none()

                if job and job.compiled_state:
                    instructions = job.compiled_state.get("instructions", "")
                    demos = job.compiled_state.get("demos", [])
                    if instructions or demos:
                        logger.info(
                            "Loaded DSPy compiled state",
                            signature=signature_name,
                            has_instructions=bool(instructions),
                            num_demos=len(demos),
                            job_id=str(job.id),
                        )
                        return instructions, demos

        except Exception as e:
            logger.debug("DSPy compiled state not available", error=str(e))

        return None, None

    # -------------------------------------------------------------------------
    # Advanced RAG Methods (GraphRAG, Agentic RAG)
    # -------------------------------------------------------------------------

    async def enhanced_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
        use_graph: bool = True,
        use_agentic: bool = False,
        db_session=None,
    ) -> RAGResponse:
        """
        Enhanced RAG query with optional GraphRAG and Agentic RAG.

        Args:
            question: User's question
            session_id: Session ID for memory
            collection_filter: Collection filter
            access_tier: User's access tier
            user_id: User ID for tracking
            use_graph: Enable GraphRAG (knowledge graph)
            use_agentic: Enable Agentic RAG (for complex queries)
            db_session: Database session for graph operations

        Returns:
            RAGResponse with answer and sources
        """
        import time
        start_time = time.time()

        # Check if we should use agentic RAG
        if use_agentic:
            try:
                from services.agentic_rag import get_agentic_rag_service

                # Initialize agentic RAG
                graph_service = None
                if use_graph and db_session:
                    from services.knowledge_graph import get_knowledge_graph_service
                    graph_service = await get_knowledge_graph_service(db_session)

                agentic_service = get_agentic_rag_service(
                    rag_service=self,
                    knowledge_graph_service=graph_service,
                )

                result = await agentic_service.process_query(
                    query=question,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    user_id=user_id,
                )

                # Convert to RAGResponse
                return RAGResponse(
                    content=result.final_answer,
                    sources=[],  # Agentic RAG handles sources differently
                    query=question,
                    model="agentic",
                    processing_time_ms=result.processing_time_ms,
                    metadata={
                        "agentic": True,
                        "iterations": result.iterations,
                        "sub_queries": len(result.sub_queries),
                    },
                    confidence_score=result.confidence,
                    confidence_level="high" if result.confidence > 0.8 else "medium" if result.confidence > 0.5 else "low",
                )

            except ImportError:
                logger.warning("Agentic RAG not available, falling back to standard RAG")

        # Standard RAG with optional graph enhancement
        response = await self.query(
            question=question,
            session_id=session_id,
            collection_filter=collection_filter,
            access_tier=access_tier,
            user_id=user_id,
        )

        # Optionally enhance with graph context
        if use_graph and db_session:
            try:
                from services.knowledge_graph import get_knowledge_graph_service
                graph_service = await get_knowledge_graph_service(db_session)

                # Get graph context
                graph_context = await graph_service.graph_search(question, max_hops=2, top_k=5)

                if graph_context.entities:
                    # Add graph summary to metadata
                    response.metadata["graph_entities"] = len(graph_context.entities)
                    response.metadata["graph_relations"] = len(graph_context.relations)

                    logger.debug(
                        "Graph context added",
                        entities=len(graph_context.entities),
                        relations=len(graph_context.relations),
                    )

            except ImportError:
                logger.debug("Knowledge graph service not available")
            except Exception as e:
                logger.warning("Graph enhancement failed", error=str(e))

        return response

    async def create_pgvector_store(
        self,
        connection_string: str,
        collection_name: str = "document_chunks",
    ):
        """
        Create and set PGVector store.

        Args:
            connection_string: PostgreSQL connection string
            collection_name: Name for the vector collection
        """
        from langchain_postgres import PGVector

        self._vector_store = PGVector(
            embeddings=self.embeddings.embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True,
        )

        logger.info(
            "Created PGVector store",
            collection_name=collection_name,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

import threading

_default_rag_service: Optional[RAGService] = None
_rag_service_lock = threading.Lock()


def get_rag_service(
    config: Optional[RAGConfig] = None,
) -> RAGService:
    """Get or create default RAG service instance (thread-safe)."""
    global _default_rag_service

    # Fast path for existing service with no config override
    if _default_rag_service is not None and config is None:
        return _default_rag_service

    with _rag_service_lock:
        # Double-check after acquiring lock
        if _default_rag_service is None or config is not None:
            _default_rag_service = RAGService(config=config)

        return _default_rag_service


def get_rag_service_dependency() -> RAGService:
    """FastAPI dependency wrapper for RAG service (no parameters for DI compatibility)."""
    return get_rag_service()


async def simple_query(
    question: str,
    top_k: int = 5,
) -> str:
    """
    Simple RAG query without session management.

    Args:
        question: Question to answer
        top_k: Number of sources to retrieve

    Returns:
        Answer string
    """
    config = RAGConfig(top_k=top_k, include_sources=False)
    service = RAGService(config=config)
    response = await service.query(question)
    return response.content
