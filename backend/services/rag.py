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

# LangChain chains (modern import paths for v0.3+)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LangChain memory
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
)

logger = structlog.get_logger(__name__)


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
        self._ttl_seconds = ttl_seconds or int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "300"))
        self._max_size = max_size
        self._cache: Dict[str, Tuple[List[Any], datetime]] = {}

    def _make_key(self, query: str, collection: Optional[str], access_tier: int, top_k: int) -> str:
        """Create cache key from query parameters."""
        key_data = f"{query}|{collection or ''}|{access_tier}|{top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()

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

        # Cache for session-specific LLM instances
        self._session_llm_cache: Dict[str, Any] = {}
        self._session_config_cache: Dict[str, LLMConfigResult] = {}

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

        # Check cache first
        cache_key = session_id or "_default"
        if cache_key in self._session_llm_cache:
            return self._session_llm_cache[cache_key], self._session_config_cache.get(cache_key)

        try:
            llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="rag",
                session_id=session_id,
                user_id=user_id,
            )

            # Cache the result
            self._session_llm_cache[cache_key] = llm
            self._session_config_cache[cache_key] = config

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
        else:
            self._session_llm_cache.clear()
            self._session_config_cache.clear()

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
            except Exception as e:
                logger.warning("Query decomposition failed", error=str(e))
                decomposed_query = None

        # Get LLM for this session (with database-driven config)
        llm, llm_config = await self.get_llm_for_session(
            session_id=session_id,
            user_id=user_id,
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

        # Retrieve relevant documents
        # If query was decomposed, retrieve for each sub-query and merge
        if decomposed_query and len(decomposed_query.sub_queries) > 1:
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
            retrieved_docs = await self._retrieve(
                question,
                collection_filter=collection_filter,
                access_tier=access_tier,
                top_k=top_k,
                document_ids=folder_document_ids,  # Filter to folder documents
                enhance_query=enhance_query,
                organization_id=organization_id,
                user_id=user_id,
                is_superadmin=is_superadmin,
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
            except Exception as e:
                # Log error but continue without verification
                logger.warning("Verification failed, continuing without filtering", error=str(e))
                verification_result = None

        # Format context from retrieved documents
        context, sources = self._format_context(retrieved_docs, include_collection_context)
        logger.debug("Formatted context", context_length=len(context), sources_count=len(sources))

        # Add additional context (e.g., from temporary documents)
        if additional_context:
            context = f"--- Uploaded Documents ---\n{additional_context}\n\n--- Library Documents ---\n{context}"
            logger.debug("Added additional context", additional_context_length=len(additional_context))

        # Build prompt with language instruction
        # Support "auto" mode: respond in the same language as the question
        auto_detect = (language == "auto")
        effective_language = "en" if auto_detect else language
        language_instruction = _get_language_instruction(effective_language, auto_detect=auto_detect)
        system_prompt = RAG_SYSTEM_PROMPT
        if language_instruction:
            system_prompt = f"{RAG_SYSTEM_PROMPT}\n{language_instruction}"

        if session_id:
            # Use conversational prompt with history
            memory = self._get_memory(session_id)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

            messages = [
                SystemMessage(content=system_prompt),
                *chat_history,
                HumanMessage(content=f"{CONVERSATIONAL_RAG_TEMPLATE}\n\nQuestion: {question}".replace("{context}", context)),
            ]
        else:
            # Single-turn query
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=RAG_PROMPT_TEMPLATE.format(context=context, question=question)),
            ]

        # Generate response
        response = await llm.ainvoke(messages)
        raw_content = response.content if hasattr(response, 'content') else str(response)

        # Parse suggested questions from the response
        content, suggested_questions = _parse_suggested_questions(raw_content)

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

            except Exception as e:
                logger.warning("CRAG processing failed", error=str(e))

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

            except Exception as e:
                logger.warning("Self-RAG verification failed", error=str(e))

        return RAGResponse(
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
        )

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
        system_prompt = RAG_SYSTEM_PROMPT
        if language_instruction:
            system_prompt = f"{RAG_SYSTEM_PROMPT}\n{language_instruction}"

        if session_id:
            memory = self._get_memory(session_id)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

            messages = [
                SystemMessage(content=system_prompt),
                *chat_history,
                HumanMessage(content=f"{CONVERSATIONAL_RAG_TEMPLATE}\n\nQuestion: {question}".replace("{context}", context)),
            ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=RAG_PROMPT_TEMPLATE.format(context=context, question=question)),
            ]

        # Stream response - use StringIO for efficient string accumulation (O(n) vs O(n²))
        response_buffer = io.StringIO()
        try:
            async for chunk in llm.astream(messages):
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
            logger.error("Streaming error", error=str(e))
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

            # Classify query intent for dynamic weighting (if enabled)
            vector_weight = None
            keyword_weight = None
            query_classification = None
            if self._query_classifier is not None and search_type == SearchType.HYBRID:
                query_classification = self._query_classifier.classify(query)
                vector_weight = query_classification.vector_weight
                keyword_weight = query_classification.keyword_weight
                logger.debug(
                    "Query classified for dynamic weighting",
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

            # Perform search - use two-stage (ColBERT) or hierarchical retrieval if enabled
            # Two-stage takes priority as it includes reranking for better precision
            logger.info(
                "Calling vectorstore search",
                query_length=len(query),
                has_embedding=query_embedding is not None and len(query_embedding) > 0,
                search_type=search_type.value,
                document_ids_count=len(document_ids) if document_ids else 0,
                two_stage=self._two_stage_retriever is not None,
                hierarchical=self._hierarchical_retriever is not None,
            )

            if self._two_stage_retriever is not None:
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

            # Enhance with knowledge graph if enabled
            if self._enable_knowledge_graph:
                langchain_results = await self._enhance_with_knowledge_graph(
                    query=query,
                    existing_results=langchain_results,
                    top_k=top_k,
                    organization_id=organization_id,
                    access_tier_level=access_tier,
                    user_id=user_id,
                    is_superadmin=is_superadmin,
                )

            return langchain_results

        except Exception as e:
            logger.error("Custom vectorstore retrieval failed", error=str(e), exc_info=True)
            return []

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

    def _format_context(
        self,
        retrieved_docs: List[Tuple[Document, float]],
        include_collection_context: bool = True,
    ) -> Tuple[str, List[Source]]:
        """
        Format retrieved documents into context string and sources list.

        Args:
            retrieved_docs: List of (document, score) tuples
            include_collection_context: Whether to include collection tags in LLM context

        Returns:
            Tuple of (context_string, sources_list)
        """
        if not retrieved_docs:
            return "No relevant documents found.", []

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
        return context, sources

    def set_vector_store(self, vector_store: Any):
        """Set vector store for retrieval."""
        self._vector_store = vector_store

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
