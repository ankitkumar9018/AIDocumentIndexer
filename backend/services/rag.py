"""
AIDocumentIndexer - RAG Service
================================

Retrieval-Augmented Generation service using LangChain.
Provides hybrid search (vector + keyword) and conversational RAG chains.
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
)
# Note: KnowledgeGraphService is imported lazily inside _enhance_with_knowledge_graph
# to avoid circular imports (knowledge_graph -> llm -> services.__init__ -> rag)

logger = structlog.get_logger(__name__)


@dataclass
class Source:
    """Source citation for RAG response."""
    document_id: str
    document_name: str
    chunk_id: str
    collection: Optional[str] = None  # Collection/tag for document grouping
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    relevance_score: float = 0.0  # RRF score for ranking (may be tiny ~0.01-0.03)
    similarity_score: float = 0.0  # Original vector cosine similarity (0-1) for display
    snippet: str = ""
    full_content: str = ""  # Full chunk content for source viewer
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Context expansion (surrounding chunks for navigation)
    prev_chunk_snippet: Optional[str] = None
    next_chunk_snippet: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class RAGResponse:
    """Response from RAG query."""
    content: str
    sources: List[Source]
    query: str
    model: str
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Verification/confidence fields
    confidence_score: Optional[float] = None  # 0-1 confidence
    confidence_level: Optional[str] = None  # "high", "medium", "low"
    verification_result: Optional[VerificationResult] = None
    # Suggested follow-up questions
    suggested_questions: List[str] = field(default_factory=list)
    # Confidence warning for UI display (empty string = no warning)
    confidence_warning: str = ""
    # CRAG result if query was refined
    crag_result: Optional[CRAGResult] = None


@dataclass
class StreamChunk:
    """A chunk in streaming response."""
    type: str  # "content", "source", "metadata", "done", "error"
    data: Any


class RAGConfig:
    """Configuration for RAG service."""

    def __init__(
        self,
        # Retrieval settings - None means "read from settings service"
        top_k: int = None,
        similarity_threshold: float = None,
        use_hybrid_search: bool = True,
        rerank_results: bool = None,

        # Response settings
        max_response_tokens: int = 2048,
        temperature: float = 0.7,
        include_sources: bool = True,

        # Memory settings
        memory_window: int = 10,  # Number of conversation turns to remember

        # Model settings
        chat_provider: str = None,  # Will use DEFAULT_LLM_PROVIDER env var
        chat_model: Optional[str] = None,
        embedding_provider: str = None,  # Will use DEFAULT_LLM_PROVIDER env var
        embedding_model: Optional[str] = None,

        # Query expansion settings (optional, improves recall by 8-12%)
        enable_query_expansion: bool = None,  # Read from env if not set
        query_expansion_count: int = None,  # Number of query variations
        query_expansion_model: str = "gpt-4o-mini",  # Cost-effective model

        # Verification settings (Self-RAG)
        enable_verification: bool = None,  # Read from env if not set
        verification_level: str = None,  # "none", "quick", "standard", "thorough"

        # Dynamic search weighting (query intent classification)
        enable_dynamic_weighting: bool = None,  # Read from env if not set

        # HyDE (Hypothetical Document Embeddings) - improves recall for short/abstract queries
        enable_hyde: bool = None,  # Read from env if not set
        hyde_min_query_words: int = 5,  # Only use HyDE for queries shorter than this

        # CRAG (Corrective RAG) - auto-refines queries on low confidence
        enable_crag: bool = None,  # Read from env if not set
        crag_confidence_threshold: float = 0.5,  # Trigger CRAG below this confidence

        # Hierarchical retrieval - document-first strategy for better diversity
        enable_hierarchical_retrieval: bool = None,  # Read from settings if not set
        hierarchical_doc_limit: int = 10,  # Max documents in stage 1
        hierarchical_chunks_per_doc: int = 3,  # Chunks per document in stage 2

        # Semantic deduplication - remove near-duplicate chunks from expanded queries
        enable_semantic_dedup: bool = None,  # Read from settings if not set
        semantic_dedup_threshold: float = 0.95,  # Similarity threshold for dedup

        # Knowledge graph integration - entity-aware retrieval
        enable_knowledge_graph: bool = None,  # Read from settings if not set
        knowledge_graph_max_hops: int = 2,  # Graph traversal depth
    ):
        import os
        from backend.services.settings import get_settings_service

        # Default provider from environment variable
        default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        settings = get_settings_service()

        # Read retrieval settings from settings service defaults
        # These are synchronous reads of default values; async DB values applied at runtime
        if top_k is None:
            top_k = settings.get_default_value("rag.top_k") or 10
        if similarity_threshold is None:
            similarity_threshold = settings.get_default_value("rag.similarity_threshold") or 0.4
        if rerank_results is None:
            rerank_results = settings.get_default_value("rag.rerank_results")
            if rerank_results is None:
                rerank_results = True

        # Read RAG settings from environment with sensible defaults
        if enable_query_expansion is None:
            enable_query_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
        if query_expansion_count is None:
            # Try settings service first, fall back to env
            query_expansion_count = settings.get_default_value("rag.query_expansion_count")
            if query_expansion_count is None:
                query_expansion_count = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
        if enable_verification is None:
            enable_verification = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"
        if verification_level is None:
            verification_level = os.getenv("VERIFICATION_LEVEL", "quick")
        if enable_dynamic_weighting is None:
            enable_dynamic_weighting = os.getenv("ENABLE_DYNAMIC_WEIGHTING", "true").lower() == "true"
        if enable_hyde is None:
            # Try settings service first, fall back to env
            enable_hyde = settings.get_default_value("rag.hyde_enabled")
            if enable_hyde is None:
                enable_hyde = os.getenv("ENABLE_HYDE", "true").lower() == "true"
        if enable_crag is None:
            # Try settings service first, fall back to env
            enable_crag = settings.get_default_value("rag.crag_enabled")
            if enable_crag is None:
                enable_crag = os.getenv("ENABLE_CRAG", "true").lower() == "true"
        # Read other HyDE/CRAG settings from settings service
        if hyde_min_query_words == 5:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.hyde_min_query_words")
            if stored_value is not None:
                hyde_min_query_words = stored_value
        if crag_confidence_threshold == 0.5:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.crag_confidence_threshold")
            if stored_value is not None:
                crag_confidence_threshold = stored_value

        # Read hierarchical retrieval settings
        if enable_hierarchical_retrieval is None:
            enable_hierarchical_retrieval = settings.get_default_value("rag.hierarchical_enabled")
            if enable_hierarchical_retrieval is None:
                enable_hierarchical_retrieval = os.getenv("ENABLE_HIERARCHICAL_RETRIEVAL", "false").lower() == "true"
        if hierarchical_doc_limit == 10:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.hierarchical_doc_limit")
            if stored_value is not None:
                hierarchical_doc_limit = stored_value
        if hierarchical_chunks_per_doc == 3:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.hierarchical_chunks_per_doc")
            if stored_value is not None:
                hierarchical_chunks_per_doc = stored_value

        # Read semantic deduplication settings
        if enable_semantic_dedup is None:
            enable_semantic_dedup = settings.get_default_value("rag.semantic_dedup_enabled")
            if enable_semantic_dedup is None:
                enable_semantic_dedup = os.getenv("ENABLE_SEMANTIC_DEDUP", "true").lower() == "true"
        if semantic_dedup_threshold == 0.95:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.semantic_dedup_threshold")
            if stored_value is not None:
                semantic_dedup_threshold = stored_value

        # Read knowledge graph settings
        if enable_knowledge_graph is None:
            enable_knowledge_graph = settings.get_default_value("rag.knowledge_graph_enabled")
            if enable_knowledge_graph is None:
                enable_knowledge_graph = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "false").lower() == "true"
        if knowledge_graph_max_hops == 2:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.knowledge_graph_max_hops")
            if stored_value is not None:
                knowledge_graph_max_hops = stored_value

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_search = use_hybrid_search
        self.rerank_results = rerank_results
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.include_sources = include_sources
        self.memory_window = memory_window
        self.chat_provider = chat_provider or default_provider
        self.chat_model = chat_model
        self.embedding_provider = embedding_provider or default_provider
        self.embedding_model = embedding_model
        # Query expansion
        self.enable_query_expansion = enable_query_expansion
        self.query_expansion_count = query_expansion_count
        self.query_expansion_model = query_expansion_model
        # Verification
        self.enable_verification = enable_verification
        self.verification_level = verification_level
        # Dynamic weighting
        self.enable_dynamic_weighting = enable_dynamic_weighting
        # HyDE
        self.enable_hyde = enable_hyde
        self.hyde_min_query_words = hyde_min_query_words
        # CRAG
        self.enable_crag = enable_crag
        self.crag_confidence_threshold = crag_confidence_threshold
        # Hierarchical retrieval
        self.enable_hierarchical_retrieval = enable_hierarchical_retrieval
        self.hierarchical_doc_limit = hierarchical_doc_limit
        self.hierarchical_chunks_per_doc = hierarchical_chunks_per_doc
        # Semantic deduplication
        self.enable_semantic_dedup = enable_semantic_dedup
        self.semantic_dedup_threshold = semantic_dedup_threshold
        # Knowledge graph
        self.enable_knowledge_graph = enable_knowledge_graph
        self.knowledge_graph_max_hops = knowledge_graph_max_hops


# =============================================================================
# Prompt Templates
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an intelligent assistant for the AI Document Indexer system.
Your role is to help users find information from their document archive and answer questions based on the retrieved content.

Guidelines:
1. Answer questions based primarily on the provided context from the documents
2. If the context doesn't contain relevant information, say so clearly
3. Always cite your sources by mentioning the document names
4. Be concise but thorough in your responses
5. If asked about something outside the document context, clarify that your knowledge comes from the indexed documents

Remember: You are helping users explore their historical document archive spanning many years of work."""

RAG_PROMPT_TEMPLATE = """Use the following context from the document archive to answer the user's question.
If the context doesn't contain relevant information to answer the question, say so.

Context:
{context}

Question: {question}

Provide a helpful, accurate answer based on the context. Cite specific documents when referencing information.

At the end of your response, on a new line, suggest 2-3 related follow-up questions the user might want to ask, prefixed with "SUGGESTED_QUESTIONS:" and separated by "|". Example:
SUGGESTED_QUESTIONS: What are the key benefits?|How does this compare to alternatives?|When was this implemented?"""

CONVERSATIONAL_RAG_TEMPLATE = """You are having a conversation with a user about their document archive.
Use the retrieved context and conversation history to provide helpful answers.

Retrieved Context:
{context}

Based on this context and our conversation, please answer the user's latest question.
If the context doesn't contain relevant information, acknowledge that and provide what help you can.

At the end of your response, on a new line, suggest 2-3 related follow-up questions the user might want to ask, prefixed with "SUGGESTED_QUESTIONS:" and separated by "|". Example:
SUGGESTED_QUESTIONS: What are the key benefits?|How does this compare to alternatives?|When was this implemented?"""


def _parse_suggested_questions(content: str) -> Tuple[str, List[str]]:
    """
    Parse suggested questions from response content.

    Args:
        content: Full response content from LLM

    Returns:
        Tuple of (cleaned_content, list_of_suggested_questions)
    """
    suggested_questions = []
    cleaned_content = content

    # Look for SUGGESTED_QUESTIONS: line
    if "SUGGESTED_QUESTIONS:" in content:
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            if line.strip().startswith("SUGGESTED_QUESTIONS:"):
                # Extract questions from this line
                questions_part = line.split("SUGGESTED_QUESTIONS:", 1)[1].strip()
                suggested_questions = [q.strip() for q in questions_part.split("|") if q.strip()]
            else:
                new_lines.append(line)
        cleaned_content = "\n".join(new_lines).rstrip()

    return cleaned_content, suggested_questions


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

        # Knowledge graph is initialized lazily per-request (requires db session)
        self._enable_knowledge_graph = self.config.enable_knowledge_graph
        self._knowledge_graph_max_hops = self.config.knowledge_graph_max_hops

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
            hierarchical_retrieval=self.config.enable_hierarchical_retrieval,
            semantic_dedup=self.config.enable_semantic_dedup,
            knowledge_graph=self.config.enable_knowledge_graph,
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

        Returns:
            List of document dicts with content, source, and score
        """
        logger.info(
            "Searching documents",
            query_length=len(query),
            limit=limit,
            collection_filter=collection_filter,
        )

        # Use _retrieve to get document chunks
        retrieved_docs = await self._retrieve(
            query=query,
            collection_filter=collection_filter,
            access_tier=access_tier,
            top_k=limit,
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
    ) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            question: User's question
            session_id: Session ID for conversation memory
            collection_filter: Filter by document collection
            access_tier: User's access tier for RLS
            user_id: User ID for usage tracking
            include_collection_context: Whether to include collection tags in LLM context
            additional_context: Additional context to include (e.g., from temp documents)
            top_k: Optional override for number of documents to retrieve

        Returns:
            RAGResponse with answer and sources
        """
        import time
        start_time = time.time()

        # Load runtime settings from database if top_k not specified
        if top_k is None:
            runtime_settings = await self.get_runtime_settings()
            top_k = runtime_settings.get("top_k", self.config.top_k)

        logger.info(
            "Processing RAG query",
            question_length=len(question),
            session_id=session_id,
            collection_filter=collection_filter,
            top_k=top_k,
        )

        # Get LLM for this session (with database-driven config)
        llm, llm_config = await self.get_llm_for_session(
            session_id=session_id,
            user_id=user_id,
        )

        # Retrieve relevant documents
        retrieved_docs = await self._retrieve(
            question,
            collection_filter=collection_filter,
            access_tier=access_tier,
            top_k=top_k,
        )
        logger.debug("Retrieved documents from vectorstore", count=len(retrieved_docs))

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

        # Build prompt
        if session_id:
            # Use conversational prompt with history
            memory = self._get_memory(session_id)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                *chat_history,
                HumanMessage(content=f"{CONVERSATIONAL_RAG_TEMPLATE}\n\nQuestion: {question}".replace("{context}", context)),
            ]
        else:
            # Single-turn query
            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
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

                if crag_result.action_taken in ["filtered", "refined_query"]:
                    logger.info(
                        "CRAG applied to low-confidence results",
                        action=crag_result.action_taken,
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
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Query RAG with streaming response.

        Args:
            question: User's question
            session_id: Session ID for memory
            collection_filter: Collection filter
            access_tier: User's access tier
            user_id: User ID for usage tracking
            include_collection_context: Whether to include collection tags in LLM context
            top_k: Optional override for number of documents to retrieve

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

        # Retrieve documents
        retrieved_docs = await self._retrieve(
            question,
            collection_filter=collection_filter,
            access_tier=access_tier,
            top_k=top_k,
        )

        context, sources = self._format_context(retrieved_docs, include_collection_context)

        # Build prompt
        if session_id:
            memory = self._get_memory(session_id)
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                *chat_history,
                HumanMessage(content=f"{CONVERSATIONAL_RAG_TEMPLATE}\n\nQuestion: {question}".replace("{context}", context)),
            ]
        else:
            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
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

    async def _retrieve(
        self,
        query: str,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for query.

        Args:
            query: Search query
            collection_filter: Filter by collection
            access_tier: User's access tier
            top_k: Number of results to return

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
        )

        # Build list of queries to search
        queries_to_search = [query]
        query_word_count = len(query.split())

        # Use HyDE for short/abstract queries (< 5 words by default)
        # HyDE generates a hypothetical document that might contain the answer
        hyde_query = None
        if (
            self._hyde_expander is not None
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
        if self._query_expander is not None and self._query_expander.should_expand(query):
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
                )
            elif self._vector_store is not None:
                return await self._retrieve_with_langchain_store(
                    query=search_query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
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
            )
            if self._custom_vectorstore is not None:
                results = await self._retrieve_with_custom_store(
                    query=query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
                )
            elif self._vector_store is not None:
                results = await self._retrieve_with_langchain_store(
                    query=query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
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
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using our custom VectorStore service.

        Args:
            query: Search query
            collection_filter: Filter by collection (or "(Untagged)" for untagged docs)
            access_tier: User's access tier
            top_k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)

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
            document_ids = None
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
                        # SQLite stores JSON arrays as text, so we use raw SQL LIKE
                        # to avoid the custom StringArrayType's JSON encoding of the pattern.
                        # e.g., tags = '["German", "Marketing"]' LIKE '%"German"%'
                        from sqlalchemy import text, cast, String
                        pattern = "%" + '"' + collection_filter + '"' + "%"
                        # Cast tags column to String to bypass StringArrayType processing
                        # and use a literal pattern to avoid parameter binding issues
                        query_stmt = select(DBDocument.id).where(
                            cast(DBDocument.tags, String).like(pattern)
                        )
                    result = await db.execute(query_stmt)
                    document_ids = [str(row[0]) for row in result.fetchall()]
                    logger.info(
                        "Collection filter results",
                        document_ids=document_ids[:5] if document_ids else [],
                        count=len(document_ids),
                    )

                    if not document_ids:
                        logger.debug(
                            "No documents match collection filter",
                            collection_filter=collection_filter,
                        )
                        return []

            # Perform search - use hierarchical retrieval if enabled
            logger.info(
                "Calling vectorstore search",
                query_length=len(query),
                has_embedding=query_embedding is not None and len(query_embedding) > 0,
                search_type=search_type.value,
                document_ids_count=len(document_ids) if document_ids else 0,
                hierarchical=self._hierarchical_retriever is not None,
            )

            if self._hierarchical_retriever is not None:
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
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using LangChain vector store.

        Args:
            query: Search query
            collection_filter: Filter by collection
            access_tier: User's access tier
            top_k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        # Build filter for RLS
        filter_dict = {"access_tier": {"$lte": access_tier}}
        if collection_filter:
            filter_dict["collection"] = collection_filter

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

                # Get graph-enhanced context
                graph_context: GraphRAGContext = await kg_service.graph_search(
                    query=query,
                    max_hops=self._knowledge_graph_max_hops,
                    top_k=top_k,
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

        Uses a simple character-level n-gram similarity (Jaccard) for efficiency.
        Results that are too similar to a higher-scored result are removed.

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
                "Semantic deduplication removed duplicates",
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
