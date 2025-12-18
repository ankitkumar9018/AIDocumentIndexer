"""
AIDocumentIndexer - RAG Service
================================

Retrieval-Augmented Generation service using LangChain.
Provides hybrid search (vector + keyword) and conversational RAG chains.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import io
import json
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
from backend.services.query_expander import (
    QueryExpander,
    QueryExpansionConfig,
    get_query_expander,
)

logger = structlog.get_logger(__name__)


@dataclass
class Source:
    """Source citation for RAG response."""
    document_id: str
    document_name: str
    chunk_id: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    relevance_score: float = 0.0
    snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class StreamChunk:
    """A chunk in streaming response."""
    type: str  # "content", "source", "metadata", "done", "error"
    data: Any


class RAGConfig:
    """Configuration for RAG service."""

    def __init__(
        self,
        # Retrieval settings
        top_k: int = 5,
        similarity_threshold: float = 0.4,  # Lower threshold for OCR'd documents
        use_hybrid_search: bool = True,
        rerank_results: bool = False,

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
        enable_query_expansion: bool = False,  # OFF by default
        query_expansion_count: int = 2,  # Number of query variations
        query_expansion_model: str = "gpt-4o-mini",  # Cost-effective model
    ):
        import os
        # Default provider from environment variable
        default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

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

Provide a helpful, accurate answer based on the context. Cite specific documents when referencing information."""

CONVERSATIONAL_RAG_TEMPLATE = """You are having a conversation with a user about their document archive.
Use the retrieved context and conversation history to provide helpful answers.

Retrieved Context:
{context}

Based on this context and our conversation, please answer the user's latest question.
If the context doesn't contain relevant information, acknowledge that and provide what help you can."""


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

        # Initialize query expander if enabled
        self._query_expander: Optional[QueryExpander] = None
        if self.config.enable_query_expansion:
            expansion_config = QueryExpansionConfig(
                enabled=True,
                expansion_count=self.config.query_expansion_count,
                model=self.config.query_expansion_model,
            )
            self._query_expander = QueryExpander(expansion_config)

        logger.info(
            "Initialized RAG service",
            chat_provider=self.config.chat_provider,
            embedding_provider=self.config.embedding_provider,
            top_k=self.config.top_k,
            use_custom_vectorstore=use_custom_vectorstore,
            use_db_config=use_db_config,
            track_usage=track_usage,
            query_expansion=self.config.enable_query_expansion,
        )

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

    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
    ) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            question: User's question
            session_id: Session ID for conversation memory
            collection_filter: Filter by document collection
            access_tier: User's access tier for RLS
            user_id: User ID for usage tracking

        Returns:
            RAGResponse with answer and sources
        """
        import time
        start_time = time.time()

        logger.info(
            "Processing RAG query",
            question_length=len(question),
            session_id=session_id,
            collection_filter=collection_filter,
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
        )

        # Format context from retrieved documents
        context, sources = self._format_context(retrieved_docs)

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
        content = response.content if hasattr(response, 'content') else str(response)

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
        )

    async def query_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        access_tier: int = 100,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Query RAG with streaming response.

        Args:
            question: User's question
            session_id: Session ID for memory
            collection_filter: Collection filter
            access_tier: User's access tier
            user_id: User ID for usage tracking

        Yields:
            StreamChunk objects with response parts
        """
        import time
        start_time = time.time()

        logger.info(
            "Processing streaming RAG query",
            question_length=len(question),
            session_id=session_id,
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
        )

        context, sources = self._format_context(retrieved_docs)

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
                        "snippet": s.snippet,
                    }
                    for s in sources
                ])

            # Get the full response from buffer
            full_response = response_buffer.getvalue()

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

        logger.debug(
            "Starting retrieval",
            query_length=len(query),
            top_k=top_k,
            has_custom_vectorstore=self._custom_vectorstore is not None,
        )

        # Expand query if enabled (generates paraphrased versions for better recall)
        queries_to_search = [query]
        if self._query_expander is not None and self._query_expander.should_expand(query):
            try:
                expansion_result = await self._query_expander.expand_query(query)
                if expansion_result.expanded_queries:
                    queries_to_search = self._query_expander.get_all_queries(expansion_result)
                    logger.debug(
                        "Query expanded",
                        original=query,
                        expanded_count=len(expansion_result.expanded_queries),
                    )
            except Exception as e:
                logger.warning("Query expansion failed, using original query", error=str(e))

        # Collect results from all queries
        all_results: List[Tuple[Document, float]] = []
        seen_chunk_ids: set = set()

        for search_query in queries_to_search:
            # Use custom vector store if available
            if self._custom_vectorstore is not None:
                results = await self._retrieve_with_custom_store(
                    query=search_query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
                )
            # Use LangChain vector store if configured
            elif self._vector_store is not None:
                results = await self._retrieve_with_langchain_store(
                    query=search_query,
                    collection_filter=collection_filter,
                    access_tier=access_tier,
                    top_k=top_k,
                )
            else:
                # Return mock results for development
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

        if len(queries_to_search) > 1:
            logger.debug(
                "Query expansion retrieval complete",
                queries_searched=len(queries_to_search),
                unique_results=len(all_results),
                returning=min(top_k, len(all_results)),
            )

        return all_results[:top_k]

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
            collection_filter: Filter by collection
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

            # Perform search
            # Note: collection_filter can be passed to search() once the vectorstore
            # supports filtering by collection/tags in the metadata
            results = await self._custom_vectorstore.search(
                query=query,
                query_embedding=query_embedding,
                search_type=search_type,
                top_k=top_k,
                access_tier_level=access_tier,
            )

            # Convert SearchResult to LangChain Document format
            langchain_results = []
            logger.debug(
                "Vector search returned results",
                total_results=len(results),
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
                        "document_name": result.document_filename or result.document_title or "Unknown Document",
                        "chunk_id": result.chunk_id,
                        "page_number": result.page_number,
                        "section_title": result.section_title,
                        **result.metadata,
                    },
                )
                langchain_results.append((doc, result.score))

            logger.info(
                "Retrieved documents with custom store",
                num_results=len(langchain_results),
                search_type=search_type.value,
            )

            return langchain_results

        except Exception as e:
            logger.error("Custom vectorstore retrieval failed", error=str(e))
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

    def _format_context(
        self,
        retrieved_docs: List[Tuple[Document, float]],
    ) -> Tuple[str, List[Source]]:
        """
        Format retrieved documents into context string and sources list.

        Args:
            retrieved_docs: List of (document, score) tuples

        Returns:
            Tuple of (context_string, sources_list)
        """
        if not retrieved_docs:
            return "No relevant documents found.", []

        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(retrieved_docs, 1):
            metadata = doc.metadata or {}

            # Build context entry
            doc_name = metadata.get("document_name", "Unknown Document")
            page_info = ""
            if metadata.get("page_number"):
                page_info = f" (Page {metadata['page_number']})"
            elif metadata.get("slide_number"):
                page_info = f" (Slide {metadata['slide_number']})"

            context_parts.append(
                f"[Source {i}: {doc_name}{page_info}]\n{doc.page_content}\n"
            )

            # Build source citation
            source = Source(
                document_id=metadata.get("document_id", f"doc-{i}"),
                document_name=doc_name,
                chunk_id=metadata.get("chunk_id", f"chunk-{i}"),
                page_number=metadata.get("page_number"),
                slide_number=metadata.get("slide_number"),
                relevance_score=score,
                snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                metadata=metadata,
            )
            sources.append(source)

        context = "\n".join(context_parts)
        return context, sources

    def set_vector_store(self, vector_store: Any):
        """Set vector store for retrieval."""
        self._vector_store = vector_store

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

_default_rag_service: Optional[RAGService] = None


def get_rag_service(
    config: Optional[RAGConfig] = None,
) -> RAGService:
    """Get or create default RAG service instance."""
    global _default_rag_service

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
