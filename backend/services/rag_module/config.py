"""
AIDocumentIndexer - RAG Configuration
======================================

Configuration for RAG service with comprehensive settings for:
- Retrieval (top_k, similarity threshold, hybrid search)
- Response generation (tokens, temperature, sources)
- Advanced features (query expansion, verification, HyDE, CRAG)
- Two-stage and hierarchical retrieval
- Knowledge graph integration
- Self-RAG verification
"""

import os
from typing import Optional


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

        # Two-stage retrieval - fast ANN + ColBERT reranking for better precision at scale
        enable_two_stage_retrieval: bool = None,  # Read from settings if not set
        two_stage_candidates: int = 150,  # Stage 1 candidates (ANN retrieval)
        use_colbert_reranker: bool = True,  # Use ColBERT (else cross-encoder) in stage 2

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

        # Smart pre-filtering for large collections (10k+ docs)
        enable_smart_filter: bool = None,  # Read from settings if not set
        smart_filter_min_docs: int = 500,  # Minimum docs to trigger pre-filtering
        smart_filter_max_candidates: int = 1000,  # Max docs after pre-filtering

        # Self-RAG (response verification and hallucination detection)
        enable_self_rag: bool = None,  # Read from settings if not set
        self_rag_min_supported_ratio: float = 0.7,  # Min ratio of supported claims
        self_rag_enable_regeneration: bool = True,  # Auto-regenerate on issues
    ):
        from backend.services.settings import get_settings_service

        settings = get_settings_service()

        # Read embedding provider from settings first, then environment, then default
        # IMPORTANT: Use EMBEDDING_PROVIDER env var, NOT DEFAULT_LLM_PROVIDER
        # Embeddings must be independent of chat LLM to allow switching LLMs without re-indexing
        if embedding_provider is None:
            embedding_provider = settings.get_default_value("embedding.provider")
            if embedding_provider is None:
                embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")

        # Read embedding model from settings first, then environment
        if embedding_model is None:
            embedding_model = settings.get_default_value("embedding.model")
            if embedding_model is None:
                embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL" if embedding_provider == "ollama" else "DEFAULT_EMBEDDING_MODEL")

        # Default provider for chat (separate from embedding)
        default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

        # Read retrieval settings from settings service defaults
        # These are synchronous reads of default values; async DB values applied at runtime
        if top_k is None:
            top_k = settings.get_default_value("rag.top_k") or 10
        if similarity_threshold is None:
            similarity_threshold = settings.get_default_value("rag.similarity_threshold") or 0.55
        if rerank_results is None:
            rerank_results = settings.get_default_value("rag.rerank_results")
            if rerank_results is None:
                rerank_results = True

        # Read RAG settings - try settings service first, then environment variables
        if enable_query_expansion is None:
            enable_query_expansion = settings.get_default_value("rag.query_expansion_enabled")
            if enable_query_expansion is None:
                enable_query_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
        if query_expansion_count is None:
            query_expansion_count = settings.get_default_value("rag.query_expansion_count")
            if query_expansion_count is None:
                query_expansion_count = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
        if enable_verification is None:
            enable_verification = settings.get_default_value("rag.verification_enabled")
            if enable_verification is None:
                enable_verification = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"
        if verification_level is None:
            verification_level = settings.get_default_value("rag.verification_level")
            if verification_level is None:
                verification_level = os.getenv("VERIFICATION_LEVEL", "quick")
        if enable_dynamic_weighting is None:
            enable_dynamic_weighting = settings.get_default_value("rag.dynamic_weighting_enabled")
            if enable_dynamic_weighting is None:
                enable_dynamic_weighting = os.getenv("ENABLE_DYNAMIC_WEIGHTING", "true").lower() == "true"
        if enable_hyde is None:
            enable_hyde = settings.get_default_value("rag.hyde_enabled")
            if enable_hyde is None:
                enable_hyde = os.getenv("ENABLE_HYDE", "true").lower() == "true"
        if enable_crag is None:
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

        # Read two-stage retrieval settings
        if enable_two_stage_retrieval is None:
            enable_two_stage_retrieval = settings.get_default_value("rag.two_stage_retrieval_enabled")
            if enable_two_stage_retrieval is None:
                enable_two_stage_retrieval = os.getenv("TWO_STAGE_RETRIEVAL_ENABLED", "false").lower() == "true"
        if two_stage_candidates == 150:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.stage1_candidates")
            if stored_value is not None:
                two_stage_candidates = stored_value
        if use_colbert_reranker:  # Default is True, check settings for override
            stored_value = settings.get_default_value("rag.use_colbert_reranker")
            if stored_value is not None:
                use_colbert_reranker = stored_value

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

        # Read smart filter settings
        if enable_smart_filter is None:
            enable_smart_filter = settings.get_default_value("rag.smart_filter_enabled")
            if enable_smart_filter is None:
                enable_smart_filter = os.getenv("ENABLE_SMART_FILTER", "true").lower() == "true"
        if smart_filter_min_docs == 500:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.smart_filter_min_docs")
            if stored_value is not None:
                smart_filter_min_docs = stored_value
        if smart_filter_max_candidates == 1000:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.smart_filter_max_candidates")
            if stored_value is not None:
                smart_filter_max_candidates = stored_value

        # Read Self-RAG settings - use core config as fallback
        if enable_self_rag is None:
            enable_self_rag = settings.get_default_value("rag.self_rag_enabled")
            if enable_self_rag is None:
                # Fall back to core settings (which reads env var with proper default)
                from backend.core.config import settings as core_settings
                enable_self_rag = getattr(core_settings, 'ENABLE_SELF_RAG', True)
        if self_rag_min_supported_ratio == 0.7:  # Default value, might be overridden
            stored_value = settings.get_default_value("rag.self_rag_min_supported_ratio")
            if stored_value is not None:
                self_rag_min_supported_ratio = stored_value
            else:
                from backend.core.config import settings as core_settings
                self_rag_min_supported_ratio = getattr(core_settings, 'SELF_RAG_MIN_SUPPORTED_RATIO', 0.7)

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
        # Two-stage retrieval
        self.enable_two_stage_retrieval = enable_two_stage_retrieval
        self.two_stage_candidates = two_stage_candidates
        self.use_colbert_reranker = use_colbert_reranker
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
        # Smart pre-filtering
        self.enable_smart_filter = enable_smart_filter
        self.smart_filter_min_docs = smart_filter_min_docs
        self.smart_filter_max_candidates = smart_filter_max_candidates
        # Self-RAG
        self.enable_self_rag = enable_self_rag
        self.self_rag_min_supported_ratio = self_rag_min_supported_ratio
        self.self_rag_enable_regeneration = self_rag_enable_regeneration
