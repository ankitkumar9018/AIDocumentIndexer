"""
AIDocumentIndexer - System Settings Service
============================================

Manages system-wide configuration settings stored in the database.
Provides default values and type conversion for settings.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context
from backend.db.models import SystemSettings

logger = structlog.get_logger(__name__)


# =============================================================================
# Types
# =============================================================================

class SettingCategory(str, Enum):
    """Setting categories for organization."""
    LLM = "llm"
    DATABASE = "database"
    SECURITY = "security"
    NOTIFICATIONS = "notifications"
    GENERAL = "general"
    RAG = "rag"  # Advanced RAG features
    OCR = "ocr"  # OCR configuration
    GENERATION = "generation"  # Document generation settings
    SCRAPING = "scraping"  # Web scraping configuration
    PROCESSING = "processing"  # Parallel processing configuration
    AUDIO = "audio"  # TTS and audio configuration
    INGESTION = "ingestion"  # Knowledge graph and entity extraction settings
    SYSTEM = "system"  # System-wide configuration (custom instructions, language, etc.)
    INFRASTRUCTURE = "infrastructure"  # Scaling & infrastructure configuration


class ValueType(str, Enum):
    """Setting value types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"


@dataclass
class SettingDefinition:
    """Definition for a setting including default value."""
    key: str
    category: SettingCategory
    default_value: Any
    value_type: ValueType
    description: str


# =============================================================================
# Default Settings Definitions
# =============================================================================

DEFAULT_SETTINGS: List[SettingDefinition] = [
    # LLM Configuration
    SettingDefinition(
        key="llm.default_model",
        category=SettingCategory.LLM,
        default_value="gpt-4",
        value_type=ValueType.STRING,
        description="Default LLM model for chat and generation"
    ),
    SettingDefinition(
        key="llm.embedding_model",
        category=SettingCategory.LLM,
        default_value="text-embedding-3-small",
        value_type=ValueType.STRING,
        description="Model used for generating embeddings"
    ),
    SettingDefinition(
        key="llm.temperature",
        category=SettingCategory.LLM,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Default temperature for LLM responses"
    ),
    SettingDefinition(
        key="llm.max_tokens",
        category=SettingCategory.LLM,
        default_value=4096,
        value_type=ValueType.NUMBER,
        description="Maximum tokens for LLM responses"
    ),
    SettingDefinition(
        key="llm.context_window",
        category=SettingCategory.LLM,
        default_value=4096,
        value_type=ValueType.NUMBER,
        description="Context window size for Ollama models (num_ctx). Controls how much text the model can process. 4096-8192 recommended for RAG. Used as fallback for models without a specific recommendation or override."
    ),
    SettingDefinition(
        key="llm.model_context_overrides",
        category=SettingCategory.LLM,
        default_value={},
        value_type=ValueType.JSON,
        description="Per-model context window overrides. JSON mapping model name to num_ctx value (e.g. {\"deepseek-r1:8b\": 6144}). Overrides both global setting and built-in recommendations."
    ),

    # ==========================================================================
    # Phase 70: LLM Resilience Settings (Circuit Breaker + Retry)
    # ==========================================================================
    SettingDefinition(
        key="llm.circuit_breaker_threshold",
        category=SettingCategory.LLM,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Number of consecutive LLM failures before circuit opens (3-10)"
    ),
    SettingDefinition(
        key="llm.circuit_breaker_recovery",
        category=SettingCategory.LLM,
        default_value=60,
        value_type=ValueType.NUMBER,
        description="Seconds before circuit breaker attempts recovery (30-300)"
    ),
    SettingDefinition(
        key="llm.max_retries",
        category=SettingCategory.LLM,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Maximum retry attempts for transient LLM failures (1-5)"
    ),
    SettingDefinition(
        key="llm.call_timeout",
        category=SettingCategory.LLM,
        default_value=120,
        value_type=ValueType.NUMBER,
        description="Timeout for LLM calls in seconds (30-300)"
    ),

    # ==========================================================================
    # Phase 73: Advanced Embedding Model Settings
    # ==========================================================================
    SettingDefinition(
        key="embedding.provider",
        category=SettingCategory.LLM,
        default_value="ollama",  # Default to Ollama for local embedding (independent of chat LLM)
        value_type=ValueType.STRING,
        description="Embedding provider (openai, voyage, jina, cohere, gte, qwen3, gemini, bge-m3, ollama, auto)"
    ),
    SettingDefinition(
        key="embedding.model",
        category=SettingCategory.LLM,
        default_value="",
        value_type=ValueType.STRING,
        description="Embedding model name (leave empty for provider default: nomic-embed-text for ollama, text-embedding-3-small for openai)"
    ),
    SettingDefinition(
        key="embedding.auto_select_enabled",
        category=SettingCategory.LLM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Phase 76: Auto-select embedding model based on content (code→voyage-code, multilingual→gemini, etc.)"
    ),
    SettingDefinition(
        key="embedding.jina_enabled",
        category=SettingCategory.LLM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Jina Embeddings v3 (89 languages, flexible dimensions)"
    ),
    SettingDefinition(
        key="embedding.jina_dimensions",
        category=SettingCategory.LLM,
        default_value=1024,
        value_type=ValueType.NUMBER,
        description="Jina output dimensions (64-1024, lower = faster but less accurate)"
    ),
    SettingDefinition(
        key="embedding.cohere_enabled",
        category=SettingCategory.LLM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Cohere Embed v3.5 (self-improving, compression support)"
    ),
    SettingDefinition(
        key="embedding.cohere_model",
        category=SettingCategory.LLM,
        default_value="embed-english-v3.0",
        value_type=ValueType.STRING,
        description="Cohere model (embed-english-v3.0, embed-multilingual-v3.0)"
    ),
    SettingDefinition(
        key="embedding.gte_enabled",
        category=SettingCategory.LLM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable GTE-Multilingual embeddings (efficient 305M params, 768 dims)"
    ),

    # Phase 77: New Embedding Models
    SettingDefinition(
        key="embedding.gemma_enabled",
        category=SettingCategory.LLM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable EmbeddingGemma (Google's specialized 768D embedding model)"
    ),
    SettingDefinition(
        key="embedding.stella_enabled",
        category=SettingCategory.LLM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Stella v3 embeddings (69+ MTEB, 1024D base, 2048D large)"
    ),
    SettingDefinition(
        key="embedding.stella_model",
        category=SettingCategory.LLM,
        default_value="stella-base",
        value_type=ValueType.STRING,
        description="Stella model variant: stella-base (1024D), stella-large (2048D)"
    ),
    SettingDefinition(
        key="embedding.dimension_reduction",
        category=SettingCategory.LLM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable dimension reduction for memory savings (PCA/MRL)"
    ),
    SettingDefinition(
        key="embedding.target_dimensions",
        category=SettingCategory.LLM,
        default_value=512,
        value_type=ValueType.NUMBER,
        description="Target dimensions after reduction (256, 512, 768)"
    ),

    # Database Configuration
    SettingDefinition(
        key="database.vector_dimensions",
        category=SettingCategory.DATABASE,
        default_value=1536,
        value_type=ValueType.NUMBER,
        description="Dimensions for vector embeddings"
    ),
    SettingDefinition(
        key="database.index_type",
        category=SettingCategory.DATABASE,
        default_value="hnsw",
        value_type=ValueType.STRING,
        description="Vector index type (hnsw, ivfflat)"
    ),
    SettingDefinition(
        key="database.max_results_per_query",
        category=SettingCategory.DATABASE,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Maximum results returned per search query"
    ),

    # Security Configuration
    SettingDefinition(
        key="security.require_email_verification",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Require email verification for new users"
    ),
    SettingDefinition(
        key="security.enable_2fa",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable two-factor authentication for admin accounts"
    ),
    SettingDefinition(
        key="security.enable_audit_logging",
        category=SettingCategory.SECURITY,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Log all user actions for compliance"
    ),
    SettingDefinition(
        key="security.session_timeout_minutes",
        category=SettingCategory.SECURITY,
        default_value=60,
        value_type=ValueType.NUMBER,
        description="Session timeout in minutes"
    ),

    # Notification Configuration
    SettingDefinition(
        key="notifications.processing_completed",
        category=SettingCategory.NOTIFICATIONS,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Notify when document processing completes"
    ),
    SettingDefinition(
        key="notifications.processing_failed",
        category=SettingCategory.NOTIFICATIONS,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Notify when document processing fails"
    ),
    SettingDefinition(
        key="notifications.cost_alerts",
        category=SettingCategory.NOTIFICATIONS,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Notify when API costs exceed threshold"
    ),

    # ==========================================================================
    # Advanced RAG Configuration
    # ==========================================================================

    # GraphRAG Settings
    SettingDefinition(
        key="rag.graphrag_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable GraphRAG for knowledge graph-based retrieval"
    ),
    SettingDefinition(
        key="rag.graph_max_hops",
        category=SettingCategory.RAG,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum hops for graph traversal (1-3)"
    ),
    SettingDefinition(
        key="rag.entity_extraction_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Extract entities during document processing"
    ),

    # Phase 77: Graph-O1 Efficient Reasoning Settings
    SettingDefinition(
        key="rag.graph_o1_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Graph-O1 efficient reasoning (3-5x faster GraphRAG with beam search)"
    ),
    SettingDefinition(
        key="rag.graph_o1_beam_width",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Graph-O1 beam width - number of paths to explore in parallel (3-10)"
    ),
    SettingDefinition(
        key="rag.graph_o1_confidence_threshold",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Graph-O1 confidence threshold for path pruning (0.5-0.9)"
    ),

    # Agentic RAG Settings
    SettingDefinition(
        key="rag.agentic_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Agentic RAG for complex multi-step queries"
    ),
    SettingDefinition(
        key="rag.agentic_max_iterations",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Maximum ReAct loop iterations (1-10)"
    ),
    SettingDefinition(
        key="rag.auto_detect_complexity",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Auto-detect complex queries for agentic mode"
    ),
    SettingDefinition(
        key="rag.agentic_timeout_seconds",
        category=SettingCategory.RAG,
        default_value=300,
        value_type=ValueType.NUMBER,
        description="Timeout for agentic RAG operations in seconds. Agent mode performs multi-step reasoning which takes longer than simple queries. Increase for complex research tasks, decrease if you want faster responses with potentially incomplete answers (60-600 seconds)"
    ),

    # ==========================================================================
    # Phase 72: Agentic RAG Parallel Execution & DRAGIN/FLARE
    # ==========================================================================
    SettingDefinition(
        key="rag.agentic_max_parallel_queries",
        category=SettingCategory.RAG,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Maximum concurrent sub-queries in agentic mode (2-8, higher = faster but more resource usage)"
    ),
    SettingDefinition(
        key="rag.agentic_dragin_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable DRAGIN/FLARE dynamic retrieval - skip retrieval when LLM is confident (reduces latency)"
    ),
    SettingDefinition(
        key="rag.agentic_retrieval_threshold",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Confidence threshold for FLARE - skip retrieval above this score (0.5-0.9)"
    ),
    SettingDefinition(
        key="rag.agentic_max_tokens",
        category=SettingCategory.RAG,
        default_value=100000,
        value_type=ValueType.NUMBER,
        description="Maximum token budget per agentic query (50000-200000, prevents runaway queries)"
    ),

    # Multimodal RAG Settings
    SettingDefinition(
        key="rag.multimodal_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable multimodal RAG for images/tables/charts"
    ),
    SettingDefinition(
        key="rag.vision_provider",
        category=SettingCategory.RAG,
        default_value="auto",
        value_type=ValueType.STRING,
        description="Vision model provider (auto, ollama, openai, anthropic)"
    ),
    SettingDefinition(
        key="rag.ollama_vision_model",
        category=SettingCategory.RAG,
        default_value="llava",
        value_type=ValueType.STRING,
        description="Ollama vision model for free multimodal (llava, bakllava)"
    ),
    SettingDefinition(
        key="rag.extract_tables",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Extract and index tables from documents"
    ),
    SettingDefinition(
        key="rag.caption_images",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Generate captions for images during indexing"
    ),

    # Image Analysis Settings (extended multimodal)
    SettingDefinition(
        key="rag.image_analysis_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable AI image analysis during document processing"
    ),
    SettingDefinition(
        key="rag.image_duplicate_detection",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Skip analyzing duplicate images (saves time and API costs)"
    ),
    SettingDefinition(
        key="rag.max_images_per_document",
        category=SettingCategory.RAG,
        default_value=50,
        value_type=ValueType.NUMBER,
        description="Maximum images to analyze per document (0 = unlimited)"
    ),
    SettingDefinition(
        key="rag.min_image_size_kb",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Minimum image size to analyze in KB (skip tiny icons)"
    ),
    SettingDefinition(
        key="rag.vision_model_warning_enabled",
        category=SettingCategory.NOTIFICATIONS,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Show warning when vision model is not configured"
    ),

    # Real-Time Updates Settings
    SettingDefinition(
        key="rag.incremental_indexing",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable incremental indexing (update only changed chunks)"
    ),
    SettingDefinition(
        key="rag.freshness_tracking",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Track document freshness and flag stale content"
    ),
    SettingDefinition(
        key="rag.freshness_threshold_days",
        category=SettingCategory.RAG,
        default_value=30,
        value_type=ValueType.NUMBER,
        description="Days until content is considered aging"
    ),
    SettingDefinition(
        key="rag.stale_threshold_days",
        category=SettingCategory.RAG,
        default_value=90,
        value_type=ValueType.NUMBER,
        description="Days until content is considered stale"
    ),

    # Query Suggestions
    SettingDefinition(
        key="rag.suggested_questions_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Generate follow-up question suggestions after answers"
    ),
    SettingDefinition(
        key="rag.suggestions_count",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of suggested questions to generate (1-5)"
    ),

    # Hybrid Search Settings
    SettingDefinition(
        key="rag.graph_weight",
        category=SettingCategory.RAG,
        default_value=0.3,
        value_type=ValueType.NUMBER,
        description="Weight for graph results in hybrid search (0-1)"
    ),
    SettingDefinition(
        key="rag.vector_weight",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Weight for vector results in hybrid search (0-1)"
    ),

    # Retrieval Settings
    SettingDefinition(
        key="rag.top_k",
        category=SettingCategory.RAG,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Number of documents to retrieve per query (higher = broader search)"
    ),
    SettingDefinition(
        key="rag.rerank_results",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable cross-encoder reranking for better relevance ordering"
    ),
    SettingDefinition(
        key="rag.query_expansion_count",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of query variations to generate (improves recall)"
    ),
    SettingDefinition(
        key="rag.similarity_threshold",
        category=SettingCategory.RAG,
        default_value=0.40,
        value_type=ValueType.NUMBER,
        description="Minimum similarity score for document retrieval (0.0-1.0). Higher = stricter filtering, lower = more results but potentially less relevant."
    ),

    # HyDE (Hypothetical Document Embeddings) Settings
    SettingDefinition(
        key="rag.hyde_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable HyDE for short/abstract queries (generates hypothetical documents to improve recall)"
    ),
    SettingDefinition(
        key="rag.hyde_min_query_words",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Use HyDE only for queries shorter than this word count (2-10)"
    ),

    # CRAG (Corrective RAG) Settings
    SettingDefinition(
        key="rag.crag_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable CRAG for auto-correcting low-confidence results"
    ),
    SettingDefinition(
        key="rag.crag_confidence_threshold",
        category=SettingCategory.RAG,
        default_value=0.5,
        value_type=ValueType.NUMBER,
        description="Trigger CRAG when confidence is below this threshold (0.0-1.0)"
    ),

    # Two-Stage Retrieval Settings (ColBERT reranking)
    SettingDefinition(
        key="rag.two_stage_retrieval_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable two-stage retrieval (fast ANN search + ColBERT reranking for 15-30% better precision)"
    ),
    SettingDefinition(
        key="rag.stage1_candidates",
        category=SettingCategory.RAG,
        default_value=150,
        value_type=ValueType.NUMBER,
        description="Number of candidates to retrieve in stage 1 (50-300, higher = better recall but slower)"
    ),
    SettingDefinition(
        key="rag.use_colbert_reranker",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Use ColBERT reranker in stage 2 (else cross-encoder, ColBERT is faster with similar quality)"
    ),

    # Hierarchical Retrieval Settings
    SettingDefinition(
        key="rag.hierarchical_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable hierarchical retrieval (document-first strategy for better diversity across large collections)"
    ),
    SettingDefinition(
        key="rag.hierarchical_doc_limit",
        category=SettingCategory.RAG,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Maximum documents to consider in hierarchical retrieval stage 1 (5-20)"
    ),
    SettingDefinition(
        key="rag.hierarchical_chunks_per_doc",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Chunks to retrieve per document in hierarchical retrieval stage 2 (1-5)"
    ),

    # Semantic Deduplication Settings
    SettingDefinition(
        key="rag.semantic_dedup_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable semantic deduplication (removes near-duplicate chunks from expanded queries)"
    ),
    SettingDefinition(
        key="rag.semantic_dedup_threshold",
        category=SettingCategory.RAG,
        default_value=0.95,
        value_type=ValueType.NUMBER,
        description="Similarity threshold for deduplication (0.0-1.0, higher = more strict)"
    ),

    # Knowledge Graph Integration Settings
    SettingDefinition(
        key="rag.knowledge_graph_enabled",
        category=SettingCategory.RAG,
        default_value=True,  # Enabled by default - feature is mature and provides +15-20% query precision
        value_type=ValueType.BOOLEAN,
        description="Enable knowledge graph-enhanced retrieval (uses entity relationships for better recall)"
    ),
    SettingDefinition(
        key="rag.knowledge_graph_max_hops",
        category=SettingCategory.RAG,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum graph traversal depth (1-3, higher = more related entities)"
    ),

    # Query Expansion Settings
    SettingDefinition(
        key="rag.query_expansion_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable query expansion (generates paraphrased queries for better recall)"
    ),
    SettingDefinition(
        key="rag.parallel_query_search",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Search expanded queries in parallel (faster but uses more resources)"
    ),

    # Query Decomposition Settings
    SettingDefinition(
        key="rag.query_decomposition_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable query decomposition for complex multi-step queries (improves accuracy on comparison/aggregation queries)"
    ),
    SettingDefinition(
        key="rag.decomposition_min_words",
        category=SettingCategory.RAG,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Minimum query word count to trigger decomposition (5-20)"
    ),

    # Verification/Self-RAG Settings
    SettingDefinition(
        key="rag.verification_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable self-verification of retrieved documents"
    ),
    SettingDefinition(
        key="rag.verification_level",
        category=SettingCategory.RAG,
        default_value="quick",
        value_type=ValueType.STRING,
        description="Verification thoroughness (none, quick, standard, thorough)"
    ),

    # Dynamic Query Weighting
    SettingDefinition(
        key="rag.dynamic_weighting_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable dynamic vector/keyword weighting based on query intent"
    ),

    # Contextual Chunking (Anthropic's contextual retrieval approach)
    SettingDefinition(
        key="rag.contextual_chunking_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable contextual chunking (49-67% reduction in failed retrievals)"
    ),
    SettingDefinition(
        key="rag.context_generation_provider",
        category=SettingCategory.RAG,
        default_value="ollama",
        value_type=ValueType.STRING,
        description="LLM provider for context generation (ollama, openai)"
    ),
    SettingDefinition(
        key="rag.context_generation_model",
        category=SettingCategory.RAG,
        default_value="llama3.2",
        value_type=ValueType.STRING,
        description="Model for generating chunk contexts"
    ),

    # Context Sufficiency (Hallucination Prevention)
    SettingDefinition(
        key="rag.context_sufficiency_threshold",
        category=SettingCategory.RAG,
        default_value=0.5,
        value_type=ValueType.NUMBER,
        description="Minimum coverage score to consider context sufficient (0-1)"
    ),
    SettingDefinition(
        key="rag.abstention_threshold",
        category=SettingCategory.RAG,
        default_value=0.3,
        value_type=ValueType.NUMBER,
        description="Coverage below this triggers 'I don't know' response (0-1)"
    ),
    SettingDefinition(
        key="rag.enable_abstention",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Allow system to say 'I don't know' when context is insufficient"
    ),
    SettingDefinition(
        key="rag.conflict_detection_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Detect conflicting information across sources"
    ),

    # ==========================================================================
    # Phase 62/63: Advanced Processing Features (Runtime Configurable)
    # ==========================================================================

    # Answer Refiner Settings
    SettingDefinition(
        key="rag.answer_refiner_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable answer refinement with Self-Refine/CRITIC (+20% quality improvement)"
    ),
    SettingDefinition(
        key="rag.answer_refiner_strategy",
        category=SettingCategory.RAG,
        default_value="self_refine",
        value_type=ValueType.STRING,
        description="Refinement strategy: self_refine (general), critic (tool-verified), cove (hallucination reduction)"
    ),
    SettingDefinition(
        key="rag.answer_refiner_max_iterations",
        category=SettingCategory.RAG,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum refinement iterations (1-5, higher = better quality but slower)"
    ),

    # TTT Compression Settings
    SettingDefinition(
        key="rag.ttt_compression_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable TTT context compression for 35x faster inference on 2M+ token contexts"
    ),
    SettingDefinition(
        key="rag.ttt_compression_ratio",
        category=SettingCategory.RAG,
        default_value=0.5,
        value_type=ValueType.NUMBER,
        description="Target compression ratio (0.3-0.8, lower = more compression)"
    ),

    # Sufficiency Checker Settings
    SettingDefinition(
        key="rag.sufficiency_checker_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable RAG sufficiency detection to skip unnecessary retrieval rounds (ICLR 2025)"
    ),
    SettingDefinition(
        key="rag.sufficiency_threshold",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Confidence threshold for context sufficiency (0.5-0.9)"
    ),

    # Fast Chunking Settings (Chonkie)
    SettingDefinition(
        key="processing.fast_chunking_enabled",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Chonkie fast chunking (33x faster than LangChain, 10-50x less memory)"
    ),
    SettingDefinition(
        key="processing.fast_chunking_strategy",
        category=SettingCategory.PROCESSING,
        default_value="auto",
        value_type=ValueType.STRING,
        description="Chunking strategy: auto, token (fastest), sentence (fast), semantic (balanced), sdpm (best quality)"
    ),
    SettingDefinition(
        key="processing.content_aware_chunking",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Phase 76: Enable content-aware auto chunking - detects code/tables/narrative to select optimal strategy"
    ),

    # Phase 76: Hierarchical Chunking Settings
    SettingDefinition(
        key="processing.hierarchical_chunking_enabled",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable hierarchical chunking for large documents (creates document→section→detail chunks)"
    ),
    SettingDefinition(
        key="processing.hierarchical_threshold_chars",
        category=SettingCategory.PROCESSING,
        default_value=100000,
        value_type=ValueType.NUMBER,
        description="Character threshold to trigger hierarchical chunking (50000-200000)"
    ),
    SettingDefinition(
        key="processing.hierarchical_levels",
        category=SettingCategory.PROCESSING,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of hierarchical levels: 2 (summary+detail) or 3 (summary+section+detail)"
    ),

    # Docling Parser Settings
    SettingDefinition(
        key="processing.docling_parser_enabled",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Docling enterprise parser for 97.9% table extraction accuracy"
    ),

    # Agent Evaluation Settings
    SettingDefinition(
        key="agent.evaluation_enabled",
        category=SettingCategory.GENERAL,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable agent evaluation metrics (Pass^k, hallucination detection, progress tracking)"
    ),

    # Phase 77: Human-in-the-Loop Settings
    SettingDefinition(
        key="agent.hitl_enabled",
        category=SettingCategory.GENERAL,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable human-in-the-loop interrupt support for agent workflows"
    ),
    SettingDefinition(
        key="agent.hitl_approval_timeout",
        category=SettingCategory.GENERAL,
        default_value=300,
        value_type=ValueType.NUMBER,
        description="Timeout in seconds for user approval requests (60-600)"
    ),
    SettingDefinition(
        key="agent.hitl_checkpoint_interval",
        category=SettingCategory.GENERAL,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Create checkpoint every N agent steps for resume capability (1-20)"
    ),
    SettingDefinition(
        key="agent.hitl_critical_actions",
        category=SettingCategory.GENERAL,
        default_value="delete,modify,send",
        value_type=ValueType.STRING,
        description="Comma-separated action keywords that require explicit approval"
    ),

    # Tree of Thoughts Settings
    SettingDefinition(
        key="rag.tree_of_thoughts_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Tree of Thoughts for complex analytical queries (multi-path reasoning)"
    ),
    SettingDefinition(
        key="rag.tot_max_depth",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Maximum reasoning tree depth (2-5)"
    ),
    SettingDefinition(
        key="rag.tot_branching_factor",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Branching factor per thought node (2-5)"
    ),

    # OCR Configuration
    SettingDefinition(
        key="ocr.provider",
        category=SettingCategory.OCR,
        default_value="tesseract",
        value_type=ValueType.STRING,
        description="OCR provider (paddleocr, easyocr, tesseract, auto)"
    ),
    SettingDefinition(
        key="ocr.paddle.variant",
        category=SettingCategory.OCR,
        default_value="server",
        value_type=ValueType.STRING,
        description="PaddleOCR model variant (server=accurate, mobile=fast)"
    ),
    SettingDefinition(
        key="ocr.paddle.languages",
        category=SettingCategory.OCR,
        default_value=["en", "de"],
        value_type=ValueType.JSON,
        description="List of language codes for OCR"
    ),
    SettingDefinition(
        key="ocr.paddle.model_dir",
        category=SettingCategory.OCR,
        default_value="./data/paddle_models",
        value_type=ValueType.STRING,
        description="Directory for PaddleOCR model cache"
    ),
    SettingDefinition(
        key="ocr.paddle.auto_download",
        category=SettingCategory.OCR,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Auto-download models on startup"
    ),
    SettingDefinition(
        key="ocr.tesseract.fallback_enabled",
        category=SettingCategory.OCR,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Fall back to Tesseract if PaddleOCR fails"
    ),
    # EasyOCR Settings
    SettingDefinition(
        key="ocr.easyocr.languages",
        category=SettingCategory.OCR,
        default_value=["en"],
        value_type=ValueType.JSON,
        description="List of language codes for EasyOCR"
    ),
    SettingDefinition(
        key="ocr.easyocr.use_gpu",
        category=SettingCategory.OCR,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Use GPU acceleration for EasyOCR (if available)"
    ),

    # ==========================================================================
    # Phase 76: Language Detection & Auto-Configuration
    # ==========================================================================
    SettingDefinition(
        key="ocr.auto_detect_language",
        category=SettingCategory.OCR,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Auto-detect document language for OCR (requires langdetect package)"
    ),
    SettingDefinition(
        key="ocr.default_language",
        category=SettingCategory.OCR,
        default_value="eng",
        value_type=ValueType.STRING,
        description="Default OCR language code when auto-detect is disabled or fails"
    ),
    SettingDefinition(
        key="ocr.multi_language_enabled",
        category=SettingCategory.OCR,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable multi-language OCR (e.g., 'eng+deu' for English + German)"
    ),

    # ==========================================================================
    # Job Queue & Caching Configuration
    # ==========================================================================

    # Celery/Redis Settings
    SettingDefinition(
        key="queue.celery_enabled",
        category=SettingCategory.GENERAL,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Celery for async document processing (requires Redis)"
    ),
    SettingDefinition(
        key="queue.redis_url",
        category=SettingCategory.GENERAL,
        default_value="redis://localhost:6379/0",
        value_type=ValueType.STRING,
        description="Redis connection URL for job queue and caching"
    ),
    SettingDefinition(
        key="queue.max_workers",
        category=SettingCategory.GENERAL,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Maximum concurrent Celery workers"
    ),

    # ==========================================================================
    # Server Configuration (Scalability)
    # ==========================================================================
    SettingDefinition(
        key="server.uvicorn_workers",
        category=SettingCategory.GENERAL,
        default_value=1,
        value_type=ValueType.NUMBER,
        description="Number of Uvicorn worker processes (1-16). More workers = more concurrent users. Requires server restart."
    ),
    SettingDefinition(
        key="server.db_pool_size",
        category=SettingCategory.DATABASE,
        default_value=30,
        value_type=ValueType.NUMBER,
        description="Database connection pool size (10-200). Increase for high concurrency."
    ),
    SettingDefinition(
        key="server.db_pool_overflow",
        category=SettingCategory.DATABASE,
        default_value=20,
        value_type=ValueType.NUMBER,
        description="Maximum overflow connections beyond pool size (10-100)."
    ),

    # Embedding Cache Settings
    SettingDefinition(
        key="cache.embedding_cache_enabled",
        category=SettingCategory.GENERAL,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable embedding deduplication cache (uses Redis if available, falls back to memory)"
    ),
    SettingDefinition(
        key="cache.embedding_cache_ttl_days",
        category=SettingCategory.GENERAL,
        default_value=7,
        value_type=ValueType.NUMBER,
        description="Embedding cache TTL in days"
    ),

    # ==========================================================================
    # Phase 75: Distributed Cache Settings
    # ==========================================================================
    SettingDefinition(
        key="cache.distributed_enabled",
        category=SettingCategory.GENERAL,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable distributed cache invalidation via Redis pub/sub (multi-instance support)"
    ),
    SettingDefinition(
        key="cache.pubsub_enabled",
        category=SettingCategory.GENERAL,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Redis pub/sub listener for cross-instance cache sync"
    ),
    SettingDefinition(
        key="cache.invalidation_broadcast",
        category=SettingCategory.GENERAL,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Broadcast cache invalidations to other instances when settings change"
    ),

    # ==========================================================================
    # Parallel Processing Configuration (Ray)
    # ==========================================================================
    SettingDefinition(
        key="processing.ray_enabled",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Ray for distributed parallel processing of large document batches"
    ),
    SettingDefinition(
        key="processing.ray_address",
        category=SettingCategory.PROCESSING,
        default_value="",
        value_type=ValueType.STRING,
        description="Ray cluster address (empty for local, or 'ray://host:10001' for remote cluster)"
    ),
    SettingDefinition(
        key="processing.ray_num_cpus",
        category=SettingCategory.PROCESSING,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Maximum CPUs for Ray tasks (0 for all available)"
    ),
    SettingDefinition(
        key="processing.ray_num_gpus",
        category=SettingCategory.PROCESSING,
        default_value=0,
        value_type=ValueType.NUMBER,
        description="Maximum GPUs for Ray tasks (for GPU-accelerated operations)"
    ),
    SettingDefinition(
        key="processing.ray_memory_limit_gb",
        category=SettingCategory.PROCESSING,
        default_value=8,
        value_type=ValueType.NUMBER,
        description="Memory limit per Ray worker in GB"
    ),

    # ==========================================================================
    # Phase 71: Parallel Processing Settings
    # ==========================================================================
    SettingDefinition(
        key="processing.sequential_upload_processing",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Process uploaded files one at a time (sequential). When disabled, files are processed in parallel which is faster but uses more CPU/memory."
    ),
    SettingDefinition(
        key="processing.max_parallel_uploads",
        category=SettingCategory.PROCESSING,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum number of files to process simultaneously when parallel processing is enabled (1-8)"
    ),
    SettingDefinition(
        key="processing.max_concurrent_pdf_pages",
        category=SettingCategory.PROCESSING,
        default_value=8,
        value_type=ValueType.NUMBER,
        description="Maximum concurrent pages to process in PDF OCR (4-16, higher = faster but more memory)"
    ),
    SettingDefinition(
        key="processing.max_concurrent_image_captions",
        category=SettingCategory.PROCESSING,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Maximum concurrent image captioning requests (2-8, limited by vision LLM rate limits)"
    ),
    SettingDefinition(
        key="processing.settings_cache_ttl",
        category=SettingCategory.PROCESSING,
        default_value=300,
        value_type=ValueType.NUMBER,
        description="TTL for settings cache in seconds (60-600, admin changes apply after expiry)"
    ),

    # ==========================================================================
    # Document Generation Configuration
    # ==========================================================================

    SettingDefinition(
        key="generation.default_format",
        category=SettingCategory.GENERATION,
        default_value="docx",
        value_type=ValueType.STRING,
        description="Default output format for document generation (docx, pptx, pdf, md, html)"
    ),
    SettingDefinition(
        key="generation.include_images",
        category=SettingCategory.GENERATION,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Include AI-generated or stock images in generated documents"
    ),
    SettingDefinition(
        key="generation.image_backend",
        category=SettingCategory.GENERATION,
        default_value="picsum",
        value_type=ValueType.STRING,
        description="Image source backend (picsum, unsplash, pexels, openai, stability, automatic1111, disabled)"
    ),
    SettingDefinition(
        key="generation.default_tone",
        category=SettingCategory.GENERATION,
        default_value="professional",
        value_type=ValueType.STRING,
        description="Default writing tone (professional, casual, academic, creative)"
    ),
    SettingDefinition(
        key="generation.default_style",
        category=SettingCategory.GENERATION,
        default_value="business",
        value_type=ValueType.STRING,
        description="Default document style (business, academic, creative, technical)"
    ),
    SettingDefinition(
        key="generation.max_sections",
        category=SettingCategory.GENERATION,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Maximum number of sections to generate (1-20)"
    ),
    SettingDefinition(
        key="generation.include_sources",
        category=SettingCategory.GENERATION,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Include source citations in generated documents"
    ),
    SettingDefinition(
        key="generation.dual_mode",
        category=SettingCategory.GENERATION,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable dual mode: combine document knowledge with general AI knowledge for richer content"
    ),
    SettingDefinition(
        key="generation.dual_mode_blend",
        category=SettingCategory.GENERATION,
        default_value="docs_first",
        value_type=ValueType.STRING,
        description="Dual mode blend strategy: 'docs_first' (documents primary, AI fills gaps) or 'merged' (equal blend)"
    ),
    SettingDefinition(
        key="generation.auto_charts",
        category=SettingCategory.GENERATION,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Auto-generate charts from data in source documents"
    ),
    SettingDefinition(
        key="generation.chart_style",
        category=SettingCategory.GENERATION,
        default_value="business",
        value_type=ValueType.STRING,
        description="Chart styling theme: business, academic, minimal"
    ),
    SettingDefinition(
        key="generation.chart_dpi",
        category=SettingCategory.GENERATION,
        default_value=150,
        value_type=ValueType.NUMBER,
        description="Chart image resolution (DPI) - higher = larger file size (100-300)"
    ),

    # Quality Review Settings
    SettingDefinition(
        key="generation.enable_quality_review",
        category=SettingCategory.GENERATION,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable LLM-based content review before rendering. Reviews slide/page content for issues like text overflow, incomplete bullets, and formatting problems."
    ),
    SettingDefinition(
        key="generation.min_quality_score",
        category=SettingCategory.GENERATION,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Minimum quality score threshold (0.0-1.0). Content below this score triggers auto-fix attempts."
    ),

    # Vision-Based Review Settings (PPTX)
    SettingDefinition(
        key="generation.enable_vision_review",
        category=SettingCategory.GENERATION,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable vision-based slide review for PPTX. Renders slides to images and uses vision LLM to detect visual issues like text overflow, poor contrast, or layout problems. Resource-intensive - requires LibreOffice for rendering."
    ),
    SettingDefinition(
        key="generation.vision_review_model",
        category=SettingCategory.GENERATION,
        default_value="auto",
        value_type=ValueType.STRING,
        description="Vision model for slide review (auto, claude-3-sonnet, gpt-4-vision, ollama-llava). 'auto' uses the configured vision provider from RAG settings."
    ),
    SettingDefinition(
        key="generation.vision_review_all_slides",
        category=SettingCategory.GENERATION,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Review all slides with vision model. When false, only reviews content slides (skips title, TOC, sources). Enable for thorough review at higher cost."
    ),

    # LLM Rewrite Settings
    SettingDefinition(
        key="generation.enable_llm_rewrite",
        category=SettingCategory.GENERATION,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Use LLM to intelligently rewrite text that exceeds slide/page constraints, preserving key insights. When disabled, uses simple truncation."
    ),
    SettingDefinition(
        key="generation.llm_rewrite_model",
        category=SettingCategory.GENERATION,
        default_value="auto",
        value_type=ValueType.STRING,
        description="LLM model for content rewriting (auto, gpt-4o-mini, claude-3-haiku). 'auto' uses a fast model for quick rewrites."
    ),

    # ==========================================================================
    # Web Scraping Configuration
    # ==========================================================================

    SettingDefinition(
        key="scraping.use_crawl4ai",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Use Crawl4AI for web scraping (recommended). When enabled: supports JavaScript-rendered pages (React, Vue, Angular), handles bot protection, produces LLM-optimized markdown. When disabled: uses basic HTTP scraping which is faster but only works for static HTML pages."
    ),
    SettingDefinition(
        key="scraping.headless_browser",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Run browser in headless mode (no visible window). Disable for debugging scraping issues."
    ),
    SettingDefinition(
        key="scraping.timeout_seconds",
        category=SettingCategory.SCRAPING,
        default_value=30,
        value_type=ValueType.NUMBER,
        description="Maximum time to wait for page load (10-120 seconds)"
    ),
    SettingDefinition(
        key="scraping.extract_links",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Extract and index links found on scraped pages"
    ),
    SettingDefinition(
        key="scraping.extract_images",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Extract image URLs from scraped pages"
    ),
    SettingDefinition(
        key="scraping.max_depth",
        category=SettingCategory.SCRAPING,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum crawl depth when following links (0=single page, 1-5 for multi-page)"
    ),
    SettingDefinition(
        key="scraping.respect_robots_txt",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Respect robots.txt rules when scraping websites"
    ),
    SettingDefinition(
        key="scraping.rate_limit_ms",
        category=SettingCategory.SCRAPING,
        default_value=1000,
        value_type=ValueType.NUMBER,
        description="Delay between requests in milliseconds to avoid overloading servers (500-5000)"
    ),
    SettingDefinition(
        key="scraping.proxy_enabled",
        category=SettingCategory.SCRAPING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable proxy rotation for web scraping. Uses a pool of proxy servers to distribute requests and avoid IP-based blocking."
    ),
    SettingDefinition(
        key="scraping.proxy_list",
        category=SettingCategory.SCRAPING,
        default_value="",
        value_type=ValueType.STRING,
        description="Comma-separated list of proxy URLs (e.g., http://proxy1:8080,socks5://proxy2:1080). Leave empty if proxy is disabled."
    ),
    SettingDefinition(
        key="scraping.proxy_rotation_strategy",
        category=SettingCategory.SCRAPING,
        default_value="round_robin",
        value_type=ValueType.STRING,
        description="Proxy rotation strategy: 'round_robin' cycles through proxies sequentially, 'random' selects randomly each request."
    ),
    SettingDefinition(
        key="scraping.jina_reader_fallback",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Use Jina Reader API (r.jina.ai) as fallback when Crawl4AI and basic HTTP scraping fail. Free tier: up to 1000 requests/day."
    ),
    SettingDefinition(
        key="scraping.adaptive_crawling",
        category=SettingCategory.SCRAPING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable adaptive crawling with confidence-based stopping. Automatically determines when a site has been sufficiently crawled based on content coverage and saturation."
    ),
    SettingDefinition(
        key="scraping.crash_recovery_enabled",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable crash recovery for long-running crawl jobs. Persists crawl state to allow resuming after failures."
    ),

    # ==========================================================================
    # Phase 65: Scaling & Performance (1M+ Documents)
    # ==========================================================================

    # HNSW Index Optimization
    SettingDefinition(
        key="vectorstore.hnsw_ef_construction",
        category=SettingCategory.DATABASE,
        default_value=200,
        value_type=ValueType.NUMBER,
        description="HNSW ef_construction parameter (100-400, higher = better recall, slower build)"
    ),
    SettingDefinition(
        key="vectorstore.hnsw_m",
        category=SettingCategory.DATABASE,
        default_value=32,
        value_type=ValueType.NUMBER,
        description="HNSW M parameter - connections per node (16-64, higher = better recall, more memory)"
    ),
    SettingDefinition(
        key="vectorstore.hnsw_ef_search",
        category=SettingCategory.DATABASE,
        default_value=128,
        value_type=ValueType.NUMBER,
        description="HNSW ef_search parameter (64-256, higher = better recall, slower search)"
    ),

    # Binary Quantization (32x memory reduction)
    SettingDefinition(
        key="vectorstore.binary_quantization_enabled",
        category=SettingCategory.DATABASE,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable binary quantization for 32x memory reduction (requires reranking for accuracy)"
    ),
    SettingDefinition(
        key="vectorstore.quantization_rerank_multiplier",
        category=SettingCategory.DATABASE,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Retrieve N times more candidates for reranking after quantized search (5-20)"
    ),

    # Late Chunking (Context Preservation)
    SettingDefinition(
        key="vectorstore.late_chunking_enabled",
        category=SettingCategory.DATABASE,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable late chunking - embed full doc first, preserves cross-chunk context (+15% recall)"
    ),

    # GPU Acceleration
    SettingDefinition(
        key="vectorstore.gpu_acceleration_enabled",
        category=SettingCategory.DATABASE,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable FAISS GPU acceleration via cuVS (12x faster indexing, 8x lower latency)"
    ),

    # ==========================================================================
    # Phase 65: Search Engine Quality Ranking
    # ==========================================================================

    # BM25 Scoring
    SettingDefinition(
        key="search.bm25_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable BM25 scoring for keyword search (better than TF-IDF for term saturation)"
    ),
    SettingDefinition(
        key="search.bm25_k1",
        category=SettingCategory.RAG,
        default_value=1.5,
        value_type=ValueType.NUMBER,
        description="BM25 k1 parameter - term frequency saturation (1.2-2.0)"
    ),
    SettingDefinition(
        key="search.bm25_b",
        category=SettingCategory.RAG,
        default_value=0.75,
        value_type=ValueType.NUMBER,
        description="BM25 b parameter - document length normalization (0.0-1.0)"
    ),

    # Field Boosting
    SettingDefinition(
        key="search.field_boosting_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable field-specific boosting (title matches score higher than body)"
    ),
    SettingDefinition(
        key="search.title_boost",
        category=SettingCategory.RAG,
        default_value=3.0,
        value_type=ValueType.NUMBER,
        description="Boost multiplier for section title matches (1.0-5.0)"
    ),
    SettingDefinition(
        key="search.document_title_boost",
        category=SettingCategory.RAG,
        default_value=2.5,
        value_type=ValueType.NUMBER,
        description="Boost multiplier for document title matches (1.0-5.0)"
    ),

    # Learning to Rank
    SettingDefinition(
        key="search.ltr_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Learning-to-Rank reranking (requires training data from click logs)"
    ),

    # Spell Correction
    SettingDefinition(
        key="search.spell_correction_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable spell correction for queries with no results"
    ),
    SettingDefinition(
        key="search.spell_correction_max_distance",
        category=SettingCategory.RAG,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum edit distance for spell correction (1-3)"
    ),

    # Freshness Boosting
    SettingDefinition(
        key="search.freshness_boost_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Boost recent documents in search results"
    ),
    SettingDefinition(
        key="search.freshness_decay_rate",
        category=SettingCategory.RAG,
        default_value=0.1,
        value_type=ValueType.NUMBER,
        description="Exponential decay rate for document freshness (0.05-0.2)"
    ),

    # Query Caching
    SettingDefinition(
        key="search.query_cache_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable query result caching for faster repeated queries"
    ),
    SettingDefinition(
        key="search.query_cache_ttl_seconds",
        category=SettingCategory.RAG,
        default_value=3600,
        value_type=ValueType.NUMBER,
        description="Query cache TTL in seconds (300-86400)"
    ),

    # ==========================================================================
    # Phase 65: Advanced RAG Pipeline
    # ==========================================================================

    # Query Classifier
    SettingDefinition(
        key="rag.query_classifier_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable query classification for intent-based routing (+25% accuracy)"
    ),

    # Self-RAG
    SettingDefinition(
        key="rag.self_rag_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Self-RAG with reflection for +18% factuality improvement"
    ),
    SettingDefinition(
        key="rag.self_rag_max_iterations",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Maximum Self-RAG reflection iterations (1-5)"
    ),

    # Adaptive RAG Routing
    SettingDefinition(
        key="rag.adaptive_routing_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable adaptive routing based on query complexity (simple/multi-hop/analytical)"
    ),

    # Speculative RAG
    SettingDefinition(
        key="rag.speculative_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Speculative RAG for 50% latency reduction via parallel draft generation"
    ),
    SettingDefinition(
        key="rag.speculative_num_drafts",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of parallel drafts to generate (2-5)"
    ),

    # Streaming with Citations
    SettingDefinition(
        key="rag.streaming_citations_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable real-time citation highlighting during streaming"
    ),

    # ==========================================================================
    # Phase 65: World-Class Web Crawler
    # ==========================================================================

    # Stealth Mode
    SettingDefinition(
        key="crawler.stealth_mode_enabled",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable stealth mode for anti-bot bypass (fingerprint spoofing, user simulation)"
    ),
    SettingDefinition(
        key="crawler.magic_mode_enabled",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Crawl4AI magic mode for advanced anti-detection"
    ),

    # LLM Extraction
    SettingDefinition(
        key="crawler.llm_extraction_enabled",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable LLM-powered content extraction from crawled pages"
    ),
    SettingDefinition(
        key="crawler.smart_extraction_enabled",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Auto-detect and extract key information (entities, facts, dates) from pages"
    ),

    # Site Crawling
    SettingDefinition(
        key="crawler.max_pages_per_site",
        category=SettingCategory.SCRAPING,
        default_value=100,
        value_type=ValueType.NUMBER,
        description="Maximum pages to crawl per site (10-1000)"
    ),
    SettingDefinition(
        key="crawler.user_agent_rotation_enabled",
        category=SettingCategory.SCRAPING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Rotate user agents to avoid detection"
    ),

    # ==========================================================================
    # Phase 65: Interactive Database Querying (Text-to-SQL)
    # ==========================================================================

    SettingDefinition(
        key="database.text_to_sql_enabled",
        category=SettingCategory.DATABASE,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable natural language to SQL conversion (DAIL-SQL style, 86.6% accuracy)"
    ),
    SettingDefinition(
        key="database.sql_validation_enabled",
        category=SettingCategory.DATABASE,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable multi-layer SQL validation (syntax, security, cost estimation)"
    ),
    SettingDefinition(
        key="database.sql_injection_prevention",
        category=SettingCategory.DATABASE,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable SQL injection prevention checks"
    ),
    SettingDefinition(
        key="database.query_timeout_seconds",
        category=SettingCategory.DATABASE,
        default_value=30,
        value_type=ValueType.NUMBER,
        description="Maximum query execution time in seconds (10-120)"
    ),
    SettingDefinition(
        key="database.query_cost_limit",
        category=SettingCategory.DATABASE,
        default_value=10000,
        value_type=ValueType.NUMBER,
        description="Maximum estimated query cost before warning (relative units)"
    ),

    # Auto-Visualization
    SettingDefinition(
        key="database.auto_visualization_enabled",
        category=SettingCategory.DATABASE,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Auto-generate charts from query results (LIDA-style)"
    ),
    SettingDefinition(
        key="database.result_summarization_enabled",
        category=SettingCategory.DATABASE,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Generate natural language summaries of query results"
    ),

    # Interactive Query Building
    SettingDefinition(
        key="database.interactive_mode_enabled",
        category=SettingCategory.DATABASE,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable interactive query building with clarification requests"
    ),
    SettingDefinition(
        key="database.query_preview_enabled",
        category=SettingCategory.DATABASE,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Show query preview and cost estimate before execution"
    ),

    # ==========================================================================
    # Phase 65: Enterprise Features
    # ==========================================================================

    # Access Control (ABAC)
    SettingDefinition(
        key="enterprise.access_control_enabled",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Attribute-Based Access Control for document retrieval"
    ),
    SettingDefinition(
        key="enterprise.access_control_mode",
        category=SettingCategory.SECURITY,
        default_value="rbac",
        value_type=ValueType.STRING,
        description="Access control mode: rbac (role-based), abac (attribute-based), rebac (relationship-based)"
    ),

    # Audit Logging
    SettingDefinition(
        key="enterprise.audit_logging_enabled",
        category=SettingCategory.SECURITY,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable comprehensive audit logging for compliance"
    ),
    SettingDefinition(
        key="enterprise.audit_log_retention_days",
        category=SettingCategory.SECURITY,
        default_value=365,
        value_type=ValueType.NUMBER,
        description="Audit log retention period in days (30-2555 for 7 years GDPR)"
    ),

    # PII Detection
    SettingDefinition(
        key="enterprise.pii_detection_enabled",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable PII detection and masking in query results"
    ),
    SettingDefinition(
        key="enterprise.pii_masking_mode",
        category=SettingCategory.SECURITY,
        default_value="redact",
        value_type=ValueType.STRING,
        description="PII masking mode: redact (remove), tokenize (replace), encrypt (reversible)"
    ),

    # Multi-Tenant Isolation
    SettingDefinition(
        key="enterprise.multi_tenant_enabled",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable multi-tenant data isolation"
    ),

    # Embedding Inversion Defense (OWASP LLM08:2025)
    SettingDefinition(
        key="security.embedding_defense_enabled",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable embedding inversion defense (noise injection, dimension shuffle, norm clipping)"
    ),
    SettingDefinition(
        key="security.defense_noise_scale",
        category=SettingCategory.SECURITY,
        default_value=0.01,
        value_type=ValueType.NUMBER,
        description="Gaussian noise standard deviation added to embeddings (0.001-0.1)"
    ),
    SettingDefinition(
        key="security.defense_clip_norm",
        category=SettingCategory.SECURITY,
        default_value=1.0,
        value_type=ValueType.NUMBER,
        description="Maximum L2 norm for stored embeddings to limit information leakage (0.5-2.0)"
    ),

    # ==========================================================================
    # Phase 66: Adaptive RAG Routing
    # ==========================================================================

    SettingDefinition(
        key="rag.adaptive_routing_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable intelligent query routing to optimal retrieval strategies (DIRECT, HYBRID, TWO_STAGE, AGENTIC, GRAPH)"
    ),
    SettingDefinition(
        key="rag.rag_fusion_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable RAG-Fusion for complex queries (multi-query with Reciprocal Rank Fusion)"
    ),
    SettingDefinition(
        key="rag.rag_fusion_variations",
        category=SettingCategory.RAG,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Number of query variations to generate for RAG-Fusion (2-8)"
    ),
    SettingDefinition(
        key="rag.lazy_graphrag_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable LazyGraphRAG for cost-efficient knowledge graph retrieval (99% cost reduction vs standard GraphRAG)"
    ),
    SettingDefinition(
        key="rag.lazy_graphrag_max_communities",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Maximum communities to summarize per query in LazyGraphRAG (1-10)"
    ),
    SettingDefinition(
        key="rag.stepback_prompting_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Step-Back prompting for complex analytical queries (abstract reasoning)"
    ),
    SettingDefinition(
        key="rag.stepback_max_background",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Maximum background chunks to retrieve for step-back context (1-5)"
    ),
    SettingDefinition(
        key="rag.context_compression_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable context compression to reduce token usage while preserving relevance"
    ),
    SettingDefinition(
        key="rag.context_compression_target_tokens",
        category=SettingCategory.RAG,
        default_value=2000,
        value_type=ValueType.NUMBER,
        description="Target token count for compressed context (500-5000)"
    ),
    SettingDefinition(
        key="rag.context_compression_use_llm",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Use LLM for semantic context compression (slower but more accurate)"
    ),

    # Phase 77: AttentionRAG Compression Settings
    SettingDefinition(
        key="rag.attention_rag_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable AttentionRAG compression (6.3x better than LLMLingua, uses attention scores)"
    ),
    SettingDefinition(
        key="rag.attention_rag_mode",
        category=SettingCategory.RAG,
        default_value="moderate",
        value_type=ValueType.STRING,
        description="AttentionRAG compression mode: light (1.25x), moderate (2x), aggressive (4x), extreme (6.6x), adaptive"
    ),
    SettingDefinition(
        key="rag.attention_rag_unit",
        category=SettingCategory.RAG,
        default_value="sentence",
        value_type=ValueType.STRING,
        description="AttentionRAG compression unit: token, sentence, paragraph"
    ),
    SettingDefinition(
        key="rag.llmlingua_compression_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable LLMLingua-2 context compression (3-6x token reduction while preserving answer quality)"
    ),
    # Speculative RAG
    SettingDefinition(
        key="rag.speculative_rag_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Speculative RAG: parallel draft generation from document subsets, verified by main model (15-50% latency reduction)"
    ),
    SettingDefinition(
        key="rag.speculative_rag_num_drafts",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of parallel draft answers to generate in Speculative RAG (2-5)"
    ),

    # Matryoshka Adaptive Retrieval
    SettingDefinition(
        key="rag.matryoshka_retrieval_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Matryoshka adaptive retrieval: two-stage search using low-dim embeddings for fast shortlisting, then full-dim reranking (5-14x speed improvement)"
    ),
    SettingDefinition(
        key="rag.matryoshka_shortlist_factor",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Matryoshka shortlist multiplier: stage 1 retrieves top_k * factor candidates for reranking (higher = better recall, slower rerank)"
    ),
    SettingDefinition(
        key="rag.matryoshka_fast_dims",
        category=SettingCategory.RAG,
        default_value=128,
        value_type=ValueType.NUMBER,
        description="Number of leading embedding dimensions for Matryoshka fast pass (128 recommended; must be < full embedding dims)"
    ),

    # Smart Model Routing
    SettingDefinition(
        key="rag.smart_model_routing_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Route queries to cost-optimal LLM models based on complexity (simple→cheap, complex→premium)"
    ),
    SettingDefinition(
        key="rag.smart_routing_simple_model",
        category=SettingCategory.RAG,
        default_value="",
        value_type=ValueType.STRING,
        description="Model for simple factual queries (e.g., 'gpt-4o-mini', 'ollama/llama3.2'). Empty = auto-detect."
    ),
    SettingDefinition(
        key="rag.smart_routing_complex_model",
        category=SettingCategory.RAG,
        default_value="",
        value_type=ValueType.STRING,
        description="Model for complex multi-hop queries (e.g., 'gpt-4o', 'anthropic/claude-3-opus'). Empty = auto-detect."
    ),
    SettingDefinition(
        key="rag.context_reorder_strategy",
        category=SettingCategory.RAG,
        default_value="sandwich",
        value_type=ValueType.STRING,
        description="Context reordering strategy to mitigate 'lost in the middle' effect (sandwich, front_loaded, alternating)"
    ),

    # ==========================================================================
    # Phase 66: RAG Evaluation
    # ==========================================================================

    SettingDefinition(
        key="rag.evaluation_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable automatic RAG response evaluation with RAGAS metrics"
    ),
    SettingDefinition(
        key="rag.evaluation_sample_rate",
        category=SettingCategory.RAG,
        default_value=0.1,
        value_type=ValueType.NUMBER,
        description="Fraction of queries to evaluate (0.0-1.0)"
    ),
    SettingDefinition(
        key="rag.min_faithfulness_score",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Minimum faithfulness score threshold (0.0-1.0) - flags potential hallucinations"
    ),

    # ==========================================================================
    # Phase 70: Agent Memory Cache Settings
    # ==========================================================================
    SettingDefinition(
        key="rag.agent_memory_cache_max_size",
        category=SettingCategory.RAG,
        default_value=1000,
        value_type=ValueType.NUMBER,
        description="Maximum entries in agent memory LRU cache (500-5000, prevents unbounded growth)"
    ),
    SettingDefinition(
        key="rag.agent_memory_cache_ttl",
        category=SettingCategory.RAG,
        default_value=3600,
        value_type=ValueType.NUMBER,
        description="TTL for agent memory cache entries in seconds (1800-7200)"
    ),
    SettingDefinition(
        key="rag.generative_cache_max_size",
        category=SettingCategory.RAG,
        default_value=10000,
        value_type=ValueType.NUMBER,
        description="Maximum entries in generative cache FAISS index (5000-50000)"
    ),
    SettingDefinition(
        key="rag.semantic_cache_rebuild_threshold",
        category=SettingCategory.RAG,
        default_value=100,
        value_type=ValueType.NUMBER,
        description="Number of new entries before triggering FAISS index rebuild (50-500)"
    ),

    # ==========================================================================
    # Phase 66: User Personalization
    # ==========================================================================

    SettingDefinition(
        key="rag.user_personalization_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable user preference learning and personalized responses"
    ),
    SettingDefinition(
        key="rag.personalization_min_feedback",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Minimum feedback entries required before adapting preferences (3-20)"
    ),
    SettingDefinition(
        key="rag.personalization_profile_ttl_days",
        category=SettingCategory.RAG,
        default_value=90,
        value_type=ValueType.NUMBER,
        description="Days to retain user profile data (30-365)"
    ),

    # ==========================================================================
    # Phase 66: Dependency-based KG Extraction
    # ==========================================================================

    SettingDefinition(
        key="kg.dependency_extraction_enabled",
        category=SettingCategory.INGESTION,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable fast dependency parsing for entity extraction (94% LLM quality, 80% cost savings)"
    ),
    SettingDefinition(
        key="kg.dependency_complexity_threshold",
        category=SettingCategory.INGESTION,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Text complexity threshold above which to use LLM extraction (0.0-1.0)"
    ),
    SettingDefinition(
        key="kg.spacy_model",
        category=SettingCategory.INGESTION,
        default_value="en_core_web_sm",
        value_type=ValueType.STRING,
        description="spaCy model for dependency parsing (en_core_web_sm, en_core_web_md, en_core_web_lg)"
    ),
    SettingDefinition(
        key="kg.extraction_concurrency",
        category=SettingCategory.INGESTION,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Number of concurrent documents to process during KG extraction (1-16)"
    ),
    SettingDefinition(
        key="kg.ray_task_timeout",
        category=SettingCategory.INGESTION,
        default_value=1800,
        value_type=ValueType.NUMBER,
        description="Timeout in seconds for each document during Ray KG extraction (300-7200)"
    ),

    # ==========================================================================
    # Phase 66: TTS Provider Configuration
    # ==========================================================================

    SettingDefinition(
        key="tts.default_provider",
        category=SettingCategory.AUDIO,
        default_value="openai",
        value_type=ValueType.STRING,
        description="Default TTS provider (openai, elevenlabs, chatterbox, cosyvoice, edge)"
    ),
    SettingDefinition(
        key="tts.fallback_chain",
        category=SettingCategory.AUDIO,
        default_value='["cosyvoice", "chatterbox", "fish_speech", "openai"]',
        value_type=ValueType.JSON,
        description="Fallback provider order when primary fails"
    ),
    SettingDefinition(
        key="tts.ultra_fast_provider",
        category=SettingCategory.AUDIO,
        default_value="cosyvoice",
        value_type=ValueType.STRING,
        description="Ultra-fast TTS provider for real-time streaming"
    ),
    SettingDefinition(
        key="tts.chatterbox_enabled",
        category=SettingCategory.AUDIO,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Chatterbox TTS (Resemble AI open-source)"
    ),
    SettingDefinition(
        key="tts.chatterbox_exaggeration",
        category=SettingCategory.AUDIO,
        default_value=0.5,
        value_type=ValueType.NUMBER,
        description="Emotional exaggeration for Chatterbox (0.0-1.0)"
    ),
    SettingDefinition(
        key="tts.chatterbox_cfg_weight",
        category=SettingCategory.AUDIO,
        default_value=0.5,
        value_type=ValueType.NUMBER,
        description="CFG weight for Chatterbox generation (0.0-1.0)"
    ),
    SettingDefinition(
        key="tts.cosyvoice_enabled",
        category=SettingCategory.AUDIO,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable CosyVoice2 TTS (Alibaba open-source, 150ms latency)"
    ),
    SettingDefinition(
        key="tts.fish_speech_enabled",
        category=SettingCategory.AUDIO,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Fish Speech TTS (multilingual, ELO 1339)"
    ),

    # ==========================================================================
    # Phase 93: DSPy Prompt Optimization
    # ==========================================================================

    SettingDefinition(
        key="rag.dspy_optimization_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable DSPy prompt optimization (admin can trigger compilation runs)"
    ),
    SettingDefinition(
        key="rag.dspy_default_optimizer",
        category=SettingCategory.RAG,
        default_value="bootstrap_few_shot",
        value_type=ValueType.STRING,
        description="Default DSPy optimizer: bootstrap_few_shot (stable, 20+ examples) or miprov2 (100+ examples)"
    ),
    SettingDefinition(
        key="rag.dspy_min_examples",
        category=SettingCategory.RAG,
        default_value=20,
        value_type=ValueType.NUMBER,
        description="Minimum training examples required before DSPy optimization can run (5-500)"
    ),
    SettingDefinition(
        key="rag.dspy_inference_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Use DSPy modules at inference time (default: compilation-only, exported as text prompts)"
    ),

    # Phase 95K: Content Freshness Scoring
    SettingDefinition(
        key="rag.content_freshness_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable content freshness scoring to boost recent documents and penalize stale ones"
    ),
    SettingDefinition(
        key="rag.freshness_decay_days",
        category=SettingCategory.RAG,
        default_value=180,
        value_type=ValueType.NUMBER,
        description="Number of days after which documents are considered stale and penalized (30-730)"
    ),
    SettingDefinition(
        key="rag.freshness_boost_factor",
        category=SettingCategory.RAG,
        default_value=1.05,
        value_type=ValueType.NUMBER,
        description="Score multiplier for documents updated in the last 30 days (1.0-1.5)"
    ),
    SettingDefinition(
        key="rag.freshness_penalty_factor",
        category=SettingCategory.RAG,
        default_value=0.95,
        value_type=ValueType.NUMBER,
        description="Score multiplier for documents older than freshness_decay_days (0.5-1.0)"
    ),

    # ==========================================================================
    # Phase 97: AI Processing Settings (frontend ai.* namespace)
    # ==========================================================================

    # Text Preprocessing
    SettingDefinition(
        key="ai.enable_preprocessing",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable text preprocessing before embedding to reduce token costs (10-20% savings)"
    ),
    SettingDefinition(
        key="ai.remove_boilerplate",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Strip headers, footers, page numbers and other boilerplate content"
    ),
    SettingDefinition(
        key="ai.normalize_whitespace",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Collapse multiple spaces and newlines to normalize whitespace"
    ),
    SettingDefinition(
        key="ai.deduplicate_content",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Remove near-duplicate chunks during ingestion"
    ),

    # Document Summarization
    SettingDefinition(
        key="ai.enable_summarization",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Generate summaries for large documents before chunking (30-40% token savings)"
    ),
    SettingDefinition(
        key="ai.summarization_threshold_pages",
        category=SettingCategory.PROCESSING,
        default_value=50,
        value_type=ValueType.NUMBER,
        description="Page count threshold above which documents are summarized before chunking"
    ),
    SettingDefinition(
        key="ai.summarization_threshold_kb",
        category=SettingCategory.PROCESSING,
        default_value=100,
        value_type=ValueType.NUMBER,
        description="Size threshold (KB) above which documents are summarized before chunking"
    ),
    SettingDefinition(
        key="ai.summarization_model",
        category=SettingCategory.PROCESSING,
        default_value="gpt-4o-mini",
        value_type=ValueType.STRING,
        description="Model to use for document summarization (gpt-4o-mini, gpt-4o, claude-3-haiku)"
    ),

    # Adaptive and Hierarchical Chunking
    SettingDefinition(
        key="ai.enable_adaptive_chunking",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Auto-detect document type and adjust chunk size accordingly (10-15% token savings)"
    ),
    SettingDefinition(
        key="ai.enable_hierarchical_chunking",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Create multi-level chunk hierarchy (document/section/detail) for large docs (20-30% savings)"
    ),
    SettingDefinition(
        key="ai.hierarchical_threshold_chars",
        category=SettingCategory.PROCESSING,
        default_value=100000,
        value_type=ValueType.NUMBER,
        description="Character threshold above which hierarchical chunking is applied"
    ),
    SettingDefinition(
        key="ai.sections_per_document",
        category=SettingCategory.PROCESSING,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Target number of section summaries per document in hierarchical chunking (3-20)"
    ),

    # ==========================================================================
    # Phase 97: Semantic Cache & Query Expansion (rag.* namespace)
    # ==========================================================================

    SettingDefinition(
        key="rag.semantic_cache_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable semantic query caching to serve similar queries from cache (up to 68% API cost reduction)"
    ),
    SettingDefinition(
        key="rag.semantic_similarity_threshold",
        category=SettingCategory.RAG,
        default_value=0.95,
        value_type=ValueType.NUMBER,
        description="Minimum cosine similarity for a cache hit (0.8-1.0, higher = stricter matching)"
    ),
    SettingDefinition(
        key="rag.max_semantic_cache_entries",
        category=SettingCategory.RAG,
        default_value=10000,
        value_type=ValueType.NUMBER,
        description="Maximum number of entries in the semantic cache before eviction"
    ),
    SettingDefinition(
        key="rag.query_expansion_model",
        category=SettingCategory.RAG,
        default_value="gpt-4o-mini",
        value_type=ValueType.STRING,
        description="Model to use for generating query variations (gpt-4o-mini, gpt-4o, claude-3-haiku)"
    ),

    # ==========================================================================
    # Phase 97: Advanced Retrieval Methods (LightRAG, RAPTOR, SELF-RAG extras)
    # ==========================================================================

    SettingDefinition(
        key="rag.lightrag_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable LightRAG dual-level retrieval with entity-based and relationship-aware search (10x token reduction)"
    ),
    SettingDefinition(
        key="rag.lightrag_mode",
        category=SettingCategory.RAG,
        default_value="hybrid",
        value_type=ValueType.STRING,
        description="LightRAG retrieval mode: local (entity-focused), global (relationship-focused), hybrid (both)"
    ),
    SettingDefinition(
        key="rag.lightrag_max_entities",
        category=SettingCategory.RAG,
        default_value=20,
        value_type=ValueType.NUMBER,
        description="Maximum entities to consider per query in LightRAG (5-50)"
    ),
    SettingDefinition(
        key="rag.raptor_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable RAPTOR tree-organized retrieval for hierarchical document understanding (97% token reduction)"
    ),
    SettingDefinition(
        key="rag.raptor_tree_depth",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of hierarchy levels in RAPTOR document tree (2-6)"
    ),
    SettingDefinition(
        key="rag.raptor_cluster_size",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Number of chunks per cluster at each RAPTOR tree level (3-15)"
    ),
    SettingDefinition(
        key="rag.raptor_use_summaries",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Generate summary nodes at each RAPTOR tree level for multi-level understanding"
    ),
    SettingDefinition(
        key="rag.self_rag_confidence_threshold",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Minimum confidence score before SELF-RAG triggers re-retrieval (0.5-0.95)"
    ),
    SettingDefinition(
        key="rag.self_rag_max_retries",
        category=SettingCategory.RAG,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum number of re-retrieval attempts for low-confidence SELF-RAG answers (1-5)"
    ),

    # Jina Reranker v3 (Listwise, 131K context)
    SettingDefinition(
        key="rerank.jina_v3_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Jina reranker v3 (listwise scoring, 131K context, BEIR 61.94 nDCG@10)"
    ),

    # ==========================================================================
    # Phase 97: Tiered Reranking Pipeline
    # ==========================================================================

    SettingDefinition(
        key="rerank.tiered_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable 3-stage tiered reranking pipeline (ColBERT → Cross-Encoder → LLM) for 87% multi-hop accuracy"
    ),
    SettingDefinition(
        key="rerank.stage1_top_k",
        category=SettingCategory.RAG,
        default_value=100,
        value_type=ValueType.NUMBER,
        description="Number of candidates from ColBERT stage 1 (20-200)"
    ),
    SettingDefinition(
        key="rerank.stage2_top_k",
        category=SettingCategory.RAG,
        default_value=20,
        value_type=ValueType.NUMBER,
        description="Number of candidates from cross-encoder stage 2 (5-50)"
    ),
    SettingDefinition(
        key="rerank.final_top_k",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Final number of results returned after all reranking stages (3-20)"
    ),
    SettingDefinition(
        key="rerank.use_llm_stage",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable LLM-based stage 3 reranking for highest quality (increases cost and latency)"
    ),

    # ==========================================================================
    # Phase 97: Memory Settings
    # ==========================================================================

    SettingDefinition(
        key="memory.enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable persistent agent memory for context retention across conversations (91% lower latency)"
    ),
    SettingDefinition(
        key="memory.provider",
        category=SettingCategory.RAG,
        default_value="mem0",
        value_type=ValueType.STRING,
        description="Memory backend provider: mem0 (recommended) or amem (agentic)"
    ),
    SettingDefinition(
        key="memory.max_entries",
        category=SettingCategory.RAG,
        default_value=1000,
        value_type=ValueType.NUMBER,
        description="Maximum memory entries per user (100-10000)"
    ),
    SettingDefinition(
        key="memory.decay_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Automatically deprioritize old memories based on access frequency and age"
    ),

    # ==========================================================================
    # Phase 97: Context Compression Settings
    # ==========================================================================

    SettingDefinition(
        key="compression.enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable context compression to reduce token costs for long conversations (32x compression, 85% savings)"
    ),
    SettingDefinition(
        key="compression.recent_turns",
        category=SettingCategory.RAG,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Number of recent conversation turns to keep uncompressed (3-20)"
    ),
    SettingDefinition(
        key="compression.level",
        category=SettingCategory.RAG,
        default_value="moderate",
        value_type=ValueType.STRING,
        description="Compression aggressiveness: minimal (keep more context), moderate (balanced), aggressive (max savings)"
    ),
    SettingDefinition(
        key="compression.enable_anchoring",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Never compress critical information like names, dates, and key facts"
    ),

    # ==========================================================================
    # Conversation Memory Settings (Dynamic window, rehydration, query rewriting)
    # ==========================================================================

    SettingDefinition(
        key="conversation.db_rehydration_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Restore conversation history from DB after backend restart. Ensures follow-up questions work across restarts."
    ),
    SettingDefinition(
        key="conversation.query_rewriting_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Rewrite follow-up questions into standalone queries using LLM. Critical for conversational RAG accuracy (+30-45% precision)."
    ),
    SettingDefinition(
        key="conversation.dynamic_memory_window",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Automatically size conversation memory window based on model size (3 turns for tiny, 6 for small, 10 for medium, 15 for large)."
    ),
    SettingDefinition(
        key="conversation.stuff_then_refine_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="When context exceeds model capacity, use iterative stuff-then-refine strategy instead of truncation. Best for small LLMs."
    ),
    SettingDefinition(
        key="conversation.token_budget_enforcement",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enforce token budgets for system prompt, history, chunks, and generation buffer. Prevents context window overflow."
    ),

    # ==========================================================================
    # System Settings (Custom Instructions, Language, etc.)
    # ==========================================================================
    SettingDefinition(
        key="system.custom_instructions_enabled",
        category=SettingCategory.SYSTEM,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable custom system instructions for RAG responses"
    ),
    SettingDefinition(
        key="system.org_system_prompt",
        category=SettingCategory.SYSTEM,
        default_value="",
        value_type=ValueType.STRING,
        description="Organization-level custom system prompt appended to RAG prompts"
    ),
    SettingDefinition(
        key="system.default_response_language",
        category=SettingCategory.SYSTEM,
        default_value="auto",
        value_type=ValueType.STRING,
        description="Default language for RAG responses (auto, en, es, fr, de, ja, zh, ko, pt, ar, hi)"
    ),
    SettingDefinition(
        key="system.custom_instructions_append_mode",
        category=SettingCategory.SYSTEM,
        default_value="append",
        value_type=ValueType.STRING,
        description="How custom instructions are combined with system prompt (prepend, append, replace)"
    ),

    # ==========================================================================
    # Generation Settings (Vision Analysis, Slide Constraints)
    # ==========================================================================
    SettingDefinition(
        key="generation.enable_template_vision_analysis",
        category=SettingCategory.GENERATION,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable vision model analysis of uploaded templates for layout matching"
    ),
    SettingDefinition(
        key="generation.template_vision_model",
        category=SettingCategory.GENERATION,
        default_value="gpt-4o",
        value_type=ValueType.STRING,
        description="Vision model to use for template analysis (gpt-4o, claude-3-5-sonnet)"
    ),
    SettingDefinition(
        key="generation.enable_per_slide_constraints",
        category=SettingCategory.GENERATION,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable per-slide content constraints for presentation generation"
    ),

    # ==========================================================================
    # Observability / Distributed Tracing (OpenTelemetry)
    # ==========================================================================
    SettingDefinition(
        key="observability.tracing_enabled",
        category=SettingCategory.GENERAL,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable OpenTelemetry distributed tracing for the RAG pipeline"
    ),
    SettingDefinition(
        key="observability.tracing_sample_rate",
        category=SettingCategory.GENERAL,
        default_value=0.1,
        value_type=ValueType.NUMBER,
        description="Head-based sampling rate for traces (0.0 = none, 1.0 = all requests)"
    ),
    SettingDefinition(
        key="observability.otlp_endpoint",
        category=SettingCategory.GENERAL,
        default_value="",
        value_type=ValueType.STRING,
        description="OTLP/gRPC collector endpoint for exporting traces (e.g. http://localhost:4317)"
    ),

    # ==========================================================================
    # Memory Management Settings (Phase 2.2)
    # ==========================================================================
    SettingDefinition(
        key="system.memory_cleanup_interval_minutes",
        category=SettingCategory.SYSTEM,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Run memory cleanup task every N minutes (5-60)"
    ),
    SettingDefinition(
        key="system.model_idle_timeout_minutes",
        category=SettingCategory.SYSTEM,
        default_value=15,
        value_type=ValueType.NUMBER,
        description="Unload ML models after N minutes of inactivity (5-60)"
    ),
    SettingDefinition(
        key="system.max_memory_usage_percent",
        category=SettingCategory.SYSTEM,
        default_value=85,
        value_type=ValueType.NUMBER,
        description="Trigger aggressive cleanup when memory usage exceeds this percentage (50-95)"
    ),

    # ==========================================================================
    # OCR Settings (Phase 2.2)
    # ==========================================================================
    SettingDefinition(
        key="ocr.timeout_seconds",
        category=SettingCategory.OCR,
        default_value=30,
        value_type=ValueType.NUMBER,
        description="OCR timeout per page in seconds (10-120)"
    ),
    SettingDefinition(
        key="ocr.max_reload_uses",
        category=SettingCategory.OCR,
        default_value=50,
        value_type=ValueType.NUMBER,
        description="Reload OCR model after N uses to prevent memory bloat (10-200)"
    ),
    SettingDefinition(
        key="ocr.force_cpu",
        category=SettingCategory.OCR,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Force CPU mode for OCR (recommended for Mac stability)"
    ),

    # ==========================================================================
    # Hybrid Search Settings (Phase 2.2 - OpenClaw-inspired)
    # ==========================================================================
    SettingDefinition(
        key="search.hybrid_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable hybrid search combining vector similarity and keyword matching"
    ),
    SettingDefinition(
        key="search.vector_weight",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Weight for vector (semantic) similarity in hybrid search (0.0-1.0)"
    ),
    SettingDefinition(
        key="search.bm25_weight",
        category=SettingCategory.RAG,
        default_value=0.3,
        value_type=ValueType.NUMBER,
        description="Weight for BM25 (keyword) matching in hybrid search (0.0-1.0)"
    ),
    SettingDefinition(
        key="search.rrf_k",
        category=SettingCategory.RAG,
        default_value=60,
        value_type=ValueType.NUMBER,
        description="Reciprocal Rank Fusion K parameter (40-100, higher = less penalty for low ranks)"
    ),

    # ==========================================================================
    # Query Cache Settings (Phase 2.2 - LightRAG-inspired)
    # ==========================================================================
    SettingDefinition(
        key="cache.query_cache_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable Q&A query caching to reduce redundant LLM calls"
    ),
    SettingDefinition(
        key="cache.query_similarity_threshold",
        category=SettingCategory.RAG,
        default_value=0.95,
        value_type=ValueType.NUMBER,
        description="Cosine similarity threshold for query cache hits (0.90-0.99)"
    ),
    SettingDefinition(
        key="cache.query_cache_ttl_hours",
        category=SettingCategory.RAG,
        default_value=24,
        value_type=ValueType.NUMBER,
        description="Query cache time-to-live in hours (1-168)"
    ),
    SettingDefinition(
        key="cache.query_cache_max_entries",
        category=SettingCategory.RAG,
        default_value=10000,
        value_type=ValueType.NUMBER,
        description="Maximum cached query-answer pairs (1000-100000)"
    ),

    # ==========================================================================
    # Ray / Distributed Processing Settings (Phase 2.2)
    # ==========================================================================
    SettingDefinition(
        key="processing.use_ray",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Use Ray for distributed processing (requires Ray installation)"
    ),
    SettingDefinition(
        key="processing.ray_num_workers",
        category=SettingCategory.PROCESSING,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Number of Ray workers for parallel processing (1-16)"
    ),
    SettingDefinition(
        key="processing.ray_fallback_to_local",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Fallback to local ThreadPoolExecutor if Ray unavailable"
    ),
    SettingDefinition(
        key="processing.max_concurrent_embeddings",
        category=SettingCategory.PROCESSING,
        default_value=4,
        value_type=ValueType.NUMBER,
        description="Max concurrent embedding batches (1-16)"
    ),

    # ==========================================================================
    # VLM / Vision Settings (Phase 2.2)
    # ==========================================================================
    SettingDefinition(
        key="rag.vlm_provider",
        category=SettingCategory.RAG,
        default_value="auto",
        value_type=ValueType.STRING,
        description="Vision model provider: auto, ollama, openai, anthropic"
    ),
    SettingDefinition(
        key="rag.vlm_model",
        category=SettingCategory.RAG,
        default_value="",
        value_type=ValueType.STRING,
        description="Vision model name (empty = auto-detect). E.g., llava, gpt-4o, claude-3-haiku"
    ),
    SettingDefinition(
        key="rag.image_duplicate_detection",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Skip re-analyzing duplicate images (use cached captions)"
    ),
    SettingDefinition(
        key="rag.max_images_per_document",
        category=SettingCategory.RAG,
        default_value=50,
        value_type=ValueType.NUMBER,
        description="Maximum images to analyze per document (0 = unlimited)"
    ),
    SettingDefinition(
        key="rag.min_image_size_kb",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Minimum image size in KB to analyze (skip tiny icons)"
    ),

    # ==========================================================================
    # SMTP / Email Notification Settings (Phase 2.2)
    # ==========================================================================
    SettingDefinition(
        key="notifications.smtp_host",
        category=SettingCategory.NOTIFICATIONS,
        default_value="",
        value_type=ValueType.STRING,
        description="SMTP server hostname for email notifications"
    ),
    SettingDefinition(
        key="notifications.smtp_port",
        category=SettingCategory.NOTIFICATIONS,
        default_value=587,
        value_type=ValueType.NUMBER,
        description="SMTP server port (25, 465 for SSL, 587 for TLS)"
    ),
    SettingDefinition(
        key="notifications.smtp_user",
        category=SettingCategory.NOTIFICATIONS,
        default_value="",
        value_type=ValueType.STRING,
        description="SMTP authentication username"
    ),
    SettingDefinition(
        key="notifications.smtp_password",
        category=SettingCategory.NOTIFICATIONS,
        default_value="",
        value_type=ValueType.STRING,
        description="SMTP authentication password (stored encrypted)"
    ),
    SettingDefinition(
        key="notifications.smtp_from_email",
        category=SettingCategory.NOTIFICATIONS,
        default_value="",
        value_type=ValueType.STRING,
        description="Default sender email address for notifications"
    ),
    SettingDefinition(
        key="notifications.smtp_from_name",
        category=SettingCategory.NOTIFICATIONS,
        default_value="AIDocumentIndexer",
        value_type=ValueType.STRING,
        description="Default sender display name for email notifications"
    ),

    # ==========================================================================
    # Adaptive Chunking Settings (Phase 2.2 - Moltbot-inspired)
    # ==========================================================================
    SettingDefinition(
        key="rag.chunk_target_tokens",
        category=SettingCategory.RAG,
        default_value=400,
        value_type=ValueType.NUMBER,
        description="Target chunk size in tokens (200-800)"
    ),
    SettingDefinition(
        key="rag.chunk_overlap_tokens",
        category=SettingCategory.RAG,
        default_value=80,
        value_type=ValueType.NUMBER,
        description="Chunk overlap in tokens (20-200)"
    ),
    SettingDefinition(
        key="rag.adaptive_chunking_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable adaptive chunking with progressive fallback for oversized chunks"
    ),

    # ==========================================================================
    # Enhanced Metadata in RAG Context
    # ==========================================================================
    SettingDefinition(
        key="rag.include_enhanced_metadata",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Include document summaries, keywords, and topics in RAG context for richer LLM responses"
    ),
    SettingDefinition(
        key="rag.tiny_model_min_score",
        category=SettingCategory.RAG,
        default_value=0.25,
        value_type=ValueType.NUMBER,
        description="Minimum retrieval score for tiny models (<3B). Below this, the LLM is skipped to prevent hallucination on irrelevant context."
    ),

    # ==========================================================================
    # Auto-Enhancement on Upload
    # ==========================================================================
    SettingDefinition(
        key="upload.auto_enhance",
        category=SettingCategory.INGESTION,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Automatically enhance documents after upload (generates summaries, keywords, and hypothetical questions)"
    ),

    # ==========================================================================
    # Connector Storage Settings
    # ==========================================================================
    SettingDefinition(
        key="connector.storage_mode",
        category=SettingCategory.INGESTION,
        default_value="download",
        value_type=ValueType.STRING,
        description="How to handle files from external connectors: 'download' = download and store locally, 'process_only' = process for indexing but don't store the file (uses external link for preview)"
    ),
    SettingDefinition(
        key="connector.store_source_metadata",
        category=SettingCategory.INGESTION,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Store source metadata (original path, connector type, external URL) for all documents"
    ),

    # ==========================================================================
    # Provider Failover Settings (Phase 2.2 - OpenClaw-inspired)
    # ==========================================================================
    SettingDefinition(
        key="llm.enable_provider_failover",
        category=SettingCategory.LLM,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Automatically failover to backup providers on error"
    ),
    SettingDefinition(
        key="llm.failover_providers",
        category=SettingCategory.LLM,
        default_value="openai,anthropic,ollama",
        value_type=ValueType.STRING,
        description="Comma-separated list of fallback LLM providers in priority order"
    ),
    SettingDefinition(
        key="embedding.enable_provider_failover",
        category=SettingCategory.LLM,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Automatically failover to backup embedding providers on error"
    ),
    SettingDefinition(
        key="embedding.failover_providers",
        category=SettingCategory.LLM,
        default_value="openai,voyage,ollama",
        value_type=ValueType.STRING,
        description="Comma-separated list of fallback embedding providers in priority order"
    ),

    # ==========================================================================
    # Phase 98: Tag Integration Settings
    # ==========================================================================
    SettingDefinition(
        key="tags.reembed_on_change",
        category=SettingCategory.PROCESSING,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Re-embed chunks when document tags change. Improves retrieval quality by updating tag context in embeddings, but costs processing time. Recommended for smaller document collections."
    ),
    SettingDefinition(
        key="tags.sync_to_kg",
        category=SettingCategory.PROCESSING,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Sync document tags to Knowledge Graph entities when tags change. Enables tag-based entity filtering in KG queries."
    ),

    # ==========================================================================
    # Infrastructure / Scaling Settings
    # ==========================================================================
    SettingDefinition(
        key="vector_store.backend",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="pgvector",
        value_type=ValueType.STRING,
        description="Vector store backend: pgvector (default, included), qdrant (1-50M scale), milvus (50M+ scale)"
    ),
    SettingDefinition(
        key="vector_store.qdrant_url",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="localhost:6333",
        value_type=ValueType.STRING,
        description="Qdrant server URL (host:port). Used when vector_store.backend is qdrant."
    ),
    SettingDefinition(
        key="vector_store.qdrant_api_key",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="",
        value_type=ValueType.STRING,
        description="Qdrant Cloud API key. Leave empty for local/self-hosted Qdrant."
    ),
    SettingDefinition(
        key="vector_store.qdrant_collection",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="documents",
        value_type=ValueType.STRING,
        description="Qdrant collection name for document embeddings."
    ),
    SettingDefinition(
        key="vector_store.milvus_host",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="localhost",
        value_type=ValueType.STRING,
        description="Milvus server hostname. Used when vector_store.backend is milvus."
    ),
    SettingDefinition(
        key="vector_store.milvus_port",
        category=SettingCategory.INFRASTRUCTURE,
        default_value=19530,
        value_type=ValueType.NUMBER,
        description="Milvus server port (default: 19530)."
    ),
    SettingDefinition(
        key="vector_store.milvus_collection",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="documents",
        value_type=ValueType.STRING,
        description="Milvus collection name for document embeddings."
    ),
    SettingDefinition(
        key="vector_store.milvus_user",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="",
        value_type=ValueType.STRING,
        description="Milvus username (optional, for authenticated clusters)."
    ),
    SettingDefinition(
        key="vector_store.milvus_password",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="",
        value_type=ValueType.STRING,
        description="Milvus password (optional, for authenticated clusters)."
    ),
    SettingDefinition(
        key="llm.inference_backend",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="ollama",
        value_type=ValueType.STRING,
        description="LLM inference backend: auto (use default provider from Providers tab), ollama (default), vllm (2-4x faster batch inference), openai, anthropic, groq, together, deepinfra, bedrock, google, cohere, custom"
    ),
    SettingDefinition(
        key="llm.vllm_api_base",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="http://localhost:8000/v1",
        value_type=ValueType.STRING,
        description="vLLM OpenAI-compatible API base URL."
    ),
    SettingDefinition(
        key="llm.vllm_api_key",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="dummy",
        value_type=ValueType.STRING,
        description="vLLM API key (typically 'dummy' for local deployment)."
    ),
    SettingDefinition(
        key="llm.vllm_model",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="",
        value_type=ValueType.STRING,
        description="Model name loaded on the vLLM server (e.g. meta-llama/Meta-Llama-3-8B-Instruct)."
    ),
    SettingDefinition(
        key="infrastructure.redis_url",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="redis://localhost:6379/0",
        value_type=ValueType.STRING,
        description="Redis connection URL for caching and task queue."
    ),
    SettingDefinition(
        key="infrastructure.redis_password",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="",
        value_type=ValueType.STRING,
        description="Redis password (optional)."
    ),
    SettingDefinition(
        key="infrastructure.scaling_profile",
        category=SettingCategory.INFRASTRUCTURE,
        default_value="development",
        value_type=ValueType.STRING,
        description="Active scaling profile: development (pgvector+ollama), production (qdrant+vllm+redis), high_scale (milvus+vllm+redis)"
    ),
]

# Build lookup dictionary
SETTINGS_DEFINITIONS: Dict[str, SettingDefinition] = {
    sd.key: sd for sd in DEFAULT_SETTINGS
}


# =============================================================================
# Settings Service
# =============================================================================

class SettingsService:
    """
    Service for managing system settings.

    Features:
    - Get/set individual settings
    - Get all settings by category
    - Type conversion and validation
    - Default values for unset settings
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_loaded = False

    def _convert_value(self, value: str, value_type: ValueType) -> Any:
        """Convert string value to appropriate type."""
        if value is None:
            return None

        try:
            if value_type == ValueType.NUMBER:
                # Try int first, then float
                try:
                    return int(value)
                except ValueError:
                    return float(value)
            elif value_type == ValueType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "1", "yes", "on")
            elif value_type == ValueType.JSON:
                return json.loads(value)
            else:
                return value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("Failed to convert setting value", error=str(e))
            return value

    def _serialize_value(self, value: Any, value_type: ValueType) -> str:
        """Serialize value to string for storage."""
        if value is None:
            return ""

        if value_type == ValueType.BOOLEAN:
            return "true" if value else "false"
        elif value_type == ValueType.JSON:
            return json.dumps(value)
        else:
            return str(value)

    async def get_setting(
        self,
        key: str,
        session: Optional[AsyncSession] = None,
    ) -> Any:
        """
        Get a single setting value.

        Returns the stored value or default if not set.
        """
        definition = SETTINGS_DEFINITIONS.get(key)
        default_value = definition.default_value if definition else None
        value_type = definition.value_type if definition else ValueType.STRING

        async def _get(db: AsyncSession) -> Any:
            query = select(SystemSettings).where(SystemSettings.key == key)
            result = await db.execute(query)
            setting = result.scalar_one_or_none()

            if setting is None:
                return default_value

            return self._convert_value(setting.value, value_type)

        if session:
            return await _get(session)

        async with async_session_context() as db:
            return await _get(db)

    async def get_settings_batch(
        self,
        keys: List[str],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Get multiple settings in a single database query.

        Returns a dict mapping each key to its value (or default if not set).
        Much more efficient than calling get_setting() for each key individually.
        """
        # Build defaults map
        defaults = {}
        type_map = {}
        for key in keys:
            definition = SETTINGS_DEFINITIONS.get(key)
            defaults[key] = definition.default_value if definition else None
            type_map[key] = definition.value_type if definition else ValueType.STRING

        async def _get_batch(db: AsyncSession) -> Dict[str, Any]:
            query = select(SystemSettings).where(SystemSettings.key.in_(keys))
            result = await db.execute(query)
            settings_rows = result.scalars().all()

            # Start with defaults, override with stored values
            values = dict(defaults)
            for setting in settings_rows:
                vtype = type_map.get(setting.key, ValueType.STRING)
                values[setting.key] = self._convert_value(setting.value, vtype)

            return values

        if session:
            return await _get_batch(session)

        async with async_session_context() as db:
            return await _get_batch(db)

    async def set_setting(
        self,
        key: str,
        value: Any,
        session: Optional[AsyncSession] = None,
    ) -> None:
        """
        Set a single setting value.

        Creates the setting if it doesn't exist.
        """
        definition = SETTINGS_DEFINITIONS.get(key)
        value_type = definition.value_type if definition else ValueType.STRING
        category = definition.category.value if definition else SettingCategory.GENERAL.value
        description = definition.description if definition else None

        serialized = self._serialize_value(value, value_type)

        async def _set(db: AsyncSession) -> None:
            query = select(SystemSettings).where(SystemSettings.key == key)
            result = await db.execute(query)
            setting = result.scalar_one_or_none()

            if setting:
                setting.value = serialized
            else:
                setting = SystemSettings(
                    key=key,
                    value=serialized,
                    category=category,
                    description=description,
                    value_type=value_type.value,
                )
                db.add(setting)

            await db.commit()

        if session:
            await _set(session)
        else:
            async with async_session_context() as db:
                await _set(db)

        # Invalidate cache
        self._cache_loaded = False

    async def get_all_settings(
        self,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Get all settings including defaults.

        Returns a dictionary with all settings, using defaults
        for any that haven't been explicitly set.
        """
        async def _get_all(db: AsyncSession) -> Dict[str, Any]:
            # Start with defaults
            result = {
                sd.key: sd.default_value
                for sd in DEFAULT_SETTINGS
            }

            # Get stored settings
            query = select(SystemSettings)
            db_result = await db.execute(query)
            stored = db_result.scalars().all()

            # Override with stored values
            for setting in stored:
                definition = SETTINGS_DEFINITIONS.get(setting.key)
                value_type = ValueType(setting.value_type) if setting.value_type else ValueType.STRING

                if definition:
                    value_type = definition.value_type

                result[setting.key] = self._convert_value(setting.value, value_type)

            return result

        if session:
            return await _get_all(session)

        async with async_session_context() as db:
            return await _get_all(db)

    async def get_settings_by_category(
        self,
        category: SettingCategory,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Get all settings for a specific category.
        """
        all_settings = await self.get_all_settings(session)

        return {
            key: value
            for key, value in all_settings.items()
            if key.startswith(f"{category.value}.")
        }

    async def update_settings(
        self,
        settings: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Update multiple settings at once.

        Returns the updated settings.
        """
        async def _update(db: AsyncSession) -> Dict[str, Any]:
            for key, value in settings.items():
                definition = SETTINGS_DEFINITIONS.get(key)
                value_type = definition.value_type if definition else ValueType.STRING
                category = definition.category.value if definition else SettingCategory.GENERAL.value
                description = definition.description if definition else None

                serialized = self._serialize_value(value, value_type)

                query = select(SystemSettings).where(SystemSettings.key == key)
                result = await db.execute(query)
                setting = result.scalar_one_or_none()

                if setting:
                    setting.value = serialized
                else:
                    setting = SystemSettings(
                        key=key,
                        value=serialized,
                        category=category,
                        description=description,
                        value_type=value_type.value,
                    )
                    db.add(setting)

            await db.commit()

            # Return all settings after update
            return await self.get_all_settings(db)

        if session:
            result = await _update(session)
        else:
            async with async_session_context() as db:
                result = await _update(db)

        # Invalidate cache
        self._cache_loaded = False

        return result

    async def reset_to_defaults(
        self,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Reset all settings to their default values.
        """
        async def _reset(db: AsyncSession) -> Dict[str, Any]:
            # Delete all stored settings
            query = select(SystemSettings)
            result = await db.execute(query)
            for setting in result.scalars().all():
                await db.delete(setting)

            await db.commit()

            # Return defaults
            return {
                sd.key: sd.default_value
                for sd in DEFAULT_SETTINGS
            }

        if session:
            return await _reset(session)

        async with async_session_context() as db:
            return await _reset(db)

    def get_default_value(self, key: str) -> Any:
        """
        Get the default value for a setting (synchronous).

        This doesn't read from the database - just returns the hardcoded default.
        Useful for initialization when async DB access isn't available.
        """
        definition = SETTINGS_DEFINITIONS.get(key)
        return definition.default_value if definition else None

    def get_setting_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all setting definitions with metadata.

        Useful for building settings UI.
        """
        return [
            {
                "key": sd.key,
                "category": sd.category.value,
                "default_value": sd.default_value,
                "value_type": sd.value_type.value,
                "description": sd.description,
            }
            for sd in DEFAULT_SETTINGS
        ]


# =============================================================================
# Singleton Instance
# =============================================================================

_settings_service: Optional[SettingsService] = None


def get_settings_service() -> SettingsService:
    """Get the settings service singleton instance."""
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service
