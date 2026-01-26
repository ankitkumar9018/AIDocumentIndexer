"""
AIDocumentIndexer - Services Module
====================================

Core services for LLM, embeddings, RAG, and document processing.
"""

from backend.services.llm import (
    get_chat_model,
    get_embeddings,
    generate_response,
    generate_embeddings,
    generate_embedding,
    test_llm_connection,
    test_embeddings_connection,
    LLMFactory,
    LLMConfig,
)

from backend.services.embeddings import (
    EmbeddingService,
    RayEmbeddingService,
    EmbeddingResult,
    get_embedding_service,
    compute_similarity,
)

from backend.services.rag import (
    RAGService,
    RAGConfig,
    RAGResponse,
    Source,
    StreamChunk,
    get_rag_service,
    simple_query,
)

from backend.services.vectorstore import (
    VectorStore,
    VectorStoreConfig,
    SearchResult,
    SearchType,
    get_vector_store,
)

from backend.services.pipeline import (
    DocumentPipeline,
    PipelineConfig,
    ProcessingResult,
    ProcessingStatus,
    BatchProcessingResult,
    get_pipeline,
    process_file,
)

from backend.services.permissions import (
    PermissionService,
    UserContext,
    Permission,
    get_permission_service,
    check_access,
    create_user_context_from_token,
    require_tier,
    require_admin,
)

from backend.services.audit import (
    AuditService,
    AuditAction,
    AuditEntry,
    LogSeverity,
    get_audit_service,
    audit_log,
    audit_service_fallback,
    audit_service_error,
)

from backend.services.generator import (
    DocumentGenerationService,
    GenerationJob,
    GenerationStatus,
    OutputFormat,
    DocumentOutline,
    Section,
    SourceReference,
    get_generation_service,
)

from backend.services.collaboration import (
    CollaborationService,
    CollaborationSession,
    CollaborationConfig,
    CollaborationMode,
    CollaborationStatus,
    ModelConfig,
    get_collaboration_service,
)

from backend.services.watcher import (
    FileWatcherService,
    WatchedDirectory,
    FileEvent,
    WatchEventType,
    WatcherStatus,
    get_watcher_service,
)

from backend.services.realtime_indexer import (
    RealTimeIndexerService,
    FreshnessLevel,
    FreshnessInfo,
    IndexingStats,
    ContentChange,
    get_realtime_indexer_service,
)

from backend.services.scraper import (
    WebScraperService,
    ScrapeJob,
    ScrapedPage,
    ScrapeConfig,
    ScrapeStatus,
    StorageMode,
    get_scraper_service,
)

from backend.services.cost_tracking import (
    CostTrackingService,
    CostPeriod,
    UsageType,
    UsageRecord,
    CostSummary,
    UserCostSummary,
    CostAlert,
    get_cost_service,
    track_usage,
)

from backend.services.chart_generator import (
    ChartGenerator,
    ChartType,
    ChartData,
    ChartStyle,
    GeneratedChart,
    DataExtractor,
    get_chart_generator,
)

from backend.services.query_decomposer import (
    QueryDecomposer,
    QueryDecomposerConfig,
    DecomposedQuery,
    get_query_decomposer,
)

# Phase 51: Service Integrations
from backend.services.recursive_lm import (
    RecursiveLMService,
    RLMConfig,
    RLMResult,
)

from backend.services.distributed_processor import (
    DistributedProcessor,
    get_distributed_processor,
)

from backend.services.vlm_processor import (
    VLMProcessor,
    VLMResult,
    get_vlm_processor,
)

# Phase 38: Context Compression
from backend.services.context_compression import (
    ContextCompressionService,
    CompressionConfig,
    CompressionResult,
    get_context_compressor,
)

# Phase 40: Mem0 Memory
from backend.services.mem0_memory import (
    Mem0MemoryService,
    MemoryConfig,
    Memory,
    MemoryType,
    get_memory_service,
)

# Phase 41: GraphRAG Enhancements
from backend.services.graphrag_enhancements import (
    GraphRAGEnhancer,
    GraphRAGConfig,
    Community,
    get_graphrag_enhancer,
)

# Phase 42: GenerativeCache
from backend.services.generative_cache import (
    GenerativeCache,
    CacheConfig,
    CacheResult,
    get_generative_cache,
)

# Phase 43: Tiered Reranking
from backend.services.tiered_reranking import (
    TieredReranker,
    RerankerConfig,
    RerankResult,
    get_tiered_reranker,
)

# Phase 47: TTT Compression
from backend.services.ttt_compression import (
    TTTCompressionService,
    TTTConfig,
    CompressionMode,
    get_ttt_compressor,
)

# Phase 48: A-Mem Agentic Memory
from backend.services.amem_memory import (
    AMemMemoryService,
    AMemConfig,
    MemoryOperation,
    MemoryImportance,
    get_amem_service,
)

# Template Service - Agent and Workflow Templates
from backend.services.template_service import (
    TemplateService,
    TemplateInfo,
    TemplateType,
    AgentType,
    AgentTemplate,
    WorkflowTemplate,
    get_template_service,
)

# Phase 60: Agent Knowledge Service
from backend.services.agent_knowledge import (
    AgentKnowledgeService,
    IncrementalUpdatePipeline,
    KnowledgeBaseStats,
    DocumentVersion,
    DocumentChange,
    DocumentChanges,
    ChangeType,
    get_agent_knowledge_service,
)

# Phase 60: Knowledge Graph Convenience Functions
from backend.services.knowledge_graph import (
    KnowledgeGraphService,
    get_knowledge_graph_service,
    get_kg_service,
    extract_knowledge_graph,
    ExtractedEntity,
    ExtractedRelation,
    GraphSearchResult,
    GraphRAGContext,
)

# Phase 62: Tree of Thoughts
from backend.services.tree_of_thoughts import (
    TreeOfThoughts,
    ToTConfig,
    ToTResult,
)

# Phase 62: Answer Refiner
from backend.services.answer_refiner import (
    AnswerRefiner,
    RefinerConfig,
    RefinementResult,
)

# Phase 62: Vision Document Processor
from backend.services.vision_document_processor import (
    VisionDocumentProcessor,
    VisionConfig,
    VisionResult,
)

# Phase 62: Advanced Reranker
from backend.services.advanced_reranker import (
    MultiStageReranker,
    RerankerConfig as AdvancedRerankerConfig,
)

# Phase 62: Contextual Embeddings
from backend.services.contextual_embeddings import (
    ContextualEmbeddingService,
    ContextualChunk,
)

# Phase 63: Sufficiency Checker
from backend.services.sufficiency_checker import (
    SufficiencyChecker,
    SufficiencyLevel,
    SufficiencyResult,
)

# Phase 63: Fast Chunking
from backend.services.chunking import (
    FastChunker,
    FastChunk,
    FastChunkingStrategy,
)

# Phase 63: Document Parser (Docling)
from backend.services.document_parser import (
    DocumentParser,
    ParsedDocument,
)

# Phase 63: Agent Evaluation
from backend.services.agent_evaluation import (
    AgentEvaluator,
    EvaluationResult,
    PersonalizationService,
)

# Phase 65: Binary Quantization (32x memory reduction)
from backend.services.binary_quantization import (
    BinaryQuantizer,
    BinaryQuantizationConfig,
    BinarySearchResult,
    get_binary_quantizer,
    quantize_embeddings,
)

# Phase 65: GPU Acceleration (FAISS + cuVS)
from backend.services.gpu_acceleration import (
    GPUVectorSearch,
    GPUSearchConfig,
    GPUSearchResult,
    GPUBackend,
    IndexType,
    get_gpu_vector_search,
    check_gpu_availability,
)

# Phase 65: Learning-to-Rank
from backend.services.learning_to_rank import (
    LTRRanker,
    LTRConfig,
    LTRFeatureExtractor,
    ClickFeedback,
    RankedResult,
    get_ltr_ranker,
)

# Phase 65: Spell Correction
from backend.services.spell_correction import (
    SpellCorrector,
    SpellCorrectionConfig,
    CorrectionResult,
    BKTree,
    get_spell_corrector,
    correct_query,
)

# Phase 65: Semantic Query Cache
from backend.services.semantic_cache import (
    SemanticQueryCache,
    SemanticCacheConfig,
    CacheEntry,
    CacheStats,
    get_semantic_cache,
    cache_get,
    cache_set,
)

# Phase 65: Streaming Citations
from backend.services.streaming_citations import (
    StreamingCitationMatcher,
    CitationConfig,
    CitationStyle,
    Citation,
    EnrichedToken,
    StreamingCitationResult,
    get_citation_matcher,
    stream_with_citations,
    enrich_streaming_response,
)

# Phase 65: Web Crawler
from backend.services.web_crawler import (
    EnhancedWebCrawler,
    WebCrawlerConfig,
    CrawlResult,
    get_web_crawler,
)

# Phase 66: Adaptive Router
from backend.services.adaptive_router import (
    AdaptiveRouter,
    RoutingDecision,
    RetrievalStrategy,
    QueryComplexity,
    get_adaptive_router,
    route_query,
)

# Phase 66: Advanced RAG Utilities
from backend.services.advanced_rag_utils import (
    RAGFusion,
    ContextCompressor,
    ContextReorderer,
    StepBackPrompter,
    FusionResult,
    CompressedContext,
    StepBackResult,
    apply_rag_fusion,
    compress_context,
    reorder_for_attention,
    apply_stepback_prompting,
)

# Phase 66: RAG Evaluation
from backend.services.rag_evaluation import (
    RAGEvaluator,
    RAGASMetrics,
    EvaluationResult as RAGEvaluationResult,
    evaluate_rag_response,
    # DeepEval integration (2025)
    DeepEvalMetric,
    DeepEvalResult,
    DeepEvalEvaluator,
    evaluate_with_deepeval,
    format_deepeval_report,
)

# Phase 66: LazyGraphRAG (99% cost reduction)
from backend.services.lazy_graphrag import (
    LazyGraphRAGService,
    LazyGraphRAGConfig,
    LazyGraphContext,
    Community,
    get_lazy_graphrag_service,
    lazy_graph_retrieve,
)

# Phase 66: User Personalization
from backend.services.user_personalization import (
    UserPersonalizationService,
    UserProfile,
    ResponsePreferences,
    get_personalization_service,
    personalize_prompt,
)

# Phase 66: Dependency-based KG Extraction
from backend.services.dependency_entity_extractor import (
    DependencyEntityExtractor,
    FastEntity,
    FastRelation,
    get_dependency_extractor,
)

# Phase 66: Query Expansion with Multi-Query Rewriting
from backend.services.query_expander import (
    QueryExpander,
    QueryExpansionConfig,
    ExpandedQuery,
    MultiQueryResult,
    get_query_expander,
)

# Mode switching utility (local/cloud mode detection)
from backend.services.mode_switch import (
    is_local_mode,
    get_current_mode,
    get_mode_info,
    ModeInfo,
    MODE_LOCAL,
    MODE_CLOUD,
)

# Agentic RAG for complex multi-step queries
from backend.services.agentic_rag import (
    AgenticRAGService,
    AgenticRAGResult,
    AgentAction,
    SubQuery,
    ReActStep,
    get_agentic_rag_service,
)

# Collaborative Annotations Service
from backend.services.annotations import (
    AnnotationService,
    Annotation,
    AnnotationType,
    AnnotationStatus,
    AnnotationReply,
    AnnotationSummary,
    get_annotation_service,
    create_highlight,
    create_comment,
)

# Service Registry for centralized service management
from backend.services.registry import (
    ServiceRegistry,
    Services,
    get_registry,
    get_rag_service as registry_get_rag_service,
    get_embedding_service as registry_get_embedding_service,
    get_generator_service,
    get_summarizer_service,
    get_query_expander_service,
)

__all__ = [
    # LLM
    "get_chat_model",
    "get_embeddings",
    "generate_response",
    "generate_embeddings",
    "generate_embedding",
    "test_llm_connection",
    "test_embeddings_connection",
    "LLMFactory",
    "LLMConfig",
    # Embeddings
    "EmbeddingService",
    "RayEmbeddingService",
    "EmbeddingResult",
    "get_embedding_service",
    "compute_similarity",
    # RAG
    "RAGService",
    "RAGConfig",
    "RAGResponse",
    "Source",
    "StreamChunk",
    "get_rag_service",
    "simple_query",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "SearchResult",
    "SearchType",
    "get_vector_store",
    # Pipeline
    "DocumentPipeline",
    "PipelineConfig",
    "ProcessingResult",
    "ProcessingStatus",
    "BatchProcessingResult",
    "get_pipeline",
    "process_file",
    # Permissions
    "PermissionService",
    "UserContext",
    "Permission",
    "get_permission_service",
    "check_access",
    "create_user_context_from_token",
    "require_tier",
    "require_admin",
    # Audit
    "AuditService",
    "AuditAction",
    "AuditEntry",
    "get_audit_service",
    "audit_log",
    # Document Generation
    "DocumentGenerationService",
    "GenerationJob",
    "GenerationStatus",
    "OutputFormat",
    "DocumentOutline",
    "Section",
    "SourceReference",
    "get_generation_service",
    # Collaboration
    "CollaborationService",
    "CollaborationSession",
    "CollaborationConfig",
    "CollaborationMode",
    "CollaborationStatus",
    "ModelConfig",
    "get_collaboration_service",
    # File Watcher
    "FileWatcherService",
    "WatchedDirectory",
    "FileEvent",
    "WatchEventType",
    "WatcherStatus",
    "get_watcher_service",
    # Real-Time Indexer
    "RealTimeIndexerService",
    "FreshnessLevel",
    "FreshnessInfo",
    "IndexingStats",
    "ContentChange",
    "get_realtime_indexer_service",
    # Web Scraper
    "WebScraperService",
    "ScrapeJob",
    "ScrapedPage",
    "ScrapeConfig",
    "ScrapeStatus",
    "StorageMode",
    "get_scraper_service",
    # Cost Tracking
    "CostTrackingService",
    "CostPeriod",
    "UsageType",
    "UsageRecord",
    "CostSummary",
    "UserCostSummary",
    "CostAlert",
    "get_cost_service",
    "track_usage",
    # Chart Generator
    "ChartGenerator",
    "ChartType",
    "ChartData",
    "ChartStyle",
    "GeneratedChart",
    "DataExtractor",
    "get_chart_generator",
    # Query Decomposer
    "QueryDecomposer",
    "QueryDecomposerConfig",
    "DecomposedQuery",
    "get_query_decomposer",
    # Phase 51: RLM Service
    "RecursiveLMService",
    "RLMConfig",
    "RLMResult",
    # Phase 51: Distributed Processor
    "DistributedProcessor",
    "get_distributed_processor",
    # Phase 51: VLM Processor
    "VLMProcessor",
    "VLMResult",
    "get_vlm_processor",
    # Phase 38: Context Compression
    "ContextCompressionService",
    "CompressionConfig",
    "CompressionResult",
    "get_context_compressor",
    # Phase 40: Mem0 Memory
    "Mem0MemoryService",
    "MemoryConfig",
    "Memory",
    "MemoryType",
    "get_memory_service",
    # Phase 41: GraphRAG Enhancements
    "GraphRAGEnhancer",
    "GraphRAGConfig",
    "Community",
    "get_graphrag_enhancer",
    # Phase 42: GenerativeCache
    "GenerativeCache",
    "CacheConfig",
    "CacheResult",
    "get_generative_cache",
    # Phase 43: Tiered Reranking
    "TieredReranker",
    "RerankerConfig",
    "RerankResult",
    "get_tiered_reranker",
    # Phase 47: TTT Compression
    "TTTCompressionService",
    "TTTConfig",
    "CompressionMode",
    "get_ttt_compressor",
    # Phase 48: A-Mem Agentic Memory
    "AMemMemoryService",
    "AMemConfig",
    "MemoryOperation",
    "MemoryImportance",
    "get_amem_service",
    # Template Service
    "TemplateService",
    "TemplateInfo",
    "TemplateType",
    "AgentType",
    "AgentTemplate",
    "WorkflowTemplate",
    "get_template_service",
    # Phase 60: Agent Knowledge Service
    "AgentKnowledgeService",
    "IncrementalUpdatePipeline",
    "KnowledgeBaseStats",
    "DocumentVersion",
    "DocumentChange",
    "DocumentChanges",
    "ChangeType",
    "get_agent_knowledge_service",
    # Phase 60: Knowledge Graph Functions
    "KnowledgeGraphService",
    "get_knowledge_graph_service",
    "get_kg_service",
    "extract_knowledge_graph",
    "ExtractedEntity",
    "ExtractedRelation",
    "GraphSearchResult",
    "GraphRAGContext",
    # Phase 62: Tree of Thoughts
    "TreeOfThoughts",
    "ToTConfig",
    "ToTResult",
    # Phase 62: Answer Refiner
    "AnswerRefiner",
    "RefinerConfig",
    "RefinementResult",
    # Phase 62: Vision Document Processor
    "VisionDocumentProcessor",
    "VisionConfig",
    "VisionResult",
    # Phase 62: Advanced Reranker
    "MultiStageReranker",
    "AdvancedRerankerConfig",
    # Phase 62: Contextual Embeddings
    "ContextualEmbeddingService",
    "ContextualChunk",
    # Phase 63: Sufficiency Checker
    "SufficiencyChecker",
    "SufficiencyLevel",
    "SufficiencyResult",
    # Phase 63: Fast Chunking
    "FastChunker",
    "FastChunk",
    "FastChunkingStrategy",
    # Phase 63: Document Parser
    "DocumentParser",
    "ParsedDocument",
    # Phase 63: Agent Evaluation
    "AgentEvaluator",
    "EvaluationResult",
    "PersonalizationService",
    # Phase 65: Binary Quantization
    "BinaryQuantizer",
    "BinaryQuantizationConfig",
    "BinarySearchResult",
    "get_binary_quantizer",
    "quantize_embeddings",
    # Phase 65: GPU Acceleration
    "GPUVectorSearch",
    "GPUSearchConfig",
    "GPUSearchResult",
    "GPUBackend",
    "IndexType",
    "get_gpu_vector_search",
    "check_gpu_availability",
    # Phase 65: Learning-to-Rank
    "LTRRanker",
    "LTRConfig",
    "LTRFeatureExtractor",
    "ClickFeedback",
    "RankedResult",
    "get_ltr_ranker",
    # Phase 65: Spell Correction
    "SpellCorrector",
    "SpellCorrectionConfig",
    "CorrectionResult",
    "BKTree",
    "get_spell_corrector",
    "correct_query",
    # Phase 65: Semantic Cache
    "SemanticQueryCache",
    "SemanticCacheConfig",
    "CacheEntry",
    "CacheStats",
    "get_semantic_cache",
    "cache_get",
    "cache_set",
    # Phase 65: Streaming Citations
    "StreamingCitationMatcher",
    "CitationConfig",
    "CitationStyle",
    "Citation",
    "EnrichedToken",
    "StreamingCitationResult",
    "get_citation_matcher",
    "stream_with_citations",
    "enrich_streaming_response",
    # Phase 65: Web Crawler
    "EnhancedWebCrawler",
    "WebCrawlerConfig",
    "CrawlResult",
    "get_web_crawler",
    # Phase 66: Adaptive Router
    "AdaptiveRouter",
    "RoutingDecision",
    "RetrievalStrategy",
    "QueryComplexity",
    "get_adaptive_router",
    "route_query",
    # Phase 66: Advanced RAG Utilities
    "RAGFusion",
    "ContextCompressor",
    "ContextReorderer",
    "StepBackPrompter",
    "FusionResult",
    "CompressedContext",
    "StepBackResult",
    "apply_rag_fusion",
    "compress_context",
    "reorder_for_attention",
    "apply_stepback_prompting",
    # Phase 66: RAG Evaluation
    "RAGEvaluator",
    "RAGASMetrics",
    "RAGEvaluationResult",
    "evaluate_rag_response",
    # DeepEval integration (2025)
    "DeepEvalMetric",
    "DeepEvalResult",
    "DeepEvalEvaluator",
    "evaluate_with_deepeval",
    "format_deepeval_report",
    # Phase 66: LazyGraphRAG
    "LazyGraphRAGService",
    "LazyGraphRAGConfig",
    "LazyGraphContext",
    "Community",
    "get_lazy_graphrag_service",
    "lazy_graph_retrieve",
    # Phase 66: User Personalization
    "UserPersonalizationService",
    "UserProfile",
    "ResponsePreferences",
    "get_personalization_service",
    "personalize_prompt",
    # Phase 66: Dependency-based KG Extraction
    "DependencyEntityExtractor",
    "FastEntity",
    "FastRelation",
    "get_dependency_extractor",
    # Phase 66: Query Expansion with Multi-Query Rewriting
    "QueryExpander",
    "QueryExpansionConfig",
    "ExpandedQuery",
    "MultiQueryResult",
    "get_query_expander",
    # Mode switching utility
    "is_local_mode",
    "get_current_mode",
    "get_mode_info",
    "ModeInfo",
    "MODE_LOCAL",
    "MODE_CLOUD",
    # Agentic RAG
    "AgenticRAGService",
    "AgenticRAGResult",
    "AgentAction",
    "SubQuery",
    "ReActStep",
    "get_agentic_rag_service",
    # Annotations
    "AnnotationService",
    "Annotation",
    "AnnotationType",
    "AnnotationStatus",
    "AnnotationReply",
    "AnnotationSummary",
    "get_annotation_service",
    "create_highlight",
    "create_comment",
    # Service Registry
    "ServiceRegistry",
    "Services",
    "get_registry",
    "registry_get_rag_service",
    "registry_get_embedding_service",
    "get_generator_service",
    "get_summarizer_service",
    "get_query_expander_service",
]
