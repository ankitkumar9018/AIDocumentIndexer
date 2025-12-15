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
    get_audit_service,
    audit_log,
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
]
