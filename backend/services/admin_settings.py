"""
Admin Settings Service - Phase 24
RBAC configuration, settings management, audit log viewing
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field
import json
import asyncio
from collections import defaultdict

from backend.services.enterprise import (
    Role, Permission, RBACService, AuditLogService, MultiTenantService
)


# =============================================================================
# SETTINGS MODELS
# =============================================================================

class SettingCategory(str, Enum):
    """Setting categories for organization"""
    GENERAL = "general"
    SECURITY = "security"
    PROCESSING = "processing"
    INTEGRATIONS = "integrations"
    NOTIFICATIONS = "notifications"
    BILLING = "billing"
    ADVANCED = "advanced"


class SettingType(str, Enum):
    """Types of settings values"""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    JSON = "json"
    SECRET = "secret"


class SettingDefinition(BaseModel):
    """Definition of a configurable setting"""
    key: str
    name: str
    description: str
    category: SettingCategory
    type: SettingType
    default_value: Any
    options: Optional[List[Dict[str, Any]]] = None  # For select types
    validation: Optional[Dict[str, Any]] = None  # min, max, pattern, etc.
    requires_permission: Permission = Permission.SETTINGS_READ
    requires_restart: bool = False
    sensitive: bool = False


class SettingValue(BaseModel):
    """Stored setting value"""
    key: str
    value: Any
    org_id: Optional[str] = None  # None = global default
    updated_by: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    previous_value: Optional[Any] = None


# =============================================================================
# PREDEFINED SETTINGS
# =============================================================================

SYSTEM_SETTINGS: List[SettingDefinition] = [
    # General Settings
    SettingDefinition(
        key="org_name",
        name="Organization Name",
        description="Display name for your organization",
        category=SettingCategory.GENERAL,
        type=SettingType.STRING,
        default_value="",
        validation={"max_length": 100}
    ),
    SettingDefinition(
        key="default_language",
        name="Default Language",
        description="Default language for the application",
        category=SettingCategory.GENERAL,
        type=SettingType.SELECT,
        default_value="en",
        options=[
            {"value": "en", "label": "English"},
            {"value": "es", "label": "Spanish"},
            {"value": "fr", "label": "French"},
            {"value": "de", "label": "German"},
            {"value": "ja", "label": "Japanese"},
        ]
    ),
    SettingDefinition(
        key="timezone",
        name="Timezone",
        description="Default timezone for dates and times",
        category=SettingCategory.GENERAL,
        type=SettingType.SELECT,
        default_value="UTC",
        options=[
            {"value": "UTC", "label": "UTC"},
            {"value": "America/New_York", "label": "Eastern Time"},
            {"value": "America/Los_Angeles", "label": "Pacific Time"},
            {"value": "Europe/London", "label": "London"},
            {"value": "Asia/Tokyo", "label": "Tokyo"},
        ]
    ),

    # Security Settings
    SettingDefinition(
        key="session_timeout_minutes",
        name="Session Timeout",
        description="Minutes of inactivity before session expires",
        category=SettingCategory.SECURITY,
        type=SettingType.NUMBER,
        default_value=60,
        validation={"min": 5, "max": 1440},
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="mfa_required",
        name="Require MFA",
        description="Require multi-factor authentication for all users",
        category=SettingCategory.SECURITY,
        type=SettingType.BOOLEAN,
        default_value=False,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="password_min_length",
        name="Minimum Password Length",
        description="Minimum characters required for passwords",
        category=SettingCategory.SECURITY,
        type=SettingType.NUMBER,
        default_value=8,
        validation={"min": 6, "max": 32},
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="allowed_ip_ranges",
        name="Allowed IP Ranges",
        description="Restrict access to specific IP ranges (CIDR notation)",
        category=SettingCategory.SECURITY,
        type=SettingType.JSON,
        default_value=[],
        requires_permission=Permission.SETTINGS_WRITE,
        sensitive=True
    ),
    SettingDefinition(
        key="api_rate_limit",
        name="API Rate Limit",
        description="Maximum API requests per minute per user",
        category=SettingCategory.SECURITY,
        type=SettingType.NUMBER,
        default_value=60,
        validation={"min": 10, "max": 1000},
        requires_permission=Permission.SETTINGS_WRITE
    ),

    # Processing Settings
    SettingDefinition(
        key="max_file_size_mb",
        name="Max File Size (MB)",
        description="Maximum file size allowed for uploads",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=100,
        validation={"min": 1, "max": 500}
    ),
    SettingDefinition(
        key="concurrent_processing_limit",
        name="Concurrent Processing Limit",
        description="Maximum documents processed simultaneously",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=10,
        validation={"min": 1, "max": 50},
        requires_restart=True
    ),
    SettingDefinition(
        key="ocr_engine",
        name="OCR Engine",
        description="Primary OCR engine for document processing",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="surya",
        options=[
            {"value": "mistral", "label": "Mistral OCR 3 (98%+ accuracy, best for complex docs)"},
            {"value": "surya", "label": "Surya (97.7% accuracy)"},
            {"value": "tesseract", "label": "Tesseract (Open Source)"},
            {"value": "claude_vision", "label": "Claude Vision (Best quality)"},
            {"value": "paddleocr", "label": "PaddleOCR (Multilingual)"},
        ]
    ),
    SettingDefinition(
        key="chunking_strategy",
        name="Chunking Strategy",
        description="How documents are split for indexing",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="semantic",
        options=[
            {"value": "semantic", "label": "Semantic (AI-powered)"},
            {"value": "fixed", "label": "Fixed Size"},
            {"value": "sentence", "label": "Sentence-based"},
            {"value": "paragraph", "label": "Paragraph-based"},
        ]
    ),
    SettingDefinition(
        key="embedding_model",
        name="Embedding Model",
        description="Model used for document embeddings",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="text-embedding-3-large",
        options=[
            {"value": "text-embedding-3-large", "label": "OpenAI Large (Best)"},
            {"value": "text-embedding-3-small", "label": "OpenAI Small (Faster)"},
            {"value": "colbert-v2", "label": "ColBERT v2 (Retrieval optimized)"},
        ],
        requires_restart=True
    ),

    # Integration Settings
    SettingDefinition(
        key="google_drive_enabled",
        name="Google Drive Integration",
        description="Enable Google Drive document import",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.BOOLEAN,
        default_value=True
    ),
    SettingDefinition(
        key="slack_webhook_url",
        name="Slack Webhook URL",
        description="Webhook for Slack notifications",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True
    ),
    SettingDefinition(
        key="webhook_endpoints",
        name="Webhook Endpoints",
        description="Custom webhook endpoints for events",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.JSON,
        default_value=[]
    ),

    # Notification Settings
    SettingDefinition(
        key="email_notifications",
        name="Email Notifications",
        description="Enable email notifications",
        category=SettingCategory.NOTIFICATIONS,
        type=SettingType.BOOLEAN,
        default_value=True
    ),
    SettingDefinition(
        key="notification_events",
        name="Notification Events",
        description="Events that trigger notifications",
        category=SettingCategory.NOTIFICATIONS,
        type=SettingType.MULTI_SELECT,
        default_value=["processing_complete", "processing_error"],
        options=[
            {"value": "processing_complete", "label": "Processing Complete"},
            {"value": "processing_error", "label": "Processing Error"},
            {"value": "user_invited", "label": "User Invited"},
            {"value": "storage_warning", "label": "Storage Warning"},
            {"value": "api_limit_warning", "label": "API Limit Warning"},
        ]
    ),

    # Advanced Settings
    SettingDefinition(
        key="cache_ttl_seconds",
        name="Cache TTL (seconds)",
        description="How long to cache query results",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=3600,
        validation={"min": 60, "max": 86400}
    ),
    SettingDefinition(
        key="debug_mode",
        name="Debug Mode",
        description="Enable detailed logging (affects performance)",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=False,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="experimental_features",
        name="Experimental Features",
        description="Enable experimental features",
        category=SettingCategory.ADVANCED,
        type=SettingType.MULTI_SELECT,
        default_value=[],
        options=[
            {"value": "recursive_lm", "label": "Recursive Language Model (10M+ context)"},
            {"value": "tree_of_thoughts", "label": "Tree of Thoughts reasoning"},
            {"value": "streaming_ingestion", "label": "Streaming document ingestion"},
            {"value": "voice_agents", "label": "Voice-enabled AI agents"},
            {"value": "warp_retrieval", "label": "WARP Engine (3x faster search)"},
            {"value": "colpali_visual", "label": "ColPali Visual Document Retrieval"},
            {"value": "sufficiency_detection", "label": "RAG Sufficiency Detection"},
            # Phase 62/63 features
            {"value": "answer_refiner", "label": "Answer Refiner (+20% quality)"},
            {"value": "ttt_compression", "label": "TTT Compression (35x faster long context)"},
            {"value": "fast_chunking", "label": "Fast Chunking (33x faster via Chonkie)"},
            {"value": "docling_parser", "label": "Docling Parser (97.9% table accuracy)"},
            {"value": "agent_evaluation", "label": "Agent Evaluation Metrics"},
        ]
    ),

    # ==========================================================================
    # LLM Provider Settings (Runtime Configurable)
    # ==========================================================================
    SettingDefinition(
        key="default_llm_provider",
        name="Default LLM Provider",
        description="Primary LLM provider for chat and generation",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="openai",
        options=[
            {"value": "openai", "label": "OpenAI (GPT-4o)"},
            {"value": "anthropic", "label": "Anthropic (Claude)"},
            {"value": "google", "label": "Google (Gemini)"},
            {"value": "groq", "label": "Groq (Fast inference)"},
            {"value": "together", "label": "Together AI"},
            {"value": "mistral", "label": "Mistral AI"},
            {"value": "deepseek", "label": "DeepSeek"},
        ]
    ),
    SettingDefinition(
        key="default_chat_model",
        name="Default Chat Model",
        description="Model to use for chat responses",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="gpt-4o",
        options=[
            {"value": "gpt-4o", "label": "GPT-4o (Best overall)"},
            {"value": "gpt-4o-mini", "label": "GPT-4o Mini (Faster)"},
            {"value": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet"},
            {"value": "claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku (Fast)"},
            {"value": "gemini-1.5-pro", "label": "Gemini 1.5 Pro"},
            {"value": "llama-3.1-70b", "label": "Llama 3.1 70B"},
        ]
    ),
    SettingDefinition(
        key="embedding_provider",
        name="Embedding Provider",
        description="Provider for document embeddings",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="openai",
        options=[
            {"value": "openai", "label": "OpenAI"},
            {"value": "voyage", "label": "Voyage AI (Best quality)"},
            {"value": "cohere", "label": "Cohere"},
            {"value": "jina", "label": "Jina AI"},
        ],
        requires_restart=True
    ),

    # ==========================================================================
    # RAG Settings (Runtime Configurable)
    # ==========================================================================
    SettingDefinition(
        key="rag_top_k",
        name="RAG Top-K Results",
        description="Number of documents to retrieve for each query",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=10,
        validation={"min": 1, "max": 50}
    ),
    SettingDefinition(
        key="rag_similarity_threshold",
        name="Similarity Threshold",
        description="Minimum similarity score for retrieved documents (0-1)",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=0.55,
        validation={"min": 0.0, "max": 1.0}
    ),
    SettingDefinition(
        key="enable_reranking",
        name="Enable Reranking",
        description="Use cross-encoder to rerank retrieved documents",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=True
    ),
    SettingDefinition(
        key="enable_hybrid_search",
        name="Enable Hybrid Search",
        description="Combine dense and sparse (BM25) retrieval",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=True
    ),
    SettingDefinition(
        key="enable_colbert",
        name="Enable ColBERT Retrieval",
        description="Use ColBERT PLAID for late interaction search",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=False
    ),

    # ==========================================================================
    # Audio/TTS Settings (Runtime Configurable)
    # ==========================================================================
    SettingDefinition(
        key="tts_provider",
        name="TTS Provider",
        description="Text-to-speech provider for audio generation",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="openai",
        options=[
            {"value": "openai", "label": "OpenAI TTS"},
            {"value": "elevenlabs", "label": "ElevenLabs (High quality)"},
            {"value": "cartesia", "label": "Cartesia (Ultra-fast)"},
        ]
    ),
    SettingDefinition(
        key="default_voice",
        name="Default Voice",
        description="Default voice for audio generation",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="alloy",
        options=[
            {"value": "alloy", "label": "Alloy (Neutral)"},
            {"value": "echo", "label": "Echo (Male)"},
            {"value": "fable", "label": "Fable (British)"},
            {"value": "onyx", "label": "Onyx (Deep)"},
            {"value": "nova", "label": "Nova (Female)"},
            {"value": "shimmer", "label": "Shimmer (Warm)"},
        ]
    ),
    SettingDefinition(
        key="enable_audio_overviews",
        name="Enable Audio Overviews",
        description="Allow generating audio summaries of documents",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=True
    ),

    # ==========================================================================
    # API Keys (Managed via Admin Panel - stored encrypted in DB)
    # ==========================================================================
    SettingDefinition(
        key="openai_api_key",
        name="OpenAI API Key",
        description="API key for OpenAI services (GPT, embeddings, TTS)",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="anthropic_api_key",
        name="Anthropic API Key",
        description="API key for Anthropic Claude models",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="voyage_api_key",
        name="Voyage AI API Key",
        description="API key for Voyage AI embeddings (Phase 29)",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="elevenlabs_api_key",
        name="ElevenLabs API Key",
        description="API key for ElevenLabs TTS",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="cohere_api_key",
        name="Cohere API Key",
        description="API key for Cohere reranking and embeddings",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="cartesia_api_key",
        name="Cartesia API Key",
        description="API key for Cartesia ultra-fast TTS",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="groq_api_key",
        name="Groq API Key",
        description="API key for Groq fast inference",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),

    # ==========================================================================
    # Feature Flags (Runtime Configurable)
    # ==========================================================================
    SettingDefinition(
        key="enable_knowledge_graph",
        name="Enable Knowledge Graph",
        description="Extract and use knowledge graphs from documents",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True
    ),
    SettingDefinition(
        key="enable_query_expansion",
        name="Enable Query Expansion",
        description="Expand queries with synonyms and related terms",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True
    ),
    SettingDefinition(
        key="enable_hyde",
        name="Enable HyDE",
        description="Use Hypothetical Document Embeddings for retrieval",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True
    ),
    SettingDefinition(
        key="enable_sufficiency_check",
        name="Enable Sufficiency Detection",
        description="Check if context is sufficient before answering (Phase 31)",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=False
    ),
    SettingDefinition(
        key="sufficiency_threshold",
        name="Sufficiency Threshold",
        description="Minimum confidence level required (0-1)",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=0.7,
        validation={"min": 0.0, "max": 1.0}
    ),

    # ==========================================================================
    # Infrastructure Settings (Admin Only - Requires Restart)
    # ==========================================================================
    SettingDefinition(
        key="database_url",
        name="Database URL",
        description="PostgreSQL connection string (requires restart)",
        category=SettingCategory.ADVANCED,
        type=SettingType.SECRET,
        default_value="postgresql+asyncpg://postgres:postgres@localhost:5432/aidocindexer",
        sensitive=True,
        requires_restart=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="redis_url",
        name="Redis URL",
        description="Redis connection string for caching and task queue (requires restart)",
        category=SettingCategory.ADVANCED,
        type=SettingType.SECRET,
        default_value="redis://localhost:6379/0",
        sensitive=True,
        requires_restart=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="secret_key",
        name="Secret Key",
        description="JWT signing key for authentication (requires restart)",
        category=SettingCategory.SECURITY,
        type=SettingType.SECRET,
        default_value="change-me-in-production",
        sensitive=True,
        requires_restart=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="sentry_dsn",
        name="Sentry DSN",
        description="Sentry error tracking DSN (leave empty to disable)",
        category=SettingCategory.ADVANCED,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_restart=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="environment",
        name="Environment",
        description="Application environment",
        category=SettingCategory.GENERAL,
        type=SettingType.SELECT,
        default_value="development",
        options=[
            {"value": "development", "label": "Development"},
            {"value": "staging", "label": "Staging"},
            {"value": "production", "label": "Production"},
        ],
        requires_restart=True
    ),

    # ==========================================================================
    # Storage Settings
    # ==========================================================================
    SettingDefinition(
        key="upload_dir",
        name="Upload Directory",
        description="Directory for uploaded files",
        category=SettingCategory.PROCESSING,
        type=SettingType.STRING,
        default_value="./uploads",
        validation={"max_length": 255},
        requires_restart=True
    ),
    SettingDefinition(
        key="audio_output_dir",
        name="Audio Output Directory",
        description="Directory for generated audio files",
        category=SettingCategory.PROCESSING,
        type=SettingType.STRING,
        default_value="./audio_output",
        validation={"max_length": 255},
        requires_restart=True
    ),

    # ==========================================================================
    # Additional API Keys
    # ==========================================================================
    SettingDefinition(
        key="google_api_key",
        name="Google AI API Key",
        description="API key for Google AI (Gemini) models",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="mistral_api_key",
        name="Mistral API Key",
        description="API key for Mistral AI models",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="together_api_key",
        name="Together AI API Key",
        description="API key for Together AI inference",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="deepseek_api_key",
        name="DeepSeek API Key",
        description="API key for DeepSeek models",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="jina_api_key",
        name="Jina AI API Key",
        description="API key for Jina AI embeddings",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="firecrawl_api_key",
        name="Firecrawl API Key",
        description="API key for Firecrawl web scraping",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),

    # ==========================================================================
    # Google Drive / OAuth
    # ==========================================================================
    SettingDefinition(
        key="google_oauth_client_id",
        name="Google OAuth Client ID",
        description="Client ID for Google OAuth (Drive integration)",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="google_oauth_client_secret",
        name="Google OAuth Client Secret",
        description="Client secret for Google OAuth",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="notion_integration_token",
        name="Notion Integration Token",
        description="Token for Notion workspace integration",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="slack_bot_token",
        name="Slack Bot Token",
        description="OAuth token for Slack bot",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="slack_signing_secret",
        name="Slack Signing Secret",
        description="Signing secret for Slack request verification",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),

    # ==========================================================================
    # Performance & Celery Settings
    # ==========================================================================
    SettingDefinition(
        key="celery_worker_concurrency",
        name="Celery Worker Concurrency",
        description="Number of concurrent Celery workers",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=4,
        validation={"min": 1, "max": 32},
        requires_restart=True
    ),
    SettingDefinition(
        key="bulk_upload_max_concurrent",
        name="Bulk Upload Concurrency",
        description="Max concurrent documents in bulk upload",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=4,
        validation={"min": 1, "max": 20}
    ),
    SettingDefinition(
        key="embedding_batch_size",
        name="Embedding Batch Size",
        description="Number of chunks to embed in a single batch",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=100,
        validation={"min": 10, "max": 500}
    ),
    SettingDefinition(
        key="embedding_cache_ttl",
        name="Embedding Cache TTL (seconds)",
        description="How long to cache embeddings (default: 7 days)",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=604800,
        validation={"min": 3600, "max": 2592000}
    ),

    # ==========================================================================
    # Phase 65: Advanced RAG Enhancements
    # ==========================================================================
    SettingDefinition(
        key="rag.ltr_enabled",
        name="Learning-to-Rank Enabled",
        description="Enable LTR reranking of search results based on user click feedback",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="rag.spell_correction_enabled",
        name="Spell Correction Enabled",
        description="Enable BK-tree based spell correction for search queries",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="rag.semantic_cache_enabled",
        name="Semantic Cache Enabled",
        description="Enable semantic query caching (cache hits for similar queries)",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="rag.semantic_cache_ttl",
        name="Semantic Cache TTL (seconds)",
        description="How long to cache query results (default: 5 minutes)",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=300,
        validation={"min": 60, "max": 3600}
    ),
    SettingDefinition(
        key="rag.semantic_cache_threshold",
        name="Semantic Cache Similarity Threshold",
        description="Minimum similarity (0-1) for semantic cache hit",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=0.92,
        validation={"min": 0.5, "max": 1.0}
    ),
    SettingDefinition(
        key="rag.streaming_citations_enabled",
        name="Streaming Citations Enabled",
        description="Enable real-time citation matching during streaming responses",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
    SettingDefinition(
        key="rag.gpu_acceleration_enabled",
        name="GPU Acceleration Enabled",
        description="Enable GPU-accelerated vector search (auto-fallback to CPU if unavailable)",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="rag.binary_quantization_enabled",
        name="Binary Quantization Enabled",
        description="Enable binary quantization for 32x memory reduction (best for 1M+ documents)",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),

    # ==========================================================================
    # Phase 68: Qwen3 Embedding Settings (70.58 MTEB - Top Performer)
    # ==========================================================================
    SettingDefinition(
        key="embedding.qwen3_enabled",
        name="Enable Qwen3 Embeddings",
        description="Use Qwen3-Embedding (70.58 MTEB score - highest accuracy)",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
    SettingDefinition(
        key="embedding.qwen3_model",
        name="Qwen3 Model",
        description="Qwen3 embedding model variant",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="Alibaba-NLP/Qwen3-Embedding-8B",
        options=[
            {"value": "Alibaba-NLP/Qwen3-Embedding-8B", "label": "Qwen3-8B (4096D, Best quality)"},
            {"value": "Alibaba-NLP/Qwen3-Embedding-4B", "label": "Qwen3-4B (2048D, Balanced)"},
            {"value": "Alibaba-NLP/Qwen3-Embedding-0.6B", "label": "Qwen3-0.6B (1024D, Fast)"},
        ],
        requires_restart=True
    ),
    SettingDefinition(
        key="embedding.qwen3_device",
        name="Qwen3 Device",
        description="Device for Qwen3 model (auto-detects GPU if available)",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="auto",
        options=[
            {"value": "auto", "label": "Auto-detect (GPU if available)"},
            {"value": "cuda", "label": "GPU (CUDA)"},
            {"value": "cpu", "label": "CPU only"},
        ],
    ),
    SettingDefinition(
        key="embedding.qwen3_use_fp16",
        name="Qwen3 Use FP16",
        description="Use half-precision for faster inference (GPU only)",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),

    # ==========================================================================
    # Phase 68: Mistral OCR 3 Settings (74% win rate)
    # ==========================================================================
    SettingDefinition(
        key="ocr.mistral_enabled",
        name="Enable Mistral OCR 3",
        description="Use Mistral OCR 3 for document processing (74% win rate, best for complex docs)",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
    SettingDefinition(
        key="ocr.mistral_api_key",
        name="Mistral API Key",
        description="API key for Mistral OCR 3 service",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="ocr.mistral_model",
        name="Mistral OCR Model",
        description="Mistral OCR model version",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="mistral-ocr-3",
        options=[
            {"value": "mistral-ocr-3", "label": "Mistral OCR 3 (Latest, best accuracy)"},
            {"value": "mistral-ocr-2", "label": "Mistral OCR 2 (Previous version)"},
        ],
    ),
    SettingDefinition(
        key="ocr.mistral_extract_tables",
        name="Mistral OCR Extract Tables",
        description="Extract structured tables from documents",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),

    # ==========================================================================
    # Phase 68: vLLM Settings (2-4x faster inference)
    # ==========================================================================
    SettingDefinition(
        key="llm.vllm_enabled",
        name="Enable vLLM",
        description="Use vLLM for 2-4x faster inference with PagedAttention",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
    SettingDefinition(
        key="llm.vllm_mode",
        name="vLLM Mode",
        description="How to connect to vLLM (API server or local engine)",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="openai_api",
        options=[
            {"value": "openai_api", "label": "OpenAI-compatible API (connect to vLLM server)"},
            {"value": "local_engine", "label": "Local Engine (load model directly)"},
            {"value": "litellm", "label": "Via LiteLLM (unified provider)"},
        ],
    ),
    SettingDefinition(
        key="llm.vllm_api_base",
        name="vLLM API Base URL",
        description="Base URL for vLLM OpenAI-compatible API",
        category=SettingCategory.PROCESSING,
        type=SettingType.STRING,
        default_value="http://localhost:8000/v1",
        validation={"pattern": r"^https?://.*"}
    ),
    SettingDefinition(
        key="llm.vllm_api_key",
        name="vLLM API Key",
        description="API key for vLLM server (if required)",
        category=SettingCategory.INTEGRATIONS,
        type=SettingType.SECRET,
        default_value="",
        sensitive=True,
        requires_permission=Permission.SETTINGS_WRITE
    ),
    SettingDefinition(
        key="llm.vllm_model",
        name="vLLM Model",
        description="Model to use with vLLM",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="meta-llama/Llama-3.1-8B-Instruct",
        options=[
            {"value": "meta-llama/Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B (Fast)"},
            {"value": "meta-llama/Llama-3.1-70B-Instruct", "label": "Llama 3.1 70B (Quality)"},
            {"value": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B"},
            {"value": "Qwen/Qwen2.5-72B-Instruct", "label": "Qwen 2.5 72B"},
            {"value": "deepseek-ai/DeepSeek-V3", "label": "DeepSeek V3"},
        ],
    ),
    SettingDefinition(
        key="llm.vllm_tensor_parallel_size",
        name="vLLM Tensor Parallel Size",
        description="Number of GPUs for tensor parallelism (local engine only)",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=1,
        validation={"min": 1, "max": 8},
        requires_restart=True
    ),
    SettingDefinition(
        key="llm.vllm_gpu_memory_utilization",
        name="vLLM GPU Memory Utilization",
        description="Fraction of GPU memory to use (0.0-1.0)",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=0.9,
        validation={"min": 0.1, "max": 0.99}
    ),
    SettingDefinition(
        key="llm.vllm_max_tokens",
        name="vLLM Max Tokens",
        description="Maximum tokens for generation",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=4096,
        validation={"min": 256, "max": 32768}
    ),

    # ==========================================================================
    # Phase 68: Qwen3 Reranker Settings (69.02 BEIR - Best Multilingual)
    # ==========================================================================
    SettingDefinition(
        key="reranker.qwen3_enabled",
        name="Enable Qwen3 Reranker",
        description="Use Qwen3-Reranker for multilingual reranking (69.02 BEIR - best open-source)",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
    SettingDefinition(
        key="reranker.qwen3_model",
        name="Qwen3 Reranker Model",
        description="Qwen3 reranker model variant",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="qwen3-reranker-8b",
        options=[
            {"value": "qwen3-reranker-8b", "label": "Qwen3-Reranker-8B (8K context, best quality)"},
            {"value": "qwen3-reranker-4b", "label": "Qwen3-Reranker-4B (balanced)"},
            {"value": "qwen3-reranker-small", "label": "Qwen3-Reranker-0.6B (fast)"},
        ],
    ),
    SettingDefinition(
        key="reranker.qwen3_max_length",
        name="Qwen3 Reranker Max Length",
        description="Maximum sequence length for reranking (Qwen3 supports up to 8K)",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=8192,
        validation={"min": 512, "max": 8192}
    ),
    SettingDefinition(
        key="reranker.qwen3_use_fp16",
        name="Qwen3 Reranker Use FP16",
        description="Use half-precision for faster inference (GPU only)",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="reranker.qwen3_batch_size",
        name="Qwen3 Reranker Batch Size",
        description="Batch size for reranking (lower for memory-constrained systems)",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=32,
        validation={"min": 1, "max": 128}
    ),

    # ==========================================================================
    # Phase 68: RAGCache v2 Settings (4x TTFT, 2.1x Throughput)
    # ==========================================================================
    SettingDefinition(
        key="cache.ragcache_v2_enabled",
        name="Enable RAGCache v2",
        description="Use RAGCache v2 for 4x faster TTFT and 2.1x throughput improvement",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
    SettingDefinition(
        key="cache.heavy_hitter_enabled",
        name="Enable Heavy-Hitter Filtering",
        description="Cache frequently accessed chunks for 90% index size reduction",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="cache.heavy_hitter_threshold",
        name="Heavy-Hitter Threshold",
        description="Minimum access count to promote chunk to heavy-hitter cache",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=5,
        validation={"min": 2, "max": 50}
    ),
    SettingDefinition(
        key="cache.heavy_hitter_cache_size",
        name="Heavy-Hitter Cache Size",
        description="Maximum number of heavy-hitters to cache",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=1000,
        validation={"min": 100, "max": 10000}
    ),
    SettingDefinition(
        key="cache.prefix_kv_enabled",
        name="Enable Prefix KV Caching",
        description="Cache KV states for common prompts to reduce TTFT",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="cache.prefix_cache_size",
        name="Prefix Cache Size",
        description="Number of prefix KV states to cache",
        category=SettingCategory.ADVANCED,
        type=SettingType.NUMBER,
        default_value=500,
        validation={"min": 50, "max": 2000}
    ),
    SettingDefinition(
        key="cache.retrieval_aware_enabled",
        name="Enable Retrieval-Aware Caching",
        description="Group and cache related chunks together for optimal efficiency",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),

    # ==========================================================================
    # Phase 68: Amazon Nova Multimodal Embeddings
    # ==========================================================================
    SettingDefinition(
        key="embedding.nova_enabled",
        name="Enable Amazon Nova Embeddings",
        description="Use Amazon Nova for unified text + image + video + audio embeddings",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
    SettingDefinition(
        key="embedding.nova_model",
        name="Nova Embedding Model",
        description="Amazon Nova embedding model variant",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="amazon.nova-embed-multimodal-v1",
        options=[
            {"value": "amazon.nova-embed-multimodal-v1", "label": "Nova Multimodal (all modalities)"},
            {"value": "amazon.nova-embed-v1", "label": "Nova Embed v1 (text + image)"},
            {"value": "amazon.nova-embed-lite-v1", "label": "Nova Lite (faster, lower dim)"},
        ],
    ),
    SettingDefinition(
        key="embedding.nova_dimension",
        name="Nova Embedding Dimension",
        description="Output embedding dimension (256-1024)",
        category=SettingCategory.PROCESSING,
        type=SettingType.NUMBER,
        default_value=1024,
        validation={"min": 256, "max": 1024}
    ),
    SettingDefinition(
        key="embedding.nova_aws_region",
        name="AWS Region for Nova",
        description="AWS region for Bedrock Nova API",
        category=SettingCategory.PROCESSING,
        type=SettingType.SELECT,
        default_value="us-east-1",
        options=[
            {"value": "us-east-1", "label": "US East (N. Virginia)"},
            {"value": "us-west-2", "label": "US West (Oregon)"},
            {"value": "eu-west-1", "label": "Europe (Ireland)"},
            {"value": "ap-northeast-1", "label": "Asia Pacific (Tokyo)"},
        ],
    ),
    SettingDefinition(
        key="embedding.nova_normalize",
        name="Normalize Nova Embeddings",
        description="Normalize embeddings to unit length",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),

    # ==========================================================================
    # Phase 68: Cost Monitoring Settings
    # ==========================================================================
    SettingDefinition(
        key="cost.tracking_enabled",
        name="Enable Cost Tracking",
        description="Track LLM, embedding, and API costs for budget management",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="cost.monthly_budget_usd",
        name="Monthly Budget (USD)",
        description="Monthly spending limit for alerts (0 = unlimited)",
        category=SettingCategory.BILLING,
        type=SettingType.NUMBER,
        default_value=0,
        validation={"min": 0, "max": 100000}
    ),
    SettingDefinition(
        key="cost.alert_threshold_percent",
        name="Alert Threshold (%)",
        description="Notify when spending reaches this percentage of budget",
        category=SettingCategory.BILLING,
        type=SettingType.NUMBER,
        default_value=80,
        validation={"min": 10, "max": 100}
    ),
    SettingDefinition(
        key="cost.cache_savings_tracking",
        name="Track Cache Savings",
        description="Estimate and track cost savings from caching",
        category=SettingCategory.ADVANCED,
        type=SettingType.BOOLEAN,
        default_value=True,
    ),
    SettingDefinition(
        key="cost.model_routing_optimization",
        name="Enable Smart Model Routing",
        description="Automatically route simple queries to cheaper models",
        category=SettingCategory.PROCESSING,
        type=SettingType.BOOLEAN,
        default_value=False,
    ),
]

# Create lookup dictionary
SETTINGS_BY_KEY: Dict[str, SettingDefinition] = {s.key: s for s in SYSTEM_SETTINGS}


# =============================================================================
# ADMIN SETTINGS SERVICE
# =============================================================================

class AdminSettingsService:
    """
    Admin settings management service.
    Handles configuration, RBAC admin, and settings persistence.
    """

    def __init__(
        self,
        rbac_service: Optional[RBACService] = None,
        audit_service: Optional[AuditLogService] = None,
        tenant_service: Optional[MultiTenantService] = None
    ):
        self._rbac = rbac_service or RBACService()
        self._audit = audit_service or AuditLogService()
        self._tenant = tenant_service or MultiTenantService()

        # In-memory storage (replace with database in production)
        self._settings: Dict[str, Dict[str, SettingValue]] = defaultdict(dict)  # org_id -> key -> value
        self._global_settings: Dict[str, SettingValue] = {}

    # -------------------------------------------------------------------------
    # Settings Management
    # -------------------------------------------------------------------------

    async def get_all_settings(
        self,
        org_id: str,
        user_id: str,
        category: Optional[SettingCategory] = None
    ) -> Dict[str, Any]:
        """Get all settings for an organization"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            user_id, org_id, Permission.SETTINGS_READ
        )
        if not has_permission:
            raise PermissionError("No permission to read settings")

        result = {}
        for definition in SYSTEM_SETTINGS:
            if category and definition.category != category:
                continue

            # Get effective value (org override or global or default)
            value = await self._get_effective_value(org_id, definition.key)

            # Mask sensitive values for non-admins
            if definition.sensitive:
                has_write = await self._rbac.check_permission(
                    user_id, org_id, Permission.SETTINGS_WRITE
                )
                if not has_write and value:
                    value = "********"

            result[definition.key] = {
                "value": value,
                "definition": definition.model_dump(),
                "is_default": definition.key not in self._settings.get(org_id, {}),
            }

        return result

    async def get_setting(
        self,
        org_id: str,
        key: str,
        user_id: Optional[str] = None
    ) -> Any:
        """Get a single setting value"""
        if key not in SETTINGS_BY_KEY:
            raise ValueError(f"Unknown setting: {key}")

        definition = SETTINGS_BY_KEY[key]

        # Check permission if user provided
        if user_id:
            has_permission = await self._rbac.check_permission(
                user_id, org_id, definition.requires_permission
            )
            if not has_permission:
                raise PermissionError(f"No permission to read setting: {key}")

        return await self._get_effective_value(org_id, key)

    async def update_setting(
        self,
        org_id: str,
        user_id: str,
        key: str,
        value: Any
    ) -> SettingValue:
        """Update a setting value"""
        if key not in SETTINGS_BY_KEY:
            raise ValueError(f"Unknown setting: {key}")

        definition = SETTINGS_BY_KEY[key]

        # Check write permission
        has_permission = await self._rbac.check_permission(
            user_id, org_id, Permission.SETTINGS_WRITE
        )
        if not has_permission:
            raise PermissionError(f"No permission to update setting: {key}")

        # Validate value
        self._validate_setting_value(definition, value)

        # Get previous value for audit
        previous_value = await self._get_effective_value(org_id, key)

        # Store new value
        setting_value = SettingValue(
            key=key,
            value=value,
            org_id=org_id,
            updated_by=user_id,
            previous_value=previous_value
        )

        if org_id not in self._settings:
            self._settings[org_id] = {}
        self._settings[org_id][key] = setting_value

        # Sync to Redis for runtime access (non-blocking)
        await self._sync_to_redis(org_id, key, value)

        # Audit log
        await self._audit.log(
            action="setting_updated",
            user_id=user_id,
            org_id=org_id,
            resource_type="setting",
            resource_id=key,
            details={
                "previous_value": previous_value if not definition.sensitive else "[REDACTED]",
                "new_value": value if not definition.sensitive else "[REDACTED]",
                "requires_restart": definition.requires_restart
            }
        )

        return setting_value

    async def reset_setting(
        self,
        org_id: str,
        user_id: str,
        key: str
    ) -> Any:
        """Reset a setting to default value"""
        if key not in SETTINGS_BY_KEY:
            raise ValueError(f"Unknown setting: {key}")

        # Check permission
        has_permission = await self._rbac.check_permission(
            user_id, org_id, Permission.SETTINGS_WRITE
        )
        if not has_permission:
            raise PermissionError(f"No permission to reset setting: {key}")

        # Remove org-specific override
        if org_id in self._settings and key in self._settings[org_id]:
            del self._settings[org_id][key]

        # Audit log
        await self._audit.log(
            action="setting_reset",
            user_id=user_id,
            org_id=org_id,
            resource_type="setting",
            resource_id=key
        )

        return SETTINGS_BY_KEY[key].default_value

    async def bulk_update_settings(
        self,
        org_id: str,
        user_id: str,
        settings: Dict[str, Any]
    ) -> Dict[str, SettingValue]:
        """Update multiple settings at once"""
        results = {}
        errors = []

        for key, value in settings.items():
            try:
                results[key] = await self.update_setting(org_id, user_id, key, value)
            except Exception as e:
                errors.append({"key": key, "error": str(e)})

        if errors:
            raise ValueError(f"Some settings failed to update: {errors}")

        return results

    async def _get_effective_value(self, org_id: str, key: str) -> Any:
        """Get effective value for a setting (Redis > org override > global > env > default)"""
        # Check Redis first for runtime settings
        try:
            from backend.core.config import get_runtime_setting
            redis_value = await get_runtime_setting(key, org_id=org_id)
            if redis_value is not None:
                return redis_value
        except Exception:
            pass  # Redis not available, continue with fallbacks

        # Check org-specific override
        if org_id in self._settings and key in self._settings[org_id]:
            return self._settings[org_id][key].value

        # Check global override
        if key in self._global_settings:
            return self._global_settings[key].value

        # Check environment variable (uppercase key)
        import os
        env_value = os.getenv(key.upper())
        if env_value is not None:
            definition = SETTINGS_BY_KEY.get(key)
            if definition:
                # Convert to appropriate type
                if definition.type == SettingType.BOOLEAN:
                    return env_value.lower() in ("true", "1", "yes", "on")
                elif definition.type == SettingType.NUMBER:
                    try:
                        return float(env_value) if "." in env_value else int(env_value)
                    except ValueError:
                        pass
            return env_value

        # Return default
        return SETTINGS_BY_KEY[key].default_value

    async def _sync_to_redis(self, org_id: str, key: str, value: Any) -> None:
        """Sync a setting to Redis for runtime access"""
        try:
            from backend.core.config import set_runtime_setting
            await set_runtime_setting(key, value, org_id=org_id)
        except Exception as e:
            # Log but don't fail - Redis sync is optional
            import structlog
            logger = structlog.get_logger(__name__)
            logger.warning(f"Failed to sync setting to Redis: {e}")

    def _validate_setting_value(self, definition: SettingDefinition, value: Any) -> None:
        """Validate a setting value against its definition"""
        validation = definition.validation or {}

        if definition.type == SettingType.STRING:
            if not isinstance(value, str):
                raise ValueError(f"Expected string for {definition.key}")
            if "max_length" in validation and len(value) > validation["max_length"]:
                raise ValueError(f"Value exceeds max length of {validation['max_length']}")
            if "pattern" in validation:
                import re
                if not re.match(validation["pattern"], value):
                    raise ValueError(f"Value does not match required pattern")

        elif definition.type == SettingType.NUMBER:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Expected number for {definition.key}")
            if "min" in validation and value < validation["min"]:
                raise ValueError(f"Value below minimum of {validation['min']}")
            if "max" in validation and value > validation["max"]:
                raise ValueError(f"Value exceeds maximum of {validation['max']}")

        elif definition.type == SettingType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"Expected boolean for {definition.key}")

        elif definition.type == SettingType.SELECT:
            valid_values = [opt["value"] for opt in (definition.options or [])]
            if value not in valid_values:
                raise ValueError(f"Invalid option for {definition.key}: {value}")

        elif definition.type == SettingType.MULTI_SELECT:
            if not isinstance(value, list):
                raise ValueError(f"Expected list for {definition.key}")
            valid_values = [opt["value"] for opt in (definition.options or [])]
            for v in value:
                if v not in valid_values:
                    raise ValueError(f"Invalid option for {definition.key}: {v}")

        elif definition.type == SettingType.JSON:
            if not isinstance(value, (dict, list)):
                raise ValueError(f"Expected JSON object or array for {definition.key}")

    # -------------------------------------------------------------------------
    # RBAC Administration
    # -------------------------------------------------------------------------

    async def list_users(
        self,
        org_id: str,
        admin_user_id: str,
        page: int = 1,
        page_size: int = 20,
        role_filter: Optional[Role] = None
    ) -> Dict[str, Any]:
        """List users in an organization with their roles"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            admin_user_id, org_id, Permission.USER_READ
        )
        if not has_permission:
            raise PermissionError("No permission to list users")

        # Get all user roles for org
        users = []
        org_roles = self._rbac._user_roles.get(org_id, {})

        for user_id, role in org_roles.items():
            if role_filter and role != role_filter:
                continue

            permissions = await self._rbac.get_user_permissions(user_id, org_id)
            users.append({
                "user_id": user_id,
                "role": role.value,
                "permissions": [p.value for p in permissions],
            })

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated = users[start:end]

        return {
            "users": paginated,
            "total": len(users),
            "page": page,
            "page_size": page_size,
            "total_pages": (len(users) + page_size - 1) // page_size
        }

    async def update_user_role(
        self,
        org_id: str,
        admin_user_id: str,
        target_user_id: str,
        new_role: Role
    ) -> Dict[str, Any]:
        """Update a user's role"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            admin_user_id, org_id, Permission.USER_MANAGE
        )
        if not has_permission:
            raise PermissionError("No permission to manage users")

        # Get current role
        old_role = self._rbac._user_roles.get(org_id, {}).get(target_user_id)

        # Prevent removing last admin
        if old_role == Role.ORG_ADMIN and new_role != Role.ORG_ADMIN:
            admin_count = sum(
                1 for r in self._rbac._user_roles.get(org_id, {}).values()
                if r == Role.ORG_ADMIN
            )
            if admin_count <= 1:
                raise ValueError("Cannot remove the last organization admin")

        # Update role
        await self._rbac.assign_role(target_user_id, org_id, new_role)

        # Audit log
        await self._audit.log(
            action="user_role_updated",
            user_id=admin_user_id,
            org_id=org_id,
            resource_type="user",
            resource_id=target_user_id,
            details={
                "old_role": old_role.value if old_role else None,
                "new_role": new_role.value
            }
        )

        permissions = await self._rbac.get_user_permissions(target_user_id, org_id)

        return {
            "user_id": target_user_id,
            "role": new_role.value,
            "permissions": [p.value for p in permissions]
        }

    async def invite_user(
        self,
        org_id: str,
        admin_user_id: str,
        email: str,
        role: Role = Role.VIEWER
    ) -> Dict[str, Any]:
        """Invite a new user to the organization"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            admin_user_id, org_id, Permission.USER_INVITE
        )
        if not has_permission:
            raise PermissionError("No permission to invite users")

        # Generate invite token
        invite_token = str(uuid4())
        invite_id = str(uuid4())

        # Store invite (would be persisted in production)
        invite = {
            "id": invite_id,
            "email": email,
            "org_id": org_id,
            "role": role.value,
            "invited_by": admin_user_id,
            "token": invite_token,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "status": "pending"
        }

        # Audit log
        await self._audit.log(
            action="user_invited",
            user_id=admin_user_id,
            org_id=org_id,
            resource_type="invite",
            resource_id=invite_id,
            details={"email": email, "role": role.value}
        )

        return invite

    async def remove_user(
        self,
        org_id: str,
        admin_user_id: str,
        target_user_id: str
    ) -> bool:
        """Remove a user from the organization"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            admin_user_id, org_id, Permission.USER_DELETE
        )
        if not has_permission:
            raise PermissionError("No permission to remove users")

        # Prevent self-removal
        if admin_user_id == target_user_id:
            raise ValueError("Cannot remove yourself")

        # Prevent removing last admin
        old_role = self._rbac._user_roles.get(org_id, {}).get(target_user_id)
        if old_role == Role.ORG_ADMIN:
            admin_count = sum(
                1 for r in self._rbac._user_roles.get(org_id, {}).values()
                if r == Role.ORG_ADMIN
            )
            if admin_count <= 1:
                raise ValueError("Cannot remove the last organization admin")

        # Remove user
        if org_id in self._rbac._user_roles:
            self._rbac._user_roles[org_id].pop(target_user_id, None)

        # Audit log
        await self._audit.log(
            action="user_removed",
            user_id=admin_user_id,
            org_id=org_id,
            resource_type="user",
            resource_id=target_user_id,
            details={"old_role": old_role.value if old_role else None}
        )

        return True

    # -------------------------------------------------------------------------
    # Audit Log Administration
    # -------------------------------------------------------------------------

    async def get_audit_logs(
        self,
        org_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Get audit logs with filtering and pagination"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            user_id, org_id, Permission.AUDIT_READ
        )
        if not has_permission:
            raise PermissionError("No permission to read audit logs")

        # Query audit service
        logs = await self._audit.query(
            org_id=org_id,
            limit=1000,  # Get more for filtering
            **(filters or {})
        )

        # Apply additional filters
        if filters:
            if "action" in filters:
                logs = [l for l in logs if l.action == filters["action"]]
            if "user_id" in filters:
                logs = [l for l in logs if l.user_id == filters["user_id"]]
            if "resource_type" in filters:
                logs = [l for l in logs if l.resource_type == filters["resource_type"]]
            if "start_date" in filters:
                start = datetime.fromisoformat(filters["start_date"])
                logs = [l for l in logs if l.timestamp >= start]
            if "end_date" in filters:
                end = datetime.fromisoformat(filters["end_date"])
                logs = [l for l in logs if l.timestamp <= end]

        # Pagination
        total = len(logs)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = logs[start:end]

        return {
            "logs": [log.model_dump() for log in paginated],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }

    async def export_audit_logs(
        self,
        org_id: str,
        user_id: str,
        format: str = "csv",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Export audit logs for compliance"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            user_id, org_id, Permission.AUDIT_EXPORT
        )
        if not has_permission:
            raise PermissionError("No permission to export audit logs")

        # Get all matching logs
        logs_response = await self.get_audit_logs(
            org_id, user_id, filters, page=1, page_size=10000
        )
        logs = logs_response["logs"]

        # Generate export
        if format == "csv":
            content = self._generate_csv_export(logs)
            content_type = "text/csv"
            filename = f"audit_logs_{org_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        elif format == "json":
            content = json.dumps(logs, indent=2, default=str)
            content_type = "application/json"
            filename = f"audit_logs_{org_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            raise ValueError(f"Unsupported export format: {format}")

        # Audit the export
        await self._audit.log(
            action="audit_logs_exported",
            user_id=user_id,
            org_id=org_id,
            resource_type="audit_export",
            resource_id=filename,
            details={"format": format, "record_count": len(logs)}
        )

        return {
            "content": content,
            "content_type": content_type,
            "filename": filename,
            "record_count": len(logs)
        }

    def _generate_csv_export(self, logs: List[Dict]) -> str:
        """Generate CSV content from audit logs"""
        if not logs:
            return "timestamp,action,user_id,org_id,resource_type,resource_id,ip_address,details\n"

        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "timestamp", "action", "user_id", "org_id",
            "resource_type", "resource_id", "ip_address", "details"
        ])

        # Rows
        for log in logs:
            writer.writerow([
                log.get("timestamp", ""),
                log.get("action", ""),
                log.get("user_id", ""),
                log.get("org_id", ""),
                log.get("resource_type", ""),
                log.get("resource_id", ""),
                log.get("ip_address", ""),
                json.dumps(log.get("details", {}))
            ])

        return output.getvalue()

    # -------------------------------------------------------------------------
    # System Statistics
    # -------------------------------------------------------------------------

    async def get_system_stats(
        self,
        org_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Get system statistics for admin dashboard"""
        # Check permission
        has_permission = await self._rbac.check_permission(
            user_id, org_id, Permission.SETTINGS_READ
        )
        if not has_permission:
            raise PermissionError("No permission to view system stats")

        # Get org info
        org = await self._tenant.get_organization(org_id)

        # Calculate stats
        user_count = len(self._rbac._user_roles.get(org_id, {}))

        # Get recent audit activity
        recent_logs = await self._audit.query(org_id=org_id, limit=100)

        # Count by action type
        action_counts = defaultdict(int)
        for log in recent_logs:
            action_counts[log.action] += 1

        return {
            "organization": {
                "id": org_id,
                "name": org.name if org else "Unknown",
                "tier": org.tier if org else "free",
            },
            "users": {
                "total": user_count,
                "by_role": self._count_users_by_role(org_id)
            },
            "activity": {
                "recent_actions": len(recent_logs),
                "by_action": dict(action_counts),
            },
            "settings": {
                "custom_overrides": len(self._settings.get(org_id, {})),
                "total_available": len(SYSTEM_SETTINGS)
            }
        }

    def _count_users_by_role(self, org_id: str) -> Dict[str, int]:
        """Count users by role"""
        counts = defaultdict(int)
        for role in self._rbac._user_roles.get(org_id, {}).values():
            counts[role.value] += 1
        return dict(counts)


# =============================================================================
# API ROUTES HELPER
# =============================================================================

def create_admin_router():
    """Create FastAPI router for admin endpoints"""
    from fastapi import APIRouter, Depends, HTTPException, Query, Response
    from typing import Annotated

    router = APIRouter(prefix="/admin", tags=["admin"])

    # Dependency injection placeholder
    async def get_admin_service() -> AdminSettingsService:
        return AdminSettingsService()

    async def get_current_user():
        # Placeholder - would be replaced with actual auth
        return {"user_id": "current_user", "org_id": "current_org"}

    @router.get("/settings")
    async def list_settings(
        category: Optional[SettingCategory] = None,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """List all settings for the organization"""
        return await service.get_all_settings(
            user["org_id"], user["user_id"], category
        )

    @router.get("/settings/{key}")
    async def get_setting(
        key: str,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Get a specific setting"""
        return await service.get_setting(user["org_id"], key, user["user_id"])

    @router.put("/settings/{key}")
    async def update_setting(
        key: str,
        value: Any,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Update a setting"""
        return await service.update_setting(
            user["org_id"], user["user_id"], key, value
        )

    @router.delete("/settings/{key}")
    async def reset_setting(
        key: str,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Reset a setting to default"""
        return await service.reset_setting(user["org_id"], user["user_id"], key)

    @router.get("/users")
    async def list_users(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        role: Optional[Role] = None,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """List organization users"""
        return await service.list_users(
            user["org_id"], user["user_id"], page, page_size, role
        )

    @router.put("/users/{target_user_id}/role")
    async def update_user_role(
        target_user_id: str,
        role: Role,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Update a user's role"""
        return await service.update_user_role(
            user["org_id"], user["user_id"], target_user_id, role
        )

    @router.post("/users/invite")
    async def invite_user(
        email: str,
        role: Role = Role.VIEWER,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Invite a new user"""
        return await service.invite_user(
            user["org_id"], user["user_id"], email, role
        )

    @router.delete("/users/{target_user_id}")
    async def remove_user(
        target_user_id: str,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Remove a user from organization"""
        return await service.remove_user(
            user["org_id"], user["user_id"], target_user_id
        )

    @router.get("/audit-logs")
    async def get_audit_logs(
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=100),
        action: Optional[str] = None,
        target_user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Get audit logs with filtering"""
        filters = {}
        if action:
            filters["action"] = action
        if target_user_id:
            filters["user_id"] = target_user_id
        if resource_type:
            filters["resource_type"] = resource_type
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date

        return await service.get_audit_logs(
            user["org_id"], user["user_id"], filters, page, page_size
        )

    @router.get("/audit-logs/export")
    async def export_audit_logs(
        format: str = Query("csv", regex="^(csv|json)$"),
        action: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Export audit logs"""
        filters = {}
        if action:
            filters["action"] = action
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date

        result = await service.export_audit_logs(
            user["org_id"], user["user_id"], format, filters
        )

        return Response(
            content=result["content"],
            media_type=result["content_type"],
            headers={
                "Content-Disposition": f"attachment; filename={result['filename']}"
            }
        )

    @router.get("/stats")
    async def get_system_stats(
        service: AdminSettingsService = Depends(get_admin_service),
        user: dict = Depends(get_current_user)
    ):
        """Get system statistics"""
        return await service.get_system_stats(user["org_id"], user["user_id"])

    return router
