"""
AIDocumentIndexer - Core Configuration
========================================

Centralized configuration using Pydantic settings with database override support.
Settings can be configured via:
1. Environment variables (.env file) - For secrets and infrastructure
2. Database/Redis-persisted settings (via Admin panel) - For runtime config
3. Defaults - Sensible defaults for all settings

Priority: Database > Environment > Defaults

Usage:
    from backend.core.config import settings, get_runtime_setting

    # Static setting from env (API keys, database URLs)
    api_key = settings.OPENAI_API_KEY

    # Runtime-configurable setting (checks database first)
    top_k = await get_runtime_setting("rag_top_k", default=10)
"""

import os
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Any, Dict, Union

from pydantic import Field
from pydantic_settings import BaseSettings

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Runtime Settings Cache (for database-backed settings)
# =============================================================================

_runtime_settings_cache: Dict[str, Any] = {}
_cache_ttl = 60  # seconds
_cache_timestamps: Dict[str, float] = {}


async def get_runtime_setting(
    key: str,
    default: Any = None,
    org_id: Optional[str] = None,
) -> Any:
    """
    Get a runtime-configurable setting.

    Checks Redis/database first, then falls back to env/default.
    Use this for settings that should be changeable without restart.

    Args:
        key: Setting key (case-insensitive)
        default: Default value if not found
        org_id: Optional organization ID for org-specific settings

    Returns:
        Setting value
    """
    cache_key = f"{org_id or 'global'}:{key.lower()}"
    now = time.time()

    # Check cache
    if cache_key in _runtime_settings_cache:
        if now - _cache_timestamps.get(cache_key, 0) < _cache_ttl:
            return _runtime_settings_cache[cache_key]

    # Try Redis
    try:
        from backend.services.redis_client import get_redis_client
        redis = await get_redis_client()
        if redis:
            redis_key = f"settings:{cache_key}"
            value = await redis.get(redis_key)
            if value is not None:
                try:
                    parsed = json.loads(value)
                    _runtime_settings_cache[cache_key] = parsed
                    _cache_timestamps[cache_key] = now
                    return parsed
                except json.JSONDecodeError:
                    _runtime_settings_cache[cache_key] = value
                    _cache_timestamps[cache_key] = now
                    return value
    except Exception as e:
        logger.debug(f"Redis setting lookup failed: {e}")

    # Fall back to environment/default
    env_value = os.getenv(key.upper())
    if env_value is not None:
        return _convert_env_value(env_value, default)

    return default


def get_runtime_setting_sync(
    key: str,
    default: Any = None,
    org_id: Optional[str] = None,
) -> Any:
    """
    Synchronous version of get_runtime_setting.

    Only checks cache and environment, not Redis.
    Use async version when possible.
    """
    cache_key = f"{org_id or 'global'}:{key.lower()}"
    now = time.time()

    # Check cache
    if cache_key in _runtime_settings_cache:
        if now - _cache_timestamps.get(cache_key, 0) < _cache_ttl:
            return _runtime_settings_cache[cache_key]

    # Fall back to environment/default
    env_value = os.getenv(key.upper())
    if env_value is not None:
        return _convert_env_value(env_value, default)

    return default


def _convert_env_value(value: str, default: Any) -> Any:
    """Convert string env value to appropriate type based on default."""
    if default is None:
        return value

    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes", "on")
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    if isinstance(default, list):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.split(",")

    return value


async def set_runtime_setting(
    key: str,
    value: Any,
    org_id: Optional[str] = None,
    ttl: Optional[int] = None,
) -> bool:
    """
    Set a runtime-configurable setting in Redis.

    Args:
        key: Setting key
        value: Setting value
        org_id: Optional organization ID
        ttl: Optional TTL in seconds (default: no expiration)

    Returns:
        True if successful
    """
    try:
        from backend.services.redis_client import get_redis_client
        redis = await get_redis_client()
        if redis:
            cache_key = f"{org_id or 'global'}:{key.lower()}"
            redis_key = f"settings:{cache_key}"

            serialized = json.dumps(value) if not isinstance(value, str) else value

            if ttl:
                await redis.setex(redis_key, ttl, serialized)
            else:
                await redis.set(redis_key, serialized)

            # Update cache
            _runtime_settings_cache[cache_key] = value
            _cache_timestamps[cache_key] = time.time()

            logger.info(f"Runtime setting updated: {key}", org_id=org_id)
            return True
    except Exception as e:
        logger.error(f"Failed to set runtime setting: {e}")

    return False


async def delete_runtime_setting(
    key: str,
    org_id: Optional[str] = None,
) -> bool:
    """
    Delete a runtime setting (reverts to env/default).

    Args:
        key: Setting key
        org_id: Optional organization ID

    Returns:
        True if successful
    """
    try:
        from backend.services.redis_client import get_redis_client
        redis = await get_redis_client()
        if redis:
            cache_key = f"{org_id or 'global'}:{key.lower()}"
            redis_key = f"settings:{cache_key}"
            await redis.delete(redis_key)

            # Clear cache
            _runtime_settings_cache.pop(cache_key, None)
            _cache_timestamps.pop(cache_key, None)

            return True
    except Exception as e:
        logger.error(f"Failed to delete runtime setting: {e}")

    return False


def clear_settings_cache():
    """Clear the runtime settings cache."""
    _runtime_settings_cache.clear()
    _cache_timestamps.clear()
    logger.info("Settings cache cleared")


async def get_all_runtime_settings(org_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all runtime settings for an organization.

    Args:
        org_id: Optional organization ID

    Returns:
        Dictionary of all settings
    """
    try:
        from backend.services.redis_client import get_redis_client
        redis = await get_redis_client()
        if redis:
            prefix = f"settings:{org_id or 'global'}:"
            keys = []
            async for key in redis.scan_iter(match=f"{prefix}*"):
                keys.append(key)

            result = {}
            for key in keys:
                setting_key = key.decode() if isinstance(key, bytes) else key
                setting_key = setting_key.replace(prefix, "")
                value = await redis.get(key)
                if value:
                    try:
                        result[setting_key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[setting_key] = value.decode() if isinstance(value, bytes) else value

            return result
    except Exception as e:
        logger.error(f"Failed to get all runtime settings: {e}")

    return {}


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    These are static settings that require restart to change.
    For runtime-configurable settings, use get_runtime_setting().

    Settings are divided into:
    - SECRETS: API keys, tokens (must be in .env, never in database)
    - INFRASTRUCTURE: Database URLs, paths (usually in .env)
    - CONFIGURABLE: Can be changed at runtime via admin panel
    """

    # ==========================================================================
    # Application
    # ==========================================================================
    APP_NAME: str = Field(default="AIDocumentIndexer", description="Application name")
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment (development, staging, production)")

    # ==========================================================================
    # API Keys - LLM Providers (SECRETS - .env only)
    # ==========================================================================
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
    GOOGLE_API_KEY: str = Field(default="", description="Google AI API key")
    COHERE_API_KEY: str = Field(default="", description="Cohere API key")
    MISTRAL_API_KEY: str = Field(default="", description="Mistral AI API key")
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    TOGETHER_API_KEY: str = Field(default="", description="Together AI API key")
    FIREWORKS_API_KEY: str = Field(default="", description="Fireworks AI API key")
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek API key")

    # ==========================================================================
    # API Keys - Embeddings (SECRETS - .env only)
    # ==========================================================================
    VOYAGE_API_KEY: str = Field(default="", description="Voyage AI API key (Phase 29)")
    JINA_API_KEY: str = Field(default="", description="Jina AI API key")

    # ==========================================================================
    # API Keys - Other Services (SECRETS - .env only)
    # ==========================================================================
    ELEVENLABS_API_KEY: str = Field(default="", description="ElevenLabs TTS API key")
    SMALLEST_API_KEY: str = Field(default="", description="Smallest.ai TTS API key")
    SLACK_BOT_TOKEN: str = Field(default="", description="Slack Bot OAuth token")
    SLACK_SIGNING_SECRET: str = Field(default="", description="Slack signing secret")

    # ==========================================================================
    # Database
    # ==========================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/aidocindexer",
        description="PostgreSQL connection URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # ==========================================================================
    # Storage
    # ==========================================================================
    UPLOAD_DIR: str = Field(default="./uploads", description="Directory for uploaded files")
    AUDIO_OUTPUT_DIR: str = Field(default="./audio_output", description="Directory for generated audio files")
    TEMP_DIR: str = Field(default="./tmp", description="Temporary files directory")

    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", description="Default LLM provider")
    DEFAULT_CHAT_MODEL: str = Field(default="gpt-4o", description="Default chat model")
    DEFAULT_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Default embedding model")

    # ==========================================================================
    # RAG Configuration
    # ==========================================================================
    RAG_TOP_K: int = Field(default=10, description="Number of documents to retrieve")
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.55, description="Minimum similarity score")
    ENABLE_QUERY_EXPANSION: bool = Field(default=True, description="Enable query expansion")
    ENABLE_VERIFICATION: bool = Field(default=True, description="Enable response verification")
    ENABLE_HYDE: bool = Field(default=True, description="Enable HyDE for retrieval")
    ENABLE_CRAG: bool = Field(default=True, description="Enable Corrective RAG")
    ENABLE_SELF_RAG: bool = Field(default=True, description="Enable SELF-RAG for hallucination detection")
    SELF_RAG_MIN_SUPPORTED_RATIO: float = Field(default=0.7, description="Min ratio of supported claims for SELF-RAG")
    ENABLE_LIGHTRAG: bool = Field(default=True, description="Enable LightRAG dual-level retrieval")
    ENABLE_RAPTOR: bool = Field(default=True, description="Enable RAPTOR tree-organized retrieval")

    # ==========================================================================
    # Audio Configuration
    # ==========================================================================
    TTS_DEFAULT_PROVIDER: str = Field(default="openai", description="Default TTS provider (openai, elevenlabs, local)")
    TTS_DEFAULT_VOICE: str = Field(default="alloy", description="Default TTS voice")
    AUDIO_FORMAT: str = Field(default="mp3", description="Default audio format")
    AUDIO_SAMPLE_RATE: int = Field(default=24000, description="Audio sample rate in Hz")

    # ==========================================================================
    # Security
    # ==========================================================================
    SECRET_KEY: str = Field(default="change-me-in-production", description="Secret key for JWT tokens")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration time")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration time")

    # ==========================================================================
    # External Integrations
    # ==========================================================================
    GOOGLE_OAUTH_CLIENT_ID: str = Field(default="", description="Google OAuth client ID")
    GOOGLE_OAUTH_CLIENT_SECRET: str = Field(default="", description="Google OAuth client secret")
    NOTION_INTEGRATION_TOKEN: str = Field(default="", description="Notion integration token")

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    ENABLE_AUDIO_OVERVIEWS: bool = Field(default=True, description="Enable audio overview generation")
    ENABLE_WORKFLOW_ENGINE: bool = Field(default=True, description="Enable workflow automation")
    ENABLE_CONNECTORS: bool = Field(default=True, description="Enable external data connectors")
    ENABLE_LLM_GATEWAY: bool = Field(default=True, description="Enable LLM gateway with budget control")

    # ==========================================================================
    # Celery / Task Queue Configuration
    # ==========================================================================
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL (Redis)"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL"
    )
    CELERY_WORKER_CONCURRENCY: int = Field(default=4, description="Celery worker concurrency")
    BULK_UPLOAD_MAX_CONCURRENT: int = Field(default=4, description="Max concurrent docs in bulk upload")
    BULK_UPLOAD_BATCH_SIZE: int = Field(default=100, description="Files per batch in bulk upload")

    # ==========================================================================
    # Performance Optimization
    # ==========================================================================
    EMBEDDING_BATCH_SIZE: int = Field(default=100, description="Batch size for embedding generation")
    EMBEDDING_CACHE_TTL: int = Field(default=604800, description="Embedding cache TTL in seconds (7 days)")
    ENABLE_EMBEDDING_CACHE: bool = Field(default=True, description="Enable Redis embedding cache")
    ENABLE_GPU_EMBEDDINGS: bool = Field(default=False, description="Enable GPU for embedding generation")
    KG_EXTRACTION_CONCURRENCY: int = Field(default=4, description="Max concurrent KG extractions")
    KG_PRE_FILTER_ENABLED: bool = Field(default=True, description="Enable KG pre-filtering to skip irrelevant docs")

    # ==========================================================================
    # Knowledge Graph Settings - Phase 60
    # ==========================================================================
    # Core KG settings
    KG_ENABLED: bool = Field(default=True, description="Enable Knowledge Graph features globally")
    KG_PROVIDER: str = Field(default="memory", description="KG storage backend: memory, neo4j, or networkx")

    # GraphRAG enhancements
    KG_USE_GRAPHRAG_ENHANCEMENTS: bool = Field(default=True, description="Enable GraphRAG optimizations")
    KG_USE_KGGEN_EXTRACTOR: bool = Field(default=True, description="Use KGGen multi-stage extraction (94% quality, 5x cheaper)")
    KG_USE_COMMUNITY_DETECTION: bool = Field(default=True, description="Enable Leiden community detection")
    KG_USE_ENTITY_STANDARDIZATION: bool = Field(default=True, description="Dedupe and normalize entities")

    # Extraction settings
    KG_BATCH_SIZE: int = Field(default=4, description="Chunks per batch for KG extraction (2-8 based on model)")
    KG_BATCH_EXTRACTION_DEFAULT: bool = Field(default=True, description="Use batch extraction by default for performance")
    KG_EXTRACTION_MODEL: str = Field(default="gpt-4o-mini", description="Model for entity extraction")

    # Query integration settings
    KG_ENABLED_IN_CHAT: bool = Field(default=True, description="Use KG context in chat queries")
    KG_ENABLED_IN_AUDIO: bool = Field(default=True, description="Use KG for audio section selection")
    KG_QUERY_EXPANSION_ENABLED: bool = Field(default=True, description="Expand queries with KG entities")
    KG_EXPANSION_MAX_ENTITIES: int = Field(default=5, description="Max entities for query expansion")
    KG_MAX_HOPS: int = Field(default=2, description="Max hops for multi-hop reasoning queries")
    KG_WEIGHT_IN_FUSION: float = Field(default=0.3, description="KG weight in hybrid retrieval fusion (0-1)")
    KG_AUDIO_WEIGHT: float = Field(default=0.2, description="KG entity weight in audio section scoring")

    # Fallback settings
    KG_LLM_FALLBACK_ENABLED: bool = Field(default=True, description="Enable multi-provider LLM fallback")
    KG_LLM_FALLBACK_CHAIN: str = Field(
        default="openai,anthropic,groq,ollama",
        description="LLM provider fallback order (comma-separated)"
    )
    KG_ENABLE_RULE_BASED_FALLBACK: bool = Field(default=True, description="Use rule-based extraction if all LLMs fail")

    # Performance settings
    USE_RAY_FOR_KG: bool = Field(default=True, description="Use Ray for distributed KG extraction")
    KG_CACHE_TTL: int = Field(default=3600, description="KG query cache TTL in seconds")
    KG_MIN_CONFIDENCE: float = Field(default=0.5, description="Minimum confidence threshold for entity inclusion")

    # ==========================================================================
    # Advanced AI Features
    # ==========================================================================
    ENABLE_CONTEXTUAL_RETRIEVAL: bool = Field(default=False, description="Enable Anthropic contextual retrieval")
    CONTEXTUAL_MODEL: str = Field(default="claude-3-5-haiku-latest", description="Model for context generation")
    CONTEXTUAL_CACHE_TTL_DAYS: int = Field(default=30, description="Context cache TTL in days")
    ENABLE_COLBERT_RETRIEVAL: bool = Field(default=False, description="Enable ColBERT PLAID retrieval")
    COLBERT_INDEX_PATH: str = Field(default="./data/colbert_index", description="ColBERT index storage path")
    ENABLE_HYBRID_SEARCH: bool = Field(default=True, description="Enable hybrid dense+sparse search")
    ENABLE_RERANKING: bool = Field(default=True, description="Enable cross-encoder reranking")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-base", description="Reranker model")

    # ==========================================================================
    # Vision & Document Processing
    # ==========================================================================
    ENABLE_VISION_PROCESSING: bool = Field(default=False, description="Enable vision model for scanned docs")
    VISION_MODEL: str = Field(default="claude-3-5-sonnet-20241022", description="Vision model for document analysis")
    PDF_PARSER: str = Field(default="auto", description="PDF parser (auto, marker, docling, pymupdf)")

    # ==========================================================================
    # External Services
    # ==========================================================================
    FIRECRAWL_API_KEY: str = Field(default="", description="Firecrawl API key for web scraping")
    CARTESIA_API_KEY: str = Field(default="", description="Cartesia API key for streaming TTS")
    E2B_API_KEY: str = Field(default="", description="E2B API key for code execution sandbox")

    # ==========================================================================
    # RLM (Recursive Language Model) Configuration - Phase 36/51
    # ==========================================================================
    ENABLE_RLM: bool = Field(default=True, description="Enable RLM for large context queries")
    RLM_THRESHOLD_TOKENS: int = Field(default=100000, description="Token threshold to trigger RLM (100K default)")
    RLM_SANDBOX: str = Field(default="auto", description="RLM sandbox type (auto, local, docker, modal, prime)")
    RLM_ROOT_MODEL: str = Field(default="gpt-4o", description="RLM root model for reasoning")
    RLM_RECURSIVE_MODEL: str = Field(default="gpt-4o-mini", description="RLM recursive model for chunks")
    RLM_MAX_ITERATIONS: int = Field(default=20, description="Max RLM REPL iterations")
    RLM_TIMEOUT_SECONDS: float = Field(default=120.0, description="RLM execution timeout")
    RLM_LOG_TRAJECTORY: bool = Field(default=False, description="Enable RLM trajectory logging")
    PRIME_API_KEY: str = Field(default="", description="Prime Intellect API key for RLM sandbox")

    # ==========================================================================
    # Ray Distributed Computing - Phase 37
    # ==========================================================================
    RAY_ADDRESS: str = Field(default="auto", description="Ray cluster address (auto for local)")
    RAY_NUM_WORKERS: int = Field(default=8, description="Number of Ray workers for actor pools")
    USE_RAY_FOR_EMBEDDINGS: bool = Field(default=True, description="Use Ray for embedding generation")
    USE_RAY_FOR_KG: bool = Field(default=True, description="Use Ray for KG extraction")
    USE_RAY_FOR_VLM: bool = Field(default=True, description="Use Ray for VLM processing")

    # ==========================================================================
    # VLM (Vision Language Model) Configuration - Phase 44/51
    # ==========================================================================
    ENABLE_VLM: bool = Field(default=True, description="Enable VLM for visual document processing")
    VLM_MODEL: str = Field(default="claude", description="VLM provider (claude, openai, qwen, local)")
    VLM_QWEN_MODEL: str = Field(default="Qwen/Qwen3-VL-7B-Instruct", description="Qwen VL model name")
    VLM_MAX_IMAGES: int = Field(default=10, description="Max images per VLM request")

    # ==========================================================================
    # Advanced Retrieval Configuration - Phase 51/62
    # ==========================================================================
    ENABLE_WARP: bool = Field(default=False, description="Enable WARP retriever (3x faster than ColBERT)")
    ENABLE_COLPALI: bool = Field(default=False, description="Enable ColPali for visual document retrieval")
    ENABLE_LIGHTRAG: bool = Field(default=True, description="Enable LightRAG dual-level retrieval")
    ENABLE_RAPTOR: bool = Field(default=True, description="Enable RAPTOR tree-organized retrieval")

    # ==========================================================================
    # Advanced Reranking - Phase 62
    # ==========================================================================
    ENABLE_ADVANCED_RERANKER: bool = Field(default=False, description="Enable multi-stage reranking with BM25")
    ENABLE_BM25_PREFILTER: bool = Field(default=True, description="Enable BM25 pre-filtering stage")

    # ==========================================================================
    # Reasoning Services - Phase 62
    # ==========================================================================
    ENABLE_TREE_OF_THOUGHTS: bool = Field(default=False, description="Enable Tree of Thoughts for complex queries")
    TOT_MAX_DEPTH: int = Field(default=3, description="Maximum depth for Tree of Thoughts exploration")
    TOT_BRANCHING_FACTOR: int = Field(default=3, description="Branching factor for Tree of Thoughts")
    TOT_SEARCH_STRATEGY: str = Field(default="beam", description="ToT search strategy (beam, dfs, bfs, mcts)")

    # ==========================================================================
    # Document Processing - Phase 62
    # ==========================================================================
    ENABLE_VISION_PROCESSOR: bool = Field(default=False, description="Enable vision document processor for scanned docs")
    VISION_OCR_ENGINE: str = Field(default="tesseract", description="OCR engine (tesseract, claude, surya)")
    ENABLE_CONTEXTUAL_EMBEDDINGS: bool = Field(default=True, description="Enable Anthropic contextual embeddings (67% error reduction)")

    # ==========================================================================
    # Answer Quality - Phase 62
    # ==========================================================================
    ENABLE_ANSWER_REFINER: bool = Field(default=False, description="Enable answer refinement (Self-Refine/CRITIC)")
    ANSWER_REFINER_STRATEGY: str = Field(default="self_refine", description="Refinement strategy (self_refine, critic, cove)")
    ANSWER_REFINER_MAX_ITERATIONS: int = Field(default=2, description="Max refinement iterations")

    # ==========================================================================
    # Additional Service Feature Flags - Phase 63
    # ==========================================================================
    ENABLE_AGENT_EVALUATION: bool = Field(default=False, description="Enable agent evaluation metrics (Pass^k, hallucination detection)")
    ENABLE_FAST_CHUNKING: bool = Field(default=False, description="Enable Chonkie fast chunking (33x faster than LangChain)")
    ENABLE_DOCLING_PARSER: bool = Field(default=False, description="Enable Docling enterprise parser (97.9% table accuracy)")
    ENABLE_SUFFICIENCY_CHECKER: bool = Field(default=False, description="Enable RAG context sufficiency detection (ICLR 2025)")
    ENABLE_TTT_COMPRESSION: bool = Field(default=False, description="Enable TTT context compression for long contexts")
    MAX_CONTEXT_LENGTH: int = Field(default=100000, description="Max context length before compression triggers")

    # ==========================================================================
    # Adaptive RAG Routing - Phase 66
    # ==========================================================================
    ENABLE_ADAPTIVE_ROUTING: bool = Field(default=True, description="Enable intelligent query routing to optimal strategies")
    ENABLE_RAG_FUSION: bool = Field(default=True, description="Enable RAG-Fusion multi-query with RRF")
    ENABLE_CONTEXT_COMPRESSION: bool = Field(default=True, description="Enable context compression for token efficiency")
    ENABLE_STEPBACK_PROMPTING: bool = Field(default=True, description="Enable step-back prompting for complex queries")
    ENABLE_CONTEXT_REORDERING: bool = Field(default=True, description="Enable context reordering to mitigate lost-in-the-middle")
    ENABLE_PHASE65: bool = Field(default=True, description="Enable Phase 65 features (spell correction, semantic cache, LTR)")
    ENABLE_USER_PERSONALIZATION: bool = Field(default=True, description="Enable user personalization based on feedback and preferences")
    ENABLE_LAZY_GRAPHRAG: bool = Field(default=True, description="Enable LazyGraphRAG for cost-efficient knowledge graph retrieval (99% cost reduction)")

    # ==========================================================================
    # pgvector HNSW Index Optimization - Phase 57
    # ==========================================================================
    HNSW_EF_SEARCH: int = Field(default=40, description="HNSW ef_search parameter (higher = better recall, slower)")
    HNSW_EF_SEARCH_HIGH_PRECISION: int = Field(default=100, description="ef_search for high-precision queries")
    PGVECTOR_ITERATIVE_SCAN: str = Field(default="relaxed_order", description="pgvector iterative scan mode (off, strict_order, relaxed_order)")
    INDEX_BUILD_MAINTENANCE_WORK_MEM: str = Field(default="2GB", description="maintenance_work_mem for index builds")
    INDEX_BUILD_PARALLEL_WORKERS: int = Field(default=4, description="Parallel workers for index builds")

    # ==========================================================================
    # VLM Extended Configuration - Phase 54
    # ==========================================================================
    VLM_PROVIDER: str = Field(default="claude", description="VLM provider (claude, openai, qwen, ollama)")
    VLM_AUTO_PROCESS: bool = Field(default=True, description="Auto-process visual documents with VLM")
    VLM_EXTRACT_TABLES: bool = Field(default=True, description="Extract tables from visual documents")
    VLM_EXTRACT_CHARTS: bool = Field(default=True, description="Extract data from charts")
    VLM_OCR_FALLBACK: bool = Field(default=True, description="Use OCR as fallback for VLM")

    # ==========================================================================
    # RLM (Recursive Language Model) Configuration - Phase 54
    # ==========================================================================
    RLM_PROVIDER: str = Field(default="anthropic", description="RLM provider")
    RLM_THRESHOLD: int = Field(default=100000, description="Context threshold for RLM (tokens)")
    RLM_MAX_CONTEXT: int = Field(default=10000000, description="Max context tokens for RLM")
    RLM_SANDBOX: str = Field(default="local", description="RLM sandbox type (local, modal, e2b)")
    RLM_SELF_REFINE: bool = Field(default=True, description="Enable RLM self-refinement")

    # ==========================================================================
    # GenerativeCache Configuration - Phase 42
    # ==========================================================================
    ENABLE_GENERATIVE_CACHE: bool = Field(default=True, description="Enable GenerativeCache for semantic caching")
    CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.92, description="Similarity threshold for cache hits")
    CACHE_MAX_SIZE_MB: int = Field(default=500, description="Max cache size in MB")
    CACHE_TTL_HOURS: int = Field(default=24, description="Cache TTL in hours")

    # ==========================================================================
    # TieredReranker Configuration - Phase 43
    # ==========================================================================
    ENABLE_TIERED_RERANKING: bool = Field(default=True, description="Enable tiered reranking pipeline")
    RERANK_STAGE1_TOP_K: int = Field(default=100, description="Stage 1 (ColBERT) top-k")
    RERANK_STAGE2_TOP_K: int = Field(default=20, description="Stage 2 (Cross-Encoder) top-k")
    RERANK_USE_LLM_STAGE: bool = Field(default=False, description="Use LLM for final reranking stage")

    # ==========================================================================
    # Memory Services Configuration - Phase 40/48
    # ==========================================================================
    ENABLE_AGENT_MEMORY: bool = Field(default=True, description="Enable agent memory system")
    MEMORY_PROVIDER: str = Field(default="mem0", description="Memory provider (mem0, amem)")
    MEMORY_MAX_ENTRIES: int = Field(default=1000, description="Max memory entries per user")
    MEMORY_DECAY_RATE: float = Field(default=0.01, description="Memory decay rate per hour")

    # ==========================================================================
    # Context Compression Configuration - Phase 38
    # ==========================================================================
    ENABLE_CONTEXT_COMPRESSION: bool = Field(default=True, description="Enable context compression")
    COMPRESSION_MAX_RECENT_TURNS: int = Field(default=10, description="Keep last N turns uncompressed")
    COMPRESSION_TARGET_RATIO: float = Field(default=0.1, description="Target compression ratio")

    # ==========================================================================
    # Ultra-Fast TTS Configuration - Phase 45/66
    # ==========================================================================
    ULTRA_FAST_TTS_PROVIDER: str = Field(default="smallest", description="Ultra-fast TTS provider")
    ENABLE_ULTRA_FAST_TTS: bool = Field(default=True, description="Enable ultra-fast TTS")

    # Chatterbox TTS (Resemble AI open-source)
    CHATTERBOX_ENABLED: bool = Field(default=True, description="Enable Chatterbox TTS provider")
    CHATTERBOX_EXAGGERATION: float = Field(default=0.5, description="Emotional exaggeration (0.0-1.0)")
    CHATTERBOX_CFG_WEIGHT: float = Field(default=0.5, description="CFG weight for generation (0.0-1.0)")
    CHATTERBOX_API_URL: str = Field(default="", description="Optional hosted Chatterbox API URL")

    # CosyVoice2 TTS (Alibaba open-source)
    COSYVOICE_ENABLED: bool = Field(default=True, description="Enable CosyVoice2 TTS provider")
    COSYVOICE_MODEL_PATH: str = Field(default="", description="Path to CosyVoice model")

    # Fish Speech TTS
    FISH_SPEECH_ENABLED: bool = Field(default=True, description="Enable Fish Speech TTS provider")
    FISH_SPEECH_API_KEY: str = Field(default="", description="Fish Speech API key")

    # TTS Fallback Chain
    TTS_FALLBACK_CHAIN: str = Field(default="cosyvoice,chatterbox,fish_speech,openai", description="TTS provider fallback order")

    # ==========================================================================
    # Monitoring & Observability
    # ==========================================================================
    SENTRY_DSN: str = Field(default="", description="Sentry DSN for error tracking")
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1, description="Sentry traces sample rate (0.0-1.0)")
    SENTRY_PROFILES_SAMPLE_RATE: float = Field(default=0.1, description="Sentry profiles sample rate (0.0-1.0)")
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics endpoint")

    def model_post_init(self, __context: Any) -> None:
        """Validate security-critical settings after initialization."""
        if self.SECRET_KEY == "change-me-in-production":
            if self.ENVIRONMENT in ("production", "staging"):
                raise ValueError(
                    "CRITICAL: SECRET_KEY is set to the default value. "
                    "Set a secure random SECRET_KEY in your .env file before running in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )
            else:
                import warnings
                warnings.warn(
                    "SECRET_KEY is set to the default value. This is acceptable for development "
                    "but MUST be changed before deploying to production.",
                    UserWarning,
                    stacklevel=2,
                )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
