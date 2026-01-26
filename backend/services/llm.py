"""
AIDocumentIndexer - LLM Service (LangChain + LiteLLM)
======================================================

Unified LLM interface using LangChain with LiteLLM for provider flexibility.
Supports OpenAI, Ollama, Anthropic, and 100+ other providers.

Features:
- Environment variable configuration (fallback)
- Database-driven provider configuration
- Operation-level provider assignment
- Per-session LLM override
- Usage tracking and cost estimation
"""

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

# LiteLLM integration with LangChain
try:
    from langchain_litellm import ChatLiteLLM
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    ChatLiteLLM = None

# Fallback to OpenAI if LiteLLM not available
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = structlog.get_logger(__name__)


# =============================================================================
# Per-Provider Rate Limiter (Token Bucket)
# =============================================================================

import asyncio
import threading
from collections import defaultdict


class ProviderRateLimiter:
    """
    Per-provider rate limiter using token bucket algorithm.

    Prevents exceeding provider API rate limits and provides
    backpressure when limits are approached.
    """

    # Default rate limits per provider (requests per minute)
    DEFAULT_LIMITS = {
        "openai": 500,
        "anthropic": 60,
        "google": 60,
        "cohere": 100,
        "ollama": 10000,  # Local — effectively unlimited
        "groq": 30,
        "together": 60,
        "mistral": 60,
        "deepseek": 60,
        "fireworks": 100,
        "default": 60,
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: Dict[str, list] = defaultdict(list)
        self._enabled = True

    def _get_limit(self, provider: str) -> int:
        return self.DEFAULT_LIMITS.get(provider.lower(), self.DEFAULT_LIMITS["default"])

    def check(self, provider: str) -> bool:
        """Check if a request is allowed. Returns True if allowed."""
        if not self._enabled:
            return True

        provider = provider.lower()
        limit = self._get_limit(provider)
        now = time.time()
        window = 60.0  # 1 minute

        with self._lock:
            # Clean old entries
            cutoff = now - window
            self._requests[provider] = [t for t in self._requests[provider] if t > cutoff]

            if len(self._requests[provider]) >= limit:
                logger.warning(
                    "Rate limit reached for provider",
                    provider=provider,
                    limit=limit,
                    window_seconds=window,
                )
                return False

            self._requests[provider].append(now)
            return True

    def wait_if_needed(self, provider: str) -> float:
        """
        Check rate limit. If exceeded, return seconds to wait.
        Returns 0 if request is allowed.
        """
        if not self._enabled:
            return 0.0

        provider = provider.lower()
        limit = self._get_limit(provider)
        now = time.time()
        window = 60.0

        with self._lock:
            cutoff = now - window
            self._requests[provider] = [t for t in self._requests[provider] if t > cutoff]

            if len(self._requests[provider]) >= limit:
                # Calculate wait time until oldest request expires
                oldest = min(self._requests[provider])
                wait = (oldest + window) - now
                return max(0.1, wait)

            self._requests[provider].append(now)
            return 0.0


# Global rate limiter instance
_provider_rate_limiter = ProviderRateLimiter()


# =============================================================================
# Configuration
# =============================================================================

class LLMConfig:
    """LLM configuration from environment variables."""

    def __init__(self):
        # Provider settings - detect available provider if not explicitly set
        self.default_provider = self._resolve_default_provider()

        # OpenAI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        # Ollama settings
        self.ollama_enabled = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_base_url = self.ollama_host  # Alias for embeddings service
        self.ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        # Anthropic settings
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        # Default settings
        self.default_chat_model = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o")
        self.default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        self.default_max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))

    def _resolve_default_provider(self) -> str:
        """
        Resolve the default LLM provider with smart detection.

        Priority:
        1. Explicitly set DEFAULT_LLM_PROVIDER env var
        2. Ollama if enabled and reachable (free, local)
        3. OpenAI if API key is set
        4. Anthropic if API key is set
        5. Ollama as final fallback (assumes user will set it up)
        """
        # If explicitly set, use that
        explicit_provider = os.getenv("DEFAULT_LLM_PROVIDER")
        if explicit_provider:
            return explicit_provider

        # Check Ollama first (free, local)
        ollama_enabled = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
        if ollama_enabled:
            # Return ollama - it will fail gracefully if not running
            return "ollama"

        # Check for valid API keys
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and not openai_key.startswith("sk-your-"):
            return "openai"

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key and not anthropic_key.startswith("sk-"):
            return "anthropic"

        # Default to ollama (user should set up a provider)
        return "ollama"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLMConfig from environment variables."""
        return cls()


# Global config
llm_config = LLMConfig()


# =============================================================================
# LLM Factory
# =============================================================================

class LLMFactory:
    """
    Factory for creating LLM instances.

    Supports multiple providers through LiteLLM abstraction.
    """

    _instances: Dict[str, BaseChatModel] = {}
    _embedding_instances: Dict[str, Embeddings] = {}

    @classmethod
    def get_chat_model(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> BaseChatModel:
        """
        Get a chat model instance.

        Args:
            provider: LLM provider (openai, ollama, anthropic, etc.)
            model: Model name (e.g., gpt-4o, llama3.2, claude-3-5-sonnet)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional model-specific parameters

        Returns:
            BaseChatModel: LangChain chat model instance
        """
        provider = provider or llm_config.default_provider
        # Use provider-specific default models
        if model is None:
            if provider == "ollama":
                model = llm_config.ollama_chat_model
            elif provider == "anthropic":
                model = llm_config.anthropic_model
            else:
                model = llm_config.default_chat_model
        temperature = temperature if temperature is not None else llm_config.default_temperature
        max_tokens = max_tokens or llm_config.default_max_tokens

        cache_key = f"{provider}:{model}:{temperature}:{max_tokens}"

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls._create_chat_model(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            logger.info(
                "Created chat model",
                provider=provider,
                model=model,
                temperature=temperature,
            )

        return cls._instances[cache_key]

    @classmethod
    def _create_chat_model(
        cls,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> BaseChatModel:
        """Create a new chat model instance."""

        # Use LiteLLM if available (recommended)
        if HAS_LITELLM:
            return cls._create_litellm_model(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        # Fallback to native LangChain integrations
        return cls._create_native_model(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @classmethod
    def _create_litellm_model(
        cls,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> BaseChatModel:
        """Create model using LiteLLM."""

        # Build LiteLLM model string
        if provider == "openai":
            litellm_model = model  # e.g., "gpt-4o"
        elif provider == "ollama":
            litellm_model = f"ollama/{model}"  # e.g., "ollama/llama3.2"
        elif provider == "anthropic":
            litellm_model = f"anthropic/{model}"  # e.g., "anthropic/claude-3-5-sonnet"
        elif provider == "vllm":
            # Phase 68: vLLM integration - 2-4x faster inference
            # Route through OpenAI-compatible API
            litellm_model = f"openai/{model}"
        else:
            litellm_model = f"{provider}/{model}"

        # Additional params for Ollama
        extra_params = {}
        if provider == "ollama":
            extra_params["api_base"] = llm_config.ollama_host
        elif provider == "vllm":
            # Phase 68: vLLM uses OpenAI-compatible API
            vllm_api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
            extra_params["api_base"] = vllm_api_base
            extra_params["api_key"] = os.getenv("VLLM_API_KEY", "dummy")

        return ChatLiteLLM(
            model=litellm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
            **kwargs,
        )

    @classmethod
    def _create_native_model(
        cls,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> BaseChatModel:
        """Create model using native LangChain integrations."""

        if provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=llm_config.openai_api_key,
                **kwargs,
            )

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=llm_config.ollama_host,
                **kwargs,
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=llm_config.anthropic_api_key,
                **kwargs,
            )

        elif provider == "vllm":
            # Phase 68: vLLM integration - use OpenAI-compatible interface
            vllm_api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=os.getenv("VLLM_API_KEY", "dummy"),
                openai_api_base=vllm_api_base,
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported provider without LiteLLM: {provider}")

    @classmethod
    def get_embeddings(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Embeddings:
        """
        Get an embeddings model instance.

        Args:
            provider: Embeddings provider
            model: Model name

        Returns:
            Embeddings: LangChain embeddings instance
        """
        provider = provider or llm_config.default_provider
        # Use provider-specific default models
        if model is None:
            if provider == "ollama":
                model = llm_config.ollama_embedding_model
            else:
                model = llm_config.default_embedding_model

        cache_key = f"embed:{provider}:{model}"

        if cache_key not in cls._embedding_instances:
            cls._embedding_instances[cache_key] = cls._create_embeddings(
                provider=provider,
                model=model,
            )
            logger.info(
                "Created embeddings model",
                provider=provider,
                model=model,
            )

        return cls._embedding_instances[cache_key]

    @classmethod
    def _create_embeddings(cls, provider: str, model: str) -> Embeddings:
        """Create embeddings model instance."""

        if provider == "openai":
            return OpenAIEmbeddings(
                model=model,
                api_key=llm_config.openai_api_key,
            )

        elif provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                model=model,
                base_url=llm_config.ollama_host,
            )

        else:
            # Default to OpenAI
            return OpenAIEmbeddings(
                model=llm_config.openai_embedding_model,
                api_key=llm_config.openai_api_key,
            )

    @classmethod
    def get_structured_model(
        cls,
        response_schema: dict,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> BaseChatModel:
        """
        Phase 82: Get a chat model configured for structured outputs (JSON schema).

        Uses OpenAI's response_format with json_schema for reliable structured extraction.
        Falls back to regular model with JSON instruction for non-OpenAI providers.

        Args:
            response_schema: JSON schema dict describing the expected output structure
            provider: LLM provider (openai recommended for native support)
            model: Model name
            temperature: Sampling temperature (default 0 for structured extraction)

        Returns:
            BaseChatModel configured for structured output
        """
        provider = provider or llm_config.default_provider
        if model is None:
            if provider == "openai":
                model = llm_config.default_chat_model
            elif provider == "anthropic":
                model = llm_config.anthropic_model
            else:
                model = llm_config.default_chat_model
        temperature = temperature if temperature is not None else 0.0

        if provider == "openai":
            # Native structured outputs via response_format
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=llm_config.default_max_tokens,
                api_key=llm_config.openai_api_key,
                model_kwargs={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_schema.get("title", "structured_output"),
                            "schema": response_schema,
                            "strict": True,
                        },
                    },
                },
            )
        else:
            # For non-OpenAI providers, use JSON mode instruction
            return cls.get_chat_model(
                provider=provider,
                model=model,
                temperature=temperature,
            )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances."""
        cls._instances.clear()
        cls._embedding_instances.clear()
        logger.info("LLM cache cleared")


# =============================================================================
# LLM Configuration Manager (Database-driven)
# =============================================================================

@dataclass
class LLMConfigResult:
    """Result from LLMConfigManager containing all config needed to create a model."""
    provider_type: str
    model: str
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    provider_id: Optional[str] = None  # Database provider ID for tracking
    source: str = "env"  # env, db_default, db_operation, db_session


class LLMConfigManager:
    """
    Unified LLM configuration manager with database integration.

    Priority order for configuration:
    1. Per-session override (if session_id provided)
    2. Operation-specific config (from db)
    3. Default provider (from db, is_default=True)
    4. Environment variables (fallback)
    """

    # Configuration cache with TTL (5 minutes default, reduces DB queries by ~60%)
    _cache: Dict[str, Tuple[Any, datetime]] = {}
    _cache_ttl_seconds: int = int(os.getenv("LLM_CONFIG_CACHE_TTL_SECONDS", "300"))

    # Environment variable mapping for API keys
    ENV_KEY_MAP = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "cohere": "COHERE_API_KEY",
    }

    @classmethod
    async def get_config_for_operation(
        cls,
        operation: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> LLMConfigResult:
        """
        Get LLM configuration for a specific operation.

        Args:
            operation: Operation type (chat, embeddings, document_processing, rag)
            session_id: Optional chat session ID for per-session override
            user_id: Optional user ID for user-specific settings (future)

        Returns:
            LLMConfigResult with all configuration needed to create the model
        """
        # 1. Check session override first
        if session_id:
            session_config = await cls._get_session_override(session_id)
            if session_config:
                return session_config

        # 2. Check operation-specific config
        operation_config = await cls._get_operation_config(operation)
        if operation_config:
            return operation_config

        # 3. Check default provider from DB
        default_config = await cls._get_default_provider()
        if default_config:
            return default_config

        # 4. Fall back to environment variables
        return cls._get_env_config(operation)

    @classmethod
    async def _get_session_override(cls, session_id: str) -> Optional[LLMConfigResult]:
        """Get per-session LLM override if exists."""
        cache_key = f"session:{session_id}"
        cached = cls._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            from backend.db.database import async_session_context
            from backend.db.models import ChatSessionLLMOverride
            from sqlalchemy import select

            async with async_session_context() as db:
                result = await db.execute(
                    select(ChatSessionLLMOverride)
                    .where(ChatSessionLLMOverride.session_id == session_id)
                )
                override = result.scalar_one_or_none()

                if override and override.provider:
                    config = await cls._build_config_from_provider(
                        override.provider,
                        model_override=override.model_override,
                        temperature_override=override.temperature_override,
                        source="db_session",
                    )
                    cls._set_cache(cache_key, config)
                    return config

        except (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning("Failed to get session override", session_id=session_id, error=str(e), error_type=type(e).__name__)

        cls._set_cache(cache_key, None)
        return None

    @classmethod
    async def _get_operation_config(cls, operation: str) -> Optional[LLMConfigResult]:
        """Get operation-specific LLM configuration."""
        cache_key = f"operation:{operation}"
        cached = cls._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            from backend.db.database import async_session_context
            from backend.db.models import LLMOperationConfig
            from sqlalchemy import select

            async with async_session_context() as db:
                result = await db.execute(
                    select(LLMOperationConfig)
                    .where(LLMOperationConfig.operation_type == operation)
                )
                op_config = result.scalar_one_or_none()

                if op_config and op_config.provider:
                    config = await cls._build_config_from_provider(
                        op_config.provider,
                        model_override=op_config.model_override,
                        temperature_override=op_config.temperature_override,
                        max_tokens_override=op_config.max_tokens_override,
                        source="db_operation",
                    )
                    cls._set_cache(cache_key, config)
                    return config

        except (ValueError, RuntimeError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning("Failed to get operation config", operation=operation, error=str(e), error_type=type(e).__name__)

        cls._set_cache(cache_key, None)
        return None

    @classmethod
    async def _get_default_provider(cls) -> Optional[LLMConfigResult]:
        """Get default LLM provider from database."""
        cache_key = "default_provider"
        cached = cls._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            from backend.db.database import async_session_context
            from backend.db.models import LLMProvider
            from sqlalchemy import select

            async with async_session_context() as db:
                result = await db.execute(
                    select(LLMProvider)
                    .where(LLMProvider.is_default == True)
                    .where(LLMProvider.is_active == True)
                )
                provider = result.scalar_one_or_none()

                if provider:
                    config = await cls._build_config_from_provider(
                        provider,
                        source="db_default",
                    )
                    cls._set_cache(cache_key, config)
                    return config

        except Exception as e:
            logger.warning("Failed to get default provider", error=str(e))

        cls._set_cache(cache_key, None)
        return None

    @classmethod
    async def get_config_for_provider_id(
        cls,
        provider_id: str,
    ) -> Optional[LLMConfigResult]:
        """
        Get LLM configuration for a specific provider by ID.

        Args:
            provider_id: Database provider ID (UUID string)

        Returns:
            LLMConfigResult if provider found and active, None otherwise
        """
        cache_key = f"provider:{provider_id}"
        cached = cls._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            from backend.db.database import async_session_context
            from backend.db.models import LLMProvider
            from sqlalchemy import select

            async with async_session_context() as db:
                result = await db.execute(
                    select(LLMProvider)
                    .where(LLMProvider.id == provider_id)
                    .where(LLMProvider.is_active == True)
                )
                provider = result.scalar_one_or_none()

                if provider:
                    config = await cls._build_config_from_provider(
                        provider,
                        source="db_explicit",
                    )
                    cls._set_cache(cache_key, config)
                    return config

        except Exception as e:
            logger.warning("Failed to get provider by ID", provider_id=provider_id, error=str(e))

        cls._set_cache(cache_key, None)
        return None

    @classmethod
    async def _build_config_from_provider(
        cls,
        provider,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
        source: str = "db",
    ) -> LLMConfigResult:
        """Build LLMConfigResult from database provider."""
        from backend.services.encryption import decrypt_value

        # Decrypt API key if stored in DB
        api_key = None
        if provider.api_key_encrypted:
            try:
                api_key = decrypt_value(provider.api_key_encrypted)
            except Exception as e:
                logger.warning("Failed to decrypt API key, falling back to env", error=str(e))

        # Fall back to environment variable if no DB key
        if not api_key:
            api_key = os.getenv(cls.ENV_KEY_MAP.get(provider.provider_type, ""), "")

        # Get settings from provider
        settings = provider.settings or {}

        return LLMConfigResult(
            provider_type=provider.provider_type,
            model=model_override or provider.default_chat_model or llm_config.default_chat_model,
            api_key=api_key or None,
            api_base_url=provider.api_base_url,
            temperature=temperature_override if temperature_override is not None else settings.get("temperature", llm_config.default_temperature),
            max_tokens=max_tokens_override or settings.get("max_tokens", llm_config.default_max_tokens),
            provider_id=str(provider.id),
            source=source,
        )

    @classmethod
    def _get_env_config(cls, operation: str = "chat") -> LLMConfigResult:
        """Get configuration from environment variables (fallback)."""
        provider = llm_config.default_provider

        # Get model based on operation
        if operation == "embeddings":
            model = llm_config.default_embedding_model
        else:
            model = llm_config.default_chat_model

        return LLMConfigResult(
            provider_type=provider,
            model=model,
            api_key=os.getenv(cls.ENV_KEY_MAP.get(provider, ""), ""),
            api_base_url=llm_config.ollama_host if provider == "ollama" else None,
            temperature=llm_config.default_temperature,
            max_tokens=llm_config.default_max_tokens,
            provider_id=None,
            source="env",
        )

    @classmethod
    def _get_from_cache(cls, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in cls._cache:
            value, expiry = cls._cache[key]
            if datetime.now() < expiry:
                return value
            del cls._cache[key]
        return None

    @classmethod
    def _set_cache(cls, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        expiry = datetime.now() + timedelta(seconds=cls._cache_ttl_seconds)
        cls._cache[key] = (value, expiry)

    @classmethod
    async def invalidate_cache(cls, key: Optional[str] = None) -> None:
        """
        Invalidate configuration cache.

        Args:
            key: Specific cache key to invalidate, or None to clear all
        """
        if key:
            cls._cache.pop(key, None)
            logger.info("LLM config cache invalidated", key=key)
        else:
            cls._cache.clear()
            logger.info("LLM config cache fully cleared")

    @classmethod
    async def invalidate_provider_cache(cls, provider_id: str) -> None:
        """Invalidate all cache entries related to a provider."""
        # Clear default provider cache (provider might be default)
        cls._cache.pop("default_provider", None)

        # Clear all operation caches (provider might be assigned to operations)
        keys_to_remove = [k for k in cls._cache.keys() if k.startswith("operation:")]
        for key in keys_to_remove:
            cls._cache.pop(key, None)

        logger.info("Provider-related cache invalidated", provider_id=provider_id)

    @classmethod
    async def get_config_with_health_check(
        cls,
        operation: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        enable_failover: bool = True,
    ) -> LLMConfigResult:
        """
        Get LLM configuration with health check and automatic failover.

        This extends get_config_for_operation by:
        1. Checking if the selected provider is healthy
        2. Automatically falling back to a healthy provider if not
        3. Using the operation's configured fallback provider if available

        Args:
            operation: Operation type (chat, embeddings, etc.)
            session_id: Optional chat session ID
            user_id: Optional user ID
            enable_failover: Whether to enable automatic failover

        Returns:
            LLMConfigResult with healthy provider configuration
        """
        # Get the initial config using normal resolution
        config = await cls.get_config_for_operation(
            operation=operation,
            session_id=session_id,
            user_id=user_id,
        )

        if not enable_failover or not config.provider_id:
            # Failover disabled or no provider ID (env-based config)
            return config

        # Check provider health
        try:
            from backend.services.provider_health import get_provider_health_checker
            from backend.db.database import async_session_context

            async with async_session_context() as db:
                health_checker = get_provider_health_checker()
                is_available = await health_checker.is_provider_available(config.provider_id)

                if is_available:
                    logger.debug(
                        "Provider health check passed",
                        provider_id=config.provider_id,
                        operation=operation,
                    )
                    return config

                # Provider is unhealthy - try failover
                logger.warning(
                    "Provider unhealthy, attempting failover",
                    provider_id=config.provider_id,
                    operation=operation,
                )

                # Try operation-specific fallback first
                fallback_config = await cls._get_operation_fallback(operation, db)
                if fallback_config:
                    fallback_available = await health_checker.is_provider_available(
                        fallback_config.provider_id
                    )
                    if fallback_available:
                        logger.info(
                            "Using operation fallback provider",
                            fallback_provider_id=fallback_config.provider_id,
                            operation=operation,
                        )
                        return fallback_config

                # Try any healthy provider
                healthy_config = await cls._get_healthy_provider(db, health_checker)
                if healthy_config:
                    logger.info(
                        "Using healthy fallback provider",
                        fallback_provider_id=healthy_config.provider_id,
                        operation=operation,
                    )
                    return healthy_config

                # All providers unhealthy - return original config anyway
                logger.error(
                    "All providers unhealthy, using original config",
                    provider_id=config.provider_id,
                    operation=operation,
                )
                return config

        except Exception as e:
            logger.warning(
                "Health check failed, using original config",
                error=str(e),
                operation=operation,
            )
            return config

    @classmethod
    async def _get_operation_fallback(
        cls,
        operation: str,
        db,
    ) -> Optional[LLMConfigResult]:
        """Get fallback provider for an operation if configured."""
        try:
            from backend.db.models import LLMOperationConfig
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            result = await db.execute(
                select(LLMOperationConfig)
                .where(LLMOperationConfig.operation_type == operation)
                .options(selectinload(LLMOperationConfig.fallback_provider))
            )
            op_config = result.scalar_one_or_none()

            if op_config and op_config.fallback_provider:
                return await cls._build_config_from_provider(
                    op_config.fallback_provider,
                    model_override=op_config.model_override,
                    temperature_override=op_config.temperature_override,
                    max_tokens_override=op_config.max_tokens_override,
                    source="db_fallback",
                )

        except Exception as e:
            logger.warning("Failed to get operation fallback", error=str(e))

        return None

    @classmethod
    async def _get_healthy_provider(
        cls,
        db,
        health_checker,
    ) -> Optional[LLMConfigResult]:
        """Get any healthy provider as fallback."""
        try:
            healthy_provider = await health_checker.get_healthy_provider_for_failover(db)
            if healthy_provider:
                from backend.db.models import LLMProvider
                from sqlalchemy import select

                result = await db.execute(
                    select(LLMProvider).where(LLMProvider.id == healthy_provider)
                )
                provider = result.scalar_one_or_none()

                if provider:
                    return await cls._build_config_from_provider(
                        provider,
                        source="db_failover",
                    )

        except Exception as e:
            logger.warning("Failed to get healthy provider", error=str(e))

        return None


# =============================================================================
# Enhanced LLM Factory with Database Support
# =============================================================================

class EnhancedLLMFactory:
    """
    Enhanced factory for creating LLM instances with database configuration
    and usage tracking support.
    """

    @classmethod
    async def get_chat_model_for_operation(
        cls,
        operation: str = "chat",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        track_usage: bool = True,
        enable_failover: bool = True,
        **kwargs,
    ) -> Tuple[BaseChatModel, LLMConfigResult]:
        """
        Get chat model for a specific operation with configuration resolution.

        Args:
            operation: Operation type (chat, rag, document_processing)
            session_id: Optional chat session ID for per-session override
            user_id: Optional user ID for tracking
            track_usage: Whether to enable usage tracking
            enable_failover: Whether to enable automatic failover on unhealthy providers
            **kwargs: Additional model parameters

        Returns:
            Tuple of (BaseChatModel, LLMConfigResult)
        """
        # Use health-checked config resolution if failover is enabled
        if enable_failover:
            config = await LLMConfigManager.get_config_with_health_check(
                operation=operation,
                session_id=session_id,
                user_id=user_id,
                enable_failover=True,
            )
        else:
            config = await LLMConfigManager.get_config_for_operation(
                operation=operation,
                session_id=session_id,
                user_id=user_id,
            )

        logger.info(
            "Resolved LLM config for operation",
            operation=operation,
            provider=config.provider_type,
            model=config.model,
            source=config.source,
        )

        # Create the model using the resolved config
        model = cls._create_model_from_config(config, **kwargs)

        return model, config

    @classmethod
    async def get_embeddings_for_operation(
        cls,
        operation: str = "embeddings",
        **kwargs,
    ) -> Tuple[Embeddings, LLMConfigResult]:
        """
        Get embeddings model for operation.

        Args:
            operation: Operation type (usually "embeddings")
            **kwargs: Additional parameters

        Returns:
            Tuple of (Embeddings, LLMConfigResult)
        """
        config = await LLMConfigManager.get_config_for_operation(
            operation=operation,
        )

        logger.info(
            "Resolved embeddings config",
            operation=operation,
            provider=config.provider_type,
            model=config.model,
            source=config.source,
        )

        # Create embeddings model
        embeddings = cls._create_embeddings_from_config(config, **kwargs)

        return embeddings, config

    @classmethod
    def _create_model_from_config(
        cls,
        config: LLMConfigResult,
        **kwargs,
    ) -> BaseChatModel:
        """Create model from LLMConfigResult."""
        # Build kwargs for model creation
        model_kwargs = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **kwargs,
        }

        if config.api_key:
            model_kwargs["api_key"] = config.api_key

        if config.api_base_url:
            if config.provider_type == "ollama":
                model_kwargs["base_url"] = config.api_base_url
            else:
                model_kwargs["api_base"] = config.api_base_url

        # Use LiteLLM if available
        if HAS_LITELLM:
            # Build LiteLLM model string
            if config.provider_type == "openai":
                litellm_model = config.model
            elif config.provider_type == "ollama":
                litellm_model = f"ollama/{config.model}"
            elif config.provider_type == "anthropic":
                litellm_model = f"anthropic/{config.model}"
            else:
                litellm_model = f"{config.provider_type}/{config.model}"

            return ChatLiteLLM(
                model=litellm_model,
                **model_kwargs,
            )

        # Fallback to native implementations
        if config.provider_type == "openai":
            return ChatOpenAI(
                model=config.model,
                **model_kwargs,
            )
        elif config.provider_type == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=config.model,
                temperature=config.temperature,
                base_url=config.api_base_url or llm_config.ollama_host,
            )
        elif config.provider_type == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported provider without LiteLLM: {config.provider_type}")

    @classmethod
    def _create_embeddings_from_config(
        cls,
        config: LLMConfigResult,
        **kwargs,
    ) -> Embeddings:
        """Create embeddings model from LLMConfigResult."""
        if config.provider_type == "openai":
            return OpenAIEmbeddings(
                model=config.model,
                api_key=config.api_key or llm_config.openai_api_key,
            )
        elif config.provider_type == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                model=config.model,
                base_url=config.api_base_url or llm_config.ollama_host,
            )
        else:
            # Default to OpenAI for embeddings
            return OpenAIEmbeddings(
                model=llm_config.default_embedding_model,
                api_key=config.api_key or llm_config.openai_api_key,
            )


# =============================================================================
# Usage Tracking Service
# =============================================================================

class LLMUsageTracker:
    """Service for tracking LLM usage and costs."""

    @classmethod
    async def log_usage(
        cls,
        provider_type: str,
        model: str,
        operation_type: str,
        input_tokens: int,
        output_tokens: int,
        provider_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log LLM usage to database.

        Args:
            provider_type: Provider type (openai, anthropic, etc.)
            model: Model name
            operation_type: Operation type (chat, embeddings, etc.)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider_id: Database provider ID
            user_id: User ID
            session_id: Chat session ID
            duration_ms: Request duration in milliseconds
            success: Whether the request succeeded
            error_message: Error message if failed

        Returns:
            Usage log ID or None if failed
        """
        try:
            from backend.db.database import async_session_context
            from backend.db.models import LLMUsageLog
            from backend.services.llm_pricing import LLMPricingService

            # Calculate costs
            costs = LLMPricingService.calculate_cost(
                provider_type=provider_type,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            async with async_session_context() as db:
                usage_log = LLMUsageLog(
                    provider_id=uuid.UUID(provider_id) if provider_id else None,
                    provider_type=provider_type,
                    model=model,
                    operation_type=operation_type,
                    user_id=uuid.UUID(user_id) if user_id else None,
                    session_id=uuid.UUID(session_id) if session_id else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    input_cost_usd=costs["input_cost_usd"],
                    output_cost_usd=costs["output_cost_usd"],
                    total_cost_usd=costs["total_cost_usd"],
                    request_duration_ms=duration_ms,
                    success=success,
                    error_message=error_message,
                )
                db.add(usage_log)
                await db.commit()

                logger.debug(
                    "Logged LLM usage",
                    model=model,
                    operation=operation_type,
                    tokens=input_tokens + output_tokens,
                    cost=costs["total_cost_usd"],
                )

                return str(usage_log.id)

        except Exception as e:
            logger.error("Failed to log LLM usage", error=str(e))
            return None

    @classmethod
    async def get_usage_summary(
        cls,
        provider_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get usage summary with aggregated stats.

        Args:
            provider_id: Filter by provider
            user_id: Filter by user
            operation_type: Filter by operation type
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dictionary with usage statistics
        """
        try:
            from backend.db.database import async_session_context
            from backend.db.models import LLMUsageLog
            from sqlalchemy import select, func

            async with async_session_context() as db:
                # Build base query
                query = select(
                    func.count(LLMUsageLog.id).label("request_count"),
                    func.sum(LLMUsageLog.input_tokens).label("total_input_tokens"),
                    func.sum(LLMUsageLog.output_tokens).label("total_output_tokens"),
                    func.sum(LLMUsageLog.total_tokens).label("total_tokens"),
                    func.sum(LLMUsageLog.total_cost_usd).label("total_cost_usd"),
                    func.avg(LLMUsageLog.request_duration_ms).label("avg_duration_ms"),
                )

                # Apply filters
                if provider_id:
                    query = query.where(LLMUsageLog.provider_id == provider_id)
                if user_id:
                    query = query.where(LLMUsageLog.user_id == user_id)
                if operation_type:
                    query = query.where(LLMUsageLog.operation_type == operation_type)
                if start_date:
                    query = query.where(LLMUsageLog.created_at >= start_date)
                if end_date:
                    query = query.where(LLMUsageLog.created_at <= end_date)

                result = await db.execute(query)
                row = result.one()

                return {
                    "request_count": row.request_count or 0,
                    "total_input_tokens": row.total_input_tokens or 0,
                    "total_output_tokens": row.total_output_tokens or 0,
                    "total_tokens": row.total_tokens or 0,
                    "total_cost_usd": float(row.total_cost_usd or 0),
                    "avg_duration_ms": float(row.avg_duration_ms or 0),
                }

        except Exception as e:
            logger.error("Failed to get usage summary", error=str(e))
            return {
                "request_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0,
                "avg_duration_ms": 0,
            }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """Get a chat model instance."""
    return LLMFactory.get_chat_model(provider=provider, model=model, **kwargs)


def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Embeddings:
    """Get an embeddings model instance."""
    return LLMFactory.get_embeddings(provider=provider, model=model)


async def generate_response(
    messages: List[BaseMessage],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate a response from the LLM.

    Args:
        messages: List of conversation messages
        provider: LLM provider
        model: Model name
        **kwargs: Additional parameters

    Returns:
        str: Generated response
    """
    effective_provider = provider or llm_config.default_provider
    wait = _provider_rate_limiter.wait_if_needed(effective_provider)
    if wait > 0:
        logger.info("Rate limit backpressure", provider=effective_provider, wait_seconds=round(wait, 1))
        await asyncio.sleep(wait)

    llm = get_chat_model(provider=provider, model=model, **kwargs)
    response = await llm.ainvoke(messages)
    return response.content


async def generate_embeddings(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        provider: Embeddings provider
        model: Model name

    Returns:
        List[List[float]]: List of embedding vectors
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return await embeddings.aembed_documents(texts)


async def generate_embedding(
    text: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        provider: Embeddings provider
        model: Model name

    Returns:
        List[float]: Embedding vector
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return await embeddings.aembed_query(text)


# =============================================================================
# Model Testing
# =============================================================================

async def test_llm_connection(provider: str = "openai") -> Dict[str, Any]:
    """
    Test LLM connection and return model info.

    Args:
        provider: Provider to test

    Returns:
        dict: Connection test results
    """
    try:
        llm = get_chat_model(provider=provider)
        response = await llm.ainvoke([
            HumanMessage(content="Say 'connection successful' in exactly those words.")
        ])

        return {
            "status": "success",
            "provider": provider,
            "response": response.content,
        }

    except Exception as e:
        logger.error("LLM connection test failed", provider=provider, error=str(e))
        return {
            "status": "error",
            "provider": provider,
            "error": str(e),
        }


async def test_embeddings_connection(provider: str = "openai") -> Dict[str, Any]:
    """
    Test embeddings connection.

    Args:
        provider: Provider to test

    Returns:
        dict: Connection test results
    """
    try:
        embeddings = get_embeddings(provider=provider)
        result = await embeddings.aembed_query("test")

        return {
            "status": "success",
            "provider": provider,
            "dimension": len(result),
        }

    except Exception as e:
        logger.error("Embeddings connection test failed", provider=provider, error=str(e))
        return {
            "status": "error",
            "provider": provider,
            "error": str(e),
        }


async def list_ollama_models(base_url: str = None) -> Dict[str, Any]:
    """
    List available models from local Ollama instance.

    Args:
        base_url: Optional Ollama API base URL. Defaults to config value.

    Returns:
        dict: Contains 'success', 'chat_models', 'embedding_models', and 'total'
    """
    import httpx

    ollama_url = base_url or llm_config.ollama_host

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ollama_url}/api/tags",
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                chat_models = []
                embedding_models = []

                for m in models:
                    name = m.get("name", "")
                    info = {
                        "name": name,
                        "size": m.get("size"),
                        "family": m.get("details", {}).get("family"),
                        "parameter_size": m.get("details", {}).get("parameter_size"),
                    }
                    if "embed" in name.lower():
                        embedding_models.append(info)
                    else:
                        chat_models.append(info)

                return {
                    "success": True,
                    "chat_models": chat_models,
                    "embedding_models": embedding_models,
                    "total": len(models),
                }
            return {
                "success": False,
                "error": f"Ollama returned {response.status_code}",
                "chat_models": [],
                "embedding_models": [],
            }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama",
            "chat_models": [],
            "embedding_models": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chat_models": [],
            "embedding_models": [],
        }


async def pull_ollama_model(model_name: str, base_url: str = None) -> Dict[str, Any]:
    """
    Pull (download) an Ollama model.

    Note: This can be a long-running operation for large models.
    With stream=False, Ollama waits for the download to complete.

    Args:
        model_name: Model name to pull (e.g., 'qwen2.5vl', 'llava:7b')
        base_url: Optional Ollama API base URL. Defaults to config value.

    Returns:
        dict: Contains 'success', 'message', and optionally 'model' or 'error'
    """
    import httpx

    ollama_url = base_url or llm_config.ollama_host

    logger.info("Pulling Ollama model", model_name=model_name, ollama_url=ollama_url)

    try:
        # Use a longer timeout for large model downloads (up to 30 minutes)
        async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
            response = await client.post(
                f"{ollama_url}/api/pull",
                json={"name": model_name, "stream": False},
            )

            if response.status_code == 200:
                logger.info("Successfully pulled Ollama model", model_name=model_name)
                return {
                    "success": True,
                    "message": f"Model '{model_name}' pulled successfully",
                    "model": model_name,
                }
            else:
                error_text = response.text
                logger.error("Failed to pull Ollama model", model_name=model_name, error=error_text)
                return {
                    "success": False,
                    "error": f"Pull failed: {error_text}",
                }
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama", ollama_url=ollama_url)
        return {"success": False, "error": "Cannot connect to Ollama. Is it running?"}
    except httpx.TimeoutException:
        logger.warning("Ollama pull timed out", model_name=model_name)
        return {
            "success": False,
            "error": "Pull timed out - model may still be downloading. Check Ollama logs.",
        }
    except Exception as e:
        logger.error("Error pulling Ollama model", model_name=model_name, error=str(e))
        return {"success": False, "error": str(e)}


async def delete_ollama_model(model_name: str, base_url: str = None) -> Dict[str, Any]:
    """
    Delete a local Ollama model.

    Args:
        model_name: Model name to delete (e.g., 'llama3.2:latest')
        base_url: Optional Ollama API base URL. Defaults to config value.

    Returns:
        dict: Contains 'success' and 'message' or 'error'
    """
    import httpx

    ollama_url = base_url or llm_config.ollama_host

    logger.info("Deleting Ollama model", model_name=model_name, ollama_url=ollama_url)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Ollama uses DELETE with JSON body for model deletion
            response = await client.request(
                "DELETE",
                f"{ollama_url}/api/delete",
                json={"name": model_name},
            )

            if response.status_code == 200:
                logger.info("Successfully deleted Ollama model", model_name=model_name)
                return {
                    "success": True,
                    "message": f"Model '{model_name}' deleted successfully",
                }
            elif response.status_code == 404:
                logger.warning("Ollama model not found", model_name=model_name)
                return {
                    "success": False,
                    "error": f"Model '{model_name}' not found",
                }
            else:
                error_text = response.text
                logger.error("Failed to delete Ollama model", model_name=model_name, error=error_text)
                return {
                    "success": False,
                    "error": f"Delete failed: {error_text}",
                }
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama", ollama_url=ollama_url)
        return {"success": False, "error": "Cannot connect to Ollama. Is it running?"}
    except Exception as e:
        logger.error("Error deleting Ollama model", model_name=model_name, error=str(e))
        return {"success": False, "error": str(e)}


# =============================================================================
# Vision Model Support
# =============================================================================

# Known vision model patterns
VISION_MODEL_PATTERNS = [
    # Ollama models
    "llava", "qwen2-vl", "qwen2.5-vl", "llama3.2-vision", "llama-3.2-vision",
    "moondream", "bakllava", "minicpm-v", "cogvlm", "yi-vl", "internlm-xcomposer",
    "phi-3-vision", "llava-llama3", "llava-phi3", "nanollava",
    # OpenAI models
    "gpt-4-vision", "gpt-4o", "gpt-4-turbo",
    # Anthropic models
    "claude-3", "claude-3.5",
    # Google models
    "gemini-1.5", "gemini-pro-vision",
]


def is_vision_model(model_name: str) -> bool:
    """Check if a model supports vision/image input."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in VISION_MODEL_PATTERNS)


def create_vision_message(
    text: str,
    image_data: bytes,
    image_type: str = "image/jpeg",
) -> HumanMessage:
    """
    Create a HumanMessage with image content for vision models.

    Args:
        text: Text prompt/question about the image
        image_data: Raw image bytes
        image_type: MIME type of the image (default: image/jpeg)

    Returns:
        HumanMessage with multimodal content
    """
    import base64

    image_b64 = base64.b64encode(image_data).decode("utf-8")

    # LangChain multimodal message format
    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_type};base64,{image_b64}",
                },
            },
        ]
    )


def create_vision_messages_from_urls(
    text: str,
    image_urls: List[str],
) -> HumanMessage:
    """
    Create a HumanMessage with multiple images from URLs.

    Args:
        text: Text prompt/question about the images
        image_urls: List of image URLs

    Returns:
        HumanMessage with multimodal content
    """
    content = [{"type": "text", "text": text}]

    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })

    return HumanMessage(content=content)


async def chat_with_vision(
    model: BaseChatModel,
    text: str,
    image_data: Optional[bytes] = None,
    image_url: Optional[str] = None,
    image_type: str = "image/jpeg",
    system_prompt: Optional[str] = None,
) -> str:
    """
    Send a vision request to a multimodal LLM.

    Args:
        model: LangChain chat model (must support vision)
        text: Text prompt/question
        image_data: Raw image bytes (mutually exclusive with image_url)
        image_url: URL to an image (mutually exclusive with image_data)
        image_type: MIME type of the image
        system_prompt: Optional system prompt

    Returns:
        Model's response as string
    """
    messages: List[BaseMessage] = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    if image_data:
        messages.append(create_vision_message(text, image_data, image_type))
    elif image_url:
        messages.append(create_vision_messages_from_urls(text, [image_url]))
    else:
        messages.append(HumanMessage(content=text))

    response = await model.ainvoke(messages)
    return response.content


async def analyze_image(
    image_data: bytes,
    prompt: str = "Describe this image in detail.",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Analyze an image using a vision model.

    Args:
        image_data: Raw image bytes
        prompt: Question or prompt about the image
        provider: LLM provider (defaults to config)
        model: Model name (defaults to provider's default vision model)

    Returns:
        Model's analysis as string
    """
    # Get appropriate vision model
    provider = provider or llm_config.default_provider
    if model is None:
        if provider == "ollama":
            model = "llava"  # Default vision model for Ollama
        elif provider == "openai":
            model = "gpt-4o"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        else:
            model = "gpt-4o"  # Fallback

    llm = LLMFactory.get_chat_model(
        provider=provider,
        model=model,
        temperature=0.3,  # Lower temperature for analysis
    )

    return await chat_with_vision(
        model=llm,
        text=prompt,
        image_data=image_data,
    )
