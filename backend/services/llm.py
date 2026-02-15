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
        self._lock = asyncio.Lock()
        self._requests: Dict[str, list] = defaultdict(list)
        self._enabled = True

    def _get_limit(self, provider: str) -> int:
        return self.DEFAULT_LIMITS.get(provider.lower(), self.DEFAULT_LIMITS["default"])

    async def check(self, provider: str) -> bool:
        """Check if a request is allowed. Returns True if allowed."""
        if not self._enabled:
            return True

        provider = provider.lower()
        limit = self._get_limit(provider)
        now = time.time()
        window = 60.0  # 1 minute

        async with self._lock:
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

    async def wait_if_needed(self, provider: str) -> float:
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

        async with self._lock:
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
# Settings-aware helpers (cached, sync-safe)
# =============================================================================

_vllm_settings_cache: Dict[str, Optional[str]] = {}
_auto_provider_cache: Dict[str, Optional[str]] = {}


def _get_vllm_setting(key: str) -> Optional[str]:
    """Get a vLLM/infrastructure setting from DB (cached, sync-safe).

    Uses sync DB session to avoid issues with running event loops in FastAPI.
    """
    if key in _vllm_settings_cache:
        return _vllm_settings_cache[key]

    try:
        from backend.db.database import get_sync_session
        from backend.db.models import SystemSettings
        from sqlalchemy import select
        session = get_sync_session()
        try:
            result = session.execute(
                select(SystemSettings.value).where(SystemSettings.key == key)
            )
            value = result.scalar_one_or_none()
            _vllm_settings_cache[key] = value
            return value
        finally:
            session.close()
    except Exception:
        return None


def invalidate_llm_infra_cache():
    """Invalidate cached LLM infrastructure settings."""
    _vllm_settings_cache.clear()
    _auto_provider_cache.clear()


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

        # Default settings — resolve model based on active provider
        _env_model = os.getenv("DEFAULT_CHAT_MODEL", "")
        if _env_model:
            self.default_chat_model = _env_model
        elif self.default_provider == "ollama":
            self.default_chat_model = self.ollama_chat_model
        elif self.default_provider == "anthropic":
            self.default_chat_model = self.anthropic_model
        else:
            self.default_chat_model = self.openai_chat_model
        self.default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        self.default_max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))

    # All valid provider types for inference backend
    VALID_PROVIDERS = frozenset({
        "ollama", "vllm", "openai", "anthropic", "azure", "google",
        "groq", "together", "cohere", "deepinfra", "bedrock", "custom", "auto",
    })

    def _resolve_default_provider(self) -> str:
        """
        Resolve the default LLM provider with smart detection.

        Priority:
        1. Settings DB (llm.inference_backend) — admin-configurable
           - "auto": use the default provider from LLMProvider DB table
           - explicit provider name: use that provider directly
        2. Explicitly set DEFAULT_LLM_PROVIDER env var (legacy)
        3. Ollama if enabled (free, local)
        4. OpenAI if API key is set
        5. Anthropic if API key is set
        6. Ollama as final fallback
        """
        # 1. Check settings DB first (admin-configurable)
        try:
            import asyncio
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            # Use sync wrapper since this is called in __init__
            try:
                loop = asyncio.get_running_loop()
                # Can't await in sync context with running loop — use cached value
            except RuntimeError:
                # No running loop — safe to run
                backend = asyncio.run(settings_svc.get_setting("llm.inference_backend"))
                if backend == "auto":
                    # Auto mode: use the default provider from LLMProvider DB table
                    resolved = self._resolve_auto_provider()
                    if resolved:
                        return resolved
                    # Fall through to env-based detection if no DB provider set
                elif backend and backend in self.VALID_PROVIDERS:
                    return backend
        except Exception:
            pass  # Settings not available yet (startup), fall through

        # 2. Explicitly set env var (legacy fallback)
        explicit_provider = os.getenv("DEFAULT_LLM_PROVIDER")
        if explicit_provider:
            return explicit_provider

        # 3. Check Ollama first (free, local)
        ollama_enabled = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
        if ollama_enabled:
            return "ollama"

        # 4. Check for valid API keys
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and not openai_key.startswith("sk-your-"):
            return "openai"

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key and not anthropic_key.startswith("sk-"):
            return "anthropic"

        # 5. Default to ollama
        return "ollama"

    @staticmethod
    def _resolve_auto_provider() -> Optional[str]:
        """
        Resolve provider from the default LLMProvider in the database.
        Returns provider_type string or None if no default provider is configured.
        Uses sync-safe pattern: returns None when event loop is running (FastAPI context).
        """
        if "default" in _auto_provider_cache:
            return _auto_provider_cache["default"]

        try:
            import asyncio
            try:
                asyncio.get_running_loop()
                # Running loop exists (FastAPI) — can't use asyncio.run(). Return None.
                return None
            except RuntimeError:
                pass  # No running loop — safe to run sync

            from sqlalchemy import select
            from backend.db.database import async_session_context
            from backend.db.models import LLMProvider

            async def _get_default():
                async with async_session_context() as session:
                    result = await session.execute(
                        select(LLMProvider.provider_type).where(
                            LLMProvider.is_default == True,
                            LLMProvider.is_active == True,
                        )
                    )
                    return result.scalar_one_or_none()

            resolved = asyncio.run(_get_default())
            _auto_provider_cache["default"] = resolved
            return resolved
        except Exception:
            return None

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

    Phase 98: Added LRU-style bounded caches to prevent memory leaks.
    Default max size is 50 instances per cache. When exceeded, oldest
    entries are evicted. Each LLM instance can hold significant memory.
    """

    _instances: Dict[str, BaseChatModel] = {}
    _embedding_instances: Dict[str, Embeddings] = {}
    _max_cache_size: int = int(os.getenv("LLM_MAX_CACHE_SIZE", "50"))

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached LLM instances. Called when provider settings change."""
        count = len(cls._instances) + len(cls._embedding_instances)
        cls._instances.clear()
        cls._embedding_instances.clear()
        _auto_provider_cache.clear()
        if count > 0:
            logger.info("Cleared LLM instance cache", evicted=count)

    _cache_lock = threading.Lock()

    @classmethod
    def _evict_oldest(cls, cache: Dict, max_size: int) -> None:
        """Evict oldest entries from cache if it exceeds max size (LRU-style)."""
        with cls._cache_lock:
            while len(cache) > max_size:
                # Python 3.7+ dicts maintain insertion order, so first key is oldest
                oldest_key = next(iter(cache))
                del cache[oldest_key]
                logger.debug("Evicted oldest LLM instance from cache", key=oldest_key)

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
            # Phase 98: Evict oldest entries to prevent unbounded memory growth
            cls._evict_oldest(cls._instances, cls._max_cache_size)
            logger.info(
                "Created chat model",
                provider=provider,
                model=model,
                temperature=temperature,
                cache_size=len(cls._instances),
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

        # Use native LangChain for Ollama — litellm has timeout issues with local Ollama
        if provider == "ollama":
            return cls._create_native_model(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        # Use LiteLLM for cloud providers (recommended)
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

    # Default API base URLs for OpenAI-compatible cloud providers
    PROVIDER_API_BASES = {
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
        "deepinfra": "https://api.deepinfra.com/v1/openai",
    }

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

        # Extract api_key and api_base from kwargs (passed from DB provider config)
        api_key = kwargs.pop("api_key", None)
        api_base = kwargs.pop("api_base", None)

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
        elif provider == "bedrock":
            litellm_model = f"bedrock/{model}"
        elif provider in ("groq", "together", "deepinfra", "custom"):
            # OpenAI-compatible providers — route through openai prefix for LiteLLM
            litellm_model = f"openai/{model}"
        else:
            litellm_model = f"{provider}/{model}"

        # Build extra params based on provider
        extra_params = {}
        if provider == "ollama":
            extra_params["api_base"] = llm_config.ollama_host
        elif provider == "vllm":
            # Phase 68: vLLM uses OpenAI-compatible API
            # Settings DB takes priority over env vars
            vllm_api_base = _get_vllm_setting("llm.vllm_api_base") or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
            vllm_api_key = _get_vllm_setting("llm.vllm_api_key") or os.getenv("VLLM_API_KEY", "dummy")
            extra_params["api_base"] = api_base or vllm_api_base
            extra_params["api_key"] = api_key or vllm_api_key
        elif provider in cls.PROVIDER_API_BASES:
            # OpenAI-compatible cloud providers — set api_base and api_key
            extra_params["api_base"] = api_base or cls.PROVIDER_API_BASES[provider]
            if api_key:
                extra_params["api_key"] = api_key
        elif provider == "custom":
            # Custom provider needs api_base
            if api_base:
                extra_params["api_base"] = api_base
            if api_key:
                extra_params["api_key"] = api_key

        # Pass through api_key for standard cloud providers (openai, anthropic, etc.)
        if api_key and "api_key" not in extra_params and provider not in ("ollama",):
            extra_params["api_key"] = api_key

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
            api_key = kwargs.pop("api_key", None) or llm_config.openai_api_key
            kwargs.pop("api_base", None)  # not used for direct openai
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **kwargs,
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            num_ctx = kwargs.pop("num_ctx", 4096)
            kwargs.pop("api_key", None)  # ollama doesn't use api_key
            custom_base = kwargs.pop("api_base", None)  # DB-configured URL override
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=custom_base or llm_config.ollama_host,
                num_ctx=num_ctx,
                **kwargs,
            )

        elif provider == "anthropic":
            api_key = kwargs.pop("api_key", None) or llm_config.anthropic_api_key
            kwargs.pop("api_base", None)  # not used for anthropic
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **kwargs,
            )

        elif provider == "vllm":
            # Phase 68: vLLM integration - use OpenAI-compatible interface
            # Settings DB takes priority over env vars
            api_key = kwargs.pop("api_key", None)
            api_base = kwargs.pop("api_base", None)
            vllm_api_base = api_base or _get_vllm_setting("llm.vllm_api_base") or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
            vllm_api_key = api_key or _get_vllm_setting("llm.vllm_api_key") or os.getenv("VLLM_API_KEY", "dummy")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=vllm_api_key,
                base_url=vllm_api_base,
                **kwargs,
            )

        elif provider in ("groq", "together", "deepinfra", "custom"):
            # OpenAI-compatible providers — use ChatOpenAI with custom base URL
            api_key = kwargs.pop("api_key", None)
            api_base = kwargs.pop("api_base", None)
            provider_base = api_base or cls.PROVIDER_API_BASES.get(provider, "")
            if not api_key and not provider_base:
                raise ValueError(f"Provider '{provider}' requires api_key or api_base. Configure in Settings > Providers.")
            native_kwargs = {}
            if provider_base:
                native_kwargs["base_url"] = provider_base
            if api_key:
                native_kwargs["api_key"] = api_key
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **native_kwargs,
                **kwargs,
            )

        elif provider == "google":
            api_key = kwargs.pop("api_key", None)
            kwargs.pop("api_base", None)  # not used for google
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    google_api_key=api_key,
                    **kwargs,
                )
            except ImportError:
                raise ValueError("Google provider requires: pip install langchain-google-genai")

        elif provider == "cohere":
            api_key = kwargs.pop("api_key", None)
            kwargs.pop("api_base", None)
            try:
                from langchain_cohere import ChatCohere
                return ChatCohere(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cohere_api_key=api_key,
                    **kwargs,
                )
            except ImportError:
                raise ValueError("Cohere provider requires: pip install langchain-cohere")

        elif provider == "azure":
            api_key = kwargs.pop("api_key", None)
            api_base = kwargs.pop("api_base", None)
            try:
                from langchain_openai import AzureChatOpenAI
                return AzureChatOpenAI(
                    azure_deployment=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    azure_endpoint=api_base or "",
                    api_key=api_key or "",
                    api_version="2024-02-01",
                    **kwargs,
                )
            except ImportError:
                raise ValueError("Azure provider requires: pip install langchain-openai")

        elif provider == "bedrock":
            api_key = kwargs.pop("api_key", None)
            api_base = kwargs.pop("api_base", None)  # region
            # Split combined key format: "ACCESS_KEY_ID:SECRET_ACCESS_KEY"
            aws_access_key = None
            aws_secret_key = None
            if api_key and ":" in api_key:
                parts = api_key.split(":", 1)
                aws_access_key = parts[0]
                aws_secret_key = parts[1]
            elif api_key:
                aws_access_key = api_key
            # Fallback to env vars (standard AWS credential chain)
            if not aws_access_key:
                aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            if not aws_secret_key:
                aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            try:
                from langchain_aws import ChatBedrock
                bedrock_kwargs: Dict[str, Any] = {
                    "model_id": model,
                    "region_name": api_base or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                    "model_kwargs": {"temperature": temperature, "max_tokens": max_tokens},
                }
                if aws_access_key and aws_secret_key:
                    import boto3
                    bedrock_kwargs["client"] = boto3.client(
                        "bedrock-runtime",
                        region_name=bedrock_kwargs["region_name"],
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key,
                    )
                return ChatBedrock(**bedrock_kwargs, **kwargs)
            except ImportError:
                raise ValueError("Bedrock provider requires: pip install langchain-aws boto3")

        else:
            raise ValueError(f"Unsupported provider without LiteLLM: {provider}. Install litellm or use a supported provider.")

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
            # Phase 98: Evict oldest entries to prevent unbounded memory growth
            cls._evict_oldest(cls._embedding_instances, cls._max_cache_size)
            logger.info(
                "Created embeddings model",
                provider=provider,
                model=model,
                cache_size=len(cls._embedding_instances),
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
            from langchain_ollama import OllamaEmbeddings
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
    # Phase 98: Added max_cache_entries and periodic cleanup to prevent memory leaks
    _cache: Dict[str, Tuple[Any, datetime]] = {}
    _cache_ttl_seconds: int = int(os.getenv("LLM_CONFIG_CACHE_TTL_SECONDS", "300"))
    _max_cache_entries: int = int(os.getenv("LLM_CONFIG_MAX_CACHE_ENTRIES", "500"))
    _cleanup_counter: int = 0
    _cleanup_interval: int = 50  # Run cleanup every 50 cache operations

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

        # Phase 98: Periodic cleanup to prevent memory leaks from expired entries
        cls._cleanup_counter += 1
        if cls._cleanup_counter >= cls._cleanup_interval:
            cls._cleanup_counter = 0
            cls._cleanup_expired_entries()

    @classmethod
    def _cleanup_expired_entries(cls) -> None:
        """Remove all expired entries from cache (Phase 98: memory leak prevention)."""
        now = datetime.now()
        expired_keys = [k for k, (_, exp) in cls._cache.items() if now >= exp]
        for key in expired_keys:
            del cls._cache[key]

        # Also enforce max cache size (LRU-style: remove oldest entries)
        if len(cls._cache) > cls._max_cache_entries:
            # Sort by expiry time and remove oldest
            sorted_keys = sorted(cls._cache.keys(), key=lambda k: cls._cache[k][1])
            to_remove = len(cls._cache) - cls._max_cache_entries
            for key in sorted_keys[:to_remove]:
                del cls._cache[key]

        if expired_keys or len(cls._cache) > cls._max_cache_entries * 0.9:
            logger.debug(
                "LLMConfigManager cache cleanup",
                expired_removed=len(expired_keys),
                current_size=len(cls._cache),
            )

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

        # Clear all session override caches (can't know which sessions reference
        # this provider without scanning, so clear all session entries)
        session_keys = [k for k in cls._cache.keys() if k.startswith("session:")]
        for key in session_keys:
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
                healthy_config = await cls._get_healthy_provider(db, health_checker, exclude_provider_id=config.provider_id)
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
        exclude_provider_id: Optional[str] = None,
    ) -> Optional[LLMConfigResult]:
        """Get any healthy provider as fallback."""
        try:
            healthy_provider = await health_checker.get_healthy_provider_for_failover(
                exclude_provider_id=exclude_provider_id
            )
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

        # Read context window for Ollama models — resolution order:
        # 1. Per-model override (from llm.model_context_overrides setting)
        # 2. Hardcoded research-backed recommendation
        # 3. Global llm.context_window setting (fallback for unknown models)
        if config.provider_type == "ollama" and "num_ctx" not in kwargs:
            try:
                from backend.services.settings import get_settings_service
                settings_svc = get_settings_service()

                resolved_ctx = None
                resolution_source = "default"

                # 1. Check per-model override
                overrides = await settings_svc.get_setting("llm.model_context_overrides")
                if isinstance(overrides, dict) and config.model in overrides:
                    resolved_ctx = int(overrides[config.model])
                    resolution_source = "per_model_override"

                # 2. Check hardcoded recommendation
                if resolved_ctx is None:
                    rec = get_recommended_context_window(config.model)
                    if rec:
                        resolved_ctx = rec["recommended"]
                        resolution_source = "recommendation"

                # 3. Fall back to global setting
                if resolved_ctx is None:
                    ctx_window = await settings_svc.get_setting("llm.context_window")
                    if ctx_window:
                        resolved_ctx = int(ctx_window)
                        resolution_source = "global_setting"

                if resolved_ctx:
                    kwargs["num_ctx"] = resolved_ctx
                    logger.info(
                        "Resolved context window",
                        model=config.model,
                        num_ctx=resolved_ctx,
                        source=resolution_source,
                    )
            except Exception:
                pass  # Use default 4096 from _create_model_from_config

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
            elif config.provider_type == "bedrock":
                litellm_model = f"bedrock/{config.model}"
            elif config.provider_type in ("groq", "together", "deepinfra", "vllm", "custom"):
                # OpenAI-compatible providers — use openai/ prefix
                litellm_model = f"openai/{config.model}"
            else:
                litellm_model = f"{config.provider_type}/{config.model}"

            # For Ollama models, pass num_ctx via model_kwargs so LiteLLM
            # forwards it to the Ollama API (controls context window size)
            if config.provider_type == "ollama":
                num_ctx = model_kwargs.pop("num_ctx", 4096)
                model_kwargs.setdefault("model_kwargs", {})
                model_kwargs["model_kwargs"]["num_ctx"] = num_ctx

            # For OpenAI-compatible providers, set api_base if not already set
            if config.provider_type in LLMFactory.PROVIDER_API_BASES and "api_base" not in model_kwargs:
                model_kwargs["api_base"] = LLMFactory.PROVIDER_API_BASES[config.provider_type]

            return ChatLiteLLM(
                model=litellm_model,
                **model_kwargs,
            )

        # Fallback to native implementations — delegate to LLMFactory._create_native_model
        # which supports all providers (openai, ollama, anthropic, vllm, groq, together,
        # deepinfra, google, cohere, azure, bedrock, custom)
        native_kwargs = {**kwargs}
        if config.api_key:
            native_kwargs["api_key"] = config.api_key
        if config.api_base_url:
            native_kwargs["api_base"] = config.api_base_url
        return LLMFactory._create_native_model(
            provider=config.provider_type,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **native_kwargs,
        )

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
            from langchain_ollama import OllamaEmbeddings
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
    wait = await _provider_rate_limiter.wait_if_needed(effective_provider)
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

async def test_llm_connection(provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Test LLM connection and return model info.

    Args:
        provider: Provider to test (defaults to llm_config.default_provider)

    Returns:
        dict: Connection test results
    """
    provider = provider or llm_config.default_provider
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


async def test_embeddings_connection(provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Test embeddings connection.

    Args:
        provider: Provider to test (defaults to llm_config.default_provider)

    Returns:
        dict: Connection test results
    """
    provider = provider or llm_config.default_provider
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


def _validate_ollama_url(url: str) -> str:
    """Validate Ollama base URL doesn't target dangerous internal networks.

    Allows localhost (Ollama runs locally) and LAN IPs (remote Ollama on LAN),
    but blocks cloud metadata, link-local, and reserved ranges.

    Returns validated URL or raises ValueError.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid Ollama URL: no hostname")

    blocked = {"metadata.google.internal", "metadata.gcp.internal", "169.254.169.254"}
    if hostname in blocked:
        raise ValueError("Ollama URL blocked: targets metadata endpoint")

    # Allow localhost — Ollama typically runs locally
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return url

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_link_local or ip.is_reserved:
            raise ValueError("Ollama URL blocked: targets link-local/reserved network")
    except ValueError as ve:
        if "blocked" in str(ve):
            raise
        # Not an IP — resolve DNS and check
        try:
            resolved = ipaddress.ip_address(socket.gethostbyname(hostname))
            if resolved.is_link_local or resolved.is_reserved:
                raise ValueError("Ollama URL blocked: resolves to link-local/reserved network")
        except socket.gaierror:
            pass

    return url


async def list_ollama_models(base_url: str = None) -> Dict[str, Any]:
    """
    List available models from local Ollama instance.

    Args:
        base_url: Optional Ollama API base URL. Defaults to config value.

    Returns:
        dict: Contains 'success', 'chat_models', 'embedding_models', 'vision_models', and 'total'
    """
    import httpx

    ollama_url = _validate_ollama_url(base_url or llm_config.ollama_host)

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
                vision_models = []

                for m in models:
                    name = m.get("name", "")
                    families = m.get("details", {}).get("families", [])
                    info = {
                        "name": name,
                        "size": m.get("size"),
                        "family": m.get("details", {}).get("family"),
                        "families": families,
                        "parameter_size": m.get("details", {}).get("parameter_size"),
                    }
                    if "embed" in name.lower():
                        embedding_models.append(info)
                    else:
                        chat_models.append(info)
                        # Also check if it's a vision model (has clip family or known vision pattern)
                        is_vision = "clip" in families or is_vision_model(name)
                        if is_vision:
                            vision_models.append(info)

                return {
                    "success": True,
                    "chat_models": chat_models,
                    "embedding_models": embedding_models,
                    "vision_models": vision_models,
                    "total": len(models),
                }
            return {
                "success": False,
                "error": f"Ollama returned {response.status_code}",
                "chat_models": [],
                "embedding_models": [],
                "vision_models": [],
            }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama",
            "chat_models": [],
            "embedding_models": [],
            "vision_models": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chat_models": [],
            "embedding_models": [],
            "vision_models": [],
        }


# =============================================================================
# Per-Model Context Window Recommendations (Research-Backed 2025-2026)
# =============================================================================
# Each entry: recommended = practical sweet spot for RAG/QA tasks
#             max = model architecture limit
#             vram = estimated VRAM at recommended ctx (Q4 quantization)
# Sub-3B: "lost in the middle" past 4K. 7-9B: sweet spot 4-8K.
# DeepSeek R1: needs headroom for chain-of-thought tokens.

MODEL_CONTEXT_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    # Llama 3.2 family
    "llama3.2:1b":       {"recommended": 2048,  "max": 128000, "vram": "~1.5-2 GB"},
    "llama-3.2-1b":      {"recommended": 2048,  "max": 128000, "vram": "~1.5-2 GB"},
    "llama3.2:3b":       {"recommended": 8192,  "max": 128000, "vram": "~3-4 GB"},
    "llama3.2:latest":   {"recommended": 8192,  "max": 128000, "vram": "~3-4 GB"},
    "llama-3.2-3b":      {"recommended": 8192,  "max": 128000, "vram": "~3-4 GB"},
    # Llama 3.1 / 3 larger
    "llama3.1:8b":       {"recommended": 8192,  "max": 128000, "vram": "~6-8 GB"},
    "llama3.1:latest":   {"recommended": 8192,  "max": 128000, "vram": "~6-8 GB"},
    "llama3:8b":         {"recommended": 8192,  "max": 128000, "vram": "~6-8 GB"},
    "llama3:70b":        {"recommended": 16384, "max": 128000, "vram": "~40-48 GB"},
    # DeepSeek R1 (distilled)
    "deepseek-r1:1.5b":  {"recommended": 2048,  "max": 128000, "vram": "~1.5-2.5 GB"},
    "deepseek-r1:7b":    {"recommended": 6144,  "max": 128000, "vram": "~5-7 GB"},
    "deepseek-r1:8b":    {"recommended": 6144,  "max": 128000, "vram": "~6-8 GB"},
    "deepseek-r1:14b":   {"recommended": 8192,  "max": 128000, "vram": "~10-13 GB"},
    "deepseek-r1:32b":   {"recommended": 8192,  "max": 128000, "vram": "~20-24 GB"},
    "deepseek-r1:70b":   {"recommended": 16384, "max": 128000, "vram": "~40-48 GB"},
    # Phi-3
    "phi3:mini":         {"recommended": 4096,  "max": 128000, "vram": "~3-4 GB"},
    "phi3:medium":       {"recommended": 8192,  "max": 128000, "vram": "~10-13 GB"},
    # Qwen 2.5
    "qwen2.5:0.5b":     {"recommended": 2048,  "max": 128000, "vram": "~0.8-1.2 GB"},
    "qwen2.5:1.5b":     {"recommended": 2048,  "max": 128000, "vram": "~1.5-2 GB"},
    "qwen2.5:3b":       {"recommended": 4096,  "max": 128000, "vram": "~3-4 GB"},
    "qwen2.5:7b":       {"recommended": 8192,  "max": 128000, "vram": "~6-8 GB"},
    "qwen2.5:14b":      {"recommended": 8192,  "max": 128000, "vram": "~10-13 GB"},
    "qwen2.5:32b":      {"recommended": 8192,  "max": 128000, "vram": "~20-24 GB"},
    # Gemma 2 (hard 8K architecture limit)
    "gemma2:2b":         {"recommended": 4096,  "max": 8192,   "vram": "~2-3 GB"},
    "gemma2:9b":         {"recommended": 8192,  "max": 8192,   "vram": "~7-9 GB"},
    "gemma2:27b":        {"recommended": 8192,  "max": 8192,   "vram": "~18-22 GB"},
    # Mistral / Mixtral
    "mistral:7b":        {"recommended": 8192,  "max": 32000,  "vram": "~5-7 GB"},
    "mistral:latest":    {"recommended": 8192,  "max": 32000,  "vram": "~5-7 GB"},
    "mixtral:8x7b":      {"recommended": 16384, "max": 32000,  "vram": "~30-32 GB"},
    "mixtral:latest":    {"recommended": 16384, "max": 32000,  "vram": "~30-32 GB"},
    # CodeLlama
    "codellama:7b":      {"recommended": 8192,  "max": 16384,  "vram": "~5-7 GB"},
    "codellama:13b":     {"recommended": 8192,  "max": 16384,  "vram": "~9-11 GB"},
    "codellama:34b":     {"recommended": 8192,  "max": 16384,  "vram": "~20-24 GB"},
}


def get_recommended_context_window(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get research-backed recommended context window for an Ollama model.

    Uses exact match first, then longest substring match for flexibility
    (e.g., "llama3.2:3b-q4_K_M" matches "llama3.2:3b").

    Returns:
        Dict with 'recommended', 'max', 'vram' or None if unknown model.
    """
    if not model_name:
        return None

    model_lower = model_name.lower()

    # Exact match first
    if model_lower in MODEL_CONTEXT_RECOMMENDATIONS:
        return MODEL_CONTEXT_RECOMMENDATIONS[model_lower]

    # Substring match (longest match wins for specificity)
    best_match = None
    best_len = 0
    for pattern, rec in MODEL_CONTEXT_RECOMMENDATIONS.items():
        if pattern in model_lower and len(pattern) > best_len:
            best_match = rec
            best_len = len(pattern)

    return best_match


async def get_ollama_model_context_length(model_name: str = None, base_url: str = None) -> Dict[str, Any]:
    """
    Get the maximum context length for an Ollama model via /api/show.

    Args:
        model_name: Model name. Defaults to configured chat model.
        base_url: Ollama API URL. Defaults to config.

    Returns:
        dict with 'context_length', 'model_name', 'parameter_size'
    """
    import httpx

    ollama_url = _validate_ollama_url(base_url or llm_config.ollama_host)

    if not model_name:
        model_name = os.getenv("DEFAULT_CHAT_MODEL", "llama3.2:latest")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ollama_url}/api/show",
                json={"name": model_name},
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                model_info = data.get("model_info", {})

                # Find context_length key (varies by model family)
                context_length = None
                for key, value in model_info.items():
                    if "context_length" in key.lower():
                        context_length = value
                        break

                parameter_size = data.get("details", {}).get("parameter_size", "")

                return {
                    "success": True,
                    "model_name": model_name,
                    "context_length": context_length,
                    "parameter_size": parameter_size,
                }
            return {
                "success": False,
                "error": f"Ollama returned {response.status_code}",
                "model_name": model_name,
                "context_length": None,
            }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama",
            "model_name": model_name,
            "context_length": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name,
            "context_length": None,
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

    ollama_url = _validate_ollama_url(base_url or llm_config.ollama_host)

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

    ollama_url = _validate_ollama_url(base_url or llm_config.ollama_host)

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
    # Get vision model from settings first, then provider-specific defaults
    provider = provider or llm_config.default_provider
    if model is None:
        # Try reading from settings
        try:
            from backend.services.settings import get_settings_service
            _settings = get_settings_service()
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                pass  # Can't do sync settings read in async context, use defaults below
            else:
                _vlm_model = asyncio.run(_settings.get_setting("rag.vlm_model"))
                if _vlm_model:
                    model = _vlm_model
        except Exception:
            pass
    if model is None:
        # Provider-specific vision model defaults
        if provider == "ollama":
            model = "llava"
        elif provider == "openai":
            model = "gpt-4o"
        elif provider == "anthropic":
            model = "claude-3-5-sonnet-20241022"
        else:
            model = "llava"  # Default to free local model

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
