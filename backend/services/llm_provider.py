"""
AIDocumentIndexer - LLM Provider Service
=========================================

Manages LLM provider configurations stored in the database.
Supports dynamic provider switching without application restart.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context
from backend.db.models import LLMProvider
from backend.services.encryption import decrypt_value, encrypt_value, mask_api_key

logger = structlog.get_logger(__name__)


# =============================================================================
# Provider Type Definitions
# =============================================================================

PROVIDER_TYPES = {
    "openai": {
        "name": "OpenAI",
        "fields": ["api_key", "organization_id"],
        "required_fields": ["api_key"],
        "chat_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "embedding_models": ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
        "default_chat_model": "gpt-4o",
        "default_embedding_model": "text-embedding-3-small",
    },
    "anthropic": {
        "name": "Anthropic",
        "fields": ["api_key"],
        "required_fields": ["api_key"],
        "chat_models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "embedding_models": [],
        "default_chat_model": "claude-3-5-sonnet-20241022",
        "default_embedding_model": None,
    },
    "ollama": {
        "name": "Ollama (Local)",
        "fields": ["api_base_url"],
        "required_fields": [],
        "chat_models": "dynamic",  # Fetched from Ollama API
        "embedding_models": "dynamic",
        "default_chat_model": "llama3.2",
        "default_embedding_model": "nomic-embed-text",
        "default_api_base_url": "http://localhost:11434",
    },
    "azure": {
        "name": "Azure OpenAI",
        "fields": ["api_key", "api_base_url", "organization_id"],
        "required_fields": ["api_key", "api_base_url"],
        "chat_models": "deployment-based",
        "embedding_models": "deployment-based",
        "default_chat_model": None,
        "default_embedding_model": None,
    },
    "google": {
        "name": "Google AI",
        "fields": ["api_key"],
        "required_fields": ["api_key"],
        "chat_models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "embedding_models": ["text-embedding-004"],
        "default_chat_model": "gemini-1.5-pro",
        "default_embedding_model": "text-embedding-004",
    },
    "groq": {
        "name": "Groq",
        "fields": ["api_key"],
        "required_fields": ["api_key"],
        "chat_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "embedding_models": [],
        "default_chat_model": "llama-3.3-70b-versatile",
        "default_embedding_model": None,
    },
    "together": {
        "name": "Together AI",
        "fields": ["api_key"],
        "required_fields": ["api_key"],
        "chat_models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        "embedding_models": ["togethercomputer/m2-bert-80M-8k-retrieval"],
        "default_chat_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "default_embedding_model": "togethercomputer/m2-bert-80M-8k-retrieval",
    },
    "cohere": {
        "name": "Cohere",
        "fields": ["api_key"],
        "required_fields": ["api_key"],
        "chat_models": ["command-r-plus", "command-r", "command"],
        "embedding_models": ["embed-english-v3.0", "embed-multilingual-v3.0"],
        "default_chat_model": "command-r-plus",
        "default_embedding_model": "embed-english-v3.0",
    },
    "custom": {
        "name": "Custom (OpenAI-compatible)",
        "fields": ["api_key", "api_base_url"],
        "required_fields": ["api_base_url"],
        "chat_models": "manual",
        "embedding_models": "manual",
        "default_chat_model": None,
        "default_embedding_model": None,
    },
}


# =============================================================================
# LLM Provider Service
# =============================================================================

class LLMProviderService:
    """Service for managing LLM provider configurations."""

    @staticmethod
    async def list_providers(session: AsyncSession) -> List[LLMProvider]:
        """List all configured LLM providers."""
        result = await session.execute(
            select(LLMProvider).order_by(LLMProvider.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_provider(session: AsyncSession, provider_id: str) -> Optional[LLMProvider]:
        """Get a specific provider by ID."""
        result = await session.execute(
            select(LLMProvider).where(LLMProvider.id == provider_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_provider_by_name(session: AsyncSession, name: str) -> Optional[LLMProvider]:
        """Get a provider by name."""
        result = await session.execute(
            select(LLMProvider).where(LLMProvider.name == name)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_default_provider(session: AsyncSession) -> Optional[LLMProvider]:
        """Get the default LLM provider."""
        result = await session.execute(
            select(LLMProvider).where(LLMProvider.is_default == True)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create_provider(
        session: AsyncSession,
        name: str,
        provider_type: str,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        organization_id: Optional[str] = None,
        default_chat_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        is_default: bool = False,
        settings: Optional[Dict[str, Any]] = None,
    ) -> LLMProvider:
        """Create a new LLM provider configuration."""
        # Validate provider type
        if provider_type not in PROVIDER_TYPES:
            raise ValueError(f"Invalid provider type: {provider_type}")

        # Encrypt API key if provided
        encrypted_api_key = encrypt_value(api_key) if api_key else None

        # Set defaults from provider type if not specified
        type_config = PROVIDER_TYPES[provider_type]
        if not default_chat_model and type_config.get("default_chat_model"):
            default_chat_model = type_config["default_chat_model"]
        if not default_embedding_model and type_config.get("default_embedding_model"):
            default_embedding_model = type_config["default_embedding_model"]
        if not api_base_url and type_config.get("default_api_base_url"):
            api_base_url = type_config["default_api_base_url"]

        # If this is being set as default, clear other defaults first
        if is_default:
            await session.execute(
                update(LLMProvider).values(is_default=False)
            )

        provider = LLMProvider(
            name=name,
            provider_type=provider_type,
            api_key_encrypted=encrypted_api_key,
            api_base_url=api_base_url,
            organization_id=organization_id,
            default_chat_model=default_chat_model,
            default_embedding_model=default_embedding_model,
            is_default=is_default,
            is_active=True,
            settings=settings or {},
        )

        session.add(provider)
        await session.flush()
        await session.refresh(provider)

        logger.info(
            "Created LLM provider",
            provider_id=str(provider.id),
            name=name,
            provider_type=provider_type,
        )

        return provider

    @staticmethod
    async def update_provider(
        session: AsyncSession,
        provider_id: str,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        organization_id: Optional[str] = None,
        default_chat_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        is_active: Optional[bool] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[LLMProvider]:
        """Update an existing LLM provider."""
        provider = await LLMProviderService.get_provider(session, provider_id)
        if not provider:
            return None

        if name is not None:
            provider.name = name
        if api_key is not None:
            provider.api_key_encrypted = encrypt_value(api_key)
        if api_base_url is not None:
            provider.api_base_url = api_base_url
        if organization_id is not None:
            provider.organization_id = organization_id
        if default_chat_model is not None:
            provider.default_chat_model = default_chat_model
        if default_embedding_model is not None:
            provider.default_embedding_model = default_embedding_model
        if is_active is not None:
            provider.is_active = is_active
        if settings is not None:
            provider.settings = settings

        await session.flush()
        await session.refresh(provider)

        logger.info("Updated LLM provider", provider_id=provider_id)
        return provider

    @staticmethod
    async def delete_provider(session: AsyncSession, provider_id: str) -> bool:
        """Delete an LLM provider."""
        provider = await LLMProviderService.get_provider(session, provider_id)
        if not provider:
            return False

        await session.delete(provider)
        logger.info("Deleted LLM provider", provider_id=provider_id)
        return True

    @staticmethod
    async def set_default_provider(session: AsyncSession, provider_id: str) -> Optional[LLMProvider]:
        """Set a provider as the default."""
        provider = await LLMProviderService.get_provider(session, provider_id)
        if not provider:
            return None

        # Clear existing default
        await session.execute(
            update(LLMProvider).values(is_default=False)
        )

        # Set new default
        provider.is_default = True
        await session.flush()
        await session.refresh(provider)

        logger.info("Set default LLM provider", provider_id=provider_id)
        return provider

    @staticmethod
    async def test_provider(session: AsyncSession, provider_id: str) -> Dict[str, Any]:
        """Test connection to an LLM provider."""
        provider = await LLMProviderService.get_provider(session, provider_id)
        if not provider:
            return {"success": False, "error": "Provider not found"}

        try:
            # Decrypt API key for testing
            api_key = decrypt_value(provider.api_key_encrypted) if provider.api_key_encrypted else None
            base_url = provider.api_base_url

            if provider.provider_type == "openai":
                return await _test_openai(api_key)
            elif provider.provider_type == "anthropic":
                return await _test_anthropic(api_key)
            elif provider.provider_type == "ollama":
                return await _test_ollama(base_url or "http://localhost:11434")
            elif provider.provider_type == "azure":
                return await _test_azure(api_key, base_url)
            elif provider.provider_type == "google":
                return await _test_google(api_key)
            elif provider.provider_type == "groq":
                return await _test_groq(api_key)
            elif provider.provider_type == "together":
                return await _test_together(api_key)
            elif provider.provider_type == "cohere":
                return await _test_cohere(api_key)
            elif provider.provider_type == "custom":
                return await _test_custom(api_key, base_url)
            else:
                return {"success": False, "error": f"Unknown provider type: {provider.provider_type}"}

        except Exception as e:
            logger.error("Provider test failed", provider_id=provider_id, error=str(e))
            return {"success": False, "error": str(e)}

    @staticmethod
    async def list_available_models(session: AsyncSession, provider_id: str) -> Dict[str, Any]:
        """List available models for a provider."""
        provider = await LLMProviderService.get_provider(session, provider_id)
        if not provider:
            return {"success": False, "error": "Provider not found"}

        try:
            api_key = decrypt_value(provider.api_key_encrypted) if provider.api_key_encrypted else None
            base_url = provider.api_base_url

            if provider.provider_type == "ollama":
                return await _list_ollama_models(base_url or "http://localhost:11434")
            elif provider.provider_type in PROVIDER_TYPES:
                type_config = PROVIDER_TYPES[provider.provider_type]
                chat_models = type_config.get("chat_models", [])
                embedding_models = type_config.get("embedding_models", [])

                if isinstance(chat_models, list) and isinstance(embedding_models, list):
                    return {
                        "success": True,
                        "chat_models": chat_models,
                        "embedding_models": embedding_models,
                    }
                else:
                    return {
                        "success": True,
                        "chat_models": [],
                        "embedding_models": [],
                        "note": "Models must be configured manually for this provider",
                    }
            else:
                return {"success": False, "error": f"Unknown provider type: {provider.provider_type}"}

        except Exception as e:
            logger.error("Failed to list models", provider_id=provider_id, error=str(e))
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_provider_types() -> Dict[str, Any]:
        """Get all supported provider types with their configurations."""
        return {
            provider_type: {
                "name": config["name"],
                "fields": config["fields"],
                "required_fields": config["required_fields"],
                "chat_models": config["chat_models"] if isinstance(config["chat_models"], list) else [],
                "embedding_models": config["embedding_models"] if isinstance(config["embedding_models"], list) else [],
                "default_chat_model": config.get("default_chat_model"),
                "default_embedding_model": config.get("default_embedding_model"),
                "default_api_base_url": config.get("default_api_base_url"),
            }
            for provider_type, config in PROVIDER_TYPES.items()
        }

    @staticmethod
    def format_provider_response(provider: LLMProvider, include_api_key: bool = False) -> Dict[str, Any]:
        """Format a provider for API response (masking sensitive data)."""
        result = {
            "id": str(provider.id),
            "name": provider.name,
            "provider_type": provider.provider_type,
            "api_base_url": provider.api_base_url,
            "organization_id": provider.organization_id,
            "is_active": provider.is_active,
            "is_default": provider.is_default,
            "default_chat_model": provider.default_chat_model,
            "default_embedding_model": provider.default_embedding_model,
            "settings": provider.settings,
            "created_at": provider.created_at.isoformat() if provider.created_at else None,
            "updated_at": provider.updated_at.isoformat() if provider.updated_at else None,
        }

        # Mask API key for display
        if provider.api_key_encrypted:
            try:
                api_key = decrypt_value(provider.api_key_encrypted)
                result["api_key_masked"] = mask_api_key(api_key)
                result["has_api_key"] = True
            except Exception:
                result["api_key_masked"] = "****"
                result["has_api_key"] = True
        else:
            result["api_key_masked"] = None
            result["has_api_key"] = False

        return result


# =============================================================================
# Provider Test Functions
# =============================================================================

async def _test_openai(api_key: str) -> Dict[str, Any]:
    """Test OpenAI connection."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if response.status_code == 200:
            return {"success": True, "message": "Connected to OpenAI successfully"}
        else:
            return {"success": False, "error": f"OpenAI API error: {response.status_code}"}


async def _test_anthropic(api_key: str) -> Dict[str, Any]:
    """Test Anthropic connection."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=10.0,
        )
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Anthropic successfully"}
        else:
            return {"success": False, "error": f"Anthropic API error: {response.status_code}"}


async def _test_ollama(base_url: str) -> Dict[str, Any]:
    """Test Ollama connection."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/api/tags", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {
                    "success": True,
                    "message": f"Connected to Ollama. Found {len(models)} models.",
                    "models": models,
                }
            else:
                return {"success": False, "error": f"Ollama error: {response.status_code}"}
        except httpx.ConnectError:
            return {"success": False, "error": "Could not connect to Ollama. Is it running?"}


async def _test_azure(api_key: str, endpoint: str) -> Dict[str, Any]:
    """Test Azure OpenAI connection."""
    if not endpoint:
        return {"success": False, "error": "Azure endpoint URL is required"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{endpoint.rstrip('/')}/openai/models?api-version=2024-02-15-preview",
            headers={"api-key": api_key},
            timeout=10.0,
        )
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Azure OpenAI successfully"}
        else:
            return {"success": False, "error": f"Azure API error: {response.status_code}"}


async def _test_google(api_key: str) -> Dict[str, Any]:
    """Test Google AI connection."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10.0,
        )
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Google AI successfully"}
        else:
            return {"success": False, "error": f"Google AI error: {response.status_code}"}


async def _test_groq(api_key: str) -> Dict[str, Any]:
    """Test Groq connection."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Groq successfully"}
        else:
            return {"success": False, "error": f"Groq API error: {response.status_code}"}


async def _test_together(api_key: str) -> Dict[str, Any]:
    """Test Together AI connection."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.together.xyz/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Together AI successfully"}
        else:
            return {"success": False, "error": f"Together AI error: {response.status_code}"}


async def _test_cohere(api_key: str) -> Dict[str, Any]:
    """Test Cohere connection."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.cohere.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if response.status_code == 200:
            return {"success": True, "message": "Connected to Cohere successfully"}
        else:
            return {"success": False, "error": f"Cohere API error: {response.status_code}"}


async def _test_custom(api_key: Optional[str], base_url: str) -> Dict[str, Any]:
    """Test custom OpenAI-compatible endpoint."""
    if not base_url:
        return {"success": False, "error": "Base URL is required for custom provider"}

    async with httpx.AsyncClient() as client:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = await client.get(
                f"{base_url.rstrip('/')}/v1/models",
                headers=headers,
                timeout=10.0,
            )
            if response.status_code == 200:
                return {"success": True, "message": "Connected to custom endpoint successfully"}
            else:
                return {"success": False, "error": f"Custom endpoint error: {response.status_code}"}
        except httpx.ConnectError:
            return {"success": False, "error": f"Could not connect to {base_url}"}


# Known Ollama vision model patterns for detection
OLLAMA_VISION_MODEL_PATTERNS = [
    "llava", "qwen2-vl", "qwen2.5-vl", "llama3.2-vision", "llama-3.2-vision",
    "moondream", "bakllava", "minicpm-v", "cogvlm", "yi-vl", "internlm-xcomposer",
    "phi-3-vision", "llava-llama3", "llava-phi3", "nanollava", "llava-1.6",
]


def _is_vision_model(model_name: str) -> bool:
    """Check if a model is a vision/multimodal model."""
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in OLLAMA_VISION_MODEL_PATTERNS)


async def _list_ollama_models(base_url: str) -> Dict[str, Any]:
    """List models available in Ollama, including vision model detection."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/api/tags", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]

                # Categorize models
                chat_models = [m for m in models if not m.endswith("-embed") and "embed" not in m.lower()]
                embedding_models = [m for m in models if "embed" in m.lower()]
                vision_models = [m for m in chat_models if _is_vision_model(m)]

                return {
                    "success": True,
                    "chat_models": chat_models,
                    "embedding_models": embedding_models,
                    "vision_models": vision_models,
                }
            else:
                return {"success": False, "error": f"Ollama error: {response.status_code}"}
        except httpx.ConnectError:
            return {"success": False, "error": "Could not connect to Ollama"}
