"""
AIDocumentIndexer - LLM Gateway
================================

OpenAI-compatible API proxy with budget enforcement and routing.

Features:
- OpenAI-compatible /v1/chat/completions endpoint
- Multi-provider routing (OpenAI, Anthropic, Ollama)
- Budget checking before requests
- Usage tracking after requests
- Virtual key validation
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel

from backend.services.base import BaseService, ServiceException
from backend.services.llm_gateway.budget import BudgetManager
from backend.services.llm_gateway.virtual_keys import VirtualKeyManager
from backend.services.llm_gateway.usage import UsageTracker

logger = structlog.get_logger(__name__)


class GatewayProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE = "azure"


@dataclass
class GatewayRequest:
    """Incoming gateway request."""
    model: str
    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayResponse:
    """Gateway response."""
    id: str
    model: str
    provider: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    cost_usd: float
    latency_ms: int
    created: int = field(default_factory=lambda: int(time.time()))

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
            "usage": self.usage,
            # Custom fields
            "x_provider": self.provider,
            "x_cost_usd": self.cost_usd,
            "x_latency_ms": self.latency_ms,
        }


# Model to provider mapping
MODEL_PROVIDERS = {
    # OpenAI models
    "gpt-4o": GatewayProvider.OPENAI,
    "gpt-4o-mini": GatewayProvider.OPENAI,
    "gpt-4-turbo": GatewayProvider.OPENAI,
    "gpt-4": GatewayProvider.OPENAI,
    "gpt-3.5-turbo": GatewayProvider.OPENAI,
    "o1-preview": GatewayProvider.OPENAI,
    "o1-mini": GatewayProvider.OPENAI,

    # Anthropic models
    "claude-3-opus": GatewayProvider.ANTHROPIC,
    "claude-3-sonnet": GatewayProvider.ANTHROPIC,
    "claude-3-haiku": GatewayProvider.ANTHROPIC,
    "claude-3-5-sonnet": GatewayProvider.ANTHROPIC,
    "claude-3-5-haiku": GatewayProvider.ANTHROPIC,

    # Ollama models (local)
    "llama3": GatewayProvider.OLLAMA,
    "llama3.1": GatewayProvider.OLLAMA,
    "mistral": GatewayProvider.OLLAMA,
    "mixtral": GatewayProvider.OLLAMA,
    "codellama": GatewayProvider.OLLAMA,
    "phi3": GatewayProvider.OLLAMA,
    "qwen2": GatewayProvider.OLLAMA,
}


class LLMGateway(BaseService):
    """
    LLM Gateway service - OpenAI-compatible API proxy.

    Features:
    - Budget enforcement (check before, record after)
    - Virtual API key validation
    - Multi-provider routing
    - Usage tracking
    - Streaming support
    """

    def __init__(
        self,
        session=None,
        organization_id=None,
        user_id=None,
    ):
        super().__init__(session, organization_id, user_id)
        self._budget_manager: Optional[BudgetManager] = None
        self._key_manager: Optional[VirtualKeyManager] = None
        self._usage_tracker: Optional[UsageTracker] = None

    async def _get_budget_manager(self) -> BudgetManager:
        """Get or create budget manager."""
        if not self._budget_manager:
            session = await self.get_session()
            self._budget_manager = BudgetManager(
                session=session,
                organization_id=self._organization_id,
                user_id=self._user_id,
            )
        return self._budget_manager

    async def _get_key_manager(self) -> VirtualKeyManager:
        """Get or create key manager."""
        if not self._key_manager:
            session = await self.get_session()
            self._key_manager = VirtualKeyManager(
                session=session,
                organization_id=self._organization_id,
                user_id=self._user_id,
            )
        return self._key_manager

    async def _get_usage_tracker(self) -> UsageTracker:
        """Get or create usage tracker."""
        if not self._usage_tracker:
            session = await self.get_session()
            self._usage_tracker = UsageTracker(
                session=session,
                organization_id=self._organization_id,
                user_id=self._user_id,
            )
        return self._usage_tracker

    def _get_provider(self, model: str) -> GatewayProvider:
        """Get the provider for a model."""
        # Check exact match first
        if model in MODEL_PROVIDERS:
            return MODEL_PROVIDERS[model]

        # Check prefix matches
        model_lower = model.lower()
        if model_lower.startswith("gpt-"):
            return GatewayProvider.OPENAI
        if model_lower.startswith("claude"):
            return GatewayProvider.ANTHROPIC
        if model_lower.startswith("o1"):
            return GatewayProvider.OPENAI

        # Default to Ollama for unknown models (assumes local)
        return GatewayProvider.OLLAMA

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate input tokens from messages."""
        # Rough estimation: ~4 chars per token
        total_chars = sum(
            len(str(m.get("content", "")))
            for m in messages
        )
        return total_chars // 4

    async def validate_request(
        self,
        request: GatewayRequest,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a request before processing.

        Args:
            request: The gateway request
            api_key: Virtual API key (if provided)

        Returns:
            Validation result with 'valid', 'error', 'warnings' keys
        """
        result = {
            "valid": True,
            "error": None,
            "warnings": [],
            "virtual_key": None,
        }

        # Validate virtual API key if provided
        if api_key:
            key_manager = await self._get_key_manager()
            is_valid, virtual_key, error = await key_manager.validate_key(
                api_key,
                model=request.model,
            )

            if not is_valid:
                result["valid"] = False
                result["error"] = error or "Invalid API key"
                return result

            result["virtual_key"] = virtual_key

            # Check key-specific limits
            if virtual_key.scope.max_tokens_per_request:
                if request.max_tokens and request.max_tokens > virtual_key.scope.max_tokens_per_request:
                    result["valid"] = False
                    result["error"] = f"max_tokens exceeds key limit of {virtual_key.scope.max_tokens_per_request}"
                    return result

        # Estimate cost for budget check
        estimated_input_tokens = self._estimate_tokens(request.messages)
        estimated_output_tokens = request.max_tokens or 1000
        usage_tracker = await self._get_usage_tracker()
        estimated_cost = usage_tracker.calculate_cost(
            request.model,
            estimated_input_tokens,
            estimated_output_tokens,
        )

        # Check budget
        budget_manager = await self._get_budget_manager()
        user_id = str(self._user_id) if self._user_id else None
        budget_check = await budget_manager.check_budget(
            estimated_cost,
            user_id=user_id,
        )

        if not budget_check["allowed"]:
            result["valid"] = False
            result["error"] = f"Budget exceeded: {budget_check['blocked_by']}"
            return result

        if budget_check["warnings"]:
            result["warnings"].extend(budget_check["warnings"])

        return result

    async def complete(
        self,
        request: GatewayRequest,
        api_key: Optional[str] = None,
    ) -> GatewayResponse:
        """
        Process a chat completion request.

        Args:
            request: The gateway request
            api_key: Virtual API key (optional)

        Returns:
            GatewayResponse
        """
        start_time = time.time()

        # Validate request
        validation = await self.validate_request(request, api_key)
        if not validation["valid"]:
            raise ServiceException(
                validation["error"],
                code="GATEWAY_VALIDATION_ERROR",
            )

        # Get provider
        provider = self._get_provider(request.model)

        # Route to appropriate provider
        try:
            if provider == GatewayProvider.OPENAI:
                response = await self._call_openai(request)
            elif provider == GatewayProvider.ANTHROPIC:
                response = await self._call_anthropic(request)
            elif provider == GatewayProvider.OLLAMA:
                response = await self._call_ollama(request)
            else:
                raise ServiceException(f"Unsupported provider: {provider}")

        except Exception as e:
            # Record failed request
            latency_ms = int((time.time() - start_time) * 1000)
            usage_tracker = await self._get_usage_tracker()
            await usage_tracker.record(
                model=request.model,
                provider=provider.value,
                endpoint="chat/completions",
                input_tokens=self._estimate_tokens(request.messages),
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
                virtual_key_id=validation["virtual_key"].id if validation.get("virtual_key") else None,
            )
            raise

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Calculate cost
        usage_tracker = await self._get_usage_tracker()
        cost = usage_tracker.calculate_cost(
            request.model,
            response["usage"]["prompt_tokens"],
            response["usage"]["completion_tokens"],
        )

        # Record usage
        await usage_tracker.record(
            model=request.model,
            provider=provider.value,
            endpoint="chat/completions",
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
            latency_ms=latency_ms,
            success=True,
            virtual_key_id=validation["virtual_key"].id if validation.get("virtual_key") else None,
            metadata={"temperature": request.temperature},
        )

        # Update virtual key usage
        if validation.get("virtual_key"):
            key_manager = await self._get_key_manager()
            await key_manager.record_usage(
                validation["virtual_key"].id,
                tokens=response["usage"]["total_tokens"],
                cost=cost,
            )

        # Record spend against budgets
        budget_manager = await self._get_budget_manager()
        await budget_manager.record_spend(
            cost,
            user_id=str(self._user_id) if self._user_id else None,
            model=request.model,
        )

        return GatewayResponse(
            id=response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            model=request.model,
            provider=provider.value,
            choices=response["choices"],
            usage=response["usage"],
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    async def complete_stream(
        self,
        request: GatewayRequest,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a streaming chat completion request.

        Args:
            request: The gateway request
            api_key: Virtual API key (optional)

        Yields:
            Streaming chunks in OpenAI format
        """
        start_time = time.time()

        # Validate request
        validation = await self.validate_request(request, api_key)
        if not validation["valid"]:
            raise ServiceException(
                validation["error"],
                code="GATEWAY_VALIDATION_ERROR",
            )

        # Get provider
        provider = self._get_provider(request.model)

        # Track tokens for usage recording
        input_tokens = self._estimate_tokens(request.messages)
        output_tokens = 0
        accumulated_content = ""

        try:
            if provider == GatewayProvider.OPENAI:
                async for chunk in self._stream_openai(request):
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            accumulated_content += delta["content"]
                    yield chunk
            elif provider == GatewayProvider.ANTHROPIC:
                async for chunk in self._stream_anthropic(request):
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            accumulated_content += delta["content"]
                    yield chunk
            elif provider == GatewayProvider.OLLAMA:
                async for chunk in self._stream_ollama(request):
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            accumulated_content += delta["content"]
                    yield chunk
            else:
                raise ServiceException(f"Unsupported provider: {provider}")

            # Estimate output tokens
            output_tokens = len(accumulated_content) // 4

        except Exception as e:
            # Record failed request
            latency_ms = int((time.time() - start_time) * 1000)
            usage_tracker = await self._get_usage_tracker()
            await usage_tracker.record(
                model=request.model,
                provider=provider.value,
                endpoint="chat/completions",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
            )
            raise

        # Record successful usage
        latency_ms = int((time.time() - start_time) * 1000)
        usage_tracker = await self._get_usage_tracker()
        cost = usage_tracker.calculate_cost(request.model, input_tokens, output_tokens)

        await usage_tracker.record(
            model=request.model,
            provider=provider.value,
            endpoint="chat/completions",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=True,
            virtual_key_id=validation["virtual_key"].id if validation.get("virtual_key") else None,
        )

        # Record spend
        budget_manager = await self._get_budget_manager()
        await budget_manager.record_spend(
            cost,
            user_id=str(self._user_id) if self._user_id else None,
            model=request.model,
        )

    async def _call_openai(self, request: GatewayRequest) -> Dict[str, Any]:
        """Call OpenAI API."""
        from openai import AsyncOpenAI
        from backend.core.config import settings

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        response = await client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )

        return {
            "id": response.id,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    async def _stream_openai(self, request: GatewayRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from OpenAI API."""
        from openai import AsyncOpenAI
        from backend.core.config import settings

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        stream = await client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            yield {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": chunk.created,
                "model": chunk.model,
                "choices": [
                    {
                        "index": c.index,
                        "delta": {
                            "role": c.delta.role,
                            "content": c.delta.content,
                        } if c.delta else {},
                        "finish_reason": c.finish_reason,
                    }
                    for c in chunk.choices
                ],
            }

    async def _call_anthropic(self, request: GatewayRequest) -> Dict[str, Any]:
        """Call Anthropic API."""
        import anthropic
        from backend.core.config import settings

        client = anthropic.AsyncAnthropic(
            api_key=getattr(settings, "ANTHROPIC_API_KEY", None)
        )

        # Convert messages to Anthropic format
        system_message = None
        messages = []
        for m in request.messages:
            if m["role"] == "system":
                system_message = m["content"]
            else:
                messages.append({
                    "role": m["role"],
                    "content": m["content"],
                })

        response = await client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens or 4096,
            system=system_message,
            messages=messages,
        )

        return {
            "id": response.id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content[0].text,
                    },
                    "finish_reason": response.stop_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }

    async def _stream_anthropic(self, request: GatewayRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Anthropic API."""
        import anthropic
        from backend.core.config import settings

        client = anthropic.AsyncAnthropic(
            api_key=getattr(settings, "ANTHROPIC_API_KEY", None)
        )

        # Convert messages to Anthropic format
        system_message = None
        messages = []
        for m in request.messages:
            if m["role"] == "system":
                system_message = m["content"]
            else:
                messages.append({
                    "role": m["role"],
                    "content": m["content"],
                })

        async with client.messages.stream(
            model=request.model,
            max_tokens=request.max_tokens or 4096,
            system=system_message,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }

    async def _call_ollama(self, request: GatewayRequest) -> Dict[str, Any]:
        """Call Ollama API."""
        import httpx
        from backend.core.config import settings

        ollama_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": request.model,
                    "messages": request.messages,
                    "options": {
                        "temperature": request.temperature,
                    },
                    "stream": False,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            # Estimate tokens for Ollama (doesn't provide them)
            prompt_tokens = self._estimate_tokens(request.messages)
            completion_tokens = len(data.get("message", {}).get("content", "")) // 4

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "choices": [
                    {
                        "index": 0,
                        "message": data.get("message", {}),
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

    async def _stream_ollama(self, request: GatewayRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Ollama API."""
        import httpx
        from backend.core.config import settings

        ollama_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": request.model,
                    "messages": request.messages,
                    "options": {
                        "temperature": request.temperature,
                    },
                    "stream": True,
                },
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "message" in data:
                            yield {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": data["message"].get("content", ""),
                                        },
                                        "finish_reason": "stop" if data.get("done") else None,
                                    }
                                ],
                            }

    async def get_models(self) -> List[Dict[str, Any]]:
        """Get available models."""
        models = []

        for model, provider in MODEL_PROVIDERS.items():
            models.append({
                "id": model,
                "provider": provider.value,
                "object": "model",
                "owned_by": provider.value,
            })

        return models
