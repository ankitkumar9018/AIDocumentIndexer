"""
AIDocumentIndexer - vLLM Integration
=====================================

vLLM provides high-throughput LLM inference with:
- 2-4x faster inference compared to standard implementations
- Dynamic batching for optimal GPU utilization
- Paged KV caching (PagedAttention) for memory efficiency
- Automatic quantization detection
- OpenAI-compatible API

Phase 68: Integrated for high-performance inference serving.

Usage:
    # As a provider in LiteLLM
    provider = "vllm"
    model = "meta-llama/Llama-3.1-70B-Instruct"

    # Direct usage
    from backend.services.vllm_provider import VLLMProvider
    provider = VLLMProvider()
    response = await provider.generate("What is AI?")
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


# Check if vLLM is available
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_ENGINE_AVAILABLE = True
except ImportError:
    VLLM_ENGINE_AVAILABLE = False
    AsyncEngineArgs = None
    AsyncLLMEngine = None
    SamplingParams = None
    RequestOutput = None


class VLLMMode(str, Enum):
    """vLLM deployment mode."""
    LOCAL_ENGINE = "local_engine"  # Direct vLLM engine
    OPENAI_API = "openai_api"  # vLLM OpenAI-compatible server
    LITELLM = "litellm"  # Through LiteLLM


@dataclass
class VLLMConfig:
    """vLLM configuration."""
    # Server mode (default: try local engine, fallback to API)
    mode: VLLMMode = VLLMMode.OPENAI_API

    # Model settings
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    dtype: str = "auto"  # auto, float16, bfloat16, float32
    quantization: Optional[str] = None  # awq, gptq, squeezellm, None

    # Server connection (for OPENAI_API mode)
    api_base: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None

    # Generation defaults
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Engine settings (for LOCAL_ENGINE mode)
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    disable_log_stats: bool = True

    @classmethod
    def from_env(cls) -> "VLLMConfig":
        """Create config from environment variables."""
        mode_str = os.getenv("VLLM_MODE", "openai_api")
        try:
            mode = VLLMMode(mode_str)
        except ValueError:
            mode = VLLMMode.OPENAI_API

        return cls(
            mode=mode,
            model=os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            quantization=os.getenv("VLLM_QUANTIZATION"),
            api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY"),
            max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("VLLM_TEMPERATURE", "0.7")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            enable_prefix_caching=os.getenv("VLLM_PREFIX_CACHING", "true").lower() == "true",
        )

    @classmethod
    async def from_admin_settings(cls) -> "VLLMConfig":
        """
        Create config from admin settings with env var fallback.

        Settings hierarchy:
        1. Admin settings (database) - takes priority
        2. Environment variables - fallback

        Returns:
            VLLMConfig instance
        """
        # Start with env var defaults
        config_data = {
            "mode": os.getenv("VLLM_MODE", "openai_api"),
            "model": os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            "dtype": os.getenv("VLLM_DTYPE", "auto"),
            "quantization": os.getenv("VLLM_QUANTIZATION"),
            "api_base": os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
            "api_key": os.getenv("VLLM_API_KEY"),
            "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("VLLM_TEMPERATURE", "0.7")),
            "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            "enable_prefix_caching": os.getenv("VLLM_PREFIX_CACHING", "true").lower() == "true",
            "enabled": os.getenv("VLLM_ENABLED", "false").lower() == "true",
        }

        try:
            from backend.services.settings import get_settings_service
            settings_service = get_settings_service()

            # Load from admin settings (these override env vars if set)
            admin_settings = await settings_service.get_all_settings()

            # Check if vLLM is enabled
            if "llm.vllm_enabled" in admin_settings:
                config_data["enabled"] = admin_settings["llm.vllm_enabled"]

            # Mode
            if admin_settings.get("llm.vllm_mode"):
                config_data["mode"] = admin_settings["llm.vllm_mode"]

            # API connection
            if admin_settings.get("llm.vllm_api_base"):
                config_data["api_base"] = admin_settings["llm.vllm_api_base"]

            if admin_settings.get("llm.vllm_api_key"):
                config_data["api_key"] = admin_settings["llm.vllm_api_key"]

            # Model settings
            if admin_settings.get("llm.vllm_model"):
                config_data["model"] = admin_settings["llm.vllm_model"]

            if admin_settings.get("llm.vllm_tensor_parallel_size"):
                config_data["tensor_parallel_size"] = int(admin_settings["llm.vllm_tensor_parallel_size"])

            if admin_settings.get("llm.vllm_gpu_memory_utilization"):
                config_data["gpu_memory_utilization"] = float(admin_settings["llm.vllm_gpu_memory_utilization"])

            if admin_settings.get("llm.vllm_max_tokens"):
                config_data["max_tokens"] = int(admin_settings["llm.vllm_max_tokens"])

            logger.debug(
                "Loaded vLLM settings from admin",
                enabled=config_data["enabled"],
                mode=config_data["mode"],
                model=config_data["model"],
            )

        except Exception as e:
            logger.warning(
                "Failed to load admin settings for vLLM, using env vars",
                error=str(e),
            )

        # Parse mode
        try:
            mode = VLLMMode(config_data["mode"])
        except ValueError:
            mode = VLLMMode.OPENAI_API

        return cls(
            mode=mode,
            model=config_data["model"],
            tensor_parallel_size=config_data["tensor_parallel_size"],
            dtype=config_data["dtype"],
            quantization=config_data["quantization"],
            api_base=config_data["api_base"],
            api_key=config_data["api_key"],
            max_tokens=config_data["max_tokens"],
            temperature=config_data["temperature"],
            gpu_memory_utilization=config_data["gpu_memory_utilization"],
            enable_prefix_caching=config_data["enable_prefix_caching"],
        )


@dataclass
class VLLMResponse:
    """Response from vLLM generation."""
    text: str
    tokens_generated: int
    prompt_tokens: int
    finish_reason: str
    latency_ms: float
    tokens_per_second: float
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VLLMProvider:
    """
    vLLM provider for high-performance LLM inference.

    Features:
    - 2-4x faster inference through PagedAttention
    - Dynamic batching for optimal throughput
    - Streaming support
    - OpenAI-compatible API

    Usage:
        provider = VLLMProvider()

        # Single generation
        response = await provider.generate(
            prompt="Explain machine learning",
            max_tokens=500,
        )

        # Streaming
        async for chunk in provider.generate_stream("Tell me a story"):
            print(chunk, end="")

        # Batch processing
        responses = await provider.generate_batch([
            "What is AI?",
            "What is ML?",
        ])
    """

    def __init__(self, config: Optional[VLLMConfig] = None):
        """
        Initialize vLLM provider.

        Args:
            config: vLLM configuration (defaults to env vars)
        """
        self.config = config or VLLMConfig.from_env()
        self._engine: Optional[Any] = None
        self._client: Optional[Any] = None
        self._initialized = False

        logger.info(
            "vLLM provider initialized",
            mode=self.config.mode.value,
            model=self.config.model,
        )

    async def _ensure_initialized(self) -> None:
        """Ensure the provider is initialized."""
        if self._initialized:
            return

        if self.config.mode == VLLMMode.LOCAL_ENGINE:
            await self._init_local_engine()
        elif self.config.mode == VLLMMode.OPENAI_API:
            await self._init_openai_client()
        elif self.config.mode == VLLMMode.LITELLM:
            await self._init_litellm()

        self._initialized = True

    async def _init_local_engine(self) -> None:
        """Initialize local vLLM engine."""
        if not VLLM_ENGINE_AVAILABLE:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )

        logger.info("Initializing vLLM local engine", model=self.config.model)

        engine_args = AsyncEngineArgs(
            model=self.config.model,
            tensor_parallel_size=self.config.tensor_parallel_size,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enforce_eager=self.config.enforce_eager,
            enable_prefix_caching=self.config.enable_prefix_caching,
            disable_log_stats=self.config.disable_log_stats,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM local engine initialized")

    async def _init_openai_client(self) -> None:
        """Initialize OpenAI-compatible client for vLLM server."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for vLLM API mode")

        self._client = httpx.AsyncClient(
            base_url=self.config.api_base,
            timeout=300.0,
            headers={
                "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else "",
                "Content-Type": "application/json",
            },
        )

        # Verify connection
        try:
            response = await self._client.get("/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                logger.info(
                    "Connected to vLLM server",
                    api_base=self.config.api_base,
                    available_models=[m.get("id") for m in models],
                )
            else:
                logger.warning(
                    "vLLM server responded but may not be fully ready",
                    status_code=response.status_code,
                )
        except Exception as e:
            logger.warning(
                "Could not verify vLLM server connection",
                error=str(e),
            )

    async def _init_litellm(self) -> None:
        """Initialize LiteLLM integration."""
        try:
            import litellm
        except ImportError:
            raise ImportError("litellm required for LiteLLM mode")

        # Configure LiteLLM to use vLLM server
        litellm.api_base = self.config.api_base
        if self.config.api_key:
            litellm.api_key = self.config.api_key

        logger.info(
            "LiteLLM configured for vLLM",
            api_base=self.config.api_base,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> VLLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            VLLMResponse with generated text
        """
        await self._ensure_initialized()
        start_time = time.time()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p

        if self.config.mode == VLLMMode.LOCAL_ENGINE:
            response = await self._generate_local(
                prompt, max_tokens, temperature, top_p, stop
            )
        else:
            response = await self._generate_api(
                prompt, max_tokens, temperature, top_p, stop
            )

        latency_ms = (time.time() - start_time) * 1000
        response.latency_ms = latency_ms
        response.tokens_per_second = (
            response.tokens_generated / (latency_ms / 1000)
            if latency_ms > 0 else 0
        )

        logger.debug(
            "vLLM generation complete",
            tokens=response.tokens_generated,
            latency_ms=round(latency_ms, 2),
            tokens_per_sec=round(response.tokens_per_second, 1),
        )

        return response

    async def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> VLLMResponse:
        """Generate using local vLLM engine."""
        import uuid

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            top_k=self.config.top_k if self.config.top_k > 0 else -1,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
        )

        request_id = str(uuid.uuid4())

        # Generate
        results = []
        async for output in self._engine.generate(prompt, sampling_params, request_id):
            results.append(output)

        if not results:
            return VLLMResponse(
                text="",
                tokens_generated=0,
                prompt_tokens=0,
                finish_reason="error",
                latency_ms=0,
                tokens_per_second=0,
                model=self.config.model,
            )

        final_output = results[-1]
        completion = final_output.outputs[0]

        return VLLMResponse(
            text=completion.text,
            tokens_generated=len(completion.token_ids),
            prompt_tokens=len(final_output.prompt_token_ids),
            finish_reason=completion.finish_reason or "stop",
            latency_ms=0,  # Will be filled in
            tokens_per_second=0,  # Will be filled in
            model=self.config.model,
        )

    async def _generate_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> VLLMResponse:
        """Generate using OpenAI-compatible API."""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if stop:
            payload["stop"] = stop

        response = await self._client.post("/chat/completions", json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM API error ({response.status_code}): {response.text}"
            )

        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return VLLMResponse(
            text=choice["message"]["content"],
            tokens_generated=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=0,  # Will be filled in
            tokens_per_second=0,  # Will be filled in
            model=data.get("model", self.config.model),
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        await self._ensure_initialized()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        if self.config.mode == VLLMMode.LOCAL_ENGINE:
            async for chunk in self._stream_local(prompt, max_tokens, temperature):
                yield chunk
        else:
            async for chunk in self._stream_api(prompt, max_tokens, temperature):
                yield chunk

    async def _stream_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream using local vLLM engine."""
        import uuid

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
        )

        request_id = str(uuid.uuid4())
        prev_text = ""

        async for output in self._engine.generate(prompt, sampling_params, request_id):
            completion = output.outputs[0]
            new_text = completion.text[len(prev_text):]
            prev_text = completion.text

            if new_text:
                yield new_text

    async def _stream_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream using OpenAI-compatible API."""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    import json
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_concurrent: int = 10,
    ) -> List[VLLMResponse]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent requests

        Returns:
            List of VLLMResponse objects
        """
        await self._ensure_initialized()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_one(prompt: str) -> VLLMResponse:
            async with semaphore:
                return await self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

        tasks = [generate_one(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch generation error",
                    prompt_index=i,
                    error=str(result),
                )
                processed_results.append(VLLMResponse(
                    text="",
                    tokens_generated=0,
                    prompt_tokens=0,
                    finish_reason="error",
                    latency_ms=0,
                    tokens_per_second=0,
                    model=self.config.model,
                    metadata={"error": str(result)},
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def health_check(self) -> Dict[str, Any]:
        """Check vLLM provider health."""
        try:
            await self._ensure_initialized()

            if self.config.mode == VLLMMode.LOCAL_ENGINE:
                return {
                    "status": "healthy",
                    "mode": "local_engine",
                    "model": self.config.model,
                    "engine_available": self._engine is not None,
                }
            else:
                # Check API health
                response = await self._client.get("/models")
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "mode": self.config.mode.value,
                        "api_base": self.config.api_base,
                        "models": [m.get("id") for m in response.json().get("data", [])],
                    }
                else:
                    return {
                        "status": "degraded",
                        "mode": self.config.mode.value,
                        "error": f"API returned {response.status_code}",
                    }

        except Exception as e:
            return {
                "status": "error",
                "mode": self.config.mode.value if self.config else "unknown",
                "error": str(e),
            }

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        await self._ensure_initialized()

        if self.config.mode == VLLMMode.LOCAL_ENGINE and self._engine:
            model_config = self._engine.engine.model_config

            return {
                "model": self.config.model,
                "max_model_len": model_config.max_model_len,
                "dtype": str(model_config.dtype),
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "quantization": self.config.quantization,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
            }
        else:
            # Try to get info from API
            try:
                response = await self._client.get(f"/models/{self.config.model}")
                if response.status_code == 200:
                    return response.json()
            except Exception:
                pass

            return {
                "model": self.config.model,
                "mode": self.config.mode.value,
                "api_base": self.config.api_base,
            }

    async def close(self) -> None:
        """Close provider resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

        self._engine = None
        self._initialized = False
        logger.info("vLLM provider closed")


# Singleton instance
_vllm_provider: Optional[VLLMProvider] = None


async def get_vllm_provider(config: Optional[VLLMConfig] = None) -> VLLMProvider:
    """
    Get or create vLLM provider singleton.

    Configuration is loaded from admin settings with env var fallback.

    Args:
        config: Optional explicit config (overrides admin settings)

    Returns:
        VLLMProvider instance
    """
    global _vllm_provider

    if _vllm_provider is None:
        if config is None:
            # Load from admin settings with env var fallback
            config = await VLLMConfig.from_admin_settings()
        _vllm_provider = VLLMProvider(config)

    return _vllm_provider


def reset_vllm_provider() -> None:
    """Reset the vLLM provider singleton to reload settings."""
    global _vllm_provider
    if _vllm_provider is not None:
        # Close existing provider
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_vllm_provider.close())
            else:
                loop.run_until_complete(_vllm_provider.close())
        except Exception:
            pass
    _vllm_provider = None
    logger.info("vLLM provider reset, will reload settings on next use")


async def get_vllm_litellm_config() -> Dict[str, Any]:
    """
    Get LiteLLM configuration for vLLM.

    Use this to configure LiteLLM to route requests to vLLM server.
    Configuration is loaded from admin settings with env var fallback.

    Returns:
        Dict with LiteLLM configuration
    """
    config = await VLLMConfig.from_admin_settings()

    return {
        "model": f"openai/{config.model}",
        "api_base": config.api_base,
        "api_key": config.api_key or "dummy",  # vLLM doesn't always require key
        "custom_llm_provider": "openai",  # Use OpenAI-compatible interface
    }


async def register_vllm_with_litellm() -> None:
    """
    Register vLLM as a custom provider in LiteLLM.

    Call this during application startup to enable vLLM through LiteLLM.
    Configuration is loaded from admin settings with env var fallback.
    """
    try:
        import litellm

        config = await VLLMConfig.from_admin_settings()

        # Set as a custom OpenAI-compatible endpoint
        litellm.api_base = config.api_base
        if config.api_key:
            litellm.api_key = config.api_key

        logger.info(
            "vLLM registered with LiteLLM",
            api_base=config.api_base,
            model=config.model,
        )

    except ImportError:
        logger.warning("LiteLLM not installed, skipping vLLM registration")
