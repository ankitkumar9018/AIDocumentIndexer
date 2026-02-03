"""
AIDocumentIndexer - Vision Language Model Processor
=====================================================

Provides VLM integration for visual document understanding.

Supported VLMs:
- Qwen3-VL (recommended): 235B-A22B flagship, 30B-A3B efficient
- GLM-4.6V: 106B with native tool calling
- Claude Vision: Via Anthropic API
- GPT-4 Vision: Via OpenAI API

Features:
- Visual document analysis (charts, tables, infographics)
- OCR with layout understanding
- Multi-image processing
- Structured output extraction (JSON, HTML)
- Fallback chain for reliability

Based on:
- Top VLMs 2026 (https://www.bentoml.com/blog)
- Qwen3-VL (https://github.com/QwenLM/Qwen3-VL)
- GLM-4.6V native tool calling

Usage:
    from backend.services.vlm_processor import get_vlm_processor

    processor = await get_vlm_processor()
    result = await processor.analyze_image(
        image_data,
        prompt="Extract all text and describe the chart"
    )
"""

import asyncio
import base64
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Phase 55: Import audit logging for fallback events
try:
    from backend.services.audit import audit_service_fallback, audit_service_error
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

class VLMProvider(str, Enum):
    """Supported VLM providers."""
    QWEN = "qwen"           # Qwen3-VL via transformers/vLLM
    GLM = "glm"             # GLM-4.6V via API
    CLAUDE = "claude"       # Claude 3.5 Sonnet vision
    OPENAI = "openai"       # GPT-4 Vision
    LOCAL = "local"         # Local model via Ollama


@dataclass
class VLMConfig:
    """Configuration for VLM processor."""
    # Provider selection
    primary_provider: VLMProvider = VLMProvider.CLAUDE
    fallback_providers: List[VLMProvider] = field(
        default_factory=lambda: [VLMProvider.OPENAI]
    )

    # Model settings
    qwen_model: str = "Qwen/Qwen3-VL-7B-Instruct"  # or Qwen3-VL-32B
    glm_model: str = "glm-4.6v"
    claude_model: str = "claude-3-5-sonnet-20241022"
    openai_model: str = "gpt-4o"
    local_model: str = "qwen3-vl"

    # Processing settings
    max_image_size: int = 4096  # pixels
    max_images_per_request: int = 10
    timeout_seconds: float = 60.0

    # Output settings
    output_format: str = "text"  # text, json, html


@dataclass
class VLMResult:
    """Result from VLM processing."""
    content: str
    provider: VLMProvider
    model: str
    structured_data: Optional[Dict[str, Any]] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None
    ocr_text: Optional[str] = None
    confidence: float = 1.0
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    success: bool = True


# =============================================================================
# Base VLM Provider Interface
# =============================================================================

class BaseVLMProvider(ABC):
    """Abstract base class for VLM providers."""

    def __init__(self, config: VLMConfig):
        self.config = config

    @abstractmethod
    async def analyze(
        self,
        images: List[bytes],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """Analyze images with the VLM."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available."""
        pass

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image to base64."""
        return base64.b64encode(image_data).decode('utf-8')

    def _get_image_mime_type(self, image_data: bytes) -> str:
        """Detect image MIME type from bytes."""
        # Check magic bytes
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        elif image_data[:3] == b'\xff\xd8\xff':
            return 'image/jpeg'
        elif image_data[:6] in (b'GIF87a', b'GIF89a'):
            return 'image/gif'
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            return 'image/webp'
        return 'image/png'  # Default


# =============================================================================
# Claude Vision Provider
# =============================================================================

class ClaudeVLMProvider(BaseVLMProvider):
    """Claude 3.5 Sonnet vision provider."""

    async def is_available(self) -> bool:
        """Check if Anthropic API is configured."""
        return bool(settings.ANTHROPIC_API_KEY)

    async def analyze(
        self,
        images: List[bytes],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """Analyze images using Claude vision."""
        import time
        start_time = time.time()

        if not await self.is_available():
            return VLMResult(
                content="",
                provider=VLMProvider.CLAUDE,
                model=self.config.claude_model,
                error="Anthropic API key not configured",
                success=False,
            )

        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

            # Build content with images
            content = []
            for img_data in images[:self.config.max_images_per_request]:
                mime_type = self._get_image_mime_type(img_data)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": self._encode_image(img_data),
                    }
                })

            # Add text prompt
            if output_format == "json":
                prompt = f"{prompt}\n\nRespond with valid JSON only."
            elif output_format == "html":
                prompt = f"{prompt}\n\nFormat the output as HTML."

            content.append({"type": "text", "text": prompt})

            # Call Claude
            response = await client.messages.create(
                model=self.config.claude_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )

            result_text = response.content[0].text

            # Parse structured output if requested
            structured_data = None
            if output_format == "json":
                try:
                    import json
                    structured_data = json.loads(result_text)
                except json.JSONDecodeError:
                    pass  # Invalid JSON, leave structured_data as None

            return VLMResult(
                content=result_text,
                provider=VLMProvider.CLAUDE,
                model=self.config.claude_model,
                structured_data=structured_data,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Claude VLM error", error=str(e))
            return VLMResult(
                content="",
                provider=VLMProvider.CLAUDE,
                model=self.config.claude_model,
                error=str(e),
                success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )


# =============================================================================
# OpenAI Vision Provider
# =============================================================================

class OpenAIVLMProvider(BaseVLMProvider):
    """GPT-4 Vision provider."""

    async def is_available(self) -> bool:
        """Check if OpenAI API is configured."""
        return bool(settings.OPENAI_API_KEY)

    async def analyze(
        self,
        images: List[bytes],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """Analyze images using GPT-4 Vision."""
        import time
        start_time = time.time()

        if not await self.is_available():
            return VLMResult(
                content="",
                provider=VLMProvider.OPENAI,
                model=self.config.openai_model,
                error="OpenAI API key not configured",
                success=False,
            )

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            # Build content with images
            content = []
            for img_data in images[:self.config.max_images_per_request]:
                mime_type = self._get_image_mime_type(img_data)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{self._encode_image(img_data)}",
                        "detail": "high",
                    }
                })

            # Add text prompt
            if output_format == "json":
                prompt = f"{prompt}\n\nRespond with valid JSON only."
            elif output_format == "html":
                prompt = f"{prompt}\n\nFormat the output as HTML."

            content.append({"type": "text", "text": prompt})

            # Call OpenAI
            response = await client.chat.completions.create(
                model=self.config.openai_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )

            result_text = response.choices[0].message.content

            # Parse structured output if requested
            structured_data = None
            if output_format == "json":
                try:
                    import json
                    structured_data = json.loads(result_text)
                except json.JSONDecodeError:
                    pass  # Invalid JSON, leave structured_data as None

            return VLMResult(
                content=result_text,
                provider=VLMProvider.OPENAI,
                model=self.config.openai_model,
                structured_data=structured_data,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("OpenAI VLM error", error=str(e))
            return VLMResult(
                content="",
                provider=VLMProvider.OPENAI,
                model=self.config.openai_model,
                error=str(e),
                success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )


# =============================================================================
# Qwen VLM Provider (Local/vLLM)
# =============================================================================

class QwenVLMProvider(BaseVLMProvider):
    """Qwen3-VL provider via transformers or vLLM."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._available = None

    async def is_available(self) -> bool:
        """Check if Qwen VL is available."""
        if self._available is not None:
            return self._available

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self._available = True
        except ImportError:
            try:
                # Try vLLM
                from vllm import LLM
                self._available = True
            except ImportError:
                self._available = False

        return self._available

    async def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self._processor = AutoProcessor.from_pretrained(
                self.config.qwen_model,
                trust_remote_code=True,
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.qwen_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

            logger.info("Qwen VL model loaded", model=self.config.qwen_model)

        except Exception as e:
            logger.error("Failed to load Qwen VL model", error=str(e))
            raise

    async def analyze(
        self,
        images: List[bytes],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """Analyze images using Qwen VL."""
        import time
        start_time = time.time()

        if not await self.is_available():
            return VLMResult(
                content="",
                provider=VLMProvider.QWEN,
                model=self.config.qwen_model,
                error="Qwen VL not available",
                success=False,
            )

        pil_images = []  # Track for cleanup
        try:
            await self._load_model()

            from PIL import Image

            # Convert bytes to PIL Images
            for img_data in images[:self.config.max_images_per_request]:
                pil_images.append(Image.open(io.BytesIO(img_data)))

            # Build messages
            if output_format == "json":
                prompt = f"{prompt}\n\nRespond with valid JSON only."

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_images],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Process inputs
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=pil_images,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)

            # Generate
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=2048,
            )

            # Decode
            result_text = self._processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0]

            # Parse structured output
            structured_data = None
            if output_format == "json":
                try:
                    import json
                    structured_data = json.loads(result_text)
                except json.JSONDecodeError:
                    pass  # Invalid JSON, leave structured_data as None

            return VLMResult(
                content=result_text,
                provider=VLMProvider.QWEN,
                model=self.config.qwen_model,
                structured_data=structured_data,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Qwen VL error", error=str(e))
            return VLMResult(
                content="",
                provider=VLMProvider.QWEN,
                model=self.config.qwen_model,
                error=str(e),
                success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        finally:
            # Clean up PIL images to prevent memory leaks
            for img in pil_images:
                try:
                    img.close()
                except Exception:
                    pass


# =============================================================================
# GLM-4.6V Provider
# =============================================================================

class GLMVLMProvider(BaseVLMProvider):
    """GLM-4.6V vision provider via ZhipuAI API."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._api_key = os.getenv("ZHIPUAI_API_KEY") or getattr(settings, "ZHIPUAI_API_KEY", None)

    async def is_available(self) -> bool:
        """Check if ZhipuAI API is configured."""
        return bool(self._api_key)

    async def analyze(
        self,
        images: List[bytes],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """Analyze images using GLM-4.6V."""
        import time
        start_time = time.time()

        if not await self.is_available():
            return VLMResult(
                content="",
                provider=VLMProvider.GLM,
                model=self.config.glm_model,
                error="ZhipuAI API key not configured",
                success=False,
            )

        try:
            import httpx

            if output_format == "json":
                prompt = f"{prompt}\n\nRespond with valid JSON only."
            elif output_format == "html":
                prompt = f"{prompt}\n\nFormat the output as HTML."

            # Build content with images
            content = []
            for img_data in images[:self.config.max_images_per_request]:
                mime_type = self._get_image_mime_type(img_data)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{self._encode_image(img_data)}",
                    }
                })

            content.append({"type": "text", "text": prompt})

            # Call ZhipuAI API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.glm_model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 4096,
                    },
                    timeout=self.config.timeout_seconds,
                )

                if response.status_code != 200:
                    raise RuntimeError(f"GLM API error: {response.text}")

                result = response.json()
                result_text = result["choices"][0]["message"]["content"]

            # Parse structured output if requested
            structured_data = None
            if output_format == "json":
                try:
                    import json
                    structured_data = json.loads(result_text)
                except json.JSONDecodeError:
                    pass  # Invalid JSON, leave structured_data as None

            return VLMResult(
                content=result_text,
                provider=VLMProvider.GLM,
                model=self.config.glm_model,
                structured_data=structured_data,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("GLM VLM error", error=str(e))
            return VLMResult(
                content="",
                provider=VLMProvider.GLM,
                model=self.config.glm_model,
                error=str(e),
                success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )


# =============================================================================
# Local VLM Provider (Ollama)
# =============================================================================

class LocalVLMProvider(BaseVLMProvider):
    """Local VLM provider via Ollama."""

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:11434/api/tags",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return any(
                        self.config.local_model in m.get("name", "")
                        for m in models
                    )
        except Exception:
            pass
        return False

    async def analyze(
        self,
        images: List[bytes],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """Analyze images using Ollama."""
        import time
        start_time = time.time()

        if not await self.is_available():
            return VLMResult(
                content="",
                provider=VLMProvider.LOCAL,
                model=self.config.local_model,
                error="Ollama not available or model not installed",
                success=False,
            )

        try:
            import httpx

            if output_format == "json":
                prompt = f"{prompt}\n\nRespond with valid JSON only."

            # Prepare images
            encoded_images = [
                self._encode_image(img)
                for img in images[:self.config.max_images_per_request]
            ]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.config.local_model,
                        "prompt": prompt,
                        "images": encoded_images,
                        "stream": False,
                    },
                    timeout=self.config.timeout_seconds,
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Ollama error: {response.text}")

                result = response.json()
                result_text = result.get("response", "")

            # Parse structured output
            structured_data = None
            if output_format == "json":
                try:
                    import json
                    structured_data = json.loads(result_text)
                except json.JSONDecodeError:
                    pass  # Invalid JSON, leave structured_data as None

            return VLMResult(
                content=result_text,
                provider=VLMProvider.LOCAL,
                model=self.config.local_model,
                structured_data=structured_data,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Local VLM error", error=str(e))
            return VLMResult(
                content="",
                provider=VLMProvider.LOCAL,
                model=self.config.local_model,
                error=str(e),
                success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )


# =============================================================================
# VLM Processor (Main Interface)
# =============================================================================

class VLMProcessor:
    """
    Main VLM processor with fallback chain.

    Provides a unified interface for visual document analysis
    with automatic failover between providers.
    """

    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self._providers: Dict[VLMProvider, BaseVLMProvider] = {}
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize VLM providers."""
        if self._initialized:
            return True

        # Initialize providers
        self._providers = {
            VLMProvider.CLAUDE: ClaudeVLMProvider(self.config),
            VLMProvider.OPENAI: OpenAIVLMProvider(self.config),
            VLMProvider.QWEN: QwenVLMProvider(self.config),
            VLMProvider.GLM: GLMVLMProvider(self.config),
            VLMProvider.LOCAL: LocalVLMProvider(self.config),
        }

        # Check primary provider
        primary = self._providers.get(self.config.primary_provider)
        if primary and await primary.is_available():
            logger.info(
                "VLM processor initialized",
                primary=self.config.primary_provider.value,
            )
        else:
            logger.warning(
                "Primary VLM provider not available",
                primary=self.config.primary_provider.value,
            )

        self._initialized = True
        return True

    async def analyze_image(
        self,
        image: Union[bytes, str, Path],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """
        Analyze a single image.

        Args:
            image: Image bytes, file path, or base64 string
            prompt: Analysis prompt
            output_format: Output format (text, json, html)

        Returns:
            VLMResult with analysis
        """
        return await self.analyze_images([image], prompt, output_format)

    async def analyze_images(
        self,
        images: List[Union[bytes, str, Path]],
        prompt: str,
        output_format: str = "text",
    ) -> VLMResult:
        """
        Analyze multiple images.

        Args:
            images: List of image bytes, file paths, or base64 strings
            prompt: Analysis prompt
            output_format: Output format (text, json, html)

        Returns:
            VLMResult with analysis
        """
        await self.initialize()

        # Convert images to bytes
        image_bytes = []
        for img in images:
            if isinstance(img, bytes):
                image_bytes.append(img)
            elif isinstance(img, (str, Path)):
                path = Path(img)
                if path.exists():
                    image_bytes.append(path.read_bytes())
                else:
                    # Assume base64
                    image_bytes.append(base64.b64decode(str(img)))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        # Try primary provider
        providers_to_try = [self.config.primary_provider] + self.config.fallback_providers

        last_error = None
        last_provider = None

        for i, provider_type in enumerate(providers_to_try):
            provider = self._providers.get(provider_type)
            if provider is None:
                continue

            if not await provider.is_available():
                logger.debug(f"VLM provider {provider_type.value} not available")
                continue

            result = await provider.analyze(image_bytes, prompt, output_format)

            if result.success:
                # Phase 55: Log fallback if this wasn't the primary provider
                if i > 0 and AUDIT_AVAILABLE and last_provider:
                    try:
                        await audit_service_fallback(
                            service_type="vlm",
                            primary_provider=last_provider,
                            fallback_provider=provider_type.value,
                            error_message=last_error or "Provider failed",
                            context={"image_count": len(image_bytes), "prompt_length": len(prompt)},
                        )
                    except Exception:
                        pass  # Don't let audit logging break VLM
                return result

            logger.warning(
                f"VLM provider {provider_type.value} failed",
                error=result.error,
            )
            last_error = result.error
            last_provider = provider_type.value

        # Phase 55: Log error when all providers failed
        if AUDIT_AVAILABLE:
            try:
                failed_providers = ", ".join(p.value for p in providers_to_try)
                await audit_service_error(
                    service_type="vlm",
                    provider=failed_providers,
                    error_message=last_error or "All VLM providers failed",
                    context={"image_count": len(image_bytes), "prompt_length": len(prompt)},
                )
            except Exception:
                pass  # Don't let audit logging break VLM

        # All providers failed
        return VLMResult(
            content="",
            provider=self.config.primary_provider,
            model="",
            error="All VLM providers failed",
            success=False,
        )

    async def extract_text(
        self,
        image: Union[bytes, str, Path],
    ) -> VLMResult:
        """
        Extract text from image (OCR).

        Args:
            image: Image data

        Returns:
            VLMResult with extracted text
        """
        prompt = """Extract all text from this image.
Preserve the layout and structure as much as possible.
Include any text in tables, charts, or diagrams.
Return the extracted text only, no explanations."""

        result = await self.analyze_image(image, prompt)
        if result.success:
            result.ocr_text = result.content
        return result

    async def describe_chart(
        self,
        image: Union[bytes, str, Path],
    ) -> VLMResult:
        """
        Describe a chart or graph.

        Args:
            image: Chart/graph image

        Returns:
            VLMResult with chart description and data
        """
        prompt = """Analyze this chart or graph and provide:
1. Type of chart (bar, line, pie, etc.)
2. Title and axis labels
3. Key data points and values
4. Trends or insights

Return as JSON with keys: chart_type, title, x_axis, y_axis, data_points, insights"""

        return await self.analyze_image(image, prompt, output_format="json")

    async def extract_table(
        self,
        image: Union[bytes, str, Path],
    ) -> VLMResult:
        """
        Extract table data from image.

        Args:
            image: Table image

        Returns:
            VLMResult with table data
        """
        prompt = """Extract the table from this image.
Return the data as a JSON array of objects where each object represents a row.
Use the header row values as keys.
Preserve all cell values exactly as shown."""

        return await self.analyze_image(image, prompt, output_format="json")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of VLM providers."""
        await self.initialize()

        status = {
            "initialized": self._initialized,
            "primary_provider": self.config.primary_provider.value,
            "providers": {},
        }

        for provider_type, provider in self._providers.items():
            available = await provider.is_available()
            status["providers"][provider_type.value] = {
                "available": available,
            }

        return status


# =============================================================================
# Global Processor
# =============================================================================

_vlm_processor: Optional[VLMProcessor] = None
_processor_lock = asyncio.Lock()


async def get_vlm_processor(
    config: Optional[VLMConfig] = None,
) -> VLMProcessor:
    """Get or create VLM processor singleton."""
    global _vlm_processor

    if _vlm_processor is not None:
        return _vlm_processor

    async with _processor_lock:
        if _vlm_processor is not None:
            return _vlm_processor

        # Build config from settings
        if config is None:
            vlm_model = getattr(settings, 'VLM_MODEL', 'claude')
            provider_map = {
                'qwen': VLMProvider.QWEN,
                'qwen3-vl': VLMProvider.QWEN,
                'glm': VLMProvider.GLM,
                'glm-4.6v': VLMProvider.GLM,
                'claude': VLMProvider.CLAUDE,
                'openai': VLMProvider.OPENAI,
                'gpt-4o': VLMProvider.OPENAI,
                'local': VLMProvider.LOCAL,
            }
            primary = provider_map.get(vlm_model.lower(), VLMProvider.CLAUDE)
            config = VLMConfig(primary_provider=primary)

        _vlm_processor = VLMProcessor(config)
        await _vlm_processor.initialize()

        return _vlm_processor


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "VLMProvider",
    "VLMConfig",
    "VLMResult",
    "VLMProcessor",
    "get_vlm_processor",
]
