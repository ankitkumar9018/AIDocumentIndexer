"""
AIDocumentIndexer - Image Generator Service
============================================

Service for generating images for document content.
Supports multiple backends:
1. Ollama with Stable Diffusion (local, free)
2. Unsplash placeholder images (fallback, free)

Image generation is DISABLED by default and must be explicitly enabled.
"""

import os
import aiohttp
import asyncio
import tempfile
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ImageBackend(str, Enum):
    """Available image generation backends."""
    AUTOMATIC1111 = "automatic1111"  # Local Stable Diffusion WebUI
    OPENAI = "openai"  # OpenAI DALL-E
    STABILITY = "stability"  # Stability AI
    UNSPLASH = "unsplash"  # Free placeholder images
    DISABLED = "disabled"


@dataclass
class ImageGeneratorConfig:
    """Configuration for image generation."""
    enabled: bool = False  # Disabled by default
    backend: ImageBackend = ImageBackend.UNSPLASH  # Default to Unsplash (always works)

    # Automatic1111 / Stable Diffusion WebUI settings
    # Install: https://github.com/AUTOMATIC1111/stable-diffusion-webui
    # Run with: ./webui.sh --api
    sd_webui_url: str = "http://localhost:7860"
    sd_model: str = ""  # Use default model
    sd_sampler: str = "DPM++ 2M Karras"
    sd_steps: int = 20

    # OpenAI DALL-E settings
    openai_api_key: str = ""  # Set via env OPENAI_API_KEY
    openai_model: str = "dall-e-3"  # dall-e-3 or dall-e-2
    openai_quality: str = "standard"  # standard or hd
    openai_style: str = "vivid"  # vivid or natural

    # Stability AI settings
    stability_api_key: str = ""  # Set via env STABILITY_API_KEY
    stability_engine: str = "stable-diffusion-xl-1024-v1-0"

    # Unsplash settings (fallback - always works, no setup needed)
    unsplash_base_url: str = "https://source.unsplash.com"

    # Output settings
    output_dir: str = "/tmp/generated_images"
    default_width: int = 800
    default_height: int = 600

    # Request settings
    timeout_seconds: int = 120
    max_retries: int = 2


@dataclass
class GeneratedImage:
    """A generated image result."""
    path: str  # Local file path
    width: int
    height: int
    prompt: str
    backend: ImageBackend
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ImageGeneratorService:
    """
    Service for generating images for document content.

    Primary backend: Ollama with Stable Diffusion models
    Fallback: Unsplash placeholder images

    Note: This feature is DISABLED by default for cost/resource reasons.
    """

    def __init__(self, config: Optional[ImageGeneratorConfig] = None):
        """Initialize image generator service."""
        self.config = config or ImageGeneratorConfig()

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Image generator initialized",
            enabled=self.config.enabled,
            backend=self.config.backend.value,
        )

    async def generate_for_section(
        self,
        section_title: str,
        section_content: str,
        document_title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[GeneratedImage]:
        """
        Generate an image for a document section.

        Args:
            section_title: Title of the section
            section_content: Content of the section (used for context)
            document_title: Optional document title for additional context
            width: Image width (default from config)
            height: Image height (default from config)

        Returns:
            GeneratedImage if successful, None if disabled or failed
        """
        if not self.config.enabled:
            logger.debug("Image generation is disabled")
            return None

        width = width or self.config.default_width
        height = height or self.config.default_height

        # Create prompt from section content
        prompt = self._create_prompt(section_title, section_content, document_title)

        logger.info(
            "Generating image",
            section_title=section_title,
            prompt_preview=prompt[:100],
            backend=self.config.backend.value,
        )

        # Try primary backend
        result = await self._generate_with_backend(
            prompt=prompt,
            width=width,
            height=height,
            backend=self.config.backend,
        )

        # Fall back to Unsplash if primary fails
        if not result or not result.success:
            if self.config.backend != ImageBackend.UNSPLASH:
                logger.warning(
                    "Primary backend failed, falling back to Unsplash",
                    primary_backend=self.config.backend.value,
                )
                result = await self._generate_with_backend(
                    prompt=prompt,
                    width=width,
                    height=height,
                    backend=ImageBackend.UNSPLASH,
                )

        return result

    async def generate_batch(
        self,
        sections: List[Tuple[str, str]],  # List of (title, content) tuples
        document_title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> List[Optional[GeneratedImage]]:
        """
        Generate images for multiple sections.

        Args:
            sections: List of (title, content) tuples
            document_title: Optional document title
            width: Image width
            height: Image height

        Returns:
            List of GeneratedImage results (may contain None for failures)
        """
        if not self.config.enabled:
            return [None] * len(sections)

        tasks = [
            self.generate_for_section(
                section_title=title,
                section_content=content,
                document_title=document_title,
                width=width,
                height=height,
            )
            for title, content in sections
        ]

        return await asyncio.gather(*tasks)

    def _create_prompt(
        self,
        section_title: str,
        section_content: str,
        document_title: Optional[str] = None,
    ) -> str:
        """Create an image generation prompt from section content."""
        # Extract key terms from content
        words = section_content.split()[:50]  # First 50 words
        content_preview = " ".join(words)

        # Build professional prompt
        context = document_title or "business document"
        prompt = (
            f"Professional presentation image for: {section_title}. "
            f"Context: {context}. "
            f"Style: clean, modern, corporate, high-quality illustration. "
            f"Theme: {content_preview[:100]}"
        )

        return prompt

    async def _generate_with_backend(
        self,
        prompt: str,
        width: int,
        height: int,
        backend: ImageBackend,
    ) -> Optional[GeneratedImage]:
        """Generate image using specified backend."""
        if backend == ImageBackend.DISABLED:
            return None
        elif backend == ImageBackend.AUTOMATIC1111:
            return await self._generate_automatic1111(prompt, width, height)
        elif backend == ImageBackend.OPENAI:
            return await self._generate_openai(prompt, width, height)
        elif backend == ImageBackend.STABILITY:
            return await self._generate_stability(prompt, width, height)
        elif backend == ImageBackend.UNSPLASH:
            return await self._generate_unsplash(prompt, width, height)
        else:
            logger.error(f"Unknown backend: {backend}")
            return None

    async def _generate_automatic1111(
        self,
        prompt: str,
        width: int,
        height: int,
    ) -> Optional[GeneratedImage]:
        """Generate image using Automatic1111 Stable Diffusion WebUI.

        Requires Stable Diffusion WebUI running with --api flag.
        Install: https://github.com/AUTOMATIC1111/stable-diffusion-webui
        Run with: ./webui.sh --api
        """
        try:
            import base64

            async with aiohttp.ClientSession() as session:
                # Check if SD WebUI is available
                try:
                    async with session.get(
                        f"{self.config.sd_webui_url}/sdapi/v1/sd-models",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        if response.status != 200:
                            logger.warning("Stable Diffusion WebUI not available")
                            return GeneratedImage(
                                path="",
                                width=width,
                                height=height,
                                prompt=prompt,
                                backend=ImageBackend.AUTOMATIC1111,
                                success=False,
                                error="Stable Diffusion WebUI not available. Run with: ./webui.sh --api",
                            )
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Could not connect to SD WebUI: {e}")
                    return GeneratedImage(
                        path="",
                        width=width,
                        height=height,
                        prompt=prompt,
                        backend=ImageBackend.AUTOMATIC1111,
                        success=False,
                        error=f"SD WebUI connection failed: {str(e)}",
                    )

                # Build txt2img payload
                payload = {
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
                    "width": width,
                    "height": height,
                    "steps": self.config.sd_steps,
                    "sampler_name": self.config.sd_sampler,
                    "cfg_scale": 7,
                    "seed": -1,  # Random seed
                }

                # Optionally set model
                if self.config.sd_model:
                    payload["override_settings"] = {
                        "sd_model_checkpoint": self.config.sd_model
                    }

                async with session.post(
                    f"{self.config.sd_webui_url}/sdapi/v1/txt2img",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return GeneratedImage(
                            path="",
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.AUTOMATIC1111,
                            success=False,
                            error=f"SD WebUI generation failed: {error_text}",
                        )

                    result = await response.json()

                    # Check for images in response
                    if "images" in result and result["images"]:
                        # Save first image to file
                        image_data = base64.b64decode(result["images"][0])
                        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                        filename = f"sd_{prompt_hash}.png"
                        filepath = os.path.join(self.config.output_dir, filename)

                        with open(filepath, "wb") as f:
                            f.write(image_data)

                        logger.info("Image generated with Stable Diffusion", path=filepath)

                        return GeneratedImage(
                            path=filepath,
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.AUTOMATIC1111,
                            success=True,
                            metadata={"seed": result.get("seed", -1)},
                        )
                    else:
                        return GeneratedImage(
                            path="",
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.AUTOMATIC1111,
                            success=False,
                            error="No image in SD WebUI response",
                        )

        except Exception as e:
            logger.error(f"Stable Diffusion image generation error: {e}")
            return GeneratedImage(
                path="",
                width=width,
                height=height,
                prompt=prompt,
                backend=ImageBackend.AUTOMATIC1111,
                success=False,
                error=str(e),
            )

    async def _generate_unsplash(
        self,
        prompt: str,
        width: int,
        height: int,
    ) -> Optional[GeneratedImage]:
        """Generate placeholder image from Unsplash."""
        try:
            # Extract keywords from prompt
            words = prompt.lower().split()
            # Filter to meaningful keywords
            stop_words = {
                "professional", "image", "for", "the", "a", "an", "and", "or",
                "of", "to", "in", "on", "at", "by", "with", "from", "as",
                "is", "it", "be", "was", "were", "been", "being", "have",
                "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "must", "shall", "can", "presentation",
                "style", "clean", "modern", "corporate", "high-quality",
                "illustration", "context", "theme",
            }
            keywords = [w for w in words if w not in stop_words and len(w) > 2][:5]
            keyword_str = ",".join(keywords) if keywords else "business,office"

            # Build Unsplash URL
            url = f"{self.config.unsplash_base_url}/{width}x{height}/?{keyword_str}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                    allow_redirects=True,
                ) as response:
                    if response.status != 200:
                        return GeneratedImage(
                            path="",
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.UNSPLASH,
                            success=False,
                            error=f"Unsplash request failed: {response.status}",
                        )

                    # Save image
                    image_data = await response.read()
                    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                    filename = f"unsplash_{prompt_hash}.jpg"
                    filepath = os.path.join(self.config.output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    logger.info(
                        "Image fetched from Unsplash",
                        path=filepath,
                        keywords=keyword_str,
                    )

                    return GeneratedImage(
                        path=filepath,
                        width=width,
                        height=height,
                        prompt=prompt,
                        backend=ImageBackend.UNSPLASH,
                        success=True,
                        metadata={"keywords": keyword_str},
                    )

        except Exception as e:
            logger.error(f"Unsplash image fetch error: {e}")
            return GeneratedImage(
                path="",
                width=width,
                height=height,
                prompt=prompt,
                backend=ImageBackend.UNSPLASH,
                success=False,
                error=str(e),
            )

    async def _generate_openai(
        self,
        prompt: str,
        width: int,
        height: int,
    ) -> Optional[GeneratedImage]:
        """Generate image using OpenAI DALL-E.

        Requires OPENAI_API_KEY environment variable or config setting.
        """
        import base64

        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not configured")
            return GeneratedImage(
                path="",
                width=width,
                height=height,
                prompt=prompt,
                backend=ImageBackend.OPENAI,
                success=False,
                error="OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
            )

        # DALL-E 3 only supports specific sizes
        if self.config.openai_model == "dall-e-3":
            if width > 1024 and height <= 1024:
                size = "1792x1024"
            elif height > 1024 and width <= 1024:
                size = "1024x1792"
            else:
                size = "1024x1024"
        else:
            # DALL-E 2 sizes
            if width <= 256 and height <= 256:
                size = "256x256"
            elif width <= 512 and height <= 512:
                size = "512x512"
            else:
                size = "1024x1024"

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.config.openai_model,
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                    "response_format": "b64_json",
                }

                # DALL-E 3 specific options
                if self.config.openai_model == "dall-e-3":
                    payload["quality"] = self.config.openai_quality
                    payload["style"] = self.config.openai_style

                async with session.post(
                    "https://api.openai.com/v1/images/generations",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error("OpenAI DALL-E API error", error=error_text)
                        return GeneratedImage(
                            path="",
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.OPENAI,
                            success=False,
                            error=f"OpenAI API error: {error_text}",
                        )

                    result = await response.json()
                    data = result.get("data", [])

                    if not data:
                        return GeneratedImage(
                            path="",
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.OPENAI,
                            success=False,
                            error="No image returned from OpenAI",
                        )

                    # Save image
                    image_data = base64.b64decode(data[0].get("b64_json", ""))
                    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                    filename = f"dalle_{prompt_hash}.png"
                    filepath = os.path.join(self.config.output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    logger.info("Image generated with OpenAI DALL-E", path=filepath)

                    return GeneratedImage(
                        path=filepath,
                        width=int(size.split("x")[0]),
                        height=int(size.split("x")[1]),
                        prompt=prompt,
                        backend=ImageBackend.OPENAI,
                        success=True,
                        metadata={
                            "model": self.config.openai_model,
                            "revised_prompt": data[0].get("revised_prompt"),
                        },
                    )

        except Exception as e:
            logger.error(f"OpenAI DALL-E generation error: {e}")
            return GeneratedImage(
                path="",
                width=width,
                height=height,
                prompt=prompt,
                backend=ImageBackend.OPENAI,
                success=False,
                error=str(e),
            )

    async def _generate_stability(
        self,
        prompt: str,
        width: int,
        height: int,
    ) -> Optional[GeneratedImage]:
        """Generate image using Stability AI.

        Requires STABILITY_API_KEY environment variable or config setting.
        """
        import base64

        api_key = self.config.stability_api_key or os.getenv("STABILITY_API_KEY")
        if not api_key:
            logger.warning("Stability AI API key not configured")
            return GeneratedImage(
                path="",
                width=width,
                height=height,
                prompt=prompt,
                backend=ImageBackend.STABILITY,
                success=False,
                error="Stability AI API key not configured. Set STABILITY_API_KEY environment variable.",
            )

        # Stability requires specific dimensions
        # Round to nearest 64
        width = (width // 64) * 64
        height = (height // 64) * 64
        width = max(512, min(width, 2048))
        height = max(512, min(height, 2048))

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text_prompts": [{"text": prompt, "weight": 1.0}],
                    "cfg_scale": 7,
                    "height": height,
                    "width": width,
                    "samples": 1,
                    "steps": 30,
                }

                url = f"https://api.stability.ai/v1/generation/{self.config.stability_engine}/text-to-image"

                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error("Stability AI API error", error=error_text)
                        return GeneratedImage(
                            path="",
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.STABILITY,
                            success=False,
                            error=f"Stability AI API error: {error_text}",
                        )

                    result = await response.json()
                    artifacts = result.get("artifacts", [])

                    if not artifacts:
                        return GeneratedImage(
                            path="",
                            width=width,
                            height=height,
                            prompt=prompt,
                            backend=ImageBackend.STABILITY,
                            success=False,
                            error="No image returned from Stability AI",
                        )

                    # Save image
                    image_data = base64.b64decode(artifacts[0].get("base64", ""))
                    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                    filename = f"stability_{prompt_hash}.png"
                    filepath = os.path.join(self.config.output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    logger.info("Image generated with Stability AI", path=filepath)

                    return GeneratedImage(
                        path=filepath,
                        width=width,
                        height=height,
                        prompt=prompt,
                        backend=ImageBackend.STABILITY,
                        success=True,
                        metadata={
                            "engine": self.config.stability_engine,
                            "seed": artifacts[0].get("seed"),
                        },
                    )

        except Exception as e:
            logger.error(f"Stability AI generation error: {e}")
            return GeneratedImage(
                path="",
                width=width,
                height=height,
                prompt=prompt,
                backend=ImageBackend.STABILITY,
                success=False,
                error=str(e),
            )

    async def check_sd_webui_available(self) -> bool:
        """Check if Stable Diffusion WebUI server is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.sd_webui_url}/sdapi/v1/sd-models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except:
            return False

    def get_available_backends(self) -> List[dict]:
        """Get list of available (configured) backends."""
        backends = []

        # Check OpenAI
        if self.config.openai_api_key or os.getenv("OPENAI_API_KEY"):
            backends.append({
                "id": ImageBackend.OPENAI.value,
                "name": "OpenAI DALL-E",
                "models": ["dall-e-3", "dall-e-2"],
                "configured": True,
            })

        # Check Stability AI
        if self.config.stability_api_key or os.getenv("STABILITY_API_KEY"):
            backends.append({
                "id": ImageBackend.STABILITY.value,
                "name": "Stability AI",
                "models": ["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6"],
                "configured": True,
            })

        # Automatic1111 is always available if configured
        backends.append({
            "id": ImageBackend.AUTOMATIC1111.value,
            "name": "Stable Diffusion WebUI (Local)",
            "models": ["varies"],
            "configured": True,
            "requires_local_server": True,
        })

        # Unsplash is always available
        backends.append({
            "id": ImageBackend.UNSPLASH.value,
            "name": "Unsplash (Placeholder Images)",
            "models": [],
            "configured": True,
            "free": True,
        })

        return backends


# Singleton instance
_image_service: Optional[ImageGeneratorService] = None


def get_image_generator(
    config: Optional[ImageGeneratorConfig] = None,
) -> ImageGeneratorService:
    """Get or create image generator service instance."""
    global _image_service

    if _image_service is None or config is not None:
        _image_service = ImageGeneratorService(config)

    return _image_service
