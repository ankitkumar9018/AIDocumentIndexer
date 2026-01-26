"""
AIDocumentIndexer - Amazon Nova Multimodal Embeddings
======================================================

Phase 68: First unified embedding model for text + images + video + audio.

Key Features:
- Cross-modal retrieval (search images with text, text with images)
- Unified embedding space for all modalities
- AWS Bedrock integration
- Support for multimodal documents (PDFs with images, videos, audio)

Benchmarks:
- Outperforms CLIP on image-text retrieval
- Unified representation reduces storage complexity
- Native video/audio understanding

Based on Amazon Nova Embeddings announcement (Dec 2025).
"""

import asyncio
import base64
import hashlib
import io
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class NovaModalityType(str, Enum):
    """Supported modality types for Nova embeddings."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class NovaModelVariant(str, Enum):
    """Nova embedding model variants."""
    NOVA_EMBED_V1 = "amazon.nova-embed-v1"  # Text + Image
    NOVA_EMBED_MULTIMODAL = "amazon.nova-embed-multimodal-v1"  # All modalities
    NOVA_EMBED_LITE = "amazon.nova-embed-lite-v1"  # Faster, lower dimension


@dataclass
class NovaEmbeddingConfig:
    """Configuration for Nova embeddings."""

    # Model selection
    model_id: str = "amazon.nova-embed-multimodal-v1"

    # AWS credentials
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_profile: Optional[str] = None

    # Embedding settings
    embedding_dimension: int = 1024  # Nova default
    normalize_embeddings: bool = True

    # Batch settings
    max_batch_size: int = 16
    max_text_length: int = 8192
    max_image_size_mb: int = 5
    max_video_length_seconds: int = 30
    max_audio_length_seconds: int = 60

    # Performance
    concurrent_requests: int = 4
    timeout_seconds: int = 60
    retry_attempts: int = 3

    @classmethod
    def from_admin_settings(cls) -> "NovaEmbeddingConfig":
        """Load config from admin settings with env var fallback."""
        return cls(
            model_id=os.getenv("NOVA_EMBEDDING_MODEL", "amazon.nova-embed-multimodal-v1"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_profile=os.getenv("AWS_PROFILE"),
            embedding_dimension=int(os.getenv("NOVA_EMBEDDING_DIM", "1024")),
        )


@dataclass
class NovaEmbeddingResult:
    """Result from Nova embedding."""
    embedding: List[float]
    modality: NovaModalityType
    input_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Nova Embeddings Service
# =============================================================================

class NovaEmbeddingService:
    """
    Amazon Nova multimodal embedding service.

    Provides unified embeddings for text, images, video, and audio
    via AWS Bedrock.

    Usage:
        service = NovaEmbeddingService()
        await service.initialize()

        # Text embedding
        text_emb = await service.embed_text("What is machine learning?")

        # Image embedding
        image_emb = await service.embed_image(image_bytes)

        # Cross-modal search
        results = await service.cross_modal_search(
            query_text="sunset over ocean",
            candidates=[image1_bytes, image2_bytes],
        )
    """

    def __init__(self, config: Optional[NovaEmbeddingConfig] = None):
        self.config = config or NovaEmbeddingConfig.from_admin_settings()
        self._client = None
        self._initialized = False
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def initialize(self) -> None:
        """Initialize the Bedrock client."""
        if self._initialized:
            return

        try:
            import boto3
            from botocore.config import Config as BotoConfig

            # Configure retry and timeout
            boto_config = BotoConfig(
                retries={"max_attempts": self.config.retry_attempts},
                read_timeout=self.config.timeout_seconds,
                connect_timeout=30,
            )

            # Create session
            session_kwargs = {"region_name": self.config.aws_region}

            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
            elif self.config.aws_profile:
                session_kwargs["profile_name"] = self.config.aws_profile

            session = boto3.Session(**session_kwargs)

            # Create Bedrock runtime client
            self._client = session.client(
                "bedrock-runtime",
                config=boto_config,
            )

            self._semaphore = asyncio.Semaphore(self.config.concurrent_requests)
            self._initialized = True

            logger.info(
                "Nova embedding service initialized",
                model=self.config.model_id,
                region=self.config.aws_region,
            )

        except ImportError:
            logger.error("boto3 not installed. Run: pip install boto3")
            raise
        except Exception as e:
            logger.error("Failed to initialize Nova embeddings", error=str(e))
            raise

    async def embed_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NovaEmbeddingResult:
        """
        Generate embedding for text.

        Args:
            text: Input text (max 8192 tokens)
            metadata: Optional metadata to include in result

        Returns:
            NovaEmbeddingResult with embedding vector
        """
        if not self._initialized:
            await self.initialize()

        # Truncate if needed
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]

        input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        async with self._semaphore:
            embedding = await self._invoke_model(
                input_type="text",
                input_data=text,
            )

        return NovaEmbeddingResult(
            embedding=embedding,
            modality=NovaModalityType.TEXT,
            input_hash=input_hash,
            metadata=metadata or {},
        )

    async def embed_image(
        self,
        image_data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NovaEmbeddingResult:
        """
        Generate embedding for an image.

        Args:
            image_data: Image bytes (PNG, JPEG, GIF, WEBP)
            metadata: Optional metadata to include in result

        Returns:
            NovaEmbeddingResult with embedding vector
        """
        if not self._initialized:
            await self.initialize()

        # Check size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > self.config.max_image_size_mb:
            raise ValueError(f"Image too large: {size_mb:.1f}MB > {self.config.max_image_size_mb}MB")

        input_hash = hashlib.sha256(image_data).hexdigest()[:16]

        async with self._semaphore:
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode()

            embedding = await self._invoke_model(
                input_type="image",
                input_data=image_base64,
            )

        return NovaEmbeddingResult(
            embedding=embedding,
            modality=NovaModalityType.IMAGE,
            input_hash=input_hash,
            metadata=metadata or {},
        )

    async def embed_video(
        self,
        video_data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NovaEmbeddingResult:
        """
        Generate embedding for a video.

        Note: Video is processed by extracting key frames.

        Args:
            video_data: Video bytes (MP4, MOV, AVI)
            metadata: Optional metadata to include in result

        Returns:
            NovaEmbeddingResult with embedding vector
        """
        if not self._initialized:
            await self.initialize()

        input_hash = hashlib.sha256(video_data).hexdigest()[:16]

        async with self._semaphore:
            video_base64 = base64.b64encode(video_data).decode()

            embedding = await self._invoke_model(
                input_type="video",
                input_data=video_base64,
            )

        return NovaEmbeddingResult(
            embedding=embedding,
            modality=NovaModalityType.VIDEO,
            input_hash=input_hash,
            metadata=metadata or {},
        )

    async def embed_audio(
        self,
        audio_data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NovaEmbeddingResult:
        """
        Generate embedding for audio.

        Args:
            audio_data: Audio bytes (MP3, WAV, FLAC)
            metadata: Optional metadata to include in result

        Returns:
            NovaEmbeddingResult with embedding vector
        """
        if not self._initialized:
            await self.initialize()

        input_hash = hashlib.sha256(audio_data).hexdigest()[:16]

        async with self._semaphore:
            audio_base64 = base64.b64encode(audio_data).decode()

            embedding = await self._invoke_model(
                input_type="audio",
                input_data=audio_base64,
            )

        return NovaEmbeddingResult(
            embedding=embedding,
            modality=NovaModalityType.AUDIO,
            input_hash=input_hash,
            metadata=metadata or {},
        )

    async def embed_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        audio: Optional[bytes] = None,
    ) -> NovaEmbeddingResult:
        """
        Generate embedding for multimodal input.

        Combines text, image, and/or audio into a single embedding.

        Args:
            text: Optional text component
            image: Optional image bytes
            audio: Optional audio bytes

        Returns:
            NovaEmbeddingResult with combined embedding
        """
        if not self._initialized:
            await self.initialize()

        if not any([text, image, audio]):
            raise ValueError("At least one input modality required")

        # Build multimodal input
        inputs = {}
        modalities = []

        if text:
            inputs["text"] = text
            modalities.append("text")

        if image:
            inputs["image"] = base64.b64encode(image).decode()
            modalities.append("image")

        if audio:
            inputs["audio"] = base64.b64encode(audio).decode()
            modalities.append("audio")

        input_hash = hashlib.sha256(
            str(inputs).encode()
        ).hexdigest()[:16]

        async with self._semaphore:
            embedding = await self._invoke_model(
                input_type="multimodal",
                input_data=inputs,
            )

        return NovaEmbeddingResult(
            embedding=embedding,
            modality=NovaModalityType.TEXT,  # Primary modality
            input_hash=input_hash,
            metadata={"modalities": modalities},
        )

    async def embed_batch(
        self,
        texts: List[str],
    ) -> List[NovaEmbeddingResult]:
        """
        Batch embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of NovaEmbeddingResult
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), self.config.max_batch_size):
            batch = texts[i:i + self.config.max_batch_size]

            # Embed batch in parallel
            batch_results = await asyncio.gather(
                *[self.embed_text(text) for text in batch],
                return_exceptions=True,
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Batch embed failed: {result}")
                    results.append(None)
                else:
                    results.append(result)

        return results

    async def cross_modal_search(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[bytes] = None,
        candidates: List[Union[str, bytes]] = None,
        candidate_type: NovaModalityType = NovaModalityType.IMAGE,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Cross-modal search: find candidates most similar to query.

        Examples:
        - Text query → Image candidates (image search)
        - Image query → Text candidates (image captioning)

        Args:
            query_text: Text query (mutually exclusive with query_image)
            query_image: Image query (mutually exclusive with query_text)
            candidates: List of candidate items to search
            candidate_type: Type of candidates (text or image)
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by score
        """
        if not candidates:
            return []

        # Get query embedding
        if query_text:
            query_result = await self.embed_text(query_text)
        elif query_image:
            query_result = await self.embed_image(query_image)
        else:
            raise ValueError("Either query_text or query_image required")

        query_embedding = query_result.embedding

        # Get candidate embeddings
        candidate_embeddings = []
        for candidate in candidates:
            if candidate_type == NovaModalityType.TEXT:
                result = await self.embed_text(candidate)
            else:
                result = await self.embed_image(candidate)
            candidate_embeddings.append(result.embedding)

        # Compute similarities
        similarities = []
        for i, cand_emb in enumerate(candidate_embeddings):
            sim = self._cosine_similarity(query_embedding, cand_emb)
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    async def _invoke_model(
        self,
        input_type: str,
        input_data: Union[str, Dict[str, Any]],
    ) -> List[float]:
        """Invoke the Bedrock model."""
        import json

        loop = asyncio.get_running_loop()

        # Build request body
        if input_type == "text":
            body = {
                "inputText": input_data,
            }
        elif input_type == "image":
            body = {
                "inputImage": input_data,
            }
        elif input_type == "video":
            body = {
                "inputVideo": input_data,
            }
        elif input_type == "audio":
            body = {
                "inputAudio": input_data,
            }
        elif input_type == "multimodal":
            body = input_data
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        # Add model config
        body["embeddingConfig"] = {
            "outputEmbeddingLength": self.config.embedding_dimension,
        }

        # Invoke model
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            ),
        )

        # Parse response
        response_body = json.loads(response["body"].read())
        embedding = response_body.get("embedding", [])

        # Normalize if configured
        if self.config.normalize_embeddings:
            embedding = self._normalize(embedding)

        return embedding

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Compute cosine similarity."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _normalize(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            if not self._initialized:
                await self.initialize()

            # Test with simple text
            result = await self.embed_text("health check")

            return {
                "status": "healthy",
                "model": self.config.model_id,
                "region": self.config.aws_region,
                "embedding_dimension": len(result.embedding),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# =============================================================================
# Singleton Instance
# =============================================================================

_nova_service: Optional[NovaEmbeddingService] = None


async def get_nova_embeddings(
    config: Optional[NovaEmbeddingConfig] = None,
) -> NovaEmbeddingService:
    """Get or create Nova embeddings singleton."""
    global _nova_service
    if _nova_service is None:
        _nova_service = NovaEmbeddingService(config)
        await _nova_service.initialize()
    return _nova_service


def reset_nova_embeddings() -> None:
    """Reset the Nova embeddings singleton."""
    global _nova_service
    _nova_service = None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "NovaModalityType",
    "NovaModelVariant",
    "NovaEmbeddingConfig",
    "NovaEmbeddingResult",
    "NovaEmbeddingService",
    "get_nova_embeddings",
    "reset_nova_embeddings",
]
