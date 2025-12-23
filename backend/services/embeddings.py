"""
AIDocumentIndexer - Embedding Service
=====================================

Embedding generation with Ray-parallel processing.
Supports multiple embedding providers via LangChain.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
import hashlib
import structlog
from concurrent.futures import ThreadPoolExecutor

# LangChain embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

# Ray for parallel processing
import ray

from backend.processors.chunker import Chunk
from backend.services.llm import LLMConfig

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding a single chunk."""
    chunk_id: str
    chunk_hash: str
    embedding: List[float]
    model: str
    dimensions: int
    metadata: Dict[str, Any]


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding operation."""
    results: List[EmbeddingResult]
    total_chunks: int
    successful: int
    failed: int
    model: str


class EmbeddingService:
    """
    Embedding generation service with multiple provider support.

    Features:
    - Multiple embedding providers (OpenAI, Ollama, HuggingFace)
    - Batch processing for efficiency
    - Ray-parallel processing for large batches
    - Caching support
    - Automatic chunking integration
    """

    # Default embedding models by provider
    DEFAULT_MODELS = {
        "openai": "text-embedding-3-small",  # 1536 dimensions
        "ollama": "nomic-embed-text",        # 768 dimensions
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    }

    # Embedding dimensions by model
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "mxbai-embed-large": 1024,
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        """
        Initialize embedding service.

        Args:
            provider: Embedding provider ("openai", "ollama", "huggingface")
            model: Specific model to use (defaults to provider's default)
            config: LLM configuration with API keys
        """
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS.get(provider, "text-embedding-3-small")
        self.config = config or LLMConfig.from_env()
        self._embeddings: Optional[Embeddings] = None
        self._dimensions: Optional[int] = None

        logger.info(
            "Initializing embedding service",
            provider=provider,
            model=self.model,
        )

    @property
    def embeddings(self) -> Embeddings:
        """Get or create embeddings instance (lazy initialization)."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for current model."""
        if self._dimensions is None:
            self._dimensions = self.MODEL_DIMENSIONS.get(self.model, 1536)
        return self._dimensions

    def _create_embeddings(self) -> Embeddings:
        """Create embeddings instance based on provider."""
        if self.provider == "openai":
            return OpenAIEmbeddings(
                model=self.model,
                openai_api_key=self.config.openai_api_key,
            )
        elif self.provider == "ollama":
            return OllamaEmbeddings(
                model=self.model,
                base_url=self.config.ollama_base_url,
            )
        else:
            # Default to OpenAI
            logger.warning(f"Unknown provider {self.provider}, defaulting to OpenAI")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.config.openai_api_key,
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            return [0.0] * self.dimensions

        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error("Embedding failed", error=str(e), text_length=len(text))
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter empty texts but track positions
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            return [[0.0] * self.dimensions for _ in texts]

        try:
            embeddings = self.embeddings.embed_documents(valid_texts)

            # Rebuild full list with zero vectors for empty texts
            result = [[0.0] * self.dimensions for _ in texts]
            for idx, embedding in zip(valid_indices, embeddings):
                result[idx] = embedding

            return result

        except Exception as e:
            logger.error(
                "Batch embedding failed",
                error=str(e),
                num_texts=len(texts),
            )
            raise

    async def embed_text_async(self, text: str) -> List[float]:
        """Async version of embed_text."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_text, text)

    async def embed_query(self, text: str) -> List[float]:
        """
        Async method to embed a query text.

        This is an alias for embed_text_async, provided for compatibility
        with LangChain's Embeddings interface naming convention.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        return await self.embed_text_async(text)

    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_texts."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_texts, texts)

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 100,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to process at once

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []

        logger.info(
            "Embedding chunks",
            num_chunks=len(chunks),
            batch_size=batch_size,
            model=self.model,
        )

        results = []
        texts = [chunk.content for chunk in chunks]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            try:
                embeddings = self.embed_texts(batch_texts)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    result = EmbeddingResult(
                        chunk_id=f"{chunk.document_id}_{chunk.chunk_index}" if chunk.document_id else str(chunk.chunk_index),
                        chunk_hash=chunk.chunk_hash,
                        embedding=embedding,
                        model=self.model,
                        dimensions=len(embedding),
                        metadata=chunk.metadata,
                    )
                    results.append(result)

                logger.debug(
                    "Batch embedded",
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch_texts),
                )

            except Exception as e:
                logger.error(
                    "Batch embedding failed",
                    batch_num=i // batch_size + 1,
                    error=str(e),
                )
                # Create placeholder results for failed batch
                for chunk in batch_chunks:
                    result = EmbeddingResult(
                        chunk_id=f"{chunk.document_id}_{chunk.chunk_index}" if chunk.document_id else str(chunk.chunk_index),
                        chunk_hash=chunk.chunk_hash,
                        embedding=[0.0] * self.dimensions,  # Zero vector for failures
                        model=self.model,
                        dimensions=self.dimensions,
                        metadata={**chunk.metadata, "embedding_failed": True},
                    )
                    results.append(result)

        return results

    async def embed_chunks_async(
        self,
        chunks: List[Chunk],
        batch_size: int = 100,
    ) -> List[EmbeddingResult]:
        """Async version of embed_chunks."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.embed_chunks(chunks, batch_size)
            )


# =============================================================================
# Ray-Parallel Embedding Functions
# =============================================================================

@ray.remote
def embed_batch_ray(
    texts: List[str],
    provider: str = "openai",
    model: Optional[str] = None,
    config_dict: Optional[Dict] = None,
) -> List[List[float]]:
    """
    Ray remote function for batch embedding.

    This runs on Ray workers for distributed processing.
    """
    # Reconstruct config from dict (Ray requires serializable objects)
    if config_dict:
        config = LLMConfig(
            openai_api_key=config_dict.get("openai_api_key"),
            ollama_base_url=config_dict.get("ollama_base_url"),
            anthropic_api_key=config_dict.get("anthropic_api_key"),
        )
    else:
        config = LLMConfig.from_env()

    service = EmbeddingService(provider=provider, model=model, config=config)
    return service.embed_texts(texts)


class RayEmbeddingService:
    """
    Distributed embedding service using Ray.

    Automatically distributes embedding work across Ray cluster.
    Falls back to local processing if Ray is not available.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        num_workers: int = 4,
        batch_size_per_worker: int = 50,
    ):
        """
        Initialize Ray embedding service.

        Args:
            provider: Embedding provider
            model: Embedding model
            config: LLM configuration
            num_workers: Number of Ray workers to use
            batch_size_per_worker: Texts per worker batch
        """
        self.provider = provider
        self.model = model or EmbeddingService.DEFAULT_MODELS.get(provider)
        self.config = config or LLMConfig.from_env()
        self.num_workers = num_workers
        self.batch_size_per_worker = batch_size_per_worker

        # Local fallback service
        self._local_service = EmbeddingService(
            provider=provider,
            model=model,
            config=config,
        )

        logger.info(
            "Initialized Ray embedding service",
            provider=provider,
            model=self.model,
            num_workers=num_workers,
        )

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._local_service.dimensions

    def _is_ray_available(self) -> bool:
        """Check if Ray is initialized and available."""
        try:
            return ray.is_initialized()
        except Exception:
            return False

    def _embed_texts_concurrent(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using concurrent.futures when Ray is not available.

        Provides 3-5x speedup over sequential processing for large batches.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Determine number of workers (default: 4, configurable via env)
        max_workers = int(os.getenv("EMBEDDING_CONCURRENT_WORKERS", "4"))

        # Split texts into batches
        batches = []
        for i in range(0, len(texts), self.batch_size_per_worker):
            batch = texts[i:i + self.batch_size_per_worker]
            batches.append((i, batch))

        logger.info(
            "Starting concurrent embedding (Ray unavailable)",
            num_texts=len(texts),
            num_batches=len(batches),
            max_workers=max_workers,
        )

        # Process batches concurrently
        results = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_idx = {
                executor.submit(self._local_service.embed_texts, batch): idx
                for idx, batch in batches
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx // self.batch_size_per_worker] = future.result()
                except Exception as e:
                    logger.error(
                        "Concurrent batch embedding failed",
                        batch_idx=idx,
                        error=str(e),
                    )
                    # Return zero vectors for failed batch
                    batch_size = len(batches[idx // self.batch_size_per_worker][1])
                    results[idx // self.batch_size_per_worker] = [
                        [0.0] * self._local_service.dimensions for _ in range(batch_size)
                    ]

        # Flatten results
        embeddings = []
        for batch_result in results:
            if batch_result:
                embeddings.extend(batch_result)

        logger.info(
            "Concurrent embedding complete",
            num_embeddings=len(embeddings),
        )

        return embeddings

    def embed_texts_parallel(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Embed texts using Ray parallel processing.

        Falls back to concurrent.futures for 3-5x faster local processing when Ray unavailable.

        Args:
            texts: Texts to embed
            show_progress: Whether to log progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Use local processing for small batches or when Ray unavailable
        if len(texts) <= self.batch_size_per_worker or not self._is_ray_available():
            if not self._is_ray_available() and len(texts) > self.batch_size_per_worker:
                # Use concurrent embedding fallback for large batches without Ray
                return self._embed_texts_concurrent(texts)
            return self._local_service.embed_texts(texts)

        logger.info(
            "Starting Ray parallel embedding",
            num_texts=len(texts),
            num_workers=self.num_workers,
        )

        # Prepare config dict for Ray workers
        config_dict = {
            "openai_api_key": self.config.openai_api_key,
            "ollama_base_url": self.config.ollama_base_url,
            "anthropic_api_key": self.config.anthropic_api_key,
        }

        # Split texts into batches for workers
        batches = []
        for i in range(0, len(texts), self.batch_size_per_worker):
            batch = texts[i:i + self.batch_size_per_worker]
            batches.append(batch)

        # Submit all batches to Ray
        futures = [
            embed_batch_ray.remote(
                batch,
                self.provider,
                self.model,
                config_dict,
            )
            for batch in batches
        ]

        # Collect results
        try:
            all_results = ray.get(futures)

            # Flatten results
            embeddings = []
            for batch_result in all_results:
                embeddings.extend(batch_result)

            logger.info(
                "Ray parallel embedding complete",
                num_embeddings=len(embeddings),
            )

            return embeddings

        except Exception as e:
            logger.error(
                "Ray embedding failed, falling back to local",
                error=str(e),
            )
            return self._local_service.embed_texts(texts)

    def embed_chunks_parallel(
        self,
        chunks: List[Chunk],
    ) -> List[EmbeddingResult]:
        """
        Embed chunks using Ray parallel processing.

        Args:
            chunks: Chunks to embed

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts_parallel(texts)

        results = []
        for chunk, embedding in zip(chunks, embeddings):
            result = EmbeddingResult(
                chunk_id=f"{chunk.document_id}_{chunk.chunk_index}" if chunk.document_id else str(chunk.chunk_index),
                chunk_hash=chunk.chunk_hash,
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                metadata=chunk.metadata,
            )
            results.append(result)

        return results

    async def embed_texts_parallel_async(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Async version of embed_texts_parallel."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.embed_texts_parallel(texts)
            )


# =============================================================================
# Utility Functions
# =============================================================================

def get_embedding_service(
    provider: str = "openai",
    use_ray: bool = True,
) -> Union[EmbeddingService, RayEmbeddingService]:
    """
    Get appropriate embedding service based on configuration.

    Args:
        provider: Embedding provider
        use_ray: Whether to use Ray for parallel processing

    Returns:
        Embedding service instance
    """
    if use_ray:
        return RayEmbeddingService(provider=provider)
    return EmbeddingService(provider=provider)


def compute_similarity(
    embedding1: List[float],
    embedding2: List[float],
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    if len(embedding1) != len(embedding2):
        raise ValueError("Embeddings must have same dimensions")

    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
