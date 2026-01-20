"""
AIDocumentIndexer - Embedding Service
=====================================

Embedding generation with Ray-parallel processing.
Supports multiple embedding providers via LangChain.

Performance optimizations:
- Adaptive batching based on provider rate limits
- Embedding cache to avoid re-embedding identical content
- Concurrent processing with ThreadPoolExecutor fallback
- Ray-parallel processing for large batches
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
import hashlib
import os
import structlog
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# LangChain embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

# Ray for parallel processing
import ray

from backend.processors.chunker import Chunk
from backend.services.llm import LLMConfig

logger = structlog.get_logger(__name__)

# Embedding cache for deduplication (content hash -> embedding)
_embedding_cache: Dict[str, List[float]] = {}
# Increased default cache size from 10k to 100k for better hit rate at scale
# At 1536 dimensions * 4 bytes * 100k = ~600MB memory usage
# For production with millions of docs, use Redis-backed cache instead
_CACHE_MAX_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "100000"))


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
            # Check for explicit dimension override (OpenAI v3 models support flexible dimensions)
            import os
            explicit_dim = os.getenv("EMBEDDING_DIMENSION")

            # OpenAI text-embedding-3-* models support dimension parameter
            if explicit_dim and ("text-embedding-3" in self.model.lower()):
                try:
                    dim = int(explicit_dim)
                    logger.info(f"Using OpenAI with reduced dimension: {dim}D")
                    return OpenAIEmbeddings(
                        model=self.model,
                        openai_api_key=self.config.openai_api_key,
                        dimensions=dim,  # OpenAI v3 supports dimension reduction
                    )
                except ValueError:
                    pass

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

    def _get_content_hash(self, text: str) -> str:
        """Generate a hash for text content for caching."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()

    def embed_texts(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use embedding cache (default True)

        Returns:
            List of embedding vectors
        """
        global _embedding_cache

        if not texts:
            return []

        # Track which texts need embedding vs are cached
        result = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                result[i] = [0.0] * self.dimensions
                continue

            if use_cache:
                content_hash = self._get_content_hash(text)
                if content_hash in _embedding_cache:
                    result[i] = _embedding_cache[content_hash]
                    continue

            texts_to_embed.append(text)
            indices_to_embed.append(i)

        # Log cache statistics
        cache_hits = len(texts) - len(texts_to_embed) - sum(1 for r in result if r is not None and r == [0.0] * self.dimensions)
        if cache_hits > 0:
            logger.debug(
                "Embedding cache hits",
                cache_hits=cache_hits,
                cache_misses=len(texts_to_embed),
            )

        # Embed texts that weren't cached
        if texts_to_embed:
            try:
                embeddings = self.embeddings.embed_documents(texts_to_embed)

                # Store in results and cache
                for idx, text, embedding in zip(indices_to_embed, texts_to_embed, embeddings):
                    result[idx] = embedding

                    if use_cache and len(_embedding_cache) < _CACHE_MAX_SIZE:
                        content_hash = self._get_content_hash(text)
                        _embedding_cache[content_hash] = embedding

            except Exception as e:
                logger.error(
                    "Batch embedding failed",
                    error=str(e),
                    num_texts=len(texts_to_embed),
                )
                raise

        # Fill any remaining None values with zero vectors
        for i in range(len(result)):
            if result[i] is None:
                result[i] = [0.0] * self.dimensions

        return result

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
        batch_size: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for document chunks with caching and optimal batching.

        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to process at once (auto-detected if None)
            use_cache: Whether to use the embedding cache

        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []

        # Use optimal batch size for provider if not specified
        if batch_size is None:
            batch_size = get_optimal_batch_size(self.provider, len(chunks))

        start_time = time.time()
        logger.info(
            "Embedding chunks",
            num_chunks=len(chunks),
            batch_size=batch_size,
            model=self.model,
            provider=self.provider,
            cache_enabled=use_cache,
        )

        results = []
        texts = [chunk.content for chunk in chunks]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            try:
                embeddings = self.embed_texts(batch_texts, use_cache=use_cache)

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

        elapsed = time.time() - start_time
        logger.info(
            "Embedding complete",
            num_chunks=len(chunks),
            num_results=len(results),
            elapsed_seconds=round(elapsed, 2),
            chunks_per_second=round(len(chunks) / elapsed, 1) if elapsed > 0 else 0,
        )

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
    # LLMConfig reads from env vars, so we create instance and override from dict if provided
    config = LLMConfig.from_env()
    if config_dict:
        if config_dict.get("openai_api_key"):
            config.openai_api_key = config_dict["openai_api_key"]
        if config_dict.get("ollama_base_url"):
            config.ollama_base_url = config_dict["ollama_base_url"]
        if config_dict.get("anthropic_api_key"):
            config.anthropic_api_key = config_dict["anthropic_api_key"]

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


def clear_embedding_cache():
    """Clear the global embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the embedding cache."""
    return {
        "size": len(_embedding_cache),
        "max_size": _CACHE_MAX_SIZE,
        "utilization_percent": (len(_embedding_cache) / _CACHE_MAX_SIZE) * 100 if _CACHE_MAX_SIZE > 0 else 0,
    }


# =============================================================================
# Persistent Embedding Cache (Redis-backed)
# =============================================================================


class EmbeddingCachePersistence:
    """
    Persistent embedding cache backed by Redis.

    Features:
    - Persists embeddings to Redis for cross-session reuse
    - Reduces API calls by caching expensive embedding operations
    - Supports bulk save/load for efficient startup preloading
    - Graceful fallback to in-memory when Redis unavailable

    Usage:
        cache = EmbeddingCachePersistence()

        # Save embedding
        await cache.save("hash123", [0.1, 0.2, ...])

        # Load embedding
        embedding = await cache.load("hash123")

        # Bulk preload on startup
        await cache.preload_to_memory()
    """

    def __init__(
        self,
        prefix: str = "emb_cache",
        ttl_days: int = 30,
        max_preload_items: int = 50000,
    ):
        """
        Initialize the persistent embedding cache.

        Args:
            prefix: Redis key prefix for embeddings
            ttl_days: Time-to-live for cached embeddings in days
            max_preload_items: Maximum items to preload from Redis on startup
        """
        self.prefix = prefix
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self.max_preload_items = max_preload_items
        self._redis_available = None  # Lazy check

        logger.info(
            "EmbeddingCachePersistence initialized",
            prefix=prefix,
            ttl_days=ttl_days,
            max_preload_items=max_preload_items,
        )

    def _make_key(self, content_hash: str) -> str:
        """Create a Redis key for an embedding hash."""
        return f"{self.prefix}:{content_hash}"

    async def _get_redis(self):
        """Get Redis client (lazy initialization)."""
        try:
            from backend.services.redis_client import get_redis_client
            return await get_redis_client()
        except Exception as e:
            logger.debug(f"Redis not available for embedding cache: {e}")
            return None

    async def is_redis_available(self) -> bool:
        """Check if Redis is available for persistent caching."""
        if self._redis_available is not None:
            return self._redis_available

        client = await self._get_redis()
        self._redis_available = client is not None
        return self._redis_available

    async def save(self, content_hash: str, embedding: List[float]) -> bool:
        """
        Save an embedding to Redis.

        Args:
            content_hash: Hash of the content that was embedded
            embedding: The embedding vector

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            client = await self._get_redis()
            if client is None:
                return False

            import json
            key = self._make_key(content_hash)
            await client.setex(key, self.ttl_seconds, json.dumps(embedding))
            return True

        except Exception as e:
            logger.debug(f"Failed to save embedding to Redis: {e}")
            return False

    async def load(self, content_hash: str) -> Optional[List[float]]:
        """
        Load an embedding from Redis.

        Args:
            content_hash: Hash of the content

        Returns:
            The embedding vector if found, None otherwise
        """
        try:
            client = await self._get_redis()
            if client is None:
                return None

            import json
            key = self._make_key(content_hash)
            data = await client.get(key)

            if data:
                return json.loads(data)
            return None

        except Exception as e:
            logger.debug(f"Failed to load embedding from Redis: {e}")
            return None

    async def load_batch(self, content_hashes: List[str]) -> Dict[str, List[float]]:
        """
        Load multiple embeddings from Redis in a single operation.

        Args:
            content_hashes: List of content hashes to load

        Returns:
            Dictionary mapping hashes to embeddings (only found entries)
        """
        if not content_hashes:
            return {}

        try:
            client = await self._get_redis()
            if client is None:
                return {}

            import json
            keys = [self._make_key(h) for h in content_hashes]
            results = {}

            # Use pipeline for efficient batch loading
            pipe = client.pipeline()
            for key in keys:
                pipe.get(key)

            values = await pipe.execute()

            for hash_val, value in zip(content_hashes, values):
                if value:
                    try:
                        results[hash_val] = json.loads(value)
                    except json.JSONDecodeError:
                        pass

            logger.debug(
                "Batch loaded embeddings from Redis",
                requested=len(content_hashes),
                found=len(results),
            )
            return results

        except Exception as e:
            logger.debug(f"Failed to batch load embeddings from Redis: {e}")
            return {}

    async def save_batch(self, embeddings: Dict[str, List[float]]) -> int:
        """
        Save multiple embeddings to Redis in a single operation.

        Args:
            embeddings: Dictionary mapping content hashes to embeddings

        Returns:
            Number of embeddings successfully saved
        """
        if not embeddings:
            return 0

        try:
            client = await self._get_redis()
            if client is None:
                return 0

            import json
            pipe = client.pipeline()

            for content_hash, embedding in embeddings.items():
                key = self._make_key(content_hash)
                pipe.setex(key, self.ttl_seconds, json.dumps(embedding))

            await pipe.execute()

            logger.debug(f"Batch saved {len(embeddings)} embeddings to Redis")
            return len(embeddings)

        except Exception as e:
            logger.debug(f"Failed to batch save embeddings to Redis: {e}")
            return 0

    async def preload_to_memory(self) -> int:
        """
        Preload embeddings from Redis into the in-memory cache.

        Call this during application startup to warm the cache.

        Returns:
            Number of embeddings loaded into memory
        """
        global _embedding_cache

        try:
            client = await self._get_redis()
            if client is None:
                logger.info("Redis not available, skipping embedding cache preload")
                return 0

            import json
            # Scan for embedding keys
            pattern = f"{self.prefix}:*"
            loaded = 0

            async for key in client.scan_iter(match=pattern, count=1000):
                if loaded >= self.max_preload_items:
                    break

                if len(_embedding_cache) >= _CACHE_MAX_SIZE:
                    break

                try:
                    value = await client.get(key)
                    if value:
                        # Extract hash from key
                        content_hash = key.replace(f"{self.prefix}:", "")
                        embedding = json.loads(value)
                        _embedding_cache[content_hash] = embedding
                        loaded += 1
                except Exception:
                    continue

            logger.info(
                "Preloaded embeddings from Redis to memory",
                loaded=loaded,
                memory_cache_size=len(_embedding_cache),
            )
            return loaded

        except Exception as e:
            logger.warning(f"Failed to preload embeddings from Redis: {e}")
            return 0

    async def persist_memory_cache(self) -> int:
        """
        Persist the current in-memory cache to Redis.

        Call this during application shutdown or periodically.

        Returns:
            Number of embeddings persisted
        """
        global _embedding_cache

        if not _embedding_cache:
            return 0

        return await self.save_batch(_embedding_cache)

    async def delete(self, content_hash: str) -> bool:
        """
        Delete an embedding from Redis.

        Args:
            content_hash: Hash of the content

        Returns:
            True if deleted, False otherwise
        """
        try:
            client = await self._get_redis()
            if client is None:
                return False

            key = self._make_key(content_hash)
            await client.delete(key)
            return True

        except Exception as e:
            logger.debug(f"Failed to delete embedding from Redis: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the persistent cache."""
        try:
            client = await self._get_redis()
            if client is None:
                return {
                    "redis_available": False,
                    "persistent_count": 0,
                    "memory_count": len(_embedding_cache),
                }

            # Count keys with our prefix
            pattern = f"{self.prefix}:*"
            count = 0
            async for _ in client.scan_iter(match=pattern, count=1000):
                count += 1
                if count > 100000:  # Cap counting at 100k to avoid long scans
                    break

            return {
                "redis_available": True,
                "persistent_count": count,
                "memory_count": len(_embedding_cache),
                "prefix": self.prefix,
                "ttl_days": self.ttl_seconds // (24 * 60 * 60),
            }

        except Exception as e:
            return {
                "redis_available": False,
                "error": str(e),
                "memory_count": len(_embedding_cache),
            }


# Singleton instance for persistent cache
_embedding_cache_persistence: Optional[EmbeddingCachePersistence] = None


def get_embedding_cache_persistence() -> EmbeddingCachePersistence:
    """Get or create the persistent embedding cache singleton."""
    global _embedding_cache_persistence
    if _embedding_cache_persistence is None:
        _embedding_cache_persistence = EmbeddingCachePersistence()
    return _embedding_cache_persistence


async def preload_embedding_cache():
    """
    Preload embeddings from Redis into memory cache on startup.

    Call this from application lifespan to warm the cache.
    """
    if os.getenv("EMBEDDING_CACHE_PRELOAD", "false").lower() == "true":
        cache = get_embedding_cache_persistence()
        await cache.preload_to_memory()


async def persist_embedding_cache():
    """
    Persist in-memory embedding cache to Redis on shutdown.

    Call this from application lifespan to save the cache.
    """
    if os.getenv("EMBEDDING_CACHE_PERSIST", "true").lower() == "true":
        cache = get_embedding_cache_persistence()
        await cache.persist_memory_cache()


# Provider-specific rate limits and optimal batch sizes
PROVIDER_BATCH_CONFIG = {
    "openai": {
        "max_batch_size": 2048,  # OpenAI supports up to 2048 texts per batch
        "requests_per_minute": 3000,  # Rate limit
        "tokens_per_minute": 1000000,  # TPM limit for embeddings
        "optimal_batch_size": 500,  # Good balance of speed and reliability
    },
    "ollama": {
        "max_batch_size": 100,  # Ollama is local, smaller batches
        "requests_per_minute": None,  # No rate limit
        "optimal_batch_size": 50,  # Keep batches small for local models
    },
    "huggingface": {
        "max_batch_size": 256,
        "requests_per_minute": 300,
        "optimal_batch_size": 100,
    },
}


def get_optimal_batch_size(provider: str, num_texts: int) -> int:
    """
    Get the optimal batch size for a provider.

    Args:
        provider: The embedding provider
        num_texts: Total number of texts to embed

    Returns:
        Optimal batch size for the provider
    """
    config = PROVIDER_BATCH_CONFIG.get(provider, PROVIDER_BATCH_CONFIG["openai"])
    optimal = config["optimal_batch_size"]

    # For small batches, use the smaller of optimal or num_texts
    if num_texts <= optimal:
        return num_texts

    return optimal
