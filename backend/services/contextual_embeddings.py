"""
AIDocumentIndexer - Contextual Embeddings Service
==================================================

Implements Anthropic's Contextual Retrieval technique for 67% error reduction.

The key insight: Chunks lack context. A chunk like "The company reported $4.2B revenue"
is useless without knowing which company and what timeframe. Contextual Retrieval
prepends a brief context to each chunk before embedding.

Research: Anthropic (2024) - "Introducing Contextual Retrieval"
Results:
- Contextual embeddings alone: 49% error reduction
- Contextual embeddings + BM25: 67% error reduction

Architecture:
1. For each chunk, generate a brief context using a fast LLM (Claude Haiku/GPT-4o-mini)
2. Prepend context to chunk before embedding
3. Store both original and contextualized versions
4. Optional: Generate contextual BM25 keywords

Cost: ~$0.0001-0.0003 per chunk with fast models
"""

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

from backend.core.config import settings
from backend.core.performance import LRUCache, gather_with_concurrency

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class ContextualConfig:
    """Configuration for contextual embeddings."""
    # Context generation model (fast, cheap)
    context_model: str = "claude-3-5-haiku-latest"
    context_provider: str = "anthropic"

    # Fallback to OpenAI if Anthropic not available
    fallback_model: str = "gpt-4o-mini"
    fallback_provider: str = "openai"

    # Context generation settings
    max_document_preview: int = 2000  # Chars of document to include
    max_context_length: int = 150    # Max generated context length
    temperature: float = 0.0         # Deterministic for caching

    # BM25 settings
    generate_bm25_keywords: bool = True
    max_bm25_keywords: int = 10

    # Caching
    cache_contexts: bool = True
    cache_ttl_days: int = 30

    # Concurrency
    max_concurrent_generations: int = 10


# Context generation prompt (from Anthropic's research)
CONTEXT_GENERATION_PROMPT = """<document>
{document_preview}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

# BM25 keyword extraction prompt
BM25_KEYWORDS_PROMPT = """Extract {max_keywords} important search keywords from this text that would help someone find this content. Return only the keywords separated by commas, nothing else.

Text:
{text}

Keywords:"""


@dataclass(slots=True)
class ContextualChunk:
    """A chunk with its generated context."""
    chunk_id: str
    original_text: str
    context: str
    contextualized_text: str
    bm25_keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# Context Cache
# =============================================================================

class ContextCache:
    """
    Cache for generated contexts.

    Uses content hash as key to avoid regenerating contexts for identical chunks.
    Supports both in-memory LRU and Redis-backed storage.
    """

    def __init__(self, use_redis: bool = True, ttl_days: int = 30):
        self.use_redis = use_redis
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self._memory_cache = LRUCache[str](capacity=10000)
        self._redis_client = None

    def _hash_key(self, document_preview: str, chunk_text: str) -> str:
        """Generate cache key from document preview and chunk text."""
        content = f"{document_preview[:500]}|{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get(self, document_preview: str, chunk_text: str) -> Optional[str]:
        """Get cached context if exists."""
        key = self._hash_key(document_preview, chunk_text)

        # Check memory cache first
        cached = await self._memory_cache.get(key)
        if cached:
            return cached

        # Check Redis if available
        if self.use_redis:
            try:
                from backend.services.redis_client import get_redis_client
                redis = await get_redis_client()
                if redis:
                    cached = await redis.get(f"ctx:{key}")
                    if cached:
                        # Populate memory cache
                        await self._memory_cache.set(key, cached)
                        return cached
            except Exception as e:
                logger.debug("Redis cache miss", error=str(e))

        return None

    async def set(self, document_preview: str, chunk_text: str, context: str) -> None:
        """Cache generated context."""
        key = self._hash_key(document_preview, chunk_text)

        # Always set in memory cache
        await self._memory_cache.set(key, context)

        # Set in Redis if available
        if self.use_redis:
            try:
                from backend.services.redis_client import get_redis_client
                redis = await get_redis_client()
                if redis:
                    await redis.setex(f"ctx:{key}", self.ttl_seconds, context)
            except Exception as e:
                logger.debug("Redis cache set failed", error=str(e))


# =============================================================================
# Contextual Embedding Service
# =============================================================================

class ContextualEmbeddingService:
    """
    Service for generating contextual embeddings.

    Implements Anthropic's Contextual Retrieval technique:
    1. Generate brief context for each chunk using fast LLM
    2. Prepend context to chunk before embedding
    3. Optionally extract BM25 keywords

    Usage:
        service = ContextualEmbeddingService()

        # Single chunk
        contextual = await service.contextualize_chunk(
            chunk_text="Revenue was $4.2B...",
            document_text="Apple Inc Annual Report 2024...",
            chunk_id="chunk_123"
        )

        # Batch processing
        results = await service.contextualize_chunks(chunks, document_text)
    """

    def __init__(self, config: Optional[ContextualConfig] = None):
        self.config = config or ContextualConfig()
        self._llm = None
        self._cache = ContextCache(
            use_redis=self.config.cache_contexts,
            ttl_days=self.config.cache_ttl_days
        )
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the LLM for context generation."""
        if self._initialized:
            return True

        try:
            from backend.services.llm import LLMFactory

            # Try primary provider first
            try:
                self._llm = LLMFactory.get_chat_model(
                    provider=self.config.context_provider,
                    model=self.config.context_model,
                    temperature=self.config.temperature,
                    max_tokens=200,
                )
                logger.info(
                    "Contextual embedding service initialized",
                    provider=self.config.context_provider,
                    model=self.config.context_model,
                )
            except Exception as e:
                logger.warning(
                    "Primary context model failed, using fallback",
                    primary=self.config.context_model,
                    fallback=self.config.fallback_model,
                    error=str(e),
                )
                self._llm = LLMFactory.get_chat_model(
                    provider=self.config.fallback_provider,
                    model=self.config.fallback_model,
                    temperature=self.config.temperature,
                    max_tokens=200,
                )

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize contextual embedding service", error=str(e))
            return False

    async def generate_context(
        self,
        chunk_text: str,
        document_text: str,
    ) -> str:
        """
        Generate context for a single chunk.

        Args:
            chunk_text: The chunk to contextualize
            document_text: Full document text (will be truncated)

        Returns:
            Generated context string
        """
        if not await self.initialize():
            logger.warning("Contextual service not initialized, returning empty context")
            return ""

        # Truncate document preview
        document_preview = document_text[:self.config.max_document_preview]
        if len(document_text) > self.config.max_document_preview:
            document_preview += "..."

        # Check cache
        cached = await self._cache.get(document_preview, chunk_text)
        if cached:
            return cached

        # Generate context
        try:
            from langchain_core.messages import HumanMessage

            prompt = CONTEXT_GENERATION_PROMPT.format(
                document_preview=document_preview,
                chunk_text=chunk_text,
            )

            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            context = response.content.strip()

            # Truncate if too long
            if len(context) > self.config.max_context_length:
                context = context[:self.config.max_context_length].rsplit(' ', 1)[0] + "..."

            # Cache the result
            await self._cache.set(document_preview, chunk_text, context)

            return context

        except Exception as e:
            logger.error("Context generation failed", error=str(e))
            return ""

    async def extract_bm25_keywords(self, text: str) -> List[str]:
        """
        Extract BM25-friendly keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        if not self.config.generate_bm25_keywords:
            return []

        if not await self.initialize():
            return []

        try:
            from langchain_core.messages import HumanMessage

            prompt = BM25_KEYWORDS_PROMPT.format(
                max_keywords=self.config.max_bm25_keywords,
                text=text[:1000],
            )

            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            keywords_str = response.content.strip()

            # Parse comma-separated keywords
            keywords = [k.strip().lower() for k in keywords_str.split(',')]
            return keywords[:self.config.max_bm25_keywords]

        except Exception as e:
            logger.debug("BM25 keyword extraction failed", error=str(e))
            return []

    async def contextualize_chunk(
        self,
        chunk_text: str,
        document_text: str,
        chunk_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextualChunk:
        """
        Create a contextualized chunk.

        Args:
            chunk_text: The chunk text
            document_text: Full document text
            chunk_id: Unique chunk identifier
            metadata: Optional metadata to include

        Returns:
            ContextualChunk with context and optionally BM25 keywords
        """
        # Generate context
        context = await self.generate_context(chunk_text, document_text)

        # Create contextualized text (context prepended)
        if context:
            contextualized_text = f"{context}\n\n{chunk_text}"
        else:
            contextualized_text = chunk_text

        # Extract BM25 keywords (from contextualized text for better coverage)
        bm25_keywords = await self.extract_bm25_keywords(contextualized_text)

        return ContextualChunk(
            chunk_id=chunk_id,
            original_text=chunk_text,
            context=context,
            contextualized_text=contextualized_text,
            bm25_keywords=bm25_keywords,
            metadata=metadata,
        )

    async def contextualize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_text: str,
    ) -> List[ContextualChunk]:
        """
        Batch contextualize multiple chunks.

        Args:
            chunks: List of dicts with 'id', 'text', and optional 'metadata'
            document_text: Full document text

        Returns:
            List of ContextualChunk objects
        """
        if not chunks:
            return []

        logger.info(
            "Contextualizing chunks",
            count=len(chunks),
            document_length=len(document_text),
        )

        # Create tasks for parallel processing
        async def process_chunk(chunk: Dict[str, Any]) -> ContextualChunk:
            return await self.contextualize_chunk(
                chunk_text=chunk.get('text', ''),
                document_text=document_text,
                chunk_id=chunk.get('id', ''),
                metadata=chunk.get('metadata'),
            )

        tasks = [process_chunk(chunk) for chunk in chunks]

        # Process with bounded concurrency
        results = await gather_with_concurrency(
            tasks,
            max_concurrent=self.config.max_concurrent_generations,
        )

        # Filter out any errors
        valid_results = [r for r in results if isinstance(r, ContextualChunk)]

        logger.info(
            "Chunks contextualized",
            total=len(chunks),
            successful=len(valid_results),
        )

        return valid_results


# =============================================================================
# Hybrid Contextual Search
# =============================================================================

async def contextual_hybrid_search(
    query: str,
    vectorstore_results: List[Any],
    bm25_results: List[Any],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combine vector and BM25 results for contextual hybrid search.

    Anthropic's research shows contextual embeddings + BM25 gives 67% error reduction.

    Args:
        query: Search query
        vectorstore_results: Results from vector search (contextualized embeddings)
        bm25_results: Results from BM25/keyword search
        vector_weight: Weight for vector scores
        bm25_weight: Weight for BM25 scores
        top_k: Number of results to return

    Returns:
        Combined and reranked results
    """
    # Build score maps
    vector_scores = {}
    bm25_scores = {}
    content_map = {}

    for r in vectorstore_results:
        chunk_id = getattr(r, 'chunk_id', r.get('chunk_id', ''))
        vector_scores[chunk_id] = getattr(r, 'score', r.get('score', 0.0))
        content_map[chunk_id] = {
            'content': getattr(r, 'content', r.get('content', '')),
            'document_id': getattr(r, 'document_id', r.get('document_id', '')),
            'metadata': getattr(r, 'metadata', r.get('metadata', {})),
        }

    for r in bm25_results:
        chunk_id = getattr(r, 'chunk_id', r.get('chunk_id', ''))
        bm25_scores[chunk_id] = getattr(r, 'score', r.get('score', 0.0))
        if chunk_id not in content_map:
            content_map[chunk_id] = {
                'content': getattr(r, 'content', r.get('content', '')),
                'document_id': getattr(r, 'document_id', r.get('document_id', '')),
                'metadata': getattr(r, 'metadata', r.get('metadata', {})),
            }

    # Normalize scores
    def normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        max_s = max(scores.values())
        min_s = min(scores.values())
        if max_s == min_s:
            return {k: 1.0 for k in scores}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

    vector_norm = normalize(vector_scores)
    bm25_norm = normalize(bm25_scores)

    # Combine scores using Reciprocal Rank Fusion style
    all_chunk_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

    combined = []
    for chunk_id in all_chunk_ids:
        v_score = vector_norm.get(chunk_id, 0.0) * vector_weight
        b_score = bm25_norm.get(chunk_id, 0.0) * bm25_weight
        combined_score = v_score + b_score

        info = content_map.get(chunk_id, {})
        combined.append({
            'chunk_id': chunk_id,
            'document_id': info.get('document_id', ''),
            'content': info.get('content', ''),
            'score': combined_score,
            'vector_score': vector_scores.get(chunk_id, 0.0),
            'bm25_score': bm25_scores.get(chunk_id, 0.0),
            'metadata': info.get('metadata', {}),
            'source': 'contextual_hybrid',
        })

    # Sort by combined score
    combined.sort(key=lambda x: x['score'], reverse=True)

    return combined[:top_k]


# =============================================================================
# Singleton Management
# =============================================================================

_contextual_service: Optional[ContextualEmbeddingService] = None
_service_lock = asyncio.Lock()


async def get_contextual_service(
    config: Optional[ContextualConfig] = None,
) -> ContextualEmbeddingService:
    """
    Get or create contextual embedding service singleton.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        ContextualEmbeddingService instance
    """
    global _contextual_service

    if _contextual_service is not None:
        return _contextual_service

    async with _service_lock:
        if _contextual_service is not None:
            return _contextual_service

        _contextual_service = ContextualEmbeddingService(config)
        return _contextual_service
