"""
AIDocumentIndexer - Response Cache Service
============================================

LLM response caching service for cost reduction and latency improvement.
Caches identical prompt/model/temperature combinations to avoid redundant API calls.

Features:
- SHA-256 prompt hashing for cache lookups
- Configurable TTL with model-specific overrides
- Temperature-based cache eligibility
- Cache statistics and hit rate tracking
- Automatic cache cleanup for expired entries
- Semantic caching: match similar queries by embedding similarity (optional)
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import CacheSettings, ResponseCache

logger = structlog.get_logger(__name__)


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic caching."""
    enabled: bool = False
    similarity_threshold: float = 0.95
    max_entries: int = 10000
    embedding_provider: str = "openai"


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    total_hits: int
    total_size_bytes: int
    hit_rate: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    estimated_savings_usd: float
    semantic_cache_entries: int = 0
    semantic_cache_hits: int = 0


@dataclass
class CachedResponse:
    """Cached response data."""
    response_text: str
    input_tokens: int
    output_tokens: int
    original_cost_usd: float
    hit_count: int
    created_at: datetime
    is_semantic_match: bool = False
    similarity_score: Optional[float] = None


class ResponseCacheService:
    """
    Service for caching LLM responses to reduce costs and latency.

    Usage:
        cache_service = ResponseCacheService()

        # Check cache before calling LLM
        cached = await cache_service.get_cached_response(
            db, prompt, model, temperature, system_prompt
        )
        if cached:
            return cached.response_text

        # Call LLM and cache response
        response = await llm.generate(prompt)
        await cache_service.cache_response(
            db, prompt, model, temperature, system_prompt,
            response, input_tokens, output_tokens, cost_usd
        )
    """

    # Default settings
    DEFAULT_TTL_SECONDS = 86400  # 24 hours
    DEFAULT_TEMPERATURE_THRESHOLD = 0.3
    MAX_PROMPT_LENGTH_FOR_CACHE = 50000  # Don't cache very long prompts

    @staticmethod
    def hash_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate SHA-256 hash of prompt for cache key.

        Args:
            prompt: The user prompt text
            system_prompt: Optional system prompt

        Returns:
            64-character hex hash string
        """
        combined = prompt
        if system_prompt:
            combined = f"{system_prompt}|||{prompt}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    @staticmethod
    def hash_system_prompt(system_prompt: str) -> str:
        """Generate SHA-256 hash of system prompt."""
        return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()

    async def get_cache_settings(self, db: AsyncSession) -> Dict[str, Any]:
        """
        Get cache settings from database.

        Args:
            db: Database session

        Returns:
            Dictionary of cache settings
        """
        try:
            result = await db.execute(select(CacheSettings).limit(1))
            settings = result.scalar_one_or_none()

            if settings:
                return {
                    "is_enabled": settings.is_enabled,
                    "default_ttl_seconds": settings.default_ttl_seconds,
                    "max_cache_size_mb": settings.max_cache_size_mb,
                    "cache_temperature_threshold": settings.cache_temperature_threshold,
                    "model_settings": settings.model_settings or {},
                    "excluded_operations": settings.excluded_operations or [],
                    # Semantic cache settings
                    "enable_semantic_cache": getattr(settings, "enable_semantic_cache", False),
                    "semantic_similarity_threshold": getattr(settings, "semantic_similarity_threshold", 0.95),
                    "max_semantic_cache_entries": getattr(settings, "max_semantic_cache_entries", 10000),
                }

            # Return defaults if no settings exist
            return {
                "is_enabled": True,
                "default_ttl_seconds": self.DEFAULT_TTL_SECONDS,
                "max_cache_size_mb": 1000,
                "cache_temperature_threshold": self.DEFAULT_TEMPERATURE_THRESHOLD,
                "model_settings": {},
                "excluded_operations": [],
                # Semantic cache defaults (OFF)
                "enable_semantic_cache": False,
                "semantic_similarity_threshold": 0.95,
                "max_semantic_cache_entries": 10000,
            }

        except Exception as e:
            logger.warning("Failed to get cache settings, using defaults", error=str(e))
            return {
                "is_enabled": True,
                "default_ttl_seconds": self.DEFAULT_TTL_SECONDS,
                "max_cache_size_mb": 1000,
                "cache_temperature_threshold": self.DEFAULT_TEMPERATURE_THRESHOLD,
                "model_settings": {},
                "excluded_operations": [],
                # Semantic cache defaults (OFF)
                "enable_semantic_cache": False,
                "semantic_similarity_threshold": 0.95,
                "max_semantic_cache_entries": 10000,
            }

    async def is_cache_eligible(
        self,
        db: AsyncSession,
        temperature: float,
        operation_type: Optional[str] = None,
        prompt_length: int = 0,
    ) -> bool:
        """
        Check if a request is eligible for caching.

        Cache is only used for low-temperature (more deterministic) requests.

        Args:
            db: Database session
            temperature: Request temperature
            operation_type: Type of operation
            prompt_length: Length of the prompt

        Returns:
            True if request should use cache
        """
        settings = await self.get_cache_settings(db)

        # Check if cache is enabled
        if not settings["is_enabled"]:
            return False

        # Check if operation is excluded
        if operation_type and operation_type in settings["excluded_operations"]:
            return False

        # Check temperature threshold
        if temperature > settings["cache_temperature_threshold"]:
            return False

        # Don't cache very long prompts
        if prompt_length > self.MAX_PROMPT_LENGTH_FOR_CACHE:
            return False

        return True

    async def get_cached_response(
        self,
        db: AsyncSession,
        prompt: str,
        model_id: str,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> Optional[CachedResponse]:
        """
        Look up cached response for a prompt.

        Args:
            db: Database session
            prompt: User prompt
            model_id: Model identifier
            temperature: Temperature setting
            system_prompt: Optional system prompt

        Returns:
            CachedResponse if found and valid, None otherwise
        """
        try:
            prompt_hash = self.hash_prompt(prompt, system_prompt)
            system_hash = self.hash_system_prompt(system_prompt) if system_prompt else None

            # Query for matching cache entry
            query = (
                select(ResponseCache)
                .where(ResponseCache.prompt_hash == prompt_hash)
                .where(ResponseCache.model_id == model_id)
                .where(ResponseCache.temperature == temperature)
            )

            if system_hash:
                query = query.where(ResponseCache.system_prompt_hash == system_hash)
            else:
                query = query.where(ResponseCache.system_prompt_hash.is_(None))

            result = await db.execute(query)
            cache_entry = result.scalar_one_or_none()

            if not cache_entry:
                logger.debug(
                    "Cache miss",
                    prompt_hash=prompt_hash[:8],
                    model=model_id,
                )
                return None

            # Check if expired
            if cache_entry.expires_at and cache_entry.expires_at < datetime.utcnow():
                logger.debug(
                    "Cache entry expired",
                    prompt_hash=prompt_hash[:8],
                    expired_at=cache_entry.expires_at,
                )
                # Delete expired entry
                await db.execute(
                    delete(ResponseCache).where(ResponseCache.id == cache_entry.id)
                )
                await db.commit()
                return None

            # Update hit count and last accessed
            await db.execute(
                update(ResponseCache)
                .where(ResponseCache.id == cache_entry.id)
                .values(
                    hit_count=ResponseCache.hit_count + 1,
                    last_accessed_at=datetime.utcnow(),
                )
            )
            await db.commit()

            logger.info(
                "Cache hit",
                prompt_hash=prompt_hash[:8],
                model=model_id,
                hit_count=cache_entry.hit_count + 1,
                saved_cost=cache_entry.original_cost_usd,
            )

            return CachedResponse(
                response_text=cache_entry.response_text,
                input_tokens=cache_entry.input_tokens,
                output_tokens=cache_entry.output_tokens,
                original_cost_usd=cache_entry.original_cost_usd or 0.0,
                hit_count=cache_entry.hit_count + 1,
                created_at=cache_entry.created_at,
            )

        except Exception as e:
            logger.error("Cache lookup failed", error=str(e))
            return None

    async def get_semantic_cached_response(
        self,
        db: AsyncSession,
        prompt: str,
        model_id: str,
        system_prompt: Optional[str] = None,
        embedding_service: Optional[Any] = None,
    ) -> Optional[CachedResponse]:
        """
        Look up cached response using semantic similarity matching.

        This enables finding cached responses for queries that are semantically
        similar but not identical (e.g., "What is the capital of France?" and
        "Tell me France's capital city").

        Args:
            db: Database session
            prompt: User prompt
            model_id: Model identifier
            system_prompt: Optional system prompt
            embedding_service: Optional embedding service instance

        Returns:
            CachedResponse if semantically similar query found, None otherwise
        """
        try:
            # Check if semantic caching is enabled
            settings = await self.get_cache_settings(db)
            if not settings.get("enable_semantic_cache", False):
                logger.debug("Semantic cache disabled")
                return None

            similarity_threshold = settings.get("semantic_similarity_threshold", 0.95)

            # Get embedding service if not provided
            if embedding_service is None:
                try:
                    import os
                    from backend.services.embeddings import get_embedding_service
                    # Use configured provider from env, with openai as fallback
                    default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
                    embedding_service = get_embedding_service(provider=default_provider, use_ray=False)
                except ImportError as e:
                    logger.warning("Could not import embedding service", error=str(e))
                    return None

            # Generate embedding for the query
            combined_query = prompt
            if system_prompt:
                combined_query = f"{system_prompt}|||{prompt}"

            try:
                query_embedding = await embedding_service.embed_text_async(combined_query)
            except Exception as e:
                logger.warning("Failed to generate query embedding", error=str(e))
                return None

            # Search for similar cached queries using vector similarity
            # Note: This uses pgvector's cosine similarity operator <=>
            # The result is cosine distance (1 - cosine_similarity), so we need to convert
            try:
                from sqlalchemy import text

                # Use raw SQL for vector similarity search
                # pgvector: <=> is cosine distance, so similarity = 1 - distance
                similarity_query = text("""
                    SELECT
                        id,
                        response_text,
                        input_tokens,
                        output_tokens,
                        original_cost_usd,
                        hit_count,
                        created_at,
                        1 - (query_embedding <=> :query_embedding) as similarity
                    FROM response_cache
                    WHERE
                        query_embedding IS NOT NULL
                        AND model_id = :model_id
                        AND (expires_at IS NULL OR expires_at > :now)
                        AND 1 - (query_embedding <=> :query_embedding) >= :threshold
                    ORDER BY similarity DESC
                    LIMIT 1
                """)

                result = await db.execute(
                    similarity_query,
                    {
                        "query_embedding": str(query_embedding),  # pgvector expects string format
                        "model_id": model_id,
                        "now": datetime.utcnow(),
                        "threshold": similarity_threshold,
                    }
                )

                row = result.fetchone()

                if row is None:
                    logger.debug(
                        "Semantic cache miss",
                        model=model_id,
                        threshold=similarity_threshold,
                    )
                    return None

                # Update hit count for the matched entry
                await db.execute(
                    update(ResponseCache)
                    .where(ResponseCache.id == row.id)
                    .values(
                        hit_count=ResponseCache.hit_count + 1,
                        last_accessed_at=datetime.utcnow(),
                    )
                )
                await db.commit()

                logger.info(
                    "Semantic cache hit",
                    model=model_id,
                    similarity=round(row.similarity, 4),
                    hit_count=row.hit_count + 1,
                    saved_cost=row.original_cost_usd,
                )

                return CachedResponse(
                    response_text=row.response_text,
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    original_cost_usd=row.original_cost_usd or 0.0,
                    hit_count=row.hit_count + 1,
                    created_at=row.created_at,
                    is_semantic_match=True,
                    similarity_score=row.similarity,
                )

            except Exception as e:
                # If pgvector is not available or query fails, fall back gracefully
                logger.warning(
                    "Semantic cache query failed (pgvector may not be available)",
                    error=str(e),
                )
                return None

        except Exception as e:
            logger.error("Semantic cache lookup failed", error=str(e))
            return None

    async def cache_response(
        self,
        db: AsyncSession,
        prompt: str,
        model_id: str,
        temperature: float,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Optional[float] = None,
        system_prompt: Optional[str] = None,
        provider_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        embedding_service: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Cache an LLM response.

        Args:
            db: Database session
            prompt: User prompt
            model_id: Model identifier
            temperature: Temperature setting
            response_text: LLM response text
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost of the request
            system_prompt: Optional system prompt
            provider_id: Provider UUID string
            ttl_seconds: Custom TTL, uses default if not specified
            embedding_service: Optional embedding service for semantic caching

        Returns:
            Cache entry ID if successful, None otherwise
        """
        try:
            settings = await self.get_cache_settings(db)

            # Determine TTL
            if ttl_seconds is None:
                # Check for model-specific TTL
                model_settings = settings.get("model_settings", {})
                if model_id in model_settings:
                    ttl_seconds = model_settings[model_id].get(
                        "ttl", settings["default_ttl_seconds"]
                    )
                else:
                    ttl_seconds = settings["default_ttl_seconds"]

            prompt_hash = self.hash_prompt(prompt, system_prompt)
            system_hash = self.hash_system_prompt(system_prompt) if system_prompt else None
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

            # Generate query embedding for semantic caching (if enabled)
            query_embedding = None
            query_text = None
            if settings.get("enable_semantic_cache", False):
                query_text = prompt
                if system_prompt:
                    query_text = f"{system_prompt}|||{prompt}"

                # Get embedding service if not provided
                if embedding_service is None:
                    try:
                        from backend.services.embeddings import get_embedding_service
                        embedding_service = get_embedding_service(provider="openai", use_ray=False)
                    except ImportError:
                        pass

                if embedding_service:
                    try:
                        query_embedding = await embedding_service.embed_text_async(query_text)
                        logger.debug("Generated query embedding for semantic cache")
                    except Exception as e:
                        logger.warning("Failed to generate query embedding", error=str(e))
                        query_embedding = None

            # Check if entry already exists (upsert)
            existing_query = (
                select(ResponseCache)
                .where(ResponseCache.prompt_hash == prompt_hash)
                .where(ResponseCache.model_id == model_id)
                .where(ResponseCache.temperature == temperature)
            )

            result = await db.execute(existing_query)
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing entry
                update_values = {
                    "response_text": response_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "original_cost_usd": cost_usd,
                    "expires_at": expires_at,
                    "last_accessed_at": datetime.utcnow(),
                }
                # Add embedding if generated and not already stored
                if query_embedding is not None and not getattr(existing, "query_embedding", None):
                    update_values["query_embedding"] = query_embedding
                    update_values["query_text"] = query_text

                await db.execute(
                    update(ResponseCache)
                    .where(ResponseCache.id == existing.id)
                    .values(**update_values)
                )
                await db.commit()

                logger.debug(
                    "Updated cache entry",
                    prompt_hash=prompt_hash[:8],
                    model=model_id,
                    has_embedding=query_embedding is not None,
                )
                return str(existing.id)

            # Create new cache entry
            cache_entry = ResponseCache(
                prompt_hash=prompt_hash,
                model_id=model_id,
                temperature=temperature,
                system_prompt_hash=system_hash,
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                original_cost_usd=cost_usd,
                expires_at=expires_at,
                provider_id=uuid.UUID(provider_id) if provider_id else None,
            )

            # Add semantic cache fields if embedding was generated
            if query_embedding is not None:
                cache_entry.query_embedding = query_embedding
                cache_entry.query_text = query_text

            db.add(cache_entry)
            await db.commit()
            await db.refresh(cache_entry)

            logger.info(
                "Cached response",
                prompt_hash=prompt_hash[:8],
                model=model_id,
                ttl_seconds=ttl_seconds,
                tokens=input_tokens + output_tokens,
                has_embedding=query_embedding is not None,
            )

            return str(cache_entry.id)

        except Exception as e:
            logger.error("Failed to cache response", error=str(e))
            await db.rollback()
            return None

    async def get_cache_stats(self, db: AsyncSession) -> CacheStats:
        """
        Get cache statistics.

        Args:
            db: Database session

        Returns:
            CacheStats object with current statistics
        """
        try:
            # Get aggregate stats
            stats_query = select(
                func.count(ResponseCache.id).label("total_entries"),
                func.sum(ResponseCache.hit_count).label("total_hits"),
                func.sum(func.length(ResponseCache.response_text)).label("total_size"),
                func.min(ResponseCache.created_at).label("oldest"),
                func.max(ResponseCache.created_at).label("newest"),
                func.sum(ResponseCache.original_cost_usd * ResponseCache.hit_count).label("savings"),
            )

            result = await db.execute(stats_query)
            row = result.one()

            total_entries = row.total_entries or 0
            total_hits = row.total_hits or 0
            total_size = row.total_size or 0

            # Calculate hit rate (hits / (hits + entries) as approximation)
            hit_rate = 0.0
            if total_entries > 0 and total_hits > 0:
                hit_rate = total_hits / (total_hits + total_entries)

            # Get semantic cache entries count (entries with embeddings)
            semantic_cache_entries = 0
            try:
                semantic_query = select(
                    func.count(ResponseCache.id)
                ).where(ResponseCache.query_embedding.isnot(None))
                semantic_result = await db.execute(semantic_query)
                semantic_cache_entries = semantic_result.scalar() or 0
            except Exception:
                # query_embedding column may not exist yet
                pass

            return CacheStats(
                total_entries=total_entries,
                total_hits=total_hits,
                total_size_bytes=total_size,
                hit_rate=hit_rate,
                oldest_entry=row.oldest,
                newest_entry=row.newest,
                estimated_savings_usd=row.savings or 0.0,
                semantic_cache_entries=semantic_cache_entries,
            )

        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return CacheStats(
                total_entries=0,
                total_hits=0,
                total_size_bytes=0,
                hit_rate=0.0,
                oldest_entry=None,
                newest_entry=None,
                estimated_savings_usd=0.0,
            )

    async def clear_cache(
        self,
        db: AsyncSession,
        model_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        older_than: Optional[datetime] = None,
    ) -> int:
        """
        Clear cache entries.

        Args:
            db: Database session
            model_id: Only clear entries for this model
            provider_id: Only clear entries for this provider
            older_than: Only clear entries older than this date

        Returns:
            Number of entries deleted
        """
        try:
            query = delete(ResponseCache)

            conditions = []
            if model_id:
                conditions.append(ResponseCache.model_id == model_id)
            if provider_id:
                conditions.append(ResponseCache.provider_id == uuid.UUID(provider_id))
            if older_than:
                conditions.append(ResponseCache.created_at < older_than)

            if conditions:
                for condition in conditions:
                    query = query.where(condition)

            result = await db.execute(query)
            await db.commit()

            deleted_count = result.rowcount

            logger.info(
                "Cleared cache",
                deleted_count=deleted_count,
                model_id=model_id,
                provider_id=provider_id,
            )

            return deleted_count

        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))
            await db.rollback()
            return 0

    async def cleanup_expired(self, db: AsyncSession) -> int:
        """
        Remove expired cache entries.

        Args:
            db: Database session

        Returns:
            Number of entries deleted
        """
        try:
            query = delete(ResponseCache).where(
                ResponseCache.expires_at < datetime.utcnow()
            )

            result = await db.execute(query)
            await db.commit()

            deleted_count = result.rowcount

            if deleted_count > 0:
                logger.info("Cleaned up expired cache entries", count=deleted_count)

            return deleted_count

        except Exception as e:
            logger.error("Failed to cleanup expired cache", error=str(e))
            await db.rollback()
            return 0

    async def update_settings(
        self,
        db: AsyncSession,
        is_enabled: Optional[bool] = None,
        default_ttl_seconds: Optional[int] = None,
        max_cache_size_mb: Optional[int] = None,
        cache_temperature_threshold: Optional[float] = None,
        model_settings: Optional[dict] = None,
        excluded_operations: Optional[list] = None,
        enable_semantic_cache: Optional[bool] = None,
        semantic_similarity_threshold: Optional[float] = None,
        max_semantic_cache_entries: Optional[int] = None,
    ) -> bool:
        """
        Update cache settings.

        Args:
            db: Database session
            is_enabled: Enable/disable cache
            default_ttl_seconds: Default TTL
            max_cache_size_mb: Maximum cache size
            cache_temperature_threshold: Temperature threshold for caching
            model_settings: Model-specific settings
            excluded_operations: Operations to exclude from caching
            enable_semantic_cache: Enable semantic cache lookups
            semantic_similarity_threshold: Similarity threshold for semantic matching
            max_semantic_cache_entries: Maximum semantic cache entries

        Returns:
            True if successful
        """
        try:
            result = await db.execute(select(CacheSettings).limit(1))
            settings = result.scalar_one_or_none()

            if not settings:
                # Create new settings
                settings = CacheSettings(
                    is_enabled=is_enabled if is_enabled is not None else True,
                    default_ttl_seconds=default_ttl_seconds or self.DEFAULT_TTL_SECONDS,
                    max_cache_size_mb=max_cache_size_mb or 1000,
                    cache_temperature_threshold=cache_temperature_threshold or self.DEFAULT_TEMPERATURE_THRESHOLD,
                    model_settings=model_settings or {},
                    excluded_operations=excluded_operations or [],
                )
                # Set semantic cache defaults
                if hasattr(settings, "enable_semantic_cache"):
                    settings.enable_semantic_cache = enable_semantic_cache if enable_semantic_cache is not None else False
                if hasattr(settings, "semantic_similarity_threshold"):
                    settings.semantic_similarity_threshold = semantic_similarity_threshold if semantic_similarity_threshold is not None else 0.95
                if hasattr(settings, "max_semantic_cache_entries"):
                    settings.max_semantic_cache_entries = max_semantic_cache_entries if max_semantic_cache_entries is not None else 10000
                db.add(settings)
            else:
                # Update existing settings
                if is_enabled is not None:
                    settings.is_enabled = is_enabled
                if default_ttl_seconds is not None:
                    settings.default_ttl_seconds = default_ttl_seconds
                if max_cache_size_mb is not None:
                    settings.max_cache_size_mb = max_cache_size_mb
                if cache_temperature_threshold is not None:
                    settings.cache_temperature_threshold = cache_temperature_threshold
                if model_settings is not None:
                    settings.model_settings = model_settings
                if excluded_operations is not None:
                    settings.excluded_operations = excluded_operations
                # Update semantic cache settings (if columns exist)
                if enable_semantic_cache is not None and hasattr(settings, "enable_semantic_cache"):
                    settings.enable_semantic_cache = enable_semantic_cache
                if semantic_similarity_threshold is not None and hasattr(settings, "semantic_similarity_threshold"):
                    settings.semantic_similarity_threshold = semantic_similarity_threshold
                if max_semantic_cache_entries is not None and hasattr(settings, "max_semantic_cache_entries"):
                    settings.max_semantic_cache_entries = max_semantic_cache_entries

            await db.commit()

            logger.info("Updated cache settings", semantic_cache_enabled=enable_semantic_cache)
            return True

        except Exception as e:
            logger.error("Failed to update cache settings", error=str(e))
            await db.rollback()
            return False


# Singleton instance
_cache_service: Optional[ResponseCacheService] = None


def get_response_cache_service() -> ResponseCacheService:
    """Get the response cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = ResponseCacheService()
    return _cache_service
