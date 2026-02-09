"""
AIDocumentIndexer - Cache Management API Routes
================================================

API endpoints for managing the generative cache system.
"""

from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import get_current_user, require_admin, AdminUser
from backend.db.models import User
from backend.services.generative_cache import (
    get_generative_cache,
    CacheConfig,
    CacheTier,
    ContentType,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    total_entries: int
    total_lookups: int
    hits: int
    misses: int
    hit_rate: float
    hits_by_tier: Dict[str, int]
    tokens_saved: int
    cost_savings_estimate: float
    avg_lookup_time_ms: float


class CacheConfigResponse(BaseModel):
    """Cache configuration response."""
    backend: str
    ttl_seconds: int
    default_threshold: float
    thresholds: Dict[str, float]
    enable_prefix_cache: bool
    max_cache_size: int
    use_compression: bool


class InvalidateCacheRequest(BaseModel):
    """Request to invalidate cache entries."""
    query: str = Field(..., description="Query to invalidate")
    context: Optional[str] = Field(None, description="Optional context to match")


class ClearCacheResponse(BaseModel):
    """Response from clearing cache."""
    entries_cleared: int
    success: bool


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    current_user: User = Depends(get_current_user),
):
    """
    Get cache statistics including hit rates and cost savings.

    Returns metrics on cache performance:
    - Hit rate by tier (exact, semantic, prefix)
    - Total tokens saved
    - Estimated cost savings
    - Average lookup latency
    """
    try:
        cache = await get_generative_cache()
        stats = cache.get_stats()

        return CacheStatsResponse(
            total_entries=stats.total_entries,
            total_lookups=stats.total_lookups,
            hits=stats.hits,
            misses=stats.misses,
            hit_rate=stats.hit_rate,
            hits_by_tier=stats.hits_by_tier,
            tokens_saved=stats.tokens_saved,
            cost_savings_estimate=stats.cost_savings_estimate,
            avg_lookup_time_ms=stats.avg_lookup_time_ms,
        )
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get cache stats: {str(e)}")


@router.get("/config", response_model=CacheConfigResponse)
async def get_cache_config(
    current_user: User = Depends(get_current_user),
):
    """
    Get current cache configuration.

    Returns the active cache settings including:
    - Storage backend (redis/memory)
    - TTL settings
    - Similarity thresholds per content type
    - Prefix caching settings
    """
    try:
        cache = await get_generative_cache()
        config = cache.config

        return CacheConfigResponse(
            backend=config.backend,
            ttl_seconds=config.ttl_seconds,
            default_threshold=config.default_threshold,
            thresholds=config.thresholds,
            enable_prefix_cache=config.enable_prefix_cache,
            max_cache_size=config.max_cache_size,
            use_compression=config.use_compression,
        )
    except Exception as e:
        logger.error("Failed to get cache config", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get cache config: {str(e)}")


@router.post("/invalidate")
async def invalidate_cache_entry(
    request: InvalidateCacheRequest,
    _user: AdminUser,
) -> Dict[str, Any]:
    """
    Invalidate a specific cache entry (admin only).

    Use this to remove stale or incorrect cached responses.
    """
    logger.info(
        "Invalidating cache entry",
        query_preview=request.query[:50],
        has_context=request.context is not None,
    )

    try:
        cache = await get_generative_cache()
        success = await cache.invalidate(request.query, request.context)

        return {
            "success": success,
            "query": request.query[:100],
            "message": "Cache entry invalidated" if success else "Entry not found",
        }
    except Exception as e:
        logger.error("Failed to invalidate cache", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to invalidate: {str(e)}")


@router.post("/clear", response_model=ClearCacheResponse)
async def clear_cache(
    _user: AdminUser,
) -> ClearCacheResponse:
    """
    Clear all cache entries (admin only).

    Warning: This removes all cached responses. Use with caution.
    """
    logger.warning("Clearing all cache entries")

    try:
        cache = await get_generative_cache()
        entries_cleared = await cache.clear()

        logger.info("Cache cleared", entries_cleared=entries_cleared)

        return ClearCacheResponse(
            entries_cleared=entries_cleared,
            success=True,
        )
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to clear cache: {str(e)}")


@router.get("/tiers")
async def list_cache_tiers() -> Dict[str, Any]:
    """
    List available cache tiers and their descriptions.
    """
    return {
        "tiers": [
            {
                "tier": CacheTier.EXACT.value,
                "name": "Exact Match",
                "description": "Hash-based exact match (instant, 100% accuracy)",
                "typical_hit_rate": "10-20%",
            },
            {
                "tier": CacheTier.SEMANTIC.value,
                "name": "Semantic Match",
                "description": "Embedding-based similarity match (fast, 95%+ accuracy)",
                "typical_hit_rate": "40-60%",
            },
            {
                "tier": CacheTier.PREFIX.value,
                "name": "Prefix Match",
                "description": "Context prefix matching (Anthropic-style)",
                "typical_hit_rate": "10-20%",
            },
        ],
        "content_types": [
            {
                "type": ct.value,
                "description": _get_content_type_description(ct),
            }
            for ct in ContentType
        ],
    }


@router.get("/health")
async def cache_health() -> Dict[str, Any]:
    """Check cache service health."""
    try:
        cache = await get_generative_cache()
        stats = cache.get_stats()

        return {
            "status": "healthy",
            "backend": cache.config.backend,
            "total_lookups": stats.total_lookups,
            "hit_rate": f"{stats.hit_rate:.1%}",
            "tokens_saved": stats.tokens_saved,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def _get_content_type_description(ct: ContentType) -> str:
    """Get description for content type."""
    descriptions = {
        ContentType.FACTUAL: "Facts and definitions (high similarity threshold)",
        ContentType.ANALYTICAL: "Analysis and reasoning (medium threshold)",
        ContentType.CREATIVE: "Creative writing (lower threshold)",
        ContentType.CODE: "Code generation (high threshold)",
        ContentType.CONVERSATIONAL: "Conversational responses (medium threshold)",
        ContentType.DEFAULT: "Default content type",
    }
    return descriptions.get(ct, "Unknown content type")


# =============================================================================
# Memory Monitoring Endpoints (All Caches)
# =============================================================================

class CacheMemoryStats(BaseModel):
    """Statistics for a single cache."""
    name: str
    entries: int
    max_entries: int
    utilization_percent: float
    estimated_memory_mb: float
    has_lru: bool


class AllCachesMemoryStats(BaseModel):
    """Combined memory statistics for all caches."""
    total_memory_mb: float
    process_memory_mb: float
    caches: list[CacheMemoryStats]
    warnings: list[str]


@router.get("/memory/stats", response_model=AllCachesMemoryStats)
async def get_all_caches_memory_stats(
    current_user: User = Depends(get_current_user),
) -> AllCachesMemoryStats:
    """
    Get memory statistics for ALL caches in the system.

    Returns detailed stats on:
    - Embedding cache
    - Session memory
    - Query cache
    - Generative cache
    - AMem cache
    - Process memory usage
    """
    import psutil

    caches = []
    warnings = []
    total_memory_mb = 0.0

    # 1. Embedding Cache
    try:
        from backend.services.embeddings import _embedding_cache, _CACHE_MAX_SIZE
        entries = len(_embedding_cache)
        max_entries = _CACHE_MAX_SIZE
        # Each embedding ~768 floats × 4 bytes = 3KB
        est_memory = (entries * 3) / 1024  # MB
        total_memory_mb += est_memory

        caches.append(CacheMemoryStats(
            name="Embedding Cache",
            entries=entries,
            max_entries=max_entries,
            utilization_percent=(entries / max_entries * 100) if max_entries > 0 else 0,
            estimated_memory_mb=round(est_memory, 2),
            has_lru=True,
        ))

        if entries > max_entries * 0.9:
            warnings.append("Embedding cache is >90% full")
    except Exception as e:
        logger.warning("Failed to get embedding cache stats", error=str(e))

    # 2. Session Memory
    try:
        from backend.services.session_memory import get_session_memory_manager
        manager = get_session_memory_manager()
        stats = manager.get_stats()
        entries = stats.get("active_sessions", 0)
        max_entries = manager.max_sessions
        # Each session ~50KB average
        est_memory = (entries * 50) / 1024  # MB
        total_memory_mb += est_memory

        caches.append(CacheMemoryStats(
            name="Session Memory",
            entries=entries,
            max_entries=max_entries,
            utilization_percent=(entries / max_entries * 100) if max_entries > 0 else 0,
            estimated_memory_mb=round(est_memory, 2),
            has_lru=True,
        ))

        if entries > max_entries * 0.9:
            warnings.append("Session memory is >90% full")
    except Exception as e:
        logger.warning("Failed to get session memory stats", error=str(e))

    # 3. Query Cache
    try:
        from backend.services.query_cache import get_query_cache
        cache = get_query_cache()
        entries = len(cache._cache)
        max_entries = cache.max_entries
        # Each query cache entry ~50KB
        est_memory = (entries * 50) / 1024  # MB
        total_memory_mb += est_memory

        caches.append(CacheMemoryStats(
            name="Query Cache",
            entries=entries,
            max_entries=max_entries,
            utilization_percent=(entries / max_entries * 100) if max_entries > 0 else 0,
            estimated_memory_mb=round(est_memory, 2),
            has_lru=True,
        ))

        if entries > max_entries * 0.9:
            warnings.append("Query cache is >90% full")
    except Exception as e:
        logger.warning("Failed to get query cache stats", error=str(e))

    # 4. Generative Cache
    try:
        cache = await get_generative_cache()
        stats = cache.get_stats()
        entries = stats.total_entries
        max_entries = cache.config.max_cache_size
        # Each entry ~100KB
        est_memory = (entries * 100) / 1024  # MB
        total_memory_mb += est_memory

        caches.append(CacheMemoryStats(
            name="Generative Cache",
            entries=entries,
            max_entries=max_entries,
            utilization_percent=(entries / max_entries * 100) if max_entries > 0 else 0,
            estimated_memory_mb=round(est_memory, 2),
            has_lru=True,
        ))
    except Exception as e:
        logger.warning("Failed to get generative cache stats", error=str(e))

    # Get process memory
    try:
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / 1024 / 1024
    except Exception:
        process_memory_mb = 0.0

    # Memory warning
    if process_memory_mb > 4000:  # 4GB
        warnings.append(f"High process memory: {process_memory_mb:.0f}MB")

    return AllCachesMemoryStats(
        total_memory_mb=round(total_memory_mb, 2),
        process_memory_mb=round(process_memory_mb, 2),
        caches=caches,
        warnings=warnings,
    )


@router.post("/memory/clear-all")
async def clear_all_caches(
    _user: AdminUser,
) -> Dict[str, Any]:
    """
    Clear ALL caches to free memory (admin only).

    Clears:
    - Embedding cache
    - Session memory
    - Query cache
    - Generative cache

    Warning: This will cause temporary performance degradation
    as caches need to be rebuilt.
    """
    results = {}
    total_cleared = 0

    # 1. Clear Embedding Cache
    try:
        from backend.services.embeddings import _embedding_cache
        count = len(_embedding_cache)
        _embedding_cache._cache.clear()
        results["embedding_cache"] = {"cleared": count, "success": True}
        total_cleared += count
    except Exception as e:
        results["embedding_cache"] = {"error": str(e), "success": False}

    # 2. Clear Session Memory
    try:
        from backend.services.session_memory import get_session_memory_manager
        manager = get_session_memory_manager()
        count = len(manager._memory_store)
        manager.clear_all()
        results["session_memory"] = {"cleared": count, "success": True}
        total_cleared += count
    except Exception as e:
        results["session_memory"] = {"error": str(e), "success": False}

    # 3. Clear Query Cache
    try:
        from backend.services.query_cache import get_query_cache
        import asyncio
        cache = get_query_cache()
        count = len(cache._cache)
        cache._cache.clear()
        cache._hits = 0
        cache._misses = 0
        results["query_cache"] = {"cleared": count, "success": True}
        total_cleared += count
    except Exception as e:
        results["query_cache"] = {"error": str(e), "success": False}

    # 4. Clear Generative Cache
    try:
        cache = await get_generative_cache()
        count = await cache.clear()
        results["generative_cache"] = {"cleared": count, "success": True}
        total_cleared += count
    except Exception as e:
        results["generative_cache"] = {"error": str(e), "success": False}

    # 5. Clear Phase 65 Semantic Cache (spell correction + query cache)
    try:
        from backend.services.phase65_integration import get_phase65_pipeline_sync
        p65 = get_phase65_pipeline_sync()
        if p65 and p65._semantic_cache:
            count = await p65._semantic_cache.clear()
            results["phase65_semantic_cache"] = {"cleared": count, "success": True}
            total_cleared += count
        else:
            results["phase65_semantic_cache"] = {"cleared": 0, "success": True}
    except Exception as e:
        results["phase65_semantic_cache"] = {"error": str(e), "success": False}

    # 6. Force garbage collection
    import gc
    gc.collect()

    # Get new memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_after_mb = process.memory_info().rss / 1024 / 1024
    except Exception:
        memory_after_mb = 0

    logger.warning(
        "All caches cleared",
        total_cleared=total_cleared,
        memory_after_mb=round(memory_after_mb, 2),
    )

    return {
        "success": True,
        "total_entries_cleared": total_cleared,
        "memory_after_mb": round(memory_after_mb, 2),
        "details": results,
    }
