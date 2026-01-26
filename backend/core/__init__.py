"""
AIDocumentIndexer - Core Module
================================

Core configuration and utilities for the backend.
"""

from backend.core.config import settings

# Performance utilities for optimization
from backend.core.performance import (
    LazyImport,
    cached_property_async,
    LRUCache,
    timed_operation,
    timed_async,
    ImmutableResult,
    ChunkData,
    SearchHit,
    chunked_iter,
    gather_with_concurrency,
    intern_string,
    clear_interned_strings,
)

__all__ = [
    "settings",
    # Performance utilities
    "LazyImport",
    "cached_property_async",
    "LRUCache",
    "timed_operation",
    "timed_async",
    "ImmutableResult",
    "ChunkData",
    "SearchHit",
    "chunked_iter",
    "gather_with_concurrency",
    "intern_string",
    "clear_interned_strings",
]
