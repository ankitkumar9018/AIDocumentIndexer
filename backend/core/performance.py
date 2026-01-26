"""
AIDocumentIndexer - Performance Utilities
==========================================

Python optimization techniques and utilities for maximum performance.

This module provides:
1. __slots__ optimized dataclasses
2. Cached decorators for expensive operations
3. Lazy import utilities
4. Memory-efficient data structures
5. Profiling context managers

Usage:
    from backend.core.performance import (
        cached_property_async,
        timed_operation,
        LazyImport,
    )
"""

import asyncio
import functools
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, TYPE_CHECKING

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


# =============================================================================
# Lazy Import Utility
# =============================================================================

class LazyImport(Generic[T]):
    """
    Lazy import for expensive modules.

    Delays import until first access, reducing startup time.

    Example:
        torch = LazyImport('torch')
        # torch is not imported yet

        model = torch.nn.Linear(10, 10)  # Now torch is imported
    """
    __slots__ = ('_module_name', '_module', '_loaded')

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module: Optional[Any] = None
        self._loaded = False

    def _load(self) -> Any:
        if not self._loaded:
            import importlib
            self._module = importlib.import_module(self._module_name)
            self._loaded = True
        return self._module

    def __getattr__(self, name: str) -> Any:
        module = self._load()
        return getattr(module, name)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "pending"
        return f"<LazyImport({self._module_name!r}, {status})>"


# =============================================================================
# Cached Async Property
# =============================================================================

def cached_property_async(func: Callable) -> property:
    """
    Async cached property decorator.

    Caches the result of an async property after first access.

    Example:
        class MyService:
            @cached_property_async
            async def expensive_data(self):
                return await fetch_expensive_data()
    """
    attr_name = f"_cached_{func.__name__}"

    @functools.wraps(func)
    async def wrapper(self):
        if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
            result = await func(self)
            setattr(self, attr_name, result)
        return getattr(self, attr_name)

    return property(lambda self: wrapper(self))


# =============================================================================
# Timing Utilities
# =============================================================================

@contextmanager
def timed_operation(operation_name: str, log_threshold_ms: float = 100.0):
    """
    Context manager for timing operations.

    Only logs if operation exceeds threshold.

    Example:
        with timed_operation("database_query"):
            result = await db.execute(query)
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000

    if elapsed_ms >= log_threshold_ms:
        logger.info(
            "Slow operation detected",
            operation=operation_name,
            duration_ms=round(elapsed_ms, 2),
        )


async def timed_async(
    coro,
    operation_name: str,
    log_threshold_ms: float = 100.0,
):
    """
    Time an async operation.

    Example:
        result = await timed_async(
            fetch_data(),
            "fetch_data",
            log_threshold_ms=50.0
        )
    """
    start = time.perf_counter()
    result = await coro
    elapsed_ms = (time.perf_counter() - start) * 1000

    if elapsed_ms >= log_threshold_ms:
        logger.info(
            "Slow async operation",
            operation=operation_name,
            duration_ms=round(elapsed_ms, 2),
        )

    return result


# =============================================================================
# Memory-Efficient Data Structures
# =============================================================================

@dataclass(slots=True, frozen=True)
class ImmutableResult:
    """
    Immutable, memory-efficient result container.

    Uses __slots__ for 40-50% memory reduction.
    Frozen for hashability and thread safety.
    """
    value: Any
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class ChunkData:
    """
    Memory-efficient chunk data structure.

    Used throughout the codebase for document chunks.
    __slots__ reduces memory by 40-50% per instance.
    """
    id: str
    content: str
    document_id: str
    embedding: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    score: float = 0.0


@dataclass(slots=True)
class SearchHit:
    """
    Memory-efficient search result.

    Optimized for high-volume search operations.
    """
    chunk_id: str
    document_id: str
    score: float
    content: str
    rank: int = 0


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def chunked_iter(iterable, size: int):
    """
    Iterate over iterable in chunks of given size.

    Memory-efficient: doesn't load entire list into memory.

    Example:
        for batch in chunked_iter(documents, 100):
            await process_batch(batch)
    """
    iterator = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(iterator))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk


async def gather_with_concurrency(
    tasks,
    max_concurrent: int = 10,
    return_exceptions: bool = True,
):
    """
    Run async tasks with bounded concurrency.

    Prevents overwhelming resources with too many concurrent operations.

    Example:
        results = await gather_with_concurrency(
            [fetch(url) for url in urls],
            max_concurrent=5,
        )
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_task(task):
        async with semaphore:
            return await task

    bounded_tasks = [bounded_task(t) for t in tasks]
    return await asyncio.gather(*bounded_tasks, return_exceptions=return_exceptions)


# =============================================================================
# Caching Utilities
# =============================================================================

class LRUCache(Generic[T]):
    """
    Simple LRU cache with __slots__ for efficiency.

    Thread-safe through asyncio lock.
    """
    __slots__ = ('_capacity', '_cache', '_lock')

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._cache: Dict[str, T] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """Get value if exists (updates access order)."""
        async with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None

    async def set(self, key: str, value: T) -> None:
        """Set value, evicting oldest if at capacity."""
        async with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._capacity:
                # Remove oldest (first) item
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = value

    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# =============================================================================
# String Interning
# =============================================================================

_interned_strings: Dict[str, str] = {}


def intern_string(s: str) -> str:
    """
    Intern a string for memory efficiency.

    Reuses string instances for repeated values.
    Useful for document IDs, collection names, etc.

    Example:
        doc_id = intern_string(doc_id)  # Reuses existing string if same value
    """
    if s not in _interned_strings:
        _interned_strings[s] = s
    return _interned_strings[s]


def clear_interned_strings() -> None:
    """Clear interned strings cache."""
    global _interned_strings
    _interned_strings = {}


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Lazy imports
    "LazyImport",
    # Caching
    "cached_property_async",
    "LRUCache",
    # Timing
    "timed_operation",
    "timed_async",
    # Data structures
    "ImmutableResult",
    "ChunkData",
    "SearchHit",
    # Batch processing
    "chunked_iter",
    "gather_with_concurrency",
    # String interning
    "intern_string",
    "clear_interned_strings",
]
