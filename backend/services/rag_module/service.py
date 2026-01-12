"""
AIDocumentIndexer - RAG Service Re-exports
============================================

Backward-compatible re-exports from the main rag.py service.
This allows gradual migration to the modular structure.

The main RAGService class remains in backend/services/rag.py for now,
as it contains significant logic that would benefit from incremental refactoring.

NOTE: This module uses lazy imports to avoid circular import issues,
since rag.py imports from rag_module (for models, config, prompts).
"""

# Lazy imports to avoid circular dependencies
_cached_exports = {}


def _get_exports():
    """Lazy load exports from rag.py to avoid circular imports."""
    global _cached_exports
    if not _cached_exports:
        from backend.services.rag import (
            RAGService as _RAGService,
            get_rag_service as _get_rag_service,
            simple_query as _simple_query,
            SearchResultCache as _SearchResultCache,
            get_search_cache as _get_search_cache,
        )
        _cached_exports = {
            "RAGService": _RAGService,
            "get_rag_service": _get_rag_service,
            "simple_query": _simple_query,
            "SearchResultCache": _SearchResultCache,
            "get_search_cache": _get_search_cache,
        }
    return _cached_exports


def __getattr__(name):
    """Lazy attribute access for re-exported symbols."""
    exports = _get_exports()
    if name in exports:
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For static type checking and IDE support
RAGService = None  # type: ignore
get_rag_service = None  # type: ignore
simple_query = None  # type: ignore
SearchResultCache = None  # type: ignore
get_search_cache = None  # type: ignore


__all__ = [
    "RAGService",
    "get_rag_service",
    "simple_query",
    "SearchResultCache",
    "get_search_cache",
]
