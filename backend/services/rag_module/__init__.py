"""
AIDocumentIndexer - RAG Module
===============================

Retrieval-Augmented Generation service using LangChain.
Provides hybrid search (vector + keyword) and conversational RAG chains.

This module is organized as follows:

- models.py: Data models (Source, RAGResponse, StreamChunk)
- config.py: RAGConfig class
- prompts.py: System prompts and language configuration
- service.py: Main RAGService class (re-exports from rag.py)

Usage:
    from backend.services.rag_module import (
        RAGService,
        get_rag_service,
        RAGConfig,
        RAGResponse,
        Source,
    )

    service = get_rag_service()
    response = await service.query("What is the quarterly revenue?")
"""

from backend.services.rag_module.models import (
    Source,
    RAGResponse,
    StreamChunk,
)
from backend.services.rag_module.config import (
    RAGConfig,
)
from backend.services.rag_module.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    CONVERSATIONAL_RAG_TEMPLATE,
    LANGUAGE_NAMES,
    get_language_instruction,
    parse_suggested_questions,
)

# Lazy imports for service to avoid circular imports with rag.py
# The main RAGService still lives in backend/services/rag.py during migration
def __getattr__(name):
    """Lazy import for RAGService, get_rag_service, simple_query."""
    if name in ("RAGService", "get_rag_service", "simple_query", "SearchResultCache", "get_search_cache"):
        from backend.services.rag_module.service import (
            RAGService,
            get_rag_service,
            simple_query,
            SearchResultCache,
            get_search_cache,
        )
        return {
            "RAGService": RAGService,
            "get_rag_service": get_rag_service,
            "simple_query": simple_query,
            "SearchResultCache": SearchResultCache,
            "get_search_cache": get_search_cache,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Models
    "Source",
    "RAGResponse",
    "StreamChunk",
    # Config
    "RAGConfig",
    # Prompts
    "RAG_SYSTEM_PROMPT",
    "RAG_PROMPT_TEMPLATE",
    "CONVERSATIONAL_RAG_TEMPLATE",
    "LANGUAGE_NAMES",
    "get_language_instruction",
    "parse_suggested_questions",
    # Service (lazy loaded)
    "RAGService",
    "get_rag_service",
    "simple_query",
    "SearchResultCache",
    "get_search_cache",
]
