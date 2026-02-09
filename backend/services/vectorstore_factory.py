"""
AIDocumentIndexer - Vector Store Factory
=========================================

Factory for creating vector store instances based on configuration.

Supports multiple backends:
- pgvector: PostgreSQL with pgvector extension (default, included)
- qdrant: Qdrant vector database (for 1-50M scale)
- milvus: Milvus distributed vector database (for 50M+ scale)

All backends are open-source:
- pgvector: PostgreSQL extension (PostgreSQL license)
- Qdrant: Apache 2.0
- Milvus: Apache 2.0

Settings-aware: Respects vector_store.backend setting.
"""

import os
from enum import Enum
from typing import Optional, Dict, Any, Protocol, List, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


class VectorStoreBackend(str, Enum):
    """Available vector store backends."""
    PGVECTOR = "pgvector"  # PostgreSQL + pgvector (default)
    QDRANT = "qdrant"      # Qdrant vector database
    MILVUS = "milvus"      # Milvus distributed vector database
    AUTO = "auto"          # Auto-detect based on available resources


# Cached settings
_vector_backend: Optional[str] = None


async def _get_vector_store_settings() -> str:
    """Get vector store backend from settings (DB first, env fallback)."""
    global _vector_backend

    if _vector_backend is not None:
        return _vector_backend

    # Settings DB takes priority over env vars
    try:
        from backend.services.settings import get_settings_service

        settings = get_settings_service()
        backend = await settings.get_setting("vector_store.backend")

        if backend and backend in ("pgvector", "qdrant", "milvus"):
            _vector_backend = backend
            return _vector_backend
    except Exception as e:
        logger.debug("Could not load vector store settings", error=str(e))

    # Fallback to environment variable
    env_backend = os.getenv("VECTOR_STORE_BACKEND", "").lower()
    if env_backend in ["pgvector", "qdrant", "milvus"]:
        _vector_backend = env_backend
        return _vector_backend

    _vector_backend = "pgvector"
    return _vector_backend


def invalidate_vector_store_settings():
    """Invalidate cached settings."""
    global _vector_backend
    _vector_backend = None


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol defining the vector store interface."""

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        access_tier_id: str,
        **kwargs,
    ) -> List[str]:
        """Add chunks with embeddings to the store."""
        ...

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        access_tier_level: int = 100,
        document_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Any]:
        """Search for similar chunks."""
        ...

    async def delete_document_chunks(
        self,
        document_id: str,
        **kwargs,
    ) -> int:
        """Delete all chunks for a document."""
        ...


class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    Usage:
        # Get the configured backend
        store = await VectorStoreFactory.create()

        # Or specify a backend
        store = await VectorStoreFactory.create(backend="qdrant")
    """

    _instances: Dict[str, Any] = {}

    @classmethod
    async def create(
        cls,
        backend: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        force_new: bool = False,
    ) -> Any:
        """
        Create or get a vector store instance.

        Args:
            backend: Vector store backend (pgvector, qdrant, milvus, auto)
            config: Optional backend-specific configuration
            force_new: Create new instance even if one exists

        Returns:
            Vector store instance implementing VectorStoreProtocol
        """
        # Determine backend
        if backend is None or backend == "auto":
            backend = await _get_vector_store_settings()

        # Normalize
        backend = backend.lower()

        # Check cache
        cache_key = f"{backend}:{hash(str(config))}"
        if not force_new and cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create instance
        if backend == "pgvector":
            instance = cls._create_pgvector(config)
        elif backend == "qdrant":
            instance = await cls._create_qdrant(config)
        elif backend == "milvus":
            instance = await cls._create_milvus(config)
        else:
            logger.warning(f"Unknown backend {backend}, falling back to pgvector")
            instance = cls._create_pgvector(config)

        cls._instances[cache_key] = instance
        logger.info("Created vector store", backend=backend)

        return instance

    @classmethod
    def _create_pgvector(cls, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create pgvector (PostgreSQL) vector store."""
        from backend.services.vectorstore import VectorStore, VectorStoreConfig

        vs_config = VectorStoreConfig()

        if config:
            if "default_top_k" in config:
                vs_config.default_top_k = config["default_top_k"]
            if "similarity_threshold" in config:
                vs_config.similarity_threshold = config["similarity_threshold"]
            if "enable_reranking" in config:
                vs_config.enable_reranking = config["enable_reranking"]

        return VectorStore(config=vs_config)

    @classmethod
    async def _create_qdrant(cls, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create Qdrant vector store. Reads config from settings if not provided."""
        try:
            from backend.services.vectorstore_qdrant import QdrantVectorStore

            if config is None:
                # Read connection config from settings (DB first, env fallback)
                try:
                    from backend.services.settings import get_settings_service
                    settings_svc = get_settings_service()
                    config = {
                        "url": await settings_svc.get_setting("vector_store.qdrant_url") or os.getenv("QDRANT_URL", "localhost:6333"),
                        "api_key": await settings_svc.get_setting("vector_store.qdrant_api_key") or os.getenv("QDRANT_API_KEY", ""),
                        "collection": await settings_svc.get_setting("vector_store.qdrant_collection") or "documents",
                    }
                except Exception:
                    config = {}

            url = config.get("url", os.getenv("QDRANT_URL", "localhost:6333"))
            api_key = config.get("api_key", os.getenv("QDRANT_API_KEY", ""))
            collection = config.get("collection", "documents")

            store = QdrantVectorStore(url=url, collection_name=collection, api_key=api_key if api_key else None)
            await store.initialize()

            return store

        except ImportError:
            logger.error("Qdrant not available. Install with: pip install qdrant-client")
            logger.warning("Falling back to pgvector")
            return cls._create_pgvector(config)

    @classmethod
    async def _create_milvus(cls, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create Milvus vector store. Reads config from settings if not provided."""
        try:
            from backend.services.vectorstore_milvus import MilvusVectorStore

            if config is None:
                # Read connection config from settings (DB first, env fallback)
                try:
                    from backend.services.settings import get_settings_service
                    settings_svc = get_settings_service()
                    config = {
                        "host": await settings_svc.get_setting("vector_store.milvus_host") or os.getenv("MILVUS_HOST", "localhost"),
                        "port": int(await settings_svc.get_setting("vector_store.milvus_port") or os.getenv("MILVUS_PORT", "19530")),
                        "collection": await settings_svc.get_setting("vector_store.milvus_collection") or "documents",
                    }
                except Exception:
                    config = {}

            host = config.get("host", os.getenv("MILVUS_HOST", "localhost"))
            port = config.get("port", int(os.getenv("MILVUS_PORT", "19530")))
            collection = config.get("collection", "documents")

            store = MilvusVectorStore(host=host, port=port, collection_name=collection)
            await store.initialize()

            return store

        except ImportError:
            logger.error("Milvus not available. Install with: pip install pymilvus")
            logger.warning("Falling back to pgvector")
            return cls._create_pgvector(config)

    @classmethod
    def clear_cache(cls):
        """Clear the instance cache."""
        cls._instances.clear()


async def get_vector_store(
    backend: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function to get a vector store instance.

    Args:
        backend: Optional backend override
        config: Optional configuration

    Returns:
        Vector store instance
    """
    return await VectorStoreFactory.create(backend=backend, config=config)
