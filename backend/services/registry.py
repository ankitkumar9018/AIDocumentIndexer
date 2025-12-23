"""
AIDocumentIndexer - Service Registry
=====================================

Centralized service management for singletons.
Provides better testability and avoids circular imports.
"""

from typing import Optional, Dict, Any, TypeVar, Type, Callable
import threading
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ServiceRegistry:
    """
    Centralized registry for service singletons.

    Features:
    - Thread-safe singleton management
    - Lazy initialization
    - Service replacement for testing
    - Cleanup support

    Usage:
        # Register a service factory
        registry.register("rag", lambda: RAGService())

        # Get a service
        rag = registry.get("rag")

        # Replace for testing
        registry.replace("rag", mock_rag_service)
    """

    _instance: Optional["ServiceRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ServiceRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if getattr(self, "_initialized", False):
            return

        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._service_lock = threading.Lock()
        self._initialized = True

        logger.info("Service registry initialized")

    def register(self, name: str, factory: Callable[[], T]) -> None:
        """
        Register a service factory.

        Args:
            name: Service identifier
            factory: Callable that creates the service instance
        """
        with self._service_lock:
            self._factories[name] = factory
            logger.debug("Service registered", service=name)

    def get(self, name: str) -> Any:
        """
        Get a service instance (creates lazily if needed).

        Args:
            name: Service identifier

        Returns:
            Service instance

        Raises:
            KeyError: If service not registered
        """
        if name in self._services:
            return self._services[name]

        with self._service_lock:
            # Double-check after acquiring lock
            if name in self._services:
                return self._services[name]

            if name not in self._factories:
                raise KeyError(f"Service '{name}' not registered")

            # Create instance
            logger.debug("Creating service instance", service=name)
            self._services[name] = self._factories[name]()
            return self._services[name]

    def get_optional(self, name: str) -> Optional[Any]:
        """
        Get a service instance if registered, None otherwise.

        Args:
            name: Service identifier

        Returns:
            Service instance or None
        """
        try:
            return self.get(name)
        except KeyError:
            return None

    def replace(self, name: str, instance: Any) -> None:
        """
        Replace a service instance (useful for testing).

        Args:
            name: Service identifier
            instance: Replacement instance
        """
        with self._service_lock:
            self._services[name] = instance
            logger.debug("Service replaced", service=name)

    def reset(self, name: str) -> None:
        """
        Reset a service (removes instance, will be recreated on next get).

        Args:
            name: Service identifier
        """
        with self._service_lock:
            if name in self._services:
                del self._services[name]
                logger.debug("Service reset", service=name)

    def reset_all(self) -> None:
        """Reset all service instances."""
        with self._service_lock:
            self._services.clear()
            logger.debug("All services reset")

    def is_registered(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._factories

    def is_initialized(self, name: str) -> bool:
        """Check if a service instance exists."""
        return name in self._services

    def list_services(self) -> Dict[str, bool]:
        """
        List all registered services and their initialization status.

        Returns:
            Dict of service_name -> is_initialized
        """
        return {
            name: name in self._services
            for name in self._factories
        }


# Global registry instance
_registry = ServiceRegistry()


def get_registry() -> ServiceRegistry:
    """Get the global service registry."""
    return _registry


# =============================================================================
# Service Names (constants to avoid typos)
# =============================================================================

class Services:
    """Service name constants."""
    RAG = "rag"
    EMBEDDING = "embedding"
    VECTOR_STORE = "vector_store"
    GENERATOR = "generator"
    IMAGE_GENERATOR = "image_generator"
    SUMMARIZER = "summarizer"
    QUERY_EXPANDER = "query_expander"
    VERIFIER = "verifier"
    PIPELINE = "pipeline"
    SCRAPER = "scraper"
    LLM_CONFIG = "llm_config"


# =============================================================================
# Helper Functions for Common Services
# =============================================================================

def get_rag_service():
    """Get the RAG service instance."""
    if not _registry.is_registered(Services.RAG):
        from backend.services.rag import RAGService
        _registry.register(Services.RAG, RAGService)
    return _registry.get(Services.RAG)


def get_embedding_service():
    """Get the embedding service instance."""
    if not _registry.is_registered(Services.EMBEDDING):
        from backend.services.embeddings import EmbeddingService
        _registry.register(Services.EMBEDDING, EmbeddingService)
    return _registry.get(Services.EMBEDDING)


def get_generator_service():
    """Get the document generator service instance."""
    if not _registry.is_registered(Services.GENERATOR):
        from backend.services.generator import DocumentGenerationService
        _registry.register(Services.GENERATOR, DocumentGenerationService)
    return _registry.get(Services.GENERATOR)


def get_summarizer_service():
    """Get the summarizer service instance."""
    if not _registry.is_registered(Services.SUMMARIZER):
        from backend.services.summarizer import Summarizer
        _registry.register(Services.SUMMARIZER, Summarizer)
    return _registry.get(Services.SUMMARIZER)


def get_query_expander_service():
    """Get the query expander service instance."""
    if not _registry.is_registered(Services.QUERY_EXPANDER):
        from backend.services.query_expander import get_query_expander
        _registry.register(Services.QUERY_EXPANDER, get_query_expander)
    return _registry.get(Services.QUERY_EXPANDER)
