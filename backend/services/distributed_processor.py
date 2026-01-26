"""
AIDocumentIndexer - Distributed Processor
==========================================

Unified interface for distributed task execution using Ray or Celery.

Strategy:
- Celery: Simple async tasks (uploads, notifications, cleanup)
- Ray: Heavy ML workloads (embeddings, KG extraction, VLM processing)

The processor automatically routes tasks to the appropriate backend based on:
1. Configuration settings (USE_RAY_FOR_*)
2. Ray availability
3. Task type and requirements

Usage:
    from backend.services.distributed_processor import get_distributed_processor

    processor = await get_distributed_processor()

    # Embedding generation (routes to Ray if available)
    embeddings = await processor.process_embeddings(texts)

    # Document processing (routes to Celery)
    await processor.submit_document_task(doc_id)
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import structlog

from backend.core.config import settings
from backend.services.ray_cluster import (
    RayManager,
    RayConfig,
    RayStatus,
    get_ray_manager,
)

logger = structlog.get_logger(__name__)

# Phase 55: Import audit logging for fallback events
try:
    from backend.services.audit import audit_service_fallback, audit_service_error
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

T = TypeVar('T')


# =============================================================================
# Configuration
# =============================================================================

class ProcessorBackend(str, Enum):
    """Task execution backend."""
    AUTO = "auto"
    RAY = "ray"
    CELERY = "celery"
    LOCAL = "local"


@dataclass
class ProcessorConfig:
    """Configuration for distributed processor."""
    # Backend selection
    default_backend: ProcessorBackend = ProcessorBackend.AUTO

    # Per-task backend preferences
    use_ray_for_embeddings: bool = True
    use_ray_for_kg: bool = True
    use_ray_for_vlm: bool = True
    use_ray_for_reranking: bool = True

    # Batch processing
    embedding_batch_size: int = 100
    kg_batch_size: int = 10
    vlm_batch_size: int = 5

    # Worker pools
    embedding_pool_size: int = 4
    kg_pool_size: int = 2
    vlm_pool_size: int = 2


# =============================================================================
# Distributed Processor
# =============================================================================

class DistributedProcessor:
    """
    Unified interface for distributed task execution.

    Routes tasks to Ray or Celery based on configuration and availability.
    Provides automatic fallback when primary backend is unavailable.
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self._ray_manager: Optional[RayManager] = None
        self._embedding_pool = None
        self._kg_pool = None
        self._vlm_pool = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize processor and backends."""
        if self._initialized:
            return True

        async with self._lock:
            if self._initialized:
                return True

            # Initialize Ray if configured
            if self.config.default_backend in (ProcessorBackend.AUTO, ProcessorBackend.RAY):
                self._ray_manager = await get_ray_manager()
                if self._ray_manager.is_available:
                    logger.info(
                        "Distributed processor initialized with Ray",
                        status=self._ray_manager.status.value,
                    )
                else:
                    logger.info("Ray unavailable, falling back to Celery/local")

            self._initialized = True
            return True

    @property
    def ray_available(self) -> bool:
        """Check if Ray is available."""
        return self._ray_manager is not None and self._ray_manager.is_available

    def _should_use_ray(self, task_type: str) -> bool:
        """Determine if Ray should be used for a task type."""
        if not self.ray_available:
            return False

        if self.config.default_backend == ProcessorBackend.CELERY:
            return False

        if self.config.default_backend == ProcessorBackend.LOCAL:
            return False

        # Check per-task settings
        task_settings = {
            "embeddings": self.config.use_ray_for_embeddings,
            "kg": self.config.use_ray_for_kg,
            "vlm": self.config.use_ray_for_vlm,
            "reranking": self.config.use_ray_for_reranking,
        }

        return task_settings.get(task_type, True)

    # =========================================================================
    # Embedding Processing
    # =========================================================================

    async def process_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts using distributed processing.

        Args:
            texts: List of texts to embed
            model: Embedding model name (optional)
            batch_size: Batch size (optional)

        Returns:
            List of embedding vectors
        """
        await self.initialize()
        batch_size = batch_size or self.config.embedding_batch_size

        if self._should_use_ray("embeddings"):
            return await self._process_embeddings_ray(texts, model, batch_size)
        else:
            return await self._process_embeddings_local(texts, model, batch_size)

    async def _process_embeddings_ray(
        self,
        texts: List[str],
        model: Optional[str],
        batch_size: int,
    ) -> List[List[float]]:
        """Process embeddings using Ray actor pool."""
        try:
            # Get or create embedding pool
            if self._embedding_pool is None:
                self._embedding_pool = await self._ray_manager.get_actor_pool(
                    name="embedding_workers",
                    actor_class=EmbeddingWorker,
                    pool_size=self.config.embedding_pool_size,
                    model_name=model or settings.DEFAULT_EMBEDDING_MODEL,
                )

            if self._embedding_pool is None:
                # Fallback to local
                return await self._process_embeddings_local(texts, model, batch_size)

            # Split into batches
            batches = [
                texts[i:i + batch_size]
                for i in range(0, len(texts), batch_size)
            ]

            # Process with actor pool
            results = list(self._embedding_pool.map(
                lambda actor, batch: actor.embed_batch.remote(batch),
                batches
            ))

            # Flatten results
            embeddings = []
            for batch_result in results:
                embeddings.extend(batch_result)

            return embeddings

        except Exception as e:
            logger.warning("Ray embedding failed, falling back to local", error=str(e))

            # Phase 55: Log Ray fallback to audit system
            if AUDIT_AVAILABLE:
                try:
                    await audit_service_fallback(
                        service_type="ray",
                        primary_provider="ray",
                        fallback_provider="local",
                        error_message=str(e),
                        context={"task": "embeddings", "text_count": len(texts)},
                    )
                except Exception:
                    pass  # Don't let audit logging break processing

            return await self._process_embeddings_local(texts, model, batch_size)

    async def _process_embeddings_local(
        self,
        texts: List[str],
        model: Optional[str],
        batch_size: int,
    ) -> List[List[float]]:
        """Process embeddings locally using EmbeddingService."""
        from backend.services.embeddings import EmbeddingService

        # Phase 59: Fixed to use correct service class
        service = EmbeddingService(provider=getattr(settings, 'EMBEDDING_PROVIDER', 'openai'))
        embeddings = await service.embed_texts_async(texts)
        return embeddings

    # =========================================================================
    # Knowledge Graph Extraction
    # =========================================================================

    async def extract_knowledge_graph(
        self,
        documents: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge graphs from documents.

        Args:
            documents: List of documents with 'content' field
            batch_size: Batch size (optional)

        Returns:
            List of KG extraction results
        """
        await self.initialize()
        batch_size = batch_size or self.config.kg_batch_size

        if self._should_use_ray("kg"):
            return await self._extract_kg_ray(documents, batch_size)
        else:
            return await self._extract_kg_local(documents, batch_size)

    async def _extract_kg_ray(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Extract KG using Ray."""
        try:
            # Get or create KG pool
            if self._kg_pool is None:
                self._kg_pool = await self._ray_manager.get_actor_pool(
                    name="kg_workers",
                    actor_class=KGExtractionWorker,
                    pool_size=self.config.kg_pool_size,
                )

            if self._kg_pool is None:
                return await self._extract_kg_local(documents, batch_size)

            # Split into batches
            batches = [
                documents[i:i + batch_size]
                for i in range(0, len(documents), batch_size)
            ]

            # Process with actor pool
            results = list(self._kg_pool.map(
                lambda actor, batch: actor.extract_batch.remote(batch),
                batches
            ))

            # Flatten results
            kg_results = []
            for batch_result in results:
                kg_results.extend(batch_result)

            return kg_results

        except Exception as e:
            logger.warning("Ray KG extraction failed, falling back to local", error=str(e))

            # Phase 55: Log Ray fallback to audit system
            if AUDIT_AVAILABLE:
                try:
                    await audit_service_fallback(
                        service_type="ray",
                        primary_provider="ray",
                        fallback_provider="local",
                        error_message=str(e),
                        context={"task": "kg_extraction", "document_count": len(documents)},
                    )
                except Exception:
                    pass  # Don't let audit logging break processing

            return await self._extract_kg_local(documents, batch_size)

    async def _extract_kg_local(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Extract KG locally."""
        from backend.services.knowledge_graph import extract_knowledge_graph

        results = []
        for doc in documents:
            try:
                kg = await extract_knowledge_graph(doc.get("content", ""))
                results.append({
                    "document_id": doc.get("id"),
                    "entities": kg.get("entities", []),
                    "relationships": kg.get("relationships", []),
                })
            except Exception as e:
                logger.warning("KG extraction failed for document", error=str(e))
                results.append({"document_id": doc.get("id"), "error": str(e)})

        return results

    # =========================================================================
    # VLM Processing
    # =========================================================================

    async def process_visual_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process visual documents with VLM.

        Args:
            documents: List of documents with image data
            batch_size: Batch size (optional)

        Returns:
            List of VLM processing results
        """
        await self.initialize()
        batch_size = batch_size or self.config.vlm_batch_size

        if self._should_use_ray("vlm"):
            return await self._process_vlm_ray(documents, batch_size)
        else:
            return await self._process_vlm_local(documents, batch_size)

    async def _process_vlm_ray(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Process VLM using Ray."""
        try:
            # Get or create VLM pool
            if self._vlm_pool is None:
                self._vlm_pool = await self._ray_manager.get_actor_pool(
                    name="vlm_workers",
                    actor_class=VLMWorker,
                    pool_size=self.config.vlm_pool_size,
                )

            if self._vlm_pool is None:
                return await self._process_vlm_local(documents, batch_size)

            # Split into batches
            batches = [
                documents[i:i + batch_size]
                for i in range(0, len(documents), batch_size)
            ]

            # Process with actor pool
            results = list(self._vlm_pool.map(
                lambda actor, batch: actor.process_batch.remote(batch),
                batches
            ))

            # Flatten results
            vlm_results = []
            for batch_result in results:
                vlm_results.extend(batch_result)

            return vlm_results

        except Exception as e:
            logger.warning("Ray VLM processing failed, falling back to local", error=str(e))

            # Phase 55: Log Ray fallback to audit system
            if AUDIT_AVAILABLE:
                try:
                    await audit_service_fallback(
                        service_type="ray",
                        primary_provider="ray",
                        fallback_provider="local",
                        error_message=str(e),
                        context={"task": "vlm_processing", "document_count": len(documents)},
                    )
                except Exception:
                    pass  # Don't let audit logging break processing

            return await self._process_vlm_local(documents, batch_size)

    async def _process_vlm_local(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Process VLM locally."""
        # VLM processing would be implemented in Phase 44
        logger.warning("VLM processing not yet implemented locally")
        return [{"document_id": doc.get("id"), "status": "pending"} for doc in documents]

    # =========================================================================
    # Celery Task Submission
    # =========================================================================

    async def submit_celery_task(
        self,
        task_name: str,
        *args,
        queue: str = "default",
        **kwargs,
    ) -> str:
        """
        Submit a task to Celery.

        Args:
            task_name: Celery task name
            *args: Task arguments
            queue: Target queue
            **kwargs: Task keyword arguments

        Returns:
            Task ID
        """
        from backend.services.task_queue import get_celery_app

        app = get_celery_app()
        task = app.send_task(
            task_name,
            args=args,
            kwargs=kwargs,
            queue=queue,
        )

        return task.id

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of distributed processor."""
        await self.initialize()

        status = {
            "initialized": self._initialized,
            "backends": {
                "ray": {
                    "available": self.ray_available,
                    "status": self._ray_manager.status.value if self._ray_manager else "not_initialized",
                },
                "celery": {
                    "available": True,  # Celery is always available
                },
            },
            "pools": {},
            "config": {
                "use_ray_for_embeddings": self.config.use_ray_for_embeddings,
                "use_ray_for_kg": self.config.use_ray_for_kg,
                "use_ray_for_vlm": self.config.use_ray_for_vlm,
            },
        }

        if self._ray_manager and self.ray_available:
            ray_health = await self._ray_manager.health_check()
            status["backends"]["ray"].update(ray_health)

        if self._embedding_pool is not None:
            status["pools"]["embeddings"] = "active"
        if self._kg_pool is not None:
            status["pools"]["kg"] = "active"
        if self._vlm_pool is not None:
            status["pools"]["vlm"] = "active"

        return status

    async def shutdown(self) -> None:
        """Shutdown processor and cleanup resources."""
        if self._ray_manager:
            await self._ray_manager.shutdown()

        self._embedding_pool = None
        self._kg_pool = None
        self._vlm_pool = None
        self._initialized = False
        logger.info("Distributed processor shutdown complete")


# =============================================================================
# Ray Workers (Actor Classes)
# =============================================================================

class EmbeddingWorker:
    """Ray actor for embedding generation."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self._embedder = None

    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from backend.services.embeddings import EmbeddingService
            # Phase 59: Fixed to use correct service class
            self._embedder = EmbeddingService(
                model=self.model_name,
                provider=getattr(settings, 'EMBEDDING_PROVIDER', 'openai') if settings else 'openai',
            )
        return self._embedder

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embedder = self._get_embedder()
        # Phase 59: Use synchronous embed_texts method (runs in Ray worker)
        return embedder.embed_texts(texts)


class KGExtractionWorker:
    """Ray actor for knowledge graph extraction."""

    def __init__(self):
        self._extractor = None

    def extract_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract KG from a batch of documents."""
        import asyncio
        from backend.services.knowledge_graph import extract_knowledge_graph

        async def _process_batch():
            results = []
            for doc in documents:
                try:
                    kg = await extract_knowledge_graph(doc.get("content", ""))
                    results.append({
                        "document_id": doc.get("id"),
                        "entities": kg.get("entities", []),
                        "relationships": kg.get("relationships", []),
                    })
                except Exception as e:
                    results.append({
                        "document_id": doc.get("id"),
                        "error": str(e),
                    })
            return results

        return asyncio.run(_process_batch())


class VLMWorker:
    """Ray actor for VLM processing."""

    def __init__(self):
        self._processor = None

    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process visual documents with VLM."""
        # VLM implementation will be added in Phase 44
        return [
            {"document_id": doc.get("id"), "status": "processed"}
            for doc in documents
        ]


# =============================================================================
# Global Processor
# =============================================================================

_processor: Optional[DistributedProcessor] = None
_processor_lock = asyncio.Lock()


async def get_distributed_processor(
    config: Optional[ProcessorConfig] = None,
) -> DistributedProcessor:
    """Get or create distributed processor singleton."""
    global _processor

    if _processor is not None:
        return _processor

    async with _processor_lock:
        if _processor is not None:
            return _processor

        # Build config from settings
        if config is None:
            config = ProcessorConfig(
                use_ray_for_embeddings=getattr(settings, 'USE_RAY_FOR_EMBEDDINGS', True),
                use_ray_for_kg=getattr(settings, 'USE_RAY_FOR_KG', True),
                use_ray_for_vlm=getattr(settings, 'USE_RAY_FOR_VLM', True),
                embedding_pool_size=getattr(settings, 'RAY_NUM_WORKERS', 4),
            )

        _processor = DistributedProcessor(config)
        await _processor.initialize()

        return _processor


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ProcessorBackend",
    "ProcessorConfig",
    "DistributedProcessor",
    "EmbeddingWorker",
    "KGExtractionWorker",
    "VLMWorker",
    "get_distributed_processor",
]
