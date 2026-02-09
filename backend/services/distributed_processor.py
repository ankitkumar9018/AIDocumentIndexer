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
        """
        Process VLM (Visual Language Model) documents locally.

        Note: Local VLM processing requires either:
        1. Ray cluster for distributed processing (preferred)
        2. Ollama with a vision model (llava, bakllava) installed locally

        For visual document processing, use the multimodal_rag service directly
        or configure a Ray cluster for distributed VLM processing.
        """
        # Check if we can use Ollama for local VLM processing
        from backend.services.settings import get_settings_service

        try:
            settings_service = get_settings_service()
            vision_provider = await settings_service.get_setting("rag.vision_provider")
            ollama_model = await settings_service.get_setting("rag.ollama_vision_model")

            if vision_provider == "ollama" and ollama_model:
                # Use multimodal RAG service for local processing
                from backend.services.multimodal_rag import get_multimodal_rag_service

                multimodal_service = get_multimodal_rag_service()
                results = []

                for doc in documents:
                    try:
                        # Process document with vision model
                        doc_result = await multimodal_service.process_document(
                            document_id=doc.get("id"),
                            content=doc.get("content", ""),
                            images=doc.get("images", []),
                        )
                        results.append({
                            "document_id": doc.get("id"),
                            "status": "completed",
                            "result": doc_result,
                        })
                    except Exception as e:
                        logger.error(f"VLM processing failed for document {doc.get('id')}: {e}")
                        results.append({
                            "document_id": doc.get("id"),
                            "status": "failed",
                            "error": str(e),
                        })

                return results

        except Exception as e:
            logger.warning(f"Could not initialize local VLM processing: {e}")

        # No local VLM available - return error status
        logger.warning(
            "Local VLM processing not available. Configure Ollama with a vision model "
            "(e.g., 'ollama pull llava') or use Ray cluster for distributed VLM processing."
        )
        return [
            {
                "document_id": doc.get("id"),
                "status": "skipped",
                "error": "Local VLM processing not configured. Install Ollama with llava model or use Ray cluster.",
            }
            for doc in documents
        ]

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
    """
    Ray actor for Vision Language Model (VLM) processing.

    Supports multiple providers:
    - Ollama (local, free): llava, bakllava, llava:13b
    - OpenAI: gpt-4o, gpt-4-vision-preview
    - Anthropic: claude-3-haiku, claude-3-sonnet

    Cross-platform: Works on Mac (MPS), Linux (CUDA), Windows (CUDA/CPU).
    """

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
    ):
        """
        Initialize VLM worker.

        Args:
            provider: "ollama", "openai", "anthropic", or "auto"
            model: Model name (provider-specific), or None for default
        """
        self.provider = provider
        self.model = model
        self._client = None
        self._initialized = False

    def _initialize(self) -> bool:
        """Lazy initialize the VLM client."""
        if self._initialized:
            return self._client is not None

        self._initialized = True

        # Auto-detect provider
        if self.provider == "auto":
            self.provider, self.model = self._detect_provider()
            if self.provider is None:
                logger.warning("No VLM provider available")
                return False

        try:
            if self.provider == "ollama":
                self._init_ollama()
            elif self.provider == "openai":
                self._init_openai()
            elif self.provider == "anthropic":
                self._init_anthropic()
            else:
                logger.error("Unknown VLM provider", provider=self.provider)
                return False
            return True
        except Exception as e:
            logger.error("Failed to initialize VLM", provider=self.provider, error=str(e))
            return False

    def _detect_provider(self) -> tuple:
        """Auto-detect best available VLM provider."""
        import os

        # 1. Check Ollama (free, local)
        try:
            import ollama
            models = ollama.list()
            vision_models = [
                m["name"] for m in models.get("models", [])
                if any(v in m["name"].lower() for v in ["llava", "bakllava", "cogvlm"])
            ]
            if vision_models:
                logger.info("Auto-detected Ollama VLM", models=vision_models)
                return "ollama", vision_models[0]
        except Exception:
            pass

        # 2. Check OpenAI (only if valid API key, not placeholder)
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if (
            openai_key
            and not openai_key.startswith("sk-your-")
            and openai_key != "disabled"
            and len(openai_key) > 20
        ):
            return "openai", "gpt-4o"

        # 3. Check Anthropic (only if valid API key, not placeholder)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if (
            anthropic_key
            and not anthropic_key.startswith("sk-ant-placeholder")
            and anthropic_key != "disabled"
            and len(anthropic_key) > 20
        ):
            return "anthropic", "claude-3-haiku-20240307"

        return None, None

    def _init_ollama(self) -> None:
        """Initialize Ollama client."""
        import ollama
        self._client = ollama.Client()
        if not self.model:
            self.model = "llava"
        logger.info("Initialized Ollama VLM", model=self.model)

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        import os
        from openai import OpenAI
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.model:
            # Check settings first, then provider-specific default
            try:
                from backend.services.settings import get_settings_service
                import asyncio
                svc = get_settings_service()
                self.model = asyncio.run(svc.get_setting("rag.vlm_model")) or "gpt-4o"
            except Exception:
                self.model = "gpt-4o"
        logger.info("Initialized OpenAI VLM", model=self.model)

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        import os
        from anthropic import Anthropic
        self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if not self.model:
            # Check settings first, then provider-specific default
            try:
                from backend.services.settings import get_settings_service
                import asyncio
                svc = get_settings_service()
                self.model = asyncio.run(svc.get_setting("rag.vlm_model")) or "claude-3-haiku-20240307"
            except Exception:
                self.model = "claude-3-haiku-20240307"
        logger.info("Initialized Anthropic VLM", model=self.model)

    def analyze_image(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        image_type: str = "image",
    ) -> Dict[str, Any]:
        """
        Analyze a single image and return description.

        Args:
            image_data: Image bytes (PNG, JPEG, etc.)
            prompt: Custom prompt, or None for default
            image_type: "image", "chart", "diagram", "table"

        Returns:
            Dict with "caption", "element_type", "confidence", etc.
        """
        if not self._initialize():
            return {"error": "VLM not available", "caption": ""}

        import base64
        b64_image = base64.b64encode(image_data).decode("utf-8")

        # Build prompt based on image type
        if prompt is None:
            prompt = self._get_default_prompt(image_type)

        try:
            if self.provider == "ollama":
                return self._analyze_ollama(b64_image, prompt, image_type)
            elif self.provider == "openai":
                return self._analyze_openai(b64_image, prompt, image_type)
            elif self.provider == "anthropic":
                return self._analyze_anthropic(b64_image, prompt, image_type)
        except Exception as e:
            logger.error("VLM analysis failed", error=str(e))
            return {"error": str(e), "caption": ""}

    def _get_default_prompt(self, image_type: str) -> str:
        """Get default prompt based on image type."""
        prompts = {
            "image": "Describe this image in detail. Include key subjects, actions, colors, and any text visible.",
            "chart": "Analyze this chart. Describe the type (bar, line, pie, etc.), the data being shown, trends, and key values.",
            "diagram": "Explain this diagram step by step. Describe the components, connections, and the process or concept it illustrates.",
            "table": "Extract the data from this table. List the headers and describe the key information in the rows.",
            "screenshot": "Describe this screenshot. Identify the application, key UI elements, and any visible text or data.",
        }
        return prompts.get(image_type, prompts["image"])

    def _analyze_ollama(self, b64_image: str, prompt: str, image_type: str) -> Dict[str, Any]:
        """Analyze image using Ollama."""
        response = self._client.chat(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }],
        )
        caption = response.get("message", {}).get("content", "")
        return {
            "caption": caption,
            "element_type": image_type,
            "provider": "ollama",
            "model": self.model,
        }

    def _analyze_openai(self, b64_image: str, prompt: str, image_type: str) -> Dict[str, Any]:
        """Analyze image using OpenAI."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }],
            max_tokens=500,
        )
        caption = response.choices[0].message.content if response.choices else ""
        return {
            "caption": caption,
            "element_type": image_type,
            "provider": "openai",
            "model": self.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    def _analyze_anthropic(self, b64_image: str, prompt: str, image_type: str) -> Dict[str, Any]:
        """Analyze image using Anthropic Claude."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        caption = response.content[0].text if response.content else ""
        return {
            "caption": caption,
            "element_type": image_type,
            "provider": "anthropic",
            "model": self.model,
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
        }

    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process visual documents with VLM.

        Args:
            documents: List of dicts with:
                - id: Document/image ID
                - image_data: Image bytes
                - image_type: "image", "chart", "diagram", "table"
                - prompt: Optional custom prompt

        Returns:
            List of results with captions and metadata
        """
        if not self._initialize():
            return [
                {"document_id": doc.get("id"), "error": "VLM not available", "caption": ""}
                for doc in documents
            ]

        results = []
        for doc in documents:
            try:
                image_data = doc.get("image_data", b"")
                image_type = doc.get("image_type", "image")
                prompt = doc.get("prompt")

                result = self.analyze_image(image_data, prompt, image_type)
                result["document_id"] = doc.get("id")
                result["status"] = "success" if result.get("caption") else "empty"
                results.append(result)

            except Exception as e:
                results.append({
                    "document_id": doc.get("id"),
                    "status": "error",
                    "error": str(e),
                    "caption": "",
                })

        logger.info(
            "VLM batch processed",
            total=len(documents),
            success=sum(1 for r in results if r.get("status") == "success"),
        )
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get VLM worker status."""
        self._initialize()
        return {
            "available": self._client is not None,
            "provider": self.provider,
            "model": self.model,
        }


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
