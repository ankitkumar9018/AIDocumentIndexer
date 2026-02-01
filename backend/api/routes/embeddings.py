"""
Embedding System API Routes
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select, func, Integer, and_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import os
import asyncio
from datetime import datetime
import threading
import uuid as uuid_module

from backend.db.database import get_async_session
from backend.db.models import Document, Chunk, get_embedding_dimension
from backend.api.middleware.auth import get_current_user, CurrentUser
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Background Job Tracking
# =============================================================================

class BackgroundJobStatus:
    """Track status of background embedding jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_type: str, user_id: str) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid_module.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "type": job_type,
                "user_id": user_id,
                "status": "pending",
                "progress": 0,
                "total": 0,
                "processed": 0,
                "errors": [],
                "created_at": datetime.utcnow().isoformat(),
                "started_at": None,
                "completed_at": None,
            }
        return job_id

    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
                if "processed" in kwargs and "total" in kwargs:
                    total = kwargs.get("total", 1)
                    if total > 0:
                        self._jobs[job_id]["progress"] = int((kwargs["processed"] / total) * 100)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by user."""
        with self._lock:
            jobs = list(self._jobs.values())
            if user_id:
                jobs = [j for j in jobs if j["user_id"] == user_id]
            return sorted(jobs, key=lambda x: x["created_at"], reverse=True)[:20]


# Global job tracker
_job_tracker = BackgroundJobStatus()


async def _run_embedding_backfill(job_id: str, batch_size: int = 50):
    """Background task to generate missing embeddings."""
    from backend.db.database import async_session_maker
    from backend.services.embeddings import get_embedding_service

    _job_tracker.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        async with async_session_maker() as db:
            # Count chunks without embeddings (use has_embedding flag for consistency)
            count_result = await db.execute(
                select(func.count(Chunk.id)).where(
                    Chunk.has_embedding == False
                )
            )
            total_missing = count_result.scalar() or 0

            _job_tracker.update_job(job_id, total=total_missing)

            if total_missing == 0:
                _job_tracker.update_job(
                    job_id,
                    status="completed",
                    completed_at=datetime.utcnow().isoformat(),
                    message="No missing embeddings found",
                )
                return

            embedding_service = get_embedding_service()
            processed = 0
            errors = []

            while processed < total_missing:
                # Fetch batch of chunks without embeddings
                result = await db.execute(
                    select(Chunk).where(
                        Chunk.has_embedding == False
                    ).limit(batch_size)
                )
                chunks = result.scalars().all()

                if not chunks:
                    break

                for chunk in chunks:
                    try:
                        # Generate embedding
                        embedding = await embedding_service.embed_text(chunk.content)
                        chunk.embedding = embedding
                        chunk.has_embedding = True  # Track for UI consistency
                        processed += 1

                    except Exception as e:
                        errors.append(f"Chunk {chunk.id}: {str(e)}")
                        logger.warning("Failed to embed chunk", chunk_id=str(chunk.id), error=str(e))

                    # Update progress every 10 items
                    if processed % 10 == 0:
                        _job_tracker.update_job(job_id, processed=processed, errors=errors[-5:])

                await db.commit()

            _job_tracker.update_job(
                job_id,
                status="completed",
                completed_at=datetime.utcnow().isoformat(),
                processed=processed,
                errors=errors[-10:],
            )

            logger.info("Embedding backfill completed", job_id=job_id, processed=processed, errors=len(errors))

    except Exception as e:
        logger.error("Embedding backfill failed", job_id=job_id, error=str(e))
        _job_tracker.update_job(
            job_id,
            status="failed",
            completed_at=datetime.utcnow().isoformat(),
            error=str(e),
        )


async def _run_document_embedding_generation(job_id: str, document_id: str):
    """Background task to generate embeddings for a specific document."""
    from backend.db.database import async_session_maker
    from backend.services.embeddings import get_embedding_service

    _job_tracker.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        async with async_session_maker() as db:
            # Count chunks without embeddings for this document
            doc_uuid = uuid_module.UUID(document_id)
            count_result = await db.execute(
                select(func.count(Chunk.id)).where(
                    and_(
                        Chunk.document_id == doc_uuid,
                        Chunk.has_embedding == False,
                    )
                )
            )
            total_missing = count_result.scalar() or 0

            _job_tracker.update_job(job_id, total=total_missing)

            if total_missing == 0:
                _job_tracker.update_job(
                    job_id,
                    status="completed",
                    completed_at=datetime.utcnow().isoformat(),
                    message="No missing embeddings found for this document",
                )
                return

            embedding_service = get_embedding_service()
            processed = 0
            errors = []

            # Fetch all chunks for this document without embeddings
            result = await db.execute(
                select(Chunk).where(
                    and_(
                        Chunk.document_id == doc_uuid,
                        Chunk.has_embedding == False,
                    )
                )
            )
            chunks = result.scalars().all()

            for chunk in chunks:
                try:
                    # Generate embedding
                    embedding = await embedding_service.embed_text(chunk.content)
                    chunk.embedding = embedding
                    chunk.has_embedding = True  # Track for UI consistency
                    processed += 1

                except Exception as e:
                    errors.append(f"Chunk {chunk.id}: {str(e)}")
                    logger.warning("Failed to embed chunk", chunk_id=str(chunk.id), error=str(e))

                # Update progress every 5 items
                if processed % 5 == 0:
                    _job_tracker.update_job(job_id, processed=processed, errors=errors[-5:])

            await db.commit()

            _job_tracker.update_job(
                job_id,
                status="completed",
                completed_at=datetime.utcnow().isoformat(),
                processed=processed,
                errors=errors[-10:],
            )

            logger.info(
                "Document embedding generation completed",
                job_id=job_id,
                document_id=document_id,
                processed=processed,
                errors=len(errors),
            )

    except Exception as e:
        logger.error("Document embedding generation failed", job_id=job_id, document_id=document_id, error=str(e))
        _job_tracker.update_job(
            job_id,
            status="failed",
            completed_at=datetime.utcnow().isoformat(),
            error=str(e),
        )


class ProviderStats(BaseModel):
    name: str
    model: str
    dimension: int
    chunk_count: int
    storage_bytes: int
    is_primary: bool


class EmbeddingStats(BaseModel):
    total_documents: int
    total_chunks: int
    chunks_with_embeddings: int
    coverage_percentage: float
    storage_bytes: int
    provider_count: int
    providers: List[ProviderStats]


@router.get("/embeddings/stats", response_model=EmbeddingStats)
async def get_embedding_stats(
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get overall embedding system statistics.

    Returns:
        - Total documents and chunks
        - Embedding coverage percentage
        - Storage usage
        - Per-provider breakdown
    """
    try:
        # Count total documents
        result = await db.execute(select(func.count(Document.id)))
        total_documents = result.scalar_one() or 0

        # Count total chunks
        result = await db.execute(select(func.count(Chunk.id)))
        total_chunks = result.scalar_one() or 0

        # Count chunks with embeddings (use has_embedding flag for consistency)
        result = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.has_embedding == True)
        )
        chunks_with_embeddings = result.scalar_one() or 0

        # Calculate coverage percentage
        coverage_percentage = (
            (chunks_with_embeddings / total_chunks * 100)
            if total_chunks > 0
            else 0
        )

        providers = []
        total_storage = 0

        # Try to get multi-embedding stats if available
        try:
            from backend.db.models_multi_embedding import ChunkEmbedding

            # Get provider breakdown from chunk_embeddings table
            result = await db.execute(
                select(
                    ChunkEmbedding.provider,
                    ChunkEmbedding.model,
                    ChunkEmbedding.dimension,
                    func.count(ChunkEmbedding.id).label('count'),
                    func.sum(ChunkEmbedding.is_primary.cast(Integer)).label('primary_count')
                )
                .group_by(ChunkEmbedding.provider, ChunkEmbedding.model, ChunkEmbedding.dimension)
            )

            for row in result:
                storage_bytes = row.count * row.dimension * 4  # 4 bytes per float
                total_storage += storage_bytes

                providers.append(ProviderStats(
                    name=row.provider,
                    model=row.model,
                    dimension=row.dimension,
                    chunk_count=row.count,
                    storage_bytes=storage_bytes,
                    is_primary=row.primary_count > 0
                ))

        except ImportError:
            # Multi-embedding not enabled, use primary embedding only
            if chunks_with_embeddings > 0:
                # Get current provider from env
                provider = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")

                if provider == "ollama":
                    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
                else:
                    model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

                dimension = get_embedding_dimension()
                storage_bytes = chunks_with_embeddings * dimension * 4
                total_storage = storage_bytes

                providers.append(ProviderStats(
                    name=provider,
                    model=model,
                    dimension=dimension,
                    chunk_count=chunks_with_embeddings,
                    storage_bytes=storage_bytes,
                    is_primary=True
                ))

        return EmbeddingStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            chunks_with_embeddings=chunks_with_embeddings,
            coverage_percentage=round(coverage_percentage, 1),
            storage_bytes=total_storage,
            provider_count=len(providers),
            providers=providers
        )

    except Exception as e:
        logger.error("Failed to get embedding stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve embedding statistics: {str(e)}"
        )


class GenerateEmbeddingsRequest(BaseModel):
    """Request to generate missing embeddings."""
    batch_size: int = 50


class JobStatusResponse(BaseModel):
    """Response with job status."""
    job_id: str
    status: str
    progress: int
    total: int
    processed: int
    errors: List[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


@router.post("/embeddings/generate-missing", response_model=JobStatusResponse)
async def generate_missing_embeddings(
    request: GenerateEmbeddingsRequest = None,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Trigger background job to generate missing embeddings.

    This starts an async job that processes chunks without embeddings.
    Use the /embeddings/jobs/{job_id} endpoint to check progress.

    Returns job ID for status tracking.
    """
    batch_size = request.batch_size if request else 50

    # Create job
    user_id = current_user.get("sub", current_user.get("id", "unknown"))
    job_id = _job_tracker.create_job("embedding_backfill", user_id)

    # Start background task
    asyncio.create_task(_run_embedding_backfill(job_id, batch_size))

    logger.info(
        "Started embedding backfill job",
        job_id=job_id,
        user_id=user_id,
        batch_size=batch_size,
    )

    job = _job_tracker.get_job(job_id)
    return JobStatusResponse(
        job_id=job["id"],
        status=job["status"],
        progress=job["progress"],
        total=job["total"],
        processed=job["processed"],
        errors=job["errors"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@router.post("/embeddings/generate/{document_id}", response_model=JobStatusResponse)
async def generate_document_embeddings(
    document_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Generate embeddings for a specific document.

    This starts a background job to generate embeddings for all chunks
    of the specified document that don't already have embeddings.

    Returns job ID for status tracking.
    """
    # Verify document exists
    try:
        doc_uuid = uuid_module.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format",
        )

    result = await db.execute(
        select(Document.id).where(Document.id == doc_uuid)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Create job
    user_id = current_user.get("sub", current_user.get("id", "unknown"))
    job_id = _job_tracker.create_job("document_embedding", user_id)

    # Start background task
    asyncio.create_task(_run_document_embedding_generation(job_id, document_id))

    logger.info(
        "Started document embedding job",
        job_id=job_id,
        document_id=document_id,
        user_id=user_id,
    )

    job = _job_tracker.get_job(job_id)
    return JobStatusResponse(
        job_id=job["id"],
        status=job["status"],
        progress=job["progress"],
        total=job["total"],
        processed=job["processed"],
        errors=job["errors"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@router.get("/embeddings/jobs/{job_id}", response_model=JobStatusResponse)
async def get_embedding_job_status(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Get status of an embedding generation job."""
    job = _job_tracker.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return JobStatusResponse(
        job_id=job["id"],
        status=job["status"],
        progress=job["progress"],
        total=job["total"],
        processed=job["processed"],
        errors=job["errors"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@router.get("/embeddings/jobs", response_model=List[JobStatusResponse])
async def list_embedding_jobs(
    current_user: CurrentUser = Depends(get_current_user)
):
    """List recent embedding generation jobs."""
    user_id = current_user.get("sub", current_user.get("id", "unknown"))
    jobs = _job_tracker.list_jobs(user_id)

    return [
        JobStatusResponse(
            job_id=job["id"],
            status=job["status"],
            progress=job["progress"],
            total=job["total"],
            processed=job["processed"],
            errors=job["errors"],
            created_at=job["created_at"],
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
        )
        for job in jobs
    ]


class ProviderInfo(BaseModel):
    """Information about an embedding provider."""
    name: str
    model: str
    dimensions: int
    description: str
    requires_api_key: bool
    requires_gpu: bool
    max_tokens: int
    batch_size: int


class EmbedTextRequest(BaseModel):
    """Request to embed text."""
    text: str
    provider: str = "openai"


class EmbedTextResponse(BaseModel):
    """Response from text embedding."""
    embedding: List[float]
    dimensions: int
    provider: str
    model: str


@router.get("/embeddings/providers", response_model=List[ProviderInfo])
async def list_embedding_providers(
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    List all available embedding providers with their configurations.

    Returns information about each provider including:
    - Model name and dimensions
    - Whether API key or GPU is required
    - Maximum token length and batch size
    """
    from backend.services.embeddings import (
        EmbeddingService,
        PROVIDER_BATCH_CONFIG,
        VOYAGEAI_AVAILABLE,
        QWEN3_AVAILABLE,
    )

    providers = [
        ProviderInfo(
            name="openai",
            model=EmbeddingService.DEFAULT_MODELS.get("openai", "text-embedding-3-small"),
            dimensions=1536,
            description="OpenAI embeddings - reliable, good quality",
            requires_api_key=True,
            requires_gpu=False,
            max_tokens=8191,
            batch_size=PROVIDER_BATCH_CONFIG.get("openai", {}).get("optimal_batch_size", 500),
        ),
        ProviderInfo(
            name="ollama",
            model=EmbeddingService.DEFAULT_MODELS.get("ollama", "nomic-embed-text"),
            dimensions=768,
            description="Local embeddings via Ollama - free, private",
            requires_api_key=False,
            requires_gpu=False,
            max_tokens=8192,
            batch_size=PROVIDER_BATCH_CONFIG.get("ollama", {}).get("optimal_batch_size", 50),
        ),
    ]

    # Add Voyage AI if available
    if VOYAGEAI_AVAILABLE:
        providers.extend([
            ProviderInfo(
                name="voyage",
                model=EmbeddingService.DEFAULT_MODELS.get("voyage", "voyage-3-large"),
                dimensions=1024,
                description="Voyage AI - top MTEB scores, RAG-optimized",
                requires_api_key=True,
                requires_gpu=False,
                max_tokens=16000,
                batch_size=PROVIDER_BATCH_CONFIG.get("voyage", {}).get("optimal_batch_size", 100),
            ),
            ProviderInfo(
                name="voyage4",
                model=EmbeddingService.DEFAULT_MODELS.get("voyage4", "voyage-4"),
                dimensions=1024,
                description="Voyage 4 - shared embedding space, latest gen",
                requires_api_key=True,
                requires_gpu=False,
                max_tokens=16000,
                batch_size=PROVIDER_BATCH_CONFIG.get("voyage4", {}).get("optimal_batch_size", 100),
            ),
        ])

    # Add Qwen3 if available
    if QWEN3_AVAILABLE:
        providers.extend([
            ProviderInfo(
                name="qwen3",
                model=EmbeddingService.DEFAULT_MODELS.get("qwen3", "Alibaba-NLP/Qwen3-Embedding-8B"),
                dimensions=4096,
                description="Qwen3-8B - highest MTEB score (70.58), 100+ languages",
                requires_api_key=False,
                requires_gpu=True,
                max_tokens=8192,
                batch_size=PROVIDER_BATCH_CONFIG.get("qwen3", {}).get("optimal_batch_size", 32),
            ),
            ProviderInfo(
                name="qwen3-4b",
                model=EmbeddingService.DEFAULT_MODELS.get("qwen3-4b", "Alibaba-NLP/Qwen3-Embedding-4B"),
                dimensions=2048,
                description="Qwen3-4B - balanced performance and efficiency",
                requires_api_key=False,
                requires_gpu=True,
                max_tokens=8192,
                batch_size=PROVIDER_BATCH_CONFIG.get("qwen3-4b", {}).get("optimal_batch_size", 64),
            ),
            ProviderInfo(
                name="qwen3-small",
                model=EmbeddingService.DEFAULT_MODELS.get("qwen3-small", "Alibaba-NLP/Qwen3-Embedding-0.6B"),
                dimensions=1024,
                description="Qwen3-0.6B - fast and lightweight",
                requires_api_key=False,
                requires_gpu=False,  # Can run on CPU
                max_tokens=8192,
                batch_size=PROVIDER_BATCH_CONFIG.get("qwen3-small", {}).get("optimal_batch_size", 128),
            ),
        ])

    return providers


@router.post("/embeddings/test", response_model=EmbedTextResponse)
async def test_embedding(
    request: EmbedTextRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Test embedding generation with a specific provider.

    Useful for verifying provider configuration and comparing output dimensions.
    """
    try:
        from backend.services.embeddings import EmbeddingService

        service = EmbeddingService(provider=request.provider)
        embedding = service.embed_text(request.text)

        return EmbedTextResponse(
            embedding=embedding,
            dimensions=len(embedding),
            provider=request.provider,
            model=service.model,
        )

    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{request.provider}' dependencies not installed: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider configuration error: {str(e)}"
        )
    except Exception as e:
        logger.error("Embedding test failed", provider=request.provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )
