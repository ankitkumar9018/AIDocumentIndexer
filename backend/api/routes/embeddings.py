"""
Embedding System API Routes
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select, func, Integer, and_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
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
# Background Job Tracking (Redis-backed for cross-process sharing)
# =============================================================================

class RedisJobTracker:
    """Track status of background embedding jobs using Redis.

    Uses Redis hashes so both the backend API process and Celery workers
    can read/write job state. Falls back to in-memory storage if Redis
    is unavailable.
    """

    REDIS_PREFIX = "reindex:job:"
    CANCEL_PREFIX = "reindex:cancel:"
    USER_JOBS_PREFIX = "reindex:user:"
    JOB_TTL = 86400  # 24 hours

    def __init__(self):
        self._lock = threading.Lock()
        self._fallback_jobs: Dict[str, Dict[str, Any]] = {}
        self._redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    def _get_sync_redis(self):
        """Get a sync Redis client for use in both async and sync contexts."""
        try:
            import redis
            return redis.Redis.from_url(self._redis_url, decode_responses=True)
        except Exception:
            return None

    def create_job(self, job_type: str, user_id: str) -> str:
        """Create a new job and return its ID."""
        import json
        job_id = str(uuid_module.uuid4())
        job_data = {
            "id": job_id,
            "type": job_type,
            "user_id": user_id,
            "status": "pending",
            "progress": "0",
            "total": "0",
            "processed": "0",
            "errors": "[]",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": "",
            "completed_at": "",
            "message": "",
            "current_document": "",
            "processing_mode": "linear",
        }

        r = self._get_sync_redis()
        if r:
            try:
                r.hset(f"{self.REDIS_PREFIX}{job_id}", mapping=job_data)
                r.expire(f"{self.REDIS_PREFIX}{job_id}", self.JOB_TTL)
                # Track user's jobs
                r.lpush(f"{self.USER_JOBS_PREFIX}{user_id}", job_id)
                r.ltrim(f"{self.USER_JOBS_PREFIX}{user_id}", 0, 19)
                r.expire(f"{self.USER_JOBS_PREFIX}{user_id}", self.JOB_TTL)
                return job_id
            except Exception as e:
                logger.warning("Redis create_job failed, using fallback", error=str(e))

        # Fallback to in-memory
        with self._lock:
            self._fallback_jobs[job_id] = {**job_data, "progress": 0, "total": 0, "processed": 0, "errors": []}
        return job_id

    def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        import json
        r = self._get_sync_redis()
        if r:
            try:
                # Convert non-string values for Redis hash
                redis_data = {}
                for k, v in kwargs.items():
                    if isinstance(v, list):
                        redis_data[k] = json.dumps(v)
                    elif v is None:
                        redis_data[k] = ""
                    else:
                        redis_data[k] = str(v)

                # Calculate progress
                if "processed" in kwargs:
                    total = int(r.hget(f"{self.REDIS_PREFIX}{job_id}", "total") or "0")
                    if "total" in kwargs:
                        total = int(kwargs["total"])
                    processed = int(kwargs["processed"])
                    if total > 0:
                        redis_data["progress"] = str(int((processed / total) * 100))

                r.hset(f"{self.REDIS_PREFIX}{job_id}", mapping=redis_data)
                return
            except Exception as e:
                logger.warning("Redis update_job failed, using fallback", error=str(e))

        # Fallback
        with self._lock:
            if job_id in self._fallback_jobs:
                self._fallback_jobs[job_id].update(kwargs)
                if "processed" in kwargs:
                    total = self._fallback_jobs[job_id].get("total", 0)
                    if isinstance(total, str):
                        total = int(total)
                    processed = int(kwargs["processed"])
                    if total > 0:
                        self._fallback_jobs[job_id]["progress"] = int((processed / total) * 100)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        import json
        r = self._get_sync_redis()
        if r:
            try:
                data = r.hgetall(f"{self.REDIS_PREFIX}{job_id}")
                if data:
                    # Convert back from Redis strings
                    data["progress"] = int(data.get("progress", "0"))
                    data["total"] = int(data.get("total", "0"))
                    data["processed"] = int(data.get("processed", "0"))
                    try:
                        data["errors"] = json.loads(data.get("errors", "[]"))
                    except (json.JSONDecodeError, TypeError):
                        data["errors"] = []
                    # Convert empty strings back to None
                    for k in ("started_at", "completed_at", "current_document"):
                        if data.get(k) == "":
                            data[k] = None
                    return data
            except Exception as e:
                logger.warning("Redis get_job failed, using fallback", error=str(e))

        with self._lock:
            return self._fallback_jobs.get(job_id)

    def list_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by user."""
        import json
        r = self._get_sync_redis()
        if r and user_id:
            try:
                job_ids = r.lrange(f"{self.USER_JOBS_PREFIX}{user_id}", 0, 19)
                jobs = []
                for jid in job_ids:
                    job = self.get_job(jid)
                    if job:
                        jobs.append(job)
                return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)
            except Exception as e:
                logger.warning("Redis list_jobs failed, using fallback", error=str(e))

        with self._lock:
            jobs = list(self._fallback_jobs.values())
            if user_id:
                jobs = [j for j in jobs if j.get("user_id") == user_id]
            return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)[:20]

    def request_cancel(self, job_id: str):
        """Signal cancellation for a job (cross-process via Redis)."""
        r = self._get_sync_redis()
        if r:
            try:
                r.set(f"{self.CANCEL_PREFIX}{job_id}", "1", ex=3600)
                return
            except Exception:
                pass
        _cancel_requests_fallback.add(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        """Check if cancellation was requested (cross-process via Redis)."""
        r = self._get_sync_redis()
        if r:
            try:
                return bool(r.get(f"{self.CANCEL_PREFIX}{job_id}"))
            except Exception:
                pass
        return job_id in _cancel_requests_fallback

    def clear_cancel(self, job_id: str):
        """Clear cancellation flag after handling it."""
        r = self._get_sync_redis()
        if r:
            try:
                r.delete(f"{self.CANCEL_PREFIX}{job_id}")
                return
            except Exception:
                pass
        _cancel_requests_fallback.discard(job_id)


# Global job tracker (Redis-backed, shared with Celery workers)
_job_tracker = RedisJobTracker()

# Fallback cancel requests set (only used if Redis unavailable)
_cancel_requests_fallback: set = set()


async def _run_embedding_backfill(job_id: str, batch_size: int = 50):
    """Background task to generate missing embeddings."""
    from backend.db.database import get_async_session_factory
    from backend.services.embeddings import get_embedding_service

    _job_tracker.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        async_session_factory = get_async_session_factory()
        async with async_session_factory() as db:
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
                        embedding = embedding_service.embed_text(chunk.content)
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
    from backend.db.database import get_async_session_factory
    from backend.services.embeddings import get_embedding_service

    _job_tracker.update_job(job_id, status="running", started_at=datetime.utcnow().isoformat())

    try:
        async_session_factory = get_async_session_factory()
        async with async_session_factory() as db:
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
                    embedding = embedding_service.embed_text(chunk.content)
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
            detail="Failed to retrieve embedding statistics"
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
    provider: Optional[str] = Field(None, description="Embedding provider (auto-detected from settings if not specified)")


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
        from backend.services.embeddings import EmbeddingService, get_embedding_service

        # Resolve provider from settings if not specified
        if request.provider:
            service = EmbeddingService(provider=request.provider)
        else:
            service = get_embedding_service(provider=None, use_ray=False)

        embedding = service.embed_text(request.text)

        return EmbedTextResponse(
            embedding=embedding,
            dimensions=len(embedding),
            provider=service.provider,
            model=service.model,
        )

    except ImportError as e:
        logger.warning("Provider dependencies missing", provider=request.provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{request.provider}' dependencies not installed"
        )
    except ValueError as e:
        logger.warning("Provider configuration error", provider=request.provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider configuration error"
        )
    except Exception as e:
        logger.error("Embedding test failed", provider=request.provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed"
        )


# =============================================================================
# Batch Re-index Endpoint (Memory-Safe)
# =============================================================================

class ReindexRequest(BaseModel):
    """Request to reindex all documents with new embeddings."""
    processing_mode: str = "linear"  # "linear" (1 at a time) or "parallel" (concurrent)
    parallel_count: int = 2  # Number of documents to process concurrently (if parallel mode)
    batch_size: int = 5  # Documents per batch before delay/GC pause
    delay_seconds: int = 15  # Delay between batches for memory cleanup
    force_reembed: bool = True  # Force re-embedding even if embeddings exist


class ReindexJobResponse(BaseModel):
    """Response for reindex job status."""
    job_id: str
    status: str
    message: str
    total_documents: int = 0
    processed: int = 0
    progress: int = 0
    processing_mode: str = "linear"
    current_document: Optional[str] = None
    errors: List[str] = []
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# Cancellation now handled via _job_tracker.request_cancel() / is_cancel_requested()


@router.post("/embeddings/reindex-all", response_model=ReindexJobResponse)
async def reindex_all_documents(
    request: ReindexRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Start a batch re-indexing job to regenerate all document embeddings.

    This is memory-safe: processes documents in small batches with delays
    between batches to allow garbage collection.

    Use this after changing embedding models or fixing embedding issues.
    """
    # Check for existing running job
    user_id = current_user.get("sub", current_user.get("id", "unknown"))
    existing_jobs = _job_tracker.list_jobs(user_id)
    for job in existing_jobs:
        if job["type"] == "reindex_all" and job["status"] in ("pending", "running"):
            return ReindexJobResponse(
                job_id=job["id"],
                status="already_running",
                message="A reindex job is already running. Wait for it to complete or cancel it.",
                total_documents=job.get("total", 0),
                processed=job.get("processed", 0),
                progress=job.get("progress", 0),
            )

    # Count total documents
    result = await db.execute(select(func.count(Document.id)))
    total_documents = result.scalar_one() or 0

    if total_documents == 0:
        return ReindexJobResponse(
            job_id="",
            status="completed",
            message="No documents to reindex.",
            total_documents=0,
        )

    # Create job
    job_id = _job_tracker.create_job("reindex_all", user_id)
    _job_tracker.update_job(
        job_id,
        total=total_documents,
        processing_mode=request.processing_mode,
    )

    # Dispatch to Celery worker so the main backend process stays responsive
    try:
        from backend.tasks.document_tasks import run_reindex_all_job
        run_reindex_all_job.delay(
            job_id=job_id,
            processing_mode=request.processing_mode,
            parallel_count=request.parallel_count,
            batch_size=request.batch_size,
            delay_seconds=request.delay_seconds,
            force_reembed=request.force_reembed,
        )
    except Exception as e:
        # If Celery is unavailable, fall back to in-process execution
        logger.warning("Celery unavailable, running reindex in-process", error=str(e))

        async def run_with_error_handling():
            try:
                await _run_batch_reindex(
                    job_id=job_id,
                    processing_mode=request.processing_mode,
                    parallel_count=request.parallel_count,
                    batch_size=request.batch_size,
                    delay_seconds=request.delay_seconds,
                    force_reembed=request.force_reembed,
                )
            except Exception as exc:
                logger.error("Background reindex task failed", job_id=job_id, error=str(exc), exc_info=True)
                _job_tracker.update_job(job_id, status="failed", error=str(exc))

        asyncio.create_task(run_with_error_handling())

    mode_desc = "sequentially" if request.processing_mode == "linear" else f"with {request.parallel_count} parallel workers"
    return ReindexJobResponse(
        job_id=job_id,
        status="started",
        message=f"Reindex job started. Processing {total_documents} documents {mode_desc}, batch size {request.batch_size}, {request.delay_seconds}s delay between batches.",
        total_documents=total_documents,
        processing_mode=request.processing_mode,
        created_at=datetime.utcnow().isoformat(),
    )


@router.post("/embeddings/reindex-all/{job_id}/cancel")
async def cancel_reindex_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Cancel a running reindex job."""
    job = _job_tracker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job['status']}")

    _job_tracker.request_cancel(job_id)
    _job_tracker.update_job(job_id, status="cancelling")

    return {"message": "Cancellation requested", "job_id": job_id}


@router.get("/embeddings/reindex-all/current", response_model=Optional[ReindexJobResponse])
async def get_current_reindex_job(
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get the currently running reindex job."""
    user_id = current_user.get("sub", current_user.get("id", "unknown"))
    jobs = _job_tracker.list_jobs(user_id)
    for job in jobs:
        if job["type"] == "reindex_all" and job["status"] in ("started", "pending", "running", "cancelling"):
            return ReindexJobResponse(
                job_id=job["id"],
                status=job["status"],
                message=job.get("message", ""),
                total_documents=job.get("total", 0),
                processed=job.get("processed", 0),
                progress=job.get("progress", 0),
                processing_mode=job.get("processing_mode", "linear"),
                current_document=job.get("current_document"),
                errors=job.get("errors", []),
                created_at=job.get("created_at"),
                started_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
            )
    return None


async def _process_single_document(
    doc_id,
    doc_name: str,
    force_reembed: bool,
    job_id: str,
    processed_counter: dict,
    errors: list,
    total: int,
):
    """Process a single document for reindexing."""
    try:
        from backend.db.database import get_async_session_factory
        from backend.db.models import Chunk, Document as DocumentModel
        from backend.db.models import ProcessingStatus as DBProcessingStatus
        from backend.services.embeddings import get_embedding_service
        from backend.services.vectorstore_local import get_chroma_vector_store
        from sqlalchemy import select
        from datetime import datetime

        async_session_factory = get_async_session_factory()
        async with async_session_factory() as db:
            # Get the document for metadata
            doc_result = await db.execute(
                select(DocumentModel).where(DocumentModel.id == doc_id)
            )
            doc = doc_result.scalar_one_or_none()

            # Get all chunks for this document
            result = await db.execute(
                select(Chunk).where(Chunk.document_id == doc_id)
            )
            chunks = result.scalars().all()

            if not chunks:
                logger.warning("No chunks found for document", doc_id=str(doc_id))
            else:
                # Get services - use_ray=False for single text embeddings
                embedding_service = get_embedding_service(use_ray=False)
                vector_store = get_chroma_vector_store()

                # Build document-level metadata for ChromaDB upserts
                doc_filename = (doc.original_filename or doc.filename or "unknown") if doc else "unknown"
                doc_access_tier_id = str(doc.access_tier_id) if doc and doc.access_tier_id else ""
                doc_tags = doc.tags if doc and doc.tags else []
                doc_org_id = str(doc.organization_id) if doc and hasattr(doc, 'organization_id') and doc.organization_id else ""
                doc_uploaded_by = str(doc.uploaded_by_id) if doc and doc.uploaded_by_id else ""
                doc_is_private = bool(doc.is_private) if doc and hasattr(doc, 'is_private') else False

                for chunk in chunks:
                    if chunk.content:
                        # Generate new embedding using embed_text (sync method)
                        new_embedding = embedding_service.embed_text(chunk.content)

                        # Update SQL database
                        chunk.embedding = new_embedding
                        chunk.has_embedding = True
                        # Update chunk_metadata with reindex timestamp
                        current_metadata = dict(chunk.chunk_metadata) if chunk.chunk_metadata else {}
                        current_metadata["embedding_reindexed_at"] = datetime.utcnow().isoformat()
                        chunk.chunk_metadata = current_metadata

                        # Build ChromaDB metadata for this chunk
                        chroma_metadata = {
                            "document_id": str(doc_id),
                            "access_tier_id": doc_access_tier_id,
                            "document_filename": doc_filename,
                            "collection": doc_tags[0] if doc_tags else "",
                            "chunk_index": chunk.chunk_index or 0,
                            "page_number": chunk.page_number or 0,
                            "section_title": chunk.section_title or "",
                            "token_count": chunk.token_count or 0,
                            "char_count": chunk.char_count or len(chunk.content),
                            "content_hash": chunk.content_hash or "",
                            "organization_id": doc_org_id,
                            "uploaded_by_id": doc_uploaded_by,
                            "is_private": doc_is_private,
                        }

                        # Update ChromaDB vector store with full metadata
                        try:
                            await vector_store.update_chunk_embedding(
                                chunk_id=str(chunk.id),
                                embedding=new_embedding,
                                document_content=chunk.content,
                                metadata=chroma_metadata,
                            )
                        except Exception as ve:
                            logger.warning("ChromaDB update failed", chunk_id=str(chunk.id), error=str(ve))

                # Update processing_status to COMPLETED if document had chunks
                # and was stuck at PENDING (fixes status for successfully embedded docs)
                if doc and doc.processing_status != DBProcessingStatus.COMPLETED:
                    doc.processing_status = DBProcessingStatus.COMPLETED
                    doc.processed_at = datetime.utcnow()
                    logger.info(
                        "Updated document status to COMPLETED after reindex",
                        doc_id=str(doc_id),
                        previous_status=str(doc.processing_status),
                    )

                await db.commit()
                logger.info(
                    "Document reindexed successfully",
                    doc_id=str(doc_id),
                    chunks_updated=len(chunks),
                )

        processed_counter["count"] += 1
        _job_tracker.update_job(
            job_id,
            processed=processed_counter["count"],
            total=total,
            current_document=doc_name or str(doc_id),
        )

    except Exception as e:
        error_msg = f"{doc_name or doc_id}: {str(e)}"
        errors.append(error_msg)
        logger.error("Reindex failed for document", doc_id=str(doc_id), error=str(e), exc_info=True)
        processed_counter["count"] += 1
        _job_tracker.update_job(
            job_id,
            processed=processed_counter["count"],
            total=total,
            errors=errors[-10:],
        )


async def _run_batch_reindex(
    job_id: str,
    processing_mode: str,
    parallel_count: int,
    batch_size: int,
    delay_seconds: int,
    force_reembed: bool,
):
    """Background task to reindex documents in memory-safe batches.

    Supports two modes:
    - linear: Process one document at a time (safest, slowest)
    - parallel: Process multiple documents concurrently (faster, more memory)
    """
    import gc
    from backend.db.database import get_async_session_factory

    logger.info(
        "Starting batch reindex task",
        job_id=job_id,
        processing_mode=processing_mode,
        parallel_count=parallel_count,
        batch_size=batch_size,
    )

    _job_tracker.update_job(
        job_id,
        status="running",
        started_at=datetime.utcnow().isoformat(),
        processing_mode=processing_mode,
    )

    try:
        async_session_factory = get_async_session_factory()
        async with async_session_factory() as db:
            # Get all document IDs
            result = await db.execute(
                select(Document.id, Document.filename).order_by(Document.created_at)
            )
            documents = result.all()

            total = len(documents)
            processed_counter = {"count": 0}  # Use dict for mutable reference
            errors = []

            logger.info(
                "Found documents to reindex",
                job_id=job_id,
                total_documents=total,
            )

            _job_tracker.update_job(job_id, total=total)

            # Process in batches
            for i in range(0, total, batch_size):
                # Check for cancellation (via Redis - works cross-process)
                if _job_tracker.is_cancel_requested(job_id):
                    _job_tracker.clear_cancel(job_id)
                    _job_tracker.update_job(
                        job_id,
                        status="cancelled",
                        completed_at=datetime.utcnow().isoformat(),
                        message=f"Cancelled after processing {processed_counter['count']}/{total} documents",
                    )
                    return

                batch = documents[i:i + batch_size]

                if processing_mode == "parallel" and parallel_count > 1:
                    # Parallel processing: process multiple docs concurrently
                    _job_tracker.update_job(
                        job_id,
                        message=f"Processing batch {i//batch_size + 1} ({len(batch)} docs, {parallel_count} parallel)",
                    )

                    # Process in sub-batches of parallel_count
                    for j in range(0, len(batch), parallel_count):
                        if _job_tracker.is_cancel_requested(job_id):
                            break

                        sub_batch = batch[j:j + parallel_count]
                        tasks = [
                            _process_single_document(
                                doc_id, doc_name, force_reembed,
                                job_id, processed_counter, errors, total
                            )
                            for doc_id, doc_name in sub_batch
                        ]
                        await asyncio.gather(*tasks, return_exceptions=True)

                else:
                    # Linear processing: one at a time (safest)
                    for doc_id, doc_name in batch:
                        if _job_tracker.is_cancel_requested(job_id):
                            break

                        _job_tracker.update_job(
                            job_id,
                            current_document=doc_name or str(doc_id),
                            message=f"Processing {doc_name or doc_id}",
                        )

                        await _process_single_document(
                            doc_id, doc_name, force_reembed,
                            job_id, processed_counter, errors, total
                        )

                # Delay between batches for memory cleanup
                if i + batch_size < total:
                    gc.collect()  # Force garbage collection
                    await asyncio.sleep(delay_seconds)

            _job_tracker.update_job(
                job_id,
                status="completed",
                completed_at=datetime.utcnow().isoformat(),
                message=f"Completed: {processed_counter['count']}/{total} documents processed, {len(errors)} errors",
                current_document=None,
            )

    except Exception as e:
        logger.error("Batch reindex job failed", job_id=job_id, error=str(e))
        _job_tracker.update_job(
            job_id,
            status="failed",
            completed_at=datetime.utcnow().isoformat(),
            error=str(e),
        )


# =============================================================================
# Vector Database Stats Endpoint
# =============================================================================

class EmbeddingStatsResponse(BaseModel):
    """Response model for embedding statistics."""
    total_chunks: int
    chunks_with_embedding: int
    chunks_without_embedding: int
    chunks_with_null_embedding: int
    orphaned_chunks: int = 0  # Chunks referencing deleted documents
    embedding_coverage_percent: float
    total_documents: int
    documents_with_issues: int
    chromadb_total_items: Optional[int] = None
    embedding_dimension: int
    problem_documents: List[Dict[str, Any]]


@router.get("/embeddings/health", response_model=EmbeddingStatsResponse)
async def get_embedding_health(
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get comprehensive health statistics about the embedding/vector database.

    Returns information about:
    - Total chunks and embedding coverage
    - Documents with missing or partial embeddings
    - ChromaDB collection status
    """
    from backend.services.vectorstore_local import get_chroma_vector_store
    from sqlalchemy import text

    # Get chunk statistics
    total_chunks_result = await db.execute(select(func.count(Chunk.id)))
    total_chunks = total_chunks_result.scalar() or 0

    with_embedding_result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.has_embedding == True)
    )
    chunks_with_embedding = with_embedding_result.scalar() or 0

    without_embedding_result = await db.execute(
        select(func.count(Chunk.id)).where(
            (Chunk.has_embedding == False) | (Chunk.has_embedding == None)
        )
    )
    chunks_without_embedding = without_embedding_result.scalar() or 0

    # Count chunks missing embeddings using has_embedding flag
    # (the embedding column in SQLite may be NULL by design when embeddings
    # are stored only in ChromaDB, so we use has_embedding as the source of truth)
    null_embedding_result = await db.execute(
        select(func.count(Chunk.id)).where(
            (Chunk.has_embedding == False) | (Chunk.has_embedding == None)
        )
    )
    chunks_with_null_embedding = null_embedding_result.scalar() or 0

    # Get document statistics
    total_docs_result = await db.execute(select(func.count(Document.id)))
    total_documents = total_docs_result.scalar() or 0

    # Get problem documents (those without embeddings based on has_embedding flag)
    problem_docs_query = """
        SELECT d.original_filename, COUNT(c.id) as problem_chunks
        FROM documents d
        JOIN chunks c ON c.document_id = d.id
        WHERE c.has_embedding = 0 OR c.has_embedding IS NULL
        GROUP BY d.id, d.original_filename
        ORDER BY problem_chunks DESC
        LIMIT 10
    """
    problem_result = await db.execute(text(problem_docs_query))
    problem_documents = [
        {"filename": row[0], "problem_chunks": row[1]}
        for row in problem_result.fetchall()
    ]

    # Count orphaned chunks (chunks referencing deleted documents)
    orphaned_query = """
        SELECT COUNT(*) FROM chunks c
        WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id)
    """
    orphaned_result = await db.execute(text(orphaned_query))
    orphaned_chunks = orphaned_result.scalar() or 0

    # ChromaDB stats
    chromadb_total_items = None
    try:
        vector_store = get_chroma_vector_store()
        if vector_store and hasattr(vector_store, '_collection') and vector_store._collection:
            chromadb_total_items = vector_store._collection.count()
    except Exception as e:
        logger.warning("Failed to get ChromaDB stats", error=str(e))

    # Calculate coverage (excluding orphaned chunks for accurate metric)
    valid_chunks = total_chunks - orphaned_chunks
    coverage = (chunks_with_embedding / valid_chunks * 100) if valid_chunks > 0 else 0

    return EmbeddingStatsResponse(
        total_chunks=total_chunks,
        chunks_with_embedding=chunks_with_embedding,
        chunks_without_embedding=chunks_without_embedding,
        chunks_with_null_embedding=chunks_with_null_embedding,
        orphaned_chunks=orphaned_chunks,
        embedding_coverage_percent=round(coverage, 2),
        total_documents=total_documents,
        documents_with_issues=len(problem_documents),
        chromadb_total_items=chromadb_total_items,
        embedding_dimension=get_embedding_dimension(),
        problem_documents=problem_documents,
    )


class CleanupResponse(BaseModel):
    """Response for cleanup operations."""
    deleted_count: int
    message: str


@router.delete("/embeddings/cleanup-orphaned", response_model=CleanupResponse)
async def cleanup_orphaned_chunks(
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Delete orphaned chunks - chunks that reference documents that no longer exist.

    This cleans up data inconsistencies that can occur when documents are deleted
    but their chunks remain in the database.
    """
    from sqlalchemy import text
    from backend.services.vectorstore_local import get_chroma_vector_store

    # First, get the IDs of orphaned chunks for ChromaDB cleanup
    orphaned_ids_query = """
        SELECT c.id FROM chunks c
        WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id)
    """
    result = await db.execute(text(orphaned_ids_query))
    orphaned_ids = [str(row[0]) for row in result.fetchall()]

    if not orphaned_ids:
        return CleanupResponse(deleted_count=0, message="No orphaned chunks found")

    # Delete from ChromaDB first
    try:
        vector_store = get_chroma_vector_store()
        if vector_store and hasattr(vector_store, '_collection') and vector_store._collection:
            # ChromaDB delete in batches to avoid issues
            batch_size = 100
            for i in range(0, len(orphaned_ids), batch_size):
                batch = orphaned_ids[i:i + batch_size]
                try:
                    vector_store._collection.delete(ids=batch)
                except Exception as e:
                    logger.warning("ChromaDB delete batch failed (may not exist)", error=str(e))
    except Exception as e:
        logger.warning("Failed to cleanup ChromaDB orphaned chunks", error=str(e))

    # Delete from SQL database
    delete_query = """
        DELETE FROM chunks
        WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = chunks.document_id)
    """
    result = await db.execute(text(delete_query))
    await db.commit()

    deleted_count = len(orphaned_ids)
    logger.info("Cleaned up orphaned chunks", deleted_count=deleted_count)

    return CleanupResponse(
        deleted_count=deleted_count,
        message=f"Successfully deleted {deleted_count} orphaned chunks"
    )


class SyncChromaResponse(BaseModel):
    """Response model for ChromaDB sync operation."""
    synced_count: int
    skipped_count: int
    error_count: int
    message: str


@router.post("/embeddings/sync-chromadb", response_model=SyncChromaResponse)
async def sync_chromadb(
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Sync missing chunks to ChromaDB using existing embeddings from SQLite.

    Finds chunks that exist in SQLite with embeddings but are missing from
    ChromaDB, and adds them with proper metadata. Does NOT re-generate
    embeddings — uses existing ones from the database.

    Use this after fixing ChromaDB corruption or path issues.
    """
    from backend.services.vectorstore_local import get_chroma_vector_store
    from backend.db.models import Document as DocumentModel

    vector_store = get_chroma_vector_store()
    if not vector_store or not hasattr(vector_store, '_collection') or not vector_store._collection:
        return SyncChromaResponse(
            synced_count=0, skipped_count=0, error_count=0,
            message="ChromaDB vector store not available"
        )

    # Get all chunk IDs currently in ChromaDB
    chroma_count = vector_store._collection.count()
    logger.info("ChromaDB sync starting", current_chroma_count=chroma_count)

    # Get all chunk IDs from ChromaDB in batches
    chroma_ids = set()
    batch_size = 1000
    offset = 0
    while True:
        batch = vector_store._collection.get(
            limit=batch_size,
            offset=offset,
            include=[],
        )
        if not batch["ids"]:
            break
        chroma_ids.update(batch["ids"])
        offset += batch_size

    logger.info("Retrieved ChromaDB IDs", count=len(chroma_ids))

    # Get all chunks with embeddings from SQLite, joined with documents for metadata
    result = await db.execute(
        select(Chunk, DocumentModel)
        .join(DocumentModel, Chunk.document_id == DocumentModel.id)
        .where(Chunk.has_embedding == True)
    )
    all_rows = result.all()

    # Find chunks missing from ChromaDB
    missing_chunks = []
    for chunk, doc in all_rows:
        if str(chunk.id) not in chroma_ids:
            missing_chunks.append((chunk, doc))

    if not missing_chunks:
        return SyncChromaResponse(
            synced_count=0, skipped_count=len(all_rows), error_count=0,
            message=f"ChromaDB is in sync. All {len(all_rows)} chunks present."
        )

    logger.info("Found chunks missing from ChromaDB", missing=len(missing_chunks), total=len(all_rows))

    # Add missing chunks to ChromaDB in batches
    synced = 0
    skipped = 0
    errors = 0
    add_batch_size = 50

    for i in range(0, len(missing_chunks), add_batch_size):
        batch = missing_chunks[i:i + add_batch_size]
        ids = []
        embeddings_list = []
        documents_list = []
        metadatas_list = []

        for chunk, doc in batch:
            # Skip if chunk has no embedding data in SQLite
            embedding = chunk.embedding
            if embedding is None or (hasattr(embedding, '__len__') and len(embedding) == 0):
                skipped += 1
                continue

            # Convert to list if numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            ids.append(str(chunk.id))
            embeddings_list.append(embedding)
            documents_list.append(chunk.content or "")

            doc_tags = doc.tags if doc.tags else []
            metadatas_list.append({
                "document_id": str(chunk.document_id),
                "access_tier_id": str(doc.access_tier_id) if doc.access_tier_id else "",
                "document_filename": doc.original_filename or doc.filename or "unknown",
                "collection": doc_tags[0] if doc_tags else "",
                "chunk_index": chunk.chunk_index or 0,
                "page_number": chunk.page_number or 0,
                "section_title": chunk.section_title or "",
                "token_count": chunk.token_count or 0,
                "char_count": chunk.char_count or (len(chunk.content) if chunk.content else 0),
                "content_hash": chunk.content_hash or "",
                "organization_id": str(doc.organization_id) if hasattr(doc, 'organization_id') and doc.organization_id else "",
                "uploaded_by_id": str(doc.uploaded_by_id) if doc.uploaded_by_id else "",
                "is_private": bool(doc.is_private) if hasattr(doc, 'is_private') else False,
            })

        if ids:
            try:
                vector_store._collection.upsert(
                    ids=ids,
                    embeddings=embeddings_list,
                    documents=documents_list,
                    metadatas=metadatas_list,
                )
                synced += len(ids)
                logger.info(
                    "Synced batch to ChromaDB",
                    batch_num=i // add_batch_size + 1,
                    batch_synced=len(ids),
                    total_synced=synced,
                )
            except Exception as e:
                errors += len(ids)
                logger.error("ChromaDB sync batch failed", error=str(e), batch_size=len(ids))

    final_count = vector_store._collection.count()
    message = f"Synced {synced} missing chunks to ChromaDB. ChromaDB now has {final_count} items."
    logger.info("ChromaDB sync completed", synced=synced, skipped=skipped, errors=errors, final_count=final_count)

    return SyncChromaResponse(
        synced_count=synced,
        skipped_count=skipped,
        error_count=errors,
        message=message,
    )
