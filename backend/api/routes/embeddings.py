"""
Embedding System API Routes
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import os

from backend.db.database import get_async_session
from backend.db.models import Document, Chunk, get_embedding_dimension
from backend.api.middleware.auth import get_current_user, CurrentUser
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()


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

        # Count chunks with embeddings
        result = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
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


@router.post("/embeddings/generate-missing")
async def generate_missing_embeddings(
    db: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Trigger background job to generate missing embeddings.

    This endpoint is a placeholder for future implementation.
    Currently, use the CLI scripts:
    - python backend/scripts/backfill_chunk_embeddings.py
    - python backend/scripts/backfill_entity_embeddings.py
    """
    # TODO: Implement background job system
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Background embedding generation not yet implemented. Use CLI scripts instead."
    )


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
