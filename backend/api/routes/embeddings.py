"""
Embedding System API Routes
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
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
            status_code=500,
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
        status_code=501,
        detail="Background embedding generation not yet implemented. Use CLI scripts instead."
    )
