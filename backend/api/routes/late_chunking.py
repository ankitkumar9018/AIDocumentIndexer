"""
AIDocumentIndexer - Late Chunking API Routes
=============================================

API endpoints for late chunking operations.

Late chunking embeds the full document first, then slices token embeddings
by chunk boundaries. This preserves full document context in each chunk
embedding, improving retrieval accuracy by 15-25%.
"""

from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.services.late_chunking import (
    get_late_chunking_engine,
    LateChunkingConfig,
    LateChunkingModel,
    PoolingStrategy,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/late-chunking", tags=["late-chunking"])


# =============================================================================
# Request/Response Models
# =============================================================================

class PoolingStrategyEnum(str, Enum):
    """Available pooling strategies for token embeddings."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAX = "max"
    CLS = "cls"
    LAST = "last"


class ModelEnum(str, Enum):
    """Available late chunking models."""
    JINA_V3 = "jinaai/jina-embeddings-v3"
    NOMIC_V1_5 = "nomic-ai/nomic-embed-text-v1.5"
    BGE_M3 = "BAAI/bge-m3"
    VOYAGE_3 = "voyage-3-large"


class ChunkResponse(BaseModel):
    """A single chunk with contextual embedding."""
    content: str
    index: int
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessDocumentRequest(BaseModel):
    """Request to process a document with late chunking."""
    text: str = Field(..., description="Document text to process")
    document_id: Optional[str] = Field(None, description="Optional document identifier")
    chunk_size: int = Field(256, ge=64, le=1024, description="Tokens per chunk")
    chunk_overlap: int = Field(32, ge=0, le=128, description="Token overlap between chunks")
    pooling_strategy: Optional[PoolingStrategyEnum] = Field(
        None, description="Token pooling strategy (default: mean)"
    )


class ProcessDocumentResponse(BaseModel):
    """Response from processing a document."""
    chunks: List[ChunkResponse]
    document_id: Optional[str]
    model_used: str
    total_tokens: int
    processing_time_ms: float


class ProcessBatchRequest(BaseModel):
    """Request to process multiple documents."""
    texts: List[str] = Field(
        ...,
        description="Document texts to process",
        min_length=1,
        max_length=100,  # Limit batch size to prevent DoS (Phase 69)
    )
    document_ids: Optional[List[str]] = Field(None, description="Optional document identifiers")
    chunk_size: int = Field(256, ge=64, le=1024)
    chunk_overlap: int = Field(32, ge=0, le=128)


class ProcessBatchResponse(BaseModel):
    """Response from batch processing."""
    results: List[ProcessDocumentResponse]
    total_documents: int
    total_chunks: int
    total_processing_time_ms: float


class CompareRequest(BaseModel):
    """Request to compare late vs traditional chunking."""
    text: str = Field(..., description="Document text")
    queries: List[str] = Field(..., description="Test queries for retrieval comparison")


class CompareResponse(BaseModel):
    """Comparison results."""
    late_chunks: int
    traditional_chunks: int
    avg_improvement_pct: float
    per_query: Dict[str, Dict[str, float]]


class ConfigResponse(BaseModel):
    """Current late chunking configuration."""
    model_name: str
    chunk_size: int
    chunk_overlap: int
    max_document_length: int
    pooling_strategy: str
    normalize_embeddings: bool


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/process", response_model=ProcessDocumentResponse)
async def process_document(request: ProcessDocumentRequest) -> ProcessDocumentResponse:
    """
    Process a document with late chunking.

    Late chunking embeds the full document first, then extracts chunk
    embeddings by slicing the token embeddings. This preserves full
    document context in each chunk.

    Benefits:
    - +15-25% retrieval accuracy
    - Better handling of cross-chunk references
    - Same storage cost as traditional chunking
    """
    logger.info(
        "Processing document with late chunking",
        document_id=request.document_id,
        text_length=len(request.text),
        chunk_size=request.chunk_size,
    )

    if len(request.text) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text too short for chunking (minimum 10 characters)",
        )

    try:
        # Create config with request parameters
        config = LateChunkingConfig(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        if request.pooling_strategy:
            config.pooling_strategy = PoolingStrategy(request.pooling_strategy.value)

        engine = await get_late_chunking_engine(config)
        result = await engine.process_document(request.text, request.document_id)

        chunks_response = [
            ChunkResponse(
                content=c.content,
                index=c.index,
                start_token=c.start_token,
                end_token=c.end_token,
                start_char=c.start_char,
                end_char=c.end_char,
                embedding=c.embedding or [],
                metadata=c.metadata,
            )
            for c in result.chunks
        ]

        return ProcessDocumentResponse(
            chunks=chunks_response,
            document_id=result.document_id,
            model_used=result.model_used,
            total_tokens=result.total_tokens,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.error("Late chunking failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Late chunking failed: {str(e)}",
        )


@router.post("/process-batch", response_model=ProcessBatchResponse)
async def process_batch(request: ProcessBatchRequest) -> ProcessBatchResponse:
    """
    Process multiple documents with late chunking.

    Efficient batch processing for multiple documents.
    """
    logger.info(
        "Processing batch with late chunking",
        num_documents=len(request.texts),
    )

    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents provided",
        )

    try:
        config = LateChunkingConfig(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        engine = await get_late_chunking_engine(config)
        results = await engine.process_batch(request.texts, request.document_ids)

        responses = []
        total_chunks = 0
        total_time = 0.0

        for result in results:
            chunks_response = [
                ChunkResponse(
                    content=c.content,
                    index=c.index,
                    start_token=c.start_token,
                    end_token=c.end_token,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    embedding=c.embedding or [],
                    metadata=c.metadata,
                )
                for c in result.chunks
            ]

            responses.append(ProcessDocumentResponse(
                chunks=chunks_response,
                document_id=result.document_id,
                model_used=result.model_used,
                total_tokens=result.total_tokens,
                processing_time_ms=result.processing_time_ms,
            ))

            total_chunks += len(result.chunks)
            total_time += result.processing_time_ms

        return ProcessBatchResponse(
            results=responses,
            total_documents=len(results),
            total_chunks=total_chunks,
            total_processing_time_ms=total_time,
        )

    except Exception as e:
        logger.error("Batch late chunking failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}",
        )


@router.post("/compare", response_model=CompareResponse)
async def compare_chunking(request: CompareRequest) -> CompareResponse:
    """
    Compare late chunking vs traditional chunking for retrieval.

    Runs the same queries against both chunking methods and reports
    the improvement in retrieval scores.
    """
    logger.info(
        "Comparing chunking methods",
        text_length=len(request.text),
        num_queries=len(request.queries),
    )

    try:
        from backend.services.late_chunking import ChunkingComparison

        comparison = ChunkingComparison()
        report = await comparison.compare(request.text, request.queries)

        return CompareResponse(
            late_chunks=report["late_chunks"],
            traditional_chunks=report["traditional_chunks"],
            avg_improvement_pct=report["avg_improvement_pct"],
            per_query=report["per_query"],
        )

    except Exception as e:
        logger.error("Comparison failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}",
        )


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get current late chunking configuration."""
    try:
        engine = await get_late_chunking_engine()
        config = engine.config

        return ConfigResponse(
            model_name=config.model_name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_document_length=config.max_document_length,
            pooling_strategy=config.pooling_strategy.value,
            normalize_embeddings=config.normalize_embeddings,
        )

    except Exception as e:
        logger.error("Failed to get config", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available late chunking models."""
    return {
        "models": [
            {
                "id": "jinaai/jina-embeddings-v3",
                "name": "Jina Embeddings v3",
                "context_length": 8192,
                "dimensions": 1024,
                "type": "local",
                "description": "Best for late chunking - native support, 8K context",
                "recommended": True,
            },
            {
                "id": "nomic-ai/nomic-embed-text-v1.5",
                "name": "Nomic Embed v1.5",
                "context_length": 8192,
                "dimensions": 768,
                "type": "local",
                "description": "Open-source, good performance, 8K context",
            },
            {
                "id": "BAAI/bge-m3",
                "name": "BGE-M3",
                "context_length": 8192,
                "dimensions": 1024,
                "type": "local",
                "description": "Multilingual, dense+sparse hybrid",
            },
            {
                "id": "voyage-3-large",
                "name": "Voyage v3 Large",
                "context_length": 16000,
                "dimensions": 1024,
                "type": "api",
                "description": "Highest quality, 16K context, requires API key",
            },
        ],
        "pooling_strategies": [
            {"id": "mean", "description": "Average all token embeddings (default)"},
            {"id": "weighted_mean", "description": "Attention-weighted average"},
            {"id": "max", "description": "Max pooling across tokens"},
            {"id": "cls", "description": "Use first (CLS) token"},
            {"id": "last", "description": "Use last token"},
        ],
    }


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Check late chunking service health."""
    try:
        engine = await get_late_chunking_engine()
        return {
            "status": "healthy",
            "initialized": engine._initialized,
            "model": engine.config.model_name,
            "device": str(engine._device) if engine._device else "api",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
