"""
AIDocumentIndexer - Compression API Routes
==========================================

API endpoints for prompt and context compression.

Supports:
- LLMLingua-2: Token-level compression (3-6x, 95%+ accuracy)
- OSCAR: Online context compression (2-5x latency reduction)
- Context rolling: Conversation compression
"""

from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import get_current_user
from backend.db.models import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/compression", tags=["compression"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CompressionModeEnum(str, Enum):
    """Compression aggressiveness modes."""
    LIGHT = "light"           # ~1.4x compression
    MODERATE = "moderate"     # ~2x compression
    AGGRESSIVE = "aggressive" # ~3x compression
    EXTREME = "extreme"       # ~5x compression
    ADAPTIVE = "adaptive"     # Content-based


class CompressTextRequest(BaseModel):
    """Request to compress text."""
    text: str = Field(..., description="Text to compress")
    mode: CompressionModeEnum = Field(
        CompressionModeEnum.MODERATE,
        description="Compression mode",
    )
    target_ratio: Optional[float] = Field(
        None, ge=0.1, le=0.9,
        description="Specific target ratio (overrides mode)",
    )


class CompressTextResponse(BaseModel):
    """Response from text compression."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time_ms: float
    mode_used: str


class CompressBatchRequest(BaseModel):
    """Request to compress multiple texts."""
    texts: List[str] = Field(
        ...,
        description="Texts to compress",
        min_length=1,
        max_length=100,  # Limit to prevent DoS
    )
    mode: CompressionModeEnum = Field(CompressionModeEnum.MODERATE)


class CompressBatchResponse(BaseModel):
    """Response from batch compression."""
    compressed_texts: List[str]
    total_original_tokens: int
    total_compressed_tokens: int
    avg_compression_ratio: float
    total_processing_time_ms: float


class CompressRAGRequest(BaseModel):
    """Request to compress RAG context."""
    query: str = Field(..., description="User query for relevance", max_length=2000)
    chunks: List[str] = Field(
        ...,
        description="Retrieved chunks to compress",
        min_length=1,
        max_length=50,  # Limit to prevent DoS
    )
    target_tokens: int = Field(2000, ge=100, le=8000, description="Target total tokens")
    preserve_query_terms: bool = Field(True, description="Preserve query-relevant content")


class CompressRAGResponse(BaseModel):
    """Response from RAG compression."""
    compressed_chunks: List[str]
    original_token_estimate: int
    compressed_token_estimate: int
    compression_ratio: float


class OSCARCompressRequest(BaseModel):
    """Request for OSCAR online compression."""
    query: str = Field(..., description="Query for relevance scoring", max_length=2000)
    chunks: List[str] = Field(
        ...,
        description="Chunks to compress",
        min_length=1,
        max_length=50,  # Limit to prevent DoS
    )
    target_tokens: int = Field(2000, ge=100, le=8000)


class OSCARCompressResponse(BaseModel):
    """Response from OSCAR compression."""
    compressed_context: str
    segments_kept: int
    segments_total: int
    compression_ratio: float
    processing_time_ms: float


# =============================================================================
# LLMLingua-2 Endpoints
# =============================================================================

@router.post("/llmlingua/compress", response_model=CompressTextResponse)
async def llmlingua_compress(
    request: CompressTextRequest,
    current_user: User = Depends(get_current_user),
) -> CompressTextResponse:
    """
    Compress text using LLMLingua-2.

    LLMLingua-2 uses a token classifier to identify important tokens,
    achieving 3-6x compression with 95-98% accuracy retention.

    Benefits:
    - 10-100x faster than LLM summarization
    - Preserves exact wording (no paraphrasing)
    - Works with any downstream LLM
    """
    logger.info(
        "LLMLingua-2 compression",
        text_length=len(request.text),
        mode=request.mode,
    )

    if len(request.text) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text too short for compression",
        )

    try:
        from backend.services.llmlingua_compression import (
            get_llmlingua_engine,
            CompressionMode,
        )

        engine = await get_llmlingua_engine()

        mode = CompressionMode(request.mode.value)
        result = await engine.compress(
            request.text,
            mode=mode,
            target_ratio=request.target_ratio,
        )

        return CompressTextResponse(
            compressed_text=result.compressed_text,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.compression_ratio,
            processing_time_ms=result.processing_time_ms,
            mode_used=result.mode_used,
        )

    except ImportError as e:
        logger.error("LLMLingua-2 dependencies missing", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLMLingua-2 not available. Install: pip install transformers torch",
        )
    except Exception as e:
        logger.error("LLMLingua-2 compression failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compression failed",
        )


@router.post("/llmlingua/compress-batch", response_model=CompressBatchResponse)
async def llmlingua_compress_batch(
    request: CompressBatchRequest,
    current_user: User = Depends(get_current_user),
) -> CompressBatchResponse:
    """
    Compress multiple texts using LLMLingua-2.

    Efficient batch processing for multiple documents.
    """
    logger.info(
        "LLMLingua-2 batch compression",
        num_texts=len(request.texts),
    )

    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No texts provided",
        )

    try:
        from backend.services.llmlingua_compression import (
            get_llmlingua_engine,
            CompressionMode,
        )

        engine = await get_llmlingua_engine()

        mode = CompressionMode(request.mode.value)
        result = await engine.compress_batch(request.texts, mode=mode)

        return CompressBatchResponse(
            compressed_texts=[r.compressed_text for r in result.results],
            total_original_tokens=result.total_original_tokens,
            total_compressed_tokens=result.total_compressed_tokens,
            avg_compression_ratio=result.avg_compression_ratio,
            total_processing_time_ms=result.total_processing_time_ms,
        )

    except Exception as e:
        logger.error("Batch compression failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch compression failed",
        )


@router.post("/llmlingua/compress-rag", response_model=CompressRAGResponse)
async def llmlingua_compress_rag(
    request: CompressRAGRequest,
    current_user: User = Depends(get_current_user),
) -> CompressRAGResponse:
    """
    Compress RAG context using LLMLingua-2.

    Compresses retrieved chunks while preserving query-relevant content.
    Useful for fitting more context into LLM token limits.
    """
    logger.info(
        "LLMLingua-2 RAG compression",
        query_preview=request.query[:50],
        num_chunks=len(request.chunks),
        target_tokens=request.target_tokens,
    )

    if not request.chunks:
        return CompressRAGResponse(
            compressed_chunks=[],
            original_token_estimate=0,
            compressed_token_estimate=0,
            compression_ratio=1.0,
        )

    try:
        from backend.services.llmlingua_compression import get_llmlingua_engine

        engine = await get_llmlingua_engine()

        compressed = await engine.compress_rag_context(
            request.query,
            request.chunks,
            request.target_tokens,
        )

        # Estimate tokens
        original_estimate = int(sum(len(c.split()) * 1.3 for c in request.chunks))
        compressed_estimate = int(sum(len(c.split()) * 1.3 for c in compressed))

        return CompressRAGResponse(
            compressed_chunks=compressed,
            original_token_estimate=original_estimate,
            compressed_token_estimate=compressed_estimate,
            compression_ratio=original_estimate / max(compressed_estimate, 1),
        )

    except Exception as e:
        logger.error("RAG compression failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG compression failed",
        )


# =============================================================================
# OSCAR Endpoints
# =============================================================================

@router.post("/oscar/compress", response_model=OSCARCompressResponse)
async def oscar_compress(
    request: OSCARCompressRequest,
    current_user: User = Depends(get_current_user),
) -> OSCARCompressResponse:
    """
    Compress context using OSCAR (Online Subspace-based Context Approximate Reasoning).

    OSCAR performs streaming context compression by:
    1. Extracting semantic segments
    2. Scoring by query relevance
    3. Removing redundancy
    4. Selecting optimal subset

    Benefits:
    - 2-5x latency reduction
    - Preserves most relevant information
    - Online (streaming) processing
    """
    logger.info(
        "OSCAR compression",
        query_preview=request.query[:50],
        num_chunks=len(request.chunks),
    )

    if not request.chunks:
        return OSCARCompressResponse(
            compressed_context="",
            segments_kept=0,
            segments_total=0,
            compression_ratio=1.0,
            processing_time_ms=0.0,
        )

    try:
        from backend.services.oscar_compression import get_oscar_compressor

        compressor = await get_oscar_compressor()

        result = await compressor.compress_context(
            query=request.query,
            chunks=request.chunks,
            target_tokens=request.target_tokens,
        )

        return OSCARCompressResponse(
            compressed_context=result.compressed_text,
            segments_kept=result.segments_kept,
            segments_total=result.segments_total,
            compression_ratio=result.compression_ratio,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.error("OSCAR compression failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OSCAR compression failed",
        )


# =============================================================================
# Info Endpoints
# =============================================================================

@router.get("/methods")
async def list_compression_methods(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """List available compression methods with comparison."""
    return {
        "methods": [
            {
                "id": "llmlingua-2",
                "name": "LLMLingua-2",
                "description": "Token-level compression using XLM-RoBERTa classifier",
                "compression_ratio": "3-6x",
                "accuracy_retention": "95-98%",
                "speed": "Fast (no LLM calls)",
                "best_for": "Prompt compression, long documents",
                "preserves_wording": True,
            },
            {
                "id": "oscar",
                "name": "OSCAR",
                "description": "Online context compression via segment scoring",
                "compression_ratio": "2-5x",
                "accuracy_retention": "90-95%",
                "speed": "Fast",
                "best_for": "RAG context, streaming",
                "preserves_wording": True,
            },
            {
                "id": "context-rolling",
                "name": "Context Rolling",
                "description": "LLM-based conversation compression",
                "compression_ratio": "5-32x",
                "accuracy_retention": "85-95%",
                "speed": "Slow (LLM calls)",
                "best_for": "Conversation history",
                "preserves_wording": False,
            },
        ],
        "modes": [
            {"id": "light", "ratio": "~1.4x", "accuracy": "~99%"},
            {"id": "moderate", "ratio": "~2x", "accuracy": "~97%"},
            {"id": "aggressive", "ratio": "~3x", "accuracy": "~95%"},
            {"id": "extreme", "ratio": "~5x", "accuracy": "~90%"},
            {"id": "adaptive", "ratio": "varies", "accuracy": "optimized"},
        ],
    }


@router.get("/health")
async def compression_health(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Check compression services health."""
    status_map = {}

    # Check LLMLingua-2
    try:
        from backend.services.llmlingua_compression import get_llmlingua_engine
        engine = await get_llmlingua_engine()
        status_map["llmlingua-2"] = {
            "status": "available",
            "initialized": engine._sync_engine._initialized,
            "device": str(engine._sync_engine._device) if engine._sync_engine._device else "not loaded",
        }
    except Exception as e:
        status_map["llmlingua-2"] = {
            "status": "unavailable",
            "error": str(e),
        }

    # Check OSCAR
    try:
        from backend.services.oscar_compression import get_oscar_compressor
        compressor = await get_oscar_compressor()
        status_map["oscar"] = {
            "status": "available",
            "initialized": True,
        }
    except Exception as e:
        status_map["oscar"] = {
            "status": "unavailable",
            "error": str(e),
        }

    overall_healthy = all(s.get("status") == "available" for s in status_map.values())

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "services": status_map,
    }
