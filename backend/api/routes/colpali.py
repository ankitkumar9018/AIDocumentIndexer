"""
AIDocumentIndexer - ColPali Visual Retrieval API Routes
========================================================

API endpoints for configuring ColPali visual document retrieval.
ColPali enables searching documents directly from images without OCR.
"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import require_admin
from backend.services.settings import get_settings_service
from backend.core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/retrieval/colpali", tags=["ColPali Visual Retrieval"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ColPaliConfigResponse(BaseModel):
    """ColPali configuration response."""
    enabled: bool = Field(..., description="Whether ColPali visual retrieval is enabled")
    model_name: str = Field(..., description="ColPali model name")
    weight: float = Field(..., description="Weight in hybrid retrieval (0-1)")
    index_path: str = Field(..., description="Path to ColPali index")
    is_available: bool = Field(..., description="Whether ColPali engine is installed")
    device: str = Field(..., description="Device for model inference")


class ColPaliConfigUpdate(BaseModel):
    """Update ColPali configuration."""
    enabled: Optional[bool] = Field(None, description="Enable/disable ColPali")
    weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Hybrid weight")
    model_name: Optional[str] = Field(None, description="Model name")


class ColPaliStatsResponse(BaseModel):
    """ColPali retrieval statistics."""
    is_available: bool
    model_loaded: bool
    index_built: bool
    indexed_images: int
    unique_documents: int
    model_name: str
    device: Optional[str]


class ColPaliSearchRequest(BaseModel):
    """Request to search with ColPali."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    document_ids: Optional[List[str]] = Field(None, description="Filter to specific documents")


class ColPaliSearchResult(BaseModel):
    """A single ColPali search result."""
    document_id: str
    image_id: str
    score: float
    rank: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ColPaliSearchResponse(BaseModel):
    """Response from ColPali search."""
    results: List[ColPaliSearchResult]
    query: str
    total_results: int


class IndexDocumentRequest(BaseModel):
    """Request to index a document with ColPali."""
    document_id: str = Field(..., description="Document ID to index")
    force_reindex: bool = Field(False, description="Force re-indexing if already indexed")


# =============================================================================
# Routes
# =============================================================================

@router.get("/config", response_model=ColPaliConfigResponse)
async def get_colpali_config(
    current_user: dict = Depends(require_admin),
):
    """
    Get ColPali visual retrieval configuration.

    ColPali (ICLR 2025) enables visual document retrieval:
    - No OCR required - searches directly on document images
    - ColBERT-style late interaction scoring
    - +40% accuracy on visual document benchmarks

    Requires admin privileges.
    """
    settings_service = get_settings_service()

    # Check if ColPali is available
    try:
        from backend.services.colpali_retriever import HAS_COLPALI
        is_available = HAS_COLPALI
    except ImportError:
        is_available = False

    return ColPaliConfigResponse(
        enabled=settings_service.get("rag.colpali_enabled", False),
        model_name=settings_service.get("rag.colpali_model", "vidore/colpali-v1.2"),
        weight=settings_service.get("rag.colpali_weight", 0.15),
        index_path=settings_service.get("rag.colpali_index_path", "./data/colpali_index"),
        is_available=is_available,
        device=settings_service.get("rag.colpali_device", "auto"),
    )


@router.put("/config", response_model=ColPaliConfigResponse)
async def update_colpali_config(
    request: ColPaliConfigUpdate,
    current_user: dict = Depends(require_admin),
):
    """
    Update ColPali configuration.

    Note: Changing model requires rebuilding the index.

    Requires admin privileges.
    """
    settings_service = get_settings_service()

    if request.enabled is not None:
        settings_service.set("rag.colpali_enabled", request.enabled)

    if request.weight is not None:
        settings_service.set("rag.colpali_weight", request.weight)

    if request.model_name is not None:
        settings_service.set("rag.colpali_model", request.model_name)

    logger.info(
        "colpali.config_updated",
        enabled=request.enabled,
        weight=request.weight,
        model=request.model_name,
        user_id=current_user.get("id"),
    )

    # Check if ColPali is available
    try:
        from backend.services.colpali_retriever import HAS_COLPALI
        is_available = HAS_COLPALI
    except ImportError:
        is_available = False

    return ColPaliConfigResponse(
        enabled=settings_service.get("rag.colpali_enabled", False),
        model_name=settings_service.get("rag.colpali_model", "vidore/colpali-v1.2"),
        weight=settings_service.get("rag.colpali_weight", 0.15),
        index_path=settings_service.get("rag.colpali_index_path", "./data/colpali_index"),
        is_available=is_available,
        device=settings_service.get("rag.colpali_device", "auto"),
    )


@router.get("/stats", response_model=ColPaliStatsResponse)
async def get_colpali_stats(
    current_user: dict = Depends(require_admin),
):
    """
    Get ColPali retriever statistics.

    Requires admin privileges.
    """
    try:
        from backend.services.colpali_retriever import get_colpali_retriever

        retriever = await get_colpali_retriever()
        stats = retriever.get_stats()

        return ColPaliStatsResponse(
            is_available=stats.get("available", False),
            model_loaded=stats.get("model_loaded", False),
            index_built=stats.get("index_built", False),
            indexed_images=stats.get("indexed_images", 0),
            unique_documents=stats.get("unique_documents", 0),
            model_name=stats.get("model_name", "unknown"),
            device=stats.get("device"),
        )

    except ImportError:
        return ColPaliStatsResponse(
            is_available=False,
            model_loaded=False,
            index_built=False,
            indexed_images=0,
            unique_documents=0,
            model_name="colpali-engine not installed",
            device=None,
        )
    except Exception as e:
        logger.error("Failed to get ColPali stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get stats",
        )


@router.post("/search", response_model=ColPaliSearchResponse)
async def search_with_colpali(
    request: ColPaliSearchRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Search using ColPali visual document retrieval.

    This searches document images directly without OCR,
    using vision-language models with ColBERT-style scoring.

    Requires admin privileges.
    """
    settings_service = get_settings_service()

    if not settings_service.get("rag.colpali_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ColPali visual retrieval is not enabled",
        )

    try:
        from backend.services.colpali_retriever import get_colpali_retriever

        retriever = await get_colpali_retriever()

        if not retriever.is_available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ColPali is not available. Install with: pip install colpali-engine",
            )

        results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        return ColPaliSearchResponse(
            results=[
                ColPaliSearchResult(
                    document_id=r.document_id,
                    image_id=r.image_id,
                    score=r.score,
                    rank=r.rank,
                    metadata=r.metadata,
                )
                for r in results
            ],
            query=request.query,
            total_results=len(results),
        )

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ColPali is not available. Install with: pip install colpali-engine",
        )
    except Exception as e:
        logger.error("ColPali search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        )


@router.post("/index")
async def index_document_images(
    request: IndexDocumentRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Index a document's images with ColPali for visual retrieval.

    This converts PDF pages to images and indexes them for
    visual document search.

    Requires admin privileges.
    """
    settings_service = get_settings_service()

    if not settings_service.get("rag.colpali_enabled", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ColPali visual retrieval is not enabled",
        )

    try:
        from backend.services.colpali_retriever import get_colpali_retriever
        from backend.db.database import async_session_context
        from backend.db.models import Document

        retriever = await get_colpali_retriever()

        if not retriever.is_available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ColPali is not available",
            )

        # Get document path from database
        async with async_session_context() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Document).where(Document.id == request.document_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {request.document_id}",
                )

            # Index PDF pages
            if document.file_path and document.file_path.lower().endswith('.pdf'):
                success = await retriever.index_pdf_pages(
                    pdf_path=document.file_path,
                    document_id=request.document_id,
                )

                if success:
                    # Save index
                    await retriever.save_index()

                    return {
                        "message": "Document indexed successfully",
                        "document_id": request.document_id,
                        "indexed": True,
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to index document images",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="ColPali indexing currently only supports PDF documents",
                )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ColPali is not available",
        )
    except Exception as e:
        logger.error("ColPali indexing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Indexing failed",
        )


@router.post("/enable")
async def enable_colpali(
    current_user: dict = Depends(require_admin),
):
    """
    Enable ColPali visual document retrieval.

    Requires admin privileges.
    """
    settings_service = get_settings_service()
    settings_service.set("rag.colpali_enabled", True)

    logger.info(
        "colpali.enabled",
        user_id=current_user.get("id"),
    )

    return {"message": "ColPali visual retrieval enabled", "status": "active"}


@router.post("/disable")
async def disable_colpali(
    current_user: dict = Depends(require_admin),
):
    """
    Disable ColPali visual document retrieval.

    Requires admin privileges.
    """
    settings_service = get_settings_service()
    settings_service.set("rag.colpali_enabled", False)

    logger.info(
        "colpali.disabled",
        user_id=current_user.get("id"),
    )

    return {"message": "ColPali visual retrieval disabled", "status": "inactive"}


@router.get("/health")
async def colpali_health():
    """Check ColPali service health."""
    try:
        from backend.services.colpali_retriever import HAS_COLPALI, get_colpali_retriever

        if not HAS_COLPALI:
            return {
                "status": "unavailable",
                "reason": "colpali-engine not installed",
                "install": "pip install colpali-engine",
            }

        retriever = await get_colpali_retriever()
        stats = retriever.get_stats()

        return {
            "status": "healthy" if stats.get("available") else "degraded",
            "model_loaded": stats.get("model_loaded", False),
            "indexed_images": stats.get("indexed_images", 0),
            "device": stats.get("device"),
        }

    except ImportError:
        return {
            "status": "unavailable",
            "reason": "colpali-engine not installed",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
