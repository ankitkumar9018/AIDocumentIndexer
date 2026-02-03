"""
AIDocumentIndexer - Visualization API Routes
=============================================

API endpoints for embedding visualization and citation mapping.

Endpoints:
- POST /project: Project embeddings to 2D/3D
- POST /citations-map: Generate citation visualization data
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.embedding_projection import (
    get_projection_service,
    ProjectionMethod,
    ProjectionConfig,
)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class ProjectionRequest(BaseModel):
    """Request to project embeddings."""
    embeddings: List[List[float]] = Field(
        ..., description="List of embedding vectors"
    )
    ids: List[str] = Field(
        ..., description="List of identifiers for each embedding"
    )
    scores: Optional[List[float]] = Field(
        None, description="Optional relevance scores"
    )
    metadata: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional metadata for each point"
    )
    method: Optional[str] = Field(
        None, description="Projection method: umap, tsne, or pca"
    )
    n_components: int = Field(
        2, description="Number of output dimensions (2 or 3)"
    )


class ProjectedPointResponse(BaseModel):
    """A projected point."""
    id: str
    x: float
    y: float
    z: Optional[float] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = {}


class ProjectionResponse(BaseModel):
    """Response with projected points."""
    points: List[ProjectedPointResponse]
    method: str
    n_components: int
    processing_time_ms: float
    cache_hit: bool
    quality_metrics: Dict[str, float] = {}


class CitationMapRequest(BaseModel):
    """Request to generate citation visualization."""
    citations: List[Dict[str, Any]] = Field(
        ..., description="List of citations with id, content, score, embedding"
    )
    query_embedding: Optional[List[float]] = Field(
        None, description="Optional query embedding for reference point"
    )
    method: str = Field(
        "umap", description="Projection method"
    )


class CitationMapResponse(BaseModel):
    """Response with citation map data."""
    points: List[Dict[str, Any]]
    query_point: Optional[Dict[str, float]] = None
    method: str
    processing_time_ms: float


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/project", response_model=ProjectionResponse)
async def project_embeddings(request: ProjectionRequest):
    """
    Project high-dimensional embeddings to 2D/3D for visualization.

    Uses UMAP by default for best quality, with fallback to PCA.

    Example:
        POST /api/v1/visualization/project
        {
            "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
            "ids": ["chunk1", "chunk2"],
            "scores": [0.95, 0.87],
            "method": "umap"
        }
    """
    if len(request.embeddings) != len(request.ids):
        raise HTTPException(
            status_code=400,
            detail="Number of embeddings must match number of ids"
        )

    if request.scores and len(request.scores) != len(request.ids):
        raise HTTPException(
            status_code=400,
            detail="Number of scores must match number of ids"
        )

    if request.n_components not in [2, 3]:
        raise HTTPException(
            status_code=400,
            detail="n_components must be 2 or 3"
        )

    # Get projection service
    service = get_projection_service()

    # Override config if needed
    if request.n_components != 2:
        service.config.n_components = request.n_components

    # Determine method
    method = None
    if request.method:
        try:
            method = ProjectionMethod(request.method.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method: {request.method}. Use umap, tsne, or pca"
            )

    # Project embeddings
    result = await service.project(
        embeddings=request.embeddings,
        ids=request.ids,
        scores=request.scores,
        metadata=request.metadata,
        method=method,
    )

    # Convert to response
    return ProjectionResponse(
        points=[
            ProjectedPointResponse(
                id=p.id,
                x=p.x,
                y=p.y,
                z=p.z,
                score=p.original_score,
                metadata=p.metadata,
            )
            for p in result.points
        ],
        method=result.method,
        n_components=result.n_components,
        processing_time_ms=result.processing_time_ms,
        cache_hit=result.cache_hit,
        quality_metrics=result.quality_metrics,
    )


@router.post("/citations-map", response_model=CitationMapResponse)
async def generate_citation_map(request: CitationMapRequest):
    """
    Generate visualization data for citations.

    Takes citation data (with embeddings) and returns 2D coordinates
    for visualization.

    Example:
        POST /api/v1/visualization/citations-map
        {
            "citations": [
                {
                    "id": "chunk1",
                    "content": "...",
                    "score": 0.95,
                    "embedding": [0.1, 0.2, ...]
                }
            ],
            "query_embedding": [0.5, 0.6, ...],
            "method": "umap"
        }
    """
    import time
    start_time = time.time()

    if not request.citations:
        return CitationMapResponse(
            points=[],
            query_point=None,
            method=request.method,
            processing_time_ms=0,
        )

    # Extract embeddings and metadata
    embeddings = []
    ids = []
    scores = []
    metadata_list = []

    for citation in request.citations:
        if "embedding" in citation and citation["embedding"]:
            embeddings.append(citation["embedding"])
            ids.append(citation.get("id", ""))
            scores.append(citation.get("score", 0.0))
            metadata_list.append({
                "content": citation.get("content", "")[:200],
                "documentTitle": citation.get("documentTitle", ""),
                "pageNumber": citation.get("pageNumber"),
                "sourceType": citation.get("sourceType", "vector"),
            })

    # Include query embedding if provided
    include_query = request.query_embedding is not None
    if include_query:
        embeddings.insert(0, request.query_embedding)
        ids.insert(0, "__query__")
        scores.insert(0, 1.0)
        metadata_list.insert(0, {"type": "query"})

    # Project embeddings
    service = get_projection_service()

    method = None
    try:
        method = ProjectionMethod(request.method.lower())
    except ValueError:
        method = ProjectionMethod.UMAP

    result = await service.project(
        embeddings=embeddings,
        ids=ids,
        scores=scores,
        metadata=metadata_list,
        method=method,
    )

    # Separate query point from citation points
    query_point = None
    points = []

    for point in result.points:
        if point.id == "__query__":
            query_point = {"x": point.x, "y": point.y}
        else:
            points.append({
                "id": point.id,
                "x": point.x,
                "y": point.y,
                "score": point.original_score,
                **point.metadata,
            })

    processing_time = (time.time() - start_time) * 1000

    return CitationMapResponse(
        points=points,
        query_point=query_point,
        method=result.method,
        processing_time_ms=processing_time,
    )


@router.get("/methods")
async def get_available_methods():
    """
    Get available projection methods.

    Returns which dimensionality reduction methods are available
    based on installed packages.
    """
    service = get_projection_service()

    methods = []
    for method in service._available_methods:
        info = {
            "name": method.value,
            "description": "",
        }

        if method == ProjectionMethod.UMAP:
            info["description"] = "UMAP - Best for preserving both local and global structure"
        elif method == ProjectionMethod.TSNE:
            info["description"] = "t-SNE - Good for visualizing clusters"
        elif method == ProjectionMethod.PCA:
            info["description"] = "PCA - Fast linear projection"

        methods.append(info)

    return {
        "methods": methods,
        "default": service.config.method.value,
    }
