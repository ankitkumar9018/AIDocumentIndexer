"""
AIDocumentIndexer - Feature Integrations API Routes
===================================================

Endpoints for discovering and executing feature integrations.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import AuthenticatedUser
from backend.services.feature_synergy import (
    get_synergy_service,
    FeatureModule,
    PRESET_PIPELINES,
)


logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class FeatureConnectionResponse(BaseModel):
    """A connection between two features."""
    source: str
    target: str
    description: str
    data_flow: str
    example: str


class FeatureCapabilitiesResponse(BaseModel):
    """Capabilities of a feature."""
    feature: str
    can_send_to: List[dict]
    can_receive_from: List[dict]
    total_integrations: int


class FeatureGraphResponse(BaseModel):
    """Full feature integration graph."""
    nodes: List[dict]
    edges: List[dict]


class IntegrationPathResponse(BaseModel):
    """Path between two features."""
    source: str
    target: str
    path: List[FeatureConnectionResponse]
    path_length: int


class PipelineStep(BaseModel):
    """A step in a pipeline."""
    feature: str = Field(..., description="Feature module to use")
    operation: str = Field(..., description="Operation to perform")
    params: dict = Field(default_factory=dict, description="Operation parameters")


class ExecutePipelineRequest(BaseModel):
    """Request to execute a multi-feature pipeline."""
    steps: List[PipelineStep] = Field(..., description="Pipeline steps to execute")
    initial_data: dict = Field(default_factory=dict, description="Initial data for the pipeline")


class PipelineResultStep(BaseModel):
    """Result of a single pipeline step."""
    step: int
    feature: str
    operation: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class ExecutePipelineResponse(BaseModel):
    """Response from pipeline execution."""
    pipeline_status: str
    steps_executed: int
    steps_successful: int
    results: List[PipelineResultStep]
    final_output: dict


class PresetPipelineResponse(BaseModel):
    """A preset integration pipeline."""
    id: str
    name: str
    description: str
    steps: List[dict]


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/connections", response_model=List[FeatureConnectionResponse])
async def list_feature_connections(
    user: AuthenticatedUser,
):
    """
    List all feature connections in the system.

    Shows how different features can integrate with each other,
    including data flow and example use cases.
    """
    service = get_synergy_service()
    return service.get_all_connections()


@router.get("/graph", response_model=FeatureGraphResponse)
async def get_feature_graph(
    user: AuthenticatedUser,
):
    """
    Get the full feature integration graph.

    Returns nodes (features) and edges (connections) for visualization.
    Can be used to render an interactive integration map in the UI.
    """
    service = get_synergy_service()
    return service.get_feature_graph()


@router.get("/features")
async def list_features(
    user: AuthenticatedUser,
):
    """
    List all available feature modules.
    """
    return {
        "features": [
            {
                "id": f.value,
                "name": f.value.replace("_", " ").title(),
                "description": _get_feature_description(f),
            }
            for f in FeatureModule
        ]
    }


@router.get("/features/{feature_id}/capabilities", response_model=FeatureCapabilitiesResponse)
async def get_feature_capabilities(
    feature_id: str,
    user: AuthenticatedUser,
):
    """
    Get the capabilities of a specific feature.

    Shows what other features it can send data to and receive data from.
    """
    try:
        feature = FeatureModule(feature_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown feature: {feature_id}",
        )

    service = get_synergy_service()
    return service.get_feature_capabilities(feature)


@router.get("/path", response_model=IntegrationPathResponse)
async def find_integration_path(
    source: str,
    target: str,
    user: AuthenticatedUser,
):
    """
    Find an integration path between two features.

    Returns the shortest sequence of integrations to connect
    the source feature to the target feature.
    """
    try:
        source_feature = FeatureModule(source)
        target_feature = FeatureModule(target)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown feature",
        )

    service = get_synergy_service()
    path = service.find_integration_path(source_feature, target_feature)

    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No integration path found from {source} to {target}",
        )

    return IntegrationPathResponse(
        source=source,
        target=target,
        path=[
            FeatureConnectionResponse(
                source=c.source.value,
                target=c.target.value,
                description=c.description,
                data_flow=c.data_flow,
                example=c.example_use_case,
            )
            for c in path
        ],
        path_length=len(path),
    )


@router.get("/presets", response_model=List[PresetPipelineResponse])
async def list_preset_pipelines(
    user: AuthenticatedUser,
):
    """
    List pre-built integration pipelines.

    These are common integration patterns that can be executed directly.
    """
    return [
        PresetPipelineResponse(
            id=key,
            name=value["name"],
            description=value["description"],
            steps=value["steps"],
        )
        for key, value in PRESET_PIPELINES.items()
    ]


@router.post("/execute", response_model=ExecutePipelineResponse)
async def execute_pipeline(
    request: ExecutePipelineRequest,
    user: AuthenticatedUser,
):
    """
    Execute a multi-feature integration pipeline.

    Runs a sequence of operations across different features,
    passing data from one step to the next.

    Example request:
    ```json
    {
        "steps": [
            {"feature": "link_groups", "operation": "get_links", "params": {"group_id": "123"}},
            {"feature": "web_scraper", "operation": "scrape", "params": {"storage_mode": "permanent"}}
        ],
        "initial_data": {}
    }
    ```
    """
    service = get_synergy_service()

    steps = [
        {
            "feature": step.feature,
            "operation": step.operation,
            "params": step.params,
        }
        for step in request.steps
    ]

    result = await service.execute_pipeline(
        steps=steps,
        initial_data=request.initial_data,
        user_id=user.user_id,
        organization_id=user.organization_id if hasattr(user, "organization_id") else None,
    )

    return ExecutePipelineResponse(
        pipeline_status=result["pipeline_status"],
        steps_executed=result["steps_executed"],
        steps_successful=result["steps_successful"],
        results=[
            PipelineResultStep(
                step=r["step"],
                feature=r["feature"],
                operation=r["operation"],
                status=r["status"],
                result=r.get("result"),
                error=r.get("error"),
            )
            for r in result["results"]
        ],
        final_output=result["final_output"],
    )


@router.post("/presets/{preset_id}/execute", response_model=ExecutePipelineResponse)
async def execute_preset_pipeline(
    preset_id: str,
    initial_data: dict,
    user: AuthenticatedUser,
):
    """
    Execute a preset integration pipeline.

    Provide initial data required by the pipeline (e.g., group_id, query, etc.).
    """
    if preset_id not in PRESET_PIPELINES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset pipeline not found: {preset_id}",
        )

    preset = PRESET_PIPELINES[preset_id]
    service = get_synergy_service()

    result = await service.execute_pipeline(
        steps=preset["steps"],
        initial_data=initial_data,
        user_id=user.user_id,
        organization_id=user.organization_id if hasattr(user, "organization_id") else None,
    )

    return ExecutePipelineResponse(
        pipeline_status=result["pipeline_status"],
        steps_executed=result["steps_executed"],
        steps_successful=result["steps_successful"],
        results=[
            PipelineResultStep(
                step=r["step"],
                feature=r["feature"],
                operation=r["operation"],
                status=r["status"],
                result=r.get("result"),
                error=r.get("error"),
            )
            for r in result["results"]
        ],
        final_output=result["final_output"],
    )


# =============================================================================
# Helpers
# =============================================================================

def _get_feature_description(feature: FeatureModule) -> str:
    """Get a human-readable description for a feature."""
    descriptions = {
        FeatureModule.LINK_GROUPS: "Organize URLs into groups for batch scraping",
        FeatureModule.WEB_SCRAPER: "Scrape web pages and extract content",
        FeatureModule.DOCUMENTS: "Store and manage uploaded documents",
        FeatureModule.COLLECTIONS: "Group documents into searchable collections",
        FeatureModule.FOLDERS: "Hierarchical folder organization with permissions",
        FeatureModule.KNOWLEDGE_GRAPH: "Extract entities and relationships from text",
        FeatureModule.SKILLS: "AI-powered reusable capabilities",
        FeatureModule.WORKFLOWS: "Orchestrate multi-step automations",
        FeatureModule.CHAT: "RAG-powered conversational interface",
        FeatureModule.RESEARCH: "Deep research with web search and analysis",
        FeatureModule.REPORTS: "Generate formatted reports from data",
        FeatureModule.CONNECTORS: "Sync with external services (Notion, GitHub, etc.)",
        FeatureModule.EXTERNAL_API: "Expose features via REST API",
        FeatureModule.EMBEDDINGS: "Generate vector embeddings for semantic search",
        FeatureModule.RERANKING: "Improve search relevance with reranking",
    }
    return descriptions.get(feature, "")
