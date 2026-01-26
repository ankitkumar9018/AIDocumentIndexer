"""
AIDocumentIndexer - Agent & Workflow Templates API Routes
==========================================================

REST API endpoints for agent and workflow templates.

Endpoints:
- GET  /agent-templates/agents              - List agent templates
- GET  /agent-templates/agents/{id}         - Get agent template details
- GET  /agent-templates/agents/{id}/config  - Get full agent config
- GET  /agent-templates/agents/{id}/enhancements - Get agent enhancements options
- POST /agent-templates/agents/{id}/create  - Create agent from template

- GET  /agent-templates/workflows           - List workflow templates
- GET  /agent-templates/workflows/{id}      - Get workflow template details
- GET  /agent-templates/workflows/{id}/config - Get full workflow config
- POST /agent-templates/workflows/{id}/create - Create workflow from template
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.api.deps import get_current_user
from backend.services.template_service import (
    get_template_service,
    TemplateInfo,
    TemplateType,
)

router = APIRouter(prefix="/agent-templates", tags=["Agent & Workflow Templates"])


# =============================================================================
# Response Models
# =============================================================================

class TemplateListItem(BaseModel):
    """Template list item."""
    id: str
    name: str
    description: str
    type: str
    version: str
    category: Optional[str] = None
    complexity: Optional[str] = None
    patterns_used: Optional[List[str]] = None


class AgentTemplateResponse(BaseModel):
    """Agent template response."""
    id: str
    name: str
    description: str
    type: str
    version: str
    system_prompt: str
    capabilities: Dict[str, bool]
    enhancements: Dict[str, Any]
    llm_config: Dict[str, Any]
    embedding_config: Optional[Dict[str, Any]] = None
    voice_config: Optional[Dict[str, Any]] = None
    ui_config: Optional[Dict[str, Any]] = None
    rate_limits: Optional[Dict[str, int]] = None
    languages: List[str]


class WorkflowTemplateResponse(BaseModel):
    """Workflow template response."""
    id: str
    name: str
    description: str
    version: str
    trigger: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    variables: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    error_handling: Optional[Dict[str, Any]] = None
    observability: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None


class EnhancementsResponse(BaseModel):
    """Agent enhancements response."""
    enhancements: Dict[str, Any]
    capabilities: Dict[str, bool]
    description: str


class CreateFromTemplateRequest(BaseModel):
    """Request to create agent/workflow from template."""
    name: Optional[str] = Field(None, description="Custom name for the created item")
    customizations: Optional[Dict[str, Any]] = Field(
        None,
        description="Customizations to apply to the template"
    )


class CreateFromTemplateResponse(BaseModel):
    """Response from creating from template."""
    success: bool
    config: Dict[str, Any]
    message: str


# =============================================================================
# Agent Templates Endpoints
# =============================================================================

@router.get("/agents", response_model=List[TemplateListItem])
async def list_agent_templates(
    current_user: dict = Depends(get_current_user),
):
    """
    List all available agent templates.

    Returns templates for chat agents, voice agents, and API-only agents.
    Each template includes enhancement options that can be customized.
    """
    service = await get_template_service()
    templates = await service.list_agent_templates()

    return [
        TemplateListItem(
            id=t.id,
            name=t.name,
            description=t.description,
            type=t.type.value,
            version=t.version,
            category=t.category,
            complexity=t.complexity,
            patterns_used=t.patterns_used,
        )
        for t in templates
    ]


@router.get("/agents/{template_id}", response_model=AgentTemplateResponse)
async def get_agent_template(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get detailed information about an agent template.

    Includes all configuration options, enhancements, and capabilities.
    """
    service = await get_template_service()
    template = await service.get_agent_template(template_id)

    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent template not found: {template_id}")

    return AgentTemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        type=template.type.value,
        version=template.version,
        system_prompt=template.system_prompt,
        capabilities=template.capabilities,
        enhancements=template.enhancements,
        llm_config=template.llm_config,
        embedding_config=template.embedding_config,
        voice_config=template.voice_config,
        ui_config=template.ui_config,
        rate_limits=template.rate_limits,
        languages=template.languages or ["en"],
    )


@router.get("/agents/{template_id}/config")
async def get_agent_template_config(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get the full raw configuration for an agent template.

    Returns the complete JSON configuration that can be used to create an agent.
    """
    service = await get_template_service()
    config = await service.get_agent_template_config(template_id)

    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent template not found: {template_id}")

    return config


@router.get("/agents/{template_id}/enhancements", response_model=EnhancementsResponse)
async def get_agent_enhancements(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get available enhancement options for an agent template.

    Enhancements include:
    - Query enhancement (expansion, HyDE, spell correction)
    - Retrieval enhancement (hybrid search, tiered reranking, GraphRAG)
    - Answer enhancement (self-refine, chain of verification, confidence scoring)
    - Memory enhancement (Mem0, decay settings)
    - Context management (compression, recursive LM)
    - Caching (generative cache)
    """
    service = await get_template_service()
    enhancements = await service.get_agent_enhancements(template_id)

    if not enhancements:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent template not found: {template_id}")

    return EnhancementsResponse(**enhancements)


@router.post("/agents/{template_id}/create", response_model=CreateFromTemplateResponse)
async def create_agent_from_template(
    template_id: str,
    request: CreateFromTemplateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Create a new agent configuration from a template.

    Apply optional customizations to override template defaults.
    Returns the complete agent configuration ready for deployment.
    """
    service = await get_template_service()

    try:
        config = await service.create_agent_from_template(
            template_id=template_id,
            customizations=request.customizations,
        )

        if request.name:
            config["name"] = request.name

        return CreateFromTemplateResponse(
            success=True,
            config=config,
            message=f"Agent configuration created from template '{template_id}'",
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create agent: {str(e)}")


# =============================================================================
# Workflow Templates Endpoints
# =============================================================================

@router.get("/workflows", response_model=List[TemplateListItem])
async def list_workflow_templates(
    current_user: dict = Depends(get_current_user),
):
    """
    List all available workflow templates.

    Returns templates for document processing, customer support, and other workflows.
    Includes metadata about complexity, patterns used, and trigger types.
    """
    service = await get_template_service()
    templates = await service.list_workflow_templates()

    return [
        TemplateListItem(
            id=t.id,
            name=t.name,
            description=t.description,
            type=t.type.value,
            version=t.version,
            category=t.category,
            complexity=t.complexity,
            patterns_used=t.patterns_used,
        )
        for t in templates
    ]


@router.get("/workflows/{template_id}", response_model=WorkflowTemplateResponse)
async def get_workflow_template(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get detailed information about a workflow template.

    Includes all nodes, edges, variables, and configuration options.
    """
    service = await get_template_service()
    template = await service.get_workflow_template(template_id)

    if not template:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow template not found: {template_id}")

    return WorkflowTemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        version=template.version,
        trigger=template.trigger,
        nodes=template.nodes,
        edges=template.edges,
        variables=template.variables,
        config=template.config,
        metadata=template.metadata,
        error_handling=template.error_handling,
        observability=template.observability,
        security=template.security,
    )


@router.get("/workflows/{template_id}/config")
async def get_workflow_template_config(
    template_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Get the full raw configuration for a workflow template.

    Returns the complete JSON configuration that can be used to create a workflow.
    """
    service = await get_template_service()
    config = await service.get_workflow_template_config(template_id)

    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workflow template not found: {template_id}")

    return config


@router.post("/workflows/{template_id}/create", response_model=CreateFromTemplateResponse)
async def create_workflow_from_template(
    template_id: str,
    request: CreateFromTemplateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Create a new workflow configuration from a template.

    Apply optional customizations to override template defaults.
    Returns the complete workflow configuration ready for deployment.
    """
    service = await get_template_service()

    try:
        config = await service.create_workflow_from_template(
            template_id=template_id,
            customizations=request.customizations,
        )

        if request.name:
            config["name"] = request.name

        return CreateFromTemplateResponse(
            success=True,
            config=config,
            message=f"Workflow configuration created from template '{template_id}'",
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create workflow: {str(e)}")


# =============================================================================
# Pattern Documentation Endpoint
# =============================================================================

@router.get("/patterns")
async def list_workflow_patterns(
    current_user: dict = Depends(get_current_user),
):
    """
    List available workflow patterns with descriptions.

    Based on 2025-2026 best practices for AI agent orchestration.
    """
    return {
        "patterns": [
            {
                "id": "orchestrator_worker",
                "name": "Orchestrator-Worker",
                "description": "Orchestrator coordinates multiple worker agents for complex tasks",
                "use_cases": ["Document processing", "Multi-step analysis", "Parallel execution"],
                "complexity": "advanced",
            },
            {
                "id": "react",
                "name": "ReAct (Reason + Act)",
                "description": "Interleaves reasoning with tool execution in a loop",
                "use_cases": ["Research tasks", "Information gathering", "Problem solving"],
                "complexity": "intermediate",
            },
            {
                "id": "reflection",
                "name": "Reflection/Self-Refine",
                "description": "Agent evaluates its own output and iteratively improves",
                "use_cases": ["Content generation", "Quality assurance", "Answer refinement"],
                "complexity": "intermediate",
            },
            {
                "id": "consensus_validation",
                "name": "Consensus Validation",
                "description": "Multiple agents validate critical outputs before finalization",
                "use_cases": ["High-stakes decisions", "Fact verification", "Quality control"],
                "complexity": "intermediate",
            },
            {
                "id": "human_in_the_loop",
                "name": "Human-in-the-Loop",
                "description": "Human approval checkpoints for critical decisions",
                "use_cases": ["Compliance", "Sensitive operations", "Quality gates"],
                "complexity": "basic",
            },
            {
                "id": "routing",
                "name": "Intent Routing",
                "description": "Route requests to specialized handlers based on detected intent",
                "use_cases": ["Customer support", "Multi-domain assistants", "Triage systems"],
                "complexity": "basic",
            },
            {
                "id": "error_recovery",
                "name": "Error Recovery",
                "description": "Graceful handling of failures with retries and fallbacks",
                "use_cases": ["Production systems", "Reliability-critical workflows"],
                "complexity": "intermediate",
            },
            {
                "id": "context_engineering",
                "name": "Context Engineering",
                "description": "Optimize context management for token efficiency and relevance",
                "use_cases": ["Long conversations", "Large document sets", "Cost optimization"],
                "complexity": "advanced",
            },
        ],
        "resources": [
            {
                "title": "The 2026 Guide to AI Agent Workflows",
                "url": "https://www.vellum.ai/blog/agentic-workflows-emerging-architectures-and-design-patterns",
            },
            {
                "title": "20 Agentic AI Workflow Patterns That Actually Work",
                "url": "https://skywork.ai/blog/agentic-ai-examples-workflow-patterns-2025/",
            },
            {
                "title": "LLM Orchestration Best Practices",
                "url": "https://orq.ai/blog/llm-orchestration",
            },
        ],
    }
