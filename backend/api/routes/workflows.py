"""
AIDocumentIndexer - Workflow API Routes
=======================================

API endpoints for workflow management and execution:
- CRUD operations for workflows
- Node and edge management
- Workflow execution (manual, scheduled, webhook)
- Execution history and monitoring
"""

import uuid
import json
import hmac
from datetime import datetime
from typing import Optional, List, Dict, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.db.database import get_async_session
from backend.db.models import (
    Workflow,
    WorkflowNode,
    WorkflowEdge,
    WorkflowExecution,
    WorkflowNodeExecution,
    WorkflowStatus,
    WorkflowNodeType,
    WorkflowTriggerType,
)
from backend.api.middleware.auth import get_user_context, UserContext, get_org_id, get_user_uuid
from backend.services.workflow_engine import (
    get_workflow_service,
    get_execution_engine,
    WorkflowService,
    WorkflowExecutionEngine,
)
from backend.services.base import NotFoundException, ValidationException, ServiceException

logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class NodeCreate(BaseModel):
    """Node creation request."""

    temp_id: Optional[str] = Field(None, description="Temporary ID for edge references")
    node_type: str = Field(..., description="Type of node")
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    position_x: float = Field(default=0.0)
    position_y: float = Field(default=0.0)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EdgeCreate(BaseModel):
    """Edge creation request."""

    source_node_id: str = Field(..., description="Source node ID (can be temp_id)")
    target_node_id: str = Field(..., description="Target node ID (can be temp_id)")
    label: Optional[str] = None
    condition: Optional[str] = None
    edge_type: str = Field(default="default")


class WorkflowCreate(BaseModel):
    """Workflow creation request."""

    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    trigger_type: str = Field(default="manual")
    trigger_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    nodes: Optional[List[NodeCreate]] = Field(default_factory=list)
    edges: Optional[List[EdgeCreate]] = Field(default_factory=list)


class WorkflowUpdate(BaseModel):
    """Workflow update request."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    trigger_type: Optional[str] = None
    trigger_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class WorkflowNodesUpdate(BaseModel):
    """Update workflow nodes and edges."""

    nodes: List[NodeCreate]
    edges: List[EdgeCreate]


class NodeResponse(BaseModel):
    """Node response model."""

    id: str
    node_type: str
    name: str
    description: Optional[str]
    position_x: float
    position_y: float
    config: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class EdgeResponse(BaseModel):
    """Edge response model."""

    id: str
    source_node_id: str
    target_node_id: str
    label: Optional[str]
    condition: Optional[str]
    edge_type: str

    class Config:
        from_attributes = True


class WorkflowResponse(BaseModel):
    """Workflow response model."""

    id: str
    name: str
    description: Optional[str]
    is_active: bool
    is_draft: bool
    version: int
    trigger_type: str
    trigger_config: Optional[Dict[str, Any]]
    config: Optional[Dict[str, Any]]
    created_by_id: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    nodes: List[NodeResponse] = Field(default_factory=list)
    edges: List[EdgeResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class WorkflowListItem(BaseModel):
    """Workflow list item with stats."""

    id: str
    name: str
    description: Optional[str]
    is_active: bool
    is_draft: bool
    trigger_type: str
    version: int
    created_at: Optional[str]
    updated_at: Optional[str]
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    last_execution_at: Optional[str] = None


class WorkflowListResponse(BaseModel):
    """Paginated workflow list response."""

    workflows: List[WorkflowListItem]
    total: int
    page: int
    page_size: int
    has_more: bool


class ExecutionRequest(BaseModel):
    """Workflow execution request."""

    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    trigger_data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class NodeExecutionResponse(BaseModel):
    """Node execution response."""

    id: str
    node_id: str
    status: str
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_ms: Optional[int]


class ExecutionResponse(BaseModel):
    """Workflow execution response."""

    id: str
    workflow_id: str
    status: str
    trigger_type: str
    trigger_data: Optional[Dict[str, Any]]
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    error_node_id: Optional[str]
    retry_count: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    triggered_by_id: Optional[str]
    created_at: Optional[datetime]
    node_executions: List[NodeExecutionResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ExecutionListResponse(BaseModel):
    """Paginated execution list response."""

    executions: List[ExecutionResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


# =============================================================================
# Helper Functions
# =============================================================================


def workflow_to_response(workflow: Workflow) -> WorkflowResponse:
    """Convert workflow model to response."""
    return WorkflowResponse(
        id=str(workflow.id),
        name=workflow.name,
        description=workflow.description,
        is_active=workflow.is_active,
        is_draft=workflow.is_draft,
        version=workflow.version,
        trigger_type=workflow.trigger_type,
        trigger_config=workflow.trigger_config,
        config=workflow.config,
        created_by_id=str(workflow.created_by_id) if workflow.created_by_id else None,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        nodes=[
            NodeResponse(
                id=str(node.id),
                node_type=node.node_type,
                name=node.name,
                description=node.description,
                position_x=node.position_x,
                position_y=node.position_y,
                config=node.config,
            )
            for node in (workflow.nodes or [])
        ],
        edges=[
            EdgeResponse(
                id=str(edge.id),
                source_node_id=str(edge.source_node_id),
                target_node_id=str(edge.target_node_id),
                label=edge.label,
                condition=edge.condition,
                edge_type=edge.edge_type,
            )
            for edge in (workflow.edges or [])
        ],
    )


def execution_to_response(
    execution: WorkflowExecution, node_executions: Optional[List[WorkflowNodeExecution]] = None
) -> ExecutionResponse:
    """Convert execution model to response."""
    return ExecutionResponse(
        id=str(execution.id),
        workflow_id=str(execution.workflow_id),
        status=execution.status,
        trigger_type=execution.trigger_type,
        trigger_data=execution.trigger_data,
        input_data=execution.input_data,
        output_data=execution.output_data,
        error_message=execution.error_message,
        error_node_id=str(execution.error_node_id) if execution.error_node_id else None,
        retry_count=execution.retry_count,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        duration_ms=execution.duration_ms,
        triggered_by_id=str(execution.triggered_by_id) if execution.triggered_by_id else None,
        created_at=execution.created_at,
        node_executions=[
            NodeExecutionResponse(
                id=str(ne.id),
                node_id=str(ne.node_id),
                status=ne.status,
                input_data=ne.input_data,
                output_data=ne.output_data,
                error_message=ne.error_message,
                started_at=ne.started_at,
                completed_at=ne.completed_at,
                duration_ms=ne.duration_ms,
            )
            for ne in (node_executions or [])
        ],
    )


# =============================================================================
# Workflow CRUD Endpoints
# =============================================================================


@router.get("", response_model=WorkflowListResponse)
async def list_workflows(
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None, description="Filter by status: active, inactive, draft"),
    trigger_type: Optional[str] = Query(default=None, description="Filter by trigger type"),
    search: Optional[str] = Query(default=None, description="Search in name/description"),
):
    """List workflows with pagination and filtering."""
    try:
        logger.info(
            "Listing workflows",
            user_id=user.user_id,
            page=page,
            page_size=page_size,
            status=status,
        )

        service = get_workflow_service(
            session=db,
            organization_id=get_org_id(user),
        )

        workflows, total = await service.list_with_stats(
            page=page,
            page_size=page_size,
            status=status,
            trigger_type=trigger_type,
            search=search,
        )

        return WorkflowListResponse(
            workflows=[WorkflowListItem(**wf) for wf in workflows],
            total=total,
            page=page,
            page_size=page_size,
            has_more=(page * page_size) < total,
        )
    except Exception as e:
        logger.error("Failed to list workflows", error=str(e), user_id=user.user_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to load workflows"
        )


@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    request: WorkflowCreate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Create a new workflow."""
    logger.info("Creating workflow", user_id=user.user_id, name=request.name)

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    try:
        workflow = await service.create_workflow(
            name=request.name,
            description=request.description,
            trigger_type=request.trigger_type,
            trigger_config=request.trigger_config,
            nodes=[n.model_dump() for n in request.nodes] if request.nodes else None,
            edges=[e.model_dump() for e in request.edges] if request.edges else None,
            created_by_id=get_user_uuid(user),
        )

        return workflow_to_response(workflow)

    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except Exception as e:
        logger.error("Failed to create workflow", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow",
        )


# =============================================================================
# Node Type Reference Endpoint (MUST be before /{workflow_id} routes)
# =============================================================================


@router.get("/node-types")
async def get_node_types():
    """
    Get available node types and their configurations.

    Returns information about each node type that can be used
    in the workflow builder.
    """
    return {
        "node_types": [
            {
                "type": WorkflowNodeType.START.value,
                "name": "Start",
                "description": "Entry point of the workflow",
                "category": "control",
                "config_schema": {},
                "max_inputs": 0,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.END.value,
                "name": "End",
                "description": "Exit point of the workflow",
                "category": "control",
                "config_schema": {},
                "max_inputs": -1,  # Unlimited
                "max_outputs": 0,
            },
            {
                "type": WorkflowNodeType.ACTION.value,
                "name": "Action",
                "description": "Perform a predefined action",
                "category": "action",
                "config_schema": {
                    "action_type": {"type": "string", "required": True},
                    "params": {"type": "object"},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.CONDITION.value,
                "name": "Condition",
                "description": "Branch based on a condition",
                "category": "control",
                "config_schema": {
                    "expression": {"type": "string", "required": True},
                },
                "max_inputs": 1,
                "max_outputs": 2,  # true/false branches
            },
            {
                "type": WorkflowNodeType.LOOP.value,
                "name": "Loop",
                "description": "Iterate over an array",
                "category": "control",
                "config_schema": {
                    "array_path": {"type": "string", "required": True},
                    "item_var": {"type": "string", "default": "item"},
                    "batch_size": {"type": "integer"},
                },
                "max_inputs": 1,
                "max_outputs": 2,  # loop body, exit
            },
            {
                "type": WorkflowNodeType.CODE.value,
                "name": "Code",
                "description": "Execute custom code",
                "category": "action",
                "config_schema": {
                    "language": {"type": "string", "enum": ["javascript", "python"]},
                    "code": {"type": "string", "required": True},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.DELAY.value,
                "name": "Delay",
                "description": "Wait for a specified time",
                "category": "control",
                "config_schema": {
                    "seconds": {"type": "integer"},
                    "until": {"type": "string", "format": "datetime"},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.HTTP.value,
                "name": "HTTP Request",
                "description": "Make an HTTP request",
                "category": "action",
                "config_schema": {
                    "url": {"type": "string", "required": True},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "headers": {"type": "object"},
                    "body": {"type": "string"},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.NOTIFICATION.value,
                "name": "Notification",
                "description": "Send a notification",
                "category": "action",
                "config_schema": {
                    "channel": {"type": "string", "enum": ["email", "slack", "webhook"]},
                    "template": {"type": "string"},
                    "recipients": {"type": "array"},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.AGENT.value,
                "name": "AI Agent",
                "description": "Execute an AI agent task",
                "category": "ai",
                "config_schema": {
                    "agent_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "input_mapping": {"type": "object"},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.VOICE_AGENT.value,
                "name": "Voice Agent",
                "description": "Execute a voice-enabled AI agent with text-to-speech",
                "category": "ai",
                "config_schema": {
                    "agent_id": {"type": "string", "description": "ID of the agent to use"},
                    "prompt": {"type": "string", "description": "User prompt/task"},
                    "tts_provider": {"type": "string", "enum": ["openai", "elevenlabs", "cartesia", "edge"], "default": "openai"},
                    "voice_id": {"type": "string", "default": "alloy"},
                    "speed": {"type": "number", "default": 1.0, "min": 0.5, "max": 2.0},
                    "use_rag": {"type": "boolean", "default": True},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.CHAT_AGENT.value,
                "name": "Chat Agent",
                "description": "Execute a chat AI agent with knowledge base access",
                "category": "ai",
                "config_schema": {
                    "agent_id": {"type": "string", "description": "ID of the agent to use"},
                    "prompt": {"type": "string", "description": "User prompt/task"},
                    "knowledge_bases": {"type": "array", "description": "List of knowledge base IDs"},
                    "use_memory": {"type": "boolean", "default": True},
                    "memory_window": {"type": "integer", "default": 10},
                    "response_style": {"type": "string", "enum": ["formal", "casual", "technical", "friendly"], "default": "friendly"},
                    "enable_citations": {"type": "boolean", "default": True},
                },
                "max_inputs": 1,
                "max_outputs": 1,
            },
            {
                "type": WorkflowNodeType.HUMAN_APPROVAL.value,
                "name": "Human Approval",
                "description": "Wait for human approval",
                "category": "control",
                "config_schema": {
                    "approvers": {"type": "array"},
                    "timeout_hours": {"type": "integer", "default": 24},
                    "message": {"type": "string"},
                },
                "max_inputs": 1,
                "max_outputs": 2,  # approved/rejected
            },
        ],
        "trigger_types": [
            {
                "type": WorkflowTriggerType.MANUAL.value,
                "name": "Manual",
                "description": "Triggered manually by user",
            },
            {
                "type": WorkflowTriggerType.SCHEDULED.value,
                "name": "Scheduled",
                "description": "Triggered on a schedule (cron)",
                "config_schema": {
                    "cron": {"type": "string", "required": True},
                    "timezone": {"type": "string", "default": "UTC"},
                },
            },
            {
                "type": WorkflowTriggerType.WEBHOOK.value,
                "name": "Webhook",
                "description": "Triggered by external webhook",
                "config_schema": {
                    "secret": {"type": "string"},
                },
            },
            {
                "type": WorkflowTriggerType.FORM.value,
                "name": "Form",
                "description": "Triggered by form submission",
                "config_schema": {
                    "fields": {"type": "array"},
                },
            },
            {
                "type": WorkflowTriggerType.EVENT.value,
                "name": "Event",
                "description": "Triggered by system event",
                "config_schema": {
                    "event_type": {"type": "string"},
                    "filters": {"type": "object"},
                },
            },
        ],
    }


# =============================================================================
# Individual Workflow Endpoints
# =============================================================================


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Get a specific workflow with nodes and edges."""
    logger.info(
        "Getting workflow",
        user_id=user.user_id,
        organization_id=user.organization_id,
        workflow_id=str(workflow_id),
    )

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)

    if not workflow:
        logger.warning(
            "Workflow not found",
            workflow_id=str(workflow_id),
            organization_id=user.organization_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    return workflow_to_response(workflow)


@router.patch("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: uuid.UUID,
    request: WorkflowUpdate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Update workflow metadata."""
    logger.info("Updating workflow", user_id=user.user_id, workflow_id=str(workflow_id))

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    try:
        workflow = await service.update_workflow(
            workflow_id=workflow_id,
            name=request.name,
            description=request.description,
            trigger_type=request.trigger_type,
            trigger_config=request.trigger_config,
            is_active=request.is_active,
            config=request.config,
        )

        return workflow_to_response(workflow)

    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


@router.put("/{workflow_id}/nodes", response_model=WorkflowResponse)
async def update_workflow_nodes(
    workflow_id: uuid.UUID,
    request: WorkflowNodesUpdate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Update workflow nodes and edges (replaces all)."""
    logger.info(
        "Updating workflow nodes",
        user_id=user.user_id,
        workflow_id=str(workflow_id),
        node_count=len(request.nodes),
        edge_count=len(request.edges),
    )

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    try:
        workflow = await service.update_nodes_and_edges(
            workflow_id=workflow_id,
            nodes=[n.model_dump() for n in request.nodes],
            edges=[e.model_dump() for e in request.edges],
        )

        return workflow_to_response(workflow)

    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )
    except Exception as e:
        logger.error("Failed to update workflow nodes", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update workflow nodes",
        )


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Delete a workflow."""
    logger.info("Deleting workflow", user_id=user.user_id, workflow_id=str(workflow_id))

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    try:
        await service.delete(workflow_id)
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )


@router.post("/{workflow_id}/duplicate", response_model=WorkflowResponse)
async def duplicate_workflow(
    workflow_id: uuid.UUID,
    new_name: Optional[str] = Query(default=None),
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Create a copy of a workflow."""
    logger.info("Duplicating workflow", user_id=user.user_id, workflow_id=str(workflow_id))

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )
    service.user_id = get_user_uuid(user)

    try:
        workflow = await service.duplicate_workflow(workflow_id, new_name)
        return workflow_to_response(workflow)

    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )


@router.post("/{workflow_id}/publish", response_model=WorkflowResponse)
async def publish_workflow(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Publish a draft workflow (make it active)."""
    logger.info("Publishing workflow", user_id=user.user_id, workflow_id=str(workflow_id))

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    try:
        workflow = await service.publish_workflow(workflow_id)
        return workflow_to_response(workflow)

    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


# =============================================================================
# Workflow Execution Endpoints
# =============================================================================


@router.post("/{workflow_id}/execute", response_model=ExecutionResponse)
async def execute_workflow(
    workflow_id: uuid.UUID,
    request: ExecutionRequest,
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute a workflow manually.

    The execution runs in the background. Use the returned execution_id
    to poll for status or subscribe to WebSocket updates.
    """
    logger.info("Executing workflow", user_id=user.user_id, workflow_id=str(workflow_id))

    engine = get_execution_engine(
        session=db,
        organization_id=get_org_id(user),
    )

    try:
        execution = await engine.execute(
            workflow_id=workflow_id,
            trigger_type=WorkflowTriggerType.MANUAL.value,
            trigger_data=request.trigger_data,
            input_data=request.input_data,
            triggered_by_id=get_user_uuid(user),
        )

        return execution_to_response(execution)

    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except Exception as e:
        logger.error("Workflow execution failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workflow execution failed",
        )


@router.post("/{workflow_id}/execute/stream")
async def execute_workflow_stream(
    workflow_id: uuid.UUID,
    request: ExecutionRequest,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute a workflow with streaming progress updates.

    Returns a Server-Sent Events stream with:
    - started: Execution started
    - node_started: Node execution started
    - node_completed: Node execution completed
    - node_failed: Node execution failed
    - completed: Workflow completed
    - error: Workflow error
    """
    logger.info(
        "Executing workflow with streaming",
        user_id=user.user_id,
        workflow_id=str(workflow_id),
    )

    engine = get_execution_engine(
        session=db,
        organization_id=get_org_id(user),
    )

    async def generate():
        try:
            async for event in engine.execute_stream(
                workflow_id=workflow_id,
                trigger_type=WorkflowTriggerType.MANUAL.value,
                trigger_data=request.trigger_data,
                input_data=request.input_data,
                triggered_by_id=get_user_uuid(user),
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error("Streaming execution failed", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'error': 'Streaming execution failed'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{workflow_id}/executions", response_model=ExecutionListResponse)
async def list_workflow_executions(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
):
    """List executions for a workflow."""
    logger.info(
        "Listing workflow executions",
        user_id=user.user_id,
        workflow_id=str(workflow_id),
    )

    from sqlalchemy import func, desc

    # Verify workflow ownership
    wf_result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = wf_result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    org_id = get_org_id(user)
    if org_id and workflow.organization_id and str(workflow.organization_id) != str(org_id):
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Build query
    query = select(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id)

    if status:
        query = query.where(WorkflowExecution.status == status)

    # Count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate
    query = query.order_by(desc(WorkflowExecution.created_at))
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    result = await db.execute(query)
    executions = list(result.scalars().all())

    return ExecutionListResponse(
        executions=[execution_to_response(e) for e in executions],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(
    execution_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Get execution details including node executions."""
    logger.info("Getting execution", user_id=user.user_id, execution_id=str(execution_id))

    # Get execution
    result = await db.execute(
        select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
    )
    execution = result.scalar_one_or_none()

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found",
        )

    # Verify ownership via parent workflow
    org_id = get_org_id(user)
    if org_id and execution.workflow_id:
        wf_result = await db.execute(select(Workflow).where(Workflow.id == execution.workflow_id))
        workflow = wf_result.scalar_one_or_none()
        if workflow and workflow.organization_id and str(workflow.organization_id) != str(org_id):
            raise HTTPException(status_code=404, detail="Execution not found")

    # Get node executions
    node_result = await db.execute(
        select(WorkflowNodeExecution)
        .where(WorkflowNodeExecution.execution_id == execution_id)
        .order_by(WorkflowNodeExecution.started_at)
    )
    node_executions = list(node_result.scalars().all())

    return execution_to_response(execution, node_executions)


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(
    execution_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Cancel a running execution."""
    logger.info("Cancelling execution", user_id=user.user_id, execution_id=str(execution_id))

    result = await db.execute(
        select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
    )
    execution = result.scalar_one_or_none()

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found",
        )

    # Verify ownership via parent workflow
    org_id = get_org_id(user)
    if org_id and execution.workflow_id:
        wf_result = await db.execute(select(Workflow).where(Workflow.id == execution.workflow_id))
        workflow = wf_result.scalar_one_or_none()
        if workflow and workflow.organization_id and str(workflow.organization_id) != str(org_id):
            raise HTTPException(status_code=404, detail="Execution not found")

    if execution.status not in [WorkflowStatus.PENDING.value, WorkflowStatus.RUNNING.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel execution in status: {execution.status}",
        )

    execution.status = WorkflowStatus.CANCELLED.value
    execution.completed_at = datetime.utcnow()
    await db.commit()

    return {"message": "Execution cancelled", "execution_id": str(execution_id)}


# =============================================================================
# Webhook Trigger Endpoint
# =============================================================================


@router.post("/webhook/{workflow_id}")
async def webhook_trigger(
    workflow_id: uuid.UUID,
    payload: Dict[str, Any],
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Trigger a workflow via webhook.

    This endpoint can be called by external systems to trigger a workflow.
    The workflow must have trigger_type='webhook' and be active.
    Requires X-Webhook-Secret header matching the workflow's configured webhook_secret.
    """
    logger.info("Webhook trigger received", workflow_id=str(workflow_id))

    # Get workflow
    service = get_workflow_service(session=db)
    workflow = await service.get_by_id(workflow_id)

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    # Verify webhook secret
    webhook_secret = (workflow.config or {}).get("webhook_secret")
    if not webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Webhook secret not configured for this workflow",
        )

    provided_secret = request.headers.get("X-Webhook-Secret", "")
    if not hmac.compare_digest(provided_secret, webhook_secret):
        logger.warning("Invalid webhook secret", workflow_id=str(workflow_id))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret",
        )

    if not workflow.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow is not active",
        )

    if workflow.trigger_type != WorkflowTriggerType.WEBHOOK.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow is not configured for webhook triggers",
        )

    # Execute workflow
    engine = get_execution_engine(session=db, organization_id=workflow.organization_id)

    try:
        execution = await engine.execute(
            workflow_id=workflow_id,
            trigger_type=WorkflowTriggerType.WEBHOOK.value,
            trigger_data={"payload": payload},
            input_data=payload,
        )

        return {
            "message": "Workflow triggered",
            "execution_id": str(execution.id),
            "status": execution.status,
        }

    except Exception as e:
        logger.error("Webhook trigger failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workflow execution failed",
        )


# =============================================================================
# Publishing Endpoints
# =============================================================================


class WorkflowInputField(BaseModel):
    """Input field definition for published workflow."""
    name: str
    type: str = "text"  # text, number, textarea, select, checkbox, date
    label: str
    description: Optional[str] = None
    required: bool = True
    default: Optional[Any] = None
    options: Optional[List[str]] = None  # For select type


class WorkflowPublishRequest(BaseModel):
    """Request to publish a workflow."""
    input_schema: List[WorkflowInputField] = Field(default_factory=list, description="Input fields for public form")
    rate_limit: int = Field(default=100, description="Max executions per minute")
    allowed_domains: List[str] = Field(default=["*"], description="Allowed origin domains")
    require_api_key: bool = Field(default=False, description="Require API key for access")
    custom_slug: Optional[str] = Field(None, description="Custom URL slug")
    branding: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom branding")


class WorkflowPublishResponse(BaseModel):
    """Response from publishing a workflow."""
    workflow_id: str
    public_slug: str
    public_url: str
    embed_code: str
    is_published: bool


def _generate_workflow_slug(name: str) -> str:
    """Generate a URL-friendly slug from name."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return f"{slug}-{uuid.uuid4().hex[:8]}"


@router.post("/{workflow_id}/deploy", response_model=WorkflowPublishResponse)
async def publish_workflow_for_external(
    workflow_id: uuid.UUID,
    request: WorkflowPublishRequest,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Publish a workflow for external access via public URL.

    Once published, the workflow can be accessed without authentication
    at /api/v1/public/workflows/{public_slug}.

    The input_schema defines what form fields will be shown to users.
    """
    from backend.core.config import settings

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    # Verify ownership
    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only publish your own workflows",
        )

    # Workflow must be active and not draft
    if workflow.is_draft:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot publish a draft workflow. Please publish it first.",
        )

    # Generate or use custom slug
    if request.custom_slug:
        # Check if slug is already taken
        existing = await db.execute(
            select(Workflow).where(
                Workflow.public_slug == request.custom_slug,
                Workflow.id != workflow.id,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Slug '{request.custom_slug}' is already taken",
            )
        public_slug = request.custom_slug
    else:
        public_slug = _generate_workflow_slug(workflow.name)

    # Update workflow
    workflow.is_published = True
    workflow.public_slug = public_slug
    workflow.publish_config = {
        "input_schema": [f.model_dump() for f in request.input_schema],
        "rate_limit": request.rate_limit,
        "allowed_domains": request.allowed_domains,
        "require_api_key": request.require_api_key,
        "branding": request.branding,
    }

    await db.commit()
    await db.refresh(workflow)

    # Generate URLs
    base_url = settings.server.frontend_url or "http://localhost:3000"
    public_url = f"{base_url}/w/{public_slug}"

    # Generate embed code
    embed_code = f'''<iframe
  src="{public_url}/embed"
  width="100%"
  height="600"
  frameborder="0"
  allow="clipboard-write"
></iframe>'''

    logger.info(
        "Workflow published for external access",
        workflow_id=str(workflow.id),
        public_slug=public_slug,
        user_id=user.user_id,
    )

    return WorkflowPublishResponse(
        workflow_id=str(workflow.id),
        public_slug=public_slug,
        public_url=public_url,
        embed_code=embed_code,
        is_published=True,
    )


@router.post("/{workflow_id}/undeploy")
async def unpublish_workflow_external(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Unpublish a workflow, removing public access."""
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only unpublish your own workflows",
        )

    workflow.is_published = False
    # Keep the slug for re-publishing

    await db.commit()

    return {"message": f"Workflow '{workflow.name}' unpublished successfully"}


@router.get("/{workflow_id}/deploy-status")
async def get_workflow_deploy_status(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Get the deployment status and public URL for a workflow."""
    from backend.core.config import settings

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    base_url = settings.server.frontend_url or "http://localhost:3000"

    return {
        "workflow_id": str(workflow.id),
        "is_published": workflow.is_published,
        "public_slug": workflow.public_slug,
        "public_url": f"{base_url}/w/{workflow.public_slug}" if workflow.public_slug else None,
        "publish_config": workflow.publish_config,
    }


# =============================================================================
# Scheduling Endpoints
# =============================================================================


class ScheduleRequest(BaseModel):
    """Request to schedule a workflow."""
    cron_expression: str = Field(..., description="Cron expression (e.g., '0 9 * * *')")
    timezone: str = Field(default="UTC", description="Timezone for the schedule")


class ScheduleResponse(BaseModel):
    """Workflow schedule information."""
    workflow_id: str
    cron: str
    timezone: str
    next_run: Optional[str] = None
    is_scheduled: bool


@router.post("/{workflow_id}/schedule", response_model=ScheduleResponse)
async def schedule_workflow(
    workflow_id: uuid.UUID,
    request: ScheduleRequest,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Schedule a workflow for periodic execution.

    The workflow will be executed automatically based on the cron expression.

    Cron expression format: minute hour day_of_month month day_of_week
    Examples:
    - "0 9 * * *" - Every day at 9:00 AM
    - "0 9 * * 1-5" - Weekdays at 9:00 AM
    - "*/15 * * * *" - Every 15 minutes
    - "0 0 1 * *" - First day of each month at midnight
    """
    from backend.services.workflow_scheduler import (
        get_workflow_scheduler,
        validate_cron_expression,
        get_next_run_time,
    )

    logger.info(
        "Scheduling workflow",
        user_id=user.user_id,
        workflow_id=str(workflow_id),
        cron=request.cron_expression,
    )

    # Validate cron expression
    if not validate_cron_expression(request.cron_expression):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid cron expression. Use format: minute hour day_of_month month day_of_week",
        )

    # Get workflow and verify ownership
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only schedule your own workflows",
        )

    if workflow.is_draft:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot schedule a draft workflow. Please publish it first.",
        )

    # Update workflow trigger config
    workflow.trigger_type = WorkflowTriggerType.SCHEDULED.value
    workflow.trigger_config = {
        "cron": request.cron_expression,
        "timezone": request.timezone,
    }

    await db.commit()

    # Register with scheduler
    scheduler = get_workflow_scheduler()
    await scheduler.schedule_workflow(
        workflow_id=str(workflow_id),
        cron_expression=request.cron_expression,
        timezone=request.timezone,
    )

    # Calculate next run time
    next_run = await get_next_run_time(request.cron_expression, request.timezone)

    return ScheduleResponse(
        workflow_id=str(workflow_id),
        cron=request.cron_expression,
        timezone=request.timezone,
        next_run=next_run.isoformat() if next_run else None,
        is_scheduled=True,
    )


@router.delete("/{workflow_id}/schedule")
async def unschedule_workflow(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Remove a workflow from the schedule."""
    from backend.services.workflow_scheduler import get_workflow_scheduler

    logger.info(
        "Unscheduling workflow",
        user_id=user.user_id,
        workflow_id=str(workflow_id),
    )

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only unschedule your own workflows",
        )

    # Update workflow trigger type back to manual
    workflow.trigger_type = WorkflowTriggerType.MANUAL.value
    workflow.trigger_config = {}

    await db.commit()

    # Remove from scheduler
    scheduler = get_workflow_scheduler()
    await scheduler.unschedule_workflow(str(workflow_id))

    return {"message": f"Workflow '{workflow.name}' unscheduled successfully"}


@router.get("/{workflow_id}/schedule", response_model=ScheduleResponse)
async def get_workflow_schedule(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Get the current schedule for a workflow."""
    from backend.services.workflow_scheduler import get_next_run_time

    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    is_scheduled = workflow.trigger_type == WorkflowTriggerType.SCHEDULED.value
    trigger_config = workflow.trigger_config or {}
    cron = trigger_config.get("cron", "")
    timezone = trigger_config.get("timezone", "UTC")

    next_run = None
    if is_scheduled and cron:
        next_run = await get_next_run_time(cron, timezone)

    return ScheduleResponse(
        workflow_id=str(workflow_id),
        cron=cron,
        timezone=timezone,
        next_run=next_run.isoformat() if next_run else None,
        is_scheduled=is_scheduled,
    )


@router.get("/schedules/upcoming")
async def list_upcoming_scheduled_executions(
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    List upcoming scheduled workflow executions.

    Returns the next scheduled runs across all scheduled workflows.
    """
    from backend.services.workflow_scheduler import get_next_run_time

    # Get all scheduled workflows for this user/org
    query = select(Workflow).where(
        Workflow.trigger_type == WorkflowTriggerType.SCHEDULED.value,
        Workflow.is_active == True,
        Workflow.is_draft == False,
    )

    org_id = get_org_id(user)
    user_uuid = get_user_uuid(user)
    if org_id:
        from sqlalchemy import or_
        query = query.where(
            or_(
                Workflow.organization_id == org_id,
                Workflow.created_by_id == user_uuid,
            )
        )

    result = await db.execute(query)
    workflows = list(result.scalars().all())

    upcoming = []
    for workflow in workflows:
        trigger_config = workflow.trigger_config or {}
        cron = trigger_config.get("cron")
        timezone = trigger_config.get("timezone", "UTC")

        if cron:
            next_run = await get_next_run_time(cron, timezone)
            if next_run:
                upcoming.append({
                    "workflow_id": str(workflow.id),
                    "workflow_name": workflow.name,
                    "cron": cron,
                    "timezone": timezone,
                    "next_run": next_run.isoformat(),
                })

    # Sort by next run time
    upcoming.sort(key=lambda x: x["next_run"])

    return {
        "upcoming": upcoming[:limit],
        "total": len(upcoming),
    }


# =============================================================================
# Sharing Endpoints
# =============================================================================


class ShareWorkflowRequest(BaseModel):
    """Request to share a workflow."""
    permission_level: str = Field(
        default="viewer",
        description="Permission level: viewer, editor, or executor",
    )
    password: Optional[str] = Field(None, description="Optional password protection")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days (null = never)")
    max_uses: Optional[int] = Field(None, description="Max uses (null = unlimited)")
    allow_copy: bool = Field(default=False, description="Allow recipient to copy the workflow")


class ShareLinkResponse(BaseModel):
    """Share link information."""
    share_id: str
    token: str
    share_url: str
    permission_level: str
    expires_at: Optional[str] = None
    max_uses: Optional[int] = None
    use_count: int = 0
    is_active: bool = True
    created_at: str


@router.post("/{workflow_id}/share", response_model=ShareLinkResponse)
async def share_workflow(
    workflow_id: uuid.UUID,
    request: ShareWorkflowRequest,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a share link for a workflow.

    Share links allow others to view, execute, or edit a workflow
    without needing to be added to the organization.

    Permission levels:
    - viewer: Can view workflow definition
    - executor: Can view and execute the workflow
    - editor: Can view, execute, and make a copy
    """
    import secrets
    import hashlib
    from backend.core.config import settings

    logger.info(
        "Creating share link for workflow",
        user_id=user.user_id,
        workflow_id=str(workflow_id),
    )

    # Get workflow and verify ownership
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only share your own workflows",
        )

    # Generate share token
    token = secrets.token_urlsafe(32)

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

    # Hash password if provided (using scrypt with random salt)
    password_hash = None
    if request.password:
        import os
        salt = os.urandom(16)
        hash_bytes = hashlib.scrypt(request.password.encode(), salt=salt, n=16384, r=8, p=1)
        password_hash = f"scrypt:{salt.hex()}:{hash_bytes.hex()}"

    # Create share link record
    share_id = uuid.uuid4()

    # Use raw SQL since ShareLink model may not exist yet
    from sqlalchemy import text

    await db.execute(
        text("""
            INSERT INTO share_links (
                id, organization_id, resource_id, resource_type, token,
                permission_level, password_hash, expires_at, max_uses,
                use_count, allow_download, require_login, created_by_id,
                is_active, created_at, updated_at
            ) VALUES (
                :id, :org_id, :resource_id, 'workflow', :token,
                :permission, :password_hash, :expires_at, :max_uses,
                0, :allow_copy, false, :created_by,
                true, :now, :now
            )
        """),
        {
            "id": str(share_id),
            "org_id": user.organization_id,
            "resource_id": str(workflow_id),
            "token": token,
            "permission": request.permission_level,
            "password_hash": password_hash,
            "expires_at": expires_at,
            "max_uses": request.max_uses,
            "allow_copy": request.allow_copy,
            "created_by": user.user_id,
            "now": datetime.utcnow(),
        },
    )
    await db.commit()

    # Generate share URL
    base_url = settings.server.frontend_url or "http://localhost:3000"
    share_url = f"{base_url}/shared/workflow/{token}"

    return ShareLinkResponse(
        share_id=str(share_id),
        token=token,
        share_url=share_url,
        permission_level=request.permission_level,
        expires_at=expires_at.isoformat() if expires_at else None,
        max_uses=request.max_uses,
        use_count=0,
        is_active=True,
        created_at=datetime.utcnow().isoformat(),
    )


@router.get("/{workflow_id}/shares")
async def list_workflow_shares(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """List all share links for a workflow."""
    from sqlalchemy import text
    from backend.core.config import settings

    # Verify ownership
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view shares for your own workflows",
        )

    # Get share links
    result = await db.execute(
        text("""
            SELECT id, token, permission_level, expires_at, max_uses,
                   use_count, is_active, created_at
            FROM share_links
            WHERE resource_id = :workflow_id
            AND resource_type = 'workflow'
            ORDER BY created_at DESC
        """),
        {"workflow_id": str(workflow_id)},
    )
    shares = result.fetchall()

    base_url = settings.server.frontend_url or "http://localhost:3000"

    return {
        "shares": [
            {
                "share_id": str(s[0]),
                "token": s[1],
                "share_url": f"{base_url}/shared/workflow/{s[1]}",
                "permission_level": s[2],
                "expires_at": s[3].isoformat() if s[3] else None,
                "max_uses": s[4],
                "use_count": s[5],
                "is_active": s[6],
                "created_at": s[7].isoformat() if s[7] else None,
            }
            for s in shares
        ],
        "total": len(shares),
    }


@router.delete("/{workflow_id}/shares/{share_id}")
async def revoke_workflow_share(
    workflow_id: uuid.UUID,
    share_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """Revoke a share link."""
    from sqlalchemy import text

    # Verify ownership
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only revoke shares for your own workflows",
        )

    # Deactivate share link
    result = await db.execute(
        text("""
            UPDATE share_links
            SET is_active = false, updated_at = :now
            WHERE id = :share_id
            AND resource_id = :workflow_id
            AND resource_type = 'workflow'
        """),
        {
            "share_id": str(share_id),
            "workflow_id": str(workflow_id),
            "now": datetime.utcnow(),
        },
    )
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found",
        )

    return {"message": "Share link revoked"}


# =============================================================================
# Versioning Endpoints
# =============================================================================


class WorkflowVersionResponse(BaseModel):
    """Workflow version information."""
    version: int
    created_at: str
    created_by: Optional[str] = None
    change_summary: Optional[str] = None
    node_count: int
    edge_count: int


@router.get("/{workflow_id}/versions")
async def list_workflow_versions(
    workflow_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List version history for a workflow.

    Each time a workflow is updated, a new version is created.
    """
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    # For now, return current version info
    # TODO: Implement full version history table
    versions = [
        WorkflowVersionResponse(
            version=workflow.version,
            created_at=workflow.updated_at.isoformat() if workflow.updated_at else workflow.created_at.isoformat(),
            created_by=str(workflow.created_by_id) if workflow.created_by_id else None,
            change_summary="Current version",
            node_count=len(workflow.nodes) if workflow.nodes else 0,
            edge_count=len(workflow.edges) if workflow.edges else 0,
        )
    ]

    return {
        "versions": [v.model_dump() for v in versions],
        "current_version": workflow.version,
        "total": len(versions),
    }


@router.post("/{workflow_id}/versions/{version}/restore")
async def restore_workflow_version(
    workflow_id: uuid.UUID,
    version: int,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Restore a workflow to a previous version.

    Creates a new version with the contents of the specified version.
    """
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only restore your own workflows",
        )

    # TODO: Implement version restore from history
    # For now, just bump version number
    if version != workflow.version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Version history not yet implemented. Only current version available.",
        )

    return {
        "message": f"Workflow restored to version {version}",
        "new_version": workflow.version,
    }


# =============================================================================
# Form Trigger Endpoints
# =============================================================================


class FormTriggerConfig(BaseModel):
    """Configuration for form-triggered workflows."""
    input_schema: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Input fields for the form",
    )
    success_message: str = Field(
        default="Form submitted successfully!",
        description="Message shown after submission",
    )
    redirect_url: Optional[str] = Field(
        None,
        description="URL to redirect after submission",
    )
    collect_metadata: bool = Field(
        default=True,
        description="Collect submission metadata (IP, timestamp)",
    )


@router.post("/{workflow_id}/form-trigger")
async def configure_form_trigger(
    workflow_id: uuid.UUID,
    config: FormTriggerConfig,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Configure a workflow to be triggered by form submission.

    Form-triggered workflows expose a form URL that anyone can submit.
    Each submission triggers a new workflow execution.
    """
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only configure your own workflows",
        )

    # Update trigger configuration
    workflow.trigger_type = WorkflowTriggerType.FORM.value
    workflow.trigger_config = {
        "input_schema": config.input_schema,
        "success_message": config.success_message,
        "redirect_url": config.redirect_url,
        "collect_metadata": config.collect_metadata,
    }

    await db.commit()

    # Generate form URL
    from backend.core.config import settings
    base_url = settings.server.frontend_url or "http://localhost:3000"

    # If workflow is published, use public slug, otherwise use form endpoint
    form_url = f"{base_url}/w/{workflow.public_slug}" if workflow.is_published and workflow.public_slug else None

    return {
        "workflow_id": str(workflow_id),
        "trigger_type": "form",
        "form_url": form_url,
        "input_schema": config.input_schema,
        "message": "Form trigger configured. Publish the workflow to get a public form URL.",
    }


@router.post("/{workflow_id}/form-submit")
async def submit_form_trigger(
    workflow_id: uuid.UUID,
    form_data: Dict[str, Any],
    db: AsyncSession = Depends(get_async_session),
):
    """
    Submit a form to trigger a workflow execution.

    This endpoint is called when a user submits a form configured
    for a form-triggered workflow.
    """
    from fastapi import Request

    # Get workflow
    service = get_workflow_service(session=db)
    workflow = await service.get_by_id(workflow_id)

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if not workflow.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow is not active",
        )

    if workflow.trigger_type != WorkflowTriggerType.FORM.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow is not configured for form triggers",
        )

    # Execute workflow
    engine = get_execution_engine(session=db, organization_id=workflow.organization_id)

    trigger_config = workflow.trigger_config or {}

    try:
        execution = await engine.execute(
            workflow_id=workflow_id,
            trigger_type=WorkflowTriggerType.FORM.value,
            trigger_data={
                "form_data": form_data,
                "submitted_at": datetime.utcnow().isoformat(),
            },
            input_data=form_data,
            triggered_by_id=workflow.created_by_id,
        )

        # Validate redirect URL — only allow relative paths (same-origin)
        redirect_url = trigger_config.get("redirect_url")
        if redirect_url:
            from urllib.parse import urlparse
            parsed = urlparse(redirect_url)
            # Block absolute URLs to prevent open redirect to external domains
            if parsed.scheme or parsed.netloc:
                redirect_url = None
            elif not redirect_url.startswith("/"):
                redirect_url = None

        return {
            "message": trigger_config.get("success_message", "Form submitted successfully!"),
            "execution_id": str(execution.id),
            "status": execution.status,
            "redirect_url": redirect_url,
        }

    except Exception as e:
        logger.error("Form trigger failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Form submission failed",
        )


# =============================================================================
# Event Trigger Endpoints
# =============================================================================


class EventTriggerConfig(BaseModel):
    """Configuration for event-triggered workflows."""
    event_types: List[str] = Field(
        ...,
        description="Event types that trigger this workflow",
    )
    filter_conditions: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional filter conditions for events",
    )


@router.post("/{workflow_id}/event-trigger")
async def configure_event_trigger(
    workflow_id: uuid.UUID,
    config: EventTriggerConfig,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Configure a workflow to be triggered by system events.

    Event types include:
    - document.uploaded: When a document is uploaded
    - document.processed: When processing completes
    - document.deleted: When a document is deleted
    - connector.sync_completed: When connector sync finishes
    - skill.executed: When a skill is executed
    - workflow.completed: When another workflow completes
    - workflow.failed: When another workflow fails
    """
    service = get_workflow_service(
        session=db,
        organization_id=get_org_id(user),
    )

    workflow = await service.get_by_id(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    if workflow.created_by_id and str(workflow.created_by_id) != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only configure your own workflows",
        )

    # Validate event types
    valid_events = {
        "document.uploaded",
        "document.processed",
        "document.deleted",
        "connector.sync_completed",
        "skill.executed",
        "workflow.completed",
        "workflow.failed",
        "chat.message",
        "user.login",
        "user.logout",
    }

    invalid_events = set(config.event_types) - valid_events
    if invalid_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event types: {', '.join(invalid_events)}. Valid types: {', '.join(sorted(valid_events))}",
        )

    # Update trigger configuration
    workflow.trigger_type = WorkflowTriggerType.EVENT.value
    workflow.trigger_config = {
        "event_types": config.event_types,
        "filter_conditions": config.filter_conditions,
    }

    await db.commit()

    return {
        "workflow_id": str(workflow_id),
        "trigger_type": "event",
        "event_types": config.event_types,
        "filter_conditions": config.filter_conditions,
        "message": "Event trigger configured successfully",
    }


@router.get("/event-triggers")
async def list_event_triggers(
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """List all event-triggered workflows for the user."""
    from sqlalchemy import or_

    query = select(Workflow).where(
        Workflow.trigger_type == WorkflowTriggerType.EVENT.value,
        Workflow.is_active == True,
    )

    org_id = get_org_id(user)
    user_uuid = get_user_uuid(user)
    if org_id:
        query = query.where(
            or_(
                Workflow.organization_id == org_id,
                Workflow.created_by_id == user_uuid,
            )
        )

    result = await db.execute(query)
    workflows = list(result.scalars().all())

    return {
        "triggers": [
            {
                "workflow_id": str(w.id),
                "workflow_name": w.name,
                "event_types": (w.trigger_config or {}).get("event_types", []),
                "filter_conditions": (w.trigger_config or {}).get("filter_conditions"),
                "is_active": w.is_active,
            }
            for w in workflows
        ],
        "total": len(workflows),
    }


async def trigger_event_workflows(
    event_type: str,
    event_data: Dict[str, Any],
    organization_id: Optional[uuid.UUID] = None,
    db: Optional[AsyncSession] = None,
) -> List[Dict[str, Any]]:
    """
    Trigger all workflows that listen for a specific event.

    This is called internally when system events occur.

    Args:
        event_type: Type of event (e.g., "document.uploaded")
        event_data: Event payload data
        organization_id: Limit to workflows in this organization
        db: Database session

    Returns:
        List of triggered execution info
    """
    from backend.db.database import get_async_session_context

    triggered = []

    async def _trigger(session: AsyncSession):
        nonlocal triggered

        query = select(Workflow).where(
            Workflow.trigger_type == WorkflowTriggerType.EVENT.value,
            Workflow.is_active == True,
            Workflow.is_draft == False,
        )

        if organization_id:
            query = query.where(Workflow.organization_id == organization_id)

        result = await session.execute(query)
        workflows = list(result.scalars().all())

        for workflow in workflows:
            trigger_config = workflow.trigger_config or {}
            event_types = trigger_config.get("event_types", [])

            if event_type not in event_types:
                continue

            # Check filter conditions
            filter_conditions = trigger_config.get("filter_conditions")
            if filter_conditions:
                # Simple key-value matching for now
                matches = all(
                    event_data.get(k) == v
                    for k, v in filter_conditions.items()
                )
                if not matches:
                    continue

            # Execute workflow
            try:
                engine = get_execution_engine(
                    session=session,
                    organization_id=workflow.organization_id,
                )

                execution = await engine.execute(
                    workflow_id=workflow.id,
                    trigger_type=WorkflowTriggerType.EVENT.value,
                    trigger_data={
                        "event_type": event_type,
                        "event_data": event_data,
                        "triggered_at": datetime.utcnow().isoformat(),
                    },
                    input_data=event_data,
                    triggered_by_id=workflow.created_by_id,
                )

                triggered.append({
                    "workflow_id": str(workflow.id),
                    "workflow_name": workflow.name,
                    "execution_id": str(execution.id),
                    "status": execution.status,
                })

                logger.info(
                    "Event triggered workflow execution",
                    event_type=event_type,
                    workflow_id=str(workflow.id),
                    execution_id=str(execution.id),
                )

            except Exception as e:
                logger.error(
                    "Failed to trigger workflow for event",
                    event_type=event_type,
                    workflow_id=str(workflow.id),
                    error=str(e),
                )

    if db:
        await _trigger(db)
    else:
        async with get_async_session_context() as session:
            await _trigger(session)

    return triggered
