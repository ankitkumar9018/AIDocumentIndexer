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
from datetime import datetime
from typing import Optional, List, Dict, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
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
from backend.api.middleware.auth import get_user_context, UserContext
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
    logger.info(
        "Listing workflows",
        user_id=user.user_id,
        page=page,
        page_size=page_size,
        status=status,
    )

    service = get_workflow_service(
        session=db,
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
    )

    try:
        workflow = await service.create_workflow(
            name=request.name,
            description=request.description,
            trigger_type=request.trigger_type,
            trigger_config=request.trigger_config,
            nodes=[n.model_dump() for n in request.nodes] if request.nodes else None,
            edges=[e.model_dump() for e in request.edges] if request.edges else None,
            created_by_id=uuid.UUID(user.user_id),
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
    )
    service.user_id = uuid.UUID(user.user_id)

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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
    )

    try:
        execution = await engine.execute(
            workflow_id=workflow_id,
            trigger_type=WorkflowTriggerType.MANUAL.value,
            trigger_data=request.trigger_data,
            input_data=request.input_data,
            triggered_by_id=uuid.UUID(user.user_id),
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
            detail=f"Workflow execution failed: {str(e)}",
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
        organization_id=uuid.UUID(user.organization_id) if user.organization_id else None,
    )

    async def generate():
        try:
            async for event in engine.execute_stream(
                workflow_id=workflow_id,
                trigger_type=WorkflowTriggerType.MANUAL.value,
                trigger_data=request.trigger_data,
                input_data=request.input_data,
                triggered_by_id=uuid.UUID(user.user_id),
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error("Streaming execution failed", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

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
    db: AsyncSession = Depends(get_async_session),
    # Note: Webhook auth should be handled separately (e.g., signature verification)
):
    """
    Trigger a workflow via webhook.

    This endpoint can be called by external systems to trigger a workflow.
    The workflow must have trigger_type='webhook' and be active.
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
            detail=f"Workflow execution failed: {str(e)}",
        )
