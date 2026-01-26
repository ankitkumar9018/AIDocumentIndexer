"""
AIDocumentIndexer - Agent API Routes
====================================

Endpoints for multi-agent orchestration system.

Features:
- Request execution through agents
- Execution mode management (agent/chat)
- Plan status and control
- Agent metrics and status
- Prompt optimization management (admin)
"""

import json
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from fastapi.responses import StreamingResponse

from backend.api.middleware.auth import AuthenticatedUser
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import structlog

from backend.db.database import get_async_session as get_db
from backend.db.models import (
    AgentDefinition,
    AgentPromptVersion,
    AgentTrajectory,
    AgentExecutionPlan,
    PromptOptimizationJob,
    ExecutionModePreference,
    ExecutionMode,
)
from backend.services.agent_orchestrator import AgentOrchestrator, create_orchestrator
from backend.services.mode_router import ModeRouter, ComplexityDetector
from backend.services.agents.trajectory_collector import TrajectoryCollector
from backend.services.agents.cost_estimator import AgentCostEstimator
from backend.services.prompt_optimization.prompt_version_manager import PromptVersionManager

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class AgentExecutionRequest(BaseModel):
    """Request for agent execution."""
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    mode: Optional[str] = Field(None, pattern="^(agent|chat)$")
    context: Optional[Dict[str, Any]] = None


class ApproveExecutionRequest(BaseModel):
    """Request to approve pending execution."""
    plan_id: str


class SetModeRequest(BaseModel):
    """Request to set execution mode."""
    mode: str = Field(..., pattern="^(agent|chat)$")


class UpdatePreferencesRequest(BaseModel):
    """Request to update mode preferences."""
    default_mode: Optional[str] = Field(None, pattern="^(agent|chat|general)$")
    agent_mode_enabled: Optional[bool] = None
    auto_detect_complexity: Optional[bool] = None
    show_cost_estimation: Optional[bool] = None
    require_approval_above_usd: Optional[float] = Field(None, ge=0)
    general_chat_enabled: Optional[bool] = None
    fallback_to_general: Optional[bool] = None


class UpdateAgentConfigRequest(BaseModel):
    """Request to update agent configuration."""
    default_provider_id: Optional[str] = None
    default_model: Optional[str] = None
    default_temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    is_active: Optional[bool] = None


class TTSConfigRequest(BaseModel):
    """TTS configuration for voice agents."""
    provider: str = Field("openai", description="TTS provider: openai, elevenlabs, cartesia, edge")
    voice_id: str = Field("alloy", description="Voice ID for the TTS provider")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""
    name: str = Field(..., min_length=1, max_length=100)
    agent_type: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    default_temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(2048, ge=256, le=128000)
    settings: Optional[Dict[str, Any]] = None
    default_provider_id: Optional[str] = None
    default_model: Optional[str] = None
    # Voice/Chat agent specific fields
    agent_mode: Optional[str] = Field(None, description="Agent mode: voice, chat, or hybrid")
    tts_config: Optional[TTSConfigRequest] = Field(None, description="TTS configuration for voice agents")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    knowledge_bases: Optional[List[str]] = Field(None, description="List of knowledge base IDs")


class UpdateAgentRequest(BaseModel):
    """Request to fully update an agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    default_temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=256, le=128000)
    settings: Optional[Dict[str, Any]] = None
    default_provider_id: Optional[str] = None
    default_model: Optional[str] = None
    is_active: Optional[bool] = None
    # Voice/Chat agent specific fields
    agent_mode: Optional[str] = Field(None, description="Agent mode: voice, chat, or hybrid")
    tts_config: Optional[TTSConfigRequest] = Field(None, description="TTS configuration for voice agents")
    knowledge_bases: Optional[List[str]] = Field(None, description="List of knowledge base IDs")
    # Prompt fields - when provided, creates a new prompt version
    system_prompt: Optional[str] = Field(None, description="System prompt - creates new prompt version if provided")
    task_prompt_template: Optional[str] = Field(None, description="Task template - creates new prompt version if provided")


class CreatePromptVersionRequest(BaseModel):
    """Request to create a prompt version."""
    system_prompt: str
    task_prompt_template: str
    change_reason: str
    few_shot_examples: Optional[List[Dict[str, str]]] = None
    output_schema: Optional[Dict[str, Any]] = None


class AgentStatusResponse(BaseModel):
    """Agent status information."""
    id: str
    name: str
    agent_type: str
    model: str
    status: str
    success_rate: Optional[float] = None


class PlanStatusResponse(BaseModel):
    """Execution plan status."""
    plan_id: str
    status: str
    user_request: str
    step_count: int
    current_step: int
    completed_steps: int
    estimated_cost_usd: Optional[float]
    actual_cost_usd: Optional[float]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class ModePreferencesResponse(BaseModel):
    """User mode preferences."""
    default_mode: str
    agent_mode_enabled: bool
    auto_detect_complexity: bool
    show_cost_estimation: bool
    require_approval_above_usd: float
    general_chat_enabled: bool = True
    fallback_to_general: bool = True
    effective_mode: str


# =============================================================================
# Service Dependencies
# =============================================================================

_orchestrator: Optional[AgentOrchestrator] = None


async def get_orchestrator(db: AsyncSession = Depends(get_db)) -> AgentOrchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = await create_orchestrator(db)
    return _orchestrator


async def get_mode_router(db: AsyncSession = Depends(get_db)) -> ModeRouter:
    """Get mode router instance with current db session."""
    # Create fresh instance with current request's db session
    # Don't cache - db session is request-scoped
    return ModeRouter(db)


def get_cost_estimator(db: AsyncSession = Depends(get_db)) -> AgentCostEstimator:
    """Get cost estimator instance with current db session."""
    # Create fresh instance with current request's db session
    return AgentCostEstimator(db)


# =============================================================================
# Execution Endpoints
# =============================================================================

@router.post("/execute")
async def execute_request(
    request: AgentExecutionRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """
    Execute a request through the multi-agent system.

    Returns the final result (non-streaming).
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = user.user_id

    result = None
    async for update in orchestrator.process_request(
        request=request.message,
        session_id=session_id,
        user_id=user_id,
        context=request.context,
    ):
        if update.get("type") == "plan_completed":
            result = update
        elif update.get("type") == "error":
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=update.get("error"))
        elif update.get("type") in ("budget_exceeded", "approval_required"):
            return update

    if result:
        return result

    return {"status": "completed", "session_id": session_id}


@router.post("/execute/stream")
async def execute_stream(
    request: AgentExecutionRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """
    Execute with SSE streaming progress updates.

    Returns Server-Sent Events with execution progress.
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = user.user_id

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for update in orchestrator.process_request(
                request=request.message,
                session_id=session_id,
                user_id=user_id,
                context=request.context,
            ):
                yield f"data: {json.dumps(update)}\n\n"

            yield "data: {\"type\": \"stream_end\"}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/execute/approve/{plan_id}")
async def approve_execution(
    plan_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Approve and execute a plan that was awaiting cost approval."""
    user_id = user.user_id

    async def event_generator() -> AsyncGenerator[str, None]:
        async for update in orchestrator.approve_execution(plan_id, user_id):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/execute/cancel/{plan_id}")
async def cancel_execution(
    plan_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Cancel pending or running execution."""
    user_id = user.user_id

    result = await orchestrator.cancel_execution(plan_id, user_id)

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))

    return result


# =============================================================================
# Plan Status Endpoints
# =============================================================================

@router.get("/plans/{plan_id}")
async def get_plan_status(
    plan_id: str,
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
) -> PlanStatusResponse:
    """Get execution plan status."""
    status = await orchestrator.get_plan_status(plan_id)

    if not status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")

    return PlanStatusResponse(**status)


@router.get("/plans")
async def list_plans(
    user: AuthenticatedUser,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """List user's execution plans."""
    user_id = user.user_id

    plans = await orchestrator.get_execution_history(user_id, limit)
    return {"plans": plans, "count": len(plans)}


# =============================================================================
# Mode Management Endpoints
# =============================================================================

@router.get("/mode")
async def get_mode(
    user: AuthenticatedUser,
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    mode_router: ModeRouter = Depends(get_mode_router),
) -> ModePreferencesResponse:
    """Get current execution mode and preferences."""
    user_id = user.user_id

    mode_info = await mode_router.get_current_mode(user_id, session_id)
    return ModePreferencesResponse(**mode_info)


@router.post("/mode")
async def set_mode(
    request: SetModeRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    mode_router: ModeRouter = Depends(get_mode_router),
):
    """Set execution mode preference."""
    user_id = user.user_id

    prefs = await mode_router.update_preferences(
        user_id, default_mode=request.mode
    )

    return {"success": True, "mode": prefs.default_mode}


@router.post("/mode/toggle")
async def toggle_agent_mode(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    mode_router: ModeRouter = Depends(get_mode_router),
):
    """Toggle agent mode on/off."""
    user_id = user.user_id

    enabled = await mode_router.toggle_agent_mode(user_id)

    return {"agent_mode_enabled": enabled}


@router.get("/preferences")
async def get_preferences(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    mode_router: ModeRouter = Depends(get_mode_router),
) -> ModePreferencesResponse:
    """Get user's agent preferences."""
    user_id = user.user_id

    mode_info = await mode_router.get_current_mode(user_id)
    return ModePreferencesResponse(**mode_info)


@router.patch("/preferences")
async def update_preferences(
    request: UpdatePreferencesRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
    mode_router: ModeRouter = Depends(get_mode_router),
):
    """Update agent preferences."""
    user_id = user.user_id

    prefs = await mode_router.update_preferences(
        user_id,
        default_mode=request.default_mode,
        agent_mode_enabled=request.agent_mode_enabled,
        auto_detect_complexity=request.auto_detect_complexity,
        show_cost_estimation=request.show_cost_estimation,
        require_approval_above_usd=request.require_approval_above_usd,
        general_chat_enabled=request.general_chat_enabled,
        fallback_to_general=request.fallback_to_general,
    )

    return {
        "success": True,
        "preferences": {
            "default_mode": prefs.default_mode,
            "agent_mode_enabled": prefs.agent_mode_enabled,
            "auto_detect_complexity": prefs.auto_detect_complexity,
            "show_cost_estimation": prefs.show_cost_estimation,
            "require_approval_above_usd": prefs.require_approval_above_usd,
            "general_chat_enabled": prefs.general_chat_enabled,
            "fallback_to_general": prefs.fallback_to_general,
        },
    }


@router.post("/analyze-complexity")
async def analyze_complexity(
    request: AgentExecutionRequest,
    db: AsyncSession = Depends(get_db),
    mode_router: ModeRouter = Depends(get_mode_router),
):
    """Analyze request complexity without executing."""
    analysis = await mode_router.analyze_request_complexity(
        request.message, request.context
    )
    return analysis


# =============================================================================
# Agent Status Endpoints
# =============================================================================

@router.get("/status")
async def get_system_status(
    db: AsyncSession = Depends(get_db),
):
    """Get status of all agents from database."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(AgentDefinition).order_by(AgentDefinition.agent_type)
    )
    agents = result.scalars().all()

    # Get active prompt versions for all agents
    agent_prompts = {}
    for agent in agents:
        prompt_version = None
        # First try the explicitly set active_prompt_version_id
        if agent.active_prompt_version_id:
            prompt_result = await db.execute(
                select(AgentPromptVersion).where(
                    AgentPromptVersion.id == agent.active_prompt_version_id
                )
            )
            prompt_version = prompt_result.scalar_one_or_none()

        # Fallback: look for prompt version with is_active=True
        if not prompt_version:
            prompt_result = await db.execute(
                select(AgentPromptVersion).where(
                    and_(
                        AgentPromptVersion.agent_id == agent.id,
                        AgentPromptVersion.is_active == True
                    )
                )
            )
            prompt_version = prompt_result.scalar_one_or_none()

        if prompt_version:
            agent_prompts[str(agent.id)] = {
                "system_prompt": prompt_version.system_prompt,
                "task_prompt_template": prompt_version.task_prompt_template,
            }

    return {
        "agents": [
            {
                "id": str(a.id),
                "name": a.name,
                "agent_type": a.agent_type,
                "description": a.description,
                "is_active": a.is_active,
                "total_executions": a.total_executions or 0,
                "success_rate": a.success_rate or 0.0,
                "avg_latency_ms": a.avg_latency_ms or 0,
                "avg_tokens_per_execution": a.avg_tokens_per_execution or 0,
                "default_model": a.default_model,
                "default_temperature": a.default_temperature,
                # Include active prompt if available
                "system_prompt": agent_prompts.get(str(a.id), {}).get("system_prompt"),
                "task_prompt_template": agent_prompts.get(str(a.id), {}).get("task_prompt_template"),
            }
            for a in agents
        ],
        "total": len(agents),
        "healthy": sum(1 for a in agents if a.is_active),
        "degraded": 0,
    }


@router.get("/agents/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    hours: int = Query(24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Get agent performance metrics."""
    metrics = await orchestrator.get_agent_metrics(agent_id, hours)
    return metrics


@router.get("/agents/{agent_id}/config")
async def get_agent_config(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get agent configuration including LLM provider."""
    from sqlalchemy import select

    result = await db.execute(
        select(AgentDefinition)
        .where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    return {
        "id": str(agent.id),
        "name": agent.name,
        "agent_type": agent.agent_type,
        "description": agent.description,
        "default_provider_id": str(agent.default_provider_id) if agent.default_provider_id else None,
        "default_model": agent.default_model,
        "default_temperature": agent.default_temperature,
        "max_tokens": agent.max_tokens,
        "is_active": agent.is_active,
        "success_rate": agent.success_rate,
        "avg_latency_ms": agent.avg_latency_ms,
    }


@router.patch("/agents/{agent_id}/config")
async def update_agent_config(
    agent_id: str,
    request: UpdateAgentConfigRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update agent LLM provider/model configuration."""
    from sqlalchemy import select

    result = await db.execute(
        select(AgentDefinition)
        .where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    if request.default_provider_id is not None:
        agent.default_provider_id = uuid.UUID(request.default_provider_id) if request.default_provider_id else None
    if request.default_model is not None:
        agent.default_model = request.default_model
    if request.default_temperature is not None:
        agent.default_temperature = request.default_temperature
    if request.max_tokens is not None:
        agent.max_tokens = request.max_tokens
    if request.is_active is not None:
        agent.is_active = request.is_active

    await db.commit()

    return {"success": True, "agent_id": agent_id}


# =============================================================================
# Agent CRUD Endpoints
# =============================================================================

@router.post("/agents")
async def create_agent(
    request: CreateAgentRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new agent definition.

    Creates a custom agent with specified configuration. The agent_type must be unique.
    """
    from sqlalchemy import select

    # Check if agent_type already exists
    existing = await db.execute(
        select(AgentDefinition).where(AgentDefinition.agent_type == request.agent_type)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent type '{request.agent_type}' already exists"
        )

    # Build settings with knowledge bases if provided
    settings = request.settings or {}
    if request.knowledge_bases:
        settings["knowledge_bases"] = request.knowledge_bases

    agent = AgentDefinition(
        name=request.name,
        agent_type=request.agent_type,
        description=request.description,
        default_temperature=request.default_temperature,
        max_tokens=request.max_tokens,
        settings=settings,
        default_provider_id=uuid.UUID(request.default_provider_id) if request.default_provider_id else None,
        default_model=request.default_model,
        is_active=True,
        # Voice/Chat agent fields
        agent_mode=request.agent_mode,
        tts_config=request.tts_config.model_dump() if request.tts_config else None,
    )

    db.add(agent)
    await db.commit()
    await db.refresh(agent)

    # Create initial prompt version if system_prompt provided
    if request.system_prompt:
        prompt_version = AgentPromptVersion(
            agent_id=agent.id,
            version_number=1,
            system_prompt=request.system_prompt,
            task_prompt_template="{{input}}",
            change_reason="Initial prompt",
            created_by="api",
        )
        db.add(prompt_version)
        await db.commit()
        await db.refresh(prompt_version)

        # Set as active prompt
        agent.active_prompt_version_id = prompt_version.id
        await db.commit()

    logger.info("Created new agent", agent_id=str(agent.id), agent_type=request.agent_type, agent_mode=request.agent_mode)

    return {
        "id": str(agent.id),
        "name": agent.name,
        "agent_type": agent.agent_type,
        "agent_mode": agent.agent_mode,
        "message": f"Agent '{agent.name}' created successfully",
    }


@router.put("/agents/{agent_id}")
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Update an agent definition (full update).

    Updates agent name, description, settings, and LLM configuration.
    If system_prompt or task_prompt_template are provided, creates a new prompt version.
    """
    from sqlalchemy import select

    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    # Update fields if provided
    if request.name is not None:
        agent.name = request.name
    if request.description is not None:
        agent.description = request.description
    if request.default_temperature is not None:
        agent.default_temperature = request.default_temperature
    if request.max_tokens is not None:
        agent.max_tokens = request.max_tokens
    if request.settings is not None:
        agent.settings = request.settings
    if request.default_provider_id is not None:
        agent.default_provider_id = uuid.UUID(request.default_provider_id) if request.default_provider_id else None
    if request.default_model is not None:
        agent.default_model = request.default_model
    if request.is_active is not None:
        agent.is_active = request.is_active

    # Handle prompt updates - create a new prompt version if prompts are provided
    prompt_version_created = False
    if request.system_prompt is not None or request.task_prompt_template is not None:
        # Get the current active prompt version to inherit values
        current_prompt_result = await db.execute(
            select(AgentPromptVersion).where(
                and_(
                    AgentPromptVersion.agent_id == agent.id,
                    AgentPromptVersion.is_active == True
                )
            )
        )
        current_prompt = current_prompt_result.scalar_one_or_none()

        # Get next version number
        version_count_result = await db.execute(
            select(func.count(AgentPromptVersion.id)).where(
                AgentPromptVersion.agent_id == agent.id
            )
        )
        version_number = (version_count_result.scalar() or 0) + 1

        # Deactivate current prompt version if exists
        if current_prompt:
            current_prompt.is_active = False

        # Create new prompt version
        new_prompt = AgentPromptVersion(
            id=uuid.uuid4(),
            agent_id=agent.id,
            version_number=version_number,
            system_prompt=request.system_prompt if request.system_prompt is not None else (current_prompt.system_prompt if current_prompt else ""),
            task_prompt_template=request.task_prompt_template if request.task_prompt_template is not None else (current_prompt.task_prompt_template if current_prompt else ""),
            few_shot_examples=current_prompt.few_shot_examples if current_prompt else None,
            output_schema=current_prompt.output_schema if current_prompt else None,
            change_reason="Updated via agent edit (enhanced prompt)",
            is_active=True,
        )
        db.add(new_prompt)

        # Update agent's active prompt version reference
        agent.active_prompt_version_id = new_prompt.id
        prompt_version_created = True

        logger.info("Created new prompt version for agent", agent_id=agent_id, version=version_number)

    await db.commit()

    logger.info("Updated agent", agent_id=agent_id)

    message = f"Agent '{agent.name}' updated successfully"
    if prompt_version_created:
        message += " with new prompt version"

    return {
        "success": True,
        "id": str(agent.id),
        "name": agent.name,
        "message": message,
    }


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    hard_delete: bool = Query(False, description="Permanently delete instead of deactivate"),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an agent definition.

    By default, performs a soft delete (deactivates the agent).
    Use hard_delete=true to permanently remove the agent.
    """
    from sqlalchemy import select

    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    agent_name = agent.name
    agent_type = agent.agent_type

    # Prevent deletion of core agents
    core_agents = {"manager", "generator", "critic", "research", "tool_executor"}
    if agent_type in core_agents and hard_delete:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot permanently delete core agent type '{agent_type}'. Use soft delete instead."
        )

    if hard_delete:
        await db.delete(agent)
        logger.info("Hard deleted agent", agent_id=agent_id, agent_type=agent_type)
        message = f"Agent '{agent_name}' permanently deleted"
    else:
        agent.is_active = False
        logger.info("Soft deleted agent", agent_id=agent_id, agent_type=agent_type)
        message = f"Agent '{agent_name}' deactivated"

    await db.commit()

    return {
        "success": True,
        "id": agent_id,
        "deleted": hard_delete,
        "message": message,
    }


# =============================================================================
# Prompt Optimization Endpoints (Admin)
# =============================================================================

async def process_optimization_job_background(job_id: str, agent_id: str):
    """
    Background task to process optimization job.
    Runs asynchronously after API returns.

    Phase 15 Integration: This ensures optimization jobs complete properly.

    Steps:
    1. pending → analyzing: Analyze failed trajectories with LLM
    2. analyzing → generating: Generate 3 prompt variants
    3. generating → awaiting_approval: Request user approval
    """
    from backend.services.prompt_optimization.prompt_builder_agent import PromptBuilderAgent
    from backend.services.agents.agent_base import AgentConfig
    from backend.db.database import get_async_session
    from sqlalchemy import select

    async with get_async_session() as db:
        try:
            logger.info(f"Starting background processing for optimization job {job_id}")

            # Get agent
            result = await db.execute(
                select(AgentDefinition)
                .where(AgentDefinition.id == uuid.UUID(agent_id))
            )
            agent = result.scalar_one_or_none()

            if not agent:
                logger.error(f"Agent {agent_id} not found for job {job_id}")
                return

            # Get job
            result = await db.execute(
                select(PromptOptimizationJob)
                .where(PromptOptimizationJob.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()

            if not job:
                logger.error(f"Job {job_id} not found")
                return

            # Create builder
            config = AgentConfig(
                agent_id=str(uuid.uuid4()),
                name="Prompt Builder",
                description="Prompt optimization agent",
            )
            builder = PromptBuilderAgent(config, db)

            # Step 1: Analyze failures
            logger.info(f"Job {job_id}: Starting failure analysis")
            job.status = "analyzing"
            await db.commit()

            analysis_result = await builder.analyze_failures(job, agent)
            job.analysis_result = analysis_result
            await db.commit()

            logger.info(f"Job {job_id}: Failure analysis complete")

            # Step 2: Generate variants
            logger.info(f"Job {job_id}: Generating prompt variants")
            job.status = "generating"
            await db.commit()

            variants = await builder.generate_prompt_variants(job, agent, analysis_result)
            job.generated_variants = variants
            await db.commit()

            logger.info(f"Job {job_id}: Generated {len(variants) if variants else 0} variants")

            # Step 3: Request approval (sets status to awaiting_approval)
            logger.info(f"Job {job_id}: Requesting approval")
            await builder.request_approval(job, variants)

            logger.info(f"Optimization job {job_id} completed and awaiting approval")

        except Exception as e:
            logger.error(f"Error processing optimization job {job_id}: {e}", exc_info=True)
            try:
                # Try to update job status to failed
                result = await db.execute(
                    select(PromptOptimizationJob)
                    .where(PromptOptimizationJob.id == uuid.UUID(job_id))
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    await db.commit()
            except Exception as commit_error:
                logger.error(f"Failed to update job status to failed: {commit_error}")


@router.post("/agents/{agent_id}/optimize")
async def trigger_optimization(
    agent_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger prompt optimization for an agent."""
    from backend.services.prompt_optimization.prompt_builder_agent import PromptBuilderAgent
    from backend.services.agents.agent_base import AgentConfig
    from sqlalchemy import select

    # Get agent
    result = await db.execute(
        select(AgentDefinition)
        .where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    # Create optimization job
    config = AgentConfig(
        agent_id=str(uuid.uuid4()),
        name="Prompt Builder",
        description="Prompt optimization agent",
    )
    builder = PromptBuilderAgent(config, db)
    job = await builder.create_optimization_job(agent)

    # Start background processing
    background_tasks.add_task(
        process_optimization_job_background,
        job_id=str(job.id),
        agent_id=agent_id
    )

    return {
        "success": True,
        "job_id": str(job.id),
        "agent_id": agent_id,
        "status": job.status,
        "message": "Optimization started in background. Job will transition through: pending → analyzing → generating → awaiting_approval",
    }


@router.get("/optimization/jobs")
async def list_optimization_jobs(
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List prompt optimization jobs."""
    from sqlalchemy import select

    query = select(PromptOptimizationJob).order_by(
        PromptOptimizationJob.created_at.desc()
    ).limit(limit)

    if status:
        query = query.where(PromptOptimizationJob.status == status)

    result = await db.execute(query)
    jobs = result.scalars().all()

    return {
        "jobs": [
            {
                "id": str(j.id),
                "agent_id": str(j.agent_id),
                "status": j.status,
                "trajectories_analyzed": j.trajectories_analyzed,
                "variants_generated": j.variants_generated,
                "baseline_success_rate": j.baseline_success_rate,
                "new_success_rate": j.new_success_rate,
                "improvement_percentage": j.improvement_percentage,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ],
        "count": len(jobs),
    }


@router.get("/optimization/jobs/{job_id}")
async def get_optimization_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get optimization job details."""
    from sqlalchemy import select

    result = await db.execute(
        select(PromptOptimizationJob)
        .where(PromptOptimizationJob.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return {
        "id": str(job.id),
        "agent_id": str(job.agent_id),
        "status": job.status,
        "analysis_window_hours": job.analysis_window_hours,
        "trajectories_analyzed": job.trajectories_analyzed,
        "failure_patterns": job.failure_patterns,
        "variants_generated": job.variants_generated,
        "variant_ids": job.variant_ids,
        "winning_variant_id": str(job.winning_variant_id) if job.winning_variant_id else None,
        "baseline_success_rate": job.baseline_success_rate,
        "new_success_rate": job.new_success_rate,
        "improvement_percentage": job.improvement_percentage,
        "approved_by": str(job.approved_by) if job.approved_by else None,
        "approved_at": job.approved_at.isoformat() if job.approved_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@router.post("/optimization/jobs/{job_id}/approve")
async def approve_optimization(
    job_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
):
    """Approve prompt optimization and promote winning variant."""
    from sqlalchemy import select

    user_id = user.user_id

    result = await db.execute(
        select(PromptOptimizationJob)
        .where(PromptOptimizationJob.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status != "awaiting_approval":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Job is not awaiting approval (status: {job.status})")

    if not job.winning_variant_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No winning variant selected")

    # Promote winning variant
    version_manager = PromptVersionManager(db)
    await version_manager.promote_version(
        str(job.agent_id),
        str(job.winning_variant_id),
        user_id,
    )

    # Update job
    job.status = "completed"
    job.approved_by = uuid.UUID(user_id)
    job.approved_at = datetime.utcnow()
    await db.commit()

    return {
        "success": True,
        "job_id": job_id,
        "promoted_variant_id": str(job.winning_variant_id),
    }


@router.post("/optimization/jobs/{job_id}/reject")
async def reject_optimization(
    job_id: str,
    reason: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
):
    """Reject prompt optimization."""
    from sqlalchemy import select

    result = await db.execute(
        select(PromptOptimizationJob)
        .where(PromptOptimizationJob.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    job.status = "rejected"
    job.rejection_reason = reason
    await db.commit()

    return {"success": True, "job_id": job_id, "status": "rejected"}


# =============================================================================
# Prompt Version Endpoints
# =============================================================================

@router.get("/agents/{agent_id}/prompts")
async def get_prompt_history(
    agent_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Get prompt version history."""
    version_manager = PromptVersionManager(db)
    history = await version_manager.get_version_history(agent_id, limit)
    return {"versions": history, "count": len(history)}


@router.get("/agents/{agent_id}/prompts/{version_id}")
async def get_prompt_version(
    agent_id: str,
    version_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get specific prompt version details."""
    version_manager = PromptVersionManager(db)
    version = await version_manager.get_version(version_id)

    if not version or str(version.agent_id) != agent_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    return {
        "id": str(version.id),
        "version_number": version.version_number,
        "system_prompt": version.system_prompt,
        "task_prompt_template": version.task_prompt_template,
        "few_shot_examples": version.few_shot_examples,
        "output_schema": version.output_schema,
        "change_reason": version.change_reason,
        "created_by": version.created_by,
        "created_at": version.created_at.isoformat() if version.created_at else None,
        "is_active": version.is_active,
        "traffic_percentage": version.traffic_percentage,
        "execution_count": version.execution_count,
        "success_count": version.success_count,
        "avg_quality_score": version.avg_quality_score,
    }


@router.post("/agents/{agent_id}/prompts")
async def create_prompt_version(
    agent_id: str,
    request: CreatePromptVersionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new prompt version manually."""
    version_manager = PromptVersionManager(db)

    version = await version_manager.create_version(
        agent_id=agent_id,
        system_prompt=request.system_prompt,
        task_prompt_template=request.task_prompt_template,
        change_reason=request.change_reason,
        created_by="manual",
        few_shot_examples=request.few_shot_examples,
        output_schema=request.output_schema,
    )

    return {
        "success": True,
        "version_id": str(version.id),
        "version_number": version.version_number,
    }


@router.post("/agents/{agent_id}/prompts/rollback")
async def rollback_prompt(
    agent_id: str,
    version_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Rollback to previous prompt version."""
    version_manager = PromptVersionManager(db)

    result = await version_manager.rollback(agent_id, version_id)

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))

    return result


# =============================================================================
# Trajectory Endpoints (Debug)
# =============================================================================

@router.get("/trajectories")
async def list_trajectories(
    agent_id: Optional[str] = None,
    success: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List agent execution trajectories with agent names."""
    from sqlalchemy import select, desc
    from datetime import datetime, timedelta

    # Build query with join to get agent name
    cutoff = datetime.utcnow() - timedelta(hours=24 * 7)  # Last 7 days

    query = (
        select(AgentTrajectory, AgentDefinition.name.label("agent_name"), AgentDefinition.agent_type)
        .outerjoin(AgentDefinition, AgentTrajectory.agent_id == AgentDefinition.id)
        .where(AgentTrajectory.created_at >= cutoff)
        .order_by(desc(AgentTrajectory.created_at))
        .limit(limit)
    )

    if agent_id:
        query = query.where(AgentTrajectory.agent_id == uuid.UUID(agent_id))

    if success is not None:
        query = query.where(AgentTrajectory.success == success)

    result = await db.execute(query)
    rows = result.all()

    return {
        "trajectories": [
            {
                "id": str(traj.id),
                "session_id": str(traj.session_id),
                "agent_id": str(traj.agent_id) if traj.agent_id else None,
                "agent_name": agent_name or "Unknown",
                "agent_type": agent_type or "unknown",
                "task_type": traj.task_type,
                "input_summary": traj.input_summary,
                "success": traj.success,
                "quality_score": traj.quality_score,
                "error_message": traj.error_message,
                "total_tokens": traj.total_tokens,
                "total_duration_ms": traj.total_duration_ms,
                "total_cost_usd": traj.total_cost_usd,
                "user_rating": traj.user_rating,
                "created_at": traj.created_at.isoformat() if traj.created_at else None,
            }
            for traj, agent_name, agent_type in rows
        ],
        "count": len(rows),
    }


@router.get("/trajectories/{trajectory_id}")
async def get_trajectory(
    trajectory_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed trajectory."""
    collector = TrajectoryCollector(db)
    trajectory = await collector.get_trajectory(trajectory_id)

    if not trajectory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trajectory not found")

    return {
        "id": str(trajectory.id),
        "session_id": str(trajectory.session_id),
        "agent_id": str(trajectory.agent_id) if trajectory.agent_id else None,
        "task_type": trajectory.task_type,
        "input_summary": trajectory.input_summary,
        "trajectory_steps": trajectory.trajectory_steps,
        "success": trajectory.success,
        "quality_score": trajectory.quality_score,
        "error_message": trajectory.error_message,
        "total_tokens": trajectory.total_tokens,
        "total_duration_ms": trajectory.total_duration_ms,
        "total_cost_usd": trajectory.total_cost_usd,
        "user_rating": trajectory.user_rating,
        "user_feedback": trajectory.user_feedback,
        "created_at": trajectory.created_at.isoformat() if trajectory.created_at else None,
    }


@router.post("/trajectories/{trajectory_id}/feedback")
async def add_trajectory_feedback(
    trajectory_id: str,
    rating: int = Query(..., ge=1, le=5),
    feedback: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Add user feedback to a trajectory."""
    collector = TrajectoryCollector(db)
    trajectory = await collector.add_user_feedback(trajectory_id, rating, feedback)

    if not trajectory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trajectory not found")

    return {"success": True, "trajectory_id": trajectory_id}


# =============================================================================
# Agent Enhancement Endpoints
# =============================================================================

class EnhancePromptRequest(BaseModel):
    """Request to enhance an agent's prompt."""
    strategy: Optional[str] = Field(
        None,
        description="Mutation strategy: rephrase_instructions, add_examples, add_guardrails, restructure_format, add_chain_of_thought, simplify, add_constraints"
    )
    custom_instructions: Optional[str] = Field(
        None,
        description="Custom instructions for the enhancement"
    )
    enhance_description: bool = Field(
        False,
        description="Whether to also enhance the agent description"
    )


class UpdateAgentSettingsRequest(BaseModel):
    """Request to update agent settings including tools and external config."""
    tools_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Tools configuration: {web_search: bool, code_execution: bool, file_access: bool, mcp_server_url: str}"
    )
    external_agent: Optional[Dict[str, Any]] = Field(
        None,
        description="External agent configuration: {api_url: str, api_key: str, enabled: bool}"
    )
    custom_settings: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional custom settings"
    )


@router.post("/agents/{agent_id}/enhance-prompt")
async def enhance_agent_prompt(
    agent_id: str,
    request: EnhancePromptRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
):
    """
    Enhance an agent's prompt using AI-powered optimization.

    Uses the PromptBuilderAgent to analyze the agent's current prompt
    and generate an improved version using GEPA-style mutations.
    """
    from backend.services.prompt_optimization.prompt_builder_agent import (
        PromptBuilderAgent,
        MutationStrategy,
    )
    from backend.services.agents.agent_base import AgentConfig

    # Get the agent
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    # Get current active prompt version
    result = await db.execute(
        select(AgentPromptVersion).where(
            and_(
                AgentPromptVersion.agent_id == agent.id,
                AgentPromptVersion.is_active == True
            )
        )
    )
    prompt_version = result.scalar_one_or_none()

    if not prompt_version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active prompt version found for this agent"
        )

    # Validate that prompts are not empty - cannot enhance empty prompts
    system_empty = not prompt_version.system_prompt or not prompt_version.system_prompt.strip()
    task_empty = not prompt_version.task_prompt_template or not prompt_version.task_prompt_template.strip()

    if system_empty and task_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent has empty prompts. Please set prompts before enhancing. "
                   "Both system_prompt and task_prompt_template are empty or whitespace-only."
        )

    try:
        # Create prompt builder agent
        config = AgentConfig(
            agent_id=str(uuid.uuid4()),
            name="Prompt Builder",
            description="Prompt optimization agent",
        )
        builder = PromptBuilderAgent(config, db)

        # Determine strategy
        strategy = None
        if request.strategy:
            try:
                strategy = MutationStrategy(request.strategy)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid strategy. Valid options: {[s.value for s in MutationStrategy]}"
                )

        # If no strategy specified, analyze and suggest the best one
        if not strategy:
            # Analyze recent failures to determine best strategy
            analysis = await builder.analyze_failures(agent, hours=168)  # Last week

            if analysis and analysis.recommendations:
                # Map recommendations to strategies
                strategy_map = {
                    "rephrase": MutationStrategy.REPHRASE_INSTRUCTIONS,
                    "example": MutationStrategy.ADD_EXAMPLES,
                    "guardrail": MutationStrategy.ADD_GUARDRAILS,
                    "format": MutationStrategy.RESTRUCTURE_FORMAT,
                    "chain": MutationStrategy.ADD_CHAIN_OF_THOUGHT,
                    "simplif": MutationStrategy.SIMPLIFY,
                    "constraint": MutationStrategy.ADD_CONSTRAINTS,
                }

                for rec in analysis.recommendations:
                    rec_lower = rec.lower()
                    for key, strat in strategy_map.items():
                        if key in rec_lower:
                            strategy = strat
                            break
                    if strategy:
                        break

            # Default to chain of thought if no strategy determined
            if not strategy:
                strategy = MutationStrategy.ADD_CHAIN_OF_THOUGHT

        # Generate enhanced prompt
        mutation = await builder.generate_mutation(
            agent_id=str(agent.id),
            strategy=strategy,
            custom_context=request.custom_instructions,
        )

        if not mutation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate enhanced prompt"
            )

        # Optionally enhance description
        enhanced_description = None
        if request.enhance_description and agent.description:
            try:
                # Use the LLM to enhance the description
                from backend.services.llm_router import LLMRouter
                llm_router = LLMRouter(db)

                description_prompt = f"""Improve this agent description to be clearer and more informative.
Keep it concise (1-3 sentences) but make it more specific about what the agent does.

Current description: {agent.description}

Agent type: {agent.agent_type}
Agent name: {agent.name}

{f"Additional context: {request.custom_instructions}" if request.custom_instructions else ""}

Respond with ONLY the improved description, nothing else."""

                response = await llm_router.chat(
                    messages=[{"role": "user", "content": description_prompt}],
                    operation_type="agent_enhancement",
                )
                enhanced_description = response.content.strip()
            except Exception as e:
                logger.warning("Failed to enhance description", error=str(e))
                # Continue without enhanced description

        return {
            "success": True,
            "enhanced_prompt": {
                "system_prompt": mutation.system_prompt,
                "task_prompt_template": mutation.task_prompt_template,
                "few_shot_examples": mutation.few_shot_examples,
            },
            "enhanced_description": enhanced_description,
            "strategy_used": mutation.strategy.value,
            "change_description": mutation.change_description,
            "expected_improvement": mutation.expected_improvement,
            "original_version_id": str(prompt_version.id),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(
            "Failed to enhance prompt",
            agent_id=agent_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enhance prompt: {str(e)}"
        )


@router.patch("/agents/{agent_id}/settings")
async def update_agent_settings(
    agent_id: str,
    request: UpdateAgentSettingsRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
):
    """
    Update agent settings including tools configuration and external agent connection.

    Tools config example:
    {
        "web_search": true,
        "code_execution": false,
        "file_access": true,
        "mcp_server_url": "http://localhost:3000/mcp"
    }

    External agent example:
    {
        "api_url": "https://external-agent.example.com/v1/chat",
        "api_key": "sk-xxx",
        "enabled": true
    }
    """
    # Get the agent
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    # Update settings
    current_settings = agent.settings or {}

    if request.tools_config is not None:
        current_settings["tools_config"] = request.tools_config

    if request.external_agent is not None:
        # Validate external agent config
        ext = request.external_agent
        if ext.get("enabled") and not ext.get("api_url"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="api_url is required when external agent is enabled"
            )
        current_settings["external_agent"] = ext

    if request.custom_settings is not None:
        current_settings.update(request.custom_settings)

    agent.settings = current_settings
    await db.commit()
    await db.refresh(agent)

    return {
        "success": True,
        "agent_id": str(agent.id),
        "settings": agent.settings,
    }


@router.get("/agents/{agent_id}/settings")
async def get_agent_settings(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get agent settings including tools configuration and external agent connection."""
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    settings = agent.settings or {}

    return {
        "agent_id": str(agent.id),
        "tools_config": settings.get("tools_config", {
            "web_search": False,
            "code_execution": False,
            "file_access": False,
            "mcp_server_url": None,
        }),
        "external_agent": settings.get("external_agent", {
            "api_url": None,
            "api_key": None,
            "enabled": False,
        }),
        "custom_settings": {
            k: v for k, v in settings.items()
            if k not in ["tools_config", "external_agent"]
        },
    }


@router.post("/agents/{agent_id}/test-external")
async def test_external_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Test connection to an external agent."""
    import httpx

    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    settings = agent.settings or {}
    external_config = settings.get("external_agent", {})

    if not external_config.get("api_url"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No external agent URL configured"
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {}
            if external_config.get("api_key"):
                headers["Authorization"] = f"Bearer {external_config['api_key']}"

            # Try a simple health check or options request
            response = await client.options(
                external_config["api_url"],
                headers=headers,
            )

            if response.status_code < 400:
                return {
                    "success": True,
                    "message": "Connection successful",
                    "status_code": response.status_code,
                }
            else:
                return {
                    "success": False,
                    "message": f"Server returned status {response.status_code}",
                    "status_code": response.status_code,
                }

    except httpx.TimeoutException:
        return {
            "success": False,
            "message": "Connection timed out",
        }
    except httpx.RequestError as e:
        return {
            "success": False,
            "message": f"Connection failed: {str(e)}",
        }


# =============================================================================
# Agent Publishing Endpoints
# =============================================================================

class PublishAgentRequest(BaseModel):
    """Request to publish an agent for external use."""
    allowed_domains: List[str] = Field(default=["*"], description="Allowed domains for embedding")
    rate_limit: int = Field(default=100, ge=1, le=10000, description="Requests per hour limit")
    welcome_message: Optional[str] = Field(None, max_length=500)
    branding: Optional[Dict[str, Any]] = Field(None, description="Custom branding config")
    # Branding: {"logo_url": "...", "primary_color": "#...", "agent_name": "..."}


class PublishAgentResponse(BaseModel):
    """Response after publishing an agent."""
    agent_id: str
    embed_token: str
    embed_code: str
    widget_url: str
    is_published: bool


class EmbedChatRequest(BaseModel):
    """Request for embedded chat interaction."""
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/agents/{agent_id}/publish", response_model=PublishAgentResponse)
async def publish_agent(
    agent_id: str,
    request: PublishAgentRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
):
    """
    Publish an agent for external embedding.

    Creates an embed token that can be used to embed the agent as a
    chat widget on external websites.
    """
    import secrets

    # Get the agent
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    # Generate embed token if not exists
    if not agent.embed_token:
        agent.embed_token = secrets.token_urlsafe(32)

    # Update publish config
    agent.is_published = True
    agent.publish_config = {
        "allowed_domains": request.allowed_domains,
        "rate_limit": request.rate_limit,
        "welcome_message": request.welcome_message,
        "branding": request.branding or {},
        "published_by": str(user.id),
        "published_at": datetime.utcnow().isoformat(),
    }

    await db.commit()
    await db.refresh(agent)

    # Generate embed code
    from backend.core.config import settings
    base_url = settings.BASE_URL or "http://localhost:8000"
    widget_url = f"{base_url}/embed/chat/{agent.embed_token}"

    embed_code = f'''<!-- AI Agent Widget -->
<script src="{base_url}/static/embed/agent-widget.js"></script>
<script>
  AIAgent.init({{
    token: "{agent.embed_token}",
    position: "bottom-right",
    theme: "light"
  }});
</script>'''

    return PublishAgentResponse(
        agent_id=str(agent.id),
        embed_token=agent.embed_token,
        embed_code=embed_code,
        widget_url=widget_url,
        is_published=True,
    )


@router.delete("/agents/{agent_id}/publish")
async def unpublish_agent(
    agent_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
):
    """Unpublish an agent, disabling external access."""
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    agent.is_published = False
    # Keep the embed_token so republishing uses the same token

    await db.commit()

    return {"success": True, "message": "Agent unpublished successfully"}


@router.get("/agents/{agent_id}/publish/status")
async def get_publish_status(
    agent_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
):
    """Get the publishing status of an agent."""
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    from backend.core.config import settings
    base_url = settings.BASE_URL or "http://localhost:8000"

    return {
        "agent_id": str(agent.id),
        "is_published": agent.is_published,
        "embed_token": agent.embed_token if agent.is_published else None,
        "widget_url": f"{base_url}/embed/chat/{agent.embed_token}" if agent.is_published and agent.embed_token else None,
        "publish_config": agent.publish_config if agent.is_published else None,
    }


# =============================================================================
# Public Embedded Agent Endpoints (No Auth Required)
# =============================================================================

@router.get("/embed/{embed_token}/config")
async def get_embed_config(
    embed_token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get configuration for embedded agent widget.
    This endpoint is public and accessed by the embed script.
    """
    result = await db.execute(
        select(AgentDefinition).where(
            and_(
                AgentDefinition.embed_token == embed_token,
                AgentDefinition.is_published == True,
                AgentDefinition.is_active == True,
            )
        )
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found or not published")

    publish_config = agent.publish_config or {}

    return {
        "agent_name": agent.name,
        "agent_type": agent.agent_type,
        "agent_mode": agent.agent_mode or "chat",
        "welcome_message": publish_config.get("welcome_message", f"Hi! I'm {agent.name}. How can I help you?"),
        "branding": publish_config.get("branding", {}),
        "tts_enabled": agent.agent_mode in ["voice", "hybrid"],
        "tts_config": agent.tts_config if agent.agent_mode in ["voice", "hybrid"] else None,
    }


@router.post("/embed/{embed_token}/chat")
async def embedded_chat(
    embed_token: str,
    request: EmbedChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Chat endpoint for embedded agent widget.
    This endpoint is public and rate-limited based on publish config.
    """
    # Get the agent
    result = await db.execute(
        select(AgentDefinition).where(
            and_(
                AgentDefinition.embed_token == embed_token,
                AgentDefinition.is_published == True,
                AgentDefinition.is_active == True,
            )
        )
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found or not published")

    # TODO: Add rate limiting based on publish_config["rate_limit"]

    try:
        from backend.services.llm import EnhancedLLMFactory
        from langchain_core.messages import HumanMessage, SystemMessage

        # Get agent's system prompt
        system_prompt = "You are a helpful AI assistant."
        if agent.active_prompt_version_id:
            prompt_result = await db.execute(
                select(AgentPromptVersion).where(
                    AgentPromptVersion.id == agent.active_prompt_version_id
                )
            )
            prompt_version = prompt_result.scalar_one_or_none()
            if prompt_version:
                system_prompt = prompt_version.system_prompt

        # Get LLM
        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="embed_chat",
            user_id=None,
            track_usage=True,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.message),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Generate audio if voice agent
        audio_url = None
        if agent.agent_mode in ["voice", "hybrid"] and agent.tts_config:
            try:
                from backend.services.audio.tts_service import TTSService, TTSProvider

                tts = TTSService()
                tts_config = agent.tts_config or {}
                provider = TTSProvider(tts_config.get("provider", "openai"))

                audio_bytes = await tts.synthesize_text(
                    text=response_text,
                    voice_id=tts_config.get("voice_id", "alloy"),
                    provider=provider,
                    speed=tts_config.get("speed", 1.0),
                )

                if audio_bytes:
                    import base64
                    audio_url = f"data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}"

            except Exception as e:
                logger.warning("Failed to generate audio for embed chat", error=str(e))

        return {
            "response": response_text,
            "conversation_id": request.conversation_id or str(uuid.uuid4()),
            "audio_url": audio_url,
            "agent_name": agent.name,
        }

    except Exception as e:
        logger.error("Embedded chat error", error=str(e), agent_id=str(agent.id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process message")


@router.post("/embed/{embed_token}/voice")
async def embedded_voice_chat(
    embed_token: str,
    request: EmbedChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Voice chat endpoint for embedded voice agent.
    Returns both text and audio response.
    """
    # Get the agent
    result = await db.execute(
        select(AgentDefinition).where(
            and_(
                AgentDefinition.embed_token == embed_token,
                AgentDefinition.is_published == True,
                AgentDefinition.is_active == True,
                AgentDefinition.agent_mode.in_(["voice", "hybrid"]),
            )
        )
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice agent not found or not published")

    try:
        from backend.services.llm import EnhancedLLMFactory
        from backend.services.audio.tts_service import TTSService, TTSProvider
        from langchain_core.messages import HumanMessage, SystemMessage

        # Get agent's system prompt
        system_prompt = "You are a helpful voice AI assistant. Keep responses concise and conversational."
        if agent.active_prompt_version_id:
            prompt_result = await db.execute(
                select(AgentPromptVersion).where(
                    AgentPromptVersion.id == agent.active_prompt_version_id
                )
            )
            prompt_version = prompt_result.scalar_one_or_none()
            if prompt_version:
                system_prompt = prompt_version.system_prompt

        # Get LLM
        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="embed_voice",
            user_id=None,
            track_usage=True,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.message),
        ]

        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Generate audio
        tts = TTSService()
        tts_config = agent.tts_config or {}
        provider = TTSProvider(tts_config.get("provider", "openai"))

        audio_bytes = await tts.synthesize_text(
            text=response_text,
            voice_id=tts_config.get("voice_id", "alloy"),
            provider=provider,
            speed=tts_config.get("speed", 1.0),
        )

        import base64
        audio_data = base64.b64encode(audio_bytes).decode() if audio_bytes else None

        return {
            "response": response_text,
            "audio_data": audio_data,
            "audio_format": "mp3",
            "conversation_id": request.conversation_id or str(uuid.uuid4()),
            "agent_name": agent.name,
            "tts_provider": tts_config.get("provider", "openai"),
        }

    except Exception as e:
        logger.error("Embedded voice chat error", error=str(e), agent_id=str(agent.id))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process voice message")


# =============================================================================
# Phase 59: Agent Evaluation & Personalization Endpoints
# =============================================================================

class RecordTrialRequest(BaseModel):
    """Request to record an evaluation trial."""
    query: str
    response: str
    ground_truth: Optional[str] = None
    latency_ms: float = 0.0
    retrieved_docs: int = 0
    correct_retrievals: int = 0
    metadata: Optional[Dict[str, Any]] = None


class RecordFeedbackRequest(BaseModel):
    """Request to record user feedback for personalization."""
    query: str
    response: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    explicit_feedback: Optional[str] = None
    follow_up_questions: int = 0
    response_was_edited: bool = False


@router.get("/agents/{agent_id}/evaluation")
async def get_agent_evaluation(
    agent_id: str,
    user: AuthenticatedUser,
):
    """
    Get comprehensive evaluation metrics for an agent.

    Returns Pass^k, progress rate, hallucination rate, invocation accuracy,
    and other performance metrics based on recorded trials.
    """
    try:
        from backend.services.agent_evaluation import get_agent_evaluator

        evaluator = await get_agent_evaluator(agent_id)
        result = await evaluator.get_full_evaluation()

        return {
            "agent_id": result.agent_id,
            "evaluation_id": result.evaluation_id,
            "trial_count": len(result.trials),
            "pass_rate": result.pass_rate,
            "metrics": result.metrics,
            "timestamp": result.timestamp.isoformat(),
            "recent_trials": [t.to_dict() for t in result.trials[-10:]],
        }

    except Exception as e:
        logger.error("Failed to get agent evaluation", error=str(e), agent_id=agent_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get evaluation: {str(e)}")


@router.post("/agents/{agent_id}/evaluation/trial")
async def record_evaluation_trial(
    agent_id: str,
    request: RecordTrialRequest,
    user: AuthenticatedUser,
):
    """
    Record a trial for agent evaluation.

    This endpoint is called after each agent interaction to track
    performance metrics like pass rate, latency, and hallucination detection.
    """
    try:
        from backend.services.agent_evaluation import get_agent_evaluator

        evaluator = await get_agent_evaluator(agent_id)
        trial = await evaluator.record_trial(
            query=request.query,
            response=request.response,
            ground_truth=request.ground_truth,
            latency_ms=request.latency_ms,
            retrieved_docs=request.retrieved_docs,
            correct_retrievals=request.correct_retrievals,
            metadata=request.metadata,
        )

        return {
            "trial_id": trial.trial_id,
            "passed": trial.passed,
            "score": trial.score,
            "hallucination_detected": trial.hallucination_detected,
            "timestamp": trial.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error("Failed to record trial", error=str(e), agent_id=agent_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to record trial: {str(e)}")


@router.get("/agents/{agent_id}/evaluation/pass-k")
async def get_pass_k_metric(
    agent_id: str,
    user: AuthenticatedUser,
    k: int = Query(3, ge=1, le=10),
):
    """
    Get Pass^k reliability metric for an agent.

    Pass^k measures the probability that at least one of k trials
    passes for each query. Target: >95% at k=3.
    """
    try:
        from backend.services.agent_evaluation import get_agent_evaluator

        evaluator = await get_agent_evaluator(agent_id)
        result = await evaluator.compute_pass_k(k=k)

        return {
            "k": result.k,
            "pass_rate": result.pass_rate,
            "trials_per_query": result.trials_per_query,
            "total_queries": result.total_queries,
            "confidence_interval": {
                "lower": result.confidence_interval[0],
                "upper": result.confidence_interval[1],
            },
            "target": 0.95,
            "meets_target": result.pass_rate >= 0.95,
        }

    except Exception as e:
        logger.error("Failed to compute Pass^k", error=str(e), agent_id=agent_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to compute Pass^k: {str(e)}")


@router.get("/agents/{agent_id}/personalization")
async def get_user_personalization(
    agent_id: str,
    user: AuthenticatedUser,
):
    """
    Get personalization settings learned from user interactions.

    Returns expertise level, communication style preferences,
    topic interests, and recommendations.
    """
    try:
        from backend.services.agent_evaluation import get_personalization_service

        service = await get_personalization_service(
            user_id=str(user.user_id),
            agent_id=agent_id,
        )

        preferences = service.preferences
        recommendations = await service.get_recommendations()

        return {
            "preferences": preferences.to_dict(),
            "recommendations": recommendations,
            "prompt_instructions": preferences.to_prompt_instructions(),
        }

    except Exception as e:
        logger.error("Failed to get personalization", error=str(e), agent_id=agent_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get personalization: {str(e)}")


@router.post("/agents/{agent_id}/personalization/feedback")
async def record_personalization_feedback(
    agent_id: str,
    request: RecordFeedbackRequest,
    user: AuthenticatedUser,
):
    """
    Record user feedback to improve personalization.

    This endpoint learns user preferences from interactions
    to adapt future responses.
    """
    try:
        from backend.services.agent_evaluation import get_personalization_service

        service = await get_personalization_service(
            user_id=str(user.user_id),
            agent_id=agent_id,
        )

        feedback = await service.record_interaction(
            query=request.query,
            response=request.response,
            rating=request.rating,
            explicit_feedback=request.explicit_feedback,
            follow_up_questions=request.follow_up_questions,
            response_was_edited=request.response_was_edited,
        )

        return {
            "interaction_id": feedback.interaction_id,
            "preferences_updated": True,
            "expertise_level": service.preferences.expertise_level.value,
            "communication_style": service.preferences.communication_style.value,
        }

    except Exception as e:
        logger.error("Failed to record feedback", error=str(e), agent_id=agent_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to record feedback: {str(e)}")
