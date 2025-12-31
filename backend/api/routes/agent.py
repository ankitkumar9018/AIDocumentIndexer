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

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse

from backend.api.middleware.auth import AuthenticatedUser
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
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
            raise HTTPException(status_code=500, detail=update.get("error"))
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
        raise HTTPException(status_code=400, detail=result.get("error"))

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
        raise HTTPException(status_code=404, detail="Plan not found")

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

    result = await db.execute(
        select(AgentDefinition).order_by(AgentDefinition.agent_type)
    )
    agents = result.scalars().all()

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
        raise HTTPException(status_code=404, detail="Agent not found")

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
        raise HTTPException(status_code=404, detail="Agent not found")

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
            status_code=400,
            detail=f"Agent type '{request.agent_type}' already exists"
        )

    agent = AgentDefinition(
        name=request.name,
        agent_type=request.agent_type,
        description=request.description,
        default_temperature=request.default_temperature,
        max_tokens=request.max_tokens,
        settings=request.settings or {},
        default_provider_id=uuid.UUID(request.default_provider_id) if request.default_provider_id else None,
        default_model=request.default_model,
        is_active=True,
    )

    db.add(agent)
    await db.commit()
    await db.refresh(agent)

    logger.info("Created new agent", agent_id=str(agent.id), agent_type=request.agent_type)

    return {
        "id": str(agent.id),
        "name": agent.name,
        "agent_type": agent.agent_type,
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
    """
    from sqlalchemy import select

    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == uuid.UUID(agent_id))
    )
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

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

    await db.commit()

    logger.info("Updated agent", agent_id=agent_id)

    return {
        "success": True,
        "id": str(agent.id),
        "name": agent.name,
        "message": f"Agent '{agent.name}' updated successfully",
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
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_name = agent.name
    agent_type = agent.agent_type

    # Prevent deletion of core agents
    core_agents = {"manager", "generator", "critic", "research", "tool_executor"}
    if agent_type in core_agents and hard_delete:
        raise HTTPException(
            status_code=400,
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
        raise HTTPException(status_code=404, detail="Agent not found")

    # Create optimization job
    config = AgentConfig(
        agent_id=str(uuid.uuid4()),
        name="Prompt Builder",
        description="Prompt optimization agent",
    )
    builder = PromptBuilderAgent(config, db)
    job = await builder.create_optimization_job(agent)

    return {
        "success": True,
        "job_id": str(job.id),
        "agent_id": agent_id,
        "status": job.status,
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
        raise HTTPException(status_code=404, detail="Job not found")

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
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Job is not awaiting approval (status: {job.status})")

    if not job.winning_variant_id:
        raise HTTPException(status_code=400, detail="No winning variant selected")

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
        raise HTTPException(status_code=404, detail="Job not found")

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
        raise HTTPException(status_code=404, detail="Version not found")

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
        raise HTTPException(status_code=400, detail=result.get("error"))

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
        raise HTTPException(status_code=404, detail="Trajectory not found")

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
        raise HTTPException(status_code=404, detail="Trajectory not found")

    return {"success": True, "trajectory_id": trajectory_id}
