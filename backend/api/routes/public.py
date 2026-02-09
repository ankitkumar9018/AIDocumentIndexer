"""
AIDocumentIndexer - Public API Routes
======================================

Public endpoints for published Skills and Workflows that can be accessed
without authentication. These are used for embedding and external integrations.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.services.llm import LLMFactory, llm_config
from backend.core.config import settings
from backend.db.database import get_async_session
from backend.db.models import (
    Skill,
    SkillExecution,
    SkillExecutionStatus,
    Workflow,
    WorkflowExecution,
    WorkflowTriggerType,
    WorkflowStatus,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Rate Limiting (Simple in-memory for now)
# =============================================================================

_rate_limit_cache: Dict[str, List[datetime]] = {}


def check_rate_limit(key: str, limit: int = 100, window_seconds: int = 60) -> bool:
    """Check if request is within rate limit. Returns True if allowed."""
    now = datetime.utcnow()
    window_start = now.timestamp() - window_seconds

    if key not in _rate_limit_cache:
        _rate_limit_cache[key] = []

    # Clean old entries
    _rate_limit_cache[key] = [
        ts for ts in _rate_limit_cache[key]
        if ts.timestamp() > window_start
    ]

    if len(_rate_limit_cache[key]) >= limit:
        return False

    _rate_limit_cache[key].append(now)
    return True


def check_cors(request: Request, allowed_domains: List[str]) -> bool:
    """Check if request origin is allowed."""
    if "*" in allowed_domains:
        return True

    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")

    for domain in allowed_domains:
        if domain in origin or domain in referer:
            return True

    return not origin and not referer  # Allow direct API calls


# =============================================================================
# Public Skill Endpoints
# =============================================================================

class PublicSkillExecuteRequest(BaseModel):
    """Request to execute a published skill."""
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    api_key: Optional[str] = Field(None, description="API key if required")


class PublicSkillResponse(BaseModel):
    """Public skill information."""
    id: str
    name: str
    description: Optional[str]
    category: str
    icon: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    branding: Optional[Dict[str, Any]]


@router.get("/skills/{public_slug}")
async def get_public_skill(
    public_slug: str,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get public skill information by slug.

    Returns the skill definition including inputs required for execution.
    """
    result = await db.execute(
        select(Skill).where(
            Skill.public_slug == public_slug,
            Skill.is_published == True,
            Skill.is_active == True,
        )
    )
    skill = result.scalar_one_or_none()

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Skill not found or not published",
        )

    # Check CORS
    allowed_domains = (skill.publish_config or {}).get("allowed_domains", ["*"])
    if not check_cors(request, allowed_domains):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not allowed",
        )

    return PublicSkillResponse(
        id=str(skill.id),
        name=skill.name,
        description=skill.description,
        category=skill.category,
        icon=skill.icon,
        inputs=skill.inputs or [],
        outputs=skill.outputs or [],
        branding=(skill.publish_config or {}).get("branding"),
    )


@router.post("/skills/{public_slug}/execute")
async def execute_public_skill(
    public_slug: str,
    execute_request: PublicSkillExecuteRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute a published skill without authentication.

    Rate limiting and CORS are enforced based on publish_config.
    """
    start_time = datetime.utcnow()

    # Get skill
    result = await db.execute(
        select(Skill).where(
            Skill.public_slug == public_slug,
            Skill.is_published == True,
            Skill.is_active == True,
        )
    )
    skill = result.scalar_one_or_none()

    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Skill not found or not published",
        )

    config = skill.publish_config or {}

    # Check CORS
    allowed_domains = config.get("allowed_domains", ["*"])
    if not check_cors(request, allowed_domains):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not allowed",
        )

    # Check API key if required
    if config.get("require_api_key"):
        if not execute_request.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
            )
        # TODO: Implement API key validation

    # Check rate limit
    rate_limit = config.get("rate_limit", 100)
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"skill:{public_slug}:{client_ip}"

    if not check_rate_limit(rate_key, rate_limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    # Create anonymous execution record
    execution = SkillExecution(
        skill_id=skill.id,
        user_id=skill.user_id,  # Use skill owner as executor for anonymous
        status=SkillExecutionStatus.RUNNING.value,
        inputs=execute_request.inputs,
    )
    db.add(execution)
    await db.flush()

    # Check if external agent skill
    if skill.system_prompt.startswith("__EXTERNAL_AGENT_CONFIG__:"):
        # Handle external agent execution (simplified)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="External agent skills not supported for public execution yet",
        )

    # Format the prompt with inputs
    try:
        format_kwargs = {}
        inputs = execute_request.inputs

        # Map common input names
        if "document" in inputs:
            doc = inputs["document"]
            format_kwargs["content"] = doc.get("content", "") if isinstance(doc, dict) else str(doc)
        if "text" in inputs:
            format_kwargs["content"] = inputs["text"]
        if "content" in inputs:
            format_kwargs["content"] = inputs["content"]
        if "length" in inputs:
            format_kwargs["length"] = inputs["length"]
        if "target_language" in inputs:
            format_kwargs["target_language"] = inputs["target_language"]
        if "document1" in inputs:
            doc1 = inputs["document1"]
            format_kwargs["document1"] = doc1.get("content", "") if isinstance(doc1, dict) else str(doc1)
        if "document2" in inputs:
            doc2 = inputs["document2"]
            format_kwargs["document2"] = doc2.get("content", "") if isinstance(doc2, dict) else str(doc2)

        formatted_prompt = skill.system_prompt.format(
            **{k: v for k, v in format_kwargs.items() if f"{{{k}}}" in skill.system_prompt}
        )
    except KeyError as e:
        execution.status = SkillExecutionStatus.FAILED.value
        execution.error_message = f"Missing required input: {e}"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required input: {e}",
        )

    # Execute with LLM
    try:
        llm = LLMFactory.get_chat_model(
            provider=None,  # Use system-configured default provider
            model=None,     # Use provider's default model
            temperature=0.7,
            max_tokens=2048,
        )

        response = await llm.ainvoke(formatted_prompt)
        output = response.content

        # Try to parse JSON responses
        if output.strip().startswith("{") or output.strip().startswith("["):
            try:
                import json
                output = json.loads(output)
            except:
                pass

        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Update execution record
        execution.status = SkillExecutionStatus.COMPLETED.value
        execution.output = {"result": output} if not isinstance(output, dict) else output
        execution.execution_time_ms = execution_time
        execution.model_used = llm_config.default_chat_model
        execution.provider_used = llm_config.default_provider

        # Update skill stats
        skill.use_count += 1
        skill.last_used_at = datetime.utcnow()

        await db.commit()

        logger.info(
            "Public skill executed",
            skill_id=str(skill.id),
            public_slug=public_slug,
            execution_time_ms=execution_time,
        )

        return {
            "output": output,
            "execution_time_ms": execution_time,
            "execution_id": str(execution.id),
        }

    except Exception as e:
        execution.status = SkillExecutionStatus.FAILED.value
        execution.error_message = str(e)
        await db.commit()

        logger.error(
            "Public skill execution failed",
            skill_id=str(skill.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}",
        )


# =============================================================================
# Public Workflow Endpoints
# =============================================================================

class PublicWorkflowResponse(BaseModel):
    """Public workflow information."""
    id: str
    name: str
    description: Optional[str]
    input_schema: List[Dict[str, Any]]
    branding: Optional[Dict[str, Any]]


class PublicWorkflowExecuteRequest(BaseModel):
    """Request to execute a published workflow."""
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    api_key: Optional[str] = Field(None, description="API key if required")


@router.get("/workflows/{public_slug}")
async def get_public_workflow(
    public_slug: str,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get public workflow information by slug.

    Returns the workflow definition including inputs required for execution.
    """
    result = await db.execute(
        select(Workflow).where(
            Workflow.public_slug == public_slug,
            Workflow.is_published == True,
            Workflow.is_active == True,
        )
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found or not published",
        )

    # Check CORS
    config = workflow.publish_config or {}
    allowed_domains = config.get("allowed_domains", ["*"])
    if not check_cors(request, allowed_domains):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not allowed",
        )

    # Get input schema from publish config or workflow config
    input_schema = config.get("input_schema", [])

    return PublicWorkflowResponse(
        id=str(workflow.id),
        name=workflow.name,
        description=workflow.description,
        input_schema=input_schema,
        branding=config.get("branding"),
    )


@router.post("/workflows/{public_slug}/execute")
async def execute_public_workflow(
    public_slug: str,
    execute_request: PublicWorkflowExecuteRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute a published workflow without authentication.

    Returns the execution ID that can be used to check status.
    """
    from backend.services.workflow_engine import get_execution_engine

    # Get workflow
    result = await db.execute(
        select(Workflow).where(
            Workflow.public_slug == public_slug,
            Workflow.is_published == True,
            Workflow.is_active == True,
        )
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found or not published",
        )

    config = workflow.publish_config or {}

    # Check CORS
    allowed_domains = config.get("allowed_domains", ["*"])
    if not check_cors(request, allowed_domains):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin not allowed",
        )

    # Check API key if required
    if config.get("require_api_key"):
        if not execute_request.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
            )

    # Check rate limit
    rate_limit = config.get("rate_limit", 100)
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"workflow:{public_slug}:{client_ip}"

    if not check_rate_limit(rate_key, rate_limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    # Execute workflow
    engine = get_execution_engine(
        session=db,
        organization_id=workflow.organization_id,
    )

    try:
        execution = await engine.execute(
            workflow_id=workflow.id,
            trigger_type=WorkflowTriggerType.FORM.value,  # Form trigger for public
            trigger_data={"source": "public_api", "slug": public_slug},
            input_data=execute_request.inputs,
            triggered_by_id=workflow.created_by_id,  # Use creator for anonymous
        )

        return {
            "execution_id": str(execution.id),
            "status": execution.status,
            "message": "Workflow execution started",
        }

    except Exception as e:
        logger.error("Public workflow execution failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}",
        )


@router.get("/workflows/{public_slug}/status/{execution_id}")
async def get_public_workflow_status(
    public_slug: str,
    execution_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """Get the status of a public workflow execution."""
    # Verify workflow exists and is published
    wf_result = await db.execute(
        select(Workflow).where(
            Workflow.public_slug == public_slug,
            Workflow.is_published == True,
        )
    )
    workflow = wf_result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    # Get execution
    try:
        exec_uuid = UUID(execution_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid execution ID",
        )

    exec_result = await db.execute(
        select(WorkflowExecution).where(
            WorkflowExecution.id == exec_uuid,
            WorkflowExecution.workflow_id == workflow.id,
        )
    )
    execution = exec_result.scalar_one_or_none()

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found",
        )

    return {
        "execution_id": str(execution.id),
        "status": execution.status,
        "started_at": execution.started_at.isoformat() if execution.started_at else None,
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "output_data": execution.output_data if execution.status == WorkflowStatus.COMPLETED.value else None,
        "error_message": execution.error_message if execution.status == WorkflowStatus.FAILED.value else None,
    }


# =============================================================================
# Shared Workflow Endpoints (via share links)
# =============================================================================


class SharedWorkflowVerifyRequest(BaseModel):
    """Request to verify shared workflow password."""
    password: str


class SharedWorkflowExecuteRequest(BaseModel):
    """Request to execute a shared workflow."""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    password: Optional[str] = None


@router.get("/shared/workflow/{token}")
async def get_shared_workflow(
    token: str,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get shared workflow information by share token.

    Returns workflow details and input schema for the share link.
    """
    from sqlalchemy import text

    # Get share link
    result = await db.execute(
        text("""
            SELECT id, resource_id, permission_level, password_hash,
                   expires_at, max_uses, use_count, is_active
            FROM share_links
            WHERE token = :token AND resource_type = 'workflow'
        """),
        {"token": token},
    )
    share = result.fetchone()

    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found or expired",
        )

    share_id, resource_id, permission_level, password_hash, expires_at, max_uses, use_count, is_active = share

    # Check if link is active
    if not is_active:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="This share link has been revoked",
        )

    # Check expiration
    if expires_at and datetime.utcnow() > expires_at:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="This share link has expired",
        )

    # Check max uses
    if max_uses and use_count >= max_uses:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="This share link has reached its maximum uses",
        )

    # Get workflow
    wf_result = await db.execute(
        select(Workflow).where(Workflow.id == UUID(resource_id))
    )
    workflow = wf_result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found",
        )

    # Get input schema from trigger config or publish config
    input_schema = []
    if workflow.trigger_config and "input_schema" in workflow.trigger_config:
        input_schema = workflow.trigger_config["input_schema"]
    elif workflow.publish_config and "input_schema" in workflow.publish_config:
        input_schema = workflow.publish_config["input_schema"]

    return {
        "workflow_id": str(workflow.id),
        "workflow_name": workflow.name,
        "workflow_description": workflow.description,
        "permission_level": permission_level,
        "input_schema": input_schema,
        "requires_password": password_hash is not None,
        "is_valid": True,
    }


@router.post("/shared/workflow/{token}/verify")
async def verify_shared_workflow_password(
    token: str,
    verify_request: SharedWorkflowVerifyRequest,
    db: AsyncSession = Depends(get_async_session),
):
    """Verify password for a password-protected share link."""
    import hashlib
    from sqlalchemy import text

    result = await db.execute(
        text("""
            SELECT password_hash FROM share_links
            WHERE token = :token AND resource_type = 'workflow' AND is_active = true
        """),
        {"token": token},
    )
    share = result.fetchone()

    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found",
        )

    password_hash = share[0]

    if not password_hash:
        return {"verified": True}

    # Verify password
    provided_hash = hashlib.sha256(verify_request.password.encode()).hexdigest()

    if provided_hash != password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    return {"verified": True}


@router.post("/shared/workflow/{token}/execute")
async def execute_shared_workflow(
    token: str,
    execute_request: SharedWorkflowExecuteRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Execute a shared workflow.

    Requires executor or editor permission level.
    """
    import hashlib
    from sqlalchemy import text
    from backend.services.workflow_engine import get_execution_engine

    # Get share link
    result = await db.execute(
        text("""
            SELECT id, resource_id, permission_level, password_hash,
                   expires_at, max_uses, use_count, is_active, organization_id
            FROM share_links
            WHERE token = :token AND resource_type = 'workflow'
        """),
        {"token": token},
    )
    share = result.fetchone()

    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found",
        )

    share_id, resource_id, permission_level, password_hash, expires_at, max_uses, use_count, is_active, org_id = share

    # Validate share link
    if not is_active:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Share link revoked")

    if expires_at and datetime.utcnow() > expires_at:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Share link expired")

    if max_uses and use_count >= max_uses:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Max uses reached")

    # Check permission
    if permission_level == "viewer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This share link only allows viewing, not execution",
        )

    # Verify password if required
    if password_hash:
        if not execute_request.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Password required",
            )
        provided_hash = hashlib.sha256(execute_request.password.encode()).hexdigest()
        if provided_hash != password_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password",
            )

    # Get workflow
    wf_result = await db.execute(
        select(Workflow).where(Workflow.id == UUID(resource_id))
    )
    workflow = wf_result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")

    if not workflow.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Workflow is not active")

    # Increment use count
    await db.execute(
        text("""
            UPDATE share_links SET use_count = use_count + 1, updated_at = :now
            WHERE id = :share_id
        """),
        {"share_id": str(share_id), "now": datetime.utcnow()},
    )

    # Execute workflow
    try:
        engine = get_execution_engine(
            session=db,
            organization_id=UUID(org_id) if org_id else workflow.organization_id,
        )

        execution = await engine.execute(
            workflow_id=workflow.id,
            trigger_type=WorkflowTriggerType.FORM.value,
            trigger_data={
                "source": "share_link",
                "token": token,
                "share_id": str(share_id),
            },
            input_data=execute_request.inputs,
            triggered_by_id=workflow.created_by_id,
        )

        return {
            "execution_id": str(execution.id),
            "status": execution.status,
            "message": "Workflow execution started",
        }

    except Exception as e:
        logger.error("Shared workflow execution failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}",
        )


@router.get("/shared/workflow/{token}/status/{execution_id}")
async def get_shared_workflow_execution_status(
    token: str,
    execution_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """Get execution status for a shared workflow."""
    from sqlalchemy import text

    # Verify share link exists
    result = await db.execute(
        text("""
            SELECT resource_id FROM share_links
            WHERE token = :token AND resource_type = 'workflow' AND is_active = true
        """),
        {"token": token},
    )
    share = result.fetchone()

    if not share:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share link not found")

    workflow_id = share[0]

    # Get execution
    try:
        exec_uuid = UUID(execution_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid execution ID")

    exec_result = await db.execute(
        select(WorkflowExecution).where(
            WorkflowExecution.id == exec_uuid,
            WorkflowExecution.workflow_id == UUID(workflow_id),
        )
    )
    execution = exec_result.scalar_one_or_none()

    if not execution:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")

    return {
        "execution_id": str(execution.id),
        "status": execution.status,
        "started_at": execution.started_at.isoformat() if execution.started_at else None,
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "output_data": execution.output_data if execution.status == WorkflowStatus.COMPLETED.value else None,
        "error_message": execution.error_message if execution.status == WorkflowStatus.FAILED.value else None,
    }
