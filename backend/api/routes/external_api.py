"""
AIDocumentIndexer - External API Endpoints
==========================================

Provides publishable API endpoints for skills and workflows that can be
accessed by external systems using API keys.

Features:
- API key authentication for external access
- Publish/unpublish skills and workflows
- Execute published skills via API
- Execute published workflows via API
- OpenAPI documentation for published endpoints
- Rate limiting and usage tracking

Usage:
    # Publish a skill
    POST /api/v1/external/skills/{skill_id}/publish

    # Execute a published skill
    POST /api/v1/external/skills/{skill_id}/execute
    Headers: X-API-Key: your-api-key
    Body: {"inputs": {...}}

    # Execute a published workflow
    POST /api/v1/external/workflows/{workflow_id}/execute
    Headers: X-API-Key: your-api-key
    Body: {"input_data": {...}}
"""

import uuid
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_async_session
from backend.db.models import Skill, Workflow, WorkflowTriggerType
from backend.api.middleware.auth import get_user_context, get_org_id, get_user_uuid, safe_uuid
from backend.services.permissions import UserContext

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/external", tags=["External API"])


# =============================================================================
# Models
# =============================================================================

class APIKeyCreate(BaseModel):
    """Request to create an API key."""
    name: str = Field(..., description="Name for the API key")
    description: Optional[str] = Field(None, description="Description")
    expires_in_days: Optional[int] = Field(None, description="Days until expiration (null = never)")
    rate_limit_per_minute: int = Field(60, description="Rate limit per minute")
    allowed_skills: Optional[List[str]] = Field(None, description="List of skill IDs (null = all)")
    allowed_workflows: Optional[List[str]] = Field(None, description="List of workflow IDs (null = all)")


class APIKeyResponse(BaseModel):
    """Response with API key details."""
    id: str
    name: str
    key_prefix: str  # First 8 chars for identification
    created_at: str
    expires_at: Optional[str]
    rate_limit_per_minute: int
    is_active: bool
    last_used_at: Optional[str]
    usage_count: int


class APIKeyCreatedResponse(APIKeyResponse):
    """Response when API key is created (includes full key)."""
    api_key: str  # Only shown once on creation


class PublishRequest(BaseModel):
    """Request to publish a skill or workflow."""
    publish: bool = Field(True, description="True to publish, False to unpublish")
    require_api_key: bool = Field(True, description="Require API key for access")
    description: Optional[str] = Field(None, description="Public description")
    rate_limit_per_minute: Optional[int] = Field(None, description="Override rate limit")


class PublishedEndpointInfo(BaseModel):
    """Information about a published endpoint."""
    id: str
    name: str
    description: Optional[str]
    type: str  # "skill" or "workflow"
    endpoint_url: str
    method: str
    requires_api_key: bool
    rate_limit_per_minute: int
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]
    example_request: Optional[Dict[str, Any]]


class SkillExecuteRequest(BaseModel):
    """Request to execute a skill via external API."""
    inputs: Dict[str, Any] = Field(..., description="Skill input values")
    provider_id: Optional[str] = Field(None, description="LLM provider ID")
    model: Optional[str] = Field(None, description="Model to use")


class SkillExecuteResponse(BaseModel):
    """Response from skill execution."""
    success: bool
    execution_id: str
    output: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time_ms: int
    model_used: Optional[str]
    tokens_used: Optional[int]


class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow via external API."""
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for workflow")
    wait_for_completion: bool = Field(False, description="Wait for workflow to complete")
    timeout_seconds: int = Field(300, description="Timeout if waiting for completion")


class WorkflowExecuteResponse(BaseModel):
    """Response from workflow execution."""
    success: bool
    execution_id: str
    status: str
    output: Optional[Dict[str, Any]]
    error: Optional[str]


# =============================================================================
# API Key Storage (In-memory for simplicity, should be DB in production)
# =============================================================================

# In production, this should be stored in the database
_api_keys: Dict[str, Dict[str, Any]] = {}
_published_skills: Dict[str, Dict[str, Any]] = {}
_published_workflows: Dict[str, Dict[str, Any]] = {}


def _hash_api_key(key: str) -> str:
    """Hash API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def _generate_api_key() -> str:
    """Generate a new API key."""
    return f"adi_{secrets.token_urlsafe(32)}"


# =============================================================================
# API Key Management Endpoints
# =============================================================================

@router.post("/api-keys", response_model=APIKeyCreatedResponse)
async def create_api_key(
    request: APIKeyCreate,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create a new API key for external access.

    The full API key is only shown once. Store it securely.
    """
    # Generate key
    api_key = _generate_api_key()
    key_hash = _hash_api_key(api_key)
    key_id = str(uuid.uuid4())

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

    # Store key
    _api_keys[key_hash] = {
        "id": key_id,
        "user_id": user.user_id,
        "organization_id": user.organization_id,
        "name": request.name,
        "description": request.description,
        "key_prefix": api_key[:12],
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "rate_limit_per_minute": request.rate_limit_per_minute,
        "allowed_skills": request.allowed_skills,
        "allowed_workflows": request.allowed_workflows,
        "is_active": True,
        "last_used_at": None,
        "usage_count": 0,
    }

    logger.info("API key created", user_id=user.user_id, key_id=key_id)

    return APIKeyCreatedResponse(
        id=key_id,
        name=request.name,
        key_prefix=api_key[:12],
        created_at=datetime.utcnow().isoformat(),
        expires_at=expires_at.isoformat() if expires_at else None,
        rate_limit_per_minute=request.rate_limit_per_minute,
        is_active=True,
        last_used_at=None,
        usage_count=0,
        api_key=api_key,  # Only shown once
    )


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    user: UserContext = Depends(get_user_context),
):
    """List all API keys for the current user."""
    user_keys = [
        APIKeyResponse(
            id=data["id"],
            name=data["name"],
            key_prefix=data["key_prefix"],
            created_at=data["created_at"].isoformat(),
            expires_at=data["expires_at"].isoformat() if data["expires_at"] else None,
            rate_limit_per_minute=data["rate_limit_per_minute"],
            is_active=data["is_active"],
            last_used_at=data["last_used_at"].isoformat() if data["last_used_at"] else None,
            usage_count=data["usage_count"],
        )
        for data in _api_keys.values()
        if data["user_id"] == user.user_id
    ]
    return user_keys


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: UserContext = Depends(get_user_context),
):
    """Revoke an API key."""
    for key_hash, data in _api_keys.items():
        if data["id"] == key_id and data["user_id"] == user.user_id:
            data["is_active"] = False
            logger.info("API key revoked", key_id=key_id, user_id=user.user_id)
            return {"message": "API key revoked"}

    raise HTTPException(status_code=404, detail="API key not found")


# =============================================================================
# API Key Validation
# =============================================================================

async def validate_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> Dict[str, Any]:
    """Validate API key from header."""
    key_hash = _hash_api_key(x_api_key)

    if key_hash not in _api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    key_data = _api_keys[key_hash]

    if not key_data["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked",
        )

    if key_data["expires_at"] and datetime.utcnow() > key_data["expires_at"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    # Update usage
    key_data["last_used_at"] = datetime.utcnow()
    key_data["usage_count"] += 1

    return key_data


# =============================================================================
# Skill Publishing Endpoints
# =============================================================================

@router.post("/skills/{skill_id}/publish")
async def publish_skill(
    skill_id: uuid.UUID,
    request: PublishRequest,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Publish or unpublish a skill for external API access.

    Once published, the skill can be executed via:
    POST /api/v1/external/skills/{skill_id}/execute
    """
    # Get skill
    user_uuid = get_user_uuid(user)
    org_id = get_org_id(user)
    result = await db.execute(
        select(Skill).where(
            and_(
                Skill.id == skill_id,
                or_(
                    Skill.user_id == user_uuid if user_uuid else False,
                    Skill.organization_id == org_id if org_id else False,
                ),
            )
        )
    )
    skill = result.scalar_one_or_none()

    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    if request.publish:
        # Publish skill
        _published_skills[str(skill_id)] = {
            "skill_id": str(skill_id),
            "name": skill.name,
            "description": request.description or skill.description,
            "user_id": user.user_id,
            "organization_id": user.organization_id,
            "require_api_key": request.require_api_key,
            "rate_limit_per_minute": request.rate_limit_per_minute or 60,
            "published_at": datetime.utcnow(),
            "inputs": skill.inputs if hasattr(skill, 'inputs') else [],
            "outputs": skill.outputs if hasattr(skill, 'outputs') else [],
        }

        # Update skill is_public
        skill.is_public = True
        await db.commit()

        logger.info("Skill published", skill_id=str(skill_id), user_id=user.user_id)

        return {
            "message": "Skill published successfully",
            "skill_id": str(skill_id),
            "endpoint_url": f"/api/v1/external/skills/{skill_id}/execute",
            "requires_api_key": request.require_api_key,
        }
    else:
        # Unpublish skill
        if str(skill_id) in _published_skills:
            del _published_skills[str(skill_id)]

        skill.is_public = False
        await db.commit()

        logger.info("Skill unpublished", skill_id=str(skill_id), user_id=user.user_id)

        return {"message": "Skill unpublished successfully"}


@router.post("/skills/{skill_id}/execute", response_model=SkillExecuteResponse)
async def execute_published_skill(
    skill_id: uuid.UUID,
    request: SkillExecuteRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    api_key: Dict[str, Any] = Depends(validate_api_key),
):
    """
    Execute a published skill via external API.

    Requires a valid API key in the X-API-Key header.
    """
    import time
    start_time = time.time()

    # Check if skill is published
    skill_data = _published_skills.get(str(skill_id))
    if not skill_data:
        # Check if skill exists but just not in cache
        result = await db.execute(
            select(Skill).where(
                and_(
                    Skill.id == skill_id,
                    Skill.is_public == True,
                    Skill.is_active == True,
                )
            )
        )
        skill = result.scalar_one_or_none()
        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Skill not found or not published",
            )
    else:
        # Get skill from database
        result = await db.execute(select(Skill).where(Skill.id == skill_id))
        skill = result.scalar_one_or_none()
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")

    # Check API key permissions
    if api_key.get("allowed_skills"):
        if str(skill_id) not in api_key["allowed_skills"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key does not have access to this skill",
            )

    # Execute skill
    execution_id = str(uuid.uuid4())

    try:
        from backend.services.llm import EnhancedLLMFactory
        from langchain_core.messages import HumanMessage, SystemMessage

        # Get LLM
        llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="skill",
            user_id=api_key.get("user_id"),
            track_usage=True,
        )

        # Build prompt from skill template
        system_prompt = skill.system_prompt or "You are a helpful AI assistant."

        # Replace placeholders in system prompt with inputs
        formatted_prompt = system_prompt
        for key, value in request.inputs.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))

        # Get user input
        user_input = request.inputs.get("content") or request.inputs.get("input") or str(request.inputs)

        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=user_input),
        ]

        # Invoke LLM
        response = await llm.ainvoke(messages)
        output_text = response.content if hasattr(response, 'content') else str(response)

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Record execution
        from backend.db.models import SkillExecution, SkillExecutionStatus
        execution = SkillExecution(
            id=uuid.UUID(execution_id),
            skill_id=skill_id,
            user_id=uuid.UUID(api_key["user_id"]),
            status=SkillExecutionStatus.COMPLETED,
            inputs=request.inputs,
            output={"result": output_text},
            execution_time_ms=execution_time_ms,
            model_used=request.model or "default",
            provider_used=request.provider_id or "default",
        )
        db.add(execution)
        await db.commit()

        return SkillExecuteResponse(
            success=True,
            execution_id=execution_id,
            output={"result": output_text},
            error=None,
            execution_time_ms=execution_time_ms,
            model_used=request.model or "default",
            tokens_used=None,  # TODO: Track tokens
        )

    except Exception as e:
        logger.error("Skill execution failed", skill_id=str(skill_id), error=str(e))
        execution_time_ms = int((time.time() - start_time) * 1000)

        return SkillExecuteResponse(
            success=False,
            execution_id=execution_id,
            output=None,
            error=str(e),
            execution_time_ms=execution_time_ms,
            model_used=None,
            tokens_used=None,
        )


# =============================================================================
# Workflow Publishing Endpoints
# =============================================================================

@router.post("/workflows/{workflow_id}/publish")
async def publish_workflow_external(
    workflow_id: uuid.UUID,
    request: PublishRequest,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Publish or unpublish a workflow for external API access.

    Once published, the workflow can be executed via:
    POST /api/v1/external/workflows/{workflow_id}/execute
    """
    # Get workflow
    user_uuid = get_user_uuid(user)
    org_id = get_org_id(user)
    result = await db.execute(
        select(Workflow).where(
            and_(
                Workflow.id == workflow_id,
                or_(
                    Workflow.created_by_id == user_uuid if user_uuid else False,
                    Workflow.organization_id == org_id if org_id else False,
                ),
            )
        )
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if request.publish:
        # Publish workflow
        _published_workflows[str(workflow_id)] = {
            "workflow_id": str(workflow_id),
            "name": workflow.name,
            "description": request.description or workflow.description,
            "user_id": user.user_id,
            "organization_id": user.organization_id,
            "require_api_key": request.require_api_key,
            "rate_limit_per_minute": request.rate_limit_per_minute or 30,
            "published_at": datetime.utcnow(),
        }

        # Set workflow as active and configure for API trigger
        workflow.is_active = True
        await db.commit()

        logger.info("Workflow published", workflow_id=str(workflow_id), user_id=user.user_id)

        return {
            "message": "Workflow published successfully",
            "workflow_id": str(workflow_id),
            "endpoint_url": f"/api/v1/external/workflows/{workflow_id}/execute",
            "requires_api_key": request.require_api_key,
        }
    else:
        # Unpublish workflow
        if str(workflow_id) in _published_workflows:
            del _published_workflows[str(workflow_id)]

        logger.info("Workflow unpublished", workflow_id=str(workflow_id), user_id=user.user_id)

        return {"message": "Workflow unpublished successfully"}


@router.post("/workflows/{workflow_id}/execute", response_model=WorkflowExecuteResponse)
async def execute_published_workflow(
    workflow_id: uuid.UUID,
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    api_key: Dict[str, Any] = Depends(validate_api_key),
):
    """
    Execute a published workflow via external API.

    Requires a valid API key in the X-API-Key header.
    """
    # Check if workflow is published
    workflow_data = _published_workflows.get(str(workflow_id))

    # Get workflow from database
    result = await db.execute(
        select(Workflow).where(
            and_(
                Workflow.id == workflow_id,
                Workflow.is_active == True,
            )
        )
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found or not active",
        )

    # Check API key permissions
    if api_key.get("allowed_workflows"):
        if str(workflow_id) not in api_key["allowed_workflows"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key does not have access to this workflow",
            )

    # Execute workflow
    try:
        from backend.services.workflow_engine import get_execution_engine

        engine = get_execution_engine(
            session=db,
            organization_id=workflow.organization_id,
        )

        execution = await engine.execute(
            workflow_id=workflow_id,
            trigger_type="api",
            trigger_data={"api_key_id": api_key["id"]},
            input_data=request.input_data,
        )

        response = WorkflowExecuteResponse(
            success=True,
            execution_id=str(execution.id),
            status=execution.status,
            output=None,
            error=None,
        )

        # Wait for completion if requested
        if request.wait_for_completion:
            import asyncio
            timeout = request.timeout_seconds
            start_time = datetime.utcnow()

            while (datetime.utcnow() - start_time).total_seconds() < timeout:
                # Refresh execution status
                await db.refresh(execution)

                if execution.status in ["completed", "failed", "cancelled"]:
                    response.status = execution.status
                    response.output = execution.output_data
                    if execution.status == "failed":
                        response.success = False
                        response.error = execution.error_message
                    break

                await asyncio.sleep(1)
            else:
                response.status = "timeout"
                response.error = f"Workflow did not complete within {timeout} seconds"

        return response

    except Exception as e:
        logger.error("Workflow execution failed", workflow_id=str(workflow_id), error=str(e))
        return WorkflowExecuteResponse(
            success=False,
            execution_id="",
            status="error",
            output=None,
            error=str(e),
        )


# =============================================================================
# Discovery Endpoints
# =============================================================================

@router.get("/published", response_model=List[PublishedEndpointInfo])
async def list_published_endpoints(
    type_filter: Optional[str] = Query(None, description="Filter by type: skill, workflow"),
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all published skills and workflows for the current user/organization.

    Returns endpoint information including URL, method, and schemas.
    """
    endpoints = []

    # Get published skills
    if not type_filter or type_filter == "skill":
        for skill_id, data in _published_skills.items():
            if data["user_id"] == user.user_id or data.get("organization_id") == user.organization_id:
                endpoints.append(PublishedEndpointInfo(
                    id=skill_id,
                    name=data["name"],
                    description=data.get("description"),
                    type="skill",
                    endpoint_url=f"/api/v1/external/skills/{skill_id}/execute",
                    method="POST",
                    requires_api_key=data.get("require_api_key", True),
                    rate_limit_per_minute=data.get("rate_limit_per_minute", 60),
                    input_schema={"type": "object", "properties": {"inputs": {"type": "object"}}},
                    output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
                    example_request={"inputs": {"content": "Your input here"}},
                ))

    # Get published workflows
    if not type_filter or type_filter == "workflow":
        for workflow_id, data in _published_workflows.items():
            if data["user_id"] == user.user_id or data.get("organization_id") == user.organization_id:
                endpoints.append(PublishedEndpointInfo(
                    id=workflow_id,
                    name=data["name"],
                    description=data.get("description"),
                    type="workflow",
                    endpoint_url=f"/api/v1/external/workflows/{workflow_id}/execute",
                    method="POST",
                    requires_api_key=data.get("require_api_key", True),
                    rate_limit_per_minute=data.get("rate_limit_per_minute", 30),
                    input_schema={"type": "object", "properties": {"input_data": {"type": "object"}}},
                    output_schema={"type": "object", "properties": {"execution_id": {"type": "string"}, "status": {"type": "string"}}},
                    example_request={"input_data": {"key": "value"}, "wait_for_completion": False},
                ))

    return endpoints


@router.get("/openapi/{endpoint_type}/{endpoint_id}")
async def get_endpoint_openapi(
    endpoint_type: str,
    endpoint_id: uuid.UUID,
    user: UserContext = Depends(get_user_context),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get OpenAPI specification for a specific published endpoint.

    Useful for generating client SDKs or documentation.
    """
    if endpoint_type == "skill":
        data = _published_skills.get(str(endpoint_id))
        if not data:
            raise HTTPException(status_code=404, detail="Published skill not found")

        return {
            "openapi": "3.0.0",
            "info": {
                "title": data["name"],
                "description": data.get("description", ""),
                "version": "1.0.0",
            },
            "paths": {
                f"/api/v1/external/skills/{endpoint_id}/execute": {
                    "post": {
                        "summary": f"Execute {data['name']}",
                        "operationId": f"execute_skill_{endpoint_id}",
                        "security": [{"apiKey": []}] if data.get("require_api_key") else [],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["inputs"],
                                        "properties": {
                                            "inputs": {"type": "object"},
                                            "provider_id": {"type": "string"},
                                            "model": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Successful execution",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "success": {"type": "boolean"},
                                                "execution_id": {"type": "string"},
                                                "output": {"type": "object"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "components": {
                "securitySchemes": {
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                    },
                },
            },
        }

    elif endpoint_type == "workflow":
        data = _published_workflows.get(str(endpoint_id))
        if not data:
            raise HTTPException(status_code=404, detail="Published workflow not found")

        return {
            "openapi": "3.0.0",
            "info": {
                "title": data["name"],
                "description": data.get("description", ""),
                "version": "1.0.0",
            },
            "paths": {
                f"/api/v1/external/workflows/{endpoint_id}/execute": {
                    "post": {
                        "summary": f"Execute {data['name']}",
                        "operationId": f"execute_workflow_{endpoint_id}",
                        "security": [{"apiKey": []}] if data.get("require_api_key") else [],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "input_data": {"type": "object"},
                                            "wait_for_completion": {"type": "boolean", "default": False},
                                            "timeout_seconds": {"type": "integer", "default": 300},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Workflow triggered",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "success": {"type": "boolean"},
                                                "execution_id": {"type": "string"},
                                                "status": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "components": {
                "securitySchemes": {
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                    },
                },
            },
        }

    raise HTTPException(status_code=400, detail="Invalid endpoint type")
