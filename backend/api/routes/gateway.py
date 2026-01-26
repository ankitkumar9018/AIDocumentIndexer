"""
AIDocumentIndexer - LLM Gateway API Routes
===========================================

OpenAI-compatible API endpoints with budget enforcement.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import (
    get_async_session as get_db_session,
    get_current_user,
    get_current_organization_id,
)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    """Chat message."""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Messages")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=128000)
    stream: bool = Field(default=False)
    user: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class BudgetCreate(BaseModel):
    """Create a budget."""
    name: str = Field(..., max_length=255)
    period: str = Field(..., description="Budget period (daily, weekly, monthly)")
    limit_amount: float = Field(..., gt=0, description="Limit in USD")
    user_id: Optional[str] = None
    soft_limit_percent: float = Field(default=80.0, ge=0, le=100)
    is_hard_limit: bool = True


class BudgetUpdate(BaseModel):
    """Update a budget."""
    limit_amount: Optional[float] = Field(default=None, gt=0)
    soft_limit_percent: Optional[float] = Field(default=None, ge=0, le=100)
    is_hard_limit: Optional[bool] = None
    is_active: Optional[bool] = None


class BudgetResponse(BaseModel):
    """Budget response."""
    id: str
    name: str
    period: str
    limit_amount: float
    spent_amount: float
    remaining: float
    percent_used: float
    status: str
    reset_at: Optional[datetime]
    is_hard_limit: bool
    is_active: bool


class VirtualKeyCreate(BaseModel):
    """Create a virtual API key."""
    name: str = Field(..., max_length=255)
    user_id: Optional[str] = None
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)
    allowed_models: Optional[List[str]] = None
    max_tokens_per_request: Optional[int] = Field(default=None, ge=1)
    max_requests_per_minute: Optional[int] = Field(default=None, ge=1)
    max_requests_per_day: Optional[int] = Field(default=None, ge=1)
    max_cost_per_request: Optional[float] = Field(default=None, gt=0)


class VirtualKeyResponse(BaseModel):
    """Virtual key response (without the actual key)."""
    id: str
    name: str
    key_prefix: str
    expires_at: Optional[datetime]
    is_active: bool
    usage_count: int
    total_tokens: int
    total_cost: float
    last_used_at: Optional[datetime]
    created_at: datetime


class VirtualKeyCreatedResponse(VirtualKeyResponse):
    """Response when creating a new key (includes the actual key)."""
    key: str  # Only returned once


class UsageStatsResponse(BaseModel):
    """Usage statistics response."""
    period: Dict[str, str]
    summary: Dict[str, Any]
    by_model: List[Dict[str, Any]]
    by_provider: List[Dict[str, Any]]


# =============================================================================
# Chat Completions (OpenAI-compatible)
# =============================================================================

@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports:
    - Budget enforcement
    - Virtual API keys
    - Multi-provider routing
    - Streaming responses
    """
    from backend.services.llm_gateway.gateway import LLMGateway, GatewayRequest

    # Extract API key from Authorization header
    api_key = None
    if authorization and authorization.startswith("Bearer vk_"):
        api_key = authorization.replace("Bearer ", "")

    gateway = LLMGateway(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    gateway_request = GatewayRequest(
        model=request.model,
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        user=request.user,
    )

    if request.stream:
        async def stream_generator():
            try:
                async for chunk in gateway.complete_stream(gateway_request, api_key):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )
    else:
        try:
            response = await gateway.complete(gateway_request, api_key)
            return response.to_openai_format()
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/v1/models")
async def list_models(
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """List available models."""
    from backend.services.llm_gateway.gateway import LLMGateway

    gateway = LLMGateway(
        session=session,
        organization_id=organization_id,
        user_id=current_user.get("sub"),
    )

    models = await gateway.get_models()
    return {"object": "list", "data": models}


# =============================================================================
# Budget Management
# =============================================================================

@router.get("/budgets", response_model=List[BudgetResponse])
async def list_budgets(
    user_id: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """List all budgets."""
    from backend.services.llm_gateway.budget import BudgetManager

    manager = BudgetManager(
        session=session,
        organization_id=organization_id,
    )

    budgets = await manager.get_active_budgets(user_id=user_id)

    # Helper to safely get enum value (handles both enum and string)
    def safe_enum_value(val, default=""):
        if val is None:
            return default
        return val.value if hasattr(val, 'value') else val

    return [
        BudgetResponse(
            id=b.id,
            name=b.name,
            period=safe_enum_value(b.period),
            limit_amount=b.limit_amount,
            spent_amount=round(b.spent_amount, 4),
            remaining=round(b.remaining, 4),
            percent_used=round(b.percent_used, 1),
            status=safe_enum_value(b.status),
            reset_at=b.reset_at,
            is_hard_limit=b.is_hard_limit,
            is_active=b.is_active,
        )
        for b in budgets
    ]


@router.post("/budgets", response_model=BudgetResponse)
async def create_budget(
    data: BudgetCreate,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Create a new budget."""
    from backend.services.llm_gateway.budget import BudgetManager, BudgetPeriod

    manager = BudgetManager(
        session=session,
        organization_id=organization_id,
    )

    budget = await manager.create_budget(
        name=data.name,
        period=BudgetPeriod(data.period),
        limit_amount=data.limit_amount,
        user_id=data.user_id,
        soft_limit_percent=data.soft_limit_percent,
        is_hard_limit=data.is_hard_limit,
    )

    # Helper to safely get enum value (handles both enum and string)
    def safe_enum_value(val, default=""):
        if val is None:
            return default
        return val.value if hasattr(val, 'value') else val

    return BudgetResponse(
        id=budget.id,
        name=budget.name,
        period=safe_enum_value(budget.period),
        limit_amount=budget.limit_amount,
        spent_amount=round(budget.spent_amount, 4),
        remaining=round(budget.remaining, 4),
        percent_used=round(budget.percent_used, 1),
        status=safe_enum_value(budget.status),
        reset_at=budget.reset_at,
        is_hard_limit=budget.is_hard_limit,
        is_active=budget.is_active,
    )


@router.patch("/budgets/{budget_id}", response_model=BudgetResponse)
async def update_budget(
    budget_id: str,
    data: BudgetUpdate,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Update a budget."""
    from backend.services.llm_gateway.budget import BudgetManager

    manager = BudgetManager(
        session=session,
        organization_id=organization_id,
    )

    budget = await manager.update_budget(
        budget_id=budget_id,
        limit_amount=data.limit_amount,
        soft_limit_percent=data.soft_limit_percent,
        is_hard_limit=data.is_hard_limit,
        is_active=data.is_active,
    )

    # Helper to safely get enum value (handles both enum and string)
    def safe_enum_value(val, default=""):
        if val is None:
            return default
        return val.value if hasattr(val, 'value') else val

    return BudgetResponse(
        id=budget.id,
        name=budget.name,
        period=safe_enum_value(budget.period),
        limit_amount=budget.limit_amount,
        spent_amount=round(budget.spent_amount, 4),
        remaining=round(budget.remaining, 4),
        percent_used=round(budget.percent_used, 1),
        status=safe_enum_value(budget.status),
        reset_at=budget.reset_at,
        is_hard_limit=budget.is_hard_limit,
        is_active=budget.is_active,
    )


@router.delete("/budgets/{budget_id}")
async def delete_budget(
    budget_id: str,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Delete a budget."""
    from backend.services.llm_gateway.budget import BudgetManager

    manager = BudgetManager(
        session=session,
        organization_id=organization_id,
    )

    await manager.delete_budget(budget_id)
    return {"status": "deleted"}


@router.get("/budgets/summary")
async def get_budget_summary(
    user_id: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Get budget summary."""
    from backend.services.llm_gateway.budget import BudgetManager

    manager = BudgetManager(
        session=session,
        organization_id=organization_id,
    )

    return await manager.get_budget_summary(user_id=user_id)


# =============================================================================
# Virtual API Keys
# =============================================================================

@router.get("/keys", response_model=List[VirtualKeyResponse])
async def list_virtual_keys(
    user_id: Optional[str] = Query(default=None),
    include_inactive: bool = Query(default=False),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """List virtual API keys."""
    from backend.services.llm_gateway.virtual_keys import VirtualKeyManager

    manager = VirtualKeyManager(
        session=session,
        organization_id=organization_id,
    )

    keys = await manager.list_keys(
        user_id=user_id,
        include_inactive=include_inactive,
    )

    return [
        VirtualKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            expires_at=k.expires_at,
            is_active=k.is_active,
            usage_count=k.usage_count,
            total_tokens=k.total_tokens,
            total_cost=round(k.total_cost, 4),
            last_used_at=k.last_used_at,
            created_at=k.created_at,
        )
        for k in keys
    ]


@router.post("/keys", response_model=VirtualKeyCreatedResponse)
async def create_virtual_key(
    data: VirtualKeyCreate,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """
    Create a new virtual API key.

    The key is only returned once in this response.
    Store it securely as it cannot be retrieved later.
    """
    from backend.services.llm_gateway.virtual_keys import VirtualKeyManager

    manager = VirtualKeyManager(
        session=session,
        organization_id=organization_id,
    )

    key, plain_key = await manager.create_key(
        name=data.name,
        user_id=data.user_id,
        expires_in_days=data.expires_in_days,
        allowed_models=data.allowed_models,
        max_tokens_per_request=data.max_tokens_per_request,
        max_requests_per_minute=data.max_requests_per_minute,
        max_requests_per_day=data.max_requests_per_day,
        max_cost_per_request=data.max_cost_per_request,
    )

    return VirtualKeyCreatedResponse(
        id=key.id,
        name=key.name,
        key_prefix=key.key_prefix,
        key=plain_key,  # Only returned once!
        expires_at=key.expires_at,
        is_active=key.is_active,
        usage_count=key.usage_count,
        total_tokens=key.total_tokens,
        total_cost=key.total_cost,
        last_used_at=key.last_used_at,
        created_at=key.created_at,
    )


@router.post("/keys/{key_id}/revoke")
async def revoke_virtual_key(
    key_id: str,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Revoke a virtual API key."""
    from backend.services.llm_gateway.virtual_keys import VirtualKeyManager

    manager = VirtualKeyManager(
        session=session,
        organization_id=organization_id,
    )

    await manager.revoke_key(key_id)
    return {"status": "revoked"}


@router.post("/keys/{key_id}/rotate", response_model=VirtualKeyCreatedResponse)
async def rotate_virtual_key(
    key_id: str,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """
    Rotate a virtual API key.

    Creates a new key with the same settings and revokes the old one.
    """
    from backend.services.llm_gateway.virtual_keys import VirtualKeyManager

    manager = VirtualKeyManager(
        session=session,
        organization_id=organization_id,
    )

    new_key, plain_key = await manager.rotate_key(key_id)

    return VirtualKeyCreatedResponse(
        id=new_key.id,
        name=new_key.name,
        key_prefix=new_key.key_prefix,
        key=plain_key,
        expires_at=new_key.expires_at,
        is_active=new_key.is_active,
        usage_count=new_key.usage_count,
        total_tokens=new_key.total_tokens,
        total_cost=new_key.total_cost,
        last_used_at=new_key.last_used_at,
        created_at=new_key.created_at,
    )


@router.delete("/keys/{key_id}")
async def delete_virtual_key(
    key_id: str,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Permanently delete a virtual API key."""
    from backend.services.llm_gateway.virtual_keys import VirtualKeyManager

    manager = VirtualKeyManager(
        session=session,
        organization_id=organization_id,
    )

    await manager.delete_key(key_id)
    return {"status": "deleted"}


# =============================================================================
# Usage Analytics
# =============================================================================

@router.get("/usage/stats", response_model=UsageStatsResponse)
async def get_usage_stats(
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Get usage statistics."""
    from backend.services.llm_gateway.usage import UsageTracker

    tracker = UsageTracker(
        session=session,
        organization_id=organization_id,
    )

    return await tracker.get_usage_stats(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        model=model,
    )


@router.get("/usage/daily")
async def get_daily_usage(
    days: int = Query(default=30, ge=1, le=90),
    user_id: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Get daily usage for the past N days."""
    from backend.services.llm_gateway.usage import UsageTracker

    tracker = UsageTracker(
        session=session,
        organization_id=organization_id,
    )

    return await tracker.get_daily_usage(days=days, user_id=user_id)


@router.get("/usage/top-users")
async def get_top_users(
    days: int = Query(default=30, ge=1, le=90),
    limit: int = Query(default=10, ge=1, le=50),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Get top users by usage."""
    from backend.services.llm_gateway.usage import UsageTracker

    tracker = UsageTracker(
        session=session,
        organization_id=organization_id,
    )

    return await tracker.get_top_users(days=days, limit=limit)


@router.get("/usage/export")
async def export_usage(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    format: str = Query(default="json", regex="^(json|csv)$"),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Export usage data."""
    from backend.services.llm_gateway.usage import UsageTracker
    from fastapi.responses import Response

    tracker = UsageTracker(
        session=session,
        organization_id=organization_id,
    )

    data = await tracker.export_usage(
        start_date=start_date,
        end_date=end_date,
        format=format,
    )

    if format == "csv":
        return Response(
            content=data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=usage_{start_date.date()}_{end_date.date()}.csv"
            },
        )

    return data
