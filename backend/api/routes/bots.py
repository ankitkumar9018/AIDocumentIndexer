"""
AIDocumentIndexer - Bot Integration API Routes
================================================

REST API endpoints for managing bot integrations (Slack, Teams, etc.).

Endpoints:
- GET    /bots/connections           - List bot connections
- POST   /bots/connections           - Create new connection
- GET    /bots/connections/{id}      - Get connection details
- DELETE /bots/connections/{id}      - Delete connection
- POST   /bots/slack/events          - Slack events webhook
- POST   /bots/slack/commands        - Slack commands webhook
- GET    /bots/slack/oauth/callback  - Slack OAuth callback
"""

import json
import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_async_session, get_current_organization_id
from backend.db.models import BotConnection, BotPlatform
from backend.core.config import settings

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateBotConnectionRequest(BaseModel):
    """Request to create a new bot connection."""
    platform: str = Field(..., description="Bot platform (slack, teams, discord)")
    workspace_id: str = Field(..., description="External workspace/team ID")
    bot_token: str = Field(..., description="Bot authentication token")
    config: Optional[dict] = Field(None, description="Platform-specific configuration")


class BotConnectionResponse(BaseModel):
    """Response model for bot connection."""
    id: str
    platform: str
    workspace_id: str
    workspace_name: Optional[str]
    is_active: bool
    config: Optional[dict]
    installed_by_id: Optional[str]
    created_at: Optional[datetime]
    last_used_at: Optional[datetime]

    class Config:
        from_attributes = True


class BotConnectionListResponse(BaseModel):
    """Response for listing bot connections."""
    connections: List[BotConnectionResponse]
    total: int


# =============================================================================
# Helper Functions
# =============================================================================

def to_response(connection: BotConnection) -> BotConnectionResponse:
    """Convert BotConnection model to response."""
    return BotConnectionResponse(
        id=str(connection.id),
        platform=connection.platform.value if connection.platform else "unknown",
        workspace_id=connection.workspace_id or "",
        workspace_name=connection.workspace_name,
        is_active=connection.is_active,
        config=connection.config,
        installed_by_id=str(connection.installed_by_id) if connection.installed_by_id else None,
        created_at=connection.created_at,
        last_used_at=connection.last_used_at,
    )


# =============================================================================
# Connection Management Endpoints
# =============================================================================

@router.get("/connections", response_model=BotConnectionListResponse)
async def list_bot_connections(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """List all bot connections for the organization."""
    query = select(BotConnection).where(
        BotConnection.organization_id == organization_id
    )

    if platform:
        try:
            platform_enum = BotPlatform(platform)
            query = query.where(BotConnection.platform == platform_enum)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid platform: {platform}")

    result = await session.execute(query)
    connections = result.scalars().all()

    return BotConnectionListResponse(
        connections=[to_response(c) for c in connections],
        total=len(connections),
    )


@router.post("/connections", response_model=BotConnectionResponse)
async def create_bot_connection(
    request: CreateBotConnectionRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """Create a new bot connection manually."""
    try:
        platform_enum = BotPlatform(request.platform)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid platform: {request.platform}. Valid: {[p.value for p in BotPlatform]}",
        )

    # Check for existing connection
    existing = await session.execute(
        select(BotConnection).where(
            BotConnection.organization_id == organization_id,
            BotConnection.platform == platform_enum,
            BotConnection.workspace_id == request.workspace_id,
        )
    )

    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A connection for this workspace already exists",
        )

    connection = BotConnection(
        id=uuid.uuid4(),
        organization_id=organization_id,
        platform=platform_enum,
        workspace_id=request.workspace_id,
        bot_token=request.bot_token,  # Should be encrypted in production
        config=request.config,
        is_active=True,
        installed_by_id=uuid.UUID(current_user.get("sub")) if current_user.get("sub") else None,
    )

    session.add(connection)
    await session.commit()
    await session.refresh(connection)

    return to_response(connection)


@router.get("/connections/{connection_id}", response_model=BotConnectionResponse)
async def get_bot_connection(
    connection_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """Get a specific bot connection."""
    query = select(BotConnection).where(
        BotConnection.id == uuid.UUID(connection_id),
        BotConnection.organization_id == organization_id,
    )

    result = await session.execute(query)
    connection = result.scalar_one_or_none()

    if not connection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot connection not found")

    return to_response(connection)


@router.delete("/connections/{connection_id}")
async def delete_bot_connection(
    connection_id: str,
    session: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """Delete a bot connection."""
    query = select(BotConnection).where(
        BotConnection.id == uuid.UUID(connection_id),
        BotConnection.organization_id == organization_id,
    )

    result = await session.execute(query)
    connection = result.scalar_one_or_none()

    if not connection:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot connection not found")

    await session.delete(connection)
    await session.commit()

    return {"message": "Bot connection deleted", "id": connection_id}


# =============================================================================
# Slack Webhook Endpoints
# =============================================================================

@router.post("/slack/events")
async def handle_slack_events(request: Request):
    """
    Handle Slack Events API webhooks.

    This endpoint receives all Slack events (messages, mentions, etc.).
    """
    from backend.integrations.slack_bot.app import get_slack_bot

    slack_bot = get_slack_bot()

    if not slack_bot or not slack_bot.handler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Slack bot not configured",
        )

    # Forward to Bolt handler
    return await slack_bot.handler.handle(request)


@router.post("/slack/commands")
async def handle_slack_commands(request: Request):
    """
    Handle Slack slash command webhooks.

    This endpoint receives all slash command invocations.
    """
    from backend.integrations.slack_bot.app import get_slack_bot

    slack_bot = get_slack_bot()

    if not slack_bot or not slack_bot.handler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Slack bot not configured",
        )

    # Forward to Bolt handler
    return await slack_bot.handler.handle(request)


@router.get("/slack/oauth/callback")
async def slack_oauth_callback(
    code: str = Query(..., description="OAuth authorization code"),
    state: Optional[str] = Query(None, description="State parameter"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Handle Slack OAuth callback.

    Exchanges the authorization code for tokens and creates the bot connection.
    """
    import httpx

    client_id = getattr(settings, "SLACK_CLIENT_ID", None)
    client_secret = getattr(settings, "SLACK_CLIENT_SECRET", None)
    redirect_uri = getattr(settings, "SLACK_REDIRECT_URI", None)

    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Slack OAuth not configured",
        )

    try:
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://slack.com/api/oauth.v2.access",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )

            data = response.json()

            if not data.get("ok"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"OAuth failed: {data.get('error', 'Unknown error')}",
                )

        # Extract tokens and workspace info
        bot_token = data.get("access_token")
        team_info = data.get("team", {})
        workspace_id = team_info.get("id")
        workspace_name = team_info.get("name")

        # Parse organization ID from state if provided
        org_id = None
        user_id = None
        if state:
            try:
                state_data = json.loads(state)
                org_id = uuid.UUID(state_data.get("org_id"))
                user_id = uuid.UUID(state_data.get("user_id"))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass  # Invalid state data, will redirect to error page

        if not org_id:
            # Redirect to error page
            return RedirectResponse(
                url="/dashboard/settings?error=missing_organization",
            )

        # Create or update bot connection
        existing = await session.execute(
            select(BotConnection).where(
                BotConnection.organization_id == org_id,
                BotConnection.platform == BotPlatform.SLACK,
                BotConnection.workspace_id == workspace_id,
            )
        )

        connection = existing.scalar_one_or_none()

        if connection:
            connection.bot_token = bot_token
            connection.workspace_name = workspace_name
            connection.is_active = True
        else:
            connection = BotConnection(
                id=uuid.uuid4(),
                organization_id=org_id,
                platform=BotPlatform.SLACK,
                workspace_id=workspace_id,
                workspace_name=workspace_name,
                bot_token=bot_token,
                is_active=True,
                installed_by_id=user_id,
                config=data,
            )
            session.add(connection)

        await session.commit()

        # Redirect to success page
        return RedirectResponse(
            url=f"/dashboard/settings?slack_connected=true&workspace={workspace_name}",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth failed: {str(e)}",
        )


@router.get("/slack/oauth/install")
async def get_slack_install_url(
    current_user: dict = Depends(get_current_user),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
):
    """
    Get the Slack OAuth installation URL.

    Returns the URL to redirect users to for Slack app installation.
    """
    client_id = getattr(settings, "SLACK_CLIENT_ID", None)
    redirect_uri = getattr(settings, "SLACK_REDIRECT_URI", None)

    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Slack OAuth not configured",
        )

    # Create state parameter with org context
    state = json.dumps({
        "org_id": str(organization_id),
        "user_id": current_user.get("sub"),
    })

    scopes = [
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "chat:write",
        "commands",
        "files:read",
        "groups:history",
        "groups:read",
        "im:history",
        "im:read",
        "im:write",
        "mpim:history",
        "mpim:read",
        "reactions:read",
        "reactions:write",
        "users:read",
    ]

    install_url = (
        f"https://slack.com/oauth/v2/authorize"
        f"?client_id={client_id}"
        f"&scope={','.join(scopes)}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
    )

    return {"install_url": install_url}
