"""
AIDocumentIndexer - Connector API Routes
=========================================

API endpoints for managing external data source connectors.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import (
    get_async_session as get_db_session,
    get_current_user,
    get_current_organization_id,
)
from backend.db.models import ConnectorInstance

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class ConnectorTypeInfo(BaseModel):
    """Information about a connector type."""
    type: str
    display_name: str
    description: str
    icon: str
    config_schema: Dict[str, Any]
    credentials_schema: Dict[str, Any]


class ConnectorInstanceCreate(BaseModel):
    """Create a new connector instance."""
    connector_type: str = Field(..., description="Type of connector (google_drive, notion, etc.)")
    name: str = Field(..., max_length=255)
    credentials: Dict[str, Any] = Field(default_factory=dict)
    sync_config: Optional[Dict[str, Any]] = Field(default=None)


class ConnectorInstanceUpdate(BaseModel):
    """Update a connector instance."""
    name: Optional[str] = None
    sync_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ConnectorInstanceResponse(BaseModel):
    """Connector instance response."""
    id: str
    connector_type: str
    name: str
    status: str
    is_active: bool
    last_sync_at: Optional[datetime]
    next_sync_at: Optional[datetime]
    error_message: Optional[str]
    sync_config: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class SyncJobResponse(BaseModel):
    """Sync job response."""
    id: str
    connector_instance_id: str
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    resources_synced: int
    resources_failed: int
    error_message: Optional[str]


class OAuthUrlResponse(BaseModel):
    """OAuth URL response."""
    url: str
    state: str


# =============================================================================
# Connector Type Endpoints
# =============================================================================

@router.get("/types", response_model=List[ConnectorTypeInfo])
async def list_connector_types():
    """List available connector types."""
    from backend.services.connectors.registry import ConnectorRegistry
    from backend.services.connectors.base import ConnectorType

    types = []
    for connector_type in ConnectorType:
        connector_class = ConnectorRegistry.get(connector_type)
        if connector_class:
            types.append(ConnectorTypeInfo(
                type=connector_type.value,
                display_name=getattr(connector_class, "display_name", connector_type.value),
                description=getattr(connector_class, "description", ""),
                icon=getattr(connector_class, "icon", "link"),
                config_schema=connector_class.get_config_schema() if hasattr(connector_class, "get_config_schema") else {},
                credentials_schema=connector_class.get_credentials_schema() if hasattr(connector_class, "get_credentials_schema") else {},
            ))

    return types


# =============================================================================
# Connector Instance Endpoints
# =============================================================================

@router.get("", response_model=List[ConnectorInstanceResponse])
async def list_connectors(
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """List all connector instances for the organization."""
    result = await session.execute(
        select(ConnectorInstance)
        .where(ConnectorInstance.organization_id == organization_id)
        .order_by(ConnectorInstance.created_at.desc())
    )
    connectors = result.scalars().all()

    return [
        ConnectorInstanceResponse(
            id=str(c.id),
            connector_type=c.connector_type,
            name=c.name,
            status=c.status or "inactive",
            is_active=c.is_active,
            last_sync_at=c.last_sync_at,
            next_sync_at=c.next_sync_at,
            error_message=c.error_message,
            sync_config=c.sync_config,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in connectors
    ]


@router.post("", response_model=ConnectorInstanceResponse)
async def create_connector(
    data: ConnectorInstanceCreate,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Create a new connector instance."""
    from backend.services.connectors.registry import ConnectorRegistry
    from backend.services.connectors.base import ConnectorType

    # Validate connector type
    try:
        connector_type = ConnectorType(data.connector_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown connector type: {data.connector_type}")

    connector_class = ConnectorRegistry.get(connector_type)
    if not connector_class:
        raise HTTPException(status_code=400, detail=f"Connector type not implemented: {data.connector_type}")

    # Create instance
    connector = ConnectorInstance(
        id=uuid.uuid4(),
        organization_id=organization_id,
        connector_type=data.connector_type,
        name=data.name,
        credentials=data.credentials,
        sync_config=data.sync_config,
        status="pending",
        is_active=False,
        created_by_id=uuid.UUID(current_user.get("sub")) if current_user.get("sub") else None,
    )

    session.add(connector)
    await session.commit()
    await session.refresh(connector)

    return ConnectorInstanceResponse(
        id=str(connector.id),
        connector_type=connector.connector_type,
        name=connector.name,
        status=connector.status,
        is_active=connector.is_active,
        last_sync_at=connector.last_sync_at,
        next_sync_at=connector.next_sync_at,
        error_message=connector.error_message,
        sync_config=connector.sync_config,
        created_at=connector.created_at,
        updated_at=connector.updated_at,
    )


@router.get("/{connector_id}", response_model=ConnectorInstanceResponse)
async def get_connector(
    connector_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Get a specific connector instance."""
    result = await session.execute(
        select(ConnectorInstance)
        .where(
            ConnectorInstance.id == connector_id,
            ConnectorInstance.organization_id == organization_id,
        )
    )
    connector = result.scalar_one_or_none()

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    return ConnectorInstanceResponse(
        id=str(connector.id),
        connector_type=connector.connector_type,
        name=connector.name,
        status=connector.status,
        is_active=connector.is_active,
        last_sync_at=connector.last_sync_at,
        next_sync_at=connector.next_sync_at,
        error_message=connector.error_message,
        sync_config=connector.sync_config,
        created_at=connector.created_at,
        updated_at=connector.updated_at,
    )


@router.patch("/{connector_id}", response_model=ConnectorInstanceResponse)
async def update_connector(
    connector_id: uuid.UUID,
    data: ConnectorInstanceUpdate,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Update a connector instance."""
    result = await session.execute(
        select(ConnectorInstance)
        .where(
            ConnectorInstance.id == connector_id,
            ConnectorInstance.organization_id == organization_id,
        )
    )
    connector = result.scalar_one_or_none()

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    if data.name is not None:
        connector.name = data.name
    if data.sync_config is not None:
        connector.sync_config = data.sync_config
    if data.is_active is not None:
        connector.is_active = data.is_active

    connector.updated_at = datetime.utcnow()
    await session.commit()
    await session.refresh(connector)

    return ConnectorInstanceResponse(
        id=str(connector.id),
        connector_type=connector.connector_type,
        name=connector.name,
        status=connector.status,
        is_active=connector.is_active,
        last_sync_at=connector.last_sync_at,
        next_sync_at=connector.next_sync_at,
        error_message=connector.error_message,
        sync_config=connector.sync_config,
        created_at=connector.created_at,
        updated_at=connector.updated_at,
    )


@router.delete("/{connector_id}")
async def delete_connector(
    connector_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Delete a connector instance."""
    result = await session.execute(
        select(ConnectorInstance)
        .where(
            ConnectorInstance.id == connector_id,
            ConnectorInstance.organization_id == organization_id,
        )
    )
    connector = result.scalar_one_or_none()

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    await session.delete(connector)
    await session.commit()

    return {"status": "deleted", "id": str(connector_id)}


# =============================================================================
# OAuth Endpoints
# =============================================================================

@router.get("/{connector_id}/oauth/url", response_model=OAuthUrlResponse)
async def get_oauth_url(
    connector_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Get OAuth authorization URL for a connector."""
    from backend.services.connectors.registry import ConnectorRegistry
    from backend.services.connectors.base import ConnectorType, ConnectorConfig

    result = await session.execute(
        select(ConnectorInstance)
        .where(
            ConnectorInstance.id == connector_id,
            ConnectorInstance.organization_id == organization_id,
        )
    )
    connector_instance = result.scalar_one_or_none()

    if not connector_instance:
        raise HTTPException(status_code=404, detail="Connector not found")

    connector_class = ConnectorRegistry.get(ConnectorType(connector_instance.connector_type))
    if not connector_class:
        raise HTTPException(status_code=400, detail="Unknown connector type")

    # Create connector instance
    connector = connector_class(
        config=ConnectorConfig(
            credentials=connector_instance.credentials or {},
            settings=connector_instance.sync_config or {},
        ),
        session=session,
        organization_id=organization_id,
    )

    # Generate state token (includes connector_id for callback)
    import secrets
    state = f"{connector_id}:{secrets.token_urlsafe(32)}"

    if hasattr(connector, "get_oauth_url"):
        url = connector.get_oauth_url(state)
        if url:
            return OAuthUrlResponse(url=url, state=state)

    raise HTTPException(status_code=400, detail="Connector does not support OAuth")


@router.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    session: AsyncSession = Depends(get_db_session),
):
    """Handle OAuth callback from provider."""
    from backend.services.connectors.registry import ConnectorRegistry
    from backend.services.connectors.base import ConnectorType, ConnectorConfig

    # Parse state to get connector_id
    try:
        connector_id_str, _ = state.split(":", 1)
        connector_id = uuid.UUID(connector_id_str)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    result = await session.execute(
        select(ConnectorInstance).where(ConnectorInstance.id == connector_id)
    )
    connector_instance = result.scalar_one_or_none()

    if not connector_instance:
        raise HTTPException(status_code=404, detail="Connector not found")

    connector_class = ConnectorRegistry.get(ConnectorType(connector_instance.connector_type))
    if not connector_class:
        raise HTTPException(status_code=400, detail="Unknown connector type")

    # Create connector instance
    connector = connector_class(
        config=ConnectorConfig(
            credentials=connector_instance.credentials or {},
            settings=connector_instance.sync_config or {},
        ),
        session=session,
        organization_id=connector_instance.organization_id,
    )

    # Exchange code for tokens
    if hasattr(connector, "handle_oauth_callback"):
        tokens = await connector.handle_oauth_callback(code, state)

        # Update credentials
        connector_instance.credentials = tokens
        connector_instance.status = "connected"
        connector_instance.is_active = True
        connector_instance.updated_at = datetime.utcnow()
        await session.commit()

        # Redirect to frontend
        return {"status": "success", "connector_id": str(connector_id)}

    raise HTTPException(status_code=400, detail="Connector does not support OAuth callback")


# =============================================================================
# Sync Endpoints
# =============================================================================

@router.post("/{connector_id}/sync", response_model=SyncJobResponse)
async def trigger_sync(
    connector_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    full_sync: bool = Query(default=False, description="Force full sync"),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Trigger a sync for a connector."""
    from backend.services.connectors.scheduler import get_scheduler

    result = await session.execute(
        select(ConnectorInstance)
        .where(
            ConnectorInstance.id == connector_id,
            ConnectorInstance.organization_id == organization_id,
        )
    )
    connector = result.scalar_one_or_none()

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    if not connector.is_active:
        raise HTTPException(status_code=400, detail="Connector is not active")

    # Trigger sync in background
    scheduler = get_scheduler()
    job = await scheduler.trigger_manual_sync(
        connector_instance_id=str(connector_id),
        organization_id=str(organization_id),
    )

    return SyncJobResponse(
        id=job.id,
        connector_instance_id=job.connector_instance_id,
        status=job.status.value,
        started_at=job.started_at,
        completed_at=job.completed_at,
        resources_synced=job.resources_synced,
        resources_failed=job.resources_failed,
        error_message=job.error_message,
    )


@router.get("/{connector_id}/sync/history", response_model=List[SyncJobResponse])
async def get_sync_history(
    connector_id: uuid.UUID,
    limit: int = Query(default=20, le=100),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Get sync history for a connector."""
    from backend.services.connectors.scheduler import get_scheduler

    result = await session.execute(
        select(ConnectorInstance)
        .where(
            ConnectorInstance.id == connector_id,
            ConnectorInstance.organization_id == organization_id,
        )
    )
    connector = result.scalar_one_or_none()

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    scheduler = get_scheduler()
    jobs = scheduler.get_job_history(
        connector_instance_id=str(connector_id),
        limit=limit,
    )

    return [
        SyncJobResponse(
            id=job.id,
            connector_instance_id=job.connector_instance_id,
            status=job.status.value,
            started_at=job.started_at,
            completed_at=job.completed_at,
            resources_synced=job.resources_synced,
            resources_failed=job.resources_failed,
            error_message=job.error_message,
        )
        for job in jobs
    ]


# =============================================================================
# Resource Browser
# =============================================================================

@router.get("/{connector_id}/browse")
async def browse_resources(
    connector_id: uuid.UUID,
    folder_id: Optional[str] = Query(default=None, description="Folder ID to browse"),
    page_token: Optional[str] = Query(default=None),
    page_size: int = Query(default=50, le=100),
    session: AsyncSession = Depends(get_db_session),
    organization_id: uuid.UUID = Depends(get_current_organization_id),
    current_user: dict = Depends(get_current_user),
):
    """Browse resources in a connected data source."""
    from backend.services.connectors.registry import ConnectorRegistry
    from backend.services.connectors.base import ConnectorType, ConnectorConfig

    result = await session.execute(
        select(ConnectorInstance)
        .where(
            ConnectorInstance.id == connector_id,
            ConnectorInstance.organization_id == organization_id,
        )
    )
    connector_instance = result.scalar_one_or_none()

    if not connector_instance:
        raise HTTPException(status_code=404, detail="Connector not found")

    if not connector_instance.is_active:
        raise HTTPException(status_code=400, detail="Connector is not active")

    connector_class = ConnectorRegistry.get(ConnectorType(connector_instance.connector_type))
    if not connector_class:
        raise HTTPException(status_code=400, detail="Unknown connector type")

    connector = connector_class(
        config=ConnectorConfig(
            credentials=connector_instance.credentials or {},
            settings=connector_instance.sync_config or {},
        ),
        session=session,
        organization_id=organization_id,
    )

    # Authenticate
    if not await connector.authenticate():
        raise HTTPException(status_code=401, detail="Connector authentication failed")

    # List resources
    resources, next_token = await connector.list_resources(
        folder_id=folder_id,
        page_token=page_token,
        page_size=page_size,
    )

    return {
        "resources": [
            {
                "id": r.id,
                "name": r.name,
                "type": r.resource_type.value,
                "mime_type": r.mime_type,
                "size_bytes": r.size_bytes,
                "parent_id": r.parent_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "modified_at": r.modified_at.isoformat() if r.modified_at else None,
                "web_url": r.web_url,
            }
            for r in resources
        ],
        "next_page_token": next_token,
    }
