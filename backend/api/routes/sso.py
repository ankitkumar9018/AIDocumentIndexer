"""
AIDocumentIndexer - SSO (Single Sign-On) API Routes
====================================================

Enterprise SSO endpoints for SAML 2.0 and OIDC authentication.

Endpoints:
- GET  /sso/providers           - List available SSO providers
- POST /sso/configurations      - Create SSO configuration (admin)
- GET  /sso/configurations      - List SSO configurations (admin)
- GET  /sso/configurations/{id} - Get SSO configuration details
- PUT  /sso/configurations/{id} - Update SSO configuration
- DELETE /sso/configurations/{id} - Delete SSO configuration
- GET  /sso/login/{config_id}   - Initiate SSO login
- POST /sso/saml/acs            - SAML Assertion Consumer Service
- GET  /sso/saml/metadata       - Get SAML SP metadata
- GET  /sso/oidc/callback       - OIDC callback endpoint
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Query, Form, status
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import jwt

from backend.api.middleware.auth import require_admin, get_user_context_optional
from backend.services.permissions import UserContext
from backend.services.sso import (
    SSOService,
    SSOConfiguration,
    SSOProvider,
    SSOProtocol,
    SSOConnectionStatus,
    SSOUser,
    get_sso_service,
)
from backend.db.database import get_async_session
from backend.core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sso", tags=["SSO"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SSOProviderInfo(BaseModel):
    """Information about an SSO provider."""
    id: str
    name: str
    protocol: str
    description: str
    logo_url: Optional[str] = None


class CreateSSOConfigRequest(BaseModel):
    """Request to create SSO configuration."""
    protocol: SSOProtocol
    provider: SSOProvider
    display_name: str = Field(..., min_length=1, max_length=100)

    # SAML settings
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None
    saml_slo_url: Optional[str] = None
    saml_certificate: Optional[str] = None

    # OIDC settings
    oidc_issuer: Optional[str] = None
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    oidc_authorization_url: Optional[str] = None
    oidc_token_url: Optional[str] = None
    oidc_userinfo_url: Optional[str] = None
    oidc_scopes: Optional[List[str]] = None

    # Options
    auto_provision_users: bool = True
    auto_update_users: bool = True
    default_role: str = "user"
    attribute_mapping: Optional[dict] = None
    role_mapping: Optional[dict] = None


class UpdateSSOConfigRequest(BaseModel):
    """Request to update SSO configuration."""
    display_name: Optional[str] = None
    enabled: Optional[bool] = None
    saml_sso_url: Optional[str] = None
    saml_slo_url: Optional[str] = None
    saml_certificate: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    auto_provision_users: Optional[bool] = None
    auto_update_users: Optional[bool] = None
    default_role: Optional[str] = None
    attribute_mapping: Optional[dict] = None
    role_mapping: Optional[dict] = None


class SSOConfigResponse(BaseModel):
    """Response for SSO configuration."""
    id: str
    organization_id: str
    protocol: str
    provider: str
    display_name: str
    enabled: bool
    status: str
    created_at: datetime
    last_used_at: Optional[datetime]

    # SAML (no secrets)
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None

    # OIDC (no secrets)
    oidc_issuer: Optional[str] = None
    oidc_client_id: Optional[str] = None
    oidc_authorization_url: Optional[str] = None

    # Options
    auto_provision_users: bool
    auto_update_users: bool
    default_role: str


class SSOLoginResponse(BaseModel):
    """Response for SSO login initiation."""
    redirect_url: str
    session_id: str


# =============================================================================
# In-memory SSO config storage (use database in production)
# =============================================================================

_sso_configs: dict[str, SSOConfiguration] = {}


def _get_config_by_id(config_id: str) -> Optional[SSOConfiguration]:
    """Get SSO config by ID."""
    return _sso_configs.get(config_id)


def _get_configs_by_org(org_id: str) -> List[SSOConfiguration]:
    """Get all SSO configs for an organization."""
    return [c for c in _sso_configs.values() if c.organization_id == org_id]


def _save_config(config: SSOConfiguration) -> None:
    """Save SSO config."""
    _sso_configs[config.id] = config


def _delete_config(config_id: str) -> bool:
    """Delete SSO config."""
    if config_id in _sso_configs:
        del _sso_configs[config_id]
        return True
    return False


# =============================================================================
# Provider Information
# =============================================================================

SSO_PROVIDERS = [
    SSOProviderInfo(
        id="okta",
        name="Okta",
        protocol="oidc",
        description="Enterprise identity management with Okta",
        logo_url="/static/sso/okta.svg",
    ),
    SSOProviderInfo(
        id="azure_ad",
        name="Microsoft Azure AD",
        protocol="oidc",
        description="Microsoft Entra ID (Azure Active Directory)",
        logo_url="/static/sso/azure.svg",
    ),
    SSOProviderInfo(
        id="google",
        name="Google Workspace",
        protocol="oidc",
        description="Sign in with Google Workspace",
        logo_url="/static/sso/google.svg",
    ),
    SSOProviderInfo(
        id="onelogin",
        name="OneLogin",
        protocol="saml",
        description="OneLogin identity platform",
        logo_url="/static/sso/onelogin.svg",
    ),
    SSOProviderInfo(
        id="generic_saml",
        name="Generic SAML 2.0",
        protocol="saml",
        description="Any SAML 2.0 compatible identity provider",
        logo_url=None,
    ),
    SSOProviderInfo(
        id="generic_oidc",
        name="Generic OIDC",
        protocol="oidc",
        description="Any OpenID Connect compatible identity provider",
        logo_url=None,
    ),
]


# =============================================================================
# Public Endpoints
# =============================================================================

@router.get("/providers", response_model=List[SSOProviderInfo])
async def list_sso_providers():
    """
    List available SSO providers.

    Returns a list of supported identity providers with their protocols.
    """
    return SSO_PROVIDERS


@router.get("/login/{config_id}")
async def initiate_sso_login(
    config_id: str,
    redirect_uri: Optional[str] = Query(None, description="Where to redirect after login"),
    sso_service: SSOService = Depends(get_sso_service),
):
    """
    Initiate SSO login flow.

    Redirects the user to the identity provider for authentication.
    """
    config = _get_config_by_id(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSO configuration not found",
        )

    if not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SSO configuration is disabled",
        )

    # Validate and default redirect URI
    if redirect_uri:
        parsed = urlparse(redirect_uri)
        base_parsed = urlparse(sso_service.base_url)
        # Block external redirects — only allow same-origin or relative paths
        if parsed.scheme and parsed.netloc and parsed.netloc != base_parsed.netloc:
            redirect_uri = f"{sso_service.base_url}/dashboard"
    else:
        redirect_uri = f"{sso_service.base_url}/dashboard"

    try:
        if config.protocol == SSOProtocol.SAML:
            auth_url, auth_request = await sso_service.initiate_saml_login(
                config=config,
                redirect_uri=redirect_uri,
            )
        else:  # OIDC
            auth_url, auth_request = await sso_service.initiate_oidc_login(
                config=config,
                redirect_uri=redirect_uri,
            )

        logger.info(
            "SSO login initiated",
            config_id=config_id,
            provider=config.provider,
        )

        return RedirectResponse(url=auth_url, status_code=302)

    except Exception as e:
        logger.error("SSO login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate SSO login",
        )


# =============================================================================
# SAML Endpoints
# =============================================================================

@router.get("/saml/metadata")
async def get_saml_metadata(
    sso_service: SSOService = Depends(get_sso_service),
):
    """
    Get SAML Service Provider metadata.

    Returns XML metadata that can be imported into identity providers.
    """
    metadata_xml = sso_service.generate_sp_metadata_xml()
    return Response(
        content=metadata_xml,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=sp-metadata.xml"},
    )


@router.post("/saml/acs")
async def saml_assertion_consumer_service(
    request: Request,
    SAMLResponse: str = Form(...),
    RelayState: str = Form(...),
    sso_service: SSOService = Depends(get_sso_service),
    session: AsyncSession = Depends(get_async_session),
):
    """
    SAML Assertion Consumer Service (ACS).

    Processes SAML responses from identity providers.
    """
    try:
        # Find config from relay state
        # In production, store config_id in relay state or session
        config_id = RelayState.split(":")[0] if ":" in RelayState else None

        if not config_id:
            # Try to find from stored auth requests
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid relay state",
            )

        config = _get_config_by_id(config_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SSO configuration not found",
            )

        # Process SAML response
        sso_user = await sso_service.process_saml_response(
            saml_response=SAMLResponse,
            relay_state=RelayState,
            config=config,
        )

        # Create or update local user
        jwt_token = await _process_sso_user(sso_user, config, session)

        # Update last used
        config.last_used_at = datetime.utcnow()
        _save_config(config)

        # Redirect to app with token
        redirect_url = f"{sso_service.base_url}/auth/sso-callback?token={jwt_token}"
        return RedirectResponse(url=redirect_url, status_code=302)

    except ValueError as e:
        logger.warning("SAML validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SAML validation failed",
        )
    except Exception as e:
        logger.error("SAML ACS error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process SAML response",
        )


# =============================================================================
# OIDC Endpoints
# =============================================================================

@router.get("/oidc/callback")
async def oidc_callback(
    code: str = Query(...),
    state: str = Query(...),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    sso_service: SSOService = Depends(get_sso_service),
    session: AsyncSession = Depends(get_async_session),
):
    """
    OIDC callback endpoint.

    Handles the authorization code exchange after IdP authentication.
    """
    if error:
        logger.warning("OIDC error", error=error, description=error_description)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_description or error,
        )

    try:
        # Extract config_id from state (format: config_id:random)
        # In production, look up from stored auth requests
        config_id = state.split(":")[0] if ":" in state else None

        # Find config from auth requests (simplified)
        config = None
        for cfg in _sso_configs.values():
            if cfg.protocol == SSOProtocol.OIDC and cfg.enabled:
                config = cfg
                break

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SSO configuration not found",
            )

        # Process OIDC callback
        sso_user = await sso_service.process_oidc_callback(
            code=code,
            state=state,
            config=config,
        )

        # Create or update local user
        jwt_token = await _process_sso_user(sso_user, config, session)

        # Update last used
        config.last_used_at = datetime.utcnow()
        _save_config(config)

        # Redirect to app with token
        redirect_url = f"{sso_service.base_url}/auth/sso-callback?token={jwt_token}"
        return RedirectResponse(url=redirect_url, status_code=302)

    except ValueError as e:
        logger.warning("OIDC validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OIDC validation failed",
        )
    except Exception as e:
        logger.error("OIDC callback error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process OIDC callback",
        )


# =============================================================================
# Admin Endpoints
# =============================================================================

@router.post("/configurations", response_model=SSOConfigResponse)
async def create_sso_configuration(
    request: CreateSSOConfigRequest,
    user: UserContext = Depends(require_admin),
):
    """
    Create a new SSO configuration.

    Requires admin access. Sets up SAML or OIDC integration.
    """
    # Get organization ID from user context
    org_id = user.organization_id or "default"

    # Create configuration
    config = SSOConfiguration(
        organization_id=org_id,
        protocol=request.protocol,
        provider=request.provider,
        display_name=request.display_name,
        saml_entity_id=request.saml_entity_id,
        saml_sso_url=request.saml_sso_url,
        saml_slo_url=request.saml_slo_url,
        saml_certificate=request.saml_certificate,
        oidc_issuer=request.oidc_issuer,
        oidc_client_id=request.oidc_client_id,
        oidc_client_secret=request.oidc_client_secret,
        oidc_authorization_url=request.oidc_authorization_url,
        oidc_token_url=request.oidc_token_url,
        oidc_userinfo_url=request.oidc_userinfo_url,
        oidc_scopes=request.oidc_scopes or ["openid", "profile", "email"],
        auto_provision_users=request.auto_provision_users,
        auto_update_users=request.auto_update_users,
        default_role=request.default_role,
        attribute_mapping=request.attribute_mapping or {},
        role_mapping=request.role_mapping or {},
        status=SSOConnectionStatus.ACTIVE,
    )

    _save_config(config)

    logger.info(
        "SSO configuration created",
        config_id=config.id,
        provider=config.provider,
        admin=user.email,
    )

    return _config_to_response(config)


@router.get("/configurations", response_model=List[SSOConfigResponse])
async def list_sso_configurations(
    user: UserContext = Depends(require_admin),
):
    """
    List all SSO configurations for the organization.

    Requires admin access.
    """
    org_id = user.organization_id or "default"
    configs = _get_configs_by_org(org_id)
    return [_config_to_response(c) for c in configs]


@router.get("/configurations/{config_id}", response_model=SSOConfigResponse)
async def get_sso_configuration(
    config_id: str,
    user: UserContext = Depends(require_admin),
):
    """
    Get SSO configuration details.

    Requires admin access.
    """
    config = _get_config_by_id(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSO configuration not found",
        )

    return _config_to_response(config)


@router.put("/configurations/{config_id}", response_model=SSOConfigResponse)
async def update_sso_configuration(
    config_id: str,
    request: UpdateSSOConfigRequest,
    user: UserContext = Depends(require_admin),
):
    """
    Update SSO configuration.

    Requires admin access.
    """
    config = _get_config_by_id(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSO configuration not found",
        )

    # Update fields
    if request.display_name is not None:
        config.display_name = request.display_name
    if request.enabled is not None:
        config.enabled = request.enabled
    if request.saml_sso_url is not None:
        config.saml_sso_url = request.saml_sso_url
    if request.saml_slo_url is not None:
        config.saml_slo_url = request.saml_slo_url
    if request.saml_certificate is not None:
        config.saml_certificate = request.saml_certificate
    if request.oidc_client_secret is not None:
        config.oidc_client_secret = request.oidc_client_secret
    if request.auto_provision_users is not None:
        config.auto_provision_users = request.auto_provision_users
    if request.auto_update_users is not None:
        config.auto_update_users = request.auto_update_users
    if request.default_role is not None:
        config.default_role = request.default_role
    if request.attribute_mapping is not None:
        config.attribute_mapping = request.attribute_mapping
    if request.role_mapping is not None:
        config.role_mapping = request.role_mapping

    config.updated_at = datetime.utcnow()
    _save_config(config)

    logger.info(
        "SSO configuration updated",
        config_id=config_id,
        admin=user.email,
    )

    return _config_to_response(config)


@router.delete("/configurations/{config_id}")
async def delete_sso_configuration(
    config_id: str,
    user: UserContext = Depends(require_admin),
):
    """
    Delete SSO configuration.

    Requires admin access.
    """
    if not _delete_config(config_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSO configuration not found",
        )

    logger.info(
        "SSO configuration deleted",
        config_id=config_id,
        admin=user.email,
    )

    return {"message": "SSO configuration deleted"}


# =============================================================================
# Helper Functions
# =============================================================================

def _config_to_response(config: SSOConfiguration) -> SSOConfigResponse:
    """Convert config to response (excluding secrets)."""
    return SSOConfigResponse(
        id=config.id,
        organization_id=config.organization_id,
        protocol=config.protocol.value,
        provider=config.provider.value,
        display_name=config.display_name,
        enabled=config.enabled,
        status=config.status.value,
        created_at=config.created_at,
        last_used_at=config.last_used_at,
        saml_entity_id=config.saml_entity_id,
        saml_sso_url=config.saml_sso_url,
        oidc_issuer=config.oidc_issuer,
        oidc_client_id=config.oidc_client_id,
        oidc_authorization_url=config.oidc_authorization_url,
        auto_provision_users=config.auto_provision_users,
        auto_update_users=config.auto_update_users,
        default_role=config.default_role,
    )


async def _process_sso_user(
    sso_user: SSOUser,
    config: SSOConfiguration,
    session: AsyncSession,
) -> str:
    """
    Process SSO user - create/update local user and generate JWT.

    Returns JWT token for authentication.
    """
    from backend.db.models import User
    from sqlalchemy import select

    # Find existing user by email
    result = await session.execute(
        select(User).where(User.email == sso_user.email)
    )
    user = result.scalar_one_or_none()

    if user:
        # Update existing user if auto_update is enabled
        if config.auto_update_users:
            if sso_user.first_name:
                user.first_name = sso_user.first_name
            if sso_user.last_name:
                user.last_name = sso_user.last_name
            user.sso_provider = config.provider.value
            user.sso_external_id = sso_user.external_id
            await session.commit()
    elif config.auto_provision_users:
        # Create new user
        from uuid import uuid4
        user = User(
            id=uuid4(),
            email=sso_user.email,
            first_name=sso_user.first_name,
            last_name=sso_user.last_name,
            full_name=sso_user.full_name,
            role=config.default_role,
            sso_provider=config.provider.value,
            sso_external_id=sso_user.external_id,
            is_active=True,
            email_verified=True,  # SSO users are pre-verified
        )
        session.add(user)
        await session.commit()
    else:
        raise ValueError("User not found and auto-provisioning is disabled")

    # Apply role mapping
    if config.role_mapping and sso_user.groups:
        for group in sso_user.groups:
            if group in config.role_mapping:
                user.role = config.role_mapping[group]
                await session.commit()
                break

    # Generate JWT token
    jwt_secret = os.getenv("JWT_SECRET", "dev-secret")
    token_payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "sso_provider": config.provider.value,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow(),
    }

    token = jwt.encode(token_payload, jwt_secret, algorithm="HS256")
    return token
