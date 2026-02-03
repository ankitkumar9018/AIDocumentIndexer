"""
AIDocumentIndexer - Authentication Middleware
==============================================

FastAPI dependencies for authentication and permission checking.
Bridges JWT tokens with the permission service for access control.
"""

import os
from typing import Optional, Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
import structlog

from backend.services.permissions import (
    PermissionService,
    UserContext,
    Permission,
    get_permission_service,
    create_user_context_from_token,
)

logger = structlog.get_logger(__name__)

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"

# Dev mode bypass (for development only)
DEV_MODE = os.getenv("DEV_MODE", "false").lower() in ("true", "1", "yes")

# Security scheme
security = HTTPBearer(auto_error=False)


# Fixed UUID for dev user - must be valid UUID format for organization_id fallback
DEV_USER_UUID = "00000000-0000-0000-0000-000000000001"


def _get_dev_user_context() -> "UserContext":
    """
    Create a default admin UserContext for development mode.
    Only used when DEV_MODE=true and token is "dev-token".
    """
    from backend.services.permissions import UserContext
    return UserContext(
        user_id=DEV_USER_UUID,
        email="admin@example.com",
        role="admin",
        access_tier_level=100,
        access_tier_name="admin",
        organization_id=None,
        is_superadmin=True,
    )


# =============================================================================
# Response Models
# =============================================================================

class CurrentUser(BaseModel):
    """Current authenticated user model for API responses."""
    id: str
    email: str
    role: str
    access_tier_level: int
    access_tier_name: str
    is_admin: bool

    @property
    def user_id(self) -> str:
        """Alias for id for backwards compatibility."""
        return self.id

    @classmethod
    def from_context(cls, ctx: UserContext) -> "CurrentUser":
        """Create CurrentUser from UserContext."""
        return cls(
            id=ctx.user_id,
            email=ctx.email,
            role=ctx.role,
            access_tier_level=ctx.access_tier_level,
            access_tier_name=ctx.access_tier_name,
            is_admin=ctx.is_admin(),
        )


# =============================================================================
# Token Utilities
# =============================================================================

def decode_jwt_token(token: str, allow_expired: bool = False) -> dict:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string
        allow_expired: If True, allow expired tokens (for refresh endpoint)

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired (when allow_expired=False)
    """
    try:
        # Build decode options
        options = {}
        if allow_expired:
            options["verify_exp"] = False

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options=options)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_for_refresh(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    Get current user from JWT token, allowing expired tokens.
    Used specifically for the token refresh endpoint.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_jwt_token(credentials.credentials, allow_expired=True)

    if not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
        )

    return payload


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    Get current authenticated user from JWT token.

    Returns the raw token payload for backward compatibility.

    In development mode (DEV_MODE=true), you can use "dev-token" as the
    bearer token to bypass JWT validation and get admin access.

    Usage:
        @router.get("/items")
        async def get_items(user: dict = Depends(get_current_user)):
            ...
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Dev mode bypass - use "dev-token" for easy development testing
    if DEV_MODE and credentials.credentials == "dev-token":
        logger.debug("Dev mode: bypassing JWT validation with dev-token")
        return {
            "sub": DEV_USER_UUID,
            "email": "admin@example.com",
            "role": "admin",
            "access_tier_level": 100,
            "access_tier_name": "admin",
            "is_superadmin": True,
        }

    payload = decode_jwt_token(credentials.credentials)

    # Ensure required fields are present
    if not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
        )

    return payload


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    Get current user if authenticated, None otherwise.

    Useful for endpoints that work for both authenticated and anonymous users.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


async def get_user_context(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> UserContext:
    """
    Get UserContext from JWT token for permission checking.

    This is the preferred dependency for new endpoints that need
    permission checking.

    In development mode (DEV_MODE=true), you can use "dev-token" as the
    bearer token to bypass JWT validation and get admin access.

    Usage:
        @router.get("/documents")
        async def list_documents(user: UserContext = Depends(get_user_context)):
            if user.can_access_tier(50):
                ...
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Dev mode bypass - use "dev-token" for easy development testing
    if DEV_MODE and credentials.credentials == "dev-token":
        logger.debug("Dev mode: bypassing JWT validation with dev-token")
        return _get_dev_user_context()

    payload = decode_jwt_token(credentials.credentials)

    # Ensure required fields are present
    if not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
        )

    # Create UserContext from token payload
    return create_user_context_from_token(payload)


async def get_user_context_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[UserContext]:
    """
    Get UserContext if authenticated, None otherwise.
    """
    if not credentials:
        return None

    try:
        return await get_user_context(credentials)
    except HTTPException:
        return None


# =============================================================================
# Permission Dependencies
# =============================================================================

def require_admin(
    user: UserContext = Depends(get_user_context),
) -> UserContext:
    """
    Require admin role for endpoint access.

    Usage:
        @router.delete("/documents/{id}")
        async def delete_document(
            id: UUID,
            user: UserContext = Depends(require_admin),
        ):
            ...
    """
    if not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def require_tier(min_tier: int):
    """
    Factory for dependency that requires minimum access tier.

    Usage:
        @router.get("/executive-reports")
        async def get_executive_reports(
            user: UserContext = Depends(require_tier(80)),
        ):
            ...
    """
    async def dependency(
        user: UserContext = Depends(get_user_context),
    ) -> UserContext:
        if not user.can_access_tier(min_tier):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires access tier {min_tier} or higher. Your tier: {user.access_tier_level}",
            )
        return user

    return dependency


def require_permission(document_id_param: str = "document_id"):
    """
    Factory for dependency that checks document-level permissions.

    This dependency:
    1. Extracts document_id from path parameters
    2. Checks if user has access to the document
    3. Returns UserContext if access is granted

    Usage:
        @router.get("/documents/{document_id}")
        async def get_document(
            document_id: UUID,
            user: UserContext = Depends(require_permission("document_id")),
        ):
            ...
    """
    async def dependency(
        user: UserContext = Depends(get_user_context),
        permission_service: PermissionService = Depends(get_permission_service_dependency),
        **path_params,
    ) -> UserContext:
        # Get document ID from path parameters
        document_id = path_params.get(document_id_param)

        if document_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing {document_id_param} parameter",
            )

        # Check document access
        has_access = await permission_service.check_document_access(
            user_context=user,
            document_id=str(document_id),
            permission=Permission.READ,
        )

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this document",
            )

        return user

    return dependency


# =============================================================================
# Service Dependencies
# =============================================================================

def get_permission_service_dependency() -> PermissionService:
    """Get permission service instance as dependency."""
    return get_permission_service()


# =============================================================================
# Organization Filtering Helper
# =============================================================================

def get_org_filter(user: UserContext) -> Optional[UUID]:
    """
    Get organization ID for filtering queries.

    Returns the organization_id to use for filtering data.
    Returns None if no organization filtering should be applied.

    Usage:
        @router.get("/documents")
        async def list_documents(user: AuthenticatedUser):
            org_id = get_org_filter(user)
            query = select(Document)
            if org_id:
                query = query.where(Document.organization_id == org_id)
            ...
    """
    if user.organization_id:
        try:
            return UUID(user.organization_id)
        except (ValueError, TypeError):
            return None
    return None


# =============================================================================
# Type Aliases for Cleaner Code
# =============================================================================

# Use these type aliases in route handlers for cleaner type hints
AuthenticatedUser = Annotated[UserContext, Depends(get_user_context)]
OptionalUser = Annotated[Optional[UserContext], Depends(get_user_context_optional)]
AdminUser = Annotated[UserContext, Depends(require_admin)]


# =============================================================================
# Helper Functions for Safe UUID Conversion
# =============================================================================

def safe_uuid(value: Optional[str]) -> Optional[UUID]:
    """
    Safely convert a string to UUID, returning None if invalid.

    Use this instead of UUID(value) to avoid ValueError on invalid UUIDs.
    """
    if not value:
        return None
    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def get_org_id(user: UserContext) -> Optional[UUID]:
    """
    Safely get organization_id as UUID from user context.

    Usage:
        org_id = get_org_id(user)  # Returns UUID or None
    """
    return safe_uuid(user.organization_id) if user else None


def get_user_uuid(user: UserContext) -> Optional[UUID]:
    """
    Safely get user_id as UUID from user context.

    Usage:
        user_uuid = get_user_uuid(user)  # Returns UUID or None
    """
    return safe_uuid(user.user_id) if user else None
