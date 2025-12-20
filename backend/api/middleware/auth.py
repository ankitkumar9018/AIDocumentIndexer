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

# Security scheme
security = HTTPBearer(auto_error=False)


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
# Type Aliases for Cleaner Code
# =============================================================================

# Use these type aliases in route handlers for cleaner type hints
AuthenticatedUser = Annotated[UserContext, Depends(get_user_context)]
OptionalUser = Annotated[Optional[UserContext], Depends(get_user_context_optional)]
AdminUser = Annotated[UserContext, Depends(require_admin)]
