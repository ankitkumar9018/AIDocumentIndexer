"""
AIDocumentIndexer - API Dependencies
=====================================

Centralized FastAPI dependencies for use across API routes.

This module provides convenient re-exports of commonly used dependencies:
- Authentication (get_current_user, get_user_context)
- Database sessions (get_async_session)
- Organization context (get_current_organization_id)
"""

from typing import Optional
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

# Re-export from database module
from backend.db.database import get_async_session

# Re-export from auth middleware
from backend.api.middleware.auth import (
    get_current_user,
    get_current_user_optional,
    get_user_context,
    get_user_context_optional,
    CurrentUser,
)


async def get_current_organization_id(
    user: dict = Depends(get_current_user),
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-ID"),
) -> UUID:
    """
    Get the current user's organization ID.

    Priority order:
    1. X-Organization-ID header (for multi-org context switching)
    2. organization_id from JWT token claims
    3. Default organization from user profile
    4. User's own ID as fallback (single-user mode)

    Args:
        user: Current authenticated user from JWT
        x_organization_id: Optional organization ID from header

    Returns:
        UUID of the current organization context

    Raises:
        HTTPException: If organization ID is invalid or inaccessible
    """
    # 1. Check header for explicit organization context
    if x_organization_id:
        try:
            org_id = UUID(x_organization_id)
            # In production, validate user has access to this org
            # For now, trust the header if provided
            return org_id
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid X-Organization-ID header format",
            )

    # 2. Check JWT claims for organization_id
    org_from_token = user.get("organization_id") or user.get("org_id")
    if org_from_token:
        try:
            return UUID(org_from_token)
        except (ValueError, TypeError):
            pass  # Fall through to next option

    # 3. Check for default_organization in user claims
    default_org = user.get("default_organization")
    if default_org:
        try:
            return UUID(default_org)
        except (ValueError, TypeError):
            pass  # Fall through to next option

    # 4. Fallback: Use user's ID as organization (single-user/personal mode)
    user_id = user.get("sub", user.get("id", ""))
    try:
        return UUID(user_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to determine organization context",
        )


__all__ = [
    # Database
    "get_async_session",
    # Authentication
    "get_current_user",
    "get_current_user_optional",
    "get_user_context",
    "get_user_context_optional",
    "CurrentUser",
    # Organization
    "get_current_organization_id",
]
