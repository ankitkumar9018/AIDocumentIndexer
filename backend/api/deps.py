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

from fastapi import Depends, HTTPException, status
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
) -> UUID:
    """
    Get the current user's organization ID.

    This is a convenience dependency that extracts the organization_id
    from the current user's context.

    For now, returns the user's ID as a placeholder since organizations
    are determined at the user level.
    """
    # TODO: When multi-org is fully implemented, this should return
    # the user's current active organization from the session/header
    # user is a dict from JWT token, get user_id from 'sub' claim
    user_id = user.get("sub", user.get("id", ""))
    try:
        return UUID(user_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in token",
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
