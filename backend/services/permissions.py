"""
AIDocumentIndexer - Permission Service
=======================================

Handles access control and permission enforcement for documents and resources.
Integrates with Row-Level Security (RLS) in PostgreSQL.
"""

import uuid
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

import structlog
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context, set_user_context, clear_user_context
from backend.db.models import User, Document, Chunk, AccessTier

logger = structlog.get_logger(__name__)


# =============================================================================
# Types
# =============================================================================

class Permission(str, Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class UserContext:
    """User context for permission checks and organization isolation."""
    user_id: str
    email: str
    role: str
    access_tier_level: int
    access_tier_name: str
    use_folder_permissions_only: bool = False
    organization_id: Optional[str] = None
    is_superadmin: bool = False

    def can_access_tier(self, tier_level: int) -> bool:
        """Check if user can access a specific tier level."""
        return self.access_tier_level >= tier_level

    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return self.role == "admin"

    def can_switch_organization(self) -> bool:
        """Check if user can switch to other organizations (superadmin only)."""
        return self.is_superadmin


# =============================================================================
# Permission Decorators
# =============================================================================

def require_tier(min_tier: int):
    """
    Decorator to require minimum access tier.

    Usage:
        @require_tier(50)
        async def get_executive_docs(user: UserContext):
            ...
    """
    def decorator(func):
        async def wrapper(*args, user: UserContext, **kwargs):
            if not user.can_access_tier(min_tier):
                raise PermissionError(
                    f"Access denied. Required tier: {min_tier}, your tier: {user.access_tier_level}"
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_admin(func):
    """
    Decorator to require admin role.

    Usage:
        @require_admin
        async def delete_all_docs(user: UserContext):
            ...
    """
    async def wrapper(*args, user: UserContext, **kwargs):
        if not user.is_admin():
            raise PermissionError("Admin access required")
        return await func(*args, user=user, **kwargs)
    return wrapper


# =============================================================================
# Permission Service
# =============================================================================

class PermissionService:
    """
    Service for managing and checking permissions.

    Features:
    - Access tier enforcement
    - Document-level permissions
    - Row-Level Security (RLS) integration
    - Permission caching
    """

    def __init__(self):
        """Initialize permission service."""
        # Cache for access tier lookups
        self._tier_cache: Dict[str, AccessTier] = {}

    async def get_user_context(
        self,
        user_id: str,
        session: Optional[AsyncSession] = None,
    ) -> Optional[UserContext]:
        """
        Get user context for permission checks.

        Args:
            user_id: User ID
            session: Optional database session

        Returns:
            UserContext or None if user not found
        """
        async def _get(db: AsyncSession) -> Optional[UserContext]:
            result = await db.execute(
                select(User, AccessTier)
                .join(AccessTier, User.access_tier_id == AccessTier.id)
                .where(User.id == uuid.UUID(user_id))
            )
            row = result.first()

            if row:
                user, tier = row
                return UserContext(
                    user_id=str(user.id),
                    email=user.email,
                    role="admin" if tier.level >= 100 else "user",
                    access_tier_level=tier.level,
                    access_tier_name=tier.name,
                )
            return None

        if session:
            return await _get(session)
        else:
            async with async_session_context() as db:
                return await _get(db)

    async def check_document_access(
        self,
        user_context: UserContext,
        document_id: str,
        permission: Permission = Permission.READ,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Check if user has access to a specific document.

        Args:
            user_context: User context
            document_id: Document ID
            permission: Required permission type
            session: Optional database session

        Returns:
            True if user has access
        """
        # Admins have full access
        if user_context.is_admin():
            return True

        async def _check(db: AsyncSession) -> bool:
            result = await db.execute(
                select(Document, AccessTier)
                .join(AccessTier, Document.access_tier_id == AccessTier.id)
                .where(Document.id == uuid.UUID(document_id))
            )
            row = result.first()

            if not row:
                return False

            doc, tier = row

            # Check tier access
            if not user_context.can_access_tier(tier.level):
                return False

            # For write/delete, also check ownership or special permissions
            if permission in [Permission.WRITE, Permission.DELETE]:
                # Document owner can modify
                if doc.uploaded_by_id and str(doc.uploaded_by_id) == user_context.user_id:
                    return True
                # Otherwise, need higher tier (admin-like)
                return user_context.access_tier_level >= 90

            return True

        if session:
            return await _check(session)
        else:
            async with async_session_context() as db:
                return await _check(db)

    async def filter_documents_by_access(
        self,
        user_context: UserContext,
        document_ids: List[str],
        session: Optional[AsyncSession] = None,
    ) -> List[str]:
        """
        Filter a list of document IDs to only those the user can access.

        Args:
            user_context: User context
            document_ids: List of document IDs
            session: Optional database session

        Returns:
            Filtered list of accessible document IDs
        """
        if not document_ids:
            return []

        # Admins have full access
        if user_context.is_admin():
            return document_ids

        async def _filter(db: AsyncSession) -> List[str]:
            doc_uuids = [uuid.UUID(d) for d in document_ids]

            result = await db.execute(
                select(Document.id)
                .join(AccessTier, Document.access_tier_id == AccessTier.id)
                .where(
                    and_(
                        Document.id.in_(doc_uuids),
                        AccessTier.level <= user_context.access_tier_level,
                    )
                )
            )

            return [str(row[0]) for row in result.all()]

        if session:
            return await _filter(session)
        else:
            async with async_session_context() as db:
                return await _filter(db)

    async def get_accessible_tier_levels(
        self,
        user_context: UserContext,
        session: Optional[AsyncSession] = None,
    ) -> List[int]:
        """
        Get all tier levels the user can access.

        Args:
            user_context: User context
            session: Optional database session

        Returns:
            List of accessible tier levels
        """
        async def _get(db: AsyncSession) -> List[int]:
            result = await db.execute(
                select(AccessTier.level)
                .where(AccessTier.level <= user_context.access_tier_level)
                .order_by(AccessTier.level)
            )
            return [row[0] for row in result.all()]

        if session:
            return await _get(session)
        else:
            async with async_session_context() as db:
                return await _get(db)

    async def can_assign_tier(
        self,
        user_context: UserContext,
        target_tier_level: int,
    ) -> bool:
        """
        Check if user can assign a specific tier to a document or user.

        Users can only assign tiers at or below their own level.

        Args:
            user_context: User context
            target_tier_level: Tier level to assign

        Returns:
            True if user can assign the tier
        """
        return user_context.access_tier_level >= target_tier_level

    async def setup_rls_context(
        self,
        user_context: UserContext,
        session: AsyncSession,
    ) -> None:
        """
        Set up RLS context for the database session.

        This sets the PostgreSQL session variable for RLS policies.

        Args:
            user_context: User context
            session: Database session
        """
        await set_user_context(session, user_context.user_id)

    async def clear_rls_context(
        self,
        session: AsyncSession,
    ) -> None:
        """
        Clear RLS context from the database session.

        Args:
            session: Database session
        """
        await clear_user_context(session)


# =============================================================================
# Convenience Functions
# =============================================================================

_permission_service: Optional[PermissionService] = None


def get_permission_service() -> PermissionService:
    """Get or create permission service instance."""
    global _permission_service

    if _permission_service is None:
        _permission_service = PermissionService()

    return _permission_service


async def check_access(
    user_id: str,
    document_id: str,
    permission: Permission = Permission.READ,
) -> bool:
    """
    Quick access check for a document.

    Args:
        user_id: User ID
        document_id: Document ID
        permission: Required permission

    Returns:
        True if user has access
    """
    service = get_permission_service()
    user_context = await service.get_user_context(user_id)

    if not user_context:
        return False

    return await service.check_document_access(
        user_context=user_context,
        document_id=document_id,
        permission=permission,
    )


def create_user_context_from_token(token_payload: Dict[str, Any]) -> UserContext:
    """
    Create UserContext from JWT token payload.

    Args:
        token_payload: Decoded JWT payload

    Returns:
        UserContext object
    """
    # Get organization_id for multi-tenant isolation
    organization_id = token_payload.get("organization_id")

    return UserContext(
        user_id=token_payload.get("sub", ""),
        email=token_payload.get("email", ""),
        role=token_payload.get("role", "user"),
        access_tier_level=token_payload.get("access_tier", 10),
        access_tier_name=token_payload.get("tier_name", "Basic"),
        use_folder_permissions_only=token_payload.get("use_folder_permissions_only", False),
        organization_id=organization_id,
        is_superadmin=token_payload.get("is_superadmin", False),
    )
