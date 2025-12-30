"""
AIDocumentIndexer - Audit Logging Service
==========================================

Tracks all security-sensitive operations for compliance and debugging.
Records who did what, when, and from where.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

import structlog
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context
from backend.db.models import AuditLog, User

logger = structlog.get_logger(__name__)


# =============================================================================
# Types
# =============================================================================

class AuditAction(str, Enum):
    """Audit action types for categorization."""
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    PASSWORD_CHANGE = "auth.password_change"
    TOKEN_REFRESH = "auth.token_refresh"

    # Documents
    DOCUMENT_CREATE = "document.create"
    DOCUMENT_READ = "document.read"
    DOCUMENT_UPDATE = "document.update"
    DOCUMENT_DELETE = "document.delete"
    DOCUMENT_REPROCESS = "document.reprocess"

    # Search
    SEARCH_QUERY = "search.query"
    CHAT_QUERY = "chat.query"

    # Uploads
    UPLOAD_SINGLE = "upload.single"
    UPLOAD_BATCH = "upload.batch"
    UPLOAD_CANCEL = "upload.cancel"

    # Admin - Users
    USER_CREATE = "admin.user.create"
    USER_UPDATE = "admin.user.update"
    USER_DELETE = "admin.user.delete"
    USER_TIER_CHANGE = "admin.user.tier_change"

    # Admin - Tiers
    TIER_CREATE = "admin.tier.create"
    TIER_UPDATE = "admin.tier.update"
    TIER_DELETE = "admin.tier.delete"

    # Admin - System
    SYSTEM_CONFIG_CHANGE = "admin.system.config"

    # Access
    ACCESS_DENIED = "access.denied"
    ACCESS_GRANTED = "access.granted"


@dataclass
class AuditEntry:
    """Audit entry data structure."""
    id: str
    action: str
    user_id: Optional[str]
    user_email: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime


# =============================================================================
# Audit Service
# =============================================================================

class AuditService:
    """
    Service for recording and querying audit logs.

    Features:
    - Structured logging with categorized actions
    - IP address and user agent tracking
    - Detailed metadata storage
    - Query support for compliance reporting
    """

    async def log(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Record an audit log entry.

        Args:
            action: The action being logged
            user_id: ID of the user performing the action
            resource_type: Type of resource being acted upon
            resource_id: ID of the resource
            details: Additional details about the action
            ip_address: Client IP address
            user_agent: Client user agent string
            session: Optional database session

        Returns:
            ID of the created audit log entry
        """
        async def _log(db: AsyncSession) -> str:
            # Create a copy of details to avoid modifying the parameter
            log_details = dict(details) if details else {}

            # Safely convert user_id to UUID (may be a non-UUID string from JWT)
            parsed_user_id = None
            if user_id:
                try:
                    parsed_user_id = uuid.UUID(user_id)
                except ValueError:
                    # Non-UUID user_id (e.g., "test-user-123") - store in details instead
                    log_details["user_id_string"] = user_id
                    logger.debug("Non-UUID user_id stored in details", user_id=user_id)

            # Safely convert resource_id to UUID
            parsed_resource_id = None
            if resource_id:
                try:
                    parsed_resource_id = uuid.UUID(resource_id)
                except ValueError:
                    # Non-UUID resource_id - store in details instead
                    log_details["resource_id_string"] = resource_id
                    logger.debug("Non-UUID resource_id stored in details", resource_id=resource_id)

            audit_entry = AuditLog(
                id=uuid.uuid4(),
                action=action.value,
                user_id=parsed_user_id,
                resource_type=resource_type,
                resource_id=parsed_resource_id,
                details=log_details if log_details else None,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            db.add(audit_entry)
            await db.flush()

            logger.info(
                "Audit log created",
                action=action.value,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
            )

            return str(audit_entry.id)

        if session:
            return await _log(session)
        else:
            async with async_session_context() as db:
                result = await _log(db)
                return result

    async def log_auth(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Log an authentication event.

        Args:
            action: Auth action (LOGIN, LOGOUT, etc.)
            user_id: User ID if known
            email: User email for failed logins
            success: Whether the action succeeded
            ip_address: Client IP
            user_agent: Client user agent
            session: Optional database session

        Returns:
            Audit log entry ID
        """
        return await self.log(
            action=action,
            user_id=user_id,
            resource_type="user",
            resource_id=user_id,
            details={"email": email, "success": success},
            ip_address=ip_address,
            user_agent=user_agent,
            session=session,
        )

    async def log_document_action(
        self,
        action: AuditAction,
        user_id: str,
        document_id: str,
        document_name: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Log a document-related action.

        Args:
            action: Document action
            user_id: User performing the action
            document_id: Document ID
            document_name: Document filename for context
            changes: What was changed (for updates)
            ip_address: Client IP
            session: Optional database session

        Returns:
            Audit log entry ID
        """
        details = {}
        if document_name:
            details["document_name"] = document_name
        if changes:
            details["changes"] = changes

        return await self.log(
            action=action,
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            details=details if details else None,
            ip_address=ip_address,
            session=session,
        )

    async def log_access_denied(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        required_tier: Optional[int] = None,
        user_tier: Optional[int] = None,
        ip_address: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Log an access denied event.

        Args:
            user_id: User who was denied
            resource_type: Type of resource
            resource_id: ID of the resource
            required_tier: Required access tier
            user_tier: User's actual tier
            ip_address: Client IP
            session: Optional database session

        Returns:
            Audit log entry ID
        """
        return await self.log(
            action=AuditAction.ACCESS_DENIED,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                "required_tier": required_tier,
                "user_tier": user_tier,
            },
            ip_address=ip_address,
            session=session,
        )

    async def log_admin_action(
        self,
        action: AuditAction,
        admin_user_id: str,
        target_user_id: Optional[str] = None,
        target_resource_type: Optional[str] = None,
        target_resource_id: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Log an admin action.

        Args:
            action: Admin action
            admin_user_id: Admin performing the action
            target_user_id: User being modified (if applicable)
            target_resource_type: Resource type
            target_resource_id: Resource ID
            changes: What was changed
            ip_address: Client IP
            session: Optional database session

        Returns:
            Audit log entry ID
        """
        details = {}
        if target_user_id:
            details["target_user_id"] = target_user_id
        if changes:
            details["changes"] = changes

        return await self.log(
            action=action,
            user_id=admin_user_id,
            resource_type=target_resource_type or "user",
            resource_id=target_resource_id or target_user_id,
            details=details if details else None,
            ip_address=ip_address,
            session=session,
        )

    async def get_logs(
        self,
        action: Optional[AuditAction] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 50,
        session: Optional[AsyncSession] = None,
    ) -> tuple[List[AuditEntry], int]:
        """
        Query audit logs with filtering.

        Args:
            action: Filter by action type
            user_id: Filter by user
            resource_type: Filter by resource type
            resource_id: Filter by specific resource
            start_date: Filter by start date
            end_date: Filter by end date
            page: Page number
            page_size: Items per page
            session: Optional database session

        Returns:
            Tuple of (list of audit entries, total count)
        """
        async def _query(db: AsyncSession) -> tuple[List[AuditEntry], int]:
            # Build base query
            query = select(AuditLog, User.email).outerjoin(
                User, AuditLog.user_id == User.id
            )

            conditions = []

            if action:
                conditions.append(AuditLog.action == action.value)
            if user_id:
                try:
                    conditions.append(AuditLog.user_id == uuid.UUID(user_id))
                except ValueError:
                    # Non-UUID user_id - won't match any records (stored in details)
                    pass
            if resource_type:
                conditions.append(AuditLog.resource_type == resource_type)
            if resource_id:
                try:
                    conditions.append(AuditLog.resource_id == uuid.UUID(resource_id))
                except ValueError:
                    # Non-UUID resource_id - won't match any records
                    pass
            if start_date:
                conditions.append(AuditLog.created_at >= start_date)
            if end_date:
                conditions.append(AuditLog.created_at <= end_date)

            if conditions:
                query = query.where(and_(*conditions))

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await db.execute(count_query)
            total = total_result.scalar() or 0

            # Apply pagination and ordering
            offset = (page - 1) * page_size
            query = query.order_by(desc(AuditLog.created_at)).offset(offset).limit(page_size)

            # Execute query
            result = await db.execute(query)
            rows = result.all()

            # Convert to AuditEntry objects
            entries = [
                AuditEntry(
                    id=str(log.id),
                    action=log.action,
                    user_id=str(log.user_id) if log.user_id else None,
                    user_email=email,
                    resource_type=log.resource_type,
                    resource_id=str(log.resource_id) if log.resource_id else None,
                    details=log.details,
                    ip_address=log.ip_address,
                    user_agent=log.user_agent,
                    created_at=log.created_at,
                )
                for log, email in rows
            ]

            return entries, total

        if session:
            return await _query(session)
        else:
            async with async_session_context() as db:
                return await _query(db)

    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
        session: Optional[AsyncSession] = None,
    ) -> List[AuditEntry]:
        """
        Get recent activity for a specific user.

        Args:
            user_id: User ID
            days: Number of days to look back
            session: Optional database session

        Returns:
            List of audit entries
        """
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(days=days)

        entries, _ = await self.get_logs(
            user_id=user_id,
            start_date=start_date,
            page_size=100,
            session=session,
        )

        return entries

    async def get_document_history(
        self,
        document_id: str,
        session: Optional[AsyncSession] = None,
    ) -> List[AuditEntry]:
        """
        Get all audit entries for a specific document.

        Args:
            document_id: Document ID
            session: Optional database session

        Returns:
            List of audit entries
        """
        entries, _ = await self.get_logs(
            resource_type="document",
            resource_id=document_id,
            page_size=100,
            session=session,
        )

        return entries

    async def get_failed_logins(
        self,
        hours: int = 24,
        session: Optional[AsyncSession] = None,
    ) -> List[AuditEntry]:
        """
        Get recent failed login attempts.

        Useful for security monitoring.

        Args:
            hours: Number of hours to look back
            session: Optional database session

        Returns:
            List of failed login audit entries
        """
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(hours=hours)

        entries, _ = await self.get_logs(
            action=AuditAction.LOGIN_FAILED,
            start_date=start_date,
            page_size=100,
            session=session,
        )

        return entries

    async def get_access_denials(
        self,
        hours: int = 24,
        session: Optional[AsyncSession] = None,
    ) -> List[AuditEntry]:
        """
        Get recent access denied events.

        Useful for identifying permission issues or unauthorized access attempts.

        Args:
            hours: Number of hours to look back
            session: Optional database session

        Returns:
            List of access denied audit entries
        """
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(hours=hours)

        entries, _ = await self.get_logs(
            action=AuditAction.ACCESS_DENIED,
            start_date=start_date,
            page_size=100,
            session=session,
        )

        return entries


# =============================================================================
# Convenience Functions
# =============================================================================

_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get or create audit service instance."""
    global _audit_service

    if _audit_service is None:
        _audit_service = AuditService()

    return _audit_service


async def audit_log(
    action: AuditAction,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> str:
    """
    Quick function to log an audit entry.

    Args:
        action: The action being logged
        user_id: ID of the user performing the action
        resource_type: Type of resource being acted upon
        resource_id: ID of the resource
        details: Additional details
        ip_address: Client IP address
        user_agent: Client user agent

    Returns:
        ID of the created audit log entry
    """
    service = get_audit_service()
    return await service.log(
        action=action,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
    )
