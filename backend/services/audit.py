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

class LogSeverity(str, Enum):
    """
    Log severity levels following syslog convention (RFC 5424).

    Used for filtering and alerting on audit logs.
    """
    DEBUG = "debug"           # Debug information, very verbose
    INFO = "info"             # Informational messages, normal operations
    NOTICE = "notice"         # Normal but significant conditions
    WARNING = "warning"       # Warning conditions, potential issues
    ERROR = "error"           # Error conditions, operation failed
    CRITICAL = "critical"     # Critical conditions, needs immediate attention


# Map actions to their default severity levels
ACTION_SEVERITY_MAP = {
    # Successful operations - INFO
    "auth.login": LogSeverity.INFO,
    "auth.logout": LogSeverity.INFO,
    "auth.password_change": LogSeverity.NOTICE,
    "auth.token_refresh": LogSeverity.DEBUG,
    "document.create": LogSeverity.INFO,
    "document.read": LogSeverity.DEBUG,
    "document.update": LogSeverity.INFO,
    "document.delete": LogSeverity.NOTICE,
    "document.reprocess": LogSeverity.INFO,
    "search.query": LogSeverity.DEBUG,
    "chat.query": LogSeverity.DEBUG,
    "upload.single": LogSeverity.INFO,
    "upload.batch": LogSeverity.INFO,
    "upload.cancel": LogSeverity.WARNING,
    "access.granted": LogSeverity.DEBUG,

    # Admin actions - NOTICE (significant but normal)
    "admin.user.create": LogSeverity.NOTICE,
    "admin.user.update": LogSeverity.NOTICE,
    "admin.user.delete": LogSeverity.WARNING,
    "admin.user.tier_change": LogSeverity.NOTICE,
    "admin.tier.create": LogSeverity.NOTICE,
    "admin.tier.update": LogSeverity.NOTICE,
    "admin.tier.delete": LogSeverity.WARNING,
    "admin.system.config": LogSeverity.WARNING,

    # Security events - WARNING/ERROR
    "auth.login_failed": LogSeverity.WARNING,
    "access.denied": LogSeverity.WARNING,

    # Service fallback events - WARNING (Phase 55)
    "service.fallback.llm": LogSeverity.WARNING,
    "service.fallback.embedding": LogSeverity.WARNING,
    "service.fallback.tts": LogSeverity.WARNING,
    "service.fallback.ocr": LogSeverity.WARNING,
    "service.fallback.vlm": LogSeverity.WARNING,
    "service.fallback.retrieval": LogSeverity.WARNING,
    "service.fallback.ray": LogSeverity.WARNING,

    # Service errors - ERROR
    "service.error.llm": LogSeverity.ERROR,
    "service.error.embedding": LogSeverity.ERROR,
    "service.error.tts": LogSeverity.ERROR,
    "service.error.ocr": LogSeverity.ERROR,
    "service.error.vlm": LogSeverity.ERROR,
    "service.error.retrieval": LogSeverity.ERROR,
    "service.error.ray": LogSeverity.ERROR,
}


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

    # Service Fallbacks (Phase 55 - fallback logging)
    SERVICE_FALLBACK_LLM = "service.fallback.llm"
    SERVICE_FALLBACK_EMBEDDING = "service.fallback.embedding"
    SERVICE_FALLBACK_TTS = "service.fallback.tts"
    SERVICE_FALLBACK_OCR = "service.fallback.ocr"
    SERVICE_FALLBACK_VLM = "service.fallback.vlm"
    SERVICE_FALLBACK_RETRIEVAL = "service.fallback.retrieval"
    SERVICE_FALLBACK_RAY = "service.fallback.ray"

    # Service Errors (Phase 55 - all fallbacks exhausted)
    SERVICE_ERROR_LLM = "service.error.llm"
    SERVICE_ERROR_EMBEDDING = "service.error.embedding"
    SERVICE_ERROR_TTS = "service.error.tts"
    SERVICE_ERROR_OCR = "service.error.ocr"
    SERVICE_ERROR_VLM = "service.error.vlm"
    SERVICE_ERROR_RETRIEVAL = "service.error.retrieval"
    SERVICE_ERROR_RAY = "service.error.ray"

    @property
    def default_severity(self) -> LogSeverity:
        """Get the default severity for this action."""
        return ACTION_SEVERITY_MAP.get(self.value, LogSeverity.INFO)


@dataclass
class AuditEntry:
    """Audit entry data structure."""
    id: str
    action: str
    severity: str  # Phase 52: Log severity level
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
        severity: Optional[LogSeverity] = None,
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
            severity: Log severity level (auto-detected from action if not provided)
            session: Optional database session

        Returns:
            ID of the created audit log entry
        """
        async def _log(db: AsyncSession) -> str:
            # Create a copy of details to avoid modifying the parameter
            log_details = dict(details) if details else {}

            # Determine severity (Phase 52)
            log_severity = severity or action.default_severity

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

            # Store severity in details (Phase 52)
            log_details["severity"] = log_severity.value

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
                severity=log_severity.value,
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
        severity: Optional[LogSeverity] = None,
        min_severity: Optional[LogSeverity] = None,
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
            severity: Filter by exact severity level (Phase 52)
            min_severity: Filter by minimum severity (e.g., WARNING+) (Phase 52)
            page: Page number
            page_size: Items per page
            session: Optional database session

        Returns:
            Tuple of (list of audit entries, total count)
        """
        # Severity order for min_severity filtering
        SEVERITY_ORDER = {
            LogSeverity.DEBUG: 0,
            LogSeverity.INFO: 1,
            LogSeverity.NOTICE: 2,
            LogSeverity.WARNING: 3,
            LogSeverity.ERROR: 4,
            LogSeverity.CRITICAL: 5,
        }

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

            # Phase 52: Severity filtering via JSON details field
            if severity:
                # Filter by exact severity in details->severity
                conditions.append(
                    AuditLog.details['severity'].astext == severity.value
                )
            elif min_severity:
                # Filter by minimum severity - include all severities at or above min
                min_order = SEVERITY_ORDER.get(min_severity, 0)
                allowed_severities = [
                    s.value for s, order in SEVERITY_ORDER.items()
                    if order >= min_order
                ]
                conditions.append(
                    AuditLog.details['severity'].astext.in_(allowed_severities)
                )

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
                    severity=(log.details or {}).get("severity", "info"),
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

    async def log_service_fallback(
        self,
        service_type: str,
        primary_provider: str,
        fallback_provider: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Log a service fallback event (Phase 55).

        Called when a primary service fails and we fall back to an alternative.

        Args:
            service_type: Type of service (llm, embedding, tts, ocr, vlm, retrieval, ray)
            primary_provider: The provider that failed
            fallback_provider: The provider we're falling back to
            error_message: Error message from the failed provider
            context: Additional context (query, document_id, etc.)
            user_id: User ID if applicable
            session: Optional database session

        Returns:
            Audit log entry ID
        """
        # Map service type to action
        action_map = {
            "llm": AuditAction.SERVICE_FALLBACK_LLM,
            "embedding": AuditAction.SERVICE_FALLBACK_EMBEDDING,
            "tts": AuditAction.SERVICE_FALLBACK_TTS,
            "ocr": AuditAction.SERVICE_FALLBACK_OCR,
            "vlm": AuditAction.SERVICE_FALLBACK_VLM,
            "retrieval": AuditAction.SERVICE_FALLBACK_RETRIEVAL,
            "ray": AuditAction.SERVICE_FALLBACK_RAY,
        }

        action = action_map.get(service_type, AuditAction.SERVICE_FALLBACK_LLM)

        details = {
            "primary_provider": primary_provider,
            "fallback_provider": fallback_provider,
            "error_message": error_message,
            "service_type": service_type,
        }
        if context:
            details["context"] = context

        return await self.log(
            action=action,
            user_id=user_id,
            resource_type="service",
            resource_id=service_type,
            details=details,
            severity=LogSeverity.WARNING,
            session=session,
        )

    async def log_service_error(
        self,
        service_type: str,
        provider: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> str:
        """
        Log a service error when all fallbacks are exhausted (Phase 55).

        Called when no fallback provider is available or all have failed.

        Args:
            service_type: Type of service (llm, embedding, tts, ocr, vlm, retrieval, ray)
            provider: The provider(s) that failed
            error_message: Error message
            context: Additional context
            user_id: User ID if applicable
            session: Optional database session

        Returns:
            Audit log entry ID
        """
        # Map service type to action
        action_map = {
            "llm": AuditAction.SERVICE_ERROR_LLM,
            "embedding": AuditAction.SERVICE_ERROR_EMBEDDING,
            "tts": AuditAction.SERVICE_ERROR_TTS,
            "ocr": AuditAction.SERVICE_ERROR_OCR,
            "vlm": AuditAction.SERVICE_ERROR_VLM,
            "retrieval": AuditAction.SERVICE_ERROR_RETRIEVAL,
            "ray": AuditAction.SERVICE_ERROR_RAY,
        }

        action = action_map.get(service_type, AuditAction.SERVICE_ERROR_LLM)

        details = {
            "provider": provider,
            "error_message": error_message,
            "service_type": service_type,
            "all_fallbacks_exhausted": True,
        }
        if context:
            details["context"] = context

        return await self.log(
            action=action,
            user_id=user_id,
            resource_type="service",
            resource_id=service_type,
            details=details,
            severity=LogSeverity.ERROR,
            session=session,
        )

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

    async def get_logs_by_severity(
        self,
        min_severity: LogSeverity = LogSeverity.WARNING,
        hours: int = 24,
        page: int = 1,
        page_size: int = 100,
        session: Optional[AsyncSession] = None,
    ) -> tuple[List[AuditEntry], int]:
        """
        Get logs at or above a minimum severity level (Phase 52).

        Useful for monitoring errors and warnings.

        Args:
            min_severity: Minimum severity to include (WARNING, ERROR, CRITICAL)
            hours: Number of hours to look back
            page: Page number
            page_size: Items per page
            session: Optional database session

        Returns:
            Tuple of (list of audit entries, total count)
        """
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(hours=hours)

        return await self.get_logs(
            start_date=start_date,
            min_severity=min_severity,
            page=page,
            page_size=page_size,
            session=session,
        )

    async def get_errors_and_warnings(
        self,
        hours: int = 24,
        session: Optional[AsyncSession] = None,
    ) -> List[AuditEntry]:
        """
        Get recent error and warning logs (Phase 52).

        Convenience method for monitoring.

        Args:
            hours: Number of hours to look back
            session: Optional database session

        Returns:
            List of warning/error/critical audit entries
        """
        entries, _ = await self.get_logs_by_severity(
            min_severity=LogSeverity.WARNING,
            hours=hours,
            page_size=200,
            session=session,
        )
        return entries

    async def get_severity_counts(
        self,
        hours: int = 24,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, int]:
        """
        Get count of logs by severity level (Phase 52).

        Useful for dashboard statistics.

        Args:
            hours: Number of hours to look back
            session: Optional database session

        Returns:
            Dict mapping severity levels to counts
        """
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(hours=hours)

        # Initialize counts
        counts = {s.value: 0 for s in LogSeverity}

        async def _query(db: AsyncSession) -> Dict[str, int]:
            # Query all logs in time range
            query = select(AuditLog.details).where(
                AuditLog.created_at >= start_date
            )
            result = await db.execute(query)
            rows = result.scalars().all()

            for details in rows:
                severity = (details or {}).get("severity", "info")
                if severity in counts:
                    counts[severity] += 1
                else:
                    counts["info"] += 1

            return counts

        if session:
            return await _query(session)
        else:
            async with async_session_context() as db:
                return await _query(db)


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


async def audit_service_fallback(
    service_type: str,
    primary_provider: str,
    fallback_provider: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Quick function to log a service fallback event (Phase 55).

    Use this when a primary service fails and you fall back to an alternative.

    Args:
        service_type: Type of service (llm, embedding, tts, ocr, vlm, retrieval, ray)
        primary_provider: The provider that failed
        fallback_provider: The provider we're falling back to
        error_message: Error message from the failed provider
        context: Additional context
        user_id: User ID if applicable

    Returns:
        Audit log entry ID

    Example:
        await audit_service_fallback(
            service_type="llm",
            primary_provider="openai",
            fallback_provider="anthropic",
            error_message="Rate limit exceeded",
            context={"query": "What is RAG?"}
        )
    """
    service = get_audit_service()
    return await service.log_service_fallback(
        service_type=service_type,
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        error_message=error_message,
        context=context,
        user_id=user_id,
    )


async def audit_service_error(
    service_type: str,
    provider: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Quick function to log a service error when all fallbacks exhausted (Phase 55).

    Use this when no fallback is available or all providers have failed.

    Args:
        service_type: Type of service (llm, embedding, tts, ocr, vlm, retrieval, ray)
        provider: The provider(s) that failed
        error_message: Error message
        context: Additional context
        user_id: User ID if applicable

    Returns:
        Audit log entry ID

    Example:
        await audit_service_error(
            service_type="embedding",
            provider="all (openai, voyage, cohere)",
            error_message="All embedding providers unavailable",
            context={"batch_size": 100}
        )
    """
    service = get_audit_service()
    return await service.log_service_error(
        service_type=service_type,
        provider=provider,
        error_message=error_message,
        context=context,
        user_id=user_id,
    )
