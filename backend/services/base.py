"""
AIDocumentIndexer - Base Service Class
======================================

Provides common patterns and utilities for all services:
- Session management
- Logging
- Error handling
- Organization context

This eliminates duplicate code across services and provides
a consistent interface for testing and maintenance.
"""

import uuid
from abc import ABC
from typing import Optional, TypeVar, Generic, Any
from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_async_session, async_session_context

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ServiceException(Exception):
    """Base exception for all service errors."""

    def __init__(self, message: str, code: str = "SERVICE_ERROR", details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ConfigurationException(ServiceException):
    """Invalid configuration error."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class ValidationException(ServiceException):
    """Input validation error."""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[dict] = None):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class NotFoundException(ServiceException):
    """Resource not found error."""

    def __init__(self, resource_type: str, resource_id: str, details: Optional[dict] = None):
        message = f"{resource_type} not found: {resource_id}"
        details = details or {}
        details["resource_type"] = resource_type
        details["resource_id"] = resource_id
        super().__init__(message, code="NOT_FOUND", details=details)


class PermissionException(ServiceException):
    """Permission denied error."""

    def __init__(self, message: str = "Permission denied", details: Optional[dict] = None):
        super().__init__(message, code="PERMISSION_DENIED", details=details)


class RateLimitException(ServiceException):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, code="RATE_LIMIT_EXCEEDED", details=details)


class ProviderException(ServiceException):
    """External provider error (LLM, storage, etc.)."""

    def __init__(self, provider: str, message: str, details: Optional[dict] = None):
        details = details or {}
        details["provider"] = provider
        super().__init__(message, code="PROVIDER_ERROR", details=details)


class BaseService(ABC):
    """
    Base class for all services.

    Provides:
    - Consistent session management (injected or context-managed)
    - Structured logging with service context
    - Organization context for multi-tenant operations
    - Common utility methods

    Usage:
        class MyService(BaseService):
            async def my_operation(self, data: dict) -> Result:
                session = await self.get_session()
                self.log_info("Starting operation", data_size=len(data))
                try:
                    # ... do work with session ...
                    return result
                except Exception as e:
                    self.log_error("Operation failed", error=str(e))
                    raise
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None,
    ):
        """
        Initialize service with optional session and context.

        Args:
            session: Optional pre-existing database session
            organization_id: Organization context for multi-tenant ops
            user_id: User context for audit and permissions
        """
        self._session = session
        self._owns_session = session is None  # Track if we need to manage session lifecycle
        self._organization_id = organization_id
        self._user_id = user_id
        self._logger = structlog.get_logger(self.__class__.__name__)

    @property
    def organization_id(self) -> Optional[uuid.UUID]:
        """Get the current organization context."""
        return self._organization_id

    @organization_id.setter
    def organization_id(self, value: Optional[uuid.UUID]):
        """Set the organization context."""
        self._organization_id = value

    @property
    def user_id(self) -> Optional[uuid.UUID]:
        """Get the current user context."""
        return self._user_id

    @user_id.setter
    def user_id(self, value: Optional[uuid.UUID]):
        """Set the user context."""
        self._user_id = value

    async def get_session(self) -> AsyncSession:
        """
        Get a database session.

        If a session was injected at construction, returns that session.
        Otherwise, gets a new session from the pool.

        Returns:
            AsyncSession: Database session for queries
        """
        if self._session is not None:
            return self._session

        # Create a new session from factory
        from backend.db.database import get_async_session_factory
        session_factory = get_async_session_factory()
        self._session = session_factory()
        return self._session

    def session_context(self):
        """
        Get an async context manager for session management.

        Use when you need explicit session lifecycle control:

            async with self.session_context() as session:
                # ... operations with session ...
                # Session is committed on success, rolled back on error
        """
        return async_session_context()

    def log_info(self, message: str, **kwargs):
        """Log info with service context."""
        self._logger.info(
            message,
            organization_id=str(self._organization_id) if self._organization_id else None,
            user_id=str(self._user_id) if self._user_id else None,
            **kwargs,
        )

    def log_debug(self, message: str, **kwargs):
        """Log debug with service context."""
        self._logger.debug(
            message,
            organization_id=str(self._organization_id) if self._organization_id else None,
            user_id=str(self._user_id) if self._user_id else None,
            **kwargs,
        )

    def log_warning(self, message: str, **kwargs):
        """Log warning with service context."""
        self._logger.warning(
            message,
            organization_id=str(self._organization_id) if self._organization_id else None,
            user_id=str(self._user_id) if self._user_id else None,
            **kwargs,
        )

    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error with service context and optional exception."""
        self._logger.error(
            message,
            organization_id=str(self._organization_id) if self._organization_id else None,
            user_id=str(self._user_id) if self._user_id else None,
            error=str(error) if error else None,
            exc_info=error is not None,
            **kwargs,
        )

    def validate_uuid(self, value: Any, field_name: str = "id") -> uuid.UUID:
        """
        Validate and convert a value to UUID.

        Args:
            value: Value to validate (str or UUID)
            field_name: Field name for error messages

        Returns:
            UUID object

        Raises:
            ValidationException: If value is not a valid UUID
        """
        if isinstance(value, uuid.UUID):
            return value

        if isinstance(value, str):
            try:
                return uuid.UUID(value)
            except ValueError:
                raise ValidationException(f"Invalid UUID format for {field_name}", field=field_name)

        raise ValidationException(f"{field_name} must be a UUID string or UUID object", field=field_name)

    def validate_required(self, value: Any, field_name: str) -> Any:
        """
        Validate that a value is not None or empty.

        Args:
            value: Value to validate
            field_name: Field name for error messages

        Returns:
            The original value if valid

        Raises:
            ValidationException: If value is None or empty
        """
        if value is None:
            raise ValidationException(f"{field_name} is required", field=field_name)

        if isinstance(value, str) and not value.strip():
            raise ValidationException(f"{field_name} cannot be empty", field=field_name)

        if isinstance(value, (list, dict)) and len(value) == 0:
            raise ValidationException(f"{field_name} cannot be empty", field=field_name)

        return value

    def validate_max_length(self, value: str, max_length: int, field_name: str) -> str:
        """
        Validate string maximum length.

        Args:
            value: String to validate
            max_length: Maximum allowed length
            field_name: Field name for error messages

        Returns:
            The original value if valid

        Raises:
            ValidationException: If value exceeds max length
        """
        if value and len(value) > max_length:
            raise ValidationException(
                f"{field_name} exceeds maximum length of {max_length}",
                field=field_name,
                details={"max_length": max_length, "actual_length": len(value)},
            )
        return value


class CRUDService(BaseService, Generic[T]):
    """
    Base class for services that provide CRUD operations on a model.

    Provides standardized create, read, update, delete operations
    with consistent error handling and logging.

    Usage:
        class WorkflowService(CRUDService[Workflow]):
            model_class = Workflow
            model_name = "Workflow"

            # Override for custom logic
            async def before_create(self, data: dict) -> dict:
                data["status"] = "draft"
                return data
    """

    model_class: type = None  # Override in subclass
    model_name: str = None  # Override in subclass (for error messages)

    async def get_by_id(self, id: uuid.UUID) -> Optional[T]:
        """
        Get a single record by ID.

        Args:
            id: UUID of the record

        Returns:
            Model instance or None if not found
        """
        from sqlalchemy import select

        session = await self.get_session()
        query = select(self.model_class).where(self.model_class.id == id)

        # Add organization filter for multi-tenant
        # PHASE 12 FIX: Include items from user's org AND items without org (legacy/shared)
        if self._organization_id and hasattr(self.model_class, "organization_id"):
            from sqlalchemy import or_
            query = query.where(
                or_(
                    self.model_class.organization_id == self._organization_id,
                    self.model_class.organization_id.is_(None),
                )
            )

        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_id_or_raise(self, id: uuid.UUID) -> T:
        """
        Get a single record by ID, raising if not found.

        Args:
            id: UUID of the record

        Returns:
            Model instance

        Raises:
            NotFoundException: If record doesn't exist
        """
        record = await self.get_by_id(id)
        if record is None:
            raise NotFoundException(self.model_name or self.model_class.__name__, str(id))
        return record

    async def list(
        self,
        page: int = 1,
        page_size: int = 20,
        order_by: str = "created_at",
        order_desc: bool = True,
        filters: Optional[dict] = None,
    ) -> tuple[list[T], int]:
        """
        List records with pagination.

        Args:
            page: Page number (1-indexed)
            page_size: Records per page
            order_by: Column to order by
            order_desc: Whether to order descending
            filters: Optional dict of field=value filters

        Returns:
            Tuple of (records list, total count)
        """
        from sqlalchemy import select, func, desc, asc

        session = await self.get_session()

        # Base query
        query = select(self.model_class)

        # Add organization filter
        # PHASE 12 FIX: Include items from user's org AND items without org (legacy/shared)
        if self._organization_id and hasattr(self.model_class, "organization_id"):
            from sqlalchemy import or_
            query = query.where(
                or_(
                    self.model_class.organization_id == self._organization_id,
                    self.model_class.organization_id.is_(None),
                )
            )

        # Apply filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.where(getattr(self.model_class, field) == value)

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Apply ordering
        order_column = getattr(self.model_class, order_by, self.model_class.created_at)
        if order_desc:
            query = query.order_by(desc(order_column))
        else:
            query = query.order_by(asc(order_column))

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        result = await session.execute(query)
        records = list(result.scalars().all())

        return records, total

    async def create(self, data: dict) -> T:
        """
        Create a new record.

        Args:
            data: Dict of field values

        Returns:
            Created model instance
        """
        session = await self.get_session()

        # Add organization ID if applicable
        if self._organization_id and hasattr(self.model_class, "organization_id"):
            data["organization_id"] = self._organization_id

        # Add default ID if not provided
        if "id" not in data:
            data["id"] = uuid.uuid4()

        # Hook for subclass customization
        data = await self.before_create(data)

        record = self.model_class(**data)
        session.add(record)
        await session.commit()
        await session.refresh(record)

        self.log_info(f"{self.model_name} created", record_id=str(record.id))

        # Hook for post-create actions
        await self.after_create(record)

        return record

    async def update(self, id: uuid.UUID, data: dict) -> T:
        """
        Update an existing record.

        Args:
            id: UUID of record to update
            data: Dict of field values to update

        Returns:
            Updated model instance

        Raises:
            NotFoundException: If record doesn't exist
        """
        session = await self.get_session()

        record = await self.get_by_id_or_raise(id)

        # Hook for subclass customization
        data = await self.before_update(record, data)

        # Update fields
        for field, value in data.items():
            if hasattr(record, field) and value is not None:
                setattr(record, field, value)

        # Update timestamp if available
        if hasattr(record, "updated_at"):
            record.updated_at = datetime.utcnow()

        await session.commit()
        await session.refresh(record)

        self.log_info(f"{self.model_name} updated", record_id=str(record.id))

        # Hook for post-update actions
        await self.after_update(record)

        return record

    async def delete(self, id: uuid.UUID, soft: bool = True) -> bool:
        """
        Delete a record.

        Args:
            id: UUID of record to delete
            soft: If True and model has is_deleted field, soft delete

        Returns:
            True if deleted

        Raises:
            NotFoundException: If record doesn't exist
        """
        session = await self.get_session()

        record = await self.get_by_id_or_raise(id)

        # Hook for subclass customization
        await self.before_delete(record)

        if soft and hasattr(record, "is_deleted"):
            record.is_deleted = True
            if hasattr(record, "deleted_at"):
                record.deleted_at = datetime.utcnow()
            await session.commit()
        else:
            await session.delete(record)
            await session.commit()

        self.log_info(f"{self.model_name} deleted", record_id=str(id), soft=soft)

        # Hook for post-delete actions
        await self.after_delete(record)

        return True

    # Hooks for subclass customization
    async def before_create(self, data: dict) -> dict:
        """Hook called before creating a record. Override to customize."""
        return data

    async def after_create(self, record: T) -> None:
        """Hook called after creating a record. Override to customize."""
        pass

    async def before_update(self, record: T, data: dict) -> dict:
        """Hook called before updating a record. Override to customize."""
        return data

    async def after_update(self, record: T) -> None:
        """Hook called after updating a record. Override to customize."""
        pass

    async def before_delete(self, record: T) -> None:
        """Hook called before deleting a record. Override to customize."""
        pass

    async def after_delete(self, record: T) -> None:
        """Hook called after deleting a record. Override to customize."""
        pass
