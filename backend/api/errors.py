"""
AIDocumentIndexer - Standardized Error Handling
================================================

Custom exception classes and error response formatting.
Provides consistent error handling across the API.
"""

from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    message: str
    status_code: int
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None


# =============================================================================
# Custom Exception Classes
# =============================================================================

class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[List[ErrorDetail]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details
        super().__init__(message)

    def to_response(self, request_id: Optional[str] = None) -> ErrorResponse:
        """Convert exception to error response."""
        return ErrorResponse(
            error=self.error_code,
            message=self.message,
            status_code=self.status_code,
            details=self.details,
            request_id=request_id,
        )


class NotFoundError(AppException):
    """Resource not found error."""

    def __init__(
        self,
        resource: str = "Resource",
        resource_id: Optional[str] = None,
    ):
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} with ID '{resource_id}' not found"
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
        )


class ValidationError(AppException):
    """Validation error."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: Optional[List[ErrorDetail]] = None,
    ):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class AuthenticationError(AppException):
    """Authentication error."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="UNAUTHORIZED",
        )


class AuthorizationError(AppException):
    """Authorization/permission error."""

    def __init__(self, message: str = "You don't have permission to access this resource"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="FORBIDDEN",
        )


class RateLimitError(AppException):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str = "Too many requests",
        retry_after: Optional[int] = None,
    ):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMITED",
        )
        self.retry_after = retry_after


class ServiceUnavailableError(AppException):
    """Service unavailable error."""

    def __init__(
        self,
        service: str = "Service",
        message: Optional[str] = None,
    ):
        super().__init__(
            message=message or f"{service} is temporarily unavailable",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
        )


class ExternalServiceError(AppException):
    """Error from external service (LLM, etc.)."""

    def __init__(
        self,
        service: str,
        message: str,
        original_error: Optional[str] = None,
    ):
        details = None
        if original_error:
            details = [ErrorDetail(message=original_error, code="EXTERNAL_ERROR")]
        super().__init__(
            message=f"{service} error: {message}",
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details,
        )


class DocumentProcessingError(AppException):
    """Document processing error."""

    def __init__(
        self,
        document_id: Optional[str] = None,
        message: str = "Document processing failed",
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="PROCESSING_ERROR",
            details=[
                ErrorDetail(
                    field="document_id",
                    message=document_id or "unknown",
                )
            ] if document_id else None,
        )


class ConfigurationError(AppException):
    """Configuration error."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
        )


class QuotaExceededError(AppException):
    """Quota exceeded error."""

    def __init__(
        self,
        resource: str = "API calls",
        limit: Optional[int] = None,
    ):
        message = f"{resource} quota exceeded"
        if limit:
            message += f" (limit: {limit})"
        super().__init__(
            message=message,
            status_code=429,
            error_code="QUOTA_EXCEEDED",
        )


# =============================================================================
# Exception Handlers for FastAPI
# =============================================================================

async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle AppException instances."""
    request_id = getattr(request.state, "request_id", None)

    logger.warning(
        "Application error",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        request_id=request_id,
        path=request.url.path,
    )

    response = exc.to_response(request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(exclude_none=True),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException instances."""
    request_id = getattr(request.state, "request_id", None)

    logger.warning(
        "HTTP error",
        status_code=exc.status_code,
        detail=exc.detail,
        request_id=request_id,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=_status_code_to_error(exc.status_code),
            message=str(exc.detail),
            status_code=exc.status_code,
            request_id=request_id,
        ).model_dump(exclude_none=True),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", None)

    logger.exception(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=request_id,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An internal error occurred",
            status_code=500,
            request_id=request_id,
        ).model_dump(exclude_none=True),
    )


def _status_code_to_error(status_code: int) -> str:
    """Convert HTTP status code to error string."""
    error_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }
    return error_map.get(status_code, "UNKNOWN_ERROR")


# =============================================================================
# Helper Functions
# =============================================================================

def create_validation_error(field: str, message: str) -> ValidationError:
    """Create a validation error for a specific field."""
    return ValidationError(
        message=f"Validation failed: {message}",
        details=[ErrorDetail(field=field, message=message)],
    )


def create_multi_field_validation_error(errors: Dict[str, str]) -> ValidationError:
    """Create a validation error for multiple fields."""
    details = [
        ErrorDetail(field=field, message=message)
        for field, message in errors.items()
    ]
    return ValidationError(
        message=f"Validation failed: {len(errors)} error(s)",
        details=details,
    )


def register_exception_handlers(app) -> None:
    """Register all exception handlers with the FastAPI app."""
    from fastapi import HTTPException

    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
