"""
AIDocumentIndexer - Request ID Middleware
==========================================

Adds unique request IDs (correlation IDs) to every request for
distributed tracing and log correlation.

Features:
- Generates UUID-based request IDs
- Accepts client-provided X-Request-ID headers
- Attaches request ID to response headers
- Makes request ID available via request.state.request_id
- Integrates with structlog for log correlation
"""

import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

logger = structlog.get_logger(__name__)

# Header names for request IDs
REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds a unique request ID to each request.

    The request ID can be:
    1. Provided by the client via X-Request-ID header
    2. Auto-generated as a UUID if not provided

    The request ID is:
    - Stored in request.state.request_id
    - Added to response headers
    - Bound to the structlog context for log correlation
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Get or generate request ID
        request_id = (
            request.headers.get(REQUEST_ID_HEADER) or
            request.headers.get(CORRELATION_ID_HEADER) or
            str(uuid.uuid4())
        )

        # Store in request state for access in route handlers
        request.state.request_id = request_id

        # Bind to structlog context for automatic log correlation
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )

        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers[REQUEST_ID_HEADER] = request_id

        return response


def get_request_id(request: Request) -> str:
    """
    Get the request ID from a request object.

    Args:
        request: The FastAPI/Starlette request object

    Returns:
        The request ID string, or "unknown" if not set
    """
    return getattr(request.state, "request_id", "unknown")
