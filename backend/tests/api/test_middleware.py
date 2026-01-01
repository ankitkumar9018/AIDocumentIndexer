"""
AIDocumentIndexer - Middleware Tests
=====================================

Tests for request ID middleware and error handling.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from backend.api.middleware.request_id import RequestIDMiddleware, get_request_id
from backend.api.errors import (
    AppException,
    NotFoundError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    register_exception_handlers,
)


# =============================================================================
# Request ID Middleware Tests
# =============================================================================

class TestRequestIDMiddleware:
    """Tests for the Request ID middleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create a FastAPI app with Request ID middleware."""
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/request-id")
        async def get_request_id_endpoint(request):
            from starlette.requests import Request
            return {"request_id": get_request_id(request)}

        return app

    def test_request_id_generated(self, app_with_middleware):
        """Test that request ID is generated and returned in headers."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        # UUID format check (36 chars with dashes)
        assert len(response.headers["X-Request-ID"]) == 36

    def test_request_id_from_client(self, app_with_middleware):
        """Test that client-provided request ID is used."""
        client = TestClient(app_with_middleware)
        custom_id = "custom-request-id-12345"
        response = client.get("/test", headers={"X-Request-ID": custom_id})

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id

    def test_correlation_id_header(self, app_with_middleware):
        """Test that X-Correlation-ID is also accepted."""
        client = TestClient(app_with_middleware)
        custom_id = "correlation-id-67890"
        response = client.get("/test", headers={"X-Correlation-ID": custom_id})

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id


# =============================================================================
# Error Handler Tests
# =============================================================================

class TestErrorHandlers:
    """Tests for standardized error handling."""

    @pytest.fixture
    def app_with_error_handlers(self):
        """Create a FastAPI app with error handlers registered."""
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)
        register_exception_handlers(app)

        @app.get("/not-found")
        async def not_found_endpoint():
            raise NotFoundError(resource="Document", resource_id="doc-123")

        @app.get("/validation-error")
        async def validation_error_endpoint():
            raise ValidationError(message="Invalid input")

        @app.get("/auth-error")
        async def auth_error_endpoint():
            raise AuthenticationError(message="Token expired")

        @app.get("/forbidden")
        async def forbidden_endpoint():
            raise AuthorizationError()

        @app.get("/rate-limit")
        async def rate_limit_endpoint():
            raise RateLimitError(retry_after=60)

        @app.get("/http-error")
        async def http_error_endpoint():
            raise HTTPException(status_code=400, detail="Bad request")

        @app.get("/generic-error")
        async def generic_error_endpoint():
            raise RuntimeError("Something went wrong")

        return app

    def test_not_found_error(self, app_with_error_handlers):
        """Test NotFoundError handling."""
        client = TestClient(app_with_error_handlers)
        response = client.get("/not-found")

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "NOT_FOUND"
        assert "doc-123" in data["message"]
        assert "request_id" in data

    def test_validation_error(self, app_with_error_handlers):
        """Test ValidationError handling."""
        client = TestClient(app_with_error_handlers)
        response = client.get("/validation-error")

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "VALIDATION_ERROR"

    def test_authentication_error(self, app_with_error_handlers):
        """Test AuthenticationError handling."""
        client = TestClient(app_with_error_handlers)
        response = client.get("/auth-error")

        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "UNAUTHORIZED"

    def test_authorization_error(self, app_with_error_handlers):
        """Test AuthorizationError handling."""
        client = TestClient(app_with_error_handlers)
        response = client.get("/forbidden")

        assert response.status_code == 403
        data = response.json()
        assert data["error"] == "FORBIDDEN"

    def test_rate_limit_error(self, app_with_error_handlers):
        """Test RateLimitError handling."""
        client = TestClient(app_with_error_handlers)
        response = client.get("/rate-limit")

        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "RATE_LIMITED"

    def test_http_exception(self, app_with_error_handlers):
        """Test HTTPException handling."""
        client = TestClient(app_with_error_handlers)
        response = client.get("/http-error")

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "BAD_REQUEST"
        assert "request_id" in data

    def test_generic_error(self, app_with_error_handlers):
        """Test generic exception handling."""
        # TestClient with raise_server_exceptions=False to catch server errors
        client = TestClient(app_with_error_handlers, raise_server_exceptions=False)
        response = client.get("/generic-error")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "INTERNAL_ERROR"
        assert "request_id" in data

    def test_request_id_in_errors(self, app_with_error_handlers):
        """Test that request ID is included in error responses."""
        client = TestClient(app_with_error_handlers)
        custom_id = "error-trace-12345"
        response = client.get("/not-found", headers={"X-Request-ID": custom_id})

        data = response.json()
        assert data["request_id"] == custom_id
