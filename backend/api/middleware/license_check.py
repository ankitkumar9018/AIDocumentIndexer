"""
AIDocumentIndexer - License Check Middleware
=============================================

FastAPI middleware and dependencies for license validation.
Ensures only licensed users can access protected features.
"""

import os
from functools import wraps
from typing import Callable, List, Optional, Set

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from backend.services.licensing import (
    LicenseInfo,
    LicenseService,
    LicenseTier,
    get_license_service,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# License key from environment (for server-side validation)
LICENSE_KEY = os.getenv("LICENSE_KEY", "")

# Skip license check for these paths
LICENSE_EXEMPT_PATHS: Set[str] = {
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/api/v1/health",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/refresh",
    "/api/v1/license/validate",
    "/api/v1/license/info",
    "/api/v1/license/activate",
    "/docs",
    "/redoc",
    "/openapi.json",
}

# Skip license check in development mode or when no license is configured
DEV_MODE = os.getenv("DEV_MODE", "false").lower() in ("true", "1", "yes")

# Enable license enforcement only when explicitly set
LICENSE_ENFORCEMENT_ENABLED = os.getenv("LICENSE_ENFORCEMENT", "false").lower() in ("true", "1", "yes")


# =============================================================================
# License Middleware
# =============================================================================

class LicenseCheckMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates license for all requests.

    In production, this ensures the server has a valid license before
    processing any requests (except exempted paths).
    """

    def __init__(self, app, license_service: Optional[LicenseService] = None):
        super().__init__(app)
        self.license_service = license_service or get_license_service()
        self._license_validated = False
        self._cached_license: Optional[LicenseInfo] = None

    async def dispatch(self, request: Request, call_next):
        """Process request and check license."""
        path = request.url.path

        # Skip exempt paths
        if self._is_exempt_path(path):
            return await call_next(request)

        # Skip in dev mode or when license enforcement is not enabled
        # This allows the app to run without any license configuration by default
        if DEV_MODE or not LICENSE_ENFORCEMENT_ENABLED:
            return await call_next(request)

        # Validate license (only when LICENSE_ENFORCEMENT=true)
        if not await self._check_license():
            return JSONResponse(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                content={
                    "detail": "Valid license required",
                    "error": "license_required",
                    "license_info": {
                        "valid": False,
                        "message": self._get_license_error_message(),
                    },
                },
            )

        # Add license info to request state for use in routes
        request.state.license = self._cached_license

        return await call_next(request)

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from license check."""
        # Exact match
        if path in LICENSE_EXEMPT_PATHS:
            return True

        # Prefix match for static files
        if path.startswith("/static/") or path.startswith("/assets/"):
            return True

        return False

    async def _check_license(self) -> bool:
        """Check if server has valid license."""
        # Use cached result if available
        if self._license_validated and self._cached_license:
            if self._cached_license.valid and not self._cached_license.is_expired():
                return True

        # Validate license
        if not LICENSE_KEY:
            logger.warning("No license key configured")
            return False

        try:
            license_info = await self.license_service.validate_license(LICENSE_KEY)
            self._cached_license = license_info
            self._license_validated = True

            if not license_info.valid:
                logger.warning(
                    "License validation failed",
                    error=license_info.error,
                )
                return False

            if license_info.is_expired():
                logger.warning(
                    "License expired",
                    expires_at=license_info.expires_at,
                )
                # Allow grace period
                return license_info.is_in_grace_period()

            return True

        except Exception as e:
            logger.error("License check failed", error=str(e))

            # Allow grace period if we have a cached license
            if self._cached_license and self._cached_license.is_in_grace_period():
                return True

            return False

    def _get_license_error_message(self) -> str:
        """Get user-friendly error message."""
        if not LICENSE_KEY:
            return "No license key configured. Please set LICENSE_KEY environment variable."

        if self._cached_license:
            if not self._cached_license.valid:
                return self._cached_license.error or "License validation failed"
            if self._cached_license.is_expired():
                return f"License expired on {self._cached_license.expires_at}"

        return "Unable to validate license. Please check your license configuration."


# =============================================================================
# Dependency Functions
# =============================================================================

async def get_license_info(request: Request) -> Optional[LicenseInfo]:
    """
    Get current license info from request state.

    Usage:
        @router.get("/some-endpoint")
        async def endpoint(license: LicenseInfo = Depends(get_license_info)):
            if license and license.has_feature("advanced_search"):
                ...
    """
    return getattr(request.state, "license", None)


async def require_license(request: Request) -> LicenseInfo:
    """
    Require a valid license for the endpoint.

    Usage:
        @router.get("/premium-endpoint")
        async def endpoint(license: LicenseInfo = Depends(require_license)):
            ...
    """
    license = await get_license_info(request)

    if not license or not license.valid:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Valid license required for this endpoint",
        )

    if license.is_expired() and not license.is_in_grace_period():
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"License expired on {license.expires_at}",
        )

    return license


def require_feature(feature: str):
    """
    Factory for dependency that requires a specific feature.

    Usage:
        @router.get("/knowledge-graph")
        async def kg_endpoint(
            license: LicenseInfo = Depends(require_feature("knowledge_graph")),
        ):
            ...
    """
    async def dependency(request: Request) -> LicenseInfo:
        license = await require_license(request)

        if not license.has_feature(feature):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Feature '{feature}' not included in your license tier ({license.tier.value})",
            )

        return license

    return dependency


def require_tier(min_tier: LicenseTier):
    """
    Factory for dependency that requires minimum license tier.

    Usage:
        @router.get("/enterprise-endpoint")
        async def endpoint(
            license: LicenseInfo = Depends(require_tier(LicenseTier.ENTERPRISE)),
        ):
            ...
    """
    tier_order = [
        LicenseTier.COMMUNITY,
        LicenseTier.PROFESSIONAL,
        LicenseTier.TEAM,
        LicenseTier.ENTERPRISE,
        LicenseTier.UNLIMITED,
    ]

    async def dependency(request: Request) -> LicenseInfo:
        license = await require_license(request)

        current_idx = tier_order.index(license.tier) if license.tier in tier_order else -1
        required_idx = tier_order.index(min_tier)

        if current_idx < required_idx:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"License tier '{min_tier.value}' or higher required. Your tier: {license.tier.value}",
            )

        return license

    return dependency


# =============================================================================
# Decorator for Feature Gating
# =============================================================================

def license_required(feature: Optional[str] = None, tier: Optional[LicenseTier] = None):
    """
    Decorator to require license validation for a function.

    Can be used for both API routes and service functions.

    Usage:
        @license_required(feature="knowledge_graph")
        async def extract_knowledge_graph(documents: List[Document]):
            ...

        @license_required(tier=LicenseTier.ENTERPRISE)
        async def configure_sso():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get license service
            service = get_license_service()
            license = await service.get_current_license()

            if not license or not license.valid:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Valid license required",
                )

            if feature and not license.has_feature(feature):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Feature '{feature}' not included in your license",
                )

            if tier:
                tier_order = [
                    LicenseTier.COMMUNITY,
                    LicenseTier.PROFESSIONAL,
                    LicenseTier.TEAM,
                    LicenseTier.ENTERPRISE,
                    LicenseTier.UNLIMITED,
                ]
                current_idx = tier_order.index(license.tier) if license.tier in tier_order else -1
                required_idx = tier_order.index(tier)

                if current_idx < required_idx:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"License tier '{tier.value}' required",
                    )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Resource Limit Checks
# =============================================================================

async def check_user_limit(
    request: Request,
    current_user_count: int,
) -> bool:
    """
    Check if adding another user exceeds license limit.

    Usage:
        @router.post("/users")
        async def create_user(request: Request):
            current_count = await get_user_count()
            if not await check_user_limit(request, current_count):
                raise HTTPException(403, "User limit reached")
            ...
    """
    license = await get_license_info(request)
    if not license:
        return False
    return license.can_add_user(current_user_count)


async def check_document_limit(
    request: Request,
    current_document_count: int,
) -> bool:
    """
    Check if adding another document exceeds license limit.

    Usage:
        @router.post("/documents")
        async def upload_document(request: Request):
            current_count = await get_document_count()
            if not await check_document_limit(request, current_count):
                raise HTTPException(403, "Document limit reached")
            ...
    """
    license = await get_license_info(request)
    if not license:
        return False
    return license.can_add_document(current_document_count)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LicenseCheckMiddleware",
    "get_license_info",
    "require_license",
    "require_feature",
    "require_tier",
    "license_required",
    "check_user_limit",
    "check_document_limit",
    "LICENSE_EXEMPT_PATHS",
]
