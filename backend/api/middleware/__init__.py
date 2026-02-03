"""
AIDocumentIndexer - API Middleware
==================================

Middleware components for authentication, permission checking,
rate limiting, and cost limits.
"""

from backend.api.middleware.auth import (
    get_current_user,
    get_current_user_optional,
    get_user_context,
    get_user_context_optional,
    require_admin,
    require_tier,
    require_permission,
    CurrentUser,
)
from backend.api.middleware.rate_limit import (
    RateLimitChecker,
    RateLimitResult,
    RateLimitSettings,
    rate_limit_dependency,
    get_rate_limit_checker,
    get_user_rate_limit_usage,
    reset_user_rate_limits,
    add_rate_limit_headers,
)
from backend.api.middleware.cost_limit import (
    CostLimitChecker,
    CostLimitResult,
    CostLimitSettings,
    cost_limit_dependency,
    get_cost_limit_checker,
    get_user_cost_status,
    check_estimated_cost,
    record_actual_cost,
    estimate_cost,
    reset_user_cost_tracking,
)
from backend.api.middleware.request_id import (
    RequestIDMiddleware,
    get_request_id,
    REQUEST_ID_HEADER,
    CORRELATION_ID_HEADER,
)
from backend.api.middleware.license_check import (
    LicenseCheckMiddleware,
    get_license_info,
    require_license,
    require_feature,
    require_tier as require_license_tier,
    license_required,
    check_user_limit,
    check_document_limit,
    LICENSE_EXEMPT_PATHS,
)

__all__ = [
    # Auth
    "get_current_user",
    "get_current_user_optional",
    "get_user_context",
    "get_user_context_optional",
    "require_admin",
    "require_tier",
    "require_permission",
    "CurrentUser",
    # Rate Limiting
    "RateLimitChecker",
    "RateLimitResult",
    "RateLimitSettings",
    "rate_limit_dependency",
    "get_rate_limit_checker",
    "get_user_rate_limit_usage",
    "reset_user_rate_limits",
    "add_rate_limit_headers",
    # Cost Limits
    "CostLimitChecker",
    "CostLimitResult",
    "CostLimitSettings",
    "cost_limit_dependency",
    "get_cost_limit_checker",
    "get_user_cost_status",
    "check_estimated_cost",
    "record_actual_cost",
    "estimate_cost",
    "reset_user_cost_tracking",
    # Request ID / Correlation
    "RequestIDMiddleware",
    "get_request_id",
    "REQUEST_ID_HEADER",
    "CORRELATION_ID_HEADER",
    # License Check
    "LicenseCheckMiddleware",
    "get_license_info",
    "require_license",
    "require_feature",
    "require_license_tier",
    "license_required",
    "check_user_limit",
    "check_document_limit",
    "LICENSE_EXEMPT_PATHS",
]
