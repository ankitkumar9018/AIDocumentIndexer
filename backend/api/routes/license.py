"""
AIDocumentIndexer - License Management API
===========================================

Endpoints for license validation, activation, and management.
"""

import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import require_admin, get_user_context
from backend.services.permissions import UserContext
from backend.services.licensing import (
    LicenseInfo,
    LicenseService,
    LicenseTier,
    MachineFingerprint,
    get_license_service,
    generate_offline_license,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/license", tags=["license"])


# =============================================================================
# Request/Response Models
# =============================================================================

class LicenseValidateRequest(BaseModel):
    """Request to validate a license key."""
    license_key: str = Field(..., min_length=10, description="License key to validate")
    machine_fingerprint: Optional[str] = Field(
        None,
        description="Machine fingerprint for device binding (auto-generated if not provided)",
    )


class LicenseActivateRequest(BaseModel):
    """Request to activate a license on this machine."""
    license_key: str = Field(..., min_length=10, description="License key to activate")


class LicenseInfoResponse(BaseModel):
    """Response with current license information."""
    valid: bool
    tier: str
    tier_display: str
    expires_at: Optional[datetime]
    days_until_expiry: Optional[int]
    machine_fingerprint: Optional[str]
    max_users: Optional[int]
    max_documents: Optional[int]
    features: List[str]
    customer_name: Optional[str]
    in_grace_period: bool
    grace_period_ends: Optional[datetime]
    last_validated: Optional[datetime]
    error: Optional[str] = None


class FingerprintResponse(BaseModel):
    """Response with machine fingerprint details."""
    fingerprint_hash: str
    machine_id: str
    hostname: str
    platform: str
    architecture: str
    cpu_count: int
    mac_address: Optional[str]


class OfflineLicenseRequest(BaseModel):
    """Request to generate an offline license file."""
    license_key: str
    tier: LicenseTier
    expires_at: datetime
    fingerprint: str
    features: Optional[List[str]] = None
    max_users: Optional[int] = None
    max_documents: Optional[int] = None
    customer: Optional[str] = None


class OfflineLicenseResponse(BaseModel):
    """Response with generated offline license."""
    license_content: str
    instructions: str


# =============================================================================
# Tier Display Names
# =============================================================================

TIER_DISPLAY_NAMES = {
    LicenseTier.COMMUNITY: "Community (Free)",
    LicenseTier.PROFESSIONAL: "Professional",
    LicenseTier.TEAM: "Team",
    LicenseTier.ENTERPRISE: "Enterprise",
    LicenseTier.UNLIMITED: "Unlimited",
}


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/validate", response_model=LicenseInfoResponse)
async def validate_license(
    request: LicenseValidateRequest,
    license_service: LicenseService = Depends(get_license_service),
):
    """
    Validate a license key.

    This endpoint can be used to check if a license key is valid before
    activating it, or to validate the currently active license.
    """
    logger.info("License validation requested", license_key=request.license_key[:8] + "...")

    try:
        license_info = await license_service.validate_license(
            license_key=request.license_key,
            force_refresh=True,
        )

        days_until_expiry = None
        if license_info.expires_at:
            delta = license_info.expires_at - datetime.utcnow()
            days_until_expiry = max(0, delta.days)

        return LicenseInfoResponse(
            valid=license_info.valid,
            tier=license_info.tier.value,
            tier_display=TIER_DISPLAY_NAMES.get(license_info.tier, license_info.tier.value),
            expires_at=license_info.expires_at,
            days_until_expiry=days_until_expiry,
            machine_fingerprint=license_info.machine_fingerprint,
            max_users=license_info.max_users,
            max_documents=license_info.max_documents,
            features=list(license_info.features),
            customer_name=license_info.customer_name,
            in_grace_period=license_info.is_in_grace_period(),
            grace_period_ends=license_info.grace_period_ends,
            last_validated=license_info.last_validated,
            error=license_info.error,
        )

    except Exception as e:
        logger.error("License validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"License validation failed: {str(e)}",
        )


@router.post("/activate", response_model=LicenseInfoResponse)
async def activate_license(
    request: LicenseActivateRequest,
    user: UserContext = Depends(require_admin),
    license_service: LicenseService = Depends(get_license_service),
):
    """
    Activate a license key on this server.

    Requires admin access. The license will be bound to this machine's
    fingerprint and validated with the license server.
    """
    logger.info(
        "License activation requested",
        license_key=request.license_key[:8] + "...",
        admin=user.email,
    )

    try:
        # Validate and activate
        license_info = await license_service.validate_license(
            license_key=request.license_key,
            force_refresh=True,
        )

        if not license_info.valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=license_info.error or "License validation failed",
            )

        # Store license key in environment for persistence
        # In production, this would be stored in database or secure config
        os.environ["LICENSE_KEY"] = request.license_key

        days_until_expiry = None
        if license_info.expires_at:
            delta = license_info.expires_at - datetime.utcnow()
            days_until_expiry = max(0, delta.days)

        logger.info(
            "License activated successfully",
            tier=license_info.tier.value,
            expires_at=license_info.expires_at,
        )

        return LicenseInfoResponse(
            valid=license_info.valid,
            tier=license_info.tier.value,
            tier_display=TIER_DISPLAY_NAMES.get(license_info.tier, license_info.tier.value),
            expires_at=license_info.expires_at,
            days_until_expiry=days_until_expiry,
            machine_fingerprint=license_info.machine_fingerprint,
            max_users=license_info.max_users,
            max_documents=license_info.max_documents,
            features=list(license_info.features),
            customer_name=license_info.customer_name,
            in_grace_period=False,
            grace_period_ends=license_info.grace_period_ends,
            last_validated=license_info.last_validated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("License activation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"License activation failed: {str(e)}",
        )


@router.get("/info", response_model=LicenseInfoResponse)
async def get_license_info(
    user: UserContext = Depends(get_user_context),
    license_service: LicenseService = Depends(get_license_service),
):
    """
    Get current license information.

    Returns details about the currently active license including tier,
    features, expiration, and limits.
    """
    license_info = await license_service.get_current_license()

    if not license_info:
        # No license activated - return community tier info
        return LicenseInfoResponse(
            valid=False,
            tier=LicenseTier.COMMUNITY.value,
            tier_display=TIER_DISPLAY_NAMES[LicenseTier.COMMUNITY],
            expires_at=None,
            days_until_expiry=None,
            machine_fingerprint=license_service.get_fingerprint().fingerprint_hash,
            max_users=1,
            max_documents=100,
            features=["basic_search", "document_upload", "basic_chat"],
            customer_name=None,
            in_grace_period=False,
            grace_period_ends=None,
            last_validated=None,
            error="No license activated",
        )

    days_until_expiry = None
    if license_info.expires_at:
        delta = license_info.expires_at - datetime.utcnow()
        days_until_expiry = max(0, delta.days)

    return LicenseInfoResponse(
        valid=license_info.valid,
        tier=license_info.tier.value,
        tier_display=TIER_DISPLAY_NAMES.get(license_info.tier, license_info.tier.value),
        expires_at=license_info.expires_at,
        days_until_expiry=days_until_expiry,
        machine_fingerprint=license_info.machine_fingerprint,
        max_users=license_info.max_users,
        max_documents=license_info.max_documents,
        features=list(license_info.features),
        customer_name=license_info.customer_name,
        in_grace_period=license_info.is_in_grace_period(),
        grace_period_ends=license_info.grace_period_ends,
        last_validated=license_info.last_validated,
        error=license_info.error,
    )


@router.get("/fingerprint", response_model=FingerprintResponse)
async def get_machine_fingerprint(
    user: UserContext = Depends(require_admin),
    license_service: LicenseService = Depends(get_license_service),
):
    """
    Get this machine's fingerprint.

    The fingerprint is used to bind licenses to specific machines.
    Useful for generating offline licenses for air-gapped deployments.
    Requires admin access.
    """
    fingerprint = license_service.get_fingerprint()

    return FingerprintResponse(
        fingerprint_hash=fingerprint.fingerprint_hash,
        machine_id=fingerprint.machine_id,
        hostname=fingerprint.hostname,
        platform=fingerprint.platform,
        architecture=fingerprint.architecture,
        cpu_count=fingerprint.cpu_count,
        mac_address=fingerprint.mac_address,
    )


@router.get("/features")
async def list_license_features(
    user: UserContext = Depends(get_user_context),
    license_service: LicenseService = Depends(get_license_service),
):
    """
    List all available features and their status.

    Shows which features are enabled/disabled based on the current license.
    """
    license_info = await license_service.get_current_license()

    # All possible features
    all_features = {
        "basic_search": "Basic semantic search",
        "advanced_search": "Advanced search with filters and facets",
        "document_upload": "Document upload and processing",
        "basic_chat": "Basic RAG chat",
        "advanced_chat": "Advanced chat with agents and tools",
        "knowledge_graph": "Knowledge graph extraction and visualization",
        "connectors": "External data source connectors",
        "collaboration": "Multi-user collaboration features",
        "workflows": "Automated workflow designer",
        "sso": "Single Sign-On (SAML/OIDC)",
        "audit_logs": "Comprehensive audit logging",
        "api_access": "Full API access",
        "custom_models": "Custom LLM model configuration",
        "priority_support": "Priority support access",
    }

    features_status = []
    for feature_id, description in all_features.items():
        enabled = False
        if license_info:
            enabled = license_info.has_feature(feature_id)

        features_status.append({
            "id": feature_id,
            "name": description,
            "enabled": enabled,
        })

    return {
        "features": features_status,
        "tier": license_info.tier.value if license_info else LicenseTier.COMMUNITY.value,
    }


@router.post("/generate-offline", response_model=OfflineLicenseResponse)
async def generate_offline_license_file(
    request: OfflineLicenseRequest,
    user: UserContext = Depends(require_admin),
):
    """
    Generate an offline license file for air-gapped deployments.

    This creates a signed license file that can be used without
    connecting to the license server. Requires admin access and
    the LICENSE_SIGNING_KEY environment variable.
    """
    signing_key = os.getenv("LICENSE_SIGNING_KEY", "").encode()

    if not signing_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LICENSE_SIGNING_KEY not configured. Cannot generate offline licenses.",
        )

    try:
        license_content = generate_offline_license(
            license_key=request.license_key,
            tier=request.tier,
            expires_at=request.expires_at,
            fingerprint=request.fingerprint,
            features=request.features,
            max_users=request.max_users,
            max_documents=request.max_documents,
            customer=request.customer,
            signing_key=signing_key,
        )

        instructions = """
To use this offline license:

1. Save this content to a file named 'license.key'
2. Place it at ~/.aidocindexer/license.key on the target machine
3. Set the environment variable: LICENSE_PROVIDER=offline
4. Restart the AIDocumentIndexer server

The license is bound to the machine fingerprint provided and cannot
be transferred to other machines.
        """.strip()

        logger.info(
            "Offline license generated",
            admin=user.email,
            customer=request.customer,
            tier=request.tier.value,
        )

        return OfflineLicenseResponse(
            license_content=license_content,
            instructions=instructions,
        )

    except Exception as e:
        logger.error("Failed to generate offline license", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate offline license: {str(e)}",
        )


@router.post("/deactivate")
async def deactivate_license(
    user: UserContext = Depends(require_admin),
    license_service: LicenseService = Depends(get_license_service),
):
    """
    Deactivate the current license.

    This will remove the license from this machine and revert to
    community tier. Requires admin access.
    """
    current_license = await license_service.get_current_license()

    if not current_license:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No license is currently active",
        )

    # Clear license key
    if "LICENSE_KEY" in os.environ:
        del os.environ["LICENSE_KEY"]

    logger.info(
        "License deactivated",
        admin=user.email,
        previous_tier=current_license.tier.value,
    )

    return {
        "message": "License deactivated successfully",
        "previous_tier": current_license.tier.value,
        "current_tier": LicenseTier.COMMUNITY.value,
    }


@router.get("/tiers")
async def list_license_tiers():
    """
    List all available license tiers and their features.

    This is a public endpoint to help users understand tier differences.
    """
    from backend.services.licensing import TIER_FEATURES

    tiers = []
    for tier in LicenseTier:
        if tier == LicenseTier.UNLIMITED:
            continue  # Internal tier, not for display

        features = list(TIER_FEATURES.get(tier, set()))

        # Parse limits from features
        max_users = None
        max_documents = None
        for feature in features:
            if feature.startswith("max_users_"):
                value = feature.replace("max_users_", "")
                max_users = None if value == "unlimited" else int(value)
            elif feature.startswith("max_documents_"):
                value = feature.replace("max_documents_", "")
                max_documents = None if value == "unlimited" else int(value)

        tiers.append({
            "id": tier.value,
            "name": TIER_DISPLAY_NAMES.get(tier, tier.value),
            "features": [f for f in features if not f.startswith("max_")],
            "max_users": max_users,
            "max_documents": max_documents,
        })

    return {"tiers": tiers}
