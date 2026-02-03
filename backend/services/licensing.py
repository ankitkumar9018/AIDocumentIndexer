"""
AIDocumentIndexer - Licensing Service
======================================

Enterprise licensing system for code protection and feature gating.
Supports multiple license providers (Cryptolens, Keygen, self-hosted).

Features:
- Machine fingerprinting for device binding
- License key validation with caching
- Feature flags based on license tier
- Grace period for offline operation
- Periodic background validation
"""

import asyncio
import hashlib
import hmac
import os
import platform
import socket
import struct
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class LicenseProvider(str, Enum):
    """Supported license providers."""
    CRYPTOLENS = "cryptolens"
    KEYGEN = "keygen"
    SELF_HOSTED = "self_hosted"
    OFFLINE = "offline"  # For air-gapped deployments


class LicenseTier(str, Enum):
    """License tiers with different feature sets."""
    COMMUNITY = "community"      # Free tier, limited features
    PROFESSIONAL = "professional"  # Single user, all features
    TEAM = "team"                # Multi-user, collaboration
    ENTERPRISE = "enterprise"     # Unlimited, SSO, audit logs
    UNLIMITED = "unlimited"      # No restrictions


# Feature flags by tier
TIER_FEATURES: Dict[LicenseTier, Set[str]] = {
    LicenseTier.COMMUNITY: {
        "basic_search",
        "document_upload",
        "basic_chat",
        "max_documents_100",
        "max_users_1",
    },
    LicenseTier.PROFESSIONAL: {
        "basic_search",
        "advanced_search",
        "document_upload",
        "basic_chat",
        "advanced_chat",
        "knowledge_graph",
        "connectors",
        "max_documents_unlimited",
        "max_users_1",
    },
    LicenseTier.TEAM: {
        "basic_search",
        "advanced_search",
        "document_upload",
        "basic_chat",
        "advanced_chat",
        "knowledge_graph",
        "connectors",
        "collaboration",
        "workflows",
        "max_documents_unlimited",
        "max_users_25",
    },
    LicenseTier.ENTERPRISE: {
        "basic_search",
        "advanced_search",
        "document_upload",
        "basic_chat",
        "advanced_chat",
        "knowledge_graph",
        "connectors",
        "collaboration",
        "workflows",
        "sso",
        "audit_logs",
        "api_access",
        "custom_models",
        "priority_support",
        "max_documents_unlimited",
        "max_users_unlimited",
    },
    LicenseTier.UNLIMITED: set(),  # All features enabled
}


# =============================================================================
# License Models
# =============================================================================

class MachineFingerprint(BaseModel):
    """Hardware fingerprint for device binding."""
    machine_id: str
    hostname: str
    platform: str
    architecture: str
    cpu_count: int
    mac_address: Optional[str] = None
    disk_serial: Optional[str] = None
    fingerprint_hash: str

    @classmethod
    def generate(cls) -> "MachineFingerprint":
        """Generate fingerprint for current machine."""
        machine_id = _get_machine_id()
        hostname = socket.gethostname()
        plat = platform.system()
        arch = platform.machine()
        cpu_count = os.cpu_count() or 1
        mac = _get_mac_address()
        disk = _get_disk_serial()

        # Create composite fingerprint hash
        components = [
            machine_id,
            hostname,
            plat,
            arch,
            str(cpu_count),
            mac or "",
            disk or "",
        ]
        fingerprint_hash = hashlib.sha256(
            "|".join(components).encode()
        ).hexdigest()[:32]

        return cls(
            machine_id=machine_id,
            hostname=hostname,
            platform=plat,
            architecture=arch,
            cpu_count=cpu_count,
            mac_address=mac,
            disk_serial=disk,
            fingerprint_hash=fingerprint_hash,
        )


class LicenseInfo(BaseModel):
    """License information and status."""
    license_key: str
    tier: LicenseTier
    valid: bool
    expires_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    machine_fingerprint: Optional[str] = None
    max_users: Optional[int] = None
    max_documents: Optional[int] = None
    features: Set[str] = Field(default_factory=set)
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    organization_id: Optional[str] = None
    error: Optional[str] = None
    last_validated: Optional[datetime] = None
    grace_period_ends: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

    def is_expired(self) -> bool:
        """Check if license has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_in_grace_period(self) -> bool:
        """Check if in grace period after validation failure."""
        if self.grace_period_ends is None:
            return False
        return datetime.utcnow() < self.grace_period_ends

    def has_feature(self, feature: str) -> bool:
        """Check if license includes a specific feature."""
        if self.tier == LicenseTier.UNLIMITED:
            return True
        return feature in self.features or feature in TIER_FEATURES.get(self.tier, set())

    def can_add_user(self, current_count: int) -> bool:
        """Check if more users can be added."""
        if self.max_users is None:
            return True
        return current_count < self.max_users

    def can_add_document(self, current_count: int) -> bool:
        """Check if more documents can be added."""
        if self.max_documents is None:
            return True
        return current_count < self.max_documents


class LicenseValidationRequest(BaseModel):
    """Request for license validation."""
    license_key: str
    machine_fingerprint: str
    product_id: str = "aidocindexer"
    version: str = "1.0.0"


class LicenseValidationResponse(BaseModel):
    """Response from license server."""
    valid: bool
    tier: Optional[LicenseTier] = None
    expires_at: Optional[datetime] = None
    features: List[str] = Field(default_factory=list)
    max_users: Optional[int] = None
    max_documents: Optional[int] = None
    customer_name: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Machine Fingerprint Helpers
# =============================================================================

def _get_machine_id() -> str:
    """Get unique machine identifier."""
    # Try platform-specific methods
    if platform.system() == "Darwin":
        # macOS: use IOPlatformSerialNumber
        try:
            import subprocess
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.split("\n"):
                if "IOPlatformSerialNumber" in line:
                    return line.split('"')[-2]
        except Exception:
            pass

    elif platform.system() == "Linux":
        # Linux: use /etc/machine-id
        try:
            with open("/etc/machine-id", "r") as f:
                return f.read().strip()
        except Exception:
            pass
        # Fallback to /var/lib/dbus/machine-id
        try:
            with open("/var/lib/dbus/machine-id", "r") as f:
                return f.read().strip()
        except Exception:
            pass

    elif platform.system() == "Windows":
        # Windows: use MachineGuid from registry
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            )
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return value
        except Exception:
            pass

    # Fallback: generate from hostname and uuid
    return hashlib.sha256(
        f"{socket.gethostname()}-{uuid.getnode()}".encode()
    ).hexdigest()[:32]


def _get_mac_address() -> Optional[str]:
    """Get primary MAC address."""
    try:
        mac = uuid.getnode()
        return ":".join(
            f"{(mac >> (8 * i)) & 0xff:02x}"
            for i in reversed(range(6))
        )
    except Exception:
        return None


def _get_disk_serial() -> Optional[str]:
    """Get primary disk serial number."""
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["diskutil", "info", "disk0"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.split("\n"):
                if "Volume UUID" in line or "Disk / Partition UUID" in line:
                    return line.split(":")[-1].strip()
        except Exception:
            pass

    elif platform.system() == "Linux":
        try:
            # Try /dev/disk/by-id
            import os
            by_id = Path("/dev/disk/by-id")
            if by_id.exists():
                for link in by_id.iterdir():
                    if "ata-" in link.name or "nvme-" in link.name:
                        return link.name.split("_")[-1][:20]
        except Exception:
            pass

    return None


# =============================================================================
# License Service
# =============================================================================

class LicenseService:
    """
    Enterprise licensing service.

    Handles license validation, caching, and feature gating.
    Supports multiple license providers and offline operation.
    """

    def __init__(
        self,
        provider: LicenseProvider = LicenseProvider.SELF_HOSTED,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        product_id: str = "aidocindexer",
        cache_ttl: int = 3600,  # 1 hour
        grace_period_hours: int = 72,  # 3 days offline grace
    ):
        self.provider = provider
        self.server_url = server_url or os.getenv("LICENSE_SERVER_URL", "https://license.example.com")
        self.api_key = api_key or os.getenv("LICENSE_API_KEY", "")
        self.product_id = product_id
        self.cache_ttl = cache_ttl
        self.grace_period_hours = grace_period_hours

        self._license_cache: Optional[LicenseInfo] = None
        self._cache_timestamp: Optional[float] = None
        self._validation_lock = asyncio.Lock()
        self._background_task: Optional[asyncio.Task] = None
        self._fingerprint: Optional[MachineFingerprint] = None

    async def initialize(self) -> None:
        """Initialize license service and start background validation."""
        self._fingerprint = MachineFingerprint.generate()
        logger.info(
            "License service initialized",
            provider=self.provider,
            machine_fingerprint=self._fingerprint.fingerprint_hash,
        )

        # Start background validation task
        self._background_task = asyncio.create_task(self._background_validation_loop())

    async def shutdown(self) -> None:
        """Shutdown license service."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

    def get_fingerprint(self) -> MachineFingerprint:
        """Get current machine fingerprint."""
        if not self._fingerprint:
            self._fingerprint = MachineFingerprint.generate()
        return self._fingerprint

    async def validate_license(
        self,
        license_key: str,
        force_refresh: bool = False,
    ) -> LicenseInfo:
        """
        Validate a license key.

        Args:
            license_key: The license key to validate
            force_refresh: If True, bypass cache and validate with server

        Returns:
            LicenseInfo with validation result
        """
        # Check cache first
        if not force_refresh and self._is_cache_valid():
            cached = self._license_cache
            if cached and cached.license_key == license_key:
                return cached

        async with self._validation_lock:
            # Double-check cache after acquiring lock
            if not force_refresh and self._is_cache_valid():
                cached = self._license_cache
                if cached and cached.license_key == license_key:
                    return cached

            # Perform validation based on provider
            try:
                if self.provider == LicenseProvider.CRYPTOLENS:
                    result = await self._validate_cryptolens(license_key)
                elif self.provider == LicenseProvider.KEYGEN:
                    result = await self._validate_keygen(license_key)
                elif self.provider == LicenseProvider.SELF_HOSTED:
                    result = await self._validate_self_hosted(license_key)
                elif self.provider == LicenseProvider.OFFLINE:
                    result = await self._validate_offline(license_key)
                else:
                    result = LicenseInfo(
                        license_key=license_key,
                        tier=LicenseTier.COMMUNITY,
                        valid=False,
                        error=f"Unknown provider: {self.provider}",
                    )
            except Exception as e:
                logger.error("License validation failed", error=str(e))
                # Check if we have a valid cached license to use during outage
                if self._license_cache and self._license_cache.is_in_grace_period():
                    logger.warning("Using cached license during validation failure")
                    return self._license_cache

                result = LicenseInfo(
                    license_key=license_key,
                    tier=LicenseTier.COMMUNITY,
                    valid=False,
                    error=f"Validation failed: {str(e)}",
                )

            # Update cache
            self._license_cache = result
            self._cache_timestamp = time.time()

            return result

    async def get_current_license(self) -> Optional[LicenseInfo]:
        """Get currently cached license info."""
        return self._license_cache

    def has_feature(self, feature: str) -> bool:
        """Check if current license has a feature."""
        if not self._license_cache:
            return False
        return self._license_cache.has_feature(feature)

    def is_valid(self) -> bool:
        """Check if current license is valid."""
        if not self._license_cache:
            return False
        return self._license_cache.valid and not self._license_cache.is_expired()

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_timestamp or not self._license_cache:
            return False
        return (time.time() - self._cache_timestamp) < self.cache_ttl

    async def _validate_cryptolens(self, license_key: str) -> LicenseInfo:
        """Validate with Cryptolens API."""
        fingerprint = self.get_fingerprint()

        async with aiohttp.ClientSession() as session:
            payload = {
                "token": self.api_key,
                "ProductId": self.product_id,
                "Key": license_key,
                "MachineCode": fingerprint.fingerprint_hash,
            }

            async with session.post(
                "https://app.cryptolens.io/api/key/Activate",
                json=payload,
            ) as response:
                data = await response.json()

                if not data.get("result") == 0:
                    return LicenseInfo(
                        license_key=license_key,
                        tier=LicenseTier.COMMUNITY,
                        valid=False,
                        error=data.get("message", "Validation failed"),
                    )

                license_data = data.get("licenseKey", {})
                expires = license_data.get("expires")

                # Parse features from data objects
                features = set()
                data_objects = license_data.get("dataObjects", [])
                for obj in data_objects:
                    if obj.get("name") == "features":
                        features = set(obj.get("stringValue", "").split(","))

                # Determine tier from features or product
                tier = self._determine_tier_from_features(features)

                return LicenseInfo(
                    license_key=license_key,
                    tier=tier,
                    valid=True,
                    expires_at=datetime.fromisoformat(expires) if expires else None,
                    activated_at=datetime.utcnow(),
                    machine_fingerprint=fingerprint.fingerprint_hash,
                    features=features,
                    customer_name=license_data.get("notes"),
                    last_validated=datetime.utcnow(),
                    grace_period_ends=datetime.utcnow() + timedelta(hours=self.grace_period_hours),
                )

    async def _validate_keygen(self, license_key: str) -> LicenseInfo:
        """Validate with Keygen.sh API."""
        fingerprint = self.get_fingerprint()
        account_id = os.getenv("KEYGEN_ACCOUNT_ID", "")

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"License {license_key}",
                "Content-Type": "application/vnd.api+json",
                "Accept": "application/vnd.api+json",
            }

            # Validate license
            async with session.post(
                f"https://api.keygen.sh/v1/accounts/{account_id}/licenses/actions/validate-key",
                headers=headers,
                json={
                    "meta": {
                        "key": license_key,
                        "scope": {
                            "fingerprint": fingerprint.fingerprint_hash,
                        },
                    },
                },
            ) as response:
                data = await response.json()

                if response.status != 200:
                    error = data.get("errors", [{}])[0].get("detail", "Validation failed")
                    return LicenseInfo(
                        license_key=license_key,
                        tier=LicenseTier.COMMUNITY,
                        valid=False,
                        error=error,
                    )

                meta = data.get("meta", {})
                license_data = data.get("data", {}).get("attributes", {})

                if not meta.get("valid"):
                    return LicenseInfo(
                        license_key=license_key,
                        tier=LicenseTier.COMMUNITY,
                        valid=False,
                        error=meta.get("detail", "License invalid"),
                    )

                expires = license_data.get("expiry")
                metadata = license_data.get("metadata", {})

                return LicenseInfo(
                    license_key=license_key,
                    tier=LicenseTier(metadata.get("tier", "professional")),
                    valid=True,
                    expires_at=datetime.fromisoformat(expires.replace("Z", "+00:00")) if expires else None,
                    activated_at=datetime.utcnow(),
                    machine_fingerprint=fingerprint.fingerprint_hash,
                    features=set(metadata.get("features", [])),
                    max_users=metadata.get("max_users"),
                    max_documents=metadata.get("max_documents"),
                    customer_name=license_data.get("name"),
                    last_validated=datetime.utcnow(),
                    grace_period_ends=datetime.utcnow() + timedelta(hours=self.grace_period_hours),
                )

    async def _validate_self_hosted(self, license_key: str) -> LicenseInfo:
        """Validate with self-hosted license server."""
        fingerprint = self.get_fingerprint()

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "license_key": license_key,
                "machine_fingerprint": fingerprint.fingerprint_hash,
                "product_id": self.product_id,
                "hostname": fingerprint.hostname,
                "platform": fingerprint.platform,
            }

            try:
                async with session.post(
                    f"{self.server_url}/api/v1/licenses/validate",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    data = await response.json()

                    if response.status != 200 or not data.get("valid"):
                        return LicenseInfo(
                            license_key=license_key,
                            tier=LicenseTier.COMMUNITY,
                            valid=False,
                            error=data.get("error", "Validation failed"),
                        )

                    return LicenseInfo(
                        license_key=license_key,
                        tier=LicenseTier(data.get("tier", "professional")),
                        valid=True,
                        expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                        activated_at=datetime.utcnow(),
                        machine_fingerprint=fingerprint.fingerprint_hash,
                        features=set(data.get("features", [])),
                        max_users=data.get("max_users"),
                        max_documents=data.get("max_documents"),
                        customer_name=data.get("customer_name"),
                        customer_email=data.get("customer_email"),
                        organization_id=data.get("organization_id"),
                        last_validated=datetime.utcnow(),
                        grace_period_ends=datetime.utcnow() + timedelta(hours=self.grace_period_hours),
                    )
            except aiohttp.ClientError as e:
                logger.warning("License server unreachable", error=str(e))
                # Return cached license if in grace period
                if self._license_cache and self._license_cache.is_in_grace_period():
                    return self._license_cache
                raise

    async def _validate_offline(self, license_key: str) -> LicenseInfo:
        """
        Validate license offline using signed license file.

        For air-gapped deployments, licenses are distributed as signed files.
        """
        license_file = Path(os.getenv("LICENSE_FILE", "~/.aidocindexer/license.key")).expanduser()

        if not license_file.exists():
            return LicenseInfo(
                license_key=license_key,
                tier=LicenseTier.COMMUNITY,
                valid=False,
                error="License file not found",
            )

        try:
            content = license_file.read_text()
            lines = content.strip().split("\n")

            # Parse license file format:
            # LICENSE_KEY=xxx
            # TIER=enterprise
            # EXPIRES=2025-12-31
            # FINGERPRINT=xxx
            # SIGNATURE=xxx (HMAC-SHA256 of above lines)

            data = {}
            signature_line = None

            for line in lines:
                if line.startswith("SIGNATURE="):
                    signature_line = line.split("=", 1)[1]
                elif "=" in line:
                    key, value = line.split("=", 1)
                    data[key] = value

            if not signature_line:
                return LicenseInfo(
                    license_key=license_key,
                    tier=LicenseTier.COMMUNITY,
                    valid=False,
                    error="Invalid license file: missing signature",
                )

            # Verify signature
            signing_key = os.getenv("LICENSE_SIGNING_KEY", "").encode()
            if not signing_key:
                return LicenseInfo(
                    license_key=license_key,
                    tier=LicenseTier.COMMUNITY,
                    valid=False,
                    error="License signing key not configured",
                )

            # Compute expected signature
            content_to_sign = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
            expected_sig = hmac.new(signing_key, content_to_sign.encode(), hashlib.sha256).hexdigest()

            if not hmac.compare_digest(signature_line, expected_sig):
                return LicenseInfo(
                    license_key=license_key,
                    tier=LicenseTier.COMMUNITY,
                    valid=False,
                    error="Invalid license signature",
                )

            # Verify fingerprint matches
            fingerprint = self.get_fingerprint()
            if data.get("FINGERPRINT") and data["FINGERPRINT"] != fingerprint.fingerprint_hash:
                return LicenseInfo(
                    license_key=license_key,
                    tier=LicenseTier.COMMUNITY,
                    valid=False,
                    error="License not valid for this machine",
                )

            # Parse features
            features = set(data.get("FEATURES", "").split(",")) if data.get("FEATURES") else set()

            return LicenseInfo(
                license_key=data.get("LICENSE_KEY", license_key),
                tier=LicenseTier(data.get("TIER", "professional").lower()),
                valid=True,
                expires_at=datetime.fromisoformat(data["EXPIRES"]) if data.get("EXPIRES") else None,
                machine_fingerprint=fingerprint.fingerprint_hash,
                features=features,
                max_users=int(data["MAX_USERS"]) if data.get("MAX_USERS") else None,
                max_documents=int(data["MAX_DOCUMENTS"]) if data.get("MAX_DOCUMENTS") else None,
                customer_name=data.get("CUSTOMER"),
                last_validated=datetime.utcnow(),
            )

        except Exception as e:
            logger.error("Failed to parse license file", error=str(e))
            return LicenseInfo(
                license_key=license_key,
                tier=LicenseTier.COMMUNITY,
                valid=False,
                error=f"Failed to parse license file: {str(e)}",
            )

    def _determine_tier_from_features(self, features: Set[str]) -> LicenseTier:
        """Determine license tier from feature set."""
        if "unlimited" in features or "all" in features:
            return LicenseTier.UNLIMITED
        if "sso" in features or "audit_logs" in features:
            return LicenseTier.ENTERPRISE
        if "collaboration" in features or "workflows" in features:
            return LicenseTier.TEAM
        if "advanced_search" in features or "knowledge_graph" in features:
            return LicenseTier.PROFESSIONAL
        return LicenseTier.COMMUNITY

    async def _background_validation_loop(self) -> None:
        """Background task to periodically revalidate license."""
        while True:
            try:
                await asyncio.sleep(self.cache_ttl)

                if self._license_cache and self._license_cache.license_key:
                    logger.debug("Background license revalidation")
                    await self.validate_license(
                        self._license_cache.license_key,
                        force_refresh=True,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Background validation failed", error=str(e))
                await asyncio.sleep(60)  # Retry after 1 minute on error


# =============================================================================
# Singleton Instance
# =============================================================================

_license_service: Optional[LicenseService] = None


def get_license_service() -> LicenseService:
    """Get or create the license service singleton."""
    global _license_service
    if _license_service is None:
        provider = LicenseProvider(os.getenv("LICENSE_PROVIDER", "self_hosted"))
        _license_service = LicenseService(provider=provider)
    return _license_service


async def initialize_license_service() -> LicenseService:
    """Initialize and return the license service."""
    service = get_license_service()
    await service.initialize()
    return service


# =============================================================================
# License File Generator (for offline deployments)
# =============================================================================

def generate_offline_license(
    license_key: str,
    tier: LicenseTier,
    expires_at: datetime,
    fingerprint: str,
    features: Optional[List[str]] = None,
    max_users: Optional[int] = None,
    max_documents: Optional[int] = None,
    customer: Optional[str] = None,
    signing_key: bytes = b"",
) -> str:
    """
    Generate a signed offline license file.

    This is used by admins to create license files for air-gapped deployments.

    Args:
        license_key: The license key
        tier: License tier
        expires_at: Expiration date
        fingerprint: Target machine fingerprint
        features: List of enabled features
        max_users: Maximum users allowed
        max_documents: Maximum documents allowed
        customer: Customer name
        signing_key: HMAC signing key

    Returns:
        Signed license file content
    """
    data = {
        "LICENSE_KEY": license_key,
        "TIER": tier.value,
        "EXPIRES": expires_at.isoformat(),
        "FINGERPRINT": fingerprint,
    }

    if features:
        data["FEATURES"] = ",".join(features)
    if max_users:
        data["MAX_USERS"] = str(max_users)
    if max_documents:
        data["MAX_DOCUMENTS"] = str(max_documents)
    if customer:
        data["CUSTOMER"] = customer

    # Create signature
    content_to_sign = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
    signature = hmac.new(signing_key, content_to_sign.encode(), hashlib.sha256).hexdigest()

    # Build file content
    lines = [f"{k}={v}" for k, v in sorted(data.items())]
    lines.append(f"SIGNATURE={signature}")

    return "\n".join(lines)
