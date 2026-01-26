"""
AIDocumentIndexer - Enterprise Features
========================================

Phase 22: Enterprise-grade features for organizations.

Features:
- Multi-tenant architecture with data isolation
- Role-Based Access Control (RBAC)
- Comprehensive audit logging
- Compliance reporting (SOC2, GDPR)
- Usage quotas and billing

Multi-Tenant Architecture:
- Organization-level data isolation
- Shared infrastructure, isolated data
- Per-tenant encryption keys (optional)
- Cross-tenant analytics (admin only)

RBAC Roles:
- Super Admin: Full system access
- Org Admin: Organization management
- Manager: Team and document management
- Editor: Create and edit documents
- Viewer: Read-only access
- API User: Programmatic access only
"""

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Roles and Permissions
# =============================================================================

class Role(str, Enum):
    """User roles in the system."""
    SUPER_ADMIN = "super_admin"  # System-wide admin
    ORG_ADMIN = "org_admin"      # Organization admin
    MANAGER = "manager"          # Team manager
    EDITOR = "editor"            # Can create/edit
    VIEWER = "viewer"            # Read-only
    API_USER = "api_user"        # API access only


class Permission(str, Enum):
    """Granular permissions."""
    # Document permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_SHARE = "document:share"

    # Collection permissions
    COLLECTION_READ = "collection:read"
    COLLECTION_CREATE = "collection:create"
    COLLECTION_UPDATE = "collection:update"
    COLLECTION_DELETE = "collection:delete"

    # Chat/Query permissions
    CHAT_ACCESS = "chat:access"
    CHAT_EXPORT = "chat:export"

    # Audio permissions
    AUDIO_GENERATE = "audio:generate"
    AUDIO_EXPORT = "audio:export"

    # User management
    USER_READ = "user:read"
    USER_INVITE = "user:invite"
    USER_MANAGE = "user:manage"

    # Organization management
    ORG_READ = "org:read"
    ORG_UPDATE = "org:update"
    ORG_BILLING = "org:billing"

    # Admin permissions
    ADMIN_AUDIT = "admin:audit"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_SYSTEM = "admin:system"

    # Settings permissions (granular)
    SETTINGS_READ = "settings:read"
    SETTINGS_WRITE = "settings:write"

    # Agent permissions
    AGENT_CREATE = "agent:create"
    AGENT_PUBLISH = "agent:publish"
    AGENT_MANAGE = "agent:manage"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.SUPER_ADMIN: set(Permission),  # All permissions

    Role.ORG_ADMIN: {
        Permission.DOCUMENT_READ, Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_UPDATE, Permission.DOCUMENT_DELETE,
        Permission.DOCUMENT_SHARE,
        Permission.COLLECTION_READ, Permission.COLLECTION_CREATE,
        Permission.COLLECTION_UPDATE, Permission.COLLECTION_DELETE,
        Permission.CHAT_ACCESS, Permission.CHAT_EXPORT,
        Permission.AUDIO_GENERATE, Permission.AUDIO_EXPORT,
        Permission.USER_READ, Permission.USER_INVITE, Permission.USER_MANAGE,
        Permission.ORG_READ, Permission.ORG_UPDATE, Permission.ORG_BILLING,
        Permission.ADMIN_AUDIT,
        Permission.SETTINGS_READ, Permission.SETTINGS_WRITE,
        Permission.AGENT_CREATE, Permission.AGENT_PUBLISH, Permission.AGENT_MANAGE,
    },

    Role.MANAGER: {
        Permission.DOCUMENT_READ, Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_UPDATE, Permission.DOCUMENT_DELETE,
        Permission.DOCUMENT_SHARE,
        Permission.COLLECTION_READ, Permission.COLLECTION_CREATE,
        Permission.COLLECTION_UPDATE,
        Permission.CHAT_ACCESS, Permission.CHAT_EXPORT,
        Permission.AUDIO_GENERATE, Permission.AUDIO_EXPORT,
        Permission.USER_READ, Permission.USER_INVITE,
        Permission.AGENT_CREATE,
    },

    Role.EDITOR: {
        Permission.DOCUMENT_READ, Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_UPDATE,
        Permission.COLLECTION_READ, Permission.COLLECTION_CREATE,
        Permission.CHAT_ACCESS,
        Permission.AUDIO_GENERATE,
    },

    Role.VIEWER: {
        Permission.DOCUMENT_READ,
        Permission.COLLECTION_READ,
        Permission.CHAT_ACCESS,
    },

    Role.API_USER: {
        Permission.DOCUMENT_READ,
        Permission.COLLECTION_READ,
        Permission.CHAT_ACCESS,
    },
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Organization:
    """Organization (tenant) in the system."""
    id: str
    name: str
    slug: str
    plan: str = "free"  # free, pro, enterprise
    settings: Dict[str, Any] = field(default_factory=dict)
    quotas: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Encryption
    encryption_key_id: Optional[str] = None  # For customer-managed keys

    # Status
    is_active: bool = True
    suspended_at: Optional[datetime] = None
    suspension_reason: Optional[str] = None


@dataclass
class OrganizationMember:
    """User membership in an organization."""
    id: str
    user_id: str
    organization_id: str
    role: Role
    custom_permissions: Set[Permission] = field(default_factory=set)
    joined_at: datetime = field(default_factory=datetime.utcnow)
    invited_by: Optional[str] = None

    @property
    def permissions(self) -> Set[Permission]:
        """Get all permissions (role + custom)."""
        base_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return base_permissions | self.custom_permissions


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance."""
    id: str
    timestamp: datetime
    organization_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class UsageQuota:
    """Usage quota tracking."""
    organization_id: str
    quota_type: str  # documents, storage, queries, etc.
    limit: int
    used: int
    period_start: datetime
    period_end: datetime

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def usage_percent(self) -> float:
        if self.limit == 0:
            return 100.0
        return (self.used / self.limit) * 100


# =============================================================================
# Multi-Tenant Service
# =============================================================================

class MultiTenantService:
    """
    Service for multi-tenant data isolation.

    Ensures all queries are scoped to the current organization.
    """

    def __init__(self):
        self._current_org_id: Optional[str] = None
        self._org_cache: Dict[str, Organization] = {}

    def set_tenant(self, organization_id: str) -> None:
        """Set the current tenant context."""
        self._current_org_id = organization_id

    def get_tenant(self) -> Optional[str]:
        """Get the current tenant ID."""
        return self._current_org_id

    def clear_tenant(self) -> None:
        """Clear the tenant context."""
        self._current_org_id = None

    async def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID with caching."""
        if org_id in self._org_cache:
            return self._org_cache[org_id]

        # In production, fetch from database
        # For now, return a mock
        org = Organization(
            id=org_id,
            name=f"Organization {org_id[:8]}",
            slug=f"org-{org_id[:8]}",
        )
        self._org_cache[org_id] = org
        return org

    def tenant_filter(self, query: Any) -> Any:
        """
        Add tenant filter to a database query.

        Usage:
            query = session.query(Document)
            query = tenant_service.tenant_filter(query)
        """
        if self._current_org_id is None:
            raise ValueError("No tenant context set")

        # In production, this would add a filter to the query
        # e.g., query.filter(Document.organization_id == self._current_org_id)
        return query

    async def check_quota(
        self,
        organization_id: str,
        quota_type: str,
        amount: int = 1,
    ) -> bool:
        """Check if organization has quota available."""
        # In production, check against database
        # For now, always allow
        return True

    async def increment_usage(
        self,
        organization_id: str,
        quota_type: str,
        amount: int = 1,
    ) -> None:
        """Increment usage counter for a quota type."""
        logger.info(
            "Usage incremented",
            org_id=organization_id,
            quota_type=quota_type,
            amount=amount,
        )


# =============================================================================
# RBAC Service
# =============================================================================

class RBACService:
    """
    Role-Based Access Control service.

    Handles permission checks and role management.
    """

    def __init__(self, tenant_service: MultiTenantService):
        self.tenant_service = tenant_service
        self._member_cache: Dict[str, OrganizationMember] = {}

    async def get_member(
        self,
        user_id: str,
        organization_id: str,
    ) -> Optional[OrganizationMember]:
        """Get organization membership for a user."""
        cache_key = f"{user_id}:{organization_id}"
        if cache_key in self._member_cache:
            return self._member_cache[cache_key]

        # In production, fetch from database
        # For now, return a mock member
        member = OrganizationMember(
            id=str(uuid.uuid4()),
            user_id=user_id,
            organization_id=organization_id,
            role=Role.EDITOR,  # Default role
        )
        self._member_cache[cache_key] = member
        return member

    async def has_permission(
        self,
        user_id: str,
        organization_id: str,
        permission: Permission,
    ) -> bool:
        """Check if user has a specific permission."""
        member = await self.get_member(user_id, organization_id)
        if member is None:
            return False

        return permission in member.permissions

    async def check_permission(
        self,
        user_id: str,
        organization_id: str,
        permission: Permission,
    ) -> None:
        """Check permission and raise if denied."""
        if not await self.has_permission(user_id, organization_id, permission):
            raise PermissionError(
                f"User {user_id} lacks permission {permission.value} "
                f"in organization {organization_id}"
            )

    async def get_user_permissions(
        self,
        user_id: str,
        organization_id: str,
    ) -> Set[Permission]:
        """Get all permissions for a user in an organization."""
        member = await self.get_member(user_id, organization_id)
        if member is None:
            return set()
        return member.permissions

    async def assign_role(
        self,
        user_id: str,
        organization_id: str,
        role: Role,
        assigned_by: str,
    ) -> OrganizationMember:
        """Assign a role to a user."""
        member = await self.get_member(user_id, organization_id)

        if member:
            member.role = role
        else:
            member = OrganizationMember(
                id=str(uuid.uuid4()),
                user_id=user_id,
                organization_id=organization_id,
                role=role,
                invited_by=assigned_by,
            )

        # Update cache
        cache_key = f"{user_id}:{organization_id}"
        self._member_cache[cache_key] = member

        logger.info(
            "Role assigned",
            user_id=user_id,
            org_id=organization_id,
            role=role.value,
            assigned_by=assigned_by,
        )

        return member

    async def grant_permission(
        self,
        user_id: str,
        organization_id: str,
        permission: Permission,
        granted_by: str,
    ) -> None:
        """Grant an additional permission to a user."""
        member = await self.get_member(user_id, organization_id)
        if member:
            member.custom_permissions.add(permission)

            logger.info(
                "Permission granted",
                user_id=user_id,
                org_id=organization_id,
                permission=permission.value,
                granted_by=granted_by,
            )

    async def revoke_permission(
        self,
        user_id: str,
        organization_id: str,
        permission: Permission,
        revoked_by: str,
    ) -> None:
        """Revoke a custom permission from a user."""
        member = await self.get_member(user_id, organization_id)
        if member and permission in member.custom_permissions:
            member.custom_permissions.remove(permission)

            logger.info(
                "Permission revoked",
                user_id=user_id,
                org_id=organization_id,
                permission=permission.value,
                revoked_by=revoked_by,
            )


# =============================================================================
# Audit Logging Service
# =============================================================================

class AuditLogService:
    """
    Comprehensive audit logging for compliance.

    Captures all significant actions for SOC2, GDPR, etc.
    """

    def __init__(self):
        self._logs: List[AuditLogEntry] = []  # In-memory for demo; use DB in production

    async def log(
        self,
        organization_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log an auditable action."""
        entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            organization_id=organization_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
        )

        self._logs.append(entry)

        # Also log to structured logger
        logger.info(
            "Audit log entry",
            audit_id=entry.id,
            org_id=organization_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
        )

        return entry

    async def get_logs(
        self,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogEntry]:
        """Query audit logs with filters."""
        filtered = [
            log for log in self._logs
            if log.organization_id == organization_id
        ]

        if start_date:
            filtered = [l for l in filtered if l.timestamp >= start_date]
        if end_date:
            filtered = [l for l in filtered if l.timestamp <= end_date]
        if user_id:
            filtered = [l for l in filtered if l.user_id == user_id]
        if action:
            filtered = [l for l in filtered if l.action == action]
        if resource_type:
            filtered = [l for l in filtered if l.resource_type == resource_type]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered[offset:offset + limit]

    async def export_logs(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
    ) -> bytes:
        """Export audit logs for compliance reporting."""
        logs = await self.get_logs(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

        if format == "json":
            data = [
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "details": log.details,
                    "ip_address": log.ip_address,
                    "success": log.success,
                }
                for log in logs
            ]
            return json.dumps(data, indent=2).encode()

        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "id", "timestamp", "user_id", "action",
                "resource_type", "resource_id", "success", "ip_address",
            ])
            for log in logs:
                writer.writerow([
                    log.id, log.timestamp.isoformat(), log.user_id,
                    log.action, log.resource_type, log.resource_id,
                    log.success, log.ip_address,
                ])
            return output.getvalue().encode()

        raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Audit Decorator
# =============================================================================

def audited(
    action: str,
    resource_type: str,
    get_resource_id: Optional[Callable] = None,
):
    """
    Decorator to automatically log function calls.

    Usage:
        @audited("document.create", "document", lambda args: args.get("document_id"))
        async def create_document(user_id, org_id, document_id, ...):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract common parameters
            user_id = kwargs.get("user_id") or (args[0] if args else None)
            org_id = kwargs.get("organization_id") or (args[1] if len(args) > 1 else None)
            resource_id = get_resource_id(kwargs) if get_resource_id else None

            audit_service = get_audit_service()

            try:
                result = await func(*args, **kwargs)

                await audit_service.log(
                    organization_id=org_id,
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    success=True,
                )

                return result

            except Exception as e:
                await audit_service.log(
                    organization_id=org_id,
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    success=False,
                    error_message=str(e),
                )
                raise

        return wrapper
    return decorator


# =============================================================================
# Singleton Instances
# =============================================================================

_tenant_service: Optional[MultiTenantService] = None
_rbac_service: Optional[RBACService] = None
_audit_service: Optional[AuditLogService] = None


def get_tenant_service() -> MultiTenantService:
    """Get tenant service singleton."""
    global _tenant_service
    if _tenant_service is None:
        _tenant_service = MultiTenantService()
    return _tenant_service


def get_rbac_service() -> RBACService:
    """Get RBAC service singleton."""
    global _rbac_service
    if _rbac_service is None:
        _rbac_service = RBACService(get_tenant_service())
    return _rbac_service


def get_audit_service() -> AuditLogService:
    """Get audit service singleton."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditLogService()
    return _audit_service


# =============================================================================
# Context Manager for Tenant Scope
# =============================================================================

class TenantContext:
    """
    Context manager for tenant-scoped operations.

    Usage:
        async with TenantContext(org_id):
            # All operations here are scoped to org_id
            documents = await get_documents()
    """

    def __init__(self, organization_id: str):
        self.organization_id = organization_id
        self._tenant_service = get_tenant_service()
        self._previous_org_id: Optional[str] = None

    async def __aenter__(self):
        self._previous_org_id = self._tenant_service.get_tenant()
        self._tenant_service.set_tenant(self.organization_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._previous_org_id:
            self._tenant_service.set_tenant(self._previous_org_id)
        else:
            self._tenant_service.clear_tenant()


# =============================================================================
# Phase 65: Attribute-Based Access Control (ABAC) for Retrieval
# =============================================================================

@dataclass
class AccessPolicy:
    """
    Access policy for a document or chunk.

    Supports attribute-based access control for fine-grained permissions.
    """
    resource_id: str
    resource_type: str = "document"  # document, chunk, collection

    # Basic permissions
    public: bool = False  # Anyone can access
    organization_only: bool = True  # Only users in the same org

    # Attribute-based rules
    required_roles: List[str] = field(default_factory=list)  # User must have one of these roles
    required_tags: List[str] = field(default_factory=list)  # User must have all these tags
    excluded_users: List[str] = field(default_factory=list)  # Users explicitly denied
    allowed_users: List[str] = field(default_factory=list)  # Users explicitly allowed

    # Time-based rules
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Data classification
    classification: str = "internal"  # public, internal, confidential, restricted

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "public": self.public,
            "organization_only": self.organization_only,
            "required_roles": self.required_roles,
            "required_tags": self.required_tags,
            "excluded_users": self.excluded_users,
            "allowed_users": self.allowed_users,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "classification": self.classification,
        }


@dataclass
class UserAttributes:
    """User attributes for ABAC evaluation."""
    user_id: str
    organization_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    clearance_level: int = 0  # 0=public, 1=internal, 2=confidential, 3=restricted
    is_superadmin: bool = False


class RetrievalAccessControl:
    """
    Access control for retrieval results.

    Filters search results based on user attributes and document policies.
    """

    CLASSIFICATION_LEVELS = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3,
    }

    def __init__(self):
        self._policy_cache: Dict[str, AccessPolicy] = {}

    async def filter_results(
        self,
        user: UserAttributes,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter retrieval results based on access policies.

        Args:
            user: User attributes for evaluation
            results: List of search results (each with 'document_id' and 'access_policy')

        Returns:
            Filtered list of results the user can access
        """
        allowed = []

        for result in results:
            policy = result.get("access_policy")
            if policy is None:
                # No policy = public access by default
                policy = AccessPolicy(
                    resource_id=result.get("document_id", ""),
                    public=True,
                )
            elif isinstance(policy, dict):
                policy = self._dict_to_policy(policy)

            if await self._evaluate_policy(user, policy):
                allowed.append(result)
                result["access_granted"] = True
            else:
                result["access_granted"] = False

        logger.debug(
            "Filtered retrieval results",
            total=len(results),
            allowed=len(allowed),
            user_id=user.user_id,
        )

        return allowed

    def _dict_to_policy(self, policy_dict: Dict[str, Any]) -> AccessPolicy:
        """Convert dictionary to AccessPolicy."""
        return AccessPolicy(
            resource_id=policy_dict.get("resource_id", ""),
            resource_type=policy_dict.get("resource_type", "document"),
            public=policy_dict.get("public", False),
            organization_only=policy_dict.get("organization_only", True),
            required_roles=policy_dict.get("required_roles", []),
            required_tags=policy_dict.get("required_tags", []),
            excluded_users=policy_dict.get("excluded_users", []),
            allowed_users=policy_dict.get("allowed_users", []),
            classification=policy_dict.get("classification", "internal"),
        )

    async def _evaluate_policy(
        self,
        user: UserAttributes,
        policy: AccessPolicy,
    ) -> bool:
        """
        Evaluate if user can access resource based on policy.

        Returns True if access should be granted.
        """
        # Superadmin bypasses all checks
        if user.is_superadmin:
            return True

        # Check explicit exclusion first
        if user.user_id in policy.excluded_users:
            return False

        # Check explicit allowlist
        if policy.allowed_users and user.user_id in policy.allowed_users:
            return True

        # Public resources are accessible to all
        if policy.public:
            return True

        # Time-based validation
        now = datetime.utcnow()
        if policy.valid_from and now < policy.valid_from:
            return False
        if policy.valid_until and now > policy.valid_until:
            return False

        # Organization check
        if policy.organization_only and user.organization_id is None:
            return False

        # Classification level check
        user_level = user.clearance_level
        required_level = self.CLASSIFICATION_LEVELS.get(policy.classification, 0)
        if user_level < required_level:
            return False

        # Role check
        if policy.required_roles:
            if not any(role in user.roles for role in policy.required_roles):
                return False

        # Tag check (must have all required tags)
        if policy.required_tags:
            if not all(tag in user.tags for tag in policy.required_tags):
                return False

        return True

    async def check_access(
        self,
        user: UserAttributes,
        resource_id: str,
        resource_type: str = "document",
    ) -> bool:
        """Check if user can access a specific resource."""
        # Get policy from cache or storage
        policy = self._policy_cache.get(resource_id)

        if policy is None:
            # Default policy: organization-only access
            policy = AccessPolicy(
                resource_id=resource_id,
                resource_type=resource_type,
                organization_only=True,
            )

        return await self._evaluate_policy(user, policy)

    def set_policy(self, resource_id: str, policy: AccessPolicy) -> None:
        """Set access policy for a resource."""
        self._policy_cache[resource_id] = policy

    def clear_policy(self, resource_id: str) -> None:
        """Clear access policy for a resource."""
        self._policy_cache.pop(resource_id, None)


# Singleton for retrieval access control
_retrieval_access_control: Optional[RetrievalAccessControl] = None


def get_retrieval_access_control() -> RetrievalAccessControl:
    """Get retrieval access control singleton."""
    global _retrieval_access_control
    if _retrieval_access_control is None:
        _retrieval_access_control = RetrievalAccessControl()
    return _retrieval_access_control


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    # Data models
    "Organization",
    "OrganizationMember",
    "AuditLogEntry",
    "UsageQuota",
    # Services
    "MultiTenantService",
    "RBACService",
    "AuditLogService",
    # Singletons
    "get_tenant_service",
    "get_rbac_service",
    "get_audit_service",
    # Utilities
    "TenantContext",
    "audited",
    # Phase 65: ABAC for Retrieval
    "AccessPolicy",
    "UserAttributes",
    "RetrievalAccessControl",
    "get_retrieval_access_control",
]
