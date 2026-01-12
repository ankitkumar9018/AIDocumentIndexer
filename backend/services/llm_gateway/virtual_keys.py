"""
AIDocumentIndexer - Virtual API Key Management
===============================================

Manage virtual API keys for controlled LLM access.

Features:
- Scoped access (allowed models, rate limits)
- Usage tracking per key
- Expiration and revocation
- Key rotation
"""

import secrets
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.base import BaseService, ServiceException, NotFoundException

logger = structlog.get_logger(__name__)


@dataclass
class KeyScope:
    """Defines what a virtual key can access."""
    allowed_models: Set[str] = field(default_factory=set)  # Empty = all
    max_tokens_per_request: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    max_requests_per_day: Optional[int] = None
    max_cost_per_request: Optional[float] = None
    allowed_endpoints: Set[str] = field(default_factory=set)  # Empty = all

    def allows_model(self, model: str) -> bool:
        """Check if model is allowed."""
        if not self.allowed_models:
            return True
        return model in self.allowed_models

    def allows_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint is allowed."""
        if not self.allowed_endpoints:
            return True
        return endpoint in self.allowed_endpoints


@dataclass
class VirtualApiKey:
    """A virtual API key for LLM access."""
    id: str
    organization_id: str
    user_id: Optional[str]
    name: str
    key_prefix: str  # First 8 chars for identification
    key_hash: str  # SHA-256 hash of full key
    scope: KeyScope
    expires_at: Optional[datetime]
    is_active: bool = True
    usage_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    last_used_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return self.expires_at <= datetime.utcnow()

    @property
    def is_valid(self) -> bool:
        """Check if key is valid for use."""
        return self.is_active and not self.is_expired


class VirtualKeyManager(BaseService):
    """
    Manages virtual API keys for LLM access control.

    Virtual keys allow:
    - Scoped access to specific models
    - Rate limiting per key
    - Usage tracking and analytics
    - Easy revocation without changing actual API keys
    """

    # Key format: vk_{prefix}_{random}
    KEY_PREFIX = "vk"

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__(session, organization_id, user_id)
        # In-memory cache for fast validation
        self._key_cache: Dict[str, VirtualApiKey] = {}
        self._hash_to_key: Dict[str, str] = {}  # hash -> key_id mapping

    def _generate_key(self) -> tuple[str, str, str]:
        """
        Generate a new virtual API key.

        Returns:
            tuple of (full_key, prefix, hash)
        """
        # Generate random bytes
        random_part = secrets.token_urlsafe(32)
        prefix = secrets.token_hex(4)  # 8 char prefix

        full_key = f"{self.KEY_PREFIX}_{prefix}_{random_part}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        return full_key, prefix, key_hash

    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def create_key(
        self,
        name: str,
        user_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        allowed_models: Optional[List[str]] = None,
        max_tokens_per_request: Optional[int] = None,
        max_requests_per_minute: Optional[int] = None,
        max_requests_per_day: Optional[int] = None,
        max_cost_per_request: Optional[float] = None,
    ) -> tuple[VirtualApiKey, str]:
        """
        Create a new virtual API key.

        Args:
            name: Key name/description
            user_id: Owner user ID
            expires_in_days: Days until expiration (None = never)
            allowed_models: List of allowed model names
            max_tokens_per_request: Token limit per request
            max_requests_per_minute: Rate limit per minute
            max_requests_per_day: Daily request limit
            max_cost_per_request: Cost limit per request

        Returns:
            tuple of (VirtualApiKey, plain_key)
            Note: plain_key is only returned once and cannot be retrieved later
        """
        session = await self.get_session()

        # Generate key
        full_key, prefix, key_hash = self._generate_key()

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Build scope
        scope = KeyScope(
            allowed_models=set(allowed_models) if allowed_models else set(),
            max_tokens_per_request=max_tokens_per_request,
            max_requests_per_minute=max_requests_per_minute,
            max_requests_per_day=max_requests_per_day,
            max_cost_per_request=max_cost_per_request,
        )

        key_id = str(uuid.uuid4())

        virtual_key = VirtualApiKey(
            id=key_id,
            organization_id=str(self._organization_id),
            user_id=user_id,
            name=name,
            key_prefix=prefix,
            key_hash=key_hash,
            scope=scope,
            expires_at=expires_at,
            is_active=True,
        )

        # Store in database
        from backend.db.models import VirtualApiKey as VirtualApiKeyModel

        db_key = VirtualApiKeyModel(
            id=uuid.UUID(key_id),
            organization_id=self._organization_id,
            user_id=uuid.UUID(user_id) if user_id else None,
            name=name,
            key_prefix=prefix,
            key_hash=key_hash,
            scope_config={
                "allowed_models": list(scope.allowed_models),
                "max_tokens_per_request": scope.max_tokens_per_request,
                "max_requests_per_minute": scope.max_requests_per_minute,
                "max_requests_per_day": scope.max_requests_per_day,
                "max_cost_per_request": scope.max_cost_per_request,
            },
            expires_at=expires_at,
            is_active=True,
        )

        session.add(db_key)
        await session.commit()

        # Update cache
        self._key_cache[key_id] = virtual_key
        self._hash_to_key[key_hash] = key_id

        self.log_info(
            "Virtual API key created",
            key_id=key_id,
            name=name,
            prefix=prefix,
            expires_at=expires_at.isoformat() if expires_at else None,
        )

        # Return key and plain text (only time it's available)
        return virtual_key, full_key

    async def validate_key(
        self,
        key: str,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> tuple[bool, Optional[VirtualApiKey], Optional[str]]:
        """
        Validate a virtual API key.

        Args:
            key: The API key to validate
            model: Model being accessed (for scope check)
            endpoint: Endpoint being accessed (for scope check)

        Returns:
            tuple of (is_valid, VirtualApiKey or None, error_message or None)
        """
        # Check key format
        if not key.startswith(f"{self.KEY_PREFIX}_"):
            return False, None, "Invalid key format"

        # Hash the key
        key_hash = self._hash_key(key)

        # Check cache first
        key_id = self._hash_to_key.get(key_hash)
        if key_id and key_id in self._key_cache:
            virtual_key = self._key_cache[key_id]
        else:
            # Load from database
            virtual_key = await self._load_key_by_hash(key_hash)
            if not virtual_key:
                return False, None, "Key not found"

        # Check if active
        if not virtual_key.is_active:
            return False, virtual_key, "Key is deactivated"

        # Check expiration
        if virtual_key.is_expired:
            return False, virtual_key, "Key has expired"

        # Check model scope
        if model and not virtual_key.scope.allows_model(model):
            return False, virtual_key, f"Model '{model}' not allowed for this key"

        # Check endpoint scope
        if endpoint and not virtual_key.scope.allows_endpoint(endpoint):
            return False, virtual_key, f"Endpoint '{endpoint}' not allowed for this key"

        return True, virtual_key, None

    async def record_usage(
        self,
        key_id: str,
        tokens: int = 0,
        cost: float = 0.0,
    ):
        """
        Record usage for a virtual key.

        Args:
            key_id: Key ID
            tokens: Tokens used
            cost: Cost in USD
        """
        session = await self.get_session()
        from backend.db.models import VirtualApiKey as VirtualApiKeyModel

        result = await session.execute(
            select(VirtualApiKeyModel).where(VirtualApiKeyModel.id == uuid.UUID(key_id))
        )
        db_key = result.scalar_one_or_none()

        if db_key:
            db_key.usage_count = (db_key.usage_count or 0) + 1
            db_key.total_tokens = (db_key.total_tokens or 0) + tokens
            db_key.total_cost = (db_key.total_cost or 0) + cost
            db_key.last_used_at = datetime.utcnow()
            await session.commit()

            # Update cache
            if key_id in self._key_cache:
                self._key_cache[key_id].usage_count += 1
                self._key_cache[key_id].total_tokens += tokens
                self._key_cache[key_id].total_cost += cost
                self._key_cache[key_id].last_used_at = datetime.utcnow()

    async def list_keys(
        self,
        user_id: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[VirtualApiKey]:
        """List all virtual keys for the organization."""
        session = await self.get_session()
        from backend.db.models import VirtualApiKey as VirtualApiKeyModel

        query = select(VirtualApiKeyModel).where(
            VirtualApiKeyModel.organization_id == self._organization_id
        )

        if user_id:
            query = query.where(VirtualApiKeyModel.user_id == uuid.UUID(user_id))

        if not include_inactive:
            query = query.where(VirtualApiKeyModel.is_active == True)

        query = query.order_by(VirtualApiKeyModel.created_at.desc())

        result = await session.execute(query)
        db_keys = result.scalars().all()

        return [self._db_to_virtual_key(k) for k in db_keys]

    async def get_key(self, key_id: str) -> Optional[VirtualApiKey]:
        """Get a virtual key by ID."""
        if key_id in self._key_cache:
            return self._key_cache[key_id]

        session = await self.get_session()
        from backend.db.models import VirtualApiKey as VirtualApiKeyModel

        result = await session.execute(
            select(VirtualApiKeyModel).where(
                VirtualApiKeyModel.id == uuid.UUID(key_id),
                VirtualApiKeyModel.organization_id == self._organization_id,
            )
        )
        db_key = result.scalar_one_or_none()

        if not db_key:
            return None

        virtual_key = self._db_to_virtual_key(db_key)
        self._key_cache[key_id] = virtual_key
        self._hash_to_key[virtual_key.key_hash] = key_id
        return virtual_key

    async def revoke_key(self, key_id: str):
        """Revoke/deactivate a virtual key."""
        session = await self.get_session()
        from backend.db.models import VirtualApiKey as VirtualApiKeyModel

        result = await session.execute(
            select(VirtualApiKeyModel).where(
                VirtualApiKeyModel.id == uuid.UUID(key_id),
                VirtualApiKeyModel.organization_id == self._organization_id,
            )
        )
        db_key = result.scalar_one_or_none()

        if not db_key:
            raise NotFoundException("VirtualApiKey", key_id)

        db_key.is_active = False
        db_key.updated_at = datetime.utcnow()
        await session.commit()

        # Update cache
        if key_id in self._key_cache:
            self._key_cache[key_id].is_active = False

        self.log_info("Virtual API key revoked", key_id=key_id)

    async def delete_key(self, key_id: str):
        """Permanently delete a virtual key."""
        session = await self.get_session()
        from backend.db.models import VirtualApiKey as VirtualApiKeyModel

        result = await session.execute(
            select(VirtualApiKeyModel).where(
                VirtualApiKeyModel.id == uuid.UUID(key_id),
                VirtualApiKeyModel.organization_id == self._organization_id,
            )
        )
        db_key = result.scalar_one_or_none()

        if not db_key:
            raise NotFoundException("VirtualApiKey", key_id)

        key_hash = db_key.key_hash
        await session.delete(db_key)
        await session.commit()

        # Remove from cache
        self._key_cache.pop(key_id, None)
        self._hash_to_key.pop(key_hash, None)

        self.log_info("Virtual API key deleted", key_id=key_id)

    async def rotate_key(self, key_id: str) -> tuple[VirtualApiKey, str]:
        """
        Rotate a virtual key (create new key with same settings, revoke old).

        Args:
            key_id: Key to rotate

        Returns:
            tuple of (new_VirtualApiKey, new_plain_key)
        """
        old_key = await self.get_key(key_id)
        if not old_key:
            raise NotFoundException("VirtualApiKey", key_id)

        # Create new key with same settings
        new_key, plain_key = await self.create_key(
            name=f"{old_key.name} (rotated)",
            user_id=old_key.user_id,
            expires_in_days=None if not old_key.expires_at else
                (old_key.expires_at - datetime.utcnow()).days,
            allowed_models=list(old_key.scope.allowed_models) if old_key.scope.allowed_models else None,
            max_tokens_per_request=old_key.scope.max_tokens_per_request,
            max_requests_per_minute=old_key.scope.max_requests_per_minute,
            max_requests_per_day=old_key.scope.max_requests_per_day,
            max_cost_per_request=old_key.scope.max_cost_per_request,
        )

        # Revoke old key
        await self.revoke_key(key_id)

        self.log_info(
            "Virtual API key rotated",
            old_key_id=key_id,
            new_key_id=new_key.id,
        )

        return new_key, plain_key

    async def _load_key_by_hash(self, key_hash: str) -> Optional[VirtualApiKey]:
        """Load a key from database by hash."""
        session = await self.get_session()
        from backend.db.models import VirtualApiKey as VirtualApiKeyModel

        result = await session.execute(
            select(VirtualApiKeyModel).where(VirtualApiKeyModel.key_hash == key_hash)
        )
        db_key = result.scalar_one_or_none()

        if not db_key:
            return None

        virtual_key = self._db_to_virtual_key(db_key)

        # Update cache
        self._key_cache[virtual_key.id] = virtual_key
        self._hash_to_key[key_hash] = virtual_key.id

        return virtual_key

    def _db_to_virtual_key(self, db_key) -> VirtualApiKey:
        """Convert database model to VirtualApiKey."""
        scope_config = db_key.scope_config or {}

        scope = KeyScope(
            allowed_models=set(scope_config.get("allowed_models", [])),
            max_tokens_per_request=scope_config.get("max_tokens_per_request"),
            max_requests_per_minute=scope_config.get("max_requests_per_minute"),
            max_requests_per_day=scope_config.get("max_requests_per_day"),
            max_cost_per_request=scope_config.get("max_cost_per_request"),
        )

        return VirtualApiKey(
            id=str(db_key.id),
            organization_id=str(db_key.organization_id),
            user_id=str(db_key.user_id) if db_key.user_id else None,
            name=db_key.name,
            key_prefix=db_key.key_prefix,
            key_hash=db_key.key_hash,
            scope=scope,
            expires_at=db_key.expires_at,
            is_active=db_key.is_active,
            usage_count=db_key.usage_count or 0,
            total_tokens=db_key.total_tokens or 0,
            total_cost=float(db_key.total_cost or 0),
            last_used_at=db_key.last_used_at,
            created_at=db_key.created_at,
        )
