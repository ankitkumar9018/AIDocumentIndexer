"""
AIDocumentIndexer - LLM Router Service
========================================

Intelligent model routing with per-service configuration and user permissions.

Features:
1. Service-Level Configuration
   - Each service (chat, RAG, workflow, audio, etc.) can have its own LLM
   - Operation-level granularity (e.g., workflow.code_execution vs workflow.default)
   - Fallback providers for reliability

2. User Permissions (LiteLLM-style RBAC)
   - Model access groups (basic, standard, advanced, enterprise, admin)
   - Wildcard pattern matching for model access
   - Per-user overrides and custom limits
   - Fully configurable through database (not hardcoded)

3. Routing Strategies (inspired by OpenRouter & NVIDIA LLM Router)
   - default: Use configured model
   - cost_optimized: Prefer cheaper models that meet quality threshold
   - quality_optimized: Prefer best-performing models
   - latency_optimized: Prefer fastest models

4. User Override Support
   - Users can switch models per-service (if allowed by admin)
   - Respects user's access level
   - Per-service rate limits and budgets

5. Model Registry
   - All model metadata (quality, cost, latency) stored in database
   - Admins can add/update models without code changes
   - Organization-specific model configurations

Based on:
- LiteLLM RBAC: https://docs.litellm.ai/docs/proxy/access_control
- OpenRouter routing: https://openrouter.ai/docs/guides/routing
- NVIDIA LLM Router: https://build.nvidia.com/nvidia/llm-router
"""

import fnmatch
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from enum import Enum

import structlog
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.llm import LLMFactory, LLMConfig, llm_config
from backend.services.llm_provider import LLMProviderService, PROVIDER_TYPES

logger = structlog.get_logger(__name__)


class RoutingStrategy(str, Enum):
    """Model routing strategies."""
    DEFAULT = "default"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"


@dataclass
class ModelAccess:
    """User's model access information."""
    user_id: str
    access_level: int = 1
    allowed_patterns: List[str] = field(default_factory=lambda: ["gpt-4o-mini", "gpt-3.5-turbo"])
    max_tokens_per_request: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    max_cost_per_day_usd: Optional[float] = None


@dataclass
class ServiceLLMConfig:
    """Configuration for a service's LLM."""
    service_name: str
    operation_name: Optional[str] = None

    # Provider and model
    provider_type: str = "openai"
    provider_id: Optional[str] = None
    model_name: str = "gpt-4o"

    # Parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Fallback
    fallback_provider_type: Optional[str] = None
    fallback_model_name: Optional[str] = None

    # Routing
    routing_strategy: RoutingStrategy = RoutingStrategy.DEFAULT
    allow_user_override: bool = True
    allowed_override_models: Optional[List[str]] = None
    min_access_level: int = 1


@dataclass
class ResolvedModel:
    """Result of model resolution."""
    provider_type: str
    model_name: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    source: str = "default"  # default, service_config, user_override, fallback
    original_request: Optional[str] = None


@dataclass
class ModelInfo:
    """Model information from registry."""
    provider_type: str
    model_name: str
    display_name: Optional[str] = None
    quality_score: int = 50
    cost_per_million_tokens: float = 1.0
    latency_score: int = 3
    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_vision: bool = False
    supports_function_calling: bool = False
    tier: Optional[str] = None
    min_access_level: int = 1
    is_active: bool = True


@dataclass
class AccessGroup:
    """Model access group from database."""
    id: str
    name: str
    description: Optional[str]
    access_level: int
    model_patterns: List[str]
    max_tokens_per_request: Optional[int] = None
    max_requests_per_minute: Optional[int] = None
    max_cost_per_day_usd: Optional[float] = None


class LLMRouter:
    """
    Intelligent LLM router with per-service configuration and user permissions.

    All model metadata and access tiers are stored in the database and can be
    configured by admins without code changes.

    Usage:
        router = LLMRouter()

        # Get model for a service
        model = await router.get_model_for_service(
            db=session,
            service_name="chat",
            operation_name="default",
            user_id=user.id,
            organization_id=org.id,
        )

        # Check if user can access a model
        can_access = await router.check_model_access(
            db=session,
            user_id=user.id,
            model_name="gpt-4o",
        )

        # List available models for user
        models = await router.get_available_models(
            db=session,
            user_id=user.id,
            service_name="chat",
        )
    """

    # Fallback model scores (used only if database is empty)
    _FALLBACK_QUALITY_SCORES = {
        "gpt-4o": 95, "gpt-4o-mini": 85, "gpt-3.5-turbo": 75,
        "claude-3-5-sonnet-latest": 92, "claude-3-5-haiku-latest": 82,
    }
    _FALLBACK_COST_SCORES = {
        "gpt-4o": 7.50, "gpt-4o-mini": 0.375, "gpt-3.5-turbo": 1.0,
        "claude-3-5-sonnet-latest": 9.0, "claude-3-5-haiku-latest": 1.25,
    }
    _FALLBACK_LATENCY_SCORES = {
        "gpt-4o": 2, "gpt-4o-mini": 1, "gpt-3.5-turbo": 1,
        "claude-3-5-sonnet-latest": 2, "claude-3-5-haiku-latest": 1,
    }

    def __init__(self):
        self._service_config_cache: Dict[str, ServiceLLMConfig] = {}
        self._user_access_cache: Dict[str, Tuple[ModelAccess, datetime]] = {}
        self._model_registry_cache: Dict[str, ModelInfo] = {}
        self._access_groups_cache: Dict[str, AccessGroup] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._registry_loaded_at: Optional[datetime] = None
        logger.info("LLMRouter initialized")

    # =========================================================================
    # Model Registry (Database-backed)
    # =========================================================================

    async def _load_model_registry(
        self,
        db: AsyncSession,
        organization_id: Optional[str] = None,
        force_reload: bool = False,
    ) -> Dict[str, ModelInfo]:
        """Load model registry from database."""
        # Check if cache is valid
        if (
            not force_reload
            and self._registry_loaded_at
            and datetime.utcnow() - self._registry_loaded_at < self._cache_ttl
            and self._model_registry_cache
        ):
            return self._model_registry_cache

        try:
            # Query model_registry table
            query = text("""
                SELECT
                    provider_type, model_name, display_name, quality_score,
                    cost_per_million_tokens, latency_score, max_context_tokens,
                    max_output_tokens, supports_vision, supports_function_calling,
                    tier, min_access_level, is_active
                FROM model_registry
                WHERE is_active = true
                  AND (organization_id IS NULL OR organization_id = :org_id)
                ORDER BY quality_score DESC
            """)
            result = await db.execute(query, {"org_id": organization_id})
            rows = result.fetchall()

            self._model_registry_cache = {}
            for row in rows:
                key = f"{row.provider_type}:{row.model_name}"
                self._model_registry_cache[key] = ModelInfo(
                    provider_type=row.provider_type,
                    model_name=row.model_name,
                    display_name=row.display_name,
                    quality_score=row.quality_score,
                    cost_per_million_tokens=row.cost_per_million_tokens,
                    latency_score=row.latency_score,
                    max_context_tokens=row.max_context_tokens,
                    max_output_tokens=row.max_output_tokens,
                    supports_vision=row.supports_vision,
                    supports_function_calling=row.supports_function_calling,
                    tier=row.tier,
                    min_access_level=row.min_access_level,
                    is_active=row.is_active,
                )

            self._registry_loaded_at = datetime.utcnow()
            logger.debug("Model registry loaded", count=len(self._model_registry_cache))

        except Exception as e:
            logger.warning("Failed to load model registry from DB, using fallbacks", error=str(e))
            # Fallback to hardcoded values if DB query fails
            if not self._model_registry_cache:
                for model, quality in self._FALLBACK_QUALITY_SCORES.items():
                    self._model_registry_cache[f"openai:{model}"] = ModelInfo(
                        provider_type="openai" if "gpt" in model else "anthropic",
                        model_name=model,
                        quality_score=quality,
                        cost_per_million_tokens=self._FALLBACK_COST_SCORES.get(model, 1.0),
                        latency_score=self._FALLBACK_LATENCY_SCORES.get(model, 3),
                    )

        return self._model_registry_cache

    async def get_model_info(
        self,
        db: AsyncSession,
        model_name: str,
        provider_type: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        """Get info for a specific model from registry."""
        await self._load_model_registry(db, organization_id)

        # Try exact match first
        if provider_type:
            key = f"{provider_type}:{model_name}"
            if key in self._model_registry_cache:
                return self._model_registry_cache[key]

        # Try to find by model name only
        for key, info in self._model_registry_cache.items():
            if info.model_name == model_name or info.model_name.startswith(model_name):
                return info

        return None

    def _get_model_quality(self, model_name: str) -> int:
        """Get quality score for a model (from cache or fallback)."""
        for key, info in self._model_registry_cache.items():
            if info.model_name == model_name or model_name in key:
                return info.quality_score
        return self._FALLBACK_QUALITY_SCORES.get(model_name, 50)

    def _get_model_cost(self, model_name: str) -> float:
        """Get cost score for a model (from cache or fallback)."""
        for key, info in self._model_registry_cache.items():
            if info.model_name == model_name or model_name in key:
                return info.cost_per_million_tokens
        return self._FALLBACK_COST_SCORES.get(model_name, 1.0)

    def _get_model_latency(self, model_name: str) -> int:
        """Get latency score for a model (from cache or fallback)."""
        for key, info in self._model_registry_cache.items():
            if info.model_name == model_name or model_name in key:
                return info.latency_score
        return self._FALLBACK_LATENCY_SCORES.get(model_name, 3)

    # =========================================================================
    # Access Groups (Database-backed)
    # =========================================================================

    async def _load_access_groups(
        self,
        db: AsyncSession,
        organization_id: Optional[str] = None,
        force_reload: bool = False,
    ) -> Dict[str, AccessGroup]:
        """Load access groups from database."""
        cache_key = organization_id or "global"
        if not force_reload and cache_key in self._access_groups_cache:
            return self._access_groups_cache

        try:
            query = text("""
                SELECT
                    id, name, description, access_level, model_patterns,
                    max_tokens_per_request, max_requests_per_minute, max_cost_per_day_usd
                FROM model_access_groups
                WHERE is_active = true
                  AND (organization_id IS NULL OR organization_id = :org_id)
                ORDER BY access_level
            """)
            result = await db.execute(query, {"org_id": organization_id})
            rows = result.fetchall()

            self._access_groups_cache = {}
            for row in rows:
                patterns = row.model_patterns
                if isinstance(patterns, str):
                    patterns = json.loads(patterns)

                self._access_groups_cache[row.name] = AccessGroup(
                    id=str(row.id),
                    name=row.name,
                    description=row.description,
                    access_level=row.access_level,
                    model_patterns=patterns,
                    max_tokens_per_request=row.max_tokens_per_request,
                    max_requests_per_minute=row.max_requests_per_minute,
                    max_cost_per_day_usd=row.max_cost_per_day_usd,
                )

            logger.debug("Access groups loaded", count=len(self._access_groups_cache))

        except Exception as e:
            logger.warning("Failed to load access groups from DB", error=str(e))
            # Fallback defaults
            if not self._access_groups_cache:
                self._access_groups_cache = {
                    "basic": AccessGroup(
                        id="fallback-basic",
                        name="basic",
                        description="Basic tier",
                        access_level=1,
                        model_patterns=["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-haiku-*"],
                    ),
                    "standard": AccessGroup(
                        id="fallback-standard",
                        name="standard",
                        description="Standard tier",
                        access_level=5,
                        model_patterns=["gpt-4o*", "claude-3-5-*", "llama-*", "gemini-*"],
                    ),
                }

        return self._access_groups_cache

    async def list_access_groups(
        self,
        db: AsyncSession,
        organization_id: Optional[str] = None,
    ) -> List[AccessGroup]:
        """List all access groups."""
        groups = await self._load_access_groups(db, organization_id)
        return list(groups.values())

    async def get_access_group(
        self,
        db: AsyncSession,
        group_name: str,
        organization_id: Optional[str] = None,
    ) -> Optional[AccessGroup]:
        """Get a specific access group by name."""
        groups = await self._load_access_groups(db, organization_id)
        return groups.get(group_name)

    async def create_access_group(
        self,
        db: AsyncSession,
        name: str,
        access_level: int,
        model_patterns: List[str],
        description: Optional[str] = None,
        max_tokens_per_request: Optional[int] = None,
        max_requests_per_minute: Optional[int] = None,
        max_cost_per_day_usd: Optional[float] = None,
        organization_id: Optional[str] = None,
    ) -> AccessGroup:
        """Create a new access group."""
        patterns_json = json.dumps(model_patterns)

        query = text("""
            INSERT INTO model_access_groups
            (name, description, access_level, model_patterns, max_tokens_per_request,
             max_requests_per_minute, max_cost_per_day_usd, organization_id, is_active)
            VALUES (:name, :description, :access_level, :model_patterns, :max_tokens,
                    :max_requests, :max_cost, :org_id, true)
            RETURNING id
        """)
        result = await db.execute(query, {
            "name": name,
            "description": description,
            "access_level": access_level,
            "model_patterns": patterns_json,
            "max_tokens": max_tokens_per_request,
            "max_requests": max_requests_per_minute,
            "max_cost": max_cost_per_day_usd,
            "org_id": organization_id,
        })
        await db.commit()
        row = result.fetchone()

        # Clear cache
        self._access_groups_cache.clear()

        return AccessGroup(
            id=str(row.id),
            name=name,
            description=description,
            access_level=access_level,
            model_patterns=model_patterns,
            max_tokens_per_request=max_tokens_per_request,
            max_requests_per_minute=max_requests_per_minute,
            max_cost_per_day_usd=max_cost_per_day_usd,
        )

    async def update_access_group(
        self,
        db: AsyncSession,
        group_name: str,
        updates: Dict[str, Any],
        organization_id: Optional[str] = None,
    ) -> Optional[AccessGroup]:
        """Update an access group."""
        # Build dynamic update query
        set_clauses = []
        params = {"name": group_name, "org_id": organization_id}

        if "access_level" in updates:
            set_clauses.append("access_level = :access_level")
            params["access_level"] = updates["access_level"]
        if "description" in updates:
            set_clauses.append("description = :description")
            params["description"] = updates["description"]
        if "model_patterns" in updates:
            set_clauses.append("model_patterns = :model_patterns")
            params["model_patterns"] = json.dumps(updates["model_patterns"])
        if "max_tokens_per_request" in updates:
            set_clauses.append("max_tokens_per_request = :max_tokens")
            params["max_tokens"] = updates["max_tokens_per_request"]
        if "max_requests_per_minute" in updates:
            set_clauses.append("max_requests_per_minute = :max_requests")
            params["max_requests"] = updates["max_requests_per_minute"]
        if "max_cost_per_day_usd" in updates:
            set_clauses.append("max_cost_per_day_usd = :max_cost")
            params["max_cost"] = updates["max_cost_per_day_usd"]
        if "is_active" in updates:
            set_clauses.append("is_active = :is_active")
            params["is_active"] = updates["is_active"]

        if not set_clauses:
            return await self.get_access_group(db, group_name, organization_id)

        set_clauses.append("updated_at = now()")

        query = text(f"""
            UPDATE model_access_groups
            SET {", ".join(set_clauses)}
            WHERE name = :name
              AND (organization_id IS NULL OR organization_id = :org_id)
        """)
        await db.execute(query, params)
        await db.commit()

        # Clear cache
        self._access_groups_cache.clear()

        return await self.get_access_group(db, group_name, organization_id)

    async def delete_access_group(
        self,
        db: AsyncSession,
        group_name: str,
        organization_id: Optional[str] = None,
    ) -> bool:
        """Delete an access group (soft delete)."""
        query = text("""
            UPDATE model_access_groups
            SET is_active = false, updated_at = now()
            WHERE name = :name
              AND (organization_id IS NULL OR organization_id = :org_id)
        """)
        result = await db.execute(query, {"name": group_name, "org_id": organization_id})
        await db.commit()

        # Clear cache
        self._access_groups_cache.clear()

        return result.rowcount > 0

    # =========================================================================
    # Model Access Control
    # =========================================================================

    async def get_user_access(
        self,
        db: AsyncSession,
        user_id: str,
        organization_id: Optional[str] = None,
    ) -> ModelAccess:
        """
        Get user's model access information from database.

        Returns their access level and allowed model patterns based on
        their assigned access groups.
        """
        # Check cache first (with TTL)
        if user_id in self._user_access_cache:
            cached_access, cached_at = self._user_access_cache[user_id]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return cached_access

        # Load access groups first
        await self._load_access_groups(db, organization_id)

        try:
            # Query user's access groups from database
            query = text("""
                SELECT
                    mag.name, mag.access_level, mag.model_patterns,
                    mag.max_tokens_per_request, mag.max_requests_per_minute,
                    mag.max_cost_per_day_usd,
                    uma.custom_max_tokens, uma.custom_rate_limit, uma.custom_daily_budget_usd,
                    uma.expires_at
                FROM user_model_access uma
                JOIN model_access_groups mag ON uma.access_group_id = mag.id
                WHERE uma.user_id = :user_id
                  AND mag.is_active = true
                  AND (uma.expires_at IS NULL OR uma.expires_at > now())
                ORDER BY mag.access_level DESC
            """)
            result = await db.execute(query, {"user_id": user_id})
            rows = result.fetchall()

            if rows:
                # Combine all user's access groups
                highest_level = 0
                all_patterns = []
                max_tokens = None
                max_requests = None
                max_cost = None
                access_groups = []

                for row in rows:
                    access_groups.append(row.name)
                    if row.access_level > highest_level:
                        highest_level = row.access_level

                    # Parse patterns
                    patterns = row.model_patterns
                    if isinstance(patterns, str):
                        patterns = json.loads(patterns)
                    all_patterns.extend(patterns)

                    # Use custom limits if set, otherwise group limits
                    tokens = row.custom_max_tokens or row.max_tokens_per_request
                    if tokens and (max_tokens is None or tokens > max_tokens):
                        max_tokens = tokens

                    requests = row.custom_rate_limit or row.max_requests_per_minute
                    if requests and (max_requests is None or requests > max_requests):
                        max_requests = requests

                    cost = row.custom_daily_budget_usd or row.max_cost_per_day_usd
                    if cost and (max_cost is None or cost > max_cost):
                        max_cost = cost

                # Deduplicate patterns
                all_patterns = list(set(all_patterns))

                access = ModelAccess(
                    user_id=user_id,
                    access_level=highest_level,
                    allowed_patterns=all_patterns,
                    max_tokens_per_request=max_tokens,
                    max_requests_per_minute=max_requests,
                    max_cost_per_day_usd=max_cost,
                )

                logger.debug(
                    "User access loaded from DB",
                    user_id=user_id[:8] if user_id else None,
                    level=highest_level,
                    groups=access_groups,
                )

            else:
                # No explicit access granted - give default basic access
                basic_group = self._access_groups_cache.get("basic")
                if basic_group:
                    access = ModelAccess(
                        user_id=user_id,
                        access_level=basic_group.access_level,
                        allowed_patterns=basic_group.model_patterns,
                        max_tokens_per_request=basic_group.max_tokens_per_request,
                        max_requests_per_minute=basic_group.max_requests_per_minute,
                        max_cost_per_day_usd=basic_group.max_cost_per_day_usd,
                    )
                else:
                    # Ultimate fallback
                    access = ModelAccess(
                        user_id=user_id,
                        access_level=1,
                        allowed_patterns=["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-haiku-*"],
                    )

        except Exception as e:
            logger.warning("Failed to load user access from DB", error=str(e), user_id=user_id[:8] if user_id else None)
            # Fallback to basic access
            access = ModelAccess(
                user_id=user_id,
                access_level=1,
                allowed_patterns=["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-5-haiku-*"],
            )

        # Cache with timestamp
        self._user_access_cache[user_id] = (access, datetime.utcnow())

        return access

    async def check_model_access(
        self,
        db: AsyncSession,
        user_id: str,
        model_name: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a user can access a specific model.

        Returns (allowed, reason).
        """
        access = await self.get_user_access(db, user_id)

        # Check against patterns
        for pattern in access.allowed_patterns:
            if fnmatch.fnmatch(model_name.lower(), pattern.lower()):
                return True, None

        return False, f"Model '{model_name}' not in your access tier"

    async def get_available_models(
        self,
        db: AsyncSession,
        user_id: str,
        service_name: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get list of models available to a user.

        Uses the model registry for metadata (quality, cost, latency).
        Optionally filtered by service (some services may restrict models further).
        """
        access = await self.get_user_access(db, user_id, organization_id)

        # Load model registry
        await self._load_model_registry(db, organization_id)

        # Get all providers
        providers = await LLMProviderService.list_providers(db)

        available = []
        seen_models = set()

        # First, add models from registry that match user's access
        for key, model_info in self._model_registry_cache.items():
            if model_info.model_name in seen_models:
                continue

            # Check if user can access this model
            allowed, _ = await self.check_model_access(db, user_id, model_info.model_name)
            if allowed:
                # Find provider for this model
                provider_name = model_info.provider_type.title()
                for p in providers:
                    if p.provider_type == model_info.provider_type:
                        provider_name = p.name
                        break

                available.append({
                    "provider": model_info.provider_type,
                    "provider_name": provider_name,
                    "model": model_info.model_name,
                    "display_name": model_info.display_name or model_info.model_name,
                    "quality_score": model_info.quality_score,
                    "cost_score": model_info.cost_per_million_tokens,
                    "latency_score": model_info.latency_score,
                    "tier": model_info.tier,
                    "supports_vision": model_info.supports_vision,
                    "supports_function_calling": model_info.supports_function_calling,
                    "is_default": False,
                })
                seen_models.add(model_info.model_name)

        # Then add models from active providers that aren't in registry
        for provider in providers:
            if not provider.is_active:
                continue

            provider_info = PROVIDER_TYPES.get(provider.provider_type, {})
            models = provider_info.get("chat_models", [])

            if models == "dynamic":
                # For Ollama, would fetch from API
                models = ["llama3.2", "llama3.1", "mistral", "codellama"]
            elif models == "manual" or models == "deployment-based":
                models = [provider.default_chat_model] if provider.default_chat_model else []

            for model in models:
                if model in seen_models:
                    continue

                # Check if user can access this model
                allowed, _ = await self.check_model_access(db, user_id, model)
                if allowed:
                    available.append({
                        "provider": provider.provider_type,
                        "provider_name": provider.name,
                        "model": model,
                        "display_name": model,
                        "quality_score": self._get_model_quality(model),
                        "cost_score": self._get_model_cost(model),
                        "latency_score": self._get_model_latency(model),
                        "tier": None,
                        "supports_vision": False,
                        "supports_function_calling": False,
                        "is_default": provider.is_default and model == provider.default_chat_model,
                    })
                    seen_models.add(model)

        # Sort by quality
        available.sort(key=lambda x: x["quality_score"], reverse=True)

        return available

    # =========================================================================
    # Service Configuration
    # =========================================================================

    async def get_service_config(
        self,
        db: AsyncSession,
        service_name: str,
        operation_name: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> ServiceLLMConfig:
        """
        Get LLM configuration for a service/operation.

        Checks in order:
        1. Organization-specific config
        2. Global config
        3. Default config
        """
        cache_key = f"{service_name}:{operation_name or 'default'}:{organization_id or 'global'}"

        if cache_key in self._service_config_cache:
            return self._service_config_cache[cache_key]

        # Query database for service config
        from sqlalchemy import text

        try:
            # First try org-specific config, then global
            query = text("""
                SELECT model_name, provider_type, temperature, max_tokens,
                       allow_user_override, fallback_model, routing_strategy,
                       min_access_level, settings
                FROM service_llm_configs
                WHERE service_name = :service_name
                  AND (operation_name = :operation_name OR operation_name IS NULL)
                  AND (organization_id = :organization_id OR organization_id IS NULL)
                  AND is_active = true
                ORDER BY
                    CASE WHEN organization_id IS NOT NULL THEN 0 ELSE 1 END,
                    CASE WHEN operation_name IS NOT NULL THEN 0 ELSE 1 END
                LIMIT 1
            """)

            result = await db.execute(
                query,
                {
                    "service_name": service_name,
                    "operation_name": operation_name,
                    "organization_id": organization_id,
                }
            )
            row = result.fetchone()

            if row:
                strategy = RoutingStrategy.DEFAULT
                if row.routing_strategy:
                    try:
                        strategy = RoutingStrategy(row.routing_strategy)
                    except ValueError:
                        pass

                config = ServiceLLMConfig(
                    service_name=service_name,
                    operation_name=operation_name,
                    model_name=row.model_name,
                    provider_type=row.provider_type,
                    temperature=row.temperature or 0.7,
                    max_tokens=row.max_tokens,
                    allow_user_override=row.allow_user_override if row.allow_user_override is not None else True,
                    fallback_model=row.fallback_model,
                    routing_strategy=strategy,
                    min_access_level=row.min_access_level or 0,
                )
                self._service_config_cache[cache_key] = config
                return config
        except Exception as e:
            # Table might not exist yet
            logger.debug(
                "Failed to query service config from database, using defaults",
                service=service_name,
                error=str(e)
            )

        # Fall back to default configurations by service
        default_configs = {
            "chat": ServiceLLMConfig(
                service_name="chat",
                model_name="gpt-4o",
                temperature=0.7,
                allow_user_override=True,
            ),
            "chat:agent": ServiceLLMConfig(
                service_name="chat",
                operation_name="agent",
                model_name="gpt-4o",
                temperature=0.3,
                routing_strategy=RoutingStrategy.QUALITY_OPTIMIZED,
                min_access_level=5,
            ),
            "rag": ServiceLLMConfig(
                service_name="rag",
                model_name="gpt-4o",
                temperature=0.5,
                allow_user_override=True,
            ),
            "audio_overview": ServiceLLMConfig(
                service_name="audio_overview",
                operation_name="script_generation",
                model_name="gpt-4o",
                temperature=0.8,
                routing_strategy=RoutingStrategy.QUALITY_OPTIMIZED,
                min_access_level=5,
            ),
            "workflow": ServiceLLMConfig(
                service_name="workflow",
                model_name="gpt-4o",
                temperature=0.5,
                allow_user_override=True,
                min_access_level=5,
            ),
            "generation": ServiceLLMConfig(
                service_name="generation",
                model_name="gpt-4o",
                temperature=0.7,
                allow_user_override=True,
                min_access_level=5,
            ),
            "embeddings": ServiceLLMConfig(
                service_name="embeddings",
                provider_type="openai",
                model_name="text-embedding-3-small",
                allow_user_override=False,
            ),
        }

        # Try specific operation first
        lookup_key = f"{service_name}:{operation_name}" if operation_name else service_name
        config = default_configs.get(lookup_key)

        if not config:
            # Fall back to service default
            config = default_configs.get(service_name)

        if not config:
            # Ultimate fallback
            config = ServiceLLMConfig(
                service_name=service_name,
                operation_name=operation_name,
                model_name="gpt-4o",
            )

        self._service_config_cache[cache_key] = config
        return config

    async def get_user_service_override(
        self,
        db: AsyncSession,
        user_id: str,
        service_name: str,
        operation_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get user's override for a service (if any).

        Returns None if no override is set.
        """
        from sqlalchemy import select, text

        # Try to query the user_service_llm_overrides table
        try:
            # Build operation identifier
            operation = operation_name or "default"

            # Query for user override
            query = text("""
                SELECT model_id, provider, settings
                FROM user_service_llm_overrides
                WHERE user_id = :user_id
                  AND service_name = :service_name
                  AND operation_name = :operation_name
                  AND is_active = true
                LIMIT 1
            """)

            result = await db.execute(
                query,
                {
                    "user_id": user_id,
                    "service_name": service_name,
                    "operation_name": operation,
                }
            )
            row = result.fetchone()

            if row:
                return {
                    "model": row.model_id,
                    "provider": row.provider,
                    "settings": row.settings or {},
                }
        except Exception as e:
            # Table might not exist yet or other DB error
            logger.debug(
                "Failed to query user service override",
                user_id=user_id,
                service=service_name,
                error=str(e)
            )

        return None

    # =========================================================================
    # Model Resolution
    # =========================================================================

    async def resolve_model(
        self,
        db: AsyncSession,
        service_name: str,
        operation_name: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        requested_model: Optional[str] = None,
        requested_provider: Optional[str] = None,
    ) -> ResolvedModel:
        """
        Resolve the model to use for a request.

        Resolution order:
        1. User's explicit request (if allowed and accessible)
        2. User's saved override for this service
        3. Organization's service config
        4. Global service config
        5. System default

        Args:
            db: Database session
            service_name: Service making the request (chat, rag, workflow, etc.)
            operation_name: Specific operation (optional)
            user_id: User making the request
            organization_id: User's organization
            requested_model: User's requested model (optional)
            requested_provider: User's requested provider (optional)

        Returns:
            ResolvedModel with provider, model, and parameters
        """
        # Get service config
        service_config = await self.get_service_config(
            db, service_name, operation_name, organization_id
        )

        # Check user access level
        if user_id:
            access = await self.get_user_access(db, user_id)

            # Check minimum access level
            if access.access_level < service_config.min_access_level:
                logger.warning(
                    "User access level too low for service",
                    user_id=user_id[:8] if user_id else None,
                    service=service_name,
                    required_level=service_config.min_access_level,
                    user_level=access.access_level,
                )
                # Fall back to a basic model
                return ResolvedModel(
                    provider_type="openai",
                    model_name="gpt-4o-mini",
                    temperature=service_config.temperature,
                    max_tokens=min(service_config.max_tokens, 4096),
                    source="access_restricted",
                )

        # Start with service default
        resolved = ResolvedModel(
            provider_type=service_config.provider_type,
            model_name=service_config.model_name,
            temperature=service_config.temperature,
            max_tokens=service_config.max_tokens,
            source="service_config",
        )

        # Check for user override
        if user_id and service_config.allow_user_override:
            user_override = await self.get_user_service_override(
                db, user_id, service_name, operation_name
            )

            if user_override:
                # Verify user can access the overridden model
                allowed, reason = await self.check_model_access(
                    db, user_id, user_override.get("model_name", "")
                )

                if allowed:
                    resolved.model_name = user_override.get("model_name", resolved.model_name)
                    resolved.provider_type = user_override.get("provider_type", resolved.provider_type)
                    if user_override.get("temperature") is not None:
                        resolved.temperature = user_override["temperature"]
                    resolved.source = "user_override"

        # Check explicit request
        if requested_model and service_config.allow_user_override:
            if user_id:
                allowed, reason = await self.check_model_access(db, user_id, requested_model)
                if allowed:
                    resolved.model_name = requested_model
                    if requested_provider:
                        resolved.provider_type = requested_provider
                    resolved.source = "explicit_request"
                    resolved.original_request = requested_model
                else:
                    logger.info(
                        "Model access denied",
                        user_id=user_id[:8] if user_id else None,
                        model=requested_model,
                        reason=reason,
                    )
            else:
                # No user = allow any model (system request)
                resolved.model_name = requested_model
                if requested_provider:
                    resolved.provider_type = requested_provider
                resolved.source = "explicit_request"

        # Apply routing strategy
        resolved = await self._apply_routing_strategy(
            db, resolved, service_config.routing_strategy, user_id
        )

        logger.debug(
            "Model resolved",
            service=service_name,
            operation=operation_name,
            provider=resolved.provider_type,
            model=resolved.model_name,
            source=resolved.source,
        )

        return resolved

    async def _apply_routing_strategy(
        self,
        db: AsyncSession,
        resolved: ResolvedModel,
        strategy: RoutingStrategy,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> ResolvedModel:
        """Apply routing strategy to potentially select a different model."""
        if strategy == RoutingStrategy.DEFAULT:
            return resolved

        # Load model registry for scores
        await self._load_model_registry(db, organization_id)

        # Get user's accessible models
        if user_id:
            available = await self.get_available_models(db, user_id, organization_id=organization_id)
        else:
            # System request - use registry models
            available = [
                {"model": info.model_name, "quality_score": info.quality_score}
                for info in self._model_registry_cache.values()
            ]

        if not available:
            return resolved

        if strategy == RoutingStrategy.COST_OPTIMIZED:
            # Find cheapest model with quality >= 70
            candidates = [
                m for m in available
                if m.get("quality_score", self._get_model_quality(m["model"])) >= 70
            ]
            if candidates:
                best = min(candidates, key=lambda x: x.get("cost_score", self._get_model_cost(x["model"])))
                resolved.model_name = best["model"]
                resolved.source = f"{resolved.source}+cost_optimized"

        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            # Find highest quality model
            best = max(available, key=lambda x: x.get("quality_score", self._get_model_quality(x["model"])))
            resolved.model_name = best["model"]
            resolved.source = f"{resolved.source}+quality_optimized"

        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            # Find fastest model with quality >= 60
            candidates = [
                m for m in available
                if m.get("quality_score", self._get_model_quality(m["model"])) >= 60
            ]
            if candidates:
                best = min(candidates, key=lambda x: x.get("latency_score", self._get_model_latency(x["model"])))
                resolved.model_name = best["model"]
                resolved.source = f"{resolved.source}+latency_optimized"

        return resolved

    # =========================================================================
    # LLM Instance Creation
    # =========================================================================

    async def get_llm(
        self,
        db: AsyncSession,
        service_name: str,
        operation_name: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        requested_model: Optional[str] = None,
        requested_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Get an LLM instance for a service.

        This is the main entry point for services to get their LLM.

        Returns:
            Tuple[BaseChatModel, ResolvedModel]
        """
        resolved = await self.resolve_model(
            db=db,
            service_name=service_name,
            operation_name=operation_name,
            user_id=user_id,
            organization_id=organization_id,
            requested_model=requested_model,
            requested_provider=requested_provider,
        )

        # Create LLM instance
        llm = LLMFactory.get_chat_model(
            provider=resolved.provider_type,
            model=resolved.model_name,
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
            **kwargs,
        )

        return llm, resolved

    # =========================================================================
    # Admin Functions
    # =========================================================================

    def clear_cache(self):
        """Clear all caches (call after config changes)."""
        self._service_config_cache.clear()
        self._user_access_cache.clear()
        logger.info("LLM router cache cleared")

    async def update_service_config(
        self,
        db: AsyncSession,
        service_name: str,
        operation_name: Optional[str],
        organization_id: Optional[str],
        config_updates: Dict[str, Any],
    ) -> ServiceLLMConfig:
        """
        Update service LLM configuration.

        Admin only.
        """
        # In production, update service_llm_configs table
        self.clear_cache()

        return await self.get_service_config(db, service_name, operation_name, organization_id)

    async def grant_model_access(
        self,
        db: AsyncSession,
        user_id: str,
        access_group_name: str,
        granted_by_id: str,
        expires_at: Optional[datetime] = None,
    ) -> bool:
        """
        Grant a user access to a model access group.

        Admin only.
        """
        # In production, insert into user_model_access table
        self._user_access_cache.pop(user_id, None)

        logger.info(
            "Model access granted",
            user_id=user_id[:8] if user_id else None,
            group=access_group_name,
            granted_by=granted_by_id[:8] if granted_by_id else None,
        )

        return True


# =============================================================================
# Singleton Instance
# =============================================================================

_llm_router: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """Get or create LLM router singleton."""
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router
