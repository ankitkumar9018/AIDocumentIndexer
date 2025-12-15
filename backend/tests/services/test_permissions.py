"""
AIDocumentIndexer - Permission Service Tests
=============================================

Unit tests for the permission service.
"""

import pytest
from backend.services.permissions import (
    Permission,
    UserContext,
    require_tier,
    require_admin,
    create_user_context_from_token,
)


# =============================================================================
# UserContext Tests
# =============================================================================

class TestUserContext:
    """Tests for UserContext class."""

    def test_create_user_context(self):
        """Test creating a UserContext."""
        ctx = UserContext(
            user_id="user-123",
            email="test@example.com",
            role="user",
            access_tier_level=50,
            access_tier_name="Standard",
        )

        assert ctx.user_id == "user-123"
        assert ctx.email == "test@example.com"
        assert ctx.role == "user"
        assert ctx.access_tier_level == 50
        assert ctx.access_tier_name == "Standard"

    def test_can_access_tier_sufficient(self):
        """Test tier access when user has sufficient tier."""
        ctx = UserContext(
            user_id="user-123",
            email="test@example.com",
            role="user",
            access_tier_level=50,
            access_tier_name="Standard",
        )

        assert ctx.can_access_tier(30) is True
        assert ctx.can_access_tier(50) is True

    def test_can_access_tier_insufficient(self):
        """Test tier access when user has insufficient tier."""
        ctx = UserContext(
            user_id="user-123",
            email="test@example.com",
            role="user",
            access_tier_level=30,
            access_tier_name="Basic",
        )

        assert ctx.can_access_tier(50) is False
        assert ctx.can_access_tier(100) is False

    def test_is_admin_true(self):
        """Test is_admin returns True for admin role."""
        ctx = UserContext(
            user_id="admin-123",
            email="admin@example.com",
            role="admin",
            access_tier_level=100,
            access_tier_name="Admin",
        )

        assert ctx.is_admin() is True

    def test_is_admin_false(self):
        """Test is_admin returns False for non-admin role."""
        ctx = UserContext(
            user_id="user-123",
            email="user@example.com",
            role="user",
            access_tier_level=50,
            access_tier_name="Standard",
        )

        assert ctx.is_admin() is False


# =============================================================================
# Decorator Tests
# =============================================================================

class TestDecorators:
    """Tests for permission decorators."""

    @pytest.mark.asyncio
    async def test_require_tier_passes(self):
        """Test require_tier decorator allows sufficient tier."""
        @require_tier(30)
        async def protected_func(user: UserContext):
            return "success"

        ctx = UserContext(
            user_id="user-123",
            email="test@example.com",
            role="user",
            access_tier_level=50,
            access_tier_name="Standard",
        )

        result = await protected_func(user=ctx)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_require_tier_fails(self):
        """Test require_tier decorator blocks insufficient tier."""
        @require_tier(100)
        async def protected_func(user: UserContext):
            return "success"

        ctx = UserContext(
            user_id="user-123",
            email="test@example.com",
            role="user",
            access_tier_level=30,
            access_tier_name="Basic",
        )

        with pytest.raises(PermissionError) as exc_info:
            await protected_func(user=ctx)

        assert "Access denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_require_admin_passes(self):
        """Test require_admin decorator allows admin."""
        @require_admin
        async def admin_func(user: UserContext):
            return "admin action"

        ctx = UserContext(
            user_id="admin-123",
            email="admin@example.com",
            role="admin",
            access_tier_level=100,
            access_tier_name="Admin",
        )

        result = await admin_func(user=ctx)
        assert result == "admin action"

    @pytest.mark.asyncio
    async def test_require_admin_fails(self):
        """Test require_admin decorator blocks non-admin."""
        @require_admin
        async def admin_func(user: UserContext):
            return "admin action"

        ctx = UserContext(
            user_id="user-123",
            email="user@example.com",
            role="user",
            access_tier_level=50,
            access_tier_name="Standard",
        )

        with pytest.raises(PermissionError) as exc_info:
            await admin_func(user=ctx)

        assert "Admin access required" in str(exc_info.value)


# =============================================================================
# Token Parsing Tests
# =============================================================================

class TestTokenParsing:
    """Tests for token parsing functions."""

    def test_create_user_context_from_token(self):
        """Test creating UserContext from JWT payload."""
        payload = {
            "sub": "user-123",
            "email": "test@example.com",
            "role": "user",
            "access_tier": 50,
            "tier_name": "Standard",
        }

        ctx = create_user_context_from_token(payload)

        assert ctx.user_id == "user-123"
        assert ctx.email == "test@example.com"
        assert ctx.role == "user"
        assert ctx.access_tier_level == 50
        assert ctx.access_tier_name == "Standard"

    def test_create_user_context_from_token_defaults(self):
        """Test creating UserContext with missing fields uses defaults."""
        payload = {}

        ctx = create_user_context_from_token(payload)

        assert ctx.user_id == ""
        assert ctx.email == ""
        assert ctx.role == "user"
        assert ctx.access_tier_level == 10
        assert ctx.access_tier_name == "Basic"


# =============================================================================
# Permission Enum Tests
# =============================================================================

class TestPermissionEnum:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """Test Permission enum has expected values."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.ADMIN.value == "admin"

    def test_permission_comparison(self):
        """Test Permission enum comparison."""
        assert Permission.READ == "read"
        assert Permission.WRITE != Permission.READ
