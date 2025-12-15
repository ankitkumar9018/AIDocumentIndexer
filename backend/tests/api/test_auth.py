"""
AIDocumentIndexer - Authentication API Tests
=============================================

Integration tests for authentication endpoints.
"""

import pytest
from httpx import AsyncClient


# =============================================================================
# Login Tests
# =============================================================================

class TestLogin:
    """Tests for the login endpoint."""

    @pytest.mark.asyncio
    async def test_login_success(self, async_client: AsyncClient):
        """Test successful login with valid credentials."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.com",
                "password": "admin123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["email"] == "admin@example.com"
        assert data["user"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_login_invalid_email(self, async_client: AsyncClient):
        """Test login with non-existent email."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert "Invalid email or password" in data["detail"]

    @pytest.mark.asyncio
    async def test_login_invalid_password(self, async_client: AsyncClient):
        """Test login with wrong password."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.com",
                "password": "wrongpassword",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert "Invalid email or password" in data["detail"]

    @pytest.mark.asyncio
    async def test_login_missing_fields(self, async_client: AsyncClient):
        """Test login with missing required fields."""
        response = await async_client.post(
            "/api/auth/login",
            json={"email": "admin@example.com"},
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# Registration Tests
# =============================================================================

class TestRegistration:
    """Tests for the registration endpoint."""

    @pytest.mark.asyncio
    async def test_register_success(self, async_client: AsyncClient):
        """Test successful user registration."""
        response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepass123",
                "full_name": "New User",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == "newuser@example.com"
        assert data["user"]["role"] == "user"
        assert data["user"]["access_tier"] == 10

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, async_client: AsyncClient):
        """Test registration with existing email."""
        response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "admin@example.com",
                "password": "password123",
                "full_name": "Duplicate User",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "already registered" in data["detail"]

    @pytest.mark.asyncio
    async def test_register_short_password(self, async_client: AsyncClient):
        """Test registration with too short password."""
        response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "shortpass@example.com",
                "password": "short",
                "full_name": "Short Pass User",
            },
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, async_client: AsyncClient):
        """Test registration with invalid email format."""
        response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "not-an-email",
                "password": "password123",
                "full_name": "Invalid Email User",
            },
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# Profile Tests
# =============================================================================

class TestProfile:
    """Tests for the user profile endpoint."""

    @pytest.mark.asyncio
    async def test_get_profile_authenticated(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test getting profile for authenticated user."""
        response = await async_client.get(
            "/api/auth/me",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "role" in data
        assert "access_tier" in data

    @pytest.mark.asyncio
    async def test_get_profile_unauthenticated(self, async_client: AsyncClient):
        """Test getting profile without authentication."""
        response = await async_client.get("/api/auth/me")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_profile_invalid_token(self, async_client: AsyncClient):
        """Test getting profile with invalid token."""
        response = await async_client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 401


# =============================================================================
# Token Verification Tests
# =============================================================================

class TestTokenVerification:
    """Tests for token verification endpoint."""

    @pytest.mark.asyncio
    async def test_verify_valid_token(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test verifying a valid token."""
        response = await async_client.get(
            "/api/auth/verify",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "user_id" in data
        assert "email" in data

    @pytest.mark.asyncio
    async def test_verify_expired_token(self, async_client: AsyncClient):
        """Test verifying an expired token."""
        import jwt
        from datetime import datetime, timedelta

        # Create an expired token
        expired_payload = {
            "sub": "user-123",
            "email": "test@example.com",
            "role": "user",
            "access_tier": 30,
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2),
        }
        expired_token = jwt.encode(
            expired_payload,
            "test-secret-key-for-testing-only",
            algorithm="HS256",
        )

        response = await async_client.get(
            "/api/auth/verify",
            headers={"Authorization": f"Bearer {expired_token}"},
        )

        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()


# =============================================================================
# Password Change Tests
# =============================================================================

class TestPasswordChange:
    """Tests for password change endpoint."""

    @pytest.mark.asyncio
    async def test_change_password_success(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test successful password change."""
        # First login to get the actual user
        login_response = await async_client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "user123"},
        )
        token = login_response.json()["access_token"]

        response = await async_client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "user123",
                "new_password": "newpassword123",
            },
        )

        assert response.status_code == 200
        assert "successfully" in response.json()["message"].lower()

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test password change with wrong current password."""
        login_response = await async_client.post(
            "/api/auth/login",
            json={"email": "admin@example.com", "password": "admin123"},
        )
        token = login_response.json()["access_token"]

        response = await async_client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "wrongpassword",
                "new_password": "newpassword123",
            },
        )

        assert response.status_code == 400
        assert "incorrect" in response.json()["detail"].lower()


# =============================================================================
# Admin User Management Tests
# =============================================================================

class TestAdminUserManagement:
    """Tests for admin user management endpoints."""

    @pytest.mark.asyncio
    async def test_list_users_as_admin(
        self, async_client: AsyncClient, admin_headers: dict
    ):
        """Test listing users as admin."""
        response = await async_client.get(
            "/api/auth/users",
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert isinstance(data["users"], list)

    @pytest.mark.asyncio
    async def test_list_users_as_regular_user(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test listing users as regular user (should fail)."""
        response = await async_client.get(
            "/api/auth/users",
            headers=auth_headers,
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_update_user_as_admin(
        self, async_client: AsyncClient, admin_headers: dict, mock_regular_user: dict
    ):
        """Test updating user role as admin."""
        response = await async_client.patch(
            f"/api/auth/users/{mock_regular_user['id']}",
            headers=admin_headers,
            params={"access_tier": 50},
        )

        assert response.status_code == 200
        assert "successfully" in response.json()["message"].lower()


# =============================================================================
# Logout Tests
# =============================================================================

class TestLogout:
    """Tests for the logout endpoint."""

    @pytest.mark.asyncio
    async def test_logout_success(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test successful logout."""
        response = await async_client.post(
            "/api/auth/logout",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert "logged out" in response.json()["message"].lower()

    @pytest.mark.asyncio
    async def test_logout_unauthenticated(self, async_client: AsyncClient):
        """Test logout without authentication."""
        response = await async_client.post("/api/auth/logout")

        assert response.status_code == 401
