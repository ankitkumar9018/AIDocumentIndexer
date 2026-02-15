"""
AIDocumentIndexer - Authentication API Routes
==============================================

Endpoints for user authentication and authorization.
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import update, select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
import jwt
import os
from passlib.context import CryptContext

from backend.db.database import get_async_session
from backend.db.models import User, AccessTier
from backend.core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()

# JWT Configuration
_jwt_secret = os.getenv("JWT_SECRET", os.getenv("SECRET_KEY", ""))
if not _jwt_secret or _jwt_secret in ("your-secret-key-change-in-production", "change-me-in-production"):
    _env = os.getenv("ENVIRONMENT", "development")
    if _env in ("production", "staging"):
        raise RuntimeError(
            "CRITICAL: JWT_SECRET or SECRET_KEY must be set to a secure value in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )
    _jwt_secret = "dev-only-insecure-key-do-not-use-in-production"
    logger.warning("Using insecure default JWT secret — acceptable for development only")

JWT_SECRET = _jwt_secret
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Security
security = HTTPBearer(auto_error=False)

# Password hashing with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pre-computed dummy hash for timing-safe comparison when user doesn't exist.
# Prevents user enumeration via response time differences (bcrypt is slow).
_DUMMY_PASSWORD_HASH = "$2b$12$LJ3m4ys3Lz.erfXkParfFux0UiUCUvZNpHBCwBR4ByDXaRTCZbxaS"


# =============================================================================
# Rate Limiting for Auth Endpoints
# =============================================================================

class AuthRateLimiter:
    """Simple in-memory rate limiter for authentication endpoints."""

    def __init__(self):
        self._attempts: Dict[str, list] = defaultdict(list)

    def _clean_old(self, key: str, window_seconds: int):
        cutoff = time.time() - window_seconds
        self._attempts[key] = [t for t in self._attempts[key] if t > cutoff]

    def check_rate_limit(self, ip: str, action: str, max_attempts: int, window_seconds: int):
        key = f"{action}:{ip}"
        self._clean_old(key, window_seconds)
        if len(self._attempts[key]) >= max_attempts:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many {action} attempts. Please try again later.",
                headers={"Retry-After": str(window_seconds)},
            )
        self._attempts[key].append(time.time())

_auth_limiter = AuthRateLimiter()

# Need Dict import for type annotation
from typing import Dict


# =============================================================================
# Pydantic Models
# =============================================================================

class LoginRequest(BaseModel):
    """Login request with email and password."""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class UserResponse(BaseModel):
    """User information response."""
    id: str
    email: str
    full_name: str
    role: str
    access_tier: int
    is_active: bool
    created_at: datetime


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class OAuthRequest(BaseModel):
    """OAuth token exchange request."""
    id_token: Optional[str] = None
    access_token: str


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


# =============================================================================
# Mock User Store (DEBUG mode only)
# =============================================================================

# In-memory user store for development/testing only.
# Only available when DEBUG=True in settings.
# In production (DEBUG=False), all users must exist in the database.
_users: dict = {}

if settings.DEBUG:
    logger.warning("DEBUG mode: Mock users enabled. Do not use in production!")
    _users = {
        "admin@example.com": {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "admin@example.com",
            "full_name": "Admin User",
            "password_hash": pwd_context.hash("admin123"),
            "role": "admin",
            "access_tier": 100,
            "is_active": True,
            "created_at": datetime.now(),
        },
        "user@example.com": {
            "id": "550e8400-e29b-41d4-a716-446655440001",
            "email": "user@example.com",
            "full_name": "Test User",
            "password_hash": pwd_context.hash("user123"),
            "role": "user",
            "access_tier": 30,
            "is_active": True,
            "created_at": datetime.now(),
        },
    }


# =============================================================================
# Helper Functions
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its bcrypt hash."""
    return pwd_context.verify(password, password_hash)


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    access_tier: int,
    organization_id: Optional[str] = None,
    is_superadmin: bool = False,
) -> str:
    """Create a JWT access token with organization context."""
    expires = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "access_tier": access_tier,
        "exp": expires,
        "iat": datetime.utcnow(),
    }
    # Include organization_id for multi-tenant isolation
    if organization_id:
        payload["organization_id"] = organization_id
    # Include superadmin flag for organization switching capability
    if is_superadmin:
        payload["is_superadmin"] = True
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str, allow_expired: bool = False) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: JWT token string
        allow_expired: If True, allow expired tokens (for refresh endpoint)
    """
    try:
        options = {}
        if allow_expired:
            options["verify_exp"] = False
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options=options)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Get current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    email = payload.get("email")

    if not email or email not in _users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    user = _users[email]
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def get_current_user_for_refresh(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Get current user from JWT token, allowing expired tokens.
    Used specifically for the token refresh endpoint.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Allow expired tokens for refresh
    payload = decode_token(credentials.credentials, allow_expired=True)
    email = payload.get("email")

    if not email or email not in _users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    user = _users[email]
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_role(required_roles: list[str]):
    """Dependency to require specific roles."""
    async def role_checker(user: dict = Depends(get_current_user)):
        if user["role"] not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user
    return role_checker


def require_tier(min_tier: int):
    """Dependency to require minimum access tier."""
    async def tier_checker(user: dict = Depends(get_current_user)):
        if user["access_tier"] < min_tier:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires access tier {min_tier} or higher",
            )
        return user
    return tier_checker


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, req: Request, db: AsyncSession = Depends(get_async_session)):
    """
    Authenticate user with email and password.

    Returns JWT token on success.
    Rate limited: 10 attempts per 60 seconds per IP.
    """
    client_ip = req.client.host if req.client else "unknown"
    _auth_limiter.check_rate_limit(client_ip, "login", max_attempts=10, window_seconds=60)

    logger.info("Login attempt", email=request.email)

    # Find user in database
    query = select(User).where(User.email == request.email)
    result = await db.execute(query)
    db_user = result.scalar_one_or_none()

    # Fallback to mock users only in DEBUG mode
    if not db_user:
        if not settings.DEBUG:
            # Always run bcrypt to prevent timing-based user enumeration
            verify_password(request.password, _DUMMY_PASSWORD_HASH)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        user = _users.get(request.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Verify password against mock user
        if not verify_password(request.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled",
            )

        # Create token with mock user data
        access_token = create_access_token(
            user_id=user["id"],
            email=user["email"],
            role=user["role"],
            access_tier=user["access_tier"],
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=JWT_EXPIRATION_HOURS * 3600,
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                full_name=user["full_name"],
                role=user["role"],
                access_tier=user["access_tier"],
                is_active=user["is_active"],
                created_at=user["created_at"],
            ),
        )

    # Verify password against database user
    if not db_user.password_hash or not verify_password(request.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user is active
    if not db_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Get access tier level
    access_tier_level = 30  # default
    if db_user.access_tier_id:
        tier_query = select(AccessTier).where(AccessTier.id == db_user.access_tier_id)
        tier_result = await db.execute(tier_query)
        tier = tier_result.scalar_one_or_none()
        if tier:
            access_tier_level = tier.level

    # Determine role based on superadmin status or access tier
    role = "admin" if db_user.is_superadmin else "user"

    # Get organization_id from user's current_organization_id
    organization_id = str(db_user.current_organization_id) if db_user.current_organization_id else None

    # Create token with actual database user ID and organization context
    access_token = create_access_token(
        user_id=str(db_user.id),
        email=db_user.email,
        role=role,
        access_tier=access_tier_level,
        organization_id=organization_id,
        is_superadmin=db_user.is_superadmin,
    )

    # Update last_login_at in database
    try:
        db_user.last_login_at = datetime.utcnow()
        await db.commit()
        logger.info("Updated last_login_at", email=request.email)
    except Exception as e:
        logger.warning("Failed to update last_login_at", email=request.email, error=str(e))
        # Don't fail the login if we can't update the timestamp

    logger.info("Login successful", email=request.email, role=role)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=UserResponse(
            id=str(db_user.id),
            email=db_user.email,
            full_name=db_user.name or "User",
            role=role,
            access_tier=access_tier_level,
            is_active=db_user.is_active,
            created_at=db_user.created_at or datetime.utcnow(),
        ),
    )


@router.post("/register", response_model=TokenResponse)
async def register(
    request: RegisterRequest,
    req: Request,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Register a new user.

    In production (DEBUG=False): Creates user in database.
    In development (DEBUG=True): Can also use in-memory mock store.

    Rate limited: 5 registrations per 3600 seconds per IP.
    """
    client_ip = req.client.host if req.client else "unknown"
    _auth_limiter.check_rate_limit(client_ip, "register", max_attempts=5, window_seconds=3600)

    logger.info("Registration attempt", email=request.email)

    # Check if email already exists in database
    existing_query = select(User).where(User.email == request.email)
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Also check mock users in DEBUG mode
    if settings.DEBUG and request.email in _users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    user_id = uuid4()
    now = datetime.utcnow()

    # Create user in database
    new_user = User(
        id=user_id,
        email=request.email,
        full_name=request.full_name,
        password_hash=hash_password(request.password),
        role="user",
        is_active=True,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Create token
    access_token = create_access_token(
        user_id=str(user_id),
        email=request.email,
        role="user",
        access_tier=10,  # Default tier
    )

    logger.info("Registration successful", email=request.email)

    return TokenResponse(
        access_token=access_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=UserResponse(
            id=str(user_id),
            email=request.email,
            full_name=request.full_name,
            role="user",
            access_tier=10,
            is_active=True,
            created_at=now,
        ),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user)):
    """
    Get current authenticated user's profile.
    """
    return UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        access_tier=user["access_tier"],
        is_active=user["is_active"],
        created_at=user["created_at"],
    )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Change current user's password.
    """
    # Verify current password
    if not verify_password(request.current_password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    new_hash = hash_password(request.new_password)

    # Update in database
    user_id = UUID(user["id"]) if isinstance(user["id"], str) else user["id"]
    stmt = update(User).where(User.id == user_id).values(password_hash=new_hash)
    await db.execute(stmt)
    await db.commit()

    # Also update mock users in DEBUG mode if applicable
    if settings.DEBUG and user["email"] in _users:
        _users[user["email"]]["password_hash"] = new_hash

    logger.info("Password changed", email=user["email"])

    return {"message": "Password changed successfully"}


@router.post("/oauth/google", response_model=TokenResponse)
async def oauth_google(
    request: OAuthRequest,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Exchange Google OAuth token for application token.

    This endpoint validates the Google token and creates/updates
    the user in the local database.

    Note: Full OAuth implementation requires:
    - Google Cloud Console project with OAuth credentials
    - GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables
    - Proper token validation using Google's public keys
    """
    logger.info("Google OAuth token exchange")

    if not settings.DEBUG:
        # Production: Implement proper Google token validation
        # This requires google-auth library and proper credentials
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth requires configuration. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )

    # DEBUG mode: Create a test OAuth user for development
    logger.warning("DEBUG mode: Using mock OAuth user - not for production!")

    email = "oauth_user@example.com"

    # Check if user exists in database
    existing_query = select(User).where(User.email == email)
    existing_result = await db.execute(existing_query)
    db_user = existing_result.scalar_one_or_none()

    if db_user:
        user_id = str(db_user.id)
        full_name = db_user.full_name
        role = db_user.role
        access_tier = 30
    else:
        # Create mock user in database for DEBUG mode
        user_id = str(uuid4())
        full_name = "OAuth User"
        role = "user"
        access_tier = 30

        new_user = User(
            id=UUID(user_id),
            email=email,
            full_name=full_name,
            password_hash="",
            role=role,
            is_active=True,
        )
        db.add(new_user)
        await db.commit()

    access_token = create_access_token(
        user_id=user_id,
        email=email,
        role=role,
        access_tier=access_tier,
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=UserResponse(
            id=user_id,
            email=email,
            full_name=full_name,
            role=role,
            access_tier=access_tier,
            is_active=db_user.is_active if db_user else True,
            created_at=str(db_user.created_at) if db_user else str(datetime.utcnow()),
        ),
    )


@router.post("/refresh")
async def refresh_token(user: dict = Depends(get_current_user_for_refresh)):
    """
    Refresh the access token.

    This endpoint accepts expired tokens and returns a new valid token.
    The signature must still be valid (not tampered with).
    """
    access_token = create_access_token(
        user_id=user.get("sub") or user.get("id"),
        email=user["email"],
        role=user["role"],
        access_tier=user["access_tier"],
        organization_id=user.get("organization_id"),
        is_superadmin=user.get("is_superadmin", False),
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
    }


@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    """
    Logout current user.

    Note: With JWT, logout is typically handled client-side by
    removing the token. This endpoint can be used for audit logging
    or token blacklisting if implemented.
    """
    logger.info("User logout", email=user["email"])

    # In production, you might want to:
    # 1. Add token to a blacklist
    # 2. Clear refresh tokens
    # 3. Log the logout event

    return {"message": "Logged out successfully"}


@router.get("/verify")
async def verify_token(user: dict = Depends(get_current_user)):
    """
    Verify that the current token is valid.

    Useful for checking authentication status.
    """
    return {
        "valid": True,
        "user_id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "access_tier": user["access_tier"],
    }


# =============================================================================
# Admin Endpoints
# =============================================================================

@router.get("/users", dependencies=[Depends(require_role(["admin"]))])
async def list_users(db: AsyncSession = Depends(get_async_session)):
    """
    List all users from database (admin only).
    """
    query = select(User).order_by(User.created_at.desc())
    result = await db.execute(query)
    db_users = result.scalars().all()

    users = [
        UserResponse(
            id=str(u.id),
            email=u.email,
            full_name=u.full_name or "",
            role=u.role or "user",
            access_tier=30,  # Default, would need to join with access_tier table
            is_active=u.is_active,
            created_at=u.created_at or datetime.utcnow(),
        )
        for u in db_users
    ]

    # Include mock users only in DEBUG mode
    if settings.DEBUG:
        for email, u in _users.items():
            # Don't duplicate if already in database
            if not any(user.email == email for user in users):
                users.append(
                    UserResponse(
                        id=u["id"],
                        email=u["email"],
                        full_name=u["full_name"],
                        role=u["role"],
                        access_tier=u["access_tier"],
                        is_active=u["is_active"],
                        created_at=u["created_at"],
                    )
                )

    return {
        "users": users,
        "total": len(users),
    }


@router.patch("/users/{user_id}", dependencies=[Depends(require_role(["admin"]))])
async def update_user(
    user_id: str,
    role: Optional[str] = None,
    access_tier: Optional[int] = None,
    is_active: Optional[bool] = None,
    admin: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Update a user's role or access tier (admin only).
    """
    # Try to find user in database first
    try:
        target_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format",
        )

    query = select(User).where(User.id == target_uuid)
    result = await db.execute(query)
    db_user = result.scalar_one_or_none()

    if db_user:
        # Prevent admin from demoting themselves
        if str(db_user.id) == admin["id"] and role and role != "admin":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change your own admin role",
            )

        # Build update values
        update_values = {}
        if role is not None:
            update_values["role"] = role
        if is_active is not None:
            update_values["is_active"] = is_active
        # Note: access_tier requires updating via access_tier_id relationship

        if access_tier is not None:
            # Admin can only assign tiers up to their own level
            if access_tier > admin["access_tier"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot assign tier higher than your own",
                )
            # For now, just log - proper implementation needs access_tier lookup

        if update_values:
            stmt = update(User).where(User.id == target_uuid).values(**update_values)
            await db.execute(stmt)
            await db.commit()

        logger.info(
            "User updated by admin",
            target_user=user_id,
            admin=admin["email"],
            changes={"role": role, "access_tier": access_tier, "is_active": is_active},
        )

        return {"message": "User updated successfully"}

    # Fallback to mock users in DEBUG mode
    if settings.DEBUG:
        target_user = None
        target_email = None
        for email, user in _users.items():
            if user["id"] == user_id:
                target_user = user
                target_email = email
                break

        if target_user:
            if target_user["id"] == admin["id"] and role and role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot change your own admin role",
                )

            if role is not None:
                _users[target_email]["role"] = role
            if access_tier is not None:
                if access_tier > admin["access_tier"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Cannot assign tier higher than your own",
                    )
                _users[target_email]["access_tier"] = access_tier
            if is_active is not None:
                _users[target_email]["is_active"] = is_active

            logger.info(
                "Mock user updated by admin",
                target_user=user_id,
                admin=admin["email"],
                changes={"role": role, "access_tier": access_tier, "is_active": is_active},
            )

            return {"message": "User updated successfully"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found",
    )
