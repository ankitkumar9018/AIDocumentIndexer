"""
AIDocumentIndexer - Authentication API Routes
==============================================

Endpoints for user authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
import structlog
import jwt
import hashlib
import os

logger = structlog.get_logger(__name__)

router = APIRouter()

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Security
security = HTTPBearer(auto_error=False)


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
# Mock User Store (Replace with database in production)
# =============================================================================

# In-memory user store for development
_users: dict = {
    "admin@example.com": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "access_tier": 100,
        "is_active": True,
        "created_at": datetime.now(),
    },
    "user@example.com": {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "email": "user@example.com",
        "full_name": "Test User",
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
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
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash


def create_access_token(user_id: str, email: str, role: str, access_tier: int) -> str:
    """Create a JWT access token."""
    expires = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "access_tier": access_tier,
        "exp": expires,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
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
async def login(request: LoginRequest):
    """
    Authenticate user with email and password.

    Returns JWT token on success.
    """
    logger.info("Login attempt", email=request.email)

    # Find user
    user = _users.get(request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Verify password
    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user is active
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Create token
    access_token = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        access_tier=user["access_tier"],
    )

    logger.info("Login successful", email=request.email, role=user["role"])

    return TokenResponse(
        access_token=access_token,
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


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """
    Register a new user.

    Note: In production, this might require admin approval or invitation.
    """
    logger.info("Registration attempt", email=request.email)

    # Check if email already exists
    if request.email in _users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    user_id = str(uuid4())
    now = datetime.now()

    _users[request.email] = {
        "id": user_id,
        "email": request.email,
        "full_name": request.full_name,
        "password_hash": hash_password(request.password),
        "role": "user",  # Default role
        "access_tier": 10,  # Default tier (lowest)
        "is_active": True,
        "created_at": now,
    }

    # Create token
    access_token = create_access_token(
        user_id=user_id,
        email=request.email,
        role="user",
        access_tier=10,
    )

    logger.info("Registration successful", email=request.email)

    return TokenResponse(
        access_token=access_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=UserResponse(
            id=user_id,
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

    # Update password
    _users[user["email"]]["password_hash"] = hash_password(request.new_password)

    logger.info("Password changed", email=user["email"])

    return {"message": "Password changed successfully"}


@router.post("/oauth/google", response_model=TokenResponse)
async def oauth_google(request: OAuthRequest):
    """
    Exchange Google OAuth token for application token.

    This endpoint validates the Google token and creates/updates
    the user in the local database.
    """
    logger.info("Google OAuth token exchange")

    # Production implementation would validate the Google token:
    # 1. Verify the token signature using Google's public keys
    # 2. Check token expiration and audience (client ID)
    # 3. Extract user info from the token payload
    # For now, using mock user for development/testing

    # Mock user for development
    email = "oauth_user@example.com"
    if email not in _users:
        user_id = str(uuid4())
        _users[email] = {
            "id": user_id,
            "email": email,
            "full_name": "OAuth User",
            "password_hash": "",
            "role": "user",
            "access_tier": 30,
            "is_active": True,
            "created_at": datetime.now(),
        }

    user = _users[email]

    access_token = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        access_tier=user["access_tier"],
    )

    return TokenResponse(
        access_token=access_token,
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


@router.post("/refresh")
async def refresh_token(user: dict = Depends(get_current_user)):
    """
    Refresh the access token.

    Returns a new token with extended expiration.
    """
    access_token = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        access_tier=user["access_tier"],
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
async def list_users():
    """
    List all users (admin only).
    """
    return {
        "users": [
            UserResponse(
                id=u["id"],
                email=u["email"],
                full_name=u["full_name"],
                role=u["role"],
                access_tier=u["access_tier"],
                is_active=u["is_active"],
                created_at=u["created_at"],
            )
            for u in _users.values()
        ],
        "total": len(_users),
    }


@router.patch("/users/{user_id}", dependencies=[Depends(require_role(["admin"]))])
async def update_user(
    user_id: str,
    role: Optional[str] = None,
    access_tier: Optional[int] = None,
    is_active: Optional[bool] = None,
    admin: dict = Depends(get_current_user),
):
    """
    Update a user's role or access tier (admin only).
    """
    # Find user by ID
    target_user = None
    target_email = None
    for email, user in _users.items():
        if user["id"] == user_id:
            target_user = user
            target_email = email
            break

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Prevent admin from demoting themselves
    if target_user["id"] == admin["id"] and role and role != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own admin role",
        )

    # Update fields
    if role is not None:
        _users[target_email]["role"] = role
    if access_tier is not None:
        # Admin can only assign tiers up to their own level
        if access_tier > admin["access_tier"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot assign tier higher than your own",
            )
        _users[target_email]["access_tier"] = access_tier
    if is_active is not None:
        _users[target_email]["is_active"] = is_active

    logger.info(
        "User updated by admin",
        target_user=user_id,
        admin=admin["email"],
        changes={"role": role, "access_tier": access_tier, "is_active": is_active},
    )

    return {"message": "User updated successfully"}
