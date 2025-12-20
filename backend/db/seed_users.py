"""
AIDocumentIndexer - Default User & Access Tier Seeding
=======================================================

Seeds the database with default access tiers and admin user on startup.
This ensures the system has proper access tiers and an admin user.
"""

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext

from backend.db.models import User, AccessTier
from backend.db.database import async_session_context

logger = structlog.get_logger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Default access tiers (matching the PostgreSQL migration)
DEFAULT_ACCESS_TIERS = [
    {"name": "intern", "level": 10, "description": "Basic access for interns", "color": "#94A3B8"},
    {"name": "staff", "level": 30, "description": "Standard staff access", "color": "#60A5FA"},
    {"name": "manager", "level": 50, "description": "Manager-level access", "color": "#34D399"},
    {"name": "executive", "level": 80, "description": "Executive-level access", "color": "#A78BFA"},
    {"name": "admin", "level": 100, "description": "Full administrative access", "color": "#F87171"},
]

# Default admin user
DEFAULT_ADMIN = {
    "email": "admin@example.com",
    "password": "admin123",  # Will be hashed with bcrypt
    "name": "Admin User",
    "is_active": True,
}


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


async def seed_access_tiers() -> None:
    """
    Seed the database with default access tiers if they don't exist.

    This function is idempotent - it only creates tiers that don't exist.
    """
    try:
        async with async_session_context() as db:
            # Check which tiers already exist
            existing_result = await db.execute(select(AccessTier.name))
            existing_names = {row[0] for row in existing_result.all()}

            # Create missing tiers
            created_count = 0
            for tier_data in DEFAULT_ACCESS_TIERS:
                if tier_data["name"] not in existing_names:
                    tier = AccessTier(
                        name=tier_data["name"],
                        level=tier_data["level"],
                        description=tier_data["description"],
                        color=tier_data["color"],
                    )
                    db.add(tier)
                    created_count += 1

            if created_count > 0:
                await db.commit()
                logger.info(
                    "Access tiers seeded successfully",
                    created=created_count,
                    total=len(DEFAULT_ACCESS_TIERS),
                )
            else:
                logger.debug("All access tiers already exist")

    except Exception as e:
        logger.error("Failed to seed access tiers", error=str(e))
        # Don't raise - this is a non-critical startup task


async def seed_admin_user() -> None:
    """
    Seed the database with the default admin user if it doesn't exist.

    This function is idempotent - it only creates the admin user if
    one doesn't already exist with the same email.

    Note: This will first seed access tiers if they don't exist.
    """
    try:
        # First, ensure access tiers exist
        await seed_access_tiers()

        async with async_session_context() as db:
            # Get the admin access tier
            tier_result = await db.execute(
                select(AccessTier).where(AccessTier.name == "admin")
            )
            admin_tier = tier_result.scalar_one_or_none()

            if not admin_tier:
                logger.error(
                    "Admin access tier not found after seeding",
                    hint="This should not happen - check database permissions",
                )
                return

            # Check if admin user already exists
            existing_result = await db.execute(
                select(User).where(User.email == DEFAULT_ADMIN["email"])
            )
            existing_user = existing_result.scalar_one_or_none()

            if existing_user:
                logger.debug(
                    "Admin user already exists",
                    email=DEFAULT_ADMIN["email"],
                )
                return

            # Create the admin user with bcrypt-hashed password
            admin_user = User(
                email=DEFAULT_ADMIN["email"],
                password_hash=hash_password(DEFAULT_ADMIN["password"]),
                name=DEFAULT_ADMIN["name"],
                is_active=DEFAULT_ADMIN["is_active"],
                access_tier_id=admin_tier.id,
                created_by_id=None,  # System-created
            )
            db.add(admin_user)
            await db.commit()

            logger.info(
                "Admin user created successfully",
                email=DEFAULT_ADMIN["email"],
                access_tier="admin",
            )

    except Exception as e:
        logger.error("Failed to seed admin user", error=str(e))
        # Don't raise - this is a non-critical startup task
        # The system can still function without the seeded user


async def get_user_by_email(email: str) -> User | None:
    """
    Get a user by their email address.

    Args:
        email: The user's email address

    Returns:
        The User object if found, None otherwise
    """
    async with async_session_context() as db:
        result = await db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
