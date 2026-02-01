"""
AIDocumentIndexer - Database Connection & Session Management
=============================================================

Supports multiple database backends:
- PostgreSQL (recommended for production with pgvector)
- SQLite (for development/testing)
- MySQL (for legacy integration)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

# Load environment variables before anything else
from dotenv import load_dotenv

# Try to load from project root .env file
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        # Check for LOCAL_MODE first - overrides other settings
        self.local_mode = os.getenv("LOCAL_MODE", "false").lower() == "true"

        if self.local_mode:
            # Force SQLite + local settings in local mode
            self.database_type = "sqlite"

            # Get project root (backend/../..)
            project_root = Path(__file__).parent.parent.parent

            # Default SQLite path in backend/data/
            default_db_path = project_root / "backend" / "data" / "aidocindexer.db"

            raw_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path}")

            # Handle relative SQLite paths - resolve them from project root
            if raw_url.startswith("sqlite:///") and not raw_url.startswith("sqlite:////"):
                # Extract the relative path (remove sqlite:///)
                rel_path = raw_url[10:]  # Remove "sqlite:///"
                abs_path = (project_root / rel_path).resolve()
                self.database_url = f"sqlite:///{abs_path}"
            else:
                self.database_url = raw_url

            # Ensure VECTOR_STORE_BACKEND is set for local mode
            if not os.getenv("VECTOR_STORE_BACKEND"):
                os.environ["VECTOR_STORE_BACKEND"] = "chroma"
            logger.info(f"Running in LOCAL_MODE - using SQLite at {self.database_url}")
        else:
            self.database_type = os.getenv("DATABASE_TYPE", "postgresql")
            self.database_url = os.getenv("DATABASE_URL", "")

        # Increased default pool size from 5 to 30 for better concurrency
        # Scale: 5 connections supports ~15 concurrent users
        #        30 connections supports ~100 concurrent users
        #        50+ connections for high-traffic production deployments
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "30"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.echo = os.getenv("APP_ENV") == "development"

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return "sqlite" in self.database_url.lower() or self.database_type == "sqlite"

    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode."""
        return self.local_mode or self.is_sqlite

    @property
    def async_url(self) -> str:
        """Convert sync URL to async URL."""
        url = self.database_url

        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        elif url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://")
        elif url.startswith("mysql://"):
            return url.replace("mysql://", "mysql+aiomysql://")

        return url

    @property
    def sync_url(self) -> str:
        """Ensure sync URL format."""
        url = self.database_url

        # Remove async drivers if present
        url = url.replace("+asyncpg", "")
        url = url.replace("+aiosqlite", "")
        url = url.replace("+aiomysql", "")

        # For PostgreSQL, ensure psycopg2 driver
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg2://")

        return url


# Global config
db_config = DatabaseConfig()


# =============================================================================
# Engine & Session Factory
# =============================================================================

# Async engine (for FastAPI)
async_engine: Optional[AsyncEngine] = None
AsyncSessionLocal: Optional[async_sessionmaker[AsyncSession]] = None

# Sync engine (for migrations, CLI tools)
sync_engine = None
SyncSessionLocal = None


def get_async_engine() -> AsyncEngine:
    """Get or create async database engine."""
    global async_engine

    if async_engine is None:
        # Build engine kwargs based on database type
        engine_kwargs = {
            "echo": db_config.echo,
        }

        # SQLite doesn't support pool_size/max_overflow with StaticPool
        # Also skip for in-memory databases
        if "sqlite" not in db_config.async_url:
            engine_kwargs["pool_size"] = db_config.pool_size
            engine_kwargs["max_overflow"] = db_config.max_overflow
            engine_kwargs["pool_pre_ping"] = True  # Check connection health

        async_engine = create_async_engine(
            db_config.async_url,
            **engine_kwargs,
        )
        logger.info(
            "Async database engine created",
            database_type=db_config.database_type,
            url_type="sqlite" if "sqlite" in db_config.async_url else "other",
            database_url=db_config.async_url,
        )

    return async_engine


def get_sync_engine():
    """Get or create sync database engine."""
    global sync_engine

    if sync_engine is None:
        # Build engine kwargs based on database type
        engine_kwargs = {
            "echo": db_config.echo,
        }

        # SQLite doesn't support pool_size/max_overflow
        if "sqlite" not in db_config.sync_url:
            engine_kwargs["pool_size"] = db_config.pool_size
            engine_kwargs["max_overflow"] = db_config.max_overflow
            engine_kwargs["pool_pre_ping"] = True

        sync_engine = create_engine(
            db_config.sync_url,
            **engine_kwargs,
        )
        logger.info(
            "Sync database engine created",
            database_type=db_config.database_type,
            url_type="sqlite" if "sqlite" in db_config.sync_url else "other",
        )

    return sync_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get async session factory."""
    global AsyncSessionLocal

    if AsyncSessionLocal is None:
        AsyncSessionLocal = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    return AsyncSessionLocal


def get_sync_session_factory() -> sessionmaker:
    """Get sync session factory."""
    global SyncSessionLocal

    if SyncSessionLocal is None:
        SyncSessionLocal = sessionmaker(
            bind=get_sync_engine(),
            autocommit=False,
            autoflush=False,
        )

    return SyncSessionLocal


# =============================================================================
# Session Dependencies
# =============================================================================

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for async database session.

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_session)):
            ...
    """
    session_factory = get_async_session_factory()

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def async_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database session.

    Usage:
        async with async_session_context() as session:
            ...
    """
    session_factory = get_async_session_factory()

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session() -> Session:
    """Get sync database session."""
    session_factory = get_sync_session_factory()
    return session_factory()


# =============================================================================
# Database Initialization
# =============================================================================

async def init_db() -> None:
    """
    Initialize database connection and create tables if needed.

    Called during application startup.
    """
    engine = get_async_engine()

    # Test connection
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    logger.info("Database connection verified")

    # Create tables (in development mode only)
    if os.getenv("APP_ENV") == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")


async def close_db() -> None:
    """
    Close database connections.

    Called during application shutdown.
    """
    global async_engine, sync_engine

    if async_engine is not None:
        await async_engine.dispose()
        async_engine = None
        logger.info("Async database engine disposed")

    if sync_engine is not None:
        sync_engine.dispose()
        sync_engine = None
        logger.info("Sync database engine disposed")


def create_all_tables() -> None:
    """
    Create all database tables (sync version for CLI/migrations).
    """
    engine = get_sync_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("All database tables created")


def drop_all_tables() -> None:
    """
    Drop all database tables (use with caution!).
    """
    engine = get_sync_engine()
    Base.metadata.drop_all(bind=engine)
    logger.info("All database tables dropped")


# =============================================================================
# RLS Helper (PostgreSQL only)
# =============================================================================

async def set_user_context(session: AsyncSession, user_id: str) -> None:
    """
    Set the current user ID for Row-Level Security policies.

    This must be called before any query that uses RLS.

    Args:
        session: Database session
        user_id: UUID of the current user
    """
    if db_config.database_type == "postgresql":
        await session.execute(
            text(f"SET app.current_user_id = '{user_id}'")
        )


async def clear_user_context(session: AsyncSession) -> None:
    """
    Clear the user context after queries.
    """
    if db_config.database_type == "postgresql":
        await session.execute(
            text("RESET app.current_user_id")
        )
