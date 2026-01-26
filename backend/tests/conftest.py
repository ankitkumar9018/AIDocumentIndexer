"""
AIDocumentIndexer - Pytest Configuration
=========================================

Shared fixtures and configuration for all tests.
"""

import os
import pytest
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock, patch

# Set test environment BEFORE any imports
os.environ["TESTING"] = "true"
os.environ["JWT_SECRET"] = "test-secret-key-for-testing-only"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["DATABASE_TYPE"] = "sqlite"

from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Import app components after setting env vars
from backend.api.routes import auth, documents, chat, upload
from backend.db.models import Base
from backend.db.database import get_async_session


# =============================================================================
# Async Event Loop
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

# Shared test engine and session factory (module-scoped for efficiency)
_test_engine = None
_test_session_factory = None


async def get_test_engine():
    """Get or create the test engine with tables."""
    global _test_engine
    if _test_engine is None:
        _test_engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with _test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    return _test_engine


async def get_test_session_factory():
    """Get or create the test session factory."""
    global _test_session_factory
    if _test_session_factory is None:
        engine = await get_test_engine()
        _test_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _test_session_factory


async def override_get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Override for FastAPI dependency injection."""
    session_factory = await get_test_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture(scope="function")
async def async_engine():
    """Create an in-memory async SQLite engine for testing."""
    engine = await get_test_engine()
    yield engine


@pytest.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    session_factory = await get_test_session_factory()
    async with session_factory() as session:
        yield session
        await session.rollback()


# =============================================================================
# FastAPI Test Client
# =============================================================================

@pytest.fixture
def test_app() -> FastAPI:
    """Create a test FastAPI application with dependency overrides."""
    app = FastAPI(title="AIDocumentIndexer Test")

    # Include routers
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(upload.router, prefix="/api/upload", tags=["upload"])

    # Override the database session dependency to use test database
    app.dependency_overrides[get_async_session] = override_get_async_session

    return app


@pytest.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def mock_admin_user() -> dict:
    """Mock admin user data."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "role": "admin",
        "access_tier": 100,
        "is_active": True,
        "created_at": datetime.now(),
    }


@pytest.fixture
def mock_regular_user() -> dict:
    """Mock regular user data."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "email": "user@example.com",
        "full_name": "Test User",
        "role": "user",
        "access_tier": 30,
        "is_active": True,
        "created_at": datetime.now(),
    }


@pytest.fixture
def admin_token(mock_admin_user: dict) -> str:
    """Generate a valid JWT token for admin user."""
    from backend.api.routes.auth import create_access_token
    return create_access_token(
        user_id=mock_admin_user["id"],
        email=mock_admin_user["email"],
        role=mock_admin_user["role"],
        access_tier=mock_admin_user["access_tier"],
    )


@pytest.fixture
def user_token(mock_regular_user: dict) -> str:
    """Generate a valid JWT token for regular user."""
    from backend.api.routes.auth import create_access_token
    return create_access_token(
        user_id=mock_regular_user["id"],
        email=mock_regular_user["email"],
        role=mock_regular_user["role"],
        access_tier=mock_regular_user["access_tier"],
    )


@pytest.fixture
def auth_headers(user_token: str) -> dict:
    """Authorization headers for regular user."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def admin_headers(admin_token: str) -> dict:
    """Authorization headers for admin user."""
    return {"Authorization": f"Bearer {admin_token}"}


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM service."""
    with patch("backend.services.llm.LLMFactory") as mock:
        mock_instance = MagicMock()
        mock_instance.get_chat_model.return_value = AsyncMock()
        mock_instance.get_embeddings.return_value = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_embeddings():
    """Mock embeddings that return fixed vectors."""
    mock = AsyncMock()
    mock.aembed_documents.return_value = [[0.1] * 1536]
    mock.aembed_query.return_value = [0.1] * 1536
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    # Patch where the function is used (in documents route), not where it's defined
    with patch("backend.api.routes.documents.get_vector_store") as mock:
        mock_store = AsyncMock()
        mock_store.search.return_value = []
        mock_store.add_documents.return_value = []
        mock.return_value = mock_store
        yield mock_store


# =============================================================================
# Document Fixtures
# =============================================================================

@pytest.fixture
def sample_document_data() -> dict:
    """Sample document creation data."""
    return {
        "name": "test_document.pdf",
        "file_type": "pdf",
        "collection": "test_collection",
        "access_tier": 30,
    }


@pytest.fixture
def sample_file_content() -> bytes:
    """Sample file content for upload tests."""
    return b"%PDF-1.4\n%Test PDF content for testing purposes"


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_document(
    id: str = "test-doc-1",
    name: str = "test.pdf",
    file_type: str = "pdf",
    status: str = "completed",
    access_tier: int = 30,
) -> dict:
    """Create a test document dictionary."""
    return {
        "id": id,
        "name": name,
        "file_type": file_type,
        "file_size": 1024,
        "file_hash": "abc123",
        "status": status,
        "access_tier": access_tier,
        "chunk_count": 10,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def create_test_chunk(
    id: str = "test-chunk-1",
    document_id: str = "test-doc-1",
    content: str = "Test chunk content",
    index: int = 0,
) -> dict:
    """Create a test chunk dictionary."""
    return {
        "id": id,
        "document_id": document_id,
        "content": content,
        "chunk_index": index,
        "page_number": 1,
        "metadata": {"section": "test"},
    }


# =============================================================================
# Phase 25: Integration Test Fixtures
# =============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client for integration tests."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.hset = AsyncMock(return_value=True)
    redis.hget = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.expire = AsyncMock(return_value=True)
    redis.incr = AsyncMock(return_value=1)
    redis.publish = AsyncMock(return_value=1)
    redis.lpush = AsyncMock(return_value=1)
    redis.lrange = AsyncMock(return_value=[])
    redis.pipeline = MagicMock(return_value=AsyncMock())
    return redis


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for integration tests."""
    service = AsyncMock()
    service.embed_texts = AsyncMock(return_value=[[0.1] * 1536 for _ in range(10)])
    service.embed_query = AsyncMock(return_value=[0.1] * 1536)
    service.embed_documents = AsyncMock(return_value=[[0.1] * 1536 for _ in range(10)])
    return service


@pytest.fixture
def sample_documents():
    """Generate sample documents for testing."""
    return [
        {
            "id": f"doc_{i}",
            "filename": f"document_{i}.pdf",
            "content": f"This is test document {i} with content about topic {i % 5}.",
            "metadata": {"source": "test", "page_count": 5},
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_chunks():
    """Generate sample chunks for testing."""
    return [
        {
            "id": f"chunk_{i}",
            "document_id": f"doc_{i // 10}",
            "content": f"This is chunk {i} with content about topic {i % 5}.",
            "metadata": {"page": i % 5, "position": i},
        }
        for i in range(1000)
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for benchmarking."""
    return [
        "What is the main topic of document 1?",
        "Summarize the key findings",
        "What are the recommendations?",
        "List all mentioned dates",
        "Who are the stakeholders?",
    ]


@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()
            return self

        def stop(self):
            self.end_time = time.perf_counter()
            return self

        @property
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return 0

    return Timer()


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        if "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
