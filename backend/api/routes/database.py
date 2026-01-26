"""
AIDocumentIndexer - Database Connector API Routes
==================================================

API endpoints for database connections and natural language querying.

Features:
- Create and manage database connections
- Test database connectivity
- Query databases using natural language
- View query history and feedback
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from backend.api.deps import (
    get_async_session as get_db_session,
    get_current_organization_id,
)
from backend.api.middleware.auth import AuthenticatedUser
from backend.db.database import async_session_context
from backend.db.models import (
    ExternalDatabaseConnection,
    DatabaseConnectorType,
    DatabaseQueryHistory,
    TextToSQLExample,
    User,
)
from backend.services.connectors.database.postgresql import PostgreSQLConnector
from backend.services.connectors.database.mysql import MySQLConnector
from backend.services.connectors.database.mongodb import MongoDBConnector
from backend.services.connectors.database.sqlite import SQLiteConnector
from backend.services.connectors.database.base import DatabaseSchema
from backend.services.text_to_sql.service import TextToSQLService

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class DatabaseConnectionCreate(BaseModel):
    """Create a new database connection."""
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    connector_type: DatabaseConnectorType
    host: str = Field(..., max_length=500)
    port: int = Field(..., ge=1, le=65535)
    database_name: str = Field(..., max_length=255)
    username: str = Field(..., max_length=255)
    password: str = Field(..., min_length=1)
    ssl_mode: Optional[str] = Field(default="prefer")
    schema_name: Optional[str] = Field(default="public")
    max_rows: int = Field(default=1000, ge=10, le=10000)
    query_timeout_seconds: int = Field(default=30, ge=5, le=300)


class ExternalDatabaseConnectionUpdate(BaseModel):
    """Update a database connection."""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    host: Optional[str] = Field(None, max_length=500)
    port: Optional[int] = Field(None, ge=1, le=65535)
    database_name: Optional[str] = Field(None, max_length=255)
    username: Optional[str] = Field(None, max_length=255)
    password: Optional[str] = None
    ssl_mode: Optional[str] = None
    schema_name: Optional[str] = None
    max_rows: Optional[int] = Field(None, ge=10, le=10000)
    query_timeout_seconds: Optional[int] = Field(None, ge=5, le=300)
    is_active: Optional[bool] = None


class ExternalDatabaseConnectionResponse(BaseModel):
    """Database connection response."""
    id: str
    name: str
    description: Optional[str]
    connector_type: DatabaseConnectorType
    host: str  # Will be partially masked
    port: int
    database_name: str
    username: str  # Will be partially masked
    ssl_mode: Optional[str]
    schema_name: Optional[str]
    max_rows: int
    query_timeout_seconds: int
    is_active: bool
    last_tested_at: Optional[datetime]
    last_test_success: Optional[bool]
    total_queries: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class DatabaseSchemaResponse(BaseModel):
    """Database schema response."""
    database_name: str
    connector_type: str
    tables: List[Dict[str, Any]]
    views: List[Dict[str, Any]]
    last_updated: datetime


class TestConnectionResponse(BaseModel):
    """Test connection response."""
    success: bool
    message: Optional[str]
    latency_ms: Optional[float]


class QueryRequest(BaseModel):
    """Natural language query request."""
    question: str = Field(..., min_length=1, max_length=2000)
    execute: bool = Field(default=True, description="Whether to execute the generated SQL")
    explain: bool = Field(default=True, description="Whether to generate an explanation")


class QueryResponse(BaseModel):
    """Query response."""
    success: bool
    natural_language_query: str
    generated_sql: Optional[str]
    explanation: Optional[str]
    columns: List[str] = []
    rows: List[List[Any]] = []
    row_count: int = 0
    execution_time_ms: float = 0
    confidence: float = 0
    error: Optional[str]
    query_id: Optional[str]  # ID of the saved query history record


class QueryHistoryResponse(BaseModel):
    """Query history response."""
    id: str
    natural_language_query: str
    generated_sql: str
    explanation: Optional[str]
    execution_success: bool
    execution_time_ms: float
    row_count: int
    confidence_score: float
    user_rating: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackRequest(BaseModel):
    """Feedback request for a query."""
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = Field(None, max_length=1000)
    is_correct: bool = Field(default=True)


# =============================================================================
# Helper Functions
# =============================================================================

def mask_string(s: str, visible_chars: int = 4) -> str:
    """Mask a string, showing only the first few characters."""
    if len(s) <= visible_chars:
        return "*" * len(s)
    return s[:visible_chars] + "*" * (len(s) - visible_chars)


def encrypt_credential(value: str) -> str:
    """
    Encrypt a credential value.

    In production, this should use proper encryption (e.g., Fernet).
    For now, using base64 as a placeholder.
    """
    import base64
    return base64.b64encode(value.encode()).decode()


def decrypt_credential(value: str) -> str:
    """
    Decrypt a credential value.

    In production, this should use proper decryption.
    """
    import base64
    return base64.b64decode(value.encode()).decode()


async def get_connector(connection: ExternalDatabaseConnection):
    """Create a connector instance from a database connection model."""
    if connection.connector_type == DatabaseConnectorType.POSTGRESQL:
        return PostgreSQLConnector(
            host=decrypt_credential(connection.host_encrypted),
            port=connection.port,
            database=connection.database_name,
            username=decrypt_credential(connection.username_encrypted),
            password=decrypt_credential(connection.password_encrypted),
            ssl_mode=connection.ssl_mode,
            schema=connection.schema_name or "public",
            max_rows=connection.max_rows,
            query_timeout_seconds=connection.query_timeout_seconds,
        )
    elif connection.connector_type == DatabaseConnectorType.MYSQL:
        return MySQLConnector(
            host=decrypt_credential(connection.host_encrypted),
            port=connection.port,
            database=connection.database_name,
            username=decrypt_credential(connection.username_encrypted),
            password=decrypt_credential(connection.password_encrypted),
            ssl_mode=connection.ssl_mode,
            max_rows=connection.max_rows,
            query_timeout_seconds=connection.query_timeout_seconds,
        )
    elif connection.connector_type == DatabaseConnectorType.MONGODB:
        # MongoDB-specific settings may be stored in schema_cache or use defaults
        extra_opts = connection.schema_cache or {}
        return MongoDBConnector(
            host=decrypt_credential(connection.host_encrypted),
            port=connection.port,
            database=connection.database_name,
            username=decrypt_credential(connection.username_encrypted),
            password=decrypt_credential(connection.password_encrypted),
            auth_source=extra_opts.get("auth_source", "admin"),
            replica_set=extra_opts.get("replica_set"),
            max_rows=connection.max_rows,
            query_timeout_seconds=connection.query_timeout_seconds,
        )
    elif connection.connector_type == DatabaseConnectorType.SQLITE:
        # For SQLite, the host field stores the database file path
        return SQLiteConnector(
            database_path=decrypt_credential(connection.host_encrypted),
            max_rows=connection.max_rows,
            query_timeout_seconds=connection.query_timeout_seconds,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported connector type: {connection.connector_type}"
        )


async def get_user_id(db: AsyncSession, user: AuthenticatedUser) -> UUID:
    """Get the database user ID from the authenticated user."""
    try:
        user_uuid = UUID(user.user_id)
        result = await db.execute(select(User).where(User.id == user_uuid))
        db_user = result.scalar_one_or_none()
        if db_user:
            return db_user.id
    except (ValueError, TypeError):
        pass

    # Try finding by email
    if user.email:
        result = await db.execute(select(User).where(User.email == user.email))
        db_user = result.scalar_one_or_none()
        if db_user:
            return db_user.id

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")


# =============================================================================
# Connection Endpoints
# =============================================================================

@router.post("/connections", response_model=ExternalDatabaseConnectionResponse)
async def create_connection(
    request: DatabaseConnectionCreate,
    user: AuthenticatedUser,
    organization_id: Optional[str] = Depends(get_current_organization_id),
):
    """Create a new database connection."""
    logger.info(
        "Creating database connection",
        name=request.name,
        connector_type=request.connector_type,
    )

    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        # Create the connection
        connection = ExternalDatabaseConnection(
            user_id=user_id,
            organization_id=UUID(organization_id) if organization_id else None,
            name=request.name,
            description=request.description,
            connector_type=request.connector_type,
            host_encrypted=encrypt_credential(request.host),
            port=request.port,
            database_name=request.database_name,
            username_encrypted=encrypt_credential(request.username),
            password_encrypted=encrypt_credential(request.password),
            ssl_mode=request.ssl_mode,
            schema_name=request.schema_name,
            max_rows=request.max_rows,
            query_timeout_seconds=request.query_timeout_seconds,
        )

        db.add(connection)
        await db.commit()
        await db.refresh(connection)

        return ExternalDatabaseConnectionResponse(
            id=str(connection.id),
            name=connection.name,
            description=connection.description,
            connector_type=connection.connector_type,
            host=mask_string(request.host),
            port=connection.port,
            database_name=connection.database_name,
            username=mask_string(request.username),
            ssl_mode=connection.ssl_mode,
            schema_name=connection.schema_name,
            max_rows=connection.max_rows,
            query_timeout_seconds=connection.query_timeout_seconds,
            is_active=connection.is_active,
            last_tested_at=connection.last_tested_at,
            last_test_success=connection.last_test_success,
            total_queries=connection.total_queries,
            created_at=connection.created_at,
            updated_at=connection.updated_at,
        )


@router.get("/connections", response_model=List[ExternalDatabaseConnectionResponse])
async def list_connections(
    user: AuthenticatedUser,
    organization_id: Optional[str] = Depends(get_current_organization_id),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List database connections for the current user."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        query = (
            select(ExternalDatabaseConnection)
            .where(
                ExternalDatabaseConnection.user_id == user_id,
                ExternalDatabaseConnection.is_active == True,
            )
            .order_by(ExternalDatabaseConnection.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        result = await db.execute(query)
        connections = result.scalars().all()

        return [
            ExternalDatabaseConnectionResponse(
                id=str(c.id),
                name=c.name,
                description=c.description,
                connector_type=c.connector_type,
                host=mask_string(decrypt_credential(c.host_encrypted)),
                port=c.port,
                database_name=c.database_name,
                username=mask_string(decrypt_credential(c.username_encrypted)),
                ssl_mode=c.ssl_mode,
                schema_name=c.schema_name,
                max_rows=c.max_rows,
                query_timeout_seconds=c.query_timeout_seconds,
                is_active=c.is_active,
                last_tested_at=c.last_tested_at,
                last_test_success=c.last_test_success,
                total_queries=c.total_queries,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in connections
        ]


@router.get("/connections/{connection_id}", response_model=ExternalDatabaseConnectionResponse)
async def get_connection(
    connection_id: UUID,
    user: AuthenticatedUser,
):
    """Get a specific database connection."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        result = await db.execute(
            select(ExternalDatabaseConnection).where(
                ExternalDatabaseConnection.id == connection_id,
                ExternalDatabaseConnection.user_id == user_id,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")

        return ExternalDatabaseConnectionResponse(
            id=str(connection.id),
            name=connection.name,
            description=connection.description,
            connector_type=connection.connector_type,
            host=mask_string(decrypt_credential(connection.host_encrypted)),
            port=connection.port,
            database_name=connection.database_name,
            username=mask_string(decrypt_credential(connection.username_encrypted)),
            ssl_mode=connection.ssl_mode,
            schema_name=connection.schema_name,
            max_rows=connection.max_rows,
            query_timeout_seconds=connection.query_timeout_seconds,
            is_active=connection.is_active,
            last_tested_at=connection.last_tested_at,
            last_test_success=connection.last_test_success,
            total_queries=connection.total_queries,
            created_at=connection.created_at,
            updated_at=connection.updated_at,
        )


@router.put("/connections/{connection_id}", response_model=ExternalDatabaseConnectionResponse)
async def update_connection(
    connection_id: UUID,
    request: ExternalDatabaseConnectionUpdate,
    user: AuthenticatedUser,
):
    """Update a database connection."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        result = await db.execute(
            select(ExternalDatabaseConnection).where(
                ExternalDatabaseConnection.id == connection_id,
                ExternalDatabaseConnection.user_id == user_id,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")

        # Update fields
        if request.name is not None:
            connection.name = request.name
        if request.description is not None:
            connection.description = request.description
        if request.host is not None:
            connection.host_encrypted = encrypt_credential(request.host)
        if request.port is not None:
            connection.port = request.port
        if request.database_name is not None:
            connection.database_name = request.database_name
        if request.username is not None:
            connection.username_encrypted = encrypt_credential(request.username)
        if request.password is not None:
            connection.password_encrypted = encrypt_credential(request.password)
        if request.ssl_mode is not None:
            connection.ssl_mode = request.ssl_mode
        if request.schema_name is not None:
            connection.schema_name = request.schema_name
        if request.max_rows is not None:
            connection.max_rows = request.max_rows
        if request.query_timeout_seconds is not None:
            connection.query_timeout_seconds = request.query_timeout_seconds
        if request.is_active is not None:
            connection.is_active = request.is_active

        # Clear cached schema if connection details changed
        connection.schema_cache = None
        connection.schema_cached_at = None

        await db.commit()
        await db.refresh(connection)

        return ExternalDatabaseConnectionResponse(
            id=str(connection.id),
            name=connection.name,
            description=connection.description,
            connector_type=connection.connector_type,
            host=mask_string(decrypt_credential(connection.host_encrypted)),
            port=connection.port,
            database_name=connection.database_name,
            username=mask_string(decrypt_credential(connection.username_encrypted)),
            ssl_mode=connection.ssl_mode,
            schema_name=connection.schema_name,
            max_rows=connection.max_rows,
            query_timeout_seconds=connection.query_timeout_seconds,
            is_active=connection.is_active,
            last_tested_at=connection.last_tested_at,
            last_test_success=connection.last_test_success,
            total_queries=connection.total_queries,
            created_at=connection.created_at,
            updated_at=connection.updated_at,
        )


@router.delete("/connections/{connection_id}")
async def delete_connection(
    connection_id: UUID,
    user: AuthenticatedUser,
):
    """Delete a database connection."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        result = await db.execute(
            select(ExternalDatabaseConnection).where(
                ExternalDatabaseConnection.id == connection_id,
                ExternalDatabaseConnection.user_id == user_id,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")

        await db.delete(connection)
        await db.commit()

        return {"message": "Connection deleted"}


# =============================================================================
# Connection Test & Schema Endpoints
# =============================================================================

@router.post("/connections/{connection_id}/test", response_model=TestConnectionResponse)
async def test_connection(
    connection_id: UUID,
    user: AuthenticatedUser,
):
    """Test a database connection."""
    import time

    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        result = await db.execute(
            select(ExternalDatabaseConnection).where(
                ExternalDatabaseConnection.id == connection_id,
                ExternalDatabaseConnection.user_id == user_id,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")

        start_time = time.time()

        try:
            connector = await get_connector(connection)
            success, error = await connector.test_connection()
            latency = (time.time() - start_time) * 1000

            # Update connection status
            connection.last_tested_at = datetime.utcnow()
            connection.last_test_success = success
            connection.last_test_error = error
            await db.commit()

            if success:
                await connector.disconnect()
                return TestConnectionResponse(
                    success=True,
                    message="Connection successful",
                    latency_ms=latency,
                )
            else:
                return TestConnectionResponse(
                    success=False,
                    message=error,
                    latency_ms=latency,
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            connection.last_tested_at = datetime.utcnow()
            connection.last_test_success = False
            connection.last_test_error = str(e)
            await db.commit()

            return TestConnectionResponse(
                success=False,
                message=str(e),
                latency_ms=latency,
            )


@router.get("/connections/{connection_id}/schema", response_model=DatabaseSchemaResponse)
async def get_database_schema(
    connection_id: UUID,
    user: AuthenticatedUser,
    refresh: bool = Query(default=False),
):
    """Get the database schema for a connection."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        result = await db.execute(
            select(ExternalDatabaseConnection).where(
                ExternalDatabaseConnection.id == connection_id,
                ExternalDatabaseConnection.user_id == user_id,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")

        # Check cache
        if not refresh and connection.schema_cache and connection.schema_cached_at:
            return DatabaseSchemaResponse(
                database_name=connection.database_name,
                connector_type=connection.connector_type.value,
                tables=connection.schema_cache.get("tables", []),
                views=connection.schema_cache.get("views", []),
                last_updated=connection.schema_cached_at,
            )

        try:
            connector = await get_connector(connection)
            schema = await connector.get_schema(refresh=True)
            await connector.disconnect()

            # Cache the schema
            connection.schema_cache = {
                "tables": [t.model_dump() for t in schema.tables],
                "views": [v.model_dump() for v in schema.views],
            }
            connection.schema_cached_at = datetime.utcnow()
            await db.commit()

            return DatabaseSchemaResponse(
                database_name=schema.database_name,
                connector_type=schema.connector_type.value,
                tables=[t.model_dump() for t in schema.tables],
                views=[v.model_dump() for v in schema.views],
                last_updated=schema.last_updated,
            )

        except Exception as e:
            logger.error("Failed to get schema", error=str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get schema: {e}")


# =============================================================================
# Query Endpoints
# =============================================================================

@router.post("/connections/{connection_id}/query", response_model=QueryResponse)
async def query_database(
    connection_id: UUID,
    request: QueryRequest,
    user: AuthenticatedUser,
):
    """Execute a natural language query against a database."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        result = await db.execute(
            select(ExternalDatabaseConnection).where(
                ExternalDatabaseConnection.id == connection_id,
                ExternalDatabaseConnection.user_id == user_id,
                ExternalDatabaseConnection.is_active == True,
            )
        )
        connection = result.scalar_one_or_none()

        if not connection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")

        try:
            connector = await get_connector(connection)
            service = TextToSQLService(connector)

            # Load verified examples for few-shot prompting
            examples_result = await db.execute(
                select(TextToSQLExample).where(
                    TextToSQLExample.connection_id == connection_id,
                    TextToSQLExample.is_verified == True,
                ).order_by(TextToSQLExample.times_used.desc()).limit(5)
            )
            examples = examples_result.scalars().all()
            for ex in examples:
                service.add_example(ex.question, ex.sql, ex.explanation)

            # Execute the query
            query_result = await service.query(
                question=request.question,
                execute=request.execute,
                explain=request.explain,
            )

            await connector.disconnect()

            # Save query history
            history = DatabaseQueryHistory(
                connection_id=connection_id,
                user_id=user_id,
                natural_language_query=request.question,
                generated_sql=query_result.generated_sql or "",
                explanation=query_result.explanation,
                execution_success=query_result.success,
                execution_time_ms=query_result.query_result.execution_time_ms if query_result.query_result else 0,
                row_count=query_result.query_result.row_count if query_result.query_result else 0,
                error_message=query_result.error,
                generation_attempts=query_result.attempts,
                confidence_score=query_result.confidence,
            )
            db.add(history)

            # Update connection stats
            connection.total_queries += 1
            if query_result.query_result:
                connection.total_query_time_ms += int(query_result.query_result.execution_time_ms)

            await db.commit()
            await db.refresh(history)

            return QueryResponse(
                success=query_result.success,
                natural_language_query=request.question,
                generated_sql=query_result.generated_sql,
                explanation=query_result.explanation,
                columns=query_result.query_result.columns if query_result.query_result else [],
                rows=query_result.query_result.rows if query_result.query_result else [],
                row_count=query_result.query_result.row_count if query_result.query_result else 0,
                execution_time_ms=query_result.query_result.execution_time_ms if query_result.query_result else 0,
                confidence=query_result.confidence,
                error=query_result.error,
                query_id=str(history.id),
            )

        except Exception as e:
            logger.error("Query failed", error=str(e), question=request.question)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query failed: {e}")


@router.get("/connections/{connection_id}/history", response_model=List[QueryHistoryResponse])
async def get_query_history(
    connection_id: UUID,
    user: AuthenticatedUser,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Get query history for a connection."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        # Verify user owns the connection
        result = await db.execute(
            select(ExternalDatabaseConnection).where(
                ExternalDatabaseConnection.id == connection_id,
                ExternalDatabaseConnection.user_id == user_id,
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")

        # Get history
        query = (
            select(DatabaseQueryHistory)
            .where(DatabaseQueryHistory.connection_id == connection_id)
            .order_by(DatabaseQueryHistory.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        result = await db.execute(query)
        history = result.scalars().all()

        return [
            QueryHistoryResponse(
                id=str(h.id),
                natural_language_query=h.natural_language_query,
                generated_sql=h.generated_sql,
                explanation=h.explanation,
                execution_success=h.execution_success,
                execution_time_ms=h.execution_time_ms,
                row_count=h.row_count,
                confidence_score=h.confidence_score,
                user_rating=h.user_rating,
                created_at=h.created_at,
            )
            for h in history
        ]


@router.post("/history/{query_id}/feedback")
async def submit_feedback(
    query_id: UUID,
    request: FeedbackRequest,
    user: AuthenticatedUser,
):
    """Submit feedback for a query."""
    async with async_session_context() as db:
        user_id = await get_user_id(db, user)

        result = await db.execute(
            select(DatabaseQueryHistory).where(
                DatabaseQueryHistory.id == query_id,
                DatabaseQueryHistory.user_id == user_id,
            )
        )
        history = result.scalar_one_or_none()

        if not history:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query not found")

        # Update feedback
        history.user_rating = request.rating
        history.user_feedback = request.feedback
        history.is_verified = request.is_correct

        # If highly rated and correct, create a verified example
        if request.rating >= 4 and request.is_correct:
            example = TextToSQLExample(
                connection_id=history.connection_id,
                question=history.natural_language_query,
                sql=history.generated_sql,
                explanation=history.explanation,
                is_verified=True,
                verified_by_id=user_id,
                verified_at=datetime.utcnow(),
            )
            db.add(example)

        await db.commit()

        return {"message": "Feedback submitted"}
