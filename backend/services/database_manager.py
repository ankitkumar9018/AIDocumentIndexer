"""
AIDocumentIndexer - Database Management Service
================================================

Manages database configuration, connection testing, data export/import,
and PostgreSQL setup with pgvector extension.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select, func, update

from backend.db.models import (
    Base,
    User,
    Document,
    Chunk,
    AccessTier,
    ChatSession,
    ChatMessage,
    ProcessingQueue,
    AuditLog,
    SystemSettings,
    ScrapedContent,
    DatabaseConnection,
)
from backend.services.encryption import encrypt_value, decrypt_value, mask_api_key

logger = structlog.get_logger(__name__)


def convert_to_async_url(db_url: str) -> str:
    """Convert sync database URL to async URL."""
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+asyncpg://")
    elif db_url.startswith("sqlite://"):
        return db_url.replace("sqlite://", "sqlite+aiosqlite://")
    elif db_url.startswith("mysql://"):
        return db_url.replace("mysql://", "mysql+aiomysql://")
    return db_url


def convert_to_sync_url(db_url: str) -> str:
    """Convert async database URL to sync URL."""
    url = db_url.replace("+asyncpg", "")
    url = url.replace("+aiosqlite", "")
    url = url.replace("+aiomysql", "")

    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg2://")
    return url


def mask_password(db_url: str) -> str:
    """Mask password in database URL for display."""
    # Pattern: protocol://user:password@host
    return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', db_url)


class DatabaseManager:
    """
    Service for managing database configuration and migrations.

    Features:
    - Get current database info and stats
    - Test database connections
    - Export/import data for migrations
    - Setup PostgreSQL with pgvector
    """

    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session

    async def get_info(self) -> Dict[str, Any]:
        """
        Get current database configuration and statistics.

        Returns:
            Dictionary with database type, status, and counts
        """
        from backend.db.database import db_config

        url = db_config.database_url
        db_type = "sqlite" if "sqlite" in url.lower() else "postgresql" if "postgresql" in url.lower() else "mysql"

        # Mask password in URL
        masked_url = mask_password(url)

        # Detect vector store type
        vector_store = "chromadb" if db_type == "sqlite" else "pgvector"

        # Get counts if we have a session
        docs_count = 0
        chunks_count = 0
        users_count = 0

        if self.session:
            try:
                docs_result = await self.session.execute(select(func.count(Document.id)))
                docs_count = docs_result.scalar() or 0

                chunks_result = await self.session.execute(select(func.count(Chunk.id)))
                chunks_count = chunks_result.scalar() or 0

                users_result = await self.session.execute(select(func.count(User.id)))
                users_count = users_result.scalar() or 0
            except Exception as e:
                logger.warning("Failed to get database counts", error=str(e))

        return {
            "type": db_type,
            "url_masked": masked_url,
            "is_connected": True,
            "vector_store": vector_store,
            "documents_count": docs_count,
            "chunks_count": chunks_count,
            "users_count": users_count,
        }

    async def test_connection(self, db_url: str) -> Dict[str, Any]:
        """
        Test if a database URL is valid and connectable.

        Args:
            db_url: Database connection URL to test

        Returns:
            Dictionary with success status and details
        """
        try:
            # Determine database type
            is_postgresql = "postgresql" in db_url.lower()

            # Create async engine
            async_url = convert_to_async_url(db_url)
            engine = create_async_engine(async_url, echo=False)

            async with engine.connect() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()

            await engine.dispose()

            # Check for pgvector if PostgreSQL
            has_pgvector = False
            pgvector_version = None

            if is_postgresql:
                has_pgvector, pgvector_version = await self._check_pgvector(db_url)

            return {
                "success": True,
                "database_type": "postgresql" if is_postgresql else "sqlite",
                "has_pgvector": has_pgvector,
                "pgvector_version": pgvector_version,
                "message": "Connection successful"
            }

        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Connection failed: {str(e)}"
            }

    async def _check_pgvector(self, db_url: str) -> tuple[bool, Optional[str]]:
        """Check if pgvector extension is available."""
        try:
            async_url = convert_to_async_url(db_url)
            engine = create_async_engine(async_url, echo=False)

            async with engine.connect() as conn:
                # Check if vector extension exists
                result = await conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
                row = result.fetchone()

                if row:
                    return True, row[0]
                return False, None

        except Exception as e:
            logger.warning("Failed to check pgvector", error=str(e))
            return False, None
        finally:
            await engine.dispose()

    async def export_data(self) -> Dict[str, Any]:
        """
        Export all data to JSON format for migration.

        Returns:
            Dictionary containing all exportable data
        """
        if not self.session:
            raise ValueError("Database session required for export")

        data = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "access_tiers": [],
            "users": [],
            "documents": [],
            "chunks": [],
            "chat_sessions": [],
            "chat_messages": [],
            "system_settings": [],
        }

        try:
            # Export access tiers
            tiers_result = await self.session.execute(select(AccessTier))
            for tier in tiers_result.scalars().all():
                data["access_tiers"].append({
                    "id": str(tier.id),
                    "name": tier.name,
                    "level": tier.level,
                    "description": tier.description,
                    "color": tier.color,
                    "created_at": tier.created_at.isoformat() if tier.created_at else None,
                })

            # Export users (without passwords for security)
            users_result = await self.session.execute(select(User))
            for user in users_result.scalars().all():
                data["users"].append({
                    "id": str(user.id),
                    "email": user.email,
                    "name": user.name,
                    "is_active": user.is_active,
                    "access_tier_id": str(user.access_tier_id) if user.access_tier_id else None,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
                })

            # Export documents
            docs_result = await self.session.execute(select(Document))
            for doc in docs_result.scalars().all():
                data["documents"].append({
                    "id": str(doc.id),
                    "filename": doc.filename,
                    "file_path": doc.file_path,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "file_hash": doc.file_hash,
                    "title": doc.title,
                    "description": doc.description,
                    "status": doc.status,
                    "access_tier_id": str(doc.access_tier_id) if doc.access_tier_id else None,
                    "uploaded_by_id": str(doc.uploaded_by_id) if doc.uploaded_by_id else None,
                    "collection": doc.collection,
                    "tags": doc.tags if hasattr(doc, 'tags') else [],
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                })

            # Export chunks (note: embeddings are not exported as they need regeneration)
            chunks_result = await self.session.execute(select(Chunk))
            for chunk in chunks_result.scalars().all():
                data["chunks"].append({
                    "id": str(chunk.id),
                    "document_id": str(chunk.document_id) if chunk.document_id else None,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number if hasattr(chunk, 'page_number') else None,
                    "section_title": chunk.section_title if hasattr(chunk, 'section_title') else None,
                    "char_count": chunk.char_count if hasattr(chunk, 'char_count') else len(chunk.content),
                    "word_count": chunk.word_count if hasattr(chunk, 'word_count') else len(chunk.content.split()),
                    "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                    # Note: embedding is NOT exported - must be regenerated
                })

            # Export chat sessions
            sessions_result = await self.session.execute(select(ChatSession))
            for session in sessions_result.scalars().all():
                data["chat_sessions"].append({
                    "id": str(session.id),
                    "user_id": str(session.user_id) if session.user_id else None,
                    "title": session.title if hasattr(session, 'title') else None,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                })

            # Export chat messages
            messages_result = await self.session.execute(select(ChatMessage))
            for msg in messages_result.scalars().all():
                data["chat_messages"].append({
                    "id": str(msg.id),
                    "session_id": str(msg.session_id) if msg.session_id else None,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": msg.sources if hasattr(msg, 'sources') else None,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                })

            # Export system settings
            settings_result = await self.session.execute(select(SystemSettings))
            for setting in settings_result.scalars().all():
                data["system_settings"].append({
                    "id": str(setting.id),
                    "key": setting.key,
                    "value": setting.value,
                    "category": setting.category,
                    "description": setting.description,
                    "value_type": setting.value_type,
                })

            logger.info(
                "Data export completed",
                tiers=len(data["access_tiers"]),
                users=len(data["users"]),
                documents=len(data["documents"]),
                chunks=len(data["chunks"]),
            )

            return data

        except Exception as e:
            logger.error("Data export failed", error=str(e))
            raise

    async def import_data(self, data: Dict[str, Any], clear_existing: bool = False) -> Dict[str, Any]:
        """
        Import data from JSON export.

        Args:
            data: Export data dictionary
            clear_existing: If True, delete existing data before import

        Returns:
            Dictionary with import results
        """
        if not self.session:
            raise ValueError("Database session required for import")

        # Validate export version
        version = data.get("version", "")
        if not version.startswith("1."):
            raise ValueError(f"Unsupported export version: {version}")

        results = {
            "success": True,
            "imported": {
                "access_tiers": 0,
                "users": 0,
                "documents": 0,
                "chunks": 0,
                "chat_sessions": 0,
                "chat_messages": 0,
                "system_settings": 0,
            },
            "errors": [],
            "warnings": [],
        }

        try:
            # Clear existing data if requested (in reverse dependency order)
            if clear_existing:
                await self._clear_all_data()
                results["warnings"].append("Existing data was cleared before import")

            # Import in dependency order
            # 1. Access tiers first (users depend on them)
            for tier_data in data.get("access_tiers", []):
                try:
                    tier = AccessTier(
                        name=tier_data["name"],
                        level=tier_data["level"],
                        description=tier_data.get("description"),
                        color=tier_data.get("color", "#6B7280"),
                    )
                    self.session.add(tier)
                    results["imported"]["access_tiers"] += 1
                except Exception as e:
                    results["errors"].append(f"Failed to import tier {tier_data.get('name')}: {e}")

            await self.session.flush()  # Flush to get tier IDs

            # 2. Users (documents depend on them)
            # Note: passwords are not exported, users will need to reset
            for user_data in data.get("users", []):
                try:
                    # Find the access tier by level or name
                    tier = None
                    if user_data.get("access_tier_id"):
                        tier_query = select(AccessTier).where(
                            AccessTier.level == data["access_tiers"][0]["level"]  # Use first tier as default
                        )
                        tier_result = await self.session.execute(tier_query)
                        tier = tier_result.scalar_one_or_none()

                    user = User(
                        email=user_data["email"],
                        name=user_data.get("name"),
                        is_active=user_data.get("is_active", True),
                        password_hash="NEEDS_RESET",  # Users must reset password
                        access_tier_id=tier.id if tier else None,
                    )
                    self.session.add(user)
                    results["imported"]["users"] += 1
                    results["warnings"].append(f"User {user_data['email']} needs to reset password")
                except Exception as e:
                    results["errors"].append(f"Failed to import user {user_data.get('email')}: {e}")

            await self.session.flush()

            # 3. Documents
            for doc_data in data.get("documents", []):
                try:
                    doc = Document(
                        filename=doc_data["filename"],
                        file_path=doc_data.get("file_path", ""),
                        file_type=doc_data.get("file_type"),
                        file_size=doc_data.get("file_size", 0),
                        file_hash=doc_data.get("file_hash"),
                        title=doc_data.get("title"),
                        description=doc_data.get("description"),
                        status=doc_data.get("status", "pending"),
                        collection=doc_data.get("collection"),
                    )
                    self.session.add(doc)
                    results["imported"]["documents"] += 1
                except Exception as e:
                    results["errors"].append(f"Failed to import document {doc_data.get('filename')}: {e}")

            await self.session.flush()

            # Note: Chunks need embeddings regenerated - import content only
            results["warnings"].append(
                "Chunk embeddings were not imported. Run document reprocessing to regenerate embeddings."
            )

            # 4. System settings
            for setting_data in data.get("system_settings", []):
                try:
                    setting = SystemSettings(
                        key=setting_data["key"],
                        value=setting_data.get("value"),
                        category=setting_data.get("category", "general"),
                        description=setting_data.get("description"),
                        value_type=setting_data.get("value_type", "string"),
                    )
                    self.session.add(setting)
                    results["imported"]["system_settings"] += 1
                except Exception as e:
                    results["errors"].append(f"Failed to import setting {setting_data.get('key')}: {e}")

            await self.session.commit()

            logger.info("Data import completed", results=results)
            return results

        except Exception as e:
            await self.session.rollback()
            logger.error("Data import failed", error=str(e))
            results["success"] = False
            results["errors"].append(f"Import failed: {str(e)}")
            return results

    async def _clear_all_data(self) -> None:
        """Clear all data from the database (in reverse dependency order)."""
        if not self.session:
            return

        # Delete in reverse dependency order
        await self.session.execute(text("DELETE FROM chat_messages"))
        await self.session.execute(text("DELETE FROM chat_sessions"))
        await self.session.execute(text("DELETE FROM chunks"))
        await self.session.execute(text("DELETE FROM documents"))
        await self.session.execute(text("DELETE FROM processing_queue"))
        await self.session.execute(text("DELETE FROM audit_log"))
        await self.session.execute(text("DELETE FROM system_settings"))
        await self.session.execute(text("DELETE FROM users"))
        await self.session.execute(text("DELETE FROM access_tiers"))
        await self.session.flush()

    async def setup_postgresql(self, db_url: str) -> Dict[str, Any]:
        """
        Setup PostgreSQL database with pgvector extension and tables.

        Args:
            db_url: PostgreSQL connection URL

        Returns:
            Dictionary with setup results
        """
        try:
            # Validate it's a PostgreSQL URL
            if "postgresql" not in db_url.lower():
                return {
                    "success": False,
                    "error": "URL must be a PostgreSQL connection string"
                }

            # Use sync engine for DDL operations (more reliable for CREATE EXTENSION)
            sync_url = convert_to_sync_url(db_url)
            engine = create_engine(sync_url, echo=False)

            with engine.connect() as conn:
                # Create pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))  # For text search
                conn.commit()

                logger.info("PostgreSQL extensions created")

            # Create all tables using sync engine
            Base.metadata.create_all(bind=engine)

            engine.dispose()

            # Verify setup with async connection
            has_pgvector, version = await self._check_pgvector(db_url)

            return {
                "success": True,
                "has_pgvector": has_pgvector,
                "pgvector_version": version,
                "message": "PostgreSQL database setup completed successfully",
                "tables_created": True,
            }

        except Exception as e:
            logger.error("PostgreSQL setup failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Setup failed: {str(e)}",
                "hint": "Make sure PostgreSQL is running and pgvector extension is available. "
                        "Install pgvector: https://github.com/pgvector/pgvector#installation"
            }

    def get_migration_instructions(self, from_type: str, to_type: str) -> Dict[str, Any]:
        """
        Get instructions for migrating between database types.

        Args:
            from_type: Current database type (sqlite, postgresql)
            to_type: Target database type

        Returns:
            Dictionary with step-by-step instructions
        """
        if from_type == "sqlite" and to_type == "postgresql":
            return {
                "title": "Migrate from SQLite to PostgreSQL",
                "steps": [
                    {
                        "step": 1,
                        "title": "Install PostgreSQL",
                        "description": "Install PostgreSQL 14+ on your system",
                        "commands": [
                            "# macOS: brew install postgresql@14",
                            "# Ubuntu: sudo apt install postgresql-14",
                            "# Start service: brew services start postgresql@14"
                        ]
                    },
                    {
                        "step": 2,
                        "title": "Install pgvector extension",
                        "description": "Install pgvector for vector similarity search",
                        "commands": [
                            "# macOS: brew install pgvector",
                            "# Ubuntu: sudo apt install postgresql-14-pgvector",
                            "# Or compile from source: https://github.com/pgvector/pgvector"
                        ]
                    },
                    {
                        "step": 3,
                        "title": "Create database",
                        "description": "Create a new PostgreSQL database",
                        "commands": [
                            "createdb aidocindexer",
                            "# Or: psql -c 'CREATE DATABASE aidocindexer;'"
                        ]
                    },
                    {
                        "step": 4,
                        "title": "Export data",
                        "description": "Use the Export button to download your data as JSON"
                    },
                    {
                        "step": 5,
                        "title": "Test connection",
                        "description": "Enter your PostgreSQL URL and click 'Test Connection'"
                    },
                    {
                        "step": 6,
                        "title": "Setup database",
                        "description": "Click 'Setup pgvector' to create extensions and tables"
                    },
                    {
                        "step": 7,
                        "title": "Update environment",
                        "description": "Update your .env file with the new database URL",
                        "env_vars": {
                            "DATABASE_TYPE": "postgresql",
                            "DATABASE_URL": "postgresql://user:password@localhost:5432/aidocindexer",
                            "VECTOR_STORE_BACKEND": "auto"
                        }
                    },
                    {
                        "step": 8,
                        "title": "Restart and import",
                        "description": "Restart the server and import your data using the Import button"
                    },
                    {
                        "step": 9,
                        "title": "Regenerate embeddings",
                        "description": "Reprocess documents to regenerate vector embeddings"
                    }
                ]
            }
        elif from_type == "postgresql" and to_type == "sqlite":
            return {
                "title": "Migrate from PostgreSQL to SQLite",
                "warning": "SQLite does not support pgvector. Vector search will use ChromaDB instead.",
                "steps": [
                    {
                        "step": 1,
                        "title": "Export data",
                        "description": "Use the Export button to download your data as JSON"
                    },
                    {
                        "step": 2,
                        "title": "Update environment",
                        "description": "Update your .env file",
                        "env_vars": {
                            "DATABASE_TYPE": "sqlite",
                            "DATABASE_URL": "sqlite:///./aidocindexer.db",
                            "VECTOR_STORE_BACKEND": "auto"
                        }
                    },
                    {
                        "step": 3,
                        "title": "Restart server",
                        "description": "Restart the application. Tables will be created automatically."
                    },
                    {
                        "step": 4,
                        "title": "Import data",
                        "description": "Use the Import button to restore your data"
                    },
                    {
                        "step": 5,
                        "title": "Regenerate embeddings",
                        "description": "Reprocess documents to regenerate vector embeddings in ChromaDB"
                    }
                ]
            }
        else:
            return {
                "title": "Migration not needed",
                "message": f"You are already using {from_type}"
            }


# =============================================================================
# Database Connection Service
# =============================================================================

DATABASE_TYPES = {
    "sqlite": {
        "name": "SQLite",
        "default_port": None,
        "default_database": "aidocindexer.db",
        "supports_vector": False,
        "vector_alternative": "chromadb",
        "url_template": "sqlite:///{database}",
        "fields": ["database"],
        "required_fields": ["database"],
    },
    "postgresql": {
        "name": "PostgreSQL",
        "default_port": 5432,
        "default_database": "aidocindexer",
        "supports_vector": True,
        "vector_extension": "pgvector",
        "url_template": "postgresql://{username}:{password}@{host}:{port}/{database}",
        "fields": ["host", "port", "database", "username", "password"],
        "required_fields": ["host", "database", "username", "password"],
    },
    "mysql": {
        "name": "MySQL",
        "default_port": 3306,
        "default_database": "aidocindexer",
        "supports_vector": False,
        "vector_alternative": "chromadb",
        "url_template": "mysql://{username}:{password}@{host}:{port}/{database}",
        "fields": ["host", "port", "database", "username", "password"],
        "required_fields": ["host", "database", "username", "password"],
    },
}


class DatabaseConnectionService:
    """Service for managing saved database connections."""

    @staticmethod
    async def list_connections(session: AsyncSession) -> List[DatabaseConnection]:
        """List all saved database connections."""
        result = await session.execute(
            select(DatabaseConnection).order_by(DatabaseConnection.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_connection(session: AsyncSession, connection_id: str) -> Optional[DatabaseConnection]:
        """Get a specific connection by ID."""
        result = await session.execute(
            select(DatabaseConnection).where(DatabaseConnection.id == connection_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_connection_by_name(session: AsyncSession, name: str) -> Optional[DatabaseConnection]:
        """Get a connection by name."""
        result = await session.execute(
            select(DatabaseConnection).where(DatabaseConnection.name == name)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_active_connection(session: AsyncSession) -> Optional[DatabaseConnection]:
        """Get the currently active database connection."""
        result = await session.execute(
            select(DatabaseConnection).where(DatabaseConnection.is_active == True)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create_connection(
        session: AsyncSession,
        name: str,
        db_type: str,
        database: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        vector_store: str = "auto",
        is_active: bool = False,
        connection_options: Optional[Dict[str, Any]] = None,
    ) -> DatabaseConnection:
        """Create a new database connection configuration."""
        # Validate database type
        if db_type not in DATABASE_TYPES:
            raise ValueError(f"Invalid database type: {db_type}")

        # Set default port if not provided
        type_config = DATABASE_TYPES[db_type]
        if port is None and type_config.get("default_port"):
            port = type_config["default_port"]

        # Encrypt password if provided
        encrypted_password = encrypt_value(password) if password else None

        # If this is being set as active, clear other active connections first
        if is_active:
            await session.execute(
                update(DatabaseConnection).values(is_active=False)
            )

        connection = DatabaseConnection(
            name=name,
            db_type=db_type,
            host=host,
            port=port,
            database=database,
            username=username,
            password_encrypted=encrypted_password,
            is_active=is_active,
            vector_store=vector_store,
            connection_options=connection_options or {},
        )

        session.add(connection)
        await session.flush()
        await session.refresh(connection)

        logger.info(
            "Created database connection",
            connection_id=str(connection.id),
            name=name,
            db_type=db_type,
        )

        return connection

    @staticmethod
    async def update_connection(
        session: AsyncSession,
        connection_id: str,
        name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        vector_store: Optional[str] = None,
        connection_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[DatabaseConnection]:
        """Update an existing database connection."""
        connection = await DatabaseConnectionService.get_connection(session, connection_id)
        if not connection:
            return None

        if name is not None:
            connection.name = name
        if host is not None:
            connection.host = host
        if port is not None:
            connection.port = port
        if database is not None:
            connection.database = database
        if username is not None:
            connection.username = username
        if password is not None:
            connection.password_encrypted = encrypt_value(password)
        if vector_store is not None:
            connection.vector_store = vector_store
        if connection_options is not None:
            connection.connection_options = connection_options

        await session.flush()
        await session.refresh(connection)

        logger.info("Updated database connection", connection_id=connection_id)
        return connection

    @staticmethod
    async def delete_connection(session: AsyncSession, connection_id: str) -> bool:
        """Delete a database connection."""
        connection = await DatabaseConnectionService.get_connection(session, connection_id)
        if not connection:
            return False

        if connection.is_active:
            raise ValueError("Cannot delete the active connection")

        await session.delete(connection)
        logger.info("Deleted database connection", connection_id=connection_id)
        return True

    @staticmethod
    async def set_active_connection(session: AsyncSession, connection_id: str) -> Optional[DatabaseConnection]:
        """Set a connection as the active connection."""
        connection = await DatabaseConnectionService.get_connection(session, connection_id)
        if not connection:
            return None

        # Clear existing active
        await session.execute(
            update(DatabaseConnection).values(is_active=False)
        )

        # Set new active
        connection.is_active = True
        await session.flush()
        await session.refresh(connection)

        logger.info("Set active database connection", connection_id=connection_id)
        return connection

    @staticmethod
    def build_connection_url(
        db_type: str,
        database: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> str:
        """Build a database connection URL from components."""
        if db_type not in DATABASE_TYPES:
            raise ValueError(f"Invalid database type: {db_type}")

        type_config = DATABASE_TYPES[db_type]
        template = type_config["url_template"]

        if db_type == "sqlite":
            return template.format(database=database)
        else:
            port = port or type_config["default_port"]
            return template.format(
                host=host or "localhost",
                port=port,
                database=database,
                username=username or "",
                password=password or "",
            )

    @staticmethod
    async def test_saved_connection(session: AsyncSession, connection_id: str) -> Dict[str, Any]:
        """Test a saved database connection."""
        connection = await DatabaseConnectionService.get_connection(session, connection_id)
        if not connection:
            return {"success": False, "error": "Connection not found"}

        # Decrypt password if present
        password = decrypt_value(connection.password_encrypted) if connection.password_encrypted else None

        # Build URL
        url = DatabaseConnectionService.build_connection_url(
            db_type=connection.db_type,
            database=connection.database,
            host=connection.host,
            port=connection.port,
            username=connection.username,
            password=password,
        )

        # Test connection
        manager = DatabaseManager()
        return await manager.test_connection(url)

    @staticmethod
    def format_connection_response(connection: DatabaseConnection) -> Dict[str, Any]:
        """Format a connection for API response (masking sensitive data)."""
        result = {
            "id": str(connection.id),
            "name": connection.name,
            "db_type": connection.db_type,
            "host": connection.host,
            "port": connection.port,
            "database": connection.database,
            "username": connection.username,
            "is_active": connection.is_active,
            "vector_store": connection.vector_store,
            "connection_options": connection.connection_options,
            "created_at": connection.created_at.isoformat() if connection.created_at else None,
            "updated_at": connection.updated_at.isoformat() if connection.updated_at else None,
        }

        # Indicate if password is set (don't show actual password)
        result["has_password"] = bool(connection.password_encrypted)

        return result

    @staticmethod
    def get_database_types() -> Dict[str, Any]:
        """Get all supported database types with their configurations."""
        return {
            db_type: {
                "name": config["name"],
                "default_port": config.get("default_port"),
                "default_database": config.get("default_database"),
                "supports_vector": config.get("supports_vector", False),
                "vector_extension": config.get("vector_extension"),
                "vector_alternative": config.get("vector_alternative"),
                "fields": config["fields"],
                "required_fields": config["required_fields"],
            }
            for db_type, config in DATABASE_TYPES.items()
        }


# Singleton instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager(session: Optional[AsyncSession] = None) -> DatabaseManager:
    """Get database manager instance."""
    return DatabaseManager(session)
