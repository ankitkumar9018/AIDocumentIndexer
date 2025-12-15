"""
AIDocumentIndexer - SQLAlchemy Models
======================================

Database models with multi-database support (PostgreSQL, SQLite, MySQL).
Uses SQLAlchemy 2.0 with async support and pgvector for embeddings.
"""

import json
import os
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    BigInteger,
    event,
    func,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import expression
from sqlalchemy.types import CHAR, TypeEngine

# Conditional import for pgvector
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None

# Check if we're using SQLite (for testing or development)
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "postgresql")
IS_SQLITE = DATABASE_TYPE == "sqlite" or "sqlite" in os.getenv("DATABASE_URL", "")


# =============================================================================
# Database-agnostic Type Decorators
# =============================================================================

class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type when available, otherwise uses
    CHAR(36) for SQLite/other databases.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect) -> TypeEngine:
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid.UUID):
                return str(value)
            return value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class JSONType(TypeDecorator):
    """Platform-independent JSON type.

    Uses PostgreSQL's JSONB when available, otherwise uses Text with JSON serialization.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect) -> TypeEngine:
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB)
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        return json.loads(value)


class StringArrayType(TypeDecorator):
    """Platform-independent string array type.

    Uses PostgreSQL's ARRAY(String) when available, otherwise stores as JSON.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect) -> TypeEngine:
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(ARRAY(String))
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        return json.loads(value)


class UUIDArrayType(TypeDecorator):
    """Platform-independent UUID array type.

    Uses PostgreSQL's ARRAY(UUID) when available, otherwise stores as JSON.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect) -> TypeEngine:
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(ARRAY(PG_UUID(as_uuid=True)))
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        return json.dumps([str(v) for v in value])

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        return [uuid.UUID(v) for v in json.loads(value)]


# =============================================================================
# Enums
# =============================================================================

class ProcessingStatus(str, PyEnum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(str, PyEnum):
    """Document processing mode."""
    FULL = "full"
    SMART = "smart"
    TEXT_ONLY = "text_only"


class StorageMode(str, PyEnum):
    """Document storage mode."""
    RAG = "rag"
    QUERY_ONLY = "query_only"


class MessageRole(str, PyEnum):
    """Chat message role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# =============================================================================
# Base Model
# =============================================================================

class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# Mixins
# =============================================================================

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDMixin:
    """Mixin for UUID primary key."""
    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )


# =============================================================================
# Models
# =============================================================================

class AccessTier(Base, UUIDMixin, TimestampMixin):
    """
    Access tier for permission management.

    Tiers have a numeric level (1-100) that determines document access.
    Admins can create custom tiers dynamically.
    """
    __tablename__ = "access_tiers"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    color: Mapped[str] = mapped_column(String(7), default="#6B7280")

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="access_tier")
    documents: Mapped[List["Document"]] = relationship("Document", back_populates="access_tier")
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="access_tier")

    def __repr__(self) -> str:
        return f"<AccessTier(name='{self.name}', level={self.level})>"


class User(Base, UUIDMixin, TimestampMixin):
    """
    User account model.

    Users are created by admins and have an associated access tier
    that determines which documents they can access.
    """
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Foreign keys
    access_tier_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("access_tiers.id"),
        nullable=False,
    )
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id"),
    )

    # Relationships
    access_tier: Mapped["AccessTier"] = relationship("AccessTier", back_populates="users")
    created_by: Mapped[Optional["User"]] = relationship("User", remote_side="User.id")
    documents: Mapped[List["Document"]] = relationship("Document", back_populates="uploaded_by")
    chat_sessions: Mapped[List["ChatSession"]] = relationship("ChatSession", back_populates="user")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(email='{self.email}', tier='{self.access_tier.name if self.access_tier else None}')>"


class Document(Base, UUIDMixin, TimestampMixin):
    """
    Document model representing an uploaded file.

    Stores metadata about documents and their processing status.
    The actual file content is stored in chunks.
    """
    __tablename__ = "documents"

    # File identification
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))

    # Processing info
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus),
        default=ProcessingStatus.PENDING,
        index=True,
    )
    processing_mode: Mapped[ProcessingMode] = mapped_column(
        Enum(ProcessingMode),
        default=ProcessingMode.SMART,
    )
    storage_mode: Mapped[StorageMode] = mapped_column(
        Enum(StorageMode),
        default=StorageMode.RAG,
    )
    processing_error: Mapped[Optional[str]] = mapped_column(Text)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Metadata
    title: Mapped[Optional[str]] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10), default="en")
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    tags: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())

    # Source info (for auto-indexed files)
    source_path: Mapped[Optional[str]] = mapped_column(String(1000))
    is_auto_indexed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Foreign keys
    access_tier_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("access_tiers.id"),
        nullable=False,
        index=True,
    )
    uploaded_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id"),
    )

    # Relationships
    access_tier: Mapped["AccessTier"] = relationship("AccessTier", back_populates="documents")
    uploaded_by: Mapped[Optional["User"]] = relationship("User", back_populates="documents")
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    processing_queue: Mapped[Optional["ProcessingQueue"]] = relationship(
        "ProcessingQueue",
        back_populates="document",
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<Document(filename='{self.filename}', status='{self.processing_status}')>"


class Chunk(Base, UUIDMixin):
    """
    Document chunk with embedding.

    Documents are split into chunks for RAG retrieval.
    Each chunk has an embedding vector for similarity search.
    """
    __tablename__ = "chunks"

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Embedding - using dynamic column type based on database
    # For PostgreSQL with pgvector, this will be a Vector type
    # For other databases, we'll store as JSON or binary
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536) if HAS_PGVECTOR else Text,  # Fallback to Text for non-pgvector
        nullable=True,
    )

    # Position info
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    section_title: Mapped[Optional[str]] = mapped_column(String(500))

    # Metadata
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    char_count: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Foreign keys
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    access_tier_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("access_tiers.id"),
        nullable=False,
        index=True,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    access_tier: Mapped["AccessTier"] = relationship("AccessTier", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk(document_id='{self.document_id}', index={self.chunk_index})>"


class ScrapedContent(Base, UUIDMixin):
    """
    Web scraped content.

    Stores content scraped from websites, optionally with embeddings
    for RAG integration.
    """
    __tablename__ = "scraped_content"

    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Embedding
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536) if HAS_PGVECTOR else Text,
        nullable=True,
    )

    # Storage preference
    stored_permanently: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Foreign keys
    access_tier_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("access_tiers.id"),
    )
    scraped_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id"),
    )

    def __repr__(self) -> str:
        return f"<ScrapedContent(url='{self.url[:50]}...', permanent={self.stored_permanently})>"


class ChatSession(Base, UUIDMixin, TimestampMixin):
    """
    Chat session for conversation history.

    Groups related chat messages together.
    """
    __tablename__ = "chat_sessions"

    title: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Foreign keys
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_sessions")
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )
    llm_override: Mapped[Optional["ChatSessionLLMOverride"]] = relationship(
        "ChatSessionLLMOverride",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id='{self.id}', title='{self.title}')>"


class ChatMessage(Base, UUIDMixin):
    """
    Individual chat message.

    Stores user queries and assistant responses with source citations.
    """
    __tablename__ = "chat_messages"

    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Source citations
    source_document_ids: Mapped[Optional[List[uuid.UUID]]] = mapped_column(UUIDArrayType())
    source_chunks: Mapped[Optional[dict]] = mapped_column(JSONType())

    # Feedback
    is_helpful: Mapped[Optional[bool]] = mapped_column(Boolean)
    feedback: Mapped[Optional[str]] = mapped_column(Text)

    # LLM info
    model_used: Mapped[Optional[str]] = mapped_column(String(100))
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Foreign keys
    session_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage(role='{self.role}', content='{self.content[:50]}...')>"


class AuditLog(Base, UUIDMixin):
    """
    Audit log for tracking user actions.

    Records all sensitive operations for security and compliance.
    """
    __tablename__ = "audit_log"

    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50))
    resource_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID())
    details: Mapped[Optional[dict]] = mapped_column(JSONType())
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 compatible
    user_agent: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Foreign keys
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id"),
        index=True,
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")

    def __repr__(self) -> str:
        return f"<AuditLog(action='{self.action}', user_id='{self.user_id}')>"


class ProcessingQueue(Base, UUIDMixin):
    """
    Document processing queue.

    Tracks documents waiting to be processed by Ray workers.
    """
    __tablename__ = "processing_queue"

    priority: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus),
        default=ProcessingStatus.PENDING,
        index=True,
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Foreign keys
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="processing_queue")

    # Composite index for queue ordering
    __table_args__ = (
        Index("idx_queue_priority_created", priority.desc(), created_at.asc()),
    )

    def __repr__(self) -> str:
        return f"<ProcessingQueue(document_id='{self.document_id}', status='{self.status}')>"


class SystemSettings(Base, UUIDMixin, TimestampMixin):
    """
    System-wide configuration settings.

    Stores key-value pairs for application configuration.
    Settings are organized by category for easier management.
    """
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    value: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="general", index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    value_type: Mapped[str] = mapped_column(String(20), default="string")  # string, number, boolean, json

    def __repr__(self) -> str:
        return f"<SystemSettings(key='{self.key}', category='{self.category}')>"


class LLMProvider(Base, UUIDMixin, TimestampMixin):
    """
    LLM provider configurations.

    Stores configuration for multiple LLM providers (OpenAI, Anthropic, Ollama, etc.)
    allowing dynamic switching between providers at runtime.
    """
    __tablename__ = "llm_providers"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    provider_type: Mapped[str] = mapped_column(String(50), nullable=False)  # openai, anthropic, ollama, azure, google, groq, together, cohere, custom
    api_key_encrypted: Mapped[Optional[str]] = mapped_column(Text)  # encrypted API key
    api_base_url: Mapped[Optional[str]] = mapped_column(String(500))  # for custom/self-hosted endpoints
    organization_id: Mapped[Optional[str]] = mapped_column(String(100))  # for OpenAI org
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # can be disabled without deleting
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)  # only one should be default
    default_chat_model: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., gpt-4o, claude-3-5-sonnet
    default_embedding_model: Mapped[Optional[str]] = mapped_column(String(100))  # e.g., text-embedding-3-small
    settings: Mapped[Optional[dict]] = mapped_column(JSONType)  # temperature, max_tokens, etc.

    __table_args__ = (
        Index("idx_llm_providers_default", is_default),
        Index("idx_llm_providers_active", is_active),
    )

    def __repr__(self) -> str:
        return f"<LLMProvider(name='{self.name}', type='{self.provider_type}', default={self.is_default})>"


class DatabaseConnection(Base, UUIDMixin, TimestampMixin):
    """
    Database connection configurations.

    Stores configuration for multiple database connections (SQLite, PostgreSQL, MySQL)
    allowing switching between databases at runtime.
    """
    __tablename__ = "database_connections"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    db_type: Mapped[str] = mapped_column(String(20), nullable=False)  # sqlite, postgresql, mysql
    host: Mapped[Optional[str]] = mapped_column(String(255))  # for postgresql/mysql
    port: Mapped[Optional[int]] = mapped_column(Integer)  # for postgresql/mysql
    database: Mapped[str] = mapped_column(String(255), nullable=False)  # database name or file path
    username: Mapped[Optional[str]] = mapped_column(String(100))  # for postgresql/mysql
    password_encrypted: Mapped[Optional[str]] = mapped_column(Text)  # encrypted password
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)  # currently active connection
    vector_store: Mapped[str] = mapped_column(String(20), default="auto")  # auto, pgvector, chromadb
    connection_options: Mapped[Optional[dict]] = mapped_column(JSONType)  # pool_size, max_overflow, etc.

    __table_args__ = (
        Index("idx_database_connections_active", is_active),
    )

    def __repr__(self) -> str:
        return f"<DatabaseConnection(name='{self.name}', type='{self.db_type}', active={self.is_active})>"


class LLMOperationConfig(Base, UUIDMixin, TimestampMixin):
    """
    Operation-level LLM configuration.

    Allows assigning different LLM providers to different operations
    (chat, embeddings, document processing, RAG, etc.)
    """
    __tablename__ = "llm_operation_configs"

    operation_type: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    # Operation types: chat, embeddings, document_processing, rag, summarization

    # Primary provider for this operation
    provider_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="SET NULL"),
    )
    model_override: Mapped[Optional[str]] = mapped_column(String(100))  # Override provider's default model
    temperature_override: Mapped[Optional[float]] = mapped_column(Float)
    max_tokens_override: Mapped[Optional[int]] = mapped_column(Integer)

    # Fallback provider if primary fails
    fallback_provider_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="SET NULL"),
    )

    # Relationships
    provider: Mapped[Optional["LLMProvider"]] = relationship(
        "LLMProvider",
        foreign_keys=[provider_id],
    )
    fallback_provider: Mapped[Optional["LLMProvider"]] = relationship(
        "LLMProvider",
        foreign_keys=[fallback_provider_id],
    )

    __table_args__ = (
        Index("idx_llm_operation_configs_type", "operation_type"),
    )

    def __repr__(self) -> str:
        return f"<LLMOperationConfig(operation='{self.operation_type}', provider_id='{self.provider_id}')>"


class ChatSessionLLMOverride(Base, UUIDMixin, TimestampMixin):
    """
    Per-session LLM configuration override.

    Allows users to select a specific LLM provider for individual chat sessions,
    overriding the default operation-level configuration.
    """
    __tablename__ = "chat_session_llm_overrides"

    session_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    provider_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_override: Mapped[Optional[str]] = mapped_column(String(100))
    temperature_override: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        back_populates="llm_override",
    )
    provider: Mapped["LLMProvider"] = relationship("LLMProvider")

    def __repr__(self) -> str:
        return f"<ChatSessionLLMOverride(session='{self.session_id}', provider='{self.provider_id}')>"


class LLMUsageLog(Base, UUIDMixin, TimestampMixin):
    """
    LLM usage tracking for analytics and cost estimation.

    Records every LLM call with token counts, costs, and performance metrics.
    """
    __tablename__ = "llm_usage_logs"

    # Provider info
    provider_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="SET NULL"),
    )
    provider_type: Mapped[str] = mapped_column(String(50), nullable=False)  # openai, anthropic, etc.
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    operation_type: Mapped[str] = mapped_column(String(50), nullable=False)  # chat, embeddings, etc.

    # User context
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID())  # Chat session if applicable

    # Token usage
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Cost calculation (in USD)
    input_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    output_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float)

    # Request metadata
    request_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    provider: Mapped[Optional["LLMProvider"]] = relationship("LLMProvider")
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_llm_usage_logs_provider", "provider_id"),
        Index("idx_llm_usage_logs_user", "user_id"),
        Index("idx_llm_usage_logs_operation", "operation_type"),
        Index("idx_llm_usage_logs_created", "created_at"),
        Index("idx_llm_usage_logs_provider_date", "provider_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<LLMUsageLog(model='{self.model}', operation='{self.operation_type}', tokens={self.total_tokens})>"


# =============================================================================
# Rate Limiting & Cost Control Models
# =============================================================================

class RateLimitConfig(Base, UUIDMixin, TimestampMixin):
    """
    Rate limiting configuration per access tier.

    Defines request limits for different operations based on user's access tier.
    """
    __tablename__ = "rate_limit_configs"

    # Tier association
    tier_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("access_tiers.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Request limits
    requests_per_minute: Mapped[int] = mapped_column(Integer, default=60)
    requests_per_hour: Mapped[int] = mapped_column(Integer, default=1000)
    requests_per_day: Mapped[int] = mapped_column(Integer, default=10000)

    # Token limits
    tokens_per_minute: Mapped[int] = mapped_column(Integer, default=100000)
    tokens_per_day: Mapped[int] = mapped_column(Integer, default=1000000)

    # Operation-specific limits (JSON: {"chat": 100, "embeddings": 500})
    operation_limits: Mapped[Optional[dict]] = mapped_column(JSONType())

    # Relationships
    tier: Mapped["AccessTier"] = relationship("AccessTier")

    def __repr__(self) -> str:
        return f"<RateLimitConfig(tier_id='{self.tier_id}', rpm={self.requests_per_minute})>"


class UserCostLimit(Base, UUIDMixin, TimestampMixin):
    """
    Per-user cost limits and tracking.

    Enforces spending caps and tracks current usage for cost control.
    """
    __tablename__ = "user_cost_limits"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Spending limits (in USD)
    daily_limit_usd: Mapped[float] = mapped_column(Float, default=10.0)
    monthly_limit_usd: Mapped[float] = mapped_column(Float, default=100.0)

    # Current usage tracking
    current_daily_usage_usd: Mapped[float] = mapped_column(Float, default=0.0)
    current_monthly_usage_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Reset timestamps
    last_daily_reset: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    last_monthly_reset: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Soft limit (warn) vs hard limit (block)
    enforce_hard_limit: Mapped[bool] = mapped_column(Boolean, default=True)

    # Alert thresholds (e.g., [50, 80, 100] for 50%, 80%, 100%)
    alert_thresholds: Mapped[Optional[list]] = mapped_column(JSONType(), default=[80, 100])

    # Relationships
    user: Mapped["User"] = relationship("User")
    alerts: Mapped[List["CostAlert"]] = relationship("CostAlert", back_populates="cost_limit")

    def __repr__(self) -> str:
        return f"<UserCostLimit(user_id='{self.user_id}', daily=${self.daily_limit_usd}, monthly=${self.monthly_limit_usd})>"


class CostAlertType(PyEnum):
    """Types of cost alerts."""
    THRESHOLD_WARNING = "threshold_warning"  # Approaching limit
    LIMIT_REACHED = "limit_reached"  # Hit the limit
    DAILY_RESET = "daily_reset"  # Daily usage reset
    MONTHLY_RESET = "monthly_reset"  # Monthly usage reset


class CostAlert(Base, UUIDMixin, TimestampMixin):
    """
    Cost alert records for notifications.

    Tracks when users approach or exceed their spending limits.
    """
    __tablename__ = "cost_alerts"

    cost_limit_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("user_cost_limits.id", ondelete="CASCADE"),
        nullable=False,
    )

    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    threshold_percent: Mapped[int] = mapped_column(Integer)  # e.g., 80, 100
    usage_at_alert_usd: Mapped[float] = mapped_column(Float)
    limit_usd: Mapped[float] = mapped_column(Float)

    # Notification status
    notified: Mapped[bool] = mapped_column(Boolean, default=False)
    notified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    cost_limit: Mapped["UserCostLimit"] = relationship("UserCostLimit", back_populates="alerts")

    __table_args__ = (
        Index("idx_cost_alerts_limit", "cost_limit_id"),
        Index("idx_cost_alerts_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<CostAlert(type='{self.alert_type}', threshold={self.threshold_percent}%)>"


class ProviderHealthStatus(PyEnum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProviderHealthLog(Base, UUIDMixin, TimestampMixin):
    """
    Provider health check logs.

    Records health check results for LLM providers for monitoring and failover.
    """
    __tablename__ = "provider_health_logs"

    provider_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Health status
    status: Mapped[str] = mapped_column(
        String(20),
        default=ProviderHealthStatus.UNKNOWN.value,
    )
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True)

    # Performance metrics
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    response_time_p50_ms: Mapped[Optional[int]] = mapped_column(Integer)
    response_time_p99_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_code: Mapped[Optional[str]] = mapped_column(String(50))
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)

    # Check metadata
    check_type: Mapped[str] = mapped_column(String(50), default="ping")  # ping, completion, embedding
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    provider: Mapped["LLMProvider"] = relationship("LLMProvider")

    __table_args__ = (
        Index("idx_provider_health_provider", "provider_id"),
        Index("idx_provider_health_checked", "checked_at"),
        Index("idx_provider_health_status", "provider_id", "is_healthy"),
    )

    def __repr__(self) -> str:
        return f"<ProviderHealthLog(provider='{self.provider_id}', healthy={self.is_healthy}, latency={self.latency_ms}ms)>"


class ProviderHealthCache(Base, UUIDMixin, TimestampMixin):
    """
    Cached current health status for providers.

    Single row per provider with latest health status for fast lookups.
    """
    __tablename__ = "provider_health_cache"

    provider_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Current status
    status: Mapped[str] = mapped_column(
        String(20),
        default=ProviderHealthStatus.UNKNOWN.value,
    )
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True)

    # Latest metrics
    last_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    avg_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)  # Rolling average

    # Failure tracking
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    last_failure_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Circuit breaker
    circuit_open: Mapped[bool] = mapped_column(Boolean, default=False)
    circuit_open_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Check metadata
    last_check_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    provider: Mapped["LLMProvider"] = relationship("LLMProvider")

    def __repr__(self) -> str:
        return f"<ProviderHealthCache(provider='{self.provider_id}', status='{self.status}')>"


# =============================================================================
# Response Caching & Prompt Templates (Phase 7)
# =============================================================================

class ResponseCache(Base, UUIDMixin, TimestampMixin):
    """
    LLM response cache for cost reduction.

    Caches identical prompt/model/temperature combinations to avoid
    redundant API calls.
    """
    __tablename__ = "response_cache"

    # Cache key components
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    system_prompt_hash: Mapped[Optional[str]] = mapped_column(String(64))

    # Cached response
    response_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Token info from original request
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    original_cost_usd: Mapped[Optional[float]] = mapped_column(Float)

    # Cache metadata
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Provider info for cache invalidation
    provider_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="SET NULL"),
    )

    __table_args__ = (
        Index("idx_response_cache_lookup", "prompt_hash", "model_id", "temperature"),
        Index("idx_response_cache_expires", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<ResponseCache(hash='{self.prompt_hash[:8]}...', model='{self.model_id}', hits={self.hit_count})>"


class CacheSettings(Base, UUIDMixin, TimestampMixin):
    """
    Global cache configuration settings.
    """
    __tablename__ = "cache_settings"

    # Cache behavior
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    default_ttl_seconds: Mapped[int] = mapped_column(Integer, default=86400)  # 24 hours
    max_cache_size_mb: Mapped[int] = mapped_column(Integer, default=1000)  # 1GB

    # Cache by temperature ranges (temperature <= threshold uses cache)
    cache_temperature_threshold: Mapped[float] = mapped_column(Float, default=0.3)

    # Model-specific settings (JSON: {"gpt-4": {"ttl": 3600}, "claude-3": {"ttl": 7200}})
    model_settings: Mapped[Optional[dict]] = mapped_column(JSONType())

    # Exclusions (operation types that should never be cached)
    excluded_operations: Mapped[Optional[list]] = mapped_column(JSONType(), default=[])

    def __repr__(self) -> str:
        return f"<CacheSettings(enabled={self.is_enabled}, ttl={self.default_ttl_seconds}s)>"


class PromptTemplate(Base, UUIDMixin, TimestampMixin):
    """
    Reusable prompt templates.

    Users can save prompts with predefined settings for quick reuse.
    """
    __tablename__ = "prompt_templates"

    # Ownership
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
    )  # NULL = system template

    # Template info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), default="general", index=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())

    # Prompt content
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)

    # Optional LLM settings
    model_id: Mapped[Optional[str]] = mapped_column(String(100))
    temperature: Mapped[Optional[float]] = mapped_column(Float)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer)

    # Visibility
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)  # Built-in templates

    # Usage stats
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Template variables (JSON: [{"name": "topic", "description": "Main topic", "default": ""}])
    variables: Mapped[Optional[list]] = mapped_column(JSONType())

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_prompt_templates_user", "user_id"),
        Index("idx_prompt_templates_public", "is_public"),
        Index("idx_prompt_templates_category", "category"),
    )

    def __repr__(self) -> str:
        return f"<PromptTemplate(name='{self.name}', category='{self.category}', public={self.is_public})>"


# =============================================================================
# Indexes (defined after models)
# =============================================================================

# Additional indexes for common queries
Index("idx_documents_created_at", Document.created_at.desc())
Index("idx_chunks_document_access", Chunk.document_id, Chunk.access_tier_id)
Index("idx_audit_user_action", AuditLog.user_id, AuditLog.action)
