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


class UploadStatus(str, PyEnum):
    """Upload job status for tracking file uploads before document creation."""
    QUEUED = "queued"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DUPLICATE = "duplicate"


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

    # Enhanced metadata from LLM analysis (for improved RAG search)
    # Contains: summary_short, summary_detailed, keywords, topics, entities,
    # hypothetical_questions, document_type, enhanced_at, model_used
    enhanced_metadata: Mapped[Optional[dict]] = mapped_column(JSONType())

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
    folder_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Relationships
    access_tier: Mapped["AccessTier"] = relationship("AccessTier", back_populates="documents")
    uploaded_by: Mapped[Optional["User"]] = relationship("User", back_populates="documents")
    folder: Mapped[Optional["Folder"]] = relationship(
        "Folder",
        back_populates="documents",
        foreign_keys=[folder_id],
    )
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

    Supports hierarchical chunking:
    - is_summary: True for document/section summaries
    - chunk_level: 0=detail, 1=section summary, 2=document summary
    - parent_chunk_id: Reference to parent chunk in hierarchy
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

    # Hierarchical chunking support (for large document optimization)
    is_summary: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    chunk_level: Mapped[int] = mapped_column(Integer, default=0)  # 0=detail, 1=section, 2=document
    parent_chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )

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
    parent_chunk: Mapped[Optional["Chunk"]] = relationship(
        "Chunk",
        remote_side="Chunk.id",
        backref="child_chunks",
    )

    def __repr__(self) -> str:
        return f"<Chunk(document_id='{self.document_id}', index={self.chunk_index}, level={self.chunk_level})>"


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


class ChatFeedback(Base, UUIDMixin, TimestampMixin):
    """
    Store user feedback on chat responses for quality improvement.

    Links to agent trajectories when feedback is for agent-mode responses,
    enabling the prompt optimization system to learn from user preferences.
    """
    __tablename__ = "chat_feedback"

    # Message identification
    message_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Feedback data
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5 scale
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Context
    mode: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # chat, general, agent

    # Link to agent trajectory (for agent mode responses)
    trajectory_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("agent_trajectories.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User")
    session: Mapped[Optional["ChatSession"]] = relationship("ChatSession")
    trajectory: Mapped[Optional["AgentTrajectory"]] = relationship("AgentTrajectory")

    __table_args__ = (
        Index("idx_chat_feedback_message", "message_id"),
        Index("idx_chat_feedback_user", "user_id"),
        Index("idx_chat_feedback_rating", "rating"),
        Index("idx_chat_feedback_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ChatFeedback(message_id='{self.message_id}', rating={self.rating})>"


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


class UploadJob(Base, UUIDMixin, TimestampMixin):
    """
    Upload job tracking.

    Persists file upload status to database so it survives server restarts.
    This model exists independently of Document - the document_id is only
    set after the document is successfully created during processing.
    """
    __tablename__ = "upload_jobs"

    # File info
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Processing status
    status: Mapped[UploadStatus] = mapped_column(
        Enum(UploadStatus),
        default=UploadStatus.QUEUED,
        index=True,
    )
    progress: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[str] = mapped_column(String(100), default="Queued")
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Processing options
    collection: Mapped[Optional[str]] = mapped_column(String(200))
    access_tier: Mapped[int] = mapped_column(Integer, default=1)
    enable_ocr: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_image_analysis: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_generate_tags: Mapped[bool] = mapped_column(Boolean, default=False)

    # Results (populated after processing)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer)
    word_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Link to created document (null until processing completes)
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    document: Mapped[Optional["Document"]] = relationship("Document")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_upload_jobs_status_created", status, "created_at"),
        Index("idx_upload_jobs_file_hash", file_hash),
    )

    def __repr__(self) -> str:
        return f"<UploadJob(id='{self.id}', filename='{self.filename}', status='{self.status}')>"


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
    redundant API calls. Supports both exact-match and semantic similarity caching.
    """
    __tablename__ = "response_cache"

    # Cache key components
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    system_prompt_hash: Mapped[Optional[str]] = mapped_column(String(64))

    # Semantic caching: store query embedding for similarity matching
    # Uses pgvector for efficient similarity search (fallback to Text for SQLite)
    query_embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536) if HAS_PGVECTOR else Text,
        nullable=True,
    )
    # Original query text (for debugging/display)
    query_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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

    # Semantic cache settings (optional - OFF by default)
    enable_semantic_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    semantic_similarity_threshold: Mapped[float] = mapped_column(Float, default=0.95)
    max_semantic_cache_entries: Mapped[int] = mapped_column(Integer, default=10000)

    # Model-specific settings (JSON: {"gpt-4": {"ttl": 3600}, "claude-3": {"ttl": 7200}})
    model_settings: Mapped[Optional[dict]] = mapped_column(JSONType())

    # Exclusions (operation types that should never be cached)
    excluded_operations: Mapped[Optional[list]] = mapped_column(JSONType(), default=[])

    def __repr__(self) -> str:
        return f"<CacheSettings(enabled={self.is_enabled}, semantic={self.enable_semantic_cache}, ttl={self.default_ttl_seconds}s)>"


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
# Agent System Models
# =============================================================================

class AgentType(str, PyEnum):
    """Agent type classification."""
    MANAGER = "manager"
    GENERATOR = "generator"
    CRITIC = "critic"
    RESEARCH = "research"
    TOOL_EXECUTOR = "tool_executor"


class ExecutionMode(str, PyEnum):
    """Execution mode for chat/agent routing."""
    AGENT = "agent"
    CHAT = "chat"
    GENERAL = "general"  # General LLM chat without RAG


class AgentDefinition(Base, UUIDMixin, TimestampMixin):
    """
    Definition of an agent in the multi-agent system.

    Each agent has a type, associated prompt versions, and LLM configuration.
    """
    __tablename__ = "agent_definitions"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Current active prompt version
    active_prompt_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID())

    # Default LLM configuration (can be overridden per agent)
    default_provider_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("llm_providers.id", ondelete="SET NULL"),
    )
    default_model: Mapped[Optional[str]] = mapped_column(String(100))
    default_temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer)

    # Agent capabilities and settings
    settings: Mapped[Optional[dict]] = mapped_column(JSONType())
    # Example: {"tools": ["search", "generate"], "max_iterations": 5}

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Performance metrics (updated after executions)
    total_executions: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    avg_tokens_per_execution: Mapped[Optional[int]] = mapped_column(Integer)

    # Relationships
    default_provider: Mapped[Optional["LLMProvider"]] = relationship("LLMProvider")

    __table_args__ = (
        Index("idx_agent_definitions_type", "agent_type"),
        Index("idx_agent_definitions_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<AgentDefinition(name='{self.name}', type='{self.agent_type}')>"


class AgentPromptVersion(Base, UUIDMixin, TimestampMixin):
    """
    Version history for agent prompts.

    Enables prompt optimization, A/B testing, and rollback capability.
    """
    __tablename__ = "agent_prompt_versions"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("agent_definitions.id", ondelete="CASCADE"),
        nullable=False,
    )

    version_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Prompt content
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    task_prompt_template: Mapped[str] = mapped_column(Text, nullable=False)
    # Template uses {{variable}} syntax for substitution

    # Optional: Few-shot examples and output schema
    few_shot_examples: Mapped[Optional[list]] = mapped_column(JSONType())
    # Example: [{"input": "...", "output": "..."}, ...]
    output_schema: Mapped[Optional[dict]] = mapped_column(JSONType())
    # JSON Schema for expected output format

    # Version metadata
    change_reason: Mapped[Optional[str]] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(50), default="system")
    # Values: "system", "manual", "prompt_builder", "rollback"

    # Performance metrics for this version
    execution_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_quality_score: Mapped[Optional[float]] = mapped_column(Float)

    # A/B testing support
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    traffic_percentage: Mapped[int] = mapped_column(Integer, default=0)
    # For A/B testing: 0-100% of traffic routed to this version

    # Parent version (for tracking lineage)
    parent_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("agent_prompt_versions.id", ondelete="SET NULL"),
    )

    # Relationships
    agent: Mapped["AgentDefinition"] = relationship("AgentDefinition")
    parent_version: Mapped[Optional["AgentPromptVersion"]] = relationship(
        "AgentPromptVersion", remote_side="AgentPromptVersion.id"
    )

    __table_args__ = (
        Index("idx_agent_prompt_versions_agent", "agent_id"),
        Index("idx_agent_prompt_versions_active", "agent_id", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<AgentPromptVersion(agent_id={self.agent_id}, v{self.version_number})>"


class AgentTrajectory(Base, UUIDMixin, TimestampMixin):
    """
    Records agent execution trajectories for analysis and self-improvement.

    Each trajectory captures the full execution path including reasoning,
    tool calls, and outcomes for later analysis by the Prompt Builder agent.
    """
    __tablename__ = "agent_trajectories"

    session_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("agent_definitions.id", ondelete="SET NULL"),
    )

    # Execution details
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    input_summary: Mapped[Optional[str]] = mapped_column(Text)
    # Hashed or summarized input for privacy

    # Full trajectory (list of steps)
    trajectory_steps: Mapped[dict] = mapped_column(JSONType(), nullable=False)
    # Structure: [
    #   {"step": 1, "action": "reasoning", "content": "...", "duration_ms": 100},
    #   {"step": 2, "action": "tool_call", "tool": "search", "result": "...", "duration_ms": 500},
    #   {"step": 3, "action": "llm_invoke", "tokens": 150, "duration_ms": 2000},
    #   {"step": 4, "action": "output", "content": "...", "duration_ms": 50}
    # ]

    # Outcome
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    # Score 0.0-5.0, can be set by CriticAgent or user feedback
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Resource usage
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[Optional[float]] = mapped_column(Float)

    # User feedback (optional)
    user_rating: Mapped[Optional[int]] = mapped_column(Integer)
    # Rating 1-5 stars
    user_feedback: Mapped[Optional[str]] = mapped_column(Text)

    # Prompt version used (for A/B testing analysis)
    prompt_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("agent_prompt_versions.id", ondelete="SET NULL"),
    )

    # Relationships
    agent: Mapped[Optional["AgentDefinition"]] = relationship("AgentDefinition")
    prompt_version: Mapped[Optional["AgentPromptVersion"]] = relationship("AgentPromptVersion")

    __table_args__ = (
        Index("idx_trajectory_agent_success", "agent_id", "success"),
        Index("idx_trajectory_created", "created_at"),
        Index("idx_trajectory_session", "session_id"),
    )

    def __repr__(self) -> str:
        return f"<AgentTrajectory(agent_id={self.agent_id}, success={self.success})>"


class AgentExecutionPlan(Base, UUIDMixin, TimestampMixin):
    """
    Persistent storage for multi-step execution plans.

    Enables recovery from failures, auditing, and cost tracking.
    """
    __tablename__ = "agent_execution_plans"

    session_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
    )

    # Original request
    user_request: Mapped[str] = mapped_column(Text, nullable=False)

    # Plan structure
    plan_steps: Mapped[dict] = mapped_column(JSONType(), nullable=False)
    # Structure: [
    #   {"step": 1, "agent": "research", "task": "Find relevant docs", "status": "completed", "dependencies": []},
    #   {"step": 2, "agent": "generator", "task": "Draft content", "status": "in_progress", "dependencies": [1]},
    #   {"step": 3, "agent": "critic", "task": "Review quality", "status": "pending", "dependencies": [2]}
    # ]

    # Execution state
    status: Mapped[str] = mapped_column(String(50), default="pending")
    # Values: "pending", "executing", "paused", "completed", "failed", "cancelled"
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    completed_steps: Mapped[int] = mapped_column(Integer, default=0)

    # Cost tracking
    estimated_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    actual_cost_usd: Mapped[Optional[float]] = mapped_column(Float)
    user_approved_cost: Mapped[bool] = mapped_column(Boolean, default=False)
    # True if user approved execution after seeing cost estimate

    # Results
    final_output: Mapped[Optional[str]] = mapped_column(Text)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_execution_plan_status", "status"),
        Index("idx_execution_plan_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<AgentExecutionPlan(session_id={self.session_id}, status='{self.status}')>"


class PromptOptimizationJob(Base, UUIDMixin, TimestampMixin):
    """
    Tracks prompt optimization jobs run by the Prompt Builder Agent.

    Records the analysis, variants generated, A/B test results, and approval status.
    """
    __tablename__ = "prompt_optimization_jobs"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("agent_definitions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Job status
    status: Mapped[str] = mapped_column(String(50), default="pending")
    # Values: "pending", "analyzing", "generating", "testing", "awaiting_approval", "completed", "rejected", "failed"

    # Analysis results
    analysis_window_hours: Mapped[int] = mapped_column(Integer, default=24)
    trajectories_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    failure_patterns: Mapped[Optional[dict]] = mapped_column(JSONType())
    # Structure: {"patterns": [{"description": "...", "count": 5, "examples": [...]}]}

    # Generated variants
    variants_generated: Mapped[int] = mapped_column(Integer, default=0)
    variant_ids: Mapped[Optional[list]] = mapped_column(JSONType())
    # List of AgentPromptVersion IDs generated

    # A/B test results
    test_start_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    test_end_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    winning_variant_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID())

    # Metrics improvement
    baseline_success_rate: Mapped[Optional[float]] = mapped_column(Float)
    new_success_rate: Mapped[Optional[float]] = mapped_column(Float)
    improvement_percentage: Mapped[Optional[float]] = mapped_column(Float)

    # Approval tracking
    approved_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    agent: Mapped["AgentDefinition"] = relationship("AgentDefinition")
    approver: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_optimization_job_agent", "agent_id"),
        Index("idx_optimization_job_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<PromptOptimizationJob(agent_id={self.agent_id}, status='{self.status}')>"


class ExecutionModePreference(Base, UUIDMixin, TimestampMixin):
    """
    User preferences for execution mode (agent vs chat).

    Controls auto-detection, cost approval thresholds, and agent mode toggle.
    """
    __tablename__ = "execution_mode_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Default mode for new sessions
    default_mode: Mapped[str] = mapped_column(String(20), default="agent")
    # Values: "agent", "chat"

    # Toggle to completely disable agent mode
    agent_mode_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Auto-detection settings
    auto_detect_complexity: Mapped[bool] = mapped_column(Boolean, default=True)
    # If True, automatically switch to agent mode for complex requests

    # Cost management
    show_cost_estimation: Mapped[bool] = mapped_column(Boolean, default=True)
    require_approval_above_usd: Mapped[float] = mapped_column(Float, default=1.0)
    # Ask user for approval if estimated cost exceeds this threshold

    # General chat mode (non-RAG)
    general_chat_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # If True, allows general LLM responses without document search
    fallback_to_general: Mapped[bool] = mapped_column(Boolean, default=True)
    # If True, automatically use general chat when no relevant documents found

    # Relationships
    user: Mapped["User"] = relationship("User")

    def __repr__(self) -> str:
        return f"<ExecutionModePreference(user_id={self.user_id}, mode='{self.default_mode}')>"


class UserPreferences(Base, UUIDMixin, TimestampMixin):
    """
    User UI and application preferences.

    Stores per-user settings for:
    - UI theme and display preferences
    - Default filters and search settings
    - Document view preferences
    """
    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # UI Theme
    theme: Mapped[str] = mapped_column(String(20), default="system")
    # Values: "light", "dark", "system"

    # Document List View
    documents_view_mode: Mapped[str] = mapped_column(String(20), default="grid")
    # Values: "grid", "list", "table"

    documents_sort_by: Mapped[str] = mapped_column(String(30), default="created_at")
    # Values: "created_at", "name", "file_size", "updated_at"

    documents_sort_order: Mapped[str] = mapped_column(String(4), default="desc")
    # Values: "asc", "desc"

    documents_page_size: Mapped[int] = mapped_column(Integer, default=20)
    # Number of documents per page

    # Default Filters
    default_collection: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # Default collection/tag filter

    default_folder_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Default folder to show

    # Search Preferences
    search_include_content: Mapped[bool] = mapped_column(Boolean, default=True)
    # Include document content in search

    search_results_per_page: Mapped[int] = mapped_column(Integer, default=10)

    # Chat/RAG Preferences
    chat_show_sources: Mapped[bool] = mapped_column(Boolean, default=True)
    # Show source documents in chat responses

    chat_expand_sources: Mapped[bool] = mapped_column(Boolean, default=False)
    # Auto-expand source references

    # Sidebar State
    sidebar_collapsed: Mapped[bool] = mapped_column(Boolean, default=False)
    # Remember sidebar collapsed state

    # Recent Items (stored as JSON)
    recent_documents: Mapped[Optional[List[str]]] = mapped_column(
        StringArrayType(),
        nullable=True,
    )
    # List of recently viewed document IDs (last 10)

    recent_searches: Mapped[Optional[List[str]]] = mapped_column(
        StringArrayType(),
        nullable=True,
    )
    # List of recent search queries (last 10)

    # Saved searches (named searches with filters)
    # Format: [{"name": "...", "query": "...", "filters": {...}, "created_at": "..."}, ...]
    saved_searches: Mapped[Optional[list]] = mapped_column(
        JSONType(),
        nullable=True,
    )
    # List of saved search configurations (max 20)

    # Relationships
    user: Mapped["User"] = relationship("User")
    default_folder: Mapped[Optional["Folder"]] = relationship("Folder")

    __table_args__ = (
        Index("idx_user_preferences_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<UserPreferences(user_id={self.user_id}, theme='{self.theme}')>"


# =============================================================================
# GraphRAG Models - Knowledge Graph for Multi-Hop Reasoning
# =============================================================================

class EntityType(str, PyEnum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    DATE = "date"
    METRIC = "metric"
    OTHER = "other"


class RelationType(str, PyEnum):
    """Types of relationships between entities."""
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    USES = "uses"
    MENTIONS = "mentions"
    BEFORE = "before"
    AFTER = "after"
    CAUSES = "causes"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    OTHER = "other"


class Entity(Base, UUIDMixin):
    """
    Knowledge graph entity extracted from documents.

    Represents a named entity (person, org, concept, etc.) that appears
    in one or more documents. Entities form nodes in the knowledge graph.
    """
    __tablename__ = "entities"

    # Entity identification
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    name_normalized: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType),
        nullable=False,
        index=True,
    )

    # Description and context
    description: Mapped[Optional[str]] = mapped_column(Text)
    aliases: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())

    # Embedding for semantic search
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(1536) if HAS_PGVECTOR else Text,
        nullable=True,
    )

    # Metadata
    properties: Mapped[Optional[dict]] = mapped_column(JSONType())
    mention_count: Mapped[int] = mapped_column(Integer, default=1)

    # Timestamps
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

    # Relationships
    source_mentions: Mapped[List["EntityMention"]] = relationship(
        "EntityMention",
        back_populates="entity",
        cascade="all, delete-orphan",
    )
    outgoing_relations: Mapped[List["EntityRelation"]] = relationship(
        "EntityRelation",
        foreign_keys="EntityRelation.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    incoming_relations: Mapped[List["EntityRelation"]] = relationship(
        "EntityRelation",
        foreign_keys="EntityRelation.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Entity(name='{self.name}', type='{self.entity_type}')>"


class EntityMention(Base, UUIDMixin):
    """
    Records where an entity is mentioned in a document/chunk.

    Links entities to their source documents for provenance tracking.
    """
    __tablename__ = "entity_mentions"

    # Foreign keys
    entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Context
    context_snippet: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    # Position in document
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    char_offset: Mapped[Optional[int]] = mapped_column(Integer)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    entity: Mapped["Entity"] = relationship("Entity", back_populates="source_mentions")
    document: Mapped["Document"] = relationship("Document")
    chunk: Mapped[Optional["Chunk"]] = relationship("Chunk")

    def __repr__(self) -> str:
        return f"<EntityMention(entity_id='{self.entity_id}', document_id='{self.document_id}')>"


class EntityRelation(Base, UUIDMixin):
    """
    Relationship between two entities in the knowledge graph.

    Represents edges in the knowledge graph, enabling multi-hop reasoning.
    """
    __tablename__ = "entity_relations"

    # Source and target entities
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Relation properties
    relation_type: Mapped[RelationType] = mapped_column(
        Enum(RelationType),
        nullable=False,
        index=True,
    )
    relation_label: Mapped[Optional[str]] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Confidence and weight
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    # Source document for provenance
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Metadata
    properties: Mapped[Optional[dict]] = mapped_column(JSONType())

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    source_entity: Mapped["Entity"] = relationship(
        "Entity",
        foreign_keys=[source_entity_id],
        back_populates="outgoing_relations",
    )
    target_entity: Mapped["Entity"] = relationship(
        "Entity",
        foreign_keys=[target_entity_id],
        back_populates="incoming_relations",
    )
    document: Mapped[Optional["Document"]] = relationship("Document")

    def __repr__(self) -> str:
        return f"<EntityRelation('{self.source_entity_id}' --{self.relation_type}--> '{self.target_entity_id}')>"


# =============================================================================
# OCR Performance Metrics
# =============================================================================

class OCRMetrics(Base, UUIDMixin, TimestampMixin):
    """
    OCR performance and usage tracking.

    Records every OCR operation with performance metrics, accuracy indicators,
    and cost tracking for analytics and optimization.
    """
    __tablename__ = "ocr_metrics"

    # Provider info
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # paddleocr, tesseract, auto
    variant: Mapped[Optional[str]] = mapped_column(String(50))  # server, mobile (for PaddleOCR)
    language: Mapped[str] = mapped_column(String(10), nullable=False)  # en, de, fr, etc.

    # Document context
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Performance metrics
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)  # Time in milliseconds
    page_count: Mapped[int] = mapped_column(Integer, default=1)  # Number of pages processed
    character_count: Mapped[Optional[int]] = mapped_column(Integer)  # Characters extracted

    # Quality indicators
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)  # Average confidence (0-1)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)  # Whether fallback to Tesseract was used

    # Cost estimation (if applicable for cloud OCR providers)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float)

    # Additional data
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONType())  # Additional provider-specific data

    # Relationships
    document: Mapped[Optional["Document"]] = relationship("Document")
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_ocr_metrics_provider", "provider"),
        Index("idx_ocr_metrics_document", "document_id"),
        Index("idx_ocr_metrics_user", "user_id"),
        Index("idx_ocr_metrics_created", "created_at"),
        Index("idx_ocr_metrics_provider_date", "provider", "created_at"),
        Index("idx_ocr_metrics_success", "success", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<OCRMetrics(provider='{self.provider}', language='{self.language}', time={self.processing_time_ms}ms)>"


# =============================================================================
# Folder Model - Hierarchical Document Organization
# =============================================================================

class Folder(Base, UUIDMixin, TimestampMixin):
    """
    Hierarchical folder structure for organizing documents.

    Uses materialized path pattern for efficient subtree queries:
    - Root folder path: "/FolderName/"
    - Nested folder path: "/Parent/Child/Grandchild/"

    Permission model:
    - Each folder has an access_tier_id
    - If inherit_permissions=True, folder uses parent's effective tier
    - Admin users (tier 100) can access all folders
    """
    __tablename__ = "folders"

    # Folder name (displayed in UI)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Materialized path for efficient hierarchy queries
    # Format: "/folder1/folder2/" - always starts and ends with /
    path: Mapped[str] = mapped_column(String(2000), nullable=False, index=True)

    # Hierarchy relationships
    parent_folder_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("folders.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    depth: Mapped[int] = mapped_column(Integer, default=0)
    # depth=0 for root folders, depth=1 for first-level children, etc.

    # Permission settings
    access_tier_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("access_tiers.id"),
        nullable=False,
        index=True,
    )
    inherit_permissions: Mapped[bool] = mapped_column(Boolean, default=True)
    # If True, effective permission = parent's effective tier
    # If False, effective permission = this folder's access_tier_id

    # Ownership tracking
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    color: Mapped[Optional[str]] = mapped_column(String(7))  # Hex color e.g., "#1E3A5F"

    # Relationships
    parent_folder: Mapped[Optional["Folder"]] = relationship(
        "Folder",
        remote_side="Folder.id",
        backref="subfolders",
        foreign_keys=[parent_folder_id],
    )
    access_tier: Mapped["AccessTier"] = relationship("AccessTier")
    created_by: Mapped[Optional["User"]] = relationship("User")
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="folder",
        foreign_keys="Document.folder_id",
    )

    __table_args__ = (
        Index("idx_folders_path_prefix", "path", postgresql_using="btree"),
        Index("idx_folders_parent_name", "parent_folder_id", "name"),
        Index("idx_folders_access_tier", "access_tier_id"),
        Index("idx_folders_created_by", "created_by_id"),
    )

    def __repr__(self) -> str:
        return f"<Folder(name='{self.name}', path='{self.path}', depth={self.depth})>"


# =============================================================================
# Generation Template Model - Saved Document Generation Configurations
# =============================================================================

class TemplateCategory(str, PyEnum):
    """Categories for generation templates."""
    REPORT = "report"
    PROPOSAL = "proposal"
    PRESENTATION = "presentation"
    MEETING_NOTES = "meeting_notes"
    DOCUMENTATION = "documentation"
    CUSTOM = "custom"


class GenerationTemplate(Base, UUIDMixin, TimestampMixin):
    """
    Saved generation configuration templates.

    Allows users to save and reuse document generation settings including:
    - Output format, theme, fonts, layouts
    - Default collections for style learning
    - Custom colors and other preferences

    System templates (user_id=NULL) are pre-built and available to all users.
    """
    __tablename__ = "generation_templates"

    # Ownership (NULL = system template available to all)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Template identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(
        String(50),
        default=TemplateCategory.CUSTOM.value,
        index=True,
    )

    # Template thumbnail (base64 encoded PNG, optional)
    thumbnail: Mapped[Optional[str]] = mapped_column(Text)

    # Generation settings (JSON)
    # Contains: output_format, theme, font_family, layout_template,
    # include_toc, include_sources, use_existing_docs, enable_animations,
    # custom_colors, etc.
    settings: Mapped[dict] = mapped_column(JSONType(), nullable=False)

    # Default collections for "Learn from Existing Documents"
    default_collections: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())

    # Visibility settings
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    # Public templates are visible to all users but owned by creator

    is_system: Mapped[bool] = mapped_column(Boolean, default=False)
    # System templates are built-in and cannot be deleted

    # Usage tracking
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Tags for filtering/searching
    tags: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_generation_templates_user", "user_id"),
        Index("idx_generation_templates_category", "category"),
        Index("idx_generation_templates_public", "is_public"),
        Index("idx_generation_templates_system", "is_system"),
        Index("idx_generation_templates_use_count", "use_count"),
    )

    def __repr__(self) -> str:
        return f"<GenerationTemplate(name='{self.name}', category='{self.category}', system={self.is_system})>"


# =============================================================================
# Indexes (defined after models)
# =============================================================================

# Additional indexes for common queries
Index("idx_documents_created_at", Document.created_at.desc())
Index("idx_chunks_document_access", Chunk.document_id, Chunk.access_tier_id)
Index("idx_audit_user_action", AuditLog.user_id, AuditLog.action)

# Performance optimization indexes (added for common query patterns)
# Documents: access tier + created_at for filtered listing
Index("idx_documents_tier_created", Document.access_tier_id, Document.created_at.desc())
# Chunks: access tier + created_at for filtered search results
Index("idx_chunks_tier_created", Chunk.access_tier_id, Chunk.created_at.desc())
# ChatMessages: session + created_at for conversation history retrieval
Index("idx_chat_messages_session_created", ChatMessage.session_id, ChatMessage.created_at.asc())

# GraphRAG indexes for knowledge graph traversal
Index("idx_entity_name_type", Entity.name_normalized, Entity.entity_type)
Index("idx_entity_relation_source", EntityRelation.source_entity_id, EntityRelation.relation_type)
Index("idx_entity_relation_target", EntityRelation.target_entity_id, EntityRelation.relation_type)
Index("idx_entity_mention_doc", EntityMention.document_id, EntityMention.entity_id)
