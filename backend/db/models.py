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


# Determine embedding dimension based on provider
def get_embedding_dimension() -> int:
    """
    Get embedding dimension based on configured provider and model.

    Supports explicit dimension override via EMBEDDING_DIMENSION env var.

    Provider-specific dimensions:
    - OpenAI text-embedding-3-small: 1536 (default) or 512-1536 (flexible)
    - OpenAI text-embedding-3-large: 3072 (default) or 256-3072 (flexible)
    - OpenAI text-embedding-ada-002: 1536 (fixed)
    - Ollama nomic-embed-text: 768
    - Ollama mxbai-embed-large: 1024
    - HuggingFace all-MiniLM-L6-v2: 384
    - HuggingFace all-mpnet-base-v2: 768
    - Cohere embed-english-v3.0: 1024
    - Voyage voyage-2: 1024
    - Mistral mistral-embed: 1024

    Returns:
        Embedding dimension (default: 768 for local models)

    Example .env configurations:
        # Explicit dimension override (any provider)
        EMBEDDING_DIMENSION=512

        # OpenAI with reduced dimension (saves storage/cost)
        DEFAULT_LLM_PROVIDER=openai
        DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
        EMBEDDING_DIMENSION=512

        # Ollama with specific model
        DEFAULT_LLM_PROVIDER=ollama
        OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # 768D

        # HuggingFace with specific model
        DEFAULT_LLM_PROVIDER=huggingface
        DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2  # 384D
    """
    # Check for explicit dimension override first
    explicit_dim = os.getenv("EMBEDDING_DIMENSION")
    if explicit_dim:
        try:
            return int(explicit_dim)
        except ValueError:
            pass  # Fall through to provider detection

    provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()

    # Provider-specific defaults
    if provider == "openai":
        model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small").lower()
        if "text-embedding-3-large" in model:
            return 3072  # Can be reduced to 256-3072
        elif "text-embedding-3-small" in model:
            return 1536  # Can be reduced to 512-1536
        elif "ada-002" in model:
            return 1536  # Fixed dimension
        return 1536  # Safe default for OpenAI

    elif provider == "ollama":
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").lower()
        if "nomic" in model:
            return 768
        elif "mxbai" in model:
            return 1024
        elif "snowflake" in model:
            return 768
        return 768  # Safe default for most Ollama models

    elif provider == "huggingface":
        model = os.getenv("DEFAULT_EMBEDDING_MODEL", "").lower()
        if "minilm" in model or "l6" in model:
            return 384
        elif "mpnet" in model or "bge" in model:
            return 768
        return 768  # Safe default

    elif provider == "cohere":
        return 1024

    elif provider == "voyage":
        return 1024

    elif provider == "mistral":
        return 1024

    else:
        # Safe default for unknown providers (most local models use 768)
        return 768


# Cache the dimension to avoid repeated env lookups
EMBEDDING_DIMENSION = get_embedding_dimension()


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
    BASIC = "basic"           # Text extraction only (fastest)
    OCR_ENABLED = "ocr"       # Text + OCR for scanned documents
    FULL = "full"             # Text + OCR + AI image analysis (most thorough)


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


class OrganizationRole(str, PyEnum):
    """Roles within an organization."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


class Organization(Base, UUIDMixin, TimestampMixin):
    """
    Organization model for multi-tenant support.

    Organizations group users and resources together with
    shared settings and feature flags.
    """
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    plan: Mapped[Optional[str]] = mapped_column(String(50), default="free")
    settings: Mapped[Optional[dict]] = mapped_column(JSONType, default=dict)
    max_users: Mapped[int] = mapped_column(Integer, default=5)
    max_storage_gb: Mapped[int] = mapped_column(Integer, default=10)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    members: Mapped[List["OrganizationMember"]] = relationship(
        "OrganizationMember", back_populates="organization", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Organization(name='{self.name}', slug='{self.slug}')>"


class OrganizationMember(Base, UUIDMixin):
    """
    Organization membership model.

    Links users to organizations with a specific role.
    """
    __tablename__ = "organization_members"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[OrganizationRole] = mapped_column(
        Enum(OrganizationRole, native_enum=False, length=20),
        default=OrganizationRole.MEMBER,
    )
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="members")
    user: Mapped["User"] = relationship("User", back_populates="organization_memberships")

    __table_args__ = (
        Index("ix_org_members_org_user", "organization_id", "user_id", unique=True),
    )

    def __repr__(self) -> str:
        return f"<OrganizationMember(org={self.organization_id}, user={self.user_id}, role={self.role})>"


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

    # Superadmin flag - superadmins can access all organizations and have full system control
    is_superadmin: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Current organization context - which org the user is currently working in
    # Superadmins can switch this to any org, regular users are limited to their memberships
    current_organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Folder permission mode
    # If True, user ONLY sees explicitly granted folders (ignores tier-based access)
    # If False (default), user sees tier-based folders + explicitly granted folders (additive)
    use_folder_permissions_only: Mapped[bool] = mapped_column(Boolean, default=False)

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
    current_organization: Mapped[Optional["Organization"]] = relationship(
        "Organization", foreign_keys=[current_organization_id]
    )
    documents: Mapped[List["Document"]] = relationship("Document", back_populates="uploaded_by")
    chat_sessions: Mapped[List["ChatSession"]] = relationship("ChatSession", back_populates="user")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user")
    organization_memberships: Mapped[List["OrganizationMember"]] = relationship(
        "OrganizationMember", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(email='{self.email}', tier='{self.access_tier.name if self.access_tier else None}')>"


class Document(Base, UUIDMixin, TimestampMixin):
    """
    Document model representing an uploaded file.

    Stores metadata about documents and their processing status.
    The actual file content is stored in chunks.
    """
    __tablename__ = "documents"

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

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
        default=ProcessingMode.FULL,
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

    # Private document flag
    # When True, only the uploaded_by user or superadmins can access this document.
    # Private documents are deleted when the owner user is deleted.
    is_private: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

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

    # Knowledge Graph extraction status
    # Tracks whether entities have been extracted from this document
    kg_extraction_status: Mapped[Optional[str]] = mapped_column(
        String(20),
        default="pending",
        index=True,
    )  # pending, processing, completed, failed, skipped
    kg_extracted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    kg_entity_count: Mapped[int] = mapped_column(Integer, default=0)
    kg_relation_count: Mapped[int] = mapped_column(Integer, default=0)

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

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Primary embedding - using dynamic column type based on database
    # For PostgreSQL with pgvector, this will be a Vector type
    # For other databases, we'll store as JSON or binary
    # Dimension is auto-detected from DEFAULT_LLM_PROVIDER env var (768 for Ollama, 1536 for OpenAI)
    # This is the "primary" embedding used for fast queries (no joins required)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,  # Fallback to Text for non-pgvector
        nullable=True,
    )

    # Metadata for primary embedding (which provider/model/dimension it uses)
    embedding_provider: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Provider for primary embedding: ollama, openai, etc."
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Model for primary embedding: nomic-embed-text, text-embedding-3-small, etc."
    )
    embedding_dimension: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Dimension of primary embedding: 384, 768, 1536, 3072, etc."
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

    # Multi-embedding support: store embeddings from multiple providers
    # This enables instant provider switching without re-indexing
    # Will be populated via migration from models_multi_embedding.py
    # multi_embeddings: Mapped[List["ChunkEmbedding"]] = relationship(
    #     "ChunkEmbedding",
    #     back_populates="chunk",
    #     cascade="all, delete-orphan",
    # )

    def __repr__(self) -> str:
        return f"<Chunk(document_id='{self.document_id}', index={self.chunk_index}, level={self.chunk_level})>"


class ScrapedContent(Base, UUIDMixin):
    """
    Web scraped content.

    Stores content scraped from websites, optionally with embeddings
    for RAG integration.
    """
    __tablename__ = "scraped_content"

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Embedding (dimension auto-detected from provider: 768D for Ollama, 1536D for OpenAI)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,
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

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

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

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

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

    # Response quality metrics
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    confidence_level: Mapped[Optional[str]] = mapped_column(String(20))  # high, medium, low

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
    temperature_manual_override: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)

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
    # Dimension auto-detected from provider (768D for Ollama, 1536D for OpenAI)
    query_embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,
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

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

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

    # Embedding for semantic search (dimension auto-detected: 768D for Ollama, 1536D for OpenAI)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,
        nullable=True,
    )

    # Language support for cross-language entity linking
    entity_language: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True, index=True
    )  # Primary language: "en", "de", "ru", etc.
    canonical_name: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True, index=True
    )  # English canonical name for cross-language linking
    language_variants: Mapped[Optional[dict]] = mapped_column(JSONType())
    # Format: {"en": "Germany", "de": "Deutschland", "fr": "Allemagne"}

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

    # Language context for the mention
    mention_language: Mapped[Optional[str]] = mapped_column(String(10))
    # Language of the source document/chunk: "en", "de", "fr", etc.
    mention_script: Mapped[Optional[str]] = mapped_column(String(20))
    # Script type: "latin", "cyrillic", "cjk", "arabic", "devanagari", etc.

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

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

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


class KGExtractionJob(Base, UUIDMixin):
    """
    Background job for knowledge graph entity extraction.

    Tracks progress of entity extraction across multiple documents,
    allowing users to start a job, leave the page, and return to
    check progress or cancel.
    """
    __tablename__ = "kg_extraction_jobs"

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # User who started the job
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Job status: queued, running, paused, completed, failed, cancelled
    status: Mapped[str] = mapped_column(
        String(20),
        default="queued",
        index=True,
    )

    # Progress tracking
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    processed_documents: Mapped[int] = mapped_column(Integer, default=0)
    failed_documents: Mapped[int] = mapped_column(Integer, default=0)

    # Extracted counts
    total_entities: Mapped[int] = mapped_column(Integer, default=0)
    total_relations: Mapped[int] = mapped_column(Integer, default=0)

    # Current document being processed
    current_document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    current_document_name: Mapped[Optional[str]] = mapped_column(String(500))

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # For ETA calculation
    avg_doc_processing_time: Mapped[Optional[float]] = mapped_column(Float)

    # Error tracking (list of {doc_id: str, error: str})
    error_log: Mapped[Optional[dict]] = mapped_column(JSONType())

    # Job configuration
    only_new_documents: Mapped[bool] = mapped_column(Boolean, default=True)
    document_ids: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())

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
    user: Mapped["User"] = relationship("User")
    current_document: Mapped[Optional["Document"]] = relationship("Document")

    def __repr__(self) -> str:
        return f"<KGExtractionJob(status='{self.status}', progress={self.processed_documents}/{self.total_documents})>"

    def get_progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100

    def get_estimated_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time based on average processing time."""
        if not self.avg_doc_processing_time or self.status != "running":
            return None
        remaining = self.total_documents - self.processed_documents
        return remaining * self.avg_doc_processing_time


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

    # Organization scope (multi-tenant isolation)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

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
    tags: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())  # Folder tags for categorization

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
# Folder Permission Model - Per-User Folder Access Control
# =============================================================================

class FolderPermissionLevel(str, PyEnum):
    """Permission levels for folder access."""
    VIEW = "view"      # Can see folder and documents
    EDIT = "edit"      # Can upload/modify documents
    MANAGE = "manage"  # Can grant permissions to others


class FolderPermission(Base, UUIDMixin, TimestampMixin):
    """
    Per-user folder permissions for fine-grained access control.

    This provides an alternative/supplement to tier-based access:
    - Users with use_folder_permissions_only=False (default) see:
      tier-based folders + explicitly granted folders (additive)
    - Users with use_folder_permissions_only=True see:
      ONLY explicitly granted folders (restrictive)

    Permission levels:
    - VIEW: Can see the folder and read documents
    - EDIT: Can upload and modify documents in the folder
    - MANAGE: Can grant/revoke permissions to other users
    """
    __tablename__ = "folder_permissions"

    # Which folder this permission applies to
    folder_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("folders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Which user has this permission
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Permission level
    permission_level: Mapped[str] = mapped_column(
        String(20),
        default=FolderPermissionLevel.VIEW.value,
        nullable=False,
    )

    # Who granted this permission
    granted_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Whether this permission cascades to child folders
    inherit_to_children: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    folder: Mapped["Folder"] = relationship("Folder", foreign_keys=[folder_id])
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    granted_by: Mapped[Optional["User"]] = relationship("User", foreign_keys=[granted_by_id])

    __table_args__ = (
        # Unique constraint: one permission per folder-user pair
        Index(
            "idx_folder_permissions_unique",
            "folder_id", "user_id",
            unique=True,
        ),
        Index("idx_folder_permissions_folder", "folder_id"),
        Index("idx_folder_permissions_user", "user_id"),
        Index("idx_folder_permissions_granted_by", "granted_by_id"),
    )

    def __repr__(self) -> str:
        return f"<FolderPermission(folder_id={self.folder_id}, user_id={self.user_id}, level='{self.permission_level}')>"


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
# Workflow Models (Phase 1A)
# =============================================================================


class WorkflowStatus(str, PyEnum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class WorkflowTriggerType(str, PyEnum):
    """Types of workflow triggers."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    FORM = "form"
    EVENT = "event"


class WorkflowNodeType(str, PyEnum):
    """Types of workflow nodes."""
    AGENT = "agent"
    ACTION = "action"
    CONDITION = "condition"
    LOOP = "loop"
    CODE = "code"
    DELAY = "delay"
    HTTP = "http"
    NOTIFICATION = "notification"
    HUMAN_APPROVAL = "human_approval"
    START = "start"
    END = "end"


class Workflow(Base, UUIDMixin, TimestampMixin):
    """
    Workflow definition for automation.

    Enables visual workflow creation with drag-and-drop nodes
    connected by edges. Supports various trigger types and
    execution modes.
    """
    __tablename__ = "workflows"

    # Organization scope (multi-tenant)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    # Workflow metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_draft: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Trigger configuration
    trigger_type: Mapped[str] = mapped_column(
        String(50),
        default=WorkflowTriggerType.MANUAL.value,
        nullable=False,
    )
    trigger_config: Mapped[Optional[dict]] = mapped_column(JSONType())
    # For scheduled: {"cron": "0 9 * * *", "timezone": "UTC"}
    # For webhook: {"secret": "...", "path": "/webhooks/workflow/..."}
    # For form: {"fields": [...]}

    # Global workflow settings
    config: Mapped[Optional[dict]] = mapped_column(JSONType())
    # Stores: timeout, retry_policy, notifications, etc.

    # Ownership
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    nodes: Mapped[List["WorkflowNode"]] = relationship(
        "WorkflowNode",
        back_populates="workflow",
        cascade="all, delete-orphan",
    )
    edges: Mapped[List["WorkflowEdge"]] = relationship(
        "WorkflowEdge",
        back_populates="workflow",
        cascade="all, delete-orphan",
    )
    executions: Mapped[List["WorkflowExecution"]] = relationship(
        "WorkflowExecution",
        back_populates="workflow",
    )
    created_by: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_workflows_organization", "organization_id"),
        Index("idx_workflows_active", "is_active"),
        Index("idx_workflows_trigger", "trigger_type"),
        Index("idx_workflows_created_by", "created_by_id"),
    )

    def __repr__(self) -> str:
        return f"<Workflow(name='{self.name}', trigger='{self.trigger_type}', active={self.is_active})>"


class WorkflowNode(Base, UUIDMixin):
    """
    Individual node in a workflow.

    Each node represents a step in the workflow that can:
    - Execute an agent task
    - Make decisions based on conditions
    - Loop over data
    - Execute custom code
    - Wait for delays or approvals
    - Make HTTP requests
    - Send notifications
    """
    __tablename__ = "workflow_nodes"

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Node identification
    node_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Visual position in the canvas
    position_x: Mapped[float] = mapped_column(Float, default=0.0)
    position_y: Mapped[float] = mapped_column(Float, default=0.0)

    # Node-specific configuration (varies by node_type)
    config: Mapped[Optional[dict]] = mapped_column(JSONType())
    # Agent: {"agent_id": "...", "input_mapping": {...}}
    # Condition: {"expression": "{{result.status}} == 'success'"}
    # Loop: {"array_path": "{{data.items}}", "batch_size": 10}
    # Code: {"language": "javascript", "code": "..."}
    # Delay: {"seconds": 60} or {"until": "{{scheduled_time}}"}
    # HTTP: {"url": "...", "method": "POST", "headers": {...}}
    # Notification: {"channel": "email", "template": "..."}
    # Human Approval: {"approvers": [...], "timeout_hours": 24}

    # Relationships
    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="nodes")

    __table_args__ = (
        Index("idx_workflow_nodes_workflow", "workflow_id"),
        Index("idx_workflow_nodes_type", "node_type"),
    )

    def __repr__(self) -> str:
        return f"<WorkflowNode(name='{self.name}', type='{self.node_type}')>"


class WorkflowEdge(Base, UUIDMixin):
    """
    Connection between workflow nodes.

    Edges define the flow of execution and can include
    conditional logic for branching.
    """
    __tablename__ = "workflow_edges"

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Source and target nodes
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("workflow_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("workflow_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Edge label for UI
    label: Mapped[Optional[str]] = mapped_column(String(100))

    # Condition for taking this edge (for branching from condition nodes)
    condition: Mapped[Optional[str]] = mapped_column(Text)
    # Example: "true", "false", "{{result}} > 10"

    # Edge type for styling
    edge_type: Mapped[str] = mapped_column(String(20), default="default")
    # default, success, error, conditional

    # Relationships
    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="edges")

    __table_args__ = (
        Index("idx_workflow_edges_workflow", "workflow_id"),
        Index("idx_workflow_edges_source", "source_node_id"),
        Index("idx_workflow_edges_target", "target_node_id"),
    )

    def __repr__(self) -> str:
        return f"<WorkflowEdge(source={self.source_node_id}, target={self.target_node_id})>"


class WorkflowExecution(Base, UUIDMixin, TimestampMixin):
    """
    Tracks individual workflow executions.

    Each execution represents one run of a workflow,
    with full tracking of status, timing, and results.
    """
    __tablename__ = "workflow_executions"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Execution status
    status: Mapped[str] = mapped_column(
        String(50),
        default=WorkflowStatus.PENDING.value,
        index=True,
    )

    # Current position in workflow
    current_node_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID())

    # Trigger information
    trigger_type: Mapped[str] = mapped_column(String(50), nullable=False)
    trigger_data: Mapped[Optional[dict]] = mapped_column(JSONType())
    # Webhook payload, form data, scheduled time, etc.

    # Execution data
    input_data: Mapped[Optional[dict]] = mapped_column(JSONType())
    output_data: Mapped[Optional[dict]] = mapped_column(JSONType())
    context: Mapped[Optional[dict]] = mapped_column(JSONType())
    # Runtime context: variables, intermediate results

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_node_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID())
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Who triggered it
    triggered_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="executions")
    triggered_by: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_workflow_executions_org", "organization_id"),
        Index("idx_workflow_executions_workflow", "workflow_id"),
        Index("idx_workflow_executions_status", "status"),
        Index("idx_workflow_executions_started", "started_at"),
    )

    def __repr__(self) -> str:
        return f"<WorkflowExecution(workflow_id={self.workflow_id}, status='{self.status}')>"


class WorkflowNodeExecution(Base, UUIDMixin):
    """
    Tracks execution of individual nodes within a workflow execution.

    Provides detailed logging for debugging and analytics.
    """
    __tablename__ = "workflow_node_executions"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("workflow_executions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    node_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("workflow_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Status
    status: Mapped[str] = mapped_column(String(50), nullable=False)

    # Input/output for this node
    input_data: Mapped[Optional[dict]] = mapped_column(JSONType())
    output_data: Mapped[Optional[dict]] = mapped_column(JSONType())

    # Error info
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    __table_args__ = (
        Index("idx_workflow_node_exec_execution", "execution_id"),
        Index("idx_workflow_node_exec_node", "node_id"),
    )


# =============================================================================
# Audio Overview Models (Phase 1B)
# =============================================================================


class AudioOverviewFormat(str, PyEnum):
    """Formats for audio overview generation."""
    DEEP_DIVE = "deep_dive"
    BRIEF = "brief"
    CRITIQUE = "critique"
    DEBATE = "debate"
    LECTURE = "lecture"
    INTERVIEW = "interview"


class AudioOverviewStatus(str, PyEnum):
    """Status of audio overview generation."""
    PENDING = "pending"
    GENERATING_SCRIPT = "generating_script"
    GENERATING_AUDIO = "generating_audio"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class AudioOverview(Base, UUIDMixin, TimestampMixin):
    """
    AI-generated audio overviews/podcasts from documents.

    Inspired by NotebookLM's Audio Overview feature.
    Generates podcast-style discussions between AI hosts
    about the content of selected documents.
    """
    __tablename__ = "audio_overviews"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    # Source documents
    document_ids: Mapped[List[uuid.UUID]] = mapped_column(
        UUIDArrayType(),
        nullable=False,
    )
    folder_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Audio metadata
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Format and configuration
    format: Mapped[str] = mapped_column(
        String(50),
        default=AudioOverviewFormat.DEEP_DIVE.value,
        nullable=False,
    )
    language: Mapped[str] = mapped_column(String(10), default="en")

    # Host configuration
    host_config: Mapped[Optional[dict]] = mapped_column(JSONType())
    # {"host_a": {"name": "Alex", "voice": "alloy", "style": "curious"},
    #  "host_b": {"name": "Jordan", "voice": "echo", "style": "analytical"}}

    # Generation settings
    target_duration_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    tone: Mapped[Optional[str]] = mapped_column(String(50))  # casual, professional, academic

    # Generated content
    script: Mapped[Optional[dict]] = mapped_column(JSONType())
    # {"segments": [{"speaker": "host_a", "text": "...", "timestamp_ms": 0}, ...]}

    transcript: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text)

    # Audio file info
    audio_url: Mapped[Optional[str]] = mapped_column(String(1000))
    storage_path: Mapped[Optional[str]] = mapped_column(String(1000))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    audio_format: Mapped[str] = mapped_column(String(10), default="mp3")

    # TTS provider used
    tts_provider: Mapped[Optional[str]] = mapped_column(String(50))
    # openai, elevenlabs, coqui, etc.

    # Processing status
    status: Mapped[str] = mapped_column(
        String(50),
        default=AudioOverviewStatus.PENDING.value,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Progress tracking
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[Optional[str]] = mapped_column(String(100))

    # Ownership
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Usage tracking
    play_count: Mapped[int] = mapped_column(Integer, default=0)
    last_played_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    created_by: Mapped[Optional["User"]] = relationship("User")
    folder: Mapped[Optional["Folder"]] = relationship("Folder")

    __table_args__ = (
        Index("idx_audio_overviews_org", "organization_id"),
        Index("idx_audio_overviews_status", "status"),
        Index("idx_audio_overviews_format", "format"),
        Index("idx_audio_overviews_created_by", "created_by_id"),
    )

    def __repr__(self) -> str:
        return f"<AudioOverview(title='{self.title}', format='{self.format}', status='{self.status}')>"


# =============================================================================
# Bot Connection Models (Phase 2)
# =============================================================================


class BotPlatform(str, PyEnum):
    """Supported bot platforms."""
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    TELEGRAM = "telegram"


class BotConnection(Base, UUIDMixin, TimestampMixin):
    """
    External bot/chat platform integrations.

    Enables AI assistant access from chat platforms
    like Slack, Microsoft Teams, Discord, etc.
    """
    __tablename__ = "bot_connections"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    # Connection details
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    platform: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # Platform-specific identifiers
    workspace_id: Mapped[Optional[str]] = mapped_column(String(255))
    # Slack: team_id, Teams: tenant_id

    bot_user_id: Mapped[Optional[str]] = mapped_column(String(255))
    # The bot's user ID in the platform

    # Credentials (encrypted at service layer)
    bot_token_encrypted: Mapped[Optional[str]] = mapped_column(Text)
    refresh_token_encrypted: Mapped[Optional[str]] = mapped_column(Text)
    signing_secret_encrypted: Mapped[Optional[str]] = mapped_column(Text)

    # Webhook URL for incoming events
    webhook_url: Mapped[Optional[str]] = mapped_column(String(1000))

    # Configuration
    config: Mapped[Optional[dict]] = mapped_column(JSONType())
    # {"allowed_channels": [...], "default_agent_id": "...",
    #  "mention_required": true, "response_visibility": "thread"}

    # Permissions
    allowed_users: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())
    allowed_channels: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_event_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Error tracking
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text)

    # Ownership
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    created_by: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_bot_connections_org", "organization_id"),
        Index("idx_bot_connections_platform", "platform"),
        Index("idx_bot_connections_active", "is_active"),
        Index("idx_bot_connections_workspace", "platform", "workspace_id"),
    )

    def __repr__(self) -> str:
        return f"<BotConnection(name='{self.name}', platform='{self.platform}', active={self.is_active})>"


# =============================================================================
# Connector Models (Phase 3)
# =============================================================================


class ConnectorType(str, PyEnum):
    """Types of data source connectors."""
    GOOGLE_DRIVE = "google_drive"
    NOTION = "notion"
    CONFLUENCE = "confluence"
    ONEDRIVE = "onedrive"
    SHAREPOINT = "sharepoint"
    SLACK_DATA = "slack_data"
    YOUTUBE = "youtube"
    GITHUB = "github"
    DROPBOX = "dropbox"
    BOX = "box"


class ConnectorStatus(str, PyEnum):
    """Status of a connector instance."""
    CONNECTED = "connected"
    SYNCING = "syncing"
    ERROR = "error"
    DISCONNECTED = "disconnected"
    RATE_LIMITED = "rate_limited"


class ConnectorInstance(Base, UUIDMixin, TimestampMixin):
    """
    Data source connector instances.

    Enables automatic synchronization of content from
    external services like Google Drive, Notion, Confluence, etc.
    """
    __tablename__ = "connector_instances"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    # Connector identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    connector_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # OAuth credentials (encrypted at service layer)
    access_token_encrypted: Mapped[Optional[str]] = mapped_column(Text)
    refresh_token_encrypted: Mapped[Optional[str]] = mapped_column(Text)
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # API key (for non-OAuth connectors)
    api_key_encrypted: Mapped[Optional[str]] = mapped_column(Text)

    # External account info
    external_account_id: Mapped[Optional[str]] = mapped_column(String(255))
    external_account_email: Mapped[Optional[str]] = mapped_column(String(255))

    # Sync configuration
    config: Mapped[Optional[dict]] = mapped_column(JSONType())
    # {"folders": ["folder_id_1", "folder_id_2"],
    #  "file_types": [".pdf", ".docx"],
    #  "exclude_patterns": ["*.tmp"],
    #  "max_file_size_mb": 100}

    # Target folder for synced documents
    target_folder_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Sync settings
    auto_sync: Mapped[bool] = mapped_column(Boolean, default=True)
    sync_interval_minutes: Mapped[int] = mapped_column(Integer, default=60)
    sync_mode: Mapped[str] = mapped_column(String(20), default="incremental")
    # full, incremental

    # Sync state
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_sync_cursor: Mapped[Optional[str]] = mapped_column(Text)
    # Cursor/token for incremental sync
    next_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Statistics
    total_resources_synced: Mapped[int] = mapped_column(Integer, default=0)
    total_bytes_synced: Mapped[int] = mapped_column(BigInteger, default=0)

    # Status
    status: Mapped[str] = mapped_column(
        String(50),
        default=ConnectorStatus.CONNECTED.value,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_count: Mapped[int] = mapped_column(Integer, default=0)

    # Rate limiting
    rate_limit_reset_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Ownership
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    synced_resources: Mapped[List["SyncedResource"]] = relationship(
        "SyncedResource",
        back_populates="connector",
        cascade="all, delete-orphan",
    )
    target_folder: Mapped[Optional["Folder"]] = relationship("Folder")
    created_by: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_connector_instances_org", "organization_id"),
        Index("idx_connector_instances_type", "connector_type"),
        Index("idx_connector_instances_status", "status"),
        Index("idx_connector_instances_next_sync", "next_sync_at"),
    )

    def __repr__(self) -> str:
        return f"<ConnectorInstance(name='{self.name}', type='{self.connector_type}', status='{self.status}')>"


class SyncedResource(Base, UUIDMixin, TimestampMixin):
    """
    Resources synced from external data sources.

    Tracks the mapping between external resources and
    local documents, enabling incremental sync.
    """
    __tablename__ = "synced_resources"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    connector_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("connector_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # External resource identity
    external_id: Mapped[str] = mapped_column(String(500), nullable=False)
    external_parent_id: Mapped[Optional[str]] = mapped_column(String(500))
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # file, folder, page, database, message, video, etc.

    # Resource metadata
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    external_url: Mapped[Optional[str]] = mapped_column(String(2000))

    # Version tracking
    external_version: Mapped[Optional[str]] = mapped_column(String(100))
    external_modified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    external_modified_by: Mapped[Optional[str]] = mapped_column(String(255))

    # Local document link
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Sync metadata
    resource_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata",  # Column name in DB is still "metadata"
        JSONType(),
    )
    # External properties, permissions, etc.

    # Sync state
    sync_status: Mapped[str] = mapped_column(String(50), default="synced")
    # synced, pending, syncing, error, deleted
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    sync_error: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    connector: Mapped["ConnectorInstance"] = relationship(
        "ConnectorInstance",
        back_populates="synced_resources",
    )
    document: Mapped[Optional["Document"]] = relationship("Document")

    __table_args__ = (
        Index("idx_synced_resources_org", "organization_id"),
        Index("idx_synced_resources_connector", "connector_id"),
        Index("idx_synced_resources_external", "connector_id", "external_id", unique=True),
        Index("idx_synced_resources_document", "document_id"),
        Index("idx_synced_resources_status", "sync_status"),
    )

    def __repr__(self) -> str:
        return f"<SyncedResource(name='{self.name}', type='{self.resource_type}', status='{self.sync_status}')>"


# =============================================================================
# LLM Gateway Models (Phase 4)
# =============================================================================


class BudgetPeriod(str, PyEnum):
    """Budget period types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class Budget(Base, UUIDMixin, TimestampMixin):
    """
    Organization and user budget management.

    Enables cost control with configurable limits
    and automatic enforcement.
    """
    __tablename__ = "budgets"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    # Optional user-specific budget (if null, applies to org)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Budget name for identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Period configuration
    period: Mapped[str] = mapped_column(
        String(20),
        default=BudgetPeriod.MONTHLY.value,
        nullable=False,
    )

    # Limits (in USD)
    limit_amount: Mapped[float] = mapped_column(Float, nullable=False)
    soft_limit_amount: Mapped[Optional[float]] = mapped_column(Float)
    # Soft limit triggers warning, hard limit blocks requests

    # Current usage
    current_spend: Mapped[float] = mapped_column(Float, default=0.0)
    current_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    current_requests: Mapped[int] = mapped_column(Integer, default=0)

    # Reset tracking
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Enforcement
    is_hard_limit: Mapped[bool] = mapped_column(Boolean, default=False)
    # If true, requests are blocked when limit reached

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    paused_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Alert configuration
    alert_thresholds: Mapped[Optional[List[int]]] = mapped_column(JSONType())
    # [50, 80, 100] - percentages to alert at
    last_alert_threshold: Mapped[Optional[int]] = mapped_column(Integer)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_budgets_org", "organization_id"),
        Index("idx_budgets_user", "user_id"),
        Index("idx_budgets_active", "is_active"),
        Index("idx_budgets_period_end", "period_end"),
    )

    def __repr__(self) -> str:
        return f"<Budget(name='{self.name}', limit={self.limit_amount}, spend={self.current_spend})>"


class VirtualApiKey(Base, UUIDMixin, TimestampMixin):
    """
    Virtual API keys for external access.

    Enables issuing scoped API keys to users or
    external applications without exposing provider credentials.
    """
    __tablename__ = "virtual_api_keys"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    # Key owner (if null, org-level key)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Key identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Key value (only prefix stored, full key shown once at creation)
    key_prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    # e.g., "sk_live_abc1"
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # SHA-256 hash for lookup

    # Permissions
    scopes: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())
    # ["chat", "documents:read", "agents:execute"]

    allowed_models: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())
    # ["gpt-4", "claude-3-opus"] - if null, all models allowed

    allowed_ips: Mapped[Optional[List[str]]] = mapped_column(StringArrayType())
    # IP whitelist

    # Rate limiting
    rate_limit_rpm: Mapped[Optional[int]] = mapped_column(Integer)
    # Requests per minute
    rate_limit_tpd: Mapped[Optional[int]] = mapped_column(Integer)
    # Tokens per day

    # Budget (optional - separate from org/user budget)
    monthly_budget: Mapped[Optional[float]] = mapped_column(Float)
    current_month_spend: Mapped[float] = mapped_column(Float, default=0.0)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Usage tracking
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_virtual_api_keys_org", "organization_id"),
        Index("idx_virtual_api_keys_user", "user_id"),
        Index("idx_virtual_api_keys_hash", "key_hash"),
        Index("idx_virtual_api_keys_active", "is_active"),
        Index("idx_virtual_api_keys_prefix", "key_prefix"),
    )

    def __repr__(self) -> str:
        return f"<VirtualApiKey(name='{self.name}', prefix='{self.key_prefix}', active={self.is_active})>"


# =============================================================================
# Image Generation Models (Phase 5)
# =============================================================================


class ImageGenerationProvider(str, PyEnum):
    """Image generation providers."""
    OPENAI_DALLE = "openai_dalle"
    STABILITY = "stability"
    LOCALAI = "localai"
    COMFYUI = "comfyui"
    REPLICATE = "replicate"


class ImageGenerationStatus(str, PyEnum):
    """Status of image generation."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class GeneratedImage(Base, UUIDMixin, TimestampMixin):
    """
    AI-generated images.

    Tracks image generation requests and results,
    supporting multiple providers and use cases.
    """
    __tablename__ = "generated_images"

    # Organization scope
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        nullable=True,
        index=True,
    )

    # Generation request
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    enhanced_prompt: Mapped[Optional[str]] = mapped_column(Text)
    # Prompt after RAG enhancement

    negative_prompt: Mapped[Optional[str]] = mapped_column(Text)

    # Configuration
    provider: Mapped[str] = mapped_column(
        String(50),
        default=ImageGenerationProvider.OPENAI_DALLE.value,
        nullable=False,
    )
    model: Mapped[Optional[str]] = mapped_column(String(100))
    # dall-e-3, stable-diffusion-xl, etc.

    # Image settings
    width: Mapped[int] = mapped_column(Integer, default=1024)
    height: Mapped[int] = mapped_column(Integer, default=1024)
    style: Mapped[Optional[str]] = mapped_column(String(50))
    # vivid, natural, etc.
    quality: Mapped[Optional[str]] = mapped_column(String(20))
    # standard, hd

    # Context (for document-aware generation)
    context_document_ids: Mapped[Optional[List[uuid.UUID]]] = mapped_column(UUIDArrayType())
    context_text: Mapped[Optional[str]] = mapped_column(Text)

    # Result
    image_url: Mapped[Optional[str]] = mapped_column(String(2000))
    storage_path: Mapped[Optional[str]] = mapped_column(String(1000))
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(1000))
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)

    # Status
    status: Mapped[str] = mapped_column(
        String(50),
        default=ImageGenerationStatus.PENDING.value,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Cost tracking
    cost_usd: Mapped[Optional[float]] = mapped_column(Float)

    # Ownership
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Usage tracking
    download_count: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    created_by: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_generated_images_org", "organization_id"),
        Index("idx_generated_images_status", "status"),
        Index("idx_generated_images_provider", "provider"),
        Index("idx_generated_images_created_by", "created_by_id"),
    )

    def __repr__(self) -> str:
        return f"<GeneratedImage(prompt='{self.prompt[:50]}...', status='{self.status}')>"


# =============================================================================
# File Watcher Models
# =============================================================================

class WatchedDirectory(Base, UUIDMixin, TimestampMixin):
    """
    Persisted configuration for file watcher directories.
    Allows directories to be watched across server restarts.
    """
    __tablename__ = "watched_directories"

    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    recursive: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_process: Mapped[bool] = mapped_column(Boolean, default=True)
    access_tier_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("access_tiers.id", ondelete="SET NULL"), nullable=True
    )
    collection: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Target folder for uploaded documents
    folder_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("folders.id", ondelete="SET NULL"), nullable=True
    )

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Organization isolation
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True
    )

    # Who created this watch config
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Stats
    files_processed: Mapped[int] = mapped_column(Integer, default=0)
    last_scan_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_event_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    access_tier: Mapped[Optional["AccessTier"]] = relationship("AccessTier")
    folder: Mapped[Optional["Folder"]] = relationship("Folder")
    organization: Mapped[Optional["Organization"]] = relationship("Organization")
    created_by: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("idx_watched_dir_org", "organization_id"),
        Index("idx_watched_dir_enabled", "enabled"),
        Index("idx_watched_dir_path", "path"),
    )

    def __repr__(self) -> str:
        return f"<WatchedDirectory(path='{self.path}', enabled={self.enabled})>"


class FileWatcherEvent(Base, UUIDMixin):
    """
    Persisted file watcher events for tracking and retry.
    """
    __tablename__ = "file_watcher_events"

    watch_dir_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("watched_directories.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)  # created, modified, deleted, moved
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Processing status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, processing, completed, failed
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Link to created document (if successful)
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    watch_dir: Mapped["WatchedDirectory"] = relationship("WatchedDirectory")
    document: Mapped[Optional["Document"]] = relationship("Document")

    __table_args__ = (
        Index("idx_watcher_event_dir", "watch_dir_id"),
        Index("idx_watcher_event_status", "status"),
        Index("idx_watcher_event_detected", "detected_at"),
        Index("idx_watcher_event_hash", "file_hash"),
    )

    def __repr__(self) -> str:
        return f"<FileWatcherEvent(file='{self.file_name}', type='{self.event_type}', status='{self.status}')>"


class FileWatcherConfig(Base, UUIDMixin, TimestampMixin):
    """
    Global file watcher configuration and state.
    Single row table for watcher settings.
    """
    __tablename__ = "file_watcher_config"

    # Service state
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_start: Mapped[bool] = mapped_column(Boolean, default=False)

    # Processing settings
    batch_size: Mapped[int] = mapped_column(Integer, default=10)
    poll_interval_seconds: Mapped[int] = mapped_column(Integer, default=5)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    # Organization (if watcher is org-specific, otherwise null for global)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<FileWatcherConfig(enabled={self.enabled}, auto_start={self.auto_start})>"


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
