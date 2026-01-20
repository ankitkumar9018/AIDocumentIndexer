"""
Multi-embedding support for chunks and entities.

Allows storing embeddings from multiple providers/models simultaneously,
enabling instant provider switching without re-indexing.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import (
    String, Integer, ForeignKey, DateTime, Index,
    UniqueConstraint, func, select
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
import uuid

from backend.db.base import Base, UUIDMixin, GUID
from backend.db.types import JSONType


class ChunkEmbedding(Base, UUIDMixin):
    """
    Multiple embeddings for a single chunk.

    Enables:
    - Storing embeddings from multiple providers (Ollama, OpenAI, etc.)
    - Different dimensions per provider (768D, 1536D, 3072D)
    - Instant provider switching without re-indexing
    - Zero-downtime migrations

    Example:
        A chunk can have:
        - Ollama nomic-embed-text (768D) - free, local
        - OpenAI text-embedding-3-small (768D) - quality, cloud
        - OpenAI text-embedding-3-large (3072D) - maximum quality

        User switches by changing .env, no re-indexing needed!
    """
    __tablename__ = "chunk_embeddings"

    # Foreign key to chunk
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Embedding metadata
    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Embedding provider: ollama, openai, huggingface, cohere, voyage, mistral"
    )
    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Model name: nomic-embed-text, text-embedding-3-small, etc."
    )
    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Embedding dimension: 384, 512, 768, 1024, 1536, 3072, etc."
    )

    # The embedding vector
    # Stored as JSON for dimension flexibility
    # For pgvector, this could be Vector(dimension) but we use JSON for universality
    embedding: Mapped[List[float]] = mapped_column(
        JSONType(),
        nullable=False,
        comment="Embedding vector as array of floats"
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Track if this is the active/primary embedding for this chunk
    is_primary: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        comment="True if this is the currently active embedding for queries"
    )

    # Relationships
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="multi_embeddings")

    # Constraints
    __table_args__ = (
        # Only one embedding per (chunk, provider, model, dimension) combination
        UniqueConstraint(
            'chunk_id', 'provider', 'model', 'dimension',
            name='uq_chunk_embedding_config'
        ),
        # Index for fast lookup by provider/model
        Index('ix_chunk_embeddings_provider_model', 'provider', 'model'),
        # Index for fast lookup of primary embeddings
        Index('ix_chunk_embeddings_primary', 'chunk_id', 'is_primary'),
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkEmbedding(chunk_id='{self.chunk_id}', "
            f"provider='{self.provider}', model='{self.model}', "
            f"dimension={self.dimension}, is_primary={self.is_primary})>"
        )


class EntityEmbedding(Base, UUIDMixin):
    """
    Multiple embeddings for a single entity.

    Similar to ChunkEmbedding but for knowledge graph entities.
    Enables semantic entity search with multiple embedding providers.
    """
    __tablename__ = "entity_embeddings"

    # Foreign key to entity
    entity_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Embedding metadata
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)

    # The embedding vector
    embedding: Mapped[List[float]] = mapped_column(JSONType(), nullable=False)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    is_primary: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
    )

    # Relationships
    entity: Mapped["Entity"] = relationship("Entity", back_populates="multi_embeddings")

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            'entity_id', 'provider', 'model', 'dimension',
            name='uq_entity_embedding_config'
        ),
        Index('ix_entity_embeddings_provider_model', 'provider', 'model'),
        Index('ix_entity_embeddings_primary', 'entity_id', 'is_primary'),
    )

    def __repr__(self) -> str:
        return (
            f"<EntityEmbedding(entity_id='{self.entity_id}', "
            f"provider='{self.provider}', model='{self.model}', "
            f"dimension={self.dimension})>"
        )


# Helper functions for embedding management

async def get_or_create_embedding(
    db,
    chunk_id: uuid.UUID,
    provider: str,
    model: str,
    dimension: int,
    embedding: List[float],
    set_as_primary: bool = False
) -> ChunkEmbedding:
    """
    Get existing embedding or create new one.
    Optionally set as primary embedding for this chunk.
    """
    # Check if embedding already exists
    result = await db.execute(
        select(ChunkEmbedding).where(
            ChunkEmbedding.chunk_id == chunk_id,
            ChunkEmbedding.provider == provider,
            ChunkEmbedding.model == model,
            ChunkEmbedding.dimension == dimension,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        # Update existing embedding
        existing.embedding = embedding
        if set_as_primary and not existing.is_primary:
            # Unset other primary embeddings for this chunk
            await db.execute(
                ChunkEmbedding.__table__.update()
                .where(ChunkEmbedding.chunk_id == chunk_id)
                .values(is_primary=False)
            )
            existing.is_primary = True
        return existing
    else:
        # Create new embedding
        if set_as_primary:
            # Unset other primary embeddings
            await db.execute(
                ChunkEmbedding.__table__.update()
                .where(ChunkEmbedding.chunk_id == chunk_id)
                .values(is_primary=False)
            )

        new_embedding = ChunkEmbedding(
            chunk_id=chunk_id,
            provider=provider,
            model=model,
            dimension=dimension,
            embedding=embedding,
            is_primary=set_as_primary,
        )
        db.add(new_embedding)
        return new_embedding


async def get_best_embedding(
    db,
    chunk_id: uuid.UUID,
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
    preferred_dimension: Optional[int] = None,
) -> Optional[ChunkEmbedding]:
    """
    Get the best available embedding for a chunk.

    Selection priority:
    1. Primary embedding (if exists)
    2. Exact match (provider + model + dimension)
    3. Same provider, any model/dimension
    4. Same dimension, any provider
    5. Any available embedding
    """
    # Try primary first
    result = await db.execute(
        select(ChunkEmbedding).where(
            ChunkEmbedding.chunk_id == chunk_id,
            ChunkEmbedding.is_primary == True,
        )
    )
    primary = result.scalar_one_or_none()
    if primary:
        return primary

    # Try exact match
    if preferred_provider and preferred_model and preferred_dimension:
        result = await db.execute(
            select(ChunkEmbedding).where(
                ChunkEmbedding.chunk_id == chunk_id,
                ChunkEmbedding.provider == preferred_provider,
                ChunkEmbedding.model == preferred_model,
                ChunkEmbedding.dimension == preferred_dimension,
            )
        )
        exact = result.scalar_one_or_none()
        if exact:
            return exact

    # Try same provider
    if preferred_provider:
        result = await db.execute(
            select(ChunkEmbedding)
            .where(
                ChunkEmbedding.chunk_id == chunk_id,
                ChunkEmbedding.provider == preferred_provider,
            )
            .order_by(ChunkEmbedding.created_at.desc())
            .limit(1)
        )
        provider_match = result.scalar_one_or_none()
        if provider_match:
            return provider_match

    # Try same dimension
    if preferred_dimension:
        result = await db.execute(
            select(ChunkEmbedding)
            .where(
                ChunkEmbedding.chunk_id == chunk_id,
                ChunkEmbedding.dimension == preferred_dimension,
            )
            .order_by(ChunkEmbedding.created_at.desc())
            .limit(1)
        )
        dim_match = result.scalar_one_or_none()
        if dim_match:
            return dim_match

    # Return any available embedding
    result = await db.execute(
        select(ChunkEmbedding)
        .where(ChunkEmbedding.chunk_id == chunk_id)
        .order_by(ChunkEmbedding.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
