# Multi-Embedding Storage Proposal

**Problem:** Currently, switching between embedding providers (e.g., Ollama 768D → OpenAI 1536D) requires re-indexing all documents because only one embedding per chunk is stored.

**Solution:** Store **multiple embeddings per chunk** so users can switch providers instantly without re-indexing.

---

## Proposed Architecture

### Option 1: Separate Embeddings Table (Recommended)

Create a new `chunk_embeddings` table to store multiple embeddings per chunk:

```python
class ChunkEmbedding(Base, UUIDMixin):
    """
    Multiple embeddings for a chunk, supporting different providers/dimensions.

    Allows storing embeddings from:
    - Ollama nomic-embed-text (768D)
    - OpenAI text-embedding-3-small (1536D or reduced)
    - OpenAI text-embedding-3-large (3072D)
    - Any other provider/model combination

    Users can switch providers without re-indexing.
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
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # "ollama", "openai", etc.
    model: Mapped[str] = mapped_column(String(100), nullable=False)    # "nomic-embed-text", "text-embedding-3-small"
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)    # 768, 1536, 3072, etc.

    # The embedding vector (stored as JSON for dimension flexibility)
    embedding: Mapped[List[float]] = mapped_column(JSONType(), nullable=False)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="embeddings")

    # Composite unique constraint: one embedding per (chunk_id, provider, model, dimension)
    __table_args__ = (
        UniqueConstraint('chunk_id', 'provider', 'model', 'dimension', name='uq_chunk_embedding'),
        Index('ix_chunk_embeddings_provider_model', 'provider', 'model'),
    )
```

### Update Chunk Model

Add relationship to embeddings:

```python
class Chunk(Base, UUIDMixin):
    # ... existing fields ...

    # Keep primary embedding for backward compatibility
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,
        nullable=True,
    )

    # New relationship for multi-embedding support
    embeddings: Mapped[List["ChunkEmbedding"]] = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )
```

---

## Migration Strategy

### Phase 1: Add New Table (Non-Breaking)

1. Create `chunk_embeddings` table via Alembic migration
2. Keep existing `chunks.embedding` column (backward compatible)
3. No data migration yet - both schemas coexist

### Phase 2: Backfill Existing Embeddings

Copy existing embeddings to new table:

```python
# backend/scripts/migrate_to_multi_embeddings.py

async def migrate_existing_embeddings():
    """
    Copy existing chunk.embedding to chunk_embeddings table.
    Preserves current embeddings as first entry in new system.
    """
    async with get_async_session() as db:
        # Get current provider config
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
        if provider == "ollama":
            model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        else:
            model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

        dimension = get_embedding_dimension()

        # Get all chunks with embeddings
        result = await db.execute(
            select(Chunk).where(Chunk.embedding.isnot(None))
        )
        chunks = result.scalars().all()

        print(f"Migrating {len(chunks)} chunk embeddings...")

        for chunk in chunks:
            # Create ChunkEmbedding record
            chunk_emb = ChunkEmbedding(
                chunk_id=chunk.id,
                provider=provider,
                model=model,
                dimension=dimension,
                embedding=chunk.embedding,  # Copy existing embedding
            )
            db.add(chunk_emb)

        await db.commit()
        print("Migration complete!")
```

### Phase 3: Generate Additional Embeddings

Add embeddings for other providers:

```python
# backend/scripts/generate_multi_provider_embeddings.py

async def generate_additional_embeddings(
    target_provider: str,
    target_model: str,
    batch_size: int = 50
):
    """
    Generate embeddings using a different provider while keeping existing ones.

    Example: Generate OpenAI 768D embeddings while keeping Ollama 768D.
    """
    async with get_async_session() as db:
        # Initialize target embedding service
        target_service = EmbeddingService(
            provider=target_provider,
            model=target_model
        )

        # Get target dimension
        test_emb = target_service.embed_texts(["test"])
        target_dim = len(test_emb[0])

        print(f"Generating {target_provider}/{target_model} ({target_dim}D) embeddings...")

        # Find chunks without this specific embedding
        # (chunks that don't have a chunk_embeddings entry for this provider/model/dimension)
        result = await db.execute(
            select(Chunk)
            .outerjoin(
                ChunkEmbedding,
                and_(
                    ChunkEmbedding.chunk_id == Chunk.id,
                    ChunkEmbedding.provider == target_provider,
                    ChunkEmbedding.model == target_model,
                    ChunkEmbedding.dimension == target_dim
                )
            )
            .where(ChunkEmbedding.id.is_(None))  # No embedding for this config
        )
        chunks = result.scalars().all()

        print(f"Found {len(chunks)} chunks needing {target_provider} embeddings")

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]

            # Generate embeddings
            embeddings = target_service.embed_texts(texts)

            # Store in chunk_embeddings table
            for chunk, embedding in zip(batch, embeddings):
                chunk_emb = ChunkEmbedding(
                    chunk_id=chunk.id,
                    provider=target_provider,
                    model=target_model,
                    dimension=target_dim,
                    embedding=embedding,
                )
                db.add(chunk_emb)

            await db.commit()
            print(f"Progress: {i + len(batch)}/{len(chunks)}")
```

---

## Usage Examples

### Example 1: Generate Both Ollama and OpenAI Embeddings

```bash
# Generate Ollama 768D embeddings (current)
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
python backend/scripts/backfill_chunk_embeddings.py

# Generate OpenAI 768D embeddings (additional)
python backend/scripts/generate_multi_provider_embeddings.py \
  --provider openai \
  --model text-embedding-3-small \
  --dimension 768

# Now you have both! Switch instantly via config
```

### Example 2: Query Service Auto-Selection

```python
class EmbeddingQueryService:
    """
    Automatically selects best available embedding for current config.
    Falls back to alternative if primary not available.
    """

    async def get_embedding_for_search(
        self,
        chunk: Chunk,
        preferred_provider: str,
        preferred_model: str,
        preferred_dimension: int
    ) -> Optional[List[float]]:
        """
        Get embedding for search, with fallback logic:
        1. Try exact match (provider + model + dimension)
        2. Try same provider, different dimension
        3. Try any provider with same dimension
        4. Return None if no embeddings available
        """

        # Try exact match first
        for emb in chunk.embeddings:
            if (emb.provider == preferred_provider
                and emb.model == preferred_model
                and emb.dimension == preferred_dimension):
                return emb.embedding

        # Fallback: same provider, any dimension
        for emb in chunk.embeddings:
            if emb.provider == preferred_provider:
                logger.info(f"Using {emb.dimension}D instead of {preferred_dimension}D")
                return emb.embedding

        # Fallback: any provider, same dimension
        for emb in chunk.embeddings:
            if emb.dimension == preferred_dimension:
                logger.info(f"Using {emb.provider} instead of {preferred_provider}")
                return emb.embedding

        # No suitable embedding found
        return None
```

### Example 3: RAG Service Integration

```python
# backend/services/rag.py

async def _vector_search(
    self,
    query: str,
    collection_ids: Optional[List[str]] = None,
    top_k: int = 10
) -> List[Chunk]:
    """
    Vector search with multi-embedding support.
    """
    # Get current embedding config
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
    if provider == "ollama":
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    else:
        model = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    dimension = get_embedding_dimension()

    # Generate query embedding
    embedding_service = EmbeddingService(provider=provider, model=model)
    query_embedding = embedding_service.embed_texts([query])[0]

    # Search using chunk_embeddings table
    stmt = (
        select(Chunk, ChunkEmbedding)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(
            ChunkEmbedding.provider == provider,
            ChunkEmbedding.model == model,
            ChunkEmbedding.dimension == dimension
        )
    )

    if collection_ids:
        stmt = stmt.join(Document).join(Collection).where(
            Collection.id.in_(collection_ids)
        )

    result = await self.db.execute(stmt)
    chunks_with_embeddings = result.all()

    # Compute cosine similarity
    scored_chunks = []
    for chunk, chunk_emb in chunks_with_embeddings:
        similarity = cosine_similarity(query_embedding, chunk_emb.embedding)
        chunk.score = similarity
        scored_chunks.append(chunk)

    # Sort by similarity and return top_k
    scored_chunks.sort(key=lambda c: c.score, reverse=True)
    return scored_chunks[:top_k]
```

---

## Storage Impact

### Current System (Single Embedding)
- 3,959 chunks × 768 floats × 4 bytes = **~12 MB**

### Multi-Embedding System
Assuming you store 2 embeddings per chunk (Ollama 768D + OpenAI 768D):
- 3,959 chunks × 2 embeddings × 768 floats × 4 bytes = **~24 MB**

For 3 embeddings (Ollama 768D, OpenAI 768D, OpenAI 1536D):
- 3,959 chunks × 3 embeddings × average 1024 floats × 4 bytes = **~49 MB**

**Conclusion:** Storage cost is acceptable (24-49 MB for multi-embedding vs 12 MB for single).

---

## Performance Impact

### Query Performance
**Before:** Single embedding lookup
```sql
SELECT * FROM chunks WHERE embedding <-> query_vector < threshold
```
**Time:** ~10-20ms for 4K chunks

**After:** Multi-embedding with filter
```sql
SELECT * FROM chunks
JOIN chunk_embeddings ON chunk_embeddings.chunk_id = chunks.id
WHERE chunk_embeddings.provider = 'ollama'
  AND chunk_embeddings.model = 'nomic-embed-text'
  AND chunk_embeddings.embedding <-> query_vector < threshold
```
**Time:** ~15-30ms for 4K chunks (slight overhead from join)

**Optimization:** Add index on `(provider, model, dimension)` to minimize overhead.

---

## Alternative: Hybrid Approach

Keep primary embedding in `chunks.embedding` for performance, use `chunk_embeddings` for alternatives:

```python
class Chunk(Base, UUIDMixin):
    # Primary embedding (fast access, no joins)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,
        nullable=True,
    )

    # Provider/model metadata for primary embedding
    embedding_provider: Mapped[Optional[str]] = mapped_column(String(50))
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100))
    embedding_dimension: Mapped[Optional[int]] = mapped_column(Integer)

    # Alternative embeddings (when switching providers)
    alternative_embeddings: Mapped[List["ChunkEmbedding"]] = relationship(...)
```

**Benefits:**
- Fast queries use primary embedding (no join)
- Alternative embeddings available when switching
- Minimal performance impact

---

## Recommendation

**Recommended Approach:** Hybrid (primary + alternatives)

**Migration Steps:**
1. Add `chunk_embeddings` table (Phase 1)
2. Add `embedding_provider`, `embedding_model`, `embedding_dimension` columns to `chunks` table
3. Backfill metadata for existing embeddings
4. Generate alternative embeddings as needed
5. Update RAG service to check primary first, fall back to alternatives

**User Experience:**
```bash
# Generate Ollama embeddings (primary)
python backend/scripts/backfill_chunk_embeddings.py

# Generate OpenAI embeddings (alternative)
python backend/scripts/generate_multi_provider_embeddings.py \
  --provider openai --model text-embedding-3-small --dimension 768

# Switch providers instantly (no re-indexing!)
# Just change .env and restart:
DEFAULT_LLM_PROVIDER=openai  # Switches to OpenAI 768D
# System automatically uses OpenAI embeddings for queries
```

**Cost:**
- Storage: +100% (2x embeddings) = ~24 MB total
- Query performance: -10% (slightly slower due to metadata check)
- Flexibility: ∞ (instant provider switching)

**Winner:** Small storage cost for huge flexibility gain.
