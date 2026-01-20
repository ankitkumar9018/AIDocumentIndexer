# Embedding Dimensions - Dynamic Configuration

AIDocumentIndexer supports **flexible embedding dimensions** that automatically adapt based on your configured embedding provider.

## How It Works

The database schema dynamically uses the embedding dimension from your `.env` configuration:

```python
# backend/db/models.py automatically detects dimension
EMBEDDING_DIMENSION = get_embedding_dimension()  # 768 or 1536 or 1024, etc.

# All embedding columns use this dynamic dimension
embedding: Mapped[Optional[List[float]]] = mapped_column(
    Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,
    nullable=True,
)
```

## Supported Providers & Dimensions

| Provider | Default Dimension | Model Examples |
|----------|------------------|----------------|
| **OpenAI** | 1536D | text-embedding-3-small, text-embedding-ada-002 |
| **Ollama** | 768D | nomic-embed-text, mxbai-embed-large |
| **HuggingFace** | 768D | sentence-transformers/all-MiniLM-L6-v2 |
| **Cohere** | 1024D | embed-english-v3.0 |

## Configuration

Set your embedding provider in `.env`:

```bash
# Example 1: Ollama (768D)
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Example 2: OpenAI (1536D)
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# Example 3: HuggingFace (768D)
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Checking Current Dimension

Run this script to see your current configuration:

```bash
python backend/scripts/check_embedding_dimension.py
```

Output:
```
============================================================
Embedding Dimension Configuration
============================================================
Provider: ollama
Default Embedding Model: not set
Ollama Embedding Model: nomic-embed-text
------------------------------------------------------------
Detected Dimension: 768D
Cached Dimension: 768D
============================================================
```

## Switching Between Providers

When you switch between providers with different dimensions (e.g., OpenAI → Ollama), you **must** re-index all data:

### Step 1: Update .env

```bash
# Old configuration (OpenAI - 1536D)
DEFAULT_LLM_PROVIDER=openai

# New configuration (Ollama - 768D)
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Step 2: Clear Old Embeddings

Run the migration script to clear incompatible embeddings:

```bash
# Dry run first to see what will be cleared
python backend/scripts/migrate_embedding_dimensions.py --dry-run

# Actually clear embeddings
python backend/scripts/migrate_embedding_dimensions.py
```

This clears embeddings from:
- ✅ Chunk embeddings (for document search)
- ✅ Document embeddings (for document-level search)
- ✅ Entity embeddings (for knowledge graph)
- ✅ Query cache embeddings (semantic caching)

### Step 3: Restart Application

The database schema will automatically use the new dimension:

```bash
# Restart your backend
# The app will now use 768D for Ollama or 1536D for OpenAI
```

### Step 4: Re-Index Data

#### 4a. Re-index Documents

Re-upload or re-process your documents so they generate new embeddings with the correct dimension.

**Via API:**
```bash
# Re-upload documents to generate new embeddings
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf" \
  -F "collection_id=your-collection-id"
```

**Via UI:**
- Go to Documents page
- Re-upload documents to the same collection
- The system will regenerate embeddings with the new dimension

#### 4b. Backfill Entity Embeddings

Run the backfill script to regenerate entity embeddings:

```bash
python backend/scripts/backfill_entity_embeddings.py
```

This will process all 1,560+ entities and generate new embeddings using the configured provider.

## Why Dimension Mismatch Matters

You **cannot** mix embedding dimensions in the same database:

### ❌ **BROKEN** - Mixed dimensions
```
Chunks: 100 chunks with 1536D embeddings (OpenAI)
Entities: 50 entities with 768D embeddings (Ollama)
Query: "Find similar chunks" → 768D query embedding

ERROR: Cannot compare 768D query to 1536D chunk embeddings!
```

### ✅ **WORKING** - Consistent dimensions
```
Chunks: 100 chunks with 768D embeddings (Ollama)
Entities: 50 entities with 768D embeddings (Ollama)
Query: "Find similar chunks" → 768D query embedding

SUCCESS: All embeddings are 768D, similarity search works!
```

## Under the Hood

The dimension detection function checks your environment:

```python
def get_embedding_dimension() -> int:
    """Get embedding dimension based on configured provider."""
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return 1536
    elif provider == "ollama":
        return 768
    elif provider == "huggingface":
        return 768
    elif provider == "cohere":
        return 1024
    else:
        return 768  # Safe default for local models
```

This value is cached at application startup:

```python
# Cached to avoid repeated env lookups
EMBEDDING_DIMENSION = get_embedding_dimension()

# Used in all embedding column definitions
embedding: Mapped[Optional[List[float]]] = mapped_column(
    Vector(EMBEDDING_DIMENSION) if HAS_PGVECTOR else Text,
    nullable=True,
)
```

## Migration Scripts

We provide two migration scripts:

### 1. Generic Migration (Recommended)
```bash
python backend/scripts/migrate_embedding_dimensions.py
```
- Detects current provider from `.env`
- Clears all embeddings (chunks, documents, entities, queries)
- Works for any provider switch

### 2. Legacy 768D Migration
```bash
python backend/scripts/migrate_entity_embeddings_768d.py
```
- Specifically for OpenAI → Ollama migration
- Only clears entity embeddings
- Kept for backward compatibility

## Best Practices

### 1. Choose One Provider and Stick With It

**Recommended**: Pick Ollama for cost-free, privacy-focused, local embeddings:
```bash
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Switching providers later requires full re-indexing, which is time-consuming for large document collections.

### 2. Development vs Production

**Development** (fast iteration, free):
```bash
DEFAULT_LLM_PROVIDER=ollama  # 768D, local, free
```

**Production** (higher quality):
```bash
DEFAULT_LLM_PROVIDER=openai  # 1536D, cloud, paid
```

If using different providers in dev vs prod, maintain **separate databases** to avoid dimension conflicts.

### 3. Test Before Migration

Always run migration with `--dry-run` first:
```bash
python backend/scripts/migrate_embedding_dimensions.py --dry-run
```

This shows you what will be cleared without actually making changes.

### 4. Backup Before Migration

```bash
# SQLite
cp aidocindexer.db aidocindexer.db.backup

# PostgreSQL
pg_dump your_database > backup.sql
```

## Troubleshooting

### Error: "expected 1536 dimensions, not 768"

**Cause**: Your database has 1536D embeddings but you're trying to add 768D embeddings.

**Solution**:
1. Check current provider: `python backend/scripts/check_embedding_dimension.py`
2. Run migration: `python backend/scripts/migrate_embedding_dimensions.py`
3. Re-index all data

### Error: "expected 768 dimensions, not 1536"

**Cause**: Opposite problem - database has 768D but trying to add 1536D.

**Solution**: Same as above, but update `.env` to use the correct provider before migrating.

### Embeddings Not Generating

**Check**: Does your provider setting match your installed models?

```bash
# If using Ollama, ensure model is pulled
ollama pull nomic-embed-text

# If using OpenAI, ensure API key is set
echo $OPENAI_API_KEY
```

## Performance Comparison

| Dimension | Storage | Search Speed | Quality |
|-----------|---------|--------------|---------|
| 384D | Smallest | Fastest | Good |
| 768D | Medium | Fast | Very Good |
| 1024D | Large | Medium | Excellent |
| 1536D | Largest | Slower | Excellent |

**Recommendation**: Use 768D (Ollama) for the best balance of quality, speed, and cost.

## Summary

✅ **Flexible**: Dimension automatically adapts to your provider
✅ **No Code Changes**: Just update `.env` and restart
✅ **Safe Migration**: Scripts ensure clean transitions
✅ **Multi-Provider**: OpenAI, Ollama, HuggingFace, Cohere supported
✅ **Documented**: Clear error messages and troubleshooting

**Key Takeaway**: Choose your embedding provider once and stick with it. Switching later is possible but requires full re-indexing.
