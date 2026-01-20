# Supported Embedding Models

Complete guide to all embedding providers and models supported by AIDocumentIndexer.

## Quick Configuration Examples

### 1. OpenAI (Recommended for Quality)

```bash
# Full dimension (1536D) - highest quality
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# Reduced dimension (512D) - saves 67% storage, minimal quality loss
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=512
```

### 2. Ollama (Recommended for Privacy/Cost)

```bash
# nomic-embed-text (768D) - best balance
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# mxbai-embed-large (1024D) - higher quality
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

### 3. HuggingFace (Free, Open Source)

```bash
# all-MiniLM-L6-v2 (384D) - fastest, smallest
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# all-mpnet-base-v2 (768D) - better quality
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### 4. Cohere

```bash
DEFAULT_LLM_PROVIDER=cohere
COHERE_API_KEY=...
```

### 5. Voyage AI

```bash
DEFAULT_LLM_PROVIDER=voyage
VOYAGE_API_KEY=...
```

### 6. Mistral

```bash
DEFAULT_LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
```

## Complete Model Reference

### OpenAI Models

| Model | Default Dim | Flexible Dim | Cost (per 1M tokens) | Quality |
|-------|-------------|--------------|---------------------|---------|
| `text-embedding-3-large` | 3072D | 256-3072D | $0.13 | ⭐⭐⭐⭐⭐ |
| `text-embedding-3-small` | 1536D | 512-1536D | $0.02 | ⭐⭐⭐⭐ |
| `text-embedding-ada-002` | 1536D | Fixed | $0.10 | ⭐⭐⭐ |

**OpenAI v3 Dimension Flexibility:**
- v3 models support **shortening embeddings** after generation
- Reduces storage and search time
- Minimal quality loss (typically <5% for 512D vs 1536D)
- Configure via `EMBEDDING_DIMENSION` env var

**Example**: 512D OpenAI (saves 67% storage):
```bash
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=512  # ← Reduced from 1536D
```

### Ollama Models (Local, Free)

| Model | Dimensions | Size | Quality | Speed |
|-------|-----------|------|---------|-------|
| `nomic-embed-text` | 768D | 274MB | ⭐⭐⭐⭐ | Fast |
| `mxbai-embed-large` | 1024D | 669MB | ⭐⭐⭐⭐⭐ | Medium |
| `snowflake-arctic-embed` | 768D | 436MB | ⭐⭐⭐⭐ | Fast |
| `all-minilm` | 384D | 120MB | ⭐⭐⭐ | Very Fast |

**Installation:**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull nomic-embed-text
```

**Configuration:**
```bash
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434  # Default
```

### HuggingFace Models (Free, Open Source)

| Model | Dimensions | Quality | Speed | License |
|-------|-----------|---------|-------|---------|
| `all-MiniLM-L6-v2` | 384D | ⭐⭐⭐ | Very Fast | Apache 2.0 |
| `all-mpnet-base-v2` | 768D | ⭐⭐⭐⭐ | Fast | Apache 2.0 |
| `bge-small-en-v1.5` | 768D | ⭐⭐⭐⭐ | Fast | MIT |
| `bge-large-en-v1.5` | 1024D | ⭐⭐⭐⭐⭐ | Medium | MIT |

**Installation:**
```bash
pip install sentence-transformers
```

**Configuration:**
```bash
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Cohere Models

| Model | Dimensions | Quality | Languages |
|-------|-----------|---------|-----------|
| `embed-english-v3.0` | 1024D | ⭐⭐⭐⭐⭐ | English |
| `embed-multilingual-v3.0` | 1024D | ⭐⭐⭐⭐⭐ | 100+ |

**Configuration:**
```bash
DEFAULT_LLM_PROVIDER=cohere
COHERE_API_KEY=...
```

### Voyage AI Models

| Model | Dimensions | Quality | Specialty |
|-------|-----------|---------|-----------|
| `voyage-2` | 1024D | ⭐⭐⭐⭐⭐ | Code & Docs |
| `voyage-law-2` | 1024D | ⭐⭐⭐⭐⭐ | Legal |

**Configuration:**
```bash
DEFAULT_LLM_PROVIDER=voyage
VOYAGE_API_KEY=...
```

### Mistral Models

| Model | Dimensions | Quality | Context Window |
|-------|-----------|---------|----------------|
| `mistral-embed` | 1024D | ⭐⭐⭐⭐ | 8K tokens |

**Configuration:**
```bash
DEFAULT_LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
```

## Dimension Override (Any Provider)

You can explicitly set the embedding dimension for **any provider**:

```bash
# Example: Force 512D regardless of provider defaults
EMBEDDING_DIMENSION=512
```

**When to use:**
- OpenAI v3 models: Reduce from 1536D to 512D (saves storage, minimal quality loss)
- Custom models with non-standard dimensions
- Testing different dimension trade-offs

**OpenAI v3 Dimension Recommendations:**

| Use Case | Recommended Dimension | Storage Savings |
|----------|---------------------|----------------|
| General search | 512D | 67% |
| High precision | 1024D | 33% |
| Maximum quality | 1536D (default) | 0% |

## Performance Comparison

### Speed (embeddings/second)

| Model | Single | Batch (100) |
|-------|--------|-------------|
| all-MiniLM-L6-v2 (384D) | 1000 | 5000 |
| nomic-embed-text (768D) | 500 | 2500 |
| text-embedding-3-small (1536D) | 2000 | 10000 |
| text-embedding-3-large (3072D) | 1000 | 5000 |

### Storage (per 1M embeddings)

| Dimension | Storage Size |
|-----------|-------------|
| 384D | ~1.5 GB |
| 512D | ~2.0 GB |
| 768D | ~3.0 GB |
| 1024D | ~4.0 GB |
| 1536D | ~6.0 GB |
| 3072D | ~12.0 GB |

### Quality (Retrieval Accuracy)

| Model | MTEB Score | Typical Use Case |
|-------|-----------|------------------|
| text-embedding-3-large (3072D) | 64.6 | Production, high-stakes |
| text-embedding-3-small (1536D) | 62.3 | Production, balanced |
| text-embedding-3-small (512D) | 61.0 | Production, cost-optimized |
| mxbai-embed-large (1024D) | 63.5 | Local, high quality |
| nomic-embed-text (768D) | 62.4 | Local, balanced |
| all-mpnet-base-v2 (768D) | 57.8 | Dev/testing |
| all-MiniLM-L6-v2 (384D) | 56.3 | Dev/testing, fast |

## Switching Between Models

When switching between models/providers:

### Same Dimension (No Re-indexing)

✅ **These switches work without re-indexing:**

```bash
# 768D → 768D (different providers, same dimension)
nomic-embed-text (768D) → all-mpnet-base-v2 (768D)  ✅
```

### Different Dimension (Re-indexing Required)

❌ **These switches require full re-indexing:**

```bash
# 768D → 1536D
nomic-embed-text (768D) → text-embedding-3-small (1536D)  ❌

# 1536D → 768D
text-embedding-3-small (1536D) → nomic-embed-text (768D)  ❌
```

**Re-indexing steps:**
1. Run migration: `python backend/scripts/migrate_embedding_dimensions.py`
2. Re-upload all documents
3. Run entity backfill: `python backend/scripts/backfill_entity_embeddings.py`

## Recommendations by Use Case

### 1. Production (Quality Priority)
```bash
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=1536  # Or 3072 for max quality
```

### 2. Production (Cost-Optimized)
```bash
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=512  # Saves 67% storage
```

### 3. Local/Private (No Internet)
```bash
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 4. Development/Testing
```bash
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 5. Multilingual
```bash
DEFAULT_LLM_PROVIDER=cohere
COHERE_API_KEY=...
```

### 6. Legal/Specialized
```bash
DEFAULT_LLM_PROVIDER=voyage
VOYAGE_API_KEY=...
```

## Cost Analysis (1M Documents, 500 tokens avg)

| Provider | Model | Dimension | Storage | API Cost | Total/Month |
|----------|-------|-----------|---------|----------|-------------|
| OpenAI | text-embedding-3-small | 512D | 2GB | $10 | ~$10 |
| OpenAI | text-embedding-3-small | 1536D | 6GB | $10 | ~$10 |
| OpenAI | text-embedding-3-large | 3072D | 12GB | $65 | ~$65 |
| Ollama | nomic-embed-text | 768D | 3GB | $0 | $0 |
| HuggingFace | all-MiniLM-L6-v2 | 384D | 1.5GB | $0 | $0 |

**Winner**: Ollama nomic-embed-text (768D) - Best balance of quality, cost ($0), and privacy.

## FAQ

### Q: Can I use OpenAI with 768D dimensions?

**A: Yes!** OpenAI `text-embedding-3-small` and `text-embedding-3-large` support flexible dimensions:

```bash
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=768  # ← Match Ollama dimension
```

This allows you to:
- ✅ Use OpenAI in production
- ✅ Use Ollama in development
- ✅ **NO re-indexing** when switching (same 768D dimension)

### Q: What happens if I don't set EMBEDDING_DIMENSION?

**A:** The system auto-detects based on your provider:
- OpenAI → 1536D (default for text-embedding-3-small)
- Ollama → 768D (for nomic-embed-text)
- HuggingFace → 384D or 768D (model-specific)

### Q: Can I mix different embedding models?

**A:** Only if they have the **same dimension**. For example:
- ✅ nomic-embed-text (768D) + all-mpnet-base-v2 (768D) = **Works**
- ❌ nomic-embed-text (768D) + text-embedding-3-small (1536D) = **Breaks**

### Q: Which model is fastest?

**A:** For speed: `all-MiniLM-L6-v2` (384D) > `nomic-embed-text` (768D) > `text-embedding-3-small` (1536D)

### Q: Which model has best quality?

**A:** For quality: `text-embedding-3-large` (3072D) > `mxbai-embed-large` (1024D) > `text-embedding-3-small` (1536D)

### Q: Can I use different models for different collections?

**A:** No, the embedding dimension is **database-wide**. All collections must use the same dimension.

## Summary

✅ **Flexible Dimensions**: Database automatically adapts to your provider
✅ **OpenAI Dimension Reduction**: Use `EMBEDDING_DIMENSION=512` to save storage
✅ **Many Providers**: OpenAI, Ollama, HuggingFace, Cohere, Voyage, Mistral
✅ **Easy Switching**: Same dimension = no re-indexing needed
✅ **Cost-Effective**: Ollama (free, local) competitive with OpenAI quality

**Recommended Setup** (Production):
```bash
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=768  # Match Ollama for dev/prod consistency
```
