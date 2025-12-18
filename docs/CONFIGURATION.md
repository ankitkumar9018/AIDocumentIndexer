# Configuration Guide

Comprehensive guide to configuring AIDocumentIndexer.

## Environment Variables

All configuration is done through environment variables. Copy `.env.example` to `.env` and customize.

---

## Core Settings

### Application

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `true` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `APP_NAME` | Application name | `AIDocumentIndexer` |
| `API_HOST` | Backend host | `0.0.0.0` |
| `API_PORT` | Backend port | `8000` |

### Security

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Secret key for JWT tokens | Required |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` |
| `JWT_EXPIRATION_HOURS` | Token expiration time | `24` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:3000` |

---

## Database

### PostgreSQL

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `DB_POOL_SIZE` | Connection pool size | `5` |
| `DB_MAX_OVERFLOW` | Max additional connections | `10` |
| `DB_POOL_TIMEOUT` | Pool timeout in seconds | `30` |

**Connection String Format:**
```
postgresql://username:password@host:port/database
```

**With SSL:**
```
postgresql://username:password@host:port/database?sslmode=require
```

### Redis

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `REDIS_PASSWORD` | Redis password | None |
| `REDIS_DB` | Redis database number | `0` |
| `REDIS_MAX_CONNECTIONS` | Max pool connections | `10` |

---

## LLM Providers

### OpenAI

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_CHAT_MODEL` | Chat model | `gpt-4o` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_ORGANIZATION` | Organization ID | None |

### Anthropic

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `ANTHROPIC_MODEL` | Claude model | `claude-3-5-sonnet-20241022` |

### Ollama

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_ENABLED` | Enable Ollama | `true` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_CHAT_MODEL` | Chat model | `llama3.2` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |

### Default LLM Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_LLM_PROVIDER` | Default provider for chat and embeddings | `openai` |
| `DEFAULT_CHAT_MODEL` | Default chat model | `gpt-4o` |
| `DEFAULT_EMBEDDING_MODEL` | Default embedding model | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `1536` |
| `DEFAULT_TEMPERATURE` | Default temperature | `0.7` |
| `DEFAULT_MAX_TOKENS` | Default max tokens | `4096` |

**Provider Options:**
- `openai` - OpenAI GPT models (requires `OPENAI_API_KEY`)
- `ollama` - Local Ollama models (no API key required)
- `anthropic` - Anthropic Claude models (requires `ANTHROPIC_API_KEY`)

**Embedding Dimensions by Model:**
| Model | Dimensions |
|-------|------------|
| `text-embedding-3-small` (OpenAI) | 1536 |
| `text-embedding-3-large` (OpenAI) | 3072 |
| `nomic-embed-text` (Ollama) | 768 |
| `all-MiniLM-L6-v2` (HuggingFace) | 384 |

---

## Document Processing

### Chunking

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `MIN_CHUNK_SIZE` | Minimum chunk size | `100` |
| `MAX_CHUNK_SIZE` | Maximum chunk size | `2000` |

### OCR (Tesseract)

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OCR` | Enable OCR for images/PDFs | `true` |
| `TESSDATA_PREFIX` | Path to Tesseract language data files | `/opt/homebrew/share/tessdata` (macOS) |
| `OCR_LANGUAGE` | OCR language(s) - use `+` for multiple | `eng` |
| `OCR_CONFIDENCE_THRESHOLD` | Minimum confidence | `0.6` |

**Multi-language OCR Example:**
```bash
# German + English
OCR_LANGUAGE=deu+eng

# French + English
OCR_LANGUAGE=fra+eng
```

**Installing Additional Languages (macOS):**
```bash
# Install all languages
brew install tesseract-lang

# Or download specific language from:
# https://github.com/tesseract-ocr/tessdata
```

### Image Optimization

| Variable | Description | Default |
|----------|-------------|---------|
| `OCR_DEFAULT_ZOOM` | Default zoom level for OCR rendering | `1.5` |
| `OCR_LARGE_PDF_ZOOM` | Zoom for large files (faster) | `1.0` |
| `LARGE_FILE_THRESHOLD_MB` | Size threshold for large file handling | `10` |
| `IMAGE_MAX_DIMENSION` | Max image dimension before resize | `2000` |
| `ENABLE_IMAGE_COMPRESSION` | Compress images before OCR | `true` |
| `CONVERT_TO_GRAYSCALE` | Convert to grayscale for faster OCR | `true` |

### Smart Processing

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_PROCESSING_MODE` | Default mode (full/smart/text_only) | `smart` |
| `SMART_CHUNKING_ENABLED` | Use semantic chunking | `true` |
| `SMART_IMAGE_HANDLING` | Auto-optimize images | `true` |
| `DETECT_DUPLICATES` | Skip duplicate files | `true` |

### File Handling

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_FILE_SIZE` | Max upload size in bytes | `104857600` (100MB) |
| `UPLOAD_DIR` | Upload directory | `./uploads` |
| `ALLOWED_FILE_TYPES` | Allowed extensions | `pdf,docx,txt,...` |

---

## RAG Settings

### Vector Store

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_STORE_BACKEND` | Vector store backend (`chroma`, `pgvector`, `auto`) | `auto` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./data/chroma` |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name | `documents` |
| `VECTOR_SIMILARITY_THRESHOLD` | Minimum similarity score | `0.4` |
| `DEFAULT_TOP_K` | Default results count | `5` |
| `MAX_TOP_K` | Maximum results count | `20` |

**Backend Selection:**
- `auto` - Automatically select based on `DATABASE_URL` (ChromaDB for SQLite, pgvector for PostgreSQL)
- `chroma` - Use ChromaDB (local, serverless, good for development)
- `pgvector` - Use PostgreSQL with pgvector extension (recommended for production)

**Similarity Threshold Note:**
The default threshold of `0.4` is optimized for OCR'd documents which may have lower similarity scores due to text extraction artifacts. Increase to `0.7` for cleaner text documents.

### Search

| Variable | Description | Default |
|----------|-------------|---------|
| `SEARCH_TYPE` | Default search type | `hybrid` |
| `HYBRID_ALPHA` | Vector/keyword balance | `0.5` |
| `RERANK_ENABLED` | Enable reranking | `true` |
| `RERANK_TOP_N` | Rerank top N results | `10` |

**Search Types:**
- `vector` - Pure vector similarity search
- `keyword` - Full-text keyword search
- `hybrid` - Combines vector and keyword using Reciprocal Rank Fusion (RRF)

---

## Distributed Processing

### Ray

| Variable | Description | Default |
|----------|-------------|---------|
| `RAY_ENABLED` | Enable Ray | `true` |
| `RAY_ADDRESS` | Ray cluster address | `auto` |
| `RAY_NUM_CPUS` | CPUs per worker | `2` |
| `RAY_NUM_GPUS` | GPUs per worker | `0` |
| `RAY_OBJECT_STORE_MEMORY` | Object store size | `2147483648` (2GB) |

---

## Web Scraper

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_WEB_SCRAPER` | Enable web scraping | `true` |
| `SCRAPER_TIMEOUT` | Request timeout | `30000` |
| `SCRAPER_MAX_DEPTH` | Max crawl depth | `3` |
| `SCRAPER_RATE_LIMIT` | Requests per second | `1` |
| `SCRAPER_USER_AGENT` | User agent string | `AIDocumentIndexer/0.1` |

---

## Cost Tracking

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_COST_TRACKING` | Enable cost tracking | `true` |
| `COST_ALERT_EMAIL` | Alert email address | None |
| `DEFAULT_COST_LIMIT` | Default monthly limit | `100.00` |

---

## File Watcher

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_FILE_WATCHER` | Enable file watching | `false` |
| `WATCH_DIRECTORIES` | Directories to watch | `./watched` |
| `WATCH_PATTERNS` | File patterns | `*.pdf,*.docx,*.txt` |
| `WATCH_IGNORE_PATTERNS` | Ignore patterns | `.git/*,node_modules/*` |

---

## Access Control

### Tiers

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_ACCESS_TIER` | New user tier | `10` |
| `MAX_ACCESS_TIER` | Maximum tier level | `100` |

**Tier Levels:**
- 10: Viewer (read public documents)
- 30: User (read all accessible documents)
- 50: Editor (read/write team documents)
- 90: Manager (manage team documents)
- 100: Admin (full access)

---

## Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |
| `NEXT_PUBLIC_WS_URL` | WebSocket URL | `ws://localhost:8000` |
| `NEXT_PUBLIC_APP_NAME` | Display name | `AIDocumentIndexer` |

---

## Example Configurations

### Development (Local with Ollama)

```bash
# Application
DEBUG=true
LOG_LEVEL=DEBUG
APP_ENV=development
DEV_MODE=true

# Database (SQLite for local dev)
DATABASE_URL=sqlite:///./aidocindexer.db

# Vector Store (ChromaDB for local)
VECTOR_STORE_BACKEND=chroma
CHROMA_PERSIST_DIRECTORY=./data/chroma

# LLM Provider (Ollama - no API key needed)
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# OCR (Tesseract)
TESSDATA_PREFIX=/opt/homebrew/share/tessdata
OCR_LANGUAGE=eng

# Security
JWT_SECRET=dev-secret-key-change-in-production
ALLOWED_ORIGINS=http://localhost:3000
```

### Development (with OpenAI)

```bash
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/aidocindexer_dev
REDIS_URL=redis://localhost:6379

# LLM Provider (OpenAI)
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=1536

JWT_SECRET=dev-secret-key
ALLOWED_ORIGINS=http://localhost:3000
```

### Production

```bash
DEBUG=false
LOG_LEVEL=INFO

# Database (PostgreSQL)
DATABASE_URL=postgresql://user:STRONG_PASS@db.example.com:5432/aidocindexer
REDIS_URL=redis://:REDIS_PASS@redis.example.com:6379

# Vector Store (pgvector)
VECTOR_STORE_BACKEND=pgvector

# LLM Provider
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_DIMENSION=1536

# OCR
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
OCR_LANGUAGE=eng

# Security
JWT_SECRET=GENERATE_STRONG_SECRET_KEY
ALLOWED_ORIGINS=https://app.example.com
```

### High-Volume Processing

```bash
RAY_ENABLED=true
RAY_NUM_CPUS=8
RAY_OBJECT_STORE_MEMORY=8589934592
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
REDIS_MAX_CONNECTIONS=50
CHUNK_SIZE=500
```

### Multi-Language OCR (German + English)

```bash
TESSDATA_PREFIX=/opt/homebrew/share/tessdata
OCR_LANGUAGE=deu+eng
```

---

## Configuration Validation

The application validates configuration on startup. Missing required variables will cause startup failure with clear error messages.

```bash
# Test configuration
python -c "from backend.core.config import settings; print('Config OK')"
```

---

## Sensitive Variables

Never commit sensitive variables to version control:
- `DATABASE_URL` (if contains password)
- `REDIS_PASSWORD`
- `JWT_SECRET`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

Use environment-specific `.env` files or secrets management (HashiCorp Vault, AWS Secrets Manager, etc.).
