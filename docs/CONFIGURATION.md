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

**Admin UI Model Management:**

You can manage Ollama models directly from the Admin Settings page:

1. Go to **Admin → Settings → LLM Providers** tab
2. Scroll to the **Ollama Local Models** section
3. Features available:
   - **View installed models** - See all chat, embedding, and vision models with size info
   - **Pull new models** - Download models from Ollama library (e.g., `qwen2.5vl`, `llava`, `nomic-embed-text`)
   - **Delete models** - Remove unused models to free disk space
   - **Quick select** - One-click buttons for recommended models

**Recommended Models:**
| Model | Type | Best For |
|-------|------|----------|
| `qwen2.5vl` | Vision | Document OCR, tables (95.7% DocVQA) |
| `llava` | Vision | General image understanding |
| `llama3.2` | Chat | Fast responses, general chat |
| `nomic-embed-text` | Embedding | Document embeddings (768 dims) |

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

### OCR Configuration

**⚡ NEW: Admin UI Configuration Available!**

OCR is now fully configurable via the Admin Settings UI. Navigate to **Dashboard → Admin → Settings → OCR Configuration** to manage OCR providers, models, and languages through a visual interface.

For complete OCR configuration details, see [OCR_CONFIGURATION.md](./OCR_CONFIGURATION.md).

#### Quick Reference

**Admin UI (Recommended):**
- Provider selection (PaddleOCR, Tesseract, Auto)
- Model variant (Server/Mobile)
- Multi-language support (8+ languages)
- One-click model downloads
- Real-time model status

**Database Settings (via Admin UI or API):**
| Setting Key | Description | Default |
|------------|-------------|---------|
| `ocr.provider` | OCR engine (paddleocr/tesseract/auto) | `paddleocr` |
| `ocr.paddle.variant` | Model variant (server/mobile) | `server` |
| `ocr.paddle.languages` | Language codes array | `["en", "de"]` |
| `ocr.paddle.auto_download` | Auto-download models on startup | `true` |
| `ocr.tesseract.fallback_enabled` | Fallback to Tesseract | `true` |

**Environment Variables (Legacy - Still Supported):**
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OCR` | Enable OCR for images/PDFs | `true` |
| `PADDLEX_HOME` | PaddleOCR model directory | `./data/paddle_models` |
| `TESSDATA_PREFIX` | Tesseract language data path | `/opt/homebrew/share/tessdata` (macOS) |
| `OCR_LANGUAGE` | Tesseract language(s) | `eng` |
| `OCR_CONFIDENCE_THRESHOLD` | Minimum confidence | `0.6` |

**PaddleOCR (Recommended):**
```bash
# Model storage (project-local)
PADDLEX_HOME=./data/paddle_models
PADDLE_HUB_HOME=./data/paddle_models/official_models
PADDLE_PDX_MODEL_SOURCE=HF  # Use HuggingFace mirror
```

**Tesseract (Legacy):**
```bash
# Multi-language example
OCR_LANGUAGE=deu+eng  # German + English

# Install additional languages (macOS)
brew install tesseract-lang
```

**Model Management:**
```bash
# Download PaddleOCR models
python backend/scripts/download_paddle_models.py

# Migrate existing models to project directory
python backend/scripts/migrate_paddle_models.py
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
| `DEFAULT_TOP_K` | Default results count | `10` |
| `MAX_TOP_K` | Maximum results count | `25` |

> **Note:** The `DEFAULT_TOP_K` has been increased from 5 to 10 for better search coverage. This can be configured per-query in the chat UI or globally in Admin Settings.

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

### Advanced RAG Features

These settings are configurable via the Admin UI (Settings > RAG Features tab) or database.

#### Retrieval Settings

**⚡ NEW: Per-Query Override in Chat UI**

Users can now adjust the number of documents to search directly in the chat interface using the "Documents to Search" control. This allows fine-tuning retrieval on a per-query basis without changing global settings.

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| `rag.top_k` | Documents to retrieve per query | `10` | 3-25 |
| `rag.rerank_results` | Enable cross-encoder reranking | `true` | - |
| `rag.query_expansion_count` | Query variations to generate | `3` | 1-5 |
| `rag.similarity_threshold` | Minimum similarity score | `0.4` | 0.1-0.9 |

**How These Settings Improve Search Quality:**

1. **Documents to Retrieve (top_k):** Higher values cast a wider net, finding more potentially relevant documents. Use higher values (15-20) when searching across many collections without filters.

2. **Result Reranking:** Uses a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) to re-score results by semantic relevance. Significantly improves result ordering at the cost of ~100-200ms latency.

3. **Query Expansions:** Generates paraphrased versions of the query to improve recall. For example, "German verbs" might also search "verben in German" and "German verb conjugation".

4. **Similarity Threshold:** Filters out results below this score. Lower values (0.3-0.4) are better for OCR'd documents with potential text artifacts.

**Per-Query Override (Chat UI):**

Users can click the "Documents to Search" button in the chat header to:
- Adjust top_k from 3-25 for the current query
- Reset to use the admin-configured default
- See "Auto" when using the default setting

#### GraphRAG (Knowledge Graph)

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.graphrag_enabled` | Enable knowledge graph-based retrieval | `true` |
| `rag.graph_max_hops` | Max traversal depth for multi-hop reasoning (1-5) | `2` |
| `rag.graph_weight` | Weight for graph results in hybrid search (0-1) | `0.3` |
| `rag.entity_extraction_enabled` | Extract entities when processing documents | `true` |

**How GraphRAG Works:**
1. Entities (people, organizations, locations, concepts) are extracted from documents
2. Relationships between entities are identified and stored
3. Queries traverse the knowledge graph to find related information
4. Results combine vector similarity with graph-based relevance

#### Agentic RAG (Complex Query Handling)

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.agentic_enabled` | Enable agentic RAG for complex queries | `false` |
| `rag.agentic_max_iterations` | Max ReAct loop iterations (1-10) | `5` |
| `rag.auto_detect_complexity` | Auto-detect when to use agentic mode | `true` |

**How Agentic RAG Works:**
1. Complex queries are decomposed into sub-questions
2. ReAct loop: Reason → Act → Observe → Iterate
3. Each sub-question retrieves relevant context
4. Final answer synthesizes all retrieved information

#### Multimodal RAG (Images & Tables)

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.multimodal_enabled` | Enable image/table processing | `true` |
| `rag.vision_provider` | Vision model provider | `auto` |
| `rag.ollama_vision_model` | Ollama vision model name | `llava` |
| `rag.caption_images` | Generate captions for images | `true` |
| `rag.extract_tables` | Extract and structure tables | `true` |

**Vision Provider Options:**
- `auto` - Automatically select based on available providers
- `ollama` - Use Ollama with LLaVA (free, local)
- `openai` - Use OpenAI GPT-4V (requires API key)
- `anthropic` - Use Anthropic Claude Vision (requires API key)

#### Real-Time Updates

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.incremental_indexing` | Only re-index changed content | `true` |
| `rag.freshness_tracking` | Track document freshness | `true` |
| `rag.freshness_threshold_days` | Days before content is considered aging | `30` |
| `rag.stale_threshold_days` | Days before content is marked stale | `90` |

#### Query Suggestions

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.suggested_questions_enabled` | Show follow-up suggestions | `true` |
| `rag.suggestions_count` | Number of suggestions to show (1-5) | `3` |

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
