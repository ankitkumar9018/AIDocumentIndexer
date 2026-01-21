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
| `DEFAULT_PROCESSING_MODE` | Default mode (full/ocr/basic) | `full` |
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

#### HyDE (Hypothetical Document Embeddings)

HyDE improves retrieval for short, abstract queries by generating a hypothetical document that might contain the answer, then using that document's embedding for search.

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.hyde_enabled` | Enable HyDE for short/abstract queries | `true` |
| `rag.hyde_min_query_words` | Use HyDE only for queries shorter than this | `5` |

**How HyDE Works:**
1. Short queries like "marketing strategy" often don't match document vocabulary well
2. HyDE generates a hypothetical answer document: "A marketing strategy involves..."
3. The hypothetical doc is embedded and searched alongside the original query
4. Results are merged and deduplicated for better recall

**When HyDE Helps:**
- Short, conceptual queries (2-4 words)
- Abstract questions that don't use specific document terminology
- Queries where you're exploring a topic rather than finding a specific fact

#### CRAG (Corrective RAG)

CRAG automatically evaluates and corrects low-confidence search results, improving answer quality when initial retrieval is uncertain.

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.crag_enabled` | Enable automatic result correction | `true` |
| `rag.crag_confidence_threshold` | Trigger CRAG below this confidence (0-1) | `0.5` |

**How CRAG Works:**
1. After retrieval, documents are evaluated for relevance to the query
2. Documents are classified as: Correct, Ambiguous, or Incorrect
3. Based on classification, CRAG takes action:
   - **Mostly correct**: Use results as-is
   - **Mixed**: Filter to keep only correct + some ambiguous
   - **Mostly incorrect**: Show confidence warning to user

**Confidence Warnings:**
When CRAG detects low-confidence results, users see warnings:
- `< 0.5`: "Moderate confidence - results may be incomplete"
- `< 0.4`: "Low confidence - documents may not fully address your question"
- `< 0.2`: "Very low confidence - consider rephrasing your question"

#### Query Expansion & Parallel Search

Query expansion generates paraphrased versions of your query to improve recall.

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.query_expansion_enabled` | Enable query paraphrasing | `true` |
| `rag.query_expansion_count` | Number of query variations | `3` |
| `rag.parallel_query_search` | Search all variations in parallel | `true` |

**Performance Note:** Parallel search uses `asyncio.gather()` to search all expanded queries simultaneously, typically reducing total search time by 40-60% compared to sequential search.

#### Verification (Self-RAG)

Self-verification filters irrelevant documents before generating responses.

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.verification_enabled` | Enable document verification | `true` |
| `rag.verification_level` | Verification thoroughness | `quick` |
| `rag.dynamic_weighting_enabled` | Adjust vector/keyword weights by query type | `true` |

**Verification Levels:**
- `none`: Skip verification (fastest)
- `quick`: Fast heuristic-based filtering (default)
- `standard`: LLM-assisted relevance scoring
- `thorough`: Detailed LLM evaluation per document (slowest)

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
DATABASE_URL=sqlite:///backend/data/aidocindexer.db

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

---

## Automated Setup Configuration

The setup script (`scripts/setup.py`) reads from `scripts/dependencies.json` to determine:
- Required system dependencies
- Optional dependencies with descriptions and install commands
- Ollama models to pull
- Service ports and configuration

### Dependencies Configuration File

```json
{
  "system": {
    "required": ["python3", "node", "npm"],
    "optional": {
      "document_conversion": ["soffice"],
      "ocr": ["tesseract"],
      "media": ["ffmpeg"],
      "pdf": ["poppler"],
      "utilities": ["jq", "redis-server"],
      "distributed": ["ray"]
    }
  },
  "ollama": {
    "models": {
      "text": ["llama3.2:latest"],
      "embedding": ["nomic-embed-text:latest"],
      "vision": ["llava:latest"]
    }
  },
  "services": {
    "backend": {"port": 8000},
    "frontend": {"port": 3000},
    "redis": {"port": 6379},
    "ollama": {"port": 11434}
  }
}
```

### Graceful Service Fallbacks

The system handles missing optional services gracefully:

| Service | Fallback Behavior |
|---------|-------------------|
| **Ollama** | Uses cloud LLM providers (OpenAI, Anthropic) |
| **Redis** | Disables caching and Celery async tasks |
| **Celery** | Processing runs synchronously |
| **Vision Models** | Falls back to cloud vision APIs |
| **Ray** | Uses standard multiprocessing |

See [SETUP.md](./SETUP.md) for complete setup script documentation.

---

## Settings Presets API

Apply configuration presets programmatically:

```bash
# Speed preset (fast responses)
curl -X POST http://localhost:8000/api/v1/settings/apply-preset/speed

# Quality preset (best accuracy)
curl -X POST http://localhost:8000/api/v1/settings/apply-preset/quality

# Balanced preset (default)
curl -X POST http://localhost:8000/api/v1/settings/apply-preset/balanced

# Offline preset (local Ollama only)
curl -X POST http://localhost:8000/api/v1/settings/apply-preset/offline
```

### Preset Configurations

| Preset | top_k | rerank | query_expansion | images |
|--------|-------|--------|-----------------|--------|
| Speed | 5 | false | false | false |
| Quality | 15 | true | true | true |
| Balanced | 10 | true | true | true |
| Offline | 10 | false | false | false |

---

## Monitoring & Observability

### Sentry (Error Tracking)

| Variable | Description | Default |
|----------|-------------|---------|
| `SENTRY_DSN` | Sentry DSN for error tracking | None (disabled) |
| `SENTRY_TRACES_SAMPLE_RATE` | Performance tracing sample rate (0.0-1.0) | `0.1` |
| `SENTRY_PROFILES_SAMPLE_RATE` | Profiling sample rate (0.0-1.0) | `0.1` |

**Setup:**
1. Create a project at [sentry.io](https://sentry.io)
2. Copy your DSN from Project Settings > Client Keys
3. Add to `.env`:
```bash
SENTRY_DSN=https://your-key@sentry.io/project-id
```

**Features:**
- Automatic error capture with stack traces
- FastAPI, SQLAlchemy, Redis, httpx integrations
- Sensitive data filtering (API keys, passwords, tokens)
- Performance monitoring with traces and profiles

### Prometheus (Metrics)

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_METRICS` | Enable Prometheus metrics endpoint | `true` |

**Metrics Endpoint:** `GET /metrics`

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests by method, endpoint, status |
| `http_request_duration_seconds` | Histogram | Request latency |
| `http_requests_active` | Gauge | Currently active requests |
| `llm_requests_total` | Counter | LLM API calls by provider, model, status |
| `llm_tokens_total` | Counter | Token usage by provider, model, type |
| `llm_request_duration_seconds` | Histogram | LLM request latency |
| `documents_processed_total` | Counter | Documents processed by status, type |
| `embeddings_generated_total` | Counter | Embeddings generated by provider |

**Prometheus Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'aidocindexer'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

---

## Database Connector

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_CONNECTOR_MAX_ROWS` | Maximum rows returned from queries | `1000` |
| `DATABASE_CONNECTOR_TIMEOUT` | Query timeout in seconds | `30` |
| `DATABASE_CONNECTOR_CACHE_ENABLED` | Cache repeated NL queries | `true` |

### Supported Databases

| Type | Connection String Format |
|------|-------------------------|
| PostgreSQL | `postgresql://user:pass@host:5432/db` |
| MySQL | `mysql://user:pass@host:3306/db` |
| SQLite | `sqlite:///path/to/file.db` |
| MongoDB | `mongodb://user:pass@host:27017/db` |

### Security

All database credentials are encrypted at rest using the application's `SECRET_KEY`. Queries are validated to ensure:
- Only SELECT/WITH statements allowed (SQL databases)
- Only find/aggregate operations allowed (MongoDB)
- No DDL (DROP, CREATE, ALTER)
- No DML (INSERT, UPDATE, DELETE)
- Dangerous patterns blocked (SLEEP, BENCHMARK, $where, etc.)

See [FEATURES.md](./FEATURES.md#database-connector) for usage guide

---

## Contextual Chunking

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CONTEXTUAL_CHUNKING_ENABLED` | Enable contextual retrieval | `false` |
| `CONTEXT_GENERATION_PROVIDER` | LLM provider (ollama, openai) | `ollama` |
| `CONTEXT_GENERATION_MODEL` | Model for context generation | `llama3.2` |

### Settings (via Admin UI)

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.contextual_chunking_enabled` | Enable contextual enhancement | `false` |
| `rag.context_generation_provider` | LLM provider | `ollama` |
| `rag.context_generation_model` | Context model | `llama3.2` |

### How It Works

Contextual chunking prepends LLM-generated context to each chunk before embedding:

1. Document is chunked normally
2. For each chunk, an LLM generates 50-100 word context
3. Context is prepended to chunk content
4. Enhanced chunk is embedded

**Research shows 49-67% reduction in failed retrievals.**

See [FEATURES.md](./FEATURES.md#contextual-chunking) for full guide
