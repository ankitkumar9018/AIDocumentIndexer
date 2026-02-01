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

### Operation-Level LLM Configuration

Different operations can use different LLM providers/models, configured via Admin UI:

**Admin UI:** Navigate to **Admin > Settings > LLM Configuration > Operation-Level Config**

| Operation | Description | Default |
|-----------|-------------|---------|
| `auto_tagging` | Automatic tag generation during upload | Uses default provider |
| `context_generation` | Contextual chunking context generation | Uses default provider |
| `entity_extraction` | Knowledge graph entity extraction | Uses default provider |
| `query_expansion` | Query paraphrasing for RAG fusion | Uses default provider |

**Example:** To configure auto-tagging to use Ollama instead of OpenAI:
1. Go to **Admin > Settings > Providers**
2. Scroll to **Operation-Level Config**
3. Set `auto_tagging` to use Ollama with `llama3.2` model

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

---

## Advanced RAG (Phase 66)

### User Personalization

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_USER_PERSONALIZATION` | Enable preference learning from feedback | `true` |

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.user_personalization_enabled` | Enable user preference learning | `true` |

### Adaptive RAG Routing

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.adaptive_routing_enabled` | Enable query-dependent strategy routing | `true` |
| `rag.routing_strategy` | Default routing (DIRECT/HYBRID/TWO_STAGE/AGENTIC/GRAPH_ENHANCED) | `auto` |

### RAG-Fusion

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.fusion_enabled` | Enable multi-query RAG-Fusion | `true` |
| `rag.fusion_query_count` | Number of query variations to generate | `3` |

### LazyGraphRAG

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.lazy_graphrag_enabled` | Enable query-time community summarization | `true` |
| `rag.lazy_graphrag_cache_ttl` | Cache TTL for community summaries (seconds) | `3600` |

### Dependency-Based Entity Extraction

| Setting | Description | Default |
|---------|-------------|---------|
| `kg.dependency_extraction_enabled` | Enable fast spaCy-based extraction | `true` |
| `kg.complexity_threshold` | Text complexity threshold for LLM fallback (0-1) | `0.7` |
| `kg.spacy_model` | spaCy model to use | `en_core_web_sm` |

### RAG Evaluation

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.evaluation_enabled` | Enable RAGAS evaluation metrics | `true` |
| `rag.evaluation_sampling_rate` | Fraction of queries to evaluate (0-1) | `0.1` |

---

## TTS Providers (Phase 66)

> **Note:** Chatterbox and CosyVoice are experimental open-source TTS providers that may require GPU and additional setup. Install optional dependencies:
> ```bash
> # From source (recommended):
> pip install git+https://github.com/resemble-ai/chatterbox.git
> pip install git+https://github.com/FunAudioLLM/CosyVoice.git
> ```
> The system gracefully falls back to HTTP API or other providers if packages are not installed.

### Chatterbox TTS

| Variable | Description | Default |
|----------|-------------|---------|
| `CHATTERBOX_ENABLED` | Enable Chatterbox TTS | `true` |
| `CHATTERBOX_EXAGGERATION` | Emotional exaggeration (0-1) | `0.5` |
| `CHATTERBOX_CFG_WEIGHT` | CFG weight for generation | `0.5` |

### CosyVoice2

| Variable | Description | Default |
|----------|-------------|---------|
| `COSYVOICE_ENABLED` | Enable CosyVoice2 TTS | `true` |

### TTS Provider Selection

| Setting | Description | Default |
|---------|-------------|---------|
| `tts.default_provider` | Default TTS provider | `openai` |
| `tts.ultra_fast_provider` | Ultra-fast TTS provider | `cosyvoice` |
| `tts.fallback_chain` | Fallback provider order | `["cosyvoice", "chatterbox", "fish_speech"]` |
| `tts.chatterbox_enabled` | Enable Chatterbox TTS | `true` |
| `tts.chatterbox_exaggeration` | Emotional exaggeration (0.0-1.0) | `0.5` |
| `tts.chatterbox_cfg_weight` | CFG weight for generation (0.0-1.0) | `0.5` |
| `tts.cosyvoice_enabled` | Enable CosyVoice2 TTS | `true` |
| `tts.fish_speech_enabled` | Enable Fish Speech TTS | `true` |

---

## LLM Resilience (Phase 70)

Settings for circuit breaker and retry logic to handle LLM provider outages gracefully.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `llm.circuit_breaker_threshold` | number | `5` | Consecutive failures before circuit opens (3-10) |
| `llm.circuit_breaker_recovery` | number | `60` | Seconds before circuit breaker attempts recovery (30-300) |
| `llm.max_retries` | number | `3` | Maximum retry attempts for transient LLM failures (1-5) |
| `llm.call_timeout` | number | `120` | Timeout for LLM calls in seconds (30-300) |

---

## Advanced Embedding Models (Phase 73-77)

Extended embedding provider configuration beyond the default OpenAI and Ollama models.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `embedding.provider` | string | `openai` | Embedding provider (openai, voyage, jina, cohere, gte, qwen3, gemini, bge-m3, ollama, auto) |
| `embedding.auto_select_enabled` | boolean | `false` | Auto-select embedding model based on content type (code, multilingual, etc.) |
| `embedding.jina_enabled` | boolean | `false` | Enable Jina Embeddings v3 (89 languages, flexible dimensions) |
| `embedding.jina_dimensions` | number | `1024` | Jina output dimensions (64-1024) |
| `embedding.cohere_enabled` | boolean | `false` | Enable Cohere Embed v3.5 (self-improving, compression support) |
| `embedding.cohere_model` | string | `embed-english-v3.0` | Cohere model (embed-english-v3.0, embed-multilingual-v3.0) |
| `embedding.gte_enabled` | boolean | `false` | Enable GTE-Multilingual embeddings (305M params, 768 dims) |
| `embedding.gemma_enabled` | boolean | `false` | Enable EmbeddingGemma (Google's specialized 768D model) |
| `embedding.stella_enabled` | boolean | `false` | Enable Stella v3 embeddings (69+ MTEB, 1024D/2048D) |
| `embedding.stella_model` | string | `stella-base` | Stella variant: stella-base (1024D), stella-large (2048D) |
| `embedding.dimension_reduction` | boolean | `false` | Enable dimension reduction for memory savings (PCA/MRL) |
| `embedding.target_dimensions` | number | `512` | Target dimensions after reduction (256, 512, 768) |

---

## DSPy Prompt Optimization (Phase 93)

DSPy enables automatic prompt optimization by compiling training examples into optimized prompts, improving RAG answer quality without manual prompt engineering.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.dspy_optimization_enabled` | boolean | `false` | Enable DSPy prompt optimization |
| `rag.dspy_default_optimizer` | string | `bootstrap_few_shot` | Default optimizer: bootstrap_few_shot (stable, 20+ examples) or miprov2 (100+ examples) |
| `rag.dspy_min_examples` | number | `20` | Minimum training examples required before optimization can run (5-500) |
| `rag.dspy_inference_enabled` | boolean | `false` | Use DSPy modules at inference time (default: compilation-only, exported as text prompts) |

**How DSPy Optimization Works:**
1. Training examples are collected from user queries and feedback
2. Admin triggers a compilation run from the Settings UI
3. DSPy optimizes prompts using the selected optimizer
4. Optimized prompts are exported and used for inference

---

## Content Freshness Scoring (Phase 95K)

Content freshness scoring adjusts search result rankings based on document age, boosting recent content and penalizing stale documents.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.content_freshness_enabled` | boolean | `false` | Enable content freshness scoring |
| `rag.freshness_decay_days` | number | `180` | Days after which content is considered stale (30-730) |
| `rag.freshness_boost_factor` | number | `1.05` | Score multiplier for documents updated in the last 30 days (1.0-1.5) |
| `rag.freshness_penalty_factor` | number | `0.95` | Score multiplier for documents older than freshness_decay_days (0.5-1.0) |

---

## Graph-O1 Efficient Reasoning (Phase 77)

Graph-O1 provides faster GraphRAG reasoning through beam search, achieving 3-5x speedup over standard graph traversal.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.graph_o1_enabled` | boolean | `false` | Enable Graph-O1 efficient reasoning (3-5x faster GraphRAG) |
| `rag.graph_o1_beam_width` | number | `5` | Beam width - parallel paths to explore (3-10) |
| `rag.graph_o1_confidence_threshold` | number | `0.7` | Confidence threshold for path pruning (0.5-0.9) |

---

## Agentic RAG Extensions (Phase 72)

Extended agentic RAG settings for parallel execution, DRAGIN/FLARE dynamic retrieval, and resource budgets.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.agentic_timeout_seconds` | number | `300` | Timeout for agentic operations in seconds (60-600) |
| `rag.agentic_max_parallel_queries` | number | `4` | Maximum concurrent sub-queries in agentic mode (2-8) |
| `rag.agentic_dragin_enabled` | boolean | `true` | Enable DRAGIN/FLARE dynamic retrieval - skip retrieval when LLM is confident |
| `rag.agentic_retrieval_threshold` | number | `0.7` | FLARE confidence threshold - skip retrieval above this score (0.5-0.9) |
| `rag.agentic_max_tokens` | number | `100000` | Maximum token budget per agentic query (50000-200000) |

---

## Two-Stage Retrieval & ColBERT Reranking

Two-stage retrieval uses fast ANN search followed by ColBERT token-level reranking for 15-30% better precision.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.two_stage_retrieval_enabled` | boolean | `false` | Enable two-stage retrieval (fast ANN + ColBERT reranking) |
| `rag.stage1_candidates` | number | `150` | Candidates to retrieve in stage 1 (50-300) |
| `rag.use_colbert_reranker` | boolean | `true` | Use ColBERT reranker in stage 2 (else cross-encoder) |

---

## Hierarchical Retrieval

Document-first retrieval strategy that ensures diverse results across large document collections.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.hierarchical_enabled` | boolean | `false` | Enable hierarchical retrieval (document-first strategy) |
| `rag.hierarchical_doc_limit` | number | `10` | Maximum documents in stage 1 (5-20) |
| `rag.hierarchical_chunks_per_doc` | number | `3` | Chunks per document in stage 2 (1-5) |

---

## Semantic Deduplication

Removes near-duplicate chunks from expanded query results to improve result diversity.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.semantic_dedup_enabled` | boolean | `true` | Enable semantic deduplication |
| `rag.semantic_dedup_threshold` | number | `0.95` | Similarity threshold for deduplication (0.0-1.0) |

---

## Knowledge Graph Integration

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.knowledge_graph_enabled` | boolean | `true` | Enable knowledge graph-enhanced retrieval (+15-20% query precision) |
| `rag.knowledge_graph_max_hops` | number | `2` | Maximum graph traversal depth (1-3) |

---

## Query Decomposition

Automatically decomposes complex multi-step queries into sub-questions for improved accuracy on comparison and aggregation queries.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.query_decomposition_enabled` | boolean | `true` | Enable query decomposition for complex queries |
| `rag.decomposition_min_words` | number | `10` | Minimum query word count to trigger decomposition (5-20) |

---

## Context Sufficiency & Hallucination Prevention

Controls how the system handles insufficient context, including abstention (saying "I don't know") and conflict detection.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.context_sufficiency_threshold` | number | `0.5` | Minimum coverage score for sufficient context (0-1) |
| `rag.abstention_threshold` | number | `0.3` | Coverage below this triggers "I don't know" response (0-1) |
| `rag.enable_abstention` | boolean | `true` | Allow system to say "I don't know" when context is insufficient |
| `rag.conflict_detection_enabled` | boolean | `true` | Detect conflicting information across sources |

---

## Answer Refinement (Phase 62/63)

Self-Refine and CRITIC strategies for iteratively improving answer quality.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.answer_refiner_enabled` | boolean | `false` | Enable answer refinement (+20% quality improvement) |
| `rag.answer_refiner_strategy` | string | `self_refine` | Strategy: self_refine (general), critic (tool-verified), cove (hallucination reduction) |
| `rag.answer_refiner_max_iterations` | number | `2` | Maximum refinement iterations (1-5) |

---

## TTT Context Compression (Phase 62/63)

Test-Time Training compression for handling very long contexts (2M+ tokens).

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.ttt_compression_enabled` | boolean | `false` | Enable TTT compression for 35x faster inference on 2M+ token contexts |
| `rag.ttt_compression_ratio` | number | `0.5` | Target compression ratio (0.3-0.8) |

---

## RAG Sufficiency Detection

Detects when existing context is sufficient, skipping unnecessary retrieval rounds (ICLR 2025).

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.sufficiency_checker_enabled` | boolean | `true` | Enable RAG sufficiency detection |
| `rag.sufficiency_threshold` | number | `0.7` | Confidence threshold for context sufficiency (0.5-0.9) |

---

## Tree of Thoughts

Multi-path reasoning for complex analytical queries.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.tree_of_thoughts_enabled` | boolean | `false` | Enable Tree of Thoughts for complex analytical queries |
| `rag.tot_max_depth` | number | `3` | Maximum reasoning tree depth (2-5) |
| `rag.tot_branching_factor` | number | `3` | Branching factor per thought node (2-5) |

---

## Advanced RAG Pipeline (Phase 65)

### Query Classification & Self-RAG

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.query_classifier_enabled` | boolean | `true` | Enable query classification for intent-based routing (+25% accuracy) |
| `rag.self_rag_enabled` | boolean | `true` | Enable Self-RAG with reflection (+18% factuality) |
| `rag.self_rag_max_iterations` | number | `3` | Maximum Self-RAG reflection iterations (1-5) |

### Speculative RAG

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.speculative_enabled` | boolean | `false` | Enable Speculative RAG for 50% latency reduction |
| `rag.speculative_num_drafts` | number | `3` | Number of parallel drafts to generate (2-5) |

### Streaming with Citations

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.streaming_citations_enabled` | boolean | `true` | Enable real-time citation highlighting during streaming |

### RAG-Fusion (Phase 66)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.rag_fusion_enabled` | boolean | `true` | Enable RAG-Fusion (multi-query with Reciprocal Rank Fusion) |
| `rag.rag_fusion_variations` | number | `4` | Number of query variations for RAG-Fusion (2-8) |

### LazyGraphRAG Extensions

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.lazy_graphrag_max_communities` | number | `5` | Maximum communities to summarize per query (1-10) |

### Step-Back Prompting

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.stepback_prompting_enabled` | boolean | `true` | Enable Step-Back prompting for complex analytical queries |
| `rag.stepback_max_background` | number | `3` | Maximum background chunks for step-back context (1-5) |

### Context Compression

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.context_compression_enabled` | boolean | `true` | Enable context compression to reduce token usage |
| `rag.context_compression_target_tokens` | number | `2000` | Target token count for compressed context (500-5000) |
| `rag.context_compression_use_llm` | boolean | `false` | Use LLM for semantic compression (slower but more accurate) |

### AttentionRAG Compression (Phase 77)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.attention_rag_enabled` | boolean | `false` | Enable AttentionRAG (6.3x better than LLMLingua) |
| `rag.attention_rag_mode` | string | `moderate` | Compression mode: light (1.25x), moderate (2x), aggressive (4x), extreme (6.6x), adaptive |
| `rag.attention_rag_unit` | string | `sentence` | Compression unit: token, sentence, paragraph |
| `rag.context_reorder_strategy` | string | `sandwich` | Reordering strategy: sandwich, front_loaded, alternating |

### RAG Evaluation (Phase 66)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.evaluation_enabled` | boolean | `false` | Enable automatic RAGAS evaluation |
| `rag.evaluation_sample_rate` | number | `0.1` | Fraction of queries to evaluate (0.0-1.0) |
| `rag.min_faithfulness_score` | number | `0.7` | Minimum faithfulness score - flags potential hallucinations (0.0-1.0) |

### Agent Memory & Caching (Phase 70)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.agent_memory_cache_max_size` | number | `1000` | Maximum entries in agent memory LRU cache (500-5000) |
| `rag.agent_memory_cache_ttl` | number | `3600` | TTL for agent memory cache entries in seconds (1800-7200) |
| `rag.generative_cache_max_size` | number | `10000` | Maximum entries in generative cache FAISS index (5000-50000) |
| `rag.semantic_cache_rebuild_threshold` | number | `100` | New entries before triggering FAISS index rebuild (50-500) |

### User Personalization (Phase 66)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.personalization_min_feedback` | number | `5` | Minimum feedback entries before adapting preferences (3-20) |
| `rag.personalization_profile_ttl_days` | number | `90` | Days to retain user profile data (30-365) |

### Hybrid Search Weights

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.vector_weight` | number | `0.7` | Weight for vector results in hybrid search (0-1) |

---

## Search Engine Quality (Phase 65)

### BM25 Scoring

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `search.bm25_enabled` | boolean | `true` | Enable BM25 scoring for keyword search |
| `search.bm25_k1` | number | `1.5` | Term frequency saturation (1.2-2.0) |
| `search.bm25_b` | number | `0.75` | Document length normalization (0.0-1.0) |

### Field Boosting

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `search.field_boosting_enabled` | boolean | `true` | Enable field-specific boosting |
| `search.title_boost` | number | `3.0` | Boost for section title matches (1.0-5.0) |
| `search.document_title_boost` | number | `2.5` | Boost for document title matches (1.0-5.0) |

### Learning to Rank

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `search.ltr_enabled` | boolean | `false` | Enable Learning-to-Rank (requires training data from click logs) |

### Spell Correction

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `search.spell_correction_enabled` | boolean | `true` | Enable spell correction for queries with no results |
| `search.spell_correction_max_distance` | number | `2` | Maximum edit distance (1-3) |

### Freshness Boosting

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `search.freshness_boost_enabled` | boolean | `true` | Boost recent documents in search results |
| `search.freshness_decay_rate` | number | `0.1` | Exponential decay rate for freshness (0.05-0.2) |

### Query Caching

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `search.query_cache_enabled` | boolean | `true` | Enable query result caching |
| `search.query_cache_ttl_seconds` | number | `3600` | Query cache TTL in seconds (300-86400) |

---

## Vector Store Scaling (Phase 65)

Settings for scaling vector operations to 1M+ documents.

### HNSW Index Optimization

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `vectorstore.hnsw_ef_construction` | number | `200` | ef_construction parameter (100-400, higher = better recall, slower build) |
| `vectorstore.hnsw_m` | number | `32` | M parameter - connections per node (16-64, higher = better recall, more memory) |
| `vectorstore.hnsw_ef_search` | number | `128` | ef_search parameter (64-256, higher = better recall, slower search) |

### Binary Quantization (Phase 92)

Binary quantization reduces memory usage by 32x, enabling million-scale vector search on commodity hardware.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `vectorstore.binary_quantization_enabled` | boolean | `false` | Enable binary quantization for 32x memory reduction |
| `vectorstore.quantization_rerank_multiplier` | number | `10` | Retrieve N times more candidates for reranking after quantized search (5-20) |

### Late Chunking

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `vectorstore.late_chunking_enabled` | boolean | `false` | Embed full doc first, preserves cross-chunk context (+15% recall) |

### GPU Acceleration

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `vectorstore.gpu_acceleration_enabled` | boolean | `false` | FAISS GPU acceleration via cuVS (12x faster indexing, 8x lower latency) |

---

## Interactive Database Querying (Phase 65)

### Text-to-SQL

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `database.text_to_sql_enabled` | boolean | `true` | Enable natural language to SQL conversion (86.6% accuracy) |
| `database.sql_validation_enabled` | boolean | `true` | Enable multi-layer SQL validation (syntax, security, cost) |
| `database.sql_injection_prevention` | boolean | `true` | Enable SQL injection prevention checks |
| `database.query_timeout_seconds` | number | `30` | Maximum query execution time in seconds (10-120) |
| `database.query_cost_limit` | number | `10000` | Maximum estimated query cost before warning |

### Auto-Visualization

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `database.auto_visualization_enabled` | boolean | `true` | Auto-generate charts from query results (LIDA-style) |
| `database.result_summarization_enabled` | boolean | `true` | Generate natural language summaries of results |

### Interactive Query Building

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `database.interactive_mode_enabled` | boolean | `true` | Enable interactive query building with clarification |
| `database.query_preview_enabled` | boolean | `true` | Show query preview and cost estimate before execution |

---

## Enterprise Security Features (Phase 65)

### Access Control

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `enterprise.access_control_enabled` | boolean | `false` | Enable Attribute-Based Access Control for document retrieval |
| `enterprise.access_control_mode` | string | `rbac` | Mode: rbac (role-based), abac (attribute-based), rebac (relationship-based) |

### Audit Logging

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `enterprise.audit_logging_enabled` | boolean | `true` | Enable comprehensive audit logging for compliance |
| `enterprise.audit_log_retention_days` | number | `365` | Retention period in days (30-2555 for 7-year GDPR) |

### PII Detection

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `enterprise.pii_detection_enabled` | boolean | `false` | Enable PII detection and masking in query results |
| `enterprise.pii_masking_mode` | string | `redact` | Masking mode: redact (remove), tokenize (replace), encrypt (reversible) |

### Multi-Tenant Isolation

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `enterprise.multi_tenant_enabled` | boolean | `false` | Enable multi-tenant data isolation |

---

## OCR Extended Settings (Phase 76)

### EasyOCR

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `ocr.easyocr.languages` | json | `["en"]` | Language codes for EasyOCR |
| `ocr.easyocr.use_gpu` | boolean | `true` | Use GPU acceleration for EasyOCR (if available) |

### Language Detection & Auto-Configuration

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `ocr.auto_detect_language` | boolean | `false` | Auto-detect document language for OCR |
| `ocr.default_language` | string | `eng` | Default OCR language code when auto-detect is disabled |
| `ocr.multi_language_enabled` | boolean | `false` | Enable multi-language OCR (e.g., eng+deu) |

---

## Document Processing Settings

### Fast Chunking (Chonkie)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `processing.fast_chunking_enabled` | boolean | `true` | Enable Chonkie fast chunking (33x faster than LangChain) |
| `processing.fast_chunking_strategy` | string | `auto` | Strategy: auto, token (fastest), sentence (fast), semantic (balanced), sdpm (best quality) |
| `processing.content_aware_chunking` | boolean | `true` | Content-aware auto chunking - detects code/tables/narrative |

### Hierarchical Chunking (Phase 76)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `processing.hierarchical_chunking_enabled` | boolean | `false` | Enable hierarchical chunking for large documents |
| `processing.hierarchical_threshold_chars` | number | `100000` | Character threshold to trigger hierarchical chunking (50000-200000) |
| `processing.hierarchical_levels` | number | `3` | Number of levels: 2 (summary+detail) or 3 (summary+section+detail) |

### Docling Parser

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `processing.docling_parser_enabled` | boolean | `false` | Enable Docling enterprise parser (97.9% table extraction accuracy) |

### Parallel Processing (Phase 71)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `processing.max_concurrent_pdf_pages` | number | `8` | Maximum concurrent PDF OCR pages (4-16) |
| `processing.max_concurrent_image_captions` | number | `4` | Maximum concurrent image captioning requests (2-8) |
| `processing.settings_cache_ttl` | number | `300` | TTL for settings cache in seconds (60-600) |

### Ray Distributed Processing

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `processing.ray_enabled` | boolean | `false` | Enable Ray for distributed parallel processing |
| `processing.ray_address` | string | `auto` | Ray cluster address (auto for local, ray://host:10001 for remote) |
| `processing.ray_num_cpus` | number | `4` | Maximum CPUs for Ray tasks (0 for all available) |
| `processing.ray_num_gpus` | number | `0` | Maximum GPUs for Ray tasks |
| `processing.ray_memory_limit_gb` | number | `8` | Memory limit per Ray worker in GB |

---

## Document Generation Settings

### Output Configuration

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `generation.default_format` | string | `docx` | Default format (docx, pptx, pdf, md, html) |
| `generation.include_images` | boolean | `true` | Include AI-generated or stock images |
| `generation.image_backend` | string | `picsum` | Image source (picsum, unsplash, pexels, openai, stability, automatic1111, disabled) |
| `generation.default_tone` | string | `professional` | Writing tone (professional, casual, academic, creative) |
| `generation.default_style` | string | `business` | Document style (business, academic, creative, technical) |
| `generation.max_sections` | number | `10` | Maximum sections to generate (1-20) |
| `generation.include_sources` | boolean | `true` | Include source citations |
| `generation.auto_charts` | boolean | `false` | Auto-generate charts from source data |
| `generation.chart_style` | string | `business` | Chart theme: business, academic, minimal |
| `generation.chart_dpi` | number | `150` | Chart resolution in DPI (100-300) |

### Quality Review

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `generation.enable_quality_review` | boolean | `true` | Enable LLM-based content review before rendering |
| `generation.min_quality_score` | number | `0.7` | Minimum quality score threshold (0.0-1.0) |

### Vision-Based Review (PPTX)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `generation.enable_vision_review` | boolean | `false` | Enable vision-based slide review (resource-intensive) |
| `generation.vision_review_model` | string | `auto` | Vision model (auto, claude-3-sonnet, gpt-4-vision, ollama-llava) |
| `generation.vision_review_all_slides` | boolean | `false` | Review all slides (vs. content slides only) |

### LLM Rewrite

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `generation.enable_llm_rewrite` | boolean | `true` | Use LLM to rewrite text exceeding constraints |
| `generation.llm_rewrite_model` | string | `auto` | LLM model for rewriting (auto, gpt-4o-mini, claude-3-haiku) |

---

## Web Scraping Settings (Database)

### Basic Scraping

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `scraping.use_crawl4ai` | boolean | `true` | Use Crawl4AI for JS-rendered pages |
| `scraping.headless_browser` | boolean | `true` | Run browser in headless mode |
| `scraping.timeout_seconds` | number | `30` | Page load timeout (10-120) |
| `scraping.extract_links` | boolean | `true` | Extract and index links from pages |
| `scraping.extract_images` | boolean | `true` | Extract image URLs from pages |
| `scraping.max_depth` | number | `2` | Maximum crawl depth (0-5) |
| `scraping.respect_robots_txt` | boolean | `true` | Respect robots.txt rules |
| `scraping.rate_limit_ms` | number | `1000` | Delay between requests in ms (500-5000) |

### Advanced Crawler (Phase 65)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `crawler.stealth_mode_enabled` | boolean | `true` | Enable stealth mode for anti-bot bypass |
| `crawler.magic_mode_enabled` | boolean | `true` | Enable Crawl4AI magic mode for advanced anti-detection |
| `crawler.llm_extraction_enabled` | boolean | `true` | Enable LLM-powered content extraction |
| `crawler.smart_extraction_enabled` | boolean | `true` | Auto-detect and extract key information |
| `crawler.max_pages_per_site` | number | `100` | Maximum pages per site (10-1000) |
| `crawler.user_agent_rotation_enabled` | boolean | `true` | Rotate user agents to avoid detection |

### Phase 96 Scraping Features

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `scraping.proxy_enabled` | boolean | `false` | Enable proxy rotation for web scraping. Uses a pool of proxy servers to distribute requests and avoid IP-based blocking |
| `scraping.proxy_list` | string | `""` | Comma-separated list of proxy URLs (e.g., `http://proxy1:8080,socks5://proxy2:1080`). Leave empty if proxy is disabled |
| `scraping.proxy_rotation_strategy` | string | `"round_robin"` | Proxy rotation strategy: `round_robin` cycles through proxies sequentially, `random` selects randomly each request |
| `scraping.jina_reader_fallback` | boolean | `true` | Use Jina Reader API (`r.jina.ai`) as fallback when Crawl4AI and basic HTTP scraping fail. Free tier: up to 1000 requests/day |
| `scraping.adaptive_crawling` | boolean | `false` | Enable adaptive crawling with confidence-based stopping. Automatically determines when a site has been sufficiently crawled based on content coverage and saturation |
| `scraping.crash_recovery_enabled` | boolean | `true` | Enable crash recovery for long-running crawl jobs. Persists crawl state to allow resuming after failures |

---

## Job Queue & Caching

### Celery/Redis Queue

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `queue.celery_enabled` | boolean | `false` | Enable Celery for async processing (requires Redis) |
| `queue.redis_url` | string | `redis://localhost:6379/0` | Redis connection URL |
| `queue.max_workers` | number | `4` | Maximum concurrent Celery workers |

### Embedding Cache

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `cache.embedding_cache_enabled` | boolean | `true` | Enable embedding deduplication cache |
| `cache.embedding_cache_ttl_days` | number | `7` | Embedding cache TTL in days |

### Distributed Cache (Phase 75)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `cache.distributed_enabled` | boolean | `true` | Enable distributed cache invalidation via Redis pub/sub |
| `cache.pubsub_enabled` | boolean | `true` | Enable Redis pub/sub listener for cross-instance sync |
| `cache.invalidation_broadcast` | boolean | `true` | Broadcast invalidations to other instances |

---

## Agent Settings

### Evaluation

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `agent.evaluation_enabled` | boolean | `false` | Enable agent evaluation metrics (Pass^k, hallucination detection) |

### Human-in-the-Loop (Phase 77)

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `agent.hitl_enabled` | boolean | `false` | Enable human-in-the-loop interrupt support |
| `agent.hitl_approval_timeout` | number | `300` | Timeout for user approval requests in seconds (60-600) |
| `agent.hitl_checkpoint_interval` | number | `5` | Create checkpoint every N agent steps (1-20) |
| `agent.hitl_critical_actions` | string | `delete,modify,send` | Action keywords requiring explicit approval |

---

## Smart Model Routing (Phase 97)

Routes RAG queries to cost-optimal LLM models based on query complexity. Achieves 40-70% LLM cost reduction on mixed workloads.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.smart_model_routing_enabled` | boolean | `false` | Enable smart model routing based on query complexity |
| `rag.smart_routing_simple_model` | string | `""` | Model for simple queries (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3-5-haiku-20241022`) |
| `rag.smart_routing_complex_model` | string | `""` | Model for complex queries (e.g., `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`) |

**Query Tiers:**
| Tier | When Used | Example Queries |
|------|-----------|-----------------|
| SIMPLE | Factual lookups, keyword queries, high confidence | "What is the definition of X?", "List the authors" |
| MODERATE | Standard queries (uses default model) | General questions, medium complexity |
| COMPLEX | Multi-hop reasoning, aggregation, creative, large context | "Compare A vs B across 15 documents", "Synthesize findings" |

---

## Embedding Inversion Defense (Phase 97)

Protects stored embeddings from text reconstruction attacks per OWASP LLM08:2025. Applies noise injection, dimension shuffling, and norm clipping while preserving search quality.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `security.embedding_defense_enabled` | boolean | `false` | Enable embedding inversion defense (recommended for sensitive data) |
| `security.defense_noise_scale` | number | `0.01` | Gaussian noise standard deviation (0.001-0.1). Higher = more privacy, slightly lower accuracy |
| `security.defense_clip_norm` | number | `1.0` | L2 norm cap for embeddings (0.5-2.0). Lower = more privacy |

**Environment Variable:**
| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_DEFENSE_SECRET` | Secret key for deterministic dimension shuffling | Random (regenerates on restart) |

**Defense Techniques:**
1. **Noise Injection**: Calibrated Gaussian noise degrades inversion fidelity
2. **Dimension Shuffle**: Secret permutation prevents mapping dimensions to linguistic features
3. **Norm Clipping**: Bounds information leakage per embedding

**Important**: Set `EMBEDDING_DEFENSE_SECRET` in production for stable permutations across restarts.

---

## Matryoshka Adaptive Retrieval (Phase 97)

Two-stage retrieval using Matryoshka embedding dimensionality reduction for 5-14x faster search.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `rag.matryoshka_retrieval_enabled` | boolean | `false` | Enable two-stage Matryoshka retrieval |
| `rag.matryoshka_shortlist_factor` | number | `5` | Multiplier for stage 1 candidates (top_k × factor) |
| `rag.matryoshka_fast_dims` | number | `128` | Dimensions for fast stage 1 search (64, 128, 256) |

**How It Works:**
1. **Stage 1 (Fast Pass)**: Search with truncated 128-dim embeddings (~12x faster than 1536-dim)
2. **Stage 2 (Precise Rerank)**: Recompute similarity using full embeddings on shortlisted candidates

**Supported Models** (Matryoshka-trained):
- OpenAI `text-embedding-3-small` / `text-embedding-3-large`
- Nomic `nomic-embed-text`
- Most modern embedding models with MRL training

**Performance**: 5-14x faster retrieval with <2% recall loss when shortlist_factor >= 5

---

## OpenTelemetry Tracing (Phase 97)

Distributed tracing for RAG pipeline operations using OpenTelemetry.

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `observability.tracing_enabled` | boolean | `false` | Enable OpenTelemetry tracing |
| `observability.tracing_sample_rate` | number | `0.1` | Head-based sampling rate (0.0-1.0). 0.1 = 10% of traces |
| `observability.otlp_endpoint` | string | `""` | OTLP/gRPC collector endpoint (e.g., `http://jaeger:4317`) |

**Environment Variables** (override DB settings):
| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | None |
| `OTEL_SERVICE_NAME` | Service name in traces | `aidocumentindexer` |

**Traced Operations:**
- `rag.query` - Full RAG query lifecycle
- `rag.retrieval` - Document retrieval stage
- `rag.reranking` - Reranking stage
- `rag.generation` - LLM generation stage

**Installation:**
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

**Collector Setup** (Jaeger example):
```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

---

## Frontend Admin Settings Tabs

The Admin Settings page (`/dashboard/admin/settings`) provides 19 configuration tabs for managing all aspects of the system:

| # | Tab | Description |
|---|-----|-------------|
| 1 | **Overview** | System status dashboard and quick settings |
| 2 | **Security** | Authentication, 2FA, audit logging, access control |
| 3 | **Notifications** | Processing alerts, cost alerts, email notifications |
| 4 | **RAG** | Retrieval settings, GraphRAG, CRAG, HyDE, query expansion |
| 5 | **Providers** | LLM provider configuration (OpenAI, Anthropic, Ollama) |
| 6 | **Models** | Model selection, embedding models, vision models |
| 7 | **Database** | Vector store, HNSW tuning, text-to-SQL, quantization |
| 8 | **OCR** | OCR provider, language, model management |
| 9 | **Audio** | TTS providers, voice settings, fallback chains |
| 10 | **Scraper** | Web scraping, Crawl4AI, stealth mode, crawl settings |
| 11 | **Generation** | Document generation, format, tone, quality review |
| 12 | **Analytics** | Usage metrics, cost tracking, query analytics |
| 13 | **Experiments** | Feature flags, A/B testing, experimental features |
| 14 | **Cache** | Embedding cache, distributed cache, Redis pub/sub |
| 15 | **Evaluation** | RAGAS metrics, faithfulness scoring, evaluation sampling |
| 16 | **Job Queue** | Celery workers, Redis queue, async processing |
| 17 | **Instructions** | System prompts, custom instructions, prompt templates |
| 18 | **Ingestion** | Knowledge graph extraction, spaCy, entity settings |
| 19 | **Embedding Dashboard** | Embedding model comparison, dimension analysis, provider status |
