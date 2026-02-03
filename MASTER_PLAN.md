# AIDocumentIndexer - Master Development Plan

> **Last Updated:** 2026-02-02
> **Status:** Active Development
> **Version:** 2.0.0

## Executive Summary

AIDocumentIndexer is an enterprise-grade RAG (Retrieval-Augmented Generation) platform designed to be the best-in-class solution for document intelligence. The system consists of three main platforms:

1. **Web Application** - Enterprise-ready with 1000+ concurrent users
2. **Desktop Application** - Cross-platform with LOCAL MODE (offline) and SERVER MODE
3. **Browser Extension** - Chrome/Firefox extension for web capture and quick search

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Feature Modules](#feature-modules)
3. [Implementation Status](#implementation-status)
4. [Performance Optimizations](#performance-optimizations)
5. [Research & Inspirations](#research--inspirations)
6. [Technical Specifications](#technical-specifications)
7. [Development Roadmap](#development-roadmap)
8. [API Reference](#api-reference)
9. [Database Schema](#database-schema)
10. [Testing Strategy](#testing-strategy)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Web App    │  │  Desktop App │  │   Browser    │  │  External    │     │
│  │   (Next.js)  │  │  (Electron)  │  │   Extension  │  │  API Clients │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │                 │
          └─────────────────┴─────────────────┴─────────────────┘
                                    │
                            ┌───────▼───────┐
                            │  Load Balancer │
                            │  (nginx/HAProxy)│
                            └───────┬───────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼───────┐          ┌───────▼───────┐          ┌───────▼───────┐
│   Backend 1   │          │   Backend 2   │          │   Backend N   │
│   (FastAPI)   │          │   (FastAPI)   │          │   (FastAPI)   │
│   + uvloop    │          │   + uvloop    │          │   + uvloop    │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────┐
    │                               │                           │
┌───▼───┐  ┌───────┐  ┌────────┐  ┌▼──────┐  ┌────────┐  ┌─────▼─────┐
│Postgres│  │ Redis │  │Celery  │  │ Ray   │  │ Qdrant │  │ Milvus    │
│  DB    │  │ Cache │  │Workers │  │Cluster│  │ Vector │  │ (optional)│
└────────┘  └───────┘  └────────┘  └───────┘  └────────┘  └───────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Next.js 14, React 18, TypeScript | Web application UI |
| Backend | FastAPI, Python 3.11+, uvloop | API server |
| Database | PostgreSQL 15+ with pgvector | Primary data store |
| Vector Store | Qdrant / Milvus / PostgreSQL pgvector | Embedding storage |
| Cache | Redis Cluster | Response & session cache |
| Task Queue | Celery + Redis | Background job processing |
| ML Runtime | Ray | Distributed ML workloads |
| Search | Elasticsearch (optional) | Full-text search |

---

## Feature Modules

### Core Modules

| Module | Status | Description |
|--------|--------|-------------|
| **Authentication** | ✅ Complete | JWT + OAuth2, multi-tenant, RBAC |
| **Documents** | ✅ Complete | Upload, parse, chunk, embed |
| **Collections** | ✅ Complete | Organize documents into searchable groups |
| **Folders** | ✅ Complete | Hierarchical organization with permissions |
| **Chat** | ✅ Complete | RAG-powered conversational interface |
| **Knowledge Graph** | ✅ Complete | Entity & relationship extraction |
| **Skills** | ✅ Complete | AI-powered reusable capabilities |
| **Workflows** | ✅ Complete | Visual workflow automation |
| **Web Scraper** | ✅ Complete | Scrape and index web content |
| **Link Groups** | ✅ Complete | Organize URLs for batch scraping |

### Advanced Features

| Feature | Status | Description |
|---------|--------|-------------|
| **GraphRAG** | ✅ Complete | Knowledge graph-enhanced retrieval |
| **Hybrid Search** | ✅ Complete | Vector + BM25 keyword search |
| **Reranking** | 🔄 In Progress | Multi-stage reranking pipeline |
| **VLM Processing** | 🔄 In Progress | Vision-language model for images |
| **Entity Resolution** | 📋 Planned | Edit distance similarity matching |
| **Contextual Chunking** | ✅ Complete | Semantic-aware chunking |
| **Late Chunking** | ✅ Complete | Colbert-style late interaction |
| **HyDE** | ✅ Complete | Hypothetical document embeddings |

### Platform Extensions

| Platform | Status | Description |
|----------|--------|-------------|
| **Desktop App** | 📋 Planned | Electron with LOCAL/SERVER modes |
| **Browser Extension** | 📋 Planned | Chrome MV3 for web capture |
| **Mobile App** | 📋 Future | React Native iOS/Android |

---

## Implementation Status

### Phase 1: Core Platform (COMPLETED)
- [x] FastAPI backend with async support
- [x] PostgreSQL database with migrations
- [x] Document upload and processing pipeline
- [x] Vector embeddings with multiple providers
- [x] Basic RAG chat interface
- [x] User authentication and authorization
- [x] Multi-tenant organization support

### Phase 2: Advanced RAG (COMPLETED)
- [x] Knowledge Graph extraction
- [x] GraphRAG integration
- [x] Hybrid search (vector + keyword)
- [x] Response caching
- [x] Query decomposition
- [x] Corrective RAG
- [x] Self-RAG verification

### Phase 3: Automation (COMPLETED)
- [x] Skills system with custom prompts
- [x] Visual workflow builder
- [x] External agent import
- [x] API publishing (skills/workflows as endpoints)
- [x] Scheduled tasks

### Phase 4: Content Sources (COMPLETED)
- [x] Web scraper with subpage crawling
- [x] Link groups with batch scraping
- [x] Connectors (Notion, GitHub, Slack, etc.)
- [x] File watcher for local directories
- [x] Database connectors

### Phase 5: Performance & Scale (IN PROGRESS)
- [x] uvloop for 2-4x async performance
- [x] ORJSON for faster JSON serialization
- [x] Connection pooling optimization
- [x] Response caching with Redis
- [ ] Tiered reranking pipeline
- [ ] Entity resolution optimization
- [ ] Batch embedding optimization

### Phase 6: Desktop & Extensions (PLANNED)
- [ ] Electron desktop application
- [ ] LOCAL MODE with Ollama
- [ ] SERVER MODE with cloud sync
- [ ] Chrome extension (MV3)
- [ ] Firefox extension

---

## Performance Optimizations

### Python Performance Enhancements

#### 1. Async I/O Optimization
```python
# Already implemented: uvloop installation
import uvloop
uvloop.install()  # 2-4x faster event loop
```

#### 2. JSON Serialization
```python
# Already implemented: ORJSON
import orjson
# 20-50% faster than standard json
```

#### 3. Database Query Optimization
```python
# Connection pooling
SQLALCHEMY_DATABASE_URL = "postgresql+asyncpg://..."
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=30,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)
```

#### 4. Embedding Batch Processing
```python
# Batch embeddings to reduce API calls
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_BATCH_MAX_TOKENS = 8000

async def batch_embed(texts: List[str]) -> List[List[float]]:
    batches = chunk_by_token_limit(texts, EMBEDDING_BATCH_MAX_TOKENS)
    results = await asyncio.gather(*[embed_batch(b) for b in batches])
    return flatten(results)
```

#### 5. Caching Strategy
```python
# Multi-level caching
class CacheStrategy:
    L1_MEMORY = "memory"      # In-process LRU (< 1ms)
    L2_REDIS = "redis"        # Distributed cache (1-5ms)
    L3_DATABASE = "database"  # Persistent cache (5-20ms)
```

#### 6. Lazy Loading
```python
# Lazy module imports for faster startup
def get_heavy_service():
    from backend.services.heavy_module import HeavyService
    return HeavyService()
```

### Planned Optimizations

| Optimization | Expected Impact | Priority |
|-------------|-----------------|----------|
| Numpy/BLAS optimization | 30% faster vector ops | High |
| Cython hot paths | 50% faster text processing | Medium |
| Connection pooling tuning | 20% fewer DB connections | High |
| Query plan optimization | 40% faster complex queries | High |
| Embedding cache by hash | 80% fewer API calls | High |
| Async batch processing | 3x throughput increase | High |

---

## Research & Inspirations

### Analyzed Projects

| Project | Key Learnings | Integration Ideas |
|---------|---------------|-------------------|
| **RAGFlow** | Entity resolution, N-hop expansion | Edit distance similarity |
| **Kotaemon** | UMAP visualization, citation UI | Citation visualization |
| **Khoj** | Desktop app architecture, Ollama | LOCAL MODE design |
| **AnythingLLM** | Workspace isolation, 38+ providers | Multi-provider support |
| **Danswer** | Chrome extension, 60+ connectors | Browser extension design |
| **OpenClaw** | Hybrid search merge, SQLite-vec | Hybrid scoring weights |

### Code Patterns to Adopt

#### From RAGFlow: Entity Resolution
```python
import editdistance

def is_similar_entity(a: str, b: str, threshold: float = 0.3) -> bool:
    """Check if two entity names are similar using edit distance."""
    distance = editdistance.eval(a.lower(), b.lower())
    max_len = max(len(a), len(b))
    similarity = 1 - (distance / max_len)
    return similarity >= (1 - threshold)
```

#### From OpenClaw: Hybrid Search Merge
```python
def merge_hybrid_results(
    vector_results: List[Result],
    keyword_results: List[Result],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[Result]:
    """Merge vector and keyword search results with weighted scoring."""
    by_id = {}

    for r in vector_results:
        by_id[r.id] = {"vector_score": r.score, "keyword_score": 0, **r.metadata}

    for r in keyword_results:
        if r.id in by_id:
            by_id[r.id]["keyword_score"] = r.score
        else:
            by_id[r.id] = {"vector_score": 0, "keyword_score": r.score, **r.metadata}

    merged = [
        {
            "id": id,
            "score": vector_weight * data["vector_score"] + keyword_weight * data["keyword_score"],
            **data
        }
        for id, data in by_id.items()
    ]

    return sorted(merged, key=lambda x: x["score"], reverse=True)
```

#### From Khoj: Local LLM Integration
```python
class OllamaClient:
    """Client for local Ollama LLM."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(self, model: str, prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            return response.json()["response"]

    async def embed(self, model: str, text: str) -> List[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            return response.json()["embedding"]
```

---

## Technical Specifications

### Desktop Application (Electron)

```
desktop/
├── package.json
├── electron-builder.json
├── src/
│   ├── main/
│   │   ├── index.ts              # Main process entry
│   │   ├── window.ts             # Window management
│   │   ├── tray.ts               # System tray
│   │   ├── hotkey.ts             # Global shortcuts
│   │   ├── ipc.ts                # IPC handlers
│   │   └── server/
│   │       ├── embedded.ts       # Embedded Python server
│   │       └── ollama.ts         # Ollama management
│   ├── preload/
│   │   └── index.ts              # Context bridge
│   └── renderer/                 # Symlink to frontend/
├── resources/
│   ├── icon.icns                 # macOS icon
│   ├── icon.ico                  # Windows icon
│   └── icon.png                  # Linux icon
└── python/
    └── server.py                 # Bundled Python server
```

#### LOCAL MODE Features
- Embedded SQLite database
- Ollama for local LLM inference
- Local embedding model (gte-small, all-MiniLM)
- LanceDB for local vector storage
- File watcher for automatic indexing
- No internet required after initial setup

#### SERVER MODE Features
- Connect to remote AIDocumentIndexer server
- Sync local documents to cloud
- Use cloud LLM providers
- Share documents across devices
- Real-time collaboration

### Browser Extension (Chrome MV3)

```
browser-extension/
├── manifest.json
├── src/
│   ├── background/
│   │   ├── index.ts              # Service worker
│   │   ├── api.ts                # API client
│   │   └── storage.ts            # Chrome storage
│   ├── content/
│   │   ├── capture.ts            # Page capture
│   │   └── highlight.ts          # Text highlighting
│   ├── popup/
│   │   ├── index.html
│   │   ├── App.tsx               # Search interface
│   │   └── styles.css
│   ├── sidepanel/
│   │   ├── index.html
│   │   ├── Panel.tsx             # Chat interface
│   │   └── Messages.tsx
│   └── options/
│       ├── index.html
│       └── Options.tsx           # Settings
├── assets/
│   └── icons/
└── _locales/
    └── en/messages.json
```

#### Extension Features
- Right-click "Save to AIDocIndexer"
- Popup for quick search
- Side panel for full chat
- Selection → Search hotkey
- Auto-capture mode (optional)
- API key management

---

## Development Roadmap

### Q1 2026 (Current)

| Week | Focus | Deliverables |
|------|-------|--------------|
| W1-2 | Performance | Tiered reranking, entity resolution |
| W3-4 | VLM | Complete vision-language processor |
| W5-6 | Desktop MVP | Electron shell, LOCAL MODE |
| W7-8 | Extension MVP | Chrome extension, basic features |

### Q2 2026

| Week | Focus | Deliverables |
|------|-------|--------------|
| W1-4 | Desktop Polish | Full feature parity, auto-update |
| W5-8 | Extension Polish | Firefox support, advanced features |
| W9-12 | Scale Testing | 1000 user load tests, optimization |

### Q3 2026

| Week | Focus | Deliverables |
|------|-------|--------------|
| W1-4 | Mobile | React Native iOS app |
| W5-8 | Mobile | React Native Android app |
| W9-12 | Enterprise | SSO, audit logs, compliance |

---

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/documents` | POST | Upload document |
| `/api/v1/chat/query` | POST | RAG query |
| `/api/v1/skills/execute` | POST | Execute skill |
| `/api/v1/workflows/{id}/run` | POST | Run workflow |
| `/api/v1/knowledge-graph/query` | POST | GraphRAG query |
| `/api/v1/scraper/jobs` | POST | Create scrape job |
| `/api/v1/link-groups/groups` | GET/POST | Manage link groups |
| `/api/v1/integrations/execute` | POST | Run integration pipeline |

### External API (Published Skills/Workflows)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/external/skills/{id}/execute` | POST | Execute published skill |
| `/api/v1/external/workflows/{id}/execute` | POST | Execute published workflow |
| `/api/v1/external/api-keys` | GET/POST | Manage API keys |

---

## Database Schema

### Core Tables

```sql
-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    content_type VARCHAR(100),
    file_size BIGINT,
    status VARCHAR(50),
    organization_id UUID REFERENCES organizations(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chunks with embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    content TEXT NOT NULL,
    embedding VECTOR(768),  -- pgvector
    chunk_index INTEGER,
    metadata JSONB
);

-- Knowledge Graph entities
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    name_normalized VARCHAR(500),
    entity_type VARCHAR(50),
    properties JSONB,
    embedding VECTOR(768)
);

-- Link Groups
CREATE TABLE link_groups (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    color VARCHAR(20),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Saved Links
CREATE TABLE saved_links (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    title VARCHAR(500),
    group_id UUID REFERENCES link_groups(id),
    last_scraped_at TIMESTAMP,
    last_scrape_status VARCHAR(50),
    scrape_count INTEGER DEFAULT 0
);
```

---

## Testing Strategy

### Unit Tests
- pytest for Python backend
- Jest for TypeScript frontend
- Coverage target: 80%

### Integration Tests
- API endpoint tests
- Database integration tests
- External service mocks

### Load Tests
- k6 for load testing
- Target: 1000 concurrent users
- Target: 100 docs/minute processing

### E2E Tests
- Playwright for web app
- Spectron for desktop app

---

## Monitoring & Observability

### Metrics (Prometheus)
- Request latency (p50, p95, p99)
- Throughput (req/sec)
- Error rate
- Queue depth
- Cache hit rate

### Logging (Structured JSON)
- Request IDs for tracing
- User context
- Error stack traces
- Performance timing

### Alerts
- Error rate > 1%
- Latency p95 > 2s
- Queue depth > 1000
- Disk usage > 80%

---

## Security Considerations

### Authentication
- JWT tokens with refresh
- OAuth2 providers
- API key authentication for external access

### Authorization
- Role-based access control (RBAC)
- Folder-level permissions
- Organization isolation

### Data Protection
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII detection and masking
- Audit logging

---

## Contributing

### Code Style
- Python: Black + isort + flake8
- TypeScript: Prettier + ESLint
- Commit messages: Conventional Commits

### Pull Request Process
1. Create feature branch
2. Write tests
3. Update documentation
4. Request review
5. Squash and merge

---

## License

Proprietary - All rights reserved

---

## Changelog

### v2.0.0 (2026-02-02)
- Added Link Groups feature
- Added External Agent import for Skills
- Added Feature Synergy service
- Added Integration pipelines API
- Performance optimizations with uvloop/orjson

### v1.5.0 (2026-01-15)
- Added Workflow automation
- Added Skills system
- Added Knowledge Graph

### v1.0.0 (2025-12-01)
- Initial release
- Document upload and processing
- RAG chat interface
- Basic authentication
