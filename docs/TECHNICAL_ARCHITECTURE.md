# Technical Architecture Documentation

> Comprehensive technical reference for AIDocumentIndexer - Enterprise RAG System

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [API Endpoint Reference](#api-endpoint-reference)
5. [Service Layer Architecture](#service-layer-architecture)
6. [Database Schema](#database-schema)
7. [Scalability & Performance](#scalability--performance)
8. [Configuration Reference](#configuration-reference)

---

## System Overview

AIDocumentIndexer is an enterprise-grade RAG (Retrieval-Augmented Generation) system designed for intelligent document management and AI-powered search. The system processes, indexes, and enables natural language querying of documents across 25+ file formats.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 15 + shadcn/ui)                        │
│  Dashboard │ Chat Interface │ Upload Portal │ Document Creator │ Admin     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKEND (FastAPI + Python)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LangChain + LiteLLM                               │   │
│  │  RAG Chains │ Memory │ Agents │ 100+ LLM Providers                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   Document  │  │  Embedding  │  │     OCR     │  │  Multi-Agent    │   │
│  │  Processors │  │  Generation │  │  (PaddleOCR)│  │  Orchestrator   │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                     │
│  PostgreSQL + pgvector │ ChromaDB (local) │ Redis Cache │ File Storage     │
│  Row-Level Security (RLS) for Permission Enforcement                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Clear boundaries between frontend, API, services, and storage
2. **Provider Agnostic**: Support for 100+ LLM providers via LiteLLM
3. **Security First**: Row-level security, JWT authentication, audit logging
4. **Scalability**: Ray-powered parallel processing, horizontal scaling support
5. **Offline Capable**: ChromaDB + Ollama for fully local operation

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Frontend** | Next.js | 15 | Server-side rendering, React framework |
| | React | 19 | UI component library |
| | shadcn/ui | latest | Accessible component system |
| | Tailwind CSS | 3.4 | Utility-first styling |
| | TanStack Query | 5.x | Data fetching and caching |
| **API** | FastAPI | 0.110+ | High-performance async REST API |
| | Pydantic | 2.x | Data validation and serialization |
| | Python | 3.11+ | Backend runtime |
| **RAG** | LangChain | 0.3.x | Chains, retrievers, memory |
| | LangGraph | 0.2.x | Agent workflows |
| | LiteLLM | latest | Multi-provider LLM routing |
| **Vector Store** | pgvector | 0.7+ | PostgreSQL vector extension |
| | ChromaDB | 0.5+ | Local vector database |
| **Database** | PostgreSQL | 16 | Primary relational database |
| | SQLite | 3.x | Local development database |
| | Redis | 7+ | Caching and session storage |
| **Processing** | Ray | 2.x | Distributed task processing |
| | PaddleOCR | 2.x | Optical character recognition |
| | PyMuPDF | 1.24+ | PDF parsing |
| **Auth** | NextAuth.js | 5.x | Frontend authentication |
| | python-jose | 3.x | JWT token handling |

---

## Data Flow Diagrams

### 1. Document Upload Pipeline

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│   Upload     │───▶│   Validate    │───▶│   File Storage   │
│   API        │    │   & Dedup     │    │   (temp)         │
└──────────────┘    └───────────────┘    └──────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACKGROUND PROCESSING QUEUE                              │
│                                                                              │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────┐  ┌────────┐  ┌────────┐ │
│  │   Extract    │─▶│   Chunk    │─▶│  Preprocess │─▶│ Embed  │─▶│ Index  │ │
│  │  (PyMuPDF)   │  │ (1000 chr) │  │  (cleanup)  │  │(OpenAI)│  │(pgvec) │ │
│  └──────────────┘  └────────────┘  └─────────────┘  └────────┘  └────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                              ┌─────────────────────────┐
                              │   WebSocket Update      │
                              │   (Status: COMPLETED)   │
                              └─────────────────────────┘
```

**Processing Steps:**

1. **Upload API** (`/api/v1/upload/single`): Receives file, validates size (max 100MB), checks file type
2. **Validation**: SHA-256 hash for deduplication, malware scanning
3. **Extraction**: Format-specific extraction (PyMuPDF for PDF, python-pptx for PPTX, etc.)
4. **Chunking**: Splits text into 1000-character chunks with 200-char overlap
5. **Preprocessing**: Removes boilerplate, normalizes whitespace (10-30% size reduction)
6. **Embedding**: Generates vectors via OpenAI text-embedding-3-small (1536 dims)
7. **Indexing**: Stores vectors in pgvector/ChromaDB with metadata
8. **Notification**: WebSocket update to frontend with completion status

### 2. RAG Query Flow

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐    ┌──────────────────────────────────────────────┐
│   Mode Selection    │───▶│   RAG Mode / General Mode / Agent Mode       │
│   (chat/agent)      │    └────────────────────┬─────────────────────────┘
└─────────────────────┘                         │
                                                │
         ┌──────────────────────────────────────┼──────────────────────────┐
         │                                      │                          │
         ▼ [RAG Mode]                           ▼ [General Mode]           ▼ [Agent Mode]
┌──────────────────────┐            ┌─────────────────────┐    ┌─────────────────────┐
│   Query Expansion    │            │   Direct LLM Call   │    │   Manager Agent     │
│   (optional)         │            │   (no retrieval)    │    │   (Plan Creation)   │
└──────────┬───────────┘            └──────────┬──────────┘    └──────────┬──────────┘
           │                                   │                          │
           ▼                                   │                          ▼
┌──────────────────────┐                       │               ┌─────────────────────┐
│   Hybrid Search      │                       │               │   Worker Agents     │
│   Vector + Keyword   │                       │               │   (Research/Gen)    │
└──────────┬───────────┘                       │               └──────────┬──────────┘
           │                                   │                          │
           ▼                                   │                          │
┌──────────────────────┐                       │                          │
│   Access Tier        │                       │                          │
│   Filter             │                       │                          │
└──────────┬───────────┘                       │                          │
           │                                   │                          │
           ▼                                   │                          │
┌──────────────────────┐                       │                          │
│   Cross-Encoder      │                       │                          │
│   Reranking          │                       │                          │
└──────────┬───────────┘                       │                          │
           │                                   │                          │
           ▼                                   ▼                          ▼
┌──────────────────────┐            ┌─────────────────────┐    ┌─────────────────────┐
│   LLM Generation     │            │   LLM Response      │    │   Final Synthesis   │
│   with Context       │            │                     │    │                     │
└──────────┬───────────┘            └──────────┬──────────┘    └──────────┬──────────┘
           │                                   │                          │
           └───────────────────────────────────┼──────────────────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │   Streaming Response    │
                                  │   + Source Citations    │
                                  └─────────────────────────┘
```

**Search Components:**

| Component | Purpose | Location |
|-----------|---------|----------|
| Query Expansion | Generates alternative queries for better recall | `rag.py:query_expansion()` |
| Hybrid Search | Combines vector similarity + BM25 keyword search | `vectorstore.py:hybrid_search()` |
| Access Tier Filter | Enforces user permission level | `vectorstore.py:_filter_by_tier()` |
| Cross-Encoder Reranking | Re-scores results for precision | `vectorstore.py:rerank()` |

### 3. Agent Execution Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           AGENT ORCHESTRATOR                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────┐                                                         │
│  │   User Request    │                                                         │
│  └─────────┬─────────┘                                                         │
│            │                                                                    │
│            ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        MANAGER AGENT                                     │   │
│  │  • Analyze request for document keywords                                 │   │
│  │  • Create execution plan with dependencies                               │   │
│  │  • Estimate costs per step                                               │   │
│  │  • CRITICAL: Include Research step if documents needed                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                    │
│            ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        EXECUTION PLAN                                    │   │
│  │  Step 1: Research (search documents)  ──────────────────┐               │   │
│  │  Step 2: Generator (create content)   ◀─────────────────┘               │   │
│  │  Step 3: Critic (evaluate quality)    ◀───────────────────────┐         │   │
│  │  Step 4: Tool (generate PPTX)         ◀───────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                    │
│            ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        WORKER AGENTS                                     │   │
│  │                                                                          │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │   │
│  │  │ Research Agent │  │Generator Agent │  │  Critic Agent  │             │   │
│  │  │ • RAG Search   │  │ • LLM Create   │  │ • LLM-as-judge │             │   │
│  │  │ • Web Search   │  │ • Use Context  │  │ • Evaluate     │             │   │
│  │  │ • Sources      │  │ • Format Output│  │ • Feedback     │             │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘             │   │
│  │                                                                          │   │
│  │  ┌────────────────┐                                                     │   │
│  │  │  Tool Agent    │                                                     │   │
│  │  │ • PPTX/DOCX    │                                                     │   │
│  │  │ • PDF Gen      │                                                     │   │
│  │  │ • File Export  │                                                     │   │
│  │  └────────────────┘                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                    │
│            ▼                                                                    │
│  ┌───────────────────┐                                                         │
│  │   Streaming SSE   │ → content, sources, step_completed, done                │
│  └───────────────────┘                                                         │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Agent Types:**

| Agent | Role | Key Features |
|-------|------|--------------|
| **Manager** | Orchestrator | Task decomposition, dependency tracking, cost estimation |
| **Research** | Information Retrieval | RAG search, web scraping, source aggregation |
| **Generator** | Content Creation | LLM-powered writing with context from research |
| **Critic** | Quality Assurance | LLM-as-judge evaluation, improvement suggestions |
| **Tool** | File Operations | PPTX/DOCX/PDF generation, markdown formatting |

---

## API Endpoint Reference

### Overview

The AIDocumentIndexer API provides 197 endpoints across 11 categories. All endpoints (except auth) require JWT authentication.

**Base URL:** `http://localhost:8000/api/v1`

**Authentication:** `Authorization: Bearer <token>`

---

### Authentication Endpoints (10)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Authenticate user with email/password |
| POST | `/auth/register` | Register new user account |
| POST | `/auth/logout` | Invalidate current session |
| POST | `/auth/refresh` | Refresh access token |
| GET | `/auth/me` | Get current user profile |
| POST | `/auth/change-password` | Change user password |
| GET | `/auth/verify` | Verify token validity |
| POST | `/auth/forgot-password` | Request password reset |
| POST | `/auth/reset-password` | Reset password with token |
| GET | `/auth/oauth/{provider}` | OAuth authentication (Google, GitHub) |

**Key Request/Response:**

```json
// POST /auth/login
{
  "email": "user@example.com",
  "password": "password123"
}

// Response
{
  "access_token": "eyJhbG...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "role": "user",
    "access_tier": 30
  }
}
```

---

### Chat Endpoints (13)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/chat/sessions` | List user's chat sessions |
| POST | `/chat/sessions` | Create new chat session |
| GET | `/chat/sessions/{session_id}` | Get session with messages |
| DELETE | `/chat/sessions/{session_id}` | Delete chat session |
| PATCH | `/chat/sessions/{session_id}` | Update session title |
| POST | `/chat/completions` | Send message, get response |
| POST | `/chat/completions/stream` | Stream response (SSE) |
| GET | `/chat/sessions/{session_id}/export` | Export session as PDF/DOCX |
| POST | `/chat/sessions/{session_id}/clear` | Clear session messages |
| GET | `/chat/sessions/{session_id}/sources` | Get all sources used |
| POST | `/chat/feedback` | Submit response feedback |
| GET | `/chat/suggestions` | Get follow-up suggestions |
| GET | `/chat/history` | Get recent chat history |

**Key Parameters for `/chat/completions`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | string | required | User's question |
| `session_id` | string | null | Session for conversation history |
| `mode` | string | "rag" | `rag`, `general`, or `agent` |
| `stream` | boolean | false | Enable SSE streaming |
| `document_ids` | array | null | Limit search to specific docs |
| `collection` | string | null | Limit search to collection |
| `include_collection_context` | boolean | true | Show collection tags to LLM |
| `search_type` | string | "hybrid" | `vector`, `keyword`, or `hybrid` |
| `top_k` | int | 5 | Number of chunks to retrieve |

**Streaming Response Events (SSE):**

| Event Type | Data | Purpose |
|------------|------|---------|
| `content` | string | Streaming response text |
| `sources` | array | Document sources with metadata |
| `agent_step` | object | Agent progress update |
| `done` | object | Final response with full content |
| `error` | string | Error message |

---

### Document Endpoints (9)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents` | List documents with pagination |
| GET | `/documents/{document_id}` | Get document details |
| PATCH | `/documents/{document_id}` | Update document metadata |
| DELETE | `/documents/{document_id}` | Delete document |
| GET | `/documents/{document_id}/chunks` | Get document chunks |
| POST | `/documents/search` | Semantic + keyword search |
| POST | `/documents/{document_id}/reprocess` | Reprocess document |
| GET | `/documents/collections/list` | List all collections |
| POST | `/documents/bulk-delete` | Delete multiple documents |

**Query Parameters for `GET /documents`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `page_size` | int | 20 | Items per page (max 100) |
| `collection` | string | null | Filter by collection |
| `file_type` | string | null | Filter by file extension |
| `status` | string | null | Filter by processing status |
| `sort_by` | string | "created_at" | Sort field |
| `sort_order` | string | "desc" | `asc` or `desc` |
| `search` | string | null | Text search in name/content |

---

### Upload Endpoints (7)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload/single` | Upload single file |
| POST | `/upload/batch` | Upload multiple files |
| GET | `/upload/status/{file_id}` | Get processing status |
| GET | `/upload/queue` | Get processing queue |
| POST | `/upload/cancel/{file_id}` | Cancel processing |
| POST | `/upload/retry/{file_id}` | Retry failed upload |
| GET | `/upload/supported-types` | List supported file types |

**Upload Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | binary | required | File to upload |
| `collection` | string | null | Collection name |
| `access_tier` | int | 30 | Permission tier (1-100) |
| `enable_ocr` | bool | true | Enable OCR for images |
| `enable_image_analysis` | bool | true | Analyze embedded images |
| `smart_chunking` | bool | true | Use semantic chunking |
| `detect_duplicates` | bool | true | Skip duplicate files |
| `processing_mode` | string | "full" | `full`, `ocr`, `basic` |

**Processing Status Response:**

```json
{
  "file_id": "uuid",
  "status": "processing",
  "progress": 65,
  "stage": "embedding",
  "message": "Generating embeddings...",
  "error": null
}
```

---

### Agent Endpoints (27)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/agent/mode` | Get current execution mode |
| POST | `/agent/mode` | Set execution mode |
| POST | `/agent/mode/toggle` | Toggle agent mode on/off |
| GET | `/agent/preferences` | Get user preferences |
| PATCH | `/agent/preferences` | Update preferences |
| POST | `/agent/execute` | Execute agent request |
| POST | `/agent/execute/stream` | Stream agent execution |
| GET | `/agent/plans` | List execution plans |
| GET | `/agent/plans/{plan_id}` | Get plan details |
| POST | `/agent/plans/{plan_id}/approve` | Approve plan execution |
| POST | `/agent/plans/{plan_id}/cancel` | Cancel plan |
| GET | `/agent/status` | Get agent system status |
| GET | `/agent/agents` | List all agents |
| GET | `/agent/agents/{agent_id}` | Get agent details |
| GET | `/agent/agents/{agent_id}/metrics` | Get agent metrics |
| GET | `/agent/agents/{agent_id}/config` | Get agent config |
| PATCH | `/agent/agents/{agent_id}/config` | Update agent config |
| POST | `/agent/agents/{agent_id}/optimize` | Start prompt optimization |
| GET | `/agent/optimization/jobs` | List optimization jobs |
| GET | `/agent/optimization/jobs/{job_id}` | Get job details |
| POST | `/agent/optimization/jobs/{job_id}/approve` | Approve optimized prompt |
| POST | `/agent/optimization/jobs/{job_id}/reject` | Reject optimized prompt |
| GET | `/agent/trajectories` | List execution trajectories |
| GET | `/agent/trajectories/{trajectory_id}` | Get trajectory details |
| POST | `/agent/estimate` | Estimate execution cost |
| GET | `/agent/capabilities` | List available capabilities |
| GET | `/agent/health` | Agent system health check |

**Agent Preferences:**

```json
{
  "default_mode": "chat",
  "agent_mode_enabled": true,
  "auto_detect_complexity": true,
  "show_cost_estimation": true,
  "require_approval_above_usd": 1.0,
  "general_chat_enabled": true,
  "fallback_to_general": true
}
```

---

### Admin Endpoints (50+)

#### User Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/users` | List all users |
| GET | `/admin/users/{user_id}` | Get user details |
| PATCH | `/admin/users/{user_id}` | Update user |
| DELETE | `/admin/users/{user_id}` | Delete user |
| POST | `/admin/users/{user_id}/reset-password` | Reset user password |
| PATCH | `/admin/users/{user_id}/role` | Change user role |
| PATCH | `/admin/users/{user_id}/tier` | Change access tier |

#### Access Tiers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/tiers` | List access tiers |
| POST | `/admin/tiers` | Create new tier |
| PATCH | `/admin/tiers/{tier_id}` | Update tier |
| DELETE | `/admin/tiers/{tier_id}` | Delete tier |

#### System Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/settings` | Get all settings |
| PATCH | `/admin/settings` | Update settings |
| GET | `/admin/settings/{category}` | Get category settings |
| POST | `/admin/settings/reset` | Reset to defaults |

#### LLM Providers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/providers` | List LLM providers |
| POST | `/admin/providers` | Add provider |
| PATCH | `/admin/providers/{provider_id}` | Update provider |
| DELETE | `/admin/providers/{provider_id}` | Remove provider |
| POST | `/admin/providers/{provider_id}/test` | Test provider connection |
| GET | `/admin/models` | List available models |

#### Processing Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/processing/config` | Get processing config |
| PATCH | `/admin/processing/config` | Update config |
| GET | `/admin/processing/queue` | View processing queue |
| POST | `/admin/processing/queue/clear` | Clear queue |
| POST | `/admin/processing/reindex` | Reindex all documents |

#### System Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/stats` | System statistics |
| GET | `/admin/health` | Detailed health check |
| GET | `/admin/logs` | System logs |
| GET | `/admin/audit-logs` | Audit trail |
| GET | `/admin/metrics` | Performance metrics |

---

### Cost Tracking Endpoints (12)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/costs/usage` | Get cost usage for period |
| GET | `/costs/history` | Get cost history |
| GET | `/costs/current` | Get current period cost |
| GET | `/costs/dashboard` | Get dashboard data |
| GET | `/costs/breakdown` | Get cost breakdown by type |
| GET | `/costs/by-model` | Get cost by model |
| GET | `/costs/alerts` | List cost alerts |
| POST | `/costs/alerts` | Create cost alert |
| PATCH | `/costs/alerts/{alert_id}` | Update alert |
| DELETE | `/costs/alerts/{alert_id}` | Delete alert |
| POST | `/costs/estimate` | Estimate request cost |
| GET | `/costs/pricing` | Get model pricing |

**Cost Dashboard Response:**

```json
{
  "current_period": {
    "total_cost_usd": 12.45,
    "token_count": 1250000,
    "request_count": 456
  },
  "by_category": {
    "chat": 8.20,
    "embedding": 2.15,
    "agent": 2.10
  },
  "trend": "increasing",
  "budget_remaining": 87.55
}
```

---

### Web Scraper Endpoints (10)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/scraper/jobs` | Create scrape job |
| GET | `/scraper/jobs` | List scrape jobs |
| GET | `/scraper/jobs/{job_id}` | Get job status |
| POST | `/scraper/jobs/{job_id}/run` | Start scrape job |
| POST | `/scraper/jobs/{job_id}/cancel` | Cancel job |
| DELETE | `/scraper/jobs/{job_id}` | Delete job |
| POST | `/scraper/immediate` | Scrape URL immediately |
| POST | `/scraper/query` | Scrape and query URL |
| POST | `/scraper/extract-links` | Extract links from URL |
| GET | `/scraper/cache` | Get cached pages |

**Scrape Job Configuration:**

```json
{
  "urls": ["https://example.com"],
  "config": {
    "max_depth": 2,
    "same_domain_only": true,
    "extract_links": true,
    "wait_for_js": true,
    "timeout": 30000
  },
  "collection": "scraped-content",
  "access_tier": 30
}
```

---

### Prompt Templates Endpoints (8)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/templates` | List templates |
| POST | `/templates` | Create template |
| GET | `/templates/{template_id}` | Get template |
| PATCH | `/templates/{template_id}` | Update template |
| DELETE | `/templates/{template_id}` | Delete template |
| POST | `/templates/{template_id}/clone` | Clone template |
| GET | `/templates/categories` | List categories |
| POST | `/templates/{template_id}/test` | Test template |

---

### Document Generation Endpoints (11)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generation/jobs` | Create generation job |
| GET | `/generation/jobs` | List jobs |
| GET | `/generation/jobs/{job_id}` | Get job details |
| POST | `/generation/jobs/{job_id}/outline` | Generate outline |
| POST | `/generation/jobs/{job_id}/approve-outline` | Approve outline |
| POST | `/generation/jobs/{job_id}/generate` | Start generation |
| POST | `/generation/jobs/{job_id}/sections/{section_id}/feedback` | Section feedback |
| POST | `/generation/jobs/{job_id}/sections/{section_id}/revise` | Revise section |
| GET | `/generation/jobs/{job_id}/download` | Download document |
| DELETE | `/generation/jobs/{job_id}` | Cancel job |
| GET | `/generation/formats` | List output formats |

**Generation Job Request:**

```json
{
  "title": "Q4 Market Analysis Report",
  "description": "Comprehensive analysis of market trends",
  "source_document_ids": ["doc-1", "doc-2"],
  "output_format": "pptx",
  "tone": "professional",
  "length": "medium"
}
```

---

### Collaboration Endpoints (9)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/collaboration/sessions` | Create collaboration session |
| GET | `/collaboration/sessions` | List sessions |
| GET | `/collaboration/sessions/{session_id}` | Get session |
| POST | `/collaboration/sessions/{session_id}/run` | Run collaboration |
| POST | `/collaboration/sessions/{session_id}/stop` | Stop session |
| DELETE | `/collaboration/sessions/{session_id}` | Delete session |
| GET | `/collaboration/sessions/{session_id}/critiques` | Get critiques |
| GET | `/collaboration/modes` | List collaboration modes |
| POST | `/collaboration/estimate` | Estimate cost |

**Collaboration Modes:**

| Mode | Description |
|------|-------------|
| `debate` | Models argue different perspectives |
| `consensus` | Models work toward agreement |
| `critique` | One model critiques another |
| `synthesis` | Combine outputs from multiple models |

---

## Service Layer Architecture

### Core Services

| Service | File | Responsibility |
|---------|------|----------------|
| **Pipeline** | `services/pipeline.py` | End-to-end document processing |
| **RAG** | `services/rag.py` | Retrieval-augmented generation with Self-RAG verification |
| **RAG Verifier** | `services/rag_verifier.py` | Answer verification and confidence scoring |
| **Semantic Chunker** | `services/semantic_chunker.py` | Context-aware chunking with section headers |
| **LLM** | `services/llm.py` | Multi-provider LLM management |
| **Embeddings** | `services/embeddings.py` | Vector embedding generation |
| **VectorStore** | `services/vectorstore.py` | Vector similarity search |
| **Permissions** | `services/permissions.py` | Access control enforcement |
| **Cost Tracking** | `services/cost_tracking.py` | Usage and cost monitoring |
| **Audit** | `services/audit.py` | Action logging and compliance |

### Agent Services

| Service | File | Responsibility |
|---------|------|----------------|
| **Orchestrator** | `services/agents/orchestrator.py` | Agent coordination |
| **Manager Agent** | `services/agents/manager_agent.py` | Task planning and decomposition |
| **Worker Agents** | `services/agents/worker_agents.py` | Task execution (Generator, Critic, Research, Tool) |
| **Agent Base** | `services/agents/agent_base.py` | Common agent functionality |

### Processing Services

| Service | File | Responsibility |
|---------|------|----------------|
| **Universal Processor** | `processors/universal.py` | Multi-format extraction |
| **Chunker** | `processors/chunker.py` | Text splitting strategies (simple, semantic, hierarchical) |
| **Semantic Chunker** | `services/semantic_chunker.py` | Advanced chunking with contextual headers |
| **Text Preprocessor** | `services/text_preprocessor.py` | Text normalization |
| **Summarizer** | `services/summarizer.py` | Document summarization |

### RAG Enhancement Services

| Service | File | Responsibility |
|---------|------|----------------|
| **RAG Verifier** | `services/rag_verifier.py` | Self-RAG verification and confidence scoring |
| **Query Expander** | `services/query_expander.py` | Multi-query expansion for improved recall |
| **Knowledge Graph** | `services/knowledge_graph.py` | Entity extraction and graph-based retrieval |
| **Agentic RAG** | `services/agentic_rag.py` | Query decomposition and ReAct loop |
| **Multimodal RAG** | `services/multimodal_rag.py` | Image captioning and table extraction |
| **Real-Time Indexer** | `services/realtime_indexer.py` | Incremental indexing and freshness tracking |
| **Adaptive Router** | `services/adaptive_router.py` | Query-dependent strategy selection (Phase 66) |
| **Advanced RAG Utils** | `services/advanced_rag_utils.py` | RAG-Fusion, step-back prompting (Phase 66) |
| **LazyGraphRAG** | `services/lazy_graphrag.py` | Query-time community summarization (Phase 66) |
| **User Personalization** | `services/user_personalization.py` | Preference learning from feedback (Phase 66) |
| **Dependency Extractor** | `services/dependency_entity_extractor.py` | Fast spaCy-based entity extraction (Phase 66) |

#### Adaptive RAG Pipeline (Phase 66)

The adaptive router selects optimal retrieval strategy based on query analysis.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Adaptive RAG Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Query                                                             │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                    │
│  │ Query Analyzer  │                                                    │
│  │ - Complexity    │                                                    │
│  │ - Entity count  │                                                    │
│  │ - Intent type   │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Adaptive Router                                │   │
│  │                                                                   │   │
│  │  Simple ──▶ DIRECT (fast, single retrieval)                      │   │
│  │  Standard ──▶ HYBRID (vector + keyword)                          │   │
│  │  Complex ──▶ TWO_STAGE (retrieval + reranking)                   │   │
│  │  Multi-step ──▶ AGENTIC (decomposition + ReAct)                  │   │
│  │  Entity-rich ──▶ GRAPH_ENHANCED (knowledge graph)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ RAG-Fusion      │ ← Optional: generates 3-5 query variations        │
│  │ (RRF Merge)     │   and merges results                              │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ User            │ ← Personalizes response format based on           │
│  │ Personalization │   learned preferences                             │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ RAGAS Evaluation│ ← Context relevance, faithfulness, answer         │
│  │ (Sampling)      │   relevance metrics                               │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### GraphRAG Architecture

GraphRAG enhances retrieval by building a knowledge graph from document content.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GraphRAG Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Document Upload                                                        │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │ Entity          │───▶│ Relationship    │───▶│ Graph           │    │
│  │ Extraction      │    │ Detection       │    │ Storage         │    │
│  │ (LLM-based)     │    │                 │    │ (PostgreSQL)    │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│                                                                         │
│  Query Time                                                             │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │ Entity          │───▶│ Graph           │───▶│ Hybrid          │    │
│  │ Recognition     │    │ Traversal       │    │ Ranking         │    │
│  │ in Query        │    │ (Multi-hop)     │    │ (Vector+Graph)  │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Entity Types:**
- PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, PRODUCT, TECHNOLOGY

**Relationship Types:**
- WORKS_FOR, LOCATED_IN, PART_OF, RELATED_TO, CREATED_BY, OWNS, etc.

**Database Tables:**
- `entities` - Knowledge graph nodes with embeddings
- `entity_mentions` - Where entities appear in chunks
- `entity_relations` - Relationships between entities

#### Agentic RAG Architecture

Agentic RAG handles complex queries through iterative reasoning.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agentic RAG Flow                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Query                                                             │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                    │
│  │ Complexity      │──── Simple ────▶ Standard RAG                     │
│  │ Detection       │                                                    │
│  └────────┬────────┘                                                    │
│           │ Complex                                                     │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ Query           │     "What products does Company X sell            │
│  │ Decomposition   │      in European markets?"                        │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────┐               │
│  │ Sub-queries:                                         │               │
│  │  1. What products does Company X make?              │               │
│  │  2. Which markets does Company X operate in?        │               │
│  │  3. What is Company X's European presence?          │               │
│  └─────────────────────────────────────────────────────┘               │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────┐               │
│  │            ReAct Loop (max iterations)              │               │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │               │
│  │  │ Reason  │─▶│   Act   │─▶│ Observe │──┐         │               │
│  │  └─────────┘  └─────────┘  └─────────┘  │         │               │
│  │       ▲                                  │         │               │
│  │       └──────────────────────────────────┘         │               │
│  └─────────────────────────────────────────────────────┘               │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ Synthesize      │                                                    │
│  │ Final Answer    │                                                    │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Multimodal RAG Architecture

Processes images and tables within documents.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Multimodal Processing                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Document with Images/Tables                                            │
│       │                                                                 │
│       ├────────────────────────────────────────┐                       │
│       │                                        │                       │
│       ▼                                        ▼                       │
│  ┌─────────────────┐                    ┌─────────────────┐            │
│  │ Image           │                    │ Table           │            │
│  │ Detection       │                    │ Detection       │            │
│  └────────┬────────┘                    └────────┬────────┘            │
│           │                                      │                      │
│           ▼                                      ▼                      │
│  ┌─────────────────┐                    ┌─────────────────┐            │
│  │ Vision Model    │                    │ Table           │            │
│  │ (LLaVA/GPT-4V)  │                    │ Extraction      │            │
│  │ Captioning      │                    │ & Structuring   │            │
│  └────────┬────────┘                    └────────┬────────┘            │
│           │                                      │                      │
│           ▼                                      ▼                      │
│  ┌─────────────────────────────────────────────────────┐               │
│  │              Text Content + Captions + Tables        │               │
│  │                      (Unified Index)                 │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Vision Model Priority:**
1. Ollama (LLaVA) - Free, local
2. OpenAI (GPT-4V) - Paid, high quality
3. Anthropic (Claude Vision) - Paid, high quality

#### Real-Time Indexer Architecture

Enables incremental updates without full re-indexing.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Incremental Indexing                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Document Update                                                        │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                    │
│  │ Content Hash    │──── No Change ────▶ Skip (Already Indexed)        │
│  │ Comparison      │                                                    │
│  └────────┬────────┘                                                    │
│           │ Changed                                                     │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ Chunk-Level     │                                                    │
│  │ Diff Detection  │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────┐               │
│  │  Only Update Changed Chunks:                        │               │
│  │  - Delete removed chunks                            │               │
│  │  - Update modified chunks                           │               │
│  │  - Add new chunks                                   │               │
│  │  - Preserve unchanged chunks                        │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Freshness Tracking:**
- `freshness_threshold_days` (30): Content aging warning
- `stale_threshold_days` (90): Content marked as stale
- Freshness indicators shown in UI

#### TTS Provider Architecture (Phase 66)

The TTS service supports multiple providers with automatic fallback.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TTS Provider Chain                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Audio Request                                                          │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Provider Selection                             │   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │ CosyVoice2  │  │ Chatterbox  │  │ Fish Speech │              │   │
│  │  │ 150ms       │  │ Emotional   │  │ Multilingual│              │   │
│  │  │ Streaming   │  │ Expressive  │  │ Fast        │              │   │
│  │  │ Free        │  │ Free        │  │ Free        │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐                                │   │
│  │  │ OpenAI TTS  │  │ ElevenLabs  │                                │   │
│  │  │ High Quality│  │ Premium     │                                │   │
│  │  │ Paid        │  │ Paid        │                                │   │
│  │  └─────────────┘  └─────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                    │
│  │ Fallback Chain  │ ← Automatic fallback if primary fails             │
│  │ cosyvoice →     │                                                    │
│  │ chatterbox →    │                                                    │
│  │ fish_speech →   │                                                    │
│  │ openai          │                                                    │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Provider Comparison:**

| Provider | Latency | Quality | Cost | Best For |
|----------|---------|---------|------|----------|
| CosyVoice2 | 150ms | Good | Free | Real-time streaming |
| Chatterbox | 800ms | High | Free | Emotional expressiveness |
| Fish Speech | 200ms | Good | Free | Multilingual support |
| OpenAI TTS | 500ms | High | $0.015/1K | Production quality |
| ElevenLabs | 300ms | Highest | $0.03/1K | Premium audio |

#### RAG Verification Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `none` | No verification (fastest) | High-throughput, trusted queries |
| `quick` | Embedding-based relevance only | Default for most queries |
| `standard` | + LLM relevance check | Important queries |
| `thorough` | + Answer grounding verification | High-stakes responses |

#### Confidence Scoring

The RAG Verifier calculates confidence scores (0-1) based on:
- Average relevance score of retrieved chunks
- Ratio of relevant to total retrieved documents
- Number of supporting sources (more sources = higher confidence)

Confidence levels:
- **High** (80%+): Answer is well-supported by retrieved documents
- **Medium** (50-80%): Some relevant information found, may be incomplete
- **Low** (<50%): Limited source support, verification recommended

---

## Database Schema

### Core Tables

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USERS & AUTH                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  users                    │  access_tiers              │  sessions          │
│  ├─ id (UUID, PK)        │  ├─ id (UUID, PK)         │  ├─ id (UUID, PK)  │
│  ├─ email (unique)       │  ├─ name (varchar)        │  ├─ user_id (FK)   │
│  ├─ hashed_password      │  ├─ level (int)           │  ├─ token (varchar)│
│  ├─ full_name            │  ├─ description           │  ├─ expires_at     │
│  ├─ role (enum)          │  └─ is_default (bool)     │  └─ created_at     │
│  ├─ access_tier_id (FK)  │                           │                     │
│  ├─ is_active (bool)     │                           │                     │
│  └─ created_at           │                           │                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                             DOCUMENTS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  documents                     │  document_chunks                           │
│  ├─ id (UUID, PK)             │  ├─ id (UUID, PK)                          │
│  ├─ name (varchar)            │  ├─ document_id (FK)                       │
│  ├─ file_path (varchar)       │  ├─ content (text)                         │
│  ├─ file_type (varchar)       │  ├─ embedding (vector[1536])               │
│  ├─ file_size (bigint)        │  ├─ chunk_index (int)                      │
│  ├─ file_hash (varchar)       │  ├─ page_number (int)                      │
│  ├─ access_tier_id (FK)       │  ├─ metadata (jsonb)                       │
│  ├─ owner_id (FK → users)     │  └─ created_at                             │
│  ├─ tags (text[])             │                                            │
│  ├─ status (enum)             │                                            │
│  ├─ metadata (jsonb)          │                                            │
│  └─ created_at                │                                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               CHAT                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  chat_sessions                 │  chat_messages                             │
│  ├─ id (UUID, PK)             │  ├─ id (UUID, PK)                          │
│  ├─ user_id (FK)              │  ├─ session_id (FK)                        │
│  ├─ title (varchar)           │  ├─ role (enum: user/assistant)            │
│  ├─ mode (varchar)            │  ├─ content (text)                         │
│  ├─ metadata (jsonb)          │  ├─ sources (jsonb)                        │
│  └─ created_at                │  ├─ token_count (int)                      │
│                               │  ├─ cost_usd (decimal)                     │
│                               │  └─ created_at                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENTS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  agent_definitions             │  agent_execution_plans                     │
│  ├─ id (varchar, PK)          │  ├─ id (UUID, PK)                          │
│  ├─ name (varchar)            │  ├─ user_id (FK)                           │
│  ├─ description (text)        │  ├─ request (text)                         │
│  ├─ agent_type (enum)         │  ├─ steps (jsonb)                          │
│  ├─ is_active (bool)          │  ├─ status (enum)                          │
│  └─ config (jsonb)            │  ├─ estimated_cost_usd (decimal)           │
│                               │  └─ created_at                             │
│                               │                                            │
│  agent_trajectories            │  agent_prompt_versions                     │
│  ├─ id (UUID, PK)             │  ├─ id (UUID, PK)                          │
│  ├─ plan_id (FK)              │  ├─ agent_id (FK)                          │
│  ├─ agent_id (FK)             │  ├─ version (int)                          │
│  ├─ input (jsonb)             │  ├─ prompt_template (text)                 │
│  ├─ output (jsonb)            │  ├─ traffic_percentage (int)               │
│  ├─ reasoning (text)          │  ├─ performance_score (float)              │
│  ├─ token_usage (jsonb)       │  ├─ is_active (bool)                       │
│  ├─ duration_ms (int)         │  └─ created_at                             │
│  ├─ success (bool)            │                                            │
│  └─ created_at                │                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Scalability & Performance

### Current Capacity

| Metric | ChromaDB (Local) | pgvector (Production) |
|--------|------------------|----------------------|
| Documents | ~10,000 | ~100,000 |
| Chunks | ~100,000 | ~1,000,000 |
| Query Latency | 50-500ms | 20-200ms |
| Concurrent Users | 10-50 | 100+ |

### Performance Optimizations

| Feature | Status | Effect |
|---------|--------|--------|
| Text Preprocessing | ON by default | 10-30% size reduction |
| Batch Embedding | ON | 100 chunks per API call |
| Hybrid Search | ON | Better recall than pure vector |
| Cross-Encoder Reranking | ON | Improved precision |
| Adaptive OCR | ON | Reduced memory for large files |
| Document Summarization | OFF by default | Optional token reduction |
| Hierarchical Chunking | OFF by default | Better for 100+ page docs |
| Semantic Chunking | ON by default | Context-aware splitting |
| Contextual Headers | ON by default | Improved retrieval accuracy |
| Self-RAG Verification | ON (quick mode) | Filters irrelevant chunks |
| Confidence Scoring | ON | Transparency for users |

### Recommended Scaling Path

1. **Development** (< 1,000 docs): ChromaDB + SQLite
2. **Small Team** (1,000-10,000 docs): ChromaDB + PostgreSQL
3. **Production** (10,000-100,000 docs): pgvector + PostgreSQL
4. **Enterprise** (100,000+ docs): Qdrant/Milvus + PostgreSQL cluster

---

## Configuration Reference

### Environment Variables by Category

#### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | AIDocumentIndexer | Application name |
| `APP_ENV` | development | Environment (development/staging/production) |
| `SECRET_KEY` | - | JWT signing key (min 32 chars) |
| `DEBUG` | false | Enable debug mode |

#### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_TYPE` | postgresql | Database type (postgresql/sqlite/mysql) |
| `DATABASE_URL` | - | Full database connection string |
| `VECTOR_STORE_BACKEND` | auto | Vector store (auto/pgvector/chroma) |
| `CHROMA_PERSIST_PATH` | ./chroma_db | ChromaDB storage path |

#### LLM Providers

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama server URL |
| `DEFAULT_CHAT_MODEL` | gpt-4o | Default chat model |
| `DEFAULT_EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |

#### Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UPLOAD_SIZE_MB` | 100 | Maximum file upload size |
| `CHUNK_SIZE` | 1000 | Default chunk size (characters) |
| `CHUNK_OVERLAP` | 200 | Chunk overlap (characters) |
| `ENABLE_OCR` | true | Enable OCR for images |
| `ENABLE_PREPROCESSING` | true | Enable text preprocessing |
| `ENABLE_SUMMARIZATION` | false | Enable document summarization |

#### Ray Cluster

| Variable | Default | Description |
|----------|---------|-------------|
| `RAY_ADDRESS` | auto | Ray cluster address |
| `RAY_NUM_CPUS` | 4 | CPUs per worker |
| `RAY_NUM_GPUS` | 0 | GPUs per worker |

#### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET_KEY` | - | JWT signing secret |
| `JWT_ALGORITHM` | HS256 | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 1440 | Token expiration (24 hours) |

---

## Appendix

### Supported File Types

| Category | Extensions |
|----------|------------|
| Documents | .pdf, .docx, .doc, .odt, .rtf |
| Presentations | .pptx, .ppt, .odp, .key |
| Spreadsheets | .xlsx, .xls, .csv, .ods |
| Images | .png, .jpg, .jpeg, .tiff, .bmp, .webp |
| Text | .txt, .md, .rst, .html, .xml, .json |
| Email | .eml, .msg |
| Archives | .zip (auto-extract) |

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid/missing token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 413 | Payload Too Large - File exceeds limit |
| 422 | Validation Error - Schema mismatch |
| 429 | Rate Limited - Too many requests |
| 500 | Internal Server Error |

### Rate Limits

| Endpoint Type | Authenticated | Unauthenticated |
|---------------|---------------|-----------------|
| Standard API | 100 req/min | 10 req/min |
| Upload | 10 req/min | N/A |
| Agent Execute | 20 req/min | N/A |
| Chat Stream | 30 req/min | N/A |

---

## Recent Architecture Updates (Phase 78-83, January 2026)

### Pipeline Integration (Phase 81)

The RAG pipeline now integrates the following services in order:

```
Query → Adaptive Router → Retrieval Strategy
      → KG Query Expansion → Graph-O1 Reasoning (optional)
      → Hybrid Search → Tiered Reranking (BM25→CrossEncoder→ColBERT→LLM)
      → Context Compression (LLMLingua / AttentionRAG)
      → Sufficiency Check → LLM Generation (with Anthropic prompt caching)
      → Generative Cache → Answer Refinement → Response
```

### Key Services Added

| Service | File | Purpose |
|---------|------|---------|
| AttentionRAG | `attention_rag.py` | 6.3x compression via attention scores |
| Graph-O1 | `graph_o1.py` | Beam search reasoning over KG |
| Human-in-the-Loop | `human_in_loop.py` | Agent workflow approval interrupts |
| Anthropic Prompt Caching | `rag.py` | 50-60% cost savings on Claude calls |
| OpenAI Structured Outputs | `llm.py` | JSON schema-validated extraction |

### Stability Improvements

- **Session LLM cache**: TTL (1 hour) + max size (200) prevents memory leaks
- **FAISS concurrency**: `asyncio.Lock` prevents concurrent index rebuilds
- **Settings cache invalidation**: All settings caches have 5-minute TTL
- **Entity graph safety**: Cycle detection prevents infinite traversal loops
- **Hash upgrade**: Cache keys use SHA-256 instead of MD5

### Test Coverage

- Import smoke test (`tests/test_imports.py`): Verifies all 145 service modules and 48 route modules import cleanly
- CI command: `pytest backend/tests/test_imports.py -v`

---

*Last updated: 2026-01-25*
