# System Architecture

![System Architecture Overview](images/system-overview.png)

## High-Level Overview

```mermaid
graph TB
    subgraph "Frontend (Next.js)"
        UI[Chat UI / Dashboard]
        API_CLIENT[API Client<br/>React Query Hooks]
    end

    subgraph "Backend (FastAPI)"
        ROUTES[API Routes<br/>chat / documents / admin]
        AUTH[Auth Middleware<br/>Bearer Token / OIDC]

        subgraph "Core Services"
            RAG[RAG Service]
            LLM_SVC[LLM Service]
            EMBED[Embedding Service]
            PIPELINE[Document Pipeline]
            KG[Knowledge Graph]
            MEMORY[Session Memory]
        end

        subgraph "Retrieval Layer"
            HYBRID[Hybrid Retriever]
            VECTOR[Vector Store<br/>ChromaDB / PGVector]
            BM25[BM25 Sparse Index]
            CACHE[Semantic Cache<br/>FAISS + LRU]
        end

        subgraph "Intelligence Layer"
            ROUTER[Adaptive Router]
            CLASSIFIER[Query Classifier]
            VERIFIER[Self-RAG Verifier]
            EXPANDER[Query Expander]
        end
    end

    subgraph "External Services"
        OLLAMA[Ollama<br/>Local LLM]
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
    end

    subgraph "Storage"
        POSTGRES[(PostgreSQL<br/>+ pgvector)]
        REDIS[(Redis Cache)]
        FILES[File Storage<br/>./storage/documents]
    end

    UI --> API_CLIENT
    API_CLIENT -->|HTTP/SSE| ROUTES
    ROUTES --> AUTH
    AUTH --> RAG
    AUTH --> PIPELINE

    RAG --> MEMORY
    RAG --> CACHE
    RAG --> CLASSIFIER
    RAG --> ROUTER
    RAG --> HYBRID
    RAG --> LLM_SVC
    RAG --> VERIFIER

    HYBRID --> VECTOR
    HYBRID --> BM25
    HYBRID --> KG

    VECTOR --> EMBED
    EMBED --> OLLAMA
    EMBED --> OPENAI

    LLM_SVC --> OLLAMA
    LLM_SVC --> OPENAI
    LLM_SVC --> ANTHROPIC

    PIPELINE --> POSTGRES
    PIPELINE --> FILES
    VECTOR --> POSTGRES
    MEMORY --> POSTGRES
    CACHE --> REDIS
```

## Component Map

| Component | File | Purpose |
|-----------|------|---------|
| **API Routes** | `backend/api/routes/chat.py` | Chat completions, streaming |
| | `backend/api/routes/documents.py` | CRUD, upload, preview, download |
| | `backend/api/routes/admin.py` | Settings, storage stats, enhancement |
| | `backend/api/routes/connectors.py` | External source connectors |
| | `backend/api/routes/memory.py` | User memory management |
| **RAG Service** | `backend/services/rag.py` | Orchestrates the full RAG pipeline |
| **LLM Service** | `backend/services/llm.py` | Model management, invocation, circuit breaker |
| **Embeddings** | `backend/services/embeddings.py` | Text-to-vector encoding |
| **Hybrid Retriever** | `backend/services/hybrid_retriever.py` | Dense + sparse + ColBERT fusion |
| **Vector Store** | `backend/services/vectorstore_local.py` | ChromaDB/PGVector CRUD |
| **Session Memory** | `backend/services/session_memory.py` | Conversation history + query rewriting |
| **Semantic Cache** | `backend/services/semantic_cache.py` | FAISS-based query dedup |
| **Text Preprocessor** | `backend/services/text_preprocessor.py` | Spell correction, normalization |
| **Document Pipeline** | `backend/services/pipeline.py` | Parse → chunk → embed → index |
| **Knowledge Graph** | `backend/services/knowledge_graph.py` | Entity/relation extraction |
| **Settings** | `backend/services/settings.py` | Runtime config from DB |
| **Connector Scheduler** | `backend/services/connectors/scheduler.py` | External source sync |

## Data Flow Diagram

![Data Flow](images/data-flow.png)

```mermaid
graph LR
    subgraph "Document Ingestion"
        UPLOAD[Upload API<br/>POST /documents]
        CONNECTOR[Connector Sync<br/>Google Drive / Notion]
    end

    subgraph "Processing Pipeline"
        PARSE[Parse<br/>PDF/DOCX/PPTX/TXT]
        CHUNK[Chunk<br/>Semantic + Sliding Window]
        EMBED_DOC[Embed<br/>Generate Vectors]
        INDEX[Index<br/>ChromaDB + BM25]
    end

    subgraph "Optional Enhancement"
        ENHANCE[Enhance<br/>Summaries / Keywords]
        KG_EXTRACT[KG Extract<br/>Entities / Relations]
        IMG_ANALYZE[Image Analysis<br/>Vision Model]
    end

    subgraph "Query Pipeline"
        QUERY[User Query]
        RETRIEVE[Retrieve<br/>Hybrid Search]
        GENERATE[Generate<br/>LLM + Context]
        RESPOND[Response<br/>+ Sources + Confidence]
    end

    UPLOAD --> PARSE
    CONNECTOR --> PARSE
    PARSE --> CHUNK
    CHUNK --> EMBED_DOC
    EMBED_DOC --> INDEX

    INDEX -.-> ENHANCE
    INDEX -.-> KG_EXTRACT
    INDEX -.-> IMG_ANALYZE

    QUERY --> RETRIEVE
    RETRIEVE -->|Top-K Chunks| GENERATE
    GENERATE --> RESPOND
```

## Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as Auth Middleware
    participant B as Backend

    U->>F: Login / Access App

    alt DEV_MODE=true
        F->>A: Bearer dev-token
        A->>A: Bypass auth, create dev user context
    else Production
        F->>A: Bearer JWT / OIDC Token
        A->>A: Validate token, extract user context
    end

    A->>B: UserContext (user_id, org_id, tier, roles)
    B->>B: Check document access tier
    B->>B: Apply org-level RLS filtering
    B-->>F: Response (filtered by permissions)
```

## Database Schema (Core Tables)

```mermaid
erDiagram
    DOCUMENT {
        uuid id PK
        string title
        string filename
        string file_type
        int file_size
        string file_path
        string processing_status
        string source_url
        string source_type
        bool is_stored_locally
        json upload_source_info
        json enhanced_metadata
        int access_tier
        uuid organization_id FK
        timestamp created_at
    }

    CHUNK {
        uuid id PK
        uuid document_id FK
        string content
        int chunk_index
        int page_number
        string content_type
        json metadata
    }

    ENTITY {
        uuid id PK
        uuid document_id FK
        string name
        string entity_type
        float confidence
    }

    ENTITY_RELATION {
        uuid id PK
        uuid source_entity_id FK
        uuid target_entity_id FK
        string relation_type
    }

    CHAT_SESSION {
        uuid id PK
        string title
        string user_id
        timestamp created_at
    }

    CHAT_MESSAGE {
        uuid id PK
        uuid session_id FK
        string role
        text content
        json source_chunks
        string model_used
        float confidence_score
    }

    DOCUMENT ||--o{ CHUNK : contains
    DOCUMENT ||--o{ ENTITY : has
    ENTITY ||--o{ ENTITY_RELATION : participates
    CHAT_SESSION ||--o{ CHAT_MESSAGE : contains
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Single Machine (Dev/Small)"
        NEXT[Next.js :3000]
        FASTAPI[FastAPI :8000]
        PG[PostgreSQL :5432]
        OLLAMA_LOCAL[Ollama :11434]
        REDIS_LOCAL[Redis :6379]
    end

    subgraph "Production (Recommended)"
        LB[Load Balancer<br/>nginx / Caddy]

        subgraph "App Tier"
            NEXT_P[Next.js<br/>2+ instances]
            FASTAPI_P[FastAPI<br/>2+ instances<br/>uvicorn workers]
        end

        subgraph "Data Tier"
            PG_P[(PostgreSQL<br/>+ pgvector<br/>+ read replicas)]
            REDIS_P[(Redis Cluster)]
        end

        subgraph "AI Tier"
            OLLAMA_P[Ollama<br/>GPU server]
            OPENAI_P[OpenAI API<br/>fallback]
        end
    end

    LB --> NEXT_P
    LB --> FASTAPI_P
    FASTAPI_P --> PG_P
    FASTAPI_P --> REDIS_P
    FASTAPI_P --> OLLAMA_P
    FASTAPI_P --> OPENAI_P
```
