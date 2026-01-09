# AIDocumentIndexer - Developer Onboarding Guide

## Quick Start for New Developers

Welcome to AIDocumentIndexer! This guide will help you understand the codebase architecture, key classes, and project flow so you can start contributing quickly.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Backend Deep Dive](#backend-deep-dive)
   - [API Layer](#api-layer)
   - [Service Layer](#service-layer)
   - [Database Layer](#database-layer)
   - [Agent System](#agent-system)
5. [Frontend Deep Dive](#frontend-deep-dive)
6. [Data Flow](#data-flow)
7. [Key Classes Reference](#key-classes-reference)
8. [Common Development Tasks](#common-development-tasks)
9. [Debugging Guide](#debugging-guide)

---

## Project Overview

AIDocumentIndexer is an intelligent document archive system with RAG (Retrieval-Augmented Generation) capabilities. It allows users to:

- **Upload documents** (PDF, DOCX, PPTX, images, etc.)
- **Index and search** using hybrid vector + keyword search
- **Chat with documents** using RAG or general LLM chat
- **Generate documents** (PPTX, DOCX, PDF) from topics using AI
- **Orchestrate agents** for complex multi-step tasks

### Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python 3.11+) |
| Frontend | Next.js 15 + React 19 + TypeScript |
| Database | PostgreSQL with pgvector |
| Vector Store | pgvector (PostgreSQL extension) |
| LLM Integration | LangChain + LiteLLM |
| Task Queue | Ray (distributed computing) |
| Auth | JWT tokens + NextAuth.js |
| Styling | Tailwind CSS + shadcn/ui |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js 15)                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Dashboard│ │  Chat   │ │Documents│ │ Upload  │ │ Create  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       └───────────┴───────────┴───────────┴───────────┘         │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │   API Client      │                        │
│                    │ (lib/api/client)  │                        │
│                    └─────────┬─────────┘                        │
└──────────────────────────────┼──────────────────────────────────┘
                               │ HTTP/WebSocket
┌──────────────────────────────┼──────────────────────────────────┐
│                        BACKEND (FastAPI)                         │
│                    ┌─────────┴─────────┐                        │
│                    │   API Routes      │                        │
│                    │ (api/routes/*)    │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│  ┌───────────────────────────┼───────────────────────────────┐  │
│  │                    SERVICE LAYER                           │  │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │  │
│  │ │   RAG   │ │Generator│ │  LLM    │ │Embedding│          │  │
│  │ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │  │
│  │      │           │           │           │                │  │
│  │ ┌────┴───────────┴───────────┴───────────┴────┐          │  │
│  │ │              Agent System                    │          │  │
│  │ │  Manager → Generator → Critic → Research    │          │  │
│  │ └─────────────────────────────────────────────┘          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │   Database Layer  │                        │
│                    │ (SQLAlchemy ORM)  │                        │
│                    └─────────┬─────────┘                        │
└──────────────────────────────┼──────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────┐
│                    PostgreSQL + pgvector                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Documents│ │ Chunks  │ │  Users  │ │Sessions │ │ Agents  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
AIDocumentIndexer/
├── backend/                    # FastAPI backend
│   ├── api/                    # API layer
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── routes/            # Route handlers by domain
│   │   │   ├── auth.py        # Authentication endpoints
│   │   │   ├── documents.py   # Document CRUD
│   │   │   ├── chat.py        # RAG chat endpoints
│   │   │   ├── upload.py      # File upload handling
│   │   │   ├── generate.py    # Document generation
│   │   │   ├── agent.py       # Agent orchestration
│   │   │   ├── admin.py       # Admin settings
│   │   │   ├── scraper.py     # Web scraping
│   │   │   └── ...
│   │   ├── middleware/        # Request middleware
│   │   │   ├── auth.py        # JWT authentication
│   │   │   ├── rate_limit.py  # Rate limiting
│   │   │   └── cost_limit.py  # Usage cost limits
│   │   ├── errors.py          # Error handlers
│   │   └── websocket.py       # WebSocket handlers
│   │
│   ├── services/              # Business logic layer
│   │   ├── rag.py             # RAG service (core!)
│   │   ├── llm.py             # LLM factory & config
│   │   ├── embeddings.py      # Embedding service
│   │   ├── generator.py       # Document generation
│   │   ├── vectorstore.py     # Vector store operations
│   │   ├── general_chat.py    # Non-RAG chat
│   │   ├── agents/            # Multi-agent system
│   │   │   ├── agent_base.py  # Base agent classes
│   │   │   ├── manager_agent.py
│   │   │   ├── worker_agents.py
│   │   │   └── cost_estimator.py
│   │   └── ...
│   │
│   ├── db/                    # Database layer
│   │   ├── database.py        # Connection & sessions
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── migrations/        # Alembic migrations
│   │   ├── seed_agents.py     # Default agent seeding
│   │   └── seed_users.py      # Admin user seeding
│   │
│   ├── processors/            # Document processing
│   │   ├── universal.py       # Universal file processor
│   │   └── chunker.py         # Text chunking
│   │
│   └── tests/                 # Test suite
│
├── frontend/                   # Next.js frontend
│   ├── app/                   # App Router pages
│   │   ├── dashboard/         # Protected dashboard
│   │   │   ├── page.tsx       # Dashboard home
│   │   │   ├── chat/          # Chat interface
│   │   │   ├── documents/     # Document management
│   │   │   ├── upload/        # File upload
│   │   │   ├── create/        # Document generation
│   │   │   └── admin/         # Admin panels
│   │   ├── login/             # Login page
│   │   └── api/auth/          # NextAuth routes
│   │
│   ├── components/            # React components
│   │   └── ui/                # shadcn/ui components
│   │
│   ├── lib/                   # Utilities
│   │   ├── api/               # API client
│   │   │   ├── client.ts      # Type-safe API calls
│   │   │   └── hooks.ts       # React Query hooks
│   │   └── auth.ts            # Auth utilities
│   │
│   └── types/                 # TypeScript types
│
├── docs/                       # Documentation
│   ├── TECHNICAL_ARCHITECTURE.md
│   ├── API.md
│   ├── AGENTS.md
│   ├── DEVELOPMENT.md
│   ├── INSTALLATION.md
│   └── DEVELOPER_ONBOARDING.md  # This file!
│
└── docker-compose.yml          # Docker setup
```

---

## Backend Deep Dive

### API Layer

The API layer is built with FastAPI and follows a clear pattern:

#### Entry Point: `backend/api/main.py`

```python
# Application factory pattern
def create_app() -> FastAPI:
    app = FastAPI(
        title="AIDocumentIndexer",
        lifespan=lifespan,  # Handles startup/shutdown
    )

    # Middleware stack
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(CORSMiddleware, ...)

    # Register all routes
    register_routes(app)

    return app
```

The `lifespan` context manager handles:
- Database initialization
- Ray cluster connection (for parallel processing)
- OCR model downloads
- Default agent seeding

#### Route Structure

Each route file follows this pattern:

```python
# backend/api/routes/documents.py
from fastapi import APIRouter, Depends
from backend.api.middleware.auth import get_current_user

router = APIRouter()

@router.get("/")
async def list_documents(
    user: User = Depends(get_current_user),  # Auth required
    page: int = 1,
    page_size: int = 20,
):
    # Business logic delegated to service layer
    return await DocumentService.list_documents(user, page, page_size)
```

**Key Routes:**

| Route File | Prefix | Purpose |
|------------|--------|---------|
| `auth.py` | `/api/v1/auth` | Login, logout, token refresh |
| `documents.py` | `/api/v1/documents` | Document CRUD, search |
| `chat.py` | `/api/v1/chat` | RAG chat, sessions |
| `upload.py` | `/api/v1/upload` | File uploads |
| `generate.py` | `/api/v1/generate` | Document generation |
| `agent.py` | `/api/v1/agent` | Agent orchestration |
| `admin.py` | `/api/v1/admin` | Settings, users, tiers |
| `scraper.py` | `/api/v1/scraper` | Web scraping |

#### Authentication Middleware

```python
# backend/api/middleware/auth.py
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Decode JWT token and return the authenticated user.
    Raises HTTPException 401 if invalid.
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    user = await db.get(User, payload["sub"])
    return user
```

---

### Service Layer

The service layer contains all business logic. Key services:

#### RAGService (`backend/services/rag.py`)

The heart of the application - handles document retrieval and response generation.

```python
class RAGService:
    """
    Retrieval-Augmented Generation service.

    Core Features:
    - Hybrid search (vector + keyword)
    - Query expansion for better recall (+8-12%)
    - HyDE (Hypothetical Document Embeddings)
    - CRAG (Corrective RAG) for query refinement

    Advanced Features (NEW):
    - Self-RAG: Response verification against sources
    - Smart pre-filtering for large collections (10k-100k docs)
    - 3-level hierarchical retrieval (Collection → Doc → Chunk)
    - Knowledge graph integration (GraphRAG)
    """

    async def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        collection_filter: Optional[str] = None,
        top_k: int = 10,
        **kwargs,
    ) -> RAGResponse:
        """
        Main entry point for RAG queries.

        Flow:
        1. Classify query intent (dynamic weighting)
        2. Expand query (synonyms, related terms)
        3. Smart pre-filter if large collection
        4. Retrieve relevant chunks (hybrid search)
        5. Rerank results (ColBERT or cross-encoder)
        6. Generate response with LLM
        7. CRAG: Refine if low confidence
        8. Self-RAG: Verify against sources (optional)
        9. Return with sources and confidence
        """
```

**Advanced RAG Components:**

| Component | File | Purpose |
|-----------|------|---------|
| Self-RAG | `self_rag.py` | Verify responses, detect hallucinations |
| Smart Filter | `smart_filter.py` | Pre-filter large collections |
| 3-Level Retriever | `retrieval_strategies.py` | Hierarchical retrieval |
| Knowledge Graph | `knowledge_graph.py` | Entity-aware retrieval |
| CRAG | `corrective_rag.py` | Query refinement on low confidence |

**Key Data Classes:**

```python
@dataclass
class Source:
    """Source citation for RAG response."""
    document_id: str
    document_name: str
    chunk_id: str
    collection: Optional[str]
    page_number: Optional[int]
    relevance_score: float      # RRF score for ranking
    similarity_score: float     # Vector cosine similarity
    snippet: str
    full_content: str

@dataclass
class RAGResponse:
    """Response from RAG query."""
    content: str                # Generated answer
    sources: List[Source]       # Citations
    query: str
    model: str
    confidence_score: float     # 0-1 confidence
    suggested_questions: List[str]
```

#### LLM Service (`backend/services/llm.py`)

Unified interface for multiple LLM providers.

```python
class LLMFactory:
    """
    Factory for creating LLM instances.

    Supports:
    - OpenAI (gpt-4o, gpt-4o-mini, etc.)
    - Anthropic (claude-3-5-sonnet, etc.)
    - Ollama (local models)
    - 100+ other providers via LiteLLM
    """

    @classmethod
    def get_chat_model(
        cls,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> BaseChatModel:
        """Get a LangChain chat model instance."""

class EnhancedLLMFactory:
    """
    Database-driven LLM configuration.

    Reads provider settings from database for:
    - Per-operation model assignment (chat, generation, etc.)
    - Per-session model override
    - Usage tracking and cost estimation
    """

    @classmethod
    async def get_chat_model_for_operation(
        cls,
        operation: str,          # "chat", "generation", "embedding"
        session_id: Optional[str],
        user_id: Optional[str],
    ) -> Tuple[BaseChatModel, LLMConfigResult]:
        """Get model based on admin configuration."""
```

#### Document Generator (`backend/services/generator.py`)

Generates PPTX, DOCX, and PDF documents from topics.

```python
class DocumentGenerator:
    """
    AI-powered document generation.

    Flow:
    1. Create generation job
    2. Generate document outline
    3. Retrieve relevant context (RAG)
    4. Generate section content
    5. Build output file (PPTX/DOCX/PDF)
    """

    async def generate_document(
        self,
        job_id: str,
        topic: str,
        output_format: str,      # "pptx", "docx", "pdf"
        theme: str = "business",
        language: str = "en",
        **options,
    ) -> GenerationResult:
```

**Theme Configuration:**

```python
THEMES = {
    "business": {
        "name": "Business Professional",
        "primary": "#1E3A5F",
        "secondary": "#3D5A80",
        "accent": "#E0E1DD",
        "slide_background": "solid",
        "header_style": "underline",
        "bullet_style": "circle",
    },
    # 12+ themes available
}
```

#### Embedding Service (`backend/services/embeddings.py`)

```python
class EmbeddingService:
    """
    Embedding generation with caching and parallel processing.

    Providers:
    - OpenAI (text-embedding-3-small/large)
    - Ollama (nomic-embed-text)
    - HuggingFace (sentence-transformers)
    """

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""

    def embed_texts(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """Batch embedding with caching."""
```

---

### Database Layer

#### Models (`backend/db/models.py`)

Key models and their relationships:

```python
class User(Base, UUIDMixin, TimestampMixin):
    """User account with access tier."""
    email: Mapped[str]
    password_hash: Mapped[str]
    access_tier_id: Mapped[uuid.UUID]  # FK to AccessTier
    use_folder_permissions_only: Mapped[bool]  # Restrict to explicit folder access

    # Relationships
    access_tier: Mapped["AccessTier"]
    documents: Mapped[List["Document"]]
    chat_sessions: Mapped[List["ChatSession"]]
    folder_permissions: Mapped[List["FolderPermission"]]  # Explicit folder access

class Document(Base, UUIDMixin, TimestampMixin):
    """Uploaded document metadata."""
    filename: Mapped[str]
    file_hash: Mapped[str]           # Unique identifier
    file_type: Mapped[str]           # pdf, docx, pptx, etc.
    processing_status: Mapped[ProcessingStatus]
    access_tier_id: Mapped[uuid.UUID]

    # Enhanced metadata from LLM analysis
    enhanced_metadata: Mapped[Optional[dict]]  # summary, keywords, etc.

    # Relationships
    chunks: Mapped[List["Chunk"]]    # Document content chunks

class Chunk(Base, UUIDMixin):
    """Document chunk with embedding vector."""
    content: Mapped[str]
    content_hash: Mapped[str]
    embedding: Mapped[List[float]]   # pgvector Vector(1536)
    chunk_index: Mapped[int]
    page_number: Mapped[Optional[int]]

    # Hierarchical chunking
    is_summary: Mapped[bool]         # True for summaries
    chunk_level: Mapped[int]         # 0=detail, 1=section, 2=document
    parent_chunk_id: Mapped[Optional[uuid.UUID]]

    # Relationships
    document: Mapped["Document"]
    access_tier: Mapped["AccessTier"]

class ChatSession(Base, UUIDMixin, TimestampMixin):
    """Chat conversation session."""
    title: Mapped[Optional[str]]
    user_id: Mapped[uuid.UUID]

    # Relationships
    user: Mapped["User"]
    messages: Mapped[List["ChatMessage"]]

class ChatMessage(Base, UUIDMixin):
    """Individual chat message."""
    role: Mapped[MessageRole]        # user, assistant, system
    content: Mapped[str]
    source_document_ids: Mapped[List[uuid.UUID]]  # Citations
    model_used: Mapped[Optional[str]]
    tokens_used: Mapped[Optional[int]]

class Folder(Base, UUIDMixin, TimestampMixin):
    """Hierarchical folder for organizing documents."""
    name: Mapped[str]
    path: Mapped[str]                # Full path like "/Marketing/2024/"
    parent_folder_id: Mapped[Optional[uuid.UUID]]
    access_tier_id: Mapped[uuid.UUID]
    tags: Mapped[List[str]]          # Folder tags for categorization

    # Relationships
    parent: Mapped[Optional["Folder"]]
    children: Mapped[List["Folder"]]
    documents: Mapped[List["Document"]]
    permissions: Mapped[List["FolderPermission"]]

class FolderPermission(Base, UUIDMixin, TimestampMixin):
    """Per-user folder access permission."""
    folder_id: Mapped[uuid.UUID]
    user_id: Mapped[uuid.UUID]
    permission_level: Mapped[str]    # "view", "edit", "manage"
    inherit_to_children: Mapped[bool]
    granted_by_id: Mapped[Optional[uuid.UUID]]

    # Relationships
    folder: Mapped["Folder"]
    user: Mapped["User"]
    granted_by: Mapped[Optional["User"]]
```

**Entity Relationship Diagram:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  AccessTier  │─────│     User     │─────│ ChatSession  │
│              │ 1:N │              │ 1:N │              │
│  - level     │     │  - email     │     │  - title     │
│  - name      │     │  - password  │     │              │
└──────────────┘     │  - folder_   │     └──────────────┘
       │            │    perm_only │            │
       │ 1:N        └──────────────┘            │ 1:N
       ▼                    │                    ▼
┌──────────────┐     ┌──────┴───────┐     ┌──────────────┐
│   Document   │─────│    Chunk     │     │ ChatMessage  │
│              │ 1:N │              │     │              │
│  - filename  │     │  - content   │     │  - role      │
│  - status    │     │  - embedding │     │  - content   │
│  - metadata  │     │  - page_num  │     │  - sources   │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       │ N:1
       ▼
┌──────────────┐     ┌──────────────┐
│    Folder    │─────│FolderPermis- │
│              │ 1:N │   sion       │
│  - name      │     │              │
│  - path      │     │  - user_id   │
│  - tags      │     │  - level     │
│  - parent_id │     │  - inherit   │
└──────────────┘     └──────────────┘
```

---

### Agent System

The multi-agent system orchestrates complex tasks with parallel execution and cross-validation.

#### Architecture (`backend/services/agents/`)

```
┌─────────────────────────────────────────────────────────┐
│                    Manager Agent                         │
│  - Decomposes complex tasks into subtasks               │
│  - Assigns tasks to worker agents                       │
│  - Executes independent steps in PARALLEL               │
│  - Coordinates execution order (respects dependencies)  │
└─────────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Generator  │     │   Critic    │     │  Research   │
│   Agent     │     │   Agent     │     │   Agent     │
│             │     │             │     │             │
│Creates text │     │ Evaluates   │     │ Retrieves   │
│from prompts │     │ quality     │     │ documents   │
└──────┬──────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Validator  │     │    Tool     │     │  Adaptive   │
│   Agent     │     │  Execution  │     │  Planner    │
│   (NEW)     │     │   Agent     │     │   (NEW)     │
│             │     │             │     │             │
│Cross-check  │     │File exports │     │Learn from   │
│vs sources   │     │PPTX/DOCX/PDF│     │past runs    │
└─────────────┘     └─────────────┘     └─────────────┘
```

#### Key Features

| Feature | Description | File |
|---------|-------------|------|
| **Parallel Execution** | Independent steps run concurrently via `asyncio.gather()` | `manager_agent.py` |
| **Cross-Validation** | Validates content against sources, detects hallucinations | `validator_agent.py` |
| **Tool Calling** | Extensible tool framework for agents | `tools.py` |
| **Adaptive Planning** | Learns from trajectory history to optimize future plans | `trajectory_collector.py` |

#### Base Agent (`backend/services/agents/agent_base.py`)

```python
@dataclass
class AgentConfig:
    """Configuration for agent instance."""
    agent_id: str
    name: str
    description: str
    provider_type: Optional[str]  # openai, anthropic, ollama
    model: Optional[str]
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: List[str] = field(default_factory=list)
    language: str = "en"

@dataclass
class AgentTask:
    """Structured task for agent execution."""
    id: str
    type: TaskType               # GENERATION, EVALUATION, RESEARCH, etc.
    name: str
    description: str
    expected_inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    success_criteria: List[str]
    fallback_strategy: FallbackStrategy
    dependencies: List[str]      # Other task IDs

@dataclass
class AgentResult:
    """Standardized result from agent."""
    task_id: str
    agent_id: str
    status: TaskStatus           # COMPLETED, FAILED, etc.
    output: Any
    reasoning_trace: List[str]   # Chain of thought
    tool_calls: List[Dict]
    tokens_used: int
    confidence_score: float
```

#### Tool Framework (`backend/services/agents/tools.py`)

The Tool Framework enables extensible tool calling for agents. Tools can search documents, perform calculations, fetch web data, and more.

```python
from backend.services.agents.tools import (
    ToolRegistry,
    ToolExecutor,
    BaseTool,
    ToolParameter,
    ToolCategory,
    get_tool_executor,
)

# Built-in tools (auto-registered)
# - search_documents: Search uploaded documents via RAG
# - calculator: Safe mathematical calculations
# - web_search: Web search integration
# - get_datetime: Current date/time

# Check available tools
if ToolRegistry.has_tool("search_documents"):
    print("Document search is available")

# Get function definitions for LLM tool calling
functions = ToolRegistry.get_function_definitions()

# Execute a tool
executor = get_tool_executor()
result = await executor.execute(
    tool_name="search_documents",
    parameters={"query": "climate change", "limit": 5},
)

# Creating custom tools
class MySummaryTool(BaseTool):
    name = "summarize"
    description = "Summarize text content"
    category = ToolCategory.COMPUTATION
    parameters = [
        ToolParameter(name="text", type="string", required=True),
        ToolParameter(name="max_length", type="number", default=200),
    ]

    async def execute(self, text: str, max_length: int = 200) -> dict:
        # Your summarization logic
        return {"summary": text[:max_length] + "..."}

# Register the custom tool
ToolRegistry.register(MySummaryTool())
```

**Integration with Worker Agents:**

The `ToolExecutionAgent` automatically checks the ToolRegistry for extensible tools:

```python
from backend.services.agents.worker_agents import create_worker_agents
from backend.services.agents.tools import get_tool_executor

# Create workers with tool executor
workers = create_worker_agents(
    rag_service=rag,
    generator_service=generator,
    tool_executor=get_tool_executor(),
)

# The "tool" worker now supports:
# 1. Registered tools (search_documents, calculator, etc.)
# 2. Built-in tools (generate_pptx, generate_docx, export_markdown)
```

#### LLM Tool Calling (`backend/services/agents/agent_base.py`)

Agents can dynamically select and call tools using LLM function calling via the `invoke_llm_with_tools` method:

```python
from langchain_core.messages import HumanMessage
from backend.services.agents.tools import ToolRegistry, get_tool_executor

# Get function definitions for LLM tool calling (OpenAI format)
tools = ToolRegistry.get_function_definitions()

# Invoke LLM with tool support - it decides which tools to call
response_text, tool_calls, in_tokens, out_tokens = await agent.invoke_llm_with_tools(
    messages=[HumanMessage(content="Search for documents about climate change")],
    tools=tools,
    tool_choice="auto",  # "auto", "none", or specific tool name
)

# Execute any tool calls the LLM decided to make
if tool_calls:
    executor = get_tool_executor()
    for call in tool_calls:
        result = await executor.execute(
            tool_name=call["name"],
            parameters=call["arguments"],
        )
        print(f"Tool {call['name']} returned: {result.output}")
```

**Supported providers:** OpenAI, Anthropic Claude, and any LangChain-compatible provider with tool calling support.

#### Agent Streaming (`backend/services/agents/manager_agent.py`)

For real-time frontend updates, use `execute_with_streaming`:

```python
from backend.services.agents.manager_agent import ManagerAgent

manager = ManagerAgent(config, workers=workers)

# Stream execution progress to frontend
async for event in manager.execute_with_streaming(
    user_request="Create a summary of all uploaded documents",
    session_id=session_id,
    user_id=user_id,
):
    event_type = event["type"]

    if event_type == "planning":
        print("Creating execution plan...")
    elif event_type == "plan_ready":
        print(f"Plan ready: {event['total_steps']} steps")
    elif event_type == "step_started":
        print(f"Starting: {event['step_name']}")
    elif event_type == "step_progress":
        print(f"Progress: {event['progress_percent']}%")
    elif event_type == "content_chunk":
        # Stream content as it's generated
        print(event["content"], end="", flush=True)
    elif event_type == "step_completed":
        print(f"Completed: {event['step_name']}")
    elif event_type == "sources":
        print(f"Sources: {len(event['sources'])} documents")
    elif event_type == "complete":
        print(f"\nDone! Cost: ${event['total_cost_usd']:.4f}")
    elif event_type == "error":
        print(f"Error: {event['error']}")
```

**Event types:**
| Event | Description |
|-------|-------------|
| `planning` | Plan creation started |
| `plan_ready` | Plan created with step overview |
| `step_started` | Step execution began |
| `step_progress` | Progress update (0-100%) |
| `content_chunk` | Partial content stream |
| `step_completed` | Step finished |
| `step_failed` | Step failed with error |
| `sources` | Document sources used |
| `synthesis_started` | Final synthesis in progress |
| `complete` | Execution finished |
| `error` | Error occurred |

---

## Frontend Deep Dive

### Page Structure

The frontend uses Next.js 15 App Router:

```
app/
├── layout.tsx              # Root layout with providers
├── page.tsx                # Landing page (redirects to login)
├── login/page.tsx          # Login form
└── dashboard/
    ├── layout.tsx          # Dashboard layout with sidebar
    ├── page.tsx            # Dashboard home (stats)
    ├── chat/page.tsx       # Chat interface
    ├── documents/
    │   ├── page.tsx        # Document list
    │   └── [id]/page.tsx   # Document detail
    ├── upload/page.tsx     # File upload
    ├── create/page.tsx     # Document generation
    └── admin/
        ├── settings/page.tsx
        ├── users/page.tsx
        └── agents/page.tsx
```

### API Client (`frontend/lib/api/client.ts`)

Type-safe API client with full TypeScript support:

```typescript
// Types mirror backend Pydantic models
export interface Document {
  id: string;
  name: string;
  file_type: string;
  file_size: number;
  status: string;
  chunk_count: number;
  enhanced_metadata?: EnhancedMetadata;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  collection_filter?: string;
  mode?: 'agent' | 'chat' | 'general' | 'vision';
  top_k?: number;
  images?: ImageAttachment[];  // For vision mode
}

export interface ImageAttachment {
  data?: string;      // Base64-encoded image
  url?: string;       // URL to image (alternative to data)
  mime_type: string;  // e.g., "image/jpeg", "image/png"
}

export interface ChatResponse {
  session_id: string;
  content: string;
  sources: ChatSource[];
  confidence_score?: number;
}

// API methods
export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(request),
  });
  return response.json();
}
```

### React Query Hooks (`frontend/lib/api/hooks.ts`)

```typescript
// Hooks for data fetching with caching
export function useDocuments(page: number, pageSize: number) {
  return useQuery({
    queryKey: ['documents', page, pageSize],
    queryFn: () => fetchDocuments(page, pageSize),
  });
}

export function useSendMessage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: sendChatMessage,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chat-sessions'] });
    },
  });
}
```

---

## Data Flow

### Document Upload Flow

```
1. User selects file(s) in Upload page
   └─> frontend/app/dashboard/upload/page.tsx

2. File sent to upload endpoint
   └─> POST /api/v1/upload/single
   └─> backend/api/routes/upload.py

3. File validated and stored
   └─> backend/processors/universal.py (extract text)
   └─> backend/processors/chunker.py (split into chunks)

4. Chunks embedded and indexed
   └─> backend/services/embeddings.py (generate vectors)
   └─> backend/services/vectorstore.py (store in pgvector)

5. Document record created
   └─> backend/db/models.py (Document, Chunk)

6. WebSocket notification sent
   └─> backend/api/websocket.py
   └─> frontend receives real-time status
```

### Chat Modes

The chat endpoint (`POST /api/v1/chat/completions`) supports four execution modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `chat` | **RAG mode** (default) - Retrieves relevant documents and generates answers with citations | Questions about uploaded documents |
| `agent` | **Agent mode** - Multi-agent orchestration for complex tasks | Document generation, multi-step research |
| `general` | **General LLM** - Direct LLM chat without document retrieval | General questions, coding help |
| `vision` | **Vision mode** - Multimodal image analysis | Analyze images, charts, screenshots |

#### Vision Mode

Vision mode enables image analysis using multimodal LLMs (GPT-4o, Claude 3.5, LLaVA, etc.):

```typescript
// Frontend: Send image for analysis
const request = {
  message: "What does this chart show?",
  mode: "vision",
  images: [
    {
      data: base64EncodedImage,  // Base64-encoded image
      mime_type: "image/png"
    }
  ]
};

// Or use URL
const request = {
  message: "Describe this diagram",
  mode: "vision",
  images: [
    {
      url: "https://example.com/diagram.png",
      mime_type: "image/png"
    }
  ]
};
```

**Backend flow:**
```
1. Request with mode="vision" + images[]
2. Backend extracts image data (base64 or fetches URL)
3. Calls chat_with_vision() from backend/services/llm.py
4. Returns analysis (no document sources)
```

**Configuration:**
```bash
# .env - Vision model settings
rag.vision_provider=auto       # auto, openai, ollama
rag.ollama_vision_model=llava  # For local Ollama vision
```

### RAG Chat Flow

```
1. User sends message in Chat page
   └─> frontend/app/dashboard/chat/page.tsx

2. Request sent to chat endpoint
   └─> POST /api/v1/chat/completions
   └─> backend/api/routes/chat.py

3. RAG service processes query
   └─> backend/services/rag.py

   a. Query Classification
      └─> backend/services/query_classifier.py

   b. Query Expansion (optional)
      └─> backend/services/query_expander.py

   c. Document Retrieval
      └─> backend/services/vectorstore.py (hybrid search)

   d. Reranking (optional)
      └─> backend/services/colbert_reranker.py

   e. Response Generation
      └─> backend/services/llm.py (invoke LLM)

   f. Verification (optional)
      └─> backend/services/rag_verifier.py

4. Response with sources returned
   └─> frontend displays answer + citations
```

### Document Generation Flow

```
1. User configures generation in Create page
   └─> frontend/app/dashboard/create/page.tsx
   └─> Selects: topic, format, theme, language, sections

2. Job created
   └─> POST /api/v1/generate/jobs
   └─> backend/api/routes/generate.py

3. Generator service runs
   └─> backend/services/generator.py

   a. Outline Generation
      └─> LLM creates document structure

   b. RAG Context Retrieval
      └─> Search uploaded docs for relevant content

   c. Section Content Generation
      └─> LLM generates each section with context

   d. Output Building
      └─> PPTX: python-pptx library
      └─> DOCX: python-docx library
      └─> PDF: ReportLab library

4. File stored and URL returned
   └─> User can download generated document
```

---

## Key Classes Reference

### Backend Classes

| Class | File | Purpose |
|-------|------|---------|
| `RAGService` | `services/rag.py` | Core RAG functionality |
| `RAGConfig` | `services/rag.py` | RAG configuration options |
| `RAGResponse` | `services/rag.py` | Response with sources |
| `LLMFactory` | `services/llm.py` | Create LLM instances |
| `EnhancedLLMFactory` | `services/llm.py` | Database-driven LLM config |
| `EmbeddingService` | `services/embeddings.py` | Generate embeddings |
| `VectorStore` | `services/vectorstore.py` | Vector storage operations |
| `DocumentGenerator` | `services/generator.py` | Document generation |
| `BaseAgent` | `services/agents/agent_base.py` | Agent base class with `invoke_llm_with_tools()` |
| `AgentTask` | `services/agents/agent_base.py` | Task definition |
| `AgentResult` | `services/agents/agent_base.py` | Task result |
| `ManagerAgent` | `services/agents/manager_agent.py` | Task orchestration with `execute_with_streaming()` |
| `ToolRegistry` | `services/agents/tools.py` | Tool registration and function definitions |
| `ToolExecutor` | `services/agents/tools.py` | Execute registered tools |
| `Document` | `db/models.py` | Document ORM model |
| `Chunk` | `db/models.py` | Chunk ORM model |
| `User` | `db/models.py` | User ORM model |
| `ChatSession` | `db/models.py` | Chat session model |
| `Folder` | `db/models.py` | Hierarchical folder model with tags |
| `FolderPermission` | `db/models.py` | Per-user folder access permission |

### Frontend Types

| Type | File | Purpose |
|------|------|---------|
| `Document` | `lib/api/client.ts` | Document type |
| `ChatRequest` | `lib/api/client.ts` | Chat request params |
| `ChatResponse` | `lib/api/client.ts` | Chat response type |
| `ChatSource` | `lib/api/client.ts` | Source citation |
| `UploadOptions` | `lib/api/client.ts` | Upload configuration |
| `GenerationRequest` | `lib/api/client.ts` | Generation params |
| `FolderPermissionResponse` | `lib/api/client.ts` | Folder permission from folder view |
| `UserFolderPermissionResponse` | `lib/api/client.ts` | Folder permission from user view |
| `AdminUser` | `lib/api/client.ts` | Admin user with folder permissions |

---

## Common Development Tasks

### Adding a New API Endpoint

1. **Create route handler** in `backend/api/routes/`:

```python
# backend/api/routes/my_feature.py
from fastapi import APIRouter, Depends
from backend.api.middleware.auth import get_current_user

router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint(
    user = Depends(get_current_user),
):
    return {"message": "Hello from my endpoint"}
```

2. **Register route** in `backend/api/main.py`:

```python
from backend.api.routes.my_feature import router as my_feature_router
app.include_router(my_feature_router, prefix="/api/v1/my-feature", tags=["My Feature"])
```

3. **Add frontend API client method** in `frontend/lib/api/client.ts`:

```typescript
export async function myEndpoint(): Promise<MyResponse> {
  return apiRequest('/my-feature/my-endpoint');
}
```

### Adding a New Database Model

1. **Define model** in `backend/db/models.py`:

```python
class MyModel(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "my_models"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Add fields...
```

2. **Create migration**:

```bash
cd backend
PYTHONPATH=. alembic revision --autogenerate -m "Add my_models table"
PYTHONPATH=. alembic upgrade head
```

### Adding a New Service

1. **Create service file** in `backend/services/`:

```python
# backend/services/my_service.py
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

class MyService:
    def __init__(self, config: Optional[MyConfig] = None):
        self.config = config or MyConfig()

    async def do_something(self, input: str) -> str:
        logger.info("Doing something", input=input)
        # Business logic here
        return result

# Singleton pattern
_my_service: Optional[MyService] = None

def get_my_service() -> MyService:
    global _my_service
    if _my_service is None:
        _my_service = MyService()
    return _my_service
```

2. **Use in route**:

```python
from backend.services.my_service import get_my_service

@router.post("/action")
async def action(request: MyRequest):
    service = get_my_service()
    result = await service.do_something(request.input)
    return {"result": result}
```

---

## Debugging Guide

### Backend Debugging

1. **Enable debug logging**:
```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=console  # or "json" for structured logs
```

2. **Check API logs**:
```bash
tail -f backend/logs/api.log
```

3. **Debug RAG queries**:
```python
# In services/rag.py, add logging
logger.debug(
    "RAG query",
    question=question,
    expanded_queries=expanded_queries,
    retrieved_chunks=len(chunks),
    top_scores=[c.score for c in chunks[:5]],
)
```

4. **Test endpoint directly**:
```bash
# Generate test token
python -c "from backend.api.middleware.auth import create_test_token; print(create_test_token())"

# Call endpoint
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "test query"}'
```

### Frontend Debugging

1. **Enable React Query devtools**:
```typescript
// Already enabled in development
// Open DevTools > React Query tab
```

2. **Check network requests**:
```typescript
// Add to lib/api/client.ts
console.log('API Request:', url, options);
console.log('API Response:', response);
```

3. **Debug state**:
```typescript
// In component
console.log('Current state:', { documents, isLoading, error });
```

### Database Debugging

1. **Check database state**:
```bash
# Connect to PostgreSQL
psql -U postgres -d aidocindexer

# List documents
SELECT id, filename, processing_status FROM documents LIMIT 10;

# Check chunk embeddings
SELECT id, document_id, char_length(content), embedding IS NOT NULL as has_embedding
FROM chunks LIMIT 10;
```

2. **Debug SQLAlchemy queries**:
```python
# Enable SQL logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

---

## Environment Setup Checklist

1. **Clone repository**
2. **Install backend dependencies**:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```
4. **Set up PostgreSQL with pgvector**
5. **Configure `.env` file** (copy from `.env.example`)
6. **Run database migrations**:
   ```bash
   cd backend
   PYTHONPATH=. alembic upgrade head
   ```
7. **Start services**:
   ```bash
   # Backend
   PYTHONPATH=. uvicorn backend.api.main:app --reload --port 8000

   # Frontend
   cd frontend && npm run dev
   ```
8. **Access application** at http://localhost:3000

---

## Additional Resources

- [CODE_REFERENCE.md](./CODE_REFERENCE.md) - **Complete function reference with call chains and user flows**
- [TECHNICAL_ARCHITECTURE.md](./TECHNICAL_ARCHITECTURE.md) - Detailed system architecture
- [API.md](./API.md) - Complete API reference
- [AGENTS.md](./AGENTS.md) - Multi-agent system documentation
- [DEVELOPMENT.md](./DEVELOPMENT.md) - Development workflow guide
- [INSTALLATION.md](./INSTALLATION.md) - Installation instructions

---

## Getting Help

- Check existing documentation in `/docs`
- Review test files for usage examples
- Look at route handlers for API patterns
- Check service files for business logic patterns

Happy coding!
