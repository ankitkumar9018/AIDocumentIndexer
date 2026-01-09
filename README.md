# AIDocumentIndexer

> **Intelligent Document Archive with RAG** - Transform 25+ years of presentations, documents, and knowledge into a searchable, AI-powered assistant.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

AIDocumentIndexer is an enterprise-grade RAG (Retrieval-Augmented Generation) system designed to serve as an intelligent archive for your organization's knowledge base. Built for teams who have accumulated years of presentations, reports, and strategic documents, it enables staff to:

- **Search & Discover**: Find relevant past work using natural language queries
- **Get AI-Powered Answers**: Ask questions and receive contextual answers with source citations
- **Generate New Content**: Create new presentations, reports, and documents inspired by existing work
- **Maintain Security**: Role-based access control ensures sensitive documents stay protected

### Key Features

| Feature | Description |
|---------|-------------|
| **Universal File Support** | PDF, PPTX, DOCX, XLSX, images, and 20+ more formats |
| **Multi-LLM Support** | OpenAI GPT-4, Ollama (local), Claude, and 100+ providers via LiteLLM |
| **Smart RAG with Self-Verification** | LangChain-powered retrieval with hybrid search + answer confidence scoring |
| **GraphRAG** | Knowledge graph-based retrieval for multi-hop reasoning across entities |
| **Agentic RAG** | Complex query decomposition with ReAct loop for iterative reasoning |
| **Multimodal RAG** | Image captioning and table extraction with free local (Ollama LLaVA) or cloud providers |
| **Semantic Chunking** | Context-aware document chunking with section headers for better retrieval |
| **Source Citations** | Every answer shows exactly which documents were used |
| **Confidence Indicators** | See how confident the AI is in each response (high/medium/low) |
| **Query Suggestions** | Intelligent follow-up question suggestions after each answer |
| **Permission Tiers** | Dynamic access control - CEO sees all, interns see only authorized files |
| **Document Generation** | Create PPTX, DOCX, PDF with human-in-the-loop approval workflow |
| **Multi-LLM Collaboration** | Multiple AI models working together for higher quality output |
| **Agent Mode** | Multi-step task execution with planning and approval workflows |
| **Web Scraping** | Import content from websites with depth control and job scheduling |
| **Cost Tracking** | Monitor LLM usage costs with detailed analytics dashboard |
| **Real-Time Indexing** | Incremental updates with content freshness tracking |
| **File Watcher** | Automatically index new files from monitored directories |
| **Advanced OCR Management** | PaddleOCR, EasyOCR, Tesseract with performance metrics & batch downloads |
| **Tabbed Admin Settings** | Organized settings UI with 8 tabs for easy configuration |
| **Folder Management** | Hierarchical folders with permissions, folder-scoped searches |
| **Saved Searches** | Save complex search queries with filters for quick access |
| **Search Operators** | AND, OR, NOT, phrase matching, and grouping support |
| **User Preferences** | Persistent theme, view mode, default collection settings |
| **Beautiful UI** | Modern, responsive interface built with Next.js 15 and shadcn/ui |

---

## Quick Start

### Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **OpenAI API Key** (or Ollama for local LLM)
- **8GB+ RAM** (16GB recommended for local LLM)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AIDocumentIndexer.git
cd AIDocumentIndexer

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use any editor

# Start all services
docker-compose up -d

# Open in browser
open http://localhost:3000
```

### Default Admin Login

```
Email: admin@example.com
Password: changeme123
```

> **Important**: Change the admin password immediately after first login!

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 15 + shadcn/ui)                    │
│  Dashboard │ Chat Interface │ Upload Portal │ Document Creator │ Admin │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKEND (FastAPI + Python)                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    LangChain + LiteLLM                          │    │
│  │  RAG Chains │ Memory │ Agents │ 100+ LLM Providers              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Document  │  │  Embedding  │  │     OCR     │  │     Ray     │    │
│  │  Processors │  │  Generation │  │  (PaddleOCR)│  │   Cluster   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                  │
│  PostgreSQL + pgvector │ S3/Local Files │ Redis Cache                   │
│  Row-Level Security (RLS) for Permission Enforcement                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Next.js 15, React 19, shadcn/ui, Tailwind CSS | Beautiful, responsive UI |
| API | FastAPI, Python 3.11+ | High-performance REST API |
| RAG Framework | LangChain, LangGraph | Chains, memory, agents, workflows |
| LLM Router | LiteLLM | Unified API for 100+ LLM providers |
| Parallel Processing | Ray | Distributed document processing |
| Vector Database | PostgreSQL + pgvector | Embeddings with RLS security |
| OCR | PaddleOCR | Best-in-class text extraction from images |
| Document Parsing | Unstructured.io | Universal document parser |
| Containerization | Docker, Docker Compose | Easy deployment anywhere |

---

## Installation

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/AIDocumentIndexer.git
cd AIDocumentIndexer

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings (see Configuration section)

# 3. Start services
docker-compose up -d

# 4. Check status
docker-compose ps

# 5. View logs
docker-compose logs -f
```

### Option 2: Local Development (with UV)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/AIDocumentIndexer.git
cd AIDocumentIndexer

# 2. Backend setup (requires uv: curl -LsSf https://astral.sh/uv/install.sh | sh)
cd backend
uv sync
uv run uvicorn backend.api.main:app --reload --port 8000

# 3. Frontend setup (in another terminal)
cd frontend
npm install
npm run dev
```

### Option 3: Manual Installation

See [INSTALLATION.md](docs/INSTALLATION.md) for detailed manual setup instructions.

### Option 4: AWS Deployment

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for production AWS deployment guide.

---

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# =============================================================================
# APPLICATION
# =============================================================================
APP_NAME=AIDocumentIndexer
APP_ENV=development  # development | staging | production
SECRET_KEY=your-secret-key-min-32-chars

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_TYPE=postgresql  # postgresql | sqlite | mysql
DATABASE_URL=postgresql://user:password@localhost:5432/aidocindexer

# =============================================================================
# LLM PROVIDERS
# =============================================================================
# OpenAI (Primary)
OPENAI_API_KEY=sk-your-openai-key

# Ollama (Local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Anthropic (Optional)
ANTHROPIC_API_KEY=sk-ant-your-key

# Default LLM Settings
DEFAULT_CHAT_MODEL=gpt-4o
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small

# =============================================================================
# RAY CLUSTER
# =============================================================================
RAY_ADDRESS=auto  # auto for local, or ray://head-node:10001
RAY_NUM_CPUS=4

# =============================================================================
# STORAGE
# =============================================================================
STORAGE_TYPE=local  # local | s3
STORAGE_PATH=/app/storage
# S3 Settings (if STORAGE_TYPE=s3)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_S3_BUCKET=

# =============================================================================
# AUTHENTICATION
# =============================================================================
NEXTAUTH_SECRET=your-nextauth-secret
NEXTAUTH_URL=http://localhost:3000

# =============================================================================
# OPTIONAL FEATURES
# =============================================================================
ENABLE_FILE_WATCHER=false
WATCH_DIRECTORIES=/path/to/watch1,/path/to/watch2
ENABLE_WEB_SCRAPING=true
```

See [CONFIGURATION.md](docs/CONFIGURATION.md) for all available options.

---

## Usage

### Uploading Documents

1. Navigate to **Upload** in the sidebar
2. Drag and drop files or click to browse
3. Select processing mode:
   - **Store for RAG**: Index documents for future search
   - **Query Only**: Ask questions without permanent storage
4. Set access tier (admin only)
5. Click **Upload & Process**

### Chatting with Your Documents

1. Go to **Chat** in the sidebar
2. Type your question in natural language
3. View AI-generated answer with source citations
4. Click **Show Sources** to see which documents were used
5. Use quick-reply suggestions for follow-up questions

### Creating New Documents

1. Navigate to **Create** in the sidebar
2. Describe what you want to create (e.g., "Create a presentation about stadium activations")
3. Review the generated outline
4. Approve or modify each section
5. Download in your preferred format (PPTX, DOCX, PDF, etc.)

### Admin Configuration

1. Go to **Admin** panel (admin users only)
2. Manage users and assign permission tiers
3. Configure LLM providers and models
4. Set up file watcher directories
5. View system health and processing logs

---

## Supported File Types

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, DOC, ODT, RTF |
| Presentations | PPTX, PPT, ODP, KEY |
| Spreadsheets | XLSX, XLS, CSV, ODS |
| Images | PNG, JPG, JPEG, TIFF, BMP, WebP |
| Text | TXT, MD, RST, HTML, XML, JSON |
| Email | EML, MSG |
| Archives | ZIP (auto-extract) |
| Audio/Video | MP3, WAV, MP4 (transcription with Whisper) |

---

## Permission System

AIDocumentIndexer uses a two-layer permission system for fine-grained access control:

### Layer 1: Tier-Based Access

| Default Tier | Level | Access |
|--------------|-------|--------|
| Intern | 10 | Public documents only |
| Staff | 30 | Internal + Public |
| Manager | 50 | Confidential + Internal + Public |
| Executive | 80 | All documents |
| Admin | 100 | All documents + System administration |

**Key Rules:**
- Admins can create custom tiers with any level (1-100)
- Users can only assign document tiers ≤ their own tier
- Users can only create users with tiers ≤ their own tier
- Permission filtering happens at database level (RLS) - LLM cannot bypass

### Layer 2: Per-User Folder Permissions

Grant specific users access to specific folders, independent of their tier:

| Permission Level | Description |
|------------------|-------------|
| **View** | Can see folder and read documents |
| **Edit** | Can upload and modify documents |
| **Manage** | Can grant permissions to others |

**Features:**
- Permissions can inherit to subfolders automatically
- Manage folder access from both folder view and user management
- "Folder Only" mode restricts users to explicitly granted folders only

**Use Cases:**
- External contractors who need access to one project folder
- Department-specific access without creating new tiers
- Temporary access for specific collaborations
- Compliance scenarios requiring strict access control

---

## API Reference

### Quick Examples

```python
# Upload a document
import requests

response = requests.post(
    "http://localhost:8000/api/v1/documents/upload",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    files={"file": open("presentation.pptx", "rb")},
    data={"store_for_rag": True, "access_tier": 30}
)

# Chat with documents
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={"message": "What activation ideas do we have for stadiums?"}
)
print(response.json()["answer"])
print(response.json()["sources"])
```

See [API.md](docs/API.md) for complete API documentation.

---

## Development

### Local Development Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Structure

```
AIDocumentIndexer/
├── frontend/           # Next.js 15 application
├── backend/            # FastAPI + LangChain + Ray
├── docker/             # Docker configurations
├── db/                 # Database migrations
├── docs/               # Documentation
└── tests/              # Test suites
```

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development guide.

**New developers**: Start with [DEVELOPER_ONBOARDING.md](docs/DEVELOPER_ONBOARDING.md) for a comprehensive guide to the codebase architecture, key classes, and project flow.

---

## Troubleshooting

### Common Issues

**Q: Docker containers won't start**
```bash
# Check logs
docker-compose logs -f

# Reset everything
docker-compose down -v
docker-compose up -d --build
```

**Q: Ollama connection failed**
```bash
# Ensure Ollama is running
ollama serve

# Pull required model
ollama pull llama3.2
```

**Q: Out of memory during document processing**
- Reduce `RAY_NUM_CPUS` in `.env`
- Use "Text-Only Mode" to skip images
- Process fewer documents at once

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

---

## Roadmap

- [x] Phase 1: Core infrastructure
- [x] Phase 2: Document processing
- [x] Phase 3: RAG & Search
- [x] Phase 4: Permission system (Tier-based RLS)
- [x] Phase 5: Document generation (PPTX, DOCX, PDF with approval workflow)
- [x] Phase 6: Advanced features
  - [x] Multi-LLM Collaboration
  - [x] Web scraping with depth control
  - [x] Agent Mode with multi-step execution
  - [x] Cost tracking and analytics
  - [x] Self-RAG with answer verification
  - [x] Semantic chunking with contextual headers
  - [x] Confidence scoring for responses
- [x] Phase 7: UI/UX polish
  - [x] Dashboard with usage analytics
  - [x] Keyboard shortcuts
  - [x] Document favorites and recently viewed
  - [x] Chat history search
  - [x] Bulk operations for documents
- [x] Phase 8: Document Organization
  - [x] Hierarchical folder management
  - [x] User preferences persistence (theme, view mode, defaults)
  - [x] Saved searches with filters
  - [x] Advanced search operators (AND, OR, NOT, phrases)
  - [x] Folder-scoped RAG queries

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [LiteLLM](https://litellm.ai/) - LLM router
- [shadcn/ui](https://ui.shadcn.com/) - UI components
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine
- [Ray](https://ray.io/) - Distributed computing
- [Unstructured](https://unstructured.io/) - Document parsing

---

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/AIDocumentIndexer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/AIDocumentIndexer/discussions)

---

<p align="center">
  <b>Built with care for teams who value their knowledge</b>
</p>
