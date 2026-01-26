# Getting Started with AIDocumentIndexer

This guide will help you get up and running with AIDocumentIndexer in just a few minutes.

## Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- Redis server
- PostgreSQL database (or SQLite for development)

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-org/AIDocumentIndexer.git
cd AIDocumentIndexer

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

### 2. Configure Environment

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aidocs
# Or for development: DATABASE_URL=sqlite:///./aidocs.db

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM Providers (at least one required)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Audio
CARTESIA_API_KEY=your_cartesia_key

# Optional: Reranking
COHERE_API_KEY=your_cohere_key

# Security
JWT_SECRET=your_random_secret_key
```

### 3. Start Services

**Start Redis:**
```bash
redis-server
```

**Start the backend:**
```bash
cd backend
uvicorn backend.api.main:app --reload --port 8000
```

**Start Celery workers (for background processing):**
```bash
cd backend
celery -A backend.services.task_queue worker -l info -c 4
```

**Start the frontend:**
```bash
cd frontend
npm run dev
```

### 4. Access the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## First Steps

### Upload Your First Document

1. Click **"Upload"** in the navigation
2. Drag and drop a PDF, Word doc, or image
3. Wait for processing to complete (you'll see real-time progress)
4. Your document is now searchable!

### Ask Your First Question

1. Go to the **Chat** interface
2. Type a question about your document
3. Get an AI-powered answer with source citations

### Create Your First AI Agent

1. Navigate to **Agents** → **Create Agent**
2. Name your agent and add a system prompt
3. Select the documents for your agent to learn from
4. Get an embed code to add to your website

## Next Steps

- [Uploading Documents](uploading-documents.md) - Learn about bulk uploads and supported formats
- [Querying Documents](querying-documents.md) - Advanced search and filtering
- [AI Agents](ai-agents.md) - Create custom chatbots
- [Architecture](../developer-guide/architecture.md) - Understand the system design

## Common Issues

### "Cannot connect to Redis"

Make sure Redis is running:
```bash
redis-cli ping
# Should return: PONG
```

### "API key not set"

Ensure you have at least one LLM API key in your `.env`:
```env
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

### "Document stuck in processing"

Check if Celery workers are running:
```bash
celery -A backend.services.task_queue inspect active
```

## Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search existing [GitHub Issues](https://github.com/your-org/AIDocumentIndexer/issues)
- Open a new issue with details about your problem
