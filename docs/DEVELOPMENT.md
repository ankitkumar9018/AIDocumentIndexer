# Development Guide

Guide for setting up and contributing to AIDocumentIndexer.

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- PostgreSQL 15+ with pgvector
- Redis 7+

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/AIDocumentIndexer.git
cd AIDocumentIndexer

# Copy environment files
cp .env.example .env
```

### 2. Backend Setup (with UV - Recommended)

```bash
cd backend

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv automatically)
uv sync

# Run migrations
uv run alembic upgrade head

# Start the server
uv run uvicorn backend.api.main:app --reload --port 8000
```

**Alternative: Using pip**

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start the server
uvicorn backend.api.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Using Docker

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

---

## Project Structure

```
AIDocumentIndexer/
├── backend/
│   ├── api/
│   │   ├── routes/          # API endpoints
│   │   └── middleware/      # Auth, CORS, etc.
│   ├── db/
│   │   ├── models.py        # SQLAlchemy models
│   │   └── database.py      # Database connection
│   ├── services/            # Business logic
│   ├── processors/          # Document processors
│   ├── langchain/           # RAG components
│   ├── ray/                 # Distributed processing
│   └── tests/               # Unit & integration tests
├── frontend/
│   ├── app/                 # Next.js App Router pages
│   ├── components/          # React components
│   ├── lib/                 # Utilities and API client
│   └── __tests__/           # Frontend tests
├── docker/                  # Docker configurations
├── docs/                    # Documentation
└── scripts/                 # Utility scripts
```

---

## Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make changes and test locally

3. Run linting and tests:
   ```bash
   # Backend (with UV)
   cd backend
   uv run ruff check .
   uv run mypy .
   uv run pytest

   # Backend (with pip/venv)
   cd backend
   ruff check .
   mypy .
   pytest

   # Frontend
   cd frontend
   npm run lint
   npm test
   ```

4. Commit with descriptive message:
   ```bash
   git commit -m "feat: add new document export feature"
   ```

5. Push and create PR

---

## Code Style

### Python (Backend)

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use `ruff` for linting
- Use `black` for formatting

```python
# Good example
async def process_document(
    document_id: str,
    options: ProcessingOptions,
) -> ProcessingResult:
    """Process a document with the given options."""
    ...
```

### TypeScript (Frontend)

- Use TypeScript strict mode
- Prefer functional components
- Use React Query for data fetching
- Follow ESLint rules

```typescript
// Good example
export function DocumentCard({ document }: DocumentCardProps) {
  const { data, isLoading } = useDocument(document.id);

  if (isLoading) return <Skeleton />;

  return <Card>{/* ... */}</Card>;
}
```

---

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/services/test_llm.py

# Run tests matching pattern
pytest -k "test_login"
```

### Frontend Tests

```bash
cd frontend

# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch

# Run specific test file
npm test -- button.test.tsx
```

---

## Database

### Migrations

```bash
cd backend

# Create a new migration
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View current revision
alembic current
```

### Seed Data

```bash
# Run seed script
python scripts/seed_data.py
```

---

## API Development

### Adding a New Endpoint

1. Create route in `backend/api/routes/`:

```python
# backend/api/routes/myfeature.py
from fastapi import APIRouter, Depends
from backend.api.middleware.auth import get_user_context

router = APIRouter()

@router.get("/myfeature")
async def get_myfeature(user=Depends(get_user_context)):
    return {"status": "ok"}
```

2. Register in main app:

```python
# backend/api/main.py
from backend.api.routes import myfeature

app.include_router(
    myfeature.router,
    prefix="/api/myfeature",
    tags=["myfeature"]
)
```

3. Add frontend hook:

```typescript
// frontend/lib/api/hooks.ts
export function useMyFeature() {
  return useQuery({
    queryKey: ['myfeature'],
    queryFn: () => api.getMyFeature(),
  });
}
```

---

## Environment Variables

See `.env.example` for all available options.

### Required Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aidocindexer

# Redis
REDIS_URL=redis://localhost:6379

# LLM
OPENAI_API_KEY=sk-...
```

### Optional Variables

```bash
# Authentication
JWT_SECRET=your-secret-key
JWT_EXPIRATION_HOURS=24

# LLM Providers
ANTHROPIC_API_KEY=sk-...
OLLAMA_HOST=http://localhost:11434

# Feature Flags
ENABLE_OCR=true
ENABLE_WEB_SCRAPER=true
```

---

## Debugging

### Backend Debugging

```python
# Use structlog for logging
import structlog
logger = structlog.get_logger(__name__)

logger.info("Processing document", document_id=doc_id)
```

### Frontend Debugging

```typescript
// Use React Query DevTools
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Add to app layout
<ReactQueryDevtools initialIsOpen={false} />
```

### Database Queries

```bash
# Connect to database
docker compose exec db psql -U postgres -d aidocindexer

# View recent queries
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

---

## Common Issues

### Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Database Connection Issues

```bash
# Check database is running
docker compose ps

# Restart database
docker compose restart db
```

### Node Modules Issues

```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
```

---

## IDE Setup

### VS Code Extensions

- Python
- Pylance
- ESLint
- Prettier
- Tailwind CSS IntelliSense

### VS Code Settings

```json
{
  "python.defaultInterpreterPath": "./backend/venv/bin/python",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
```

---

## Performance Tips

1. **Use connection pooling** for database connections
2. **Enable Redis caching** for frequently accessed data
3. **Use Ray** for parallel document processing
4. **Optimize embeddings** with batch processing
5. **Use SSR** for SEO-critical pages

---

## Getting Help

- Check existing issues on GitHub
- Review documentation in `/docs`
- Join community Discord
- Open a new issue with reproduction steps
