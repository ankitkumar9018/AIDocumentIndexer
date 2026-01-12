# Installation Guide

This guide covers all installation methods for AIDocumentIndexer.

---

## Table of Contents

1. [Quick Start (Docker)](#quick-start-docker)
2. [Manual Installation (No Docker)](#manual-installation-no-docker)
3. [Hybrid Setup](#hybrid-setup)
4. [Cloud Deployment](#cloud-deployment)
5. [Development Setup](#development-setup)

---

## Quick Start (Docker)

The easiest way to run AIDocumentIndexer with all dependencies.

### Prerequisites
- Docker Engine 24.0+
- Docker Compose 2.0+
- 8GB RAM minimum (16GB recommended)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AIDocumentIndexer.git
cd AIDocumentIndexer

# 2. Copy and configure environment
cp .env.example .env
nano .env  # Add your OpenAI API key

# 3. Start all services
docker-compose -f docker/docker-compose.yml up -d

# 4. Run database migrations (required for new installations)
docker-compose -f docker/docker-compose.yml exec backend \
  alembic -c backend/alembic.ini upgrade head

# 5. Check status
docker-compose -f docker/docker-compose.yml ps

# 6. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# Ray Dashboard: http://localhost:8265
```

**Note:** Migrations run automatically on startup in development mode. For production, run migrations explicitly before deploying new versions.

---

## Manual Installation (No Docker)

Run AIDocumentIndexer directly on your system without Docker.

### Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **PostgreSQL 15+** with pgvector extension
- **Redis 7+**
- **Tesseract OCR** (optional, required for scanned PDF support)
- **Ollama** (optional, for local LLM)

### Step 1: Install System Dependencies

#### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 node@20 postgresql@16 redis

# Install pgvector extension
brew install pgvector

# Install Tesseract OCR (required for scanned PDF support)
brew install tesseract

# Start services
brew services start postgresql@16
brew services start redis
```

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs

# Install PostgreSQL with pgvector
sudo apt install postgresql-16 postgresql-16-pgvector

# Install Redis
sudo apt install redis-server

# Install Tesseract OCR (required for scanned PDF support)
sudo apt install tesseract-ocr tesseract-ocr-eng

# Start services
sudo systemctl start postgresql
sudo systemctl start redis
```

#### Windows
```powershell
# Install using winget or Chocolatey
winget install Python.Python.3.11
winget install OpenJS.NodeJS.LTS
winget install PostgreSQL.PostgreSQL
winget install Redis.Redis

# Install Tesseract OCR (required for scanned PDF support)
winget install UB-Mannheim.TesseractOCR
# Add to PATH: C:\Program Files\Tesseract-OCR

# Note: pgvector requires manual installation on Windows
# See: https://github.com/pgvector/pgvector#windows
```

### Step 2: Set Up PostgreSQL

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE USER aidoc WITH PASSWORD 'your_password';
CREATE DATABASE aidocindexer OWNER aidoc;

# Connect to the database
\c aidocindexer

# Enable pgvector extension
CREATE EXTENSION vector;

# Run initialization script
\i /path/to/AIDocumentIndexer/db/init.sql

# Exit
\q
```

### Step 3: Set Up Python Backend

```bash
# Navigate to project
cd AIDocumentIndexer

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Install PaddleOCR (may require additional setup)
pip install paddlepaddle paddleocr
```

### Step 3.5: Initialize the Database

AIDocumentIndexer supports two database modes. Choose based on your needs:

---

#### Option A: SQLite (Development / Quick Start)

SQLite is the simplest option for development and testing. Tables are created automatically.

**Step 1: Configure environment**
```bash
cd AIDocumentIndexer

# Copy environment template
cp .env.example .env

# Edit .env and set SQLite database:
# DATABASE_URL=sqlite:////path/to/AIDocumentIndexer/aidocindexer.db
```

**Step 2: Set environment variables**
```bash
# Set environment variables (or add to .env)
export DATABASE_URL=sqlite:////Users/yourname/AIDocumentIndexer/aidocindexer.db
export APP_ENV=development
export PYTHONPATH=.
```

**Step 3: Start the backend (tables created automatically)**
```bash
source venv/bin/activate
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Tables are created automatically via SQLAlchemy on first startup. No migrations needed for SQLite.

**SQLite Limitations:**
- No pgvector (vector search uses slower brute-force method)
- No full-text search indexes
- No row-level security
- Single-user only (file locking)
- **Not recommended for production**

---

#### Option B: PostgreSQL (Production / Full Features)

PostgreSQL with pgvector provides the best performance and all features.

**Step 1: Install PostgreSQL and pgvector**

```bash
# macOS
brew install postgresql@16 pgvector
brew services start postgresql@16

# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector
sudo systemctl start postgresql

# Windows
# Download PostgreSQL from https://www.postgresql.org/download/windows/
# Install pgvector manually: https://github.com/pgvector/pgvector#windows
```

**Step 2: Create database and enable pgvector**

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Run these SQL commands:
CREATE USER aidoc WITH PASSWORD 'your_secure_password';
CREATE DATABASE aidocindexer OWNER aidoc;
\c aidocindexer
CREATE EXTENSION vector;
GRANT ALL PRIVILEGES ON DATABASE aidocindexer TO aidoc;
\q
```

**Step 3: Configure environment**

```bash
cd AIDocumentIndexer
cp .env.example .env

# Edit .env and set:
# DATABASE_URL=postgresql://aidoc:your_secure_password@localhost:5432/aidocindexer
```

**Step 4: Run database migrations**

```bash
source venv/bin/activate
export DATABASE_URL=postgresql://aidoc:your_secure_password@localhost:5432/aidocindexer
export APP_ENV=development
export PYTHONPATH=.

# Run all migrations
alembic -c backend/alembic.ini upgrade head

# Verify migration success (should show 20260102_012)
alembic -c backend/alembic.ini current
```

**Step 5: Start the backend**

```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

### PostgreSQL Migration Reference

**Full migration chain:**
```
001_initial_schema           → Core tables, pgvector, RLS policies
002_ai_optimization          → AI optimization fields
003_ocr_settings             → OCR configuration
20251230_004                 → OCR metrics
20251230_005                 → EasyOCR settings
20251230_006                 → Performance indexes
20251230_007                 → Chat feedback
20260102_008                 → HNSW + FTS indexes (critical for scale)
20260102_009                 → Upload jobs table
20260102_010                 → Folders table
20260102_011                 → User preferences table
20260102_012 (head)          → Saved searches column
```

**Common migration commands:**
```bash
# Check current version
alembic -c backend/alembic.ini current

# Run all pending migrations
alembic -c backend/alembic.ini upgrade head

# Run migrations to specific version
alembic -c backend/alembic.ini upgrade 20260102_008

# Rollback one migration
alembic -c backend/alembic.ini downgrade -1

# View migration history
alembic -c backend/alembic.ini history --verbose

# Generate new migration after model changes
alembic -c backend/alembic.ini revision --autogenerate -m "description"

# Mark database as up-to-date (skip migrations)
alembic -c backend/alembic.ini stamp head
```

**Troubleshooting migrations:**

| Error | Cause | Solution |
|-------|-------|----------|
| "relation already exists" | Table created outside migrations | Run `alembic stamp head` |
| "extension vector does not exist" | pgvector not installed | Install pgvector and run `CREATE EXTENSION vector;` |
| "no such revision" | Migration chain broken | Check down_revision in migration files |
| "CONCURRENTLY cannot be executed from a function" | Migration in transaction | Run index creation manually outside alembic |

### Step 4: Set Up Node.js Frontend

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Build for production (optional)
npm run build
```

### Step 5: Install Ray (Optional but Recommended)

```bash
# Install Ray
pip install ray[default]

# Start Ray head node (in a separate terminal)
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# To stop Ray later:
# ray stop
```

### Step 6: Install Ollama (Optional, for Local LLM)

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

### Step 7: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Key settings for non-Docker setup:
```bash
# Database (local PostgreSQL)
DATABASE_URL=postgresql://aidoc:your_password@localhost:5432/aidocindexer

# Redis (local)
REDIS_URL=redis://localhost:6379/0

# Ray (local)
RAY_ADDRESS=auto

# Ollama (local)
OLLAMA_HOST=http://localhost:11434
```

### Step 8: Run the Application

**Option A: Development Mode (with hot reload)**

```bash
# Terminal 1: Backend
source venv/bin/activate
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

**Option B: Production Mode**

```bash
# Terminal 1: Backend
source venv/bin/activate
gunicorn backend.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Terminal 2: Frontend
cd frontend
npm run build
npm start
```

### Step 9: Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Ray Dashboard**: http://localhost:8265 (if Ray is running)

---

## Hybrid Setup

Use Docker for some services and run others locally.

### Example: Docker for Databases, Local for Application

```bash
# Start only database services
docker-compose -f docker/docker-compose.yml up -d postgres redis

# Run backend locally
source venv/bin/activate
uvicorn backend.api.main:app --reload

# Run frontend locally
cd frontend && npm run dev
```

### Example: Local PostgreSQL, Docker for Everything Else

Edit `.env`:
```bash
DATABASE_URL=postgresql://aidoc:password@host.docker.internal:5432/aidocindexer
```

---

## Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed cloud deployment instructions:

- AWS (ECS, RDS, ElastiCache)
- Google Cloud (Cloud Run, Cloud SQL)
- Azure (Container Apps, Azure Database)
- Railway, Render, Fly.io

---

## Development Setup

For contributing to AIDocumentIndexer:

```bash
# Clone with git
git clone https://github.com/yourusername/AIDocumentIndexer.git
cd AIDocumentIndexer

# Install development dependencies
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt  # If available

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest backend/tests/

# Run linting
ruff check backend/
mypy backend/

# Frontend development
cd frontend
npm install
npm run lint
npm test
```

---

## Troubleshooting

### PostgreSQL pgvector Not Found

```bash
# macOS
brew install pgvector

# Ubuntu
sudo apt install postgresql-16-pgvector

# From source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install
```

Then enable in PostgreSQL:
```sql
CREATE EXTENSION vector;
```

### PaddleOCR Installation Issues

```bash
# Install with CUDA support (if you have NVIDIA GPU)
pip install paddlepaddle-gpu paddleocr

# CPU only
pip install paddlepaddle paddleocr

# If issues persist, try:
pip install paddlepaddle==2.5.2 paddleocr==2.7.0
```

### Ray Connection Issues

```bash
# Check if Ray is running
ray status

# Stop existing Ray instance
ray stop

# Start fresh
ray start --head
```

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce `RAY_NUM_CPUS` in `.env`
2. Use `basic` processing mode
3. Process documents in smaller batches
4. Increase system swap space

---

## System Requirements

### Minimum
- 4 CPU cores
- 8GB RAM
- 20GB storage
- Python 3.11+
- Node.js 20+

### Recommended
- 8+ CPU cores
- 16GB+ RAM
- 100GB+ SSD storage
- GPU (for local LLM inference)

### For 1000+ Documents
- 16+ CPU cores
- 32GB+ RAM
- 500GB+ SSD storage
- Ray cluster with multiple workers
