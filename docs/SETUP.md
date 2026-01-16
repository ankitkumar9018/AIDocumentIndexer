# Automated Setup Guide

This guide covers the automated setup script for AIDocumentIndexer, which handles all dependency installation, service configuration, and startup in a single command.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What the Script Does](#what-the-script-does)
3. [Command Line Options](#command-line-options)
4. [Dependencies Configuration](#dependencies-configuration)
5. [Service Management](#service-management)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

Run the setup script from the project root:

```bash
# Full setup with all services
python scripts/setup.py

# Setup with verbose output for debugging
python scripts/setup.py --verbose

# Setup without starting services (just install dependencies)
python scripts/setup.py --skip-services

# Setup without Ollama (use cloud LLM providers)
python scripts/setup.py --skip-ollama
```

After successful setup, access:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Ollama**: http://localhost:11434
- **Redis**: localhost:6379

---

## What the Script Does

The setup script performs the following steps in order:

### 1. Check System Dependencies
Verifies required tools are installed:
- **Required**: `python3`, `node`, `npm`
- **Optional**: `soffice` (LibreOffice), `tesseract`, `ffmpeg`, `poppler`, `redis-server`, `ray`

### 2. Stop Existing Services
Kills any processes running on ports 8000, 3000, 6379, 11434 to prevent conflicts.

### 3. Install Python Dependencies
Uses UV (preferred) or pip to install backend packages:
```bash
cd backend && uv sync
```

### 4. Install Node.js Dependencies
Installs frontend packages:
```bash
cd frontend && npm install
```

### 5. Setup Ollama (Optional)
- Installs Ollama if not present
- Starts Ollama service
- Pulls required models (skips if already installed):
  - `llama3.2:latest` - Text model for chat and generation
  - `nomic-embed-text:latest` - Embedding model for RAG
  - `llava:latest` - Vision model for image understanding

### 6. Create Environment Files
Generates `.env` files if missing:
- `backend/.env` (from `.env.example`)
- `frontend/.env.local` (auto-generated with secrets)

### 7. Run Database Migrations
Applies Alembic migrations:
```bash
uv run alembic upgrade head
```

### 8. Start Redis (Optional)
Starts Redis server for caching and Celery task queue.

### 9. Start Celery Worker (Optional)
Starts Celery worker for async document processing (requires Redis).

### 10. Start Backend & Frontend
Launches services as background processes with logs in `logs/` directory.

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `--skip-services` | Skip starting backend/frontend services |
| `--skip-ollama` | Skip Ollama installation and model pulling |
| `--skip-redis` | Skip Redis startup (also skips Celery) |
| `--skip-celery` | Skip Celery worker startup |
| `--pull-optional` | Also pull optional Ollama models (mistral, codellama, llama3.3:70b) |
| `--verbose`, `-v` | Enable verbose/debug output |
| `--log-file PATH` | Write logs to specified file (default: `logs/setup.log`) |

### Examples

```bash
# Development setup without heavy models
python scripts/setup.py --skip-ollama --verbose

# Production setup with all optional models
python scripts/setup.py --pull-optional

# Just install dependencies, don't start anything
python scripts/setup.py --skip-services --skip-ollama --skip-redis

# Debug installation issues
python scripts/setup.py --verbose --log-file /tmp/setup-debug.log
```

---

## Dependencies Configuration

All dependencies are configured in `scripts/dependencies.json`:

### System Dependencies

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
  }
}
```

### Ollama Models

```json
{
  "ollama": {
    "models": {
      "text": ["llama3.2:latest"],
      "embedding": ["nomic-embed-text:latest"],
      "vision": ["llava:latest"]
    },
    "optional_models": {
      "text": ["llama3.1:8b", "mistral:latest", "codellama:latest"],
      "vision": ["llava:13b", "bakllava:latest"],
      "large": ["llama3.3:70b"]
    }
  }
}
```

### Service Ports

| Service | Port | Required |
|---------|------|----------|
| Backend | 8000 | Yes |
| Frontend | 3000 | Yes |
| Ollama | 11434 | No |
| Redis | 6379 | No |

---

## Service Management

### Starting Services Manually

If you need to start services manually after setup:

```bash
# Backend (from project root)
cd backend && uv run uvicorn backend.api.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev

# Redis
redis-server

# Celery worker
uv run celery -A backend.services.task_queue worker --loglevel=info

# Ollama
ollama serve
```

### Viewing Logs

Service logs are stored in `logs/`:
- `logs/backend.log` - Backend uvicorn output
- `logs/frontend.log` - Frontend Next.js output
- `logs/redis.log` - Redis server output
- `logs/celery.log` - Celery worker output
- `logs/setup.log` - Setup script output

```bash
# Follow backend logs
tail -f logs/backend.log

# Follow all logs
tail -f logs/*.log
```

### Stopping Services

```bash
# Stop process on specific port
lsof -ti :8000 | xargs kill -9  # Backend
lsof -ti :3000 | xargs kill -9  # Frontend

# Stop Redis
redis-cli shutdown

# Stop Ollama
ollama stop
```

---

## Troubleshooting

### Setup Script Fails

**Check the log file:**
```bash
cat logs/setup.log
```

**Run with verbose output:**
```bash
python scripts/setup.py --verbose
```

### Missing System Dependencies

The script will warn about missing optional dependencies. Install them based on your OS:

**macOS (Homebrew):**
```bash
brew install libreoffice tesseract ffmpeg poppler redis jq
pip install ray[default]
```

**Ubuntu/Debian:**
```bash
sudo apt install libreoffice tesseract-ocr ffmpeg poppler-utils redis-server jq
pip install ray[default]
```

### Ollama Model Pull Fails

If model pulling fails, try manually:
```bash
# Check Ollama is running
ollama list

# Pull model manually
ollama pull llama3.2:latest

# Check disk space (models are 4-8GB each)
df -h
```

### Database Migration Fails

If migrations fail:
```bash
cd backend

# Check current migration status
uv run alembic current

# View migration history
uv run alembic history

# Try upgrading to latest
uv run alembic upgrade head
```

### Port Already in Use

If a port is already in use:
```bash
# Find process using port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or let the setup script handle it (it kills existing processes automatically)
```

### Redis Connection Issues

If Redis won't start:
```bash
# Check if Redis is already running
redis-cli ping

# Check Redis logs
cat logs/redis.log

# Start Redis manually
redis-server --daemonize yes
```

### UV Not Found

If UV is not installed:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"

# Or use pip fallback
pip install -r backend/requirements.txt
```

---

## Platform-Specific Notes

### macOS

```bash
# Install system dependencies with Homebrew
brew install python@3.11 node@20 redis
brew install --cask libreoffice
brew install tesseract ffmpeg poppler jq pango

# Start services on boot (optional)
brew services start redis
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv nodejs npm redis-server
sudo apt install libreoffice tesseract-ocr ffmpeg poppler-utils jq
sudo apt install libpango-1.0-0 libharfbuzz0b  # For WeasyPrint PDF generation

# Enable services
sudo systemctl enable redis-server
```

### Windows

```powershell
# Install with winget
winget install Python.Python.3.11
winget install OpenJS.NodeJS.LTS
winget install Gyan.FFmpeg
winget install UB-Mannheim.TesseractOCR

# Run setup
python scripts\setup.py
```

---

## Related Documentation

- [INSTALLATION.md](./INSTALLATION.md) - Manual installation guide
- [DEVELOPMENT.md](./DEVELOPMENT.md) - Development workflow
- [CONFIGURATION.md](./CONFIGURATION.md) - Environment variables
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
