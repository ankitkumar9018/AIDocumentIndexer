# AIDocumentIndexer Installation Guide

This guide covers installation and setup for all three platforms:
1. **Web App** - Enterprise-ready web application
2. **Desktop App** - Local-first with offline mode (Tauri)
3. **Browser Extension** - Chrome extension for web capture

---

## Prerequisites

### System Requirements
- **Node.js** 18.x or higher
- **Python** 3.10 or higher
- **Rust** 1.70+ (for Desktop App only)
- **PostgreSQL** 14+ (for Web App backend)
- **Redis** 6+ (optional, for caching)

### Install Dependencies

```bash
# macOS
brew install node python rust postgresql redis

# Ubuntu/Debian
sudo apt update
sudo apt install nodejs npm python3 python3-pip postgresql redis-server
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Windows (using Chocolatey)
choco install nodejs python rust postgresql redis-64
```

---

## 1. Web App Installation

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/AIDocumentIndexer.git
cd AIDocumentIndexer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r backend/requirements.txt
# Or using uv (faster)
pip install uv
uv pip install -r backend/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration:
# - DATABASE_URL
# - OPENAI_API_KEY (or other LLM provider)
# - REDIS_URL (optional)

# Initialize database
python -m backend.db.init

# Run migrations
alembic upgrade head

# Start the backend server
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local
# Edit .env.local with:
# NEXT_PUBLIC_API_URL=http://localhost:8000

# Run development server
npm run dev

# Or build for production
npm run build
npm start
```

### Production Deployment

```bash
# Backend (with multiple workers)
gunicorn backend.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or using Docker
docker-compose up -d

# Frontend (production build)
cd frontend
npm run build
npm start
```

### Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# Or build individual images
docker build -t aidocindexer-backend -f Dockerfile.backend .
docker build -t aidocindexer-frontend -f Dockerfile.frontend .
```

### Kubernetes Deployment

AIDocumentIndexer is cloud-native with built-in support for Kubernetes health probes, graceful shutdown, and cloud logging.

**Prerequisites:**
- Kubernetes cluster (EKS, GKE, AKS, or self-hosted)
- kubectl configured
- Helm (optional, for chart deployment)

**Quick Start with kubectl:**
```bash
# Create namespace
kubectl create namespace aidocindexer

# Apply configurations
kubectl apply -f k8s/configmap.yaml -n aidocindexer
kubectl apply -f k8s/secret.yaml -n aidocindexer
kubectl apply -f k8s/deployment.yaml -n aidocindexer
kubectl apply -f k8s/service.yaml -n aidocindexer
kubectl apply -f k8s/ingress.yaml -n aidocindexer
```

**Example Deployment (k8s/deployment.yaml):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aidocindexer-api
  labels:
    app: aidocindexer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aidocindexer
  template:
    metadata:
      labels:
        app: aidocindexer
    spec:
      containers:
      - name: api
        image: aidocindexer:latest
        ports:
        - containerPort: 8000
        env:
        # Cloud context for logging
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        # Performance optimizations
        - name: PERF_COMPILE_CYTHON
          value: "true"
        - name: PERF_INIT_GPU
          value: "true"
        - name: PERF_INIT_MINHASH
          value: "true"
        # Database
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aidocindexer-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          failureThreshold: 3
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

**Health Endpoints:**
| Endpoint | Purpose | K8s Probe |
|----------|---------|-----------|
| `/health/live` | Is the process alive? | livenessProbe |
| `/health/ready` | Can it accept traffic? | readinessProbe |
| `/health` | Full health status | Monitoring |

### AWS Deployment

**ECS with Fargate:**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name aidocindexer-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster aidocindexer-cluster \
  --service-name aidocindexer-api \
  --task-definition aidocindexer:1 \
  --desired-count 3 \
  --launch-type FARGATE
```

**ECS Task Definition (ecs-task-definition.json):**
```json
{
  "family": "aidocindexer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-account.dkr.ecr.us-west-2.amazonaws.com/aidocindexer:latest",
      "portMappings": [
        {"containerPort": 8000, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "PERF_COMPILE_CYTHON", "value": "true"},
        {"name": "PERF_INIT_GPU", "value": "false"},
        {"name": "PERF_INIT_MINHASH", "value": "true"},
        {"name": "AWS_REGION", "value": "us-west-2"}
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:123456789:secret:aidocindexer/db-url"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health/ready || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/aidocindexer",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

**ALB Health Check Configuration:**
- Path: `/health/ready`
- Interval: 30 seconds
- Healthy threshold: 2
- Unhealthy threshold: 3

---

## 2. Desktop App Installation (Tauri)

The desktop app runs completely offline with local LLM support via Ollama.

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Tauri CLI
cargo install tauri-cli

# Install Ollama (for local LLM)
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

### Build from Source

```bash
# Navigate to desktop app directory
cd desktop-tauri

# Install frontend dependencies
npm install

# Install Rust dependencies (automatic on first build)

# Development mode
npm run tauri:dev

# Production build
npm run tauri:build
```

### Build Output Locations

After running `npm run tauri:build`, find installers at:

| Platform | Location |
|----------|----------|
| macOS | `desktop-tauri/src-tauri/target/release/bundle/dmg/` |
| Windows | `desktop-tauri/src-tauri/target/release/bundle/msi/` |
| Linux | `desktop-tauri/src-tauri/target/release/bundle/deb/` or `appimage/` |

### Generate Platform Icons

```bash
# Generate .ico (Windows) and .icns (macOS) from PNG
cd desktop-tauri
npx tauri icon public/icon.png
```

### Desktop App Configuration

On first launch:
1. Choose mode: **LOCAL** (offline) or **SERVER** (connected)
2. For LOCAL mode:
   - Ensure Ollama is running: `ollama serve`
   - Pull a model: `ollama pull llama3.2`
3. For SERVER mode:
   - Enter your server URL (e.g., `http://localhost:8000`)

---

## 3. Browser Extension Installation

### Development Installation (Chrome)

```bash
# Navigate to extension directory
cd browser-extension

# Install dependencies
npm install

# Build the extension
npm run build

# The built extension is in browser-extension/dist/
```

**Load in Chrome:**
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `browser-extension/dist` folder

### Development Mode (Hot Reload)

```bash
cd browser-extension
npm run dev
```

### Production Build for Chrome Web Store

```bash
cd browser-extension
npm run build

# Create ZIP for Chrome Web Store submission
cd dist
zip -r ../aidocindexer-extension.zip .
```

### Extension Configuration

1. Click the extension icon in Chrome toolbar
2. Go to Settings (gear icon)
3. Enter your server URL: `http://localhost:8000`
4. (Optional) Enter API key if authentication is enabled

### Firefox Support (Future)

The extension uses Manifest V3 which is Chrome-specific. For Firefox:
```bash
# Build for Firefox (requires manifest modifications)
npm run build:firefox
```

---

## Quick Start Guide

### Option A: Full Stack (Web + Backend)

```bash
# 1. Start backend
cd AIDocumentIndexer
source venv/bin/activate
uvicorn backend.api.main:app --port 8000

# 2. Start frontend (new terminal)
cd frontend
npm run dev

# 3. Open http://localhost:3000
```

### Option B: Desktop App (Offline)

```bash
# 1. Start Ollama
ollama serve

# 2. Pull a model (first time only)
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Run desktop app
cd desktop-tauri
npm run tauri:dev
```

### Option C: Browser Extension + Backend

```bash
# 1. Start backend
uvicorn backend.api.main:app --port 8000

# 2. Load extension in Chrome (see above)

# 3. Configure extension to point to http://localhost:8000
```

---

## Environment Variables

### Backend (.env)

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/aidocindexer

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379

# Optional: Embedding model
EMBEDDING_MODEL=text-embedding-3-small

# Security
SECRET_KEY=your-secret-key-here
```

### Performance Optimization Variables

```bash
# Cython Extensions (10-100x speedup for similarity computations)
PERF_COMPILE_CYTHON=true      # Compile Cython at startup (default: true)

# GPU Acceleration (5-20x speedup with CUDA/MPS)
PERF_INIT_GPU=true            # Initialize GPU accelerator (default: true)
PERF_GPU_PREFER=true          # Prefer GPU over CPU (default: true)
PERF_MIXED_PRECISION=true     # Use FP16 for 2x throughput (default: true)
PERF_WARMUP_GPU=false         # Run GPU warmup at startup (default: false)

# MinHash Deduplication (O(n) instead of O(n²))
PERF_INIT_MINHASH=true        # Initialize MinHash deduplicator (default: true)
PERF_MINHASH_PERMS=128        # Permutations (more = accurate, slower)
PERF_MINHASH_THRESHOLD=0.8    # Similarity threshold for duplicates
```

### Cloud/Kubernetes Variables

```bash
# Cloud Context (auto-detected if running in K8s)
POD_NAME=aidocindexer-abc123  # Kubernetes pod name (for logging)
POD_NAMESPACE=production      # Kubernetes namespace
NODE_NAME=node-1              # Kubernetes node name
CONTAINER_NAME=api            # Container name
AWS_REGION=us-west-2          # AWS region
CLUSTER_NAME=prod-cluster     # Cluster name

# Observability
OTLP_ENDPOINT=http://jaeger:4317  # OpenTelemetry collector endpoint
TRACING_ENABLED=true              # Enable distributed tracing
TRACING_SAMPLE_RATE=0.1           # Trace 10% of requests
```

### Frontend (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Desktop App (Settings UI)

Configured through the Settings page in the app:
- Server URL (for SERVER mode)
- Ollama Model (for LOCAL mode)
- Embedding Model
- Chunk Size / Overlap

---

## Troubleshooting

### Backend Issues

```bash
# Check if PostgreSQL is running
pg_isready

# Reset database
python -m backend.db.reset

# Check logs
tail -f logs/backend.log
```

### Desktop App Issues

```bash
# Check Ollama status
ollama list
curl http://localhost:11434/api/tags

# Rebuild Tauri
cd desktop-tauri
cargo clean
npm run tauri:build
```

### Extension Issues

```bash
# Rebuild extension
cd browser-extension
npm run clean
npm run build

# Check Chrome console for errors
# Right-click extension icon > Inspect popup
```

---

## Updating

### Web App

```bash
git pull origin main
pip install -r backend/requirements.txt
alembic upgrade head
cd frontend && npm install
```

### Desktop App

```bash
git pull origin main
cd desktop-tauri
npm install
npm run tauri:build
```

### Browser Extension

```bash
git pull origin main
cd browser-extension
npm install
npm run build
# Reload in chrome://extensions/
```

---

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/AIDocumentIndexer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/AIDocumentIndexer/discussions)
