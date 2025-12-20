# Server Commands Reference

Quick reference for starting, stopping, and restarting the AIDocumentIndexer servers.

## Quick Reference

| Action | Backend (Port 8000) | Frontend (Port 3000) |
|--------|---------------------|----------------------|
| Start | `source backend/.venv/bin/activate && PYTHONPATH=$(pwd) uvicorn backend.api.main:app --host 0.0.0.0 --port 8000` | `npm run dev --prefix frontend` |
| Stop | `pkill -f uvicorn` | `pkill -f "next-server"` |
| Health Check | `curl http://localhost:8000/health` | `curl -s -o /dev/null -w "%{http_code}" http://localhost:3000` |

---

## Backend Server

The backend runs on **port 8000** using FastAPI with Uvicorn.

### Start Backend

```bash
# From project root directory
source backend/.venv/bin/activate && PYTHONPATH=$(pwd) uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

**With auto-reload (development):**
```bash
source backend/.venv/bin/activate && PYTHONPATH=$(pwd) uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Run in background:**
```bash
source backend/.venv/bin/activate && PYTHONPATH=$(pwd) uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
```

### Stop Backend

```bash
# Option 1: Kill by process name
pkill -f uvicorn

# Option 2: Kill by port (forceful)
lsof -ti:8000 | xargs kill -9
```

### Restart Backend

```bash
# Stop then start
pkill -f uvicorn; sleep 2; source backend/.venv/bin/activate && PYTHONPATH=$(pwd) uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

### Verify Backend

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"AIDocumentIndexer","version":"0.1.0"}
```

---

## Frontend Server

The frontend runs on **port 3000** using Next.js.

### Start Frontend

```bash
# From project root directory
npm run dev --prefix frontend
```

**Or from frontend directory:**
```bash
cd frontend && npm run dev
```

**Run in background:**
```bash
npm run dev --prefix frontend > /tmp/frontend.log 2>&1 &
```

### Stop Frontend

```bash
# Option 1: Kill by process name
pkill -f "next-server"

# Option 2: Kill Node processes related to Next.js
pkill -f "node.*next"

# Option 3: Kill by port (forceful)
lsof -ti:3000 | xargs kill -9
```

### Restart Frontend

```bash
# Stop then start
pkill -f "next-server"; sleep 2; npm run dev --prefix frontend
```

### Verify Frontend

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
# Expected: 200
```

---

## Combined Commands

### Start Both Servers

```bash
# Start backend in background, then frontend
source backend/.venv/bin/activate && PYTHONPATH=$(pwd) uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
npm run dev --prefix frontend > /tmp/frontend.log 2>&1 &
echo "Servers starting... Backend: http://localhost:8000 | Frontend: http://localhost:3000"
```

### Stop Both Servers

```bash
pkill -f uvicorn; pkill -f "next-server"; lsof -ti:8000 | xargs kill -9 2>/dev/null; lsof -ti:3000 | xargs kill -9 2>/dev/null
echo "All servers stopped"
```

### Restart Both Servers

```bash
# Stop all
pkill -f uvicorn; pkill -f "next-server"; lsof -ti:8000 | xargs kill -9 2>/dev/null; lsof -ti:3000 | xargs kill -9 2>/dev/null
sleep 2
# Start all
source backend/.venv/bin/activate && PYTHONPATH=$(pwd) uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
npm run dev --prefix frontend > /tmp/frontend.log 2>&1 &
echo "Servers restarting..."
```

---

## Viewing Logs

### Backend Logs

```bash
# If running in background with log file
tail -f /tmp/backend.log

# View last 50 lines
tail -50 /tmp/backend.log
```

### Frontend Logs

```bash
# If running in background with log file
tail -f /tmp/frontend.log

# View last 50 lines
tail -50 /tmp/frontend.log
```

---

## Health Checks

### Check Both Servers

```bash
echo "=== Backend ===" && curl -s http://localhost:8000/health && echo "" && echo "=== Frontend ===" && curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:3000
```

### Check What's Running on Ports

```bash
# Check port 8000 (backend)
lsof -i :8000

# Check port 3000 (frontend)
lsof -i :3000
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Backend Won't Start

1. **Check virtual environment:**
   ```bash
   source backend/.venv/bin/activate
   which python  # Should point to backend/.venv/bin/python
   ```

2. **Check PYTHONPATH:**
   ```bash
   # Must be set to project root
   export PYTHONPATH=$(pwd)
   ```

3. **Test import:**
   ```bash
   source backend/.venv/bin/activate && PYTHONPATH=$(pwd) python -c "from backend.api.main import app; print('OK')"
   ```

### Frontend Won't Start

1. **Check Node modules:**
   ```bash
   cd frontend && npm install
   ```

2. **Clear Next.js cache:**
   ```bash
   rm -rf frontend/.next
   npm run dev --prefix frontend
   ```

### Connection Timeout

If the backend starts but `curl` times out:

1. Check if using `--reload` flag (can be slow on first load)
2. Wait longer for startup (15-20 seconds)
3. Check logs for errors: `tail -50 /tmp/backend.log`

---

## Docker Alternative

If you prefer Docker, see [DEPLOYMENT.md](DEPLOYMENT.md) for Docker Compose commands:

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f
```
