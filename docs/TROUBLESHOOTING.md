# Troubleshooting Guide

Common issues and solutions for AIDocumentIndexer.

---

## Installation Issues

### Python Dependency Errors

**Problem:** `pip install` fails with dependency conflicts.

**Solution:**
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

### Node.js/npm Errors

**Problem:** `npm install` fails or hangs.

**Solution:**
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# If using specific Node version
nvm use 20
npm install
```

### Docker Build Failures

**Problem:** Docker build fails or runs out of memory.

**Solution:**
```bash
# Increase Docker memory (Docker Desktop)
# Settings → Resources → Memory → 8GB+

# Build with no cache
docker compose build --no-cache

# Check disk space
docker system df
docker system prune -a
```

---

## Database Issues

### Connection Refused

**Problem:** `Connection refused` error.

**Causes & Solutions:**

1. PostgreSQL not running:
   ```bash
   docker compose up -d db
   # or
   brew services start postgresql
   ```

2. Wrong port/host:
   ```bash
   # Check DATABASE_URL in .env
   DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
   ```

3. Firewall blocking:
   ```bash
   # Allow port 5432
   sudo ufw allow 5432
   ```

### Authentication Failed

**Problem:** `authentication failed for user`

**Solution:**
```bash
# Reset password in PostgreSQL
docker compose exec db psql -U postgres
ALTER USER username WITH PASSWORD 'newpassword';
```

### pgvector Extension Missing

**Problem:** `extension "vector" is not available`

**Solution:**
```bash
# Use pgvector Docker image
# In docker-compose.yml:
# image: pgvector/pgvector:pg15

# Or install manually
docker compose exec db psql -U postgres -d aidocindexer
CREATE EXTENSION vector;
```

### Migration Failures

**Problem:** Alembic migration fails.

**Solution:**
```bash
# Check current state
alembic current

# Stamp current revision
alembic stamp head

# Create fresh migration
alembic revision --autogenerate -m "Fix schema"
alembic upgrade head
```

---

## API Issues

### 401 Unauthorized

**Problem:** All API requests return 401.

**Causes & Solutions:**

1. Token expired:
   ```javascript
   // Refresh token
   const response = await fetch('/api/auth/refresh', {
     method: 'POST',
     headers: { Authorization: `Bearer ${token}` }
   });
   ```

2. Wrong JWT secret:
   ```bash
   # Ensure same JWT_SECRET in all services
   echo $JWT_SECRET
   ```

3. Token malformed:
   ```bash
   # Check token format
   # Should be: Authorization: Bearer <token>
   ```

### 403 Forbidden

**Problem:** User can't access resource.

**Solution:**
```bash
# Check user's access tier
curl -H "Authorization: Bearer $TOKEN" /api/auth/me

# Document requires higher tier
# Contact admin to upgrade access tier
```

### 500 Internal Server Error

**Problem:** Unexpected server errors.

**Solution:**
```bash
# Check backend logs
docker compose logs backend -f

# Enable debug mode
DEBUG=true uvicorn backend.api.main:app --reload
```

### CORS Errors

**Problem:** CORS policy blocked request.

**Solution:**
```bash
# Add origin to ALLOWED_ORIGINS
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Check if credentials are being sent
# Frontend must include credentials: 'include'
```

---

## LLM/Embedding Issues

### OpenAI API Errors

**Problem:** OpenAI requests fail.

**Causes & Solutions:**

1. Invalid API key:
   ```bash
   # Verify key format starts with sk-
   echo $OPENAI_API_KEY
   ```

2. Rate limited:
   ```python
   # Implement retry with backoff
   from tenacity import retry, wait_exponential

   @retry(wait=wait_exponential(min=1, max=60))
   async def call_openai():
       ...
   ```

3. Model not available:
   ```bash
   # Check available models
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### Ollama Connection Failed

**Problem:** Can't connect to Ollama.

**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull required model
ollama pull llama3.2
```

### Embedding Dimension Mismatch

**Problem:** `vector dimension mismatch`

**Solution:**
```bash
# Ensure EMBEDDING_DIMENSION matches model output
# OpenAI text-embedding-3-small: 1536
# nomic-embed-text: 768

EMBEDDING_DIMENSION=1536

# May need to recreate embeddings if changed
```

---

## Document Processing Issues

### File Upload Fails

**Problem:** File upload rejected.

**Causes & Solutions:**

1. File too large:
   ```bash
   # Increase limit
   MAX_FILE_SIZE=209715200  # 200MB
   ```

2. Unsupported type:
   ```bash
   # Check supported types
   curl /api/upload/supported-types
   ```

3. Disk full:
   ```bash
   df -h
   # Clean up old files
   ```

### Processing Stuck

**Problem:** Document stays in "processing" state.

**Solution:**
```bash
# Check Ray workers
ray status

# Restart processing
curl -X POST /api/documents/{id}/reprocess

# Check for errors
docker compose logs backend | grep ERROR
```

### OCR Not Working

**Problem:** Scanned PDFs not indexed.

**Solution:**
```bash
# Ensure OCR is enabled
ENABLE_OCR=true

# Check PaddleOCR installation
python -c "from paddleocr import PaddleOCR; print('OK')"

# Try with explicit language
OCR_LANGUAGE=en,ch
```

---

## Frontend Issues

### Page Won't Load

**Problem:** Blank page or loading forever.

**Solution:**
```bash
# Check for JavaScript errors
# Open browser DevTools (F12) → Console

# Rebuild frontend
npm run build

# Check API connection
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Theme Not Persisting

**Problem:** Dark/light mode resets on refresh.

**Solution:**
```javascript
// Ensure ThemeProvider wraps app
// Check localStorage for theme value
localStorage.getItem('theme')
```

### Search Not Returning Results

**Problem:** Search returns empty despite documents.

**Causes & Solutions:**

1. Documents not indexed:
   ```bash
   # Check document status
   curl /api/documents?status=completed
   ```

2. Access tier too low:
   ```bash
   # Documents may have higher access tier
   ```

3. Query too specific:
   ```bash
   # Try broader search terms
   ```

---

## Performance Issues

### Slow Queries

**Problem:** API responses are slow.

**Solution:**
```bash
# Enable query logging
LOG_LEVEL=DEBUG

# Add database indexes
CREATE INDEX idx_documents_status ON documents(processing_status);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);

# Enable Redis caching
REDIS_URL=redis://localhost:6379
```

### High Memory Usage

**Problem:** Services consuming too much memory.

**Solution:**
```bash
# Limit connection pools
DB_POOL_SIZE=5
REDIS_MAX_CONNECTIONS=10

# Reduce batch sizes
EMBEDDING_BATCH_SIZE=50

# Enable garbage collection
import gc
gc.collect()
```

### Slow Document Processing

**Problem:** Documents take too long to process.

**Solution:**
```bash
# Enable Ray for parallel processing
RAY_ENABLED=true

# Reduce chunk size
CHUNK_SIZE=500

# Process in batches
EMBEDDING_BATCH_SIZE=100
```

---

## Docker Issues

### Container Keeps Restarting

**Problem:** Container restarts in loop.

**Solution:**
```bash
# Check logs
docker compose logs backend

# Common causes:
# - Missing environment variables
# - Database not ready
# - Port already in use
```

### Volumes Not Persisting

**Problem:** Data lost after restart.

**Solution:**
```yaml
# docker-compose.yml
volumes:
  postgres_data:
    driver: local

services:
  db:
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

### Network Issues Between Containers

**Problem:** Containers can't communicate.

**Solution:**
```bash
# Check network
docker network ls

# Containers should be on same network
docker compose config | grep networks
```

---

## Getting Help

### Collect Debug Information

```bash
# System info
uname -a
python --version
node --version
docker --version

# Environment (remove sensitive values)
env | grep -E "DATABASE|REDIS|OPENAI" | sed 's/=.*/=***/'

# Recent logs
docker compose logs --tail=100 > logs.txt
```

### Report an Issue

1. Check existing issues on GitHub
2. Collect debug information above
3. Create minimal reproduction steps
4. Open new issue with:
   - Expected behavior
   - Actual behavior
   - Environment details
   - Logs and error messages
