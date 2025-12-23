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

# Pull required models
ollama pull llama3.2           # Chat model
ollama pull nomic-embed-text   # Embedding model

# List available models
ollama list
```

### Ollama Embeddings Not Working

**Problem:** Using Ollama but embeddings fail with OpenAI API key error.

**Causes & Solutions:**

1. **Wrong provider configured:**
   ```bash
   # Ensure DEFAULT_LLM_PROVIDER is set to ollama
   DEFAULT_LLM_PROVIDER=ollama
   ```

2. **RAG service using wrong provider:**
   The RAG service now reads `DEFAULT_LLM_PROVIDER` from environment. Ensure your `.env` is correct and restart the backend.

3. **Embedding model not pulled:**
   ```bash
   ollama pull nomic-embed-text
   ```

### Embedding Dimension Mismatch

**Problem:** `vector dimension mismatch` or embeddings don't match

**Causes:**
- Switched embedding models after indexing documents
- `EMBEDDING_DIMENSION` doesn't match actual model output

**Solution:**
```bash
# Ensure EMBEDDING_DIMENSION matches model output
# OpenAI text-embedding-3-small: 1536
# OpenAI text-embedding-3-large: 3072
# Ollama nomic-embed-text: 768
# HuggingFace all-MiniLM-L6-v2: 384

EMBEDDING_DIMENSION=768  # For Ollama nomic-embed-text

# If you changed models, you need to:
# 1. Clear ChromaDB: rm -rf ./data/chroma
# 2. Reprocess all documents
```

### LangChain Deprecation Warnings

**Problem:** Logs show `LangChainDeprecationWarning: The class 'OllamaEmbeddings' was deprecated`

**Solution:** These are warnings, not errors. The code will continue to work. To suppress warnings, you can upgrade langchain-ollama:
```bash
pip install -U langchain-ollama
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

### Tesseract OCR Not Installed

**Problem:** Scanned PDFs process with 0 chunks, backend logs show `tesseract is not installed or it's not in your PATH`.

**Solution:**

Tesseract OCR is a system binary that must be installed separately from Python packages.

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-eng

# Windows
winget install UB-Mannheim.TesseractOCR
# Add to PATH: C:\Program Files\Tesseract-OCR

# Verify installation
tesseract --version
```

**Note:** The `pytesseract` Python package is just a wrapper - the Tesseract binary must be installed on the system. After installation, restart the backend server.

### Tesseract Language Data Not Found

**Problem:** PDF shows status "completed" but has 0 chunks and 0 words. Backend logs show:
```
Error opening data file /opt/homebrew/share/tessdata/en.traineddata
Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory.
Failed loading language 'en'
```

**Solution:**

1. Set the `TESSDATA_PREFIX` environment variable in `.env`:
   ```bash
   # macOS (Homebrew)
   TESSDATA_PREFIX=/opt/homebrew/share/tessdata

   # Linux (apt)
   TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

   # Windows
   TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
   ```

2. Verify the language data files exist:
   ```bash
   ls $TESSDATA_PREFIX
   # Should show: eng.traineddata, osd.traineddata, etc.
   ```

3. Restart the backend server after changing `.env`.

### OCR Language Not Installed

**Problem:** Non-English PDFs (e.g., German, French) show 0 words after processing.

**Solution:**

1. Install additional language packs:
   ```bash
   # macOS - install all languages
   brew install tesseract-lang

   # Ubuntu/Debian - specific language
   sudo apt install tesseract-ocr-deu  # German
   sudo apt install tesseract-ocr-fra  # French
   ```

2. Configure the OCR language in `.env`:
   ```bash
   # German + English
   OCR_LANGUAGE=deu+eng

   # French + English
   OCR_LANGUAGE=fra+eng
   ```

3. Reprocess the document via the UI "Reprocess" button or API.

### PDF Shows "Completed" But Has 0 Chunks

**Problem:** Document processing completes successfully but word_count and chunk_count are 0.

**Causes & Solutions:**

1. **OCR failing silently** - Check for Tesseract errors in logs:
   ```bash
   grep -i "tesseract\|ocr" /tmp/backend.log
   ```

2. **PDF contains only images** (scanned document):
   - Ensure `ENABLE_OCR=true`
   - Verify `TESSDATA_PREFIX` is set correctly
   - Check `OCR_LANGUAGE` matches document language

3. **Text extraction returning empty**:
   ```bash
   # Test PDF text extraction manually
   PYTHONPATH=. python -c "
   from backend.processors.universal import UniversalProcessor
   p = UniversalProcessor()
   result = p.process_file('path/to/file.pdf')
   print(f'Text length: {len(result.text)}')
   "
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

### Document Names Showing "unknown"

**Problem:** Documents page shows "unknown" as document name, or Sources panel in Chat shows "unknown" instead of actual filenames.

**Causes & Solutions:**

1. **Sources panel showing "unknown":**
   This was caused by a metadata key mismatch in `backend/services/rag.py`. The vector store uses `document_filename` but RAG service was looking for `document_name`. Fixed in December 2024 to check both keys.

2. **Documents page showing "unknown":**
   The `original_filename` field was "unknown" in the database. This can happen when:
   - Documents were reprocessed without preserving original filenames
   - Files were uploaded with missing metadata

3. **Fix for existing documents:**
   ```bash
   # Match files from input_files folder by hash
   # This admin endpoint recovers original filenames
   curl -X POST "http://localhost:8000/api/v1/documents/fix-names" \
     -H "Authorization: Bearer $TOKEN"
   ```

4. **PDF metadata showing instead of filename:**
   If documents show titles like "Document.pdf" or "Mastervorlage 16:9", this is PDF metadata. The fix prioritizes `original_filename` over `title`:
   ```python
   # In backend/api/routes/documents.py
   name=doc.original_filename or doc.title  # Filename first
   ```

### Search/Chat Not Returning Results

**Problem:** Chat says "no relevant documents found" even though documents are indexed.

**Causes & Solutions:**

1. **Documents not indexed:**
   ```bash
   # Check document status
   curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/documents?status=completed
   ```

2. **Embedding provider mismatch:**
   The chat/RAG service might be using a different embedding provider than the one used to index documents.
   ```bash
   # Ensure DEFAULT_LLM_PROVIDER matches what was used during indexing
   DEFAULT_LLM_PROVIDER=ollama  # or openai
   ```

3. **Similarity threshold too high:**
   OCR'd documents may have lower similarity scores. The default threshold has been lowered to 0.4.
   ```bash
   # Check vectorstore.py VectorStoreConfig
   similarity_threshold: float = 0.4  # Lower for OCR documents
   ```

4. **Embedding dimension mismatch:**
   If you switched embedding models, old embeddings won't match.
   ```bash
   # Check EMBEDDING_DIMENSION matches current model
   EMBEDDING_DIMENSION=768   # nomic-embed-text (Ollama)
   EMBEDDING_DIMENSION=1536  # text-embedding-3-small (OpenAI)
   ```

5. **ChromaDB collection empty or corrupted:**
   ```bash
   # Check ChromaDB document count
   PYTHONPATH=. python -c "
   import chromadb
   client = chromadb.PersistentClient(path='./data/chroma')
   collection = client.get_collection('documents')
   print(f'Documents in ChromaDB: {collection.count()}')
   "
   ```

### RAG Returns "EmbeddingService has no attribute" Error

**Problem:** Backend logs show `'EmbeddingService' object has no attribute 'embed_query'`

**Solution:** This was a bug that has been fixed. The `EmbeddingService` class now has the `embed_query` async method. Update your code to the latest version.

### Chat History Not Appearing

**Problem:** Clicking "History" button shows no conversations. Chat sessions are not persisting.

**Causes & Solutions:**

1. **Sessions not being created:**
   ```bash
   # Check if sessions exist in database
   sqlite3 aidocindexer.db "SELECT id, title, created_at FROM chat_sessions LIMIT 5"
   ```

2. **User ID mismatch:**
   The JWT token's `sub` field may contain a non-UUID identifier (e.g., "test-user-123") while the database expects UUID foreign keys. This was fixed in December 2024 with a `get_db_user_id()` helper that:
   - Looks up user by email in database
   - Falls back to admin user if not found
   - Returns valid UUID for session creation

3. **Database error in session creation:**
   ```bash
   # Check backend logs for errors
   grep -i "session\|chat" /tmp/backend.log | tail -20
   ```

4. **Fix applied:** If you're running an older version, update to get the `get_db_user_id()` helper in `backend/api/routes/chat.py`.

**Verification:**
```bash
# After chatting, verify sessions are created
sqlite3 aidocindexer.db "SELECT COUNT(*) FROM chat_sessions"
# Should show > 0

# Check messages are saved
sqlite3 aidocindexer.db "SELECT COUNT(*) FROM chat_messages"
```

### Documents Found But Not Relevant

**Problem:** RAG returns sources but they're from wrong documents (e.g., test documents instead of real content).

**Causes & Solutions:**

1. **Documents indexed but chunks stored incorrectly:**
   ```bash
   # Check if chunks exist in SQLite
   sqlite3 aidocindexer.db "SELECT COUNT(*) FROM chunks WHERE document_id='YOUR_DOC_ID'"
   ```

2. **ChromaDB and SQLite out of sync:**
   ```bash
   # Verify both have the same count
   # SQLite
   sqlite3 aidocindexer.db "SELECT COUNT(*) FROM chunks"

   # ChromaDB
   PYTHONPATH=. python -c "
   import chromadb
   client = chromadb.PersistentClient(path='./data/chroma')
   print(client.get_collection('documents').count())
   "
   ```

3. **Reindex documents:**
   Use the "Reprocess" button in the UI or call the reprocess API endpoint.

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
