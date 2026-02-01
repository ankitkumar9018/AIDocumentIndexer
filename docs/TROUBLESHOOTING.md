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
pip install --no-cache-dir -e ".[dev]"
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

### Image Captioning Using OpenAI Instead of Ollama

**Problem:** Image captioning fails with `401 - Incorrect API key provided: sk-your-***` even though Ollama is configured.

**Causes & Solutions:**

1. **Vision provider not explicitly set to `ollama`:**
   ```bash
   # Check current setting in Admin UI:
   # Admin > Settings > RAG Configuration > rag.vision_provider

   # Or check database directly:
   sqlite3 backend/data/aidocindexer.db \
     "SELECT value FROM system_settings WHERE key='rag.vision_provider';"

   # Should return: ollama
   ```

2. **Celery workers running old code (see Celery section below)**

3. **Ollama llava model not installed:**
   ```bash
   # Pull the vision model
   ollama pull llava

   # Verify it's installed
   ollama list | grep llava
   ```

4. **Cached image captions from previous failed attempts:**
   ```bash
   # Clear cached error captions
   sqlite3 backend/data/aidocindexer.db \
     "DELETE FROM analyzed_images WHERE caption LIKE '%401%' OR caption LIKE '%error%';"
   ```

**Verifying Correct Configuration:**
After fixing, check Celery logs for these messages:
```
Vision model selection         db_model=llava db_provider=ollama
Using Ollama vision model (from DB setting): llava at http://localhost:11434
```

If you see these messages, image captioning is correctly using Ollama.

---

## Celery Worker Issues

### Celery Workers Not Picking Up Code Changes

**Problem:** After modifying Python files, Celery workers still run old code. Changes like logging statements don't appear in logs.

**Root Cause:** Python bytecode cache (`.pyc` files and `__pycache__` directories) can cause workers to load stale compiled code even after restart.

**Solution:**
```bash
# Stop Celery workers
pkill -f "celery.*worker"

# Clear Python bytecode cache
find backend -name "*.pyc" -delete
find backend -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Restart with setup.py (recommended - automatically clears cache)
python scripts/setup.py

# OR restart manually:
PYTHONPATH=/path/to/project \
  uv run --project backend celery -A backend.services.task_queue:celery_app \
  worker --loglevel=info --pool=threads --concurrency=4
```

**Note:** As of the latest update, `setup.py` automatically clears the bytecode cache before starting Celery workers to prevent this issue.

### Celery Worker Crashes or Won't Start

**Problem:** Celery worker exits immediately or shows import errors.

**Solution:**
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Start Redis if needed
redis-server &

# Check for import errors by running Celery in foreground
PYTHONPATH=/path/to/project \
  uv run --project backend celery -A backend.services.task_queue:celery_app \
  worker --loglevel=debug

# Common fixes:
# 1. Ensure PYTHONPATH includes project root
# 2. Clear bytecode cache (see above)
# 3. Verify database URL is correct
```

### Tasks Not Being Processed

**Problem:** Documents uploaded but processing never starts.

**Causes & Solutions:**

1. **Celery not connected to Redis:**
   ```bash
   # Check Celery log for:
   # "Connected to redis://localhost:6379/0"
   # "celery@hostname ready."
   ```

2. **No workers available:**
   ```bash
   # Check running workers
   ps aux | grep "celery.*worker"
   ```

3. **Task queue full or stuck:**
   ```bash
   # Check Redis queue length
   redis-cli LLEN celery

   # Clear stuck tasks (careful in production!)
   redis-cli DEL celery
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

## Document Generation Issues

### Outline Generates Generic Section Titles

**Problem:** PPTX/DOCX outline has generic titles like "Key Findings and Insights for {topic}" instead of specific section names.

**Causes & Solutions:**

1. **LLM not available:**
   ```bash
   # Check if Ollama is running (if using local LLM)
   curl http://localhost:11434/api/tags

   # Check LLM provider health
   curl http://localhost:8000/health | jq '.llm_health'
   ```

2. **No documents uploaded:**
   The outline generator needs source documents to create relevant sections.
   ```bash
   sqlite3 aidocindexer.db "SELECT COUNT(*) FROM documents WHERE processing_status='completed'"
   ```

3. **Collection filter not matching:**
   If you selected a specific collection but no documents match, the outline will be generic.

### Per-Slide Sources Missing in Notes

**Problem:** The Sources & References slide shows documents, but individual slide notes say "No specific sources for this section."

**Explanation:** This happens because the system uses two different search strategies:

1. **Outline search (broad):** Uses `{document_title}: {document_description}` to find general sources for the whole document. These appear on the Sources & References slide.

2. **Section search (specific):** Uses `{document_title} - {section_title}: {section_description}` to find sources relevant to each specific section. These appear in slide notes.

**Solutions:**

1. **Ensure documents match section topics:**
   Section-specific searches need documents that match the section's subject matter.

2. **Check the backend logs:**
   ```bash
   # Look for "Section sources search completed" messages
   grep "Section sources" /tmp/backend.log
   # Shows: section_title, sources_found, source_names
   ```

3. **Lower relevance threshold (admin setting):**
   In Settings → RAG, reduce the minimum relevance score to allow more results.

### LLM Generation Fails

**Problem:** Error "Cannot connect to host localhost:11434" or similar.

**Solution:**
```bash
# Start Ollama if using local LLM
ollama serve

# Or configure a cloud provider in Settings → LLM Providers
# Add OpenAI or Anthropic as fallback
```

---

## DSPy Optimization Issues

### DSPy Optimization Job Fails

**Problem:** DSPy optimization job fails or hangs.

**Causes & Solutions:**

1. **Insufficient training examples:**
   ```bash
   # Check example count — minimum 10 required
   curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/dspy/training-data/stats
   ```

2. **LLM provider unavailable:**
   - DSPy optimization requires an active LLM provider
   - Check provider health in admin settings

3. **Timeout exceeded:**
   - Default timeout is configurable via `rag.dspy_optimization_timeout`
   - Long-running jobs may need longer timeouts for large datasets

### DSPy Inference Not Working

**Problem:** Compiled DSPy prompts not being used in RAG responses.

**Solution:**
```bash
# Verify DSPy inference is enabled
# Admin Settings → Intelligence → DSPy Inference Enabled = true
# Or check via API:
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/settings/rag.dspy_inference_enabled
```

Ensure at least one optimization job has `deployed` status.

---

## Binary Quantization Issues

### Memory Not Reduced After Enabling

**Problem:** Binary quantization is enabled but memory usage hasn't decreased.

**Solution:**
- Binary quantization only applies to new embeddings — existing vectors must be re-indexed
- Re-process documents to generate binary-quantized embeddings
- Check `rag.binary_quantization_enabled` is `true` in admin settings

### Search Quality Degraded

**Problem:** Search results are less relevant after enabling binary quantization.

**Solution:**
- Binary quantization trades some precision for 32x memory savings
- Use binary quantization as a first-pass filter with re-ranking enabled
- If quality is unacceptable, disable binary quantization and use full-precision vectors

---

## BYOK (Bring Your Own Key) Issues

### BYOK Key Not Working

**Problem:** Provider returns authentication error when using BYOK key.

**Causes & Solutions:**

1. **Invalid API key:**
   - Use the "Test Key" button in BYOK settings to validate
   - Ensure the key has not been revoked by the provider

2. **Key not saved correctly:**
   - BYOK keys are stored in browser localStorage
   - Clear localStorage and re-enter the key
   - Check that the master BYOK toggle is enabled

3. **Provider mismatch:**
   - Ensure the key matches the selected provider
   - OpenAI keys start with `sk-`, Anthropic keys start with `sk-ant-`

### BYOK Mode Not Activating

**Problem:** Requests still use server-side keys despite BYOK being enabled.

**Solution:**
```javascript
// Check browser localStorage
localStorage.getItem('byok-enabled')  // should be "true"
localStorage.getItem('byok-keys')     // should contain provider keys
```

Ensure the master BYOK toggle is ON in the BYOK settings panel.

---

## Content Freshness Issues

### Old Documents Getting Low Scores

**Problem:** Older documents are being penalized too aggressively.

**Solution:**
- Adjust `rag.freshness_decay_days` (default: 180) to a longer period
- Reduce `rag.freshness_penalty_factor` (default: 0.95) closer to 1.0
- Disable content freshness entirely via `rag.content_freshness_enabled = false`

### Recent Documents Not Boosted

**Problem:** Recently updated documents don't appear higher in results.

**Solution:**
- Verify `rag.content_freshness_enabled` is `true`
- Check that document `updated_at` timestamps are being set correctly
- Increase `rag.freshness_boost_factor` (default: 1.05) for stronger boosting

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
