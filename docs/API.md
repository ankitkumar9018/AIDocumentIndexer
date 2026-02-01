# AIDocumentIndexer API Reference

Complete API documentation for the AIDocumentIndexer backend.

## Base URL

```
http://localhost:8000/api
```

## Authentication

All endpoints (except `/auth/login` and `/auth/register`) require a JWT token in the Authorization header:

```
Authorization: Bearer <token>
```

---

## Authentication Endpoints

### POST /auth/login

Authenticate user with email and password.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbG...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "full_name": "User Name",
    "role": "user",
    "access_tier": 30,
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### POST /auth/register

Register a new user.

**Request:**
```json
{
  "email": "newuser@example.com",
  "password": "securepassword",
  "full_name": "New User"
}
```

### GET /auth/me

Get current user profile.

### POST /auth/change-password

Change user password.

**Request:**
```json
{
  "current_password": "oldpassword",
  "new_password": "newpassword"
}
```

### POST /auth/refresh

Refresh access token.

### POST /auth/logout

Logout current user.

### GET /auth/verify

Verify token validity.

---

## Document Endpoints

### GET /documents

List documents with pagination and filtering.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 20, max: 100)
- `collection` (string): Filter by collection
- `file_type` (string): Filter by file type
- `status` (string): Filter by processing status
- `sort_by` (string): Sort field (default: created_at)
- `sort_order` (string): asc or desc (default: desc)

**Response:**
```json
{
  "documents": [...],
  "total": 156,
  "page": 1,
  "page_size": 20,
  "has_more": true
}
```

### GET /documents/{document_id}

Get a specific document.

### PATCH /documents/{document_id}

Update document metadata.

**Request:**
```json
{
  "name": "New Name",
  "collection": "new-collection",
  "access_tier": 50,
  "tags": ["tag1", "tag2"]
}
```

### DELETE /documents/{document_id}

Delete a document (soft delete by default).

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hard_delete` | bool | false | Permanently delete document and all chunks (admin only) |

**Soft Delete (default):**
- Marks document as deleted but preserves data
- Can be restored from Admin Settings > Database > Deleted Documents
- Chunks remain in vector store

**Hard Delete (admin only):**
- Permanently removes document from database
- Removes all chunks from vector store
- Cannot be undone

### GET /documents/{document_id}/chunks

Get document chunks.

**Query Parameters:**
- `page` (int): Page number
- `page_size` (int): Chunks per page (max: 200)

### POST /documents/search

Search documents using semantic and keyword search.

**Request:**
```json
{
  "query": "search terms",
  "collection": "optional-collection",
  "file_types": ["pdf", "docx"],
  "min_tier": 1,
  "max_tier": 100,
  "limit": 20
}
```

### POST /documents/{document_id}/reprocess

Reprocess a document.

### POST /documents/{document_id}/auto-tag

Automatically generate tags/collection for a document using AI analysis.

The auto-tagger uses the configured LLM provider (defaults to Ollama with the `DEFAULT_CHAT_MODEL`) to analyze the document's content and generate relevant tags. It samples the first few chunks of the document to understand its content and context.

**Request:**
```json
{
  "max_tags": 5
}
```

**Request Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tags` | int | 5 | Maximum number of tags to generate (1-10) |

**Response:**
```json
{
  "document_id": "uuid",
  "tags": ["Project Management", "Software Development", "Guide", "Best Practices"],
  "collection": "Project Management"
}
```

**How it works:**
1. Retrieves the first 3 chunks of the document for content analysis
2. Sends the content sample to the LLM with a tagging prompt
3. LLM generates relevant tags based on topic, industry, document type, and content
4. The first tag is automatically set as the document's collection
5. Tags are merged with any existing user-defined tags (existing tags preserved)

**Note:** Auto-tagging can also be enabled during upload by setting `auto_generate_tags=true` in the upload request. When enabled, tags are automatically generated after document processing completes.

### GET /documents/collections/list

List all collections.

---

## Chat Endpoints

### GET /chat/sessions

List chat sessions.

### POST /chat/sessions

Create a new chat session.

**Request:**
```json
{
  "title": "Optional Title"
}
```

### GET /chat/sessions/{session_id}

Get a specific chat session with messages.

### DELETE /chat/sessions/{session_id}

Delete a chat session.

### POST /chat/completions

Send a chat message and get a response. Supports four modes:
- `chat` (default): RAG mode with document search and citations
- `general`: Pure LLM mode without document search
- `agent`: Multi-agent orchestration for complex tasks
- `vision`: Multimodal image analysis using vision-capable LLMs

**Request:**
```json
{
  "message": "What is the main topic of the Q4 report?",
  "session_id": "optional-session-id",
  "document_ids": ["doc-1", "doc-2"],
  "collection_filter": "optional-collection",
  "collection_filters": ["collection1", "collection2"],
  "search_type": "hybrid",
  "top_k": 10,
  "stream": false,
  "mode": "chat",
  "include_collection_context": true,
  "temp_session_id": "optional-temp-session-id"
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | string | required | The user's question or request |
| `session_id` | string | null | Session ID for conversation history |
| `mode` | string | "chat" | Execution mode: `chat`, `general`, `agent`, or `vision` |
| `stream` | boolean | false | Enable Server-Sent Events streaming |
| `document_ids` | array | null | Limit search to specific documents |
| `collection_filter` | string | null | Limit search to a single collection (backward compatible) |
| `collection_filters` | array | null | Limit search to multiple collections |
| `include_collection_context` | boolean | true | Include collection tags in LLM context |
| `search_type` | string | "hybrid" | Search type: `vector`, `keyword`, `hybrid` |
| `top_k` | int | null | Number of documents to retrieve (3-25). Uses admin setting if not specified. |
| `temp_session_id` | string | null | Temporary document session ID for quick chat |
| `use_graph` | boolean | true | Enable GraphRAG for knowledge graph retrieval |
| `use_agentic` | boolean | false | Enable Agentic RAG for complex multi-step queries |
| `images` | array | null | Image attachments for vision mode (see below) |

**Image Attachment Object (for vision mode):**

| Field | Type | Description |
|-------|------|-------------|
| `data` | string | Base64-encoded image data |
| `url` | string | URL to image (alternative to data) |
| `mime_type` | string | MIME type: `image/jpeg`, `image/png`, `image/webp`, `image/gif` |

**Per-Query Retrieval Override:**

The `top_k` parameter allows you to adjust how many documents are searched on a per-query basis:
- If not specified (`null`), uses the admin-configured default (typically 10)
- Range: 3-25 documents
- Higher values = broader search, finds more potentially relevant documents
- Lower values = faster, more focused results

This is useful when:
- Searching across many collections without filters (use higher top_k: 15-20)
- Looking for specific information with filters applied (use lower top_k: 5-10)

**General Chat Mode Example:**
```json
{
  "message": "What is the capital of France?",
  "mode": "general"
}
```

**Agent Mode Example:**
```json
{
  "message": "Create a summary of all German lessons in my documents",
  "mode": "agent"
}
```

**Vision Mode Example:**
```json
{
  "message": "What does this chart show? Summarize the key findings.",
  "mode": "vision",
  "images": [
    {
      "data": "iVBORw0KGgoAAAANSUhEUgAA...",
      "mime_type": "image/png"
    }
  ]
}
```

Or using a URL:
```json
{
  "message": "Describe this diagram",
  "mode": "vision",
  "images": [
    {
      "url": "https://example.com/diagram.png",
      "mime_type": "image/png"
    }
  ]
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "message_id": "uuid",
  "content": "The capital of France is Paris...",
  "sources": [
    {
      "document_id": "uuid",
      "filename": "document.pdf",
      "page_number": 5,
      "snippet": "Relevant text excerpt...",
      "full_content": "Complete chunk content for detailed viewing...",
      "similarity": 0.92,
      "relevance_score": 0.88,
      "collection": "my-collection",
      "chunk_index": 3
    }
  ],
  "is_general_response": false,
  "confidence_score": 0.85,
  "confidence_level": "high",
  "suggested_questions": [
    "What other cities are important in France?",
    "What is the population of Paris?",
    "What landmarks are in Paris?"
  ],
  "created_at": "2025-01-01T00:00:00Z"
}
```

**Additional Response Fields (Phase 95):**

| Field | Type | Description |
|-------|------|-------------|
| `hallucination_score` | float (0-1) | Hallucination detection score. 0 = fully grounded, 1 = likely hallucinated. Added in Phase 95J. |
| `confidence_score` | float (0-1) | Overall confidence in the response quality, combining source relevance and faithfulness. Added in Phase 95J. |
| `freshness_scores` | array | Per-source freshness scores indicating how current each retrieved document is. Each entry contains `document_id`, `score` (0-1), and `last_updated`. Added in Phase 95K. |
| `conversation_context_used` | boolean | Whether prior conversation context was used to augment the current query. Added in Phase 95L. |

**Confidence Levels:**

| Level | Score Range | Description |
|-------|-------------|-------------|
| `high` | 80%+ | Answer is well-supported by retrieved documents |
| `medium` | 50-80% | Some relevant information found, may be incomplete |
| `low` | <50% | Limited source support, verification recommended |

### POST /chat/completions/stream

Stream a chat response using Server-Sent Events (SSE).

**Request:** Same parameters as `/chat/completions` with `stream: true`

**SSE Event Types:**

| Event Type | Data Structure | Description |
|------------|----------------|-------------|
| `content` | `{ "data": "text chunk" }` | Streaming response text |
| `sources` | `{ "data": [{ "document_id": "...", ... }] }` | Document sources with metadata |
| `confidence` | `{ "score": 0.85, "level": "high" }` | Confidence score for the response |
| `suggested_questions` | `{ "questions": ["...", "..."] }` | Follow-up query suggestions |
| `agent_step` | `{ "step": "Research", "status": "completed" }` | Agent mode step progress |
| `done` | `{ "message_id": "uuid", "content": "full text" }` | Final message with complete content |
| `error` | `{ "message": "error description" }` | Error during processing |

**Example SSE Stream (Agent Mode):**
```
data: {"type": "agent_step", "step": "Research", "status": "in_progress"}

data: {"type": "content", "data": "**Research**\n\nBased on your documents..."}

data: {"type": "sources", "data": [{"document_id": "...", "filename": "german_lesson.pdf", "collection": "German Lessons"}]}

data: {"type": "agent_step", "step": "Generator", "status": "completed"}

data: {"type": "content", "data": "**Final Summary**\n\nHere is the generated content..."}

data: {"type": "done", "message_id": "uuid", "content": "Complete response text"}
```

---

## Upload Endpoints

### POST /upload

Upload a single file.

**Request:** multipart/form-data
- `file`: File to upload
- `collection` (optional): Collection name
- `access_tier` (optional): Access tier level (1-100)
- `is_private` (optional, default: false): Mark document as private (only visible to uploader and admins)
- `enable_ocr` (optional, default: true): Enable OCR for scanned documents
- `enable_image_analysis` (optional, default: true): Enable image analysis
- `smart_image_handling` (optional, default: true): Optimize images for faster processing
- `smart_chunking` (optional, default: true): Use semantic chunking
- `detect_duplicates` (optional, default: true): Skip duplicate files based on content hash. Only flags as duplicate if the original document still exists in the system
- `auto_generate_tags` (optional, default: false): Automatically generate tags using AI analysis after processing. Uses the configured LLM (e.g., Ollama) to analyze document content and generate up to 5 relevant tags. The first tag is also set as the document's collection
- `processing_mode` (optional): "full" (default), "ocr", or "basic"
- `chunking_strategy` (optional, default: "semantic"): "simple", "semantic", or "hierarchical"
- `enable_contextual_headers` (optional, default: true): Prepend document context to each chunk

**Example with Private Document:**
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@sensitive-report.pdf" \
  -F "is_private=true" \
  -F "collection=confidential"
```

### POST /upload/batch

Upload multiple files. Accepts same options as single upload.

### GET /upload/status/{file_id}

Get processing status for a file.

**Response:**
```json
{
  "file_id": "uuid",
  "status": "processing",
  "progress": 65,
  "stage": "embedding",
  "message": "Generating embeddings..."
}
```

### GET /upload/queue

Get the processing queue status.

### POST /upload/cancel/{file_id}

Cancel processing for a file.

### POST /upload/retry/{file_id}

Retry processing for a failed file.

### GET /upload/supported-types

Get list of supported file types.

---

## Document Generation Endpoints

### POST /generation/jobs

Create a new document generation job.

**Request:**
```json
{
  "title": "Document Title",
  "description": "Description of what to generate",
  "source_document_ids": ["doc-1", "doc-2"],
  "output_format": "pdf",
  "tone": "professional",
  "length": "medium"
}
```

### GET /generation/jobs

List generation jobs.

### GET /generation/jobs/{job_id}

Get job status and details.

### POST /generation/jobs/{job_id}/outline

Generate an outline.

### POST /generation/jobs/{job_id}/approve-outline

Approve the generated outline.

### POST /generation/jobs/{job_id}/generate

Start content generation.

### POST /generation/jobs/{job_id}/sections/{section_id}/feedback

Provide feedback on a section.

### POST /generation/jobs/{job_id}/sections/{section_id}/revise

Request revision of a section.

### GET /generation/jobs/{job_id}/download

Download the generated document.

### DELETE /generation/jobs/{job_id}

Cancel a generation job.

### GET /generation/formats

List available output formats.

---

## Collaboration Endpoints

### POST /collaboration/sessions

Create a multi-LLM collaboration session.

**Request:**
```json
{
  "prompt": "Analyze this topic from multiple perspectives",
  "mode": "debate",
  "models": ["gpt-4o", "claude-3-5-sonnet", "llama3.2"],
  "document_ids": ["doc-1"],
  "max_rounds": 3
}
```

### GET /collaboration/sessions

List collaboration sessions.

### GET /collaboration/sessions/{session_id}

Get session details.

### POST /collaboration/sessions/{session_id}/run

Run the collaboration.

### GET /collaboration/sessions/{session_id}/critiques

Get critiques from the session.

### GET /collaboration/modes

List available collaboration modes.

### POST /collaboration/estimate

Estimate cost for a collaboration.

---

## Web Scraper Endpoints

### POST /scraper/jobs

Create a scrape job.

**Request:**
```json
{
  "urls": ["https://example.com"],
  "storage_mode": "permanent",
  "crawl_subpages": false,
  "max_depth": 2,
  "same_domain_only": true,
  "config": {
    "extract_links": true,
    "extract_images": false,
    "extract_metadata": true,
    "timeout": 30,
    "wait_for_js": true
  },
  "collection": "scraped-content",
  "access_tier": 30
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | array | required | URLs to scrape (max 20) |
| `storage_mode` | string | "immediate" | `immediate` or `permanent` |
| `crawl_subpages` | boolean | false | Enable recursive subpage crawling |
| `max_depth` | int | 2 | Maximum crawl depth (1-5) when crawl_subpages is enabled |
| `same_domain_only` | boolean | true | Only crawl pages from the same domain |
| `collection` | string | null | Collection to store scraped content |
| `access_tier` | int | 1 | Access tier for stored content (1-100) |

**Storage Modes:**

| Mode | Description |
|------|-------------|
| `immediate` | Content is scraped and returned but not indexed. Can be indexed later via `/jobs/{job_id}/index` |
| `permanent` | Content is automatically indexed into RAG pipeline (embeddings + vector store + Knowledge Graph) |

### GET /scraper/jobs

List scrape jobs.

### GET /scraper/jobs/{job_id}

Get job status.

### POST /scraper/jobs/{job_id}/run

Run a pending scrape job with optional subpage crawling.

**Request:**
```json
{
  "crawl_subpages": true,
  "max_depth": 3,
  "same_domain_only": true
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crawl_subpages` | boolean | false | Enable recursive subpage crawling |
| `max_depth` | int | 2 | Maximum crawl depth (1-5) |
| `same_domain_only` | boolean | true | Only crawl pages from the same domain |

### POST /scraper/jobs/{job_id}/index

Index a completed job's content into the RAG pipeline. This allows content scraped with `immediate` mode to be permanently indexed later.

**Response:**
```json
{
  "status": "success",
  "documents_indexed": 15,
  "entities_extracted": 42,
  "chunks_processed": 15,
  "error": null
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `success`, `partial`, or `error` |
| `documents_indexed` | int | Number of chunks indexed in vector store |
| `entities_extracted` | int | Number of Knowledge Graph entities extracted |
| `chunks_processed` | int | Total chunks processed |
| `error` | string | Error message if any |

**Use Case:** Users can scrape content with "Quick Scrape" (immediate mode), review it, then decide to permanently save it to the knowledge base by calling this endpoint.

### POST /scraper/index-pages

Index scraped pages directly without a job. Useful for indexing content from quick scrapes.

**Request:**
```json
{
  "pages": [
    {
      "url": "https://example.com/page1",
      "title": "Page Title",
      "content": "Page content in markdown...",
      "metadata": {},
      "word_count": 500,
      "scraped_at": "2025-01-20T10:30:00Z"
    }
  ],
  "source_id": "optional-identifier"
}
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pages` | array | Yes | List of scraped page objects to index |
| `source_id` | string | No | Optional identifier for the source |

**Response:** Same as `/jobs/{job_id}/index`

### POST /scraper/scrape

Scrape a URL immediately (Quick Scrape).

**Request:**
```json
{
  "url": "https://example.com",
  "config": {
    "wait_for_js": true,
    "timeout": 30,
    "crawl_subpages": false,
    "max_depth": 3,
    "same_domain_only": true
  }
}
```

**Response (single page):**
```json
{
  "url": "https://example.com",
  "title": "Example Page",
  "content": "Markdown content...",
  "word_count": 1500,
  "links_count": 25,
  "images_count": 5,
  "scraped_at": "2025-01-20T10:30:00Z",
  "metadata": {}
}
```

**Response (multiple pages when crawl_subpages=true):**
```json
{
  "pages": [...],
  "total_pages": 10,
  "total_word_count": 15000
}
```

### POST /scraper/scrape-and-query

Scrape a URL and query the content with an LLM.

**Request:**
```json
{
  "url": "https://example.com",
  "query": "What is the main topic of this page?",
  "config": {
    "crawl_subpages": false
  }
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "title": "Example Page",
  "content": "Truncated content...",
  "word_count": 1500,
  "scraped_at": "2025-01-20T10:30:00Z",
  "query": "What is the main topic of this page?",
  "answer": "The main topic is...",
  "model": "gpt-4o-mini",
  "processing_time_ms": 2500,
  "context_ready": true
}
```

### POST /scraper/extract-links

Extract links from a URL for discovery.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | URL to extract links from |
| `max_depth` | int | 3 | Maximum depth to follow (1-10) |
| `same_domain_only` | boolean | true | Only extract same-domain links |

**Response:**
```json
{
  "url": "https://example.com",
  "links": ["https://example.com/page1", "https://example.com/page2"],
  "count": 25
}
```

### GET /scraper/jobs/{job_id}/documents

Get scraped content as RAG-ready chunked documents.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1000 | Chunk size in characters (200-4000) |

**Response:**
```json
{
  "job_id": "uuid",
  "documents": [
    {
      "content": "Chunk content...",
      "metadata": {
        "source": "web_scrape",
        "url": "https://example.com",
        "title": "Page Title",
        "chunk_index": 0,
        "total_chunks": 5
      }
    }
  ],
  "total": 15
}
```

### POST /scraper/sitemap-crawl (Phase 96)

Crawl a website using its sitemap.xml for URL discovery. Fetches the site's sitemap.xml, extracts URLs, and crawls them. URLs are prioritized by lastmod date (newest first). Results can be stored permanently in the RAG knowledge base.

**Request:**
```json
{
  "url": "https://example.com",
  "max_pages": 50,
  "storage_mode": "permanent",
  "access_tier": 1
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | Base URL of the site (e.g., https://example.com) |
| `max_pages` | int | 50 | Maximum pages to crawl from sitemap (1-500) |
| `storage_mode` | string | "permanent" | `immediate` or `permanent` |
| `access_tier` | int | 1 | Access tier for stored content (1-100) |

**Response:**
```json
{
  "url": "https://example.com",
  "pages_found": 120,
  "pages_successful": 48,
  "total_words": 75000,
  "storage_mode": "permanent",
  "pages": [
    {
      "url": "https://example.com/page1",
      "title": "Page Title",
      "success": true,
      "word_count": 1500,
      "error": null
    }
  ]
}
```

### GET /scraper/jobs/{job_id}/stream (Phase 96)

Stream real-time progress of a scrape job via Server-Sent Events (SSE). Connect using `EventSource` in the browser.

**Content-Type:** `text/event-stream`

**Path Parameters:**
- `job_id` (string): UUID of the scrape job

**SSE Event Types:**

| Event Type | Data Structure | Description |
|------------|----------------|-------------|
| `status` | `{ "type": "status", "status": "running", "job_id": "uuid" }` | Job status changed |
| `page_complete` | `{ "type": "page_complete", "url": "...", "title": "...", "word_count": 1500, "pages_scraped": 5, "total_pages": 20 }` | A page finished scraping |
| `complete` | `{ "type": "complete", "status": "completed", "pages_scraped": 20, "pages_failed": 2, "total_words": 35000 }` | Job finished (completed or failed) |
| `error` | `{ "type": "error", "message": "Job not found" }` | Error occurred |

**Example SSE Stream:**
```
data: {"type": "status", "status": "running", "job_id": "abc-123"}

data: {"type": "page_complete", "url": "https://example.com/page1", "title": "Page 1", "word_count": 1500, "pages_scraped": 1, "total_pages": 10}

data: {"type": "page_complete", "url": "https://example.com/page2", "title": "Page 2", "word_count": 800, "pages_scraped": 2, "total_pages": 10}

data: {"type": "complete", "status": "completed", "pages_scraped": 10, "pages_failed": 0, "total_words": 12000}
```

### POST /scraper/search-crawl (Phase 96)

Search the web for relevant pages and crawl them. Uses DuckDuckGo to find pages matching the query, then crawls each result to extract content. Optionally stores results in the RAG knowledge base.

**Request:**
```json
{
  "query": "machine learning tutorials",
  "max_results": 5,
  "storage_mode": "immediate",
  "access_tier": 1
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query for finding relevant pages (min 3 characters) |
| `max_results` | int | 5 | Maximum search results to crawl (1-20) |
| `storage_mode` | string | "immediate" | `immediate` or `permanent` |
| `access_tier` | int | 1 | Access tier for stored content (1-100) |

**Response:**
```json
{
  "query": "machine learning tutorials",
  "results_found": 5,
  "results_successful": 4,
  "total_words": 12000,
  "storage_mode": "immediate",
  "results": [
    {
      "url": "https://example.com/ml-guide",
      "title": "ML Guide",
      "success": true,
      "word_count": 3000,
      "error": null,
      "snippet": "An introduction to machine learning concepts..."
    }
  ]
}
```

### GET /scraper/robots-txt (Phase 96)

Parse and return robots.txt rules for a domain. Returns allowed/disallowed paths, crawl delay, and sitemap URLs.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | URL to check robots.txt for |

**Response:**
```json
{
  "url": "https://example.com",
  "allowed_paths": ["/", "/public/"],
  "disallowed_paths": ["/admin/", "/private/", "/api/"],
  "crawl_delay": 1,
  "sitemaps": ["https://example.com/sitemap.xml"]
}
```

### Scheduled Crawls (Phase 96)

Endpoints for managing recurring/scheduled crawls. Scheduled crawls run on cron schedules via Celery Beat. Content is hashed between runs to detect changes and avoid redundant re-indexing.

### POST /scraper/scheduled (Phase 96)

Create a new scheduled/recurring crawl.

**Request:**
```json
{
  "url": "https://example.com/blog",
  "schedule": "0 */6 * * *",
  "crawl_config": {
    "max_pages": 50,
    "max_depth": 3,
    "storage_mode": "permanent"
  },
  "enabled": true
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | URL to crawl on schedule |
| `schedule` | string | required | Cron expression (e.g., `0 */6 * * *` for every 6 hours) |
| `crawl_config` | object | `{"max_pages": 50, "max_depth": 3, "storage_mode": "permanent"}` | Crawl configuration |
| `enabled` | boolean | true | Whether the schedule is active |

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "url": "https://example.com/blog",
  "schedule": "0 */6 * * *",
  "crawl_config": {
    "max_pages": 50,
    "max_depth": 3,
    "storage_mode": "permanent"
  },
  "enabled": true,
  "last_run": null,
  "next_run": "2026-01-26T18:00:00Z",
  "last_content_hash": null,
  "created_at": "2026-01-26T12:00:00Z",
  "updated_at": "2026-01-26T12:00:00Z",
  "created_by": "user-uuid"
}
```

### GET /scraper/scheduled (Phase 96)

List all scheduled crawls. Returns schedules for the current user; admins see all schedules.

**Response:**
```json
{
  "schedules": [
    {
      "id": "uuid",
      "url": "https://example.com/blog",
      "schedule": "0 */6 * * *",
      "crawl_config": {},
      "enabled": true,
      "last_run": "2026-01-26T06:00:00Z",
      "next_run": "2026-01-26T12:00:00Z",
      "last_content_hash": "abc123",
      "created_at": "2026-01-20T10:00:00Z",
      "updated_at": "2026-01-26T06:00:00Z",
      "created_by": "user-uuid"
    }
  ],
  "total": 1
}
```

### GET /scraper/scheduled/{schedule_id} (Phase 96)

Get a specific scheduled crawl by ID. Only accessible by the schedule owner or admins.

**Path Parameters:**
- `schedule_id` (string): UUID of the scheduled crawl

**Response:** Same `ScheduledCrawlResponse` object as `POST /scraper/scheduled`.

### PUT /scraper/scheduled/{schedule_id} (Phase 96)

Update an existing scheduled crawl. Supports partial updates -- only provided fields are changed. If the cron schedule or enabled state changes, the Celery Beat registration is updated accordingly.

**Path Parameters:**
- `schedule_id` (string): UUID of the scheduled crawl

**Request:**
```json
{
  "schedule": "0 0 * * *",
  "enabled": false
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | null | Updated URL |
| `schedule` | string | null | Updated cron expression |
| `crawl_config` | object | null | Updated crawl configuration |
| `enabled` | boolean | null | Enable or disable the schedule |

All fields are optional. At least one field must be provided.

**Response:** Updated `ScheduledCrawlResponse` object.

### DELETE /scraper/scheduled/{schedule_id} (Phase 96)

Delete a scheduled crawl. Removes the schedule and unregisters the corresponding Celery Beat task.

**Path Parameters:**
- `schedule_id` (string): UUID of the scheduled crawl

**Response:**
```json
{
  "message": "Scheduled crawl {schedule_id} deleted successfully"
}
```

### POST /scraper/scheduled/{schedule_id}/run (Phase 96)

Manually trigger a scheduled crawl execution. Runs the crawl immediately regardless of the cron schedule. Computes a content hash and re-indexes only if content has changed since the last execution.

**Path Parameters:**
- `schedule_id` (string): UUID of the scheduled crawl

**Response:**
```json
{
  "status": "completed",
  "pages_crawled": 25,
  "content_changed": true,
  "content_hash": "sha256-hash",
  "indexed": true
}
```

---

## Cost Optimization Endpoints (Phase 68)

Cost monitoring dashboard for tracking and reducing LLM costs by 30-50%. Provides cost-per-query tracking, token usage analytics, cache hit rate monitoring, model usage mix analysis, budget alerts, and GPU utilization metrics.

**Authentication:** All cost optimization endpoints require a valid JWT token.

### GET /costs/analysis

Get comprehensive cost analysis with breakdown by category, model usage, cache performance, and daily cost trends.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_range` | string | "week" | Time range: `hour`, `day`, `week`, or `month` |

**Response:**
```json
{
  "time_range": "week",
  "start_date": "2026-01-19T00:00:00Z",
  "end_date": "2026-01-26T00:00:00Z",
  "total_cost_usd": 12.5432,
  "breakdown": [
    {
      "category": "llm_inference",
      "total_cost_usd": 8.1234,
      "percentage": 64.8,
      "request_count": 450,
      "avg_cost_per_request": 0.018052
    }
  ],
  "model_usage": [
    {
      "model_id": "gpt-4o",
      "provider": "openai",
      "request_count": 200,
      "total_tokens": 500000,
      "input_tokens": 350000,
      "output_tokens": 150000,
      "total_cost_usd": 6.25,
      "avg_latency_ms": 1250.5
    }
  ],
  "cache_stats": [
    {
      "cache_type": "semantic",
      "hit_count": 1234,
      "miss_count": 567,
      "hit_rate": 0.685,
      "savings_usd": 1.234
    }
  ],
  "daily_costs": [
    { "date": "2026-01-20", "cost_usd": 1.8234 }
  ]
}
```

**Cost Categories:**

| Category | Description |
|----------|-------------|
| `llm_inference` | LLM chat/completion costs |
| `embedding` | Embedding generation costs |
| `reranking` | Reranking model costs |
| `ocr` | OCR processing costs |
| `tts` | Text-to-speech costs |
| `storage` | Vector/document storage costs |
| `compute` | General compute costs |

### GET /costs/recommendations

Get cost optimization recommendations based on usage pattern analysis.

**Response:**
```json
[
  {
    "id": "enable-semantic-cache",
    "title": "Enable Semantic Caching",
    "description": "Your cache hit rate is low. Enable semantic caching to match similar queries.",
    "potential_savings_usd": 5.0,
    "priority": "high",
    "action": "Enable 'rag.semantic_cache_enabled' in settings",
    "implemented": false
  }
]
```

**Priority Levels:** `high`, `medium`, `low`

### GET /costs/budget-alerts

List all configured budget alerts with current spend status.

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "Monthly LLM Budget",
    "threshold_usd": 100.0,
    "period": "month",
    "current_spend_usd": 45.23,
    "triggered": false,
    "triggered_at": null
  }
]
```

### POST /costs/budget-alerts

Create a new budget alert.

**Request:**
```json
{
  "name": "Monthly LLM Budget",
  "threshold_usd": 100.0,
  "period": "month",
  "notify_email": "admin@example.com",
  "notify_slack": false
}
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Alert name |
| `threshold_usd` | float | Yes | Budget threshold in USD (must be > 0) |
| `period` | string | No | Time period: `hour`, `day`, `week`, or `month` (default: month) |
| `notify_email` | string | No | Email address for notifications |
| `notify_slack` | boolean | No | Enable Slack notifications (default: false) |

**Response:** `201 Created` with the created `BudgetAlert` object.

### DELETE /costs/budget-alerts/{alert_id}

Delete a budget alert.

**Response:** `204 No Content`

### GET /costs/tokens

Get token usage breakdown with input vs output tokens, tokens per query, and cost per 1K tokens.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_range` | string | "day" | Time range: `hour`, `day`, `week`, or `month` |

**Response:**
```json
{
  "time_range": "day",
  "start_date": "2026-01-25T00:00:00Z",
  "end_date": "2026-01-26T00:00:00Z",
  "total_tokens": 125000,
  "input_tokens": 87500,
  "output_tokens": 37500,
  "total_cost_usd": 1.5432,
  "query_count": 85,
  "avg_tokens_per_query": 1470.6,
  "cost_per_1k_tokens": 0.012346
}
```

### GET /costs/gpu

Get GPU utilization metrics including memory usage and inference statistics.

**Response:**
```json
{
  "available": true,
  "device_count": 1,
  "cuda_version": "12.1",
  "devices": [
    {
      "id": 0,
      "name": "NVIDIA RTX 4090",
      "total_memory_gb": 24.0,
      "allocated_memory_gb": 8.5,
      "reserved_memory_gb": 10.0,
      "utilization_percent": 35.4
    }
  ]
}
```

If no GPU is available, returns `{ "available": false, "device_count": 0, "devices": [] }`.

### POST /costs/record

Record a cost event. Used internally by services to track costs.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `category` | string | Yes | Cost category (see categories above) |
| `cost_usd` | float | Yes | Cost in USD |
| `tokens` | int | No | Token count (default: 0) |
| `model_id` | string | No | Model identifier |
| `latency_ms` | float | No | Request latency in milliseconds (default: 0) |

**Response:**
```json
{ "status": "recorded" }
```

### POST /costs/cache-event

Record a cache hit or miss event. Used internally by cache services.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cache_type` | string | Yes | Cache type identifier (e.g., "semantic", "embedding") |
| `hit` | boolean | Yes | Whether this was a cache hit |

**Response:**
```json
{ "status": "recorded" }
```

### GET /costs/summary

Get a quick cost summary for dashboard display.

**Response:**
```json
{
  "today_cost_usd": 2.3456,
  "month_cost_usd": 45.2345,
  "total_queries_today": 85,
  "total_queries_month": 2150,
  "cache_hit_rate": 0.685,
  "estimated_cache_savings_usd": 1.234,
  "top_model": {
    "model_id": "gpt-4o",
    "cost_usd": 6.25
  },
  "active_alerts": 0
}
```

---

## Agent Endpoints

### GET /agent/mode

Get current execution mode and preferences.

**Response:**
```json
{
  "mode": "agent",
  "agent_mode_enabled": true,
  "auto_detect_complexity": true
}
```

### POST /agent/mode/toggle

Toggle agent mode on/off.

### GET /agent/preferences

Get user's agent preferences.

**Response:**
```json
{
  "default_mode": "chat",
  "agent_mode_enabled": true,
  "auto_detect_complexity": true,
  "show_cost_estimation": true,
  "require_approval_above_usd": 1.0,
  "general_chat_enabled": true,
  "fallback_to_general": true
}
```

### PATCH /agent/preferences

Update agent preferences.

**Request:**
```json
{
  "default_mode": "agent",
  "agent_mode_enabled": true,
  "auto_detect_complexity": true,
  "show_cost_estimation": true,
  "require_approval_above_usd": 1.0,
  "general_chat_enabled": true,
  "fallback_to_general": true
}
```

### POST /agent/execute

Execute a request through the multi-agent system.

**Request:**
```json
{
  "message": "Generate a comprehensive market analysis",
  "session_id": "optional-session-id",
  "context": {}
}
```

---

## Admin Endpoints

### GET /admin/users

List all users (admin only).

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 20)

**Response:**
```json
{
  "users": [
    {
      "id": "uuid",
      "email": "user@example.com",
      "name": "User Name",
      "is_active": true,
      "access_tier_id": "uuid",
      "access_tier_name": "Staff",
      "access_tier_level": 30,
      "use_folder_permissions_only": false,
      "created_at": "2025-01-01T00:00:00Z",
      "last_login_at": "2025-01-08T10:30:00Z"
    }
  ],
  "total": 50,
  "page": 1,
  "page_size": 20
}
```

### POST /admin/users

Create a new user (admin only).

**Request:**
```json
{
  "email": "newuser@example.com",
  "password": "securepassword",
  "full_name": "New User",
  "access_tier_id": "uuid",
  "use_folder_permissions_only": false,
  "initial_folder_permissions": [
    {
      "folder_id": "uuid",
      "permission_level": "view",
      "inherit_to_children": true
    }
  ]
}
```

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `email` | string | Yes | User email address |
| `password` | string | Yes | Password (min 8 characters) |
| `full_name` | string | No | User's full name |
| `access_tier_id` | uuid | Yes | Access tier ID |
| `use_folder_permissions_only` | bool | No | Restrict to folder-only access (default: false) |
| `initial_folder_permissions` | array | No | Initial folder access grants |

**Folder Permission Object:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `folder_id` | uuid | Yes | Folder to grant access to |
| `permission_level` | string | No | `view`, `edit`, or `manage` (default: view) |
| `inherit_to_children` | bool | No | Apply to subfolders (default: true) |

### PATCH /admin/users/{user_id}

Update user settings (admin only).

**Request:**
```json
{
  "access_tier_id": "uuid",
  "is_active": true,
  "use_folder_permissions_only": false
}
```

**Request Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `access_tier_id` | uuid | New access tier |
| `is_active` | bool | Active status |
| `use_folder_permissions_only` | bool | Restrict to folder-only access |

**Note on `use_folder_permissions_only`:**
When enabled, the user can ONLY access folders they have been explicitly granted access to via folder permissions. Tier-based access is bypassed entirely. This is useful for:
- External contractors needing access to specific project folders
- Temporary collaborators with limited scope
- Compliance scenarios requiring strict access control

### GET /admin/ocr/settings

Get OCR configuration and model information (admin only).

**Response:**
```json
{
  "settings": {
    "ocr.provider": "paddleocr",
    "ocr.paddle.variant": "server",
    "ocr.paddle.languages": ["en", "de"],
    "ocr.paddle.model_dir": "./data/paddle_models",
    "ocr.paddle.auto_download": true,
    "ocr.tesseract.fallback_enabled": true
  },
  "models": {
    "downloaded": [
      {
        "name": "inference",
        "type": ".pdiparams",
        "size": "83.9 MB",
        "path": "official_models/PP-OCRv5_server_det/inference.pdiparams"
      }
    ],
    "total_size": "118.4 MB",
    "model_dir": "data/paddle_models",
    "status": "installed"
  }
}
```

### PATCH /admin/ocr/settings

Update OCR configuration (admin only).

**Request Body:**
```json
{
  "ocr.provider": "paddleocr",
  "ocr.paddle.variant": "mobile",
  "ocr.paddle.languages": ["en", "de", "fr"]
}
```

**Response:**
```json
{
  "status": "updated",
  "settings": { ... },
  "download_triggered": true
}
```

### POST /admin/ocr/models/download

Download PaddleOCR models for specified languages (admin only).

**Request Body:**
```json
{
  "languages": ["en", "de"],
  "variant": "server"
}
```

**Response:**
```json
{
  "status": "success",
  "downloaded": ["en", "de"],
  "failed": [],
  "model_info": {
    "downloaded": [...],
    "total_size": "118.4 MB",
    "status": "installed"
  }
}
```

### GET /admin/ocr/models/info

Get information about downloaded PaddleOCR models (admin only).

**Response:**
```json
{
  "downloaded": [...],
  "total_size": "118.4 MB",
  "model_dir": "data/paddle_models",
  "status": "installed"
}
```

**Note:** For complete OCR configuration details, see [OCR_CONFIGURATION.md](./OCR_CONFIGURATION.md).

### GET /admin/llm/ollama-models

List all locally installed Ollama models, categorized by type (admin only).

**Query Parameters:**
- `base_url` (string): Ollama server URL (default: `http://localhost:11434`)

**Response:**
```json
{
  "success": true,
  "chat_models": [
    {
      "name": "llama3.2:latest",
      "parameter_size": "3B",
      "family": "llama",
      "size": 2147483648
    },
    {
      "name": "qwen2.5vl:7b",
      "parameter_size": "7B",
      "family": "qwen2",
      "size": 4831838208
    }
  ],
  "embedding_models": [
    {
      "name": "nomic-embed-text:latest",
      "parameter_size": "274M",
      "size": 137000000
    }
  ],
  "vision_models": [
    {
      "name": "qwen2.5vl:7b",
      "parameter_size": "7B",
      "family": "qwen2",
      "size": 4831838208
    }
  ]
}
```

### POST /admin/llm/ollama-models/pull

Download (pull) an Ollama model from the library (admin only).

**Request Body:**
```json
{
  "model_name": "qwen2.5vl",
  "base_url": "http://localhost:11434"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model 'qwen2.5vl' pulled successfully",
  "model": "qwen2.5vl"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Cannot connect to Ollama. Is it running?"
}
```

**Note:** Model downloads can take several minutes for large models (7B+). The endpoint has a 30-minute timeout.

### DELETE /admin/llm/ollama-models/{model_name}

Delete a locally installed Ollama model (admin only).

**Path Parameters:**
- `model_name` (string): Name of the model to delete (e.g., `llama3.2:latest`)

**Query Parameters:**
- `base_url` (string): Ollama server URL (default: `http://localhost:11434`)

**Response:**
```json
{
  "success": true,
  "message": "Model 'llama3.2:latest' deleted successfully"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Model 'unknown-model' not found"
}
```

---

## Folder Endpoints

### GET /folders

List folders with optional parent filter.

**Query Parameters:**
- `parent_id` (string): Filter by parent folder ID (null for root)
- `include_counts` (bool): Include document counts (default: true)

**Response:**
```json
{
  "folders": [
    {
      "id": "uuid",
      "name": "Marketing",
      "path": "/Marketing/",
      "parent_folder_id": null,
      "depth": 0,
      "access_tier_level": 30,
      "document_count": 15,
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 5
}
```

### POST /folders

Create a new folder.

**Request:**
```json
{
  "name": "Q1 Reports",
  "parent_folder_id": "optional-parent-uuid",
  "access_tier_id": "tier-uuid",
  "description": "First quarter financial reports",
  "color": "#3D5A80"
}
```

### GET /folders/tree

Get the full folder hierarchy tree.

**Response:**
```json
{
  "tree": [
    {
      "id": "uuid",
      "name": "Marketing",
      "path": "/Marketing/",
      "children": [
        {
          "id": "uuid",
          "name": "Campaigns",
          "path": "/Marketing/Campaigns/",
          "children": []
        }
      ]
    }
  ]
}
```

### GET /folders/{folder_id}

Get a specific folder.

### PATCH /folders/{folder_id}

Update folder properties.

**Request:**
```json
{
  "name": "New Name",
  "description": "Updated description",
  "color": "#1E3A5F"
}
```

### DELETE /folders/{folder_id}

Delete a folder.

**Query Parameters:**
- `recursive` (bool): Delete folder and all contents (default: false)

### POST /folders/{folder_id}/move

Move a folder to a new parent.

**Request:**
```json
{
  "new_parent_id": "target-folder-uuid"
}
```

### GET /folders/{folder_id}/documents

List documents in a folder.

**Query Parameters:**
- `include_subfolders` (bool): Include documents from subfolders (default: false)
- `page` (int): Page number
- `page_size` (int): Items per page

---

## Folder Permission Endpoints

Per-user folder permissions allow granting specific users access to specific folders, independent of their tier-based access.

### GET /folders/{folder_id}/permissions

Get all users with explicit access to a folder.

**Response:**
```json
[
  {
    "id": "uuid",
    "folder_id": "uuid",
    "folder_name": "Marketing",
    "folder_path": "/Marketing/",
    "user_id": "uuid",
    "user_email": "user@example.com",
    "user_name": "User Name",
    "permission_level": "view",
    "inherit_to_children": true,
    "granted_by_id": "uuid",
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

### POST /folders/{folder_id}/permissions

Grant folder access to a user.

**Request:**
```json
{
  "user_id": "uuid",
  "permission_level": "view",
  "inherit_to_children": true
}
```

**Permission Levels:**
| Level | Description |
|-------|-------------|
| `view` | Can see folder and read documents |
| `edit` | Can upload and modify documents |
| `manage` | Can grant permissions to others |

**Response:**
```json
{
  "id": "uuid",
  "folder_id": "uuid",
  "user_id": "uuid",
  "permission_level": "view",
  "inherit_to_children": true,
  "created_at": "2025-01-01T00:00:00Z"
}
```

### DELETE /folders/{folder_id}/permissions/{user_id}

Revoke folder access from a user.

**Response:**
```json
{
  "message": "Permission revoked successfully"
}
```

### GET /admin/users/{user_id}/folder-permissions

Get all folders a user has explicit access to (admin only).

**Response:**
```json
[
  {
    "id": "uuid",
    "folder_id": "uuid",
    "folder_name": "Marketing",
    "folder_path": "/Marketing/",
    "permission_level": "edit",
    "inherit_to_children": true,
    "created_at": "2025-01-01T00:00:00Z"
  }
]
```

---

## User Preferences Endpoints

### GET /preferences

Get current user's preferences.

**Response:**
```json
{
  "theme": "system",
  "documents_view_mode": "grid",
  "documents_sort_by": "created_at",
  "documents_sort_order": "desc",
  "documents_page_size": 20,
  "default_collection": null,
  "default_folder_id": null,
  "search_include_content": true,
  "search_results_per_page": 10,
  "chat_show_sources": true,
  "chat_expand_sources": false,
  "sidebar_collapsed": false,
  "recent_documents": ["doc-uuid-1", "doc-uuid-2"],
  "recent_searches": ["search term 1", "search term 2"]
}
```

### PATCH /preferences

Update user preferences.

**Request:**
```json
{
  "theme": "dark",
  "documents_view_mode": "list",
  "documents_page_size": 50
}
```

**Valid Values:**
- `theme`: "light", "dark", "system"
- `documents_view_mode`: "grid", "list", "table"
- `documents_sort_by`: "created_at", "name", "file_size", "updated_at"
- `documents_sort_order`: "asc", "desc"
- `documents_page_size`: 5-100

### POST /preferences/recent

Add a recent item (document or search).

**Request:**
```json
{
  "item_type": "document",
  "item": "document-uuid"
}
```

### DELETE /preferences/recent/{item_type}

Clear recent documents or searches.

**Path Parameters:**
- `item_type`: "documents" or "searches"

### POST /preferences/reset

Reset all preferences to defaults.

---

## Saved Searches Endpoints

### GET /preferences/searches

List all saved searches.

**Response:**
```json
{
  "searches": [
    {
      "name": "Marketing PDFs",
      "query": "marketing campaign",
      "collection": "Marketing",
      "folder_id": null,
      "include_subfolders": true,
      "file_types": ["pdf", "pptx"],
      "search_mode": "hybrid",
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "count": 3
}
```

### POST /preferences/searches

Save a search configuration.

**Request:**
```json
{
  "name": "German Lessons",
  "query": "german vocabulary",
  "collection": "Learning",
  "folder_id": "optional-folder-uuid",
  "include_subfolders": true,
  "date_from": "2024-01-01",
  "date_to": "2024-12-31",
  "file_types": ["pdf", "docx"],
  "search_mode": "hybrid"
}
```

**Valid Values:**
- `search_mode`: "hybrid", "vector", "keyword"
- Maximum 20 saved searches per user

### GET /preferences/searches/{name}

Get a specific saved search by name.

### DELETE /preferences/searches/{name}

Delete a saved search.

---

## Search Operators

The search system supports advanced operators for keyword search:

| Operator | Syntax | Example | Description |
|----------|--------|---------|-------------|
| AND | `term1 AND term2` | `marketing AND strategy` | Both terms required |
| OR | `term1 OR term2` | `budget OR finance` | Either term matches |
| NOT | `NOT term` | `NOT draft` | Exclude term |
| Phrase | `"exact phrase"` | `"quarterly report"` | Match exact phrase |
| Grouping | `(term1 OR term2)` | `(Q1 OR Q2) AND report` | Group expressions |

**Examples:**
```
marketing AND (strategy OR plan)
"quarterly report" NOT draft
budget OR finance OR accounting
German AND vocabulary NOT test
```

---

## Health Check

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "database": "connected",
  "redis": "connected"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

### Common Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

---

## RAG Evaluation Endpoints

### POST /evaluation/evaluate

Evaluate a single RAG response using RAGAS metrics.

**Request:**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a branch of AI...",
  "contexts": ["Retrieved document content 1...", "Retrieved document content 2..."],
  "ground_truth": "Optional expected answer for recall calculation"
}
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a branch of AI...",
  "contexts": ["..."],
  "metrics": {
    "context_relevance": 0.85,
    "faithfulness": 0.92,
    "answer_relevance": 0.88,
    "context_recall": 0.80,
    "overall_score": 0.88,
    "quality_level": "excellent",
    "evaluation_time_ms": 1250.5
  },
  "ground_truth": null,
  "issues": [],
  "suggestions": [],
  "timestamp": "2026-01-24T10:30:00Z"
}
```

### POST /evaluation/evaluate-query

Evaluate a RAG query by running it through the RAG service first.

**Request:**
```json
{
  "query": "What is machine learning?",
  "session_id": "optional-session-id",
  "ground_truth": "Optional expected answer"
}
```

**Response:** Same as `/evaluation/evaluate`

### POST /evaluation/benchmark

Run a benchmark suite with multiple test cases.

**Request:**
```json
{
  "test_cases": [
    {
      "query": "Query 1...",
      "ground_truth": "Expected answer 1...",
      "contexts": ["Optional pre-defined contexts..."]
    },
    {
      "query": "Query 2...",
      "ground_truth": "Expected answer 2..."
    }
  ],
  "collection_filter": "optional-collection-name",
  "use_rag_service": true
}
```

**Response:**
```json
{
  "test_count": 2,
  "passing_rate": 0.85,
  "duration_ms": 5230.5,
  "aggregate_metrics": {
    "context_relevance": 0.82,
    "context_relevance_std": 0.05,
    "faithfulness": 0.89,
    "faithfulness_std": 0.03,
    "answer_relevance": 0.85,
    "answer_relevance_std": 0.04,
    "overall_score": 0.85,
    "overall_score_std": 0.04
  },
  "timestamp": "2026-01-24T10:30:00Z"
}
```

### GET /evaluation/metrics/recent

Get recent evaluation metrics summary.

**Query Parameters:**
- `hours` (int): Number of hours to look back (default: 24)

**Response:**
```json
{
  "count": 150,
  "avg_overall": 0.84,
  "avg_faithfulness": 0.87,
  "avg_relevance": 0.82,
  "period_hours": 24
}
```

### GET /evaluation/metrics/trend

Get evaluation metrics trend over time.

**Query Parameters:**
- `metric` (string): Metric to track (default: "overall_score")
- `periods` (int): Number of time periods (default: 7)
- `period_hours` (int): Hours per period (default: 24)

**Response:**
```json
{
  "metric": "overall_score",
  "periods": 7,
  "period_hours": 24,
  "trend": [0.82, 0.83, 0.85, 0.84, 0.86, 0.85, 0.87],
  "average": 0.846,
  "direction": "improving"
}
```

### GET /evaluation/health

Check evaluation service health.

**Response:**
```json
{
  "status": "healthy",
  "tracker_size": 150,
  "llm_available": true,
  "embedding_service_available": true
}
```

---

## Agentic RAG Endpoints (Phase 72+)

### POST /agentic/query

Execute an agentic RAG query with multi-step reasoning (DRAGIN/FLARE dynamic retrieval).

**Request:**
```json
{
  "query": "Compare the financial performance across all quarterly reports",
  "session_id": "optional-session-id",
  "max_iterations": 5,
  "collection_filter": "financial-reports"
}
```

**Response:**
```json
{
  "answer": "Based on the quarterly reports...",
  "reasoning_steps": [...],
  "sub_queries": [...],
  "sources": [...],
  "iterations_used": 3,
  "token_budget_used": 4500
}
```

### GET /agentic/status

Get agentic RAG service status and configuration.

---

## Cache Management Endpoints (Phase 75)

### GET /cache/stats

Get cache statistics across all cache tiers (semantic, generative, embedding, query).

**Response:**
```json
{
  "semantic_cache": { "hits": 1234, "misses": 567, "hit_rate": 0.685, "size": 5000 },
  "generative_cache": { "hits": 890, "misses": 234, "hit_rate": 0.792, "size": 3000 },
  "embedding_cache": { "hits": 45000, "misses": 2000, "hit_rate": 0.957 },
  "redis_connected": true
}
```

### POST /cache/invalidate

Invalidate cache entries by type and optional key pattern.

**Request:**
```json
{
  "cache_type": "semantic",
  "pattern": "collection:finance*",
  "reason": "documents updated"
}
```

### POST /cache/clear

Clear all caches (admin only). Returns count of cleared entries.

---

## Reranking Endpoints (Phase 74)

### POST /reranking/rerank

Apply tiered reranking to a set of search results.

**Request:**
```json
{
  "query": "What is the revenue?",
  "results": [
    { "content": "...", "score": 0.85 },
    { "content": "...", "score": 0.72 }
  ],
  "stages": ["bm25", "cross_encoder", "colbert", "llm"]
}
```

### GET /reranking/config

Get current reranking pipeline configuration (stages, models, thresholds).

---

## Compression Endpoints (Phase 66+)

### POST /compression/compress

Compress context using available compression methods.

**Request:**
```json
{
  "query": "summary query",
  "context": "long context text...",
  "method": "attention_rag",
  "target_ratio": 0.3
}
```

**Methods:** `attention_rag`, `llmlingua`, `ttt`, `oscar`, `rcc`

### GET /compression/stats

Get compression statistics and method performance.

---

## Vision Processing Endpoints (Phase 76)

### POST /vision/process

Process an image through the vision pipeline (OCR, captioning, table extraction).

**Request:** `multipart/form-data` with `file` field.

### POST /vision/caption

Generate a caption for an uploaded image.

### GET /vision/status

Get vision processing service status and available models.

---

## RAG Security Endpoints (Phase 84-85)

### POST /security/scan

Scan a query for prompt injection and other security threats.

**Request:**
```json
{
  "query": "user query text",
  "scan_types": ["prompt_injection", "pii", "jailbreak"]
}
```

**Response:**
```json
{
  "safe": true,
  "threats_detected": [],
  "confidence": 0.95,
  "scan_time_ms": 12
}
```

### GET /security/config

Get current security configuration (threat detection thresholds, enabled checks).

---

## Experiments & Feature Flags Endpoints (Phase 82)

### GET /experiments

List all available experiments and their status.

**Response:**
```json
{
  "experiments": [
    { "name": "attention_rag", "enabled": true, "description": "6.3x context compression" },
    { "name": "graph_o1", "enabled": false, "description": "Beam search GraphRAG reasoning" }
  ]
}
```

### PUT /experiments/{experiment_name}

Toggle an experiment on/off (admin only).

**Request:**
```json
{
  "enabled": true
}
```

---

## Diagnostics Endpoints (Phase 88)

### GET /diagnostics/health

Comprehensive health check for all external services.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "postgresql": { "connected": true, "latency_ms": 5 },
    "redis": { "connected": true, "latency_ms": 2 },
    "ollama": { "connected": true, "models": 3 },
    "ray": { "connected": false, "reason": "not configured" },
    "chromadb": { "connected": true, "collections": 5 }
  },
  "uptime_seconds": 86400
}
```

### GET /diagnostics/circuit-breakers

Get circuit breaker status for all providers.

### GET /diagnostics/rate-limits

Get current rate limit status for all LLM providers.

---

## Late Chunking Endpoints (Phase 66)

### POST /late-chunking/process

Process text using late chunking (context-preserving chunking with +15-25% accuracy).

**Request:**
```json
{
  "text": "long document text...",
  "chunk_size": 512,
  "model": "jina-embeddings-v3"
}
```

---

## DSPy Prompt Optimization Endpoints (Phase 93)

Admin API for DSPy-based prompt optimization. Supports automated optimization of RAG prompts using BootstrapFewShot and MIPROv2 optimizers, training example management, and deployment of optimized prompts via A/B testing.

**Authentication:** All DSPy endpoints require admin privileges (JWT token with admin role).

**Base Path:** `/admin/dspy`

### POST /admin/dspy/optimize

Trigger DSPy prompt optimization for a specific signature. Creates and runs an optimization job that collects training examples, runs the selected optimizer, evaluates improvement, and optionally exports to the prompt version manager.

**Request:**
```json
{
  "signature": "rag_answer",
  "optimizer": "bootstrap_few_shot",
  "max_examples": 50,
  "agent_id": "optional-agent-id"
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | string | required | Signature to optimize (see valid values below) |
| `optimizer` | string | "bootstrap_few_shot" | Optimizer type: `bootstrap_few_shot` or `miprov2` |
| `max_examples` | int | 50 | Maximum training examples to use (5-1000) |
| `agent_id` | string | null | Agent ID for prompt version export |

**Valid Signatures:**
- `rag_answer` -- Main RAG answer generation
- `query_expansion` -- Query expansion for improved retrieval
- `query_decomposition` -- Complex query decomposition into sub-queries
- `react_reasoning` -- ReAct-style reasoning steps
- `answer_synthesis` -- Final answer synthesis from multiple sources

**Response:**
```json
{
  "id": "uuid",
  "signature_name": "rag_answer",
  "optimizer_type": "bootstrap_few_shot",
  "status": "completed",
  "num_train_examples": 40,
  "num_dev_examples": 10,
  "baseline_score": 0.72,
  "optimized_score": 0.85,
  "improvement_pct": 18.1,
  "error_message": null,
  "prompt_version_id": "uuid-or-null",
  "created_at": "2026-01-26T10:30:00Z"
}
```

**Job Status Values:** `running`, `completed`, `failed`, `deployed`

### GET /admin/dspy/status/{job_id}

Get the status of a DSPy optimization job.

**Path Parameters:**
- `job_id` (string): UUID of the optimization job

**Response:** Same `JobResponse` schema as `POST /admin/dspy/optimize`.

**Error:** `404` if job not found.

### GET /admin/dspy/jobs

List DSPy optimization jobs with optional filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | string | null | Filter by signature name |
| `limit` | int | 20 | Maximum number of jobs to return |

**Response:** Array of `JobResponse` objects, ordered by creation date (newest first).

### GET /admin/dspy/examples

List training examples used for optimization.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | string | null | Filter by signature name |
| `source` | string | null | Filter by source (e.g., `manual`, `chat_feedback`, `trajectory`) |
| `active_only` | boolean | true | Only return active examples |
| `limit` | int | 50 | Maximum number of examples to return |

**Response:**
```json
[
  {
    "id": "uuid",
    "signature_name": "rag_answer",
    "inputs": { "question": "What is AI?", "context": "..." },
    "outputs": { "answer": "AI is..." },
    "source": "chat_feedback",
    "quality_score": 0.95,
    "is_active": true,
    "created_at": "2026-01-25T14:00:00Z"
  }
]
```

### POST /admin/dspy/examples

Add a manual training example for optimization.

**Request:**
```json
{
  "signature_name": "rag_answer",
  "inputs": { "question": "What is machine learning?", "context": "ML is a subset of AI..." },
  "outputs": { "answer": "Machine learning is a branch of artificial intelligence..." },
  "quality_score": 1.0
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature_name` | string | required | Target signature name |
| `inputs` | object | required | Input fields for the signature |
| `outputs` | object | required | Expected output fields |
| `quality_score` | float | 1.0 | Quality score (0.0-1.0) |

**Response:** The created `ExampleResponse` object.

### DELETE /admin/dspy/examples/{example_id}

Deactivate a training example (soft delete).

**Path Parameters:**
- `example_id` (string): UUID of the example to deactivate

**Response:**
```json
{
  "status": "deactivated",
  "id": "example-uuid"
}
```

**Error:** `404` if example not found.

### GET /admin/dspy/example-counts

Get training example counts grouped by source.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | string | null | Filter by signature name |

**Response:**
```json
{
  "chat_feedback": 25,
  "trajectory": 15,
  "manual": 10,
  "total": 50
}
```

### POST /admin/dspy/deploy/{job_id}

Deploy a completed optimization result by creating a new prompt version for A/B testing via the PromptVersionManager.

**Path Parameters:**
- `job_id` (string): UUID of the completed optimization job

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_id` | string | Yes | Agent ID to deploy the optimized prompt to |

**Response:**
```json
{
  "status": "deployed",
  "job_id": "uuid",
  "prompt_version_id": "uuid",
  "improvement_pct": 18.1
}
```

**Errors:**
- `404` -- Job not found
- `400` -- Job status is not `completed` or no compiled state available
- `500` -- Deployment failed

---

## Rate Limiting

API requests are limited to:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated endpoints
- 10 login attempts per minute per IP (Phase 85)
- 5 registration attempts per hour per IP (Phase 85)

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time until limit resets
