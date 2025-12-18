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

Delete a document.

**Query Parameters:**
- `hard_delete` (bool): Permanently delete (admin only)

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

Send a chat message and get a response. Supports three modes:
- `chat` (default): RAG mode with document search
- `general`: Pure LLM mode without document search
- `agent`: Multi-agent orchestration for complex tasks

**Request:**
```json
{
  "message": "What is the main topic of the Q4 report?",
  "session_id": "optional-session-id",
  "document_ids": ["doc-1", "doc-2"],
  "collection": "optional-collection",
  "search_type": "hybrid",
  "top_k": 5,
  "stream": false,
  "mode": "chat"
}
```

**General Chat Mode Example:**
```json
{
  "message": "What is the capital of France?",
  "mode": "general"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "message_id": "uuid",
  "content": "The capital of France is Paris...",
  "sources": [],
  "is_general_response": true,
  "created_at": "2025-01-01T00:00:00Z"
}
```

### POST /chat/completions/stream

Stream a chat response (Server-Sent Events).

---

## Upload Endpoints

### POST /upload

Upload a single file.

**Request:** multipart/form-data
- `file`: File to upload
- `collection` (optional): Collection name
- `access_tier` (optional): Access tier level (1-100)
- `enable_ocr` (optional, default: true): Enable OCR for scanned documents
- `enable_image_analysis` (optional, default: true): Enable image analysis
- `smart_image_handling` (optional, default: true): Optimize images for faster processing
- `smart_chunking` (optional, default: true): Use semantic chunking
- `detect_duplicates` (optional, default: true): Skip duplicate files
- `processing_mode` (optional): "full", "smart" (default), or "text_only"

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
  "config": {
    "max_depth": 2,
    "same_domain_only": true,
    "extract_links": true
  },
  "collection": "scraped-content",
  "access_tier": 30
}
```

### GET /scraper/jobs

List scrape jobs.

### GET /scraper/jobs/{job_id}

Get job status.

### POST /scraper/jobs/{job_id}/run

Start the scrape job.

### POST /scraper/immediate

Scrape a URL immediately.

**Request:**
```json
{
  "url": "https://example.com",
  "config": {
    "wait_for_js": true,
    "timeout": 30000
  }
}
```

### POST /scraper/query

Scrape and query a URL.

### POST /scraper/extract-links

Extract links from a URL.

---

## Cost Tracking Endpoints

### GET /costs/usage

Get cost usage for a period.

**Query Parameters:**
- `period` (string): day, week, month (default: month)

### GET /costs/history

Get cost history.

### GET /costs/current

Get current period cost.

### GET /costs/dashboard

Get cost dashboard data.

### GET /costs/alerts

List cost alerts.

### POST /costs/alerts

Create a cost alert.

**Request:**
```json
{
  "threshold": 50.00,
  "period": "month"
}
```

### DELETE /costs/alerts/{alert_id}

Delete a cost alert.

### POST /costs/estimate

Estimate cost for a request.

### GET /costs/pricing

Get model pricing information.

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

### GET /auth/users

List all users (admin only).

### PATCH /auth/users/{user_id}

Update user role/tier (admin only).

**Query Parameters:**
- `role` (string): New role
- `access_tier` (int): New access tier
- `is_active` (bool): Active status

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

## Rate Limiting

API requests are limited to:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated endpoints

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time until limit resets
