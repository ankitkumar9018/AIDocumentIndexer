# AIDocumentIndexer - Security Model

This document describes the security architecture, threat model, and hardening measures implemented in AIDocumentIndexer.

---

## Table of Contents

1. [Authentication & Authorization](#authentication--authorization)
2. [Input Validation](#input-validation)
3. [RAG Security](#rag-security)
4. [Sandbox Execution](#sandbox-execution)
5. [Rate Limiting](#rate-limiting)
6. [Data Protection](#data-protection)
7. [LLM Provider Security](#llm-provider-security)
8. [Configuration Security](#configuration-security)
9. [Audit Logging](#audit-logging)
10. [Security Hardening Checklist](#security-hardening-checklist)

---

## Authentication & Authorization

### JWT Authentication

All API endpoints (except `/auth/login` and `/auth/register`) require a valid JWT token.

- **Algorithm:** HS256 (configurable via `JWT_ALGORITHM`)
- **Expiration:** 24 hours (configurable via `JWT_EXPIRATION_HOURS`)
- **Secret Key:** Set via `SECRET_KEY` environment variable

**Phase 85:** The application rejects the default `SECRET_KEY` value (`change-me-in-production`) in production and staging environments at startup.

### Access Tier System

Users are assigned access tiers controlling document visibility:

| Tier | Role | Access |
|------|------|--------|
| 10 | Viewer | Read public documents only |
| 30 | User | Read all accessible documents |
| 50 | Editor | Read/write team documents |
| 90 | Manager | Manage team documents |
| 100 | Admin | Full access including admin panel |

Documents have an `access_tier` field. Users can only access documents with a tier equal to or below their own.

### Organization-Based Access

Multi-tenant isolation ensures users only see documents belonging to their organization. Superadmins can access all organizations.

---

## Input Validation

### Query Length Limits (Phase 85)

| Endpoint | Field | Max Length |
|----------|-------|-----------|
| Chat | `message` | 100,000 characters |
| Chat | `content` (per message) | 500,000 characters |
| Chat | `collection_filter` | 100 characters |
| Documents | `name` | 500 characters |
| Documents | `collection` | 100 characters (alphanumeric + `_-. `) |

### Collection Name Validation (Phase 85)

Collection names are validated with regex: `^[a-zA-Z0-9_\-\s\.]+$`

This prevents:
- Path traversal attacks via collection names
- SQL injection in collection filters
- Directory traversal in file storage paths

---

## RAG Security

### Prompt Injection Detection

The `rag_security.py` service implements OWASP LLM Top 10 protections:

1. **Prompt injection detection:** Scans user queries for injection patterns (e.g., "ignore previous instructions", "system prompt override")
2. **PII detection:** Identifies and optionally masks personal identifiable information in queries and responses
3. **Jailbreak detection:** Detects attempts to bypass model safety filters
4. **Content safety scoring:** Rates queries and responses for safety

### Security Pipeline

Every RAG query passes through:
1. Input sanitization (strip control characters, normalize Unicode)
2. Prompt injection scan (configurable threshold)
3. PII detection (optional masking)
4. Standard RAG pipeline execution
5. Output validation (hallucination check, safety score)

### Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `security.prompt_injection_enabled` | Enable injection detection | `true` |
| `security.pii_detection_enabled` | Enable PII detection | `true` |
| `security.injection_threshold` | Detection confidence threshold (0-1) | `0.7` |

---

## Sandbox Execution

### Code Execution Security (Phase 84)

Three services execute user/LLM-generated code in sandboxed environments:

#### 1. RLM Sandbox (`rlm_sandbox.py`)

Used for Recursive Language Model code execution:
- **Safe wrappers:** `SafeRegex` and `SafeJson` classes replace raw module access
- **Blocked:** `__import__`, `eval`, `exec`, `open`, `os`, `sys`, `subprocess`
- **Attribute blocking:** `__getattr__` raises `AttributeError` for any non-whitelisted attribute access
- **Subprocess isolation:** Code runs in a separate process with resource limits

#### 2. Recursive LM (`recursive_lm.py`)

- Uses same `SafeRegex`/`SafeJson` wrappers from `rlm_sandbox`
- **Execution timeout:** Code execution is wrapped in `ThreadPoolExecutor` with configurable timeout (default: 120 seconds)
- **Safe globals:** Only whitelisted builtins (`len`, `str`, `int`, `float`, `list`, `dict`, `range`, `enumerate`, `zip`, `sorted`, `min`, `max`, `abs`, `round`, `sum`, `any`, `all`, `map`, `filter`, `isinstance`, `type`, `print`)

#### 3. Workflow Engine (`workflow_engine.py`)

Two execution modes:
- **RestrictedPython mode:** Uses RestrictedPython compiler with safe wrapper classes (`_SafeMath`, `_SafeJson`, `_SafeRegex`)
- **Basic mode:** AST validation that blocks:
  - `Import`, `ImportFrom` nodes
  - `Exec`, `Eval` function calls
  - Dunder attribute access (`__class__`, `__bases__`, `__subclasses__`, `__globals__`, `__code__`, `__builtins__`, `__import__`, etc.)
  - Dangerous builtins (`exec`, `eval`, `compile`, `__import__`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `dir`, `breakpoint`)

### Sandbox Escape Prevention

The following attack vectors are blocked:
- `re.__class__.__bases__[0].__subclasses__()` → `AttributeError` from `SafeRegex.__getattr__`
- `json.__builtins__.__import__('os')` → `AttributeError` from `SafeJson.__getattr__`
- `().__class__.__bases__[0].__subclasses__()` → Blocked by AST attribute chain validation
- `getattr(str, '__class__')` → `getattr` blocked in builtins

---

## Rate Limiting

### Authentication Rate Limits (Phase 85)

| Endpoint | Limit | Window |
|----------|-------|--------|
| Login | 10 attempts | 60 seconds per IP |
| Register | 5 attempts | 3600 seconds per IP |

Exceeded limits return HTTP 429 with `Retry-After` header.

### LLM Provider Rate Limits (Phase 86)

Per-provider token bucket rate limiting prevents API cost overruns:

| Provider | Default RPM | Configurable |
|----------|-------------|--------------|
| OpenAI | 500 | Yes |
| Anthropic | 60 | Yes |
| Google | 60 | Yes |
| Groq | 30 | Yes |
| Cohere | 100 | Yes |
| Ollama | 10000 (local) | Yes |

When a provider's rate limit is reached, requests are queued with backpressure (async sleep) rather than rejected.

### API Rate Limits

- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated endpoints

---

## Data Protection

### Database Credential Encryption

External database connector credentials are encrypted at rest using the application's `SECRET_KEY` via Fernet symmetric encryption.

### SQL Injection Prevention

- All database queries use SQLAlchemy ORM with parameterized queries
- External database connectors validate queries:
  - Only `SELECT`/`WITH` statements allowed (SQL databases)
  - Only `find`/`aggregate` operations allowed (MongoDB)
  - DDL (`DROP`, `CREATE`, `ALTER`) blocked
  - DML (`INSERT`, `UPDATE`, `DELETE`) blocked
  - Dangerous patterns blocked (`SLEEP`, `BENCHMARK`, `$where`, etc.)

### PII Handling

- PII detection available in RAG pipeline
- Sensitive data filtered from Sentry error reports (API keys, passwords, tokens)
- Session data has TTL-based expiration (1 hour default)

---

## LLM Provider Security

### API Key Management

- API keys stored in database with Fernet encryption
- Keys never logged or exposed in API responses
- Environment variable fallback for development

### Circuit Breaker Pattern (Phase 70)

Resilient LLM calls use circuit breaker pattern:
- **Closed:** Normal operation
- **Open:** After 5 consecutive failures, stops calling provider for 60 seconds
- **Half-open:** After cooldown, allows one test request

This prevents:
- Cascading failures across services
- Runaway API costs from retrying failed providers
- Event loop blocking from unresponsive providers

### Retry Budget (Phase 86)

Per-request retry budget prevents compound retries:
- Maximum 3 total retries per user request across all services
- Retry counter passed via context to prevent query retry → embedding retry → reranker retry chains

---

## Configuration Security

### Secret Key Validation (Phase 85)

```
SECRET_KEY=change-me-in-production  # REJECTED in production/staging
SECRET_KEY=<your-strong-random-key>  # Required for production
```

Generate a secure key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

### Environment-Based Security

| Setting | Development | Production |
|---------|-------------|------------|
| `SECRET_KEY` | Warning logged | Must not be default |
| `DEBUG` | `true` | Must be `false` |
| `ALLOWED_ORIGINS` | `localhost:3000` | Specific domain(s) |
| `JWT_EXPIRATION_HOURS` | 24 | Consider shorter |

### Sensitive Variables

Never commit to version control:
- `SECRET_KEY`, `JWT_SECRET`
- `DATABASE_URL` (if contains password)
- `REDIS_PASSWORD`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `SENTRY_DSN`

Use `.env` files (gitignored) or secrets management (HashiCorp Vault, AWS Secrets Manager).

---

## Audit Logging

### Structured Logging

All security-relevant events are logged with structured context using `structlog`:

```json
{
  "event": "login_failed",
  "ip": "192.168.1.1",
  "email": "user@example.com",
  "error_type": "InvalidCredentials",
  "timestamp": "2026-01-26T12:00:00Z"
}
```

### Logged Events

- Authentication attempts (success/failure)
- Rate limit violations
- Sandbox execution attempts
- Prompt injection detections
- Admin setting changes
- Document access by tier
- Circuit breaker state changes
- Cache invalidation events

### Monitoring Integration

- **Sentry:** Automatic error capture with stack traces and context
- **Prometheus:** Metrics for HTTP requests, LLM calls, token usage, document processing
- **Structured logs:** JSON-formatted for ELK/Grafana Loki ingestion

---

## Security Hardening Checklist

### Before Production Deployment

- [ ] Set a strong, unique `SECRET_KEY` (not the default)
- [ ] Set `DEBUG=false`
- [ ] Set `APP_ENV=production`
- [ ] Configure `ALLOWED_ORIGINS` to your specific domain(s)
- [ ] Use PostgreSQL (not SQLite) for the database
- [ ] Enable Redis for caching and session management
- [ ] Configure `SENTRY_DSN` for error tracking
- [ ] Review and set appropriate rate limits
- [ ] Enable prompt injection detection (`security.prompt_injection_enabled`)
- [ ] Configure PII detection if handling sensitive documents
- [ ] Set up SSL/TLS termination (via reverse proxy)
- [ ] Restrict network access to internal services (Redis, PostgreSQL, Ollama)
- [ ] Review file upload size limits (`MAX_FILE_SIZE`)
- [ ] Enable audit logging
- [ ] Set up monitoring dashboards (Prometheus + Grafana)

### Regular Maintenance

- [ ] Rotate `SECRET_KEY` periodically
- [ ] Update dependencies (`pip install --upgrade`)
- [ ] Review circuit breaker and rate limit configurations
- [ ] Monitor Sentry for security-related errors
- [ ] Review audit logs for suspicious activity
- [ ] Test sandbox escape prevention after code changes
