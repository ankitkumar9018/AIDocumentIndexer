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
9. [DSPy Security Considerations](#dspy-security-considerations-phase-93)
10. [Audit Logging](#audit-logging)
11. [Security Hardening Checklist](#security-hardening-checklist)

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

### Private Documents

Documents can be marked as **private** during upload, providing an additional layer of access control beyond access tiers.

**How Private Documents Work:**

| Scenario | Can Access? |
|----------|-------------|
| Document owner | ✅ Yes |
| Superadmin | ✅ Yes |
| Other users in organization | ❌ No |
| Users in other organizations | ❌ No |

**Setting a Document as Private:**

1. **During Upload:** Toggle "Private Document" in the Processing Options section
2. **Via API:** Set `is_private: true` in the upload request body
3. **After Upload:** Edit document properties in the Documents page

**Private Document Behavior:**

- Private documents are **excluded from search results** for other users
- Private documents **do not appear** in document lists for other users
- Knowledge graph entities from private documents are **filtered at query time** - entities are still extracted but only visible to authorized users
- Superadmins can access all private documents for administrative purposes

**API Example:**

```bash
# Upload a private document
curl -X POST "/api/upload" \
  -F "file=@document.pdf" \
  -F "is_private=true"

# Bulk upload with private flag
curl -X POST "/api/upload/batch" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "is_private=true"
```

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

### Hallucination Scoring (Phase 95J)

Multi-signal hallucination detection is integrated into the RAG pipeline to identify and flag responses that are not grounded in retrieved sources:

- **Source Faithfulness:** Each claim in the generated response is scored against the retrieved source chunks. Claims that cannot be attributed to any source receive a low faithfulness score.
- **Claim Verification:** Individual factual claims are extracted and cross-referenced with the source documents. Unverifiable claims are flagged.
- **Confidence Scoring:** An overall hallucination confidence score (0-1) is computed from the combined signals. Lower scores indicate higher likelihood of hallucination.

Hallucination scores are exposed in API responses for transparency, allowing downstream consumers to make informed decisions about response trustworthiness. Configurable thresholds enable automated flagging: responses scoring below the configured threshold are annotated with a hallucination warning in the API response payload.

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.hallucination_detection_enabled` | Enable hallucination scoring | `true` |
| `rag.hallucination_threshold` | Score below which responses are flagged (0-1) | `0.5` |

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
- **RestrictedPython mode:** Uses RestrictedPython compiler with safe wrapper classes (`_SafeMath`, `_SafeJson`, `_SafeRegex`, `_SafeDatetime`)
- **Basic mode:** AST validation that blocks:
  - `Import`, `ImportFrom` nodes
  - `Exec`, `Eval` function calls
  - Dunder attribute access (`__class__`, `__bases__`, `__subclasses__`, `__globals__`, `__code__`, `__builtins__`, `__import__`, etc.)
  - Dangerous builtins (`exec`, `eval`, `compile`, `__import__`, `open`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `dir`, `breakpoint`)

#### Phase 91/94 Sandbox Hardening

**`_SafeDatetime` Wrapper (Phase 94):**
The `_SafeDatetime` class wraps all `datetime` module access to prevent sandbox escape via datetime module attributes. Only whitelisted operations (`now`, `utcnow`, `today`, `strftime`, `strptime`, `timedelta`, `date`, `time`, `datetime`) are permitted. Any attempt to access dunder attributes or internal module attributes raises `AttributeError`, closing an escape vector where attackers could traverse `datetime.datetime.__class__.__bases__` to reach the object hierarchy.

**AST-Level `blocked_attrs` List (Phase 91):**
The AST validator maintains an explicit `blocked_attrs` list that rejects any attribute access node referencing the following dunder attributes:
- `__class__`, `__bases__`, `__subclasses__`
- `__globals__`, `__code__`, `__builtins__`
- `__import__`, `__getattribute__`, `__setattr__`, `__delattr__`

This provides defense-in-depth: even if a safe wrapper is bypassed, the AST validator will reject the code before execution.

**Phase 91 Security Tests:**
Comprehensive unit tests validate sandbox escape prevention:
- `test_sandbox.py` — Tests all blocked builtins, import attempts, and attribute traversal attacks
- `test_security.py` — Tests prompt injection detection, PII masking, and content safety scoring
- `test_agent_memory.py` — Tests that agent memory operations cannot be exploited for code execution

### Sandbox Escape Prevention

The following attack vectors are blocked:
- `re.__class__.__bases__[0].__subclasses__()` → `AttributeError` from `SafeRegex.__getattr__`
- `json.__builtins__.__import__('os')` → `AttributeError` from `SafeJson.__getattr__`
- `().__class__.__bases__[0].__subclasses__()` → Blocked by AST attribute chain validation
- `getattr(str, '__class__')` → `getattr` blocked in builtins

#### Phase 91/94 Blocked Patterns

The following additional escape vectors are blocked by Phase 91 and Phase 94 hardening:

| Attack Pattern | Blocked By | Mechanism |
|----------------|-----------|-----------|
| `datetime.datetime.__class__.__bases__` | `_SafeDatetime.__getattr__` | Wrapper rejects all dunder attribute access on datetime objects |
| `re.compile.__globals__` | `SafeRegex.__getattr__` | Wrapper rejects `__globals__` access on regex function references |
| `json.dumps.__globals__['__builtins__']` | `SafeJson.__getattr__` | Wrapper rejects `__globals__` traversal to reach builtins |

These safe wrappers operate as a first line of defense. The AST-level `blocked_attrs` validator (Phase 91) acts as a second line, rejecting code containing any `__class__`, `__bases__`, `__subclasses__`, `__globals__`, `__code__`, `__builtins__`, `__import__`, `__getattribute__`, `__setattr__`, or `__delattr__` attribute access before the code is ever executed.

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

### Bring Your Own Key (BYOK) Security Model (Phase 95R)

BYOK allows users to provide their own LLM provider API keys instead of relying on server-managed keys. This model has specific security properties:

**Supported Providers:** OpenAI, Anthropic, Google AI, Mistral, Groq, Together

**Client-Side Key Storage:**
- Keys are stored in encrypted `localStorage` with a master BYOK toggle
- A master toggle enables/disables BYOK mode globally for the user session
- Password-style input fields with show/hide toggle prevent shoulder surfing during key entry

**Key Isolation:**
- BYOK keys are **never sent to the backend server** -- they are injected client-side into API requests directly from the browser
- This eliminates the risk of server-side key leakage, database compromise exposure, or logging of user keys
- The backend never stores, processes, or has access to user-provided BYOK keys

**Key Validation:**
- A dedicated key validation endpoint tests keys against the provider's API before saving
- Invalid or revoked keys are rejected at entry time with clear error messages
- Validation requests are made directly from the client to the provider (not proxied through the backend)

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

## DSPy Security Considerations (Phase 93)

DSPy optimization provides automated prompt tuning and few-shot example management. The following security measures are enforced:

### Access Control

- All DSPy optimization endpoints require **admin-only access** (JWT with `admin` role)
- Non-admin users cannot trigger optimization jobs, view training data, or access compiled prompts
- Role enforcement is applied at the API route level via JWT middleware

### Training Data Safety

- Training examples are **validated for content safety** before storage
- Examples containing prompt injection patterns, PII, or unsafe content are rejected
- Validation uses the same `rag_security` pipeline applied to user queries

### Compiled Prompt Storage

- Compiled/optimized prompts are stored in the **database as data**, not as executable code
- Prompts are treated as inert text and are never `eval`/`exec`'d
- Database storage ensures prompts are subject to the same access control and encryption as other application data

### Resource Limits

- Optimization jobs run with **timeout limits** to prevent resource exhaustion
- Long-running optimization tasks are terminated after the configured timeout
- Job status is tracked to prevent concurrent optimization runs from overwhelming system resources

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
