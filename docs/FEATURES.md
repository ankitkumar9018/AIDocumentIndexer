# AIDocumentIndexer - New Features Guide

This guide covers the new features introduced in AIDocumentIndexer, including the Visual Workflow Builder, Audio Overviews, Connectors, and LLM Gateway.

---

## Table of Contents

1. [Visual Workflow Builder](#visual-workflow-builder)
2. [Audio Overviews](#audio-overviews)
3. [Connectors](#connectors)
4. [Database Connector](#database-connector)
5. [Web Scraper](#web-scraper)
6. [LLM Gateway](#llm-gateway)
7. [Knowledge Graph](#knowledge-graph)
8. [Contextual Chunking](#contextual-chunking)
9. [Settings Presets](#settings-presets)
10. [Phase 78-83 Features](#phase-78-83-features-january-2026)
11. [Phase 94: Cross-Section RAG](#cross-section-rag-integration-phase-94)
12. [Phase 95: UI Overhaul](#phase-95-ui-overhaul--competitor-inspired-features)
    - [Skills Marketplace](#skills-marketplace-95s)
    - [Mood Board Creator](#mood-board-creator-95t)
    - [Deep Research Mode](#deep-research-mode-95u)
    - [Provider-Aware LLM Selection](#provider-aware-llm-selection-global-feature)
13. [Phase 97: Performance & Security](#phase-97-performance--security-optimizations-january-2026)
14. [Recent Enhancements (February 2026)](#recent-enhancements-february-2026)
    - [Private Documents](#private-documents)
    - [Memory Manager](#memory-manager-openclaw-inspired)
    - [Adaptive Chunking](#adaptive-chunking-moltbot-inspired)
    - [Auto-Tagging](#auto-tagging)
    - [AI Image Analysis](#ai-image-analysis-multimodal-rag)
    - [Duplicate Detection Improvements](#duplicate-detection-improvements)
15. [Future Roadmap (Research-Based)](#future-roadmap-research-based)

---

## Visual Workflow Builder

The Workflow Builder provides a drag-and-drop visual interface to create automated task workflows. This feature is inspired by tools like n8n and Langdock, allowing you to chain together actions, conditions, and AI agents.

### Accessing the Workflow Builder

Navigate to **Dashboard > Workflows** to see all your workflows and create new ones.

### Creating a New Workflow

1. Click **New Workflow** button in the top right
2. Enter a workflow name and optional description
3. Select a trigger type:
   - **Manual**: Run workflow on-demand
   - **Scheduled**: Run on a cron schedule
   - **Webhook**: Trigger via HTTP webhook
   - **Form**: Trigger from form submission
   - **Event**: Trigger on system events

4. Click **Create Workflow** to open the visual editor

### Workflow Editor Interface

The workflow editor uses React Flow and consists of:

- **Canvas**: The main area where you build your workflow
- **Node Palette (Left Panel)**: Contains nodes organized in three tabs:
  - **Control**: START, END, Condition, Loop, Delay
  - **Action**: HTTP Request, Notification, Code
  - **AI**: Agent nodes for AI processing

- **Config Panel (Right Sheet)**: Opens when you click a node to configure it
- **Toolbar (Top)**: Save, Publish, Execute buttons

### Available Node Types

| Node Type | Category | Description |
|-----------|----------|-------------|
| START | Control | Entry point for the workflow |
| END | Control | Exit point for the workflow |
| ACTION | Action | General action node (create/update/delete docs, webhooks) |
| CONDITION | Control | Branch based on expressions (has true/false outputs) |
| LOOP | Control | Iterate over collections |
| CODE | Action | Run custom Python or JavaScript code |
| DELAY | Control | Wait for specified duration |
| HTTP | Action | Make HTTP requests to external APIs |
| NOTIFICATION | Action | Send notifications (email, Slack, webhook) |
| AGENT | AI | Run an AI agent with a prompt template |
| HUMAN_APPROVAL | Control | Pause for manual approval |

### Connecting Nodes

1. Drag from a node's output handle (bottom) to another node's input handle (top)
2. Condition and Loop nodes have two outputs:
   - **Green (left)**: True/Yes path
   - **Red (right)**: False/No path

### Configuring Nodes

Click any node to open its configuration panel. Common settings include:

**Action Node:**
- Action Type: create_document, update_document, delete_document, send_email, webhook, custom

**Condition Node:**
- Condition Type: expression, equals, contains, exists
- Expression: Use `{{variable}}` syntax for data references

**Loop Node:**
- Loop Type: for_each, while, count
- Items Source: Reference to array (e.g., `{{input.documents}}`)
- Max Iterations: Safety limit

**HTTP Node:**
- Method: GET, POST, PUT, PATCH, DELETE
- URL: Target endpoint
- Headers: JSON object
- Body: JSON with variable substitution

**Agent Node:**
- Agent ID: Select or enter agent identifier
- Prompt Template: Instructions with `{{input.data}}` placeholders
- Wait for Result: Whether to block until agent completes

### Variable Substitution

Use double curly braces for dynamic values:
- `{{input.field}}` - Access input data
- `{{node_id.output}}` - Reference another node's output
- `{{workflow.name}}` - Workflow metadata

### Saving and Publishing

1. **Save**: Saves current state (can be a draft)
2. **Publish**: Makes the workflow executable
3. **Execute**: Runs the workflow (requires published state)

### Execution History

Click **History** button to view past executions with:
- Status (success/failed/running)
- Start/end times
- Input data
- Output/error messages

---

## Audio Overviews

Audio Overviews generates AI podcast-style discussions from your documents, similar to NotebookLM. This feature uses LLM to create engaging scripts and text-to-speech to produce audio.

### Accessing Audio Overviews

Navigate to **Dashboard > Audio** to create and manage audio overviews.

### Available Formats

| Format | Duration | Description |
|--------|----------|-------------|
| Deep Dive | 15-20 min | Comprehensive exploration with two hosts |
| Brief Summary | 5 min | Quick overview of key points |
| Critique | 10-15 min | Analysis of strengths and weaknesses |
| Debate | 12-15 min | Two hosts with contrasting viewpoints |
| Lecture | 10-15 min | Educational single-speaker format |
| Interview | 12-15 min | Q&A style with expert |

### Creating an Audio Overview

1. Click **New Audio** button
2. **Select Format**: Choose the discussion style
3. **Select Documents**: Pick 1 or more documents to discuss
4. **Title (optional)**: Custom title for the audio
5. **Custom Instructions (optional)**: Guide the AI on topics, style, focus areas
6. Review the **Cost Estimate** showing:
   - Estimated duration
   - Script generation cost
   - Text-to-speech cost
   - Total cost
7. Click **Generate Audio**

### Audio Status States

- **Pending**: Waiting in queue
- **Writing Script**: LLM generating the dialogue
- **Generating Audio**: TTS converting script to audio
- **Ready**: Audio is ready to play
- **Failed**: Error occurred (can retry)

### Playing Audio

1. Click on any completed audio overview card
2. Audio player opens with:
   - Play/pause controls
   - Progress bar
   - Volume control
   - Transcript sync (highlights current segment)
3. Click **Close** to close the player

### Cost Management

Audio generation incurs costs for:
1. **Script Generation**: LLM API calls to create dialogue
2. **Text-to-Speech**: OpenAI TTS or other providers

Costs are estimated before generation so you can decide if it fits your budget.

---

## Connectors

Connectors allow you to sync documents from external data sources automatically. Supported connectors include Google Drive, Notion, Confluence, OneDrive/SharePoint, and YouTube.

### Accessing Connectors

Navigate to **Dashboard > Connectors** to manage your connections.

### Available Connectors

| Connector | Auth Type | Description |
|-----------|-----------|-------------|
| Google Drive | OAuth 2.0 | Sync files from Google Drive folders |
| Notion | API Token | Sync pages and databases from Notion |
| Confluence | API Token | Sync pages from Atlassian Confluence |
| OneDrive/SharePoint | OAuth 2.0 | Sync files from Microsoft 365 |
| YouTube | API Key | Transcribe and index YouTube videos |

### Adding a Connector

1. Click **Add Connector**
2. Select the connector type
3. Enter a name for this connection
4. Click **Create**
5. For OAuth connectors (Google Drive, OneDrive):
   - You'll be redirected to the provider's login page
   - Grant permissions
   - You'll return to the app with a connected status

### Connector Configuration

After creation, configure sync settings:

- **Sync Folders**: Select which folders/pages to sync
- **File Types**: Filter by file extensions
- **Sync Schedule**: How often to check for updates
  - Manual only
  - Every hour
  - Every 6 hours
  - Daily
  - Weekly

### Connector Status

| Status | Meaning |
|--------|---------|
| Connected | Active and ready to sync |
| Syncing | Currently syncing files |
| Pending Setup | OAuth not completed |
| Error | Sync failed (check error message) |
| Inactive | Manually paused |

### Manual Sync

Click the refresh icon on any connector card to trigger an immediate sync.

### Managing Connectors

- **Toggle Active**: Enable/disable automatic syncing
- **Settings**: Open configuration dialog
- **Delete**: Remove connector and stop syncing

### Sync History

Each connector shows:
- Last sync time
- Number of documents synced
- Errors encountered

---

## Database Connector

The Database Connector allows you to connect to external databases (PostgreSQL, MySQL, MongoDB, SQLite) and query them using natural language. An LLM translates your questions into SQL (or MongoDB aggregations), executes the query safely, and returns results with explanations.

### Accessing Database Connector

Navigate to **Dashboard > Connectors > Database** or use the database page directly.

### Supported Databases

| Database | Driver | Features |
|----------|--------|----------|
| PostgreSQL | asyncpg | Full schema introspection, JSON support |
| MySQL | aiomysql | Full schema introspection |
| MongoDB | motor | Collection introspection, aggregation pipelines |
| SQLite | aiosqlite | File-based databases |

### Connecting to a Database

1. Click **Add Connection** in the Database Connector section
2. Select the database type (PostgreSQL, MySQL, SQLite)
3. Enter connection details:
   - **Name**: A friendly name for this connection
   - **Host**: Database server hostname (e.g., `localhost`, `db.example.com`)
   - **Port**: Database port (default: 5432 for PostgreSQL, 3306 for MySQL)
   - **Database**: Database name
   - **Username**: Database user
   - **Password**: Database password (stored encrypted)
   - **SSL Mode** (PostgreSQL): `disable`, `require`, `verify-ca`, `verify-full`
4. Click **Test Connection** to verify connectivity
5. Click **Save** to store the connection

### Querying with Natural Language

Once connected, you can ask questions in plain English:

**Example Questions:**
- "Show me all customers from Germany"
- "What are the top 10 products by revenue?"
- "How many orders were placed last month?"
- "List employees who joined in 2024"
- "What's the average order value by country?"

**How It Works:**

1. **Schema Introspection**: The system reads your database schema (tables, columns, types, relationships)
2. **Query Generation**: An LLM generates a SQL query based on your question and schema
3. **Validation**: The query is validated for safety (only SELECT queries allowed)
4. **Execution**: The query runs with a timeout and row limit
5. **Results**: Data is returned in a table format with an explanation

### Query Interface

The query interface shows:

| Tab | Content |
|-----|---------|
| **Results** | Data table with sorting and filtering |
| **SQL** | Generated SQL query with syntax highlighting |
| **Explanation** | Natural language explanation of what the query does |

### Safety Features

All queries are validated for security:

- **Read-only**: Only SELECT queries are allowed
- **No DDL**: DROP, CREATE, ALTER statements are blocked
- **No DML**: INSERT, UPDATE, DELETE statements are blocked
- **Timeout**: Queries timeout after configurable seconds (default: 30s)
- **Row Limit**: Maximum rows returned (default: 1000)
- **Credential Encryption**: Passwords are encrypted at rest

### Query History

Click **History** to see past queries with:
- Natural language question asked
- Generated SQL
- Execution time
- Row count
- Feedback rating (if provided)

### Feedback & Few-Shot Learning

You can rate query results with thumbs up/down. Good examples are stored for few-shot learning, improving future query generation accuracy.

### Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `DATABASE_CONNECTOR_MAX_ROWS` | Maximum rows returned | 1000 |
| `DATABASE_CONNECTOR_TIMEOUT` | Query timeout in seconds | 30 |
| `DATABASE_CONNECTOR_CACHE_ENABLED` | Cache repeated queries | true |

### API Endpoints

```
POST   /api/v1/database/connections           # Create connection
GET    /api/v1/database/connections           # List connections
GET    /api/v1/database/connections/{id}      # Get connection details
DELETE /api/v1/database/connections/{id}      # Delete connection
POST   /api/v1/database/connections/{id}/test # Test connectivity
GET    /api/v1/database/connections/{id}/schema # Get database schema
POST   /api/v1/database/connections/{id}/query  # Natural language query
GET    /api/v1/database/connections/{id}/history # Query history
POST   /api/v1/database/history/{id}/feedback   # Submit feedback
```

### Example: Querying a PostgreSQL Database

```bash
# 1. Create a connection
curl -X POST http://localhost:8000/api/v1/database/connections \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production DB",
    "connector_type": "postgresql",
    "host": "db.example.com",
    "port": 5432,
    "database": "myapp",
    "username": "readonly_user",
    "password": "secret",
    "ssl_mode": "require"
  }'

# 2. Test the connection
curl -X POST http://localhost:8000/api/v1/database/connections/{id}/test \
  -H "Authorization: Bearer $TOKEN"

# 3. Query with natural language
curl -X POST http://localhost:8000/api/v1/database/connections/{id}/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top 5 customers by total order value?",
    "execute": true,
    "explain": true
  }'
```

**Response:**
```json
{
  "success": true,
  "natural_language_query": "What are the top 5 customers by total order value?",
  "generated_sql": "SELECT c.name, SUM(o.total) as total_value FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_value DESC LIMIT 5",
  "explanation": "This query joins the customers and orders tables, sums the order totals for each customer, and returns the top 5 customers ranked by their total order value.",
  "query_result": {
    "success": true,
    "columns": ["name", "total_value"],
    "rows": [
      ["Acme Corp", 125000.00],
      ["TechStart Inc", 98500.00],
      ...
    ],
    "row_count": 5,
    "execution_time_ms": 45
  },
  "confidence": 0.85
}
```

---

## Web Scraper

The Web Scraper allows you to extract content from web pages and add it to your knowledge base for RAG queries, document generation, and more.

### Accessing the Web Scraper

Navigate to **Dashboard > Web Scraper** to scrape web content.

### Scraping Modes

| Mode | Description |
|------|-------------|
| **Quick Scrape** | Extract content immediately without saving. Review first, then optionally save to knowledge base. |
| **Save to Index** | Scrape and immediately index content into the RAG pipeline (embeddings + vector store + Knowledge Graph). |
| **Scrape & Query** | Scrape a URL and immediately ask questions about the content using an LLM. |
| **Extract Links** | Discover all links from a page for later scraping. |
| **Sitemap Crawl** | Crawl a website using its sitemap.xml for URL discovery. Prioritizes pages by `lastmod` date. |
| **Search & Crawl** | Search the web via DuckDuckGo, then crawl the top results. Useful for topic-based content acquisition. |

### Quick Scrape Workflow

1. Enter a URL and click **Scrape** with "Quick Scrape" mode selected
2. Review the scraped content in the Results panel
3. If you want to keep the content, click **Save to Knowledge Base**
4. The content is chunked, embedded, and indexed for future RAG queries

This workflow is useful when you want to review content before committing it to your knowledge base.

### Subpage Crawling

Enable **Crawl Subpages** to automatically follow links and scrape multiple pages:

- **Max Depth**: How many levels deep to follow links (1-10)
- **Same Domain Only**: Only crawl pages from the same domain (recommended)

### What Happens When Content is Indexed

When you save scraped content (either via "Save to Index" mode or the "Save to Knowledge Base" button):

1. **Chunking**: Content is split into semantic chunks (~1000 characters)
2. **Embedding**: Each chunk is converted to a vector embedding
3. **Vector Store**: Embeddings are indexed for semantic search
4. **Knowledge Graph**: Entities and relationships are extracted and stored

Once indexed, the content becomes searchable in:
- Chat/RAG queries
- Document generation (PPTX, DOCX, PDF)
- Audio overviews
- Workflow RAG nodes

### Configuration Options

| Option | Description |
|--------|-------------|
| `extract_links` | Extract hyperlinks from pages |
| `extract_images` | Extract image URLs |
| `extract_metadata` | Extract meta tags (title, description, etc.) |
| `wait_for_js` | Wait for JavaScript to load before scraping |
| `timeout` | Request timeout in seconds |

### Best Practices

1. **Review before saving**: Use Quick Scrape to preview content quality
2. **Use subpage crawling sparingly**: Start with max_depth=2 to avoid scraping too many pages
3. **Enable same_domain_only**: Prevents accidentally scraping unrelated sites
4. **Check for duplicates**: The system tracks content hashes to avoid duplicate storage

### Phase 96 Enhancements

#### Sitemap Crawling
- Fetches and parses `sitemap.xml` (including sitemap index files and `.xml.gz` compressed sitemaps)
- Discovers sitemap URLs from `robots.txt` automatically
- Prioritizes pages by `lastmod` date (newest first)
- Configurable max pages limit (1-500)
- Supports permanent storage mode for automatic RAG indexing

#### Search & Crawl
- Uses DuckDuckGo search (free, no API key required) for URL discovery
- Enter a search query to find relevant web pages, then automatically crawl the results
- Configurable max results (1-20)
- Results can be stored permanently in the knowledge base

#### Scheduled / Recurring Crawls
- Set up cron-based recurring crawl schedules (e.g., "every 6 hours", "daily at midnight")
- Powered by Celery Beat task scheduling
- Content hash tracking between runs — only re-indexes changed content
- CRUD management via API: create, list, update, delete, and manually trigger schedules
- Useful for monitoring competitor sites, news sources, or documentation

#### SSE Progress Streaming
- Real-time crawl progress via Server-Sent Events (SSE)
- Frontend displays: progress bar, current URL being crawled, pages found/crawled count
- Event types: `status` (job state changes), `page_complete` (per-page results), `complete` (final summary)
- Auto-connects when starting job-mode crawls

#### Proxy Rotation
- Configurable proxy pool with round-robin or random rotation strategies
- Supports HTTP and SOCKS5 proxy URLs
- Enable via `scraping.proxy_enabled` setting
- Helps avoid IP-based rate limiting for large crawl operations

#### Jina Reader Fallback
- Automatic fallback to Jina Reader API (`r.jina.ai`) when Crawl4AI and basic HTTP scraping fail
- Zero-infrastructure setup — works without a headless browser
- High-quality markdown output with automatic boilerplate removal
- Free tier: up to 1000 requests/day
- Enable/disable via `scraping.jina_reader_fallback` setting

#### Robots.txt Compliance
- Full `robots.txt` parsing with `urllib.robotparser`
- Respects `Disallow` rules and `Crawl-delay` directives
- API endpoint to inspect a domain's robots.txt rules before crawling
- Connected to the `scraping.respect_robots_txt` admin setting

#### Crash Recovery
- Persists crawl state for long-running jobs
- Resume interrupted crawls without re-crawling already-completed pages
- Enabled by default via `scraping.crash_recovery_enabled` setting

#### Adaptive Crawling
- Confidence-based auto-stop using crawl4ai's AdaptiveCrawler
- Three-layer scoring: content coverage, consistency, and saturation
- Automatically determines when a site has been sufficiently crawled
- Enable via `scraping.adaptive_crawling` setting

---

## LLM Gateway

The LLM Gateway provides centralized management for LLM API usage with budget enforcement and API key management. This is useful for controlling costs and monitoring usage across your organization.

### Accessing the Gateway

Navigate to **Dashboard > Gateway** to manage budgets and API keys.

### Overview Dashboard

The Gateway page shows:
- **Total Spending**: Current period spending
- **Budget Utilization**: Percentage of budget used
- **API Calls**: Number of requests made
- **Average Latency**: Response time metrics

### Budget Management

Budgets help you control LLM API spending with soft or hard limits.

#### Creating a Budget

1. Click **Create Budget**
2. Fill in:
   - **Name**: Descriptive name (e.g., "Monthly Production")
   - **Period**: Daily, Weekly, or Monthly
   - **Limit Amount**: Budget cap in USD
   - **Hard Limit**:
     - **Off**: Warn when exceeded but allow requests
     - **On**: Block requests when exceeded
3. Click **Create Budget**

#### Budget Status

- **Green**: Under 75% utilization
- **Yellow**: 75-90% utilization
- **Red**: Over 90% or exceeded

### API Key Management

Virtual API keys allow you to distribute access while tracking usage.

#### Creating an API Key

1. Click **Create API Key**
2. Enter:
   - **Name**: Identify the key's purpose
   - **Scopes** (optional): Limit to specific models
   - **Expires** (optional): Days until expiration
3. Click **Create**
4. **IMPORTANT**: Copy the key immediately! It won't be shown again.

#### Key Operations

- **Copy**: Copy key to clipboard
- **Revoke**: Immediately invalidate the key
- **Rotate**: Generate a new key (old key still works briefly)

### Usage Stats

The usage tab shows:
- Requests over time (chart)
- Token usage by model
- Cost breakdown
- Top users/keys

### API Gateway Endpoint

Use the gateway endpoint in your applications:

```
POST /api/v1/gateway/v1/chat/completions
Authorization: Bearer your-virtual-api-key
```

The gateway is OpenAI-compatible, so you can use standard OpenAI client libraries by changing the base URL.

---

## Keyboard Shortcuts

### Workflow Editor
- `Delete` / `Backspace`: Delete selected node
- `Ctrl/Cmd + S`: Save workflow
- `Ctrl/Cmd + Z`: Undo (via React Flow)

### Audio Player
- `Space`: Play/Pause
- Arrow keys: Seek forward/back

---

## Troubleshooting

### Workflow Issues

**Workflow won't save:**
- Check network connection
- Ensure all required node configs are filled
- Try refreshing the page

**Execution fails:**
- Check execution history for error details
- Verify external API credentials
- Ensure loops have proper termination conditions

### Audio Issues

**Generation stuck:**
- Check if backend is running
- View logs for TTS provider errors
- Try a shorter document first

**Audio won't play:**
- Verify audio file was generated (check status)
- Try different browser
- Check browser audio permissions

### Connector Issues

**OAuth fails:**
- Ensure callback URLs are configured correctly
- Check provider app credentials
- Try revoking and re-connecting

**Sync not working:**
- Verify credentials haven't expired
- Check folder permissions in source system
- Look at connector error message

### Gateway Issues

**API key not working:**
- Verify key hasn't been revoked
- Check key hasn't expired
- Ensure scopes allow the requested model

**Budget not enforcing:**
- Confirm budget is active (not paused)
- Check period (daily resets at midnight UTC)
- Verify hard_limit is enabled if you want blocking

---

## API Reference

For detailed API documentation, see [API.md](./API.md).

### Quick Reference

**Workflows:**
- `GET /api/v1/workflows` - List workflows
- `POST /api/v1/workflows` - Create workflow
- `POST /api/v1/workflows/{id}/execute` - Execute workflow
- `POST /api/v1/workflows/{id}/deploy` - Publish workflow for external access
- `POST /api/v1/workflows/{id}/share` - Create share link
- `POST /api/v1/workflows/{id}/schedule` - Schedule recurring execution
- `GET /api/v1/workflows/{id}/versions` - List version history

**Skills:**
- `GET /api/v1/skills/list` - List all skills
- `POST /api/v1/skills` - Create custom skill
- `POST /api/v1/skills/execute` - Execute a skill
- `POST /api/v1/skills/{id}/publish` - Publish skill for external access

**Audio:**
- `GET /api/v1/audio/overviews` - List audio overviews
- `POST /api/v1/audio/overviews` - Create audio overview
- `POST /api/v1/audio/overviews/{id}/generate` - Generate audio

**Connectors:**
- `GET /api/v1/connectors` - List connectors
- `POST /api/v1/connectors` - Create connector
- `POST /api/v1/connectors/{id}/sync` - Trigger sync

**Gateway:**
- `GET /api/v1/gateway/budgets` - List budgets
- `POST /api/v1/gateway/budgets` - Create budget
- `GET /api/v1/gateway/keys` - List API keys
- `POST /api/v1/gateway/keys` - Create API key

---

## Workflow Deployment & Sharing

Workflows can be published for external access and shared with team members or external users through share links with granular permissions.

### Workflow Deployment (Publishing)

Deploy workflows to make them accessible via a public URL without authentication.

**Features:**
- **Public URL**: Each deployed workflow gets a unique public slug (e.g., `/w/my-workflow`)
- **Automatic Form Generation**: Input fields are automatically rendered based on the workflow's input schema
- **Execution Tracking**: All executions are logged with status and outputs
- **Branding Support**: Optional company logo, primary color, and branding customization

**Deploying a Workflow:**
1. Navigate to **Dashboard > Workflows**
2. Open the workflow you want to deploy
3. Click **Deploy** in the workflow settings
4. Configure the public slug and branding options
5. Copy the public URL to share

**Public Workflow Page:**
- Displays workflow name and description
- Shows all input fields with appropriate UI controls (text, number, select, checkbox, textarea)
- Executes the workflow and displays results
- Polls for status updates during execution

**API Endpoints:**
```bash
# Deploy workflow
POST /api/v1/workflows/{workflow_id}/deploy
{
  "public_slug": "my-workflow",
  "branding": {
    "logo": "https://...",
    "primaryColor": "#8b5cf6",
    "companyName": "My Company"
  }
}

# Undeploy workflow
POST /api/v1/workflows/{workflow_id}/undeploy

# Get deploy status
GET /api/v1/workflows/{workflow_id}/deploy-status
```

### Workflow Sharing (Share Links)

Create secure share links with fine-grained permissions, password protection, and expiration.

**Permission Levels:**

| Level | Capabilities |
|-------|-------------|
| `viewer` | View workflow details and input schema only |
| `executor` | View and execute the workflow |
| `editor` | View, execute, and duplicate the workflow |

**Security Options:**
- **Password Protection**: Optional password required to access the share link
- **Expiration**: Set an expiration date/time after which the link becomes invalid
- **Max Uses**: Limit the number of times the share link can be used

**Creating a Share Link:**
```bash
POST /api/v1/workflows/{workflow_id}/share
{
  "permission_level": "executor",
  "password": "optional-password",
  "expires_at": "2026-03-01T00:00:00Z",
  "max_uses": 100
}
```

**Response:**
```json
{
  "share_id": "uuid",
  "token": "abc123xyz",
  "share_url": "https://app.example.com/shared/workflow/abc123xyz",
  "permission_level": "executor",
  "expires_at": "2026-03-01T00:00:00Z",
  "max_uses": 100,
  "use_count": 0
}
```

**Managing Shares:**
```bash
# List all shares for a workflow
GET /api/v1/workflows/{workflow_id}/shares

# Revoke a share link
DELETE /api/v1/workflows/{workflow_id}/shares/{share_id}
```

---

## Workflow Scheduling

Schedule workflows to run automatically on a recurring basis using cron expressions.

### Features

- **Cron-Based Scheduling**: Standard 5-field cron expressions (minute, hour, day, month, weekday)
- **Timezone Support**: Schedule in any timezone
- **Celery Beat Integration**: Reliable execution via Celery Beat
- **Execution History**: Track all scheduled executions

### Common Cron Patterns

| Pattern | Description |
|---------|-------------|
| `0 9 * * *` | Daily at 9:00 AM |
| `0 9 * * 1-5` | Weekdays at 9:00 AM |
| `*/15 * * * *` | Every 15 minutes |
| `0 0 1 * *` | Monthly on the 1st at midnight |
| `0 */6 * * *` | Every 6 hours |

### Scheduling a Workflow

**Via UI:**
1. Open the workflow in the designer
2. Click **Schedule** in the toolbar
3. Enter a cron expression or use the preset options
4. Select timezone
5. Save the schedule

**Via API:**
```bash
POST /api/v1/workflows/{workflow_id}/schedule
{
  "cron_expression": "0 9 * * *",
  "timezone": "America/New_York",
  "input_data": {
    "param1": "default-value"
  }
}
```

**Response:**
```json
{
  "schedule_id": "uuid",
  "workflow_id": "workflow-uuid",
  "cron_expression": "0 9 * * *",
  "timezone": "America/New_York",
  "next_run": "2026-02-03T09:00:00-05:00",
  "is_active": true
}
```

### Managing Schedules

```bash
# Get current schedule
GET /api/v1/workflows/{workflow_id}/schedule

# Delete schedule
DELETE /api/v1/workflows/{workflow_id}/schedule

# List upcoming scheduled executions (all workflows)
GET /api/v1/workflows/schedules/upcoming?limit=10
```

---

## Workflow Triggers

Workflows can be triggered in multiple ways beyond manual execution.

### Form Triggers

Create a public form that triggers workflow execution when submitted.

**Features:**
- Auto-generated form based on workflow input schema
- Custom form title and description
- Submission confirmation message
- Rate limiting options

**Configuration:**
```bash
POST /api/v1/workflows/{workflow_id}/form-trigger
{
  "enabled": true,
  "title": "Contact Request Form",
  "description": "Submit your inquiry",
  "success_message": "Thank you! We'll be in touch.",
  "rate_limit_per_minute": 10
}
```

**Form Submission:**
```bash
POST /api/v1/workflows/{workflow_id}/form-submit
{
  "name": "John Doe",
  "email": "john@example.com",
  "message": "I have a question..."
}
```

### Event Triggers

Trigger workflows based on system events from connectors or other integrations.

**Supported Events:**
| Event Type | Description |
|------------|-------------|
| `document.created` | New document indexed |
| `document.updated` | Document content changed |
| `connector.sync_completed` | Connector sync finished |
| `chat.message_received` | New chat message received |

**Configuration:**
```bash
POST /api/v1/workflows/{workflow_id}/event-trigger
{
  "event_type": "document.created",
  "filter_conditions": {
    "source": "notion",
    "tags": ["important"]
  }
}
```

**Listing Event Triggers:**
```bash
GET /api/v1/workflows/event-triggers
```

### Webhook Triggers

Trigger workflows via HTTP webhook from external systems.

```bash
POST /api/v1/workflows/webhook/{workflow_id}
{
  "custom_param": "value",
  "data": { ... }
}
```

---

## Workflow Versioning

Track changes to workflows with version history and rollback capability.

### Features

- **Automatic Versioning**: Each save creates a new version
- **Version Comparison**: View changes between versions
- **Rollback**: Restore any previous version
- **Version Metadata**: Timestamps, author, and change notes

### Viewing Version History

```bash
GET /api/v1/workflows/{workflow_id}/versions
```

**Response:**
```json
{
  "versions": [
    {
      "version": 3,
      "created_at": "2026-02-02T10:30:00Z",
      "created_by": "user-uuid",
      "node_count": 5,
      "is_current": true
    },
    {
      "version": 2,
      "created_at": "2026-02-01T15:00:00Z",
      "created_by": "user-uuid",
      "node_count": 4,
      "is_current": false
    }
  ]
}
```

### Restoring a Version

```bash
POST /api/v1/workflows/{workflow_id}/versions/{version}/restore
```

This creates a new version with the restored content, preserving the full history.

---

## Skills Publishing

Publish custom skills for external access, enabling third-party integrations and public APIs.

### Publishing a Skill

**Via UI:**
1. Navigate to **Dashboard > Skills**
2. Open the skill you want to publish
3. Click **Publish** in the skill settings
4. Configure the public slug
5. Copy the public URL

**Via API:**
```bash
POST /api/v1/skills/{skill_id}/publish
{
  "public_slug": "my-summarizer"
}
```

**Response:**
```json
{
  "skill_id": "uuid",
  "public_slug": "my-summarizer",
  "public_url": "https://app.example.com/api/v1/public/skills/my-summarizer",
  "is_published": true
}
```

### Public Skill Endpoints

Once published, skills are accessible via public API:

```bash
# Get skill info (public)
GET /api/v1/public/skills/{public_slug}

# Execute skill (public)
POST /api/v1/public/skills/{public_slug}/execute
{
  "inputs": {
    "text": "Content to process..."
  }
}
```

### Managing Published Skills

```bash
# Get publish status
GET /api/v1/skills/{skill_id}/publish-status

# Unpublish skill
POST /api/v1/skills/{skill_id}/unpublish
```

---

## Knowledge Graph

The Knowledge Graph feature extracts entities and relationships from your documents, enabling graph-based exploration and queries.

### Accessing the Knowledge Graph

Navigate to **Dashboard > Knowledge Graph** to view and explore entities.

### Features

#### Entity Extraction
- Automatically extracts named entities (People, Organizations, Locations, Concepts, etc.)
- Identifies relationships between entities
- Supports multiple languages with cross-language entity linking

#### Graph Visualization

The knowledge graph offers two rendering modes:

| Mode | Technology | Best For |
|------|------------|----------|
| 2D | Canvas | Quick overview, simple graphs |
| 3D (WebGL) | Three.js + react-force-graph-3d | Large graphs (1000+ nodes), immersive exploration |

**Switching Views:**
- Use the **2D/3D toggle** in the graph header
- 3D mode provides smooth rotation, zoom, and node selection

**Controls (3D mode):**
- **Drag**: Rotate the graph
- **Scroll**: Zoom in/out
- **Click**: Select node for details

#### Batch Entity Extraction (Performance)

For large documents, the system uses batch extraction to process multiple chunks per LLM call, reducing processing time by 3-5x.

**How it works:**
1. Document is split into chunks
2. Chunks are batched (3 per LLM call by default)
3. Entities are merged and deduplicated
4. Relationships are stored in the graph database

**Configuration:**
```python
# In backend/services/knowledge_graph.py
await kg_service.extract_entities_batch(
    chunks=chunk_texts,
    batch_size=3,  # Chunks per LLM call
)
```

### Entity Types

| Type | Description | Example |
|------|-------------|---------|
| PERSON | People, individuals | "John Smith" |
| ORGANIZATION | Companies, institutions | "Anthropic" |
| LOCATION | Places, cities, countries | "San Francisco" |
| CONCEPT | Abstract ideas, methodologies | "Machine Learning" |
| EVENT | Occurrences, meetings | "Q4 Review" |
| PRODUCT | Products, services | "Claude AI" |
| TECHNOLOGY | Technologies, tools | "FastAPI" |
| DATE | Dates, time periods | "2024 Q1" |

### Graceful Fallbacks

The knowledge graph handles LLM unavailability gracefully:
- If Ollama is not running, entity extraction is skipped (not crashed)
- Cloud LLM providers can be used as fallback
- Existing entities remain accessible

---

## Contextual Chunking

Contextual chunking implements Anthropic's contextual retrieval approach, which achieves **49-67% reduction in failed retrievals** compared to traditional chunking.

### How It Works

Traditional chunking splits documents into pieces without context. When you search for information, the chunk might contain the answer but lack the context needed to understand it fully.

Contextual chunking solves this by:
1. **Taking each chunk** from a document
2. **Generating a brief context** (50-100 words) using an LLM
3. **Prepending the context** to the chunk before embedding

**Example:**

Without contextual chunking:
> "The company reported $12.5M in revenue for Q3."

With contextual chunking:
> "This excerpt is from Acme Corp's Q3 2024 earnings report, discussing financial performance. The company reported $12.5M in revenue for Q3."

### Enabling Contextual Chunking

Navigate to **Dashboard > Settings > RAG** and enable:
- **Contextual Chunking**: Toggle to enable
- **Context Provider**: Choose `ollama` (free) or `openai`
- **Context Model**: Model to generate contexts (e.g., `llama3.2`)

Or via environment variables:
```bash
CONTEXTUAL_CHUNKING_ENABLED=true
CONTEXT_GENERATION_PROVIDER=ollama
CONTEXT_GENERATION_MODEL=llama3.2
```

### Performance Considerations

| Aspect | Impact |
|--------|--------|
| **Indexing Time** | 2-5x longer (LLM call per chunk) |
| **Embedding Size** | ~20% larger (context added) |
| **Retrieval Quality** | 49-67% fewer failed retrievals |
| **Cost** | Free with Ollama, ~$0.001/chunk with OpenAI |

### When to Use

**Recommended for:**
- Technical documentation where context matters
- Legal/compliance documents with specific terminology
- Research papers with domain-specific concepts
- Any corpus where chunks lack standalone context

**May skip for:**
- Simple Q&A datasets
- Short documents where chunks have natural context
- Real-time indexing requirements

### Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.contextual_chunking_enabled` | Enable contextual enhancement | `false` |
| `rag.context_generation_provider` | LLM provider | `ollama` |
| `rag.context_generation_model` | Model for context generation | `llama3.2` |

---

## Settings Presets

Quick configuration presets are available in **Dashboard > Settings**:

| Preset | Description |
|--------|-------------|
| **Speed** | Fast responses, reduced accuracy (fewer RAG results, no reranking) |
| **Quality** | Best accuracy, slower (more RAG results, reranking enabled) |
| **Balanced** | Default settings |
| **Offline** | Local models only (Ollama required) |

### Applying Presets

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/settings/apply-preset/speed

# Via UI
Dashboard > Settings > Presets tab > Click preset button
```

---

## Advanced RAG Features (Phase 66)

### User Personalization

The system learns your preferences over time based on feedback and usage patterns.

**What It Learns:**
- Response length preference (concise vs detailed)
- Format preference (bullet points vs prose)
- Expertise level (beginner vs expert)
- Citation style preferences

**How It Works:**
1. Submit feedback on responses (thumbs up/down)
2. System analyzes feedback patterns
3. Future responses are tailored to your preferences

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.user_personalization_enabled` | Enable preference learning | `true` |

### Adaptive RAG Routing

Automatically routes queries to the optimal retrieval strategy based on query complexity.

**Routing Strategies:**
| Strategy | When Used | Description |
|----------|-----------|-------------|
| DIRECT | Simple factual queries | Fast single-shot retrieval |
| HYBRID | Standard queries | Vector + keyword search |
| TWO_STAGE | Complex queries | Retrieval + reranking |
| AGENTIC | Multi-step queries | Query decomposition + ReAct loop |
| GRAPH_ENHANCED | Entity-rich queries | Knowledge graph traversal |

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.adaptive_routing_enabled` | Enable query-dependent routing | `true` |

### RAG-Fusion

Generates multiple query variations and merges results using Reciprocal Rank Fusion.

**How It Works:**
1. Original query is paraphrased into 3-5 variations
2. Each variation runs through retrieval
3. Results are merged using RRF scoring
4. Improves recall by 20-40%

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.fusion_enabled` | Enable RAG-Fusion | `true` |
| `rag.fusion_query_count` | Number of query variations | `3` |

### LazyGraphRAG

Query-time community summarization for knowledge graphs. Achieves 99% cost reduction compared to eager summarization.

**How It Works:**
1. Communities are detected in the knowledge graph
2. Summaries are generated on-demand when queried
3. Summaries are cached for subsequent queries

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.lazy_graphrag_enabled` | Enable lazy community summarization | `true` |

### Dependency-Based Entity Extraction

Fast entity extraction using spaCy dependency parsing. Achieves 94% of LLM performance at 80% lower cost.

**When It's Used:**
- High-volume entity extraction
- Real-time processing
- Cost-sensitive deployments

**Automatic LLM Fallback:**
For complex technical text, the system automatically falls back to LLM extraction.

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `kg.dependency_extraction_enabled` | Enable fast dependency parsing | `true` |
| `kg.complexity_threshold` | Text complexity threshold for LLM fallback | `0.7` |

### RAG Evaluation (RAGAS Metrics)

Comprehensive evaluation of RAG quality using industry-standard RAGAS metrics.

**Metrics:**
| Metric | Description |
|--------|-------------|
| Context Relevance | How relevant retrieved documents are to the query |
| Faithfulness | How grounded the response is in the retrieved context |
| Answer Relevance | How well the response answers the question |

**API Endpoints:**
```bash
# Evaluate a single query
POST /api/v1/evaluation/query
{
  "query": "What is machine learning?",
  "response": "Machine learning is...",
  "context": ["Document 1 content...", "Document 2 content..."]
}

# Batch evaluation
POST /api/v1/evaluation/batch
{
  "evaluations": [...]
}

# Get evaluation history
GET /api/v1/evaluation/history
```

---

## TTS Providers

### Chatterbox TTS (NEW)

Ultra-realistic open-source TTS from Resemble AI with emotional expressiveness.

**Features:**
- Emotional speech synthesis
- Voice cloning support
- Zero API cost (open-source)
- Configurable exaggeration and CFG weight

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `tts.chatterbox_enabled` | Enable Chatterbox TTS | `true` |
| `tts.chatterbox_exaggeration` | Emotional exaggeration (0-1) | `0.5` |
| `tts.chatterbox_cfg_weight` | CFG weight for generation | `0.5` |

### CosyVoice2 (NEW)

Alibaba's open-source streaming TTS with 150ms latency.

**Features:**
- Ultra-low latency (150ms)
- Streaming audio generation
- Zero API cost (open-source)
- Multi-language support

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `tts.cosyvoice_enabled` | Enable CosyVoice2 | `true` |

### TTS Provider Comparison

| Provider | Latency | Quality | Cost | Best For |
|----------|---------|---------|------|----------|
| OpenAI TTS | 500-1000ms | High | $0.015/1K chars | Production |
| ElevenLabs | 300-600ms | Highest | $0.03/1K chars | Premium audio |
| Chatterbox | 800-1500ms | High | Free | Emotional speech |
| CosyVoice2 | 150ms | Good | Free | Real-time streaming |
| Fish Speech | 200ms | Good | Free | Multilingual |

---

## Phase 78-83 Features (January 2026)

### AttentionRAG Compression

6.3x more efficient context compression than LLMLingua, using attention scores to identify the most relevant context segments.

**How It Works:**
1. Forward pass with query + context through attention model
2. Extract attention scores from query tokens to context tokens
3. Aggregate attention to sentence/chunk level
4. Keep top-k most attended segments

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.attention_rag_enabled` | Enable attention-based compression | `false` |
| `rag.attention_rag_mode` | Compression aggressiveness (light/moderate/aggressive) | `moderate` |

### Graph-O1 Reasoning

Efficient beam search reasoning over the knowledge graph, providing 3-5x faster reasoning than naive traversal while maintaining 95%+ accuracy.

**How It Works:**
1. Parse query to identify entity and relationship targets
2. Expand search beam across knowledge graph hops
3. Score and prune paths based on confidence
4. Return evidence-backed reasoning chain

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.graph_o1_enabled` | Enable Graph-O1 beam search reasoning | `false` |
| `rag.graph_o1_beam_width` | Number of parallel paths to explore | `5` |
| `rag.graph_o1_confidence_threshold` | Minimum confidence to continue path | `0.7` |

### Anthropic Prompt Caching

Automatically caches system prompts when using Claude models, saving 50-60% on repeated API calls. No configuration needed — enabled automatically for Anthropic providers.

### OpenAI Structured Outputs

`LLMFactory.get_structured_model()` creates models configured with JSON schema validation for reliable structured extraction. Uses OpenAI's native `response_format` with `json_schema` for strict output compliance.

### Stability Improvements (Phase 79)

- **Session LLM cache TTL**: 1-hour TTL + 200 max entries prevents unbounded memory growth
- **FAISS rebuild lock**: Prevents concurrent index rebuilds in GenerativeCache
- **SHA-256 cache keys**: Replaced MD5 with SHA-256 for cache key generation (collision resistance at scale)
- **Settings cache TTL**: All settings caches auto-invalidate after 5 minutes (admin changes take effect without restart)

### Import Smoke Test (Phase 83)

CI-friendly test that imports every module in `backend/services/` and `backend/api/routes/` to catch broken imports at build time:

```bash
pytest backend/tests/test_imports.py -v
```

### Security Hardening (Phase 84-85)

**Sandbox Execution Security (Phase 84):**
- Safe wrapper classes (`SafeRegex`, `SafeJson`) replace raw module access in sandboxed code execution
- AST validation blocks dunder attribute chains (`__class__.__bases__.__subclasses__`)
- Code execution timeout via `ThreadPoolExecutor` prevents infinite loops
- Blocks `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `dir`, `breakpoint` in sandboxed code

**Authentication & Input Validation (Phase 85):**
- Default `SECRET_KEY` rejected at startup in production/staging environments
- Query length limits on chat endpoints (100K message, 500K content)
- Collection name regex validation (`^[a-zA-Z0-9_\-\s\.]+$`)
- Auth endpoint rate limiting (10 logins/min, 5 registrations/hour per IP)

See [SECURITY.md](SECURITY.md) for comprehensive security documentation.

### LLM Provider Rate Limiting (Phase 86)

Per-provider token bucket rate limiting prevents API cost overruns and cascading failures:
- Configurable RPM limits per provider (OpenAI: 500, Anthropic: 60, etc.)
- Backpressure mechanism (async sleep) when limits are approached
- Automatic tracking of request rates across the application

### Dependency Updates (Phase 87)

- **Google GenAI SDK migration**: Replaced deprecated `google-generativeai` with unified `google-genai` SDK
- **LangChain audit**: All imports verified correct for LangChain 0.3.x
- **Dependency pinning**: Upper bounds added to critical deps (FastAPI, Pydantic, SQLAlchemy, Ray)
- **Ray 2.10+**: Updated for better async support and memory management

### Error Handling & Observability (Phase 88)

- **Narrowed exception handling**: Top 22 broad `except Exception` blocks in critical paths (rag.py, redis_client.py, llm.py) replaced with specific exception types
- **Structured error context**: All error logs now include `error_type` for easier debugging and monitoring
- **Redis-specific exceptions**: Cache operations catch `ConnectionError`, `TimeoutError`, `JSONDecodeError` separately for better failure diagnosis

### Cross-Section RAG Integration (Phase 94)

RAG features are now integrated across all major sections, not just chat:

**Document Generation:**
- Knowledge graph integration fixed (`get_graph_rag_context` wrapper method added)
- Graph-enhanced source retrieval for section content
- Entity context used for better source matching

**Web Crawler:**
- Automatic entity extraction from crawled pages (stored in knowledge graph)
- Background KG enrichment runs asynchronously to not block crawl responses
- Controlled by `rag.knowledge_graph_enabled` setting

**Text-to-SQL / Database Queries:**
- Knowledge graph entity lookup for WHERE clause exact matching
- Entity aliases resolved for better query accuracy
- Proper noun context injected into SQL generation prompt

**Knowledge Graph Queries:**
- Query expansion before entity lookup (finds 20-30% more relevant entities)
- LRU cache (128 entries) for graph_search results avoids redundant traversals
- Expanded queries merged with deduplication, capped at 10 entities

**Admin Settings UI:**
- 13 new RAG feature toggles (Query Intelligence, Advanced Compression, Storage Optimization)
- 38-step Query Pipeline Visualization with live status indicators
- Audio Settings tab (TTS providers, Chatterbox/CosyVoice/Fish Speech)
- Web Scraper Settings tab (Crawl4AI, anti-detection, content extraction)
- Knowledge Graph Ingestion tab (dependency extraction, spaCy model)

**Sandbox Security Hardening:**
- Raw `datetime` module replaced with `_SafeDatetime` wrapper in workflow sandbox
- Prevents `datetime.__class__.__bases__[0].__subclasses__()` sandbox escape
- Only safe static methods exposed (now, utcnow, fromisoformat, strptime, today)

**DSPy Inference Integration:**
- Compiled DSPy modules (instructions + few-shot demos) injected into RAG prompt
- Loaded from DSPyOptimizationJob table with deployed/completed status
- Controlled by `rag.dspy_inference_enabled` setting

**Default Settings Fixed:**
- `rag.adaptive_routing_enabled` → True (was silently off due to DB override)
- `rag.self_rag_enabled` → True (hallucination detection)
- `rag.sufficiency_checker_enabled` → True (quality gate)
- `processing.fast_chunking_enabled` → True (33x faster chunking)
- Adaptive routing fallback added when router is disabled

---

## Phase 95: UI Overhaul + Competitor-Inspired Features

### UI Component Consistency (95A)
- Replaced all native HTML `<input type="checkbox">` with shadcn/ui `<Switch>` across 12+ settings tabs
- Replaced all native `<select>` with shadcn/ui `<Select>` across settings and document pages
- Consistent component library usage throughout the application

### Admin Settings Vertical Navigation (95B)
- Replaced horizontal tab overflow with vertical sidebar navigation
- Tabs organized into categories: System, AI & Models, Processing, Intelligence, Integrations, Security
- Mobile-responsive dropdown fallback
- URL-persisted active tab state

### Dark Mode Polish (95C)
- Added `dark:` variants to all hardcoded color classes across dashboard, documents, privacy, and settings pages
- Consistent color treatment: light `-50`/`-100` backgrounds map to `dark:-950/30`/`-900/30`
- Text colors: `-600`/`-700` map to `dark:-400`/`-300` for readability

### Inline Source Citations (95D)
- Numbered `[1]`, `[2]`, `[3]` superscript citation badges inline in AI responses
- Hover popover showing source document name, page, similarity score, and snippet
- Click to navigate to source in the sources panel
- Backend system prompt updated to instruct numbered citation format

### Thinking/Reasoning Transparency (95E)
- Collapsible "Thinking" block above AI responses showing RAG pipeline steps
- Per-step timing display: searching, finding chunks, reranking, generating
- Animated streaming state with bouncing dots during processing
- Total processing time summary in collapsed header

### Chat Canvas / Artifacts Panel (95N)
- Side-by-side Sheet panel for viewing and editing AI-generated code and content
- View mode with line numbers for code, edit mode with monospace textarea
- Copy to clipboard, download as file, and AI edit request input
- Auto-detects code blocks and long responses for "Open in Canvas" button

### Document Preview Split Pane (95G)
- Right-side Sheet panel opens on document click instead of page navigation
- Tabbed view: Preview (using DocumentPreview component) and Metadata
- Quick actions: Full View, Download, Edit Tags
- Displays filename, type, size, collection, tags, creation date, chunk count

### Floating Batch Action Bar (95H)
- Fixed-position floating bar at bottom-center when documents are selected
- Backdrop-blur glass effect with slide-up animation
- Actions: Auto-tag, Enhance, Move, Download, Delete, Clear selection
- Replaces inline bulk actions for cleaner document management UX

### Prompt Library (95M)
- Full CRUD page for managing reusable prompt templates
- Card grid with search, category filter, use count badges
- Template variables with `{{variable}}` syntax and runtime substitution
- Public/private sharing, system templates, duplicate functionality
- Integrated with existing backend PromptTemplate model and API

### Agent Usage Insights (95O)
- Per-agent analytics component showing usage metrics
- 4 metric cards: total queries, avg response time, satisfaction %, active users
- Time period selector (7d, 30d, all time)
- Recent activity list with query summaries and feedback indicators

### Custom Instructions / System Prompts (95P)
- Admin settings tab for organization-wide default system prompts
- Variables support: `{{user_name}}`, `{{user_role}}`, `{{date}}`
- Append mode configuration: prepend, append, or replace RAG prompt
- Default response language selection (11 languages)
- Live preview of final merged system prompt

### Projects / Chat Folders (95Q)
- Client-side chat organization with project folders
- Collapsible folder tree with session count badges
- Drag sessions between projects via dropdown menu
- Ungrouped section for unassigned chats
- LocalStorage persistence, search filtering

### BYOK - Bring Your Own Key (95R)
- Per-provider API key management panel
- Supported providers: OpenAI, Anthropic, Google AI, Mistral, Groq, Together
- Key validation testing, status badges (Active/Invalid/Untested)
- Password-style input with show/hide toggle
- Encrypted localStorage storage with master BYOK toggle

### Skills Marketplace (95S)

AI-powered skills for document processing, extending the platform with pluggable capabilities.

**Features:**
- **6 Built-in Skills**: Document Summarizer, Fact Checker, Translator, Sentiment Analyzer, Entity Extractor, Document Comparison
- **Dynamic LLM Selection**: Only shows models from providers configured in Settings (OpenAI, Anthropic, Google, Ollama)
- **Multiple Input Types**: Text, Document, File, JSON, Image inputs with appropriate UI controls
- **Create Custom Skills**: Define custom skills with system prompts and input/output schemas
- **Import Skills**: Import skill definitions from JSON files or URLs
- **Execution Tracking**: Usage count, average latency, status indicators

**Skill Categories:**
| Category | Description |
|----------|-------------|
| Analysis | Extract insights from documents |
| Generation | Create new content |
| Extraction | Pull specific data |
| Transformation | Convert formats |
| Validation | Verify content |
| Integration | Connect with external services |

**Accessing Skills:**
Navigate to **Dashboard > Skills** to browse, create, and execute skills.

### Mood Board Creator (95T)

AI-powered visual inspiration board generator for design and branding projects.

**Location:** Available in the **Create** section alongside other document formats (DOCX, PPTX, PDF, etc.)

**Features:**
- **Color Palette Builder**: Add up to 8 colors with color picker and hex codes
- **Style Keywords**: Tag-based input for mood descriptors (minimalist, bold, playful, etc.)
- **Reference Images**: Upload multiple reference images for visual inspiration
- **AI-Generated Suggestions**: Typography recommendations, complementary colors, mood keywords, and inspiration notes

**How to Use:**
1. Navigate to **Dashboard > Create**
2. Select **Mood Board** from the format options
3. Enter a board name and optional description
4. Add colors to your palette (click to change, up to 8)
5. Add style keywords to describe your aesthetic
6. Optionally upload reference images
7. Click **Generate AI Mood Board** to get suggestions
8. Export or save your completed mood board

### Deep Research Mode (95U)

Multi-round verification research with cross-model fact-checking for high-confidence answers.

**Features:**
- **Multi-LLM Verification**: Select multiple models for cross-checking facts
- **Dynamic Provider Filtering**: Only shows models from configured providers in Settings
- **Configurable Rounds**: 1-5 verification rounds for varying thoroughness
- **Verification Chain**: Visual representation of each verification step with verdict (supported/contradicted/uncertain)
- **Confidence Scoring**: Aggregate confidence from all verification steps
- **Conflict Detection**: Alerts when models find contradictory evidence

**Settings Popover:**
- Select/deselect individual models by provider
- Reset to default model selection
- Shows count of selected models

**Verdicts:**
| Verdict | Meaning | Icon |
|---------|---------|------|
| Supported | Evidence confirms the claim | ✓ Green |
| Contradicted | Evidence refutes the claim | ✗ Red |
| Uncertain | Insufficient or conflicting evidence | ⚠ Yellow |

### Provider-Aware LLM Selection (Global Feature)

All features with LLM selection now filter available models based on Settings configuration.

**How It Works:**
1. Admin configures LLM providers in **Dashboard > Settings > Providers**
2. Each provider has an `is_active` flag
3. Features (Skills, Deep Research, etc.) fetch the provider list via `useLLMProviders` hook
4. Model dropdowns only show models from active providers
5. If no providers are configured, all models are shown as fallback

**Affected Features:**
- Skills Marketplace (skill execution)
- Deep Research Mode
- Agent Configuration

### Backend: Hallucination + Confidence Scoring (95J)
- `_compute_hallucination_score()`: Reranker-based grounding check (0=grounded, 1=hallucinated)
- `_compute_confidence_score()`: Multi-signal confidence from source count, rerank scores, retrieval density
- Scores injected into RAG response metadata

### Backend: Content Freshness Scoring (95K)
- Configurable freshness boost for recently updated documents (default 1.05x within 30 days)
- Configurable staleness penalty for old documents (default 0.95x after 180 days)
- Admin settings: `rag.content_freshness_enabled`, decay days, boost/penalty factors

### Backend: Conversation-Aware Retrieval (95L)
- Enriches retrieval query with last 3 user messages from conversation history
- Improves embedding quality for follow-up questions
- Original question preserved for LLM prompt generation

---

## Phase 97: Performance & Security Optimizations (January 2026)

### Smart Model Routing

Routes queries to cost-optimal LLM models based on query complexity, achieving 40-70% cost reduction.

**How It Works:**
1. Query classifier analyzes complexity (factual, analytical, multi-hop, creative)
2. Simple queries → cheaper/faster model (GPT-4o-mini, Claude Haiku)
3. Complex queries → premium model (GPT-4o, Claude Sonnet)
4. Moderate queries → default configured model

**Query Tier Classification:**
| Signal | Simple | Moderate | Complex |
|--------|--------|----------|---------|
| Intent | factual, keyword, definition | standard | analytical, multi_hop, creative |
| Context length | < 2K tokens | 2K-50K | > 50K |
| Document count | ≤ 3 | 4-15 | > 15 |

**Configuration:**
```yaml
rag.smart_model_routing_enabled: true
rag.smart_routing_simple_model: "openai/gpt-4o-mini"
rag.smart_routing_complex_model: "openai/gpt-4o"
```

### Embedding Inversion Defense (OWASP LLM08:2025)

Protects stored embeddings from text reconstruction attacks using three defense techniques.

**Why It Matters:**
Research has shown that embeddings can be inverted to recover original text, posing privacy risks for sensitive documents. This implementation follows OWASP LLM08:2025 guidance.

**Defense Techniques:**
1. **Noise Injection**: Adds calibrated Gaussian noise (default σ=0.01) that degrades inversion fidelity while preserving ranking
2. **Dimension Shuffle**: Secret, deterministic permutation prevents adversaries from mapping dimensions to linguistic features
3. **Norm Clipping**: Caps L2 norm to bound information leakage per embedding

**Symmetric Application:**
The same `protect()` transformation is applied at both index time and query time, so cosine similarity between defended vectors is preserved.

**Configuration:**
```yaml
security.embedding_defense_enabled: true
security.defense_noise_scale: 0.01  # Gaussian noise stddev
security.defense_clip_norm: 1.0     # L2 norm cap
```

**Environment:**
```bash
EMBEDDING_DEFENSE_SECRET=your-secret-key  # Stable permutations across restarts
```

### Matryoshka Adaptive Retrieval

Two-stage retrieval using Matryoshka representation learning for 5-14x faster search.

**Background:**
Matryoshka Representation Learning (NeurIPS 2022) trains embeddings where the first N dimensions form valid lower-dimensional representations. Modern models like OpenAI text-embedding-3-* support this natively.

**How It Works:**
1. **Stage 1 (Fast Pass)**: Truncate embeddings to 128 dimensions, search for broad shortlist
2. **Stage 2 (Precise Rerank)**: Recompute full-dimensional similarity on shortlist

**Performance:**
- 128-dim search is ~12x faster than 1536-dim search
- <2% recall loss when shortlist factor ≥ 5x
- Works with any Matryoshka-trained embedding model

**Configuration:**
```yaml
rag.matryoshka_retrieval_enabled: true
rag.matryoshka_shortlist_factor: 5   # Retrieve 5x more candidates in stage 1
rag.matryoshka_fast_dims: 128        # Fast pass dimension count
```

### Speculative RAG (Google Research ICLR 2025)

Generates draft responses from document subsets in parallel, then selects the best via verification.

**Research Reference:** [Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/abs/2407.08223)

**How It Works:**
1. Split retrieved documents into N subsets (default 3)
2. Generate draft answers in parallel using smaller/faster drafter model
3. Verifier model evaluates and selects the best draft
4. Return selected answer with higher confidence

**Benefits:**
- 15-50% latency reduction (parallel drafting)
- 5-13% accuracy improvement (best-of-N selection)
- Better utilization of document subsets

**Configuration:**
```yaml
rag.speculative_enabled: true
rag.speculative_num_drafts: 3  # Number of parallel drafts
```

### OpenTelemetry RAG Tracing

Distributed tracing for RAG pipeline operations with per-stage latency tracking.

**Traced Operations:**
| Span | Attributes |
|------|------------|
| `rag.query` | query length, session ID, user ID, search type |
| `rag.retrieval` | top_k, search type, collection, result count |
| `rag.reranking` | doc count, model, top_n |
| `rag.generation` | model, doc count, prompt tokens, streaming |

**Integration:**
```python
from backend.services.otel_tracing import get_rag_tracer

tracer = get_rag_tracer()
with tracer.start_query_span(query="What is X?", user_id="123") as span:
    results = await rag_service.query(...)
    span.set_attribute("rag.result_count", len(results))
```

**Configuration:**
```yaml
observability.tracing_enabled: true
observability.tracing_sample_rate: 0.1  # 10% sampling
observability.otlp_endpoint: "http://jaeger:4317"
```

**Exporters:**
- OTLP/gRPC (Jaeger, Tempo, Zipkin)
- Console (development)
- Graceful degradation when OpenTelemetry not installed

---

## Recent Enhancements (February 2026)

### Private Documents

Documents can now be marked as **private** during upload, restricting visibility to only the document owner and superadmins.

**How to Use:**
1. During upload, toggle "Private Document" in the Processing Options
2. Private documents won't appear in other users' searches or document lists
3. Knowledge graph entities from private documents are filtered at query time

**API Usage:**
```bash
curl -X POST "/api/upload" \
  -F "file=@sensitive.pdf" \
  -F "is_private=true"
```

See [Security Documentation](SECURITY.md#private-documents) for full details.

---

### Memory Manager (OpenClaw-Inspired)

Centralized memory management with dirty tracking and debounced sync, inspired by OpenClaw/Moltbot patterns.

**Features:**
- **Dirty tracking** with configurable debounce (default 1.5 seconds)
- **Atomic reindexing** with safe temp file swap
- **Automatic sync triggers** on settings/model changes
- **LRU cache pruning** for memory control

**Usage:**
```python
from backend.services.memory_manager import get_memory_manager

manager = get_memory_manager()

# Mark data as dirty (triggers debounced sync)
await manager.mark_dirty("embeddings")

# Force immediate sync
await manager.sync_now()

# Atomic reindexing with validation
await manager.atomic_reindex(
    source_path="path/to/index",
    build_func=build_new_index,
    validate_func=validate_index,
)
```

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `system.memory_cleanup_interval_minutes` | Memory cleanup frequency | `10` |
| `system.model_idle_timeout_minutes` | Unload idle models after | `15` |

---

### Adaptive Chunking (Moltbot-Inspired)

Progressive fallback chunking that targets ~400 tokens per chunk with automatic size adjustment.

**Features:**
- Target token counts with progressive fallback
- Sentence-aware splitting for natural breaks
- Long line splitting to keep embeddings under limits
- Token counting via tiktoken (or character estimation fallback)

**How It Works:**
1. Start with target chunk size (~400 tokens)
2. If chunks are too large, reduce by 20% and retry
3. After max retries, fall back to sentence-based splitting
4. Always preserve sentence boundaries where possible

**Usage:**
```python
from backend.services.adaptive_chunking import AdaptiveChunker

chunker = AdaptiveChunker(
    target_tokens=400,
    overlap_tokens=80,
    max_retries=3,
)
chunks = chunker.chunk(text)

# Get statistics
stats = chunker.get_stats(chunks)
# {'count': 15, 'avg_tokens': 387, 'max_tokens': 412, ...}
```

**Configuration:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.adaptive_chunking_enabled` | Enable adaptive chunking | `true` |
| `rag.adaptive_chunking_target_tokens` | Target tokens per chunk | `400` |
| `rag.adaptive_chunking_overlap_tokens` | Overlap between chunks | `80` |

---

### Auto-Tagging

LLM-powered automatic document tagging that analyzes document content and generates relevant tags during upload.

**Features:**
- **Content-Aware Tagging**: Analyzes document name and content sample to generate relevant tags
- **Existing Collection Suggestions**: Prefers matching existing collections when relevant
- **Configurable Tag Count**: Generate 1-5 tags per document (default: 5)
- **Primary Category First**: First tag represents the primary collection/category

**How to Use:**
1. During upload, enable "Auto-Generate Tags" in Processing Options
2. Tags are generated after document processing completes
3. Generated tags are merged with any manually specified tags

**UI Integration:**
- Toggle available in Upload page under "Processing Options"
- Checkbox: "Auto-Generate Tags" with description "Use AI to automatically generate relevant tags based on document content"

**API Usage:**
```bash
curl -X POST "/api/v1/upload/single" \
  -F "file=@document.pdf" \
  -F "auto_generate_tags=true"
```

**How It Works:**
1. After document is chunked and processed, first ~2000 characters are extracted
2. LLM analyzes document name + content sample
3. Tags are generated focusing on: topic, industry, document type, project/client name
4. Tags are merged with any existing/manual tags on the document

**Configuration:**
The LLM provider and model for auto-tagging is configured via Admin UI:
- Navigate to **Admin > Settings > LLM Configuration**
- Configure the "auto_tagging" operation under Operation-Level Config

**Example Generated Tags:**
- For a project management guide: `["Project Management", "Software Development", "Guide", "Best Practices"]`
- For a marketing presentation: `["Marketing", "FedEx Campaign", "Presentation", "Q4 2023"]`

---

### AI Image Analysis (Multimodal RAG)

Vision-based image captioning that makes images in your documents searchable by generating AI descriptions.

**Features:**
- **Local Vision Models**: Uses Ollama llava by default (free, private, no cloud costs)
- **Automatic Detection**: Extracts images from PDFs and documents during processing
- **Intelligent Caching**: Same images across documents are only analyzed once (deduplication)
- **Searchable Captions**: Generated descriptions are indexed and searchable via chat
- **Provider Flexibility**: Supports Ollama (recommended), OpenAI GPT-4V, or Anthropic Claude Vision

**How to Use:**
1. Enable "AI Image Analysis" during upload (enabled by default)
2. Images are automatically extracted during document processing
3. Vision model generates descriptions for each significant image
4. Descriptions are embedded and stored with document chunks

**Single File Upload:**
```bash
curl -X POST "/api/v1/upload/single" \
  -F "file=@document.pdf" \
  -F "enable_image_analysis=true"
```

**Batch Upload (Multiple Files):**
```bash
curl -X POST "/api/v1/upload/batch" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "enable_image_analysis=true" \
  -F "auto_generate_tags=true"
```

**Vision Provider Configuration:**

Configure the vision model via **Admin > Settings > RAG Configuration**:

| Setting | Description | Default |
|---------|-------------|---------|
| `rag.vision_provider` | Vision model provider | `auto` |
| `rag.ollama_vision_model` | Ollama vision model name | `llava` |

**Provider Options:**
- `ollama` - **Recommended**: Uses Ollama locally (free, private). When set, system uses Ollama ONLY with no fallback to cloud APIs
- `auto` - Automatically selects best available provider
- `openai` - Uses GPT-4V (requires valid API key)
- `anthropic` - Uses Claude Vision (requires valid API key)

**Recommended Ollama Vision Models:**
| Model | Best For |
|-------|----------|
| `llava` | General image understanding (default) |
| `qwen2.5vl` | Document OCR, tables (highest accuracy) |
| `llava-phi3` | Fast inference on resource-constrained systems |

**Verifying Image Analysis:**
Check Celery worker logs for successful image processing:
```
Vision model selection         db_model=llava db_provider=ollama
Using Ollama vision model (from DB setting): llava at http://localhost:11434
Image processing complete      images_analyzed=5 images_found=8
```

**Troubleshooting:**
- If you see `401 Incorrect API key` errors, ensure `rag.vision_provider` is set to `ollama` in Admin Settings
- For "Image - captioning error" messages, pull the vision model: `ollama pull llava`
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#image-captioning-using-openai-instead-of-ollama) for detailed solutions

---

### Duplicate Detection Improvements

Enhanced duplicate file detection that properly handles deleted documents.

**Features:**
- **Content Hash Detection**: SHA-256 hash of file content for accurate duplicate detection
- **Deleted Document Awareness**: Re-uploading a previously deleted file no longer flags as duplicate
- **Upload Job History**: Maintains upload history while respecting document lifecycle

**How It Works:**
1. On upload, file content hash (SHA-256) is computed
2. System checks both UploadJob table and Document table for existing hash
3. If hash exists in UploadJob, verifies the referenced document still exists
4. If document was deleted, allows re-upload as a new document
5. If document exists, returns duplicate status with existing document info

**API Response for Duplicates:**
```json
{
  "status": "duplicate",
  "file_id": "existing-doc-uuid",
  "filename": "original-filename.pdf",
  "document_id": "existing-doc-uuid"
}
```

**Configuration:**
```bash
# Enable/disable duplicate detection during upload
curl -X POST "/api/v1/upload/single" \
  -F "file=@document.pdf" \
  -F "detect_duplicates=true"  # Default: true
```

---

## Performance Optimizations (February 2026)

AIDocumentIndexer includes comprehensive performance optimizations that are automatically initialized at server startup. All optimizations have graceful fallbacks if dependencies are missing.

### Overview

| Optimization | Speedup | Fallback |
|-------------|---------|----------|
| Cython Extensions | 10-100x | NumPy-based pure Python |
| GPU Acceleration | 5-20x | CPU-based NumPy |
| MinHash Deduplication | O(n) vs O(n²) | Exact Jaccard similarity |
| orjson Serialization | 2-3x | Standard json module |
| GZip Compression | 60-70% size reduction | Uncompressed responses |

### Cython Extensions

High-performance similarity computations using Cython with runtime compilation.

**Features:**
- **Runtime Compilation**: Cython extensions are compiled automatically on first server start (if Cython is available)
- **Thread-Safe Initialization**: Only one thread performs initialization
- **Graceful Fallback**: If Cython/C compiler unavailable, falls back to NumPy implementations
- **Zero Hard Dependencies**: Works without Cython installed (just slower)

**Accelerated Functions:**
| Function | Cython Speedup | Use Case |
|----------|---------------|----------|
| `cosine_similarity_batch` | 10-50x | Query-document similarity |
| `cosine_similarity_matrix` | 20-100x | Document-document similarity |
| `mmr_selection` | 10-50x | Maximal Marginal Relevance |
| `hamming_distance_batch` | 5-50x | Binary quantization |
| `weighted_mean_pooling` | 20-40x | Embedding pooling |

**Checking Status:**
```python
from backend.services.cython_extensions import get_optimization_status, is_using_fallback

status = get_optimization_status()
print(f"Using Cython: {status['using_cython']}")
print(f"Speedup: {status['speedup_factor']}")

if is_using_fallback():
    print("Using NumPy fallbacks (install Cython for 10-100x speedup)")
```

**Installing Cython for Maximum Performance:**
```bash
pip install cython numpy

# Extensions compile automatically on next server start
# Check logs for: "Cython extensions loaded successfully (10-100x faster)"
```

### GPU Acceleration

PyTorch-based GPU acceleration for similarity computations with automatic CPU fallback.

**Features:**
- **Automatic Device Detection**: Detects CUDA (NVIDIA) or MPS (Apple Silicon)
- **Mixed Precision**: FP16 operations for 2x throughput on supported GPUs
- **OOM Handling**: Automatically falls back to CPU on out-of-memory errors
- **Warmup Support**: Optional GPU warmup at startup for consistent latency

**Supported Devices:**
| Device | Detection | Notes |
|--------|-----------|-------|
| NVIDIA CUDA | `torch.cuda.is_available()` | Best performance |
| Apple MPS | `torch.backends.mps.is_available()` | macOS M1/M2/M3 |
| CPU | Always available | Fallback |

**Configuration (Environment Variables):**
```bash
# Enable/disable GPU initialization (default: true)
PERF_INIT_GPU=true

# Prefer GPU over CPU when available (default: true)
PERF_GPU_PREFER=true

# Use FP16 mixed precision for 2x throughput (default: true)
PERF_MIXED_PRECISION=true

# Run GPU warmup at startup for consistent latency (default: false)
PERF_WARMUP_GPU=false
```

**Checking GPU Status:**
```python
from backend.services.gpu_acceleration import check_gpu_availability, get_similarity_accelerator

availability = check_gpu_availability()
print(f"CUDA available: {availability['cuda_available']}")
print(f"MPS available: {availability['mps_available']}")

accelerator = get_similarity_accelerator()
status = accelerator.get_status()
print(f"Active device: {status['device']}")
print(f"Mixed precision: {status['mixed_precision']}")
```

### MinHash LSH Deduplication

O(n) approximate deduplication using MinHash signatures and Locality-Sensitive Hashing.

**Features:**
- **O(n) Complexity**: Instead of O(n²) pairwise comparison
- **Configurable Accuracy**: Trade accuracy for speed with permutation count
- **Memory Efficient**: Stores compact signatures instead of full documents
- **Graceful Fallback**: Falls back to exact Jaccard if datasketch unavailable

**Configuration (Environment Variables):**
```bash
# Enable MinHash initialization (default: true)
PERF_INIT_MINHASH=true

# Number of permutations - more = accurate, slower (default: 128)
PERF_MINHASH_PERMS=128

# Similarity threshold for duplicates (default: 0.8)
PERF_MINHASH_THRESHOLD=0.8
```

**Usage:**
```python
from backend.services.minhash_dedup import get_minhash_deduplicator

dedup = get_minhash_deduplicator()

# Add documents
dedup.add_document("doc1", "Document text content...")
dedup.add_document("doc2", "Similar document content...")

# Check for duplicates
result = dedup.is_duplicate("New document to check", threshold=0.8)
if result.is_duplicate:
    print(f"Duplicate of: {result.similar_doc_id}")
    print(f"Similarity: {result.similarity}")

# Find all duplicate clusters
clusters = dedup.find_all_duplicates()
for cluster in clusters:
    print(f"Duplicate group: {cluster.member_ids}")
```

**Statistics:**
```python
stats = dedup.get_stats()
# {
#   "num_documents": 1000,
#   "using_minhash": true,
#   "method": "minhash_lsh",
#   "complexity": "O(n)",
#   "num_permutations": 128
# }
```

### Performance Initialization

All performance optimizations are initialized automatically during FastAPI startup.

**Startup Sequence:**
1. **Cython Extensions**: Compile (if needed) and load optimized functions
2. **GPU Acceleration**: Detect devices, initialize accelerator, optional warmup
3. **MinHash Deduplicator**: Initialize LSH index with configured parameters

**Environment Variables Summary:**
```bash
# Cython
PERF_COMPILE_CYTHON=true    # Compile Cython extensions

# GPU
PERF_INIT_GPU=true          # Initialize GPU accelerator
PERF_GPU_PREFER=true        # Prefer GPU over CPU
PERF_MIXED_PRECISION=true   # Use FP16 for 2x throughput
PERF_WARMUP_GPU=false       # Run warmup at startup

# MinHash
PERF_INIT_MINHASH=true      # Initialize MinHash deduplicator
PERF_MINHASH_PERMS=128      # Permutation count
PERF_MINHASH_THRESHOLD=0.8  # Similarity threshold
```

**Checking Overall Status:**
```python
from backend.services.performance_init import get_performance_status

status = get_performance_status()
# {
#   "cython": {"using_cython": true, "speedup_factor": "10-100x"},
#   "gpu": {"device": "cuda", "mixed_precision": true},
#   "minhash": {"using_minhash": true, "complexity": "O(n)"}
# }
```

---

## Cloud & Kubernetes Features (February 2026)

AIDocumentIndexer is designed for cloud-native deployments with built-in support for Kubernetes, AWS, and other cloud platforms.

### Health Endpoints

Three health endpoints for different monitoring scenarios:

| Endpoint | Purpose | Use Case |
|----------|---------|----------|
| `/health` | Comprehensive health check | General monitoring |
| `/health/live` | Kubernetes liveness probe | Pod restart decisions |
| `/health/ready` | Kubernetes readiness probe | Traffic routing |

**`/health` - Comprehensive Check:**
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "timestamp": "2026-02-02T10:30:00Z",
  "version": "1.0.0",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "celery": "healthy",
    "disk_space": "healthy"
  },
  "performance": {
    "cython_enabled": true,
    "gpu_enabled": true,
    "minhash_enabled": true
  }
}
```

**`/health/live` - Liveness Probe:**
```bash
curl http://localhost:8000/health/live
```
```json
{"status": "alive", "timestamp": "2026-02-02T10:30:00Z"}
```

**`/health/ready` - Readiness Probe:**
```bash
curl http://localhost:8000/health/ready
```
```json
{
  "status": "ready",
  "timestamp": "2026-02-02T10:30:00Z",
  "checks": {
    "database": true,
    "redis": true
  }
}
```

### Kubernetes Deployment

**Deployment YAML Example:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aidocindexer-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: aidocindexer:latest
        ports:
        - containerPort: 8000
        env:
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
        # Performance settings
        - name: PERF_INIT_GPU
          value: "true"
        - name: PERF_COMPILE_CYTHON
          value: "true"
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

### AWS Integration

**ALB Health Check Configuration:**
- **Path**: `/health/ready`
- **Interval**: 30 seconds
- **Timeout**: 5 seconds
- **Healthy threshold**: 2
- **Unhealthy threshold**: 3

**ECS Task Definition (excerpt):**
```json
{
  "healthCheck": {
    "command": ["CMD-SHELL", "curl -f http://localhost:8000/health/ready || exit 1"],
    "interval": 30,
    "timeout": 5,
    "retries": 3,
    "startPeriod": 60
  }
}
```

### Graceful Shutdown

The server handles SIGTERM and SIGINT signals for graceful shutdown, ensuring:
- In-flight requests complete
- Database connections are properly closed
- Background tasks are given time to finish

**Shutdown Sequence:**
1. Receive SIGTERM/SIGINT signal
2. Stop accepting new requests
3. Wait for in-flight requests (up to 30 seconds)
4. Close database connections
5. Exit cleanly

### Cloud Context Logging

Structured JSON logs automatically include cloud context when available:

```json
{
  "timestamp": "2026-02-02T10:30:00Z",
  "level": "info",
  "message": "Request processed",
  "pod_name": "aidocindexer-api-7d5f8b9c4-abc12",
  "namespace": "production",
  "node_name": "ip-10-0-1-42.ec2.internal",
  "container_name": "api",
  "cluster_name": "prod-cluster",
  "region": "us-west-2"
}
```

**Environment Variables for Cloud Context:**
```bash
POD_NAME=aidocindexer-api-xyz      # Kubernetes pod name
POD_NAMESPACE=production            # Kubernetes namespace
NODE_NAME=node-1                    # Kubernetes node
CONTAINER_NAME=api                  # Container name
AWS_REGION=us-west-2               # AWS region
CLUSTER_NAME=prod-cluster          # Cluster name
```

### Prometheus Metrics

Performance metrics are exposed at `/metrics` in Prometheus format:

```bash
curl http://localhost:8000/metrics
```

**Performance Metrics:**
```prometheus
# Cython optimization status
aidoc_cython_enabled 1

# GPU acceleration status
aidoc_gpu_enabled 1

# MinHash deduplication status
aidoc_minhash_enabled 1

# Memory usage
aidoc_memory_rss_bytes 2147483648
aidoc_memory_vms_bytes 4294967296

# Request metrics
aidoc_requests_total{method="GET",endpoint="/chat"} 1523
aidoc_request_duration_seconds{method="GET",endpoint="/chat"} 0.245
```

### Response Compression

GZip compression is automatically applied to responses larger than 500 bytes:

- **Threshold**: 500 bytes minimum
- **Compression Level**: Default (balanced speed/ratio)
- **Content Types**: All JSON and text responses
- **Size Reduction**: Typically 60-70%

**Verifying Compression:**
```bash
curl -H "Accept-Encoding: gzip" http://localhost:8000/api/v1/documents -v
# Response header: Content-Encoding: gzip
```

### Connection Pool Optimization

Database connection pool is optimized for cloud environments:

```python
# Automatic configuration
pool_recycle=3600    # Recycle connections after 1 hour (AWS RDS compatibility)
pool_timeout=30      # Wait up to 30 seconds for connection
pool_pre_ping=True   # Verify connections before use
```

---

## Future Roadmap (Research-Based)

The following improvements were identified through research of the current AI/RAG ecosystem. These are documented for future implementation and are NOT yet built.

### Critical (Enterprise Readiness)

1. **NeMo Guardrails** — NVIDIA's framework for prompt injection defense and content safety. Addresses OWASP #1 LLM vulnerability. Would add programmable guardrails for input/output filtering, topic control, and hallucination prevention at the framework level.

2. **Langfuse Integration** — Self-hosted LLM observability platform. Would provide tracing, evaluation, prompt management, and cost tracking across all LLM calls. Biggest operational gap for production deployments.

### High Priority

3. **pgvector 0.8.0+ Upgrade** — Latest pgvector adds iterative index scans, improved HNSW performance, and better filtered vector queries. Would improve search latency and accuracy for filtered retrieval scenarios.

4. **MCP Server/Client** — Expose RAG, search, and knowledge graph capabilities as Model Context Protocol tools. Would allow any MCP-compatible AI assistant to use AIDocumentIndexer as a tool.

5. **LangGraph 1.0 Migration** — LangGraph 1.0 is now GA with durable execution, human-in-the-loop, node caching, deferred nodes, and pre/post model hooks. Migrate current agent implementations for checkpointing, time-travel debugging, and fault tolerance.

6. **BGE-M3 Embeddings** — Single model supporting dense, sparse, and multi-vector retrieval simultaneously. Would simplify the embedding pipeline while maintaining hybrid search quality.

7. **Embedding Quantization (INT8/Binary)** — Apply INT8 and binary quantization to stored embeddings for up to 32x storage savings in pgvector. Complements existing binary quantization work.

8. **NodeRAG** — Novel graph-based RAG paradigm achieving 100% accuracy on MultiHop-RAG benchmark. Heterogeneous graph structure combining nodes, relationships, and communities for superior multi-hop reasoning.

9. **PageIndex (Vectorless RAG)** — Reasoning-based retrieval without vector search, achieving 98.7% accuracy on FinanceBench. Uses LLM reasoning to navigate document pages directly.

### Medium Priority

10. **DSPy Activation** — Activate the existing DSPy prompt optimization infrastructure (Phase 93) for production use with automated evaluation pipelines.

11. **Multimodal RAG via ColPali** — Leverage the existing ColPali engine dependency for end-to-end multimodal retrieval without separate OCR pipelines.

12. **Generative UI** — Stream React components from the backend for rich, interactive answer displays (charts, tables, interactive elements).

13. **Vercel AI SDK v6 Upgrade** — Upgrade to latest AI SDK for improved streaming, tool calling, and multi-modal support.

14. **Human-in-the-Loop for Agents** — Add approval workflows and human checkpoints for agent actions with side effects.

15. **Agent Evaluation & Simulation** — Automated testing framework for agent behaviors using simulated conversations and metrics.

16. **Enhanced Semantic Caching** — Advanced semantic cache strategies showing 68.8% cost reduction in benchmarks. Improve existing GPTCache integration with better similarity thresholds and cache invalidation.

17. **Contextual Memory for Agents** — Add persistent contextual memory to agentic RAG for maintaining long-term context across sessions and improving personalized retrieval.

---

## Cross-References

- **API Documentation**: [API.md](API.md) — Complete REST API reference
- **Configuration**: [CONFIGURATION.md](CONFIGURATION.md) — All settings with defaults
- **Security**: [SECURITY.md](SECURITY.md) — Security model and hardening
- **Architecture**: [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) — System architecture
- **Developer Guide**: [DEVELOPER_ONBOARDING.md](DEVELOPER_ONBOARDING.md) — Getting started as a developer
