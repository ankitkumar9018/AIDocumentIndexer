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
