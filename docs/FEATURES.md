# AIDocumentIndexer - New Features Guide

This guide covers the new features introduced in AIDocumentIndexer, including the Visual Workflow Builder, Audio Overviews, Connectors, and LLM Gateway.

---

## Table of Contents

1. [Visual Workflow Builder](#visual-workflow-builder)
2. [Audio Overviews](#audio-overviews)
3. [Connectors](#connectors)
4. [Web Scraper](#web-scraper)
5. [LLM Gateway](#llm-gateway)
6. [Knowledge Graph](#knowledge-graph)
7. [Settings Presets](#settings-presets)

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
