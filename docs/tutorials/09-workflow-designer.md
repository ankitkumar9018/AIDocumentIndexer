# Workflow Designer Tutorial

Build visual automation workflows with drag-and-drop nodes.

## Overview

The Workflow Designer lets you create complex automation pipelines by connecting nodes visually. Each node performs a specific action, and data flows between nodes automatically.

## Getting Started

1. Navigate to **Workflows** in the sidebar
2. Click **Create Workflow**
3. The designer opens with a canvas and node palette

## Available Node Types

### Control Nodes

| Node | Description |
|------|-------------|
| **START** | Entry point, defines input schema |
| **END** | Exit point, returns final output |
| **Condition** | Branch based on true/false logic |
| **Loop** | Iterate over arrays or repeat actions |
| **Delay** | Wait for a duration or until a time |

### AI Nodes

| Node | Description |
|------|-------------|
| **AI Agent** | Execute RAG-powered AI agent |
| **Voice Agent** | AI agent with TTS audio output |
| **Chat Agent** | Conversational agent with memory |

### Action Nodes

| Node | Description |
|------|-------------|
| **HTTP** | Make external API calls |
| **Code** | Run Python or JavaScript |
| **Action** | Document operations (create, update, delete) |
| **Notification** | Send emails, Slack, Teams messages |

### Approval Nodes

| Node | Description |
|------|-------------|
| **Human Approval** | Pause for human review |

## Building Your First Workflow

### Step 1: Add START Node

The START node defines what inputs your workflow accepts:

```json
{
  "document_id": "string",
  "user_query": "string"
}
```

### Step 2: Add Processing Nodes

Drag nodes from the palette and connect them:

1. Click a node in the palette
2. Click on the canvas to place it
3. Drag from output port to input port to connect

### Step 3: Configure Each Node

Click a node to open its configuration panel:

- **Name**: Descriptive label
- **Description**: What the node does
- **Settings**: Node-specific configuration

### Step 4: Add END Node

The END node specifies what to return:

- **Output Path**: JSONPath to extract (e.g., `$.result`)
- **Include Metadata**: Add execution info

## Node Configuration Examples

### AI Agent Node

```yaml
Agent Type: default
Prompt: |
  Analyze this document and extract key points:
  {{input.document}}
Model: gpt-4o
Temperature: 0.7
Use RAG: true
```

### Voice Agent Node

```yaml
Prompt: Summarize the following for audio playback
TTS Provider: cartesia
Voice: sonic-female
Speed: 1.1
Use RAG: true
Audio Format: mp3
```

### Chat Agent Node

```yaml
Prompt: Answer user questions about the project
Response Style: professional
Knowledge Bases: kb-support, kb-faq
Use Memory: true
Memory Type: session
Max History: 10
```

### Condition Node

```yaml
Condition Type: expression
Expression: input.score > 0.8
True Label: High Confidence
False Label: Low Confidence
```

### Loop Node

```yaml
Loop Type: for_each
Items Source: {{input.documents}}
Item Variable: doc
Max Iterations: 100
Parallel: true
Concurrency: 5
```

### HTTP Node

```yaml
Method: POST
URL: https://api.example.com/webhook
Auth Type: bearer
Token: {{secrets.api_token}}
Body: |
  {
    "result": "{{nodes.agent.output}}"
  }
```

## Data Flow & Variables

### Accessing Input Data

Use `{{input.fieldName}}` to access workflow inputs:

```
{{input.document_id}}
{{input.user_query}}
{{input.options.format}}
```

### Accessing Node Outputs

Use `{{nodes.nodeId.output}}` to access previous node results:

```
{{nodes.agent_1.output}}
{{nodes.http_request.response.data}}
{{nodes.condition_1.result}}
```

### Built-in Variables

| Variable | Description |
|----------|-------------|
| `{{workflow.id}}` | Current workflow ID |
| `{{workflow.name}}` | Workflow name |
| `{{workflow.url}}` | Workflow execution URL |
| `{{now}}` | Current timestamp |
| `{{loop.index}}` | Current loop iteration |
| `{{loop.item}}` | Current loop item |

## Workflow Patterns

### Sequential Processing

```
START → Agent → Transform → HTTP → END
```

### Conditional Branching

```
START → Agent → Condition ─┬─ (true) → Action A → END
                           └─ (false) → Action B → END
```

### Parallel Processing

```
START → Loop (parallel) → Agent → Merge → END
          ↓
    [doc1, doc2, doc3]
```

### Human-in-the-Loop

```
START → Agent → Human Approval ─┬─ (approved) → Deploy → END
                                └─ (rejected) → Notify → END
```

## Executing Workflows

### Manual Execution

1. Open the workflow
2. Click **Run**
3. Enter input values
4. Monitor execution progress

### API Execution

```bash
POST /api/v1/workflows/{workflow_id}/execute
Content-Type: application/json

{
  "inputs": {
    "document_id": "doc-123",
    "user_query": "What are the key findings?"
  }
}
```

### Scheduled Execution

Configure triggers in workflow settings:
- **Cron Schedule**: `0 9 * * 1-5` (9 AM weekdays)
- **Webhook**: Trigger via HTTP POST
- **Event**: On document upload, etc.

## Best Practices

1. **Name nodes descriptively** - Makes debugging easier
2. **Add descriptions** - Document what each node does
3. **Handle errors** - Configure error handling on each node
4. **Test incrementally** - Test each node before adding more
5. **Use conditions wisely** - Keep branching logic simple
6. **Set timeouts** - Prevent runaway workflows

## Troubleshooting

### Workflow won't start
- Check START node has valid input schema
- Verify all required connections exist
- Ensure no circular dependencies

### Node execution fails
- Check node configuration
- Verify input data format
- Review error message in execution log

### Slow execution
- Enable parallel processing where possible
- Reduce unnecessary nodes
- Optimize AI agent prompts

## Advanced Features

### Subworkflows

Call other workflows as nodes:
1. Create a reusable workflow
2. Reference it in another workflow
3. Pass inputs and receive outputs

### Error Handling

Per-node error handling:
- **Stop Workflow**: Halt on error
- **Continue**: Skip and proceed
- **Retry**: Retry with backoff

### Versioning

- Workflows are versioned automatically
- Roll back to previous versions
- Compare version differences

---

## Publishing & Deployment

Deploy workflows for public access, allowing external users to execute them without authentication.

### Deploying a Workflow

1. Open the workflow in the designer
2. Click the **Deploy** button in the toolbar
3. Configure deployment settings:
   - **Public Slug**: URL-friendly name (e.g., `my-workflow`)
   - **Branding**: Optional logo, primary color, company name
4. Click **Deploy**
5. Copy the public URL to share

### Public Workflow Page

The deployed workflow is accessible at `/w/{public_slug}`:

- Displays workflow name and description
- Auto-generates input form from workflow schema
- Executes workflow and shows results
- Shows loading states and error messages

### Undeploying

1. Open the deployed workflow
2. Click **Undeploy**
3. The public URL becomes inactive

### API Access

```bash
# Deploy
POST /api/v1/workflows/{workflow_id}/deploy
{
  "public_slug": "my-workflow",
  "branding": {
    "logo": "https://...",
    "primaryColor": "#8b5cf6"
  }
}

# Check status
GET /api/v1/workflows/{workflow_id}/deploy-status

# Undeploy
POST /api/v1/workflows/{workflow_id}/undeploy
```

---

## Sharing Workflows

Create secure share links with fine-grained permissions for team collaboration.

### Permission Levels

| Level | What Users Can Do |
|-------|------------------|
| **Viewer** | View workflow details and input schema |
| **Executor** | View and execute the workflow |
| **Editor** | View, execute, and duplicate the workflow |

### Creating a Share Link

1. Open the workflow
2. Click **Share** in the toolbar
3. Configure:
   - **Permission Level**: viewer, executor, or editor
   - **Password** (optional): Require password to access
   - **Expiration** (optional): Set an expiry date
   - **Max Uses** (optional): Limit number of accesses
4. Click **Create Link**
5. Copy the share URL

### Managing Shares

View and revoke share links:
1. Open the workflow
2. Click **Share** → **Manage Links**
3. See all active shares with usage stats
4. Click **Revoke** to disable a link

### Share Link Access

When accessing via share link:
- If password-protected, user must enter password first
- User sees workflow based on permission level
- Usage is tracked (count, last accessed)

---

## Scheduling

Schedule workflows to run automatically on a recurring basis.

### Setting Up a Schedule

1. Open the workflow
2. Click **Schedule** in the toolbar
3. Enter a cron expression or select a preset:
   - Every hour: `0 * * * *`
   - Daily at 9 AM: `0 9 * * *`
   - Weekdays at 9 AM: `0 9 * * 1-5`
   - Weekly on Monday: `0 9 * * 1`
   - Monthly on 1st: `0 0 1 * *`
4. Select timezone
5. (Optional) Set default input values
6. Click **Save Schedule**

### Cron Expression Format

```
┌───────────── minute (0-59)
│ ┌─────────── hour (0-23)
│ │ ┌───────── day of month (1-31)
│ │ │ ┌─────── month (1-12)
│ │ │ │ ┌───── day of week (0-6, Sun=0)
│ │ │ │ │
* * * * *
```

**Examples:**
- `*/15 * * * *` — Every 15 minutes
- `0 */6 * * *` — Every 6 hours
- `30 2 * * 1` — 2:30 AM every Monday
- `0 9 15 * *` — 9 AM on the 15th of each month

### Viewing Scheduled Executions

1. Go to **Workflows** page
2. Look for the clock icon on scheduled workflows
3. Click to see next run time and history

### Removing a Schedule

1. Open the scheduled workflow
2. Click **Schedule**
3. Click **Remove Schedule**

---

## Form Triggers

Create public forms that trigger workflow execution when submitted.

### Configuring a Form Trigger

1. Open the workflow
2. Click **Triggers** → **Form**
3. Configure:
   - **Title**: Form heading
   - **Description**: Instructions for users
   - **Success Message**: Shown after submission
   - **Rate Limit**: Max submissions per minute
4. Click **Enable**
5. Copy the form URL

### Form Features

- Input fields auto-generated from workflow input schema
- Field validation (required, type checking)
- Rate limiting to prevent abuse
- Customizable success message
- Submission tracking

---

## Event Triggers

Automatically trigger workflows based on system events.

### Supported Events

| Event Type | When It Fires |
|------------|---------------|
| `document.created` | New document indexed |
| `document.updated` | Document content changed |
| `connector.sync_completed` | Connector sync finished |
| `chat.message_received` | New chat message received |

### Configuring an Event Trigger

1. Open the workflow
2. Click **Triggers** → **Event**
3. Select event type
4. (Optional) Add filter conditions:
   - Filter by source (e.g., only Notion documents)
   - Filter by tags
   - Custom field matching
5. Click **Save**

### Filter Conditions

```json
{
  "source": "notion",
  "tags": ["important", "urgent"],
  "metadata.priority": "high"
}
```

Only events matching ALL conditions will trigger the workflow.

---

## Webhook Triggers

Trigger workflows from external systems via HTTP POST.

### Webhook URL

Each workflow has a unique webhook endpoint:

```
POST /api/v1/workflows/webhook/{workflow_id}
```

### Usage Example

```bash
curl -X POST "https://app.example.com/api/v1/workflows/webhook/{id}" \
  -H "Content-Type: application/json" \
  -d '{
    "event": "new_order",
    "order_id": "12345",
    "customer": "John Doe"
  }'
```

The entire payload becomes available as workflow input.

### Webhook Security

- Validate the source using custom headers
- Use secret tokens in the payload
- Configure IP allowlisting in settings

---

## Version Management

Track changes and restore previous versions.

### Viewing Version History

1. Open the workflow
2. Click **Versions** in the toolbar
3. See list of all versions with:
   - Version number
   - Creation timestamp
   - Author
   - Node count

### Restoring a Version

1. Click on a version in the history
2. Click **Restore This Version**
3. A new version is created with the restored content

Note: Restoring doesn't delete history—it creates a new version.

### Best Practices

- Make descriptive changes in logical units
- Test before making changes to production workflows
- Use restore for rollbacks, not manual re-creation
