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
