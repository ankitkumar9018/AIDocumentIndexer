# Multi-Agent System Documentation

## Overview

AIDocumentIndexer includes a production-grade multi-agent system that enables complex document processing tasks through specialized AI agents. The system features:

- **Manager Agent**: Orchestrates task decomposition and worker coordination
- **Worker Agents**: Specialized agents for generation, evaluation, research, and tool execution
- **Self-Improvement**: Background prompt optimization with A/B testing
- **Mode Router**: Intelligent routing between agent mode, chat mode, and general mode
- **Cost Management**: Budget enforcement and cost estimation

## Execution Modes

The system supports three distinct execution modes:

### Agent Mode
Multi-agent orchestration for complex tasks:
- Task decomposition into subtasks
- Parallel worker agent execution
- Quality assurance through critic agent
- Best for: complex analysis, document generation, multi-step research

### Chat Mode (RAG)
Document-aware conversation with retrieval:
- Semantic search across indexed documents
- Source citations with page numbers
- Conversation memory per session
- Best for: document Q&A, finding specific information

### General Mode
Pure LLM conversation without document search:
- Direct LLM responses using general knowledge
- No RAG retrieval overhead
- Fast responses for general questions
- Best for: general knowledge questions, explanations, coding help

### Mode Selection

Users can select modes via:
1. **Chat UI**: Toggle between "Documents" and "General" modes
2. **API**: Set `mode` parameter in `/chat/completions`
3. **Settings**: Configure default mode and auto-detection preferences

### Smart Fallback

When `fallback_to_general` is enabled:
- If RAG search finds no relevant documents
- System automatically falls back to General mode
- User is notified that response is from general knowledge

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API Layer (/api/v1/agent/)                          │
│  POST /execute  |  GET /mode  |  POST /mode/toggle  |  GET /status         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Complexity Detector & Mode Router                       │
│  • Analyzes request complexity (keywords, length, structure)                │
│  • Routes: AGENT mode (complex) vs CHAT mode (simple)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                                       │
     ┌──────────────┴──────────────┐          ┌────────────┴────────────┐
     ▼                             ▼          ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  Budget Checker     │   │    Manager Agent    │   │   Normal Chat Mode  │
│  • Estimate cost    │──▶│  • Task decompose   │   │  • Direct RAGService│
│  • Check limits     │   │  • Worker dispatch  │   │  • Streaming        │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
     │ Generator Agent │   │  Critic Agent   │   │ Research Agent  │
     │ • Content draft │   │ • LLM-as-judge  │   │ • RAG search    │
     └─────────────────┘   └─────────────────┘   └─────────────────┘
```

## Agent Types

### Manager Agent
The orchestrator that coordinates all worker agents:
- Analyzes user requests to understand intent
- Decomposes complex tasks into subtasks
- Assigns tasks to appropriate worker agents
- Monitors progress and handles failures
- Aggregates results into final output

**Recent Improvements (2025-12/2026-01):**
- **Fallback Research Step**: If the user's request contains document-related keywords (e.g., "documents", "files", "my data") but the LLM plan doesn't include a research step, the Manager automatically prepends one
- **Context Passing**: Research findings are now passed to the Generator agent as context, improving output quality
- **Streaming Output**: Step outputs are streamed in real-time via SSE `content` events
- **Source Attribution**: Research agent sources are propagated to the frontend for display
- **Multi-Step Synthesis Fix** (2026-01): Fixed an issue where multi-step tasks (e.g., "research X and create Y") would only return research findings without the creative/generated output. The synthesis now properly combines all step outputs and emphasizes completing the user's full request

### Generator Agent
Handles content creation tasks:
- Document generation
- Report writing
- Summary creation
- Uses structured prompts with few-shot examples

### Critic Agent
Quality evaluation using LLM-as-judge pattern:
- Evaluates generated content quality
- Provides scores across multiple dimensions
- Suggests improvements
- Validates factual accuracy

### Research Agent
Information retrieval and synthesis:
- RAG-based document search
- Web scraping integration
- Source aggregation
- Citation management

**Source Data Returned:**
```json
{
  "findings": "Summary of research findings...",
  "sources": [
    {
      "source": "document_name.pdf",
      "document_id": "uuid",
      "chunk_id": "uuid",
      "page_number": 5,
      "content": "Relevant excerpt...",
      "full_content": "Complete chunk content...",
      "score": 0.92,
      "relevance_score": 0.88,
      "collection": "My Collection"
    }
  ],
  "result_count": 10,
  "suggested_questions": ["Follow-up question 1?", "Follow-up question 2?"]
}
```

**Advanced RAG Features (2025-12):**
- **GraphRAG Integration**: Research agent can leverage knowledge graph for multi-hop reasoning
- **Agentic RAG**: Complex research queries are automatically decomposed into sub-questions
- **Multimodal Support**: Can retrieve and reason about images and tables in documents
- **Confidence Scoring**: Each source has relevance scores for transparency
- **Query Suggestions**: Intelligent follow-up questions generated after research

**Recent Improvements:**
- Sources now include full metadata (document_id, chunk_id, page_number, collection)
- Sources are streamed to frontend via SSE for real-time display
- Collection context is included in search results for better LLM context
- Full chunk content available for detailed source viewing

### Tool Executor Agent
File operations and exports:
- PPTX generation
- DOCX creation
- PDF export
- Markdown formatting

## Execution Modes

### Agent Mode (Default)
- Multi-agent orchestration for complex tasks
- Task decomposition and parallel execution
- Quality assurance through critic agent
- Progress tracking and cost estimation

### Chat Mode
- Direct conversation with RAG-powered responses
- Lower latency for simple queries
- No multi-step processing
- Ideal for quick lookups

### Auto-Detection
The system automatically detects query complexity based on:
- Keywords (generate, analyze, compare, research, etc.)
- Query length
- Multi-part requests
- References to multiple documents

## API Endpoints

### Execution

```
POST /api/v1/agent/execute
```
Execute a request through the multi-agent system.

Request body:
```json
{
  "request": "Generate a comprehensive market analysis report",
  "session_id": "optional-session-id",
  "mode": "agent",
  "require_approval": true
}
```

Response:
```json
{
  "plan_id": "uuid",
  "session_id": "uuid",
  "status": "cost_approval_required",
  "estimated_cost_usd": 0.15,
  "plan_summary": "3-step plan: research, analyze, generate",
  "steps": [...]
}
```

### Streaming Execution

```
POST /api/v1/agent/execute/stream
```
Execute with Server-Sent Events for real-time progress updates.

### Mode Management

```
GET /api/v1/agent/mode
POST /api/v1/agent/mode
POST /api/v1/agent/mode/toggle
```
Get, set, or toggle the execution mode.

### User Preferences

```
GET /api/v1/agent/preferences
PATCH /api/v1/agent/preferences
```
Manage user preferences for agent behavior:
- `default_mode`: "agent" or "chat"
- `agent_mode_enabled`: boolean
- `auto_detect_complexity`: boolean
- `show_cost_estimation`: boolean
- `require_approval_above_usd`: number

### Agent Status

```
GET /api/v1/agent/status
GET /api/v1/agent/agents/{agent_id}/metrics
GET /api/v1/agent/agents/{agent_id}/config
```
Monitor agent health and performance metrics.

### Prompt Optimization (Admin)

```
POST /api/v1/agent/agents/{agent_id}/optimize
GET /api/v1/agent/optimization/jobs
POST /api/v1/agent/optimization/jobs/{job_id}/approve
POST /api/v1/agent/optimization/jobs/{job_id}/reject
```
Manage prompt optimization jobs.

## Cost Management

### Cost Estimation
Before execution, the system estimates costs based on:
- Model pricing (per-token rates)
- Expected input/output token counts
- Number of agent steps

### Budget Enforcement
- Users have configurable daily limits
- Approval required above threshold
- Real-time cost tracking
- Automatic budget alerts

### Model Pricing
Current supported models with pricing:
- GPT-4o: $2.50/M input, $10.00/M output
- GPT-4o-mini: $0.15/M input, $0.60/M output
- Claude 3.5 Sonnet: $3.00/M input, $15.00/M output

## Self-Improvement System

### Trajectory Collection
All agent executions are recorded with:
- Input/output data
- Reasoning traces
- Token usage
- Duration metrics
- Success/failure status
- Quality scores

### Prompt Builder Agent
Background agent that runs daily to:
1. Analyze failed trajectories from past 24 hours
2. Cluster similar failure patterns
3. Generate improved prompt variants using mutation strategies
4. Run A/B tests with rainbow deployment (10% → 25% → 50% → 100%)
5. Request human approval before promotion

### A/B Testing
Prompt variants are tested with traffic splitting:
- Multiple variants tested simultaneously
- Statistical significance tracking
- Automatic winner detection
- Manual approval required for production

### Rollback
Previous prompt versions are preserved for:
- Quick rollback on performance degradation
- Version history with performance metrics
- Audit trail of changes

## Database Schema

### Agent Tables

1. **agent_definitions**: Agent configurations
2. **agent_prompt_versions**: Prompt versions with A/B test traffic
3. **agent_trajectories**: Execution traces for analysis
4. **agent_execution_plans**: Task decomposition plans
5. **prompt_optimization_jobs**: Background optimization jobs
6. **execution_mode_preferences**: User preferences

## Frontend Integration

### Mode Toggle
The chat interface includes a mode toggle that allows users to:
- Switch between Agent and Chat modes
- Configure auto-detection preferences
- Set cost approval thresholds

### Cost Approval Dialog
When execution cost exceeds threshold:
- Shows estimated cost breakdown
- Displays execution steps
- Approve or cancel options

### Agent Execution Progress
Real-time progress display showing:
- Current step and overall progress
- Agent status indicators
- Cost tracking per step

### Streaming Events
Agent mode streams various events via SSE:

| Event Type | Purpose | Data |
|------------|---------|------|
| `agent_step` | Step status updates | `{ step: "Research", status: "in_progress" }` |
| `content` | Step output display | `{ data: "**Research**\n\nFindings..." }` |
| `sources` | Document citations | `{ data: [{ document_id, filename, ... }] }` |
| `done` | Execution complete | `{ message_id, content }` |

**Frontend Integration:**
```typescript
// Handle streaming in chat component
switch (chunk.type) {
  case "content":
    streamContent += chunk.data;
    break;
  case "sources":
    message.sources.push(...chunk.data);
    break;
  case "agent_step":
    currentStep = chunk.step;
    break;
}
```

### Admin Dashboard
For administrators to:
- Monitor agent health
- Review optimization jobs
- Approve prompt improvements
- View execution trajectories

## Configuration

### Environment Variables

```env
# Agent defaults
DEFAULT_AGENT_TEMPERATURE=0.7
DEFAULT_AGENT_MAX_TOKENS=4096

# Cost limits
DEFAULT_DAILY_COST_LIMIT=10.0
DEFAULT_APPROVAL_THRESHOLD=1.0

# Optimization
PROMPT_OPTIMIZATION_ENABLED=true
OPTIMIZATION_ANALYSIS_HOURS=24
OPTIMIZATION_MIN_TRAJECTORIES=50
```

### Per-Agent Configuration
Each agent can be configured with:
- Custom LLM provider
- Model override
- Temperature settings
- Max tokens
- Active/inactive status

## Best Practices

### When to Use Agent Mode
- Complex multi-step tasks
- Document generation
- Research and analysis
- Content comparison
- Quality-critical outputs

### When to Use Chat Mode
- Quick questions
- Simple lookups
- Clarification queries
- Low-latency requirements

### Cost Optimization
- Set appropriate approval thresholds
- Use GPT-4o-mini for simpler tasks
- Monitor daily usage
- Review trajectory costs

## Troubleshooting

### Common Issues

**High Costs**
- Review agent configurations
- Check model selections
- Set lower approval thresholds

**Slow Responses**
- Check if agent mode is needed
- Reduce max tokens
- Use faster models for non-critical steps

**Failed Executions**
- Check trajectory logs
- Review error messages
- Verify API keys and limits

### Monitoring
- Agent health dashboard shows degraded agents
- Optimization jobs highlight underperforming prompts
- Cost dashboard tracks usage trends
