# Memory System

![Memory Tiers](images/memory-tiers.png)

## Overview

The system has two distinct memory subsystems:

1. **Session Memory** — per-conversation history for follow-up questions
2. **Global User Memory** — persistent facts/preferences that span conversations

```mermaid
flowchart TD
    subgraph "Session Memory (per conversation)"
        SM[ConversationBufferWindowMemory]
        SM --> QR[Query Rewriting<br/>follow-up → standalone]
        SM --> CTX[History in LLM context]
        SM --> DB_R[DB Rehydration<br/>survives restart]
    end

    subgraph "Global User Memory (persistent)"
        GM[User Memory Store]
        GM --> FACTS[Facts about user]
        GM --> PREFS[Preferences]
        GM --> ENTITIES_M[Known entities]
        GM --> PROCEDURES[How-to knowledge]
    end

    USER[User Query] --> SM
    USER --> GM
    SM --> ENRICHED[Enriched Query<br/>+ context]
    GM --> ENRICHED
    ENRICHED --> RAG[RAG Pipeline]
```

## Session Memory

**File:** `backend/services/session_memory.py`

### Memory Tiers by Model Size

```mermaid
flowchart TD
    MODEL[Model Name] --> PARSE[Parse size<br/>from name]
    PARSE --> TIER{Size?}

    TIER -->|≤3B| TINY["Tiny Tier<br/>k=3 turns<br/>budget: 10/10/60/20%"]
    TIER -->|3-9B| SMALL["Small Tier<br/>k=6 turns<br/>budget: 10/15/55/20%"]
    TIER -->|9-34B| MEDIUM["Medium Tier<br/>k=10 turns<br/>budget: 10/15/60/15%"]
    TIER -->|>34B| LARGE["Large Tier<br/>k=15 turns<br/>budget: 10/15/60/15%"]
```

### Query Rewriting

```mermaid
flowchart TD
    Q["How many of them?"] --> DETECT{Follow-up<br/>question?}
    DETECT -->|No| PASS[Use as-is]
    DETECT -->|Yes| SIZE{Model<br/>size?}
    SIZE -->|< 9B| HEURISTIC["Heuristic Rewrite<br/>pronoun replacement<br/>+ keyword injection"]
    SIZE -->|≥ 14B| LLM_RW["LLM Rewrite<br/>generate standalone"]

    HEURISTIC --> RESULT["'How many of the 5<br/>planetary boundaries<br/>have been breached?'"]
    LLM_RW --> RESULT
```

**Why heuristic for small models?** 3B models can't do meta-tasks — they answer the rewrite prompt as if it were a question instead of rewriting.

### DB Rehydration

After a backend restart, conversation history is restored from the database:

```mermaid
sequenceDiagram
    participant U as User
    participant S as SessionMemory
    participant DB as ChatMessage Table

    U->>S: Continue conversation (session_id)
    S->>S: Check in-memory buffer
    Note over S: Empty after restart

    S->>DB: SELECT * FROM chat_messages<br/>WHERE session_id = ?<br/>ORDER BY created_at DESC<br/>LIMIT k*2

    DB-->>S: Last k conversation pairs
    S->>S: Load into ConversationBufferWindowMemory
    S-->>U: Context restored, ready for follow-ups
```

## Global User Memory

**File:** `backend/services/mem0_memory.py`, `backend/api/routes/memory.py`

### Memory Types

| Type | Example | Priority |
|------|---------|----------|
| `fact` | "User works at ACME Corp" | high |
| `preference` | "Prefers concise answers" | medium |
| `context` | "Working on Q4 report" | low |
| `procedure` | "Use pytest for testing" | medium |
| `entity` | "John = team lead" | medium |
| `relationship` | "Reports to Jane" | low |

### Memory Lifecycle

```mermaid
flowchart TD
    EXTRACT[LLM extracts memory<br/>from conversation] --> CLASSIFY[Classify type<br/>+ priority]
    CLASSIFY --> DEDUP{Similar memory<br/>already exists?}
    DEDUP -->|Yes| MERGE[Merge/Update<br/>existing memory]
    DEDUP -->|No| CREATE[Create new<br/>memory entry]

    MERGE --> STORE[Memory Store]
    CREATE --> STORE

    STORE --> DECAY[Decay over time<br/>if not accessed]
    DECAY --> PRUNE{Decay score<br/>< threshold?}
    PRUNE -->|Yes| DELETE[Auto-delete]
    PRUNE -->|No| KEEP[Keep alive]
```

### Memory API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/memory` | List all memories (paginated, filterable) |
| GET | `/api/v1/memory/stats` | Memory statistics |
| GET | `/api/v1/memory/export` | Export all memories as JSON |
| GET | `/api/v1/memory/{id}` | Get single memory |
| PATCH | `/api/v1/memory/{id}` | Update content or priority |
| DELETE | `/api/v1/memory/{id}` | Delete single memory |
| DELETE | `/api/v1/memory?confirm=true` | Clear all memories |

### Frontend Memory Page

Located at Dashboard > Memory, provides:
- Searchable, filterable list of all memories
- Type and priority badges
- Edit and delete actions
- Export to JSON download
- Clear all with confirmation
- Statistics overview (total, by type, by priority)
