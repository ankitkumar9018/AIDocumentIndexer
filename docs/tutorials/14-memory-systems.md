# Memory Systems

Configure persistent memory for conversations and agents.

## Overview

AIDocumentIndexer supports multiple memory systems:

- **Mem0**: Simple key-value memory storage
- **A-Mem**: Agentic memory with importance ranking
- **Session Memory**: In-conversation context

## Mem0 Memory

Basic memory system for storing facts and preferences.

### Configuration

```env
ENABLE_AGENT_MEMORY=true
MEMORY_PROVIDER=mem0
```

### Features

- User preference storage
- Conversation history
- Fact extraction
- Cross-session persistence

## A-Mem (Agentic Memory)

Advanced memory with intelligent importance ranking.

### Configuration

```env
MEMORY_PROVIDER=amem
```

### Features

- **Importance Scoring**: Automatically ranks memory importance
- **Decay**: Old memories fade over time
- **Consolidation**: Similar memories merge
- **Active Recall**: Relevant memories surface automatically

### Memory Types

- **Episodic**: Specific events and conversations
- **Semantic**: Facts and knowledge
- **Procedural**: How-to information

## Memory Configuration

### Memory Limits

```env
MEMORY_MAX_ENTRIES=1000
MEMORY_DECAY_RATE=0.01  # Per hour
```

### Memory Filtering

```env
MEMORY_MIN_IMPORTANCE=0.3
MEMORY_CONSOLIDATION_THRESHOLD=0.85
```

## Using Memory in Agents

Agents automatically use memory for:

1. **Personalization**: Remember user preferences
2. **Context**: Recall previous conversations
3. **Learning**: Improve over time

### Example

User: "I prefer concise answers"
*Agent remembers preference*

Later...
Agent provides shorter responses automatically.

## Privacy Controls

- **Memory Clearing**: Users can clear their memory
- **Opt-out**: Disable memory per user
- **Retention**: Set memory expiration

## Debugging Memory

View stored memories:

```bash
curl http://localhost:8000/api/v1/memory/{user_id}
```

## Next Steps

- [Custom Agents](07-custom-agents.md) - Build agents that use memory
- [Advanced RAG](11-advanced-rag.md) - Enhance retrieval with context
