# Knowledge Graph

Explore document relationships through the knowledge graph visualization.

## What is the Knowledge Graph?

The knowledge graph automatically extracts:
- **Entities**: People, organizations, locations, concepts
- **Relationships**: How entities connect to each other
- **Properties**: Attributes and metadata

## Viewing the Knowledge Graph

1. Navigate to the Knowledge Graph page
2. Select a document or collection
3. Explore the interactive visualization

## Graph Features

### Entity Types

- **Person**: Individuals mentioned in documents
- **Organization**: Companies, agencies, institutions
- **Location**: Places, addresses, regions
- **Concept**: Abstract ideas, topics, themes
- **Event**: Dated occurrences, meetings, milestones

### Relationship Types

- WORKS_FOR, LOCATED_IN, PART_OF
- MENTIONS, REFERENCES, CITES
- RELATED_TO, DEPENDS_ON, PRECEDES

## Querying the Graph

### Natural Language

Ask questions that leverage relationships:
> "Who works for Company X according to the documents?"

### Graph Search

Use the search box to find specific entities.

## Configuration

Enable enhanced graph extraction:

```env
ENABLE_KG_EXTRACTION=true
KG_EXTRACTION_MODEL=gpt-4o-mini
```

## GraphRAG Integration

The knowledge graph enhances RAG queries by:
- Finding related entities
- Traversing relationships
- Providing context from connected documents

## Next Steps

- [Advanced RAG](11-advanced-rag.md) - Learn about GraphRAG and hybrid retrieval
