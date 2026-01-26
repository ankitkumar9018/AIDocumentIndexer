# Advanced RAG

Master advanced retrieval techniques for better answers.

## RAG Pipeline Overview

AIDocumentIndexer uses a sophisticated retrieval pipeline:

1. **Query Classification**: Understand query intent
2. **Hybrid Retrieval**: Combine multiple retrieval methods
3. **Reranking**: Score and filter results
4. **Generation**: Create response with citations

## Retrieval Methods

### Dense Retrieval (Default)

Semantic search using embeddings. Best for:
- Conceptual questions
- Paraphrased queries
- General understanding

### Sparse Retrieval (BM25)

Keyword matching. Best for:
- Exact term searches
- Technical terminology
- Acronyms and codes

### Hybrid Retrieval

Combines dense and sparse for better coverage:

```env
ENABLE_HYBRID_SEARCH=true
HYBRID_ALPHA=0.7  # 70% dense, 30% sparse
```

## Advanced Retrievers

### ColBERT

Late interaction model for precise matching:

```env
ENABLE_COLBERT=true
```

### LightRAG

Dual-level retrieval with entity extraction:

```env
ENABLE_LIGHTRAG=true
```

### RAPTOR

Tree-organized retrieval for hierarchical documents:

```env
ENABLE_RAPTOR=true
```

## Reranking

### Tiered Reranking

Multi-stage scoring pipeline:

1. **BM25 Pre-filter**: Fast keyword matching (100 → 50)
2. **Cross-Encoder**: Semantic scoring (50 → 20)
3. **ColBERT**: Fine-grained matching (20 → 10)
4. **LLM Verification**: Final validation (optional)

```env
ENABLE_TIERED_RERANKING=true
RERANK_STAGE1_TOP_K=100
RERANK_STAGE2_TOP_K=20
```

## GraphRAG

Enhance retrieval with knowledge graph:

- Entity-based context expansion
- Relationship traversal
- Community detection

```env
ENABLE_GRAPHRAG=true
```

## Tree of Thoughts

For complex analytical queries:

```env
ENABLE_TREE_OF_THOUGHTS=true
TOT_MAX_DEPTH=3
TOT_BRANCHING_FACTOR=3
```

## Answer Refinement

Post-generation quality improvement:

```env
ENABLE_ANSWER_REFINER=true
ANSWER_REFINER_STRATEGY=self_refine
```

## Adaptive RAG Routing (Phase 66)

Automatically routes queries to optimal strategy:

```env
ENABLE_ADAPTIVE_ROUTING=true
```

**Routing Strategies:**
- **DIRECT**: Simple factual queries → fast single-shot
- **HYBRID**: Standard queries → vector + keyword
- **TWO_STAGE**: Complex queries → retrieval + reranking
- **AGENTIC**: Multi-step queries → query decomposition
- **GRAPH_ENHANCED**: Entity-rich queries → knowledge graph

## RAG-Fusion (Phase 66)

Multi-query generation with Reciprocal Rank Fusion:

```env
ENABLE_RAG_FUSION=true
RAG_FUSION_QUERY_COUNT=3
```

Improves recall by 20-40% by searching multiple query variations.

## LazyGraphRAG (Phase 66)

Query-time community summarization:

```env
ENABLE_LAZY_GRAPHRAG=true
```

Achieves 99% cost reduction by generating community summaries on-demand.

## User Personalization (Phase 66)

Learn user preferences from feedback:

```env
ENABLE_USER_PERSONALIZATION=true
```

System learns:
- Response length preferences
- Format preferences (bullets vs prose)
- Expertise level
- Citation style

## Dependency-Based Entity Extraction (Phase 66)

Fast spaCy-based extraction (94% of LLM quality):

```env
ENABLE_DEPENDENCY_EXTRACTION=true
COMPLEXITY_THRESHOLD=0.7
```

Automatic LLM fallback for complex text.

## RAG Evaluation (RAGAS)

Evaluate RAG quality with industry metrics:

```bash
# Evaluate single query
curl -X POST http://localhost:8000/api/v1/evaluation/query \
  -d '{"query": "...", "response": "...", "context": [...]}'

# View evaluation stats
curl http://localhost:8000/api/v1/evaluation/stats
```

Metrics: Context Relevance, Faithfulness, Answer Relevance

## Next Steps

- [Memory Systems](14-memory-systems.md) - Persistent conversation memory
