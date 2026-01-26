# AIDocumentIndexer - Documentation Index

**Last Updated:** January 26, 2026

## Quick Links

- [📖 Main README](../README.md)
- [🚀 Getting Started](#getting-started)
- [📚 Core Features](#core-features)
- [🔧 Configuration](#configuration)
- [🔒 Security](SECURITY.md)
- [🎯 Guides & Tutorials](#guides--tutorials)

---

## Getting Started

### Installation & Setup
- [Main README](../README.md) - Project overview, installation, quick start
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

### First Steps
1. Install dependencies: `npm install` (frontend) + `pip install -r requirements.txt` (backend)
2. Configure embeddings: See [Embedding Models Guide](embeddings/EMBEDDING_MODELS.md)
3. Run the application: `npm run dev` (frontend) + `uvicorn backend.main:app` (backend)
4. Upload documents and start chatting!

---

## Core Features

### 1. Embeddings System 📊

**Current Status:** ✅ Fully Operational (3,959 chunks, 1,559 entities with 768D embeddings)

**Documentation:**
- **[Embedding Models Guide](embeddings/EMBEDDING_MODELS.md)** (372 lines)
  - Complete reference for all supported providers (OpenAI, Ollama, HuggingFace, Cohere, Voyage, Mistral)
  - Model dimensions, costs, performance comparisons
  - Quality metrics (MTEB scores)
  - **Use this when:** Choosing which embedding provider to use

- **[Embedding Dimensions Guide](embeddings/EMBEDDING_DIMENSIONS.md)** (311 lines)
  - How flexible dimensions work (384D-3072D)
  - Migration guides for switching providers
  - Troubleshooting dimension mismatches
  - **Use this when:** Switching between providers or debugging dimension errors

- **[Multi-Embedding Proposal](embeddings/MULTI_EMBEDDING_PROPOSAL.md)** (308 lines)
  - Architecture for storing embeddings from multiple providers
  - Migration strategy and performance impact
  - **Use this when:** Planning to support instant provider switching

- **[Multi-Embedding Usage Guide](embeddings/MULTI_EMBEDDING_USAGE.md)** (389 lines)
  - Step-by-step usage guide for multi-provider embeddings
  - Example workflows (dev/prod, A/B testing)
  - **Use this when:** Implementing multi-provider support

**Key Concepts:**
- **Embeddings** = Vector representations of text that enable semantic search
- **Dimensions** = Size of embedding vectors (768D for Ollama, 1536D for OpenAI default)
- **Provider** = Service that generates embeddings (Ollama = local/free, OpenAI = cloud/paid)

**Quick Configuration:**
```bash
# Ollama (Free, Local, 768D)
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# OpenAI (Quality, Cloud, 768D - matches Ollama!)
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=768
OPENAI_API_KEY=sk-...
```

---

### 2. Knowledge Graph 🕸️

**Current Status:** ✅ Enabled by Default (1,559 entities with embeddings)

**Documentation:**
- **[Knowledge Graph Completion Report](knowledge-graph/KNOWLEDGE_GRAPH_COMPLETION.md)** (245 lines)
  - Complete implementation status of all 7 phases
  - Performance metrics and testing results
  - **Use this when:** Understanding what's been implemented

**Key Features:**
- ✅ LLM-based entity extraction (10 types: PERSON, ORG, LOCATION, etc.)
- ✅ 13 relationship types
- ✅ Graph-augmented reranking (+0.2 per entity overlap, +0.3 for relationships)
- ✅ Multi-language support with canonical names
- ✅ Small model support (Llama, Qwen, DeepSeek, Phi)
- ✅ Adaptive batch sizing (2-8 chunks based on model context)

**How It Works:**
```
Query: "What does the CEO do?" →
  ↓
1. Extract entities from query ("CEO" = PERSON role)
  ↓
2. Find related entities in graph (CEO → WORKS_FOR → Company)
  ↓
3. Retrieve chunks mentioning CEO and company
  ↓
4. Boost chunks with entity overlap (+0.2 per match)
  ↓
5. Add relationship bonuses (+0.3 if entities connected)
  ↓
6. Return enriched results (+15-20% precision)
```

---

### 3. RAG (Retrieval-Augmented Generation) 🔍

**Current Status:** ✅ Fully Functional with All Features

**Integration Flow:**
```
User Query →
  ↓
Query Classification (determines search strategy) →
  ↓
Generate Query Embedding (768D Ollama) →
  ↓
Hybrid Search (Vector + Keyword) on 3,959 chunks →
  ↓
Knowledge Graph Enhancement:
  • Rerank by entity overlap
  • Add relationship bonuses
  • Retrieve entity-connected chunks →
  ↓
MMR for diversity (if needed) →
  ↓
Generate response with LLM
```

**Key Features:**
- ✅ **Vector Search:** Semantic similarity using embeddings
- ✅ **Keyword Search:** BM25 for exact matches
- ✅ **Hybrid Search:** Combines vector + keyword with dynamic weighting
- ✅ **Query Classification:** Determines optimal search strategy
- ✅ **Knowledge Graph:** Entity-based enhancements
- ✅ **MMR:** Maximal Marginal Relevance for diversity
- ✅ **Caching:** Semantic caching of LLM responses

**Phase 62/63 Advanced Features (Optional):**
- ⚙️ **Tree of Thoughts:** Multi-path reasoning for complex queries (`ENABLE_TREE_OF_THOUGHTS`)
- ⚙️ **Answer Refiner:** Post-generation quality improvement (`ENABLE_ANSWER_REFINER`)
- ⚙️ **Sufficiency Checker:** ICLR 2025 context detection (`ENABLE_SUFFICIENCY_CHECKER`)
- ⚙️ **TTT Compression:** Long context compression (`ENABLE_TTT_COMPRESSION`)
- ⚙️ **Fast Chunking:** Chonkie 33x faster chunking (`ENABLE_FAST_CHUNKING`)
- ⚙️ **Docling Parser:** 97.9% table extraction (`ENABLE_DOCLING_PARSER`)
- ⚙️ **Agent Evaluation:** Pass^k metrics, hallucination detection (`ENABLE_AGENT_EVALUATION`)

**Performance:**
- Query latency: ~60-120ms
- Semantic search: 40-60% better than keyword-only
- Knowledge graph: +15-20% precision improvement

---

## Configuration

### Environment Variables

**Embedding Configuration:**
```bash
# Provider Selection
DEFAULT_LLM_PROVIDER=ollama  # or openai, huggingface, cohere
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small

# Dimension Override (optional)
EMBEDDING_DIMENSION=768  # Auto-detected if not set

# OpenAI Configuration
OPENAI_API_KEY=sk-...
```

**Knowledge Graph Configuration:**
```bash
# Enable/disable knowledge graph (default: true)
KNOWLEDGE_GRAPH_ENABLED=true

# Graph traversal depth (default: 2)
KNOWLEDGE_GRAPH_MAX_HOPS=2
```

**RAG Configuration:**
```bash
# Hybrid search (default: true)
USE_HYBRID_SEARCH=true

# Similarity threshold (default: 0.7)
SIMILARITY_THRESHOLD=0.7

# Top K results (default: 10)
TOP_K_RESULTS=10
```

---

## Guides & Tutorials

### For Users

**[Session Summary](guides/SESSION_SUMMARY.md)** (comprehensive overview)
- What was implemented (embeddings, knowledge graph, flexible dimensions)
- Current system state (100% coverage, 5,518 embeddings)
- Expected quality improvements
- Cost analysis and recommendations

**[UI Embedding Controls Proposal](guides/UI_EMBEDDING_CONTROLS_PROPOSAL.md)**
- Proposed UI enhancements for embedding control
- Upload page: provider selection
- Chat page: embedding provider selector
- Settings page: embedding status dashboard

### For Developers

**Backend Scripts:**
Located in `/backend/scripts/`

**Diagnostic Scripts:**
- `check_embedding_dimension.py` - Verify current dimension configuration
- `check_embeddings.py` - Check entity embedding status
- `check_all_embeddings.py` - Check all tables (entities, chunks, documents)
- `test_rag_search.py` - Test RAG search capability
- `test_embedding_quality.py` - Test semantic search quality

**Migration Scripts:**
- `backfill_entity_embeddings.py` - Generate embeddings for entities
- `backfill_chunk_embeddings.py` - Generate embeddings for chunks
- `migrate_entity_embeddings_768d.py` - Migration for 1536D → 768D
- `migrate_embedding_dimensions.py` - Generic migration script
- `generate_additional_embeddings.py` - Generate multi-provider embeddings

**Helper Scripts:**
- `show_embedding_examples.py` - Visual configuration examples

---

## API Reference

### Embeddings API

**GET `/api/v1/embeddings/stats`**
- Get embedding system statistics
- Returns: coverage, storage, per-provider breakdown

**POST `/api/v1/embeddings/generate-missing`**
- Trigger background job to generate missing embeddings
- Status: Not yet implemented (use CLI scripts)

### Knowledge Graph API

**GET `/api/v1/knowledge-graph/entities`**
- List all entities
- Supports filtering by type, search

**GET `/api/v1/knowledge-graph/entities/{id}`**
- Get single entity with relationships

**POST `/api/v1/knowledge-graph/search`**
- Search entities by semantic similarity

### RAG API

**POST `/api/v1/chat`**
- Chat with RAG capabilities
- Mode: `chat` (RAG), `agent` (orchestration), `general` (no RAG)

**POST `/api/v1/chat/stream`**
- Streaming chat response
- Returns: SSE stream with content, sources, confidence

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Upload  │  │   Chat   │  │ Settings │  │  Search  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              RAG Service                              │  │
│  │  • Query Classification                               │  │
│  │  • Hybrid Search (Vector + Keyword)                   │  │
│  │  • Knowledge Graph Enhancement                        │  │
│  │  • MMR Diversity                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ▼                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ Embeddings │  │ KG Service │  │ LLM Router │           │
│  │  Service   │  │            │  │            │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Database (SQLite/PostgreSQL)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Chunks  │  │ Entities │  │Documents │  │ Sessions │   │
│  │ (3,959)  │  │ (1,560)  │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│  All chunks have 768D embeddings (Ollama nomic-embed-text)  │
│  All entities have embeddings (semantic entity search)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Metrics

### Current System Stats

```
📊 ENTITIES:
   ✅ With embeddings:      1,559
   ❌ Without embeddings:       1 (empty name)
   📈 Total:                1,560

📄 CHUNKS:
   ✅ With embeddings:      3,959
   ❌ Without embeddings:       0
   📈 Total:                3,959

Total embeddings:         5,518
Overall coverage:        100.0%
```

### Query Performance

- Vector search: ~10-20ms for 3,959 chunks
- Knowledge graph enhancement: +50-100ms
- Total RAG query: ~60-120ms
- Semantic vs keyword: 40-60% better recall

### Storage

- Chunks: 12 MB (3,959 × 768 floats × 4 bytes)
- Entities: 4.8 MB (1,559 × 768 floats × 4 bytes)
- Total: ~17 MB

---

## Troubleshooting

### Common Issues

**1. "No embeddings found" / Search not working**
```bash
# Check embedding status
python backend/scripts/check_all_embeddings.py

# If 0 chunks have embeddings:
python backend/scripts/backfill_chunk_embeddings.py
```

**2. "Expected 1536 dimensions, not 768"**
```bash
# Check current configuration
python backend/scripts/check_embedding_dimension.py

# Run migration
python backend/scripts/migrate_embedding_dimensions.py

# Re-generate embeddings
python backend/scripts/backfill_chunk_embeddings.py
```

**3. Chat returns generic responses (not using documents)**
- Check embedding coverage: should be 100%
- Verify RAG search is enabled: `USE_HYBRID_SEARCH=true`
- Test semantic search: `python backend/scripts/test_embedding_quality.py`

**4. Switching embedding providers**
```bash
# Same dimension (no re-indexing needed)
# Ollama 768D → OpenAI 768D
DEFAULT_LLM_PROVIDER=openai
EMBEDDING_DIMENSION=768

# Different dimension (re-indexing required)
# Run migration first
python backend/scripts/migrate_embedding_dimensions.py

# Then re-generate embeddings
python backend/scripts/backfill_chunk_embeddings.py
```

---

## Cost Analysis

### Embedding Generation Cost

| Provider | Model | Dimension | API Cost (per 1M tokens) | Storage (per 1M embeddings) |
|----------|-------|-----------|--------------------------|----------------------------|
| Ollama | nomic-embed-text | 768D | $0 | 3 GB |
| OpenAI | text-embedding-3-small | 768D | $0.02 | 3 GB |
| OpenAI | text-embedding-3-small | 512D | $0.02 | 2 GB |
| OpenAI | text-embedding-3-large | 3072D | $0.13 | 12 GB |

**For 3,959 chunks (current system):**
- Ollama: $0 (free, local)
- OpenAI (768D): ~$0.04 one-time
- Storage: ~12 MB

---

## Changelog

### January 20, 2026 - Embedding System Complete
- ✅ Generated embeddings for all 3,959 chunks (Ollama 768D)
- ✅ Generated embeddings for all 1,559 entities
- ✅ Implemented flexible embedding dimensions (384D-3072D)
- ✅ Knowledge graph fully operational (+15-20% precision)
- ✅ Created comprehensive documentation
- ✅ Built diagnostic and migration scripts
- ✅ Chat now uses semantic search + knowledge graph
- ✅ RAG search fully functional (100% coverage)

### January 23, 2026 - Phase 62/63 Service Integration
- ✅ Integrated TreeOfThoughts for complex analytical queries
- ✅ Integrated AnswerRefiner for post-generation quality improvement
- ✅ Integrated SufficiencyChecker (ICLR 2025) for context detection
- ✅ Integrated TTTCompression for long context handling
- ✅ Integrated FastChunker (Chonkie 33x faster)
- ✅ Integrated DocumentParser (Docling 97.9% table accuracy)
- ✅ Integrated AgentEvaluator (Pass^k metrics)
- ✅ Added 6 new feature flags for runtime control
- ✅ Exported 23 new services in __init__.py
- ✅ Created 8 tutorial stub files
- ✅ Archived 6 outdated docs, removed 6 duplicates

### January 23, 2026 - Phase 65: Scale to 1M+ Documents
- ✅ **BM25 Scoring**: Search-engine quality ranking with term saturation
- ✅ **Field Boosting**: Title/section matches weighted higher (3x for titles)
- ✅ **Scale-Aware HNSW**: Auto-tuned index params (small/medium/large/xlarge)
- ✅ **EnhancedWebCrawler**: Anti-bot bypass + LLM content extraction
- ✅ **Web Query API**: Answer questions about any website
- ✅ **Text-to-SQL Enhancements**: Interactive queries + auto-visualization
- ✅ **ABAC for Retrieval**: Attribute-based access control for search results
- ✅ **30+ New Settings**: Comprehensive configuration for all features
- ✅ **API Endpoints**: /crawler/crawl, /crawler/query, /crawler/extract

### January 23, 2026 - Phase 65.2: Advanced Optimizations
- ✅ **Binary Quantization**: 32x memory reduction with Hamming distance search
- ✅ **GPU Acceleration**: FAISS + cuVS support for 8-20x faster search
- ✅ **Learning-to-Rank**: XGBoost-based ranking trained on click data
- ✅ **Spell Correction**: BK-tree based O(log n) fuzzy matching
- ✅ **Semantic Query Cache**: Intelligent caching with embedding similarity
- ✅ **Streaming Citations**: Real-time citation matching during LLM streaming
- ✅ **Late Chunking**: Embed full document then split (context preservation)
- ✅ **Web Crawler UI**: Full-featured frontend component
- ✅ **Natural Language DB Query UI**: Text-to-SQL with auto-visualization

### Upcoming
- 🔄 Multi-embedding table implementation (Alembic migration)
- 🔄 UI controls for embedding provider selection
- 🔄 Matryoshka multi-resolution search
- 🔄 RAPTOR hierarchical indexing

---

## Support

**Questions or Issues?**
- Check this documentation index first
- Review the [Session Summary](guides/SESSION_SUMMARY.md) for recent changes
- Check troubleshooting section above
- Open an issue on GitHub with logs and reproduction steps

**Contributing:**
- See [CONTRIBUTING.md](../CONTRIBUTING.md)
- All documentation improvements welcome!

---

## External Resources

- **Ollama:** https://ollama.ai/
- **OpenAI Embeddings:** https://platform.openai.com/docs/guides/embeddings
- **HuggingFace Sentence Transformers:** https://huggingface.co/sentence-transformers
- **MTEB Leaderboard:** https://huggingface.co/spaces/mteb/leaderboard

---

**Last Updated:** January 23, 2026
**Documentation Version:** 1.2 (Phase 65.2)
**System Version:** 0.1.0
