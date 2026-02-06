# Scalability Analysis

## Can This System Handle Millions of Documents?

**Short answer:** The current single-machine architecture handles up to ~500K documents well. Millions require infrastructure changes (mostly swapping storage backends), not major code rewrites.

## Current Capacity Estimates

```mermaid
graph LR
    subgraph "Current Architecture"
        A[Single Machine<br/>16GB RAM]
    end

    subgraph "Capacity"
        B[500K Documents]
        C[5M Chunks]
        D[~4GB Vector Index]
        E[~50 QPS]
    end

    A --> B
    A --> C
    A --> D
    A --> E
```

| Metric | Current Limit | Bottleneck |
|--------|--------------|------------|
| Documents | ~500K | Vector index RAM |
| Chunks | ~5M | ChromaDB HNSW memory |
| Concurrent queries | ~50 QPS | Single FastAPI process |
| Query latency (p50) | ~2s | LLM inference |
| Query latency (p99) | ~8s | Complex RAG + verification |
| Index build time | ~2hr/100K docs | Embedding generation |

## Bottleneck Analysis

```mermaid
flowchart TD
    subgraph "Critical Bottlenecks"
        VS[Vector Store<br/>ChromaDB in-memory<br/>RAM-limited]
        BM[BM25 Index<br/>PostgreSQL FTS<br/>table scan at scale]
    end

    subgraph "Major Bottlenecks"
        QE[Query Expansion<br/>3× LLM calls per query]
        VER[Self-RAG Verification<br/>10× LLM calls per query]
        RL[LLM Rate Limits<br/>API provider throttling]
    end

    subgraph "Medium Bottlenecks"
        SM[Session Memory<br/>in-process RAM]
        KG_T[KG Traversal<br/>2-hop on large graphs]
        CC[Context Assembly<br/>chunk concatenation]
    end

    VS -->|Solution| PGV[PGVector +<br/>HNSW disk index]
    BM -->|Solution| ES[Elasticsearch /<br/>OpenSearch]
    QE -->|Solution| QC[Query cache +<br/>batch LLM]
    VER -->|Solution| CE[Cross-encoder<br/>lightweight model]
    RL -->|Solution| LB_LLM[Multi-provider<br/>load balancing]
    SM -->|Solution| RED[Redis for<br/>session store]
```

## Scaling Tiers

### Tier 1: Up to 500K Documents (Current)

```
Single machine, 16-32GB RAM
├── PostgreSQL + pgvector
├── ChromaDB (in-process)
├── Ollama (local GPU)
├── Redis (optional, in-memory fallback)
└── FastAPI (single process, multiple workers)
```

**No changes needed.** This is the current architecture.

### Tier 2: 500K - 5M Documents

```mermaid
graph TB
    subgraph "App Tier (2-4 instances)"
        F1[FastAPI Worker 1]
        F2[FastAPI Worker 2]
        F3[FastAPI Worker 3]
    end

    subgraph "Search Tier"
        PGV[(PGVector<br/>dedicated<br/>+ read replicas)]
        ES[(Elasticsearch<br/>BM25 + FTS)]
    end

    subgraph "AI Tier"
        OL1[Ollama GPU 1<br/>embedding]
        OL2[Ollama GPU 2<br/>chat]
    end

    subgraph "Cache Tier"
        RED[(Redis Cluster<br/>sessions + cache)]
    end

    F1 --> PGV
    F2 --> PGV
    F3 --> PGV
    F1 --> ES
    F2 --> ES
    F3 --> ES
    F1 --> RED
    F2 --> RED
    F3 --> RED
    F1 --> OL1
    F1 --> OL2
```

**Required changes:**
1. **Replace ChromaDB with PGVector HNSW** — disk-based index, shared across workers
2. **Add Elasticsearch** for BM25 — distributed full-text search
3. **Redis cluster** for session memory and semantic cache
4. **Multiple FastAPI workers** behind load balancer
5. **Separate embedding GPU** from chat GPU

**Code changes:** Minimal — `vectorstore_local.py` already abstracts the vector store interface. Add Elasticsearch adapter for `hybrid_retriever.py`.

### Tier 3: 5M - 50M Documents

```mermaid
graph TB
    subgraph "Edge / CDN"
        CDN[Static assets<br/>+ API cache]
    end

    subgraph "App Cluster"
        LB[Load Balancer]
        F1[FastAPI Pod 1]
        F2[FastAPI Pod 2]
        F3[FastAPI Pod N]
    end

    subgraph "Vector DB Cluster"
        QD1[Qdrant Shard 1]
        QD2[Qdrant Shard 2]
        QD3[Qdrant Shard N]
    end

    subgraph "Search Cluster"
        ES1[OpenSearch Node 1]
        ES2[OpenSearch Node 2]
        ES3[OpenSearch Node N]
    end

    subgraph "DB Cluster"
        PG_W[(PostgreSQL<br/>Write Primary)]
        PG_R1[(Read Replica 1)]
        PG_R2[(Read Replica 2)]
    end

    subgraph "AI Cluster"
        VLLM[vLLM Cluster<br/>batch inference]
        EMB_SVC[Embedding Service<br/>TEI / Infinity]
    end

    CDN --> LB
    LB --> F1
    LB --> F2
    LB --> F3
    F1 --> QD1
    F2 --> QD2
    F3 --> QD3
    F1 --> ES1
    F2 --> ES2
    F1 --> PG_W
    F2 --> PG_R1
    F3 --> PG_R2
    F1 --> VLLM
    F2 --> EMB_SVC
```

**Required changes:**
1. **Qdrant or Weaviate** — distributed vector DB with automatic sharding
2. **OpenSearch serverless** — managed full-text search
3. **vLLM** — batched LLM inference (10-50× throughput)
4. **Text Embedding Inference (TEI)** — dedicated embedding service
5. **Collection-based sharding** — partition by document collection
6. **Read replicas** for PostgreSQL

### Tier 4: 50M+ Documents (Enterprise)

Same as Tier 3 plus:
- **Multi-region deployment** with geo-routing
- **Tiered storage** — hot (SSD) / warm (HDD) / cold (S3) vectors
- **Async indexing pipeline** with Celery + RabbitMQ
- **Approximate pre-filtering** before ANN search
- **Quantized vectors** (int8/binary) for 4-8× memory reduction

## Performance Optimization Roadmap

```mermaid
gantt
    title Scaling Roadmap
    dateFormat  YYYY-MM-DD
    section Quick Wins
    Quantized embeddings (4× less RAM)    :a1, 2024-01-01, 7d
    Query result caching (Redis)          :a2, 2024-01-01, 5d
    Batch embedding generation            :a3, 2024-01-08, 5d
    section Medium Effort
    PGVector HNSW migration               :b1, 2024-01-15, 14d
    Elasticsearch for BM25                :b2, 2024-01-15, 14d
    Redis session memory                  :b3, 2024-01-29, 7d
    section Major Effort
    Qdrant distributed vector DB          :c1, 2024-02-12, 21d
    vLLM batch inference                  :c2, 2024-02-12, 14d
    Collection-based sharding             :c3, 2024-03-05, 21d
```

## Cost Estimates (Cloud Deployment)

| Scale | Infrastructure | Monthly Cost (est.) |
|-------|---------------|-------------------|
| 100K docs | 1× 32GB server + 1× GPU | $200-400 |
| 500K docs | 2× 32GB + managed PG + 1× GPU | $500-1,000 |
| 2M docs | 4× servers + PG cluster + 2× GPU | $2,000-4,000 |
| 10M docs | K8s cluster + Qdrant + ES + vLLM | $8,000-15,000 |
| 50M docs | Multi-region + enterprise DBs | $30,000-60,000 |

## What Works Well at Scale Already

The current architecture has several scale-friendly design decisions:

1. **Async everywhere** — SQLAlchemy async, async LLM calls, no blocking I/O
2. **Batch settings loading** — All 30+ settings in one DB query, cached
3. **Semantic caching** — Avoids full pipeline for repeated queries
4. **Adaptive routing** — Simple queries skip expensive operations
5. **Token budgeting** — Dynamic allocation prevents context overflow
6. **Circuit breaker** — Graceful degradation when LLM providers fail
7. **Content-type penalties** — Skip low-value chunks early in pipeline
8. **Lazy initialization** — Services created on first use, not startup
9. **Abstract interfaces** — VectorStore, EmbeddingProvider, LLMProvider all pluggable
