# RAG Query Pipeline — End-to-End

![RAG Pipeline Flow](images/rag-pipeline.png)

This document traces every step from user query to final answer.

## Pipeline Overview

```mermaid
flowchart TD
    START([User sends query]) --> CACHE{Semantic<br/>Cache Hit?}

    CACHE -->|Hit| RETURN_CACHED[Return cached response]
    CACHE -->|Miss| SPELL[Spell Correction<br/>Phase 65]

    SPELL --> CLASSIFY[Query Classification<br/>intent + complexity]

    CLASSIFY --> ROUTE{Adaptive<br/>Router}

    ROUTE -->|Simple| DENSE_ONLY[Dense Vector<br/>Search Only]
    ROUTE -->|Standard| HYBRID_SEARCH[Hybrid Search<br/>Dense + BM25]
    ROUTE -->|Complex| ADVANCED[Advanced RAG<br/>Fusion + HyDE + KG]

    DENSE_ONLY --> MERGE[Merge & Rank<br/>RRF Fusion]
    HYBRID_SEARCH --> MERGE
    ADVANCED --> MERGE

    MERGE --> VERIFY{Self-RAG<br/>Verification?}

    VERIFY -->|Yes| CHECK[Verify chunk<br/>relevance + grounding]
    VERIFY -->|No| ASSEMBLE

    CHECK --> FILTER[Filter low-quality<br/>chunks]
    FILTER --> ASSEMBLE[Assemble Context<br/>+ Memory + KG]

    ASSEMBLE --> SELECT_PROMPT[Select Prompt<br/>Template]
    SELECT_PROMPT --> INVOKE_LLM[Invoke LLM<br/>with context]

    INVOKE_LLM --> PARSE[Parse Response<br/>citations + questions]
    PARSE --> SCORE[Confidence<br/>Scoring]
    SCORE --> SAVE[Save to Cache<br/>+ DB + Memory]
    SAVE --> RESPOND([Return RAGResponse])

    style START fill:#4CAF50,color:white
    style RESPOND fill:#4CAF50,color:white
    style RETURN_CACHED fill:#FF9800,color:white
```

## Step 1: Chat Endpoint

**File:** `backend/api/routes/chat.py` — `create_chat_completion()`

```mermaid
flowchart LR
    REQ[ChatRequest] --> MODE{mode?}
    MODE -->|agent| AGENT[Agent Orchestrator]
    MODE -->|general| GENERAL[Direct LLM<br/>no retrieval]
    MODE -->|vision| VISION[Vision Analysis]
    MODE -->|chat/default| RAG[RAG Service]
```

The endpoint receives a `ChatRequest` with:
- `message` — the user's question
- `session_id` — conversation continuity
- `mode` — routing: `chat` (RAG), `general` (no docs), `agent` (multi-step), `vision` (images)
- `collection_filter` / `folder_id` — scope which documents to search
- `intelligence_level` — basic / standard / enhanced / maximum
- `temperature_override` — per-request creativity control
- `enable_cot`, `enable_verification`, `ensemble_voting` — intelligence toggles

## Step 2: Semantic Cache Check

**File:** `backend/services/semantic_cache.py`

```mermaid
flowchart TD
    Q[Query] --> EMB[Generate query<br/>embedding]
    EMB --> FAISS[FAISS ANN search<br/>against cached queries]
    FAISS --> SIM{Similarity<br/>> 0.92?}
    SIM -->|Yes| HIT[Cache HIT<br/>return stored answer]
    SIM -->|No| MISS[Cache MISS<br/>continue pipeline]
```

- **Index:** FAISS in-memory with up to 1,000 cached query embeddings
- **Matching:** Dual-threshold — precision (0.95) first, then recall (0.85)
- **TTL:** 300 seconds (5 minutes)
- **Benefit:** Avoids full RAG pipeline for repeated/similar questions

## Step 3: Spell Correction (Phase 65)

**File:** `backend/services/text_preprocessor.py`

Corrects typos in the query before retrieval. Has safety guards:
- **Word overlap check:** If corrected version shares <50% words with original, revert
- **Character similarity:** If <60% similar by edit distance, revert
- **Reason:** Prevents catastrophic corrections (e.g., "of them" → "too the java")

## Step 4: Query Classification

**File:** `backend/services/query_classifier.py`

```mermaid
flowchart LR
    Q[Query] --> CLS[Classifier]
    CLS --> INT[Intent]
    CLS --> CONF[Confidence]
    CLS --> HINTS[Retrieval Hints]

    INT --> S[SUMMARY]
    INT --> C[COMPARISON]
    INT --> L[LIST]
    INT --> A[ANALYTICAL]
    INT --> F[FACTUAL]

    HINTS --> TK[suggested_top_k]
    HINTS --> MMR[use_mmr]
    HINTS --> COT[use_cot]
    HINTS --> KGH[use_kg_enhancement]
    HINTS --> TPL[prompt_template]
```

Classification drives downstream behavior:
- **LIST queries** → higher top_k (15), list-specific prompt template
- **COMPARISON queries** → retrieve from multiple docs, comparison template
- **SUMMARY queries** → broader retrieval, summary template
- **FACTUAL queries** → precise retrieval, direct answer template

## Step 5: Adaptive Routing (Phase 66)

**File:** `backend/services/adaptive_router.py`

```mermaid
flowchart TD
    Q[Query + Classification] --> ANALYZE[Complexity Analysis]

    ANALYZE --> SIMPLE{Simple?}
    SIMPLE -->|Yes| FAST[DENSE_ONLY<br/>top_k=5, no reranking]

    SIMPLE -->|No| MODERATE{Moderate?}
    MODERATE -->|Yes| STD[HYBRID<br/>top_k=10, BM25+dense]

    MODERATE -->|No| COMPLEX_PATH[FUSION<br/>top_k=20, all engines]

    COMPLEX_PATH --> HYDE[+ HyDE<br/>hypothetical doc embedding]
    COMPLEX_PATH --> RAGF[+ RAG-Fusion<br/>multi-query variations]
    COMPLEX_PATH --> SB[+ Step-Back<br/>abstract → specific]
```

| Strategy | When Used | Latency | Quality |
|----------|-----------|---------|---------|
| `DENSE_ONLY` | Short factual queries | ~50ms | Good |
| `HYBRID` | Standard queries | ~100ms | Better |
| `FUSION` | Complex multi-part queries | ~300ms | Best |
| `TWO_STAGE` | High-precision needs | ~200ms | Very Good |
| `HIERARCHICAL` | Document-spanning queries | ~250ms | Best for long docs |

## Step 6: Document Retrieval

![Hybrid Retrieval](images/hybrid-retrieval.png)

**File:** `backend/services/hybrid_retriever.py`

```mermaid
flowchart TD
    subgraph "Stage 1: Dense Search"
        QE[Query Embedding] --> HNSW[HNSW ANN Search<br/>ChromaDB / PGVector]
        HNSW --> DENSE_RESULTS[Top-15 by cosine sim]
    end

    subgraph "Stage 2: Sparse Search"
        QT[Query Tokens] --> BM25_IDX[BM25 Index<br/>PostgreSQL FTS]
        BM25_IDX --> SPARSE_RESULTS[Top-15 by BM25 score]
    end

    subgraph "Stage 3: Optional Advanced"
        COLBERT[ColBERT/WARP<br/>Late Interaction]
        LIGHTRAG[LightRAG<br/>Dual-Level]
        RAPTOR[RAPTOR<br/>Tree-Organized]
    end

    subgraph "Stage 4: Fusion"
        DENSE_RESULTS --> RRF[Reciprocal Rank Fusion<br/>RRF score = Σ 1/k+rank]
        SPARSE_RESULTS --> RRF
        COLBERT --> RRF
        LIGHTRAG --> RRF
        RAPTOR --> RRF
        RRF --> DEDUP[Deduplicate by chunk_id]
        DEDUP --> TOP_K[Return Top-K<br/>default: 10]
    end
```

### RRF Fusion Formula

```
RRF(document) = Σ  1 / (k + rank_i(document))
                i∈sources
```

Where `k=30` (dampening). This is position-based, not score-based, making it robust across different score distributions.

### Query Expansion (if enabled)

```mermaid
flowchart LR
    Q[Original Query] --> LLM[LLM generates<br/>2-3 variations]
    LLM --> V1["'ML benefits'"]
    LLM --> V2["'machine learning advantages'"]
    LLM --> V3["'why use artificial intelligence'"]
    V1 --> SEARCH[Each variation<br/>searched separately]
    V2 --> SEARCH
    V3 --> SEARCH
    SEARCH --> FUSE[Results fused<br/>with RRF]
```

### HyDE (Hypothetical Document Embeddings)

For abstract queries (< 5 words), the LLM generates a hypothetical document that would answer the query. The embedding of this synthetic document is used as an additional search vector.

## Step 7: Post-Retrieval Processing

```mermaid
flowchart TD
    CHUNKS[Retrieved Chunks] --> KG_ENH{KG Enhancement<br/>enabled?}

    KG_ENH -->|Yes| KG_TRAVERSE[Traverse entity graph<br/>2-hop neighbors]
    KG_ENH -->|No| VERIFY_CHECK

    KG_TRAVERSE --> ADD_KG[Add KG-related<br/>chunks to context]
    ADD_KG --> VERIFY_CHECK

    VERIFY_CHECK{Self-RAG<br/>Verification?}
    VERIFY_CHECK -->|Yes| VERIFY_EACH[For each chunk:<br/>relevance? grounding?]
    VERIFY_CHECK -->|No| DEDUP_CHECK

    VERIFY_EACH --> FILTER_LOW[Remove chunks<br/>scoring < threshold]
    FILTER_LOW --> DEDUP_CHECK

    DEDUP_CHECK{Semantic<br/>Dedup?}
    DEDUP_CHECK -->|Yes| REMOVE_DUPS[Remove >95%<br/>similar chunks]
    DEDUP_CHECK -->|No| REORDER

    REMOVE_DUPS --> REORDER[Context Reordering<br/>sandwich strategy]
    REORDER --> FINAL[Final Context<br/>ready for LLM]
```

### Content-Type Penalties

Certain chunk types are penalized during scoring:
- **Glossary/reference:** -30% score (usually not directly answering questions)
- **Table of contents:** -50% score
- **Image credits:** -40% score
- **Hypothetical question chunks:** filtered at vector store level

## Step 8: Conversation Memory

**File:** `backend/services/session_memory.py`

```mermaid
flowchart TD
    SESSION[Session ID] --> REHYDRATE{DB Rehydration<br/>enabled?}

    REHYDRATE -->|Yes| LOAD_DB[Load last K messages<br/>from ChatMessage table]
    REHYDRATE -->|No| EMPTY[Start with<br/>empty memory]

    LOAD_DB --> TIER{Model Size<br/>Tier?}
    EMPTY --> TIER

    TIER -->|Tiny ≤3B| K3[k=3 turns<br/>history: 10% budget]
    TIER -->|Small 3-9B| K6[k=6 turns<br/>history: 15% budget]
    TIER -->|Medium 9-34B| K10[k=10 turns<br/>history: 15% budget]
    TIER -->|Large >34B| K15[k=15 turns<br/>history: 15% budget]

    K3 --> REWRITE
    K6 --> REWRITE
    K10 --> REWRITE
    K15 --> REWRITE

    REWRITE{Follow-up<br/>question?}
    REWRITE -->|Yes, Small model| HEURISTIC[Heuristic rewrite<br/>pronoun replacement]
    REWRITE -->|Yes, Large model| LLM_REWRITE[LLM rewrite<br/>standalone question]
    REWRITE -->|No| PASS[Use original query]
```

### Token Budget Allocation

```
┌─────────────────────────────────────────────────┐
│              Context Window (e.g., 8000 tokens)  │
├──────────┬──────────┬──────────────┬────────────┤
│ System   │ History  │ Chunks       │ Generation │
│ 10-15%   │ 10-15%  │ 55-60%       │ 15-20%     │
│ ~1000    │ ~1000   │ ~4500        │ ~1500      │
└──────────┴──────────┴──────────────┴────────────┘
```

## Step 9: LLM Invocation

**File:** `backend/services/llm.py`

```mermaid
flowchart TD
    CTX[Context + Question<br/>+ History + System Prompt] --> SELECT[Select Provider<br/>+ Model]

    SELECT --> CONFIG[Apply Sampling Config<br/>temp, top_p, top_k]

    CONFIG --> CIRCUIT{Circuit Breaker<br/>Status?}
    CIRCUIT -->|Open| FALLBACK[Use fallback provider]
    CIRCUIT -->|Closed/Half| INVOKE[Invoke LLM]

    INVOKE --> TIMEOUT{Response<br/>within 120s?}
    TIMEOUT -->|Yes| PARSE_RESP[Parse response]
    TIMEOUT -->|No| RETRY{Retries<br/>remaining?}

    RETRY -->|Yes| BACKOFF[Exponential backoff<br/>+ jitter]
    RETRY -->|No| ERROR[Return error]

    BACKOFF --> INVOKE
    PARSE_RESP --> DONE[LLM Response]
```

### Model-Specific Configurations

| Model | Context Window | Temperature | top_p | Notes |
|-------|---------------|-------------|-------|-------|
| llama3.2:latest | 8,000 | 0.3 | 0.7 | Quote-first prompting |
| llama3.1:latest | 8,000 | 0.5 | 0.8 | Standard |
| mistral:latest | 8,000 | 0.5 | 0.8 | Standard |
| qwen2.5:latest | 32,000 | 0.7 | 0.95 | Higher top_p for stability |
| gpt-4 | 128,000 | 0.7 | 0.9 | Full context |
| claude-3-opus | 200,000 | 0.7 | 0.9 | Largest context |

### Prompt Template Selection

Based on query classification:

| Intent | Template | Key Instructions |
|--------|----------|-----------------|
| FACTUAL | `RAG_PROMPT_TEMPLATE` | "Direct answer first, cite [Source N]" |
| SUMMARY | `SUMMARY_TEMPLATE` | "2-3 sentence summary, then key points" |
| COMPARISON | `COMPARISON_TEMPLATE` | "Structured comparison with citations" |
| LIST | `LIST_TEMPLATE` | "Enumerate ALL items, cite each" |
| ANALYTICAL | `ANALYTICAL_TEMPLATE` | "Evidence → analysis → conclusion" |

## Step 10: Response Assembly

```mermaid
flowchart TD
    RAW[Raw LLM Output] --> STRIP[Strip preambles<br/>"Sure, I'll help:"]

    STRIP --> EXTRACT_Q[Extract suggested<br/>questions from<br/>SUGGESTED_QUESTIONS: q1|q2|q3]

    EXTRACT_Q --> NORMALIZE[Normalize citations<br/>[source 1] → [Source 1]]

    NORMALIZE --> CONFIDENCE[Calculate confidence<br/>relevance × grounding]

    CONFIDENCE --> SUFFICIENCY[Context sufficiency<br/>check: coverage score]

    SUFFICIENCY --> SOURCES[Assemble source<br/>citations with snippets]

    SOURCES --> CACHE_SAVE[Save to semantic<br/>cache + response cache]

    CACHE_SAVE --> DB_SAVE[Persist to DB<br/>ChatMessage table]

    DB_SAVE --> MEM_SAVE[Update session<br/>memory buffer]

    MEM_SAVE --> RESPONSE[Return RAGResponse<br/>content + sources +<br/>confidence + suggestions]
```

### RAGResponse Structure

```json
{
  "content": "The answer text with [Source 1] citations...",
  "sources": [
    {
      "document_id": "uuid",
      "document_name": "report.pdf",
      "chunk_id": "uuid",
      "page_number": 5,
      "snippet": "First 200 chars of relevant chunk...",
      "relevance_score": 0.85,
      "similarity_score": 0.72
    }
  ],
  "confidence_score": 0.82,
  "confidence_level": "high",
  "suggested_questions": ["Follow-up 1?", "Follow-up 2?"],
  "context_sufficiency": {
    "is_sufficient": true,
    "coverage_score": 0.88,
    "has_conflicts": false,
    "missing_aspects": []
  }
}
```

## Intelligence Levels

```mermaid
flowchart LR
    subgraph "Basic"
        B1[Dense search only]
        B2[No verification]
        B3[No expansion]
    end

    subgraph "Standard"
        S1[Hybrid search]
        S2[Basic verification]
        S3[No expansion]
    end

    subgraph "Enhanced"
        E1[Hybrid + KG]
        E2[Full verification]
        E3[Query expansion]
        E4[Chain-of-thought]
    end

    subgraph "Maximum"
        M1[All engines]
        M2[Ensemble voting]
        M3[Extended thinking]
        M4[Multi-query fusion]
    end
```

| Level | Features Enabled | Latency | Best For |
|-------|-----------------|---------|----------|
| Basic | Dense search, direct answer | ~1s | Quick lookups |
| Standard | Hybrid search, basic verify | ~2s | General Q&A |
| Enhanced | + KG, expansion, CoT, verify | ~4s | Complex research |
| Maximum | + Ensemble, extended thinking | ~8s | Critical decisions |

## Configuration Reference

All settings are stored in the database and can be changed at runtime via Admin > Settings.

### Retrieval Settings

| Setting Key | Default | Description |
|-------------|---------|-------------|
| `rag.top_k` | 10 | Chunks to retrieve per query |
| `rag.similarity_threshold` | 0.40 | Minimum cosine similarity |
| `rag.use_hybrid_search` | true | Enable BM25 + dense fusion |
| `rag.query_expansion_enabled` | false | Expand queries with LLM |
| `rag.hyde_enabled` | false | Hypothetical document embeddings |
| `rag.knowledge_graph_enabled` | false | KG-enhanced retrieval |
| `rag.verification_enabled` | true | Self-RAG chunk verification |
| `rag.rrf_k` | 30 | RRF dampening parameter |

### Cache Settings

| Setting Key | Default | Description |
|-------------|---------|-------------|
| `cache.enabled` | true | Enable semantic cache |
| `cache.max_entries` | 1000 | Max cached queries |
| `cache.default_ttl_seconds` | 300 | Cache entry lifetime |
| `cache.threshold_mode` | adaptive | precision/recall/adaptive |

### Conversation Settings

| Setting Key | Default | Description |
|-------------|---------|-------------|
| `conversation.query_rewriting_enabled` | true | Rewrite follow-ups |
| `conversation.db_rehydration_enabled` | true | Restore history from DB |
| `conversation.max_history_turns` | 15 | Max turns to keep |
