# Document Lifecycle

![Document Lifecycle](images/document-lifecycle.png)

## Overview

Documents go through a multi-stage lifecycle from upload to queryable state.

```mermaid
stateDiagram-v2
    [*] --> Uploaded : Upload API / Connector Sync
    Uploaded --> Parsing : Auto-triggered
    Parsing --> Chunking : Text extracted
    Chunking --> Embedding : Chunks created
    Embedding --> Indexed : Vectors stored
    Indexed --> Enhanced : Optional
    Enhanced --> KG_Extracted : Optional
    KG_Extracted --> Queryable
    Indexed --> Queryable : Skip enhancement

    Parsing --> Failed : Parse error
    Embedding --> Failed : Embed error
    Failed --> Parsing : Retry

    Queryable --> Deleted : Soft delete
    Deleted --> Queryable : Restore
    Deleted --> [*] : Hard delete
```

## Stage 1: Upload / Ingest

### Local Upload

**File:** `backend/api/routes/documents.py` — POST `/documents`

```mermaid
flowchart TD
    FILE[File Upload<br/>multipart/form-data] --> VALIDATE[Validate<br/>size, type, duplicates]
    VALIDATE --> SAVE[Save to<br/>./storage/documents/uuid/]
    SAVE --> CREATE_DB[Create Document record<br/>status=pending]
    CREATE_DB --> CAPTURE[Capture source metadata<br/>IP, user-agent, method]
    CAPTURE --> TRIGGER[Trigger pipeline<br/>async processing]
```

Source metadata captured:
```json
{
  "upload_method": "web_upload",
  "uploaded_by": "user-uuid",
  "uploaded_at": "2024-01-15T10:30:00Z",
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "original_filename": "report.pdf"
}
```

### Connector Sync

**File:** `backend/services/connectors/scheduler.py`

```mermaid
flowchart TD
    SCHED[Scheduler<br/>periodic / manual] --> DISCOVER[List resources<br/>from external source]
    DISCOVER --> DIFF[Compare with<br/>SyncedResource table]
    DIFF --> NEW{New or<br/>Modified?}
    NEW -->|Yes| DOWNLOAD[Download content]
    NEW -->|No| SKIP[Skip]

    DOWNLOAD --> MODE{Storage<br/>Mode?}
    MODE -->|Download| STORE[Save file locally<br/>is_stored_locally=true]
    MODE -->|Process Only| PROCESS[Process & delete file<br/>is_stored_locally=false]

    STORE --> CREATE[Create Document<br/>+ SyncedResource records]
    PROCESS --> CREATE
```

Storage mode can be set:
1. **Global default:** Admin > Settings > Ingestion > Default Storage Mode
2. **Per-connector override:** Connector settings > Storage Mode
3. **Priority:** Per-connector > Global > "download" (default)

## Stage 2: Document Parsing

**File:** `backend/services/document_parser.py`

```mermaid
flowchart TD
    DOC[Document File] --> TYPE{File Type?}

    TYPE -->|PDF| PDF_PARSE[PyPDF2 + pdfplumber<br/>OCR fallback for scanned]
    TYPE -->|DOCX| DOCX_PARSE[python-docx<br/>paragraphs + tables]
    TYPE -->|PPTX| PPTX_PARSE[python-pptx<br/>slides + notes]
    TYPE -->|TXT/MD| TEXT_PARSE[Direct read<br/>encoding detection]
    TYPE -->|HTML| HTML_PARSE[BeautifulSoup<br/>clean text extraction]
    TYPE -->|CSV/XLSX| TABLE_PARSE[pandas<br/>row-based extraction]
    TYPE -->|Images| VISION_PARSE[Vision model<br/>OCR + description]

    PDF_PARSE --> CLEAN[Clean & normalize<br/>whitespace, encoding]
    DOCX_PARSE --> CLEAN
    PPTX_PARSE --> CLEAN
    TEXT_PARSE --> CLEAN
    HTML_PARSE --> CLEAN
    TABLE_PARSE --> CLEAN
    VISION_PARSE --> CLEAN

    CLEAN --> META[Extract metadata<br/>page count, word count,<br/>language, structure]
    META --> TEXT[Structured text output<br/>with page boundaries]
```

## Stage 3: Chunking

**File:** `backend/services/chunking.py`

```mermaid
flowchart TD
    TEXT[Extracted Text] --> STRATEGY{Chunking<br/>Strategy?}

    STRATEGY -->|Semantic| SEM[Semantic Chunking<br/>split by meaning shifts]
    STRATEGY -->|Fixed| FIXED[Fixed Size<br/>with overlap]
    STRATEGY -->|Recursive| REC[Recursive Character<br/>paragraph → sentence → char]

    SEM --> CHUNKS[Chunks<br/>~500-1500 chars each]
    FIXED --> CHUNKS
    REC --> CHUNKS

    CHUNKS --> TAG[Tag each chunk:<br/>content_type, page_num,<br/>section, index]
    TAG --> CLASSIFY_TYPE[Classify content type:<br/>prose, table, list,<br/>heading, code, image_caption]
    CLASSIFY_TYPE --> DB_SAVE[Save to Chunk table]
```

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Target characters per chunk |
| `chunk_overlap` | 200 | Overlap between adjacent chunks |
| `min_chunk_size` | 100 | Minimum chunk size (avoid tiny fragments) |
| `strategy` | recursive | semantic / fixed / recursive |

## Stage 4: Embedding

**File:** `backend/services/embeddings.py`

```mermaid
flowchart TD
    CHUNKS[Document Chunks] --> BATCH[Batch chunks<br/>up to 100 per request]
    BATCH --> PROVIDER{Embedding<br/>Provider?}

    PROVIDER -->|Ollama| OLLAMA[nomic-embed-text<br/>768 dimensions]
    PROVIDER -->|OpenAI| OPENAI[text-embedding-3-small<br/>1536 dimensions]
    PROVIDER -->|Voyage| VOYAGE[voyage-3<br/>1024 dimensions]
    PROVIDER -->|Gemini| GEMINI[text-embedding-004<br/>768 dimensions]

    OLLAMA --> VECTORS[Vector arrays]
    OPENAI --> VECTORS
    VOYAGE --> VECTORS
    GEMINI --> VECTORS

    VECTORS --> STORE[Store in vector DB<br/>ChromaDB / PGVector]
    STORE --> INDEX[Update HNSW index<br/>ef=200, M=32]
```

## Stage 5: Enhancement (Optional)

**File:** `backend/api/routes/admin.py` — POST `/admin/enhance-documents/{doc_id}`

```mermaid
flowchart TD
    DOC[Document] --> LLM[LLM Analysis]

    LLM --> SUMMARY[Summary<br/>short + detailed]
    LLM --> KEYWORDS[Keywords<br/>5-15 terms]
    LLM --> TOPICS[Topics<br/>3-8 categories]
    LLM --> ENTITIES[Named Entities<br/>people, orgs, locations]
    LLM --> QUESTIONS[Hypothetical Questions<br/>5-10 that this doc answers]

    SUMMARY --> META[Store in<br/>enhanced_metadata JSON]
    KEYWORDS --> META
    TOPICS --> META
    ENTITIES --> META
    QUESTIONS --> META

    META --> FLAG[Set is_enhanced=true<br/>enhanced_at=now]
```

Enhancement metadata structure:
```json
{
  "summary_short": "Brief 1-2 sentence summary",
  "summary_detailed": "Detailed multi-paragraph summary",
  "keywords": ["keyword1", "keyword2"],
  "topics": ["Topic A", "Topic B"],
  "entities": {
    "people": ["John Doe"],
    "organizations": ["ACME Corp"],
    "locations": ["New York"]
  },
  "hypothetical_questions": [
    "What are the main findings of this report?"
  ],
  "language": "en",
  "document_type": "research_paper",
  "enhanced_at": "2024-01-15T10:30:00Z",
  "model_used": "llama3.2:latest"
}
```

## Stage 6: Knowledge Graph Extraction (Optional)

**File:** `backend/services/knowledge_graph.py`

```mermaid
flowchart TD
    DOC[Document Chunks] --> NER[Named Entity<br/>Recognition]
    NER --> ENTITIES[Entities<br/>PERSON, ORG, LOCATION,<br/>CONCEPT, TECHNOLOGY...]
    ENTITIES --> RE[Relation<br/>Extraction]
    RE --> RELATIONS[Relations<br/>WORKS_FOR, LOCATED_IN,<br/>PART_OF, USES...]
    RELATIONS --> DB[Store in<br/>Entity + EntityRelation<br/>+ EntityMention tables]
    DB --> GRAPH[Knowledge Graph<br/>available for retrieval]
```

## Document Source Tracking

```mermaid
flowchart LR
    subgraph "Local Upload"
        L1[is_stored_locally = true]
        L2[source_type = local_upload]
        L3[file on disk]
    end

    subgraph "Connector - Download Mode"
        C1[is_stored_locally = true]
        C2[source_type = google_drive]
        C3[file on disk + source_url]
    end

    subgraph "Connector - Process Only"
        P1[is_stored_locally = false]
        P2[source_type = notion]
        P3[source_url only, no file]
    end

    P3 -->|Import Copy| C3
```

### Preview Behavior by Storage Mode

| Storage | File Preview | External Preview | Import Copy |
|---------|-------------|-----------------|-------------|
| Local (file on disk) | Native PDF/image/DOCX viewer | N/A | N/A |
| External (link only) | N/A | iframe attempt, then placeholder | Downloads & stores locally |
| External (Google Drive) | N/A | Google Docs Viewer embed | Downloads & stores locally |
