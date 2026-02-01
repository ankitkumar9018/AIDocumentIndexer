# AIDocumentIndexer - Complete Code Reference

## Purpose

This document provides a **complete code-level reference** for developers to understand:
1. **User Flows** - Step-by-step journey through the application
2. **Function Call Chains** - Which function calls what, in what order
3. **All Classes & Functions** - Detailed signatures and purposes
4. **Data Transformations** - How data flows through the system

---

## Table of Contents

1. [User Flow: Document Upload](#user-flow-document-upload)
2. [User Flow: RAG Chat Query](#user-flow-rag-chat-query)
3. [User Flow: Document Generation](#user-flow-document-generation)
4. [User Flow: Agent Mode](#user-flow-agent-mode)
5. [Complete Class Reference](#complete-class-reference)
6. [Complete Function Reference](#complete-function-reference)
7. [Database Operations Reference](#database-operations-reference)
8. [Phase 84-96 Service Reference](#phase-84-96-service-reference)

---

## User Flow: Document Upload

### Overview
User uploads a file → File is processed → Text extracted → Chunks created → Embeddings generated → Stored in vector database

### Step-by-Step Call Chain

```
USER ACTION: Clicks "Upload" and selects a file

┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: frontend/app/dashboard/upload/page.tsx                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. User selects file(s)                                                     │
│ 2. handleUpload() called                                                    │
│    └─> uploadDocument(file, options)  [lib/api/client.ts:uploadDocument]   │
│        └─> POST /api/v1/upload/single                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BACKEND API: backend/api/routes/upload.py                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ @router.post("/single")                                                     │
│ async def upload_single_file(                                               │
│     file: UploadFile,                                                       │
│     collection: str = None,                                                 │
│     access_tier: int = 1,                                                   │
│     user: User = Depends(get_current_user)                                  │
│ ):                                                                          │
│     # 1. Validate file type and size                                        │
│     validate_file(file)                                                     │
│                                                                             │
│     # 2. Check for duplicates                                               │
│     file_hash = compute_file_hash(file)                                     │
│     if await check_duplicate(file_hash):                                    │
│         return {"status": "duplicate", ...}                                 │
│                                                                             │
│     # 3. Save file to disk                                                  │
│     file_path = save_uploaded_file(file)                                    │
│                                                                             │
│     # 4. Create document record                                             │
│     document = await create_document_record(                                │
│         filename=file.filename,                                             │
│         file_path=file_path,                                                │
│         file_hash=file_hash,                                                │
│         access_tier_id=access_tier_id,                                      │
│         user_id=user.id                                                     │
│     )                                                                       │
│                                                                             │
│     # 5. Queue for processing (async)                                       │
│     await queue_document_processing(document.id)                            │
│                                                                             │
│     return UploadResponse(file_id=document.id, status="processing")         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BACKGROUND TASK: backend/tasks/document_tasks.py                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ async def process_document(document_id: str):                               │
│     """Main document processing pipeline"""                                 │
│                                                                             │
│     # 1. Load document record                                               │
│     document = await get_document(document_id)                              │
│     update_status(document, "processing")                                   │
│                                                                             │
│     # 2. Extract content using UniversalProcessor                           │
│     processor = UniversalProcessor(                                         │
│         enable_ocr=True,                                                    │
│         enable_image_analysis=True                                          │
│     )                                                                       │
│     ──────────────────────────────────────────────────────────────────────  │
│     │ CALLS: backend/processors/universal.py                               │
│     │                                                                       │
│     │ extracted = processor.process(file_path, mode="full")                │
│     │   └─> _process_pdf() / _process_docx() / _process_pptx() etc.       │
│     │       └─> For PDFs with images: _ocr_pages_parallel()                │
│     │           └─> ThreadPoolExecutor with 4 workers                      │
│     │           └─> _ocr_image() per page                                  │
│     │               └─> PaddleOCR.ocr() or Tesseract fallback             │
│     │                                                                       │
│     │ Returns: ExtractedContent(                                           │
│     │     text="...",                                                      │
│     │     metadata={...},                                                  │
│     │     pages=[{page_num, text, images}...],                             │
│     │     images=[ExtractedImage...],                                      │
│     │     word_count=N,                                                    │
│     │     page_count=N                                                     │
│     │ )                                                                    │
│     ──────────────────────────────────────────────────────────────────────  │
│                                                                             │
│     # 3. Chunk the content                                                  │
│     chunker = DocumentChunker(config=ChunkingConfig(                        │
│         chunk_size=1000,                                                    │
│         chunk_overlap=200,                                                  │
│         strategy="recursive"                                                │
│     ))                                                                      │
│     ──────────────────────────────────────────────────────────────────────  │
│     │ CALLS: backend/processors/chunker.py                                 │
│     │                                                                       │
│     │ chunks = chunker.chunk_with_pages(extracted.pages)                   │
│     │   └─> detect_document_type() if adaptive_chunking enabled            │
│     │   └─> _chunk_code() / _chunk_html() / chunk() based on type         │
│     │   └─> For large docs (>100k chars): chunk_hierarchical()             │
│     │       └─> Creates 3 levels: document → section → detail             │
│     │                                                                       │
│     │ Returns: List[Chunk(                                                 │
│     │     content="...",                                                   │
│     │     chunk_index=N,                                                   │
│     │     page_number=N,                                                   │
│     │     chunk_hash="...",                                                │
│     │     is_summary=False,                                                │
│     │     chunk_level=0  # 0=detail, 1=section, 2=document                 │
│     │ )]                                                                   │
│     ──────────────────────────────────────────────────────────────────────  │
│                                                                             │
│     # 4. Generate embeddings                                                │
│     embedding_service = get_embedding_service(provider="openai")            │
│     ──────────────────────────────────────────────────────────────────────  │
│     │ CALLS: backend/services/embeddings.py                                │
│     │                                                                       │
│     │ # If Ray available, use parallel processing                          │
│     │ if RayEmbeddingService available:                                    │
│     │     results = ray_service.embed_chunks_parallel(chunks)              │
│     │       └─> Distributes to N workers                                   │
│     │       └─> Each worker: embed_batch_ray() remote function             │
│     │ else:                                                                │
│     │     results = embedding_service.embed_chunks(chunks)                 │
│     │       └─> embed_texts() with batching                                │
│     │       └─> OpenAIEmbeddings.embed_documents()                         │
│     │                                                                       │
│     │ Returns: List[EmbeddingResult(                                       │
│     │     chunk_id="...",                                                  │
│     │     embedding=[0.1, 0.2, ...],  # 1536 dimensions                    │
│     │     model="text-embedding-3-small"                                   │
│     │ )]                                                                   │
│     ──────────────────────────────────────────────────────────────────────  │
│                                                                             │
│     # 5. Store in vector database                                           │
│     vector_store = get_vector_store()                                       │
│     ──────────────────────────────────────────────────────────────────────  │
│     │ CALLS: backend/services/vectorstore.py                               │
│     │                                                                       │
│     │ chunk_ids = await vector_store.add_chunks(                           │
│     │     chunks=[{                                                        │
│     │         "content": chunk.content,                                    │
│     │         "embedding": embedding,                                      │
│     │         "metadata": {...}                                            │
│     │     }...],                                                           │
│     │     document_id=document.id,                                         │
│     │     access_tier_id=document.access_tier_id                           │
│     │ )                                                                    │
│     │   └─> INSERT INTO chunks (content, embedding, ...)                   │
│     │   └─> pgvector stores embedding as Vector(1536)                      │
│     ──────────────────────────────────────────────────────────────────────  │
│                                                                             │
│     # 6. Update document status                                             │
│     await update_document(document_id, status="completed")                  │
│                                                                             │
│     # 7. Send WebSocket notification                                        │
│     await notify_upload_complete(document_id, user_id)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ WEBSOCKET: backend/api/websocket.py                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ await manager.send_to_user(                                                 │
│     user_id=user_id,                                                        │
│     message={                                                               │
│         "type": "document_processed",                                       │
│         "document_id": document_id,                                         │
│         "status": "completed"                                               │
│     }                                                                       │
│ )                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: Receives WebSocket message                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ useWebSocket hook receives message                                          │
│   └─> Updates document list via React Query invalidation                    │
│   └─> Shows success toast notification                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Transformations

```
File (binary)
    ↓ UniversalProcessor.process()
ExtractedContent { text, pages[], images[], metadata }
    ↓ DocumentChunker.chunk_with_pages()
List[Chunk] { content, chunk_index, page_number, metadata }
    ↓ EmbeddingService.embed_chunks()
List[Chunk + embedding[1536]]
    ↓ VectorStore.add_chunks()
PostgreSQL chunks table with pgvector
```

---

## User Flow: RAG Chat Query

### Overview
User asks question → Query processed → Documents retrieved → LLM generates answer → Response with sources returned

### Step-by-Step Call Chain

```
USER ACTION: Types question in chat and hits Enter

┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: frontend/app/dashboard/chat/page.tsx                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. User types message and submits                                           │
│ 2. handleSendMessage() called                                               │
│    └─> sendChatMessage(request)  [lib/api/client.ts]                       │
│        request = {                                                          │
│            message: "What are the key findings?",                           │
│            session_id: "uuid",                                              │
│            mode: "chat",           // or "agent", "general"                │
│            collection_filter: "Research",                                   │
│            include_sources: true,                                           │
│            language: "auto"        // auto-detect or "en", "de", etc.      │
│        }                                                                    │
│        └─> POST /api/v1/chat/completions                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BACKEND API: backend/api/routes/chat.py                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ @router.post("/completions")                                                │
│ async def chat_completions(                                                 │
│     request: ChatRequest,                                                   │
│     user: User = Depends(get_current_user)                                  │
│ ):                                                                          │
│     # 1. Determine chat mode                                                │
│     if request.mode == "general":                                           │
│         → Route to GeneralChatService (no RAG)                              │
│     elif request.mode == "agent":                                           │
│         → Route to AgentOrchestrator                                        │
│     else:  # "chat" mode - RAG                                              │
│         → Route to RAGService                                               │
│                                                                             │
│     # 2. Get or create session                                              │
│     session_id = request.session_id or create_session(user.id)              │
│                                                                             │
│     # 3. Call RAG service                                                   │
│     rag_service = get_rag_service()                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAG SERVICE: backend/services/rag.py                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ class RAGService:                                                           │
│                                                                             │
│ async def query(                                                            │
│     self,                                                                   │
│     question: str,                                                          │
│     session_id: str = None,                                                 │
│     collection_filter: str = None,                                          │
│     access_tier: int = 100,                                                 │
│     language: str = "en"                                                    │
│ ) -> RAGResponse:                                                           │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 1: QUERY CLASSIFICATION                                          │
│     # ═══════════════════════════════════════════════════════════════════   │
│     if self.config.enable_dynamic_weighting:                                │
│         classifier = get_query_classifier()                                 │
│         ────────────────────────────────────────────────────────────────    │
│         │ CALLS: backend/services/query_classifier.py                      │
│         │                                                                   │
│         │ classification = await classifier.classify(question)             │
│         │   └─> Uses LLM to detect: factual, analytical, exploratory      │
│         │   └─> Adjusts vector_weight vs keyword_weight                    │
│         │                                                                   │
│         │ Returns: QueryClassification(                                    │
│         │     intent=QueryIntent.FACTUAL,                                  │
│         │     vector_weight=0.7,                                           │
│         │     keyword_weight=0.3                                           │
│         │ )                                                                │
│         ────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 2: QUERY EXPANSION (Optional)                                    │
│     # ═══════════════════════════════════════════════════════════════════   │
│     expanded_queries = [question]                                           │
│     if self.config.enable_query_expansion:                                  │
│         expander = get_query_expander()                                     │
│         ────────────────────────────────────────────────────────────────    │
│         │ CALLS: backend/services/query_expander.py                        │
│         │                                                                   │
│         │ expanded = await expander.expand(question, count=3)              │
│         │   └─> Uses LLM to generate query variations                      │
│         │   └─> Adds synonyms, related terms                               │
│         │                                                                   │
│         │ Returns: ["original query", "variation 1", "variation 2"]        │
│         ────────────────────────────────────────────────────────────────    │
│         expanded_queries = expanded                                         │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 3: HYDE - Hypothetical Document Embeddings (Optional)            │
│     # ═══════════════════════════════════════════════════════════════════   │
│     if self.config.enable_hyde and len(question.split()) < 5:               │
│         hyde = get_hyde_expander()                                          │
│         ────────────────────────────────────────────────────────────────    │
│         │ CALLS: backend/services/hyde.py                                  │
│         │                                                                   │
│         │ hypothetical_doc = await hyde.generate(question)                 │
│         │   └─> LLM generates a hypothetical answer document               │
│         │   └─> This doc is embedded instead of short query                │
│         │                                                                   │
│         │ Returns: "Based on the analysis, the key findings include..."    │
│         ────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 4: GENERATE QUERY EMBEDDING                                      │
│     # ═══════════════════════════════════════════════════════════════════   │
│     embedding_service = EmbeddingService(provider="openai")                 │
│     query_embedding = await embedding_service.embed_query(question)         │
│       └─> OpenAIEmbeddings.embed_query()                                   │
│       └─> Returns: List[float] of 1536 dimensions                          │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 5: HYBRID SEARCH (Vector + Keyword + Enhanced)                   │
│     # ═══════════════════════════════════════════════════════════════════   │
│     vector_store = get_vector_store()                                       │
│     ────────────────────────────────────────────────────────────────────    │
│     │ CALLS: backend/services/vectorstore.py                               │
│     │                                                                       │
│     │ results = await vector_store.hybrid_search(                          │
│     │     query=question,                                                  │
│     │     query_embedding=query_embedding,                                 │
│     │     top_k=10,                                                        │
│     │     access_tier_level=access_tier,                                   │
│     │     vector_weight=0.7,                                               │
│     │     keyword_weight=0.3                                               │
│     │ )                                                                    │
│     │                                                                       │
│     │ INTERNALLY:                                                          │
│     │ ┌─────────────────────────────────────────────────────────────────┐  │
│     │ │ 1. VECTOR SEARCH (similarity_search)                            │  │
│     │ │    SELECT *, 1 - (embedding <=> query_embedding) as score       │  │
│     │ │    FROM chunks                                                  │  │
│     │ │    WHERE access_tier_level <= user_tier                         │  │
│     │ │    ORDER BY embedding <=> query_embedding                       │  │
│     │ │    LIMIT top_k * 2                                              │  │
│     │ │                                                                 │  │
│     │ │ 2. KEYWORD SEARCH (keyword_search)                              │  │
│     │ │    SELECT *, ts_rank(to_tsvector(content), query) as score      │  │
│     │ │    FROM chunks                                                  │  │
│     │ │    WHERE to_tsvector(content) @@ plainto_tsquery(query)         │  │
│     │ │    LIMIT top_k * 2                                              │  │
│     │ │                                                                 │  │
│     │ │ 3. ENHANCED METADATA SEARCH (if enabled)                        │  │
│     │ │    Searches document summaries, keywords, hypothetical Qs       │  │
│     │ │                                                                 │  │
│     │ │ 4. RECIPROCAL RANK FUSION (RRF)                                 │  │
│     │ │    Combines all results with weighted scoring:                  │  │
│     │ │    score = Σ (weight / (rank + k)) for each search type         │  │
│     │ └─────────────────────────────────────────────────────────────────┘  │
│     │                                                                       │
│     │ Returns: List[SearchResult(                                          │
│     │     chunk_id="...",                                                  │
│     │     content="...",                                                   │
│     │     score=0.85,              # RRF combined score                    │
│     │     similarity_score=0.92,   # Original vector similarity            │
│     │     document_id="...",                                               │
│     │     document_name="...",                                             │
│     │     page_number=5                                                    │
│     │ )]                                                                   │
│     ────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 6: RERANKING (Optional - ColBERT or Cross-Encoder)               │
│     # ═══════════════════════════════════════════════════════════════════   │
│     if self.config.rerank_results:                                          │
│         ────────────────────────────────────────────────────────────────    │
│         │ CALLS: backend/services/colbert_reranker.py                      │
│         │                                                                   │
│         │ results = await vector_store._rerank_with_colbert(               │
│         │     query=question,                                              │
│         │     results=results,                                             │
│         │     top_k=top_k                                                  │
│         │ )                                                                │
│         │   └─> ColBERT late-interaction scoring                           │
│         │   └─> More accurate than vector similarity alone                 │
│         ────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 7: CONTEXT EXPANSION (Get surrounding chunks)                    │
│     # ═══════════════════════════════════════════════════════════════════   │
│     results = await vector_store.expand_context(results)                    │
│       └─> Adds prev_chunk_snippet and next_chunk_snippet                   │
│       └─> Helps user navigate to adjacent content                          │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 8: BUILD CONTEXT FOR LLM                                         │
│     # ═══════════════════════════════════════════════════════════════════   │
│     context = self._build_context(results)                                  │
│     # Format: "Source 1 (doc_name, page 5):\n{content}\n\nSource 2:..."     │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 9: GET LLM & GENERATE RESPONSE                                   │
│     # ═══════════════════════════════════════════════════════════════════   │
│     llm, config = await self.get_llm_for_session(session_id)                │
│     ────────────────────────────────────────────────────────────────────    │
│     │ CALLS: backend/services/llm.py                                       │
│     │                                                                       │
│     │ EnhancedLLMFactory.get_chat_model_for_operation(                     │
│     │     operation="chat",                                                │
│     │     session_id=session_id                                            │
│     │ )                                                                    │
│     │   └─> LLMConfigManager.get_config_for_operation()                    │
│     │       └─> Check session override (per-session model)                 │
│     │       └─> Check operation config (admin-configured)                  │
│     │       └─> Check default provider                                     │
│     │       └─> Fallback to environment variables                          │
│     │   └─> Returns (ChatOpenAI instance, LLMConfigResult)                 │
│     ────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # Build prompt with language instruction                                │
│     system_prompt = RAG_SYSTEM_PROMPT                                       │
│     if language != "en":                                                    │
│         system_prompt += _get_language_instruction(language)                │
│                                                                             │
│     messages = [                                                            │
│         SystemMessage(content=system_prompt),                               │
│         *chat_history,  # From session memory                               │
│         HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")│
│     ]                                                                       │
│                                                                             │
│     # Generate response                                                     │
│     response = await llm.ainvoke(messages)                                  │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 10: VERIFICATION (Optional - Self-RAG)                           │
│     # ═══════════════════════════════════════════════════════════════════   │
│     if self.config.enable_verification:                                     │
│         verifier = RAGVerifier()                                            │
│         ────────────────────────────────────────────────────────────────    │
│         │ CALLS: backend/services/rag_verifier.py                          │
│         │                                                                   │
│         │ verification = await verifier.verify(                            │
│         │     question=question,                                           │
│         │     answer=response.content,                                     │
│         │     sources=results                                              │
│         │ )                                                                │
│         │   └─> Checks if answer is supported by sources                   │
│         │   └─> Calculates confidence score (0-1)                          │
│         │                                                                   │
│         │ Returns: VerificationResult(                                     │
│         │     is_supported=True,                                           │
│         │     confidence=0.85,                                             │
│         │     reasoning="Answer aligns with sources..."                    │
│         │ )                                                                │
│         ────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 11: CRAG - Corrective RAG (If low confidence)                    │
│     # ═══════════════════════════════════════════════════════════════════   │
│     if self.config.enable_crag and confidence < 0.5:                        │
│         crag = get_corrective_rag()                                         │
│         ────────────────────────────────────────────────────────────────    │
│         │ CALLS: backend/services/corrective_rag.py                        │
│         │                                                                   │
│         │ crag_result = await crag.refine_query(                           │
│         │     original_query=question,                                     │
│         │     original_results=results,                                    │
│         │     confidence=confidence                                        │
│         │ )                                                                │
│         │   └─> Reformulates query based on what was found                 │
│         │   └─> Re-searches with refined query                             │
│         │   └─> Regenerates answer                                         │
│         ────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 12: PARSE SUGGESTED QUESTIONS                                    │
│     # ═══════════════════════════════════════════════════════════════════   │
│     content, suggested_questions = _parse_suggested_questions(response)     │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 13: SAVE TO SESSION MEMORY                                       │
│     # ═══════════════════════════════════════════════════════════════════   │
│     if session_id:                                                          │
│         memory = self._get_memory(session_id)                               │
│         memory.save_context(                                                │
│             {"input": question},                                            │
│             {"output": content}                                             │
│         )                                                                   │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 14: TRACK USAGE                                                  │
│     # ═══════════════════════════════════════════════════════════════════   │
│     await LLMUsageTracker.log_usage(                                        │
│         provider_type=config.provider_type,                                 │
│         model=config.model,                                                 │
│         operation_type="rag_chat",                                          │
│         input_tokens=count_tokens(messages),                                │
│         output_tokens=count_tokens(response)                                │
│     )                                                                       │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # RETURN RESPONSE                                                       │
│     # ═══════════════════════════════════════════════════════════════════   │
│     return RAGResponse(                                                     │
│         content=content,                                                    │
│         sources=[Source(...) for r in results],                             │
│         query=question,                                                     │
│         model=config.model,                                                 │
│         confidence_score=confidence,                                        │
│         suggested_questions=suggested_questions                             │
│     )                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BACKEND API: backend/api/routes/chat.py (continued)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│     # Save messages to database                                             │
│     await save_chat_messages(                                               │
│         session_id=session_id,                                              │
│         user_message=request.message,                                       │
│         assistant_message=response.content,                                 │
│         sources=response.sources                                            │
│     )                                                                       │
│                                                                             │
│     return ChatResponse(                                                    │
│         session_id=session_id,                                              │
│         content=response.content,                                           │
│         sources=[...],                                                      │
│         confidence_score=response.confidence_score                          │
│     )                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: Receives and displays response                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Response received from API                                               │
│ 2. Message added to chat history state                                      │
│ 3. Sources displayed with expandable panels                                 │
│ 4. Confidence indicator shown (high/medium/low)                             │
│ 5. Suggested follow-up questions displayed as clickable chips               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Transformations

```
User Question (string)
    ↓ QueryClassifier.classify()
QueryClassification { intent, weights }
    ↓ QueryExpander.expand()
List[expanded_queries]
    ↓ EmbeddingService.embed_query()
query_embedding[1536]
    ↓ VectorStore.hybrid_search()
List[SearchResult] { chunk_id, content, score, document_info }
    ↓ ColBERTReranker.rerank()
List[SearchResult] (reordered by relevance)
    ↓ _build_context()
context_string (formatted sources)
    ↓ LLM.ainvoke(messages)
response_string
    ↓ RAGVerifier.verify()
VerificationResult { confidence, is_supported }
    ↓
RAGResponse { content, sources[], confidence, suggested_questions[] }
```

---

## User Flow: Document Generation

### Overview
User provides topic → Outline generated → User approves → Sections generated → Output file created (PPTX/DOCX/PDF)

### Step-by-Step Call Chain

```
USER ACTION: Enters topic and clicks "Generate"

┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: frontend/app/dashboard/create/page.tsx                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ State variables:                                                            │
│   - topic: string                                                           │
│   - outputFormat: "pptx" | "docx" | "pdf"                                   │
│   - theme: "business" | "creative" | "modern" | ...                        │
│   - language: "en" | "de" | "es" | ...                                     │
│   - enableAnimations: boolean                                               │
│   - slideCount: number                                                      │
│                                                                             │
│ handleGenerate():                                                           │
│   └─> createGenerationJob(request)  [lib/api/client.ts]                    │
│       request = {                                                           │
│           topic: "Marketing Plan for Q1",                                   │
│           output_format: "pptx",                                            │
│           collection_filter: "Marketing",                                   │
│           metadata: {                                                       │
│               theme: "business",                                            │
│               output_language: "en",                                        │
│               slide_count: 10,                                              │
│               enable_animations: true                                       │
│           }                                                                 │
│       }                                                                     │
│       └─> POST /api/v1/generate/jobs                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BACKEND API: backend/api/routes/generate.py                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ @router.post("/jobs")                                                       │
│ async def create_generation_job(                                            │
│     request: CreateJobRequest,                                              │
│     user: User = Depends(get_current_user)                                  │
│ ):                                                                          │
│     generator = get_generation_service()                                    │
│     job = await generator.create_job(                                       │
│         query=request.topic,                                                │
│         output_format=request.output_format,                                │
│         metadata=request.metadata                                           │
│     )                                                                       │
│     return {"job_id": job.id, "status": job.status}                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GENERATOR SERVICE: backend/services/generator.py                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ class DocumentGenerationService:                                            │
│                                                                             │
│ async def create_job(...) -> GenerationJob:                                 │
│     # 1. Create job record                                                  │
│     job = GenerationJob(                                                    │
│         id=str(uuid.uuid4()),                                               │
│         query=query,                                                        │
│         output_format=output_format,                                        │
│         status=GenerationStatus.PENDING,                                    │
│         metadata=metadata                                                   │
│     )                                                                       │
│     self._jobs[job.id] = job                                                │
│                                                                             │
│     # 2. Search for relevant source documents                               │
│     ────────────────────────────────────────────────────────────────────    │
│     │ CALLS: _search_sources() → RAGService                                │
│     │                                                                       │
│     │ sources = await self._search_sources(query, limit=20)                │
│     │   └─> rag_service.search(query, top_k=20)                            │
│     │   └─> Returns relevant document chunks                               │
│     │                                                                       │
│     │ Returns: List[SourceReference(                                       │
│     │     document_id="...",                                               │
│     │     title="Marketing Strategy 2024.pptx",                            │
│     │     relevance_score=0.85                                             │
│     │ )]                                                                   │
│     ────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     # 3. Generate document outline                                          │
│     ────────────────────────────────────────────────────────────────────    │
│     │ CALLS: _generate_outline_with_llm()                                  │
│     │                                                                       │
│     │ outline = await self._generate_outline_with_llm(query, sources)      │
│     │                                                                       │
│     │ PROMPT:                                                              │
│     │ """                                                                  │
│     │ Create a document outline for: {query}                               │
│     │                                                                       │
│     │ Available source material:                                           │
│     │ - {source1.title}: {source1.snippet}                                 │
│     │ - {source2.title}: {source2.snippet}                                 │
│     │ ...                                                                  │
│     │                                                                       │
│     │ Generate {slide_count} sections with titles and descriptions.        │
│     │ Output as JSON: {"sections": [{"title": "...", "description": "..."}]}│
│     │ """                                                                  │
│     │                                                                       │
│     │ llm = await EnhancedLLMFactory.get_chat_model_for_operation(         │
│     │     operation="content_generation"                                   │
│     │ )                                                                    │
│     │ response = await llm.ainvoke(messages)                               │
│     │ outline = parse_json(response.content)                               │
│     │                                                                       │
│     │ Returns: DocumentOutline(                                            │
│     │     sections=[                                                       │
│     │         Section(title="Executive Summary", description="..."),       │
│     │         Section(title="Market Analysis", description="..."),         │
│     │         Section(title="Strategy", description="..."),                │
│     │         ...                                                          │
│     │     ]                                                                │
│     │ )                                                                    │
│     ────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     job.outline = outline                                                   │
│     job.status = GenerationStatus.OUTLINE_READY                             │
│     return job                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: User reviews outline and approves                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Outline displayed with editable sections                                 │
│ 2. User can reorder, add, remove, or edit sections                          │
│ 3. User clicks "Approve & Generate"                                         │
│    └─> approveOutline(jobId, outline)                                      │
│        └─> POST /api/v1/generate/jobs/{id}/approve                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GENERATOR SERVICE: Content Generation                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ async def generate_content(job: GenerationJob) -> GenerationJob:            │
│     job.status = GenerationStatus.GENERATING                                │
│                                                                             │
│     # Get language setting                                                  │
│     output_language = job.metadata.get("output_language", "en")             │
│     language_name = LANGUAGE_NAMES.get(output_language, "English")          │
│                                                                             │
│     # Generate each section                                                 │
│     for i, section in enumerate(job.outline.sections):                      │
│         ────────────────────────────────────────────────────────────────    │
│         │ CALLS: _generate_section()                                       │
│         │                                                                   │
│         │ # 1. Search for section-specific sources                         │
│         │ # Query includes job title for better context matching           │
│         │ section_query = f"{job.title} - {section.title}"                 │
│         │ if section.description:                                          │
│         │     section_query += f": {section.description}"                  │
│         │ sources = await self._search_sources(section_query, max=5)       │
│         │                                                                   │
│         │ # 2. Build context from sources                                  │
│         │ source_context = self._build_source_context(sources)             │
│         │                                                                   │
│         │ # 3. Build cross-section context (what came before)              │
│         │ cross_context = self._build_cross_section_context(job, i)        │
│         │                                                                   │
│         │ # 3. Generate content with LLM                                   │
│         │ PROMPT:                                                          │
│         │ """                                                              │
│         │ LANGUAGE REQUIREMENT:                                            │
│         │ - Generate ALL content in {language_name}                        │
│         │ - Translate source material if in different language             │
│         │                                                                   │
│         │ Document: {job.title}                                            │
│         │ Section: {section.title}                                         │
│         │ Description: {section.description}                               │
│         │                                                                   │
│         │ Source material:                                                 │
│         │ {source_context}                                                 │
│         │                                                                   │
│         │ Previous sections context:                                       │
│         │ {cross_context}                                                  │
│         │                                                                   │
│         │ Write detailed content for this section.                         │
│         │ Use bullet points for key information.                           │
│         │ """                                                              │
│         │                                                                   │
│         │ content = await llm.ainvoke(messages)                            │
│         │                                                                   │
│         │ # 4. Filter LLM meta-text                                        │
│         │ content = filter_llm_metatext(content)                           │
│         │ content = filter_title_echo(content, section.title)              │
│         │                                                                   │
│         │ section.content = content                                        │
│         │ section.status = "generated"                                     │
│         ────────────────────────────────────────────────────────────────    │
│                                                                             │
│     job.status = GenerationStatus.CONTENT_READY                             │
│     return job                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GENERATOR SERVICE: Output File Generation                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ async def _generate_output_file(job: GenerationJob) -> str:                 │
│     if job.output_format == OutputFormat.PPTX:                              │
│         return await self._generate_pptx(job, filename)                     │
│     elif job.output_format == OutputFormat.DOCX:                            │
│         return await self._generate_docx(job, filename)                     │
│     elif job.output_format == OutputFormat.PDF:                             │
│         return await self._generate_pdf(job, filename)                      │
│                                                                             │
│ ════════════════════════════════════════════════════════════════════════    │
│ PPTX GENERATION (_generate_pptx):                                           │
│ ════════════════════════════════════════════════════════════════════════    │
│                                                                             │
│ from pptx import Presentation                                               │
│ from pptx.util import Inches, Pt                                            │
│ from pptx.dml.color import RGBColor                                         │
│                                                                             │
│ prs = Presentation()                                                        │
│ prs.slide_width = Inches(13.333)  # 16:9 widescreen                        │
│ prs.slide_height = Inches(7.5)                                              │
│                                                                             │
│ theme = get_theme_colors(job.metadata.get("theme", "business"))             │
│                                                                             │
│ # 1. Title Slide                                                            │
│ slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout         │
│ add_title_text(slide, job.title, theme)                                     │
│ add_subtitle(slide, job.subtitle, theme)                                    │
│                                                                             │
│ # 2. Table of Contents                                                      │
│ if job.metadata.get("include_toc", True):                                   │
│     slide = prs.slides.add_slide(prs.slide_layouts[6])                      │
│     add_toc_slide(slide, job.outline.sections, theme)                       │
│                                                                             │
│ # 3. Content Slides                                                         │
│ for section in job.outline.sections:                                        │
│     slide = prs.slides.add_slide(prs.slide_layouts[6])                      │
│                                                                             │
│     # Add title with theme styling                                          │
│     title_shape = add_slide_title(slide, section.title, theme)              │
│                                                                             │
│     # Parse content into bullets                                            │
│     bullets = parse_bullets(section.content)                                │
│                                                                             │
│     # Add bullet points with hierarchy                                      │
│     text_frame = add_text_frame(slide, theme)                               │
│     for bullet in bullets:                                                  │
│         level = bullet.get("level", 0)                                      │
│         text = bullet.get("text", "")                                       │
│         p = text_frame.add_paragraph()                                      │
│         p.text = text                                                       │
│         p.level = level                                                     │
│         p.font.size = Pt(18 - level * 2)                                    │
│         p.font.color.rgb = RGBColor.from_string(theme["text"])              │
│                                                                             │
│     # Add slide transitions if enabled                                      │
│     if job.metadata.get("enable_animations"):                               │
│         add_slide_transition(slide, "fade")                                 │
│                                                                             │
│ # 4. References Slide (if sources used)                                     │
│ if job.sources:                                                             │
│     slide = prs.slides.add_slide(prs.slide_layouts[6])                      │
│     add_references_slide(slide, job.sources, theme)                         │
│                                                                             │
│ # Save file                                                                 │
│ output_path = f"generated/{job.id}_{filename}.pptx"                         │
│ prs.save(output_path)                                                       │
│ return output_path                                                          │
│                                                                             │
│ ════════════════════════════════════════════════════════════════════════    │
│ DOCX GENERATION (_generate_docx):                                           │
│ ════════════════════════════════════════════════════════════════════════    │
│                                                                             │
│ from docx import Document                                                   │
│ from docx.shared import Inches, Pt, RGBColor                                │
│ from docx.enum.text import WD_ALIGN_PARAGRAPH                               │
│                                                                             │
│ doc = Document()                                                            │
│                                                                             │
│ # Title                                                                     │
│ title = doc.add_heading(job.title, 0)                                       │
│ title.alignment = WD_ALIGN_PARAGRAPH.CENTER                                 │
│                                                                             │
│ # Table of Contents                                                         │
│ doc.add_heading("Table of Contents", level=1)                               │
│ for i, section in enumerate(job.outline.sections):                          │
│     doc.add_paragraph(f"{i+1}. {section.title}")                            │
│ doc.add_page_break()                                                        │
│                                                                             │
│ # Content Sections                                                          │
│ for section in job.outline.sections:                                        │
│     doc.add_heading(section.title, level=1)                                 │
│     # Parse and add content with formatting                                 │
│     add_formatted_content(doc, section.content)                             │
│                                                                             │
│ output_path = f"generated/{job.id}_{filename}.docx"                         │
│ doc.save(output_path)                                                       │
│ return output_path                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: User downloads generated file                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Poll job status until "completed"                                        │
│ 2. Display download button                                                  │
│ 3. User clicks download                                                     │
│    └─> GET /api/v1/generate/jobs/{id}/download                             │
│        └─> Returns file as binary with Content-Disposition header          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User Flow: Agent Mode

### Overview
User asks complex question → Manager agent decomposes task → Worker agents execute → Results synthesized → Response returned

### Step-by-Step Call Chain

```
USER ACTION: Asks complex question with mode="agent"

┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: frontend/app/dashboard/chat/page.tsx                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ request = {                                                                 │
│     message: "Analyze our Q3 performance and suggest improvements",         │
│     mode: "agent",                                                          │
│     agent_options: {                                                        │
│         search_documents: true,                                             │
│         include_web_search: false,                                          │
│         require_approval: true,                                             │
│         max_steps: 5,                                                       │
│         language: "en"                                                      │
│     }                                                                       │
│ }                                                                           │
│ └─> POST /api/v1/chat/completions                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BACKEND API: backend/api/routes/chat.py                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ # Mode is "agent" - route to orchestrator                                   │
│ orchestrator = await create_orchestrator(db=db, rag_service=rag_service)    │
│                                                                             │
│ agent_context = {                                                           │
│     "collection_filter": request.collection_filter,                         │
│     "language": request.agent_options.language,                             │
│     "options": request.agent_options.model_dump()                           │
│ }                                                                           │
│                                                                             │
│ async for update in orchestrator.process_request(                           │
│     request=request.message,                                                │
│     session_id=session_id,                                                  │
│     user_id=user_id,                                                        │
│     context=agent_context                                                   │
│ ):                                                                          │
│     # Stream updates to frontend                                            │
│     yield update                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT ORCHESTRATOR: backend/services/agent_orchestrator.py                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ class AgentOrchestrator:                                                    │
│                                                                             │
│ async def process_request(                                                  │
│     self,                                                                   │
│     request: str,                                                           │
│     session_id: str,                                                        │
│     context: Dict                                                           │
│ ) -> AsyncGenerator[Dict, None]:                                            │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 1: MANAGER AGENT - Task Decomposition                            │
│     # ═══════════════════════════════════════════════════════════════════   │
│     manager = await self._get_manager_agent()                               │
│     ────────────────────────────────────────────────────────────────────    │
│     │ CALLS: backend/services/agents/manager_agent.py                      │
│     │                                                                       │
│     │ class ManagerAgent(BaseAgent):                                       │
│     │     """Decomposes complex tasks into subtasks"""                     │
│     │                                                                       │
│     │ decomposition = await manager.decompose_task(                        │
│     │     task=request,                                                    │
│     │     context=context                                                  │
│     │ )                                                                    │
│     │                                                                       │
│     │ PROMPT:                                                              │
│     │ """                                                                  │
│     │ You are a task planning agent. Decompose this request into          │
│     │ discrete, actionable subtasks.                                       │
│     │                                                                       │
│     │ Request: {request}                                                   │
│     │ Available tools: search_documents, analyze_data, synthesize         │
│     │                                                                       │
│     │ Output as JSON:                                                      │
│     │ {                                                                    │
│     │   "plan": [                                                          │
│     │     {"step": 1, "task": "...", "agent": "research", "deps": []},    │
│     │     {"step": 2, "task": "...", "agent": "generator", "deps": [1]}   │
│     │   ]                                                                  │
│     │ }                                                                    │
│     │ """                                                                  │
│     │                                                                       │
│     │ Returns: TaskPlan(                                                   │
│     │     steps=[                                                          │
│     │         AgentTask(                                                   │
│     │             id="task-1",                                             │
│     │             type=TaskType.RESEARCH,                                  │
│     │             name="Search Q3 reports",                                │
│     │             agent="research",                                        │
│     │             dependencies=[]                                          │
│     │         ),                                                           │
│     │         AgentTask(                                                   │
│     │             id="task-2",                                             │
│     │             type=TaskType.GENERATION,                                │
│     │             name="Analyze performance data",                         │
│     │             agent="generator",                                       │
│     │             dependencies=["task-1"]                                  │
│     │         ),                                                           │
│     │         AgentTask(                                                   │
│     │             id="task-3",                                             │
│     │             type=TaskType.EVALUATION,                                │
│     │             name="Generate improvement suggestions",                 │
│     │             agent="critic",                                          │
│     │             dependencies=["task-2"]                                  │
│     │         )                                                            │
│     │     ]                                                                │
│     │ )                                                                    │
│     ────────────────────────────────────────────────────────────────────    │
│                                                                             │
│     yield {"type": "plan", "data": decomposition.to_dict()}                 │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 2: APPROVAL (if required)                                        │
│     # ═══════════════════════════════════════════════════════════════════   │
│     if context.get("options", {}).get("require_approval"):                  │
│         yield {"type": "awaiting_approval", "plan": decomposition}          │
│         # Wait for user approval via separate endpoint                      │
│         # POST /api/v1/agent/approve/{execution_id}                         │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 3: EXECUTE TASKS IN DEPENDENCY ORDER                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     results = {}                                                            │
│     for task in self._topological_sort(decomposition.steps):                │
│                                                                             │
│         yield {"type": "task_start", "task_id": task.id, "name": task.name} │
│                                                                             │
│         # Get appropriate worker agent                                      │
│         agent = await self._get_worker_agent(task.agent)                    │
│                                                                             │
│         # Gather dependency results                                         │
│         dep_context = {                                                     │
│             dep_id: results[dep_id]                                         │
│             for dep_id in task.dependencies                                 │
│         }                                                                   │
│                                                                             │
│         # ───────────────────────────────────────────────────────────────   │
│         # RESEARCH AGENT                                                    │
│         # ───────────────────────────────────────────────────────────────   │
│         if task.type == TaskType.RESEARCH:                                  │
│             ────────────────────────────────────────────────────────────    │
│             │ CALLS: backend/services/agents/worker_agents.py              │
│             │                                                               │
│             │ class ResearchAgent(BaseAgent):                              │
│             │     """Retrieves information from documents"""               │
│             │                                                               │
│             │ result = await agent.execute(                                │
│             │     task=task,                                               │
│             │     context=dep_context                                      │
│             │ )                                                            │
│             │                                                               │
│             │ INTERNALLY:                                                  │
│             │ 1. Parse task to determine search queries                    │
│             │ 2. Call RAGService.search() for each query                   │
│             │ 3. Aggregate and deduplicate results                         │
│             │ 4. Format findings with citations                            │
│             │                                                               │
│             │ Returns: AgentResult(                                        │
│             │     task_id="task-1",                                        │
│             │     status=TaskStatus.COMPLETED,                             │
│             │     output={                                                 │
│             │         "findings": [...],                                   │
│             │         "sources": [...],                                    │
│             │         "summary": "..."                                     │
│             │     },                                                       │
│             │     reasoning_trace=["Searched for...", "Found..."]          │
│             │ )                                                            │
│             ────────────────────────────────────────────────────────────    │
│                                                                             │
│         # ───────────────────────────────────────────────────────────────   │
│         # GENERATOR AGENT                                                   │
│         # ───────────────────────────────────────────────────────────────   │
│         elif task.type == TaskType.GENERATION:                              │
│             ────────────────────────────────────────────────────────────    │
│             │ class GeneratorAgent(BaseAgent):                             │
│             │     """Creates content from prompts and context"""           │
│             │                                                               │
│             │ result = await agent.execute(                                │
│             │     task=task,                                               │
│             │     context=dep_context  # Includes research findings        │
│             │ )                                                            │
│             │                                                               │
│             │ INTERNALLY:                                                  │
│             │ 1. Build prompt with task description                        │
│             │ 2. Include context from dependencies                         │
│             │ 3. Add language instruction if not English                   │
│             │ 4. Generate content with LLM                                 │
│             │ 5. Validate output meets success criteria                    │
│             │                                                               │
│             │ Returns: AgentResult(                                        │
│             │     output={"content": "Analysis shows...", ...}             │
│             │ )                                                            │
│             ────────────────────────────────────────────────────────────    │
│                                                                             │
│         # ───────────────────────────────────────────────────────────────   │
│         # CRITIC AGENT                                                      │
│         # ───────────────────────────────────────────────────────────────   │
│         elif task.type == TaskType.EVALUATION:                              │
│             ────────────────────────────────────────────────────────────    │
│             │ class CriticAgent(BaseAgent):                                │
│             │     """Evaluates and improves content quality"""             │
│             │                                                               │
│             │ result = await agent.execute(                                │
│             │     task=task,                                               │
│             │     context=dep_context  # Includes generated content        │
│             │ )                                                            │
│             │                                                               │
│             │ INTERNALLY:                                                  │
│             │ 1. Review generated content for quality                      │
│             │ 2. Check factual accuracy against sources                    │
│             │ 3. Suggest improvements if needed                            │
│             │ 4. Score confidence level                                    │
│             │                                                               │
│             │ Returns: AgentResult(                                        │
│             │     output={                                                 │
│             │         "evaluation": "Good quality with minor issues",      │
│             │         "suggestions": ["Add more specifics about..."],      │
│             │         "score": 0.85                                        │
│             │     }                                                        │
│             │ )                                                            │
│             ────────────────────────────────────────────────────────────    │
│                                                                             │
│         results[task.id] = result                                           │
│         yield {                                                             │
│             "type": "task_complete",                                        │
│             "task_id": task.id,                                             │
│             "result": result.output                                         │
│         }                                                                   │
│                                                                             │
│     # ═══════════════════════════════════════════════════════════════════   │
│     # STEP 4: SYNTHESIZE FINAL RESPONSE                                     │
│     # ═══════════════════════════════════════════════════════════════════   │
│     final_response = await manager.synthesize_results(                      │
│         original_request=request,                                           │
│         task_results=results                                                │
│     )                                                                       │
│                                                                             │
│     yield {                                                                 │
│         "type": "final_response",                                           │
│         "content": final_response.content,                                  │
│         "sources": final_response.sources,                                  │
│         "reasoning_trace": final_response.reasoning_trace                   │
│     }                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Class Reference

### Database Models (`backend/db/models.py`)

```python
class AccessTier(Base, UUIDMixin, TimestampMixin):
    """Access tier for permission management (1-100 levels)"""
    __tablename__ = "access_tiers"

    name: str                    # e.g., "Public", "Internal", "Confidential"
    level: int                   # 1 (lowest) to 100 (highest)
    description: Optional[str]
    color: str                   # Hex color for UI display

    # Relationships
    users: List["User"]
    documents: List["Document"]
    chunks: List["Chunk"]


class User(Base, UUIDMixin, TimestampMixin):
    """User account model"""
    __tablename__ = "users"

    email: str                   # Unique email address
    password_hash: str           # Bcrypt hashed password
    name: Optional[str]
    is_active: bool              # Account status
    last_login_at: Optional[datetime]
    access_tier_id: UUID         # FK to AccessTier

    # Relationships
    access_tier: "AccessTier"
    documents: List["Document"]
    chat_sessions: List["ChatSession"]
    audit_logs: List["AuditLog"]


class Document(Base, UUIDMixin, TimestampMixin):
    """Uploaded document metadata"""
    __tablename__ = "documents"

    # File identification
    file_hash: str               # SHA-256 hash (unique)
    filename: str                # Display name
    original_filename: str       # Original upload name
    file_path: str               # Storage path
    file_type: str               # pdf, docx, pptx, etc.
    file_size: int               # Bytes
    mime_type: Optional[str]

    # Processing info
    processing_status: ProcessingStatus  # pending, processing, completed, failed
    processing_mode: ProcessingMode      # basic, ocr, full
    storage_mode: StorageMode            # rag, query_only
    processing_error: Optional[str]
    processed_at: Optional[datetime]

    # Metadata
    title: Optional[str]
    description: Optional[str]
    language: str                # ISO language code
    page_count: Optional[int]
    word_count: Optional[int]
    tags: Optional[List[str]]

    # Enhanced metadata (from LLM analysis)
    enhanced_metadata: Optional[dict]
    # Contains: summary_short, summary_detailed, keywords, topics,
    #           entities, hypothetical_questions, document_type

    # Foreign keys
    access_tier_id: UUID
    uploaded_by_id: Optional[UUID]
    folder_id: Optional[UUID]

    # Relationships
    chunks: List["Chunk"]


class Chunk(Base, UUIDMixin):
    """Document chunk with embedding vector"""
    __tablename__ = "chunks"

    content: str                 # Chunk text content
    content_hash: str            # MD5 hash for deduplication
    embedding: Optional[List[float]]  # pgvector Vector(1536)
    chunk_index: int             # Position in document
    page_number: Optional[int]
    section_title: Optional[str]
    token_count: Optional[int]
    char_count: Optional[int]

    # Hierarchical chunking
    is_summary: bool             # True for summary chunks
    chunk_level: int             # 0=detail, 1=section, 2=document
    parent_chunk_id: Optional[UUID]

    # Foreign keys
    document_id: UUID
    access_tier_id: UUID

    # Relationships
    document: "Document"
    access_tier: "AccessTier"
    parent_chunk: Optional["Chunk"]


class ChatSession(Base, UUIDMixin, TimestampMixin):
    """Chat conversation session"""
    __tablename__ = "chat_sessions"

    title: Optional[str]
    is_active: bool
    user_id: UUID

    # Relationships
    user: "User"
    messages: List["ChatMessage"]
    llm_override: Optional["ChatSessionLLMOverride"]


class ChatMessage(Base, UUIDMixin):
    """Individual chat message"""
    __tablename__ = "chat_messages"

    role: MessageRole            # user, assistant, system
    content: str
    source_document_ids: Optional[List[UUID]]
    source_chunks: Optional[dict]

    # Feedback
    is_helpful: Optional[bool]
    feedback: Optional[str]

    # LLM info
    model_used: Optional[str]
    tokens_used: Optional[int]
    latency_ms: Optional[int]

    # Foreign keys
    session_id: UUID
    created_at: datetime


class GenerationJob(Base, UUIDMixin, TimestampMixin):
    """Document generation job"""
    __tablename__ = "generation_jobs"

    query: str                   # User's topic/prompt
    output_format: str           # pptx, docx, pdf
    status: str                  # pending, processing, completed, failed
    outline_json: Optional[dict]
    sections_json: Optional[dict]
    output_file_path: Optional[str]
    error_message: Optional[str]
    metadata: Optional[dict]     # theme, language, slide_count, etc.

    # Foreign keys
    user_id: UUID
    collection_id: Optional[UUID]


class LLMProvider(Base, UUIDMixin, TimestampMixin):
    """LLM provider configuration"""
    __tablename__ = "llm_providers"

    name: str                    # Display name
    provider_type: str           # openai, anthropic, ollama
    api_key_encrypted: Optional[str]
    api_base_url: Optional[str]
    default_model: str
    default_temperature: float
    default_max_tokens: int
    is_active: bool
    is_default: bool
    health_check_at: Optional[datetime]
    health_status: Optional[str]


class LLMOperationConfig(Base, UUIDMixin, TimestampMixin):
    """Operation-specific LLM configuration"""
    __tablename__ = "llm_operation_configs"

    operation: str               # chat, generation, embedding, etc.
    provider_id: UUID            # FK to LLMProvider
    model_override: Optional[str]
    temperature_override: Optional[float]
    max_tokens_override: Optional[int]
    is_active: bool


class AgentConfig(Base, UUIDMixin, TimestampMixin):
    """Agent configuration stored in database"""
    __tablename__ = "agent_configs"

    agent_type: str              # manager, generator, critic, research
    name: str
    description: str
    system_prompt: str
    provider_id: Optional[UUID]
    model_override: Optional[str]
    temperature: float
    max_tokens: int
    tools: List[str]
    is_active: bool
```

### Service Classes

#### RAGService (`backend/services/rag.py`)

```python
class RAGService:
    """
    Retrieval-Augmented Generation service.

    Main entry point for document-based Q&A with:
    - Hybrid search (vector + keyword)
    - Query expansion and HyDE
    - Response verification (Self-RAG)
    - Corrective RAG for low confidence
    """

    def __init__(
        self,
        config: RAGConfig = None,      # Configuration options
        vector_store: VectorStore = None,
        session_id: str = None
    ):
        """Initialize RAG service with optional configuration"""

    async def query(
        self,
        question: str,                  # User's question
        session_id: str = None,         # For conversation memory
        collection_filter: str = None,  # Filter by collection/tag
        access_tier: int = 100,         # User's access level
        language: str = "en",           # Response language
        top_k: int = None,              # Number of results (default from config)
        folder_id: str = None,          # Folder scope
        include_subfolders: bool = True
    ) -> RAGResponse:
        """
        Execute RAG query and return response with sources.

        Returns:
            RAGResponse with content, sources, confidence, suggestions
        """

    async def query_stream(
        self,
        question: str,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream RAG response for real-time display"""

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_type: SearchType = SearchType.HYBRID,
        access_tier_level: int = 100,
        document_ids: List[str] = None
    ) -> List[SearchResult]:
        """Search documents without LLM generation"""

    def clear_memory(self, session_id: str):
        """Clear conversation memory for session"""
```

#### LLMFactory (`backend/services/llm.py`)

```python
class LLMFactory:
    """
    Factory for creating LLM instances with caching.

    Supports multiple providers via LiteLLM abstraction.
    """

    @classmethod
    def get_chat_model(
        cls,
        provider: str = None,           # openai, anthropic, ollama
        model: str = None,              # gpt-4o, claude-3-5-sonnet, etc.
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> BaseChatModel:
        """Get or create cached chat model instance"""

    @classmethod
    def get_embeddings(
        cls,
        provider: str = None,
        model: str = None
    ) -> Embeddings:
        """Get or create cached embeddings instance"""

    @classmethod
    def clear_cache(cls):
        """Clear all cached model instances"""


class EnhancedLLMFactory:
    """
    Database-driven LLM configuration with usage tracking.

    Priority: session override > operation config > default > environment
    """

    @classmethod
    async def get_chat_model_for_operation(
        cls,
        operation: str = "chat",        # chat, generation, embedding
        session_id: str = None,         # For per-session override
        user_id: str = None,            # For usage tracking
        track_usage: bool = True,
        enable_failover: bool = True,   # Auto-failover on health issues
        **kwargs
    ) -> Tuple[BaseChatModel, LLMConfigResult]:
        """
        Get chat model with database-driven configuration.

        Returns:
            Tuple of (model instance, configuration used)
        """


class LLMUsageTracker:
    """Track LLM usage and costs"""

    @staticmethod
    async def log_usage(
        provider_type: str,
        model: str,
        operation_type: str,
        input_tokens: int,
        output_tokens: int,
        provider_id: str = None,
        user_id: str = None,
        session_id: str = None,
        duration_ms: int = None,
        success: bool = True,
        error_message: str = None
    ) -> Optional[str]:
        """Log usage to database with cost calculation"""

    @staticmethod
    async def get_usage_summary(
        provider_id: str = None,
        user_id: str = None,
        operation_type: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Get aggregated usage statistics"""
```

#### EmbeddingService (`backend/services/embeddings.py`)

```python
class EmbeddingService:
    """
    Embedding generation with multiple providers.

    Features:
    - Caching to avoid re-embedding identical content
    - Batch processing for efficiency
    - Multiple providers (OpenAI, Ollama, HuggingFace)
    """

    DEFAULT_MODELS = {
        "openai": "text-embedding-3-small",    # 1536 dimensions
        "ollama": "nomic-embed-text",          # 768 dimensions
        "huggingface": "all-MiniLM-L6-v2"      # 384 dimensions
    }

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        config: LLMConfig = None
    ):
        """Initialize embedding service"""

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""

    def embed_texts(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with caching"""

    async def embed_query(self, text: str) -> List[float]:
        """Async embedding for query text"""

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = None,
        use_cache: bool = True
    ) -> List[EmbeddingResult]:
        """Generate embeddings for document chunks"""

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for current model"""


class RayEmbeddingService(EmbeddingService):
    """Distributed embedding using Ray parallel processing"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        num_workers: int = 4,
        batch_size_per_worker: int = 50
    ):
        """Initialize Ray embedding service"""

    def embed_texts_parallel(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Embed texts using Ray (falls back to ThreadPool if unavailable)"""
```

#### VectorStore (`backend/services/vectorstore.py`)

```python
class VectorStore:
    """
    Vector storage and retrieval using PostgreSQL + pgvector.

    Supports:
    - Vector similarity search (cosine distance)
    - Full-text keyword search (PostgreSQL ts_vector)
    - Hybrid search with Reciprocal Rank Fusion
    - ColBERT/cross-encoder reranking
    """

    def __init__(self, config: VectorStoreConfig = None):
        """Initialize vector store"""

    async def add_chunks(
        self,
        chunks: List[Dict],             # {content, embedding, metadata}
        document_id: str,
        access_tier_id: str,
        session: AsyncSession = None
    ) -> List[str]:
        """Add chunks with embeddings to store"""

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        access_tier_level: int = 100,
        document_ids: List[str] = None,
        similarity_threshold: float = None
    ) -> List[SearchResult]:
        """Vector similarity search using cosine distance"""

    async def keyword_search(
        self,
        query: str,
        top_k: int = None,
        access_tier_level: int = 100,
        document_ids: List[str] = None
    ) -> List[SearchResult]:
        """Full-text keyword search with AND/OR/NOT/phrase support"""

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = None,
        access_tier_level: int = 100,
        vector_weight: float = None,    # Default 0.7
        keyword_weight: float = None    # Default 0.3
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector + keyword + enhanced metadata.
        Uses Reciprocal Rank Fusion (RRF) for score combination.
        """

    async def expand_context(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Add prev/next chunk snippets for context navigation"""

    async def delete_document_chunks(
        self,
        document_id: str
    ) -> int:
        """Delete all chunks for a document"""
```

#### DocumentGenerationService (`backend/services/generator.py`)

```python
class DocumentGenerationService:
    """
    AI-powered document generation.

    Supports PPTX, DOCX, PDF output with:
    - RAG-based content from uploaded documents
    - 12+ themes with customizable colors
    - Multi-language output
    - Human-in-the-loop approval workflow
    """

    def __init__(self, config: GenerationConfig = None):
        """Initialize generation service"""

    async def create_job(
        self,
        query: str,                     # Topic/prompt
        output_format: OutputFormat,    # PPTX, DOCX, PDF
        documents: List[str] = None,    # Specific doc IDs to use
        metadata: Dict = None           # theme, language, slide_count, etc.
    ) -> GenerationJob:
        """Create new generation job with outline"""

    async def approve_outline(
        self,
        job: GenerationJob,
        outline: DocumentOutline
    ) -> GenerationJob:
        """Approve outline and proceed to content generation"""

    async def generate_content(
        self,
        job: GenerationJob
    ) -> GenerationJob:
        """Generate all section content"""

    async def revise_section(
        self,
        job: GenerationJob,
        section_index: int,
        feedback: str
    ) -> Section:
        """Regenerate section based on user feedback"""

    async def get_output_file(
        self,
        job_id: str
    ) -> Tuple[bytes, str, str]:
        """Get generated file (bytes, filename, content_type)"""
```

#### Agent Classes (`backend/services/agents/`)

```python
# === BASE CLASSES (agent_base.py) ===

@dataclass
class AgentConfig:
    """Configuration for agent instance"""
    agent_id: str
    name: str
    description: str
    provider_type: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: List[str] = field(default_factory=list)
    language: str = "en"


@dataclass
class AgentTask:
    """Structured task for agent execution"""
    id: str
    type: TaskType                      # GENERATION, EVALUATION, RESEARCH, etc.
    name: str
    description: str
    expected_inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    success_criteria: List[str]
    fallback_strategy: FallbackStrategy
    dependencies: List[str]             # Task IDs this depends on


@dataclass
class AgentResult:
    """Result from agent execution"""
    task_id: str
    agent_id: str
    status: TaskStatus                  # COMPLETED, FAILED, etc.
    output: Any
    reasoning_trace: List[str]          # Chain of thought
    tool_calls: List[Dict]
    tokens_used: int
    confidence_score: float


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name

    @abstractmethod
    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any] = None
    ) -> AgentResult:
        """Execute a task and return result"""

    async def _get_llm(self) -> BaseChatModel:
        """Get LLM instance for this agent"""

    def _build_messages(
        self,
        task: AgentTask,
        context: Dict
    ) -> List[BaseMessage]:
        """Build LangChain messages for task"""

    async def invoke_llm_with_tools(
        self,
        messages: List[Any],
        tools: List[Dict[str, Any]],
        record: bool = True,
        tool_choice: str = "auto",
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], int, int]:
        """
        Invoke LLM with tool/function calling support.

        Enables agents to dynamically select and call tools using LLM
        function calling. Compatible with OpenAI, Anthropic, and other
        providers that support tool calling.

        Args:
            messages: LangChain messages to send
            tools: List of tool schemas (OpenAI function calling format)
            record: Whether to record in trajectory
            tool_choice: "auto", "none", or specific tool name

        Returns:
            Tuple of (response_text, tool_calls, input_tokens, output_tokens)
            - response_text: Text response from LLM
            - tool_calls: List of tool calls [{id, name, arguments}]
            - input_tokens: Input token count
            - output_tokens: Output token count

        Example:
            tools = ToolRegistry.get_function_definitions()
            response, calls, in_tok, out_tok = await agent.invoke_llm_with_tools(
                messages=[HumanMessage(content="Search for climate data")],
                tools=tools,
            )
            if calls:
                for call in calls:
                    result = await executor.execute(call["name"], call["arguments"])
        """


# === WORKER AGENTS (worker_agents.py) ===

class ResearchAgent(BaseAgent):
    """
    Retrieves information from documents using RAG.

    Tools: search_documents, get_document_content
    """

    async def execute(self, task: AgentTask, context: Dict) -> AgentResult:
        """Search documents and aggregate findings"""


class GeneratorAgent(BaseAgent):
    """
    Creates content from prompts and context.

    Tools: generate_text, format_output
    """

    async def execute(self, task: AgentTask, context: Dict) -> AgentResult:
        """Generate content based on task and dependencies"""


class CriticAgent(BaseAgent):
    """
    Evaluates content quality and suggests improvements.

    Tools: evaluate_content, check_facts, suggest_improvements
    """

    async def execute(self, task: AgentTask, context: Dict) -> AgentResult:
        """Review content and provide evaluation"""


class ToolExecutionAgent(BaseAgent):
    """
    Executes file operations and exports.

    Tools: create_file, read_file, export_format
    """

    async def execute(self, task: AgentTask, context: Dict) -> AgentResult:
        """Execute tool operations"""


# === MANAGER AGENT (manager_agent.py) ===

class ManagerAgent(BaseAgent):
    """
    Orchestrates task decomposition and execution.

    Responsibilities:
    - Decompose complex requests into subtasks
    - Assign tasks to appropriate worker agents
    - Manage execution order based on dependencies
    - Execute independent steps in PARALLEL (via asyncio.gather)
    - Synthesize final response from results
    """

    async def decompose_task(
        self,
        task: str,
        context: Dict
    ) -> TaskPlan:
        """Break complex task into subtasks with dependencies"""

    async def synthesize_results(
        self,
        original_request: str,
        task_results: Dict[str, AgentResult]
    ) -> AgentResult:
        """Combine all task results into final response"""

    async def execute_with_streaming(
        self,
        user_request: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a user request with enhanced streaming support.

        Provides real-time progress updates for frontend integrations
        that need live feedback during execution.

        Args:
            user_request: The user's request to process
            session_id: Optional session ID
            user_id: Optional user ID
            context: Additional context

        Yields:
            Streaming events with types:
            - "planning": Plan is being created
            - "plan_ready": Plan created with step overview
            - "step_started": Step execution began
            - "step_progress": Intermediate progress (0-100%)
            - "content_chunk": Partial content as it's generated
            - "step_completed": Step finished successfully
            - "step_failed": Step failed with error
            - "sources": Document sources used
            - "synthesis_started": Final synthesis in progress
            - "complete": Final result ready
            - "error": Execution error

        Example:
            manager = ManagerAgent(config)
            async for event in manager.execute_with_streaming(
                "Create a report about climate change"
            ):
                if event["type"] == "step_started":
                    print(f"Starting: {event['step_name']}")
                elif event["type"] == "content_chunk":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "complete":
                    print(f"Done! Cost: ${event['total_cost_usd']:.4f}")
        """
```

#### Phase 84-96 Service Classes

##### BinaryQuantizationManager (`backend/services/rag.py`)

```python
class BinaryQuantizationManager:
    """
    INT8/binary vector quantization for 32x storage savings.

    Converts float32 embeddings to INT8 or binary representations,
    dramatically reducing memory and storage requirements while
    maintaining retrieval accuracy through a reranking step.
    """

    def quantize_vectors(
        self,
        vectors: List[List[float]],
        mode: str = "binary"           # "int8" or "binary"
    ) -> Any:
        """Quantize float32 vectors to INT8 or binary format"""

    def search_quantized(
        self,
        query_vector: List[float],
        top_k: int = 100
    ) -> List[Dict]:
        """Fast approximate search using quantized vectors"""

    def rerank_candidates(
        self,
        query_vector: List[float],
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """Rerank quantized search candidates with full-precision vectors"""
```

##### ColBERTRetriever (`backend/services/rag.py`)

```python
class ColBERTRetriever:
    """
    Late interaction retrieval with per-token matching (ColBERT).

    Implements MaxSim scoring between query and document token
    embeddings for higher accuracy than single-vector retrieval.
    Includes WARP engine for 3x faster multi-vector retrieval.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """Retrieve documents using ColBERT late interaction scoring"""
```

##### ContentFreshnessScorer (`backend/services/rag.py`)

```python
class ContentFreshnessScorer:
    """
    Time-decay scoring for document relevance.

    Applies freshness boosts to recently updated documents and
    staleness penalties to old documents. Configurable via settings:
    - freshness_decay_days: Days before freshness boost decays
    - freshness_boost_factor: Multiplier for fresh documents
    - freshness_penalty_factor: Multiplier for stale documents
    """

    def score(
        self,
        document_age_days: float,
        base_score: float
    ) -> float:
        """Apply freshness-adjusted score to a document"""
```

##### CircuitBreaker (`backend/services/rag.py`)

```python
class CircuitBreaker:
    """
    Fault tolerance for external service calls.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is down, requests fail fast
    - HALF_OPEN: Testing recovery, limited requests allowed

    Configurable failure thresholds and recovery timeouts prevent
    cascading failures when external services (LLM APIs, etc.) are
    unavailable.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1
    ):
        """Initialize circuit breaker with configurable thresholds"""

    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection"""
```

##### AgenticRAGEngine (`backend/services/rag.py`)

```python
class AgenticRAGEngine:
    """
    Multi-step reasoning with tool use for complex queries.

    Extends standard RAG with:
    - Agent routing: Determines if query needs agentic processing
    - Query decomposition: Breaks complex queries into sub-queries
    - Iterative retrieval: Multiple search rounds with refinement
    - Tool use: Integrates search, calculation, and other tools
    """

    async def process(
        self,
        query: str,
        context: Dict = None
    ) -> Dict:
        """Process query with agentic reasoning pipeline"""
```

##### CitationEngine (`backend/services/rag.py`)

```python
class CitationEngine:
    """
    Numbered source references in LLM responses.

    Injects [1], [2], [3] inline citation markers into generated
    responses, with verification that citations map to actual sources.
    Supports hover popovers showing source document name, page,
    similarity score, and snippet.
    """

    def inject_citations(
        self,
        response: str,
        sources: List[Dict]
    ) -> str:
        """Add numbered citation markers to response text"""

    def verify_citations(
        self,
        cited_response: str,
        sources: List[Dict]
    ) -> Dict:
        """Verify all citations map to valid sources"""
```

##### EmbeddingQuantizer (`backend/services/rag.py`)

```python
class EmbeddingQuantizer:
    """
    INT8/binary quantization for embeddings with Matryoshka support.

    Supports Matryoshka embeddings (variable-dimension embeddings)
    and quantization to INT8 or binary formats for reduced storage
    and faster similarity computation.
    """

    def quantize(
        self,
        embeddings: List[List[float]],
        target_format: str = "int8"    # "int8" or "binary"
    ) -> Any:
        """Quantize embeddings to target format"""
```

##### DSPyOptimizer (`backend/services/rag.py`)

```python
class DSPyOptimizer:
    """
    Automated prompt compilation using DSPy framework.

    Optimizes RAG prompts using demonstration-based compilation:
    - BootstrapFewShot: Few-shot example selection
    - MIPRO: Multi-instruction prompt optimization
    - BayesianSignatureOptimizer: Bayesian prompt tuning

    Settings:
    - dspy_optimization_enabled: Enable/disable optimization
    - dspy_default_optimizer: Which optimizer to use
    - dspy_min_examples: Minimum examples before optimization
    """

    async def optimize(
        self,
        module: Any,
        train_data: List[Dict],
        optimizer_name: str = "BootstrapFewShot"
    ) -> Any:
        """Compile/optimize a DSPy module with training data"""
```

##### HallucinationDetector (`backend/services/rag.py`)

```python
class HallucinationDetector:
    """
    Reranker-based grounding check for LLM responses.

    Scores responses on a 0-1 scale where 0 = fully grounded
    and 1 = fully hallucinated. Uses reranker cross-encoder scores
    to evaluate how well each claim is supported by retrieved sources.
    """

    def compute_hallucination_score(
        self,
        response: str,
        sources: List[Dict]
    ) -> float:
        """Compute hallucination score (0=grounded, 1=hallucinated)"""

    def compute_confidence_score(
        self,
        source_count: int,
        rerank_scores: List[float],
        retrieval_density: float
    ) -> float:
        """Multi-signal confidence from source count, rerank scores, retrieval density"""
```

##### SettingsService (`backend/services/settings.py`)

```python
class SettingsService:
    """
    Centralized settings management with 180+ settings.

    Provides unified access to all application settings with:
    - SettingDefinition dataclass with validation, categories, UI metadata
    - 19 frontend settings tab categories with vertical navigation
    - Type validation (str, int, float, bool, json)
    - Default values with database override
    - Category-based organization for admin UI

    Frontend tab categories:
    - providers-tab: LLM provider configuration
    - rag-tab: RAG pipeline settings
    - security-tab: Authentication and access control
    - generation-tab: Document generation options
    - notifications-tab: Alert and notification settings
    - audio-tab: TTS provider configuration
    - ingestion-tab: Document processing settings
    - scraper-tab: Web scraper configuration
    """

    async def get_setting(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get setting value with fallback to default"""

    async def set_setting(
        self,
        key: str,
        value: Any
    ) -> None:
        """Set setting value with type validation"""

    def get_definitions_by_category(
        self,
        category: str
    ) -> List:
        """Get all setting definitions for a category"""
```

##### Enhanced WebScraperService (`backend/services/web_crawler.py`)

```python
class WebScraperService:
    """
    Enhanced web scraping service with crawl4ai v0.8.0.

    Phase 96 enhancements:
    - ProxyManager: Proxy rotation with round_robin/random strategies
    - Sitemap crawling: XML sitemap parsing with gzip and sitemapindex recursion
    - robots.txt compliance: urllib.robotparser integration
    - Search-based crawling: DuckDuckGo-based URL discovery
    - Crash recovery: JSON state persistence for interrupted crawls
    - Jina Reader fallback: Alternative extraction via Jina API
    """

    async def crawl_sitemap(
        self,
        sitemap_url: str,
        max_pages: int = 100
    ) -> List[Dict]:
        """Parse XML sitemap with gzip support and sitemapindex recursion"""

    def parse_robots_txt(
        self,
        robots_url: str
    ) -> Dict:
        """Parse robots.txt for compliance using urllib.robotparser"""

    async def search_and_crawl(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict]:
        """DuckDuckGo-based URL discovery and crawling"""

    async def crawl_with_recovery(
        self,
        urls: List[str],
        state_file: str = None
    ) -> List[Dict]:
        """Crawl with crash recovery via JSON state persistence"""

    async def _crawl_with_jina(
        self,
        url: str
    ) -> Dict:
        """Jina Reader API fallback for failed crawls"""


class ProxyManager:
    """
    Proxy rotation for web scraping.

    Strategies:
    - round_robin: Cycle through proxies sequentially
    - random: Select proxy randomly
    """

    def get_next_proxy(self) -> str:
        """Get next proxy based on rotation strategy"""
```

##### CrawlSchedulerService (`backend/services/crawl_scheduler.py`)

```python
class CrawlSchedulerService:
    """
    Celery Beat scheduled crawl management.

    Provides CRUD operations for scheduled crawl jobs that run
    on configurable intervals via Celery Beat.
    """

    async def create_schedule(
        self,
        url: str,
        interval: str,
        config: Dict = None
    ) -> Dict:
        """Create a new scheduled crawl job"""

    async def list_schedules(self) -> List[Dict]:
        """List all scheduled crawl jobs"""

    async def delete_schedule(
        self,
        schedule_id: str
    ) -> None:
        """Delete a scheduled crawl job"""
```

##### _SafeDatetime (`backend/services/text_to_sql/service.py`)

```python
class _SafeDatetime:
    """
    Safe datetime wrapper for sandbox execution.

    Prevents sandbox escape via datetime.__class__.__bases__[0].__subclasses__()
    by only exposing safe static methods:
    - now(), utcnow(), fromisoformat(), strptime(), today()

    AST-level blocked_attrs prevents access to dunder attributes
    in sandboxed code execution.
    """
```

---

## Complete Function Reference

### Document Processing Functions

```python
# backend/processors/universal.py

class UniversalProcessor:
    """Process any document format"""

    def process(
        self,
        file_path: str,
        processing_mode: str = "full"  # "full", "ocr", "basic"
    ) -> ExtractedContent:
        """
        Extract content from document.

        Args:
            file_path: Path to document file
            processing_mode:
                - "full": Text + OCR + AI image analysis (most thorough)
                - "ocr": Text + OCR for scanned documents
                - "basic": Text extraction only (fastest)

        Returns:
            ExtractedContent with text, pages, images, metadata
        """

    def _process_pdf(self, file_path: str, mode: str) -> ExtractedContent:
        """Process PDF with PyMuPDF + parallel OCR"""

    def _process_docx(self, file_path: str, mode: str) -> ExtractedContent:
        """Process DOCX with python-docx"""

    def _process_pptx(self, file_path: str, mode: str) -> ExtractedContent:
        """Process PPTX with python-pptx"""

    def _ocr_pages_parallel(
        self,
        file_path: str,
        page_indices: List[int],
        file_size_mb: float = 0
    ) -> Dict[int, str]:
        """OCR multiple pages in parallel using ThreadPoolExecutor"""

    def _ocr_image(self, image_path: str) -> str:
        """OCR single image using PaddleOCR with Tesseract fallback"""


# backend/processors/chunker.py

class DocumentChunker:
    """Chunk documents for embedding"""

    def chunk(
        self,
        text: str,
        strategy: ChunkingStrategy = None,
        metadata: Dict = None,
        document_id: str = None
    ) -> List[Chunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Document text
            strategy: RECURSIVE, SEMANTIC, TOKEN, MARKDOWN, HTML, CODE, SLIDE
            metadata: Additional metadata to attach
            document_id: Parent document ID

        Returns:
            List of Chunk objects
        """

    def chunk_with_pages(
        self,
        pages: List[Dict],              # [{page_num, text, images}...]
        strategy: ChunkingStrategy = None,
        metadata: Dict = None,
        document_id: str = None
    ) -> List[Chunk]:
        """Chunk document preserving page information"""

    def chunk_hierarchical(
        self,
        text: str,
        metadata: Dict = None,
        document_id: str = None
    ) -> List[Chunk]:
        """
        Create multi-level chunks for large documents.

        Levels:
        - 2: Document summary (1 chunk)
        - 1: Section summaries (N chunks)
        - 0: Detail chunks (many chunks)
        """

    def detect_document_type(
        self,
        text: str,
        filename: str = None,
        mime_type: str = None
    ) -> DocumentType:
        """Auto-detect document type for adaptive chunking"""

    def chunk_adaptive(
        self,
        text: str,
        filename: str = None,
        mime_type: str = None,
        metadata: Dict = None,
        document_id: str = None
    ) -> List[Chunk]:
        """Chunk with automatic type detection and strategy selection"""
```

### RAG Enhancement Functions

```python
# backend/services/query_expander.py

class QueryExpander:
    """Expand queries for better recall"""

    async def expand(
        self,
        query: str,
        count: int = 3
    ) -> List[str]:
        """
        Generate query variations using LLM.

        Returns list including original query + variations with:
        - Synonyms
        - Related terms
        - Alternative phrasings
        """


# backend/services/hyde.py

class HyDEExpander:
    """Hypothetical Document Embeddings"""

    async def generate(self, query: str) -> str:
        """
        Generate hypothetical answer document.

        For short/abstract queries, generates a hypothetical
        document that would answer the query. This document
        is then embedded instead of the query for better
        semantic matching.
        """


# backend/services/rag_verifier.py

class RAGVerifier:
    """Verify RAG response accuracy (Self-RAG)"""

    async def verify(
        self,
        question: str,
        answer: str,
        sources: List[SearchResult]
    ) -> VerificationResult:
        """
        Verify answer is supported by sources.

        Returns:
            VerificationResult with:
            - is_supported: bool
            - confidence: float (0-1)
            - reasoning: str
            - unsupported_claims: List[str]
        """


# backend/services/corrective_rag.py

class CorrectiveRAG:
    """Auto-correct low-confidence RAG responses"""

    async def refine_query(
        self,
        original_query: str,
        original_results: List[SearchResult],
        confidence: float
    ) -> CRAGResult:
        """
        Refine query when confidence is low.

        Process:
        1. Analyze why original query failed
        2. Reformulate query based on available content
        3. Re-search with refined query
        4. Generate improved response
        """


# backend/services/query_classifier.py

class QueryClassifier:
    """Classify query intent for dynamic weighting"""

    async def classify(self, query: str) -> QueryClassification:
        """
        Classify query intent.

        Returns weights for search types:
        - FACTUAL: Higher keyword weight (0.4 vector, 0.6 keyword)
        - ANALYTICAL: Balanced (0.6 vector, 0.4 keyword)
        - EXPLORATORY: Higher vector weight (0.8 vector, 0.2 keyword)
        """
```

### Helper Functions

```python
# backend/services/generator.py

def filter_llm_metatext(text: str) -> str:
    """
    Remove LLM meta-text from generated content.

    Filters patterns like:
    - "Here's the content you requested:"
    - "I'll create a section about..."
    - "Let me explain..."
    """

def filter_title_echo(content: str, section_title: str) -> str:
    """Remove title repetition from section content"""

def smart_truncate(text: str, max_chars: int) -> str:
    """Truncate text intelligently at word boundaries"""

def get_theme_colors(theme_key: str, custom_colors: Dict = None) -> Dict:
    """
    Get color theme for document generation.

    Available themes:
    - business, creative, modern, nature, elegant
    - vibrant, tech, warm, minimalist, dark
    - colorful, academic
    """

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color (#RRGGBB) to RGB tuple"""

def sanitize_filename(title: str, max_length: int = 50) -> str:
    """Create valid filename from title"""


# backend/services/rag.py

def _get_language_instruction(language: str, auto_detect: bool = False) -> str:
    """
    Get system instruction for response language.

    Args:
        language: Target language code (en, de, es, fr, etc.)
        auto_detect: If True, respond in same language as question

    Returns:
        System instruction string to append to prompt
    """

def _parse_suggested_questions(content: str) -> Tuple[str, List[str]]:
    """
    Parse suggested follow-up questions from LLM response.

    Extracts questions marked with [SUGGESTED_QUESTIONS] tag.

    Returns:
        Tuple of (cleaned content, list of questions)
    """


# backend/services/vectorstore.py

def parse_search_query(query: str) -> Tuple[str, bool]:
    """
    Parse search query with operators into PostgreSQL tsquery.

    Supported operators:
    - AND: term1 AND term2
    - OR: term1 OR term2
    - NOT: NOT term
    - Phrases: "exact phrase"
    - Grouping: (term1 OR term2) AND term3

    Returns:
        Tuple of (tsquery string, has_operators flag)
    """


# backend/services/embeddings.py

def compute_similarity(
    embedding1: List[float],
    embedding2: List[float]
) -> float:
    """
    Compute cosine similarity between embeddings.

    Returns: Float between 0 (dissimilar) and 1 (identical)
    """

def get_optimal_batch_size(provider: str, num_texts: int) -> int:
    """
    Get optimal batch size based on provider rate limits.

    - OpenAI: 2048 texts per batch
    - Ollama: 100 texts per batch
    - HuggingFace: 64 texts per batch
    """
```

---

## Database Operations Reference

### Common Query Patterns

```python
# Get document with chunks
async with async_session_context() as db:
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()


# Search chunks with access control
async with async_session_context() as db:
    result = await db.execute(
        select(Chunk)
        .join(Document)
        .join(AccessTier)
        .where(AccessTier.level <= user_access_level)
        .where(Document.processing_status == ProcessingStatus.COMPLETED)
        .order_by(Chunk.chunk_index)
    )
    chunks = result.scalars().all()


# Vector similarity search (pgvector)
async with async_session_context() as db:
    result = await db.execute(
        text("""
            SELECT
                c.id,
                c.content,
                c.document_id,
                d.filename,
                1 - (c.embedding <=> :query_embedding) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            JOIN access_tiers at ON c.access_tier_id = at.id
            WHERE at.level <= :access_level
            ORDER BY c.embedding <=> :query_embedding
            LIMIT :top_k
        """),
        {
            "query_embedding": str(query_embedding),
            "access_level": access_level,
            "top_k": top_k
        }
    )


# Full-text search
async with async_session_context() as db:
    result = await db.execute(
        text("""
            SELECT
                c.id,
                c.content,
                ts_rank(to_tsvector('english', c.content), query) as rank
            FROM chunks c,
                 plainto_tsquery('english', :search_query) query
            WHERE to_tsvector('english', c.content) @@ query
            ORDER BY rank DESC
            LIMIT :top_k
        """),
        {"search_query": query, "top_k": top_k}
    )


# Create document with chunks (transaction)
async with async_session_context() as db:
    try:
        # Create document
        document = Document(
            filename=filename,
            file_hash=file_hash,
            access_tier_id=access_tier_id
        )
        db.add(document)
        await db.flush()  # Get document.id

        # Create chunks
        for i, chunk_data in enumerate(chunks):
            chunk = Chunk(
                document_id=document.id,
                content=chunk_data["content"],
                embedding=chunk_data["embedding"],
                chunk_index=i,
                access_tier_id=access_tier_id
            )
            db.add(chunk)

        await db.commit()
    except Exception:
        await db.rollback()
        raise


# Update document status
async with async_session_context() as db:
    await db.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(
            processing_status=ProcessingStatus.COMPLETED,
            processed_at=datetime.utcnow()
        )
    )
    await db.commit()


# Delete document and cascade to chunks
async with async_session_context() as db:
    # Chunks deleted automatically due to cascade="all, delete-orphan"
    await db.execute(
        delete(Document).where(Document.id == document_id)
    )
    await db.commit()
```

### Migration Commands

```bash
# Create new migration
cd backend
PYTHONPATH=. alembic revision --autogenerate -m "Add new_field to documents"

# Apply migrations
PYTHONPATH=. alembic upgrade head

# Rollback one migration
PYTHONPATH=. alembic downgrade -1

# View migration history
PYTHONPATH=. alembic history
```

---

## Quick Reference: Finding Code

| What you're looking for | Where to find it |
|------------------------|------------------|
| API endpoint definition | `backend/api/routes/{domain}.py` |
| Business logic | `backend/services/{service}.py` |
| Database models | `backend/db/models.py` |
| Database migrations | `backend/db/migrations/versions/` |
| Agent implementations | `backend/services/agents/` |
| Document processing | `backend/processors/` |
| Frontend pages | `frontend/app/dashboard/{page}/page.tsx` |
| API client | `frontend/lib/api/client.ts` |
| React hooks | `frontend/lib/api/hooks.ts` |
| UI components | `frontend/components/ui/` |
| RAG pipeline classes (Phases 84-95) | `backend/services/rag.py` |
| Settings service (Phase 94) | `backend/services/settings.py` |
| Web scraper enhancements (Phase 96) | `backend/services/web_crawler.py` |
| Crawl scheduler (Phase 96) | `backend/services/crawl_scheduler.py` |
| Scraper API routes (Phase 96) | `backend/api/routes/scraper.py` |
| SQL sandbox hardening (Phase 91) | `backend/services/text_to_sql/service.py` |
| Settings UI tabs (Phase 94) | `frontend/app/dashboard/admin/settings/components/` |

---

## Advanced RAG Features

### Self-RAG (Response Verification)

**File:** `backend/services/self_rag.py`

Self-RAG verifies LLM responses against source documents to detect hallucinations.

```python
from backend.services.self_rag import get_self_rag, SelfRAGResult

# Initialize Self-RAG
self_rag = get_self_rag(
    min_supported_ratio=0.7,      # Min ratio of claims that must be supported
    enable_regeneration=True,      # Auto-regenerate on issues
)

# Verify a response
result: SelfRAGResult = await self_rag.verify_response(
    response="Generated response text",
    sources=search_results,        # List[SearchResult]
    query="Original user query",
    llm=llm_instance,
)

# Check results
if result.needs_regeneration:
    print(f"Hallucinations detected: {result.hallucination_count}")
    print(f"Confidence: {result.overall_confidence}")
    print(f"Issues: {result.regeneration_feedback}")
```

**Key Classes:**
- `SelfRAG` - Main verification class
- `SelfRAGResult` - Verification result with confidence, hallucination count
- `ClaimAnalysis` - Per-claim verification result
- `SupportLevel` - Enum: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, CONTRADICTED

**Configuration (RAGConfig):**
```python
RAGConfig(
    enable_self_rag=True,
    self_rag_min_supported_ratio=0.7,
    self_rag_enable_regeneration=True,
)
```

### Smart Pre-Filtering

**File:** `backend/services/smart_filter.py`

Pre-filters large document collections (10k-100k docs) before vector search.

```python
from backend.services.smart_filter import get_smart_filter, SmartFilterConfig

filter_config = SmartFilterConfig(
    metadata_max_candidates=1000,
    summary_top_k=500,
    enable_keyword_filter=True,
)

smart_filter = get_smart_filter(config=filter_config)

# Filter documents before search
result = await smart_filter.filter_documents(
    query="search query",
    query_embedding=embedding_vector,
    total_docs=50000,
    db=db_session,
)

# Use filtered document IDs for vector search
filtered_ids = result.document_ids
```

### Three-Level Hierarchical Retrieval

**File:** `backend/services/retrieval_strategies.py`

Implements Collection → Document → Chunk retrieval for large collections.

```python
from backend.services.retrieval_strategies import (
    ThreeLevelRetriever,
    ThreeLevelConfig,
    get_three_level_retriever,
)

config = ThreeLevelConfig(
    max_collections=5,
    docs_per_collection=10,
    chunks_per_doc=3,
    final_top_k=10,
)

retriever = get_three_level_retriever(vectorstore, db_session, config)

results = await retriever.retrieve(
    query="search query",
    query_embedding=embedding,
    search_type="hybrid",
    top_k=10,
)
```

### Knowledge Graph (GraphRAG)

**File:** `backend/services/knowledge_graph.py`

Entity extraction and graph-based retrieval for complex queries.

```python
from backend.services.knowledge_graph import get_knowledge_graph_service

kg_service = await get_knowledge_graph_service(db_session)

# Extract entities from document
stats = await kg_service.process_document_for_graph(document_id)

# Graph-based search
context = await kg_service.graph_search(
    query="query about entities",
    max_hops=2,
    top_k=10,
)

# Hybrid search (vector + graph)
results = await kg_service.hybrid_search(
    query="search query",
    vector_results=vector_results,
    graph_weight=0.3,
)
```

---

## Agent System Reference

### Available Agents

| Agent | File | Purpose |
|-------|------|---------|
| `generator` | `worker_agents.py` | Content generation using LLM |
| `critic` | `worker_agents.py` | Quality evaluation and feedback |
| `research` | `worker_agents.py` | Document search and retrieval |
| `tool` | `worker_agents.py` | File operations (PPTX, DOCX, PDF) |
| `validator` | `validator_agent.py` | Cross-validation for hallucination detection |

### Validator Agent

**File:** `backend/services/agents/validator_agent.py`

Cross-validates generated content against source documents.

```python
from backend.services.agents.validator_agent import create_validator_agent

validator = create_validator_agent(trajectory_collector=collector)

# Validate content
result = await validator.execute(
    task=validation_task,
    context={
        "content": "Content to validate",
        "sources": source_documents,
        "original_query": "User's query",
    }
)

# Quick validation
is_valid, confidence, issues = await validator.quick_check(content, sources)
```

**ValidationReport Fields:**
- `is_valid: bool` - Overall validation result
- `overall_confidence: float` - 0.0 to 1.0
- `verified_claims: int` - Claims supported by sources
- `unsupported_claims: int` - Potential hallucinations
- `contradicted_claims: int` - Claims contradicting sources
- `improvements: List[str]` - Suggested fixes

### Tool Calling Framework

**File:** `backend/services/agents/tools.py`

Extensible tool system for agent capabilities.

```python
from backend.services.agents.tools import (
    ToolRegistry,
    ToolExecutor,
    BaseTool,
    register_default_tools,
)

# Register default tools
register_default_tools(rag_service=rag, scraper_service=scraper)

# Get tool registry
registry = ToolRegistry()

# List available tools
tools = registry.list_tools(category=ToolCategory.RETRIEVAL)

# Execute a tool
executor = ToolExecutor()
result = await executor.execute(
    tool_name="search_documents",
    parameters={"query": "search text", "limit": 5},
)
```

**Built-in Tools:**
- `search_documents` - Search user's uploaded documents
- `calculator` - Safe mathematical calculations
- `web_search` - Web search integration
- `get_datetime` - Current date/time

**Registry Methods:**
```python
# Check if a tool exists
if ToolRegistry.has_tool("search_documents"):
    # Tool is available

# Get OpenAI-compatible function definitions for LLM tool calling
functions = ToolRegistry.get_function_definitions()
# Returns list of tool schemas compatible with OpenAI/Anthropic function calling

# Get formatted descriptions for prompts
descriptions = ToolRegistry.get_descriptions()
```

**Integration with ToolExecutionAgent:**

The ToolExecutionAgent automatically checks the ToolRegistry for extensible tools before falling back to built-in document generation tools:

```python
from backend.services.agents.tools import get_tool_executor
from backend.services.agents.worker_agents import create_worker_agents

# Create executor (auto-registers default tools)
executor = get_tool_executor()

# Pass to worker factory
workers = create_worker_agents(
    rag_service=rag,
    generator_service=generator,
    tool_executor=executor,  # Enables extensible tool calling
)

# ToolExecutionAgent now supports both:
# 1. Registered tools (search_documents, calculator, etc.)
# 2. Built-in tools (generate_pptx, generate_docx, export_markdown)
```

**Creating Custom Tools:**
```python
class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    category = ToolCategory.COMPUTATION
    parameters = [
        ToolParameter(name="input", type="string", required=True),
    ]

    async def execute(self, input: str) -> Any:
        return {"result": f"Processed: {input}"}

ToolRegistry.register(MyCustomTool())
```

### Adaptive Planning

**File:** `backend/services/agents/trajectory_collector.py`

Learns from execution history to optimize future plans.

```python
from backend.services.agents.trajectory_collector import (
    get_adaptive_planner,
    PlanningHints,
)

# Create planner
planner = get_adaptive_planner(trajectory_collector)

# Get planning hints for a task type
hints: PlanningHints = await planner.get_planning_hints(
    task_type="document_generation",
    agent_type="manager",
)

# Use hints
print(f"Recommended steps: {hints.recommended_steps}")
print(f"Average tokens: {hints.avg_tokens}")
print(f"Success rate: {hints.success_rate}")
print(f"Optimization tips: {hints.optimization_tips}")
```

**PlanningHints Fields:**
- `recommended_steps: List[str]` - Common successful step patterns
- `common_patterns: List[str]` - High-level execution patterns
- `avg_tokens: int` - Average token usage
- `avg_duration_ms: int` - Average execution time
- `success_rate: float` - Historical success rate
- `common_failures: List[str]` - Frequent error patterns
- `optimization_tips: List[str]` - Suggested optimizations

### Parallel Task Execution

**File:** `backend/services/agents/manager_agent.py`

The manager agent executes independent steps in parallel using `asyncio.gather()`.

```python
# In execute_plan(), when multiple steps are ready:
if len(ready_steps) > 1:
    # Execute in parallel
    parallel_results = await asyncio.gather(
        *[execute_step_parallel(step) for step in ready_steps],
        return_exceptions=True,
    )
else:
    # Execute sequentially
    result = await self._execute_step(step, context)
```

---

## Configuration Reference

### RAG Configuration Options

```python
RAGConfig(
    # Core retrieval
    top_k=10,
    similarity_threshold=0.55,
    use_hybrid_search=True,
    rerank_results=True,

    # Query expansion (improves recall 8-12%)
    enable_query_expansion=True,
    query_expansion_count=3,

    # HyDE (Hypothetical Document Embeddings)
    enable_hyde=True,
    hyde_min_query_words=5,

    # CRAG (Corrective RAG)
    enable_crag=True,
    crag_confidence_threshold=0.5,

    # Self-RAG (Response verification)
    enable_self_rag=False,  # Set True to enable
    self_rag_min_supported_ratio=0.7,

    # Smart pre-filtering (for large collections)
    enable_smart_filter=True,
    smart_filter_min_docs=500,
    smart_filter_max_candidates=1000,

    # Hierarchical retrieval
    enable_hierarchical_retrieval=False,
    hierarchical_doc_limit=10,
    hierarchical_chunks_per_doc=3,

    # Knowledge graph
    enable_knowledge_graph=False,
    knowledge_graph_max_hops=2,
)
```

### Environment Variables

```bash
# Self-RAG
ENABLE_SELF_RAG=false

# Smart filtering
ENABLE_SMART_FILTER=true

# Query expansion
ENABLE_QUERY_EXPANSION=true
QUERY_EXPANSION_COUNT=3

# CRAG
ENABLE_CRAG=true

# HyDE
ENABLE_HYDE=true

# Knowledge Graph
ENABLE_KNOWLEDGE_GRAPH=false
```

---

## Phase 84-96 Service Reference

This section provides a consolidated reference for all services and enhancements introduced in Phases 84 through 96.

### Phase 84: Binary Quantization

**File:** `backend/services/rag.py`

`BinaryQuantizationManager` provides INT8/binary vector quantization for 32x storage savings. Converts float32 embeddings (6 KB per vector) to binary (192 bytes per vector), enabling massive index compression with minimal accuracy loss through a rerank step.

**Key Methods:**
- `quantize_vectors(vectors, mode)` - Quantize to INT8 or binary
- `search_quantized(query_vector, top_k)` - Fast approximate search on quantized index
- `rerank_candidates(query_vector, candidates, top_k)` - Rerank with full-precision vectors

### Phase 85: Late Interaction Models (ColBERT)

**File:** `backend/services/rag.py`

`ColBERTRetriever` implements late interaction retrieval with per-token matching. Instead of compressing documents into single vectors, ColBERT retains per-token embeddings and computes MaxSim scores between query and document tokens for higher accuracy.

**Key Features:**
- Per-token embedding storage and MaxSim scoring
- WARP engine for 3x faster multi-vector retrieval
- Integration with existing hybrid search pipeline

### Phase 86: Content Freshness Scoring

**File:** `backend/services/rag.py`

`ContentFreshnessScorer` applies time-decay scoring to document relevance, boosting recently updated content and penalizing stale documents.

**Settings:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.content_freshness_enabled` | Enable freshness scoring | `true` |
| `rag.freshness_decay_days` | Days before freshness boost decays | `30` |
| `rag.freshness_boost_factor` | Multiplier for fresh documents | `1.05` |
| `rag.freshness_penalty_factor` | Multiplier for stale documents (>180 days) | `0.95` |

### Phase 87: Dependency Updates & Stability

**File:** `backend/pyproject.toml`

Comprehensive dependency audit and update:
- **google-genai migration**: Replaced deprecated `google-generativeai` with unified `google-genai` SDK
- **Upper bounds on critical deps**: FastAPI, Pydantic, SQLAlchemy, Ray pinned with upper bounds
- **Ray 2.10+**: Updated for better async support and memory management
- **LangChain 0.3.x audit**: All imports verified correct

### Phase 88: Circuit Breaker Pattern

**File:** `backend/services/rag.py`

`CircuitBreaker` implements fault tolerance for external service calls (LLM APIs, embedding services, etc.).

**States:**
| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation; requests pass through. Transitions to OPEN after N consecutive failures. |
| **OPEN** | Fast-fail mode; requests rejected immediately. Transitions to HALF_OPEN after recovery timeout. |
| **HALF_OPEN** | Recovery testing; limited requests allowed. Returns to CLOSED on success, OPEN on failure. |

**Configuration:**
- `failure_threshold`: Number of failures before opening circuit (default: 5)
- `recovery_timeout`: Seconds to wait before testing recovery (default: 30)
- `half_open_max_calls`: Max calls in half-open state (default: 1)

### Phase 89: Agentic RAG

**File:** `backend/services/rag.py`

`AgenticRAGEngine` extends standard RAG with multi-step reasoning and tool use for complex queries that require iterative retrieval or computation.

**Pipeline:**
1. **Agent Routing** - Classifies whether query needs agentic processing vs. standard RAG
2. **Query Decomposition** - Breaks complex queries into sub-queries
3. **Iterative Retrieval** - Multiple search rounds with refinement based on intermediate results
4. **Tool Use** - Integrates search, calculation, and other tools
5. **Response Synthesis** - Combines results from all sub-queries

### Phase 90: Inline Citations

**File:** `backend/services/rag.py`

`CitationEngine` adds numbered source references (`[1]`, `[2]`, `[3]`) inline in LLM responses with verification that citations map to actual retrieved sources.

**Features:**
- Numbered superscript citation badges inline in responses
- Citation verification ensuring all markers map to real sources
- Source mapping with document name, page number, similarity score, and snippet
- Frontend hover popovers and click-to-navigate to source

### Phase 91: Sandbox Hardening

**File:** `backend/services/text_to_sql/service.py`

Enhanced SQL sandbox security with `_SafeDatetime` wrapper and AST-level attribute blocking.

**Security Measures:**
- `_SafeDatetime`: Wraps `datetime` module to prevent `.__class__.__bases__[0].__subclasses__()` sandbox escape
- Only safe static methods exposed: `now()`, `utcnow()`, `fromisoformat()`, `strptime()`, `today()`
- AST-level `blocked_attrs` list prevents access to dunder attributes in sandboxed code
- Blocks `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `dir`, `breakpoint`

### Phase 92: Embedding Quantization

**File:** `backend/services/rag.py`

`EmbeddingQuantizer` provides INT8/binary quantization for embeddings with Matryoshka embeddings support.

**Features:**
- INT8 quantization: 4x storage reduction with minimal accuracy loss
- Binary quantization: 32x storage reduction for large-scale indices
- Matryoshka embeddings: Variable-dimension embeddings that allow trading off accuracy for speed by truncating embedding dimensions

### Phase 93: DSPy Prompt Optimization

**File:** `backend/services/rag.py`

`DSPyOptimizer` automates prompt compilation using the DSPy framework, replacing hand-crafted prompts with data-driven optimized versions.

**Optimizers:**
| Optimizer | Description |
|-----------|-------------|
| `BootstrapFewShot` | Selects optimal few-shot examples from training data |
| `MIPRO` | Multi-instruction prompt optimization via search |
| `BayesianSignatureOptimizer` | Bayesian tuning of prompt signatures and instructions |

**Settings:**
| Setting | Description | Default |
|---------|-------------|---------|
| `rag.dspy_optimization_enabled` | Enable DSPy optimization | `false` |
| `rag.dspy_default_optimizer` | Default optimizer to use | `BootstrapFewShot` |
| `rag.dspy_min_examples` | Minimum examples before optimization runs | `10` |
| `rag.dspy_inference_enabled` | Use compiled DSPy modules in inference | `false` |

### Phase 94: Settings Service Overhaul

**File:** `backend/services/settings.py`

`SettingsService` provides centralized management for 180+ application settings with type validation, categories, and UI metadata.

**Architecture:**
- `SettingDefinition` dataclass with key, type, default, description, category, UI metadata
- Database-backed overrides with in-memory caching (5-minute TTL)
- 19 frontend settings tab categories organized into groups

**Frontend Components:**
| Component | File | Purpose |
|-----------|------|---------|
| `providers-tab` | `frontend/app/dashboard/admin/settings/components/providers-tab.tsx` | LLM provider configuration |
| `rag-tab` | `frontend/app/dashboard/admin/settings/components/rag-tab.tsx` | RAG pipeline settings |
| `security-tab` | `frontend/app/dashboard/admin/settings/components/security-tab.tsx` | Authentication and access control |
| `generation-tab` | `frontend/app/dashboard/admin/settings/components/generation-tab.tsx` | Document generation options |
| `notifications-tab` | `frontend/app/dashboard/admin/settings/components/notifications-tab.tsx` | Alert and notification settings |
| `audio-tab` | `frontend/app/dashboard/admin/settings/components/audio-tab.tsx` | TTS provider configuration |
| `ingestion-tab` | `frontend/app/dashboard/admin/settings/components/ingestion-tab.tsx` | Document processing settings |
| `scraper-tab` | `frontend/app/dashboard/admin/settings/components/scraper-tab.tsx` | Web scraper configuration |

**Settings Page:** `frontend/app/dashboard/admin/settings/page.tsx` - Vertical sidebar navigation with category grouping (System, AI & Models, Processing, Intelligence, Integrations, Security).

### Phase 95: Multi-feature Release

**File:** `backend/services/rag.py` and frontend components

#### Phase 95J: Hallucination Scoring
`HallucinationDetector` in `rag.py` provides reranker-based grounding checks:
- `_compute_hallucination_score()`: Scores 0 (grounded) to 1 (hallucinated)
- `_compute_confidence_score()`: Multi-signal confidence from source count, rerank scores, retrieval density
- Scores injected into RAG response metadata

#### Phase 95K: Content Freshness
Time-based document scoring with configurable freshness boost (default 1.05x within 30 days) and staleness penalty (default 0.95x after 180 days). Controlled by `rag.content_freshness_enabled`.

#### Phase 95L: Conversation-Aware Retrieval
Enriches retrieval query with last 3 user messages from conversation history, improving embedding quality for follow-up questions. Original question preserved for LLM prompt generation.

#### Phase 95O: Agent Insights
Per-agent analytics component showing usage metrics:
- 4 metric cards: total queries, avg response time, satisfaction %, active users
- Time period selector (7d, 30d, all time)
- Recent activity list with query summaries and feedback indicators

#### Phase 95R: BYOK (Bring Your Own Key)
Client-side API key management for per-provider keys:
- Supported providers: OpenAI, Anthropic, Google AI, Mistral, Groq, Together
- Key validation testing, status badges (Active/Invalid/Untested)
- Encrypted localStorage storage with master BYOK toggle

#### Phase 95S: Prompt Library
Full CRUD management for reusable prompt templates:
- Card grid with search, category filter, use count badges
- Template variables with `{{variable}}` syntax and runtime substitution
- Public/private sharing, system templates, duplicate functionality

#### Phase 95T: Canvas Panel
Side-by-side artifact viewing and editing:
- Sheet panel for AI-generated code and content
- View mode with line numbers, edit mode with monospace textarea
- Copy to clipboard, download as file, AI edit request input

### Phase 96: Web Scraper Upgrade

**Files:** `backend/services/web_crawler.py`, `backend/services/crawl_scheduler.py`, `backend/api/routes/scraper.py`, `backend/services/settings.py`

Major upgrade to the web scraping infrastructure with crawl4ai v0.8.0.

#### Enhanced WebScraperService (`services/web_crawler.py`)

| Method | Description |
|--------|-------------|
| `crawl_sitemap()` | XML sitemap parsing with gzip support and sitemapindex recursion |
| `parse_robots_txt()` | robots.txt compliance using `urllib.robotparser` |
| `search_and_crawl()` | DuckDuckGo-based URL discovery and crawling |
| `crawl_with_recovery()` | Crash recovery with JSON state persistence |
| `_crawl_with_jina()` | Jina Reader API fallback for failed crawls |

**ProxyManager:** Proxy rotation with `round_robin` and `random` strategies for anti-detection.

#### CrawlSchedulerService (`services/crawl_scheduler.py`)
Celery Beat integration for scheduled crawl jobs with CRUD management.

#### New API Endpoints (`api/routes/scraper.py`)

```
POST   /api/v1/scraper/sitemap-crawl      # Crawl URLs from XML sitemap
GET    /api/v1/scraper/stream              # SSE streaming for crawl progress
POST   /api/v1/scraper/search-crawl        # Search-based URL discovery and crawl
GET    /api/v1/scraper/robots-txt          # Parse robots.txt for a domain
POST   /api/v1/scraper/schedules           # Create scheduled crawl
GET    /api/v1/scraper/schedules           # List scheduled crawls
PUT    /api/v1/scraper/schedules/{id}      # Update scheduled crawl
DELETE /api/v1/scraper/schedules/{id}      # Delete scheduled crawl
```

#### New Scraper Settings (`services/settings.py`)

| Setting | Description | Default |
|---------|-------------|---------|
| `scraper.proxy_enabled` | Enable proxy rotation | `false` |
| `scraper.proxy_list` | JSON array of proxy URLs | `[]` |
| `scraper.proxy_strategy` | Rotation strategy (`round_robin` / `random`) | `round_robin` |
| `scraper.jina_fallback_enabled` | Enable Jina Reader API fallback | `false` |
| `scraper.adaptive_crawling_enabled` | Enable adaptive crawl speed | `true` |
| `scraper.crash_recovery_enabled` | Enable JSON state persistence | `true` |

---

## See Also

- [DEVELOPER_ONBOARDING.md](./DEVELOPER_ONBOARDING.md) - Architecture overview
- [TECHNICAL_ARCHITECTURE.md](./TECHNICAL_ARCHITECTURE.md) - System design
- [API.md](./API.md) - REST API reference
- [AGENTS.md](./AGENTS.md) - Multi-agent system
