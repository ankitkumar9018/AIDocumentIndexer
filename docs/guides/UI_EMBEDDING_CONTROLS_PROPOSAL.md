# UI Embedding Controls - Proposal

## Current State

### ✅ What EXISTS
**Settings Page** (`/dashboard/admin/settings`):
- Configure default embedding model per LLM provider
- View available embedding models (especially for Ollama)
- System-wide configuration (admin only)

### ❌ What's MISSING
**Upload Page:**
- No embedding provider selection during upload
- No option to generate multiple embeddings simultaneously
- Always uses system default from `.env`

**Chat/Search:**
- No way to switch embedding provider per query
- No embedding status indicator

**User Experience:**
- Users don't know which embedding provider is being used
- No visibility into embedding coverage
- No control over embedding quality vs cost tradeoffs

---

## Proposed UI Enhancements

### Enhancement 1: Upload Page - Embedding Provider Selection

**Location:** `/dashboard/upload/page.tsx`

**UI Design:**

```typescript
// Add to upload form (collapsible advanced options)
<Collapsible>
  <CollapsibleTrigger className="flex items-center gap-2">
    <Settings className="h-4 w-4" />
    Advanced Options
    <ChevronDown className="h-4 w-4" />
  </CollapsibleTrigger>

  <CollapsibleContent className="space-y-4 mt-4">
    {/* Embedding Provider Selection */}
    <div className="space-y-3">
      <Label className="text-base font-semibold">
        Embedding Providers
      </Label>
      <p className="text-sm text-muted-foreground">
        Generate embeddings from multiple providers for instant switching.
        Each provider has different trade-offs for quality, cost, and privacy.
      </p>

      <div className="space-y-2">
        {/* Option 1: Ollama (Free, Local) */}
        <div className="flex items-start space-x-2 p-3 border rounded-lg hover:bg-accent">
          <Checkbox
            id="embed-ollama"
            checked={embeddingProviders.includes("ollama")}
            onCheckedChange={(checked) => {
              if (checked) {
                setEmbeddingProviders([...embeddingProviders, "ollama"]);
              } else {
                setEmbeddingProviders(embeddingProviders.filter(p => p !== "ollama"));
              }
            }}
          />
          <div className="grid gap-1.5 leading-none flex-1">
            <label
              htmlFor="embed-ollama"
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 flex items-center gap-2"
            >
              <Shield className="h-4 w-4 text-green-600" />
              Ollama (nomic-embed-text)
              <Badge variant="outline" className="ml-auto">768D</Badge>
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                FREE
              </Badge>
            </label>
            <p className="text-xs text-muted-foreground">
              ✅ Free, private, runs locally
              <br />
              ✅ Good quality (comparable to OpenAI)
              <br />
              ✅ No API cost
            </p>
          </div>
        </div>

        {/* Option 2: OpenAI (Quality, Cloud) */}
        <div className="flex items-start space-x-2 p-3 border rounded-lg hover:bg-accent">
          <Checkbox
            id="embed-openai"
            checked={embeddingProviders.includes("openai")}
            onCheckedChange={(checked) => {
              if (checked) {
                setEmbeddingProviders([...embeddingProviders, "openai"]);
              } else {
                setEmbeddingProviders(embeddingProviders.filter(p => p !== "openai"));
              }
            }}
          />
          <div className="grid gap-1.5 leading-none flex-1">
            <label
              htmlFor="embed-openai"
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 flex items-center gap-2"
            >
              <Sparkles className="h-4 w-4 text-blue-600" />
              OpenAI (text-embedding-3-small)
              <Badge variant="outline" className="ml-auto">768D</Badge>
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                $0.02/1M tokens
              </Badge>
            </label>
            <p className="text-xs text-muted-foreground">
              ✅ High quality
              <br />
              ✅ Same 768D dimension (switch without re-indexing)
              <br />
              💰 ~$0.01 per 500 documents
            </p>
          </div>
        </div>

        {/* Option 3: OpenAI Large (Maximum Quality) */}
        <div className="flex items-start space-x-2 p-3 border rounded-lg hover:bg-accent">
          <Checkbox
            id="embed-openai-large"
            checked={embeddingProviders.includes("openai-large")}
            onCheckedChange={(checked) => {
              if (checked) {
                setEmbeddingProviders([...embeddingProviders, "openai-large"]);
              } else {
                setEmbeddingProviders(embeddingProviders.filter(p => p !== "openai-large"));
              }
            }}
          />
          <div className="grid gap-1.5 leading-none flex-1">
            <label
              htmlFor="embed-openai-large"
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 flex items-center gap-2"
            >
              <Zap className="h-4 w-4 text-yellow-600" />
              OpenAI (text-embedding-3-large)
              <Badge variant="outline" className="ml-auto">3072D</Badge>
              <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">
                $0.13/1M tokens
              </Badge>
            </label>
            <p className="text-xs text-muted-foreground">
              ✅ Best quality available
              <br />
              ⚠️  Higher storage (3072D vs 768D)
              <br />
              💰 ~$0.07 per 500 documents
            </p>
          </div>
        </div>
      </div>

      {/* Cost Estimate */}
      {embeddingProviders.length > 0 && (
        <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
          <p className="text-sm font-medium mb-1">
            Estimated Cost for {selectedFiles.length} file(s):
          </p>
          <div className="text-xs space-y-1">
            {embeddingProviders.includes("ollama") && (
              <div className="flex justify-between">
                <span>Ollama (local):</span>
                <span className="font-medium text-green-600">$0.00</span>
              </div>
            )}
            {embeddingProviders.includes("openai") && (
              <div className="flex justify-between">
                <span>OpenAI Small:</span>
                <span className="font-medium">
                  ${(selectedFiles.length * 0.00002).toFixed(3)}
                </span>
              </div>
            )}
            {embeddingProviders.includes("openai-large") && (
              <div className="flex justify-between">
                <span>OpenAI Large:</span>
                <span className="font-medium">
                  ${(selectedFiles.length * 0.00014).toFixed(3)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Info Alert */}
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Why multiple embeddings?</AlertTitle>
        <AlertDescription>
          Generating embeddings from multiple providers enables instant
          switching between them without re-processing documents. Use
          free Ollama for testing, then switch to OpenAI for production
          with zero downtime.
        </AlertDescription>
      </Alert>
    </div>
  </CollapsibleContent>
</Collapsible>
```

**Backend API Update:**

```typescript
// Update upload API request
interface UploadRequest {
  files: File[];
  collection_id: string;
  folder_id?: string;
  access_tier_id: string;

  // NEW: Embedding provider selection
  embedding_providers?: string[];  // ["ollama", "openai", "openai-large"]
}

// Default to system default if not specified
embeddingProviders = embeddingProviders || [process.env.DEFAULT_LLM_PROVIDER]
```

---

### Enhancement 2: Chat Page - Embedding Provider Selector

**Location:** `/dashboard/chat/page.tsx`

**UI Design:**

```typescript
// Add to chat settings dropdown
<DropdownMenu>
  <DropdownMenuTrigger asChild>
    <Button variant="ghost" size="icon">
      <Settings className="h-4 w-4" />
    </Button>
  </DropdownMenuTrigger>
  <DropdownMenuContent align="end" className="w-80">
    {/* ... existing settings ... */}

    <DropdownMenuSeparator />
    <DropdownMenuLabel>Embedding Provider</DropdownMenuLabel>

    <div className="p-2 space-y-2">
      <RadioGroup value={selectedEmbeddingProvider} onValueChange={setSelectedEmbeddingProvider}>
        {/* Auto (use system default) */}
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="auto" id="embed-auto" />
          <Label htmlFor="embed-auto" className="flex items-center gap-2 flex-1">
            <Zap className="h-4 w-4" />
            Auto (System Default)
            <Badge variant="outline" className="ml-auto">
              {systemDefaultProvider}
            </Badge>
          </Label>
        </div>

        {/* Ollama */}
        {availableProviders.includes("ollama") && (
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="ollama" id="embed-ollama-chat" />
            <Label htmlFor="embed-ollama-chat" className="flex items-center gap-2 flex-1">
              <Shield className="h-4 w-4 text-green-600" />
              Ollama (Privacy)
              <Badge variant="outline" className="ml-auto">768D</Badge>
            </Label>
          </div>
        )}

        {/* OpenAI */}
        {availableProviders.includes("openai") && (
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="openai" id="embed-openai-chat" />
            <Label htmlFor="embed-openai-chat" className="flex items-center gap-2 flex-1">
              <Sparkles className="h-4 w-4 text-blue-600" />
              OpenAI (Quality)
              <Badge variant="outline" className="ml-auto">768D</Badge>
            </Label>
          </div>
        )}
      </RadioGroup>

      <p className="text-xs text-muted-foreground mt-2">
        Choose which embedding provider to use for semantic search.
        Only providers with available embeddings are shown.
      </p>
    </div>
  </DropdownMenuContent>
</DropdownMenu>
```

**Backend API Update:**

```typescript
// Update chat API request
interface ChatRequest {
  message: string;
  collection_ids?: string[];

  // NEW: Per-query embedding provider override
  embedding_provider?: string;  // "auto" | "ollama" | "openai" | "openai-large"
}
```

---

### Enhancement 3: Document Status - Embedding Coverage Indicator

**Location:** `/dashboard/documents/page.tsx` (document list)

**UI Design:**

```typescript
// Add embedding status column to document table
<TableCell>
  <div className="flex flex-col gap-1">
    {/* Status badges */}
    <div className="flex gap-1">
      {doc.embedding_status?.ollama && (
        <Badge variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200">
          <Shield className="h-3 w-3 mr-1" />
          Ollama
        </Badge>
      )}
      {doc.embedding_status?.openai && (
        <Badge variant="outline" className="text-xs bg-blue-50 text-blue-700 border-blue-200">
          <Sparkles className="h-3 w-3 mr-1" />
          OpenAI
        </Badge>
      )}
    </div>

    {/* Coverage indicator */}
    {(!doc.embedding_status?.ollama && !doc.embedding_status?.openai) && (
      <Badge variant="destructive" className="text-xs">
        <AlertCircle className="h-3 w-3 mr-1" />
        No embeddings
      </Badge>
    )}
  </div>
</TableCell>

// Add action to generate missing embeddings
<DropdownMenuItem onClick={() => generateEmbeddings(doc.id)}>
  <Sparkles className="h-4 w-4 mr-2" />
  Generate Missing Embeddings
</DropdownMenuItem>
```

---

### Enhancement 4: Settings Page - Embedding Dashboard

**Location:** `/dashboard/admin/settings/page.tsx`

**New Section: Embedding Status**

```typescript
<Card>
  <CardHeader>
    <CardTitle>Embedding System Status</CardTitle>
    <CardDescription>
      Overview of embedding coverage and providers
    </CardDescription>
  </CardHeader>
  <CardContent className="space-y-4">
    {/* Overall Coverage */}
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{embeddingStats.total_documents}</div>
          <p className="text-xs text-muted-foreground">
            {embeddingStats.total_chunks} chunks
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Embedding Coverage</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {embeddingStats.coverage_percentage}%
          </div>
          <Progress value={embeddingStats.coverage_percentage} className="mt-2" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {formatFileSize(embeddingStats.storage_bytes)}
          </div>
          <p className="text-xs text-muted-foreground">
            {embeddingStats.provider_count} provider(s)
          </p>
        </CardContent>
      </Card>
    </div>

    {/* Provider Breakdown */}
    <div className="space-y-3">
      <h3 className="text-sm font-medium">Provider Breakdown</h3>

      {embeddingStats.providers.map(provider => (
        <div key={provider.name} className="flex items-center justify-between p-3 border rounded-lg">
          <div className="flex items-center gap-3">
            {provider.name === "ollama" && <Shield className="h-5 w-5 text-green-600" />}
            {provider.name === "openai" && <Sparkles className="h-5 w-5 text-blue-600" />}

            <div>
              <div className="font-medium">{provider.name}</div>
              <div className="text-sm text-muted-foreground">
                {provider.model} ({provider.dimension}D)
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="font-medium">{provider.chunk_count} chunks</div>
            <div className="text-sm text-muted-foreground">
              {formatFileSize(provider.storage_bytes)}
            </div>
            {provider.is_primary && (
              <Badge variant="outline" className="mt-1">
                PRIMARY
              </Badge>
            )}
          </div>
        </div>
      ))}
    </div>

    {/* Actions */}
    <div className="flex gap-2">
      <Button variant="outline" onClick={() => generateMissingEmbeddings()}>
        <Sparkles className="h-4 w-4 mr-2" />
        Generate Missing Embeddings
      </Button>

      <Button variant="outline" onClick={() => addAdditionalProvider()}>
        <Plus className="h-4 w-4 mr-2" />
        Add Additional Provider
      </Button>
    </div>
  </CardContent>
</Card>
```

---

## Implementation Priority

### Phase 1: Basic Visibility (Easiest)
1. **Settings Page - Embedding Status Dashboard**
   - Show overall coverage statistics
   - Show provider breakdown
   - Display current primary provider

**Effort:** 2-3 hours
**Impact:** High (users understand system state)

### Phase 2: Upload Controls (Medium)
2. **Upload Page - Provider Selection**
   - Add collapsible advanced options
   - Checkbox for Ollama, OpenAI Small, OpenAI Large
   - Cost estimation

**Effort:** 4-6 hours
**Impact:** High (users control embedding generation)

### Phase 3: Query Controls (Advanced)
3. **Chat Page - Provider Selector**
   - Radio buttons for embedding provider
   - Only show providers with available embeddings
   - Per-query override

**Effort:** 3-4 hours
**Impact:** Medium (power users benefit)

4. **Document List - Embedding Status**
   - Show embedding coverage per document
   - Action to generate missing embeddings

**Effort:** 2-3 hours
**Impact:** Medium (troubleshooting visibility)

---

## Backend API Changes Needed

### 1. Embedding Stats Endpoint

```python
# backend/api/routes/embeddings.py (new file)

@router.get("/embeddings/stats")
async def get_embedding_stats(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get overall embedding system statistics.
    """
    # Count total documents and chunks
    total_documents = await db.scalar(select(func.count(Document.id)))
    total_chunks = await db.scalar(select(func.count(Chunk.id)))

    # Count chunks with embeddings
    chunks_with_embeddings = await db.scalar(
        select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
    )

    coverage_percentage = (chunks_with_embeddings / total_chunks * 100) if total_chunks > 0 else 0

    # Get provider breakdown (if multi-embedding table exists)
    providers = []
    try:
        from backend.db.models_multi_embedding import ChunkEmbedding

        result = await db.execute(
            select(
                ChunkEmbedding.provider,
                ChunkEmbedding.model,
                ChunkEmbedding.dimension,
                func.count(ChunkEmbedding.id).label('count'),
                func.sum(ChunkEmbedding.is_primary.cast(Integer)).label('primary_count')
            )
            .group_by(ChunkEmbedding.provider, ChunkEmbedding.model, ChunkEmbedding.dimension)
        )

        for row in result:
            providers.append({
                "name": row.provider,
                "model": row.model,
                "dimension": row.dimension,
                "chunk_count": row.count,
                "storage_bytes": row.count * row.dimension * 4,  # 4 bytes per float
                "is_primary": row.primary_count > 0
            })
    except ImportError:
        # Multi-embedding not enabled, use primary embedding only
        if chunks_with_embeddings > 0:
            # Detect from env
            import os
            provider = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
            dimension = get_embedding_dimension()

            providers.append({
                "name": provider,
                "model": "default",
                "dimension": dimension,
                "chunk_count": chunks_with_embeddings,
                "storage_bytes": chunks_with_embeddings * dimension * 4,
                "is_primary": True
            })

    total_storage = sum(p["storage_bytes"] for p in providers)

    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "chunks_with_embeddings": chunks_with_embeddings,
        "coverage_percentage": round(coverage_percentage, 1),
        "storage_bytes": total_storage,
        "provider_count": len(providers),
        "providers": providers
    }
```

### 2. Update Upload Endpoint

```python
# backend/api/routes/documents.py

class UploadRequest(BaseModel):
    # ... existing fields ...
    embedding_providers: Optional[List[str]] = None  # ["ollama", "openai"]

@router.post("/upload")
async def upload_document(
    # ... existing params ...
    embedding_providers: Optional[List[str]] = Form(None),
):
    # Parse embedding providers from form or use default
    providers = embedding_providers or [os.getenv("DEFAULT_LLM_PROVIDER", "ollama")]

    # Process document and generate embeddings for each provider
    for provider in providers:
        await generate_embeddings_for_provider(document, provider)
```

### 3. Update Chat Endpoint

```python
# backend/api/routes/chat.py

class ChatRequest(BaseModel):
    # ... existing fields ...
    embedding_provider: Optional[str] = "auto"  # "auto", "ollama", "openai"

@router.post("/chat")
async def chat(
    request: ChatRequest,
    # ... existing params ...
):
    # Use specified provider or fall back to default
    if request.embedding_provider == "auto":
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
    else:
        provider = request.embedding_provider

    # Pass to RAG service
    results = await rag_service.search(
        query=request.message,
        embedding_provider=provider
    )
```

---

## Summary

**Current State:** ❌ No UI controls for embeddings

**Proposed State:** ✅ Full user control

**Benefits:**
- Users can choose embedding provider during upload
- Users can switch providers per query (chat)
- Visibility into embedding coverage
- Cost estimation before generation
- Multi-provider support for flexibility

**Implementation Effort:**
- Phase 1 (Visibility): 2-3 hours
- Phase 2 (Upload Controls): 4-6 hours
- Phase 3 (Query Controls): 5-7 hours
- **Total: 11-16 hours** for full implementation

**Recommended Approach:** Start with Phase 1 (visibility) to show users what's happening, then add Phase 2 (upload controls) for most value.
