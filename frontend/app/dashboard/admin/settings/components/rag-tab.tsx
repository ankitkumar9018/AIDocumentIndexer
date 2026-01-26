"use client";

import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { EmbeddingDashboard } from "@/components/embedding-dashboard";
import { Sparkles, FileText, MessageSquare, Search, Zap, Cog, Activity, Workflow, Bot, RefreshCw, DollarSign, Shield, Layers, TreePine } from "lucide-react";

interface RagTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function RagTab({ localSettings, handleSettingChange }: RagTabProps) {
  return (
    <TabsContent value="rag" className="space-y-6">
      {/* Embedding System Status Dashboard */}
      <EmbeddingDashboard />

      {/* AI Optimization Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            AI Optimization
          </CardTitle>
          <CardDescription>
            Configure AI features to reduce costs and improve quality
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Text Preprocessing */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Text Preprocessing
            </h4>
            <p className="text-sm text-muted-foreground">
              Clean and normalize text before embedding to reduce token costs (10-20% savings)
            </p>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Enable Preprocessing</p>
                  <p className="text-sm text-muted-foreground">
                    Clean text before embedding
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.enable_preprocessing"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("ai.enable_preprocessing", e.target.checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Remove Boilerplate</p>
                  <p className="text-sm text-muted-foreground">
                    Strip headers, footers, page numbers
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.remove_boilerplate"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("ai.remove_boilerplate", e.target.checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Normalize Whitespace</p>
                  <p className="text-sm text-muted-foreground">
                    Collapse multiple spaces/newlines
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.normalize_whitespace"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("ai.normalize_whitespace", e.target.checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Deduplicate Content</p>
                  <p className="text-sm text-muted-foreground">
                    Remove near-duplicate chunks
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.deduplicate_content"] as boolean ?? false}
                  onChange={(e) => handleSettingChange("ai.deduplicate_content", e.target.checked)}
                />
              </div>
            </div>
          </div>

          {/* Document Summarization */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Document Summarization
            </h4>
            <p className="text-sm text-muted-foreground">
              Generate summaries for large documents to reduce embedding tokens (30-40% savings)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Summarization</p>
                <p className="text-sm text-muted-foreground">
                  Summarize large docs before chunking
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_summarization"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_summarization", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Page Threshold</label>
                <Input
                  type="number"
                  placeholder="50"
                  value={localSettings["ai.summarization_threshold_pages"] as number ?? 50}
                  onChange={(e) => handleSettingChange("ai.summarization_threshold_pages", parseInt(e.target.value) || 10)}
                />
                <p className="text-xs text-muted-foreground">Summarize docs with more pages</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">KB Threshold</label>
                <Input
                  type="number"
                  placeholder="100"
                  value={localSettings["ai.summarization_threshold_kb"] as number ?? 100}
                  onChange={(e) => handleSettingChange("ai.summarization_threshold_kb", parseInt(e.target.value) || 500)}
                />
                <p className="text-xs text-muted-foreground">Summarize docs larger than this</p>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Summary Model</label>
              <select
                className="w-full h-10 px-3 rounded-md border bg-background"
                value={localSettings["ai.summarization_model"] as string || "gpt-4o-mini"}
                onChange={(e) => handleSettingChange("ai.summarization_model", e.target.value)}
              >
                <option value="gpt-4o-mini">GPT-4o Mini (Cost-effective)</option>
                <option value="gpt-4o">GPT-4o (Higher quality)</option>
                <option value="claude-3-haiku">Claude 3 Haiku (Fast)</option>
              </select>
            </div>
          </div>

          {/* Semantic Caching */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Search className="h-4 w-4" />
              Semantic Caching
            </h4>
            <p className="text-sm text-muted-foreground">
              Cache responses for semantically similar queries (up to 68% API cost reduction)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Semantic Cache</p>
                <p className="text-sm text-muted-foreground">
                  Match similar queries by embedding
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_semantic_cache"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_semantic_cache", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Similarity Threshold</label>
                <Input
                  type="number"
                  step="0.01"
                  min="0.8"
                  max="1.0"
                  placeholder="0.95"
                  value={localSettings["ai.semantic_similarity_threshold"] as number ?? 0.95}
                  onChange={(e) => handleSettingChange("ai.semantic_similarity_threshold", parseFloat(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Higher = stricter matching (0.8-1.0)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Cache Entries</label>
                <Input
                  type="number"
                  placeholder="10000"
                  value={localSettings["ai.max_semantic_cache_entries"] as number ?? 10000}
                  onChange={(e) => handleSettingChange("ai.max_semantic_cache_entries", parseInt(e.target.value) || 1000)}
                />
                <p className="text-xs text-muted-foreground">Limit semantic cache size</p>
              </div>
            </div>
          </div>

          {/* Query Expansion */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Query Expansion
            </h4>
            <p className="text-sm text-muted-foreground">
              Generate query variations to improve search recall (8-12% accuracy improvement)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Query Expansion</p>
                <p className="text-sm text-muted-foreground">
                  Generate paraphrased queries for better retrieval
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_query_expansion"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_query_expansion", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Query Variations</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  placeholder="2"
                  value={localSettings["ai.query_expansion_count"] as number ?? 2}
                  onChange={(e) => handleSettingChange("ai.query_expansion_count", parseInt(e.target.value) || 2)}
                />
                <p className="text-xs text-muted-foreground">Number of variations to generate (1-5)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Expansion Model</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["ai.query_expansion_model"] as string || "gpt-4o-mini"}
                  onChange={(e) => handleSettingChange("ai.query_expansion_model", e.target.value)}
                >
                  <option value="gpt-4o-mini">GPT-4o Mini (Cost-effective)</option>
                  <option value="gpt-4o">GPT-4o (Higher quality)</option>
                  <option value="claude-3-haiku">Claude 3 Haiku (Fast)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Adaptive Chunking */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Cog className="h-4 w-4" />
              Adaptive Chunking
            </h4>
            <p className="text-sm text-muted-foreground">
              Automatically optimize chunk size based on document type (10-15% token savings)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Adaptive Chunking</p>
                <p className="text-sm text-muted-foreground">
                  Auto-detect document type and adjust chunk size
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_adaptive_chunking"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_adaptive_chunking", e.target.checked)}
              />
            </div>
            <div className="bg-muted/30 rounded-lg p-3">
              <p className="text-xs font-medium mb-2">Type-Specific Chunk Sizes</p>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>Code: 256 chars (preserve function boundaries)</li>
                <li>Legal: 1024 chars (preserve paragraphs)</li>
                <li>Technical: 512 chars (preserve sections)</li>
                <li>Academic: 800 chars (preserve citations)</li>
                <li>General: 1000 chars (balanced)</li>
              </ul>
            </div>
          </div>

          {/* Hierarchical Chunking */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Hierarchical Chunking
            </h4>
            <p className="text-sm text-muted-foreground">
              Create multi-level chunk hierarchy for very large documents (20-30% token savings)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Hierarchical Chunking</p>
                <p className="text-sm text-muted-foreground">
                  Create document/section/detail chunk levels
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_hierarchical_chunking"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_hierarchical_chunking", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Size Threshold (KB)</label>
                <Input
                  type="number"
                  min="50"
                  placeholder="100"
                  value={(localSettings["ai.hierarchical_threshold_chars"] as number ?? 100000) / 1000}
                  onChange={(e) => handleSettingChange("ai.hierarchical_threshold_chars", parseInt(e.target.value) * 1000)}
                />
                <p className="text-xs text-muted-foreground">Enable for docs larger than this</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Target Sections</label>
                <Input
                  type="number"
                  min="3"
                  max="20"
                  placeholder="10"
                  value={localSettings["ai.sections_per_document"] as number ?? 10}
                  onChange={(e) => handleSettingChange("ai.sections_per_document", parseInt(e.target.value) || 5)}
                />
                <p className="text-xs text-muted-foreground">Number of section summaries</p>
              </div>
            </div>
          </div>

          {/* Cost Savings Info */}
          <div className="pt-4 border-t">
            <div className="bg-muted/50 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <DollarSign className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium text-sm">Estimated Savings</p>
                  <ul className="text-sm text-muted-foreground mt-1 space-y-1">
                    <li>Text Preprocessing: 10-20% token reduction</li>
                    <li>Document Summarization: 30-40% for large files</li>
                    <li>Semantic Caching: Up to 68% fewer API calls</li>
                    <li>Query Expansion: 8-12% accuracy improvement</li>
                    <li>Adaptive Chunking: 10-15% token savings</li>
                    <li>Hierarchical Chunking: 20-30% for large docs</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced RAG Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Advanced RAG Features
          </CardTitle>
          <CardDescription>Configure GraphRAG, Agentic RAG, Multimodal, and real-time features</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Retrieval Settings */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Search className="h-4 w-4" />
              Retrieval Settings
            </h4>
            <p className="text-sm text-muted-foreground">
              Configure how documents are retrieved and ranked for chat queries
            </p>

            <div className="grid gap-4 sm:grid-cols-2">
              {/* Top K Documents */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Documents to Retrieve</label>
                <Input
                  type="number"
                  min="3"
                  max="25"
                  value={localSettings["rag.top_k"] as number ?? 10}
                  onChange={(e) => handleSettingChange("rag.top_k", parseInt(e.target.value) || 5)}
                />
                <p className="text-xs text-muted-foreground">
                  How many documents to search (3-25). Higher = broader but slower.
                </p>
              </div>

              {/* Query Expansion Count */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Query Expansions</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  value={localSettings["rag.query_expansion_count"] as number ?? 3}
                  onChange={(e) => handleSettingChange("rag.query_expansion_count", parseInt(e.target.value) || 3)}
                />
                <p className="text-xs text-muted-foreground">
                  Query variations to try (1-5). More = better recall.
                </p>
              </div>

              {/* Similarity Threshold */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Similarity Threshold</label>
                <Input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="0.9"
                  value={localSettings["rag.similarity_threshold"] as number ?? 0.4}
                  onChange={(e) => handleSettingChange("rag.similarity_threshold", parseFloat(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum relevance score (0.1-0.9). Lower = more results.
                </p>
              </div>
            </div>

            {/* Reranking Toggle */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Result Reranking</p>
                <p className="text-sm text-muted-foreground">
                  Use cross-encoder AI to reorder results by semantic relevance (recommended)
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.rerank_results"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.rerank_results", e.target.checked)}
              />
            </div>
          </div>

          {/* GraphRAG */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Workflow className="h-4 w-4" />
              GraphRAG (Knowledge Graph)
            </h4>
            <p className="text-sm text-muted-foreground">
              Build knowledge graphs from documents for multi-hop reasoning
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable GraphRAG</p>
                <p className="text-sm text-muted-foreground">
                  Extract entities and relationships for graph-based retrieval
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.graphrag_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.graphrag_enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Graph Hops</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  value={localSettings["rag.graph_max_hops"] as number ?? 2}
                  onChange={(e) => handleSettingChange("rag.graph_max_hops", parseInt(e.target.value) || 2)}
                />
                <p className="text-xs text-muted-foreground">Traversal depth (1-5)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Graph Weight</label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localSettings["rag.graph_weight"] as number ?? 0.3}
                  onChange={(e) => handleSettingChange("rag.graph_weight", parseFloat(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Hybrid search weight (0-1)</p>
              </div>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Entity Extraction on Upload</p>
                <p className="text-sm text-muted-foreground">
                  Extract entities when processing documents
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.entity_extraction_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.entity_extraction_enabled", e.target.checked)}
              />
            </div>
          </div>

          {/* Agentic RAG */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Bot className="h-4 w-4" />
              Agentic RAG
            </h4>
            <p className="text-sm text-muted-foreground">
              Use AI agents for complex multi-step queries (ReAct loop)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Agentic RAG</p>
                <p className="text-sm text-muted-foreground">
                  Decompose complex queries into sub-questions
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.agentic_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("rag.agentic_enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Iterations</label>
                <Input
                  type="number"
                  min="1"
                  max="10"
                  value={localSettings["rag.agentic_max_iterations"] as number ?? 5}
                  onChange={(e) => handleSettingChange("rag.agentic_max_iterations", parseInt(e.target.value) || 5)}
                />
                <p className="text-xs text-muted-foreground">ReAct loop limit (1-10)</p>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Auto-detect Complexity</p>
                  <p className="text-sm text-muted-foreground">
                    Trigger agentic mode automatically
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["rag.auto_detect_complexity"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("rag.auto_detect_complexity", e.target.checked)}
                />
              </div>
            </div>
          </div>

          {/* Multimodal RAG */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Multimodal RAG
            </h4>
            <p className="text-sm text-muted-foreground">
              Process images, tables, and charts in documents
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Multimodal Processing</p>
                <p className="text-sm text-muted-foreground">
                  Caption images and extract tables during indexing
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.multimodal_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.multimodal_enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Vision Provider</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["rag.vision_provider"] as string || "auto"}
                  onChange={(e) => handleSettingChange("rag.vision_provider", e.target.value)}
                >
                  <option value="auto">Auto (Free first)</option>
                  <option value="ollama">Ollama (Free - Local)</option>
                  <option value="openai">OpenAI (Paid)</option>
                  <option value="anthropic">Anthropic (Paid)</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Ollama Vision Model</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["rag.ollama_vision_model"] as string || "llava"}
                  onChange={(e) => handleSettingChange("rag.ollama_vision_model", e.target.value)}
                >
                  <option value="llava">LLaVA (Recommended)</option>
                  <option value="bakllava">BakLLaVA</option>
                  <option value="llava:13b">LLaVA 13B</option>
                </select>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["rag.caption_images"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("rag.caption_images", e.target.checked)}
                />
                <span className="text-sm">Caption Images</span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["rag.extract_tables"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("rag.extract_tables", e.target.checked)}
                />
                <span className="text-sm">Extract Tables</span>
              </div>
            </div>
          </div>

          {/* Real-Time Updates */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Real-Time Updates
            </h4>
            <p className="text-sm text-muted-foreground">
              Incremental indexing and content freshness tracking
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Incremental Indexing</p>
                <p className="text-sm text-muted-foreground">
                  Only update changed chunks when reprocessing
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.incremental_indexing"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.incremental_indexing", e.target.checked)}
              />
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Freshness Tracking</p>
                <p className="text-sm text-muted-foreground">
                  Flag stale content based on age
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.freshness_tracking"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.freshness_tracking", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Aging Threshold (days)</label>
                <Input
                  type="number"
                  min="7"
                  value={localSettings["rag.freshness_threshold_days"] as number ?? 30}
                  onChange={(e) => handleSettingChange("rag.freshness_threshold_days", parseInt(e.target.value) || 7)}
                />
                <p className="text-xs text-muted-foreground">Days until content is aging</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Stale Threshold (days)</label>
                <Input
                  type="number"
                  min="30"
                  value={localSettings["rag.stale_threshold_days"] as number ?? 90}
                  onChange={(e) => handleSettingChange("rag.stale_threshold_days", parseInt(e.target.value) || 90)}
                />
                <p className="text-xs text-muted-foreground">Days until content is stale</p>
              </div>
            </div>
          </div>

          {/* Memory Services (Phase 40/48) */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Agent Memory (Mem0/A-Mem)
            </h4>
            <p className="text-sm text-muted-foreground">
              Enable persistent memory for context retention across conversations (91% lower latency, 90% token savings)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Agent Memory</p>
                <p className="text-sm text-muted-foreground">
                  Remember facts and preferences across sessions
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["memory.enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("memory.enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Memory Provider</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={localSettings["memory.provider"] as string ?? "mem0"}
                  onChange={(e) => handleSettingChange("memory.provider", e.target.value)}
                >
                  <option value="mem0">Mem0 (Recommended)</option>
                  <option value="amem">A-Mem (Agentic)</option>
                </select>
                <p className="text-xs text-muted-foreground">Memory backend to use</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Memory Entries</label>
                <Input
                  type="number"
                  min="100"
                  max="10000"
                  value={localSettings["memory.max_entries"] as number ?? 1000}
                  onChange={(e) => handleSettingChange("memory.max_entries", parseInt(e.target.value) || 1000)}
                />
                <p className="text-xs text-muted-foreground">Per-user memory limit</p>
              </div>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Memory Decay</p>
                <p className="text-sm text-muted-foreground">
                  Automatically deprioritize old memories
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["memory.decay_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("memory.decay_enabled", e.target.checked)}
              />
            </div>
          </div>

          {/* Context Compression (Phase 38) */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Context Compression
            </h4>
            <p className="text-sm text-muted-foreground">
              Compress long conversations to reduce token costs (32x compression, 85% cost reduction)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Context Compression</p>
                <p className="text-sm text-muted-foreground">
                  Summarize older turns while keeping recent ones verbatim
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["compression.enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("compression.enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Recent Turns (Verbatim)</label>
                <Input
                  type="number"
                  min="3"
                  max="20"
                  value={localSettings["compression.recent_turns"] as number ?? 10}
                  onChange={(e) => handleSettingChange("compression.recent_turns", parseInt(e.target.value) || 10)}
                />
                <p className="text-xs text-muted-foreground">Keep last N turns uncompressed</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Compression Level</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={localSettings["compression.level"] as string ?? "moderate"}
                  onChange={(e) => handleSettingChange("compression.level", e.target.value)}
                >
                  <option value="minimal">Minimal (Keep more context)</option>
                  <option value="moderate">Moderate (Balanced)</option>
                  <option value="aggressive">Aggressive (Max savings)</option>
                </select>
                <p className="text-xs text-muted-foreground">How aggressively to compress</p>
              </div>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Anchor Important Facts</p>
                <p className="text-sm text-muted-foreground">
                  Never compress critical information (names, dates, etc.)
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["compression.enable_anchoring"] as boolean ?? true}
                onChange={(e) => handleSettingChange("compression.enable_anchoring", e.target.checked)}
              />
            </div>
          </div>

          {/* Tiered Reranking (Phase 43) - Enhanced */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Tiered Reranking Pipeline
            </h4>
            <p className="text-sm text-muted-foreground">
              Multi-stage reranking for optimal retrieval (ColBERT → Cross-Encoder → LLM)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Tiered Reranking</p>
                <p className="text-sm text-muted-foreground">
                  Use 3-stage pipeline for 87% multi-hop accuracy
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rerank.tiered_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rerank.tiered_enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Stage 1 Top-K</label>
                <Input
                  type="number"
                  min="20"
                  max="200"
                  value={localSettings["rerank.stage1_top_k"] as number ?? 100}
                  onChange={(e) => handleSettingChange("rerank.stage1_top_k", parseInt(e.target.value) || 100)}
                />
                <p className="text-xs text-muted-foreground">ColBERT candidates</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Stage 2 Top-K</label>
                <Input
                  type="number"
                  min="5"
                  max="50"
                  value={localSettings["rerank.stage2_top_k"] as number ?? 20}
                  onChange={(e) => handleSettingChange("rerank.stage2_top_k", parseInt(e.target.value) || 20)}
                />
                <p className="text-xs text-muted-foreground">Cross-encoder output</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Final Top-K</label>
                <Input
                  type="number"
                  min="3"
                  max="20"
                  value={localSettings["rerank.final_top_k"] as number ?? 5}
                  onChange={(e) => handleSettingChange("rerank.final_top_k", parseInt(e.target.value) || 5)}
                />
                <p className="text-xs text-muted-foreground">Results returned</p>
              </div>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable LLM Reranking (Stage 3)</p>
                <p className="text-sm text-muted-foreground">
                  Use LLM for final reranking (higher quality, more cost)
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rerank.use_llm_stage"] as boolean ?? false}
                onChange={(e) => handleSettingChange("rerank.use_llm_stage", e.target.checked)}
              />
            </div>
          </div>

          {/* Advanced Retrieval Methods (Phase 46/49) */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Advanced Retrieval Methods
            </h4>
            <p className="text-sm text-muted-foreground">
              State-of-the-art retrieval architectures for improved accuracy and reduced hallucinations
            </p>

            {/* SELF-RAG */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium flex items-center gap-2">
                  <Shield className="h-4 w-4 text-blue-500" />
                  SELF-RAG (Self-Reflective RAG)
                </p>
                <p className="text-sm text-muted-foreground">
                  Reduces hallucinations by 30% with self-verification and correction
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.self_rag_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.self_rag_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["rag.self_rag_enabled"] as boolean ?? true) && (
              <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Confidence Threshold</label>
                    <Input
                      type="number"
                      step="0.1"
                      min="0.5"
                      max="0.95"
                      value={localSettings["rag.self_rag_confidence_threshold"] as number ?? 0.7}
                      onChange={(e) => handleSettingChange("rag.self_rag_confidence_threshold", parseFloat(e.target.value))}
                    />
                    <p className="text-xs text-muted-foreground">Min confidence for verification (0.5-0.95)</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Retries</label>
                    <Input
                      type="number"
                      min="1"
                      max="5"
                      value={localSettings["rag.self_rag_max_retries"] as number ?? 2}
                      onChange={(e) => handleSettingChange("rag.self_rag_max_retries", parseInt(e.target.value) || 2)}
                    />
                    <p className="text-xs text-muted-foreground">Retry count for low-confidence answers</p>
                  </div>
                </div>
              </div>
            )}

            {/* LightRAG */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium flex items-center gap-2">
                  <Layers className="h-4 w-4 text-purple-500" />
                  LightRAG (Dual-Level Retrieval)
                </p>
                <p className="text-sm text-muted-foreground">
                  10x token reduction with entity-based and relationship-aware retrieval
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.lightrag_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.lightrag_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["rag.lightrag_enabled"] as boolean ?? true) && (
              <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">LightRAG Mode</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={localSettings["rag.lightrag_mode"] as string ?? "hybrid"}
                      onChange={(e) => handleSettingChange("rag.lightrag_mode", e.target.value)}
                    >
                      <option value="local">Local (Entity-focused)</option>
                      <option value="global">Global (Relationship-focused)</option>
                      <option value="hybrid">Hybrid (Both - Recommended)</option>
                    </select>
                    <p className="text-xs text-muted-foreground">Retrieval strategy</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Entities</label>
                    <Input
                      type="number"
                      min="5"
                      max="50"
                      value={localSettings["rag.lightrag_max_entities"] as number ?? 20}
                      onChange={(e) => handleSettingChange("rag.lightrag_max_entities", parseInt(e.target.value) || 20)}
                    />
                    <p className="text-xs text-muted-foreground">Max entities per query</p>
                  </div>
                </div>
              </div>
            )}

            {/* RAPTOR */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium flex items-center gap-2">
                  <TreePine className="h-4 w-4 text-green-500" />
                  RAPTOR (Tree-Organized Retrieval)
                </p>
                <p className="text-sm text-muted-foreground">
                  97% token reduction with hierarchical document tree for multi-level understanding
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.raptor_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.raptor_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["rag.raptor_enabled"] as boolean ?? true) && (
              <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Tree Depth</label>
                    <Input
                      type="number"
                      min="2"
                      max="6"
                      value={localSettings["rag.raptor_tree_depth"] as number ?? 3}
                      onChange={(e) => handleSettingChange("rag.raptor_tree_depth", parseInt(e.target.value) || 3)}
                    />
                    <p className="text-xs text-muted-foreground">Hierarchy levels (2-6)</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Cluster Size</label>
                    <Input
                      type="number"
                      min="3"
                      max="15"
                      value={localSettings["rag.raptor_cluster_size"] as number ?? 5}
                      onChange={(e) => handleSettingChange("rag.raptor_cluster_size", parseInt(e.target.value) || 5)}
                    />
                    <p className="text-xs text-muted-foreground">Chunks per cluster</p>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border bg-background">
                  <div>
                    <p className="font-medium text-sm">Use Summary Nodes</p>
                    <p className="text-xs text-muted-foreground">
                      Generate summaries at each tree level
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={localSettings["rag.raptor_use_summaries"] as boolean ?? true}
                    onChange={(e) => handleSettingChange("rag.raptor_use_summaries", e.target.checked)}
                  />
                </div>
              </div>
            )}

            {/* Advanced Retrieval Info Box */}
            <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Shield className="h-5 w-5 text-blue-500 mt-0.5" />
                <div>
                  <p className="font-medium text-sm">How These Work Together</p>
                  <ul className="text-sm text-muted-foreground mt-1 space-y-1">
                    <li><strong>SELF-RAG:</strong> Verifies answers and re-retrieves if confidence is low</li>
                    <li><strong>LightRAG:</strong> Uses entity graphs for relationship-aware retrieval</li>
                    <li><strong>RAPTOR:</strong> Retrieves from hierarchical summaries for broad understanding</li>
                    <li>All three can be enabled together for maximum accuracy</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Query Suggestions */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Query Suggestions
            </h4>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Suggestions</p>
                <p className="text-sm text-muted-foreground">
                  Suggest follow-up questions after answers
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.suggested_questions_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.suggested_questions_enabled", e.target.checked)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Suggestions Count</label>
              <Input
                type="number"
                min="1"
                max="5"
                value={localSettings["rag.suggestions_count"] as number ?? 3}
                onChange={(e) => handleSettingChange("rag.suggestions_count", parseInt(e.target.value) || 3)}
              />
              <p className="text-xs text-muted-foreground">Number of suggestions (1-5)</p>
            </div>
          </div>

          {/* Phase 62/63 Advanced Processing */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Advanced Processing (Phase 62/63)
            </h4>
            <p className="text-sm text-muted-foreground">
              Cutting-edge document processing and answer quality features
            </p>

            {/* Answer Refiner */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Answer Refiner</p>
                <p className="text-sm text-muted-foreground">
                  Self-Refine/CRITIC for +20% answer quality (NeurIPS 2023)
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.answer_refiner_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("rag.answer_refiner_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["rag.answer_refiner_enabled"] as boolean) && (
              <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Refinement Strategy</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={localSettings["rag.answer_refiner_strategy"] as string ?? "self_refine"}
                      onChange={(e) => handleSettingChange("rag.answer_refiner_strategy", e.target.value)}
                    >
                      <option value="self_refine">Self-Refine (General quality)</option>
                      <option value="critic">CRITIC (Tool-verified facts)</option>
                      <option value="cove">CoVe (Hallucination reduction)</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Iterations</label>
                    <Input
                      type="number"
                      min="1"
                      max="5"
                      value={localSettings["rag.answer_refiner_max_iterations"] as number ?? 2}
                      onChange={(e) => handleSettingChange("rag.answer_refiner_max_iterations", parseInt(e.target.value) || 2)}
                    />
                    <p className="text-xs text-muted-foreground">Refinement passes (1-5)</p>
                  </div>
                </div>
              </div>
            )}

            {/* TTT Compression */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable TTT Compression</p>
                <p className="text-sm text-muted-foreground">
                  Test-Time Training for 35x faster inference on 2M+ context (NVIDIA)
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.ttt_compression_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("rag.ttt_compression_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["rag.ttt_compression_enabled"] as boolean) && (
              <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
                <label className="text-sm font-medium">Compression Ratio</label>
                <Input
                  type="number"
                  min="0.3"
                  max="0.8"
                  step="0.1"
                  value={localSettings["rag.ttt_compression_ratio"] as number ?? 0.5}
                  onChange={(e) => handleSettingChange("rag.ttt_compression_ratio", parseFloat(e.target.value) || 0.5)}
                />
                <p className="text-xs text-muted-foreground">Target ratio (0.3-0.8, lower = more compression)</p>
              </div>
            )}

            {/* Sufficiency Checker */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Sufficiency Checker</p>
                <p className="text-sm text-muted-foreground">
                  ICLR 2025 technique to skip unnecessary retrieval rounds
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.sufficiency_checker_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("rag.sufficiency_checker_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["rag.sufficiency_checker_enabled"] as boolean) && (
              <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
                <label className="text-sm font-medium">Sufficiency Threshold</label>
                <Input
                  type="number"
                  min="0.5"
                  max="0.9"
                  step="0.05"
                  value={localSettings["rag.sufficiency_threshold"] as number ?? 0.7}
                  onChange={(e) => handleSettingChange("rag.sufficiency_threshold", parseFloat(e.target.value) || 0.7)}
                />
                <p className="text-xs text-muted-foreground">Confidence threshold (0.5-0.9)</p>
              </div>
            )}

            {/* Tree of Thoughts */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Tree of Thoughts</p>
                <p className="text-sm text-muted-foreground">
                  Multi-path reasoning for complex analytical queries
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.tree_of_thoughts_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("rag.tree_of_thoughts_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["rag.tree_of_thoughts_enabled"] as boolean) && (
              <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Depth</label>
                    <Input
                      type="number"
                      min="2"
                      max="5"
                      value={localSettings["rag.tot_max_depth"] as number ?? 3}
                      onChange={(e) => handleSettingChange("rag.tot_max_depth", parseInt(e.target.value) || 3)}
                    />
                    <p className="text-xs text-muted-foreground">Tree depth (2-5)</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Branching Factor</label>
                    <Input
                      type="number"
                      min="2"
                      max="5"
                      value={localSettings["rag.tot_branching_factor"] as number ?? 3}
                      onChange={(e) => handleSettingChange("rag.tot_branching_factor", parseInt(e.target.value) || 3)}
                    />
                    <p className="text-xs text-muted-foreground">Branches per node (2-5)</p>
                  </div>
                </div>
              </div>
            )}

            {/* Fast Chunking */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Fast Chunking</p>
                <p className="text-sm text-muted-foreground">
                  Chonkie-based chunking (33x faster, 10-50x less memory)
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["processing.fast_chunking_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("processing.fast_chunking_enabled", e.target.checked)}
              />
            </div>
            {(localSettings["processing.fast_chunking_enabled"] as boolean) && (
              <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
                <label className="text-sm font-medium">Chunking Strategy</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["processing.fast_chunking_strategy"] as string ?? "auto"}
                  onChange={(e) => handleSettingChange("processing.fast_chunking_strategy", e.target.value)}
                >
                  <option value="auto">Auto (Select based on doc size)</option>
                  <option value="token">Token (Fastest)</option>
                  <option value="sentence">Sentence (Fast)</option>
                  <option value="semantic">Semantic (Balanced)</option>
                  <option value="sdpm">SDPM (Best quality)</option>
                </select>
              </div>
            )}

            {/* Docling Parser */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Docling Parser</p>
                <p className="text-sm text-muted-foreground">
                  Enterprise document parsing with 97.9% table accuracy
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["processing.docling_parser_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("processing.docling_parser_enabled", e.target.checked)}
              />
            </div>

            {/* Agent Evaluation */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Agent Evaluation</p>
                <p className="text-sm text-muted-foreground">
                  Pass^k metrics, hallucination detection, progress tracking
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["agent.evaluation_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("agent.evaluation_enabled", e.target.checked)}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
