"use client";

import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { EmbeddingDashboard } from "@/components/embedding-dashboard";
import { Sparkles, FileText, MessageSquare, Search, Zap, Cog, Activity, Workflow, Bot, RefreshCw, DollarSign, Shield, Layers, TreePine, Brain, HardDrive, GitBranch } from "lucide-react";

interface RagTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function RagTab({ localSettings, handleSettingChange }: RagTabProps) {
  return (
    <TabsContent value="rag" className="space-y-6">
      {/* Embedding System Status Dashboard */}
      <EmbeddingDashboard />

      {/* Embedding Provider Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Embedding Provider
          </CardTitle>
          <CardDescription>
            Configure which provider and model to use for generating document embeddings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Provider</label>
              <Select
                value={localSettings["embedding.provider"] as string ?? "ollama"}
                onValueChange={(value) => handleSettingChange("embedding.provider", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ollama">Ollama (Local)</SelectItem>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="voyage">Voyage AI</SelectItem>
                  <SelectItem value="cohere">Cohere</SelectItem>
                  <SelectItem value="gemini">Google Gemini</SelectItem>
                  <SelectItem value="auto">Auto-select</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Ollama is free and runs locally. Cloud providers require API keys configured in Providers tab.
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Model</label>
              <Input
                value={localSettings["embedding.model"] as string ?? ""}
                onChange={(e) => handleSettingChange("embedding.model", e.target.value)}
                placeholder={
                  (localSettings["embedding.provider"] as string) === "ollama"
                    ? "nomic-embed-text"
                    : "text-embedding-3-small"
                }
              />
              <p className="text-xs text-muted-foreground">
                Leave empty to use provider default. For Ollama: nomic-embed-text (768D), mxbai-embed-large (1024D)
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Auto-select Model</p>
              <p className="text-sm text-muted-foreground">
                Automatically choose embedding model based on content type (code, multilingual, etc.)
              </p>
            </div>
            <Switch
              checked={localSettings["embedding.auto_select_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("embedding.auto_select_enabled", checked)}
            />
          </div>

          <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg text-sm text-muted-foreground">
            Changes take effect for new document uploads. Existing documents retain their original embeddings.
            Use the &quot;Reprocess&quot; option on individual documents to regenerate embeddings with new settings.
          </div>
        </CardContent>
      </Card>

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
                <Switch
                  checked={localSettings["ai.enable_preprocessing"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("ai.enable_preprocessing", checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Remove Boilerplate</p>
                  <p className="text-sm text-muted-foreground">
                    Strip headers, footers, page numbers
                  </p>
                </div>
                <Switch
                  checked={localSettings["ai.remove_boilerplate"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("ai.remove_boilerplate", checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Normalize Whitespace</p>
                  <p className="text-sm text-muted-foreground">
                    Collapse multiple spaces/newlines
                  </p>
                </div>
                <Switch
                  checked={localSettings["ai.normalize_whitespace"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("ai.normalize_whitespace", checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Deduplicate Content</p>
                  <p className="text-sm text-muted-foreground">
                    Remove near-duplicate chunks
                  </p>
                </div>
                <Switch
                  checked={localSettings["ai.deduplicate_content"] as boolean ?? false}
                  onCheckedChange={(checked) => handleSettingChange("ai.deduplicate_content", checked)}
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
              <Switch
                checked={localSettings["ai.enable_summarization"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("ai.enable_summarization", checked)}
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
              <Select
                value={localSettings["ai.summarization_model"] as string || "gpt-4o-mini"}
                onValueChange={(value) => handleSettingChange("ai.summarization_model", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4o-mini">GPT-4o Mini (Cost-effective)</SelectItem>
                  <SelectItem value="gpt-4o">GPT-4o (Higher quality)</SelectItem>
                  <SelectItem value="claude-3-haiku">Claude 3 Haiku (Fast)</SelectItem>
                </SelectContent>
              </Select>
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
              <Switch
                checked={localSettings["rag.semantic_cache_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("rag.semantic_cache_enabled", checked)}
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
                  value={localSettings["rag.semantic_similarity_threshold"] as number ?? 0.95}
                  onChange={(e) => handleSettingChange("rag.semantic_similarity_threshold", parseFloat(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Higher = stricter matching (0.8-1.0)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Cache Entries</label>
                <Input
                  type="number"
                  placeholder="10000"
                  value={localSettings["rag.max_semantic_cache_entries"] as number ?? 10000}
                  onChange={(e) => handleSettingChange("rag.max_semantic_cache_entries", parseInt(e.target.value) || 1000)}
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
              <Switch
                checked={localSettings["rag.query_expansion_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("rag.query_expansion_enabled", checked)}
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
                  value={localSettings["rag.query_expansion_count"] as number ?? 2}
                  onChange={(e) => handleSettingChange("rag.query_expansion_count", parseInt(e.target.value) || 2)}
                />
                <p className="text-xs text-muted-foreground">Number of variations to generate (1-5)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Expansion Model</label>
                <Select
                  value={localSettings["rag.query_expansion_model"] as string || "gpt-4o-mini"}
                  onValueChange={(value) => handleSettingChange("rag.query_expansion_model", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gpt-4o-mini">GPT-4o Mini (Cost-effective)</SelectItem>
                    <SelectItem value="gpt-4o">GPT-4o (Higher quality)</SelectItem>
                    <SelectItem value="claude-3-haiku">Claude 3 Haiku (Fast)</SelectItem>
                  </SelectContent>
                </Select>
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
              <Switch
                checked={localSettings["ai.enable_adaptive_chunking"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("ai.enable_adaptive_chunking", checked)}
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
              <Switch
                checked={localSettings["ai.enable_hierarchical_chunking"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("ai.enable_hierarchical_chunking", checked)}
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
              <Switch
                checked={localSettings["rag.rerank_results"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.rerank_results", checked)}
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
              <Switch
                checked={localSettings["rag.graphrag_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.graphrag_enabled", checked)}
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
              <Switch
                checked={localSettings["rag.entity_extraction_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.entity_extraction_enabled", checked)}
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
              <Switch
                checked={localSettings["rag.agentic_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("rag.agentic_enabled", checked)}
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
                <Switch
                  checked={localSettings["rag.auto_detect_complexity"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("rag.auto_detect_complexity", checked)}
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
              <Switch
                checked={localSettings["rag.multimodal_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.multimodal_enabled", checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Vision Provider</label>
                <Select
                  value={localSettings["rag.vision_provider"] as string || "auto"}
                  onValueChange={(value) => handleSettingChange("rag.vision_provider", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Free first)</SelectItem>
                    <SelectItem value="ollama">Ollama (Free - Local)</SelectItem>
                    <SelectItem value="openai">OpenAI (Paid)</SelectItem>
                    <SelectItem value="anthropic">Anthropic (Paid)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Ollama Vision Model</label>
                <Select
                  value={localSettings["rag.ollama_vision_model"] as string || "llava"}
                  onValueChange={(value) => handleSettingChange("rag.ollama_vision_model", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="llava">LLaVA (Recommended)</SelectItem>
                    <SelectItem value="bakllava">BakLLaVA</SelectItem>
                    <SelectItem value="llava:13b">LLaVA 13B</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Switch
                  checked={localSettings["rag.caption_images"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("rag.caption_images", checked)}
                />
                <span className="text-sm">Caption Images</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  checked={localSettings["rag.extract_tables"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("rag.extract_tables", checked)}
                />
                <span className="text-sm">Extract Tables</span>
              </div>
            </div>

            {/* Image Analysis Settings */}
            <div className="space-y-4 pt-3 mt-3 border-t border-dashed">
              <h5 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Image Analysis Options
              </h5>

              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Enable Image Analysis</p>
                  <p className="text-sm text-muted-foreground">
                    Analyze images with AI during document processing
                  </p>
                </div>
                <Switch
                  checked={localSettings["rag.image_analysis_enabled"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("rag.image_analysis_enabled", checked)}
                />
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Skip Duplicate Images</p>
                  <p className="text-sm text-muted-foreground">
                    Reuse cached analysis for identical images (saves time and cost)
                  </p>
                </div>
                <Switch
                  checked={localSettings["rag.image_duplicate_detection"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("rag.image_duplicate_detection", checked)}
                />
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Max Images per Document</label>
                  <Input
                    type="number"
                    min="0"
                    max="200"
                    value={localSettings["rag.max_images_per_document"] as number ?? 50}
                    onChange={(e) => handleSettingChange("rag.max_images_per_document", parseInt(e.target.value) || 50)}
                  />
                  <p className="text-xs text-muted-foreground">Limit images to analyze (0 = unlimited)</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Min Image Size (KB)</label>
                  <Input
                    type="number"
                    min="0"
                    max="100"
                    value={localSettings["rag.min_image_size_kb"] as number ?? 5}
                    onChange={(e) => handleSettingChange("rag.min_image_size_kb", parseInt(e.target.value) || 5)}
                  />
                  <p className="text-xs text-muted-foreground">Skip tiny icons and decorations</p>
                </div>
              </div>

              <div className="bg-amber-50 dark:bg-amber-950/30 rounded-lg p-3 text-xs">
                <p className="font-medium text-amber-700 dark:text-amber-400">Free Vision with Ollama</p>
                <p className="text-amber-600 dark:text-amber-500 mt-1">
                  Run <code className="bg-amber-100 dark:bg-amber-900/50 px-1 rounded">ollama pull llava</code> for free local image analysis. No API costs!
                </p>
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
              <Switch
                checked={localSettings["rag.incremental_indexing"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.incremental_indexing", checked)}
              />
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Freshness Tracking</p>
                <p className="text-sm text-muted-foreground">
                  Flag stale content based on age
                </p>
              </div>
              <Switch
                checked={localSettings["rag.freshness_tracking"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.freshness_tracking", checked)}
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
              <Switch
                checked={localSettings["memory.enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("memory.enabled", checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Memory Provider</label>
                <Select
                  value={localSettings["memory.provider"] as string ?? "mem0"}
                  onValueChange={(value) => handleSettingChange("memory.provider", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mem0">Mem0 (Recommended)</SelectItem>
                    <SelectItem value="amem">A-Mem (Agentic)</SelectItem>
                  </SelectContent>
                </Select>
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
              <Switch
                checked={localSettings["memory.decay_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("memory.decay_enabled", checked)}
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
              <Switch
                checked={localSettings["compression.enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("compression.enabled", checked)}
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
                <Select
                  value={localSettings["compression.level"] as string ?? "moderate"}
                  onValueChange={(value) => handleSettingChange("compression.level", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="minimal">Minimal (Keep more context)</SelectItem>
                    <SelectItem value="moderate">Moderate (Balanced)</SelectItem>
                    <SelectItem value="aggressive">Aggressive (Max savings)</SelectItem>
                  </SelectContent>
                </Select>
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
              <Switch
                checked={localSettings["compression.enable_anchoring"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("compression.enable_anchoring", checked)}
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
              <Switch
                checked={localSettings["rerank.tiered_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rerank.tiered_enabled", checked)}
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
              <Switch
                checked={localSettings["rerank.use_llm_stage"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("rerank.use_llm_stage", checked)}
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
              <Switch
                checked={localSettings["rag.self_rag_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.self_rag_enabled", checked)}
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
              <Switch
                checked={localSettings["rag.lightrag_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.lightrag_enabled", checked)}
              />
            </div>
            {(localSettings["rag.lightrag_enabled"] as boolean ?? true) && (
              <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">LightRAG Mode</label>
                    <Select
                      value={localSettings["rag.lightrag_mode"] as string ?? "hybrid"}
                      onValueChange={(value) => handleSettingChange("rag.lightrag_mode", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="local">Local (Entity-focused)</SelectItem>
                        <SelectItem value="global">Global (Relationship-focused)</SelectItem>
                        <SelectItem value="hybrid">Hybrid (Both - Recommended)</SelectItem>
                      </SelectContent>
                    </Select>
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
              <Switch
                checked={localSettings["rag.raptor_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.raptor_enabled", checked)}
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
                  <Switch
                    checked={localSettings["rag.raptor_use_summaries"] as boolean ?? true}
                    onCheckedChange={(checked) => handleSettingChange("rag.raptor_use_summaries", checked)}
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
              <Switch
                checked={localSettings["rag.suggested_questions_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.suggested_questions_enabled", checked)}
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
              <Switch
                checked={localSettings["rag.answer_refiner_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("rag.answer_refiner_enabled", checked)}
              />
            </div>
            {(localSettings["rag.answer_refiner_enabled"] as boolean) && (
              <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Refinement Strategy</label>
                    <Select
                      value={localSettings["rag.answer_refiner_strategy"] as string ?? "self_refine"}
                      onValueChange={(value) => handleSettingChange("rag.answer_refiner_strategy", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="self_refine">Self-Refine (General quality)</SelectItem>
                        <SelectItem value="critic">CRITIC (Tool-verified facts)</SelectItem>
                        <SelectItem value="cove">CoVe (Hallucination reduction)</SelectItem>
                      </SelectContent>
                    </Select>
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
              <Switch
                checked={localSettings["rag.ttt_compression_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("rag.ttt_compression_enabled", checked)}
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
              <Switch
                checked={localSettings["rag.sufficiency_checker_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("rag.sufficiency_checker_enabled", checked)}
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
              <Switch
                checked={localSettings["rag.tree_of_thoughts_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("rag.tree_of_thoughts_enabled", checked)}
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
              <Switch
                checked={localSettings["processing.fast_chunking_enabled"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("processing.fast_chunking_enabled", checked)}
              />
            </div>
            {(localSettings["processing.fast_chunking_enabled"] as boolean) && (
              <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
                <label className="text-sm font-medium">Chunking Strategy</label>
                <Select
                  value={localSettings["processing.fast_chunking_strategy"] as string ?? "auto"}
                  onValueChange={(value) => handleSettingChange("processing.fast_chunking_strategy", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Select based on doc size)</SelectItem>
                    <SelectItem value="token">Token (Fastest)</SelectItem>
                    <SelectItem value="sentence">Sentence (Fast)</SelectItem>
                    <SelectItem value="semantic">Semantic (Balanced)</SelectItem>
                    <SelectItem value="sdpm">SDPM (Best quality)</SelectItem>
                  </SelectContent>
                </Select>
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
              <Switch
                checked={localSettings["processing.docling_parser_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("processing.docling_parser_enabled", checked)}
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
              <Switch
                checked={localSettings["agent.evaluation_enabled"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("agent.evaluation_enabled", checked)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Query Intelligence */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Query Intelligence
          </CardTitle>
          <CardDescription>
            Smart query processing — routing, decomposition, fusion, and expansion techniques
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Adaptive Query Routing */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-indigo-500" />
                Adaptive Query Routing
              </p>
              <p className="text-sm text-muted-foreground">
                Automatically routes queries to optimal strategy (DIRECT, HYBRID, TWO_STAGE, AGENTIC, GRAPH)
              </p>
            </div>
            <Switch
              checked={localSettings["rag.adaptive_routing_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.adaptive_routing_enabled", checked)}
            />
          </div>

          {/* Query Decomposition */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Query Decomposition</p>
              <p className="text-sm text-muted-foreground">
                Break complex queries into sub-queries for better accuracy on comparison/aggregation questions
              </p>
            </div>
            <Switch
              checked={localSettings["rag.query_decomposition_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.query_decomposition_enabled", checked)}
            />
          </div>
          {(localSettings["rag.query_decomposition_enabled"] as boolean ?? true) && (
            <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
              <label className="text-sm font-medium">Min Query Words</label>
              <Input
                type="number"
                min="5"
                max="20"
                value={localSettings["rag.decomposition_min_words"] as number ?? 10}
                onChange={(e) => handleSettingChange("rag.decomposition_min_words", parseInt(e.target.value) || 10)}
              />
              <p className="text-xs text-muted-foreground">Only decompose queries longer than this (5-20 words)</p>
            </div>
          )}

          {/* RAG-Fusion */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">RAG-Fusion</p>
              <p className="text-sm text-muted-foreground">
                Generate multiple query variations and fuse results with Reciprocal Rank Fusion
              </p>
            </div>
            <Switch
              checked={localSettings["rag.rag_fusion_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.rag_fusion_enabled", checked)}
            />
          </div>
          {(localSettings["rag.rag_fusion_enabled"] as boolean ?? true) && (
            <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
              <label className="text-sm font-medium">Query Variations</label>
              <Input
                type="number"
                min="2"
                max="8"
                value={localSettings["rag.rag_fusion_variations"] as number ?? 4}
                onChange={(e) => handleSettingChange("rag.rag_fusion_variations", parseInt(e.target.value) || 4)}
              />
              <p className="text-xs text-muted-foreground">Number of query variations to generate (2-8)</p>
            </div>
          )}

          {/* Step-Back Prompting */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Step-Back Prompting</p>
              <p className="text-sm text-muted-foreground">
                Abstract reasoning for complex analytical queries — retrieves background context first
              </p>
            </div>
            <Switch
              checked={localSettings["rag.stepback_prompting_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.stepback_prompting_enabled", checked)}
            />
          </div>
          {(localSettings["rag.stepback_prompting_enabled"] as boolean ?? true) && (
            <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
              <label className="text-sm font-medium">Max Background Chunks</label>
              <Input
                type="number"
                min="1"
                max="5"
                value={localSettings["rag.stepback_max_background"] as number ?? 3}
                onChange={(e) => handleSettingChange("rag.stepback_max_background", parseInt(e.target.value) || 3)}
              />
              <p className="text-xs text-muted-foreground">Background chunks for step-back context (1-5)</p>
            </div>
          )}

          {/* HyDE */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">HyDE (Hypothetical Document Embeddings)</p>
              <p className="text-sm text-muted-foreground">
                Generate hypothetical documents for short/abstract queries to improve recall
              </p>
            </div>
            <Switch
              checked={localSettings["rag.hyde_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.hyde_enabled", checked)}
            />
          </div>
          {(localSettings["rag.hyde_enabled"] as boolean ?? true) && (
            <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
              <label className="text-sm font-medium">Min Query Words</label>
              <Input
                type="number"
                min="2"
                max="10"
                value={localSettings["rag.hyde_min_query_words"] as number ?? 5}
                onChange={(e) => handleSettingChange("rag.hyde_min_query_words", parseInt(e.target.value) || 5)}
              />
              <p className="text-xs text-muted-foreground">Use HyDE only for queries shorter than this (2-10)</p>
            </div>
          )}

          {/* Learning to Rank */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Learning to Rank</p>
              <p className="text-sm text-muted-foreground">
                ML-based reranking trained on click logs and user feedback
              </p>
            </div>
            <Switch
              checked={localSettings["search.ltr_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("search.ltr_enabled", checked)}
            />
          </div>

          {/* Info Box */}
          <div className="bg-indigo-50 dark:bg-indigo-950/30 rounded-lg p-4 mt-2">
            <div className="flex items-start gap-3">
              <GitBranch className="h-5 w-5 text-indigo-500 mt-0.5" />
              <div>
                <p className="font-medium text-sm">How These Work Together</p>
                <p className="text-sm text-muted-foreground mt-1">
                  When Adaptive Routing is enabled, it automatically selects the optimal combination of these features per query. Simple queries skip expensive steps, while complex multi-hop queries use decomposition + fusion + step-back together.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Compression & Retrieval */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Advanced Compression & Retrieval
          </CardTitle>
          <CardDescription>
            Context compression and retrieval optimization techniques
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* AttentionRAG */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium flex items-center gap-2">
                <Zap className="h-4 w-4 text-amber-500" />
                AttentionRAG Compression
              </p>
              <p className="text-sm text-muted-foreground">
                6.3x better compression than LLMLingua — uses attention scores for intelligent context pruning
              </p>
            </div>
            <Switch
              checked={localSettings["rag.attention_rag_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("rag.attention_rag_enabled", checked)}
            />
          </div>
          {(localSettings["rag.attention_rag_enabled"] as boolean) && (
            <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Compression Mode</label>
                  <Select
                    value={localSettings["rag.attention_rag_mode"] as string ?? "moderate"}
                    onValueChange={(value) => handleSettingChange("rag.attention_rag_mode", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light (1.25x)</SelectItem>
                      <SelectItem value="moderate">Moderate (2x)</SelectItem>
                      <SelectItem value="aggressive">Aggressive (4x)</SelectItem>
                      <SelectItem value="extreme">Extreme (6.6x)</SelectItem>
                      <SelectItem value="adaptive">Adaptive (auto)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Higher = more compression, less context</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Compression Unit</label>
                  <Select
                    value={localSettings["rag.attention_rag_unit"] as string ?? "sentence"}
                    onValueChange={(value) => handleSettingChange("rag.attention_rag_unit", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="token">Token (Fine-grained)</SelectItem>
                      <SelectItem value="sentence">Sentence (Balanced)</SelectItem>
                      <SelectItem value="paragraph">Paragraph (Coarse)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Granularity of compression</p>
                </div>
              </div>
            </div>
          )}

          {/* CRAG */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium flex items-center gap-2">
                <RefreshCw className="h-4 w-4 text-orange-500" />
                CRAG (Corrective RAG)
              </p>
              <p className="text-sm text-muted-foreground">
                Auto-corrects low-confidence results — rewrites query and re-retrieves when needed
              </p>
            </div>
            <Switch
              checked={localSettings["rag.crag_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.crag_enabled", checked)}
            />
          </div>
          {(localSettings["rag.crag_enabled"] as boolean ?? true) && (
            <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
              <label className="text-sm font-medium">Confidence Threshold</label>
              <Input
                type="number"
                step="0.1"
                min="0.1"
                max="0.9"
                value={localSettings["rag.crag_confidence_threshold"] as number ?? 0.5}
                onChange={(e) => handleSettingChange("rag.crag_confidence_threshold", parseFloat(e.target.value))}
              />
              <p className="text-xs text-muted-foreground">Trigger CRAG when confidence is below this (0.1-0.9)</p>
            </div>
          )}

          {/* Graph-O1 */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium flex items-center gap-2">
                <Workflow className="h-4 w-4 text-cyan-500" />
                Graph-O1 Reasoning
              </p>
              <p className="text-sm text-muted-foreground">
                3-5x faster GraphRAG with beam search — explores multiple reasoning paths efficiently
              </p>
            </div>
            <Switch
              checked={localSettings["rag.graph_o1_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("rag.graph_o1_enabled", checked)}
            />
          </div>
          {(localSettings["rag.graph_o1_enabled"] as boolean) && (
            <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Beam Width</label>
                  <Input
                    type="number"
                    min="3"
                    max="10"
                    value={localSettings["rag.graph_o1_beam_width"] as number ?? 5}
                    onChange={(e) => handleSettingChange("rag.graph_o1_beam_width", parseInt(e.target.value) || 5)}
                  />
                  <p className="text-xs text-muted-foreground">Parallel reasoning paths (3-10)</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Confidence Threshold</label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0.5"
                    max="0.9"
                    value={localSettings["rag.graph_o1_confidence_threshold"] as number ?? 0.7}
                    onChange={(e) => handleSettingChange("rag.graph_o1_confidence_threshold", parseFloat(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">Path pruning threshold (0.5-0.9)</p>
                </div>
              </div>
            </div>
          )}

          {/* LazyGraphRAG */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium flex items-center gap-2">
                <DollarSign className="h-4 w-4 text-green-500" />
                LazyGraphRAG
              </p>
              <p className="text-sm text-muted-foreground">
                99% cost reduction vs standard GraphRAG — builds graph lazily at query time
              </p>
            </div>
            <Switch
              checked={localSettings["rag.lazy_graphrag_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.lazy_graphrag_enabled", checked)}
            />
          </div>
          {(localSettings["rag.lazy_graphrag_enabled"] as boolean ?? true) && (
            <div className="ml-6 space-y-2 p-3 bg-muted/30 rounded-lg">
              <label className="text-sm font-medium">Max Communities</label>
              <Input
                type="number"
                min="1"
                max="10"
                value={localSettings["rag.lazy_graphrag_max_communities"] as number ?? 5}
                onChange={(e) => handleSettingChange("rag.lazy_graphrag_max_communities", parseInt(e.target.value) || 5)}
              />
              <p className="text-xs text-muted-foreground">Communities to summarize per query (1-10)</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Storage & Indexing Optimization */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Storage & Indexing Optimization
          </CardTitle>
          <CardDescription>
            Vectorstore optimization and prompt tuning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Binary Quantization */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Binary Quantization Pre-filter</p>
              <p className="text-sm text-muted-foreground">
                10-100x faster initial search on 1M+ vectors — uses 1-bit representations for candidate selection
              </p>
            </div>
            <Switch
              checked={localSettings["vectorstore.binary_quantization_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("vectorstore.binary_quantization_enabled", checked)}
            />
          </div>

          {/* Late Chunking */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Late Chunking</p>
              <p className="text-sm text-muted-foreground">
                Embed full document first, then chunk — preserves cross-chunk context (+15% recall)
              </p>
            </div>
            <Switch
              checked={localSettings["vectorstore.late_chunking_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("vectorstore.late_chunking_enabled", checked)}
            />
          </div>

          {/* DSPy Prompt Optimization */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-violet-500" />
                DSPy Prompt Optimization
              </p>
              <p className="text-sm text-muted-foreground">
                Automated prompt tuning via DSPy compilation — discovers optimal instructions from user feedback
              </p>
            </div>
            <Switch
              checked={localSettings["rag.dspy_optimization_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("rag.dspy_optimization_enabled", checked)}
            />
          </div>
          {(localSettings["rag.dspy_optimization_enabled"] as boolean) && (
            <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Default Optimizer</label>
                  <Select
                    value={localSettings["rag.dspy_default_optimizer"] as string ?? "bootstrap_few_shot"}
                    onValueChange={(value) => handleSettingChange("rag.dspy_default_optimizer", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="bootstrap_few_shot">BootstrapFewShot (20+ examples)</SelectItem>
                      <SelectItem value="miprov2">MIPROv2 (100+ examples)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Optimization algorithm</p>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Min Training Examples</label>
                  <Input
                    type="number"
                    min="5"
                    max="500"
                    value={localSettings["rag.dspy_min_examples"] as number ?? 20}
                    onChange={(e) => handleSettingChange("rag.dspy_min_examples", parseInt(e.target.value) || 20)}
                  />
                  <p className="text-xs text-muted-foreground">Required before optimization can run (5-500)</p>
                </div>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border bg-background">
                <div>
                  <p className="font-medium text-sm">DSPy Inference Mode</p>
                  <p className="text-xs text-muted-foreground">
                    Use DSPy modules at inference time (default: compilation-only, exported as text prompts)
                  </p>
                </div>
                <Switch
                  checked={localSettings["rag.dspy_inference_enabled"] as boolean ?? false}
                  onCheckedChange={(checked) => handleSettingChange("rag.dspy_inference_enabled", checked)}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Conversation Memory */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Conversation Memory
          </CardTitle>
          <CardDescription>
            Smart conversation handling — adapts memory, query rewriting, and context management by model size
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Dynamic Memory Window</p>
              <p className="text-sm text-muted-foreground">
                Adjust conversation history length based on model size (tiny: 3 turns, small: 6, medium: 10, large: 15)
              </p>
            </div>
            <Switch
              checked={localSettings["conversation.dynamic_memory_window"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("conversation.dynamic_memory_window", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Query Rewriting</p>
              <p className="text-sm text-muted-foreground">
                Resolve pronouns and references in follow-up questions (heuristic for small models, LLM for 14B+)
              </p>
            </div>
            <Switch
              checked={localSettings["conversation.query_rewriting_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("conversation.query_rewriting_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">DB Rehydration</p>
              <p className="text-sm text-muted-foreground">
                Restore conversation history from database after backend restart
              </p>
            </div>
            <Switch
              checked={localSettings["conversation.db_rehydration_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("conversation.db_rehydration_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Token Budget Enforcement</p>
              <p className="text-sm text-muted-foreground">
                Enforce per-component token budgets to prevent context window overflow
              </p>
            </div>
            <Switch
              checked={localSettings["conversation.token_budget_enforcement"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("conversation.token_budget_enforcement", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Stuff-Then-Refine</p>
              <p className="text-sm text-muted-foreground">
                Process chunks iteratively when they exceed small model context windows
              </p>
            </div>
            <Switch
              checked={localSettings["conversation.stuff_then_refine_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("conversation.stuff_then_refine_enabled", checked)}
            />
          </div>

          <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-4 mt-2">
            <div className="flex items-start gap-3">
              <Brain className="h-5 w-5 text-blue-500 mt-0.5" />
              <div>
                <p className="font-medium text-sm">Model-Adaptive Behavior</p>
                <p className="text-sm text-muted-foreground mt-1">
                  These features automatically adapt to your model size. Small models (1-3B) get shorter history windows and heuristic query rewriting. Medium models (9-34B) get LLM-based rewriting. Large models (34B+) get full history and advanced context management.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Query Pipeline Visualization */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Workflow className="h-5 w-5" />
            Query Pipeline Visualization
          </CardTitle>
          <CardDescription>
            See how your query flows through the 38-step RAG pipeline. Green = enabled, gray = disabled, blue = always-on.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-1">
            {[
              { step: 1, name: "Spell Correction", desc: "Fix typos before search", key: "search.spell_correction_enabled" },
              { step: 2, name: "Semantic Cache Check", desc: "Return cached answer if similar query found", key: "rag.semantic_cache_enabled" },
              { step: 3, name: "Query Classification", desc: "Determine query type and complexity", key: null },
              { step: 4, name: "Adaptive Routing", desc: "Route to optimal retrieval strategy", key: "rag.adaptive_routing_enabled" },
              { step: 5, name: "Tree of Thoughts", desc: "Multi-path reasoning for complex queries", key: "rag.tree_of_thoughts_enabled" },
              { step: 6, name: "Aggregation Detection", desc: "Detect comparison/aggregation patterns", key: null },
              { step: 7, name: "Query Decomposition", desc: "Break into sub-queries if complex", key: "rag.query_decomposition_enabled" },
              { step: 8, name: "Model Selection", desc: "Choose optimal LLM for this query", key: null },
              { step: 9, name: "Dynamic Chunk Capping", desc: "Adapt context window to query needs", key: null },
              { step: 10, name: "Folder Scoping", desc: "Filter to relevant document folders", key: null },
              { step: 11, name: "RAG-Fusion", desc: "Multi-query with Reciprocal Rank Fusion", key: "rag.rag_fusion_enabled" },
              { step: 12, name: "Step-Back Prompting", desc: "Abstract reasoning for analytical queries", key: "rag.stepback_prompting_enabled" },
              { step: 13, name: "HyDE Expansion", desc: "Generate hypothetical documents", key: "rag.hyde_enabled" },
              { step: 14, name: "Query Expansion", desc: "Expand with synonyms and related terms", key: "rag.query_expansion_enabled" },
              { step: 15, name: "KG Query Expansion", desc: "Expand via knowledge graph entities", key: "rag.knowledge_graph_enabled" },
              { step: 16, name: "Binary Quantization Pre-filter", desc: "Fast 1-bit candidate selection", key: "vectorstore.binary_quantization_enabled" },
              { step: 17, name: "Hybrid/Multi-Source Retrieval", desc: "Retrieve from vector + keyword + graph", key: null },
              { step: 18, name: "Tiered Reranking", desc: "ColBERT → Cross-Encoder → LLM pipeline", key: "rerank.tiered_enabled" },
              { step: 19, name: "KG Augmentation", desc: "Enrich results with knowledge graph context", key: "rag.knowledge_graph_enabled" },
              { step: 20, name: "LazyGraphRAG", desc: "Cost-efficient graph-based retrieval", key: "rag.lazy_graphrag_enabled" },
              { step: 21, name: "MMR Diversity", desc: "Maximize Marginal Relevance for diverse results", key: null },
              { step: 22, name: "Result Verification", desc: "Verify retrieved document relevance", key: "rag.verification_enabled" },
              { step: 23, name: "Context Formatting", desc: "Format chunks with metadata for LLM", key: null },
              { step: 24, name: "Graph-O1 Reasoning", desc: "Beam search over knowledge graph", key: "rag.graph_o1_enabled" },
              { step: 25, name: "Sufficiency Check", desc: "Skip extra retrieval if context sufficient", key: "rag.sufficiency_checker_enabled" },
              { step: 26, name: "Context Compression", desc: "Compress context to reduce tokens", key: "rag.context_compression_enabled" },
              { step: 27, name: "AttentionRAG", desc: "Attention-based context pruning", key: "rag.attention_rag_enabled" },
              { step: 28, name: "Prompt Construction", desc: "Build final prompt with system + context", key: null },
              { step: 29, name: "Generative Cache Check", desc: "Check for cached LLM response", key: "rag.semantic_cache_enabled" },
              { step: 30, name: "LLM Invocation", desc: "Generate answer via language model", key: null },
              { step: 31, name: "Answer Refinement", desc: "Self-Refine/CRITIC quality improvement", key: "rag.answer_refiner_enabled" },
              { step: 32, name: "Suggested Questions", desc: "Generate follow-up question suggestions", key: "rag.suggested_questions_enabled" },
              { step: 33, name: "Usage Tracking", desc: "Log token usage and costs", key: null },
              { step: 34, name: "Memory Update", desc: "Store conversation in agent memory", key: "memory.enabled" },
              { step: 35, name: "CRAG Recovery", desc: "Corrective re-retrieval if low confidence", key: "rag.crag_enabled" },
              { step: 36, name: "Self-RAG Verification", desc: "Self-reflective answer validation", key: "rag.self_rag_enabled" },
              { step: 37, name: "Semantic Cache Store", desc: "Cache answer for future similar queries", key: "rag.semantic_cache_enabled" },
              { step: 38, name: "Response Return", desc: "Stream or return final response", key: null },
            ].map((item) => {
              const isAlwaysOn = item.key === null;
              const isEnabled = isAlwaysOn || (localSettings[item.key!] as boolean ?? false);
              const dotColor = isAlwaysOn
                ? "bg-blue-500"
                : isEnabled
                  ? "bg-green-500"
                  : "bg-gray-300 dark:bg-gray-600";

              return (
                <div
                  key={item.step}
                  className="flex items-center gap-3 py-1.5 px-2 rounded hover:bg-muted/50 transition-colors"
                >
                  <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${dotColor}`} />
                  <span className="text-xs font-mono text-muted-foreground w-6 flex-shrink-0">
                    {String(item.step).padStart(2, "0")}
                  </span>
                  <span className="text-sm font-medium flex-shrink-0">{item.name}</span>
                  <span className="text-xs text-muted-foreground truncate">{item.desc}</span>
                </div>
              );
            })}
          </div>
          <div className="flex items-center gap-6 mt-4 pt-3 border-t text-xs text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
              <span>Enabled</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-gray-300 dark:bg-gray-600" />
              <span>Disabled</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
              <span>Always Active</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
