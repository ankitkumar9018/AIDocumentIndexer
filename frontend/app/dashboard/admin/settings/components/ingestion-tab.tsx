"use client";

import { useState } from "react";
import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { GitBranch, Brain, Zap, Cpu, Loader2, Sparkles } from "lucide-react";
import { api } from "@/lib/api/client";

interface IngestionTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function IngestionTab({ localSettings, handleSettingChange }: IngestionTabProps) {
  const [detecting, setDetecting] = useState(false);

  const detectKgConcurrency = async () => {
    setDetecting(true);
    try {
      const { data } = await api.get<{ recommended: Record<string, number> }>("/diagnostics/hardware/recommended-settings");
      const recommended = data.recommended["kg.extraction_concurrency"];
      if (recommended !== undefined) {
        handleSettingChange("kg.extraction_concurrency", recommended);
      }
    } catch {
      // Silently fail - the value just won't change
    } finally {
      setDetecting(false);
    }
  };

  return (
    <TabsContent value="ingestion" className="space-y-6">
      {/* Document Enhancement */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Document Enhancement
          </CardTitle>
          <CardDescription>
            Automatically enhance documents with LLM-extracted summaries, keywords, and hypothetical questions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Auto-Enhance on Upload</p>
              <p className="text-sm text-muted-foreground">
                Automatically enhance documents after upload completes
              </p>
            </div>
            <Switch
              checked={localSettings["upload.auto_enhance"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("upload.auto_enhance", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Include Enhanced Metadata in RAG</p>
              <p className="text-sm text-muted-foreground">
                Include document summaries, keywords, and topics in RAG context for richer answers
              </p>
            </div>
            <Switch
              checked={localSettings["rag.include_enhanced_metadata"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.include_enhanced_metadata", checked)}
            />
          </div>

          <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg text-sm text-muted-foreground">
            Enhancement generates summaries, keywords, topics, entities, and hypothetical questions using an LLM.
            Hypothetical questions are embedded into the vector store for improved question-based retrieval.
            Documents without enhancement are still fully queryable — enhancement only makes answers richer.
          </div>
        </CardContent>
      </Card>

      {/* Knowledge Graph Ingestion */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <GitBranch className="h-5 w-5" />
            Knowledge Graph Ingestion
          </CardTitle>
          <CardDescription>
            Configure entity extraction and relationship building during document ingestion
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Fast Dependency Extraction</p>
              <p className="text-sm text-muted-foreground">
                Use spaCy dependency parsing for entity extraction (94% LLM quality, 80% cost savings)
              </p>
            </div>
            <Switch
              checked={localSettings["kg.dependency_extraction_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("kg.dependency_extraction_enabled", checked)}
            />
          </div>

          <div className="space-y-3 p-3 bg-muted/30 rounded-lg">
            <div>
              <label className="text-sm font-medium">Complexity Threshold</label>
              <Input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={localSettings["kg.dependency_complexity_threshold"] as number ?? 0.7}
                onChange={(e) => handleSettingChange("kg.dependency_complexity_threshold", parseFloat(e.target.value))}
                className="mt-1 w-32"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Text above this complexity uses full LLM extraction instead of dependency parsing (0.0-1.0)
              </p>
            </div>

            <div>
              <label className="text-sm font-medium">spaCy Model</label>
              <Select
                value={localSettings["kg.spacy_model"] as string ?? "en_core_web_sm"}
                onValueChange={(value) => handleSettingChange("kg.spacy_model", value)}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="en_core_web_sm">en_core_web_sm (Fast, 12MB)</SelectItem>
                  <SelectItem value="en_core_web_md">en_core_web_md (Balanced, 40MB)</SelectItem>
                  <SelectItem value="en_core_web_lg">en_core_web_lg (Accurate, 560MB)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                Larger models improve entity recognition accuracy
              </p>
            </div>
          </div>

          <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg text-sm text-muted-foreground">
            When dependency extraction is enabled, simple text uses fast spaCy parsing while complex passages
            (above the threshold) are routed to the LLM for higher-quality extraction.
          </div>
        </CardContent>
      </Card>

      {/* KG Extraction Performance */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="h-5 w-5" />
                KG Extraction Performance
              </CardTitle>
              <CardDescription>
                Control concurrency and timeouts for knowledge graph extraction jobs
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={detectKgConcurrency}
              disabled={detecting}
            >
              {detecting ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Cpu className="h-4 w-4 mr-1" />
              )}
              Auto-detect
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2 p-3 bg-muted/30 rounded-lg">
            <label className="text-sm font-medium">Extraction Concurrency</label>
            <Input
              type="number"
              min="1"
              max="16"
              step="1"
              value={localSettings["kg.extraction_concurrency"] as number ?? 4}
              onChange={(e) => handleSettingChange("kg.extraction_concurrency", parseInt(e.target.value) || 4)}
              className="w-32"
            />
            <p className="text-xs text-muted-foreground">
              Number of documents processed in parallel during KG extraction (1-16)
            </p>
          </div>

          <div className="space-y-2 p-3 bg-muted/30 rounded-lg">
            <label className="text-sm font-medium">Ray Task Timeout (seconds)</label>
            <Input
              type="number"
              min="300"
              max="7200"
              step="60"
              value={localSettings["kg.ray_task_timeout"] as number ?? 1800}
              onChange={(e) => handleSettingChange("kg.ray_task_timeout", parseInt(e.target.value) || 1800)}
              className="w-32"
            />
            <p className="text-xs text-muted-foreground">
              Maximum time to wait for each document during Ray-distributed extraction (300-7200s)
            </p>
          </div>

          <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg text-sm text-muted-foreground">
            Higher concurrency processes documents faster but uses more memory. If using a local LLM (Ollama),
            increase the timeout to avoid premature batch failures.
          </div>
        </CardContent>
      </Card>

      {/* Knowledge Graph Settings from RAG */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Graph Query Settings
          </CardTitle>
          <CardDescription>
            Control how the knowledge graph is used during queries (also configurable in RAG tab)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Knowledge Graph Enabled</p>
              <p className="text-sm text-muted-foreground">
                Use entity relationships to enhance retrieval results
              </p>
            </div>
            <Switch
              checked={localSettings["rag.knowledge_graph_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.knowledge_graph_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">LazyGraphRAG</p>
              <p className="text-sm text-muted-foreground">
                Cost-efficient graph retrieval with community detection (99% cost reduction)
              </p>
            </div>
            <Switch
              checked={localSettings["rag.lazy_graphrag_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("rag.lazy_graphrag_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Graph-O1 Reasoning</p>
              <p className="text-sm text-muted-foreground">
                Beam search reasoning over knowledge graph for complex queries
              </p>
            </div>
            <Switch
              checked={localSettings["rag.graph_o1_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("rag.graph_o1_enabled", checked)}
            />
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
