"use client";

import { useState, useEffect } from "react";
import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { GitBranch, Brain, Zap, Cpu, Loader2, Sparkles, HardDrive, ExternalLink, RefreshCw } from "lucide-react";
import { api } from "@/lib/api/client";

interface IngestionTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

type StorageStats = {
  local_count: number;
  external_count: number;
  local_size_bytes: number;
  external_size_bytes: number;
  total_count: number;
  total_size_bytes: number;
  by_source_type: Record<string, { count: number; size: number; external_count: number }>;
};

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

export function IngestionTab({ localSettings, handleSettingChange }: IngestionTabProps) {
  const [detecting, setDetecting] = useState(false);
  const [storageStats, setStorageStats] = useState<StorageStats | null>(null);
  const [loadingStats, setLoadingStats] = useState(false);

  const loadStorageStats = async () => {
    setLoadingStats(true);
    try {
      const stats = await api.getStorageStats();
      setStorageStats(stats);
    } catch (err) {
      console.error("Failed to load storage stats:", err);
    } finally {
      setLoadingStats(false);
    }
  };

  useEffect(() => {
    loadStorageStats();
  }, []);

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

      {/* Connector Storage Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Connector Storage Settings
          </CardTitle>
          <CardDescription>
            Control how external documents are stored when synced via connectors
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Default Storage Mode */}
          <div className="space-y-2">
            <p className="font-medium">Default Storage Mode</p>
            <Select
              value={String(localSettings["connector.storage_mode"] ?? "download")}
              onValueChange={(v) => handleSettingChange("connector.storage_mode", v)}
            >
              <SelectTrigger className="h-9">
                <SelectValue placeholder="Select mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="download">Download & Store (recommended)</SelectItem>
                <SelectItem value="process_only">Process Only (save storage, keep external link)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {String(localSettings["connector.storage_mode"]) === "process_only"
                ? "Files are processed and indexed but not stored locally. Preview uses external source link."
                : "Files are downloaded and stored locally for preview and reprocessing."}
            </p>
          </div>

          {/* Store Source Metadata */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Store Source Metadata</p>
              <p className="text-sm text-muted-foreground">
                Record origin info (connector, external ID, URL) for each synced document
              </p>
            </div>
            <Switch
              checked={Boolean(localSettings["connector.store_source_metadata"] ?? true)}
              onCheckedChange={(v) => handleSettingChange("connector.store_source_metadata", v)}
            />
          </div>

          {/* Storage Breakdown */}
          <div className="space-y-3 pt-2 border-t">
            <div className="flex items-center justify-between">
              <p className="font-medium text-sm">Storage Breakdown</p>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs gap-1"
                onClick={loadStorageStats}
                disabled={loadingStats}
              >
                <RefreshCw className={`h-3 w-3 ${loadingStats ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>

            {storageStats ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Local</p>
                    <p className="text-lg font-bold">{storageStats.local_count}</p>
                    <p className="text-xs text-muted-foreground">{formatBytes(storageStats.local_size_bytes)}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      External <ExternalLink className="h-3 w-3" />
                    </p>
                    <p className="text-lg font-bold">{storageStats.external_count}</p>
                    <p className="text-xs text-muted-foreground">{formatBytes(storageStats.external_size_bytes)}</p>
                  </div>
                </div>

                {storageStats.total_count > 0 && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Local ({Math.round((storageStats.local_count / storageStats.total_count) * 100)}%)</span>
                      <span>External ({Math.round((storageStats.external_count / storageStats.total_count) * 100)}%)</span>
                    </div>
                    <Progress
                      value={(storageStats.local_count / storageStats.total_count) * 100}
                      className="h-2"
                    />
                  </div>
                )}

                {/* By Source Type */}
                {Object.keys(storageStats.by_source_type).length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-medium text-muted-foreground">By Source</p>
                    {Object.entries(storageStats.by_source_type).map(([type, data]) => (
                      <div key={type} className="flex items-center justify-between text-sm p-1.5 rounded bg-muted/30">
                        <span className="truncate">{type.replace(/_/g, ' ')}</span>
                        <div className="flex items-center gap-2 shrink-0">
                          <span className="text-muted-foreground">{data.count} docs ({formatBytes(data.size)})</span>
                          {data.external_count > 0 && (
                            <Badge variant="outline" className="text-[10px] px-1.5">
                              <ExternalLink className="h-2.5 w-2.5 mr-0.5" />
                              {data.external_count}
                            </Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">
                {loadingStats ? "Loading..." : "No storage data available"}
              </p>
            )}
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
