"use client";

import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { GitBranch, Brain } from "lucide-react";

interface IngestionTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function IngestionTab({ localSettings, handleSettingChange }: IngestionTabProps) {
  return (
    <TabsContent value="ingestion" className="space-y-6">
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
