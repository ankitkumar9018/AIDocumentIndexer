"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  TestTube,
  Sparkles,
  Loader2,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Zap,
  Shield,
  Brain,
} from "lucide-react";
import { api } from "@/lib/api";

interface Experiment {
  name: string;
  enabled: boolean;
  description: string;
  category: string;
  status: "stable" | "beta" | "experimental";
}

const DEFAULT_EXPERIMENTS: Experiment[] = [
  {
    name: "attention_rag",
    enabled: false,
    description: "6.3x context compression using attention scoring (AttentionRAG). Reduces context size while maintaining relevance.",
    category: "Compression",
    status: "beta",
  },
  {
    name: "graph_o1",
    enabled: false,
    description: "Beam search reasoning over the knowledge graph (Graph-O1). 3-5x faster graph reasoning with 95%+ accuracy.",
    category: "Retrieval",
    status: "experimental",
  },
  {
    name: "tiered_reranking",
    enabled: true,
    description: "4-stage reranking pipeline: BM25 → CrossEncoder → ColBERT → LLM. Doubles precision for complex queries.",
    category: "Retrieval",
    status: "stable",
  },
  {
    name: "adaptive_routing",
    enabled: true,
    description: "Query-dependent strategy routing (DIRECT/HYBRID/TWO_STAGE/AGENTIC/GRAPH). Selects optimal retrieval based on query complexity.",
    category: "Retrieval",
    status: "stable",
  },
  {
    name: "late_chunking",
    enabled: false,
    description: "Context-preserving chunking with +15-25% retrieval accuracy. Uses full document context for chunk embeddings.",
    category: "Processing",
    status: "beta",
  },
  {
    name: "tree_of_thoughts",
    enabled: false,
    description: "Multi-path reasoning with beam search exploration. Generates and evaluates multiple reasoning chains.",
    category: "Reasoning",
    status: "experimental",
  },
  {
    name: "generative_cache",
    enabled: true,
    description: "Semantic caching with 9x speedup over GPTCache. 68.8% cache hit rate for repeated queries.",
    category: "Performance",
    status: "stable",
  },
  {
    name: "llmlingua_compression",
    enabled: false,
    description: "LLMLingua-2 context compression (3-6x). Reduces token usage while preserving answer quality.",
    category: "Compression",
    status: "beta",
  },
];

const statusColors: Record<string, string> = {
  stable: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 border-green-200 dark:border-green-800",
  beta: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 border-yellow-200 dark:border-yellow-800",
  experimental: "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800",
};

const categoryIcons: Record<string, React.ReactNode> = {
  Compression: <Zap className="h-4 w-4" />,
  Retrieval: <Sparkles className="h-4 w-4" />,
  Processing: <Brain className="h-4 w-4" />,
  Reasoning: <Brain className="h-4 w-4" />,
  Performance: <Zap className="h-4 w-4" />,
  Security: <Shield className="h-4 w-4" />,
};

export function ExperimentsTab() {
  const [experiments, setExperiments] = useState<Experiment[]>(DEFAULT_EXPERIMENTS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toggling, setToggling] = useState<string | null>(null);

  const fetchExperiments = async () => {
    try {
      setLoading(true);
      setError(null);
      const { data } = await api.get<{ experiments?: Experiment[] }>("/experiments/");
      if (data.experiments) {
        setExperiments(data.experiments);
      }
    } catch {
      // Use defaults if API not available
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const toggleExperiment = async (name: string, enabled: boolean) => {
    setToggling(name);
    try {
      await api.put(`/experiments/${name}`, { enabled });
      setExperiments((prev) =>
        prev.map((e) => (e.name === name ? { ...e, enabled } : e))
      );
    } catch {
      setError(`Failed to toggle ${name}`);
    } finally {
      setToggling(null);
    }
  };

  // Group by category
  const categories = experiments.reduce<Record<string, Experiment[]>>((acc, exp) => {
    if (!acc[exp.category]) acc[exp.category] = [];
    acc[exp.category].push(exp);
    return acc;
  }, {});

  const enabledCount = experiments.filter((e) => e.enabled).length;

  return (
    <TabsContent value="experiments" className="space-y-6">
      {/* Summary */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <TestTube className="h-5 w-5" />
                Experimental Features
              </CardTitle>
              <CardDescription>
                Enable or disable experimental RAG features. Stable features are production-ready.
                Beta and experimental features may impact performance.
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={fetchExperiments}>
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500 dark:text-green-400" />
              <span className="text-sm">{enabledCount} enabled</span>
            </div>
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">{experiments.length - enabledCount} disabled</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Feature Categories */}
      {Object.entries(categories).map(([category, features]) => (
        <Card key={category}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              {categoryIcons[category] || <Sparkles className="h-4 w-4" />}
              {category}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {features.map((feature) => (
              <div
                key={feature.name}
                className="flex items-start justify-between p-3 rounded-lg border hover:border-primary/50 transition-colors"
              >
                <div className="flex-1 mr-4">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm">{feature.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</span>
                    <Badge variant="outline" className={statusColors[feature.status]}>
                      {feature.status}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
                <div className="flex items-center gap-2">
                  {toggling === feature.name ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Switch
                      checked={feature.enabled}
                      onCheckedChange={(checked) => toggleExperiment(feature.name, checked)}
                    />
                  )}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      ))}
    </TabsContent>
  );
}
