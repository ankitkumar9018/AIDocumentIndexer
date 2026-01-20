"use client";

import { useState, useEffect } from "react";
import { Shield, Sparkles, Zap, RefreshCw, AlertCircle, CheckCircle, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { toast } from "sonner";
import { api } from "@/lib/api";

interface EmbeddingProvider {
  name: string;
  model: string;
  dimension: number;
  chunk_count: number;
  storage_bytes: number;
  is_primary: boolean;
}

interface EmbeddingStats {
  total_documents: number;
  total_chunks: number;
  chunks_with_embeddings: number;
  coverage_percentage: number;
  storage_bytes: number;
  provider_count: number;
  providers: EmbeddingProvider[];
}

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const getProviderIcon = (name: string) => {
  if (name === "ollama") return Shield;
  if (name === "openai") return Sparkles;
  return Database;
};

const getProviderColor = (name: string) => {
  if (name === "ollama") return "text-green-600";
  if (name === "openai") return "text-blue-600";
  return "text-gray-600";
};

export function EmbeddingDashboard() {
  const [stats, setStats] = useState<EmbeddingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStats = async () => {
    try {
      setRefreshing(true);
      const data = await api.getEmbeddingStats();
      setStats(data);
    } catch (error) {
      console.error("Failed to fetch embedding stats:", error);
      toast.error("Failed to load embedding statistics");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Embedding System Status</CardTitle>
          <CardDescription>Loading statistics...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!stats) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Embedding System Status</CardTitle>
          <CardDescription>Failed to load statistics</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              Could not load embedding statistics. Please try refreshing the page.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const coverageColor = stats.coverage_percentage >= 90 ? "text-green-600" :
                        stats.coverage_percentage >= 50 ? "text-yellow-600" :
                        "text-red-600";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Embedding System Status
          </CardTitle>
          <CardDescription>
            Overview of embedding coverage and providers
          </CardDescription>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchStats}
          disabled={refreshing}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Overall Coverage Cards */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Documents
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_documents.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {stats.total_chunks.toLocaleString()} chunks
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Embedding Coverage
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${coverageColor}`}>
                {stats.coverage_percentage}%
              </div>
              <Progress value={stats.coverage_percentage} className="mt-2" />
              <p className="text-xs text-muted-foreground mt-2">
                {stats.chunks_with_embeddings.toLocaleString()} / {stats.total_chunks.toLocaleString()} chunks
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Storage Used
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {formatBytes(stats.storage_bytes)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {stats.provider_count} provider{stats.provider_count !== 1 ? 's' : ''}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Status Alert */}
        {stats.coverage_percentage < 100 && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Incomplete Coverage</AlertTitle>
            <AlertDescription>
              {stats.total_chunks - stats.chunks_with_embeddings} chunks are missing embeddings.
              This may affect search quality. Consider running the embedding backfill script.
            </AlertDescription>
          </Alert>
        )}

        {stats.coverage_percentage === 100 && (
          <Alert className="border-green-200 bg-green-50 dark:bg-green-950/30">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertTitle className="text-green-900 dark:text-green-100">
              Full Coverage
            </AlertTitle>
            <AlertDescription className="text-green-800 dark:text-green-200">
              All chunks have embeddings. Your semantic search is fully operational!
            </AlertDescription>
          </Alert>
        )}

        {/* Provider Breakdown */}
        {stats.providers && stats.providers.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Provider Breakdown</h3>

            <div className="space-y-3">
              {stats.providers.map((provider) => {
                const Icon = getProviderIcon(provider.name);
                const colorClass = getProviderColor(provider.name);

                return (
                  <div
                    key={`${provider.name}-${provider.model}-${provider.dimension}`}
                    className="flex items-center justify-between p-4 border rounded-lg bg-card hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <Icon className={`h-5 w-5 ${colorClass}`} />

                      <div>
                        <div className="font-medium flex items-center gap-2">
                          {provider.name}
                          <Badge variant="outline" className="text-xs">
                            {provider.dimension}D
                          </Badge>
                          {provider.is_primary && (
                            <Badge variant="default" className="text-xs bg-blue-600">
                              PRIMARY
                            </Badge>
                          )}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {provider.model}
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="font-medium">
                        {provider.chunk_count.toLocaleString()} chunks
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {formatBytes(provider.storage_bytes)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Help Text */}
        <div className="pt-4 border-t">
          <h4 className="text-sm font-medium mb-2">About Embeddings</h4>
          <p className="text-sm text-muted-foreground">
            Embeddings are vector representations of text that enable semantic search.
            The system uses embeddings to find contextually relevant content, not just
            keyword matches. Higher coverage means better search quality.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
