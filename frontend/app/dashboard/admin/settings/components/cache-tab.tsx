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
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  HardDrive,
  Loader2,
  RefreshCw,
  Trash2,
  AlertCircle,
  CheckCircle,
  Database,
  Zap,
  Activity,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

interface CacheTierStats {
  hits: number;
  misses: number;
  hit_rate: number;
  size: number;
  errors?: number;
}

interface CacheStats {
  semantic_cache?: CacheTierStats;
  generative_cache?: CacheTierStats;
  embedding_cache?: CacheTierStats;
  query_cache?: CacheTierStats;
  redis_connected: boolean;
  total_memory_mb?: number;
}

export function CacheTab() {
  const [stats, setStats] = useState<CacheStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/cache/stats`, {
        credentials: "include",
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      } else {
        // Use placeholder stats if endpoint not available
        setStats({
          redis_connected: false,
          semantic_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
          generative_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
          embedding_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
        });
      }
    } catch {
      setStats({
        redis_connected: false,
        semantic_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
        generative_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
        embedding_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 30000); // Auto-refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const clearCache = async (cacheType?: string) => {
    const target = cacheType || "all";
    setClearing(target);
    setError(null);
    setSuccess(null);
    try {
      const url = cacheType
        ? `${API_BASE}/cache/invalidate`
        : `${API_BASE}/cache/clear`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ cache_type: cacheType || "all" }),
      });
      if (response.ok) {
        setSuccess(`${target} cache cleared successfully`);
        await fetchStats();
      } else {
        setError(`Failed to clear ${target} cache`);
      }
    } catch {
      setError(`Failed to clear ${target} cache`);
    } finally {
      setClearing(null);
    }
  };

  const formatNumber = (n: number) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
  };

  const formatHitRate = (rate: number) => `${(rate * 100).toFixed(1)}%`;

  const hitRateColor = (rate: number) => {
    if (rate >= 0.7) return "text-green-600";
    if (rate >= 0.4) return "text-yellow-600";
    return "text-red-600";
  };

  const cacheTiers = [
    { key: "semantic_cache", name: "Semantic Cache", icon: <Zap className="h-4 w-4" />, description: "Caches similar queries for instant responses" },
    { key: "generative_cache", name: "Generative Cache", icon: <Activity className="h-4 w-4" />, description: "Caches LLM-generated responses (9x speedup)" },
    { key: "embedding_cache", name: "Embedding Cache", icon: <Database className="h-4 w-4" />, description: "Caches computed embeddings to avoid re-embedding" },
    { key: "query_cache", name: "Query Cache", icon: <HardDrive className="h-4 w-4" />, description: "Caches expanded query results" },
  ];

  return (
    <TabsContent value="cache" className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <HardDrive className="h-5 w-5" />
                Cache Management
              </CardTitle>
              <CardDescription>
                Monitor and manage caching across all tiers. Higher hit rates reduce latency and API costs.
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={fetchStats}>
                <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
                Refresh
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="sm">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Clear All
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Clear All Caches?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will clear all cache tiers (semantic, generative, embedding, query).
                      Queries will be slower until caches warm up again.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={() => clearCache()}>
                      {clearing === "all" ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                      Clear All Caches
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {stats?.redis_connected ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <AlertCircle className="h-4 w-4 text-yellow-500" />
              )}
              <span className="text-sm">
                Redis: {stats?.redis_connected ? "Connected" : "In-memory fallback"}
              </span>
            </div>
            {stats?.total_memory_mb && (
              <Badge variant="outline">
                {stats.total_memory_mb.toFixed(1)} MB total
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* Cache Tier Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        {cacheTiers.map((tier) => {
          const tierStats = stats?.[tier.key as keyof CacheStats] as CacheTierStats | undefined;
          return (
            <Card key={tier.key}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2 text-base">
                    {tier.icon}
                    {tier.name}
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => clearCache(tier.key)}
                    disabled={clearing === tier.key}
                  >
                    {clearing === tier.key ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Trash2 className="h-3 w-3" />
                    )}
                  </Button>
                </div>
                <CardDescription className="text-xs">{tier.description}</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center py-4">
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  </div>
                ) : tierStats ? (
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-xs text-muted-foreground">Hit Rate</p>
                      <p className={`text-lg font-semibold ${hitRateColor(tierStats.hit_rate)}`}>
                        {formatHitRate(tierStats.hit_rate)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Hits / Misses</p>
                      <p className="text-sm font-medium">
                        {formatNumber(tierStats.hits)} / {formatNumber(tierStats.misses)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Entries</p>
                      <p className="text-sm font-medium">{formatNumber(tierStats.size)}</p>
                    </div>
                    {tierStats.errors !== undefined && tierStats.errors > 0 && (
                      <div className="col-span-3">
                        <Badge variant="destructive" className="text-xs">
                          {tierStats.errors} errors
                        </Badge>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No data available</p>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </TabsContent>
  );
}
