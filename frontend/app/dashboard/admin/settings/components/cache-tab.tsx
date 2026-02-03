"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
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

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

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

interface CacheMemoryStats {
  name: string;
  entries: number;
  max_entries: number;
  utilization_percent: number;
  estimated_memory_mb: number;
  has_lru: boolean;
}

interface AllCachesMemoryStats {
  total_memory_mb: number;
  process_memory_mb: number;
  caches: CacheMemoryStats[];
  warnings: string[];
}

export function CacheTab() {
  const { data: session } = useSession();
  const [stats, setStats] = useState<CacheStats | null>(null);
  const [memoryStats, setMemoryStats] = useState<AllCachesMemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const accessToken = (session as any)?.accessToken as string | undefined;

  const fetchStats = useCallback(async () => {
    if (!accessToken) {
      // Use placeholder stats if not authenticated
      setStats({
        redis_connected: false,
        semantic_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
        generative_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
        embedding_cache: { hits: 0, misses: 0, hit_rate: 0, size: 0 },
      });
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/cache/stats`, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
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
  }, [accessToken]);

  const fetchMemoryStats = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await fetch(`${API_BASE}/cache/memory/stats`, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setMemoryStats(data);
      }
    } catch (e) {
      console.error("Failed to fetch memory stats:", e);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchStats();
    fetchMemoryStats();
    const interval = setInterval(() => {
      fetchStats();
      fetchMemoryStats();
    }, 30000); // Auto-refresh every 30s
    return () => clearInterval(interval);
  }, [fetchStats, fetchMemoryStats]);

  const clearCache = async (cacheType?: string) => {
    if (!accessToken) return;
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
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ cache_type: cacheType || "all" }),
      });
      if (response.ok) {
        setSuccess(`${target} cache cleared successfully`);
        await fetchStats();
        await fetchMemoryStats();
      } else {
        setError(`Failed to clear ${target} cache`);
      }
    } catch {
      setError(`Failed to clear ${target} cache`);
    } finally {
      setClearing(null);
    }
  };

  const clearAllCaches = async () => {
    if (!accessToken) return;
    setClearing("memory-all");
    setError(null);
    setSuccess(null);
    try {
      const response = await fetch(`${API_BASE}/cache/memory/clear-all`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setSuccess(`Cleared ${data.total_entries_cleared} entries. Memory: ${data.memory_after_mb}MB`);
        await fetchStats();
        await fetchMemoryStats();
      } else {
        setError("Failed to clear all caches");
      }
    } catch {
      setError("Failed to clear all caches");
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

      {/* Memory Monitor Card */}
      {memoryStats && (
        <Card className="border-2 border-primary/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  Memory Monitor
                </CardTitle>
                <CardDescription>
                  Real-time memory usage across all caches. Clear caches if memory usage is high.
                </CardDescription>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="sm">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Clear ALL Caches
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Clear ALL Memory Caches?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will clear ALL caches: embeddings, sessions, queries, and generative cache.
                      This frees memory but queries will be slower until caches rebuild.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={clearAllCaches}>
                      {clearing === "memory-all" ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                      Clear All & Free Memory
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Memory Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-xs text-muted-foreground">Process Memory</p>
                <p className={`text-xl font-bold ${memoryStats.process_memory_mb > 4000 ? "text-red-600" : memoryStats.process_memory_mb > 2000 ? "text-yellow-600" : "text-green-600"}`}>
                  {memoryStats.process_memory_mb.toFixed(0)} MB
                </p>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-xs text-muted-foreground">Cache Memory</p>
                <p className="text-xl font-bold">{memoryStats.total_memory_mb.toFixed(0)} MB</p>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-xs text-muted-foreground">Active Caches</p>
                <p className="text-xl font-bold">{memoryStats.caches.length}</p>
              </div>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-xs text-muted-foreground">Status</p>
                <p className={`text-xl font-bold ${memoryStats.warnings.length > 0 ? "text-yellow-600" : "text-green-600"}`}>
                  {memoryStats.warnings.length > 0 ? "⚠️ Warnings" : "✓ Healthy"}
                </p>
              </div>
            </div>

            {/* Warnings */}
            {memoryStats.warnings.length > 0 && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  <ul className="list-disc list-inside">
                    {memoryStats.warnings.map((warning, i) => (
                      <li key={i}>{warning}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {/* Individual Cache Stats */}
            <div className="space-y-2">
              <p className="text-sm font-medium">Cache Utilization</p>
              {memoryStats.caches.map((cache) => (
                <div key={cache.name} className="flex items-center gap-3">
                  <span className="text-sm w-36 truncate">{cache.name}</span>
                  <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        cache.utilization_percent > 90
                          ? "bg-red-500"
                          : cache.utilization_percent > 70
                          ? "bg-yellow-500"
                          : "bg-green-500"
                      }`}
                      style={{ width: `${Math.min(cache.utilization_percent, 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-24 text-right">
                    {cache.entries.toLocaleString()} / {cache.max_entries.toLocaleString()}
                  </span>
                  <span className="text-xs w-16 text-right">
                    {cache.estimated_memory_mb.toFixed(1)} MB
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
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
