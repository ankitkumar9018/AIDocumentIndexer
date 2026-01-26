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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart3,
  Loader2,
  RefreshCw,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  CheckCircle,
  Activity,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

interface EvalMetrics {
  count: number;
  avg_overall: number;
  avg_faithfulness: number;
  avg_relevance: number;
  avg_context_relevance?: number;
  avg_answer_relevance?: number;
  period_hours: number;
}

interface MetricTrend {
  metric: string;
  periods: number;
  period_hours: number;
  trend: number[];
  average: number;
  direction: "improving" | "declining" | "stable";
}

export function EvaluationTab() {
  const [metrics, setMetrics] = useState<EvalMetrics | null>(null);
  const [trend, setTrend] = useState<MetricTrend | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [trendMetric, setTrendMetric] = useState("overall_score");
  const [trendPeriod, setTrendPeriod] = useState("7");

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      setError(null);

      const [metricsRes, trendRes] = await Promise.allSettled([
        fetch(`${API_BASE}/evaluation/metrics/summary?period_hours=24`, {
          credentials: "include",
        }),
        fetch(
          `${API_BASE}/evaluation/metrics/trend?metric=${trendMetric}&periods=${trendPeriod}&period_hours=24`,
          { credentials: "include" }
        ),
      ]);

      if (metricsRes.status === "fulfilled" && metricsRes.value.ok) {
        setMetrics(await metricsRes.value.json());
      }
      if (trendRes.status === "fulfilled" && trendRes.value.ok) {
        setTrend(await trendRes.value.json());
      }
    } catch {
      setError("Failed to fetch evaluation metrics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, [trendMetric, trendPeriod]);

  const scoreColor = (score: number) => {
    if (score >= 0.8) return "text-green-600";
    if (score >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const scoreBadge = (score: number) => {
    if (score >= 0.8) return { label: "Good", variant: "default" as const, className: "bg-green-100 text-green-700" };
    if (score >= 0.6) return { label: "Fair", variant: "outline" as const, className: "bg-yellow-100 text-yellow-700" };
    return { label: "Needs Work", variant: "destructive" as const, className: "bg-red-100 text-red-700" };
  };

  const directionIcon = (dir: string) => {
    if (dir === "improving") return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (dir === "declining") return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <Minus className="h-4 w-4 text-gray-400" />;
  };

  const metricCards = metrics
    ? [
        { label: "Overall Score", value: metrics.avg_overall, icon: <Target className="h-4 w-4" /> },
        { label: "Faithfulness", value: metrics.avg_faithfulness, icon: <CheckCircle className="h-4 w-4" /> },
        { label: "Relevance", value: metrics.avg_relevance, icon: <Activity className="h-4 w-4" /> },
        ...(metrics.avg_context_relevance
          ? [{ label: "Context Relevance", value: metrics.avg_context_relevance, icon: <BarChart3 className="h-4 w-4" /> }]
          : []),
        ...(metrics.avg_answer_relevance
          ? [{ label: "Answer Relevance", value: metrics.avg_answer_relevance, icon: <BarChart3 className="h-4 w-4" /> }]
          : []),
      ]
    : [];

  return (
    <TabsContent value="evaluation" className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                RAG Evaluation Dashboard
              </CardTitle>
              <CardDescription>
                Monitor retrieval and generation quality metrics (RAGAS framework).
                Metrics are computed from sampled queries.
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={fetchMetrics}>
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        {metrics && (
          <CardContent>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span>{metrics.count} queries evaluated in the last {metrics.period_hours}h</span>
            </div>
          </CardContent>
        )}
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Metric Cards */}
      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : metrics ? (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            {metricCards.map((card) => {
              const badge = scoreBadge(card.value);
              return (
                <Card key={card.label}>
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        {card.icon}
                        {card.label}
                      </div>
                      <Badge variant={badge.variant} className={badge.className}>
                        {badge.label}
                      </Badge>
                    </div>
                    <p className={`text-3xl font-bold ${scoreColor(card.value)}`}>
                      {(card.value * 100).toFixed(1)}%
                    </p>
                    {/* Simple bar visualization */}
                    <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          card.value >= 0.8
                            ? "bg-green-500"
                            : card.value >= 0.6
                            ? "bg-yellow-500"
                            : "bg-red-500"
                        }`}
                        style={{ width: `${card.value * 100}%` }}
                      />
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Trend Section */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Metric Trend</CardTitle>
                <div className="flex gap-2">
                  <Select value={trendMetric} onValueChange={setTrendMetric}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="overall_score">Overall Score</SelectItem>
                      <SelectItem value="faithfulness">Faithfulness</SelectItem>
                      <SelectItem value="relevance">Relevance</SelectItem>
                      <SelectItem value="context_relevance">Context Relevance</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={trendPeriod} onValueChange={setTrendPeriod}>
                    <SelectTrigger className="w-[120px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="7">7 days</SelectItem>
                      <SelectItem value="14">14 days</SelectItem>
                      <SelectItem value="30">30 days</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {trend ? (
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    {directionIcon(trend.direction)}
                    <span className="text-sm capitalize">{trend.direction}</span>
                    <span className="text-sm text-muted-foreground">
                      Average: {(trend.average * 100).toFixed(1)}%
                    </span>
                  </div>
                  {/* Simple bar chart visualization */}
                  <div className="flex items-end gap-1 h-32">
                    {trend.trend.map((value, i) => (
                      <div
                        key={i}
                        className="flex-1 rounded-t transition-all hover:opacity-80"
                        style={{
                          height: `${Math.max(value * 100, 5)}%`,
                          backgroundColor:
                            value >= 0.8 ? "#22c55e" : value >= 0.6 ? "#eab308" : "#ef4444",
                        }}
                        title={`Day ${i + 1}: ${(value * 100).toFixed(1)}%`}
                      />
                    ))}
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{trend.periods} days ago</span>
                    <span>Today</span>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No trend data available. Enable evaluation sampling in RAG settings.
                </p>
              )}
            </CardContent>
          </Card>
        </>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <BarChart3 className="h-12 w-12 mb-4" />
            <p className="text-lg font-medium mb-2">No evaluation data</p>
            <p className="text-sm">
              Enable RAG evaluation sampling in the RAG Features tab to start collecting metrics.
            </p>
          </CardContent>
        </Card>
      )}
    </TabsContent>
  );
}
