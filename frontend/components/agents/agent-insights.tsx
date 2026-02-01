"use client";

/**
 * AIDocumentIndexer - Agent Usage Insights
 * =========================================
 *
 * Per-agent analytics component displaying usage metrics computed
 * from agent trajectories and execution data.
 *
 * Features:
 * - Summary metric cards (total queries, avg response time, satisfaction %, active users)
 * - Time period selector (7d, 30d, all time)
 * - Recent activity list showing last 10 queries
 * - Loading skeleton and empty state handling
 */

import * as React from "react";
import { useState, useMemo } from "react";
import {
  BarChart3,
  Clock,
  ThumbsUp,
  ThumbsDown,
  Users,
  Activity,
  TrendingUp,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { useAgentTrajectories, useAgentMetrics } from "@/lib/api/hooks";
import type { AgentTrajectory } from "@/lib/api/client";

// =============================================================================
// TYPES
// =============================================================================

interface AgentInsightsProps {
  agentId: string;
  agentName: string;
}

type TimePeriod = "7d" | "30d" | "all";

interface ComputedMetrics {
  totalQueries: number;
  avgResponseTimeMs: number;
  satisfactionPercent: number;
  activeUsers: number;
}

// =============================================================================
// HELPERS
// =============================================================================

function periodToHours(period: TimePeriod): number | undefined {
  switch (period) {
    case "7d":
      return 7 * 24;
    case "30d":
      return 30 * 24;
    case "all":
      return undefined;
  }
}

function filterByPeriod(
  trajectories: AgentTrajectory[],
  period: TimePeriod
): AgentTrajectory[] {
  if (period === "all") return trajectories;

  const hours = periodToHours(period);
  if (!hours) return trajectories;

  const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
  return trajectories.filter(
    (t) => new Date(t.created_at) >= cutoff
  );
}

function computeMetrics(trajectories: AgentTrajectory[]): ComputedMetrics {
  if (trajectories.length === 0) {
    return {
      totalQueries: 0,
      avgResponseTimeMs: 0,
      satisfactionPercent: 0,
      activeUsers: 0,
    };
  }

  const totalQueries = trajectories.length;

  const totalDuration = trajectories.reduce(
    (sum, t) => sum + (t.total_duration_ms || 0),
    0
  );
  const avgResponseTimeMs =
    totalQueries > 0 ? totalDuration / totalQueries : 0;

  // Satisfaction: trajectories with user_rating >= 4 or success === true and no negative rating
  const ratedTrajectories = trajectories.filter(
    (t) => t.user_rating !== null && t.user_rating !== undefined
  );
  let satisfactionPercent: number;
  if (ratedTrajectories.length > 0) {
    const positive = ratedTrajectories.filter(
      (t) => t.user_rating !== null && t.user_rating >= 4
    ).length;
    satisfactionPercent = Math.round(
      (positive / ratedTrajectories.length) * 100
    );
  } else {
    // Fallback to success rate if no user ratings
    const successful = trajectories.filter((t) => t.success).length;
    satisfactionPercent = Math.round((successful / totalQueries) * 100);
  }

  // Active users: count unique session_ids as a proxy for users
  const uniqueSessions = new Set(trajectories.map((t) => t.session_id));
  const activeUsers = uniqueSessions.size;

  return {
    totalQueries,
    avgResponseTimeMs,
    satisfactionPercent,
    activeUsers,
  };
}

function formatDuration(ms: number): string {
  if (ms === 0) return "0s";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function formatTimeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 30) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function truncateText(text: string, maxLength: number): string {
  if (!text) return "";
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + "...";
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function MetricCard({
  icon: Icon,
  value,
  label,
  trend,
  className,
}: {
  icon: React.ElementType;
  value: string;
  label: string;
  trend?: "up" | "down" | "neutral";
  className?: string;
}) {
  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-2xl font-bold tracking-tight">{value}</p>
            <p className="text-xs text-muted-foreground">{label}</p>
          </div>
          <div className="rounded-full bg-muted p-2.5">
            <Icon className="h-4 w-4 text-muted-foreground" />
          </div>
        </div>
        {trend && trend !== "neutral" && (
          <div className="mt-2 flex items-center gap-1">
            <TrendingUp
              className={cn(
                "h-3 w-3",
                trend === "up" ? "text-emerald-500" : "text-red-500 rotate-180"
              )}
            />
            <span
              className={cn(
                "text-xs font-medium",
                trend === "up" ? "text-emerald-500" : "text-red-500"
              )}
            >
              {trend === "up" ? "Improving" : "Declining"}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MetricCardSkeleton() {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <Skeleton className="h-7 w-16" />
            <Skeleton className="h-3 w-20" />
          </div>
          <Skeleton className="h-9 w-9 rounded-full" />
        </div>
      </CardContent>
    </Card>
  );
}

function ActivityItem({ trajectory }: { trajectory: AgentTrajectory }) {
  const isPositive =
    trajectory.user_rating !== null && trajectory.user_rating !== undefined
      ? trajectory.user_rating >= 4
      : trajectory.success;

  return (
    <div className="flex items-center gap-3 rounded-md border px-3 py-2.5 text-sm transition-colors hover:bg-muted/50">
      <div className="flex-1 min-w-0">
        <p className="truncate font-medium">
          {truncateText(trajectory.input_summary || trajectory.task_type, 60)}
        </p>
        <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            {trajectory.task_type}
          </Badge>
          {trajectory.total_tokens > 0 && (
            <span>{trajectory.total_tokens.toLocaleString()} tokens</span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-xs text-muted-foreground tabular-nums">
          {formatDuration(trajectory.total_duration_ms)}
        </span>
        {isPositive ? (
          <ThumbsUp className="h-3.5 w-3.5 text-emerald-500" />
        ) : (
          <ThumbsDown className="h-3.5 w-3.5 text-red-500" />
        )}
        <span className="text-xs text-muted-foreground whitespace-nowrap">
          {formatTimeAgo(trajectory.created_at)}
        </span>
      </div>
    </div>
  );
}

function ActivityItemSkeleton() {
  return (
    <div className="flex items-center gap-3 rounded-md border px-3 py-2.5">
      <div className="flex-1 space-y-1.5">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/3" />
      </div>
      <div className="flex items-center gap-2">
        <Skeleton className="h-3 w-10" />
        <Skeleton className="h-3.5 w-3.5 rounded-full" />
        <Skeleton className="h-3 w-12" />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="rounded-full bg-muted p-4 mb-4">
        <Activity className="h-8 w-8 text-muted-foreground" />
      </div>
      <h3 className="text-sm font-semibold">No usage data yet</h3>
      <p className="mt-1 text-sm text-muted-foreground max-w-xs">
        Usage insights will appear here once this agent starts processing
        queries. Try interacting with the agent to generate data.
      </p>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function AgentInsights({ agentId, agentName }: AgentInsightsProps) {
  const [period, setPeriod] = useState<TimePeriod>("7d");

  // Fetch trajectories for this agent (up to 100 for local computation)
  const {
    data: trajectoriesData,
    isLoading: trajectoriesLoading,
    isError: trajectoriesError,
  } = useAgentTrajectories(
    { agent_id: agentId, limit: 100 },
    { enabled: !!agentId }
  );

  // Also fetch server-side metrics as a complement
  const metricsHours = periodToHours(period) ?? 24 * 365;
  const {
    data: serverMetrics,
    isLoading: metricsLoading,
  } = useAgentMetrics(agentId, metricsHours, { enabled: !!agentId });

  const isLoading = trajectoriesLoading || metricsLoading;

  // Filter trajectories by selected period
  const filteredTrajectories = useMemo(() => {
    const all = trajectoriesData?.trajectories ?? [];
    return filterByPeriod(all, period);
  }, [trajectoriesData, period]);

  // Compute metrics from trajectories
  const metrics = useMemo(
    () => computeMetrics(filteredTrajectories),
    [filteredTrajectories]
  );

  // Use server metrics when available to enrich client-side computation
  const displayMetrics = useMemo(() => {
    if (serverMetrics && serverMetrics.total_executions > 0) {
      return {
        totalQueries: serverMetrics.total_executions,
        avgResponseTimeMs: serverMetrics.avg_latency_ms,
        satisfactionPercent: Math.round(serverMetrics.success_rate * 100),
        activeUsers: metrics.activeUsers, // Server doesn't provide this
      };
    }
    return metrics;
  }, [serverMetrics, metrics]);

  // Recent activity: last 10 from filtered trajectories, sorted newest first
  const recentActivity = useMemo(() => {
    return [...filteredTrajectories]
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )
      .slice(0, 10);
  }, [filteredTrajectories]);

  const hasData =
    !isLoading && !trajectoriesError && displayMetrics.totalQueries > 0;
  const isEmpty =
    !isLoading && !trajectoriesError && displayMetrics.totalQueries === 0;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base">Usage Insights</CardTitle>
            {agentName && (
              <Badge variant="secondary" className="text-xs font-normal">
                {agentName}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            {(["7d", "30d", "all"] as TimePeriod[]).map((p) => (
              <Button
                key={p}
                variant={period === p ? "default" : "ghost"}
                size="sm"
                className="h-7 px-2.5 text-xs"
                onClick={() => setPeriod(p)}
              >
                {p === "all" ? "All" : p}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Metric Cards Grid */}
        {isLoading ? (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </div>
        ) : isEmpty ? (
          <EmptyState />
        ) : hasData ? (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard
                icon={BarChart3}
                value={displayMetrics.totalQueries.toLocaleString()}
                label="Total Queries"
              />
              <MetricCard
                icon={Clock}
                value={formatDuration(displayMetrics.avgResponseTimeMs)}
                label="Avg Response Time"
              />
              <MetricCard
                icon={ThumbsUp}
                value={`${displayMetrics.satisfactionPercent}%`}
                label="Satisfaction"
                trend={
                  displayMetrics.satisfactionPercent >= 80
                    ? "up"
                    : displayMetrics.satisfactionPercent >= 50
                      ? "neutral"
                      : "down"
                }
              />
              <MetricCard
                icon={Users}
                value={displayMetrics.activeUsers.toLocaleString()}
                label="Active Users"
              />
            </div>

            {/* Recent Activity */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-semibold">Recent Activity</h4>
                <span className="text-xs text-muted-foreground">
                  ({recentActivity.length} most recent)
                </span>
              </div>

              {recentActivity.length > 0 ? (
                <ScrollArea className="h-[320px]">
                  <div className="space-y-2 pr-3">
                    {recentActivity.map((trajectory) => (
                      <ActivityItem
                        key={trajectory.id}
                        trajectory={trajectory}
                      />
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <p className="text-sm text-muted-foreground py-4 text-center">
                  No recent activity in this period.
                </p>
              )}
            </div>
          </>
        ) : null}

        {/* Error state */}
        {trajectoriesError && (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="rounded-full bg-red-100 dark:bg-red-900/20 p-3 mb-3">
              <Activity className="h-6 w-6 text-red-500" />
            </div>
            <p className="text-sm text-muted-foreground">
              Failed to load usage data. The analytics endpoint may not be
              available.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default AgentInsights;
