"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

/**
 * Skeleton for stats card
 */
export function StatsCardSkeleton() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-4" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-8 w-20 mb-1" />
        <Skeleton className="h-3 w-32" />
      </CardContent>
    </Card>
  );
}

/**
 * Skeleton for stats grid
 */
export function StatsGridSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <StatsCardSkeleton key={i} />
      ))}
    </div>
  );
}

/**
 * Skeleton for chart card
 */
export function ChartSkeleton({ height = 300 }: { height?: number }) {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-5 w-32 mb-1" />
        <Skeleton className="h-4 w-48" />
      </CardHeader>
      <CardContent>
        <Skeleton className={`w-full rounded-lg`} style={{ height }} />
      </CardContent>
    </Card>
  );
}

/**
 * Skeleton for activity list
 */
export function ActivityListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-5 w-28" />
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {Array.from({ length: count }).map((_, i) => (
            <div key={i} className="flex items-start gap-3">
              <Skeleton className="h-8 w-8 rounded-full shrink-0" />
              <div className="flex-1 space-y-1.5">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-3 w-20" />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Skeleton for quick actions
 */
export function QuickActionsSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-5 w-28" />
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-12 rounded-lg" />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Full dashboard page skeleton
 */
export function DashboardPageSkeleton() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-1">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-5 w-72" />
      </div>

      {/* Stats Grid */}
      <StatsGridSkeleton count={4} />

      {/* Charts Row */}
      <div className="grid gap-4 md:grid-cols-2">
        <ChartSkeleton height={250} />
        <ChartSkeleton height={250} />
      </div>

      {/* Bottom Row */}
      <div className="grid gap-4 md:grid-cols-2">
        <ActivityListSkeleton count={5} />
        <QuickActionsSkeleton />
      </div>
    </div>
  );
}
