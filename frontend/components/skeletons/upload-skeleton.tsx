"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

/**
 * Skeleton for file upload dropzone
 */
export function UploadDropzoneSkeleton() {
  return (
    <Card>
      <CardContent className="p-8">
        <div className="flex flex-col items-center justify-center py-10 border-2 border-dashed rounded-lg">
          <Skeleton className="h-12 w-12 rounded-full mb-4" />
          <Skeleton className="h-5 w-64 mb-2" />
          <Skeleton className="h-4 w-48 mb-4" />
          <Skeleton className="h-9 w-28" />
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Skeleton for upload queue item
 */
export function UploadQueueItemSkeleton() {
  return (
    <div className="flex items-center gap-3 p-3 border rounded-lg">
      <Skeleton className="h-10 w-10 rounded" />
      <div className="flex-1 space-y-2">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-4 w-16" />
        </div>
        <Skeleton className="h-2 w-full rounded-full" />
      </div>
      <Skeleton className="h-8 w-8 rounded" />
    </div>
  );
}

/**
 * Skeleton for upload queue
 */
export function UploadQueueSkeleton({ count = 3 }: { count?: number }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <Skeleton className="h-5 w-28" />
        <Skeleton className="h-4 w-16" />
      </CardHeader>
      <CardContent className="space-y-3">
        {Array.from({ length: count }).map((_, i) => (
          <UploadQueueItemSkeleton key={i} />
        ))}
      </CardContent>
    </Card>
  );
}

/**
 * Skeleton for upload settings
 */
export function UploadSettingsSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-5 w-32" />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-10 w-full" />
        </div>
        <div className="space-y-2">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-10 w-full" />
        </div>
        <div className="flex items-center gap-2">
          <Skeleton className="h-4 w-4" />
          <Skeleton className="h-4 w-32" />
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Full upload page skeleton
 */
export function UploadPageSkeleton() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-1">
        <Skeleton className="h-9 w-40" />
        <Skeleton className="h-5 w-80" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Main upload area */}
        <div className="lg:col-span-2 space-y-6">
          <UploadDropzoneSkeleton />
          <UploadQueueSkeleton count={2} />
        </div>

        {/* Settings sidebar */}
        <div>
          <UploadSettingsSkeleton />
        </div>
      </div>
    </div>
  );
}
