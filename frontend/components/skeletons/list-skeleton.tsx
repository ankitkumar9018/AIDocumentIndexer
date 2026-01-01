"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

/**
 * Skeleton for a generic list item with icon and text
 */
export function ListItemSkeleton() {
  return (
    <div className="flex items-center justify-between p-2 rounded-lg">
      <div className="flex items-center gap-3">
        <Skeleton className="h-5 w-5 rounded" />
        <div className="space-y-1.5">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-20" />
        </div>
      </div>
      <Skeleton className="h-8 w-14 rounded" />
    </div>
  );
}

/**
 * Skeleton for a list of items in a card
 */
export function ListCardSkeleton({
  title,
  count = 5,
}: {
  title?: string;
  count?: number;
}) {
  return (
    <Card>
      <CardHeader>
        {title ? (
          <>
            <Skeleton className="h-5 w-36" />
            <Skeleton className="h-4 w-48" />
          </>
        ) : (
          <>
            <Skeleton className="h-5 w-36" />
            <Skeleton className="h-4 w-48" />
          </>
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {Array.from({ length: count }).map((_, i) => (
            <ListItemSkeleton key={i} />
          ))}
        </div>
        <Skeleton className="h-9 w-full mt-4" />
      </CardContent>
    </Card>
  );
}

/**
 * Skeleton for processing queue item with status indicator
 */
export function QueueItemSkeleton() {
  return (
    <div className="flex items-center gap-3 p-2 rounded-lg">
      <Skeleton className="h-8 w-8 rounded-full" />
      <div className="flex-1 min-w-0 space-y-1">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-3 w-16" />
      </div>
      <Skeleton className="h-4 w-8" />
    </div>
  );
}

/**
 * Skeleton for processing queue card
 */
export function QueueCardSkeleton({ count = 5 }: { count?: number }) {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-5 w-36" />
        <Skeleton className="h-4 w-56" />
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {Array.from({ length: count }).map((_, i) => (
            <QueueItemSkeleton key={i} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
