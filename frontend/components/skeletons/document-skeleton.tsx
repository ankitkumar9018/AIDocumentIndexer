"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent } from "@/components/ui/card";

/**
 * Skeleton loader for document list view
 */
export function DocumentListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <Card>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="w-12 p-3">
                  <Skeleton className="h-4 w-4" />
                </th>
                <th className="text-left p-3">
                  <Skeleton className="h-4 w-16" />
                </th>
                <th className="text-left p-3">
                  <Skeleton className="h-4 w-20" />
                </th>
                <th className="text-left p-3">
                  <Skeleton className="h-4 w-12" />
                </th>
                <th className="text-left p-3">
                  <Skeleton className="h-4 w-12" />
                </th>
                <th className="w-12 p-3"></th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: count }).map((_, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="p-3">
                    <Skeleton className="h-4 w-4" />
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-3">
                      <Skeleton className="h-5 w-5 rounded" />
                      <div className="space-y-1.5">
                        <Skeleton className="h-4 w-48" />
                        <Skeleton className="h-3 w-20" />
                      </div>
                    </div>
                  </td>
                  <td className="p-3">
                    <Skeleton className="h-6 w-24 rounded-full" />
                  </td>
                  <td className="p-3">
                    <Skeleton className="h-4 w-16" />
                  </td>
                  <td className="p-3">
                    <Skeleton className="h-4 w-20" />
                  </td>
                  <td className="p-3">
                    <Skeleton className="h-8 w-8 rounded" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Skeleton loader for document grid view
 */
export function DocumentGridSkeleton({ count = 8 }: { count?: number }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i}>
          <CardContent className="pt-6">
            <div className="flex items-start justify-between">
              <Skeleton className="h-14 w-14 rounded-lg" />
              <Skeleton className="h-8 w-8 rounded" />
            </div>
            <div className="mt-4 space-y-2">
              <Skeleton className="h-5 w-3/4" />
              <Skeleton className="h-4 w-16" />
            </div>
            <div className="flex flex-col gap-2 mt-4 pt-4 border-t">
              <div className="flex gap-1">
                <Skeleton className="h-5 w-16 rounded-full" />
                <Skeleton className="h-5 w-12 rounded-full" />
              </div>
              <Skeleton className="h-3 w-24" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

/**
 * Skeleton for document stats cards
 */
export function DocumentStatsSkeleton() {
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {Array.from({ length: 3 }).map((_, i) => (
        <Card key={i}>
          <CardContent className="pt-6">
            <Skeleton className="h-8 w-20 mb-1" />
            <Skeleton className="h-4 w-28" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

/**
 * Full documents page skeleton
 */
export function DocumentsPageSkeleton() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <Skeleton className="h-9 w-40" />
          <Skeleton className="h-5 w-72" />
        </div>
        <div className="flex items-center gap-3">
          <Skeleton className="h-10 w-48 rounded-lg" />
          <Skeleton className="h-9 w-24" />
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex flex-col sm:flex-row gap-4">
        <Skeleton className="h-10 flex-1" />
        <div className="flex items-center gap-2">
          <Skeleton className="h-10 w-28" />
          <Skeleton className="h-10 w-28" />
          <Skeleton className="h-10 w-28" />
          <Skeleton className="h-10 w-36" />
          <Skeleton className="h-10 w-20" />
        </div>
      </div>

      {/* Document List */}
      <DocumentListSkeleton count={5} />

      {/* Pagination */}
      <div className="flex items-center justify-between py-4">
        <Skeleton className="h-8 w-48" />
        <div className="flex items-center gap-1">
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-24" />
          <Skeleton className="h-8 w-8" />
          <Skeleton className="h-8 w-8" />
        </div>
      </div>

      {/* Stats */}
      <DocumentStatsSkeleton />
    </div>
  );
}
