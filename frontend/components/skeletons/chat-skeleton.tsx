"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

/**
 * Skeleton for individual chat message
 */
export function ChatMessageSkeleton({ isUser = false }: { isUser?: boolean }) {
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Skeleton className="h-8 w-8 rounded-full shrink-0" />
      <div className={`space-y-2 max-w-[80%] ${isUser ? "items-end" : ""}`}>
        <div className="flex items-center gap-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-3 w-12" />
        </div>
        <Skeleton className={`h-16 ${isUser ? "w-48" : "w-64"} rounded-lg`} />
      </div>
    </div>
  );
}

/**
 * Skeleton for chat messages list
 */
export function ChatMessagesSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="space-y-6 p-4">
      {Array.from({ length: count }).map((_, i) => (
        <ChatMessageSkeleton key={i} isUser={i % 2 === 1} />
      ))}
    </div>
  );
}

/**
 * Skeleton for chat session list
 */
export function ChatSessionListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-2 p-2">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="p-3 rounded-lg border">
          <div className="flex items-start justify-between">
            <div className="space-y-1.5 flex-1">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
            <Skeleton className="h-6 w-6 rounded" />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Skeleton for source panel
 */
export function SourcePanelSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-3 p-4">
      <div className="flex items-center justify-between">
        <Skeleton className="h-5 w-20" />
        <Skeleton className="h-4 w-12" />
      </div>
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i}>
          <CardContent className="p-3">
            <div className="flex items-start gap-2">
              <Skeleton className="h-4 w-4 mt-0.5 shrink-0" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-3 w-3/4" />
                <div className="flex gap-2 mt-2">
                  <Skeleton className="h-5 w-16 rounded-full" />
                  <Skeleton className="h-5 w-12 rounded-full" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

/**
 * Full chat page skeleton
 */
export function ChatPageSkeleton() {
  return (
    <div className="flex h-[calc(100vh-10rem)] gap-4">
      {/* Sessions sidebar */}
      <div className="w-64 border rounded-lg shrink-0 hidden lg:block">
        <div className="p-3 border-b">
          <Skeleton className="h-9 w-full" />
        </div>
        <ChatSessionListSkeleton count={6} />
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col border rounded-lg">
        {/* Chat header */}
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Skeleton className="h-5 w-5" />
            <Skeleton className="h-5 w-40" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton className="h-8 w-8 rounded" />
            <Skeleton className="h-8 w-8 rounded" />
          </div>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-hidden">
          <ChatMessagesSkeleton count={4} />
        </div>

        {/* Input area */}
        <div className="p-4 border-t">
          <div className="flex items-center gap-2">
            <Skeleton className="h-10 flex-1 rounded-lg" />
            <Skeleton className="h-10 w-10 rounded-lg" />
          </div>
        </div>
      </div>

      {/* Sources sidebar */}
      <div className="w-72 border rounded-lg shrink-0 hidden xl:block">
        <div className="p-3 border-b">
          <Skeleton className="h-5 w-24" />
        </div>
        <SourcePanelSkeleton count={3} />
      </div>
    </div>
  );
}
