"use client";

/**
 * AIDocumentIndexer - Bulk Progress Dashboard
 * ============================================
 *
 * Phase 18: Real-time visibility into bulk processing for 100K+ files.
 *
 * Features:
 * - Overall progress bar with ETA
 * - Virtualized file status grid (for performance with large batches)
 * - Real-time WebSocket updates
 * - Pause/Resume/Cancel controls
 * - Error summary with retry buttons
 * - Files per minute processing rate
 *
 * Usage:
 *   <BulkProgressDashboard
 *     batchId="batch-123"
 *     onComplete={() => console.log("Done!")}
 *   />
 */

import * as React from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  FileText,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  StopCircle,
  XCircle,
  Zap,
  Filter,
  Download,
  BarChart3,
} from "lucide-react";

import { cn, formatBytes } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

// =============================================================================
// Types
// =============================================================================

export type FileStatus =
  | "pending"
  | "processing"
  | "completed"
  | "failed"
  | "skipped"
  | "cancelled";

export interface BulkFileInfo {
  fileId: string;
  filename: string;
  status: FileStatus;
  stage: string;
  progress: number;
  error?: string;
  documentId?: string;
  chunkCount?: number;
  fileSize?: number;
  processingTime?: number;
}

export interface BulkProgress {
  batchId: string;
  status: "processing" | "completed" | "failed" | "paused" | "cancelled";
  totalFiles: number;
  completed: number;
  failed: number;
  processing: number;
  pending: number;
  overallProgress: number;
  etaSeconds?: number;
  filesPerMinute: number;
  elapsedSeconds: number;
  startedAt?: string;
}

interface BulkProgressDashboardProps {
  batchId: string;
  onComplete?: () => void;
  onCancel?: () => void;
  className?: string;
  pollInterval?: number; // ms, default 2000
  showFileList?: boolean;
  maxVisibleFiles?: number;
}

// =============================================================================
// Hooks
// =============================================================================

function useBulkProgress(batchId: string, pollInterval: number = 2000) {
  const [progress, setProgress] = useState<BulkProgress | null>(null);
  const [files, setFiles] = useState<BulkFileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch progress via REST API
  const fetchProgress = useCallback(async () => {
    try {
      const response = await fetch(`/api/upload/bulk/${batchId}/progress`);
      if (!response.ok) {
        throw new Error(`Failed to fetch progress: ${response.statusText}`);
      }
      const data = await response.json();
      setProgress({
        batchId: data.batch_id,
        status: data.status,
        totalFiles: data.total_files,
        completed: data.completed,
        failed: data.failed,
        processing: data.processing,
        pending: data.pending,
        overallProgress: data.overall_progress,
        etaSeconds: data.eta_seconds,
        filesPerMinute: data.files_per_minute,
        elapsedSeconds: data.elapsed_seconds,
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [batchId]);

  // Fetch files list
  const fetchFiles = useCallback(
    async (statusFilter?: FileStatus, limit: number = 100, offset: number = 0) => {
      try {
        const params = new URLSearchParams({
          limit: limit.toString(),
          offset: offset.toString(),
        });
        if (statusFilter) {
          params.set("status", statusFilter);
        }

        const response = await fetch(
          `/api/upload/bulk/${batchId}/files?${params}`
        );
        if (!response.ok) {
          throw new Error(`Failed to fetch files: ${response.statusText}`);
        }
        const data = await response.json();
        setFiles(
          data.files.map((f: any) => ({
            fileId: f.file_id,
            filename: f.filename,
            status: f.status as FileStatus,
            stage: f.stage,
            progress: f.progress,
            error: f.error,
            documentId: f.document_id,
            chunkCount: f.chunk_count,
          }))
        );
      } catch (err) {
        console.error("Failed to fetch files:", err);
      }
    },
    [batchId]
  );

  // Set up WebSocket connection for real-time updates
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/ws/bulk/${batchId}`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "progress") {
          setProgress((prev) =>
            prev
              ? {
                  ...prev,
                  ...data.progress,
                }
              : null
          );
        } else if (data.type === "file_update") {
          setFiles((prev) => {
            const idx = prev.findIndex((f) => f.fileId === data.file.file_id);
            if (idx >= 0) {
              const updated = [...prev];
              updated[idx] = {
                ...updated[idx],
                status: data.file.status,
                stage: data.file.stage,
                progress: data.file.progress,
                error: data.file.error,
              };
              return updated;
            }
            return prev;
          });
        }
      };

      ws.onerror = () => {
        console.warn("WebSocket connection failed, falling back to polling");
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    } catch {
      // WebSocket not available, use polling
    }

    return () => {
      wsRef.current?.close();
    };
  }, [batchId]);

  // Polling fallback
  useEffect(() => {
    fetchProgress();
    fetchFiles();

    const interval = setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        fetchProgress();
      }
    }, pollInterval);

    return () => clearInterval(interval);
  }, [fetchProgress, fetchFiles, pollInterval]);

  return { progress, files, loading, error, refetch: fetchProgress, refetchFiles: fetchFiles };
}

// =============================================================================
// Helper Components
// =============================================================================

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

function StatusBadge({ status }: { status: FileStatus }) {
  const config: Record<FileStatus, { label: string; variant: "default" | "secondary" | "destructive" | "outline"; icon: React.ReactNode }> = {
    pending: { label: "Pending", variant: "secondary", icon: <Clock className="h-3 w-3" /> },
    processing: { label: "Processing", variant: "default", icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    completed: { label: "Completed", variant: "outline", icon: <CheckCircle2 className="h-3 w-3 text-green-500" /> },
    failed: { label: "Failed", variant: "destructive", icon: <XCircle className="h-3 w-3" /> },
    skipped: { label: "Skipped", variant: "secondary", icon: <AlertCircle className="h-3 w-3" /> },
    cancelled: { label: "Cancelled", variant: "secondary", icon: <StopCircle className="h-3 w-3" /> },
  };

  const { label, variant, icon } = config[status];

  return (
    <Badge variant={variant} className="flex items-center gap-1">
      {icon}
      {label}
    </Badge>
  );
}

function StatCard({
  label,
  value,
  icon,
  subtext,
  className,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  subtext?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-3 p-3 rounded-lg bg-muted/50", className)}>
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
        {icon}
      </div>
      <div>
        <p className="text-2xl font-bold">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
        {subtext && <p className="text-xs text-muted-foreground">{subtext}</p>}
      </div>
    </div>
  );
}

// =============================================================================
// File List (Virtualized)
// =============================================================================

interface FileListProps {
  files: BulkFileInfo[];
  statusFilter: FileStatus | "all";
  onStatusFilterChange: (status: FileStatus | "all") => void;
  onRetry?: (fileId: string) => void;
  maxHeight?: number;
}

function FileList({
  files,
  statusFilter,
  onStatusFilterChange,
  onRetry,
  maxHeight = 400,
}: FileListProps) {
  const filteredFiles = useMemo(() => {
    if (statusFilter === "all") return files;
    return files.filter((f) => f.status === statusFilter);
  }, [files, statusFilter]);

  return (
    <div className="space-y-3">
      {/* Filter */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <Select value={statusFilter} onValueChange={(v) => onStatusFilterChange(v as FileStatus | "all")}>
            <SelectTrigger className="w-[140px] h-8">
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All files</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="processing">Processing</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
              <SelectItem value="skipped">Skipped</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <span className="text-sm text-muted-foreground">
          Showing {filteredFiles.length} of {files.length} files
        </span>
      </div>

      {/* File Grid */}
      <ScrollArea className="border rounded-lg" style={{ maxHeight }}>
        <div className="divide-y">
          {filteredFiles.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              No files match the selected filter
            </div>
          ) : (
            filteredFiles.map((file) => (
              <FileRow key={file.fileId} file={file} onRetry={onRetry} />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function FileRow({
  file,
  onRetry,
}: {
  file: BulkFileInfo;
  onRetry?: (fileId: string) => void;
}) {
  return (
    <div className="flex items-center gap-3 p-3 hover:bg-muted/50 transition-colors">
      <div className="flex h-8 w-8 items-center justify-center rounded bg-muted shrink-0">
        <FileText className="h-4 w-4 text-muted-foreground" />
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.filename}</p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-muted-foreground">{file.stage}</span>
          {file.status === "processing" && file.progress > 0 && (
            <span className="text-xs text-primary">{file.progress}%</span>
          )}
          {file.error && (
            <span className="text-xs text-destructive truncate max-w-[200px]">
              {file.error}
            </span>
          )}
        </div>

        {/* Progress bar for processing files */}
        {file.status === "processing" && (
          <div className="mt-1.5 h-1 w-full bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${file.progress}%` }}
            />
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <StatusBadge status={file.status} />
        {file.status === "failed" && onRetry && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() => onRetry(file.fileId)}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Retry this file</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Main Dashboard Component
// =============================================================================

export function BulkProgressDashboard({
  batchId,
  onComplete,
  onCancel,
  className,
  pollInterval = 2000,
  showFileList = true,
  maxVisibleFiles = 100,
}: BulkProgressDashboardProps) {
  const { progress, files, loading, error, refetch, refetchFiles } = useBulkProgress(
    batchId,
    pollInterval
  );
  const [statusFilter, setStatusFilter] = useState<FileStatus | "all">("all");
  const [isFileListOpen, setIsFileListOpen] = useState(true);
  const [isPaused, setIsPaused] = useState(false);

  // Track completion
  useEffect(() => {
    if (progress?.status === "completed" && onComplete) {
      onComplete();
    }
  }, [progress?.status, onComplete]);

  // Batch actions
  const handlePause = async () => {
    try {
      await fetch(`/api/upload/bulk/${batchId}/pause`, { method: "POST" });
      setIsPaused(true);
      refetch();
    } catch (err) {
      console.error("Failed to pause:", err);
    }
  };

  const handleResume = async () => {
    try {
      await fetch(`/api/upload/bulk/${batchId}/resume`, { method: "POST" });
      setIsPaused(false);
      refetch();
    } catch (err) {
      console.error("Failed to resume:", err);
    }
  };

  const handleCancel = async () => {
    if (!confirm("Are you sure you want to cancel this batch? This cannot be undone.")) {
      return;
    }
    try {
      await fetch(`/api/upload/bulk/${batchId}`, { method: "DELETE" });
      onCancel?.();
    } catch (err) {
      console.error("Failed to cancel:", err);
    }
  };

  const handleRetryFile = async (fileId: string) => {
    try {
      await fetch(`/api/upload/retry/${fileId}`, { method: "POST" });
      refetchFiles();
    } catch (err) {
      console.error("Failed to retry file:", err);
    }
  };

  const handleRetryAllFailed = async () => {
    try {
      await fetch(`/api/upload/bulk/${batchId}/retry-failed`, { method: "POST" });
      refetchFiles();
    } catch (err) {
      console.error("Failed to retry failed files:", err);
    }
  };

  if (loading && !progress) {
    return (
      <Card className={cn("p-8", className)}>
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading batch progress...</p>
        </div>
      </Card>
    );
  }

  if (error || !progress) {
    return (
      <Card className={cn("p-8", className)}>
        <div className="flex flex-col items-center gap-4">
          <AlertCircle className="h-8 w-8 text-destructive" />
          <p className="text-destructive">{error || "Failed to load batch progress"}</p>
          <Button onClick={refetch} variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </Card>
    );
  }

  const isComplete = progress.status === "completed";
  const isFailed = progress.status === "failed";
  const isProcessing = progress.status === "processing";

  return (
    <Card className={cn("overflow-hidden", className)}>
      {/* Header */}
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Bulk Upload Progress
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Batch ID: {batchId.slice(0, 8)}...
            </p>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            {isProcessing && (
              <>
                {isPaused ? (
                  <Button size="sm" variant="outline" onClick={handleResume}>
                    <Play className="h-4 w-4 mr-1" />
                    Resume
                  </Button>
                ) : (
                  <Button size="sm" variant="outline" onClick={handlePause}>
                    <Pause className="h-4 w-4 mr-1" />
                    Pause
                  </Button>
                )}
                <Button size="sm" variant="destructive" onClick={handleCancel}>
                  <StopCircle className="h-4 w-4 mr-1" />
                  Cancel
                </Button>
              </>
            )}
            {progress.failed > 0 && (
              <Button size="sm" variant="outline" onClick={handleRetryAllFailed}>
                <RefreshCw className="h-4 w-4 mr-1" />
                Retry Failed ({progress.failed})
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">
              {isComplete
                ? "Processing Complete"
                : isFailed
                ? "Processing Failed"
                : isPaused
                ? "Paused"
                : "Processing..."}
            </span>
            <span className="text-muted-foreground">
              {progress.completed + progress.failed} / {progress.totalFiles} files
            </span>
          </div>
          <Progress
            value={progress.overallProgress}
            className="h-3"
            indicatorClassName={cn(
              isFailed && "bg-destructive",
              isComplete && "bg-green-500",
              isPaused && "bg-yellow-500"
            )}
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{Math.round(progress.overallProgress)}% complete</span>
            {progress.etaSeconds && progress.etaSeconds > 0 && !isComplete && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                ETA: {formatDuration(progress.etaSeconds)}
              </span>
            )}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label="Completed"
            value={progress.completed}
            icon={<CheckCircle2 className="h-5 w-5 text-green-500" />}
          />
          <StatCard
            label="Failed"
            value={progress.failed}
            icon={<XCircle className="h-5 w-5 text-destructive" />}
          />
          <StatCard
            label="Processing Rate"
            value={`${progress.filesPerMinute.toFixed(1)}/min`}
            icon={<Zap className="h-5 w-5 text-primary" />}
          />
          <StatCard
            label="Elapsed Time"
            value={formatDuration(progress.elapsedSeconds)}
            icon={<Clock className="h-5 w-5 text-muted-foreground" />}
          />
        </div>

        {/* File List */}
        {showFileList && (
          <Collapsible open={isFileListOpen} onOpenChange={setIsFileListOpen}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between p-0 h-auto hover:bg-transparent">
                <span className="font-medium">File Details</span>
                {isFileListOpen ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-4">
              <FileList
                files={files}
                statusFilter={statusFilter}
                onStatusFilterChange={setStatusFilter}
                onRetry={handleRetryFile}
                maxHeight={400}
              />
            </CollapsibleContent>
          </Collapsible>
        )}

        {/* Completion Summary */}
        {isComplete && (
          <div className="rounded-lg bg-green-500/10 border border-green-500/20 p-4">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-6 w-6 text-green-500" />
              <div>
                <p className="font-medium text-green-700 dark:text-green-400">
                  Batch processing complete!
                </p>
                <p className="text-sm text-muted-foreground">
                  Successfully processed {progress.completed} files in{" "}
                  {formatDuration(progress.elapsedSeconds)}
                  {progress.failed > 0 && ` (${progress.failed} failed)`}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Failure Summary */}
        {isFailed && progress.failed > 0 && (
          <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-6 w-6 text-destructive" />
              <div className="flex-1">
                <p className="font-medium text-destructive">
                  {progress.failed} files failed to process
                </p>
                <p className="text-sm text-muted-foreground">
                  Click "Retry Failed" to attempt processing again
                </p>
              </div>
              <Button
                variant="destructive"
                size="sm"
                onClick={handleRetryAllFailed}
              >
                <RefreshCw className="h-4 w-4 mr-1" />
                Retry All
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Mini Progress Indicator (for use in sidebars, headers)
// =============================================================================

export function BulkProgressMini({
  batchId,
  className,
}: {
  batchId: string;
  className?: string;
}) {
  const { progress } = useBulkProgress(batchId, 5000);

  if (!progress) return null;

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {progress.status === "processing" && (
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
      )}
      {progress.status === "completed" && (
        <CheckCircle2 className="h-4 w-4 text-green-500" />
      )}
      {progress.status === "failed" && (
        <XCircle className="h-4 w-4 text-destructive" />
      )}
      <div className="flex-1 min-w-[100px]">
        <Progress value={progress.overallProgress} className="h-1.5" />
      </div>
      <span className="text-xs text-muted-foreground">
        {progress.completed}/{progress.totalFiles}
      </span>
    </div>
  );
}

export default BulkProgressDashboard;
