"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import {
  RefreshCw,
  Database,
  Network,
  Play,
  Square,
  Loader2,
  CheckCircle,
  AlertCircle,
  Clock,
  Trash2,
  ChevronDown,
  ChevronRight,
  FileText,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { TabsContent } from "@/components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { toast } from "sonner";

interface ExtractionJobDocumentDetail {
  document_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  chunks_processed: number | null;
  kg_entity_count: number;
  kg_relation_count: number;
}

// API base URL - use backend directly to avoid Next.js routing issues
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface ReindexJob {
  job_id: string;
  status: string;
  message: string;
  total_documents: number;
  processed: number;
  progress: number;
  processing_mode?: string;
  current_document?: string;
  errors: string[];
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

interface KGExtractionJob {
  job_id: string;
  status: string;
  progress_percent: number;
  total_documents: number;
  processed_documents: number;
  failed_documents: number;
  total_entities: number;
  total_relations: number;
  current_document?: string;
  error_count: number;
  estimated_remaining_seconds?: number;
  avg_doc_processing_time?: number;
  started_at?: string;
  completed_at?: string;
  can_cancel: boolean;
  can_pause: boolean;
  can_resume: boolean;
}

interface EmbeddingStats {
  total_chunks: number;
  chunks_with_embedding: number;
  chunks_without_embedding: number;
  chunks_with_null_embedding: number;
  orphaned_chunks: number;
  embedding_coverage_percent: number;
  total_documents: number;
  documents_with_issues: number;
  chromadb_total_items: number | null;
  embedding_dimension: number;
  problem_documents: Array<{ filename: string; problem_chunks: number }>;
}

export function MaintenanceTab() {
  const { data: session } = useSession();
  const accessToken = (session as any)?.accessToken as string | undefined;

  // Reindex state
  const [reindexJob, setReindexJob] = useState<ReindexJob | null>(null);
  const [processingMode, setProcessingMode] = useState<"linear" | "parallel">("linear");
  const [parallelCount, setParallelCount] = useState(2);
  const [reindexBatchSize, setReindexBatchSize] = useState(5);
  const [reindexDelay, setReindexDelay] = useState(15);
  const [isStartingReindex, setIsStartingReindex] = useState(false);

  // KG Extraction state
  const [kgJob, setKgJob] = useState<KGExtractionJob | null>(null);
  const [isStartingKG, setIsStartingKG] = useState(false);
  const [kgOnlyNew, setKgOnlyNew] = useState(true);

  // KG document details state
  const [kgDocDetails, setKgDocDetails] = useState<ExtractionJobDocumentDetail[] | null>(null);
  const [isDocDetailsExpanded, setIsDocDetailsExpanded] = useState(false);

  // Embedding stats state
  const [embeddingStats, setEmbeddingStats] = useState<EmbeddingStats | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [isCleaningOrphans, setIsCleaningOrphans] = useState(false);
  const [isSyncingChroma, setIsSyncingChroma] = useState(false);

  // Cleanup orphaned chunks
  const cleanupOrphanedChunks = async () => {
    if (!accessToken) return;
    setIsCleaningOrphans(true);
    try {
      const response = await fetch(`${API_BASE_URL}/embeddings/cleanup-orphaned`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (response.ok) {
        const data = await response.json();
        toast.success("Cleanup Complete", {
          description: data.message,
        });
        fetchEmbeddingStats(); // Refresh stats
      } else {
        const error = await response.json();
        toast.error("Cleanup Failed", {
          description: error.detail || "Unknown error",
        });
      }
    } catch (error) {
      console.error("Failed to cleanup orphaned chunks:", error);
      toast.error("Cleanup Failed", {
        description: "Network error during cleanup",
      });
    } finally {
      setIsCleaningOrphans(false);
    }
  };

  // Sync missing chunks to ChromaDB
  const syncChromaDB = async () => {
    if (!accessToken) return;
    setIsSyncingChroma(true);
    try {
      const response = await fetch(`${API_BASE_URL}/embeddings/sync-chromadb`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });
      if (response.ok) {
        const data = await response.json();
        toast.success("ChromaDB Sync Complete", {
          description: data.message,
        });
        fetchEmbeddingStats();
      } else {
        const error = await response.json();
        toast.error("Sync Failed", {
          description: error.detail || error.message || "Unknown error",
        });
      }
    } catch (error) {
      console.error("Failed to sync ChromaDB:", error);
      toast.error("Sync Failed", {
        description: "Network error during sync",
      });
    } finally {
      setIsSyncingChroma(false);
    }
  };

  // Fetch embedding stats
  const fetchEmbeddingStats = useCallback(async () => {
    if (!accessToken) return;
    setIsLoadingStats(true);
    try {
      const response = await fetch(`${API_BASE_URL}/embeddings/health`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (response.ok) {
        const data = await response.json();
        setEmbeddingStats(data);
      }
    } catch (error) {
      console.error("Failed to fetch embedding stats:", error);
    } finally {
      setIsLoadingStats(false);
    }
  }, [accessToken]);

  // Fetch current job status
  const fetchReindexStatus = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await fetch(`${API_BASE_URL}/embeddings/reindex-all/current`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (response.ok) {
        const data = await response.json();
        setReindexJob(data);
      } else {
        setReindexJob(null);
      }
    } catch (error) {
      console.error("Failed to fetch reindex status:", error);
    }
  }, [accessToken]);

  const fetchKGStatus = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await fetch(`${API_BASE_URL}/knowledge-graph/extraction-jobs/current`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (response.ok) {
        const data = await response.json();
        setKgJob(data);
      } else {
        setKgJob(null);
      }
    } catch (error) {
      console.error("Failed to fetch KG extraction status:", error);
    }
  }, [accessToken]);

  const fetchKGDocDetails = useCallback(async () => {
    if (!accessToken || !kgJob?.job_id) return;
    try {
      const response = await fetch(
        `${API_BASE_URL}/knowledge-graph/extraction-jobs/${kgJob.job_id}/documents`,
        { headers: { Authorization: `Bearer ${accessToken}` } }
      );
      if (response.ok) {
        const data = await response.json();
        setKgDocDetails(data.documents);
      }
    } catch (error) {
      console.error("Failed to fetch KG document details:", error);
    }
  }, [accessToken, kgJob?.job_id]);

  // Poll for document details when expanded
  useEffect(() => {
    if (!isDocDetailsExpanded || !kgJob?.job_id) return;

    // Fetch immediately on expand
    fetchKGDocDetails();

    // Poll only while job is active
    const isActive = kgJob.status === "running" || kgJob.status === "paused" || kgJob.status === "pending";
    if (!isActive) return;

    const interval = setInterval(fetchKGDocDetails, 5000);
    return () => clearInterval(interval);
  }, [isDocDetailsExpanded, kgJob?.job_id, kgJob?.status, fetchKGDocDetails]);

  // Poll for status while jobs are running
  useEffect(() => {
    fetchReindexStatus();
    fetchKGStatus();
    fetchEmbeddingStats();

    const interval = setInterval(() => {
      // Poll for all active job states (started, pending, running, cancelling)
      if (reindexJob?.status === "started" || reindexJob?.status === "pending" || reindexJob?.status === "running" || reindexJob?.status === "cancelling") {
        fetchReindexStatus();
      }
      if (kgJob?.status === "pending" || kgJob?.status === "running" || kgJob?.status === "paused") {
        fetchKGStatus();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [fetchReindexStatus, fetchKGStatus, fetchEmbeddingStats, reindexJob?.status, kgJob?.status]);

  // Start reindex
  const startReindex = async () => {
    if (!accessToken) return;
    setIsStartingReindex(true);
    try {
      const response = await fetch(`${API_BASE_URL}/embeddings/reindex-all`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          processing_mode: processingMode,
          parallel_count: parallelCount,
          batch_size: reindexBatchSize,
          delay_seconds: reindexDelay,
          force_reembed: true,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setReindexJob(data);
        toast.success("Reindex Started", {
          description: data.message,
        });
      } else {
        const error = await response.json();
        toast.error("Failed to start reindex", {
          description: error.detail || "Unknown error",
        });
      }
    } catch (error) {
      toast.error("Error", {
        description: "Failed to start reindex job",
      });
    } finally {
      setIsStartingReindex(false);
    }
  };

  // Cancel reindex
  const cancelReindex = async () => {
    if (!reindexJob?.job_id || !accessToken) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/embeddings/reindex-all/${reindexJob.job_id}/cancel`,
        {
          method: "POST",
          headers: { Authorization: `Bearer ${accessToken}` },
        }
      );

      if (response.ok) {
        toast.info("Cancellation Requested", {
          description: "The job will stop after the current document.",
        });
        fetchReindexStatus();
      }
    } catch (error) {
      toast.error("Error", {
        description: "Failed to cancel job",
      });
    }
  };

  // Start KG extraction
  const startKGExtraction = async () => {
    if (!accessToken) return;
    setIsStartingKG(true);
    try {
      const response = await fetch(`${API_BASE_URL}/knowledge-graph/extraction-jobs`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          only_new_documents: kgOnlyNew,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        toast.success("KG Extraction Started", {
          description: data.message,
        });
        fetchKGStatus();
      } else {
        const error = await response.json();
        toast.error("Failed to start extraction", {
          description: error.detail || error.message || "Unknown error",
        });
      }
    } catch (error) {
      toast.error("Error", {
        description: "Failed to start KG extraction",
      });
    } finally {
      setIsStartingKG(false);
    }
  };

  // Cancel KG extraction
  const cancelKGExtraction = async () => {
    if (!kgJob?.job_id || !accessToken) return;

    try {
      await fetch(`${API_BASE_URL}/knowledge-graph/extraction-jobs/${kgJob.job_id}/cancel`, {
        method: "POST",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      toast.info("Cancellation Requested", {
        description: "The extraction will stop after the current document.",
      });
      fetchKGStatus();
    } catch (error) {
      toast.error("Error", {
        description: "Failed to cancel extraction",
      });
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "started":
      case "pending":
        return <Badge className="bg-blue-400">Starting...</Badge>;
      case "running":
        return <Badge className="bg-blue-500">Running</Badge>;
      case "completed":
        return <Badge className="bg-green-500">Completed</Badge>;
      case "completed_with_errors":
        return <Badge className="bg-orange-500">Completed with errors</Badge>;
      case "failed":
        return <Badge className="bg-red-500">Failed</Badge>;
      case "cancelled":
        return <Badge className="bg-yellow-500">Cancelled</Badge>;
      case "cancelling":
        return <Badge className="bg-yellow-500">Cancelling...</Badge>;
      case "paused":
        return <Badge className="bg-orange-500">Paused</Badge>;
      default:
        return <Badge>{status}</Badge>;
    }
  };

  return (
    <TabsContent value="maintenance" className="space-y-6">
      {/* Vector Database Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Vector Database Status
          </CardTitle>
          <CardDescription>
            Overview of embedding coverage and vector database health
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {isLoadingStats ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : embeddingStats ? (
            <>
              {/* Stats Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">Total Chunks</p>
                  <p className="text-2xl font-bold">{embeddingStats.total_chunks.toLocaleString()}</p>
                </div>
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">With Embeddings</p>
                  <p className="text-2xl font-bold text-green-600">{embeddingStats.chunks_with_embedding.toLocaleString()}</p>
                </div>
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">Missing/Null</p>
                  <p className="text-2xl font-bold text-red-600">{embeddingStats.chunks_with_null_embedding}</p>
                </div>
                <div className="p-3 rounded-lg bg-muted">
                  <p className="text-sm text-muted-foreground">ChromaDB Items</p>
                  <p className="text-2xl font-bold">{embeddingStats.chromadb_total_items?.toLocaleString() || "N/A"}</p>
                </div>
              </div>

              {/* Coverage Progress */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Embedding Coverage</span>
                  <span className={embeddingStats.embedding_coverage_percent === 100 ? "text-green-600 font-medium" : "text-yellow-600 font-medium"}>
                    {embeddingStats.embedding_coverage_percent}%
                  </span>
                </div>
                <Progress
                  value={embeddingStats.embedding_coverage_percent}
                  className={embeddingStats.embedding_coverage_percent === 100 ? "[&>div]:bg-green-500" : "[&>div]:bg-yellow-500"}
                />
              </div>

              {/* Additional Info */}
              <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                <span>Documents: {embeddingStats.total_documents}</span>
                <span>Dimension: {embeddingStats.embedding_dimension}</span>
                {embeddingStats.documents_with_issues > 0 && (
                  <span className="text-red-500">
                    {embeddingStats.documents_with_issues} docs with issues
                  </span>
                )}
              </div>

              {/* Problem Documents */}
              {embeddingStats.problem_documents.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-red-600 flex items-center gap-1">
                    <AlertCircle className="h-4 w-4" />
                    Documents with Missing Embeddings
                  </p>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {embeddingStats.problem_documents.map((doc, i) => (
                      <div key={i} className="text-sm flex justify-between px-2 py-1 bg-red-50 dark:bg-red-950 rounded">
                        <span className="truncate flex-1">{doc.filename}</span>
                        <span className="text-red-600 ml-2">{doc.problem_chunks} chunks</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Orphaned Chunks Warning */}
              {embeddingStats.orphaned_chunks > 0 && (
                <div className="space-y-3 p-3 rounded-lg bg-yellow-50 dark:bg-yellow-950 border border-yellow-200 dark:border-yellow-800">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-yellow-700 dark:text-yellow-300">
                      <AlertCircle className="h-5 w-5" />
                      <span className="font-medium">
                        {embeddingStats.orphaned_chunks.toLocaleString()} Orphaned Chunks
                      </span>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={cleanupOrphanedChunks}
                      disabled={isCleaningOrphans}
                      className="border-yellow-500 text-yellow-700 hover:bg-yellow-100 dark:hover:bg-yellow-900"
                    >
                      {isCleaningOrphans ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4 mr-2" />
                      )}
                      {isCleaningOrphans ? "Cleaning..." : "Clean Up"}
                    </Button>
                  </div>
                  <p className="text-sm text-yellow-600 dark:text-yellow-400">
                    These chunks reference deleted documents and should be cleaned up.
                  </p>
                </div>
              )}

              {/* ChromaDB Out of Sync Warning */}
              {embeddingStats.chromadb_total_items !== null && embeddingStats.chromadb_total_items < embeddingStats.chunks_with_embedding && (
                <div className="space-y-3 p-3 rounded-lg bg-orange-50 dark:bg-orange-950 border border-orange-200 dark:border-orange-800">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-orange-700 dark:text-orange-300">
                      <AlertCircle className="h-5 w-5" />
                      <span className="font-medium">
                        ChromaDB Out of Sync ({embeddingStats.chromadb_total_items.toLocaleString()} / {embeddingStats.chunks_with_embedding.toLocaleString()})
                      </span>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={syncChromaDB}
                      disabled={isSyncingChroma}
                      className="border-orange-500 text-orange-700 hover:bg-orange-100 dark:hover:bg-orange-900"
                    >
                      {isSyncingChroma ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Database className="h-4 w-4 mr-2" />
                      )}
                      {isSyncingChroma ? "Syncing..." : "Sync Now"}
                    </Button>
                  </div>
                  <p className="text-sm text-orange-600 dark:text-orange-400">
                    {embeddingStats.chunks_with_embedding - embeddingStats.chromadb_total_items} chunks have embeddings in the database but are missing from ChromaDB.
                    Sync will add them without re-generating embeddings.
                  </p>
                </div>
              )}

              {/* Success message if all good */}
              {embeddingStats.embedding_coverage_percent === 100 && embeddingStats.chunks_with_null_embedding === 0 && embeddingStats.orphaned_chunks === 0 && (embeddingStats.chromadb_total_items === null || embeddingStats.chromadb_total_items >= embeddingStats.chunks_with_embedding) && (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300">
                  <CheckCircle className="h-5 w-5" />
                  <span>All chunks have valid embeddings</span>
                </div>
              )}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Failed to load stats</p>
          )}

          <Button variant="outline" onClick={fetchEmbeddingStats} disabled={isLoadingStats}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoadingStats ? "animate-spin" : ""}`} />
            Refresh Stats
          </Button>
        </CardContent>
      </Card>

      {/* Embedding Re-index Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Re-index Embeddings
          </CardTitle>
          <CardDescription>
            Regenerate all document embeddings. Use this after changing embedding
            models or fixing embedding issues. Processes documents in small batches
            to avoid memory issues.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Configuration */}
          {(!reindexJob || reindexJob.status === "completed" || reindexJob.status === "failed" || reindexJob.status === "cancelled") && (
            <div className="space-y-4">
              {/* Processing Mode Selection */}
              <div className="space-y-2">
                <Label>Processing Mode</Label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="processing-mode"
                      checked={processingMode === "linear"}
                      onChange={() => setProcessingMode("linear")}
                      className="rounded-full"
                    />
                    <span>Linear (1 at a time)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="processing-mode"
                      checked={processingMode === "parallel"}
                      onChange={() => setProcessingMode("parallel")}
                      className="rounded-full"
                    />
                    <span>Parallel (faster, more memory)</span>
                  </label>
                </div>
                <p className="text-xs text-muted-foreground">
                  Linear is safest. Parallel uses more memory but is faster.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-4">
                {/* Parallel Count - only show if parallel mode */}
                {processingMode === "parallel" && (
                  <div className="space-y-2">
                    <Label htmlFor="parallel-count">Parallel Workers</Label>
                    <Input
                      id="parallel-count"
                      type="number"
                      min={2}
                      max={8}
                      value={parallelCount}
                      onChange={(e) => setParallelCount(parseInt(e.target.value) || 2)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Concurrent documents (2-8)
                    </p>
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="batch-size">Batch Size</Label>
                  <Input
                    id="batch-size"
                    type="number"
                    min={1}
                    max={20}
                    value={reindexBatchSize}
                    onChange={(e) => setReindexBatchSize(parseInt(e.target.value) || 5)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Docs before GC pause
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="delay">Delay (seconds)</Label>
                  <Input
                    id="delay"
                    type="number"
                    min={5}
                    max={120}
                    value={reindexDelay}
                    onChange={(e) => setReindexDelay(parseInt(e.target.value) || 15)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Pause between batches
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Progress display */}
          {reindexJob && reindexJob.status !== "completed" && reindexJob.status !== "failed" && reindexJob.status !== "cancelled" && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getStatusBadge(reindexJob.status)}
                  <span className="text-sm text-muted-foreground">
                    {reindexJob.processed} / {reindexJob.total_documents} documents
                  </span>
                </div>
                <span className="text-sm font-medium">{reindexJob.progress}%</span>
              </div>
              <Progress value={reindexJob.progress} />
              {reindexJob.current_document && (
                <p className="text-sm text-muted-foreground">
                  Current: {reindexJob.current_document}
                </p>
              )}
              {reindexJob.errors.length > 0 && (
                <div className="text-sm text-red-500">
                  {reindexJob.errors.length} errors (latest: {reindexJob.errors[reindexJob.errors.length - 1]})
                </div>
              )}
            </div>
          )}

          {/* Completed/Failed status */}
          {reindexJob && (reindexJob.status === "completed" || reindexJob.status === "failed" || reindexJob.status === "cancelled") && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-muted">
              {reindexJob.status === "completed" ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : (
                <AlertCircle className="h-5 w-5 text-red-500" />
              )}
              <span>{reindexJob.message}</span>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-2">
            {(!reindexJob || reindexJob.status === "completed" || reindexJob.status === "failed" || reindexJob.status === "cancelled") ? (
              <Button onClick={startReindex} disabled={isStartingReindex}>
                {isStartingReindex ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                Start Re-index
              </Button>
            ) : (
              <Button variant="destructive" onClick={cancelReindex}>
                <Square className="h-4 w-4 mr-2" />
                Cancel
              </Button>
            )}
            <Button variant="outline" onClick={fetchReindexStatus}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Knowledge Graph Extraction Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            Knowledge Graph Extraction
          </CardTitle>
          <CardDescription>
            Extract entities and relationships from documents to build the
            knowledge graph. This enhances search with semantic understanding.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Configuration */}
          {(!kgJob || kgJob.status === "completed" || kgJob.status === "completed_with_errors" || kgJob.status === "failed" || kgJob.status === "cancelled") && (
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="only-new"
                checked={kgOnlyNew}
                onChange={(e) => setKgOnlyNew(e.target.checked)}
                className="rounded border-gray-300"
              />
              <Label htmlFor="only-new">Only process new documents (skip already extracted)</Label>
            </div>
          )}

          {/* Progress display */}
          {kgJob && kgJob.status !== "completed" && kgJob.status !== "failed" && kgJob.status !== "cancelled" && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getStatusBadge(kgJob.status)}
                  <span className="text-sm text-muted-foreground">
                    {kgJob.processed_documents} / {kgJob.total_documents} documents
                    {kgJob.failed_documents > 0 && (
                      <span className="text-red-500 ml-1">({kgJob.failed_documents} failed)</span>
                    )}
                  </span>
                </div>
                <span className="text-sm font-medium">
                  {Math.round(kgJob.progress_percent || 0)}%
                </span>
              </div>
              <Progress value={kgJob.progress_percent || 0} />
              {kgJob.current_document && (
                <p className="text-sm text-muted-foreground">
                  Current: {kgJob.current_document}
                </p>
              )}
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Extracted: {kgJob.total_entities} entities, {kgJob.total_relations} relations</span>
                {kgJob.estimated_remaining_seconds && kgJob.estimated_remaining_seconds > 0 && (
                  <span>
                    ETA: {Math.round(kgJob.estimated_remaining_seconds / 60)} min
                  </span>
                )}
              </div>
              {kgJob.avg_doc_processing_time && (
                <p className="text-xs text-muted-foreground">
                  Avg: {kgJob.avg_doc_processing_time.toFixed(1)}s per document
                </p>
              )}
            </div>
          )}

          {/* Completed/Failed status */}
          {kgJob && (kgJob.status === "completed" || kgJob.status === "completed_with_errors" || kgJob.status === "failed" || kgJob.status === "cancelled") && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-muted">
              {kgJob.status === "completed" ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : kgJob.status === "completed_with_errors" ? (
                <AlertCircle className="h-5 w-5 text-orange-500" />
              ) : (
                <AlertCircle className="h-5 w-5 text-red-500" />
              )}
              <span>
                {kgJob.status === "completed"
                  ? `Completed: ${kgJob.total_entities} entities, ${kgJob.total_relations} relations extracted from ${kgJob.processed_documents} docs`
                  : kgJob.status === "completed_with_errors"
                  ? `Completed with errors: ${kgJob.total_entities} entities, ${kgJob.total_relations} relations from ${kgJob.processed_documents} docs (${kgJob.failed_documents} failed)`
                  : kgJob.status === "cancelled"
                  ? `Cancelled after ${kgJob.processed_documents} docs`
                  : `Failed: ${kgJob.error_count} errors`}
              </span>
            </div>
          )}

          {/* Collapsible per-document details */}
          {kgJob && kgJob.total_documents > 0 && (
            <Collapsible open={isDocDetailsExpanded} onOpenChange={setIsDocDetailsExpanded}>
              <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors w-full py-1">
                {isDocDetailsExpanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
                <span>View document details ({kgJob.total_documents} documents)</span>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="mt-2 max-h-64 overflow-y-auto rounded-lg border">
                  {kgDocDetails === null ? (
                    <div className="flex items-center justify-center py-4 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Loading...
                    </div>
                  ) : kgDocDetails.length === 0 ? (
                    <div className="py-4 text-center text-sm text-muted-foreground">
                      No documents found
                    </div>
                  ) : (
                    <div className="divide-y">
                      {kgDocDetails.map((doc) => (
                        <div
                          key={doc.document_id}
                          className="flex items-center gap-3 px-3 py-2 text-sm"
                        >
                          {/* Status icon */}
                          {doc.status === "processing" ? (
                            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-500" />
                          ) : doc.status === "completed" ? (
                            <CheckCircle className="h-4 w-4 shrink-0 text-green-500" />
                          ) : doc.status === "failed" ? (
                            <XCircle className="h-4 w-4 shrink-0 text-red-500" />
                          ) : (
                            <Clock className="h-4 w-4 shrink-0 text-muted-foreground" />
                          )}

                          {/* Filename */}
                          <span className="truncate min-w-0 flex-1" title={doc.filename}>
                            {doc.filename}
                          </span>

                          {/* Progress / counts */}
                          <span className="shrink-0 text-xs text-muted-foreground">
                            {doc.status === "processing" && doc.chunks_processed !== null
                              ? `Chunk ${doc.chunks_processed}/${doc.chunk_count}`
                              : doc.status === "completed"
                              ? `${doc.kg_entity_count}E / ${doc.kg_relation_count}R`
                              : doc.status === "failed"
                              ? "Failed"
                              : doc.chunk_count > 0
                              ? `${doc.chunk_count} chunks`
                              : "Pending"}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}

          {/* Action buttons */}
          <div className="flex gap-2">
            {(!kgJob || kgJob.status === "completed" || kgJob.status === "completed_with_errors" || kgJob.status === "failed" || kgJob.status === "cancelled") ? (
              <Button onClick={startKGExtraction} disabled={isStartingKG}>
                {isStartingKG ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                Start Extraction
              </Button>
            ) : (
              <Button variant="destructive" onClick={cancelKGExtraction}>
                <Square className="h-4 w-4 mr-2" />
                Cancel
              </Button>
            )}
            <Button variant="outline" onClick={fetchKGStatus}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
