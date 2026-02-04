"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useSession } from "next-auth/react";
import { toast } from "sonner";
import { getErrorMessage } from "@/lib/errors";
import { useWebSocket, FileUpdateMessage, WebSocketMessage } from "@/lib/websocket";
import {
  Upload,
  FileText,
  Image,
  FileSpreadsheet,
  Presentation,
  File,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  RefreshCw,
  Trash2,
  Play,
  Pause,
  RotateCcw,
  X,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  FolderOpen,
  Settings,
  Zap,
  Eye,
  FileSearch,
  Wifi,
  WifiOff,
  Sparkles,
  Shield,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  useUploadFile,
  useUploadBatch,
  useProcessingStatus,
  useProcessingQueue,
  useCancelProcessing,
  useRetryProcessing,
  useSupportedFileTypes,
  useAccessTiers,
  useVisionStatus,
  ProcessingStatus,
  api,
} from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { FolderSelector } from "@/components/folder-selector";

const getFileIconFromName = (filename: string) => {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'tiff', 'bmp'].includes(ext)) return Image;
  if (ext === 'pdf') return FileText;
  if (['xlsx', 'xls', 'csv'].includes(ext)) return FileSpreadsheet;
  if (['pptx', 'ppt'].includes(ext)) return Presentation;
  return File;
};

const getFileIconFromType = (fileType: string) => {
  if (fileType.includes("image")) return Image;
  if (fileType.includes("pdf")) return FileText;
  if (fileType.includes("spreadsheet") || fileType.includes("excel") || fileType.includes("csv"))
    return FileSpreadsheet;
  if (fileType.includes("presentation") || fileType.includes("powerpoint"))
    return Presentation;
  return File;
};

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const getStatusColor = (status: string) => {
  switch (status) {
    case "completed":
      return "text-green-500";
    case "failed":
      return "text-red-500";
    case "processing":
      return "text-blue-500";
    default:
      return "text-yellow-500";
  }
};

export default function UploadPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [collection, setCollection] = useState("");
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  // Track file IDs being processed for WebSocket notifications
  const processingFileIds = useRef<Set<string>>(new Set());
  const notifiedCompletions = useRef<Set<string>>(new Set());

  // WebSocket for real-time processing updates
  const { subscribe, unsubscribe, isConnected, onMessage } = useWebSocket();

  // Processing options state
  const [processingOptions, setProcessingOptions] = useState({
    enable_ocr: true,
    enable_image_analysis: true,
    smart_image_handling: true,
    smart_chunking: true,
    detect_duplicates: true,
    auto_generate_tags: false,
    auto_enhance: false,
    is_private: false,
    processing_mode: "full" as "full" | "ocr" | "basic",
    access_tier: undefined as number | undefined,
  });

  // Queries - only fetch when authenticated
  const { data: queue, isLoading: queueLoading, refetch: refetchQueue } = useProcessingQueue({ enabled: isAuthenticated });
  const { data: supportedTypes } = useSupportedFileTypes();
  const { data: tiersData } = useAccessTiers({ enabled: isAuthenticated });
  const tiers = tiersData?.tiers ?? [];
  const { data: visionStatus } = useVisionStatus({ enabled: isAuthenticated });

  // Mutations
  const uploadFile = useUploadFile();
  const uploadBatch = useUploadBatch();
  const cancelProcessing = useCancelProcessing();
  const retryProcessing = useRetryProcessing();

  // WebSocket message handler for processing updates
  useEffect(() => {
    if (!isConnected) return;

    // Define message handler
    const handleMessage = (message: WebSocketMessage) => {
      // Only handle file update messages
      if (message.type !== "file_update") return;

      const fileMessage = message as FileUpdateMessage;
      const fileId = fileMessage.file_id;
      const filename = fileMessage.filename || "Document";

      // Only show notifications for files we're tracking
      if (!processingFileIds.current.has(fileId)) return;

      if (fileMessage.status === "completed" && !notifiedCompletions.current.has(fileId)) {
        notifiedCompletions.current.add(fileId);
        processingFileIds.current.delete(fileId);
        toast.success(`"${filename}" processed successfully`, {
          description: `${fileMessage.chunk_count || 0} chunks created`,
        });
        refetchQueue();
      } else if (fileMessage.status === "failed" && !notifiedCompletions.current.has(fileId)) {
        notifiedCompletions.current.add(fileId);
        processingFileIds.current.delete(fileId);
        toast.error(`"${filename}" processing failed`, {
          description: fileMessage.error || "An error occurred during processing",
        });
        refetchQueue();
      } else if (fileMessage.status === "processing" && fileMessage.progress === 0) {
        // Show toast when processing starts (first update)
        toast.info(`Processing "${filename}"...`, {
          description: fileMessage.current_step || "Starting...",
        });
      }
    };

    // Subscribe to WebSocket messages
    const unsubscribeMsg = onMessage(handleMessage);

    return () => {
      unsubscribeMsg();
    };
  }, [isConnected, onMessage, refetchQueue]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    setSelectedFiles((prev) => [...prev, ...files]);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles((prev) => [...prev, ...files]);
    }
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    console.log('[Upload] handleUpload called, selectedFiles:', selectedFiles.length);
    if (selectedFiles.length === 0) {
      console.log('[Upload] No files selected, returning early');
      return;
    }

    // Build upload options including processing settings
    const uploadOptions = {
      ...(collection && { collection }),
      ...(selectedFolderId && { folder_id: selectedFolderId }),
      ...processingOptions,
    };
    console.log('[Upload] Upload options:', uploadOptions);

    try {
      if (selectedFiles.length === 1) {
        console.log('[Upload] Uploading single file:', selectedFiles[0].name);
        const result = await uploadFile.mutateAsync({
          file: selectedFiles[0],
          options: Object.keys(uploadOptions).length > 0 ? uploadOptions : undefined,
        });
        console.log('[Upload] Single file upload result:', result);
        // Check if file was a duplicate
        if (result?.status === 'duplicate') {
          toast.info("Duplicate file detected", {
            description: "This file already exists in the system. Upload skipped.",
          });
        } else {
          // Track file ID for WebSocket notifications
          if (result?.file_id) {
            const fileId = String(result.file_id);
            processingFileIds.current.add(fileId);
            subscribe(fileId);
          }
          toast.success("File uploaded successfully", {
            description: "Processing will begin shortly.",
          });
        }
      } else {
        const result = await uploadBatch.mutateAsync({
          files: selectedFiles,
          options: Object.keys(uploadOptions).length > 0 ? uploadOptions : undefined,
        });
        // Track all file IDs for WebSocket notifications
        result.files?.forEach((file: any) => {
          if (file?.file_id && file.status !== 'failed') {
            const fileId = String(file.file_id);
            processingFileIds.current.add(fileId);
            subscribe(fileId);
          }
        });
        toast.success(`${result.successful} files uploaded successfully`, {
          description: result.failed > 0 ? `${result.failed} files failed.` : "Processing will begin shortly.",
        });
      }
      setSelectedFiles([]);
      refetchQueue();
    } catch (error) {
      console.error("Upload failed:", error);
      const errorMessage = getErrorMessage(error, "Upload failed");
      // Check if it's an auth error by looking for 401 in the message or specific error patterns
      if (errorMessage.toLowerCase().includes("unauthorized") || errorMessage.includes("401")) {
        toast.error("Authentication error", {
          description: "Your session may have expired. Please sign out and sign in again.",
        });
      } else {
        toast.error("Upload failed", {
          description: errorMessage,
        });
      }
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await cancelProcessing.mutateAsync(id);
      toast.success("Processing cancelled");
      refetchQueue();
    } catch (error) {
      console.error("Cancel failed:", error);
      toast.error("Failed to cancel processing", {
        description: getErrorMessage(error),
      });
    }
  };

  const handleRetry = async (id: string) => {
    try {
      await retryProcessing.mutateAsync(id);
      toast.success("Retrying processing");
      refetchQueue();
    } catch (error) {
      console.error("Retry failed:", error);
      toast.error("Failed to retry processing", {
        description: getErrorMessage(error),
      });
    }
  };

  const toggleExpanded = (id: string) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const [isClearingCompleted, setIsClearingCompleted] = useState(false);

  const handleClearCompleted = async () => {
    try {
      setIsClearingCompleted(true);
      const result = await api.clearCompletedUploads();
      toast.success("Completed uploads cleared", {
        description: `${result.deleted_count} completed items removed from queue`,
      });
      refetchQueue();
    } catch (error) {
      console.error("Failed to clear completed uploads:", error);
      toast.error("Failed to clear completed uploads", {
        description: getErrorMessage(error),
      });
    } finally {
      setIsClearingCompleted(false);
    }
  };

  const isUploading = uploadFile.isPending || uploadBatch.isPending;

  // Use real queue data from API (no mock fallback)
  const queueItems: ProcessingStatus[] = queue?.items ?? [];

  // Active processing statuses from backend UploadStatus enum
  // These are the intermediate statuses that indicate active processing
  const ACTIVE_PROCESSING_STATUSES = [
    "validating",
    "extracting",
    "chunking",
    "embedding",
    "indexing"
  ];

  const stats = {
    total: queueItems.length,
    processing: queueItems.filter((i) => ACTIVE_PROCESSING_STATUSES.includes(i.status.toLowerCase())).length,
    queued: queueItems.filter((i) => i.status.toLowerCase() === "queued").length,
    completed: queueItems.filter((i) => i.status.toLowerCase() === "completed").length,
    failed: queueItems.filter((i) => ["failed", "cancelled"].includes(i.status.toLowerCase())).length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Upload & Processing</h1>
          <p className="text-muted-foreground">
            Upload documents and monitor the processing queue
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* WebSocket connection status */}
          <div className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs ${
            isConnected
              ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
              : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
          }`}>
            {isConnected ? (
              <>
                <Wifi className="h-3 w-3" />
                <span>Live</span>
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3" />
                <span>Connecting...</span>
              </>
            )}
          </div>
          <Button onClick={() => refetchQueue()} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          {stats.completed > 0 && (
            <Button
              onClick={handleClearCompleted}
              variant="outline"
              size="sm"
              disabled={isClearingCompleted}
            >
              {isClearingCompleted ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4 mr-2" />
              )}
              Clear Completed ({stats.completed})
            </Button>
          )}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Processing</p>
                <p className="text-2xl font-bold">{stats.processing}</p>
              </div>
              <Loader2 className={`h-8 w-8 text-blue-500 ${stats.processing > 0 ? 'animate-spin' : ''}`} />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Queued</p>
                <p className="text-2xl font-bold">{stats.queued}</p>
              </div>
              <Clock className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold">{stats.completed}</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Failed</p>
                <p className="text-2xl font-bold">{stats.failed}</p>
              </div>
              <XCircle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Upload Area */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Documents
            </CardTitle>
            <CardDescription>
              Drag and drop files or click to browse
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Drop Zone */}
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging
                  ? "border-primary bg-primary/5"
                  : "border-muted-foreground/25 hover:border-primary/50"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-sm text-muted-foreground mb-2">
                Drag and drop files here, or
              </p>
              <label>
                <Button variant="secondary" size="sm" asChild>
                  <span>
                    <FolderOpen className="h-4 w-4 mr-2" />
                    Browse Files
                  </span>
                </Button>
                <input
                  type="file"
                  multiple
                  className="hidden"
                  onChange={handleFileSelect}
                  accept={supportedTypes?.supported_extensions?.map((ext: string) => `.${ext}`).join(",") || "*"}
                />
              </label>
              <p className="text-xs text-muted-foreground mt-4">
                Supports PDF, DOCX, PPTX, XLSX, images, and more
              </p>
            </div>

            {/* Collection Input */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Collection (optional)</label>
              <Input
                placeholder="Enter collection name..."
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
              />
            </div>

            {/* Auto-generate Tags - Visible by default since it's commonly used */}
            <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/30">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <div>
                  <Label className="text-sm font-medium">Auto-generate Tags</Label>
                  <p className="text-xs text-muted-foreground">
                    Use AI to automatically generate relevant tags based on document content
                  </p>
                </div>
              </div>
              <Switch
                checked={processingOptions.auto_generate_tags}
                onCheckedChange={(checked) =>
                  setProcessingOptions((prev) => ({ ...prev, auto_generate_tags: checked }))
                }
              />
            </div>

            {/* Folder Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <FolderOpen className="h-4 w-4 text-muted-foreground" />
                Upload to Folder (optional)
              </label>
              <FolderSelector
                value={selectedFolderId}
                onChange={setSelectedFolderId}
                placeholder="Root (no folder)"
              />
              <p className="text-xs text-muted-foreground">
                Leave empty to upload to root level
              </p>
            </div>

            {/* Access Tier Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <Shield className="h-4 w-4 text-muted-foreground" />
                Access Tier
              </label>
              <select
                className="w-full h-10 px-3 rounded-md border bg-background text-sm"
                value={processingOptions.access_tier ?? ""}
                onChange={(e) =>
                  setProcessingOptions((prev) => ({
                    ...prev,
                    access_tier: e.target.value ? parseInt(e.target.value) : undefined,
                  }))
                }
              >
                <option value="">Default (lowest tier)</option>
                {tiers
                  .sort((a: { level: number }, b: { level: number }) => a.level - b.level)
                  .map((tier: { id: string; name: string; level: number }) => (
                    <option key={tier.id} value={tier.level}>
                      {tier.name} (Level {tier.level})
                    </option>
                  ))}
              </select>
              <p className="text-xs text-muted-foreground">
                Set the minimum access level required to view this document
              </p>
            </div>

            {/* Processing Options */}
            <Collapsible open={showAdvancedOptions} onOpenChange={setShowAdvancedOptions}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between p-2 h-auto">
                  <div className="flex items-center gap-2">
                    <Settings className="h-4 w-4" />
                    <span className="text-sm font-medium">Processing Options</span>
                  </div>
                  {showAdvancedOptions ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-4 pt-4">
                {/* Processing Mode */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Processing Mode</Label>
                  <select
                    className="w-full h-10 px-3 rounded-md border bg-background text-sm"
                    value={processingOptions.processing_mode}
                    onChange={(e) =>
                      setProcessingOptions((prev) => ({
                        ...prev,
                        processing_mode: e.target.value as "full" | "ocr" | "basic",
                      }))
                    }
                  >
                    <option value="full">Full (Recommended)</option>
                    <option value="ocr">OCR Enabled</option>
                    <option value="basic">Basic (Fastest)</option>
                  </select>
                  <p className="text-xs text-muted-foreground">
                    Full mode includes text extraction, OCR, and AI image analysis
                  </p>
                </div>

                {/* Toggle Options */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Eye className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <Label className="text-sm">OCR Processing</Label>
                        <p className="text-xs text-muted-foreground">
                          Extract text from images & scanned PDFs
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={processingOptions.enable_ocr}
                      onCheckedChange={(checked) =>
                        setProcessingOptions((prev) => ({ ...prev, enable_ocr: checked }))
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <Label className="text-sm">Smart Image Handling</Label>
                        <p className="text-xs text-muted-foreground">
                          Optimize images for faster processing
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={processingOptions.smart_image_handling}
                      onCheckedChange={(checked) =>
                        setProcessingOptions((prev) => ({ ...prev, smart_image_handling: checked }))
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <FileSearch className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <Label className="text-sm">Duplicate Detection</Label>
                        <p className="text-xs text-muted-foreground">
                          Skip files already in the system
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={processingOptions.detect_duplicates}
                      onCheckedChange={(checked) =>
                        setProcessingOptions((prev) => ({ ...prev, detect_duplicates: checked }))
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Image className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <Label className="text-sm">AI Image Analysis</Label>
                        <p className="text-xs text-muted-foreground">
                          Describe images for searchability
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={processingOptions.enable_image_analysis}
                      onCheckedChange={(checked) =>
                        setProcessingOptions((prev) => ({ ...prev, enable_image_analysis: checked }))
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50 border border-purple-200 dark:border-purple-800">
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-purple-500" />
                      <div>
                        <Label className="text-sm">AI Enhancement</Label>
                        <p className="text-xs text-muted-foreground">
                          Auto-generate summaries, keywords &amp; questions
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={processingOptions.auto_enhance}
                      onCheckedChange={(checked) =>
                        setProcessingOptions((prev) => ({ ...prev, auto_enhance: checked }))
                      }
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50 border border-orange-200 dark:border-orange-800">
                    <div className="flex items-center gap-2">
                      <Shield className="h-4 w-4 text-orange-500" />
                      <div>
                        <Label className="text-sm">Private Document</Label>
                        <p className="text-xs text-muted-foreground">
                          Only visible to you and admins
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={processingOptions.is_private}
                      onCheckedChange={(checked) =>
                        setProcessingOptions((prev) => ({ ...prev, is_private: checked }))
                      }
                    />
                  </div>

                  {/* Private Document Info */}
                  {processingOptions.is_private && (
                    <Alert variant="default" className="border-orange-200 bg-orange-50 dark:bg-orange-950/30">
                      <Shield className="h-4 w-4 text-orange-500" />
                      <AlertTitle className="text-orange-700 dark:text-orange-300">Private Document</AlertTitle>
                      <AlertDescription className="text-orange-600 dark:text-orange-400 text-xs">
                        This document will only be visible to you and superadmins. Other users won&apos;t see it in searches or document lists.
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* Vision Model Warning */}
                  {processingOptions.enable_image_analysis && !visionStatus?.available && (
                    <Alert variant="default" className="border-orange-200 bg-orange-50 dark:bg-orange-950/30">
                      <AlertCircle className="h-4 w-4 text-orange-500" />
                      <AlertTitle className="text-orange-700 dark:text-orange-300">Vision Model Not Configured</AlertTitle>
                      <AlertDescription className="text-orange-600 dark:text-orange-400 text-xs">
                        Images will be extracted but not analyzed. {visionStatus?.recommendation}
                        <Button
                          variant="link"
                          size="sm"
                          className="p-0 h-auto text-orange-700 dark:text-orange-300 ml-1"
                          onClick={() => window.location.href = '/dashboard/admin/settings'}
                        >
                          Configure in Settings
                        </Button>
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>

            {/* Selected Files */}
            {selectedFiles.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">
                    Selected Files ({selectedFiles.length})
                  </label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedFiles([])}
                  >
                    Clear All
                  </Button>
                </div>
                <div className="max-h-48 overflow-y-auto rounded-md border border-border">
                  <div className="space-y-2 p-2">
                    {selectedFiles.map((file, index) => {
                      const FileIcon = getFileIconFromType(file.type);
                      return (
                        <div
                          key={index}
                          className="flex items-center justify-between p-2 rounded-lg bg-muted"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <FileIcon className="h-4 w-4 text-muted-foreground shrink-0" />
                            <span className="text-sm truncate">{file.name}</span>
                            <span className="text-xs text-muted-foreground shrink-0">
                              ({formatFileSize(file.size)})
                            </span>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 shrink-0"
                            onClick={() => handleRemoveFile(index)}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Upload Button */}
            <Button
              onClick={handleUpload}
              disabled={selectedFiles.length === 0 || isUploading}
              className="w-full"
            >
              {isUploading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Upload className="h-4 w-4 mr-2" />
              )}
              Upload {selectedFiles.length > 0 && `(${selectedFiles.length} files)`}
            </Button>
          </CardContent>
        </Card>

        {/* Processing Queue */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Processing Queue
            </CardTitle>
            <CardDescription>
              Real-time status of document processing
            </CardDescription>
          </CardHeader>
          <CardContent>
            {queueLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : queueItems.length > 0 ? (
              <ScrollArea className="h-[400px]">
                <div className="space-y-3">
                  {queueItems.map((item) => {
                    const FileIcon = getFileIconFromName(item.filename);
                    const isExpanded = expandedItems.has(item.file_id);

                    return (
                      <div
                        key={item.file_id}
                        className="border rounded-lg overflow-hidden"
                      >
                        {/* Main Row */}
                        <div
                          className="flex items-center gap-3 p-3 cursor-pointer hover:bg-muted/50"
                          onClick={() => toggleExpanded(item.file_id)}
                        >
                          <FileIcon className="h-5 w-5 text-muted-foreground shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">
                              {item.filename}
                            </p>
                            <div className="flex items-center gap-2 mt-1">
                              <span
                                className={`text-xs capitalize ${getStatusColor(
                                  item.status
                                )}`}
                              >
                                {item.status === "processing" ? (
                                  <span className="flex items-center gap-1">
                                    <Loader2 className="h-3 w-3 animate-spin" />
                                    Processing
                                  </span>
                                ) : (
                                  item.status
                                )}
                              </span>
                              {item.current_step && (
                                <span className="text-xs text-muted-foreground">
                                  {item.current_step}
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Progress or Status Icon */}
                          <div className="flex items-center gap-2">
                            {item.status === "processing" && (
                              <div className="w-20">
                                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-blue-500 rounded-full transition-all"
                                    style={{ width: `${item.progress}%` }}
                                  />
                                </div>
                                <p className="text-xs text-muted-foreground text-center mt-0.5">
                                  {item.progress}%
                                </p>
                              </div>
                            )}
                            {item.status === "completed" && (
                              <CheckCircle className="h-5 w-5 text-green-500" />
                            )}
                            {item.status === "failed" && (
                              <XCircle className="h-5 w-5 text-red-500" />
                            )}
                            {item.status === "queued" && (
                              <Clock className="h-5 w-5 text-yellow-500" />
                            )}
                            {isExpanded ? (
                              <ChevronUp className="h-4 w-4 text-muted-foreground" />
                            ) : (
                              <ChevronDown className="h-4 w-4 text-muted-foreground" />
                            )}
                          </div>
                        </div>

                        {/* Expanded Details */}
                        {isExpanded && (
                          <div className="px-3 pb-3 pt-0 border-t bg-muted/30">
                            <div className="grid grid-cols-2 gap-2 text-xs mt-2">
                              <div>
                                <span className="text-muted-foreground">Started:</span>
                                <span className="ml-1">
                                  {new Date(item.created_at).toLocaleString()}
                                </span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Updated:</span>
                                <span className="ml-1">
                                  {new Date(item.updated_at).toLocaleString()}
                                </span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Progress:</span>
                                <span className="ml-1">{item.progress}%</span>
                              </div>
                            </div>

                            {item.error && (
                              <div className="mt-2 p-2 rounded bg-red-500/10 text-red-600 text-xs flex items-start gap-2">
                                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                                {item.error}
                              </div>
                            )}

                            <div className="flex gap-2 mt-3">
                              {item.status === "failed" && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleRetry(item.file_id);
                                  }}
                                >
                                  <RotateCcw className="h-3 w-3 mr-1" />
                                  Retry
                                </Button>
                              )}
                              {(item.status === "processing" ||
                                item.status === "queued") && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleCancel(item.file_id);
                                  }}
                                >
                                  <X className="h-3 w-3 mr-1" />
                                  Cancel
                                </Button>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </ScrollArea>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Clock className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No items in queue</p>
                <p className="text-sm">Upload files to start processing</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Supported File Types */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Supported File Types</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {[
              "PDF",
              "DOCX",
              "DOC",
              "PPTX",
              "PPT",
              "XLSX",
              "XLS",
              "CSV",
              "TXT",
              "MD",
              "HTML",
              "JSON",
              "XML",
              "PNG",
              "JPG",
              "JPEG",
              "GIF",
              "WEBP",
              "TIFF",
              "EML",
              "MSG",
            ].map((type) => (
              <span
                key={type}
                className="px-2 py-1 text-xs bg-muted rounded-md font-mono"
              >
                .{type.toLowerCase()}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
