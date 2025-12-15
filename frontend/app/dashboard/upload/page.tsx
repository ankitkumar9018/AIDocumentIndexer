"use client";

import { useState, useCallback } from "react";
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
import {
  useUploadFile,
  useUploadBatch,
  useProcessingStatus,
  useProcessingQueue,
  useCancelProcessing,
  useRetryProcessing,
  useSupportedFileTypes,
  ProcessingStatus,
} from "@/lib/api";

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
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [collection, setCollection] = useState("");
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  // Queries
  const { data: queue, isLoading: queueLoading, refetch: refetchQueue } = useProcessingQueue();
  const { data: supportedTypes } = useSupportedFileTypes();

  // Mutations
  const uploadFile = useUploadFile();
  const uploadBatch = useUploadBatch();
  const cancelProcessing = useCancelProcessing();
  const retryProcessing = useRetryProcessing();

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
    if (selectedFiles.length === 0) return;

    try {
      if (selectedFiles.length === 1) {
        await uploadFile.mutateAsync({
          file: selectedFiles[0],
          options: collection ? { collection } : undefined,
        });
      } else {
        await uploadBatch.mutateAsync({
          files: selectedFiles,
          options: collection ? { collection } : undefined,
        });
      }
      setSelectedFiles([]);
      refetchQueue();
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await cancelProcessing.mutateAsync(id);
      refetchQueue();
    } catch (error) {
      console.error("Cancel failed:", error);
    }
  };

  const handleRetry = async (id: string) => {
    try {
      await retryProcessing.mutateAsync(id);
      refetchQueue();
    } catch (error) {
      console.error("Retry failed:", error);
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

  const isUploading = uploadFile.isPending || uploadBatch.isPending;

  // Use real queue data from API (no mock fallback)
  const queueItems: ProcessingStatus[] = queue?.items ?? [];

  const stats = {
    total: queueItems.length,
    processing: queueItems.filter((i) => i.status === "processing").length,
    queued: queueItems.filter((i) => i.status === "queued").length,
    completed: queueItems.filter((i) => i.status === "completed").length,
    failed: queueItems.filter((i) => i.status === "failed").length,
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
        <Button onClick={() => refetchQueue()} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
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
                  accept={supportedTypes?.supported_extensions?.join(",") || "*"}
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
                <ScrollArea className="max-h-48">
                  <div className="space-y-2">
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
                </ScrollArea>
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
