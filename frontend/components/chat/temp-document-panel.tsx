"use client";

import { useState, useRef, useCallback } from "react";
import {
  FileText,
  Upload,
  X,
  Trash2,
  Save,
  Loader2,
  FileType,
  AlertCircle,
  CheckCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface TempDocument {
  id: string;
  filename: string;
  token_count: number;
  file_size: number;
  file_type: string;
  has_chunks: boolean;
  has_embeddings: boolean;
}

interface TempDocumentPanelProps {
  sessionId: string | null;
  documents: TempDocument[];
  totalTokens: number;
  isLoading?: boolean;
  onUpload: (file: File) => Promise<void>;
  onRemove: (docId: string) => Promise<void>;
  onSave: (docId: string, collection?: string) => Promise<void>;
  onDiscard: () => Promise<void>;
  className?: string;
}

const MAX_CONTEXT_TOKENS = 100000;

export function TempDocumentPanel({
  sessionId,
  documents,
  totalTokens,
  isLoading = false,
  onUpload,
  onRemove,
  onSave,
  onDiscard,
  className,
}: TempDocumentPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [savingDocs, setSavingDocs] = useState<Set<string>>(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      if (files.length === 0) return;

      setIsUploading(true);
      try {
        for (const file of files) {
          await onUpload(file);
        }
        toast.success(`Uploaded ${files.length} file(s)`);
      } catch (error) {
        toast.error("Failed to upload files");
      } finally {
        setIsUploading(false);
      }
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;

      setIsUploading(true);
      try {
        for (const file of Array.from(files)) {
          await onUpload(file);
        }
        toast.success(`Uploaded ${files.length} file(s)`);
      } catch (error) {
        toast.error("Failed to upload files");
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [onUpload]
  );

  const handleSave = useCallback(
    async (docId: string) => {
      setSavingDocs((prev) => new Set(prev).add(docId));
      try {
        await onSave(docId);
        toast.success("Document saved to library");
      } catch (error) {
        toast.error("Failed to save document");
      } finally {
        setSavingDocs((prev) => {
          const next = new Set(prev);
          next.delete(docId);
          return next;
        });
      }
    },
    [onSave]
  );

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatTokenCount = (tokens: number) => {
    if (tokens < 1000) return tokens.toString();
    return `${(tokens / 1000).toFixed(1)}K`;
  };

  const getFileTypeIcon = (type: string) => {
    return <FileType className="h-4 w-4" />;
  };

  const contextUsagePercent = Math.min(
    100,
    (totalTokens / MAX_CONTEXT_TOKENS) * 100
  );

  // Always render when shown - the upload zone should be visible even without a session
  return (
    <Card className={cn("border-dashed", className)}>
      <CardHeader className="py-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Quick Chat Documents
            {documents.length > 0 && (
              <Badge variant="secondary" className="text-xs">
                {documents.length}
              </Badge>
            )}
          </CardTitle>
          {documents.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onDiscard}
              className="h-7 text-xs text-destructive hover:text-destructive"
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear All
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="pt-0 space-y-3">
        {/* Context Usage Bar */}
        {documents.length > 0 && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Context Usage</span>
              <span>
                {formatTokenCount(totalTokens)} / {formatTokenCount(MAX_CONTEXT_TOKENS)} tokens
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full transition-all",
                  contextUsagePercent > 90
                    ? "bg-destructive"
                    : contextUsagePercent > 70
                    ? "bg-yellow-500"
                    : "bg-primary"
                )}
                style={{ width: `${contextUsagePercent}%` }}
              />
            </div>
            {contextUsagePercent > 90 && (
              <p className="text-xs text-destructive flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                Documents will be chunked for queries
              </p>
            )}
          </div>
        )}

        {/* Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "border-2 border-dashed rounded-lg p-4 text-center transition-colors cursor-pointer",
            isDragging
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-muted-foreground/50"
          )}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={handleFileSelect}
            accept=".pdf,.doc,.docx,.txt,.md,.html,.csv,.xlsx,.xls,.pptx,.ppt"
          />
          {isUploading || isLoading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Processing...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="h-6 w-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                Drop files here or click to upload
              </p>
              <p className="text-xs text-muted-foreground/70">
                PDF, Word, Excel, PowerPoint, Text, Markdown
              </p>
            </div>
          )}
        </div>

        {/* Document List */}
        {documents.length > 0 && (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="border rounded-lg p-2 bg-muted/30"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {getFileTypeIcon(doc.file_type)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {doc.filename}
                      </p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{formatFileSize(doc.file_size)}</span>
                        <span>|</span>
                        <span>{formatTokenCount(doc.token_count)} tokens</span>
                        {doc.has_chunks && (
                          <>
                            <span>|</span>
                            <Badge variant="outline" className="text-xs h-4">
                              Chunked
                            </Badge>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSave(doc.id)}
                      disabled={savingDocs.has(doc.id)}
                      className="h-7 w-7 p-0"
                      title="Save to library"
                    >
                      {savingDocs.has(doc.id) ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Save className="h-3 w-3" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onRemove(doc.id)}
                      className="h-7 w-7 p-0 text-destructive hover:text-destructive"
                      title="Remove"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Info */}
        {documents.length === 0 && (
          <p className="text-xs text-muted-foreground text-center">
            Upload documents to chat with them instantly. Save to library later if needed.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
