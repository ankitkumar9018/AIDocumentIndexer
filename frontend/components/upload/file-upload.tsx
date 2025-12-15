"use client";

import * as React from "react";
import {
  Upload,
  X,
  FileText,
  Image,
  FileSpreadsheet,
  Presentation,
  File,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";

import { cn, formatBytes, getFileExtension } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

export interface UploadFile {
  id: string;
  file: File;
  status: "pending" | "uploading" | "processing" | "completed" | "error";
  progress: number;
  error?: string;
}

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  onRemoveFile: (fileId: string) => void;
  files: UploadFile[];
  maxFiles?: number;
  maxSize?: number; // in bytes
  acceptedTypes?: string[];
  className?: string;
}

const DEFAULT_MAX_SIZE = 100 * 1024 * 1024; // 100MB
const DEFAULT_ACCEPTED_TYPES = [
  // Documents
  ".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt",
  // Presentations
  ".ppt", ".pptx", ".odp", ".key",
  // Spreadsheets
  ".xls", ".xlsx", ".csv", ".ods",
  // Images
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg",
  // Archives
  ".zip", ".rar", ".7z", ".tar", ".gz",
  // Audio/Video
  ".mp3", ".wav", ".mp4", ".mov", ".avi", ".mkv",
  // Other
  ".json", ".xml", ".html", ".htm",
];

export function FileUpload({
  onFilesSelected,
  onRemoveFile,
  files,
  maxFiles = 50,
  maxSize = DEFAULT_MAX_SIZE,
  acceptedTypes = DEFAULT_ACCEPTED_TYPES,
  className,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      handleFiles(selectedFiles);
    }
    // Reset input
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const handleFiles = (newFiles: File[]) => {
    // Filter out files that exceed max size
    const validFiles = newFiles.filter((file) => {
      if (file.size > maxSize) {
        console.warn(`File ${file.name} exceeds max size of ${formatBytes(maxSize)}`);
        return false;
      }
      return true;
    });

    // Check max files limit
    const remainingSlots = maxFiles - files.length;
    const filesToAdd = validFiles.slice(0, remainingSlots);

    if (filesToAdd.length > 0) {
      onFilesSelected(filesToAdd);
    }
  };

  const pendingFiles = files.filter((f) => f.status === "pending" || f.status === "uploading" || f.status === "processing");
  const completedFiles = files.filter((f) => f.status === "completed");
  const errorFiles = files.filter((f) => f.status === "error");

  return (
    <div className={cn("space-y-4", className)}>
      {/* Drop Zone */}
      <div
        className={cn(
          "relative border-2 border-dashed rounded-lg p-8 text-center transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50",
          files.length >= maxFiles && "opacity-50 pointer-events-none"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={acceptedTypes.join(",")}
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={files.length >= maxFiles}
        />

        <div className="flex flex-col items-center gap-2">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
            <Upload className="h-6 w-6 text-primary" />
          </div>
          <div>
            <p className="font-medium">
              {isDragging ? "Drop files here" : "Drag & drop files here"}
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse
            </p>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Supports PDF, Office documents, images, and more (max {formatBytes(maxSize)} per file)
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <Card>
          <div className="p-4 border-b">
            <div className="flex items-center justify-between">
              <h3 className="font-medium">
                Files ({files.length}/{maxFiles})
              </h3>
              <div className="flex gap-2 text-sm text-muted-foreground">
                {completedFiles.length > 0 && (
                  <span className="flex items-center gap-1 text-green-600">
                    <CheckCircle2 className="h-4 w-4" />
                    {completedFiles.length} completed
                  </span>
                )}
                {errorFiles.length > 0 && (
                  <span className="flex items-center gap-1 text-destructive">
                    <AlertCircle className="h-4 w-4" />
                    {errorFiles.length} failed
                  </span>
                )}
              </div>
            </div>
          </div>
          <ScrollArea className="max-h-[300px]">
            <div className="divide-y">
              {files.map((file) => (
                <FileItem
                  key={file.id}
                  file={file}
                  onRemove={() => onRemoveFile(file.id)}
                />
              ))}
            </div>
          </ScrollArea>
        </Card>
      )}
    </div>
  );
}

interface FileItemProps {
  file: UploadFile;
  onRemove: () => void;
}

function FileItem({ file, onRemove }: FileItemProps) {
  const extension = getFileExtension(file.file.name);
  const Icon = getFileIcon(extension);

  return (
    <div className="flex items-center gap-3 p-3 hover:bg-muted/50 transition-colors">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted shrink-0">
        <Icon className="h-5 w-5 text-muted-foreground" />
      </div>

      <div className="flex-1 min-w-0">
        <p className="font-medium text-sm truncate">{file.file.name}</p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-muted-foreground">
            {formatBytes(file.file.size)}
          </span>
          {file.status === "uploading" && (
            <span className="text-xs text-primary">
              Uploading... {file.progress}%
            </span>
          )}
          {file.status === "processing" && (
            <span className="text-xs text-primary flex items-center gap-1">
              <Loader2 className="h-3 w-3 animate-spin" />
              Processing...
            </span>
          )}
          {file.status === "completed" && (
            <span className="text-xs text-green-600 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Completed
            </span>
          )}
          {file.status === "error" && (
            <span className="text-xs text-destructive flex items-center gap-1">
              <AlertCircle className="h-3 w-3" />
              {file.error || "Failed"}
            </span>
          )}
        </div>

        {/* Progress Bar */}
        {(file.status === "uploading" || file.status === "processing") && (
          <div className="mt-2 h-1 w-full bg-muted rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full transition-all duration-300",
                file.status === "processing" ? "bg-primary animate-pulse" : "bg-primary"
              )}
              style={{ width: file.status === "processing" ? "100%" : `${file.progress}%` }}
            />
          </div>
        )}
      </div>

      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0"
        onClick={onRemove}
        disabled={file.status === "uploading" || file.status === "processing"}
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}

function getFileIcon(extension: string): React.ElementType {
  const ext = extension.toLowerCase();

  // Documents
  if (["pdf", "doc", "docx", "txt", "md", "rtf", "odt"].includes(ext)) {
    return FileText;
  }

  // Images
  if (["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "svg"].includes(ext)) {
    return Image;
  }

  // Spreadsheets
  if (["xls", "xlsx", "csv", "ods"].includes(ext)) {
    return FileSpreadsheet;
  }

  // Presentations
  if (["ppt", "pptx", "odp", "key"].includes(ext)) {
    return Presentation;
  }

  return File;
}
