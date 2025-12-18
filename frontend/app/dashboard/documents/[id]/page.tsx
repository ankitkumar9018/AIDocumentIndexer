"use client";

import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import {
  ArrowLeft,
  FileText,
  Calendar,
  HardDrive,
  Layers,
  FolderOpen,
  Hash,
  RefreshCw,
  Trash2,
  Download,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import { useDocument, useDeleteDocument, useReprocessDocument, api } from "@/lib/api";

const formatFileSize = (bytes: number) => {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const getStatusIcon = (status: string) => {
  switch (status?.toLowerCase()) {
    case "completed":
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case "failed":
      return <XCircle className="h-4 w-4 text-destructive" />;
    case "processing":
      return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
    default:
      return <AlertCircle className="h-4 w-4 text-muted-foreground" />;
  }
};

const getStatusColor = (status: string) => {
  switch (status?.toLowerCase()) {
    case "completed":
      return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    case "failed":
      return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
    case "processing":
      return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
    default:
      return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200";
  }
};

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const documentId = params.id as string;

  const { data: document, isLoading, error, refetch } = useDocument(documentId);

  const deleteDocument = useDeleteDocument();
  const reprocessDocument = useReprocessDocument();

  const handleDelete = async () => {
    if (!document) return;

    if (!confirm(`Are you sure you want to delete "${document.name}"?`)) {
      return;
    }

    try {
      await deleteDocument.mutateAsync(documentId);
      toast.success("Document deleted");
      router.push("/dashboard/documents");
    } catch (error: any) {
      toast.error("Failed to delete document", {
        description: error?.detail || error?.message || "An error occurred",
      });
    }
  };

  const handleReprocess = async () => {
    try {
      await reprocessDocument.mutateAsync(documentId);
      toast.success("Document queued for reprocessing");
      refetch();
    } catch (error: any) {
      toast.error("Failed to reprocess document", {
        description: error?.detail || error?.message || "An error occurred",
      });
    }
  };

  const handleDownload = async () => {
    if (!document) return;
    try {
      await api.downloadDocument(documentId, document.name);
      toast.success("Download started", {
        description: `Downloading "${document.name}"...`,
      });
    } catch (error: any) {
      console.error("Download failed:", error);
      toast.error("Failed to download document", {
        description: error?.detail || error?.message || "An error occurred",
      });
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Skeleton className="h-10 w-10" />
          <div className="space-y-2">
            <Skeleton className="h-6 w-64" />
            <Skeleton className="h-4 w-32" />
          </div>
        </div>
        <div className="grid gap-6 md:grid-cols-2">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className="space-y-6">
        <Button variant="ghost" onClick={() => router.back()}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center py-12">
              <XCircle className="h-12 w-12 mx-auto text-destructive opacity-50 mb-3" />
              <h3 className="text-lg font-medium">Document not found</h3>
              <p className="text-muted-foreground mt-1">
                The document you're looking for doesn't exist or has been deleted.
              </p>
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => router.push("/dashboard/documents")}
              >
                Go to Documents
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.back()}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <FileText className="h-8 w-8 text-muted-foreground" />
              <div>
                <h1 className="text-2xl font-bold tracking-tight">{document.name}</h1>
                <p className="text-muted-foreground text-sm">
                  Uploaded {new Date(document.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReprocess}
            disabled={reprocessDocument.isPending}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${reprocessDocument.isPending ? "animate-spin" : ""}`} />
            Reprocess
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={handleDelete}
            disabled={deleteDocument.isPending}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Document Info */}
        <Card>
          <CardHeader>
            <CardTitle>Document Information</CardTitle>
            <CardDescription>Basic details about this document</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between py-2 border-b">
              <div className="flex items-center gap-2 text-muted-foreground">
                <FileText className="h-4 w-4" />
                <span>File Type</span>
              </div>
              <Badge variant="secondary">{document.file_type?.toUpperCase() || "Unknown"}</Badge>
            </div>

            <div className="flex items-center justify-between py-2 border-b">
              <div className="flex items-center gap-2 text-muted-foreground">
                <HardDrive className="h-4 w-4" />
                <span>File Size</span>
              </div>
              <span className="font-medium">{formatFileSize(document.file_size)}</span>
            </div>

            <div className="flex items-center justify-between py-2 border-b">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Layers className="h-4 w-4" />
                <span>Chunks</span>
              </div>
              <span className="font-medium">{document.chunk_count || 0}</span>
            </div>

            {document.page_count !== undefined && (
              <div className="flex items-center justify-between py-2 border-b">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <FileText className="h-4 w-4" />
                  <span>Pages</span>
                </div>
                <span className="font-medium">{document.page_count}</span>
              </div>
            )}

            {document.word_count !== undefined && (
              <div className="flex items-center justify-between py-2 border-b">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Hash className="h-4 w-4" />
                  <span>Words</span>
                </div>
                <span className="font-medium">{document.word_count?.toLocaleString()}</span>
              </div>
            )}

            {document.collection && (
              <div className="flex items-center justify-between py-2 border-b">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <FolderOpen className="h-4 w-4" />
                  <span>Collection</span>
                </div>
                <Badge variant="outline">{document.collection}</Badge>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Processing Status */}
        <Card>
          <CardHeader>
            <CardTitle>Processing Status</CardTitle>
            <CardDescription>Document processing information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between py-2 border-b">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span>Status</span>
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(document.status)}
                <Badge className={getStatusColor(document.status)}>
                  {document.status}
                </Badge>
              </div>
            </div>

            <div className="flex items-center justify-between py-2 border-b">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Calendar className="h-4 w-4" />
                <span>Created</span>
              </div>
              <span className="font-medium">
                {new Date(document.created_at).toLocaleString()}
              </span>
            </div>

            {document.updated_at && (
              <div className="flex items-center justify-between py-2 border-b">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <RefreshCw className="h-4 w-4" />
                  <span>Last Updated</span>
                </div>
                <span className="font-medium">
                  {new Date(document.updated_at).toLocaleString()}
                </span>
              </div>
            )}

            {document.access_tier_name && (
              <div className="flex items-center justify-between py-2 border-b">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Layers className="h-4 w-4" />
                  <span>Access Tier</span>
                </div>
                <Badge variant="outline">{document.access_tier_name}</Badge>
              </div>
            )}

            {document.tags && document.tags.length > 0 && (
              <div className="py-2">
                <div className="flex items-center gap-2 text-muted-foreground mb-2">
                  <Hash className="h-4 w-4" />
                  <span>Tags</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {document.tags.map((tag: string) => (
                    <Badge key={tag} variant="secondary">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* File Hash */}
      {document.file_hash && (
        <Card>
          <CardHeader>
            <CardTitle>File Hash</CardTitle>
            <CardDescription>SHA-256 hash for duplicate detection</CardDescription>
          </CardHeader>
          <CardContent>
            <code className="text-xs bg-muted p-2 rounded block overflow-x-auto">
              {document.file_hash}
            </code>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
