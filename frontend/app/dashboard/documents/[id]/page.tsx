"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { getErrorMessage } from "@/lib/errors";
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
  Shield,
  Edit2,
  Check,
  X,
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
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { useDocument, useDeleteDocument, useReprocessDocument, useUpdateDocument, useAccessTiers, useDocumentChunks, api } from "@/lib/api";
import { useUser } from "@/lib/auth";

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
  const { user } = useUser();

  const documentId = params.id as string;

  // State for tier editing
  const [isEditingTier, setIsEditingTier] = useState(false);
  const [selectedTier, setSelectedTier] = useState<number | undefined>(undefined);

  const { data: document, isLoading, error, refetch } = useDocument(documentId);
  const { data: tiersData } = useAccessTiers({ enabled: isAuthenticated });
  const { data: chunks, isLoading: chunksLoading } = useDocumentChunks(documentId, isAuthenticated && !!documentId);
  const tiers = tiersData?.tiers ?? [];

  const deleteDocument = useDeleteDocument();
  const reprocessDocument = useReprocessDocument();
  const updateDocument = useUpdateDocument();

  // Get document's current tier level
  const documentTierLevel = document?.access_tier ?? 1;
  // Get user's tier level
  const userTierLevel = user?.accessTier ?? 1;
  // Can edit tier if user's tier >= document's tier
  const canEditTier = userTierLevel >= documentTierLevel;
  // Filter tiers to only show tiers at or below user's level
  const availableTiers = tiers.filter((t: { id: string; name: string; level: number }) => t.level <= userTierLevel);

  const handleDelete = async () => {
    if (!document) return;

    const hardDelete = confirm(
      `Are you sure you want to delete "${document.name}"?\n\nClick OK for soft delete (recoverable) or use the Documents page for permanent deletion.`
    );
    if (!hardDelete && !confirm(`Proceed with soft delete of "${document.name}"?`)) {
      return;
    }

    try {
      await deleteDocument.mutateAsync({ id: documentId, hardDelete: false });
      toast.success("Document deleted");
      router.push("/dashboard/documents");
    } catch (error) {
      toast.error("Failed to delete document", {
        description: getErrorMessage(error, "An error occurred"),
      });
    }
  };

  const handleReprocess = async () => {
    try {
      await reprocessDocument.mutateAsync(documentId);
      toast.success("Document queued for reprocessing");
      refetch();
    } catch (error) {
      toast.error("Failed to reprocess document", {
        description: getErrorMessage(error, "An error occurred"),
      });
    }
  };

  const handleStartEditTier = () => {
    setSelectedTier(documentTierLevel);
    setIsEditingTier(true);
  };

  const handleCancelEditTier = () => {
    setIsEditingTier(false);
    setSelectedTier(undefined);
  };

  const handleSaveTier = async () => {
    if (selectedTier === undefined || selectedTier === documentTierLevel) {
      handleCancelEditTier();
      return;
    }

    try {
      await updateDocument.mutateAsync({
        id: documentId,
        data: { access_tier: selectedTier },
      });
      toast.success("Access tier updated");
      setIsEditingTier(false);
      setSelectedTier(undefined);
      refetch();
    } catch (error) {
      toast.error("Failed to update access tier", {
        description: getErrorMessage(error, "An error occurred"),
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
    } catch (error) {
      console.error("Download failed:", error);
      toast.error("Failed to download document", {
        description: getErrorMessage(error, "An error occurred"),
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

            {/* Access Tier - Editable */}
            <div className="flex items-center justify-between py-2 border-b">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Shield className="h-4 w-4" />
                <span>Access Tier</span>
              </div>
              {isEditingTier ? (
                <div className="flex items-center gap-2">
                  <select
                    className="h-8 px-2 rounded-md border bg-background text-sm"
                    value={selectedTier ?? documentTierLevel}
                    onChange={(e) => setSelectedTier(parseInt(e.target.value))}
                  >
                    {availableTiers
                      .sort((a: { level: number }, b: { level: number }) => a.level - b.level)
                      .map((tier: { id: string; name: string; level: number }) => (
                        <option key={tier.id} value={tier.level}>
                          {tier.name} (Level {tier.level})
                        </option>
                      ))}
                  </select>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7"
                    onClick={handleSaveTier}
                    disabled={updateDocument.isPending}
                  >
                    <Check className="h-4 w-4 text-green-500" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7"
                    onClick={handleCancelEditTier}
                    disabled={updateDocument.isPending}
                  >
                    <X className="h-4 w-4 text-red-500" />
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Badge variant="outline">
                    {document?.access_tier_name || "Unknown"}
                  </Badge>
                  {canEditTier && (
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7"
                      onClick={handleStartEditTier}
                      title="Edit access tier"
                    >
                      <Edit2 className="h-3 w-3 text-muted-foreground" />
                    </Button>
                  )}
                </div>
              )}
            </div>

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

      {/* Document Content */}
      <Card>
        <CardHeader>
          <CardTitle>Document Content</CardTitle>
          <CardDescription>
            Extracted text content ({chunks?.length || 0} chunks)
          </CardDescription>
        </CardHeader>
        <CardContent>
          {chunksLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-20 w-full" />
            </div>
          ) : chunks && chunks.length > 0 ? (
            <ScrollArea className="h-[500px] pr-4">
              <div className="space-y-4">
                {chunks.map((chunk) => (
                  <div
                    key={chunk.id}
                    className="p-4 bg-muted/30 rounded-lg border border-border/50"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="secondary" className="text-xs">
                        Chunk {chunk.chunk_index + 1}
                      </Badge>
                      {chunk.page_number && (
                        <Badge variant="outline" className="text-xs">
                          Page {chunk.page_number}
                        </Badge>
                      )}
                      {chunk.token_count && (
                        <span className="text-xs text-muted-foreground">
                          {chunk.token_count} tokens
                        </span>
                      )}
                    </div>
                    <p className="text-sm whitespace-pre-wrap leading-relaxed">
                      {chunk.content}
                    </p>
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No content chunks available</p>
              <p className="text-xs mt-1">
                The document may still be processing or has no extractable text.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
