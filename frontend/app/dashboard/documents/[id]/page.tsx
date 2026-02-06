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
  Database,
  Network,
  Sparkles,
  ImageIcon,
  Globe,
  ArrowRight,
  Wand2,
  Loader2,
  Pencil,
  Plus,
  Save,
  ChevronDown,
  Eye,
  ExternalLink,
  Upload,
  Server,
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
import { Progress } from "@/components/ui/progress";
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
import { toast } from "sonner";
import { Textarea } from "@/components/ui/textarea";
import { useDocument, useDeleteDocument, useReprocessDocument, useUpdateDocument, useAccessTiers, useDocumentChunks, useDocumentEntities, useEnhanceSingleDocument, useStartExtractionJob, api } from "@/lib/api";
import { useUser } from "@/lib/auth";
import { UploadedDocumentPreview } from "@/components/preview/UploadedDocumentPreview";

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

  // Collapsible section states
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    info: true,
    source_info: false,
    status_overview: true,
    summary: true,
    insights: false,
    questions: false,
    kg: false,
    preview: false,
    content: false,
  });
  const toggleSection = (key: string) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));

  // State for tier editing
  const [isEditingTier, setIsEditingTier] = useState(false);
  const [selectedTier, setSelectedTier] = useState<number | undefined>(undefined);
  const [isEditingQuestions, setIsEditingQuestions] = useState(false);
  const [editedQuestions, setEditedQuestions] = useState<string[]>([]);
  const [isSavingQuestions, setIsSavingQuestions] = useState(false);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);

  const { data: document, isLoading, error, refetch } = useDocument(documentId);
  const { data: tiersData } = useAccessTiers({ enabled: isAuthenticated });
  const { data: chunks, isLoading: chunksLoading } = useDocumentChunks(documentId, isAuthenticated && !!documentId);
  const { data: kgData, isLoading: kgLoading } = useDocumentEntities(documentId, { enabled: isAuthenticated && !!documentId });
  const tiers = tiersData?.tiers ?? [];

  const enhanced = document?.enhanced_metadata;
  const hasEnhanced = !!enhanced && !!enhanced.summary_short;

  const deleteDocument = useDeleteDocument();
  const reprocessDocument = useReprocessDocument();
  const updateDocument = useUpdateDocument();
  const enhanceDocument = useEnhanceSingleDocument();
  const startExtractionJob = useStartExtractionJob();
  const [isExtractingKG, setIsExtractingKG] = useState(false);
  const [isGeneratingEmbeddings, setIsGeneratingEmbeddings] = useState(false);

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

  const handleEnhance = async (force: boolean = false) => {
    try {
      await enhanceDocument.mutateAsync({ documentId, force });
      toast.success(force ? "Document re-enhanced" : "Document enhanced");
      refetch();
    } catch (error) {
      toast.error("Failed to enhance document", {
        description: getErrorMessage(error, "An error occurred"),
      });
    }
  };

  const handleExtractKG = async () => {
    setIsExtractingKG(true);
    try {
      const result = await startExtractionJob.mutateAsync({
        document_ids: [documentId],
        only_new_documents: false,
      });
      if (result.status === "already_running") {
        toast.info("Extraction already in progress", { description: result.message });
      } else {
        toast.success("KG extraction started");
      }
      refetch();
    } catch (error) {
      toast.error("Failed to extract knowledge graph", {
        description: getErrorMessage(error, "An error occurred"),
      });
    } finally {
      setIsExtractingKG(false);
    }
  };

  const handleGenerateEmbeddings = async () => {
    setIsGeneratingEmbeddings(true);
    try {
      await api.post(`/embeddings/generate/${documentId}`);
      toast.success("Embedding generation started");
      setTimeout(() => refetch(), 3000);
    } catch (error) {
      toast.error("Failed to generate embeddings", {
        description: getErrorMessage(error, "An error occurred"),
      });
    } finally {
      setIsGeneratingEmbeddings(false);
    }
  };

  const handleStartEditQuestions = () => {
    if (enhanced?.hypothetical_questions) {
      setEditedQuestions([...enhanced.hypothetical_questions]);
    } else {
      setEditedQuestions([""]);
    }
    setIsEditingQuestions(true);
  };

  const handleSaveQuestions = async () => {
    const filtered = editedQuestions.filter(q => q.trim().length > 0);
    if (filtered.length === 0) {
      toast.error("At least one question is required");
      return;
    }
    setIsSavingQuestions(true);
    try {
      await api.updateHypotheticalQuestions(documentId, filtered);
      toast.success("Hypothetical questions updated");
      setIsEditingQuestions(false);
      refetch();
    } catch (e: any) {
      toast.error("Failed to save questions", { description: e.message });
    } finally {
      setIsSavingQuestions(false);
    }
  };

  const handleGenerateMore = async () => {
    setIsGeneratingQuestions(true);
    try {
      const result = await api.generateMoreQuestions(documentId, 5);
      toast.success(`Generated ${result.count} new question${result.count !== 1 ? "s" : ""}`);
      refetch();
    } catch (e: any) {
      toast.error("Failed to generate questions", { description: e.message });
    } finally {
      setIsGeneratingQuestions(false);
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
    <div className="space-y-4">
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
                <div className="flex items-center gap-2">
                  <h1 className="text-2xl font-bold tracking-tight">{document.name}</h1>
                  {!document.is_stored_locally && (
                    <Badge variant="outline" className="text-xs gap-1 shrink-0">
                      <Globe className="h-3 w-3" />
                      External
                    </Badge>
                  )}
                </div>
                <p className="text-muted-foreground text-sm">
                  Uploaded {new Date(document.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleEnhance(hasEnhanced)}
                  disabled={enhanceDocument.isPending}
                >
                  {enhanceDocument.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Sparkles className="h-4 w-4 mr-2" />
                  )}
                  {hasEnhanced ? "Re-enhance" : "Enhance"}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Generate summaries, keywords, topics, and questions with LLM</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExtractKG}
                  disabled={isExtractingKG || document.kg_extraction_status === "processing"}
                >
                  {isExtractingKG ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Network className="h-4 w-4 mr-2" />
                  )}
                  {document.kg_extraction_status === "completed" ? "Re-extract KG" : "Extract KG"}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Extract entities and relationships into knowledge graph</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
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

      {/* Document Info & Processing Status */}
      <Collapsible open={openSections.info} onOpenChange={() => toggleSection("info")}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
              <div className="flex items-center justify-between">
                <CardTitle>Document Information</CardTitle>
                <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.info ? "rotate-180" : ""}`} />
              </div>
              <CardDescription>Basic details and processing status</CardDescription>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                {/* Document Info */}
                <div className="space-y-3">
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
                  {document.file_hash && (
                    <div className="flex items-center justify-between py-2">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Hash className="h-4 w-4" />
                        <span>SHA-256</span>
                      </div>
                      <code className="text-[10px] bg-muted px-1.5 py-0.5 rounded max-w-[200px] truncate" title={document.file_hash}>
                        {document.file_hash}
                      </code>
                    </div>
                  )}
                </div>

                {/* Processing Status */}
                <div className="space-y-3">
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
                    <span className="font-medium text-sm">
                      {new Date(document.created_at).toLocaleString()}
                    </span>
                  </div>
                  {document.updated_at && (
                    <div className="flex items-center justify-between py-2 border-b">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <RefreshCw className="h-4 w-4" />
                        <span>Last Updated</span>
                      </div>
                      <span className="font-medium text-sm">
                        {new Date(document.updated_at).toLocaleString()}
                      </span>
                    </div>
                  )}
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
                        <Button size="icon" variant="ghost" className="h-7 w-7" onClick={handleSaveTier} disabled={updateDocument.isPending}>
                          <Check className="h-4 w-4 text-green-500" />
                        </Button>
                        <Button size="icon" variant="ghost" className="h-7 w-7" onClick={handleCancelEditTier} disabled={updateDocument.isPending}>
                          <X className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{document?.access_tier_name || "Unknown"}</Badge>
                        {canEditTier && (
                          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={handleStartEditTier} title="Edit access tier">
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
                          <Badge key={tag} variant="secondary">{tag}</Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Source Information — Provenance & origin tracking */}
      {(document.source_type || document.upload_source_info || document.source_url) && (
        <Collapsible open={openSections.source_info} onOpenChange={() => toggleSection("source_info")}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CardTitle>Source Information</CardTitle>
                    <Badge variant={document.is_stored_locally ? "secondary" : "outline"} className="text-xs">
                      {document.is_stored_locally ? "Stored Locally" : "External Source"}
                    </Badge>
                  </div>
                  <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.source_info ? "rotate-180" : ""}`} />
                </div>
                <CardDescription>How and where this document was uploaded from</CardDescription>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Upload Method */}
                  {!!document.upload_source_info?.upload_method && (
                    <div className="flex items-start gap-3">
                      <Upload className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Upload Method</p>
                        <p className="text-sm text-muted-foreground">
                          {String(document.upload_source_info.upload_method).replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Source Type */}
                  {document.source_type && (
                    <div className="flex items-start gap-3">
                      <Server className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Source Type</p>
                        <p className="text-sm text-muted-foreground">
                          {document.source_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Uploaded By */}
                  {!!document.upload_source_info?.uploaded_by && (
                    <div className="flex items-start gap-3">
                      <Shield className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Uploaded By</p>
                        <p className="text-sm text-muted-foreground truncate">
                          {String(document.upload_source_info.uploaded_by).slice(0, 16)}...
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Upload Time */}
                  {!!document.upload_source_info?.uploaded_at && (
                    <div className="flex items-start gap-3">
                      <Clock className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Upload Time</p>
                        <p className="text-sm text-muted-foreground">
                          {new Date(String(document.upload_source_info.uploaded_at)).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Client Info */}
                  {!!document.upload_source_info?.client_ip && (
                    <div className="flex items-start gap-3">
                      <Globe className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Client IP</p>
                        <p className="text-sm text-muted-foreground">
                          {String(document.upload_source_info.client_ip)}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* User Agent */}
                  {!!document.upload_source_info?.user_agent && (
                    <div className="flex items-start gap-3 sm:col-span-2">
                      <HardDrive className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium">User Agent</p>
                        <p className="text-xs text-muted-foreground truncate">
                          {String(document.upload_source_info.user_agent)}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Connector Info (for synced docs) */}
                  {!!document.upload_source_info?.connector_type && (
                    <div className="flex items-start gap-3">
                      <Database className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Connector</p>
                        <p className="text-sm text-muted-foreground">
                          {String(document.upload_source_info.connector_name || document.upload_source_info.connector_type)}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* External URL */}
                  {document.source_url && (
                    <div className="flex items-start gap-3 sm:col-span-2">
                      <ExternalLink className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium">Source URL</p>
                        <a
                          href={document.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-blue-600 dark:text-blue-400 hover:underline truncate block"
                        >
                          {document.source_url}
                        </a>
                      </div>
                    </div>
                  )}

                  {/* Original Path */}
                  {!!document.upload_source_info?.original_path && (
                    <div className="flex items-start gap-3 sm:col-span-2">
                      <FolderOpen className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium">Original Path</p>
                        <p className="text-xs text-muted-foreground truncate">
                          {String(document.upload_source_info.original_path)}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Storage Mode */}
                  {!!document.upload_source_info?.storage_mode && (
                    <div className="flex items-start gap-3">
                      <HardDrive className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Storage Mode</p>
                        <p className="text-sm text-muted-foreground">
                          {String(document.upload_source_info.storage_mode).replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* Status Overview - Embeddings, KG, Enhancement, Images */}
      <Collapsible open={openSections.status_overview} onOpenChange={() => toggleSection("status_overview")}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
              <div className="flex items-center justify-between">
                <CardTitle>Status Overview</CardTitle>
                <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.status_overview ? "rotate-180" : ""}`} />
              </div>
              <CardDescription>Embedding, enhancement, and extraction status</CardDescription>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {/* Embedding Status */}
                <div className="p-4 rounded-lg border bg-muted/30 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Database className={`h-5 w-5 ${document.has_all_embeddings ? "text-green-500" : "text-amber-500"}`} />
                      <p className="font-medium text-sm">Embeddings</p>
                    </div>
                    {document.has_all_embeddings ? (
                      <Badge variant="outline" className="text-xs text-green-600 border-green-300 dark:border-green-800">Complete</Badge>
                    ) : (
                      <Badge variant="outline" className="text-xs text-amber-600 border-amber-300 dark:border-amber-800">Partial</Badge>
                    )}
                  </div>
                  {document.chunk_count > 0 ? (
                    <>
                      <Progress value={document.embedding_coverage || 0} className="h-2" />
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{document.embedding_coverage?.toFixed(0) || 0}% coverage</span>
                        <span>{document.embedding_count}/{document.chunk_count} chunks</span>
                      </div>
                      {!document.has_all_embeddings && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="w-full h-7 text-xs mt-1"
                          onClick={handleGenerateEmbeddings}
                          disabled={isGeneratingEmbeddings}
                        >
                          {isGeneratingEmbeddings ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Database className="h-3 w-3 mr-1" />}
                          Generate Missing
                        </Button>
                      )}
                    </>
                  ) : (
                    <p className="text-xs text-muted-foreground">No chunks available</p>
                  )}
                </div>

                {/* KG Status */}
                <div className="p-4 rounded-lg border bg-muted/30 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Network className={`h-5 w-5 ${document.kg_extraction_status === "completed" ? "text-green-500" : document.kg_extraction_status === "processing" ? "text-blue-500 animate-pulse" : "text-muted-foreground"}`} />
                      <p className="font-medium text-sm">Knowledge Graph</p>
                    </div>
                    {document.kg_extraction_status === "completed" && (
                      <Badge variant="outline" className="text-xs text-green-600 border-green-300 dark:border-green-800">Extracted</Badge>
                    )}
                    {document.kg_extraction_status === "processing" && (
                      <Badge variant="outline" className="text-xs text-blue-600 border-blue-300 dark:border-blue-800 animate-pulse">Processing</Badge>
                    )}
                    {document.kg_extraction_status === "failed" && (
                      <Badge variant="outline" className="text-xs text-red-600 border-red-300 dark:border-red-800">Failed</Badge>
                    )}
                  </div>
                  {document.kg_extraction_status === "completed" ? (
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-center p-2 rounded bg-background border">
                        <p className="text-lg font-bold">{document.kg_entity_count}</p>
                        <p className="text-xs text-muted-foreground">Entities</p>
                      </div>
                      <div className="text-center p-2 rounded bg-background border">
                        <p className="text-lg font-bold">{document.kg_relation_count}</p>
                        <p className="text-xs text-muted-foreground">Relations</p>
                      </div>
                    </div>
                  ) : document.kg_extraction_status !== "processing" ? (
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full h-7 text-xs"
                      onClick={handleExtractKG}
                      disabled={isExtractingKG}
                    >
                      {isExtractingKG ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Network className="h-3 w-3 mr-1" />}
                      {document.kg_extraction_status === "failed" ? "Retry Extraction" : "Extract KG"}
                    </Button>
                  ) : (
                    <p className="text-xs text-blue-600">Extraction in progress...</p>
                  )}
                </div>

                {/* Enhancement Status */}
                <div className="p-4 rounded-lg border bg-muted/30 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Sparkles className={`h-5 w-5 ${hasEnhanced ? "text-amber-500" : "text-muted-foreground"}`} />
                      <p className="font-medium text-sm">Enhancement</p>
                    </div>
                    {hasEnhanced && (
                      <Badge variant="outline" className="text-xs text-amber-600 border-amber-300 dark:border-amber-800">Enhanced</Badge>
                    )}
                  </div>
                  {hasEnhanced ? (
                    <div className="space-y-1">
                      <div className="flex flex-wrap gap-1">
                        {enhanced?.keywords && <Badge variant="secondary" className="text-[10px] h-5">{enhanced.keywords.length} keywords</Badge>}
                        {enhanced?.topics && <Badge variant="secondary" className="text-[10px] h-5">{enhanced.topics.length} topics</Badge>}
                        {enhanced?.hypothetical_questions && <Badge variant="secondary" className="text-[10px] h-5">{enhanced.hypothetical_questions.length} questions</Badge>}
                      </div>
                      {enhanced?.enhanced_at && (
                        <p className="text-xs text-muted-foreground">{new Date(enhanced.enhanced_at).toLocaleDateString()}</p>
                      )}
                    </div>
                  ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full h-7 text-xs"
                      onClick={() => handleEnhance(false)}
                      disabled={enhanceDocument.isPending}
                    >
                      {enhanceDocument.isPending ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Sparkles className="h-3 w-3 mr-1" />}
                      Enhance Document
                    </Button>
                  )}
                </div>

                {/* Image Analysis Status */}
                <div className="p-4 rounded-lg border bg-muted/30 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ImageIcon className={`h-5 w-5 ${document.image_analysis_status === "completed" ? "text-green-500" : document.images_extracted_count > 0 ? "text-amber-500" : "text-muted-foreground"}`} />
                      <p className="font-medium text-sm">Images</p>
                    </div>
                    {document.image_analysis_status === "completed" && (
                      <Badge variant="outline" className="text-xs text-green-600 border-green-300 dark:border-green-800">Analyzed</Badge>
                    )}
                  </div>
                  {document.images_extracted_count > 0 ? (
                    <>
                      {document.image_analysis_status === "completed" ? (
                        <div className="text-center p-2 rounded bg-background border">
                          <p className="text-lg font-bold">{document.images_analyzed_count}<span className="text-sm font-normal text-muted-foreground">/{document.images_extracted_count}</span></p>
                          <p className="text-xs text-muted-foreground">images analyzed</p>
                        </div>
                      ) : document.image_analysis_status === "processing" ? (
                        <p className="text-xs text-blue-600">Analyzing images...</p>
                      ) : (
                        <p className="text-xs text-muted-foreground">{document.images_extracted_count} images found, not analyzed</p>
                      )}
                    </>
                  ) : (
                    <p className="text-xs text-muted-foreground">No images detected</p>
                  )}
                </div>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Document Preview */}
      <Collapsible open={openSections.preview} onOpenChange={() => toggleSection("preview")}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Eye className="h-5 w-5 text-blue-500" />
                  Document Preview
                </CardTitle>
                <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.preview ? "rotate-180" : ""}`} />
              </div>
              <CardDescription>View the original document (PDF, images, text, etc.)</CardDescription>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent>
              <div className="rounded-lg border overflow-hidden" style={{ minHeight: 400 }}>
                <UploadedDocumentPreview
                  documentId={documentId}
                  fileName={document.name}
                  fileType={document.file_type || ""}
                  className="h-[600px]"
                  isStoredLocally={document.is_stored_locally}
                  sourceUrl={document.source_url}
                  sourceType={document.source_type}
                  summaryShort={document.enhanced_metadata?.summary_short}
                />
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Enhanced Summary */}
      <Collapsible open={openSections.summary} onOpenChange={() => toggleSection("summary")}>
        {hasEnhanced ? (
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-amber-500" />
                      Enhanced Summary
                    </CardTitle>
                    <CardDescription>
                      AI-generated summary and classification
                      {enhanced?.model_used && (
                        <span className="ml-2 text-xs">({enhanced.model_used})</span>
                      )}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => { e.stopPropagation(); handleEnhance(true); }}
                      disabled={enhanceDocument.isPending}
                      className="text-xs"
                    >
                      {enhanceDocument.isPending ? <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5 mr-1" />}
                      Re-enhance
                    </Button>
                    <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.summary ? "rotate-180" : ""}`} />
                  </div>
                </div>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="space-y-4">
                {enhanced?.summary_short && (
                  <div className="p-4 rounded-lg bg-gradient-to-r from-amber-50/50 to-orange-50/50 dark:from-amber-950/20 dark:to-orange-950/20 border border-amber-200/50 dark:border-amber-800/30">
                    <p className="text-sm leading-relaxed">{enhanced.summary_short}</p>
                    {enhanced.summary_detailed && enhanced.summary_detailed !== enhanced.summary_short && (
                      <details className="mt-3">
                        <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                          Show detailed summary
                        </summary>
                        <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                          {enhanced.summary_detailed}
                        </p>
                      </details>
                    )}
                  </div>
                )}
                {(enhanced?.language || enhanced?.document_type) && (
                  <div className="flex items-center gap-2 pt-2 border-t">
                    {enhanced.language && (
                      <Badge variant="outline" className="gap-1">
                        <Globe className="h-3 w-3" />
                        {enhanced.language}
                      </Badge>
                    )}
                    {enhanced.document_type && (
                      <Badge variant="outline" className="gap-1">
                        <FileText className="h-3 w-3" />
                        {enhanced.document_type}
                      </Badge>
                    )}
                  </div>
                )}
              </CardContent>
            </CollapsibleContent>
          </Card>
        ) : (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-10 text-center">
              <Sparkles className="h-12 w-12 text-muted-foreground/30 mb-4" />
              <h3 className="font-medium text-lg mb-1">Not Yet Enhanced</h3>
              <p className="text-sm text-muted-foreground mb-4 max-w-md">
                Enhance this document to generate summaries, keywords, topics, named entities, and hypothetical questions using AI.
              </p>
              <Button
                onClick={() => handleEnhance(false)}
                disabled={enhanceDocument.isPending}
                className="gap-2"
              >
                {enhanceDocument.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                {enhanceDocument.isPending ? "Enhancing..." : "Enhance Document"}
              </Button>
            </CardContent>
          </Card>
        )}
      </Collapsible>

      {/* Insights - Keywords, Topics, Entities */}
      {hasEnhanced && (
        <Collapsible open={openSections.insights} onOpenChange={() => toggleSection("insights")}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Keywords, Topics & Entities</CardTitle>
                    <CardDescription>
                      {enhanced?.keywords?.length || 0} keywords, {enhanced?.topics?.length || 0} topics, {Object.values(enhanced?.entities || {}).flat().length} entities
                    </CardDescription>
                  </div>
                  <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.insights ? "rotate-180" : ""}`} />
                </div>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent>
                <div className="grid gap-6 md:grid-cols-2">
                  {/* Keywords & Topics */}
                  <div className="space-y-4">
                    {enhanced?.keywords && enhanced.keywords.length > 0 && (
                      <div>
                        <p className="text-sm font-medium text-muted-foreground mb-2">Keywords</p>
                        <div className="flex flex-wrap gap-1.5">
                          {enhanced.keywords.map((kw: string) => (
                            <Badge key={kw} variant="secondary" className="text-xs">{kw}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    {enhanced?.topics && enhanced.topics.length > 0 && (
                      <div>
                        <p className="text-sm font-medium text-muted-foreground mb-2">Topics</p>
                        <div className="flex flex-wrap gap-1.5">
                          {enhanced.topics.map((topic: string) => (
                            <Badge key={topic} variant="outline" className="text-xs bg-blue-50 dark:bg-blue-950/30">{topic}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Named Entities */}
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-2">Named Entities</p>
                    {enhanced?.entities && Object.keys(enhanced.entities).length > 0 ? (
                      <div className="space-y-3">
                        {Object.entries(enhanced.entities).map(([type, items]) => (
                          <div key={type}>
                            <p className="text-xs font-medium text-muted-foreground capitalize mb-1">{type}</p>
                            <div className="flex flex-wrap gap-1">
                              {(items as string[]).map((item: string) => (
                                <Badge key={item} variant="outline" className="text-xs">{item}</Badge>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No named entities found.</p>
                    )}
                  </div>
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* Hypothetical Questions */}
      {hasEnhanced && (
        <Collapsible open={openSections.questions} onOpenChange={() => toggleSection("questions")}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Hypothetical Questions</CardTitle>
                    <CardDescription>
                      {enhanced?.hypothetical_questions?.length || 0} AI-generated questions for improved retrieval
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    {!isEditingQuestions && (
                      <>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => { e.stopPropagation(); handleGenerateMore(); }}
                          disabled={isGeneratingQuestions}
                        >
                          {isGeneratingQuestions ? (
                            <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                          ) : (
                            <Wand2 className="h-3.5 w-3.5 mr-1.5" />
                          )}
                          {isGeneratingQuestions ? "Generating..." : "Generate More"}
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={(e) => { e.stopPropagation(); handleStartEditQuestions(); }}
                        >
                          <Pencil className="h-3.5 w-3.5 mr-1.5" />
                          Edit
                        </Button>
                      </>
                    )}
                    <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.questions ? "rotate-180" : ""}`} />
                  </div>
                </div>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent>
                {isEditingQuestions ? (
                  <div className="space-y-3">
                    {editedQuestions.map((q, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className="text-sm text-muted-foreground mt-2 min-w-[24px]">{i + 1}.</span>
                        <Textarea
                          value={q}
                          onChange={(e) => {
                            const updated = [...editedQuestions];
                            updated[i] = e.target.value;
                            setEditedQuestions(updated);
                          }}
                          className="min-h-[60px] text-sm"
                          placeholder="Enter a question..."
                        />
                        {editedQuestions.length > 1 && (
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-8 w-8 mt-0.5 text-destructive hover:text-destructive"
                            onClick={() => setEditedQuestions(editedQuestions.filter((_, idx) => idx !== i))}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    ))}
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full"
                      onClick={() => setEditedQuestions([...editedQuestions, ""])}
                    >
                      <Plus className="h-3.5 w-3.5 mr-1.5" />
                      Add Question
                    </Button>
                    <div className="flex items-center justify-between pt-2 border-t">
                      <p className="text-xs text-muted-foreground italic">
                        Changes will re-embed questions in the vector store.
                      </p>
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setIsEditingQuestions(false)}
                          disabled={isSavingQuestions}
                        >
                          <X className="h-3.5 w-3.5 mr-1" />
                          Cancel
                        </Button>
                        <Button
                          size="sm"
                          onClick={handleSaveQuestions}
                          disabled={isSavingQuestions}
                        >
                          {isSavingQuestions ? (
                            <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                          ) : (
                            <Save className="h-3.5 w-3.5 mr-1.5" />
                          )}
                          Save
                        </Button>
                      </div>
                    </div>
                  </div>
                ) : enhanced?.hypothetical_questions && enhanced.hypothetical_questions.length > 0 ? (
                  <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                    {enhanced.hypothetical_questions.map((q: string, i: number) => (
                      <li key={i} className="leading-relaxed">{q}</li>
                    ))}
                  </ol>
                ) : (
                  <div className="text-center py-6 text-muted-foreground">
                    <p className="text-sm">No hypothetical questions generated yet.</p>
                    <Button
                      size="sm"
                      variant="outline"
                      className="mt-3"
                      onClick={handleGenerateMore}
                      disabled={isGeneratingQuestions}
                    >
                      {isGeneratingQuestions ? (
                        <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                      ) : (
                        <Wand2 className="h-3.5 w-3.5 mr-1.5" />
                      )}
                      Generate with LLM
                    </Button>
                  </div>
                )}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* Knowledge Graph */}
      <Collapsible open={openSections.kg} onOpenChange={() => toggleSection("kg")}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Network className="h-5 w-5 text-blue-500" />
                    Knowledge Graph
                  </CardTitle>
                  <CardDescription>
                    {kgData?.entities?.length
                      ? `${kgData.entities.length} entities, ${kgData.relations?.length || 0} relations`
                      : "Entities and relationships extracted from this document"}
                  </CardDescription>
                </div>
                <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.kg ? "rotate-180" : ""}`} />
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <CardContent>
              {kgLoading ? (
                <div className="space-y-3">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : kgData?.entities && kgData.entities.length > 0 ? (
                <div className="space-y-6">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground mb-3">
                      Entities ({kgData.entities.length})
                    </p>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {kgData.entities.map((entity: any) => (
                        <div key={entity.id} className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-sm">{entity.name}</span>
                            <Badge variant="outline" className="text-xs capitalize">{entity.entity_type}</Badge>
                          </div>
                          {entity.description && (
                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{entity.description}</p>
                          )}
                          <p className="text-xs text-muted-foreground mt-1">
                            {entity.mention_count} mention{entity.mention_count !== 1 ? "s" : ""}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {kgData.relations && kgData.relations.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-muted-foreground mb-3">
                        Relations ({kgData.relations.length})
                      </p>
                      <div className="space-y-2">
                        {kgData.relations.map((rel: any) => (
                          <div key={rel.id} className="flex items-center gap-2 p-2.5 rounded-lg border bg-muted/30 text-sm">
                            <span className="font-medium">{rel.source_entity_name}</span>
                            <ArrowRight className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                            <Badge variant="outline" className="text-xs flex-shrink-0">{rel.relation_type}</Badge>
                            <ArrowRight className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                            <span className="font-medium">{rel.target_entity_name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Network className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No knowledge graph data</p>
                  <p className="text-xs mt-1">
                    Extract knowledge graph to see entities and relationships from this document.
                  </p>
                </div>
              )}
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* Document Content (Chunks) */}
      <Collapsible open={openSections.content} onOpenChange={() => toggleSection("content")}>
        <Card>
          <CollapsibleTrigger asChild>
            <CardHeader className="cursor-pointer hover:bg-muted/30 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Layers className="h-5 w-5 text-muted-foreground" />
                    Document Chunks
                  </CardTitle>
                  <CardDescription>
                    Extracted text content ({chunks?.length || 0} chunks)
                  </CardDescription>
                </div>
                <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${openSections.content ? "rotate-180" : ""}`} />
              </div>
            </CardHeader>
          </CollapsibleTrigger>
          <CollapsibleContent>
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
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  );
}
