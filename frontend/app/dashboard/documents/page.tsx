"use client";

import { useState, useMemo } from "react";
import { useSession } from "next-auth/react";
import {
  FileText,
  Search,
  Filter,
  Grid,
  List,
  MoreVertical,
  Trash2,
  Download,
  Eye,
  Edit,
  FolderOpen,
  Image,
  FileSpreadsheet,
  Presentation,
  File,
  ChevronDown,
  CheckSquare,
  Square,
  RefreshCw,
  SortAsc,
  SortDesc,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Sparkles,
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "sonner";
import {
  useDocuments,
  useDocument,
  useUpdateDocument,
  useDeleteDocument,
  useSearchDocuments,
  useCollections,
  useEnhanceDocuments,
  useEstimateEnhancementCost,
  useLLMProviders,
  useLLMOperations,
  useSetLLMOperationConfig,
  Document,
  api,
} from "@/lib/api";
import { Wand2, Zap, Settings, Info, Bot, Tag } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { X, Plus } from "lucide-react";

type ViewMode = "grid" | "list";

const getFileIcon = (fileType: string) => {
  if (fileType?.includes("image")) return Image;
  if (fileType?.includes("pdf")) return FileText;
  if (fileType?.includes("spreadsheet") || fileType?.includes("excel") || fileType?.includes("csv"))
    return FileSpreadsheet;
  if (fileType?.includes("presentation") || fileType?.includes("powerpoint"))
    return Presentation;
  return File;
};

const formatFileSize = (bytes: number) => {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const PAGE_SIZE_OPTIONS = [10, 25, 50, 100];

// File type filter options
const FILE_TYPE_OPTIONS = [
  { value: "", label: "All Types" },
  { value: "pdf", label: "PDF" },
  { value: "docx", label: "Word" },
  { value: "xlsx", label: "Excel" },
  { value: "pptx", label: "PowerPoint" },
  { value: "txt", label: "Text" },
  { value: "image", label: "Images" },
];

export default function DocumentsPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<string>("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [selectedFileType, setSelectedFileType] = useState<string>("");
  const [autoTagEnabled, setAutoTagEnabled] = useState(false);

  // Queries - only fetch when authenticated
  const { data: documentsData, isLoading, refetch } = useDocuments({
    collection: selectedCollection || undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
  }, { enabled: isAuthenticated });
  const { data: collections } = useCollections({ enabled: isAuthenticated });

  // LLM configuration for document enhancement
  const { data: providersData } = useLLMProviders({ enabled: isAuthenticated });
  const { data: operationsData, refetch: refetchOperations } = useLLMOperations({ enabled: isAuthenticated });
  const setOperationConfig = useSetLLMOperationConfig();

  // Get the current LLM configured for document enhancement
  const enhancementConfig = operationsData?.operations?.find(
    (op: any) => op.operation_type === "document_enhancement"
  );
  const enhancementProvider = enhancementConfig?.provider_id
    ? providersData?.providers?.find((p: any) => p.id === enhancementConfig.provider_id)
    : providersData?.providers?.find((p: any) => p.is_default);
  const enhancementModel = enhancementConfig?.model_override || enhancementProvider?.default_chat_model || "default";

  // Get active providers for the dropdown
  const activeProviders = providersData?.providers?.filter((p: any) => p.is_active) || [];

  // Mutations
  const deleteDocument = useDeleteDocument();

  // Use real documents from API (no mock fallback)
  const documents: Document[] = documentsData?.documents ?? [];

  // Filter documents by search query and file type
  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesFileType = !selectedFileType || doc.file_type?.toLowerCase().includes(selectedFileType.toLowerCase());
      return matchesSearch && matchesFileType;
    });
  }, [documents, searchQuery, selectedFileType]);

  // Pagination calculations
  const totalPages = Math.ceil(filteredDocuments.length / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  const paginatedDocuments = filteredDocuments.slice(startIndex, endIndex);

  // Reset to page 1 when filters change
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    setCurrentPage(1);
  };

  const handleFileTypeChange = (value: string) => {
    setSelectedFileType(value);
    setCurrentPage(1);
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
    setCurrentPage(1);
  };

  const toggleSelectDocument = (id: string) => {
    setSelectedDocuments((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedDocuments.size === filteredDocuments.length) {
      setSelectedDocuments(new Set());
    } else {
      setSelectedDocuments(new Set(filteredDocuments.map((d) => d.id)));
    }
  };

  const handleDeleteSelected = async () => {
    for (const id of selectedDocuments) {
      await deleteDocument.mutateAsync(id);
    }
    setSelectedDocuments(new Set());
    refetch();
  };

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(field);
      setSortOrder("desc");
    }
  };

  // Use real collections from API (no mock fallback)
  const collectionsList = collections?.collections?.map((c: { name: string }) => c.name) ?? [];

  const handleDeleteDocument = async (docId: string, docName: string) => {
    try {
      await deleteDocument.mutateAsync(docId);
      toast.success("Document deleted", {
        description: `"${docName}" has been deleted.`,
      });
      refetch();
    } catch (error: any) {
      console.error("Delete failed:", error);
      toast.error("Failed to delete document", {
        description: error?.detail || error?.message || "An error occurred",
      });
    }
  };

  const handleDownloadDocument = async (docId: string, docName: string) => {
    try {
      await api.downloadDocument(docId, docName);
      toast.success("Download started", {
        description: `Downloading "${docName}"...`,
      });
    } catch (error: any) {
      console.error("Download failed:", error);
      toast.error("Failed to download document", {
        description: error?.detail || error?.message || "An error occurred",
      });
    }
  };

  const handleViewDocument = (docId: string) => {
    // Navigate to document detail view
    window.location.href = `/dashboard/documents/${docId}`;
  };

  const [isAutoTagging, setIsAutoTagging] = useState(false);
  const [isEnhancing, setIsEnhancing] = useState(false);

  // Edit tags dialog state
  const [editTagsDialogOpen, setEditTagsDialogOpen] = useState(false);
  const [editingDocument, setEditingDocument] = useState<Document | null>(null);
  const [editingTags, setEditingTags] = useState<string[]>([]);
  const [newTagInput, setNewTagInput] = useState("");
  const [isSavingTags, setIsSavingTags] = useState(false);

  // Enhancement mutation
  const enhanceDocuments = useEnhanceDocuments();
  const updateDocument = useUpdateDocument();

  const handleAutoTagDocument = async (docId: string, docName: string) => {
    try {
      setIsAutoTagging(true);
      const result = await api.autoTagDocument(docId);
      toast.success("Tags generated", {
        description: `Generated ${result.tags.length} tags for "${docName}": ${result.tags.join(", ")}`,
      });
      refetch();
    } catch (error: any) {
      console.error("Auto-tag failed:", error);
      toast.error("Failed to auto-tag document", {
        description: error?.detail || error?.message || "An error occurred",
      });
    } finally {
      setIsAutoTagging(false);
    }
  };

  const handleBulkAutoTag = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      setIsAutoTagging(true);
      const result = await api.bulkAutoTag(Array.from(selectedDocuments));
      toast.success("Bulk auto-tagging complete", {
        description: `Successfully tagged ${result.successful} of ${result.processed} documents`,
      });
      setSelectedDocuments(new Set());
      refetch();
    } catch (error: any) {
      console.error("Bulk auto-tag failed:", error);
      toast.error("Failed to auto-tag documents", {
        description: error?.detail || error?.message || "An error occurred",
      });
    } finally {
      setIsAutoTagging(false);
    }
  };

  const handleBulkEnhance = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      setIsEnhancing(true);
      const result = await enhanceDocuments.mutateAsync({
        document_ids: Array.from(selectedDocuments),
        auto_tag: autoTagEnabled,
      });
      toast.success("Document enhancement complete", {
        description: result.message || `Enhanced ${result.successful} of ${result.total} documents. Estimated cost: $${result.estimated_cost_usd.toFixed(4)}`,
      });
      setSelectedDocuments(new Set());
      refetch();
    } catch (error: any) {
      console.error("Bulk enhance failed:", error);
      toast.error("Failed to enhance documents", {
        description: error?.detail || error?.message || "An error occurred",
      });
    } finally {
      setIsEnhancing(false);
    }
  };

  const handleEnhanceAll = async () => {
    try {
      setIsEnhancing(true);
      const result = await enhanceDocuments.mutateAsync({
        collection: selectedCollection || undefined,
        auto_tag: autoTagEnabled,
      });
      toast.success("Document enhancement complete", {
        description: result.message || `Enhanced ${result.successful} of ${result.total} documents. Estimated cost: $${result.estimated_cost_usd.toFixed(4)}`,
      });
      refetch();
    } catch (error: any) {
      console.error("Enhance all failed:", error);
      toast.error("Failed to enhance documents", {
        description: error?.detail || error?.message || "An error occurred",
      });
    } finally {
      setIsEnhancing(false);
    }
  };

  const handleProviderChange = async (providerId: string) => {
    try {
      await setOperationConfig.mutateAsync({
        operationType: "document_enhancement",
        data: { provider_id: providerId },
      });
      toast.success("Enhancement LLM updated");
      refetchOperations();
    } catch (error: any) {
      toast.error("Failed to update LLM", {
        description: error?.detail || error?.message || "An error occurred",
      });
    }
  };

  // Edit tags functions
  const handleOpenEditTags = (doc: Document) => {
    setEditingDocument(doc);
    setEditingTags(doc.tags ? [...doc.tags] : []);
    setNewTagInput("");
    setEditTagsDialogOpen(true);
  };

  const handleAddTag = () => {
    const tag = newTagInput.trim();
    if (tag && !editingTags.includes(tag)) {
      setEditingTags([...editingTags, tag]);
      setNewTagInput("");
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setEditingTags(editingTags.filter((t) => t !== tagToRemove));
  };

  const handleSaveTags = async () => {
    if (!editingDocument) return;

    try {
      setIsSavingTags(true);
      await updateDocument.mutateAsync({
        id: editingDocument.id,
        data: { tags: editingTags },
      });
      toast.success("Tags updated", {
        description: `Updated tags for "${editingDocument.name}"`,
      });
      setEditTagsDialogOpen(false);
      refetch();
    } catch (error: any) {
      console.error("Failed to update tags:", error);
      toast.error("Failed to update tags", {
        description: error?.detail || error?.message || "An error occurred",
      });
    } finally {
      setIsSavingTags(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Documents</h1>
          <p className="text-muted-foreground">
            Manage your indexed documents and collections
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Enhancement Controls */}
          <div className="flex items-center gap-2 p-2 rounded-lg border bg-muted/30">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={handleEnhanceAll}
                    variant="outline"
                    size="sm"
                    disabled={isEnhancing || setOperationConfig.isPending}
                  >
                    <Wand2 className="h-4 w-4 mr-2" />
                    {isEnhancing ? "Enhancing..." : "Enhance All"}
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-xs">
                  <p className="text-sm">
                    Enhance documents with LLM-extracted metadata for better RAG search.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* LLM Provider Selector */}
            <Select
              value={enhancementProvider?.id || ""}
              onValueChange={handleProviderChange}
              disabled={setOperationConfig.isPending}
            >
              <SelectTrigger className="w-[140px] h-8 text-xs">
                <Bot className="h-3 w-3 mr-1" />
                <SelectValue placeholder="Select LLM" />
              </SelectTrigger>
              <SelectContent>
                {activeProviders.map((provider: any) => (
                  <SelectItem key={provider.id} value={provider.id}>
                    {provider.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Auto-tag Checkbox */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5">
                    <Checkbox
                      id="auto-tag"
                      checked={autoTagEnabled}
                      onCheckedChange={(checked: boolean | "indeterminate") => setAutoTagEnabled(checked === true)}
                    />
                    <Label htmlFor="auto-tag" className="text-xs cursor-pointer flex items-center gap-1">
                      <Tag className="h-3 w-3" />
                      Auto-tag
                    </Label>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  <p className="text-sm">Also generate tags/collections after enhancement</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          <Button onClick={() => refetch()} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="pl-9"
            aria-label="Search documents"
          />
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* File Type Filter */}
          <select
            value={selectedFileType}
            onChange={(e) => handleFileTypeChange(e.target.value)}
            className="h-10 px-3 rounded-md border bg-background text-sm"
            aria-label="Filter by file type"
          >
            {FILE_TYPE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          {/* Collection Filter */}
          <select
            value={selectedCollection || ""}
            onChange={(e) => {
              setSelectedCollection(e.target.value || null);
              setCurrentPage(1);
            }}
            className="h-10 px-3 rounded-md border bg-background text-sm"
            aria-label="Filter by collection"
          >
            <option value="">All Collections</option>
            {collectionsList.map((col: string) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>

          {/* View Toggle */}
          <div className="flex rounded-lg border bg-muted p-1">
            <Button
              variant={viewMode === "list" ? "secondary" : "ghost"}
              size="sm"
              className="h-8 px-2"
              onClick={() => setViewMode("list")}
              aria-label="List view"
            >
              <List className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === "grid" ? "secondary" : "ghost"}
              size="sm"
              className="h-8 px-2"
              onClick={() => setViewMode("grid")}
              aria-label="Grid view"
            >
              <Grid className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Bulk Actions */}
      {selectedDocuments.size > 0 && (
        <div className="flex items-center gap-4 p-3 rounded-lg bg-muted">
          <span className="text-sm font-medium">
            {selectedDocuments.size} selected
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={handleBulkAutoTag}
            disabled={isAutoTagging}
          >
            <Sparkles className="h-4 w-4 mr-2" />
            {isAutoTagging ? "Tagging..." : "Auto-tag"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleBulkEnhance}
            disabled={isEnhancing}
          >
            <Wand2 className="h-4 w-4 mr-2" />
            {isEnhancing ? "Enhancing..." : "Enhance for RAG"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDeleteSelected}
            disabled={deleteDocument.isPending}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedDocuments(new Set())}
          >
            Clear
          </Button>
        </div>
      )}

      {/* Documents */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : viewMode === "list" ? (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="w-12 p-3">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={toggleSelectAll}
                      >
                        {selectedDocuments.size === filteredDocuments.length ? (
                          <CheckSquare className="h-4 w-4" />
                        ) : (
                          <Square className="h-4 w-4" />
                        )}
                      </Button>
                    </th>
                    <th
                      className="text-left p-3 text-sm font-medium cursor-pointer hover:bg-muted"
                      onClick={() => handleSort("name")}
                    >
                      <div className="flex items-center gap-1">
                        Name
                        {sortBy === "name" &&
                          (sortOrder === "asc" ? (
                            <SortAsc className="h-3 w-3" />
                          ) : (
                            <SortDesc className="h-3 w-3" />
                          ))}
                      </div>
                    </th>
                    <th className="text-left p-3 text-sm font-medium">Collection</th>
                    <th
                      className="text-left p-3 text-sm font-medium cursor-pointer hover:bg-muted"
                      onClick={() => handleSort("size")}
                    >
                      <div className="flex items-center gap-1">
                        Size
                        {sortBy === "size" &&
                          (sortOrder === "asc" ? (
                            <SortAsc className="h-3 w-3" />
                          ) : (
                            <SortDesc className="h-3 w-3" />
                          ))}
                      </div>
                    </th>
                    <th
                      className="text-left p-3 text-sm font-medium cursor-pointer hover:bg-muted"
                      onClick={() => handleSort("created_at")}
                    >
                      <div className="flex items-center gap-1">
                        Date
                        {sortBy === "created_at" &&
                          (sortOrder === "asc" ? (
                            <SortAsc className="h-3 w-3" />
                          ) : (
                            <SortDesc className="h-3 w-3" />
                          ))}
                      </div>
                    </th>
                    <th className="w-12 p-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedDocuments.map((doc) => {
                    const FileIcon = getFileIcon(doc.file_type);
                    return (
                      <tr
                        key={doc.id}
                        className="border-b last:border-0 hover:bg-muted/50"
                      >
                        <td className="p-3">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => toggleSelectDocument(doc.id)}
                            aria-label={selectedDocuments.has(doc.id) ? "Deselect document" : "Select document"}
                          >
                            {selectedDocuments.has(doc.id) ? (
                              <CheckSquare className="h-4 w-4" />
                            ) : (
                              <Square className="h-4 w-4" />
                            )}
                          </Button>
                        </td>
                        <td className="p-3">
                          <div className="flex items-center gap-3">
                            <FileIcon className="h-5 w-5 text-muted-foreground shrink-0" />
                            <div>
                              <div className="flex items-center gap-2">
                                <p className="font-medium">{doc.name}</p>
                                {doc.is_enhanced && (
                                  <TooltipProvider>
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 bg-green-50 text-green-700 border-green-200">
                                          <Zap className="h-2.5 w-2.5 mr-0.5" />
                                          Enhanced
                                        </Badge>
                                      </TooltipTrigger>
                                      <TooltipContent side="right" className="max-w-sm">
                                        <p className="font-medium text-sm mb-1">Enhanced Metadata</p>
                                        {doc.enhanced_metadata?.summary_short && (
                                          <p className="text-xs text-muted-foreground mb-1">
                                            {doc.enhanced_metadata.summary_short}
                                          </p>
                                        )}
                                        {doc.enhanced_metadata?.keywords && doc.enhanced_metadata.keywords.length > 0 && (
                                          <p className="text-xs">
                                            <span className="font-medium">Keywords:</span>{" "}
                                            {doc.enhanced_metadata.keywords.slice(0, 5).join(", ")}
                                          </p>
                                        )}
                                        {doc.enhanced_metadata?.model_used && (
                                          <p className="text-[10px] text-muted-foreground mt-1">
                                            Model: {doc.enhanced_metadata.model_used}
                                          </p>
                                        )}
                                      </TooltipContent>
                                    </Tooltip>
                                  </TooltipProvider>
                                )}
                              </div>
                              <p className="text-xs text-muted-foreground">
                                {doc.chunk_count} chunks
                              </p>
                            </div>
                          </div>
                        </td>
                        <td className="p-3">
                          <div className="flex flex-wrap gap-1">
                            {doc.tags && doc.tags.length > 0 ? (
                              doc.tags.map((tag, idx) => (
                                <span
                                  key={idx}
                                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs ${
                                    idx === 0 ? 'bg-primary/10 text-primary' : 'bg-muted'
                                  }`}
                                >
                                  {idx === 0 && <FolderOpen className="h-3 w-3" />}
                                  {tag}
                                </span>
                              ))
                            ) : doc.collection ? (
                              <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted text-xs">
                                <FolderOpen className="h-3 w-3" />
                                {doc.collection}
                              </span>
                            ) : null}
                          </div>
                        </td>
                        <td className="p-3 text-sm text-muted-foreground">
                          {formatFileSize(doc.file_size)}
                        </td>
                        <td className="p-3 text-sm text-muted-foreground">
                          {new Date(doc.created_at).toLocaleDateString()}
                        </td>
                        <td className="p-3">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="More options">
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={() => handleViewDocument(doc.id)}>
                                <Eye className="h-4 w-4 mr-2" />
                                View Details
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => handleDownloadDocument(doc.id, doc.name)}>
                                <Download className="h-4 w-4 mr-2" />
                                Download
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={() => handleAutoTagDocument(doc.id, doc.name)}
                                disabled={isAutoTagging}
                              >
                                <Sparkles className="h-4 w-4 mr-2" />
                                Auto-generate Tags
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => handleOpenEditTags(doc)}>
                                <Edit className="h-4 w-4 mr-2" />
                                Edit Tags
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                className="text-destructive focus:text-destructive"
                                onClick={() => handleDeleteDocument(doc.id, doc.name)}
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {paginatedDocuments.map((doc) => {
            const FileIcon = getFileIcon(doc.file_type);
            return (
              <Card
                key={doc.id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedDocuments.has(doc.id) ? "ring-2 ring-primary" : ""
                }`}
                onClick={() => toggleSelectDocument(doc.id)}
              >
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between">
                    <div className="p-3 rounded-lg bg-muted">
                      <FileIcon className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={(e) => e.stopPropagation()}
                          aria-label="More options"
                        >
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleViewDocument(doc.id); }}>
                          <Eye className="h-4 w-4 mr-2" />
                          View Details
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleDownloadDocument(doc.id, doc.name); }}>
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => { e.stopPropagation(); handleAutoTagDocument(doc.id, doc.name); }}
                          disabled={isAutoTagging}
                        >
                          <Sparkles className="h-4 w-4 mr-2" />
                          Auto-generate Tags
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleOpenEditTags(doc); }}>
                          <Edit className="h-4 w-4 mr-2" />
                          Edit Tags
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          className="text-destructive focus:text-destructive"
                          onClick={(e) => { e.stopPropagation(); handleDeleteDocument(doc.id, doc.name); }}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                  <div className="mt-4">
                    <div className="flex items-center gap-2">
                      <p className="font-medium truncate flex-1">{doc.name}</p>
                      {doc.is_enhanced && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 bg-green-50 text-green-700 border-green-200 shrink-0">
                                <Zap className="h-2.5 w-2.5" />
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="top" className="max-w-sm">
                              <p className="font-medium text-sm mb-1">Enhanced</p>
                              {doc.enhanced_metadata?.summary_short && (
                                <p className="text-xs text-muted-foreground">
                                  {doc.enhanced_metadata.summary_short}
                                </p>
                              )}
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {formatFileSize(doc.file_size)}
                    </p>
                  </div>
                  <div className="flex flex-col gap-2 mt-4 pt-4 border-t">
                    <div className="flex flex-wrap gap-1">
                      {doc.tags && doc.tags.length > 0 ? (
                        doc.tags.slice(0, 3).map((tag, idx) => (
                          <span
                            key={idx}
                            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs ${
                              idx === 0 ? 'bg-primary/10 text-primary' : 'bg-muted'
                            }`}
                          >
                            {idx === 0 && <FolderOpen className="h-2.5 w-2.5" />}
                            {tag}
                          </span>
                        ))
                      ) : doc.collection ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-muted text-xs">
                          <FolderOpen className="h-2.5 w-2.5" />
                          {doc.collection}
                        </span>
                      ) : null}
                      {doc.tags && doc.tags.length > 3 && (
                        <span className="text-xs text-muted-foreground">+{doc.tags.length - 3} more</span>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Pagination Controls */}
      {filteredDocuments.length > 0 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>
              Showing {startIndex + 1}-{Math.min(endIndex, filteredDocuments.length)} of {filteredDocuments.length}
            </span>
            <span className="mx-2">•</span>
            <span>Page size:</span>
            <select
              value={pageSize}
              onChange={(e) => handlePageSizeChange(Number(e.target.value))}
              className="h-8 px-2 rounded-md border bg-background text-sm"
              aria-label="Items per page"
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
              aria-label="First page"
            >
              <ChevronsLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              aria-label="Previous page"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-1 px-2">
              <span className="text-sm">
                Page {currentPage} of {totalPages || 1}
              </span>
            </div>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              aria-label="Next page"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage >= totalPages}
              aria-label="Last page"
            >
              <ChevronsRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && filteredDocuments.length === 0 && (
        <div className="text-center py-12">
          <FileText className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-3" />
          <h3 className="text-lg font-medium">No documents found</h3>
          <p className="text-muted-foreground mt-1">
            {searchQuery
              ? "Try a different search term"
              : "Upload some documents to get started"}
          </p>
        </div>
      )}

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{documents.length}</div>
            <p className="text-sm text-muted-foreground">Total Documents</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {collectionsList.length}
            </div>
            <p className="text-sm text-muted-foreground">Collections</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {formatFileSize(documents.reduce((acc, d) => acc + d.file_size, 0))}
            </div>
            <p className="text-sm text-muted-foreground">Total Size</p>
          </CardContent>
        </Card>
      </div>

      {/* Edit Tags Dialog */}
      <Dialog open={editTagsDialogOpen} onOpenChange={setEditTagsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Tags</DialogTitle>
            <DialogDescription>
              Manage tags for &quot;{editingDocument?.name}&quot;
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {/* Current Tags */}
            <div className="flex flex-wrap gap-2 min-h-[40px] p-2 border rounded-md bg-muted/30">
              {editingTags.length > 0 ? (
                editingTags.map((tag, idx) => (
                  <Badge
                    key={idx}
                    variant={idx === 0 ? "default" : "secondary"}
                    className="flex items-center gap-1 pr-1"
                  >
                    {idx === 0 && <FolderOpen className="h-3 w-3 mr-0.5" />}
                    {tag}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-4 w-4 p-0 hover:bg-transparent"
                      onClick={() => handleRemoveTag(tag)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </Badge>
                ))
              ) : (
                <span className="text-sm text-muted-foreground">No tags yet</span>
              )}
            </div>

            {/* Add New Tag */}
            <div className="flex gap-2">
              <Input
                placeholder="Add a tag..."
                value={newTagInput}
                onChange={(e) => setNewTagInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    handleAddTag();
                  }
                }}
                className="flex-1"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={handleAddTag}
                disabled={!newTagInput.trim()}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {/* Hint */}
            <p className="text-xs text-muted-foreground">
              The first tag is used as the primary collection. Press Enter or click + to add.
            </p>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setEditTagsDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleSaveTags} disabled={isSavingTags}>
              {isSavingTags ? "Saving..." : "Save Tags"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
