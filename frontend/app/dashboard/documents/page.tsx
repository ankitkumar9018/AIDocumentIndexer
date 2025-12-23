"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { getErrorMessage } from "@/lib/errors";
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
  Star,
  Clock,
  Bookmark,
  X,
  Plus,
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

// File size filter options
const FILE_SIZE_OPTIONS = [
  { value: "", label: "Any Size", minBytes: 0, maxBytes: Infinity },
  { value: "small", label: "< 100 KB", minBytes: 0, maxBytes: 100 * 1024 },
  { value: "medium", label: "100 KB - 1 MB", minBytes: 100 * 1024, maxBytes: 1024 * 1024 },
  { value: "large", label: "1 MB - 10 MB", minBytes: 1024 * 1024, maxBytes: 10 * 1024 * 1024 },
  { value: "xlarge", label: "> 10 MB", minBytes: 10 * 1024 * 1024, maxBytes: Infinity },
];

// Date range filter options
const DATE_RANGE_OPTIONS = [
  { value: "", label: "Any Time" },
  { value: "today", label: "Today" },
  { value: "week", label: "Last 7 Days" },
  { value: "month", label: "Last 30 Days" },
  { value: "quarter", label: "Last 90 Days" },
  { value: "year", label: "Last Year" },
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
  const [selectedSizeFilter, setSelectedSizeFilter] = useState<string>("");
  const [selectedDateRange, setSelectedDateRange] = useState<string>("");
  const [autoTagEnabled, setAutoTagEnabled] = useState(false);

  // Favorites and recently viewed
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [recentlyViewed, setRecentlyViewed] = useState<string[]>([]);
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);

  // Load favorites and recently viewed from localStorage
  useEffect(() => {
    try {
      const savedFavorites = localStorage.getItem("document_favorites");
      if (savedFavorites) {
        setFavorites(new Set(JSON.parse(savedFavorites)));
      }
      const savedRecent = localStorage.getItem("document_recent");
      if (savedRecent) {
        setRecentlyViewed(JSON.parse(savedRecent));
      }
    } catch (error) {
      console.error("Failed to load favorites/recent from localStorage:", error);
    }
  }, []);

  // Save favorites to localStorage
  const saveFavorites = useCallback((newFavorites: Set<string>) => {
    try {
      localStorage.setItem("document_favorites", JSON.stringify([...newFavorites]));
    } catch (error) {
      console.error("Failed to save favorites:", error);
    }
  }, []);

  // Save recently viewed to localStorage
  const saveRecentlyViewed = useCallback((recent: string[]) => {
    try {
      localStorage.setItem("document_recent", JSON.stringify(recent));
    } catch (error) {
      console.error("Failed to save recent:", error);
    }
  }, []);

  // Toggle favorite
  const toggleFavorite = useCallback((docId: string) => {
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
        toast.info("Removed from favorites");
      } else {
        next.add(docId);
        toast.success("Added to favorites");
      }
      saveFavorites(next);
      return next;
    });
  }, [saveFavorites]);

  // Add to recently viewed
  const addToRecentlyViewed = useCallback((docId: string) => {
    setRecentlyViewed((prev) => {
      const filtered = prev.filter((id) => id !== docId);
      const updated = [docId, ...filtered].slice(0, 10); // Keep last 10
      saveRecentlyViewed(updated);
      return updated;
    });
  }, [saveRecentlyViewed]);

  // Saved searches
  interface SavedSearch {
    id: string;
    name: string;
    query: string;
    fileType: string;
    sizeFilter: string;
    dateRange: string;
    collection: string | null;
  }

  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [showSavedSearches, setShowSavedSearches] = useState(false);

  // Load saved searches from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem("document_saved_searches");
      if (saved) {
        setSavedSearches(JSON.parse(saved));
      }
    } catch (error) {
      console.error("Failed to load saved searches:", error);
    }
  }, []);

  // Save a new search
  const handleSaveSearch = useCallback(() => {
    const hasFilters = searchQuery || selectedFileType || selectedSizeFilter || selectedDateRange || selectedCollection;
    if (!hasFilters) {
      toast.error("No filters to save", { description: "Apply some filters first" });
      return;
    }

    const name = prompt("Enter a name for this search:");
    if (!name) return;

    const newSearch: SavedSearch = {
      id: Date.now().toString(),
      name,
      query: searchQuery,
      fileType: selectedFileType,
      sizeFilter: selectedSizeFilter,
      dateRange: selectedDateRange,
      collection: selectedCollection,
    };

    setSavedSearches((prev) => {
      const updated = [...prev, newSearch];
      localStorage.setItem("document_saved_searches", JSON.stringify(updated));
      return updated;
    });

    toast.success("Search saved", { description: `"${name}" saved to your searches` });
  }, [searchQuery, selectedFileType, selectedSizeFilter, selectedDateRange, selectedCollection]);

  // Apply a saved search
  const handleApplySavedSearch = useCallback((search: SavedSearch) => {
    setSearchQuery(search.query);
    setSelectedFileType(search.fileType);
    setSelectedSizeFilter(search.sizeFilter);
    setSelectedDateRange(search.dateRange);
    setSelectedCollection(search.collection);
    setCurrentPage(1);
    setShowSavedSearches(false);
    toast.success("Search applied", { description: `Applied "${search.name}"` });
  }, []);

  // Delete a saved search
  const handleDeleteSavedSearch = useCallback((searchId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSavedSearches((prev) => {
      const updated = prev.filter((s) => s.id !== searchId);
      localStorage.setItem("document_saved_searches", JSON.stringify(updated));
      return updated;
    });
    toast.info("Search deleted");
  }, []);

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

  // Filter documents by search query, file type, size, and date
  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      // Search filter
      const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase());

      // File type filter
      const matchesFileType = !selectedFileType || doc.file_type?.toLowerCase().includes(selectedFileType.toLowerCase());

      // Size filter
      let matchesSize = true;
      if (selectedSizeFilter) {
        const sizeOption = FILE_SIZE_OPTIONS.find((o) => o.value === selectedSizeFilter);
        if (sizeOption) {
          matchesSize = doc.file_size >= sizeOption.minBytes && doc.file_size < sizeOption.maxBytes;
        }
      }

      // Date range filter
      let matchesDate = true;
      if (selectedDateRange) {
        const now = new Date();
        const docDate = new Date(doc.created_at);
        const dayMs = 24 * 60 * 60 * 1000;

        switch (selectedDateRange) {
          case "today":
            matchesDate = now.toDateString() === docDate.toDateString();
            break;
          case "week":
            matchesDate = (now.getTime() - docDate.getTime()) < 7 * dayMs;
            break;
          case "month":
            matchesDate = (now.getTime() - docDate.getTime()) < 30 * dayMs;
            break;
          case "quarter":
            matchesDate = (now.getTime() - docDate.getTime()) < 90 * dayMs;
            break;
          case "year":
            matchesDate = (now.getTime() - docDate.getTime()) < 365 * dayMs;
            break;
        }
      }

      // Favorites filter
      const matchesFavorites = !showFavoritesOnly || favorites.has(doc.id);

      return matchesSearch && matchesFileType && matchesSize && matchesDate && matchesFavorites;
    });
  }, [documents, searchQuery, selectedFileType, selectedSizeFilter, selectedDateRange, showFavoritesOnly, favorites]);

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

  const handleSizeFilterChange = (value: string) => {
    setSelectedSizeFilter(value);
    setCurrentPage(1);
  };

  const handleDateRangeChange = (value: string) => {
    setSelectedDateRange(value);
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
    } catch (error) {
      console.error("Delete failed:", error);
      toast.error("Failed to delete document", {
        description: getErrorMessage(error, "An error occurred"),
      });
    }
  };

  const handleDownloadDocument = async (docId: string, docName: string) => {
    try {
      await api.downloadDocument(docId, docName);
      toast.success("Download started", {
        description: `Downloading "${docName}"...`,
      });
    } catch (error) {
      console.error("Download failed:", error);
      toast.error("Failed to download document", {
        description: getErrorMessage(error, "An error occurred"),
      });
    }
  };

  const handleViewDocument = (docId: string) => {
    // Track recently viewed
    addToRecentlyViewed(docId);
    // Navigate to document detail view
    window.location.href = `/dashboard/documents/${docId}`;
  };

  const [isAutoTagging, setIsAutoTagging] = useState(false);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

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
    } catch (error) {
      console.error("Auto-tag failed:", error);
      toast.error("Failed to auto-tag document", {
        description: getErrorMessage(error, "An error occurred"),
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
    } catch (error) {
      console.error("Bulk auto-tag failed:", error);
      toast.error("Failed to auto-tag documents", {
        description: getErrorMessage(error, "An error occurred"),
      });
    } finally {
      setIsAutoTagging(false);
    }
  };

  const handleBulkDownload = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      setIsDownloading(true);
      await api.bulkDownloadDocuments(Array.from(selectedDocuments));
      toast.success("Download started", {
        description: `Downloading ${selectedDocuments.size} documents as ZIP...`,
      });
    } catch (error) {
      console.error("Bulk download failed:", error);
      toast.error("Failed to download documents", {
        description: getErrorMessage(error, "An error occurred"),
      });
    } finally {
      setIsDownloading(false);
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
    } catch (error) {
      console.error("Bulk enhance failed:", error);
      toast.error("Failed to enhance documents", {
        description: getErrorMessage(error, "An error occurred"),
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
    } catch (error) {
      console.error("Enhance all failed:", error);
      toast.error("Failed to enhance documents", {
        description: getErrorMessage(error, "An error occurred"),
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
    } catch (error) {
      toast.error("Failed to update LLM", {
        description: getErrorMessage(error, "An error occurred"),
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
    } catch (error) {
      console.error("Failed to update tags:", error);
      toast.error("Failed to update tags", {
        description: getErrorMessage(error, "An error occurred"),
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
        {/* Search with Saved Searches */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="pl-9 pr-20"
            aria-label="Search documents"
          />
          <div className="absolute right-1 top-1/2 -translate-y-1/2 flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handleSaveSearch}
              title="Save current search"
            >
              <Bookmark className="h-3.5 w-3.5" />
            </Button>
            {savedSearches.length > 0 && (
              <Button
                variant={showSavedSearches ? "secondary" : "ghost"}
                size="icon"
                className="h-7 w-7"
                onClick={() => setShowSavedSearches(!showSavedSearches)}
                title="Saved searches"
              >
                <ChevronDown className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>

          {/* Saved Searches Dropdown */}
          {showSavedSearches && savedSearches.length > 0 && (
            <Card className="absolute z-10 top-full mt-1 w-full p-2 shadow-lg">
              <div className="flex items-center justify-between px-2 pb-2 border-b mb-2">
                <span className="text-sm font-medium">Saved Searches</span>
                <span className="text-xs text-muted-foreground">{savedSearches.length} saved</span>
              </div>
              <div className="max-h-48 overflow-y-auto space-y-1">
                {savedSearches.map((search) => (
                  <div
                    key={search.id}
                    className="flex items-center justify-between p-2 rounded hover:bg-muted cursor-pointer"
                    onClick={() => handleApplySavedSearch(search)}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <Bookmark className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium truncate">{search.name}</p>
                        <p className="text-xs text-muted-foreground truncate">
                          {[
                            search.query && `"${search.query}"`,
                            search.fileType && search.fileType,
                            search.sizeFilter && search.sizeFilter,
                            search.dateRange && search.dateRange,
                            search.collection && search.collection,
                          ].filter(Boolean).join(" • ") || "No filters"}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={(e) => handleDeleteSavedSearch(search.id, e)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </Card>
          )}
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

          {/* Size Filter */}
          <select
            value={selectedSizeFilter}
            onChange={(e) => handleSizeFilterChange(e.target.value)}
            className="h-10 px-3 rounded-md border bg-background text-sm"
            aria-label="Filter by file size"
          >
            {FILE_SIZE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          {/* Date Range Filter */}
          <select
            value={selectedDateRange}
            onChange={(e) => handleDateRangeChange(e.target.value)}
            className="h-10 px-3 rounded-md border bg-background text-sm"
            aria-label="Filter by date"
          >
            {DATE_RANGE_OPTIONS.map((option) => (
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

          {/* Favorites Toggle */}
          <Button
            variant={showFavoritesOnly ? "default" : "outline"}
            size="sm"
            className="h-10"
            onClick={() => {
              setShowFavoritesOnly(!showFavoritesOnly);
              setCurrentPage(1);
            }}
            aria-label="Show favorites only"
          >
            <Star className={`h-4 w-4 mr-1 ${showFavoritesOnly ? "fill-current" : ""}`} />
            Favorites {favorites.size > 0 && `(${favorites.size})`}
          </Button>

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

      {/* Recently Viewed Section */}
      {recentlyViewed.length > 0 && !showFavoritesOnly && !searchQuery && (
        <Card className="bg-muted/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Recently Viewed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-3 overflow-x-auto pb-2">
              {recentlyViewed
                .slice(0, 5)
                .map((docId) => {
                  const doc = documents.find((d) => d.id === docId);
                  if (!doc) return null;
                  const FileIcon = getFileIcon(doc.file_type);
                  return (
                    <div
                      key={docId}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg border bg-background hover:bg-muted cursor-pointer shrink-0"
                      onClick={() => handleViewDocument(doc.id)}
                    >
                      <FileIcon className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium truncate max-w-[150px]">
                        {doc.name}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 shrink-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleFavorite(doc.id);
                        }}
                        aria-label={favorites.has(doc.id) ? "Remove from favorites" : "Add to favorites"}
                      >
                        <Star
                          className={`h-3 w-3 ${
                            favorites.has(doc.id)
                              ? "fill-yellow-400 text-yellow-400"
                              : "text-muted-foreground"
                          }`}
                        />
                      </Button>
                    </div>
                  );
                })
                .filter(Boolean)}
            </div>
          </CardContent>
        </Card>
      )}

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
          <Button
            variant="outline"
            size="sm"
            onClick={handleBulkDownload}
            disabled={isDownloading}
          >
            <Download className="h-4 w-4 mr-2" />
            {isDownloading ? "Downloading..." : "Download"}
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
                                {favorites.has(doc.id) && (
                                  <Star className="h-4 w-4 fill-yellow-400 text-yellow-400 shrink-0" />
                                )}
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
                              <DropdownMenuItem onClick={() => toggleFavorite(doc.id)}>
                                <Star className={`h-4 w-4 mr-2 ${favorites.has(doc.id) ? "fill-yellow-400 text-yellow-400" : ""}`} />
                                {favorites.has(doc.id) ? "Remove from Favorites" : "Add to Favorites"}
                              </DropdownMenuItem>
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
                        <DropdownMenuItem onClick={(e) => { e.stopPropagation(); toggleFavorite(doc.id); }}>
                          <Star className={`h-4 w-4 mr-2 ${favorites.has(doc.id) ? "fill-yellow-400 text-yellow-400" : ""}`} />
                          {favorites.has(doc.id) ? "Remove from Favorites" : "Add to Favorites"}
                        </DropdownMenuItem>
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
