"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { getErrorMessage } from "@/lib/errors";
import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors,
  closestCenter,
} from "@dnd-kit/core";
import { useDraggable, useDroppable } from "@dnd-kit/core";
import { CSS } from "@dnd-kit/utilities";
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
  PanelLeftClose,
  PanelLeft,
  FolderTree as FolderTreeIcon,
  GripVertical,
  Database,
  AlertCircle,
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
  useVisionStatus,
  useReprocessDocument,
  Document,
  api,
} from "@/lib/api";
import { ReanalyzeImagesModal } from "@/components/documents/reanalyze-images-modal";
import { Wand2, Zap, Settings, Info, Bot, Tag, FolderInput, ImageOff, Images } from "lucide-react";
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
import {
  DocumentListSkeleton,
  DocumentGridSkeleton,
  DocumentStatsSkeleton,
} from "@/components/skeletons";
import { FolderTree } from "@/components/folder-tree";
import { FolderSelector } from "@/components/folder-selector";
import { SavedSearchesPanel, type SearchFilters } from "@/components/saved-searches-panel";
import { SmartFolders, DocumentInsights } from "@/components/documents/smart-folders";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { UploadedDocumentPreview } from "@/components/preview/UploadedDocumentPreview";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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

  // Folder navigation
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Favorites and recently viewed
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [recentlyViewed, setRecentlyViewed] = useState<string[]>([]);
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [showImagesPending, setShowImagesPending] = useState(false);

  // Delete confirmation dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<{ id: string; name: string } | null>(null);
  const [hardDeleteEnabled, setHardDeleteEnabled] = useState(false);

  // Move to folder dialog
  const [moveDialogOpen, setMoveDialogOpen] = useState(false);
  const [moveFolderId, setMoveFolderId] = useState<string | null>(null);
  const [isMoving, setIsMoving] = useState(false);

  // Document preview panel
  const [previewDocument, setPreviewDocument] = useState<any>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  // Image analysis modal
  const [imageAnalysisDocument, setImageAnalysisDocument] = useState<any>(null);
  const [isImageAnalysisOpen, setIsImageAnalysisOpen] = useState(false);

  // Drag and drop state
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [dragOverFolderId, setDragOverFolderId] = useState<string | null>(null);

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

  // Handle applying saved search from the SavedSearchesPanel
  const handleApplySavedSearch = useCallback((filters: SearchFilters) => {
    setSearchQuery(filters.query || "");
    setSelectedCollection(filters.collection || null);
    setSelectedFolderId(filters.folder_id || null);
    setSelectedDateRange("");  // Convert date_from/date_to to range if needed
    setSelectedFileType(filters.file_types?.[0] || "");
    setCurrentPage(1);
  }, []);

  // Get current search filters for the SavedSearchesPanel
  const currentFilters: SearchFilters = useMemo(() => ({
    query: searchQuery,
    collection: selectedCollection,
    folder_id: selectedFolderId,
    date_from: null,
    date_to: null,
    file_types: selectedFileType ? [selectedFileType] : null,
    search_mode: "hybrid",
  }), [searchQuery, selectedCollection, selectedFolderId, selectedFileType]);

  // Queries - only fetch when authenticated
  const { data: documentsData, isLoading, refetch } = useDocuments({
    page: currentPage,
    page_size: pageSize,
    collection: selectedCollection || undefined,
    folder_id: selectedFolderId || undefined,
    include_subfolders: true,
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
  const reprocessDocument = useReprocessDocument();

  // Use real documents from API (no mock fallback)
  const documents: Document[] = documentsData?.documents ?? [];

  // DnD sensors - require minimum distance before starting drag to allow clicking
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8, // 8px minimum movement before drag starts
      },
    })
  );

  // Handle drag start
  const handleDragStart = useCallback((event: DragStartEvent) => {
    const { active } = event;
    setActiveDocId(active.id as string);
  }, []);

  // Handle drag end - move document to folder
  const handleDragEnd = useCallback(async (event: DragEndEvent) => {
    const { active, over } = event;
    setActiveDocId(null);
    setDragOverFolderId(null);

    if (!over) return;

    const documentId = active.id as string;
    const folderId = String(over.id);

    // Skip if not a folder target
    if (!folderId.startsWith("folder-")) return;

    const targetFolderId = folderId === "folder-root" ? null : folderId.replace("folder-", "");

    try {
      setIsMoving(true);
      const doc = documents.find((d) => d.id === documentId);
      const result = await api.moveDocument(documentId, targetFolderId);
      if (result.success) {
        toast.success("Document moved", {
          description: `Moved "${doc?.name || "document"}" to ${targetFolderId ? "folder" : "root level"}`,
        });
        refetch();
      }
    } catch (error) {
      console.error("Failed to move document:", error);
      toast.error("Failed to move document", {
        description: getErrorMessage(error, "An error occurred"),
      });
    } finally {
      setIsMoving(false);
    }
  }, [documents, refetch]);

  // Handle drag over folder
  const handleDragOver = useCallback((event: DragEndEvent) => {
    if (event.over && String(event.over.id).startsWith("folder-")) {
      setDragOverFolderId(String(event.over.id));
    } else {
      setDragOverFolderId(null);
    }
  }, []);

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

      // Images pending filter
      const matchesImagesPending = !showImagesPending || (
        (doc as any).images_extracted_count > 0 &&
        (doc as any).image_analysis_status !== 'completed'
      );

      return matchesSearch && matchesFileType && matchesSize && matchesDate && matchesFavorites && matchesImagesPending;
    });
  }, [documents, searchQuery, selectedFileType, selectedSizeFilter, selectedDateRange, showFavoritesOnly, favorites, showImagesPending]);

  // Pagination calculations - use server-side total count
  const totalCount = documentsData?.total ?? 0;
  const totalPages = Math.ceil(totalCount / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, totalCount);
  // Documents are already paginated by the server, no need to slice again
  const paginatedDocuments = filteredDocuments;

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
      await deleteDocument.mutateAsync({ id, hardDelete: false });
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

  // Open delete confirmation dialog
  const handleDeleteDocument = (docId: string, docName: string) => {
    setDocumentToDelete({ id: docId, name: docName });
    setHardDeleteEnabled(false); // Reset hard delete option
    setDeleteDialogOpen(true);
  };

  // Confirm and execute delete
  const confirmDeleteDocument = async () => {
    if (!documentToDelete) return;

    try {
      await deleteDocument.mutateAsync({
        id: documentToDelete.id,
        hardDelete: hardDeleteEnabled,
      });
      toast.success(hardDeleteEnabled ? "Document permanently deleted" : "Document deleted", {
        description: hardDeleteEnabled
          ? `"${documentToDelete.name}" has been permanently removed.`
          : `"${documentToDelete.name}" has been deleted (can be recovered by admin).`,
      });
      refetch();
    } catch (error) {
      console.error("Delete failed:", error);
      toast.error("Failed to delete document", {
        description: getErrorMessage(error, "An error occurred"),
      });
    } finally {
      setDeleteDialogOpen(false);
      setDocumentToDelete(null);
      setHardDeleteEnabled(false);
    }
  };

  // Reprocess a document (re-run processing pipeline)
  const handleReprocessDocument = async (docId: string, docName: string) => {
    try {
      toast.loading(`Reprocessing "${docName}"...`, { id: `reprocess-${docId}` });
      await reprocessDocument.mutateAsync(docId);
      toast.success("Document reprocessing started", {
        id: `reprocess-${docId}`,
        description: `"${docName}" is being reprocessed. This may take a moment.`,
      });
      refetch();
    } catch (error) {
      console.error("Reprocess failed:", error);
      toast.error("Failed to reprocess document", {
        id: `reprocess-${docId}`,
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
    // Open preview panel instead of navigating
    const doc = filteredDocuments.find(d => d.id === docId);
    if (doc) {
      setPreviewDocument(doc);
      setIsPreviewOpen(true);
    }
  };

  const handleOpenFullView = (docId: string) => {
    window.location.href = `/dashboard/documents/${docId}`;
  };

  // Generate embeddings for a specific document
  const handleGenerateEmbeddings = async (docId: string) => {
    try {
      toast.info("Starting embedding generation...");
      const response = await api.post(`/embeddings/generate/${docId}`);
      if (response.data.job_id) {
        toast.success("Embedding generation started", {
          description: "You can check the status in the embeddings dashboard.",
        });
        // Refresh documents after a delay to show updated status
        setTimeout(() => {
          refetch();
        }, 3000);
      }
    } catch (error) {
      toast.error("Failed to start embedding generation", {
        description: getErrorMessage(error),
      });
    }
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

    if (selectedDocuments.size > 20) {
      toast.error("Too many documents selected", {
        description: "Maximum 20 documents per bulk auto-tag request. Please select fewer documents.",
      });
      return;
    }

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

  const handleBulkMove = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      setIsMoving(true);
      const result = await api.bulkMoveDocuments(
        Array.from(selectedDocuments),
        moveFolderId
      );
      if (result.moved_count > 0) {
        toast.success("Documents moved", {
          description: `Moved ${result.moved_count} document${result.moved_count > 1 ? "s" : ""} to ${moveFolderId ? "folder" : "root"}`,
        });
      }
      if (result.failed_count > 0) {
        toast.error(`Failed to move ${result.failed_count} document${result.failed_count > 1 ? "s" : ""}`);
      }
      setSelectedDocuments(new Set());
      setMoveDialogOpen(false);
      setMoveFolderId(null);
      refetch();
    } catch (error) {
      console.error("Bulk move failed:", error);
      toast.error("Failed to move documents", {
        description: getErrorMessage(error, "An error occurred"),
      });
    } finally {
      setIsMoving(false);
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

  // Draggable document row component
  const DraggableDocumentRow = ({ doc, children }: { doc: Document; children: React.ReactNode }) => {
    const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
      id: doc.id,
    });

    const style = {
      transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
      opacity: isDragging ? 0.5 : 1,
    };

    return (
      <tr
        ref={setNodeRef}
        style={style}
        className={`border-b last:border-0 hover:bg-muted/50 ${isDragging ? "shadow-lg" : ""}`}
      >
        <td className="p-3 cursor-grab" {...attributes} {...listeners}>
          <GripVertical className="h-4 w-4 text-muted-foreground hover:text-foreground" />
        </td>
        {children}
      </tr>
    );
  };

  // Draggable document card component for grid view
  const DraggableDocumentCard = ({ doc, children }: { doc: Document; children: React.ReactNode }) => {
    const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
      id: doc.id,
    });

    const style = {
      transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
      opacity: isDragging ? 0.5 : 1,
    };

    return (
      <div ref={setNodeRef} style={style} className={isDragging ? "shadow-lg z-50" : ""}>
        <div
          className="absolute top-2 left-2 cursor-grab p-1 rounded hover:bg-muted"
          {...attributes}
          {...listeners}
        >
          <GripVertical className="h-4 w-4 text-muted-foreground" />
        </div>
        {children}
      </div>
    );
  };

  // Get the currently dragged document for the overlay
  const activeDraggedDoc = activeDocId ? documents.find((d) => d.id === activeDocId) : null;

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

      {/* AI-Powered Smart Folders Section */}
      <div className="grid gap-6 lg:grid-cols-2">
        <SmartFolders />
        <DocumentInsights />
      </div>

      {/* Main Content with Sidebar - wrapped in DndContext for drag-drop */}
      <DndContext
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        onDragOver={handleDragOver}
      >
      <div className="flex gap-6">
        {/* Sidebar - Folder Tree & Saved Searches */}
        {sidebarOpen && (
          <div className="w-64 shrink-0 space-y-4">
            <Card className="p-3">
              <FolderTree
                selectedFolderId={selectedFolderId}
                onFolderSelect={(folderId) => {
                  setSelectedFolderId(folderId);
                  setCurrentPage(1);
                }}
                dragOverFolderId={dragOverFolderId}
                isDropTarget={true}
              />
            </Card>
            <Card className="p-3">
              <SavedSearchesPanel
                currentFilters={currentFilters}
                onApplySearch={handleApplySavedSearch}
                collections={collections?.collections}
              />
            </Card>
          </div>
        )}

        {/* Main Content Area */}
        <div className="flex-1 min-w-0 space-y-4">
          {/* Toolbar */}
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Sidebar Toggle + Search */}
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
              >
                {sidebarOpen ? (
                  <PanelLeftClose className="h-4 w-4" />
                ) : (
                  <PanelLeft className="h-4 w-4" />
                )}
              </Button>
              <div className="relative flex-1 min-w-0">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search documents..."
                  value={searchQuery}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  className="pl-9"
                  aria-label="Search documents"
                />
              </div>
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

              {/* Images Pending Toggle */}
              <Button
                variant={showImagesPending ? "default" : "outline"}
                size="sm"
                className="h-10"
                onClick={() => {
                  setShowImagesPending(!showImagesPending);
                  setCurrentPage(1);
                }}
                aria-label="Show documents with pending image analysis"
              >
                <ImageOff className={`h-4 w-4 mr-1 ${showImagesPending ? "text-orange-300" : "text-orange-500"}`} />
                Images Pending
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
                              ? "fill-yellow-400 text-yellow-400 dark:fill-yellow-300 dark:text-yellow-300"
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

      {/* Floating Batch Action Bar */}
      {selectedDocuments.size > 0 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 animate-in slide-in-from-bottom-4 duration-300">
          <div className="flex items-center gap-3 px-5 py-3 rounded-xl border bg-background/80 backdrop-blur-lg shadow-lg">
            <span className="text-sm font-medium whitespace-nowrap">
              {selectedDocuments.size} selected
            </span>
            <div className="w-px h-6 bg-border" />
            <Button
              variant="outline"
              size="sm"
              onClick={handleBulkAutoTag}
              disabled={isAutoTagging}
            >
              <Sparkles className="h-4 w-4 mr-1.5" />
              {isAutoTagging ? "Tagging..." : "Auto-tag"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleBulkEnhance}
              disabled={isEnhancing}
            >
              <Wand2 className="h-4 w-4 mr-1.5" />
              {isEnhancing ? "Enhancing..." : "Enhance"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setMoveDialogOpen(true)}
              disabled={isMoving}
            >
              <FolderInput className="h-4 w-4 mr-1.5" />
              {isMoving ? "Moving..." : "Move"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleBulkDownload}
              disabled={isDownloading}
            >
              <Download className="h-4 w-4 mr-1.5" />
              {isDownloading ? "Downloading..." : "Download"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="text-destructive hover:text-destructive"
              onClick={handleDeleteSelected}
              disabled={deleteDocument.isPending}
            >
              <Trash2 className="h-4 w-4 mr-1.5" />
              Delete
            </Button>
            <div className="w-px h-6 bg-border" />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedDocuments(new Set())}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Move to Folder Dialog */}
      <Dialog open={moveDialogOpen} onOpenChange={setMoveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Move Documents to Folder</DialogTitle>
            <DialogDescription>
              Select a folder to move {selectedDocuments.size} document{selectedDocuments.size > 1 ? "s" : ""} to.
              Select &quot;Root&quot; to move to the root level.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <FolderSelector
              value={moveFolderId}
              onChange={setMoveFolderId}
              placeholder="Select folder (or leave empty for root)"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setMoveDialogOpen(false);
                setMoveFolderId(null);
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleBulkMove}
              disabled={isMoving}
            >
              {isMoving ? "Moving..." : "Move"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Documents */}
      {isLoading ? (
        viewMode === "list" ? (
          <DocumentListSkeleton count={pageSize > 10 ? 10 : pageSize} />
        ) : (
          <DocumentGridSkeleton count={8} />
        )
      ) : viewMode === "list" ? (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="w-8 p-3" title="Drag to move">
                      {/* Drag handle column header */}
                    </th>
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
                      <DraggableDocumentRow key={doc.id} doc={doc}>
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
                                  <Star className="h-4 w-4 fill-yellow-400 text-yellow-400 dark:fill-yellow-300 dark:text-yellow-300 shrink-0" />
                                )}
                                <p className="font-medium">{doc.name}</p>
                                {doc.is_enhanced && (
                                  <TooltipProvider>
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 bg-green-50 text-green-700 border-green-200 dark:bg-green-950/30 dark:text-green-300 dark:border-green-800">
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
                                {/* Image Analysis Status Badge */}
                                {(doc as any).images_extracted_count > 0 && (
                                  <TooltipProvider>
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <Badge
                                          variant="outline"
                                          className={`text-[10px] px-1.5 py-0 h-4 cursor-pointer ${
                                            (doc as any).image_analysis_status === 'completed'
                                              ? 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/30 dark:text-blue-300 dark:border-blue-800'
                                              : 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950/30 dark:text-orange-300 dark:border-orange-800'
                                          }`}
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            setImageAnalysisDocument(doc);
                                            setIsImageAnalysisOpen(true);
                                          }}
                                        >
                                          <Images className="h-2.5 w-2.5 mr-0.5" />
                                          {(doc as any).images_analyzed_count || 0}/{(doc as any).images_extracted_count}
                                        </Badge>
                                      </TooltipTrigger>
                                      <TooltipContent side="right">
                                        <p className="font-medium text-sm mb-1">Image Analysis</p>
                                        <p className="text-xs text-muted-foreground">
                                          {(doc as any).image_analysis_status === 'completed'
                                            ? `${(doc as any).images_analyzed_count} images analyzed`
                                            : `${(doc as any).images_extracted_count - ((doc as any).images_analyzed_count || 0)} images pending`}
                                        </p>
                                        <p className="text-[10px] text-muted-foreground mt-1">Click to analyze</p>
                                      </TooltipContent>
                                    </Tooltip>
                                  </TooltipProvider>
                                )}
                              </div>
                              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                <span>{doc.chunk_count} chunks</span>
                                {doc.chunk_count > 0 && (
                                  <TooltipProvider>
                                    <Tooltip>
                                      <TooltipTrigger asChild>
                                        <span
                                          className={`flex items-center gap-1 cursor-pointer ${
                                            doc.has_all_embeddings
                                              ? "text-green-600 dark:text-green-400"
                                              : "text-amber-600 dark:text-amber-400"
                                          }`}
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            if (!doc.has_all_embeddings) {
                                              handleGenerateEmbeddings(doc.id);
                                            }
                                          }}
                                        >
                                          {doc.has_all_embeddings ? (
                                            <Database className="h-3 w-3" />
                                          ) : (
                                            <AlertCircle className="h-3 w-3" />
                                          )}
                                          {doc.embedding_coverage.toFixed(0)}%
                                        </span>
                                      </TooltipTrigger>
                                      <TooltipContent>
                                        <p className="font-medium">
                                          {doc.has_all_embeddings
                                            ? "All embeddings generated"
                                            : `${doc.embedding_count}/${doc.chunk_count} chunks embedded`}
                                        </p>
                                        {!doc.has_all_embeddings && (
                                          <p className="text-[10px] text-muted-foreground mt-1">
                                            Click to generate missing embeddings
                                          </p>
                                        )}
                                      </TooltipContent>
                                    </Tooltip>
                                  </TooltipProvider>
                                )}
                              </div>
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
                                <Star className={`h-4 w-4 mr-2 ${favorites.has(doc.id) ? "fill-yellow-400 text-yellow-400 dark:fill-yellow-300 dark:text-yellow-300" : ""}`} />
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
                              {(doc as any).images_extracted_count > 0 && (
                                <DropdownMenuItem
                                  onClick={() => {
                                    setImageAnalysisDocument(doc);
                                    setIsImageAnalysisOpen(true);
                                  }}
                                >
                                  <Images className="h-4 w-4 mr-2" />
                                  Analyze Images
                                </DropdownMenuItem>
                              )}
                              {!doc.has_all_embeddings && doc.chunk_count > 0 && (
                                <DropdownMenuItem
                                  onClick={() => handleGenerateEmbeddings(doc.id)}
                                >
                                  <Database className="h-4 w-4 mr-2" />
                                  Generate Embeddings
                                </DropdownMenuItem>
                              )}
                              <DropdownMenuItem
                                onClick={() => handleReprocessDocument(doc.id, doc.name)}
                                disabled={reprocessDocument.isPending}
                              >
                                <RefreshCw className={`h-4 w-4 mr-2 ${reprocessDocument.isPending ? "animate-spin" : ""}`} />
                                Reprocess Document
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
                      </DraggableDocumentRow>
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
                          <Star className={`h-4 w-4 mr-2 ${favorites.has(doc.id) ? "fill-yellow-400 text-yellow-400 dark:fill-yellow-300 dark:text-yellow-300" : ""}`} />
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
                        {(doc as any).images_extracted_count > 0 && (
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              setImageAnalysisDocument(doc);
                              setIsImageAnalysisOpen(true);
                            }}
                          >
                            <Images className="h-4 w-4 mr-2" />
                            Analyze Images
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuItem
                          onClick={(e) => { e.stopPropagation(); handleReprocessDocument(doc.id, doc.name); }}
                          disabled={reprocessDocument.isPending}
                        >
                          <RefreshCw className={`h-4 w-4 mr-2 ${reprocessDocument.isPending ? "animate-spin" : ""}`} />
                          Reprocess Document
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
                              <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 bg-green-50 text-green-700 border-green-200 dark:bg-green-950/30 dark:text-green-300 dark:border-green-800 shrink-0">
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
                      {(doc as any).images_extracted_count > 0 && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge
                                variant="outline"
                                className={`text-[10px] px-1.5 py-0 h-4 shrink-0 cursor-pointer ${
                                  (doc as any).image_analysis_status === 'completed'
                                    ? 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/30 dark:text-blue-300 dark:border-blue-800'
                                    : 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950/30 dark:text-orange-300 dark:border-orange-800'
                                }`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setImageAnalysisDocument(doc);
                                  setIsImageAnalysisOpen(true);
                                }}
                              >
                                <Images className="h-2.5 w-2.5" />
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              <p className="text-xs">
                                {(doc as any).images_analyzed_count || 0}/{(doc as any).images_extracted_count} images
                              </p>
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
      {totalCount > 0 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>
              Showing {startIndex + 1}-{endIndex} of {totalCount}
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
      {!isLoading && totalCount === 0 && (
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
            <div className="text-2xl font-bold">{totalCount}</div>
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
        {/* End of Main Content Area */}
        </div>
      {/* End of Sidebar + Content flex container */}
      </div>

      {/* Drag Overlay - shows preview of dragged document */}
      <DragOverlay>
        {activeDraggedDoc && (
          <div className="px-4 py-2 bg-background border rounded-lg shadow-lg flex items-center gap-2">
            <GripVertical className="h-4 w-4 text-muted-foreground" />
            <File className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium text-sm">{activeDraggedDoc.name}</span>
          </div>
        )}
      </DragOverlay>
      </DndContext>

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

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Document</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{documentToDelete?.name}&quot;?
            </DialogDescription>
          </DialogHeader>

          <div className="py-4 space-y-4">
            <div className="p-3 bg-muted rounded-lg text-sm">
              <p className="font-medium mb-2">Delete Options:</p>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                <li><strong>Soft delete (default):</strong> Document is hidden but can be recovered by an admin</li>
                <li><strong>Hard delete:</strong> Document is permanently removed and cannot be recovered</li>
              </ul>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="hard-delete"
                checked={hardDeleteEnabled}
                onCheckedChange={(checked) => setHardDeleteEnabled(checked === true)}
              />
              <Label
                htmlFor="hard-delete"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Permanently delete (cannot be undone)
              </Label>
            </div>

            {hardDeleteEnabled && (
              <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-sm text-destructive">
                <strong>Warning:</strong> This will permanently delete the document, its chunks, and all associated data. This action cannot be undone.
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setDeleteDialogOpen(false);
                setDocumentToDelete(null);
                setHardDeleteEnabled(false);
              }}
            >
              Cancel
            </Button>
            <Button
              variant={hardDeleteEnabled ? "destructive" : "default"}
              onClick={confirmDeleteDocument}
              disabled={deleteDocument.isPending}
            >
              {deleteDocument.isPending
                ? "Deleting..."
                : hardDeleteEnabled
                ? "Permanently Delete"
                : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Document Preview Panel */}
      <Sheet open={isPreviewOpen} onOpenChange={setIsPreviewOpen}>
        <SheetContent side="right" className="w-[480px] sm:w-[540px] sm:max-w-[50vw] p-0">
          {previewDocument && (
            <div className="flex flex-col h-full">
              <SheetHeader className="px-6 py-4 border-b">
                <div className="flex items-center justify-between">
                  <SheetTitle className="text-base truncate pr-4">
                    {previewDocument.name || previewDocument.title}
                  </SheetTitle>
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleOpenFullView(previewDocument.id)}
                  >
                    <Eye className="h-3 w-3 mr-1" />
                    Full View
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleDownloadDocument(previewDocument.id, previewDocument.name)}
                  >
                    <Download className="h-3 w-3 mr-1" />
                    Download
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setEditingDocument(previewDocument);
                      setEditingTags(previewDocument.tags || []);
                      setEditTagsDialogOpen(true);
                    }}
                  >
                    <Edit className="h-3 w-3 mr-1" />
                    Tags
                  </Button>
                </div>
              </SheetHeader>
              <Tabs defaultValue="preview" className="flex-1 flex flex-col min-h-0">
                <TabsList className="mx-6 mt-3 w-fit">
                  <TabsTrigger value="preview">Preview</TabsTrigger>
                  <TabsTrigger value="metadata">Metadata</TabsTrigger>
                </TabsList>
                <TabsContent value="preview" className="flex-1 min-h-0 px-6 py-3">
                  <ScrollArea className="h-full">
                    <UploadedDocumentPreview
                      documentId={previewDocument.id}
                      fileName={previewDocument.name || previewDocument.original_filename}
                      fileType={previewDocument.file_type || previewDocument.name?.split('.').pop() || 'txt'}
                      className="min-h-[300px]"
                    />
                  </ScrollArea>
                </TabsContent>
                <TabsContent value="metadata" className="flex-1 min-h-0 px-6 py-3">
                  <ScrollArea className="h-full">
                    <div className="space-y-4 text-sm">
                      <div>
                        <p className="font-medium text-muted-foreground">Filename</p>
                        <p>{previewDocument.name}</p>
                      </div>
                      <div>
                        <p className="font-medium text-muted-foreground">Type</p>
                        <p>{previewDocument.file_type || 'Unknown'}</p>
                      </div>
                      {previewDocument.file_size && (
                        <div>
                          <p className="font-medium text-muted-foreground">Size</p>
                          <p>{(previewDocument.file_size / 1024).toFixed(1)} KB</p>
                        </div>
                      )}
                      {previewDocument.collection && (
                        <div>
                          <p className="font-medium text-muted-foreground">Collection</p>
                          <p>{previewDocument.collection}</p>
                        </div>
                      )}
                      {previewDocument.tags && previewDocument.tags.length > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground">Tags</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {previewDocument.tags.map((tag: string) => (
                              <Badge key={tag} variant="secondary" className="text-xs">{tag}</Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {previewDocument.created_at && (
                        <div>
                          <p className="font-medium text-muted-foreground">Created</p>
                          <p>{new Date(previewDocument.created_at).toLocaleString()}</p>
                        </div>
                      )}
                      {previewDocument.chunk_count !== undefined && (
                        <div>
                          <p className="font-medium text-muted-foreground">Chunks</p>
                          <p>{previewDocument.chunk_count}</p>
                        </div>
                      )}
                      {previewDocument.chunk_count > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground">Embedding Status</p>
                          <div className="flex items-center gap-2">
                            <span
                              className={`flex items-center gap-1 ${
                                previewDocument.has_all_embeddings
                                  ? "text-green-600 dark:text-green-400"
                                  : "text-amber-600 dark:text-amber-400"
                              }`}
                            >
                              {previewDocument.has_all_embeddings ? (
                                <Database className="h-4 w-4" />
                              ) : (
                                <AlertCircle className="h-4 w-4" />
                              )}
                              {previewDocument.embedding_coverage?.toFixed(0) || 0}% ({previewDocument.embedding_count}/{previewDocument.chunk_count})
                            </span>
                            {!previewDocument.has_all_embeddings && (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleGenerateEmbeddings(previewDocument.id)}
                                className="h-6 text-xs"
                              >
                                Generate
                              </Button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            </div>
          )}
        </SheetContent>
      </Sheet>

      {/* Image Analysis Modal */}
      {imageAnalysisDocument && (
        <ReanalyzeImagesModal
          open={isImageAnalysisOpen}
          onOpenChange={setIsImageAnalysisOpen}
          document={imageAnalysisDocument}
          onSuccess={() => {
            // Refresh the documents list
            refetch();
          }}
        />
      )}
    </div>
  );
}
