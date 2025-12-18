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
  Document,
  api,
} from "@/lib/api";

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

  // Queries - only fetch when authenticated
  const { data: documentsData, isLoading, refetch } = useDocuments({
    collection: selectedCollection || undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
  }, { enabled: isAuthenticated });
  const { data: collections } = useCollections({ enabled: isAuthenticated });

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
        <Button onClick={() => refetch()} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
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
                              <p className="font-medium">{doc.name}</p>
                              <p className="text-xs text-muted-foreground">
                                {doc.chunk_count} chunks
                              </p>
                            </div>
                          </div>
                        </td>
                        <td className="p-3">
                          {doc.collection && (
                            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted text-xs">
                              <FolderOpen className="h-3 w-3" />
                              {doc.collection}
                            </span>
                          )}
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
                    <p className="font-medium truncate">{doc.name}</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {formatFileSize(doc.file_size)}
                    </p>
                  </div>
                  <div className="flex items-center justify-between mt-4 pt-4 border-t">
                    {doc.collection && (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted text-xs">
                        <FolderOpen className="h-3 w-3" />
                        {doc.collection}
                      </span>
                    )}
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
    </div>
  );
}
