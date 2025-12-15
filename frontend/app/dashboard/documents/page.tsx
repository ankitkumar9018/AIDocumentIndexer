"use client";

import { useState } from "react";
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
  useDocuments,
  useDocument,
  useUpdateDocument,
  useDeleteDocument,
  useSearchDocuments,
  useCollections,
  Document,
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

export default function DocumentsPage() {
  const [viewMode, setViewMode] = useState<ViewMode>("list");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<string>("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  // Queries
  const { data: documentsData, isLoading, refetch } = useDocuments({
    collection: selectedCollection || undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
  });
  const { data: collections } = useCollections();

  // Mutations
  const deleteDocument = useDeleteDocument();

  // Use real documents from API (no mock fallback)
  const documents: Document[] = documentsData?.documents ?? [];

  const filteredDocuments = documents.filter((doc) =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

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
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Collection Filter */}
        <div className="flex items-center gap-2">
          <select
            value={selectedCollection || ""}
            onChange={(e) => setSelectedCollection(e.target.value || null)}
            className="h-10 px-3 rounded-md border bg-background text-sm"
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
            >
              <List className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === "grid" ? "secondary" : "ghost"}
              size="sm"
              className="h-8 px-2"
              onClick={() => setViewMode("grid")}
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
                  {filteredDocuments.map((doc) => {
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
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
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
          {filteredDocuments.map((doc) => {
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
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
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
