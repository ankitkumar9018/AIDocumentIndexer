"use client";

import { useState } from "react";
import {
  Bookmark,
  Plus,
  Trash2,
  Search,
  Loader2,
  ChevronDown,
  ChevronUp,
  Calendar,
  FileType,
  Folder,
  Filter,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import {
  useSavedSearches,
  useSaveSearch,
  useDeleteSavedSearch,
  type SavedSearchResponse,
  type SavedSearchRequest,
} from "@/lib/api";

export interface SearchFilters {
  query: string;
  collection?: string | null;
  folder_id?: string | null;
  include_subfolders?: boolean;
  date_from?: string | null;
  date_to?: string | null;
  file_types?: string[] | null;
  search_mode?: "hybrid" | "vector" | "keyword";
}

interface SavedSearchesPanelProps {
  currentFilters: SearchFilters;
  onApplySearch: (filters: SearchFilters) => void;
  collections?: Array<{ name: string; document_count: number }>;
}

export function SavedSearchesPanel({
  currentFilters,
  onApplySearch,
  collections = [],
}: SavedSearchesPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [searchName, setSearchName] = useState("");
  const [selectedSearchMode, setSelectedSearchMode] = useState<"hybrid" | "vector" | "keyword">("hybrid");

  // API hooks
  const { data: savedSearchesData, isLoading } = useSavedSearches();
  const saveSearchMutation = useSaveSearch();
  const deleteSearchMutation = useDeleteSavedSearch();

  const savedSearches = savedSearchesData?.searches ?? [];

  // Check if there are filters to save
  const hasFilters =
    currentFilters.query ||
    currentFilters.collection ||
    currentFilters.folder_id ||
    currentFilters.date_from ||
    currentFilters.date_to ||
    (currentFilters.file_types && currentFilters.file_types.length > 0);

  const handleSaveSearch = async () => {
    if (!searchName.trim()) {
      toast.error("Please enter a name for this search");
      return;
    }

    if (!hasFilters) {
      toast.error("No filters to save", {
        description: "Apply some filters first",
      });
      return;
    }

    try {
      const searchRequest: SavedSearchRequest = {
        name: searchName.trim(),
        query: currentFilters.query || "",
        collection: currentFilters.collection || undefined,
        folder_id: currentFilters.folder_id || undefined,
        include_subfolders: currentFilters.include_subfolders ?? true,
        date_from: currentFilters.date_from || undefined,
        date_to: currentFilters.date_to || undefined,
        file_types: currentFilters.file_types || undefined,
        search_mode: selectedSearchMode,
      };

      await saveSearchMutation.mutateAsync(searchRequest);
      toast.success("Search saved", {
        description: `"${searchName}" saved to your searches`,
      });
      setSearchName("");
      setSaveDialogOpen(false);
    } catch (error: any) {
      toast.error("Failed to save search", {
        description: error?.message || "Please try again",
      });
    }
  };

  const handleApplySearch = (search: SavedSearchResponse) => {
    onApplySearch({
      query: search.query,
      collection: search.collection,
      folder_id: search.folder_id,
      include_subfolders: search.include_subfolders,
      date_from: search.date_from,
      date_to: search.date_to,
      file_types: search.file_types,
      search_mode: search.search_mode as "hybrid" | "vector" | "keyword",
    });
    toast.success("Search applied", {
      description: `Applied "${search.name}"`,
    });
  };

  const handleDeleteSearch = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSearchMutation.mutateAsync(name);
      toast.success("Search deleted");
    } catch (error: any) {
      toast.error("Failed to delete search", {
        description: error?.message || "Please try again",
      });
    }
  };

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="gap-2">
            <Bookmark className="h-4 w-4" />
            Saved Searches
            {savedSearches.length > 0 && (
              <Badge variant="secondary" className="ml-1 h-5 px-1.5 text-xs">
                {savedSearches.length}
              </Badge>
            )}
            {isOpen ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </CollapsibleTrigger>

        <Dialog open={saveDialogOpen} onOpenChange={setSaveDialogOpen}>
          <DialogTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              className="gap-2"
              disabled={!hasFilters}
            >
              <Plus className="h-4 w-4" />
              Save Current
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Save Search</DialogTitle>
              <DialogDescription>
                Save your current filters as a named search for quick access later.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="search-name">Search Name</Label>
                <Input
                  id="search-name"
                  placeholder="e.g., Recent PDFs, Marketing Docs..."
                  value={searchName}
                  onChange={(e) => setSearchName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="search-mode">Search Mode</Label>
                <Select
                  value={selectedSearchMode}
                  onValueChange={(v) => setSelectedSearchMode(v as any)}
                >
                  <SelectTrigger id="search-mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="hybrid">Hybrid (Best)</SelectItem>
                    <SelectItem value="vector">Semantic</SelectItem>
                    <SelectItem value="keyword">Keyword</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {/* Show current filters preview */}
              <div className="rounded-md border p-3 text-sm">
                <p className="font-medium mb-2">Current Filters:</p>
                <div className="space-y-1 text-muted-foreground">
                  {currentFilters.query && (
                    <p className="flex items-center gap-2">
                      <Search className="h-3 w-3" /> Query: "{currentFilters.query}"
                    </p>
                  )}
                  {currentFilters.collection && (
                    <p className="flex items-center gap-2">
                      <Folder className="h-3 w-3" /> Collection: {currentFilters.collection}
                    </p>
                  )}
                  {currentFilters.file_types && currentFilters.file_types.length > 0 && (
                    <p className="flex items-center gap-2">
                      <FileType className="h-3 w-3" /> Types: {currentFilters.file_types.join(", ")}
                    </p>
                  )}
                  {(currentFilters.date_from || currentFilters.date_to) && (
                    <p className="flex items-center gap-2">
                      <Calendar className="h-3 w-3" /> Date:{" "}
                      {currentFilters.date_from || "Any"} to{" "}
                      {currentFilters.date_to || "Any"}
                    </p>
                  )}
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setSaveDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                onClick={handleSaveSearch}
                disabled={saveSearchMutation.isPending || !searchName.trim()}
              >
                {saveSearchMutation.isPending && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Save Search
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <CollapsibleContent className="pt-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : savedSearches.length === 0 ? (
          <div className="rounded-md border border-dashed p-4 text-center text-sm text-muted-foreground">
            <Bookmark className="mx-auto h-8 w-8 mb-2 opacity-50" />
            <p>No saved searches yet</p>
            <p className="text-xs mt-1">
              Apply filters and click "Save Current" to save a search
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {savedSearches.map((search) => (
              <div
                key={search.name}
                className="flex items-center justify-between rounded-md border p-2 hover:bg-accent cursor-pointer transition-colors"
                onClick={() => handleApplySearch(search)}
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{search.name}</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {search.query && (
                      <Badge variant="secondary" className="text-xs">
                        <Search className="h-2.5 w-2.5 mr-1" />
                        {search.query.length > 20
                          ? search.query.slice(0, 20) + "..."
                          : search.query}
                      </Badge>
                    )}
                    {search.collection && (
                      <Badge variant="secondary" className="text-xs">
                        <Folder className="h-2.5 w-2.5 mr-1" />
                        {search.collection}
                      </Badge>
                    )}
                    {search.file_types && search.file_types.length > 0 && (
                      <Badge variant="secondary" className="text-xs">
                        <FileType className="h-2.5 w-2.5 mr-1" />
                        {search.file_types.length} types
                      </Badge>
                    )}
                    <Badge variant="outline" className="text-xs">
                      {search.search_mode}
                    </Badge>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0 text-muted-foreground hover:text-destructive"
                  onClick={(e) => handleDeleteSearch(search.name, e)}
                  disabled={deleteSearchMutation.isPending}
                >
                  {deleteSearchMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>
              </div>
            ))}
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
