"use client";

import { useState, useCallback, useEffect } from "react";
import {
  Brain,
  Search,
  Trash2,
  Download,
  Pencil,
  X,
  Check,
  AlertTriangle,
  BarChart3,
  Clock,
  Hash,
  Filter,
  RefreshCw,
  Loader2,
} from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";

import {
  useMemories,
  useMemoryStats,
  useUpdateMemory,
  useDeleteMemory,
  useClearAllMemories,
  api,
} from "@/lib/api";
import type { MemoryEntry } from "@/lib/api";

// ── Constants ────────────────────────────────────────────────────────────

const MEMORY_TYPES = [
  { value: "fact", label: "Fact" },
  { value: "preference", label: "Preference" },
  { value: "context", label: "Context" },
  { value: "procedure", label: "Procedure" },
  { value: "entity", label: "Entity" },
  { value: "relationship", label: "Relationship" },
];

const PRIORITY_LEVELS = [
  { value: "critical", label: "Critical", color: "text-red-500" },
  { value: "high", label: "High", color: "text-orange-500" },
  { value: "medium", label: "Medium", color: "text-yellow-500" },
  { value: "low", label: "Low", color: "text-gray-400" },
];

const PRIORITY_BADGE_VARIANT: Record<string, "destructive" | "default" | "secondary" | "outline"> = {
  critical: "destructive",
  high: "default",
  medium: "secondary",
  low: "outline",
};

const TYPE_COLORS: Record<string, string> = {
  fact: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  preference: "bg-purple-500/10 text-purple-500 border-purple-500/20",
  context: "bg-green-500/10 text-green-500 border-green-500/20",
  procedure: "bg-orange-500/10 text-orange-500 border-orange-500/20",
  entity: "bg-cyan-500/10 text-cyan-500 border-cyan-500/20",
  relationship: "bg-pink-500/10 text-pink-500 border-pink-500/20",
};

// ── Page Component ───────────────────────────────────────────────────────

export default function MemoryManagementPage() {
  // Filters
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [typeFilter, setTypeFilter] = useState<string>("");
  const [priorityFilter, setPriorityFilter] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");

  // Edit state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [editPriority, setEditPriority] = useState("");

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Dialog state
  const [showClearDialog, setShowClearDialog] = useState(false);
  const [showDeleteSelectedDialog, setShowDeleteSelectedDialog] = useState(false);

  // Debounced search
  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value);
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Queries
  const {
    data: memoriesData,
    isLoading: memoriesLoading,
    refetch: refetchMemories,
  } = useMemories({
    page,
    page_size: pageSize,
    memory_type: typeFilter || undefined,
    priority: priorityFilter || undefined,
    search: debouncedSearch || undefined,
  });

  const { data: stats, isLoading: statsLoading } = useMemoryStats();

  // Mutations
  const updateMemory = useUpdateMemory();
  const deleteMemory = useDeleteMemory();
  const clearAll = useClearAllMemories();

  // Handlers
  const startEdit = (memory: MemoryEntry) => {
    setEditingId(memory.id);
    setEditContent(memory.content);
    setEditPriority(memory.priority);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditContent("");
    setEditPriority("");
  };

  const saveEdit = async () => {
    if (!editingId) return;
    try {
      await updateMemory.mutateAsync({
        memoryId: editingId,
        data: { content: editContent, priority: editPriority },
      });
      toast.success("Memory updated");
      cancelEdit();
    } catch (e: unknown) {
      toast.error("Failed to update memory", { description: e instanceof Error ? e.message : "Unknown error" });
    }
  };

  const handleDelete = async (memoryId: string) => {
    try {
      await deleteMemory.mutateAsync(memoryId);
      toast.success("Memory deleted");
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(memoryId);
        return next;
      });
    } catch (e: unknown) {
      toast.error("Failed to delete memory", { description: e instanceof Error ? e.message : "Unknown error" });
    }
  };

  const handleDeleteSelected = async () => {
    const ids = Array.from(selectedIds);
    let deleted = 0;
    for (const id of ids) {
      try {
        await deleteMemory.mutateAsync(id);
        deleted++;
      } catch {
        // continue
      }
    }
    toast.success(`Deleted ${deleted} of ${ids.length} memories`);
    setSelectedIds(new Set());
    setShowDeleteSelectedDialog(false);
  };

  const handleClearAll = async () => {
    try {
      const result = await clearAll.mutateAsync();
      toast.success(`Cleared ${result.count} memories`);
      setShowClearDialog(false);
    } catch (e: unknown) {
      toast.error("Failed to clear memories", { description: e instanceof Error ? e.message : "Unknown error" });
    }
  };

  const handleExport = async () => {
    try {
      const blob = await api.exportMemories();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `memories_export_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Memories exported");
    } catch (e: unknown) {
      toast.error("Failed to export memories", { description: e instanceof Error ? e.message : "Unknown error" });
    }
  };

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (!memoriesData?.memories) return;
    const allIds = memoriesData.memories.map((m) => m.id);
    const allSelected = allIds.every((id) => selectedIds.has(id));
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(allIds));
    }
  };

  const clearFilters = () => {
    setTypeFilter("");
    setPriorityFilter("");
    setSearchQuery("");
    setDebouncedSearch("");
    setPage(1);
  };

  const memories = memoriesData?.memories || [];
  const totalMemories = memoriesData?.total || 0;
  const totalPages = Math.ceil(totalMemories / pageSize);
  const hasFilters = typeFilter || priorityFilter || debouncedSearch;

  return (
    <TooltipProvider>
      <div className="space-y-6 p-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="h-7 w-7 text-primary" />
            <div>
              <h1 className="text-2xl font-bold">Memory Management</h1>
              <p className="text-sm text-muted-foreground">
                View, edit, and manage your AI memory entries
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="sm" onClick={() => refetchMemories()}>
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Refresh</TooltipContent>
            </Tooltip>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setShowClearDialog(true)}
              disabled={totalMemories === 0}
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Clear All
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Brain className="h-4 w-4" /> Total Memories
              </CardTitle>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <Skeleton className="h-8 w-20" />
              ) : (
                <div className="text-2xl font-bold">{stats?.total_memories ?? 0}</div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <BarChart3 className="h-4 w-4" /> Avg Access Count
              </CardTitle>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <Skeleton className="h-8 w-20" />
              ) : (
                <div className="text-2xl font-bold">
                  {stats?.avg_access_count?.toFixed(1) ?? "0"}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Hash className="h-4 w-4" /> By Type
              </CardTitle>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <Skeleton className="h-8 w-40" />
              ) : (
                <div className="flex flex-wrap gap-1">
                  {stats?.memories_by_type &&
                    Object.entries(stats.memories_by_type).map(([type, count]) => (
                      <Badge
                        key={type}
                        variant="outline"
                        className={`text-xs ${TYPE_COLORS[type] || ""}`}
                      >
                        {type}: {count as number}
                      </Badge>
                    ))}
                  {(!stats?.memories_by_type ||
                    Object.keys(stats.memories_by_type).length === 0) && (
                    <span className="text-xs text-muted-foreground">No data</span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Clock className="h-4 w-4" /> Activity Range
              </CardTitle>
            </CardHeader>
            <CardContent>
              {statsLoading ? (
                <Skeleton className="h-8 w-40" />
              ) : (
                <div className="text-xs text-muted-foreground space-y-1">
                  <div>
                    Oldest:{" "}
                    {stats?.oldest_memory
                      ? new Date(stats.oldest_memory).toLocaleDateString()
                      : "—"}
                  </div>
                  <div>
                    Newest:{" "}
                    {stats?.newest_memory
                      ? new Date(stats.newest_memory).toLocaleDateString()
                      : "—"}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Filters Bar */}
        <Card>
          <CardContent className="pt-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative flex-1 min-w-[200px]">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search memories..."
                  value={searchQuery}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  className="pl-9"
                />
              </div>

              <Select value={typeFilter} onValueChange={(v) => { setTypeFilter(v === "all" ? "" : v); setPage(1); }}>
                <SelectTrigger className="w-[160px]">
                  <Filter className="h-4 w-4 mr-2 text-muted-foreground" />
                  <SelectValue placeholder="All Types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  {MEMORY_TYPES.map((t) => (
                    <SelectItem key={t.value} value={t.value}>
                      {t.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={priorityFilter} onValueChange={(v) => { setPriorityFilter(v === "all" ? "" : v); setPage(1); }}>
                <SelectTrigger className="w-[160px]">
                  <SelectValue placeholder="All Priorities" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Priorities</SelectItem>
                  {PRIORITY_LEVELS.map((p) => (
                    <SelectItem key={p.value} value={p.value}>
                      <span className={p.color}>{p.label}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {hasFilters && (
                <Button variant="ghost" size="sm" onClick={clearFilters}>
                  <X className="h-4 w-4 mr-1" />
                  Clear
                </Button>
              )}

              {selectedIds.size > 0 && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => setShowDeleteSelectedDialog(true)}
                >
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete {selectedIds.size} Selected
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Memory Table */}
        <Card>
          <CardContent className="p-0">
            <ScrollArea className="h-[calc(100vh-480px)] min-h-[300px]">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-background border-b z-10">
                  <tr>
                    <th className="text-left p-3 w-10">
                      <input
                        type="checkbox"
                        className="rounded border-muted-foreground/30"
                        checked={
                          memories.length > 0 &&
                          memories.every((m) => selectedIds.has(m.id))
                        }
                        onChange={toggleSelectAll}
                      />
                    </th>
                    <th className="text-left p-3 font-medium text-muted-foreground">
                      Content
                    </th>
                    <th className="text-left p-3 font-medium text-muted-foreground w-28">
                      Type
                    </th>
                    <th className="text-left p-3 font-medium text-muted-foreground w-24">
                      Priority
                    </th>
                    <th className="text-left p-3 font-medium text-muted-foreground w-32">
                      Last Accessed
                    </th>
                    <th className="text-center p-3 font-medium text-muted-foreground w-20">
                      Hits
                    </th>
                    <th className="text-center p-3 font-medium text-muted-foreground w-20">
                      Decay
                    </th>
                    <th className="text-right p-3 font-medium text-muted-foreground w-24">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {memoriesLoading ? (
                    Array.from({ length: 5 }).map((_, i) => (
                      <tr key={i} className="border-b">
                        <td className="p-3"><Skeleton className="h-4 w-4" /></td>
                        <td className="p-3"><Skeleton className="h-4 w-full" /></td>
                        <td className="p-3"><Skeleton className="h-4 w-16" /></td>
                        <td className="p-3"><Skeleton className="h-4 w-16" /></td>
                        <td className="p-3"><Skeleton className="h-4 w-24" /></td>
                        <td className="p-3"><Skeleton className="h-4 w-8 mx-auto" /></td>
                        <td className="p-3"><Skeleton className="h-4 w-8 mx-auto" /></td>
                        <td className="p-3"><Skeleton className="h-4 w-16 ml-auto" /></td>
                      </tr>
                    ))
                  ) : memories.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="text-center p-12 text-muted-foreground">
                        <Brain className="h-12 w-12 mx-auto mb-3 opacity-20" />
                        <p className="text-lg font-medium">No memories found</p>
                        <p className="text-sm">
                          {hasFilters
                            ? "Try adjusting your filters"
                            : "Memories are created as you chat with the AI"}
                        </p>
                      </td>
                    </tr>
                  ) : (
                    memories.map((memory) => (
                      <tr
                        key={memory.id}
                        className={`border-b hover:bg-muted/50 transition-colors ${
                          selectedIds.has(memory.id) ? "bg-muted/30" : ""
                        }`}
                      >
                        <td className="p-3">
                          <input
                            type="checkbox"
                            className="rounded border-muted-foreground/30"
                            checked={selectedIds.has(memory.id)}
                            onChange={() => toggleSelect(memory.id)}
                          />
                        </td>
                        <td className="p-3 max-w-md">
                          {editingId === memory.id ? (
                            <Textarea
                              value={editContent}
                              onChange={(e) => setEditContent(e.target.value)}
                              className="min-h-[60px] text-sm"
                              autoFocus
                            />
                          ) : (
                            <div className="truncate" title={memory.content}>
                              {memory.content}
                            </div>
                          )}
                          {memory.entities.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {memory.entities.slice(0, 3).map((e) => (
                                <Badge
                                  key={e}
                                  variant="outline"
                                  className="text-[10px] px-1 py-0"
                                >
                                  {e}
                                </Badge>
                              ))}
                              {memory.entities.length > 3 && (
                                <Badge variant="outline" className="text-[10px] px-1 py-0">
                                  +{memory.entities.length - 3}
                                </Badge>
                              )}
                            </div>
                          )}
                        </td>
                        <td className="p-3">
                          <Badge
                            variant="outline"
                            className={`text-xs ${TYPE_COLORS[memory.memory_type] || ""}`}
                          >
                            {memory.memory_type}
                          </Badge>
                        </td>
                        <td className="p-3">
                          {editingId === memory.id ? (
                            <Select value={editPriority} onValueChange={setEditPriority}>
                              <SelectTrigger className="h-7 text-xs">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {PRIORITY_LEVELS.map((p) => (
                                  <SelectItem key={p.value} value={p.value}>
                                    {p.label}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          ) : (
                            <Badge variant={PRIORITY_BADGE_VARIANT[memory.priority] || "outline"}>
                              {memory.priority}
                            </Badge>
                          )}
                        </td>
                        <td className="p-3 text-xs text-muted-foreground">
                          {new Date(memory.last_accessed).toLocaleDateString()}
                        </td>
                        <td className="p-3 text-center text-xs">
                          {memory.access_count}
                        </td>
                        <td className="p-3 text-center">
                          <span
                            className={`text-xs font-mono ${
                              memory.decay_score > 0.7
                                ? "text-green-500"
                                : memory.decay_score > 0.4
                                ? "text-yellow-500"
                                : "text-red-500"
                            }`}
                          >
                            {memory.decay_score.toFixed(2)}
                          </span>
                        </td>
                        <td className="p-3 text-right">
                          {editingId === memory.id ? (
                            <div className="flex items-center gap-1 justify-end">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7"
                                onClick={saveEdit}
                                disabled={updateMemory.isPending}
                              >
                                {updateMemory.isPending ? (
                                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                ) : (
                                  <Check className="h-3.5 w-3.5 text-green-500" />
                                )}
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7"
                                onClick={cancelEdit}
                              >
                                <X className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1 justify-end">
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7"
                                    onClick={() => startEdit(memory)}
                                  >
                                    <Pencil className="h-3.5 w-3.5" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>Edit</TooltipContent>
                              </Tooltip>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 text-destructive hover:text-destructive"
                                    onClick={() => handleDelete(memory.id)}
                                    disabled={deleteMemory.isPending}
                                  >
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>Delete</TooltipContent>
                              </Tooltip>
                            </div>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </ScrollArea>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between border-t px-4 py-3">
                <span className="text-sm text-muted-foreground">
                  Showing {(page - 1) * pageSize + 1}–
                  {Math.min(page * pageSize, totalMemories)} of {totalMemories}
                </span>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={page <= 1}
                    onClick={() => setPage((p) => p - 1)}
                  >
                    Previous
                  </Button>
                  <span className="text-sm">
                    Page {page} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={page >= totalPages}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Clear All Confirmation Dialog */}
        <Dialog open={showClearDialog} onOpenChange={setShowClearDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                Clear All Memories
              </DialogTitle>
              <DialogDescription>
                This will permanently delete all {totalMemories} memories. This
                action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowClearDialog(false)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleClearAll}
                disabled={clearAll.isPending}
              >
                {clearAll.isPending ? (
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4 mr-1" />
                )}
                Clear All
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Selected Confirmation Dialog */}
        <Dialog
          open={showDeleteSelectedDialog}
          onOpenChange={setShowDeleteSelectedDialog}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                Delete Selected Memories
              </DialogTitle>
              <DialogDescription>
                Delete {selectedIds.size} selected memories? This action cannot be
                undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setShowDeleteSelectedDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteSelected}
                disabled={deleteMemory.isPending}
              >
                {deleteMemory.isPending ? (
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4 mr-1" />
                )}
                Delete {selectedIds.size}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  );
}
