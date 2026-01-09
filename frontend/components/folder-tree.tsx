"use client";

import { useState, useCallback } from "react";
import { useDroppable } from "@dnd-kit/core";
import {
  Folder,
  FolderOpen,
  FolderPlus,
  ChevronRight,
  ChevronDown,
  MoreVertical,
  Pencil,
  Trash2,
  Move,
  Loader2,
  Home,
  Tag,
  X,
  Share2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import {
  useFolderTree,
  useCreateFolder,
  useUpdateFolder,
  useDeleteFolder,
  useMoveFolder,
  type FolderTreeNode,
  type CreateFolderRequest,
} from "@/lib/api";
import { FolderPermissionsDialog } from "./folder-permissions-dialog";

interface FolderTreeProps {
  selectedFolderId?: string | null;
  onFolderSelect: (folderId: string | null) => void;
  className?: string;
  // Drag-and-drop support
  dragOverFolderId?: string | null;
  isDropTarget?: boolean;
}

// Color picker options for folders
const FOLDER_COLORS = [
  { value: "#3B82F6", label: "Blue" },
  { value: "#10B981", label: "Green" },
  { value: "#F59E0B", label: "Amber" },
  { value: "#EF4444", label: "Red" },
  { value: "#8B5CF6", label: "Purple" },
  { value: "#EC4899", label: "Pink" },
  { value: "#6366F1", label: "Indigo" },
  { value: "#06B6D4", label: "Cyan" },
];

// Droppable root folder (for moving documents to root level)
function DroppableRootFolder({
  selectedFolderId,
  onFolderSelect,
  dragOverFolderId,
  isDropTarget,
}: {
  selectedFolderId?: string | null;
  onFolderSelect: (folderId: string | null) => void;
  dragOverFolderId?: string | null;
  isDropTarget?: boolean;
}) {
  const { setNodeRef, isOver } = useDroppable({
    id: "folder-root",
    disabled: !isDropTarget,
  });

  const isDraggedOver = isOver || dragOverFolderId === "folder-root";

  return (
    <div
      ref={isDropTarget ? setNodeRef : undefined}
      className={cn(
        "flex items-center gap-2 rounded-md px-2 py-1.5 cursor-pointer hover:bg-accent transition-colors",
        selectedFolderId === null && "bg-accent",
        isDraggedOver && "ring-2 ring-primary bg-primary/10"
      )}
      onClick={() => onFolderSelect(null)}
    >
      <Home className="h-4 w-4 text-muted-foreground" />
      <span className="text-sm">All Documents</span>
      {isDraggedOver && (
        <span className="text-xs text-primary ml-auto">Drop here</span>
      )}
    </div>
  );
}

function FolderTreeItem({
  folder,
  level,
  selectedFolderId,
  onFolderSelect,
  expandedFolders,
  toggleExpanded,
  onEdit,
  onDelete,
  onMove,
  onCreateSubfolder,
  onShare,
  dragOverFolderId,
  isDropTarget,
}: {
  folder: FolderTreeNode;
  level: number;
  selectedFolderId?: string | null;
  onFolderSelect: (folderId: string | null) => void;
  expandedFolders: Set<string>;
  toggleExpanded: (folderId: string) => void;
  onEdit: (folder: FolderTreeNode) => void;
  onDelete: (folder: FolderTreeNode) => void;
  onMove: (folder: FolderTreeNode) => void;
  onCreateSubfolder: (parentId: string) => void;
  onShare: (folder: FolderTreeNode) => void;
  dragOverFolderId?: string | null;
  isDropTarget?: boolean;
}) {
  const isExpanded = expandedFolders.has(folder.id);
  const isSelected = selectedFolderId === folder.id;

  // Make folder droppable
  const { setNodeRef, isOver } = useDroppable({
    id: `folder-${folder.id}`,
    disabled: !isDropTarget,
  });

  const isDraggedOver = isOver || dragOverFolderId === `folder-${folder.id}`;
  const hasChildren = folder.children && folder.children.length > 0;

  return (
    <div>
      <div
        ref={isDropTarget ? setNodeRef : undefined}
        className={cn(
          "group flex items-center gap-1 rounded-md px-2 py-1.5 cursor-pointer hover:bg-accent transition-colors",
          isSelected && "bg-accent",
          isDraggedOver && "ring-2 ring-primary bg-primary/10"
        )}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        {/* Expand/Collapse Button */}
        <button
          className="p-0.5 hover:bg-accent-foreground/10 rounded"
          onClick={(e) => {
            e.stopPropagation();
            toggleExpanded(folder.id);
          }}
        >
          {hasChildren ? (
            isExpanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )
          ) : (
            <span className="w-4" />
          )}
        </button>

        {/* Folder Icon */}
        <div onClick={() => onFolderSelect(folder.id)} className="flex items-center gap-2 flex-1 min-w-0">
          {isExpanded ? (
            <FolderOpen
              className="h-4 w-4 shrink-0"
              style={{ color: folder.color || "#3B82F6" }}
            />
          ) : (
            <Folder
              className="h-4 w-4 shrink-0"
              style={{ color: folder.color || "#3B82F6" }}
            />
          )}
          <span className="truncate text-sm">{folder.name}</span>
        </div>

        {/* Actions Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={(e) => e.stopPropagation()}
            >
              <MoreVertical className="h-3.5 w-3.5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onCreateSubfolder(folder.id)}>
              <FolderPlus className="mr-2 h-4 w-4" />
              New Subfolder
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => onEdit(folder)}>
              <Pencil className="mr-2 h-4 w-4" />
              Rename
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onMove(folder)}>
              <Move className="mr-2 h-4 w-4" />
              Move
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onShare(folder)}>
              <Share2 className="mr-2 h-4 w-4" />
              Share
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-destructive focus:text-destructive"
              onClick={() => onDelete(folder)}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Children */}
      {isExpanded && hasChildren && (
        <div>
          {folder.children.map((child) => (
            <FolderTreeItem
              key={child.id}
              folder={child}
              level={level + 1}
              selectedFolderId={selectedFolderId}
              onFolderSelect={onFolderSelect}
              expandedFolders={expandedFolders}
              toggleExpanded={toggleExpanded}
              onEdit={onEdit}
              onDelete={onDelete}
              onMove={onMove}
              onCreateSubfolder={onCreateSubfolder}
              onShare={onShare}
              dragOverFolderId={dragOverFolderId}
              isDropTarget={isDropTarget}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function FolderTree({
  selectedFolderId,
  onFolderSelect,
  className,
  dragOverFolderId,
  isDropTarget,
}: FolderTreeProps) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [moveDialogOpen, setMoveDialogOpen] = useState(false);
  const [parentFolderId, setParentFolderId] = useState<string | null>(null);
  const [selectedFolder, setSelectedFolder] = useState<FolderTreeNode | null>(null);
  const [folderName, setFolderName] = useState("");
  const [folderColor, setFolderColor] = useState("#3B82F6");
  const [folderTags, setFolderTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");
  const [deleteRecursive, setDeleteRecursive] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);

  // API hooks
  const { data: folderTree, isLoading } = useFolderTree();
  const createFolderMutation = useCreateFolder();
  const updateFolderMutation = useUpdateFolder();
  const deleteFolderMutation = useDeleteFolder();
  const moveFolderMutation = useMoveFolder();

  const toggleExpanded = useCallback((folderId: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) {
        next.delete(folderId);
      } else {
        next.add(folderId);
      }
      return next;
    });
  }, []);

  const handleCreateFolder = async () => {
    if (!folderName.trim()) {
      toast.error("Please enter a folder name");
      return;
    }

    try {
      const request: CreateFolderRequest = {
        name: folderName.trim(),
        parent_folder_id: parentFolderId,
        color: folderColor,
        tags: folderTags.length > 0 ? folderTags : undefined,
      };
      await createFolderMutation.mutateAsync(request);
      toast.success("Folder created");
      setFolderName("");
      setFolderTags([]);
      setCreateDialogOpen(false);
      // Expand parent if creating subfolder
      if (parentFolderId) {
        setExpandedFolders((prev) => new Set([...prev, parentFolderId]));
      }
    } catch (error: any) {
      toast.error("Failed to create folder", {
        description: error?.message || "Please try again",
      });
    }
  };

  const handleUpdateFolder = async () => {
    if (!selectedFolder || !folderName.trim()) return;

    try {
      await updateFolderMutation.mutateAsync({
        folderId: selectedFolder.id,
        request: {
          name: folderName.trim(),
          color: folderColor,
          tags: folderTags,
        },
      });
      toast.success("Folder updated");
      setEditDialogOpen(false);
    } catch (error: any) {
      toast.error("Failed to update folder", {
        description: error?.message || "Please try again",
      });
    }
  };

  const handleDeleteFolder = async () => {
    if (!selectedFolder) return;

    try {
      await deleteFolderMutation.mutateAsync({
        folderId: selectedFolder.id,
        recursive: deleteRecursive,
      });
      toast.success("Folder deleted");
      setDeleteDialogOpen(false);
      // Clear selection if deleted folder was selected
      if (selectedFolderId === selectedFolder.id) {
        onFolderSelect(null);
      }
    } catch (error: any) {
      toast.error("Failed to delete folder", {
        description: error?.message || "Please try again",
      });
    }
  };

  const handleMoveFolder = async (newParentId: string | null) => {
    if (!selectedFolder) return;

    try {
      await moveFolderMutation.mutateAsync({
        folderId: selectedFolder.id,
        newParentId,
      });
      toast.success("Folder moved");
      setMoveDialogOpen(false);
    } catch (error: any) {
      toast.error("Failed to move folder", {
        description: error?.message || "Please try again",
      });
    }
  };

  const openCreateDialog = (parentId: string | null) => {
    setParentFolderId(parentId);
    setFolderName("");
    setFolderColor("#3B82F6");
    setFolderTags([]);
    setNewTag("");
    setCreateDialogOpen(true);
  };

  const openEditDialog = (folder: FolderTreeNode) => {
    setSelectedFolder(folder);
    setFolderName(folder.name);
    setFolderColor(folder.color || "#3B82F6");
    setFolderTags(folder.tags || []);
    setNewTag("");
    setEditDialogOpen(true);
  };

  const openDeleteDialog = (folder: FolderTreeNode) => {
    setSelectedFolder(folder);
    setDeleteRecursive(false);
    setDeleteDialogOpen(true);
  };

  const openMoveDialog = (folder: FolderTreeNode) => {
    setSelectedFolder(folder);
    setMoveDialogOpen(true);
  };

  const openShareDialog = (folder: FolderTreeNode) => {
    setSelectedFolder(folder);
    setShareDialogOpen(true);
  };

  const handleAddTag = () => {
    const tag = newTag.trim().toLowerCase();
    if (tag && !folderTags.includes(tag)) {
      setFolderTags([...folderTags, tag]);
    }
    setNewTag("");
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setFolderTags(folderTags.filter((t) => t !== tagToRemove));
  };

  const handleTagKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddTag();
    }
  };

  if (isLoading) {
    return (
      <div className={cn("flex items-center justify-center py-8", className)}>
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-2">
        <h3 className="text-sm font-medium">Folders</h3>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={() => openCreateDialog(null)}
        >
          <FolderPlus className="h-4 w-4" />
        </Button>
      </div>

      {/* All Documents (root) */}
      <DroppableRootFolder
        selectedFolderId={selectedFolderId}
        onFolderSelect={onFolderSelect}
        dragOverFolderId={dragOverFolderId}
        isDropTarget={isDropTarget}
      />

      {/* Folder Tree */}
      {folderTree && folderTree.length > 0 ? (
        <div className="space-y-0.5">
          {folderTree.map((folder) => (
            <FolderTreeItem
              key={folder.id}
              folder={folder}
              level={0}
              selectedFolderId={selectedFolderId}
              onFolderSelect={onFolderSelect}
              expandedFolders={expandedFolders}
              toggleExpanded={toggleExpanded}
              onEdit={openEditDialog}
              onDelete={openDeleteDialog}
              onMove={openMoveDialog}
              onCreateSubfolder={openCreateDialog}
              onShare={openShareDialog}
              dragOverFolderId={dragOverFolderId}
              isDropTarget={isDropTarget}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-4 text-sm text-muted-foreground">
          No folders yet
        </div>
      )}

      {/* Create Folder Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Folder</DialogTitle>
            <DialogDescription>
              {parentFolderId
                ? "Create a new subfolder"
                : "Create a new root folder"}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="folder-name">Folder Name</Label>
              <Input
                id="folder-name"
                placeholder="Enter folder name"
                value={folderName}
                onChange={(e) => setFolderName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Color</Label>
              <div className="flex gap-2 flex-wrap">
                {FOLDER_COLORS.map((color) => (
                  <button
                    key={color.value}
                    className={cn(
                      "w-6 h-6 rounded-full border-2 transition-all",
                      folderColor === color.value
                        ? "border-foreground scale-110"
                        : "border-transparent"
                    )}
                    style={{ backgroundColor: color.value }}
                    onClick={() => setFolderColor(color.value)}
                    title={color.label}
                  />
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <Label>Tags</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="Add tag and press Enter"
                  value={newTag}
                  onChange={(e) => setNewTag(e.target.value)}
                  onKeyDown={handleTagKeyDown}
                  className="flex-1"
                />
                <Button type="button" variant="outline" size="icon" onClick={handleAddTag}>
                  <Tag className="h-4 w-4" />
                </Button>
              </div>
              {folderTags.length > 0 && (
                <div className="flex gap-1 flex-wrap mt-2">
                  {folderTags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="gap-1">
                      {tag}
                      <button
                        type="button"
                        onClick={() => handleRemoveTag(tag)}
                        className="ml-1 hover:text-destructive"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setCreateDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateFolder}
              disabled={createFolderMutation.isPending}
            >
              {createFolderMutation.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Folder Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Folder</DialogTitle>
            <DialogDescription>Update folder name, color, and tags</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-folder-name">Folder Name</Label>
              <Input
                id="edit-folder-name"
                value={folderName}
                onChange={(e) => setFolderName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Color</Label>
              <div className="flex gap-2 flex-wrap">
                {FOLDER_COLORS.map((color) => (
                  <button
                    key={color.value}
                    className={cn(
                      "w-6 h-6 rounded-full border-2 transition-all",
                      folderColor === color.value
                        ? "border-foreground scale-110"
                        : "border-transparent"
                    )}
                    style={{ backgroundColor: color.value }}
                    onClick={() => setFolderColor(color.value)}
                    title={color.label}
                  />
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <Label>Tags</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="Add tag and press Enter"
                  value={newTag}
                  onChange={(e) => setNewTag(e.target.value)}
                  onKeyDown={handleTagKeyDown}
                  className="flex-1"
                />
                <Button type="button" variant="outline" size="icon" onClick={handleAddTag}>
                  <Tag className="h-4 w-4" />
                </Button>
              </div>
              {folderTags.length > 0 && (
                <div className="flex gap-1 flex-wrap mt-2">
                  {folderTags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="gap-1">
                      {tag}
                      <button
                        type="button"
                        onClick={() => handleRemoveTag(tag)}
                        className="ml-1 hover:text-destructive"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleUpdateFolder}
              disabled={updateFolderMutation.isPending}
            >
              {updateFolderMutation.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Folder Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Folder</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{selectedFolder?.name}"?
              {selectedFolder?.children && selectedFolder.children.length > 0 && (
                <span className="block mt-2 text-amber-600">
                  This folder contains subfolders.
                </span>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="flex items-center space-x-2 py-2">
            <Checkbox
              id="recursive-delete"
              checked={deleteRecursive}
              onCheckedChange={(checked) => setDeleteRecursive(!!checked)}
            />
            <Label htmlFor="recursive-delete" className="text-sm">
              Delete all subfolders and unlink documents
            </Label>
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={handleDeleteFolder}
              disabled={deleteFolderMutation.isPending}
            >
              {deleteFolderMutation.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Move Folder Dialog */}
      <Dialog open={moveDialogOpen} onOpenChange={setMoveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Move Folder</DialogTitle>
            <DialogDescription>
              Select a new location for "{selectedFolder?.name}"
            </DialogDescription>
          </DialogHeader>
          <div className="py-4 max-h-[300px] overflow-y-auto">
            {/* Move to Root */}
            <div
              className={cn(
                "flex items-center gap-2 rounded-md px-2 py-1.5 cursor-pointer hover:bg-accent transition-colors"
              )}
              onClick={() => handleMoveFolder(null)}
            >
              <Home className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">Root (No parent)</span>
            </div>

            {/* Available folders */}
            {folderTree?.map((folder) => (
              <MoveTargetFolder
                key={folder.id}
                folder={folder}
                level={0}
                currentFolderId={selectedFolder?.id}
                onSelect={handleMoveFolder}
                isMoving={moveFolderMutation.isPending}
              />
            ))}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setMoveDialogOpen(false)}>
              Cancel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Folder Permissions Dialog */}
      {selectedFolder && (
        <FolderPermissionsDialog
          folderId={selectedFolder.id}
          folderName={selectedFolder.name}
          open={shareDialogOpen}
          onOpenChange={setShareDialogOpen}
        />
      )}
    </div>
  );
}

// Helper component for move dialog
function MoveTargetFolder({
  folder,
  level,
  currentFolderId,
  onSelect,
  isMoving,
}: {
  folder: FolderTreeNode;
  level: number;
  currentFolderId?: string;
  onSelect: (folderId: string) => void;
  isMoving: boolean;
}) {
  const isDisabled =
    folder.id === currentFolderId ||
    folder.path.includes(`/${currentFolderId}/`);

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors",
          isDisabled
            ? "opacity-50 cursor-not-allowed"
            : "cursor-pointer hover:bg-accent"
        )}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={() => !isDisabled && !isMoving && onSelect(folder.id)}
      >
        <Folder
          className="h-4 w-4 shrink-0"
          style={{ color: folder.color || "#3B82F6" }}
        />
        <span className="text-sm truncate">{folder.name}</span>
      </div>
      {folder.children?.map((child) => (
        <MoveTargetFolder
          key={child.id}
          folder={child}
          level={level + 1}
          currentFolderId={currentFolderId}
          onSelect={onSelect}
          isMoving={isMoving}
        />
      ))}
    </div>
  );
}
