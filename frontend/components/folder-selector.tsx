"use client";

import { useState, useCallback } from "react";
import {
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  Home,
  Loader2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  useFolderTree,
  type FolderTreeNode,
} from "@/lib/api";

interface FolderSelectorProps {
  value: string | null;
  onChange: (folderId: string | null) => void;
  includeSubfolders?: boolean;
  onIncludeSubfoldersChange?: (include: boolean) => void;
  showSubfoldersToggle?: boolean;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

function FolderTreeItemSelect({
  folder,
  level,
  selectedFolderId,
  onSelect,
  expandedFolders,
  toggleExpanded,
}: {
  folder: FolderTreeNode;
  level: number;
  selectedFolderId: string | null;
  onSelect: (folderId: string | null) => void;
  expandedFolders: Set<string>;
  toggleExpanded: (folderId: string) => void;
}) {
  const isExpanded = expandedFolders.has(folder.id);
  const isSelected = selectedFolderId === folder.id;
  const hasChildren = folder.children && folder.children.length > 0;

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-1 rounded-md px-2 py-1.5 cursor-pointer hover:bg-accent transition-colors",
          isSelected && "bg-accent"
        )}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={() => onSelect(folder.id)}
      >
        {/* Expand/Collapse Button */}
        {hasChildren ? (
          <button
            className="p-0.5 hover:bg-accent-foreground/10 rounded"
            onClick={(e) => {
              e.stopPropagation();
              toggleExpanded(folder.id);
            }}
          >
            {isExpanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
            )}
          </button>
        ) : (
          <span className="w-4" />
        )}

        {/* Folder Icon */}
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

      {/* Children */}
      {isExpanded && hasChildren && (
        <div>
          {folder.children.map((child) => (
            <FolderTreeItemSelect
              key={child.id}
              folder={child}
              level={level + 1}
              selectedFolderId={selectedFolderId}
              onSelect={onSelect}
              expandedFolders={expandedFolders}
              toggleExpanded={toggleExpanded}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function FolderSelector({
  value,
  onChange,
  includeSubfolders = true,
  onIncludeSubfoldersChange,
  showSubfoldersToggle = false,
  placeholder = "Select folder...",
  className,
  disabled = false,
}: FolderSelectorProps) {
  const [open, setOpen] = useState(false);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());

  const { data: folderTree, isLoading } = useFolderTree();

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

  const handleSelect = (folderId: string | null) => {
    onChange(folderId);
    setOpen(false);
  };

  // Find the selected folder name
  const findFolderName = (folders: FolderTreeNode[] | undefined, id: string): string | null => {
    if (!folders) return null;
    for (const folder of folders) {
      if (folder.id === id) return folder.name;
      if (folder.children) {
        const found = findFolderName(folder.children, id);
        if (found) return found;
      }
    }
    return null;
  };

  const selectedFolderName = value ? findFolderName(folderTree, value) : null;

  return (
    <div className={cn("space-y-2", className)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal"
            disabled={disabled}
          >
            <span className="flex items-center gap-2 truncate">
              {value ? (
                <>
                  <Folder className="h-4 w-4 shrink-0 text-blue-500" />
                  <span className="truncate">{selectedFolderName || "Unknown folder"}</span>
                </>
              ) : (
                <>
                  <Home className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="text-muted-foreground">{placeholder}</span>
                </>
              )}
            </span>
            {value && (
              <X
                className="h-4 w-4 shrink-0 opacity-50 hover:opacity-100"
                onClick={(e) => {
                  e.stopPropagation();
                  onChange(null);
                }}
              />
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[280px] p-0" align="start">
          <div className="max-h-[300px] overflow-y-auto p-2">
            {isLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <>
                {/* Root option */}
                <div
                  className={cn(
                    "flex items-center gap-2 rounded-md px-2 py-1.5 cursor-pointer hover:bg-accent transition-colors",
                    value === null && "bg-accent"
                  )}
                  onClick={() => handleSelect(null)}
                >
                  <Home className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Root (All Documents)</span>
                </div>

                {/* Folder Tree */}
                {folderTree && folderTree.length > 0 ? (
                  <div className="mt-1 space-y-0.5">
                    {folderTree.map((folder) => (
                      <FolderTreeItemSelect
                        key={folder.id}
                        folder={folder}
                        level={0}
                        selectedFolderId={value}
                        onSelect={handleSelect}
                        expandedFolders={expandedFolders}
                        toggleExpanded={toggleExpanded}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-4 text-sm text-muted-foreground">
                    No folders created yet
                  </div>
                )}
              </>
            )}
          </div>
        </PopoverContent>
      </Popover>

      {/* Include Subfolders Toggle */}
      {showSubfoldersToggle && value && onIncludeSubfoldersChange && (
        <div className="flex items-center space-x-2">
          <Checkbox
            id="include-subfolders"
            checked={includeSubfolders}
            onCheckedChange={(checked) => onIncludeSubfoldersChange(!!checked)}
          />
          <Label htmlFor="include-subfolders" className="text-sm text-muted-foreground">
            Include documents in subfolders
          </Label>
        </div>
      )}
    </div>
  );
}
