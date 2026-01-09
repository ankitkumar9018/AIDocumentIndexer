"use client";

import { useState } from "react";
import {
  Folder,
  FolderPlus,
  Trash2,
  Loader2,
  Shield,
  Eye,
  Edit,
  Crown,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import {
  useUserFolderPermissions,
  useFolderTree,
  useGrantFolderPermission,
  useRevokeFolderPermission,
  folderPermissionQueryKeys,
  type UserFolderPermissionResponse,
  type FolderTreeNode,
} from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

interface UserFolderPermissionsDialogProps {
  userId: string;
  userName: string;
  userEmail: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const PERMISSION_LEVELS = [
  { value: "view", label: "View", icon: Eye, description: "Can see folder and read documents" },
  { value: "edit", label: "Edit", icon: Edit, description: "Can upload and modify documents" },
  { value: "manage", label: "Manage", icon: Crown, description: "Can grant permissions to others" },
];

function getPermissionIcon(level: string) {
  const perm = PERMISSION_LEVELS.find((p) => p.value === level);
  return perm?.icon || Eye;
}

function getPermissionLabel(level: string) {
  const perm = PERMISSION_LEVELS.find((p) => p.value === level);
  return perm?.label || level;
}

function getPermissionColor(level: string) {
  switch (level) {
    case "manage":
      return "bg-amber-500";
    case "edit":
      return "bg-blue-500";
    default:
      return "bg-gray-500";
  }
}

// Flatten folder tree to a list for the select dropdown
function flattenFolderTree(nodes: FolderTreeNode[], depth = 0): { id: string; name: string; path: string; depth: number }[] {
  const result: { id: string; name: string; path: string; depth: number }[] = [];
  for (const node of nodes) {
    result.push({ id: node.id, name: node.name, path: node.path || node.name, depth });
    if (node.children && node.children.length > 0) {
      result.push(...flattenFolderTree(node.children, depth + 1));
    }
  }
  return result;
}

export function UserFolderPermissionsDialog({
  userId,
  userName,
  userEmail,
  open,
  onOpenChange,
}: UserFolderPermissionsDialogProps) {
  const [showAddFolder, setShowAddFolder] = useState(false);
  const [selectedFolderId, setSelectedFolderId] = useState("");
  const [selectedPermissionLevel, setSelectedPermissionLevel] = useState("view");
  const [inheritToChildren, setInheritToChildren] = useState(true);

  const queryClient = useQueryClient();

  // API hooks
  const { data: permissions, isLoading: loadingPermissions, refetch } = useUserFolderPermissions(
    userId,
    { enabled: open }
  );
  const { data: folderTree, isLoading: loadingFolders } = useFolderTree(
    undefined,
    { enabled: open && showAddFolder }
  );
  const grantMutation = useGrantFolderPermission();
  const revokeMutation = useRevokeFolderPermission();

  const handleGrantPermission = async () => {
    if (!selectedFolderId) {
      toast.error("Please select a folder");
      return;
    }

    try {
      await grantMutation.mutateAsync({
        folderId: selectedFolderId,
        request: {
          user_id: userId,
          permission_level: selectedPermissionLevel,
          inherit_to_children: inheritToChildren,
        },
      });
      toast.success("Folder access granted");
      setShowAddFolder(false);
      setSelectedFolderId("");
      // Invalidate user permissions query
      queryClient.invalidateQueries({
        queryKey: folderPermissionQueryKeys.user(userId),
      });
      refetch();
    } catch (error: any) {
      toast.error("Failed to grant folder access", {
        description: error?.message || "Please try again",
      });
    }
  };

  const handleRevokePermission = async (folderId: string, folderName: string) => {
    try {
      await revokeMutation.mutateAsync({ folderId, userId });
      toast.success(`Access to "${folderName}" revoked`);
      // Invalidate user permissions query
      queryClient.invalidateQueries({
        queryKey: folderPermissionQueryKeys.user(userId),
      });
      refetch();
    } catch (error: any) {
      toast.error("Failed to revoke access", {
        description: error?.message || "Please try again",
      });
    }
  };

  // Flatten folder tree and filter out folders user already has access to
  const existingFolderIds = new Set(permissions?.map((p) => p.folder_id) || []);
  const allFolders = folderTree ? flattenFolderTree(folderTree) : [];
  const availableFolders = allFolders.filter((f) => !existingFolderIds.has(f.id));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Folder Access for {userName || userEmail}
          </DialogTitle>
          <DialogDescription>
            Manage which folders {userName || userEmail} can access
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Current Folder Access */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">Current Folder Access</Label>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAddFolder(!showAddFolder)}
              >
                <FolderPlus className="h-4 w-4 mr-2" />
                Add Folder
              </Button>
            </div>

            {loadingPermissions ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : permissions && permissions.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Folder</TableHead>
                    <TableHead>Permission</TableHead>
                    <TableHead>Inherits</TableHead>
                    <TableHead className="w-[80px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {permissions.map((perm) => {
                    const Icon = getPermissionIcon(perm.permission_level);
                    return (
                      <TableRow key={perm.id}>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Folder className="h-4 w-4 text-muted-foreground" />
                            <div>
                              <p className="text-sm font-medium">{perm.folder_name}</p>
                              {perm.folder_path && perm.folder_path !== perm.folder_name && (
                                <p className="text-xs text-muted-foreground">{perm.folder_path}</p>
                              )}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="secondary"
                            className={`gap-1 ${getPermissionColor(perm.permission_level)} text-white`}
                          >
                            <Icon className="h-3 w-3" />
                            {getPermissionLabel(perm.permission_level)}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant={perm.inherit_to_children ? "default" : "outline"}>
                            {perm.inherit_to_children ? "Yes" : "No"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-destructive hover:text-destructive"
                            onClick={() => handleRevokePermission(perm.folder_id, perm.folder_name)}
                            disabled={revokeMutation.isPending}
                          >
                            {revokeMutation.isPending ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Folder className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No explicit folder access</p>
                <p className="text-xs">User has tier-based access only</p>
              </div>
            )}
          </div>

          {/* Add Folder Form */}
          {showAddFolder && (
            <div className="border rounded-lg p-4 space-y-4 bg-muted/50">
              <Label className="text-sm font-medium">Grant Access to Folder</Label>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="folder-select" className="text-xs">Folder</Label>
                  <Select value={selectedFolderId} onValueChange={setSelectedFolderId}>
                    <SelectTrigger id="folder-select">
                      <SelectValue placeholder="Select folder..." />
                    </SelectTrigger>
                    <SelectContent>
                      {loadingFolders ? (
                        <div className="flex justify-center py-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                        </div>
                      ) : availableFolders.length === 0 ? (
                        <div className="text-center py-2 text-sm text-muted-foreground">
                          No folders available
                        </div>
                      ) : (
                        availableFolders.map((folder) => (
                          <SelectItem key={folder.id} value={folder.id}>
                            <div className="flex items-center gap-1">
                              {folder.depth > 0 && (
                                <span className="text-muted-foreground mr-1">
                                  {"  ".repeat(folder.depth)}
                                  <ChevronRight className="h-3 w-3 inline" />
                                </span>
                              )}
                              <Folder className="h-3 w-3 mr-1" />
                              {folder.name}
                            </div>
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="permission-select" className="text-xs">Permission Level</Label>
                  <Select value={selectedPermissionLevel} onValueChange={setSelectedPermissionLevel}>
                    <SelectTrigger id="permission-select">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PERMISSION_LEVELS.map((level) => (
                        <SelectItem key={level.value} value={level.value}>
                          <div className="flex items-center gap-2">
                            <level.icon className="h-4 w-4" />
                            <span>{level.label}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  id="inherit-permissions"
                  checked={inheritToChildren}
                  onCheckedChange={setInheritToChildren}
                />
                <Label htmlFor="inherit-permissions" className="text-sm">
                  Apply to all subfolders
                </Label>
              </div>

              <div className="flex gap-2 justify-end">
                <Button variant="outline" size="sm" onClick={() => setShowAddFolder(false)}>
                  Cancel
                </Button>
                <Button size="sm" onClick={handleGrantPermission} disabled={grantMutation.isPending}>
                  {grantMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Grant Access
                </Button>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
