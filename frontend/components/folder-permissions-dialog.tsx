"use client";

import { useState } from "react";
import {
  Users,
  UserPlus,
  Trash2,
  Loader2,
  Shield,
  Eye,
  Edit,
  Crown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  useFolderPermissions,
  useGrantFolderPermission,
  useRevokeFolderPermission,
  useAdminUsers,
  type FolderPermissionResponse,
} from "@/lib/api";

interface FolderPermissionsDialogProps {
  folderId: string;
  folderName: string;
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

export function FolderPermissionsDialog({
  folderId,
  folderName,
  open,
  onOpenChange,
}: FolderPermissionsDialogProps) {
  const [showAddUser, setShowAddUser] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [selectedPermissionLevel, setSelectedPermissionLevel] = useState("view");
  const [inheritToChildren, setInheritToChildren] = useState(true);

  // API hooks
  const { data: permissions, isLoading: loadingPermissions, refetch } = useFolderPermissions(
    folderId,
    { enabled: open }
  );
  const { data: usersData, isLoading: loadingUsers } = useAdminUsers(
    { page_size: 100 },
    { enabled: open && showAddUser }
  );
  const grantMutation = useGrantFolderPermission();
  const revokeMutation = useRevokeFolderPermission();

  const handleGrantPermission = async () => {
    if (!selectedUserId) {
      toast.error("Please select a user");
      return;
    }

    try {
      await grantMutation.mutateAsync({
        folderId,
        request: {
          user_id: selectedUserId,
          permission_level: selectedPermissionLevel,
          inherit_to_children: inheritToChildren,
        },
      });
      toast.success("Permission granted");
      setShowAddUser(false);
      setSelectedUserId("");
      refetch();
    } catch (error: any) {
      toast.error("Failed to grant permission", {
        description: error?.message || "Please try again",
      });
    }
  };

  const handleRevokePermission = async (userId: string, userEmail: string) => {
    try {
      await revokeMutation.mutateAsync({ folderId, userId });
      toast.success(`Permission revoked for ${userEmail}`);
      refetch();
    } catch (error: any) {
      toast.error("Failed to revoke permission", {
        description: error?.message || "Please try again",
      });
    }
  };

  // Filter out users who already have permission
  const existingUserIds = new Set(permissions?.map((p) => p.user_id) || []);
  const allUsers = usersData?.users || [];
  const availableUsers = allUsers.filter((u) => !existingUserIds.has(u.id));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Folder Permissions
          </DialogTitle>
          <DialogDescription>
            Manage who can access "{folderName}" and what they can do
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Current Permissions */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">Current Access</Label>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAddUser(!showAddUser)}
              >
                <UserPlus className="h-4 w-4 mr-2" />
                Add User
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
                    <TableHead>User</TableHead>
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
                            <Users className="h-4 w-4 text-muted-foreground" />
                            <div>
                              <p className="text-sm font-medium">{perm.user_email}</p>
                              {perm.user_name && (
                                <p className="text-xs text-muted-foreground">{perm.user_name}</p>
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
                            onClick={() => handleRevokePermission(perm.user_id, perm.user_email)}
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
                <Shield className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No explicit permissions set</p>
                <p className="text-xs">Access is determined by tier-based permissions</p>
              </div>
            )}
          </div>

          {/* Add User Form */}
          {showAddUser && (
            <div className="border rounded-lg p-4 space-y-4 bg-muted/50">
              <Label className="text-sm font-medium">Grant Access to User</Label>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="user-select" className="text-xs">User</Label>
                  <Select value={selectedUserId} onValueChange={setSelectedUserId}>
                    <SelectTrigger id="user-select">
                      <SelectValue placeholder="Select user..." />
                    </SelectTrigger>
                    <SelectContent>
                      {loadingUsers ? (
                        <div className="flex justify-center py-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                        </div>
                      ) : availableUsers.length === 0 ? (
                        <div className="text-center py-2 text-sm text-muted-foreground">
                          No users available
                        </div>
                      ) : (
                        availableUsers.map((user) => (
                          <SelectItem key={user.id} value={user.id}>
                            {user.email}
                            {user.name && ` (${user.name})`}
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
                <Button variant="outline" size="sm" onClick={() => setShowAddUser(false)}>
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
