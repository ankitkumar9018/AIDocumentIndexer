"use client";

import { useState } from "react";
import { getErrorMessage } from "@/lib/errors";
import { toast } from "sonner";
import {
  Users,
  Search,
  MoreVertical,
  Shield,
  ShieldCheck,
  ShieldAlert,
  Mail,
  Calendar,
  UserPlus,
  RefreshCw,
  Loader2,
  AlertCircle,
  Plus,
  Trash2,
  FolderKey,
  Folder,
  ChevronRight,
  X,
  Lock,
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
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  useAdminUsers,
  useAccessTiers,
  useUpdateAdminUser,
  useAdminStats,
  useCreateUser,
  useUpdateAccessTier,
  useCreateAccessTier,
  useDeleteAccessTier,
  useFolderTree,
  useGrantFolderPermission,
  folderPermissionQueryKeys,
} from "@/lib/api/hooks";
import type { FolderTreeNode } from "@/lib/api";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { useQueryClient } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useUser } from "@/lib/auth";
import { UserFolderPermissionsDialog } from "@/components/user-folder-permissions-dialog";

const getTierIcon = (level: number) => {
  if (level >= 80) return <ShieldCheck className="h-4 w-4 text-green-500" />;
  if (level >= 50) return <Shield className="h-4 w-4 text-blue-500" />;
  return <ShieldAlert className="h-4 w-4 text-yellow-500" />;
};

export default function AdminUsersPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(1);
  const { isAuthenticated, isLoading: authLoading } = useUser();

  // Add User dialog state
  const [showAddUserDialog, setShowAddUserDialog] = useState(false);
  const [newUserEmail, setNewUserEmail] = useState("");
  const [newUserPassword, setNewUserPassword] = useState("");
  const [newUserName, setNewUserName] = useState("");
  const [newUserTierId, setNewUserTierId] = useState("");
  const [newUserFolderOnly, setNewUserFolderOnly] = useState(false);
  const [newUserFolderPermissions, setNewUserFolderPermissions] = useState<Array<{
    folder_id: string;
    folder_name: string;
    permission_level: 'view' | 'edit' | 'manage';
    inherit_to_children: boolean;
  }>>([]);
  const [selectedFolderId, setSelectedFolderId] = useState("");
  const [selectedPermissionLevel, setSelectedPermissionLevel] = useState<'view' | 'edit' | 'manage'>("view");

  // Configure Tier dialog state
  const [showConfigureTierDialog, setShowConfigureTierDialog] = useState(false);
  const [editingTier, setEditingTier] = useState<{ id: string; name: string; level: number; description: string; color: string } | null>(null);
  const [tierName, setTierName] = useState("");
  const [tierLevel, setTierLevel] = useState(0);
  const [tierDescription, setTierDescription] = useState("");
  const [tierColor, setTierColor] = useState("#6B7280");

  // Change User Tier state
  const [showChangeTierDialog, setShowChangeTierDialog] = useState(false);
  const [changingUser, setChangingUser] = useState<{ id: string; name: string; currentTierId: string } | null>(null);
  const [selectedNewTierId, setSelectedNewTierId] = useState("");

  // Add Tier dialog state
  const [showAddTierDialog, setShowAddTierDialog] = useState(false);
  const [newTierName, setNewTierName] = useState("");

  // Folder Permissions dialog state
  const [folderPermissionsUser, setFolderPermissionsUser] = useState<{ id: string; name: string; email: string } | null>(null);
  const [newTierLevel, setNewTierLevel] = useState(50);
  const [newTierDescription, setNewTierDescription] = useState("");
  const [newTierColor, setNewTierColor] = useState("#6B7280");

  // Real API calls - only fetch when authenticated
  const { data: usersData, isLoading: usersLoading, error: usersError, refetch: refetchUsers } = useAdminUsers({
    page,
    page_size: 20,
    search: searchQuery || undefined,
  }, { enabled: isAuthenticated });

  const { data: tiersData, isLoading: tiersLoading } = useAccessTiers({ enabled: isAuthenticated });
  const { data: statsData, isLoading: statsLoading } = useAdminStats({ enabled: isAuthenticated });
  const { data: folderTree } = useFolderTree(undefined, { enabled: isAuthenticated && showAddUserDialog });
  const queryClient = useQueryClient();
  const updateUser = useUpdateAdminUser();
  const createUser = useCreateUser();
  const updateTier = useUpdateAccessTier();
  const createTier = useCreateAccessTier();
  const deleteTier = useDeleteAccessTier();

  const users = usersData?.users ?? [];
  const tiers = tiersData?.tiers ?? [];

  // Stats from real API
  const stats = {
    total: statsData?.users?.total ?? 0,
    active: statsData?.users?.active ?? 0,
    pending: (statsData?.users?.total ?? 0) - (statsData?.users?.active ?? 0),
    admins: tiers.filter(t => t.level >= 80).reduce((sum, t) => sum + t.user_count, 0),
  };

  const handleToggleActive = async (userId: string, currentStatus: boolean) => {
    try {
      await updateUser.mutateAsync({
        userId,
        data: { is_active: !currentStatus },
      });
    } catch (err) {
      console.error("Failed to update user:", err);
    }
  };

  // Helper to flatten folder tree
  const flattenFolderTree = (nodes: FolderTreeNode[], depth = 0): { id: string; name: string; path: string; depth: number }[] => {
    const result: { id: string; name: string; path: string; depth: number }[] = [];
    for (const node of nodes) {
      result.push({ id: node.id, name: node.name, path: node.path || node.name, depth });
      if (node.children && node.children.length > 0) {
        result.push(...flattenFolderTree(node.children, depth + 1));
      }
    }
    return result;
  };

  const flatFolders = folderTree ? flattenFolderTree(folderTree) : [];

  // Add User handlers
  const handleOpenAddUser = () => {
    setNewUserEmail("");
    setNewUserPassword("");
    setNewUserName("");
    setNewUserTierId(tiers[0]?.id || "");
    setNewUserFolderOnly(false);
    setNewUserFolderPermissions([]);
    setSelectedFolderId("");
    setSelectedPermissionLevel("view");
    setShowAddUserDialog(true);
  };

  const handleCreateUser = async () => {
    if (!newUserEmail || !newUserPassword || !newUserTierId) return;
    try {
      await createUser.mutateAsync({
        email: newUserEmail,
        password: newUserPassword,
        name: newUserName || undefined,
        access_tier_id: newUserTierId,
        use_folder_permissions_only: newUserFolderOnly,
        initial_folder_permissions: newUserFolderPermissions.length > 0
          ? newUserFolderPermissions.map((p) => ({
              folder_id: p.folder_id,
              permission_level: p.permission_level,
              inherit_to_children: p.inherit_to_children,
            }))
          : undefined,
      });
      setShowAddUserDialog(false);
      toast.success("User created successfully");
      refetchUsers();
    } catch (err: unknown) {
      console.error("Failed to create user:", err);
      toast.error("Failed to create user", {
        description: getErrorMessage(err),
      });
    }
  };

  const handleAddFolderPermission = () => {
    if (!selectedFolderId) return;
    const folder = flatFolders.find((f) => f.id === selectedFolderId);
    if (!folder) return;
    if (newUserFolderPermissions.some((p) => p.folder_id === selectedFolderId)) {
      toast.error("Folder already added");
      return;
    }
    setNewUserFolderPermissions([
      ...newUserFolderPermissions,
      {
        folder_id: selectedFolderId,
        folder_name: folder.name,
        permission_level: selectedPermissionLevel,
        inherit_to_children: true,
      },
    ]);
    setSelectedFolderId("");
  };

  const handleRemoveFolderPermission = (folderId: string) => {
    setNewUserFolderPermissions(newUserFolderPermissions.filter((p) => p.folder_id !== folderId));
  };

  const handleToggleFolderOnly = async (userId: string, currentValue: boolean) => {
    try {
      await updateUser.mutateAsync({
        userId,
        data: { use_folder_permissions_only: !currentValue },
      });
      toast.success(!currentValue ? "User restricted to folder access only" : "User tier-based access restored");
      refetchUsers();
    } catch (err: unknown) {
      console.error("Failed to update user:", err);
      toast.error("Failed to update user", {
        description: getErrorMessage(err),
      });
    }
  };

  // Configure Tier handlers
  const handleOpenConfigureTier = (tier: typeof tiers[0]) => {
    setEditingTier({
      id: tier.id,
      name: tier.name,
      level: tier.level,
      description: tier.description || "",
      color: tier.color || "#6B7280",
    });
    setTierName(tier.name);
    setTierLevel(tier.level);
    setTierDescription(tier.description || "");
    setTierColor(tier.color || "#6B7280");
    setShowConfigureTierDialog(true);
  };

  const handleUpdateTier = async () => {
    if (!editingTier) return;
    try {
      await updateTier.mutateAsync({
        tierId: editingTier.id,
        data: {
          name: tierName,
          level: tierLevel,
          description: tierDescription || undefined,
          color: tierColor,
        },
      });
      setShowConfigureTierDialog(false);
      toast.success("Access tier updated successfully");
    } catch (err: unknown) {
      console.error("Failed to update tier:", err);
      toast.error("Failed to update tier", {
        description: getErrorMessage(err),
      });
    }
  };

  // Change User Tier handlers
  const handleOpenChangeTier = (user: typeof users[0]) => {
    setChangingUser({
      id: user.id,
      name: user.name || user.email,
      currentTierId: user.access_tier_id,
    });
    setSelectedNewTierId(user.access_tier_id);
    setShowChangeTierDialog(true);
  };

  const handleChangeUserTier = async () => {
    if (!changingUser || !selectedNewTierId) return;
    try {
      await updateUser.mutateAsync({
        userId: changingUser.id,
        data: { access_tier_id: selectedNewTierId },
      });
      setShowChangeTierDialog(false);
      toast.success("User tier updated successfully");
      refetchUsers();
    } catch (err: unknown) {
      console.error("Failed to change user tier:", err);
      toast.error("Failed to change user tier", {
        description: getErrorMessage(err),
      });
    }
  };

  // Add Tier handlers
  const handleOpenAddTier = () => {
    setNewTierName("");
    setNewTierLevel(50);
    setNewTierDescription("");
    setNewTierColor("#6B7280");
    setShowAddTierDialog(true);
  };

  const handleCreateTier = async () => {
    if (!newTierName) return;
    try {
      await createTier.mutateAsync({
        name: newTierName,
        level: newTierLevel,
        description: newTierDescription || undefined,
        color: newTierColor,
      });
      setShowAddTierDialog(false);
      toast.success("Access tier created successfully");
    } catch (err: unknown) {
      console.error("Failed to create tier:", err);
      toast.error("Failed to create tier", {
        description: getErrorMessage(err),
      });
    }
  };

  const handleDeleteTier = async (tierId: string, tierName: string, userCount: number) => {
    if (userCount > 0) {
      toast.error("Cannot delete tier", {
        description: `${tierName} has ${userCount} users. Reassign users first.`,
      });
      return;
    }

    if (!confirm(`Are you sure you want to delete the "${tierName}" tier? This action cannot be undone.`)) {
      return;
    }

    try {
      await deleteTier.mutateAsync(tierId);
      toast.success("Access tier deleted successfully");
    } catch (err: unknown) {
      console.error("Failed to delete tier:", err);
      toast.error("Failed to delete tier", {
        description: getErrorMessage(err),
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">User Management</h1>
          <p className="text-muted-foreground">
            Manage users and their access permissions
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={() => refetchUsers()}>
            <RefreshCw className={`h-4 w-4 ${usersLoading ? "animate-spin" : ""}`} />
          </Button>
          <Button onClick={handleOpenAddUser}>
            <UserPlus className="h-4 w-4 mr-2" />
            Add User
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Users</p>
                {statsLoading ? (
                  <Loader2 className="h-6 w-6 animate-spin mt-1" />
                ) : (
                  <p className="text-2xl font-bold">{stats.total}</p>
                )}
              </div>
              <Users className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active</p>
                {statsLoading ? (
                  <Loader2 className="h-6 w-6 animate-spin mt-1" />
                ) : (
                  <p className="text-2xl font-bold">{stats.active}</p>
                )}
              </div>
              <div className="h-8 w-8 rounded-full bg-green-500/10 flex items-center justify-center">
                <div className="h-3 w-3 rounded-full bg-green-500" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Inactive</p>
                {statsLoading ? (
                  <Loader2 className="h-6 w-6 animate-spin mt-1" />
                ) : (
                  <p className="text-2xl font-bold">{statsData?.users?.inactive ?? 0}</p>
                )}
              </div>
              <div className="h-8 w-8 rounded-full bg-gray-500/10 flex items-center justify-center">
                <div className="h-3 w-3 rounded-full bg-gray-500" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Access Tiers</p>
                {tiersLoading ? (
                  <Loader2 className="h-6 w-6 animate-spin mt-1" />
                ) : (
                  <p className="text-2xl font-bold">{tiers.length}</p>
                )}
              </div>
              <ShieldCheck className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search users by name or email..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
            setPage(1); // Reset to first page on search
          }}
          className="pl-9"
        />
      </div>

      {/* Users Table */}
      <Card>
        <CardContent className="p-0">
          {usersError ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Failed to load users</h3>
              <p className="text-muted-foreground mb-4">
                {getErrorMessage(usersError, "Unable to fetch users. Please try again.")}
              </p>
              <Button onClick={() => refetchUsers()}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry
              </Button>
            </div>
          ) : authLoading || usersLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : users.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Users className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No users found</h3>
              <p className="text-muted-foreground">
                {searchQuery
                  ? "No users match your search criteria."
                  : "No users have been created yet."}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-4 text-sm font-medium">User</th>
                    <th className="text-left p-4 text-sm font-medium">Access Tier</th>
                    <th className="text-left p-4 text-sm font-medium">Status</th>
                    <th className="text-left p-4 text-sm font-medium">Joined</th>
                    <th className="text-left p-4 text-sm font-medium">Last Login</th>
                    <th className="w-12 p-4"></th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => (
                    <tr key={user.id} className="border-b last:border-0 hover:bg-muted/50">
                      <td className="p-4">
                        <div className="flex items-center gap-3">
                          <Avatar className="h-9 w-9">
                            <AvatarFallback>
                              {(user.name || user.email)
                                .split(/[\s@]/)
                                .map((n) => n[0]?.toUpperCase())
                                .filter(Boolean)
                                .slice(0, 2)
                                .join("")}
                            </AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="font-medium">{user.name || "Unnamed User"}</p>
                            <p className="text-sm text-muted-foreground flex items-center gap-1">
                              <Mail className="h-3 w-3" />
                              {user.email}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          {getTierIcon(user.access_tier_level)}
                          <span className="text-sm">{user.access_tier_name}</span>
                          <span className="text-xs text-muted-foreground">
                            (Level {user.access_tier_level})
                          </span>
                          {user.use_folder_permissions_only && (
                            <Badge variant="outline" className="text-xs gap-1 text-amber-600 border-amber-300">
                              <Lock className="h-3 w-3" />
                              Folder Only
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="p-4">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                            user.is_active
                              ? "bg-green-500/10 text-green-600"
                              : "bg-gray-500/10 text-gray-600"
                          }`}
                        >
                          <div
                            className={`h-1.5 w-1.5 rounded-full ${
                              user.is_active ? "bg-green-500" : "bg-gray-500"
                            }`}
                          />
                          {user.is_active ? "Active" : "Inactive"}
                        </span>
                      </td>
                      <td className="p-4 text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          {new Date(user.created_at).toLocaleDateString()}
                        </div>
                      </td>
                      <td className="p-4 text-sm text-muted-foreground">
                        {user.last_login_at
                          ? new Date(user.last_login_at).toLocaleDateString()
                          : "Never"}
                      </td>
                      <td className="p-4">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8">
                              <MoreVertical className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem
                              onClick={() => handleToggleActive(user.id, user.is_active)}
                            >
                              {user.is_active ? "Deactivate User" : "Activate User"}
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => handleOpenChangeTier(user)}>
                              Change Access Tier
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => setFolderPermissionsUser({
                                id: user.id,
                                name: user.name || "",
                                email: user.email,
                              })}
                            >
                              <FolderKey className="h-4 w-4 mr-2" />
                              Manage Folder Access
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onClick={() => handleToggleFolderOnly(user.id, user.use_folder_permissions_only)}
                            >
                              <Lock className="h-4 w-4 mr-2" />
                              {user.use_folder_permissions_only
                                ? "Enable Tier-Based Access"
                                : "Restrict to Folders Only"}
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>

        {/* Pagination */}
        {usersData && usersData.total > 20 && (
          <div className="flex items-center justify-between px-4 py-3 border-t">
            <p className="text-sm text-muted-foreground">
              Showing {((page - 1) * 20) + 1} to {Math.min(page * 20, usersData.total)} of {usersData.total} users
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => p + 1)}
                disabled={!usersData.has_more}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </Card>

      {/* Access Tiers */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-lg">Access Tiers</CardTitle>
            <CardDescription>Document access permissions by tier</CardDescription>
          </div>
          <Button size="sm" onClick={handleOpenAddTier}>
            <Plus className="h-4 w-4 mr-2" />
            Add Tier
          </Button>
        </CardHeader>
        <CardContent>
          {tiersLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : tiers.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No access tiers configured
            </div>
          ) : (
            <div className="space-y-4">
              {tiers
                .sort((a, b) => b.level - a.level)
                .map((tier) => (
                  <div
                    key={tier.id}
                    className="flex items-center justify-between p-3 rounded-lg border"
                  >
                    <div className="flex items-center gap-3">
                      {getTierIcon(tier.level)}
                      <div>
                        <p className="font-medium">
                          {tier.name}
                          <span className="text-sm font-normal text-muted-foreground ml-2">
                            Level {tier.level}
                          </span>
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {tier.description || "No description"}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="text-right text-sm mr-2">
                        <p className="font-medium">{tier.user_count} users</p>
                        <p className="text-muted-foreground">{tier.document_count} docs</p>
                      </div>
                      <Button variant="outline" size="sm" onClick={() => handleOpenConfigureTier(tier)}>
                        Configure
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-8 w-8 text-red-500 hover:text-red-600 hover:bg-red-50"
                        onClick={() => handleDeleteTier(tier.id, tier.name, tier.user_count)}
                        disabled={deleteTier.isPending}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Add User Dialog */}
      <Dialog open={showAddUserDialog} onOpenChange={setShowAddUserDialog}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Add New User</DialogTitle>
            <DialogDescription>
              Create a new user account with specified access tier and folder permissions.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="email">Email *</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="user@example.com"
                  value={newUserEmail}
                  onChange={(e) => setNewUserEmail(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password *</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter password"
                  value={newUserPassword}
                  onChange={(e) => setNewUserPassword(e.target.value)}
                />
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  placeholder="John Doe"
                  value={newUserName}
                  onChange={(e) => setNewUserName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="tier">Access Tier *</Label>
                <Select value={newUserTierId} onValueChange={setNewUserTierId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select tier" />
                  </SelectTrigger>
                  <SelectContent>
                    {tiers.map((tier) => (
                      <SelectItem key={tier.id} value={tier.id}>
                        {tier.name} (Level {tier.level})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Folder Access Mode */}
            <div className="border rounded-lg p-4 space-y-4 bg-muted/30">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-sm font-medium flex items-center gap-2">
                    <Lock className="h-4 w-4" />
                    Restrict to Folder Access Only
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    User will ONLY see explicitly granted folders (ignores tier-based access)
                  </p>
                </div>
                <Switch
                  checked={newUserFolderOnly}
                  onCheckedChange={setNewUserFolderOnly}
                />
              </div>

              {/* Initial Folder Permissions */}
              <div className="space-y-2">
                <Label className="text-sm font-medium">Initial Folder Access</Label>
                <div className="flex gap-2">
                  <Select value={selectedFolderId} onValueChange={setSelectedFolderId}>
                    <SelectTrigger className="flex-1">
                      <SelectValue placeholder="Select folder..." />
                    </SelectTrigger>
                    <SelectContent>
                      {flatFolders
                        .filter((f) => !newUserFolderPermissions.some((p) => p.folder_id === f.id))
                        .map((folder) => (
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
                        ))}
                    </SelectContent>
                  </Select>
                  <Select
                    value={selectedPermissionLevel}
                    onValueChange={(v) => setSelectedPermissionLevel(v as 'view' | 'edit' | 'manage')}
                  >
                    <SelectTrigger className="w-28">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="view">View</SelectItem>
                      <SelectItem value="edit">Edit</SelectItem>
                      <SelectItem value="manage">Manage</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={handleAddFolderPermission}
                    disabled={!selectedFolderId}
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>

                {/* Selected Folders List */}
                {newUserFolderPermissions.length > 0 && (
                  <div className="space-y-1 mt-2">
                    {newUserFolderPermissions.map((perm) => (
                      <div
                        key={perm.folder_id}
                        className="flex items-center justify-between bg-background border rounded px-3 py-2"
                      >
                        <div className="flex items-center gap-2">
                          <Folder className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm">{perm.folder_name}</span>
                          <Badge variant="secondary" className="text-xs">
                            {perm.permission_level}
                          </Badge>
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => handleRemoveFolderPermission(perm.folder_id)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddUserDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateUser}
              disabled={!newUserEmail || !newUserPassword || !newUserTierId || createUser.isPending}
            >
              {createUser.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Create User
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Configure Tier Dialog */}
      <Dialog open={showConfigureTierDialog} onOpenChange={setShowConfigureTierDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Configure Access Tier</DialogTitle>
            <DialogDescription>
              Update the tier settings. Changes affect all users with this tier.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="tierName">Name</Label>
              <Input
                id="tierName"
                value={tierName}
                onChange={(e) => setTierName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="tierLevel">Level (0-100)</Label>
              <Input
                id="tierLevel"
                type="number"
                min={0}
                max={100}
                value={tierLevel}
                onChange={(e) => setTierLevel(parseInt(e.target.value) || 0)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="tierDescription">Description</Label>
              <Input
                id="tierDescription"
                placeholder="Tier description..."
                value={tierDescription}
                onChange={(e) => setTierDescription(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="tierColor">Color</Label>
              <div className="flex gap-2">
                <Input
                  id="tierColor"
                  type="color"
                  className="w-12 h-10 p-1"
                  value={tierColor}
                  onChange={(e) => setTierColor(e.target.value)}
                />
                <Input
                  value={tierColor}
                  onChange={(e) => setTierColor(e.target.value)}
                  placeholder="#6B7280"
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowConfigureTierDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleUpdateTier} disabled={!tierName || updateTier.isPending}>
              {updateTier.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Change User Tier Dialog */}
      <Dialog open={showChangeTierDialog} onOpenChange={setShowChangeTierDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Change Access Tier</DialogTitle>
            <DialogDescription>
              Change access tier for {changingUser?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>New Access Tier</Label>
              <Select value={selectedNewTierId} onValueChange={setSelectedNewTierId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select tier" />
                </SelectTrigger>
                <SelectContent>
                  {tiers.map((tier) => (
                    <SelectItem key={tier.id} value={tier.id}>
                      {tier.name} (Level {tier.level})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowChangeTierDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleChangeUserTier} disabled={!selectedNewTierId || updateUser.isPending}>
              {updateUser.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Update Tier
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add Tier Dialog */}
      <Dialog open={showAddTierDialog} onOpenChange={setShowAddTierDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add New Access Tier</DialogTitle>
            <DialogDescription>
              Create a new access tier for controlling document permissions.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="newTierName">Name *</Label>
              <Input
                id="newTierName"
                placeholder="e.g., Manager, Viewer, Premium"
                value={newTierName}
                onChange={(e) => setNewTierName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="newTierLevel">Level (0-100) *</Label>
              <Input
                id="newTierLevel"
                type="number"
                min={0}
                max={100}
                value={newTierLevel}
                onChange={(e) => setNewTierLevel(parseInt(e.target.value) || 0)}
              />
              <p className="text-xs text-muted-foreground">
                Higher levels can access documents from lower levels
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="newTierDescription">Description</Label>
              <Input
                id="newTierDescription"
                placeholder="Tier description..."
                value={newTierDescription}
                onChange={(e) => setNewTierDescription(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="newTierColor">Color</Label>
              <div className="flex gap-2">
                <Input
                  id="newTierColor"
                  type="color"
                  className="w-12 h-10 p-1"
                  value={newTierColor}
                  onChange={(e) => setNewTierColor(e.target.value)}
                />
                <Input
                  value={newTierColor}
                  onChange={(e) => setNewTierColor(e.target.value)}
                  placeholder="#6B7280"
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddTierDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateTier}
              disabled={!newTierName || createTier.isPending}
            >
              {createTier.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Create Tier
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* User Folder Permissions Dialog */}
      {folderPermissionsUser && (
        <UserFolderPermissionsDialog
          userId={folderPermissionsUser.id}
          userName={folderPermissionsUser.name}
          userEmail={folderPermissionsUser.email}
          open={!!folderPermissionsUser}
          onOpenChange={(open) => !open && setFolderPermissionsUser(null)}
        />
      )}
    </div>
  );
}
