"use client";

import { useState } from "react";
import { getErrorMessage } from "@/lib/errors";
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
} from "@/lib/api/hooks";
import { useUser } from "@/lib/auth";

const getTierIcon = (level: number) => {
  if (level >= 80) return <ShieldCheck className="h-4 w-4 text-green-500" />;
  if (level >= 50) return <Shield className="h-4 w-4 text-blue-500" />;
  return <ShieldAlert className="h-4 w-4 text-yellow-500" />;
};

export default function AdminUsersPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(1);
  const { isAuthenticated, isLoading: authLoading } = useUser();

  // Real API calls - only fetch when authenticated
  const { data: usersData, isLoading: usersLoading, error: usersError, refetch: refetchUsers } = useAdminUsers({
    page,
    page_size: 20,
    search: searchQuery || undefined,
  }, { enabled: isAuthenticated });

  const { data: tiersData, isLoading: tiersLoading } = useAccessTiers({ enabled: isAuthenticated });
  const { data: statsData, isLoading: statsLoading } = useAdminStats({ enabled: isAuthenticated });
  const updateUser = useUpdateAdminUser();

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
          <Button disabled>
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
                            <DropdownMenuItem disabled className="text-muted-foreground">
                              Change Access Tier
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
        <CardHeader>
          <CardTitle className="text-lg">Access Tiers</CardTitle>
          <CardDescription>Document access permissions by tier</CardDescription>
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
                    <div className="flex items-center gap-4">
                      <div className="text-right text-sm">
                        <p className="font-medium">{tier.user_count} users</p>
                        <p className="text-muted-foreground">{tier.document_count} docs</p>
                      </div>
                      <Button variant="outline" size="sm" disabled>
                        Configure
                      </Button>
                    </div>
                  </div>
                ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
