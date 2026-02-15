"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Building2,
  Plus,
  Search,
  MoreHorizontal,
  Users,
  Settings,
  Trash2,
  Edit,
  Shield,
  CheckCircle,
  XCircle,
  Loader2,
  BarChart3,
  HardDrive,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { api } from "@/lib/api/client";

// Types
interface Organization {
  id: string;
  name: string;
  slug: string;
  plan: string;
  settings: Record<string, unknown>;
  max_users: number;
  max_storage_gb: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  member_count: number;
  document_count?: number;
  storage_used_gb?: number;
}

interface OrganizationStats {
  total_organizations: number;
  active_organizations: number;
  total_users: number;
  total_documents: number;
  total_storage_gb: number;
  organizations_by_plan: Record<string, number>;
}

interface Member {
  id: string;
  user_id: string;
  email: string;
  name: string | null;
  role: string;
  joined_at: string;
}

// API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// Helper to get auth headers - token is passed from component
const getAuthHeaders = (token: string | undefined): HeadersInit => {
  if (!token) {
    throw new Error("Not authenticated");
  }
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };
};

// API functions - all require token parameter
const fetchOrganizations = async (token: string, params: {
  page: number;
  page_size: number;
  search?: string;
  plan?: string;
  is_active?: boolean;
}) => {
  const searchParams = new URLSearchParams();
  searchParams.set("page", String(params.page));
  searchParams.set("page_size", String(params.page_size));
  if (params.search) searchParams.set("search", params.search);
  if (params.plan && params.plan !== "all") searchParams.set("plan", params.plan);
  if (params.is_active !== undefined) searchParams.set("is_active", String(params.is_active));

  const response = await fetch(`${API_BASE_URL}/organizations?${searchParams.toString()}`, {
    headers: getAuthHeaders(token),
    credentials: "include",
  });
  if (!response.ok) throw new Error("Failed to fetch organizations");
  return response.json();
};

const fetchOrganizationStats = async (token: string): Promise<OrganizationStats> => {
  console.log("[Stats] Fetching from:", `${API_BASE_URL}/organizations/stats`);
  const response = await fetch(`${API_BASE_URL}/organizations/stats`, {
    headers: getAuthHeaders(token),
    credentials: "include",
  });
  if (!response.ok) {
    console.error("[Stats] Failed:", response.status, response.statusText);
    throw new Error("Failed to fetch stats");
  }
  const data = await response.json();
  console.log("[Stats] Response:", data);
  return data;
};

const fetchOrganizationMembers = async (token: string, orgId: string): Promise<Member[]> => {
  const response = await fetch(`${API_BASE_URL}/organizations/${orgId}/members`, {
    headers: getAuthHeaders(token),
    credentials: "include",
  });
  if (!response.ok) throw new Error("Failed to fetch members");
  return response.json();
};

const fetchOrganizationFeatures = async (token: string, orgId: string) => {
  const response = await fetch(`${API_BASE_URL}/organizations/${orgId}/features`, {
    headers: getAuthHeaders(token),
    credentials: "include",
  });
  if (!response.ok) throw new Error("Failed to fetch features");
  return response.json();
};

interface UserContext {
  user_id: string;
  email: string;
  name: string | null;
  is_superadmin: boolean;
  current_organization_id: string | null;
}

export default function OrganizationsPage() {
  const { data: session, status: sessionStatus } = useSession();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [planFilter, setPlanFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [page, setPage] = useState(1);
  const [selectedOrg, setSelectedOrg] = useState<Organization | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isMembersOpen, setIsMembersOpen] = useState(false);
  const [isFeaturesOpen, setIsFeaturesOpen] = useState(false);
  const [userContext, setUserContext] = useState<UserContext | null>(null);
  const [isCheckingAccess, setIsCheckingAccess] = useState(true);

  // Get access token from session
  const accessToken = (session as any)?.accessToken as string | undefined;
  const isAuthenticated = sessionStatus === "authenticated" && !!accessToken;

  // Check superadmin status on mount when session is available
  useEffect(() => {
    const checkAuth = async () => {
      if (!accessToken) {
        setIsCheckingAccess(false);
        return;
      }

      try {
        const response = await fetch(`${API_BASE_URL}/organizations/me/context`, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          console.log("User context loaded:", data);
          setUserContext(data);
        } else {
          const errorText = await response.text();
          console.error("Failed to fetch user context:", response.status, errorText);
        }
      } catch (error) {
        console.error("Failed to check access:", error);
      }
      setIsCheckingAccess(false);
    };

    if (sessionStatus !== "loading") {
      checkAuth();
    }
  }, [accessToken, sessionStatus]);

  // Form state
  const [formData, setFormData] = useState({
    name: "",
    slug: "",
    plan: "free",
    max_users: 5,
    max_storage_gb: 10,
  });

  // Queries - only run when authenticated with token
  const { data: statsData } = useQuery({
    queryKey: ["organizations", "stats", accessToken],
    queryFn: () => fetchOrganizationStats(accessToken!),
    enabled: isAuthenticated && !!accessToken,
  });

  const { data: orgsData, isLoading } = useQuery({
    queryKey: ["organizations", { page, search, plan: planFilter, is_active: statusFilter }, accessToken],
    queryFn: () =>
      fetchOrganizations(accessToken!, {
        page,
        page_size: 20,
        search: search || undefined,
        plan: planFilter !== "all" ? planFilter : undefined,
        is_active: statusFilter !== "all" ? statusFilter === "active" : undefined,
      }),
    enabled: isAuthenticated && !!accessToken,
  });

  const { data: membersData } = useQuery({
    queryKey: ["organizations", selectedOrg?.id, "members", accessToken],
    queryFn: () => fetchOrganizationMembers(accessToken!, selectedOrg!.id),
    enabled: isAuthenticated && !!accessToken && !!selectedOrg && isMembersOpen,
  });

  const { data: featuresData } = useQuery({
    queryKey: ["organizations", selectedOrg?.id, "features", accessToken],
    queryFn: () => fetchOrganizationFeatures(accessToken!, selectedOrg!.id),
    enabled: isAuthenticated && !!accessToken && !!selectedOrg && isFeaturesOpen,
  });

  // Mutations
  const createOrg = useMutation({
    mutationFn: async (data: typeof formData) => {
      const response = await fetch(`${API_BASE_URL}/organizations`, {
        method: "POST",
        headers: getAuthHeaders(accessToken),
        credentials: "include",
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to create organization");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations"] });
      setIsCreateOpen(false);
      setFormData({ name: "", slug: "", plan: "free", max_users: 5, max_storage_gb: 10 });
      toast.success("Organization created successfully");
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });

  const updateOrg = useMutation({
    mutationFn: async ({ id, data }: { id: string; data: Partial<Organization> }) => {
      const response = await fetch(`${API_BASE_URL}/organizations/${id}`, {
        method: "PATCH",
        headers: getAuthHeaders(accessToken),
        credentials: "include",
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error("Failed to update organization");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations"] });
      setIsEditOpen(false);
      toast.success("Organization updated successfully");
    },
    onError: () => {
      toast.error("Failed to update organization");
    },
  });

  const deleteOrg = useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`${API_BASE_URL}/organizations/${id}`, {
        method: "DELETE",
        headers: getAuthHeaders(accessToken),
        credentials: "include",
      });
      if (!response.ok) throw new Error("Failed to delete organization");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations"] });
      toast.success("Organization deleted successfully");
    },
    onError: () => {
      toast.error("Failed to delete organization");
    },
  });

  const updateFeatures = useMutation({
    mutationFn: async ({ id, flags }: { id: string; flags: Record<string, boolean> }) => {
      const response = await fetch(`${API_BASE_URL}/organizations/${id}/features`, {
        method: "PATCH",
        headers: getAuthHeaders(accessToken),
        credentials: "include",
        body: JSON.stringify({ flags }),
      });
      if (!response.ok) throw new Error("Failed to update features");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations", selectedOrg?.id, "features"] });
      toast.success("Features updated successfully");
    },
    onError: () => {
      toast.error("Failed to update features");
    },
  });

  const handleCreateOrg = () => {
    createOrg.mutate(formData);
  };

  const handleEditOrg = (org: Organization) => {
    setSelectedOrg(org);
    setFormData({
      name: org.name,
      slug: org.slug,
      plan: org.plan,
      max_users: org.max_users,
      max_storage_gb: org.max_storage_gb,
    });
    setIsEditOpen(true);
  };

  const handleUpdateOrg = () => {
    if (!selectedOrg) return;
    updateOrg.mutate({
      id: selectedOrg.id,
      data: {
        name: formData.name,
        plan: formData.plan,
        max_users: formData.max_users,
        max_storage_gb: formData.max_storage_gb,
      },
    });
  };

  const handleToggleFeature = (feature: string, enabled: boolean) => {
    if (!selectedOrg) return;
    updateFeatures.mutate({
      id: selectedOrg.id,
      flags: { [feature]: enabled },
    });
  };

  const getPlanBadgeVariant = (plan: string) => {
    switch (plan) {
      case "enterprise":
        return "default";
      case "pro":
        return "secondary";
      default:
        return "outline";
    }
  };

  // Show loading state while checking access or session
  if (isCheckingAccess || sessionStatus === "loading") {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // Show access denied for non-superadmin users
  if (!userContext?.is_superadmin) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <Shield className="h-16 w-16 text-muted-foreground" />
        <h2 className="text-2xl font-bold">Access Denied</h2>
        <p className="text-muted-foreground text-center max-w-md">
          Only superadmin users can access the Organizations management page.
          Contact your system administrator if you need access.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Organizations</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{statsData?.total_organizations ?? 0}</div>
            <p className="text-xs text-muted-foreground">
              {statsData?.active_organizations ?? 0} active
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{statsData?.total_users ?? 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Documents</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{statsData?.total_documents ?? 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{statsData?.total_storage_gb?.toFixed(2) ?? 0} GB</div>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Actions */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4 flex-1">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search organizations..."
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              className="pl-10"
            />
          </div>
          <Select value={planFilter} onValueChange={(v) => { setPlanFilter(v); setPage(1); }}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="All Plans" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Plans</SelectItem>
              <SelectItem value="free">Free</SelectItem>
              <SelectItem value="pro">Pro</SelectItem>
              <SelectItem value="enterprise">Enterprise</SelectItem>
            </SelectContent>
          </Select>
          <Select value={statusFilter} onValueChange={(v) => { setStatusFilter(v); setPage(1); }}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="All Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="inactive">Inactive</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              New Organization
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Organization</DialogTitle>
              <DialogDescription>Add a new organization to the system.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="Acme Corp"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="slug">Slug</Label>
                <Input
                  id="slug"
                  value={formData.slug}
                  onChange={(e) => setFormData({ ...formData, slug: e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "-") })}
                  placeholder="acme-corp"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="plan">Plan</Label>
                <Select value={formData.plan} onValueChange={(v) => setFormData({ ...formData, plan: v })}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="free">Free</SelectItem>
                    <SelectItem value="pro">Pro</SelectItem>
                    <SelectItem value="enterprise">Enterprise</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="max_users">Max Users</Label>
                  <Input
                    id="max_users"
                    type="number"
                    value={formData.max_users}
                    onChange={(e) => setFormData({ ...formData, max_users: parseInt(e.target.value) || 5 })}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="max_storage_gb">Max Storage (GB)</Label>
                  <Input
                    id="max_storage_gb"
                    type="number"
                    value={formData.max_storage_gb}
                    onChange={(e) => setFormData({ ...formData, max_storage_gb: parseInt(e.target.value) || 10 })}
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateOrg} disabled={createOrg.isPending}>
                {createOrg.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Organizations Table */}
      <Card>
        <CardHeader>
          <CardTitle>Organizations</CardTitle>
          <CardDescription>Manage all organizations in the system.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Organization</TableHead>
                  <TableHead>Plan</TableHead>
                  <TableHead>Members</TableHead>
                  <TableHead>Limits</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {orgsData?.organizations?.map((org: Organization) => (
                  <TableRow key={org.id}>
                    <TableCell>
                      <div>
                        <div className="font-medium">{org.name}</div>
                        <div className="text-sm text-muted-foreground">{org.slug}</div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getPlanBadgeVariant(org.plan)}>{org.plan}</Badge>
                    </TableCell>
                    <TableCell>{org.member_count}</TableCell>
                    <TableCell>
                      <div className="text-sm">
                        <div>{org.max_users} users</div>
                        <div className="text-muted-foreground">{org.max_storage_gb} GB</div>
                      </div>
                    </TableCell>
                    <TableCell>
                      {org.is_active ? (
                        <Badge variant="default" className="bg-green-500">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Active
                        </Badge>
                      ) : (
                        <Badge variant="destructive">
                          <XCircle className="h-3 w-3 mr-1" />
                          Inactive
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>{new Date(org.created_at).toLocaleDateString()}</TableCell>
                    <TableCell className="text-right">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handleEditOrg(org)}>
                            <Edit className="h-4 w-4 mr-2" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => {
                              setSelectedOrg(org);
                              setIsMembersOpen(true);
                            }}
                          >
                            <Users className="h-4 w-4 mr-2" />
                            Members
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => {
                              setSelectedOrg(org);
                              setIsFeaturesOpen(true);
                            }}
                          >
                            <Shield className="h-4 w-4 mr-2" />
                            Features
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={() => {
                              if (confirm("Are you sure you want to delete this organization?")) {
                                deleteOrg.mutate(org.id);
                              }
                            }}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
                {(!orgsData?.organizations || orgsData.organizations.length === 0) && (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                      No organizations found
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
          {orgsData?.has_more && (
            <div className="flex justify-center mt-4">
              <Button variant="outline" onClick={() => setPage(page + 1)}>
                Load More
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Edit Organization Dialog */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Organization</DialogTitle>
            <DialogDescription>Update organization details.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="edit-name">Name</Label>
              <Input
                id="edit-name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-plan">Plan</Label>
              <Select value={formData.plan} onValueChange={(v) => setFormData({ ...formData, plan: v })}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="free">Free</SelectItem>
                  <SelectItem value="pro">Pro</SelectItem>
                  <SelectItem value="enterprise">Enterprise</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="grid gap-2">
                <Label htmlFor="edit-max_users">Max Users</Label>
                <Input
                  id="edit-max_users"
                  type="number"
                  value={formData.max_users}
                  onChange={(e) => setFormData({ ...formData, max_users: parseInt(e.target.value) || 5 })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="edit-max_storage_gb">Max Storage (GB)</Label>
                <Input
                  id="edit-max_storage_gb"
                  type="number"
                  value={formData.max_storage_gb}
                  onChange={(e) => setFormData({ ...formData, max_storage_gb: parseInt(e.target.value) || 10 })}
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleUpdateOrg} disabled={updateOrg.isPending}>
              {updateOrg.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Members Dialog */}
      <Dialog open={isMembersOpen} onOpenChange={setIsMembersOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Organization Members</DialogTitle>
            <DialogDescription>
              {selectedOrg?.name} - {selectedOrg?.member_count} members
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>User</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Joined</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {membersData?.map((member) => (
                  <TableRow key={member.id}>
                    <TableCell>
                      <div>
                        <div className="font-medium">{member.name || "No name"}</div>
                        <div className="text-sm text-muted-foreground">{member.email}</div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={member.role === "owner" ? "default" : "secondary"}>
                        {member.role}
                      </Badge>
                    </TableCell>
                    <TableCell>{new Date(member.joined_at).toLocaleDateString()}</TableCell>
                  </TableRow>
                ))}
                {(!membersData || membersData.length === 0) && (
                  <TableRow>
                    <TableCell colSpan={3} className="text-center text-muted-foreground py-8">
                      No members found
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </DialogContent>
      </Dialog>

      {/* Features Dialog */}
      <Dialog open={isFeaturesOpen} onOpenChange={setIsFeaturesOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Feature Flags</DialogTitle>
            <DialogDescription>
              Manage features for {selectedOrg?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="py-4 space-y-4">
            {featuresData?.features &&
              Object.entries(featuresData.features).map(([feature, enabled]) => (
                <div key={feature} className="flex items-center justify-between">
                  <div>
                    <div className="font-medium capitalize">{feature.replace(/_/g, " ")}</div>
                  </div>
                  <Switch
                    checked={enabled as boolean}
                    onCheckedChange={(checked) => handleToggleFeature(feature, checked)}
                  />
                </div>
              ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
