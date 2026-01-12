"use client";

import { useState, useEffect } from "react";
import { Building2, ChevronDown, Check, Plus, Shield } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { useSession } from "next-auth/react";

// =============================================================================
// Types
// =============================================================================

interface Organization {
  id: string;
  name: string;
  slug: string;
  plan: string;
  is_active: boolean;
}

interface OrganizationContext {
  user_id: string;
  email: string;
  name: string | null;
  is_superadmin: boolean;
  current_organization_id: string | null;
  current_organization: Organization | null;
  available_organizations: Organization[];
}

// =============================================================================
// Helper Functions
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// =============================================================================
// Organization Switcher Component
// =============================================================================

export function OrganizationSwitcher() {
  const [context, setContext] = useState<OrganizationContext | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSwitching, setIsSwitching] = useState(false);
  const router = useRouter();
  const { data: session, status } = useSession();

  // Get auth headers using session token
  const getAuthHeaders = (): HeadersInit => {
    const token = (session as any)?.accessToken;
    if (!token) {
      throw new Error("Not authenticated");
    }
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    };
  };

  // Fetch organization context when session is available
  useEffect(() => {
    const fetchContext = async () => {
      const token = (session as any)?.accessToken;
      if (!token) {
        // Token not ready yet, don't fetch
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/organizations/me/context`, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setContext(data);
        } else if (response.status === 401) {
          // Not authenticated, might not have the feature
          setContext(null);
        }
      } catch (error) {
        console.error("Failed to fetch organization context:", error);
      } finally {
        setIsLoading(false);
      }
    };

    // Check if we have a session with token before fetching
    if (status === "authenticated" && (session as any)?.accessToken) {
      fetchContext();
    } else if (status !== "loading") {
      setIsLoading(false);
    }
  }, [session, status]);

  // Switch organization
  const switchOrganization = async (orgId: string) => {
    if (isSwitching) return;

    setIsSwitching(true);
    try {
      const response = await fetch(`${API_BASE}/organizations/me/switch`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({ organization_id: orgId }),
      });

      if (response.ok) {
        const data = await response.json();
        setContext(data);
        toast.success(`Switched to ${data.current_organization?.name || "organization"}`);
        // Refresh the page to update all data for the new organization context
        router.refresh();
      } else {
        const error = await response.json();
        toast.error(error.detail || "Failed to switch organization");
      }
    } catch (error) {
      console.error("Failed to switch organization:", error);
      toast.error("Failed to switch organization");
    } finally {
      setIsSwitching(false);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="px-3 py-2">
        <Skeleton className="h-10 w-full rounded-lg" />
      </div>
    );
  }

  // No context or not a multi-org user
  if (!context || context.available_organizations.length === 0) {
    return null;
  }

  // Don't show if only one organization and not superadmin
  if (context.available_organizations.length === 1 && !context.is_superadmin) {
    return null;
  }

  const currentOrg = context.current_organization || context.available_organizations[0];

  return (
    <div className="px-3 py-2">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className="w-full justify-between h-auto py-2 px-3"
            disabled={isSwitching}
          >
            <div className="flex items-center gap-2 min-w-0">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                <Building2 className="h-4 w-4 text-primary" />
              </div>
              <div className="flex flex-col items-start min-w-0">
                <span className="text-sm font-medium truncate max-w-[120px]">
                  {currentOrg?.name || "Select Organization"}
                </span>
                <span className="text-xs text-muted-foreground capitalize">
                  {currentOrg?.plan || "free"} plan
                </span>
              </div>
            </div>
            <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-64">
          <DropdownMenuLabel className="flex items-center justify-between">
            <span>Organizations</span>
            {context.is_superadmin && (
              <Badge variant="outline" className="text-xs">
                <Shield className="h-3 w-3 mr-1" />
                Superadmin
              </Badge>
            )}
          </DropdownMenuLabel>
          <DropdownMenuSeparator />

          {/* Organization List */}
          <div className="max-h-64 overflow-y-auto">
            {context.available_organizations.map((org) => (
              <DropdownMenuItem
                key={org.id}
                onClick={() => switchOrganization(org.id)}
                className={cn(
                  "cursor-pointer flex items-center justify-between",
                  context.current_organization_id === org.id && "bg-accent"
                )}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <div className="flex h-7 w-7 items-center justify-center rounded bg-muted">
                    <Building2 className="h-3.5 w-3.5" />
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="text-sm truncate max-w-[150px]">{org.name}</span>
                    <span className="text-xs text-muted-foreground capitalize">{org.plan}</span>
                  </div>
                </div>
                {context.current_organization_id === org.id && (
                  <Check className="h-4 w-4 text-primary shrink-0" />
                )}
              </DropdownMenuItem>
            ))}
          </div>

          {/* Superadmin: Create New Organization */}
          {context.is_superadmin && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => router.push("/dashboard/admin/organizations")}
                className="cursor-pointer"
              >
                <Plus className="h-4 w-4 mr-2" />
                Manage Organizations
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
