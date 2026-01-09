"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import {
  Shield,
  Search,
  Filter,
  RefreshCw,
  User,
  Clock,
  FileText,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Globe,
  Key,
  Upload,
  Trash2,
  Edit,
  Eye,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
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
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  useAuditLogs,
  useAuditActions,
  useSecurityAuditLogs,
  type AuditLogEntry,
} from "@/lib/api";

// Action type styling
const actionConfig: Record<string, { color: string; icon: React.ElementType; label: string }> = {
  login: { color: "text-green-600", icon: Key, label: "Login" },
  logout: { color: "text-gray-600", icon: Key, label: "Logout" },
  login_failed: { color: "text-red-600", icon: AlertTriangle, label: "Login Failed" },
  user_created: { color: "text-blue-600", icon: User, label: "User Created" },
  user_updated: { color: "text-yellow-600", icon: Edit, label: "User Updated" },
  user_deleted: { color: "text-red-600", icon: Trash2, label: "User Deleted" },
  document_uploaded: { color: "text-blue-600", icon: Upload, label: "Document Uploaded" },
  document_deleted: { color: "text-red-600", icon: Trash2, label: "Document Deleted" },
  document_viewed: { color: "text-gray-600", icon: Eye, label: "Document Viewed" },
  settings_updated: { color: "text-yellow-600", icon: Edit, label: "Settings Updated" },
  chat_session: { color: "text-purple-600", icon: FileText, label: "Chat Session" },
  generation_job: { color: "text-indigo-600", icon: FileText, label: "Generation Job" },
};

function getActionConfig(action: string) {
  return actionConfig[action] || { color: "text-gray-500", icon: FileText, label: action };
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return formatDate(dateString);
}

export default function AuditLogsPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [page, setPage] = useState(1);
  const [pageSize] = useState(25);
  const [actionFilter, setActionFilter] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [showSecurityOnly, setShowSecurityOnly] = useState(false);
  const [selectedLog, setSelectedLog] = useState<AuditLogEntry | null>(null);

  // Queries
  const { data: auditLogs, isLoading, refetch } = useAuditLogs(
    {
      page,
      page_size: pageSize,
      action: actionFilter !== "all" ? actionFilter : undefined,
    },
    { enabled: isAuthenticated && !showSecurityOnly }
  );

  const { data: securityLogs, isLoading: securityLoading } = useSecurityAuditLogs({
    enabled: isAuthenticated && showSecurityOnly,
  });

  const { data: actions } = useAuditActions({ enabled: isAuthenticated });

  // Ensure logs is always an array (handle API response format changes)
  const rawSecurityLogs = Array.isArray(securityLogs) ? securityLogs : [];
  const rawAuditLogs = Array.isArray(auditLogs?.logs) ? auditLogs.logs : [];
  const logs = showSecurityOnly ? rawSecurityLogs : rawAuditLogs;
  const total = showSecurityOnly ? rawSecurityLogs.length : (auditLogs?.total || 0);
  const loading = showSecurityOnly ? securityLoading : isLoading;

  const handleRefresh = () => {
    refetch();
  };

  const filteredLogs = logs.filter((log) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      log.user_email?.toLowerCase().includes(query) ||
      log.action.toLowerCase().includes(query) ||
      log.resource_type?.toLowerCase().includes(query) ||
      log.ip_address?.toLowerCase().includes(query)
    );
  });

  if (status === "loading") {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-muted-foreground">Please sign in to view audit logs.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Shield className="h-6 w-6" />
            Audit Logs
          </h1>
          <p className="text-muted-foreground">
            Track and monitor system activity and security events
          </p>
        </div>
        <Button variant="outline" onClick={handleRefresh} disabled={loading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Events
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{total}</div>
          </CardContent>
        </Card>

        <Card
          className={`cursor-pointer transition-colors ${showSecurityOnly ? "border-primary" : ""}`}
          onClick={() => setShowSecurityOnly(!showSecurityOnly)}
        >
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Security Events
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {showSecurityOnly ? "Active" : "View"}
            </div>
            <p className="text-xs text-muted-foreground">Click to toggle</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Action Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{actions?.length || 0}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Page
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {page} / {Math.ceil(total / pageSize) || 1}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by user, action, or IP..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <Select value={actionFilter} onValueChange={setActionFilter}>
              <SelectTrigger className="w-[180px]">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Filter by action" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Actions</SelectItem>
                {actions?.map((action) => (
                  <SelectItem key={action} value={action}>
                    {action.replace(/_/g, " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Audit Log Table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center justify-between">
            <span>
              {showSecurityOnly ? "Security Events" : "Activity Log"}
              {filteredLogs && ` (${filteredLogs.length} entries)`}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : filteredLogs && filteredLogs.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[180px]">Timestamp</TableHead>
                  <TableHead className="w-[150px]">Action</TableHead>
                  <TableHead>User</TableHead>
                  <TableHead>Resource</TableHead>
                  <TableHead className="w-[120px]">IP Address</TableHead>
                  <TableHead className="w-[100px]">Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredLogs.map((log) => {
                  const config = getActionConfig(log.action);
                  const Icon = config.icon;
                  return (
                    <TableRow key={log.id}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Clock className="h-3 w-3 text-muted-foreground" />
                          <span className="text-sm" title={formatDate(log.created_at)}>
                            {formatTimeAgo(log.created_at)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Icon className={`h-4 w-4 ${config.color}`} />
                          <Badge variant="outline" className="text-xs">
                            {config.label}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <User className="h-3 w-3 text-muted-foreground" />
                          <span className="text-sm truncate max-w-[150px]">
                            {log.user_email || "System"}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        {log.resource_type && (
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary" className="text-xs">
                              {log.resource_type}
                            </Badge>
                            {log.resource_id && (
                              <span className="text-xs text-muted-foreground truncate max-w-[100px]">
                                {log.resource_id.slice(0, 8)}...
                              </span>
                            )}
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        {log.ip_address && (
                          <div className="flex items-center gap-1">
                            <Globe className="h-3 w-3 text-muted-foreground" />
                            <span className="text-xs font-mono">{log.ip_address}</span>
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        {log.details && Object.keys(log.details).length > 0 && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2"
                            onClick={() => setSelectedLog(log)}
                          >
                            <Eye className="h-3 w-3" />
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Shield className="h-12 w-12 mb-4 opacity-50" />
              <p>No audit log entries found</p>
              {searchQuery && <p className="text-sm">Try adjusting your search</p>}
            </div>
          )}

          {/* Pagination */}
          {!showSecurityOnly && auditLogs && auditLogs.total > pageSize && (
            <div className="flex items-center justify-between mt-4 pt-4 border-t">
              <p className="text-sm text-muted-foreground">
                Showing {(page - 1) * pageSize + 1} to{" "}
                {Math.min(page * pageSize, auditLogs.total)} of {auditLogs.total} entries
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => p + 1)}
                  disabled={!auditLogs.has_more}
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Audit Log Details Dialog */}
      <Dialog open={!!selectedLog} onOpenChange={(open) => !open && setSelectedLog(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Audit Log Details
            </DialogTitle>
          </DialogHeader>
          {selectedLog && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Action:</span>
                  <Badge className="ml-2" variant="outline">
                    {selectedLog.action}
                  </Badge>
                </div>
                <div>
                  <span className="text-muted-foreground">Timestamp:</span>
                  <span className="ml-2">{formatDate(selectedLog.created_at)}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">User:</span>
                  <span className="ml-2">{selectedLog.user_email || "System"}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">IP Address:</span>
                  <span className="ml-2 font-mono">{selectedLog.ip_address || "N/A"}</span>
                </div>
                {selectedLog.resource_type && (
                  <div>
                    <span className="text-muted-foreground">Resource Type:</span>
                    <Badge className="ml-2" variant="secondary">
                      {selectedLog.resource_type}
                    </Badge>
                  </div>
                )}
                {selectedLog.resource_id && (
                  <div>
                    <span className="text-muted-foreground">Resource ID:</span>
                    <span className="ml-2 font-mono text-xs">{selectedLog.resource_id}</span>
                  </div>
                )}
              </div>
              <div>
                <span className="text-sm text-muted-foreground block mb-2">Details:</span>
                <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto max-h-96 whitespace-pre-wrap">
                  {JSON.stringify(selectedLog.details, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
