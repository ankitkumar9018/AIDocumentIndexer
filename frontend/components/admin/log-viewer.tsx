"use client";

/**
 * AIDocumentIndexer - Superadmin Log Viewer (Phase 52)
 * =====================================================
 *
 * Enhanced audit log viewer with severity filtering for superadmins.
 *
 * Features:
 * - Severity filter chips (DEBUG, INFO, NOTICE, WARNING, ERROR, CRITICAL)
 * - Color-coded log entries
 * - Date range picker
 * - Search/filter by action type
 * - Export to CSV
 * - Auto-refresh toggle
 * - Real-time severity counts
 */

import * as React from "react";
import {
  AlertCircle,
  AlertTriangle,
  Bug,
  Check,
  ChevronDown,
  Download,
  Filter,
  Info,
  RefreshCw,
  Search,
  Shield,
  XCircle,
  Clock,
  User,
  Activity,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

// =============================================================================
// TYPES
// =============================================================================

type LogSeverity = "debug" | "info" | "notice" | "warning" | "error" | "critical";

interface AuditLog {
  id: string;
  action: string;
  severity: LogSeverity;
  user_id?: string;
  user_email?: string;
  resource_type?: string;
  resource_id?: string;
  details?: Record<string, any>;
  ip_address?: string;
  created_at: string;
}

interface SeverityCounts {
  debug: number;
  info: number;
  notice: number;
  warning: number;
  error: number;
  critical: number;
}

interface LogViewerProps {
  className?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

const SEVERITY_CONFIG: Record<
  LogSeverity,
  {
    label: string;
    icon: React.ElementType;
    color: string;
    bgColor: string;
    borderColor: string;
  }
> = {
  debug: {
    label: "Debug",
    icon: Bug,
    color: "text-gray-600",
    bgColor: "bg-gray-100",
    borderColor: "border-gray-300",
  },
  info: {
    label: "Info",
    icon: Info,
    color: "text-blue-600",
    bgColor: "bg-blue-100",
    borderColor: "border-blue-300",
  },
  notice: {
    label: "Notice",
    icon: Check,
    color: "text-green-600",
    bgColor: "bg-green-100",
    borderColor: "border-green-300",
  },
  warning: {
    label: "Warning",
    icon: AlertTriangle,
    color: "text-yellow-600",
    bgColor: "bg-yellow-100",
    borderColor: "border-yellow-300",
  },
  error: {
    label: "Error",
    icon: XCircle,
    color: "text-red-600",
    bgColor: "bg-red-100",
    borderColor: "border-red-300",
  },
  critical: {
    label: "Critical",
    icon: AlertCircle,
    color: "text-red-800",
    bgColor: "bg-red-200",
    borderColor: "border-red-500",
  },
};

const SEVERITY_ORDER: LogSeverity[] = [
  "debug",
  "info",
  "notice",
  "warning",
  "error",
  "critical",
];

// Quick filter presets for common monitoring scenarios
const QUICK_FILTERS = [
  { label: "Service Fallbacks", filter: "service.fallback", color: "bg-yellow-100 text-yellow-800" },
  { label: "Service Errors", filter: "service.error", color: "bg-red-100 text-red-800" },
  { label: "Auth Issues", filter: "auth", color: "bg-orange-100 text-orange-800" },
  { label: "Admin Actions", filter: "admin", color: "bg-purple-100 text-purple-800" },
];

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function fetchLogs(params: {
  page?: number;
  page_size?: number;
  severity?: string;
  min_severity?: string;
  action?: string;
  start_date?: string;
  end_date?: string;
}): Promise<{ logs: AuditLog[]; total: number; has_more: boolean }> {
  try {
    const searchParams = new URLSearchParams();
    if (params.page) searchParams.set("page", params.page.toString());
    if (params.page_size) searchParams.set("page_size", params.page_size.toString());
    if (params.severity) searchParams.set("severity", params.severity);
    if (params.min_severity) searchParams.set("min_severity", params.min_severity);
    if (params.action) searchParams.set("action", params.action);
    if (params.start_date) searchParams.set("start_date", params.start_date);
    if (params.end_date) searchParams.set("end_date", params.end_date);

    const response = await fetch(`${API_BASE}/admin/audit-logs?${searchParams}`, {
      credentials: "include",
    });
    if (!response.ok) throw new Error("Failed to fetch logs");
    return await response.json();
  } catch (error) {
    console.error("Error fetching logs:", error);
    return { logs: [], total: 0, has_more: false };
  }
}

async function fetchSeverityCounts(hours: number = 24): Promise<SeverityCounts> {
  try {
    const response = await fetch(
      `${API_BASE}/admin/audit-logs/severity-counts?hours=${hours}`,
      { credentials: "include" }
    );
    if (!response.ok) throw new Error("Failed to fetch severity counts");
    return await response.json();
  } catch (error) {
    console.error("Error fetching severity counts:", error);
    return { debug: 0, info: 0, notice: 0, warning: 0, error: 0, critical: 0 };
  }
}

async function fetchAuditActions(): Promise<{ value: string; name: string }[]> {
  try {
    const response = await fetch(`${API_BASE}/admin/audit-logs/actions`, {
      credentials: "include",
    });
    if (!response.ok) throw new Error("Failed to fetch actions");
    const data = await response.json();
    return data.actions || [];
  } catch (error) {
    console.error("Error fetching actions:", error);
    return [];
  }
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function LogViewer({ className }: LogViewerProps) {
  // State
  const [logs, setLogs] = React.useState<AuditLog[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [pageSize] = React.useState(50);
  const [hasMore, setHasMore] = React.useState(false);

  // Filters
  const [selectedSeverity, setSelectedSeverity] = React.useState<LogSeverity | "all">("all");
  const [minSeverity, setMinSeverity] = React.useState<LogSeverity | "">("");
  const [selectedAction, setSelectedAction] = React.useState<string>("");
  const [dateRange, setDateRange] = React.useState<string>("24h");
  const [searchQuery, setSearchQuery] = React.useState<string>("");

  // Counts
  const [severityCounts, setSeverityCounts] = React.useState<SeverityCounts>({
    debug: 0,
    info: 0,
    notice: 0,
    warning: 0,
    error: 0,
    critical: 0,
  });

  // Actions
  const [actions, setActions] = React.useState<{ value: string; name: string }[]>([]);

  // Auto-refresh
  const [autoRefresh, setAutoRefresh] = React.useState(false);
  const [lastRefresh, setLastRefresh] = React.useState<Date>(new Date());

  // Calculate date range
  const getDateRange = React.useCallback(() => {
    const now = new Date();
    let startDate: Date;

    switch (dateRange) {
      case "1h":
        startDate = new Date(now.getTime() - 60 * 60 * 1000);
        break;
      case "24h":
        startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        break;
      case "7d":
        startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case "30d":
        startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
      default:
        startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    }

    return {
      start_date: startDate.toISOString(),
      end_date: now.toISOString(),
    };
  }, [dateRange]);

  // Load logs
  const loadLogs = React.useCallback(async () => {
    setLoading(true);
    const dates = getDateRange();

    const params: Parameters<typeof fetchLogs>[0] = {
      page,
      page_size: pageSize,
      ...dates,
    };

    if (selectedSeverity !== "all") {
      params.severity = selectedSeverity;
    } else if (minSeverity) {
      params.min_severity = minSeverity;
    }

    if (selectedAction) {
      params.action = selectedAction;
    }

    const result = await fetchLogs(params);
    setLogs(result.logs);
    setTotal(result.total);
    setHasMore(result.has_more);
    setLastRefresh(new Date());
    setLoading(false);
  }, [page, pageSize, selectedSeverity, minSeverity, selectedAction, getDateRange]);

  // Load severity counts
  const loadCounts = React.useCallback(async () => {
    const hours = dateRange === "1h" ? 1 : dateRange === "24h" ? 24 : dateRange === "7d" ? 168 : 720;
    const counts = await fetchSeverityCounts(hours);
    setSeverityCounts(counts);
  }, [dateRange]);

  // Load actions
  React.useEffect(() => {
    fetchAuditActions().then(setActions);
  }, []);

  // Load logs on filter change
  React.useEffect(() => {
    loadLogs();
    loadCounts();
  }, [loadLogs, loadCounts]);

  // Auto-refresh
  React.useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadLogs();
      loadCounts();
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, loadLogs, loadCounts]);

  // Filter logs by search query (client-side)
  const filteredLogs = React.useMemo(() => {
    if (!searchQuery) return logs;

    const query = searchQuery.toLowerCase();
    return logs.filter(
      (log) =>
        log.action.toLowerCase().includes(query) ||
        log.user_email?.toLowerCase().includes(query) ||
        log.user_id?.toLowerCase().includes(query) ||
        log.resource_type?.toLowerCase().includes(query) ||
        log.resource_id?.toLowerCase().includes(query) ||
        log.ip_address?.toLowerCase().includes(query)
    );
  }, [logs, searchQuery]);

  // Export logs
  const handleExport = async (format: "csv" | "json") => {
    const dates = getDateRange();
    const dataToExport = filteredLogs;

    if (format === "json") {
      const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit-logs-${new Date().toISOString().split("T")[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      const headers = [
        "ID",
        "Timestamp",
        "Severity",
        "Action",
        "User Email",
        "User ID",
        "Resource Type",
        "Resource ID",
        "IP Address",
        "Details",
      ];
      const rows = dataToExport.map((log) => [
        log.id,
        log.created_at,
        log.severity,
        log.action,
        log.user_email || "",
        log.user_id || "",
        log.resource_type || "",
        log.resource_id || "",
        log.ip_address || "",
        JSON.stringify(log.details || {}),
      ]);
      const csv = [headers, ...rows].map((row) => row.join(",")).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit-logs-${new Date().toISOString().split("T")[0]}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  // Calculate total counts
  const totalEvents = Object.values(severityCounts).reduce((a, b) => a + b, 0);
  const warningAndAbove = severityCounts.warning + severityCounts.error + severityCounts.critical;

  return (
    <div className={cn("space-y-6", className)}>
      {/* Severity Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {SEVERITY_ORDER.map((severity) => {
          const config = SEVERITY_CONFIG[severity];
          const Icon = config.icon;
          const count = severityCounts[severity];
          const isSelected = selectedSeverity === severity;

          return (
            <Card
              key={severity}
              className={cn(
                "cursor-pointer transition-all hover:shadow-md",
                isSelected && "ring-2 ring-primary"
              )}
              onClick={() =>
                setSelectedSeverity(isSelected ? "all" : severity)
              }
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className={cn("p-2 rounded-lg", config.bgColor)}>
                    <Icon className={cn("h-4 w-4", config.color)} />
                  </div>
                  <span className="text-2xl font-bold">{count}</span>
                </div>
                <p className="text-sm text-muted-foreground mt-2 capitalize">
                  {config.label}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Filters Bar */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-4">
            {/* Search */}
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search logs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>

            {/* Min Severity Filter */}
            <Select
              value={minSeverity}
              onValueChange={(v) => {
                setMinSeverity(v as LogSeverity | "");
                setSelectedSeverity("all");
              }}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Min Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Severities</SelectItem>
                {SEVERITY_ORDER.map((severity) => (
                  <SelectItem key={severity} value={severity} className="capitalize">
                    {SEVERITY_CONFIG[severity].label}+
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Action Filter */}
            <Select value={selectedAction} onValueChange={setSelectedAction}>
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="All Actions" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All Actions</SelectItem>
                {actions.map((action) => (
                  <SelectItem key={action.value} value={action.value}>
                    {action.name.replace(/_/g, " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Date Range */}
            <Select value={dateRange} onValueChange={setDateRange}>
              <SelectTrigger className="w-[150px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">Last hour</SelectItem>
                <SelectItem value="24h">Last 24 hours</SelectItem>
                <SelectItem value="7d">Last 7 days</SelectItem>
                <SelectItem value="30d">Last 30 days</SelectItem>
              </SelectContent>
            </Select>

            {/* Auto-refresh */}
            <div className="flex items-center gap-2">
              <Switch
                id="auto-refresh"
                checked={autoRefresh}
                onCheckedChange={setAutoRefresh}
              />
              <Label htmlFor="auto-refresh" className="text-sm">
                Auto-refresh
              </Label>
            </div>

            {/* Refresh Button */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => {
                      loadLogs();
                      loadCounts();
                    }}
                    disabled={loading}
                  >
                    <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Last refreshed: {lastRefresh.toLocaleTimeString()}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Export */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="gap-2">
                  <Download className="h-4 w-4" />
                  Export
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => handleExport("csv")}>
                  Export as CSV
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleExport("json")}>
                  Export as JSON
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Quick Filters for common monitoring scenarios */}
          <div className="flex items-center gap-2 mt-4 pt-4 border-t">
            <span className="text-sm text-muted-foreground">Quick filters:</span>
            {QUICK_FILTERS.map((qf) => (
              <Badge
                key={qf.filter}
                variant="outline"
                className={cn(
                  "cursor-pointer hover:opacity-80 transition-opacity",
                  selectedAction.startsWith(qf.filter) && qf.color
                )}
                onClick={() => {
                  // Find the first action that matches the filter prefix
                  const matchingAction = actions.find((a) => a.value.startsWith(qf.filter));
                  if (matchingAction) {
                    setSelectedAction(selectedAction === matchingAction.value ? "" : matchingAction.value);
                  } else {
                    // If no exact match, set a prefix filter that will show in search
                    setSearchQuery(qf.filter);
                  }
                }}
              >
                {qf.label}
              </Badge>
            ))}
          </div>

          {/* Active filters summary */}
          {(selectedSeverity !== "all" || minSeverity || selectedAction) && (
            <div className="flex items-center gap-2 mt-4 pt-4 border-t">
              <span className="text-sm text-muted-foreground">Active filters:</span>
              {selectedSeverity !== "all" && (
                <Badge
                  variant="secondary"
                  className="gap-1 cursor-pointer"
                  onClick={() => setSelectedSeverity("all")}
                >
                  Severity: {selectedSeverity}
                  <XCircle className="h-3 w-3" />
                </Badge>
              )}
              {minSeverity && (
                <Badge
                  variant="secondary"
                  className="gap-1 cursor-pointer"
                  onClick={() => setMinSeverity("")}
                >
                  Min: {minSeverity}+
                  <XCircle className="h-3 w-3" />
                </Badge>
              )}
              {selectedAction && (
                <Badge
                  variant="secondary"
                  className="gap-1 cursor-pointer"
                  onClick={() => setSelectedAction("")}
                >
                  Action: {selectedAction}
                  <XCircle className="h-3 w-3" />
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSelectedSeverity("all");
                  setMinSeverity("");
                  setSelectedAction("");
                }}
              >
                Clear all
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Logs Table */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Audit Logs</CardTitle>
              <CardDescription>
                {total} total events, {warningAndAbove} warnings/errors
              </CardDescription>
            </div>
            <Badge variant="outline">{filteredLogs.length} shown</Badge>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : filteredLogs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
              <Activity className="h-12 w-12 mb-4" />
              <p>No logs found matching your filters</p>
            </div>
          ) : (
            <ScrollArea className="h-[600px]">
              <Table>
                <TableHeader className="sticky top-0 bg-background">
                  <TableRow>
                    <TableHead className="w-[180px]">Timestamp</TableHead>
                    <TableHead className="w-[100px]">Severity</TableHead>
                    <TableHead className="w-[180px]">Action</TableHead>
                    <TableHead>User</TableHead>
                    <TableHead>Resource</TableHead>
                    <TableHead className="w-[120px]">IP Address</TableHead>
                    <TableHead className="w-[100px]">Details</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredLogs.map((log) => (
                    <LogRow key={log.id} log={log} />
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>

        {/* Pagination */}
        {total > pageSize && (
          <div className="flex items-center justify-between p-4 border-t">
            <p className="text-sm text-muted-foreground">
              Showing {(page - 1) * pageSize + 1} - {Math.min(page * pageSize, total)} of {total}
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
                disabled={!hasMore}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

// =============================================================================
// LOG ROW COMPONENT
// =============================================================================

function LogRow({ log }: { log: AuditLog }) {
  const severity = log.severity as LogSeverity;
  const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.info;
  const Icon = config.icon;

  return (
    <TableRow className="hover:bg-muted/50">
      <TableCell className="text-muted-foreground text-sm">
        <div className="flex items-center gap-2">
          <Clock className="h-3 w-3" />
          {new Date(log.created_at).toLocaleString()}
        </div>
      </TableCell>
      <TableCell>
        <Badge
          className={cn(
            "gap-1",
            config.bgColor,
            config.color,
            config.borderColor,
            "border"
          )}
        >
          <Icon className="h-3 w-3" />
          {config.label}
        </Badge>
      </TableCell>
      <TableCell>
        <span className="font-medium">{log.action.replace(/_/g, " ").replace(/\./g, " > ")}</span>
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          <User className="h-3 w-3 text-muted-foreground" />
          <span className="truncate max-w-[150px]">{log.user_email || log.user_id || "-"}</span>
        </div>
      </TableCell>
      <TableCell>
        {log.resource_type && (
          <span className="text-muted-foreground text-sm">
            {log.resource_type}
            {log.resource_id && (
              <>
                /<span className="text-foreground">{log.resource_id.slice(0, 8)}...</span>
              </>
            )}
          </span>
        )}
      </TableCell>
      <TableCell className="text-muted-foreground text-sm">
        {log.ip_address || "-"}
      </TableCell>
      <TableCell>
        {log.details && Object.keys(log.details).length > 0 && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                {/* Special display for service fallback/error events */}
                {log.action.startsWith("service.fallback.") && log.details.primary_provider ? (
                  <Badge variant="outline" className="cursor-help gap-1 bg-yellow-50">
                    {log.details.primary_provider} → {log.details.fallback_provider}
                  </Badge>
                ) : log.action.startsWith("service.error.") && log.details.provider ? (
                  <Badge variant="outline" className="cursor-help gap-1 bg-red-50">
                    {log.details.provider} failed
                  </Badge>
                ) : (
                  <Badge variant="outline" className="cursor-help">
                    {Object.keys(log.details).length} fields
                  </Badge>
                )}
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-md">
                {/* Enhanced display for service events */}
                {(log.action.startsWith("service.fallback.") || log.action.startsWith("service.error.")) ? (
                  <div className="text-xs space-y-2">
                    <div className="font-semibold">
                      {log.action.startsWith("service.fallback.") ? "Service Fallback" : "Service Error"}
                    </div>
                    {log.details.primary_provider && (
                      <div><span className="text-muted-foreground">Primary:</span> {log.details.primary_provider}</div>
                    )}
                    {log.details.fallback_provider && (
                      <div><span className="text-muted-foreground">Fallback:</span> {log.details.fallback_provider}</div>
                    )}
                    {log.details.provider && (
                      <div><span className="text-muted-foreground">Provider:</span> {log.details.provider}</div>
                    )}
                    {log.details.error_message && (
                      <div className="text-red-600 mt-1">{log.details.error_message}</div>
                    )}
                    {log.details.context && (
                      <div className="mt-2 pt-2 border-t">
                        <div className="text-muted-foreground mb-1">Context:</div>
                        <pre className="whitespace-pre-wrap">{JSON.stringify(log.details.context, null, 2)}</pre>
                      </div>
                    )}
                  </div>
                ) : (
                  <pre className="text-xs whitespace-pre-wrap">
                    {JSON.stringify(log.details, null, 2)}
                  </pre>
                )}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </TableCell>
    </TableRow>
  );
}

export default LogViewer;
