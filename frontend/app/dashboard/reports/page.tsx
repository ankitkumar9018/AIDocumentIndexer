"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import {
  Plus,
  FileText,
  Search,
  Clock,
  Star,
  MoreVertical,
  Trash2,
  Copy,
  Download,
  Share2,
  Eye,
  Filter,
  SortAsc,
  Loader2,
  AlertCircle,
  RefreshCw,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { useSession } from "next-auth/react";

interface Report {
  id: string;
  title: string;
  description: string;
  createdAt: Date;
  updatedAt: Date;
  status: "draft" | "published" | "archived";
  sections: number;
  citations: number;
  starred: boolean;
  thumbnail?: string;
}

const statusColors: Record<Report["status"], string> = {
  draft: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
  published: "bg-green-500/10 text-green-600 border-green-500/20",
  archived: "bg-gray-500/10 text-gray-600 border-gray-500/20",
};

export default function ReportsPage() {
  const { data: session } = useSession();
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("all");

  const fetchReports = useCallback(async () => {
    if (!session?.user) return;

    try {
      setLoading(true);
      setError(null);
      const response = await api.get<{ reports: any[] }>("/api/v1/reports/list");

      if (response.data?.reports) {
        // Transform API response to frontend Report type
        const transformedReports: Report[] = response.data.reports.map((r: any) => ({
          id: r.id,
          title: r.title,
          description: r.description || "",
          createdAt: new Date(r.created_at),
          updatedAt: new Date(r.updated_at),
          status: r.status as Report["status"],
          sections: r.section_count,
          citations: r.citation_count,
          starred: r.is_starred,
          thumbnail: r.thumbnail_url,
        }));
        setReports(transformedReports);
      }
    } catch (err: any) {
      console.error("Failed to fetch reports:", err);
      setReports([]);
      setError(err?.response?.data?.detail || "Failed to load reports. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [session?.user]);

  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  const filteredReports = reports.filter((report) => {
    const matchesSearch =
      report.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      report.description.toLowerCase().includes(searchQuery.toLowerCase());

    if (activeTab === "starred") return matchesSearch && report.starred;
    if (activeTab === "drafts") return matchesSearch && report.status === "draft";
    if (activeTab === "published") return matchesSearch && report.status === "published";
    return matchesSearch;
  });

  const toggleStar = async (id: string) => {
    try {
      const response = await api.patch<{ is_starred: boolean }>(`/api/v1/reports/${id}/star`);
      if (response.data) {
        setReports((prev) =>
          prev.map((r) => (r.id === id ? { ...r, starred: response.data.is_starred } : r))
        );
      }
    } catch (err) {
      console.error("Failed to toggle star:", err);
      // Optimistic update fallback
      setReports((prev) =>
        prev.map((r) => (r.id === id ? { ...r, starred: !r.starred } : r))
      );
    }
  };

  const deleteReport = async (id: string) => {
    try {
      await api.delete(`/api/v1/reports/${id}`);
      setReports((prev) => prev.filter((r) => r.id !== id));
    } catch (err) {
      console.error("Failed to delete report:", err);
      // Still remove from UI for better UX, but log error
      setReports((prev) => prev.filter((r) => r.id !== id));
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Sparkpages</h1>
          <p className="text-muted-foreground mt-1">
            Create dynamic, AI-generated reports with rich citations
          </p>
        </div>
        <Link href="/dashboard/reports/new">
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            New Report
          </Button>
        </Link>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription className="flex items-center justify-between">
            <span>{error}</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setError(null);
                fetchReports();
              }}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Search and Filters */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search reports..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>
          <Button variant="outline" size="sm">
            <SortAsc className="h-4 w-4 mr-2" />
            Sort
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="all" value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="all">
            All
            <Badge variant="secondary" className="ml-2">
              {reports.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="starred">
            <Star className="h-4 w-4 mr-1" />
            Starred
          </TabsTrigger>
          <TabsTrigger value="drafts">Drafts</TabsTrigger>
          <TabsTrigger value="published">Published</TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-6">
          {loading ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground">Loading reports...</p>
              </CardContent>
            </Card>
          ) : filteredReports.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <FileText className="h-12 w-12 text-muted-foreground/50 mb-4" />
                <h3 className="text-lg font-medium">No reports found</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {searchQuery
                    ? "Try adjusting your search query"
                    : "Create your first report to get started"}
                </p>
                {!searchQuery && (
                  <Link href="/dashboard/reports/new" className="mt-4">
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Report
                    </Button>
                  </Link>
                )}
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredReports.map((report) => (
                <ReportCard
                  key={report.id}
                  report={report}
                  onToggleStar={() => toggleStar(report.id)}
                  onDelete={() => deleteReport(report.id)}
                />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

interface ReportCardProps {
  report: Report;
  onToggleStar: () => void;
  onDelete: () => void;
}

function ReportCard({ report, onToggleStar, onDelete }: ReportCardProps) {
  return (
    <Card className="group hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <Link
              href={`/dashboard/reports/${report.id}`}
              className="hover:underline"
            >
              <CardTitle className="text-lg truncate">{report.title}</CardTitle>
            </Link>
            <CardDescription className="mt-1 line-clamp-2">
              {report.description}
            </CardDescription>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem asChild>
                <Link href={`/dashboard/reports/${report.id}`}>
                  <Eye className="h-4 w-4 mr-2" />
                  View
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onToggleStar}>
                <Star
                  className={`h-4 w-4 mr-2 ${
                    report.starred ? "fill-yellow-500 text-yellow-500" : ""
                  }`}
                />
                {report.starred ? "Unstar" : "Star"}
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Copy className="h-4 w-4 mr-2" />
                Duplicate
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Download className="h-4 w-4 mr-2" />
                Export
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Share2 className="h-4 w-4 mr-2" />
                Share
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive"
                onClick={onDelete}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent>
        {/* Metadata */}
        <div className="flex items-center gap-4 text-sm text-muted-foreground mb-3">
          <div className="flex items-center gap-1">
            <FileText className="h-4 w-4" />
            {report.sections} sections
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs">@</span>
            {report.citations} citations
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-3 border-t">
          <Badge variant="outline" className={statusColors[report.status]}>
            {report.status}
          </Badge>
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="h-3.5 w-3.5" />
            {formatDistanceToNow(report.updatedAt, { addSuffix: true })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
