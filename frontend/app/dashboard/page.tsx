"use client";

import {
  FileText,
  MessageSquare,
  Upload,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertCircle,
  Activity,
  Zap,
  FolderOpen,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Loader2,
  Minus,
  Rocket,
  BookOpen,
  Search,
  Brain,
} from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useDocuments, useProcessingQueue, useCostDashboard, useChatSessions, useHealthCheck } from "@/lib/api";
import { useUser } from "@/lib/auth";

// Helper to display values with null/undefined handling
const formatNumber = (value: number | null | undefined): string => {
  if (value == null) return "—";
  return value.toLocaleString();
};

const formatCurrency = (value: number | null | undefined): string => {
  if (value == null) return "—";
  return `$${value.toFixed(2)}`;
};

export default function DashboardPage() {
  const { isAuthenticated, isLoading: authLoading } = useUser();

  // Only fetch data when authenticated
  const { data: documents, isLoading: docsLoading } = useDocuments({ page_size: 5 }, { enabled: isAuthenticated });
  const { data: queue, isLoading: queueLoading } = useProcessingQueue({ enabled: isAuthenticated });
  const { data: costs, isLoading: costsLoading } = useCostDashboard({ enabled: isAuthenticated });
  const { data: chatSessions, isLoading: chatsLoading } = useChatSessions({ page_size: 100 }, { enabled: isAuthenticated });
  const { data: health, isLoading: healthLoading } = useHealthCheck({ enabled: isAuthenticated });

  // Calculate real stats from API data
  const totalDocuments = documents?.total ?? null;
  const totalChunks = documents?.documents?.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0) ?? null;
  const activeChats = chatSessions?.total ?? null;
  const processingJobs = queue?.active_tasks ?? null;
  const completedItems = queue?.items?.filter(item => item.status === 'completed') ?? [];
  const failedItems = queue?.items?.filter(item => item.status === 'failed') ?? [];
  const costThisMonth = costs?.costs?.last_month ?? null;

  // Calculate cost change percentage from daily costs if available
  const costChange = (() => {
    if (!costs?.daily_costs || costs.daily_costs.length < 2) return null;
    const recent = costs.daily_costs.slice(-7);
    const older = costs.daily_costs.slice(-14, -7);
    if (recent.length === 0 || older.length === 0) return null;
    const recentTotal = recent.reduce((sum, d) => sum + d.cost, 0);
    const olderTotal = older.reduce((sum, d) => sum + d.cost, 0);
    if (olderTotal === 0) return null;
    return ((recentTotal - olderTotal) / olderTotal) * 100;
  })();

  // Use actual documents from API, show empty state if none
  const recentDocuments = documents?.documents ?? [];

  // Check if this is a new/empty workspace
  const isEmptyWorkspace = totalDocuments === 0 && (activeChats === 0 || activeChats === null);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Welcome back! Here's an overview of your document archive.
        </p>
      </div>

      {/* Getting Started Banner - Show only for empty workspaces */}
      {isEmptyWorkspace && !docsLoading && (
        <Card className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent border-primary/20">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
              <div className="p-4 rounded-full bg-primary/10">
                <Rocket className="h-8 w-8 text-primary" />
              </div>
              <div className="flex-1">
                <h2 className="text-xl font-semibold mb-2">Get Started with AI Document Indexer</h2>
                <p className="text-muted-foreground mb-4">
                  Upload your first documents to start building your knowledge base. Once indexed, you can chat with your documents using AI.
                </p>
                <div className="flex flex-wrap gap-3">
                  <Link href="/dashboard/upload">
                    <Button>
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Documents
                    </Button>
                  </Link>
                  <Link href="/dashboard/scraper">
                    <Button variant="outline">
                      <Search className="h-4 w-4 mr-2" />
                      Scrape Web Content
                    </Button>
                  </Link>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {docsLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : formatNumber(totalDocuments)}
            </div>
            <p className="text-xs text-muted-foreground">
              {totalChunks != null ? `${formatNumber(totalChunks)} chunks indexed` : "No chunks indexed yet"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {queueLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : formatNumber(processingJobs)}
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-green-600 flex items-center">
                <CheckCircle className="h-3 w-3 mr-1" />
                {completedItems.length} completed
              </span>
              {failedItems.length > 0 && (
                <span className="text-red-600 flex items-center">
                  <AlertCircle className="h-3 w-3 mr-1" />
                  {failedItems.length} failed
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Chat Sessions</CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {chatsLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : formatNumber(activeChats)}
            </div>
            <p className="text-xs text-muted-foreground">
              Total sessions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Costs</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {costsLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : formatCurrency(costThisMonth)}
            </div>
            <p className="text-xs text-muted-foreground flex items-center">
              {costChange != null ? (
                costChange > 0 ? (
                  <>
                    <ArrowUpRight className="h-3 w-3 text-red-500 mr-1" />
                    <span className="text-red-500">+{costChange.toFixed(1)}%</span>
                    <span className="ml-1">from last week</span>
                  </>
                ) : (
                  <>
                    <ArrowDownRight className="h-3 w-3 text-green-500 mr-1" />
                    <span className="text-green-500">{costChange.toFixed(1)}%</span>
                    <span className="ml-1">from last week</span>
                  </>
                )
              ) : (
                <span>This month</span>
              )}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <div className="grid gap-4 md:grid-cols-4">
        <Link href="/dashboard/chat">
          <Card className="hover:bg-muted/50 transition-colors cursor-pointer h-full">
            <CardContent className="pt-6 flex flex-col items-center text-center">
              <div className="p-3 rounded-full bg-primary/10 mb-3">
                <MessageSquare className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-medium">Start Chat</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Ask questions about your documents
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link href="/dashboard/upload">
          <Card className="hover:bg-muted/50 transition-colors cursor-pointer h-full">
            <CardContent className="pt-6 flex flex-col items-center text-center">
              <div className="p-3 rounded-full bg-green-500/10 mb-3">
                <Upload className="h-6 w-6 text-green-600" />
              </div>
              <h3 className="font-medium">Upload Files</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Add documents to your archive
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link href="/dashboard/create">
          <Card className="hover:bg-muted/50 transition-colors cursor-pointer h-full">
            <CardContent className="pt-6 flex flex-col items-center text-center">
              <div className="p-3 rounded-full bg-purple-500/10 mb-3">
                <FileText className="h-6 w-6 text-purple-600" />
              </div>
              <h3 className="font-medium">Create Document</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Generate new content with AI
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link href="/dashboard/documents">
          <Card className="hover:bg-muted/50 transition-colors cursor-pointer h-full">
            <CardContent className="pt-6 flex flex-col items-center text-center">
              <div className="p-3 rounded-full bg-orange-500/10 mb-3">
                <FolderOpen className="h-6 w-6 text-orange-600" />
              </div>
              <h3 className="font-medium">Browse Documents</h3>
              <p className="text-sm text-muted-foreground mt-1">
                View and manage your files
              </p>
            </CardContent>
          </Card>
        </Link>
      </div>

      {/* Recent Documents & Activity */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Recent Documents */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Recent Documents</CardTitle>
            <CardDescription>Latest files added to your archive</CardDescription>
          </CardHeader>
          <CardContent>
            {docsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : recentDocuments.length === 0 ? (
              <div className="text-center py-6 text-muted-foreground">
                <div className="p-4 rounded-full bg-muted/50 w-fit mx-auto mb-3">
                  <FileText className="h-8 w-8 opacity-50" />
                </div>
                <p className="text-sm font-medium">No documents yet</p>
                <p className="text-xs mt-1 mb-4">Upload your first document to get started</p>
                <Link href="/dashboard/upload">
                  <Button size="sm" variant="outline">
                    <Upload className="h-3 w-3 mr-2" />
                    Upload Now
                  </Button>
                </Link>
              </div>
            ) : (
              <div className="space-y-3">
                {recentDocuments.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50"
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="text-sm font-medium">{doc.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(doc.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <Link href={`/dashboard/documents/${doc.id}`}>
                      <Button variant="ghost" size="sm">
                        View
                      </Button>
                    </Link>
                  </div>
                ))}
              </div>
            )}
            <Link href="/dashboard/documents">
              <Button variant="outline" size="sm" className="w-full mt-4">
                View All Documents
              </Button>
            </Link>
          </CardContent>
        </Card>

        {/* Processing Queue Status */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Processing Queue</CardTitle>
            <CardDescription>Current document processing status</CardDescription>
          </CardHeader>
          <CardContent>
            {queueLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : !queue?.items || queue.items.length === 0 ? (
              <div className="text-center py-6 text-muted-foreground">
                <div className="p-4 rounded-full bg-muted/50 w-fit mx-auto mb-3">
                  <Activity className="h-8 w-8 opacity-50" />
                </div>
                <p className="text-sm font-medium">Queue is empty</p>
                <p className="text-xs mt-1">Documents will appear here when processing</p>
                <div className="flex items-center justify-center gap-4 mt-4 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="h-2 w-2 rounded-full bg-green-500" />
                    <span>Completed</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="h-2 w-2 rounded-full bg-blue-500" />
                    <span>Processing</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="h-2 w-2 rounded-full bg-yellow-500" />
                    <span>Pending</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {queue.items.slice(0, 5).map((item) => (
                  <div
                    key={item.file_id}
                    className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50"
                  >
                    <div
                      className={`p-2 rounded-full ${
                        item.status === "completed"
                          ? "bg-green-500/10"
                          : item.status === "failed"
                          ? "bg-red-500/10"
                          : item.status === "processing"
                          ? "bg-blue-500/10"
                          : "bg-gray-500/10"
                      }`}
                    >
                      {item.status === "completed" && (
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      )}
                      {item.status === "failed" && (
                        <AlertCircle className="h-4 w-4 text-red-600" />
                      )}
                      {item.status === "processing" && (
                        <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
                      )}
                      {item.status === "pending" && (
                        <Clock className="h-4 w-4 text-gray-600" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm truncate">{item.filename || item.file_id}</p>
                      <p className="text-xs text-muted-foreground capitalize">{item.status}</p>
                    </div>
                    {item.progress != null && item.status === "processing" && (
                      <span className="text-xs text-muted-foreground">{item.progress}%</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* How It Works - Show only for empty workspaces */}
      {isEmptyWorkspace && !docsLoading && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              How It Works
            </CardTitle>
            <CardDescription>
              Learn how to use AI Document Indexer in three simple steps
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-3">
              <div className="flex flex-col items-center text-center p-4 rounded-lg bg-muted/30">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10 text-primary font-bold mb-3">
                  1
                </div>
                <h3 className="font-medium mb-2">Upload Documents</h3>
                <p className="text-sm text-muted-foreground">
                  Upload PDFs, Word docs, text files, or scrape web content to build your knowledge base.
                </p>
              </div>
              <div className="flex flex-col items-center text-center p-4 rounded-lg bg-muted/30">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10 text-primary font-bold mb-3">
                  2
                </div>
                <h3 className="font-medium mb-2">AI Processing</h3>
                <p className="text-sm text-muted-foreground">
                  Documents are automatically chunked and indexed using advanced AI embeddings.
                </p>
              </div>
              <div className="flex flex-col items-center text-center p-4 rounded-lg bg-muted/30">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10 text-primary font-bold mb-3">
                  3
                </div>
                <h3 className="font-medium mb-2">Chat & Discover</h3>
                <p className="text-sm text-muted-foreground">
                  Ask questions in natural language and get AI-powered answers from your documents.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">System Status</CardTitle>
          <CardDescription>
            {healthLoading ? "Checking..." : health?.status === "healthy" ? "All systems operational" : "System status unknown"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {healthLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="flex items-center gap-3">
                <div className={`h-2 w-2 rounded-full ${health?.status === "healthy" ? "bg-green-500" : "bg-yellow-500"}`} />
                <div>
                  <span className="text-sm">API Server</span>
                  {health?.version && (
                    <p className="text-xs text-muted-foreground">v{health.version}</p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className={`h-2 w-2 rounded-full ${health?.status === "healthy" ? "bg-green-500" : "bg-gray-400"}`} />
                <span className="text-sm">Backend Services</span>
              </div>
              <div className="flex items-center gap-3">
                <div className={`h-2 w-2 rounded-full ${queue ? "bg-green-500" : "bg-gray-400"}`} />
                <span className="text-sm">Processing Queue</span>
              </div>
              <div className="flex items-center gap-3">
                <div className={`h-2 w-2 rounded-full ${documents ? "bg-green-500" : "bg-gray-400"}`} />
                <span className="text-sm">Database</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
