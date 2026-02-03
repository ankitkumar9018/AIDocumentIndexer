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
  Sparkles,
  Database,
  Network,
  Shield,
  PieChart,
  Users,
  Globe,
  ChevronRight,
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
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useDocuments, useProcessingQueue, useCostDashboard, useChatSessions, useHealthCheck } from "@/lib/api";
import { useUser } from "@/lib/auth";
import { StatsCardSkeleton, ListCardSkeleton, QueueCardSkeleton } from "@/components/skeletons";
import { cn } from "@/lib/utils";

// Helper to display values with null/undefined handling
const formatNumber = (value: number | null | undefined): string => {
  if (value == null) return "—";
  return value.toLocaleString();
};

const formatCurrency = (value: number | null | undefined): string => {
  if (value == null) return "—";
  return `$${value.toFixed(2)}`;
};

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

export default function DashboardPage() {
  const { isAuthenticated, isLoading: authLoading, user } = useUser();

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
  const totalSize = documents?.documents?.reduce((sum, doc) => sum + (doc.file_size || 0), 0) ?? 0;

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

  // Greeting based on time of day
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 18) return "Good afternoon";
    return "Good evening";
  };

  return (
    <div className="space-y-8 page-transition">
      {/* Header with Greeting */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            {getGreeting()}{user?.name ? `, ${user.name.split(' ')[0]}` : ''}
          </h1>
          <p className="text-muted-foreground mt-1">
            Here's what's happening with your knowledge base today.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/dashboard/upload">
            <Button className="btn-gradient">
              <Upload className="h-4 w-4 mr-2" />
              Upload Documents
            </Button>
          </Link>
        </div>
      </div>

      {/* Getting Started Banner - Show only for empty workspaces */}
      {isEmptyWorkspace && !docsLoading && (
        <Card className="glass border-primary/20 overflow-hidden bounce-in">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary via-primary/50 to-transparent" />
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
              <div className="p-4 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 float">
                <Rocket className="h-10 w-10 text-primary" />
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold mb-2">Welcome to AI Document Indexer</h2>
                <p className="text-muted-foreground mb-4 max-w-2xl">
                  Transform your documents into an intelligent, searchable knowledge base. Upload your first documents to start building your AI-powered archive.
                </p>
                <div className="flex flex-wrap gap-3">
                  <Link href="/dashboard/upload">
                    <Button className="btn-gradient">
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Documents
                    </Button>
                  </Link>
                  <Link href="/dashboard/scraper">
                    <Button variant="outline" className="btn-glass">
                      <Globe className="h-4 w-4 mr-2" />
                      Scrape Web Content
                    </Button>
                  </Link>
                  <Link href="/dashboard/connectors">
                    <Button variant="outline" className="btn-glass">
                      <Network className="h-4 w-4 mr-2" />
                      Connect Sources
                    </Button>
                  </Link>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stats Grid - Modern Design */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 stagger-children">
        {/* Total Documents */}
        {docsLoading ? (
          <StatsCardSkeleton />
        ) : (
          <Card className="stats-card group card-hover">
            <CardContent className="pt-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="stats-label">Total Documents</p>
                  <p className="stats-value mt-1">{formatNumber(totalDocuments)}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {totalChunks != null ? `${formatNumber(totalChunks)} chunks` : "No chunks yet"}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-colors">
                  <FileText className="h-5 w-5 text-primary" />
                </div>
              </div>
              <div className="mt-4 flex items-center gap-2 text-xs">
                <Badge variant="secondary" className="rounded-full">
                  {formatBytes(totalSize)}
                </Badge>
                <span className="text-muted-foreground">storage used</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Processing Status */}
        {queueLoading ? (
          <StatsCardSkeleton />
        ) : (
          <Card className="stats-card group card-hover">
            <CardContent className="pt-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="stats-label">Processing</p>
                  <p className="stats-value mt-1">{formatNumber(processingJobs)}</p>
                  <p className="text-xs text-muted-foreground mt-1">active tasks</p>
                </div>
                <div className={cn(
                  "p-3 rounded-xl transition-colors",
                  processingJobs && processingJobs > 0
                    ? "bg-blue-500/10 group-hover:bg-blue-500/20"
                    : "bg-green-500/10 group-hover:bg-green-500/20"
                )}>
                  <Activity className={cn(
                    "h-5 w-5",
                    processingJobs && processingJobs > 0 ? "text-blue-500 animate-pulse" : "text-green-500"
                  )} />
                </div>
              </div>
              <div className="mt-4 flex items-center gap-3 text-xs">
                <div className="flex items-center gap-1 text-green-600 dark:text-green-400">
                  <CheckCircle className="h-3 w-3" />
                  <span>{completedItems.length}</span>
                </div>
                {failedItems.length > 0 && (
                  <div className="flex items-center gap-1 text-red-600 dark:text-red-400">
                    <AlertCircle className="h-3 w-3" />
                    <span>{failedItems.length}</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Chat Sessions */}
        {chatsLoading ? (
          <StatsCardSkeleton />
        ) : (
          <Card className="stats-card group card-hover">
            <CardContent className="pt-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="stats-label">Chat Sessions</p>
                  <p className="stats-value mt-1">{formatNumber(activeChats)}</p>
                  <p className="text-xs text-muted-foreground mt-1">conversations</p>
                </div>
                <div className="p-3 rounded-xl bg-purple-500/10 group-hover:bg-purple-500/20 transition-colors">
                  <MessageSquare className="h-5 w-5 text-purple-500" />
                </div>
              </div>
              <div className="mt-4">
                <Link href="/dashboard/chat">
                  <Button variant="ghost" size="sm" className="h-7 px-2 text-xs hover:bg-purple-500/10">
                    Start new chat
                    <ChevronRight className="h-3 w-3 ml-1" />
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        )}

        {/* API Costs */}
        {costsLoading ? (
          <StatsCardSkeleton />
        ) : (
          <Card className="stats-card group card-hover">
            <CardContent className="pt-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="stats-label">API Costs</p>
                  <p className="stats-value mt-1">{formatCurrency(costThisMonth)}</p>
                  <p className="text-xs text-muted-foreground mt-1">this month</p>
                </div>
                <div className="p-3 rounded-xl bg-amber-500/10 group-hover:bg-amber-500/20 transition-colors">
                  <Zap className="h-5 w-5 text-amber-500" />
                </div>
              </div>
              <div className="mt-4 flex items-center gap-1 text-xs">
                {costChange != null ? (
                  costChange > 0 ? (
                    <>
                      <ArrowUpRight className="h-3 w-3 text-red-500" />
                      <span className="stats-trend negative">+{costChange.toFixed(1)}%</span>
                      <span className="text-muted-foreground">vs last week</span>
                    </>
                  ) : (
                    <>
                      <ArrowDownRight className="h-3 w-3 text-green-500" />
                      <span className="stats-trend positive">{costChange.toFixed(1)}%</span>
                      <span className="text-muted-foreground">vs last week</span>
                    </>
                  )
                ) : (
                  <span className="text-muted-foreground">No comparison data</span>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Quick Actions - Feature Cards */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 stagger-children">
          <Link href="/dashboard/chat" className="block">
            <Card className="feature-card h-full group">
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-4 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 mb-4 group-hover:scale-110 transition-transform">
                  <MessageSquare className="h-7 w-7 text-primary" />
                </div>
                <h3 className="font-semibold">Start Chat</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Ask questions about your documents using AI
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/dashboard/upload" className="block">
            <Card className="feature-card h-full group">
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-4 rounded-2xl bg-gradient-to-br from-green-500/20 to-green-500/5 mb-4 group-hover:scale-110 transition-transform">
                  <Upload className="h-7 w-7 text-green-500" />
                </div>
                <h3 className="font-semibold">Upload Files</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Add documents to your knowledge base
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/dashboard/create" className="block">
            <Card className="feature-card h-full group">
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-4 rounded-2xl bg-gradient-to-br from-purple-500/20 to-purple-500/5 mb-4 group-hover:scale-110 transition-transform">
                  <Sparkles className="h-7 w-7 text-purple-500" />
                </div>
                <h3 className="font-semibold">Create Content</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  Generate new documents with AI assistance
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/dashboard/documents" className="block">
            <Card className="feature-card h-full group">
              <CardContent className="pt-6 flex flex-col items-center text-center">
                <div className="p-4 rounded-2xl bg-gradient-to-br from-orange-500/20 to-orange-500/5 mb-4 group-hover:scale-110 transition-transform">
                  <FolderOpen className="h-7 w-7 text-orange-500" />
                </div>
                <h3 className="font-semibold">Browse Documents</h3>
                <p className="text-sm text-muted-foreground mt-2">
                  View and manage your indexed files
                </p>
              </CardContent>
            </Card>
          </Link>
        </div>
      </div>

      {/* Analytics Overview */}
      <Card className="glass">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                Analytics Overview
              </CardTitle>
              <CardDescription>Your activity and usage patterns</CardDescription>
            </div>
            <Link href="/dashboard/costs">
              <Button variant="ghost" size="sm">
                View Details
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4 stagger-children">
            {/* Documents by Type */}
            <div className="p-4 rounded-xl border bg-card/50">
              <div className="flex items-center gap-2 mb-4">
                <PieChart className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-medium">By Type</h4>
              </div>
              <div className="space-y-3">
                {(() => {
                  const typeCounts = (documents?.documents ?? []).reduce((acc, doc) => {
                    const type = doc.file_type?.toLowerCase() || 'other';
                    acc[type] = (acc[type] || 0) + 1;
                    return acc;
                  }, {} as Record<string, number>);
                  const types = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]).slice(0, 4);
                  const total = types.reduce((sum, [, count]) => sum + count, 0);
                  if (types.length === 0) {
                    return <p className="text-xs text-muted-foreground">No documents yet</p>;
                  }
                  return types.map(([type, count]) => (
                    <div key={type} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className="capitalize">{type}</span>
                        <span className="font-medium">{count}</span>
                      </div>
                      <Progress value={(count / total) * 100} className="h-1" />
                    </div>
                  ));
                })()}
              </div>
            </div>

            {/* Storage Usage */}
            <div className="p-4 rounded-xl border bg-card/50">
              <div className="flex items-center gap-2 mb-4">
                <Database className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-medium">Storage</h4>
              </div>
              <div className="text-3xl font-bold">{formatBytes(totalSize)}</div>
              <p className="text-xs text-muted-foreground mt-1">Total document size</p>
              <div className="mt-4">
                <Progress value={Math.min((totalSize / (100 * 1024 * 1024)) * 100, 100)} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">of 100MB free tier</p>
              </div>
            </div>

            {/* Chat Activity */}
            <div className="p-4 rounded-xl border bg-card/50">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-medium">AI Activity</h4>
              </div>
              <div className="text-3xl font-bold">{formatNumber(activeChats)}</div>
              <p className="text-xs text-muted-foreground mt-1">Chat sessions</p>
              <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
                <Sparkles className="h-3 w-3" />
                <span>AI-powered conversations</span>
              </div>
            </div>

            {/* Cost Summary */}
            <div className="p-4 rounded-xl border bg-card/50">
              <div className="flex items-center gap-2 mb-4">
                <Zap className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-medium">Costs</h4>
              </div>
              <div className="text-3xl font-bold">{formatCurrency(costThisMonth)}</div>
              <p className="text-xs text-muted-foreground mt-1">This month</p>
              <div className="mt-4">
                <Link href="/dashboard/costs">
                  <Button variant="link" size="sm" className="h-auto p-0 text-xs">
                    View breakdown →
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity & Processing */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Recent Documents */}
        {docsLoading ? (
          <ListCardSkeleton count={5} />
        ) : (
          <Card className="glass">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg">Recent Documents</CardTitle>
                  <CardDescription>Latest files in your archive</CardDescription>
                </div>
                <Link href="/dashboard/documents">
                  <Button variant="ghost" size="sm">View All</Button>
                </Link>
              </div>
            </CardHeader>
            <CardContent>
              {recentDocuments.length === 0 ? (
                <div className="text-center py-8">
                  <div className="p-4 rounded-2xl bg-muted/50 w-fit mx-auto mb-4">
                    <FileText className="h-10 w-10 text-muted-foreground/50" />
                  </div>
                  <p className="font-medium">No documents yet</p>
                  <p className="text-sm text-muted-foreground mt-1 mb-4">
                    Upload your first document to get started
                  </p>
                  <Link href="/dashboard/upload">
                    <Button size="sm">
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Now
                    </Button>
                  </Link>
                </div>
              ) : (
                <div className="space-y-2 stagger-children">
                  {recentDocuments.map((doc) => (
                    <Link
                      key={doc.id}
                      href={`/dashboard/documents/${doc.id}`}
                      className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors group"
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-primary/10">
                          <FileText className="h-4 w-4 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm font-medium group-hover:text-primary transition-colors">
                            {doc.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(doc.created_at).toLocaleDateString()} · {doc.chunk_count || 0} chunks
                          </p>
                        </div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                    </Link>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Processing Queue */}
        {queueLoading ? (
          <QueueCardSkeleton count={5} />
        ) : (
          <Card className="glass">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg">Processing Queue</CardTitle>
                  <CardDescription>Document processing status</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  {processingJobs && processingJobs > 0 && (
                    <Badge variant="secondary" className="animate-pulse">
                      {processingJobs} active
                    </Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {!queue?.items || queue.items.length === 0 ? (
                <div className="text-center py-8">
                  <div className="p-4 rounded-2xl bg-muted/50 w-fit mx-auto mb-4">
                    <Activity className="h-10 w-10 text-muted-foreground/50" />
                  </div>
                  <p className="font-medium">Queue is empty</p>
                  <p className="text-sm text-muted-foreground mt-1 mb-4">
                    Documents will appear here when processing
                  </p>
                  <div className="flex items-center justify-center gap-4 text-xs">
                    <div className="flex items-center gap-1">
                      <div className="h-2 w-2 rounded-full bg-green-500" />
                      <span>Completed</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                      <span>Processing</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="h-2 w-2 rounded-full bg-amber-500" />
                      <span>Pending</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-2 stagger-children">
                  {queue.items.slice(0, 5).map((item) => (
                    <div
                      key={item.file_id}
                      className="flex items-center gap-3 p-3 rounded-lg bg-muted/30"
                    >
                      <div
                        className={cn(
                          "p-2 rounded-lg",
                          item.status === "completed" && "bg-green-500/10",
                          item.status === "failed" && "bg-red-500/10",
                          item.status === "processing" && "bg-blue-500/10",
                          item.status === "pending" && "bg-amber-500/10"
                        )}
                      >
                        {item.status === "completed" && (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        )}
                        {item.status === "failed" && (
                          <AlertCircle className="h-4 w-4 text-red-500" />
                        )}
                        {item.status === "processing" && (
                          <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
                        )}
                        {item.status === "pending" && (
                          <Clock className="h-4 w-4 text-amber-500" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">
                          {item.filename || item.file_id}
                        </p>
                        <p className="text-xs text-muted-foreground capitalize">{item.status}</p>
                      </div>
                      {item.progress != null && item.status === "processing" && (
                        <div className="w-20">
                          <Progress value={item.progress} className="h-1.5" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      {/* How It Works - Show only for empty workspaces */}
      {isEmptyWorkspace && !docsLoading && (
        <Card className="glass slide-in-up">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              How It Works
            </CardTitle>
            <CardDescription>
              Get started in three simple steps
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-3 stagger-children">
              <div className="relative p-6 rounded-xl bg-gradient-to-br from-primary/10 to-transparent border">
                <div className="flex items-center justify-center w-12 h-12 rounded-full bg-primary text-primary-foreground font-bold mb-4">
                  1
                </div>
                <h3 className="font-semibold mb-2">Upload Documents</h3>
                <p className="text-sm text-muted-foreground">
                  Upload PDFs, Word docs, text files, or scrape web content to build your knowledge base.
                </p>
              </div>
              <div className="relative p-6 rounded-xl bg-gradient-to-br from-purple-500/10 to-transparent border">
                <div className="flex items-center justify-center w-12 h-12 rounded-full bg-purple-500 text-white font-bold mb-4">
                  2
                </div>
                <h3 className="font-semibold mb-2">AI Processing</h3>
                <p className="text-sm text-muted-foreground">
                  Documents are automatically chunked and indexed using advanced AI embeddings.
                </p>
              </div>
              <div className="relative p-6 rounded-xl bg-gradient-to-br from-green-500/10 to-transparent border">
                <div className="flex items-center justify-center w-12 h-12 rounded-full bg-green-500 text-white font-bold mb-4">
                  3
                </div>
                <h3 className="font-semibold mb-2">Chat & Discover</h3>
                <p className="text-sm text-muted-foreground">
                  Ask questions in natural language and get AI-powered answers from your documents.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Status - Compact */}
      <Card className="glass">
        <CardContent className="py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">System Status</span>
            </div>
            {healthLoading ? (
              <div className="flex items-center gap-2">
                <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Checking...</span>
              </div>
            ) : (
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "h-2 w-2 rounded-full",
                    health?.status === "healthy" ? "bg-green-500" : "bg-amber-500"
                  )} />
                  <span>API {health?.version && `v${health.version}`}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "h-2 w-2 rounded-full",
                    health?.status === "healthy" ? "bg-green-500" : "bg-gray-400"
                  )} />
                  <span>Services</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "h-2 w-2 rounded-full",
                    queue ? "bg-green-500" : "bg-gray-400"
                  )} />
                  <span>Queue</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={cn(
                    "h-2 w-2 rounded-full",
                    documents ? "bg-green-500" : "bg-gray-400"
                  )} />
                  <span>Database</span>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
