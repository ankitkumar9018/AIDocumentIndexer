"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import {
  Bot,
  FileEdit,
  MessageSquare,
  Search,
  Wrench,
  Activity,
  Clock,
  DollarSign,
  CheckCircle2,
  XCircle,
  AlertCircle,
  RefreshCw,
  Settings2,
  History,
  Zap,
  TrendingUp,
  TrendingDown,
  MoreHorizontal,
  Play,
  Pause,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  useAgentStatus,
  useOptimizationJobs,
  useApproveOptimization,
  useRejectOptimization,
  useTriggerPromptOptimization,
  useAgentTrajectories,
  useUpdateAgentConfig,
} from "@/lib/api/hooks";
import type { AgentDefinition, PromptOptimizationJob, AgentTrajectory } from "@/lib/api/client";

const agentIcons: Record<string, React.ReactNode> = {
  manager: <Bot className="h-5 w-5" />,
  generator: <FileEdit className="h-5 w-5" />,
  critic: <MessageSquare className="h-5 w-5" />,
  research: <Search className="h-5 w-5" />,
  tool_executor: <Wrench className="h-5 w-5" />,
};

const agentColors: Record<string, string> = {
  manager: "bg-violet-500",
  generator: "bg-blue-500",
  critic: "bg-amber-500",
  research: "bg-emerald-500",
  tool_executor: "bg-pink-500",
};

function formatDuration(ms: number | undefined | null): string {
  if (ms == null) return "0ms";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function formatNumber(num: number | undefined | null): string {
  if (num == null) return "0";
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

export default function AgentsAdminPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [selectedAgent, setSelectedAgent] = useState<AgentDefinition | null>(null);
  const [showOptimizationDialog, setShowOptimizationDialog] = useState(false);
  const [selectedJob, setSelectedJob] = useState<PromptOptimizationJob | null>(null);

  // Queries - only fetch when authenticated
  const { data: statusData, isLoading: statusLoading, refetch: refetchStatus } = useAgentStatus({ enabled: isAuthenticated });
  const { data: jobsData, isLoading: jobsLoading } = useOptimizationJobs(undefined, { enabled: isAuthenticated });
  const { data: trajectoriesData } = useAgentTrajectories({ limit: 50 }, { enabled: isAuthenticated });

  // Mutations
  const approveOptimization = useApproveOptimization();
  const rejectOptimization = useRejectOptimization();
  const triggerOptimization = useTriggerPromptOptimization();
  const updateAgentConfig = useUpdateAgentConfig();

  const agents = statusData?.agents || [];
  const jobs = jobsData?.jobs || [];
  const trajectories = trajectoriesData?.trajectories || [];

  const pendingJobs = jobs.filter((j) => j.status === "awaiting_approval");
  const activeJobs = jobs.filter((j) => !["completed", "rejected", "awaiting_approval"].includes(j.status));

  const handleToggleAgent = async (agent: AgentDefinition) => {
    try {
      await updateAgentConfig.mutateAsync({
        agentId: agent.id,
        data: { is_active: !agent.is_active },
      });
    } catch (error) {
      console.error("Failed to toggle agent:", error);
    }
  };

  const handleTriggerOptimization = async (agentId: string) => {
    try {
      await triggerOptimization.mutateAsync(agentId);
    } catch (error) {
      console.error("Failed to trigger optimization:", error);
    }
  };

  const handleApproveJob = async (jobId: string) => {
    try {
      await approveOptimization.mutateAsync(jobId);
      setShowOptimizationDialog(false);
      setSelectedJob(null);
    } catch (error) {
      console.error("Failed to approve optimization:", error);
    }
  };

  const handleRejectJob = async (jobId: string) => {
    try {
      await rejectOptimization.mutateAsync({ jobId, reason: "Manual rejection" });
      setShowOptimizationDialog(false);
      setSelectedJob(null);
    } catch (error) {
      console.error("Failed to reject optimization:", error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Agent Management</h1>
          <p className="text-muted-foreground">
            Monitor and configure multi-agent system performance
          </p>
        </div>
        <Button onClick={() => refetchStatus()} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
            <Bot className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statusLoading ? <Skeleton className="h-8 w-16" /> : statusData?.total || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {statusData?.healthy || 0} healthy, {statusData?.degraded || 0} degraded
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Approvals</CardTitle>
            <AlertCircle className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {jobsLoading ? <Skeleton className="h-8 w-16" /> : pendingJobs.length}
            </div>
            <p className="text-xs text-muted-foreground">
              Prompt optimizations awaiting review
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Optimizations</CardTitle>
            <Zap className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {jobsLoading ? <Skeleton className="h-8 w-16" /> : activeJobs.length}
            </div>
            <p className="text-xs text-muted-foreground">
              Currently running A/B tests
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Executions</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {statusLoading ? (
                <Skeleton className="h-8 w-16" />
              ) : (
                formatNumber(agents.reduce((sum, a) => sum + a.total_executions, 0))
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Across all agents (24h)
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="agents" className="space-y-4">
        <TabsList>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="optimizations">
            Optimizations
            {pendingJobs.length > 0 && (
              <Badge variant="destructive" className="ml-2 h-5 w-5 rounded-full p-0 text-xs">
                {pendingJobs.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="trajectories">Trajectories</TabsTrigger>
        </TabsList>

        {/* Agents Tab */}
        <TabsContent value="agents" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Agent Overview</CardTitle>
              <CardDescription>
                View and manage individual agent configurations and performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              {statusLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-20 w-full" />
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  {agents.map((agent) => (
                    <Card key={agent.id} className="p-4">
                      <div className="flex items-start gap-4">
                        {/* Agent Icon */}
                        <div
                          className={cn(
                            "p-3 rounded-lg text-white",
                            agentColors[agent.agent_type] || "bg-gray-500"
                          )}
                        >
                          {agentIcons[agent.agent_type] || <Bot className="h-5 w-5" />}
                        </div>

                        {/* Agent Info */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <h4 className="font-semibold capitalize">{agent.name}</h4>
                            <Badge variant={agent.is_active ? "default" : "secondary"}>
                              {agent.is_active ? "Active" : "Inactive"}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground line-clamp-1">
                            {agent.description}
                          </p>

                          {/* Stats */}
                          <div className="flex items-center gap-6 mt-3 text-sm">
                            <div className="flex items-center gap-1">
                              <Activity className="h-4 w-4 text-muted-foreground" />
                              <span>{formatNumber(agent.total_executions)} executions</span>
                            </div>
                            <div className="flex items-center gap-1">
                              {(agent.success_rate ?? 0) >= 0.8 ? (
                                <TrendingUp className="h-4 w-4 text-green-500" />
                              ) : (agent.success_rate ?? 0) >= 0.5 ? (
                                <Activity className="h-4 w-4 text-amber-500" />
                              ) : (
                                <TrendingDown className="h-4 w-4 text-red-500" />
                              )}
                              <span>{((agent.success_rate ?? 0) * 100).toFixed(1)}% success</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="h-4 w-4 text-muted-foreground" />
                              <span>{formatDuration(agent.avg_latency_ms)} avg</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <DollarSign className="h-4 w-4 text-muted-foreground" />
                              <span>{agent.avg_tokens_per_execution ?? 0} tokens/exec</span>
                            </div>
                          </div>

                          {/* Success Rate Progress */}
                          <div className="mt-3">
                            <div className="flex items-center justify-between text-xs mb-1">
                              <span className="text-muted-foreground">Success Rate</span>
                              <span
                                className={cn(
                                  "font-medium",
                                  (agent.success_rate ?? 0) >= 0.8
                                    ? "text-green-500"
                                    : (agent.success_rate ?? 0) >= 0.5
                                    ? "text-amber-500"
                                    : "text-red-500"
                                )}
                              >
                                {((agent.success_rate ?? 0) * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress
                              value={(agent.success_rate ?? 0) * 100}
                              className={cn(
                                "h-2",
                                (agent.success_rate ?? 0) >= 0.8
                                  ? "[&>div]:bg-green-500"
                                  : (agent.success_rate ?? 0) >= 0.5
                                  ? "[&>div]:bg-amber-500"
                                  : "[&>div]:bg-red-500"
                              )}
                            />
                          </div>
                        </div>

                        {/* Actions */}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuLabel>Actions</DropdownMenuLabel>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => handleToggleAgent(agent)}>
                              {agent.is_active ? (
                                <>
                                  <Pause className="h-4 w-4 mr-2" />
                                  Disable Agent
                                </>
                              ) : (
                                <>
                                  <Play className="h-4 w-4 mr-2" />
                                  Enable Agent
                                </>
                              )}
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleTriggerOptimization(agent.id)}>
                              <Zap className="h-4 w-4 mr-2" />
                              Trigger Optimization
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <History className="h-4 w-4 mr-2" />
                              View Prompt History
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <Settings2 className="h-4 w-4 mr-2" />
                              Configure
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Optimizations Tab */}
        <TabsContent value="optimizations" className="space-y-4">
          {/* Pending Approvals */}
          {pendingJobs.length > 0 && (
            <Card className="border-amber-500/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-amber-500" />
                  Pending Approvals
                </CardTitle>
                <CardDescription>
                  Review and approve prompt optimizations before they go live
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {pendingJobs.map((job) => (
                    <div
                      key={job.id}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div>
                        <p className="font-medium">{job.agent_name || "Unknown Agent"}</p>
                        <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                          <span>{job.trajectories_analyzed ?? 0} trajectories analyzed</span>
                          <span>{job.variants_generated ?? 0} variants generated</span>
                          <span className="flex items-center gap-1">
                            {(job.improvement_percentage ?? 0) > 0 ? (
                              <TrendingUp className="h-4 w-4 text-green-500" />
                            ) : (
                              <TrendingDown className="h-4 w-4 text-red-500" />
                            )}
                            {(job.improvement_percentage ?? 0).toFixed(1)}% improvement
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleRejectJob(job.id)}
                          disabled={rejectOptimization.isPending}
                        >
                          <XCircle className="h-4 w-4 mr-1" />
                          Reject
                        </Button>
                        <Button
                          size="sm"
                          onClick={() => handleApproveJob(job.id)}
                          disabled={approveOptimization.isPending}
                        >
                          <CheckCircle2 className="h-4 w-4 mr-1" />
                          Approve
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* All Jobs */}
          <Card>
            <CardHeader>
              <CardTitle>Optimization History</CardTitle>
              <CardDescription>
                All prompt optimization jobs and their status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Agent</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Trajectories</TableHead>
                    <TableHead>Variants</TableHead>
                    <TableHead>Improvement</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {jobs.map((job) => (
                    <TableRow key={job.id}>
                      <TableCell className="font-medium">
                        {job.agent_name || "Unknown"}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            job.status === "completed"
                              ? "default"
                              : job.status === "awaiting_approval"
                              ? "secondary"
                              : job.status === "rejected"
                              ? "destructive"
                              : "outline"
                          }
                        >
                          {job.status}
                        </Badge>
                      </TableCell>
                      <TableCell>{job.trajectories_analyzed ?? 0}</TableCell>
                      <TableCell>{job.variants_generated ?? 0}</TableCell>
                      <TableCell>
                        <span
                          className={cn(
                            (job.improvement_percentage ?? 0) > 0
                              ? "text-green-500"
                              : (job.improvement_percentage ?? 0) < 0
                              ? "text-red-500"
                              : ""
                          )}
                        >
                          {(job.improvement_percentage ?? 0) > 0 ? "+" : ""}
                          {(job.improvement_percentage ?? 0).toFixed(1)}%
                        </span>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {new Date(job.created_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setSelectedJob(job);
                            setShowOptimizationDialog(true);
                          }}
                        >
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {jobs.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                        No optimization jobs found
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trajectories Tab */}
        <TabsContent value="trajectories" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Trajectories</CardTitle>
              <CardDescription>
                View agent execution traces for debugging and analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Agent</TableHead>
                    <TableHead>Task Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Quality</TableHead>
                    <TableHead>Tokens</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Cost</TableHead>
                    <TableHead>Time</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {trajectories.map((traj) => (
                    <TableRow key={traj.id}>
                      <TableCell className="font-medium">
                        {traj.agent_name || "Unknown"}
                      </TableCell>
                      <TableCell>{traj.task_type}</TableCell>
                      <TableCell>
                        {traj.success ? (
                          <Badge variant="default" className="bg-green-500">
                            <CheckCircle2 className="h-3 w-3 mr-1" />
                            Success
                          </Badge>
                        ) : (
                          <Badge variant="destructive">
                            <XCircle className="h-3 w-3 mr-1" />
                            Failed
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Progress value={(traj.quality_score ?? 0) * 20} className="w-16 h-2" />
                          <span className="text-sm">{(traj.quality_score ?? 0).toFixed(1)}/5</span>
                        </div>
                      </TableCell>
                      <TableCell>{formatNumber(traj.total_tokens)}</TableCell>
                      <TableCell>{formatDuration(traj.total_duration_ms)}</TableCell>
                      <TableCell>${(traj.total_cost_usd ?? 0).toFixed(4)}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {new Date(traj.created_at).toLocaleTimeString()}
                      </TableCell>
                    </TableRow>
                  ))}
                  {trajectories.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={8} className="text-center text-muted-foreground py-8">
                        No trajectories found
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Optimization Details Dialog */}
      <Dialog open={showOptimizationDialog} onOpenChange={setShowOptimizationDialog}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Optimization Details</DialogTitle>
            <DialogDescription>
              Review the optimization job details and results
            </DialogDescription>
          </DialogHeader>
          {selectedJob && (
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Agent</p>
                  <p className="font-medium">{selectedJob.agent_name || "Unknown"}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Status</p>
                  <Badge>{selectedJob.status}</Badge>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Trajectories Analyzed</p>
                  <p className="font-medium">{selectedJob.trajectories_analyzed ?? 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Variants Generated</p>
                  <p className="font-medium">{selectedJob.variants_generated ?? 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Baseline Success Rate</p>
                  <p className="font-medium">
                    {((selectedJob.baseline_success_rate ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">New Success Rate</p>
                  <p className="font-medium">
                    {((selectedJob.new_success_rate ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <div className="pt-2 border-t">
                <p className="text-sm text-muted-foreground mb-2">Improvement</p>
                <div className="flex items-center gap-2">
                  {(selectedJob.improvement_percentage ?? 0) > 0 ? (
                    <TrendingUp className="h-5 w-5 text-green-500" />
                  ) : (
                    <TrendingDown className="h-5 w-5 text-red-500" />
                  )}
                  <span
                    className={cn(
                      "text-2xl font-bold",
                      (selectedJob.improvement_percentage ?? 0) > 0
                        ? "text-green-500"
                        : "text-red-500"
                    )}
                  >
                    {(selectedJob.improvement_percentage ?? 0) > 0 ? "+" : ""}
                    {(selectedJob.improvement_percentage ?? 0).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowOptimizationDialog(false)}>
              Close
            </Button>
            {selectedJob?.status === "awaiting_approval" && (
              <>
                <Button
                  variant="outline"
                  onClick={() => handleRejectJob(selectedJob.id)}
                  disabled={rejectOptimization.isPending}
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  Reject
                </Button>
                <Button
                  onClick={() => handleApproveJob(selectedJob.id)}
                  disabled={approveOptimization.isPending}
                >
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                  Approve
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
