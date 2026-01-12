"use client";

import { useState, useEffect } from "react";
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
  Loader2,
  Plus,
  Pencil,
  Trash2,
  Wand2,
  Globe,
  Code,
  FolderOpen,
  Link,
  Key,
  TestTube,
  Check,
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import {
  useAgentStatus,
  useOptimizationJobs,
  useApproveOptimization,
  useRejectOptimization,
  useTriggerPromptOptimization,
  useAgentTrajectories,
  useUpdateAgentConfig,
  useLLMProviders,
  useLLMProviderModels,
  useCreateAgent,
  useUpdateAgent,
  useDeleteAgent,
  useAgentSettings,
  useUpdateAgentSettings,
  useEnhanceAgentPrompt,
  useTestExternalAgent,
} from "@/lib/api/hooks";
import type { AgentDefinition, PromptOptimizationJob, AgentTrajectory, AgentToolsConfig, AgentExternalConfig } from "@/lib/api/client";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";

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

  // Configure dialog state
  const [showConfigureDialog, setShowConfigureDialog] = useState(false);
  const [configuringAgent, setConfiguringAgent] = useState<AgentDefinition | null>(null);
  const [configTemperature, setConfigTemperature] = useState(0.7);
  const [configMaxTokens, setConfigMaxTokens] = useState(2048);
  const [configProviderId, setConfigProviderId] = useState<string>("");
  const [configModel, setConfigModel] = useState<string>("");

  // Create Agent dialog state
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newAgentName, setNewAgentName] = useState("");
  const [newAgentType, setNewAgentType] = useState("");
  const [newAgentDescription, setNewAgentDescription] = useState("");
  const [newAgentTemperature, setNewAgentTemperature] = useState(0.7);
  const [newAgentMaxTokens, setNewAgentMaxTokens] = useState(2048);

  // Edit Agent dialog state
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [editingAgent, setEditingAgent] = useState<AgentDefinition | null>(null);
  const [editAgentName, setEditAgentName] = useState("");
  const [editAgentDescription, setEditAgentDescription] = useState("");
  const [editAgentTemperature, setEditAgentTemperature] = useState(0.7);
  const [editAgentMaxTokens, setEditAgentMaxTokens] = useState(2048);

  // Delete Agent dialog state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deletingAgent, setDeletingAgent] = useState<AgentDefinition | null>(null);

  // Enhancement dialog state
  const [showEnhanceDialog, setShowEnhanceDialog] = useState(false);
  const [enhancingAgent, setEnhancingAgent] = useState<AgentDefinition | null>(null);
  const [enhanceStrategy, setEnhanceStrategy] = useState<string>("auto");
  const [enhanceInstructions, setEnhanceInstructions] = useState("");
  const [enhanceDescription, setEnhanceDescription] = useState(false);
  const [enhancedPrompt, setEnhancedPrompt] = useState<{
    system_prompt: string;
    task_prompt_template: string;
    few_shot_examples: Array<{ input: string; output: string }>;
  } | null>(null);
  const [enhancedDescription, setEnhancedDescription] = useState<string | null>(null);
  // Editable versions of enhanced prompts
  const [editedSystemPrompt, setEditedSystemPrompt] = useState<string>("");
  const [editedTaskTemplate, setEditedTaskTemplate] = useState<string>("");
  const [editedDescription, setEditedDescription] = useState<string>("");

  // Tools & External Agent dialog state
  const [showToolsDialog, setShowToolsDialog] = useState(false);
  const [toolsAgent, setToolsAgent] = useState<AgentDefinition | null>(null);
  const [toolsConfig, setToolsConfig] = useState<AgentToolsConfig>({
    web_search: false,
    code_execution: false,
    file_access: false,
    mcp_server_url: null,
  });
  const [externalConfig, setExternalConfig] = useState<AgentExternalConfig>({
    api_url: null,
    api_key: null,
    enabled: false,
  });

  // Trajectory detail dialog state
  const [showTrajectoryDialog, setShowTrajectoryDialog] = useState(false);
  const [selectedTrajectory, setSelectedTrajectory] = useState<AgentTrajectory | null>(null);

  // Queries - only fetch when authenticated
  const { data: statusData, isLoading: statusLoading, refetch: refetchStatus } = useAgentStatus({ enabled: isAuthenticated });
  const { data: jobsData, isLoading: jobsLoading } = useOptimizationJobs(undefined, { enabled: isAuthenticated });
  const { data: trajectoriesData } = useAgentTrajectories({ limit: 50 }, { enabled: isAuthenticated });
  const { data: providersData } = useLLMProviders({ enabled: isAuthenticated });
  const { data: providerModelsData } = useLLMProviderModels(configProviderId, {
    enabled: isAuthenticated && !!configProviderId,
  });

  // Mutations
  const approveOptimization = useApproveOptimization();
  const rejectOptimization = useRejectOptimization();
  const triggerOptimization = useTriggerPromptOptimization();
  const updateAgentConfig = useUpdateAgentConfig();
  const createAgentMutation = useCreateAgent();
  const updateAgentMutation = useUpdateAgent();
  const deleteAgentMutation = useDeleteAgent();
  const enhancePromptMutation = useEnhanceAgentPrompt();
  const updateSettingsMutation = useUpdateAgentSettings();
  const testExternalMutation = useTestExternalAgent();

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
      toast.success(
        agent.is_active ? "Agent Disabled" : "Agent Enabled",
        { description: `${agent.name} has been ${agent.is_active ? "disabled" : "enabled"}.` }
      );
      refetchStatus();
    } catch (error) {
      console.error("Failed to toggle agent:", error);
      toast.error("Error", { description: "Failed to update agent status. Please try again." });
    }
  };

  const handleTriggerOptimization = async (agentId: string) => {
    try {
      await triggerOptimization.mutateAsync(agentId);
      toast.success("Optimization Started", { description: "Prompt optimization has been triggered." });
    } catch (error) {
      console.error("Failed to trigger optimization:", error);
      toast.error("Error", { description: "Failed to trigger optimization. Please try again." });
    }
  };

  const handleApproveJob = async (jobId: string) => {
    try {
      await approveOptimization.mutateAsync(jobId);
      setShowOptimizationDialog(false);
      setSelectedJob(null);
      toast.success("Optimization Approved", { description: "The new prompt version has been activated." });
    } catch (error) {
      console.error("Failed to approve optimization:", error);
      toast.error("Error", { description: "Failed to approve optimization. Please try again." });
    }
  };

  const handleRejectJob = async (jobId: string) => {
    try {
      await rejectOptimization.mutateAsync({ jobId, reason: "Manual rejection" });
      setShowOptimizationDialog(false);
      setSelectedJob(null);
      toast.success("Optimization Rejected", { description: "The prompt optimization has been rejected." });
    } catch (error) {
      console.error("Failed to reject optimization:", error);
      toast.error("Error", { description: "Failed to reject optimization. Please try again." });
    }
  };

  // Configure dialog handlers
  const handleOpenConfigure = (agent: AgentDefinition) => {
    setConfiguringAgent(agent);
    setConfigTemperature(agent.default_temperature ?? 0.7);
    setConfigMaxTokens(agent.max_tokens ?? 2048);
    setConfigProviderId(agent.default_provider_id ?? "");
    setConfigModel(agent.default_model ?? "");
    setShowConfigureDialog(true);
  };

  const handleSaveConfig = async () => {
    if (!configuringAgent) return;
    try {
      await updateAgentConfig.mutateAsync({
        agentId: configuringAgent.id,
        data: {
          temperature: configTemperature,
          max_tokens: configMaxTokens,
          provider_id: configProviderId || undefined,
          model: configModel || undefined,
        },
      });
      toast.success("Configuration Saved", {
        description: `${configuringAgent.name} settings have been updated.`,
      });
      setShowConfigureDialog(false);
      setConfiguringAgent(null);
      refetchStatus();
    } catch (error) {
      console.error("Failed to save configuration:", error);
      toast.error("Error", { description: "Failed to save configuration. Please try again." });
    }
  };

  // Create Agent handlers
  const handleOpenCreate = () => {
    setNewAgentName("");
    setNewAgentType("");
    setNewAgentDescription("");
    setNewAgentTemperature(0.7);
    setNewAgentMaxTokens(2048);
    setShowCreateDialog(true);
  };

  const handleCreateAgent = async () => {
    if (!newAgentName.trim() || !newAgentType.trim()) {
      toast.error("Error", { description: "Name and type are required." });
      return;
    }
    try {
      await createAgentMutation.mutateAsync({
        name: newAgentName.trim(),
        agent_type: newAgentType.trim().toLowerCase().replace(/\s+/g, "_"),
        description: newAgentDescription.trim() || undefined,
        default_temperature: newAgentTemperature,
        max_tokens: newAgentMaxTokens,
      });
      toast.success("Agent Created", { description: `${newAgentName} has been created.` });
      setShowCreateDialog(false);
      refetchStatus();
    } catch (error: unknown) {
      console.error("Failed to create agent:", error);
      const message = error instanceof Error ? error.message : "Failed to create agent.";
      toast.error("Error", { description: message });
    }
  };

  // Edit Agent handlers
  const handleOpenEdit = (agent: AgentDefinition) => {
    setEditingAgent(agent);
    setEditAgentName(agent.name);
    setEditAgentDescription(agent.description || "");
    setEditAgentTemperature(agent.default_temperature ?? 0.7);
    setEditAgentMaxTokens(agent.max_tokens ?? 2048);
    setShowEditDialog(true);
  };

  const handleUpdateAgent = async () => {
    if (!editingAgent || !editAgentName.trim()) {
      toast.error("Error", { description: "Name is required." });
      return;
    }
    try {
      await updateAgentMutation.mutateAsync({
        agentId: editingAgent.id,
        data: {
          name: editAgentName.trim(),
          description: editAgentDescription.trim() || undefined,
          default_temperature: editAgentTemperature,
          max_tokens: editAgentMaxTokens,
        },
      });
      toast.success("Agent Updated", { description: `${editAgentName} has been updated.` });
      setShowEditDialog(false);
      setEditingAgent(null);
      refetchStatus();
    } catch (error: unknown) {
      console.error("Failed to update agent:", error);
      const message = error instanceof Error ? error.message : "Failed to update agent.";
      toast.error("Error", { description: message });
    }
  };

  // Delete Agent handlers
  const handleOpenDelete = (agent: AgentDefinition) => {
    setDeletingAgent(agent);
    setShowDeleteDialog(true);
  };

  const handleDeleteAgent = async (hardDelete: boolean = false) => {
    if (!deletingAgent) return;
    try {
      await deleteAgentMutation.mutateAsync({ agentId: deletingAgent.id, hardDelete });
      const action = hardDelete ? "permanently deleted" : "deactivated";
      toast.success("Agent Deleted", { description: `${deletingAgent.name} has been ${action}.` });
      setShowDeleteDialog(false);
      setDeletingAgent(null);
      refetchStatus();
    } catch (error: unknown) {
      console.error("Failed to delete agent:", error);
      const message = error instanceof Error ? error.message : "Failed to delete agent.";
      toast.error("Error", { description: message });
    }
  };

  // Enhancement handlers
  const handleOpenEnhance = (agent: AgentDefinition) => {
    setEnhancingAgent(agent);
    setEnhanceStrategy("auto");
    setEnhanceInstructions("");
    setEnhanceDescription(!!agent.description); // Default to true if agent has description
    setEnhancedPrompt(null);
    setEnhancedDescription(null);
    setShowEnhanceDialog(true);
  };

  const handleEnhancePrompt = async () => {
    if (!enhancingAgent) return;
    try {
      const result = await enhancePromptMutation.mutateAsync({
        agentId: enhancingAgent.id,
        request: {
          strategy: enhanceStrategy === "auto" ? undefined : enhanceStrategy,
          custom_instructions: enhanceInstructions || undefined,
          enhance_description: enhanceDescription,
        },
      });
      setEnhancedPrompt(result.enhanced_prompt);
      setEnhancedDescription(result.enhanced_description || null);
      // Set editable versions
      setEditedSystemPrompt(result.enhanced_prompt.system_prompt || "");
      setEditedTaskTemplate(result.enhanced_prompt.task_prompt_template || "");
      setEditedDescription(result.enhanced_description || "");
      toast.success("Prompt Enhanced", {
        description: `Strategy: ${result.strategy_used}. ${result.change_description}`,
      });
    } catch (error: unknown) {
      console.error("Failed to enhance prompt:", error);
      const message = error instanceof Error ? error.message : "Failed to enhance prompt.";
      toast.error("Error", { description: message });
    }
  };

  const handleApplyEnhancedPrompt = async () => {
    if (!enhancingAgent || !enhancedPrompt) return;
    try {
      // Use edited values (which may have been modified by the user)
      await updateAgentMutation.mutateAsync({
        agentId: enhancingAgent.id,
        data: {
          system_prompt: editedSystemPrompt,
          task_prompt_template: editedTaskTemplate,
          // Include enhanced description if available and was edited
          ...(editedDescription && { description: editedDescription }),
        },
      });
      toast.success("Enhancements Applied", {
        description: `Enhanced prompt${editedDescription ? " and description" : ""} saved to ${enhancingAgent.name}.`,
      });
      setShowEnhanceDialog(false);
      setEnhancedPrompt(null);
      setEnhancedDescription(null);
      setEditedSystemPrompt("");
      setEditedTaskTemplate("");
      setEditedDescription("");
      refetchStatus();
    } catch (error: unknown) {
      console.error("Failed to apply enhanced prompt:", error);
      const message = error instanceof Error ? error.message : "Failed to apply prompt.";
      toast.error("Error", { description: message });
    }
  };

  // Tools & External Agent handlers
  const handleOpenTools = async (agent: AgentDefinition) => {
    setToolsAgent(agent);
    // Load current settings from agent.settings or use defaults
    const settings = (agent as { settings?: { tools_config?: AgentToolsConfig; external_agent?: AgentExternalConfig } }).settings || {};
    setToolsConfig(settings.tools_config || {
      web_search: false,
      code_execution: false,
      file_access: false,
      mcp_server_url: null,
    });
    setExternalConfig(settings.external_agent || {
      api_url: null,
      api_key: null,
      enabled: false,
    });
    setShowToolsDialog(true);
  };

  const handleSaveTools = async () => {
    if (!toolsAgent) return;
    try {
      await updateSettingsMutation.mutateAsync({
        agentId: toolsAgent.id,
        data: {
          tools_config: toolsConfig,
          external_agent: externalConfig,
        },
      });
      toast.success("Settings Saved", { description: "Agent tools and external configuration updated." });
      setShowToolsDialog(false);
      refetchStatus();
    } catch (error: unknown) {
      console.error("Failed to save settings:", error);
      const message = error instanceof Error ? error.message : "Failed to save settings.";
      toast.error("Error", { description: message });
    }
  };

  const handleTestExternal = async () => {
    if (!toolsAgent) return;
    try {
      const result = await testExternalMutation.mutateAsync(toolsAgent.id);
      if (result.success) {
        toast.success("Connection Successful", { description: result.message });
      } else {
        toast.error("Connection Failed", { description: result.message });
      }
    } catch (error: unknown) {
      console.error("Failed to test external agent:", error);
      const message = error instanceof Error ? error.message : "Failed to test connection.";
      toast.error("Error", { description: message });
    }
  };

  // Get the selected provider's models from the models endpoint
  const availableModels = providerModelsData?.chat_models || [];

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
        <div className="flex items-center gap-2">
          <Button onClick={handleOpenCreate} size="sm">
            <Plus className="h-4 w-4 mr-2" />
            Add Agent
          </Button>
          <Button onClick={() => refetchStatus()} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
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
                            <DropdownMenuItem onClick={() => handleOpenEnhance(agent)}>
                              <Wand2 className="h-4 w-4 mr-2" />
                              Enhance Prompt
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <History className="h-4 w-4 mr-2" />
                              View Prompt History
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleOpenConfigure(agent)}>
                              <Settings2 className="h-4 w-4 mr-2" />
                              Configure LLM
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleOpenTools(agent)}>
                              <Wrench className="h-4 w-4 mr-2" />
                              Tools & External Agent
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => handleOpenEdit(agent)}>
                              <Pencil className="h-4 w-4 mr-2" />
                              Edit Agent
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => handleOpenDelete(agent)}
                              className="text-destructive focus:text-destructive"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete Agent
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
                    <TableRow
                      key={traj.id}
                      className="cursor-pointer hover:bg-muted/50"
                      onClick={() => {
                        setSelectedTrajectory(traj);
                        setShowTrajectoryDialog(true);
                      }}
                    >
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

      {/* Configure Agent Dialog */}
      <Dialog open={showConfigureDialog} onOpenChange={setShowConfigureDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Configure {configuringAgent?.name}</DialogTitle>
            <DialogDescription>
              Adjust the LLM settings for this agent. Changes apply to all future executions.
            </DialogDescription>
          </DialogHeader>
          {configuringAgent && (
            <div className="space-y-6 py-4">
              {/* Provider Selection */}
              <div className="space-y-2">
                <Label htmlFor="provider">LLM Provider</Label>
                <Select
                  value={configProviderId || "__default__"}
                  onValueChange={(value) => {
                    setConfigProviderId(value === "__default__" ? "" : value);
                    setConfigModel(""); // Reset model when provider changes
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Use default provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__default__">Use Default</SelectItem>
                    {providersData?.providers?.map((provider) => (
                      <SelectItem key={provider.id} value={provider.id}>
                        {provider.name} ({provider.provider_type})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Select a specific provider or use the system default.
                </p>
              </div>

              {/* Model Selection */}
              {configProviderId && availableModels.length > 0 && (
                <div className="space-y-2">
                  <Label htmlFor="model">Model</Label>
                  <Select
                    value={configModel || "__default__"}
                    onValueChange={(value) => setConfigModel(value === "__default__" ? "" : value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Use provider default" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__default__">Use Provider Default</SelectItem>
                      {availableModels.map((model: string) => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              {/* Temperature Slider */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Temperature</Label>
                  <span className="text-sm text-muted-foreground">{configTemperature.toFixed(2)}</span>
                </div>
                <Slider
                  value={[configTemperature]}
                  onValueChange={([value]) => setConfigTemperature(value)}
                  min={0}
                  max={2}
                  step={0.05}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Precise (0)</span>
                  <span>Balanced (0.7)</span>
                  <span>Creative (2)</span>
                </div>
              </div>

              {/* Max Tokens */}
              <div className="space-y-2">
                <Label htmlFor="maxTokens">Max Tokens</Label>
                <Input
                  id="maxTokens"
                  type="number"
                  min={256}
                  max={128000}
                  step={256}
                  value={configMaxTokens}
                  onChange={(e) => setConfigMaxTokens(parseInt(e.target.value) || 2048)}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum tokens in the response. Higher values allow longer outputs.
                </p>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowConfigureDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveConfig} disabled={updateAgentConfig.isPending}>
              {updateAgentConfig.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Agent Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Agent</DialogTitle>
            <DialogDescription>
              Create a custom agent with specific capabilities and settings.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="newAgentName">Agent Name *</Label>
              <Input
                id="newAgentName"
                placeholder="e.g., Summarizer Agent"
                value={newAgentName}
                onChange={(e) => setNewAgentName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="newAgentType">Agent Type *</Label>
              <Input
                id="newAgentType"
                placeholder="e.g., summarizer (lowercase, no spaces)"
                value={newAgentType}
                onChange={(e) => setNewAgentType(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Unique identifier for this agent type. Use lowercase with underscores.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="newAgentDescription">Description</Label>
              <Textarea
                id="newAgentDescription"
                placeholder="Describe what this agent does..."
                value={newAgentDescription}
                onChange={(e) => setNewAgentDescription(e.target.value)}
                rows={3}
              />
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Temperature</Label>
                <span className="text-sm text-muted-foreground">{newAgentTemperature.toFixed(2)}</span>
              </div>
              <Slider
                value={[newAgentTemperature]}
                onValueChange={([value]) => setNewAgentTemperature(value)}
                min={0}
                max={2}
                step={0.05}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="newAgentMaxTokens">Max Tokens</Label>
              <Input
                id="newAgentMaxTokens"
                type="number"
                min={256}
                max={128000}
                value={newAgentMaxTokens}
                onChange={(e) => setNewAgentMaxTokens(parseInt(e.target.value) || 2048)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateAgent}
              disabled={createAgentMutation.isPending || !newAgentName.trim() || !newAgentType.trim()}
            >
              {createAgentMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Create Agent
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Agent Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Agent</DialogTitle>
            <DialogDescription>
              Update {editingAgent?.name} settings. Type cannot be changed after creation.
            </DialogDescription>
          </DialogHeader>
          {editingAgent && (
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="editAgentName">Agent Name *</Label>
                <Input
                  id="editAgentName"
                  value={editAgentName}
                  onChange={(e) => setEditAgentName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Agent Type</Label>
                <Input value={editingAgent.agent_type} disabled className="bg-muted" />
                <p className="text-xs text-muted-foreground">
                  Agent type cannot be changed after creation.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="editAgentDescription">Description</Label>
                <Textarea
                  id="editAgentDescription"
                  value={editAgentDescription}
                  onChange={(e) => setEditAgentDescription(e.target.value)}
                  rows={3}
                />
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Temperature</Label>
                  <span className="text-sm text-muted-foreground">{editAgentTemperature.toFixed(2)}</span>
                </div>
                <Slider
                  value={[editAgentTemperature]}
                  onValueChange={([value]) => setEditAgentTemperature(value)}
                  min={0}
                  max={2}
                  step={0.05}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="editAgentMaxTokens">Max Tokens</Label>
                <Input
                  id="editAgentMaxTokens"
                  type="number"
                  min={256}
                  max={128000}
                  value={editAgentMaxTokens}
                  onChange={(e) => setEditAgentMaxTokens(parseInt(e.target.value) || 2048)}
                />
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEditDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleUpdateAgent}
              disabled={updateAgentMutation.isPending || !editAgentName.trim()}
            >
              {updateAgentMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Agent Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Agent</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete {deletingAgent?.name}?
            </DialogDescription>
          </DialogHeader>
          {deletingAgent && (
            <div className="py-4">
              <div className="rounded-lg border p-4 bg-muted/50">
                <div className="flex items-center gap-3">
                  <div className={cn("p-2 rounded-lg", agentColors[deletingAgent.agent_type] || "bg-gray-500")}>
                    {agentIcons[deletingAgent.agent_type] || <Bot className="h-5 w-5 text-white" />}
                  </div>
                  <div>
                    <p className="font-medium">{deletingAgent.name}</p>
                    <p className="text-sm text-muted-foreground">{deletingAgent.agent_type}</p>
                  </div>
                </div>
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                <strong>Deactivate</strong> will disable the agent but keep its data.
                <br />
                <strong>Delete Permanently</strong> will remove the agent and all associated data.
              </p>
              {["manager", "generator", "critic", "research", "tool_executor"].includes(deletingAgent.agent_type) && (
                <p className="mt-2 text-sm text-amber-600 dark:text-amber-400">
                  Note: Core agents cannot be permanently deleted.
                </p>
              )}
            </div>
          )}
          <DialogFooter className="flex-col sm:flex-row gap-2">
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
              Cancel
            </Button>
            <Button
              variant="secondary"
              onClick={() => handleDeleteAgent(false)}
              disabled={deleteAgentMutation.isPending}
            >
              {deleteAgentMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Deactivate
            </Button>
            {deletingAgent && !["manager", "generator", "critic", "research", "tool_executor"].includes(deletingAgent.agent_type) && (
              <Button
                variant="destructive"
                onClick={() => handleDeleteAgent(true)}
                disabled={deleteAgentMutation.isPending}
              >
                {deleteAgentMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Delete Permanently
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Enhance Prompt Dialog */}
      <Dialog open={showEnhanceDialog} onOpenChange={setShowEnhanceDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Wand2 className="h-5 w-5" />
              Enhance Agent Prompt
            </DialogTitle>
            <DialogDescription>
              Use AI to analyze and improve the agent&apos;s prompt using GEPA-style optimization.
            </DialogDescription>
          </DialogHeader>
          {enhancingAgent && (
            <div className="space-y-4 py-4">
              <div className="rounded-lg border p-4 bg-muted/50">
                <div className="flex items-center gap-3">
                  <div className={cn("p-2 rounded-lg", agentColors[enhancingAgent.agent_type] || "bg-gray-500")}>
                    {agentIcons[enhancingAgent.agent_type] || <Bot className="h-5 w-5 text-white" />}
                  </div>
                  <div>
                    <p className="font-medium">{enhancingAgent.name}</p>
                    <p className="text-sm text-muted-foreground">{enhancingAgent.agent_type}</p>
                  </div>
                </div>
              </div>

              {/* Current Prompt Display */}
              <div className="space-y-3 border rounded-lg p-4 bg-muted/30">
                <h4 className="font-medium text-sm">Current Prompt</h4>
                <div className="space-y-2">
                  {/* Description - metadata about what the agent does */}
                  {enhancingAgent.description && (
                    <div>
                      <Label className="text-xs text-muted-foreground">Description (metadata)</Label>
                      <pre className="text-xs bg-background p-3 rounded border max-h-24 overflow-auto whitespace-pre-wrap">
                        {enhancingAgent.description}
                      </pre>
                    </div>
                  )}
                  <div>
                    <Label className="text-xs text-muted-foreground">System Prompt</Label>
                    <pre className="text-xs bg-background p-3 rounded border max-h-48 overflow-auto whitespace-pre-wrap">
                      {enhancingAgent.system_prompt || "(No system prompt set - uses description or default)"}
                    </pre>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Task Template</Label>
                    <pre className="text-xs bg-background p-3 rounded border max-h-48 overflow-auto whitespace-pre-wrap">
                      {enhancingAgent.task_prompt_template || "(No task template set)"}
                    </pre>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="enhance-strategy">Enhancement Strategy</Label>
                <Select value={enhanceStrategy} onValueChange={setEnhanceStrategy}>
                  <SelectTrigger>
                    <SelectValue placeholder="Auto-detect best strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto-detect best strategy</SelectItem>
                    <SelectItem value="rephrase_instructions">Rephrase Instructions</SelectItem>
                    <SelectItem value="add_examples">Add Examples</SelectItem>
                    <SelectItem value="add_guardrails">Add Guardrails</SelectItem>
                    <SelectItem value="restructure_format">Restructure Format</SelectItem>
                    <SelectItem value="add_chain_of_thought">Add Chain of Thought</SelectItem>
                    <SelectItem value="simplify">Simplify</SelectItem>
                    <SelectItem value="add_constraints">Add Constraints</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="enhance-instructions">Custom Instructions (optional)</Label>
                <Textarea
                  id="enhance-instructions"
                  placeholder="Additional context or specific improvements you want..."
                  value={enhanceInstructions}
                  onChange={(e) => setEnhanceInstructions(e.target.value)}
                  rows={3}
                />
              </div>

              {/* Enhance Description Checkbox */}
              {enhancingAgent.description && (
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="enhance-description"
                    checked={enhanceDescription}
                    onCheckedChange={(checked) => setEnhanceDescription(checked === true)}
                  />
                  <Label htmlFor="enhance-description" className="text-sm cursor-pointer">
                    Also enhance agent description
                  </Label>
                </div>
              )}

              {enhancedPrompt && (
                <div className="space-y-3 border rounded-lg p-4 bg-green-50 dark:bg-green-950/20">
                  <h4 className="font-medium text-green-700 dark:text-green-400">Enhanced Prompt Generated</h4>
                  <p className="text-xs text-muted-foreground">Edit the prompts below before applying.</p>
                  <div className="space-y-3">
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">Enhanced System Prompt</Label>
                      <Textarea
                        value={editedSystemPrompt}
                        onChange={(e) => setEditedSystemPrompt(e.target.value)}
                        className="text-xs font-mono min-h-[120px] max-h-[200px]"
                        placeholder="System prompt..."
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs text-muted-foreground">Enhanced Task Template</Label>
                      <Textarea
                        value={editedTaskTemplate}
                        onChange={(e) => setEditedTaskTemplate(e.target.value)}
                        className="text-xs font-mono min-h-[120px] max-h-[200px]"
                        placeholder="Task template..."
                      />
                    </div>
                    {enhancedPrompt.few_shot_examples.length > 0 && (
                      <p className="text-xs text-muted-foreground">
                        + {enhancedPrompt.few_shot_examples.length} few-shot examples (not editable here)
                      </p>
                    )}
                    {enhancedDescription && (
                      <div className="space-y-1">
                        <Label className="text-xs text-muted-foreground">Enhanced Description</Label>
                        <Textarea
                          value={editedDescription}
                          onChange={(e) => setEditedDescription(e.target.value)}
                          className="text-xs min-h-[60px] max-h-[100px]"
                          placeholder="Agent description..."
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
          <DialogFooter className="flex-col sm:flex-row gap-2">
            <Button variant="outline" onClick={() => setShowEnhanceDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleEnhancePrompt}
              disabled={enhancePromptMutation.isPending || updateAgentMutation.isPending}
              variant={enhancedPrompt ? "outline" : "default"}
            >
              {enhancePromptMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Enhancing...
                </>
              ) : (
                <>
                  <Wand2 className="h-4 w-4 mr-2" />
                  {enhancedPrompt ? "Re-enhance" : "Enhance Prompt"}
                </>
              )}
            </Button>
            {enhancedPrompt && (
              <Button
                onClick={handleApplyEnhancedPrompt}
                disabled={updateAgentMutation.isPending}
              >
                {updateAgentMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Applying...
                  </>
                ) : (
                  <>
                    <Check className="h-4 w-4 mr-2" />
                    Apply Enhanced Prompt
                  </>
                )}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Tools & External Agent Dialog */}
      <Dialog open={showToolsDialog} onOpenChange={setShowToolsDialog}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Wrench className="h-5 w-5" />
              Tools & External Agent
            </DialogTitle>
            <DialogDescription>
              Configure agent capabilities and external agent connections.
            </DialogDescription>
          </DialogHeader>
          {toolsAgent && (
            <div className="space-y-6 py-4">
              {/* Tools Configuration */}
              <div className="space-y-4">
                <h4 className="font-medium">Tools & Capabilities</h4>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Globe className="h-4 w-4 text-muted-foreground" />
                      <Label htmlFor="web-search">Web Search</Label>
                    </div>
                    <Switch
                      id="web-search"
                      checked={toolsConfig.web_search}
                      onCheckedChange={(checked) => setToolsConfig((prev) => ({ ...prev, web_search: checked }))}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Code className="h-4 w-4 text-muted-foreground" />
                      <Label htmlFor="code-exec">Code Execution</Label>
                    </div>
                    <Switch
                      id="code-exec"
                      checked={toolsConfig.code_execution}
                      onCheckedChange={(checked) => setToolsConfig((prev) => ({ ...prev, code_execution: checked }))}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FolderOpen className="h-4 w-4 text-muted-foreground" />
                      <Label htmlFor="file-access">File Access</Label>
                    </div>
                    <Switch
                      id="file-access"
                      checked={toolsConfig.file_access}
                      onCheckedChange={(checked) => setToolsConfig((prev) => ({ ...prev, file_access: checked }))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="mcp-url">MCP Server URL</Label>
                    <Input
                      id="mcp-url"
                      placeholder="http://localhost:3000/mcp"
                      value={toolsConfig.mcp_server_url || ""}
                      onChange={(e) => setToolsConfig((prev) => ({ ...prev, mcp_server_url: e.target.value || null }))}
                    />
                  </div>
                </div>
              </div>

              <div className="border-t my-2" />

              {/* External Agent Configuration */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">External Agent</h4>
                  <Switch
                    id="external-enabled"
                    checked={externalConfig.enabled}
                    onCheckedChange={(checked) => setExternalConfig((prev) => ({ ...prev, enabled: checked }))}
                  />
                </div>
                {externalConfig.enabled && (
                  <div className="space-y-3">
                    <div className="space-y-2">
                      <Label htmlFor="ext-url">
                        <Link className="h-3 w-3 inline mr-1" />
                        API URL
                      </Label>
                      <Input
                        id="ext-url"
                        placeholder="https://external-agent.example.com/v1/chat"
                        value={externalConfig.api_url || ""}
                        onChange={(e) => setExternalConfig((prev) => ({ ...prev, api_url: e.target.value || null }))}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="ext-key">
                        <Key className="h-3 w-3 inline mr-1" />
                        API Key
                      </Label>
                      <Input
                        id="ext-key"
                        type="password"
                        placeholder="sk-..."
                        value={externalConfig.api_key || ""}
                        onChange={(e) => setExternalConfig((prev) => ({ ...prev, api_key: e.target.value || null }))}
                      />
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleTestExternal}
                      disabled={testExternalMutation.isPending || !externalConfig.api_url}
                    >
                      {testExternalMutation.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <TestTube className="h-4 w-4 mr-2" />
                      )}
                      Test Connection
                    </Button>
                  </div>
                )}
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowToolsDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleSaveTools}
              disabled={updateSettingsMutation.isPending}
            >
              {updateSettingsMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save Settings
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Trajectory Detail Dialog */}
      <Dialog open={showTrajectoryDialog} onOpenChange={setShowTrajectoryDialog}>
        <DialogContent className="sm:max-w-3xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Execution Trajectory
            </DialogTitle>
            <DialogDescription>
              View detailed execution steps and results
            </DialogDescription>
          </DialogHeader>
          {selectedTrajectory && (
            <ScrollArea className="max-h-[60vh] pr-4">
              <div className="space-y-4">
                {/* Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Agent</p>
                    <p className="font-medium">{selectedTrajectory.agent_name || "Unknown"}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Task Type</p>
                    <p className="font-medium">{selectedTrajectory.task_type}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Status</p>
                    {selectedTrajectory.success ? (
                      <Badge variant="default" className="bg-green-500">Success</Badge>
                    ) : (
                      <Badge variant="destructive">Failed</Badge>
                    )}
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Quality Score</p>
                    <p className="font-medium">{(selectedTrajectory.quality_score ?? 0).toFixed(1)}/5</p>
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-4 p-3 bg-muted/50 rounded-lg">
                  <div className="text-center">
                    <p className="text-2xl font-bold">{formatNumber(selectedTrajectory.total_tokens)}</p>
                    <p className="text-xs text-muted-foreground">Tokens</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">{formatDuration(selectedTrajectory.total_duration_ms)}</p>
                    <p className="text-xs text-muted-foreground">Duration</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">${(selectedTrajectory.total_cost_usd ?? 0).toFixed(4)}</p>
                    <p className="text-xs text-muted-foreground">Cost</p>
                  </div>
                </div>

                {/* Input Summary */}
                <div className="space-y-2">
                  <h4 className="font-medium flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />
                    Input Summary
                  </h4>
                  <div className="p-3 bg-muted/50 rounded-lg text-sm">
                    {selectedTrajectory.input_summary || "No input summary available"}
                  </div>
                </div>

                {/* Error Message */}
                {selectedTrajectory.error_message && (
                  <div className="space-y-2">
                    <h4 className="font-medium flex items-center gap-2 text-destructive">
                      <AlertCircle className="h-4 w-4" />
                      Error
                    </h4>
                    <div className="p-3 bg-destructive/10 text-destructive rounded-lg text-sm">
                      {selectedTrajectory.error_message}
                    </div>
                  </div>
                )}

                {/* Trajectory Steps */}
                <div className="space-y-2">
                  <h4 className="font-medium flex items-center gap-2">
                    <History className="h-4 w-4" />
                    Execution Steps ({Array.isArray(selectedTrajectory.trajectory_steps) ? selectedTrajectory.trajectory_steps.length : 0})
                  </h4>
                  <div className="space-y-2">
                    {Array.isArray(selectedTrajectory.trajectory_steps) && selectedTrajectory.trajectory_steps.length > 0 ? (
                      selectedTrajectory.trajectory_steps.map((step, index) => {
                        const stepObj = step as Record<string, unknown>;
                        return (
                          <div key={index} className="p-3 border rounded-lg space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="font-medium text-sm">
                                Step {index + 1}: {String(stepObj.action || stepObj.type || "Unknown")}
                              </span>
                              {stepObj.duration_ms ? (
                                <span className="text-xs text-muted-foreground">
                                  {formatDuration(Number(stepObj.duration_ms))}
                                </span>
                              ) : null}
                            </div>
                            {stepObj.input ? (
                              <div className="text-xs">
                                <span className="text-muted-foreground">Input: </span>
                                <code className="bg-muted px-1 rounded">
                                  {typeof stepObj.input === "string"
                                    ? (stepObj.input as string).slice(0, 100) + ((stepObj.input as string).length > 100 ? "..." : "")
                                    : JSON.stringify(stepObj.input).slice(0, 100)}
                                </code>
                              </div>
                            ) : null}
                            {stepObj.output ? (
                              <div className="text-xs">
                                <span className="text-muted-foreground">Output: </span>
                                <code className="bg-muted px-1 rounded">
                                  {typeof stepObj.output === "string"
                                    ? (stepObj.output as string).slice(0, 100) + ((stepObj.output as string).length > 100 ? "..." : "")
                                    : JSON.stringify(stepObj.output).slice(0, 100)}
                                </code>
                              </div>
                            ) : null}
                            {stepObj.error ? (
                              <div className="text-xs text-destructive">
                                <span>Error: </span>
                                {String(stepObj.error)}
                              </div>
                            ) : null}
                          </div>
                        );
                      })
                    ) : (
                      <div className="p-3 text-center text-muted-foreground text-sm">
                        No execution steps recorded
                      </div>
                    )}
                  </div>
                </div>

                {/* User Feedback */}
                {(selectedTrajectory.user_rating || selectedTrajectory.user_feedback) && (
                  <div className="space-y-2">
                    <h4 className="font-medium">User Feedback</h4>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      {selectedTrajectory.user_rating && (
                        <p className="text-sm">Rating: {selectedTrajectory.user_rating}/5</p>
                      )}
                      {selectedTrajectory.user_feedback && (
                        <p className="text-sm mt-1">{selectedTrajectory.user_feedback}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Metadata */}
                <div className="text-xs text-muted-foreground pt-2 border-t">
                  <p>Session ID: {selectedTrajectory.session_id}</p>
                  <p>Created: {new Date(selectedTrajectory.created_at).toLocaleString()}</p>
                </div>
              </div>
            </ScrollArea>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowTrajectoryDialog(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
