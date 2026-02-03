"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Plus,
  Search,
  GitBranch,
  Play,
  Pause,
  Trash2,
  Copy,
  MoreHorizontal,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Filter,
  Calendar,
  Webhook,
  FormInput,
  Zap,
  ExternalLink,
  Check,
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  useWorkflows,
  useCreateWorkflow,
  useDeleteWorkflow,
  useDuplicateWorkflow,
  usePublishWorkflow,
  useUpdateWorkflow,
  type WorkflowListItem,
  type WorkflowTriggerType,
} from "@/lib/api";
import { toast } from "sonner";
import { formatDistanceToNow } from "date-fns";

const triggerIcons: Record<WorkflowTriggerType, React.ElementType> = {
  manual: Play,
  scheduled: Calendar,
  webhook: Webhook,
  form: FormInput,
  event: Zap,
};

const triggerLabels: Record<WorkflowTriggerType, string> = {
  manual: "Manual",
  scheduled: "Scheduled",
  webhook: "Webhook",
  form: "Form",
  event: "Event",
};

function getStatusBadge(workflow: WorkflowListItem) {
  if (workflow.is_draft) {
    return <Badge variant="outline">Draft</Badge>;
  }
  if (workflow.is_active) {
    return <Badge className="bg-green-500/10 text-green-600 border-green-200">Active</Badge>;
  }
  return <Badge variant="secondary">Inactive</Badge>;
}

function WorkflowCard({
  workflow,
  onEdit,
  onDelete,
  onDuplicate,
  onToggleActive,
  onDeploy,
}: {
  workflow: WorkflowListItem;
  onEdit: () => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onToggleActive: () => void;
  onDeploy: () => void;
}) {
  const TriggerIcon = triggerIcons[workflow.trigger_type] || Play;
  const successRate =
    workflow.total_executions > 0
      ? Math.round((workflow.successful_executions / workflow.total_executions) * 100)
      : null;

  return (
    <Card
      className="cursor-pointer hover:border-primary/50 transition-colors"
      onClick={onEdit}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-lg">{workflow.name}</CardTitle>
            <CardDescription className="line-clamp-2">
              {workflow.description || "No description"}
            </CardDescription>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onEdit(); }}>
                <GitBranch className="mr-2 h-4 w-4" />
                Edit Workflow
              </DropdownMenuItem>
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onDuplicate(); }}>
                <Copy className="mr-2 h-4 w-4" />
                Duplicate
              </DropdownMenuItem>
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onDeploy(); }}>
                <ExternalLink className="mr-2 h-4 w-4" />
                Deploy
              </DropdownMenuItem>
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onToggleActive(); }}>
                {workflow.is_active ? (
                  <>
                    <Pause className="mr-2 h-4 w-4" />
                    Deactivate
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Activate
                  </>
                )}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={(e) => { e.stopPropagation(); onDelete(); }}
                className="text-red-600"
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <TriggerIcon className="h-4 w-4" />
            <span>{triggerLabels[workflow.trigger_type]}</span>
          </div>
          {getStatusBadge(workflow)}
          <span className="text-xs">v{workflow.version}</span>
        </div>

        <div className="mt-4 flex items-center justify-between text-sm">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <Play className="h-3.5 w-3.5 text-muted-foreground" />
              <span>{workflow.total_executions} runs</span>
            </div>
            {successRate !== null && (
              <div className="flex items-center gap-1">
                {successRate >= 80 ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                ) : successRate >= 50 ? (
                  <Clock className="h-3.5 w-3.5 text-yellow-500" />
                ) : (
                  <XCircle className="h-3.5 w-3.5 text-red-500" />
                )}
                <span>{successRate}% success</span>
              </div>
            )}
          </div>
          {workflow.last_execution_at && (
            <span className="text-xs text-muted-foreground">
              Last run {formatDistanceToNow(new Date(workflow.last_execution_at), { addSuffix: true })}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default function WorkflowsPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string | undefined>();
  const [triggerFilter, setTriggerFilter] = useState<string | undefined>();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [workflowToDelete, setWorkflowToDelete] = useState<WorkflowListItem | null>(null);
  const [deployDialogOpen, setDeployDialogOpen] = useState(false);
  const [workflowToDeploy, setWorkflowToDeploy] = useState<WorkflowListItem | null>(null);
  const [deploySlug, setDeploySlug] = useState("");
  const [deployAllowedDomains, setDeployAllowedDomains] = useState("*");
  const [deployRateLimit, setDeployRateLimit] = useState("100");
  const [deployRequireApiKey, setDeployRequireApiKey] = useState(false);
  const [deployStatus, setDeployStatus] = useState<{
    is_deployed: boolean;
    public_slug?: string;
    public_url?: string;
  } | null>(null);
  const [isDeploying, setIsDeploying] = useState(false);
  const [newWorkflowName, setNewWorkflowName] = useState("");
  const [newWorkflowDescription, setNewWorkflowDescription] = useState("");
  const [newWorkflowTrigger, setNewWorkflowTrigger] = useState<WorkflowTriggerType>("manual");

  const { data, isLoading, refetch } = useWorkflows({
    search: searchQuery || undefined,
    status: statusFilter,
    trigger_type: triggerFilter,
  });

  const createWorkflow = useCreateWorkflow();
  const deleteWorkflow = useDeleteWorkflow();
  const duplicateWorkflow = useDuplicateWorkflow();
  const publishWorkflow = usePublishWorkflow();
  const updateWorkflow = useUpdateWorkflow();

  const handleCreateWorkflow = async () => {
    if (!newWorkflowName.trim()) {
      toast.error("Please enter a workflow name");
      return;
    }

    try {
      const workflow = await createWorkflow.mutateAsync({
        name: newWorkflowName,
        description: newWorkflowDescription || undefined,
        trigger_type: newWorkflowTrigger,
        nodes: [
          {
            temp_id: "start",
            node_type: "start",
            name: "Start",
            position_x: 250,
            position_y: 50,
          },
          {
            temp_id: "end",
            node_type: "end",
            name: "End",
            position_x: 250,
            position_y: 300,
          },
        ],
        edges: [
          {
            source_node_id: "start",
            target_node_id: "end",
          },
        ],
      });

      toast.success("Workflow created");
      setCreateDialogOpen(false);
      setNewWorkflowName("");
      setNewWorkflowDescription("");
      setNewWorkflowTrigger("manual");

      router.push(`/dashboard/workflows/${workflow.id}`);
    } catch {
      toast.error("Failed to create workflow");
    }
  };

  const handleDeleteWorkflow = async () => {
    if (!workflowToDelete) return;

    try {
      await deleteWorkflow.mutateAsync(workflowToDelete.id);
      toast.success("Workflow deleted");
      setDeleteDialogOpen(false);
      setWorkflowToDelete(null);
      refetch();
    } catch {
      toast.error("Failed to delete workflow");
    }
  };

  const handleDuplicateWorkflow = async (workflow: WorkflowListItem) => {
    try {
      const duplicated = await duplicateWorkflow.mutateAsync({
        workflowId: workflow.id,
        newName: `${workflow.name} (Copy)`,
      });
      toast.success("Workflow duplicated");
      router.push(`/dashboard/workflows/${duplicated.id}`);
    } catch {
      toast.error("Failed to duplicate workflow");
    }
  };

  const handleToggleActive = async (workflow: WorkflowListItem) => {
    try {
      if (workflow.is_draft) {
        await publishWorkflow.mutateAsync(workflow.id);
        toast.success("Workflow published");
      } else {
        // Toggle active state through update
        await updateWorkflow.mutateAsync({
          workflowId: workflow.id,
          data: { is_active: !workflow.is_active },
        });
        toast.success(workflow.is_active ? "Workflow deactivated" : "Workflow activated");
      }
      refetch();
    } catch {
      toast.error("Failed to update workflow");
    }
  };

  const fetchDeployStatus = async (workflowId: string) => {
    try {
      const response = await fetch(`/api/v1/workflows/${workflowId}/deploy-status`);
      if (response.ok) {
        const status = await response.json();
        setDeployStatus(status);
        if (status.is_deployed && status.public_slug) {
          setDeploySlug(status.public_slug);
        }
      }
    } catch {
      console.error("Failed to fetch deploy status");
    }
  };

  const handleOpenDeployDialog = async (workflow: WorkflowListItem) => {
    setWorkflowToDeploy(workflow);
    setDeploySlug(workflow.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""));
    setDeployAllowedDomains("*");
    setDeployRateLimit("100");
    setDeployRequireApiKey(false);
    setDeployStatus(null);
    setDeployDialogOpen(true);
    await fetchDeployStatus(workflow.id);
  };

  const handleDeploy = async () => {
    if (!workflowToDeploy) return;

    if (!deploySlug.trim()) {
      toast.error("Please enter a public slug");
      return;
    }

    setIsDeploying(true);
    try {
      const response = await fetch(`/api/v1/workflows/${workflowToDeploy.id}/deploy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          public_slug: deploySlug,
          allowed_domains: deployAllowedDomains.split(",").map(d => d.trim()).filter(Boolean),
          rate_limit: parseInt(deployRateLimit) || 100,
          require_api_key: deployRequireApiKey,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to deploy");
      }

      const result = await response.json();
      setDeployStatus({
        is_deployed: true,
        public_slug: result.public_slug,
        public_url: result.public_url,
      });
      toast.success("Workflow deployed successfully");
      refetch();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to deploy workflow");
    } finally {
      setIsDeploying(false);
    }
  };

  const handleUndeploy = async () => {
    if (!workflowToDeploy) return;

    setIsDeploying(true);
    try {
      const response = await fetch(`/api/v1/workflows/${workflowToDeploy.id}/undeploy`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Failed to undeploy");
      }

      setDeployStatus({ is_deployed: false });
      toast.success("Workflow undeployed");
      refetch();
    } catch {
      toast.error("Failed to undeploy workflow");
    } finally {
      setIsDeploying(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Workflows</h1>
          <p className="text-muted-foreground">
            Automate tasks with visual workflow builder
          </p>
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          New Workflow
        </Button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search workflows..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              <Filter className="h-4 w-4 mr-2" />
              Status
              {statusFilter && <Badge variant="secondary" className="ml-2">{statusFilter}</Badge>}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => setStatusFilter(undefined)}>All</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setStatusFilter("active")}>Active</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setStatusFilter("inactive")}>Inactive</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setStatusFilter("draft")}>Draft</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              <Zap className="h-4 w-4 mr-2" />
              Trigger
              {triggerFilter && <Badge variant="secondary" className="ml-2">{triggerFilter}</Badge>}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => setTriggerFilter(undefined)}>All</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setTriggerFilter("manual")}>Manual</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setTriggerFilter("scheduled")}>Scheduled</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setTriggerFilter("webhook")}>Webhook</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setTriggerFilter("form")}>Form</DropdownMenuItem>
            <DropdownMenuItem onClick={() => setTriggerFilter("event")}>Event</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Workflow Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : data?.workflows && data.workflows.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {data.workflows.map((workflow) => (
            <WorkflowCard
              key={workflow.id}
              workflow={workflow}
              onEdit={() => router.push(`/dashboard/workflows/${workflow.id}`)}
              onDelete={() => {
                setWorkflowToDelete(workflow);
                setDeleteDialogOpen(true);
              }}
              onDuplicate={() => handleDuplicateWorkflow(workflow)}
              onToggleActive={() => handleToggleActive(workflow)}
              onDeploy={() => handleOpenDeployDialog(workflow)}
            />
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <GitBranch className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No workflows yet</h3>
            <p className="text-muted-foreground text-center max-w-sm mb-4">
              Create your first workflow to automate tasks like document processing,
              notifications, and more.
            </p>
            <Button onClick={() => setCreateDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Workflow
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Create Workflow Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Workflow</DialogTitle>
            <DialogDescription>
              Set up a new workflow to automate your tasks
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Name</label>
              <Input
                placeholder="My Workflow"
                value={newWorkflowName}
                onChange={(e) => setNewWorkflowName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Description (optional)</label>
              <Input
                placeholder="Describe what this workflow does..."
                value={newWorkflowDescription}
                onChange={(e) => setNewWorkflowDescription(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Trigger Type</label>
              <div className="grid grid-cols-2 gap-2">
                {(Object.keys(triggerLabels) as WorkflowTriggerType[]).map((trigger) => {
                  const Icon = triggerIcons[trigger];
                  return (
                    <Button
                      key={trigger}
                      variant={newWorkflowTrigger === trigger ? "default" : "outline"}
                      className="justify-start"
                      onClick={() => setNewWorkflowTrigger(trigger)}
                    >
                      <Icon className="h-4 w-4 mr-2" />
                      {triggerLabels[trigger]}
                    </Button>
                  );
                })}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateWorkflow} disabled={createWorkflow.isPending}>
              {createWorkflow.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Create Workflow
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Workflow</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{workflowToDelete?.name}&quot;? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteWorkflow}
              disabled={deleteWorkflow.isPending}
            >
              {deleteWorkflow.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Deploy Workflow Dialog */}
      <Dialog open={deployDialogOpen} onOpenChange={setDeployDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Deploy Workflow</DialogTitle>
            <DialogDescription>
              Make &quot;{workflowToDeploy?.name}&quot; accessible via a public URL
            </DialogDescription>
          </DialogHeader>

          {deployStatus?.is_deployed ? (
            <div className="space-y-4 py-4">
              <div className="flex items-center gap-2 text-green-600 bg-green-50 p-3 rounded-lg">
                <Check className="h-5 w-5" />
                <span className="font-medium">Workflow is deployed</span>
              </div>

              <div className="space-y-2">
                <Label>Public URL</Label>
                <div className="flex gap-2">
                  <Input
                    value={`${typeof window !== "undefined" ? window.location.origin : ""}/w/${deployStatus.public_slug}`}
                    readOnly
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      copyToClipboard(
                        `${typeof window !== "undefined" ? window.location.origin : ""}/w/${deployStatus.public_slug}`
                      )
                    }
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label>Embed Code</Label>
                <Textarea
                  value={`<iframe src="${typeof window !== "undefined" ? window.location.origin : ""}/w/${deployStatus.public_slug}" width="100%" height="600" frameborder="0"></iframe>`}
                  readOnly
                  rows={3}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    copyToClipboard(
                      `<iframe src="${typeof window !== "undefined" ? window.location.origin : ""}/w/${deployStatus.public_slug}" width="100%" height="600" frameborder="0"></iframe>`
                    )
                  }
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy Embed Code
                </Button>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label>API Endpoint</Label>
                <div className="flex gap-2">
                  <Input
                    value={`${typeof window !== "undefined" ? window.location.origin : ""}/api/v1/public/workflows/${deployStatus.public_slug}/execute`}
                    readOnly
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      copyToClipboard(
                        `${typeof window !== "undefined" ? window.location.origin : ""}/api/v1/public/workflows/${deployStatus.public_slug}/execute`
                      )
                    }
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="deploy-slug">Public Slug</Label>
                <Input
                  id="deploy-slug"
                  placeholder="my-workflow"
                  value={deploySlug}
                  onChange={(e) => setDeploySlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ""))}
                />
                <p className="text-xs text-muted-foreground">
                  URL: {typeof window !== "undefined" ? window.location.origin : ""}/w/{deploySlug || "my-workflow"}
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="deploy-domains">Allowed Domains</Label>
                <Input
                  id="deploy-domains"
                  placeholder="*, example.com"
                  value={deployAllowedDomains}
                  onChange={(e) => setDeployAllowedDomains(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Comma-separated list. Use * to allow all domains.
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="deploy-rate">Rate Limit (requests/minute)</Label>
                <Input
                  id="deploy-rate"
                  type="number"
                  value={deployRateLimit}
                  onChange={(e) => setDeployRateLimit(e.target.value)}
                />
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="deploy-apikey"
                  checked={deployRequireApiKey}
                  onChange={(e) => setDeployRequireApiKey(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300"
                />
                <Label htmlFor="deploy-apikey">Require API Key</Label>
              </div>
            </div>
          )}

          <DialogFooter>
            {deployStatus?.is_deployed ? (
              <>
                <Button variant="outline" onClick={() => setDeployDialogOpen(false)}>
                  Close
                </Button>
                <Button
                  variant="destructive"
                  onClick={handleUndeploy}
                  disabled={isDeploying}
                >
                  {isDeploying && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Undeploy
                </Button>
              </>
            ) : (
              <>
                <Button variant="outline" onClick={() => setDeployDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleDeploy} disabled={isDeploying}>
                  {isDeploying && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Deploy Workflow
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
