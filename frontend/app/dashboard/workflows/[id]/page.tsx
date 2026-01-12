"use client";

import { use, useState, useCallback, useMemo, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  MarkerType,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  ArrowLeft,
  Save,
  Play,
  Loader2,
  Settings,
  History,
  Trash2,
  Plus,
  Rocket,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useWorkflow,
  useUpdateWorkflow,
  useUpdateWorkflowNodes,
  useExecuteWorkflow,
  usePublishWorkflow,
  useWorkflowNodeTypes,
  useWorkflowExecutions,
  type WorkflowNode,
  type WorkflowEdge,
  type NodeTypeInfo,
} from "@/lib/api";
import { toast } from "sonner";
import { WorkflowNodeComponent, type WorkflowNodeData } from "@/components/workflow-builder/WorkflowNode";
import { NodeConfigPanel } from "@/components/workflow-builder/NodeConfigPanel";
import { ExecutionHistoryPanel } from "@/components/workflow-builder/ExecutionHistoryPanel";

// Custom data types for workflow edges
interface WorkflowEdgeData extends Record<string, unknown> {
  condition?: string | null;
  edgeType?: string;
}

type FlowNode = Node<WorkflowNodeData>;
type FlowEdge = Edge<WorkflowEdgeData>;

// Custom node types for React Flow
const nodeTypes = {
  workflowNode: WorkflowNodeComponent,
};

// Convert backend node to React Flow node
function toFlowNode(node: WorkflowNode): FlowNode {
  return {
    id: node.id,
    type: "workflowNode",
    position: { x: node.position_x, y: node.position_y },
    data: {
      label: node.name,
      nodeType: node.node_type,
      description: node.description,
      config: node.config,
    },
  };
}

// Convert backend edge to React Flow edge
function toFlowEdge(edge: WorkflowEdge): FlowEdge {
  return {
    id: edge.id,
    source: edge.source_node_id,
    target: edge.target_node_id,
    label: edge.label || undefined,
    type: "smoothstep",
    animated: true,
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
    data: {
      condition: edge.condition,
      edgeType: edge.edge_type,
    },
  };
}

export default function WorkflowEditorPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: workflowId } = use(params);
  const router = useRouter();

  // State
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<FlowEdge>([]);
  const [selectedNode, setSelectedNode] = useState<FlowNode | null>(null);
  const [workflowName, setWorkflowName] = useState("");
  const [workflowDescription, setWorkflowDescription] = useState("");
  const [configPanelOpen, setConfigPanelOpen] = useState(false);
  const [historyPanelOpen, setHistoryPanelOpen] = useState(false);
  const [executeDialogOpen, setExecuteDialogOpen] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [inputData, setInputData] = useState("{}");

  // Queries
  const { data: workflow, isLoading, refetch } = useWorkflow(workflowId);
  const { data: nodeTypesData } = useWorkflowNodeTypes();
  const { data: executions } = useWorkflowExecutions(workflowId);

  // Mutations
  const updateWorkflow = useUpdateWorkflow();
  const updateNodes = useUpdateWorkflowNodes();
  const executeWorkflow = useExecuteWorkflow();
  const publishWorkflow = usePublishWorkflow();

  // Initialize nodes and edges from workflow data
  useEffect(() => {
    if (workflow) {
      setWorkflowName(workflow.name);
      setWorkflowDescription(workflow.description || "");
      setNodes(workflow.nodes.map(toFlowNode));
      setEdges(workflow.edges.map(toFlowEdge));
    }
  }, [workflow, setNodes, setEdges]);

  // Track unsaved changes
  useEffect(() => {
    if (workflow) {
      const currentNodeIds = nodes.map((n) => n.id).sort().join(",");
      const workflowNodeIds = workflow.nodes.map((n) => n.id).sort().join(",");
      const currentEdgeIds = edges.map((e) => e.id).sort().join(",");
      const workflowEdgeIds = workflow.edges.map((e) => e.id).sort().join(",");

      const hasChanges =
        currentNodeIds !== workflowNodeIds ||
        currentEdgeIds !== workflowEdgeIds ||
        workflowName !== workflow.name ||
        workflowDescription !== (workflow.description || "");

      setHasUnsavedChanges(hasChanges);
    }
  }, [nodes, edges, workflowName, workflowDescription, workflow]);

  // Handle edge connection
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            type: "smoothstep",
            animated: true,
            markerEnd: { type: MarkerType.ArrowClosed },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback((_event: React.MouseEvent, node: FlowNode) => {
    setSelectedNode(node);
    setConfigPanelOpen(true);
  }, []);

  // Handle node delete
  const onNodesDelete = useCallback((deletedNodes: FlowNode[]) => {
    const deletedIds = new Set(deletedNodes.map((n) => n.id));
    setEdges((eds) =>
      eds.filter((e) => !deletedIds.has(e.source) && !deletedIds.has(e.target))
    );
    if (selectedNode && deletedIds.has(selectedNode.id)) {
      setSelectedNode(null);
      setConfigPanelOpen(false);
    }
  }, [selectedNode, setEdges]);

  // Add new node
  const addNode = useCallback(
    (nodeType: NodeTypeInfo) => {
      const newNode: FlowNode = {
        id: `node_${Date.now()}`,
        type: "workflowNode",
        position: { x: 250, y: 150 + nodes.length * 100 },
        data: {
          label: nodeType.name,
          nodeType: nodeType.type,
          description: nodeType.description,
          config: {},
        },
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [nodes.length, setNodes]
  );

  // Save workflow
  const handleSave = async () => {
    try {
      // Update metadata
      await updateWorkflow.mutateAsync({
        workflowId,
        data: {
          name: workflowName,
          description: workflowDescription || undefined,
        },
      });

      // Update nodes and edges
      await updateNodes.mutateAsync({
        workflowId,
        data: {
          nodes: nodes.map((node) => ({
            temp_id: node.id,
            node_type: node.data.nodeType,
            name: node.data.label,
            description: node.data.description ?? undefined,
            position_x: node.position.x,
            position_y: node.position.y,
            config: node.data.config ?? undefined,
          })),
          edges: edges.map((edge) => ({
            source_node_id: edge.source,
            target_node_id: edge.target,
            label: edge.label as string | undefined,
            condition: edge.data?.condition ?? undefined,
            edge_type: edge.data?.edgeType || "default",
          })),
        },
      });

      toast.success("Workflow saved");
      refetch();
    } catch {
      toast.error("Failed to save workflow");
    }
  };

  // Publish workflow
  const handlePublish = async () => {
    try {
      await handleSave();
      await publishWorkflow.mutateAsync(workflowId);
      toast.success("Workflow published");
      refetch();
    } catch {
      toast.error("Failed to publish workflow");
    }
  };

  // Execute workflow
  const handleExecute = async () => {
    try {
      let parsedInput = {};
      try {
        parsedInput = JSON.parse(inputData);
      } catch {
        toast.error("Invalid JSON input");
        return;
      }

      const execution = await executeWorkflow.mutateAsync({
        workflowId,
        data: { input_data: parsedInput },
      });

      toast.success(`Workflow execution started: ${execution.id}`);
      setExecuteDialogOpen(false);
      setHistoryPanelOpen(true);
    } catch {
      toast.error("Failed to execute workflow");
    }
  };

  // Update selected node config
  const updateNodeConfig = useCallback(
    (nodeId: string, updates: Partial<WorkflowNodeData>) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, ...updates } }
            : node
        )
      );
    },
    [setNodes]
  );

  // Node type categories
  const nodeCategories = useMemo(() => {
    if (!nodeTypesData) return {};
    return nodeTypesData.node_types.reduce(
      (acc, nodeType) => {
        const category = nodeType.category || "other";
        if (!acc[category]) acc[category] = [];
        acc[category].push(nodeType);
        return acc;
      },
      {} as Record<string, NodeTypeInfo[]>
    );
  }, [nodeTypesData]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!workflow) {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-8rem)]">
        <h2 className="text-xl font-semibold mb-2">Workflow not found</h2>
        <Button onClick={() => router.push("/dashboard/workflows")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Workflows
        </Button>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.push("/dashboard/workflows")}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <Input
              value={workflowName}
              onChange={(e) => setWorkflowName(e.target.value)}
              className="text-lg font-semibold border-none p-0 h-auto focus-visible:ring-0"
              placeholder="Workflow Name"
            />
            <Input
              value={workflowDescription}
              onChange={(e) => setWorkflowDescription(e.target.value)}
              className="text-sm text-muted-foreground border-none p-0 h-auto focus-visible:ring-0"
              placeholder="Add description..."
            />
          </div>
          {workflow.is_draft && <Badge variant="outline">Draft</Badge>}
          {workflow.is_active && (
            <Badge className="bg-green-500/10 text-green-600 border-green-200">Active</Badge>
          )}
          {hasUnsavedChanges && (
            <Badge variant="secondary" className="text-yellow-600">
              Unsaved changes
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => setHistoryPanelOpen(true)}>
            <History className="h-4 w-4 mr-2" />
            History
            {executions?.total ? (
              <Badge variant="secondary" className="ml-2">
                {executions.total}
              </Badge>
            ) : null}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleSave}
            disabled={updateWorkflow.isPending || updateNodes.isPending}
          >
            {(updateWorkflow.isPending || updateNodes.isPending) ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save
          </Button>
          {workflow.is_draft ? (
            <Button size="sm" onClick={handlePublish} disabled={publishWorkflow.isPending}>
              {publishWorkflow.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Rocket className="h-4 w-4 mr-2" />
              )}
              Publish
            </Button>
          ) : (
            <Button size="sm" onClick={() => setExecuteDialogOpen(true)}>
              <Play className="h-4 w-4 mr-2" />
              Execute
            </Button>
          )}
        </div>
      </div>

      {/* Workflow Canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onNodesDelete={onNodesDelete}
          nodeTypes={nodeTypes}
          fitView
          snapToGrid
          snapGrid={[15, 15]}
          defaultEdgeOptions={{
            type: "smoothstep",
            animated: true,
            markerEnd: { type: MarkerType.ArrowClosed },
          }}
        >
          <Background variant={BackgroundVariant.Dots} gap={15} size={1} />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              const nodeType = (node.data?.nodeType as string)?.toUpperCase();
              switch (nodeType) {
                case "START":
                  return "#22c55e";
                case "END":
                  return "#ef4444";
                case "CONDITION":
                  return "#f59e0b";
                case "AGENT":
                  return "#8b5cf6";
                default:
                  return "#3b82f6";
              }
            }}
          />

          {/* Add Node Panel */}
          <Panel position="top-left" className="bg-card border rounded-lg p-3 shadow-lg">
            <div className="text-sm font-medium mb-2">Add Node</div>
            <Tabs defaultValue="control" className="w-48">
              <TabsList className="grid w-full grid-cols-3 h-8">
                <TabsTrigger value="control" className="text-xs">Control</TabsTrigger>
                <TabsTrigger value="action" className="text-xs">Action</TabsTrigger>
                <TabsTrigger value="ai" className="text-xs">AI</TabsTrigger>
              </TabsList>
              {Object.entries(nodeCategories).map(([category, types]) => (
                <TabsContent key={category} value={category} className="space-y-1 mt-2">
                  {types.map((nodeType) => (
                    <Button
                      key={nodeType.type}
                      variant="ghost"
                      size="sm"
                      className="w-full justify-start text-xs h-7"
                      onClick={() => addNode(nodeType)}
                    >
                      <Plus className="h-3 w-3 mr-2" />
                      {nodeType.name}
                    </Button>
                  ))}
                </TabsContent>
              ))}
            </Tabs>
          </Panel>
        </ReactFlow>
      </div>

      {/* Node Configuration Panel */}
      <Sheet open={configPanelOpen} onOpenChange={setConfigPanelOpen}>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>
              {selectedNode?.data?.label || "Node Configuration"}
            </SheetTitle>
            <SheetDescription>
              Configure this node&apos;s settings
            </SheetDescription>
          </SheetHeader>
          {selectedNode && (
            <NodeConfigPanel
              node={selectedNode}
              nodeTypes={nodeTypesData?.node_types || []}
              onUpdate={(updates) => updateNodeConfig(selectedNode.id, updates)}
              onDelete={() => {
                setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
                setConfigPanelOpen(false);
                setSelectedNode(null);
              }}
            />
          )}
        </SheetContent>
      </Sheet>

      {/* Execution History Panel */}
      <Sheet open={historyPanelOpen} onOpenChange={setHistoryPanelOpen}>
        <SheetContent className="w-[500px] sm:max-w-[500px]">
          <SheetHeader>
            <SheetTitle>Execution History</SheetTitle>
            <SheetDescription>
              View past workflow executions
            </SheetDescription>
          </SheetHeader>
          <ExecutionHistoryPanel
            workflowId={workflowId}
            executions={executions?.executions || []}
          />
        </SheetContent>
      </Sheet>

      {/* Execute Dialog */}
      <Dialog open={executeDialogOpen} onOpenChange={setExecuteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Execute Workflow</DialogTitle>
            <DialogDescription>
              Provide input data for this workflow execution
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Input Data (JSON)</label>
              <textarea
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                className="w-full h-32 p-3 rounded-md border bg-background font-mono text-sm"
                placeholder='{"key": "value"}'
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setExecuteDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleExecute} disabled={executeWorkflow.isPending}>
              {executeWorkflow.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Execute
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
