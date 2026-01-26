"use client";

import * as React from "react";
import { useState, useCallback, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
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
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Play,
  Square,
  Circle,
  Diamond,
  RotateCw,
  Code,
  Clock,
  Globe,
  Bell,
  Bot,
  UserCheck,
  Plus,
  Trash2,
  Save,
  Settings,
  ZoomIn,
  ZoomOut,
  Maximize2,
  GripVertical,
  ArrowRight,
  MoreVertical,
  Copy,
  Mic,
  MessageSquare,
  Wand2,
  Loader2,
  Check,
  Layers,
} from "lucide-react";

// =============================================================================
// Types
// =============================================================================

interface Position {
  x: number;
  y: number;
}

interface WorkflowNode {
  id: string;
  type: string;
  name: string;
  description?: string;
  position: Position;
  config: Record<string, any>;
}

interface WorkflowEdge {
  id: string;
  sourceNodeId: string;
  targetNodeId: string;
  label?: string;
  condition?: string;
  edgeType: string;
}

interface Workflow {
  id?: string;
  name: string;
  description?: string;
  triggerType: string;
  triggerConfig: Record<string, any>;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  isActive: boolean;
  isDraft: boolean;
}

interface NodeType {
  type: string;
  name: string;
  description: string;
  category: string;
  icon: React.ReactNode;
  color: string;
  configSchema: Record<string, any>;
  maxInputs: number;
  maxOutputs: number;
}

// =============================================================================
// Node Type Definitions
// =============================================================================

const NODE_TYPES: NodeType[] = [
  {
    type: "start",
    name: "Start",
    description: "Entry point of the workflow",
    category: "control",
    icon: <Play className="h-4 w-4" />,
    color: "bg-green-500",
    configSchema: {},
    maxInputs: 0,
    maxOutputs: 1,
  },
  {
    type: "end",
    name: "End",
    description: "Exit point of the workflow",
    category: "control",
    icon: <Square className="h-4 w-4" />,
    color: "bg-red-500",
    configSchema: {},
    maxInputs: -1,
    maxOutputs: 0,
  },
  {
    type: "action",
    name: "Action",
    description: "Perform a predefined action",
    category: "action",
    icon: <Circle className="h-4 w-4" />,
    color: "bg-blue-500",
    configSchema: { action_type: { type: "string" }, params: { type: "object" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "condition",
    name: "Condition",
    description: "Branch based on a condition",
    category: "control",
    icon: <Diamond className="h-4 w-4" />,
    color: "bg-yellow-500",
    configSchema: { expression: { type: "string" } },
    maxInputs: 1,
    maxOutputs: 2,
  },
  {
    type: "loop",
    name: "Loop",
    description: "Iterate over an array",
    category: "control",
    icon: <RotateCw className="h-4 w-4" />,
    color: "bg-purple-500",
    configSchema: { array_path: { type: "string" }, item_var: { type: "string" } },
    maxInputs: 1,
    maxOutputs: 2,
  },
  {
    type: "code",
    name: "Code",
    description: "Execute custom code",
    category: "action",
    icon: <Code className="h-4 w-4" />,
    color: "bg-gray-600",
    configSchema: { language: { type: "string" }, code: { type: "string" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "delay",
    name: "Delay",
    description: "Wait for a specified time",
    category: "control",
    icon: <Clock className="h-4 w-4" />,
    color: "bg-orange-500",
    configSchema: { seconds: { type: "integer" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "http",
    name: "HTTP Request",
    description: "Make an HTTP request",
    category: "action",
    icon: <Globe className="h-4 w-4" />,
    color: "bg-cyan-500",
    configSchema: { url: { type: "string" }, method: { type: "string" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "notification",
    name: "Notification",
    description: "Send a notification",
    category: "action",
    icon: <Bell className="h-4 w-4" />,
    color: "bg-pink-500",
    configSchema: { channel: { type: "string" }, template: { type: "string" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "agent",
    name: "AI Agent",
    description: "Execute an AI agent task",
    category: "ai",
    icon: <Bot className="h-4 w-4" />,
    color: "bg-indigo-500",
    configSchema: { agent_id: { type: "string" }, prompt: { type: "string" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "voice_agent",
    name: "Voice Agent",
    description: "Execute a voice-enabled AI agent",
    category: "ai",
    icon: <Mic className="h-4 w-4" />,
    color: "bg-violet-500",
    configSchema: { agent_id: { type: "string" }, tts_provider: { type: "string" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "chat_agent",
    name: "Chat Agent",
    description: "Execute a chat AI agent",
    category: "ai",
    icon: <MessageSquare className="h-4 w-4" />,
    color: "bg-emerald-500",
    configSchema: { agent_id: { type: "string" }, knowledge_bases: { type: "array" } },
    maxInputs: 1,
    maxOutputs: 1,
  },
  {
    type: "human_approval",
    name: "Human Approval",
    description: "Wait for human approval",
    category: "control",
    icon: <UserCheck className="h-4 w-4" />,
    color: "bg-amber-500",
    configSchema: { approvers: { type: "array" }, timeout_hours: { type: "integer" } },
    maxInputs: 1,
    maxOutputs: 2,
  },
];

const TRIGGER_TYPES = [
  { type: "manual", name: "Manual", description: "Triggered manually by user" },
  { type: "scheduled", name: "Scheduled", description: "Triggered on a schedule (cron)" },
  { type: "webhook", name: "Webhook", description: "Triggered by external webhook" },
  { type: "form", name: "Form", description: "Triggered by form submission" },
  { type: "event", name: "Event", description: "Triggered by system event" },
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

interface WorkflowTemplateItem {
  id: string;
  name: string;
  description: string;
  type: string;
  version: string;
  category?: string;
  complexity?: string;
  patterns_used?: string[];
}

// =============================================================================
// Node Palette Component
// =============================================================================

interface NodePaletteProps {
  onDragStart: (nodeType: NodeType, e: React.DragEvent) => void;
}

function NodePalette({ onDragStart }: NodePaletteProps) {
  const categories = [
    { id: "control", name: "Control Flow" },
    { id: "action", name: "Actions" },
    { id: "ai", name: "AI Agents" },
  ];

  return (
    <div className="w-64 border-r bg-muted/30 p-4 overflow-y-auto">
      <h3 className="font-semibold mb-4">Node Types</h3>
      {categories.map((category) => (
        <div key={category.id} className="mb-4">
          <h4 className="text-sm font-medium text-muted-foreground mb-2">
            {category.name}
          </h4>
          <div className="space-y-2">
            {NODE_TYPES.filter((n) => n.category === category.id).map((nodeType) => (
              <div
                key={nodeType.type}
                draggable
                onDragStart={(e) => onDragStart(nodeType, e)}
                className="flex items-center gap-2 p-2 rounded-md border bg-background cursor-grab hover:border-primary transition-colors"
              >
                <div className={cn("p-1.5 rounded text-white", nodeType.color)}>
                  {nodeType.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">{nodeType.name}</div>
                  <div className="text-xs text-muted-foreground truncate">
                    {nodeType.description}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// =============================================================================
// Canvas Node Component
// =============================================================================

interface CanvasNodeProps {
  node: WorkflowNode;
  nodeType: NodeType | undefined;
  isSelected: boolean;
  isConnecting: boolean;
  onSelect: () => void;
  onDragStart: (e: React.MouseEvent) => void;
  onConnectionStart: (nodeId: string, isOutput: boolean) => void;
  onConnectionEnd: (nodeId: string) => void;
  onDelete: () => void;
  onConfigure: () => void;
}

function CanvasNode({
  node,
  nodeType,
  isSelected,
  isConnecting,
  onSelect,
  onDragStart,
  onConnectionStart,
  onConnectionEnd,
  onDelete,
  onConfigure,
}: CanvasNodeProps) {
  if (!nodeType) return null;

  return (
    <div
      className={cn(
        "absolute bg-background border-2 rounded-lg shadow-md min-w-[160px] cursor-move transition-all",
        isSelected ? "border-primary ring-2 ring-primary/20" : "border-border",
        isConnecting && "ring-2 ring-blue-500/50"
      )}
      style={{
        left: node.position.x,
        top: node.position.y,
        transform: "translate(-50%, -50%)",
      }}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      onMouseDown={onDragStart}
    >
      {/* Input connector */}
      {nodeType.maxInputs !== 0 && (
        <div
          className="absolute -left-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-blue-500 border-2 border-background cursor-crosshair flex items-center justify-center hover:scale-110 transition-transform"
          onClick={(e) => {
            e.stopPropagation();
            onConnectionEnd(node.id);
          }}
        >
          <div className="w-2 h-2 rounded-full bg-white" />
        </div>
      )}

      {/* Node content */}
      <div className="p-3">
        <div className="flex items-center gap-2 mb-1">
          <div className={cn("p-1.5 rounded text-white", nodeType.color)}>
            {nodeType.icon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium truncate">{node.name}</div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-6 w-6">
                <MoreVertical className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onConfigure}>
                <Settings className="h-4 w-4 mr-2" />
                Configure
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Copy className="h-4 w-4 mr-2" />
                Duplicate
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onDelete} className="text-destructive">
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        {node.description && (
          <div className="text-xs text-muted-foreground truncate">{node.description}</div>
        )}
      </div>

      {/* Output connector */}
      {nodeType.maxOutputs !== 0 && (
        <div
          className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-green-500 border-2 border-background cursor-crosshair flex items-center justify-center hover:scale-110 transition-transform"
          onClick={(e) => {
            e.stopPropagation();
            onConnectionStart(node.id, true);
          }}
        >
          <ArrowRight className="w-3 h-3 text-white" />
        </div>
      )}

      {/* Multiple outputs for condition/loop nodes */}
      {nodeType.maxOutputs === 2 && (
        <div
          className="absolute -right-3 top-3/4 w-6 h-6 rounded-full bg-red-500 border-2 border-background cursor-crosshair flex items-center justify-center hover:scale-110 transition-transform"
          onClick={(e) => {
            e.stopPropagation();
            onConnectionStart(node.id, true);
          }}
        >
          <ArrowRight className="w-3 h-3 text-white" />
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Edge Component (SVG)
// =============================================================================

interface CanvasEdgeProps {
  edge: WorkflowEdge;
  sourceNode: WorkflowNode | undefined;
  targetNode: WorkflowNode | undefined;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

function CanvasEdge({
  edge,
  sourceNode,
  targetNode,
  isSelected,
  onSelect,
  onDelete,
}: CanvasEdgeProps) {
  if (!sourceNode || !targetNode) return null;

  const startX = sourceNode.position.x + 80;
  const startY = sourceNode.position.y;
  const endX = targetNode.position.x - 80;
  const endY = targetNode.position.y;

  // Bezier curve control points
  const midX = (startX + endX) / 2;
  const path = `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`;

  return (
    <g onClick={onSelect} className="cursor-pointer">
      {/* Invisible wider path for easier clicking */}
      <path
        d={path}
        fill="none"
        stroke="transparent"
        strokeWidth={20}
      />
      {/* Visible path */}
      <path
        d={path}
        fill="none"
        stroke={isSelected ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))"}
        strokeWidth={2}
        strokeDasharray={edge.condition ? "5,5" : undefined}
        className="transition-colors"
      />
      {/* Arrow head */}
      <polygon
        points={`${endX},${endY} ${endX - 8},${endY - 5} ${endX - 8},${endY + 5}`}
        fill={isSelected ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))"}
      />
      {/* Edge label */}
      {edge.label && (
        <text
          x={midX}
          y={(startY + endY) / 2 - 10}
          textAnchor="middle"
          className="text-xs fill-muted-foreground"
        >
          {edge.label}
        </text>
      )}
    </g>
  );
}

// =============================================================================
// Node Configuration Sheet
// =============================================================================

interface NodeConfigSheetProps {
  node: WorkflowNode | null;
  nodeType: NodeType | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (node: WorkflowNode) => void;
}

function NodeConfigSheet({
  node,
  nodeType,
  open,
  onOpenChange,
  onSave,
}: NodeConfigSheetProps) {
  const [editedNode, setEditedNode] = useState<WorkflowNode | null>(null);

  useEffect(() => {
    if (node) {
      setEditedNode({ ...node, config: { ...node.config } });
    }
  }, [node]);

  if (!editedNode || !nodeType) return null;

  const handleSave = () => {
    onSave(editedNode);
    onOpenChange(false);
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-[400px] sm:w-[540px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <div className={cn("p-1.5 rounded text-white", nodeType.color)}>
              {nodeType.icon}
            </div>
            Configure {nodeType.name}
          </SheetTitle>
          <SheetDescription>{nodeType.description}</SheetDescription>
        </SheetHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="node-name">Name</Label>
            <Input
              id="node-name"
              value={editedNode.name}
              onChange={(e) =>
                setEditedNode({ ...editedNode, name: e.target.value })
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="node-description">Description</Label>
            <Textarea
              id="node-description"
              value={editedNode.description || ""}
              onChange={(e) =>
                setEditedNode({ ...editedNode, description: e.target.value })
              }
              rows={2}
            />
          </div>

          {/* Node-specific configuration */}
          {nodeType.type === "agent" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="agent-id">Agent ID</Label>
                <Input
                  id="agent-id"
                  placeholder="Enter agent ID or select from list"
                  value={editedNode.config.agent_id || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, agent_id: e.target.value },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="agent-prompt">Prompt Template</Label>
                <Textarea
                  id="agent-prompt"
                  placeholder="Enter prompt template with {{variables}}"
                  value={editedNode.config.prompt || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, prompt: e.target.value },
                    })
                  }
                  rows={4}
                />
              </div>
            </>
          )}

          {nodeType.type === "voice_agent" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="voice-agent-id">Voice Agent ID</Label>
                <Input
                  id="voice-agent-id"
                  placeholder="Enter voice agent ID"
                  value={editedNode.config.agent_id || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, agent_id: e.target.value },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="tts-provider">TTS Provider</Label>
                <Select
                  value={editedNode.config.tts_provider || "cartesia"}
                  onValueChange={(value) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, tts_provider: value },
                    })
                  }
                >
                  <SelectTrigger id="tts-provider">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cartesia">Cartesia Sonic</SelectItem>
                    <SelectItem value="elevenlabs">ElevenLabs</SelectItem>
                    <SelectItem value="openai">OpenAI TTS</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </>
          )}

          {nodeType.type === "chat_agent" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="chat-agent-id">Chat Agent ID</Label>
                <Input
                  id="chat-agent-id"
                  placeholder="Enter chat agent ID"
                  value={editedNode.config.agent_id || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, agent_id: e.target.value },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="knowledge-bases">Knowledge Bases (comma-separated)</Label>
                <Input
                  id="knowledge-bases"
                  placeholder="kb-1, kb-2"
                  value={(editedNode.config.knowledge_bases || []).join(", ")}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: {
                        ...editedNode.config,
                        knowledge_bases: e.target.value.split(",").map((s) => s.trim()).filter(Boolean),
                      },
                    })
                  }
                />
              </div>
            </>
          )}

          {nodeType.type === "condition" && (
            <div className="space-y-2">
              <Label htmlFor="condition-expression">Condition Expression</Label>
              <Textarea
                id="condition-expression"
                placeholder="e.g., input.value > 10"
                value={editedNode.config.expression || ""}
                onChange={(e) =>
                  setEditedNode({
                    ...editedNode,
                    config: { ...editedNode.config, expression: e.target.value },
                  })
                }
                rows={3}
              />
              <p className="text-xs text-muted-foreground">
                Use JavaScript-like expressions. Available variables: input, context
              </p>
            </div>
          )}

          {nodeType.type === "http" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="http-url">URL</Label>
                <Input
                  id="http-url"
                  placeholder="https://api.example.com/endpoint"
                  value={editedNode.config.url || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, url: e.target.value },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="http-method">Method</Label>
                <Select
                  value={editedNode.config.method || "GET"}
                  onValueChange={(value) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, method: value },
                    })
                  }
                >
                  <SelectTrigger id="http-method">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="GET">GET</SelectItem>
                    <SelectItem value="POST">POST</SelectItem>
                    <SelectItem value="PUT">PUT</SelectItem>
                    <SelectItem value="DELETE">DELETE</SelectItem>
                    <SelectItem value="PATCH">PATCH</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="http-body">Request Body (JSON)</Label>
                <Textarea
                  id="http-body"
                  placeholder='{"key": "value"}'
                  value={editedNode.config.body || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, body: e.target.value },
                    })
                  }
                  rows={4}
                />
              </div>
            </>
          )}

          {nodeType.type === "delay" && (
            <div className="space-y-2">
              <Label htmlFor="delay-seconds">Delay (seconds)</Label>
              <Input
                id="delay-seconds"
                type="number"
                min={1}
                value={editedNode.config.seconds || 60}
                onChange={(e) =>
                  setEditedNode({
                    ...editedNode,
                    config: { ...editedNode.config, seconds: parseInt(e.target.value) || 60 },
                  })
                }
              />
            </div>
          )}

          {nodeType.type === "code" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="code-language">Language</Label>
                <Select
                  value={editedNode.config.language || "javascript"}
                  onValueChange={(value) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, language: value },
                    })
                  }
                >
                  <SelectTrigger id="code-language">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="javascript">JavaScript</SelectItem>
                    <SelectItem value="python">Python</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="code-content">Code</Label>
                <Textarea
                  id="code-content"
                  className="font-mono text-sm"
                  placeholder="// Your code here"
                  value={editedNode.config.code || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, code: e.target.value },
                    })
                  }
                  rows={10}
                />
              </div>
            </>
          )}

          {nodeType.type === "notification" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="notification-channel">Channel</Label>
                <Select
                  value={editedNode.config.channel || "email"}
                  onValueChange={(value) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, channel: value },
                    })
                  }
                >
                  <SelectTrigger id="notification-channel">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="email">Email</SelectItem>
                    <SelectItem value="slack">Slack</SelectItem>
                    <SelectItem value="webhook">Webhook</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="notification-template">Message Template</Label>
                <Textarea
                  id="notification-template"
                  placeholder="Hello {{name}}, your workflow completed."
                  value={editedNode.config.template || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, template: e.target.value },
                    })
                  }
                  rows={4}
                />
              </div>
            </>
          )}

          {nodeType.type === "human_approval" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="approval-message">Approval Message</Label>
                <Textarea
                  id="approval-message"
                  placeholder="Please review and approve this action."
                  value={editedNode.config.message || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, message: e.target.value },
                    })
                  }
                  rows={3}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="approval-timeout">Timeout (hours)</Label>
                <Input
                  id="approval-timeout"
                  type="number"
                  min={1}
                  value={editedNode.config.timeout_hours || 24}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, timeout_hours: parseInt(e.target.value) || 24 },
                    })
                  }
                />
              </div>
            </>
          )}

          {nodeType.type === "loop" && (
            <>
              <div className="space-y-2">
                <Label htmlFor="loop-array">Array Path</Label>
                <Input
                  id="loop-array"
                  placeholder="input.items"
                  value={editedNode.config.array_path || ""}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, array_path: e.target.value },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="loop-item-var">Item Variable Name</Label>
                <Input
                  id="loop-item-var"
                  placeholder="item"
                  value={editedNode.config.item_var || "item"}
                  onChange={(e) =>
                    setEditedNode({
                      ...editedNode,
                      config: { ...editedNode.config, item_var: e.target.value },
                    })
                  }
                />
              </div>
            </>
          )}
        </div>
        <div className="flex justify-end gap-2 pt-4 border-t">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save Changes</Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}

// =============================================================================
// Workflow Settings Dialog
// =============================================================================

interface WorkflowSettingsDialogProps {
  workflow: Workflow;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (workflow: Workflow) => void;
}

function WorkflowSettingsDialog({
  workflow,
  open,
  onOpenChange,
  onSave,
}: WorkflowSettingsDialogProps) {
  const [editedWorkflow, setEditedWorkflow] = useState<Workflow>(workflow);

  useEffect(() => {
    setEditedWorkflow(workflow);
  }, [workflow]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Workflow Settings</DialogTitle>
          <DialogDescription>
            Configure the workflow name, description, and trigger settings.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="workflow-name">Name</Label>
            <Input
              id="workflow-name"
              value={editedWorkflow.name}
              onChange={(e) =>
                setEditedWorkflow({ ...editedWorkflow, name: e.target.value })
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="workflow-description">Description</Label>
            <Textarea
              id="workflow-description"
              value={editedWorkflow.description || ""}
              onChange={(e) =>
                setEditedWorkflow({ ...editedWorkflow, description: e.target.value })
              }
              rows={3}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="workflow-trigger">Trigger Type</Label>
            <Select
              value={editedWorkflow.triggerType}
              onValueChange={(value) =>
                setEditedWorkflow({ ...editedWorkflow, triggerType: value })
              }
            >
              <SelectTrigger id="workflow-trigger">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TRIGGER_TYPES.map((trigger) => (
                  <SelectItem key={trigger.type} value={trigger.type}>
                    {trigger.name} - {trigger.description}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {editedWorkflow.triggerType === "scheduled" && (
            <div className="space-y-2">
              <Label htmlFor="cron-expression">Cron Expression</Label>
              <Input
                id="cron-expression"
                placeholder="0 9 * * * (daily at 9am)"
                value={editedWorkflow.triggerConfig.cron || ""}
                onChange={(e) =>
                  setEditedWorkflow({
                    ...editedWorkflow,
                    triggerConfig: { ...editedWorkflow.triggerConfig, cron: e.target.value },
                  })
                }
              />
              <p className="text-xs text-muted-foreground">
                Standard cron format: minute hour day-of-month month day-of-week
              </p>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              onSave(editedWorkflow);
              onOpenChange(false);
            }}
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// Main Workflow Designer Component
// =============================================================================

interface WorkflowDesignerProps {
  className?: string;
  initialWorkflow?: Workflow;
  onSave?: (workflow: Workflow) => Promise<void>;
  onCancel?: () => void;
}

export function WorkflowDesigner({
  className,
  initialWorkflow,
  onSave,
  onCancel,
}: WorkflowDesignerProps) {
  // Workflow state
  const [workflow, setWorkflow] = useState<Workflow>(
    initialWorkflow || {
      name: "New Workflow",
      description: "",
      triggerType: "manual",
      triggerConfig: {},
      nodes: [],
      edges: [],
      isActive: false,
      isDraft: true,
    }
  );

  // Selection state
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  // Connection state
  const [connectingFromNode, setConnectingFromNode] = useState<string | null>(null);

  // UI state
  const [configNodeId, setConfigNodeId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [templateSelectorOpen, setTemplateSelectorOpen] = useState(!initialWorkflow);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [saving, setSaving] = useState(false);

  // Template state
  const [templates, setTemplates] = useState<WorkflowTemplateItem[]>([]);
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);

  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);
  const draggedNodeRef = useRef<{ nodeId: string; startPos: Position; mouseStart: Position } | null>(null);

  // Load templates
  useEffect(() => {
    async function loadTemplates() {
      setLoadingTemplates(true);
      try {
        const response = await fetch(`${API_BASE}/v1/agent-templates/workflows`, {
          credentials: "include",
        });
        if (response.ok) {
          const data = await response.json();
          setTemplates(data || []);
        }
      } catch (err) {
        console.error("Failed to load workflow templates:", err);
      } finally {
        setLoadingTemplates(false);
      }
    }
    loadTemplates();
  }, []);

  // Load workflow from template
  const loadFromTemplate = async (templateId: string) => {
    setLoadingTemplates(true);
    try {
      const response = await fetch(`${API_BASE}/v1/agent-templates/workflows/${templateId}`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error("Failed to load template");

      const templateData = await response.json();

      // Map template to workflow format
      const newWorkflow: Workflow = {
        name: templateData.name || "New Workflow",
        description: templateData.description || "",
        triggerType: (templateData.trigger || "manual").toLowerCase(),
        triggerConfig: templateData.config || {},
        nodes: (templateData.nodes || []).map((n: any, index: number) => ({
          id: n.id || `node_${index}`,
          type: (n.type || "action").toLowerCase(),
          name: n.name || n.type,
          description: n.description || "",
          position: n.position || { x: 200 + index * 200, y: 200 },
          config: n.config || {},
        })),
        edges: (templateData.edges || []).map((e: any, index: number) => ({
          id: e.id || `edge_${index}`,
          sourceNodeId: e.from || e.sourceNodeId,
          targetNodeId: e.to || e.targetNodeId,
          label: e.label || e.condition,
          condition: e.condition,
          edgeType: "default",
        })),
        isActive: false,
        isDraft: true,
      };

      setWorkflow(newWorkflow);
      setSelectedTemplate(templateId);
      setTemplateSelectorOpen(false);
    } catch (err) {
      console.error("Failed to load template:", err);
    } finally {
      setLoadingTemplates(false);
    }
  };

  // Get node type helper
  const getNodeType = useCallback((type: string) => {
    return NODE_TYPES.find((nt) => nt.type === type);
  }, []);

  // Generate unique ID
  const generateId = useCallback(() => {
    return `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Handle drag start from palette
  const handlePaletteDragStart = useCallback(
    (nodeType: NodeType, e: React.DragEvent) => {
      e.dataTransfer.setData("nodeType", nodeType.type);
      e.dataTransfer.effectAllowed = "copy";
    },
    []
  );

  // Handle drop on canvas
  const handleCanvasDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const nodeTypeStr = e.dataTransfer.getData("nodeType");
      if (!nodeTypeStr || !canvasRef.current) return;

      const nodeType = getNodeType(nodeTypeStr);
      if (!nodeType) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left - pan.x) / zoom;
      const y = (e.clientY - rect.top - pan.y) / zoom;

      const newNode: WorkflowNode = {
        id: generateId(),
        type: nodeType.type,
        name: nodeType.name,
        description: "",
        position: { x, y },
        config: {},
      };

      setWorkflow((prev) => ({
        ...prev,
        nodes: [...prev.nodes, newNode],
      }));
    },
    [generateId, getNodeType, pan, zoom]
  );

  // Handle node drag within canvas
  const handleNodeDragStart = useCallback(
    (nodeId: string, e: React.MouseEvent) => {
      if (e.button !== 0) return;
      e.preventDefault();

      const node = workflow.nodes.find((n) => n.id === nodeId);
      if (!node) return;

      draggedNodeRef.current = {
        nodeId,
        startPos: { ...node.position },
        mouseStart: { x: e.clientX, y: e.clientY },
      };

      const handleMouseMove = (moveEvent: MouseEvent) => {
        if (!draggedNodeRef.current) return;

        const dx = (moveEvent.clientX - draggedNodeRef.current.mouseStart.x) / zoom;
        const dy = (moveEvent.clientY - draggedNodeRef.current.mouseStart.y) / zoom;

        setWorkflow((prev) => ({
          ...prev,
          nodes: prev.nodes.map((n) =>
            n.id === nodeId
              ? {
                  ...n,
                  position: {
                    x: draggedNodeRef.current!.startPos.x + dx,
                    y: draggedNodeRef.current!.startPos.y + dy,
                  },
                }
              : n
          ),
        }));
      };

      const handleMouseUp = () => {
        draggedNodeRef.current = null;
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [workflow.nodes, zoom]
  );

  // Handle connection start
  const handleConnectionStart = useCallback((nodeId: string) => {
    setConnectingFromNode(nodeId);
  }, []);

  // Handle connection end
  const handleConnectionEnd = useCallback(
    (targetNodeId: string) => {
      if (!connectingFromNode || connectingFromNode === targetNodeId) {
        setConnectingFromNode(null);
        return;
      }

      // Check if edge already exists
      const exists = workflow.edges.some(
        (e) => e.sourceNodeId === connectingFromNode && e.targetNodeId === targetNodeId
      );

      if (!exists) {
        const newEdge: WorkflowEdge = {
          id: `edge_${Date.now()}`,
          sourceNodeId: connectingFromNode,
          targetNodeId: targetNodeId,
          edgeType: "default",
        };

        setWorkflow((prev) => ({
          ...prev,
          edges: [...prev.edges, newEdge],
        }));
      }

      setConnectingFromNode(null);
    },
    [connectingFromNode, workflow.edges]
  );

  // Delete node
  const handleDeleteNode = useCallback((nodeId: string) => {
    setWorkflow((prev) => ({
      ...prev,
      nodes: prev.nodes.filter((n) => n.id !== nodeId),
      edges: prev.edges.filter(
        (e) => e.sourceNodeId !== nodeId && e.targetNodeId !== nodeId
      ),
    }));
    setSelectedNodeId(null);
  }, []);

  // Delete edge
  const handleDeleteEdge = useCallback((edgeId: string) => {
    setWorkflow((prev) => ({
      ...prev,
      edges: prev.edges.filter((e) => e.id !== edgeId),
    }));
    setSelectedEdgeId(null);
  }, []);

  // Update node
  const handleUpdateNode = useCallback((updatedNode: WorkflowNode) => {
    setWorkflow((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) => (n.id === updatedNode.id ? updatedNode : n)),
    }));
  }, []);

  // Save workflow
  const handleSave = useCallback(async () => {
    if (!onSave) return;

    setSaving(true);
    try {
      await onSave(workflow);
    } finally {
      setSaving(false);
    }
  }, [onSave, workflow]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedNodeId) {
          handleDeleteNode(selectedNodeId);
        } else if (selectedEdgeId) {
          handleDeleteEdge(selectedEdgeId);
        }
      } else if (e.key === "Escape") {
        setSelectedNodeId(null);
        setSelectedEdgeId(null);
        setConnectingFromNode(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedNodeId, selectedEdgeId, handleDeleteNode, handleDeleteEdge]);

  // Selected node for config
  const configNode = configNodeId
    ? workflow.nodes.find((n) => n.id === configNodeId)
    : null;
  const configNodeType = configNode ? getNodeType(configNode.type) : undefined;

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Template Selector Dialog */}
      <Dialog open={templateSelectorOpen} onOpenChange={setTemplateSelectorOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Wand2 className="h-5 w-5" />
              Start from Template
            </DialogTitle>
            <DialogDescription>
              Choose a pre-built workflow template or start from scratch
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 py-4">
            {/* Start Fresh Option */}
            <button
              type="button"
              onClick={() => {
                setSelectedTemplate(null);
                setTemplateSelectorOpen(false);
              }}
              className={cn(
                "p-4 rounded-lg border-2 text-left transition-all",
                !selectedTemplate
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              )}
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gray-100">
                  <Plus className="h-5 w-5 text-gray-600" />
                </div>
                <div>
                  <div className="font-medium">Start Fresh</div>
                  <div className="text-xs text-muted-foreground">
                    Build your workflow from scratch
                  </div>
                </div>
              </div>
            </button>

            {/* Templates */}
            {templates.map((template) => (
              <button
                key={template.id}
                type="button"
                onClick={() => loadFromTemplate(template.id)}
                disabled={loadingTemplates}
                className={cn(
                  "p-4 rounded-lg border-2 text-left transition-all",
                  selectedTemplate === template.id
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                )}
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-indigo-100">
                    <Layers className="h-5 w-5 text-indigo-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{template.name}</div>
                    <div className="text-xs text-muted-foreground line-clamp-2">
                      {template.description}
                    </div>
                  </div>
                  {selectedTemplate === template.id && (
                    <Check className="h-5 w-5 text-primary shrink-0" />
                  )}
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {template.complexity && (
                    <Badge variant="secondary" className="text-xs">
                      {template.complexity}
                    </Badge>
                  )}
                  {template.patterns_used?.slice(0, 2).map((pattern) => (
                    <Badge key={pattern} variant="outline" className="text-xs">
                      {pattern}
                    </Badge>
                  ))}
                </div>
              </button>
            ))}
          </div>

          {loadingTemplates && (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">Loading template...</span>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setTemplateSelectorOpen(false)}>
              Skip
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Toolbar */}
      <div className="flex items-center justify-between p-2 border-b bg-background">
        <div className="flex items-center gap-2">
          <h2 className="font-semibold">{workflow.name}</h2>
          <Badge variant={workflow.isDraft ? "secondary" : "default"}>
            {workflow.isDraft ? "Draft" : "Published"}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setZoom((z) => Math.min(z + 0.1, 2))}
                >
                  <ZoomIn className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom In</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <span className="text-sm text-muted-foreground w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setZoom((z) => Math.max(z - 0.1, 0.5))}
                >
                  <ZoomOut className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom Out</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => {
                    setZoom(1);
                    setPan({ x: 0, y: 0 });
                  }}
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Reset View</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <div className="w-px h-6 bg-border mx-2" />
          <Button variant="outline" size="sm" onClick={() => setTemplateSelectorOpen(true)}>
            <Wand2 className="h-4 w-4 mr-2" />
            Templates
          </Button>
          <Button variant="outline" size="sm" onClick={() => setSettingsOpen(true)}>
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
          {onCancel && (
            <Button variant="outline" size="sm" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <Button size="sm" onClick={handleSave} disabled={saving}>
            <Save className="h-4 w-4 mr-2" />
            {saving ? "Saving..." : "Save"}
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Node palette */}
        <NodePalette onDragStart={handlePaletteDragStart} />

        {/* Canvas */}
        <div
          ref={canvasRef}
          className="flex-1 relative overflow-hidden bg-muted/20"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleCanvasDrop}
          onClick={() => {
            setSelectedNodeId(null);
            setSelectedEdgeId(null);
            setConnectingFromNode(null);
          }}
        >
          {/* Grid background */}
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `radial-gradient(circle, hsl(var(--muted-foreground) / 0.2) 1px, transparent 1px)`,
              backgroundSize: `${20 * zoom}px ${20 * zoom}px`,
              backgroundPosition: `${pan.x}px ${pan.y}px`,
            }}
          />

          {/* Edges (SVG layer) */}
          <svg
            className="absolute inset-0 pointer-events-none"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "0 0",
            }}
          >
            {workflow.edges.map((edge) => (
              <CanvasEdge
                key={edge.id}
                edge={edge}
                sourceNode={workflow.nodes.find((n) => n.id === edge.sourceNodeId)}
                targetNode={workflow.nodes.find((n) => n.id === edge.targetNodeId)}
                isSelected={selectedEdgeId === edge.id}
                onSelect={() => {
                  setSelectedEdgeId(edge.id);
                  setSelectedNodeId(null);
                }}
                onDelete={() => handleDeleteEdge(edge.id)}
              />
            ))}
          </svg>

          {/* Nodes */}
          <div
            className="absolute inset-0"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "0 0",
            }}
          >
            {workflow.nodes.map((node) => (
              <CanvasNode
                key={node.id}
                node={node}
                nodeType={getNodeType(node.type)}
                isSelected={selectedNodeId === node.id}
                isConnecting={connectingFromNode !== null}
                onSelect={() => {
                  setSelectedNodeId(node.id);
                  setSelectedEdgeId(null);
                }}
                onDragStart={(e) => handleNodeDragStart(node.id, e)}
                onConnectionStart={handleConnectionStart}
                onConnectionEnd={handleConnectionEnd}
                onDelete={() => handleDeleteNode(node.id)}
                onConfigure={() => setConfigNodeId(node.id)}
              />
            ))}
          </div>

          {/* Empty state */}
          {workflow.nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <p className="text-lg font-medium mb-2">Start building your workflow</p>
                <p className="text-sm">Drag nodes from the left panel to get started</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Node configuration sheet */}
      <NodeConfigSheet
        node={configNode || null}
        nodeType={configNodeType}
        open={configNodeId !== null}
        onOpenChange={(open) => !open && setConfigNodeId(null)}
        onSave={handleUpdateNode}
      />

      {/* Workflow settings dialog */}
      <WorkflowSettingsDialog
        workflow={workflow}
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        onSave={setWorkflow}
      />
    </div>
  );
}

// =============================================================================
// Exports for Pages
// =============================================================================

export function WorkflowDesignerPage() {
  const handleSave = async (workflow: Workflow) => {
    console.log("Saving workflow:", workflow);
    // API call would go here
  };

  return (
    <div className="h-screen">
      <WorkflowDesigner onSave={handleSave} />
    </div>
  );
}
