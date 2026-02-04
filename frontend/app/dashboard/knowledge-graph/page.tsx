"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useSession } from "next-auth/react";
import dynamic from "next/dynamic";
import {
  Network,
  Search,
  Filter,
  Users,
  Building2,
  MapPin,
  Lightbulb,
  Calendar,
  Package,
  Cpu,
  FileText,
  Loader2,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Info,
  ChevronRight,
  X,
  Sparkles,
  Trash2,
  Box,
  Square,
  Pause,
  Play,
  StopCircle,
  Clock,
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  Plus,
  Pencil,
  List,
  ArrowRight,
  Save,
} from "lucide-react";

// Dynamically import WebGL graph to avoid SSR issues
const WebGLGraph = dynamic(
  () => import("@/components/knowledge-graph/WebGLGraph"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    ),
  }
);
import { toast } from "sonner";
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
  useKnowledgeGraphStats,
  useKnowledgeGraphData,
  useSearchEntities,
  useEntityNeighborhood,
  useEntityTypes,
  useCurrentExtractionJob,
  usePendingExtractionCount,
  useStartExtractionJob,
  useCancelExtractionJob,
  usePauseExtractionJob,
  useResumeExtractionJob,
  useCreateEntity,
  useUpdateEntity,
  useDeleteEntity,
  useCreateRelation,
  useDeleteRelation,
  useLLMProviders,
  knowledgeGraphQueryKeys,
  api,
  type EntityResponse,
  type GraphNode,
  type GraphEdge,
  type ExtractionJobProgress,
} from "@/lib/api";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import { useQueryClient } from "@tanstack/react-query";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

// Entity type colors and icons
const entityTypeConfig: Record<string, { color: string; icon: React.ElementType; bgColor: string }> = {
  PERSON: { color: "#3b82f6", icon: Users, bgColor: "bg-blue-100 dark:bg-blue-900/30" },
  ORGANIZATION: { color: "#8b5cf6", icon: Building2, bgColor: "bg-purple-100 dark:bg-purple-900/30" },
  LOCATION: { color: "#10b981", icon: MapPin, bgColor: "bg-green-100 dark:bg-green-900/30" },
  CONCEPT: { color: "#f59e0b", icon: Lightbulb, bgColor: "bg-yellow-100 dark:bg-yellow-900/30" },
  DATE: { color: "#ec4899", icon: Calendar, bgColor: "bg-pink-100 dark:bg-pink-900/30" },
  EVENT: { color: "#ef4444", icon: Calendar, bgColor: "bg-red-100 dark:bg-red-900/30" },
  PRODUCT: { color: "#06b6d4", icon: Package, bgColor: "bg-cyan-100 dark:bg-cyan-900/30" },
  TECHNOLOGY: { color: "#6366f1", icon: Cpu, bgColor: "bg-indigo-100 dark:bg-indigo-900/30" },
  DOCUMENT: { color: "#78716c", icon: FileText, bgColor: "bg-stone-100 dark:bg-stone-900/30" },
};

function getEntityConfig(type: string) {
  return entityTypeConfig[type] || entityTypeConfig.CONCEPT;
}

// Simple Canvas-based Graph Visualization
function GraphVisualization({
  nodes,
  edges,
  onNodeClick,
  selectedNodeId,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick: (nodeId: string) => void;
  selectedNodeId: string | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [nodePositions, setNodePositions] = useState<Map<string, { x: number; y: number }>>(new Map());

  // Initialize node positions using force-directed layout
  useEffect(() => {
    if (nodes.length === 0) return;

    const positions = new Map<string, { x: number; y: number }>();
    const width = containerRef.current?.clientWidth || 800;
    const height = containerRef.current?.clientHeight || 600;
    const centerX = width / 2;
    const centerY = height / 2;

    // Initial positions in a circle
    nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / nodes.length;
      const radius = Math.min(width, height) * 0.35;
      positions.set(node.id, {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      });
    });

    // Simple force simulation
    for (let iter = 0; iter < 50; iter++) {
      // Repulsion between nodes
      nodes.forEach((node1) => {
        nodes.forEach((node2) => {
          if (node1.id === node2.id) return;
          const pos1 = positions.get(node1.id)!;
          const pos2 = positions.get(node2.id)!;
          const dx = pos2.x - pos1.x;
          const dy = pos2.y - pos1.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = 5000 / (dist * dist);
          pos1.x -= (dx / dist) * force;
          pos1.y -= (dy / dist) * force;
          pos2.x += (dx / dist) * force;
          pos2.y += (dy / dist) * force;
        });
      });

      // Attraction along edges
      edges.forEach((edge) => {
        const pos1 = positions.get(edge.from);
        const pos2 = positions.get(edge.to);
        if (!pos1 || !pos2) return;
        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = dist * 0.01;
        pos1.x += (dx / dist) * force;
        pos1.y += (dy / dist) * force;
        pos2.x -= (dx / dist) * force;
        pos2.y -= (dy / dist) * force;
      });

      // Center attraction
      nodes.forEach((node) => {
        const pos = positions.get(node.id)!;
        pos.x += (centerX - pos.x) * 0.01;
        pos.y += (centerY - pos.y) * 0.01;
      });
    }

    setNodePositions(positions);
  }, [nodes, edges]);

  // Draw the graph
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || nodePositions.size === 0) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(scale, scale);

    // Draw edges
    edges.forEach((edge) => {
      const from = nodePositions.get(edge.from);
      const to = nodePositions.get(edge.to);
      if (!from || !to) return;

      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Draw edge label
      if (edge.label) {
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        ctx.font = "10px sans-serif";
        ctx.fillStyle = "#64748b";
        ctx.textAlign = "center";
        ctx.fillText(edge.label, midX, midY - 5);
      }
    });

    // Draw nodes
    nodes.forEach((node) => {
      const pos = nodePositions.get(node.id);
      if (!pos) return;

      const config = getEntityConfig(node.type);
      const isSelected = node.id === selectedNodeId;
      const radius = isSelected ? 28 : 24;

      // Node circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isSelected ? config.color : `${config.color}cc`;
      ctx.fill();
      if (isSelected) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Node label
      ctx.font = "11px sans-serif";
      ctx.fillStyle = "#1e293b";
      ctx.textAlign = "center";
      const label = node.label.length > 12 ? node.label.slice(0, 12) + "..." : node.label;
      ctx.fillText(label, pos.x, pos.y + radius + 14);
    });

    ctx.restore();
  }, [nodes, edges, nodePositions, scale, offset, selectedNodeId]);

  // Handle canvas resize
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const resizeObserver = new ResizeObserver(() => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    });

    resizeObserver.observe(container);
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    return () => resizeObserver.disconnect();
  }, []);

  // Handle click on node
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left - offset.x) / scale;
      const y = (e.clientY - rect.top - offset.y) / scale;

      // Find clicked node
      for (const node of nodes) {
        const pos = nodePositions.get(node.id);
        if (!pos) continue;
        const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
        if (dist <= 24) {
          onNodeClick(node.id);
          return;
        }
      }
    },
    [nodes, nodePositions, scale, offset, onNodeClick]
  );

  // Drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[400px]">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onClick={handleCanvasClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Zoom controls */}
      <div className="absolute bottom-4 right-4 flex gap-2">
        <Button
          variant="outline"
          size="icon"
          onClick={() => setScale((s) => Math.min(s * 1.2, 3))}
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          onClick={() => setScale((s) => Math.max(s / 1.2, 0.3))}
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          onClick={() => {
            setScale(1);
            setOffset({ x: 0, y: 0 });
          }}
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
      </div>

      {/* Legend */}
      <div className="absolute top-4 left-4 bg-background/90 backdrop-blur-sm border rounded-lg p-3 space-y-1">
        <p className="text-xs font-medium text-muted-foreground mb-2">Entity Types</p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {Object.entries(entityTypeConfig).slice(0, 6).map(([type, config]) => (
            <div key={type} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: config.color }}
              />
              <span className="text-xs">{type}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Table View Component
function TableView({
  nodes,
  edges,
  onNodeClick,
  selectedNodeId,
  onDeleteRelation,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick: (nodeId: string) => void;
  selectedNodeId: string | null;
  onDeleteRelation?: (relationId: string) => void;
}) {
  const [activeTab, setActiveTab] = useState<"entities" | "relations">("entities");
  const [sortBy, setSortBy] = useState<"name" | "type">("name");

  const sortedNodes = [...nodes].sort((a, b) => {
    if (sortBy === "name") return a.label.localeCompare(b.label);
    return a.type.localeCompare(b.type) || a.label.localeCompare(b.label);
  });

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-2 p-3 border-b">
        <Button
          variant={activeTab === "entities" ? "secondary" : "ghost"}
          size="sm"
          onClick={() => setActiveTab("entities")}
        >
          Entities ({nodes.length})
        </Button>
        <Button
          variant={activeTab === "relations" ? "secondary" : "ghost"}
          size="sm"
          onClick={() => setActiveTab("relations")}
        >
          Relations ({edges.length})
        </Button>
        {activeTab === "entities" && (
          <Select value={sortBy} onValueChange={(v) => setSortBy(v as "name" | "type")}>
            <SelectTrigger className="w-[120px] h-8 ml-auto">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="name">Sort: Name</SelectItem>
              <SelectItem value="type">Sort: Type</SelectItem>
            </SelectContent>
          </Select>
        )}
      </div>
      <div className="flex-1 overflow-auto">
        {activeTab === "entities" ? (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-background border-b">
              <tr>
                <th className="text-left p-2 font-medium text-muted-foreground">Name</th>
                <th className="text-left p-2 font-medium text-muted-foreground">Type</th>
                <th className="text-left p-2 font-medium text-muted-foreground hidden md:table-cell">Description</th>
              </tr>
            </thead>
            <tbody>
              {sortedNodes.map((node) => {
                const config = getEntityConfig(node.type);
                return (
                  <tr
                    key={node.id}
                    className={`border-b cursor-pointer hover:bg-muted/50 ${
                      node.id === selectedNodeId ? "bg-primary/10" : ""
                    }`}
                    onClick={() => onNodeClick(node.id)}
                  >
                    <td className="p-2">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                          style={{ backgroundColor: config.color }}
                        />
                        <span className="font-medium">{node.label}</span>
                      </div>
                    </td>
                    <td className="p-2">
                      <Badge variant="outline" className="text-xs" style={{ borderColor: config.color, color: config.color }}>
                        {node.type}
                      </Badge>
                    </td>
                    <td className="p-2 text-muted-foreground truncate max-w-[300px] hidden md:table-cell">
                      {(node as GraphNode & { description?: string }).description || "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-background border-b">
              <tr>
                <th className="text-left p-2 font-medium text-muted-foreground">Source</th>
                <th className="text-center p-2 font-medium text-muted-foreground">Relation</th>
                <th className="text-left p-2 font-medium text-muted-foreground">Target</th>
                {onDeleteRelation && (
                  <th className="text-right p-2 font-medium text-muted-foreground w-10"></th>
                )}
              </tr>
            </thead>
            <tbody>
              {edges.map((edge, i) => {
                const sourceNode = nodes.find((n) => n.id === edge.from);
                const targetNode = nodes.find((n) => n.id === edge.to);
                return (
                  <tr key={`${edge.from}-${edge.to}-${i}`} className="border-b hover:bg-muted/50">
                    <td className="p-2">
                      <span
                        className="cursor-pointer hover:underline font-medium"
                        onClick={() => onNodeClick(edge.from)}
                      >
                        {sourceNode?.label || edge.from.slice(0, 8)}
                      </span>
                    </td>
                    <td className="p-2 text-center">
                      <div className="flex items-center justify-center gap-1 text-muted-foreground">
                        <ArrowRight className="h-3 w-3" />
                        <span className="text-xs text-primary">{edge.label || "related_to"}</span>
                        <ArrowRight className="h-3 w-3" />
                      </div>
                    </td>
                    <td className="p-2">
                      <span
                        className="cursor-pointer hover:underline font-medium"
                        onClick={() => onNodeClick(edge.to)}
                      >
                        {targetNode?.label || edge.to.slice(0, 8)}
                      </span>
                    </td>
                    {onDeleteRelation && (edge as GraphEdge & { id?: string }).id && (
                      <td className="p-2 text-right">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 text-muted-foreground hover:text-destructive"
                          onClick={() => onDeleteRelation((edge as GraphEdge & { id?: string }).id!)}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// Entity Details Panel
function EntityDetailsPanel({
  entityId,
  onClose,
  onEdit,
  onDelete,
}: {
  entityId: string;
  onClose: () => void;
  onEdit?: (entity: EntityResponse) => void;
  onDelete?: (entityId: string) => void;
}) {
  const { data: neighborhood, isLoading } = useEntityNeighborhood(entityId, {
    max_hops: 1,
    max_neighbors: 10,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-4 w-full mb-2" />
          <Skeleton className="h-4 w-3/4" />
        </CardContent>
      </Card>
    );
  }

  if (!neighborhood) return null;

  const entity = neighborhood.entity;
  const config = getEntityConfig(entity.entity_type);
  const Icon = config.icon;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${config.bgColor}`}>
              <Icon className="h-5 w-5" style={{ color: config.color }} />
            </div>
            <div>
              <CardTitle className="text-lg">{entity.name}</CardTitle>
              <CardDescription className="flex items-center gap-2">
                <Badge variant="outline" style={{ borderColor: config.color, color: config.color }}>
                  {entity.entity_type}
                </Badge>
                <span className="text-xs">{entity.mention_count} mentions</span>
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-1">
            {onEdit && (
              <Button variant="ghost" size="icon" onClick={() => onEdit(entity)} title="Edit entity">
                <Pencil className="h-4 w-4" />
              </Button>
            )}
            {onDelete && (
              <Button variant="ghost" size="icon" onClick={() => onDelete(entityId)} title="Delete entity"
                className="text-muted-foreground hover:text-destructive">
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {entity.description && (
          <p className="text-sm text-muted-foreground">{entity.description}</p>
        )}

        {entity.aliases.length > 0 && (
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">Also known as</p>
            <div className="flex flex-wrap gap-1">
              {entity.aliases.map((alias, i) => (
                <Badge key={i} variant="secondary" className="text-xs">
                  {alias}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {neighborhood.neighbors.length > 0 && (
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-2">
              Connected Entities ({neighborhood.neighbors.length})
            </p>
            <div className="space-y-1">
              {neighborhood.neighbors.slice(0, 5).map((neighbor) => {
                const nConfig = getEntityConfig(neighbor.entity_type);
                return (
                  <div
                    key={neighbor.id}
                    className="flex items-center gap-2 text-sm p-1.5 rounded hover:bg-muted cursor-pointer"
                  >
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: nConfig.color }}
                    />
                    <span>{neighbor.name}</span>
                    <ChevronRight className="h-3 w-3 ml-auto text-muted-foreground" />
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {neighborhood.relations.length > 0 && (
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-2">
              Relationships ({neighborhood.relations.length})
            </p>
            <div className="space-y-1 text-xs text-muted-foreground">
              {neighborhood.relations.slice(0, 5).map((rel) => (
                <div key={rel.id} className="flex items-center gap-1">
                  <span className="font-medium text-foreground">
                    {rel.source_entity_name}
                  </span>
                  <span className="text-primary">{rel.relation_label || rel.relation_type}</span>
                  <span className="font-medium text-foreground">
                    {rel.target_entity_name}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Extraction Job Panel Component
function ExtractionJobPanel({
  job,
  onCancel,
  onPause,
  onResume,
  isCancelling,
  isPausing,
  isResuming,
}: {
  job: ExtractionJobProgress;
  onCancel: () => void;
  onPause: () => void;
  onResume: () => void;
  isCancelling: boolean;
  isPausing: boolean;
  isResuming: boolean;
}) {
  const [showErrors, setShowErrors] = useState(false);

  const formatTime = (seconds: number | null) => {
    if (!seconds) return null;
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  };

  const formatRelativeTime = (isoString: string | null | undefined) => {
    if (!isoString) return null;
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);

    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffMin < 60) return `${diffMin}m ago`;
    return `${Math.floor(diffMin / 60)}h ${diffMin % 60}m ago`;
  };

  // Detect if job might be stuck (no activity for 5+ minutes while running)
  const isStuck = (() => {
    if (job.status !== "running") return false;
    if (!job.last_activity_at) return false;
    const lastActivity = new Date(job.last_activity_at).getTime();
    const now = Date.now();
    const minutesSinceActivity = (now - lastActivity) / 1000 / 60;
    return minutesSinceActivity > 5;
  })();

  const getStatusColor = (status: string) => {
    if (isStuck) return "bg-orange-500"; // Override for stuck jobs
    switch (status) {
      case "running":
        return "bg-blue-500";
      case "paused":
        return "bg-yellow-500";
      case "completed":
        return "bg-green-500";
      case "completed_with_errors":
        return "bg-yellow-600";
      case "cancelled":
        return "bg-gray-500";
      case "failed":
        return "bg-red-500";
      default:
        return "bg-gray-400";
    }
  };

  // Calculate total done (processed + failed)
  const totalDone = job.processed_documents + job.failed_documents;

  return (
    <Card className="border-primary/20 bg-primary/5">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            Entity Extraction
          </CardTitle>
          <Badge
            variant="outline"
            className={`${getStatusColor(job.status)} text-white border-0`}
          >
            {isStuck ? "stuck?" : job.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>Progress</span>
            <span className="font-medium">
              {job.processed_documents} completed
              {job.failed_documents > 0 && `, ${job.failed_documents} failed`}
              {" / "}{job.total_documents} total
            </span>
          </div>
          <Progress value={job.progress_percent} className="h-2" />
        </div>

        {job.current_document && job.status === "running" && (
          <div className="text-xs text-muted-foreground flex items-center gap-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            Processing: {job.current_document}
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Entities:</span>
            <span className="font-medium">{job.total_entities}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Relations:</span>
            <span className="font-medium">{job.total_relations}</span>
          </div>
          {job.last_activity_at && (
            <div className="flex items-center gap-1 col-span-2">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">
                Last activity: {formatRelativeTime(job.last_activity_at)}
              </span>
            </div>
          )}
          {job.estimated_remaining_seconds && job.status === "running" && !isStuck && (
            <div className="flex items-center gap-1 col-span-2">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span>~{formatTime(job.estimated_remaining_seconds)} remaining</span>
            </div>
          )}
        </div>

        {/* Stuck job warning */}
        {isStuck && (
          <div className="mt-2 p-2 bg-orange-500/10 rounded text-xs text-orange-600 dark:text-orange-400">
            <div className="flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" />
              <span className="font-medium">Job may be stuck</span>
            </div>
            <p className="mt-1 text-muted-foreground">
              No progress for over 5 minutes. Consider cancelling and restarting.
            </p>
          </div>
        )}

        {/* Error summary with expandable details */}
        {job.error_count > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowErrors(!showErrors)}
              className="flex items-center gap-1 text-xs text-red-500 hover:text-red-600"
            >
              <AlertCircle className="h-3 w-3" />
              <span>{job.error_count} error{job.error_count > 1 ? "s" : ""}</span>
              <ChevronDown className={`h-3 w-3 transition-transform ${showErrors ? "rotate-180" : ""}`} />
            </button>
            {showErrors && job.errors && job.errors.length > 0 && (
              <div className="mt-1 p-2 bg-red-500/10 rounded text-xs max-h-24 overflow-y-auto">
                {job.errors.map((err: { doc_id?: string; doc_name?: string; error: string }, i: number) => (
                  <div key={i} className="text-red-600 dark:text-red-400 mb-1">
                    <span className="font-medium">{err.doc_name || err.doc_id}:</span>{" "}
                    {err.error.length > 100 ? `${err.error.slice(0, 100)}...` : err.error}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {(job.can_cancel || job.can_pause || job.can_resume) && (
          <div className="flex gap-2 pt-2">
            {job.can_pause && (
              <Button
                size="sm"
                variant="outline"
                onClick={onPause}
                disabled={isPausing}
                className="flex-1"
              >
                {isPausing ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <Pause className="h-3 w-3 mr-1" />
                )}
                Pause
              </Button>
            )}
            {job.can_resume && (
              <Button
                size="sm"
                variant="outline"
                onClick={onResume}
                disabled={isResuming}
                className="flex-1"
              >
                {isResuming ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <Play className="h-3 w-3 mr-1" />
                )}
                Resume
              </Button>
            )}
            {job.can_cancel && (
              <Button
                size="sm"
                variant="destructive"
                onClick={onCancel}
                disabled={isCancelling}
                className="flex-1"
              >
                {isCancelling ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <StopCircle className="h-3 w-3 mr-1" />
                )}
                Cancel
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function KnowledgeGraphPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";
  const queryClient = useQueryClient();

  const [searchQuery, setSearchQuery] = useState("");
  const [entityTypeFilter, setEntityTypeFilter] = useState<string>("all");
  const [nodeLimit, setNodeLimit] = useState(100);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [isCleaning, setIsCleaning] = useState(false);
  const [viewMode, setViewMode] = useState<"2d" | "3d" | "table">("2d");
  const [onlyNewDocs, setOnlyNewDocs] = useState(true);
  const [selectedProvider, setSelectedProvider] = useState<string>("default");

  // Entity/Relation editing state
  const [showEntityDialog, setShowEntityDialog] = useState(false);
  const [showRelationDialog, setShowRelationDialog] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [editingEntity, setEditingEntity] = useState<EntityResponse | null>(null);
  const [deleteEntityId, setDeleteEntityId] = useState<string | null>(null);
  const [entityForm, setEntityForm] = useState({ name: "", entity_type: "CONCEPT", description: "", aliases: "" });
  const [relationForm, setRelationForm] = useState({ source_entity_id: "", target_entity_id: "", relation_type: "related_to", relation_label: "" });

  // Knowledge Graph Queries
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useKnowledgeGraphStats({
    enabled: isAuthenticated,
  });

  const { data: graphData, isLoading: graphLoading, refetch: refetchGraph } = useKnowledgeGraphData(
    {
      limit: nodeLimit,
      entity_type: entityTypeFilter !== "all" ? entityTypeFilter : undefined,
    },
    { enabled: isAuthenticated }
  );

  const { data: entityTypes } = useEntityTypes({ enabled: isAuthenticated });

  const { data: searchResults, isLoading: searchLoading } = useSearchEntities(
    {
      query: searchQuery,
      entity_type: entityTypeFilter !== "all" ? entityTypeFilter : undefined,
      page_size: 20,
    },
    { enabled: isAuthenticated && searchQuery.length > 0 }
  );

  // Extraction Job Queries
  const {
    data: currentJob,
    refetch: refetchCurrentJob,
  } = useCurrentExtractionJob({
    enabled: isAuthenticated,
    refetchInterval: 2000, // Poll every 2 seconds when job is running
  });

  const { data: pendingInfo } = usePendingExtractionCount({
    enabled: isAuthenticated,
  });

  // LLM Providers for extraction
  const { data: llmProviders } = useLLMProviders({
    enabled: isAuthenticated,
  });

  // Extraction Job Mutations
  const startExtractionMutation = useStartExtractionJob();
  const cancelExtractionMutation = useCancelExtractionJob();
  const pauseExtractionMutation = usePauseExtractionJob();
  const resumeExtractionMutation = useResumeExtractionJob();

  // Entity/Relation CRUD Mutations
  const createEntityMutation = useCreateEntity();
  const updateEntityMutation = useUpdateEntity();
  const deleteEntityMutation = useDeleteEntity();
  const createRelationMutation = useCreateRelation();
  const deleteRelationMutation = useDeleteRelation();

  // Refresh stats when job completes
  useEffect(() => {
    if (currentJob?.status === "completed" || currentJob?.status === "completed_with_errors") {
      refetchStats();
      refetchGraph();
    }
  }, [currentJob?.status, refetchStats, refetchGraph]);

  const handleRefresh = () => {
    refetchStats();
    refetchGraph();
  };

  const handleNodeClick = (nodeId: string) => {
    setSelectedEntityId(nodeId);
  };

  const handleStartExtraction = async () => {
    try {
      const result = await startExtractionMutation.mutateAsync({
        only_new_documents: onlyNewDocs,
        provider_id: selectedProvider !== "default" ? selectedProvider : undefined,
      });

      if (result.status === "already_running") {
        toast.info(result.message);
      } else {
        toast.success("Extraction job started! You can navigate away and check progress later.");
      }
      refetchCurrentJob();
    } catch (error) {
      console.error("Failed to start extraction:", error);
      toast.error(error instanceof Error ? error.message : "Failed to start extraction");
    }
  };

  const handleCancelExtraction = async () => {
    if (!currentJob) return;
    try {
      await cancelExtractionMutation.mutateAsync(currentJob.job_id);
      toast.success("Extraction job cancelled");
      refetchCurrentJob();
    } catch (error) {
      console.error("Failed to cancel extraction:", error);
      toast.error(error instanceof Error ? error.message : "Failed to cancel extraction");
    }
  };

  const handlePauseExtraction = async () => {
    if (!currentJob) return;
    try {
      await pauseExtractionMutation.mutateAsync(currentJob.job_id);
      toast.success("Extraction job paused");
      refetchCurrentJob();
    } catch (error) {
      console.error("Failed to pause extraction:", error);
      toast.error(error instanceof Error ? error.message : "Failed to pause extraction");
    }
  };

  const handleResumeExtraction = async () => {
    if (!currentJob) return;
    try {
      await resumeExtractionMutation.mutateAsync(currentJob.job_id);
      toast.success("Extraction job resumed");
      refetchCurrentJob();
    } catch (error) {
      console.error("Failed to resume extraction:", error);
      toast.error(error instanceof Error ? error.message : "Failed to resume extraction");
    }
  };

  // Entity CRUD handlers
  const openAddEntityDialog = () => {
    setEditingEntity(null);
    setEntityForm({ name: "", entity_type: "CONCEPT", description: "", aliases: "" });
    setShowEntityDialog(true);
  };

  const openEditEntityDialog = (entity: EntityResponse) => {
    setEditingEntity(entity);
    setEntityForm({
      name: entity.name,
      entity_type: entity.entity_type,
      description: entity.description || "",
      aliases: entity.aliases?.join(", ") || "",
    });
    setShowEntityDialog(true);
  };

  const handleSaveEntity = async () => {
    const data = {
      name: entityForm.name.trim(),
      entity_type: entityForm.entity_type,
      description: entityForm.description.trim() || undefined,
      aliases: entityForm.aliases.trim() ? entityForm.aliases.split(",").map((a) => a.trim()).filter(Boolean) : undefined,
    };

    try {
      if (editingEntity) {
        await updateEntityMutation.mutateAsync({ entityId: editingEntity.id, data });
        toast.success(`Entity "${data.name}" updated`);
      } else {
        await createEntityMutation.mutateAsync(data);
        toast.success(`Entity "${data.name}" created`);
      }
      setShowEntityDialog(false);
      refetchGraph();
      refetchStats();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to save entity");
    }
  };

  const handleConfirmDeleteEntity = async () => {
    if (!deleteEntityId) return;
    try {
      await deleteEntityMutation.mutateAsync({ entityId: deleteEntityId, cascade: true });
      toast.success("Entity deleted");
      setSelectedEntityId(null);
      setShowDeleteConfirm(false);
      setDeleteEntityId(null);
      refetchGraph();
      refetchStats();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to delete entity");
    }
  };

  const handleDeleteEntityPrompt = (entityId: string) => {
    setDeleteEntityId(entityId);
    setShowDeleteConfirm(true);
  };

  // Relation CRUD handlers
  const openAddRelationDialog = () => {
    setRelationForm({
      source_entity_id: selectedEntityId || "",
      target_entity_id: "",
      relation_type: "related_to",
      relation_label: "",
    });
    setShowRelationDialog(true);
  };

  const handleSaveRelation = async () => {
    try {
      await createRelationMutation.mutateAsync({
        source_entity_id: relationForm.source_entity_id,
        target_entity_id: relationForm.target_entity_id,
        relation_type: relationForm.relation_type,
        relation_label: relationForm.relation_label.trim() || undefined,
      });
      toast.success("Relation created");
      setShowRelationDialog(false);
      refetchGraph();
      refetchStats();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to create relation");
    }
  };

  const handleDeleteRelation = async (relationId: string) => {
    try {
      await deleteRelationMutation.mutateAsync(relationId);
      toast.success("Relation deleted");
      refetchGraph();
      refetchStats();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to delete relation");
    }
  };

  const handleCleanup = async () => {
    setIsCleaning(true);
    try {
      const result = await api.cleanupKnowledgeGraph(false);
      if (result.orphan_entities_removed > 0 || result.orphan_relations_removed > 0) {
        toast.success(result.message || "Cleanup completed");
      } else {
        toast.info("No orphan entities found - graph is clean!");
      }
      // Refresh the stats and graph after cleanup
      setTimeout(() => {
        refetchStats();
        refetchGraph();
      }, 500);
    } catch (error) {
      console.error("Cleanup error:", error);
      toast.error(error instanceof Error ? error.message : "Failed to cleanup graph");
    } finally {
      setIsCleaning(false);
    }
  };

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
        <p className="text-muted-foreground">Please sign in to view the knowledge graph.</p>
      </div>
    );
  }

  const hasActiveJob = !!(currentJob && ["queued", "running", "paused"].includes(currentJob.status));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Network className="h-6 w-6" />
            Knowledge Graph
          </h1>
          <p className="text-muted-foreground">
            Explore entities and relationships extracted from your documents
            {pendingInfo && pendingInfo.pending_count > 0 && !hasActiveJob && (
              <span className="ml-2 text-primary">
                ({pendingInfo.pending_count} documents pending extraction)
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {!hasActiveJob && (
            <>
              <div className="flex items-center gap-2 mr-2">
                <Checkbox
                  id="only-new"
                  checked={onlyNewDocs}
                  onCheckedChange={(checked) => setOnlyNewDocs(checked as boolean)}
                />
                <label htmlFor="only-new" className="text-sm cursor-pointer">
                  Only new documents
                </label>
              </div>
              {/* LLM Provider Selector */}
              <Select value={selectedProvider} onValueChange={setSelectedProvider}>
                <SelectTrigger className="w-[180px]">
                  <Cpu className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Select LLM" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default Provider</SelectItem>
                  {llmProviders?.providers
                    ?.filter((p: { is_active: boolean }) => p.is_active)
                    .map((provider: { id: string; name: string; provider_type: string }) => (
                      <SelectItem key={provider.id} value={provider.id}>
                        {provider.name} ({provider.provider_type})
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
              <Button
                variant="default"
                onClick={handleStartExtraction}
                disabled={startExtractionMutation.isPending}
                className="gap-2"
              >
                {startExtractionMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                {startExtractionMutation.isPending ? "Starting..." : "Extract Entities"}
              </Button>
            </>
          )}
          <Button
            variant="outline"
            onClick={handleCleanup}
            disabled={isCleaning || hasActiveJob}
            className="gap-2"
            title="Remove orphan entities (entities with no document references)"
          >
            {isCleaning ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            {isCleaning ? "Cleaning..." : "Cleanup"}
          </Button>
          <Button variant="outline" onClick={handleRefresh} disabled={graphLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${graphLoading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Extraction Job Progress Panel */}
      {currentJob && ["queued", "running", "paused", "completed", "completed_with_errors"].includes(currentJob.status) && (
        <ExtractionJobPanel
          job={currentJob}
          onCancel={handleCancelExtraction}
          onPause={handlePauseExtraction}
          onResume={handleResumeExtraction}
          isCancelling={cancelExtractionMutation.isPending}
          isPausing={pauseExtractionMutation.isPending}
          isResuming={resumeExtractionMutation.isPending}
        />
      )}

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Entities
            </CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className="text-2xl font-bold">{stats?.total_entities || 0}</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Relationships
            </CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className="text-2xl font-bold">{stats?.total_relations || 0}</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Entity Mentions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className="text-2xl font-bold">{stats?.total_mentions || 0}</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Documents Indexed
            </CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <div className="text-2xl font-bold">{stats?.documents_with_entities || 0}</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-4">
        {/* Left Panel - Filters & Search */}
        <div className="space-y-4">
          {/* Search */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Search Entities</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by name..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>

              <Select value={entityTypeFilter} onValueChange={setEntityTypeFilter}>
                <SelectTrigger>
                  <Filter className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Filter by type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  {entityTypes?.entity_types.map((type) => (
                    <SelectItem key={type} value={type}>
                      {type}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">Max nodes: {nodeLimit}</label>
                <input
                  type="range"
                  min="20"
                  max="200"
                  step="10"
                  value={nodeLimit}
                  onChange={(e) => setNodeLimit(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            </CardContent>
          </Card>

          {/* Search Results */}
          {searchQuery && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center justify-between">
                  Results
                  {searchLoading && <Loader2 className="h-3 w-3 animate-spin" />}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {searchResults?.entities.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No entities found</p>
                ) : (
                  <div className="space-y-1">
                    {searchResults?.entities.map((entity) => {
                      const config = getEntityConfig(entity.entity_type);
                      return (
                        <div
                          key={entity.id}
                          className="flex items-center gap-2 p-2 rounded hover:bg-muted cursor-pointer"
                          onClick={() => setSelectedEntityId(entity.id)}
                        >
                          <div
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: config.color }}
                          />
                          <span className="text-sm truncate flex-1">{entity.name}</span>
                          <Badge variant="outline" className="text-xs">
                            {entity.mention_count}
                          </Badge>
                        </div>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Top Entities */}
          {!searchQuery && stats?.top_entities && stats.top_entities.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Top Entities</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-1">
                  {stats.top_entities.slice(0, 10).map((entity) => {
                    const config = getEntityConfig(entity.entity_type);
                    return (
                      <div
                        key={entity.id}
                        className="flex items-center gap-2 p-2 rounded hover:bg-muted cursor-pointer"
                        onClick={() => setSelectedEntityId(entity.id)}
                      >
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: config.color }}
                        />
                        <span className="text-sm truncate flex-1">{entity.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {entity.mention_count}
                        </Badge>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Entity Type Distribution */}
          {stats?.entity_type_distribution && Object.keys(stats.entity_type_distribution).length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Entity Types</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(stats.entity_type_distribution)
                    .sort((a, b) => b[1] - a[1])
                    .map(([type, count]) => {
                      const config = getEntityConfig(type);
                      const Icon = config.icon;
                      const percentage = (count / stats.total_entities) * 100;
                      return (
                        <div key={type} className="space-y-1">
                          <div className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              <Icon className="h-3 w-3" style={{ color: config.color }} />
                              <span>{type}</span>
                            </div>
                            <span className="text-muted-foreground">{count}</span>
                          </div>
                          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${percentage}%`,
                                backgroundColor: config.color,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Panel - Graph Visualization */}
        <div className="lg:col-span-3 space-y-4">
          {/* Graph */}
          <Card className="overflow-hidden">
            <CardHeader className="pb-3 border-b">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Network className="h-4 w-4" />
                  Graph Visualization
                  {graphData && (
                    <span className="text-muted-foreground font-normal">
                      ({graphData.nodes.length} nodes, {graphData.edges.length} edges)
                    </span>
                  )}
                </CardTitle>
                <div className="flex items-center gap-4">
                  {/* Add Entity/Relation buttons */}
                  <div className="flex items-center gap-1">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 gap-1"
                      onClick={openAddEntityDialog}
                    >
                      <Plus className="h-3 w-3" />
                      <span className="text-xs">Entity</span>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 gap-1"
                      onClick={openAddRelationDialog}
                      disabled={!graphData || graphData.nodes.length < 2}
                    >
                      <Plus className="h-3 w-3" />
                      <span className="text-xs">Relation</span>
                    </Button>
                  </div>
                  {/* 2D/3D/Table Toggle */}
                  <div className="flex items-center gap-1 border rounded-lg p-0.5">
                    <Button
                      variant={viewMode === "2d" ? "secondary" : "ghost"}
                      size="sm"
                      className="h-7 px-2 gap-1"
                      onClick={() => setViewMode("2d")}
                    >
                      <Square className="h-3 w-3" />
                      <span className="text-xs">2D</span>
                    </Button>
                    <Button
                      variant={viewMode === "3d" ? "secondary" : "ghost"}
                      size="sm"
                      className="h-7 px-2 gap-1"
                      onClick={() => setViewMode("3d")}
                    >
                      <Box className="h-3 w-3" />
                      <span className="text-xs">3D</span>
                    </Button>
                    <Button
                      variant={viewMode === "table" ? "secondary" : "ghost"}
                      size="sm"
                      className="h-7 px-2 gap-1"
                      onClick={() => setViewMode("table")}
                    >
                      <List className="h-3 w-3" />
                      <span className="text-xs">Table</span>
                    </Button>
                  </div>
                  <div className="hidden lg:flex items-center gap-2 text-xs text-muted-foreground">
                    <Info className="h-3 w-3" />
                    <span>
                      {viewMode === "3d"
                        ? "Drag to rotate. Scroll to zoom."
                        : viewMode === "table"
                        ? "Click rows to view details."
                        : "Click nodes to view details. Drag to pan."}
                    </span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0 h-[calc(100vh-320px)] min-h-[500px]">
              {graphLoading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : graphData && graphData.nodes.length > 0 ? (
                viewMode === "3d" ? (
                  <WebGLGraph
                    nodes={graphData.nodes}
                    edges={graphData.edges}
                    onNodeClick={handleNodeClick}
                    selectedNodeId={selectedEntityId}
                  />
                ) : viewMode === "table" ? (
                  <TableView
                    nodes={graphData.nodes}
                    edges={graphData.edges}
                    onNodeClick={handleNodeClick}
                    selectedNodeId={selectedEntityId}
                    onDeleteRelation={handleDeleteRelation}
                  />
                ) : (
                  <GraphVisualization
                    nodes={graphData.nodes}
                    edges={graphData.edges}
                    onNodeClick={handleNodeClick}
                    selectedNodeId={selectedEntityId}
                  />
                )
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <Network className="h-12 w-12 mb-4 opacity-50" />
                  <p className="font-medium">No entities found in the knowledge graph</p>
                  <p className="text-sm mb-4">Extract entities from your uploaded documents to populate the graph</p>
                  <div className="flex gap-2">
                    <Button
                      onClick={handleStartExtraction}
                      disabled={startExtractionMutation.isPending || hasActiveJob}
                      className="gap-2"
                    >
                      {startExtractionMutation.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Sparkles className="h-4 w-4" />
                      )}
                      {startExtractionMutation.isPending ? "Starting..." : "Extract from Documents"}
                    </Button>
                    <Button variant="outline" onClick={openAddEntityDialog} className="gap-2">
                      <Plus className="h-4 w-4" />
                      Add Entity Manually
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Entity Details */}
          {selectedEntityId && (
            <EntityDetailsPanel
              entityId={selectedEntityId}
              onClose={() => setSelectedEntityId(null)}
              onEdit={openEditEntityDialog}
              onDelete={handleDeleteEntityPrompt}
            />
          )}
        </div>
      </div>

      {/* Create/Edit Entity Dialog */}
      <Dialog open={showEntityDialog} onOpenChange={setShowEntityDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>{editingEntity ? "Edit Entity" : "Add Entity"}</DialogTitle>
            <DialogDescription>
              {editingEntity ? "Update the entity details." : "Manually add a new entity to the knowledge graph."}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="entity-name">Name</Label>
              <Input
                id="entity-name"
                value={entityForm.name}
                onChange={(e) => setEntityForm((f) => ({ ...f, name: e.target.value }))}
                placeholder="Entity name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="entity-type">Type</Label>
              <Select
                value={entityForm.entity_type}
                onValueChange={(v) => setEntityForm((f) => ({ ...f, entity_type: v }))}
              >
                <SelectTrigger id="entity-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(entityTypeConfig).map((type) => (
                    <SelectItem key={type} value={type}>{type}</SelectItem>
                  ))}
                  <SelectItem value="OTHER">OTHER</SelectItem>
                  <SelectItem value="METRIC">METRIC</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="entity-desc">Description</Label>
              <Textarea
                id="entity-desc"
                value={entityForm.description}
                onChange={(e) => setEntityForm((f) => ({ ...f, description: e.target.value }))}
                placeholder="Optional description"
                rows={3}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="entity-aliases">Aliases (comma-separated)</Label>
              <Input
                id="entity-aliases"
                value={entityForm.aliases}
                onChange={(e) => setEntityForm((f) => ({ ...f, aliases: e.target.value }))}
                placeholder="alias1, alias2, alias3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowEntityDialog(false)}>Cancel</Button>
            <Button
              onClick={handleSaveEntity}
              disabled={!entityForm.name.trim() || createEntityMutation.isPending || updateEntityMutation.isPending}
            >
              {(createEntityMutation.isPending || updateEntityMutation.isPending) && (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              )}
              <Save className="h-4 w-4 mr-2" />
              {editingEntity ? "Update" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Relation Dialog */}
      <Dialog open={showRelationDialog} onOpenChange={setShowRelationDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Add Relation</DialogTitle>
            <DialogDescription>
              Create a relationship between two entities.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label>Source Entity</Label>
              <Select
                value={relationForm.source_entity_id}
                onValueChange={(v) => setRelationForm((f) => ({ ...f, source_entity_id: v }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select source entity" />
                </SelectTrigger>
                <SelectContent className="max-h-[200px]">
                  {graphData?.nodes.map((node) => (
                    <SelectItem key={node.id} value={node.id}>
                      {node.label} ({node.type})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Relation Type</Label>
              <Select
                value={relationForm.relation_type}
                onValueChange={(v) => setRelationForm((f) => ({ ...f, relation_type: v }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["related_to", "works_for", "located_in", "part_of", "created_by",
                    "uses", "mentions", "before", "after", "causes", "contains", "similar_to", "other"
                  ].map((type) => (
                    <SelectItem key={type} value={type}>{type.replace(/_/g, " ")}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Target Entity</Label>
              <Select
                value={relationForm.target_entity_id}
                onValueChange={(v) => setRelationForm((f) => ({ ...f, target_entity_id: v }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select target entity" />
                </SelectTrigger>
                <SelectContent className="max-h-[200px]">
                  {graphData?.nodes
                    .filter((n) => n.id !== relationForm.source_entity_id)
                    .map((node) => (
                      <SelectItem key={node.id} value={node.id}>
                        {node.label} ({node.type})
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="relation-label">Label (optional)</Label>
              <Input
                id="relation-label"
                value={relationForm.relation_label}
                onChange={(e) => setRelationForm((f) => ({ ...f, relation_label: e.target.value }))}
                placeholder="Custom label for this relation"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowRelationDialog(false)}>Cancel</Button>
            <Button
              onClick={handleSaveRelation}
              disabled={!relationForm.source_entity_id || !relationForm.target_entity_id || createRelationMutation.isPending}
            >
              {createRelationMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
              <Save className="h-4 w-4 mr-2" />
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Entity Confirmation */}
      <AlertDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Entity</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this entity and all its relations. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleConfirmDeleteEntity}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleteEntityMutation.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
