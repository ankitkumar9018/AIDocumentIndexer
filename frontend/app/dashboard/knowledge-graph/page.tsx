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
  api,
  type EntityResponse,
  type GraphNode,
  type GraphEdge,
} from "@/lib/api";

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

// Entity Details Panel
function EntityDetailsPanel({
  entityId,
  onClose,
}: {
  entityId: string;
  onClose: () => void;
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
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
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

export default function KnowledgeGraphPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [searchQuery, setSearchQuery] = useState("");
  const [entityTypeFilter, setEntityTypeFilter] = useState<string>("all");
  const [nodeLimit, setNodeLimit] = useState(100);
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isCleaning, setIsCleaning] = useState(false);
  const [use3DView, setUse3DView] = useState(false); // Toggle between 2D canvas and 3D WebGL

  // Queries
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

  const handleRefresh = () => {
    refetchStats();
    refetchGraph();
  };

  const handleNodeClick = (nodeId: string) => {
    setSelectedEntityId(nodeId);
  };

  const handleExtractAll = async () => {
    setIsExtracting(true);
    try {
      const result = await api.extractEntitiesFromAllDocuments();
      toast.success(result.message || "Entity extraction completed");
      // Refresh the stats and graph after extraction
      setTimeout(() => {
        refetchStats();
        refetchGraph();
      }, 1000);
    } catch (error) {
      console.error("Entity extraction error:", error);
      toast.error(error instanceof Error ? error.message : "Failed to extract entities");
    } finally {
      setIsExtracting(false);
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
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="default"
            onClick={handleExtractAll}
            disabled={isExtracting}
            className="gap-2"
          >
            {isExtracting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {isExtracting ? "Extracting..." : "Extract Entities"}
          </Button>
          <Button
            variant="outline"
            onClick={handleCleanup}
            disabled={isCleaning}
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
                  {/* 2D/3D Toggle */}
                  <div className="flex items-center gap-1 border rounded-lg p-0.5">
                    <Button
                      variant={!use3DView ? "secondary" : "ghost"}
                      size="sm"
                      className="h-7 px-2 gap-1"
                      onClick={() => setUse3DView(false)}
                    >
                      <Square className="h-3 w-3" />
                      <span className="text-xs">2D</span>
                    </Button>
                    <Button
                      variant={use3DView ? "secondary" : "ghost"}
                      size="sm"
                      className="h-7 px-2 gap-1"
                      onClick={() => setUse3DView(true)}
                    >
                      <Box className="h-3 w-3" />
                      <span className="text-xs">3D</span>
                    </Button>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Info className="h-3 w-3" />
                    <span>{use3DView ? "Drag to rotate. Scroll to zoom." : "Click nodes to view details. Drag to pan."}</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0 h-[500px]">
              {graphLoading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : graphData && graphData.nodes.length > 0 ? (
                use3DView ? (
                  <WebGLGraph
                    nodes={graphData.nodes}
                    edges={graphData.edges}
                    onNodeClick={handleNodeClick}
                    selectedNodeId={selectedEntityId}
                    height={500}
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
                  <Button
                    onClick={handleExtractAll}
                    disabled={isExtracting}
                    className="gap-2"
                  >
                    {isExtracting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4" />
                    )}
                    {isExtracting ? "Extracting..." : "Extract Entities from Documents"}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Entity Details */}
          {selectedEntityId && (
            <EntityDetailsPanel
              entityId={selectedEntityId}
              onClose={() => setSelectedEntityId(null)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
