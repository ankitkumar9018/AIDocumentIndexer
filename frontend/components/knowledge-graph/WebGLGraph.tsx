"use client";

import { useRef, useEffect, useCallback, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { ForceGraphMethods, NodeObject, LinkObject } from "react-force-graph-3d";

// Import THREE statically to ensure single instance
// This prevents the "Multiple instances of Three.js" warning
import * as THREE from "three";

// Ensure THREE is available globally for react-force-graph-3d
if (typeof window !== "undefined") {
  (window as unknown as { THREE?: typeof THREE }).THREE = THREE;
}

// Entity type colors matching the main knowledge graph page
const entityTypeColors: Record<string, string> = {
  PERSON: "#3b82f6",
  ORGANIZATION: "#8b5cf6",
  LOCATION: "#10b981",
  CONCEPT: "#f59e0b",
  DATE: "#ec4899",
  EVENT: "#ef4444",
  PRODUCT: "#06b6d4",
  TECHNOLOGY: "#6366f1",
  DOCUMENT: "#78716c",
};

function getEntityColor(type: string): string {
  return entityTypeColors[type] || entityTypeColors.CONCEPT;
}

// Types for graph data
export interface GraphNode {
  id: string;
  label: string;
  type: string;
}

export interface GraphEdge {
  from: string;
  to: string;
  label?: string;
  weight?: number;
}

interface WebGLGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick: (nodeId: string) => void;
  selectedNodeId: string | null;
  height?: number;
}

// Dynamically import ForceGraph3D to avoid SSR issues
const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
    </div>
  ),
});

// Internal node/link types for the force graph
interface FGNode extends NodeObject {
  id: string;
  label: string;
  type: string;
  color: string;
}

interface FGLink extends LinkObject {
  source: string | FGNode;
  target: string | FGNode;
  label?: string;
}

export default function WebGLGraph({
  nodes,
  edges,
  onNodeClick,
  selectedNodeId,
  height: propHeight,
}: WebGLGraphProps) {
  const fgRef = useRef<ForceGraphMethods | undefined>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: propHeight || 600 });

  // Auto-detect container dimensions
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const updateDimensions = () => {
      const parent = el.parentElement;
      const w = el.clientWidth || parent?.clientWidth || 800;
      const h = propHeight || parent?.clientHeight || 600;
      if (w > 100 && h > 100) {
        setDimensions({ width: w, height: h });
      }
    };

    // Initial + delayed measurement (after layout)
    updateDimensions();
    const timer = setTimeout(updateDimensions, 100);

    const observer = new ResizeObserver(updateDimensions);
    observer.observe(el.parentElement || el);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [propHeight]);

  // Transform data for force-graph format
  const graphData = useMemo(() => {
    const fgNodes: FGNode[] = nodes.map((node) => ({
      id: node.id,
      label: node.label,
      type: node.type,
      color: getEntityColor(node.type),
    }));

    const fgLinks: FGLink[] = edges.map((edge) => ({
      source: edge.from,
      target: edge.to,
      label: edge.label,
    }));

    return { nodes: fgNodes, links: fgLinks };
  }, [nodes, edges]);

  // Focus on selected node
  useEffect(() => {
    if (selectedNodeId && fgRef.current) {
      const node = graphData.nodes.find((n) => n.id === selectedNodeId);
      if (node && node.x !== undefined && node.y !== undefined && node.z !== undefined) {
        const distance = 200;
        const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
        fgRef.current.cameraPosition(
          { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
          { x: node.x, y: node.y, z: node.z },
          1000
        );
      }
    }
  }, [selectedNodeId, graphData.nodes]);

  // Handle node click
  const handleNodeClick = useCallback(
    (node: NodeObject) => {
      if (node && typeof node.id === "string") {
        onNodeClick(node.id);
      }
    },
    [onNodeClick]
  );

  // Custom node rendering with Three.js
  const nodeThreeObject = useCallback(
    (node: NodeObject) => {
      // Use the globally imported THREE instance (not require)
      const fgNode = node as FGNode;
      const isSelected = fgNode.id === selectedNodeId;

      // Create a group to hold sphere and label
      const group = new THREE.Group();

      // Create sphere for node
      const sphereGeometry = new THREE.SphereGeometry(isSelected ? 8 : 6, 16, 16);
      const sphereMaterial = new THREE.MeshLambertMaterial({
        color: fgNode.color,
        transparent: true,
        opacity: isSelected ? 1 : 0.85,
      });
      const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      group.add(sphere);

      // Add highlight ring for selected node
      if (isSelected) {
        const ringGeometry = new THREE.RingGeometry(10, 12, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.8,
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = Math.PI / 2;
        group.add(ring);
      }

      // Create text sprite for label
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (context) {
        canvas.width = 256;
        canvas.height = 64;
        context.fillStyle = "rgba(0, 0, 0, 0)";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = "bold 24px Arial";
        context.fillStyle = "#ffffff";
        context.textAlign = "center";
        context.textBaseline = "middle";

        const label =
          fgNode.label.length > 15 ? fgNode.label.slice(0, 15) + "..." : fgNode.label;
        context.fillText(label, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({
          map: texture,
          transparent: true,
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(40, 10, 1);
        sprite.position.y = -12;
        group.add(sprite);
      }

      return group;
    },
    [selectedNodeId]
  );

  // Custom link rendering
  const linkColor = useCallback(() => "#94a3b8", []);
  const linkWidth = useCallback(() => 1, []);

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        nodeId="id"
        nodeLabel={(node) => {
          const fgNode = node as FGNode;
          return `${fgNode.label} (${fgNode.type})`;
        }}
        nodeThreeObject={nodeThreeObject}
        nodeThreeObjectExtend={false}
        onNodeClick={handleNodeClick}
        linkSource="source"
        linkTarget="target"
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkOpacity={0.4}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={1}
        linkDirectionalParticleSpeed={0.005}
        backgroundColor="rgba(0, 0, 0, 0)"
        showNavInfo={false}
        enableNodeDrag={true}
        enableNavigationControls={true}
        controlType="orbit"
        warmupTicks={50}
        cooldownTicks={100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
      />

      {/* Legend */}
      <div className="absolute top-4 left-4 bg-background/90 backdrop-blur-sm border rounded-lg p-3 space-y-1 z-10">
        <p className="text-xs font-medium text-muted-foreground mb-2">Entity Types (3D)</p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {Object.entries(entityTypeColors)
            .slice(0, 6)
            .map(([type, color]) => (
              <div key={type} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-xs">{type}</span>
              </div>
            ))}
        </div>
      </div>

      {/* Controls hint */}
      <div className="absolute bottom-4 left-4 text-xs text-muted-foreground bg-background/80 backdrop-blur-sm px-2 py-1 rounded z-10">
        Drag to rotate | Scroll to zoom | Click nodes for details
      </div>
    </div>
  );
}
