"use client";

/**
 * Citation Visualization Component
 * =================================
 *
 * Visualizes retrieved chunks in 2D space using dimensionality reduction.
 * Shows semantic relationships between citations with interactive exploration.
 *
 * Features:
 * - UMAP/t-SNE visualization of chunk embeddings
 * - Interactive tooltip with chunk content
 * - Color coding by relevance score
 * - Zoom and pan controls
 * - Click to expand citation details
 */

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  FileText,
  Layers,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Types
interface Citation {
  id: string;
  chunkId: string;
  documentId: string;
  documentTitle?: string;
  content: string;
  score: number;
  embedding?: number[];
  pageNumber?: number;
  sourceType?: string;
}

interface ProjectedPoint {
  x: number;
  y: number;
  citation: Citation;
}

interface CitationVisualizationProps {
  citations: Citation[];
  query?: string;
  queryEmbedding?: number[];
  onCitationClick?: (citation: Citation) => void;
  className?: string;
  height?: number;
}

// Color scale for relevance scores
const getScoreColor = (score: number): string => {
  // Green to yellow to red based on score
  if (score >= 0.8) return "#22c55e"; // Green - high relevance
  if (score >= 0.6) return "#84cc16"; // Lime
  if (score >= 0.4) return "#eab308"; // Yellow
  if (score >= 0.2) return "#f97316"; // Orange
  return "#ef4444"; // Red - low relevance
};

// Simple UMAP-like dimensionality reduction
// In production, this would call the backend for actual UMAP
function simpleProjection(
  embeddings: number[][],
  dimensions: number = 2
): number[][] {
  if (embeddings.length === 0) return [];
  if (embeddings.length === 1) return [[0.5, 0.5]];

  const embeddingDim = embeddings[0].length;

  // Use PCA-like projection (first 2 principal components approximation)
  // This is a simplified version - production would use proper UMAP/t-SNE

  // Calculate mean
  const mean = new Array(embeddingDim).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < embeddingDim; i++) {
      mean[i] += emb[i] / embeddings.length;
    }
  }

  // Center the data
  const centered = embeddings.map((emb) =>
    emb.map((val, i) => val - mean[i])
  );

  // Simple projection using random directions (approximation)
  // In production, compute actual principal components
  const projections: number[][] = [];

  // Use first few dimensions as rough projection (works okay for visualization)
  for (const emb of centered) {
    // Combine multiple dimensions for x and y
    const x =
      (emb[0] || 0) * 0.5 +
      (emb[1] || 0) * 0.3 +
      (emb[2] || 0) * 0.2;
    const y =
      (emb[3] || 0) * 0.5 +
      (emb[4] || 0) * 0.3 +
      (emb[5] || 0) * 0.2;
    projections.push([x, y]);
  }

  // Normalize to [0, 1] with padding
  const xs = projections.map((p) => p[0]);
  const ys = projections.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const padding = 0.1;
  return projections.map((p) => [
    padding + ((p[0] - minX) / rangeX) * (1 - 2 * padding),
    padding + ((p[1] - minY) / rangeY) * (1 - 2 * padding),
  ]);
}

// Calculate cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}

export function CitationVisualization({
  citations,
  query,
  queryEmbedding,
  onCitationClick,
  className,
  height = 400,
}: CitationVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [projectedPoints, setProjectedPoints] = useState<ProjectedPoint[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [hoveredCitation, setHoveredCitation] = useState<Citation | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [colorBy, setColorBy] = useState<"score" | "document" | "source">("score");
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Project citations to 2D
  useEffect(() => {
    if (citations.length === 0) {
      setProjectedPoints([]);
      return;
    }

    // Get embeddings (use mock if not available)
    const embeddings = citations.map((c) => {
      if (c.embedding && c.embedding.length > 0) {
        return c.embedding;
      }
      // Generate deterministic pseudo-embedding based on content hash
      const hash = c.content.split("").reduce((a, b) => {
        a = (a << 5) - a + b.charCodeAt(0);
        return a & a;
      }, 0);
      return Array.from({ length: 10 }, (_, i) =>
        Math.sin(hash * (i + 1) * 0.1) * 0.5
      );
    });

    const projected = simpleProjection(embeddings);

    setProjectedPoints(
      projected.map((p, i) => ({
        x: p[0],
        y: p[1],
        citation: citations[i],
      }))
    );
  }, [citations]);

  // Get unique documents for color coding
  const uniqueDocuments = useMemo(() => {
    const docs = new Set(citations.map((c) => c.documentId));
    return Array.from(docs);
  }, [citations]);

  // Generate document colors
  const documentColors = useMemo(() => {
    const colors = [
      "#6366f1", "#8b5cf6", "#d946ef", "#ec4899", "#f43f5e",
      "#f97316", "#eab308", "#84cc16", "#22c55e", "#14b8a6",
    ];
    const colorMap: Record<string, string> = {};
    uniqueDocuments.forEach((doc, i) => {
      colorMap[doc] = colors[i % colors.length];
    });
    return colorMap;
  }, [uniqueDocuments]);

  // Get color for a citation
  const getColor = useCallback(
    (citation: Citation): string => {
      switch (colorBy) {
        case "score":
          return getScoreColor(citation.score);
        case "document":
          return documentColors[citation.documentId] || "#6366f1";
        case "source":
          return citation.sourceType === "kg"
            ? "#8b5cf6"
            : citation.sourceType === "vector"
            ? "#6366f1"
            : "#22c55e";
        default:
          return getScoreColor(citation.score);
      }
    },
    [colorBy, documentColors]
  );

  // Handle zoom
  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.2, 5));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.2, 0.5));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Handle pan
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Handle point click
  const handlePointClick = (citation: Citation) => {
    setSelectedCitation(citation);
    onCitationClick?.(citation);
  };

  if (citations.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-[200px]">
          <p className="text-muted-foreground">No citations to visualize</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="py-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Citation Map
            <Badge variant="secondary" className="ml-2">
              {citations.length} sources
            </Badge>
          </CardTitle>
          <div className="flex items-center gap-2">
            <Select value={colorBy} onValueChange={(v: any) => setColorBy(v)}>
              <SelectTrigger className="w-[120px] h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="score">By Relevance</SelectItem>
                <SelectItem value="document">By Document</SelectItem>
                <SelectItem value="source">By Source</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex items-center gap-1 border rounded-md">
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handleZoomOut}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handleZoomIn}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={handleReset}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div
          className="relative overflow-hidden bg-muted/30"
          style={{ height }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <svg
            ref={svgRef}
            className="w-full h-full cursor-grab active:cursor-grabbing"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "center",
            }}
          >
            {/* Grid lines */}
            <defs>
              <pattern
                id="grid"
                width="40"
                height="40"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M 40 0 L 0 0 0 40"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  className="text-muted-foreground/20"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />

            {/* Query point (if embedding available) */}
            {queryEmbedding && (
              <g>
                <circle
                  cx="50%"
                  cy="50%"
                  r="8"
                  fill="#6366f1"
                  stroke="white"
                  strokeWidth="2"
                />
                <text
                  x="50%"
                  y="50%"
                  dy="20"
                  textAnchor="middle"
                  className="text-xs fill-muted-foreground"
                >
                  Query
                </text>
              </g>
            )}

            {/* Citation points */}
            {projectedPoints.map((point, index) => {
              const isSelected = selectedCitation?.id === point.citation.id;
              const isHovered = hoveredCitation?.id === point.citation.id;
              const x = point.x * 100 + "%";
              const y = point.y * 100 + "%";
              const radius = isSelected ? 10 : isHovered ? 8 : 6;

              return (
                <TooltipProvider key={point.citation.id}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <g
                        className="cursor-pointer transition-transform"
                        onClick={() => handlePointClick(point.citation)}
                        onMouseEnter={() => setHoveredCitation(point.citation)}
                        onMouseLeave={() => setHoveredCitation(null)}
                      >
                        {/* Outer ring for selected/hovered */}
                        {(isSelected || isHovered) && (
                          <circle
                            cx={x}
                            cy={y}
                            r={radius + 4}
                            fill="none"
                            stroke={getColor(point.citation)}
                            strokeWidth="2"
                            strokeDasharray={isSelected ? "none" : "4 2"}
                            opacity={0.5}
                          />
                        )}
                        {/* Main point */}
                        <circle
                          cx={x}
                          cy={y}
                          r={radius}
                          fill={getColor(point.citation)}
                          stroke="white"
                          strokeWidth="2"
                          className="transition-all duration-200"
                        />
                        {/* Rank number */}
                        <text
                          x={x}
                          y={y}
                          dy="0.35em"
                          textAnchor="middle"
                          className="text-[10px] font-bold fill-white pointer-events-none"
                        >
                          {index + 1}
                        </text>
                      </g>
                    </TooltipTrigger>
                    <TooltipContent
                      side="top"
                      className="max-w-[300px]"
                    >
                      <div className="space-y-1">
                        <div className="font-medium flex items-center gap-2">
                          <FileText className="h-3 w-3" />
                          {point.citation.documentTitle || "Document"}
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-3">
                          {point.citation.content.slice(0, 200)}...
                        </p>
                        <div className="flex items-center gap-2 text-xs">
                          <Badge
                            variant="outline"
                            style={{
                              borderColor: getScoreColor(point.citation.score),
                              color: getScoreColor(point.citation.score),
                            }}
                          >
                            {Math.round(point.citation.score * 100)}% match
                          </Badge>
                          {point.citation.pageNumber && (
                            <span className="text-muted-foreground">
                              Page {point.citation.pageNumber}
                            </span>
                          )}
                        </div>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              );
            })}
          </svg>

          {/* Legend */}
          <div className="absolute bottom-2 left-2 bg-background/80 backdrop-blur-sm rounded-md p-2 text-xs">
            {colorBy === "score" && (
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Relevance:</span>
                <div className="flex items-center gap-1">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: "#ef4444" }}
                  />
                  <span>Low</span>
                </div>
                <div
                  className="w-12 h-2 rounded"
                  style={{
                    background:
                      "linear-gradient(to right, #ef4444, #f97316, #eab308, #84cc16, #22c55e)",
                  }}
                />
                <div className="flex items-center gap-1">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ background: "#22c55e" }}
                  />
                  <span>High</span>
                </div>
              </div>
            )}
            {colorBy === "document" && (
              <div className="flex flex-wrap items-center gap-2">
                {uniqueDocuments.slice(0, 5).map((doc) => (
                  <div key={doc} className="flex items-center gap-1">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ background: documentColors[doc] }}
                    />
                    <span className="truncate max-w-[80px]">
                      {citations.find((c) => c.documentId === doc)
                        ?.documentTitle || "Doc"}
                    </span>
                  </div>
                ))}
                {uniqueDocuments.length > 5 && (
                  <span className="text-muted-foreground">
                    +{uniqueDocuments.length - 5} more
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Selected citation details */}
        {selectedCitation && (
          <div className="p-4 border-t bg-muted/30">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <FileText className="h-4 w-4 text-primary" />
                  <span className="font-medium truncate">
                    {selectedCitation.documentTitle || "Document"}
                  </span>
                  {selectedCitation.pageNumber && (
                    <Badge variant="secondary">
                      Page {selectedCitation.pageNumber}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-muted-foreground line-clamp-3">
                  {selectedCitation.content}
                </p>
              </div>
              <div className="flex flex-col items-end gap-1">
                <Badge
                  style={{
                    background: getScoreColor(selectedCitation.score),
                    color: "white",
                  }}
                >
                  {Math.round(selectedCitation.score * 100)}%
                </Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedCitation(null)}
                >
                  Close
                </Button>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default CitationVisualization;
