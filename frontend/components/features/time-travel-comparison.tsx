"use client";

/**
 * AIDocumentIndexer - Time Travel Document Comparison
 * =====================================================
 *
 * Phase 20: Differentiation Features - Unique feature nobody else has.
 *
 * Time Travel allows users to:
 * - Compare how their knowledge base has evolved over time
 * - See what information was added, removed, or changed
 * - Query against historical snapshots
 * - Track document version changes
 *
 * Features:
 * - Timeline visualization of document changes
 * - Side-by-side comparison view
 * - Diff highlighting for text changes
 * - Historical query execution
 * - Change impact analysis
 */

import * as React from "react";
import { useState, useEffect, useMemo } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Calendar,
  ChevronLeft,
  ChevronRight,
  Clock,
  Diff,
  Eye,
  FileText,
  GitBranch,
  History,
  Minus,
  Plus,
  RefreshCw,
  Search,
  SplitSquareHorizontal,
} from "lucide-react";
import { format, formatDistanceToNow } from "date-fns";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";

// =============================================================================
// Types
// =============================================================================

export interface DocumentSnapshot {
  id: string;
  timestamp: string;
  documentCount: number;
  totalChunks: number;
  changes: {
    added: number;
    modified: number;
    removed: number;
  };
  documents: SnapshotDocument[];
}

export interface SnapshotDocument {
  documentId: string;
  filename: string;
  version: number;
  status: "added" | "modified" | "removed" | "unchanged";
  chunkCount: number;
  lastModified: string;
  contentPreview?: string;
}

export interface DocumentDiff {
  documentId: string;
  filename: string;
  beforeVersion: number;
  afterVersion: number;
  additions: string[];
  deletions: string[];
  modifications: DiffChunk[];
}

export interface DiffChunk {
  type: "context" | "addition" | "deletion" | "modification";
  before?: string;
  after?: string;
  lineNumber: number;
}

interface TimeTravelComparisonProps {
  organizationId: string;
  className?: string;
}

// =============================================================================
// Mock Data (would come from API in real implementation)
// =============================================================================

const mockSnapshots: DocumentSnapshot[] = [
  {
    id: "snap-001",
    timestamp: new Date(Date.now() - 86400000 * 7).toISOString(),
    documentCount: 45,
    totalChunks: 1250,
    changes: { added: 45, modified: 0, removed: 0 },
    documents: [],
  },
  {
    id: "snap-002",
    timestamp: new Date(Date.now() - 86400000 * 5).toISOString(),
    documentCount: 52,
    totalChunks: 1480,
    changes: { added: 7, modified: 3, removed: 0 },
    documents: [],
  },
  {
    id: "snap-003",
    timestamp: new Date(Date.now() - 86400000 * 3).toISOString(),
    documentCount: 58,
    totalChunks: 1620,
    changes: { added: 8, modified: 5, removed: 2 },
    documents: [],
  },
  {
    id: "snap-004",
    timestamp: new Date(Date.now() - 86400000).toISOString(),
    documentCount: 65,
    totalChunks: 1890,
    changes: { added: 10, modified: 2, removed: 3 },
    documents: [],
  },
  {
    id: "snap-005",
    timestamp: new Date().toISOString(),
    documentCount: 72,
    totalChunks: 2100,
    changes: { added: 9, modified: 4, removed: 2 },
    documents: [],
  },
];

// =============================================================================
// Timeline Component
// =============================================================================

function Timeline({
  snapshots,
  selectedIndex,
  onSelect,
}: {
  snapshots: DocumentSnapshot[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="relative py-4">
      {/* Timeline line */}
      <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-muted -translate-y-1/2" />

      {/* Timeline points */}
      <div className="relative flex justify-between">
        {snapshots.map((snapshot, index) => {
          const isSelected = index === selectedIndex;
          const hasChanges = snapshot.changes.added > 0 ||
            snapshot.changes.modified > 0 ||
            snapshot.changes.removed > 0;

          return (
            <TooltipProvider key={snapshot.id}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    onClick={() => onSelect(index)}
                    className={cn(
                      "relative flex flex-col items-center gap-2 transition-all",
                      isSelected && "scale-110"
                    )}
                  >
                    <div
                      className={cn(
                        "h-4 w-4 rounded-full border-2 transition-colors",
                        isSelected
                          ? "bg-primary border-primary"
                          : hasChanges
                          ? "bg-background border-primary"
                          : "bg-muted border-muted-foreground/30"
                      )}
                    />
                    <span
                      className={cn(
                        "text-xs whitespace-nowrap",
                        isSelected ? "font-medium" : "text-muted-foreground"
                      )}
                    >
                      {format(new Date(snapshot.timestamp), "MMM d")}
                    </span>
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="text-sm">
                    <p className="font-medium">
                      {format(new Date(snapshot.timestamp), "PPP")}
                    </p>
                    <p className="text-muted-foreground">
                      {snapshot.documentCount} documents
                    </p>
                    {hasChanges && (
                      <div className="flex gap-2 mt-1">
                        {snapshot.changes.added > 0 && (
                          <span className="text-green-500">+{snapshot.changes.added}</span>
                        )}
                        {snapshot.changes.modified > 0 && (
                          <span className="text-yellow-500">~{snapshot.changes.modified}</span>
                        )}
                        {snapshot.changes.removed > 0 && (
                          <span className="text-red-500">-{snapshot.changes.removed}</span>
                        )}
                      </div>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// Comparison View
// =============================================================================

function ComparisonView({
  before,
  after,
}: {
  before: DocumentSnapshot;
  after: DocumentSnapshot;
}) {
  const [viewMode, setViewMode] = useState<"split" | "unified">("split");

  const stats = useMemo(() => ({
    docsAdded: after.documentCount - before.documentCount + after.changes.removed,
    docsRemoved: after.changes.removed,
    chunksAdded: after.totalChunks - before.totalChunks,
    timeDiff: formatDistanceToNow(new Date(before.timestamp), { addSuffix: false }),
  }), [before, after]);

  return (
    <div className="space-y-4">
      {/* Stats Banner */}
      <div className="grid grid-cols-4 gap-4">
        <StatBox
          label="Documents Added"
          value={stats.docsAdded}
          icon={<Plus className="h-4 w-4" />}
          variant="success"
        />
        <StatBox
          label="Documents Removed"
          value={stats.docsRemoved}
          icon={<Minus className="h-4 w-4" />}
          variant="danger"
        />
        <StatBox
          label="Chunks Changed"
          value={Math.abs(stats.chunksAdded)}
          icon={<Diff className="h-4 w-4" />}
          variant={stats.chunksAdded >= 0 ? "success" : "danger"}
        />
        <StatBox
          label="Time Span"
          value={stats.timeDiff}
          icon={<Clock className="h-4 w-4" />}
          variant="neutral"
        />
      </div>

      {/* View Toggle */}
      <div className="flex items-center justify-between">
        <h3 className="font-medium">Document Changes</h3>
        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === "split" ? "default" : "outline"}
            size="sm"
            onClick={() => setViewMode("split")}
          >
            <SplitSquareHorizontal className="h-4 w-4 mr-1" />
            Split
          </Button>
          <Button
            variant={viewMode === "unified" ? "default" : "outline"}
            size="sm"
            onClick={() => setViewMode("unified")}
          >
            <Diff className="h-4 w-4 mr-1" />
            Unified
          </Button>
        </div>
      </div>

      {/* Comparison Content */}
      {viewMode === "split" ? (
        <SplitView before={before} after={after} />
      ) : (
        <UnifiedView before={before} after={after} />
      )}
    </div>
  );
}

function StatBox({
  label,
  value,
  icon,
  variant,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  variant: "success" | "danger" | "neutral";
}) {
  const colors = {
    success: "text-green-500 bg-green-500/10",
    danger: "text-red-500 bg-red-500/10",
    neutral: "text-muted-foreground bg-muted",
  };

  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
      <div className={cn("p-2 rounded-lg", colors[variant])}>
        {icon}
      </div>
      <div>
        <p className="text-xl font-bold">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}

function SplitView({
  before,
  after,
}: {
  before: DocumentSnapshot;
  after: DocumentSnapshot;
}) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <Card>
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium">
              {format(new Date(before.timestamp), "PPP")}
            </CardTitle>
            <Badge variant="outline">{before.documentCount} docs</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[300px]">
            <div className="space-y-2">
              {/* Example document list */}
              {Array.from({ length: before.documentCount }).slice(0, 10).map((_, i) => (
                <div key={i} className="flex items-center gap-2 p-2 rounded bg-muted/50">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Document_{i + 1}.pdf</span>
                </div>
              ))}
              {before.documentCount > 10 && (
                <p className="text-sm text-muted-foreground text-center py-2">
                  +{before.documentCount - 10} more documents
                </p>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium">
              {format(new Date(after.timestamp), "PPP")}
            </CardTitle>
            <Badge variant="outline">{after.documentCount} docs</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[300px]">
            <div className="space-y-2">
              {/* Show added documents first */}
              {Array.from({ length: after.changes.added }).map((_, i) => (
                <div key={`added-${i}`} className="flex items-center gap-2 p-2 rounded bg-green-500/10">
                  <Plus className="h-4 w-4 text-green-500" />
                  <span className="text-sm">New_Document_{i + 1}.pdf</span>
                  <Badge variant="outline" className="ml-auto text-green-500">Added</Badge>
                </div>
              ))}
              {/* Show modified documents */}
              {Array.from({ length: after.changes.modified }).map((_, i) => (
                <div key={`modified-${i}`} className="flex items-center gap-2 p-2 rounded bg-yellow-500/10">
                  <RefreshCw className="h-4 w-4 text-yellow-500" />
                  <span className="text-sm">Modified_Doc_{i + 1}.pdf</span>
                  <Badge variant="outline" className="ml-auto text-yellow-500">Modified</Badge>
                </div>
              ))}
              {/* Show remaining documents */}
              {Array.from({ length: Math.max(0, after.documentCount - after.changes.added - after.changes.modified) })
                .slice(0, 5)
                .map((_, i) => (
                  <div key={`unchanged-${i}`} className="flex items-center gap-2 p-2 rounded bg-muted/50">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm">Document_{i + 1}.pdf</span>
                  </div>
                ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

function UnifiedView({
  before,
  after,
}: {
  before: DocumentSnapshot;
  after: DocumentSnapshot;
}) {
  return (
    <Card>
      <CardContent className="pt-4">
        <ScrollArea className="h-[400px]">
          <div className="space-y-1 font-mono text-sm">
            {/* Removed documents */}
            {Array.from({ length: after.changes.removed }).map((_, i) => (
              <div key={`removed-${i}`} className="flex items-center gap-2 px-3 py-1 bg-red-500/10 text-red-600">
                <Minus className="h-3 w-3" />
                <span>- Removed_Document_{i + 1}.pdf</span>
              </div>
            ))}

            {/* Added documents */}
            {Array.from({ length: after.changes.added }).map((_, i) => (
              <div key={`added-${i}`} className="flex items-center gap-2 px-3 py-1 bg-green-500/10 text-green-600">
                <Plus className="h-3 w-3" />
                <span>+ New_Document_{i + 1}.pdf</span>
              </div>
            ))}

            {/* Modified documents */}
            {Array.from({ length: after.changes.modified }).map((_, i) => (
              <div key={`modified-${i}`} className="flex items-center gap-2 px-3 py-1 bg-yellow-500/10 text-yellow-600">
                <RefreshCw className="h-3 w-3" />
                <span>~ Modified_Document_{i + 1}.pdf (v2 → v3)</span>
              </div>
            ))}

            {/* Unchanged context */}
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={`context-${i}`} className="flex items-center gap-2 px-3 py-1 text-muted-foreground">
                <span className="w-3" />
                <span>  Document_{i + 10}.pdf</span>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Historical Query
// =============================================================================

function HistoricalQuery({
  snapshot,
}: {
  snapshot: DocumentSnapshot;
}) {
  const [query, setQuery] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);

  const handleQuery = async () => {
    if (!query.trim()) return;
    setIsQuerying(true);
    // Simulate query
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setIsQuerying(false);
  };

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <History className="h-4 w-4" />
          Query Against Historical Snapshot
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Input
            placeholder="Ask a question about this version..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleQuery()}
          />
          <Button onClick={handleQuery} disabled={isQuerying}>
            {isQuerying ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Search className="h-4 w-4" />
            )}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Query using the knowledge base as it existed on{" "}
          {format(new Date(snapshot.timestamp), "PPP")}
        </p>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function TimeTravelComparison({
  organizationId,
  className,
}: TimeTravelComparisonProps) {
  const [snapshots, setSnapshots] = useState<DocumentSnapshot[]>(mockSnapshots);
  const [selectedIndex, setSelectedIndex] = useState(snapshots.length - 1);
  const [compareIndex, setCompareIndex] = useState(0);
  const [isComparing, setIsComparing] = useState(false);

  const selectedSnapshot = snapshots[selectedIndex];
  const compareSnapshot = snapshots[compareIndex];

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <GitBranch className="h-6 w-6" />
            Time Travel
          </h2>
          <p className="text-muted-foreground">
            Compare how your knowledge base has evolved over time
          </p>
        </div>
        <Button
          variant={isComparing ? "default" : "outline"}
          onClick={() => setIsComparing(!isComparing)}
        >
          <Diff className="h-4 w-4 mr-2" />
          {isComparing ? "Exit Compare" : "Compare Versions"}
        </Button>
      </div>

      {/* Timeline */}
      <Card>
        <CardContent className="pt-6">
          <Timeline
            snapshots={snapshots}
            selectedIndex={isComparing ? compareIndex : selectedIndex}
            onSelect={isComparing ? setCompareIndex : setSelectedIndex}
          />

          {isComparing && (
            <div className="flex items-center justify-center gap-4 mt-4 pt-4 border-t">
              <div className="text-sm">
                <span className="text-muted-foreground">Comparing: </span>
                <Badge variant="outline">
                  {format(new Date(compareSnapshot.timestamp), "MMM d")}
                </Badge>
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              <div className="text-sm">
                <Badge variant="outline">
                  {format(new Date(selectedSnapshot.timestamp), "MMM d")}
                </Badge>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Comparison or Single View */}
      {isComparing ? (
        <ComparisonView before={compareSnapshot} after={selectedSnapshot} />
      ) : (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Snapshot Details */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                Snapshot Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="space-y-3">
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Date</dt>
                  <dd className="font-medium">
                    {format(new Date(selectedSnapshot.timestamp), "PPPp")}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Documents</dt>
                  <dd className="font-medium">{selectedSnapshot.documentCount}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Total Chunks</dt>
                  <dd className="font-medium">{selectedSnapshot.totalChunks.toLocaleString()}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Changes</dt>
                  <dd className="flex gap-2">
                    <Badge variant="outline" className="text-green-500">
                      +{selectedSnapshot.changes.added}
                    </Badge>
                    <Badge variant="outline" className="text-yellow-500">
                      ~{selectedSnapshot.changes.modified}
                    </Badge>
                    <Badge variant="outline" className="text-red-500">
                      -{selectedSnapshot.changes.removed}
                    </Badge>
                  </dd>
                </div>
              </dl>
            </CardContent>
          </Card>

          {/* Historical Query */}
          <HistoricalQuery snapshot={selectedSnapshot} />
        </div>
      )}

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Button
          variant="outline"
          disabled={selectedIndex === 0}
          onClick={() => setSelectedIndex((prev) => Math.max(0, prev - 1))}
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          Previous
        </Button>
        <span className="text-sm text-muted-foreground">
          {selectedIndex + 1} of {snapshots.length} snapshots
        </span>
        <Button
          variant="outline"
          disabled={selectedIndex === snapshots.length - 1}
          onClick={() =>
            setSelectedIndex((prev) => Math.min(snapshots.length - 1, prev + 1))
          }
        >
          Next
          <ChevronRight className="h-4 w-4 ml-1" />
        </Button>
      </div>
    </div>
  );
}

export default TimeTravelComparison;
