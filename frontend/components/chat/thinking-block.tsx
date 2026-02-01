"use client";

import React, { useState, useMemo } from "react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  ChevronRight,
  Check,
  Loader2,
  SkipForward,
  Clock,
  Brain,
  Sparkles,
  Search,
  FileText,
  Database,
  Globe,
  Terminal,
  Zap,
  AlertCircle,
  XCircle,
  PauseCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface PipelineStep {
  name: string;
  status: "pending" | "running" | "completed" | "skipped" | "error";
  durationMs?: number;
  details?: string;
  substeps?: PipelineStep[];
}

export interface ToolInvocation {
  id: string;
  name: string;
  type: "search" | "retrieval" | "database" | "web" | "code" | "file" | "analysis" | "other";
  status: "pending" | "running" | "streaming" | "completed" | "error" | "cancelled";
  input?: string;
  output?: string;
  streamingText?: string;
  durationMs?: number;
  progress?: number;
}

interface ThinkingBlockProps {
  steps: PipelineStep[];
  toolInvocations?: ToolInvocation[];
  totalTime?: number;
  isStreaming?: boolean;
  showToolDetails?: boolean;
}

function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

const TOOL_ICONS: Record<ToolInvocation["type"], React.ReactNode> = {
  search: <Search className="h-3 w-3" />,
  retrieval: <FileText className="h-3 w-3" />,
  database: <Database className="h-3 w-3" />,
  web: <Globe className="h-3 w-3" />,
  code: <Terminal className="h-3 w-3" />,
  file: <FileText className="h-3 w-3" />,
  analysis: <Brain className="h-3 w-3" />,
  other: <Zap className="h-3 w-3" />,
};

const TOOL_COLORS: Record<ToolInvocation["type"], string> = {
  search: "text-blue-500",
  retrieval: "text-purple-500",
  database: "text-orange-500",
  web: "text-green-500",
  code: "text-yellow-500",
  file: "text-cyan-500",
  analysis: "text-pink-500",
  other: "text-gray-500",
};

function StepStatusIcon({ status }: { status: PipelineStep["status"] }) {
  switch (status) {
    case "completed":
      return (
        <div className="flex h-4 w-4 items-center justify-center rounded-full bg-green-500/15">
          <Check className="h-3 w-3 text-green-600 dark:text-green-400" />
        </div>
      );
    case "running":
      return (
        <div className="flex h-4 w-4 items-center justify-center">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-600 dark:text-blue-400" />
        </div>
      );
    case "skipped":
      return (
        <div className="flex h-4 w-4 items-center justify-center">
          <SkipForward className="h-3 w-3 text-muted-foreground" />
        </div>
      );
    case "error":
      return (
        <div className="flex h-4 w-4 items-center justify-center rounded-full bg-red-500/15">
          <XCircle className="h-3 w-3 text-red-600 dark:text-red-400" />
        </div>
      );
    case "pending":
    default:
      return (
        <div className="flex h-4 w-4 items-center justify-center">
          <div className="h-2.5 w-2.5 rounded-full border-2 border-muted-foreground/40" />
        </div>
      );
  }
}

function ToolStatusIcon({ status }: { status: ToolInvocation["status"] }) {
  switch (status) {
    case "completed":
      return <Check className="h-3 w-3 text-green-500" />;
    case "running":
      return <Loader2 className="h-3 w-3 animate-spin text-blue-500" />;
    case "streaming":
      return <Zap className="h-3 w-3 animate-pulse text-yellow-500" />;
    case "error":
      return <XCircle className="h-3 w-3 text-red-500" />;
    case "cancelled":
      return <PauseCircle className="h-3 w-3 text-orange-500" />;
    case "pending":
    default:
      return <Clock className="h-3 w-3 text-muted-foreground" />;
  }
}

function StreamingDots() {
  return (
    <span className="inline-flex items-center gap-0.5 ml-1">
      <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:0ms]" />
      <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:150ms]" />
      <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:300ms]" />
    </span>
  );
}

function StreamingCursor() {
  return (
    <span className="inline-block w-1.5 h-3 bg-primary/70 animate-pulse ml-0.5 rounded-sm" />
  );
}

function ToolInvocationItem({
  tool,
  showDetails = false,
}: {
  tool: ToolInvocation;
  showDetails?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const isActive = tool.status === "running" || tool.status === "streaming";
  const hasOutput = tool.output || tool.streamingText;

  return (
    <div
      className={cn(
        "rounded-md border transition-all",
        isActive && "border-primary/30 bg-primary/5",
        tool.status === "error" && "border-red-500/30 bg-red-500/5"
      )}
    >
      <button
        type="button"
        onClick={() => showDetails && hasOutput && setExpanded(!expanded)}
        className={cn(
          "flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs",
          showDetails && hasOutput && "cursor-pointer hover:bg-muted/50"
        )}
      >
        <span className={cn("flex-shrink-0", TOOL_COLORS[tool.type])}>
          {TOOL_ICONS[tool.type]}
        </span>

        <span className="flex-1 truncate font-medium">{tool.name}</span>

        <div className="flex items-center gap-2">
          {isActive && tool.progress !== undefined && (
            <div className="w-12">
              <Progress value={tool.progress} className="h-1" />
            </div>
          )}

          <ToolStatusIcon status={tool.status} />

          {tool.durationMs !== undefined && (
            <span className="text-muted-foreground tabular-nums text-[10px]">
              {formatDuration(tool.durationMs)}
            </span>
          )}

          {showDetails && hasOutput && (
            <ChevronRight
              className={cn(
                "h-3 w-3 text-muted-foreground transition-transform",
                expanded && "rotate-90"
              )}
            />
          )}
        </div>
      </button>

      {/* Expanded Details */}
      {expanded && showDetails && (
        <div className="border-t px-2 py-1.5">
          {tool.input && (
            <div className="mb-1">
              <span className="text-[10px] text-muted-foreground font-medium">Input:</span>
              <div className="text-[10px] text-foreground/80 font-mono mt-0.5 truncate">
                {tool.input}
              </div>
            </div>
          )}

          {(tool.output || tool.streamingText) && (
            <div>
              <span className="text-[10px] text-muted-foreground font-medium">Output:</span>
              <div className="text-[10px] text-foreground/80 font-mono mt-0.5 max-h-20 overflow-auto whitespace-pre-wrap">
                {tool.status === "streaming" ? (
                  <>
                    {tool.streamingText}
                    <StreamingCursor />
                  </>
                ) : (
                  tool.output
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ThinkingBlock({
  steps,
  toolInvocations = [],
  totalTime,
  isStreaming = false,
  showToolDetails = true,
}: ThinkingBlockProps) {
  const [isOpen, setIsOpen] = useState(false);

  const activeTools = toolInvocations.filter(
    (t) => t.status === "running" || t.status === "streaming"
  );
  const completedTools = toolInvocations.filter((t) => t.status === "completed");
  const errorTools = toolInvocations.filter((t) => t.status === "error");

  const headerLabel = useMemo(() => {
    if (isStreaming) {
      if (activeTools.length > 0) {
        return `Processing with ${activeTools.length} tool${activeTools.length > 1 ? "s" : ""}`;
      }
      return "Thinking";
    }
    if (totalTime != null) {
      return `Thought for ${formatDuration(totalTime)}`;
    }
    const computedTotal = steps.reduce((sum, step) => {
      if (step.status === "completed" && step.durationMs != null) {
        return sum + step.durationMs;
      }
      return sum;
    }, 0);
    if (computedTotal > 0) {
      return `Thought for ${formatDuration(computedTotal)}`;
    }
    return "Thought";
  }, [isStreaming, totalTime, steps, activeTools.length]);

  const showToolSummary = toolInvocations.length > 0;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="rounded-lg border bg-muted/50 text-sm">
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className={cn(
              "flex w-full items-center gap-2 px-3 py-2 text-left text-sm",
              "text-muted-foreground hover:text-foreground transition-colors",
              "rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            )}
          >
            <ChevronRight
              className={cn(
                "h-3.5 w-3.5 shrink-0 transition-transform duration-200",
                isOpen && "rotate-90"
              )}
            />

            {isStreaming ? (
              <Brain className="h-3.5 w-3.5 shrink-0 text-primary animate-pulse" />
            ) : (
              <Clock className="h-3.5 w-3.5 shrink-0" />
            )}

            <span className="text-xs font-medium flex-1">
              {headerLabel}
              {isStreaming && <StreamingDots />}
            </span>

            {/* Tool Summary Badges */}
            {showToolSummary && (
              <div className="flex items-center gap-1.5">
                {activeTools.length > 0 && (
                  <Badge
                    variant="outline"
                    className="h-5 px-1.5 text-[10px] bg-blue-500/10 text-blue-500 border-blue-500/30"
                  >
                    <Loader2 className="h-2.5 w-2.5 mr-0.5 animate-spin" />
                    {activeTools.length}
                  </Badge>
                )}
                {completedTools.length > 0 && (
                  <Badge
                    variant="outline"
                    className="h-5 px-1.5 text-[10px] bg-green-500/10 text-green-500 border-green-500/30"
                  >
                    <Check className="h-2.5 w-2.5 mr-0.5" />
                    {completedTools.length}
                  </Badge>
                )}
                {errorTools.length > 0 && (
                  <Badge
                    variant="outline"
                    className="h-5 px-1.5 text-[10px] bg-red-500/10 text-red-500 border-red-500/30"
                  >
                    <AlertCircle className="h-2.5 w-2.5 mr-0.5" />
                    {errorTools.length}
                  </Badge>
                )}
              </div>
            )}
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="border-t px-3 pb-3 pt-2 space-y-3">
            {/* Pipeline Steps */}
            {steps.length > 0 && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground mb-1.5">
                  Pipeline Steps
                </div>
                <ul className="space-y-1.5">
                  {steps.map((step, index) => (
                    <li
                      key={index}
                      className="flex items-center justify-between gap-3 text-xs"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <StepStatusIcon status={step.status} />
                        <span
                          className={cn(
                            "truncate",
                            step.status === "skipped"
                              ? "text-muted-foreground"
                              : step.status === "running"
                                ? "text-blue-600 dark:text-blue-400"
                                : step.status === "error"
                                  ? "text-red-600 dark:text-red-400"
                                  : "text-foreground"
                          )}
                        >
                          {step.name}
                        </span>
                      </div>
                      <span className="shrink-0 tabular-nums text-muted-foreground">
                        {step.status === "skipped"
                          ? "skipped"
                          : step.status === "error"
                            ? "error"
                            : step.durationMs != null
                              ? formatDuration(step.durationMs)
                              : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Tool Invocations */}
            {toolInvocations.length > 0 && (
              <div>
                <div className="text-[10px] font-medium text-muted-foreground mb-1.5 flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  Tool Invocations
                </div>
                <div className="space-y-1.5">
                  {toolInvocations.map((tool) => (
                    <ToolInvocationItem
                      key={tool.id}
                      tool={tool}
                      showDetails={showToolDetails}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Total Time */}
            {totalTime != null && (steps.length > 0 || toolInvocations.length > 0) && (
              <div className="flex justify-end border-t pt-2">
                <span className="text-xs tabular-nums text-muted-foreground">
                  Total: {formatDuration(totalTime)}
                </span>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

// Enhanced version with real-time streaming support
export function StreamingThinkingBlock({
  initialSteps = [],
  initialTools = [],
  onStepsChange,
  onToolsChange,
}: {
  initialSteps?: PipelineStep[];
  initialTools?: ToolInvocation[];
  onStepsChange?: (steps: PipelineStep[]) => void;
  onToolsChange?: (tools: ToolInvocation[]) => void;
}) {
  const [steps, setSteps] = useState<PipelineStep[]>(initialSteps);
  const [tools, setTools] = useState<ToolInvocation[]>(initialTools);
  const [isStreaming, setIsStreaming] = useState(true);

  // Helper to update a specific step
  const updateStep = (index: number, updates: Partial<PipelineStep>) => {
    setSteps((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], ...updates };
      onStepsChange?.(next);
      return next;
    });
  };

  // Helper to add/update a tool invocation
  const updateTool = (id: string, updates: Partial<ToolInvocation>) => {
    setTools((prev) => {
      const index = prev.findIndex((t) => t.id === id);
      if (index === -1) {
        const newTools = [...prev, { id, ...updates } as ToolInvocation];
        onToolsChange?.(newTools);
        return newTools;
      }
      const next = [...prev];
      next[index] = { ...next[index], ...updates };
      onToolsChange?.(next);
      return next;
    });
  };

  // Append streaming text to a tool
  const appendToolStream = (id: string, text: string) => {
    setTools((prev) => {
      const index = prev.findIndex((t) => t.id === id);
      if (index === -1) return prev;
      const next = [...prev];
      const current = next[index].streamingText || "";
      next[index] = { ...next[index], streamingText: current + text, status: "streaming" };
      return next;
    });
  };

  const completeStreaming = () => {
    setIsStreaming(false);
    // Mark all running tools as completed
    setTools((prev) =>
      prev.map((t) =>
        t.status === "running" || t.status === "streaming"
          ? { ...t, status: "completed", output: t.streamingText }
          : t
      )
    );
  };

  // Compute total time
  const totalTime = useMemo(() => {
    let total = 0;
    steps.forEach((s) => {
      if (s.durationMs) total += s.durationMs;
    });
    tools.forEach((t) => {
      if (t.durationMs) total += t.durationMs;
    });
    return total > 0 ? total : undefined;
  }, [steps, tools]);

  return (
    <ThinkingBlock
      steps={steps}
      toolInvocations={tools}
      totalTime={isStreaming ? undefined : totalTime}
      isStreaming={isStreaming}
      showToolDetails={true}
    />
  );
}
