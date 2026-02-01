"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Search,
  FileText,
  Database,
  Globe,
  Code,
  Terminal,
  Sparkles,
  CheckCircle,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Copy,
  Clock,
  Zap,
  Brain,
  Image,
  FileSearch,
  Network,
  Calculator,
  BookOpen,
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

// Tool types and their metadata
export type ToolType =
  | "search"
  | "retrieval"
  | "database"
  | "web_fetch"
  | "code_execution"
  | "file_read"
  | "file_write"
  | "image_analysis"
  | "calculation"
  | "knowledge_graph"
  | "summarization"
  | "translation"
  | "fact_check"
  | "entity_extraction";

export type ToolStatus = "pending" | "running" | "streaming" | "completed" | "error" | "cancelled";

export interface ToolExecution {
  id: string;
  type: ToolType;
  name: string;
  description?: string;
  status: ToolStatus;
  input?: Record<string, unknown>;
  output?: string;
  streamingOutput?: string;
  error?: string;
  startTime?: Date;
  endTime?: Date;
  progress?: number;
  metadata?: {
    documentsFound?: number;
    tokensUsed?: number;
    model?: string;
    confidence?: number;
  };
}

const TOOL_ICONS: Record<ToolType, React.ReactNode> = {
  search: <Search className="h-4 w-4" />,
  retrieval: <FileSearch className="h-4 w-4" />,
  database: <Database className="h-4 w-4" />,
  web_fetch: <Globe className="h-4 w-4" />,
  code_execution: <Terminal className="h-4 w-4" />,
  file_read: <FileText className="h-4 w-4" />,
  file_write: <FileText className="h-4 w-4" />,
  image_analysis: <Image className="h-4 w-4" />,
  calculation: <Calculator className="h-4 w-4" />,
  knowledge_graph: <Network className="h-4 w-4" />,
  summarization: <BookOpen className="h-4 w-4" />,
  translation: <Globe className="h-4 w-4" />,
  fact_check: <CheckCircle className="h-4 w-4" />,
  entity_extraction: <Brain className="h-4 w-4" />,
};

const TOOL_COLORS: Record<ToolType, string> = {
  search: "text-blue-500 bg-blue-500/10",
  retrieval: "text-purple-500 bg-purple-500/10",
  database: "text-orange-500 bg-orange-500/10",
  web_fetch: "text-green-500 bg-green-500/10",
  code_execution: "text-yellow-500 bg-yellow-500/10",
  file_read: "text-cyan-500 bg-cyan-500/10",
  file_write: "text-cyan-500 bg-cyan-500/10",
  image_analysis: "text-pink-500 bg-pink-500/10",
  calculation: "text-indigo-500 bg-indigo-500/10",
  knowledge_graph: "text-teal-500 bg-teal-500/10",
  summarization: "text-emerald-500 bg-emerald-500/10",
  translation: "text-rose-500 bg-rose-500/10",
  fact_check: "text-lime-500 bg-lime-500/10",
  entity_extraction: "text-violet-500 bg-violet-500/10",
};

const STATUS_STYLES: Record<ToolStatus, { icon: React.ReactNode; className: string; label: string }> = {
  pending: {
    icon: <Clock className="h-3 w-3" />,
    className: "text-muted-foreground",
    label: "Pending",
  },
  running: {
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
    className: "text-blue-500",
    label: "Running",
  },
  streaming: {
    icon: <Zap className="h-3 w-3 animate-pulse" />,
    className: "text-yellow-500",
    label: "Streaming",
  },
  completed: {
    icon: <CheckCircle className="h-3 w-3" />,
    className: "text-green-500",
    label: "Completed",
  },
  error: {
    icon: <XCircle className="h-3 w-3" />,
    className: "text-red-500",
    label: "Error",
  },
  cancelled: {
    icon: <AlertCircle className="h-3 w-3" />,
    className: "text-orange-500",
    label: "Cancelled",
  },
};

interface ToolStreamingBlockProps {
  execution: ToolExecution;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
  onCancel?: () => void;
  onRetry?: () => void;
}

export function ToolStreamingBlock({
  execution,
  isExpanded = false,
  onToggleExpand,
  onCancel,
  onRetry,
}: ToolStreamingBlockProps) {
  const [copied, setCopied] = useState(false);
  const outputRef = useRef<HTMLDivElement>(null);

  // Auto-scroll streaming output
  useEffect(() => {
    if (execution.status === "streaming" && outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [execution.streamingOutput, execution.status]);

  const statusStyle = STATUS_STYLES[execution.status];
  const toolColor = TOOL_COLORS[execution.type];
  const isActive = execution.status === "running" || execution.status === "streaming";

  const duration = execution.startTime && execution.endTime
    ? Math.round((execution.endTime.getTime() - execution.startTime.getTime()) / 1000 * 10) / 10
    : null;

  const handleCopy = async () => {
    const textToCopy = execution.output || execution.streamingOutput || "";
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const textArea = document.createElement("textarea");
      textArea.value = textToCopy;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div
      className={cn(
        "border rounded-lg overflow-hidden transition-all",
        isActive && "border-primary/50 shadow-sm",
        execution.status === "error" && "border-red-500/50"
      )}
    >
      {/* Header */}
      <Collapsible open={isExpanded} onOpenChange={onToggleExpand}>
        <CollapsibleTrigger asChild>
          <div
            className={cn(
              "flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-muted/50 transition-colors",
              isActive && "bg-muted/30"
            )}
          >
            {/* Tool Icon */}
            <div className={cn("p-1.5 rounded", toolColor)}>
              {TOOL_ICONS[execution.type]}
            </div>

            {/* Tool Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm truncate">{execution.name}</span>
                <Badge variant="outline" className={cn("text-xs", statusStyle.className)}>
                  {statusStyle.icon}
                  <span className="ml-1">{statusStyle.label}</span>
                </Badge>
              </div>
              {execution.description && (
                <p className="text-xs text-muted-foreground truncate mt-0.5">
                  {execution.description}
                </p>
              )}
            </div>

            {/* Progress / Duration */}
            <div className="flex items-center gap-2">
              {isActive && execution.progress !== undefined && (
                <div className="w-20">
                  <Progress value={execution.progress} className="h-1.5" />
                </div>
              )}
              {duration !== null && (
                <span className="text-xs text-muted-foreground">{duration}s</span>
              )}
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
            </div>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="border-t">
            {/* Input Parameters */}
            {execution.input && Object.keys(execution.input).length > 0 && (
              <div className="px-3 py-2 border-b bg-muted/20">
                <div className="text-xs font-medium text-muted-foreground mb-1">Input</div>
                <div className="font-mono text-xs space-y-0.5">
                  {Object.entries(execution.input).map(([key, value]) => (
                    <div key={key} className="flex">
                      <span className="text-blue-500 min-w-[80px]">{key}:</span>
                      <span className="text-foreground truncate">
                        {typeof value === "string" ? value : JSON.stringify(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Streaming / Output */}
            {(execution.output || execution.streamingOutput || execution.error) && (
              <div className="px-3 py-2">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-xs font-medium text-muted-foreground">
                    {execution.error ? "Error" : "Output"}
                  </div>
                  <div className="flex items-center gap-1">
                    {(execution.output || execution.streamingOutput) && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-xs"
                        onClick={handleCopy}
                      >
                        {copied ? (
                          <>
                            <CheckCircle className="h-3 w-3 mr-1 text-green-500" />
                            Copied
                          </>
                        ) : (
                          <>
                            <Copy className="h-3 w-3 mr-1" />
                            Copy
                          </>
                        )}
                      </Button>
                    )}
                  </div>
                </div>

                <div
                  ref={outputRef}
                  className={cn(
                    "font-mono text-xs rounded p-2 max-h-[200px] overflow-auto",
                    execution.error
                      ? "bg-red-500/10 text-red-600"
                      : "bg-muted/50"
                  )}
                >
                  {execution.error ? (
                    <span>{execution.error}</span>
                  ) : execution.status === "streaming" ? (
                    <span className="whitespace-pre-wrap">
                      {execution.streamingOutput}
                      <span className="inline-block w-1.5 h-4 bg-primary animate-pulse ml-0.5" />
                    </span>
                  ) : (
                    <span className="whitespace-pre-wrap">{execution.output}</span>
                  )}
                </div>
              </div>
            )}

            {/* Metadata */}
            {execution.metadata && Object.keys(execution.metadata).length > 0 && (
              <div className="px-3 py-2 border-t bg-muted/20">
                <div className="flex flex-wrap gap-3">
                  {execution.metadata.documentsFound !== undefined && (
                    <div className="flex items-center gap-1 text-xs">
                      <FileText className="h-3 w-3 text-muted-foreground" />
                      <span>{execution.metadata.documentsFound} docs</span>
                    </div>
                  )}
                  {execution.metadata.tokensUsed !== undefined && (
                    <div className="flex items-center gap-1 text-xs">
                      <Sparkles className="h-3 w-3 text-muted-foreground" />
                      <span>{execution.metadata.tokensUsed.toLocaleString()} tokens</span>
                    </div>
                  )}
                  {execution.metadata.model && (
                    <div className="flex items-center gap-1 text-xs">
                      <Brain className="h-3 w-3 text-muted-foreground" />
                      <span>{execution.metadata.model}</span>
                    </div>
                  )}
                  {execution.metadata.confidence !== undefined && (
                    <div className="flex items-center gap-1 text-xs">
                      <CheckCircle className="h-3 w-3 text-muted-foreground" />
                      <span>{Math.round(execution.metadata.confidence * 100)}%</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Actions */}
            {(onCancel || onRetry) && (
              <div className="px-3 py-2 border-t flex items-center gap-2">
                {isActive && onCancel && (
                  <Button variant="outline" size="sm" className="h-7 text-xs" onClick={onCancel}>
                    <Pause className="h-3 w-3 mr-1" />
                    Cancel
                  </Button>
                )}
                {execution.status === "error" && onRetry && (
                  <Button variant="outline" size="sm" className="h-7 text-xs" onClick={onRetry}>
                    <RotateCcw className="h-3 w-3 mr-1" />
                    Retry
                  </Button>
                )}
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// Container for multiple tool executions
interface ToolStreamingContainerProps {
  executions: ToolExecution[];
  title?: string;
  onCancelTool?: (id: string) => void;
  onRetryTool?: (id: string) => void;
}

export function ToolStreamingContainer({
  executions,
  title = "Tool Executions",
  onCancelTool,
  onRetryTool,
}: ToolStreamingContainerProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  // Auto-expand running tools
  useEffect(() => {
    executions.forEach((exec) => {
      if (exec.status === "running" || exec.status === "streaming") {
        setExpandedIds((prev) => new Set(prev).add(exec.id));
      }
    });
  }, [executions]);

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const activeCount = executions.filter(
    (e) => e.status === "running" || e.status === "streaming"
  ).length;
  const completedCount = executions.filter((e) => e.status === "completed").length;
  const errorCount = executions.filter((e) => e.status === "error").length;

  if (executions.length === 0) return null;

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium">{title}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {activeCount > 0 && (
            <Badge variant="outline" className="text-blue-500 border-blue-500/30 bg-blue-500/10">
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
              {activeCount} running
            </Badge>
          )}
          {completedCount > 0 && (
            <Badge variant="outline" className="text-green-500 border-green-500/30 bg-green-500/10">
              <CheckCircle className="h-3 w-3 mr-1" />
              {completedCount} done
            </Badge>
          )}
          {errorCount > 0 && (
            <Badge variant="outline" className="text-red-500 border-red-500/30 bg-red-500/10">
              <XCircle className="h-3 w-3 mr-1" />
              {errorCount} failed
            </Badge>
          )}
        </div>
      </div>

      {/* Tool List */}
      <div className="space-y-2">
        {executions.map((execution) => (
          <ToolStreamingBlock
            key={execution.id}
            execution={execution}
            isExpanded={expandedIds.has(execution.id)}
            onToggleExpand={() => toggleExpand(execution.id)}
            onCancel={onCancelTool ? () => onCancelTool(execution.id) : undefined}
            onRetry={onRetryTool ? () => onRetryTool(execution.id) : undefined}
          />
        ))}
      </div>
    </div>
  );
}

// Demo component showing usage
export function ToolStreamingDemo() {
  const [executions, setExecutions] = useState<ToolExecution[]>([
    {
      id: "1",
      type: "retrieval",
      name: "Document Search",
      description: "Searching knowledge base for relevant documents",
      status: "completed",
      input: { query: "machine learning optimization" },
      output: "Found 5 relevant documents:\n1. ML Optimization Techniques (95% match)\n2. Gradient Descent Methods (89% match)\n3. Neural Network Training (85% match)",
      startTime: new Date(Date.now() - 2500),
      endTime: new Date(Date.now() - 500),
      metadata: { documentsFound: 5, confidence: 0.92 },
    },
    {
      id: "2",
      type: "code_execution",
      name: "Python Analysis",
      description: "Running code analysis on retrieved content",
      status: "streaming",
      input: { script: "analyze_documents.py" },
      streamingOutput: "Analyzing document 1/5...\nExtracting key concepts...\nIdentified 12 main topics\nProcessing embeddings",
      startTime: new Date(Date.now() - 1000),
      progress: 65,
      metadata: { model: "code-analyzer-v2" },
    },
    {
      id: "3",
      type: "knowledge_graph",
      name: "Graph Update",
      description: "Updating knowledge graph with new relationships",
      status: "pending",
    },
  ]);

  // Simulate streaming
  useEffect(() => {
    const interval = setInterval(() => {
      setExecutions((prev) =>
        prev.map((exec) => {
          if (exec.status === "streaming" && exec.streamingOutput) {
            const newProgress = (exec.progress || 0) + 5;
            if (newProgress >= 100) {
              return {
                ...exec,
                status: "completed" as ToolStatus,
                output: exec.streamingOutput + "\n\nAnalysis complete!",
                streamingOutput: undefined,
                endTime: new Date(),
                progress: 100,
              };
            }
            return {
              ...exec,
              progress: newProgress,
              streamingOutput: exec.streamingOutput + `\nProgress: ${newProgress}%...`,
            };
          }
          return exec;
        })
      );
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 max-w-2xl">
      <ToolStreamingContainer
        executions={executions}
        onCancelTool={(id) => {
          setExecutions((prev) =>
            prev.map((e) => (e.id === id ? { ...e, status: "cancelled" as ToolStatus } : e))
          );
        }}
        onRetryTool={(id) => {
          setExecutions((prev) =>
            prev.map((e) => (e.id === id ? { ...e, status: "running" as ToolStatus, error: undefined } : e))
          );
        }}
      />
    </div>
  );
}
