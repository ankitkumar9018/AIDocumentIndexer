"use client";

import { useState } from "react";
import { formatDistanceToNow, format } from "date-fns";
import {
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  ChevronDown,
  ChevronRight,
  StopCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  useWorkflowExecution,
  useCancelWorkflowExecution,
  type WorkflowExecution,
  type WorkflowStatus,
} from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const statusConfig: Record<
  WorkflowStatus,
  { icon: React.ElementType; color: string; label: string }
> = {
  pending: { icon: Clock, color: "text-gray-500", label: "Pending" },
  running: { icon: Loader2, color: "text-blue-500", label: "Running" },
  completed: { icon: CheckCircle2, color: "text-green-500", label: "Completed" },
  failed: { icon: XCircle, color: "text-red-500", label: "Failed" },
  cancelled: { icon: StopCircle, color: "text-orange-500", label: "Cancelled" },
  paused: { icon: Clock, color: "text-yellow-500", label: "Paused" },
};

interface ExecutionHistoryPanelProps {
  workflowId: string;
  executions: WorkflowExecution[];
}

function ExecutionItem({
  execution,
  expanded,
  onToggle,
}: {
  execution: WorkflowExecution;
  expanded: boolean;
  onToggle: () => void;
}) {
  const { data: detailedExecution, isLoading } = useWorkflowExecution(
    execution.id,
    {
      enabled: expanded,
      refetchInterval: execution.status === "running" ? 2000 : undefined,
    }
  );
  const cancelExecution = useCancelWorkflowExecution();

  const statusInfo = statusConfig[execution.status] || statusConfig.pending;
  const StatusIcon = statusInfo.icon;
  const isRunning = execution.status === "running";

  const handleCancel = async () => {
    try {
      await cancelExecution.mutateAsync(execution.id);
      toast.success("Execution cancelled");
    } catch {
      toast.error("Failed to cancel execution");
    }
  };

  const displayExecution = detailedExecution || execution;

  return (
    <Collapsible open={expanded} onOpenChange={onToggle}>
      <CollapsibleTrigger asChild>
        <div className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 cursor-pointer">
          <div className="flex items-center gap-3">
            <StatusIcon
              className={cn(
                "h-5 w-5",
                statusInfo.color,
                isRunning && "animate-spin"
              )}
            />
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">
                  {execution.started_at
                    ? format(new Date(execution.started_at), "MMM d, HH:mm")
                    : "Pending"}
                </span>
                <Badge
                  variant="outline"
                  className={cn(
                    "text-xs",
                    execution.status === "completed" && "border-green-200 text-green-600",
                    execution.status === "failed" && "border-red-200 text-red-600",
                    execution.status === "running" && "border-blue-200 text-blue-600"
                  )}
                >
                  {statusInfo.label}
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {execution.trigger_type} trigger
                {execution.duration_ms && (
                  <span className="ml-2">
                    {(execution.duration_ms / 1000).toFixed(2)}s
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isRunning && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  handleCancel();
                }}
                disabled={cancelExecution.isPending}
              >
                <StopCircle className="h-4 w-4" />
              </Button>
            )}
            {expanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-2 p-3 bg-muted/30 rounded-lg space-y-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <>
              {/* Error Message */}
              {displayExecution.error_message && (
                <div className="p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                  <strong>Error:</strong> {displayExecution.error_message}
                </div>
              )}

              {/* Node Executions */}
              {displayExecution.node_executions &&
                displayExecution.node_executions.length > 0 && (
                  <div className="space-y-2">
                    <h5 className="text-xs font-medium text-muted-foreground">
                      Node Executions
                    </h5>
                    {displayExecution.node_executions.map((nodeExec) => {
                      const nodeStatus = statusConfig[nodeExec.status] || statusConfig.pending;
                      const NodeStatusIcon = nodeStatus.icon;
                      return (
                        <div
                          key={nodeExec.id}
                          className="flex items-center justify-between text-xs p-2 bg-background rounded border"
                        >
                          <div className="flex items-center gap-2">
                            <NodeStatusIcon
                              className={cn(
                                "h-3.5 w-3.5",
                                nodeStatus.color,
                                nodeExec.status === "running" && "animate-spin"
                              )}
                            />
                            <span>Node {nodeExec.node_id.slice(0, 8)}...</span>
                          </div>
                          <div className="flex items-center gap-2 text-muted-foreground">
                            {nodeExec.duration_ms && (
                              <span>{(nodeExec.duration_ms / 1000).toFixed(2)}s</span>
                            )}
                            <Badge variant="outline" className="text-xs">
                              {nodeStatus.label}
                            </Badge>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

              {/* Input/Output Data */}
              <div className="grid grid-cols-2 gap-2">
                {displayExecution.input_data && (
                  <div className="space-y-1">
                    <h5 className="text-xs font-medium text-muted-foreground">Input</h5>
                    <pre className="text-xs bg-background p-2 rounded border overflow-auto max-h-24">
                      {JSON.stringify(displayExecution.input_data, null, 2)}
                    </pre>
                  </div>
                )}
                {displayExecution.output_data && (
                  <div className="space-y-1">
                    <h5 className="text-xs font-medium text-muted-foreground">Output</h5>
                    <pre className="text-xs bg-background p-2 rounded border overflow-auto max-h-24">
                      {JSON.stringify(displayExecution.output_data, null, 2)}
                    </pre>
                  </div>
                )}
              </div>

              {/* Timing */}
              <div className="text-xs text-muted-foreground">
                <div>
                  Started:{" "}
                  {displayExecution.started_at
                    ? format(new Date(displayExecution.started_at), "PPpp")
                    : "Not started"}
                </div>
                {displayExecution.completed_at && (
                  <div>
                    Completed:{" "}
                    {format(new Date(displayExecution.completed_at), "PPpp")}
                  </div>
                )}
                {displayExecution.retry_count > 0 && (
                  <div>Retries: {displayExecution.retry_count}</div>
                )}
              </div>
            </>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function ExecutionHistoryPanel({
  workflowId,
  executions,
}: ExecutionHistoryPanelProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (executions.length === 0) {
    return (
      <div className="py-8 text-center text-muted-foreground">
        <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No executions yet</p>
        <p className="text-sm">Run the workflow to see execution history</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-[calc(100vh-12rem)]">
      <div className="space-y-2 pr-4">
        {executions.map((execution) => (
          <ExecutionItem
            key={execution.id}
            execution={execution}
            expanded={expandedId === execution.id}
            onToggle={() =>
              setExpandedId(expandedId === execution.id ? null : execution.id)
            }
          />
        ))}
      </div>
    </ScrollArea>
  );
}
