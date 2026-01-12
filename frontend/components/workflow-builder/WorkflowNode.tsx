"use client";

import { memo } from "react";
import { Handle, Position, type Node } from "@xyflow/react";
import {
  Play,
  Square,
  GitBranch,
  Repeat,
  Code,
  Clock,
  Globe,
  Bell,
  Bot,
  UserCheck,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

const nodeTypeIcons: Record<string, React.ElementType> = {
  START: Play,
  END: Square,
  ACTION: Zap,
  CONDITION: GitBranch,
  LOOP: Repeat,
  CODE: Code,
  DELAY: Clock,
  HTTP: Globe,
  NOTIFICATION: Bell,
  AGENT: Bot,
  HUMAN_APPROVAL: UserCheck,
};

const nodeTypeColors: Record<string, { bg: string; border: string; text: string }> = {
  START: { bg: "bg-green-50", border: "border-green-500", text: "text-green-700" },
  END: { bg: "bg-red-50", border: "border-red-500", text: "text-red-700" },
  ACTION: { bg: "bg-blue-50", border: "border-blue-500", text: "text-blue-700" },
  CONDITION: { bg: "bg-amber-50", border: "border-amber-500", text: "text-amber-700" },
  LOOP: { bg: "bg-purple-50", border: "border-purple-500", text: "text-purple-700" },
  CODE: { bg: "bg-gray-50", border: "border-gray-500", text: "text-gray-700" },
  DELAY: { bg: "bg-cyan-50", border: "border-cyan-500", text: "text-cyan-700" },
  HTTP: { bg: "bg-indigo-50", border: "border-indigo-500", text: "text-indigo-700" },
  NOTIFICATION: { bg: "bg-pink-50", border: "border-pink-500", text: "text-pink-700" },
  AGENT: { bg: "bg-violet-50", border: "border-violet-500", text: "text-violet-700" },
  HUMAN_APPROVAL: { bg: "bg-orange-50", border: "border-orange-500", text: "text-orange-700" },
};

export interface WorkflowNodeData extends Record<string, unknown> {
  label: string;
  nodeType: string;
  description?: string | null;
  config?: Record<string, unknown> | null;
}

export type WorkflowNodeType = Node<WorkflowNodeData>;

interface WorkflowNodeProps {
  data: WorkflowNodeData;
  selected?: boolean;
}

function WorkflowNodeBase({ data, selected }: WorkflowNodeProps) {
  // Normalize nodeType to uppercase for display consistency
  const normalizedType = data.nodeType?.toUpperCase() || "ACTION";
  const Icon = nodeTypeIcons[normalizedType] || Zap;
  const colors = nodeTypeColors[normalizedType] || nodeTypeColors.ACTION;

  const isStartNode = normalizedType === "START";
  const isEndNode = normalizedType === "END";
  const isConditionNode = normalizedType === "CONDITION";
  const isLoopNode = normalizedType === "LOOP";

  return (
    <div
      className={cn(
        "px-4 py-3 rounded-lg border-2 shadow-sm min-w-[150px] max-w-[200px]",
        colors.bg,
        colors.border,
        selected && "ring-2 ring-primary ring-offset-2"
      )}
    >
      {/* Input Handle */}
      {!isStartNode && (
        <Handle
          type="target"
          position={Position.Top}
          className="!bg-gray-400 !w-3 !h-3"
        />
      )}

      <div className="flex items-center gap-2">
        <div className={cn("p-1.5 rounded-md", colors.bg, colors.text)}>
          <Icon className="h-4 w-4" />
        </div>
        <div className="flex-1 min-w-0">
          <div className={cn("text-sm font-medium truncate", colors.text)}>
            {data.label}
          </div>
          {data.description && (
            <div className="text-xs text-muted-foreground truncate">
              {data.description}
            </div>
          )}
        </div>
      </div>

      {/* Config indicator */}
      {data.config && Object.keys(data.config).length > 0 && (
        <div className="mt-2 text-xs text-muted-foreground">
          {Object.keys(data.config).length} configured
        </div>
      )}

      {/* Output Handles */}
      {!isEndNode && (
        <>
          {isConditionNode || isLoopNode ? (
            <>
              {/* True/Yes output */}
              <Handle
                type="source"
                position={Position.Bottom}
                id="true"
                className="!bg-green-500 !w-3 !h-3"
                style={{ left: "30%" }}
              />
              {/* False/No output */}
              <Handle
                type="source"
                position={Position.Bottom}
                id="false"
                className="!bg-red-500 !w-3 !h-3"
                style={{ left: "70%" }}
              />
            </>
          ) : (
            <Handle
              type="source"
              position={Position.Bottom}
              className="!bg-gray-400 !w-3 !h-3"
            />
          )}
        </>
      )}
    </div>
  );
}

export const WorkflowNodeComponent = memo(WorkflowNodeBase);
