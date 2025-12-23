"use client";

import {
  CheckCircle2,
  Circle,
  Loader2,
  XCircle,
  Bot,
  Search,
  FileEdit,
  MessageSquare,
  Wrench,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { PlanStep } from "@/lib/api/client";

interface AgentExecutionProgressProps {
  steps: PlanStep[];
  currentStep: number;
  isExecuting: boolean;
  className?: string;
}

const agentIcons: Record<string, React.ReactNode> = {
  manager: <Bot className="h-4 w-4" />,
  generator: <FileEdit className="h-4 w-4" />,
  critic: <MessageSquare className="h-4 w-4" />,
  research: <Search className="h-4 w-4" />,
  tool_executor: <Wrench className="h-4 w-4" />,
};

function getStepIcon(status: string, isCurrentStep: boolean) {
  if (status === "completed") {
    return <CheckCircle2 className="h-4 w-4 text-green-500" />;
  }
  if (status === "failed") {
    return <XCircle className="h-4 w-4 text-destructive" />;
  }
  if (isCurrentStep || status === "in_progress") {
    return <Loader2 className="h-4 w-4 text-primary animate-spin" />;
  }
  return <Circle className="h-4 w-4 text-muted-foreground/40" />;
}

function getAgentColor(agent: string): string {
  const colors: Record<string, string> = {
    manager: "bg-violet-500",
    generator: "bg-blue-500",
    critic: "bg-amber-500",
    research: "bg-emerald-500",
    tool_executor: "bg-pink-500",
  };
  return colors[agent.toLowerCase()] || "bg-gray-500";
}

export function AgentExecutionProgress({
  steps,
  currentStep,
  isExecuting,
  className,
}: AgentExecutionProgressProps) {
  const completedSteps = steps.filter((s) => s.status === "completed").length;
  const progress = steps.length > 0 ? (completedSteps / steps.length) * 100 : 0;

  return (
    <Card className={cn("p-4", className)}>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" />
            <h4 className="font-medium">Agent Execution</h4>
          </div>
          <Badge variant={isExecuting ? "default" : "secondary"}>
            {isExecuting ? "Executing" : "Complete"}
          </Badge>
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-medium">
              {completedSteps} / {steps.length} steps
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Steps List */}
        <div className="space-y-2">
          {steps.map((step, idx) => {
            const isCurrentStepActive = idx === currentStep && isExecuting;
            const isPast = idx < currentStep;
            const isFuture = idx > currentStep && isExecuting;

            return (
              <div
                key={step.step_id || idx}
                className={cn(
                  "flex items-start gap-3 p-2 rounded-md transition-colors",
                  isCurrentStepActive && "bg-primary/5 border border-primary/20",
                  isPast && step.status === "completed" && "bg-green-50 dark:bg-green-950/20",
                  step.status === "failed" && "bg-destructive/5 border border-destructive/20",
                  isFuture && "opacity-50"
                )}
              >
                {/* Step Number & Status */}
                <div className="flex flex-col items-center gap-1">
                  <div
                    className={cn(
                      "w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium text-white",
                      step.status === "completed" && "bg-green-500",
                      step.status === "failed" && "bg-destructive",
                      step.status === "in_progress" && "bg-primary",
                      step.status === "pending" && "bg-muted-foreground/30"
                    )}
                  >
                    {idx + 1}
                  </div>
                  {idx < steps.length - 1 && (
                    <div
                      className={cn(
                        "w-0.5 h-4",
                        isPast ? "bg-green-500" : "bg-muted-foreground/20"
                      )}
                    />
                  )}
                </div>

                {/* Step Content */}
                <div className="flex-1 min-w-0 pt-0.5">
                  <div className="flex items-center gap-2 mb-1">
                    <div
                      className={cn(
                        "p-1 rounded",
                        getAgentColor(step.agent)
                      )}
                    >
                      {agentIcons[step.agent.toLowerCase()] || <Bot className="h-3 w-3 text-white" />}
                    </div>
                    <span className="font-medium text-sm capitalize">
                      {step.agent} Agent
                    </span>
                    {getStepIcon(step.status, isCurrentStepActive)}
                  </div>
                  <p className="text-xs text-muted-foreground line-clamp-2">
                    {step.task}
                  </p>

                  {/* Cost Info */}
                  <div className="flex items-center gap-2 mt-1">
                    {step.actual_cost_usd != null ? (
                      <Badge variant="outline" className="text-xs">
                        ${step.actual_cost_usd.toFixed(4)}
                      </Badge>
                    ) : step.estimated_cost_usd != null ? (
                      <Badge variant="outline" className="text-xs text-muted-foreground">
                        ~${step.estimated_cost_usd.toFixed(4)}
                      </Badge>
                    ) : null}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
