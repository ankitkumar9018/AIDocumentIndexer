"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Clock,
  Loader2,
  Brain,
  Cog,
  MessageSquare,
  AlertCircle,
  Lightbulb,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";

interface PlanStep {
  step_id: string;
  step_number: number;
  agent: string;
  task: string;
  name?: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  estimated_cost_usd?: number | null;
  actual_cost_usd?: number | null;
  dependencies?: string[];
  output?: string;  // Output preview from completed step
}

interface AgentResponseSectionsProps {
  planningDetails?: string;
  executionSteps?: PlanStep[];
  finalAnswer: string;
  isExecuting?: boolean;
  currentStep?: number;
  totalSteps?: number;
  thinkingContent?: string;  // Agent's reasoning/thinking process
}

const getAgentIcon = (agent: string) => {
  switch (agent?.toLowerCase()) {
    case "manager":
      return Brain;
    case "research":
      return MessageSquare;
    case "generator":
    case "critic":
    case "tool_executor":
    default:
      return Cog;
  }
};

const getAgentColor = (agent: string) => {
  switch (agent?.toLowerCase()) {
    case "manager":
      return "bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300";
    case "generator":
      return "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300";
    case "critic":
      return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300";
    case "research":
      return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300";
    case "tool_executor":
      return "bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-300";
    default:
      return "bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-300";
  }
};

const getStatusIcon = (status: string, isCurrentStep?: boolean) => {
  if (isCurrentStep && status === "in_progress") {
    return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
  }
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-4 w-4 text-green-500" />;
    case "failed":
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    case "in_progress":
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    default:
      return <Clock className="h-4 w-4 text-muted-foreground" />;
  }
};

export function AgentResponseSections({
  planningDetails,
  executionSteps = [],
  finalAnswer,
  isExecuting = false,
  currentStep,
  totalSteps,
  thinkingContent,
}: AgentResponseSectionsProps) {
  const [isPlanningOpen, setIsPlanningOpen] = useState(false);
  const [isThinkingOpen, setIsThinkingOpen] = useState(false);  // Collapsed by default per user preference
  const [isExecutionOpen, setIsExecutionOpen] = useState(false);
  const [isFinalOpen, setIsFinalOpen] = useState(true);
  // Track which step outputs are expanded (all collapsed by default)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  const toggleStepOutput = (stepId: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  };

  const completedSteps = executionSteps.filter(
    (s) => s.status === "completed"
  ).length;
  const hasSteps = executionSteps.length > 0;
  const hasPlanningDetails = planningDetails && planningDetails.trim().length > 0;
  const hasThinking = thinkingContent && thinkingContent.trim().length > 0;

  return (
    <div className="space-y-2">
      {/* Planning Phase Section */}
      {hasPlanningDetails && (
        <Collapsible open={isPlanningOpen} onOpenChange={setIsPlanningOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-between px-3 py-2 h-auto",
                "bg-muted/50 hover:bg-muted/70 rounded-lg",
                "border border-border/50"
              )}
            >
              <div className="flex items-center gap-2">
                {isPlanningOpen ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                )}
                <Brain className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium text-muted-foreground">
                  Planning Phase
                </span>
              </div>
              <Badge variant="secondary" className="text-xs">
                Analyzed
              </Badge>
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 p-3 bg-muted/30 rounded-lg border border-border/30">
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {planningDetails}
              </p>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Agent Thinking/Reasoning Section */}
      {hasThinking && (
        <Collapsible open={isThinkingOpen} onOpenChange={setIsThinkingOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-between px-3 py-2 h-auto",
                "bg-amber-50/50 hover:bg-amber-50/70 dark:bg-amber-950/20 dark:hover:bg-amber-950/30",
                "rounded-lg border border-amber-200/50 dark:border-amber-800/30"
              )}
            >
              <div className="flex items-center gap-2">
                {isThinkingOpen ? (
                  <ChevronDown className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                )}
                <Lightbulb className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                <span className="text-sm font-medium text-amber-700 dark:text-amber-300">
                  Agent Reasoning
                </span>
              </div>
              <Badge
                variant="secondary"
                className="text-xs bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
              >
                Thinking
              </Badge>
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 p-3 bg-amber-50/30 dark:bg-amber-950/10 rounded-lg border border-amber-200/30 dark:border-amber-800/20">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {thinkingContent}
                </ReactMarkdown>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Execution Steps Section */}
      {hasSteps && (
        <Collapsible open={isExecutionOpen} onOpenChange={setIsExecutionOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-between px-3 py-2 h-auto",
                "bg-blue-50/50 hover:bg-blue-50/70 dark:bg-blue-950/20 dark:hover:bg-blue-950/30",
                "rounded-lg border border-blue-200/50 dark:border-blue-800/30"
              )}
            >
              <div className="flex items-center gap-2">
                {isExecutionOpen ? (
                  <ChevronDown className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                )}
                <Cog className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                  Execution Steps
                </span>
              </div>
              <div className="flex items-center gap-2">
                {isExecuting && (
                  <Loader2 className="h-3 w-3 animate-spin text-blue-600" />
                )}
                <Badge
                  variant="secondary"
                  className={cn(
                    "text-xs",
                    completedSteps === executionSteps.length
                      ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                      : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                  )}
                >
                  {completedSteps}/{executionSteps.length} completed
                </Badge>
              </div>
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 space-y-2 p-3 bg-blue-50/30 dark:bg-blue-950/10 rounded-lg border border-blue-200/30 dark:border-blue-800/20">
              {executionSteps.map((step, index) => {
                const AgentIcon = getAgentIcon(step.agent);
                const isCurrentStepActive =
                  currentStep !== undefined && index + 1 === currentStep;
                const isStepExpanded = expandedSteps.has(step.step_id);
                const hasOutput = step.output && step.output.trim().length > 0;

                return (
                  <div
                    key={step.step_id}
                    className={cn(
                      "rounded-md transition-colors",
                      isCurrentStepActive && "bg-primary/5 border border-primary/20",
                      step.status === "completed" && !isStepExpanded && "opacity-80"
                    )}
                  >
                    <div
                      className={cn(
                        "flex items-start gap-3 p-2 cursor-pointer hover:bg-muted/50 rounded-md",
                        hasOutput && "cursor-pointer"
                      )}
                      onClick={() => hasOutput && toggleStepOutput(step.step_id)}
                    >
                      <div className="flex-shrink-0 mt-0.5">
                        {getStatusIcon(step.status, isCurrentStepActive)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-xs font-medium text-muted-foreground">
                            Step {step.step_number}
                          </span>
                          <Badge
                            variant="secondary"
                            className={cn("text-xs px-1.5 py-0", getAgentColor(step.agent))}
                          >
                            <AgentIcon className="h-3 w-3 mr-1" />
                            {step.agent}
                          </Badge>
                          {hasOutput && (
                            <span className="text-xs text-muted-foreground flex items-center gap-1">
                              {isStepExpanded ? (
                                <ChevronDown className="h-3 w-3" />
                              ) : (
                                <ChevronRight className="h-3 w-3" />
                              )}
                              Output
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-foreground mt-0.5 line-clamp-2">
                          {step.name || step.task}
                        </p>
                        {step.actual_cost_usd != null && (
                          <span className="text-xs text-muted-foreground">
                            Cost: ${step.actual_cost_usd.toFixed(4)}
                          </span>
                        )}
                      </div>
                    </div>
                    {/* Collapsible output section with markdown rendering */}
                    {hasOutput && isStepExpanded && (
                      <div className="mx-2 mb-2 p-3 bg-muted/30 rounded border border-border/50 max-h-[400px] overflow-y-auto">
                        <div className="prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {step.output}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Final Answer Section */}
      {finalAnswer && (
        <Collapsible open={isFinalOpen} onOpenChange={setIsFinalOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-between px-3 py-2 h-auto",
                "bg-background hover:bg-accent/50",
                "rounded-lg border border-primary/20",
                isFinalOpen && "border-primary/40"
              )}
            >
              <div className="flex items-center gap-2">
                {isFinalOpen ? (
                  <ChevronDown className="h-4 w-4 text-primary" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-primary" />
                )}
                <MessageSquare className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium text-foreground">
                  Final Answer
                </span>
              </div>
              {!isFinalOpen && (
                <Badge variant="outline" className="text-xs">
                  Click to expand
                </Badge>
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 p-4 bg-background rounded-lg border border-primary/20 shadow-sm">
              <div className="prose prose-sm dark:prose-invert max-w-none prose-chat">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {finalAnswer}
                </ReactMarkdown>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
}

export default AgentResponseSections;
