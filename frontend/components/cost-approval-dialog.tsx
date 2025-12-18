"use client";

import { useState } from "react";
import {
  DollarSign,
  AlertCircle,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Bot,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { PlanStep } from "@/lib/api/client";

interface CostApprovalDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  planId: string;
  planSummary: string | null;
  estimatedCost: number;
  steps: PlanStep[] | null;
  remainingBudget?: number;
  onApprove: () => void;
  onCancel: () => void;
  isApproving?: boolean;
  isCancelling?: boolean;
}

export function CostApprovalDialog({
  open,
  onOpenChange,
  planId,
  planSummary,
  estimatedCost,
  steps,
  remainingBudget,
  onApprove,
  onCancel,
  isApproving,
  isCancelling,
}: CostApprovalDialogProps) {
  const [showSteps, setShowSteps] = useState(false);

  const willExceedBudget = remainingBudget !== undefined && estimatedCost > remainingBudget;
  const budgetUsagePercent = remainingBudget
    ? Math.min((estimatedCost / remainingBudget) * 100, 100)
    : 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <DollarSign className="h-5 w-5 text-amber-500" />
            Cost Approval Required
          </DialogTitle>
          <DialogDescription>
            This operation requires approval before proceeding.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Plan Summary */}
          {planSummary && (
            <div className="p-3 bg-muted rounded-lg">
              <p className="text-sm font-medium mb-1">Plan Summary</p>
              <p className="text-sm text-muted-foreground">{planSummary}</p>
            </div>
          )}

          {/* Cost Estimate */}
          <div className="p-4 border rounded-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Estimated Cost</span>
              <span className="text-2xl font-bold">
                ${estimatedCost.toFixed(4)}
              </span>
            </div>

            {remainingBudget !== undefined && (
              <>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Daily Budget Remaining</span>
                    <span className={cn(
                      "font-medium",
                      willExceedBudget && "text-destructive"
                    )}>
                      ${remainingBudget.toFixed(2)}
                    </span>
                  </div>
                  <Progress
                    value={budgetUsagePercent}
                    className={cn(
                      "h-2",
                      willExceedBudget && "[&>div]:bg-destructive"
                    )}
                  />
                </div>

                {willExceedBudget && (
                  <div className="flex items-start gap-2 p-2 bg-destructive/10 rounded-md text-sm text-destructive">
                    <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>
                      This operation would exceed your remaining daily budget.
                    </span>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Execution Steps */}
          {steps && steps.length > 0 && (
            <Collapsible open={showSteps} onOpenChange={setShowSteps}>
              <CollapsibleTrigger asChild>
                <Button
                  variant="ghost"
                  className="w-full justify-between px-3"
                  size="sm"
                >
                  <span className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    {steps.length} Execution Steps
                  </span>
                  {showSteps ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {steps.map((step, idx) => (
                    <div
                      key={step.step_id || idx}
                      className="flex items-center gap-3 p-2 bg-muted/50 rounded-md text-sm"
                    >
                      <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium">
                        {idx + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <Bot className="h-3 w-3 text-muted-foreground" />
                          <span className="font-medium capitalize">
                            {step.agent}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {step.task}
                        </p>
                      </div>
                      {step.estimated_cost_usd !== null && (
                        <Badge variant="outline" className="text-xs">
                          ${step.estimated_cost_usd.toFixed(4)}
                        </Badge>
                      )}
                    </div>
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button
            variant="outline"
            onClick={onCancel}
            disabled={isApproving || isCancelling}
          >
            {isCancelling ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Cancelling...
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4 mr-2" />
                Cancel
              </>
            )}
          </Button>
          <Button
            onClick={onApprove}
            disabled={isApproving || isCancelling || willExceedBudget}
          >
            {isApproving ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Approving...
              </>
            ) : (
              <>
                <CheckCircle2 className="h-4 w-4 mr-2" />
                Approve & Execute
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
