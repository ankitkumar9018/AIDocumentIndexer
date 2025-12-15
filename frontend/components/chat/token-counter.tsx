"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import {
  countTokens,
  getContextLimit,
  getContextStatus,
  formatTokenCount,
  estimateCost,
} from "@/lib/utils/tokenizer";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";
import { Coins, Zap } from "lucide-react";

interface TokenCounterProps {
  text: string;
  modelId?: string;
  previousTokens?: number;
  className?: string;
  showCost?: boolean;
}

export function TokenCounter({
  text,
  modelId = "gpt-4o",
  previousTokens = 0,
  className,
  showCost = true,
}: TokenCounterProps) {
  const [tokenCount, setTokenCount] = React.useState(0);

  // Debounced token counting for performance
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setTokenCount(countTokens(text));
    }, 100);

    return () => clearTimeout(timer);
  }, [text]);

  const totalTokens = previousTokens + tokenCount;
  const contextStatus = getContextStatus(totalTokens, modelId);
  const estimatedResponseTokens = Math.min(tokenCount * 2, 4096);
  const cost = estimateCost(totalTokens, estimatedResponseTokens, modelId);

  const statusColors = {
    ok: "text-green-600 dark:text-green-400",
    warning: "text-yellow-600 dark:text-yellow-400",
    critical: "text-red-600 dark:text-red-400",
  };

  const progressColors = {
    ok: "bg-green-500",
    warning: "bg-yellow-500",
    critical: "bg-red-500",
  };

  return (
    <TooltipProvider>
      <div className={cn("flex items-center gap-3 text-xs", className)}>
        {/* Token count */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className={cn("flex items-center gap-1", statusColors[contextStatus.status])}>
              <Zap className="h-3 w-3" />
              <span className="font-medium">
                {formatTokenCount(tokenCount)} tokens
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-xs">
            <div className="space-y-2">
              <p className="font-medium">Token Usage</p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Current input:</span>
                  <span>{tokenCount.toLocaleString()}</span>
                </div>
                {previousTokens > 0 && (
                  <div className="flex justify-between gap-4">
                    <span className="text-muted-foreground">Conversation history:</span>
                    <span>{previousTokens.toLocaleString()}</span>
                  </div>
                )}
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Total:</span>
                  <span className="font-medium">{totalTokens.toLocaleString()}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Context limit:</span>
                  <span>{formatTokenCount(contextStatus.limit)}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">Remaining:</span>
                  <span>{formatTokenCount(contextStatus.remaining)}</span>
                </div>
              </div>
            </div>
          </TooltipContent>
        </Tooltip>

        {/* Context usage bar */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-2 min-w-[100px]">
              <Progress
                value={contextStatus.percent}
                className="h-1.5 w-full"
                indicatorClassName={progressColors[contextStatus.status]}
              />
              <span className={cn("text-xs tabular-nums", statusColors[contextStatus.status])}>
                {contextStatus.percent.toFixed(0)}%
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="top">
            <p>Context window usage: {contextStatus.percent.toFixed(1)}%</p>
            <p className="text-xs text-muted-foreground">
              {formatTokenCount(totalTokens)} / {formatTokenCount(contextStatus.limit)}
            </p>
          </TooltipContent>
        </Tooltip>

        {/* Cost estimate */}
        {showCost && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1 text-muted-foreground">
                <Coins className="h-3 w-3" />
                <span>~${cost < 0.01 ? "<0.01" : cost.toFixed(3)}</span>
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p>Estimated cost for this request</p>
              <p className="text-xs text-muted-foreground">
                Based on {modelId} pricing
              </p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
    </TooltipProvider>
  );
}
