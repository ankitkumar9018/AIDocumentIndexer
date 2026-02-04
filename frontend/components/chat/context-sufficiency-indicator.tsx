"use client";

import * as React from "react";
import { Info, AlertTriangle, CheckCircle, AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/**
 * Context sufficiency information from the API.
 * Shows how well the retrieved documents cover the query.
 */
export interface ContextSufficiencyInfo {
  is_sufficient: boolean;
  coverage_score: number; // 0-1
  has_conflicts: boolean;
  missing_aspects: string[];
  confidence_level: "high" | "medium" | "low";
  explanation: string;
}

interface ContextSufficiencyIndicatorProps {
  sufficiency: ContextSufficiencyInfo | null | undefined;
  className?: string;
  compact?: boolean; // Show only the indicator dot in compact mode
}

/**
 * Context Sufficiency Indicator
 *
 * Displays the quality of retrieved context for a RAG response.
 * Research shows this helps users understand when the AI might be making
 * educated guesses vs. when it has strong source material.
 *
 * Color coding:
 * - Green (high): 70%+ coverage, no conflicts
 * - Yellow (medium): 40-70% coverage or some gaps
 * - Red (low): <40% coverage or has conflicts
 */
export function ContextSufficiencyIndicator({
  sufficiency,
  className,
  compact = false,
}: ContextSufficiencyIndicatorProps) {
  if (!sufficiency) {
    return null;
  }

  const { coverage_score, has_conflicts, missing_aspects, confidence_level, explanation } =
    sufficiency;

  // Determine status color and icon
  const getStatusConfig = () => {
    if (has_conflicts) {
      return {
        color: "bg-orange-500",
        textColor: "text-orange-600 dark:text-orange-400",
        icon: AlertTriangle,
        label: "Conflicting sources",
        variant: "destructive" as const,
      };
    }

    if (confidence_level === "high" || coverage_score >= 0.65) {
      return {
        color: "bg-green-500",
        textColor: "text-green-600 dark:text-green-400",
        icon: CheckCircle,
        label: "High confidence",
        variant: "secondary" as const,
      };
    }

    if (confidence_level === "medium" || coverage_score >= 0.35) {
      return {
        color: "bg-yellow-500",
        textColor: "text-yellow-600 dark:text-yellow-400",
        icon: AlertCircle,
        label: "Partial context",
        variant: "outline" as const,
      };
    }

    return {
      color: "bg-red-500",
      textColor: "text-red-600 dark:text-red-400",
      icon: AlertCircle,
      label: "Limited context",
      variant: "destructive" as const,
    };
  };

  const status = getStatusConfig();
  const Icon = status.icon;
  const coveragePercent = Math.round(coverage_score * 100);

  // Build tooltip content
  const tooltipContent = (
    <div className="max-w-xs space-y-2 text-sm">
      <div className="font-medium">{status.label}</div>

      <div className="flex items-center gap-2">
        <span className="text-muted-foreground">Coverage:</span>
        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={cn("h-full transition-all", status.color)}
            style={{ width: `${coveragePercent}%` }}
          />
        </div>
        <span className="text-xs font-medium">{coveragePercent}%</span>
      </div>

      {explanation && (
        <p className="text-muted-foreground text-xs">{explanation}</p>
      )}

      {has_conflicts && (
        <div className="flex items-center gap-1 text-orange-600 dark:text-orange-400">
          <AlertTriangle className="h-3 w-3" />
          <span className="text-xs">Sources contain conflicting information</span>
        </div>
      )}

      {missing_aspects && missing_aspects.length > 0 && (
        <div className="text-xs">
          <span className="text-muted-foreground">May not fully cover: </span>
          <span>{missing_aspects.slice(0, 3).join(", ")}</span>
          {missing_aspects.length > 3 && (
            <span className="text-muted-foreground">
              {" "}
              +{missing_aspects.length - 3} more
            </span>
          )}
        </div>
      )}
    </div>
  );

  // Compact mode: just show the indicator dot
  if (compact) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div
              className={cn(
                "w-2 h-2 rounded-full cursor-help transition-all hover:scale-125",
                status.color,
                className
              )}
              aria-label={`Context sufficiency: ${status.label}`}
            />
          </TooltipTrigger>
          <TooltipContent side="top" className="p-3">
            {tooltipContent}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  // Full mode: show badge with label
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              "inline-flex items-center gap-1.5 text-xs cursor-help",
              status.textColor,
              className
            )}
          >
            <div className={cn("w-2 h-2 rounded-full", status.color)} />
            <span className="hidden sm:inline">{status.label}</span>
            <Info className="h-3 w-3 opacity-60" />
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="p-3">
          {tooltipContent}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Context Sufficiency Badge
 *
 * A more prominent badge-style display for context sufficiency.
 * Useful for source viewers or detailed response panels.
 */
export function ContextSufficiencyBadge({
  sufficiency,
  className,
}: {
  sufficiency: ContextSufficiencyInfo | null | undefined;
  className?: string;
}) {
  if (!sufficiency) {
    return null;
  }

  const { coverage_score, has_conflicts, confidence_level } = sufficiency;
  const coveragePercent = Math.round(coverage_score * 100);

  const getVariant = () => {
    if (has_conflicts) return "destructive";
    if (confidence_level === "high") return "secondary";
    if (confidence_level === "medium") return "outline";
    return "destructive";
  };

  const getLabel = () => {
    if (has_conflicts) return "Conflicting sources";
    if (confidence_level === "high") return `High confidence (${coveragePercent}%)`;
    if (confidence_level === "medium") return `Partial context (${coveragePercent}%)`;
    return `Limited context (${coveragePercent}%)`;
  };

  return (
    <Badge variant={getVariant()} className={cn("text-xs", className)}>
      {getLabel()}
    </Badge>
  );
}
