"use client";

import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

interface QueryEnhancementToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  variant?: "inline" | "popover";
  disabled?: boolean;
  className?: string;
}

export function QueryEnhancementToggle({
  enabled,
  onChange,
  variant = "inline",
  disabled = false,
  className,
}: QueryEnhancementToggleProps) {
  if (variant === "inline") {
    return (
      <Button
        variant={enabled ? "default" : "outline"}
        size="sm"
        onClick={() => onChange(!enabled)}
        disabled={disabled}
        className={cn("gap-1.5 h-8", className)}
        title={enabled ? "Query enhancement enabled - click to disable" : "Enable query enhancement"}
      >
        <Sparkles className={cn("h-4 w-4", enabled && "text-yellow-300")} />
        <span className="hidden sm:inline">Enhance</span>
      </Button>
    );
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant={enabled ? "default" : "outline"}
          size="sm"
          className={cn("gap-2", className)}
          disabled={disabled}
        >
          <Sparkles className={cn("h-4 w-4", enabled && "text-yellow-300")} />
          Query Enhancement
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-72" align="end">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Query Enhancement</span>
            <Switch checked={enabled} onCheckedChange={onChange} />
          </div>
          <p className="text-xs text-muted-foreground">
            Improves search by expanding queries with synonyms and generating
            hypothetical document embeddings (HyDE) for better semantic matching.
          </p>
        </div>
      </PopoverContent>
    </Popover>
  );
}
