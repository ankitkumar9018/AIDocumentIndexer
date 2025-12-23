"use client";

import { cn } from "@/lib/utils";
import { Mic, Volume2, Loader2, CircleOff } from "lucide-react";

export type VoiceState = "idle" | "listening" | "thinking" | "speaking";

interface VoiceConversationIndicatorProps {
  state: VoiceState;
  className?: string;
}

const stateConfig = {
  idle: {
    icon: CircleOff,
    label: "Voice mode off",
    bgColor: "bg-muted",
    iconColor: "text-muted-foreground",
    animate: false,
  },
  listening: {
    icon: Mic,
    label: "Listening...",
    bgColor: "bg-red-500/10",
    iconColor: "text-red-500",
    animate: true,
  },
  thinking: {
    icon: Loader2,
    label: "Processing...",
    bgColor: "bg-yellow-500/10",
    iconColor: "text-yellow-500",
    animate: true,
  },
  speaking: {
    icon: Volume2,
    label: "Speaking...",
    bgColor: "bg-blue-500/10",
    iconColor: "text-blue-500",
    animate: true,
  },
};

export function VoiceConversationIndicator({
  state,
  className,
}: VoiceConversationIndicatorProps) {
  const config = stateConfig[state];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-3 py-1.5 rounded-full transition-all duration-300",
        config.bgColor,
        className
      )}
    >
      <div className={cn("relative", config.animate && "animate-pulse")}>
        <Icon
          className={cn(
            "h-4 w-4 transition-colors",
            config.iconColor,
            state === "thinking" && "animate-spin"
          )}
        />
        {state === "listening" && (
          <>
            <span className="absolute inset-0 rounded-full bg-red-500/20 animate-ping" />
            <span className="absolute inset-0 rounded-full bg-red-500/10 animate-pulse" />
          </>
        )}
      </div>
      <span
        className={cn(
          "text-xs font-medium transition-colors",
          config.iconColor
        )}
      >
        {config.label}
      </span>
    </div>
  );
}
