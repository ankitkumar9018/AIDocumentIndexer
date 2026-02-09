"use client";

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Paintbrush } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BackgroundVariantName } from "./types";

interface BackgroundPickerProps {
  variant: BackgroundVariantName;
  bgColor: string;
  patternColor: string;
  onVariantChange: (v: BackgroundVariantName) => void;
  onBgColorChange: (color: string) => void;
  onPatternColorChange: (color: string) => void;
}

const PATTERN_PRESETS: { id: BackgroundVariantName; label: string; icon: string }[] = [
  { id: "dots", label: "Dots", icon: "\u2022\u2022\u2022" },
  { id: "lines", label: "Grid", icon: "\u2500\u2502" },
  { id: "cross", label: "Cross", icon: "\u253C" },
  { id: "none", label: "Clean", icon: "\u2014" },
];

const BG_PRESETS = [
  { color: "#ffffff", label: "White" },
  { color: "#fafafa", label: "Light" },
  { color: "#f1f5f9", label: "Slate" },
  { color: "#0f172a", label: "Navy" },
  { color: "#18181b", label: "Dark" },
  { color: "#fefce8", label: "Cream" },
];

export function BackgroundPicker({
  variant,
  bgColor,
  patternColor,
  onVariantChange,
  onBgColorChange,
  onPatternColorChange,
}: BackgroundPickerProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="w-full justify-start text-xs h-7">
          <Paintbrush className="h-3.5 w-3.5 mr-2 text-cyan-500" /> Background
        </Button>
      </PopoverTrigger>
      <PopoverContent side="right" align="start" className="w-56 p-3 space-y-3">
        {/* Pattern */}
        <div>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest mb-1.5">Pattern</p>
          <div className="grid grid-cols-4 gap-1">
            {PATTERN_PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => onVariantChange(p.id)}
                className={cn(
                  "h-8 rounded border text-xs flex flex-col items-center justify-center gap-0.5 transition-colors",
                  variant === p.id
                    ? "border-blue-400 bg-blue-50 dark:bg-blue-950 text-blue-600"
                    : "border-zinc-200 dark:border-zinc-700 hover:bg-muted text-muted-foreground"
                )}
              >
                <span className="text-[10px] leading-none">{p.icon}</span>
                <span className="text-[8px]">{p.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Canvas Color */}
        <div>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest mb-1.5">Canvas</p>
          <div className="flex gap-1 items-center">
            {BG_PRESETS.map((p) => (
              <button
                key={p.color}
                onClick={() => onBgColorChange(p.color)}
                title={p.label}
                className={cn(
                  "w-6 h-6 rounded-full border-2 transition-transform hover:scale-110",
                  bgColor === p.color ? "border-blue-400 scale-110" : "border-zinc-300 dark:border-zinc-600"
                )}
                style={{ backgroundColor: p.color }}
              />
            ))}
            <input
              type="color"
              value={bgColor}
              onChange={(e) => onBgColorChange(e.target.value)}
              className="w-6 h-6 rounded-full cursor-pointer border-0 p-0"
              title="Custom color"
            />
          </div>
        </div>

        {/* Pattern Color */}
        {variant !== "none" && (
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest mb-1.5">Pattern Color</p>
            <div className="flex items-center gap-2">
              <input
                type="color"
                value={patternColor}
                onChange={(e) => onPatternColorChange(e.target.value)}
                className="w-7 h-7 rounded cursor-pointer border border-zinc-200 dark:border-zinc-700 p-0"
              />
              <span className="text-[10px] font-mono text-muted-foreground">{patternColor}</span>
            </div>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
