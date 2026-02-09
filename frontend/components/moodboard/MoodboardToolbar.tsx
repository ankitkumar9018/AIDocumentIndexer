"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Palette,
  Type,
  StickyNote,
  FileText,
  Tags,
  ImagePlus,
  Frame,
  LayoutGrid,
  Download,
  Camera,
  Save,
  Wand2,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { BackgroundPicker } from "./BackgroundPicker";
import type { BackgroundVariantName } from "./types";

interface MoodboardToolbarProps {
  onAddNode: (type: string) => void;
  onAddFrame: () => void;
  onAutoLayout: () => void;
  onSave: () => void;
  onExportJSON: () => void;
  onExportImage: () => void;
  onEnhance?: () => void;
  isSaving?: boolean;
  isEnhancing?: boolean;
  isDirty?: boolean;
  bgVariant: BackgroundVariantName;
  bgColor: string;
  bgPatternColor: string;
  onBgVariantChange: (v: BackgroundVariantName) => void;
  onBgColorChange: (c: string) => void;
  onBgPatternColorChange: (c: string) => void;
}

const ADD_ITEMS = [
  { type: "colorSwatch", label: "Color", icon: Palette, color: "text-violet-500" },
  { type: "typography", label: "Font", icon: Type, color: "text-blue-500" },
  { type: "stickyNote", label: "Sticky Note", icon: StickyNote, color: "text-yellow-500" },
  { type: "textBlock", label: "Text Block", icon: FileText, color: "text-emerald-500" },
  { type: "tagCloud", label: "Tags", icon: Tags, color: "text-orange-500" },
  { type: "imageNode", label: "Image", icon: ImagePlus, color: "text-pink-500" },
];

export function MoodboardToolbar({
  onAddNode,
  onAddFrame,
  onAutoLayout,
  onSave,
  onExportJSON,
  onExportImage,
  onEnhance,
  isSaving,
  isEnhancing,
  isDirty,
  bgVariant,
  bgColor,
  bgPatternColor,
  onBgVariantChange,
  onBgColorChange,
  onBgPatternColorChange,
}: MoodboardToolbarProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <TooltipProvider delayDuration={300}>
      <div
        className="bg-card/95 backdrop-blur-sm border rounded-lg shadow-lg flex flex-col overflow-hidden transition-all duration-200 ease-in-out"
        style={{ width: isExpanded ? 170 : 44 }}
      >
        {/* Toggle */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center justify-center h-8 hover:bg-muted/50 transition-colors border-b border-border/50"
        >
          {isExpanded ? <ChevronLeft className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        </button>

        {/* Add Elements */}
        <div className="p-1 space-y-0.5">
          {ADD_ITEMS.map((item) => (
            <Tooltip key={item.type}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => onAddNode(item.type)}
                  className="flex items-center gap-2 w-full rounded-md hover:bg-muted/80 transition-colors h-7 px-2"
                >
                  <item.icon className={`h-3.5 w-3.5 shrink-0 ${item.color}`} />
                  {isExpanded && <span className="text-xs truncate">{item.label}</span>}
                </button>
              </TooltipTrigger>
              {!isExpanded && (
                <TooltipContent side="right" className="text-xs">
                  {item.label}
                </TooltipContent>
              )}
            </Tooltip>
          ))}

          {/* Frame */}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={onAddFrame}
                className="flex items-center gap-2 w-full rounded-md hover:bg-muted/80 transition-colors h-7 px-2"
              >
                <Frame className="h-3.5 w-3.5 shrink-0 text-indigo-500" />
                {isExpanded && <span className="text-xs">Frame</span>}
              </button>
            </TooltipTrigger>
            {!isExpanded && (
              <TooltipContent side="right" className="text-xs">
                Frame
              </TooltipContent>
            )}
          </Tooltip>
        </div>

        <Separator className="mx-1" />

        {/* Tools */}
        <div className="p-1 space-y-0.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={onAutoLayout}
                className="flex items-center gap-2 w-full rounded-md hover:bg-muted/80 transition-colors h-7 px-2"
              >
                <LayoutGrid className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                {isExpanded && <span className="text-xs">Auto Layout</span>}
              </button>
            </TooltipTrigger>
            {!isExpanded && <TooltipContent side="right" className="text-xs">Auto Layout</TooltipContent>}
          </Tooltip>

          {onEnhance && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={onEnhance}
                  disabled={isEnhancing}
                  className="flex items-center gap-2 w-full rounded-md hover:bg-muted/80 transition-colors h-7 px-2 disabled:opacity-50"
                >
                  <Wand2 className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  {isExpanded && <span className="text-xs">{isEnhancing ? "Enhancing..." : "AI Enhance"}</span>}
                </button>
              </TooltipTrigger>
              {!isExpanded && <TooltipContent side="right" className="text-xs">AI Enhance</TooltipContent>}
            </Tooltip>
          )}

          {/* Background picker — only show popover trigger inline */}
          <BackgroundPicker
            variant={bgVariant}
            bgColor={bgColor}
            patternColor={bgPatternColor}
            onVariantChange={onBgVariantChange}
            onBgColorChange={onBgColorChange}
            onPatternColorChange={onBgPatternColorChange}
          />
        </div>

        <Separator className="mx-1" />

        {/* Save & Export */}
        <div className="p-1 space-y-0.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={onSave}
                disabled={isSaving}
                className={`flex items-center gap-2 w-full rounded-md transition-colors h-7 px-2 disabled:opacity-50 ${
                  isDirty ? "bg-primary/10 hover:bg-primary/20 text-primary" : "hover:bg-muted/80"
                }`}
              >
                <Save className="h-3.5 w-3.5 shrink-0" />
                {isExpanded && <span className="text-xs">{isSaving ? "Saving..." : "Save"}</span>}
              </button>
            </TooltipTrigger>
            {!isExpanded && <TooltipContent side="right" className="text-xs">Save</TooltipContent>}
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={onExportJSON}
                className="flex items-center gap-2 w-full rounded-md hover:bg-muted/80 transition-colors h-7 px-2"
              >
                <Download className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                {isExpanded && <span className="text-xs">Export JSON</span>}
              </button>
            </TooltipTrigger>
            {!isExpanded && <TooltipContent side="right" className="text-xs">Export JSON</TooltipContent>}
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={onExportImage}
                className="flex items-center gap-2 w-full rounded-md hover:bg-muted/80 transition-colors h-7 px-2"
              >
                <Camera className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                {isExpanded && <span className="text-xs">Export Image</span>}
              </button>
            </TooltipTrigger>
            {!isExpanded && <TooltipContent side="right" className="text-xs">Export Image</TooltipContent>}
          </Tooltip>
        </div>
      </div>
    </TooltipProvider>
  );
}
