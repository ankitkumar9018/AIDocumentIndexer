"use client";

import { memo, useState, useCallback } from "react";
import { type NodeProps, type Node, useReactFlow } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { TextBlockData } from "../types";

const variantStyles = {
  inspiration: {
    accent: "border-l-amber-400",
    icon: "\u201C",
    titleColor: "text-amber-600 dark:text-amber-400",
    bg: "bg-amber-50/50 dark:bg-amber-950/20",
    contentClass: "italic",
  },
  direction: {
    accent: "border-l-blue-400",
    icon: "\u279C",
    titleColor: "text-blue-600 dark:text-blue-400",
    bg: "bg-blue-50/40 dark:bg-blue-950/20",
    contentClass: "",
  },
  system: {
    accent: "border-l-emerald-400",
    icon: "\u2699",
    titleColor: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50/40 dark:bg-emerald-950/20",
    contentClass: "font-mono text-[11px]",
  },
  "anti-patterns": {
    accent: "border-l-red-500",
    icon: "\u26A0",
    titleColor: "text-red-600 dark:text-red-400",
    bg: "bg-red-50/50 dark:bg-red-950/20",
    contentClass: "",
  },
  custom: {
    accent: "border-l-zinc-400",
    icon: "\u2022",
    titleColor: "text-zinc-600 dark:text-zinc-400",
    bg: "bg-white dark:bg-zinc-900",
    contentClass: "",
  },
};

function TextBlockNodeBase({ id, data, selected }: NodeProps<Node<TextBlockData>>) {
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(data.title);
  const [editContent, setEditContent] = useState(data.content);
  const { setNodes } = useReactFlow();

  const style = variantStyles[data.variant] || variantStyles.custom;
  const nodeWidth = data.width || 350;

  const commitEdit = useCallback(() => {
    setNodes((nds) => nds.map((n) => n.id === id ? { ...n, data: { ...n.data, title: editTitle, content: editContent } } : n));
    setIsEditing(false);
  }, [id, editTitle, editContent, setNodes]);

  return (
    <div
      className={cn(
        "rounded-xl border border-zinc-200 dark:border-zinc-700 shadow-sm overflow-hidden border-l-4",
        style.bg,
        style.accent,
        selected && "ring-2 ring-blue-500 ring-offset-2"
      )}
      style={{ width: nodeWidth, minHeight: 120 }}
      onDoubleClick={() => { setEditTitle(data.title); setEditContent(data.content); setIsEditing(true); }}
    >
      <div className="p-4">
        {isEditing ? (
          <div className="space-y-2">
            <input
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              className="w-full bg-transparent border-b border-zinc-300 dark:border-zinc-600 text-sm font-semibold text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-blue-400"
              autoFocus
            />
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              onBlur={commitEdit}
              onKeyDown={(e) => { if (e.key === "Escape") setIsEditing(false); }}
              className="w-full bg-transparent resize-none border-0 outline-none text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed min-h-[80px]"
            />
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2 mb-2">
              <span className={cn("text-xs font-semibold uppercase tracking-wider", style.titleColor)}>
                {style.icon} {data.title}
              </span>
            </div>
            <p className={cn(
              "text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed whitespace-pre-line",
              style.contentClass
            )}>
              {data.variant === "inspiration" ? `\u201C${data.content}\u201D` : data.content}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

export const TextBlockNode = memo(TextBlockNodeBase);
