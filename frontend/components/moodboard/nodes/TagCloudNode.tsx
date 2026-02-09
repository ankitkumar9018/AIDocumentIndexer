"use client";

import { memo, useState, useCallback } from "react";
import { type NodeProps, type Node, useReactFlow } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import type { TagCloudData } from "../types";

function TagCloudNodeBase({ id, data, selected }: NodeProps<Node<TagCloudData>>) {
  const [newTag, setNewTag] = useState("");
  const [isAdding, setIsAdding] = useState(false);
  const { setNodes } = useReactFlow();

  const removeTag = useCallback((idx: number) => {
    setNodes((nds) => nds.map((n) => {
      if (n.id !== id) return n;
      const tags = [...(n.data as TagCloudData).tags];
      tags.splice(idx, 1);
      return { ...n, data: { ...n.data, tags } };
    }));
  }, [id, setNodes]);

  const addTag = useCallback(() => {
    if (!newTag.trim()) return;
    setNodes((nds) => nds.map((n) => {
      if (n.id !== id) return n;
      return { ...n, data: { ...n.data, tags: [...(n.data as TagCloudData).tags, newTag.trim()] } };
    }));
    setNewTag("");
  }, [id, newTag, setNodes]);

  const isDark = data.variant === "dark";

  return (
    <div
      className={cn(
        "rounded-2xl overflow-hidden p-4 flex flex-col items-center justify-center shadow-sm",
        isDark ? "bg-zinc-900 dark:bg-zinc-950" : "bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700",
        selected && "ring-2 ring-blue-500 ring-offset-2"
      )}
      style={{ width: 260, minHeight: 140 }}
    >
      <div className="flex flex-wrap gap-1.5 justify-center">
        {data.tags.map((tag, i) => (
          <span
            key={`${tag}-${i}`}
            className={cn(
              "inline-flex items-center gap-1 px-2 py-0.5 rounded-full group/tag",
              i === 0
                ? (isDark ? "text-sm font-semibold text-white/90 bg-white/10" : "text-sm font-semibold text-zinc-800 bg-zinc-100")
                : (isDark ? "text-[11px] text-white/50 bg-white/5" : "text-[11px] text-zinc-500 bg-zinc-50")
            )}
            style={{ textTransform: "uppercase", letterSpacing: "0.15em" }}
          >
            {tag}
            <button
              onClick={() => removeTag(i)}
              className={cn(
                "opacity-0 group-hover/tag:opacity-100 transition-opacity",
                isDark ? "text-white/30 hover:text-red-400" : "text-zinc-400 hover:text-red-500"
              )}
            >
              <X className="h-2.5 w-2.5" />
            </button>
          </span>
        ))}
      </div>

      {/* Add tag */}
      {isAdding ? (
        <div className="mt-2 flex gap-1">
          <input
            type="text"
            value={newTag}
            onChange={(e) => setNewTag(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { addTag(); } if (e.key === "Escape") setIsAdding(false); }}
            onBlur={() => { if (newTag.trim()) addTag(); setIsAdding(false); }}
            placeholder="tag..."
            className={cn(
              "text-[10px] rounded px-2 py-0.5 w-20 text-center border-0 outline-none focus:ring-1",
              isDark ? "bg-white/10 text-white/80 placeholder:text-white/30 focus:ring-white/30" : "bg-zinc-100 text-zinc-700 placeholder:text-zinc-400 focus:ring-zinc-300"
            )}
            autoFocus
          />
        </div>
      ) : (
        <button
          onClick={() => setIsAdding(true)}
          className={cn(
            "mt-2 text-[10px] opacity-0 group-hover:opacity-100 transition-opacity",
            isDark ? "text-white/30 hover:text-white/50" : "text-zinc-400 hover:text-zinc-600"
          )}
        >
          + add tag
        </button>
      )}
    </div>
  );
}

export const TagCloudNode = memo(TagCloudNodeBase);
