"use client";

import { memo, useState, useCallback } from "react";
import { type NodeProps, type Node, useReactFlow, NodeResizer } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { FrameNodeData } from "../types";

function FrameNodeBase({ id, data, selected }: NodeProps<Node<FrameNodeData>>) {
  const [isEditing, setIsEditing] = useState(false);
  const [titleValue, setTitleValue] = useState(data.title);
  const { setNodes } = useReactFlow();

  const commitTitle = useCallback(() => {
    if (titleValue.trim()) {
      setNodes((nds) => nds.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, title: titleValue.trim() } } : n
      ));
    }
    setIsEditing(false);
  }, [id, titleValue, setNodes]);

  return (
    <>
      <NodeResizer
        minWidth={300}
        minHeight={200}
        isVisible={selected}
        lineClassName="!border-blue-400"
        handleClassName="!h-3 !w-3 !bg-white !border-2 !border-blue-400 !rounded"
      />
      <div
        className={cn(
          "w-full h-full rounded-xl border-2 border-dashed overflow-visible",
          selected ? "border-blue-400/60" : "border-zinc-300/60 dark:border-zinc-600/60"
        )}
        style={{ backgroundColor: data.backgroundColor || "transparent" }}
      >
        {/* Title bar — above the frame */}
        {data.showTitle !== false && (
          <div
            className="absolute -top-7 left-3 flex items-center gap-2"
            onDoubleClick={() => { setTitleValue(data.title); setIsEditing(true); }}
          >
            {isEditing ? (
              <input
                type="text"
                value={titleValue}
                onChange={(e) => setTitleValue(e.target.value)}
                onBlur={commitTitle}
                onKeyDown={(e) => { if (e.key === "Enter") commitTitle(); if (e.key === "Escape") setIsEditing(false); }}
                className="text-xs font-semibold bg-card/90 backdrop-blur-sm border rounded px-2 py-0.5 outline-none focus:border-blue-400 text-zinc-700 dark:text-zinc-300"
                autoFocus
              />
            ) : (
              <span className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 select-none tracking-wide uppercase">
                {data.title}
              </span>
            )}
          </div>
        )}
      </div>
    </>
  );
}

export const FrameNode = memo(FrameNodeBase);
