"use client";

import { memo, useState, useCallback } from "react";
import { type NodeProps, type Node, useReactFlow } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { TypographyData } from "../types";

function TypographyNodeBase({ id, data, selected }: NodeProps<Node<TypographyData>>) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(data.fontFamily);
  const { setNodes } = useReactFlow();

  const commitEdit = useCallback(() => {
    if (editValue.trim()) {
      setNodes((nds) => nds.map((n) => n.id === id ? { ...n, data: { ...n.data, fontFamily: editValue.trim() } } : n));
    }
    setIsEditing(false);
  }, [id, editValue, setNodes]);

  const fontUrl = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(data.fontFamily)}:wght@300;400;700&display=swap`;

  return (
    <div
      className={cn(
        "rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 shadow-sm overflow-hidden flex flex-col items-center justify-center p-4 group relative",
        selected && "ring-2 ring-blue-500 ring-offset-2"
      )}
      style={{ width: 300, height: 160 }}
      onDoubleClick={() => { setEditValue(data.fontFamily); setIsEditing(true); }}
    >
      {/* eslint-disable-next-line @next/next/no-page-custom-font */}
      <link href={fontUrl} rel="stylesheet" />

      {isEditing ? (
        <input
          type="text"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={(e) => e.key === "Enter" && commitEdit()}
          className="text-center bg-transparent border-b-2 border-blue-400 text-zinc-800 dark:text-zinc-200 focus:outline-none text-2xl w-full"
          style={{ fontFamily: `'${editValue}', sans-serif` }}
          autoFocus
        />
      ) : (
        <>
          <p
            style={{ fontFamily: `'${data.fontFamily}', sans-serif`, fontSize: data.sampleText ? "clamp(20px, 3vw, 32px)" : "clamp(28px, 4vw, 48px)", fontWeight: 700, lineHeight: 1.1 }}
            className="text-zinc-800 dark:text-zinc-200 select-none text-center"
          >
            {data.sampleText || data.fontFamily.split(" ")[0]}
          </p>
          <p className="text-[9px] text-zinc-400 dark:text-zinc-500 mt-2 tracking-[0.3em] uppercase">
            {data.fontFamily}
          </p>
        </>
      )}

      {/* Role badge */}
      {data.role && (
        <span className="absolute top-2 right-2 text-[9px] uppercase tracking-wider text-zinc-400 bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
          {data.role}
        </span>
      )}

      {/* Rationale on hover */}
      {data.rationale && (
        <div className="absolute bottom-0 left-0 right-0 bg-white/95 dark:bg-zinc-900/95 p-2 opacity-0 group-hover:opacity-100 transition-opacity border-t border-zinc-100 dark:border-zinc-800">
          <p className="text-[10px] text-zinc-500 dark:text-zinc-400 italic leading-snug">{data.rationale}</p>
        </div>
      )}
    </div>
  );
}

export const TypographyNode = memo(TypographyNodeBase);
