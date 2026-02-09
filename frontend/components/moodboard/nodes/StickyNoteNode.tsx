"use client";

import { memo, useState, useCallback, useMemo } from "react";
import { type NodeProps, type Node, useReactFlow } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { StickyNoteData } from "../types";
import { STICKY_COLORS } from "../types";

function StickyNoteNodeBase({ id, data, selected }: NodeProps<Node<StickyNoteData>>) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(data.text);
  const { setNodes } = useReactFlow();

  const colorScheme = STICKY_COLORS[data.noteColor] || STICKY_COLORS.yellow;

  // Slight random rotation for personality
  const rotation = useMemo(() => {
    const hash = id.charCodeAt(0) + id.charCodeAt(id.length - 1);
    return ((hash % 7) - 3) * 0.5; // -1.5 to 1.5 degrees
  }, [id]);

  const commitEdit = useCallback(() => {
    setNodes((nds) => nds.map((n) => n.id === id ? { ...n, data: { ...n.data, text: editValue } } : n));
    setIsEditing(false);
  }, [id, editValue, setNodes]);

  const cycleColor = useCallback(() => {
    const colors = Object.keys(STICKY_COLORS) as Array<keyof typeof STICKY_COLORS>;
    const idx = colors.indexOf(data.noteColor);
    const next = colors[(idx + 1) % colors.length];
    setNodes((nds) => nds.map((n) => n.id === id ? { ...n, data: { ...n.data, noteColor: next } } : n));
  }, [id, data.noteColor, setNodes]);

  return (
    <div
      className={cn(
        "rounded-sm overflow-hidden shadow-lg relative group",
        selected && "ring-2 ring-blue-500 ring-offset-2"
      )}
      style={{
        width: 220,
        height: 160,
        backgroundColor: colorScheme.bg,
        borderTop: `4px solid ${colorScheme.border}`,
        transform: `rotate(${rotation}deg)`,
        boxShadow: "2px 3px 10px rgba(0,0,0,0.08), 1px 1px 3px rgba(0,0,0,0.05)",
      }}
      onDoubleClick={() => { setEditValue(data.text); setIsEditing(true); }}
    >
      {/* Color cycle button */}
      <button
        onClick={cycleColor}
        className="absolute top-1.5 right-1.5 w-4 h-4 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ backgroundColor: colorScheme.border }}
        title="Change color"
      />

      <div className="p-4 h-full flex items-start">
        {isEditing ? (
          <textarea
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={(e) => { if (e.key === "Escape") { setEditValue(data.text); setIsEditing(false); } }}
            className="w-full h-full bg-transparent resize-none border-0 outline-none text-sm text-zinc-700 leading-relaxed"
            autoFocus
          />
        ) : (
          <p className="text-sm text-zinc-700 leading-relaxed whitespace-pre-wrap">
            {data.text || "Double-click to add a note..."}
          </p>
        )}
      </div>
    </div>
  );
}

export const StickyNoteNode = memo(StickyNoteNodeBase);
