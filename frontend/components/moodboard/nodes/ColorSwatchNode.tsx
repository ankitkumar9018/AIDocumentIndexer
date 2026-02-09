"use client";

import { memo, useState, useCallback } from "react";
import { type NodeProps, type Node, useReactFlow } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { ColorSwatchData } from "../types";

function ColorSwatchNodeBase({ id, data, selected }: NodeProps<Node<ColorSwatchData>>) {
  const [isEditing, setIsEditing] = useState(false);
  const { setNodes } = useReactFlow();

  const updateColor = useCallback((color: string) => {
    setNodes((nds) => nds.map((n) => n.id === id ? { ...n, data: { ...n.data, color } } : n));
  }, [id, setNodes]);

  return (
    <div
      className={cn(
        "rounded-2xl overflow-hidden shadow-md transition-all relative group",
        selected && "ring-2 ring-blue-500 ring-offset-2"
      )}
      style={{ width: 180, height: 180, backgroundColor: data.color }}
      onDoubleClick={() => setIsEditing(true)}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent pointer-events-none" />

      {/* Role badge */}
      {data.role && (
        <span className="absolute top-2 right-2 text-[9px] uppercase tracking-wider text-white/60 bg-black/20 px-1.5 py-0.5 rounded backdrop-blur-sm">
          {data.role}
        </span>
      )}

      {/* Hex label + psychology */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/30 to-transparent p-3 opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="text-xs font-mono text-white/90 drop-shadow-sm">{data.color}</span>
        {data.label && <span className="text-[10px] text-white/70 block">{data.label}</span>}
        {data.psychology && <span className="text-[9px] text-white/50 block mt-0.5 italic">{data.psychology}</span>}
      </div>

      {/* Color picker overlay */}
      {isEditing && (
        <input
          type="color"
          value={data.color}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          autoFocus
          onChange={(e) => updateColor(e.target.value)}
          onBlur={() => setIsEditing(false)}
        />
      )}
    </div>
  );
}

export const ColorSwatchNode = memo(ColorSwatchNodeBase);
