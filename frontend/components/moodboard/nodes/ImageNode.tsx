"use client";

import { memo, useState, useCallback } from "react";
import { type NodeProps, type Node, useReactFlow } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { ImagePlus } from "lucide-react";
import type { ImageNodeData } from "../types";

function ImageNodeBase({ id, data, selected }: NodeProps<Node<ImageNodeData>>) {
  const [isEditingCaption, setIsEditingCaption] = useState(false);
  const [captionValue, setCaptionValue] = useState(data.caption || "");
  const { setNodes } = useReactFlow();

  const updateCaption = useCallback(() => {
    setNodes((nds) => nds.map((n) => n.id === id ? { ...n, data: { ...n.data, caption: captionValue } } : n));
    setIsEditingCaption(false);
  }, [id, captionValue, setNodes]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file?.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
      setNodes((nds) => nds.map((n) => n.id === id ? { ...n, data: { ...n.data, url: reader.result as string } } : n));
    };
    reader.readAsDataURL(file);
  }, [id, setNodes]);

  return (
    <div
      className={cn(
        "rounded-2xl overflow-hidden shadow-sm border border-zinc-200 dark:border-zinc-700 group relative",
        selected && "ring-2 ring-blue-500 ring-offset-2"
      )}
      style={{ width: 250, height: 250 }}
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      {data.url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={data.url}
          alt={data.caption || "Moodboard image"}
          className="w-full h-full object-cover"
        />
      ) : (
        <div className="w-full h-full bg-zinc-100 dark:bg-zinc-800 flex flex-col items-center justify-center text-zinc-400">
          <ImagePlus className="h-8 w-8 mb-2" />
          <span className="text-xs">Drop an image here</span>
        </div>
      )}

      {/* Caption overlay */}
      {(data.caption || isEditingCaption) && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-3">
          {isEditingCaption ? (
            <input
              type="text"
              value={captionValue}
              onChange={(e) => setCaptionValue(e.target.value)}
              onBlur={updateCaption}
              onKeyDown={(e) => e.key === "Enter" && updateCaption()}
              className="w-full bg-transparent text-white text-xs border-b border-white/40 focus:outline-none"
              autoFocus
            />
          ) : (
            <p
              className="text-xs text-white/80 cursor-pointer"
              onDoubleClick={() => { setCaptionValue(data.caption || ""); setIsEditingCaption(true); }}
            >
              {data.caption}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export const ImageNode = memo(ImageNodeBase);
