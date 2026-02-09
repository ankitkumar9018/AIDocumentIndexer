"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  useNodesState,
  BackgroundVariant,
  type Node,
  type NodeChange,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ArrowLeft, Maximize2, Minimize2, Download, Camera, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { moodboardNodeTypes } from "./nodes";
import { MoodboardToolbar } from "./MoodboardToolbar";
import { autoLayoutMoodboard } from "./MoodboardAutoLayout";
import type {
  MoodboardBoard,
  CanvasData,
  ColorSwatchData,
  TypographyData,
  TagCloudData,
  FrameNodeData,
  BackgroundVariantName,
} from "./types";
import { api } from "@/lib/api/client";

// ── Props ──────────────────────────────────────────────────────────

interface MoodboardCanvasProps {
  board: MoodboardBoard;
  onClose: () => void;
  onBoardUpdate?: (board: MoodboardBoard) => void;
}

// ── Background variant mapping ──────────────────────────────────────
const BG_VARIANT_MAP: Record<string, BackgroundVariant> = {
  dots: BackgroundVariant.Dots,
  lines: BackgroundVariant.Lines,
  cross: BackgroundVariant.Cross,
};

// ── Inner component (needs ReactFlow context) ──────────────────────

function MoodboardCanvasInner({ board, onClose, onBoardUpdate }: MoodboardCanvasProps) {
  const { fitView, getViewport } = useReactFlow();

  // Initialize nodes from canvas_data or auto-layout
  const initialNodes = useMemo(() => {
    if (board.canvas_data?.nodes?.length) {
      return board.canvas_data.nodes.map((n) => ({
        ...n,
        // Restore frame dimensions and z-ordering
        ...(n.type === "frame"
          ? { style: { width: n.width || 800, height: n.height || 600 }, zIndex: -1 }
          : {}),
      })) as Node[];
    }
    return autoLayoutMoodboard(board);
  }, [board]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [boardName, setBoardName] = useState(board.name);
  const [boardDesc, setBoardDesc] = useState(board.description || "");
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  // Background state
  const [bgVariant, setBgVariant] = useState<BackgroundVariantName>(
    board.canvas_data?.background?.variant || "dots"
  );
  const [bgColor, setBgColor] = useState(board.canvas_data?.background?.color || "#ffffff");
  const [bgPatternColor, setBgPatternColor] = useState(
    board.canvas_data?.background?.patternColor || "#e5e7eb"
  );

  // Frame context menu
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);

  // ── Serialize canvas data ──
  const buildCanvasData = useCallback((): CanvasData => {
    const viewport = getViewport();
    return {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.type || "colorSwatch",
        position: n.position,
        data: n.data as Record<string, unknown>,
        width: n.measured?.width || (n.style?.width as number | undefined),
        height: n.measured?.height || (n.style?.height as number | undefined),
        parentId: n.parentId,
        extent: n.extent === "parent" ? "parent" : undefined,
        expandParent: n.expandParent,
        style: n.type === "frame" ? { width: n.style?.width, height: n.style?.height } : undefined,
      })),
      viewport: { x: viewport.x, y: viewport.y, zoom: viewport.zoom },
      version: 2,
      background: { variant: bgVariant, color: bgColor, patternColor: bgPatternColor },
    };
  }, [nodes, getViewport, bgVariant, bgColor, bgPatternColor]);

  // Track node changes for dirty state
  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    onNodesChange(changes);
    const hasPositionChange = changes.some((c) => c.type === "position" && c.dragging === false);
    if (hasPositionChange) setIsDirty(true);
  }, [onNodesChange]);

  // Auto-save canvas positions (debounced 3s)
  useEffect(() => {
    if (!isDirty || !board.id) return;
    if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current);
    autoSaveTimer.current = setTimeout(async () => {
      try {
        await api.put(`/moodboard/${board.id}/canvas`, { canvas_data: buildCanvasData() });
        setIsDirty(false);
      } catch {
        // Silent fail — auto-save is best-effort
      }
    }, 3000);
    return () => { if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current); };
  }, [isDirty, nodes, board.id, buildCanvasData]);

  // ── Add Node ──
  const handleAddNode = useCallback((type: string) => {
    const viewport = getViewport();
    const centerX = (-viewport.x + 400) / viewport.zoom;
    const centerY = (-viewport.y + 300) / viewport.zoom;
    const id = `${type}_${Date.now()}`;

    const defaultData: Record<string, Record<string, unknown>> = {
      colorSwatch: { color: "#6366f1", label: "New Color", role: "accent" },
      typography: { fontFamily: "Inter", role: "body" },
      stickyNote: { text: "", noteColor: "yellow" },
      textBlock: { title: "New Note", content: "", variant: "custom" },
      tagCloud: { tags: ["new tag"], variant: "dark" },
      imageNode: { url: "", caption: "" },
    };

    setNodes((nds) => [...nds, { id, type, position: { x: centerX, y: centerY }, data: defaultData[type] || {} }]);
    setIsDirty(true);
  }, [getViewport, setNodes]);

  // ── Add Frame ──
  const handleAddFrame = useCallback(() => {
    const viewport = getViewport();
    const centerX = (-viewport.x + 200) / viewport.zoom;
    const centerY = (-viewport.y + 150) / viewport.zoom;
    const id = `frame_${Date.now()}`;

    const newFrame: Node = {
      id,
      type: "frame",
      position: { x: centerX, y: centerY },
      data: {
        title: "New Frame",
        backgroundColor: "rgba(248,250,252,0.5)",
        borderColor: "#d1d5db",
        showTitle: true,
      } satisfies FrameNodeData,
      style: { width: 800, height: 600 },
      zIndex: -1,
    };

    setNodes((nds) => [newFrame, ...nds]);
    setIsDirty(true);
  }, [getViewport, setNodes]);

  // ── Node-into-frame detection ──
  const handleNodeDragStop = useCallback((_event: React.MouseEvent, node: Node) => {
    if (node.type === "frame") return;

    const frames = nodes.filter((n) => n.type === "frame");
    let newParentId: string | undefined;

    for (const frame of frames) {
      const fw = (frame.style?.width as number) || frame.measured?.width || 800;
      const fh = (frame.style?.height as number) || frame.measured?.height || 600;

      // Check if node center is inside frame (use absolute positions)
      const nodeAbsX = node.parentId ? node.position.x + (frames.find(f => f.id === node.parentId)?.position.x || 0) : node.position.x;
      const nodeAbsY = node.parentId ? node.position.y + (frames.find(f => f.id === node.parentId)?.position.y || 0) : node.position.y;

      if (
        nodeAbsX >= frame.position.x &&
        nodeAbsY >= frame.position.y &&
        nodeAbsX <= frame.position.x + fw &&
        nodeAbsY <= frame.position.y + fh
      ) {
        newParentId = frame.id;
        break;
      }
    }

    if (newParentId && node.parentId !== newParentId) {
      // Reparent: convert to frame-relative position
      const frame = frames.find((f) => f.id === newParentId)!;
      const oldFrame = node.parentId ? frames.find((f) => f.id === node.parentId) : null;
      const absX = oldFrame ? node.position.x + oldFrame.position.x : node.position.x;
      const absY = oldFrame ? node.position.y + oldFrame.position.y : node.position.y;

      setNodes((nds) => nds.map((n) =>
        n.id === node.id
          ? { ...n, parentId: newParentId, extent: "parent" as const, expandParent: true, position: { x: absX - frame.position.x, y: absY - frame.position.y } }
          : n
      ));
      setIsDirty(true);
    } else if (!newParentId && node.parentId) {
      // Unparent: convert back to absolute
      const oldFrame = frames.find((f) => f.id === node.parentId);
      if (oldFrame) {
        setNodes((nds) => nds.map((n) =>
          n.id === node.id
            ? { ...n, parentId: undefined, extent: undefined, expandParent: undefined, position: { x: n.position.x + oldFrame.position.x, y: n.position.y + oldFrame.position.y } }
            : n
        ));
        setIsDirty(true);
      }
    }
  }, [nodes, setNodes]);

  // ── Auto Layout ──
  const handleAutoLayout = useCallback(() => {
    const layoutNodes = autoLayoutMoodboard(board);
    setNodes(layoutNodes);
    setIsDirty(true);
    setTimeout(() => fitView({ padding: 0.2 }), 100);
  }, [board, setNodes, fitView]);

  // ── Save ──
  const handleSave = useCallback(async () => {
    setIsSaving(true);
    try {
      const colorNodes = nodes.filter((n) => n.type === "colorSwatch");
      const fontNodes = nodes.filter((n) => n.type === "typography");
      const tagNodes = nodes.filter((n) => n.type === "tagCloud");

      const colors = colorNodes.map((n) => (n.data as ColorSwatchData).color);
      const themes = fontNodes.map((n) => (n.data as TypographyData).fontFamily);
      const tags = tagNodes.flatMap((n) => (n.data as TagCloudData).tags);

      const updates: Record<string, unknown> = {
        name: boardName,
        description: boardDesc,
        color_palette: colors.length > 0 ? colors : undefined,
        themes: themes.length > 0 ? themes : undefined,
        style_tags: tags.length > 0 ? tags : undefined,
        canvas_data: buildCanvasData(),
      };

      const resp = await api.patch(`/moodboard/${board.id}`, updates);
      setIsDirty(false);
      toast.success("Board saved");
      if (onBoardUpdate) onBoardUpdate(resp.data as MoodboardBoard);
    } catch {
      toast.error("Failed to save");
    }
    setIsSaving(false);
  }, [nodes, boardName, boardDesc, board.id, buildCanvasData, onBoardUpdate]);

  // ── Enhance ──
  const handleEnhance = useCallback(async () => {
    setIsEnhancing(true);
    try {
      const resp = await api.post(`/moodboard/${board.id}/enhance`);
      const updated = resp.data as MoodboardBoard;
      const layoutNodes = autoLayoutMoodboard(updated);
      setNodes(layoutNodes);
      setIsDirty(true);
      toast.success("Board enhanced with AI insights");
      if (onBoardUpdate) onBoardUpdate(updated);
    } catch {
      toast.error("Enhancement failed");
    }
    setIsEnhancing(false);
  }, [board.id, setNodes, onBoardUpdate]);

  // ── Export JSON ──
  const handleExportJSON = useCallback(() => {
    const blob = new Blob([JSON.stringify(board, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `moodboard-${boardName.replace(/\s+/g, "-").toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [board, boardName]);

  // ── Export Image ──
  const handleExportImage = useCallback(async () => {
    if (!canvasRef.current) return;
    try {
      const html2canvas = (await import("html2canvas")).default;
      const canvas = await html2canvas(canvasRef.current, { scale: 2, useCORS: true });
      const url = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = url;
      a.download = `moodboard-${boardName.replace(/\s+/g, "-").toLowerCase()}.png`;
      a.click();
      toast.success("Image exported");
    } catch {
      toast.error("Failed to export image");
    }
  }, [boardName]);

  // ── Export Frame JSON ──
  const handleExportFrameJSON = useCallback((frameId: string) => {
    const frameNode = nodes.find((n) => n.id === frameId);
    if (!frameNode) return;
    const childNodes = nodes.filter((n) => n.parentId === frameId);
    const frameData = { frame: frameNode, children: childNodes, exportedAt: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(frameData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `frame-${((frameNode.data as FrameNodeData).title || "frame").replace(/\s+/g, "-").toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [nodes]);

  // ── Delete Frame (and unparent children) ──
  const handleDeleteFrame = useCallback((frameId: string) => {
    const frame = nodes.find((n) => n.id === frameId);
    if (!frame) return;
    setNodes((nds) => nds
      .filter((n) => n.id !== frameId)
      .map((n) => n.parentId === frameId
        ? { ...n, parentId: undefined, extent: undefined, expandParent: undefined, position: { x: n.position.x + frame.position.x, y: n.position.y + frame.position.y } }
        : n
      )
    );
    setIsDirty(true);
  }, [nodes, setNodes]);

  // ── Frame context menu ──
  const handleNodeContextMenu = useCallback((event: React.MouseEvent, node: Node) => {
    if (node.type !== "frame") return;
    event.preventDefault();
    setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
  }, []);

  // Close context menu on click
  useEffect(() => {
    if (!contextMenu) return;
    const handler = () => setContextMenu(null);
    window.addEventListener("click", handler);
    return () => window.removeEventListener("click", handler);
  }, [contextMenu]);

  // ── Escape key for fullscreen ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape" && isFullscreen) setIsFullscreen(false); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isFullscreen]);

  // ── Background change handlers ──
  const handleBgVariantChange = useCallback((v: BackgroundVariantName) => { setBgVariant(v); setIsDirty(true); }, []);
  const handleBgColorChange = useCallback((c: string) => { setBgColor(c); setIsDirty(true); }, []);
  const handleBgPatternColorChange = useCallback((c: string) => { setBgPatternColor(c); setIsDirty(true); }, []);

  // ── MiniMap node color ──
  const miniMapNodeColor = useCallback((node: Node) => {
    switch (node.type) {
      case "colorSwatch": return (node.data as ColorSwatchData).color || "#6366f1";
      case "typography": return "#f3f4f6";
      case "stickyNote": return "#fef9c3";
      case "textBlock": return "#ffffff";
      case "tagCloud": return "#1f2937";
      case "imageNode": return "#dbeafe";
      case "frame": return "rgba(99,102,241,0.15)";
      default: return "#e5e7eb";
    }
  }, []);

  const canvasContent = (
    <div
      ref={canvasRef}
      className={`w-full ${isFullscreen ? "h-screen" : "h-[calc(100vh-12rem)]"} rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700`}
      style={{ backgroundColor: bgColor }}
    >
      <ReactFlow
        nodes={nodes}
        edges={[]}
        onNodesChange={handleNodesChange}
        onNodeDragStop={handleNodeDragStop}
        onNodeContextMenu={handleNodeContextMenu}
        nodeTypes={moodboardNodeTypes}
        fitView
        deleteKeyCode={["Backspace", "Delete"]}
        multiSelectionKeyCode="Shift"
        panOnScroll
        selectionOnDrag
        proOptions={{ hideAttribution: true }}
      >
        {bgVariant !== "none" && BG_VARIANT_MAP[bgVariant] && (
          <Background variant={BG_VARIANT_MAP[bgVariant]} gap={20} size={bgVariant === "lines" ? 0.5 : 1} color={bgPatternColor} />
        )}
        <Controls position="bottom-right" showInteractive={false} />
        <MiniMap
          nodeColor={miniMapNodeColor}
          position="bottom-left"
          pannable
          zoomable
          style={{ width: 160, height: 100 }}
        />

        {/* ── Toolbar Panel ── */}
        <Panel position="top-left">
          <MoodboardToolbar
            onAddNode={handleAddNode}
            onAddFrame={handleAddFrame}
            onAutoLayout={handleAutoLayout}
            onSave={handleSave}
            onExportJSON={handleExportJSON}
            onExportImage={handleExportImage}
            onEnhance={handleEnhance}
            isSaving={isSaving}
            isEnhancing={isEnhancing}
            isDirty={isDirty}
            bgVariant={bgVariant}
            bgColor={bgColor}
            bgPatternColor={bgPatternColor}
            onBgVariantChange={handleBgVariantChange}
            onBgColorChange={handleBgColorChange}
            onBgPatternColorChange={handleBgPatternColorChange}
          />
        </Panel>

        {/* ── Header Panel ── */}
        <Panel position="top-right">
          <div className="bg-card/95 backdrop-blur-sm border rounded-lg p-3 shadow-sm space-y-2 max-w-[260px]">
            <Input
              value={boardName}
              onChange={(e) => { setBoardName(e.target.value); setIsDirty(true); }}
              className="font-semibold text-sm h-8"
              placeholder="Board name"
            />
            <Input
              value={boardDesc}
              onChange={(e) => { setBoardDesc(e.target.value); setIsDirty(true); }}
              className="text-xs h-7 text-muted-foreground"
              placeholder="Description..."
            />
            <div className="flex gap-1.5">
              <Button variant="outline" size="sm" className="text-xs h-7 flex-1" onClick={() => setIsFullscreen(!isFullscreen)}>
                {isFullscreen ? <Minimize2 className="h-3 w-3 mr-1" /> : <Maximize2 className="h-3 w-3 mr-1" />}
                {isFullscreen ? "Exit" : "Fullscreen"}
              </Button>
              <Button variant="ghost" size="sm" className="text-xs h-7" onClick={onClose}>
                <ArrowLeft className="h-3 w-3 mr-1" /> Back
              </Button>
            </div>
            {isDirty && (
              <p className="text-[10px] text-amber-500 text-center">Unsaved changes</p>
            )}
          </div>
        </Panel>
      </ReactFlow>

      {/* ── Frame Context Menu ── */}
      {contextMenu && (
        <div
          className="fixed z-[100] bg-card border rounded-lg shadow-lg p-1 min-w-[160px] animate-in fade-in-0 zoom-in-95"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            className="flex items-center gap-2 w-full text-left text-xs px-3 py-1.5 rounded hover:bg-muted transition-colors"
            onClick={() => { handleExportFrameJSON(contextMenu.nodeId); setContextMenu(null); }}
          >
            <Download className="h-3 w-3" /> Export Frame JSON
          </button>
          <button
            className="flex items-center gap-2 w-full text-left text-xs px-3 py-1.5 rounded hover:bg-muted transition-colors"
            onClick={() => { handleExportImage(); setContextMenu(null); }}
          >
            <Camera className="h-3 w-3" /> Export Canvas Image
          </button>
          <div className="h-px bg-border my-1" />
          <button
            className="flex items-center gap-2 w-full text-left text-xs px-3 py-1.5 rounded hover:bg-destructive/10 text-destructive transition-colors"
            onClick={() => { handleDeleteFrame(contextMenu.nodeId); setContextMenu(null); }}
          >
            <Trash2 className="h-3 w-3" /> Delete Frame
          </button>
        </div>
      )}
    </div>
  );

  if (isFullscreen) {
    return (
      <div className="fixed inset-0 z-50" style={{ backgroundColor: bgColor }}>
        {canvasContent}
      </div>
    );
  }

  return canvasContent;
}

// ── Wrapper with ReactFlowProvider ──────────────────────────────────

export function MoodboardCanvas(props: MoodboardCanvasProps) {
  return (
    <ReactFlowProvider>
      <MoodboardCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
