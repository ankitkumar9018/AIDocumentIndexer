import type { Node } from "@xyflow/react";

// ── Node Data Interfaces ──────────────────────────────────────────

export interface ColorSwatchData extends Record<string, unknown> {
  color: string;
  label?: string;
  role?: string;       // "primary" | "secondary" | "accent" | "background" | "text"
  psychology?: string;
  pairWith?: string;
  pairContext?: string;
}

export interface TypographyData extends Record<string, unknown> {
  fontFamily: string;
  role?: string;       // "heading" | "body" | "accent" | "display"
  rationale?: string;
  sampleText?: string;
}

export interface StickyNoteData extends Record<string, unknown> {
  text: string;
  noteColor: "yellow" | "pink" | "blue" | "green" | "purple" | "orange";
}

export interface TextBlockData extends Record<string, unknown> {
  title: string;
  content: string;
  variant: "inspiration" | "direction" | "system" | "anti-patterns" | "custom";
}

export interface TagCloudData extends Record<string, unknown> {
  tags: string[];
  variant: "dark" | "light";
}

export interface ImageNodeData extends Record<string, unknown> {
  url: string;
  caption?: string;
}

export interface FrameNodeData extends Record<string, unknown> {
  title: string;
  backgroundColor: string;
  borderColor: string;
  showTitle: boolean;
}

// ── Node Types ────────────────────────────────────────────────────

export type MoodboardNodeData =
  | ColorSwatchData
  | TypographyData
  | StickyNoteData
  | TextBlockData
  | TagCloudData
  | ImageNodeData
  | FrameNodeData;

export type MoodboardNode = Node<MoodboardNodeData>;

export const NODE_TYPES = {
  colorSwatch: "colorSwatch",
  typography: "typography",
  stickyNote: "stickyNote",
  textBlock: "textBlock",
  tagCloud: "tagCloud",
  imageNode: "imageNode",
  frame: "frame",
} as const;

// ── Canvas Persistence ────────────────────────────────────────────

export type BackgroundVariantName = "dots" | "lines" | "cross" | "none";

export interface CanvasBackground {
  variant: BackgroundVariantName;
  color: string;         // canvas background color
  patternColor: string;  // pattern element color
}

export interface CanvasData {
  nodes: Array<{
    id: string;
    type: string;
    position: { x: number; y: number };
    data: Record<string, unknown>;
    width?: number;
    height?: number;
    parentId?: string;
    extent?: "parent" | null;
    expandParent?: boolean;
    style?: Record<string, unknown>;
  }>;
  viewport: { x: number; y: number; zoom: number };
  version: number;
  background?: CanvasBackground;
}

export interface MoodboardBoard {
  id: string;
  name: string;
  description?: string;
  color_palette: string[];
  themes: string[];
  style_tags: string[];
  generated_suggestions: Record<string, any>;
  canvas_data?: CanvasData | null;
  created_at: string;
  updated_at: string;
}

// ── Sticky Note Colors ────────────────────────────────────────────

export const STICKY_COLORS = {
  yellow:  { bg: "#fef9c3", border: "#facc15" },
  pink:    { bg: "#fce7f3", border: "#f472b6" },
  blue:    { bg: "#dbeafe", border: "#60a5fa" },
  green:   { bg: "#dcfce7", border: "#4ade80" },
  purple:  { bg: "#ede9fe", border: "#a78bfa" },
  orange:  { bg: "#ffedd5", border: "#fb923c" },
} as const;
