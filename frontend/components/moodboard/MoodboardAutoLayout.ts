import type { MoodboardBoard } from "./types";
import type { Node } from "@xyflow/react";

const SECTION_GAP = 60;
const COLOR_W = 180;
const COLOR_H = 180;
const COLOR_OVERLAP = 140; // overlapping fan effect
const FONT_W = 300;
const FONT_H = 160;
const FONT_STAGGER = 20; // cascade offset
const TEXT_W = 350;
const TEXT_H = 140;
const TAG_W = 260;
const TAG_H = 140;
const FRAME_PAD = 30;

/**
 * Convert flat moodboard data into positioned ReactFlow nodes.
 * Used when a board has no canvas_data (legacy boards or "Auto Layout").
 *
 * v2: Staggered typography, overlapping swatches, auto-frame wrapping.
 *
 * IMPORTANT: Parent (frame) nodes MUST appear before their children in
 * the returned array — ReactFlow requires this ordering.
 */
export function autoLayoutMoodboard(board: MoodboardBoard): Node[] {
  const nodes: Node[] = [];
  let y = 0;

  const suggestions = board.generated_suggestions || {};
  const totalElements =
    (board.color_palette?.length || 0) +
    (board.themes?.length || 0) +
    (board.style_tags?.length ? 1 : 0) +
    (suggestions.visual_narrative || suggestions.inspiration_notes ? 1 : 0) +
    (suggestions.design_direction ? 1 : 0) +
    (suggestions.design_system ? 1 : 0) +
    (suggestions.anti_patterns?.length ? 1 : 0);

  const useFrames = totalElements >= 5;

  // ── Row 1: Color swatches ──
  const colors = board.color_palette || [];
  if (colors.length > 0) {
    const psych = suggestions.color_psychology || {};
    const colorStartY = y;
    const paletteWidth = FRAME_PAD * 2 + (colors.length - 1) * COLOR_OVERLAP + COLOR_W;
    const paletteHeight = FRAME_PAD * 2 + COLOR_H;

    // Frame FIRST (parent before children)
    if (useFrames) {
      nodes.push({
        id: "frame_palette",
        type: "frame",
        position: { x: 0, y: colorStartY },
        data: { title: "Color Palette", backgroundColor: "#fafafa", borderColor: "#e5e7eb", showTitle: true },
        style: { width: paletteWidth, height: paletteHeight },
        zIndex: -1,
      });
    }

    // Then children
    colors.forEach((color, i) => {
      const psychInfo = psych[color];
      const meaning = typeof psychInfo === "object" ? psychInfo?.meaning : psychInfo;

      nodes.push({
        id: `color_${i}`,
        type: "colorSwatch",
        position: useFrames
          ? { x: FRAME_PAD + i * COLOR_OVERLAP, y: FRAME_PAD }
          : { x: i * (COLOR_OVERLAP + 40), y },
        data: {
          color,
          label: i === 0 ? "Primary" : i === 1 ? "Secondary" : `Accent ${i - 1}`,
          role: i === 0 ? "primary" : i === 1 ? "secondary" : "accent",
          psychology: meaning || undefined,
        },
        ...(useFrames ? { parentId: "frame_palette", extent: "parent" as const, expandParent: true } : {}),
      });
    });

    y += (useFrames ? paletteHeight : COLOR_H) + SECTION_GAP;
  }

  // ── Row 2: Typography (staggered cascade) ──
  const fonts = board.themes || [];
  if (fonts.length > 0) {
    const richTypo = suggestions.typography || [];
    const typoStartY = y;
    const typoWidth = FRAME_PAD * 2 + (fonts.length - 1) * (FONT_W + 20) + FONT_W;
    const typoHeight = FRAME_PAD * 2 + FONT_H + (fonts.length - 1) * FONT_STAGGER;

    // Frame FIRST
    if (useFrames) {
      nodes.push({
        id: "frame_typography",
        type: "frame",
        position: { x: 0, y: typoStartY },
        data: { title: "Typography", backgroundColor: "#fafafa", borderColor: "#e5e7eb", showTitle: true },
        style: { width: typoWidth, height: typoHeight },
        zIndex: -1,
      });
    }

    // Then children
    fonts.forEach((font, i) => {
      const richEntry = Array.isArray(richTypo) && richTypo[i];
      const isRich = richEntry && typeof richEntry === "object";

      nodes.push({
        id: `typography_${i}`,
        type: "typography",
        position: useFrames
          ? { x: FRAME_PAD + i * (FONT_W + 20), y: FRAME_PAD + i * FONT_STAGGER }
          : { x: i * (FONT_W + 30), y: typoStartY + i * FONT_STAGGER },
        data: {
          fontFamily: font,
          role: isRich ? richEntry.role : (i === 0 ? "heading" : i === 1 ? "body" : "accent"),
          rationale: isRich ? richEntry.rationale : undefined,
          sampleText: isRich ? richEntry.sample_text : undefined,
        },
        ...(useFrames ? { parentId: "frame_typography", extent: "parent" as const, expandParent: true } : {}),
      });
    });

    y += (useFrames ? typoHeight : FONT_H) + SECTION_GAP;
  }

  // ── Row 3: Creative direction (tags + text blocks) ──
  const directionChildren: Node[] = [];
  let dx = FRAME_PAD;
  const directionStartY = y;

  // Tags
  const tags = board.style_tags || [];
  if (tags.length > 0) {
    directionChildren.push({
      id: "tagcloud_0",
      type: "tagCloud",
      position: { x: dx, y: FRAME_PAD },
      data: { tags, variant: "dark" as const },
    });
    dx += TAG_W + 30;
  }

  // Inspiration / visual narrative
  const narrative = suggestions.visual_narrative || suggestions.inspiration_notes;
  if (narrative) {
    directionChildren.push({
      id: "text_inspiration",
      type: "textBlock",
      position: { x: dx, y: FRAME_PAD },
      data: {
        title: "Inspiration",
        content: narrative,
        variant: "inspiration" as const,
      },
    });
    dx += TEXT_W + 30;
  }

  // Design direction
  const direction = suggestions.design_direction;
  if (direction) {
    directionChildren.push({
      id: "text_direction",
      type: "textBlock",
      position: { x: dx, y: FRAME_PAD },
      data: {
        title: "Design Direction",
        content: direction,
        variant: "direction" as const,
      },
    });
    dx += TEXT_W + 30;
  }

  if (directionChildren.length > 0) {
    const frameWidth = dx + FRAME_PAD;
    const frameHeight = FRAME_PAD * 2 + Math.max(TAG_H, TEXT_H);

    if (useFrames) {
      // Frame FIRST
      nodes.push({
        id: "frame_direction",
        type: "frame",
        position: { x: 0, y: directionStartY },
        data: { title: "Creative Direction", backgroundColor: "#fafafa", borderColor: "#e5e7eb", showTitle: true },
        style: { width: frameWidth, height: frameHeight },
        zIndex: -1,
      });
      directionChildren.forEach((n) => {
        n.parentId = "frame_direction";
        n.extent = "parent" as const;
        n.expandParent = true;
      });
    } else {
      // Offset to absolute coords when not using frames
      directionChildren.forEach((n) => {
        n.position.y += directionStartY;
        n.position.x -= FRAME_PAD;
      });
    }
    // Then children
    nodes.push(...directionChildren);
    y += (useFrames ? frameHeight : Math.max(TAG_H, TEXT_H)) + SECTION_GAP;
  }

  // ── Row 4: Design system + anti-patterns ──
  let sx = 0;

  const designSystem = suggestions.design_system;
  if (designSystem && typeof designSystem === "object") {
    const systemText = Object.entries(designSystem)
      .map(([key, val]) => `${key.replace(/_/g, " ")}: ${val}`)
      .join("\n");
    nodes.push({
      id: "text_system",
      type: "textBlock",
      position: { x: sx, y },
      data: {
        title: "Design System",
        content: systemText,
        variant: "system" as const,
      },
    });
    sx += TEXT_W + 30;
  }

  const antiPatterns = suggestions.anti_patterns;
  if (Array.isArray(antiPatterns) && antiPatterns.length > 0) {
    nodes.push({
      id: "text_antipatterns",
      type: "textBlock",
      position: { x: sx, y },
      data: {
        title: "Avoid",
        content: antiPatterns.map((a: string) => `\u2022 ${a}`).join("\n"),
        variant: "anti-patterns" as const,
      },
    });
  }

  return nodes;
}
