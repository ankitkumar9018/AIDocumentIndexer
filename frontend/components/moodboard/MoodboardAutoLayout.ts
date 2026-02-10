import type { MoodboardBoard } from "./types";
import type { Node } from "@xyflow/react";

const SECTION_GAP = 80;
const COLOR_W = 180;
const COLOR_H = 180;
const COLOR_OVERLAP = 140; // overlapping fan effect (reduced dynamically for large palettes)
const FONT_W = 300;
const FONT_H = 160;
const FONT_STAGGER = 20; // cascade offset
const TEXT_W = 350;
const TEXT_H = 140;
const TAG_W = 260;
const TAG_H = 140;
const FRAME_PAD = 30;
const IMG_REF_W = 220;
const IMG_REF_H = 160;
const STICKY_W = 160;
const STICKY_H = 120;

/**
 * Convert flat moodboard data into positioned ReactFlow nodes.
 * Used when a board has no canvas_data (legacy boards or "Auto Layout").
 *
 * v2: Staggered typography, overlapping swatches, auto-frame wrapping.
 * v3: Image reference nodes, palette-accented frames, improved titles.
 * v4: Sticky notes, frame minHeight for expandParent, reduced swatch overlap,
 *     Design System frame wrapper, increased section gap.
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
    (suggestions.anti_patterns?.length ? 1 : 0) +
    (Array.isArray(suggestions.image_search_terms) && suggestions.image_search_terms.length > 0 ? 1 : 0);

  const useFrames = totalElements >= 5;

  // Use first palette color as frame accent (fallback to neutral gray)
  const frameAccentColor = (board.color_palette && board.color_palette.length > 0)
    ? board.color_palette[0]
    : "#e5e7eb";

  // ── Row 1: Color swatches ──
  // Merge user palette with AI additional_colors (dedup by hex)
  const baseColors = board.color_palette || [];
  const additionalColors: Array<{ hex: string; role?: string; name?: string }> =
    Array.isArray(suggestions.additional_colors) ? suggestions.additional_colors : [];
  const existingHexes = new Set(baseColors.map((c: string) => c.toLowerCase()));
  const mergedAdditional = additionalColors
    .filter((ac) => typeof ac === "object" && ac.hex && !existingHexes.has(ac.hex.toLowerCase()))
    .slice(0, 4); // Max 4 additional
  const colors = [...baseColors, ...mergedAdditional.map((ac) => ac.hex)];
  // Build a map for additional color metadata (role, name) for label assignment
  const additionalColorMeta: Record<string, { role?: string; name?: string }> = {};
  mergedAdditional.forEach((ac) => { additionalColorMeta[ac.hex.toLowerCase()] = { role: ac.role, name: ac.name }; });
  if (colors.length > 0) {
    const psych = suggestions.color_psychology || {};
    const colorStartY = y;
    // Reduce overlap for large palettes so role badges stay readable
    const effectiveOverlap = colors.length > 5 ? 100 : COLOR_OVERLAP;
    const paletteWidth = FRAME_PAD * 2 + (colors.length - 1) * effectiveOverlap + COLOR_W;
    const paletteHeight = FRAME_PAD * 2 + COLOR_H + 20;

    // Frame FIRST (parent before children)
    if (useFrames) {
      nodes.push({
        id: "frame_palette",
        type: "frame",
        position: { x: 0, y: colorStartY },
        data: { title: "Color Palette", backgroundColor: "#fafafa", borderColor: frameAccentColor, showTitle: true },
        style: { width: paletteWidth, minHeight: paletteHeight },
        zIndex: -1,
      });
    }

    // Then children
    colors.forEach((color, i) => {
      const psychInfo = psych[color];
      const isRichPsych = typeof psychInfo === "object" && psychInfo !== null;
      const meaning = isRichPsych ? psychInfo?.meaning : psychInfo;
      const pairWith = isRichPsych ? psychInfo?.pair_with : undefined;
      const pairContext = isRichPsych ? psychInfo?.pair_context : undefined;

      // Use metadata from additional colors if available, otherwise default labels
      const acMeta = additionalColorMeta[color.toLowerCase()];
      const defaultLabel = i === 0 ? "Primary" : i === 1 ? "Secondary" : `Accent ${i - 1}`;
      const defaultRole = i === 0 ? "primary" : i === 1 ? "secondary" : "accent";

      nodes.push({
        id: `color_${i}`,
        type: "colorSwatch",
        position: useFrames
          ? { x: FRAME_PAD + i * effectiveOverlap, y: FRAME_PAD }
          : { x: i * (effectiveOverlap + 40), y },
        data: {
          color,
          label: acMeta?.name || defaultLabel,
          role: acMeta?.role || defaultRole,
          psychology: meaning || undefined,
          pairWith: pairWith || undefined,
          pairContext: pairContext || undefined,
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
    const typoHeight = FRAME_PAD * 2 + FONT_H + (fonts.length - 1) * FONT_STAGGER + 20;

    // Frame FIRST
    if (useFrames) {
      nodes.push({
        id: "frame_typography",
        type: "frame",
        position: { x: 0, y: typoStartY },
        data: { title: "Typography & Type System", backgroundColor: "#fafafa", borderColor: frameAccentColor, showTitle: true },
        style: { width: typoWidth, minHeight: typoHeight },
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
    const frameHeight = FRAME_PAD * 2 + Math.max(TAG_H, TEXT_H) + 20;

    if (useFrames) {
      // Frame FIRST
      nodes.push({
        id: "frame_direction",
        type: "frame",
        position: { x: 0, y: directionStartY },
        data: { title: "Creative Direction & Mood", backgroundColor: "#fafafa", borderColor: frameAccentColor, showTitle: true },
        style: { width: frameWidth, minHeight: frameHeight },
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

  // ── Row 4: Design system + anti-patterns (now in a frame) ──
  const designSystem = suggestions.design_system;
  const antiPatterns = suggestions.anti_patterns;
  const hasDesignSystem = designSystem && typeof designSystem === "object";
  const hasAntiPatterns = Array.isArray(antiPatterns) && antiPatterns.length > 0;

  if (hasDesignSystem || hasAntiPatterns) {
    const systemChildren: Node[] = [];
    let sx = FRAME_PAD;

    if (hasDesignSystem) {
      const systemText = Object.entries(designSystem)
        .map(([key, val]) => `${key.replace(/_/g, " ")}: ${val}`)
        .join("\n");
      systemChildren.push({
        id: "text_system",
        type: "textBlock",
        position: { x: sx, y: FRAME_PAD },
        data: {
          title: "Design System",
          content: systemText,
          variant: "system" as const,
        },
      });
      sx += TEXT_W + 30;
    }

    if (hasAntiPatterns) {
      systemChildren.push({
        id: "text_antipatterns",
        type: "textBlock",
        position: { x: sx, y: FRAME_PAD },
        data: {
          title: "Avoid",
          content: antiPatterns.map((a: string) => `\u2022 ${a}`).join("\n"),
          variant: "anti-patterns" as const,
        },
      });
      sx += TEXT_W + 30;
    }

    const systemFrameWidth = sx + FRAME_PAD;
    const systemFrameHeight = FRAME_PAD * 2 + TEXT_H + 20;

    if (useFrames) {
      // Frame FIRST
      nodes.push({
        id: "frame_system",
        type: "frame",
        position: { x: 0, y },
        data: { title: "Design System & Guidelines", backgroundColor: "#fafafa", borderColor: frameAccentColor, showTitle: true },
        style: { width: systemFrameWidth, minHeight: systemFrameHeight },
        zIndex: -1,
      });
      systemChildren.forEach((n) => {
        n.parentId = "frame_system";
        n.extent = "parent" as const;
        n.expandParent = true;
      });
    } else {
      // Offset to absolute coords when not using frames
      systemChildren.forEach((n) => {
        n.position.y += y;
        n.position.x -= FRAME_PAD;
      });
    }
    nodes.push(...systemChildren);
    y += (useFrames ? systemFrameHeight : TEXT_H) + SECTION_GAP;
  }

  // ── Row 5: Mood sticky notes ──
  const moodKeywords: string[] = suggestions.mood_keywords || [];
  if (moodKeywords.length > 0) {
    const stickyColors: Array<"yellow" | "pink" | "blue" | "green" | "purple" | "orange"> =
      ["yellow", "pink", "blue", "green", "purple", "orange"];

    const stickyChildren: Node[] = [];
    let stx = FRAME_PAD;

    moodKeywords.slice(0, 6).forEach((kw: string, i: number) => {
      // Stagger odd items for a more organic, hand-placed feel
      const yOffset = i % 2 === 1 ? 15 : 0;
      const xOffset = i % 2 === 1 ? 10 : 0;
      stickyChildren.push({
        id: `sticky_mood_${i}`,
        type: "stickyNote",
        position: { x: stx + xOffset, y: FRAME_PAD + yOffset },
        data: { text: kw, noteColor: stickyColors[i % stickyColors.length] },
      });
      stx += STICKY_W + 15;
    });

    if (stickyChildren.length > 0) {
      const stickyFrameWidth = stx + FRAME_PAD;
      const stickyFrameHeight = FRAME_PAD * 2 + STICKY_H + 30; // extra for stagger

      if (useFrames) {
        nodes.push({
          id: "frame_mood",
          type: "frame",
          position: { x: 0, y },
          data: { title: "Mood & Themes", backgroundColor: "#fafafa", borderColor: frameAccentColor, showTitle: true },
          style: { width: stickyFrameWidth, minHeight: stickyFrameHeight },
          zIndex: -1,
        });
        stickyChildren.forEach((n) => {
          n.parentId = "frame_mood";
          n.extent = "parent" as const;
          n.expandParent = true;
        });
      } else {
        stickyChildren.forEach((n) => {
          n.position.y += y;
          n.position.x -= FRAME_PAD;
        });
      }
      nodes.push(...stickyChildren);
      y += (useFrames ? stickyFrameHeight : STICKY_H) + SECTION_GAP;
    }
  }

  // ── Row 6: Image search inspiration ──
  const imageTerms = suggestions.image_search_terms;
  if (Array.isArray(imageTerms) && imageTerms.length > 0) {
    const imgChildren: Node[] = [];
    let ix = FRAME_PAD;

    imageTerms.slice(0, 5).forEach((term: string, i: number) => {
      imgChildren.push({
        id: `image_search_${i}`,
        type: "textBlock",
        position: { x: ix, y: FRAME_PAD },
        data: {
          title: `Reference ${i + 1}`,
          content: term,
          variant: "inspiration" as const,
          width: IMG_REF_W,
        },
      });
      ix += IMG_REF_W + 20;
    });

    if (imgChildren.length > 0) {
      const frameWidth = ix + FRAME_PAD;
      const frameHeight = FRAME_PAD * 2 + IMG_REF_H + 20;

      if (useFrames) {
        nodes.push({
          id: "frame_imagery",
          type: "frame",
          position: { x: 0, y },
          data: { title: "Image References & Inspiration", backgroundColor: "#fafafa", borderColor: frameAccentColor, showTitle: true },
          style: { width: frameWidth, minHeight: frameHeight },
          zIndex: -1,
        });
        imgChildren.forEach((n) => {
          n.parentId = "frame_imagery";
          n.extent = "parent" as const;
          n.expandParent = true;
        });
      } else {
        imgChildren.forEach((n) => {
          n.position.y += y;
          n.position.x -= FRAME_PAD;
        });
      }
      nodes.push(...imgChildren);
    }
  }

  return nodes;
}
