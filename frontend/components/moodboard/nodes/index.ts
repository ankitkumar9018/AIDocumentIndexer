export { ColorSwatchNode } from "./ColorSwatchNode";
export { TypographyNode } from "./TypographyNode";
export { StickyNoteNode } from "./StickyNoteNode";
export { TextBlockNode } from "./TextBlockNode";
export { TagCloudNode } from "./TagCloudNode";
export { ImageNode } from "./ImageNode";
export { FrameNode } from "./FrameNode";

import { ColorSwatchNode } from "./ColorSwatchNode";
import { TypographyNode } from "./TypographyNode";
import { StickyNoteNode } from "./StickyNoteNode";
import { TextBlockNode } from "./TextBlockNode";
import { TagCloudNode } from "./TagCloudNode";
import { ImageNode } from "./ImageNode";
import { FrameNode } from "./FrameNode";

export const moodboardNodeTypes = {
  colorSwatch: ColorSwatchNode,
  typography: TypographyNode,
  stickyNote: StickyNoteNode,
  textBlock: TextBlockNode,
  tagCloud: TagCloudNode,
  imageNode: ImageNode,
  frame: FrameNode,
} as const;
