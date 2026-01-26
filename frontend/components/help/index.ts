/**
 * Help Components
 * ================
 *
 * Phase 26: In-app help system components.
 *
 * Exports:
 * - HelpCenter: Full help center dialog with searchable articles
 * - HelpTrigger: Button that opens help center (responds to F1)
 * - ContextualHelp: Wrapper for adding help tooltips/popovers to any element
 * - HelpTooltip: Simple tooltip with help content
 * - HelpPopover: Rich popover with help content
 * - InlineHelp: Expandable inline help text
 * - FeatureTip: One-time feature discovery tip
 * - HelpBadge: Label with help icon
 * - HELP_CONTENT: Pre-defined help content registry
 * - getHelpContent: Helper to get help content by key
 */

export {
  HelpCenter,
  HelpTrigger,
  type HelpArticle,
  type HelpCategory,
} from "./help-center";

export {
  ContextualHelp,
  HelpTooltip,
  HelpPopover,
  InlineHelp,
  FeatureTip,
  HelpBadge,
  HELP_CONTENT,
  getHelpContent,
  type HelpContent,
} from "./contextual-help";
