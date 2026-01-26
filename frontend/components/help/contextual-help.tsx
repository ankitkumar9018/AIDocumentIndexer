"use client";

/**
 * AIDocumentIndexer - Contextual Help Tooltips
 * =============================================
 *
 * Phase 26: Enhanced tooltips with help content for UI elements.
 *
 * Features:
 * - Rich help tooltips with descriptions
 * - Keyboard shortcut hints
 * - Links to full documentation
 * - Dismissible tips for experienced users
 * - Smart positioning
 */

import * as React from "react";
import { useState, useCallback, useEffect } from "react";
import {
  HelpCircle,
  Lightbulb,
  ExternalLink,
  X,
  Keyboard,
  ChevronRight,
  Info,
  Sparkles,
} from "lucide-react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import * as PopoverPrimitive from "@radix-ui/react-popover";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

// =============================================================================
// Types
// =============================================================================

export interface HelpContent {
  title: string;
  description: string;
  tip?: string;
  shortcut?: string[];
  learnMoreUrl?: string;
  learnMoreArticle?: string;
}

interface ContextualHelpProps {
  content: HelpContent;
  children: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  showIcon?: boolean;
  iconPosition?: "before" | "after";
  variant?: "tooltip" | "popover" | "inline";
  dismissable?: boolean;
  storageKey?: string; // For remembering dismissed state
  className?: string;
}

interface HelpTooltipProps {
  content: HelpContent;
  children: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
}

interface HelpPopoverProps {
  content: HelpContent;
  children: React.ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  dismissable?: boolean;
  storageKey?: string;
}

interface InlineHelpProps {
  content: HelpContent;
  className?: string;
}

// =============================================================================
// Help Content Registry
// =============================================================================

/**
 * Pre-defined help content for common UI elements.
 * Use these IDs to get consistent help content across the app.
 */
export const HELP_CONTENT: Record<string, HelpContent> = {
  // Chat
  "chat-input": {
    title: "Ask a Question",
    description: "Type your question in natural language. The AI will search your documents and provide an answer with sources.",
    tip: "Use @mentions to reference specific documents",
    shortcut: ["Cmd", "Enter"],
    learnMoreArticle: "chat-basics",
  },
  "chat-voice": {
    title: "Voice Input",
    description: "Click to speak your question. The AI will transcribe and process your voice input.",
    shortcut: ["Cmd", "Shift", "V"],
    learnMoreArticle: "chat-basics",
  },
  "chat-sources": {
    title: "Source References",
    description: "Click on a source to view the original passage in your document. Sources are ranked by relevance.",
    tip: "Multiple sources increase answer confidence",
    learnMoreArticle: "chat-basics",
  },
  "chat-export": {
    title: "Export Conversation",
    description: "Download this conversation as PDF or Markdown for sharing or archiving.",
    shortcut: ["Cmd", "Shift", "E"],
  },

  // Upload
  "upload-dropzone": {
    title: "Upload Documents",
    description: "Drag and drop files here, or click to browse. Supports PDF, Word, PowerPoint, Excel, and text files up to 100MB.",
    tip: "Processing starts immediately - no submit button needed!",
    shortcut: ["Cmd", "U"],
    learnMoreArticle: "upload-documents",
  },
  "upload-google-drive": {
    title: "Google Drive Import",
    description: "Connect your Google Drive to import documents directly. Documents will sync automatically when updated.",
    learnMoreArticle: "upload-documents",
  },
  "upload-url": {
    title: "URL Import",
    description: "Paste a URL to import web pages as documents. Great for articles, documentation, and blog posts.",
    learnMoreArticle: "upload-documents",
  },

  // Documents
  "document-search": {
    title: "Search Documents",
    description: "Search by filename, content, or tags. Use filters to narrow results by type, date, or collection.",
    shortcut: ["Cmd", "K"],
  },
  "document-collection": {
    title: "Collections",
    description: "Organize documents into collections for focused analysis. Chat queries can be scoped to specific collections.",
    learnMoreArticle: "collections",
  },
  "document-actions": {
    title: "Document Actions",
    description: "Download, move, delete, or process documents. Select multiple with Cmd+Click for bulk operations.",
  },

  // Features
  "audio-generate": {
    title: "Generate Audio Overview",
    description: "Create a podcast-style audio summary of your documents. Perfect for reviewing on the go.",
    tip: "Works best with text-heavy documents",
    learnMoreArticle: "audio-overview",
  },
  "knowledge-graph": {
    title: "Knowledge Graph",
    description: "Visualize relationships between entities in your documents. Discover hidden connections and patterns.",
    learnMoreArticle: "knowledge-graph",
  },
  "report-generate": {
    title: "Generate Report",
    description: "Create comprehensive reports from your document knowledge. Choose from templates or customize your own.",
    learnMoreArticle: "document-generation",
  },

  // Settings
  "settings-api-key": {
    title: "API Key",
    description: "Your API key is required for AI features. Keep it secure and don't share it publicly.",
    tip: "Rotate your key periodically for security",
  },
  "settings-model": {
    title: "AI Model Selection",
    description: "Choose which AI model to use for answers. Different models have different capabilities and costs.",
    tip: "GPT-4 is more accurate but slower than GPT-3.5",
  },
  "settings-privacy": {
    title: "Privacy Settings",
    description: "Control how your data is stored and processed. Your documents are encrypted and never used for training.",
    learnMoreArticle: "faq-privacy",
  },

  // Advanced
  "query-enhancement": {
    title: "Query Enhancement",
    description: "When enabled, your questions are automatically expanded with related terms to improve search results.",
    tip: "Disable for exact phrase matching",
  },
  "context-window": {
    title: "Context Window",
    description: "The amount of document context included in AI responses. Larger windows provide more context but may be slower.",
  },
  "agent-mode": {
    title: "Agent Mode",
    description: "Enable agentic processing for complex questions. The AI will plan, search, and reason step-by-step.",
    tip: "Best for multi-step analysis tasks",
  },
};

// =============================================================================
// Contextual Help Component (Main Export)
// =============================================================================

export function ContextualHelp({
  content,
  children,
  side = "top",
  align = "center",
  showIcon = false,
  iconPosition = "after",
  variant = "tooltip",
  dismissable = false,
  storageKey,
  className,
}: ContextualHelpProps) {
  // Check if dismissed
  const [isDismissed, setIsDismissed] = useState(() => {
    if (!dismissable || !storageKey) return false;
    if (typeof window === "undefined") return false;
    return localStorage.getItem(`help-dismissed-${storageKey}`) === "true";
  });

  const handleDismiss = useCallback(() => {
    setIsDismissed(true);
    if (storageKey) {
      localStorage.setItem(`help-dismissed-${storageKey}`, "true");
    }
  }, [storageKey]);

  if (isDismissed && variant !== "tooltip") {
    return <>{children}</>;
  }

  const helpIcon = showIcon && (
    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors cursor-help" />
  );

  const wrappedChildren = showIcon ? (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      {iconPosition === "before" && helpIcon}
      {children}
      {iconPosition === "after" && helpIcon}
    </span>
  ) : (
    children
  );

  if (variant === "tooltip") {
    return (
      <HelpTooltip content={content} side={side} align={align}>
        {wrappedChildren}
      </HelpTooltip>
    );
  }

  if (variant === "popover") {
    return (
      <HelpPopover
        content={content}
        side={side}
        align={align}
        dismissable={dismissable}
        storageKey={storageKey}
      >
        {wrappedChildren}
      </HelpPopover>
    );
  }

  // Inline variant
  return (
    <div className={cn("flex items-start gap-2", className)}>
      {children}
      <InlineHelp content={content} />
    </div>
  );
}

// =============================================================================
// Help Tooltip
// =============================================================================

export function HelpTooltip({
  content,
  children,
  side = "top",
  align = "center",
}: HelpTooltipProps) {
  return (
    <TooltipPrimitive.Provider delayDuration={300}>
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            side={side}
            align={align}
            sideOffset={6}
            className={cn(
              "z-50 w-72 rounded-lg border bg-popover p-3 shadow-lg",
              "animate-in fade-in-0 zoom-in-95",
              "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
              "data-[side=bottom]:slide-in-from-top-2",
              "data-[side=left]:slide-in-from-right-2",
              "data-[side=right]:slide-in-from-left-2",
              "data-[side=top]:slide-in-from-bottom-2"
            )}
          >
            <HelpContentCard content={content} compact />
            <TooltipPrimitive.Arrow className="fill-popover" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}

// =============================================================================
// Help Popover
// =============================================================================

export function HelpPopover({
  content,
  children,
  side = "right",
  align = "start",
  dismissable = false,
  storageKey,
}: HelpPopoverProps) {
  const [open, setOpen] = useState(false);

  const handleDismiss = useCallback(() => {
    setOpen(false);
    if (storageKey) {
      localStorage.setItem(`help-dismissed-${storageKey}`, "true");
    }
  }, [storageKey]);

  return (
    <PopoverPrimitive.Root open={open} onOpenChange={setOpen}>
      <PopoverPrimitive.Trigger asChild>{children}</PopoverPrimitive.Trigger>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          side={side}
          align={align}
          sideOffset={8}
          className={cn(
            "z-50 w-80 rounded-lg border bg-popover shadow-lg",
            "animate-in fade-in-0 zoom-in-95",
            "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
            "data-[side=bottom]:slide-in-from-top-2",
            "data-[side=left]:slide-in-from-right-2",
            "data-[side=right]:slide-in-from-left-2",
            "data-[side=top]:slide-in-from-bottom-2"
          )}
        >
          <div className="p-4">
            <HelpContentCard content={content} />
          </div>
          {dismissable && (
            <div className="border-t px-4 py-2 bg-muted/50 flex justify-between items-center">
              <span className="text-xs text-muted-foreground">
                Tip for this feature
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={handleDismiss}
              >
                Don't show again
              </Button>
            </div>
          )}
          <PopoverPrimitive.Arrow className="fill-popover" />
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}

// =============================================================================
// Inline Help
// =============================================================================

export function InlineHelp({ content, className }: InlineHelpProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={cn("inline-block", className)}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="text-muted-foreground hover:text-foreground transition-colors"
        aria-label="Toggle help"
      >
        <Info className="h-4 w-4" />
      </button>
      {isExpanded && (
        <div className="mt-2 p-3 rounded-lg border bg-muted/50 text-sm">
          <HelpContentCard content={content} compact />
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Help Content Card
// =============================================================================

function HelpContentCard({
  content,
  compact = false,
}: {
  content: HelpContent;
  compact?: boolean;
}) {
  return (
    <div className={cn("space-y-2", compact && "space-y-1.5")}>
      <div className="flex items-start gap-2">
        <div className={cn("shrink-0 text-primary", compact ? "mt-0.5" : "mt-0")}>
          <Lightbulb className={compact ? "h-4 w-4" : "h-5 w-5"} />
        </div>
        <div className="flex-1 min-w-0">
          <p className={cn("font-medium", compact && "text-sm")}>{content.title}</p>
          <p
            className={cn(
              "text-muted-foreground",
              compact ? "text-xs" : "text-sm"
            )}
          >
            {content.description}
          </p>
        </div>
      </div>

      {content.tip && (
        <div
          className={cn(
            "flex items-start gap-2 rounded-md bg-primary/5 p-2",
            compact && "p-1.5"
          )}
        >
          <Sparkles
            className={cn(
              "shrink-0 text-primary",
              compact ? "h-3 w-3 mt-0.5" : "h-4 w-4"
            )}
          />
          <p className={cn("text-primary", compact ? "text-xs" : "text-sm")}>
            {content.tip}
          </p>
        </div>
      )}

      {content.shortcut && (
        <div className="flex items-center gap-2">
          <Keyboard className={cn("text-muted-foreground", compact ? "h-3 w-3" : "h-4 w-4")} />
          <div className="flex items-center gap-1">
            {content.shortcut.map((key, idx) => (
              <Badge
                key={idx}
                variant="outline"
                className={cn("font-mono", compact ? "h-5 px-1 text-[10px]" : "h-6 px-1.5 text-xs")}
              >
                {key}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {(content.learnMoreUrl || content.learnMoreArticle) && (
        <a
          href={content.learnMoreUrl || `#help-${content.learnMoreArticle}`}
          className={cn(
            "flex items-center gap-1 text-primary hover:underline",
            compact ? "text-xs" : "text-sm"
          )}
        >
          Learn more
          <ChevronRight className="h-3 w-3" />
        </a>
      )}
    </div>
  );
}

// =============================================================================
// Feature Tip Component (for onboarding/discovery)
// =============================================================================

interface FeatureTipProps {
  id: string;
  title: string;
  description: string;
  position?: "top-left" | "top-right" | "bottom-left" | "bottom-right";
  onDismiss?: () => void;
  children: React.ReactNode;
}

export function FeatureTip({
  id,
  title,
  description,
  position = "bottom-right",
  onDismiss,
  children,
}: FeatureTipProps) {
  const [isDismissed, setIsDismissed] = useState(() => {
    if (typeof window === "undefined") return true;
    return localStorage.getItem(`feature-tip-${id}`) === "true";
  });

  const handleDismiss = () => {
    setIsDismissed(true);
    localStorage.setItem(`feature-tip-${id}`, "true");
    onDismiss?.();
  };

  if (isDismissed) {
    return <>{children}</>;
  }

  const positionClasses = {
    "top-left": "bottom-full left-0 mb-2",
    "top-right": "bottom-full right-0 mb-2",
    "bottom-left": "top-full left-0 mt-2",
    "bottom-right": "top-full right-0 mt-2",
  };

  return (
    <div className="relative inline-block">
      {children}
      <div
        className={cn(
          "absolute z-50 w-64 rounded-lg border bg-popover p-3 shadow-lg",
          "animate-in fade-in-0 slide-in-from-top-2",
          positionClasses[position]
        )}
      >
        <button
          onClick={handleDismiss}
          className="absolute top-2 right-2 text-muted-foreground hover:text-foreground"
          aria-label="Dismiss tip"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="flex items-start gap-2 pr-6">
          <div className="shrink-0 h-6 w-6 rounded-full bg-primary/10 flex items-center justify-center">
            <Sparkles className="h-3 w-3 text-primary" />
          </div>
          <div>
            <p className="font-medium text-sm">{title}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
          </div>
        </div>

        <div className="flex justify-end mt-2">
          <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={handleDismiss}>
            Got it
          </Button>
        </div>

        {/* Arrow */}
        <div
          className={cn(
            "absolute w-3 h-3 rotate-45 border bg-popover",
            position.startsWith("top") && "top-full -mt-1.5 border-t-0 border-l-0",
            position.startsWith("bottom") && "bottom-full -mb-1.5 border-b-0 border-r-0",
            position.endsWith("left") && "left-4",
            position.endsWith("right") && "right-4"
          )}
        />
      </div>
    </div>
  );
}

// =============================================================================
// Help Badge Component (for labels with help)
// =============================================================================

interface HelpBadgeProps {
  helpKey: keyof typeof HELP_CONTENT;
  children: React.ReactNode;
  className?: string;
}

export function HelpBadge({ helpKey, children, className }: HelpBadgeProps) {
  const content = HELP_CONTENT[helpKey];
  if (!content) {
    return <span className={className}>{children}</span>;
  }

  return (
    <ContextualHelp content={content} showIcon iconPosition="after">
      <span className={className}>{children}</span>
    </ContextualHelp>
  );
}

// =============================================================================
// Helper to get help content by key
// =============================================================================

export function getHelpContent(key: keyof typeof HELP_CONTENT): HelpContent | undefined {
  return HELP_CONTENT[key];
}

export default ContextualHelp;
