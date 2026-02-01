"use client";

import React, { useState, useMemo } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  FileText,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  BookOpen,
  Sparkles,
  Eye,
  Link2,
  Hash,
  Layers,
} from "lucide-react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Source {
  id?: string;
  filename: string;
  pageNumber?: number;
  snippet: string;
  fullContent?: string;
  similarity: number;
  collection?: string;
  documentId?: string;
  chunkIndex?: number;
  metadata?: {
    author?: string;
    createdAt?: string;
    updatedAt?: string;
    fileType?: string;
    wordCount?: number;
    [key: string]: unknown;
  };
}

export interface InlineCitationProps {
  citationNumber: number;
  source: Source;
  onCitationClick?: (citationNumber: number) => void;
  onViewInDocument?: (documentId: string, pageNumber?: number) => void;
}

/** Segment produced by `parseCitations`. */
export interface TextSegment {
  type: "text" | "citation";
  content: string;
  citationNumber?: number;
}

// ---------------------------------------------------------------------------
// parseCitations – splits markdown response text into text / citation parts
// ---------------------------------------------------------------------------

/**
 * Takes markdown response text and finds citation patterns like `[1]`, `[2]`,
 * etc.  Only bare `[N]` references with 1–2 digit numbers are treated as
 * citations.  Brackets that are part of markdown links (`[text](url)`) or
 * labelled source references (`[Source 1:]`) are left as plain text.
 *
 * Returns an ordered array of `TextSegment` objects that can be mapped over to
 * render mixed text + citation badges.
 */
export function parseCitations(text: string): TextSegment[] {
  const citationPattern = /\[(\d{1,2})\]/g;
  const segments: TextSegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationPattern.exec(text)) !== null) {
    const fullMatch = match[0];
    const digitStr = match[1];
    const matchStart = match.index;
    const matchEnd = matchStart + fullMatch.length;

    // Skip if this looks like a markdown link: [1](http://...)
    if (text[matchEnd] === "(") {
      continue;
    }

    // Skip if preceded by word character (part of a label)
    if (matchStart > 0 && /\w/.test(text[matchStart - 1])) {
      continue;
    }

    if (matchStart > lastIndex) {
      segments.push({
        type: "text",
        content: text.slice(lastIndex, matchStart),
      });
    }

    segments.push({
      type: "citation",
      content: fullMatch,
      citationNumber: parseInt(digitStr, 10),
    });

    lastIndex = matchEnd;
  }

  if (lastIndex < text.length) {
    segments.push({
      type: "text",
      content: text.slice(lastIndex),
    });
  }

  return segments;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Return relevance tier based on similarity score */
function getRelevanceTier(similarity: number): {
  label: string;
  color: string;
  bgColor: string;
} {
  if (similarity >= 0.9) {
    return {
      label: "Highly Relevant",
      color: "text-green-600 dark:text-green-400",
      bgColor: "bg-green-500",
    };
  }
  if (similarity >= 0.75) {
    return {
      label: "Relevant",
      color: "text-emerald-600 dark:text-emerald-400",
      bgColor: "bg-emerald-500",
    };
  }
  if (similarity >= 0.5) {
    return {
      label: "Somewhat Relevant",
      color: "text-yellow-600 dark:text-yellow-400",
      bgColor: "bg-yellow-500",
    };
  }
  return {
    label: "Low Relevance",
    color: "text-orange-600 dark:text-orange-400",
    bgColor: "bg-orange-500",
  };
}

/** Format a 0–1 similarity score as a rounded percentage string. */
function formatSimilarity(similarity: number): string {
  return `${Math.round(similarity * 100)}%`;
}

/** Truncate a string to a given length adding an ellipsis when needed. */
function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength).trimEnd() + "\u2026";
}

/** Get file type icon color */
function getFileTypeColor(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "pdf":
      return "text-red-500";
    case "doc":
    case "docx":
      return "text-blue-500";
    case "xls":
    case "xlsx":
      return "text-green-500";
    case "md":
      return "text-purple-500";
    case "txt":
      return "text-gray-500";
    default:
      return "text-muted-foreground";
  }
}

// ---------------------------------------------------------------------------
// RelevanceBar - Visual similarity score
// ---------------------------------------------------------------------------

function RelevanceBar({ similarity }: { similarity: number }) {
  const tier = getRelevanceTier(similarity);
  const percentage = Math.round(similarity * 100);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">Relevance</span>
        <span className={cn("font-medium", tier.color)}>
          {formatSimilarity(similarity)}
        </span>
      </div>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn("h-full transition-all", tier.bgColor)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className={cn("text-[10px]", tier.color)}>{tier.label}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ExpandableSnippet - Collapsible full content view
// ---------------------------------------------------------------------------

function ExpandableSnippet({
  snippet,
  fullContent,
}: {
  snippet: string;
  fullContent?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const displayContent = isExpanded && fullContent ? fullContent : snippet;
  const canExpand = fullContent && fullContent.length > snippet.length;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(fullContent || snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const textArea = document.createElement("textarea");
      textArea.value = fullContent || snippet;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground flex items-center gap-1">
          <BookOpen className="h-3 w-3" />
          Source Context
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
          onClick={handleCopy}
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 mr-1 text-green-500" />
              Copied
            </>
          ) : (
            <>
              <Copy className="h-3 w-3 mr-1" />
              Copy
            </>
          )}
        </Button>
      </div>

      <div
        className={cn(
          "text-xs leading-relaxed p-2 rounded-md bg-muted/50 border",
          isExpanded ? "max-h-60 overflow-auto" : "max-h-24 overflow-hidden"
        )}
      >
        <p className="whitespace-pre-wrap">{displayContent}</p>
      </div>

      {canExpand && (
        <Button
          variant="ghost"
          size="sm"
          className="w-full h-6 text-xs text-muted-foreground hover:text-foreground"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Show Less
            </>
          ) : (
            <>
              <ChevronRight className="h-3 w-3 mr-1" />
              Show Full Context
            </>
          )}
        </Button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// InlineCitation component - Enhanced with Sparkpage-style features
// ---------------------------------------------------------------------------

export function InlineCitation({
  citationNumber,
  source,
  onCitationClick,
  onViewInDocument,
}: InlineCitationProps) {
  const handleClick = () => {
    onCitationClick?.(citationNumber);
  };

  const handleViewInDocument = () => {
    if (source.documentId && onViewInDocument) {
      onViewInDocument(source.documentId, source.pageNumber);
    }
  };

  const tier = getRelevanceTier(source.similarity);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          onClick={handleClick}
          className={cn(
            "inline-flex items-center justify-center min-w-[18px] h-[18px] px-1",
            "text-[10px] font-bold rounded-full",
            "bg-primary/20 text-primary hover:bg-primary/30",
            "cursor-pointer align-super mx-0.5 transition-all",
            "hover:scale-110 hover:shadow-sm"
          )}
          aria-label={`Citation ${citationNumber}`}
        >
          {citationNumber}
        </button>
      </PopoverTrigger>

      <PopoverContent
        side="top"
        align="center"
        sideOffset={8}
        className="w-96 p-0"
      >
        {/* Header */}
        <div className="p-3 border-b bg-muted/30">
          <div className="flex items-start gap-2">
            <FileText
              className={cn(
                "h-5 w-5 mt-0.5 shrink-0",
                getFileTypeColor(source.filename)
              )}
            />
            <div className="flex-1 min-w-0">
              <p className="font-medium text-sm leading-tight truncate">
                {source.filename}
              </p>
              <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                {source.pageNumber !== undefined && (
                  <span className="flex items-center gap-0.5">
                    <Hash className="h-3 w-3" />
                    Page {source.pageNumber}
                  </span>
                )}
                {source.chunkIndex !== undefined && (
                  <span className="flex items-center gap-0.5">
                    <Layers className="h-3 w-3" />
                    Chunk {source.chunkIndex + 1}
                  </span>
                )}
              </div>
            </div>
            <Badge
              variant="secondary"
              className={cn("text-[10px] shrink-0", tier.color)}
            >
              <Sparkles className="h-2.5 w-2.5 mr-0.5" />
              {formatSimilarity(source.similarity)}
            </Badge>
          </div>
        </div>

        {/* Body */}
        <div className="p-3 space-y-3">
          {/* Collection Tag */}
          {source.collection && (
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                <Link2 className="h-3 w-3 mr-1" />
                {source.collection}
              </Badge>
              {source.metadata?.fileType && (
                <Badge variant="secondary" className="text-xs uppercase">
                  {source.metadata.fileType}
                </Badge>
              )}
            </div>
          )}

          {/* Relevance Visualization */}
          <RelevanceBar similarity={source.similarity} />

          {/* Expandable Content */}
          <ExpandableSnippet
            snippet={source.snippet}
            fullContent={source.fullContent}
          />

          {/* Metadata */}
          {source.metadata && (
            <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground pt-2 border-t">
              {source.metadata.author && (
                <div>
                  <span className="font-medium">Author:</span>{" "}
                  {source.metadata.author}
                </div>
              )}
              {source.metadata.wordCount && (
                <div>
                  <span className="font-medium">Words:</span>{" "}
                  {source.metadata.wordCount.toLocaleString()}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="p-2 border-t bg-muted/20 flex items-center gap-2">
          {source.documentId && onViewInDocument && (
            <Button
              variant="outline"
              size="sm"
              className="flex-1 h-7 text-xs"
              onClick={handleViewInDocument}
            >
              <Eye className="h-3 w-3 mr-1" />
              View in Document
            </Button>
          )}
          <Button variant="outline" size="sm" className="h-7 text-xs" asChild>
            <a
              href={`/dashboard/documents?id=${source.documentId || ""}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <ExternalLink className="h-3 w-3 mr-1" />
              Open
            </a>
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}

// ---------------------------------------------------------------------------
// CitationGroup - Group citations by document (Sparkpage-style)
// ---------------------------------------------------------------------------

interface CitationGroupProps {
  sources: Array<Source & { citationNumber: number }>;
  onCitationClick?: (citationNumber: number) => void;
  onViewInDocument?: (documentId: string, pageNumber?: number) => void;
}

export function CitationGroup({
  sources,
  onCitationClick,
  onViewInDocument,
}: CitationGroupProps) {
  // Group sources by document
  const groupedSources = useMemo(() => {
    const groups = new Map<string, Array<Source & { citationNumber: number }>>();

    sources.forEach((source) => {
      const key = source.documentId || source.filename;
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(source);
    });

    return Array.from(groups.entries()).map(([key, items]) => ({
      documentKey: key,
      filename: items[0].filename,
      collection: items[0].collection,
      documentId: items[0].documentId,
      citations: items.sort((a, b) => a.citationNumber - b.citationNumber),
      avgSimilarity:
        items.reduce((sum, s) => sum + s.similarity, 0) / items.length,
    }));
  }, [sources]);

  return (
    <div className="space-y-2">
      {groupedSources.map((group) => (
        <Collapsible key={group.documentKey}>
          <CollapsibleTrigger asChild>
            <div className="flex items-center justify-between p-2 rounded-lg border bg-card hover:bg-muted/50 cursor-pointer transition-colors">
              <div className="flex items-center gap-2 min-w-0">
                <FileText
                  className={cn(
                    "h-4 w-4 shrink-0",
                    getFileTypeColor(group.filename)
                  )}
                />
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{group.filename}</p>
                  {group.collection && (
                    <p className="text-xs text-muted-foreground">
                      {group.collection}
                    </p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Badge variant="secondary" className="text-xs">
                  {group.citations.length} citation
                  {group.citations.length > 1 ? "s" : ""}
                </Badge>
                <Badge
                  variant="outline"
                  className={cn("text-xs", getRelevanceTier(group.avgSimilarity).color)}
                >
                  {formatSimilarity(group.avgSimilarity)} avg
                </Badge>
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              </div>
            </div>
          </CollapsibleTrigger>

          <CollapsibleContent>
            <div className="ml-6 mt-1 space-y-1 border-l-2 pl-3">
              {group.citations.map((source) => (
                <div
                  key={source.citationNumber}
                  className="flex items-start gap-2 py-1.5"
                >
                  <button
                    type="button"
                    onClick={() => onCitationClick?.(source.citationNumber)}
                    className="flex items-center justify-center w-5 h-5 text-[10px] font-bold rounded-full bg-primary/20 text-primary hover:bg-primary/30 shrink-0"
                  >
                    {source.citationNumber}
                  </button>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {truncate(source.snippet, 150)}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      {source.pageNumber !== undefined && (
                        <span className="text-[10px] text-muted-foreground">
                          Page {source.pageNumber}
                        </span>
                      )}
                      <span
                        className={cn(
                          "text-[10px] font-medium",
                          getRelevanceTier(source.similarity).color
                        )}
                      >
                        {formatSimilarity(source.similarity)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}

              {group.documentId && onViewInDocument && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full h-7 text-xs mt-1"
                  onClick={() => onViewInDocument(group.documentId!, undefined)}
                >
                  <Eye className="h-3 w-3 mr-1" />
                  View Document
                </Button>
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CitationSidebar - Full citation panel (Sparkpage-style)
// ---------------------------------------------------------------------------

interface CitationSidebarProps {
  sources: Array<Source & { citationNumber: number }>;
  isOpen: boolean;
  onClose: () => void;
  onCitationClick?: (citationNumber: number) => void;
  onViewInDocument?: (documentId: string, pageNumber?: number) => void;
}

export function CitationSidebar({
  sources,
  isOpen,
  onClose,
  onCitationClick,
  onViewInDocument,
}: CitationSidebarProps) {
  const sortedSources = useMemo(
    () => [...sources].sort((a, b) => b.similarity - a.similarity),
    [sources]
  );

  const avgSimilarity = useMemo(() => {
    if (sources.length === 0) return 0;
    return sources.reduce((sum, s) => sum + s.similarity, 0) / sources.length;
  }, [sources]);

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-lg max-h-[80vh] flex flex-col p-0">
        <DialogHeader className="px-4 py-3 border-b">
          <DialogTitle className="flex items-center gap-2 text-base">
            <BookOpen className="h-5 w-5 text-primary" />
            Sources & Citations
            <Badge variant="secondary" className="ml-auto">
              {sources.length} sources
            </Badge>
          </DialogTitle>
        </DialogHeader>

        {/* Summary Stats */}
        <div className="px-4 py-2 border-b bg-muted/30">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-lg font-bold">{sources.length}</p>
              <p className="text-xs text-muted-foreground">Citations</p>
            </div>
            <div>
              <p className="text-lg font-bold">
                {new Set(sources.map((s) => s.documentId || s.filename)).size}
              </p>
              <p className="text-xs text-muted-foreground">Documents</p>
            </div>
            <div>
              <p
                className={cn(
                  "text-lg font-bold",
                  getRelevanceTier(avgSimilarity).color
                )}
              >
                {formatSimilarity(avgSimilarity)}
              </p>
              <p className="text-xs text-muted-foreground">Avg Relevance</p>
            </div>
          </div>
        </div>

        {/* Grouped Citations */}
        <ScrollArea className="flex-1 px-4 py-3">
          <CitationGroup
            sources={sortedSources}
            onCitationClick={onCitationClick}
            onViewInDocument={onViewInDocument}
          />
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

export default InlineCitation;
