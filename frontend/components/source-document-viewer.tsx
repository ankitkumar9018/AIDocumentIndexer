"use client";

import { useState, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  FileText,
  Copy,
  Check,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  Download,
} from "lucide-react";
import { api } from "@/lib/api/client";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface SourceChunk {
  documentId: string;
  filename: string;
  pageNumber?: number;
  snippet: string;
  fullContent?: string;
  similarity: number;
  collection?: string;
  chunkId?: string;
}

interface SourceDocumentViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sources: SourceChunk[];
  currentIndex?: number;
  query?: string;
  onOpenDocument?: (documentId: string) => void;
}

/**
 * Highlights matching words/phrases in text based on query terms.
 */
function highlightText(text: string, query?: string): React.ReactNode[] {
  if (!query || !text) {
    return [text];
  }

  // Split query into words, filter out very common stop words only
  // Keep domain-specific terms that might be relevant
  const stopWords = new Set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need",
    "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "and", "but", "if", "or", "because", "until", "while",
    "about", "what", "which", "who", "this", "that", "these", "those",
  ]);

  const queryWords = query
    .toLowerCase()
    .split(/\s+/)
    // Allow words with 2+ characters (was > 2, now >= 2)
    .filter(word => word.length >= 2 && !stopWords.has(word))
    .map(word => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")); // Escape regex special chars

  if (queryWords.length === 0) {
    return [text];
  }

  // Create regex pattern with word boundaries for better matching
  // Use \b for word boundary to avoid partial matches within words
  const pattern = new RegExp(`\\b(${queryWords.join("|")})\\b`, "gi");

  const parts = text.split(pattern);
  const result: React.ReactNode[] = [];

  parts.forEach((part, index) => {
    // Check if this part matches any query word (case-insensitive)
    const isMatch = queryWords.some(word =>
      part.toLowerCase() === word.toLowerCase()
    );

    if (isMatch) {
      result.push(
        <mark
          key={index}
          className="bg-yellow-200 dark:bg-yellow-800/60 text-inherit px-0.5 rounded font-medium"
        >
          {part}
        </mark>
      );
    } else {
      result.push(part);
    }
  });

  return result;
}

export function SourceDocumentViewer({
  open,
  onOpenChange,
  sources,
  currentIndex = 0,
  query,
  onOpenDocument,
}: SourceDocumentViewerProps) {
  const [activeIndex, setActiveIndex] = useState(currentIndex);
  const [copied, setCopied] = useState(false);

  const currentSource = sources[activeIndex];

  // Reset active index when sources change
  useMemo(() => {
    if (currentIndex >= 0 && currentIndex < sources.length) {
      setActiveIndex(currentIndex);
    }
  }, [currentIndex, sources.length]);

  if (!currentSource) {
    return null;
  }

  const handleCopy = async () => {
    const content = currentSource.fullContent || currentSource.snippet;
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = async () => {
    try {
      await api.downloadDocument(currentSource.documentId, currentSource.filename);
      toast.success("Download started", {
        description: `Downloading "${currentSource.filename}"...`,
      });
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      toast.error("Download failed", {
        description: errorMessage,
      });
    }
  };

  const handlePrevious = () => {
    if (activeIndex > 0) {
      setActiveIndex(activeIndex - 1);
    }
  };

  const handleNext = () => {
    if (activeIndex < sources.length - 1) {
      setActiveIndex(activeIndex + 1);
    }
  };

  const displayContent = currentSource.fullContent || currentSource.snippet;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col">
        <DialogHeader className="flex-shrink-0">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-2 min-w-0">
              <FileText className="h-5 w-5 text-primary shrink-0" />
              <DialogTitle className="truncate">
                {currentSource.filename}
              </DialogTitle>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              {onOpenDocument && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onOpenDocument(currentSource.documentId)}
                >
                  <ExternalLink className="h-4 w-4 mr-1" />
                  View Full
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
              >
                <Download className="h-4 w-4 mr-1" />
                Download
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopy}
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4 mr-1" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4 mr-1" />
                    Copy
                  </>
                )}
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap mt-2">
            {currentSource.pageNumber && (
              <Badge variant="secondary">
                Page {currentSource.pageNumber}
              </Badge>
            )}
            {currentSource.collection && (
              <Badge variant="outline">
                {currentSource.collection}
              </Badge>
            )}
            <Badge
              variant="secondary"
              className={cn(
                "gap-1",
                currentSource.similarity >= 0.8
                  ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                  : currentSource.similarity >= 0.5
                  ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300"
                  : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
              )}
            >
              {Math.round(currentSource.similarity * 100)}% match
            </Badge>
          </div>
        </DialogHeader>

        {/* Content area with highlighted text */}
        <ScrollArea className="flex-1 min-h-0 mt-4">
          <div className="prose prose-sm dark:prose-invert max-w-none p-4 bg-muted/30 rounded-lg border">
            <p className="whitespace-pre-wrap leading-relaxed text-foreground">
              {highlightText(displayContent, query)}
            </p>
          </div>
        </ScrollArea>

        {/* Navigation footer */}
        {sources.length > 1 && (
          <div className="flex items-center justify-between pt-4 border-t mt-4 flex-shrink-0">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePrevious}
              disabled={activeIndex === 0}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </Button>

            <div className="flex items-center gap-2">
              {sources.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setActiveIndex(index)}
                  className={cn(
                    "w-2 h-2 rounded-full transition-colors",
                    index === activeIndex
                      ? "bg-primary"
                      : "bg-muted-foreground/30 hover:bg-muted-foreground/50"
                  )}
                  aria-label={`Go to source ${index + 1}`}
                />
              ))}
              <span className="text-sm text-muted-foreground ml-2">
                {activeIndex + 1} of {sources.length}
              </span>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={handleNext}
              disabled={activeIndex === sources.length - 1}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        )}

        {/* Query context */}
        {query && (
          <div className="pt-4 border-t mt-4 flex-shrink-0">
            <p className="text-xs text-muted-foreground">
              <span className="font-medium">Query:</span> {query}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Highlighted words are terms from your query that appear in this source.
            </p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default SourceDocumentViewer;
