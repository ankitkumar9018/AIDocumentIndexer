"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  FileText,
  Link as LinkIcon,
  Image as ImageIcon,
  Info,
  Copy,
  Check,
  ExternalLink,
  Clock,
  Hash,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface ScrapedPage {
  url: string;
  title?: string;
  content?: string;
  word_count?: number;
  scraped_at?: string;
  links?: string[];
  images?: string[];
  metadata?: Record<string, unknown>;
}

interface ScrapedContentModalProps {
  page: ScrapedPage | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ScrapedContentModal({
  page,
  open,
  onOpenChange,
}: ScrapedContentModalProps) {
  const [copied, setCopied] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState("content");

  const handleCopyContent = async () => {
    if (!page?.content) return;

    try {
      await navigator.clipboard.writeText(page.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // Reset tab when modal opens with new page
  React.useEffect(() => {
    if (open) {
      setActiveTab("content");
    }
  }, [open, page]);

  if (!page) return null;

  const hasLinks = page.links && page.links.length > 0;
  const hasImages = page.images && page.images.length > 0;
  const hasMetadata = page.metadata && Object.keys(page.metadata).length > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 pr-8">
            <FileText className="h-5 w-5 text-primary" />
            <span className="truncate">{page.title || "Scraped Content"}</span>
          </DialogTitle>
          <DialogDescription asChild>
            <div className="flex items-center gap-2 text-sm">
              <a
                href={page.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline flex items-center gap-1 truncate max-w-md"
              >
                {page.url}
                <ExternalLink className="h-3 w-3 flex-shrink-0" />
              </a>
            </div>
          </DialogDescription>
        </DialogHeader>

        {/* Stats Bar */}
        <div className="flex flex-wrap gap-2 py-2 border-b">
          {page.word_count && (
            <Badge variant="secondary" className="gap-1">
              <Hash className="h-3 w-3" />
              {page.word_count.toLocaleString()} words
            </Badge>
          )}
          {page.scraped_at && (
            <Badge variant="outline" className="gap-1">
              <Clock className="h-3 w-3" />
              {new Date(page.scraped_at).toLocaleString()}
            </Badge>
          )}
          {hasLinks && (
            <Badge variant="outline" className="gap-1">
              <LinkIcon className="h-3 w-3" />
              {page.links!.length} links
            </Badge>
          )}
          {hasImages && (
            <Badge variant="outline" className="gap-1">
              <ImageIcon className="h-3 w-3" />
              {page.images!.length} images
            </Badge>
          )}
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="content" className="gap-1">
              <FileText className="h-4 w-4" />
              Content
            </TabsTrigger>
            <TabsTrigger value="links" disabled={!hasLinks} className="gap-1">
              <LinkIcon className="h-4 w-4" />
              Links
            </TabsTrigger>
            <TabsTrigger value="images" disabled={!hasImages} className="gap-1">
              <ImageIcon className="h-4 w-4" />
              Images
            </TabsTrigger>
            <TabsTrigger value="metadata" disabled={!hasMetadata} className="gap-1">
              <Info className="h-4 w-4" />
              Metadata
            </TabsTrigger>
          </TabsList>

          {/* Content Tab */}
          <TabsContent value="content" className="mt-4">
            <div className="flex justify-end mb-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopyContent}
                disabled={!page.content}
                className="gap-1"
              >
                {copied ? (
                  <>
                    <Check className="h-3 w-3" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    Copy Content
                  </>
                )}
              </Button>
            </div>
            <ScrollArea className="h-[400px] rounded-md border p-4 bg-muted/30">
              {page.content ? (
                <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed">
                  {page.content}
                </pre>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  No content available
                </p>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Links Tab */}
          <TabsContent value="links" className="mt-4">
            <ScrollArea className="h-[400px] rounded-md border p-4">
              {hasLinks ? (
                <div className="space-y-2">
                  {page.links!.map((link, i) => (
                    <a
                      key={i}
                      href={link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-sm text-primary hover:underline hover:bg-muted/50 p-2 rounded-md transition-colors"
                    >
                      <ExternalLink className="h-3 w-3 flex-shrink-0" />
                      <span className="truncate">{link}</span>
                    </a>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  No links found
                </p>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Images Tab */}
          <TabsContent value="images" className="mt-4">
            <ScrollArea className="h-[400px] rounded-md border p-4">
              {hasImages ? (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {page.images!.map((src, i) => (
                    <a
                      key={i}
                      href={src}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block group relative overflow-hidden rounded-lg border bg-muted hover:ring-2 ring-primary transition-all"
                    >
                      <img
                        src={src}
                        alt={`Image ${i + 1}`}
                        className="w-full h-32 object-cover"
                        onError={(e) => {
                          (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ccc' width='100' height='100'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23666' font-size='12'%3ENo Preview%3C/text%3E%3C/svg%3E";
                        }}
                      />
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <ExternalLink className="h-6 w-6 text-white" />
                      </div>
                    </a>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  No images found
                </p>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Metadata Tab */}
          <TabsContent value="metadata" className="mt-4">
            <ScrollArea className="h-[400px] rounded-md border p-4">
              {hasMetadata ? (
                <div className="space-y-4">
                  {Object.entries(page.metadata!).map(([key, value]) => (
                    <div key={key} className="border-b pb-3 last:border-0">
                      <dt className="text-sm font-medium text-muted-foreground mb-1">
                        {key}
                      </dt>
                      <dd className="text-sm font-mono break-all">
                        {typeof value === "object"
                          ? JSON.stringify(value, null, 2)
                          : String(value)}
                      </dd>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  No metadata available
                </p>
              )}
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}

export default ScrapedContentModal;
