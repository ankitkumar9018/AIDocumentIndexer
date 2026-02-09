"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Search, FileText } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { DocumentFilterPanel } from "@/components/chat/document-filter-panel";
import { api } from "@/lib/api/client";

interface PreviewDoc {
  title: string;
  snippet: string;
  score: number;
  doc_id: string;
}

interface DocumentPickerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  collections: Array<{ name: string; document_count: number }>;
  selectedCollections: string[];
  onCollectionsChange: (collections: string[]) => void;
  boardName: string;
  keywords: string[];
  onConfirm: (docs: PreviewDoc[]) => void;
}

export function DocumentPickerDialog({
  open,
  onOpenChange,
  collections,
  selectedCollections,
  onCollectionsChange,
  boardName,
  keywords,
  onConfirm,
}: DocumentPickerDialogProps) {
  const [previewDocs, setPreviewDocs] = useState<PreviewDoc[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handlePreview = useCallback(async () => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams();
      if (boardName) params.append("name", boardName);
      keywords.forEach((k) => params.append("keywords", k));
      if (selectedCollections.length > 0) {
        selectedCollections.forEach((c) => params.append("collection_filters", c));
      }
      const resp = await api.post(`/moodboard/document-preview?${params.toString()}`);
      setPreviewDocs((resp.data as { documents: PreviewDoc[] }).documents || []);
    } catch {
      setPreviewDocs([]);
    }
    setIsLoading(false);
  }, [boardName, keywords, selectedCollections]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Select Document Sources</DialogTitle>
          <p className="text-sm text-muted-foreground">
            Choose which document collections to use as inspiration for your moodboard
          </p>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-4 flex-1 overflow-hidden">
          {/* Left: Collection filter */}
          <div className="overflow-y-auto">
            <DocumentFilterPanel
              collections={collections}
              selectedCollections={selectedCollections}
              onCollectionsChange={onCollectionsChange}
            />
          </div>

          {/* Right: Preview */}
          <div className="overflow-y-auto space-y-2">
            <Button
              onClick={handlePreview}
              disabled={isLoading}
              size="sm"
              className="w-full"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Search className="h-4 w-4 mr-2" />
              )}
              Preview Matches
            </Button>

            {previewDocs.length === 0 && !isLoading && (
              <div className="text-center py-8 text-muted-foreground">
                <FileText className="h-8 w-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">Click preview to see matching documents</p>
              </div>
            )}

            {previewDocs.map((doc, i) => (
              <div key={i} className="p-2.5 rounded-lg border bg-card text-xs space-y-1">
                <p className="font-medium text-foreground truncate">{doc.title}</p>
                <p className="text-muted-foreground line-clamp-2 leading-relaxed">{doc.snippet}</p>
                <Badge variant="outline" className="text-[10px]">
                  Relevance: {(doc.score * 100).toFixed(0)}%
                </Badge>
              </div>
            ))}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={() => { onConfirm(previewDocs); onOpenChange(false); }}>
            Use {selectedCollections.length > 0 ? `${selectedCollections.length} collections` : "All Documents"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
