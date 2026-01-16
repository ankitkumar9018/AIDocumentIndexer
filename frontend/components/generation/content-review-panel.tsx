'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import {
  CheckCircle2,
  Circle,
  Edit3,
  Trash2,
  RefreshCw,
  ChevronRight,
  ChevronLeft,
  CheckCheck,
  FileText,
  Presentation,
  Table,
  Loader2,
  AlertCircle,
  Sparkles,
  Minimize2,
  Maximize2,
} from 'lucide-react';
import { toast } from 'sonner';
import { api } from '@/lib/api';
import type {
  ContentReviewSession,
  ContentReviewItem,
  SlideContentReview,
  DocumentSectionReview,
  SheetContentReview,
  ContentReviewAction,
} from '@/lib/api';
import { SlideEditor } from './slide-editor';
import { SectionEditor } from './section-editor';

interface ContentReviewPanelProps {
  sessionId: string;
  contentType: 'pptx' | 'docx' | 'xlsx';
  onComplete: (content: Record<string, unknown>) => void;
  onCancel: () => void;
}

export function ContentReviewPanel({
  sessionId,
  contentType,
  onComplete,
  onCancel,
}: ContentReviewPanelProps) {
  const [session, setSession] = useState<ContentReviewSession | null>(null);
  const [items, setItems] = useState<ContentReviewItem[]>([]);
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const [selectedItemDetail, setSelectedItemDetail] = useState<
    SlideContentReview | DocumentSectionReview | SheetContentReview | null
  >(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isApproving, setIsApproving] = useState(false);
  const [isApproveAll, setIsApproveAll] = useState(false);

  // Fetch session status and items
  const fetchData = useCallback(async () => {
    try {
      const [statusResponse, itemsResponse] = await Promise.all([
        api.getContentReviewStatus(sessionId),
        api.listContentReviewItems(sessionId),
      ]);
      setSession(statusResponse);
      setItems(itemsResponse.items);

      // Auto-select first item if none selected
      if (!selectedItemId && itemsResponse.items.length > 0) {
        setSelectedItemId(itemsResponse.items[0].item_id);
      }
    } catch (error) {
      console.error('Failed to fetch review data:', error);
      toast.error('Failed to load review session');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, selectedItemId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Fetch item detail when selection changes
  useEffect(() => {
    if (selectedItemId) {
      api.getContentReviewItemDetail(sessionId, selectedItemId)
        .then((response) => {
          // Handle wrapped response format: { type, data, constraints, validation }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const rawResponse = response as any;
          if (rawResponse && rawResponse.data) {
            // Backend returns wrapped format, extract the data
            setSelectedItemDetail(rawResponse.data);
          } else {
            // Direct format (already the item itself)
            setSelectedItemDetail(response);
          }
        })
        .catch((error) => {
          console.error('Failed to fetch item detail:', error);
          toast.error('Failed to load item details');
        });
    }
  }, [sessionId, selectedItemId]);

  // Handle approve single item
  const handleApprove = async (itemId: string) => {
    setIsApproving(true);
    try {
      await api.approveContentReviewItem(sessionId, itemId);
      await fetchData();
      toast.success('Item approved');

      // Move to next item if available
      const currentIndex = items.findIndex(i => i.item_id === itemId);
      if (currentIndex < items.length - 1) {
        setSelectedItemId(items[currentIndex + 1].item_id);
      }
    } catch (error) {
      console.error('Failed to approve item:', error);
      toast.error('Failed to approve item');
    } finally {
      setIsApproving(false);
    }
  };

  // Handle approve all
  const handleApproveAll = async () => {
    setIsApproveAll(true);
    try {
      await api.approveAllContentReviewItems(sessionId);
      await fetchData();
      toast.success('All items approved');
    } catch (error) {
      console.error('Failed to approve all:', error);
      toast.error('Failed to approve all items');
    } finally {
      setIsApproveAll(false);
    }
  };

  // Handle edit action
  const handleEditAction = async (itemId: string, action: ContentReviewAction, feedback?: string) => {
    try {
      await api.editContentReviewItem(sessionId, itemId, {
        item_id: itemId,
        action,
        feedback,
      });
      await fetchData();

      // Refresh item detail
      if (selectedItemId === itemId) {
        const detail = await api.getContentReviewItemDetail(sessionId, itemId);
        setSelectedItemDetail(detail);
      }

      toast.success(`Action "${action}" completed`);
    } catch (error) {
      console.error('Failed to perform action:', error);
      toast.error(`Failed to ${action}`);
    }
  };

  // Handle slide update
  const handleSlideUpdate = async (updates: Partial<SlideContentReview>) => {
    if (!selectedItemId) return;

    try {
      const result = await api.updateSlide(sessionId, selectedItemId, updates);
      setSelectedItemDetail(result.updated_slide);
      await fetchData();
      toast.success('Slide updated');
    } catch (error) {
      console.error('Failed to update slide:', error);
      toast.error('Failed to update slide');
    }
  };

  // Handle complete - get approved content and call onComplete
  const handleComplete = async () => {
    if (!session?.can_render) {
      toast.error('Please approve all items before completing');
      return;
    }

    try {
      const result = await api.getApprovedContent(sessionId);
      onComplete(result.content);
    } catch (error) {
      console.error('Failed to get approved content:', error);
      toast.error('Failed to complete review');
    }
  };

  // Navigate between items
  const handlePrevious = () => {
    const currentIndex = items.findIndex(i => i.item_id === selectedItemId);
    if (currentIndex > 0) {
      setSelectedItemId(items[currentIndex - 1].item_id);
    }
  };

  const handleNext = () => {
    const currentIndex = items.findIndex(i => i.item_id === selectedItemId);
    if (currentIndex < items.length - 1) {
      setSelectedItemId(items[currentIndex + 1].item_id);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'approved':
      case 'final':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'rejected':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'deleted':
        return <Trash2 className="h-4 w-4 text-muted-foreground" />;
      case 'edited':
      case 'regenerating':
        return <RefreshCw className="h-4 w-4 text-blue-500" />;
      case 'draft':
      case 'pending':
      case 'pending_review':
      default:
        return <Circle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getContentTypeIcon = () => {
    switch (contentType) {
      case 'pptx':
        return <Presentation className="h-5 w-5" />;
      case 'docx':
        return <FileText className="h-5 w-5" />;
      case 'xlsx':
        return <Table className="h-5 w-5" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const currentIndex = items.findIndex(i => i.item_id === selectedItemId);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-3">
          {getContentTypeIcon()}
          <div>
            <h2 className="text-lg font-semibold">Content Review</h2>
            <p className="text-sm text-muted-foreground">
              Review and edit content before generating the document
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            variant="outline"
            onClick={handleApproveAll}
            disabled={isApproveAll || session?.can_render}
          >
            {isApproveAll ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <CheckCheck className="h-4 w-4 mr-2" />
            )}
            Approve All
          </Button>
          <Button
            onClick={handleComplete}
            disabled={!session?.can_render}
          >
            Complete Review
          </Button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="p-4 border-b bg-muted/30">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">
            Progress: {session?.approved_items || 0} / {session?.total_items || 0} approved
          </span>
          <span className="text-sm text-muted-foreground">
            {Math.round(session?.progress_percent || 0)}%
          </span>
        </div>
        <Progress value={session?.progress_percent || 0} className="h-2" />
      </div>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Item List Sidebar */}
        <div className="w-72 border-r flex flex-col">
          <div className="p-3 border-b">
            <h3 className="text-sm font-medium">
              {contentType === 'pptx' ? 'Slides' : contentType === 'docx' ? 'Sections' : 'Sheets'}
            </h3>
          </div>
          <ScrollArea className="flex-1">
            <div className="p-2 space-y-1">
              {items.map((item, index) => (
                <button
                  key={item.item_id || `item-${index}`}
                  onClick={() => setSelectedItemId(item.item_id)}
                  className={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-colors ${
                    selectedItemId === item.item_id
                      ? 'bg-primary/10 border border-primary/20'
                      : 'hover:bg-muted'
                  }`}
                >
                  {getStatusIcon(item.status)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-muted-foreground">
                        #{index + 1}
                      </span>
                      <span className="text-sm font-medium truncate">
                        {item.title}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground truncate mt-0.5">
                      {item.preview_text}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Item Detail / Editor */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Navigation */}
          <div className="flex items-center justify-between p-3 border-b">
            <Button
              variant="ghost"
              size="sm"
              onClick={handlePrevious}
              disabled={currentIndex <= 0}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </Button>
            <span className="text-sm text-muted-foreground">
              {currentIndex + 1} of {items.length}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleNext}
              disabled={currentIndex >= items.length - 1}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>

          {/* Editor Area */}
          <ScrollArea className="flex-1 p-4">
            {selectedItemDetail && contentType === 'pptx' && (
              <SlideEditor
                slide={selectedItemDetail as SlideContentReview}
                onUpdate={handleSlideUpdate}
                onApprove={() => handleApprove(selectedItemId!)}
                onAction={(action, feedback) => handleEditAction(selectedItemId!, action, feedback)}
                isApproving={isApproving}
              />
            )}
            {selectedItemDetail && contentType === 'docx' && (
              <SectionEditor
                section={selectedItemDetail as DocumentSectionReview}
                onApprove={() => handleApprove(selectedItemId!)}
                onAction={(action, feedback) => handleEditAction(selectedItemId!, action, feedback)}
                isApproving={isApproving}
              />
            )}
            {selectedItemDetail && contentType === 'xlsx' && (
              <Card>
                <CardHeader>
                  <CardTitle>Sheet: {(selectedItemDetail as SheetContentReview).sheet_name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Spreadsheet editing coming soon...
                  </p>
                </CardContent>
              </Card>
            )}
          </ScrollArea>

          {/* Action Bar */}
          <div className="p-4 border-t bg-muted/30">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleEditAction(selectedItemId!, 'enhance')}
                  disabled={!selectedItemId}
                >
                  <Sparkles className="h-4 w-4 mr-1" />
                  Enhance
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleEditAction(selectedItemId!, 'shorten')}
                  disabled={!selectedItemId}
                >
                  <Minimize2 className="h-4 w-4 mr-1" />
                  Shorten
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleEditAction(selectedItemId!, 'expand')}
                  disabled={!selectedItemId}
                >
                  <Maximize2 className="h-4 w-4 mr-1" />
                  Expand
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleEditAction(selectedItemId!, 'regenerate')}
                  disabled={!selectedItemId}
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Regenerate
                </Button>
              </div>
              <Button
                onClick={() => handleApprove(selectedItemId!)}
                disabled={!selectedItemId || isApproving}
              >
                {isApproving ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                )}
                Approve & Next
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
