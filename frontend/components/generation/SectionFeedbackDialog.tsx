'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Star, RefreshCw, ThumbsUp, ThumbsDown, Loader2 } from 'lucide-react';
import { useProvideSectionFeedback, useReviseSection } from '@/lib/api/hooks';
import { toast } from 'sonner';

interface SectionFeedbackDialogProps {
  jobId: string;
  sectionId: string;
  sectionTitle: string;
  sectionContent: string;
  qualityScore?: number;
  qualitySummary?: string;
  isOpen: boolean;
  onClose: () => void;
  onFeedbackSubmitted?: () => void;
}

export function SectionFeedbackDialog({
  jobId,
  sectionId,
  sectionTitle,
  sectionContent,
  qualityScore,
  qualitySummary,
  isOpen,
  onClose,
  onFeedbackSubmitted,
}: SectionFeedbackDialogProps) {
  const [feedback, setFeedback] = useState('');
  const [rating, setRating] = useState<number>(0);

  const feedbackMutation = useProvideSectionFeedback();
  const reviseMutation = useReviseSection();

  const handleApprove = async () => {
    try {
      await feedbackMutation.mutateAsync({
        jobId,
        sectionId,
        feedback: { approved: true, feedback: feedback || undefined },
      });
      toast.success('Section approved');
      onClose();
      onFeedbackSubmitted?.();
    } catch (error) {
      toast.error('Failed to approve section');
    }
  };

  const handleRequestRevision = async () => {
    if (!feedback.trim()) {
      toast.error('Please provide feedback for the revision');
      return;
    }

    try {
      await feedbackMutation.mutateAsync({
        jobId,
        sectionId,
        feedback: { approved: false, feedback },
      });
      toast.success('Revision requested');
      onClose();
      onFeedbackSubmitted?.();
    } catch (error) {
      toast.error('Failed to request revision');
    }
  };

  const handleRegenerate = async () => {
    try {
      await feedbackMutation.mutateAsync({
        jobId,
        sectionId,
        feedback: { approved: false, feedback: feedback || 'Please regenerate this section with improvements.' },
      });
      await reviseMutation.mutateAsync({ jobId, sectionId });
      toast.success('Section regenerated');
      onClose();
      onFeedbackSubmitted?.();
    } catch (error) {
      toast.error('Failed to regenerate section');
    }
  };

  const isLoading = feedbackMutation.isPending || reviseMutation.isPending;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Section Feedback: {sectionTitle}</DialogTitle>
          <DialogDescription>
            Review this section and provide feedback or approve it.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Quality Score Display */}
          {qualityScore !== undefined && (
            <div className="p-3 bg-muted rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Quality Score</span>
                <div className="flex items-center gap-2">
                  <div className="flex">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <Star
                        key={star}
                        className={`h-4 w-4 ${
                          star <= Math.round(qualityScore * 5)
                            ? 'fill-yellow-400 text-yellow-400'
                            : 'text-muted-foreground'
                        }`}
                      />
                    ))}
                  </div>
                  <span className="text-sm text-muted-foreground">
                    ({(qualityScore * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
              {qualitySummary && (
                <p className="text-xs text-muted-foreground mt-1">{qualitySummary}</p>
              )}
            </div>
          )}

          {/* Content Preview */}
          <div className="border rounded-lg p-4 max-h-[200px] overflow-y-auto bg-background">
            <h4 className="font-medium text-sm mb-2">Content Preview</h4>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">
              {sectionContent.length > 500
                ? sectionContent.slice(0, 500) + '...'
                : sectionContent}
            </p>
          </div>

          {/* Rating */}
          <div>
            <label className="text-sm font-medium">Rate this section</label>
            <div className="flex gap-1 mt-1">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setRating(star)}
                  className="p-1 hover:scale-110 transition-transform"
                >
                  <Star
                    className={`h-6 w-6 ${
                      star <= rating
                        ? 'fill-yellow-400 text-yellow-400'
                        : 'text-muted-foreground hover:text-yellow-300'
                    }`}
                  />
                </button>
              ))}
            </div>
          </div>

          {/* Feedback Input */}
          <div>
            <label className="text-sm font-medium">
              Feedback (required for revision)
            </label>
            <Textarea
              placeholder="What would you like to change? Be specific about improvements needed..."
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              className="mt-1"
              rows={4}
            />
          </div>
        </div>

        <DialogFooter className="flex-col sm:flex-row gap-2">
          <Button
            variant="outline"
            onClick={handleRegenerate}
            disabled={isLoading}
            className="w-full sm:w-auto"
          >
            {reviseMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Regenerate
          </Button>
          <Button
            variant="destructive"
            onClick={handleRequestRevision}
            disabled={isLoading || !feedback.trim()}
            className="w-full sm:w-auto"
          >
            <ThumbsDown className="h-4 w-4 mr-2" />
            Request Revision
          </Button>
          <Button
            onClick={handleApprove}
            disabled={isLoading}
            className="w-full sm:w-auto"
          >
            {feedbackMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <ThumbsUp className="h-4 w-4 mr-2" />
            )}
            Approve
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
