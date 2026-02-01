'use client';

import { useState } from 'react';
import { Loader2, AlertTriangle, Sparkles, Image as ImageIcon, Settings } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import Link from 'next/link';
import { useVisionStatus, useReanalyzeDocumentImages } from '@/lib/api';
import type { ImageAnalysisStats } from '@/lib/api';

interface ReanalyzeImagesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  document: {
    id: string;
    name: string;
    images_extracted_count?: number;
    images_analyzed_count?: number;
    image_analysis_status?: string;
  };
  onSuccess?: () => void;
}

export function ReanalyzeImagesModal({
  open,
  onOpenChange,
  document,
  onSuccess,
}: ReanalyzeImagesModalProps) {
  const [forceReanalyze, setForceReanalyze] = useState(false);
  const [skipDuplicates, setSkipDuplicates] = useState(true);
  const [analysisResult, setAnalysisResult] = useState<{
    success: boolean;
    stats?: ImageAnalysisStats;
    error?: string | null;
  } | null>(null);

  const { data: visionStatus, isLoading: isLoadingVision } = useVisionStatus();
  const reanalyzeMutation = useReanalyzeDocumentImages();

  const imagesExtracted = document.images_extracted_count ?? 0;
  const imagesAnalyzed = document.images_analyzed_count ?? 0;
  const pendingImages = imagesExtracted - imagesAnalyzed;

  const handleAnalyze = async () => {
    setAnalysisResult(null);

    try {
      const result = await reanalyzeMutation.mutateAsync({
        documentId: document.id,
        options: {
          force_reanalyze: forceReanalyze,
          skip_duplicates: skipDuplicates,
        },
      });

      setAnalysisResult({
        success: result.success,
        stats: result.stats,
        error: result.error,
      });

      if (result.success && onSuccess) {
        onSuccess();
      }
    } catch (error: any) {
      setAnalysisResult({
        success: false,
        error: error?.detail || 'Analysis failed',
      });
    }
  };

  const handleClose = () => {
    setAnalysisResult(null);
    onOpenChange(false);
  };

  const isAnalyzing = reanalyzeMutation.isPending;

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ImageIcon className="h-5 w-5" />
            Analyze Document Images
          </DialogTitle>
          <DialogDescription>
            Use AI vision to analyze and describe images in this document for better search.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Vision Model Warning */}
          {!isLoadingVision && !visionStatus?.available && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Vision Model Not Configured</AlertTitle>
              <AlertDescription className="space-y-2">
                <p>{visionStatus?.recommendation || 'No vision model is available for image analysis.'}</p>
                <Button variant="link" size="sm" className="p-0 h-auto" asChild>
                  <Link href="/dashboard/admin/settings">
                    <Settings className="h-3 w-3 mr-1" />
                    Configure in Settings
                  </Link>
                </Button>
              </AlertDescription>
            </Alert>
          )}

          {/* Vision Model Info */}
          {visionStatus?.available && (
            <Alert>
              <Sparkles className="h-4 w-4" />
              <AlertTitle>Vision Model Ready</AlertTitle>
              <AlertDescription>
                Using {visionStatus.provider} ({visionStatus.model})
              </AlertDescription>
            </Alert>
          )}

          {/* Document Stats */}
          <div className="grid grid-cols-2 gap-4 p-4 bg-muted rounded-lg">
            <div>
              <Label className="text-muted-foreground text-xs">Images Found</Label>
              <p className="text-2xl font-bold">{imagesExtracted}</p>
            </div>
            <div>
              <Label className="text-muted-foreground text-xs">Already Analyzed</Label>
              <p className="text-2xl font-bold">{imagesAnalyzed}</p>
            </div>
            {pendingImages > 0 && (
              <div className="col-span-2">
                <Label className="text-muted-foreground text-xs">Pending Analysis</Label>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">{pendingImages} images</Badge>
                </div>
              </div>
            )}
          </div>

          {/* Options */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="force">Force re-analyze all</Label>
                <p className="text-xs text-muted-foreground">
                  Re-analyze even images already processed
                </p>
              </div>
              <Switch
                id="force"
                checked={forceReanalyze}
                onCheckedChange={setForceReanalyze}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="dedup">Skip duplicates</Label>
                <p className="text-xs text-muted-foreground">
                  Use cached results for identical images
                </p>
              </div>
              <Switch
                id="dedup"
                checked={skipDuplicates}
                onCheckedChange={setSkipDuplicates}
              />
            </div>
          </div>

          {/* Analysis Result */}
          {analysisResult && (
            <div className="space-y-2">
              {analysisResult.success ? (
                <Alert>
                  <Sparkles className="h-4 w-4" />
                  <AlertTitle>Analysis Complete</AlertTitle>
                  <AlertDescription>
                    <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                      <div>Newly analyzed: {analysisResult.stats?.newly_analyzed ?? 0}</div>
                      <div>From cache: {analysisResult.stats?.cached_used ?? 0}</div>
                      <div>Skipped (small): {analysisResult.stats?.images_skipped_small ?? 0}</div>
                      <div>Failed: {analysisResult.stats?.images_failed ?? 0}</div>
                    </div>
                    {analysisResult.stats?.total_time_ms && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Completed in {(analysisResult.stats.total_time_ms / 1000).toFixed(1)}s
                      </p>
                    )}
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Analysis Failed</AlertTitle>
                  <AlertDescription>{analysisResult.error}</AlertDescription>
                </Alert>
              )}
            </div>
          )}

          {/* Progress */}
          {isAnalyzing && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Analyzing images...</span>
              </div>
              <Progress value={undefined} className="w-full" />
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isAnalyzing}>
            {analysisResult?.success ? 'Close' : 'Cancel'}
          </Button>
          {!analysisResult?.success && (
            <Button
              onClick={handleAnalyze}
              disabled={!visionStatus?.available || isAnalyzing || imagesExtracted === 0}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Analyze Images
                </>
              )}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
