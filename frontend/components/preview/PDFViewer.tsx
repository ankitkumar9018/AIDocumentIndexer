'use client';

import { useState } from 'react';
import { ChevronLeft, ChevronRight, Loader2, AlertCircle, ZoomIn, ZoomOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { usePreviewMetadata, getPreviewPageUrl } from '@/lib/api';

interface PDFViewerProps {
  jobId: string;
  className?: string;
}

export function PDFViewer({ jobId, className }: PDFViewerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [zoom, setZoom] = useState(100);
  const [imageLoading, setImageLoading] = useState(true);
  const [imageError, setImageError] = useState(false);

  const { data: metadata, isLoading, error } = usePreviewMetadata(jobId);

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center h-[600px] bg-muted rounded-lg', className)}>
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !metadata?.supported) {
    return (
      <div className={cn('flex flex-col items-center justify-center h-[600px] bg-muted rounded-lg gap-2', className)}>
        <AlertCircle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-muted-foreground">
          {metadata?.error || 'PDF preview not available'}
        </p>
      </div>
    );
  }

  const totalPages = metadata.page_count || 1;
  const pageUrl = getPreviewPageUrl(jobId, currentPage);

  const goToPrevious = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
      setImageLoading(true);
      setImageError(false);
    }
  };

  const goToNext = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
      setImageLoading(true);
      setImageError(false);
    }
  };

  const zoomIn = () => setZoom((prev) => Math.min(prev + 25, 200));
  const zoomOut = () => setZoom((prev) => Math.max(prev - 25, 50));

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      {/* Toolbar */}
      <div className="flex items-center justify-between bg-muted/50 rounded-lg px-4 py-2">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={goToPrevious}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm">
            Page {currentPage} of {totalPages}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={goToNext}
            disabled={currentPage === totalPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={zoomOut} disabled={zoom <= 50}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-sm w-12 text-center">{zoom}%</span>
          <Button variant="ghost" size="sm" onClick={zoomIn} disabled={zoom >= 200}>
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* PDF Page */}
      <div className="relative bg-gray-200 rounded-lg overflow-auto h-[600px] flex justify-center">
        {imageLoading && !imageError && (
          <div className="absolute inset-0 flex items-center justify-center bg-muted/50">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}
        {imageError ? (
          <div className="flex flex-col items-center justify-center h-full gap-2">
            <AlertCircle className="h-8 w-8 text-destructive" />
            <p className="text-sm text-muted-foreground">Failed to load page</p>
          </div>
        ) : (
          <img
            src={pageUrl}
            alt={`Page ${currentPage}`}
            style={{ width: `${zoom}%`, height: 'auto' }}
            className="shadow-lg"
            onLoad={() => setImageLoading(false)}
            onError={() => {
              setImageLoading(false);
              setImageError(true);
            }}
          />
        )}
      </div>
    </div>
  );
}
