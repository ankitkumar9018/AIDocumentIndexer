'use client';

import { Loader2, AlertCircle, FileText, Presentation, File } from 'lucide-react';
import { cn } from '@/lib/utils';
import { usePreviewMetadata } from '@/lib/api';
import { SlideViewer } from './SlideViewer';
import { PDFViewer } from './PDFViewer';
import { DocxViewer } from './DocxViewer';
import { TextPreview } from './TextPreview';

interface DocumentPreviewProps {
  jobId: string;
  format: string;
  className?: string;
}

export function DocumentPreview({ jobId, format, className }: DocumentPreviewProps) {
  const { data: metadata, isLoading, error } = usePreviewMetadata(jobId);

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center h-[400px] bg-muted rounded-lg', className)}>
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('flex flex-col items-center justify-center h-[400px] bg-muted rounded-lg gap-2', className)}>
        <AlertCircle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-muted-foreground">Failed to load preview</p>
      </div>
    );
  }

  if (!metadata?.supported) {
    return (
      <div className={cn('flex flex-col items-center justify-center h-[400px] bg-muted rounded-lg gap-4', className)}>
        {format.toLowerCase() === 'pptx' && <Presentation className="h-16 w-16 text-muted-foreground" />}
        {format.toLowerCase() === 'docx' && <FileText className="h-16 w-16 text-muted-foreground" />}
        {format.toLowerCase() === 'pdf' && <File className="h-16 w-16 text-muted-foreground" />}
        {!['pptx', 'docx', 'pdf'].includes(format.toLowerCase()) && (
          <File className="h-16 w-16 text-muted-foreground" />
        )}
        <div className="text-center">
          <p className="text-sm font-medium">Preview not available</p>
          <p className="text-xs text-muted-foreground mt-1">
            {metadata?.error || 'Download the document to view it'}
          </p>
        </div>
      </div>
    );
  }

  const formatLower = format.toLowerCase();

  // Render appropriate viewer based on format
  if (formatLower === 'pptx') {
    return <SlideViewer jobId={jobId} className={className} />;
  }

  if (formatLower === 'pdf') {
    return <PDFViewer jobId={jobId} className={className} />;
  }

  if (formatLower === 'docx') {
    return <DocxViewer jobId={jobId} className={className} />;
  }

  if (['md', 'markdown', 'html', 'txt'].includes(formatLower)) {
    return <TextPreview jobId={jobId} format={formatLower} className={className} />;
  }

  // Fallback for unsupported formats
  return (
    <div className={cn('flex flex-col items-center justify-center h-[400px] bg-muted rounded-lg gap-2', className)}>
      <File className="h-16 w-16 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">
        Preview not available for {format.toUpperCase()} files
      </p>
    </div>
  );
}
