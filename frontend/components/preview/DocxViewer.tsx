'use client';

import { useState, useEffect } from 'react';
import { Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getPreviewPageUrl } from '@/lib/api';

interface DocxViewerProps {
  jobId: string;
  className?: string;
}

export function DocxViewer({ jobId, className }: DocxViewerProps) {
  const [htmlContent, setHtmlContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHtml = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const url = getPreviewPageUrl(jobId, 1);
        const response = await fetch(url, {
          credentials: 'include',
        });

        if (!response.ok) {
          throw new Error(`Failed to load preview: ${response.statusText}`);
        }

        const html = await response.text();
        setHtmlContent(html);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load preview');
      } finally {
        setIsLoading(false);
      }
    };

    fetchHtml();
  }, [jobId]);

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center h-[500px] bg-muted rounded-lg', className)}>
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('flex flex-col items-center justify-center h-[500px] bg-muted rounded-lg gap-2', className)}>
        <AlertCircle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-muted-foreground">{error}</p>
      </div>
    );
  }

  if (!htmlContent) {
    return (
      <div className={cn('flex items-center justify-center h-[500px] bg-muted rounded-lg', className)}>
        <p className="text-sm text-muted-foreground">No preview available</p>
      </div>
    );
  }

  return (
    <div className={cn('bg-white rounded-lg border shadow-sm overflow-hidden', className)}>
      <iframe
        srcDoc={htmlContent}
        className="w-full h-[600px] border-0"
        title="Document Preview"
        sandbox="allow-same-origin"
      />
    </div>
  );
}
