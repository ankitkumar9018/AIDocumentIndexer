'use client';

import { useState, useEffect } from 'react';
import { Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getPreviewPageUrl } from '@/lib/api';

interface TextPreviewProps {
  jobId: string;
  format: string;
  className?: string;
}

export function TextPreview({ jobId, format, className }: TextPreviewProps) {
  const [content, setContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchContent = async () => {
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

        const text = await response.text();
        setContent(text);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load preview');
      } finally {
        setIsLoading(false);
      }
    };

    fetchContent();
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

  if (!content) {
    return (
      <div className={cn('flex items-center justify-center h-[500px] bg-muted rounded-lg', className)}>
        <p className="text-sm text-muted-foreground">No preview available</p>
      </div>
    );
  }

  // For HTML, render in iframe
  if (format === 'html') {
    return (
      <div className={cn('bg-white rounded-lg border shadow-sm overflow-hidden', className)}>
        <iframe
          srcDoc={content}
          className="w-full h-[600px] border-0"
          title="HTML Preview"
          sandbox="allow-same-origin"
        />
      </div>
    );
  }

  // For Markdown and plain text, render as preformatted text
  return (
    <div className={cn('bg-white rounded-lg border shadow-sm overflow-hidden', className)}>
      <div className="p-4 max-h-[600px] overflow-auto">
        <pre className="text-sm font-mono whitespace-pre-wrap break-words">
          {content}
        </pre>
      </div>
    </div>
  );
}
