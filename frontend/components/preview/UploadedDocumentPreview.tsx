'use client';

import { useState, useEffect } from 'react';
import { Loader2, AlertCircle, FileText, Presentation, File, ExternalLink, Download, Image as ImageIcon, ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface UploadedDocumentPreviewProps {
  documentId: string;
  fileName: string;
  fileType: string;
  className?: string;
}

interface PreviewMetadata {
  supported: boolean;
  type?: 'pdf' | 'slides' | 'html' | 'image' | 'text' | 'iframe' | 'unsupported';
  page_count?: number;
  format?: string;
  error?: string;
}

/**
 * Preview component for uploaded documents.
 *
 * Supports:
 * - PDF: Inline via iframe
 * - Images: Inline display
 * - PPTX: Rendered slides with navigation
 * - DOCX: HTML rendering
 * - Text files: Raw content display
 */
export function UploadedDocumentPreview({
  documentId,
  fileName,
  fileType,
  className
}: UploadedDocumentPreviewProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<PreviewMetadata | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [slideImages, setSlideImages] = useState<string[]>([]);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
  const DEV_MODE = process.env.NEXT_PUBLIC_DEV_MODE === 'true';

  // Build URLs with auth token for direct browser access
  const addToken = (url: string) => {
    if (DEV_MODE) {
      const separator = url.includes('?') ? '&' : '?';
      return `${url}${separator}token=dev-token`;
    }
    return url;
  };

  const previewBaseUrl = `${API_BASE_URL}/documents/${documentId}/download?preview=true`;
  const downloadBaseUrl = `${API_BASE_URL}/documents/${documentId}/download`;
  const metadataUrl = `${API_BASE_URL}/documents/${documentId}/preview/metadata`;
  const pageUrl = (page: number) => `${API_BASE_URL}/documents/${documentId}/preview/page/${page}`;

  const viewUrl = addToken(previewBaseUrl);
  const downloadUrl = addToken(downloadBaseUrl);

  const formatLower = fileType?.toLowerCase() || fileName?.split('.').pop()?.toLowerCase() || '';

  // Check if format supports native browser preview (PDF, images)
  const supportsNativePreview = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(formatLower);
  const isImage = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(formatLower);
  const isPdf = formatLower === 'pdf';
  const isPptx = formatLower === 'pptx';
  const isDocx = formatLower === 'docx';

  useEffect(() => {
    async function fetchMetadata() {
      setIsLoading(true);
      setError(null);

      try {
        // For PPTX/DOCX, fetch preview metadata first
        if (isPptx || isDocx || !supportsNativePreview) {
          const response = await fetch(addToken(metadataUrl));
          if (response.ok) {
            const data = await response.json();
            setMetadata(data);

            // For PPTX, fetch all slide thumbnails
            if (data.type === 'slides' && data.supported) {
              const slidesUrl = `${API_BASE_URL}/documents/${documentId}/preview/slides`;
              const slidesResponse = await fetch(addToken(slidesUrl));
              if (slidesResponse.ok) {
                const slidesData = await slidesResponse.json();
                setSlideImages(slidesData.slides || []);
              }
            }
          } else {
            setMetadata({ supported: false, error: 'Failed to fetch preview info' });
          }
        } else {
          // Native preview - set basic metadata
          setMetadata({
            supported: true,
            type: isPdf ? 'pdf' : isImage ? 'image' : 'unsupported',
            format: formatLower
          });
        }
      } catch (err) {
        console.error('Preview metadata error:', err);
        setMetadata({ supported: false, error: 'Failed to load preview' });
      } finally {
        setIsLoading(false);
      }
    }

    fetchMetadata();
  }, [documentId, formatLower, isPptx, isDocx, supportsNativePreview]);

  const handleDownload = () => {
    window.open(downloadUrl, '_blank');
  };

  const handleOpenInNewTab = () => {
    window.open(viewUrl, '_blank');
  };

  const handlePrevPage = () => {
    if (currentPage > 1) setCurrentPage(currentPage - 1);
  };

  const handleNextPage = () => {
    if (metadata?.page_count && currentPage < metadata.page_count) {
      setCurrentPage(currentPage + 1);
    }
  };

  // Get icon based on format
  const getFormatIcon = () => {
    switch (formatLower) {
      case 'pdf':
        return <FileText className="h-16 w-16 text-red-500" />;
      case 'pptx':
      case 'ppt':
        return <Presentation className="h-16 w-16 text-orange-500" />;
      case 'docx':
      case 'doc':
        return <FileText className="h-16 w-16 text-blue-500" />;
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'gif':
      case 'webp':
      case 'svg':
        return <ImageIcon className="h-16 w-16 text-green-500" />;
      default:
        return <File className="h-16 w-16 text-muted-foreground" />;
    }
  };

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
        <p className="text-sm text-muted-foreground">{error}</p>
        <Button variant="outline" size="sm" onClick={handleDownload}>
          <Download className="h-4 w-4 mr-2" />
          Download Instead
        </Button>
      </div>
    );
  }

  // PDF Preview (native iframe)
  if (isPdf) {
    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex justify-end gap-2 mb-2">
          <Button variant="outline" size="sm" onClick={handleOpenInNewTab}>
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in New Tab
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
        <iframe
          src={viewUrl}
          className="flex-1 w-full min-h-[400px] border rounded-lg"
          title={`Preview of ${fileName}`}
        />
      </div>
    );
  }

  // Image Preview (native)
  if (isImage) {
    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex justify-end gap-2 mb-2">
          <Button variant="outline" size="sm" onClick={handleOpenInNewTab}>
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in New Tab
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
        <div className="flex-1 flex items-center justify-center bg-muted rounded-lg overflow-hidden">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={viewUrl}
            alt={`Preview of ${fileName}`}
            className="max-w-full max-h-[500px] object-contain"
            onError={() => setError('Failed to load image preview')}
          />
        </div>
      </div>
    );
  }

  // PPTX Preview (rendered slides)
  if (isPptx && metadata?.supported && metadata.type === 'slides') {
    const pageCount = metadata.page_count || slideImages.length || 1;

    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex justify-between items-center mb-2">
          {/* Slide navigation */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePrevPage}
              disabled={currentPage <= 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm text-muted-foreground">
              Slide {currentPage} of {pageCount}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={handleNextPage}
              disabled={currentPage >= pageCount}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleOpenInNewTab}>
              <ExternalLink className="h-4 w-4 mr-2" />
              Open in PowerPoint
            </Button>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          </div>
        </div>

        {/* Current slide - rendered as image */}
        <div className="flex-1 flex items-center justify-center bg-muted rounded-lg overflow-hidden min-h-[400px]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={addToken(pageUrl(currentPage))}
            alt={`Slide ${currentPage}`}
            className="max-w-full max-h-[500px] object-contain shadow-lg"
            onError={() => setError('Failed to render slide')}
          />
        </div>

        {/* Slide thumbnails */}
        {slideImages.length > 1 && (
          <div className="mt-3 flex gap-2 overflow-x-auto pb-2">
            {slideImages.map((base64, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentPage(idx + 1)}
                className={cn(
                  'flex-shrink-0 border-2 rounded overflow-hidden transition-all',
                  currentPage === idx + 1
                    ? 'border-primary ring-2 ring-primary/20'
                    : 'border-transparent hover:border-muted-foreground/30'
                )}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/png;base64,${base64}`}
                  alt={`Slide ${idx + 1}`}
                  className="w-24 h-14 object-contain bg-white"
                />
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  // DOCX Preview (HTML rendering)
  if (isDocx && metadata?.supported && metadata.type === 'html') {
    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex justify-end gap-2 mb-2">
          <Button variant="outline" size="sm" onClick={handleOpenInNewTab}>
            <ExternalLink className="h-4 w-4 mr-2" />
            Open in Word
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
        <iframe
          src={addToken(pageUrl(1))}
          className="flex-1 w-full min-h-[400px] border rounded-lg bg-white"
          title={`Preview of ${fileName}`}
        />
      </div>
    );
  }

  // Text file preview
  if (metadata?.supported && metadata.type === 'text') {
    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex justify-end gap-2 mb-2">
          <Button variant="outline" size="sm" onClick={handleOpenInNewTab}>
            <ExternalLink className="h-4 w-4 mr-2" />
            Open
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        </div>
        <iframe
          src={addToken(pageUrl(1))}
          className="flex-1 w-full min-h-[400px] border rounded-lg bg-white font-mono"
          title={`Preview of ${fileName}`}
        />
      </div>
    );
  }

  // Fallback for unsupported formats - show download option
  return (
    <div className={cn('flex flex-col items-center justify-center h-[400px] bg-muted rounded-lg gap-4', className)}>
      {getFormatIcon()}
      <div className="text-center">
        <p className="text-sm font-medium mb-1">{fileName}</p>
        <p className="text-xs text-muted-foreground mb-4">
          Preview not available for {formatLower.toUpperCase()} files
        </p>
      </div>
      <div className="flex gap-2">
        <Button variant="outline" onClick={handleOpenInNewTab}>
          <ExternalLink className="h-4 w-4 mr-2" />
          Open
        </Button>
        <Button onClick={handleDownload}>
          <Download className="h-4 w-4 mr-2" />
          Download
        </Button>
      </div>
    </div>
  );
}
