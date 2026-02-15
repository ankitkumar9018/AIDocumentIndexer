'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  Download,
  Maximize2,
  Minimize2,
  Search,
  X,
  RotateCw,
  FileText,
  Loader2,
} from 'lucide-react';

// PDF.js types
interface PDFDocumentProxy {
  numPages: number;
  getPage(pageNumber: number): Promise<PDFPageProxy>;
  destroy(): void;
}

interface PDFPageProxy {
  getViewport(params: { scale: number; rotation?: number }): PDFPageViewport;
  render(params: { canvasContext: CanvasRenderingContext2D; viewport: PDFPageViewport }): PDFRenderTask;
  getTextContent(): Promise<PDFTextContent>;
}

interface PDFPageViewport {
  width: number;
  height: number;
}

interface PDFRenderTask {
  promise: Promise<void>;
  cancel(): void;
}

interface PDFTextContent {
  items: Array<{ str: string; transform: number[] }>;
}

interface PDFViewerProps {
  /** URL to the PDF document */
  url: string;
  /** Initial page number (1-indexed) */
  initialPage?: number;
  /** Initial zoom level */
  initialZoom?: number;
  /** Callback when page changes */
  onPageChange?: (page: number) => void;
  /** Callback when text is selected */
  onTextSelect?: (text: string, page: number) => void;
  /** Highlights to show (from search) */
  highlights?: Array<{
    page: number;
    text: string;
    position?: { x: number; y: number };
  }>;
  /** Height of the viewer */
  height?: string | number;
  /** CSS class name */
  className?: string;
}

export function PDFViewer({
  url,
  initialPage = 1,
  initialZoom = 1.0,
  onPageChange,
  onTextSelect,
  highlights = [],
  height = '100%',
  className = '',
}: PDFViewerProps) {
  // State
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [numPages, setNumPages] = useState(0);
  const [zoom, setZoom] = useState(initialZoom);
  const [rotation, setRotation] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [searchResults, setSearchResults] = useState<number[]>([]);
  const [showSearch, setShowSearch] = useState(false);

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<PDFRenderTask | null>(null);

  // Load PDF.js library dynamically
  useEffect(() => {
    const loadPdfJs = async () => {
      if (typeof window === 'undefined') return;

      // Check if already loaded
      if ((window as any).pdfjsLib) return;

      // Load PDF.js from CDN
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
      script.async = true;
      document.head.appendChild(script);

      await new Promise<void>((resolve) => {
        script.onload = () => {
          // Set worker path
          (window as any).pdfjsLib.GlobalWorkerOptions.workerSrc =
            'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
          resolve();
        };
      });
    };

    loadPdfJs();
  }, []);

  // Load PDF document
  useEffect(() => {
    let pollInterval: ReturnType<typeof setInterval> | null = null;

    const loadDocument = async () => {
      if (typeof window === 'undefined' || !(window as any).pdfjsLib) {
        // Wait for PDF.js to load
        pollInterval = setInterval(() => {
          if ((window as any).pdfjsLib) {
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = null;
            loadDocument();
          }
        }, 100);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const pdfjsLib = (window as any).pdfjsLib;
        const loadingTask = pdfjsLib.getDocument(url);
        const doc = await loadingTask.promise;

        setPdfDoc(doc);
        setNumPages(doc.numPages);
        setCurrentPage(Math.min(initialPage, doc.numPages));
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to load PDF:', err);
        setError('Failed to load PDF document');
        setIsLoading(false);
      }
    };

    loadDocument();

    return () => {
      if (pollInterval) clearInterval(pollInterval);
      pdfDoc?.destroy();
    };
  }, [url]);

  // Render current page
  const renderPage = useCallback(async () => {
    if (!pdfDoc || !canvasRef.current) return;

    // Cancel any ongoing render
    if (renderTaskRef.current) {
      renderTaskRef.current.cancel();
    }

    try {
      const page = await pdfDoc.getPage(currentPage);
      const viewport = page.getViewport({ scale: zoom, rotation });

      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      if (!context) return;

      canvas.width = viewport.width;
      canvas.height = viewport.height;

      const renderTask = page.render({
        canvasContext: context,
        viewport,
      });
      renderTaskRef.current = renderTask;

      await renderTask.promise;

      // Render text layer for selection
      if (textLayerRef.current) {
        const textContent = await page.getTextContent();
        renderTextLayer(textContent, viewport);
      }
    } catch (err: any) {
      if (err?.name !== 'RenderingCancelledException') {
        console.error('Failed to render page:', err);
      }
    }
  }, [pdfDoc, currentPage, zoom, rotation]);

  // Render text layer for text selection
  const renderTextLayer = (textContent: PDFTextContent, viewport: PDFPageViewport) => {
    if (!textLayerRef.current) return;

    textLayerRef.current.innerHTML = '';
    textLayerRef.current.style.width = `${viewport.width}px`;
    textLayerRef.current.style.height = `${viewport.height}px`;

    textContent.items.forEach((item) => {
      const span = document.createElement('span');
      span.textContent = item.str;
      span.style.position = 'absolute';
      span.style.left = `${item.transform[4] * zoom}px`;
      span.style.bottom = `${item.transform[5] * zoom}px`;
      span.style.fontSize = `${Math.abs(item.transform[0]) * zoom}px`;
      span.style.color = 'transparent';
      span.style.cursor = 'text';
      textLayerRef.current?.appendChild(span);
    });
  };

  useEffect(() => {
    renderPage();
  }, [renderPage]);

  // Notify page change
  useEffect(() => {
    onPageChange?.(currentPage);
  }, [currentPage, onPageChange]);

  // Navigation handlers
  const goToPage = (page: number) => {
    const newPage = Math.max(1, Math.min(page, numPages));
    setCurrentPage(newPage);
  };

  const previousPage = () => goToPage(currentPage - 1);
  const nextPage = () => goToPage(currentPage + 1);

  // Zoom handlers
  const zoomIn = () => setZoom((z) => Math.min(z + 0.25, 3.0));
  const zoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.25));
  const resetZoom = () => setZoom(1.0);

  // Rotation handler
  const rotate = () => setRotation((r) => (r + 90) % 360);

  // Fullscreen handler
  const toggleFullscreen = () => {
    if (!containerRef.current) return;

    if (!isFullscreen) {
      containerRef.current.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
    setIsFullscreen(!isFullscreen);
  };

  // Download handler
  const download = () => {
    const link = document.createElement('a');
    link.href = url;
    link.download = url.split('/').pop() || 'document.pdf';
    link.click();
  };

  // Search handler
  const handleSearch = async () => {
    if (!pdfDoc || !searchText.trim()) return;

    const results: number[] = [];

    for (let i = 1; i <= numPages; i++) {
      const page = await pdfDoc.getPage(i);
      const textContent = await page.getTextContent();
      const text = textContent.items.map((item) => item.str).join(' ');

      if (text.toLowerCase().includes(searchText.toLowerCase())) {
        results.push(i);
      }
    }

    setSearchResults(results);
    if (results.length > 0) {
      goToPage(results[0]);
    }
  };

  // Handle text selection
  const handleTextSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      onTextSelect?.(selection.toString().trim(), currentPage);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      switch (e.key) {
        case 'ArrowLeft':
          previousPage();
          break;
        case 'ArrowRight':
          nextPage();
          break;
        case '+':
        case '=':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomIn();
          }
          break;
        case '-':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomOut();
          }
          break;
        case 'f':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            setShowSearch(true);
          }
          break;
        case 'Escape':
          setShowSearch(false);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentPage, numPages]);

  if (error) {
    return (
      <div className={`flex flex-col items-center justify-center p-8 ${className}`} style={{ height }}>
        <FileText className="w-16 h-16 text-muted-foreground mb-4" />
        <p className="text-lg font-medium text-foreground">Failed to load PDF</p>
        <p className="text-sm text-muted-foreground mt-2">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-lg"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`flex flex-col bg-muted rounded-lg overflow-hidden ${className}`}
      style={{ height }}
    >
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-card border-b border-border">
        {/* Left: Page navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={previousPage}
            disabled={currentPage <= 1}
            className="p-1.5 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            title="Previous page"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-1">
            <input
              type="number"
              value={currentPage}
              onChange={(e) => goToPage(parseInt(e.target.value) || 1)}
              className="w-12 px-2 py-1 text-center text-sm rounded bg-muted border border-input"
              min={1}
              max={numPages}
            />
            <span className="text-sm text-muted-foreground">/ {numPages}</span>
          </div>

          <button
            onClick={nextPage}
            disabled={currentPage >= numPages}
            className="p-1.5 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
            title="Next page"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>

        {/* Center: Zoom controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={zoomOut}
            disabled={zoom <= 0.25}
            className="p-1.5 rounded hover:bg-muted disabled:opacity-50"
            title="Zoom out"
          >
            <ZoomOut className="w-5 h-5" />
          </button>

          <button
            onClick={resetZoom}
            className="px-2 py-1 text-sm rounded hover:bg-muted"
            title="Reset zoom"
          >
            {Math.round(zoom * 100)}%
          </button>

          <button
            onClick={zoomIn}
            disabled={zoom >= 3.0}
            className="p-1.5 rounded hover:bg-muted disabled:opacity-50"
            title="Zoom in"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowSearch(!showSearch)}
            className={`p-1.5 rounded hover:bg-muted ${showSearch ? 'bg-muted' : ''}`}
            title="Search (Ctrl+F)"
          >
            <Search className="w-5 h-5" />
          </button>

          <button
            onClick={rotate}
            className="p-1.5 rounded hover:bg-muted"
            title="Rotate"
          >
            <RotateCw className="w-5 h-5" />
          </button>

          <button
            onClick={download}
            className="p-1.5 rounded hover:bg-muted"
            title="Download"
          >
            <Download className="w-5 h-5" />
          </button>

          <button
            onClick={toggleFullscreen}
            className="p-1.5 rounded hover:bg-muted"
            title="Toggle fullscreen"
          >
            {isFullscreen ? (
              <Minimize2 className="w-5 h-5" />
            ) : (
              <Maximize2 className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Search bar */}
      {showSearch && (
        <div className="flex items-center gap-2 px-4 py-2 bg-card border-b border-border">
          <Search className="w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search in document..."
            className="flex-1 px-2 py-1 text-sm bg-muted rounded border border-input focus:outline-none focus:ring-2 focus:ring-ring"
            autoFocus
          />
          <span className="text-xs text-muted-foreground">
            {searchResults.length > 0 && `${searchResults.length} results`}
          </span>
          <button
            onClick={() => setShowSearch(false)}
            className="p-1 rounded hover:bg-muted"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* PDF Viewer */}
      <div className="flex-1 overflow-auto flex items-center justify-center p-4 bg-muted/50">
        {isLoading ? (
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Loading PDF...</p>
          </div>
        ) : (
          <div className="relative shadow-lg" onMouseUp={handleTextSelection}>
            <canvas ref={canvasRef} className="max-w-full" />
            <div
              ref={textLayerRef}
              className="absolute top-0 left-0 pointer-events-auto select-text"
              style={{ userSelect: 'text' }}
            />
          </div>
        )}
      </div>

      {/* Highlight indicators */}
      {highlights.filter((h) => h.page === currentPage).length > 0 && (
        <div className="absolute bottom-4 right-4 bg-yellow-500/20 text-yellow-800 dark:text-yellow-200 px-3 py-1 rounded-full text-sm">
          {highlights.filter((h) => h.page === currentPage).length} highlights on this page
        </div>
      )}
    </div>
  );
}

export default PDFViewer;
