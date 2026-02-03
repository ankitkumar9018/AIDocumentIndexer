"use client";

/**
 * Full-Screen Document Viewer
 * ============================
 *
 * A comprehensive document viewer that supports:
 * - PDF viewing with page navigation
 * - Image viewing with zoom/pan
 * - Text/Markdown preview
 * - Office document preview (via conversion)
 * - Full-screen mode
 * - Zoom controls
 * - Keyboard shortcuts
 * - Annotations
 * - Search within document
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  X,
  ZoomIn,
  ZoomOut,
  RotateCw,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Minimize2,
  Download,
  Search,
  Printer,
  Highlighter,
  MessageSquare,
  ChevronUp,
  ChevronDown,
  FileText,
  Image as ImageIcon,
  File,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// Types
interface DocumentViewerProps {
  documentId: string;
  documentUrl: string;
  documentName: string;
  documentType: string; // pdf, image, text, markdown, docx, xlsx, pptx
  onClose: () => void;
  initialPage?: number;
  annotations?: Annotation[];
  onAnnotationAdd?: (annotation: Annotation) => void;
}

interface Annotation {
  id: string;
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
  type: "highlight" | "comment" | "underline";
  content?: string;
  color?: string;
}

interface ViewerState {
  zoom: number;
  rotation: number;
  currentPage: number;
  totalPages: number;
  isFullScreen: boolean;
  isSearchOpen: boolean;
  searchQuery: string;
  searchResults: number[];
  currentSearchResult: number;
}

// Constants
const ZOOM_LEVELS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3];
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 5;
const ZOOM_STEP = 0.25;

export function FullScreenViewer({
  documentId,
  documentUrl,
  documentName,
  documentType,
  onClose,
  initialPage = 1,
  annotations = [],
  onAnnotationAdd,
}: DocumentViewerProps) {
  // State
  const [state, setState] = useState<ViewerState>({
    zoom: 1,
    rotation: 0,
    currentPage: initialPage,
    totalPages: 1,
    isFullScreen: false,
    isSearchOpen: false,
    searchQuery: "",
    searchResults: [],
    currentSearchResult: 0,
  });

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pdfDocument, setPdfDocument] = useState<any>(null);
  const [textContent, setTextContent] = useState<string>("");
  const [imageUrl, setImageUrl] = useState<string>("");

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<HTMLDivElement>(null);

  // Determine viewer type
  const getViewerType = (): "pdf" | "image" | "text" | "office" => {
    const type = documentType.toLowerCase();
    if (type === "pdf" || type === "application/pdf") return "pdf";
    if (type.startsWith("image/") || ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(type)) return "image";
    if (["txt", "md", "markdown", "json", "xml", "csv", "html", "text/plain"].includes(type)) return "text";
    return "office"; // docx, xlsx, pptx
  };

  const viewerType = getViewerType();

  // Load document
  useEffect(() => {
    const loadDocument = async () => {
      setIsLoading(true);
      setError(null);

      try {
        switch (viewerType) {
          case "pdf":
            await loadPdf();
            break;
          case "image":
            setImageUrl(documentUrl);
            setIsLoading(false);
            break;
          case "text":
            await loadText();
            break;
          case "office":
            await loadOfficePreview();
            break;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load document");
        setIsLoading(false);
      }
    };

    loadDocument();
  }, [documentUrl, documentType]);

  // PDF loading
  const loadPdf = async () => {
    try {
      // Dynamic import of PDF.js with ESM module
      // See: https://github.com/mozilla/pdf.js/issues/17319
      const pdfjsLib = await import("pdfjs-dist/build/pdf.mjs");

      // Use CDN worker with .mjs extension for modern browsers
      pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

      const loadingTask = pdfjsLib.getDocument(documentUrl);
      const pdf = await loadingTask.promise;

      setPdfDocument(pdf);
      setState((prev) => ({ ...prev, totalPages: pdf.numPages }));
      setIsLoading(false);
    } catch (err) {
      console.error("Failed to load PDF:", err);
      throw new Error("Failed to load PDF");
    }
  };

  // Text loading
  const loadText = async () => {
    try {
      const response = await fetch(documentUrl);
      const text = await response.text();
      setTextContent(text);
      setIsLoading(false);
    } catch (err) {
      throw new Error("Failed to load text file");
    }
  };

  // Office preview (via server conversion)
  const loadOfficePreview = async () => {
    try {
      // Server converts to HTML or images
      const response = await fetch(`/api/v1/documents/${documentId}/preview`);
      if (!response.ok) throw new Error("Preview not available");

      const data = await response.json();
      if (data.type === "html") {
        setTextContent(data.content);
      } else if (data.type === "images") {
        setState((prev) => ({ ...prev, totalPages: data.pages.length }));
        setImageUrl(data.pages[0]);
      }
      setIsLoading(false);
    } catch (err) {
      throw new Error("Failed to load office document preview");
    }
  };

  // Render PDF page
  useEffect(() => {
    const renderPage = async () => {
      if (!pdfDocument || !canvasRef.current) return;

      try {
        const page = await pdfDocument.getPage(state.currentPage);
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");
        if (!context) return;

        const viewport = page.getViewport({
          scale: state.zoom,
          rotation: state.rotation,
        });

        canvas.width = viewport.width;
        canvas.height = viewport.height;

        await page.render({
          canvasContext: context,
          viewport: viewport,
        }).promise;
      } catch (err) {
        console.error("Failed to render page:", err);
      }
    };

    if (viewerType === "pdf") {
      renderPage();
    }
  }, [pdfDocument, state.currentPage, state.zoom, state.rotation]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if search is focused
      if (state.isSearchOpen && e.target instanceof HTMLInputElement) return;

      switch (e.key) {
        case "Escape":
          if (state.isFullScreen) {
            toggleFullScreen();
          } else {
            onClose();
          }
          break;
        case "ArrowLeft":
        case "PageUp":
          goToPreviousPage();
          break;
        case "ArrowRight":
        case "PageDown":
          goToNextPage();
          break;
        case "+":
        case "=":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomIn();
          }
          break;
        case "-":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomOut();
          }
          break;
        case "0":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            resetZoom();
          }
          break;
        case "f":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            setState((prev) => ({ ...prev, isSearchOpen: !prev.isSearchOpen }));
          }
          break;
        case "Home":
          goToPage(1);
          break;
        case "End":
          goToPage(state.totalPages);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [state.isFullScreen, state.isSearchOpen, state.totalPages]);

  // Navigation functions
  const goToPage = (page: number) => {
    const newPage = Math.max(1, Math.min(page, state.totalPages));
    setState((prev) => ({ ...prev, currentPage: newPage }));
  };

  const goToNextPage = () => goToPage(state.currentPage + 1);
  const goToPreviousPage = () => goToPage(state.currentPage - 1);

  // Zoom functions
  const zoomIn = () => {
    setState((prev) => ({
      ...prev,
      zoom: Math.min(prev.zoom + ZOOM_STEP, MAX_ZOOM),
    }));
  };

  const zoomOut = () => {
    setState((prev) => ({
      ...prev,
      zoom: Math.max(prev.zoom - ZOOM_STEP, MIN_ZOOM),
    }));
  };

  const resetZoom = () => {
    setState((prev) => ({ ...prev, zoom: 1 }));
  };

  const setZoom = (zoom: number) => {
    setState((prev) => ({
      ...prev,
      zoom: Math.max(MIN_ZOOM, Math.min(zoom, MAX_ZOOM)),
    }));
  };

  // Rotation
  const rotate = () => {
    setState((prev) => ({
      ...prev,
      rotation: (prev.rotation + 90) % 360,
    }));
  };

  // Full screen
  const toggleFullScreen = async () => {
    if (!containerRef.current) return;

    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen();
        setState((prev) => ({ ...prev, isFullScreen: true }));
      } else {
        await document.exitFullscreen();
        setState((prev) => ({ ...prev, isFullScreen: false }));
      }
    } catch (err) {
      console.error("Fullscreen error:", err);
    }
  };

  // Download
  const downloadDocument = () => {
    const link = document.createElement("a");
    link.href = documentUrl;
    link.download = documentName;
    link.click();
  };

  // Print
  const printDocument = () => {
    const printWindow = window.open(documentUrl, "_blank");
    printWindow?.print();
  };

  // Search
  const handleSearch = useCallback(
    async (query: string) => {
      if (!query.trim()) {
        setState((prev) => ({
          ...prev,
          searchResults: [],
          currentSearchResult: 0,
        }));
        return;
      }

      // For PDF, search through all pages
      if (viewerType === "pdf" && pdfDocument) {
        const results: number[] = [];
        for (let i = 1; i <= state.totalPages; i++) {
          const page = await pdfDocument.getPage(i);
          const textContent = await page.getTextContent();
          const text = textContent.items.map((item: any) => item.str).join(" ");
          if (text.toLowerCase().includes(query.toLowerCase())) {
            results.push(i);
          }
        }
        setState((prev) => ({
          ...prev,
          searchResults: results,
          currentSearchResult: 0,
        }));
        if (results.length > 0) {
          goToPage(results[0]);
        }
      }

      // For text, highlight matches
      if (viewerType === "text") {
        const matches = textContent
          .toLowerCase()
          .split(query.toLowerCase()).length - 1;
        // Just show match count for text
        setState((prev) => ({
          ...prev,
          searchResults: Array(matches).fill(1),
          currentSearchResult: 0,
        }));
      }
    },
    [pdfDocument, state.totalPages, textContent, viewerType]
  );

  const goToNextSearchResult = () => {
    if (state.searchResults.length === 0) return;
    const next = (state.currentSearchResult + 1) % state.searchResults.length;
    setState((prev) => ({ ...prev, currentSearchResult: next }));
    if (viewerType === "pdf") {
      goToPage(state.searchResults[next]);
    }
  };

  const goToPreviousSearchResult = () => {
    if (state.searchResults.length === 0) return;
    const prev =
      (state.currentSearchResult - 1 + state.searchResults.length) %
      state.searchResults.length;
    setState((prev) => ({ ...prev, currentSearchResult: prev }));
    if (viewerType === "pdf") {
      goToPage(state.searchResults[prev]);
    }
  };

  // Render
  return (
    <div
      ref={containerRef}
      className={cn(
        "fixed inset-0 z-50 flex flex-col bg-background",
        state.isFullScreen && "fullscreen"
      )}
    >
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b px-4 py-2 bg-muted/50">
        {/* Left: Document info */}
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-5 w-5" />
          </Button>
          <div className="flex items-center gap-2">
            {viewerType === "pdf" && <FileText className="h-4 w-4 text-red-500" />}
            {viewerType === "image" && <ImageIcon className="h-4 w-4 text-blue-500" />}
            {viewerType === "text" && <FileText className="h-4 w-4 text-gray-500" />}
            {viewerType === "office" && <File className="h-4 w-4 text-green-500" />}
            <span className="font-medium truncate max-w-[300px]">{documentName}</span>
          </div>
        </div>

        {/* Center: Navigation (for multi-page docs) */}
        {state.totalPages > 1 && (
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={goToPreviousPage}
              disabled={state.currentPage <= 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-1">
              <Input
                type="number"
                min={1}
                max={state.totalPages}
                value={state.currentPage}
                onChange={(e) => goToPage(parseInt(e.target.value) || 1)}
                className="w-16 text-center h-8"
              />
              <span className="text-sm text-muted-foreground">
                / {state.totalPages}
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={goToNextPage}
              disabled={state.currentPage >= state.totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Right: Tools */}
        <div className="flex items-center gap-1">
          {/* Search */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={state.isSearchOpen ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      isSearchOpen: !prev.isSearchOpen,
                    }))
                  }
                >
                  <Search className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Search (Ctrl+F)</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <div className="w-px h-6 bg-border mx-1" />

          {/* Zoom controls */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={zoomOut}>
                  <ZoomOut className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom Out (Ctrl+-)</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <select
            value={state.zoom}
            onChange={(e) => setZoom(parseFloat(e.target.value))}
            className="h-8 px-2 text-sm border rounded bg-background"
          >
            {ZOOM_LEVELS.map((level) => (
              <option key={level} value={level}>
                {Math.round(level * 100)}%
              </option>
            ))}
          </select>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={zoomIn}>
                  <ZoomIn className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom In (Ctrl++)</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <div className="w-px h-6 bg-border mx-1" />

          {/* Rotate */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={rotate}>
                  <RotateCw className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Rotate</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <div className="w-px h-6 bg-border mx-1" />

          {/* Download */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={downloadDocument}>
                  <Download className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Download</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {/* Print */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={printDocument}>
                  <Printer className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Print</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <div className="w-px h-6 bg-border mx-1" />

          {/* Fullscreen */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={toggleFullScreen}>
                  {state.isFullScreen ? (
                    <Minimize2 className="h-4 w-4" />
                  ) : (
                    <Maximize2 className="h-4 w-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {state.isFullScreen ? "Exit Fullscreen" : "Fullscreen"}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      {/* Search bar */}
      {state.isSearchOpen && (
        <div className="flex items-center gap-2 px-4 py-2 border-b bg-muted/30">
          <Search className="h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search in document..."
            value={state.searchQuery}
            onChange={(e) => {
              setState((prev) => ({ ...prev, searchQuery: e.target.value }));
              handleSearch(e.target.value);
            }}
            className="flex-1 h-8"
            autoFocus
          />
          {state.searchResults.length > 0 && (
            <>
              <span className="text-sm text-muted-foreground whitespace-nowrap">
                {state.currentSearchResult + 1} of {state.searchResults.length}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={goToPreviousSearchResult}
              >
                <ChevronUp className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={goToNextSearchResult}
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            </>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() =>
              setState((prev) => ({
                ...prev,
                isSearchOpen: false,
                searchQuery: "",
                searchResults: [],
              }))
            }
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Document content */}
      <div
        ref={viewerRef}
        className="flex-1 overflow-auto bg-muted/20 flex items-center justify-center p-4"
      >
        {isLoading && (
          <div className="flex flex-col items-center gap-4">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-muted-foreground">Loading document...</span>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center gap-4 text-destructive">
            <FileText className="h-12 w-12" />
            <span>{error}</span>
          </div>
        )}

        {!isLoading && !error && viewerType === "pdf" && (
          <div
            className="bg-white shadow-lg"
            style={{
              transform: `rotate(${state.rotation}deg)`,
              transition: "transform 0.3s",
            }}
          >
            <canvas ref={canvasRef} />
          </div>
        )}

        {!isLoading && !error && viewerType === "image" && (
          <img
            src={imageUrl}
            alt={documentName}
            className="max-w-full max-h-full object-contain shadow-lg"
            style={{
              transform: `scale(${state.zoom}) rotate(${state.rotation}deg)`,
              transition: "transform 0.2s",
            }}
          />
        )}

        {!isLoading && !error && viewerType === "text" && (
          <div
            className="w-full max-w-4xl bg-white dark:bg-gray-900 p-8 shadow-lg rounded-lg overflow-auto"
            style={{ transform: `scale(${state.zoom})` }}
          >
            <pre className="whitespace-pre-wrap font-mono text-sm">
              {state.searchQuery
                ? textContent.split(new RegExp(`(${state.searchQuery})`, "gi")).map(
                    (part, i) =>
                      part.toLowerCase() === state.searchQuery.toLowerCase() ? (
                        <mark key={i} className="bg-yellow-300 dark:bg-yellow-700">
                          {part}
                        </mark>
                      ) : (
                        part
                      )
                  )
                : textContent}
            </pre>
          </div>
        )}

        {!isLoading && !error && viewerType === "office" && textContent && (
          <div
            className="w-full max-w-4xl bg-white dark:bg-gray-900 p-8 shadow-lg rounded-lg overflow-auto"
            style={{ transform: `scale(${state.zoom})` }}
            dangerouslySetInnerHTML={{ __html: textContent }}
          />
        )}
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="flex items-center justify-center gap-6 px-4 py-1 border-t text-xs text-muted-foreground bg-muted/30">
        <span>
          <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">←</kbd>{" "}
          <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">→</kbd> Navigate
        </span>
        <span>
          <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Ctrl</kbd>+
          <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">+/-</kbd> Zoom
        </span>
        <span>
          <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Ctrl</kbd>+
          <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">F</kbd> Search
        </span>
        <span>
          <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Esc</kbd> Close
        </span>
      </div>
    </div>
  );
}

export default FullScreenViewer;
