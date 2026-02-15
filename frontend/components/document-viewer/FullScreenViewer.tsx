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
import DOMPurify from "dompurify";
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
  PanelRightOpen,
  PanelRightClose,
  Sparkles,
  Network,
  Globe,
  ArrowRight,
  Database,
  AlertCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDocumentEntities } from "@/lib/api";
import { api } from "@/lib/api/client";
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
  documentData?: any; // Full document object for metadata sidebar
}

export interface Annotation {
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
  documentData,
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
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState("overview");

  // Sidebar metadata
  const enhanced = documentData?.enhanced_metadata;
  const hasEnhanced = !!enhanced && !!enhanced.summary_short;
  const hasKg = (documentData?.kg_entity_count || 0) > 0;

  // Lazy-load KG entities only when sidebar KG tab is active
  const {
    data: kgData,
    isLoading: kgLoading,
  } = useDocumentEntities(documentId, {
    enabled: isSidebarOpen && sidebarTab === "knowledge-graph" && !!documentData,
  });

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
      const response = await api.fetchWithAuth(documentUrl);
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
      const response = await api.fetchWithAuth(`/api/v1/documents/${documentId}/preview`);
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
    setState((prev) => {
      const newPage = Math.max(1, Math.min(page, prev.totalPages));
      return { ...prev, currentPage: newPage };
    });
  };

  const goToNextPage = () => {
    setState((prev) => ({ ...prev, currentPage: Math.min(prev.currentPage + 1, prev.totalPages) }));
  };
  const goToPreviousPage = () => {
    setState((prev) => ({ ...prev, currentPage: Math.max(prev.currentPage - 1, 1) }));
  };

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
    setState((prevState) => ({ ...prevState, currentSearchResult: prev }));
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

          {/* Details sidebar toggle */}
          {documentData && (
            <>
              <div className="w-px h-6 bg-border mx-1" />
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={isSidebarOpen ? "secondary" : "ghost"}
                      size="icon"
                      onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                    >
                      {isSidebarOpen ? (
                        <PanelRightClose className="h-4 w-4" />
                      ) : (
                        <PanelRightOpen className="h-4 w-4" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {isSidebarOpen ? "Hide Details" : "Show Details"}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </>
          )}
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

      {/* Main content area with optional sidebar */}
      <div className="flex-1 flex min-h-0">
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
                  ? textContent.split(new RegExp(`(${state.searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi")).map(
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
              dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(textContent) }}
            />
          )}
        </div>

        {/* Metadata Sidebar */}
        {isSidebarOpen && documentData && (
          <div className="w-[380px] border-l bg-background flex flex-col">
            <Tabs value={sidebarTab} onValueChange={setSidebarTab} className="flex-1 flex flex-col min-h-0">
              <TabsList className="mx-3 mt-3 w-fit">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="insights">
                  Insights
                  {hasEnhanced && <Sparkles className="h-3 w-3 ml-1 text-amber-500" />}
                </TabsTrigger>
                <TabsTrigger value="knowledge-graph">
                  KG
                  {hasKg && <Network className="h-3 w-3 ml-1 text-blue-500" />}
                </TabsTrigger>
              </TabsList>

              {/* Overview Tab */}
              <TabsContent value="overview" className="flex-1 min-h-0 px-4 py-3">
                <ScrollArea className="h-full">
                  <div className="space-y-4 text-sm">
                    {enhanced?.summary_short && (
                      <div className="p-3 bg-muted/50 rounded-lg">
                        <p className="font-medium text-muted-foreground mb-1">Summary</p>
                        <p>{enhanced.summary_short}</p>
                        {enhanced.summary_detailed && enhanced.summary_detailed !== enhanced.summary_short && (
                          <details className="mt-2">
                            <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                              Show detailed summary
                            </summary>
                            <p className="mt-2 text-muted-foreground">{enhanced.summary_detailed}</p>
                          </details>
                        )}
                      </div>
                    )}
                    <div>
                      <p className="font-medium text-muted-foreground">Filename</p>
                      <p>{documentData.name}</p>
                    </div>
                    <div className="flex gap-6">
                      <div>
                        <p className="font-medium text-muted-foreground">Type</p>
                        <p>{documentData.file_type || "Unknown"}</p>
                      </div>
                      {documentData.file_size > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground">Size</p>
                          <p>
                            {documentData.file_size > 1048576
                              ? `${(documentData.file_size / 1048576).toFixed(1)} MB`
                              : `${(documentData.file_size / 1024).toFixed(1)} KB`}
                          </p>
                        </div>
                      )}
                    </div>
                    {(enhanced?.language || enhanced?.document_type) && (
                      <div className="flex items-center gap-2">
                        {enhanced.language && (
                          <Badge variant="outline" className="gap-1">
                            <Globe className="h-3 w-3" />
                            {enhanced.language}
                          </Badge>
                        )}
                        {enhanced.document_type && (
                          <Badge variant="outline" className="gap-1">
                            <FileText className="h-3 w-3" />
                            {enhanced.document_type}
                          </Badge>
                        )}
                      </div>
                    )}
                    {documentData.collection && (
                      <div>
                        <p className="font-medium text-muted-foreground">Collection</p>
                        <p>{documentData.collection}</p>
                      </div>
                    )}
                    {documentData.tags && documentData.tags.length > 0 && (
                      <div>
                        <p className="font-medium text-muted-foreground">Tags</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {documentData.tags.map((tag: string) => (
                            <Badge key={tag} variant="secondary" className="text-xs">{tag}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    {documentData.created_at && (
                      <div>
                        <p className="font-medium text-muted-foreground">Created</p>
                        <p>{new Date(documentData.created_at).toLocaleString()}</p>
                      </div>
                    )}
                    <div className="space-y-2 pt-2 border-t">
                      <p className="font-medium text-muted-foreground">Status</p>
                      {documentData.chunk_count > 0 && (
                        <div className="flex items-center gap-2 text-xs">
                          <span className={`flex items-center gap-1 ${documentData.has_all_embeddings ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400"}`}>
                            {documentData.has_all_embeddings ? (
                              <Database className="h-3.5 w-3.5" />
                            ) : (
                              <AlertCircle className="h-3.5 w-3.5" />
                            )}
                            Embeddings: {documentData.embedding_coverage?.toFixed(0) || 0}% ({documentData.embedding_count}/{documentData.chunk_count})
                          </span>
                        </div>
                      )}
                      <div className="flex items-center gap-2 text-xs">
                        <Network className="h-3.5 w-3.5 text-muted-foreground" />
                        <span>
                          Knowledge Graph:{" "}
                          {documentData.kg_extraction_status === "completed" ? (
                            <span className="text-green-600 dark:text-green-400">{documentData.kg_entity_count} entities, {documentData.kg_relation_count} relations</span>
                          ) : documentData.kg_extraction_status === "processing" ? (
                            <span className="text-blue-600">Processing...</span>
                          ) : (
                            <span className="text-muted-foreground">Not extracted</span>
                          )}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
                        <span>
                          Enhancement:{" "}
                          {hasEnhanced ? (
                            <span className="text-green-600 dark:text-green-400">
                              Enhanced
                              {enhanced?.enhanced_at && (
                                <span className="text-muted-foreground ml-1">
                                  ({new Date(enhanced.enhanced_at).toLocaleDateString()})
                                </span>
                              )}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">Not enhanced</span>
                          )}
                        </span>
                      </div>
                      {documentData.images_extracted_count > 0 && (
                        <div className="flex items-center gap-2 text-xs">
                          <Database className="h-3.5 w-3.5 text-muted-foreground" />
                          <span>
                            Images:{" "}
                            {documentData.image_analysis_status === "completed" ? (
                              <span className="text-green-600 dark:text-green-400">
                                {documentData.images_analyzed_count}/{documentData.images_extracted_count} analyzed
                              </span>
                            ) : documentData.image_analysis_status === "processing" ? (
                              <span className="text-blue-600">Analyzing...</span>
                            ) : (
                              <span className="text-muted-foreground">
                                {documentData.images_extracted_count} found, not analyzed
                              </span>
                            )}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </ScrollArea>
              </TabsContent>

              {/* Insights Tab */}
              <TabsContent value="insights" className="flex-1 min-h-0 px-4 py-3">
                <ScrollArea className="h-full">
                  {hasEnhanced ? (
                    <div className="space-y-5 text-sm">
                      {enhanced?.keywords && enhanced.keywords.length > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground mb-2">Keywords</p>
                          <div className="flex flex-wrap gap-1.5">
                            {enhanced.keywords.map((kw: string) => (
                              <Badge key={kw} variant="secondary" className="text-xs">{kw}</Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {enhanced?.topics && enhanced.topics.length > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground mb-2">Topics</p>
                          <div className="flex flex-wrap gap-1.5">
                            {enhanced.topics.map((topic: string) => (
                              <Badge key={topic} variant="outline" className="text-xs bg-blue-50 dark:bg-blue-950/30">{topic}</Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {enhanced?.entities && Object.keys(enhanced.entities).length > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground mb-2">Entities</p>
                          <div className="space-y-2">
                            {Object.entries(enhanced.entities).map(([type, items]) => (
                              <div key={type}>
                                <p className="text-xs font-medium text-muted-foreground capitalize mb-1">{type}</p>
                                <div className="flex flex-wrap gap-1">
                                  {(items as string[]).map((item: string) => (
                                    <Badge key={item} variant="outline" className="text-xs">{item}</Badge>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {enhanced?.hypothetical_questions && enhanced.hypothetical_questions.length > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground mb-2">Hypothetical Questions</p>
                          <ol className="list-decimal list-inside space-y-1.5 text-sm text-muted-foreground">
                            {enhanced.hypothetical_questions.map((q: string, i: number) => (
                              <li key={i}>{q}</li>
                            ))}
                          </ol>
                        </div>
                      )}
                      {enhanced?.model_used && (
                        <div className="pt-2 border-t text-xs text-muted-foreground">
                          Enhanced with {enhanced.model_used}
                          {enhanced.enhanced_at && <> on {new Date(enhanced.enhanced_at).toLocaleString()}</>}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
                      <Sparkles className="h-10 w-10 mb-3 opacity-30" />
                      <p className="font-medium">Not yet enhanced</p>
                      <p className="text-sm mt-1">Enhance this document to see insights.</p>
                    </div>
                  )}
                </ScrollArea>
              </TabsContent>

              {/* Knowledge Graph Tab */}
              <TabsContent value="knowledge-graph" className="flex-1 min-h-0 px-4 py-3">
                <ScrollArea className="h-full">
                  {kgLoading ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full" />
                    </div>
                  ) : kgData?.entities && kgData.entities.length > 0 ? (
                    <div className="space-y-5 text-sm">
                      <div>
                        <p className="font-medium text-muted-foreground mb-2">Entities ({kgData.entities.length})</p>
                        <div className="space-y-2">
                          {kgData.entities.map((entity: any) => (
                            <div key={entity.id} className="p-2 rounded border bg-muted/30">
                              <div className="flex items-center justify-between">
                                <span className="font-medium">{entity.name}</span>
                                <Badge variant="outline" className="text-xs capitalize">{entity.entity_type}</Badge>
                              </div>
                              {entity.description && (
                                <p className="text-xs text-muted-foreground mt-1">{entity.description}</p>
                              )}
                              <p className="text-xs text-muted-foreground mt-1">
                                {entity.mention_count} mention{entity.mention_count !== 1 ? "s" : ""}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                      {kgData.relations && kgData.relations.length > 0 && (
                        <div>
                          <p className="font-medium text-muted-foreground mb-2">Relations ({kgData.relations.length})</p>
                          <div className="space-y-1.5">
                            {kgData.relations.map((rel: any) => (
                              <div key={rel.id} className="flex items-center gap-2 p-2 rounded border bg-muted/30 text-xs">
                                <span className="font-medium">{rel.source_entity_name}</span>
                                <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                                <Badge variant="outline" className="text-xs flex-shrink-0">{rel.relation_type}</Badge>
                                <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                                <span className="font-medium">{rel.target_entity_name}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
                      <Network className="h-10 w-10 mb-3 opacity-30" />
                      <p className="font-medium">No knowledge graph data</p>
                      <p className="text-sm mt-1">Extract knowledge graph to see entities and relationships.</p>
                    </div>
                  )}
                </ScrollArea>
              </TabsContent>
            </Tabs>
          </div>
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
