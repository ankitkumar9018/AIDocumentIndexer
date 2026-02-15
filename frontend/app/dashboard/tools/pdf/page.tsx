"use client";

/**
 * PDF Tools Page
 * ===============
 *
 * Provides a UI for PDF manipulation tools:
 * - Merge PDFs
 * - Split PDF
 * - Extract pages
 * - Rotate pages
 * - Compress PDF
 * - Convert to/from images
 * - Edit metadata
 * - Add watermark
 * - Rearrange pages
 */

import React, { useState, useCallback } from "react";
import {
  FileText,
  Merge,
  Split,
  RotateCw,
  Minimize2,
  Image as ImageIcon,
  FileEdit,
  Droplet,
  ArrowUpDown,
  Upload,
  Download,
  Trash2,
  Plus,
  GripVertical,
  X,
  Check,
  Loader2,
  Database,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { api } from "@/lib/api";

// Types
interface UploadedFile {
  id: string;
  file: File;
  name: string;
  size: number;
  preview?: string;
}

type ToolType =
  | "merge"
  | "split"
  | "extract"
  | "rotate"
  | "compress"
  | "to-images"
  | "from-images"
  | "metadata"
  | "watermark"
  | "rearrange";

export default function PDFToolsPage() {
  const [activeTool, setActiveTool] = useState<ToolType>("merge");
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [indexAfterProcess, setIndexAfterProcess] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [lastProcessedResult, setLastProcessedResult] = useState<{ blob: Blob; filename: string } | null>(null);

  // Tool-specific state
  const [splitRanges, setSplitRanges] = useState("1-5,6-10");
  const [extractPages, setExtractPages] = useState("1,3,5");
  const [rotationAngle, setRotationAngle] = useState(90);
  const [rotatePages, setRotatePages] = useState("");
  const [compressionQuality, setCompressionQuality] = useState(75);
  const [imageDpi, setImageDpi] = useState(150);
  const [imageFormat, setImageFormat] = useState("png");
  const [watermarkText, setWatermarkText] = useState("CONFIDENTIAL");
  const [watermarkOpacity, setWatermarkOpacity] = useState(0.3);
  const [watermarkAngle, setWatermarkAngle] = useState(45);
  const [metadata, setMetadata] = useState({
    title: "",
    author: "",
    subject: "",
    keywords: "",
  });
  const [pageOrder, setPageOrder] = useState("");

  // File handling
  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = e.target.files;
      if (!selectedFiles) return;

      const newFiles: UploadedFile[] = Array.from(selectedFiles).map((file) => ({
        id: Math.random().toString(36).substr(2, 9),
        file,
        name: file.name,
        size: file.size,
      }));

      setFiles((prev) => [...prev, ...newFiles]);
    },
    []
  );

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFiles = e.dataTransfer.files;
    if (!droppedFiles) return;

    const newFiles: UploadedFile[] = Array.from(droppedFiles)
      .filter(
        (file) =>
          file.type === "application/pdf" || file.type.startsWith("image/")
      )
      .map((file) => ({
        id: Math.random().toString(36).substr(2, 9),
        file,
        name: file.name,
        size: file.size,
      }));

    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const clearFiles = () => {
    setFiles([]);
  };

  const moveFile = (fromIndex: number, toIndex: number) => {
    setFiles((prev) => {
      const newFiles = [...prev];
      const [removed] = newFiles.splice(fromIndex, 1);
      newFiles.splice(toIndex, 0, removed);
      return newFiles;
    });
  };

  // Process PDFs
  const processFiles = async () => {
    if (files.length === 0) {
      toast.error("Please upload at least one file.");
      return;
    }

    setIsProcessing(true);

    try {
      const formData = new FormData();

      // Add files
      files.forEach((f) => {
        formData.append("files", f.file);
      });

      // Add tool-specific parameters
      let endpoint = `/api/v1/pdf-tools/${activeTool}`;
      let responseType: "blob" | "json" = "blob";

      switch (activeTool) {
        case "merge":
          // Files already added
          break;
        case "split":
          endpoint = `/api/v1/pdf-tools/split`;
          formData.set("file", files[0].file);
          formData.set("ranges", splitRanges);
          break;
        case "extract":
          endpoint = `/api/v1/pdf-tools/extract`;
          formData.set("file", files[0].file);
          formData.set("pages", extractPages);
          break;
        case "rotate":
          endpoint = `/api/v1/pdf-tools/rotate`;
          formData.set("file", files[0].file);
          formData.set("rotation", rotationAngle.toString());
          if (rotatePages) formData.set("pages", rotatePages);
          break;
        case "compress":
          endpoint = `/api/v1/pdf-tools/compress`;
          formData.set("file", files[0].file);
          formData.set("quality", compressionQuality.toString());
          break;
        case "to-images":
          endpoint = `/api/v1/pdf-tools/to-images`;
          formData.set("file", files[0].file);
          formData.set("dpi", imageDpi.toString());
          formData.set("format", imageFormat);
          break;
        case "from-images":
          endpoint = `/api/v1/pdf-tools/from-images`;
          // Files already added
          break;
        case "metadata":
          endpoint = `/api/v1/pdf-tools/metadata`;
          formData.set("file", files[0].file);
          if (metadata.title) formData.set("title", metadata.title);
          if (metadata.author) formData.set("author", metadata.author);
          if (metadata.subject) formData.set("subject", metadata.subject);
          if (metadata.keywords) formData.set("keywords", metadata.keywords);
          break;
        case "watermark":
          endpoint = `/api/v1/pdf-tools/watermark`;
          formData.set("file", files[0].file);
          formData.set("text", watermarkText);
          formData.set("opacity", watermarkOpacity.toString());
          formData.set("angle", watermarkAngle.toString());
          break;
        case "rearrange":
          endpoint = `/api/v1/pdf-tools/rearrange`;
          formData.set("file", files[0].file);
          formData.set("order", pageOrder);
          break;
      }

      const response = await api.fetchWithAuth(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      // Get the result
      const blob = await response.blob();

      // Determine filename
      const contentDisposition = response.headers.get("Content-Disposition");
      let filename = `${activeTool}_result`;
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="(.+)"/);
        if (match) filename = match[1];
      } else {
        const contentType = response.headers.get("Content-Type");
        if (contentType?.includes("pdf")) filename += ".pdf";
        else if (contentType?.includes("zip")) filename += ".zip";
      }

      // Store the result for potential indexing
      setLastProcessedResult({ blob, filename });

      // Download the result
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      // If "Index after processing" is enabled and result is a PDF, upload to index
      if (indexAfterProcess && filename.endsWith(".pdf")) {
        setIsIndexing(true);
        try {
          const indexFormData = new FormData();
          const file = new File([blob], filename, { type: "application/pdf" });
          indexFormData.append("file", file);

          const indexResponse = await api.fetchWithAuth("/api/v1/upload/single", {
            method: "POST",
            body: indexFormData,
          });

          if (indexResponse.ok) {
            toast.success(`${filename} has been added to your document index!`);
          } else {
            toast.error("Failed to index the processed document.");
          }
        } catch (indexError) {
          console.error("Indexing error:", indexError);
          toast.error("Failed to index the processed document.");
        } finally {
          setIsIndexing(false);
        }
      }

      toast.success(`Your ${activeTool} operation completed successfully.`);
    } catch (error) {
      console.error("Processing error:", error);
      toast.error(error instanceof Error ? error.message : "Processing failed");
    } finally {
      setIsProcessing(false);
    }
  };

  // Tool configurations
  const tools = [
    { id: "merge", label: "Merge", icon: Merge, description: "Combine multiple PDFs" },
    { id: "split", label: "Split", icon: Split, description: "Split by page ranges" },
    { id: "extract", label: "Extract", icon: FileText, description: "Extract specific pages" },
    { id: "rotate", label: "Rotate", icon: RotateCw, description: "Rotate PDF pages" },
    { id: "compress", label: "Compress", icon: Minimize2, description: "Reduce file size" },
    { id: "to-images", label: "To Images", icon: ImageIcon, description: "Convert to images" },
    { id: "from-images", label: "From Images", icon: FileText, description: "Create PDF from images" },
    { id: "metadata", label: "Metadata", icon: FileEdit, description: "Edit PDF properties" },
    { id: "watermark", label: "Watermark", icon: Droplet, description: "Add text watermark" },
    { id: "rearrange", label: "Rearrange", icon: ArrowUpDown, description: "Reorder pages" },
  ];

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">PDF Tools</h1>
        <p className="text-muted-foreground">
          Powerful tools for working with PDF documents
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Tool Selection */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Tools</CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            <div className="space-y-1">
              {tools.map((tool) => (
                <button
                  key={tool.id}
                  onClick={() => setActiveTool(tool.id as ToolType)}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
                    activeTool === tool.id
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-muted"
                  }`}
                >
                  <tool.icon className="h-4 w-4" />
                  <div>
                    <div className="font-medium text-sm">{tool.label}</div>
                    <div
                      className={`text-xs ${
                        activeTool === tool.id
                          ? "text-primary-foreground/70"
                          : "text-muted-foreground"
                      }`}
                    >
                      {tool.description}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-6">
          {/* File Upload */}
          <Card>
            <CardHeader>
              <CardTitle>
                {activeTool === "from-images" ? "Upload Images" : "Upload PDF(s)"}
              </CardTitle>
              <CardDescription>
                {activeTool === "merge"
                  ? "Select multiple PDFs to merge"
                  : activeTool === "from-images"
                  ? "Select images to convert to PDF"
                  : "Select a PDF to process"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary/50 transition-colors"
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
              >
                <input
                  type="file"
                  accept={
                    activeTool === "from-images"
                      ? "image/*"
                      : "application/pdf"
                  }
                  multiple={activeTool === "merge" || activeTool === "from-images"}
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center gap-2"
                >
                  <Upload className="h-10 w-10 text-muted-foreground" />
                  <span className="text-lg font-medium">
                    Drop files here or click to upload
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {activeTool === "from-images"
                      ? "PNG, JPG, JPEG, GIF, WebP"
                      : "PDF files only"}
                  </span>
                </label>
              </div>

              {/* File List */}
              {files.length > 0 && (
                <div className="mt-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">
                      {files.length} file(s) selected
                    </span>
                    <Button variant="ghost" size="sm" onClick={clearFiles}>
                      <Trash2 className="h-4 w-4 mr-1" />
                      Clear all
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {files.map((file, index) => (
                      <div
                        key={file.id}
                        className="flex items-center gap-3 p-3 bg-muted rounded-lg"
                      >
                        {activeTool === "merge" && (
                          <GripVertical className="h-4 w-4 text-muted-foreground cursor-move" />
                        )}
                        <FileText className="h-5 w-5 text-red-500" />
                        <div className="flex-1 min-w-0">
                          <div className="font-medium truncate">{file.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {formatFileSize(file.size)}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => removeFile(file.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Tool Options */}
          <Card>
            <CardHeader>
              <CardTitle>Options</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {activeTool === "split" && (
                <div className="space-y-2">
                  <Label htmlFor="ranges">Page Ranges</Label>
                  <Input
                    id="ranges"
                    placeholder="e.g., 1-5, 6-10, 11-"
                    value={splitRanges}
                    onChange={(e) => setSplitRanges(e.target.value)}
                  />
                  <p className="text-sm text-muted-foreground">
                    Separate ranges with commas. Use "11-" for pages 11 to end.
                  </p>
                </div>
              )}

              {activeTool === "extract" && (
                <div className="space-y-2">
                  <Label htmlFor="pages">Pages to Extract</Label>
                  <Input
                    id="pages"
                    placeholder="e.g., 1, 3, 5, 7-10"
                    value={extractPages}
                    onChange={(e) => setExtractPages(e.target.value)}
                  />
                </div>
              )}

              {activeTool === "rotate" && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Rotation Angle</Label>
                    <Select
                      value={rotationAngle.toString()}
                      onValueChange={(v) => setRotationAngle(parseInt(v))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="90">90° Clockwise</SelectItem>
                        <SelectItem value="180">180°</SelectItem>
                        <SelectItem value="270">90° Counter-clockwise</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="rotate-pages">Pages (optional)</Label>
                    <Input
                      id="rotate-pages"
                      placeholder="Leave empty for all pages"
                      value={rotatePages}
                      onChange={(e) => setRotatePages(e.target.value)}
                    />
                  </div>
                </div>
              )}

              {activeTool === "compress" && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Image Quality: {compressionQuality}%</Label>
                    <Slider
                      value={[compressionQuality]}
                      onValueChange={(v) => setCompressionQuality(v[0])}
                      min={10}
                      max={100}
                      step={5}
                    />
                    <p className="text-sm text-muted-foreground">
                      Lower quality = smaller file size
                    </p>
                  </div>
                </div>
              )}

              {activeTool === "to-images" && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>DPI: {imageDpi}</Label>
                    <Slider
                      value={[imageDpi]}
                      onValueChange={(v) => setImageDpi(v[0])}
                      min={72}
                      max={600}
                      step={10}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Format</Label>
                    <Select value={imageFormat} onValueChange={setImageFormat}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="png">PNG</SelectItem>
                        <SelectItem value="jpeg">JPEG</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {activeTool === "metadata" && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="meta-title">Title</Label>
                    <Input
                      id="meta-title"
                      value={metadata.title}
                      onChange={(e) =>
                        setMetadata((prev) => ({ ...prev, title: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="meta-author">Author</Label>
                    <Input
                      id="meta-author"
                      value={metadata.author}
                      onChange={(e) =>
                        setMetadata((prev) => ({ ...prev, author: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="meta-subject">Subject</Label>
                    <Input
                      id="meta-subject"
                      value={metadata.subject}
                      onChange={(e) =>
                        setMetadata((prev) => ({ ...prev, subject: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="meta-keywords">Keywords</Label>
                    <Input
                      id="meta-keywords"
                      value={metadata.keywords}
                      onChange={(e) =>
                        setMetadata((prev) => ({
                          ...prev,
                          keywords: e.target.value,
                        }))
                      }
                    />
                  </div>
                </div>
              )}

              {activeTool === "watermark" && (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="watermark-text">Watermark Text</Label>
                    <Input
                      id="watermark-text"
                      value={watermarkText}
                      onChange={(e) => setWatermarkText(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Opacity: {Math.round(watermarkOpacity * 100)}%</Label>
                    <Slider
                      value={[watermarkOpacity * 100]}
                      onValueChange={(v) => setWatermarkOpacity(v[0] / 100)}
                      min={10}
                      max={100}
                      step={5}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Angle: {watermarkAngle}°</Label>
                    <Slider
                      value={[watermarkAngle]}
                      onValueChange={(v) => setWatermarkAngle(v[0])}
                      min={0}
                      max={90}
                      step={5}
                    />
                  </div>
                </div>
              )}

              {activeTool === "rearrange" && (
                <div className="space-y-2">
                  <Label htmlFor="page-order">New Page Order</Label>
                  <Input
                    id="page-order"
                    placeholder="e.g., 3, 1, 2, 4"
                    value={pageOrder}
                    onChange={(e) => setPageOrder(e.target.value)}
                  />
                  <p className="text-sm text-muted-foreground">
                    Enter page numbers in the desired order. Pages can be
                    duplicated or omitted.
                  </p>
                </div>
              )}

              {activeTool === "merge" && files.length > 1 && (
                <p className="text-sm text-muted-foreground">
                  Drag files to reorder them. PDFs will be merged in the order
                  shown.
                </p>
              )}

              {activeTool === "from-images" && (
                <p className="text-sm text-muted-foreground">
                  Images will be converted to PDF in the order shown.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Index Option & Process Button */}
          <div className="space-y-3">
            {/* Show index option only for tools that produce PDFs */}
            {!["to-images"].includes(activeTool) && (
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="index-after"
                  checked={indexAfterProcess}
                  onCheckedChange={(checked) => setIndexAfterProcess(checked === true)}
                />
                <label
                  htmlFor="index-after"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 flex items-center gap-2"
                >
                  <Database className="h-4 w-4" />
                  Also add to document index
                </label>
              </div>
            )}

            <Button
              size="lg"
              className="w-full"
              onClick={processFiles}
              disabled={files.length === 0 || isProcessing || isIndexing}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Processing...
                </>
              ) : isIndexing ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Indexing...
                </>
              ) : (
                <>
                  <Download className="h-5 w-5 mr-2" />
                  {indexAfterProcess && !["to-images"].includes(activeTool)
                    ? "Process, Download & Index"
                    : "Process & Download"}
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
