"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import {
  PenTool,
  FileText,
  Presentation,
  FileSpreadsheet,
  ChevronRight,
  ChevronLeft,
  Check,
  Loader2,
  Sparkles,
  List,
  Edit3,
  Download,
  RefreshCw,
  Plus,
  Trash2,
  GripVertical,
  Image,
  AlertCircle,
  Filter,
  BookOpen,
  Palette,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  useGenerationJobs,
  useGenerationJob,
  useOutputFormats,
  useCreateGenerationJob,
  useGenerateOutline,
  useApproveOutline,
  useGenerateContent,
  useDownloadGeneratedDocument,
  useCancelGenerationJob,
  useCollections,
  useGetThemes,
  useSuggestTheme,
} from "@/lib/api";
import type { ThemeInfo, StyleGuide } from "@/lib/api";
import { toast } from "sonner";
import { DocumentFilterPanel } from "@/components/chat/document-filter-panel";
import { FolderSelector } from "@/components/folder-selector";

type Step = "format" | "topic" | "outline" | "content" | "download";
type OutputFormat = "docx" | "pptx" | "pdf" | "markdown" | "html" | "txt" | "xlsx";

const formatOptions: { id: OutputFormat; name: string; icon: React.ElementType; description: string }[] = [
  { id: "docx", name: "Word Document", icon: FileText, description: "Microsoft Word format (.docx)" },
  { id: "pptx", name: "PowerPoint", icon: Presentation, description: "Presentation slides (.pptx)" },
  { id: "pdf", name: "PDF", icon: FileText, description: "Portable Document Format (.pdf)" },
  { id: "xlsx", name: "Excel Spreadsheet", icon: FileSpreadsheet, description: "Structured data format (.xlsx)" },
  { id: "markdown", name: "Markdown", icon: FileText, description: "Plain text with formatting (.md)" },
  { id: "html", name: "HTML", icon: FileText, description: "Web page format (.html)" },
  { id: "txt", name: "Plain Text", icon: FileText, description: "Simple text file (.txt)" },
];

const steps: { id: Step; name: string; description: string }[] = [
  { id: "format", name: "Format", description: "Choose output format" },
  { id: "topic", name: "Topic", description: "Describe your document" },
  { id: "outline", name: "Outline", description: "Review & edit structure" },
  { id: "content", name: "Generate", description: "AI creates content" },
  { id: "download", name: "Download", description: "Get your document" },
];

interface OutlineSection {
  id: string;
  title: string;
  description: string;
}

export default function CreatePage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [currentStep, setCurrentStep] = useState<Step>("format");
  const [selectedFormat, setSelectedFormat] = useState<OutputFormat | null>(null);
  const [topic, setTopic] = useState("");
  const [context, setContext] = useState("");
  const [tone, setTone] = useState("professional");
  const [includeImages, setIncludeImages] = useState(false); // Disabled by default
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [includeSubfolders, setIncludeSubfolders] = useState(true);
  const [showFilters, setShowFilters] = useState(false);
  const [outline, setOutline] = useState<OutlineSection[]>([]);
  const [documentTitle, setDocumentTitle] = useState("");
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState<number | null>(null); // null = auto mode
  const [showCustomPageCount, setShowCustomPageCount] = useState(false);
  const [customPageCountValue, setCustomPageCountValue] = useState<number>(5);
  const [selectedTheme, setSelectedTheme] = useState<string>("business");
  const [themeSuggestionReason, setThemeSuggestionReason] = useState<string | null>(null);
  const [isLoadingThemeSuggestion, setIsLoadingThemeSuggestion] = useState(false);
  const [themeManuallyChanged, setThemeManuallyChanged] = useState(false);
  const [includeSources, setIncludeSources] = useState(true);

  // Style learning from existing documents
  const [useExistingDocs, setUseExistingDocs] = useState(false);
  const [styleCollections, setStyleCollections] = useState<string[]>([]);
  const [styleFolderId, setStyleFolderId] = useState<string | null>(null);
  const [includeStyleSubfolders, setIncludeStyleSubfolders] = useState(true);

  // Queries - only fetch when authenticated
  const { data: job, isLoading: jobLoading } = useGenerationJob(currentJobId || "");
  const { data: recentJobs } = useGenerationJobs();
  const { data: collectionsData, isLoading: isLoadingCollections, refetch: refetchCollections } = useCollections({ enabled: isAuthenticated });
  const { data: themesData } = useGetThemes();
  const suggestTheme = useSuggestTheme();

  // Mutations
  const createJob = useCreateGenerationJob();
  const generateOutline = useGenerateOutline();
  const approveOutline = useApproveOutline();
  const generateContent = useGenerateContent();
  const downloadDocument = useDownloadGeneratedDocument();
  const cancelJob = useCancelGenerationJob();

  const currentStepIndex = steps.findIndex((s) => s.id === currentStep);

  // Auto-suggest theme when topic and context are provided (debounced)
  useEffect(() => {
    if (topic.length > 10 && currentStep === "topic") {
      const timer = setTimeout(async () => {
        setIsLoadingThemeSuggestion(true);
        try {
          const suggestion = await suggestTheme.mutateAsync({
            title: topic,
            description: context || topic,
          });
          setSelectedTheme(suggestion.recommended);
          setThemeSuggestionReason(suggestion.reason);
        } catch {
          // Silently fail - theme suggestion is optional
        }
        setIsLoadingThemeSuggestion(false);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [topic, context, currentStep]);

  // Re-suggest theme after outline is generated (has more context from section titles/descriptions)
  useEffect(() => {
    if (job?.outline && currentStep === "outline" && outline.length > 0 && !themeManuallyChanged) {
      const outlineContext = outline.map(s => `${s.title}: ${s.description}`).join('\n');
      const timer = setTimeout(async () => {
        setIsLoadingThemeSuggestion(true);
        try {
          const suggestion = await suggestTheme.mutateAsync({
            title: job.title || topic,
            description: outlineContext,  // Use full outline for better context
          });
          // Only update if user hasn't manually changed theme
          if (!themeManuallyChanged) {
            setSelectedTheme(suggestion.recommended);
            setThemeSuggestionReason(suggestion.reason);
          }
        } catch {
          // Silently fail - theme suggestion is optional
        }
        setIsLoadingThemeSuggestion(false);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [job?.outline, currentStep, outline, themeManuallyChanged]);

  const handleNext = async () => {
    switch (currentStep) {
      case "format":
        if (selectedFormat) setCurrentStep("topic");
        break;
      case "topic":
        if (topic) {
          try {
            const newJob = await createJob.mutateAsync({
              title: topic,
              description: context || topic,
              output_format: selectedFormat!,
              include_images: includeImages,
              collection_filters: selectedCollections.length > 0 ? selectedCollections : undefined,
              folder_id: selectedFolderId || undefined,
              include_subfolders: includeSubfolders,
              page_count: pageCount, // null = auto mode
              theme: selectedTheme,
              include_sources: includeSources,
              // Style learning from existing documents
              use_existing_docs: useExistingDocs,
              style_collection_filters: useExistingDocs && styleCollections.length > 0 ? styleCollections : undefined,
              style_folder_id: useExistingDocs && styleFolderId ? styleFolderId : undefined,
              include_style_subfolders: useExistingDocs ? includeStyleSubfolders : undefined,
            });
            setCurrentJobId(newJob.id);
            await generateOutline.mutateAsync({
              jobId: newJob.id,
              numSections: pageCount ?? undefined  // Pass user's selection, undefined for auto
            });
            setCurrentStep("outline");
          } catch (error) {
            console.error("Failed to create job:", error);
          }
        }
        break;
      case "outline":
        if (currentJobId) {
          try {
            const approvedJob = await approveOutline.mutateAsync({
              jobId: currentJobId,
              modifications: {
                title: documentTitle || undefined,
                sections: outline.length > 0 ? outline : undefined,
                tone,
                theme: selectedTheme,  // Pass theme (allows changing after outline)
              },
            });
            // Only generate content if the job is in the correct state
            if (approvedJob.status === "outline_approved") {
              await generateContent.mutateAsync(currentJobId);
            }
            setCurrentStep("content");
          } catch (error: any) {
            console.error("Failed to approve outline:", error);
            const errorDetail = error?.detail || error?.message || "";
            // If the job is already completed, just move to download step
            if (errorDetail.includes("COMPLETED")) {
              toast.info("Document already generated", {
                description: "Moving to download step.",
              });
              setCurrentStep("download");
            } else if (errorDetail.includes("OUTLINE_APPROVED") || errorDetail.includes("GENERATING")) {
              // Job is already approved or generating, move to content step
              toast.info("Generation in progress", {
                description: "Please wait for content generation to complete.",
              });
              setCurrentStep("content");
            } else {
              toast.error("Failed to approve outline", {
                description: errorDetail || "An error occurred",
              });
            }
          }
        }
        break;
      case "content":
        setCurrentStep("download");
        break;
    }
  };

  const handleBack = () => {
    const prevIndex = currentStepIndex - 1;
    if (prevIndex >= 0) {
      setCurrentStep(steps[prevIndex].id);
    }
  };

  const handleDownload = async () => {
    if (!currentJobId) return;
    try {
      const blob = await downloadDocument.mutateAsync(currentJobId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      // Use the original job title (from creation), not the outline title which may have been modified
      a.download = `${job?.title || documentTitle || "document"}.${selectedFormat}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to download:", error);
    }
  };

  const handleAddSection = () => {
    setOutline([
      ...outline,
      { id: crypto.randomUUID(), title: "New Section", description: "" },
    ]);
  };

  const handleRemoveSection = (id: string) => {
    setOutline(outline.filter((s) => s.id !== id));
  };

  const handleUpdateSection = (id: string, field: "title" | "description", value: string) => {
    setOutline(outline.map((s) => (s.id === id ? { ...s, [field]: value } : s)));
  };

  const handleStartNew = () => {
    setCurrentStep("format");
    setSelectedFormat(null);
    setTopic("");
    setContext("");
    setOutline([]);
    setDocumentTitle("");
    setCurrentJobId(null);
  };

  // Update outline from job data
  if (job?.outline && outline.length === 0 && currentStep === "outline") {
    setDocumentTitle(job.outline.title || "");
    setOutline(
      job.outline.sections?.map((s: { title: string; description: string }, i: number) => ({
        id: crypto.randomUUID(),
        title: s.title,
        description: s.description,
      })) || []
    );
  }

  const isLoading =
    createJob.isPending ||
    generateOutline.isPending ||
    approveOutline.isPending ||
    generateContent.isPending;

  const isGenerating = job?.status === "generating" || job?.status === "processing";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Document Creator</h1>
          <p className="text-muted-foreground">
            Generate documents from your knowledge base with AI
          </p>
        </div>
        {currentJobId && (
          <Button onClick={handleStartNew} variant="outline">
            <Plus className="h-4 w-4 mr-2" />
            New Document
          </Button>
        )}
      </div>

      {/* Progress Steps */}
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center">
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors ${
                  index < currentStepIndex
                    ? "bg-primary border-primary text-primary-foreground"
                    : index === currentStepIndex
                    ? "border-primary text-primary"
                    : "border-muted text-muted-foreground"
                }`}
              >
                {index < currentStepIndex ? (
                  <Check className="h-5 w-5" />
                ) : (
                  <span>{index + 1}</span>
                )}
              </div>
              <div className="mt-2 text-center">
                <p className="text-sm font-medium">{step.name}</p>
                <p className="text-xs text-muted-foreground hidden sm:block">
                  {step.description}
                </p>
              </div>
            </div>
            {index < steps.length - 1 && (
              <div
                className={`w-12 sm:w-24 h-0.5 mx-2 ${
                  index < currentStepIndex ? "bg-primary" : "bg-muted"
                }`}
              />
            )}
          </div>
        ))}
      </div>

      {/* Step Content */}
      <Card className="min-h-[400px]">
        <CardContent className="pt-6">
          {/* Step 1: Format Selection */}
          {currentStep === "format" && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold">Choose Output Format</h2>
              <p className="text-muted-foreground">
                Select the format for your generated document
              </p>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {formatOptions.map((format) => (
                  <Card
                    key={format.id}
                    className={`cursor-pointer transition-all ${
                      selectedFormat === format.id
                        ? "border-primary ring-2 ring-primary/20"
                        : "hover:border-primary/50"
                    }`}
                    onClick={() => setSelectedFormat(format.id)}
                  >
                    <CardContent className="pt-6">
                      <div className="flex items-center gap-3">
                        <format.icon className="h-8 w-8 text-primary" />
                        <div>
                          <p className="font-medium">{format.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {format.description}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Step 2: Topic Input */}
          {currentStep === "topic" && (
            <div className="space-y-6 max-w-2xl">
              <h2 className="text-xl font-semibold">Describe Your Document</h2>
              <p className="text-muted-foreground">
                Tell us what you want to create. The AI will use your knowledge base to generate content.
              </p>

              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Topic / Title</label>
                  <Input
                    placeholder="e.g., Quarterly Sales Report, Project Proposal, Technical Documentation"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Additional Context (optional)</label>
                  <textarea
                    placeholder="Add any specific requirements, focus areas, or instructions..."
                    value={context}
                    onChange={(e) => setContext(e.target.value)}
                    className="w-full min-h-[100px] p-3 rounded-md border bg-background resize-none"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Tone</label>
                  <select
                    value={tone}
                    onChange={(e) => setTone(e.target.value)}
                    className="w-full h-10 px-3 rounded-md border bg-background"
                  >
                    <option value="professional">Professional</option>
                    <option value="casual">Casual</option>
                    <option value="academic">Academic</option>
                    <option value="technical">Technical</option>
                    <option value="creative">Creative</option>
                  </select>
                </div>

                {/* Page/Slide Count Selector */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    Number of {selectedFormat === "pptx" ? "Slides" : "Pages/Sections"}
                  </label>
                  {showCustomPageCount ? (
                    <div className="flex gap-2">
                      <Input
                        type="number"
                        min={1}
                        max={20}
                        value={customPageCountValue}
                        onChange={(e) => setCustomPageCountValue(Math.min(20, Math.max(1, parseInt(e.target.value) || 1)))}
                        className="w-24"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setPageCount(customPageCountValue);
                          setShowCustomPageCount(false);
                        }}
                      >
                        Set
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowCustomPageCount(false)}
                      >
                        Cancel
                      </Button>
                    </div>
                  ) : (
                    <select
                      value={pageCount === null ? "auto" : pageCount.toString()}
                      onChange={(e) => {
                        const value = e.target.value;
                        if (value === "custom") {
                          setShowCustomPageCount(true);
                        } else if (value === "auto") {
                          setPageCount(null);
                        } else {
                          setPageCount(parseInt(value));
                        }
                      }}
                      className="w-full h-10 px-3 rounded-md border bg-background"
                    >
                      <option value="auto">✨ Auto (AI decides based on content)</option>
                      <option value="3">3 {selectedFormat === "pptx" ? "slides" : "sections"}</option>
                      <option value="5">5 {selectedFormat === "pptx" ? "slides" : "sections"}</option>
                      <option value="7">7 {selectedFormat === "pptx" ? "slides" : "sections"}</option>
                      <option value="10">10 {selectedFormat === "pptx" ? "slides" : "sections"}</option>
                      <option value="12">12 {selectedFormat === "pptx" ? "slides" : "sections"}</option>
                      <option value="15">15 {selectedFormat === "pptx" ? "slides" : "sections"}</option>
                      <option value="20">20 {selectedFormat === "pptx" ? "slides" : "sections"}</option>
                      <option value="custom">⚙️ Custom (1-20)...</option>
                    </select>
                  )}
                  <p className="text-xs text-muted-foreground">
                    {pageCount === null
                      ? "AI will determine the optimal length based on topic complexity"
                      : `Document will have exactly ${pageCount} content ${selectedFormat === "pptx" ? "slides" : "sections"}`}
                  </p>
                </div>

                {/* Source Filter Toggle */}
                <div className="space-y-3">
                  <div
                    className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                      showFilters
                        ? "bg-primary/5 border-primary/30"
                        : "bg-muted/30 border-border hover:bg-muted/50"
                    }`}
                    onClick={() => setShowFilters(!showFilters)}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${showFilters ? "bg-primary/10" : "bg-muted"}`}>
                        <Filter className={`h-4 w-4 ${showFilters ? "text-primary" : "text-muted-foreground"}`} />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Filter Source Documents</p>
                        <p className="text-xs text-muted-foreground">
                          Select collections and folders to use as reference
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {(selectedCollections.length > 0 || selectedFolderId) && (
                        <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded">
                          {selectedCollections.length > 0 && `${selectedCollections.length} collections`}
                          {selectedCollections.length > 0 && selectedFolderId && " + "}
                          {selectedFolderId && "folder"}
                        </span>
                      )}
                      <ChevronRight className={`h-4 w-4 transition-transform ${showFilters ? "rotate-90" : ""}`} />
                    </div>
                  </div>

                  {/* Expanded Filter Panel */}
                  {showFilters && (
                    <div className="space-y-3 pl-2 border-l-2 border-primary/20">
                      {/* Collections Filter */}
                      <DocumentFilterPanel
                        collections={collectionsData?.collections || []}
                        selectedCollections={selectedCollections}
                        onCollectionsChange={setSelectedCollections}
                        totalDocuments={collectionsData?.total_documents}
                        isLoading={isLoadingCollections}
                        onRefresh={() => refetchCollections()}
                      />

                      {/* Folder Filter */}
                      <div className="p-3 rounded-lg border bg-card">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-sm font-medium">Folder Scope</span>
                        </div>
                        <FolderSelector
                          value={selectedFolderId}
                          onChange={setSelectedFolderId}
                          includeSubfolders={includeSubfolders}
                          onIncludeSubfoldersChange={setIncludeSubfolders}
                          showSubfoldersToggle={true}
                          placeholder="All folders"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Image Generation Toggle */}
                <div className="space-y-2 pt-2">
                  <div
                    className={`flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-colors ${
                      includeImages
                        ? "bg-primary/5 border-primary/30"
                        : "bg-muted/30 border-border hover:bg-muted/50"
                    }`}
                    onClick={() => setIncludeImages(!includeImages)}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${includeImages ? "bg-primary/10" : "bg-muted"}`}>
                        <Image className={`h-5 w-5 ${includeImages ? "text-primary" : "text-muted-foreground"}`} />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Include AI-Generated Images</p>
                        <p className="text-xs text-muted-foreground">
                          Add relevant images to each section using local AI
                        </p>
                      </div>
                    </div>
                    <div
                      className={`w-11 h-6 rounded-full transition-colors flex items-center ${
                        includeImages ? "bg-primary justify-end" : "bg-muted-foreground/30 justify-start"
                      }`}
                    >
                      <div className="w-5 h-5 bg-white rounded-full shadow-sm mx-0.5" />
                    </div>
                  </div>
                  {includeImages && (
                    <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/50">
                      <AlertCircle className="h-4 w-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                      <p className="text-xs text-amber-700 dark:text-amber-300">
                        Image generation requires Ollama with a Stable Diffusion model installed locally.
                        Generation may take longer and images will be placeholders if unavailable.
                      </p>
                    </div>
                  )}
                </div>

                {/* Learn from Existing Documents Toggle */}
                <div className="space-y-2">
                  <div
                    className={`flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-colors ${
                      useExistingDocs
                        ? "bg-primary/5 border-primary/30"
                        : "bg-muted/30 border-border hover:bg-muted/50"
                    }`}
                    onClick={() => setUseExistingDocs(!useExistingDocs)}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${useExistingDocs ? "bg-primary/10" : "bg-muted"}`}>
                        <BookOpen className={`h-5 w-5 ${useExistingDocs ? "text-primary" : "text-muted-foreground"}`} />
                      </div>
                      <div>
                        <p className="text-sm font-medium">Learn from Existing Documents</p>
                        <p className="text-xs text-muted-foreground">
                          Use your documents to guide style, tone, and content structure
                        </p>
                      </div>
                    </div>
                    <div
                      className={`w-11 h-6 rounded-full transition-colors flex items-center ${
                        useExistingDocs ? "bg-primary justify-end" : "bg-muted-foreground/30 justify-start"
                      }`}
                    >
                      <div className="w-5 h-5 bg-white rounded-full shadow-sm mx-0.5" />
                    </div>
                  </div>
                  {useExistingDocs && (
                    <div className="pl-4 space-y-3 mt-3">
                      <div className="text-xs text-muted-foreground mb-2">
                        <Sparkles className="h-3 w-3 inline mr-1" />
                        AI will analyze selected documents to match their writing style
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Select source collections/folders</label>
                        <p className="text-xs text-muted-foreground">
                          The AI will learn formatting, tone, and style from these documents
                        </p>
                        {/* Collection selector for style learning */}
                        <div className="space-y-2">
                          {isLoadingCollections ? (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              Loading collections...
                            </div>
                          ) : (
                            <div className="flex flex-wrap gap-2">
                              {(collectionsData?.collections || []).map((collection: { name: string; document_count: number }) => (
                                <button
                                  key={collection.name}
                                  type="button"
                                  onClick={() => {
                                    if (styleCollections.includes(collection.name)) {
                                      setStyleCollections(styleCollections.filter(c => c !== collection.name));
                                    } else {
                                      setStyleCollections([...styleCollections, collection.name]);
                                    }
                                  }}
                                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                                    styleCollections.includes(collection.name)
                                      ? "bg-primary text-primary-foreground"
                                      : "bg-muted hover:bg-muted/80 text-muted-foreground"
                                  }`}
                                >
                                  {collection.name} ({collection.document_count})
                                </button>
                              ))}
                              {(collectionsData?.collections || []).length === 0 && (
                                <p className="text-xs text-muted-foreground">
                                  No collections available. Upload documents first.
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                        {styleCollections.length > 0 && (
                          <p className="text-xs text-primary">
                            {styleCollections.length} collection{styleCollections.length > 1 ? "s" : ""} selected for style learning
                          </p>
                        )}
                      </div>

                      {/* Include Sources Toggle - only shown when Learn from Existing Docs is enabled */}
                      <div
                        className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                          includeSources
                            ? "bg-primary/5 border-primary/30"
                            : "bg-muted/30 border-border hover:bg-muted/50"
                        }`}
                        onClick={() => setIncludeSources(!includeSources)}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${includeSources ? "bg-primary/10" : "bg-muted"}`}>
                            <FileText className={`h-4 w-4 ${includeSources ? "text-primary" : "text-muted-foreground"}`} />
                          </div>
                          <div>
                            <p className="text-sm font-medium">Include Sources & References</p>
                            <p className="text-xs text-muted-foreground">
                              Add a {selectedFormat === "pptx" ? "slide" : selectedFormat === "xlsx" ? "sheet" : "page"} listing referenced documents
                            </p>
                          </div>
                        </div>
                        <div
                          className={`w-10 h-5 rounded-full transition-colors flex items-center ${
                            includeSources ? "bg-primary justify-end" : "bg-muted-foreground/30 justify-start"
                          }`}
                        >
                          <div className="w-4 h-4 bg-white rounded-full shadow-sm mx-0.5" />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Theme Selector */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Document Theme</label>
                    {isLoadingThemeSuggestion && (
                      <span className="text-xs text-muted-foreground flex items-center gap-1">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Suggesting theme...
                      </span>
                    )}
                  </div>
                  {themeSuggestionReason && (
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      <Sparkles className="h-3 w-3" />
                      AI suggests: {themeSuggestionReason}
                    </p>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    {(themesData?.themes || [
                      { key: "business", name: "Business Professional", description: "Clean, corporate look", primary: "#1E3A5F", secondary: "#3D5A80", accent: "#E0E1DD", text: "#2D3A45" },
                      { key: "creative", name: "Creative & Bold", description: "Vibrant marketing style", primary: "#6B4C9A", secondary: "#9B6B9E", accent: "#F4E4BA", text: "#333333" },
                      { key: "modern", name: "Modern Minimal", description: "Sleek, contemporary design", primary: "#212529", secondary: "#495057", accent: "#00B4D8", text: "#212529" },
                      { key: "nature", name: "Nature & Organic", description: "Earthy, sustainable tones", primary: "#2D5016", secondary: "#5A7D3A", accent: "#F5F0E1", text: "#2D3A2E" },
                    ] as ThemeInfo[]).map((theme: ThemeInfo) => (
                      <div
                        key={theme.key}
                        onClick={() => {
                          setSelectedTheme(theme.key);
                          setThemeSuggestionReason(null);
                          setThemeManuallyChanged(true);  // User manually selected a theme
                        }}
                        className={`p-3 rounded-lg border cursor-pointer transition-all ${
                          selectedTheme === theme.key
                            ? "border-primary ring-2 ring-primary/20"
                            : "border-border hover:border-primary/50"
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          {/* Color swatches */}
                          <div
                            className="w-4 h-4 rounded-full"
                            style={{ backgroundColor: theme.primary }}
                          />
                          <div
                            className="w-4 h-4 rounded-full"
                            style={{ backgroundColor: theme.secondary }}
                          />
                          <div
                            className="w-4 h-4 rounded-full border"
                            style={{ backgroundColor: theme.accent }}
                          />
                        </div>
                        <p className="text-sm font-medium">{theme.name}</p>
                        <p className="text-xs text-muted-foreground">{theme.description}</p>
                        {themeSuggestionReason && selectedTheme === theme.key && (
                          <span className="text-xs text-primary mt-1 inline-block">✨ Recommended</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Outline Review */}
          {currentStep === "outline" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">Review & Edit Outline</h2>
                  <p className="text-muted-foreground">
                    Customize the structure before generating content
                  </p>
                </div>
                <Button onClick={handleAddSection} variant="outline" size="sm">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Section
                </Button>
              </div>

              <div className="space-y-4">
                {/* Style Analysis Display */}
                {(() => {
                  const styleGuide = job?.metadata?.style_guide as StyleGuide | undefined;
                  if (!styleGuide) return null;
                  return (
                    <div className="bg-muted/30 rounded-lg p-4 border">
                      <h4 className="text-sm font-medium flex items-center gap-2 mb-3">
                        <Palette className="h-4 w-4 text-primary" />
                        Style Analysis from Existing Documents
                      </h4>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Tone:</span>
                          <span className="font-medium capitalize">{styleGuide.tone}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Vocabulary:</span>
                          <span className="font-medium capitalize">{styleGuide.vocabulary_level}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Structure:</span>
                          <span className="font-medium capitalize">{styleGuide.structure_pattern?.replace(/-/g, ' ')}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Sentences:</span>
                          <span className="font-medium capitalize">{styleGuide.sentence_style?.replace(/-/g, ' ')}</span>
                        </div>
                      </div>
                      {styleGuide.source_documents && styleGuide.source_documents.length > 0 && (
                        <p className="text-xs text-muted-foreground mt-3 pt-3 border-t">
                          <Sparkles className="h-3 w-3 inline mr-1" />
                          Based on: {styleGuide.source_documents.slice(0, 3).join(', ')}
                          {styleGuide.source_documents.length > 3 && ` +${styleGuide.source_documents.length - 3} more`}
                        </p>
                      )}
                    </div>
                  );
                })()}

                <div className="space-y-2">
                  <label className="text-sm font-medium">Document Title</label>
                  <Input
                    value={documentTitle}
                    onChange={(e) => setDocumentTitle(e.target.value)}
                    placeholder="Enter document title"
                  />
                </div>

                <div className="space-y-3">
                  <label className="text-sm font-medium">Sections</label>
                  {outline.map((section, index) => (
                    <div
                      key={section.id}
                      className="flex items-start gap-3 p-4 rounded-lg border bg-muted/30"
                    >
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <GripVertical className="h-5 w-5 cursor-grab" />
                        <span className="text-sm font-medium">{index + 1}</span>
                      </div>
                      <div className="flex-1 space-y-2">
                        <Input
                          value={section.title}
                          onChange={(e) =>
                            handleUpdateSection(section.id, "title", e.target.value)
                          }
                          placeholder="Section title"
                          className="font-medium"
                        />
                        <Input
                          value={section.description}
                          onChange={(e) =>
                            handleUpdateSection(section.id, "description", e.target.value)
                          }
                          placeholder="Brief description of this section"
                          className="text-sm"
                        />
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleRemoveSection(section.id)}
                      >
                        <Trash2 className="h-4 w-4 text-muted-foreground" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Content Generation */}
          {currentStep === "content" && (
            <div className="flex flex-col items-center justify-center py-12 space-y-6">
              {isGenerating || jobLoading ? (
                <>
                  <div className="relative">
                    <Sparkles className="h-16 w-16 text-primary animate-pulse" />
                    <Loader2 className="h-8 w-8 text-primary animate-spin absolute -bottom-1 -right-1" />
                  </div>
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">Generating Your Document</h2>
                    <p className="text-muted-foreground mt-2">
                      AI is creating content based on your knowledge base...
                    </p>
                  </div>
                  <div className="w-full max-w-md">
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-primary rounded-full animate-progress" />
                    </div>
                    <p className="text-sm text-muted-foreground text-center mt-2">
                      {job?.status === "generating" ? "Generating sections..." : "Processing..."}
                    </p>
                  </div>
                </>
              ) : job?.status === "completed" ? (
                <>
                  <Check className="h-16 w-16 text-green-500" />
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">Document Ready!</h2>
                    <p className="text-muted-foreground mt-2">
                      Your document has been generated successfully
                    </p>
                  </div>
                </>
              ) : job?.status === "failed" ? (
                <>
                  <div className="text-center">
                    <h2 className="text-xl font-semibold text-red-500">Generation Failed</h2>
                    <p className="text-muted-foreground mt-2">
                      {job.error_message || "An error occurred during generation"}
                    </p>
                  </div>
                  <Button onClick={() => setCurrentStep("outline")} variant="outline">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Try Again
                  </Button>
                </>
              ) : null}
            </div>
          )}

          {/* Step 5: Download */}
          {currentStep === "download" && (
            <div className="flex flex-col items-center justify-center py-12 space-y-6">
              <div className="p-6 rounded-full bg-primary/10">
                <Download className="h-16 w-16 text-primary" />
              </div>
              <div className="text-center">
                <h2 className="text-xl font-semibold">Download Your Document</h2>
                <p className="text-muted-foreground mt-2">
                  {documentTitle || "Your document"} is ready to download
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  onClick={handleDownload}
                  size="lg"
                  disabled={downloadDocument.isPending}
                >
                  {downloadDocument.isPending ? (
                    <Loader2 className="h-5 w-5 animate-spin mr-2" />
                  ) : (
                    <Download className="h-5 w-5 mr-2" />
                  )}
                  Download {selectedFormat?.toUpperCase()}
                </Button>
                <Button onClick={handleStartNew} variant="outline" size="lg">
                  <Plus className="h-5 w-5 mr-2" />
                  Create Another
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Navigation Buttons */}
      {currentStep !== "download" && (
        <div className="flex justify-between">
          <Button
            onClick={handleBack}
            variant="outline"
            disabled={currentStepIndex === 0}
          >
            <ChevronLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Button
            onClick={handleNext}
            disabled={
              (currentStep === "format" && !selectedFormat) ||
              (currentStep === "topic" && !topic) ||
              (currentStep === "content" && isGenerating) ||
              isLoading
            }
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : currentStep === "content" && job?.status === "completed" ? (
              <>
                Continue
                <ChevronRight className="h-4 w-4 ml-2" />
              </>
            ) : (
              <>
                {currentStep === "topic" ? "Generate Outline" : "Next"}
                <ChevronRight className="h-4 w-4 ml-2" />
              </>
            )}
          </Button>
        </div>
      )}

      {/* Recent Jobs */}
      {recentJobs?.jobs && recentJobs.jobs.length > 0 && !currentJobId && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Recent Documents</CardTitle>
            <CardDescription>Your previously generated documents</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {recentJobs.jobs.slice(0, 5).map((j) => (
                <div
                  key={j.id}
                  className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 cursor-pointer"
                  onClick={() => {
                    setCurrentJobId(j.id);
                    setSelectedFormat(j.output_format as OutputFormat);
                    if (j.status === "completed") {
                      setCurrentStep("download");
                    }
                  }}
                >
                  <div className="flex items-center gap-3">
                    <FileText className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">{j.title}</p>
                      <p className="text-xs text-muted-foreground">
                        {j.output_format.toUpperCase()} - {new Date(j.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      j.status === "completed"
                        ? "bg-green-500/10 text-green-600"
                        : j.status === "failed"
                        ? "bg-red-500/10 text-red-600"
                        : "bg-yellow-500/10 text-yellow-600"
                    }`}
                  >
                    {j.status}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
