"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
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
  Image,
  AlertCircle,
  Filter,
  BookOpen,
  Palette,
  FolderOpen,
  Save,
  SpellCheck,
  Info,
  BarChart3,
  Eye,
  Cog,
  CheckCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { calculateOptimizedTemperature } from "@/lib/utils";
import { TemplateSelector, SaveTemplateDialog, BuiltInTemplateSelector } from "@/components/generation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  api,
  useGenerationJobs,
  useGenerationJob,
  useOutputFormats,
  useCreateGenerationJob,
  useGenerateOutline,
  useApproveOutline,
  useApproveSectionPlans,
  useGenerateContent,
  useDownloadGeneratedDocument,
  useCancelGenerationJob,
  useCollections,
  useGetThemes,
  useSuggestTheme,
  useSpellCheckJob,
  useLLMProviders,
} from "@/lib/api";
import type { ThemeInfo, StyleGuide, GenerationTemplate, TemplateSettings, GenerationSection, SpellCheckResponse, DocumentTemplateMetadata } from "@/lib/api";
import { toast } from "sonner";
import { DocumentFilterPanel } from "@/components/chat/document-filter-panel";
import { FolderSelector } from "@/components/folder-selector";
import { SectionFeedbackDialog } from "@/components/generation";
import { QueryEnhancementToggle } from "@/components/chat/query-enhancement-toggle";
import { MoodboardCanvas } from "@/components/moodboard/MoodboardCanvas";

type Step = "format" | "topic" | "outline" | "content" | "download";
type OutputFormat = "docx" | "pptx" | "pdf" | "markdown" | "html" | "txt" | "xlsx" | "moodboard";

const formatOptions: { id: OutputFormat; name: string; icon: React.ElementType; description: string }[] = [
  { id: "docx", name: "Word Document", icon: FileText, description: "Microsoft Word format (.docx)" },
  { id: "pptx", name: "PowerPoint", icon: Presentation, description: "Presentation slides (.pptx)" },
  { id: "pdf", name: "PDF", icon: FileText, description: "Portable Document Format (.pdf)" },
  { id: "xlsx", name: "Excel Spreadsheet", icon: FileSpreadsheet, description: "Structured data format (.xlsx)" },
  { id: "markdown", name: "Markdown", icon: FileText, description: "Plain text with formatting (.md)" },
  { id: "html", name: "HTML", icon: FileText, description: "Web page format (.html)" },
  { id: "txt", name: "Plain Text", icon: FileText, description: "Simple text file (.txt)" },
  { id: "moodboard", name: "Mood Board", icon: Palette, description: "AI-powered visual inspiration board" },
];

// Language options for document generation
const LANGUAGES = [
  { code: "auto", name: "Auto-detect (match source language)" },
  { code: "en", name: "English" },
  { code: "de", name: "German (Deutsch)" },
  { code: "es", name: "Spanish (Español)" },
  { code: "fr", name: "French (Français)" },
  { code: "it", name: "Italian (Italiano)" },
  { code: "pt", name: "Portuguese (Português)" },
  { code: "nl", name: "Dutch (Nederlands)" },
  { code: "pl", name: "Polish (Polski)" },
  { code: "ru", name: "Russian (Русский)" },
  { code: "zh", name: "Chinese (中文)" },
  { code: "ja", name: "Japanese (日本語)" },
  { code: "ko", name: "Korean (한국어)" },
  { code: "ar", name: "Arabic (العربية)" },
  { code: "hi", name: "Hindi (हिन्दी)" },
];

const steps: { id: Step; name: string; description: string }[] = [
  { id: "format", name: "Format", description: "Choose output format" },
  { id: "topic", name: "Topic", description: "Describe your document" },
  { id: "outline", name: "Outline", description: "Review & edit sections" },
  { id: "content", name: "Generate", description: "AI creates content" },
  { id: "download", name: "Download", description: "Get your document" },
];

interface OutlineSection {
  id: string;
  title: string;
  description: string;
  approved: boolean;
}

function CreatePageContent() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";
  const searchParams = useSearchParams();

  const [currentStep, setCurrentStep] = useState<Step>("format");
  const [selectedFormat, setSelectedFormat] = useState<OutputFormat | null>(null);
  const [topic, setTopic] = useState("");
  const [context, setContext] = useState("");
  const [tone, setTone] = useState("professional");
  const [outputLanguage, setOutputLanguage] = useState("auto");
  const [includeImages, setIncludeImages] = useState(false); // Disabled by default
  const [autoCharts, setAutoCharts] = useState(false); // Auto-generate charts/tables from data
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

  // Enhanced theming - font, layout, animations
  const [selectedFontFamily, setSelectedFontFamily] = useState<string>("modern");
  const [selectedLayout, setSelectedLayout] = useState<string>("standard");
  const [enableAnimations, setEnableAnimations] = useState<boolean>(false);
  const [animationSpeed, setAnimationSpeed] = useState<'very_slow' | 'slow' | 'med' | 'fast' | 'very_fast'>('med');
  const [useCustomDuration, setUseCustomDuration] = useState<boolean>(false);
  const [customDuration, setCustomDuration] = useState<number>(750);
  const [enableQualityReview, setEnableQualityReview] = useState<boolean>(false);
  // AI Proofreading with CriticAgent
  const [enableCriticReview, setEnableCriticReview] = useState<boolean>(false);
  const [qualityThreshold, setQualityThreshold] = useState<number>(0.7);
  const [fixStyling, setFixStyling] = useState<boolean>(true);
  const [fixIncomplete, setFixIncomplete] = useState<boolean>(true);
  // Include AI explanations in notes (PPTX speaker notes)
  const [includeNotesExplanation, setIncludeNotesExplanation] = useState<boolean>(false);
  // Query Enhancement - controls query expansion + HyDE for source search
  const [enhanceQuery, setEnhanceQuery] = useState<boolean | null>(null);

  // Vision Analysis (PPTX only) - per-document overrides for template analysis
  const [enableTemplateVisionAnalysis, setEnableTemplateVisionAnalysis] = useState<boolean | null>(null);
  const [templateVisionModel, setTemplateVisionModel] = useState<string>("auto");
  const [enableVisionReview, setEnableVisionReview] = useState<boolean | null>(null);
  const [visionReviewModel, setVisionReviewModel] = useState<string>("auto");

  // Advanced LLM Settings - Phase 15 Optimization (Optional per-document overrides)
  const [overrideTemperature, setOverrideTemperature] = useState<number | null>(null);

  const [availableFonts, setAvailableFonts] = useState<Record<string, any>>({});
  const [availableLayouts, setAvailableLayouts] = useState<Record<string, any>>({});

  // Custom colors - override theme colors
  const [useCustomColors, setUseCustomColors] = useState(false);
  const [customPrimaryColor, setCustomPrimaryColor] = useState<string>("#3D5A80");
  const [customSecondaryColor, setCustomSecondaryColor] = useState<string>("#98C1D9");
  const [customAccentColor, setCustomAccentColor] = useState<string>("#EE6C4D");

  // Style learning from existing documents
  const [useExistingDocs, setUseExistingDocs] = useState(false);
  const [styleCollections, setStyleCollections] = useState<string[]>([]);
  const [styleFolderId, setStyleFolderId] = useState<string | null>(null);
  const [includeStyleSubfolders, setIncludeStyleSubfolders] = useState(true);

  // PPTX Template (visual styling from existing PPTX)
  const [pptxTemplates, setPptxTemplates] = useState<Array<{ template_id: string; filename?: string; slide_count: number; created_at: string }>>([]);
  const [selectedPptxTemplate, setSelectedPptxTemplate] = useState<string | null>(null);
  const [isUploadingTemplate, setIsUploadingTemplate] = useState(false);
  const [templateSuggestions, setTemplateSuggestions] = useState<Array<{
    template_id: string;
    filename: string;
    slide_count: number;
    score: number;
    reason: string;
  }>>([]);
  const [isFetchingSuggestions, setIsFetchingSuggestions] = useState(false);

  // Built-in Template (from template library)
  const [selectedBuiltInTemplate, setSelectedBuiltInTemplate] = useState<DocumentTemplateMetadata | null>(null);

  // Template dialogs
  const [showTemplateSelector, setShowTemplateSelector] = useState(false);
  const [showSaveTemplate, setShowSaveTemplate] = useState(false);

  // Section feedback
  const [feedbackSection, setFeedbackSection] = useState<GenerationSection | null>(null);
  const [showSectionReview, setShowSectionReview] = useState(false);

  // Spell check state
  const [spellCheckResult, setSpellCheckResult] = useState<SpellCheckResponse | null>(null);
  const [isCheckingSpelling, setIsCheckingSpelling] = useState(false);
  const spellCheckJob = useSpellCheckJob();

  // LLM provider/model selection
  const [selectedProviderId, setSelectedProviderId] = useState<string | null>(null);
  const [selectedModelName, setSelectedModelName] = useState<string | null>(null);
  const [temperatureMode, setTemperatureMode] = useState<"auto" | "manual">("auto");
  const [manualTemperature, setManualTemperature] = useState(0.7);

  // Dual Mode (RAG + General)
  const [dualModeEnabled, setDualModeEnabled] = useState(false);
  const [dualModeBlend, setDualModeBlend] = useState<"merged" | "docs_first">("merged");

  // Moodboard gallery
  const [moodBoardTab, setMoodBoardTab] = useState<"create" | "gallery">("create");
  const [savedMoodBoards, setSavedMoodBoards] = useState<any[]>([]);
  const [isLoadingMoodBoards, setIsLoadingMoodBoards] = useState(false);
  const [viewingMoodBoard, setViewingMoodBoard] = useState<any>(null);

  // Mood Board state
  const [moodBoardColors, setMoodBoardColors] = useState<string[]>(["#6366f1", "#8b5cf6", "#ec4899"]);
  const [moodBoardKeywords, setMoodBoardKeywords] = useState<string[]>([]);
  const [moodBoardKeywordInput, setMoodBoardKeywordInput] = useState("");
  const [moodBoardImages, setMoodBoardImages] = useState<Array<{ url: string; file?: File }>>([]);
  const [moodBoardStyleNotes, setMoodBoardStyleNotes] = useState("");
  const [generatedMoodBoard, setGeneratedMoodBoard] = useState<any>(null);
  const [isGeneratingMoodBoard, setIsGeneratingMoodBoard] = useState(false);
  const [multiLLMEnabled, setMultiLLMEnabled] = useState(false);
  const [multiLLMProviders, setMultiLLMProviders] = useState<Array<{ provider_id: string; model: string }>>([]);
  const [multiLLMStrategy, setMultiLLMStrategy] = useState<"best_of" | "merge">("merge");

  // Queries - only fetch when authenticated
  const { data: job, isLoading: jobLoading, isError: jobError, error: jobQueryError } = useGenerationJob(currentJobId || "");
  const { data: recentJobs } = useGenerationJobs();
  const { data: collectionsData, isLoading: isLoadingCollections, refetch: refetchCollections } = useCollections({ enabled: isAuthenticated });
  const { data: providersData } = useLLMProviders({ enabled: isAuthenticated });
  const { data: themesData } = useGetThemes();
  const suggestTheme = useSuggestTheme();

  // Mutations
  const createJob = useCreateGenerationJob();
  const generateOutline = useGenerateOutline();
  const approveOutline = useApproveOutline();
  const approveSectionPlans = useApproveSectionPlans();
  const generateContent = useGenerateContent();
  const downloadDocument = useDownloadGeneratedDocument();
  const cancelJob = useCancelGenerationJob();

  const currentStepIndex = steps.findIndex((s) => s.id === currentStep);

  // Load Query Enhancement preference from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("create_enhance_query");
    if (saved !== null) setEnhanceQuery(JSON.parse(saved));
  }, []);

  // Handle URL parameters (job resumption from content review)
  useEffect(() => {
    const jobIdParam = searchParams.get('job');
    const reviewComplete = searchParams.get('reviewComplete');

    if (jobIdParam) {
      setCurrentJobId(jobIdParam);

      // If review is complete, skip to download step
      if (reviewComplete === 'true') {
        setCurrentStep('download');
        toast.success('Content review completed!', {
          description: 'Your document is ready for download.',
        });
      }
    }
  }, [searchParams]);

  // Handle job not found error (e.g., after backend restart)
  useEffect(() => {
    if (jobError && currentJobId) {
      // Check if it's a 404 error (job no longer exists)
      const errorMessage = jobQueryError instanceof Error ? jobQueryError.message : String(jobQueryError);
      const is404 = errorMessage.includes('404') || errorMessage.toLowerCase().includes('not found');

      if (is404) {
        toast.error('Session expired', {
          description: 'The generation job was not found. This may happen after a server restart. Please start a new document.',
        });
        // Reset state to allow starting fresh
        setCurrentJobId(null);
        setCurrentStep('format');
        setOutline([]);
        setDocumentTitle('');
      }
    }
  }, [jobError, jobQueryError, currentJobId]);

  const handleEnhanceQueryChange = (enabled: boolean) => {
    setEnhanceQuery(enabled);
    localStorage.setItem("create_enhance_query", JSON.stringify(enabled));
  };

  // Mood Board helper functions
  const addMoodBoardColor = () => {
    if (moodBoardColors.length < 8) {
      const randomColor = `#${Math.floor(Math.random()*16777215).toString(16).padStart(6, '0')}`;
      setMoodBoardColors([...moodBoardColors, randomColor]);
    }
  };

  const removeMoodBoardColor = (index: number) => {
    if (moodBoardColors.length > 1) {
      setMoodBoardColors(moodBoardColors.filter((_, i) => i !== index));
    }
  };

  const updateMoodBoardColor = (index: number, color: string) => {
    const newColors = [...moodBoardColors];
    newColors[index] = color;
    setMoodBoardColors(newColors);
  };

  const addMoodBoardKeyword = () => {
    const trimmed = moodBoardKeywordInput.trim();
    if (trimmed && !moodBoardKeywords.includes(trimmed)) {
      setMoodBoardKeywords([...moodBoardKeywords, trimmed]);
      setMoodBoardKeywordInput("");
    }
  };

  const removeMoodBoardKeyword = (keyword: string) => {
    setMoodBoardKeywords(moodBoardKeywords.filter(k => k !== keyword));
  };

  const handleMoodBoardImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = (event) => {
          setMoodBoardImages(prev => [...prev, {
            url: event.target?.result as string,
            file,
          }]);
        };
        reader.readAsDataURL(file);
      });
    }
  };

  const removeMoodBoardImage = (index: number) => {
    setMoodBoardImages(moodBoardImages.filter((_, i) => i !== index));
  };

  const handleGenerateMoodBoard = async () => {
    if (!topic.trim()) {
      toast.error("Please provide a name for your mood board");
      return;
    }

    setIsGeneratingMoodBoard(true);

    try {
      // Call the backend Mood Board API
      const response = await api.post("/moodboard/generate", {
        name: topic,
        description: context || undefined,
        colors: moodBoardColors,
        keywords: moodBoardKeywords,
        style_notes: moodBoardStyleNotes || undefined,
        provider_id: selectedProviderId || undefined,
        model: selectedModelName || undefined,
        use_existing_docs: useExistingDocs,
        collection_filters: useExistingDocs && styleCollections.length > 0 ? styleCollections : undefined,
        dual_mode: useExistingDocs && dualModeEnabled,
        multi_llm: multiLLMEnabled && multiLLMProviders.length > 1 ? {
          providers: multiLLMProviders.map(p => ({ provider_id: p.provider_id, model: p.model })),
          merge_strategy: multiLLMStrategy,
        } : undefined,
      });

      const data = response.data as any;

      setGeneratedMoodBoard({
        id: data.id,
        name: data.name,
        description: data.description,
        colors: data.colors,
        keywords: data.keywords,
        styleNotes: data.style_notes,
        generatedSuggestions: {
          typography: data.generated_suggestions.typography,
          additionalColors: data.generated_suggestions.additional_colors,
          moodKeywords: data.generated_suggestions.mood_keywords,
          inspirationNotes: data.generated_suggestions.inspiration_notes,
          designDirection: data.generated_suggestions.design_direction,
          colorPsychology: data.generated_suggestions.color_psychology,
        },
        createdAt: data.created_at,
        modelUsed: data.model_used,
      });

      toast.success("Mood board generated!", {
        description: "AI suggestions have been added to your board.",
      });
    } catch (err: any) {
      // Fallback to local simulation if API fails
      if (err.response?.status === 404 || err.message?.includes("Network")) {
        setGeneratedMoodBoard({
          name: topic,
          description: context,
          colors: moodBoardColors,
          keywords: moodBoardKeywords,
          styleNotes: moodBoardStyleNotes,
          generatedSuggestions: {
            typography: ["Inter", "Playfair Display", "Space Grotesk"],
            additionalColors: ["#f97316", "#22c55e", "#0ea5e9"],
            moodKeywords: ["modern", "elegant", "bold"],
            inspirationNotes: "Based on the colors and keywords provided, this mood board suggests a contemporary, sophisticated aesthetic with vibrant accent colors.",
          },
          createdAt: new Date().toISOString(),
        });
        toast.success("Mood board generated!", {
          description: "AI suggestions have been added (offline mode).",
        });
      } else {
        toast.error(err.response?.data?.detail || err.message || "Failed to generate mood board");
      }
    } finally {
      setIsGeneratingMoodBoard(false);
    }
  };

  const resetMoodBoard = () => {
    setMoodBoardColors(["#6366f1", "#8b5cf6", "#ec4899"]);
    setMoodBoardKeywords([]);
    setMoodBoardKeywordInput("");
    setMoodBoardImages([]);
    setMoodBoardStyleNotes("");
    setGeneratedMoodBoard(null);
  };

  // Load PPTX templates when PPTX format is selected
  useEffect(() => {
    if (selectedFormat === "pptx" && isAuthenticated) {
      const loadTemplates = async () => {
        try {
          const { api } = await import("@/lib/api");
          const templates = await api.listPptxTemplates();
          setPptxTemplates(templates);
        } catch (error) {
          console.error("Failed to load PPTX templates:", error);
        }
      };
      loadTemplates();
    }
  }, [selectedFormat, isAuthenticated]);

  // Fetch AI template suggestions when topic/context changes (debounced)
  useEffect(() => {
    if (selectedFormat !== "pptx" || !isAuthenticated || !topic || !context) {
      setTemplateSuggestions([]);
      return;
    }

    // Debounce the API call
    const timer = setTimeout(async () => {
      if (pptxTemplates.length === 0) return; // No templates to suggest from

      setIsFetchingSuggestions(true);
      try {
        const { api } = await import("@/lib/api");
        const result = await api.suggestPptxTemplates(topic, context);
        setTemplateSuggestions(result.suggestions);
      } catch (error) {
        console.error("Failed to get template suggestions:", error);
        setTemplateSuggestions([]);
      } finally {
        setIsFetchingSuggestions(false);
      }
    }, 1000); // Wait 1 second after typing stops

    return () => clearTimeout(timer);
  }, [selectedFormat, isAuthenticated, topic, context, pptxTemplates.length]);

  // Handle PPTX template upload
  const handleTemplateUpload = async (file: File) => {
    setIsUploadingTemplate(true);
    try {
      const { api } = await import("@/lib/api");
      const result = await api.uploadPptxTemplate(file);
      toast.success("Template uploaded", {
        description: result.message,
      });
      // Refresh template list
      const templates = await api.listPptxTemplates();
      setPptxTemplates(templates);
      setSelectedPptxTemplate(result.template_id);
    } catch (error: any) {
      toast.error("Failed to upload template", {
        description: error.message || "An error occurred",
      });
    } finally {
      setIsUploadingTemplate(false);
    }
  };

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
          // Set enhanced suggestions
          if (suggestion.font_family) setSelectedFontFamily(suggestion.font_family);
          if (suggestion.layout) setSelectedLayout(suggestion.layout);
          if (suggestion.animations !== undefined) setEnableAnimations(suggestion.animations);
          // Store available options for UI
          if (suggestion.available_fonts) setAvailableFonts(suggestion.available_fonts);
          if (suggestion.available_layouts) setAvailableLayouts(suggestion.available_layouts);
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
            // Set enhanced suggestions
            if (suggestion.font_family) setSelectedFontFamily(suggestion.font_family);
            if (suggestion.layout) setSelectedLayout(suggestion.layout);
            if (suggestion.animations !== undefined) setEnableAnimations(suggestion.animations);
            // Store available options for UI
            if (suggestion.available_fonts) setAvailableFonts(suggestion.available_fonts);
            if (suggestion.available_layouts) setAvailableLayouts(suggestion.available_layouts);
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
        // Mood board has its own generation flow - skip job creation
        if (selectedFormat === "moodboard") {
          if (generatedMoodBoard) {
            setCurrentStep("download");
          }
          return;
        }
        if (topic) {
          try {
            const newJob = await createJob.mutateAsync({
              title: topic,
              description: context || topic,
              output_format: selectedFormat!,
              output_language: outputLanguage,
              include_images: includeImages,
              auto_charts: selectedFormat === "pptx" ? autoCharts : undefined,
              collection_filters: selectedCollections.length > 0 ? selectedCollections : undefined,
              folder_id: selectedFolderId || undefined,
              include_subfolders: includeSubfolders,
              page_count: pageCount, // null = auto mode
              theme: selectedTheme,
              include_sources: includeSources,
              // Enhanced theming options
              font_family: selectedFontFamily || undefined,
              layout: selectedLayout || undefined,
              animations: enableAnimations,
              animation_speed: enableAnimations ? (useCustomDuration ? "custom" : animationSpeed) : undefined,
              animation_duration_ms: enableAnimations && useCustomDuration ? customDuration : undefined,
              enable_quality_review: enableQualityReview,
              // AI Proofreading with CriticAgent
              enable_critic_review: enableCriticReview,
              quality_threshold: enableCriticReview ? qualityThreshold : undefined,
              fix_styling: enableCriticReview ? fixStyling : undefined,
              fix_incomplete: enableCriticReview ? fixIncomplete : undefined,
              // Include AI explanations in speaker notes
              include_notes_explanation: includeNotesExplanation,
              // Custom colors override theme
              custom_colors: useCustomColors ? {
                primary: customPrimaryColor,
                secondary: customSecondaryColor,
                accent: customAccentColor,
              } : undefined,
              // Style learning from existing documents
              use_existing_docs: useExistingDocs,
              style_collection_filters: useExistingDocs && styleCollections.length > 0 ? styleCollections : undefined,
              style_folder_id: useExistingDocs && styleFolderId ? styleFolderId : undefined,
              include_style_subfolders: useExistingDocs ? includeStyleSubfolders : undefined,
              // Query Enhancement for source search
              enhance_query: enhanceQuery ?? undefined,
              // LLM provider/model selection
              provider_id: selectedProviderId || undefined,
              model: selectedModelName || undefined,
              temperature_override: temperatureMode === "manual" ? manualTemperature : undefined,
              // Dual Mode (RAG + General)
              dual_mode: useExistingDocs && dualModeEnabled ? true : undefined,
              dual_mode_blend: useExistingDocs && dualModeEnabled ? dualModeBlend : undefined,
              // PPTX Template for visual styling
              template_pptx_id: selectedFormat === "pptx" && selectedPptxTemplate ? selectedPptxTemplate : undefined,
              // Vision Analysis (PPTX only) - per-document overrides
              enable_template_vision_analysis: selectedFormat === "pptx" && enableTemplateVisionAnalysis !== null ? enableTemplateVisionAnalysis : undefined,
              template_vision_model: selectedFormat === "pptx" && enableTemplateVisionAnalysis ? templateVisionModel : undefined,
              enable_vision_review: selectedFormat === "pptx" && enableVisionReview !== null ? enableVisionReview : undefined,
              vision_review_model: selectedFormat === "pptx" && enableVisionReview ? visionReviewModel : undefined,
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
          // Skip if already generating or completed
          if (job?.status === "generating" || job?.status === "processing" || job?.status === "completed") {
            setCurrentStep("content");
            break;
          }
          try {
            // Approve outline with sections that include approval status
            const approvedJob = await approveOutline.mutateAsync({
              jobId: currentJobId,
              modifications: {
                title: documentTitle || undefined,
                sections: outline.length > 0 ? outline : undefined,
                tone,
                theme: selectedTheme,
              },
            });

            // After outline is approved, approve sections and start generation
            if (approvedJob.status === "sections_planning" || approvedJob.status === "outline_approved") {
              // Build section approvals from our outline state
              const sectionApprovals = outline.map((s, index) => ({
                section_id: approvedJob.sections?.[index]?.id || s.id,
                approved: s.approved,
                title: s.title,
                description: s.description,
              }));

              await approveSectionPlans.mutateAsync({
                jobId: currentJobId,
                sectionApprovals: sectionApprovals,
              });
            }
            // Trigger content generation after section plans are approved
            setCurrentStep("content");
            await generateContent.mutateAsync(currentJobId);
          } catch (error: any) {
            console.error("Failed to approve outline:", error);
            const errorDetail = error?.detail || error?.message || "";
            // If the job is already completed, just move to download step
            if (errorDetail.includes("COMPLETED")) {
              toast.info("Document already generated", {
                description: "Moving to download step.",
              });
              setCurrentStep("download");
            } else if (errorDetail.includes("OUTLINE_APPROVED") || errorDetail.includes("GENERATING") || errorDetail.includes("SECTIONS_PLANNING")) {
              // Job is already in progress, move to appropriate step
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

  const handleSpellCheck = async () => {
    if (!currentJobId) return;
    setIsCheckingSpelling(true);
    setSpellCheckResult(null);
    try {
      const result = await spellCheckJob.mutateAsync(currentJobId);
      setSpellCheckResult(result);
      if (!result.has_issues || result.issues.length === 0) {
        toast.success("No Spelling Errors", {
          description: "Your document has no spelling errors.",
        });
      } else {
        toast.info("Spelling Check Complete", {
          description: `Found ${result.issues.length} potential spelling ${result.issues.length === 1 ? 'issue' : 'issues'}.`,
        });
      }
    } catch (error) {
      console.error("Spell check failed:", error);
      toast.error("Spell Check Failed", {
        description: "Could not check spelling. Please try again.",
      });
    } finally {
      setIsCheckingSpelling(false);
    }
  };

  const handleAddSection = () => {
    setOutline([
      ...outline,
      { id: crypto.randomUUID(), title: "New Section", description: "", approved: true },
    ]);
  };

  const handleRemoveSection = (id: string) => {
    setOutline(outline.filter((s) => s.id !== id));
  };

  const handleUpdateSection = (id: string, field: "title" | "description", value: string) => {
    setOutline(outline.map((s) => (s.id === id ? { ...s, [field]: value } : s)));
  };

  const handleToggleSectionApproval = (id: string) => {
    setOutline(outline.map((s) => (s.id === id ? { ...s, approved: !s.approved } : s)));
  };

  const handleStartNew = () => {
    setCurrentStep("format");
    setSelectedFormat(null);
    setTopic("");
    setContext("");
    setOutputLanguage("en");
    setOutline([]);
    setDocumentTitle("");
    setCurrentJobId(null);
  };

  // Apply a template's settings
  const handleApplyTemplate = (template: GenerationTemplate) => {
    const settings = template.settings as TemplateSettings;

    // Apply format
    if (settings.output_format) {
      setSelectedFormat(settings.output_format as OutputFormat);
    }

    // Apply theme and styling
    if (settings.theme) setSelectedTheme(settings.theme);
    if (settings.font_family) setSelectedFontFamily(settings.font_family);
    if (settings.layout_template) setSelectedLayout(settings.layout_template);
    if (typeof settings.include_toc === 'boolean') {
      // Note: TOC setting is stored but UI may not have this toggle yet
    }
    if (typeof settings.include_sources === 'boolean') setIncludeSources(settings.include_sources);
    if (typeof settings.use_existing_docs === 'boolean') setUseExistingDocs(settings.use_existing_docs);
    if (typeof settings.enable_animations === 'boolean') setEnableAnimations(settings.enable_animations);
    if (settings.animation_speed) {
      const speed = settings.animation_speed as 'very_slow' | 'slow' | 'med' | 'fast' | 'very_fast';
      if (['very_slow', 'slow', 'med', 'fast', 'very_fast'].includes(speed)) {
        setAnimationSpeed(speed);
      }
    }
    if (settings.animation_duration_ms) {
      setUseCustomDuration(true);
      setCustomDuration(settings.animation_duration_ms);
    }
    if (typeof settings.enable_quality_review === 'boolean') setEnableQualityReview(settings.enable_quality_review);

    // Apply custom colors
    if (settings.custom_colors) {
      setUseCustomColors(true);
      if (settings.custom_colors.primary) setCustomPrimaryColor(settings.custom_colors.primary);
      if (settings.custom_colors.secondary) setCustomSecondaryColor(settings.custom_colors.secondary);
      if (settings.custom_colors.accent) setCustomAccentColor(settings.custom_colors.accent);
    }

    // Apply default collections for style learning
    if (template.default_collections && template.default_collections.length > 0) {
      setStyleCollections(template.default_collections);
    }

    // Apply vision analysis settings (PPTX only)
    if (typeof settings.enable_template_vision_analysis === 'boolean') {
      setEnableTemplateVisionAnalysis(settings.enable_template_vision_analysis);
    }
    if (settings.template_vision_model) {
      setTemplateVisionModel(settings.template_vision_model);
    }
    if (typeof settings.enable_vision_review === 'boolean') {
      setEnableVisionReview(settings.enable_vision_review);
    }
    if (settings.vision_review_model) {
      setVisionReviewModel(settings.vision_review_model);
    }

    toast.success(`Applied template: ${template.name}`);

    // Move to topic step if format is set
    if (settings.output_format) {
      setCurrentStep("topic");
    }
  };

  // Get current settings for saving as template
  const getCurrentSettings = (): TemplateSettings => ({
    output_format: selectedFormat || "docx",
    theme: selectedTheme,
    font_family: selectedFontFamily,
    layout_template: selectedLayout,
    include_toc: false,
    include_sources: includeSources,
    use_existing_docs: useExistingDocs,
    enable_animations: enableAnimations,
    animation_speed: animationSpeed,
    enable_quality_review: enableQualityReview,
    custom_colors: useCustomColors ? {
      primary: customPrimaryColor,
      secondary: customSecondaryColor,
      accent: customAccentColor,
    } : undefined,
    // Vision Analysis (PPTX only)
    enable_template_vision_analysis: enableTemplateVisionAnalysis ?? undefined,
    template_vision_model: templateVisionModel !== "auto" ? templateVisionModel : undefined,
    enable_vision_review: enableVisionReview ?? undefined,
    vision_review_model: visionReviewModel !== "auto" ? visionReviewModel : undefined,
  });

  // Update outline from job data
  if (job?.outline && outline.length === 0 && currentStep === "outline") {
    setDocumentTitle(job.outline.title || "");
    setOutline(
      job.outline.sections?.map((s: { title: string; description?: string }, i: number) => ({
        id: crypto.randomUUID(),
        title: s.title || "",
        description: s.description || "",
        approved: true, // Default all sections to approved
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
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">Choose Output Format</h2>
                  <p className="text-muted-foreground">
                    Select the format for your generated document
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => setShowTemplateSelector(true)}
                  className="flex items-center gap-2"
                >
                  <FolderOpen className="h-4 w-4" />
                  Use Template
                </Button>
              </div>
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
          {currentStep === "topic" && selectedFormat === "moodboard" && (
            <div className="space-y-6 max-w-3xl">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <Palette className="h-5 w-5" />
                  Mood Board
                </h2>
                <div className="flex items-center p-1 bg-muted rounded-lg">
                  <Button
                    variant={moodBoardTab === "create" ? "default" : "ghost"}
                    size="sm"
                    className="text-xs h-7"
                    onClick={() => setMoodBoardTab("create")}
                  >
                    Create New
                  </Button>
                  <Button
                    variant={moodBoardTab === "gallery" ? "default" : "ghost"}
                    size="sm"
                    className="text-xs h-7"
                    onClick={async () => {
                      setMoodBoardTab("gallery");
                      setIsLoadingMoodBoards(true);
                      try {
                        const resp = await api.get("/moodboard/list?limit=50");
                        setSavedMoodBoards((resp.data as any).mood_boards || []);
                      } catch { /* ignore */ }
                      setIsLoadingMoodBoards(false);
                    }}
                  >
                    <Eye className="h-3 w-3 mr-1" />
                    My Boards
                  </Button>
                </div>
              </div>

              {/* Gallery View */}
              {moodBoardTab === "gallery" && (
                <div className="space-y-4">
                  {isLoadingMoodBoards ? (
                    <div className="flex items-center justify-center p-8">
                      <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                    </div>
                  ) : savedMoodBoards.length === 0 ? (
                    <div className="text-center p-8 text-muted-foreground">
                      <Palette className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No saved mood boards yet.</p>
                      <p className="text-xs">Create one and click Save to see it here.</p>
                    </div>
                  ) : (
                    <>
                      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                        {savedMoodBoards.map((board: any) => (
                          <Card key={board.id} className="cursor-pointer hover:border-primary/50 transition-colors">
                            <CardContent className="p-4 space-y-3">
                              {/* Color swatches */}
                              <div className="flex gap-1">
                                {(board.color_palette || []).slice(0, 6).map((color: string, i: number) => (
                                  <div
                                    key={i}
                                    className="h-6 flex-1 rounded"
                                    style={{ backgroundColor: color }}
                                  />
                                ))}
                              </div>
                              <div>
                                <p className="font-medium text-sm truncate">{board.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {(board.color_palette || []).length} colors
                                  {(board.themes || []).length > 0 && <> &middot; {(board.themes || []).length} fonts</>}
                                  {" "}&middot; {new Date(board.created_at).toLocaleDateString()}
                                </p>
                              </div>
                              {(board.style_tags || []).length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {(board.style_tags || []).slice(0, 3).map((tag: string) => (
                                    <span key={tag} className="px-1.5 py-0.5 rounded bg-primary/10 text-primary text-[10px]">{tag}</span>
                                  ))}
                                  {(board.style_tags || []).length > 3 && (
                                    <span className="text-[10px] text-muted-foreground">+{(board.style_tags || []).length - 3}</span>
                                  )}
                                </div>
                              )}
                              <div className="flex gap-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="text-xs flex-1"
                                  onClick={async (e) => {
                                    e.stopPropagation();
                                    try {
                                      const resp = await api.get(`/moodboard/${board.id}`);
                                      setViewingMoodBoard(resp.data);
                                    } catch {
                                      setViewingMoodBoard(board);
                                    }
                                  }}
                                >
                                  <Eye className="h-3 w-3 mr-1" />
                                  View
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="text-xs text-destructive hover:text-destructive"
                                  onClick={async (e) => {
                                    e.stopPropagation();
                                    try {
                                      await api.delete(`/moodboard/${board.id}`);
                                      setSavedMoodBoards(savedMoodBoards.filter((b: any) => b.id !== board.id));
                                      toast.success("Mood board deleted");
                                    } catch {
                                      toast.error("Failed to delete");
                                    }
                                  }}
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>

                      {/* Moodboard Viewer — Infinite Canvas */}
                      {viewingMoodBoard && (
                        <MoodboardCanvas
                          board={viewingMoodBoard}
                          onClose={() => setViewingMoodBoard(null)}
                          onBoardUpdate={(updated) => {
                            setViewingMoodBoard(updated);
                            setSavedMoodBoards((prev: any[]) => prev.map((b: any) => b.id === updated.id ? { ...b, ...updated } : b));
                          }}
                        />
                      )}

                    </>
                  )}
                </div>
              )}

              {/* Create View */}
              {moodBoardTab === "create" && (<>
              <div>
                <p className="text-muted-foreground">
                  Generate an AI-powered visual inspiration board for your design projects
                </p>
              </div>

              {/* Style Presets */}
              <div className="space-y-2">
                <Label className="text-sm font-medium">Quick Start — Style Presets</Label>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                  {([
                    { name: "Modern Startup", colors: ["#0066ff", "#00c2ff", "#f0f4f8", "#1a1a2e", "#ffffff"], keywords: ["modern", "clean", "innovative", "tech"], desc: "Clean, tech-forward visual identity" },
                    { name: "Luxury Brand", colors: ["#1c1c1c", "#c9a96e", "#f5f0eb", "#2d2d2d", "#8b7355"], keywords: ["elegant", "premium", "sophisticated", "refined"], desc: "Elevated, high-end aesthetic" },
                    { name: "Nature & Eco", colors: ["#2d6a4f", "#95d5b2", "#fefae0", "#606c38", "#dda15e"], keywords: ["organic", "sustainable", "earthy", "natural"], desc: "Earthy tones, organic feel" },
                    { name: "Creative Agency", colors: ["#ff006e", "#8338ec", "#3a86ff", "#ffbe0b", "#fb5607"], keywords: ["bold", "creative", "vibrant", "energetic"], desc: "Bold colors, high energy" },
                    { name: "Corporate", colors: ["#003049", "#d62828", "#f8f9fa", "#264653", "#e9ecef"], keywords: ["professional", "trustworthy", "structured", "formal"], desc: "Professional, business-ready" },
                    { name: "Pastel Soft", colors: ["#ffc8dd", "#bde0fe", "#cdb4db", "#a2d2ff", "#ffafcc"], keywords: ["soft", "gentle", "warm", "friendly"], desc: "Soft, approachable, warm tones" },
                    { name: "Dark Minimal", colors: ["#121212", "#1e1e1e", "#bb86fc", "#03dac6", "#cf6679"], keywords: ["dark", "minimal", "modern", "sleek"], desc: "Dark mode, Material-inspired" },
                    { name: "Academic", colors: ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"], keywords: ["scholarly", "analytical", "structured", "intellectual"], desc: "Research-focused, scholarly tone" },
                  ] as const).map((preset) => (
                    <button
                      key={preset.name}
                      onClick={() => {
                        setTopic(preset.name);
                        setContext(preset.desc);
                        setMoodBoardColors([...preset.colors]);
                        setMoodBoardKeywords([...preset.keywords]);
                      }}
                      className="text-left p-3 rounded-lg border hover:border-primary/50 transition-all group"
                    >
                      <div className="flex gap-0.5 h-4 rounded overflow-hidden mb-2">
                        {preset.colors.map((c, i) => (
                          <div key={i} className="flex-1" style={{ backgroundColor: c }} />
                        ))}
                      </div>
                      <p className="text-xs font-medium group-hover:text-primary transition-colors">{preset.name}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-1">{preset.desc}</p>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-6">
                {/* Basic Info */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Board Name *</Label>
                    <Input
                      placeholder="e.g., Tech Startup Brand Identity, Wedding Theme, Product Launch"
                      value={topic}
                      onChange={(e) => setTopic(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Description / Concept</Label>
                    <textarea
                      placeholder="Describe the mood, feeling, or concept you're going for..."
                      value={context}
                      onChange={(e) => setContext(e.target.value)}
                      className="w-full min-h-[80px] p-3 rounded-md border bg-background resize-none"
                    />
                  </div>
                </div>

                {/* Color Palette */}
                <div className="space-y-3 p-4 rounded-lg border bg-muted/30">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">Color Palette</Label>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={addMoodBoardColor}
                      disabled={moodBoardColors.length >= 8}
                    >
                      <Plus className="h-3 w-3 mr-1" />
                      Add Color
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    {moodBoardColors.map((color, index) => (
                      <div key={index} className="flex items-center gap-2 p-2 rounded-lg border bg-background">
                        <input
                          type="color"
                          value={color}
                          onChange={(e) => updateMoodBoardColor(index, e.target.value)}
                          className="w-10 h-10 rounded border cursor-pointer"
                        />
                        <span className="text-xs font-mono text-muted-foreground w-16">{color}</span>
                        {moodBoardColors.length > 1 && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => removeMoodBoardColor(index)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Select 1-8 colors that define your aesthetic. Click on a color to change it.
                  </p>
                </div>

                {/* Keywords / Tags */}
                <div className="space-y-3 p-4 rounded-lg border bg-muted/30">
                  <Label className="text-sm font-medium">Keywords / Style Tags</Label>
                  <div className="flex gap-2">
                    <Input
                      placeholder="Add a keyword..."
                      value={moodBoardKeywordInput}
                      onChange={(e) => setMoodBoardKeywordInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          addMoodBoardKeyword();
                        }
                      }}
                    />
                    <Button variant="outline" onClick={addMoodBoardKeyword}>
                      Add
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 min-h-[32px]">
                    {moodBoardKeywords.map((keyword) => (
                      <span
                        key={keyword}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-primary/10 text-primary text-sm"
                      >
                        {keyword}
                        <button
                          onClick={() => removeMoodBoardKeyword(keyword)}
                          className="ml-1 hover:text-destructive"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                    {moodBoardKeywords.length === 0 && (
                      <p className="text-xs text-muted-foreground">
                        Add keywords like: minimalist, bold, playful, corporate, vintage, etc.
                      </p>
                    )}
                  </div>
                </div>

                {/* Reference Images */}
                <div className="space-y-3 p-4 rounded-lg border bg-muted/30">
                  <Label className="text-sm font-medium">Reference Images (Optional)</Label>
                  <div className="grid grid-cols-4 gap-3">
                    {moodBoardImages.map((img, index) => (
                      <div key={index} className="relative group">
                        <img
                          src={img.url}
                          alt={`Reference ${index + 1}`}
                          className="w-full h-24 object-cover rounded-lg border"
                        />
                        <Button
                          variant="destructive"
                          size="icon"
                          className="absolute top-1 right-1 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => removeMoodBoardImage(index)}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                    <label className="border-2 border-dashed rounded-lg h-24 flex flex-col items-center justify-center cursor-pointer hover:bg-muted/50 transition-colors">
                      <input
                        type="file"
                        className="hidden"
                        accept="image/*"
                        multiple
                        onChange={handleMoodBoardImageUpload}
                      />
                      <Image className="h-6 w-6 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground mt-1">Add Image</span>
                    </label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Upload images that inspire your mood board's direction
                  </p>
                </div>

                {/* Style Notes */}
                <div className="space-y-2">
                  <Label className="text-sm font-medium">Additional Style Notes</Label>
                  <textarea
                    placeholder="Any specific requirements, brand guidelines, or preferences..."
                    value={moodBoardStyleNotes}
                    onChange={(e) => setMoodBoardStyleNotes(e.target.value)}
                    className="w-full min-h-[80px] p-3 rounded-md border bg-background resize-none"
                  />
                </div>

                {/* Learn from Existing Documents (Moodboard) */}
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
                        <p className="text-sm font-medium">Use Documents as Inspiration</p>
                        <p className="text-xs text-muted-foreground">
                          Reference your documents for themes, styles, and content direction
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
                        AI will use selected documents to enrich mood board suggestions
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Select source collections</label>
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
                            <p className="text-xs text-muted-foreground">No collections available.</p>
                          )}
                        </div>
                      </div>

                      {/* Dual Mode Toggle */}
                      <div
                        className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                          dualModeEnabled
                            ? "bg-blue-500/5 border-blue-500/30"
                            : "bg-muted/30 border-border hover:bg-muted/50"
                        }`}
                        onClick={() => setDualModeEnabled(!dualModeEnabled)}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`p-1.5 rounded-lg ${dualModeEnabled ? "bg-blue-500/10" : "bg-muted"}`}>
                            <Sparkles className={`h-4 w-4 ${dualModeEnabled ? "text-blue-500" : "text-muted-foreground"}`} />
                          </div>
                          <div>
                            <p className="text-sm font-medium">Dual Mode (Documents + General AI)</p>
                            <p className="text-xs text-muted-foreground">
                              Combine document knowledge with general AI for richer suggestions
                            </p>
                          </div>
                        </div>
                        <div
                          className={`w-10 h-5 rounded-full transition-colors flex items-center ${
                            dualModeEnabled ? "bg-blue-500 justify-end" : "bg-muted-foreground/30 justify-start"
                          }`}
                        >
                          <div className="w-4 h-4 bg-white rounded-full shadow-sm mx-0.5" />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* LLM Configuration */}
                <div className="space-y-3 p-4 rounded-lg border bg-muted/30">
                  <div className="flex items-center gap-2">
                    <Cog className="h-4 w-4 text-muted-foreground" />
                    <Label className="text-sm font-medium">LLM Configuration</Label>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <label className="text-xs text-muted-foreground">Provider</label>
                      <select
                        value={selectedProviderId || ""}
                        onChange={(e) => {
                          setSelectedProviderId(e.target.value || null);
                          setSelectedModelName(null);
                        }}
                        className="w-full h-9 px-3 rounded-md border bg-background text-sm"
                      >
                        <option value="">System Default</option>
                        {(providersData?.providers || []).filter((p: any) => p.is_active).map((p: any) => (
                          <option key={p.id} value={p.id}>{p.name} ({p.provider_type})</option>
                        ))}
                      </select>
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs text-muted-foreground">Model</label>
                      <select
                        value={selectedModelName || ""}
                        onChange={(e) => setSelectedModelName(e.target.value || null)}
                        className="w-full h-9 px-3 rounded-md border bg-background text-sm"
                      >
                        <option value="">Auto (provider default)</option>
                        {selectedProviderId && (() => {
                          const provider = (providersData?.providers || []).find((p: any) => p.id === selectedProviderId);
                          if (provider?.default_chat_model) {
                            return <option key={provider.default_chat_model} value={provider.default_chat_model}>{provider.default_chat_model}</option>;
                          }
                          return null;
                        })()}
                      </select>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-muted-foreground">Temperature</label>
                      <select
                        value={temperatureMode}
                        onChange={(e) => setTemperatureMode(e.target.value as "auto" | "manual")}
                        className="h-7 px-2 rounded border bg-background text-xs"
                      >
                        <option value="auto">Auto (format-optimized)</option>
                        <option value="manual">Manual</option>
                      </select>
                    </div>
                    {temperatureMode === "manual" && (
                      <div className="flex items-center gap-3">
                        <Slider
                          value={[manualTemperature]}
                          onValueChange={([v]) => setManualTemperature(v)}
                          min={0}
                          max={1.5}
                          step={0.05}
                          className="flex-1"
                        />
                        <span className="text-xs font-mono w-8 text-right">{manualTemperature.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Multi-LLM Toggle */}
                {(providersData?.providers || []).filter((p: any) => p.is_active).length > 1 && (
                  <div className="space-y-3 p-4 rounded-lg border bg-muted/30">
                    <div
                      className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                        multiLLMEnabled
                          ? "bg-purple-500/5 border-purple-500/30"
                          : "bg-muted/30 border-border hover:bg-muted/50"
                      }`}
                      onClick={() => {
                        const next = !multiLLMEnabled;
                        setMultiLLMEnabled(next);
                        if (next && multiLLMProviders.length === 0) {
                          const active = (providersData?.providers || []).filter((p: any) => p.is_active).slice(0, 2);
                          setMultiLLMProviders(active.map((p: any) => ({ provider_id: p.id, model: p.default_chat_model || "" })));
                        }
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`p-1.5 rounded-lg ${multiLLMEnabled ? "bg-purple-500/10" : "bg-muted"}`}>
                          <Sparkles className={`h-4 w-4 ${multiLLMEnabled ? "text-purple-500" : "text-muted-foreground"}`} />
                        </div>
                        <div>
                          <p className="text-sm font-medium">Multi-LLM Generation</p>
                          <p className="text-xs text-muted-foreground">
                            Combine ideas from multiple AI models for richer results
                          </p>
                        </div>
                      </div>
                      <div
                        className={`w-10 h-5 rounded-full transition-colors flex items-center ${
                          multiLLMEnabled ? "bg-purple-500 justify-end" : "bg-muted-foreground/30 justify-start"
                        }`}
                      >
                        <div className="w-4 h-4 bg-white rounded-full shadow-sm mx-0.5" />
                      </div>
                    </div>

                    {multiLLMEnabled && (
                      <div className="space-y-3 pl-2">
                        <div className="flex items-center gap-3">
                          <label className="text-xs text-muted-foreground">Merge Strategy</label>
                          <select
                            value={multiLLMStrategy}
                            onChange={(e) => setMultiLLMStrategy(e.target.value as "best_of" | "merge")}
                            className="h-7 px-2 rounded border bg-background text-xs"
                          >
                            <option value="merge">Merge best parts</option>
                            <option value="best_of">Pick best overall</option>
                          </select>
                        </div>

                        {multiLLMProviders.map((mp, idx) => (
                          <div key={idx} className="flex gap-2 items-center">
                            <span className="text-xs text-muted-foreground w-4">{idx + 1}.</span>
                            <select
                              value={mp.provider_id}
                              onChange={(e) => {
                                const updated = [...multiLLMProviders];
                                const prov = (providersData?.providers || []).find((p: any) => p.id === e.target.value);
                                updated[idx] = { provider_id: e.target.value, model: prov?.default_chat_model || "" };
                                setMultiLLMProviders(updated);
                              }}
                              className="flex-1 h-8 px-2 rounded border bg-background text-xs"
                            >
                              {(providersData?.providers || []).filter((p: any) => p.is_active).map((p: any) => (
                                <option key={p.id} value={p.id}>{p.name}</option>
                              ))}
                            </select>
                            {multiLLMProviders.length > 2 && (
                              <button
                                onClick={() => setMultiLLMProviders(multiLLMProviders.filter((_, i) => i !== idx))}
                                className="text-xs text-red-400 hover:text-red-500"
                              >
                                Remove
                              </button>
                            )}
                          </div>
                        ))}

                        {multiLLMProviders.length < 3 && (
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-xs"
                            onClick={() => {
                              const active = (providersData?.providers || []).filter((p: any) => p.is_active);
                              if (active.length > 0) {
                                setMultiLLMProviders([...multiLLMProviders, { provider_id: active[0].id, model: active[0].default_chat_model || "" }]);
                              }
                            }}
                          >
                            <Plus className="h-3 w-3 mr-1" /> Add Provider
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Generated Mood Board Result */}
                {generatedMoodBoard && (
                  <div className="space-y-4 p-4 rounded-lg border bg-gradient-to-br from-primary/5 to-primary/10">
                    <h4 className="font-medium flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-primary" />
                      AI-Generated Suggestions
                    </h4>

                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-2">
                        <p className="text-sm font-medium">Suggested Typography</p>
                        <div className="flex flex-wrap gap-2">
                          {generatedMoodBoard.generatedSuggestions.typography.map((font: string) => (
                            <span key={font} className="px-2 py-1 rounded bg-background border text-sm">{font}</span>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <p className="text-sm font-medium">Complementary Colors</p>
                        <div className="flex gap-2">
                          {generatedMoodBoard.generatedSuggestions.additionalColors.map((color: string) => (
                            <div key={color} className="flex items-center gap-1">
                              <div
                                className="w-6 h-6 rounded border"
                                style={{ backgroundColor: color }}
                              />
                              <span className="text-xs font-mono text-muted-foreground">{color}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <p className="text-sm font-medium">Mood Keywords</p>
                        <div className="flex flex-wrap gap-2">
                          {generatedMoodBoard.generatedSuggestions.moodKeywords.map((keyword: string) => (
                            <span key={keyword} className="px-2 py-1 rounded bg-primary/10 text-primary text-sm">{keyword}</span>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-2 sm:col-span-2">
                        <p className="text-sm font-medium">Inspiration Notes</p>
                        <p className="text-sm text-muted-foreground bg-background p-3 rounded border">
                          {generatedMoodBoard.generatedSuggestions.inspirationNotes}
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-2 pt-2">
                      <Button variant="outline" size="sm" onClick={resetMoodBoard}>
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Start Over
                      </Button>
                      <Button
                        size="sm"
                        onClick={async () => {
                          try {
                            await api.post("/moodboard/save", {
                              mood_board: {
                                name: generatedMoodBoard.name,
                                description: generatedMoodBoard.description,
                                colors: generatedMoodBoard.colors || moodBoardColors,
                                keywords: generatedMoodBoard.keywords || moodBoardKeywords,
                                style_notes: generatedMoodBoard.styleNotes || moodBoardStyleNotes,
                                generated_suggestions: {
                                  typography: generatedMoodBoard.generatedSuggestions.typography,
                                  additional_colors: generatedMoodBoard.generatedSuggestions.additionalColors,
                                  mood_keywords: generatedMoodBoard.generatedSuggestions.moodKeywords,
                                  inspiration_notes: generatedMoodBoard.generatedSuggestions.inspirationNotes,
                                  design_direction: generatedMoodBoard.generatedSuggestions.designDirection,
                                  color_psychology: generatedMoodBoard.generatedSuggestions.colorPsychology,
                                },
                              },
                            });
                            toast.success("Mood board saved!", { description: "You can view it from My Boards." });
                          } catch (err: any) {
                            toast.error(err.response?.data?.detail || "Failed to save mood board");
                          }
                        }}
                      >
                        <Save className="h-4 w-4 mr-2" />
                        Save Mood Board
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const sug = generatedMoodBoard.generatedSuggestions;
                          const exportData = {
                            name: generatedMoodBoard.name,
                            description: generatedMoodBoard.description,
                            color_palette: generatedMoodBoard.colors || moodBoardColors,
                            style_tags: generatedMoodBoard.keywords || moodBoardKeywords,
                            themes: sug?.typography || [],
                            generated_suggestions: {
                              typography: sug?.typography || [],
                              additional_colors: sug?.additionalColors || [],
                              mood_keywords: sug?.moodKeywords || [],
                              inspiration_notes: sug?.inspirationNotes || "",
                              design_direction: sug?.designDirection || "",
                              color_psychology: sug?.colorPsychology || {},
                            },
                            created_at: generatedMoodBoard.createdAt,
                            model_used: generatedMoodBoard.modelUsed,
                          };
                          const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement("a");
                          a.href = url;
                          a.download = `moodboard-${topic.replace(/\s+/g, "-").toLowerCase()}.json`;
                          a.click();
                          URL.revokeObjectURL(url);
                        }}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Export JSON
                      </Button>
                    </div>
                  </div>
                )}

                {/* Generate Button */}
                {!generatedMoodBoard && (
                  <Button
                    onClick={handleGenerateMoodBoard}
                    disabled={isGeneratingMoodBoard || !topic.trim()}
                    className="w-full"
                    size="lg"
                  >
                    {isGeneratingMoodBoard ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Generating Mood Board...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4 mr-2" />
                        Generate AI Mood Board
                      </>
                    )}
                  </Button>
                )}
              </div>
              </>)}
            </div>
          )}

          {/* Step 2: Topic Input (for non-moodboard formats) */}
          {currentStep === "topic" && selectedFormat !== "moodboard" && (
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

                {/* Output Language Selector */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Output Language</label>
                  <select
                    value={outputLanguage}
                    onChange={(e) => setOutputLanguage(e.target.value)}
                    className="w-full h-10 px-3 rounded-md border bg-background"
                  >
                    {LANGUAGES.map((lang) => (
                      <option key={lang.code} value={lang.code}>
                        {lang.name}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-muted-foreground">
                    All generated content will be in this language
                  </p>
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

                  {/* Query Enhancement Toggle - inline within filters */}
                  <div className="flex items-center justify-between p-3 rounded-lg border bg-card mt-3">
                    <div className="flex items-center gap-3">
                      <Sparkles className={`h-4 w-4 ${(enhanceQuery ?? true) ? "text-yellow-500" : "text-muted-foreground"}`} />
                      <div>
                        <p className="text-sm font-medium">Query Enhancement</p>
                        <p className="text-xs text-muted-foreground">
                          Improve source search with query expansion and HyDE
                        </p>
                      </div>
                    </div>
                    <div
                      onClick={() => handleEnhanceQueryChange(!(enhanceQuery ?? true))}
                      className={`w-10 h-5 rounded-full transition-colors flex items-center cursor-pointer ${
                        (enhanceQuery ?? true) ? "bg-primary justify-end" : "bg-muted-foreground/30 justify-start"
                      }`}
                    >
                      <div className="w-4 h-4 bg-white rounded-full shadow-sm mx-0.5" />
                    </div>
                  </div>
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

                {/* Auto Charts/Tables Toggle - Only for PPTX */}
                {selectedFormat === "pptx" && (
                  <div className="space-y-2">
                    <div
                      className={`flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-colors ${
                        autoCharts
                          ? "bg-primary/5 border-primary/30"
                          : "bg-muted/30 border-border hover:bg-muted/50"
                      }`}
                      onClick={() => setAutoCharts(!autoCharts)}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg ${autoCharts ? "bg-primary/10" : "bg-muted"}`}>
                          <BarChart3 className={`h-5 w-5 ${autoCharts ? "text-primary" : "text-muted-foreground"}`} />
                        </div>
                        <div>
                          <p className="text-sm font-medium">Auto-Generate Charts & Tables</p>
                          <p className="text-xs text-muted-foreground">
                            Automatically convert numeric data into native PowerPoint charts and tables
                          </p>
                        </div>
                      </div>
                      <div
                        className={`w-11 h-6 rounded-full transition-colors flex items-center ${
                          autoCharts ? "bg-primary justify-end" : "bg-muted-foreground/30 justify-start"
                        }`}
                      >
                        <div className="w-5 h-5 bg-white rounded-full shadow-sm mx-0.5" />
                      </div>
                    </div>
                    {autoCharts && (
                      <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800/50">
                        <Info className="h-4 w-4 text-blue-600 dark:text-blue-400 shrink-0 mt-0.5" />
                        <p className="text-xs text-blue-700 dark:text-blue-300">
                          When enabled, the AI will detect numeric patterns and structured data in content,
                          automatically rendering them as native PowerPoint charts (bar, line, pie) or tables
                          instead of bullet points.
                        </p>
                      </div>
                    )}
                  </div>
                )}

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

                      {/* Dual Mode Toggle */}
                      <div
                        className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                          dualModeEnabled
                            ? "bg-blue-500/5 border-blue-500/30"
                            : "bg-muted/30 border-border hover:bg-muted/50"
                        }`}
                        onClick={() => setDualModeEnabled(!dualModeEnabled)}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`p-1.5 rounded-lg ${dualModeEnabled ? "bg-blue-500/10" : "bg-muted"}`}>
                            <Sparkles className={`h-4 w-4 ${dualModeEnabled ? "text-blue-500" : "text-muted-foreground"}`} />
                          </div>
                          <div>
                            <p className="text-sm font-medium">Dual Mode (Documents + General AI)</p>
                            <p className="text-xs text-muted-foreground">
                              Combine document knowledge with general AI for richer content
                            </p>
                          </div>
                        </div>
                        <div
                          className={`w-10 h-5 rounded-full transition-colors flex items-center ${
                            dualModeEnabled ? "bg-blue-500 justify-end" : "bg-muted-foreground/30 justify-start"
                          }`}
                        >
                          <div className="w-4 h-4 bg-white rounded-full shadow-sm mx-0.5" />
                        </div>
                      </div>
                      {dualModeEnabled && (
                        <div className="flex items-center gap-2 pl-2">
                          <label className="text-xs text-muted-foreground">Blend mode:</label>
                          <select
                            value={dualModeBlend}
                            onChange={(e) => setDualModeBlend(e.target.value as "merged" | "docs_first")}
                            className="h-7 px-2 rounded border bg-background text-xs"
                          >
                            <option value="merged">Merged (AI synthesizes both sources)</option>
                            <option value="docs_first">Documents First (supplement with AI)</option>
                          </select>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* LLM Configuration */}
                <div className="space-y-3 p-4 rounded-lg border bg-muted/30">
                  <div className="flex items-center gap-2">
                    <Cog className="h-4 w-4 text-muted-foreground" />
                    <Label className="text-sm font-medium">LLM Configuration</Label>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1.5">
                      <label className="text-xs text-muted-foreground">Provider</label>
                      <select
                        value={selectedProviderId || ""}
                        onChange={(e) => {
                          setSelectedProviderId(e.target.value || null);
                          setSelectedModelName(null);
                        }}
                        className="w-full h-9 px-3 rounded-md border bg-background text-sm"
                      >
                        <option value="">System Default</option>
                        {(providersData?.providers || []).filter((p: any) => p.is_active).map((p: any) => (
                          <option key={p.id} value={p.id}>{p.name} ({p.provider_type})</option>
                        ))}
                      </select>
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs text-muted-foreground">Model</label>
                      <select
                        value={selectedModelName || ""}
                        onChange={(e) => setSelectedModelName(e.target.value || null)}
                        className="w-full h-9 px-3 rounded-md border bg-background text-sm"
                      >
                        <option value="">Auto (provider default)</option>
                        {selectedProviderId && (() => {
                          const provider = (providersData?.providers || []).find((p: any) => p.id === selectedProviderId);
                          if (provider?.default_chat_model) {
                            return <option key={provider.default_chat_model} value={provider.default_chat_model}>{provider.default_chat_model}</option>;
                          }
                          return null;
                        })()}
                      </select>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-muted-foreground">Temperature</label>
                      <select
                        value={temperatureMode}
                        onChange={(e) => setTemperatureMode(e.target.value as "auto" | "manual")}
                        className="h-7 px-2 rounded border bg-background text-xs"
                      >
                        <option value="auto">Auto (format-optimized)</option>
                        <option value="manual">Manual</option>
                      </select>
                    </div>
                    {temperatureMode === "manual" && (
                      <div className="flex items-center gap-3">
                        <Slider
                          value={[manualTemperature]}
                          onValueChange={([v]) => setManualTemperature(v)}
                          min={0}
                          max={1.5}
                          step={0.05}
                          className="flex-1"
                        />
                        <span className="text-xs font-mono w-8 text-right">{manualTemperature.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Theme Selector - Hidden when using PPTX template */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Document Theme</label>
                    {selectedFormat === "pptx" && selectedPptxTemplate && (
                      <span className="text-xs text-blue-600 flex items-center gap-1">
                        <Info className="h-3 w-3" />
                        Using template styling
                      </span>
                    )}
                    {isLoadingThemeSuggestion && !selectedPptxTemplate && (
                      <span className="text-xs text-muted-foreground flex items-center gap-1">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Suggesting theme...
                      </span>
                    )}
                  </div>
                  {/* Show message when PPTX template is selected */}
                  {selectedFormat === "pptx" && selectedPptxTemplate ? (
                    <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 dark:bg-blue-950/20 dark:border-blue-800">
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        Theme colors are disabled when using an uploaded PPTX template.
                        The generated presentation will use the template&apos;s styling.
                      </p>
                    </div>
                  ) : (
                    <>
                  {themeSuggestionReason && (
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      <Sparkles className="h-3 w-3" />
                      AI suggests: {themeSuggestionReason}
                    </p>
                  )}
                  <div className="grid grid-cols-3 gap-2">
                    {(themesData?.themes || [
                      // Row 1: Professional themes
                      { key: "business", name: "Business Professional", description: "Clean corporate look", primary: "#1E3A5F", secondary: "#3D5A80", accent: "#E0E1DD", text: "#2D3A45" },
                      { key: "elegant", name: "Elegant & Refined", description: "Sophisticated executive", primary: "#2C3E50", secondary: "#7F8C8D", accent: "#BDC3C7", text: "#2C3E50" },
                      { key: "academic", name: "Academic & Scholarly", description: "Research & education", primary: "#2C3E50", secondary: "#8E44AD", accent: "#ECF0F1", text: "#2C3E50" },
                      // Row 2: Modern/Minimal themes
                      { key: "modern", name: "Modern Minimal", description: "Sleek contemporary", primary: "#212529", secondary: "#495057", accent: "#00B4D8", text: "#212529" },
                      { key: "minimalist", name: "Ultra Minimalist", description: "Maximum whitespace", primary: "#333333", secondary: "#666666", accent: "#F5F5F5", text: "#222222" },
                      { key: "dark", name: "Dark Mode", description: "Low-light viewing", primary: "#1A1A2E", secondary: "#16213E", accent: "#0F3460", text: "#E4E4E4" },
                      // Row 3: Creative themes
                      { key: "creative", name: "Creative & Bold", description: "Vibrant marketing", primary: "#6B4C9A", secondary: "#9B6B9E", accent: "#F4E4BA", text: "#333333" },
                      { key: "vibrant", name: "Vibrant & Energetic", description: "High-energy content", primary: "#E74C3C", secondary: "#F39C12", accent: "#FDF2E9", text: "#2D3436" },
                      { key: "colorful", name: "Colorful & Fun", description: "Engaging & memorable", primary: "#FF6B6B", secondary: "#4ECDC4", accent: "#FFE66D", text: "#2C3E50" },
                      // Row 4: Specialty themes
                      { key: "tech", name: "Tech & Digital", description: "Digital aesthetics", primary: "#0984E3", secondary: "#6C5CE7", accent: "#DFE6E9", text: "#2D3436" },
                      { key: "nature", name: "Nature & Organic", description: "Earthy sustainable", primary: "#2D5016", secondary: "#5A7D3A", accent: "#F5F0E1", text: "#2D3A2E" },
                      { key: "warm", name: "Warm & Inviting", description: "Cozy community feel", primary: "#D35400", secondary: "#E67E22", accent: "#FDEBD0", text: "#2C3E50" },
                    ] as ThemeInfo[]).map((theme: ThemeInfo) => (
                      <div
                        key={theme.key}
                        onClick={() => {
                          setSelectedTheme(theme.key);
                          setThemeSuggestionReason(null);
                          setThemeManuallyChanged(true);  // User manually selected a theme
                        }}
                        className={`p-2 rounded-lg border cursor-pointer transition-all ${
                          selectedTheme === theme.key
                            ? "border-primary ring-2 ring-primary/20"
                            : "border-border hover:border-primary/50"
                        }`}
                      >
                        {/* Theme preview mini-card */}
                        <div
                          className="w-full h-8 rounded mb-2 relative overflow-hidden"
                          style={{
                            background: theme.key === "dark"
                              ? `linear-gradient(135deg, ${theme.primary} 0%, ${theme.secondary} 100%)`
                              : theme.key === "colorful" || theme.key === "vibrant" || theme.key === "creative"
                              ? `linear-gradient(135deg, ${theme.primary} 0%, ${theme.secondary} 50%, ${theme.accent} 100%)`
                              : theme.primary
                          }}
                        >
                          {/* Accent bar */}
                          <div
                            className="absolute bottom-0 left-0 right-0 h-1"
                            style={{ backgroundColor: theme.accent }}
                          />
                        </div>
                        {/* Color swatches */}
                        <div className="flex items-center gap-1 mb-1">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: theme.primary }}
                          />
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: theme.secondary }}
                          />
                          <div
                            className="w-3 h-3 rounded-full border"
                            style={{ backgroundColor: theme.accent }}
                          />
                        </div>
                        <p className="text-xs font-medium truncate">{theme.name}</p>
                        <p className="text-[10px] text-muted-foreground truncate">{theme.description}</p>
                        {themeSuggestionReason && selectedTheme === theme.key && (
                          <span className="text-[10px] text-primary mt-1 inline-block">✨ AI Pick</span>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Custom Colors Toggle */}
                  <div className="mt-4 p-3 rounded-lg border border-dashed">
                    <div className="flex items-center gap-2 mb-2">
                      <input
                        type="checkbox"
                        id="useCustomColors"
                        checked={useCustomColors ?? false}
                        onChange={(e) => {
                          setUseCustomColors(e.target.checked);
                          setThemeManuallyChanged(true);
                        }}
                        className="h-4 w-4 rounded border-border"
                      />
                      <label htmlFor="useCustomColors" className="text-sm font-medium">
                        Use custom colors
                      </label>
                      <span className="text-xs text-muted-foreground">(override theme)</span>
                    </div>
                    {useCustomColors && (
                      <div className="grid grid-cols-3 gap-3 mt-3">
                        <div>
                          <label className="block text-xs text-muted-foreground mb-1">Primary</label>
                          <div className="flex items-center gap-2">
                            <input
                              type="color"
                              value={customPrimaryColor}
                              onChange={(e) => setCustomPrimaryColor(e.target.value)}
                              className="w-8 h-8 rounded cursor-pointer border-0"
                            />
                            <input
                              type="text"
                              value={customPrimaryColor}
                              onChange={(e) => setCustomPrimaryColor(e.target.value)}
                              className="w-20 text-xs px-2 py-1 border rounded"
                              placeholder="#000000"
                            />
                          </div>
                        </div>
                        <div>
                          <label className="block text-xs text-muted-foreground mb-1">Secondary</label>
                          <div className="flex items-center gap-2">
                            <input
                              type="color"
                              value={customSecondaryColor}
                              onChange={(e) => setCustomSecondaryColor(e.target.value)}
                              className="w-8 h-8 rounded cursor-pointer border-0"
                            />
                            <input
                              type="text"
                              value={customSecondaryColor}
                              onChange={(e) => setCustomSecondaryColor(e.target.value)}
                              className="w-20 text-xs px-2 py-1 border rounded"
                              placeholder="#000000"
                            />
                          </div>
                        </div>
                        <div>
                          <label className="block text-xs text-muted-foreground mb-1">Accent</label>
                          <div className="flex items-center gap-2">
                            <input
                              type="color"
                              value={customAccentColor}
                              onChange={(e) => setCustomAccentColor(e.target.value)}
                              className="w-8 h-8 rounded cursor-pointer border-0"
                            />
                            <input
                              type="text"
                              value={customAccentColor}
                              onChange={(e) => setCustomAccentColor(e.target.value)}
                              className="w-20 text-xs px-2 py-1 border rounded"
                              placeholder="#000000"
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                    </>
                  )}
                </div>

                {/* Font & Layout Selection (only for PPTX/DOCX/PDF) - Hide font when using PPTX template */}
                {(selectedFormat === "pptx" || selectedFormat === "docx" || selectedFormat === "pdf") && Object.keys(availableFonts).length > 0 && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    {/* Font Family Selection - Hidden when using PPTX template (template fonts will be used) */}
                    {selectedFormat === "pptx" && selectedPptxTemplate ? (
                      <div>
                        <label className="block text-sm font-medium mb-2">Font Style</label>
                        <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 dark:bg-blue-950/20 dark:border-blue-800">
                          <p className="text-sm text-blue-700 dark:text-blue-300">
                            Font styling is inherited from the template.
                          </p>
                        </div>
                      </div>
                    ) : (
                    <div>
                      <label className="block text-sm font-medium mb-2">Font Style</label>
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(availableFonts).map(([key, font]: [string, any]) => (
                          <div
                            key={key}
                            onClick={() => {
                              setSelectedFontFamily(key);
                              setThemeManuallyChanged(true);
                            }}
                            className={`p-2 rounded-lg border cursor-pointer transition-all text-center ${
                              selectedFontFamily === key
                                ? "border-primary ring-2 ring-primary/20"
                                : "border-border hover:border-primary/50"
                            }`}
                          >
                            <p className="text-sm font-medium">{font.name}</p>
                            <p className="text-xs text-muted-foreground">{font.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                    )}

                    {/* Layout Selection (PPTX only) */}
                    {selectedFormat === "pptx" && Object.keys(availableLayouts).length > 0 && (
                      <div>
                        <label className="block text-sm font-medium mb-2">Slide Layout</label>
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(availableLayouts).map(([key, layout]: [string, any]) => (
                            <div
                              key={key}
                              onClick={() => {
                                setSelectedLayout(key);
                                setThemeManuallyChanged(true);
                              }}
                              className={`p-2 rounded-lg border cursor-pointer transition-all text-center ${
                                selectedLayout === key
                                  ? "border-primary ring-2 ring-primary/20"
                                  : "border-border hover:border-primary/50"
                              }`}
                            >
                              <p className="text-sm font-medium">{layout.name}</p>
                              <p className="text-xs text-muted-foreground">{layout.description}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Slide Transitions Toggle (PPTX only) */}
                {selectedFormat === "pptx" && (
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="enableAnimations"
                        checked={enableAnimations ?? false}
                        onChange={(e) => {
                          setEnableAnimations(e.target.checked);
                          setThemeManuallyChanged(true);
                        }}
                        className="h-4 w-4 rounded border-border"
                      />
                      <label htmlFor="enableAnimations" className="text-sm font-medium">
                        Enable slide transitions
                      </label>
                      <span className="text-xs text-muted-foreground">(fade, wipe, push effects between slides)</span>
                    </div>
                    {enableAnimations && (
                      <div className="ml-6 space-y-3">
                        <div className="flex items-center gap-4">
                          <label className="text-sm">Transition speed:</label>
                          <select
                            value={animationSpeed}
                            onChange={(e) => setAnimationSpeed(e.target.value as 'very_slow' | 'slow' | 'med' | 'fast' | 'very_fast')}
                            className="px-3 py-1 rounded border border-border bg-background text-sm"
                            disabled={useCustomDuration}
                          >
                            <option value="very_slow">Very Slow (2s)</option>
                            <option value="slow">Slow (1.5s)</option>
                            <option value="med">Medium (0.75s)</option>
                            <option value="fast">Fast (0.4s)</option>
                            <option value="very_fast">Very Fast (0.2s)</option>
                          </select>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="useCustomDuration"
                            checked={useCustomDuration ?? false}
                            onChange={(e) => setUseCustomDuration(e.target.checked)}
                            className="h-4 w-4 rounded border-border"
                          />
                          <label htmlFor="useCustomDuration" className="text-sm">Custom transition duration</label>
                        </div>
                        {useCustomDuration && (
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <label className="text-sm">Duration: {customDuration}ms ({(customDuration/1000).toFixed(1)}s)</label>
                            </div>
                            <input
                              type="range"
                              min={200}
                              max={3000}
                              step={100}
                              value={customDuration}
                              onChange={(e) => setCustomDuration(Number(e.target.value))}
                              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>0.2s</span>
                              <span>3s</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Quality Review Toggle - available for all formats */}
                <div className="flex items-center gap-2 mt-4 pt-4 border-t">
                  <input
                    type="checkbox"
                    id="enableQualityReview"
                    checked={enableQualityReview ?? false}
                    onChange={(e) => setEnableQualityReview(e.target.checked)}
                    className="h-4 w-4 rounded border-border"
                  />
                  <label htmlFor="enableQualityReview" className="text-sm font-medium">
                    Enable AI quality review
                  </label>
                  <span className="text-xs text-muted-foreground">
                    {selectedFormat === "pptx"
                      ? "(reviews slides for text overflow, layout issues & auto-fixes)"
                      : "(auto-improves low-quality sections)"}
                  </span>
                </div>

                {/* AI Proofreading with CriticAgent */}
                <div className="mt-4 pt-4 border-t space-y-4">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="enableCriticReview"
                      checked={enableCriticReview ?? false}
                      onChange={(e) => setEnableCriticReview(e.target.checked)}
                      className="h-4 w-4 rounded border-border"
                    />
                    <label htmlFor="enableCriticReview" className="text-sm font-medium">
                      Enable AI Proofreading
                    </label>
                    <span className="text-xs text-muted-foreground">(reviews and auto-fixes quality issues)</span>
                  </div>

                  {enableCriticReview && (
                    <div className="ml-6 space-y-4 bg-muted/30 p-4 rounded-lg">
                      {/* Quality Threshold Slider */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <label className="text-sm font-medium">Quality Threshold</label>
                          <span className="text-sm text-muted-foreground">{Math.round(qualityThreshold * 100)}%</span>
                        </div>
                        <input
                          type="range"
                          min="0.6"
                          max="0.9"
                          step="0.05"
                          value={qualityThreshold}
                          onChange={(e) => setQualityThreshold(parseFloat(e.target.value))}
                          className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
                        />
                        <p className="text-xs text-muted-foreground">
                          Content scoring below this threshold will be automatically improved
                        </p>
                      </div>

                      {/* Fix Options */}
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="fixStyling"
                            checked={fixStyling ?? false}
                            onChange={(e) => setFixStyling(e.target.checked)}
                            className="h-4 w-4 rounded border-border"
                          />
                          <label htmlFor="fixStyling" className="text-sm">
                            Fix styling and formatting issues
                          </label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            id="fixIncomplete"
                            checked={fixIncomplete ?? false}
                            onChange={(e) => setFixIncomplete(e.target.checked)}
                            className="h-4 w-4 rounded border-border"
                          />
                          <label htmlFor="fixIncomplete" className="text-sm">
                            Complete incomplete bullet points and sentences
                          </label>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Speaker Notes Options (PPTX only) */}
                {selectedFormat === "pptx" && (
                  <div className="mt-4 pt-4 border-t space-y-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="includeNotesExplanation"
                        checked={includeNotesExplanation ?? false}
                        onChange={(e) => setIncludeNotesExplanation(e.target.checked)}
                        className="h-4 w-4 rounded border-border"
                      />
                      <label htmlFor="includeNotesExplanation" className="text-sm font-medium">
                        Include AI explanations in speaker notes
                      </label>
                    </div>
                    <p className="text-xs text-muted-foreground ml-6">
                      Adds detailed AI reasoning and source references to each slide&apos;s speaker notes.
                      Title slide always includes generation info (model, date, theme).
                    </p>
                  </div>
                )}

                {/* Vision Analysis (PPTX only) */}
                {selectedFormat === "pptx" && (
                  <div className="mt-4 pt-4 border-t space-y-4">
                    <div>
                      <h3 className="text-sm font-medium flex items-center gap-2">
                        <Eye className="h-4 w-4" />
                        Vision Analysis
                      </h3>
                      <p className="text-xs text-muted-foreground">
                        Use AI vision to analyze templates and review generated slides
                      </p>
                    </div>

                    {/* Template Vision Analysis */}
                    <div className="space-y-2 bg-muted/30 p-3 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium">Template Vision Analysis</label>
                          <p className="text-xs text-muted-foreground">
                            Analyze template slides visually to learn styling and layout
                          </p>
                        </div>
                        <select
                          className="h-8 px-2 text-sm rounded-md border bg-background"
                          value={enableTemplateVisionAnalysis === null ? "system" : enableTemplateVisionAnalysis ? "enabled" : "disabled"}
                          onChange={(e) => {
                            if (e.target.value === "system") setEnableTemplateVisionAnalysis(null);
                            else if (e.target.value === "enabled") setEnableTemplateVisionAnalysis(true);
                            else setEnableTemplateVisionAnalysis(false);
                          }}
                        >
                          <option value="system">Use System Default</option>
                          <option value="enabled">Enabled</option>
                          <option value="disabled">Disabled</option>
                        </select>
                      </div>

                      {enableTemplateVisionAnalysis && (
                        <div className="mt-2 space-y-1">
                          <label className="text-xs text-muted-foreground">Vision Model:</label>
                          <select
                            className="w-full h-8 px-2 text-sm rounded-md border bg-background"
                            value={templateVisionModel}
                            onChange={(e) => setTemplateVisionModel(e.target.value)}
                          >
                            <option value="auto">Auto (Use default vision model)</option>
                            <option value="gpt-4o">GPT-4o</option>
                            <option value="gpt-4o-mini">GPT-4o Mini</option>
                            <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                            <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                            <option value="llava">LLaVA (Local)</option>
                          </select>
                        </div>
                      )}
                    </div>

                    {/* Vision-Based Slide Review */}
                    <div className="space-y-2 bg-muted/30 p-3 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium">Vision-Based Slide Review</label>
                          <p className="text-xs text-muted-foreground">
                            Review generated slides for visual issues (overlaps, truncation)
                          </p>
                        </div>
                        <select
                          className="h-8 px-2 text-sm rounded-md border bg-background"
                          value={enableVisionReview === null ? "system" : enableVisionReview ? "enabled" : "disabled"}
                          onChange={(e) => {
                            if (e.target.value === "system") setEnableVisionReview(null);
                            else if (e.target.value === "enabled") setEnableVisionReview(true);
                            else setEnableVisionReview(false);
                          }}
                        >
                          <option value="system">Use System Default</option>
                          <option value="enabled">Enabled</option>
                          <option value="disabled">Disabled</option>
                        </select>
                      </div>

                      {enableVisionReview && (
                        <div className="mt-2 space-y-1">
                          <label className="text-xs text-muted-foreground">Vision Model:</label>
                          <select
                            className="w-full h-8 px-2 text-sm rounded-md border bg-background"
                            value={visionReviewModel}
                            onChange={(e) => setVisionReviewModel(e.target.value)}
                          >
                            <option value="auto">Auto (Use default vision model)</option>
                            <option value="gpt-4o">GPT-4o</option>
                            <option value="gpt-4o-mini">GPT-4o Mini</option>
                            <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                            <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                            <option value="llava">LLaVA (Local)</option>
                          </select>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Advanced LLM Settings - Phase 15 Optimization */}
                <div className="mt-4 pt-4 border-t space-y-4">
                  <div>
                    <h3 className="text-sm font-medium flex items-center gap-2">
                      <Cog className="h-4 w-4" />
                      Advanced LLM Settings (Optional)
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      Override content generation temperature for this document
                    </p>
                  </div>

                  <div className="space-y-3 bg-muted/30 p-3 rounded-lg">
                    {(() => {
                      // Calculate optimized temperature (use default model for now since we don't expose provider selection)
                      const optimizedTemp = 0.7; // Default for large models, will be calculated by backend
                      const currentTemp = overrideTemperature ?? optimizedTemp;
                      const isManualOverride = overrideTemperature !== null;

                      return (
                        <>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <Label className="text-sm">Content Quality Temperature</Label>
                              <span className="text-sm font-medium">{currentTemp.toFixed(2)}</span>
                            </div>
                            <Slider
                              value={[currentTemp]}
                              onValueChange={([value]) => setOverrideTemperature(value)}
                              min={0}
                              max={1}
                              step={0.05}
                              className="w-full"
                            />
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>Precise (0.0)</span>
                              <span>Balanced (0.7)</span>
                              <span>Creative (1.0)</span>
                            </div>
                          </div>

                          <div className="text-xs space-y-2">
                            {!isManualOverride ? (
                              <div className="flex items-center gap-2 text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/30 p-2 rounded">
                                <CheckCircle className="h-3 w-3 flex-shrink-0" />
                                <span>Using system default temperature (optimized per model type)</span>
                              </div>
                            ) : (
                              <div className="space-y-2">
                                <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/30 p-2 rounded">
                                  <Info className="h-3 w-3 flex-shrink-0" />
                                  <span>Custom temperature override active: {currentTemp.toFixed(2)}</span>
                                </div>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="w-full h-8 text-xs"
                                  onClick={() => setOverrideTemperature(null)}
                                >
                                  Reset to System Default
                                </Button>
                              </div>
                            )}

                            <div className="text-muted-foreground space-y-1 pt-2 border-t">
                              <p className="font-medium">Temperature Guide:</p>
                              <ul className="list-disc list-inside space-y-0.5 ml-2">
                                <li><strong>0.2-0.4:</strong> Precise, factual content (legal, medical, technical docs)</li>
                                <li><strong>0.5-0.7:</strong> Balanced, professional content (reports, presentations)</li>
                                <li><strong>0.8-1.0:</strong> Creative, varied content (marketing, brainstorming)</li>
                              </ul>
                              <p className="pt-1 text-xs">
                                <strong>Note:</strong> System automatically optimizes temperature based on your selected LLM provider.
                                Small models (Llama, Qwen) use lower temperature to reduce hallucinations.
                              </p>
                            </div>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                </div>

                {/* Built-in Templates (PPTX/DOCX/XLSX) */}
                {(selectedFormat === "pptx" || selectedFormat === "docx" || selectedFormat === "xlsx") && (
                  <div className="mt-4 pt-4 border-t space-y-3">
                    <BuiltInTemplateSelector
                      fileType={selectedFormat}
                      onSelectTemplate={(template) => {
                        setSelectedBuiltInTemplate(template);
                        // Clear uploaded template selection when built-in is selected
                        if (selectedFormat === "pptx") {
                          setSelectedPptxTemplate(null);
                        }
                      }}
                      selectedTemplateId={selectedBuiltInTemplate?.id}
                    />
                    {selectedBuiltInTemplate && (
                      <div className="flex items-start gap-2 p-3 rounded-lg bg-primary/5 border border-primary/20">
                        <Palette className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <p className="text-xs text-primary font-medium">
                            Using: {selectedBuiltInTemplate.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {selectedBuiltInTemplate.description}
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-xs"
                          onClick={() => setSelectedBuiltInTemplate(null)}
                        >
                          Clear
                        </Button>
                      </div>
                    )}
                  </div>
                )}

                {/* PPTX Template from Uploaded Documents (PPTX only) */}
                {selectedFormat === "pptx" && (
                  <div className="mt-4 pt-4 border-t space-y-3">
                    <div>
                      <h3 className="text-sm font-medium">Or Use Uploaded PPTX as Template</h3>
                      <p className="text-xs text-muted-foreground">
                        Select a previously uploaded presentation to inherit its styling and design
                      </p>
                    </div>

                    {pptxTemplates.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <label className="text-xs text-muted-foreground">Available Templates:</label>
                          {isFetchingSuggestions && (
                            <span className="text-xs text-muted-foreground flex items-center gap-1">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Analyzing templates...
                            </span>
                          )}
                          {!isFetchingSuggestions && templateSuggestions.length > 0 && (
                            <span className="text-xs text-primary flex items-center gap-1">
                              <Sparkles className="h-3 w-3" />
                              AI recommendations available
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <div
                            onClick={() => setSelectedPptxTemplate(null)}
                            className={`p-3 rounded-lg border cursor-pointer transition-all ${
                              selectedPptxTemplate === null
                                ? "border-primary ring-2 ring-primary/20"
                                : "border-border hover:border-primary/50"
                            }`}
                          >
                            <p className="text-sm font-medium">No Template</p>
                            <p className="text-xs text-muted-foreground">Use default styling</p>
                          </div>
                          {/* Sort templates: AI-recommended first (by score descending), then others */}
                          {[...pptxTemplates]
                            .sort((a, b) => {
                              const suggestionA = templateSuggestions.find(s => s.template_id === a.template_id);
                              const suggestionB = templateSuggestions.find(s => s.template_id === b.template_id);
                              const scoreA = suggestionA?.score ?? -1;
                              const scoreB = suggestionB?.score ?? -1;
                              return scoreB - scoreA;
                            })
                            .map((template) => {
                              const suggestion = templateSuggestions.find(s => s.template_id === template.template_id);
                              const isRecommended = suggestion && suggestion.score >= 70;
                              return (
                                <div
                                  key={template.template_id}
                                  onClick={() => setSelectedPptxTemplate(template.template_id)}
                                  className={`p-3 rounded-lg border cursor-pointer transition-all relative ${
                                    selectedPptxTemplate === template.template_id
                                      ? "border-primary ring-2 ring-primary/20"
                                      : isRecommended
                                        ? "border-green-500/50 hover:border-green-500"
                                        : "border-border hover:border-primary/50"
                                  }`}
                                >
                                  {isRecommended && (
                                    <div className="absolute -top-2 -right-2 bg-green-500 text-white text-[10px] px-1.5 py-0.5 rounded-full flex items-center gap-0.5">
                                      <Sparkles className="h-2.5 w-2.5" />
                                      {suggestion.score}%
                                    </div>
                                  )}
                                  <div className="flex items-center justify-between">
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm font-medium truncate">
                                        {template.filename || `Template ${template.template_id.slice(0, 8)}`}
                                      </p>
                                      <p className="text-xs text-muted-foreground">
                                        {template.slide_count} slides • {new Date(template.created_at).toLocaleDateString()}
                                      </p>
                                      {suggestion && suggestion.reason && (
                                        <p className="text-xs text-green-600 dark:text-green-400 mt-1 line-clamp-2">
                                          {suggestion.reason}
                                        </p>
                                      )}
                                    </div>
                                    {/* Info icon - these are uploaded documents, not deletable from here */}
                                    <div
                                      className="p-1 rounded text-muted-foreground"
                                      title="This is an uploaded document that can be used as a template"
                                    >
                                      <FileText className="h-3 w-3" />
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    )}

                    {pptxTemplates.length === 0 && (
                      <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/50 border">
                        <FileText className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
                        <p className="text-xs text-muted-foreground">
                          No PPTX documents found. Upload presentations in the Documents section to use them as templates.
                        </p>
                      </div>
                    )}

                    {selectedPptxTemplate && (
                      <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800/50">
                        <Sparkles className="h-4 w-4 text-blue-600 dark:text-blue-400 shrink-0 mt-0.5" />
                        <p className="text-xs text-blue-700 dark:text-blue-300">
                          The generated presentation will use the slide master, fonts, and colors from your selected template.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Step 3: Outline Review */}
          {currentStep === "outline" && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">Review & Edit Sections</h2>
                  <p className="text-muted-foreground">
                    Toggle sections on/off, edit content, or add new sections
                  </p>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-muted-foreground">
                    {outline.filter(s => s.approved).length} of {outline.length} sections selected
                  </span>
                  <Button onClick={handleAddSection} variant="outline" size="sm">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Section
                  </Button>
                </div>
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
                    value={documentTitle || ""}
                    onChange={(e) => setDocumentTitle(e.target.value)}
                    placeholder="Enter document title"
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Sections</label>
                  </div>
                  {outline.map((section, index) => (
                    <div
                      key={section.id}
                      className={`flex items-start gap-3 p-4 rounded-lg border transition-colors ${
                        section.approved ? 'bg-muted/30' : 'bg-muted/10 opacity-60'
                      }`}
                    >
                      <div className="flex items-center gap-2 pt-2">
                        <input
                          type="checkbox"
                          checked={section.approved ?? false}
                          onChange={() => handleToggleSectionApproval(section.id)}
                          className="h-4 w-4 rounded border-gray-300"
                        />
                        <span className="text-sm font-medium text-muted-foreground">{index + 1}</span>
                      </div>
                      <div className="flex-1 space-y-2">
                        <Input
                          value={section.title || ""}
                          onChange={(e) =>
                            handleUpdateSection(section.id, "title", e.target.value)
                          }
                          placeholder="Section title"
                          className="font-medium"
                          disabled={!section.approved}
                        />
                        <Input
                          value={section.description || ""}
                          onChange={(e) =>
                            handleUpdateSection(section.id, "description", e.target.value)
                          }
                          placeholder="Brief description of this section"
                          className="text-sm"
                          disabled={!section.approved}
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

                {outline.filter(s => s.approved).length === 0 && outline.length > 0 && (
                  <div className="text-center py-4 text-muted-foreground">
                    <AlertCircle className="h-6 w-6 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Please select at least one section to generate content for.</p>
                  </div>
                )}

                <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-4 text-sm">
                  <Info className="h-4 w-4 inline mr-2 text-blue-500" />
                  <span className="text-blue-700 dark:text-blue-300">
                    Unchecked sections will be skipped during generation. You can edit titles and descriptions to guide the AI.
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Section Plans Review (Pre-generation) */}

          {/* Step 5: Content Generation */}
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
                  {/* Review & Edit option for PPTX and DOCX */}
                  {(selectedFormat === "pptx" || selectedFormat === "docx") && (
                    <div className="flex flex-col items-center gap-2 mt-4 p-4 bg-muted/50 rounded-lg">
                      <p className="text-sm text-muted-foreground text-center max-w-md">
                        Want to review and edit the content before downloading?
                        You can modify slides, sections, and text before final generation.
                      </p>
                      <Button
                        variant="outline"
                        onClick={async () => {
                          try {
                            // Create a review session with the generated content
                            const contentType = selectedFormat as "pptx" | "docx";

                            // Build content from sections
                            if (contentType === "pptx") {
                              const presentationContent = {
                                title: documentTitle,
                                subtitle: context || undefined,
                                slides: job.sections.map((s, i) => {
                                  // Split content into bullets and limit to 8 max
                                  const allBullets = (s.content || "").split("\n").filter(Boolean);
                                  const limitedBullets = allBullets.slice(0, 8);
                                  return {
                                    slide_number: i + 1,
                                    layout: "title_content",
                                    title: s.title,
                                    bullets: limitedBullets.map((text: string) => ({ text, sub_bullets: [] as string[] })),
                                    status: "draft" as const,
                                  };
                                }),
                              };
                              const result = await api.createContentReviewSession({
                                content_type: contentType,
                                job_id: currentJobId || undefined,
                                presentation: presentationContent,
                              });
                              window.location.href = `/dashboard/create/review?session=${result.session_id}&type=${contentType}&job=${currentJobId}`;
                            } else {
                              const documentContent = {
                                title: documentTitle,
                                sections: job.sections.map((s, i) => ({
                                  section_number: i + 1,
                                  title: s.title,
                                  content: s.content || "",
                                  level: 1,
                                  status: "draft" as const,
                                })),
                              };
                              const result = await api.createContentReviewSession({
                                content_type: contentType,
                                job_id: currentJobId || undefined,
                                document: documentContent,
                              });
                              window.location.href = `/dashboard/create/review?session=${result.session_id}&type=${contentType}&job=${currentJobId}`;
                            }
                          } catch (error) {
                            console.error("Failed to create review session:", error);
                            toast.error("Failed to start review session");
                          }
                        }}
                      >
                        <Edit3 className="h-4 w-4 mr-2" />
                        Review & Edit Before Download
                      </Button>
                    </div>
                  )}
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
              ) : jobError ? (
                // Error state - job not found (e.g., after server restart)
                <>
                  <AlertCircle className="h-16 w-16 text-yellow-500" />
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">Session Expired</h2>
                    <p className="text-muted-foreground mt-2">
                      The generation job was not found. This may happen after a server restart.
                    </p>
                  </div>
                  <Button onClick={handleStartNew}>
                    <Plus className="h-4 w-4 mr-2" />
                    Start New Document
                  </Button>
                </>
              ) : (
                // Unknown state - shouldn't happen but handle gracefully
                <>
                  <AlertCircle className="h-16 w-16 text-muted-foreground" />
                  <div className="text-center">
                    <h2 className="text-xl font-semibold">No Active Generation</h2>
                    <p className="text-muted-foreground mt-2">
                      Start a new document to continue.
                    </p>
                  </div>
                  <Button onClick={handleStartNew}>
                    <Plus className="h-4 w-4 mr-2" />
                    Start New Document
                  </Button>
                </>
              )}
            </div>
          )}

          {/* Step 5: Download */}
          {currentStep === "download" && (
            <div className="space-y-6">
              <div className="flex flex-col items-center justify-center py-8 space-y-4">
                <div className="p-4 rounded-full bg-primary/10">
                  <Download className="h-12 w-12 text-primary" />
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

              {/* Section Review - allow reviewing generated sections */}
              {job?.sections && job.sections.length > 0 && (
                <div className="border-t pt-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="font-medium">Review Sections</h3>
                      <p className="text-sm text-muted-foreground">
                        Click on a section to review and provide feedback
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleSpellCheck}
                        disabled={isCheckingSpelling}
                      >
                        {isCheckingSpelling ? (
                          <Loader2 className="h-4 w-4 animate-spin mr-1" />
                        ) : (
                          <SpellCheck className="h-4 w-4 mr-1" />
                        )}
                        Check Spelling
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowSectionReview(!showSectionReview)}
                      >
                        {showSectionReview ? "Hide" : "Show"} Sections
                      </Button>
                    </div>
                  </div>

                  {/* Spell Check Results */}
                  {spellCheckResult && spellCheckResult.issues.length > 0 && (
                    <div className="mb-4 p-4 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800">
                      <div className="flex items-start gap-3">
                        <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
                        <div className="flex-1">
                          <h4 className="font-medium text-yellow-800 dark:text-yellow-300">
                            {spellCheckResult.issues.length} Spelling {spellCheckResult.issues.length === 1 ? 'Issue' : 'Issues'} Found
                          </h4>
                          <div className="mt-2 space-y-2">
                            {spellCheckResult.issues.slice(0, 5).map((issue, idx) => (
                              <div key={idx} className="text-sm">
                                <span className="font-mono bg-yellow-100 dark:bg-yellow-800 px-1 rounded text-yellow-900 dark:text-yellow-200">
                                  {issue.word}
                                </span>
                                {issue.suggestion && (
                                  <span className="text-muted-foreground ml-2">
                                    Suggestion: {issue.suggestion}
                                  </span>
                                )}
                              </div>
                            ))}
                            {spellCheckResult.issues.length > 5 && (
                              <p className="text-xs text-muted-foreground">
                                ...and {spellCheckResult.issues.length - 5} more
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {showSectionReview && (
                    <div className="space-y-3">
                      {job.sections.map((section: GenerationSection) => (
                        <div
                          key={section.id}
                          className="p-4 rounded-lg border hover:border-primary/50 cursor-pointer transition-colors"
                          onClick={() => setFeedbackSection(section)}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h4 className="font-medium text-sm">{section.title}</h4>
                              <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                {section.content.slice(0, 200)}...
                              </p>
                            </div>
                            <div className="flex items-center gap-2 ml-4">
                              {section.metadata?.quality_score !== undefined && (
                                <span className={`text-xs px-2 py-1 rounded ${
                                  section.metadata.quality_score >= 0.8
                                    ? 'bg-green-100 text-green-700'
                                    : section.metadata.quality_score >= 0.6
                                    ? 'bg-yellow-100 text-yellow-700'
                                    : 'bg-red-100 text-red-700'
                                }`}>
                                  {(section.metadata.quality_score * 100).toFixed(0)}%
                                </span>
                              )}
                              {section.approved ? (
                                <span className="text-xs px-2 py-1 rounded bg-green-100 text-green-700">
                                  Approved
                                </span>
                              ) : section.feedback ? (
                                <span className="text-xs px-2 py-1 rounded bg-yellow-100 text-yellow-700">
                                  Needs Revision
                                </span>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Navigation Buttons */}
      {currentStep !== "download" && (
        <div className="flex justify-between">
          <div className="flex gap-2">
            <Button
              onClick={handleBack}
              variant="outline"
              disabled={currentStepIndex === 0}
            >
              <ChevronLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            {currentStep === "topic" && selectedFormat && (
              <Button
                onClick={() => setShowSaveTemplate(true)}
                variant="ghost"
                size="sm"
                className="text-muted-foreground"
              >
                <Save className="h-4 w-4 mr-2" />
                Save as Template
              </Button>
            )}
          </div>
          {/* Hide Next button for mood board at topic step - it has its own generate button */}
          {!(currentStep === "topic" && selectedFormat === "moodboard") && (
            <Button
              onClick={handleNext}
              disabled={
                (currentStep === "format" && !selectedFormat) ||
                (currentStep === "topic" && !topic) ||
                (currentStep === "outline" && (outline.filter(s => s.approved).length === 0 || isGenerating)) ||
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
                  {currentStep === "topic" ? "Generate Outline" : currentStep === "outline" ? "Generate Content" : "Next"}
                  <ChevronRight className="h-4 w-4 ml-2" />
                </>
              )}
            </Button>
          )}
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

      {/* Template Dialogs */}
      <TemplateSelector
        open={showTemplateSelector}
        onOpenChange={setShowTemplateSelector}
        onSelect={handleApplyTemplate}
      />
      <SaveTemplateDialog
        open={showSaveTemplate}
        onOpenChange={setShowSaveTemplate}
        currentSettings={getCurrentSettings()}
        defaultCollections={styleCollections.length > 0 ? styleCollections : undefined}
      />
      {feedbackSection && currentJobId && (
        <SectionFeedbackDialog
          isOpen={!!feedbackSection}
          onClose={() => setFeedbackSection(null)}
          jobId={currentJobId}
          sectionId={feedbackSection.id}
          sectionTitle={feedbackSection.title}
          sectionContent={feedbackSection.content}
          qualityScore={feedbackSection.metadata?.quality_score}
          qualitySummary={feedbackSection.metadata?.quality_summary}
          onFeedbackSubmitted={() => {
            // Refresh job data after feedback
            setFeedbackSection(null);
          }}
        />
      )}
    </div>
  );
}

export default function CreatePage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh]">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <CreatePageContent />
    </Suspense>
  );
}
