"use client";

import { useState, useEffect } from "react";
import {
  Plus,
  Search,
  Headphones,
  Play,
  Trash2,
  MoreHorizontal,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Filter,
  FileText,
  Mic,
  MessageSquare,
  BookOpen,
  Users,
  GraduationCap,
  RefreshCw,
  DollarSign,
  Download,
  HardDrive,
  AlertCircle,
  Settings,
  ChevronDown,
  ChevronUp,
  Info,
  Volume2,
  Calendar,
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { AudioPlayer } from "@/components/audio-player";
import {
  useDocuments,
  useAudioOverviews,
  useAudioOverview,
  useCreateAudioOverview,
  useUpdateAudioOverview,
  useGenerateAudioOverview,
  useDeleteAudioOverview,
  useEstimateAudioCost,
  useCoquiModels,
  useDownloadCoquiModel,
  useDeleteCoquiModel,
  useTTSSettings,
  useAudioVoices,
  type AudioOverview,
  type TranscriptSegment,
  type VoiceInfo,
} from "@/lib/api";
import Link from "next/link";
import { toast } from "sonner";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";

// Edge TTS voice options - only currently available voices
const EDGE_TTS_VOICES = [
  // Premium multilingual voices (recommended)
  { id: "en-US-AvaMultilingualNeural", name: "Ava", gender: "female", style: "natural, warm" },
  { id: "en-US-AndrewMultilingualNeural", name: "Andrew", gender: "male", style: "natural, friendly" },
  { id: "en-US-EmmaMultilingualNeural", name: "Emma", gender: "female", style: "natural, clear" },
  { id: "en-US-BrianMultilingualNeural", name: "Brian", gender: "male", style: "natural, professional" },
  // High-quality US voices
  { id: "en-US-AvaNeural", name: "Ava (US)", gender: "female", style: "expressive, natural" },
  { id: "en-US-AndrewNeural", name: "Andrew (US)", gender: "male", style: "warm, engaging" },
  { id: "en-US-EmmaNeural", name: "Emma (US)", gender: "female", style: "friendly, clear" },
  { id: "en-US-BrianNeural", name: "Brian (US)", gender: "male", style: "professional" },
  { id: "en-US-JennyNeural", name: "Jenny", gender: "female", style: "friendly, conversational" },
  { id: "en-US-GuyNeural", name: "Guy", gender: "male", style: "conversational" },
  { id: "en-US-AriaNeural", name: "Aria", gender: "female", style: "professional" },
  { id: "en-US-ChristopherNeural", name: "Christopher", gender: "male", style: "reliable, clear" },
  { id: "en-US-EricNeural", name: "Eric", gender: "male", style: "calm, reassuring" },
  { id: "en-US-MichelleNeural", name: "Michelle", gender: "female", style: "warm, engaging" },
  { id: "en-US-RogerNeural", name: "Roger", gender: "male", style: "energetic" },
  { id: "en-US-SteffanNeural", name: "Steffan", gender: "male", style: "casual, friendly" },
  { id: "en-US-AnaNeural", name: "Ana", gender: "female", style: "child voice" },
  // British voices
  { id: "en-GB-SoniaNeural", name: "Sonia (UK)", gender: "female", style: "professional, british" },
  { id: "en-GB-RyanNeural", name: "Ryan (UK)", gender: "male", style: "friendly, british" },
  { id: "en-GB-MaisieNeural", name: "Maisie (UK)", gender: "female", style: "child voice, british" },
  { id: "en-GB-ThomasNeural", name: "Thomas (UK)", gender: "male", style: "warm, british" },
  // Australian voices
  { id: "en-AU-NatashaNeural", name: "Natasha (AU)", gender: "female", style: "friendly, australian" },
  { id: "en-AU-WilliamNeural", name: "William (AU)", gender: "male", style: "warm, australian" },
];

// Audio format configurations with icons
const audioFormatConfig: Record<string, { icon: typeof Headphones; name: string; description: string; duration: string }> = {
  deep_dive: {
    icon: Headphones,
    name: "Deep Dive",
    description: "Comprehensive exploration with two hosts",
    duration: "15-20 min",
  },
  brief: {
    icon: Clock,
    name: "Brief Summary",
    description: "Quick overview of key points",
    duration: "5 min",
  },
  critique: {
    icon: MessageSquare,
    name: "Critique",
    description: "Analysis of strengths and weaknesses",
    duration: "10-15 min",
  },
  debate: {
    icon: Users,
    name: "Debate",
    description: "Two hosts with contrasting viewpoints",
    duration: "12-15 min",
  },
  lecture: {
    icon: GraduationCap,
    name: "Lecture",
    description: "Educational single-speaker format",
    duration: "10-15 min",
  },
  interview: {
    icon: Mic,
    name: "Interview",
    description: "Q&A style with expert",
    duration: "12-15 min",
  },
};

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function getStatusBadge(status: string) {
  switch (status) {
    case "completed":
    case "ready":
      return (
        <Badge className="bg-green-500/10 text-green-600 border-green-200">
          <CheckCircle2 className="h-3 w-3 mr-1" />
          Ready
        </Badge>
      );
    case "generating_script":
      return (
        <Badge className="bg-blue-500/10 text-blue-600 border-blue-200">
          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
          Writing Script
        </Badge>
      );
    case "generating_audio":
      return (
        <Badge className="bg-purple-500/10 text-purple-600 border-purple-200">
          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
          Generating Audio
        </Badge>
      );
    case "failed":
      return (
        <Badge className="bg-red-500/10 text-red-600 border-red-200">
          <XCircle className="h-3 w-3 mr-1" />
          Failed
        </Badge>
      );
    default:
      return (
        <Badge variant="outline" className="bg-yellow-500/10 text-yellow-600 border-yellow-200">
          <Play className="h-3 w-3 mr-1" />
          Click to Generate
        </Badge>
      );
  }
}

export default function AudioOverviewsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [formatFilter, setFormatFilter] = useState<string | undefined>();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [selectedFormat, setSelectedFormat] = useState("deep_dive");
  const [selectedTTSProvider, setSelectedTTSProvider] = useState("edge"); // Default to free Edge TTS
  const [customTitle, setCustomTitle] = useState("");
  const [customInstructions, setCustomInstructions] = useState("");
  const [playerOverviewId, setPlayerOverviewId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [editingOverview, setEditingOverview] = useState<AudioOverview | null>(null);
  const [editTTSProvider, setEditTTSProvider] = useState("edge");
  const [showDetails, setShowDetails] = useState(false);
  const [selectedHost1Voice, setSelectedHost1Voice] = useState<string>("");
  const [selectedHost2Voice, setSelectedHost2Voice] = useState<string>("");
  const [host1Name, setHost1Name] = useState<string>("Alex");
  const [host2Name, setHost2Name] = useState<string>("Jordan");
  const [selectedDuration, setSelectedDuration] = useState<"short" | "standard" | "extended">("standard");

  // Fetch audio overviews
  const {
    data: overviewsData,
    isLoading: overviewsLoading,
    refetch: refetchOverviews,
  } = useAudioOverviews(
    {
      format: formatFilter === "all" ? undefined : formatFilter,
    },
    {
      refetchInterval: 5000, // Poll for status updates
    }
  );

  // Fetch specific overview for player (with transcript)
  const { data: playerOverview } = useAudioOverview(
    playerOverviewId || "",
    { enabled: !!playerOverviewId }
  );

  // Fetch documents for selection
  const { data: documentsData } = useDocuments({ page_size: 100 });

  // Mutations
  const createMutation = useCreateAudioOverview();
  const updateMutation = useUpdateAudioOverview();
  const generateMutation = useGenerateAudioOverview();
  const deleteMutation = useDeleteAudioOverview();
  const estimateCostMutation = useEstimateAudioCost();

  // TTS settings and provider availability
  const { data: ttsSettings } = useTTSSettings();

  // Available voices for the selected provider
  const { data: availableVoices } = useAudioVoices(selectedTTSProvider);

  // Coqui TTS models
  const { data: coquiModels, isLoading: coquiModelsLoading } = useCoquiModels();
  const downloadCoquiModel = useDownloadCoquiModel();
  const deleteCoquiModel = useDeleteCoquiModel();

  // Helper to check if a TTS provider is available
  const getProviderStatus = (providerId: string) => {
    const provider = ttsSettings?.available_providers.find((p) => p.id === providerId);
    return provider || null;
  };

  // Get voices list for the current provider (backend returns { provider: VoiceInfo[] })
  // Use fallback EDGE_TTS_VOICES if API doesn't return voices for edge provider
  const apiVoices = availableVoices?.[selectedTTSProvider];
  const voicesList = selectedTTSProvider === "edge"
    ? (apiVoices && apiVoices.length > 0 ? apiVoices : EDGE_TTS_VOICES)
    : (apiVoices || []);

  // Estimate cost when documents are selected
  useEffect(() => {
    if (selectedDocuments.length > 0 && createDialogOpen) {
      estimateCostMutation.mutate({
        document_ids: selectedDocuments,
        format: selectedFormat as "deep_dive" | "brief" | "critique" | "debate" | "lecture" | "interview",
        tts_provider: selectedTTSProvider as "openai" | "elevenlabs" | "edge" | "coqui",
      });
    }
  }, [selectedDocuments, selectedFormat, selectedTTSProvider, createDialogOpen]);

  // Filter overviews locally by search query
  const filteredOverviews = overviewsData?.overviews.filter((overview) => {
    if (searchQuery && !overview.title?.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    return true;
  }) || [];

  const handleCreateOverview = async () => {
    if (selectedDocuments.length === 0) {
      toast.error("Please select at least one document");
      return;
    }

    try {
      // Build voice selection with custom names
      const voices = (selectedHost1Voice || selectedHost2Voice || host1Name !== "Alex" || host2Name !== "Jordan") ? {
        host1_voice: selectedHost1Voice || undefined,
        host2_voice: selectedHost2Voice || undefined,
        host1_name: host1Name || "Alex",
        host2_name: host2Name || "Jordan",
      } : undefined;

      // Create the overview
      const overview = await createMutation.mutateAsync({
        document_ids: selectedDocuments,
        format: selectedFormat as "deep_dive" | "brief" | "critique" | "debate" | "lecture" | "interview",
        title: customTitle || undefined,
        custom_instructions: customInstructions || undefined,
        tts_provider: selectedTTSProvider as "openai" | "elevenlabs" | "edge" | "coqui",
        voices,
        duration_preference: selectedDuration,
      });

      // Start generation in background
      await generateMutation.mutateAsync({
        overviewId: overview.id,
        background: true
      });

      toast.success("Audio overview generation started");
      setCreateDialogOpen(false);
      setSelectedDocuments([]);
      setCustomTitle("");
      setCustomInstructions("");
      setSelectedHost1Voice("");
      setSelectedHost2Voice("");
      setSelectedDuration("standard");
      refetchOverviews();
    } catch (error) {
      toast.error("Failed to create audio overview");
      console.error(error);
    }
  };

  const handleDelete = async (overviewId: string) => {
    try {
      await deleteMutation.mutateAsync(overviewId);
      toast.success("Audio overview deleted");
      setDeleteConfirmId(null);
      if (playerOverviewId === overviewId) {
        setPlayerOverviewId(null);
      }
    } catch (error) {
      toast.error("Failed to delete audio overview");
      console.error(error);
    }
  };

  const handleRetryGeneration = async (overviewId: string) => {
    try {
      await generateMutation.mutateAsync({ overviewId, background: true });
      toast.success("Audio generation started");
      refetchOverviews();
    } catch (error) {
      toast.error("Failed to start generation");
      console.error(error);
    }
  };

  const handleEditOverview = (overview: AudioOverview) => {
    setEditingOverview(overview);
    // Set the current TTS provider from the overview config or default
    const currentProvider = (overview as { tts_provider?: string }).tts_provider || "edge";
    setEditTTSProvider(currentProvider);
  };

  const handleSaveEdit = async () => {
    if (!editingOverview) return;

    try {
      await updateMutation.mutateAsync({
        overviewId: editingOverview.id,
        data: { tts_provider: editTTSProvider as "openai" | "elevenlabs" | "edge" | "coqui" },
      });
      toast.success("Audio overview updated, starting generation...");

      // Start generation after updating
      await generateMutation.mutateAsync({ overviewId: editingOverview.id, background: true });

      setEditingOverview(null);
      refetchOverviews();
    } catch (error) {
      toast.error("Failed to update or generate audio overview");
      console.error(error);
    }
  };

  const toggleDocumentSelection = (docId: string) => {
    setSelectedDocuments((prev) =>
      prev.includes(docId)
        ? prev.filter((id) => id !== docId)
        : [...prev, docId]
    );
  };

  // Parse transcript from script if available
  // The backend stores dialogue as 'turns', not 'segments'
  const transcript: TranscriptSegment[] = playerOverview?.script?.turns || playerOverview?.script?.segments || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Audio Overviews</h1>
          <p className="text-muted-foreground">
            Generate AI podcast-style discussions from your documents
          </p>
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          New Audio
        </Button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search audio overviews..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={formatFilter || "all"} onValueChange={(v) => setFormatFilter(v === "all" ? undefined : v)}>
          <SelectTrigger className="w-40">
            <Filter className="h-4 w-4 mr-2" />
            <SelectValue placeholder="All formats" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All formats</SelectItem>
            {Object.entries(audioFormatConfig).map(([id, format]) => (
              <SelectItem key={id} value={id}>
                {format.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button variant="outline" size="icon" onClick={() => refetchOverviews()}>
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      {/* Audio Player (if playing) */}
      {playerOverview && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">{playerOverview.title || "Untitled Audio"}</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPlayerOverviewId(null)}
              >
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <AudioPlayer
              src={playerOverview.audio_url || ""}
              title={playerOverview.title || "Untitled Audio"}
              subtitle={`${audioFormatConfig[playerOverview.format]?.name || playerOverview.format} - ${formatDuration(playerOverview.duration_seconds || 0)}`}
              transcript={transcript}
              speakers={playerOverview.script?.speakers}
            />

            {/* Details Section */}
            <Collapsible open={showDetails} onOpenChange={setShowDetails}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full justify-between hover:bg-muted/50">
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <Info className="h-4 w-4" />
                    Audio Details
                  </span>
                  {showDetails ? (
                    <ChevronUp className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  )}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="mt-3 space-y-4 rounded-lg border bg-muted/30 p-4">
                  {/* Source Documents */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      Source Documents
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {playerOverview.document_ids.map((docId) => {
                        const doc = documentsData?.documents.find((d) => d.id === docId);
                        return (
                          <Badge key={docId} variant="secondary" className="text-xs">
                            {doc?.name || docId.slice(0, 8) + "..."}
                          </Badge>
                        );
                      })}
                    </div>
                  </div>

                  {/* Generation Info Grid */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {/* TTS Provider */}
                    <div className="space-y-1">
                      <span className="text-muted-foreground flex items-center gap-1">
                        <Volume2 className="h-3 w-3" />
                        Voice Provider
                      </span>
                      <Badge variant="outline" className="capitalize">
                        {playerOverview.tts_provider === "edge" && "Microsoft Edge TTS"}
                        {playerOverview.tts_provider === "openai" && "OpenAI TTS"}
                        {playerOverview.tts_provider === "elevenlabs" && "ElevenLabs"}
                        {playerOverview.tts_provider === "coqui" && "Coqui (Local)"}
                        {!playerOverview.tts_provider && "Unknown"}
                      </Badge>
                    </div>

                    {/* Format */}
                    <div className="space-y-1">
                      <span className="text-muted-foreground flex items-center gap-1">
                        <Headphones className="h-3 w-3" />
                        Format
                      </span>
                      <span className="font-medium">
                        {audioFormatConfig[playerOverview.format]?.name || playerOverview.format}
                      </span>
                    </div>

                    {/* Duration */}
                    <div className="space-y-1">
                      <span className="text-muted-foreground flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        Duration
                      </span>
                      <span className="font-medium">
                        {formatDuration(playerOverview.duration_seconds || 0)}
                      </span>
                    </div>

                    {/* Word Count */}
                    {playerOverview.script?.total_words && (
                      <div className="space-y-1">
                        <span className="text-muted-foreground flex items-center gap-1">
                          <BookOpen className="h-3 w-3" />
                          Script Words
                        </span>
                        <span className="font-medium">
                          {playerOverview.script.total_words.toLocaleString()}
                        </span>
                      </div>
                    )}

                    {/* Created Date */}
                    {playerOverview.created_at && (
                      <div className="space-y-1">
                        <span className="text-muted-foreground flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          Created
                        </span>
                        <span className="font-medium">
                          {formatDistanceToNow(new Date(playerOverview.created_at), { addSuffix: true })}
                        </span>
                      </div>
                    )}

                    {/* Hosts */}
                    {playerOverview.script?.hosts && playerOverview.script.hosts.length > 0 && (
                      <div className="space-y-1">
                        <span className="text-muted-foreground flex items-center gap-1">
                          <Users className="h-3 w-3" />
                          Hosts
                        </span>
                        <div className="flex gap-1">
                          {playerOverview.script.hosts.map((host, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {host.name}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </CardContent>
        </Card>
      )}

      {/* Loading State */}
      {overviewsLoading && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </CardHeader>
              <CardContent>
                <div className="flex justify-between">
                  <Skeleton className="h-4 w-20" />
                  <Skeleton className="h-5 w-16" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Audio Overview Grid */}
      {!overviewsLoading && filteredOverviews.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredOverviews.map((overview) => {
            const format = audioFormatConfig[overview.format];
            const FormatIcon = format?.icon || Headphones;

            return (
              <Card
                key={overview.id}
                className={cn(
                  "cursor-pointer transition-colors",
                  (overview.status === "completed" || overview.status === "ready" || overview.status === "pending") && "hover:border-primary/50"
                )}
                onClick={() => {
                  if (overview.status === "completed" || overview.status === "ready") {
                    setPlayerOverviewId(overview.id);
                  } else if (overview.status === "pending") {
                    handleRetryGeneration(overview.id);
                  }
                }}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <FormatIcon className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <CardTitle className="text-base line-clamp-1">
                          {overview.title || "Untitled Audio"}
                        </CardTitle>
                        <CardDescription>{format?.name || overview.format}</CardDescription>
                      </div>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {(overview.status === "completed" || overview.status === "ready") && (
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation();
                            setPlayerOverviewId(overview.id);
                          }}>
                            <Play className="mr-2 h-4 w-4" />
                            Play
                          </DropdownMenuItem>
                        )}
                        {overview.status === "pending" && (
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation();
                            handleRetryGeneration(overview.id);
                          }}>
                            <Play className="mr-2 h-4 w-4" />
                            Generate
                          </DropdownMenuItem>
                        )}
                        {overview.status === "failed" && (
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation();
                            handleRetryGeneration(overview.id);
                          }}>
                            <RefreshCw className="mr-2 h-4 w-4" />
                            Retry
                          </DropdownMenuItem>
                        )}
                        {(overview.status === "pending" || overview.status === "failed") && (
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation();
                            handleEditOverview(overview);
                          }}>
                            <Settings className="mr-2 h-4 w-4" />
                            Edit Settings
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          className="text-red-600"
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteConfirmId(overview.id);
                          }}
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-4 text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <FileText className="h-3.5 w-3.5" />
                        {overview.document_ids?.length || 0} docs
                      </span>
                      {overview.duration_seconds && (
                        <span>{formatDuration(overview.duration_seconds)}</span>
                      )}
                    </div>
                    {getStatusBadge(overview.status)}
                  </div>
                  {overview.error_message && (
                    <p className="text-xs text-red-500 mt-2 line-clamp-2">
                      {overview.error_message}
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground mt-2">
                    Created {overview.created_at ? formatDistanceToNow(new Date(overview.created_at), { addSuffix: true }) : "recently"}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : !overviewsLoading ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Headphones className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No audio overviews yet</h3>
            <p className="text-muted-foreground text-center max-w-sm mb-4">
              Create your first audio overview to generate engaging podcast-style
              discussions from your documents.
            </p>
            <Button onClick={() => setCreateDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Audio Overview
            </Button>
          </CardContent>
        </Card>
      ) : null}

      {/* Create Audio Overview Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] flex flex-col">
          <DialogHeader className="flex-shrink-0">
            <DialogTitle>Create Audio Overview</DialogTitle>
            <DialogDescription>
              Select documents and choose a format to generate an AI podcast discussion
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4 overflow-y-auto flex-1 pr-2">
            {/* Format Selection */}
            <div className="space-y-3">
              <Label>Format</Label>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(audioFormatConfig).map(([id, format]) => {
                  const Icon = format.icon;
                  return (
                    <div
                      key={id}
                      className={cn(
                        "flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                        selectedFormat === id
                          ? "border-primary bg-primary/5"
                          : "hover:border-muted-foreground/50"
                      )}
                      onClick={() => setSelectedFormat(id)}
                    >
                      <Icon className="h-5 w-5 text-primary mt-0.5" />
                      <div>
                        <div className="font-medium text-sm">{format.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {format.description}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          ~{format.duration}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Duration Control */}
            <div className="space-y-3">
              <Label>Duration</Label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: "short", label: "Short", description: "~50% shorter", icon: "⚡" },
                  { id: "standard", label: "Standard", description: "Default length", icon: "📻" },
                  { id: "extended", label: "Extended", description: "~50% longer", icon: "📚" },
                ].map((option) => (
                  <div
                    key={option.id}
                    className={cn(
                      "flex flex-col items-center p-3 rounded-lg border cursor-pointer transition-colors text-center",
                      selectedDuration === option.id
                        ? "border-primary bg-primary/5"
                        : "hover:border-muted-foreground/50"
                    )}
                    onClick={() => setSelectedDuration(option.id as "short" | "standard" | "extended")}
                  >
                    <span className="text-xl mb-1">{option.icon}</span>
                    <div className="font-medium text-sm">{option.label}</div>
                    <div className="text-xs text-muted-foreground">{option.description}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Document Selection */}
            <div className="space-y-3">
              <Label>
                Select Documents ({selectedDocuments.length} selected)
              </Label>
              <ScrollArea className="h-48 border rounded-lg p-2">
                {documentsData?.documents && documentsData.documents.length > 0 ? (
                  <div className="space-y-2">
                    {documentsData.documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="flex items-center gap-3 p-2 rounded hover:bg-muted cursor-pointer"
                        onClick={() => toggleDocumentSelection(doc.id)}
                      >
                        <Checkbox
                          checked={selectedDocuments.includes(doc.id)}
                          onCheckedChange={() => toggleDocumentSelection(doc.id)}
                        />
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm truncate">{doc.name}</div>
                          <div className="text-xs text-muted-foreground">
                            {doc.file_type} - {doc.word_count?.toLocaleString() || 0} words
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-muted-foreground">
                    No documents available
                  </div>
                )}
              </ScrollArea>
            </div>

            {/* Custom Title */}
            <div className="space-y-2">
              <Label>Title (optional)</Label>
              <Input
                placeholder="Auto-generated from documents..."
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
              />
            </div>

            {/* TTS Provider Selection */}
            <div className="space-y-2">
              <Label>Voice Provider</Label>
              <Select value={selectedTTSProvider} onValueChange={setSelectedTTSProvider}>
                <SelectTrigger>
                  <SelectValue placeholder="Select voice provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="edge">
                    <div className="flex items-center gap-2">
                      <span>Microsoft Edge TTS</span>
                      <Badge variant="outline" className="text-xs bg-green-500/10 text-green-600">Free</Badge>
                    </div>
                  </SelectItem>
                  <SelectItem value="openai">
                    <div className="flex items-center gap-2">
                      <span>OpenAI TTS</span>
                      <Badge variant="outline" className="text-xs">API Key Required</Badge>
                    </div>
                  </SelectItem>
                  <SelectItem value="elevenlabs">
                    <div className="flex items-center gap-2">
                      <span>ElevenLabs</span>
                      <Badge variant="outline" className="text-xs">Premium</Badge>
                    </div>
                  </SelectItem>
                  <SelectItem value="coqui">
                    <div className="flex items-center gap-2">
                      <span>Local (Coqui TTS)</span>
                      <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-600">Self-hosted</Badge>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {selectedTTSProvider === "edge" && "Free high-quality voices from Microsoft. No API key needed."}
                {selectedTTSProvider === "openai" && "Requires OPENAI_API_KEY. High quality, natural sounding."}
                {selectedTTSProvider === "elevenlabs" && "Premium voices with voice cloning. Requires API key."}
                {selectedTTSProvider === "coqui" && "Runs locally using Coqui TTS. Requires 'pip install TTS'."}
              </p>
            </div>

            {/* Voice Selection - show when Edge TTS is selected */}
            {selectedTTSProvider === "edge" && (
              <div className="space-y-3 p-3 border rounded-lg bg-muted/30">
                <div className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4 text-muted-foreground" />
                  <Label>Voice Selection</Label>
                </div>
                {/* Speaker Names */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      {selectedFormat === "debate" ? "Debater 1 Name" : selectedFormat === "interview" ? "Interviewer Name" : "Host 1 Name"}
                    </Label>
                    <Input
                      value={host1Name}
                      onChange={(e) => setHost1Name(e.target.value)}
                      placeholder="Alex"
                      className="h-9"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      {selectedFormat === "debate" ? "Debater 2 Name" : selectedFormat === "interview" ? "Expert Name" : "Host 2 Name"}
                    </Label>
                    <Input
                      value={host2Name}
                      onChange={(e) => setHost2Name(e.target.value)}
                      placeholder="Jordan"
                      className="h-9"
                    />
                  </div>
                </div>
                {/* Voice Selection */}
                <div className="grid grid-cols-2 gap-3">
                  {/* Host 1 Voice */}
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      {host1Name || "Host 1"} Voice
                    </Label>
                    <Select value={selectedHost1Voice || "default"} onValueChange={(v) => setSelectedHost1Voice(v === "default" ? "" : v)}>
                      <SelectTrigger className="h-9">
                        <SelectValue placeholder="Default (Andrew)" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="default">Default (Andrew - Natural Male)</SelectItem>
                        <div className="px-2 py-1 text-xs font-medium text-muted-foreground">Male Voices</div>
                        {voicesList
                          .filter((v) => v.gender === "male")
                          .map((voice) => (
                            <SelectItem key={voice.id} value={voice.id}>
                              <div className="flex items-center gap-2">
                                <span>{voice.name}</span>
                                <span className="text-xs text-muted-foreground">({voice.style})</span>
                              </div>
                            </SelectItem>
                          ))}
                        <div className="px-2 py-1 text-xs font-medium text-muted-foreground">Female Voices</div>
                        {voicesList
                          .filter((v) => v.gender === "female")
                          .map((voice) => (
                            <SelectItem key={voice.id} value={voice.id}>
                              <div className="flex items-center gap-2">
                                <span>{voice.name}</span>
                                <span className="text-xs text-muted-foreground">({voice.style})</span>
                              </div>
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                  {/* Host 2 Voice */}
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      {host2Name || "Host 2"} Voice
                    </Label>
                    <Select value={selectedHost2Voice || "default"} onValueChange={(v) => setSelectedHost2Voice(v === "default" ? "" : v)}>
                      <SelectTrigger className="h-9">
                        <SelectValue placeholder="Default (Ava)" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="default">Default (Ava - Natural Female)</SelectItem>
                        <div className="px-2 py-1 text-xs font-medium text-muted-foreground">Female Voices</div>
                        {voicesList
                          .filter((v) => v.gender === "female")
                          .map((voice) => (
                            <SelectItem key={voice.id} value={voice.id}>
                              <div className="flex items-center gap-2">
                                <span>{voice.name}</span>
                                <span className="text-xs text-muted-foreground">({voice.style})</span>
                              </div>
                            </SelectItem>
                          ))}
                        <div className="px-2 py-1 text-xs font-medium text-muted-foreground">Male Voices</div>
                        {voicesList
                          .filter((v) => v.gender === "male")
                          .map((voice) => (
                            <SelectItem key={voice.id} value={voice.id}>
                              <div className="flex items-center gap-2">
                                <span>{voice.name}</span>
                                <span className="text-xs text-muted-foreground">({voice.style})</span>
                              </div>
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  Choose different voices for each speaker. Premium voices (Ava, Andrew, Emma, Brian) sound most natural.
                </p>
              </div>
            )}

            {/* OpenAI API Key Status */}
            {selectedTTSProvider === "openai" && getProviderStatus("openai")?.is_available && (
              <div className="flex items-center gap-2 p-2 border border-green-200 rounded-lg bg-green-500/10">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <span className="text-sm text-green-700">OpenAI API key configured</span>
              </div>
            )}
            {selectedTTSProvider === "openai" && !getProviderStatus("openai")?.is_available && (
              <div className="flex items-start gap-3 p-3 border border-yellow-200 rounded-lg bg-yellow-500/10">
                <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-yellow-800">OpenAI API Key Not Configured</p>
                  <p className="text-xs text-yellow-700 mt-1">
                    To use OpenAI TTS, add your API key to the backend <code className="bg-yellow-100 px-1 rounded">.env</code> file:
                  </p>
                  <code className="text-xs bg-yellow-100 px-2 py-1 rounded mt-2 block text-yellow-800">
                    OPENAI_API_KEY=sk-your-key-here
                  </code>
                  <div className="flex items-center gap-3 mt-3">
                    <Link href="/dashboard/admin/settings" className="text-xs font-medium text-yellow-800 underline hover:text-yellow-900">
                      Go to Admin Settings →
                    </Link>
                    <span className="text-xs text-yellow-600">or use Edge TTS (free)</span>
                  </div>
                </div>
              </div>
            )}

            {/* ElevenLabs API Key Warning */}
            {/* ElevenLabs API Key Status */}
            {selectedTTSProvider === "elevenlabs" && getProviderStatus("elevenlabs")?.is_available && (
              <div className="flex items-center gap-2 p-2 border border-green-200 rounded-lg bg-green-500/10">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <span className="text-sm text-green-700">ElevenLabs API key configured</span>
              </div>
            )}
            {selectedTTSProvider === "elevenlabs" && !getProviderStatus("elevenlabs")?.is_available && (
              <div className="flex items-start gap-3 p-3 border border-yellow-200 rounded-lg bg-yellow-500/10">
                <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-yellow-800">ElevenLabs API Key Not Configured</p>
                  <p className="text-xs text-yellow-700 mt-1">
                    To use ElevenLabs TTS, add your API key to the backend <code className="bg-yellow-100 px-1 rounded">.env</code> file:
                  </p>
                  <code className="text-xs bg-yellow-100 px-2 py-1 rounded mt-2 block text-yellow-800">
                    ELEVENLABS_API_KEY=your-key-here
                  </code>
                  <div className="flex items-center gap-3 mt-3">
                    <a
                      href="https://elevenlabs.io"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs font-medium text-yellow-800 underline hover:text-yellow-900"
                    >
                      Get API key at elevenlabs.io →
                    </a>
                    <Link href="/dashboard/admin/settings" className="text-xs font-medium text-yellow-800 underline hover:text-yellow-900">
                      Admin Settings
                    </Link>
                  </div>
                </div>
              </div>
            )}

            {/* Coqui Model Management - show only when Coqui is selected */}
            {selectedTTSProvider === "coqui" && (
              <div className="space-y-3 p-3 border rounded-lg bg-muted/30">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4 text-muted-foreground" />
                    <Label>Coqui TTS Models</Label>
                  </div>
                  {coquiModelsLoading && <Loader2 className="h-4 w-4 animate-spin" />}
                </div>

                {coquiModels && coquiModels.length > 0 ? (
                  <ScrollArea className="h-32">
                    <div className="space-y-2">
                      {coquiModels.map((model) => (
                        <div
                          key={model.name}
                          className="flex items-center justify-between p-2 rounded border bg-background"
                        >
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{model.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {model.language} • {model.size_mb}MB
                            </p>
                          </div>
                          <div className="flex items-center gap-2">
                            {model.is_installed ? (
                              <>
                                <Badge variant="outline" className="bg-green-500/10 text-green-600 text-xs">
                                  Installed
                                </Badge>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7 text-red-500 hover:text-red-600"
                                  onClick={() => {
                                    deleteCoquiModel.mutate(model.name, {
                                      onSuccess: () => toast.success(`Model "${model.name}" deleted`),
                                      onError: () => toast.error(`Failed to delete model`),
                                    });
                                  }}
                                  disabled={deleteCoquiModel.isPending}
                                >
                                  {deleteCoquiModel.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <Trash2 className="h-4 w-4" />
                                  )}
                                </Button>
                              </>
                            ) : (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-7 text-xs"
                                onClick={() => {
                                  downloadCoquiModel.mutate(model.name, {
                                    onSuccess: () => toast.success(`Model "${model.name}" downloaded`),
                                    onError: () => toast.error(`Failed to download model`),
                                  });
                                }}
                                disabled={downloadCoquiModel.isPending}
                              >
                                {downloadCoquiModel.isPending ? (
                                  <Loader2 className="h-4 w-4 animate-spin mr-1" />
                                ) : (
                                  <Download className="h-3 w-3 mr-1" />
                                )}
                                Download
                              </Button>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : !coquiModelsLoading ? (
                  <div className="flex flex-col items-start py-3 px-2">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium">Coqui TTS Not Installed</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          To use local TTS, install Coqui TTS in the backend:
                        </p>
                        <code className="text-xs bg-muted px-2 py-1 rounded mt-2 block">
                          pip install TTS
                        </code>
                        <p className="text-xs text-muted-foreground mt-2">
                          This requires ~2GB disk space and a GPU for best performance.
                        </p>
                        <div className="flex items-center gap-3 mt-3">
                          <Link href="/dashboard/admin/settings" className="text-xs font-medium text-primary underline">
                            Manage in Admin Settings →
                          </Link>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            )}

            {/* Custom Instructions */}
            <div className="space-y-2">
              <Label>Custom Instructions (optional)</Label>
              <Textarea
                placeholder="Any specific topics to focus on, style preferences, etc..."
                value={customInstructions}
                onChange={(e) => setCustomInstructions(e.target.value)}
                rows={3}
              />
            </div>

            {/* Cost Estimate */}
            {estimateCostMutation.data && selectedDocuments.length > 0 && (
              <div className="bg-muted/50 rounded-lg p-3 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <DollarSign className="h-4 w-4" />
                  Cost Estimate
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="text-muted-foreground">Estimated duration:</div>
                  <div>
                    {Math.floor((estimateCostMutation.data.estimated_duration?.min_seconds || 0) / 60)}-
                    {Math.ceil((estimateCostMutation.data.estimated_duration?.max_seconds || 0) / 60)} min
                  </div>
                  <div className="text-muted-foreground">Script generation:</div>
                  <div>${(estimateCostMutation.data.estimated_costs?.script_generation_usd || 0).toFixed(2)}</div>
                  <div className="text-muted-foreground">Text-to-speech:</div>
                  <div>${(estimateCostMutation.data.estimated_costs?.tts_usd || 0).toFixed(2)}</div>
                  <div className="text-muted-foreground font-medium">Total:</div>
                  <div className="font-medium">${(estimateCostMutation.data.estimated_costs?.total_usd || 0).toFixed(2)}</div>
                </div>
              </div>
            )}
          </div>

          <DialogFooter className="flex-shrink-0 border-t pt-4 mt-4">
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateOverview}
              disabled={selectedDocuments.length === 0 || createMutation.isPending || generateMutation.isPending}
            >
              {(createMutation.isPending || generateMutation.isPending) && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Generate Audio
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={!!deleteConfirmId} onOpenChange={() => setDeleteConfirmId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Audio Overview</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this audio overview? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteConfirmId(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deleteConfirmId && handleDelete(deleteConfirmId)}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Settings Dialog */}
      <Dialog open={!!editingOverview} onOpenChange={() => setEditingOverview(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Audio Settings</DialogTitle>
            <DialogDescription>
              Change the TTS provider for this audio overview. After saving, you can generate the audio.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Voice Provider</Label>
              <Select value={editTTSProvider} onValueChange={setEditTTSProvider}>
                <SelectTrigger>
                  <SelectValue placeholder="Select voice provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="edge">
                    <div className="flex items-center gap-2">
                      <span>Microsoft Edge TTS</span>
                      <Badge variant="outline" className="text-xs bg-green-500/10 text-green-600">Free</Badge>
                    </div>
                  </SelectItem>
                  <SelectItem value="openai">
                    <div className="flex items-center gap-2">
                      <span>OpenAI TTS</span>
                      <Badge variant="outline" className="text-xs">API Key Required</Badge>
                    </div>
                  </SelectItem>
                  <SelectItem value="elevenlabs">
                    <div className="flex items-center gap-2">
                      <span>ElevenLabs</span>
                      <Badge variant="outline" className="text-xs">Premium</Badge>
                    </div>
                  </SelectItem>
                  <SelectItem value="coqui">
                    <div className="flex items-center gap-2">
                      <span>Local (Coqui TTS)</span>
                      <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-600">Self-hosted</Badge>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {editTTSProvider === "edge" && "Free high-quality voices from Microsoft. No API key needed."}
                {editTTSProvider === "openai" && "Requires OPENAI_API_KEY. High quality, natural sounding."}
                {editTTSProvider === "elevenlabs" && "Premium voices with voice cloning. Requires API key."}
                {editTTSProvider === "coqui" && "Runs locally using Coqui TTS. Requires 'pip install TTS'."}
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingOverview(null)}>
              Cancel
            </Button>
            <Button
              onClick={handleSaveEdit}
              disabled={updateMutation.isPending}
            >
              {updateMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save & Generate
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
