"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { useSession } from "next-auth/react";
import {
  Send,
  Bot,
  User,
  FileText,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  Copy,
  Check,
  PanelRightOpen,
  PanelRightClose,
  ExternalLink,
  Clock,
  MessageSquare,
  Trash2,
  ChevronDown,
  ChevronRight,
  Search,
  Sparkles,
  Settings2,
  Cpu,
  Thermometer,
  Zap,
  Brain,
  FileSearch,
  Tag,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  Filter,
  Mic,
  Upload,
  Globe,
  BrainCog,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import { getErrorMessage } from "@/lib/errors";
import {
  useChatSessions,
  useChatSession,
  useCreateChatSession,
  useDeleteChatSession,
  useDeleteAllChatSessions,
  useSendChatMessage,
} from "@/lib/api";
import {
  useLLMProviders,
  useSessionLLMConfig,
  useSetSessionLLMConfig,
  useAgentMode,
  useApproveAgentExecution,
  useCancelAgentExecution,
  useCollections,
} from "@/lib/api/hooks";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CostApprovalDialog } from "@/components/cost-approval-dialog";
import { AgentExecutionProgress } from "@/components/agent-execution-progress";
import { AgentResponseSections } from "@/components/agent-response-sections";
import { SourceDocumentViewer } from "@/components/source-document-viewer";
import { DocumentFilterPanel } from "@/components/chat/document-filter-panel";
import { VoiceInput } from "@/components/chat/voice-input";
import { TextToSpeech } from "@/components/chat/text-to-speech";
import { VoiceConversationIndicator, type VoiceState } from "@/components/chat/voice-conversation-indicator";
import { TempDocumentPanel } from "@/components/chat/temp-document-panel";
import { ImageUploadCompact, ImagePreviewBar } from "@/components/chat/image-upload";
import { FolderSelector } from "@/components/folder-selector";
import { QueryEnhancementToggle } from "@/components/chat/query-enhancement-toggle";
import { api, type PlanStep, type ExecutionMode } from "@/lib/api/client";

/**
 * Highlights matching query terms in text for source snippets.
 * Returns React nodes with highlighted terms wrapped in <mark> tags.
 */
function highlightQueryTerms(text: string, query?: string): React.ReactNode {
  if (!query || !text) {
    return text;
  }

  // English stop words to ignore - keep domain-specific terms
  const stopWords = new Set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need",
    "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "and", "but", "if", "or", "because", "until", "while",
    "about", "what", "which", "who", "this", "that", "these", "those",
  ]);

  // Extract words from query - keep words 2+ chars that aren't stopwords
  // Also preserve punctuation-attached words like "German?" -> "german"
  const queryWords = query
    .toLowerCase()
    .replace(/[?!.,;:'"]/g, "") // Remove punctuation
    .split(/\s+/)
    .filter(word => word.length >= 2 && !stopWords.has(word))
    .map(word => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")); // Escape regex special chars

  if (queryWords.length === 0) {
    return text;
  }

  // Check if query contains non-ASCII characters (German ä, ö, ü, etc.)
  // \b word boundary doesn't work with Unicode - use simple matching instead
  const hasNonAscii = queryWords.some(w => /[^\x00-\x7F]/.test(w));

  try {
    // Create pattern - for non-ASCII languages, match anywhere
    // For ASCII, use word boundaries for precise matching
    const pattern = hasNonAscii
      ? new RegExp(`(${queryWords.join("|")})`, "gi")
      : new RegExp(`\\b(${queryWords.join("|")})\\b`, "gi");

    const parts = text.split(pattern);

    return parts.map((part, index) => {
      // Check if this part matches any query word (case-insensitive)
      const isMatch = queryWords.some(word =>
        part.toLowerCase() === word.toLowerCase()
      );
      if (isMatch) {
        return (
          <mark
            key={index}
            className="bg-yellow-300 dark:bg-yellow-600 text-black dark:text-yellow-100 px-0.5 rounded font-medium"
          >
            {part}
          </mark>
        );
      }
      return part;
    });
  } catch {
    // If regex fails, return plain text
    return text;
  }
}

// Language options for chat output
const CHAT_LANGUAGES = [
  { code: "auto", name: "Auto (match question)" },
  { code: "en", name: "English" },
  { code: "de", name: "German" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "nl", name: "Dutch" },
  { code: "pl", name: "Polish" },
  { code: "ru", name: "Russian" },
  { code: "zh", name: "Chinese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
];

// Temp document interface matching backend response
interface TempDocument {
  id: string;
  filename: string;
  token_count: number;
  file_size: number;
  file_type: string;
  has_chunks: boolean;
  has_embeddings: boolean;
}

// Image attachment for vision chat
interface ImageAttachment {
  id: string;
  data: string;
  mimeType: string;
  name: string;
  preview: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
  isStreaming?: boolean;
  isGeneralResponse?: boolean;
  confidenceScore?: number;  // 0-1 confidence score
  confidenceLevel?: "high" | "medium" | "low";  // Confidence level
  suggestedQuestions?: string[];  // Follow-up question suggestions
  // Agent mode specific fields
  isAgentResponse?: boolean;  // Whether this is from agent mode
  planningDetails?: string;  // Planning phase summary
  executionSteps?: PlanStep[];  // Steps executed
  thinkingContent?: string;  // Agent's reasoning/thinking process
  query?: string;  // Original user query for this response (used for source highlighting)
}

interface Source {
  documentId: string;
  filename: string;
  pageNumber?: number;
  snippet: string;
  fullContent?: string;  // Full chunk content for source viewer modal
  similarity: number;
  collection?: string;
}

export default function ChatPage() {
  // Auth state
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hello! I'm your AI document assistant. I can help you find information across your document archive, answer questions, and even help generate new content. What would you like to know?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showSourcePanel, setShowSourcePanel] = useState(true);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [agentExecutionMode, setAgentExecutionMode] = useState<ExecutionMode>("agent");
  // Source mode: documents (RAG search) vs general (no document search)
  const [sourceMode, setSourceMode] = useState<"documents" | "general">("documents");
  // Agent mode: enable multi-agent orchestration (can be combined with either source mode)
  const [agentEnabled, setAgentEnabled] = useState(false);
  // Agent mode options - user-configurable settings for agent execution
  const [agentOptions, setAgentOptions] = useState({
    search_documents: true,      // Force search uploaded documents first
    include_web_search: false,   // Include web search in research
    require_approval: false,     // Show plan and require approval before execution
    max_steps: 5,                // Maximum number of steps in execution plan
    collection: null as string | null,  // Target specific collection (null = all)
  });
  // Collection context toggle - shows collection tags to LLM for better document disambiguation
  const [includeCollectionContext, setIncludeCollectionContext] = useState(true);
  const [showCostApproval, setShowCostApproval] = useState(false);
  const [pendingPlanId, setPendingPlanId] = useState<string | null>(null);
  const [pendingPlanSummary, setPendingPlanSummary] = useState<string | null>(null);
  const [pendingEstimatedCost, setPendingEstimatedCost] = useState<number>(0);
  const [pendingSteps, setPendingSteps] = useState<PlanStep[] | null>(null);
  const [agentSteps, setAgentSteps] = useState<PlanStep[]>([]);
  const [currentAgentStep, setCurrentAgentStep] = useState<number>(0);
  const [isAgentExecuting, setIsAgentExecuting] = useState(false);
  const [expandedSourceId, setExpandedSourceId] = useState<string | null>(null);
  const [sourceViewerOpen, setSourceViewerOpen] = useState(false);
  const [sourceViewerIndex, setSourceViewerIndex] = useState(0);
  const [lastUserQuery, setLastUserQuery] = useState<string>("");
  const [shouldLoadHistory, setShouldLoadHistory] = useState(false);
  const [historySearchQuery, setHistorySearchQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  // Folder filter for scoped queries
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [includeSubfolders, setIncludeSubfolders] = useState(true);
  // Voice conversation mode
  const [voiceModeEnabled, setVoiceModeEnabled] = useState(false);
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [lastAssistantMessage, setLastAssistantMessage] = useState<string | null>(null);
  // Temp document upload (Quick Upload)
  const [showTempPanel, setShowTempPanel] = useState(false);
  const [tempSessionId, setTempSessionId] = useState<string | null>(null);
  const [tempDocuments, setTempDocuments] = useState<TempDocument[]>([]);
  const [tempTotalTokens, setTempTotalTokens] = useState(0);
  const [isTempUploading, setIsTempUploading] = useState(false);
  // Documents to search (per-query override, null = use admin setting)
  const [topK, setTopK] = useState<number | null>(null);
  // Output language for chat responses
  const [outputLanguage, setOutputLanguage] = useState<string>("auto");
  // Query enhancement toggle (expansion + HyDE)
  const [enhanceQuery, setEnhanceQuery] = useState<boolean | null>(null);
  // Image attachments for vision mode
  const [attachedImages, setAttachedImages] = useState<ImageAttachment[]>([]);
  // Restrict to documents only - disables LLM's pre-trained knowledge in general mode
  const [restrictToDocuments, setRestrictToDocuments] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Queries - wait for auth to be ready before making API calls
  const { data: sessions } = useChatSessions(undefined, { enabled: isAuthenticated });
  const { data: currentSession, error: sessionError } = useChatSession(currentSessionId);
  const { data: providersData } = useLLMProviders({ enabled: isAuthenticated });
  const { data: sessionLLMConfig, refetch: refetchSessionLLM } = useSessionLLMConfig(
    currentSessionId || "",
    { enabled: isAuthenticated && !!currentSessionId }
  );

  // Agent mode query
  const { data: agentModeData } = useAgentMode(currentSessionId || undefined, { enabled: isAuthenticated });

  // Collections for filtering
  const { data: collectionsData, refetch: refetchCollections, isLoading: isLoadingCollections } = useCollections({ enabled: isAuthenticated });

  // Handle 404 errors - session doesn't exist, clear and start fresh
  useEffect(() => {
    if (sessionError && typeof sessionError === 'object' && 'status' in sessionError) {
      const status = (sessionError as { status: number }).status;
      if (status === 404 && currentSessionId) {
        console.log('[Chat] Session not found, clearing session ID');
        setCurrentSessionId(null);
      }
    }
  }, [sessionError, currentSessionId]);

  // Sync messages when session data is loaded from history
  // Only runs when user explicitly clicks a history session (shouldLoadHistory flag)
  useEffect(() => {
    if (shouldLoadHistory && currentSessionId && currentSession?.messages && currentSession.messages.length > 0) {
      // Transform API messages to local Message format, including sources
      const loadedMessages: Message[] = currentSession.messages.map((msg, index) => {
        const msgId = `session-msg-${index}`;
        // Get sources for this message if available (keyed by message index)
        const msgSources = currentSession.sources?.[String(index)] || [];
        const transformedSources: Source[] = msgSources.map((src) => ({
          documentId: src.document_id,
          filename: src.document_name,
          pageNumber: src.page_number,
          snippet: src.snippet,
          fullContent: src.full_content,
          similarity: src.similarity_score ?? src.relevance_score,
          collection: src.collection,
        }));

        return {
          id: msgId,
          role: msg.role as "user" | "assistant",
          content: msg.content,
          timestamp: new Date(),
          sources: transformedSources.length > 0 ? transformedSources : undefined,
          // Note: confidenceScore, isAgentResponse, executionSteps etc. are not persisted
          // to the database, so they won't be available when loading from history
        };
      });
      setMessages(loadedMessages);
      setShouldLoadHistory(false); // Reset flag after loading
    }
  }, [shouldLoadHistory, currentSessionId, currentSession]);

  // Mutations
  const createSession = useCreateChatSession();
  const deleteSession = useDeleteChatSession();
  const deleteAllSessions = useDeleteAllChatSessions();
  const sendMessage = useSendChatMessage();
  const setSessionLLM = useSetSessionLLMConfig();
  const approveExecution = useApproveAgentExecution();
  const cancelExecution = useCancelAgentExecution();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + Enter to send message
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && input.trim() && !isLoading) {
        e.preventDefault();
        handleSubmit(e as unknown as React.FormEvent);
      }
      // Cmd/Ctrl + K to focus input
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      // Cmd/Ctrl + Shift + E to export chat
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "e") {
        e.preventDefault();
        handleExportChat();
      }
      // Cmd/Ctrl + N for new chat
      if ((e.metaKey || e.ctrlKey) && e.key === "n") {
        e.preventDefault();
        handleNewChat();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [input, isLoading]);

  // Sync temperature with session config
  useEffect(() => {
    if (sessionLLMConfig?.temperature !== null && sessionLLMConfig?.temperature !== undefined) {
      setTemperature(sessionLLMConfig.temperature);
    }
  }, [sessionLLMConfig?.temperature]);

  // Load query enhancement preference from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("chat_enhance_query");
    if (saved !== null) {
      setEnhanceQuery(JSON.parse(saved));
    }
  }, []);

  // Handler for query enhancement toggle
  const handleEnhanceQueryChange = (enabled: boolean) => {
    setEnhanceQuery(enabled);
    localStorage.setItem("chat_enhance_query", JSON.stringify(enabled));
  };

  // Helper to get temperature description
  const getTemperatureLabel = (temp: number) => {
    if (temp <= 0.3) return "Precise";
    if (temp <= 0.7) return "Balanced";
    if (temp <= 1.0) return "Creative";
    return "Very Creative";
  };

  // Helper to get confidence icon and color
  const getConfidenceDisplay = (level?: "high" | "medium" | "low", score?: number) => {
    if (!level) return null;

    const config = {
      high: {
        icon: ShieldCheck,
        color: "text-green-500",
        bgColor: "bg-green-500/10",
        label: "High Confidence",
      },
      medium: {
        icon: ShieldQuestion,
        color: "text-yellow-500",
        bgColor: "bg-yellow-500/10",
        label: "Medium Confidence",
      },
      low: {
        icon: ShieldAlert,
        color: "text-red-500",
        bgColor: "bg-red-500/10",
        label: "Low Confidence",
      },
    };

    return { ...config[level], score: score ? Math.round(score * 100) : null };
  };

  // Handle temperature change
  const handleTemperatureChange = async (value: number[]) => {
    const newTemp = value[0];
    setTemperature(newTemp);

    // Update session config if we have a session
    if (currentSessionId && sessionLLMConfig?.provider_id) {
      try {
        await setSessionLLM.mutateAsync({
          sessionId: currentSessionId,
          data: {
            provider_id: sessionLLMConfig.provider_id,
            model_override: sessionLLMConfig.model || undefined,
            temperature_override: newTemp,
          },
        });
      } catch (error) {
        console.error("Failed to update temperature:", error);
      }
    }
  };

  // Get sources from selected message
  const selectedMessage = messages.find((m) => m.id === selectedMessageId);
  const selectedSources = selectedMessage?.sources || [];
  const isSelectedMessageGeneral = selectedMessage?.isGeneralResponse;

  // Filter sessions by search query
  const filteredSessions = useMemo(() => {
    if (!sessions?.sessions) return [];
    if (!historySearchQuery.trim()) return sessions.sessions;

    const query = historySearchQuery.toLowerCase();
    return sessions.sessions.filter((session: { title: string }) =>
      session.title.toLowerCase().includes(query)
    );
  }, [sessions?.sessions, historySearchQuery]);

  // Group sources by document for better display
  const groupedSources = useMemo(() => {
    const groups: Record<string, Source[]> = {};
    selectedSources.forEach((source) => {
      const key = source.documentId;
      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(source);
    });
    return Object.entries(groups).map(([documentId, sources]) => ({
      documentId,
      filename: sources[0].filename,
      sources: sources.sort((a, b) => (a.pageNumber || 0) - (b.pageNumber || 0)),
      avgSimilarity: sources.reduce((acc, s) => acc + s.similarity, 0) / sources.length,
    }));
  }, [selectedSources]);

  // Navigate to document detail page
  const handleSourceClick = (documentId: string) => {
    window.open(`/dashboard/documents/${documentId}`, '_blank');
  };

  // Temp document handlers
  const handleTempUpload = async (file: File) => {
    try {
      let sessionId = tempSessionId;

      // Create session if needed
      if (!sessionId) {
        const sessionData = await api.createTempSession();
        sessionId = sessionData.session_id;
        setTempSessionId(sessionId);
      }

      // Upload file
      setIsTempUploading(true);
      await api.uploadTempDocument(sessionId, file, true);

      // Refresh session info
      await refreshTempSession(sessionId);
    } catch (error) {
      console.error("Temp upload failed:", error);
      throw error;
    } finally {
      setIsTempUploading(false);
    }
  };

  const refreshTempSession = async (sessionId: string | null) => {
    if (!sessionId) return;

    try {
      const data = await api.getTempSessionInfo(sessionId);
      setTempDocuments(data.documents || []);
      setTempTotalTokens(data.total_tokens || 0);
    } catch (error: unknown) {
      // Session expired or not found
      if (error && typeof error === 'object' && 'status' in error && (error as {status: number}).status === 404) {
        setTempSessionId(null);
        setTempDocuments([]);
        setTempTotalTokens(0);
      } else {
        console.error("Failed to refresh temp session:", error);
      }
    }
  };

  const handleTempRemove = async (docId: string) => {
    if (!tempSessionId) return;

    try {
      await api.removeTempDocument(tempSessionId, docId);
      await refreshTempSession(tempSessionId);
    } catch (error) {
      console.error("Failed to remove temp document:", error);
      throw error;
    }
  };

  const handleTempSave = async (docId: string, collection?: string) => {
    if (!tempSessionId) return;

    try {
      await api.saveTempDocument(tempSessionId, docId, collection);
      await refreshTempSession(tempSessionId);
      // Refresh collections since a new document was added
      refetchCollections();
    } catch (error) {
      console.error("Failed to save temp document:", error);
      throw error;
    }
  };

  const handleTempDiscard = async () => {
    if (!tempSessionId) return;

    try {
      await api.deleteTempSession(tempSessionId);
      setTempSessionId(null);
      setTempDocuments([]);
      setTempTotalTokens(0);
    } catch (error) {
      console.error("Failed to discard temp session:", error);
      throw error;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Capture current images before clearing
    const imagesToSend = [...attachedImages];
    const hasImages = imagesToSend.length > 0;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: hasImages ? `[Image${imagesToSend.length > 1 ? 's' : ''} attached] ${input.trim()}` : input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageText = input.trim();
    setLastUserQuery(messageText); // Track the last user query for source highlighting
    setInput("");
    setAttachedImages([]); // Clear attached images after sending
    setIsLoading(true);

    // Update voice state when submitting in voice mode
    if (voiceModeEnabled) {
      setVoiceState("thinking");
    }

    // Reset agent state when starting new request
    if (agentEnabled) {
      setAgentSteps([]);
      setCurrentAgentStep(0);
      setIsAgentExecuting(false);
    }

    // Compute API mode from combined state
    const apiMode = agentEnabled
      ? "agent"
      : sourceMode === "documents"
        ? "chat"
        : "general";

    // Add streaming assistant message
    const assistantId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        isStreaming: true,
        query: messageText, // Store the user query for source highlighting
      },
    ]);

    try {
      // Use streaming for agent mode to receive real-time updates
      if (agentEnabled) {
        const { api } = await import("@/lib/api/client");
        let streamContent = "";

        for await (const chunk of api.streamChatCompletion({
          message: messageText,
          session_id: currentSessionId || undefined,
          mode: "agent",
          agent_options: {
            ...agentOptions,
            search_documents: sourceMode === "documents", // Sync with source mode
          },
          include_collection_context: includeCollectionContext,
          collection_filters: selectedCollections.length > 0 ? selectedCollections : undefined,
          folder_id: selectedFolderId || undefined,
          include_subfolders: includeSubfolders,
          top_k: topK || undefined,
          language: outputLanguage, // Output language for response
          enhance_query: enhanceQuery ?? undefined, // Per-query enhancement override
        })) {
          switch (chunk.type) {
            case "session":
              if (chunk.session_id && !currentSessionId) {
                setCurrentSessionId(chunk.session_id);
              }
              break;

            case "planning":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content: "Planning execution steps...",
                        isAgentResponse: true,
                        planningDetails: "Analyzing request and creating execution plan..."
                      }
                    : m
                )
              );
              break;

            case "plan_created":
              if (chunk.steps) {
                setAgentSteps(chunk.steps);
                // Update message with plan details
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          isAgentResponse: true,
                          planningDetails: chunk.plan_summary || `Created plan with ${chunk.steps?.length || 0} steps`,
                          executionSteps: chunk.steps
                        }
                      : m
                  )
                );
              }
              break;

            case "approval_required":
              // Show cost approval dialog
              setPendingPlanId(chunk.plan_id || "");
              setPendingPlanSummary(chunk.plan_summary || null);
              setPendingEstimatedCost(chunk.estimated_cost || 0);
              setPendingSteps(chunk.steps || null);
              setShowCostApproval(true);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: "Awaiting approval...", isStreaming: false }
                    : m
                )
              );
              // Keep loading state until user approves/cancels
              return;

            case "agent_step":
              setIsAgentExecuting(true);
              if (chunk.step_index !== undefined) {
                setCurrentAgentStep(chunk.step_index);
              }
              if (chunk.step) {
                setAgentSteps((prev) => {
                  const updated = [...prev];
                  const idx = chunk.step_index ?? prev.length;
                  if (updated[idx]) {
                    updated[idx] = { ...updated[idx], ...chunk.step, status: "in_progress" };
                  }
                  return updated;
                });
              }
              break;

            case "step_completed":
              if (chunk.step_index !== undefined) {
                // Use full_output if available, fallback to output_preview
                const stepOutput = chunk.full_output || chunk.output_preview;
                setAgentSteps((prev) => {
                  const updated = [...prev];
                  if (updated[chunk.step_index!]) {
                    updated[chunk.step_index!] = {
                      ...updated[chunk.step_index!],
                      status: "completed",
                      output: stepOutput,
                    };
                  }
                  return updated;
                });
                // Update message with completed steps and output
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          executionSteps: prev.find(msg => msg.id === assistantId)?.executionSteps?.map((step, idx) =>
                            idx === chunk.step_index ? { ...step, status: "completed" as const, output: stepOutput } : step
                          )
                        }
                      : m
                  )
                );
              }
              break;

            case "content":
              if (chunk.data) {
                streamContent += chunk.data as string;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: streamContent }
                      : m
                  )
                );
              }
              break;

            case "sources":
              if (chunk.data && Array.isArray(chunk.data)) {
                const newSources: Source[] = chunk.data.map((s: Record<string, unknown>) => {
                  const docId = (s.document_id as string) || (s.source as string) || "unknown";
                  // Use similarity_score (original vector similarity 0-1) for display,
                  // fallback to relevance_score (RRF score ~0.01-0.03) if not available
                  const similarityValue = (s.similarity_score as number) || (s.score as number) || (s.relevance_score as number) || 0;
                  return {
                    documentId: docId,
                    filename: (s.document_name as string) || (s.source as string) || `Document ${docId.slice(0, 8)}`,
                    pageNumber: s.page_number as number | undefined,
                    snippet: ((s.snippet as string) || (s.content as string) || "").substring(0, 500),
                    fullContent: (s.full_content as string) || (s.content as string) || "",
                    similarity: similarityValue,
                    collection: (s.collection as string) || undefined,
                  };
                });

                // Update message with sources
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, sources: [...(m.sources || []), ...newSources] }
                      : m
                  )
                );

                // Auto-select for source panel
                if (newSources.length > 0) {
                  setSelectedMessageId(assistantId);
                }
              }
              break;

            case "suggestions":
              // Handle suggested follow-up questions
              if (chunk.data && Array.isArray(chunk.data)) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, suggestedQuestions: chunk.data as string[] }
                      : m
                  )
                );
              }
              break;

            case "execution_complete":
              setIsAgentExecuting(false);
              if (chunk.result) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          content: chunk.result!,
                          isStreaming: false,
                          isAgentResponse: true,
                          // Mark all steps as completed
                          executionSteps: m.executionSteps?.map(step => ({
                            ...step,
                            status: "completed" as const
                          }))
                        }
                      : m
                  )
                );
              }
              break;

            case "done":
              setMessages((prev) =>
                prev.map((m) => {
                  if (m.id !== assistantId) return m;
                  // Clean up SUGGESTED_QUESTIONS line from content if present
                  let cleanedContent = m.content;
                  if (cleanedContent.includes("SUGGESTED_QUESTIONS:")) {
                    cleanedContent = cleanedContent
                      .split("\n")
                      .filter(line => !line.trim().startsWith("SUGGESTED_QUESTIONS:"))
                      .join("\n")
                      .trim();
                  }
                  return { ...m, content: cleanedContent, isStreaming: false };
                })
              );
              break;

            case "thinking":
              // Append thinking/reasoning content
              if (chunk.content) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          thinkingContent: (m.thinkingContent || "") + chunk.content,
                          isAgentResponse: true,
                        }
                      : m
                  )
                );
              }
              break;

            case "error":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: `Error: ${chunk.error || "Unknown error"}`, isStreaming: false }
                    : m
                )
              );
              break;

            case "cancelled":
              setIsAgentExecuting(false);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: "Execution cancelled.", isStreaming: false }
                    : m
                )
              );
              break;
          }
        }
      } else {
        // Use non-streaming for documents/general mode
        // Prepare request with optional image attachments
        const requestPayload: Parameters<typeof sendMessage.mutateAsync>[0] = {
          message: messageText,
          session_id: currentSessionId || undefined,
          mode: hasImages ? "vision" : apiMode, // Use vision mode if images attached, otherwise computed mode
          include_collection_context: includeCollectionContext,
          collection_filters: selectedCollections.length > 0 ? selectedCollections : undefined,
          folder_id: selectedFolderId || undefined,
          include_subfolders: includeSubfolders,
          temp_session_id: tempSessionId || undefined,
          top_k: topK || undefined,
          language: outputLanguage, // Output language for response
          enhance_query: enhanceQuery ?? undefined, // Per-query enhancement override
          restrict_to_documents: sourceMode === "general" && restrictToDocuments, // Block AI knowledge in general mode
        };

        // Add images if present
        if (hasImages) {
          (requestPayload as unknown as Record<string, unknown>).images = imagesToSend.map((img) => ({
            data: img.data,
            mime_type: img.mimeType,
          }));
        }

        const response = await sendMessage.mutateAsync(requestPayload);

        // Update session ID if new
        if (response.session_id && !currentSessionId) {
          setCurrentSessionId(response.session_id);
        }

        // Transform sources
        const sources: Source[] =
          response.sources?.map((s) => ({
            documentId: s.document_id,
            filename: s.document_name || `Document ${s.document_id?.slice(0, 8) || 'unknown'}`,
            pageNumber: s.page_number,
            snippet: s.snippet,
            fullContent: s.full_content,
            // Use similarity_score (original vector similarity 0-1) for display,
            // fallback to relevance_score (RRF score ~0.01-0.03) if not available
            similarity: s.similarity_score ?? s.relevance_score,
            collection: s.collection,
          })) || [];

        // Update message with response including confidence
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: response.content,
                  sources,
                  isStreaming: false,
                  isGeneralResponse: response.is_general_response,
                  confidenceScore: response.confidence_score,
                  confidenceLevel: response.confidence_level as "high" | "medium" | "low" | undefined,
                }
              : m
          )
        );

        // Auto-select for source panel
        if (sources.length > 0) {
          setSelectedMessageId(assistantId);
        }
      }
    } catch (error) {
      console.error("Error:", error);
      // Show error message to user
      const errorMessage = getErrorMessage(error, "Failed to get a response. Please check your connection and try again.");

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: `Sorry, I encountered an error: ${errorMessage}`,
                isStreaming: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
      setIsAgentExecuting(false);
      inputRef.current?.focus();
    }
  };

  const handleCopy = async (content: string, id: string) => {
    await navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Export chat as markdown
  const handleExportChat = async () => {
    if (messages.length <= 1) return;

    const markdown = messages
      .filter(m => m.id !== "welcome")
      .map(m => {
        const role = m.role === "user" ? "**User:**" : "**Assistant:**";
        const sources = m.sources?.length
          ? `\n\n*Sources: ${m.sources.map(s => s.filename).join(", ")}*`
          : "";
        return `${role}\n\n${m.content}${sources}`;
      })
      .join("\n\n---\n\n");

    const header = `# Chat Export\n\nExported: ${new Date().toLocaleString()}\n\n---\n\n`;
    const content = header + markdown;

    // Copy to clipboard
    await navigator.clipboard.writeText(content);
    setCopiedId("export");
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Handle feedback submission
  const handleFeedback = async (messageId: string, isPositive: boolean) => {
    try {
      // Call feedback API
      const response = await fetch("/api/chat/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message_id: messageId,
          rating: isPositive ? 5 : 1,
        }),
      });

      if (response.ok) {
        // Visual feedback - you could add a toast notification here
        console.log("Feedback submitted:", { messageId, isPositive });
      }
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    }
  };

  const handleNewChat = async () => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content:
          "Hello! I'm your AI document assistant. How can I help you today?",
        timestamp: new Date(),
      },
    ]);
    setCurrentSessionId(null);
    setSelectedMessageId(null);
  };

  const handleLoadSession = (sessionId: string) => {
    setShouldLoadHistory(true); // Flag that we want to load history messages
    setCurrentSessionId(sessionId);
    setShowHistory(false);
  };

  // Agent approval handlers
  const handleApproveExecution = async () => {
    if (!pendingPlanId) return;
    try {
      await approveExecution.mutateAsync(pendingPlanId);
      setShowCostApproval(false);
      setIsAgentExecuting(true);
      // The execution will continue via streaming or polling
    } catch (error) {
      console.error("Failed to approve execution:", error);
    }
  };

  const handleCancelExecution = async () => {
    if (!pendingPlanId) return;
    try {
      await cancelExecution.mutateAsync(pendingPlanId);
      setShowCostApproval(false);
      setPendingPlanId(null);
      setPendingSteps(null);
      setIsLoading(false);
    } catch (error) {
      console.error("Failed to cancel execution:", error);
    }
  };

  const suggestedQuestions = [
    "What were our main achievements last quarter?",
    "Show me past stadium activation ideas",
    "What's our brand strategy for 2024?",
    "Find budget templates from previous events",
  ];

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-2 sm:gap-4 overflow-hidden">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div className="shrink-0">
            <h1 className="text-2xl font-bold">Chat with Documents</h1>
            <p className="text-muted-foreground">
              Ask questions and get answers from your knowledge base
            </p>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-end overflow-x-auto max-w-full pb-1">
            {/* Temperature Settings */}
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2">
                  <Thermometer className="h-4 w-4" />
                  <span className="hidden sm:inline">{temperature.toFixed(1)}</span>
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-72" align="end">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm font-medium">Temperature</Label>
                      <span className="text-sm text-muted-foreground">
                        {getTemperatureLabel(temperature)}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground w-6">0</span>
                      <Slider
                        value={[temperature]}
                        onValueChange={handleTemperatureChange}
                        min={0}
                        max={2}
                        step={0.1}
                        className="flex-1"
                      />
                      <span className="text-xs text-muted-foreground w-6">2</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Lower values make responses more focused and deterministic.
                      Higher values make responses more creative and varied.
                    </p>
                  </div>
                  <div className="pt-2 border-t">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Current value:</span>
                      <span className="font-medium">{temperature.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </PopoverContent>
            </Popover>

            {/* Documents to Search */}
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant={topK ? "default" : "outline"}
                  size="sm"
                  className="gap-2"
                  title="Documents to search per query"
                >
                  <FileSearch className="h-4 w-4" />
                  <span className="text-xs sm:text-sm">{topK ? `${topK}` : ""}</span>
                  <span className="hidden sm:inline">{topK ? "docs" : "Auto"}</span>
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-72" align="end">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm font-medium">Documents to Search</Label>
                      <span className="text-sm text-muted-foreground">
                        {topK ? `${topK} documents` : "Default"}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground w-6">3</span>
                      <Slider
                        value={[topK || 10]}
                        onValueChange={(value) => setTopK(value[0])}
                        min={3}
                        max={25}
                        step={1}
                        className="flex-1"
                      />
                      <span className="text-xs text-muted-foreground w-6">25</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      How many documents to search for each query. Higher = broader search but slower.
                    </p>
                  </div>
                  <div className="pt-2 border-t flex justify-between items-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setTopK(null)}
                      className="text-xs"
                    >
                      Reset to Default
                    </Button>
                    <span className="text-xs text-muted-foreground">
                      {topK ? `${topK} docs` : "Using admin setting"}
                    </span>
                  </div>
                </div>
              </PopoverContent>
            </Popover>

            {/* Query Enhancement Toggle (popover variant) */}
            {sourceMode === "documents" && (
              <QueryEnhancementToggle
                enabled={enhanceQuery ?? true}
                onChange={handleEnhanceQueryChange}
                variant="popover"
                disabled={isLoading}
              />
            )}

            {/* Output Language Selector - BEFORE Voice for visibility */}
            <Select value={outputLanguage} onValueChange={setOutputLanguage}>
              <SelectTrigger className="w-[100px] h-8 text-xs">
                <Globe className="h-3 w-3 mr-1 flex-shrink-0" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CHAT_LANGUAGES.map((lang) => (
                  <SelectItem key={lang.code} value={lang.code}>
                    {lang.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Voice Mode Toggle */}
            <Button
              variant={voiceModeEnabled ? "default" : "outline"}
              size="sm"
              onClick={() => {
                setVoiceModeEnabled(!voiceModeEnabled);
                if (!voiceModeEnabled) {
                  setVoiceState("idle");
                } else {
                  setVoiceState("idle");
                }
              }}
              className="gap-2"
              title={voiceModeEnabled ? "Disable voice conversation mode" : "Enable voice conversation mode"}
            >
              <Mic className={cn("h-4 w-4", voiceModeEnabled && "text-primary-foreground")} />
              <span className="hidden sm:inline">
                {voiceModeEnabled ? "Voice On" : "Voice"}
              </span>
            </Button>

            {/* Voice State Indicator - Only show when voice mode is enabled */}
            {voiceModeEnabled && voiceState !== "idle" && (
              <VoiceConversationIndicator state={voiceState} />
            )}

            {/* Agent Options Popover - Only show when agent mode is enabled */}
            {agentEnabled && (
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="outline" size="sm" className="gap-2">
                    <Settings2 className="h-4 w-4" />
                    <span className="hidden sm:inline">Agent Settings</span>
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-80" align="end">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <h4 className="font-medium leading-none flex items-center gap-2">
                        <Brain className="h-4 w-4" />
                        Agent Mode Options
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        Configure how the AI agent processes your requests.
                      </p>
                    </div>

                    {/* Search Documents Toggle */}
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm font-medium flex items-center gap-2">
                          <FileSearch className="h-3 w-3" />
                          Search Documents
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          Always search your uploaded documents first
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={agentOptions.search_documents}
                        onChange={(e) =>
                          setAgentOptions((prev) => ({
                            ...prev,
                            search_documents: e.target.checked,
                          }))
                        }
                        className="h-4 w-4 rounded border-gray-300"
                      />
                    </div>

                    {/* Include Web Search Toggle */}
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm font-medium flex items-center gap-2">
                          <Search className="h-3 w-3" />
                          Include Web Search
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          Also search the web for information
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={agentOptions.include_web_search}
                        onChange={(e) =>
                          setAgentOptions((prev) => ({
                            ...prev,
                            include_web_search: e.target.checked,
                          }))
                        }
                        className="h-4 w-4 rounded border-gray-300"
                      />
                    </div>

                    {/* Require Approval Toggle */}
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm font-medium flex items-center gap-2">
                          <Check className="h-3 w-3" />
                          Require Approval
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          Show plan before executing
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={agentOptions.require_approval}
                        onChange={(e) =>
                          setAgentOptions((prev) => ({
                            ...prev,
                            require_approval: e.target.checked,
                          }))
                        }
                        className="h-4 w-4 rounded border-gray-300"
                      />
                    </div>

                    {/* Max Steps Slider */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-sm font-medium flex items-center gap-2">
                          <Zap className="h-3 w-3" />
                          Max Steps
                        </Label>
                        <span className="text-sm text-muted-foreground">
                          {agentOptions.max_steps}
                        </span>
                      </div>
                      <Slider
                        value={[agentOptions.max_steps]}
                        onValueChange={([value]) =>
                          setAgentOptions((prev) => ({
                            ...prev,
                            max_steps: value,
                          }))
                        }
                        min={1}
                        max={10}
                        step={1}
                        className="w-full"
                      />
                      <p className="text-xs text-muted-foreground">
                        Limit complexity of execution plan
                      </p>
                    </div>

                    {/* Collection Context Toggle */}
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm font-medium flex items-center gap-2">
                          <Tag className="h-3 w-3" />
                          Show Collection Tags
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          Include collection names in AI context for better document grouping
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={includeCollectionContext}
                        onChange={(e) => setIncludeCollectionContext(e.target.checked)}
                        className="h-4 w-4 rounded border-gray-300"
                      />
                    </div>
                  </div>
                </PopoverContent>
              </Popover>
            )}

            {/* LLM Model Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2">
                  <Cpu className="h-4 w-4" />
                  <span className="hidden sm:inline">
                    {sessionLLMConfig?.provider_name || "Default LLM"}
                  </span>
                  {sessionLLMConfig?.model && (
                    <Badge variant="secondary" className="text-xs">
                      {sessionLLMConfig.model.split("/").pop()}
                    </Badge>
                  )}
                  <ChevronDown className="h-3 w-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-64">
                <DropdownMenuLabel>Select LLM Provider</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={async () => {
                    if (currentSessionId) {
                      // Use default by removing override
                      try {
                        await setSessionLLM.mutateAsync({
                          sessionId: currentSessionId,
                          data: { provider_id: "" },
                        });
                        refetchSessionLLM();
                      } catch {
                        // Ignore error if no override exists
                      }
                    }
                  }}
                  className="cursor-pointer"
                >
                  <div className="flex items-center justify-between w-full">
                    <span>Use Default</span>
                    {!sessionLLMConfig?.provider_id && (
                      <Check className="h-4 w-4 text-green-500" />
                    )}
                  </div>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                {providersData?.providers?.map((provider) => (
                  <DropdownMenuItem
                    key={provider.id}
                    onClick={async () => {
                      if (!currentSessionId) {
                        // Create a session first
                        const newSession = await createSession.mutateAsync(
                          "New Chat"
                        );
                        setCurrentSessionId(newSession.id);
                        // Then set the LLM
                        await setSessionLLM.mutateAsync({
                          sessionId: newSession.id,
                          data: {
                            provider_id: provider.id,
                            model_override: provider.default_chat_model || undefined,
                          },
                        });
                      } else {
                        await setSessionLLM.mutateAsync({
                          sessionId: currentSessionId,
                          data: {
                            provider_id: provider.id,
                            model_override: provider.default_chat_model || undefined,
                          },
                        });
                        refetchSessionLLM();
                      }
                    }}
                    className="cursor-pointer"
                  >
                    <div className="flex items-center justify-between w-full">
                      <div className="flex flex-col">
                        <span className="font-medium">{provider.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {provider.provider_type} • {provider.default_chat_model}
                        </span>
                      </div>
                      {sessionLLMConfig?.provider_id === provider.id && (
                        <Check className="h-4 w-4 text-green-500" />
                      )}
                    </div>
                  </DropdownMenuItem>
                ))}
                {(!providersData?.providers || providersData.providers.length === 0) && (
                  <div className="px-2 py-4 text-center text-sm text-muted-foreground">
                    No LLM providers configured.
                    <br />
                    <a href="/dashboard/admin/settings" className="text-primary underline">
                      Add providers in settings
                    </a>
                  </div>
                )}
              </DropdownMenuContent>
            </DropdownMenu>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowHistory(!showHistory)}
            >
              <Clock className="h-4 w-4 mr-2" />
              History
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleNewChat}
              title="New Chat (⌘N)"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              New Chat
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportChat}
              disabled={messages.length <= 1}
              title="Export Chat (⌘⇧E)"
            >
              {copiedId === "export" ? (
                <Check className="h-4 w-4 mr-2" />
              ) : (
                <Copy className="h-4 w-4 mr-2" />
              )}
              {copiedId === "export" ? "Copied!" : "Export"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSourcePanel(!showSourcePanel)}
              className="hidden lg:flex"
            >
              {showSourcePanel ? (
                <PanelRightClose className="h-4 w-4" />
              ) : (
                <PanelRightOpen className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* History Dropdown */}
        {showHistory && sessions && sessions.sessions?.length > 0 && (
          <Card className="absolute z-10 top-24 sm:top-32 right-2 sm:right-6 w-[min(calc(100vw-1rem),20rem)] p-2 shadow-lg max-h-[60vh] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-2 py-1 mb-2">
              <span className="text-sm font-medium text-muted-foreground">
                Recent Conversations
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs text-destructive hover:text-destructive"
                onClick={() => {
                  if (confirm(`Delete all ${sessions.sessions.length} conversations? This cannot be undone.`)) {
                    deleteAllSessions.mutate(undefined, {
                      onSuccess: () => {
                        setCurrentSessionId(null);
                        setMessages([{
                          id: "welcome",
                          role: "assistant",
                          content: "Hello! I'm your AI document assistant. I can help you find information across your document archive, answer questions, and even help generate new content. What would you like to know?",
                          timestamp: new Date(),
                        }]);
                      },
                    });
                  }
                }}
                disabled={deleteAllSessions.isPending}
              >
                {deleteAllSessions.isPending ? "Clearing..." : "Clear All"}
              </Button>
            </div>

            {/* Search History */}
            <div className="px-2 pb-2">
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  value={historySearchQuery}
                  onChange={(e) => setHistorySearchQuery(e.target.value)}
                  placeholder="Search conversations..."
                  className="h-8 pl-8 text-sm"
                  aria-label="Search chat history"
                />
              </div>
            </div>

            <ScrollArea className="flex-1 min-h-0">
              {filteredSessions.length > 0 ? (
                filteredSessions.map((session: { id: string; title: string; created_at: string }) => (
                  <div
                    key={session.id}
                    className="flex items-center justify-between p-2 rounded hover:bg-muted cursor-pointer"
                    onClick={() => handleLoadSession(session.id)}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <MessageSquare className="h-4 w-4 text-muted-foreground shrink-0" />
                      <span className="text-sm truncate">{session.title}</span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSession.mutate(session.id);
                      }}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                ))
              ) : (
                <div className="p-4 text-center text-sm text-muted-foreground">
                  No conversations match "{historySearchQuery}"
                </div>
              )}
            </ScrollArea>
          </Card>
        )}

        {/* Chat Messages */}
        <Card className="flex-1 flex flex-col overflow-hidden">
          <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
            <div className="space-y-6 max-w-3xl mx-auto" role="log" aria-label="Chat messages" aria-live="polite">
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-3 message-appear",
                    message.role === "user" ? "flex-row-reverse" : "flex-row"
                  )}
                >
                  <Avatar className="h-8 w-8 shrink-0">
                    <AvatarFallback
                      className={cn(
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted"
                      )}
                    >
                      {message.role === "user" ? (
                        <User className="h-4 w-4" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                    </AvatarFallback>
                  </Avatar>

                  <div
                    className={cn(
                      "flex-1 space-y-2",
                      message.role === "user" ? "text-right" : "text-left"
                    )}
                  >
                    <div
                      className={cn(
                        "inline-block rounded-lg px-4 py-2 max-w-full",
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : message.isAgentResponse
                          ? "bg-transparent p-0"
                          : "bg-muted"
                      )}
                    >
                      {/* Agent mode responses use collapsible sections */}
                      {message.isAgentResponse && !message.isStreaming ? (
                        <AgentResponseSections
                          planningDetails={message.planningDetails}
                          executionSteps={message.executionSteps}
                          finalAnswer={message.content}
                          isExecuting={false}
                          thinkingContent={message.thinkingContent}
                        />
                      ) : (
                        <div className="prose prose-sm dark:prose-invert max-w-none prose-chat text-left">
                          {message.isStreaming ? (
                            <>
                              {message.content}
                              <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
                            </>
                          ) : (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {message.content}
                            </ReactMarkdown>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Message Actions */}
                    {message.role === "assistant" && !message.isStreaming && message.content && (
                      <div className="flex items-center gap-2">
                        <TextToSpeech
                          text={message.content}
                          size="sm"
                          className="h-7 px-2"
                          autoPlay={voiceModeEnabled && index === messages.length - 1 && message.content !== lastAssistantMessage}
                          onStart={() => {
                            if (voiceModeEnabled) {
                              setVoiceState("speaking");
                              setLastAssistantMessage(message.content);
                            }
                          }}
                          onComplete={() => {
                            if (voiceModeEnabled) {
                              setVoiceState("listening");
                            }
                          }}
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2"
                          onClick={() => handleCopy(message.content, message.id)}
                        >
                          {copiedId === message.id ? (
                            <Check className="h-3 w-3" />
                          ) : (
                            <Copy className="h-3 w-3" />
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2"
                          onClick={() => handleFeedback(message.id, true)}
                          aria-label="Mark as helpful"
                        >
                          <ThumbsUp className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2"
                          onClick={() => handleFeedback(message.id, false)}
                          aria-label="Mark as not helpful"
                        >
                          <ThumbsDown className="h-3 w-3" />
                        </Button>
                        {message.sources && message.sources.length > 0 && (
                          <Button
                            variant={selectedMessageId === message.id ? "secondary" : "ghost"}
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() =>
                              setSelectedMessageId(
                                selectedMessageId === message.id ? null : message.id
                              )
                            }
                          >
                            <FileText className="h-3 w-3 mr-1" />
                            {message.sources.length} sources
                          </Button>
                        )}
                        {message.isGeneralResponse && (
                          <Button
                            variant={selectedMessageId === message.id ? "secondary" : "ghost"}
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() =>
                              setSelectedMessageId(
                                selectedMessageId === message.id ? null : message.id
                              )
                            }
                          >
                            <Brain className="h-3 w-3 mr-1" />
                            General
                          </Button>
                        )}
                        {/* Confidence Indicator */}
                        {message.confidenceLevel && (() => {
                          const confidence = getConfidenceDisplay(message.confidenceLevel, message.confidenceScore);
                          if (!confidence) return null;
                          const Icon = confidence.icon;
                          return (
                            <Popover>
                              <PopoverTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className={cn("h-7 px-2 text-xs gap-1", confidence.color)}
                                  title={confidence.label}
                                >
                                  <Icon className="h-3 w-3" />
                                  {confidence.score !== null && `${confidence.score}%`}
                                </Button>
                              </PopoverTrigger>
                              <PopoverContent className="w-64" align="start">
                                <div className="space-y-2">
                                  <div className="flex items-center gap-2">
                                    <Icon className={cn("h-5 w-5", confidence.color)} />
                                    <div>
                                      <p className="font-medium text-sm">{confidence.label}</p>
                                      {confidence.score !== null && (
                                        <p className="text-xs text-muted-foreground">
                                          Confidence score: {confidence.score}%
                                        </p>
                                      )}
                                    </div>
                                  </div>
                                  <p className="text-xs text-muted-foreground">
                                    {message.confidenceLevel === "high"
                                      ? "The answer is well-supported by the retrieved documents."
                                      : message.confidenceLevel === "medium"
                                      ? "Some relevant information found, but answer may be incomplete."
                                      : "Limited source support. Please verify this information."}
                                  </p>
                                </div>
                              </PopoverContent>
                            </Popover>
                          );
                        })()}
                      </div>
                    )}

                    {/* Suggested Follow-up Questions */}
                    {message.role === "assistant" && !message.isStreaming && message.suggestedQuestions && message.suggestedQuestions.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-muted">
                        <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
                          <Sparkles className="h-3 w-3" />
                          Related questions
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {message.suggestedQuestions.map((question, qIndex) => (
                            <Button
                              key={qIndex}
                              variant="outline"
                              size="sm"
                              className="h-auto py-1.5 px-3 text-xs text-left whitespace-normal"
                              onClick={() => {
                                setInput(question);
                              }}
                            >
                              {question}
                            </Button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Loading Indicator */}
              {isLoading && messages[messages.length - 1]?.content === "" && (
                <div className="flex gap-3">
                  <Avatar className="h-8 w-8 shrink-0">
                    <AvatarFallback className="bg-muted">
                      <Bot className="h-4 w-4" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="bg-muted rounded-lg px-4 py-3">
                    <div className="flex items-center gap-2">
                      {agentEnabled ? (
                        <Bot className="h-4 w-4 animate-pulse" />
                      ) : (
                        <Search className="h-4 w-4 animate-pulse" />
                      )}
                      <span className="text-sm text-muted-foreground">
                        {agentEnabled
                          ? isAgentExecuting && agentSteps.length > 0
                            ? `Executing step ${(currentAgentStep ?? 0) + 1}/${agentSteps.length}: ${agentSteps[currentAgentStep ?? 0]?.name || "Processing..."}`
                            : "Agents planning task..."
                          : sourceMode === "general"
                          ? "Thinking..."
                          : "Searching documents..."}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Suggested Questions */}
          {messages.length === 1 && (
            <div className="p-4 border-t bg-muted/50">
              <p className="text-sm text-muted-foreground mb-2">Try asking:</p>
              <div className="flex flex-wrap gap-2">
                {suggestedQuestions.map((question, idx) => (
                  <Button
                    key={idx}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    onClick={() => setInput(question)}
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="p-4 border-t">
            {/* Mode Selection Controls */}
            <div className="flex items-center justify-center gap-3 mb-3 flex-wrap">
              {/* Source Mode Toggle: Documents vs General */}
              <div className="flex items-center gap-1 p-1 bg-muted rounded-lg">
                <Button
                  variant={sourceMode === "documents" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setSourceMode("documents")}
                  className="gap-1.5 h-8"
                >
                  <FileSearch className="h-4 w-4" />
                  <span className="hidden sm:inline">Documents</span>
                </Button>
                <Button
                  variant={sourceMode === "general" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setSourceMode("general")}
                  className="gap-1.5 h-8"
                >
                  <Brain className="h-4 w-4" />
                  <span className="hidden sm:inline">General</span>
                </Button>
              </div>

              {/* No AI Knowledge Toggle - Only visible for general mode */}
              {sourceMode === "general" && (
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant={restrictToDocuments ? "default" : "outline"}
                      size="sm"
                      className={cn(
                        "gap-1.5 h-8",
                        restrictToDocuments && "bg-amber-500 hover:bg-amber-600 text-white"
                      )}
                      aria-label="Toggle AI knowledge restriction"
                    >
                      <BrainCog className="h-4 w-4" aria-hidden="true" />
                      <span className="hidden sm:inline">No AI Knowledge</span>
                      {restrictToDocuments && (
                        <Badge variant="secondary" className="text-[10px] px-1 py-0 bg-white/20 ml-1">
                          ON
                        </Badge>
                      )}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-80" align="start">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label htmlFor="restrict-toggle" className="text-sm font-medium">
                            Disable AI Knowledge
                          </Label>
                          <p className="text-xs text-muted-foreground">
                            Block pre-trained knowledge
                          </p>
                        </div>
                        <Switch
                          id="restrict-toggle"
                          checked={restrictToDocuments}
                          onCheckedChange={setRestrictToDocuments}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground border-t pt-2">
                        When enabled, the AI will not use its general knowledge and will only respond
                        based on your uploaded documents. Useful for ensuring answers come strictly
                        from your document base.
                      </p>
                    </div>
                  </PopoverContent>
                </Popover>
              )}

              {/* Agent Mode Toggle */}
              <Button
                variant={agentEnabled ? "default" : "outline"}
                size="sm"
                onClick={() => setAgentEnabled(!agentEnabled)}
                className={cn(
                  "gap-1.5 h-8",
                  agentEnabled && "bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600"
                )}
              >
                <Bot className="h-4 w-4" />
                <span className="hidden sm:inline">Agent</span>
                {agentEnabled && (
                  <Badge variant="secondary" className="text-[10px] px-1 py-0 bg-white/20 ml-1">
                    ON
                  </Badge>
                )}
              </Button>

              {/* Filter Toggle Button - Only visible for document mode */}
              {sourceMode === "documents" && (
                <Button
                  variant={showFilters ? "default" : "outline"}
                  size="sm"
                  onClick={() => setShowFilters(!showFilters)}
                  className="gap-1.5 h-8"
                  aria-label="Toggle document filters"
                  aria-expanded={showFilters}
                >
                  <Filter className="h-4 w-4" aria-hidden="true" />
                  <span className="hidden sm:inline">Filters</span>
                  {selectedCollections.length > 0 && (
                    <Badge variant="secondary" className="ml-1 h-5 px-1.5 text-xs">
                      {selectedCollections.length}
                    </Badge>
                  )}
                </Button>
              )}
              {/* Quick Upload Toggle Button */}
              <Button
                variant={showTempPanel ? "default" : "outline"}
                size="sm"
                onClick={() => setShowTempPanel(!showTempPanel)}
                className="gap-1.5 h-8"
                aria-label="Toggle quick document upload"
                aria-expanded={showTempPanel}
              >
                <Upload className="h-4 w-4" aria-hidden="true" />
                <span className="hidden sm:inline">Quick Upload</span>
                {tempDocuments.length > 0 && (
                  <Badge variant="secondary" className="ml-1 h-5 px-1.5 text-xs">
                    {tempDocuments.length}
                  </Badge>
                )}
              </Button>

              {/* Query Enhancement Toggle - Only for documents mode */}
              {sourceMode === "documents" && (
                <QueryEnhancementToggle
                  enabled={enhanceQuery ?? true}
                  onChange={handleEnhanceQueryChange}
                  variant="inline"
                  disabled={isLoading}
                />
              )}
            </div>

            {/* Voice Conversation Indicator */}
            {voiceModeEnabled && voiceState !== "idle" && (
              <div className="max-w-3xl mx-auto mb-3 flex justify-center">
                <VoiceConversationIndicator state={voiceState} />
              </div>
            )}

            {/* Document Filter Panel */}
            {showFilters && sourceMode === "documents" && (
              <div className="max-w-3xl mx-auto mb-3 max-h-48 sm:max-h-none overflow-y-auto space-y-3">
                <DocumentFilterPanel
                  collections={collectionsData?.collections || []}
                  selectedCollections={selectedCollections}
                  onCollectionsChange={setSelectedCollections}
                  totalDocuments={collectionsData?.total_documents}
                  isLoading={isLoadingCollections}
                  onRefresh={() => refetchCollections()}
                />
                {/* Folder Filter */}
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                  <span className="text-sm font-medium whitespace-nowrap">Folder scope:</span>
                  <div className="flex-1">
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
              </div>
            )}

            {/* Temp Document Panel (Quick Upload) */}
            {showTempPanel && (
              <div className="max-w-3xl mx-auto mb-3">
                <TempDocumentPanel
                  sessionId={tempSessionId}
                  documents={tempDocuments}
                  totalTokens={tempTotalTokens}
                  isLoading={isTempUploading}
                  onUpload={handleTempUpload}
                  onRemove={handleTempRemove}
                  onSave={handleTempSave}
                  onDiscard={handleTempDiscard}
                />
              </div>
            )}

            {/* Image Preview Bar */}
            {attachedImages.length > 0 && (
              <div className="max-w-3xl mx-auto mb-2">
                <ImagePreviewBar
                  images={attachedImages}
                  onRemove={(id) => setAttachedImages((prev) => prev.filter((img) => img.id !== id))}
                  disabled={isLoading}
                />
              </div>
            )}

            <form onSubmit={handleSubmit} className="flex gap-2 max-w-3xl mx-auto" role="search">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  attachedImages.length > 0
                    ? "Describe what you want to know about the image..."
                    : agentEnabled
                    ? "Describe a task for the agents to complete..."
                    : sourceMode === "general"
                    ? "Ask anything..."
                    : "Ask anything about your documents..."
                }
                disabled={isLoading || isAgentExecuting}
                className="flex-1"
                aria-label={
                  attachedImages.length > 0
                    ? "Ask about the attached image"
                    : agentEnabled
                    ? "Describe a task for AI agents"
                    : sourceMode === "general"
                    ? "Ask a general question"
                    : "Ask about your documents"
                }
              />
              <ImageUploadCompact
                images={attachedImages}
                onImagesChange={setAttachedImages}
                disabled={isLoading || isAgentExecuting}
                maxImages={4}
              />
              <VoiceInput
                onTranscript={(transcript) => {
                  if (voiceModeEnabled) {
                    // In voice mode, set input and trigger auto-send
                    setInput(transcript);
                  } else {
                    // Normal mode: append transcript
                    setInput((prev) => (prev ? `${prev} ${transcript}` : transcript));
                    inputRef.current?.focus();
                  }
                }}
                disabled={isLoading || isAgentExecuting}
                continuousMode={voiceModeEnabled}
                autoSend={voiceModeEnabled}
                onAutoSend={() => {
                  // Auto-send triggered by voice input
                  if (voiceModeEnabled && input.trim()) {
                    setVoiceState("thinking");
                    // Submit the form programmatically
                    const form = inputRef.current?.form;
                    if (form) {
                      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                    }
                  }
                }}
                onListeningStart={() => {
                  if (voiceModeEnabled) {
                    setVoiceState("listening");
                  }
                }}
                onListeningEnd={() => {
                  if (voiceModeEnabled && voiceState === "listening") {
                    setVoiceState("idle");
                  }
                }}
              />
              <Button
                type="submit"
                disabled={isLoading || !input.trim()}
                aria-label="Send message"
              >
                <Send className="h-4 w-4" />
              </Button>
            </form>
            <p className="text-xs text-muted-foreground text-center mt-2">
              {agentEnabled
                ? sourceMode === "documents"
                  ? "Multi-agent system will search your documents and complete complex tasks."
                  : "Multi-agent system will use general knowledge to complete complex tasks."
                : sourceMode === "documents"
                ? "Responses are generated from your document archive."
                : "Responses use general AI knowledge without document search."}
              <span className="hidden sm:inline ml-2 opacity-70">
                Press ⌘Enter to send • ⌘K to focus
              </span>
            </p>
          </div>
        </Card>
      </div>

      {/* Source Panel */}
      {showSourcePanel && (
        <div className="hidden lg:block w-80 shrink-0">
          <Card className="h-full flex flex-col">
            <div className="p-4 border-b">
              <h3 className="font-semibold flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Sources
              </h3>
              <p className="text-xs text-muted-foreground mt-1">
                {groupedSources.length > 0
                  ? `${groupedSources.length} ${groupedSources.length === 1 ? 'document' : 'documents'} (${selectedSources.length} chunks)`
                  : "Select a message to view sources"}
              </p>
            </div>

            <ScrollArea className="flex-1">
              {isSelectedMessageGeneral ? (
                <div className="p-8 text-center">
                  <Brain className="h-12 w-12 mx-auto mb-3 text-purple-500 opacity-70" />
                  <p className="text-sm font-medium">General Knowledge</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    This response was generated from AI general knowledge, not from your documents.
                  </p>
                </div>
              ) : groupedSources.length > 0 ? (
                <div className="p-4 space-y-3">
                  {groupedSources.map((group) => (
                    <Card
                      key={group.documentId}
                      className="overflow-hidden"
                    >
                      {/* Document Header - Click to expand/collapse */}
                      <div
                        className="p-3 hover:bg-muted/50 transition-colors cursor-pointer"
                        onClick={() => setExpandedSourceId(
                          expandedSourceId === group.documentId ? null : group.documentId
                        )}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            {expandedSourceId === group.documentId ? (
                              <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
                            ) : (
                              <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                            )}
                            <FileText className="h-4 w-4 text-primary shrink-0" />
                            <span className="text-sm font-medium truncate">
                              {group.filename}
                            </span>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 shrink-0"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSourceClick(group.documentId);
                            }}
                            title="Open document"
                          >
                            <ExternalLink className="h-3 w-3" />
                          </Button>
                        </div>

                        <div className="flex items-center gap-2 mt-2 ml-8 flex-wrap">
                          <Badge variant="secondary" className="text-xs">
                            {group.sources.length} {group.sources.length === 1 ? 'chunk' : 'chunks'}
                          </Badge>
                          {group.sources[0]?.collection && (
                            <Badge variant="outline" className="text-xs">
                              {group.sources[0].collection}
                            </Badge>
                          )}
                          <Badge
                            variant="secondary"
                            className={cn(
                              "text-xs",
                              group.avgSimilarity >= 0.8
                                ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                                : group.avgSimilarity >= 0.5
                                ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300"
                                : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                            )}
                          >
                            {Math.round(group.avgSimilarity * 100)}% match
                          </Badge>
                        </div>
                      </div>

                      {/* Expanded: Show individual chunks */}
                      {expandedSourceId === group.documentId && (
                        <div className="border-t bg-muted/30">
                          {group.sources.map((source, idx) => {
                            // Calculate global index for this source across all groups
                            const globalIndex = selectedSources.findIndex(
                              s => s.documentId === source.documentId && s.snippet === source.snippet
                            );
                            return (
                              <div
                                key={idx}
                                className="p-3 border-b last:border-b-0 cursor-pointer hover:bg-muted/50 transition-colors"
                                onClick={() => {
                                  setSourceViewerIndex(globalIndex >= 0 ? globalIndex : 0);
                                  setSourceViewerOpen(true);
                                }}
                                title="Click to view full content"
                              >
                                <div className="flex items-center gap-2 mb-2">
                                  {source.pageNumber && (
                                    <Badge variant="outline" className="text-xs">
                                      Page {source.pageNumber}
                                    </Badge>
                                  )}
                                  <Badge
                                    variant="secondary"
                                    className={cn(
                                      "text-xs",
                                      source.similarity >= 0.8
                                        ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                                        : source.similarity >= 0.5
                                        ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300"
                                        : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                                    )}
                                  >
                                    {Math.round(source.similarity * 100)}% match
                                  </Badge>
                                </div>
                                {/* Snippet with query term highlighting - show more lines */}
                                <p className="text-sm text-foreground/80 leading-relaxed">
                                  {highlightQueryTerms(
                                    source.snippet.length > 400
                                      ? source.snippet.substring(0, 400) + "..."
                                      : source.snippet,
                                    selectedMessage?.query || lastUserQuery
                                  )}
                                </p>
                                <p className="text-xs text-muted-foreground mt-2 italic">
                                  Click to view full content
                                </p>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="p-8 text-center text-muted-foreground">
                  <Sparkles className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">No sources selected</p>
                  <p className="text-xs mt-1">
                    Click "sources" on a message to view referenced documents
                  </p>
                </div>
              )}
            </ScrollArea>

            {/* Source Panel Footer */}
            {selectedSources.length > 0 && (
              <div className="p-4 border-t">
                <Button variant="outline" size="sm" className="w-full">
                  <Search className="h-4 w-4 mr-2" />
                  View All in Documents
                </Button>
              </div>
            )}

            {/* Agent Execution Progress - shown when agent is executing */}
            {isAgentExecuting && agentSteps.length > 0 && (
              <div className="p-4 border-t">
                <AgentExecutionProgress
                  steps={agentSteps}
                  currentStep={currentAgentStep}
                  isExecuting={isAgentExecuting}
                />
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Cost Approval Dialog */}
      <CostApprovalDialog
        open={showCostApproval}
        onOpenChange={setShowCostApproval}
        planId={pendingPlanId || ""}
        planSummary={pendingPlanSummary}
        estimatedCost={pendingEstimatedCost}
        steps={pendingSteps}
        onApprove={handleApproveExecution}
        onCancel={handleCancelExecution}
        isApproving={approveExecution.isPending}
        isCancelling={cancelExecution.isPending}
      />

      {/* Source Document Viewer Modal */}
      <SourceDocumentViewer
        open={sourceViewerOpen}
        onOpenChange={setSourceViewerOpen}
        sources={selectedSources.map(s => ({
          documentId: s.documentId,
          filename: s.filename,
          pageNumber: s.pageNumber,
          snippet: s.snippet,
          fullContent: s.fullContent,
          similarity: s.similarity,
          collection: s.collection,
        }))}
        currentIndex={sourceViewerIndex}
        query={selectedMessage?.query || lastUserQuery}
        onOpenDocument={handleSourceClick}
      />
    </div>
  );
}
