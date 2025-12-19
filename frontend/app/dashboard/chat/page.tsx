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
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import {
  useChatSessions,
  useChatSession,
  useCreateChatSession,
  useDeleteChatSession,
  useSendChatMessage,
} from "@/lib/api";
import {
  useLLMProviders,
  useSessionLLMConfig,
  useSetSessionLLMConfig,
  useAgentMode,
  useApproveAgentExecution,
  useCancelAgentExecution,
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
import { AgentModeToggle } from "@/components/agent-mode-toggle";
import { CostApprovalDialog } from "@/components/cost-approval-dialog";
import { AgentExecutionProgress } from "@/components/agent-execution-progress";
import type { PlanStep, ExecutionMode } from "@/lib/api/client";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
  isStreaming?: boolean;
  isGeneralResponse?: boolean;
}

interface Source {
  documentId: string;
  filename: string;
  pageNumber?: number;
  snippet: string;
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
  const [chatMode, setChatMode] = useState<"chat" | "general" | "agent">("chat");
  // Agent mode options - user-configurable settings for agent execution
  const [agentOptions, setAgentOptions] = useState({
    search_documents: true,      // Force search uploaded documents first
    include_web_search: false,   // Include web search in research
    require_approval: false,     // Show plan and require approval before execution
    max_steps: 5,                // Maximum number of steps in execution plan
    collection: null as string | null,  // Target specific collection (null = all)
  });
  const [showCostApproval, setShowCostApproval] = useState(false);
  const [pendingPlanId, setPendingPlanId] = useState<string | null>(null);
  const [pendingPlanSummary, setPendingPlanSummary] = useState<string | null>(null);
  const [pendingEstimatedCost, setPendingEstimatedCost] = useState<number>(0);
  const [pendingSteps, setPendingSteps] = useState<PlanStep[] | null>(null);
  const [agentSteps, setAgentSteps] = useState<PlanStep[]>([]);
  const [currentAgentStep, setCurrentAgentStep] = useState<number>(0);
  const [isAgentExecuting, setIsAgentExecuting] = useState(false);
  const [expandedSourceId, setExpandedSourceId] = useState<string | null>(null);
  const [shouldLoadHistory, setShouldLoadHistory] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Queries - wait for auth to be ready before making API calls
  const { data: sessions } = useChatSessions(undefined, { enabled: isAuthenticated });
  const { data: currentSession } = useChatSession(currentSessionId || "");
  const { data: providersData } = useLLMProviders({ enabled: isAuthenticated });
  const { data: sessionLLMConfig, refetch: refetchSessionLLM } = useSessionLLMConfig(
    currentSessionId || "",
    { enabled: isAuthenticated && !!currentSessionId }
  );

  // Agent mode query
  const { data: agentModeData } = useAgentMode(currentSessionId || undefined, { enabled: isAuthenticated });

  // Sync messages when session data is loaded from history
  // Only runs when user explicitly clicks a history session (shouldLoadHistory flag)
  useEffect(() => {
    if (shouldLoadHistory && currentSessionId && currentSession?.messages && currentSession.messages.length > 0) {
      // Transform API messages to local Message format
      const loadedMessages: Message[] = currentSession.messages.map((msg, index) => ({
        id: `session-msg-${index}`,
        role: msg.role as "user" | "assistant",
        content: msg.content,
        timestamp: new Date(),
      }));
      setMessages(loadedMessages);
      setShouldLoadHistory(false); // Reset flag after loading
    }
  }, [shouldLoadHistory, currentSessionId, currentSession]);

  // Mutations
  const createSession = useCreateChatSession();
  const deleteSession = useDeleteChatSession();
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

  // Helper to get temperature description
  const getTemperatureLabel = (temp: number) => {
    if (temp <= 0.3) return "Precise";
    if (temp <= 0.7) return "Balanced";
    if (temp <= 1.0) return "Creative";
    return "Very Creative";
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageText = input.trim();
    setInput("");
    setIsLoading(true);

    // Reset agent state when starting new request
    if (chatMode === "agent") {
      setAgentSteps([]);
      setCurrentAgentStep(0);
      setIsAgentExecuting(false);
    }

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
      },
    ]);

    try {
      // Use streaming for agent mode to receive real-time updates
      if (chatMode === "agent") {
        const { api } = await import("@/lib/api/client");
        let streamContent = "";

        for await (const chunk of api.streamChatCompletion({
          message: messageText,
          session_id: currentSessionId || undefined,
          mode: "agent",
          agent_options: agentOptions,
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
                    ? { ...m, content: "Planning execution steps..." }
                    : m
                )
              );
              break;

            case "plan_created":
              if (chunk.steps) {
                setAgentSteps(chunk.steps);
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
                setAgentSteps((prev) => {
                  const updated = [...prev];
                  if (updated[chunk.step_index!]) {
                    updated[chunk.step_index!] = { ...updated[chunk.step_index!], status: "completed" };
                  }
                  return updated;
                });
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
                const newSources: Source[] = chunk.data.map((s: Record<string, unknown>) => ({
                  documentId: (s.document_id as string) || (s.source as string) || "unknown",
                  filename: (s.source as string) || (s.document_name as string) || "Document",
                  pageNumber: s.page_number as number | undefined,
                  snippet: ((s.content as string) || "").substring(0, 200),
                  similarity: (s.score as number) || 0,
                }));

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

            case "execution_complete":
              setIsAgentExecuting(false);
              if (chunk.result) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: chunk.result!, isStreaming: false }
                      : m
                  )
                );
              }
              break;

            case "done":
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, isStreaming: false }
                    : m
                )
              );
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
        const response = await sendMessage.mutateAsync({
          message: messageText,
          session_id: currentSessionId || undefined,
          mode: chatMode,
        });

        // Update session ID if new
        if (response.session_id && !currentSessionId) {
          setCurrentSessionId(response.session_id);
        }

        // Transform sources
        const sources: Source[] =
          response.sources?.map((s) => ({
            documentId: s.document_id,
            filename: s.document_name,
            pageNumber: s.page_number,
            snippet: s.snippet,
            similarity: s.relevance_score,
            collection: undefined,
          })) || [];

        // Update message with response
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: response.content,
                  sources,
                  isStreaming: false,
                  isGeneralResponse: response.is_general_response,
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
      const errorMessage = (error as any)?.detail || "Failed to get a response. Please check your connection and try again.";

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

  const handleModeChange = (mode: ExecutionMode) => {
    setAgentExecutionMode(mode);
  };

  const suggestedQuestions = [
    "What were our main achievements last quarter?",
    "Show me past stadium activation ideas",
    "What's our brand strategy for 2024?",
    "Find budget templates from previous events",
  ];

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold">Chat with Documents</h1>
            <p className="text-muted-foreground">
              Ask questions and get answers from your knowledge base
            </p>
          </div>
          <div className="flex items-center gap-2">
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

            {/* Agent Mode Toggle */}
            <AgentModeToggle
              sessionId={currentSessionId || undefined}
              showLabel={false}
              onModeChange={handleModeChange}
            />

            {/* Agent Options Popover - Only show when in agent mode */}
            {chatMode === "agent" && (
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
          <Card className="absolute z-10 top-32 right-6 w-80 p-2 shadow-lg">
            <div className="text-sm font-medium px-2 py-1 text-muted-foreground">
              Recent Conversations
            </div>
            <ScrollArea className="max-h-64">
              {sessions.sessions.map((session: { id: string; title: string; created_at: string }) => (
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
              ))}
            </ScrollArea>
          </Card>
        )}

        {/* Chat Messages */}
        <Card className="flex-1 flex flex-col overflow-hidden">
          <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
            <div className="space-y-6 max-w-3xl mx-auto" role="log" aria-label="Chat messages" aria-live="polite">
              {messages.map((message) => (
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
                          : "bg-muted"
                      )}
                    >
                      <div className="prose-chat whitespace-pre-wrap text-left">
                        {message.content}
                        {message.isStreaming && (
                          <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
                        )}
                      </div>
                    </div>

                    {/* Message Actions */}
                    {message.role === "assistant" && !message.isStreaming && message.content && (
                      <div className="flex items-center gap-2">
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
                      {chatMode === "agent" ? (
                        <Bot className="h-4 w-4 animate-pulse" />
                      ) : (
                        <Search className="h-4 w-4 animate-pulse" />
                      )}
                      <span className="text-sm text-muted-foreground">
                        {chatMode === "agent"
                          ? isAgentExecuting && agentSteps.length > 0
                            ? `Executing step ${(currentAgentStep ?? 0) + 1}/${agentSteps.length}: ${agentSteps[currentAgentStep ?? 0]?.name || "Processing..."}`
                            : "Agents planning task..."
                          : chatMode === "general"
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
            {/* Mode Selector */}
            <div className="flex justify-center mb-3" role="tablist" aria-label="Chat mode">
              <div className="inline-flex rounded-lg bg-muted p-1">
                <Button
                  variant={chatMode === "chat" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setChatMode("chat")}
                  className="gap-1.5"
                  role="tab"
                  aria-selected={chatMode === "chat"}
                  aria-label="Document chat mode - search your documents"
                >
                  <FileSearch className="h-4 w-4" aria-hidden="true" />
                  Documents
                </Button>
                <Button
                  variant={chatMode === "general" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setChatMode("general")}
                  className="gap-1.5"
                  disabled={!agentModeData?.general_chat_enabled}
                  title={!agentModeData?.general_chat_enabled ? "Enable General Chat in settings" : undefined}
                  role="tab"
                  aria-selected={chatMode === "general"}
                  aria-label="General chat mode - AI knowledge without document search"
                >
                  <Brain className="h-4 w-4" aria-hidden="true" />
                  General
                </Button>
                <Button
                  variant={chatMode === "agent" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setChatMode("agent")}
                  className="gap-1.5"
                  disabled={!agentModeData?.agent_mode_enabled}
                  title={!agentModeData?.agent_mode_enabled ? "Enable Agent Mode in settings" : undefined}
                  role="tab"
                  aria-selected={chatMode === "agent"}
                  aria-label="Agent mode - multi-step task execution with AI agents"
                >
                  <Bot className="h-4 w-4" aria-hidden="true" />
                  Agent
                </Button>
              </div>
            </div>
            <form onSubmit={handleSubmit} className="flex gap-2 max-w-3xl mx-auto" role="search">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  chatMode === "agent"
                    ? "Describe a task for the agents to complete..."
                    : chatMode === "general"
                    ? "Ask anything..."
                    : "Ask anything about your documents..."
                }
                disabled={isLoading || isAgentExecuting}
                className="flex-1"
                aria-label={
                  chatMode === "agent"
                    ? "Describe a task for AI agents"
                    : chatMode === "general"
                    ? "Ask a general question"
                    : "Ask about your documents"
                }
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
              {chatMode === "agent"
                ? "Multi-agent system will plan and execute complex tasks step by step."
                : chatMode === "general"
                ? "Responses use general AI knowledge without document search."
                : "Responses are generated from your document archive. Always verify important information."}
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

                        <div className="flex items-center gap-2 mt-2 ml-8">
                          <Badge variant="secondary" className="text-xs">
                            {group.sources.length} {group.sources.length === 1 ? 'chunk' : 'chunks'}
                          </Badge>
                          <div className="flex items-center gap-1 flex-1">
                            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden max-w-20">
                              <div
                                className="h-full bg-primary rounded-full"
                                style={{ width: `${group.avgSimilarity * 100}%` }}
                              />
                            </div>
                            <span className="text-xs text-muted-foreground">
                              {Math.round(group.avgSimilarity * 100)}% avg
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Expanded: Show individual chunks */}
                      {expandedSourceId === group.documentId && (
                        <div className="border-t bg-muted/30">
                          {group.sources.map((source, idx) => (
                            <div
                              key={idx}
                              className="p-3 border-b last:border-b-0"
                            >
                              {source.pageNumber && (
                                <p className="text-xs text-muted-foreground mb-1">
                                  Page {source.pageNumber}
                                </p>
                              )}
                              <p className="text-sm text-muted-foreground line-clamp-3">
                                {source.snippet}
                              </p>
                              <div className="flex items-center gap-2 mt-2">
                                <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-primary/70 rounded-full"
                                    style={{ width: `${source.similarity * 100}%` }}
                                  />
                                </div>
                                <span className="text-xs text-muted-foreground">
                                  {Math.round(source.similarity * 100)}%
                                </span>
                              </div>
                            </div>
                          ))}
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
    </div>
  );
}
