"use client";

import { useState, useRef, useEffect } from "react";
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
  Search,
  Sparkles,
  Settings2,
  Cpu,
  Thermometer,
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

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
  isStreaming?: boolean;
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

  // Mutations
  const createSession = useCreateChatSession();
  const deleteSession = useDeleteChatSession();
  const sendMessage = useSendChatMessage();
  const setSessionLLM = useSetSessionLLMConfig();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

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
  const selectedSources = messages.find((m) => m.id === selectedMessageId)?.sources || [];

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
    setInput("");
    setIsLoading(true);

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
      // Call actual API
      const response = await sendMessage.mutateAsync({
        message: input.trim(),
        session_id: currentSessionId || undefined,
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
              }
            : m
        )
      );

      // Auto-select for source panel
      if (sources.length > 0) {
        setSelectedMessageId(assistantId);
      }
    } catch (error) {
      console.error("Error:", error);
      // Show error message to user
      const errorMessage = (error as any)?.detail || "Failed to get a response. Please check your connection and try again.";

      const assistantId = Date.now().toString() + "-error";
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: `Sorry, I encountered an error: ${errorMessage}`,
          timestamp: new Date(),
          isStreaming: false,
        },
      ]);
      setIsLoading(false);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleCopy = async (content: string, id: string) => {
    await navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
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
    setCurrentSessionId(sessionId);
    setShowHistory(false);
    // Session messages would be loaded via useChatSession
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
            <Button variant="outline" size="sm" onClick={handleNewChat}>
              <RefreshCw className="h-4 w-4 mr-2" />
              New Chat
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
            <div className="space-y-6 max-w-3xl mx-auto">
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
                        <Button variant="ghost" size="sm" className="h-7 px-2">
                          <ThumbsUp className="h-3 w-3" />
                        </Button>
                        <Button variant="ghost" size="sm" className="h-7 px-2">
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
                      <Search className="h-4 w-4 animate-pulse" />
                      <span className="text-sm text-muted-foreground">
                        Searching documents...
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
            <form onSubmit={handleSubmit} className="flex gap-2 max-w-3xl mx-auto">
              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything about your documents..."
                disabled={isLoading}
                className="flex-1"
              />
              <Button type="submit" disabled={isLoading || !input.trim()}>
                <Send className="h-4 w-4" />
              </Button>
            </form>
            <p className="text-xs text-muted-foreground text-center mt-2">
              Responses are generated from your document archive. Always verify
              important information.
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
                {selectedSources.length > 0
                  ? `${selectedSources.length} documents referenced`
                  : "Select a message to view sources"}
              </p>
            </div>

            <ScrollArea className="flex-1">
              {selectedSources.length > 0 ? (
                <div className="p-4 space-y-3">
                  {selectedSources.map((source, idx) => (
                    <Card
                      key={idx}
                      className="p-3 hover:bg-muted/50 transition-colors cursor-pointer"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <FileText className="h-4 w-4 text-primary shrink-0" />
                          <span className="text-sm font-medium truncate">
                            {source.filename}
                          </span>
                        </div>
                        <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0">
                          <ExternalLink className="h-3 w-3" />
                        </Button>
                      </div>

                      {source.pageNumber && (
                        <p className="text-xs text-muted-foreground mt-1 ml-6">
                          Page {source.pageNumber}
                        </p>
                      )}

                      <p className="text-sm text-muted-foreground mt-2 line-clamp-3">
                        {source.snippet}
                      </p>

                      <div className="flex items-center gap-2 mt-2">
                        <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary rounded-full"
                            style={{ width: `${source.similarity * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium text-primary">
                          {Math.round(source.similarity * 100)}%
                        </span>
                      </div>

                      {source.collection && (
                        <span className="inline-block text-xs bg-muted px-2 py-0.5 rounded mt-2">
                          {source.collection}
                        </span>
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
          </Card>
        </div>
      )}
    </div>
  );
}
