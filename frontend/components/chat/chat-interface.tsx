"use client";

import * as React from "react";
import { Send, Paperclip, StopCircle, FileText, ExternalLink, Loader2, Settings2, Download } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card } from "@/components/ui/card";
import { TokenCounter } from "./token-counter";
import { PromptTemplatesDialog } from "./prompt-templates-dialog";
import { ExportChatButton } from "./export-chat-button";
import { countChatTokens } from "@/lib/utils/tokenizer";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
  isStreaming?: boolean;
}

export interface Source {
  documentId: string;
  documentName: string;
  chunkId: string;
  pageNumber?: number;
  relevanceScore: number;
  snippet: string;
}

interface ChatInterfaceProps {
  sessionId?: string;
  className?: string;
  modelId?: string;
}

export function ChatInterface({ sessionId: initialSessionId, className, modelId = "gpt-4o" }: ChatInterfaceProps) {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const [isLoading, setIsLoading] = React.useState(false);
  const [showSources, setShowSources] = React.useState(true);
  const [sessionId, setSessionId] = React.useState(initialSessionId || crypto.randomUUID());
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const abortControllerRef = React.useRef<AbortController | null>(null);

  // Calculate total tokens from conversation history
  const previousTokens = React.useMemo(() => {
    if (messages.length === 0) return 0;
    return countChatTokens(
      messages.map((m) => ({ role: m.role, content: m.content }))
    );
  }, [messages]);

  // Auto-scroll to bottom on new messages
  React.useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Create placeholder for assistant response
    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, assistantMessage]);

    // Create abort controller for this request
    abortControllerRef.current = new AbortController();

    try {
      // Demo mode: simulating streaming response
      // In production, replace with: fetch('/api/chat/completions/stream', { signal: abortControllerRef.current.signal, ... })
      const mockResponse = "Based on the documents in your knowledge base, I found relevant information about your query. The Q4 Strategy Presentation mentions that the key focus areas for the upcoming quarter include digital transformation initiatives and customer experience improvements. Additionally, the Marketing Report 2024 highlights the importance of data-driven decision making.";

      const mockSources: Source[] = [
        {
          documentId: "doc-1",
          documentName: "Q4 Strategy Presentation.pptx",
          chunkId: "chunk-1",
          pageNumber: 5,
          relevanceScore: 0.92,
          snippet: "Key focus areas for Q4: Digital transformation, customer experience...",
        },
        {
          documentId: "doc-2",
          documentName: "Marketing Report 2024.pdf",
          chunkId: "chunk-2",
          pageNumber: 12,
          relevanceScore: 0.87,
          snippet: "Data-driven decision making is essential for modern marketing...",
        },
      ];

      // Simulate streaming with abort support
      let streamedContent = "";
      for (let i = 0; i < mockResponse.length; i++) {
        // Check if request was aborted
        if (abortControllerRef.current?.signal.aborted) {
          break;
        }
        streamedContent += mockResponse[i];
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessage.id
              ? { ...msg, content: streamedContent }
              : msg
          )
        );
        await new Promise((resolve) => setTimeout(resolve, 20));
      }

      // Add sources after streaming completes
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? { ...msg, isStreaming: false, sources: mockSources }
            : msg
        )
      );
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: "Sorry, an error occurred while processing your request. Please try again.",
                isStreaming: false,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleStop = () => {
    // Abort the current streaming request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Chat Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div>
          <h2 className="font-semibold">AI Assistant</h2>
          <p className="text-sm text-muted-foreground">
            Ask questions about your documents
          </p>
        </div>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <ExportChatButton sessionId={sessionId} />
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSources(!showSources)}
          >
            <Settings2 className="mr-2 h-4 w-4" />
            {showSources ? "Hide Sources" : "Show Sources"}
          </Button>
        </div>
      </div>

      {/* Messages Area */}
      <ScrollArea ref={scrollRef} className="flex-1 p-4">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                showSources={showSources}
              />
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <PromptTemplatesDialog
            onApply={(text) => setInput(text)}
            trigger={
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="shrink-0"
                title="Prompt templates"
              >
                <FileText className="h-5 w-5" />
              </Button>
            }
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="shrink-0"
            title="Attach files (coming soon)"
            disabled
          >
            <Paperclip className="h-5 w-5" />
          </Button>
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={isLoading}
            className="flex-1"
          />
          {isLoading ? (
            <Button
              type="button"
              variant="destructive"
              size="icon"
              onClick={handleStop}
              className="shrink-0"
            >
              <StopCircle className="h-5 w-5" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim()}
              className="shrink-0"
            >
              <Send className="h-5 w-5" />
            </Button>
          )}
        </form>
        {/* Token Counter */}
        <div className="mt-2 flex items-center justify-between">
          <TokenCounter
            text={input}
            modelId={modelId}
            previousTokens={previousTokens}
            showCost={true}
          />
          <p className="text-xs text-muted-foreground">
            Verify important information
          </p>
        </div>
      </div>
    </div>
  );
}

interface ChatMessageProps {
  message: Message;
  showSources: boolean;
}

function ChatMessage({ message, showSources }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <Avatar className="h-8 w-8 shrink-0">
        <AvatarFallback className={isUser ? "bg-primary text-primary-foreground" : "bg-muted"}>
          {isUser ? "U" : "AI"}
        </AvatarFallback>
      </Avatar>

      <div
        className={cn(
          "flex flex-col gap-2 max-w-[80%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div
          className={cn(
            "rounded-lg px-4 py-2",
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted"
          )}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
          {message.isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
          )}
        </div>

        {/* Sources */}
        {showSources && message.sources && message.sources.length > 0 && (
          <div className="w-full space-y-2">
            <p className="text-xs font-medium text-muted-foreground">
              Sources ({message.sources.length})
            </p>
            <div className="grid gap-2">
              {message.sources.map((source) => (
                <SourceCard key={source.chunkId} source={source} />
              ))}
            </div>
          </div>
        )}

        <span className="text-xs text-muted-foreground">
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </span>
      </div>
    </div>
  );
}

interface SourceCardProps {
  source: Source;
}

function SourceCard({ source }: SourceCardProps) {
  return (
    <Card className="p-3 text-sm">
      <div className="flex items-start gap-2">
        <FileText className="h-4 w-4 mt-0.5 shrink-0 text-muted-foreground" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <p className="font-medium truncate">{source.documentName}</p>
            <span className="text-xs text-muted-foreground shrink-0">
              {Math.round(source.relevanceScore * 100)}% match
            </span>
          </div>
          {source.pageNumber && (
            <p className="text-xs text-muted-foreground">
              Page {source.pageNumber}
            </p>
          )}
          <p className="mt-1 text-muted-foreground line-clamp-2">
            {source.snippet}
          </p>
        </div>
        <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0">
          <ExternalLink className="h-3 w-3" />
        </Button>
      </div>
    </Card>
  );
}

function EmptyState() {
  const suggestions = [
    "What were the key initiatives from last quarter?",
    "Summarize the main findings from recent reports",
    "What events did we organize in 2024?",
    "Find presentations about digital transformation",
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 mb-4">
        <FileText className="h-8 w-8 text-primary" />
      </div>
      <h3 className="text-lg font-semibold mb-2">
        Start a Conversation
      </h3>
      <p className="text-muted-foreground mb-6 max-w-md">
        Ask questions about your documents and get answers with source citations.
        The AI will search through your entire knowledge base.
      </p>
      <div className="grid gap-2 w-full max-w-md">
        <p className="text-sm font-medium text-muted-foreground">
          Try asking:
        </p>
        {suggestions.map((suggestion, i) => (
          <Button
            key={i}
            variant="outline"
            className="justify-start h-auto py-2 px-3 text-left"
          >
            <span className="truncate">{suggestion}</span>
          </Button>
        ))}
      </div>
    </div>
  );
}
