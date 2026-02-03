import { useState, useRef, useEffect } from 'react';
import {
  Send,
  Loader2,
  Trash2,
  FileText,
  ChevronDown,
  MessageSquare,
  Sparkles,
  Bot,
  User,
  Copy,
  Check,
  RefreshCw,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useAppStore, type Message, type Source } from '../lib/store';
import { chatLocal, chatServer } from '../lib/tauri';
import { cn, generateId } from '../lib/utils';

export function ChatPage() {
  const { mode, messages, addMessage, clearMessages, isProcessing, setIsProcessing } =
    useAppStore();
  const [input, setInput] = useState('');
  const [expandedSources, setExpandedSources] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    addMessage(userMessage);
    setInput('');
    setIsProcessing(true);

    try {
      const response =
        mode === 'local'
          ? await chatLocal(userMessage.content)
          : await chatServer(userMessage.content);

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        sources: response.sources,
      };

      addMessage(assistantMessage);
    } catch (error) {
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
        timestamp: new Date(),
      };
      addMessage(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleCopy = async (content: string, id: string) => {
    await navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex flex-col h-full page-enter">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-border/50 glass">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-gradient-to-br from-primary/20 to-purple-500/10">
            <MessageSquare className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">Chat</h1>
            <p className="text-xs text-muted-foreground">
              {messages.length} messages
            </p>
          </div>
        </div>
        <button
          onClick={clearMessages}
          disabled={messages.length === 0}
          className={cn(
            'flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium',
            'bg-muted/50 hover:bg-destructive/10 hover:text-destructive transition-all',
            'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-muted/50 disabled:hover:text-foreground'
          )}
        >
          <Trash2 className="w-4 h-4" />
          Clear
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 ? (
          <div className="empty-state h-full">
            <div className="empty-state-icon">
              <Sparkles className="w-8 h-8 text-primary/60" />
            </div>
            <h2 className="text-xl font-semibold mb-2">Start a conversation</h2>
            <p className="text-muted-foreground max-w-sm">
              Ask questions about your indexed documents and get AI-powered answers
            </p>
            <div className="mt-6 flex flex-wrap gap-2 justify-center">
              {['What documents do I have?', 'Summarize my notes', 'Find related topics'].map(
                (suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(suggestion)}
                    className="px-3 py-1.5 rounded-full text-sm bg-muted/50 hover:bg-muted transition-colors"
                  >
                    {suggestion}
                  </button>
                )
              )}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              expanded={expandedSources === message.id}
              onToggleSources={() =>
                setExpandedSources(
                  expandedSources === message.id ? null : message.id
                )
              }
              onCopy={() => handleCopy(message.content, message.id)}
              isCopied={copiedId === message.id}
              isLatest={index === messages.length - 1}
            />
          ))
        )}
        {isProcessing && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-border/50 glass">
        <div className="flex gap-3 items-end">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              className={cn(
                'w-full px-4 py-3 pr-12 rounded-2xl resize-none',
                'input-modern',
                'min-h-[52px] max-h-[200px]',
                'focus:outline-none focus:ring-2 focus:ring-primary/20'
              )}
              rows={1}
              disabled={isProcessing}
            />
            <div className="absolute right-3 bottom-3 text-xs text-muted-foreground/60">
              {mode === 'local' ? 'Ollama' : 'Cloud'}
            </div>
          </div>
          <button
            type="submit"
            disabled={!input.trim() || isProcessing}
            className={cn(
              'flex items-center justify-center w-12 h-12 rounded-xl',
              'btn-gradient',
              'disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none disabled:transform-none'
            )}
          >
            {isProcessing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
        <p className="text-[10px] text-muted-foreground/60 mt-2 text-center">
          Press Enter to send, Shift+Enter for new line
        </p>
      </form>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 message-appear">
      <div className="p-2 rounded-xl bg-gradient-to-br from-primary/20 to-purple-500/10 shrink-0">
        <Bot className="w-4 h-4 text-primary" />
      </div>
      <div className="message-assistant px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            <span className="typing-dot w-2 h-2 rounded-full bg-primary/60" />
            <span className="typing-dot w-2 h-2 rounded-full bg-primary/60" />
            <span className="typing-dot w-2 h-2 rounded-full bg-primary/60" />
          </div>
          <span className="text-sm text-muted-foreground">Thinking...</span>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  expanded,
  onToggleSources,
  onCopy,
  isCopied,
  isLatest,
}: {
  message: Message;
  expanded: boolean;
  onToggleSources: () => void;
  onCopy: () => void;
  isCopied: boolean;
  isLatest: boolean;
}) {
  const isUser = message.role === 'user';

  return (
    <div
      className={cn(
        'flex items-start gap-3',
        isUser ? 'flex-row-reverse' : '',
        isLatest && 'message-appear'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'p-2 rounded-xl shrink-0',
          isUser
            ? 'bg-gradient-to-br from-primary to-primary/80'
            : 'bg-gradient-to-br from-primary/20 to-purple-500/10'
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-primary-foreground" />
        ) : (
          <Bot className="w-4 h-4 text-primary" />
        )}
      </div>

      {/* Message */}
      <div
        className={cn(
          'max-w-[85%] group',
          isUser ? 'message-user' : 'message-assistant',
          'px-4 py-3'
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        ) : (
          <div className="message-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  const isInline = !match;
                  return isInline ? (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  ) : (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                      customStyle={{
                        borderRadius: '0.75rem',
                        margin: '0.75rem 0',
                      }}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Actions */}
        {!isUser && (
          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border/30 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={onCopy}
              className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
            >
              {isCopied ? (
                <>
                  <Check className="w-3 h-3 text-green-500" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-border/30">
            <button
              onClick={onToggleSources}
              className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>{message.sources.length} sources</span>
              <ChevronDown
                className={cn(
                  'w-3.5 h-3.5 transition-transform duration-200',
                  expanded && 'rotate-180'
                )}
              />
            </button>

            {expanded && (
              <div className="mt-3 space-y-2 slide-in-up">
                {message.sources.map((source, index) => (
                  <SourceCard key={index} source={source} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function SourceCard({ source }: { source: Source }) {
  return (
    <div className="source-card">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-sm font-medium truncate flex items-center gap-2">
          <FileText className="w-3.5 h-3.5 text-primary/60" />
          {source.document_name}
        </span>
        <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
          {(source.score * 100).toFixed(0)}%
        </span>
      </div>
      <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
        {source.content}
      </p>
    </div>
  );
}
