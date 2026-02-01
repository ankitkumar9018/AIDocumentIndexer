"use client";

import React, { useState, useCallback, useRef, useMemo, useEffect } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Copy,
  Download,
  X,
  Pencil,
  Check,
  Code,
  FileText,
  Send,
  SplitSquareVertical,
  GitCompare,
  Eye,
  Play,
  Undo2,
  Redo2,
  WrapText,
  Plus,
  Minus,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface CanvasPanelProps {
  content: string;
  language?: string;
  title?: string;
  isOpen: boolean;
  onClose: () => void;
  onContentChange?: (content: string) => void;
  onRequestEdit?: (instruction: string) => void;
  previousContent?: string; // For diff view
}

const CODE_LANGUAGES = new Set([
  "python",
  "javascript",
  "typescript",
  "java",
  "c",
  "cpp",
  "csharp",
  "go",
  "rust",
  "ruby",
  "php",
  "swift",
  "kotlin",
  "scala",
  "shell",
  "bash",
  "sql",
  "html",
  "css",
  "json",
  "yaml",
  "xml",
  "toml",
  "dockerfile",
  "makefile",
  "jsx",
  "tsx",
]);

const PREVIEW_LANGUAGES = new Set(["html", "jsx", "tsx", "css"]);

// Syntax highlighting colors per token type
const SYNTAX_COLORS: Record<string, string> = {
  keyword: "text-purple-500 dark:text-purple-400",
  string: "text-green-600 dark:text-green-400",
  number: "text-orange-500 dark:text-orange-400",
  comment: "text-gray-500 italic",
  function: "text-blue-500 dark:text-blue-400",
  class: "text-yellow-600 dark:text-yellow-400",
  operator: "text-pink-500 dark:text-pink-400",
  type: "text-cyan-500 dark:text-cyan-400",
  variable: "text-foreground",
  punctuation: "text-muted-foreground",
  property: "text-teal-500 dark:text-teal-400",
  tag: "text-red-500 dark:text-red-400",
  attribute: "text-orange-400 dark:text-orange-300",
  default: "text-foreground",
};

// Language-specific keyword patterns
const LANGUAGE_PATTERNS: Record<string, { keywords: string[]; builtins?: string[] }> = {
  python: {
    keywords: ["def", "class", "import", "from", "return", "if", "elif", "else", "for", "while", "try", "except", "finally", "with", "as", "yield", "raise", "pass", "break", "continue", "lambda", "and", "or", "not", "in", "is", "None", "True", "False", "async", "await", "global", "nonlocal"],
    builtins: ["print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple", "open", "type", "isinstance", "hasattr", "getattr", "setattr"],
  },
  javascript: {
    keywords: ["const", "let", "var", "function", "return", "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "try", "catch", "finally", "throw", "new", "class", "extends", "import", "export", "from", "default", "async", "await", "yield", "typeof", "instanceof", "in", "of", "this", "super", "null", "undefined", "true", "false"],
    builtins: ["console", "document", "window", "fetch", "Promise", "Array", "Object", "String", "Number", "Boolean", "JSON", "Math", "Date", "Error"],
  },
  typescript: {
    keywords: ["const", "let", "var", "function", "return", "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "try", "catch", "finally", "throw", "new", "class", "extends", "import", "export", "from", "default", "async", "await", "yield", "typeof", "instanceof", "in", "of", "this", "super", "null", "undefined", "true", "false", "interface", "type", "enum", "implements", "abstract", "public", "private", "protected", "readonly", "static", "as", "is", "keyof", "infer", "never", "unknown", "any"],
    builtins: ["console", "document", "window", "fetch", "Promise", "Array", "Object", "String", "Number", "Boolean", "JSON", "Math", "Date", "Error"],
  },
  html: {
    keywords: [],
  },
  css: {
    keywords: ["@import", "@media", "@keyframes", "@font-face", "@supports", "!important"],
  },
  json: {
    keywords: ["true", "false", "null"],
  },
  sql: {
    keywords: ["SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AND", "OR", "NOT", "IN", "IS", "NULL", "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "OFFSET", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "INDEX", "UNIQUE", "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "CASCADE", "AS", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "UNION", "ALL", "EXISTS", "BETWEEN", "LIKE", "ASC", "DESC"],
  },
};

function getFileExtension(language?: string, title?: string): string {
  if (title) {
    const dotIndex = title.lastIndexOf(".");
    if (dotIndex !== -1) return title.substring(dotIndex);
  }

  const extensionMap: Record<string, string> = {
    python: ".py",
    javascript: ".js",
    typescript: ".ts",
    java: ".java",
    c: ".c",
    cpp: ".cpp",
    csharp: ".cs",
    go: ".go",
    rust: ".rs",
    ruby: ".rb",
    php: ".php",
    swift: ".swift",
    kotlin: ".kt",
    scala: ".scala",
    shell: ".sh",
    bash: ".sh",
    sql: ".sql",
    html: ".html",
    css: ".css",
    json: ".json",
    yaml: ".yaml",
    xml: ".xml",
    toml: ".toml",
    markdown: ".md",
    jsx: ".jsx",
    tsx: ".tsx",
    dockerfile: "",
    makefile: "",
  };

  if (language && language in extensionMap) return extensionMap[language];
  return ".txt";
}

// Simple syntax highlighter
function highlightCode(code: string, language: string): React.ReactNode[] {
  const langLower = language.toLowerCase();
  const patterns = LANGUAGE_PATTERNS[langLower];
  const keywords = new Set(patterns?.keywords || []);
  const builtins = new Set(patterns?.builtins || []);

  const lines = code.split("\n");

  return lines.map((line, lineIndex) => {
    const tokens: React.ReactNode[] = [];
    let remaining = line;
    let tokenKey = 0;

    while (remaining.length > 0) {
      let matched = false;

      // HTML tags
      if (langLower === "html" || langLower === "jsx" || langLower === "tsx" || langLower === "xml") {
        const tagMatch = remaining.match(/^(<\/?[\w-]+)/);
        if (tagMatch) {
          tokens.push(<span key={tokenKey++} className={SYNTAX_COLORS.tag}>{tagMatch[0]}</span>);
          remaining = remaining.substring(tagMatch[0].length);
          matched = true;
          continue;
        }
        const attrMatch = remaining.match(/^(\s+[\w-]+)(=)/);
        if (attrMatch) {
          tokens.push(<span key={tokenKey++}>{attrMatch[1]}</span>);
          tokens.push(<span key={tokenKey++} className={SYNTAX_COLORS.operator}>{attrMatch[2]}</span>);
          remaining = remaining.substring(attrMatch[0].length);
          matched = true;
          continue;
        }
      }

      // Comments
      const commentMatch = remaining.match(/^(\/\/.*|#.*|\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->)/);
      if (commentMatch) {
        tokens.push(<span key={tokenKey++} className={SYNTAX_COLORS.comment}>{commentMatch[0]}</span>);
        remaining = remaining.substring(commentMatch[0].length);
        matched = true;
        continue;
      }

      // Strings
      const stringMatch = remaining.match(/^("[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*'|`[^`\\]*(?:\\.[^`\\]*)*`)/);
      if (stringMatch) {
        tokens.push(<span key={tokenKey++} className={SYNTAX_COLORS.string}>{stringMatch[0]}</span>);
        remaining = remaining.substring(stringMatch[0].length);
        matched = true;
        continue;
      }

      // Numbers
      const numberMatch = remaining.match(/^(\b\d+\.?\d*(?:e[+-]?\d+)?\b)/i);
      if (numberMatch) {
        tokens.push(<span key={tokenKey++} className={SYNTAX_COLORS.number}>{numberMatch[0]}</span>);
        remaining = remaining.substring(numberMatch[0].length);
        matched = true;
        continue;
      }

      // Keywords and identifiers
      const wordMatch = remaining.match(/^(\b[\w$]+\b)/);
      if (wordMatch) {
        const word = wordMatch[0];
        let colorClass = SYNTAX_COLORS.default;

        if (keywords.has(word) || keywords.has(word.toUpperCase())) {
          colorClass = SYNTAX_COLORS.keyword;
        } else if (builtins.has(word)) {
          colorClass = SYNTAX_COLORS.function;
        } else if (remaining.substring(word.length).trimStart().startsWith("(")) {
          colorClass = SYNTAX_COLORS.function;
        } else if (word[0] === word[0].toUpperCase() && word[0] !== word[0].toLowerCase()) {
          colorClass = SYNTAX_COLORS.class;
        }

        tokens.push(<span key={tokenKey++} className={colorClass}>{word}</span>);
        remaining = remaining.substring(word.length);
        matched = true;
        continue;
      }

      // Operators
      const operatorMatch = remaining.match(/^([+\-*/%=<>!&|^~?:]+|\.{3})/);
      if (operatorMatch) {
        tokens.push(<span key={tokenKey++} className={SYNTAX_COLORS.operator}>{operatorMatch[0]}</span>);
        remaining = remaining.substring(operatorMatch[0].length);
        matched = true;
        continue;
      }

      // Punctuation
      const punctMatch = remaining.match(/^([{}[\]();,.])/);
      if (punctMatch) {
        tokens.push(<span key={tokenKey++} className={SYNTAX_COLORS.punctuation}>{punctMatch[0]}</span>);
        remaining = remaining.substring(punctMatch[0].length);
        matched = true;
        continue;
      }

      // Default: take one character
      if (!matched) {
        tokens.push(<span key={tokenKey++}>{remaining[0]}</span>);
        remaining = remaining.substring(1);
      }
    }

    return (
      <div key={lineIndex} className="flex">
        <span className="inline-block w-10 text-right pr-4 text-muted-foreground select-none flex-shrink-0 text-xs leading-relaxed">
          {lineIndex + 1}
        </span>
        <span className="flex-1 break-all">{tokens.length > 0 ? tokens : "\n"}</span>
      </div>
    );
  });
}

// Diff computation
interface DiffLine {
  type: "added" | "removed" | "unchanged";
  content: string;
  oldLineNum?: number;
  newLineNum?: number;
}

function computeDiff(oldText: string, newText: string): DiffLine[] {
  const oldLines = oldText.split("\n");
  const newLines = newText.split("\n");
  const result: DiffLine[] = [];

  // Simple line-by-line diff (LCS-based would be better for production)
  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);

  let oldIdx = 0;
  let newIdx = 0;
  let oldLineNum = 1;
  let newLineNum = 1;

  while (oldIdx < oldLines.length || newIdx < newLines.length) {
    if (oldIdx >= oldLines.length) {
      // All remaining new lines are additions
      result.push({ type: "added", content: newLines[newIdx], newLineNum: newLineNum++ });
      newIdx++;
    } else if (newIdx >= newLines.length) {
      // All remaining old lines are removals
      result.push({ type: "removed", content: oldLines[oldIdx], oldLineNum: oldLineNum++ });
      oldIdx++;
    } else if (oldLines[oldIdx] === newLines[newIdx]) {
      // Lines match
      result.push({ type: "unchanged", content: oldLines[oldIdx], oldLineNum: oldLineNum++, newLineNum: newLineNum++ });
      oldIdx++;
      newIdx++;
    } else if (!newSet.has(oldLines[oldIdx])) {
      // Old line was removed
      result.push({ type: "removed", content: oldLines[oldIdx], oldLineNum: oldLineNum++ });
      oldIdx++;
    } else if (!oldSet.has(newLines[newIdx])) {
      // New line was added
      result.push({ type: "added", content: newLines[newIdx], newLineNum: newLineNum++ });
      newIdx++;
    } else {
      // Both lines exist elsewhere, treat as removal then addition
      result.push({ type: "removed", content: oldLines[oldIdx], oldLineNum: oldLineNum++ });
      oldIdx++;
    }
  }

  return result;
}

// Diff View Component
function DiffView({ oldContent, newContent }: { oldContent: string; newContent: string }) {
  const diff = useMemo(() => computeDiff(oldContent, newContent), [oldContent, newContent]);

  const stats = useMemo(() => {
    let added = 0, removed = 0;
    diff.forEach(line => {
      if (line.type === "added") added++;
      if (line.type === "removed") removed++;
    });
    return { added, removed };
  }, [diff]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4 text-sm px-2">
        <Badge variant="secondary" className="bg-green-500/10 text-green-600 border-green-500/20">
          <Plus className="h-3 w-3 mr-1" />
          {stats.added} added
        </Badge>
        <Badge variant="secondary" className="bg-red-500/10 text-red-600 border-red-500/20">
          <Minus className="h-3 w-3 mr-1" />
          {stats.removed} removed
        </Badge>
      </div>

      <div className="font-mono text-sm border rounded-lg overflow-hidden">
        {diff.map((line, index) => (
          <div
            key={index}
            className={cn(
              "flex",
              line.type === "added" && "bg-green-500/10 border-l-2 border-green-500",
              line.type === "removed" && "bg-red-500/10 border-l-2 border-red-500",
              line.type === "unchanged" && "border-l-2 border-transparent"
            )}
          >
            <span className="w-8 text-right pr-2 text-muted-foreground select-none text-xs py-0.5 flex-shrink-0">
              {line.oldLineNum || ""}
            </span>
            <span className="w-8 text-right pr-2 text-muted-foreground select-none text-xs py-0.5 flex-shrink-0 border-r">
              {line.newLineNum || ""}
            </span>
            <span className={cn(
              "w-5 text-center select-none text-xs py-0.5 flex-shrink-0",
              line.type === "added" && "text-green-600",
              line.type === "removed" && "text-red-600"
            )}>
              {line.type === "added" ? "+" : line.type === "removed" ? "-" : " "}
            </span>
            <span className="flex-1 px-2 py-0.5 break-all whitespace-pre-wrap">
              {line.content || " "}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Split View Component
function SplitView({
  leftContent,
  rightContent,
  leftTitle,
  rightTitle,
  language
}: {
  leftContent: string;
  rightContent: string;
  leftTitle: string;
  rightTitle: string;
  language?: string;
}) {
  const isCode = language ? CODE_LANGUAGES.has(language.toLowerCase()) : false;

  const renderContent = (content: string) => {
    if (isCode && language) {
      return (
        <code className="block">
          {highlightCode(content, language)}
        </code>
      );
    }
    return <span className="whitespace-pre-wrap">{content}</span>;
  };

  return (
    <div className="grid grid-cols-2 gap-2 h-full">
      <div className="border rounded-lg overflow-hidden flex flex-col">
        <div className="px-3 py-2 border-b bg-muted/50 flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">{leftTitle}</span>
          <Badge variant="outline" className="text-xs">Original</Badge>
        </div>
        <ScrollArea className="flex-1">
          <pre className={cn("p-3 text-sm", isCode && "font-mono")}>
            {renderContent(leftContent)}
          </pre>
        </ScrollArea>
      </div>

      <div className="border rounded-lg overflow-hidden flex flex-col">
        <div className="px-3 py-2 border-b bg-muted/50 flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">{rightTitle}</span>
          <Badge variant="outline" className="text-xs bg-primary/10 text-primary border-primary/20">Current</Badge>
        </div>
        <ScrollArea className="flex-1">
          <pre className={cn("p-3 text-sm", isCode && "font-mono")}>
            {renderContent(rightContent)}
          </pre>
        </ScrollArea>
      </div>
    </div>
  );
}

// Live Preview Component for HTML/CSS/JS
function LivePreview({ content, language }: { content: string; language: string }) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!iframeRef.current) return;

    try {
      let htmlContent = content;
      const langLower = language.toLowerCase();

      if (langLower === "css") {
        htmlContent = `
          <!DOCTYPE html>
          <html>
            <head><style>${content}</style></head>
            <body>
              <div class="preview-container">
                <h1>CSS Preview</h1>
                <p>This is a paragraph with <a href="#">a link</a>.</p>
                <button>Button</button>
                <div class="box">Box Element</div>
              </div>
            </body>
          </html>
        `;
      } else if (langLower === "jsx" || langLower === "tsx") {
        // For JSX/TSX, show a message that live preview isn't available
        htmlContent = `
          <!DOCTYPE html>
          <html>
            <head>
              <style>
                body { font-family: system-ui; padding: 20px; color: #666; }
                .info { background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 16px; }
              </style>
            </head>
            <body>
              <div class="info">
                <strong>JSX/TSX Preview</strong>
                <p>Live preview for React components requires a build step. The code is displayed in the editor.</p>
              </div>
            </body>
          </html>
        `;
      } else if (langLower !== "html") {
        htmlContent = `
          <!DOCTYPE html>
          <html>
            <head>
              <style>body { font-family: system-ui; padding: 20px; }</style>
            </head>
            <body>
              <pre>${content.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</pre>
            </body>
          </html>
        `;
      }

      const blob = new Blob([htmlContent], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      iframeRef.current.src = url;
      setError(null);

      return () => URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Preview error");
    }
  }, [content, language]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
        <span>Preview error: {error}</span>
      </div>
    );
  }

  return (
    <iframe
      ref={iframeRef}
      className="w-full h-full border-0 bg-white rounded-lg"
      sandbox="allow-scripts"
      title="Live Preview"
    />
  );
}

export function CanvasPanel({
  content,
  language,
  title,
  isOpen,
  onClose,
  onContentChange,
  onRequestEdit,
  previousContent,
}: CanvasPanelProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(content);
  const [copied, setCopied] = useState(false);
  const [aiInstruction, setAiInstruction] = useState("");
  const [viewMode, setViewMode] = useState<"code" | "diff" | "split" | "preview">("code");
  const [wordWrap, setWordWrap] = useState(true);
  const [history, setHistory] = useState<string[]>([content]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const aiInputRef = useRef<HTMLInputElement>(null);

  const isCode = language ? CODE_LANGUAGES.has(language.toLowerCase()) : false;
  const canPreview = language ? PREVIEW_LANGUAGES.has(language.toLowerCase()) : false;
  const hasPreviousContent = !!previousContent && previousContent !== content;
  const displayTitle = title || (isCode ? `code.${language || "txt"}` : "Document");

  // Sync edit content when content prop changes and not actively editing
  useEffect(() => {
    if (!isEditing) {
      setEditContent(content);
      setHistory(prev => {
        if (prev[prev.length - 1] !== content) {
          return [...prev, content];
        }
        return prev;
      });
      setHistoryIndex(history.length);
    }
  }, [content, isEditing]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const textArea = document.createElement("textarea");
      textArea.value = content;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [content]);

  const handleDownload = useCallback(() => {
    const ext = getFileExtension(language, title);
    const filename = title || `canvas-content${ext}`;
    const mimeType = isCode ? "text/plain" : "text/plain";
    const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [content, language, title, isCode]);

  const handleToggleEdit = useCallback(() => {
    if (isEditing) {
      if (editContent !== content && onContentChange) {
        onContentChange(editContent);
        setHistory(prev => [...prev.slice(0, historyIndex + 1), editContent]);
        setHistoryIndex(prev => prev + 1);
      }
      setIsEditing(false);
    } else {
      setEditContent(content);
      setIsEditing(true);
    }
  }, [isEditing, editContent, content, onContentChange, historyIndex]);

  const handleCancelEdit = useCallback(() => {
    setEditContent(content);
    setIsEditing(false);
  }, [content]);

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      const previousState = history[newIndex];
      setEditContent(previousState);
      if (onContentChange) {
        onContentChange(previousState);
      }
    }
  }, [historyIndex, history, onContentChange]);

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      const nextState = history[newIndex];
      setEditContent(nextState);
      if (onContentChange) {
        onContentChange(nextState);
      }
    }
  }, [historyIndex, history, onContentChange]);

  const handleAiEditSubmit = useCallback(() => {
    const instruction = aiInstruction.trim();
    if (!instruction || !onRequestEdit) return;
    onRequestEdit(instruction);
    setAiInstruction("");
  }, [aiInstruction, onRequestEdit]);

  const handleAiEditKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleAiEditSubmit();
      }
    },
    [handleAiEditSubmit]
  );

  const highlightedContent = useMemo(() => {
    if (isCode && language) {
      return highlightCode(content, language);
    }
    return null;
  }, [content, language, isCode]);

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        side="right"
        className="w-[600px] sm:w-[700px] sm:max-w-[60vw] flex flex-col p-0 gap-0"
      >
        {/* Header */}
        <SheetHeader className="px-6 py-4 border-b flex-shrink-0">
          <div className="flex items-center justify-between pr-8">
            <SheetTitle className="flex items-center gap-2 text-base">
              {isCode ? (
                <Code className="h-4 w-4 text-muted-foreground" />
              ) : (
                <FileText className="h-4 w-4 text-muted-foreground" />
              )}
              <span className="truncate">{displayTitle}</span>
              {language && (
                <span className="text-xs text-muted-foreground font-normal bg-muted px-2 py-0.5 rounded">
                  {language}
                </span>
              )}
            </SheetTitle>
          </div>
        </SheetHeader>

        {/* View Mode Tabs */}
        <div className="px-4 py-2 border-b flex items-center justify-between">
          <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as typeof viewMode)} className="w-auto">
            <TabsList className="h-8">
              <TabsTrigger value="code" className="text-xs gap-1 px-2 h-7">
                <Code className="h-3 w-3" />
                Code
              </TabsTrigger>
              {hasPreviousContent && (
                <>
                  <TabsTrigger value="diff" className="text-xs gap-1 px-2 h-7">
                    <GitCompare className="h-3 w-3" />
                    Diff
                  </TabsTrigger>
                  <TabsTrigger value="split" className="text-xs gap-1 px-2 h-7">
                    <SplitSquareVertical className="h-3 w-3" />
                    Split
                  </TabsTrigger>
                </>
              )}
              {canPreview && (
                <TabsTrigger value="preview" className="text-xs gap-1 px-2 h-7">
                  <Eye className="h-3 w-3" />
                  Preview
                </TabsTrigger>
              )}
            </TabsList>
          </Tabs>

          <div className="flex items-center gap-1">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={wordWrap ? "secondary" : "ghost"}
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => setWordWrap(!wordWrap)}
                  >
                    <WrapText className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Word Wrap</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={handleUndo}
                    disabled={historyIndex <= 0}
                  >
                    <Undo2 className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Undo</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={handleRedo}
                    disabled={historyIndex >= history.length - 1}
                  >
                    <Redo2 className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Redo</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 min-h-0">
          {viewMode === "code" && (
            <ScrollArea className="h-full">
              <div className="p-4">
                {isEditing ? (
                  <Textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className={cn(
                      "min-h-[400px] resize-none border-0 focus-visible:ring-0 p-0 shadow-none",
                      isCode && "font-mono text-sm"
                    )}
                    placeholder="Enter content..."
                    autoFocus
                  />
                ) : (
                  <pre
                    className={cn(
                      "text-sm leading-relaxed",
                      isCode && "font-mono",
                      wordWrap ? "whitespace-pre-wrap" : "whitespace-pre overflow-x-auto"
                    )}
                  >
                    {highlightedContent ? (
                      <code className="block">{highlightedContent}</code>
                    ) : (
                      <span>{content}</span>
                    )}
                  </pre>
                )}
              </div>
            </ScrollArea>
          )}

          {viewMode === "diff" && hasPreviousContent && (
            <ScrollArea className="h-full">
              <div className="p-4">
                <DiffView oldContent={previousContent!} newContent={content} />
              </div>
            </ScrollArea>
          )}

          {viewMode === "split" && hasPreviousContent && (
            <div className="h-full p-4">
              <SplitView
                leftContent={previousContent!}
                rightContent={content}
                leftTitle="Previous Version"
                rightTitle="Current Version"
                language={language}
              />
            </div>
          )}

          {viewMode === "preview" && canPreview && (
            <div className="h-full p-4">
              <LivePreview content={content} language={language || "html"} />
            </div>
          )}
        </div>

        {/* Action Bar */}
        <div className="flex-shrink-0 border-t px-4 py-3 space-y-3">
          <div className="flex items-center gap-2">
            <Button
              variant={isEditing ? "default" : "outline"}
              size="sm"
              onClick={handleToggleEdit}
              className="gap-1.5"
            >
              {isEditing ? (
                <>
                  <Check className="h-3.5 w-3.5" />
                  Save
                </>
              ) : (
                <>
                  <Pencil className="h-3.5 w-3.5" />
                  Edit
                </>
              )}
            </Button>

            {isEditing && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCancelEdit}
                className="gap-1.5 text-muted-foreground"
              >
                <X className="h-3.5 w-3.5" />
                Cancel
              </Button>
            )}

            <div className="flex-1" />

            <Button
              variant="outline"
              size="sm"
              onClick={handleCopy}
              className="gap-1.5"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-green-500" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  Copy
                </>
              )}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              className="gap-1.5"
            >
              <Download className="h-3.5 w-3.5" />
              Download
            </Button>
          </div>

          {/* AI Edit Input */}
          {onRequestEdit && (
            <div className="flex items-center gap-2">
              <Input
                ref={aiInputRef}
                value={aiInstruction}
                onChange={(e) => setAiInstruction(e.target.value)}
                onKeyDown={handleAiEditKeyDown}
                placeholder="Ask AI to edit: &quot;make it shorter&quot;, &quot;add error handling&quot;..."
                className="flex-1 text-sm"
              />
              <Button
                size="sm"
                onClick={handleAiEditSubmit}
                disabled={!aiInstruction.trim()}
                className="gap-1.5 flex-shrink-0"
              >
                <Send className="h-3.5 w-3.5" />
                Send
              </Button>
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
