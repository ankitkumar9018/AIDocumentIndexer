"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  FileText,
  MessageSquare,
  Upload,
  Settings,
  Users,
  Bot,
  BarChart3,
  FolderOpen,
  Moon,
  Sun,
  Keyboard,
  LogOut,
  HelpCircle,
  RefreshCw,
  Download,
  Trash2,
  Plus,
  Loader2,
  ChevronRight,
} from "lucide-react";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Command {
  id: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
  category: "navigation" | "actions" | "settings" | "help";
  keywords?: string[];
  action: () => void | Promise<void>;
  shortcut?: string[];
}

interface SearchResult {
  document_id: string;
  document_name: string;
  chunk_id: string;
  content: string;
  score: number;
  page_number?: number;
  collection?: string;
}

interface CommandPaletteProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const [isOpen, setIsOpen] = React.useState(open || false);
  const [query, setQuery] = React.useState("");
  const [searchResults, setSearchResults] = React.useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = React.useState(false);
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const [mode, setMode] = React.useState<"commands" | "search">("commands");
  const router = useRouter();
  const inputRef = React.useRef<HTMLInputElement>(null);
  const { theme, setTheme } = useTheme();

  // Define available commands
  const commands: Command[] = React.useMemo(() => [
    // Navigation
    {
      id: "nav-dashboard",
      label: "Go to Dashboard",
      description: "View your dashboard overview",
      icon: <BarChart3 className="h-4 w-4" />,
      category: "navigation",
      keywords: ["home", "overview", "stats"],
      action: () => router.push("/dashboard"),
      shortcut: ["⌘", "1"],
    },
    {
      id: "nav-chat",
      label: "Go to Chat",
      description: "Start a conversation with your documents",
      icon: <MessageSquare className="h-4 w-4" />,
      category: "navigation",
      keywords: ["ai", "ask", "question", "conversation"],
      action: () => router.push("/dashboard/chat"),
      shortcut: ["⌘", "2"],
    },
    {
      id: "nav-documents",
      label: "Go to Documents",
      description: "Browse and manage your documents",
      icon: <FolderOpen className="h-4 w-4" />,
      category: "navigation",
      keywords: ["files", "library", "browse"],
      action: () => router.push("/dashboard/documents"),
      shortcut: ["⌘", "3"],
    },
    {
      id: "nav-upload",
      label: "Go to Upload",
      description: "Upload new documents",
      icon: <Upload className="h-4 w-4" />,
      category: "navigation",
      keywords: ["add", "import", "new"],
      action: () => router.push("/dashboard/upload"),
      shortcut: ["⌘", "4"],
    },
    {
      id: "nav-costs",
      label: "Go to Costs",
      description: "View usage and cost analytics",
      icon: <BarChart3 className="h-4 w-4" />,
      category: "navigation",
      keywords: ["billing", "usage", "analytics", "spending"],
      action: () => router.push("/dashboard/costs"),
    },
    {
      id: "nav-profile",
      label: "Go to Profile",
      description: "Manage your account settings",
      icon: <Users className="h-4 w-4" />,
      category: "navigation",
      keywords: ["account", "user", "preferences"],
      action: () => router.push("/dashboard/profile"),
    },
    {
      id: "nav-admin",
      label: "Go to Admin Settings",
      description: "System administration",
      icon: <Settings className="h-4 w-4" />,
      category: "navigation",
      keywords: ["system", "config", "administration"],
      action: () => router.push("/dashboard/admin/settings"),
    },
    // Actions
    {
      id: "action-new-chat",
      label: "New Chat Session",
      description: "Start a fresh conversation",
      icon: <Plus className="h-4 w-4" />,
      category: "actions",
      keywords: ["create", "start", "fresh"],
      action: () => {
        router.push("/dashboard/chat");
        // Could trigger new chat action here
      },
      shortcut: ["⌘", "N"],
    },
    {
      id: "action-search-docs",
      label: "Search Documents",
      description: "Find documents by content",
      icon: <Search className="h-4 w-4" />,
      category: "actions",
      keywords: ["find", "query", "lookup"],
      action: () => {
        setMode("search");
        setQuery("");
      },
    },
    {
      id: "action-upload",
      label: "Upload Document",
      description: "Add a new document to your library",
      icon: <Upload className="h-4 w-4" />,
      category: "actions",
      keywords: ["add", "import", "new", "file"],
      action: () => router.push("/dashboard/upload"),
    },
    // Settings
    {
      id: "settings-theme-toggle",
      label: theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode",
      description: "Toggle between light and dark themes",
      icon: theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />,
      category: "settings",
      keywords: ["dark", "light", "appearance", "color"],
      action: () => setTheme(theme === "dark" ? "light" : "dark"),
    },
    {
      id: "settings-shortcuts",
      label: "Keyboard Shortcuts",
      description: "View all available shortcuts",
      icon: <Keyboard className="h-4 w-4" />,
      category: "settings",
      keywords: ["keys", "hotkeys", "bindings"],
      action: () => {
        handleClose();
        // Trigger keyboard shortcuts dialog
        window.dispatchEvent(new KeyboardEvent("keydown", { key: "?" }));
      },
      shortcut: ["?"],
    },
    // Help
    {
      id: "help-docs",
      label: "Documentation",
      description: "Read the user guide",
      icon: <HelpCircle className="h-4 w-4" />,
      category: "help",
      keywords: ["guide", "manual", "instructions"],
      action: () => { window.open("/docs", "_blank"); },
    },
  ], [router, theme, setTheme]);

  // Filter commands based on query
  const filteredCommands = React.useMemo(() => {
    if (!query.trim()) return commands;

    const searchTerms = query.toLowerCase().split(" ");
    return commands.filter((command) => {
      const searchText = [
        command.label,
        command.description,
        ...(command.keywords || []),
      ].join(" ").toLowerCase();

      return searchTerms.every((term) => searchText.includes(term));
    });
  }, [commands, query]);

  // Group filtered commands by category
  const groupedCommands = React.useMemo(() => {
    const groups: Record<string, Command[]> = {
      navigation: [],
      actions: [],
      settings: [],
      help: [],
    };

    filteredCommands.forEach((command) => {
      groups[command.category].push(command);
    });

    return groups;
  }, [filteredCommands]);

  // Calculate total items for keyboard navigation
  const totalItems = React.useMemo(() => {
    if (mode === "search") {
      return searchResults.length;
    }
    return filteredCommands.length;
  }, [mode, searchResults, filteredCommands]);

  // Keyboard shortcut to open palette (Cmd+K or Ctrl+K)
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsOpen(true);
        setMode("commands");
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Focus input when dialog opens
  React.useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Sync with external open state
  React.useEffect(() => {
    if (open !== undefined) {
      setIsOpen(open);
    }
  }, [open]);

  // Search documents when in search mode
  React.useEffect(() => {
    if (mode !== "search" || !query.trim()) {
      setSearchResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const token = localStorage.getItem("auth_token");
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/documents/search?query=${encodeURIComponent(query)}&limit=10`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.ok) {
          const data = await response.json();
          setSearchResults(data.results || []);
          setSelectedIndex(0);
        }
      } catch (error) {
        console.error("Search failed:", error);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [mode, query]);

  // Reset selected index when query changes
  React.useEffect(() => {
    setSelectedIndex(0);
  }, [query, mode]);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, totalItems - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (mode === "search" && searchResults[selectedIndex]) {
        router.push(`/dashboard/documents/${searchResults[selectedIndex].document_id}`);
        handleClose();
      } else if (mode === "commands" && filteredCommands[selectedIndex]) {
        executeCommand(filteredCommands[selectedIndex]);
      }
    } else if (e.key === "Escape") {
      if (mode === "search") {
        setMode("commands");
        setQuery("");
      } else {
        handleClose();
      }
    } else if (e.key === "Backspace" && !query && mode === "search") {
      setMode("commands");
    }
  };

  const executeCommand = async (command: Command) => {
    handleClose();
    await command.action();
  };

  const handleClose = () => {
    setIsOpen(false);
    setQuery("");
    setSearchResults([]);
    setMode("commands");
    onOpenChange?.(false);
  };

  const handleOpenChange = (open: boolean) => {
    setIsOpen(open);
    if (!open) {
      setQuery("");
      setSearchResults([]);
      setMode("commands");
    }
    onOpenChange?.(open);
  };

  const categoryLabels: Record<string, string> = {
    navigation: "Navigation",
    actions: "Actions",
    settings: "Settings",
    help: "Help",
  };

  return (
    <>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={cn(
          "flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground",
          "rounded-md border border-input bg-background",
          "hover:bg-accent hover:text-accent-foreground",
          "transition-colors"
        )}
      >
        <Search className="h-4 w-4" />
        <span className="hidden sm:inline">Search...</span>
        <kbd className="hidden sm:inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
          <span className="text-xs">⌘</span>K
        </kbd>
      </button>

      {/* Command Palette Dialog */}
      <Dialog open={isOpen} onOpenChange={handleOpenChange}>
        <DialogContent className="max-w-2xl p-0 gap-0 overflow-hidden">
          <DialogHeader className="p-4 pb-0">
            <DialogTitle className="sr-only">Command Palette</DialogTitle>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={mode === "search" ? "Search documents..." : "Type a command or search..."}
                className="pl-10"
              />
              {mode === "search" && (
                <Badge
                  variant="secondary"
                  className="absolute right-3 top-1/2 -translate-y-1/2"
                >
                  Documents
                </Badge>
              )}
            </div>
          </DialogHeader>

          <div className="border-t mt-4">
            {mode === "search" ? (
              // Search Results View
              isSearching ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : searchResults.length > 0 ? (
                <ScrollArea className="max-h-[400px]">
                  <div className="p-2">
                    {searchResults.map((result, index) => (
                      <button
                        key={`${result.document_id}-${result.chunk_id}`}
                        onClick={() => {
                          router.push(`/dashboard/documents/${result.document_id}`);
                          handleClose();
                        }}
                        className={cn(
                          "w-full text-left p-3 rounded-md flex items-start gap-3",
                          "hover:bg-accent transition-colors",
                          index === selectedIndex && "bg-accent"
                        )}
                      >
                        <FileText className="h-5 w-5 text-muted-foreground mt-0.5 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium truncate">
                              {result.document_name}
                            </span>
                            {result.collection && (
                              <Badge variant="secondary" className="shrink-0">
                                {result.collection}
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
                            {result.content}
                          </p>
                        </div>
                        <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0 self-center" />
                      </button>
                    ))}
                  </div>
                </ScrollArea>
              ) : query ? (
                <div className="py-8 text-center text-muted-foreground">
                  No documents found for &quot;{query}&quot;
                </div>
              ) : (
                <div className="py-8 text-center text-muted-foreground">
                  <p>Start typing to search your documents</p>
                </div>
              )
            ) : (
              // Commands View
              <ScrollArea className="max-h-[400px]">
                <div className="p-2">
                  {Object.entries(groupedCommands).map(([category, cmds]) => {
                    if (cmds.length === 0) return null;

                    // Calculate the starting index for this category
                    let startIndex = 0;
                    for (const [cat, c] of Object.entries(groupedCommands)) {
                      if (cat === category) break;
                      startIndex += c.length;
                    }

                    return (
                      <div key={category} className="mb-4">
                        <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                          {categoryLabels[category]}
                        </div>
                        {cmds.map((command, idx) => {
                          const globalIndex = startIndex + idx;
                          return (
                            <button
                              key={command.id}
                              onClick={() => executeCommand(command)}
                              className={cn(
                                "w-full text-left p-2 rounded-md flex items-center gap-3",
                                "hover:bg-accent transition-colors",
                                globalIndex === selectedIndex && "bg-accent"
                              )}
                            >
                              <div className="flex items-center justify-center w-8 h-8 rounded-md bg-muted">
                                {command.icon}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="font-medium">{command.label}</div>
                                {command.description && (
                                  <div className="text-sm text-muted-foreground truncate">
                                    {command.description}
                                  </div>
                                )}
                              </div>
                              {command.shortcut && (
                                <div className="flex items-center gap-1">
                                  {command.shortcut.map((key, keyIdx) => (
                                    <kbd
                                      key={keyIdx}
                                      className="h-5 min-w-5 px-1.5 inline-flex items-center justify-center rounded border bg-muted font-mono text-[10px] font-medium text-muted-foreground"
                                    >
                                      {key}
                                    </kbd>
                                  ))}
                                </div>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    );
                  })}
                  {filteredCommands.length === 0 && (
                    <div className="py-8 text-center text-muted-foreground">
                      No commands found for &quot;{query}&quot;
                    </div>
                  )}
                </div>
              </ScrollArea>
            )}
          </div>

          <div className="border-t p-2 flex items-center justify-between text-xs text-muted-foreground bg-muted/50">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <kbd className="h-5 px-1.5 inline-flex items-center justify-center rounded border bg-background font-mono text-[10px]">↑</kbd>
                <kbd className="h-5 px-1.5 inline-flex items-center justify-center rounded border bg-background font-mono text-[10px]">↓</kbd>
                <span className="ml-1">Navigate</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd className="h-5 px-1.5 inline-flex items-center justify-center rounded border bg-background font-mono text-[10px]">↵</kbd>
                <span className="ml-1">Select</span>
              </div>
              <div className="flex items-center gap-1">
                <kbd className="h-5 px-1.5 inline-flex items-center justify-center rounded border bg-background font-mono text-[10px]">esc</kbd>
                <span className="ml-1">{mode === "search" ? "Back" : "Close"}</span>
              </div>
            </div>
            <div>
              {mode === "commands" && (
                <span>Type to filter commands or search documents</span>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
