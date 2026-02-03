"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Command, CommandEmpty, CommandGroup, CommandItem, CommandList } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverAnchor } from "@/components/ui/popover";
import {
  Sparkles,
  Brain,
  CheckCircle,
  MessageSquare,
  Calculator,
  Calendar,
  HelpCircle,
  Bot,
  Zap,
  Settings,
  FileSearch,
  Network,
  Layers
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SlashCommand {
  name: string;
  aliases: string[];
  description: string;
  usage: string;
  examples: string[];
  category: string;
  requires_args: boolean;
}

interface SlashCommandAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  onCommandSelect?: (command: SlashCommand) => void;
  inputRef: React.RefObject<HTMLInputElement | HTMLTextAreaElement>;
  className?: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchCommands(): Promise<SlashCommand[]> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/chat/commands/list`,
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("authToken") || ""}`,
        },
      }
    );

    if (!response.ok) return [];

    const data = await response.json();
    return data.commands || [];
  } catch (error) {
    console.error("Failed to fetch slash commands:", error);
    return [];
  }
}

function getIconForCategory(category: string) {
  switch (category.toLowerCase()) {
    case "reasoning":
      return <Brain className="h-4 w-4" />;
    case "verification":
      return <CheckCircle className="h-4 w-4" />;
    case "conversation":
      return <MessageSquare className="h-4 w-4" />;
    case "utility":
      return <Calculator className="h-4 w-4" />;
    case "date":
      return <Calendar className="h-4 w-4" />;
    case "help":
      return <HelpCircle className="h-4 w-4" />;
    case "agent":
      return <Bot className="h-4 w-4" />;
    case "advanced":
      return <Zap className="h-4 w-4" />;
    case "search":
      return <FileSearch className="h-4 w-4" />;
    case "graph":
      return <Network className="h-4 w-4" />;
    case "workflow":
      return <Layers className="h-4 w-4" />;
    case "settings":
      return <Settings className="h-4 w-4" />;
    default:
      return <Sparkles className="h-4 w-4" />;
  }
}

function getCategoryLabel(category: string): string {
  return category.charAt(0).toUpperCase() + category.slice(1);
}

export function SlashCommandAutocomplete({
  value,
  onChange,
  onCommandSelect,
  inputRef,
  className,
}: SlashCommandAutocompleteProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [partialCommand, setPartialCommand] = React.useState("");
  const [commandStart, setCommandStart] = React.useState(-1);
  const [selectedIndex, setSelectedIndex] = React.useState(0);

  // Fetch all commands once
  const { data: allCommands = [], isLoading } = useQuery({
    queryKey: ["slashCommands"],
    queryFn: fetchCommands,
    staleTime: 60000, // Cache for 1 minute
  });

  // Detect /command being typed at start of input
  React.useEffect(() => {
    // Only show autocomplete if / is at the start of the message
    if (value.startsWith("/")) {
      // Extract the partial command (everything from / to first space or end)
      const spaceIndex = value.indexOf(" ");
      const partial = spaceIndex >= 0 ? value.slice(0, spaceIndex) : value;

      setPartialCommand(partial.slice(1).toLowerCase()); // Remove leading /
      setCommandStart(0);
      setIsOpen(true);
      setSelectedIndex(0);
    } else {
      setIsOpen(false);
    }
  }, [value]);

  // Filter commands based on partial input
  const filteredCommands = React.useMemo(() => {
    if (!partialCommand) return allCommands;

    return allCommands.filter((cmd) => {
      const searchTerm = partialCommand.toLowerCase();
      // Match command name or any alias
      return (
        cmd.name.toLowerCase().includes(searchTerm) ||
        cmd.aliases.some((alias) => alias.toLowerCase().includes(searchTerm))
      );
    });
  }, [allCommands, partialCommand]);

  // Group commands by category
  const groupedCommands = React.useMemo(() => {
    const groups: Record<string, SlashCommand[]> = {};

    filteredCommands.forEach((cmd) => {
      if (!groups[cmd.category]) {
        groups[cmd.category] = [];
      }
      groups[cmd.category].push(cmd);
    });

    return groups;
  }, [filteredCommands]);

  // Handle keyboard navigation
  React.useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < filteredCommands.length - 1 ? prev + 1 : 0
        );
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : filteredCommands.length - 1
        );
      } else if (e.key === "Enter" && filteredCommands.length > 0) {
        e.preventDefault();
        handleSelect(filteredCommands[selectedIndex]);
      } else if (e.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, selectedIndex, filteredCommands]);

  // Handle selection
  const handleSelect = (command: SlashCommand) => {
    if (commandStart >= 0) {
      // Replace the partial command with the selected one
      const spaceIndex = value.indexOf(" ");
      const afterCommand = spaceIndex >= 0 ? value.slice(spaceIndex) : "";

      // Add a space after the command if it requires args and doesn't have one
      const newValue = `/${command.name}${command.requires_args && !afterCommand.startsWith(" ") ? " " : ""}${afterCommand}`;
      onChange(newValue);

      onCommandSelect?.(command);
    }

    setIsOpen(false);

    // Focus back on input
    setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  if (!isOpen || filteredCommands.length === 0) {
    return null;
  }

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverAnchor asChild>
        <div className="absolute bottom-full left-0 w-full h-0" />
      </PopoverAnchor>
      <PopoverContent
        className={cn("w-96 p-0 max-h-80 overflow-auto", className)}
        align="start"
        side="top"
        sideOffset={8}
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <Command className="rounded-lg border shadow-md">
          <CommandList>
            {isLoading && (
              <CommandEmpty>Loading commands...</CommandEmpty>
            )}
            {!isLoading && filteredCommands.length === 0 && (
              <CommandEmpty>No commands found.</CommandEmpty>
            )}
            {Object.entries(groupedCommands).map(([category, commands]) => (
              <CommandGroup key={category} heading={getCategoryLabel(category)}>
                {commands.map((command, index) => {
                  // Calculate global index for selection highlighting
                  let globalIndex = 0;
                  for (const [cat, cmds] of Object.entries(groupedCommands)) {
                    if (cat === category) {
                      globalIndex += index;
                      break;
                    }
                    globalIndex += cmds.length;
                  }

                  return (
                    <CommandItem
                      key={command.name}
                      onSelect={() => handleSelect(command)}
                      className={cn(
                        "flex items-start gap-2 py-2 cursor-pointer",
                        globalIndex === selectedIndex && "bg-accent"
                      )}
                    >
                      <div className="flex-shrink-0 mt-0.5 text-muted-foreground">
                        {getIconForCategory(command.category)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-primary">/{command.name}</span>
                          {command.aliases.length > 0 && (
                            <span className="text-xs text-muted-foreground">
                              ({command.aliases.map(a => `/${a}`).join(", ")})
                            </span>
                          )}
                        </div>
                        <div className="text-sm text-muted-foreground truncate">
                          {command.description}
                        </div>
                        {command.usage && (
                          <div className="text-xs text-muted-foreground/70 font-mono mt-0.5">
                            {command.usage}
                          </div>
                        )}
                      </div>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

// Export for integration with chat interface
export type { SlashCommand, SlashCommandAutocompleteProps };
