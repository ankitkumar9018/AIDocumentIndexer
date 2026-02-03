"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Command, CommandEmpty, CommandGroup, CommandItem, CommandList } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverAnchor } from "@/components/ui/popover";
import { Folder, FileText, Tag, Clock, Globe } from "lucide-react";
import { cn } from "@/lib/utils";

interface MentionSuggestion {
  type: string;
  value: string;
  display: string;
  description?: string;
}

interface MentionAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  onMentionSelect?: (mention: MentionSuggestion) => void;
  inputRef: React.RefObject<HTMLInputElement | HTMLTextAreaElement>;
  className?: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchMentionSuggestions(partial: string): Promise<MentionSuggestion[]> {
  if (!partial || partial.length < 1) return [];

  try {
    const response = await fetch(
      `${API_BASE_URL}/chat/mentions/autocomplete?partial=${encodeURIComponent(partial)}`,
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("authToken") || ""}`,
        },
      }
    );

    if (!response.ok) return [];

    const data = await response.json();
    return data.suggestions || [];
  } catch (error) {
    console.error("Failed to fetch mention suggestions:", error);
    return [];
  }
}

function getIconForType(type: string) {
  switch (type) {
    case "folder":
      return <Folder className="h-4 w-4" />;
    case "document":
      return <FileText className="h-4 w-4" />;
    case "tag":
      return <Tag className="h-4 w-4" />;
    case "recent":
      return <Clock className="h-4 w-4" />;
    case "type":
      return <Globe className="h-4 w-4" />;
    default:
      return <FileText className="h-4 w-4" />;
  }
}

export function MentionAutocomplete({
  value,
  onChange,
  onMentionSelect,
  inputRef,
  className,
}: MentionAutocompleteProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [partialMention, setPartialMention] = React.useState("");
  const [mentionStart, setMentionStart] = React.useState(-1);

  // Detect @mention being typed
  React.useEffect(() => {
    // Find the last @ in the value
    const lastAtIndex = value.lastIndexOf("@");

    if (lastAtIndex >= 0) {
      // Check if there's a space after the @ (meaning the mention is complete)
      const textAfterAt = value.slice(lastAtIndex);
      const hasSpaceAfter = textAfterAt.includes(" ") && textAfterAt.indexOf(" ") > textAfterAt.indexOf(":");

      if (!hasSpaceAfter) {
        // Extract the partial mention
        const partial = textAfterAt.trim();
        if (partial.length > 0) {
          setPartialMention(partial);
          setMentionStart(lastAtIndex);
          setIsOpen(true);
        }
      } else {
        setIsOpen(false);
      }
    } else {
      setIsOpen(false);
    }
  }, [value]);

  // Fetch suggestions
  const { data: suggestions = [], isLoading } = useQuery({
    queryKey: ["mentionSuggestions", partialMention],
    queryFn: () => fetchMentionSuggestions(partialMention),
    enabled: isOpen && partialMention.length >= 1,
    staleTime: 1000,
  });

  // Handle selection
  const handleSelect = (suggestion: MentionSuggestion) => {
    if (mentionStart >= 0) {
      // Replace the partial mention with the selected value
      const beforeMention = value.slice(0, mentionStart);
      const afterMention = value.slice(mentionStart + partialMention.length);

      // Add a space after the mention if there isn't one
      const newValue = `${beforeMention}${suggestion.value}${afterMention.startsWith(" ") ? "" : " "}${afterMention}`;
      onChange(newValue);

      onMentionSelect?.(suggestion);
    }

    setIsOpen(false);

    // Focus back on input
    setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  if (!isOpen || suggestions.length === 0) {
    return null;
  }

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverAnchor asChild>
        <div className="absolute bottom-full left-0 w-full h-0" />
      </PopoverAnchor>
      <PopoverContent
        className={cn("w-80 p-0", className)}
        align="start"
        side="top"
        sideOffset={8}
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <Command className="rounded-lg border shadow-md">
          <CommandList>
            {isLoading && (
              <CommandEmpty>Loading suggestions...</CommandEmpty>
            )}
            {!isLoading && suggestions.length === 0 && (
              <CommandEmpty>No suggestions found.</CommandEmpty>
            )}
            <CommandGroup heading="Mentions">
              {suggestions.map((suggestion, index) => (
                <CommandItem
                  key={`${suggestion.type}-${suggestion.value}-${index}`}
                  onSelect={() => handleSelect(suggestion)}
                  className="flex items-start gap-2 py-2"
                >
                  <div className="flex-shrink-0 mt-0.5 text-muted-foreground">
                    {getIconForType(suggestion.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{suggestion.display}</div>
                    {suggestion.description && (
                      <div className="text-xs text-muted-foreground truncate">
                        {suggestion.description}
                      </div>
                    )}
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

// Helper component that combines input with autocomplete
interface MentionInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: () => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

export function MentionInput({
  value,
  onChange,
  onSubmit,
  placeholder = "Type a message... Use @folder: @doc: @tag: to filter",
  className,
  disabled,
}: MentionInputProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && onSubmit) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
      />
      <MentionAutocomplete
        value={value}
        onChange={onChange}
        inputRef={inputRef}
      />
    </div>
  );
}

// Parse mentions from a message for display
export function parseMentions(text: string): { cleanText: string; mentions: string[] } {
  const mentionPattern = /@(?:folder|document|doc|tag|recent|all):[^\s]+|@all/gi;
  const mentions: string[] = [];

  const cleanText = text.replace(mentionPattern, (match) => {
    mentions.push(match);
    return "";
  }).trim().replace(/\s+/g, " ");

  return { cleanText, mentions };
}
