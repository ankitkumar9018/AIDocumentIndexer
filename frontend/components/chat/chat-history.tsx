"use client";

import * as React from "react";
import { Plus, MessageSquare, Trash2, MoreHorizontal } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatRelativeTime } from "@/lib/utils";

export interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  updatedAt: Date;
  messageCount: number;
}

interface ChatHistoryProps {
  sessions: ChatSession[];
  activeSessionId?: string;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
  className?: string;
}

export function ChatHistory({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  className,
}: ChatHistoryProps) {
  return (
    <div className={cn("flex flex-col h-full border-r", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="font-semibold">Chat History</h3>
        <Button
          variant="ghost"
          size="icon"
          onClick={onNewSession}
          title="New chat"
        >
          <Plus className="h-5 w-5" />
        </Button>
      </div>

      {/* Sessions List */}
      <ScrollArea className="flex-1">
        {sessions.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground">
            <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No chat history yet</p>
            <p className="text-xs mt-1">Start a new conversation</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {sessions.map((session) => (
              <SessionItem
                key={session.id}
                session={session}
                isActive={session.id === activeSessionId}
                onSelect={() => onSelectSession(session.id)}
                onDelete={() => onDeleteSession(session.id)}
              />
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}

interface SessionItemProps {
  session: ChatSession;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

function SessionItem({ session, isActive, onSelect, onDelete }: SessionItemProps) {
  const [showActions, setShowActions] = React.useState(false);

  return (
    <div
      className={cn(
        "group relative rounded-lg p-3 cursor-pointer transition-colors",
        isActive
          ? "bg-primary/10 text-primary"
          : "hover:bg-muted"
      )}
      onClick={onSelect}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className="flex items-start gap-3">
        <MessageSquare className="h-5 w-5 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate">{session.title}</p>
          <p className="text-xs text-muted-foreground truncate mt-0.5">
            {session.lastMessage}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {formatRelativeTime(session.updatedAt)} · {session.messageCount} messages
          </p>
        </div>
      </div>

      {/* Actions */}
      {showActions && (
        <div className="absolute right-2 top-2 flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      )}
    </div>
  );
}
