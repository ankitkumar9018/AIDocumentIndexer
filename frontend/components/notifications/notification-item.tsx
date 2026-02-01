"use client";

import { formatDistanceToNow } from "date-fns";
import {
  FileText,
  Upload,
  MessageSquare,
  AlertCircle,
  CheckCircle,
  Info,
  AlertTriangle,
  Download,
  Database,
  Trash2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export type NotificationType =
  | "document_indexed"
  | "document_deleted"
  | "upload_complete"
  | "upload_failed"
  | "chat_mention"
  | "query_complete"
  | "sync_complete"
  | "sync_failed"
  | "export_ready"
  | "system_info"
  | "system_warning"
  | "system_error";

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  link?: string;
  metadata?: Record<string, unknown>;
}

interface NotificationItemProps {
  notification: Notification;
  onMarkRead: (id: string) => void;
  onDismiss: (id: string) => void;
  onClick?: (notification: Notification) => void;
}

const typeConfig: Record<
  NotificationType,
  { icon: React.ComponentType<{ className?: string }>; color: string }
> = {
  document_indexed: { icon: FileText, color: "text-blue-500" },
  document_deleted: { icon: Trash2, color: "text-red-500" },
  upload_complete: { icon: Upload, color: "text-green-500" },
  upload_failed: { icon: AlertCircle, color: "text-red-500" },
  chat_mention: { icon: MessageSquare, color: "text-purple-500" },
  query_complete: { icon: Database, color: "text-blue-500" },
  sync_complete: { icon: CheckCircle, color: "text-green-500" },
  sync_failed: { icon: AlertCircle, color: "text-red-500" },
  export_ready: { icon: Download, color: "text-green-500" },
  system_info: { icon: Info, color: "text-blue-500" },
  system_warning: { icon: AlertTriangle, color: "text-yellow-500" },
  system_error: { icon: AlertCircle, color: "text-red-500" },
};

export function NotificationItem({
  notification,
  onMarkRead,
  onDismiss,
  onClick,
}: NotificationItemProps) {
  const config = typeConfig[notification.type] || typeConfig.system_info;
  const Icon = config.icon;

  const handleClick = () => {
    if (!notification.read) {
      onMarkRead(notification.id);
    }
    onClick?.(notification);
  };

  return (
    <div
      className={cn(
        "relative flex items-start gap-3 p-3 rounded-lg transition-colors cursor-pointer group",
        notification.read
          ? "bg-background hover:bg-muted/50"
          : "bg-muted/80 hover:bg-muted"
      )}
      onClick={handleClick}
    >
      {/* Unread indicator */}
      {!notification.read && (
        <div className="absolute left-1 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-primary" />
      )}

      {/* Icon */}
      <div className={cn("mt-0.5", config.color)}>
        <Icon className="h-5 w-5" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className={cn("text-sm font-medium", !notification.read && "font-semibold")}>
          {notification.title}
        </p>
        <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
          {notification.message}
        </p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
        </p>
      </div>

      {/* Dismiss button */}
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={(e) => {
          e.stopPropagation();
          onDismiss(notification.id);
        }}
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}
