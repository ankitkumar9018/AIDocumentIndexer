"use client";

import { useState, useEffect, useCallback } from "react";
import { Bell, CheckCheck, Settings, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  Notification,
  NotificationItem,
  NotificationType,
} from "./notification-item";

// Mock notifications for demo - in production, these would come from an API/WebSocket
const mockNotifications: Notification[] = [
  {
    id: "1",
    type: "upload_complete",
    title: "Upload Complete",
    message: "5 documents have been successfully uploaded and indexed.",
    timestamp: new Date(Date.now() - 5 * 60 * 1000), // 5 mins ago
    read: false,
    link: "/dashboard/documents",
  },
  {
    id: "2",
    type: "sync_complete",
    title: "Google Drive Sync Complete",
    message: "Synced 23 new files from your connected Google Drive.",
    timestamp: new Date(Date.now() - 30 * 60 * 1000), // 30 mins ago
    read: false,
    link: "/dashboard/connectors",
  },
  {
    id: "3",
    type: "export_ready",
    title: "Export Ready",
    message: 'Your PDF export "Q4 Report" is ready for download.',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    read: true,
    link: "/dashboard/create",
  },
  {
    id: "4",
    type: "system_info",
    title: "New Features Available",
    message: "Check out the new Mind Map generation feature in document creation!",
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000), // 1 day ago
    read: true,
  },
];

export function NotificationCenter() {
  const [notifications, setNotifications] = useState<Notification[]>(mockNotifications);
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("all");

  const unreadCount = notifications.filter((n) => !n.read).length;

  const filteredNotifications = notifications.filter((n) => {
    if (activeTab === "unread") return !n.read;
    return true;
  });

  const handleMarkRead = useCallback((id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  }, []);

  const handleDismiss = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const handleMarkAllRead = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }, []);

  const handleClearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  const handleNotificationClick = useCallback((notification: Notification) => {
    if (notification.link) {
      window.location.href = notification.link;
    }
  }, []);

  // WebSocket connection for real-time notifications (placeholder)
  useEffect(() => {
    // In production, connect to WebSocket for real-time updates
    // const ws = new WebSocket(`${process.env.NEXT_PUBLIC_WS_URL}/notifications`);
    // ws.onmessage = (event) => {
    //   const newNotification = JSON.parse(event.data);
    //   setNotifications((prev) => [newNotification, ...prev]);
    // };
    // return () => ws.close();
  }, []);

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex h-4 w-4 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground">
                {unreadCount > 9 ? "9+" : unreadCount}
              </span>
            </span>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-96 p-0" align="end">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="font-semibold">Notifications</h3>
          <div className="flex items-center gap-1">
            {unreadCount > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 text-xs"
                onClick={handleMarkAllRead}
              >
                <CheckCheck className="h-4 w-4 mr-1" />
                Mark all read
              </Button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="all" value={activeTab} onValueChange={setActiveTab}>
          <div className="px-4 pt-2">
            <TabsList className="w-full grid grid-cols-2">
              <TabsTrigger value="all">
                All
                {notifications.length > 0 && (
                  <span className="ml-1.5 text-muted-foreground">
                    ({notifications.length})
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger value="unread">
                Unread
                {unreadCount > 0 && (
                  <span className="ml-1.5 text-muted-foreground">
                    ({unreadCount})
                  </span>
                )}
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="all" className="m-0">
            <NotificationList
              notifications={filteredNotifications}
              onMarkRead={handleMarkRead}
              onDismiss={handleDismiss}
              onClick={handleNotificationClick}
            />
          </TabsContent>
          <TabsContent value="unread" className="m-0">
            <NotificationList
              notifications={filteredNotifications}
              onMarkRead={handleMarkRead}
              onDismiss={handleDismiss}
              onClick={handleNotificationClick}
            />
          </TabsContent>
        </Tabs>

        {/* Footer */}
        {notifications.length > 0 && (
          <div className="flex items-center justify-between px-4 py-2 border-t">
            <Button
              variant="ghost"
              size="sm"
              className="text-xs text-muted-foreground hover:text-destructive"
              onClick={handleClearAll}
            >
              <Trash2 className="h-3.5 w-3.5 mr-1" />
              Clear all
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-xs"
              onClick={() => {
                setIsOpen(false);
                window.location.href = "/dashboard/admin/settings";
              }}
            >
              <Settings className="h-3.5 w-3.5 mr-1" />
              Settings
            </Button>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

interface NotificationListProps {
  notifications: Notification[];
  onMarkRead: (id: string) => void;
  onDismiss: (id: string) => void;
  onClick: (notification: Notification) => void;
}

function NotificationList({
  notifications,
  onMarkRead,
  onDismiss,
  onClick,
}: NotificationListProps) {
  if (notifications.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <Bell className="h-10 w-10 text-muted-foreground/50 mb-2" />
        <p className="text-sm text-muted-foreground">No notifications</p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          You&apos;re all caught up!
        </p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-[300px]">
      <div className="p-2 space-y-1">
        {notifications.map((notification) => (
          <NotificationItem
            key={notification.id}
            notification={notification}
            onMarkRead={onMarkRead}
            onDismiss={onDismiss}
            onClick={onClick}
          />
        ))}
      </div>
    </ScrollArea>
  );
}
