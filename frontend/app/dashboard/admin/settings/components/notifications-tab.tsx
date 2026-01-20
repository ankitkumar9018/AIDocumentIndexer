"use client";

import { Bell } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";

interface NotificationsTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function NotificationsTab({ localSettings, handleSettingChange }: NotificationsTabProps) {
  return (
    <TabsContent value="notifications" className="space-y-6">
      {/* Notification Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notifications
          </CardTitle>
          <CardDescription>Configure notification preferences</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Processing Completed</p>
              <p className="text-sm text-muted-foreground">
                Notify when document processing finishes
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["notifications.processing_completed"] as boolean ?? true}
              onChange={(e) => handleSettingChange("notifications.processing_completed", e.target.checked)}
            />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Processing Failed</p>
              <p className="text-sm text-muted-foreground">
                Notify when document processing fails
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["notifications.processing_failed"] as boolean ?? true}
              onChange={(e) => handleSettingChange("notifications.processing_failed", e.target.checked)}
            />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Cost Alerts</p>
              <p className="text-sm text-muted-foreground">
                Notify when API costs exceed threshold
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["notifications.cost_alerts"] as boolean ?? true}
              onChange={(e) => handleSettingChange("notifications.cost_alerts", e.target.checked)}
            />
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
