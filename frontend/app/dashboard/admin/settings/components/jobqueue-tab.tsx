"use client";

import { TabsContent } from "@/components/ui/tabs";

interface JobQueueTabProps {
  JobQueueSettings: React.ComponentType<{
    localSettings: Record<string, unknown>;
    handleSettingChange: (key: string, value: unknown) => void;
  }>;
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function JobQueueTab({ JobQueueSettings, localSettings, handleSettingChange }: JobQueueTabProps) {
  return (
    <TabsContent value="jobqueue" className="space-y-6">
      <JobQueueSettings
        localSettings={localSettings}
        handleSettingChange={handleSettingChange}
      />
    </TabsContent>
  );
}
