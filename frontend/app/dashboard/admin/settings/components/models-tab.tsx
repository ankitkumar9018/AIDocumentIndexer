"use client";

import { TabsContent } from "@/components/ui/tabs";
import type { LLMProvider } from "@/lib/api/client";

interface ModelsTabProps {
  ModelConfigurationSection: React.ComponentType<{ providers: LLMProvider[] }>;
  providers: LLMProvider[];
}

export function ModelsTab({ ModelConfigurationSection, providers }: ModelsTabProps) {
  return (
    <TabsContent value="models" className="space-y-6">
      <ModelConfigurationSection providers={providers} />
    </TabsContent>
  );
}
