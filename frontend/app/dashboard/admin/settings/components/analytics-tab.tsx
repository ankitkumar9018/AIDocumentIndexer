"use client";

import { TabsContent } from "@/components/ui/tabs";

interface AnalyticsTabProps {
  UsageAnalyticsCard: React.ComponentType;
  ProviderHealthCard: React.ComponentType;
  CostAlertsCard: React.ComponentType;
}

export function AnalyticsTab({ UsageAnalyticsCard, ProviderHealthCard, CostAlertsCard }: AnalyticsTabProps) {
  return (
    <TabsContent value="analytics" className="space-y-6">
      {/* LLM Usage Analytics Card */}
      <UsageAnalyticsCard />

      {/* Provider Health Card */}
      <ProviderHealthCard />

      {/* Cost Alerts Card */}
      <CostAlertsCard />
    </TabsContent>
  );
}
