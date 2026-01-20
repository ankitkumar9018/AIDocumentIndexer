"use client";

import { useState } from "react";
import { getErrorMessage } from "@/lib/errors";
import {
  Server,
  Loader2,
  CheckCircle,
  AlertCircle,
  Zap,
  Sparkles,
  Activity,
  Globe,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";
import { useSettingsData } from "../hooks/use-settings-data";
import { useSettingsActions } from "../hooks/use-settings-actions";
import { useUser } from "@/lib/auth";

export function OverviewTab() {
  const { isAuthenticated } = useUser();
  const [applyingPreset, setApplyingPreset] = useState<string | null>(null);

  const {
    health: healthData,
    healthLoading,
    presets: presetsData,
    refetchSettings,
  } = useSettingsData(isAuthenticated);

  const { applyPreset } = useSettingsActions();

  const getServiceIcon = (status: string) => {
    switch (status) {
      case "online":
      case "connected":
      case "configured":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "not_configured":
      case "unavailable":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-red-500" />;
    }
  };

  const getServiceStatusColor = (status: string) => {
    switch (status) {
      case "online":
      case "connected":
      case "configured":
        return "text-green-600";
      case "not_configured":
      case "unavailable":
        return "text-yellow-600";
      default:
        return "text-red-600";
    }
  };

  return (
    <TabsContent value="overview" className="space-y-6">
      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Server className="h-5 w-5" />
            System Status
          </CardTitle>
          <CardDescription>Current status of all services</CardDescription>
        </CardHeader>
        <CardContent>
          {healthLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : healthData?.services ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {Object.entries(healthData.services).map(([name, service]) => (
                <div
                  key={name}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div className="flex items-center gap-2">
                    {getServiceIcon(service.status)}
                    <span className="text-sm capitalize">
                      {name.replace(/_/g, " ")}
                    </span>
                  </div>
                  <span className={`text-xs ${getServiceStatusColor(service.status)}`}>
                    {service.status === "connected" || service.status === "online"
                      ? service.type?.toUpperCase() || "Online"
                      : service.status === "configured"
                      ? service.providers?.join(", ") || "Configured"
                      : service.status === "not_configured"
                      ? "Not Configured"
                      : service.status}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground">Unable to fetch system status</p>
          )}
        </CardContent>
      </Card>

      {/* Quick Presets */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Quick Presets
          </CardTitle>
          <CardDescription>
            Apply pre-configured settings bundles optimized for different use cases
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!presetsData ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : presetsData?.presets ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {presetsData.presets.map((preset) => (
                <div
                  key={preset.id}
                  className="flex flex-col p-4 rounded-lg border hover:border-primary/50 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    {preset.id === "speed" && <Zap className="h-4 w-4 text-yellow-500" />}
                    {preset.id === "quality" && <Sparkles className="h-4 w-4 text-purple-500" />}
                    {preset.id === "balanced" && <Activity className="h-4 w-4 text-blue-500" />}
                    {preset.id === "offline" && <Globe className="h-4 w-4 text-green-500" />}
                    <span className="font-medium">{preset.name}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mb-3 flex-grow">
                    {preset.description}
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={async () => {
                      setApplyingPreset(preset.id);
                      try {
                        await applyPreset.mutateAsync(preset.id);
                        refetchSettings();
                      } catch (err) {
                        console.error("Failed to apply preset:", err);
                        alert(`Failed to apply preset: ${getErrorMessage(err)}`);
                      } finally {
                        setApplyingPreset(null);
                      }
                    }}
                    disabled={applyingPreset !== null}
                    className="w-full"
                  >
                    {applyingPreset === preset.id ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Apply
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground">No presets available</p>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  );
}
