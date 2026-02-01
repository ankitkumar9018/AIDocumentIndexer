"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Slider } from "@/components/ui/slider";
import {
  Zap,
  Loader2,
  RefreshCw,
  Save,
  AlertCircle,
  CheckCircle,
  Server,
  Cpu,
  HardDrive,
  Activity,
  RotateCcw,
  Play,
  Square,
  Power,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface RayDetailedStatus {
  status: string;
  ray_available: boolean;
  ray_connected: boolean;
  message?: string;
  cluster_resources?: {
    total_cpus: number;
    available_cpus: number;
    total_gpus: number;
    available_gpus: number;
    total_memory_gb: number;
    available_memory_gb: number;
    object_store_memory_gb: number;
  };
  nodes?: {
    total: number;
    active: number;
  };
}

interface RaySettings {
  "processing.ray_enabled": boolean;
  "processing.ray_address": string;
  "processing.ray_num_cpus": number;
  "processing.ray_num_gpus": number;
  "processing.ray_memory_limit_gb": number;
  "processing.ray_num_workers": number;
  "processing.ray_fallback_to_local": boolean;
}

const defaultSettings: RaySettings = {
  "processing.ray_enabled": true,
  "processing.ray_address": "",
  "processing.ray_num_cpus": 4,
  "processing.ray_num_gpus": 0,
  "processing.ray_memory_limit_gb": 8,
  "processing.ray_num_workers": 8,
  "processing.ray_fallback_to_local": true,
};

export function RayTab() {
  const { data: session } = useSession();
  const [status, setStatus] = useState<RayDetailedStatus | null>(null);
  const [settings, setSettings] = useState<RaySettings>(defaultSettings);
  const [originalSettings, setOriginalSettings] = useState<RaySettings>(defaultSettings);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [controlAction, setControlAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [hasChanges, setHasChanges] = useState(false);

  const accessToken = (session as any)?.accessToken as string | undefined;

  const fetchStatus = useCallback(async () => {
    if (!accessToken) {
      setStatus({
        status: "unknown",
        ray_available: false,
        ray_connected: false,
        message: "Not authenticated",
      });
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/diagnostics/ray/status`, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch {
      console.error("Failed to fetch Ray status");
    }
  }, [accessToken]);

  const fetchSettings = useCallback(async () => {
    if (!accessToken) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/admin/settings`, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        const newSettings: RaySettings = { ...defaultSettings };

        Object.keys(defaultSettings).forEach((key) => {
          if (data[key] !== undefined) {
            (newSettings as any)[key] = data[key];
          }
        });

        setSettings(newSettings);
        setOriginalSettings(newSettings);
      }
    } catch {
      console.error("Failed to fetch settings");
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchStatus();
    fetchSettings();
    const interval = setInterval(fetchStatus, 5000); // Auto-refresh every 5s
    return () => clearInterval(interval);
  }, [fetchStatus, fetchSettings]);

  useEffect(() => {
    const changed = Object.keys(settings).some(
      (key) => (settings as any)[key] !== (originalSettings as any)[key]
    );
    setHasChanges(changed);
  }, [settings, originalSettings]);

  const handleRayControl = async (action: "start" | "stop" | "restart") => {
    if (!accessToken) return;

    setControlAction(action);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(`${API_BASE}/diagnostics/ray/${action}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });

      const data = await response.json();

      if (data.success) {
        setSuccess(data.message);
        // Refresh status after action
        setTimeout(fetchStatus, 1000);
      } else {
        setError(data.message || `Failed to ${action} Ray`);
      }
    } catch {
      setError(`Failed to ${action} Ray`);
    } finally {
      setControlAction(null);
    }
  };

  const handleSave = async () => {
    if (!accessToken) return;

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(`${API_BASE}/admin/settings`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify(settings),
      });

      if (response.ok) {
        setSuccess("Settings saved. Click 'Restart Ray' to apply changes.");
        setOriginalSettings({ ...settings });
        setHasChanges(false);
      } else {
        const data = await response.json();
        setError(data.detail || "Failed to save settings");
      }
    } catch {
      setError("Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setSettings({ ...originalSettings });
    setHasChanges(false);
  };

  const updateSetting = (key: keyof RaySettings, value: any) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const getStatusColor = () => {
    if (status?.ray_connected) return "bg-green-500";
    if (status?.ray_available) return "bg-yellow-500";
    return "bg-red-500";
  };

  const getStatusText = () => {
    if (status?.ray_connected) return "Running";
    if (status?.ray_available) return "Stopped";
    return "Unavailable";
  };

  return (
    <TabsContent value="ray" className="space-y-6">
      {/* Status & Control Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Ray Cluster Status
              </CardTitle>
              <CardDescription>
                Monitor and control the Ray distributed processing cluster
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={fetchStatus}>
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Status Display */}
          <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
            <div className="flex items-center gap-4">
              <div className={`h-3 w-3 rounded-full ${getStatusColor()}`} />
              <div>
                <p className="font-medium">{getStatusText()}</p>
                <p className="text-sm text-muted-foreground">
                  {status?.message || (status?.ray_connected ? "Cluster is healthy" : "Cluster is not running")}
                </p>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="flex gap-2">
              {!status?.ray_connected ? (
                <Button
                  onClick={() => handleRayControl("start")}
                  disabled={controlAction !== null || !status?.ray_available}
                  className="bg-green-600 hover:bg-green-700"
                >
                  {controlAction === "start" ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Start
                </Button>
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={() => handleRayControl("restart")}
                    disabled={controlAction !== null}
                  >
                    {controlAction === "restart" ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <RotateCcw className="h-4 w-4 mr-2" />
                    )}
                    Restart
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={() => handleRayControl("stop")}
                    disabled={controlAction !== null}
                  >
                    {controlAction === "stop" ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Square className="h-4 w-4 mr-2" />
                    )}
                    Stop
                  </Button>
                </>
              )}
            </div>
          </div>

          {/* Cluster Resources */}
          {status?.ray_connected && status?.cluster_resources && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-muted rounded-lg">
                <Cpu className="h-5 w-5 mx-auto mb-1 text-blue-500" />
                <p className="text-2xl font-bold">{status.cluster_resources.available_cpus}/{status.cluster_resources.total_cpus}</p>
                <p className="text-xs text-muted-foreground">CPUs Available</p>
              </div>
              <div className="text-center p-3 bg-muted rounded-lg">
                <Activity className="h-5 w-5 mx-auto mb-1 text-purple-500" />
                <p className="text-2xl font-bold">{status.cluster_resources.available_gpus}/{status.cluster_resources.total_gpus}</p>
                <p className="text-xs text-muted-foreground">GPUs Available</p>
              </div>
              <div className="text-center p-3 bg-muted rounded-lg">
                <HardDrive className="h-5 w-5 mx-auto mb-1 text-green-500" />
                <p className="text-2xl font-bold">{status.cluster_resources.available_memory_gb.toFixed(1)}</p>
                <p className="text-xs text-muted-foreground">Memory (GB)</p>
              </div>
              <div className="text-center p-3 bg-muted rounded-lg">
                <Server className="h-5 w-5 mx-auto mb-1 text-orange-500" />
                <p className="text-2xl font-bold">{status.nodes?.active || 0}/{status.nodes?.total || 0}</p>
                <p className="text-xs text-muted-foreground">Active Nodes</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* Settings */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Basic Settings */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Configuration</CardTitle>
                <CardDescription>Ray cluster settings</CardDescription>
              </div>
              {hasChanges && (
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleReset}>
                    <RotateCcw className="h-4 w-4 mr-1" />
                    Reset
                  </Button>
                  <Button size="sm" onClick={handleSave} disabled={saving}>
                    {saving ? (
                      <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    ) : (
                      <Save className="h-4 w-4 mr-1" />
                    )}
                    Save
                  </Button>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="ray-enabled">Auto-Start Ray</Label>
                <p className="text-xs text-muted-foreground">
                  Start Ray automatically when server starts
                </p>
              </div>
              <Switch
                id="ray-enabled"
                checked={settings["processing.ray_enabled"]}
                onCheckedChange={(checked) =>
                  updateSetting("processing.ray_enabled", checked)
                }
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ray-address">Cluster Address</Label>
              <Input
                id="ray-address"
                value={settings["processing.ray_address"]}
                onChange={(e) =>
                  updateSetting("processing.ray_address", e.target.value)
                }
                placeholder="Leave empty for local cluster"
              />
              <p className="text-xs text-muted-foreground">
                Empty = local cluster, or ray://host:10001 for remote
              </p>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="ray-fallback">Local Fallback</Label>
                <p className="text-xs text-muted-foreground">
                  Use ThreadPool if Ray unavailable
                </p>
              </div>
              <Switch
                id="ray-fallback"
                checked={settings["processing.ray_fallback_to_local"]}
                onCheckedChange={(checked) =>
                  updateSetting("processing.ray_fallback_to_local", checked)
                }
              />
            </div>
          </CardContent>
        </Card>

        {/* Resource Limits */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Resource Limits</CardTitle>
            <CardDescription>Control resource allocation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>CPUs</Label>
                <span className="text-sm font-medium">
                  {settings["processing.ray_num_cpus"] || "Auto"}
                </span>
              </div>
              <Slider
                value={[settings["processing.ray_num_cpus"]]}
                onValueChange={([value]) =>
                  updateSetting("processing.ray_num_cpus", value)
                }
                min={0}
                max={32}
                step={1}
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>GPUs</Label>
                <span className="text-sm font-medium">
                  {settings["processing.ray_num_gpus"]}
                </span>
              </div>
              <Slider
                value={[settings["processing.ray_num_gpus"]]}
                onValueChange={([value]) =>
                  updateSetting("processing.ray_num_gpus", value)
                }
                min={0}
                max={8}
                step={1}
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Memory Limit (GB)</Label>
                <span className="text-sm font-medium">
                  {settings["processing.ray_memory_limit_gb"]} GB
                </span>
              </div>
              <Slider
                value={[settings["processing.ray_memory_limit_gb"]]}
                onValueChange={([value]) =>
                  updateSetting("processing.ray_memory_limit_gb", value)
                }
                min={1}
                max={64}
                step={1}
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Workers</Label>
                <span className="text-sm font-medium">
                  {settings["processing.ray_num_workers"]}
                </span>
              </div>
              <Slider
                value={[settings["processing.ray_num_workers"]]}
                onValueChange={([value]) =>
                  updateSetting("processing.ray_num_workers", value)
                }
                min={1}
                max={32}
                step={1}
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </TabsContent>
  );
}
