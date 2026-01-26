"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Eye,
  Brain,
  Server,
  Cpu,
  MemoryStick,
  HardDrive,
  Activity,
  Zap,
  Settings,
  RefreshCw,
  Play,
  Pause,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Database,
  Network,
  Layers,
  GitBranch,
  Terminal,
  BarChart3,
  TrendingUp,
  Gauge,
  Info,
  Save,
} from "lucide-react";

// =============================================================================
// Types
// =============================================================================

interface ServiceStatus {
  status: "connected" | "disconnected" | "degraded" | "initializing";
  message?: string;
  lastCheck?: string;
}

interface RayClusterInfo {
  status: ServiceStatus;
  headNode: {
    address: string;
    uptime: string;
    cpuUsage: number;
    memoryUsage: number;
    gpuCount: number;
  };
  workers: Array<{
    id: string;
    address: string;
    status: "alive" | "dead" | "pending";
    cpuUsage: number;
    memoryUsage: number;
    taskCount: number;
  }>;
  resources: {
    totalCPU: number;
    availableCPU: number;
    totalMemory: number;
    availableMemory: number;
    totalGPU: number;
    availableGPU: number;
  };
  tasks: {
    pending: number;
    running: number;
    completed: number;
    failed: number;
  };
}

interface VLMConfig {
  enabled: boolean;
  provider: string;
  model: string;
  maxImagesPerRequest: number;
  enableForVisualDocs: boolean;
  fallbackToOCR: boolean;
  confidenceThreshold: number;
}

interface RLMConfig {
  enabled: boolean;
  provider: string;
  model: string;
  contextThreshold: number;
  maxRecursionDepth: number;
  sandboxMode: string;
  enableSelfRefinement: boolean;
}

interface MLServiceMetrics {
  service: string;
  requestsToday: number;
  avgLatencyMs: number;
  successRate: number;
  costToday: number;
  tokensUsed: number;
}

// =============================================================================
// API Helpers
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

async function fetchRayStatus(): Promise<RayClusterInfo | null> {
  try {
    const response = await fetch(`${API_BASE}/admin/ray/status`, {
      credentials: "include",
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

async function fetchVLMConfig(): Promise<VLMConfig> {
  try {
    const response = await fetch(`${API_BASE}/admin/vlm/config`, {
      credentials: "include",
    });
    if (!response.ok) throw new Error();
    return await response.json();
  } catch {
    return {
      enabled: false,
      provider: "anthropic",
      model: "claude-3-5-sonnet-20241022",
      maxImagesPerRequest: 20,
      enableForVisualDocs: true,
      fallbackToOCR: true,
      confidenceThreshold: 0.8,
    };
  }
}

async function fetchRLMConfig(): Promise<RLMConfig> {
  try {
    const response = await fetch(`${API_BASE}/admin/rlm/config`, {
      credentials: "include",
    });
    if (!response.ok) throw new Error();
    return await response.json();
  } catch {
    return {
      enabled: false,
      provider: "anthropic",
      model: "claude-3-5-sonnet-20241022",
      contextThreshold: 100000,
      maxRecursionDepth: 5,
      sandboxMode: "local",
      enableSelfRefinement: true,
    };
  }
}

// =============================================================================
// Ray Cluster Panel
// =============================================================================

function RayClusterPanel() {
  const [clusterInfo, setClusterInfo] = useState<RayClusterInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Config state
  const [config, setConfig] = useState({
    address: "auto",
    numWorkers: 4,
    useForEmbeddings: true,
    useForKG: true,
    useForVLM: false,
    workerConcurrency: 2,
  });

  useEffect(() => {
    loadClusterInfo();
    const interval = setInterval(loadClusterInfo, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadClusterInfo = async () => {
    const info = await fetchRayStatus();
    setClusterInfo(info);
    setLoading(false);
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadClusterInfo();
    setRefreshing(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  const isConnected = clusterInfo?.status.status === "connected";

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-2 rounded-lg",
                isConnected ? "bg-green-100" : "bg-red-100"
              )}>
                <Server className={cn(
                  "h-5 w-5",
                  isConnected ? "text-green-600" : "text-red-600"
                )} />
              </div>
              <div>
                <CardTitle>Ray Cluster</CardTitle>
                <CardDescription>
                  Distributed computing for ML workloads
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={isConnected ? "default" : "destructive"}>
                {isConnected ? "Connected" : "Disconnected"}
              </Badge>
              <Button
                variant="outline"
                size="icon"
                onClick={handleRefresh}
                disabled={refreshing}
              >
                <RefreshCw className={cn("h-4 w-4", refreshing && "animate-spin")} />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {!isConnected ? (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Ray cluster not connected</AlertTitle>
              <AlertDescription>
                Configure the Ray cluster address below or start a local Ray cluster.
                Tasks will fall back to Celery when Ray is unavailable.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-4 gap-4">
              <MetricCard
                label="CPU Usage"
                value={`${clusterInfo!.headNode.cpuUsage}%`}
                icon={Cpu}
                trend={clusterInfo!.headNode.cpuUsage > 80 ? "warning" : "normal"}
              />
              <MetricCard
                label="Memory"
                value={`${clusterInfo!.headNode.memoryUsage}%`}
                icon={MemoryStick}
                trend={clusterInfo!.headNode.memoryUsage > 80 ? "warning" : "normal"}
              />
              <MetricCard
                label="Workers"
                value={`${clusterInfo!.workers.filter(w => w.status === "alive").length}/${clusterInfo!.workers.length}`}
                icon={Network}
              />
              <MetricCard
                label="Tasks Running"
                value={clusterInfo!.tasks.running.toString()}
                icon={Activity}
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Ray Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="ray-address">Ray Address</Label>
              <Input
                id="ray-address"
                value={config.address}
                onChange={(e) => setConfig({ ...config, address: e.target.value })}
                placeholder="auto or ray://hostname:port"
              />
              <p className="text-xs text-muted-foreground">
                Use "auto" for local cluster or specify remote address
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="num-workers">Number of Workers</Label>
              <Input
                id="num-workers"
                type="number"
                min={1}
                max={32}
                value={config.numWorkers}
                onChange={(e) => setConfig({ ...config, numWorkers: parseInt(e.target.value) || 4 })}
              />
            </div>
          </div>

          <Separator />

          <div className="space-y-4">
            <h4 className="font-medium">Route Tasks to Ray</h4>
            <div className="grid grid-cols-3 gap-4">
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Embeddings</span>
                </div>
                <Switch
                  checked={config.useForEmbeddings}
                  onCheckedChange={(checked) =>
                    setConfig({ ...config, useForEmbeddings: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-2">
                  <GitBranch className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">Knowledge Graph</span>
                </div>
                <Switch
                  checked={config.useForKG}
                  onCheckedChange={(checked) =>
                    setConfig({ ...config, useForKG: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-2">
                  <Eye className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">VLM Processing</span>
                </div>
                <Switch
                  checked={config.useForVLM}
                  onCheckedChange={(checked) =>
                    setConfig({ ...config, useForVLM: checked })
                  }
                />
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <Button>
              <Save className="h-4 w-4 mr-2" />
              Save Configuration
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Workers List */}
      {isConnected && clusterInfo!.workers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5" />
              Worker Nodes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {clusterInfo!.workers.map((worker) => (
                <div
                  key={worker.id}
                  className="flex items-center justify-between p-3 border rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "w-2 h-2 rounded-full",
                      worker.status === "alive" && "bg-green-500",
                      worker.status === "dead" && "bg-red-500",
                      worker.status === "pending" && "bg-yellow-500"
                    )} />
                    <div>
                      <p className="font-mono text-sm">{worker.id}</p>
                      <p className="text-xs text-muted-foreground">{worker.address}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <div className="flex items-center gap-2">
                      <Cpu className="h-4 w-4 text-muted-foreground" />
                      <span>{worker.cpuUsage}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <MemoryStick className="h-4 w-4 text-muted-foreground" />
                      <span>{worker.memoryUsage}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-muted-foreground" />
                      <span>{worker.taskCount} tasks</span>
                    </div>
                    <Badge variant={worker.status === "alive" ? "secondary" : "destructive"}>
                      {worker.status}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// =============================================================================
// VLM Panel
// =============================================================================

function VLMPanel() {
  const [config, setConfig] = useState<VLMConfig>({
    enabled: false,
    provider: "anthropic",
    model: "claude-3-5-sonnet-20241022",
    maxImagesPerRequest: 20,
    enableForVisualDocs: true,
    fallbackToOCR: true,
    confidenceThreshold: 0.8,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    const data = await fetchVLMConfig();
    setConfig(data);
    setLoading(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await fetch(`${API_BASE}/admin/vlm/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(config),
      });
    } catch (error) {
      console.error("Failed to save VLM config:", error);
    }
    setSaving(false);
  };

  const handleTest = async () => {
    setTestResult(null);
    try {
      const response = await fetch(`${API_BASE}/admin/vlm/test`, {
        method: "POST",
        credentials: "include",
      });
      const result = await response.json();
      setTestResult({
        success: result.success,
        message: result.message || (result.success ? "VLM is working correctly" : "VLM test failed"),
      });
    } catch {
      setTestResult({ success: false, message: "Failed to connect to VLM service" });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-2 rounded-lg",
                config.enabled ? "bg-violet-100" : "bg-gray-100"
              )}>
                <Eye className={cn(
                  "h-5 w-5",
                  config.enabled ? "text-violet-600" : "text-gray-400"
                )} />
              </div>
              <div>
                <CardTitle>Vision Language Model (VLM)</CardTitle>
                <CardDescription>
                  Process images, charts, and visual documents with AI
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Switch
                checked={config.enabled}
                onCheckedChange={(checked) =>
                  setConfig({ ...config, enabled: checked })
                }
              />
              <Badge variant={config.enabled ? "default" : "secondary"}>
                {config.enabled ? "Enabled" : "Disabled"}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {!config.enabled && (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertTitle>VLM is disabled</AlertTitle>
              <AlertDescription>
                Enable VLM to extract information from charts, infographics, and scanned documents.
                Requires a compatible API key (Anthropic, OpenAI, or local Qwen model).
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            VLM Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label>Provider</Label>
              <Select
                value={config.provider}
                onValueChange={(value) => setConfig({ ...config, provider: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="anthropic">Anthropic (Claude)</SelectItem>
                  <SelectItem value="openai">OpenAI (GPT-4V)</SelectItem>
                  <SelectItem value="qwen">Qwen (Local/Ollama)</SelectItem>
                  <SelectItem value="google">Google (Gemini)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Model</Label>
              <Select
                value={config.model}
                onValueChange={(value) => setConfig({ ...config, model: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {config.provider === "anthropic" && (
                    <>
                      <SelectItem value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</SelectItem>
                      <SelectItem value="claude-3-5-haiku-20241022">Claude 3.5 Haiku (Fast)</SelectItem>
                    </>
                  )}
                  {config.provider === "openai" && (
                    <>
                      <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                      <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                    </>
                  )}
                  {config.provider === "qwen" && (
                    <>
                      <SelectItem value="qwen2-vl-72b">Qwen2-VL 72B</SelectItem>
                      <SelectItem value="qwen2-vl-7b">Qwen2-VL 7B</SelectItem>
                    </>
                  )}
                  {config.provider === "google" && (
                    <>
                      <SelectItem value="gemini-1.5-pro">Gemini 1.5 Pro</SelectItem>
                      <SelectItem value="gemini-1.5-flash">Gemini 1.5 Flash</SelectItem>
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Separator />

          <div className="space-y-4">
            <h4 className="font-medium">Processing Options</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <p className="text-sm font-medium">Auto-detect visual documents</p>
                  <p className="text-xs text-muted-foreground">
                    Automatically use VLM for charts, diagrams, and infographics
                  </p>
                </div>
                <Switch
                  checked={config.enableForVisualDocs}
                  onCheckedChange={(checked) =>
                    setConfig({ ...config, enableForVisualDocs: checked })
                  }
                />
              </div>
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <p className="text-sm font-medium">Fallback to OCR</p>
                  <p className="text-xs text-muted-foreground">
                    Use OCR if VLM fails or is unavailable
                  </p>
                </div>
                <Switch
                  checked={config.fallbackToOCR}
                  onCheckedChange={(checked) =>
                    setConfig({ ...config, fallbackToOCR: checked })
                  }
                />
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Max Images per Request</Label>
              <span className="text-sm text-muted-foreground">{config.maxImagesPerRequest}</span>
            </div>
            <Slider
              value={[config.maxImagesPerRequest]}
              onValueChange={([value]) =>
                setConfig({ ...config, maxImagesPerRequest: value })
              }
              min={1}
              max={50}
              step={1}
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Confidence Threshold</Label>
              <span className="text-sm text-muted-foreground">{(config.confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
            <Slider
              value={[config.confidenceThreshold * 100]}
              onValueChange={([value]) =>
                setConfig({ ...config, confidenceThreshold: value / 100 })
              }
              min={50}
              max={100}
              step={5}
            />
            <p className="text-xs text-muted-foreground">
              Minimum confidence required before including VLM results
            </p>
          </div>

          {testResult && (
            <Alert variant={testResult.success ? "default" : "destructive"}>
              {testResult.success ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <XCircle className="h-4 w-4" />
              )}
              <AlertTitle>{testResult.success ? "Success" : "Failed"}</AlertTitle>
              <AlertDescription>{testResult.message}</AlertDescription>
            </Alert>
          )}

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={handleTest}>
              <Play className="h-4 w-4 mr-2" />
              Test Connection
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              <Save className="h-4 w-4 mr-2" />
              {saving ? "Saving..." : "Save Configuration"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// =============================================================================
// RLM Panel
// =============================================================================

function RLMPanel() {
  const [config, setConfig] = useState<RLMConfig>({
    enabled: false,
    provider: "anthropic",
    model: "claude-3-5-sonnet-20241022",
    contextThreshold: 100000,
    maxRecursionDepth: 5,
    sandboxMode: "local",
    enableSelfRefinement: true,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    const data = await fetchRLMConfig();
    setConfig(data);
    setLoading(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await fetch(`${API_BASE}/admin/rlm/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(config),
      });
    } catch (error) {
      console.error("Failed to save RLM config:", error);
    }
    setSaving(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-2 rounded-lg",
                config.enabled ? "bg-indigo-100" : "bg-gray-100"
              )}>
                <Brain className={cn(
                  "h-5 w-5",
                  config.enabled ? "text-indigo-600" : "text-gray-400"
                )} />
              </div>
              <div>
                <CardTitle>Recursive Language Model (RLM)</CardTitle>
                <CardDescription>
                  Process 10M+ token contexts with O(log N) complexity
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Switch
                checked={config.enabled}
                onCheckedChange={(checked) =>
                  setConfig({ ...config, enabled: checked })
                }
              />
              <Badge variant={config.enabled ? "default" : "secondary"}>
                {config.enabled ? "Enabled" : "Disabled"}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {!config.enabled && (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertTitle>RLM is disabled</AlertTitle>
              <AlertDescription>
                Enable RLM to handle queries requiring massive context windows (10M+ tokens).
                Based on MIT/Prime Intellect research for logarithmic-complexity context processing.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            RLM Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label>LLM Provider</Label>
              <Select
                value={config.provider}
                onValueChange={(value) => setConfig({ ...config, provider: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="anthropic">Anthropic (Claude)</SelectItem>
                  <SelectItem value="openai">OpenAI</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Model</Label>
              <Select
                value={config.model}
                onValueChange={(value) => setConfig({ ...config, model: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {config.provider === "anthropic" && (
                    <>
                      <SelectItem value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet (Recommended)</SelectItem>
                      <SelectItem value="claude-3-opus-20240229">Claude 3 Opus</SelectItem>
                    </>
                  )}
                  {config.provider === "openai" && (
                    <>
                      <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                      <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Separator />

          <div className="space-y-4">
            <h4 className="font-medium">Context & Recursion Settings</h4>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Context Threshold (tokens)</Label>
                  <p className="text-xs text-muted-foreground">
                    Use RLM when context exceeds this threshold
                  </p>
                </div>
                <span className="text-sm font-mono">{config.contextThreshold.toLocaleString()}</span>
              </div>
              <Slider
                value={[config.contextThreshold / 1000]}
                onValueChange={([value]) =>
                  setConfig({ ...config, contextThreshold: value * 1000 })
                }
                min={50}
                max={500}
                step={10}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>50K tokens</span>
                <span>500K tokens</span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Max Recursion Depth</Label>
                  <p className="text-xs text-muted-foreground">
                    Maximum levels of recursive context navigation
                  </p>
                </div>
                <span className="text-sm">{config.maxRecursionDepth}</span>
              </div>
              <Slider
                value={[config.maxRecursionDepth]}
                onValueChange={([value]) =>
                  setConfig({ ...config, maxRecursionDepth: value })
                }
                min={1}
                max={10}
                step={1}
              />
            </div>
          </div>

          <Separator />

          <div className="space-y-4">
            <h4 className="font-medium">Execution Settings</h4>

            <div className="space-y-2">
              <Label>Sandbox Mode</Label>
              <Select
                value={config.sandboxMode}
                onValueChange={(value) => setConfig({ ...config, sandboxMode: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="local">Local (RestrictedPython)</SelectItem>
                  <SelectItem value="modal">Modal (Cloud Sandbox)</SelectItem>
                  <SelectItem value="e2b">E2B (Secure Execution)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Code execution environment for RLM context navigation
              </p>
            </div>

            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <p className="text-sm font-medium">Enable Self-Refinement</p>
                <p className="text-xs text-muted-foreground">
                  Allow RLM to iteratively improve answers
                </p>
              </div>
              <Switch
                checked={config.enableSelfRefinement}
                onCheckedChange={(checked) =>
                  setConfig({ ...config, enableSelfRefinement: checked })
                }
              />
            </div>
          </div>

          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Resource Warning</AlertTitle>
            <AlertDescription>
              RLM can be expensive for very large contexts. Monitor usage in the Cost Tracking panel.
              Consider setting up cost alerts before enabling for production use.
            </AlertDescription>
          </Alert>

          <div className="flex justify-end">
            <Button onClick={handleSave} disabled={saving}>
              <Save className="h-4 w-4 mr-2" />
              {saving ? "Saving..." : "Save Configuration"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* RLM Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            RLM Usage Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            <MetricCard
              label="Queries Today"
              value="23"
              icon={Activity}
            />
            <MetricCard
              label="Avg Latency"
              value="4.2s"
              icon={Clock}
            />
            <MetricCard
              label="Success Rate"
              value="98.5%"
              icon={CheckCircle2}
              trend="good"
            />
            <MetricCard
              label="Cost Today"
              value="$12.45"
              icon={TrendingUp}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// =============================================================================
// Shared Components
// =============================================================================

interface MetricCardProps {
  label: string;
  value: string;
  icon: React.ElementType;
  trend?: "good" | "warning" | "normal";
}

function MetricCard({ label, value, icon: Icon, trend = "normal" }: MetricCardProps) {
  return (
    <div className="p-4 border rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <Icon className="h-4 w-4 text-muted-foreground" />
        {trend === "good" && <Badge variant="secondary" className="text-green-600 bg-green-50">Good</Badge>}
        {trend === "warning" && <Badge variant="secondary" className="text-amber-600 bg-amber-50">High</Badge>}
      </div>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-sm text-muted-foreground">{label}</p>
    </div>
  );
}

// =============================================================================
// Main ML Services Panel
// =============================================================================

interface MLServicesPanelProps {
  className?: string;
}

export function MLServicesPanel({ className }: MLServicesPanelProps) {
  const [activeTab, setActiveTab] = useState("ray");

  return (
    <div className={cn("h-full flex flex-col", className)}>
      <div className="flex items-center justify-between p-6 border-b">
        <div>
          <h1 className="text-2xl font-bold">ML Services</h1>
          <p className="text-muted-foreground">
            Configure and monitor Ray, VLM, and RLM services
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <div className="border-b px-6">
          <TabsList className="h-12">
            <TabsTrigger value="ray" className="gap-2">
              <Server className="h-4 w-4" />
              Ray Cluster
            </TabsTrigger>
            <TabsTrigger value="vlm" className="gap-2">
              <Eye className="h-4 w-4" />
              Vision LM
            </TabsTrigger>
            <TabsTrigger value="rlm" className="gap-2">
              <Brain className="h-4 w-4" />
              Recursive LM
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-6">
            <TabsContent value="ray" className="m-0">
              <RayClusterPanel />
            </TabsContent>
            <TabsContent value="vlm" className="m-0">
              <VLMPanel />
            </TabsContent>
            <TabsContent value="rlm" className="m-0">
              <RLMPanel />
            </TabsContent>
          </div>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

export default MLServicesPanel;
