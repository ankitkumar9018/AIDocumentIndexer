"use client";

import { useState } from "react";
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
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Cpu,
  Server,
  Cloud,
  Laptop,
  Database,
  Zap,
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Eye,
  EyeOff,
} from "lucide-react";
import {
  useInfrastructureStatus,
  useInfrastructureProfiles,
  useTestInfrastructure,
  useApplyInfrastructureProfile,
} from "@/lib/api";

interface InfrastructureTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

const PROFILE_ICONS: Record<string, React.ReactNode> = {
  laptop: <Laptop className="h-5 w-5" />,
  server: <Server className="h-5 w-5" />,
  cloud: <Cloud className="h-5 w-5" />,
};

function StatusDot({ connected, latency }: { connected: boolean; latency?: number | null }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm">
      <span
        className={`h-2 w-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
      />
      <span className={connected ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>
        {connected ? "Connected" : "Disconnected"}
      </span>
      {latency != null && connected && (
        <span className="text-muted-foreground">({latency}ms)</span>
      )}
    </span>
  );
}

export function InfrastructureTab({ localSettings, handleSettingChange }: InfrastructureTabProps) {
  const [testResults, setTestResults] = useState<Record<string, { success: boolean; message: string; latency_ms?: number }>>({});
  const [testing, setTesting] = useState<string | null>(null);
  const [showPasswords, setShowPasswords] = useState<Record<string, boolean>>({});

  const { data: infraStatus, isLoading: statusLoading } = useInfrastructureStatus();
  const { data: profilesData } = useInfrastructureProfiles();
  const testInfra = useTestInfrastructure();
  const applyProfile = useApplyInfrastructureProfile();

  const vsBackend = (localSettings["vector_store.backend"] as string) || "pgvector";
  const llmBackend = (localSettings["llm.inference_backend"] as string) || "ollama";
  const activeProfile = infraStatus?.active_profile || "development";

  const handleTest = async (backend: string, config: Record<string, unknown>) => {
    setTesting(backend);
    try {
      const result = await testInfra.mutateAsync({ backend, config });
      setTestResults(prev => ({ ...prev, [backend]: result }));
    } catch {
      setTestResults(prev => ({
        ...prev,
        [backend]: { success: false, message: "Request failed" },
      }));
    }
    setTesting(null);
  };

  const handleApplyProfile = async (profileId: string) => {
    if (!confirm(
      `Apply "${profileId}" profile?\n\nThis will change your vector store, LLM, and queue settings. ` +
      `If the vector store backend changes, you will need to re-index all documents.`
    )) return;
    try {
      const result = await applyProfile.mutateAsync(profileId);
      // Update local settings to reflect the applied profile
      if (result.applied_settings) {
        for (const [key, value] of Object.entries(result.applied_settings)) {
          handleSettingChange(key, value);
        }
      }
    } catch (err) {
      console.error("Failed to apply profile:", err);
    }
  };

  const togglePassword = (key: string) => {
    setShowPasswords(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <TabsContent value="infrastructure" className="space-y-6">
      {/* Section 1: Scaling Profiles */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Scaling Profiles
          </CardTitle>
          <CardDescription>
            One-click profiles to configure your infrastructure for different scales.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {(profilesData?.profiles || []).map((profile) => (
              <div
                key={profile.id}
                className={`relative rounded-lg border-2 p-4 transition-colors ${
                  activeProfile === profile.id
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-muted-foreground/30"
                }`}
              >
                {activeProfile === profile.id && (
                  <Badge className="absolute -top-2 right-2" variant="default">Active</Badge>
                )}
                <div className="flex items-center gap-2 mb-2">
                  {PROFILE_ICONS[profile.icon] || <Server className="h-5 w-5" />}
                  <h4 className="font-semibold">{profile.name}</h4>
                </div>
                <p className="text-sm text-muted-foreground mb-3">{profile.description}</p>
                <div className="text-xs text-muted-foreground mb-3 space-y-1">
                  {Object.entries(profile.settings).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span>{key.split('.').pop()}</span>
                      <span className="font-mono">{String(value)}</span>
                    </div>
                  ))}
                </div>
                <Button
                  variant={activeProfile === profile.id ? "outline" : "default"}
                  size="sm"
                  className="w-full"
                  disabled={activeProfile === profile.id || applyProfile.isPending}
                  onClick={() => handleApplyProfile(profile.id)}
                >
                  {applyProfile.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1" />
                  ) : null}
                  {activeProfile === profile.id ? "Active" : "Apply"}
                </Button>
              </div>
            ))}
          </div>
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Switching profiles changes vector store, LLM, and queue settings. If the vector store backend changes, you must re-index all documents.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      {/* Section 2: Vector Store Backend */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Database className="h-5 w-5" />
                Vector Store Backend
              </CardTitle>
              <CardDescription>
                Where document embeddings are stored and searched.
              </CardDescription>
            </div>
            {infraStatus && (
              <StatusDot
                connected={infraStatus.vector_store.connected}
                latency={infraStatus.vector_store.latency_ms}
              />
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Backend</Label>
            <Select
              value={vsBackend}
              onValueChange={(v) => handleSettingChange("vector_store.backend", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pgvector">pgvector (PostgreSQL, default)</SelectItem>
                <SelectItem value="qdrant">Qdrant (1-50M docs, Rust-based)</SelectItem>
                <SelectItem value="milvus">Milvus (50M+ docs, distributed)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {vsBackend === "pgvector" && (
            <p className="text-sm text-muted-foreground">Using your existing PostgreSQL database. No additional configuration needed.</p>
          )}

          {vsBackend === "qdrant" && (
            <div className="space-y-3 pl-2 border-l-2 border-primary/20">
              <div className="space-y-1">
                <Label>Qdrant URL</Label>
                <Input
                  value={(localSettings["vector_store.qdrant_url"] as string) || "localhost:6333"}
                  onChange={(e) => handleSettingChange("vector_store.qdrant_url", e.target.value)}
                  placeholder="localhost:6333"
                />
              </div>
              <div className="space-y-1">
                <Label>API Key (optional, for Qdrant Cloud)</Label>
                <div className="flex gap-2">
                  <Input
                    type={showPasswords["qdrant_api_key"] ? "text" : "password"}
                    value={(localSettings["vector_store.qdrant_api_key"] as string) || ""}
                    onChange={(e) => handleSettingChange("vector_store.qdrant_api_key", e.target.value)}
                    placeholder="Leave empty for local"
                  />
                  <Button variant="ghost" size="icon" onClick={() => togglePassword("qdrant_api_key")}>
                    {showPasswords["qdrant_api_key"] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
              <div className="space-y-1">
                <Label>Collection Name</Label>
                <Input
                  value={(localSettings["vector_store.qdrant_collection"] as string) || "documents"}
                  onChange={(e) => handleSettingChange("vector_store.qdrant_collection", e.target.value)}
                  placeholder="documents"
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={testing === "qdrant"}
                onClick={() => handleTest("qdrant", {
                  url: (localSettings["vector_store.qdrant_url"] as string) || "localhost:6333",
                  api_key: (localSettings["vector_store.qdrant_api_key"] as string) || "",
                })}
              >
                {testing === "qdrant" ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                Test Connection
              </Button>
              {testResults["qdrant"] && (
                <div className={`flex items-center gap-2 text-sm ${testResults["qdrant"].success ? "text-green-600" : "text-red-600"}`}>
                  {testResults["qdrant"].success ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                  {testResults["qdrant"].message}
                  {testResults["qdrant"].latency_ms && ` (${testResults["qdrant"].latency_ms}ms)`}
                </div>
              )}
            </div>
          )}

          {vsBackend === "milvus" && (
            <div className="space-y-3 pl-2 border-l-2 border-primary/20">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>Host</Label>
                  <Input
                    value={(localSettings["vector_store.milvus_host"] as string) || "localhost"}
                    onChange={(e) => handleSettingChange("vector_store.milvus_host", e.target.value)}
                    placeholder="localhost"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Port</Label>
                  <Input
                    type="number"
                    value={(localSettings["vector_store.milvus_port"] as number) || 19530}
                    onChange={(e) => handleSettingChange("vector_store.milvus_port", parseInt(e.target.value) || 19530)}
                    placeholder="19530"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <Label>Collection Name</Label>
                <Input
                  value={(localSettings["vector_store.milvus_collection"] as string) || "documents"}
                  onChange={(e) => handleSettingChange("vector_store.milvus_collection", e.target.value)}
                  placeholder="documents"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>Username (optional)</Label>
                  <Input
                    value={(localSettings["vector_store.milvus_user"] as string) || ""}
                    onChange={(e) => handleSettingChange("vector_store.milvus_user", e.target.value)}
                    placeholder="Optional"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Password (optional)</Label>
                  <Input
                    type={showPasswords["milvus_password"] ? "text" : "password"}
                    value={(localSettings["vector_store.milvus_password"] as string) || ""}
                    onChange={(e) => handleSettingChange("vector_store.milvus_password", e.target.value)}
                    placeholder="Optional"
                  />
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={testing === "milvus"}
                onClick={() => handleTest("milvus", {
                  host: (localSettings["vector_store.milvus_host"] as string) || "localhost",
                  port: (localSettings["vector_store.milvus_port"] as number) || 19530,
                })}
              >
                {testing === "milvus" ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                Test Connection
              </Button>
              {testResults["milvus"] && (
                <div className={`flex items-center gap-2 text-sm ${testResults["milvus"].success ? "text-green-600" : "text-red-600"}`}>
                  {testResults["milvus"].success ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                  {testResults["milvus"].message}
                  {testResults["milvus"].latency_ms && ` (${testResults["milvus"].latency_ms}ms)`}
                </div>
              )}
            </div>
          )}

          {vsBackend !== "pgvector" && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                Switching vector stores requires re-indexing all documents. Your existing pgvector data will remain intact.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Section 3: LLM Inference Backend */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                LLM Inference Backend
              </CardTitle>
              <CardDescription>
                Where language model inference runs.
              </CardDescription>
            </div>
            {infraStatus && (
              <StatusDot
                connected={infraStatus.llm_inference.connected}
                latency={infraStatus.llm_inference.latency_ms}
              />
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Backend</Label>
            <Select
              value={llmBackend}
              onValueChange={(v) => handleSettingChange("llm.inference_backend", v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto (use default provider)</SelectItem>
                <SelectItem value="ollama">Ollama (local, free)</SelectItem>
                <SelectItem value="vllm">vLLM (2-4x faster batch inference)</SelectItem>
                <SelectItem value="openai">OpenAI (cloud)</SelectItem>
                <SelectItem value="anthropic">Anthropic (cloud)</SelectItem>
                <SelectItem value="groq">Groq (fast cloud inference)</SelectItem>
                <SelectItem value="together">Together AI (cloud)</SelectItem>
                <SelectItem value="deepinfra">DeepInfra (cloud)</SelectItem>
                <SelectItem value="bedrock">AWS Bedrock (cloud)</SelectItem>
                <SelectItem value="google">Google AI (cloud)</SelectItem>
                <SelectItem value="cohere">Cohere (cloud)</SelectItem>
                <SelectItem value="custom">Custom (OpenAI-compatible)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {llmBackend === "vllm" && (
            <div className="space-y-3 pl-2 border-l-2 border-primary/20">
              <div className="space-y-1">
                <Label>API Base URL</Label>
                <Input
                  value={(localSettings["llm.vllm_api_base"] as string) || "http://localhost:8000/v1"}
                  onChange={(e) => handleSettingChange("llm.vllm_api_base", e.target.value)}
                  placeholder="http://localhost:8000/v1"
                />
              </div>
              <div className="space-y-1">
                <Label>API Key</Label>
                <div className="flex gap-2">
                  <Input
                    type={showPasswords["vllm_api_key"] ? "text" : "password"}
                    value={(localSettings["llm.vllm_api_key"] as string) || "dummy"}
                    onChange={(e) => handleSettingChange("llm.vllm_api_key", e.target.value)}
                    placeholder="dummy (for local)"
                  />
                  <Button variant="ghost" size="icon" onClick={() => togglePassword("vllm_api_key")}>
                    {showPasswords["vllm_api_key"] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
              <div className="space-y-1">
                <Label>Model Name</Label>
                <Input
                  value={(localSettings["llm.vllm_model"] as string) || ""}
                  onChange={(e) => handleSettingChange("llm.vllm_model", e.target.value)}
                  placeholder="e.g. meta-llama/Meta-Llama-3-8B-Instruct"
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={testing === "vllm"}
                onClick={() => handleTest("vllm", {
                  api_base: (localSettings["llm.vllm_api_base"] as string) || "http://localhost:8000/v1",
                  api_key: (localSettings["llm.vllm_api_key"] as string) || "dummy",
                })}
              >
                {testing === "vllm" ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                Test Connection
              </Button>
              {testResults["vllm"] && (
                <div className={`flex items-center gap-2 text-sm ${testResults["vllm"].success ? "text-green-600" : "text-red-600"}`}>
                  {testResults["vllm"].success ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                  {testResults["vllm"].message}
                  {testResults["vllm"].latency_ms && ` (${testResults["vllm"].latency_ms}ms)`}
                </div>
              )}
            </div>
          )}

          {llmBackend === "auto" && (
            <p className="text-sm text-muted-foreground">
              Auto mode uses the default provider configured in the Providers tab.
              Add providers and mark one as default to get started. The system auto-detects
              the provider type, API keys, and model settings from your provider configuration.
            </p>
          )}

          {llmBackend && !["auto", "vllm"].includes(llmBackend) && (
            <p className="text-sm text-muted-foreground">
              {llmBackend === "ollama"
                ? "Configured via Ollama. Manage models in the Providers tab."
                : `Using ${llmBackend} cloud API. Configure API keys in the Providers tab.`}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Section 4: Redis */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Redis
              </CardTitle>
              <CardDescription>
                Caching, session storage, and task queue.
              </CardDescription>
            </div>
            {infraStatus && (
              <StatusDot
                connected={infraStatus.redis.connected}
                latency={infraStatus.redis.latency_ms}
              />
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1">
            <Label>Redis URL</Label>
            <Input
              value={(localSettings["infrastructure.redis_url"] as string) || "redis://localhost:6379/0"}
              onChange={(e) => handleSettingChange("infrastructure.redis_url", e.target.value)}
              placeholder="redis://localhost:6379/0"
            />
          </div>
          <div className="space-y-1">
            <Label>Password (optional)</Label>
            <div className="flex gap-2">
              <Input
                type={showPasswords["redis_password"] ? "text" : "password"}
                value={(localSettings["infrastructure.redis_password"] as string) || ""}
                onChange={(e) => handleSettingChange("infrastructure.redis_password", e.target.value)}
                placeholder="Leave empty if no auth"
              />
              <Button variant="ghost" size="icon" onClick={() => togglePassword("redis_password")}>
                {showPasswords["redis_password"] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            disabled={testing === "redis"}
            onClick={() => handleTest("redis", {
              url: (localSettings["infrastructure.redis_url"] as string) || "redis://localhost:6379/0",
              password: (localSettings["infrastructure.redis_password"] as string) || "",
            })}
          >
            {testing === "redis" ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
            Test Connection
          </Button>
          {testResults["redis"] && (
            <div className={`flex items-center gap-2 text-sm ${testResults["redis"].success ? "text-green-600" : "text-red-600"}`}>
              {testResults["redis"].success ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
              {testResults["redis"].message}
              {testResults["redis"].latency_ms && ` (${testResults["redis"].latency_ms}ms)`}
            </div>
          )}
          {!infraStatus?.redis.connected && (
            <p className="text-sm text-muted-foreground">
              Redis is optional. Without it, the system uses in-memory caching (data lost on restart).
            </p>
          )}
        </CardContent>
      </Card>

      {/* Section 5: Infrastructure Status Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Server className="h-5 w-5" />
            Current Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          {statusLoading ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading status...
            </div>
          ) : infraStatus ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="text-sm font-medium">Vector Store</p>
                  <p className="text-xs text-muted-foreground font-mono">{infraStatus.vector_store.backend}</p>
                </div>
                <StatusDot connected={infraStatus.vector_store.connected} latency={infraStatus.vector_store.latency_ms} />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="text-sm font-medium">LLM Inference</p>
                  <p className="text-xs text-muted-foreground font-mono">{infraStatus.llm_inference.backend}</p>
                </div>
                <StatusDot connected={infraStatus.llm_inference.connected} latency={infraStatus.llm_inference.latency_ms} />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="text-sm font-medium">Redis</p>
                  <p className="text-xs text-muted-foreground font-mono">{infraStatus.redis.connected ? "active" : "inactive"}</p>
                </div>
                <StatusDot connected={infraStatus.redis.connected} latency={infraStatus.redis.latency_ms} />
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Unable to load infrastructure status.</p>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  );
}
