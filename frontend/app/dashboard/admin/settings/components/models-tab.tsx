"use client";

import { useState, useCallback, useEffect } from "react";
import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Cpu, Thermometer, MessageSquare, Clock, Shield, RefreshCw, Loader2, Info,
  ChevronDown, ChevronUp, Layers, Check, X, HardDrive, Settings2, Save, RotateCcw,
} from "lucide-react";
import { toast } from "sonner";
import type { LLMProvider } from "@/lib/api/client";
import {
  useOllamaContextLength,
  useModelContextRecommendations,
  useLLMOperations,
  useSetLLMOperationConfig,
  useDeleteLLMOperationConfig,
} from "@/lib/api/hooks";

interface ModelsTabProps {
  ModelConfigurationSection: React.ComponentType<{ providers: LLMProvider[] }>;
  providers: LLMProvider[];
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function ModelsTab({ ModelConfigurationSection, providers, localSettings, handleSettingChange }: ModelsTabProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showPerModel, setShowPerModel] = useState(true);

  // Auto-detect Ollama model's max context length
  const { data: ollamaContextData, isLoading: contextLoading, refetch: refetchContext } = useOllamaContextLength();

  // Per-model context recommendations
  const { data: ctxRecommendations, isLoading: ctxRecLoading, refetch: refetchCtxRec } = useModelContextRecommendations();

  const maxContextLength = ollamaContextData?.context_length || null;
  const currentContextWindow = (localSettings["llm.context_window"] as number) ?? 4096;
  const currentTemperature = (localSettings["llm.temperature"] as number) ?? 0.7;
  const currentMaxTokens = (localSettings["llm.max_tokens"] as number) ?? 4096;
  const currentCallTimeout = (localSettings["llm.call_timeout"] as number) ?? 120;
  const currentMaxRetries = (localSettings["llm.max_retries"] as number) ?? 3;
  const currentCircuitThreshold = (localSettings["llm.circuit_breaker_threshold"] as number) ?? 5;
  const currentCircuitRecovery = (localSettings["llm.circuit_breaker_recovery"] as number) ?? 60;

  // Per-model context overrides from local settings
  const currentOverrides = (localSettings["llm.model_context_overrides"] as Record<string, number>) ?? {};

  const handleModelContextOverride = (modelName: string, value: number | null) => {
    const newOverrides = { ...currentOverrides };
    if (value === null) {
      delete newOverrides[modelName];
    } else {
      newOverrides[modelName] = value;
    }
    handleSettingChange("llm.model_context_overrides", newOverrides);
  };

  const getSourceBadge = (source: string) => {
    switch (source) {
      case "override":
        return <Badge variant="default" className="text-[10px] px-1.5 py-0">override</Badge>;
      case "recommendation":
        return <Badge variant="secondary" className="text-[10px] px-1.5 py-0">recommended</Badge>;
      default:
        return <Badge variant="outline" className="text-[10px] px-1.5 py-0">global</Badge>;
    }
  };

  return (
    <TabsContent value="models" className="space-y-6">
      <ModelConfigurationSection providers={providers} />

      {/* LLM Inference Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            LLM Inference Settings
          </CardTitle>
          <CardDescription>
            Control how the LLM generates responses — context window, temperature, token limits, and timeouts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Default Context Window (fallback) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                <label className="text-sm font-medium">Default Context Window (num_ctx)</label>
              </div>
              <div className="flex items-center gap-2">
                {contextLoading ? (
                  <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                ) : maxContextLength ? (
                  <Badge variant="outline" className="text-xs">
                    Model max: {maxContextLength.toLocaleString()}
                    {ollamaContextData?.model_name && ` (${ollamaContextData.model_name})`}
                  </Badge>
                ) : (
                  <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => refetchContext()}>
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Detect max
                  </Button>
                )}
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Slider
                value={[currentContextWindow]}
                onValueChange={([value]) => handleSettingChange("llm.context_window", value)}
                min={2048}
                max={maxContextLength || 131072}
                step={1024}
                className="flex-1"
              />
              <Input
                type="number"
                min={2048}
                max={maxContextLength || 131072}
                step={1024}
                value={currentContextWindow}
                onChange={(e) => handleSettingChange("llm.context_window", parseInt(e.target.value) || 4096)}
                className="w-28"
              />
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Info className="h-3 w-3" />
              <span>
                Fallback for models without a specific recommendation or override below.
                {maxContextLength && currentContextWindow > maxContextLength && (
                  <span className="text-destructive ml-1">
                    Warning: exceeds model maximum ({maxContextLength.toLocaleString()})
                  </span>
                )}
              </span>
            </div>
          </div>

          {/* Per-Model Context Windows */}
          <div className="pt-2 border-t">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowPerModel(!showPerModel)}
              className="flex items-center gap-2 text-muted-foreground mb-2"
            >
              <Layers className="h-4 w-4" />
              Per-Model Context Windows
              {ctxRecommendations?.models && (
                <Badge variant="outline" className="text-[10px] ml-1">
                  {ctxRecommendations.models.length} models
                </Badge>
              )}
              {showPerModel ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </Button>

            {showPerModel && (
              <div className="space-y-2">
                {ctxRecLoading ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground py-4 justify-center">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading installed models...
                  </div>
                ) : ctxRecommendations?.models && ctxRecommendations.models.length > 0 ? (
                  <>
                    {/* Header */}
                    <div className="grid grid-cols-[1fr_80px_100px_110px_100px_auto] gap-2 px-2 py-1 text-[11px] text-muted-foreground font-medium uppercase tracking-wider">
                      <span>Model</span>
                      <span>Size</span>
                      <span>Effective</span>
                      <span>Recommended</span>
                      <span>Override</span>
                      <span>VRAM</span>
                    </div>

                    {/* Model rows */}
                    {ctxRecommendations.models.map((model) => {
                      const hasOverride = currentOverrides[model.model_name] != null;
                      const overrideValue = currentOverrides[model.model_name];
                      // Compute effective based on local overrides (may differ from server-computed)
                      const localEffective = hasOverride
                        ? overrideValue
                        : model.recommended ?? currentContextWindow;
                      const localSource = hasOverride
                        ? "override"
                        : model.recommended
                        ? "recommendation"
                        : "global";

                      return (
                        <div
                          key={model.model_name}
                          className="grid grid-cols-[1fr_80px_100px_110px_100px_auto] gap-2 px-2 py-2 rounded-md hover:bg-muted/50 items-center border-b last:border-0"
                        >
                          {/* Model name */}
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="text-sm font-medium truncate">{model.model_name}</span>
                          </div>

                          {/* Parameter size */}
                          <span className="text-xs text-muted-foreground">
                            {model.parameter_size || "—"}
                          </span>

                          {/* Effective context + source badge */}
                          <div className="flex flex-col gap-0.5">
                            <span className="text-sm font-mono">{localEffective?.toLocaleString()}</span>
                            {getSourceBadge(localSource)}
                          </div>

                          {/* Recommended value */}
                          <div className="flex items-center gap-1">
                            {model.recommended ? (
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      className="h-7 px-2 text-xs font-mono"
                                      onClick={() => handleModelContextOverride(model.model_name, model.recommended!)}
                                    >
                                      {model.recommended.toLocaleString()}
                                      <Check className="h-3 w-3 ml-1 text-green-500" />
                                    </Button>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p>Click to use recommended value</p>
                                    {model.max && <p className="text-xs">Max: {model.max.toLocaleString()}</p>}
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            ) : (
                              <span className="text-xs text-muted-foreground">—</span>
                            )}
                          </div>

                          {/* Override input */}
                          <div className="flex items-center gap-1">
                            <Input
                              type="number"
                              min={1024}
                              max={model.max || 131072}
                              step={1024}
                              placeholder={model.recommended ? String(model.recommended) : "—"}
                              value={hasOverride ? overrideValue : ""}
                              onChange={(e) => {
                                const val = parseInt(e.target.value);
                                handleModelContextOverride(model.model_name, isNaN(val) ? null : val);
                              }}
                              className="w-20 h-7 text-xs"
                            />
                            {hasOverride && (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 w-7 p-0"
                                onClick={() => handleModelContextOverride(model.model_name, null)}
                              >
                                <X className="h-3 w-3 text-muted-foreground" />
                              </Button>
                            )}
                          </div>

                          {/* VRAM */}
                          <div className="flex items-center gap-1">
                            {model.vram ? (
                              <span className="text-[11px] text-muted-foreground flex items-center gap-1">
                                <HardDrive className="h-3 w-3" />
                                {model.vram}
                              </span>
                            ) : (
                              <span className="text-xs text-muted-foreground">—</span>
                            )}
                          </div>
                        </div>
                      );
                    })}

                    {/* Refresh button */}
                    <div className="flex justify-end pt-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-xs"
                        onClick={() => refetchCtxRec()}
                      >
                        <RefreshCw className="h-3 w-3 mr-1" />
                        Refresh models
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="text-sm text-muted-foreground py-4 text-center">
                    No Ollama models detected. Make sure Ollama is running.
                    <Button variant="ghost" size="sm" className="ml-2" onClick={() => refetchCtxRec()}>
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Retry
                    </Button>
                  </div>
                )}

                <div className="flex items-center gap-2 text-xs text-muted-foreground px-2">
                  <Info className="h-3 w-3 flex-shrink-0" />
                  <span>
                    Recommended values are research-backed defaults for RAG tasks. Override per model or use the global fallback above.
                    Resolution: override &gt; recommendation &gt; global fallback.
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Temperature */}
          <div className="space-y-3 pt-2 border-t">
            <div className="flex items-center gap-2">
              <Thermometer className="h-4 w-4" />
              <label className="text-sm font-medium">Temperature</label>
              <Badge variant="secondary" className="text-xs">{currentTemperature.toFixed(2)}</Badge>
            </div>
            <Slider
              value={[currentTemperature]}
              onValueChange={([value]) => handleSettingChange("llm.temperature", parseFloat(value.toFixed(2)))}
              min={0}
              max={2}
              step={0.05}
            />
            <p className="text-xs text-muted-foreground">
              Lower = more focused and deterministic. Higher = more creative and varied. 0.7 is a good default for RAG.
            </p>
          </div>

          {/* Max Tokens */}
          <div className="space-y-3 pt-2 border-t">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              <label className="text-sm font-medium">Max Response Tokens</label>
            </div>
            <div className="flex items-center gap-4">
              <Slider
                value={[currentMaxTokens]}
                onValueChange={([value]) => handleSettingChange("llm.max_tokens", value)}
                min={256}
                max={16384}
                step={256}
                className="flex-1"
              />
              <Input
                type="number"
                min={256}
                max={16384}
                step={256}
                value={currentMaxTokens}
                onChange={(e) => handleSettingChange("llm.max_tokens", parseInt(e.target.value) || 4096)}
                className="w-28"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Maximum number of tokens the LLM can generate in a single response.
            </p>
          </div>

          {/* Call Timeout */}
          <div className="space-y-3 pt-2 border-t">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              <label className="text-sm font-medium">Call Timeout (seconds)</label>
            </div>
            <div className="flex items-center gap-4">
              <Slider
                value={[currentCallTimeout]}
                onValueChange={([value]) => handleSettingChange("llm.call_timeout", value)}
                min={30}
                max={300}
                step={10}
                className="flex-1"
              />
              <Input
                type="number"
                min={30}
                max={300}
                step={10}
                value={currentCallTimeout}
                onChange={(e) => handleSettingChange("llm.call_timeout", parseInt(e.target.value) || 120)}
                className="w-28"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              How long to wait for an LLM response before timing out. Increase for large context windows or slow hardware.
            </p>
          </div>

          {/* Advanced: Resilience Settings */}
          <div className="pt-2 border-t">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-muted-foreground"
            >
              <Shield className="h-4 w-4" />
              Resilience Settings
              {showAdvanced ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </Button>

            {showAdvanced && (
              <div className="space-y-4 mt-3 pl-2">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Retries</label>
                    <Input
                      type="number"
                      min={1}
                      max={5}
                      value={currentMaxRetries}
                      onChange={(e) => handleSettingChange("llm.max_retries", parseInt(e.target.value) || 3)}
                    />
                    <p className="text-xs text-muted-foreground">Retry attempts for transient failures (1-5)</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Circuit Breaker Threshold</label>
                    <Input
                      type="number"
                      min={3}
                      max={10}
                      value={currentCircuitThreshold}
                      onChange={(e) => handleSettingChange("llm.circuit_breaker_threshold", parseInt(e.target.value) || 5)}
                    />
                    <p className="text-xs text-muted-foreground">Consecutive failures before circuit opens (3-10)</p>
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Circuit Recovery Time (seconds)</label>
                  <Input
                    type="number"
                    min={30}
                    max={300}
                    value={currentCircuitRecovery}
                    onChange={(e) => handleSettingChange("llm.circuit_breaker_recovery", parseInt(e.target.value) || 60)}
                  />
                  <p className="text-xs text-muted-foreground">Seconds before circuit breaker attempts recovery (30-300)</p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
      {/* Per-Operation Model Configuration */}
      <OperationModelConfigSection providers={providers} />
    </TabsContent>
  );
}

// =============================================================================
// Operation-Specific Model Configuration Section
// =============================================================================

const OPERATION_LABELS: Record<string, { label: string; description: string; defaultTemp: number | null }> = {
  chat: { label: "Chat", description: "Conversational responses", defaultTemp: 0.7 },
  rag: { label: "RAG", description: "Retrieval-augmented generation", defaultTemp: 0.3 },
  doc_generation: { label: "Doc Generation", description: "Document generation tasks", defaultTemp: 0.5 },
  summarization: { label: "Summarization", description: "Text summarization", defaultTemp: 0.3 },
  embeddings: { label: "Embeddings", description: "Vector embeddings", defaultTemp: null },
  agent: { label: "Agent", description: "Autonomous agent tasks", defaultTemp: 0.5 },
};

interface OperationRowState {
  provider_id: string;
  model_override: string;
  temperature_override: string;
  max_tokens_override: string;
  fallback_provider_id: string;
}

function OperationModelConfigSection({ providers }: { providers: LLMProvider[] }) {
  const { data: opsData, isLoading: opsLoading } = useLLMOperations();
  const setOpConfig = useSetLLMOperationConfig();
  const deleteOpConfig = useDeleteLLMOperationConfig();

  // Local state for each operation row (keyed by operation_type)
  const [rowState, setRowState] = useState<Record<string, OperationRowState>>({});
  const [initialized, setInitialized] = useState(false);

  const activeProviders = providers.filter((p) => p.is_active);

  // Build the valid operations list from API or fallback to our known set
  const validOperations = opsData?.valid_operations ?? Object.keys(OPERATION_LABELS);

  // Initialize local row state from fetched data
  useEffect(() => {
    if (!opsData || initialized) return;
    const initial: Record<string, OperationRowState> = {};
    for (const op of validOperations) {
      const existing = opsData.operations.find((o) => o.operation_type === op);
      initial[op] = {
        provider_id: existing?.provider_id ?? "",
        model_override: existing?.model_override ?? "",
        temperature_override: existing?.temperature_override != null ? String(existing.temperature_override) : "",
        max_tokens_override: existing?.max_tokens_override != null ? String(existing.max_tokens_override) : "",
        fallback_provider_id: existing?.fallback_provider_id ?? "",
      };
    }
    setRowState(initial);
    setInitialized(true);
  }, [opsData, validOperations, initialized]);

  // Re-initialize when opsData changes after a mutation
  const reinitRow = useCallback((op: string, data: { provider_id?: string | null; model_override?: string | null; temperature_override?: number | null; max_tokens_override?: number | null; fallback_provider_id?: string | null } | null) => {
    setRowState((prev) => ({
      ...prev,
      [op]: {
        provider_id: data?.provider_id ?? "",
        model_override: data?.model_override ?? "",
        temperature_override: data?.temperature_override != null ? String(data.temperature_override) : "",
        max_tokens_override: data?.max_tokens_override != null ? String(data.max_tokens_override) : "",
        fallback_provider_id: data?.fallback_provider_id ?? "",
      },
    }));
  }, []);

  const updateRow = useCallback((op: string, field: keyof OperationRowState, value: string) => {
    setRowState((prev) => ({
      ...prev,
      [op]: { ...prev[op], [field]: value },
    }));
  }, []);

  const handleSave = useCallback(async (op: string) => {
    const row = rowState[op];
    if (!row) return;

    const data: Record<string, unknown> = {};
    if (row.provider_id) data.provider_id = row.provider_id;
    if (row.model_override) data.model_override = row.model_override;
    if (row.temperature_override !== "") {
      const t = parseFloat(row.temperature_override);
      if (!isNaN(t)) data.temperature_override = t;
    }
    if (row.max_tokens_override !== "") {
      const m = parseInt(row.max_tokens_override);
      if (!isNaN(m)) data.max_tokens_override = m;
    }
    if (row.fallback_provider_id) data.fallback_provider_id = row.fallback_provider_id;

    try {
      const result = await setOpConfig.mutateAsync({ operationType: op, data });
      reinitRow(op, result);
      toast.success(`Saved ${OPERATION_LABELS[op]?.label ?? op} configuration`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      toast.error(`Failed to save: ${msg}`);
    }
  }, [rowState, setOpConfig, reinitRow]);

  const handleReset = useCallback(async (op: string) => {
    try {
      await deleteOpConfig.mutateAsync(op);
      reinitRow(op, null);
      toast.success(`Reset ${OPERATION_LABELS[op]?.label ?? op} to system defaults`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      toast.error(`Failed to reset: ${msg}`);
    }
  }, [deleteOpConfig, reinitRow]);

  const hasExistingConfig = useCallback((op: string) => {
    return opsData?.operations.some((o) => o.operation_type === op) ?? false;
  }, [opsData]);

  const rowIsDirty = useCallback((op: string) => {
    const row = rowState[op];
    if (!row) return false;
    const existing = opsData?.operations.find((o) => o.operation_type === op);
    if (!existing) {
      // No saved config -- dirty if any field is non-empty
      return !!(row.provider_id || row.model_override || row.temperature_override || row.max_tokens_override || row.fallback_provider_id);
    }
    return (
      (row.provider_id || "") !== (existing.provider_id || "") ||
      (row.model_override || "") !== (existing.model_override || "") ||
      (row.temperature_override || "") !== (existing.temperature_override != null ? String(existing.temperature_override) : "") ||
      (row.max_tokens_override || "") !== (existing.max_tokens_override != null ? String(existing.max_tokens_override) : "") ||
      (row.fallback_provider_id || "") !== (existing.fallback_provider_id || "")
    );
  }, [rowState, opsData]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Settings2 className="h-5 w-5" />
          Operation-Specific Model Configuration
        </CardTitle>
        <CardDescription>
          Configure different models for different operations. Overrides global defaults for specific tasks.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {opsLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading operation configurations...
          </div>
        ) : (
          <div className="space-y-1">
            {/* Table Header */}
            <div className="grid grid-cols-[140px_1fr_1fr_100px_100px_1fr_auto] gap-3 px-3 py-2 text-[11px] text-muted-foreground font-medium uppercase tracking-wider border-b">
              <span>Operation</span>
              <span>Provider</span>
              <span>Model</span>
              <span>Temperature</span>
              <span>Max Tokens</span>
              <span>Fallback</span>
              <span>Actions</span>
            </div>

            {/* Operation Rows */}
            {validOperations.map((op) => {
              const meta = OPERATION_LABELS[op] ?? { label: op, description: "", defaultTemp: 0.7 };
              const row = rowState[op];
              if (!row) return null;
              const isConfigured = hasExistingConfig(op);
              const dirty = rowIsDirty(op);
              const saving = setOpConfig.isPending;
              const deleting = deleteOpConfig.isPending;

              return (
                <div
                  key={op}
                  className="grid grid-cols-[140px_1fr_1fr_100px_100px_1fr_auto] gap-3 px-3 py-2.5 rounded-md hover:bg-muted/50 items-center border-b last:border-0"
                >
                  {/* Operation label */}
                  <div className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-medium">{meta.label}</span>
                      {isConfigured && (
                        <Badge variant="default" className="text-[9px] px-1 py-0">custom</Badge>
                      )}
                    </div>
                    {meta.description && (
                      <span className="text-[10px] text-muted-foreground">{meta.description}</span>
                    )}
                  </div>

                  {/* Provider select */}
                  <Select
                    value={row.provider_id || "__default__"}
                    onValueChange={(v) => updateRow(op, "provider_id", v === "__default__" ? "" : v)}
                  >
                    <SelectTrigger className="h-8 text-xs">
                      <SelectValue placeholder="System Default" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__default__">System Default</SelectItem>
                      {activeProviders.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}{p.is_default ? " (default)" : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  {/* Model override input */}
                  <Input
                    type="text"
                    placeholder="Provider default"
                    value={row.model_override}
                    onChange={(e) => updateRow(op, "model_override", e.target.value)}
                    className="h-8 text-xs"
                  />

                  {/* Temperature input */}
                  {meta.defaultTemp !== null ? (
                    <Input
                      type="number"
                      min={0}
                      max={2}
                      step={0.05}
                      placeholder={String(meta.defaultTemp)}
                      value={row.temperature_override}
                      onChange={(e) => updateRow(op, "temperature_override", e.target.value)}
                      className="h-8 text-xs"
                    />
                  ) : (
                    <div className="flex items-center justify-center text-xs text-muted-foreground">
                      --
                    </div>
                  )}

                  {/* Max tokens input */}
                  <Input
                    type="number"
                    min={256}
                    max={16384}
                    step={256}
                    placeholder="Default"
                    value={row.max_tokens_override}
                    onChange={(e) => updateRow(op, "max_tokens_override", e.target.value)}
                    className="h-8 text-xs"
                  />

                  {/* Fallback provider select */}
                  <Select
                    value={row.fallback_provider_id || "__none__"}
                    onValueChange={(v) => updateRow(op, "fallback_provider_id", v === "__none__" ? "" : v)}
                  >
                    <SelectTrigger className="h-8 text-xs">
                      <SelectValue placeholder="None" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__none__">None</SelectItem>
                      {activeProviders.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  {/* Actions */}
                  <div className="flex items-center gap-1">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant={dirty ? "default" : "outline"}
                            size="sm"
                            className="h-7 w-7 p-0"
                            disabled={!dirty || saving}
                            onClick={() => handleSave(op)}
                          >
                            {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Save override</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 w-7 p-0"
                            disabled={!isConfigured || deleting}
                            onClick={() => handleReset(op)}
                          >
                            {deleting ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCcw className="h-3 w-3" />}
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Reset to system default</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                </div>
              );
            })}

            {/* Info footer */}
            <div className="flex items-center gap-2 text-xs text-muted-foreground px-3 pt-3">
              <Info className="h-3 w-3 flex-shrink-0" />
              <span>
                Overrides apply to specific operation types. Leave fields empty to use system defaults.
                Use &quot;Reset&quot; to remove all overrides for an operation.
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
