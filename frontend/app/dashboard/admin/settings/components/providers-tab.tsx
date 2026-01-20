"use client";

import { getErrorMessage } from "@/lib/errors";
import {
  Bot,
  Plus,
  Star,
  Edit2,
  TestTube,
  Trash2,
  Loader2,
  Eye,
  EyeOff,
  Save,
  HardDrive,
  RefreshCw,
  Download,
  MessageSquare,
  Database,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";
import type { LLMProvider, LLMProviderType } from "@/lib/api/client";

interface OllamaModel {
  name: string;
  parameter_size?: string;
  family?: string;
  size?: number;
}

interface OllamaModelsData {
  chat_models?: OllamaModel[];
  embedding_models?: OllamaModel[];
}

interface OllamaLocalModels {
  success: boolean;
  chat_models?: OllamaModel[];
  embedding_models?: OllamaModel[];
  error?: string;
}

interface ProviderTestResult {
  success: boolean;
  message?: string;
  error?: string;
}

interface NewProviderForm {
  name: string;
  provider_type: string;
  api_key: string;
  api_base_url: string;
  organization_id: string;
  default_chat_model: string;
  default_embedding_model: string;
  is_default: boolean;
}

interface PullResult {
  success: boolean;
  message: string;
}

interface ProviderTypesData {
  provider_types?: Record<string, LLMProviderType>;
}

interface ProvidersData {
  providers?: LLMProvider[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MutationHook = any;

export interface ProvidersTabProps {
  // Provider data and loading states
  providersLoading: boolean;
  providersData: ProvidersData | undefined;
  refetchProviders: () => void;

  // Provider types
  providerTypesData: ProviderTypesData | undefined;

  // Ollama models from registry
  ollamaModelsData: OllamaModelsData | undefined;

  // Ollama local models
  ollamaLocalModels: OllamaLocalModels | undefined;
  ollamaLocalModelsLoading: boolean;
  refetchOllamaModels: () => void;

  // Add provider form state
  showAddProvider: boolean;
  setShowAddProvider: (show: boolean) => void;
  newProvider: NewProviderForm;
  setNewProvider: (provider: NewProviderForm) => void;

  // API key visibility
  showApiKey: boolean;
  setShowApiKey: (show: boolean) => void;

  // Provider test results
  providerTestResults: Record<string, ProviderTestResult>;
  setProviderTestResults: (results: Record<string, ProviderTestResult>) => void;

  // Editing provider
  editingProvider: LLMProvider | null;
  setEditingProvider: (provider: LLMProvider | null) => void;

  // Ollama base URL
  ollamaBaseUrl: string;
  setOllamaBaseUrl: (url: string) => void;

  // New model pulling
  newModelName: string;
  setNewModelName: (name: string) => void;
  pullingModel: boolean;
  setPullingModel: (pulling: boolean) => void;
  pullResult: PullResult | null;
  setPullResult: (result: PullResult | null) => void;

  // Model deletion
  deletingModel: string | null;
  setDeletingModel: (model: string | null) => void;

  // Mutation hooks
  testProvider: MutationHook;
  createProvider: MutationHook;
  updateProvider: MutationHook;
  deleteProvider: MutationHook;
  setDefaultProvider: MutationHook;
  pullOllamaModel: MutationHook;
  deleteOllamaModel: MutationHook;

  // Handler functions
  handleEditProvider: (provider: LLMProvider) => void;
  handleTestProvider: (providerId: string) => void;
  handleSetDefaultProvider: (providerId: string) => void;
  handleDeleteProvider: (providerId: string) => void;
  handleSaveProvider: () => void;
  handleCancelProviderForm: () => void;

  // Helper function
  getProviderTypeConfig: (type: string) => LLMProviderType | undefined;
}

export function ProvidersTab({
  providersLoading,
  providersData,
  providerTypesData,
  ollamaModelsData,
  ollamaLocalModels,
  ollamaLocalModelsLoading,
  refetchOllamaModels,
  showAddProvider,
  setShowAddProvider,
  newProvider,
  setNewProvider,
  showApiKey,
  setShowApiKey,
  providerTestResults,
  editingProvider,
  ollamaBaseUrl,
  setOllamaBaseUrl,
  newModelName,
  setNewModelName,
  pullingModel,
  setPullingModel,
  pullResult,
  setPullResult,
  deletingModel,
  setDeletingModel,
  testProvider,
  createProvider,
  updateProvider,
  deleteProvider,
  setDefaultProvider,
  pullOllamaModel,
  deleteOllamaModel,
  handleEditProvider,
  handleTestProvider,
  handleSetDefaultProvider,
  handleDeleteProvider,
  handleSaveProvider,
  handleCancelProviderForm,
  getProviderTypeConfig,
}: ProvidersTabProps) {
  return (
    <TabsContent value="providers" className="space-y-6">
      {/* LLM Providers Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <Bot className="h-5 w-5" />
                LLM Providers
              </CardTitle>
              <CardDescription>Manage AI model providers and connections</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAddProvider(!showAddProvider)}
            >
              <Plus className="h-4 w-4 mr-1" />
              Add Provider
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Provider List */}
          {providersLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : providersData?.providers && providersData.providers.length > 0 ? (
            <div className="space-y-2">
              {providersData.providers.map((provider) => (
                <div
                  key={provider.id}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex flex-col">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{provider.name}</span>
                        {provider.is_default && (
                          <Badge variant="default" className="text-xs">
                            <Star className="h-3 w-3 mr-1" />
                            Default
                          </Badge>
                        )}
                        {!provider.is_active && (
                          <Badge variant="secondary" className="text-xs">Inactive</Badge>
                        )}
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {getProviderTypeConfig(provider.provider_type)?.name || provider.provider_type}
                        {provider.default_chat_model && ` - ${provider.default_chat_model}`}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {providerTestResults[provider.id] && (
                      <span className={`text-xs ${providerTestResults[provider.id].success ? "text-green-600" : "text-red-600"}`}>
                        {providerTestResults[provider.id].message || providerTestResults[provider.id].error}
                      </span>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleEditProvider(provider)}
                      title="Edit provider"
                    >
                      <Edit2 className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleTestProvider(provider.id)}
                      disabled={testProvider.isPending}
                    >
                      <TestTube className="h-4 w-4" />
                    </Button>
                    {!provider.is_default && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleSetDefaultProvider(provider.id)}
                        disabled={setDefaultProvider.isPending}
                        title="Set as default"
                      >
                        <Star className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteProvider(provider.id)}
                      disabled={deleteProvider.isPending || provider.is_default}
                      className="text-red-500 hover:text-red-600"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">
              No providers configured. Add a provider to get started.
            </p>
          )}

          {/* Add/Edit Provider Form */}
          {showAddProvider && (
            <div className="p-4 rounded-lg border bg-muted/50 space-y-4">
              <h4 className="font-medium">{editingProvider ? "Edit Provider" : "Add New Provider"}</h4>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Display Name</label>
                  <Input
                    placeholder="My OpenAI Account"
                    value={newProvider.name}
                    onChange={(e) => setNewProvider({ ...newProvider, name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Provider Type</label>
                  <select
                    className="w-full h-10 px-3 rounded-md border bg-background disabled:opacity-50"
                    value={newProvider.provider_type}
                    disabled={!!editingProvider}
                    onChange={(e) => {
                      const type = e.target.value;
                      const config = getProviderTypeConfig(type);
                      setNewProvider({
                        ...newProvider,
                        provider_type: type,
                        default_chat_model: config?.default_chat_model || "",
                        default_embedding_model: config?.default_embedding_model || "",
                        api_base_url: config?.default_api_base_url || "",
                      });
                    }}
                  >
                    {providerTypesData?.provider_types &&
                      Object.entries(providerTypesData.provider_types).map(([key, config]) => (
                        <option key={key} value={key}>
                          {config.name}
                        </option>
                      ))}
                  </select>
                </div>
              </div>
              {getProviderTypeConfig(newProvider.provider_type)?.fields.includes("api_key") && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    API Key
                    {editingProvider && (
                      <span className="text-muted-foreground font-normal ml-1">(leave empty to keep current)</span>
                    )}
                  </label>
                  <div className="relative">
                    <Input
                      type={showApiKey ? "text" : "password"}
                      placeholder={editingProvider ? "----------------" : "sk-..."}
                      value={newProvider.api_key}
                      onChange={(e) => setNewProvider({ ...newProvider, api_key: e.target.value })}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-1 top-1 h-8 w-8 p-0"
                      onClick={() => setShowApiKey(!showApiKey)}
                    >
                      {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
              )}
              {getProviderTypeConfig(newProvider.provider_type)?.fields.includes("api_base_url") && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">API Base URL</label>
                  <Input
                    placeholder={getProviderTypeConfig(newProvider.provider_type)?.default_api_base_url || "https://api.example.com"}
                    value={newProvider.api_base_url}
                    onChange={(e) => setNewProvider({ ...newProvider, api_base_url: e.target.value })}
                  />
                </div>
              )}
              {getProviderTypeConfig(newProvider.provider_type)?.fields.includes("organization_id") && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">Organization ID (optional)</label>
                  <Input
                    placeholder="org-..."
                    value={newProvider.organization_id}
                    onChange={(e) => setNewProvider({ ...newProvider, organization_id: e.target.value })}
                  />
                </div>
              )}
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Chat Model</label>
                  <select
                    className="w-full h-10 px-3 rounded-md border bg-background"
                    value={newProvider.default_chat_model}
                    onChange={(e) => setNewProvider({ ...newProvider, default_chat_model: e.target.value })}
                  >
                    <option value="">Select model...</option>
                    {newProvider.provider_type === "ollama" && ollamaModelsData?.chat_models ? (
                      ollamaModelsData.chat_models.map((model) => (
                        <option key={model.name} value={model.name}>
                          {model.name} {model.parameter_size && `(${model.parameter_size})`}
                        </option>
                      ))
                    ) : (
                      getProviderTypeConfig(newProvider.provider_type)?.chat_models.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))
                    )}
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Embedding Model</label>
                  <select
                    className="w-full h-10 px-3 rounded-md border bg-background"
                    value={newProvider.default_embedding_model}
                    onChange={(e) => setNewProvider({ ...newProvider, default_embedding_model: e.target.value })}
                  >
                    <option value="">Select model...</option>
                    {newProvider.provider_type === "ollama" && ollamaModelsData?.embedding_models ? (
                      ollamaModelsData.embedding_models.map((model) => (
                        <option key={model.name} value={model.name}>
                          {model.name} {model.parameter_size && `(${model.parameter_size})`}
                        </option>
                      ))
                    ) : (
                      getProviderTypeConfig(newProvider.provider_type)?.embedding_models.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))
                    )}
                  </select>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is_default_provider"
                  checked={newProvider.is_default}
                  onChange={(e) => setNewProvider({ ...newProvider, is_default: e.target.checked })}
                />
                <label htmlFor="is_default_provider" className="text-sm">
                  Set as default provider
                </label>
              </div>
              <div className="flex gap-2">
                <Button
                  onClick={handleSaveProvider}
                  disabled={!newProvider.name || createProvider.isPending || updateProvider.isPending}
                >
                  {(createProvider.isPending || updateProvider.isPending) ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : editingProvider ? (
                    <Save className="h-4 w-4 mr-2" />
                  ) : (
                    <Plus className="h-4 w-4 mr-2" />
                  )}
                  {editingProvider ? "Update Provider" : "Add Provider"}
                </Button>
                <Button variant="outline" onClick={handleCancelProviderForm}>
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Ollama Local Models Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <HardDrive className="h-5 w-5" />
                Ollama Local Models
              </CardTitle>
              <CardDescription>Manage locally installed Ollama models</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetchOllamaModels()}
              disabled={ollamaLocalModelsLoading}
            >
              {ollamaLocalModelsLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Ollama Base URL */}
          <div className="space-y-2">
            <Label>Ollama Base URL</Label>
            <div className="flex items-center gap-2">
              <Input
                placeholder="http://localhost:11434"
                value={ollamaBaseUrl}
                onChange={(e) => setOllamaBaseUrl(e.target.value)}
                className="flex-1"
              />
            </div>
          </div>

          {/* Pull New Model */}
          <div className="p-4 rounded-lg border bg-muted/30 space-y-3">
            <Label>Pull New Model</Label>
            <div className="flex items-center gap-2">
              <Input
                placeholder="Model name (e.g., qwen2.5vl, llava:7b, nomic-embed-text)"
                value={newModelName}
                onChange={(e) => setNewModelName(e.target.value)}
                className="flex-1"
              />
              <Button
                onClick={async () => {
                  if (!newModelName.trim()) return;
                  setPullingModel(true);
                  setPullResult(null);
                  try {
                    const result = await pullOllamaModel.mutateAsync({
                      modelName: newModelName.trim(),
                      baseUrl: ollamaBaseUrl,
                    });
                    if (result.success) {
                      setPullResult({ success: true, message: result.message || "Model pulled successfully" });
                      setNewModelName("");
                      refetchOllamaModels();
                    } else {
                      setPullResult({ success: false, message: result.error || "Pull failed" });
                    }
                  } catch (err) {
                    setPullResult({ success: false, message: getErrorMessage(err, "Pull failed") });
                  } finally {
                    setPullingModel(false);
                  }
                }}
                disabled={!newModelName.trim() || pullingModel}
              >
                {pullingModel ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Download className="h-4 w-4 mr-2" />
                )}
                Pull Model
              </Button>
            </div>
            {pullResult && (
              <p className={`text-sm ${pullResult.success ? "text-green-600" : "text-red-600"}`}>
                {pullResult.message}
              </p>
            )}
            {/* Recommended Models */}
            <div className="pt-2">
              <p className="text-xs text-muted-foreground mb-2">Quick select recommended models:</p>
              <div className="flex flex-wrap gap-2">
                {[
                  { name: "qwen2.5vl", desc: "Vision (Best DocVQA)" },
                  { name: "llava", desc: "Vision (General)" },
                  { name: "llama3.2", desc: "Fast Chat" },
                  { name: "nomic-embed-text", desc: "Embeddings" },
                ].map((rec) => (
                  <Button
                    key={rec.name}
                    variant="outline"
                    size="sm"
                    onClick={() => setNewModelName(rec.name)}
                    className="text-xs h-7"
                  >
                    {rec.name}
                    <span className="text-muted-foreground ml-1">({rec.desc})</span>
                  </Button>
                ))}
              </div>
            </div>
          </div>

          {/* Model List */}
          {ollamaLocalModelsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : ollamaLocalModels?.success ? (
            <div className="space-y-4">
              {/* Chat Models */}
              {ollamaLocalModels.chat_models && ollamaLocalModels.chat_models.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />
                    Chat Models ({ollamaLocalModels.chat_models.length})
                  </h4>
                  <div className="space-y-2">
                    {ollamaLocalModels.chat_models.map((model: OllamaModel) => (
                      <div
                        key={model.name}
                        className="flex items-center justify-between p-3 rounded-lg border"
                      >
                        <div className="flex items-center gap-3">
                          <Bot className="h-4 w-4 text-primary" />
                          <div>
                            <p className="font-medium">{model.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {model.parameter_size && `${model.parameter_size} - `}
                              {model.family && `${model.family} - `}
                              {model.size && `${(model.size / 1024 / 1024 / 1024).toFixed(1)} GB`}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {(model.name.includes("llava") || model.name.includes("vision") ||
                            model.name.includes("qwen") && model.name.includes("vl") || model.name.includes("moondream")) && (
                            <Badge variant="secondary" className="text-xs">
                              <Eye className="h-3 w-3 mr-1" />
                              Vision
                            </Badge>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={async () => {
                              if (!confirm(`Delete model "${model.name}"? This will free up disk space.`)) return;
                              setDeletingModel(model.name);
                              try {
                                const result = await deleteOllamaModel.mutateAsync({
                                  modelName: model.name,
                                  baseUrl: ollamaBaseUrl,
                                });
                                if (result.success) {
                                  refetchOllamaModels();
                                } else {
                                  alert(result.error || "Delete failed");
                                }
                              } catch (err) {
                                alert(getErrorMessage(err, "Delete failed"));
                              } finally {
                                setDeletingModel(null);
                              }
                            }}
                            disabled={deletingModel === model.name}
                            className="text-red-500 hover:text-red-600"
                          >
                            {deletingModel === model.name ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Embedding Models */}
              {ollamaLocalModels.embedding_models && ollamaLocalModels.embedding_models.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Database className="h-4 w-4" />
                    Embedding Models ({ollamaLocalModels.embedding_models.length})
                  </h4>
                  <div className="space-y-2">
                    {ollamaLocalModels.embedding_models.map((model: OllamaModel) => (
                      <div
                        key={model.name}
                        className="flex items-center justify-between p-3 rounded-lg border"
                      >
                        <div className="flex items-center gap-3">
                          <Database className="h-4 w-4 text-blue-500" />
                          <div>
                            <p className="font-medium">{model.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {model.parameter_size && `${model.parameter_size} - `}
                              {model.size && `${(model.size / 1024 / 1024 / 1024).toFixed(1)} GB`}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={async () => {
                            if (!confirm(`Delete model "${model.name}"? This will free up disk space.`)) return;
                            setDeletingModel(model.name);
                            try {
                              const result = await deleteOllamaModel.mutateAsync({
                                modelName: model.name,
                                baseUrl: ollamaBaseUrl,
                              });
                              if (result.success) {
                                refetchOllamaModels();
                              } else {
                                alert(result.error || "Delete failed");
                              }
                            } catch (err) {
                              alert(getErrorMessage(err, "Delete failed"));
                            } finally {
                              setDeletingModel(null);
                            }
                          }}
                          disabled={deletingModel === model.name}
                          className="text-red-500 hover:text-red-600"
                        >
                          {deletingModel === model.name ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Empty state */}
              {(!ollamaLocalModels.chat_models?.length && !ollamaLocalModels.embedding_models?.length) && (
                <div className="text-center py-8 text-muted-foreground">
                  <Bot className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No models installed</p>
                  <p className="text-xs mt-1">Pull a model using the form above</p>
                </div>
              )}
            </div>
          ) : (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {ollamaLocalModels?.error || "Cannot connect to Ollama. Is it running?"}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  );
}