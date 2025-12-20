"use client";

import { useState, useEffect, useRef } from "react";
import {
  Database,
  Server,
  Key,
  Bell,
  Shield,
  Save,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Loader2,
  RotateCcw,
  HardDrive,
  Download,
  Upload,
  Bot,
  Plus,
  Trash2,
  Edit2,
  Star,
  TestTube,
  Eye,
  EyeOff,
  BarChart3,
  Cog,
  MessageSquare,
  FileText,
  Search,
  Sparkles,
  Activity,
  DollarSign,
  XCircle,
  CheckCheck,
  Clock,
  Zap,
  AlertTriangle,
  Tag,
  PenTool,
  Users,
  Globe,
  Workflow,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  useSettings,
  useUpdateSettings,
  useResetSettings,
  useSystemHealth,
  useDatabaseInfo,
  useTestDatabaseConnection,
  useExportDatabase,
  useImportDatabase,
  useSetupPostgresql,
  useLLMProviders,
  useLLMProviderTypes,
  useCreateLLMProvider,
  useDeleteLLMProvider,
  useTestLLMProvider,
  useSetDefaultLLMProvider,
  useUpdateLLMProvider,
  useDatabaseConnections,
  useDatabaseConnectionTypes,
  useCreateDatabaseConnection,
  useDeleteDatabaseConnection,
  useTestDatabaseConnectionById,
  useActivateDatabaseConnection,
  useLLMOperations,
  useSetLLMOperationConfig,
  useDeleteLLMOperationConfig,
  useLLMUsageSummary,
  useLLMUsageByProvider,
  useLLMUsageByOperation,
  useInvalidateLLMCache,
  useProviderHealth,
  useTriggerHealthCheck,
  useResetProviderCircuit,
  useCostAlertsAdmin,
  useAcknowledgeCostAlert,
  useOllamaModels,
} from "@/lib/api/hooks";
import type { LLMProvider, LLMProviderType, DatabaseConnectionType } from "@/lib/api/client";
import { useUser } from "@/lib/auth";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

export default function AdminSettingsPage() {
  const [localSettings, setLocalSettings] = useState<Record<string, unknown>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [newDbUrl, setNewDbUrl] = useState("");
  const [testResult, setTestResult] = useState<{
    success: boolean;
    has_pgvector?: boolean;
    pgvector_version?: string | null;
    error?: string;
    message: string;
  } | null>(null);
  const importRef = useRef<HTMLInputElement>(null);
  const { isAuthenticated, isLoading: authLoading } = useUser();

  // LLM Provider state
  const [showAddProvider, setShowAddProvider] = useState(false);
  const [newProvider, setNewProvider] = useState({
    name: "",
    provider_type: "openai",
    api_key: "",
    api_base_url: "",
    organization_id: "",
    default_chat_model: "",
    default_embedding_model: "",
    is_default: false,
  });
  const [showApiKey, setShowApiKey] = useState(false);
  const [providerTestResults, setProviderTestResults] = useState<Record<string, { success: boolean; message?: string; error?: string }>>({});
  const [editingProvider, setEditingProvider] = useState<LLMProvider | null>(null);

  // Database Connection state
  const [showAddConnection, setShowAddConnection] = useState(false);
  const [newConnection, setNewConnection] = useState({
    name: "",
    db_type: "postgresql",
    host: "localhost",
    port: 5432,
    database: "aidocindexer",
    username: "",
    password: "",
    vector_store: "auto",
    is_active: false,
  });
  const [connectionTestResults, setConnectionTestResults] = useState<Record<string, { success: boolean; message?: string; error?: string }>>({});

  // Real API calls - only fetch when authenticated
  const { data: settingsData, isLoading: settingsLoading, error: settingsError, refetch: refetchSettings } = useSettings({ enabled: isAuthenticated });
  const { data: healthData, isLoading: healthLoading } = useSystemHealth({ enabled: isAuthenticated });
  const { data: dbInfo, isLoading: dbInfoLoading, refetch: refetchDbInfo } = useDatabaseInfo({ enabled: isAuthenticated });
  const updateSettings = useUpdateSettings();
  const resetSettings = useResetSettings();
  const testConnection = useTestDatabaseConnection();
  const exportDatabase = useExportDatabase();
  const importDatabase = useImportDatabase();
  const setupPostgres = useSetupPostgresql();

  // LLM Provider hooks
  const { data: providersData, isLoading: providersLoading, refetch: refetchProviders } = useLLMProviders({ enabled: isAuthenticated });
  const { data: providerTypesData } = useLLMProviderTypes({ enabled: isAuthenticated });
  const createProvider = useCreateLLMProvider();
  const updateProvider = useUpdateLLMProvider();
  const deleteProvider = useDeleteLLMProvider();
  const testProvider = useTestLLMProvider();
  const setDefaultProvider = useSetDefaultLLMProvider();

  // Ollama models - fetch when provider type is ollama
  const { data: ollamaModelsData } = useOllamaModels(
    newProvider.api_base_url || undefined,
    { enabled: isAuthenticated && newProvider.provider_type === "ollama" }
  );

  // Database Connection hooks
  const { data: connectionsData, isLoading: connectionsLoading, refetch: refetchConnections } = useDatabaseConnections({ enabled: isAuthenticated });
  const { data: connectionTypesData } = useDatabaseConnectionTypes({ enabled: isAuthenticated });
  const createConnection = useCreateDatabaseConnection();
  const deleteConnection = useDeleteDatabaseConnection();
  const testConnectionById = useTestDatabaseConnectionById();
  const activateConnection = useActivateDatabaseConnection();

  // Initialize local settings when data loads
  useEffect(() => {
    if (settingsData?.settings) {
      setLocalSettings(settingsData.settings);
      setHasChanges(false);
    }
  }, [settingsData]);

  const handleSettingChange = (key: string, value: unknown) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    try {
      await updateSettings.mutateAsync(localSettings);
      setHasChanges(false);
    } catch (err) {
      console.error("Failed to save settings:", err);
    }
  };

  const handleReset = async () => {
    if (confirm("Are you sure you want to reset all settings to defaults?")) {
      try {
        await resetSettings.mutateAsync();
        setHasChanges(false);
      } catch (err) {
        console.error("Failed to reset settings:", err);
      }
    }
  };

  const handleTestConnection = async () => {
    if (!newDbUrl) return;
    setTestResult(null);
    try {
      const result = await testConnection.mutateAsync(newDbUrl);
      setTestResult(result);
    } catch (err) {
      setTestResult({
        success: false,
        error: (err as Error).message,
        message: "Connection test failed",
      });
    }
  };

  const handleSetupPostgres = async () => {
    if (!newDbUrl) return;
    try {
      const result = await setupPostgres.mutateAsync(newDbUrl);
      if (result.success) {
        setTestResult({
          success: true,
          has_pgvector: result.has_pgvector,
          pgvector_version: result.pgvector_version,
          message: result.message,
        });
      } else {
        setTestResult({
          success: false,
          error: result.error,
          message: result.message,
        });
      }
    } catch (err) {
      setTestResult({
        success: false,
        error: (err as Error).message,
        message: "PostgreSQL setup failed",
      });
    }
  };

  const handleExport = async () => {
    try {
      const data = await exportDatabase.mutateAsync();
      // Download as JSON file
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `aidocindexer_export_${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
      alert("Export failed: " + (err as Error).message);
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const data = JSON.parse(text);

      const clearExisting = confirm(
        "Do you want to clear existing data before importing?\n\n" +
        "Click OK to clear existing data first.\n" +
        "Click Cancel to merge with existing data."
      );

      const result = await importDatabase.mutateAsync({ data, clearExisting });

      if (result.success) {
        alert(
          `Import completed!\n\n` +
          `Imported:\n` +
          Object.entries(result.imported)
            .map(([key, count]) => `  ${key}: ${count}`)
            .join("\n") +
          (result.warnings.length > 0 ? `\n\nWarnings:\n${result.warnings.join("\n")}` : "")
        );
        refetchDbInfo();
      } else {
        alert(`Import failed:\n${result.errors.join("\n")}`);
      }
    } catch (err) {
      console.error("Import failed:", err);
      alert("Import failed: " + (err as Error).message);
    }

    // Reset file input
    if (importRef.current) {
      importRef.current.value = "";
    }
  };

  // LLM Provider handlers
  const handleAddProvider = async () => {
    try {
      await createProvider.mutateAsync({
        name: newProvider.name,
        provider_type: newProvider.provider_type,
        api_key: newProvider.api_key || undefined,
        api_base_url: newProvider.api_base_url || undefined,
        organization_id: newProvider.organization_id || undefined,
        default_chat_model: newProvider.default_chat_model || undefined,
        default_embedding_model: newProvider.default_embedding_model || undefined,
        is_default: newProvider.is_default,
      });
      setShowAddProvider(false);
      setNewProvider({
        name: "",
        provider_type: "openai",
        api_key: "",
        api_base_url: "",
        organization_id: "",
        default_chat_model: "",
        default_embedding_model: "",
        is_default: false,
      });
      refetchProviders();
    } catch (err) {
      console.error("Failed to create provider:", err);
      alert("Failed to create provider: " + (err as Error).message);
    }
  };

  const handleDeleteProvider = async (providerId: string) => {
    if (!confirm("Are you sure you want to delete this provider?")) return;
    try {
      await deleteProvider.mutateAsync(providerId);
      refetchProviders();
    } catch (err) {
      console.error("Failed to delete provider:", err);
      alert("Failed to delete provider: " + (err as Error).message);
    }
  };

  const handleTestProvider = async (providerId: string) => {
    setProviderTestResults((prev) => ({ ...prev, [providerId]: { success: true, message: "Testing..." } }));
    try {
      const result = await testProvider.mutateAsync(providerId);
      setProviderTestResults((prev) => ({
        ...prev,
        [providerId]: { success: result.success, message: result.message, error: result.error },
      }));
    } catch (err) {
      setProviderTestResults((prev) => ({
        ...prev,
        [providerId]: { success: false, error: (err as Error).message },
      }));
    }
  };

  const handleSetDefaultProvider = async (providerId: string) => {
    try {
      await setDefaultProvider.mutateAsync(providerId);
      refetchProviders();
    } catch (err) {
      console.error("Failed to set default provider:", err);
      alert("Failed to set default provider: " + (err as Error).message);
    }
  };

  const handleEditProvider = (provider: LLMProvider) => {
    setEditingProvider(provider);
    setNewProvider({
      name: provider.name,
      provider_type: provider.provider_type,
      api_key: "", // Keep empty for security - only send if user enters new key
      api_base_url: provider.api_base_url || "",
      organization_id: provider.organization_id || "",
      default_chat_model: provider.default_chat_model || "",
      default_embedding_model: provider.default_embedding_model || "",
      is_default: provider.is_default,
    });
    setShowAddProvider(true);
  };

  const handleSaveProvider = async () => {
    if (editingProvider) {
      // Update existing provider
      try {
        await updateProvider.mutateAsync({
          providerId: editingProvider.id,
          data: {
            name: newProvider.name !== editingProvider.name ? newProvider.name : undefined,
            api_key: newProvider.api_key || undefined, // Only send if changed
            api_base_url: newProvider.api_base_url || undefined,
            organization_id: newProvider.organization_id || undefined,
            default_chat_model: newProvider.default_chat_model || undefined,
            default_embedding_model: newProvider.default_embedding_model || undefined,
            is_active: true,
          },
        });
        setEditingProvider(null);
        setShowAddProvider(false);
        setNewProvider({
          name: "",
          provider_type: "openai",
          api_key: "",
          api_base_url: "",
          organization_id: "",
          default_chat_model: "",
          default_embedding_model: "",
          is_default: false,
        });
        refetchProviders();
      } catch (err) {
        console.error("Failed to update provider:", err);
        alert("Failed to update provider: " + (err as Error).message);
      }
    } else {
      // Create new provider
      await handleAddProvider();
    }
  };

  const handleCancelProviderForm = () => {
    setShowAddProvider(false);
    setEditingProvider(null);
    setNewProvider({
      name: "",
      provider_type: "openai",
      api_key: "",
      api_base_url: "",
      organization_id: "",
      default_chat_model: "",
      default_embedding_model: "",
      is_default: false,
    });
  };

  // Database Connection handlers
  const handleAddConnection = async () => {
    try {
      await createConnection.mutateAsync({
        name: newConnection.name,
        db_type: newConnection.db_type,
        database: newConnection.database,
        host: newConnection.db_type !== "sqlite" ? newConnection.host : undefined,
        port: newConnection.db_type !== "sqlite" ? newConnection.port : undefined,
        username: newConnection.db_type !== "sqlite" ? newConnection.username : undefined,
        password: newConnection.db_type !== "sqlite" ? newConnection.password : undefined,
        vector_store: newConnection.vector_store,
        is_active: newConnection.is_active,
      });
      setShowAddConnection(false);
      setNewConnection({
        name: "",
        db_type: "postgresql",
        host: "localhost",
        port: 5432,
        database: "aidocindexer",
        username: "",
        password: "",
        vector_store: "auto",
        is_active: false,
      });
      refetchConnections();
    } catch (err) {
      console.error("Failed to create connection:", err);
      alert("Failed to create connection: " + (err as Error).message);
    }
  };

  const handleDeleteConnection = async (connectionId: string) => {
    if (!confirm("Are you sure you want to delete this connection?")) return;
    try {
      await deleteConnection.mutateAsync(connectionId);
      refetchConnections();
    } catch (err) {
      console.error("Failed to delete connection:", err);
      alert("Failed to delete connection: " + (err as Error).message);
    }
  };

  const handleTestSavedConnection = async (connectionId: string) => {
    setConnectionTestResults((prev) => ({ ...prev, [connectionId]: { success: true, message: "Testing..." } }));
    try {
      const result = await testConnectionById.mutateAsync(connectionId);
      setConnectionTestResults((prev) => ({
        ...prev,
        [connectionId]: { success: result.success, message: result.message, error: result.error },
      }));
    } catch (err) {
      setConnectionTestResults((prev) => ({
        ...prev,
        [connectionId]: { success: false, error: (err as Error).message },
      }));
    }
  };

  const handleActivateConnection = async (connectionId: string) => {
    try {
      await activateConnection.mutateAsync(connectionId);
      refetchConnections();
      refetchDbInfo();
      alert("Connection activated. Please restart the server to use this connection.");
    } catch (err) {
      console.error("Failed to activate connection:", err);
      alert("Failed to activate connection: " + (err as Error).message);
    }
  };

  const getProviderTypeConfig = (type: string): LLMProviderType | undefined => {
    return providerTypesData?.provider_types?.[type];
  };

  const getDbTypeConfig = (type: string): DatabaseConnectionType | undefined => {
    return connectionTypesData?.database_types?.[type];
  };

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

  if (authLoading || settingsLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (settingsError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
        <h3 className="text-lg font-semibold mb-2">Failed to load settings</h3>
        <p className="text-muted-foreground mb-4">
          {(settingsError as any)?.detail || "Unable to fetch settings. Please try again."}
        </p>
        <Button onClick={() => refetchSettings()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">
            Configure system settings and integrations
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={handleReset}
            disabled={resetSettings.isPending}
          >
            {resetSettings.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RotateCcw className="h-4 w-4 mr-2" />
            )}
            Reset to Defaults
          </Button>
          <Button
            onClick={handleSave}
            disabled={!hasChanges || updateSettings.isPending}
          >
            {updateSettings.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save Changes
          </Button>
        </div>
      </div>

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
                        {provider.default_chat_model && ` • ${provider.default_chat_model}`}
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
                      placeholder={editingProvider ? "••••••••••••••••" : "sk-..."}
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

      {/* Database Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Database className="h-5 w-5" />
            Database Settings
          </CardTitle>
          <CardDescription>Vector database configuration</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Vector Dimensions</label>
              <Input
                type="number"
                value={localSettings["database.vector_dimensions"] as number ?? 1536}
                onChange={(e) => handleSettingChange("database.vector_dimensions", parseInt(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Index Type</label>
              <select
                className="w-full h-10 px-3 rounded-md border bg-background"
                value={localSettings["database.index_type"] as string || "hnsw"}
                onChange={(e) => handleSettingChange("database.index_type", e.target.value)}
              >
                <option value="hnsw">HNSW</option>
                <option value="ivfflat">IVFFlat</option>
              </select>
            </div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Max Results per Query</label>
            <Input
              type="number"
              value={localSettings["database.max_results_per_query"] as number ?? 10}
              onChange={(e) => handleSettingChange("database.max_results_per_query", parseInt(e.target.value))}
            />
          </div>
        </CardContent>
      </Card>

      {/* Database Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Database Configuration
          </CardTitle>
          <CardDescription>
            View and manage database connection
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Current Status */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Current Database</h4>
            {dbInfoLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : dbInfo ? (
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="text-sm">Type</span>
                  <Badge variant={dbInfo.type === "postgresql" ? "default" : "secondary"}>
                    {dbInfo.type === "postgresql" ? "PostgreSQL" : "SQLite"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="text-sm">Vector Store</span>
                  <Badge variant="outline">
                    {dbInfo.vector_store === "pgvector" ? "pgvector" : "ChromaDB"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="text-sm">Documents</span>
                  <span className="font-medium">{dbInfo.documents_count}</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="text-sm">Chunks</span>
                  <span className="font-medium">{dbInfo.chunks_count}</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="text-sm">Users</span>
                  <span className="font-medium">{dbInfo.users_count}</span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="text-sm">Status</span>
                  <div className="flex items-center gap-1">
                    {dbInfo.is_connected ? (
                      <>
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm text-green-600">Connected</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-4 w-4 text-red-500" />
                        <span className="text-sm text-red-600">Disconnected</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Unable to fetch database info</p>
            )}
          </div>

          {/* Saved Connections */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Saved Connections</h4>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAddConnection(!showAddConnection)}
              >
                <Plus className="h-4 w-4 mr-1" />
                Add Connection
              </Button>
            </div>
            {connectionsLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : connectionsData?.connections && connectionsData.connections.length > 0 ? (
              <div className="space-y-2">
                {connectionsData.connections.map((conn) => (
                  <div
                    key={conn.id}
                    className="flex items-center justify-between p-3 rounded-lg border"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{conn.name}</span>
                          {conn.is_active && (
                            <Badge variant="default" className="text-xs">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Active
                            </Badge>
                          )}
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {getDbTypeConfig(conn.db_type)?.name || conn.db_type}
                          {conn.host && ` • ${conn.host}:${conn.port}`}
                          {` • ${conn.database}`}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {connectionTestResults[conn.id] && (
                        <span className={`text-xs ${connectionTestResults[conn.id].success ? "text-green-600" : "text-red-600"}`}>
                          {connectionTestResults[conn.id].message || connectionTestResults[conn.id].error}
                        </span>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleTestSavedConnection(conn.id)}
                        disabled={testConnectionById.isPending}
                        title="Test connection"
                      >
                        <TestTube className="h-4 w-4" />
                      </Button>
                      {!conn.is_active && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleActivateConnection(conn.id)}
                          disabled={activateConnection.isPending}
                          title="Activate connection"
                        >
                          <CheckCircle className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDeleteConnection(conn.id)}
                        disabled={deleteConnection.isPending || conn.is_active}
                        className="text-red-500 hover:text-red-600"
                        title="Delete connection"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-4">
                No saved connections. Add a connection to get started.
              </p>
            )}

            {/* Add Connection Form */}
            {showAddConnection && (
              <div className="p-4 rounded-lg border bg-muted/50 space-y-4">
                <h4 className="font-medium">Add New Connection</h4>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Connection Name</label>
                    <Input
                      placeholder="Production Database"
                      value={newConnection.name}
                      onChange={(e) => setNewConnection({ ...newConnection, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Database Type</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={newConnection.db_type}
                      onChange={(e) => {
                        const type = e.target.value;
                        const config = getDbTypeConfig(type);
                        setNewConnection({
                          ...newConnection,
                          db_type: type,
                          port: config?.default_port || 5432,
                          database: config?.default_database || "aidocindexer",
                        });
                      }}
                    >
                      {connectionTypesData?.database_types &&
                        Object.entries(connectionTypesData.database_types).map(([key, config]) => (
                          <option key={key} value={key}>
                            {config.name}
                          </option>
                        ))}
                    </select>
                  </div>
                </div>
                {newConnection.db_type !== "sqlite" && (
                  <>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Host</label>
                        <Input
                          placeholder="localhost"
                          value={newConnection.host}
                          onChange={(e) => setNewConnection({ ...newConnection, host: e.target.value })}
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Port</label>
                        <Input
                          type="number"
                          placeholder="5432"
                          value={newConnection.port}
                          onChange={(e) => setNewConnection({ ...newConnection, port: parseInt(e.target.value) })}
                        />
                      </div>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Username</label>
                        <Input
                          placeholder="postgres"
                          value={newConnection.username}
                          onChange={(e) => setNewConnection({ ...newConnection, username: e.target.value })}
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Password</label>
                        <Input
                          type="password"
                          placeholder="••••••••"
                          value={newConnection.password}
                          onChange={(e) => setNewConnection({ ...newConnection, password: e.target.value })}
                        />
                      </div>
                    </div>
                  </>
                )}
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Database Name</label>
                    <Input
                      placeholder={newConnection.db_type === "sqlite" ? "/path/to/database.db" : "aidocindexer"}
                      value={newConnection.database}
                      onChange={(e) => setNewConnection({ ...newConnection, database: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Vector Store</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={newConnection.vector_store}
                      onChange={(e) => setNewConnection({ ...newConnection, vector_store: e.target.value })}
                    >
                      <option value="auto">Auto-detect</option>
                      <option value="pgvector">pgvector</option>
                      <option value="chromadb">ChromaDB</option>
                    </select>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button
                    onClick={handleAddConnection}
                    disabled={!newConnection.name || !newConnection.database || createConnection.isPending}
                  >
                    {createConnection.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Plus className="h-4 w-4 mr-2" />
                    )}
                    Add Connection
                  </Button>
                  <Button variant="outline" onClick={() => setShowAddConnection(false)}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </div>

          {/* Migration Tools */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Data Migration</h4>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                onClick={handleExport}
                disabled={exportDatabase.isPending}
              >
                {exportDatabase.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Download className="h-4 w-4 mr-2" />
                )}
                Export Data
              </Button>
              <Button
                variant="outline"
                onClick={() => importRef.current?.click()}
                disabled={importDatabase.isPending}
              >
                {importDatabase.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Upload className="h-4 w-4 mr-2" />
                )}
                Import Data
              </Button>
              <input
                ref={importRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleImport}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Export/import all data as JSON for migration between databases.
            </p>
          </div>

          {/* Switch Database */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Switch Database</h4>
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Changing databases requires updating environment variables and restarting the server.
                Export your data first to avoid data loss.
              </AlertDescription>
            </Alert>
            <div className="p-4 rounded-lg border space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">PostgreSQL Connection URL</label>
                <Input
                  placeholder="postgresql://user:password@localhost:5432/aidocindexer"
                  value={newDbUrl}
                  onChange={(e) => {
                    setNewDbUrl(e.target.value);
                    setTestResult(null);
                  }}
                />
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  onClick={handleTestConnection}
                  disabled={!newDbUrl || testConnection.isPending}
                >
                  {testConnection.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Server className="h-4 w-4 mr-2" />
                  )}
                  Test Connection
                </Button>
                <Button
                  variant="outline"
                  onClick={handleSetupPostgres}
                  disabled={!newDbUrl || setupPostgres.isPending}
                >
                  {setupPostgres.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Database className="h-4 w-4 mr-2" />
                  )}
                  Setup pgvector
                </Button>
              </div>
              {testResult && (
                <div className={`p-3 rounded-lg ${testResult.success ? "bg-green-500/10 border border-green-500/20" : "bg-red-500/10 border border-red-500/20"}`}>
                  <div className={`flex items-center gap-2 text-sm ${testResult.success ? "text-green-600" : "text-red-600"}`}>
                    {testResult.success ? (
                      <CheckCircle className="h-4 w-4" />
                    ) : (
                      <AlertCircle className="h-4 w-4" />
                    )}
                    <span>{testResult.message}</span>
                  </div>
                  {testResult.success && testResult.has_pgvector !== undefined && (
                    <p className="text-xs mt-1 text-muted-foreground">
                      pgvector: {testResult.has_pgvector ? `Installed (v${testResult.pgvector_version})` : "Not installed"}
                    </p>
                  )}
                  {!testResult.success && testResult.error && (
                    <p className="text-xs mt-1 text-red-500">{testResult.error}</p>
                  )}
                </div>
              )}
              <div className="text-xs text-muted-foreground space-y-1">
                <p>After testing, update your <code className="px-1 py-0.5 bg-muted rounded">.env</code> file:</p>
                <pre className="p-2 bg-muted rounded text-xs overflow-x-auto">
{`DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://user:password@localhost:5432/aidocindexer
VECTOR_STORE_BACKEND=auto`}
                </pre>
                <p>Then restart the server.</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Optimization Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            AI Optimization
          </CardTitle>
          <CardDescription>
            Configure AI features to reduce costs and improve quality
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Text Preprocessing */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Text Preprocessing
            </h4>
            <p className="text-sm text-muted-foreground">
              Clean and normalize text before embedding to reduce token costs (10-20% savings)
            </p>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Enable Preprocessing</p>
                  <p className="text-sm text-muted-foreground">
                    Clean text before embedding
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.enable_preprocessing"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("ai.enable_preprocessing", e.target.checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Remove Boilerplate</p>
                  <p className="text-sm text-muted-foreground">
                    Strip headers, footers, page numbers
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.remove_boilerplate"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("ai.remove_boilerplate", e.target.checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Normalize Whitespace</p>
                  <p className="text-sm text-muted-foreground">
                    Collapse multiple spaces/newlines
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.normalize_whitespace"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("ai.normalize_whitespace", e.target.checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Deduplicate Content</p>
                  <p className="text-sm text-muted-foreground">
                    Remove near-duplicate chunks
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["ai.deduplicate_content"] as boolean ?? false}
                  onChange={(e) => handleSettingChange("ai.deduplicate_content", e.target.checked)}
                />
              </div>
            </div>
          </div>

          {/* Document Summarization */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Document Summarization
            </h4>
            <p className="text-sm text-muted-foreground">
              Generate summaries for large documents to reduce embedding tokens (30-40% savings)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Summarization</p>
                <p className="text-sm text-muted-foreground">
                  Summarize large docs before chunking
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_summarization"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_summarization", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Page Threshold</label>
                <Input
                  type="number"
                  placeholder="50"
                  value={localSettings["ai.summarization_threshold_pages"] as number ?? 50}
                  onChange={(e) => handleSettingChange("ai.summarization_threshold_pages", parseInt(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Summarize docs with more pages</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">KB Threshold</label>
                <Input
                  type="number"
                  placeholder="100"
                  value={localSettings["ai.summarization_threshold_kb"] as number ?? 100}
                  onChange={(e) => handleSettingChange("ai.summarization_threshold_kb", parseInt(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Summarize docs larger than this</p>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Summary Model</label>
              <select
                className="w-full h-10 px-3 rounded-md border bg-background"
                value={localSettings["ai.summarization_model"] as string || "gpt-4o-mini"}
                onChange={(e) => handleSettingChange("ai.summarization_model", e.target.value)}
              >
                <option value="gpt-4o-mini">GPT-4o Mini (Cost-effective)</option>
                <option value="gpt-4o">GPT-4o (Higher quality)</option>
                <option value="claude-3-haiku">Claude 3 Haiku (Fast)</option>
              </select>
            </div>
          </div>

          {/* Semantic Caching */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Search className="h-4 w-4" />
              Semantic Caching
            </h4>
            <p className="text-sm text-muted-foreground">
              Cache responses for semantically similar queries (up to 68% API cost reduction)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Semantic Cache</p>
                <p className="text-sm text-muted-foreground">
                  Match similar queries by embedding
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_semantic_cache"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_semantic_cache", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Similarity Threshold</label>
                <Input
                  type="number"
                  step="0.01"
                  min="0.8"
                  max="1.0"
                  placeholder="0.95"
                  value={localSettings["ai.semantic_similarity_threshold"] as number ?? 0.95}
                  onChange={(e) => handleSettingChange("ai.semantic_similarity_threshold", parseFloat(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Higher = stricter matching (0.8-1.0)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Cache Entries</label>
                <Input
                  type="number"
                  placeholder="10000"
                  value={localSettings["ai.max_semantic_cache_entries"] as number ?? 10000}
                  onChange={(e) => handleSettingChange("ai.max_semantic_cache_entries", parseInt(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Limit semantic cache size</p>
              </div>
            </div>
          </div>

          {/* Query Expansion */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Query Expansion
            </h4>
            <p className="text-sm text-muted-foreground">
              Generate query variations to improve search recall (8-12% accuracy improvement)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Query Expansion</p>
                <p className="text-sm text-muted-foreground">
                  Generate paraphrased queries for better retrieval
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_query_expansion"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_query_expansion", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Query Variations</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  placeholder="2"
                  value={localSettings["ai.query_expansion_count"] as number ?? 2}
                  onChange={(e) => handleSettingChange("ai.query_expansion_count", parseInt(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Number of variations to generate (1-5)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Expansion Model</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["ai.query_expansion_model"] as string || "gpt-4o-mini"}
                  onChange={(e) => handleSettingChange("ai.query_expansion_model", e.target.value)}
                >
                  <option value="gpt-4o-mini">GPT-4o Mini (Cost-effective)</option>
                  <option value="gpt-4o">GPT-4o (Higher quality)</option>
                  <option value="claude-3-haiku">Claude 3 Haiku (Fast)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Adaptive Chunking */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Cog className="h-4 w-4" />
              Adaptive Chunking
            </h4>
            <p className="text-sm text-muted-foreground">
              Automatically optimize chunk size based on document type (10-15% token savings)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Adaptive Chunking</p>
                <p className="text-sm text-muted-foreground">
                  Auto-detect document type and adjust chunk size
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_adaptive_chunking"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_adaptive_chunking", e.target.checked)}
              />
            </div>
            <div className="bg-muted/30 rounded-lg p-3">
              <p className="text-xs font-medium mb-2">Type-Specific Chunk Sizes</p>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>Code: 256 chars (preserve function boundaries)</li>
                <li>Legal: 1024 chars (preserve paragraphs)</li>
                <li>Technical: 512 chars (preserve sections)</li>
                <li>Academic: 800 chars (preserve citations)</li>
                <li>General: 1000 chars (balanced)</li>
              </ul>
            </div>
          </div>

          {/* Hierarchical Chunking */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Hierarchical Chunking
            </h4>
            <p className="text-sm text-muted-foreground">
              Create multi-level chunk hierarchy for very large documents (20-30% token savings)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Hierarchical Chunking</p>
                <p className="text-sm text-muted-foreground">
                  Create document/section/detail chunk levels
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["ai.enable_hierarchical_chunking"] as boolean ?? false}
                onChange={(e) => handleSettingChange("ai.enable_hierarchical_chunking", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Size Threshold (KB)</label>
                <Input
                  type="number"
                  min="50"
                  placeholder="100"
                  value={(localSettings["ai.hierarchical_threshold_chars"] as number ?? 100000) / 1000}
                  onChange={(e) => handleSettingChange("ai.hierarchical_threshold_chars", parseInt(e.target.value) * 1000)}
                />
                <p className="text-xs text-muted-foreground">Enable for docs larger than this</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Target Sections</label>
                <Input
                  type="number"
                  min="3"
                  max="20"
                  placeholder="10"
                  value={localSettings["ai.sections_per_document"] as number ?? 10}
                  onChange={(e) => handleSettingChange("ai.sections_per_document", parseInt(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Number of section summaries</p>
              </div>
            </div>
          </div>

          {/* Cost Savings Info */}
          <div className="pt-4 border-t">
            <div className="bg-muted/50 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <DollarSign className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <p className="font-medium text-sm">Estimated Savings</p>
                  <ul className="text-sm text-muted-foreground mt-1 space-y-1">
                    <li>Text Preprocessing: 10-20% token reduction</li>
                    <li>Document Summarization: 30-40% for large files</li>
                    <li>Semantic Caching: Up to 68% fewer API calls</li>
                    <li>Query Expansion: 8-12% accuracy improvement</li>
                    <li>Adaptive Chunking: 10-15% token savings</li>
                    <li>Hierarchical Chunking: 20-30% for large docs</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Security Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Security Settings
          </CardTitle>
          <CardDescription>Authentication and access control</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Require Email Verification</p>
              <p className="text-sm text-muted-foreground">
                New users must verify their email
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["security.require_email_verification"] as boolean ?? false}
              onChange={(e) => handleSettingChange("security.require_email_verification", e.target.checked)}
            />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Two-Factor Authentication</p>
              <p className="text-sm text-muted-foreground">
                Require 2FA for admin accounts
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["security.enable_2fa"] as boolean ?? false}
              onChange={(e) => handleSettingChange("security.enable_2fa", e.target.checked)}
            />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Audit Logging</p>
              <p className="text-sm text-muted-foreground">
                Log all user actions for compliance
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["security.enable_audit_logging"] as boolean ?? true}
              onChange={(e) => handleSettingChange("security.enable_audit_logging", e.target.checked)}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Session Timeout (minutes)</label>
            <Input
              type="number"
              value={localSettings["security.session_timeout_minutes"] as number ?? 60}
              onChange={(e) => handleSettingChange("security.session_timeout_minutes", parseInt(e.target.value))}
            />
          </div>
        </CardContent>
      </Card>

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

      {/* Operation-Level LLM Configuration Card */}
      <OperationLevelConfigCard providers={providersData?.providers || []} />

      {/* LLM Usage Analytics Card */}
      <UsageAnalyticsCard />

      {/* Provider Health Card */}
      <ProviderHealthCard />

      {/* Cost Alerts Card */}
      <CostAlertsCard />

      {/* Unsaved Changes Warning */}
      {hasChanges && (
        <div className="fixed bottom-4 right-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-yellow-500" />
          <span className="text-sm">You have unsaved changes</span>
          <Button size="sm" onClick={handleSave} disabled={updateSettings.isPending}>
            {updateSettings.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              "Save"
            )}
          </Button>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Operation-Level Configuration Card Component
// =============================================================================

const OPERATION_ICONS: Record<string, React.ReactNode> = {
  chat: <MessageSquare className="h-4 w-4" />,
  embeddings: <Search className="h-4 w-4" />,
  document_processing: <FileText className="h-4 w-4" />,
  rag: <Sparkles className="h-4 w-4" />,
  summarization: <FileText className="h-4 w-4" />,
  document_enhancement: <Sparkles className="h-4 w-4" />,
  auto_tagging: <Tag className="h-4 w-4" />,
  content_generation: <PenTool className="h-4 w-4" />,
  collaboration: <Users className="h-4 w-4" />,
  web_scraping: <Globe className="h-4 w-4" />,
  agent_planning: <Workflow className="h-4 w-4" />,
  agent_execution: <Play className="h-4 w-4" />,
};

const OPERATION_LABELS: Record<string, string> = {
  chat: "Chat",
  embeddings: "Embeddings",
  document_processing: "Document Processing",
  rag: "RAG",
  summarization: "Summarization",
  document_enhancement: "Document Enhancement",
  auto_tagging: "Auto Tagging",
  content_generation: "Content Generation",
  collaboration: "Model Collaboration",
  web_scraping: "Web Scraping",
  agent_planning: "Agent Planning",
  agent_execution: "Agent Execution",
};

function OperationLevelConfigCard({ providers }: { providers: LLMProvider[] }) {
  const { data: operationsData, isLoading } = useLLMOperations();
  const setOperationConfig = useSetLLMOperationConfig();
  const deleteOperationConfig = useDeleteLLMOperationConfig();
  const invalidateCache = useInvalidateLLMCache();

  const [editingOperation, setEditingOperation] = useState<string | null>(null);
  const [operationConfigs, setOperationConfigs] = useState<Record<string, { provider_id: string; model_override: string }>>({});

  // Initialize operation configs from data
  useEffect(() => {
    if (operationsData?.operations) {
      const configs: Record<string, { provider_id: string; model_override: string }> = {};
      for (const op of operationsData.operations) {
        configs[op.operation_type] = {
          provider_id: op.provider_id || "",
          model_override: op.model_override || "",
        };
      }
      setOperationConfigs(configs);
    }
  }, [operationsData]);

  const validOperations = operationsData?.valid_operations || ["chat", "embeddings", "document_processing", "rag", "summarization"];
  const activeProviders = providers.filter(p => p.is_active);

  const handleSaveOperation = async (operationType: string) => {
    const config = operationConfigs[operationType];
    if (!config?.provider_id) return;

    await setOperationConfig.mutateAsync({
      operationType,
      data: {
        provider_id: config.provider_id,
        model_override: config.model_override || undefined,
      },
    });
    setEditingOperation(null);
  };

  const handleResetOperation = async (operationType: string) => {
    await deleteOperationConfig.mutateAsync(operationType);
    setOperationConfigs(prev => {
      const updated = { ...prev };
      delete updated[operationType];
      return updated;
    });
  };

  const getOperationConfig = (operationType: string) => {
    return operationsData?.operations.find(op => op.operation_type === operationType);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Cog className="h-5 w-5" />
          Operation-Level LLM Configuration
        </CardTitle>
        <CardDescription>
          Assign different LLM providers to different operations for granular control
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {validOperations.map((operationType) => {
                const config = getOperationConfig(operationType);
                const isEditing = editingOperation === operationType;
                const localConfig = operationConfigs[operationType] || { provider_id: "", model_override: "" };

                return (
                  <div key={operationType} className="flex items-center justify-between p-3 rounded-lg border">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-muted">
                        {OPERATION_ICONS[operationType]}
                      </div>
                      <div>
                        <p className="font-medium">{OPERATION_LABELS[operationType]}</p>
                        {config?.provider_name ? (
                          <p className="text-sm text-muted-foreground">
                            {config.provider_name} {config.model_override && `(${config.model_override})`}
                          </p>
                        ) : (
                          <p className="text-sm text-muted-foreground">Using default provider</p>
                        )}
                      </div>
                    </div>

                    {isEditing ? (
                      <div className="flex items-center gap-2">
                        <select
                          className="h-8 rounded-md border bg-background px-2 text-sm"
                          value={localConfig.provider_id}
                          onChange={(e) => setOperationConfigs(prev => ({
                            ...prev,
                            [operationType]: { ...prev[operationType], provider_id: e.target.value },
                          }))}
                        >
                          <option value="">Use Default</option>
                          {activeProviders.map(p => (
                            <option key={p.id} value={p.id}>
                              {p.name} ({p.provider_type})
                            </option>
                          ))}
                        </select>
                        <Input
                          placeholder="Model override (optional)"
                          className="h-8 w-40"
                          value={localConfig.model_override || ""}
                          onChange={(e) => setOperationConfigs(prev => ({
                            ...prev,
                            [operationType]: { ...prev[operationType], model_override: e.target.value },
                          }))}
                        />
                        <Button
                          size="sm"
                          variant="default"
                          onClick={() => handleSaveOperation(operationType)}
                          disabled={setOperationConfig.isPending}
                        >
                          {setOperationConfig.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : "Save"}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setEditingOperation(null)}
                        >
                          Cancel
                        </Button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setEditingOperation(operationType)}
                        >
                          <Edit2 className="h-3 w-3 mr-1" />
                          Configure
                        </Button>
                        {config && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleResetOperation(operationType)}
                            disabled={deleteOperationConfig.isPending}
                          >
                            <RotateCcw className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="pt-3 border-t">
              <Button
                variant="outline"
                size="sm"
                onClick={() => invalidateCache.mutate()}
                disabled={invalidateCache.isPending}
              >
                {invalidateCache.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <RefreshCw className="h-4 w-4 mr-2" />
                )}
                Clear LLM Cache
              </Button>
              <p className="text-xs text-muted-foreground mt-2">
                Clear the configuration cache to apply changes immediately
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Usage Analytics Card Component
// =============================================================================

// Chart colors
const CHART_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];

function UsageAnalyticsCard() {
  const { data: summary, isLoading: summaryLoading } = useLLMUsageSummary();
  const { data: byProvider, isLoading: providerLoading } = useLLMUsageByProvider();
  const { data: byOperation, isLoading: operationLoading } = useLLMUsageByOperation();

  const isLoading = summaryLoading || providerLoading || operationLoading;

  const formatCost = (cost: number | null | undefined) => {
    if (cost === null || cost === undefined) return "$0.00";
    return `$${cost.toFixed(4)}`;
  };

  const formatTokens = (tokens: number | null | undefined) => {
    if (!tokens) return "0";
    if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(1)}M`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
    return tokens.toString();
  };

  // Prepare chart data
  const providerChartData = byProvider?.usage_by_provider?.map((item) => ({
    name: item.provider_name || item.provider_type,
    cost: item.total_cost_usd || 0,
    tokens: item.total_tokens || 0,
    requests: item.request_count || 0,
  })) || [];

  const operationChartData = byOperation?.usage_by_operation?.map((item) => ({
    name: OPERATION_LABELS[item.operation_type] || item.operation_type,
    cost: item.total_cost_usd || 0,
    tokens: item.total_tokens || 0,
    requests: item.request_count || 0,
  })) || [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          LLM Usage Analytics
        </CardTitle>
        <CardDescription>
          Track usage and costs across providers and operations
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold">{summary?.request_count || 0}</p>
                <p className="text-sm text-muted-foreground">Total Requests</p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold">{formatTokens(summary?.total_tokens)}</p>
                <p className="text-sm text-muted-foreground">Total Tokens</p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold">{formatCost(summary?.total_cost_usd)}</p>
                <p className="text-sm text-muted-foreground">Total Cost</p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-2xl font-bold">{summary?.avg_duration_ms?.toFixed(0) || 0}ms</p>
                <p className="text-sm text-muted-foreground">Avg Duration</p>
              </div>
            </div>

            {/* Charts Row */}
            {(providerChartData.length > 0 || operationChartData.length > 0) && (
              <div className="grid md:grid-cols-2 gap-6">
                {/* Cost by Provider Pie Chart */}
                {providerChartData.length > 0 && (
                  <div className="p-4 rounded-lg border">
                    <h4 className="text-sm font-medium mb-4">Cost by Provider</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={providerChartData}
                          cx="50%"
                          cy="50%"
                          innerRadius={40}
                          outerRadius={80}
                          paddingAngle={2}
                          dataKey="cost"
                          nameKey="name"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          labelLine={false}
                        >
                          {providerChartData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value: number) => formatCost(value)} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Requests by Operation Bar Chart */}
                {operationChartData.length > 0 && (
                  <div className="p-4 rounded-lg border">
                    <h4 className="text-sm font-medium mb-4">Requests by Operation</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={operationChartData} layout="vertical">
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="name" width={100} />
                        <Tooltip />
                        <Bar dataKey="requests" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* Usage by Provider */}
            {byProvider?.usage_by_provider && byProvider.usage_by_provider.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-3">Usage by Provider</h4>
                <div className="space-y-2">
                  {byProvider.usage_by_provider.map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                        />
                        <Bot className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium">{item.provider_name || item.provider_type}</span>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-muted-foreground">{item.request_count} requests</span>
                        <span className="text-muted-foreground">{formatTokens(item.total_tokens)} tokens</span>
                        <span className="font-medium">{formatCost(item.total_cost_usd)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Usage by Operation */}
            {byOperation?.usage_by_operation && byOperation.usage_by_operation.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-3">Usage by Operation</h4>
                <div className="space-y-2">
                  {byOperation.usage_by_operation.map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center gap-2">
                        {OPERATION_ICONS[item.operation_type] || <Cog className="h-4 w-4" />}
                        <span className="font-medium">{OPERATION_LABELS[item.operation_type] || item.operation_type}</span>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-muted-foreground">{item.request_count} requests</span>
                        <span className="text-muted-foreground">{formatTokens(item.total_tokens)} tokens</span>
                        <span className="font-medium">{formatCost(item.total_cost_usd)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Empty State */}
            {(!summary || summary.request_count === 0) && (
              <div className="text-center py-8 text-muted-foreground">
                <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No usage data yet</p>
                <p className="text-sm">Start using LLM features to see analytics here</p>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Provider Health Card Component
// =============================================================================

function ProviderHealthCard() {
  const { data: healthData, isLoading } = useProviderHealth({ refetchInterval: 30000 });
  const triggerHealthCheck = useTriggerHealthCheck();
  const resetCircuit = useResetProviderCircuit();

  const getHealthIcon = (isHealthy: boolean, circuitOpen: boolean) => {
    if (circuitOpen) return <XCircle className="h-4 w-4 text-red-500" />;
    if (isHealthy) return <CheckCircle className="h-4 w-4 text-green-500" />;
    return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
  };

  const getStatusBadge = (isHealthy: boolean, circuitOpen: boolean) => {
    if (circuitOpen) {
      return <Badge variant="destructive">Circuit Open</Badge>;
    }
    if (isHealthy) {
      return <Badge variant="default" className="bg-green-500">Healthy</Badge>;
    }
    return <Badge variant="secondary">Degraded</Badge>;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Provider Health
            </CardTitle>
            <CardDescription>Real-time health status of LLM providers</CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => triggerHealthCheck.mutate(undefined)}
            disabled={triggerHealthCheck.isPending}
          >
            {triggerHealthCheck.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
            <span className="ml-2 hidden sm:inline">Check All</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : healthData?.providers && healthData.providers.length > 0 ? (
          <div className="space-y-3">
            {healthData.providers.map((provider) => (
              <div
                key={provider.provider_id}
                className="flex items-center justify-between p-3 rounded-lg border"
              >
                <div className="flex items-center gap-3">
                  {getHealthIcon(provider.is_healthy, provider.circuit_open)}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{provider.provider_name}</span>
                      {getStatusBadge(provider.is_healthy, provider.circuit_open)}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <span>{provider.provider_type}</span>
                      {provider.latency_ms && (
                        <span className="flex items-center gap-1">
                          <Zap className="h-3 w-3" />
                          {provider.latency_ms}ms
                        </span>
                      )}
                      {provider.consecutive_failures > 0 && (
                        <span className="text-red-500">
                          {provider.consecutive_failures} failures
                        </span>
                      )}
                      {provider.last_check && (
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {new Date(provider.last_check).toLocaleTimeString()}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {provider.circuit_open && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => resetCircuit.mutate(provider.provider_id)}
                      disabled={resetCircuit.isPending}
                    >
                      Reset Circuit
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => triggerHealthCheck.mutate(provider.provider_id)}
                    disabled={triggerHealthCheck.isPending}
                  >
                    <TestTube className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Activity className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No providers configured</p>
            <p className="text-sm">Add LLM providers to see health status</p>
          </div>
        )}

        {/* Health Legend */}
        <div className="mt-4 pt-4 border-t">
          <p className="text-xs text-muted-foreground mb-2">Status Legend:</p>
          <div className="flex flex-wrap gap-4 text-xs">
            <div className="flex items-center gap-1">
              <CheckCircle className="h-3 w-3 text-green-500" />
              <span>Healthy - Provider responding normally</span>
            </div>
            <div className="flex items-center gap-1">
              <AlertTriangle className="h-3 w-3 text-yellow-500" />
              <span>Degraded - High latency or errors</span>
            </div>
            <div className="flex items-center gap-1">
              <XCircle className="h-3 w-3 text-red-500" />
              <span>Circuit Open - Provider temporarily disabled</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Cost Alerts Card Component
// =============================================================================

function CostAlertsCard() {
  const { data: alertsData, isLoading, refetch } = useCostAlertsAdmin(false);
  const acknowledgeAlert = useAcknowledgeCostAlert();

  const getAlertIcon = (alertType: string) => {
    if (alertType === "daily") return <Clock className="h-4 w-4" />;
    return <DollarSign className="h-4 w-4" />;
  };

  const getThresholdBadge = (threshold: number) => {
    if (threshold >= 100) {
      return <Badge variant="destructive">Limit Reached</Badge>;
    }
    if (threshold >= 80) {
      return <Badge variant="default" className="bg-orange-500">Warning</Badge>;
    }
    return <Badge variant="secondary">{threshold}%</Badge>;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              Cost Alerts
            </CardTitle>
            <CardDescription>Monitor and manage cost limit alerts</CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : alertsData?.alerts && alertsData.alerts.length > 0 ? (
          <div className="space-y-3">
            {alertsData.alerts.map((alert) => (
              <div
                key={alert.id}
                className={`flex items-center justify-between p-3 rounded-lg border ${
                  !alert.acknowledged ? "bg-yellow-50 dark:bg-yellow-950/20 border-yellow-200 dark:border-yellow-800" : ""
                }`}
              >
                <div className="flex items-center gap-3">
                  {getAlertIcon(alert.alert_type)}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{alert.user_email}</span>
                      {getThresholdBadge(alert.threshold_percent)}
                      <Badge variant="outline" className="text-xs">
                        {alert.alert_type}
                      </Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      ${alert.usage_at_alert_usd.toFixed(2)} at trigger • {formatDate(alert.created_at)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {alert.acknowledged ? (
                    <span className="flex items-center gap-1 text-xs text-green-600">
                      <CheckCheck className="h-3 w-3" />
                      Acknowledged
                    </span>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => acknowledgeAlert.mutate(alert.id)}
                      disabled={acknowledgeAlert.isPending}
                    >
                      {acknowledgeAlert.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>
                          <CheckCheck className="h-4 w-4 mr-1" />
                          Acknowledge
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <DollarSign className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No cost alerts</p>
            <p className="text-sm">Alerts will appear when users approach their cost limits</p>
          </div>
        )}

        {/* Summary */}
        {alertsData?.alerts && alertsData.alerts.length > 0 && (
          <div className="mt-4 pt-4 border-t">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Total Alerts:</span>
              <span className="font-medium">{alertsData.total}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Unacknowledged:</span>
              <span className="font-medium text-yellow-600">
                {alertsData.alerts.filter(a => !a.acknowledged).length}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
