"use client";

import { useState, useEffect, useRef } from "react";
import { getErrorMessage } from "@/lib/errors";
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
  Scan,
  Volume2,
  Mic,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
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
  usePullOllamaModel,
  useDeleteOllamaModel,
  useOCRSettings,
  useUpdateOCRSettings,
  useDownloadOCRModels,
  useOCRModelInfo,
  useRedisStatus,
  useCeleryStatus,
  useInvalidateRedisCache,
  useTTSSettings,
  useUpdateTTSSettings,
  useCoquiModels,
  useDownloadCoquiModel,
  useDeleteCoquiModel,
  useSettingsPresets,
  useApplySettingsPreset,
} from "@/lib/api/hooks";
import type { LLMProvider, LLMProviderType, DatabaseConnectionType } from "@/lib/api/client";
import { useUser } from "@/lib/auth";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
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

  // Deleted Documents state
  const [deletedDocs, setDeletedDocs] = useState<Array<{
    id: string;
    name: string;
    file_type: string;
    file_size: number;
    deleted_at: string | null;
    created_at: string | null;
  }>>([]);
  const [deletedDocsTotal, setDeletedDocsTotal] = useState(0);
  const [deletedDocsPage, setDeletedDocsPage] = useState(1);
  const [deletedDocsLoading, setDeletedDocsLoading] = useState(false);
  const [deletedDocsError, setDeletedDocsError] = useState<string | null>(null);
  const [restoringDocId, setRestoringDocId] = useState<string | null>(null);
  const [hardDeletingDocId, setHardDeletingDocId] = useState<string | null>(null);
  // Bulk selection state for deleted documents
  const [selectedDeletedDocs, setSelectedDeletedDocs] = useState<Set<string>>(new Set());
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);
  const [isBulkRestoring, setIsBulkRestoring] = useState(false);

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

  // OCR Configuration hooks
  const { data: ocrData, isLoading: ocrLoading, refetch: refetchOCR } = useOCRSettings({ enabled: isAuthenticated });
  const updateOCRSettings = useUpdateOCRSettings();
  const downloadModels = useDownloadOCRModels();
  const [downloadingModels, setDownloadingModels] = useState(false);
  const [downloadResult, setDownloadResult] = useState<{ success: boolean; message: string } | null>(null);

  // Ollama Model Management state
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState("http://localhost:11434");
  const [newModelName, setNewModelName] = useState("");
  const [pullingModel, setPullingModel] = useState(false);
  const [pullResult, setPullResult] = useState<{ success: boolean; message: string } | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);

  // Ollama Model Management hooks
  const { data: ollamaLocalModels, isLoading: ollamaLocalModelsLoading, refetch: refetchOllamaModels } = useOllamaModels(
    ollamaBaseUrl,
    { enabled: isAuthenticated }
  );
  const pullOllamaModel = usePullOllamaModel();
  const deleteOllamaModel = useDeleteOllamaModel();

  // Settings Presets hooks
  const { data: presetsData, isLoading: presetsLoading } = useSettingsPresets({ enabled: isAuthenticated });
  const applyPreset = useApplySettingsPreset();
  const [applyingPreset, setApplyingPreset] = useState<string | null>(null);

  // Initialize local settings when data loads
  useEffect(() => {
    if (settingsData?.settings) {
      setLocalSettings(settingsData.settings);
      setHasChanges(false);
    }
  }, [settingsData]);

  // Fetch deleted documents
  const fetchDeletedDocs = async (page: number = 1) => {
    setDeletedDocsLoading(true);
    setDeletedDocsError(null);
    try {
      const { api } = await import("@/lib/api/client");
      const result = await api.listDeletedDocuments(page, 10);
      setDeletedDocs(result.documents.map(doc => ({
        id: doc.id,
        name: doc.name,
        file_type: doc.file_type,
        file_size: doc.file_size,
        deleted_at: doc.deleted_at || null,
        created_at: doc.created_at || null,
      })));
      setDeletedDocsTotal(result.total);
      setDeletedDocsPage(page);
    } catch (err) {
      const errorMsg = getErrorMessage(err, "Failed to fetch deleted documents");
      console.error("Failed to fetch deleted documents:", errorMsg, err);
      setDeletedDocsError(errorMsg);
    } finally {
      setDeletedDocsLoading(false);
    }
  };

  // Restore deleted document
  const handleRestoreDocument = async (documentId: string) => {
    setRestoringDocId(documentId);
    try {
      const { api } = await import("@/lib/api/client");
      await api.restoreDeletedDocument(documentId);
      // Refresh the list
      await fetchDeletedDocs(deletedDocsPage);
      // Also refresh database info to update counts
      refetchDbInfo();
    } catch (err) {
      console.error("Failed to restore document:", err);
      alert(`Failed to restore document: ${getErrorMessage(err)}`);
    } finally {
      setRestoringDocId(null);
    }
  };

  // Permanently delete a soft-deleted document
  const handleHardDeleteDocument = async (documentId: string, docName: string) => {
    if (!confirm(`Permanently delete "${docName}"? This cannot be undone.`)) {
      return;
    }
    setHardDeletingDocId(documentId);
    try {
      const { api } = await import("@/lib/api/client");
      await api.deleteDocument(documentId, true); // hard_delete=true
      // Refresh the list
      await fetchDeletedDocs(deletedDocsPage);
      // Also refresh database info to update counts
      refetchDbInfo();
    } catch (err) {
      console.error("Failed to permanently delete document:", err);
      alert(`Failed to delete: ${getErrorMessage(err)}`);
    } finally {
      setHardDeletingDocId(null);
    }
  };

  // Bulk restore selected deleted documents
  const handleBulkRestore = async () => {
    if (selectedDeletedDocs.size === 0) return;

    setIsBulkRestoring(true);
    const ids = Array.from(selectedDeletedDocs);
    let restoredCount = 0;
    let failedCount = 0;

    try {
      const { api } = await import("@/lib/api/client");
      for (const id of ids) {
        try {
          await api.restoreDeletedDocument(id);
          restoredCount++;
        } catch (err) {
          console.error(`Failed to restore document ${id}:`, err);
          failedCount++;
        }
      }

      if (failedCount > 0) {
        alert(`Restored ${restoredCount} documents. ${failedCount} failed.`);
      } else {
        alert(`Successfully restored ${restoredCount} documents.`);
      }

      // Clear selection and refresh
      setSelectedDeletedDocs(new Set());
      await fetchDeletedDocs(deletedDocsPage);
      refetchDbInfo();
    } catch (err) {
      console.error("Bulk restore failed:", err);
      alert(`Bulk restore failed: ${getErrorMessage(err)}`);
    } finally {
      setIsBulkRestoring(false);
    }
  };

  // Bulk permanently delete selected deleted documents
  const handleBulkPermanentDelete = async () => {
    if (selectedDeletedDocs.size === 0) return;

    if (!confirm(`Permanently delete ${selectedDeletedDocs.size} documents? This cannot be undone.`)) {
      return;
    }

    setIsBulkDeleting(true);
    const ids = Array.from(selectedDeletedDocs);
    let deletedCount = 0;
    let failedCount = 0;

    try {
      const { api } = await import("@/lib/api/client");
      for (const id of ids) {
        try {
          await api.deleteDocument(id, true); // hard_delete=true
          deletedCount++;
        } catch (err) {
          console.error(`Failed to permanently delete document ${id}:`, err);
          failedCount++;
        }
      }

      if (failedCount > 0) {
        alert(`Permanently deleted ${deletedCount} documents. ${failedCount} failed.`);
      } else {
        alert(`Successfully permanently deleted ${deletedCount} documents.`);
      }

      // Clear selection and refresh
      setSelectedDeletedDocs(new Set());
      await fetchDeletedDocs(deletedDocsPage);
      refetchDbInfo();
    } catch (err) {
      console.error("Bulk delete failed:", err);
      alert(`Bulk delete failed: ${getErrorMessage(err)}`);
    } finally {
      setIsBulkDeleting(false);
    }
  };

  // Toggle selection of a deleted document
  const toggleDeletedDocSelection = (docId: string) => {
    setSelectedDeletedDocs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(docId)) {
        newSet.delete(docId);
      } else {
        newSet.add(docId);
      }
      return newSet;
    });
  };

  // Toggle select all deleted documents
  const toggleSelectAllDeletedDocs = () => {
    if (selectedDeletedDocs.size === deletedDocs.length && deletedDocs.length > 0) {
      setSelectedDeletedDocs(new Set());
    } else {
      setSelectedDeletedDocs(new Set(deletedDocs.map(d => d.id)));
    }
  };

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
          {getErrorMessage(settingsError, "Unable to fetch settings. Please try again.")}
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

      {/* Tabs Navigation */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="flex flex-wrap h-auto gap-1 p-1">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <Server className="h-4 w-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="providers" className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            LLM Providers
          </TabsTrigger>
          <TabsTrigger value="database" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Database
          </TabsTrigger>
          <TabsTrigger value="rag" className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            RAG Features
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Security
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            Notifications
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="ocr" className="flex items-center gap-2">
            <Scan className="h-4 w-4" />
            OCR Configuration
          </TabsTrigger>
          <TabsTrigger value="models" className="flex items-center gap-2">
            <Cog className="h-4 w-4" />
            Model Configuration
          </TabsTrigger>
          <TabsTrigger value="jobqueue" className="flex items-center gap-2">
            <Workflow className="h-4 w-4" />
            Job Queue
          </TabsTrigger>
          <TabsTrigger value="generation" className="flex items-center gap-2">
            <PenTool className="h-4 w-4" />
            Document Generation
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
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
              {presetsLoading ? (
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

        {/* LLM Providers Tab */}
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
                        {ollamaLocalModels.chat_models.map((model: { name: string; parameter_size?: string; family?: string; size?: number }) => (
                          <div
                            key={model.name}
                            className="flex items-center justify-between p-3 rounded-lg border"
                          >
                            <div className="flex items-center gap-3">
                              <Bot className="h-4 w-4 text-primary" />
                              <div>
                                <p className="font-medium">{model.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {model.parameter_size && `${model.parameter_size} · `}
                                  {model.family && `${model.family} · `}
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
                        {ollamaLocalModels.embedding_models.map((model: { name: string; parameter_size?: string; size?: number }) => (
                          <div
                            key={model.name}
                            className="flex items-center justify-between p-3 rounded-lg border"
                          >
                            <div className="flex items-center gap-3">
                              <Database className="h-4 w-4 text-blue-500" />
                              <div>
                                <p className="font-medium">{model.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {model.parameter_size && `${model.parameter_size} · `}
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

        {/* Database Tab */}
        <TabsContent value="database" className="space-y-6">
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
                onChange={(e) => handleSettingChange("database.vector_dimensions", parseInt(e.target.value) || 1536)}
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
              onChange={(e) => handleSettingChange("database.max_results_per_query", parseInt(e.target.value) || 10)}
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

          {/* Deleted Documents */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Trash2 className="h-5 w-5" />
                Deleted Documents
              </CardTitle>
              <CardDescription>
                View and restore soft-deleted documents
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  Soft-deleted documents can be restored here. Hard-deleted documents are permanently removed.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fetchDeletedDocs(1)}
                  disabled={deletedDocsLoading}
                >
                  {deletedDocsLoading ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4 mr-2" />
                  )}
                  Load Deleted Docs
                </Button>
              </div>

              {deletedDocsError && (
                <div className="flex items-center gap-2 p-3 rounded-lg border border-destructive/50 bg-destructive/10 text-destructive">
                  <AlertCircle className="h-4 w-4 flex-shrink-0" />
                  <span className="text-sm">{deletedDocsError}</span>
                </div>
              )}

              {deletedDocsTotal > 0 && (
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium">
                    {deletedDocsTotal} deleted document{deletedDocsTotal !== 1 ? "s" : ""} found
                  </p>
                  {/* Bulk action bar - shown when items are selected */}
                  {selectedDeletedDocs.size > 0 && (
                    <div className="flex items-center gap-2 p-2 bg-muted rounded-md">
                      <span className="text-sm font-medium">{selectedDeletedDocs.size} selected</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleBulkRestore}
                        disabled={isBulkRestoring || isBulkDeleting}
                      >
                        {isBulkRestoring ? (
                          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        ) : (
                          <RotateCcw className="h-4 w-4 mr-1" />
                        )}
                        Restore Selected
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={handleBulkPermanentDelete}
                        disabled={isBulkRestoring || isBulkDeleting}
                      >
                        {isBulkDeleting ? (
                          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4 mr-1" />
                        )}
                        Delete Selected
                      </Button>
                    </div>
                  )}
                </div>
              )}

              {deletedDocsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : deletedDocs.length > 0 ? (
                <div className="space-y-2">
                  {/* Select all header */}
                  <div className="flex items-center gap-2 p-2 border-b">
                    <Checkbox
                      checked={selectedDeletedDocs.size === deletedDocs.length && deletedDocs.length > 0}
                      onCheckedChange={toggleSelectAllDeletedDocs}
                      disabled={isBulkRestoring || isBulkDeleting}
                    />
                    <span className="text-sm text-muted-foreground">
                      {selectedDeletedDocs.size === deletedDocs.length && deletedDocs.length > 0
                        ? "Deselect all"
                        : "Select all"}
                    </span>
                  </div>
                  {deletedDocs.map((doc) => (
                    <div
                      key={doc.id}
                      className={`flex items-center justify-between p-3 rounded-lg border ${
                        selectedDeletedDocs.has(doc.id) ? "bg-primary/5 border-primary/50" : "bg-muted/50"
                      }`}
                    >
                      <div className="flex items-center gap-3 min-w-0 flex-1">
                        <Checkbox
                          checked={selectedDeletedDocs.has(doc.id)}
                          onCheckedChange={() => toggleDeletedDocSelection(doc.id)}
                          disabled={isBulkRestoring || isBulkDeleting || restoringDocId === doc.id || hardDeletingDocId === doc.id}
                        />
                        <div className="flex flex-col min-w-0 flex-1">
                          <span className="font-medium truncate">{doc.name}</span>
                          <span className="text-xs text-muted-foreground">
                            {doc.file_type} • {(doc.file_size / 1024).toFixed(1)} KB
                            {doc.deleted_at && ` • Deleted: ${new Date(doc.deleted_at).toLocaleDateString()}`}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 ml-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleRestoreDocument(doc.id)}
                          disabled={restoringDocId === doc.id || hardDeletingDocId === doc.id || isBulkRestoring || isBulkDeleting}
                        >
                          {restoringDocId === doc.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              <RotateCcw className="h-4 w-4 mr-1" />
                              Restore
                            </>
                          )}
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => handleHardDeleteDocument(doc.id, doc.name)}
                          disabled={restoringDocId === doc.id || hardDeletingDocId === doc.id || isBulkRestoring || isBulkDeleting}
                        >
                          {hardDeletingDocId === doc.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              <Trash2 className="h-4 w-4 mr-1" />
                              Delete
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  ))}

                  {/* Pagination */}
                  {deletedDocsTotal > 10 && (
                    <div className="flex items-center justify-center gap-2 pt-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => fetchDeletedDocs(deletedDocsPage - 1)}
                        disabled={deletedDocsPage <= 1 || deletedDocsLoading}
                      >
                        Previous
                      </Button>
                      <span className="text-sm text-muted-foreground">
                        Page {deletedDocsPage} of {Math.ceil(deletedDocsTotal / 10)}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => fetchDeletedDocs(deletedDocsPage + 1)}
                        disabled={deletedDocsPage >= Math.ceil(deletedDocsTotal / 10) || deletedDocsLoading}
                      >
                        Next
                      </Button>
                    </div>
                  )}
                </div>
              ) : deletedDocsTotal === 0 && !deletedDocsLoading ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Trash2 className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No deleted documents found</p>
                  <p className="text-xs mt-1">Click &quot;Load Deleted Docs&quot; to check for deleted documents</p>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        {/* RAG Features Tab */}
        <TabsContent value="rag" className="space-y-6">
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
                  onChange={(e) => handleSettingChange("ai.summarization_threshold_pages", parseInt(e.target.value) || 10)}
                />
                <p className="text-xs text-muted-foreground">Summarize docs with more pages</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">KB Threshold</label>
                <Input
                  type="number"
                  placeholder="100"
                  value={localSettings["ai.summarization_threshold_kb"] as number ?? 100}
                  onChange={(e) => handleSettingChange("ai.summarization_threshold_kb", parseInt(e.target.value) || 500)}
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
                  onChange={(e) => handleSettingChange("ai.max_semantic_cache_entries", parseInt(e.target.value) || 1000)}
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
                  onChange={(e) => handleSettingChange("ai.query_expansion_count", parseInt(e.target.value) || 2)}
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
                  onChange={(e) => handleSettingChange("ai.sections_per_document", parseInt(e.target.value) || 5)}
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

      {/* Advanced RAG Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Advanced RAG Features
          </CardTitle>
          <CardDescription>Configure GraphRAG, Agentic RAG, Multimodal, and real-time features</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Retrieval Settings */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Search className="h-4 w-4" />
              Retrieval Settings
            </h4>
            <p className="text-sm text-muted-foreground">
              Configure how documents are retrieved and ranked for chat queries
            </p>

            <div className="grid gap-4 sm:grid-cols-2">
              {/* Top K Documents */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Documents to Retrieve</label>
                <Input
                  type="number"
                  min="3"
                  max="25"
                  value={localSettings["rag.top_k"] as number ?? 10}
                  onChange={(e) => handleSettingChange("rag.top_k", parseInt(e.target.value) || 5)}
                />
                <p className="text-xs text-muted-foreground">
                  How many documents to search (3-25). Higher = broader but slower.
                </p>
              </div>

              {/* Query Expansion Count */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Query Expansions</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  value={localSettings["rag.query_expansion_count"] as number ?? 3}
                  onChange={(e) => handleSettingChange("rag.query_expansion_count", parseInt(e.target.value) || 3)}
                />
                <p className="text-xs text-muted-foreground">
                  Query variations to try (1-5). More = better recall.
                </p>
              </div>

              {/* Similarity Threshold */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Similarity Threshold</label>
                <Input
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="0.9"
                  value={localSettings["rag.similarity_threshold"] as number ?? 0.4}
                  onChange={(e) => handleSettingChange("rag.similarity_threshold", parseFloat(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">
                  Minimum relevance score (0.1-0.9). Lower = more results.
                </p>
              </div>
            </div>

            {/* Reranking Toggle */}
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Result Reranking</p>
                <p className="text-sm text-muted-foreground">
                  Use cross-encoder AI to reorder results by semantic relevance (recommended)
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.rerank_results"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.rerank_results", e.target.checked)}
              />
            </div>
          </div>

          {/* GraphRAG */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Workflow className="h-4 w-4" />
              GraphRAG (Knowledge Graph)
            </h4>
            <p className="text-sm text-muted-foreground">
              Build knowledge graphs from documents for multi-hop reasoning
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable GraphRAG</p>
                <p className="text-sm text-muted-foreground">
                  Extract entities and relationships for graph-based retrieval
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.graphrag_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.graphrag_enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Graph Hops</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  value={localSettings["rag.graph_max_hops"] as number ?? 2}
                  onChange={(e) => handleSettingChange("rag.graph_max_hops", parseInt(e.target.value) || 2)}
                />
                <p className="text-xs text-muted-foreground">Traversal depth (1-5)</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Graph Weight</label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localSettings["rag.graph_weight"] as number ?? 0.3}
                  onChange={(e) => handleSettingChange("rag.graph_weight", parseFloat(e.target.value))}
                />
                <p className="text-xs text-muted-foreground">Hybrid search weight (0-1)</p>
              </div>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Entity Extraction on Upload</p>
                <p className="text-sm text-muted-foreground">
                  Extract entities when processing documents
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.entity_extraction_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.entity_extraction_enabled", e.target.checked)}
              />
            </div>
          </div>

          {/* Agentic RAG */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Bot className="h-4 w-4" />
              Agentic RAG
            </h4>
            <p className="text-sm text-muted-foreground">
              Use AI agents for complex multi-step queries (ReAct loop)
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Agentic RAG</p>
                <p className="text-sm text-muted-foreground">
                  Decompose complex queries into sub-questions
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.agentic_enabled"] as boolean ?? false}
                onChange={(e) => handleSettingChange("rag.agentic_enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Iterations</label>
                <Input
                  type="number"
                  min="1"
                  max="10"
                  value={localSettings["rag.agentic_max_iterations"] as number ?? 5}
                  onChange={(e) => handleSettingChange("rag.agentic_max_iterations", parseInt(e.target.value) || 5)}
                />
                <p className="text-xs text-muted-foreground">ReAct loop limit (1-10)</p>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Auto-detect Complexity</p>
                  <p className="text-sm text-muted-foreground">
                    Trigger agentic mode automatically
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["rag.auto_detect_complexity"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("rag.auto_detect_complexity", e.target.checked)}
                />
              </div>
            </div>
          </div>

          {/* Multimodal RAG */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Multimodal RAG
            </h4>
            <p className="text-sm text-muted-foreground">
              Process images, tables, and charts in documents
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Multimodal Processing</p>
                <p className="text-sm text-muted-foreground">
                  Caption images and extract tables during indexing
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.multimodal_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.multimodal_enabled", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Vision Provider</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["rag.vision_provider"] as string || "auto"}
                  onChange={(e) => handleSettingChange("rag.vision_provider", e.target.value)}
                >
                  <option value="auto">Auto (Free first)</option>
                  <option value="ollama">Ollama (Free - Local)</option>
                  <option value="openai">OpenAI (Paid)</option>
                  <option value="anthropic">Anthropic (Paid)</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Ollama Vision Model</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["rag.ollama_vision_model"] as string || "llava"}
                  onChange={(e) => handleSettingChange("rag.ollama_vision_model", e.target.value)}
                >
                  <option value="llava">LLaVA (Recommended)</option>
                  <option value="bakllava">BakLLaVA</option>
                  <option value="llava:13b">LLaVA 13B</option>
                </select>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["rag.caption_images"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("rag.caption_images", e.target.checked)}
                />
                <span className="text-sm">Caption Images</span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["rag.extract_tables"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("rag.extract_tables", e.target.checked)}
                />
                <span className="text-sm">Extract Tables</span>
              </div>
            </div>
          </div>

          {/* Real-Time Updates */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Real-Time Updates
            </h4>
            <p className="text-sm text-muted-foreground">
              Incremental indexing and content freshness tracking
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Incremental Indexing</p>
                <p className="text-sm text-muted-foreground">
                  Only update changed chunks when reprocessing
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.incremental_indexing"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.incremental_indexing", e.target.checked)}
              />
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Freshness Tracking</p>
                <p className="text-sm text-muted-foreground">
                  Flag stale content based on age
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.freshness_tracking"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.freshness_tracking", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Aging Threshold (days)</label>
                <Input
                  type="number"
                  min="7"
                  value={localSettings["rag.freshness_threshold_days"] as number ?? 30}
                  onChange={(e) => handleSettingChange("rag.freshness_threshold_days", parseInt(e.target.value) || 7)}
                />
                <p className="text-xs text-muted-foreground">Days until content is aging</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Stale Threshold (days)</label>
                <Input
                  type="number"
                  min="30"
                  value={localSettings["rag.stale_threshold_days"] as number ?? 90}
                  onChange={(e) => handleSettingChange("rag.stale_threshold_days", parseInt(e.target.value) || 90)}
                />
                <p className="text-xs text-muted-foreground">Days until content is stale</p>
              </div>
            </div>
          </div>

          {/* Query Suggestions */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Query Suggestions
            </h4>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Suggestions</p>
                <p className="text-sm text-muted-foreground">
                  Suggest follow-up questions after answers
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["rag.suggested_questions_enabled"] as boolean ?? true}
                onChange={(e) => handleSettingChange("rag.suggested_questions_enabled", e.target.checked)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Suggestions Count</label>
              <Input
                type="number"
                min="1"
                max="5"
                value={localSettings["rag.suggestions_count"] as number ?? 3}
                onChange={(e) => handleSettingChange("rag.suggestions_count", parseInt(e.target.value) || 3)}
              />
              <p className="text-xs text-muted-foreground">Number of suggestions (1-5)</p>
            </div>
          </div>
        </CardContent>
          </Card>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="space-y-6">
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
                onChange={(e) => handleSettingChange("security.session_timeout_minutes", parseInt(e.target.value) || 60)}
              />
            </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notifications" className="space-y-6">
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
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          {/* LLM Usage Analytics Card */}
          <UsageAnalyticsCard />

          {/* Provider Health Card */}
          <ProviderHealthCard />

          {/* Cost Alerts Card */}
          <CostAlertsCard />
        </TabsContent>

        {/* OCR Configuration Tab */}
        <TabsContent value="ocr" className="space-y-6">
          {ocrLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <>
              {/* OCR Provider Selection */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Scan className="h-5 w-5" />
                    OCR Provider Configuration
                  </CardTitle>
                  <CardDescription>
                    Configure OCR engine, models, and languages for document processing
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Provider Selection */}
                  <div className="space-y-2">
                    <Label htmlFor="ocr-provider">OCR Provider</Label>
                    <Select
                      value={ocrData?.settings?.['ocr.provider'] as string || 'paddleocr'}
                      onValueChange={(value) => {
                        updateOCRSettings.mutate({ 'ocr.provider': value });
                      }}
                    >
                      <SelectTrigger id="ocr-provider">
                        <SelectValue placeholder="Select OCR provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="paddleocr">
                          <div className="flex items-center gap-2">
                            <Sparkles className="h-4 w-4" />
                            <div>
                              <p className="font-medium">PaddleOCR</p>
                              <p className="text-xs text-muted-foreground">Deep learning-based, high accuracy</p>
                            </div>
                          </div>
                        </SelectItem>
                        <SelectItem value="tesseract">
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4" />
                            <div>
                              <p className="font-medium">Tesseract</p>
                              <p className="text-xs text-muted-foreground">Traditional OCR, fast and lightweight</p>
                            </div>
                          </div>
                        </SelectItem>
                        <SelectItem value="auto">
                          <div className="flex items-center gap-2">
                            <Zap className="h-4 w-4" />
                            <div>
                              <p className="font-medium">Auto (Try Both)</p>
                              <p className="text-xs text-muted-foreground">PaddleOCR with Tesseract fallback</p>
                            </div>
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* PaddleOCR Settings (conditional) */}
                  {ocrData?.settings?.['ocr.provider'] !== 'tesseract' && (
                    <>
                      {/* Model Variant */}
                      <div className="space-y-2">
                        <Label htmlFor="ocr-variant">Model Variant</Label>
                        <Select
                          value={ocrData?.settings?.['ocr.paddle.variant'] as string || 'server'}
                          onValueChange={(value) => {
                            updateOCRSettings.mutate({ 'ocr.paddle.variant': value });
                          }}
                        >
                          <SelectTrigger id="ocr-variant">
                            <SelectValue placeholder="Select model variant" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="server">
                              <div>
                                <p className="font-medium">Server (Accurate)</p>
                                <p className="text-xs text-muted-foreground">Higher accuracy, slower processing</p>
                              </div>
                            </SelectItem>
                            <SelectItem value="mobile">
                              <div>
                                <p className="font-medium">Mobile (Fast)</p>
                                <p className="text-xs text-muted-foreground">Faster processing, lower accuracy</p>
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Languages */}
                      <div className="space-y-2">
                        <Label>Languages</Label>
                        <div className="flex flex-wrap gap-2">
                          {['en', 'de', 'fr', 'es', 'zh', 'ja', 'ko', 'ar'].map((lang) => {
                            const labels: Record<string, string> = {
                              en: 'English',
                              de: 'German',
                              fr: 'French',
                              es: 'Spanish',
                              zh: 'Chinese',
                              ja: 'Japanese',
                              ko: 'Korean',
                              ar: 'Arabic',
                            };
                            const currentLanguages = (ocrData?.settings?.['ocr.paddle.languages'] as string[]) || ['en', 'de'];
                            const isSelected = currentLanguages.includes(lang);

                            return (
                              <Button
                                key={lang}
                                variant={isSelected ? 'default' : 'outline'}
                                size="sm"
                                onClick={() => {
                                  const newLanguages = isSelected
                                    ? currentLanguages.filter(l => l !== lang)
                                    : [...currentLanguages, lang];
                                  updateOCRSettings.mutate({ 'ocr.paddle.languages': newLanguages });
                                }}
                              >
                                {labels[lang]}
                              </Button>
                            );
                          })}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          Selected: {((ocrData?.settings?.['ocr.paddle.languages'] as string[]) || []).join(', ')}
                        </p>
                      </div>

                      {/* Auto Download Toggle */}
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label htmlFor="auto-download">Auto-Download Models</Label>
                          <p className="text-sm text-muted-foreground">
                            Automatically download missing models on startup
                          </p>
                        </div>
                        <Switch
                          id="auto-download"
                          checked={ocrData?.settings?.['ocr.paddle.auto_download'] as boolean || true}
                          onCheckedChange={(checked) => {
                            updateOCRSettings.mutate({ 'ocr.paddle.auto_download': checked });
                          }}
                        />
                      </div>
                    </>
                  )}

                  {/* Tesseract Fallback */}
                  {ocrData?.settings?.['ocr.provider'] === 'paddleocr' && (
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label htmlFor="tesseract-fallback">Tesseract Fallback</Label>
                        <p className="text-sm text-muted-foreground">
                          Fall back to Tesseract if PaddleOCR fails
                        </p>
                      </div>
                      <Switch
                        id="tesseract-fallback"
                        checked={ocrData?.settings?.['ocr.tesseract.fallback_enabled'] as boolean || true}
                        onCheckedChange={(checked) => {
                          updateOCRSettings.mutate({ 'ocr.tesseract.fallback_enabled': checked });
                        }}
                      />
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Model Management */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <HardDrive className="h-5 w-5" />
                    Downloaded Models
                  </CardTitle>
                  <CardDescription>
                    Manage PaddleOCR model downloads and storage
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Model Status */}
                  <div className="grid gap-4">
                    <div className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center gap-3">
                        <Database className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <p className="font-medium">Model Directory</p>
                          <p className="text-sm text-muted-foreground font-mono">
                            {ocrData?.models?.model_dir || './data/paddle_models'}
                          </p>
                        </div>
                      </div>
                      <Badge variant={ocrData?.models?.status === 'installed' ? 'default' : 'secondary'}>
                        {ocrData?.models?.status || 'unknown'}
                      </Badge>
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-lg border">
                      <div className="flex items-center gap-3">
                        <HardDrive className="h-5 w-5 text-muted-foreground" />
                        <div>
                          <p className="font-medium">Total Size</p>
                          <p className="text-sm text-muted-foreground">
                            {ocrData?.models?.total_size || '0 MB'}
                          </p>
                        </div>
                      </div>
                      <Badge variant="outline">
                        {ocrData?.models?.downloaded?.length || 0} models
                      </Badge>
                    </div>
                  </div>

                  {/* Downloaded Models List */}
                  {ocrData?.models?.downloaded && ocrData.models.downloaded.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Model Files:</p>
                      <div className="space-y-1 max-h-60 overflow-y-auto">
                        {ocrData.models.downloaded.map((model, idx) => (
                          <div
                            key={idx}
                            className="flex items-center justify-between p-2 rounded border text-sm"
                          >
                            <div className="flex items-center gap-2">
                              <CheckCircle className="h-4 w-4 text-green-500" />
                              <span className="font-mono text-xs">{model.name}</span>
                            </div>
                            <span className="text-muted-foreground text-xs">{model.size}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Download Button */}
                  <div className="pt-4 border-t">
                    <Button
                      onClick={async () => {
                        setDownloadingModels(true);
                        setDownloadResult(null);
                        try {
                          const result = await downloadModels.mutateAsync({
                            languages: (ocrData?.settings?.['ocr.paddle.languages'] as string[]) || ['en', 'de'],
                            variant: (ocrData?.settings?.['ocr.paddle.variant'] as string) || 'server',
                          });
                          setDownloadResult({
                            success: result.status === 'success',
                            message: `Downloaded ${result.downloaded.length} languages successfully`,
                          });
                          refetchOCR();
                        } catch (error) {
                          setDownloadResult({
                            success: false,
                            message: getErrorMessage(error),
                          });
                        } finally {
                          setDownloadingModels(false);
                        }
                      }}
                      disabled={downloadingModels}
                      className="w-full"
                    >
                      {downloadingModels ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Downloading Models...
                        </>
                      ) : (
                        <>
                          <Download className="h-4 w-4 mr-2" />
                          Download Selected Models
                        </>
                      )}
                    </Button>

                    {downloadResult && (
                      <Alert className="mt-3">
                        {downloadResult.success ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <AlertCircle className="h-4 w-4 text-red-500" />
                        )}
                        <AlertDescription>{downloadResult.message}</AlertDescription>
                      </Alert>
                    )}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Model Configuration Tab */}
        <TabsContent value="models" className="space-y-6">
          <ModelConfigurationSection providers={providersData?.providers || []} />
        </TabsContent>

        {/* Job Queue Tab */}
        <TabsContent value="jobqueue" className="space-y-6">
          <JobQueueSettings
            localSettings={localSettings}
            handleSettingChange={handleSettingChange}
          />
        </TabsContent>

        {/* Document Generation Tab */}
        <TabsContent value="generation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <PenTool className="h-5 w-5" />
                Document Generation Settings
              </CardTitle>
              <CardDescription>
                Configure AI-powered document and presentation generation features
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Output Settings */}
              <div className="space-y-4">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Output Settings
                </h4>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="flex items-center justify-between p-3 rounded-lg border">
                    <div>
                      <p className="font-medium">Include Sources</p>
                      <p className="text-sm text-muted-foreground">
                        Add source citations to documents
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      className="h-4 w-4"
                      checked={localSettings["generation.include_sources"] as boolean ?? true}
                      onChange={(e) => handleSettingChange("generation.include_sources", e.target.checked)}
                    />
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg border">
                    <div>
                      <p className="font-medium">Include Images</p>
                      <p className="text-sm text-muted-foreground">
                        Auto-generate relevant images
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      className="h-4 w-4"
                      checked={localSettings["generation.include_images"] as boolean ?? true}
                      onChange={(e) => handleSettingChange("generation.include_images", e.target.checked)}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Image Backend</label>
                  <select
                    className="w-full h-10 px-3 rounded-md border bg-background"
                    value={localSettings["generation.image_backend"] as string || "picsum"}
                    onChange={(e) => handleSettingChange("generation.image_backend", e.target.value)}
                  >
                    <option value="picsum">Picsum (Free - No API Key)</option>
                    <option value="unsplash">Unsplash (Requires API Key)</option>
                    <option value="pexels">Pexels (Requires API Key)</option>
                    <option value="openai">OpenAI DALL-E (Requires API Key)</option>
                    <option value="stability">Stability AI (Requires API Key)</option>
                    <option value="automatic1111">Automatic1111 (Local SD)</option>
                    <option value="disabled">Disabled</option>
                  </select>
                  <p className="text-xs text-muted-foreground">Source for auto-generated images</p>
                </div>
              </div>

              {/* Quality Review Settings */}
              <div className="space-y-4 pt-4 border-t">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <CheckCircle className="h-4 w-4" />
                  Quality Review
                </h4>
                <p className="text-sm text-muted-foreground">
                  LLM-based review of generated content for quality assurance
                </p>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <div>
                    <p className="font-medium">Enable Quality Review</p>
                    <p className="text-sm text-muted-foreground">
                      Use LLM to review generated content quality
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={localSettings["generation.enable_quality_review"] as boolean ?? true}
                    onChange={(e) => handleSettingChange("generation.enable_quality_review", e.target.checked)}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Minimum Quality Score</label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    placeholder="0.7"
                    value={localSettings["generation.min_quality_score"] as number ?? 0.7}
                    onChange={(e) => handleSettingChange("generation.min_quality_score", parseFloat(e.target.value))}
                  />
                  <p className="text-xs text-muted-foreground">Score threshold (0-1) for content to pass review</p>
                </div>
              </div>

              {/* Vision Analysis Settings */}
              <div className="space-y-4 pt-4 border-t">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Eye className="h-4 w-4" />
                  Vision Analysis (PPTX)
                </h4>
                <p className="text-sm text-muted-foreground">
                  Use vision models to analyze templates and review generated slides for layout issues
                </p>

                {/* Template Vision Analysis */}
                <div className="space-y-3 p-3 rounded-lg border bg-muted/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Template Vision Analysis</p>
                      <p className="text-sm text-muted-foreground">
                        Analyze template slides to learn visual styling and layout
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      className="h-4 w-4"
                      checked={localSettings["generation.enable_template_vision_analysis"] as boolean ?? false}
                      onChange={(e) => handleSettingChange("generation.enable_template_vision_analysis", e.target.checked)}
                    />
                  </div>
                  <div className="space-y-2 pt-2">
                    <label className="text-sm font-medium">Template Vision Model</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={localSettings["generation.template_vision_model"] as string || "auto"}
                      onChange={(e) => handleSettingChange("generation.template_vision_model", e.target.value)}
                    >
                      <option value="auto">Auto (Use default vision model)</option>
                      <optgroup label="OpenAI (Vision-capable)">
                        {providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                          <>
                            <option value="gpt-4o">GPT-4o</option>
                            <option value="gpt-4o-mini">GPT-4o Mini</option>
                            <option value="gpt-4-vision-preview">GPT-4 Vision</option>
                          </>
                        )}
                        {!providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                          <option disabled>No OpenAI provider configured</option>
                        )}
                      </optgroup>
                      <optgroup label="Anthropic (Vision-capable)">
                        {providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                          <>
                            <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                            <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                            <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
                            <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                          </>
                        )}
                        {!providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                          <option disabled>No Anthropic provider configured</option>
                        )}
                      </optgroup>
                      <optgroup label="Ollama (Local - Downloaded)">
                        {ollamaLocalModels?.chat_models?.filter(m =>
                          m.name.includes("llava") || m.name.includes("bakllava") ||
                          m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                          m.name.includes("minicpm-v")
                        ).map(model => (
                          <option key={model.name} value={model.name}>
                            {model.name} {model.parameter_size && `(${model.parameter_size})`}
                          </option>
                        ))}
                        {(!ollamaLocalModels?.chat_models || ollamaLocalModels.chat_models.filter(m =>
                          m.name.includes("llava") || m.name.includes("bakllava") ||
                          m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                          m.name.includes("minicpm-v")
                        ).length === 0) && (
                          <option disabled>No vision models downloaded (try: ollama pull llava)</option>
                        )}
                      </optgroup>
                    </select>
                    <p className="text-xs text-muted-foreground">Vision model for analyzing template styling</p>
                  </div>
                </div>

                {/* Slide Review */}
                <div className="space-y-3 p-3 rounded-lg border bg-muted/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Vision-Based Slide Review</p>
                      <p className="text-sm text-muted-foreground">
                        Review generated slides for visual issues (overlaps, truncation)
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      className="h-4 w-4"
                      checked={localSettings["generation.enable_vision_review"] as boolean ?? false}
                      onChange={(e) => handleSettingChange("generation.enable_vision_review", e.target.checked)}
                    />
                  </div>
                  <div className="space-y-3 pt-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Vision Review Model</label>
                      <select
                        className="w-full h-10 px-3 rounded-md border bg-background"
                        value={localSettings["generation.vision_review_model"] as string || "auto"}
                        onChange={(e) => handleSettingChange("generation.vision_review_model", e.target.value)}
                      >
                        <option value="auto">Auto (Use default vision model)</option>
                        <optgroup label="OpenAI (Vision-capable)">
                          {providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                            <>
                              <option value="gpt-4o">GPT-4o</option>
                              <option value="gpt-4o-mini">GPT-4o Mini</option>
                              <option value="gpt-4-vision-preview">GPT-4 Vision</option>
                            </>
                          )}
                          {!providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                            <option disabled>No OpenAI provider configured</option>
                          )}
                        </optgroup>
                        <optgroup label="Anthropic (Vision-capable)">
                          {providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                            <>
                              <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                              <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                              <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
                              <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                            </>
                          )}
                          {!providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                            <option disabled>No Anthropic provider configured</option>
                          )}
                        </optgroup>
                        <optgroup label="Ollama (Local - Downloaded)">
                          {ollamaLocalModels?.chat_models?.filter(m =>
                            m.name.includes("llava") || m.name.includes("bakllava") ||
                            m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                            m.name.includes("minicpm-v")
                          ).map(model => (
                            <option key={model.name} value={model.name}>
                              {model.name} {model.parameter_size && `(${model.parameter_size})`}
                            </option>
                          ))}
                          {(!ollamaLocalModels?.chat_models || ollamaLocalModels.chat_models.filter(m =>
                            m.name.includes("llava") || m.name.includes("bakllava") ||
                            m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                            m.name.includes("minicpm-v")
                          ).length === 0) && (
                            <option disabled>No vision models downloaded (try: ollama pull llava)</option>
                          )}
                        </optgroup>
                      </select>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">Review All Slides</p>
                        <p className="text-xs text-muted-foreground">
                          If off, only reviews slides at risk of issues
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        className="h-4 w-4"
                        checked={localSettings["generation.vision_review_all_slides"] as boolean ?? false}
                        onChange={(e) => handleSettingChange("generation.vision_review_all_slides", e.target.checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Per-Slide Constraints */}
              <div className="space-y-4 pt-4 border-t">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Content Fitting
                </h4>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <div>
                    <p className="font-medium">Per-Slide Constraints</p>
                    <p className="text-sm text-muted-foreground">
                      Learn layout constraints from template for each slide type
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={localSettings["generation.enable_per_slide_constraints"] as boolean ?? true}
                    onChange={(e) => handleSettingChange("generation.enable_per_slide_constraints", e.target.checked)}
                  />
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <div>
                    <p className="font-medium">LLM Content Rewriting</p>
                    <p className="text-sm text-muted-foreground">
                      Use LLM to condense overflowing content intelligently
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={localSettings["generation.enable_llm_rewrite"] as boolean ?? true}
                    onChange={(e) => handleSettingChange("generation.enable_llm_rewrite", e.target.checked)}
                  />
                </div>
              </div>

              {/* Chart Generation */}
              <div className="space-y-4 pt-4 border-t">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Chart Generation
                </h4>
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <div>
                    <p className="font-medium">Auto-Generate Charts</p>
                    <p className="text-sm text-muted-foreground">
                      Automatically create charts from data in content
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={localSettings["generation.auto_charts"] as boolean ?? false}
                    onChange={(e) => handleSettingChange("generation.auto_charts", e.target.checked)}
                  />
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Chart Style</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={localSettings["generation.chart_style"] as string || "business"}
                      onChange={(e) => handleSettingChange("generation.chart_style", e.target.value)}
                    >
                      <option value="business">Business (Professional)</option>
                      <option value="modern">Modern (Clean lines)</option>
                      <option value="minimal">Minimal (Simple)</option>
                      <option value="colorful">Colorful (Vibrant)</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Chart DPI</label>
                    <Input
                      type="number"
                      min="72"
                      max="300"
                      placeholder="150"
                      value={localSettings["generation.chart_dpi"] as number ?? 150}
                      onChange={(e) => handleSettingChange("generation.chart_dpi", parseInt(e.target.value) || 150)}
                    />
                    <p className="text-xs text-muted-foreground">Resolution for chart images</p>
                  </div>
                </div>
              </div>

              {/* Style Defaults */}
              <div className="space-y-4 pt-4 border-t">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <PenTool className="h-4 w-4" />
                  Style Defaults
                </h4>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Default Tone</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={localSettings["generation.default_tone"] as string || "professional"}
                      onChange={(e) => handleSettingChange("generation.default_tone", e.target.value)}
                    >
                      <option value="professional">Professional</option>
                      <option value="casual">Casual</option>
                      <option value="technical">Technical</option>
                      <option value="friendly">Friendly</option>
                      <option value="academic">Academic</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Default Style</label>
                    <select
                      className="w-full h-10 px-3 rounded-md border bg-background"
                      value={localSettings["generation.default_style"] as string || "business"}
                      onChange={(e) => handleSettingChange("generation.default_style", e.target.value)}
                    >
                      <option value="business">Business</option>
                      <option value="creative">Creative</option>
                      <option value="minimal">Minimal</option>
                      <option value="bold">Bold</option>
                    </select>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

      </Tabs>

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
  audio_script: <Volume2 className="h-4 w-4" />,
  knowledge_graph: <Scan className="h-4 w-4" />,
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
  audio_script: "Audio Script Generation",
  knowledge_graph: "Knowledge Graph",
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

// Job Queue Settings Component
function JobQueueSettings({
  localSettings,
  handleSettingChange,
}: {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}) {
  const { data: redisStatus, isLoading: redisLoading } = useRedisStatus();
  const { data: celeryStatus, isLoading: celeryLoading } = useCeleryStatus();
  const invalidateCache = useInvalidateRedisCache();

  return (
    <>
      {/* Connection Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Connection Status
          </CardTitle>
          <CardDescription>
            Real-time status of Redis and Celery connections
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            {/* Redis Status */}
            <div className="p-4 rounded-lg border">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Redis</span>
                {redisLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <div className="flex items-center gap-2">
                    <div
                      className={`h-2 w-2 rounded-full ${
                        redisStatus?.connected
                          ? "bg-green-500"
                          : redisStatus?.enabled
                          ? "bg-yellow-500"
                          : "bg-gray-400"
                      }`}
                    />
                    <span className="text-sm">
                      {redisStatus?.connected
                        ? "Connected"
                        : redisStatus?.enabled
                        ? "Disconnected"
                        : "Disabled"}
                    </span>
                  </div>
                )}
              </div>
              {redisStatus?.url && (
                <p className="text-xs text-muted-foreground">{redisStatus.url}</p>
              )}
              {redisStatus?.reason && !redisStatus?.connected && (
                <p className="text-xs text-yellow-600 mt-1">{redisStatus.reason}</p>
              )}
            </div>

            {/* Celery Status */}
            <div className="p-4 rounded-lg border">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Celery Workers</span>
                {celeryLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <div className="flex items-center gap-2">
                    <div
                      className={`h-2 w-2 rounded-full ${
                        celeryStatus?.available
                          ? "bg-green-500"
                          : celeryStatus?.enabled
                          ? "bg-yellow-500"
                          : "bg-gray-400"
                      }`}
                    />
                    <span className="text-sm">
                      {celeryStatus?.available
                        ? `${celeryStatus.worker_count} workers`
                        : celeryStatus?.enabled
                        ? "No workers"
                        : "Disabled"}
                    </span>
                  </div>
                )}
              </div>
              {celeryStatus?.active_tasks !== undefined && celeryStatus.active_tasks > 0 && (
                <p className="text-xs text-muted-foreground">
                  {celeryStatus.active_tasks} active task(s)
                </p>
              )}
              {celeryStatus?.message && (
                <p className="text-xs text-yellow-600 mt-1">{celeryStatus.message}</p>
              )}
            </div>
          </div>

          <div className="mt-4">
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
              Apply Settings Changes
            </Button>
            <p className="text-xs text-muted-foreground mt-1">
              Click after saving settings to apply Redis/Celery configuration changes
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Celery Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Workflow className="h-5 w-5" />
            Job Queue Configuration
          </CardTitle>
          <CardDescription>
            Configure Celery for async document processing (requires Redis)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Enable Celery Toggle */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Celery Job Queue</p>
              <p className="text-sm text-muted-foreground">
                Process documents asynchronously with Celery workers
              </p>
            </div>
            <Switch
              checked={(localSettings["queue.celery_enabled"] as boolean) ?? false}
              onCheckedChange={(checked) =>
                handleSettingChange("queue.celery_enabled", checked)
              }
            />
          </div>

          {/* Redis URL (shown when Celery enabled) */}
          {Boolean(localSettings["queue.celery_enabled"]) && (
            <div className="space-y-2">
              <Label htmlFor="redis-url">Redis URL</Label>
              <Input
                id="redis-url"
                value={
                  (localSettings["queue.redis_url"] as string) ??
                  "redis://localhost:6379/0"
                }
                onChange={(e) =>
                  handleSettingChange("queue.redis_url", e.target.value)
                }
                placeholder="redis://localhost:6379/0"
              />
              <p className="text-xs text-muted-foreground">
                Redis connection URL for job queue and caching
              </p>
            </div>
          )}

          {/* Max Workers */}
          {Boolean(localSettings["queue.celery_enabled"]) && (
            <div className="space-y-2">
              <Label htmlFor="max-workers">Max Celery Workers</Label>
              <Input
                id="max-workers"
                type="number"
                min={1}
                max={16}
                value={(localSettings["queue.max_workers"] as number) ?? 4}
                onChange={(e) =>
                  handleSettingChange("queue.max_workers", parseInt(e.target.value) || 4)
                }
              />
              <p className="text-xs text-muted-foreground">
                Maximum concurrent workers for document processing
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Embedding Cache Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Embedding Cache
          </CardTitle>
          <CardDescription>
            Cache embeddings to reduce API calls and speed up processing
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Enable Cache Toggle */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Embedding Cache</p>
              <p className="text-sm text-muted-foreground">
                Cache embeddings to reduce API calls (uses Redis if available, falls back to memory)
              </p>
            </div>
            <Switch
              checked={(localSettings["cache.embedding_cache_enabled"] as boolean) ?? true}
              onCheckedChange={(checked) =>
                handleSettingChange("cache.embedding_cache_enabled", checked)
              }
            />
          </div>

          {/* Cache TTL */}
          {(localSettings["cache.embedding_cache_enabled"] as boolean) !== false && (
            <div className="space-y-2">
              <Label htmlFor="cache-ttl">Cache TTL (days)</Label>
              <Input
                id="cache-ttl"
                type="number"
                min={1}
                max={30}
                value={(localSettings["cache.embedding_cache_ttl_days"] as number) ?? 7}
                onChange={(e) =>
                  handleSettingChange(
                    "cache.embedding_cache_ttl_days",
                    parseInt(e.target.value) || 7
                  )
                }
              />
              <p className="text-xs text-muted-foreground">
                How long to cache embeddings before they expire
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}

// =============================================================================
// Model Configuration Section
// =============================================================================

function ModelConfigurationSection({ providers }: { providers: LLMProvider[] }) {
  const [activeTab, setActiveTab] = useState<"llm" | "tts">("llm");

  return (
    <div className="space-y-6">
      {/* Sub-tabs for LLM Config and TTS */}
      <div className="flex gap-2 border-b pb-2">
        <Button
          variant={activeTab === "llm" ? "default" : "ghost"}
          size="sm"
          onClick={() => setActiveTab("llm")}
          className="flex items-center gap-2"
        >
          <MessageSquare className="h-4 w-4" />
          LLM Configuration
        </Button>
        <Button
          variant={activeTab === "tts" ? "default" : "ghost"}
          size="sm"
          onClick={() => setActiveTab("tts")}
          className="flex items-center gap-2"
        >
          <Volume2 className="h-4 w-4" />
          Text-to-Speech
        </Button>
      </div>

      {activeTab === "llm" && (
        <OperationLevelConfigCard providers={providers} />
      )}

      {activeTab === "tts" && (
        <TTSConfigurationSection />
      )}
    </div>
  );
}

// TTS Configuration Section with default provider and Coqui model management
function TTSConfigurationSection() {
  const { data: ttsSettings, isLoading: settingsLoading } = useTTSSettings();
  const { data: coquiModels, isLoading: modelsLoading } = useCoquiModels();
  const updateSettings = useUpdateTTSSettings();
  const downloadModel = useDownloadCoquiModel();
  const deleteModel = useDeleteCoquiModel();

  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);

  // Update selected provider when settings load
  useEffect(() => {
    if (ttsSettings?.default_provider && !selectedProvider) {
      setSelectedProvider(ttsSettings.default_provider);
    }
  }, [ttsSettings, selectedProvider]);

  const handleProviderChange = async (provider: string) => {
    setSelectedProvider(provider);
    try {
      await updateSettings.mutateAsync(provider);
    } catch (error) {
      console.error("Failed to update TTS provider:", error);
    }
  };

  const handleDownloadModel = async (modelName: string) => {
    setDownloadingModel(modelName);
    try {
      await downloadModel.mutateAsync(modelName);
    } catch (error) {
      console.error("Failed to download model:", error);
    } finally {
      setDownloadingModel(null);
    }
  };

  const handleDeleteModel = async (modelName: string) => {
    setDeletingModel(modelName);
    try {
      await deleteModel.mutateAsync(modelName);
    } catch (error) {
      console.error("Failed to delete model:", error);
    } finally {
      setDeletingModel(null);
    }
  };

  if (settingsLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Default TTS Provider */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Mic className="h-5 w-5" />
            Default TTS Provider
          </CardTitle>
          <CardDescription>
            Select the default text-to-speech provider for audio overview generation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Provider</Label>
            <Select
              value={selectedProvider}
              onValueChange={handleProviderChange}
              disabled={updateSettings.isPending}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a TTS provider" />
              </SelectTrigger>
              <SelectContent>
                {ttsSettings?.available_providers?.map((provider) => (
                  <SelectItem key={provider.id} value={provider.id}>
                    <div className="flex items-center gap-2">
                      <span>{provider.name}</span>
                      {!provider.requires_api_key && (
                        <Badge variant="secondary" className="text-xs">Free</Badge>
                      )}
                      {!provider.is_available && (
                        <Badge variant="destructive" className="text-xs">Unavailable</Badge>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Provider Info */}
          {selectedProvider && ttsSettings?.available_providers && (
            <div className="p-3 rounded-lg bg-muted/50 space-y-2">
              {(() => {
                const provider = ttsSettings.available_providers.find(p => p.id === selectedProvider);
                if (!provider) return null;
                return (
                  <>
                    <p className="text-sm">{provider.description}</p>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>Cost: {provider.cost}</span>
                      {provider.requires_api_key && (
                        <span className="flex items-center gap-1">
                          <Key className="h-3 w-3" />
                          API Key Required
                        </span>
                      )}
                    </div>
                  </>
                );
              })()}
            </div>
          )}

          {updateSettings.isPending && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Saving...
            </div>
          )}
        </CardContent>
      </Card>

      {/* Coqui TTS Model Management - only show if Coqui is available */}
      {ttsSettings?.available_providers?.find(p => p.id === "coqui")?.is_available && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <HardDrive className="h-5 w-5" />
              Coqui TTS Models
            </CardTitle>
            <CardDescription>
              Manage locally installed Coqui TTS models for offline text-to-speech
            </CardDescription>
          </CardHeader>
          <CardContent>
            {modelsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : coquiModels && coquiModels.length > 0 ? (
              <div className="space-y-3">
                {coquiModels.map((model) => (
                  <div
                    key={model.name}
                    className="flex items-center justify-between p-3 rounded-lg border"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{model.name}</p>
                        <Badge variant={model.is_installed ? "default" : "secondary"}>
                          {model.is_installed ? "Installed" : "Available"}
                        </Badge>
                        <Badge variant="outline">{model.language}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{model.description}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Size: {model.size_mb} MB
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      {model.is_installed ? (
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => handleDeleteModel(model.name)}
                          disabled={deletingModel === model.name}
                        >
                          {deletingModel === model.name ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDownloadModel(model.name)}
                          disabled={downloadingModel === model.name}
                        >
                          {downloadingModel === model.name ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Download className="h-4 w-4" />
                          )}
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>No Coqui models available</p>
                <p className="text-sm mt-1">
                  Install Coqui TTS to enable local text-to-speech models
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Info about TTS options */}
      <Alert>
        <Volume2 className="h-4 w-4" />
        <AlertDescription>
          <strong>TTS Provider Options:</strong>
          <ul className="mt-2 space-y-1 text-sm list-disc list-inside">
            <li><strong>Edge TTS</strong> - Free Microsoft voices, no API key required</li>
            <li><strong>OpenAI TTS</strong> - High quality, requires API key</li>
            <li><strong>ElevenLabs</strong> - Premium realistic voices, requires API key</li>
            <li><strong>Coqui TTS</strong> - Local/offline, requires GPU for best performance</li>
          </ul>
        </AlertDescription>
      </Alert>
    </div>
  );
}
