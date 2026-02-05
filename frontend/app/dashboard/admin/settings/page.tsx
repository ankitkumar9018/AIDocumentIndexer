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
  Square,
  Scan,
  Volume2,
  Mic,
  GitBranch,
  Cpu,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { EmbeddingDashboard } from "@/components/embedding-dashboard";
import { OverviewTab } from "./components/overview-tab";
import { SecurityTab } from "./components/security-tab";
import { NotificationsTab } from "./components/notifications-tab";
import { RagTab } from "./components/rag-tab";
import { AnalyticsTab } from "./components/analytics-tab";
import { ModelsTab } from "./components/models-tab";
import { JobQueueTab } from "./components/jobqueue-tab";
import { ProvidersTab } from "./components/providers-tab";
import { DatabaseTab } from "./components/database-tab";
import { OcrTab } from "./components/ocr-tab";
import { GenerationTab } from "./components/generation-tab";
import { ExperimentsTab } from "./components/experiments-tab";
import { CacheTab } from "./components/cache-tab";
import { EvaluationTab } from "./components/evaluation-tab";
import { AudioTab } from "./components/audio-tab";
import { ScraperTab } from "./components/scraper-tab";
import { IngestionTab } from "./components/ingestion-tab";
import { InstructionsTab } from "./components/instructions-tab";
import { RayTab } from "./components/ray-tab";
import { MaintenanceTab } from "./components/maintenance-tab";
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
  useVlmConfig,
  useUpdateVlmConfig,
  useTestVlm,
  useRedisStatus,
  useCeleryStatus,
  useInvalidateRedisCache,
  useStartCeleryWorker,
  useStopCeleryWorker,
  useTTSSettings,
  useUpdateTTSSettings,
  useCoquiModels,
  useDownloadCoquiModel,
  useDeleteCoquiModel,
  useSettingsPresets,
  useApplySettingsPreset,
} from "@/lib/api/hooks";
import { api } from "@/lib/api/client";
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
import { Slider } from "@/components/ui/slider";
import { calculateOptimizedTemperature } from "@/lib/utils";
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

  // Hardware auto-detect state
  const [detectingQueue, setDetectingQueue] = useState(false);

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

  // VLM Configuration hooks
  const { data: vlmConfig, isLoading: vlmLoading } = useVlmConfig({ enabled: isAuthenticated });
  const updateVlmConfig = useUpdateVlmConfig();
  const testVlm = useTestVlm();

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

  const detectQueueWorkers = async () => {
    setDetectingQueue(true);
    try {
      const { data } = await api.get<{ recommended: Record<string, number> }>("/diagnostics/hardware/recommended-settings");
      const recommended = data.recommended?.["queue.max_workers"];
      if (recommended !== undefined) {
        handleSettingChange("queue.max_workers", recommended);
      }
    } catch {
      // Silently fail
    } finally {
      setDetectingQueue(false);
    }
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
      <Tabs defaultValue="overview" orientation="vertical" className="flex gap-6 min-h-[600px]">
        {/* Vertical Sidebar Navigation */}
        <div className="hidden md:block w-56 shrink-0">
          <nav className="sticky top-4 space-y-6">
            {/* SYSTEM */}
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-3">System</p>
              <TabsList className="flex flex-col h-auto w-full bg-transparent gap-0.5">
                <TabsTrigger value="overview" className="w-full justify-start gap-2 px-3">
                  <Server className="h-4 w-4" />
                  Overview
                </TabsTrigger>
                <TabsTrigger value="database" className="w-full justify-start gap-2 px-3">
                  <Database className="h-4 w-4" />
                  Database
                </TabsTrigger>
                <TabsTrigger value="cache" className="w-full justify-start gap-2 px-3">
                  <HardDrive className="h-4 w-4" />
                  Cache
                </TabsTrigger>
                <TabsTrigger value="maintenance" className="w-full justify-start gap-2 px-3">
                  <RefreshCw className="h-4 w-4" />
                  Maintenance
                </TabsTrigger>
              </TabsList>
            </div>
            {/* AI & MODELS */}
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-3">AI & Models</p>
              <TabsList className="flex flex-col h-auto w-full bg-transparent gap-0.5">
                <TabsTrigger value="providers" className="w-full justify-start gap-2 px-3">
                  <Bot className="h-4 w-4" />
                  LLM Providers
                </TabsTrigger>
                <TabsTrigger value="rag" className="w-full justify-start gap-2 px-3">
                  <Sparkles className="h-4 w-4" />
                  RAG Features
                </TabsTrigger>
                <TabsTrigger value="models" className="w-full justify-start gap-2 px-3">
                  <Cog className="h-4 w-4" />
                  Models
                </TabsTrigger>
              </TabsList>
            </div>
            {/* PROCESSING */}
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-3">Processing</p>
              <TabsList className="flex flex-col h-auto w-full bg-transparent gap-0.5">
                <TabsTrigger value="ingestion" className="w-full justify-start gap-2 px-3">
                  <GitBranch className="h-4 w-4" />
                  Ingestion
                </TabsTrigger>
                <TabsTrigger value="ocr" className="w-full justify-start gap-2 px-3">
                  <Scan className="h-4 w-4" />
                  OCR
                </TabsTrigger>
                <TabsTrigger value="generation" className="w-full justify-start gap-2 px-3">
                  <PenTool className="h-4 w-4" />
                  Generation
                </TabsTrigger>
                <TabsTrigger value="jobqueue" className="w-full justify-start gap-2 px-3">
                  <Workflow className="h-4 w-4" />
                  Job Queue
                </TabsTrigger>
                <TabsTrigger value="ray" className="w-full justify-start gap-2 px-3">
                  <Zap className="h-4 w-4" />
                  Ray
                </TabsTrigger>
              </TabsList>
            </div>
            {/* INTELLIGENCE */}
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-3">Intelligence</p>
              <TabsList className="flex flex-col h-auto w-full bg-transparent gap-0.5">
                <TabsTrigger value="experiments" className="w-full justify-start gap-2 px-3">
                  <TestTube className="h-4 w-4" />
                  Experiments
                </TabsTrigger>
                <TabsTrigger value="evaluation" className="w-full justify-start gap-2 px-3">
                  <BarChart3 className="h-4 w-4" />
                  Evaluation
                </TabsTrigger>
                <TabsTrigger value="analytics" className="w-full justify-start gap-2 px-3">
                  <BarChart3 className="h-4 w-4" />
                  Analytics
                </TabsTrigger>
              </TabsList>
            </div>
            {/* INTEGRATIONS */}
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-3">Integrations</p>
              <TabsList className="flex flex-col h-auto w-full bg-transparent gap-0.5">
                <TabsTrigger value="audio" className="w-full justify-start gap-2 px-3">
                  <Volume2 className="h-4 w-4" />
                  Audio / TTS
                </TabsTrigger>
                <TabsTrigger value="scraper" className="w-full justify-start gap-2 px-3">
                  <Globe className="h-4 w-4" />
                  Scraper
                </TabsTrigger>
              </TabsList>
            </div>
            {/* SECURITY */}
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-3">Security</p>
              <TabsList className="flex flex-col h-auto w-full bg-transparent gap-0.5">
                <TabsTrigger value="security" className="w-full justify-start gap-2 px-3">
                  <Shield className="h-4 w-4" />
                  Security
                </TabsTrigger>
                <TabsTrigger value="notifications" className="w-full justify-start gap-2 px-3">
                  <Bell className="h-4 w-4" />
                  Notifications
                </TabsTrigger>
                <TabsTrigger value="instructions" className="w-full justify-start gap-2 px-3">
                  <MessageSquare className="h-4 w-4" />
                  Instructions
                </TabsTrigger>
              </TabsList>
            </div>
          </nav>
        </div>

        {/* Mobile Navigation - Dropdown */}
        <div className="md:hidden w-full">
          <TabsList className="flex flex-wrap h-auto gap-1 p-1 w-full">
            <TabsTrigger value="overview" className="flex items-center gap-1 text-xs">
              <Server className="h-3 w-3" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="providers" className="flex items-center gap-1 text-xs">
              <Bot className="h-3 w-3" />
              Providers
            </TabsTrigger>
            <TabsTrigger value="database" className="flex items-center gap-1 text-xs">
              <Database className="h-3 w-3" />
              Database
            </TabsTrigger>
            <TabsTrigger value="rag" className="flex items-center gap-1 text-xs">
              <Sparkles className="h-3 w-3" />
              RAG
            </TabsTrigger>
            <TabsTrigger value="security" className="flex items-center gap-1 text-xs">
              <Shield className="h-3 w-3" />
              Security
            </TabsTrigger>
            <TabsTrigger value="notifications" className="flex items-center gap-1 text-xs">
              <Bell className="h-3 w-3" />
              Alerts
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-1 text-xs">
              <BarChart3 className="h-3 w-3" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="ocr" className="flex items-center gap-1 text-xs">
              <Scan className="h-3 w-3" />
              OCR
            </TabsTrigger>
            <TabsTrigger value="models" className="flex items-center gap-1 text-xs">
              <Cog className="h-3 w-3" />
              Models
            </TabsTrigger>
            <TabsTrigger value="jobqueue" className="flex items-center gap-1 text-xs">
              <Workflow className="h-3 w-3" />
              Jobs
            </TabsTrigger>
            <TabsTrigger value="generation" className="flex items-center gap-1 text-xs">
              <PenTool className="h-3 w-3" />
              Generation
            </TabsTrigger>
            <TabsTrigger value="experiments" className="flex items-center gap-1 text-xs">
              <TestTube className="h-3 w-3" />
              Experiments
            </TabsTrigger>
            <TabsTrigger value="cache" className="flex items-center gap-1 text-xs">
              <HardDrive className="h-3 w-3" />
              Cache
            </TabsTrigger>
            <TabsTrigger value="evaluation" className="flex items-center gap-1 text-xs">
              <BarChart3 className="h-3 w-3" />
              Evaluation
            </TabsTrigger>
            <TabsTrigger value="audio" className="flex items-center gap-1 text-xs">
              <Volume2 className="h-3 w-3" />
              Audio
            </TabsTrigger>
            <TabsTrigger value="scraper" className="flex items-center gap-1 text-xs">
              <Globe className="h-3 w-3" />
              Scraper
            </TabsTrigger>
            <TabsTrigger value="ingestion" className="flex items-center gap-1 text-xs">
              <GitBranch className="h-3 w-3" />
              Ingestion
            </TabsTrigger>
            <TabsTrigger value="instructions" className="flex items-center gap-1 text-xs">
              <MessageSquare className="h-3 w-3" />
              Instructions
            </TabsTrigger>
            <TabsTrigger value="ray" className="flex items-center gap-1 text-xs">
              <Zap className="h-3 w-3" />
              Ray
            </TabsTrigger>
          </TabsList>
        </div>

        {/* Tab Content Area */}
        <div className="flex-1 min-w-0">

        {/* Overview Tab */}
        <OverviewTab />

        {/* LLM Providers Tab */}
        <ProvidersTab
          providersLoading={providersLoading}
          providersData={providersData}
          refetchProviders={refetchProviders}
          providerTypesData={providerTypesData}
          ollamaModelsData={ollamaModelsData}
          ollamaLocalModels={ollamaLocalModels}
          ollamaLocalModelsLoading={ollamaLocalModelsLoading}
          refetchOllamaModels={refetchOllamaModels}
          showAddProvider={showAddProvider}
          setShowAddProvider={setShowAddProvider}
          newProvider={newProvider}
          setNewProvider={setNewProvider}
          showApiKey={showApiKey}
          setShowApiKey={setShowApiKey}
          providerTestResults={providerTestResults}
          setProviderTestResults={setProviderTestResults}
          editingProvider={editingProvider}
          setEditingProvider={setEditingProvider}
          ollamaBaseUrl={ollamaBaseUrl}
          setOllamaBaseUrl={setOllamaBaseUrl}
          newModelName={newModelName}
          setNewModelName={setNewModelName}
          pullingModel={pullingModel}
          setPullingModel={setPullingModel}
          pullResult={pullResult}
          setPullResult={setPullResult}
          deletingModel={deletingModel}
          setDeletingModel={setDeletingModel}
          testProvider={testProvider}
          createProvider={createProvider}
          updateProvider={updateProvider}
          deleteProvider={deleteProvider}
          setDefaultProvider={setDefaultProvider}
          pullOllamaModel={pullOllamaModel}
          deleteOllamaModel={deleteOllamaModel}
          handleEditProvider={handleEditProvider}
          handleTestProvider={handleTestProvider}
          handleSetDefaultProvider={handleSetDefaultProvider}
          handleDeleteProvider={handleDeleteProvider}
          handleSaveProvider={handleSaveProvider}
          handleCancelProviderForm={handleCancelProviderForm}
          getProviderTypeConfig={getProviderTypeConfig}
        />

        {/* Database Tab */}
        <DatabaseTab
          localSettings={localSettings}
          handleSettingChange={handleSettingChange}
          dbInfo={dbInfo}
          dbInfoLoading={dbInfoLoading}
          connectionsData={connectionsData}
          connectionsLoading={connectionsLoading}
          connectionTypesData={connectionTypesData}
          showAddConnection={showAddConnection}
          setShowAddConnection={setShowAddConnection}
          newConnection={newConnection}
          setNewConnection={setNewConnection}
          connectionTestResults={connectionTestResults}
          deletedDocs={deletedDocs}
          deletedDocsTotal={deletedDocsTotal}
          deletedDocsPage={deletedDocsPage}
          deletedDocsLoading={deletedDocsLoading}
          deletedDocsError={deletedDocsError}
          restoringDocId={restoringDocId}
          hardDeletingDocId={hardDeletingDocId}
          selectedDeletedDocs={selectedDeletedDocs}
          isBulkDeleting={isBulkDeleting}
          isBulkRestoring={isBulkRestoring}
          newDbUrl={newDbUrl}
          setNewDbUrl={setNewDbUrl}
          testResult={testResult}
          setTestResult={setTestResult}
          importRef={importRef}
          testConnection={testConnection}
          setupPostgres={setupPostgres}
          exportDatabase={exportDatabase}
          importDatabase={importDatabase}
          createConnection={createConnection}
          deleteConnection={deleteConnection}
          activateConnection={activateConnection}
          testConnectionById={testConnectionById}
          getDbTypeConfig={getDbTypeConfig}
          handleTestConnection={handleTestConnection}
          handleSetupPostgres={handleSetupPostgres}
          handleExport={handleExport}
          handleImport={handleImport}
          handleAddConnection={handleAddConnection}
          handleTestSavedConnection={handleTestSavedConnection}
          handleActivateConnection={handleActivateConnection}
          handleDeleteConnection={handleDeleteConnection}
          fetchDeletedDocs={fetchDeletedDocs}
          handleRestoreDocument={handleRestoreDocument}
          handleHardDeleteDocument={handleHardDeleteDocument}
          handleBulkRestore={handleBulkRestore}
          handleBulkPermanentDelete={handleBulkPermanentDelete}
          toggleSelectAllDeletedDocs={toggleSelectAllDeletedDocs}
          toggleDeletedDocSelection={toggleDeletedDocSelection}
        />

        {/* RAG Features Tab */}
        <RagTab localSettings={localSettings} handleSettingChange={handleSettingChange} />

        {/* Security Tab */}
        <SecurityTab localSettings={localSettings} handleSettingChange={handleSettingChange} />

        {/* Notifications Tab */}
        <NotificationsTab localSettings={localSettings} handleSettingChange={handleSettingChange} />

        {/* Analytics Tab */}
        <AnalyticsTab
          UsageAnalyticsCard={UsageAnalyticsCard}
          ProviderHealthCard={ProviderHealthCard}
          CostAlertsCard={CostAlertsCard}
        />

        {/* OCR Configuration Tab */}
        <OcrTab
          ocrLoading={ocrLoading}
          ocrData={ocrData}
          refetchOCR={refetchOCR}
          updateOCRSettings={updateOCRSettings}
          downloadModels={downloadModels}
          downloadingModels={downloadingModels}
          setDownloadingModels={setDownloadingModels}
          downloadResult={downloadResult}
          setDownloadResult={setDownloadResult}
          vlmConfig={vlmConfig}
          vlmLoading={vlmLoading}
          updateVlmConfig={updateVlmConfig}
          testVlm={testVlm}
          visionModels={ollamaLocalModels?.vision_models}
        />

        {/* Model Configuration Tab */}
        <ModelsTab
          ModelConfigurationSection={ModelConfigurationSection}
          providers={providersData?.providers || []}
          localSettings={localSettings}
          handleSettingChange={handleSettingChange}
        />

        {/* Job Queue Tab */}
        <JobQueueTab
          JobQueueSettings={JobQueueSettings}
          localSettings={localSettings}
          handleSettingChange={handleSettingChange}
        />

        {/* Document Generation Tab */}
        <GenerationTab
          localSettings={localSettings}
          handleSettingChange={handleSettingChange}
          providersData={providersData}
          ollamaLocalModels={ollamaLocalModels}
        />

        {/* Experiments Tab (Phase 90) */}
        <ExperimentsTab />

        {/* Cache Management Tab (Phase 90) */}
        <CacheTab />

        {/* Maintenance Tab - Reindex & KG Extraction (Phase 89) */}
        <MaintenanceTab />

        {/* Evaluation Dashboard Tab (Phase 90) */}
        <EvaluationTab />

        {/* Audio Settings Tab */}
        <AudioTab localSettings={localSettings} handleSettingChange={handleSettingChange} />

        {/* Web Scraper Settings Tab */}
        <ScraperTab localSettings={localSettings} handleSettingChange={handleSettingChange} />

        {/* Knowledge Graph Ingestion Tab */}
        <IngestionTab localSettings={localSettings} handleSettingChange={handleSettingChange} />

        {/* Custom Instructions Tab */}
        <InstructionsTab localSettings={localSettings} handleSettingChange={handleSettingChange} isLoading={settingsLoading} hasChanges={hasChanges} />

        {/* Ray Distributed Processing Tab */}
        <RayTab />

        </div> {/* Close flex-1 content area */}
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
  const [operationConfigs, setOperationConfigs] = useState<Record<string, {
    provider_id: string;
    model_override: string;
    temperature_override: number | null;
  }>>({});

  // Initialize operation configs from data
  useEffect(() => {
    if (operationsData?.operations) {
      const configs: Record<string, { provider_id: string; model_override: string; temperature_override: number | null }> = {};
      for (const op of operationsData.operations) {
        configs[op.operation_type] = {
          provider_id: op.provider_id || "",
          model_override: op.model_override || "",
          temperature_override: op.temperature_override ?? null,
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
        temperature_override: config.temperature_override ?? undefined,
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
                const localConfig = operationConfigs[operationType] || { provider_id: "", model_override: "", temperature_override: null };

                return (
                  <div key={operationType} className={`p-3 rounded-lg border ${isEditing ? "flex flex-col gap-3" : "flex items-center justify-between"}`}>
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-muted">
                        {OPERATION_ICONS[operationType]}
                      </div>
                      <div>
                        <p className="font-medium">{OPERATION_LABELS[operationType]}</p>
                        {config?.provider_name ? (
                          <div className="flex items-center gap-2">
                            <p className="text-sm text-muted-foreground">
                              {config.provider_name} {config.model_override && `(${config.model_override})`}
                            </p>
                            {config.temperature_override !== null && config.temperature_override !== undefined && (
                              <Badge variant="outline" className="text-xs">
                                Temp: {config.temperature_override.toFixed(2)}
                              </Badge>
                            )}
                          </div>
                        ) : (
                          <p className="text-sm text-muted-foreground">Using default provider</p>
                        )}
                      </div>
                    </div>

                    {isEditing ? (
                      <div className="flex flex-col gap-3 flex-1">
                        <div className="flex items-center gap-2">
                          <select
                            className="h-8 rounded-md border bg-background px-2 text-sm flex-1"
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
                        </div>

                        {/* Phase 15 Temperature Control */}
                        {(() => {
                          const selectedProvider = activeProviders.find(p => p.id === localConfig.provider_id);
                          const effectiveModel = localConfig.model_override || selectedProvider?.default_chat_model;
                          const optimizedTemp = calculateOptimizedTemperature(effectiveModel || null);
                          const currentTemp = localConfig.temperature_override ?? optimizedTemp;
                          const isManualOverride = localConfig.temperature_override !== null &&
                                                  localConfig.temperature_override !== undefined &&
                                                  Math.abs(currentTemp - optimizedTemp) > 0.01;

                          return (
                            <div className="space-y-2 pl-2 pr-2">
                              <div className="flex items-center justify-between">
                                <Label className="text-xs">Temperature</Label>
                                <span className="text-xs font-medium">{currentTemp.toFixed(2)}</span>
                              </div>
                              <Slider
                                value={[currentTemp]}
                                onValueChange={([value]) => setOperationConfigs(prev => ({
                                  ...prev,
                                  [operationType]: { ...prev[operationType], temperature_override: value },
                                }))}
                                min={0}
                                max={1}
                                step={0.05}
                                className="w-full"
                              />

                              {/* Phase 15 Optimization Indicator */}
                              {effectiveModel && (
                                <div className="text-xs space-y-1">
                                  <div className="flex items-center justify-between text-muted-foreground">
                                    <span>Optimized for {effectiveModel}:</span>
                                    <span className="font-medium text-blue-600 dark:text-blue-400">{optimizedTemp.toFixed(2)}</span>
                                  </div>

                                  {!isManualOverride ? (
                                    <div className="flex items-center gap-1 text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/30 p-1.5 rounded">
                                      <CheckCircle className="h-3 w-3" />
                                      <span className="text-xs">Research-backed temperature active</span>
                                    </div>
                                  ) : (
                                    <div className="space-y-1">
                                      <div className="flex items-center gap-1 text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 p-1.5 rounded">
                                        <AlertCircle className="h-3 w-3" />
                                        <span className="text-xs">Manual override active</span>
                                      </div>
                                      <Button
                                        size="sm"
                                        variant="outline"
                                        className="h-6 text-xs w-full"
                                        onClick={() => setOperationConfigs(prev => ({
                                          ...prev,
                                          [operationType]: { ...prev[operationType], temperature_override: optimizedTemp },
                                        }))}
                                      >
                                        Reset to Optimized ({optimizedTemp.toFixed(2)})
                                      </Button>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })()}

                        <div className="flex items-center gap-2 justify-end">
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
  const startCelery = useStartCeleryWorker();
  const stopCelery = useStopCeleryWorker();
  const [detectingQueue, setDetectingQueue] = useState(false);

  const detectQueueWorkers = async () => {
    setDetectingQueue(true);
    try {
      const { data } = await api.get<{ recommended: Record<string, number> }>("/diagnostics/hardware/recommended-settings");
      const recommended = data.recommended?.["queue.max_workers"];
      if (recommended !== undefined) {
        handleSettingChange("queue.max_workers", recommended);
      }
    } catch {
      // Silently fail
    } finally {
      setDetectingQueue(false);
    }
  };

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
              {/* Start/Stop Buttons */}
              <div className="flex gap-2 mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => startCelery.mutate()}
                  disabled={startCelery.isPending || celeryStatus?.available}
                >
                  {startCelery.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1" />
                  ) : (
                    <Play className="h-4 w-4 mr-1" />
                  )}
                  Start
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => stopCelery.mutate()}
                  disabled={stopCelery.isPending || !celeryStatus?.available}
                >
                  {stopCelery.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-1" />
                  ) : (
                    <Square className="h-4 w-4 mr-1" />
                  )}
                  Stop
                </Button>
              </div>
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
              <div className="flex items-center justify-between">
                <Label htmlFor="max-workers">Max Celery Workers</Label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={detectQueueWorkers}
                  disabled={detectingQueue}
                >
                  {detectingQueue ? (
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                  ) : (
                    <Cpu className="h-3 w-3 mr-1" />
                  )}
                  Auto-detect
                </Button>
              </div>
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

      {/* Ray Parallel Processing Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Ray Parallel Processing
          </CardTitle>
          <CardDescription>
            Distributed computing for large-scale document processing using Ray
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Enable Ray Toggle */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Ray Cluster</p>
              <p className="text-sm text-muted-foreground">
                Use Ray for parallel processing of large document batches
              </p>
            </div>
            <Switch
              checked={(localSettings["processing.ray_enabled"] as boolean) ?? false}
              onCheckedChange={(checked) =>
                handleSettingChange("processing.ray_enabled", checked)
              }
            />
          </div>

          {/* Ray Configuration - only show if enabled */}
          {Boolean(localSettings["processing.ray_enabled"]) && (
            <>
              {/* Ray Address */}
              <div className="space-y-2">
                <Label htmlFor="ray-address">Ray Cluster Address</Label>
                <Input
                  id="ray-address"
                  value={(localSettings["processing.ray_address"] as string) ?? "auto"}
                  onChange={(e) =>
                    handleSettingChange("processing.ray_address", e.target.value)
                  }
                  placeholder="auto or ray://cluster:10001"
                />
                <p className="text-xs text-muted-foreground">
                  Use &quot;auto&quot; for local cluster or specify remote address
                </p>
              </div>

              {/* CPU/GPU Configuration */}
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="ray-cpus">Max CPUs</Label>
                  <Input
                    id="ray-cpus"
                    type="number"
                    min={0}
                    max={64}
                    value={(localSettings["processing.ray_num_cpus"] as number) ?? 4}
                    onChange={(e) =>
                      handleSettingChange("processing.ray_num_cpus", parseInt(e.target.value) || 4)
                    }
                  />
                  <p className="text-xs text-muted-foreground">
                    0 = use all available
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="ray-gpus">Max GPUs</Label>
                  <Input
                    id="ray-gpus"
                    type="number"
                    min={0}
                    max={8}
                    value={(localSettings["processing.ray_num_gpus"] as number) ?? 0}
                    onChange={(e) =>
                      handleSettingChange("processing.ray_num_gpus", parseInt(e.target.value) || 0)
                    }
                  />
                  <p className="text-xs text-muted-foreground">
                    For GPU-accelerated operations
                  </p>
                </div>
              </div>

              {/* Memory Limit */}
              <div className="space-y-2">
                <Label htmlFor="ray-memory">Memory Limit (GB)</Label>
                <Input
                  id="ray-memory"
                  type="number"
                  min={1}
                  max={128}
                  value={(localSettings["processing.ray_memory_limit_gb"] as number) ?? 8}
                  onChange={(e) =>
                    handleSettingChange("processing.ray_memory_limit_gb", parseInt(e.target.value) || 8)
                  }
                />
                <p className="text-xs text-muted-foreground">
                  Memory limit per Ray worker
                </p>
              </div>
            </>
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
