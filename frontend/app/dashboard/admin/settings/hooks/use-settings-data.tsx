/**
 * Shared data fetching hooks for Settings components
 * Consolidates all data fetching logic in one place
 */

import {
  useSettings,
  useSystemHealth,
  useDatabaseInfo,
  useLLMProviders,
  useLLMProviderTypes,
  useDatabaseConnections,
  useDatabaseConnectionTypes,
  useLLMOperations,
  useLLMUsageSummary,
  useLLMUsageByProvider,
  useLLMUsageByOperation,
  useProviderHealth,
  useCostAlertsAdmin,
  useOllamaModels,
  useOCRSettings,
  useOCRModelInfo,
  useRedisStatus,
  useCeleryStatus,
  useTTSSettings,
  useCoquiModels,
  useSettingsPresets,
} from "@/lib/api/hooks";

export function useSettingsData(isAuthenticated: boolean) {
  // Core settings
  const settingsQuery = useSettings({ enabled: isAuthenticated });
  const healthQuery = useSystemHealth({ enabled: isAuthenticated });
  const dbInfoQuery = useDatabaseInfo({ enabled: isAuthenticated });

  // LLM Providers
  const providersQuery = useLLMProviders({ enabled: isAuthenticated });
  const providerTypesQuery = useLLMProviderTypes({ enabled: isAuthenticated });

  // Database Connections
  const connectionsQuery = useDatabaseConnections({ enabled: isAuthenticated });
  const connectionTypesQuery = useDatabaseConnectionTypes({ enabled: isAuthenticated });

  // LLM Operations
  const operationsQuery = useLLMOperations();
  const usageSummaryQuery = useLLMUsageSummary();
  const usageByProviderQuery = useLLMUsageByProvider();
  const usageByOperationQuery = useLLMUsageByOperation();
  const providerHealthQuery = useProviderHealth();
  const costAlertsQuery = useCostAlertsAdmin();

  // OCR Configuration
  const ocrQuery = useOCRSettings({ enabled: isAuthenticated });
  const ocrModelInfoQuery = useOCRModelInfo({ enabled: isAuthenticated });

  // System Status
  const redisStatusQuery = useRedisStatus({ enabled: isAuthenticated });
  const celeryStatusQuery = useCeleryStatus({ enabled: isAuthenticated });

  // TTS Configuration
  const ttsQuery = useTTSSettings();
  const coquiModelsQuery = useCoquiModels();

  // Settings Presets
  const presetsQuery = useSettingsPresets();

  return {
    // Core
    settings: settingsQuery.data,
    settingsLoading: settingsQuery.isLoading,
    settingsError: settingsQuery.error,
    refetchSettings: settingsQuery.refetch,

    health: healthQuery.data,
    healthLoading: healthQuery.isLoading,

    dbInfo: dbInfoQuery.data,
    dbInfoLoading: dbInfoQuery.isLoading,
    refetchDbInfo: dbInfoQuery.refetch,

    // Providers
    providers: providersQuery.data,
    providersLoading: providersQuery.isLoading,
    refetchProviders: providersQuery.refetch,

    providerTypes: providerTypesQuery.data,

    // Connections
    connections: connectionsQuery.data,
    connectionsLoading: connectionsQuery.isLoading,
    refetchConnections: connectionsQuery.refetch,

    connectionTypes: connectionTypesQuery.data,

    // Operations
    operations: operationsQuery.data,
    operationsLoading: operationsQuery.isLoading,
    refetchOperations: operationsQuery.refetch,

    usageSummary: usageSummaryQuery.data,
    usageByProvider: usageByProviderQuery.data,
    usageByOperation: usageByOperationQuery.data,

    providerHealth: providerHealthQuery.data,
    refetchProviderHealth: providerHealthQuery.refetch,

    costAlerts: costAlertsQuery.data,
    refetchCostAlerts: costAlertsQuery.refetch,

    // OCR
    ocr: ocrQuery.data,
    ocrLoading: ocrQuery.isLoading,
    refetchOCR: ocrQuery.refetch,

    ocrModelInfo: ocrModelInfoQuery.data,
    refetchOCRModelInfo: ocrModelInfoQuery.refetch,

    // System
    redisStatus: redisStatusQuery.data,
    refetchRedisStatus: redisStatusQuery.refetch,

    celeryStatus: celeryStatusQuery.data,
    refetchCeleryStatus: celeryStatusQuery.refetch,

    // TTS
    tts: ttsQuery.data,
    ttsLoading: ttsQuery.isLoading,
    refetchTTS: ttsQuery.refetch,

    coquiModels: coquiModelsQuery.data,
    refetchCoquiModels: coquiModelsQuery.refetch,

    // Presets
    presets: presetsQuery.data,
    refetchPresets: presetsQuery.refetch,
  };
}

// Hook for Ollama models (conditional based on provider type)
export function useOllamaModelsData(
  baseUrl: string | undefined,
  enabled: boolean
) {
  return useOllamaModels(baseUrl, { enabled });
}
