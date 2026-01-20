/**
 * Shared action hooks for Settings components
 * Consolidates all mutation hooks in one place
 */

import {
  useUpdateSettings,
  useResetSettings,
  useTestDatabaseConnection,
  useExportDatabase,
  useImportDatabase,
  useSetupPostgresql,
  useCreateLLMProvider,
  useUpdateLLMProvider,
  useDeleteLLMProvider,
  useTestLLMProvider,
  useSetDefaultLLMProvider,
  useCreateDatabaseConnection,
  useDeleteDatabaseConnection,
  useTestDatabaseConnectionById,
  useActivateDatabaseConnection,
  useSetLLMOperationConfig,
  useDeleteLLMOperationConfig,
  useInvalidateLLMCache,
  useTriggerHealthCheck,
  useResetProviderCircuit,
  useAcknowledgeCostAlert,
  usePullOllamaModel,
  useDeleteOllamaModel,
  useUpdateOCRSettings,
  useDownloadOCRModels,
  useInvalidateRedisCache,
  useUpdateTTSSettings,
  useDownloadCoquiModel,
  useDeleteCoquiModel,
  useApplySettingsPreset,
} from "@/lib/api/hooks";

export function useSettingsActions() {
  // Core settings
  const updateSettings = useUpdateSettings();
  const resetSettings = useResetSettings();

  // Database
  const testConnection = useTestDatabaseConnection();
  const exportDatabase = useExportDatabase();
  const importDatabase = useImportDatabase();
  const setupPostgres = useSetupPostgresql();

  // LLM Providers
  const createProvider = useCreateLLMProvider();
  const updateProvider = useUpdateLLMProvider();
  const deleteProvider = useDeleteLLMProvider();
  const testProvider = useTestLLMProvider();
  const setDefaultProvider = useSetDefaultLLMProvider();

  // Database Connections
  const createConnection = useCreateDatabaseConnection();
  const deleteConnection = useDeleteDatabaseConnection();
  const testConnectionById = useTestDatabaseConnectionById();
  const activateConnection = useActivateDatabaseConnection();

  // LLM Operations
  const setOperationConfig = useSetLLMOperationConfig();
  const deleteOperationConfig = useDeleteLLMOperationConfig();
  const invalidateLLMCache = useInvalidateLLMCache();

  // Provider Health
  const triggerHealthCheck = useTriggerHealthCheck();
  const resetProviderCircuit = useResetProviderCircuit();

  // Cost Alerts
  const acknowledgeCostAlert = useAcknowledgeCostAlert();

  // Ollama
  const pullOllamaModel = usePullOllamaModel();
  const deleteOllamaModel = useDeleteOllamaModel();

  // OCR
  const updateOCRSettings = useUpdateOCRSettings();
  const downloadOCRModels = useDownloadOCRModels();

  // Redis
  const invalidateRedisCache = useInvalidateRedisCache();

  // TTS
  const updateTTSSettings = useUpdateTTSSettings();
  const downloadCoquiModel = useDownloadCoquiModel();
  const deleteCoquiModel = useDeleteCoquiModel();

  // Presets
  const applyPreset = useApplySettingsPreset();

  return {
    // Core
    updateSettings,
    resetSettings,

    // Database
    testConnection,
    exportDatabase,
    importDatabase,
    setupPostgres,

    // Providers
    createProvider,
    updateProvider,
    deleteProvider,
    testProvider,
    setDefaultProvider,

    // Connections
    createConnection,
    deleteConnection,
    testConnectionById,
    activateConnection,

    // Operations
    setOperationConfig,
    deleteOperationConfig,
    invalidateLLMCache,

    // Health
    triggerHealthCheck,
    resetProviderCircuit,

    // Cost
    acknowledgeCostAlert,

    // Ollama
    pullOllamaModel,
    deleteOllamaModel,

    // OCR
    updateOCRSettings,
    downloadOCRModels,

    // Redis
    invalidateRedisCache,

    // TTS
    updateTTSSettings,
    downloadCoquiModel,
    deleteCoquiModel,

    // Presets
    applyPreset,
  };
}
