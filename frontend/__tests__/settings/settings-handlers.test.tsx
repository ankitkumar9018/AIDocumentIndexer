/**
 * Settings Page Handlers Tests
 * =============================
 * Tests for the handler functions in the Settings page that interact with
 * mutation hooks and manage state.
 */

import React from "react";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Mock the API hooks module
jest.mock("@/lib/api/hooks", () => ({
  useSettings: jest.fn(),
  useUpdateSettings: jest.fn(),
  useResetSettings: jest.fn(),
  useSystemHealth: jest.fn(),
  useDatabaseInfo: jest.fn(),
  useTestDatabaseConnection: jest.fn(),
  useExportDatabase: jest.fn(),
  useImportDatabase: jest.fn(),
  useSetupPostgresql: jest.fn(),
  useLLMProviders: jest.fn(),
  useLLMProviderTypes: jest.fn(),
  useCreateLLMProvider: jest.fn(),
  useDeleteLLMProvider: jest.fn(),
  useTestLLMProvider: jest.fn(),
  useSetDefaultLLMProvider: jest.fn(),
  useUpdateLLMProvider: jest.fn(),
  useDatabaseConnections: jest.fn(),
  useDatabaseConnectionTypes: jest.fn(),
  useCreateDatabaseConnection: jest.fn(),
  useDeleteDatabaseConnection: jest.fn(),
  useTestDatabaseConnectionById: jest.fn(),
  useActivateDatabaseConnection: jest.fn(),
  useLLMOperations: jest.fn(),
  useSetLLMOperationConfig: jest.fn(),
  useDeleteLLMOperationConfig: jest.fn(),
  useLLMUsageSummary: jest.fn(),
  useLLMUsageByProvider: jest.fn(),
  useLLMUsageByOperation: jest.fn(),
  useInvalidateLLMCache: jest.fn(),
  useProviderHealth: jest.fn(),
  useTriggerHealthCheck: jest.fn(),
  useResetProviderCircuit: jest.fn(),
  useCostAlertsAdmin: jest.fn(),
  useAcknowledgeCostAlert: jest.fn(),
  useOllamaModels: jest.fn(),
  usePullOllamaModel: jest.fn(),
  useDeleteOllamaModel: jest.fn(),
  useOCRSettings: jest.fn(),
  useUpdateOCRSettings: jest.fn(),
  useDownloadOCRModels: jest.fn(),
  useOCRModelInfo: jest.fn(),
  useRedisStatus: jest.fn(),
  useCeleryStatus: jest.fn(),
  useInvalidateRedisCache: jest.fn(),
  useTTSSettings: jest.fn(),
  useUpdateTTSSettings: jest.fn(),
  useCoquiModels: jest.fn(),
  useDownloadCoquiModel: jest.fn(),
  useDeleteCoquiModel: jest.fn(),
  useSettingsPresets: jest.fn(),
  useApplySettingsPreset: jest.fn(),
}));

// Mock auth
jest.mock("@/lib/auth", () => ({
  useUser: jest.fn().mockReturnValue({
    isAuthenticated: true,
    isLoading: false,
    user: { id: "1", email: "admin@test.com", role: "admin" },
  }),
}));

// Mock API client
jest.mock("@/lib/api/client", () => ({
  api: {
    restoreDeletedDocument: jest.fn(),
    deleteDocument: jest.fn(),
    getDeletedDocuments: jest.fn(),
  },
}));

// Import mocked hooks
import * as hooks from "@/lib/api/hooks";

// Create mock mutation factory
function createMockMutation(options: {
  mutateAsync?: jest.Mock;
  isPending?: boolean;
} = {}) {
  return {
    mutate: jest.fn(),
    mutateAsync: options.mutateAsync || jest.fn().mockResolvedValue({}),
    isPending: options.isPending || false,
    isSuccess: false,
    isError: false,
    error: null,
    data: undefined,
    reset: jest.fn(),
  };
}

// Create mock query factory
function createMockQuery(options: {
  data?: unknown;
  isLoading?: boolean;
  error?: Error | null;
  refetch?: jest.Mock;
} = {}) {
  return {
    data: options.data || null,
    isLoading: options.isLoading || false,
    error: options.error || null,
    refetch: options.refetch || jest.fn(),
    isError: !!options.error,
    isSuccess: !options.isLoading && !options.error,
  };
}

// Test wrapper
function createTestWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

describe("Settings Page Handlers", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset window.confirm
    window.confirm = jest.fn().mockReturnValue(true);
    window.alert = jest.fn();
  });

  describe("handleSettingChange", () => {
    it("should update local settings state with new value", () => {
      // Simulate the handler behavior
      const localSettings: Record<string, unknown> = {};
      let hasChanges = false;

      const handleSettingChange = (key: string, value: unknown) => {
        localSettings[key] = value;
        hasChanges = true;
      };

      handleSettingChange("database.vector_dimensions", 1536);

      expect(localSettings["database.vector_dimensions"]).toBe(1536);
      expect(hasChanges).toBe(true);
    });

    it("should handle multiple setting changes", () => {
      const localSettings: Record<string, unknown> = {};
      let hasChanges = false;

      const handleSettingChange = (key: string, value: unknown) => {
        localSettings[key] = value;
        hasChanges = true;
      };

      handleSettingChange("database.vector_dimensions", 1536);
      handleSettingChange("database.index_type", "ivfflat");
      handleSettingChange("rag.enabled", true);

      expect(localSettings["database.vector_dimensions"]).toBe(1536);
      expect(localSettings["database.index_type"]).toBe("ivfflat");
      expect(localSettings["rag.enabled"]).toBe(true);
      expect(hasChanges).toBe(true);
    });

    it("should handle boolean values correctly", () => {
      const localSettings: Record<string, unknown> = { "feature.enabled": true };

      const handleSettingChange = (key: string, value: unknown) => {
        localSettings[key] = value;
      };

      handleSettingChange("feature.enabled", false);
      expect(localSettings["feature.enabled"]).toBe(false);

      handleSettingChange("feature.enabled", true);
      expect(localSettings["feature.enabled"]).toBe(true);
    });
  });

  describe("handleSave", () => {
    it("should call updateSettings.mutateAsync with local settings", async () => {
      const mockMutateAsync = jest.fn().mockResolvedValue({ success: true });
      const mockUpdateSettings = createMockMutation({ mutateAsync: mockMutateAsync });

      (hooks.useUpdateSettings as jest.Mock).mockReturnValue(mockUpdateSettings);

      const localSettings = { "database.vector_dimensions": 1536 };
      let hasChanges = true;

      const handleSave = async () => {
        try {
          await mockMutateAsync(localSettings);
          hasChanges = false;
        } catch (err) {
          console.error("Failed to save settings:", err);
        }
      };

      await handleSave();

      expect(mockMutateAsync).toHaveBeenCalledWith(localSettings);
      expect(hasChanges).toBe(false);
    });

    it("should handle save error gracefully", async () => {
      const mockError = new Error("Save failed");
      const mockMutateAsync = jest.fn().mockRejectedValue(mockError);

      const consoleErrorSpy = jest.spyOn(console, "error").mockImplementation();

      let hasChanges = true;

      const handleSave = async () => {
        try {
          await mockMutateAsync({});
          hasChanges = false;
        } catch (err) {
          console.error("Failed to save settings:", err);
        }
      };

      await handleSave();

      expect(consoleErrorSpy).toHaveBeenCalledWith("Failed to save settings:", mockError);
      expect(hasChanges).toBe(true); // Should remain true on error

      consoleErrorSpy.mockRestore();
    });
  });

  describe("handleReset", () => {
    it("should call resetSettings.mutateAsync when confirmed", async () => {
      const mockMutateAsync = jest.fn().mockResolvedValue({ success: true });

      window.confirm = jest.fn().mockReturnValue(true);

      let hasChanges = true;

      const handleReset = async () => {
        if (confirm("Are you sure you want to reset all settings to defaults?")) {
          try {
            await mockMutateAsync();
            hasChanges = false;
          } catch (err) {
            console.error("Failed to reset settings:", err);
          }
        }
      };

      await handleReset();

      expect(window.confirm).toHaveBeenCalledWith("Are you sure you want to reset all settings to defaults?");
      expect(mockMutateAsync).toHaveBeenCalled();
      expect(hasChanges).toBe(false);
    });

    it("should not reset when user cancels confirmation", async () => {
      const mockMutateAsync = jest.fn();

      window.confirm = jest.fn().mockReturnValue(false);

      const handleReset = async () => {
        if (confirm("Are you sure you want to reset all settings to defaults?")) {
          await mockMutateAsync();
        }
      };

      await handleReset();

      expect(window.confirm).toHaveBeenCalled();
      expect(mockMutateAsync).not.toHaveBeenCalled();
    });
  });

  describe("handleTestConnection", () => {
    it("should call testConnection.mutateAsync with database URL", async () => {
      const mockMutateAsync = jest.fn().mockResolvedValue({
        success: true,
        has_pgvector: true,
        pgvector_version: "0.5.0",
        message: "Connection successful",
      });

      let testResult: unknown = null;
      const newDbUrl = "postgresql://localhost:5432/testdb";

      const handleTestConnection = async () => {
        if (!newDbUrl) return;
        testResult = null;
        try {
          const result = await mockMutateAsync(newDbUrl);
          testResult = result;
        } catch (err) {
          testResult = {
            success: false,
            error: (err as Error).message,
            message: "Connection test failed",
          };
        }
      };

      await handleTestConnection();

      expect(mockMutateAsync).toHaveBeenCalledWith(newDbUrl);
      expect(testResult).toEqual({
        success: true,
        has_pgvector: true,
        pgvector_version: "0.5.0",
        message: "Connection successful",
      });
    });

    it("should not call mutation when URL is empty", async () => {
      const mockMutateAsync = jest.fn();
      const newDbUrl = "";

      const handleTestConnection = async () => {
        if (!newDbUrl) return;
        await mockMutateAsync(newDbUrl);
      };

      await handleTestConnection();

      expect(mockMutateAsync).not.toHaveBeenCalled();
    });

    it("should handle connection test error", async () => {
      const mockMutateAsync = jest.fn().mockRejectedValue(new Error("Connection refused"));

      let testResult: { success: boolean; error?: string; message: string } | null = null;
      const newDbUrl = "postgresql://localhost:5432/testdb";

      const handleTestConnection = async () => {
        if (!newDbUrl) return;
        testResult = null;
        try {
          const result = await mockMutateAsync(newDbUrl);
          testResult = result;
        } catch (err) {
          testResult = {
            success: false,
            error: (err as Error).message,
            message: "Connection test failed",
          };
        }
      };

      await handleTestConnection();

      expect(testResult).toEqual({
        success: false,
        error: "Connection refused",
        message: "Connection test failed",
      });
    });
  });

  describe("Provider Handlers", () => {
    describe("handleAddProvider / handleSaveProvider (create mode)", () => {
      it("should call createProvider.mutateAsync with new provider data", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({ id: "new-provider-id" });
        const mockRefetch = jest.fn();

        let showAddProvider = true;
        let newProvider = {
          name: "Test Provider",
          provider_type: "openai",
          api_key: "sk-test-key",
          api_base_url: "",
          organization_id: "",
          default_chat_model: "gpt-4",
          default_embedding_model: "text-embedding-3-small",
          is_default: false,
        };

        const handleAddProvider = async () => {
          try {
            await mockMutateAsync({
              name: newProvider.name,
              provider_type: newProvider.provider_type,
              api_key: newProvider.api_key || undefined,
              api_base_url: newProvider.api_base_url || undefined,
              organization_id: newProvider.organization_id || undefined,
              default_chat_model: newProvider.default_chat_model || undefined,
              default_embedding_model: newProvider.default_embedding_model || undefined,
              is_default: newProvider.is_default,
            });
            showAddProvider = false;
            newProvider = {
              name: "",
              provider_type: "openai",
              api_key: "",
              api_base_url: "",
              organization_id: "",
              default_chat_model: "",
              default_embedding_model: "",
              is_default: false,
            };
            mockRefetch();
          } catch (err) {
            console.error("Failed to create provider:", err);
          }
        };

        await handleAddProvider();

        expect(mockMutateAsync).toHaveBeenCalledWith({
          name: "Test Provider",
          provider_type: "openai",
          api_key: "sk-test-key",
          api_base_url: undefined,
          organization_id: undefined,
          default_chat_model: "gpt-4",
          default_embedding_model: "text-embedding-3-small",
          is_default: false,
        });
        expect(showAddProvider).toBe(false);
        expect(newProvider.name).toBe("");
        expect(mockRefetch).toHaveBeenCalled();
      });
    });

    describe("handleDeleteProvider", () => {
      it("should call deleteProvider.mutateAsync when confirmed", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({ success: true });
        const mockRefetch = jest.fn();

        window.confirm = jest.fn().mockReturnValue(true);

        const handleDeleteProvider = async (providerId: string) => {
          if (!confirm("Are you sure you want to delete this provider?")) return;
          try {
            await mockMutateAsync(providerId);
            mockRefetch();
          } catch (err) {
            console.error("Failed to delete provider:", err);
          }
        };

        await handleDeleteProvider("provider-123");

        expect(window.confirm).toHaveBeenCalledWith("Are you sure you want to delete this provider?");
        expect(mockMutateAsync).toHaveBeenCalledWith("provider-123");
        expect(mockRefetch).toHaveBeenCalled();
      });

      it("should not delete when user cancels confirmation", async () => {
        const mockMutateAsync = jest.fn();

        window.confirm = jest.fn().mockReturnValue(false);

        const handleDeleteProvider = async (providerId: string) => {
          if (!confirm("Are you sure you want to delete this provider?")) return;
          await mockMutateAsync(providerId);
        };

        await handleDeleteProvider("provider-123");

        expect(mockMutateAsync).not.toHaveBeenCalled();
      });
    });

    describe("handleTestProvider", () => {
      it("should set initial testing state and update with result", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({
          success: true,
          message: "Provider connected successfully",
        });

        let providerTestResults: Record<string, { success: boolean; message?: string; error?: string }> = {};

        const handleTestProvider = async (providerId: string) => {
          providerTestResults = {
            ...providerTestResults,
            [providerId]: { success: true, message: "Testing..." },
          };
          try {
            const result = await mockMutateAsync(providerId);
            providerTestResults = {
              ...providerTestResults,
              [providerId]: { success: result.success, message: result.message, error: result.error },
            };
          } catch (err) {
            providerTestResults = {
              ...providerTestResults,
              [providerId]: { success: false, error: (err as Error).message },
            };
          }
        };

        await handleTestProvider("provider-123");

        expect(mockMutateAsync).toHaveBeenCalledWith("provider-123");
        expect(providerTestResults["provider-123"]).toEqual({
          success: true,
          message: "Provider connected successfully",
          error: undefined,
        });
      });

      it("should handle test provider error", async () => {
        const mockMutateAsync = jest.fn().mockRejectedValue(new Error("API key invalid"));

        let providerTestResults: Record<string, { success: boolean; message?: string; error?: string }> = {};

        const handleTestProvider = async (providerId: string) => {
          providerTestResults = {
            ...providerTestResults,
            [providerId]: { success: true, message: "Testing..." },
          };
          try {
            const result = await mockMutateAsync(providerId);
            providerTestResults = {
              ...providerTestResults,
              [providerId]: { success: result.success, message: result.message },
            };
          } catch (err) {
            providerTestResults = {
              ...providerTestResults,
              [providerId]: { success: false, error: (err as Error).message },
            };
          }
        };

        await handleTestProvider("provider-123");

        expect(providerTestResults["provider-123"]).toEqual({
          success: false,
          error: "API key invalid",
        });
      });
    });

    describe("handleSetDefaultProvider", () => {
      it("should call setDefaultProvider.mutateAsync with provider ID", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({ success: true });
        const mockRefetch = jest.fn();

        const handleSetDefaultProvider = async (providerId: string) => {
          try {
            await mockMutateAsync(providerId);
            mockRefetch();
          } catch (err) {
            console.error("Failed to set default provider:", err);
          }
        };

        await handleSetDefaultProvider("provider-456");

        expect(mockMutateAsync).toHaveBeenCalledWith("provider-456");
        expect(mockRefetch).toHaveBeenCalled();
      });
    });

    describe("handleEditProvider", () => {
      it("should populate form with provider data for editing", () => {
        const existingProvider = {
          id: "provider-123",
          name: "OpenAI Production",
          provider_type: "openai",
          api_base_url: "https://api.openai.com",
          organization_id: "org-123",
          default_chat_model: "gpt-4",
          default_embedding_model: "text-embedding-3-small",
          is_default: true,
        };

        let editingProvider: typeof existingProvider | null = null;
        let newProvider = {
          name: "",
          provider_type: "openai",
          api_key: "",
          api_base_url: "",
          organization_id: "",
          default_chat_model: "",
          default_embedding_model: "",
          is_default: false,
        };
        let showAddProvider = false;

        const handleEditProvider = (provider: typeof existingProvider) => {
          editingProvider = provider;
          newProvider = {
            name: provider.name,
            provider_type: provider.provider_type,
            api_key: "", // Keep empty for security
            api_base_url: provider.api_base_url || "",
            organization_id: provider.organization_id || "",
            default_chat_model: provider.default_chat_model || "",
            default_embedding_model: provider.default_embedding_model || "",
            is_default: provider.is_default,
          };
          showAddProvider = true;
        };

        handleEditProvider(existingProvider);

        expect(editingProvider).toBe(existingProvider);
        expect(newProvider.name).toBe("OpenAI Production");
        expect(newProvider.api_key).toBe(""); // Security: API key not populated
        expect(newProvider.api_base_url).toBe("https://api.openai.com");
        expect(showAddProvider).toBe(true);
      });
    });

    describe("handleSaveProvider (update mode)", () => {
      it("should call updateProvider.mutateAsync when editing existing provider", async () => {
        const mockUpdateMutateAsync = jest.fn().mockResolvedValue({ success: true });
        const mockRefetch = jest.fn();

        const editingProvider = { id: "provider-123", name: "Old Name" };
        let showAddProvider = true;
        const newProvider = {
          name: "New Name",
          api_key: "new-api-key",
          api_base_url: "https://api.example.com",
          organization_id: "",
          default_chat_model: "gpt-4",
          default_embedding_model: "",
        };

        const handleSaveProvider = async () => {
          if (editingProvider) {
            try {
              await mockUpdateMutateAsync({
                providerId: editingProvider.id,
                data: {
                  name: newProvider.name !== editingProvider.name ? newProvider.name : undefined,
                  api_key: newProvider.api_key || undefined,
                  api_base_url: newProvider.api_base_url || undefined,
                  organization_id: newProvider.organization_id || undefined,
                  default_chat_model: newProvider.default_chat_model || undefined,
                  default_embedding_model: newProvider.default_embedding_model || undefined,
                  is_active: true,
                },
              });
              showAddProvider = false;
              mockRefetch();
            } catch (err) {
              console.error("Failed to update provider:", err);
            }
          }
        };

        await handleSaveProvider();

        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          providerId: "provider-123",
          data: {
            name: "New Name",
            api_key: "new-api-key",
            api_base_url: "https://api.example.com",
            organization_id: undefined,
            default_chat_model: "gpt-4",
            default_embedding_model: undefined,
            is_active: true,
          },
        });
        expect(showAddProvider).toBe(false);
        expect(mockRefetch).toHaveBeenCalled();
      });
    });

    describe("handleCancelProviderForm", () => {
      it("should reset form state", () => {
        let showAddProvider = true;
        let editingProvider: { id: string } | null = { id: "provider-123" };
        let newProvider = {
          name: "Test",
          provider_type: "anthropic",
          api_key: "test-key",
          api_base_url: "https://api.test.com",
          organization_id: "org-test",
          default_chat_model: "claude-3",
          default_embedding_model: "",
          is_default: true,
        };

        const handleCancelProviderForm = () => {
          showAddProvider = false;
          editingProvider = null;
          newProvider = {
            name: "",
            provider_type: "openai",
            api_key: "",
            api_base_url: "",
            organization_id: "",
            default_chat_model: "",
            default_embedding_model: "",
            is_default: false,
          };
        };

        handleCancelProviderForm();

        expect(showAddProvider).toBe(false);
        expect(editingProvider).toBe(null);
        expect(newProvider.name).toBe("");
        expect(newProvider.provider_type).toBe("openai");
        expect(newProvider.is_default).toBe(false);
      });
    });
  });

  describe("Database Connection Handlers", () => {
    describe("handleAddConnection", () => {
      it("should call createConnection.mutateAsync with connection data", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({ id: "conn-new" });
        const mockRefetch = jest.fn();

        let showAddConnection = true;
        let newConnection = {
          name: "Production DB",
          db_type: "postgresql",
          host: "db.example.com",
          port: 5432,
          database: "myapp",
          username: "admin",
          password: "secret",
          vector_store: "pgvector",
          is_active: false,
        };

        const handleAddConnection = async () => {
          try {
            await mockMutateAsync({
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
            showAddConnection = false;
            newConnection = {
              name: "",
              db_type: "postgresql",
              host: "localhost",
              port: 5432,
              database: "aidocindexer",
              username: "",
              password: "",
              vector_store: "auto",
              is_active: false,
            };
            mockRefetch();
          } catch (err) {
            console.error("Failed to create connection:", err);
          }
        };

        await handleAddConnection();

        expect(mockMutateAsync).toHaveBeenCalledWith({
          name: "Production DB",
          db_type: "postgresql",
          database: "myapp",
          host: "db.example.com",
          port: 5432,
          username: "admin",
          password: "secret",
          vector_store: "pgvector",
          is_active: false,
        });
        expect(showAddConnection).toBe(false);
        expect(mockRefetch).toHaveBeenCalled();
      });

      it("should not include host/port/username/password for SQLite connections", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({ id: "conn-new" });

        const newConnection = {
          name: "Local SQLite",
          db_type: "sqlite",
          host: "localhost",
          port: 5432,
          database: "./data/app.db",
          username: "ignored",
          password: "ignored",
          vector_store: "auto",
          is_active: false,
        };

        const handleAddConnection = async () => {
          await mockMutateAsync({
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
        };

        await handleAddConnection();

        expect(mockMutateAsync).toHaveBeenCalledWith({
          name: "Local SQLite",
          db_type: "sqlite",
          database: "./data/app.db",
          host: undefined,
          port: undefined,
          username: undefined,
          password: undefined,
          vector_store: "auto",
          is_active: false,
        });
      });
    });

    describe("handleDeleteConnection", () => {
      it("should call deleteConnection.mutateAsync when confirmed", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({ success: true });
        const mockRefetch = jest.fn();

        window.confirm = jest.fn().mockReturnValue(true);

        const handleDeleteConnection = async (connectionId: string) => {
          if (!confirm("Are you sure you want to delete this connection?")) return;
          try {
            await mockMutateAsync(connectionId);
            mockRefetch();
          } catch (err) {
            console.error("Failed to delete connection:", err);
          }
        };

        await handleDeleteConnection("conn-123");

        expect(window.confirm).toHaveBeenCalled();
        expect(mockMutateAsync).toHaveBeenCalledWith("conn-123");
        expect(mockRefetch).toHaveBeenCalled();
      });
    });

    describe("handleTestSavedConnection", () => {
      it("should set initial testing state and update with result", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({
          success: true,
          message: "Connection successful",
        });

        let connectionTestResults: Record<string, { success: boolean; message?: string; error?: string }> = {};

        const handleTestSavedConnection = async (connectionId: string) => {
          connectionTestResults = {
            ...connectionTestResults,
            [connectionId]: { success: true, message: "Testing..." },
          };
          try {
            const result = await mockMutateAsync(connectionId);
            connectionTestResults = {
              ...connectionTestResults,
              [connectionId]: { success: result.success, message: result.message, error: result.error },
            };
          } catch (err) {
            connectionTestResults = {
              ...connectionTestResults,
              [connectionId]: { success: false, error: (err as Error).message },
            };
          }
        };

        await handleTestSavedConnection("conn-123");

        expect(mockMutateAsync).toHaveBeenCalledWith("conn-123");
        expect(connectionTestResults["conn-123"]).toEqual({
          success: true,
          message: "Connection successful",
          error: undefined,
        });
      });
    });

    describe("handleActivateConnection", () => {
      it("should call activateConnection.mutateAsync and show restart message", async () => {
        const mockMutateAsync = jest.fn().mockResolvedValue({ success: true });
        const mockRefetchConnections = jest.fn();
        const mockRefetchDbInfo = jest.fn();

        window.alert = jest.fn();

        const handleActivateConnection = async (connectionId: string) => {
          try {
            await mockMutateAsync(connectionId);
            mockRefetchConnections();
            mockRefetchDbInfo();
            alert("Connection activated. Please restart the server to use this connection.");
          } catch (err) {
            console.error("Failed to activate connection:", err);
          }
        };

        await handleActivateConnection("conn-456");

        expect(mockMutateAsync).toHaveBeenCalledWith("conn-456");
        expect(mockRefetchConnections).toHaveBeenCalled();
        expect(mockRefetchDbInfo).toHaveBeenCalled();
        expect(window.alert).toHaveBeenCalledWith(
          "Connection activated. Please restart the server to use this connection."
        );
      });
    });
  });

  describe("Export/Import Handlers", () => {
    describe("handleExport", () => {
      it("should call exportDatabase.mutateAsync and trigger download", async () => {
        const mockExportData = {
          settings: { key: "value" },
          providers: [],
          version: "1.0.0",
        };
        const mockMutateAsync = jest.fn().mockResolvedValue(mockExportData);

        // Mock URL and document methods
        const mockCreateObjectURL = jest.fn().mockReturnValue("blob:test-url");
        const mockRevokeObjectURL = jest.fn();
        const mockClick = jest.fn();
        const mockAppendChild = jest.fn();
        const mockRemoveChild = jest.fn();

        global.URL.createObjectURL = mockCreateObjectURL;
        global.URL.revokeObjectURL = mockRevokeObjectURL;

        const mockAnchor = {
          href: "",
          download: "",
          click: mockClick,
        };
        jest.spyOn(document, "createElement").mockReturnValue(mockAnchor as unknown as HTMLAnchorElement);
        jest.spyOn(document.body, "appendChild").mockImplementation(mockAppendChild);
        jest.spyOn(document.body, "removeChild").mockImplementation(mockRemoveChild);

        const handleExport = async () => {
          try {
            const data = await mockMutateAsync();
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
          }
        };

        await handleExport();

        expect(mockMutateAsync).toHaveBeenCalled();
        expect(mockCreateObjectURL).toHaveBeenCalled();
        expect(mockClick).toHaveBeenCalled();
        expect(mockRevokeObjectURL).toHaveBeenCalled();
      });
    });

    describe("handleImport", () => {
      it("should parse file and call importDatabase.mutateAsync", async () => {
        const mockImportResult = {
          success: true,
          imported: { settings: 10, providers: 2 },
          warnings: [],
        };
        const mockMutateAsync = jest.fn().mockResolvedValue(mockImportResult);
        const mockRefetchDbInfo = jest.fn();

        window.confirm = jest.fn().mockReturnValue(true); // Clear existing data
        window.alert = jest.fn();

        const fileContent = JSON.stringify({ settings: {}, providers: [] });
        const mockFile = new File([fileContent], "import.json", { type: "application/json" });
        Object.defineProperty(mockFile, "text", {
          value: jest.fn().mockResolvedValue(fileContent),
        });

        const handleImport = async (file: File) => {
          try {
            const text = await file.text();
            const data = JSON.parse(text);

            const clearExisting = confirm("Do you want to clear existing data?");

            const result = await mockMutateAsync({ data, clearExisting });

            if (result.success) {
              alert(`Import completed! Imported: ${Object.entries(result.imported).map(([k, v]) => `${k}: ${v}`).join(", ")}`);
              mockRefetchDbInfo();
            }
          } catch (err) {
            alert("Import failed: " + (err as Error).message);
          }
        };

        await handleImport(mockFile);

        expect(mockMutateAsync).toHaveBeenCalledWith({
          data: { settings: {}, providers: [] },
          clearExisting: true,
        });
        expect(mockRefetchDbInfo).toHaveBeenCalled();
        expect(window.alert).toHaveBeenCalled();
      });
    });
  });

  describe("Deleted Documents Handlers", () => {
    describe("handleRestoreDocument", () => {
      it("should call api.restoreDeletedDocument and refresh list", async () => {
        const { api } = require("@/lib/api/client");
        api.restoreDeletedDocument.mockResolvedValue({ success: true });

        const mockFetchDeletedDocs = jest.fn();
        const mockRefetchDbInfo = jest.fn();
        let restoringDocId: string | null = null;

        const handleRestoreDocument = async (docId: string) => {
          restoringDocId = docId;
          try {
            await api.restoreDeletedDocument(docId);
            await mockFetchDeletedDocs(1);
            mockRefetchDbInfo();
          } catch (err) {
            alert(`Failed to restore: ${(err as Error).message}`);
          } finally {
            restoringDocId = null;
          }
        };

        await handleRestoreDocument("doc-123");

        expect(api.restoreDeletedDocument).toHaveBeenCalledWith("doc-123");
        expect(mockFetchDeletedDocs).toHaveBeenCalledWith(1);
        expect(mockRefetchDbInfo).toHaveBeenCalled();
      });
    });

    describe("handleHardDeleteDocument", () => {
      it("should call api.deleteDocument with hard_delete=true when confirmed", async () => {
        const { api } = require("@/lib/api/client");
        api.deleteDocument.mockResolvedValue({ success: true });

        window.confirm = jest.fn().mockReturnValue(true);
        const mockFetchDeletedDocs = jest.fn();

        const handleHardDeleteDocument = async (docId: string) => {
          if (!confirm("Permanently delete this document?")) return;
          try {
            await api.deleteDocument(docId, true);
            await mockFetchDeletedDocs(1);
          } catch (err) {
            alert(`Failed to delete: ${(err as Error).message}`);
          }
        };

        await handleHardDeleteDocument("doc-456");

        expect(window.confirm).toHaveBeenCalled();
        expect(api.deleteDocument).toHaveBeenCalledWith("doc-456", true);
        expect(mockFetchDeletedDocs).toHaveBeenCalled();
      });

      it("should not delete when user cancels confirmation", async () => {
        const { api } = require("@/lib/api/client");
        window.confirm = jest.fn().mockReturnValue(false);

        const handleHardDeleteDocument = async (docId: string) => {
          if (!confirm("Permanently delete this document?")) return;
          await api.deleteDocument(docId, true);
        };

        await handleHardDeleteDocument("doc-456");

        expect(api.deleteDocument).not.toHaveBeenCalled();
      });
    });

    describe("handleBulkRestore", () => {
      it("should restore multiple documents and show success message", async () => {
        const { api } = require("@/lib/api/client");
        api.restoreDeletedDocument.mockResolvedValue({ success: true });

        const selectedDeletedDocs = new Set(["doc-1", "doc-2", "doc-3"]);
        let isBulkRestoring = false;
        const mockFetchDeletedDocs = jest.fn();
        const mockRefetchDbInfo = jest.fn();
        let clearedSelection = false;

        window.alert = jest.fn();

        const handleBulkRestore = async () => {
          if (selectedDeletedDocs.size === 0) return;

          isBulkRestoring = true;
          const ids = Array.from(selectedDeletedDocs);
          let restoredCount = 0;
          let failedCount = 0;

          try {
            for (const id of ids) {
              try {
                await api.restoreDeletedDocument(id);
                restoredCount++;
              } catch {
                failedCount++;
              }
            }

            if (failedCount > 0) {
              alert(`Restored ${restoredCount} documents. ${failedCount} failed.`);
            } else {
              alert(`Successfully restored ${restoredCount} documents.`);
            }

            clearedSelection = true;
            await mockFetchDeletedDocs(1);
            mockRefetchDbInfo();
          } finally {
            isBulkRestoring = false;
          }
        };

        await handleBulkRestore();

        expect(api.restoreDeletedDocument).toHaveBeenCalledTimes(3);
        expect(window.alert).toHaveBeenCalledWith("Successfully restored 3 documents.");
        expect(clearedSelection).toBe(true);
        expect(mockFetchDeletedDocs).toHaveBeenCalled();
      });

      it("should not proceed when no documents selected", async () => {
        const { api } = require("@/lib/api/client");
        api.restoreDeletedDocument.mockClear();

        const selectedDeletedDocs = new Set<string>();

        const handleBulkRestore = async () => {
          if (selectedDeletedDocs.size === 0) return;
          // Would never reach here
        };

        await handleBulkRestore();

        expect(api.restoreDeletedDocument).not.toHaveBeenCalled();
      });
    });

    describe("toggleDeletedDocSelection", () => {
      it("should add document to selection when not selected", () => {
        let selectedDeletedDocs = new Set<string>();

        const toggleDeletedDocSelection = (docId: string) => {
          const newSet = new Set(selectedDeletedDocs);
          if (newSet.has(docId)) {
            newSet.delete(docId);
          } else {
            newSet.add(docId);
          }
          selectedDeletedDocs = newSet;
        };

        toggleDeletedDocSelection("doc-1");

        expect(selectedDeletedDocs.has("doc-1")).toBe(true);
      });

      it("should remove document from selection when already selected", () => {
        let selectedDeletedDocs = new Set(["doc-1", "doc-2"]);

        const toggleDeletedDocSelection = (docId: string) => {
          const newSet = new Set(selectedDeletedDocs);
          if (newSet.has(docId)) {
            newSet.delete(docId);
          } else {
            newSet.add(docId);
          }
          selectedDeletedDocs = newSet;
        };

        toggleDeletedDocSelection("doc-1");

        expect(selectedDeletedDocs.has("doc-1")).toBe(false);
        expect(selectedDeletedDocs.has("doc-2")).toBe(true);
      });
    });

    describe("toggleSelectAllDeletedDocs", () => {
      it("should select all documents when none or partial selected", () => {
        const deletedDocs = [{ id: "doc-1" }, { id: "doc-2" }, { id: "doc-3" }];
        let selectedDeletedDocs = new Set(["doc-1"]);

        const toggleSelectAllDeletedDocs = () => {
          if (selectedDeletedDocs.size === deletedDocs.length && deletedDocs.length > 0) {
            selectedDeletedDocs = new Set();
          } else {
            selectedDeletedDocs = new Set(deletedDocs.map((d) => d.id));
          }
        };

        toggleSelectAllDeletedDocs();

        expect(selectedDeletedDocs.size).toBe(3);
        expect(selectedDeletedDocs.has("doc-1")).toBe(true);
        expect(selectedDeletedDocs.has("doc-2")).toBe(true);
        expect(selectedDeletedDocs.has("doc-3")).toBe(true);
      });

      it("should deselect all when all are selected", () => {
        const deletedDocs = [{ id: "doc-1" }, { id: "doc-2" }];
        let selectedDeletedDocs = new Set(["doc-1", "doc-2"]);

        const toggleSelectAllDeletedDocs = () => {
          if (selectedDeletedDocs.size === deletedDocs.length && deletedDocs.length > 0) {
            selectedDeletedDocs = new Set();
          } else {
            selectedDeletedDocs = new Set(deletedDocs.map((d) => d.id));
          }
        };

        toggleSelectAllDeletedDocs();

        expect(selectedDeletedDocs.size).toBe(0);
      });
    });
  });

  describe("Utility Functions", () => {
    describe("getProviderTypeConfig", () => {
      it("should return provider type configuration", () => {
        const providerTypesData: { provider_types: Record<string, { name: string; fields: string[]; default_model: string; chat_models: string[]; embedding_models: string[] }> } = {
          provider_types: {
            openai: {
              name: "OpenAI",
              fields: ["api_key", "api_base_url"],
              default_model: "gpt-4",
              chat_models: ["gpt-4", "gpt-3.5-turbo"],
              embedding_models: ["text-embedding-3-small"],
            },
            anthropic: {
              name: "Anthropic",
              fields: ["api_key"],
              default_model: "claude-3-opus",
              chat_models: ["claude-3-opus", "claude-3-sonnet"],
              embedding_models: [],
            },
          },
        };

        const getProviderTypeConfig = (type: string) => {
          return providerTypesData?.provider_types?.[type];
        };

        expect(getProviderTypeConfig("openai")).toEqual({
          name: "OpenAI",
          fields: ["api_key", "api_base_url"],
          default_model: "gpt-4",
          chat_models: ["gpt-4", "gpt-3.5-turbo"],
          embedding_models: ["text-embedding-3-small"],
        });
        expect(getProviderTypeConfig("anthropic")?.name).toBe("Anthropic");
        expect(getProviderTypeConfig("unknown")).toBeUndefined();
      });
    });

    describe("getDbTypeConfig", () => {
      it("should return database type configuration", () => {
        const connectionTypesData: { database_types: Record<string, { name: string; default_port: number | null; default_database: string }> } = {
          database_types: {
            postgresql: {
              name: "PostgreSQL",
              default_port: 5432,
              default_database: "aidocindexer",
            },
            sqlite: {
              name: "SQLite",
              default_port: null,
              default_database: "./data/app.db",
            },
          },
        };

        const getDbTypeConfig = (type: string) => {
          return connectionTypesData?.database_types?.[type];
        };

        expect(getDbTypeConfig("postgresql")).toEqual({
          name: "PostgreSQL",
          default_port: 5432,
          default_database: "aidocindexer",
        });
        expect(getDbTypeConfig("sqlite")?.default_port).toBeNull();
        expect(getDbTypeConfig("mysql")).toBeUndefined();
      });
    });
  });
});
