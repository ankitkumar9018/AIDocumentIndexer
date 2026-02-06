/**
 * Settings Tests - Test Utilities
 * ================================
 * Shared utilities, mocks, and wrappers for settings tab tests.
 */

import React from "react";
import { render, RenderOptions } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Create a wrapper with React Query provider
export function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

interface WrapperProps {
  children: React.ReactNode;
}

export function createWrapper() {
  const queryClient = createTestQueryClient();
  return function Wrapper({ children }: WrapperProps) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

// Custom render with providers
export function renderWithProviders(
  ui: React.ReactElement,
  options?: Omit<RenderOptions, "wrapper">
) {
  const Wrapper = createWrapper();
  return render(ui, { wrapper: Wrapper, ...options });
}

// Mock mutation hook factory
export function createMockMutation(overrides = {}) {
  return {
    mutate: jest.fn(),
    mutateAsync: jest.fn().mockResolvedValue({}),
    isPending: false,
    isSuccess: false,
    isError: false,
    error: null,
    data: undefined,
    reset: jest.fn(),
    ...overrides,
  };
}

// Mock data factories
export const mockLocalSettings: Record<string, unknown> = {
  "database.vector_dimensions": 1536,
  "database.index_type": "hnsw",
  "database.max_results_per_query": 10,
  "auth.require_email_verification": false,
  "notifications.email_enabled": true,
  "notifications.system_alerts": true,
  "generation.include_sources": true,
  "generation.include_images": true,
  "generation.image_backend": "picsum",
  "generation.enable_quality_review": true,
  "generation.min_quality_score": 0.7,
};

export const mockDbInfo = {
  type: "postgresql",
  vector_store: "pgvector",
  documents_count: 150,
  chunks_count: 2500,
  users_count: 5,
  is_connected: true,
};

export const mockProvidersData = {
  providers: [
    {
      id: "provider-1",
      name: "OpenAI",
      provider_type: "openai",
      is_active: true,
      is_default: true,
      api_base_url: null,
      organization_id: null,
      default_chat_model: "gpt-4",
      default_embedding_model: "text-embedding-3-small",
      settings: null,
      api_key_masked: "sk-...abc",
      has_api_key: true,
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
    },
    {
      id: "provider-2",
      name: "Anthropic",
      provider_type: "anthropic",
      is_active: true,
      is_default: false,
      api_base_url: null,
      organization_id: null,
      default_chat_model: "claude-sonnet-4-5-20250929",
      default_embedding_model: null,
      settings: null,
      api_key_masked: "sk-...xyz",
      has_api_key: true,
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
    },
  ],
};

export const mockConnectionsData = {
  connections: [
    {
      id: "conn-1",
      name: "Production DB",
      db_type: "postgresql",
      host: "localhost",
      port: 5432,
      database: "aidocindexer",
      is_active: true,
    },
  ],
};

export const mockConnectionTypesData = {
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

export const mockOcrData = {
  settings: {
    "ocr.provider": "paddleocr",
    "ocr.paddle.variant": "server",
    "ocr.paddle.languages": ["en", "de"],
    "ocr.paddle.auto_download": true,
    "ocr.tesseract.fallback_enabled": true,
  },
  models: {
    model_dir: "./data/paddle_models",
    status: "installed",
    total_size: "250 MB",
    downloaded: [
      { name: "en_PP-OCRv4_det", size: "4.5 MB" },
      { name: "en_PP-OCRv4_rec", size: "12 MB" },
    ],
  },
};

export const mockDeletedDocs = [
  {
    id: "doc-1",
    name: "test-document.pdf",
    file_type: "pdf",
    file_size: 1024000,
    deleted_at: "2024-01-15T10:30:00Z",
  },
  {
    id: "doc-2",
    name: "report.docx",
    file_type: "docx",
    file_size: 512000,
    deleted_at: "2024-01-14T14:20:00Z",
  },
];

export const mockHealthData = {
  status: "healthy",
  services: {
    database: { status: "healthy", message: "Connected" },
    vector_store: { status: "healthy", message: "pgvector ready" },
    llm: { status: "healthy", message: "OpenAI connected" },
  },
};

export const mockPresetsData = {
  presets: [
    { id: "preset-1", name: "High Performance", description: "Optimized for speed" },
    { id: "preset-2", name: "High Quality", description: "Optimized for accuracy" },
  ],
};

// Handler mock factory
export function createMockHandlers() {
  return {
    handleSettingChange: jest.fn(),
    handleTestConnection: jest.fn(),
    handleSetupPostgres: jest.fn(),
    handleExport: jest.fn(),
    handleImport: jest.fn(),
    handleAddConnection: jest.fn(),
    handleTestSavedConnection: jest.fn(),
    handleActivateConnection: jest.fn(),
    handleDeleteConnection: jest.fn(),
    fetchDeletedDocs: jest.fn(),
    handleRestoreDocument: jest.fn(),
    handleHardDeleteDocument: jest.fn(),
    handleBulkRestore: jest.fn(),
    handleBulkPermanentDelete: jest.fn(),
    toggleSelectAllDeletedDocs: jest.fn(),
    toggleDeletedDocSelection: jest.fn(),
    handleEditProvider: jest.fn(),
    handleTestProvider: jest.fn(),
    handleSetDefaultProvider: jest.fn(),
    handleDeleteProvider: jest.fn(),
    handleSaveProvider: jest.fn(),
    handleCancelProviderForm: jest.fn(),
    getProviderTypeConfig: jest.fn().mockReturnValue({
      name: "OpenAI",
      fields: ["api_key", "api_base_url", "organization_id"],
      default_model: "gpt-4",
      default_chat_model: "gpt-4",
      default_embedding_model: "text-embedding-3-small",
      chat_models: ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
      embedding_models: ["text-embedding-3-small", "text-embedding-3-large"],
    }),
    getDbTypeConfig: jest.fn().mockReturnValue({ name: "PostgreSQL", default_port: 5432 }),
  };
}

// Re-export testing library utilities
export * from "@testing-library/react";

// Wrapper for tab components that need Tabs context
interface TabsWrapperProps {
  children: React.ReactNode;
  defaultValue: string;
}

export function TabsWrapper({ children, defaultValue }: TabsWrapperProps) {
  // Import Tabs from radix at runtime to avoid circular dependencies
  const { Tabs } = require("@/components/ui/tabs");
  return <Tabs defaultValue={defaultValue}>{children}</Tabs>;
}

// Custom render with Tabs context for tab components
export function renderTabComponent(
  ui: React.ReactElement,
  tabValue: string,
  options?: Omit<RenderOptions, "wrapper">
) {
  const Wrapper = createWrapper();
  const { Tabs } = require("@/components/ui/tabs");

  return render(
    <Wrapper>
      <Tabs defaultValue={tabValue}>{ui}</Tabs>
    </Wrapper>,
    options
  );
}
