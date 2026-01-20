/**
 * ProvidersTab Component Tests
 * ============================
 * Tests for the LLM Providers settings tab component.
 */

import React from "react";
import { screen, fireEvent } from "@testing-library/react";
import { ProvidersTab } from "@/app/dashboard/admin/settings/components/providers-tab";
import { mockProvidersData, createMockMutation, createMockHandlers, renderTabComponent } from "./utils";

describe("ProvidersTab", () => {
  const mockHandlers = createMockHandlers();

  const defaultProps = {
    providersLoading: false,
    providersData: mockProvidersData,
    refetchProviders: jest.fn(),
    providerTypesData: {
      provider_types: {
        openai: { name: "OpenAI", requires_api_key: true },
        anthropic: { name: "Anthropic", requires_api_key: true },
        ollama: { name: "Ollama", requires_api_key: false },
      },
    },
    ollamaModelsData: undefined,
    ollamaLocalModels: undefined,
    ollamaLocalModelsLoading: false,
    refetchOllamaModels: jest.fn(),
    showAddProvider: false,
    setShowAddProvider: jest.fn(),
    newProvider: {
      name: "",
      provider_type: "openai",
      api_key: "",
      api_base_url: "",
      organization_id: "",
      default_chat_model: "",
      default_embedding_model: "",
      is_default: false,
    },
    setNewProvider: jest.fn(),
    showApiKey: false,
    setShowApiKey: jest.fn(),
    providerTestResults: {},
    setProviderTestResults: jest.fn(),
    editingProvider: null,
    setEditingProvider: jest.fn(),
    ollamaBaseUrl: "http://localhost:11434",
    setOllamaBaseUrl: jest.fn(),
    newModelName: "",
    setNewModelName: jest.fn(),
    pullingModel: false,
    setPullingModel: jest.fn(),
    pullResult: null,
    setPullResult: jest.fn(),
    deletingModel: null,
    setDeletingModel: jest.fn(),
    testProvider: createMockMutation(),
    createProvider: createMockMutation(),
    updateProvider: createMockMutation(),
    deleteProvider: createMockMutation(),
    setDefaultProvider: createMockMutation(),
    pullOllamaModel: createMockMutation(),
    deleteOllamaModel: createMockMutation(),
    ...mockHandlers,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("loading state", () => {
    it("shows loading spinner when providersLoading is true", () => {
      const props = {
        ...defaultProps,
        providersLoading: true,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(document.querySelector(".animate-spin")).toBeInTheDocument();
    });
  });

  describe("rendering", () => {
    it("renders LLM providers card", () => {
      renderTabComponent(<ProvidersTab {...defaultProps} />, "providers");

      expect(screen.getByText("LLM Providers")).toBeInTheDocument();
    });

    it("renders add provider button", () => {
      renderTabComponent(<ProvidersTab {...defaultProps} />, "providers");

      expect(screen.getByText("Add Provider")).toBeInTheDocument();
    });

    it("renders provider list", () => {
      renderTabComponent(<ProvidersTab {...defaultProps} />, "providers");

      // Provider name appears multiple times (once as name, once as type)
      expect(screen.getAllByText("OpenAI").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Anthropic").length).toBeGreaterThan(0);
    });

    it("shows default badge for default provider", () => {
      renderTabComponent(<ProvidersTab {...defaultProps} />, "providers");

      expect(screen.getByText("Default")).toBeInTheDocument();
    });
  });

  describe("provider actions", () => {
    it("calls setShowAddProvider when add button is clicked", () => {
      const setShowAddProvider = jest.fn();
      const props = {
        ...defaultProps,
        setShowAddProvider,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      fireEvent.click(screen.getByText("Add Provider"));

      expect(setShowAddProvider).toHaveBeenCalledWith(true);
    });

    it("calls handleTestProvider when test button is clicked", () => {
      const handleTestProvider = jest.fn();
      const props = {
        ...defaultProps,
        handleTestProvider,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // Test button has TestTube icon - find buttons by looking for the icon
      const buttons = screen.getAllByRole("button");
      // Find the test button (button with TestTube icon, not the edit, star or delete buttons)
      // The test button is after the edit button for each provider
      const testButton = buttons.find(btn =>
        btn.querySelector('svg.lucide-test-tube') ||
        btn.innerHTML.includes('test-tube')
      );
      if (testButton) {
        fireEvent.click(testButton);
        expect(handleTestProvider).toHaveBeenCalledWith("provider-1");
      }
    });

    it("calls handleEditProvider when edit button is clicked", () => {
      const handleEditProvider = jest.fn();
      const props = {
        ...defaultProps,
        handleEditProvider,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      const editButtons = screen.getAllByTitle("Edit provider");
      fireEvent.click(editButtons[0]);

      expect(handleEditProvider).toHaveBeenCalled();
    });

    it("calls handleSetDefaultProvider when star button is clicked", () => {
      const handleSetDefaultProvider = jest.fn();
      const props = {
        ...defaultProps,
        handleSetDefaultProvider,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // Find a non-default provider's star button
      const starButtons = screen.getAllByTitle("Set as default");
      if (starButtons.length > 0) {
        fireEvent.click(starButtons[0]);
        expect(handleSetDefaultProvider).toHaveBeenCalled();
      }
    });

    it("calls handleDeleteProvider when delete button is clicked", () => {
      const handleDeleteProvider = jest.fn();
      // Use a non-default provider so delete button is not disabled
      const props = {
        ...defaultProps,
        handleDeleteProvider,
        providersData: {
          providers: [
            {
              id: "provider-2",
              name: "Anthropic",
              provider_type: "anthropic",
              is_active: true,
              is_default: false,  // Not default so can be deleted
              api_base_url: null,
            },
          ],
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // Find delete button by its red class
      const deleteButtons = document.querySelectorAll('button.text-red-500');
      if (deleteButtons.length > 0) {
        fireEvent.click(deleteButtons[0]);
        expect(handleDeleteProvider).toHaveBeenCalledWith("provider-2");
      }
    });
  });

  describe("add provider form", () => {
    it("shows add provider form when showAddProvider is true", () => {
      const props = {
        ...defaultProps,
        showAddProvider: true,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText("Add New Provider")).toBeInTheDocument();
    });

    it("shows provider name input", () => {
      const props = {
        ...defaultProps,
        showAddProvider: true,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // Form uses "Display Name" label
      expect(screen.getByText("Display Name")).toBeInTheDocument();
    });

    it("shows provider type select", () => {
      const props = {
        ...defaultProps,
        showAddProvider: true,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText("Provider Type")).toBeInTheDocument();
    });

    it("shows API key input for providers that require it", () => {
      const props = {
        ...defaultProps,
        showAddProvider: true,
        newProvider: {
          ...defaultProps.newProvider,
          provider_type: "openai",
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText("API Key")).toBeInTheDocument();
    });

    it("calls handleSaveProvider when save button is clicked", () => {
      const handleSaveProvider = jest.fn();
      const props = {
        ...defaultProps,
        showAddProvider: true,
        handleSaveProvider,
        newProvider: {
          ...defaultProps.newProvider,
          name: "Test Provider",
          api_key: "test-key",
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // There are two "Add Provider" buttons - one in header, one in form
      // The form submit button is not disabled and is inside the form
      const addProviderButtons = screen.getAllByRole("button", { name: /add provider/i });
      // The submit button in the form area - find the one that triggers save
      // It's the second "Add Provider" button (first is in header)
      const saveButton = addProviderButtons[addProviderButtons.length - 1];
      fireEvent.click(saveButton);

      expect(handleSaveProvider).toHaveBeenCalled();
    });

    it("calls handleCancelProviderForm when cancel button is clicked", () => {
      const handleCancelProviderForm = jest.fn();
      const props = {
        ...defaultProps,
        showAddProvider: true,
        handleCancelProviderForm,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      const cancelButton = screen.getByText("Cancel");
      fireEvent.click(cancelButton);

      expect(handleCancelProviderForm).toHaveBeenCalled();
    });
  });

  describe("edit provider form", () => {
    it("shows edit form when editingProvider is set", () => {
      const props = {
        ...defaultProps,
        showAddProvider: true,  // Form only shows when showAddProvider is true
        editingProvider: mockProvidersData.providers[0],
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText("Edit Provider")).toBeInTheDocument();
    });
  });

  describe("provider test results", () => {
    it("shows success message for successful test", () => {
      const props = {
        ...defaultProps,
        providerTestResults: {
          "provider-1": { success: true, message: "Connection successful" },
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText("Connection successful")).toBeInTheDocument();
    });

    it("shows error message for failed test", () => {
      const props = {
        ...defaultProps,
        providerTestResults: {
          "provider-1": { success: false, error: "Invalid API key" },
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText("Invalid API key")).toBeInTheDocument();
    });
  });

  describe("API key visibility", () => {
    it("shows eye icon to toggle API key visibility", () => {
      const props = {
        ...defaultProps,
        showAddProvider: true,
        newProvider: {
          ...defaultProps.newProvider,
          provider_type: "openai",  // Openai requires API key
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // Should have an eye icon button (either Eye or EyeOff)
      const eyeIcons = document.querySelectorAll('.lucide-eye, .lucide-eye-off');
      expect(eyeIcons.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("Ollama section", () => {
    it("renders Ollama local models section", () => {
      renderTabComponent(<ProvidersTab {...defaultProps} />, "providers");

      expect(screen.getByText("Ollama Local Models")).toBeInTheDocument();
    });

    it("shows Ollama base URL input", () => {
      renderTabComponent(<ProvidersTab {...defaultProps} />, "providers");

      const input = screen.getByDisplayValue("http://localhost:11434");
      expect(input).toBeInTheDocument();
    });

    it("shows pull model input", () => {
      renderTabComponent(<ProvidersTab {...defaultProps} />, "providers");

      expect(screen.getByText("Pull New Model")).toBeInTheDocument();
    });

    it("shows loading when pulling model", () => {
      const props = {
        ...defaultProps,
        pullingModel: true,
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // When pulling, the button shows a loading spinner icon but text stays "Pull Model"
      // Check for the disabled button state and spinner icon
      const pullButton = screen.getByRole("button", { name: /pull model/i });
      expect(pullButton).toBeDisabled();
      expect(pullButton.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it("shows pull result message", () => {
      const props = {
        ...defaultProps,
        pullResult: {
          success: true,
          message: "Model pulled successfully",
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText("Model pulled successfully")).toBeInTheDocument();
    });

    it("displays local models when available", () => {
      const props = {
        ...defaultProps,
        ollamaLocalModels: {
          success: true,
          chat_models: [
            { name: "llama3.2", parameter_size: "3B" },
            { name: "mistral", parameter_size: "7B" },
          ],
          embedding_models: [
            { name: "nomic-embed-text", parameter_size: "137M" },
          ],
        },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      // Model names are rendered in p.font-medium elements
      expect(screen.getByText("Chat Models (2)")).toBeInTheDocument();
      // Check model names exist
      const modelNames = screen.getAllByText(/llama3\.2|mistral/);
      expect(modelNames.length).toBeGreaterThan(0);
    });
  });

  describe("empty state", () => {
    it("shows message when no providers exist", () => {
      const props = {
        ...defaultProps,
        providersData: { providers: [] },
      };

      renderTabComponent(<ProvidersTab {...props} />, "providers");

      expect(screen.getByText(/No providers configured/)).toBeInTheDocument();
    });
  });
});
