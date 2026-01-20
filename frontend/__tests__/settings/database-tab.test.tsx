/**
 * DatabaseTab Component Tests
 * ===========================
 * Tests for the Database settings tab component.
 */

import React from "react";
import { screen, fireEvent } from "@testing-library/react";
import { DatabaseTab } from "@/app/dashboard/admin/settings/components/database-tab";
import {
  mockLocalSettings,
  mockDbInfo,
  mockConnectionsData,
  mockConnectionTypesData,
  mockDeletedDocs,
  createMockMutation,
  createMockHandlers,
  renderTabComponent,
} from "./utils";

describe("DatabaseTab", () => {
  const mockHandlers = createMockHandlers();
  const importRef = React.createRef<HTMLInputElement>();

  const defaultProps = {
    localSettings: mockLocalSettings,
    handleSettingChange: jest.fn(),
    dbInfo: mockDbInfo,
    dbInfoLoading: false,
    connectionsData: mockConnectionsData,
    connectionsLoading: false,
    connectionTypesData: mockConnectionTypesData,
    showAddConnection: false,
    setShowAddConnection: jest.fn(),
    newConnection: {
      name: "",
      db_type: "postgresql",
      host: "localhost",
      port: 5432,
      username: "",
      password: "",
      database: "aidocindexer",
      vector_store: "auto",
      is_active: false,
    },
    setNewConnection: jest.fn(),
    connectionTestResults: {},
    deletedDocs: [],
    deletedDocsTotal: 0,
    deletedDocsPage: 1,
    deletedDocsLoading: false,
    deletedDocsError: null,
    restoringDocId: null,
    hardDeletingDocId: null,
    selectedDeletedDocs: new Set<string>(),
    isBulkDeleting: false,
    isBulkRestoring: false,
    newDbUrl: "",
    setNewDbUrl: jest.fn(),
    testResult: null,
    setTestResult: jest.fn(),
    importRef,
    testConnection: createMockMutation(),
    setupPostgres: createMockMutation(),
    exportDatabase: createMockMutation(),
    importDatabase: createMockMutation(),
    createConnection: createMockMutation(),
    deleteConnection: createMockMutation(),
    activateConnection: createMockMutation(),
    testConnectionById: createMockMutation(),
    ...mockHandlers,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders database settings card", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Database Settings")).toBeInTheDocument();
    });

    it("renders database configuration card", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Database Configuration")).toBeInTheDocument();
    });

    it("renders deleted documents card", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Deleted Documents")).toBeInTheDocument();
    });

    it("renders vector dimensions input", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Vector Dimensions")).toBeInTheDocument();
    });

    it("renders index type select", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Index Type")).toBeInTheDocument();
    });

    it("renders max results input", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Max Results per Query")).toBeInTheDocument();
    });
  });

  describe("database info display", () => {
    it("shows loading state when dbInfoLoading is true", () => {
      const props = {
        ...defaultProps,
        dbInfoLoading: true,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(document.querySelector(".animate-spin")).toBeInTheDocument();
    });

    it("displays database type", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("PostgreSQL")).toBeInTheDocument();
    });

    it("displays vector store type", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("pgvector")).toBeInTheDocument();
    });

    it("displays document count", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("150")).toBeInTheDocument();
    });

    it("displays chunks count", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("2500")).toBeInTheDocument();
    });

    it("displays users count", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("5")).toBeInTheDocument();
    });

    it("displays connected status", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Connected")).toBeInTheDocument();
    });

    it("displays disconnected status when not connected", () => {
      const props = {
        ...defaultProps,
        dbInfo: {
          ...mockDbInfo,
          is_connected: false,
        },
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("Disconnected")).toBeInTheDocument();
    });
  });

  describe("connections list", () => {
    it("shows loading state when connections are loading", () => {
      const props = {
        ...defaultProps,
        connectionsLoading: true,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      // Multiple spinners might exist, just check that at least one exists
      const spinners = document.querySelectorAll(".animate-spin");
      expect(spinners.length).toBeGreaterThan(0);
    });

    it("displays saved connections", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Production DB")).toBeInTheDocument();
    });

    it("shows active badge for active connection", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Active")).toBeInTheDocument();
    });

    it("shows no connections message when empty", () => {
      const props = {
        ...defaultProps,
        connectionsData: { connections: [] },
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText(/No saved connections/)).toBeInTheDocument();
    });
  });

  describe("add connection form", () => {
    it("shows add connection button", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Add Connection")).toBeInTheDocument();
    });

    it("calls setShowAddConnection when add button is clicked", () => {
      const setShowAddConnection = jest.fn();
      const props = {
        ...defaultProps,
        setShowAddConnection,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      fireEvent.click(screen.getByText("Add Connection"));

      expect(setShowAddConnection).toHaveBeenCalledWith(true);
    });

    it("shows add connection form when showAddConnection is true", () => {
      const props = {
        ...defaultProps,
        showAddConnection: true,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("Add New Connection")).toBeInTheDocument();
      expect(screen.getByText("Connection Name")).toBeInTheDocument();
      expect(screen.getByText("Database Type")).toBeInTheDocument();
    });

    it("shows host and port fields for postgresql", () => {
      const props = {
        ...defaultProps,
        showAddConnection: true,
        newConnection: {
          ...defaultProps.newConnection,
          db_type: "postgresql",
        },
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("Host")).toBeInTheDocument();
      expect(screen.getByText("Port")).toBeInTheDocument();
    });
  });

  describe("connection actions", () => {
    it("calls handleTestSavedConnection when test button is clicked", () => {
      const handleTestSavedConnection = jest.fn();
      const props = {
        ...defaultProps,
        handleTestSavedConnection,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      const testButtons = screen.getAllByTitle("Test connection");
      fireEvent.click(testButtons[0]);

      expect(handleTestSavedConnection).toHaveBeenCalledWith("conn-1");
    });

    it("calls handleDeleteConnection when delete button is clicked", () => {
      const handleDeleteConnection = jest.fn();
      const props = {
        ...defaultProps,
        handleDeleteConnection,
        connectionsData: {
          connections: [
            {
              ...mockConnectionsData.connections[0],
              is_active: false,
            },
          ],
        },
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      const deleteButtons = screen.getAllByTitle("Delete connection");
      fireEvent.click(deleteButtons[0]);

      expect(handleDeleteConnection).toHaveBeenCalledWith("conn-1");
    });
  });

  describe("data migration", () => {
    it("renders export button", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Export Data")).toBeInTheDocument();
    });

    it("renders import button", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Import Data")).toBeInTheDocument();
    });

    it("calls handleExport when export button is clicked", () => {
      const handleExport = jest.fn();
      const props = {
        ...defaultProps,
        handleExport,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      fireEvent.click(screen.getByText("Export Data"));

      expect(handleExport).toHaveBeenCalled();
    });
  });

  describe("switch database section", () => {
    it("renders PostgreSQL connection URL input", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("PostgreSQL Connection URL")).toBeInTheDocument();
    });

    it("renders test connection button", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Test Connection")).toBeInTheDocument();
    });

    it("renders setup pgvector button", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Setup pgvector")).toBeInTheDocument();
    });

    it("calls handleTestConnection when test button is clicked", () => {
      const handleTestConnection = jest.fn();
      const props = {
        ...defaultProps,
        handleTestConnection,
        newDbUrl: "postgresql://user:pass@localhost:5432/db",
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      fireEvent.click(screen.getByText("Test Connection"));

      expect(handleTestConnection).toHaveBeenCalled();
    });

    it("shows test result when available", () => {
      const props = {
        ...defaultProps,
        testResult: {
          success: true,
          message: "Connection successful!",
          has_pgvector: true,
          pgvector_version: "0.5.1",
        },
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("Connection successful!")).toBeInTheDocument();
    });
  });

  describe("deleted documents", () => {
    it("shows load deleted docs button", () => {
      renderTabComponent(<DatabaseTab {...defaultProps} />, "database");

      expect(screen.getByText("Load Deleted Docs")).toBeInTheDocument();
    });

    it("calls fetchDeletedDocs when button is clicked", () => {
      const fetchDeletedDocs = jest.fn();
      const props = {
        ...defaultProps,
        fetchDeletedDocs,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      fireEvent.click(screen.getByText("Load Deleted Docs"));

      expect(fetchDeletedDocs).toHaveBeenCalledWith(1);
    });

    it("shows deleted docs count", () => {
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText(/2 deleted documents? found/)).toBeInTheDocument();
    });

    it("displays deleted document names", () => {
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("test-document.pdf")).toBeInTheDocument();
      expect(screen.getByText("report.docx")).toBeInTheDocument();
    });

    it("shows restore button for each document", () => {
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      const restoreButtons = screen.getAllByText("Restore");
      expect(restoreButtons.length).toBe(2);
    });

    it("calls handleRestoreDocument when restore is clicked", () => {
      const handleRestoreDocument = jest.fn();
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
        handleRestoreDocument,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      const restoreButtons = screen.getAllByText("Restore");
      fireEvent.click(restoreButtons[0]);

      expect(handleRestoreDocument).toHaveBeenCalledWith("doc-1");
    });

    it("shows error message when deletedDocsError is set", () => {
      const props = {
        ...defaultProps,
        deletedDocsError: "Failed to load deleted documents",
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("Failed to load deleted documents")).toBeInTheDocument();
    });

    it("shows loading state when loading deleted docs", () => {
      const props = {
        ...defaultProps,
        deletedDocsLoading: true,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      // Button should show loading state
      expect(screen.getByText("Load Deleted Docs").closest("button")).toBeDisabled();
    });
  });

  describe("bulk actions", () => {
    it("shows bulk action bar when documents are selected", () => {
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
        selectedDeletedDocs: new Set(["doc-1"]),
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("1 selected")).toBeInTheDocument();
    });

    it("shows restore selected button", () => {
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
        selectedDeletedDocs: new Set(["doc-1"]),
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("Restore Selected")).toBeInTheDocument();
    });

    it("shows delete selected button", () => {
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
        selectedDeletedDocs: new Set(["doc-1"]),
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      expect(screen.getByText("Delete Selected")).toBeInTheDocument();
    });

    it("calls handleBulkRestore when restore selected is clicked", () => {
      const handleBulkRestore = jest.fn();
      const props = {
        ...defaultProps,
        deletedDocs: mockDeletedDocs,
        deletedDocsTotal: 2,
        selectedDeletedDocs: new Set(["doc-1"]),
        handleBulkRestore,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      fireEvent.click(screen.getByText("Restore Selected"));

      expect(handleBulkRestore).toHaveBeenCalled();
    });
  });

  describe("settings inputs", () => {
    it("calls handleSettingChange when vector dimensions is changed", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      const input = screen.getByDisplayValue("1536");
      fireEvent.change(input, { target: { value: "768" } });

      expect(handleSettingChange).toHaveBeenCalledWith(
        "database.vector_dimensions",
        768
      );
    });

    it("calls handleSettingChange when index type is changed", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
      };

      renderTabComponent(<DatabaseTab {...props} />, "database");

      // Select shows "HNSW" text, not "hnsw" value
      // Find the select by finding the element with option "HNSW"
      const selects = document.querySelectorAll('select');
      const indexTypeSelect = Array.from(selects).find(select =>
        Array.from(select.options).some(opt => opt.value === "hnsw")
      );
      expect(indexTypeSelect).toBeInTheDocument();
      fireEvent.change(indexTypeSelect!, { target: { value: "ivfflat" } });

      expect(handleSettingChange).toHaveBeenCalledWith(
        "database.index_type",
        "ivfflat"
      );
    });
  });
});
