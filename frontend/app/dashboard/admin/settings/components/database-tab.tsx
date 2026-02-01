"use client";

import { RefObject } from "react";
import {
  Database,
  HardDrive,
  Plus,
  Trash2,
  TestTube,
  CheckCircle,
  AlertCircle,
  Loader2,
  Download,
  Upload,
  Server,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyData = any;

interface ConnectionTestResult {
  success: boolean;
  message?: string;
  error?: string;
}

interface TestResult {
  success: boolean;
  message: string;
  has_pgvector?: boolean;
  pgvector_version?: string;
  error?: string;
}

interface DeletedDoc {
  id: string;
  name: string;
  file_type: string;
  file_size: number;
  deleted_at?: string;
}

interface NewConnection {
  name: string;
  db_type: string;
  host: string;
  port: number;
  username: string;
  password: string;
  database: string;
  vector_store: string;
  is_active: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MutationHook = any;

interface DatabaseTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
  dbInfo: AnyData;
  dbInfoLoading: boolean;
  connectionsData: AnyData;
  connectionsLoading: boolean;
  connectionTypesData: AnyData;
  showAddConnection: boolean;
  setShowAddConnection: (value: boolean) => void;
  newConnection: AnyData;
  setNewConnection: (value: AnyData) => void;
  connectionTestResults: AnyData;
  deletedDocs: AnyData;
  deletedDocsTotal: number;
  deletedDocsPage: number;
  deletedDocsLoading: boolean;
  deletedDocsError: string | null;
  restoringDocId: string | null;
  hardDeletingDocId: string | null;
  selectedDeletedDocs: Set<string>;
  isBulkDeleting: boolean;
  isBulkRestoring: boolean;
  newDbUrl: string;
  setNewDbUrl: (value: string) => void;
  testResult: AnyData;
  setTestResult: (value: AnyData) => void;
  importRef: RefObject<HTMLInputElement>;
  testConnection: MutationHook;
  setupPostgres: MutationHook;
  exportDatabase: MutationHook;
  importDatabase: MutationHook;
  createConnection: MutationHook;
  deleteConnection: MutationHook;
  activateConnection: MutationHook;
  testConnectionById: MutationHook;
  getDbTypeConfig: (type: string) => AnyData;
  handleTestConnection: () => void;
  handleSetupPostgres: () => void;
  handleExport: () => void;
  handleImport: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleAddConnection: () => void;
  handleTestSavedConnection: (id: string) => void;
  handleActivateConnection: (id: string) => void;
  handleDeleteConnection: (id: string) => void;
  fetchDeletedDocs: (page: number) => void;
  handleRestoreDocument: (id: string) => void;
  handleHardDeleteDocument: (id: string, name: string) => void;
  handleBulkRestore: () => void;
  handleBulkPermanentDelete: () => void;
  toggleSelectAllDeletedDocs: () => void;
  toggleDeletedDocSelection: (id: string) => void;
}

export function DatabaseTab({
  localSettings,
  handleSettingChange,
  dbInfo,
  dbInfoLoading,
  connectionsData,
  connectionsLoading,
  connectionTypesData,
  showAddConnection,
  setShowAddConnection,
  newConnection,
  setNewConnection,
  connectionTestResults,
  deletedDocs,
  deletedDocsTotal,
  deletedDocsPage,
  deletedDocsLoading,
  deletedDocsError,
  restoringDocId,
  hardDeletingDocId,
  selectedDeletedDocs,
  isBulkDeleting,
  isBulkRestoring,
  newDbUrl,
  setNewDbUrl,
  testResult,
  setTestResult,
  importRef,
  testConnection,
  setupPostgres,
  exportDatabase,
  importDatabase,
  createConnection,
  deleteConnection,
  activateConnection,
  testConnectionById,
  getDbTypeConfig,
  handleTestConnection,
  handleSetupPostgres,
  handleExport,
  handleImport,
  handleAddConnection,
  handleTestSavedConnection,
  handleActivateConnection,
  handleDeleteConnection,
  fetchDeletedDocs,
  handleRestoreDocument,
  handleHardDeleteDocument,
  handleBulkRestore,
  handleBulkPermanentDelete,
  toggleSelectAllDeletedDocs,
  toggleDeletedDocSelection,
}: DatabaseTabProps) {
  return (
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
              <Select
                value={localSettings["database.index_type"] as string || "hnsw"}
                onValueChange={(value) => handleSettingChange("database.index_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hnsw">HNSW</SelectItem>
                  <SelectItem value="ivfflat">IVFFlat</SelectItem>
                </SelectContent>
              </Select>
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
                {connectionsData.connections.map((conn: AnyData) => (
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
                    <Select
                      value={newConnection.db_type}
                      onValueChange={(type) => {
                        const config = getDbTypeConfig(type);
                        setNewConnection({
                          ...newConnection,
                          db_type: type,
                          port: config?.default_port || 5432,
                          database: config?.default_database || "aidocindexer",
                        });
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {connectionTypesData?.database_types &&
                          Object.entries(connectionTypesData.database_types).map(([key, config]: [string, AnyData]) => (
                            <SelectItem key={key} value={key}>
                              {(config as AnyData).name}
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
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
                    <Select
                      value={newConnection.vector_store}
                      onValueChange={(value) => setNewConnection({ ...newConnection, vector_store: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="auto">Auto-detect</SelectItem>
                        <SelectItem value="pgvector">pgvector</SelectItem>
                        <SelectItem value="chromadb">ChromaDB</SelectItem>
                      </SelectContent>
                    </Select>
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
              {deletedDocs.map((doc: AnyData) => (
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
  );
}
