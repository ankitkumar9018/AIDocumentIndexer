"use client";

import { useState } from "react";
import {
  Database,
  Plus,
  Play,
  History,
  Settings,
  Trash2,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Copy,
  ThumbsUp,
  ThumbsDown,
  Table,
  Code,
  MessageSquare,
  Loader2,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Save,
  Sparkles,
  SlidersHorizontal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table as UITable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { toast } from "sonner";
import { api } from "@/lib/api/client";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { databaseConnectionSchema, databaseQuerySchema, type DatabaseConnectionInput, type DatabaseQueryInput } from "@/lib/validations";

// Types
interface DatabaseConnection {
  id: string;
  name: string;
  description?: string;
  connector_type: string;
  host: string;
  port: number;
  database_name: string;
  username: string;
  ssl_mode?: string;
  schema_name?: string;
  max_rows: number;
  query_timeout_seconds: number;
  is_active: boolean;
  last_tested_at?: string;
  last_test_success?: boolean;
  total_queries: number;
  created_at: string;
  updated_at?: string;
}

interface QueryResult {
  success: boolean;
  natural_language_query: string;
  generated_sql?: string;
  explanation?: string;
  answer?: string;
  columns: string[];
  rows: any[][];
  row_count: number;
  execution_time_ms: number;
  confidence: number;
  error?: string;
  query_id?: string;
  // Enhanced metadata
  complexity?: string;
  schema_tables_used?: string[];
  candidates_generated?: number;
  features_used?: string[];
}

interface SchemaAnnotations {
  tables: Record<string, { description?: string; columns?: Record<string, string> }>;
  glossary: Record<string, string>;
}

interface SchemaTable {
  name: string;
  columns: { name: string; data_type: string; description?: string }[];
  description?: string;
}

interface QueryHistoryItem {
  id: string;
  natural_language_query: string;
  generated_sql: string;
  explanation?: string;
  execution_success: boolean;
  execution_time_ms: number;
  row_count: number;
  confidence_score: number;
  user_rating?: number;
  created_at: string;
}

// Database type icons
const databaseIcons: Record<string, string> = {
  postgresql: "🐘",
  mysql: "🐬",
  mongodb: "🍃",
  sqlite: "📦",
};

export default function DatabasePage() {
  const { data: session } = useSession();
  const accessToken = (session as any)?.accessToken as string | undefined;
  const queryClient = useQueryClient();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [selectedConnection, setSelectedConnection] = useState<DatabaseConnection | null>(null);
  const [query, setQuery] = useState("");
  const [activeTab, setActiveTab] = useState("results");
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  // Schema annotation editor state
  const [isSchemaEditorOpen, setIsSchemaEditorOpen] = useState(false);
  const [annotations, setAnnotations] = useState<SchemaAnnotations>({ tables: {}, glossary: {} });
  const [schemaData, setSchemaData] = useState<SchemaTable[]>([]);
  const [expandedTables, setExpandedTables] = useState<Set<string>>(new Set());
  const [glossaryNewTerm, setGlossaryNewTerm] = useState("");
  const [glossaryNewDef, setGlossaryNewDef] = useState("");
  const [isSavingAnnotations, setIsSavingAnnotations] = useState(false);
  const [isRawDataOpen, setIsRawDataOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("__default__");

  // Advanced options state
  const [schemaLinking, setSchemaLinking] = useState(true);
  const [sampleRows, setSampleRows] = useState(true);
  const [selfConsistency, setSelfConsistency] = useState(false);
  const [dynamicExamples, setDynamicExamples] = useState(true);
  const [complexityRouting, setComplexityRouting] = useState(true);
  const [numExamples, setNumExamples] = useState(3);

  // Form with Zod validation for new connection
  const connectionForm = useForm<DatabaseConnectionInput>({
    resolver: zodResolver(databaseConnectionSchema),
    defaultValues: {
      name: "",
      connectorType: "postgresql",
      host: "",
      port: 5432,
      database: "",
      username: "",
      password: "",
      sslMode: "prefer",
    },
  });

  // Legacy state for compatibility (bridging to form)
  const newConnection = {
    name: connectionForm.watch("name"),
    description: "",
    connector_type: connectionForm.watch("connectorType"),
    host: connectionForm.watch("host"),
    port: connectionForm.watch("port"),
    database_name: connectionForm.watch("database"),
    username: connectionForm.watch("username"),
    password: connectionForm.watch("password"),
    ssl_mode: connectionForm.watch("sslMode"),
    schema_name: "public",
    max_rows: 1000,
    query_timeout_seconds: 30,
  };

  // Update form when connector type changes (adjust default port)
  const watchConnectorType = connectionForm.watch("connectorType");
  const updatePortForConnector = (type: string) => {
    const ports: Record<string, number> = {
      postgresql: 5432,
      mysql: 3306,
      mongodb: 27017,
      sqlite: 0,
    };
    connectionForm.setValue("port", ports[type] || 5432);
  };

  // Fetch connections
  const { data: connections = [], isLoading, refetch } = useQuery({
    queryKey: ["database-connections"],
    queryFn: async () => {
      if (!accessToken) return [];
      const response = await fetch(`${API_BASE}/database/connections`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!response.ok) throw new Error("Failed to fetch connections");
      return response.json();
    },
    enabled: !!accessToken,
  });

  // Fetch query history for selected connection
  const { data: queryHistory = [] } = useQuery({
    queryKey: ["database-history", selectedConnection?.id],
    queryFn: async () => {
      if (!selectedConnection || !accessToken) return [];
      const response = await fetch(
        `${API_BASE}/database/connections/${selectedConnection.id}/history`,
        { headers: { Authorization: `Bearer ${accessToken}` } }
      );
      if (!response.ok) throw new Error("Failed to fetch history");
      return response.json();
    },
    enabled: !!selectedConnection && !!accessToken,
  });

  // Fetch available models from configured LLM providers
  const { data: availableModels = [] } = useQuery({
    queryKey: ["provider-models"],
    queryFn: async () => {
      try {
        const { providers } = await api.listLLMProviders();
        const activeProviders = providers.filter((p) => p.is_active);
        // Only show models explicitly configured as default_chat_model on each provider
        const seen = new Set<string>();
        const models: { name: string; provider: string }[] = [];
        for (const provider of activeProviders) {
          if (provider.default_chat_model && !seen.has(provider.default_chat_model)) {
            seen.add(provider.default_chat_model);
            models.push({ name: provider.default_chat_model, provider: provider.name });
          }
        }
        return models;
      } catch {
        return [];
      }
    },
    staleTime: 60000,
  });

  // Create connection mutation
  const createMutation = useMutation({
    mutationFn: async (data: typeof newConnection) => {
      const response = await fetch(`${API_BASE}/database/connections`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to create connection");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["database-connections"] });
      setIsCreateDialogOpen(false);
      resetForm();
      toast.success("Database connection created");
    },
    onError: (error) => {
      toast.error(`Failed to create connection: ${error.message}`);
    },
  });

  // Test connection mutation
  const testMutation = useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`${API_BASE}/database/connections/${id}/test`, {
        method: "POST",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!response.ok) throw new Error("Test failed");
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["database-connections"] });
      if (data.success) {
        toast.success(`Connection successful (${data.latency_ms?.toFixed(0)}ms)`);
      } else {
        toast.error(`Connection failed: ${data.message}`);
      }
    },
    onError: () => {
      toast.error("Failed to test connection");
    },
  });

  // Delete connection mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(`${API_BASE}/database/connections/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!response.ok) throw new Error("Delete failed");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["database-connections"] });
      if (selectedConnection) setSelectedConnection(null);
      toast.success("Connection deleted");
    },
    onError: () => {
      toast.error("Failed to delete connection");
    },
  });

  // Submit feedback mutation
  const feedbackMutation = useMutation({
    mutationFn: async ({
      queryId,
      rating,
      isCorrect,
    }: {
      queryId: string;
      rating: number;
      isCorrect: boolean;
    }) => {
      const response = await fetch(`${API_BASE}/database/history/${queryId}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ rating, is_correct: isCorrect }),
      });
      if (!response.ok) throw new Error("Feedback failed");
      return response.json();
    },
    onSuccess: () => {
      toast.success("Feedback submitted");
    },
  });

  const resetForm = () => {
    connectionForm.reset({
      name: "",
      connectorType: "postgresql",
      host: "",
      port: 5432,
      database: "",
      username: "",
      password: "",
      sslMode: "prefer",
    });
  };

  const handleCreateConnection = connectionForm.handleSubmit((data) => {
    // Map form data to API format
    const apiData = {
      name: data.name,
      description: "",
      connector_type: data.connectorType,
      host: data.host,
      port: data.port,
      database_name: data.database,
      username: data.username,
      password: data.password,
      ssl_mode: data.sslMode,
      schema_name: "public",
      max_rows: 1000,
      query_timeout_seconds: 30,
      // MongoDB specific
      auth_source: data.authSource,
      replica_set: data.replicaSet,
    };
    createMutation.mutate(apiData);
  });

  const handleQuery = async () => {
    if (!selectedConnection || !query.trim() || !accessToken) return;

    setIsQuerying(true);
    setQueryResult(null);

    try {
      const response = await fetch(
        `${API_BASE}/database/connections/${selectedConnection.id}/query`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          body: JSON.stringify({
            question: query,
            execute: true,
            explain: true,
            model_override: selectedModel !== "__default__" ? selectedModel : undefined,
            schema_linking: schemaLinking,
            sample_rows: sampleRows,
            self_consistency: selfConsistency,
            dynamic_examples: dynamicExamples,
            complexity_routing: complexityRouting,
            num_examples: numExamples,
          }),
        }
      );

      const result = await response.json();
      setQueryResult(result);

      if (result.success) {
        setActiveTab("results");
        toast.success(`Query returned ${result.row_count} rows`);
      } else {
        toast.error(result.error || "Query failed");
      }

      // Refresh history
      queryClient.invalidateQueries({ queryKey: ["database-history", selectedConnection.id] });
    } catch (error) {
      toast.error("Failed to execute query");
    } finally {
      setIsQuerying(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const getStatusBadge = (connection: DatabaseConnection) => {
    if (!connection.is_active) {
      return <Badge variant="secondary">Inactive</Badge>;
    }
    if (connection.last_test_success === true) {
      return <Badge className="bg-green-500">Connected</Badge>;
    }
    if (connection.last_test_success === false) {
      return <Badge variant="destructive">Error</Badge>;
    }
    return <Badge variant="outline">Not Tested</Badge>;
  };

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Database Connector</h1>
          <p className="text-muted-foreground">
            Query your databases using natural language
          </p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Add Database
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle>Add Database Connection</DialogTitle>
              <DialogDescription>
                Connect to a database to query it using natural language
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleCreateConnection}>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Database Type</Label>
                  <Select
                    value={connectionForm.watch("connectorType")}
                    onValueChange={(value: "postgresql" | "mysql" | "mongodb" | "sqlite") => {
                      connectionForm.setValue("connectorType", value);
                      updatePortForConnector(value);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="postgresql">
                        🐘 PostgreSQL
                      </SelectItem>
                      <SelectItem value="mysql">🐬 MySQL</SelectItem>
                      <SelectItem value="mongodb">
                        🍃 MongoDB
                      </SelectItem>
                      <SelectItem value="sqlite">
                        📦 SQLite
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  {connectionForm.formState.errors.connectorType && (
                    <p className="text-sm text-destructive">{connectionForm.formState.errors.connectorType.message}</p>
                  )}
                </div>
                <div className="space-y-2">
                  <Label>Connection Name</Label>
                  <Input
                    placeholder="Production DB"
                    {...connectionForm.register("name")}
                  />
                  {connectionForm.formState.errors.name && (
                    <p className="text-sm text-destructive">{connectionForm.formState.errors.name.message}</p>
                  )}
                </div>
              </div>

              {/* SQLite-specific field: File Path */}
              {connectionForm.watch("connectorType") === "sqlite" ? (
                <div className="space-y-2">
                  <Label>Database File Path</Label>
                  <Input
                    placeholder="/path/to/database.db or :memory:"
                    {...connectionForm.register("host")}
                  />
                  <p className="text-xs text-muted-foreground">
                    Enter the full path to your SQLite database file, or use :memory: for an in-memory database
                  </p>
                  {connectionForm.formState.errors.host && (
                    <p className="text-sm text-destructive">{connectionForm.formState.errors.host.message}</p>
                  )}
                </div>
              ) : (
                <>
                  {/* Standard connection fields for PostgreSQL, MySQL, MongoDB */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="col-span-2 space-y-2">
                      <Label>Host</Label>
                      <Input
                        placeholder="localhost"
                        {...connectionForm.register("host")}
                      />
                      {connectionForm.formState.errors.host && (
                        <p className="text-sm text-destructive">{connectionForm.formState.errors.host.message}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Port</Label>
                      <Input
                        type="number"
                        {...connectionForm.register("port", { valueAsNumber: true })}
                      />
                      {connectionForm.formState.errors.port && (
                        <p className="text-sm text-destructive">{connectionForm.formState.errors.port.message}</p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Database Name</Label>
                    <Input
                      placeholder="myapp_production"
                      {...connectionForm.register("database")}
                    />
                    {connectionForm.formState.errors.database && (
                      <p className="text-sm text-destructive">{connectionForm.formState.errors.database.message}</p>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Username</Label>
                      <Input
                        placeholder="postgres"
                        {...connectionForm.register("username")}
                      />
                      {connectionForm.formState.errors.username && (
                        <p className="text-sm text-destructive">{connectionForm.formState.errors.username.message}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Password</Label>
                      <Input
                        type="password"
                        placeholder="••••••••"
                        {...connectionForm.register("password")}
                      />
                      {connectionForm.formState.errors.password && (
                        <p className="text-sm text-destructive">{connectionForm.formState.errors.password.message}</p>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Schema (PostgreSQL/MySQL only)</Label>
                      <Input
                        placeholder="public"
                        defaultValue="public"
                        disabled={connectionForm.watch("connectorType") === "mongodb"}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>SSL Mode</Label>
                      <Select
                        value={connectionForm.watch("sslMode")}
                        onValueChange={(value: "disable" | "allow" | "prefer" | "require" | "verify-ca" | "verify-full") =>
                          connectionForm.setValue("sslMode", value)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="disable">Disable</SelectItem>
                          <SelectItem value="allow">Allow</SelectItem>
                          <SelectItem value="prefer">Prefer</SelectItem>
                          <SelectItem value="require">Require</SelectItem>
                          <SelectItem value="verify-ca">Verify CA</SelectItem>
                          <SelectItem value="verify-full">Verify Full</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </>
              )}

              {/* MongoDB-specific fields */}
              {connectionForm.watch("connectorType") === "mongodb" && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Auth Source</Label>
                    <Input
                      placeholder="admin"
                      {...connectionForm.register("authSource")}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Replica Set (optional)</Label>
                    <Input
                      placeholder="rs0"
                      {...connectionForm.register("replicaSet")}
                    />
                  </div>
                </div>
              )}
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsCreateDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={createMutation.isPending}
              >
                {createMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  "Create Connection"
                )}
              </Button>
            </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Connections Sidebar */}
        <div className="lg:col-span-1 space-y-4">
          <h2 className="text-lg font-semibold">Connections</h2>
          {isLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <Card key={i} className="animate-pulse">
                  <CardContent className="p-4">
                    <div className="h-5 bg-muted rounded w-3/4" />
                    <div className="h-4 bg-muted rounded w-1/2 mt-2" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : connections.length === 0 ? (
            <Card className="py-8">
              <CardContent className="text-center">
                <Database className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground">
                  No database connections yet
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-2">
              {connections.map((conn: DatabaseConnection) => (
                <Card
                  key={conn.id}
                  className={`cursor-pointer transition-colors hover:border-primary ${
                    selectedConnection?.id === conn.id ? "border-primary" : ""
                  }`}
                  onClick={() => setSelectedConnection(conn)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">
                          {databaseIcons[conn.connector_type] || "🗄️"}
                        </span>
                        <div>
                          <p className="font-medium">{conn.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {conn.database_name}
                          </p>
                        </div>
                      </div>
                      {getStatusBadge(conn)}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Main Query Area */}
        <div className="lg:col-span-3 space-y-4">
          {!selectedConnection ? (
            <Card className="py-12">
              <CardContent className="text-center">
                <Database className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">
                  Select a Database Connection
                </h3>
                <p className="text-muted-foreground mb-4">
                  Choose a connection from the sidebar or create a new one to
                  start querying
                </p>
                <Button onClick={() => setIsCreateDialogOpen(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Database
                </Button>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Connection Info Bar */}
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-3xl">
                        {databaseIcons[selectedConnection.connector_type] || "🗄️"}
                      </span>
                      <div>
                        <h3 className="font-semibold">{selectedConnection.name}</h3>
                        <p className="text-sm text-muted-foreground">
                          {selectedConnection.host}:{selectedConnection.port}/
                          {selectedConnection.database_name}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => testMutation.mutate(selectedConnection.id)}
                        disabled={testMutation.isPending}
                      >
                        {testMutation.isPending ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <RefreshCw className="h-4 w-4" />
                        )}
                        <span className="ml-1">Test</span>
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={async () => {
                          setIsSchemaEditorOpen(true);
                          // Fetch annotations and schema independently
                          try {
                            const { data: annData } = await api.get<SchemaAnnotations>(`/database/connections/${selectedConnection.id}/annotations`);
                            const ann = annData as SchemaAnnotations | null;
                            setAnnotations({ tables: ann?.tables || {}, glossary: ann?.glossary || {} });
                          } catch (e) {
                            console.error("Failed to fetch annotations:", e);
                            setAnnotations({ tables: {}, glossary: {} });
                          }
                          try {
                            const { data: schData } = await api.get<{ tables: SchemaTable[] }>(`/database/connections/${selectedConnection.id}/schema`);
                            const sch = schData as { tables?: SchemaTable[] } | null;
                            setSchemaData(sch?.tables || []);
                          } catch (e) {
                            console.error("Failed to fetch schema:", e);
                            setSchemaData([]);
                          }
                        }}
                      >
                        <BookOpen className="h-4 w-4" />
                        <span className="ml-1">Teach Schema</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-destructive"
                        onClick={() => {
                          if (confirm("Delete this connection?")) {
                            deleteMutation.mutate(selectedConnection.id);
                          }
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Query Input */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Ask a Question</CardTitle>
                  <CardDescription>
                    Type your question in natural language and we'll generate
                    the SQL
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-2">
                    <Textarea
                      placeholder="e.g., Show me the top 10 customers by total orders"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      className="min-h-[80px]"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                          handleQuery();
                        }
                      }}
                    />
                    <Button
                      className="self-end"
                      onClick={handleQuery}
                      disabled={isQuerying || !query.trim()}
                    >
                      {isQuerying ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                      <span className="ml-1">Run</span>
                    </Button>
                  </div>
                  <div className="flex items-center gap-3 mt-2">
                    <div className="flex items-center gap-2">
                      <Label className="text-xs text-muted-foreground whitespace-nowrap">Model:</Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger className="h-7 w-[260px] text-xs">
                          <SelectValue placeholder="Default" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__default__">Default</SelectItem>
                          {availableModels.map((m: { name: string; provider: string }) => (
                            <SelectItem key={`${m.provider}-${m.name}`} value={m.name}>
                              <span className="flex items-center gap-1.5">
                                <span>{m.name}</span>
                                <span className="text-muted-foreground">({m.provider})</span>
                              </span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" size="sm" className="h-7 text-xs gap-1">
                          <SlidersHorizontal className="h-3 w-3" />
                          Advanced
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-72 p-4" align="start">
                        <div className="space-y-4">
                          <h4 className="font-medium text-sm">Query Options</h4>

                          <div className="flex items-center justify-between">
                            <div>
                              <Label className="text-xs">Schema Linking</Label>
                              <p className="text-[10px] text-muted-foreground">Prune to relevant tables</p>
                            </div>
                            <Switch checked={schemaLinking} onCheckedChange={setSchemaLinking} />
                          </div>

                          <div className="flex items-center justify-between">
                            <div>
                              <Label className="text-xs">Sample Rows</Label>
                              <p className="text-[10px] text-muted-foreground">Include data samples in prompt</p>
                            </div>
                            <Switch checked={sampleRows} onCheckedChange={setSampleRows} />
                          </div>

                          <div className="flex items-center justify-between">
                            <div>
                              <Label className="text-xs">Self-Consistency</Label>
                              <p className="text-[10px] text-muted-foreground">Multiple candidates (slower)</p>
                            </div>
                            <Switch checked={selfConsistency} onCheckedChange={setSelfConsistency} />
                          </div>

                          <div className="flex items-center justify-between">
                            <div>
                              <Label className="text-xs">Smart Examples</Label>
                              <p className="text-[10px] text-muted-foreground">Semantic example selection</p>
                            </div>
                            <Switch checked={dynamicExamples} onCheckedChange={setDynamicExamples} />
                          </div>

                          <div className="flex items-center justify-between">
                            <div>
                              <Label className="text-xs">Complexity Routing</Label>
                              <p className="text-[10px] text-muted-foreground">Adaptive prompt templates</p>
                            </div>
                            <Switch checked={complexityRouting} onCheckedChange={setComplexityRouting} />
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <Label className="text-xs">Examples</Label>
                              <span className="text-xs text-muted-foreground">{numExamples}</span>
                            </div>
                            <Slider
                              value={[numExamples]}
                              onValueChange={([v]) => setNumExamples(v)}
                              min={0}
                              max={5}
                              step={1}
                              className="w-full"
                            />
                          </div>
                        </div>
                      </PopoverContent>
                    </Popover>

                    <p className="text-xs text-muted-foreground">
                      Press ⌘+Enter or Ctrl+Enter to run
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Results */}
              {queryResult && (
                <Card>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Results</CardTitle>
                      {queryResult.success && queryResult.query_id && (
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">
                            Helpful?
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() =>
                              feedbackMutation.mutate({
                                queryId: queryResult.query_id!,
                                rating: 5,
                                isCorrect: true,
                              })
                            }
                          >
                            <ThumbsUp className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() =>
                              feedbackMutation.mutate({
                                queryId: queryResult.query_id!,
                                rating: 1,
                                isCorrect: false,
                              })
                            }
                          >
                            <ThumbsDown className="h-4 w-4" />
                          </Button>
                        </div>
                      )}
                    </div>
                    {/* Feature metadata badges */}
                    {queryResult.success && (queryResult.complexity || (queryResult.features_used && queryResult.features_used.length > 0)) && (
                      <div className="flex flex-wrap gap-1.5 mt-1">
                        {queryResult.complexity && (
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            {queryResult.complexity}
                          </Badge>
                        )}
                        {queryResult.schema_tables_used && queryResult.schema_tables_used.length > 0 && (
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            {queryResult.schema_tables_used.length} tables
                          </Badge>
                        )}
                        {queryResult.candidates_generated && queryResult.candidates_generated > 1 && (
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            {queryResult.candidates_generated} candidates
                          </Badge>
                        )}
                        {queryResult.features_used?.map((f) => (
                          <Badge key={f} variant="secondary" className="text-[10px] px-1.5 py-0">
                            {f.replace(/_/g, " ")}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </CardHeader>
                  <CardContent>
                    <Tabs value={activeTab} onValueChange={setActiveTab}>
                      <TabsList>
                        <TabsTrigger value="results">
                          <Table className="h-4 w-4 mr-1" />
                          Results ({queryResult.row_count})
                        </TabsTrigger>
                        <TabsTrigger value="sql">
                          <Code className="h-4 w-4 mr-1" />
                          SQL
                        </TabsTrigger>
                        <TabsTrigger value="explanation">
                          <MessageSquare className="h-4 w-4 mr-1" />
                          Explanation
                        </TabsTrigger>
                      </TabsList>

                      <TabsContent value="results" className="mt-4">
                        {queryResult.error ? (
                          <div className="p-4 bg-destructive/10 text-destructive rounded-lg flex items-start gap-2">
                            <AlertCircle className="h-5 w-5 mt-0.5" />
                            <span>{queryResult.error}</span>
                          </div>
                        ) : (
                          <div className="space-y-3">
                            {/* Human-readable answer */}
                            {queryResult.answer && (
                              <div className="bg-primary/5 border border-primary/20 p-4 rounded-lg">
                                <div className="flex items-center gap-2 mb-2">
                                  <Sparkles className="h-4 w-4 text-primary" />
                                  <span className="font-medium text-sm text-primary">Answer</span>
                                </div>
                                <p className="text-sm whitespace-pre-wrap">{queryResult.answer}</p>
                              </div>
                            )}

                            {/* Raw data table - collapsible when answer is present */}
                            {queryResult.rows.length === 0 ? (
                              <div className="text-center py-8 text-muted-foreground">
                                No results found
                              </div>
                            ) : (
                              <div>
                                <button
                                  className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-2"
                                  onClick={() => setIsRawDataOpen(!isRawDataOpen)}
                                >
                                  {isRawDataOpen ? (
                                    <ChevronDown className="h-4 w-4" />
                                  ) : (
                                    <ChevronRight className="h-4 w-4" />
                                  )}
                                  Raw Data ({queryResult.row_count} rows)
                                </button>
                                {(isRawDataOpen || !queryResult.answer) && (
                                  <div className="border rounded-lg overflow-auto max-h-[400px]">
                                    <UITable>
                                      <TableHeader>
                                        <TableRow>
                                          {queryResult.columns.map((col) => (
                                            <TableHead key={col}>{col}</TableHead>
                                          ))}
                                        </TableRow>
                                      </TableHeader>
                                      <TableBody>
                                        {queryResult.rows.map((row, i) => (
                                          <TableRow key={i}>
                                            {row.map((cell, j) => (
                                              <TableCell key={j}>
                                                {cell === null ? (
                                                  <span className="text-muted-foreground italic">
                                                    null
                                                  </span>
                                                ) : typeof cell === "object" ? (
                                                  JSON.stringify(cell)
                                                ) : (
                                                  String(cell)
                                                )}
                                              </TableCell>
                                            ))}
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </UITable>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                        {queryResult.success && (
                          <div className="mt-2 text-sm text-muted-foreground flex items-center gap-4">
                            <span>{queryResult.row_count} rows</span>
                            <span>{queryResult.execution_time_ms.toFixed(0)}ms</span>
                            <span>
                              Confidence: {(queryResult.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                      </TabsContent>

                      <TabsContent value="sql" className="mt-4">
                        {queryResult.generated_sql ? (
                          <div className="relative">
                            <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                              <code>{queryResult.generated_sql}</code>
                            </pre>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="absolute top-2 right-2"
                              onClick={() =>
                                copyToClipboard(queryResult.generated_sql || "")
                              }
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                          </div>
                        ) : (
                          <p className="text-muted-foreground">
                            No SQL generated
                          </p>
                        )}
                      </TabsContent>

                      <TabsContent value="explanation" className="mt-4">
                        {queryResult.explanation ? (
                          <div className="bg-muted p-4 rounded-lg">
                            <p>{queryResult.explanation}</p>
                          </div>
                        ) : (
                          <p className="text-muted-foreground">
                            No explanation available
                          </p>
                        )}
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              )}

              {/* Query History */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <History className="h-5 w-5" />
                    Recent Queries
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {queryHistory.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No query history yet
                    </p>
                  ) : (
                    <div className="space-y-2">
                      {queryHistory.slice(0, 5).map((item: QueryHistoryItem) => (
                        <div
                          key={item.id}
                          className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 cursor-pointer"
                          onClick={() => {
                            setQuery(item.natural_language_query);
                          }}
                        >
                          <div className="flex items-center gap-3">
                            {item.execution_success ? (
                              <CheckCircle className="h-4 w-4 text-green-500" />
                            ) : (
                              <AlertCircle className="h-4 w-4 text-destructive" />
                            )}
                            <div>
                              <p className="text-sm font-medium">
                                {item.natural_language_query}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {item.row_count} rows •{" "}
                                {item.execution_time_ms.toFixed(0)}ms •{" "}
                                {new Date(item.created_at).toLocaleString()}
                              </p>
                            </div>
                          </div>
                          {item.user_rating && (
                            <Badge
                              variant={
                                item.user_rating >= 4 ? "default" : "secondary"
                              }
                            >
                              {item.user_rating >= 4 ? "👍" : "👎"}
                            </Badge>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Schema Annotation Editor Dialog */}
      <Dialog open={isSchemaEditorOpen} onOpenChange={setIsSchemaEditorOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              Teach Schema
            </DialogTitle>
            <DialogDescription>
              Help the AI understand your database by adding descriptions and a business glossary.
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 overflow-y-auto pr-4">
            <div className="space-y-6">
              {/* Glossary Section */}
              <div>
                <h4 className="font-medium text-sm mb-2">Business Glossary</h4>
                <p className="text-xs text-muted-foreground mb-3">
                  Define business terms so the AI maps them to the right columns.
                </p>
                <div className="space-y-2">
                  {Object.entries(annotations.glossary).map(([term, def]) => (
                    <div key={term} className="flex items-center gap-2">
                      <Input
                        className="w-1/3"
                        value={term}
                        readOnly
                      />
                      <Input
                        className="flex-1"
                        value={def}
                        onChange={(e) => {
                          const updated = { ...annotations.glossary, [term]: e.target.value };
                          setAnnotations({ ...annotations, glossary: updated });
                        }}
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const updated = { ...annotations.glossary };
                          delete updated[term];
                          setAnnotations({ ...annotations, glossary: updated });
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  ))}
                  <div className="flex items-center gap-2">
                    <Input
                      className="w-1/3"
                      placeholder="Term (e.g. MRR)"
                      value={glossaryNewTerm}
                      onChange={(e) => setGlossaryNewTerm(e.target.value)}
                    />
                    <Input
                      className="flex-1"
                      placeholder="Definition (e.g. Monthly Recurring Revenue)"
                      value={glossaryNewDef}
                      onChange={(e) => setGlossaryNewDef(e.target.value)}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!glossaryNewTerm.trim() || !glossaryNewDef.trim()}
                      onClick={() => {
                        setAnnotations({
                          ...annotations,
                          glossary: { ...annotations.glossary, [glossaryNewTerm.trim()]: glossaryNewDef.trim() },
                        });
                        setGlossaryNewTerm("");
                        setGlossaryNewDef("");
                      }}
                    >
                      <Plus className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              </div>

              {/* Tables Section */}
              <div>
                <h4 className="font-medium text-sm mb-2">Table & Column Descriptions</h4>
                <p className="text-xs text-muted-foreground mb-3">
                  Describe what tables and columns mean in your business context.
                </p>
                {schemaData.length === 0 ? (
                  <p className="text-sm text-muted-foreground italic">
                    No schema loaded. Test the connection first.
                  </p>
                ) : (
                  <div className="space-y-1">
                    {schemaData.map((table) => (
                      <div key={table.name} className="border rounded-lg">
                        <button
                          className="w-full flex items-center gap-2 p-3 text-sm font-medium hover:bg-muted/50"
                          onClick={() => {
                            const next = new Set(expandedTables);
                            if (next.has(table.name)) next.delete(table.name);
                            else next.add(table.name);
                            setExpandedTables(next);
                          }}
                        >
                          {expandedTables.has(table.name) ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                          {table.name}
                          <Badge variant="secondary" className="ml-auto text-xs">
                            {table.columns.length} cols
                          </Badge>
                        </button>
                        {expandedTables.has(table.name) && (
                          <div className="px-3 pb-3 space-y-2">
                            <div>
                              <Label className="text-xs">Table Description</Label>
                              <Input
                                placeholder="What does this table store?"
                                value={annotations.tables[table.name]?.description || ""}
                                onChange={(e) => {
                                  const tableAnn = annotations.tables[table.name] || {};
                                  setAnnotations({
                                    ...annotations,
                                    tables: {
                                      ...annotations.tables,
                                      [table.name]: { ...tableAnn, description: e.target.value },
                                    },
                                  });
                                }}
                              />
                            </div>
                            <div className="space-y-1">
                              {table.columns.map((col) => (
                                <div key={col.name} className="flex items-center gap-2">
                                  <span className="text-xs font-mono w-1/3 truncate" title={col.name}>
                                    {col.name}
                                    <span className="text-muted-foreground ml-1">({col.data_type})</span>
                                  </span>
                                  <Input
                                    className="flex-1 h-8 text-xs"
                                    placeholder={col.description || "Describe this column..."}
                                    value={annotations.tables[table.name]?.columns?.[col.name] || ""}
                                    onChange={(e) => {
                                      const tableAnn = annotations.tables[table.name] || {};
                                      const cols = tableAnn.columns || {};
                                      setAnnotations({
                                        ...annotations,
                                        tables: {
                                          ...annotations.tables,
                                          [table.name]: {
                                            ...tableAnn,
                                            columns: { ...cols, [col.name]: e.target.value },
                                          },
                                        },
                                      });
                                    }}
                                  />
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsSchemaEditorOpen(false)}>
              Cancel
            </Button>
            <Button
              disabled={isSavingAnnotations}
              onClick={async () => {
                if (!selectedConnection) return;
                setIsSavingAnnotations(true);
                try {
                  // Clean empty values before saving
                  const cleanTables: Record<string, any> = {};
                  for (const [tName, tAnn] of Object.entries(annotations.tables)) {
                    const cleanCols: Record<string, string> = {};
                    for (const [cName, cDesc] of Object.entries(tAnn.columns || {})) {
                      if (cDesc.trim()) cleanCols[cName] = cDesc.trim();
                    }
                    if (tAnn.description?.trim() || Object.keys(cleanCols).length > 0) {
                      cleanTables[tName] = {
                        ...(tAnn.description?.trim() ? { description: tAnn.description.trim() } : {}),
                        ...(Object.keys(cleanCols).length > 0 ? { columns: cleanCols } : {}),
                      };
                    }
                  }
                  const cleanGlossary: Record<string, string> = {};
                  for (const [term, def] of Object.entries(annotations.glossary)) {
                    if (term.trim() && def.trim()) cleanGlossary[term.trim()] = def.trim();
                  }

                  await api.put(`/database/connections/${selectedConnection.id}/annotations`, {
                    tables: cleanTables,
                    glossary: cleanGlossary,
                  });
                  toast.success("Schema annotations saved");
                  setIsSchemaEditorOpen(false);
                } catch (error) {
                  toast.error(error instanceof Error ? error.message : "Failed to save annotations");
                } finally {
                  setIsSavingAnnotations(false);
                }
              }}
            >
              {isSavingAnnotations ? (
                <Loader2 className="h-4 w-4 animate-spin mr-1" />
              ) : (
                <Save className="h-4 w-4 mr-1" />
              )}
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
