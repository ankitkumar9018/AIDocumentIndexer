"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  useConnectorTypes,
  useConnectors,
  useCreateConnector,
  useUpdateConnector,
  useDeleteConnector,
  useTriggerConnectorSync,
  type ConnectorType,
  type ConnectorInstance,
} from "@/lib/api";
import {
  Plus,
  RefreshCw,
  Settings,
  Trash2,
  ExternalLink,
  Cloud,
  CheckCircle,
  AlertCircle,
  Clock,
  FolderSync,
  Link2,
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
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { api } from "@/lib/api/client";

// Connector type icons
const connectorIcons: Record<string, React.ElementType> = {
  google_drive: Cloud,
  notion: FolderSync,
  confluence: Link2,
  onedrive: Cloud,
  slack: Cloud,
  youtube: Cloud,
};

export default function ConnectorsPage() {
  const queryClient = useQueryClient();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [selectedConnector, setSelectedConnector] = useState<ConnectorInstance | null>(null);
  const [newConnectorType, setNewConnectorType] = useState<string>("");
  const [newConnectorName, setNewConnectorName] = useState<string>("");
  const [newStorageMode, setNewStorageMode] = useState<string>("global_default");

  // Fetch connector types and instances using new hooks
  const { data: connectorTypes = [], isError: typesError } = useConnectorTypes();
  const { data: connectors = [], isLoading, isError, refetch } = useConnectors();

  // Mutations using new hooks
  const createConnectorMutation = useCreateConnector();
  const deleteConnectorMutation = useDeleteConnector();
  const triggerSyncMutation = useTriggerConnectorSync();
  const updateConnectorMutation = useUpdateConnector();

  const handleCreateConnector = async () => {
    try {
      const data = await createConnectorMutation.mutateAsync({
        connector_type: newConnectorType,
        name: newConnectorName,
        sync_config: newStorageMode !== "global_default" ? { storage_mode: newStorageMode } : undefined,
      });
      setIsCreateDialogOpen(false);
      setNewConnectorType("");
      setNewConnectorName("");
      setNewStorageMode("global_default");
      toast.success("Connector created");

      // If it's OAuth-based, redirect to OAuth flow
      if (data.connector_type === "google_drive") {
        initiateOAuth(data.id);
      }
    } catch (error) {
      toast.error(`Failed to create connector: ${(error as Error).message}`);
    }
  };

  const handleDeleteConnector = async (id: string) => {
    try {
      await deleteConnectorMutation.mutateAsync(id);
      toast.success("Connector deleted");
    } catch (error) {
      toast.error(`Failed to delete connector: ${(error as Error).message}`);
    }
  };

  const handleTriggerSync = async (id: string) => {
    try {
      await triggerSyncMutation.mutateAsync(id);
      toast.success("Sync started");
    } catch (error) {
      toast.error(`Failed to start sync: ${(error as Error).message}`);
    }
  };

  const handleToggleActive = async (id: string, is_active: boolean) => {
    try {
      await updateConnectorMutation.mutateAsync({ id, data: { is_active } });
    } catch (error) {
      toast.error(`Failed to update connector: ${(error as Error).message}`);
    }
  };

  const initiateOAuth = async (connectorId: string) => {
    try {
      const response = await api.getConnectorOAuthUrl(connectorId);
      window.location.href = response.url;
    } catch (error) {
      toast.error("Failed to initiate OAuth");
    }
  };

  const getStatusBadge = (status: string, isActive: boolean) => {
    if (!isActive) {
      return <Badge variant="secondary">Inactive</Badge>;
    }
    switch (status) {
      case "connected":
      case "active":
        return <Badge className="bg-green-500">Connected</Badge>;
      case "syncing":
        return <Badge className="bg-blue-500">Syncing</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      case "pending":
        return <Badge variant="outline">Pending Setup</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "Never";
    return new Date(dateString).toLocaleString();
  };

  const getConnectorIcon = (type: string) => {
    const Icon = connectorIcons[type] || Cloud;
    return <Icon className="h-8 w-8" />;
  };

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Connectors</h1>
          <p className="text-muted-foreground">
            Connect external data sources to automatically sync documents
          </p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Add Connector
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Connector</DialogTitle>
              <DialogDescription>
                Connect an external data source to sync documents automatically
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Connector Type</Label>
                <Select value={newConnectorType} onValueChange={setNewConnectorType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a connector type" />
                  </SelectTrigger>
                  <SelectContent>
                    {connectorTypes.map((type) => (
                      <SelectItem key={type.type} value={type.type}>
                        <div className="flex items-center gap-2">
                          {getConnectorIcon(type.type)}
                          <span>{type.display_name}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Name</Label>
                <Input
                  placeholder="My Google Drive"
                  value={newConnectorName}
                  onChange={(e) => setNewConnectorName(e.target.value)}
                />
              </div>
              {/* Storage Mode Override */}
              <div className="space-y-2">
                <Label>Storage Mode</Label>
                <Select value={newStorageMode} onValueChange={setNewStorageMode}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select storage mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="global_default">Use Global Default</SelectItem>
                    <SelectItem value="download">Download & Store</SelectItem>
                    <SelectItem value="process_only">Process Only (link only)</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Override the global storage setting for this connector
                </p>
              </div>
              {newConnectorType && (
                <div className="p-4 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground">
                    {connectorTypes.find((t) => t.type === newConnectorType)?.description}
                  </p>
                </div>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleCreateConnector}
                disabled={!newConnectorType || !newConnectorName || createConnectorMutation.isPending}
              >
                {createConnectorMutation.isPending ? "Creating..." : "Create & Connect"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Connector Cards */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-6 bg-muted rounded w-1/2" />
                <div className="h-4 bg-muted rounded w-3/4 mt-2" />
              </CardHeader>
            </Card>
          ))}
        </div>
      ) : isError || typesError ? (
        <Card className="py-12">
          <CardContent className="text-center">
            <AlertCircle className="mx-auto h-12 w-12 text-destructive mb-4" />
            <h3 className="text-lg font-medium mb-2">Failed to Load Connectors</h3>
            <p className="text-muted-foreground mb-4">
              There was an error loading connectors. Please try again.
            </p>
            <Button variant="outline" onClick={() => refetch()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : connectors.length === 0 ? (
        <Card className="py-12">
          <CardContent className="text-center">
            <Cloud className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No Connectors</h3>
            <p className="text-muted-foreground mb-4">
              Add a connector to start syncing documents from external sources
            </p>
            <Button onClick={() => setIsCreateDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Your First Connector
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {connectors.map((connector) => (
            <Card key={connector.id} className="relative">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-muted rounded-lg">
                      {getConnectorIcon(connector.connector_type)}
                    </div>
                    <div>
                      <CardTitle className="text-lg">{connector.name}</CardTitle>
                      <CardDescription className="capitalize">
                        {connector.connector_type.replace("_", " ")}
                      </CardDescription>
                    </div>
                  </div>
                  {getStatusBadge(connector.status, connector.is_active)}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Sync Info */}
                <div className="text-sm space-y-1">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span>Last sync: {formatDate(connector.last_sync_at)}</span>
                  </div>
                  {connector.next_sync_at && (
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <RefreshCw className="h-4 w-4" />
                      <span>Next sync: {formatDate(connector.next_sync_at)}</span>
                    </div>
                  )}
                </div>

                {/* Error Message */}
                {connector.error_message && (
                  <div className="flex items-start gap-2 p-2 bg-destructive/10 text-destructive rounded text-sm">
                    <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                    <span>{connector.error_message}</span>
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center justify-between pt-2 border-t">
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={connector.is_active}
                      onCheckedChange={(checked) =>
                        handleToggleActive(connector.id, checked)
                      }
                    />
                    <span className="text-sm text-muted-foreground">
                      {connector.is_active ? "Active" : "Paused"}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {connector.status === "pending" ? (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => initiateOAuth(connector.id)}
                      >
                        <ExternalLink className="h-4 w-4 mr-1" />
                        Connect
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleTriggerSync(connector.id)}
                        disabled={!connector.is_active || triggerSyncMutation.isPending}
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSelectedConnector(connector)}
                    >
                      <Settings className="h-4 w-4" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive"
                      onClick={() => {
                        if (confirm("Are you sure you want to delete this connector?")) {
                          handleDeleteConnector(connector.id);
                        }
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Available Connector Types */}
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">Available Connectors</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {connectorTypes.map((type) => (
            <Card
              key={type.type}
              className="cursor-pointer hover:border-primary transition-colors"
              onClick={() => {
                setNewConnectorType(type.type);
                setIsCreateDialogOpen(true);
              }}
            >
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-muted rounded-lg">
                    {getConnectorIcon(type.type)}
                  </div>
                  <div>
                    <h3 className="font-medium">{type.display_name}</h3>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {type.description}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Settings Dialog */}
      {selectedConnector && (
        <Dialog open={!!selectedConnector} onOpenChange={() => setSelectedConnector(null)}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Connector Settings</DialogTitle>
              <DialogDescription>
                Configure sync settings for {selectedConnector.name}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Sync Schedule</Label>
                <Select defaultValue="hourly">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="every5min">Every 5 minutes</SelectItem>
                    <SelectItem value="every15min">Every 15 minutes</SelectItem>
                    <SelectItem value="hourly">Every hour</SelectItem>
                    <SelectItem value="daily">Daily</SelectItem>
                    <SelectItem value="manual">Manual only</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>File Types to Sync</Label>
                <div className="flex flex-wrap gap-2">
                  {["PDF", "DOCX", "XLSX", "PPTX", "TXT", "MD"].map((type) => (
                    <Badge
                      key={type}
                      variant="outline"
                      className="cursor-pointer hover:bg-primary hover:text-primary-foreground"
                    >
                      {type}
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <Label>Sync History</Label>
                <div className="border rounded-lg divide-y max-h-48 overflow-y-auto">
                  <div className="p-3 flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Completed</span>
                    </div>
                    <span className="text-muted-foreground">2 hours ago</span>
                    <span>42 files synced</span>
                  </div>
                  <div className="p-3 flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span>Completed</span>
                    </div>
                    <span className="text-muted-foreground">1 day ago</span>
                    <span>156 files synced</span>
                  </div>
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setSelectedConnector(null)}>
                Close
              </Button>
              <Button>Save Settings</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
