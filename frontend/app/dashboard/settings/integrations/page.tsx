"use client";

import { useState, useEffect } from "react";
import {
  Plus,
  Trash2,
  Settings,
  ExternalLink,
  CheckCircle,
  Loader2,
  RefreshCw,
  MessageSquare,
  Hash,
  Users,
  Bot,
  Slack,
  MessageCircle,
  Webhook,
  Copy,
  Eye,
  EyeOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  useBotConnections,
  useCreateBotConnection,
  useDeleteBotConnection,
  useSlackOAuthUrl,
  type BotConnection,
  type BotPlatform,
} from "@/lib/api";
import { toast } from "sonner";
import { formatDistanceToNow } from "date-fns";

// Platform icons and metadata
const platformConfig: Record<string, {
  icon: typeof Slack;
  name: string;
  description: string;
  color: string;
  features: string[];
}> = {
  slack: {
    icon: Slack,
    name: "Slack",
    description: "Connect your Slack workspace to query documents and run agents from Slack channels",
    color: "bg-[#4A154B]",
    features: [
      "/ask - Query your documents",
      "/search - Search across all documents",
      "/summarize - Summarize a thread",
      "/agent - Run an agent task",
      "@mention the bot for quick questions",
    ],
  },
  teams: {
    icon: Users,
    name: "Microsoft Teams",
    description: "Connect Microsoft Teams to access your documents and AI assistant",
    color: "bg-[#6264A7]",
    features: [
      "Chat with the bot in channels",
      "Query documents with @mention",
      "Summarize conversations",
      "Run agent tasks",
    ],
  },
  discord: {
    icon: MessageCircle,
    name: "Discord",
    description: "Add a bot to your Discord server for document queries",
    color: "bg-[#5865F2]",
    features: [
      "!ask command for questions",
      "!search for document search",
      "Thread-based conversations",
    ],
  },
  webhook: {
    icon: Webhook,
    name: "Webhook",
    description: "Generic webhook integration for custom applications",
    color: "bg-gray-600",
    features: [
      "Send queries via HTTP POST",
      "Receive responses as JSON",
      "Custom authentication",
    ],
  },
};

export default function IntegrationsSettingsPage() {
  const [selectedPlatform, setSelectedPlatform] = useState<string | null>(null);
  const [manualSetupOpen, setManualSetupOpen] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [selectedConnection, setSelectedConnection] = useState<BotConnection | null>(null);
  const [showToken, setShowToken] = useState(false);

  // SSR-safe origin
  const [origin, setOrigin] = useState("");
  useEffect(() => {
    if (typeof window !== "undefined") {
      setOrigin(window.location.origin);
    }
  }, []);

  // Manual setup form state
  const [manualPlatform, setManualPlatform] = useState<BotPlatform>("slack");
  const [workspaceId, setWorkspaceId] = useState("");
  const [botToken, setBotToken] = useState("");

  // Fetch bot connections using proper API hooks
  const {
    data: connectionsData,
    isLoading,
    refetch,
  } = useBotConnections();

  // Get Slack install URL
  const slackOAuthMutation = useSlackOAuthUrl();

  // Create connection manually
  const createConnectionMutation = useCreateBotConnection();

  // Delete connection
  const deleteConnectionMutation = useDeleteBotConnection();

  const handleOAuthInstall = (platform: string) => {
    if (platform === "slack") {
      slackOAuthMutation.mutate(undefined, {
        onSuccess: (data) => {
          window.location.href = data.url;
        },
        onError: (error: Error) => {
          toast.error(`Failed to start Slack installation: ${error.message}`);
        },
      });
    } else {
      toast.info(`${platformConfig[platform]?.name || platform} OAuth not yet implemented`);
    }
  };

  const handleManualSetup = () => {
    if (!workspaceId || !botToken) {
      toast.error("Please fill in all required fields");
      return;
    }
    createConnectionMutation.mutate(
      {
        platform: manualPlatform,
        workspace_id: workspaceId,
        bot_token: botToken,
      },
      {
        onSuccess: () => {
          setManualSetupOpen(false);
          setWorkspaceId("");
          setBotToken("");
          toast.success("Bot connection created successfully");
        },
        onError: (error: Error) => {
          toast.error(error.message);
        },
      }
    );
  };

  const handleDeleteConnection = (id: string) => {
    deleteConnectionMutation.mutate(id, {
      onSuccess: () => {
        setDeleteConfirmId(null);
        toast.success("Bot connection deleted");
      },
      onError: (error: Error) => {
        toast.error(`Failed to delete connection: ${error.message}`);
      },
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const connections = connectionsData?.connections || [];

  return (
    <div className="container mx-auto py-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Bot Integrations</h1>
          <p className="text-muted-foreground">
            Connect chat platforms to access your AI assistant and documents
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button onClick={() => setManualSetupOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Manual Setup
          </Button>
        </div>
      </div>

      {/* Active Connections */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Active Connections</h2>

        {isLoading ? (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2].map((i) => (
              <Card key={i}>
                <CardHeader>
                  <Skeleton className="h-6 w-32" />
                  <Skeleton className="h-4 w-48 mt-2" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-4 w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : connections.length === 0 ? (
          <Card className="py-12">
            <CardContent className="text-center">
              <Bot className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No Bot Connections</h3>
              <p className="text-muted-foreground mb-4">
                Connect a chat platform to start using your AI assistant
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {connections.map((connection) => {
              const platform = platformConfig[connection.platform] || platformConfig.webhook;
              const PlatformIcon = platform.icon;

              return (
                <Card key={connection.id} className="relative">
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-lg ${platform.color} text-white`}>
                          <PlatformIcon className="h-5 w-5" />
                        </div>
                        <div>
                          <CardTitle className="text-lg">
                            {connection.workspace_name || connection.workspace_id}
                          </CardTitle>
                          <CardDescription className="capitalize">
                            {platform.name}
                          </CardDescription>
                        </div>
                      </div>
                      <Badge className={connection.is_active ? "bg-green-500" : "bg-gray-500"}>
                        {connection.is_active ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="text-sm text-muted-foreground space-y-1">
                      <div className="flex items-center gap-2">
                        <Hash className="h-4 w-4" />
                        <span>ID: {connection.workspace_id}</span>
                      </div>
                      {connection.last_used_at && (
                        <div className="flex items-center gap-2">
                          <MessageSquare className="h-4 w-4" />
                          <span>
                            Last used: {formatDistanceToNow(new Date(connection.last_used_at), { addSuffix: true })}
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between pt-2 border-t">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setSelectedConnection(connection)}
                      >
                        <Settings className="h-4 w-4 mr-1" />
                        Settings
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive"
                        onClick={() => setDeleteConfirmId(connection.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>

      {/* Available Platforms */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Available Platforms</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {Object.entries(platformConfig).map(([key, platform]) => {
            const PlatformIcon = platform.icon;
            const isConnected = connections.some((c) => c.platform === key);

            return (
              <Card key={key} className="overflow-hidden">
                <div className="flex">
                  <div className={`w-2 ${platform.color}`} />
                  <div className="flex-1">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${platform.color} text-white`}>
                            <PlatformIcon className="h-6 w-6" />
                          </div>
                          <div>
                            <CardTitle>{platform.name}</CardTitle>
                            <CardDescription className="mt-1">
                              {platform.description}
                            </CardDescription>
                          </div>
                        </div>
                        {isConnected && (
                          <Badge variant="outline" className="text-green-600 border-green-600">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Connected
                          </Badge>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <p className="text-sm font-medium">Features:</p>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          {platform.features.map((feature, i) => (
                            <li key={i} className="flex items-center gap-2">
                              <CheckCircle className="h-3 w-3 text-green-500" />
                              {feature}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          onClick={() => handleOAuthInstall(key)}
                          disabled={slackOAuthMutation.isPending && key === "slack"}
                          className={platform.color}
                        >
                          {slackOAuthMutation.isPending && key === "slack" ? (
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          ) : (
                            <ExternalLink className="h-4 w-4 mr-2" />
                          )}
                          {isConnected ? "Add Another Workspace" : `Connect ${platform.name}`}
                        </Button>
                      </div>
                    </CardContent>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Manual Setup Dialog */}
      <Dialog open={manualSetupOpen} onOpenChange={setManualSetupOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Manual Bot Setup</DialogTitle>
            <DialogDescription>
              Configure a bot connection manually with your own tokens
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Platform</Label>
              <Select value={manualPlatform} onValueChange={(val) => setManualPlatform(val as BotPlatform)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(platformConfig).map(([key, platform]) => (
                    <SelectItem key={key} value={key}>
                      {platform.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Workspace ID</Label>
              <Input
                placeholder="e.g., T0123456789"
                value={workspaceId}
                onChange={(e) => setWorkspaceId(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                The unique identifier for your workspace/team
              </p>
            </div>
            <div className="space-y-2">
              <Label>Bot Token</Label>
              <div className="relative">
                <Input
                  type={showToken ? "text" : "password"}
                  placeholder="xoxb-..."
                  value={botToken}
                  onChange={(e) => setBotToken(e.target.value)}
                  className="pr-10"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-1 top-1 h-7 w-7"
                  onClick={() => setShowToken(!showToken)}
                >
                  {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                The bot OAuth token from your app settings
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setManualSetupOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleManualSetup}
              disabled={createConnectionMutation.isPending}
            >
              {createConnectionMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Create Connection
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Connection Settings Dialog */}
      <Dialog open={!!selectedConnection} onOpenChange={() => setSelectedConnection(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Connection Settings</DialogTitle>
            <DialogDescription>
              {selectedConnection?.workspace_name || selectedConnection?.workspace_id}
            </DialogDescription>
          </DialogHeader>
          {selectedConnection && (
            <Tabs defaultValue="general" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="general">General</TabsTrigger>
                <TabsTrigger value="webhooks">Webhooks</TabsTrigger>
              </TabsList>
              <TabsContent value="general" className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label>Status</Label>
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <span>Bot Active</span>
                    <Switch checked={selectedConnection.is_active} />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Workspace ID</Label>
                  <div className="flex items-center gap-2">
                    <Input value={selectedConnection.workspace_id} readOnly />
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => copyToClipboard(selectedConnection.workspace_id)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Created</Label>
                  <Input
                    value={
                      selectedConnection.created_at
                        ? new Date(selectedConnection.created_at).toLocaleDateString()
                        : "Unknown"
                    }
                    readOnly
                  />
                </div>
              </TabsContent>
              <TabsContent value="webhooks" className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label>Events Webhook URL</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      value={`${origin}/api/v1/bots/slack/events`}
                      readOnly
                      className="text-sm"
                    />
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() =>
                        copyToClipboard(`${origin}/api/v1/bots/slack/events`)
                      }
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Use this URL in your Slack app's Event Subscriptions
                  </p>
                </div>
                <div className="space-y-2">
                  <Label>Commands Webhook URL</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      value={`${origin}/api/v1/bots/slack/commands`}
                      readOnly
                      className="text-sm"
                    />
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() =>
                        copyToClipboard(`${origin}/api/v1/bots/slack/commands`)
                      }
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Use this URL for Slack slash commands
                  </p>
                </div>
              </TabsContent>
            </Tabs>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedConnection(null)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteConfirmId} onOpenChange={() => setDeleteConfirmId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Bot Connection?</AlertDialogTitle>
            <AlertDialogDescription>
              This will disconnect the bot from this workspace. You can reconnect later,
              but any workspace-specific settings will be lost.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => deleteConfirmId && handleDeleteConnection(deleteConfirmId)}
            >
              {deleteConnectionMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
