"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/components/ui/use-toast";
import {
  Key,
  Plus,
  Copy,
  Trash2,
  Eye,
  EyeOff,
  RefreshCw,
  Code2,
  Zap,
  Activity,
  Settings,
  ExternalLink,
  FileJson,
  Book,
  Shield,
  Clock,
} from "lucide-react";

interface APIKey {
  id: string;
  name: string;
  key_prefix: string;
  created_at: string;
  expires_at: string | null;
  rate_limit_per_minute: number;
  is_active: boolean;
  last_used_at: string | null;
  usage_count: number;
}

interface PublishedEndpoint {
  id: string;
  name: string;
  description: string | null;
  type: "skill" | "workflow";
  endpoint_url: string;
  method: string;
  requires_api_key: boolean;
  rate_limit_per_minute: number;
  input_schema: object | null;
  output_schema: object | null;
  example_request: object | null;
}

export default function APIAccessPage() {
  const { toast } = useToast();
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [endpoints, setEndpoints] = useState<PublishedEndpoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [selectedEndpoint, setSelectedEndpoint] = useState<PublishedEndpoint | null>(null);

  // New API key form state
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyDescription, setNewKeyDescription] = useState("");
  const [newKeyExpiry, setNewKeyExpiry] = useState<string>("never");
  const [newKeyRateLimit, setNewKeyRateLimit] = useState(60);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [keysRes, endpointsRes] = await Promise.all([
        fetch("/api/v1/external/api-keys", { credentials: "include" }),
        fetch("/api/v1/external/published", { credentials: "include" }),
      ]);

      if (keysRes.ok) {
        setApiKeys(await keysRes.json());
      }
      if (endpointsRes.ok) {
        setEndpoints(await endpointsRes.json());
      }
    } catch (error) {
      console.error("Failed to fetch API data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const createApiKey = async () => {
    try {
      const res = await fetch("/api/v1/external/api-keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          name: newKeyName,
          description: newKeyDescription,
          expires_in_days: newKeyExpiry === "never" ? null : parseInt(newKeyExpiry),
          rate_limit_per_minute: newKeyRateLimit,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setNewApiKey(data.api_key);
        toast({
          title: "API Key Created",
          description: "Copy your API key now - it won't be shown again!",
        });
        fetchData();
      } else {
        throw new Error("Failed to create API key");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to create API key",
        variant: "destructive",
      });
    }
  };

  const revokeApiKey = async (keyId: string) => {
    try {
      const res = await fetch(`/api/v1/external/api-keys/${keyId}`, {
        method: "DELETE",
        credentials: "include",
      });

      if (res.ok) {
        toast({ title: "API Key Revoked" });
        fetchData();
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to revoke API key",
        variant: "destructive",
      });
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({ title: "Copied to clipboard" });
  };

  const getBaseUrl = () => {
    return typeof window !== "undefined" ? window.location.origin : "";
  };

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">External API Access</h1>
          <p className="text-muted-foreground">
            Manage API keys and publish skills/workflows for external integration
          </p>
        </div>
        <Button onClick={() => setShowCreateDialog(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create API Key
        </Button>
      </div>

      <Tabs defaultValue="keys" className="space-y-4">
        <TabsList>
          <TabsTrigger value="keys">
            <Key className="mr-2 h-4 w-4" />
            API Keys
          </TabsTrigger>
          <TabsTrigger value="endpoints">
            <Zap className="mr-2 h-4 w-4" />
            Published Endpoints
          </TabsTrigger>
          <TabsTrigger value="docs">
            <Book className="mr-2 h-4 w-4" />
            Documentation
          </TabsTrigger>
          <TabsTrigger value="usage">
            <Activity className="mr-2 h-4 w-4" />
            Usage & Monitoring
          </TabsTrigger>
        </TabsList>

        {/* API Keys Tab */}
        <TabsContent value="keys" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Your API Keys</CardTitle>
              <CardDescription>
                Manage API keys for external access to your published skills and workflows
              </CardDescription>
            </CardHeader>
            <CardContent>
              {apiKeys.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Key className="mx-auto h-12 w-12 mb-4 opacity-50" />
                  <p>No API keys yet</p>
                  <p className="text-sm">Create an API key to start using the external API</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Key</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Expires</TableHead>
                      <TableHead>Rate Limit</TableHead>
                      <TableHead>Usage</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {apiKeys.map((key) => (
                      <TableRow key={key.id}>
                        <TableCell className="font-medium">{key.name}</TableCell>
                        <TableCell>
                          <code className="text-sm bg-muted px-2 py-1 rounded">
                            {key.key_prefix}...
                          </code>
                        </TableCell>
                        <TableCell>
                          {new Date(key.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          {key.expires_at
                            ? new Date(key.expires_at).toLocaleDateString()
                            : "Never"}
                        </TableCell>
                        <TableCell>{key.rate_limit_per_minute}/min</TableCell>
                        <TableCell>{key.usage_count} calls</TableCell>
                        <TableCell>
                          <Badge variant={key.is_active ? "default" : "secondary"}>
                            {key.is_active ? "Active" : "Revoked"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                disabled={!key.is_active}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                              <AlertDialogHeader>
                                <AlertDialogTitle>Revoke API Key?</AlertDialogTitle>
                                <AlertDialogDescription>
                                  This will immediately revoke access for any systems using
                                  this API key. This action cannot be undone.
                                </AlertDialogDescription>
                              </AlertDialogHeader>
                              <AlertDialogFooter>
                                <AlertDialogCancel>Cancel</AlertDialogCancel>
                                <AlertDialogAction
                                  onClick={() => revokeApiKey(key.id)}
                                  className="bg-destructive text-destructive-foreground"
                                >
                                  Revoke
                                </AlertDialogAction>
                              </AlertDialogFooter>
                            </AlertDialogContent>
                          </AlertDialog>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Published Endpoints Tab */}
        <TabsContent value="endpoints" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Published Endpoints</CardTitle>
              <CardDescription>
                Skills and workflows available for external API access
              </CardDescription>
            </CardHeader>
            <CardContent>
              {endpoints.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Zap className="mx-auto h-12 w-12 mb-4 opacity-50" />
                  <p>No published endpoints</p>
                  <p className="text-sm">
                    Publish a skill or workflow to make it available via API
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {endpoints.map((endpoint) => (
                    <Card key={endpoint.id} className="border">
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge variant={endpoint.type === "skill" ? "default" : "secondary"}>
                              {endpoint.type}
                            </Badge>
                            <CardTitle className="text-lg">{endpoint.name}</CardTitle>
                          </div>
                          <div className="flex items-center gap-2">
                            {endpoint.requires_api_key && (
                              <Badge variant="outline">
                                <Shield className="mr-1 h-3 w-3" />
                                API Key Required
                              </Badge>
                            )}
                            <Badge variant="outline">
                              <Clock className="mr-1 h-3 w-3" />
                              {endpoint.rate_limit_per_minute}/min
                            </Badge>
                          </div>
                        </div>
                        {endpoint.description && (
                          <CardDescription>{endpoint.description}</CardDescription>
                        )}
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline">{endpoint.method}</Badge>
                            <code className="flex-1 text-sm bg-muted px-3 py-1 rounded font-mono">
                              {getBaseUrl()}{endpoint.endpoint_url}
                            </code>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() =>
                                copyToClipboard(`${getBaseUrl()}${endpoint.endpoint_url}`)
                              }
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => setSelectedEndpoint(endpoint)}
                            >
                              <Code2 className="mr-2 h-4 w-4" />
                              View Integration Code
                            </Button>
                            <Button variant="outline" size="sm" asChild>
                              <a
                                href={`/api/v1/external/openapi/${endpoint.type}/${endpoint.id}`}
                                target="_blank"
                              >
                                <FileJson className="mr-2 h-4 w-4" />
                                OpenAPI Spec
                              </a>
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Documentation Tab */}
        <TabsContent value="docs" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Integration Guide</CardTitle>
              <CardDescription>
                How to integrate your published skills and workflows with external services
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">1. Create an API Key</h3>
                <p className="text-muted-foreground mb-4">
                  Create an API key with the appropriate permissions and rate limits for
                  your use case.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">2. Authentication</h3>
                <p className="text-muted-foreground mb-2">
                  Include your API key in the <code>X-API-Key</code> header:
                </p>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`curl -X POST "${getBaseUrl()}/api/v1/external/skills/{skill_id}/execute" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -d '{"inputs": {"content": "Your input here"}}'`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">3. Execute a Skill</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`// JavaScript/TypeScript Example
const response = await fetch('${getBaseUrl()}/api/v1/external/skills/{skill_id}/execute', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.API_KEY,
  },
  body: JSON.stringify({
    inputs: {
      content: 'Text to process...',
    },
  }),
});

const result = await response.json();
console.log(result.output);`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">4. Execute a Workflow</h3>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
{`// Python Example
import requests

response = requests.post(
    '${getBaseUrl()}/api/v1/external/workflows/{workflow_id}/execute',
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': os.environ['API_KEY'],
    },
    json={
        'input_data': {'key': 'value'},
        'wait_for_completion': True,
        'timeout_seconds': 300,
    }
)

result = response.json()
print(result['status'], result['output'])`}
                </pre>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-2">5. Error Handling</h3>
                <p className="text-muted-foreground mb-2">
                  Handle common error responses:
                </p>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Status Code</TableHead>
                      <TableHead>Meaning</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow>
                      <TableCell><code>401</code></TableCell>
                      <TableCell>Invalid or missing API key</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><code>403</code></TableCell>
                      <TableCell>API key doesn't have access to this endpoint</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><code>404</code></TableCell>
                      <TableCell>Skill or workflow not found or not published</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><code>429</code></TableCell>
                      <TableCell>Rate limit exceeded</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Usage & Monitoring Tab */}
        <TabsContent value="usage" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total API Calls</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {apiKeys.reduce((sum, key) => sum + key.usage_count, 0)}
                </div>
                <p className="text-xs text-muted-foreground">
                  Across all API keys
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active API Keys</CardTitle>
                <Key className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {apiKeys.filter((k) => k.is_active).length}
                </div>
                <p className="text-xs text-muted-foreground">
                  {apiKeys.filter((k) => !k.is_active).length} revoked
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Published Endpoints</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{endpoints.length}</div>
                <p className="text-xs text-muted-foreground">
                  {endpoints.filter((e) => e.type === "skill").length} skills,{" "}
                  {endpoints.filter((e) => e.type === "workflow").length} workflows
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Recent API Activity</CardTitle>
              <CardDescription>Last 24 hours of API usage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <Activity className="mx-auto h-12 w-12 mb-4 opacity-50" />
                <p>Usage monitoring coming soon</p>
                <p className="text-sm">
                  Detailed logs and analytics will be available here
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Create API Key Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Create API Key</DialogTitle>
            <DialogDescription>
              Create a new API key for external access to your published endpoints.
            </DialogDescription>
          </DialogHeader>
          {newApiKey ? (
            <div className="space-y-4">
              <div className="rounded-lg border bg-muted p-4">
                <Label className="text-sm font-medium">Your API Key</Label>
                <p className="text-xs text-muted-foreground mb-2">
                  Copy this key now - it won't be shown again!
                </p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-sm bg-background px-3 py-2 rounded break-all">
                    {newApiKey}
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard(newApiKey)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <Button
                className="w-full"
                onClick={() => {
                  setShowCreateDialog(false);
                  setNewApiKey(null);
                  setNewKeyName("");
                  setNewKeyDescription("");
                }}
              >
                Done
              </Button>
            </div>
          ) : (
            <>
              <div className="grid gap-4 py-4">
                <div className="grid gap-2">
                  <Label htmlFor="name">Name</Label>
                  <Input
                    id="name"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    placeholder="My API Key"
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="description">Description (optional)</Label>
                  <Textarea
                    id="description"
                    value={newKeyDescription}
                    onChange={(e) => setNewKeyDescription(e.target.value)}
                    placeholder="What is this key for?"
                    rows={2}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="expiry">Expiration</Label>
                  <Select value={newKeyExpiry} onValueChange={setNewKeyExpiry}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="never">Never expires</SelectItem>
                      <SelectItem value="30">30 days</SelectItem>
                      <SelectItem value="90">90 days</SelectItem>
                      <SelectItem value="180">180 days</SelectItem>
                      <SelectItem value="365">1 year</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="rate-limit">Rate Limit (requests per minute)</Label>
                  <Input
                    id="rate-limit"
                    type="number"
                    value={newKeyRateLimit}
                    onChange={(e) => setNewKeyRateLimit(parseInt(e.target.value) || 60)}
                    min={1}
                    max={1000}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setShowCreateDialog(false)}
                >
                  Cancel
                </Button>
                <Button onClick={createApiKey} disabled={!newKeyName}>
                  Create API Key
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Integration Code Dialog */}
      <Dialog
        open={!!selectedEndpoint}
        onOpenChange={() => setSelectedEndpoint(null)}
      >
        <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Integration Code for {selectedEndpoint?.name}</DialogTitle>
            <DialogDescription>
              Copy these code snippets to integrate with your application
            </DialogDescription>
          </DialogHeader>
          {selectedEndpoint && (
            <div className="space-y-4">
              <div>
                <Label className="text-sm font-medium">Endpoint URL</Label>
                <div className="flex items-center gap-2 mt-1">
                  <code className="flex-1 text-sm bg-muted px-3 py-2 rounded font-mono">
                    {getBaseUrl()}{selectedEndpoint.endpoint_url}
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      copyToClipboard(`${getBaseUrl()}${selectedEndpoint.endpoint_url}`)
                    }
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">cURL</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() =>
                      copyToClipboard(
                        `curl -X ${selectedEndpoint.method} "${getBaseUrl()}${selectedEndpoint.endpoint_url}" \\\n  -H "Content-Type: application/json" \\\n  -H "X-API-Key: YOUR_API_KEY" \\\n  -d '${JSON.stringify(selectedEndpoint.example_request || {}, null, 2)}'`
                      )
                    }
                  >
                    <Copy className="mr-2 h-3 w-3" />
                    Copy
                  </Button>
                </div>
                <pre className="bg-muted p-3 rounded-lg overflow-x-auto text-xs mt-1">
{`curl -X ${selectedEndpoint.method} "${getBaseUrl()}${selectedEndpoint.endpoint_url}" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -d '${JSON.stringify(selectedEndpoint.example_request || {}, null, 2)}'`}
                </pre>
              </div>

              <div>
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">JavaScript/TypeScript</Label>
                  <Button variant="ghost" size="sm">
                    <Copy className="mr-2 h-3 w-3" />
                    Copy
                  </Button>
                </div>
                <pre className="bg-muted p-3 rounded-lg overflow-x-auto text-xs mt-1">
{`const response = await fetch('${getBaseUrl()}${selectedEndpoint.endpoint_url}', {
  method: '${selectedEndpoint.method}',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.API_KEY,
  },
  body: JSON.stringify(${JSON.stringify(selectedEndpoint.example_request || {}, null, 4).replace(/\n/g, '\n    ')}),
});

const result = await response.json();`}
                </pre>
              </div>

              <div>
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">Python</Label>
                  <Button variant="ghost" size="sm">
                    <Copy className="mr-2 h-3 w-3" />
                    Copy
                  </Button>
                </div>
                <pre className="bg-muted p-3 rounded-lg overflow-x-auto text-xs mt-1">
{`import requests

response = requests.${selectedEndpoint.method.toLowerCase()}(
    '${getBaseUrl()}${selectedEndpoint.endpoint_url}',
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': os.environ['API_KEY'],
    },
    json=${JSON.stringify(selectedEndpoint.example_request || {}, null, 4).replace(/\n/g, '\n    ')},
)

result = response.json()`}
                </pre>
              </div>

              {selectedEndpoint.input_schema && (
                <div>
                  <Label className="text-sm font-medium">Input Schema</Label>
                  <pre className="bg-muted p-3 rounded-lg overflow-x-auto text-xs mt-1">
                    {JSON.stringify(selectedEndpoint.input_schema, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
