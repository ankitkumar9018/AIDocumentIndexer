"use client";

import { useState } from "react";
import {
  Plus,
  Key,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  Copy,
  RefreshCw,
  Trash2,
  MoreHorizontal,
  Eye,
  EyeOff,
  Clock,
  BarChart3,
  Loader2,
  CheckCircle2,
  XCircle,
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { formatDistanceToNow } from "date-fns";
import {
  useGatewayBudgets,
  useGatewayApiKeys,
  useGatewayUsageStats,
  useCreateGatewayBudget,
  useDeleteGatewayBudget,
  useCreateGatewayApiKey,
  useRevokeGatewayApiKey,
  useRotateGatewayApiKey,
  type Budget,
  type BudgetPeriod,
} from "@/lib/api";

export default function GatewayPage() {
  // Fetch data using real API hooks
  const { data: budgets = [], isLoading: isLoadingBudgets, refetch: refetchBudgets } = useGatewayBudgets();
  const { data: apiKeys = [], isLoading: isLoadingKeys, refetch: refetchKeys } = useGatewayApiKeys();
  const { data: usageStats, isLoading: isLoadingStats } = useGatewayUsageStats();

  // Mutations
  const createBudgetMutation = useCreateGatewayBudget();
  const deleteBudgetMutation = useDeleteGatewayBudget();
  const createKeyMutation = useCreateGatewayApiKey();
  const revokeKeyMutation = useRevokeGatewayApiKey();
  const rotateKeyMutation = useRotateGatewayApiKey();

  // Dialog states
  const [createBudgetOpen, setCreateBudgetOpen] = useState(false);
  const [createKeyOpen, setCreateKeyOpen] = useState(false);
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);
  const [showKeyValue, setShowKeyValue] = useState(false);

  // Form states
  const [budgetForm, setBudgetForm] = useState({
    name: "",
    period: "monthly" as BudgetPeriod,
    limit_amount: "",
    hard_limit: false,
  });

  const [keyForm, setKeyForm] = useState({
    name: "",
    scopes: [] as string[],
    expires_days: "",
  });

  const handleCreateBudget = async () => {
    if (!budgetForm.name || !budgetForm.limit_amount) {
      toast.error("Please fill in all required fields");
      return;
    }

    try {
      await createBudgetMutation.mutateAsync({
        name: budgetForm.name,
        period: budgetForm.period,
        limit_amount: parseFloat(budgetForm.limit_amount),
        hard_limit: budgetForm.hard_limit,
      });
      toast.success("Budget created successfully");
      setCreateBudgetOpen(false);
      setBudgetForm({ name: "", period: "monthly", limit_amount: "", hard_limit: false });
    } catch (error) {
      toast.error(`Failed to create budget: ${(error as Error).message}`);
    }
  };

  const handleCreateApiKey = async () => {
    if (!keyForm.name) {
      toast.error("Please enter a key name");
      return;
    }

    try {
      const result = await createKeyMutation.mutateAsync({
        name: keyForm.name,
        scopes: keyForm.scopes.length > 0 ? keyForm.scopes : undefined,
        expires_in_days: keyForm.expires_days ? parseInt(keyForm.expires_days) : undefined,
      });
      setNewKeyValue(result.key);
      toast.success("API key created successfully");
    } catch (error) {
      toast.error(`Failed to create API key: ${(error as Error).message}`);
    }
  };

  const handleCopyKey = () => {
    if (newKeyValue) {
      navigator.clipboard.writeText(newKeyValue);
      toast.success("API key copied to clipboard");
    }
  };

  const handleRevokeKey = async (keyId: string) => {
    try {
      await revokeKeyMutation.mutateAsync(keyId);
      toast.success("API key revoked");
    } catch (error) {
      toast.error(`Failed to revoke key: ${(error as Error).message}`);
    }
  };

  const handleRotateKey = async (keyId: string) => {
    try {
      const result = await rotateKeyMutation.mutateAsync(keyId);
      setNewKeyValue(result.key);
      setCreateKeyOpen(true);
      toast.success("API key rotated - copy the new key");
    } catch (error) {
      toast.error(`Failed to rotate key: ${(error as Error).message}`);
    }
  };

  const handleDeleteBudget = async (budgetId: string) => {
    try {
      await deleteBudgetMutation.mutateAsync(budgetId);
      toast.success("Budget deleted");
    } catch (error) {
      toast.error(`Failed to delete budget: ${(error as Error).message}`);
    }
  };

  const getUsagePercentage = (budget: Budget) => {
    return Math.min(100, (budget.spent_amount / budget.limit_amount) * 100);
  };

  const getUsageColor = (percentage: number) => {
    if (percentage >= 90) return "text-red-500";
    if (percentage >= 75) return "text-yellow-500";
    return "text-green-500";
  };

  const isLoading = isLoadingBudgets || isLoadingKeys || isLoadingStats;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">LLM Gateway</h1>
          <p className="text-muted-foreground">
            Manage budgets, API keys, and monitor usage
          </p>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Spend</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoadingStats ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold">${(usageStats?.total_cost_usd ?? 0).toFixed(2)}</div>
                <p className="text-xs text-muted-foreground">This billing period</p>
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoadingStats ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold">{(usageStats?.total_requests ?? 0).toLocaleString()}</div>
                <p className="text-xs text-muted-foreground">API calls made</p>
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tokens Used</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoadingStats ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold">{((usageStats?.total_tokens ?? 0) / 1000000).toFixed(1)}M</div>
                <p className="text-xs text-muted-foreground">Total tokens</p>
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Budgets</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoadingBudgets ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold">{budgets.filter((b) => b.is_active).length}</div>
                <p className="text-xs text-muted-foreground">of {budgets.length} total</p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="budgets">
        <TabsList>
          <TabsTrigger value="budgets">
            <DollarSign className="h-4 w-4 mr-2" />
            Budgets
          </TabsTrigger>
          <TabsTrigger value="api-keys">
            <Key className="h-4 w-4 mr-2" />
            API Keys
          </TabsTrigger>
          <TabsTrigger value="usage">
            <BarChart3 className="h-4 w-4 mr-2" />
            Usage
          </TabsTrigger>
        </TabsList>

        {/* Budgets Tab */}
        <TabsContent value="budgets" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-semibold">Budget Management</h2>
            <Button onClick={() => setCreateBudgetOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Budget
            </Button>
          </div>

          {isLoading ? (
            <div className="space-y-3">
              {[1, 2].map((i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : budgets.length > 0 ? (
            <div className="grid gap-4">
              {budgets.map((budget) => {
                const percentage = getUsagePercentage(budget);
                const colorClass = getUsageColor(percentage);

                return (
                  <Card key={budget.id}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${budget.is_active ? "bg-green-100" : "bg-gray-100"}`}>
                            <DollarSign className={`h-5 w-5 ${budget.is_active ? "text-green-600" : "text-gray-400"}`} />
                          </div>
                          <div>
                            <CardTitle className="text-base">{budget.name}</CardTitle>
                            <CardDescription className="capitalize">
                              {budget.period} budget
                            </CardDescription>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={budget.is_active ? "default" : "secondary"}>
                            {budget.is_active ? "Active" : "Inactive"}
                          </Badge>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem>Edit</DropdownMenuItem>
                              <DropdownMenuItem>View Details</DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                className="text-red-600"
                                onClick={() => handleDeleteBudget(budget.id)}
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">
                            ${budget.spent_amount.toFixed(2)} of ${budget.limit_amount.toFixed(2)}
                          </span>
                          <span className={colorClass}>{percentage.toFixed(1)}%</span>
                        </div>
                        <Progress value={percentage} className="h-2" />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Resets: {formatDistanceToNow(new Date(budget.reset_at), { addSuffix: true })}</span>
                          {budget.hard_limit && (
                            <span className="text-yellow-600">Hard limit enabled</span>
                          )}
                        </div>
                        {percentage >= 80 && (
                          <div className="flex items-center gap-2 text-yellow-600 text-sm">
                            <AlertTriangle className="h-4 w-4" />
                            Approaching budget limit
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <DollarSign className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No budgets configured</h3>
                <p className="text-muted-foreground text-center max-w-sm mb-4">
                  Create budgets to control spending and get alerts when limits are reached.
                </p>
                <Button onClick={() => setCreateBudgetOpen(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Budget
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* API Keys Tab */}
        <TabsContent value="api-keys" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-semibold">API Keys</h2>
            <Button onClick={() => setCreateKeyOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create API Key
            </Button>
          </div>

          {apiKeys.length > 0 ? (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Key</TableHead>
                    <TableHead>Scopes</TableHead>
                    <TableHead>Usage</TableHead>
                    <TableHead>Last Used</TableHead>
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
                          {key.key_prefix}
                        </code>
                      </TableCell>
                      <TableCell>
                        <div className="flex gap-1 flex-wrap">
                          {key.scopes.map((scope) => (
                            <Badge key={scope} variant="outline" className="text-xs">
                              {scope}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="text-sm">
                          <div>{key.usage_count.toLocaleString()} requests</div>
                          <div className="text-muted-foreground">{(key.total_tokens / 1000).toFixed(0)}k tokens</div>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {key.last_used_at
                          ? formatDistanceToNow(new Date(key.last_used_at), { addSuffix: true })
                          : "Never"}
                      </TableCell>
                      <TableCell>
                        <Badge variant={key.is_active ? "default" : "secondary"}>
                          {key.is_active ? (
                            <>
                              <CheckCircle2 className="h-3 w-3 mr-1" />
                              Active
                            </>
                          ) : (
                            <>
                              <XCircle className="h-3 w-3 mr-1" />
                              Revoked
                            </>
                          )}
                        </Badge>
                        {key.expires_at && (
                          <div className="text-xs text-muted-foreground mt-1">
                            Expires {formatDistanceToNow(new Date(key.expires_at), { addSuffix: true })}
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleRotateKey(key.id)}>
                              <RefreshCw className="h-4 w-4 mr-2" />
                              Rotate Key
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              className="text-red-600"
                              onClick={() => handleRevokeKey(key.id)}
                            >
                              <XCircle className="h-4 w-4 mr-2" />
                              Revoke
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Key className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No API keys</h3>
                <p className="text-muted-foreground text-center max-w-sm mb-4">
                  Create API keys to access the LLM gateway from your applications.
                </p>
                <Button onClick={() => setCreateKeyOpen(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Create API Key
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Usage Tab */}
        <TabsContent value="usage" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Usage by Model</CardTitle>
              <CardDescription>
                Token usage breakdown by model
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingStats ? (
                <Skeleton className="h-48 w-full" />
              ) : usageStats?.by_model && Object.keys(usageStats.by_model).length > 0 ? (
                <div className="space-y-4">
                  {Object.entries(usageStats.by_model).map(([model, stats]) => (
                    <div key={model} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <div className="font-medium">{model}</div>
                        <div className="text-sm text-muted-foreground">
                          {stats.requests.toLocaleString()} requests
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">${stats.cost_usd.toFixed(2)}</div>
                        <div className="text-sm text-muted-foreground">
                          {(stats.tokens / 1000).toFixed(0)}k tokens
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4" />
                  <p>No usage data available yet</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Usage by Operation</CardTitle>
              <CardDescription>
                Breakdown by operation type
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingStats ? (
                <Skeleton className="h-48 w-full" />
              ) : usageStats?.by_operation && Object.keys(usageStats.by_operation).length > 0 ? (
                <div className="space-y-4">
                  {Object.entries(usageStats.by_operation).map(([operation, stats]) => (
                    <div key={operation} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <div className="font-medium capitalize">{operation.replace(/_/g, " ")}</div>
                        <div className="text-sm text-muted-foreground">
                          {stats.requests.toLocaleString()} requests
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">${stats.cost_usd.toFixed(2)}</div>
                        <div className="text-sm text-muted-foreground">
                          {(stats.tokens / 1000).toFixed(0)}k tokens
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4" />
                  <p>No operation data available yet</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Create Budget Dialog */}
      <Dialog open={createBudgetOpen} onOpenChange={setCreateBudgetOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Budget</DialogTitle>
            <DialogDescription>
              Set spending limits and get alerts when thresholds are reached.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="budget-name">Budget Name</Label>
              <Input
                id="budget-name"
                placeholder="e.g., Monthly Organization Budget"
                value={budgetForm.name}
                onChange={(e) => setBudgetForm({ ...budgetForm, name: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="budget-period">Period</Label>
              <Select
                value={budgetForm.period}
                onValueChange={(v) => setBudgetForm({ ...budgetForm, period: v as BudgetPeriod })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="daily">Daily</SelectItem>
                  <SelectItem value="weekly">Weekly</SelectItem>
                  <SelectItem value="monthly">Monthly</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="limit-amount">Limit Amount ($)</Label>
              <Input
                id="limit-amount"
                type="number"
                placeholder="500"
                value={budgetForm.limit_amount}
                onChange={(e) => setBudgetForm({ ...budgetForm, limit_amount: e.target.value })}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Hard Limit</Label>
                <p className="text-xs text-muted-foreground">
                  Block requests when limit is reached
                </p>
              </div>
              <Switch
                checked={budgetForm.hard_limit}
                onCheckedChange={(checked) => setBudgetForm({ ...budgetForm, hard_limit: checked })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateBudgetOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateBudget} disabled={createBudgetMutation.isPending}>
              {createBudgetMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Create Budget
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create API Key Dialog */}
      <Dialog open={createKeyOpen} onOpenChange={(open) => {
        setCreateKeyOpen(open);
        if (!open) {
          setNewKeyValue(null);
          setKeyForm({ name: "", scopes: [], expires_days: "" });
        }
      }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {newKeyValue ? "API Key Created" : "Create API Key"}
            </DialogTitle>
            <DialogDescription>
              {newKeyValue
                ? "Copy your API key now. You won't be able to see it again."
                : "Create a new API key to access the LLM gateway."}
            </DialogDescription>
          </DialogHeader>
          {newKeyValue ? (
            <div className="space-y-4 py-4">
              <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                <code className="flex-1 text-sm break-all">
                  {showKeyValue ? newKeyValue : "sk-" + "*".repeat(40)}
                </code>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowKeyValue(!showKeyValue)}
                >
                  {showKeyValue ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
                <Button variant="ghost" size="icon" onClick={handleCopyKey}>
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center gap-2 text-yellow-600 text-sm">
                <AlertTriangle className="h-4 w-4" />
                Make sure to copy your API key. You won't see it again!
              </div>
            </div>
          ) : (
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="key-name">Key Name</Label>
                <Input
                  id="key-name"
                  placeholder="e.g., Production API"
                  value={keyForm.name}
                  onChange={(e) => setKeyForm({ ...keyForm, name: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label>Scopes</Label>
                <div className="grid grid-cols-2 gap-2">
                  {["chat", "rag", "documents", "audio", "workflows", "admin"].map((scope) => (
                    <div key={scope} className="flex items-center gap-2">
                      <Switch
                        checked={keyForm.scopes.includes(scope)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setKeyForm({ ...keyForm, scopes: [...keyForm.scopes, scope] });
                          } else {
                            setKeyForm({ ...keyForm, scopes: keyForm.scopes.filter((s) => s !== scope) });
                          }
                        }}
                      />
                      <span className="text-sm capitalize">{scope}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="expires">Expiration (days)</Label>
                <Input
                  id="expires"
                  type="number"
                  placeholder="Leave empty for no expiration"
                  value={keyForm.expires_days}
                  onChange={(e) => setKeyForm({ ...keyForm, expires_days: e.target.value })}
                />
              </div>
            </div>
          )}
          <DialogFooter>
            {newKeyValue ? (
              <Button onClick={() => setCreateKeyOpen(false)}>Done</Button>
            ) : (
              <>
                <Button variant="outline" onClick={() => setCreateKeyOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateApiKey} disabled={!keyForm.name || createKeyMutation.isPending}>
                  {createKeyMutation.isPending && (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  )}
                  Create Key
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
