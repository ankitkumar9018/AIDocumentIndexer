"use client";

import { useState } from "react";
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Bell,
  Plus,
  Trash2,
  BarChart3,
  PieChart,
  Calendar,
  Zap,
  Loader2,
  RefreshCw,
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
  useCostUsage,
  useCostHistory,
  useCurrentCost,
  useCostDashboard,
  useCostAlerts,
  useCreateCostAlert,
  useDeleteCostAlert,
  useEstimateCost,
  useModelPricing,
} from "@/lib/api";

type Period = "day" | "week" | "month" | "year";

export default function CostsPage() {
  const [period, setPeriod] = useState<Period>("month");
  const [alertThreshold, setAlertThreshold] = useState("");
  const [estimateModel, setEstimateModel] = useState("gpt-4");
  const [estimatePrompt, setEstimatePrompt] = useState("");

  // Queries
  const { data: usage, isLoading: usageLoading } = useCostUsage(period);
  const { data: history } = useCostHistory(30);
  const { data: currentCost } = useCurrentCost(period);
  const { data: dashboard, isLoading: dashboardLoading, refetch: refetchDashboard } = useCostDashboard();
  const { data: alerts, refetch: refetchAlerts } = useCostAlerts();
  const { data: pricing } = useModelPricing();

  // Mutations
  const createAlert = useCreateCostAlert();
  const deleteAlert = useDeleteCostAlert();
  const estimateCost = useEstimateCost();

  const handleCreateAlert = async () => {
    if (!alertThreshold) return;
    try {
      await createAlert.mutateAsync({
        threshold: parseFloat(alertThreshold),
        period,
      });
      setAlertThreshold("");
      refetchAlerts();
    } catch (error) {
      console.error("Failed to create alert:", error);
    }
  };

  const handleDeleteAlert = async (alertId: string) => {
    try {
      await deleteAlert.mutateAsync(alertId);
      refetchAlerts();
    } catch (error) {
      console.error("Failed to delete alert:", error);
    }
  };

  const handleEstimate = async () => {
    if (!estimatePrompt) return;
    try {
      await estimateCost.mutateAsync({
        model: estimateModel,
        prompt: estimatePrompt,
      });
    } catch (error) {
      console.error("Failed to estimate cost:", error);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 4,
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat("en-US").format(num);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Cost Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor and manage your LLM usage costs
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Period Selector */}
          <div className="flex rounded-lg border bg-muted p-1">
            {(["day", "week", "month", "year"] as Period[]).map((p) => (
              <Button
                key={p}
                variant={period === p ? "default" : "ghost"}
                size="sm"
                onClick={() => setPeriod(p)}
                className="capitalize"
              >
                {p}
              </Button>
            ))}
          </div>
          <Button onClick={() => refetchDashboard()} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Spent</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dashboardLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                formatCurrency(dashboard?.costs?.last_month || 0)
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              This {period}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tokens</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dashboardLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                "-"
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Token data not available
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Calls</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dashboardLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                formatNumber(dashboard?.requests?.last_month || 0)
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Total requests this {period}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Cost/Request</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dashboardLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : dashboard?.requests?.last_month ? (
                formatCurrency((dashboard?.costs?.last_month || 0) / dashboard.requests.last_month)
              ) : (
                "$0.0000"
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Average per API call
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Usage by Model */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChart className="h-5 w-5" />
              Usage by Model
            </CardTitle>
            <CardDescription>Cost breakdown by LLM model</CardDescription>
          </CardHeader>
          <CardContent>
            {dashboard?.by_model && Object.keys(dashboard.by_model).length > 0 ? (
              <div className="space-y-4">
                {Object.entries(dashboard.by_model).map(([model, cost]) => (
                  <div key={model} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{model}</span>
                      <span className="text-sm text-muted-foreground">
                        {formatCurrency(cost)}
                      </span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary rounded-full"
                        style={{
                          width: `${((cost / (dashboard?.costs?.last_month || 1)) * 100).toFixed(1)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <PieChart className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No usage data yet</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Cost Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Cost Alerts
            </CardTitle>
            <CardDescription>Get notified when spending exceeds thresholds</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Create Alert Form */}
            <div className="flex gap-2">
              <div className="relative flex-1">
                <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  type="number"
                  step="0.01"
                  placeholder="Threshold amount"
                  value={alertThreshold}
                  onChange={(e) => setAlertThreshold(e.target.value)}
                  className="pl-9"
                />
              </div>
              <Button
                onClick={handleCreateAlert}
                disabled={!alertThreshold || createAlert.isPending}
              >
                {createAlert.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Plus className="h-4 w-4" />
                )}
              </Button>
            </div>

            {/* Alerts List */}
            {alerts?.alerts && alerts.alerts.length > 0 ? (
              <div className="space-y-2">
                {alerts.alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`flex items-center justify-between p-3 rounded-lg border ${
                      alert.notification_sent ? "border-yellow-500 bg-yellow-500/10" : ""
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      {alert.notification_sent ? (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      ) : (
                        <Bell className="h-4 w-4 text-muted-foreground" />
                      )}
                      <div>
                        <p className="text-sm font-medium">
                          {formatCurrency(alert.threshold)} / {alert.period}
                        </p>
                        {alert.notification_sent && (
                          <p className="text-xs text-yellow-600">
                            Notification sent
                          </p>
                        )}
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteAlert(alert.id)}
                      disabled={deleteAlert.isPending}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-4 text-muted-foreground text-sm">
                No alerts configured
              </div>
            )}
          </CardContent>
        </Card>

        {/* Cost Estimator */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Cost Estimator
            </CardTitle>
            <CardDescription>Estimate cost before making API calls</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Model</label>
              <select
                value={estimateModel}
                onChange={(e) => setEstimateModel(e.target.value)}
                className="w-full h-10 px-3 rounded-md border bg-background"
              >
                {pricing && Object.keys(pricing).length > 0 ? (
                  Object.keys(pricing).map((model) => (
                    <option key={model} value={model}>{model}</option>
                  ))
                ) : (
                  <>
                    <option value="gpt-4">gpt-4</option>
                    <option value="gpt-4-turbo">gpt-4-turbo</option>
                    <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                    <option value="claude-3-opus">claude-3-opus</option>
                    <option value="claude-3-sonnet">claude-3-sonnet</option>
                  </>
                )}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Prompt</label>
              <textarea
                placeholder="Enter your prompt to estimate cost..."
                value={estimatePrompt}
                onChange={(e) => setEstimatePrompt(e.target.value)}
                className="w-full min-h-[100px] p-3 rounded-md border bg-background resize-none"
              />
            </div>
            <Button
              onClick={handleEstimate}
              disabled={!estimatePrompt || estimateCost.isPending}
              className="w-full"
            >
              {estimateCost.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <DollarSign className="h-4 w-4 mr-2" />
              )}
              Estimate Cost
            </Button>

            {estimateCost.data && (
              <div className="p-4 rounded-lg bg-muted space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Input tokens:</span>
                  <span className="text-sm font-medium">
                    {formatNumber(estimateCost.data.estimated_input_tokens)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Est. output tokens:</span>
                  <span className="text-sm font-medium">
                    {formatNumber(estimateCost.data.estimated_output_tokens)}
                  </span>
                </div>
                <div className="flex justify-between border-t pt-2 mt-2">
                  <span className="text-sm font-medium">Estimated cost:</span>
                  <span className="text-sm font-bold text-primary">
                    {formatCurrency(estimateCost.data.estimated_cost)}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Model Pricing */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              Model Pricing
            </CardTitle>
            <CardDescription>Cost per 1M tokens by model</CardDescription>
          </CardHeader>
          <CardContent>
            {pricing?.pricing && Object.keys(pricing.pricing).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(pricing.pricing).map(([model, prices]) => (
                  <div
                    key={model}
                    className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50"
                  >
                    <span className="text-sm font-medium">{model}</span>
                    <div className="text-right text-xs text-muted-foreground">
                      <p>In: ${prices.input}/1M</p>
                      <p>Out: ${prices.output}/1M</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                <DollarSign className="h-8 w-8 mb-2 opacity-50" />
                <p className="text-sm">Pricing data unavailable</p>
                <p className="text-xs">Check backend configuration</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Usage History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Recent Usage
          </CardTitle>
          <CardDescription>Last 30 API calls</CardDescription>
        </CardHeader>
        <CardContent>
          {history?.records && history.records.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 font-medium">Time</th>
                    <th className="text-left py-2 font-medium">Model</th>
                    <th className="text-left py-2 font-medium">Type</th>
                    <th className="text-right py-2 font-medium">Tokens</th>
                    <th className="text-right py-2 font-medium">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {history.records.slice(0, 10).map((record) => (
                    <tr key={record.id} className="border-b last:border-0">
                      <td className="py-2 text-muted-foreground">
                        {new Date(record.timestamp).toLocaleString()}
                      </td>
                      <td className="py-2">{record.model}</td>
                      <td className="py-2 capitalize">{record.usage_type}</td>
                      <td className="py-2 text-right">
                        {formatNumber(record.input_tokens + record.output_tokens)}
                      </td>
                      <td className="py-2 text-right font-medium">
                        {formatCurrency(record.cost)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No usage history yet</p>
              <p className="text-sm">Start using the API to see your usage here</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
