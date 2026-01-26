"use client";

/**
 * AIDocumentIndexer - Feature Control Panel (Phase 55)
 * =====================================================
 *
 * Superadmin panel for managing system-wide feature toggles.
 *
 * Features:
 * - Enable/disable features at runtime
 * - View feature dependencies
 * - Check API key requirements
 * - Monitor feature health status
 */

import * as React from "react";
import {
  AlertCircle,
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Info,
  Loader2,
  Power,
  PowerOff,
  RefreshCw,
  Settings,
  Shield,
  Zap,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

// =============================================================================
// TYPES
// =============================================================================

interface Feature {
  id: string;
  name: string;
  description: string;
  category: string;
  enabled: boolean;
  requires_api_key: string | null;
  dependencies: string[];
  status: "available" | "unavailable" | "degraded";
}

interface FeatureHealth {
  counts: {
    available: number;
    unavailable: number;
    degraded: number;
    enabled: number;
    disabled: number;
  };
  issues: Array<{
    feature_id: string;
    feature_name: string;
    issue: string;
  }>;
  total_features: number;
}

interface FeatureControlPanelProps {
  className?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

const CATEGORY_ICONS: Record<string, React.ElementType> = {
  "Document Processing": Settings,
  "Advanced RAG": Zap,
  "Distributed Processing": RefreshCw,
  "Audio & Voice": Info,
  "Workflow & Agents": Shield,
  "Integrations": ExternalLink,
};

const STATUS_CONFIG = {
  available: {
    color: "text-green-600",
    bgColor: "bg-green-100",
    label: "Available",
  },
  unavailable: {
    color: "text-red-600",
    bgColor: "bg-red-100",
    label: "Unavailable",
  },
  degraded: {
    color: "text-yellow-600",
    bgColor: "bg-yellow-100",
    label: "Degraded",
  },
};

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function fetchFeatures(): Promise<{ features: Feature[]; categories: string[] }> {
  const response = await fetch(`${API_BASE}/admin/features`, {
    credentials: "include",
  });
  if (!response.ok) throw new Error("Failed to fetch features");
  return response.json();
}

async function fetchFeatureHealth(): Promise<FeatureHealth> {
  const response = await fetch(`${API_BASE}/admin/features/health/check`, {
    credentials: "include",
  });
  if (!response.ok) throw new Error("Failed to fetch feature health");
  return response.json();
}

async function toggleFeature(featureId: string, enabled: boolean): Promise<any> {
  const response = await fetch(`${API_BASE}/admin/features/${featureId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to toggle feature");
  }
  return response.json();
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function FeatureControlPanel({ className }: FeatureControlPanelProps) {
  const [features, setFeatures] = React.useState<Feature[]>([]);
  const [categories, setCategories] = React.useState<string[]>([]);
  const [health, setHealth] = React.useState<FeatureHealth | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [toggling, setToggling] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = React.useState<Set<string>>(new Set());

  // Confirmation dialog state
  const [confirmDialog, setConfirmDialog] = React.useState<{
    open: boolean;
    feature: Feature | null;
    action: "enable" | "disable";
  }>({ open: false, feature: null, action: "enable" });

  // Load data
  const loadData = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [featuresData, healthData] = await Promise.all([
        fetchFeatures(),
        fetchFeatureHealth(),
      ]);
      setFeatures(featuresData.features);
      setCategories(featuresData.categories);
      setHealth(healthData);
      // Expand first category by default
      if (featuresData.categories.length > 0) {
        setExpandedCategories(new Set([featuresData.categories[0]]));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load features");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle toggle
  const handleToggle = async (feature: Feature, enabled: boolean) => {
    // For disabling, show confirmation
    if (!enabled && feature.enabled) {
      setConfirmDialog({ open: true, feature, action: "disable" });
      return;
    }

    await executeToggle(feature, enabled);
  };

  const executeToggle = async (feature: Feature, enabled: boolean) => {
    setToggling(feature.id);
    setError(null);

    try {
      const result = await toggleFeature(feature.id, enabled);

      // Update local state
      setFeatures((prev) =>
        prev.map((f) =>
          f.id === feature.id ? { ...f, enabled: result.enabled } : f
        )
      );

      // Show restart notice if needed
      if (result.requires_restart) {
        setError(`Feature updated. Server restart required for full effect.`);
      }

      // Refresh health
      const healthData = await fetchFeatureHealth();
      setHealth(healthData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to toggle feature");
    } finally {
      setToggling(null);
      setConfirmDialog({ open: false, feature: null, action: "enable" });
    }
  };

  // Group features by category
  const featuresByCategory = React.useMemo(() => {
    const grouped: Record<string, Feature[]> = {};
    for (const feature of features) {
      if (!grouped[feature.category]) {
        grouped[feature.category] = [];
      }
      grouped[feature.category].push(feature);
    }
    return grouped;
  }, [features]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Power className="h-6 w-6" />
            Feature Controls
          </h2>
          <p className="text-muted-foreground mt-1">
            Enable or disable system features. Some changes require server restart.
          </p>
        </div>
        <Button variant="outline" onClick={loadData} className="gap-2">
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant={error.includes("restart") ? "default" : "destructive"}>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>{error.includes("restart") ? "Notice" : "Error"}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Health Overview */}
      {health && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-green-600">{health.counts.enabled}</div>
              <p className="text-sm text-muted-foreground">Enabled</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-gray-600">{health.counts.disabled}</div>
              <p className="text-sm text-muted-foreground">Disabled</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-green-600">{health.counts.available}</div>
              <p className="text-sm text-muted-foreground">Available</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-yellow-600">{health.counts.degraded}</div>
              <p className="text-sm text-muted-foreground">Degraded</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold text-red-600">{health.counts.unavailable}</div>
              <p className="text-sm text-muted-foreground">Unavailable</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Issues Alert */}
      {health && health.issues.length > 0 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Configuration Issues ({health.issues.length})</AlertTitle>
          <AlertDescription>
            <ul className="mt-2 space-y-1">
              {health.issues.slice(0, 5).map((issue) => (
                <li key={issue.feature_id} className="text-sm">
                  <strong>{issue.feature_name}:</strong> {issue.issue}
                </li>
              ))}
              {health.issues.length > 5 && (
                <li className="text-sm text-muted-foreground">
                  ...and {health.issues.length - 5} more issues
                </li>
              )}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Feature Categories */}
      <div className="space-y-4">
        {categories.map((category) => {
          const categoryFeatures = featuresByCategory[category] || [];
          const enabledCount = categoryFeatures.filter((f) => f.enabled).length;
          const Icon = CATEGORY_ICONS[category] || Settings;
          const isExpanded = expandedCategories.has(category);

          return (
            <Collapsible
              key={category}
              open={isExpanded}
              onOpenChange={(open) => {
                setExpandedCategories((prev) => {
                  const next = new Set(prev);
                  if (open) {
                    next.add(category);
                  } else {
                    next.delete(category);
                  }
                  return next;
                });
              }}
            >
              <Card>
                <CollapsibleTrigger asChild>
                  <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-primary/10">
                          <Icon className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <CardTitle className="text-lg">{category}</CardTitle>
                          <CardDescription>
                            {enabledCount}/{categoryFeatures.length} features enabled
                          </CardDescription>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">
                          {categoryFeatures.length} features
                        </Badge>
                        {isExpanded ? (
                          <ChevronDown className="h-5 w-5" />
                        ) : (
                          <ChevronRight className="h-5 w-5" />
                        )}
                      </div>
                    </div>
                  </CardHeader>
                </CollapsibleTrigger>

                <CollapsibleContent>
                  <CardContent className="pt-0">
                    <div className="space-y-4">
                      {categoryFeatures.map((feature) => (
                        <FeatureRow
                          key={feature.id}
                          feature={feature}
                          features={features}
                          toggling={toggling === feature.id}
                          onToggle={(enabled) => handleToggle(feature, enabled)}
                        />
                      ))}
                    </div>
                  </CardContent>
                </CollapsibleContent>
              </Card>
            </Collapsible>
          );
        })}
      </div>

      {/* Confirmation Dialog */}
      <Dialog
        open={confirmDialog.open}
        onOpenChange={(open) =>
          setConfirmDialog({ ...confirmDialog, open })
        }
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Disable Feature?</DialogTitle>
            <DialogDescription>
              Are you sure you want to disable{" "}
              <strong>{confirmDialog.feature?.name}</strong>?
              {confirmDialog.feature?.dependencies &&
                confirmDialog.feature.dependencies.length > 0 && (
                  <span className="block mt-2 text-yellow-600">
                    This may affect dependent features.
                  </span>
                )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() =>
                setConfirmDialog({ open: false, feature: null, action: "enable" })
              }
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() =>
                confirmDialog.feature &&
                executeToggle(confirmDialog.feature, false)
              }
            >
              Disable Feature
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// =============================================================================
// FEATURE ROW COMPONENT
// =============================================================================

interface FeatureRowProps {
  feature: Feature;
  features: Feature[];
  toggling: boolean;
  onToggle: (enabled: boolean) => void;
}

function FeatureRow({ feature, features, toggling, onToggle }: FeatureRowProps) {
  const statusConfig = STATUS_CONFIG[feature.status];

  // Find dependent features
  const dependentFeatures = features.filter((f) =>
    f.dependencies.includes(feature.id)
  );

  return (
    <div
      className={cn(
        "flex items-start justify-between p-4 rounded-lg border",
        feature.enabled && "bg-green-50/50 border-green-200",
        !feature.enabled && "bg-gray-50/50",
        feature.status === "unavailable" && "opacity-60"
      )}
    >
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium">{feature.name}</span>
          <Badge
            variant="outline"
            className={cn(statusConfig.bgColor, statusConfig.color, "text-xs")}
          >
            {statusConfig.label}
          </Badge>
          {feature.enabled && (
            <Badge variant="secondary" className="text-xs gap-1">
              <Check className="h-3 w-3" />
              Enabled
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground mt-1">{feature.description}</p>

        {/* Dependencies */}
        {feature.dependencies.length > 0 && (
          <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
            <span>Requires:</span>
            {feature.dependencies.map((dep) => {
              const depFeature = features.find((f) => f.id === dep);
              return (
                <Badge
                  key={dep}
                  variant="outline"
                  className={cn(
                    "text-xs",
                    depFeature?.enabled
                      ? "bg-green-50 text-green-700"
                      : "bg-red-50 text-red-700"
                  )}
                >
                  {depFeature?.name || dep}
                </Badge>
              );
            })}
          </div>
        )}

        {/* Dependent features warning */}
        {feature.enabled && dependentFeatures.length > 0 && (
          <div className="mt-2 text-xs text-yellow-600">
            Used by: {dependentFeatures.map((f) => f.name).join(", ")}
          </div>
        )}

        {/* API Key requirement */}
        {feature.requires_api_key && feature.status === "unavailable" && (
          <div className="mt-2 text-xs text-red-600">
            Requires: {feature.requires_api_key}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 ml-4">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div>
                <Switch
                  checked={feature.enabled}
                  onCheckedChange={onToggle}
                  disabled={
                    toggling ||
                    feature.status === "unavailable" ||
                    (feature.status === "degraded" && !feature.enabled)
                  }
                />
              </div>
            </TooltipTrigger>
            <TooltipContent>
              {feature.status === "unavailable"
                ? "Configure required API keys first"
                : feature.status === "degraded"
                ? "Enable dependencies first"
                : feature.enabled
                ? "Click to disable"
                : "Click to enable"}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {toggling && <Loader2 className="h-4 w-4 animate-spin" />}
      </div>
    </div>
  );
}

export default FeatureControlPanel;
