"use client";

import { useState, useEffect } from "react";
import {
  Shield,
  Eye,
  EyeOff,
  History,
  Brain,
  Trash2,
  Download,
  Clock,
  AlertTriangle,
  Settings,
  MessageSquare,
  Lock,
  Loader2,
  CheckCircle2,
  FileJson,
  FileSpreadsheet,
  FolderArchive,
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
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { toast } from "sonner";
import {
  usePrivacySettings,
  useUpdatePrivacySettings,
  useExportUserData,
  useDeleteUserData,
  type PrivacySettings,
} from "@/lib/api";

export default function PrivacyPage() {
  const { data: privacyData, isLoading: isLoadingSettings } = usePrivacySettings();
  const updateSettingsMutation = useUpdatePrivacySettings();
  const exportMutation = useExportUserData();
  const deleteMutation = useDeleteUserData();

  const [settings, setSettings] = useState<PrivacySettings>({
    chat_history_enabled: true,
    chat_history_admin_visible: false,
    ai_memory_enabled: true,
    memory_retention_days: 90,
    auto_delete_history_days: null,
    incognito_mode: false,
  });

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteType, setDeleteType] = useState<"history" | "memory" | "all">("history");
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState<"json" | "csv" | "zip">("json");

  // Sync settings from API
  useEffect(() => {
    if (privacyData) {
      setSettings(privacyData);
    }
  }, [privacyData]);

  const handleSettingChange = async (
    key: keyof PrivacySettings,
    value: boolean | number | null
  ) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);

    try {
      await updateSettingsMutation.mutateAsync({ [key]: value });
      toast.success("Privacy settings updated");
    } catch {
      // Revert on error
      setSettings(settings);
      toast.error("Failed to update settings");
    }
  };

  const handleExport = async () => {
    try {
      const blob = await exportMutation.mutateAsync(exportFormat);
      // Download the blob
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `my-data.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success(`Data exported as ${exportFormat.toUpperCase()}`);
      setExportDialogOpen(false);
    } catch {
      toast.error("Export failed");
    }
  };

  const handleDelete = async () => {
    try {
      await deleteMutation.mutateAsync(deleteType);
      const messages = {
        history: "Chat history deleted",
        memory: "AI memory cleared",
        all: "All data deleted",
      };
      toast.success(messages[deleteType]);
      setDeleteDialogOpen(false);
    } catch {
      toast.error("Delete failed");
    }
  };

  const isLoading = updateSettingsMutation.isPending;
  const isExporting = exportMutation.isPending;
  const isDeleting = deleteMutation.isPending;

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Privacy & Data</h1>
        <p className="text-muted-foreground">
          Control how your data is stored and used
        </p>
      </div>

      {/* Chat History Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-100">
              <History className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <CardTitle>Chat History</CardTitle>
              <CardDescription>
                Control how your conversation history is stored
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Save Chat History</Label>
              <p className="text-sm text-muted-foreground">
                Keep a record of your conversations
              </p>
            </div>
            <Switch
              checked={settings.chat_history_enabled}
              onCheckedChange={(checked) =>
                handleSettingChange("chat_history_enabled", checked)
              }
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Admin Visibility</Label>
              <p className="text-sm text-muted-foreground">
                Allow organization admins to view your chat history
              </p>
            </div>
            <Switch
              checked={settings.chat_history_admin_visible}
              onCheckedChange={(checked) =>
                handleSettingChange("chat_history_admin_visible", checked)
              }
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Auto-Delete History</Label>
              <p className="text-sm text-muted-foreground">
                Automatically delete old conversations
              </p>
            </div>
            <Select
              value={settings.auto_delete_history_days?.toString() || "never"}
              onValueChange={(v) =>
                handleSettingChange(
                  "auto_delete_history_days",
                  v === "never" ? null : parseInt(v)
                )
              }
            >
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="never">Never</SelectItem>
                <SelectItem value="7">After 7 days</SelectItem>
                <SelectItem value="30">After 30 days</SelectItem>
                <SelectItem value="90">After 90 days</SelectItem>
                <SelectItem value="365">After 1 year</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* AI Memory Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-100">
              <Brain className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <CardTitle>AI Memory</CardTitle>
              <CardDescription>
                Control what the AI remembers about you
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Enable AI Memory</Label>
              <p className="text-sm text-muted-foreground">
                Allow the AI to remember context from previous conversations
              </p>
            </div>
            <Switch
              checked={settings.ai_memory_enabled}
              onCheckedChange={(checked) =>
                handleSettingChange("ai_memory_enabled", checked)
              }
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Memory Retention</Label>
              <p className="text-sm text-muted-foreground">
                How long to keep AI memory entries
              </p>
            </div>
            <Select
              value={settings.memory_retention_days?.toString() || "forever"}
              onValueChange={(v) =>
                handleSettingChange(
                  "memory_retention_days",
                  v === "forever" ? null : parseInt(v)
                )
              }
              disabled={!settings.ai_memory_enabled}
            >
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="30">30 days</SelectItem>
                <SelectItem value="90">90 days</SelectItem>
                <SelectItem value="180">180 days</SelectItem>
                <SelectItem value="365">1 year</SelectItem>
                <SelectItem value="forever">Forever</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {settings.ai_memory_enabled && (
            <div className="bg-muted/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <span>AI memory is active and learning from your conversations</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Incognito Mode Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gray-100">
              <EyeOff className="h-5 w-5 text-gray-600" />
            </div>
            <div>
              <CardTitle>Incognito Mode</CardTitle>
              <CardDescription>
                Start a private session that won't be saved
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Enable Incognito Mode</Label>
              <p className="text-sm text-muted-foreground">
                Your next chat will not be saved to history or affect AI memory
              </p>
            </div>
            <Switch
              checked={settings.incognito_mode}
              onCheckedChange={(checked) =>
                handleSettingChange("incognito_mode", checked)
              }
            />
          </div>
          {settings.incognito_mode && (
            <div className="mt-4 flex items-center gap-2 text-sm text-yellow-600 bg-yellow-50 rounded-lg p-3">
              <Lock className="h-4 w-4" />
              Incognito mode is active. Your chats will not be saved.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Data Management Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-green-100">
              <Settings className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <CardTitle>Data Management</CardTitle>
              <CardDescription>
                Export or delete your data
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <Button
              variant="outline"
              className="flex-1"
              onClick={() => setExportDialogOpen(true)}
            >
              <Download className="h-4 w-4 mr-2" />
              Export My Data
            </Button>
            <Button
              variant="outline"
              className="flex-1 text-red-600 hover:text-red-700 hover:bg-red-50"
              onClick={() => setDeleteDialogOpen(true)}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete Data
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            You can export all your data at any time. Deletion is permanent and cannot be undone.
          </p>
        </CardContent>
      </Card>

      {/* Privacy Summary */}
      <Card className="bg-muted/30">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <Shield className="h-5 w-5 text-primary mt-0.5" />
            <div className="space-y-2">
              <h4 className="font-medium">Your Privacy Summary</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li className="flex items-center gap-2">
                  {settings.chat_history_enabled ? (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  )}
                  Chat history is {settings.chat_history_enabled ? "enabled" : "disabled"}
                </li>
                <li className="flex items-center gap-2">
                  {settings.ai_memory_enabled ? (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  )}
                  AI memory is {settings.ai_memory_enabled ? "enabled" : "disabled"}
                </li>
                <li className="flex items-center gap-2">
                  {!settings.chat_history_admin_visible ? (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  ) : (
                    <Eye className="h-4 w-4 text-yellow-500" />
                  )}
                  Chats are {settings.chat_history_admin_visible ? "visible" : "hidden"} to admins
                </li>
                {settings.incognito_mode && (
                  <li className="flex items-center gap-2">
                    <Lock className="h-4 w-4 text-primary" />
                    Incognito mode is active
                  </li>
                )}
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export Your Data</DialogTitle>
            <DialogDescription>
              Download a copy of your data in your preferred format.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label className="mb-3 block">Export Format</Label>
            <RadioGroup
              value={exportFormat}
              onValueChange={(v) => setExportFormat(v as any)}
              className="grid grid-cols-3 gap-4"
            >
              <div>
                <RadioGroupItem
                  value="json"
                  id="json"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="json"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                >
                  <FileJson className="mb-3 h-6 w-6" />
                  <span className="text-sm font-medium">JSON</span>
                </Label>
              </div>
              <div>
                <RadioGroupItem
                  value="csv"
                  id="csv"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="csv"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                >
                  <FileSpreadsheet className="mb-3 h-6 w-6" />
                  <span className="text-sm font-medium">CSV</span>
                </Label>
              </div>
              <div>
                <RadioGroupItem
                  value="zip"
                  id="zip"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="zip"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                >
                  <FolderArchive className="mb-3 h-6 w-6" />
                  <span className="text-sm font-medium">ZIP (All)</span>
                </Label>
              </div>
            </RadioGroup>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setExportDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={isExporting}>
              {isExporting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Export
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-red-600">Delete Data</DialogTitle>
            <DialogDescription>
              This action cannot be undone. Please choose what to delete.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <RadioGroup
              value={deleteType}
              onValueChange={(v) => setDeleteType(v as any)}
              className="space-y-3"
            >
              <div className="flex items-center space-x-3 rounded-lg border p-4">
                <RadioGroupItem value="history" id="delete-history" />
                <Label htmlFor="delete-history" className="flex-1 cursor-pointer">
                  <div className="font-medium">Chat History Only</div>
                  <div className="text-sm text-muted-foreground">
                    Delete all conversation history
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-3 rounded-lg border p-4">
                <RadioGroupItem value="memory" id="delete-memory" />
                <Label htmlFor="delete-memory" className="flex-1 cursor-pointer">
                  <div className="font-medium">AI Memory Only</div>
                  <div className="text-sm text-muted-foreground">
                    Clear what the AI remembers about you
                  </div>
                </Label>
              </div>
              <div className="flex items-center space-x-3 rounded-lg border border-red-200 p-4 bg-red-50">
                <RadioGroupItem value="all" id="delete-all" />
                <Label htmlFor="delete-all" className="flex-1 cursor-pointer">
                  <div className="font-medium text-red-600">Delete Everything</div>
                  <div className="text-sm text-muted-foreground">
                    Delete all history, memory, and preferences
                  </div>
                </Label>
              </div>
            </RadioGroup>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={isDeleting}>
              {isDeleting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
