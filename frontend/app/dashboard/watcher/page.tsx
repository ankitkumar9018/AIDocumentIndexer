"use client";

import { useState, useCallback } from "react";
import { useSession } from "next-auth/react";
import {
  FolderSearch,
  Play,
  Pause,
  Square,
  RefreshCw,
  Plus,
  Trash2,
  Power,
  PowerOff,
  Eye,
  Loader2,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  FileText,
  Folder,
  Settings2,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface WatcherStats {
  status: string;
  directories_watched: number;
  events_queued: number;
  events_processed: number;
  events_failed: number;
  last_event_time: string | null;
  uptime_seconds: number;
}

interface WatchDirectory {
  id: string;
  path: string;
  recursive: boolean;
  auto_process: boolean;
  access_tier: number;
  collection: string | null;
  folder_id: string | null;
  enabled: boolean;
  created_at: string;
}

interface FileEvent {
  id: string;
  event_type: string;
  file_path: string;
  file_name: string;
  file_extension: string;
  file_size: number;
  file_hash: string | null;
  watch_dir_id: string;
  timestamp: string;
  processed: boolean;
  processing_error: string | null;
}

export default function WatcherPage() {
  const { data: session } = useSession();
  const accessToken = (session as any)?.accessToken as string | undefined;
  const queryClient = useQueryClient();

  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newDir, setNewDir] = useState({
    path: "",
    recursive: true,
    auto_process: true,
    access_tier: 1,
    collection: "",
  });

  const headers = {
    Authorization: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
  };

  // Fetch watcher status
  const { data: status, isLoading: statusLoading, error: statusError } = useQuery<WatcherStats>({
    queryKey: ["watcher-status"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/status`, { headers });
      if (!res.ok) throw new Error("Failed to fetch status");
      return res.json();
    },
    enabled: !!accessToken,
    refetchInterval: 5000,
  });

  // Fetch directories
  const { data: directories, isLoading: dirsLoading } = useQuery<WatchDirectory[]>({
    queryKey: ["watcher-directories"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/directories`, { headers });
      if (!res.ok) throw new Error("Failed to fetch directories");
      return res.json();
    },
    enabled: !!accessToken,
  });

  // Fetch events
  const { data: events, isLoading: eventsLoading } = useQuery<FileEvent[]>({
    queryKey: ["watcher-events"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/events?limit=50&include_processed=true`, { headers });
      if (!res.ok) throw new Error("Failed to fetch events");
      return res.json();
    },
    enabled: !!accessToken,
    refetchInterval: 10000,
  });

  // Mutations
  const startWatcher = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/start`, { method: "POST", headers });
      if (!res.ok) throw new Error("Failed to start watcher");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watcher-status"] }),
  });

  const stopWatcher = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/stop`, { method: "POST", headers });
      if (!res.ok) throw new Error("Failed to stop watcher");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watcher-status"] }),
  });

  const pauseWatcher = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/pause`, { method: "POST", headers });
      if (!res.ok) throw new Error("Failed to pause watcher");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watcher-status"] }),
  });

  const resumeWatcher = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/resume`, { method: "POST", headers });
      if (!res.ok) throw new Error("Failed to resume watcher");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watcher-status"] }),
  });

  const addDirectory = useMutation({
    mutationFn: async (data: typeof newDir) => {
      const res = await fetch(`${API_BASE}/watcher/directories`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          path: data.path,
          recursive: data.recursive,
          auto_process: data.auto_process,
          access_tier: data.access_tier,
          collection: data.collection || null,
        }),
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Failed to add directory");
      }
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["watcher-directories"] });
      setAddDialogOpen(false);
      setNewDir({ path: "", recursive: true, auto_process: true, access_tier: 1, collection: "" });
    },
  });

  const removeDirectory = useMutation({
    mutationFn: async (id: string) => {
      const res = await fetch(`${API_BASE}/watcher/directories/${id}`, { method: "DELETE", headers });
      if (!res.ok) throw new Error("Failed to remove directory");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watcher-directories"] }),
  });

  const toggleDirectory = useMutation({
    mutationFn: async (id: string) => {
      const res = await fetch(`${API_BASE}/watcher/directories/${id}/toggle`, { method: "POST", headers });
      if (!res.ok) throw new Error("Failed to toggle directory");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watcher-directories"] }),
  });

  const scanDirectory = useMutation({
    mutationFn: async (id: string) => {
      const res = await fetch(`${API_BASE}/watcher/directories/${id}/scan`, {
        method: "POST",
        headers,
        body: JSON.stringify({ process_existing: true }),
      });
      if (!res.ok) throw new Error("Failed to scan directory");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["watcher-events"] });
      queryClient.invalidateQueries({ queryKey: ["watcher-status"] });
    },
  });

  const retryEvent = useMutation({
    mutationFn: async (id: string) => {
      const res = await fetch(`${API_BASE}/watcher/events/${id}/retry`, { method: "POST", headers });
      if (!res.ok) throw new Error("Failed to retry event");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watcher-events"] }),
  });

  const clearEvents = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/watcher/events/clear`, { method: "POST", headers });
      if (!res.ok) throw new Error("Failed to clear events");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["watcher-events"] });
      queryClient.invalidateQueries({ queryKey: ["watcher-status"] });
    },
  });

  const formatUptime = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "bg-green-500";
      case "paused":
        return "bg-yellow-500";
      case "stopped":
        return "bg-gray-400";
      default:
        return "bg-gray-400";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "running":
        return <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">Running</Badge>;
      case "paused":
        return <Badge className="bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400">Paused</Badge>;
      case "stopped":
        return <Badge variant="secondary">Stopped</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  if (!accessToken) {
    return (
      <div className="p-6">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Please sign in to access the file watcher.</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <FolderSearch className="h-6 w-6" />
            File Watcher
          </h1>
          <p className="text-muted-foreground">
            Automatically index files from monitored directories
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            queryClient.invalidateQueries({ queryKey: ["watcher-status"] });
            queryClient.invalidateQueries({ queryKey: ["watcher-directories"] });
            queryClient.invalidateQueries({ queryKey: ["watcher-events"] });
          }}
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {statusError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to connect to file watcher service. Make sure the backend is running.
          </AlertDescription>
        </Alert>
      )}

      {/* Status Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5" />
                Watcher Status
              </CardTitle>
              <CardDescription>Monitor and control the file watcher service</CardDescription>
            </div>
            {status && getStatusBadge(status.status)}
          </div>
        </CardHeader>
        <CardContent>
          {statusLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : status ? (
            <div className="space-y-4">
              {/* Stats Grid */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{status.directories_watched}</p>
                  <p className="text-xs text-muted-foreground">Directories</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{status.events_queued}</p>
                  <p className="text-xs text-muted-foreground">Queued</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{status.events_processed}</p>
                  <p className="text-xs text-muted-foreground">Processed</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold text-red-600">{status.events_failed}</p>
                  <p className="text-xs text-muted-foreground">Failed</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{formatUptime(status.uptime_seconds)}</p>
                  <p className="text-xs text-muted-foreground">Uptime</p>
                </div>
              </div>

              {/* Controls */}
              <div className="flex items-center gap-2">
                {status.status === "stopped" ? (
                  <Button onClick={() => startWatcher.mutate()} disabled={startWatcher.isPending}>
                    {startWatcher.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                    Start
                  </Button>
                ) : status.status === "running" ? (
                  <>
                    <Button variant="outline" onClick={() => pauseWatcher.mutate()} disabled={pauseWatcher.isPending}>
                      {pauseWatcher.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Pause className="h-4 w-4 mr-2" />}
                      Pause
                    </Button>
                    <Button variant="destructive" onClick={() => stopWatcher.mutate()} disabled={stopWatcher.isPending}>
                      {stopWatcher.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Square className="h-4 w-4 mr-2" />}
                      Stop
                    </Button>
                  </>
                ) : status.status === "paused" ? (
                  <>
                    <Button onClick={() => resumeWatcher.mutate()} disabled={resumeWatcher.isPending}>
                      {resumeWatcher.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                      Resume
                    </Button>
                    <Button variant="destructive" onClick={() => stopWatcher.mutate()} disabled={stopWatcher.isPending}>
                      {stopWatcher.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Square className="h-4 w-4 mr-2" />}
                      Stop
                    </Button>
                  </>
                ) : null}
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {/* Watched Directories */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Folder className="h-5 w-5" />
                Watched Directories
              </CardTitle>
              <CardDescription>Directories being monitored for new files</CardDescription>
            </div>
            <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
              <DialogTrigger asChild>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Directory
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Watch Directory</DialogTitle>
                  <DialogDescription>
                    Add a new directory to monitor for file changes.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label>Directory Path</Label>
                    <Input
                      placeholder="/path/to/watch"
                      value={newDir.path}
                      onChange={(e) => setNewDir({ ...newDir, path: e.target.value })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>Watch Subdirectories</Label>
                    <Switch
                      checked={newDir.recursive}
                      onCheckedChange={(c) => setNewDir({ ...newDir, recursive: c })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>Auto-process Files</Label>
                    <Switch
                      checked={newDir.auto_process}
                      onCheckedChange={(c) => setNewDir({ ...newDir, auto_process: c })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Access Tier</Label>
                    <Select
                      value={String(newDir.access_tier)}
                      onValueChange={(v) => setNewDir({ ...newDir, access_tier: parseInt(v) })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">1 - Public</SelectItem>
                        <SelectItem value="2">2 - Internal</SelectItem>
                        <SelectItem value="3">3 - Confidential</SelectItem>
                        <SelectItem value="4">4 - Restricted</SelectItem>
                        <SelectItem value="5">5 - Top Secret</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Collection (optional)</Label>
                    <Input
                      placeholder="Default collection"
                      value={newDir.collection}
                      onChange={(e) => setNewDir({ ...newDir, collection: e.target.value })}
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setAddDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={() => addDirectory.mutate(newDir)}
                    disabled={!newDir.path || addDirectory.isPending}
                  >
                    {addDirectory.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
                    Add Directory
                  </Button>
                </DialogFooter>
                {addDirectory.error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{addDirectory.error.message}</AlertDescription>
                  </Alert>
                )}
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          {dirsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : directories && directories.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Path</TableHead>
                  <TableHead>Options</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {directories.map((dir) => (
                  <TableRow key={dir.id}>
                    <TableCell className="font-mono text-sm">{dir.path}</TableCell>
                    <TableCell>
                      <div className="flex gap-1 flex-wrap">
                        {dir.recursive && <Badge variant="outline">Recursive</Badge>}
                        {dir.auto_process && <Badge variant="outline">Auto-process</Badge>}
                        <Badge variant="secondary">Tier {dir.access_tier}</Badge>
                      </div>
                    </TableCell>
                    <TableCell>
                      {dir.enabled ? (
                        <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                          Enabled
                        </Badge>
                      ) : (
                        <Badge variant="secondary">Disabled</Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => scanDirectory.mutate(dir.id)}
                          disabled={scanDirectory.isPending}
                          title="Scan for existing files"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleDirectory.mutate(dir.id)}
                          disabled={toggleDirectory.isPending}
                          title={dir.enabled ? "Disable" : "Enable"}
                        >
                          {dir.enabled ? <PowerOff className="h-4 w-4" /> : <Power className="h-4 w-4" />}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeDirectory.mutate(dir.id)}
                          disabled={removeDirectory.isPending}
                          title="Remove"
                        >
                          <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Folder className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No directories being watched</p>
              <p className="text-sm">Click "Add Directory" to start monitoring</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* File Events */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Recent Events
              </CardTitle>
              <CardDescription>File changes detected by the watcher</CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => clearEvents.mutate()}
              disabled={clearEvents.isPending}
            >
              {clearEvents.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Trash2 className="h-4 w-4 mr-2" />}
              Clear Processed
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {eventsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : events && events.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>File</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Time</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {events.map((event) => (
                  <TableRow key={event.id}>
                    <TableCell className="font-medium">{event.file_name}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{event.event_type}</Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">{formatFileSize(event.file_size)}</TableCell>
                    <TableCell className="text-muted-foreground">
                      {new Date(event.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      {event.processing_error ? (
                        <Badge variant="destructive">Failed</Badge>
                      ) : event.processed ? (
                        <Badge className="bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                          Processed
                        </Badge>
                      ) : (
                        <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                          Pending
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {event.processing_error && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => retryEvent.mutate(event.id)}
                          disabled={retryEvent.isPending}
                          title="Retry"
                        >
                          <RotateCcw className="h-4 w-4" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Clock className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No events yet</p>
              <p className="text-sm">File changes will appear here</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
