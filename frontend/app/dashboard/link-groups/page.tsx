"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import {
  Plus,
  Folder,
  Link as LinkIcon,
  Trash2,
  Edit2,
  Globe,
  ExternalLink,
  Search,
  Loader2,
  Check,
  X,
  FileText,
  Clock,
  Play,
  RefreshCw,
  Eye,
  FolderPlus,
  ChevronRight,
  ChevronDown,
  MoreVertical,
  Copy,
  Download,
  Database,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
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
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface LinkGroup {
  id: string;
  name: string;
  description: string | null;
  color: string | null;
  icon: string | null;
  is_shared: boolean;
  sort_order: number;
  link_count: number;
  created_at: string;
  updated_at: string;
}

interface SavedLink {
  id: string;
  url: string;
  title: string | null;
  description: string | null;
  favicon_url: string | null;
  group_id: string;
  group_name: string | null;
  tags: string[] | null;
  auto_scrape: boolean;
  scrape_frequency: string | null;
  last_scraped_at: string | null;
  last_scrape_status: string | null;
  scrape_count: number;
  cached_word_count: number | null;
  cached_content_preview: string | null;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

interface ScrapedContent {
  id: string;
  saved_link_id: string;
  url: string;
  title: string | null;
  content: string | null;
  word_count: number;
  scraped_at: string;
  status: string;
  error_message: string | null;
  links_found: number;
  images_found: number;
  metadata: Record<string, any> | null;
  indexed_to_rag: boolean;
}

// Color presets for groups
const GROUP_COLORS = [
  { name: "Blue", value: "#3b82f6" },
  { name: "Green", value: "#22c55e" },
  { name: "Purple", value: "#a855f7" },
  { name: "Orange", value: "#f97316" },
  { name: "Pink", value: "#ec4899" },
  { name: "Teal", value: "#14b8a6" },
  { name: "Red", value: "#ef4444" },
  { name: "Yellow", value: "#eab308" },
];

export default function LinkGroupsPage() {
  const { data: session, status } = useSession();
  const isAuthenticated = status === "authenticated";

  // State
  const [groups, setGroups] = useState<LinkGroup[]>([]);
  const [links, setLinks] = useState<SavedLink[]>([]);
  const [selectedGroup, setSelectedGroup] = useState<LinkGroup | null>(null);
  const [selectedLink, setSelectedLink] = useState<SavedLink | null>(null);
  const [scrapeHistory, setScrapeHistory] = useState<ScrapedContent[]>([]);

  // Loading states
  const [groupsLoading, setGroupsLoading] = useState(true);
  const [linksLoading, setLinksLoading] = useState(false);
  const [scraping, setScraping] = useState<string | null>(null);
  const [groupScraping, setGroupScraping] = useState<string | null>(null);

  // Dialog states
  const [createGroupOpen, setCreateGroupOpen] = useState(false);
  const [editGroupOpen, setEditGroupOpen] = useState(false);
  const [addLinkOpen, setAddLinkOpen] = useState(false);
  const [bulkAddOpen, setBulkAddOpen] = useState(false);
  const [contentViewerOpen, setContentViewerOpen] = useState(false);
  const [viewingContent, setViewingContent] = useState<ScrapedContent | null>(null);

  // Form states
  const [newGroup, setNewGroup] = useState({
    name: "",
    description: "",
    color: "#3b82f6",
    is_shared: false,
  });
  const [newLink, setNewLink] = useState({
    url: "",
    title: "",
    description: "",
    auto_scrape: false,
    scrape_frequency: "",
  });
  const [bulkUrls, setBulkUrls] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  // Expanded groups for collapsible view
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  // Auth header
  const getAuthHeader = () => {
    const token = (session as any)?.accessToken;
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  // Fetch groups
  const fetchGroups = async () => {
    if (!isAuthenticated) return;
    setGroupsLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/groups`, {
        headers: { ...getAuthHeader() },
      });
      if (res.ok) {
        const data = await res.json();
        setGroups(data.groups || []);
      }
    } catch (error) {
      console.error("Failed to fetch groups:", error);
    } finally {
      setGroupsLoading(false);
    }
  };

  // Fetch links for a group
  const fetchLinks = async (groupId?: string) => {
    if (!isAuthenticated) return;
    setLinksLoading(true);
    try {
      const url = groupId
        ? `${API_BASE_URL}/link-groups/links?group_id=${groupId}`
        : `${API_BASE_URL}/link-groups/links`;
      const res = await fetch(url, {
        headers: { ...getAuthHeader() },
      });
      if (res.ok) {
        const data = await res.json();
        setLinks(data.links || []);
      }
    } catch (error) {
      console.error("Failed to fetch links:", error);
    } finally {
      setLinksLoading(false);
    }
  };

  // Fetch scrape history for a link
  const fetchScrapeHistory = async (linkId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/links/${linkId}/history`, {
        headers: { ...getAuthHeader() },
      });
      if (res.ok) {
        const data = await res.json();
        setScrapeHistory(data.history || []);
      }
    } catch (error) {
      console.error("Failed to fetch scrape history:", error);
    }
  };

  // Create group
  const handleCreateGroup = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/groups`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader(),
        },
        body: JSON.stringify(newGroup),
      });
      if (res.ok) {
        setCreateGroupOpen(false);
        setNewGroup({ name: "", description: "", color: "#3b82f6", is_shared: false });
        fetchGroups();
      }
    } catch (error) {
      console.error("Failed to create group:", error);
    }
  };

  // Update group
  const handleUpdateGroup = async () => {
    if (!selectedGroup) return;
    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/groups/${selectedGroup.id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader(),
        },
        body: JSON.stringify(newGroup),
      });
      if (res.ok) {
        setEditGroupOpen(false);
        setSelectedGroup(null);
        fetchGroups();
      }
    } catch (error) {
      console.error("Failed to update group:", error);
    }
  };

  // Delete group
  const handleDeleteGroup = async (groupId: string) => {
    if (!confirm("Delete this group and all its links?")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/groups/${groupId}`, {
        method: "DELETE",
        headers: { ...getAuthHeader() },
      });
      if (res.ok) {
        setSelectedGroup(null);
        fetchGroups();
        fetchLinks();
      }
    } catch (error) {
      console.error("Failed to delete group:", error);
    }
  };

  // Add link
  const handleAddLink = async () => {
    if (!selectedGroup) return;
    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/links`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader(),
        },
        body: JSON.stringify({
          ...newLink,
          group_id: selectedGroup.id,
        }),
      });
      if (res.ok) {
        setAddLinkOpen(false);
        setNewLink({ url: "", title: "", description: "", auto_scrape: false, scrape_frequency: "" });
        fetchLinks(selectedGroup.id);
        fetchGroups(); // Update link count
      }
    } catch (error) {
      console.error("Failed to add link:", error);
    }
  };

  // Bulk add links
  const handleBulkAddLinks = async () => {
    if (!selectedGroup) return;
    const urls = bulkUrls
      .split("\n")
      .map((u) => u.trim())
      .filter((u) => u.length > 0);
    if (urls.length === 0) return;

    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/links/bulk`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeader(),
        },
        body: JSON.stringify({
          urls,
          group_id: selectedGroup.id,
          auto_scrape: false,
        }),
      });
      if (res.ok) {
        setBulkAddOpen(false);
        setBulkUrls("");
        fetchLinks(selectedGroup.id);
        fetchGroups();
      }
    } catch (error) {
      console.error("Failed to add links:", error);
    }
  };

  // Delete link
  const handleDeleteLink = async (linkId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/link-groups/links/${linkId}`, {
        method: "DELETE",
        headers: { ...getAuthHeader() },
      });
      if (res.ok) {
        fetchLinks(selectedGroup?.id);
        fetchGroups();
      }
    } catch (error) {
      console.error("Failed to delete link:", error);
    }
  };

  // Scrape single link
  const handleScrapeLink = async (linkId: string, indexToRag: boolean = false) => {
    setScraping(linkId);
    try {
      const res = await fetch(
        `${API_BASE_URL}/link-groups/links/${linkId}/scrape?save_history=true&index_to_rag=${indexToRag}`,
        {
          method: "POST",
          headers: { ...getAuthHeader() },
        }
      );
      if (res.ok) {
        const data = await res.json();
        // Update the link in the list
        setLinks((prev) =>
          prev.map((l) =>
            l.id === linkId
              ? {
                  ...l,
                  last_scraped_at: data.scraped_at,
                  last_scrape_status: data.success ? "success" : "failed",
                  cached_word_count: data.word_count,
                  cached_content_preview: data.content?.slice(0, 500),
                  scrape_count: l.scrape_count + 1,
                }
              : l
          )
        );
        // If we have a selected link, update history
        if (selectedLink?.id === linkId) {
          fetchScrapeHistory(linkId);
        }
      }
    } catch (error) {
      console.error("Failed to scrape link:", error);
    } finally {
      setScraping(null);
    }
  };

  // Scrape all links in a group
  const handleScrapeGroup = async (groupId: string) => {
    setGroupScraping(groupId);
    try {
      const res = await fetch(
        `${API_BASE_URL}/link-groups/groups/${groupId}/scrape?storage_mode=permanent`,
        {
          method: "POST",
          headers: { ...getAuthHeader() },
        }
      );
      if (res.ok) {
        const data = await res.json();
        alert(`Started scraping ${data.urls_count} links from "${data.group_name}". Job ID: ${data.job_id}`);
      }
    } catch (error) {
      console.error("Failed to scrape group:", error);
    } finally {
      setGroupScraping(null);
    }
  };

  // View scraped content
  const handleViewContent = (content: ScrapedContent) => {
    setViewingContent(content);
    setContentViewerOpen(true);
  };

  // Toggle group expansion
  const toggleGroupExpansion = (groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  };

  // Initial fetch
  useEffect(() => {
    if (isAuthenticated) {
      fetchGroups();
      fetchLinks();
    }
  }, [isAuthenticated]);

  // Fetch links when group is selected
  useEffect(() => {
    if (selectedGroup) {
      fetchLinks(selectedGroup.id);
    }
  }, [selectedGroup]);

  // Fetch history when link is selected
  useEffect(() => {
    if (selectedLink) {
      fetchScrapeHistory(selectedLink.id);
    }
  }, [selectedLink]);

  // Filter links by search
  const filteredLinks = links.filter(
    (link) =>
      link.url.toLowerCase().includes(searchQuery.toLowerCase()) ||
      link.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      link.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Links grouped by group_id for the all links view
  const linksByGroup = links.reduce((acc, link) => {
    const groupId = link.group_id;
    if (!acc[groupId]) {
      acc[groupId] = [];
    }
    acc[groupId].push(link);
    return acc;
  }, {} as Record<string, SavedLink[]>);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Link Groups</h1>
          <p className="text-muted-foreground">
            Organize and manage your saved links for web scraping
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={() => fetchGroups()} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Dialog open={createGroupOpen} onOpenChange={setCreateGroupOpen}>
            <DialogTrigger asChild>
              <Button>
                <FolderPlus className="h-4 w-4 mr-2" />
                New Group
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Link Group</DialogTitle>
                <DialogDescription>
                  Create a new group to organize your links
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="group-name">Name</Label>
                  <Input
                    id="group-name"
                    placeholder="e.g., Marketing Resources"
                    value={newGroup.name}
                    onChange={(e) => setNewGroup({ ...newGroup, name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="group-desc">Description</Label>
                  <Textarea
                    id="group-desc"
                    placeholder="Optional description..."
                    value={newGroup.description}
                    onChange={(e) => setNewGroup({ ...newGroup, description: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Color</Label>
                  <div className="flex flex-wrap gap-2">
                    {GROUP_COLORS.map((color) => (
                      <button
                        key={color.value}
                        className={`w-8 h-8 rounded-full border-2 transition-all ${
                          newGroup.color === color.value
                            ? "border-foreground scale-110"
                            : "border-transparent"
                        }`}
                        style={{ backgroundColor: color.value }}
                        onClick={() => setNewGroup({ ...newGroup, color: color.value })}
                        title={color.name}
                      />
                    ))}
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="group-shared"
                    checked={newGroup.is_shared}
                    onCheckedChange={(checked) => setNewGroup({ ...newGroup, is_shared: checked })}
                  />
                  <Label htmlFor="group-shared">Share with organization</Label>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setCreateGroupOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateGroup} disabled={!newGroup.name}>
                  Create Group
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar - Groups List */}
        <div className="w-64 border-r flex flex-col">
          <div className="p-4 border-b">
            <Button
              variant={selectedGroup === null ? "secondary" : "ghost"}
              className="w-full justify-start"
              onClick={() => {
                setSelectedGroup(null);
                fetchLinks();
              }}
            >
              <Globe className="h-4 w-4 mr-2" />
              All Links
              <Badge variant="secondary" className="ml-auto">
                {links.length}
              </Badge>
            </Button>
          </div>
          <ScrollArea className="flex-1">
            <div className="p-2 space-y-1">
              {groupsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : groups.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  <Folder className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No groups yet</p>
                  <p>Create one to get started</p>
                </div>
              ) : (
                groups.map((group) => (
                  <div
                    key={group.id}
                    className={`group flex items-center gap-2 rounded-lg px-3 py-2 cursor-pointer transition-colors ${
                      selectedGroup?.id === group.id
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-muted"
                    }`}
                    onClick={() => setSelectedGroup(group)}
                  >
                    <div
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: group.color || "#3b82f6" }}
                    />
                    <span className="flex-1 truncate text-sm">{group.name}</span>
                    <Badge variant="secondary" className="text-xs">
                      {group.link_count}
                    </Badge>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 opacity-0 group-hover:opacity-100"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedGroup(group);
                            setNewGroup({
                              name: group.name,
                              description: group.description || "",
                              color: group.color || "#3b82f6",
                              is_shared: group.is_shared,
                            });
                            setEditGroupOpen(true);
                          }}
                        >
                          <Edit2 className="h-4 w-4 mr-2" />
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            handleScrapeGroup(group.id);
                          }}
                          disabled={groupScraping === group.id}
                        >
                          {groupScraping === group.id ? (
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          ) : (
                            <Play className="h-4 w-4 mr-2" />
                          )}
                          Scrape All
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          className="text-destructive"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteGroup(group.id);
                          }}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Links Header */}
          <div className="p-4 border-b flex items-center justify-between gap-4">
            <div className="flex-1 flex items-center gap-2">
              <Search className="h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search links..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="max-w-sm"
              />
            </div>
            {selectedGroup && (
              <div className="flex items-center gap-2">
                <Dialog open={bulkAddOpen} onOpenChange={setBulkAddOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Copy className="h-4 w-4 mr-2" />
                      Bulk Add
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Bulk Add Links</DialogTitle>
                      <DialogDescription>
                        Paste multiple URLs, one per line
                      </DialogDescription>
                    </DialogHeader>
                    <Textarea
                      placeholder="https://example.com&#10;https://another-site.com&#10;..."
                      value={bulkUrls}
                      onChange={(e) => setBulkUrls(e.target.value)}
                      className="min-h-[200px] font-mono text-sm"
                    />
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setBulkAddOpen(false)}>
                        Cancel
                      </Button>
                      <Button onClick={handleBulkAddLinks}>
                        Add {bulkUrls.split("\n").filter((u) => u.trim()).length} Links
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
                <Dialog open={addLinkOpen} onOpenChange={setAddLinkOpen}>
                  <DialogTrigger asChild>
                    <Button size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Add Link
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Add Link</DialogTitle>
                      <DialogDescription>
                        Add a new link to "{selectedGroup.name}"
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <div className="space-y-2">
                        <Label htmlFor="link-url">URL</Label>
                        <Input
                          id="link-url"
                          placeholder="https://example.com"
                          value={newLink.url}
                          onChange={(e) => setNewLink({ ...newLink, url: e.target.value })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="link-title">Title (optional)</Label>
                        <Input
                          id="link-title"
                          placeholder="Page title"
                          value={newLink.title}
                          onChange={(e) => setNewLink({ ...newLink, title: e.target.value })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="link-desc">Description (optional)</Label>
                        <Textarea
                          id="link-desc"
                          placeholder="Optional description..."
                          value={newLink.description}
                          onChange={(e) => setNewLink({ ...newLink, description: e.target.value })}
                        />
                      </div>
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="link-auto"
                          checked={newLink.auto_scrape}
                          onCheckedChange={(checked) =>
                            setNewLink({ ...newLink, auto_scrape: checked })
                          }
                        />
                        <Label htmlFor="link-auto">Include in scheduled scraping</Label>
                      </div>
                      {newLink.auto_scrape && (
                        <div className="space-y-2">
                          <Label>Scrape Frequency</Label>
                          <Select
                            value={newLink.scrape_frequency}
                            onValueChange={(value) =>
                              setNewLink({ ...newLink, scrape_frequency: value })
                            }
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select frequency" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="daily">Daily</SelectItem>
                              <SelectItem value="weekly">Weekly</SelectItem>
                              <SelectItem value="monthly">Monthly</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                    </div>
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setAddLinkOpen(false)}>
                        Cancel
                      </Button>
                      <Button onClick={handleAddLink} disabled={!newLink.url}>
                        Add Link
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            )}
          </div>

          {/* Links List */}
          <ScrollArea className="flex-1">
            <div className="p-4">
              {linksLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : filteredLinks.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <LinkIcon className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p className="text-lg font-medium">No links yet</p>
                  <p className="text-sm">
                    {selectedGroup
                      ? `Add links to "${selectedGroup.name}" to get started`
                      : "Create a group and add links to get started"}
                  </p>
                </div>
              ) : selectedGroup ? (
                // Single group view - flat list
                <div className="space-y-2">
                  {filteredLinks.map((link) => (
                    <LinkCard
                      key={link.id}
                      link={link}
                      scraping={scraping === link.id}
                      onScrape={() => handleScrapeLink(link.id)}
                      onScrapeAndIndex={() => handleScrapeLink(link.id, true)}
                      onDelete={() => handleDeleteLink(link.id)}
                      onSelect={() => setSelectedLink(link)}
                      isSelected={selectedLink?.id === link.id}
                    />
                  ))}
                </div>
              ) : (
                // All links view - grouped by group
                <div className="space-y-4">
                  {groups.map((group) => {
                    const groupLinks = linksByGroup[group.id] || [];
                    if (groupLinks.length === 0) return null;

                    const isExpanded = expandedGroups.has(group.id);
                    const filteredGroupLinks = groupLinks.filter(
                      (link) =>
                        link.url.toLowerCase().includes(searchQuery.toLowerCase()) ||
                        link.title?.toLowerCase().includes(searchQuery.toLowerCase())
                    );

                    if (filteredGroupLinks.length === 0) return null;

                    return (
                      <Collapsible
                        key={group.id}
                        open={isExpanded}
                        onOpenChange={() => toggleGroupExpansion(group.id)}
                      >
                        <CollapsibleTrigger asChild>
                          <div className="flex items-center gap-2 p-2 rounded-lg hover:bg-muted cursor-pointer">
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: group.color || "#3b82f6" }}
                            />
                            <span className="font-medium">{group.name}</span>
                            <Badge variant="secondary" className="ml-auto">
                              {filteredGroupLinks.length} links
                            </Badge>
                          </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <div className="pl-8 space-y-2 mt-2">
                            {filteredGroupLinks.map((link) => (
                              <LinkCard
                                key={link.id}
                                link={link}
                                scraping={scraping === link.id}
                                onScrape={() => handleScrapeLink(link.id)}
                                onScrapeAndIndex={() => handleScrapeLink(link.id, true)}
                                onDelete={() => handleDeleteLink(link.id)}
                                onSelect={() => setSelectedLink(link)}
                                isSelected={selectedLink?.id === link.id}
                              />
                            ))}
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                    );
                  })}
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Right Panel - Link Details & History */}
        {selectedLink && (
          <div className="w-96 border-l flex flex-col">
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="font-semibold truncate">{selectedLink.title || "Link Details"}</h3>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSelectedLink(null)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-4 space-y-4">
                {/* Link Info */}
                <div className="space-y-2">
                  <Label className="text-muted-foreground">URL</Label>
                  <a
                    href={selectedLink.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-primary hover:underline flex items-center gap-1 break-all"
                  >
                    {selectedLink.url}
                    <ExternalLink className="h-3 w-3 flex-shrink-0" />
                  </a>
                </div>

                {selectedLink.description && (
                  <div className="space-y-2">
                    <Label className="text-muted-foreground">Description</Label>
                    <p className="text-sm">{selectedLink.description}</p>
                  </div>
                )}

                {/* Stats */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <Label className="text-muted-foreground">Scrape Count</Label>
                    <p className="font-medium">{selectedLink.scrape_count}</p>
                  </div>
                  <div>
                    <Label className="text-muted-foreground">Last Status</Label>
                    <p className="font-medium">
                      {selectedLink.last_scrape_status === "success" ? (
                        <span className="text-green-600 flex items-center gap-1">
                          <Check className="h-3 w-3" /> Success
                        </span>
                      ) : selectedLink.last_scrape_status === "failed" ? (
                        <span className="text-red-600 flex items-center gap-1">
                          <X className="h-3 w-3" /> Failed
                        </span>
                      ) : (
                        <span className="text-muted-foreground">Never scraped</span>
                      )}
                    </p>
                  </div>
                  {selectedLink.cached_word_count && (
                    <div>
                      <Label className="text-muted-foreground">Word Count</Label>
                      <p className="font-medium">{selectedLink.cached_word_count.toLocaleString()}</p>
                    </div>
                  )}
                  {selectedLink.last_scraped_at && (
                    <div>
                      <Label className="text-muted-foreground">Last Scraped</Label>
                      <p className="font-medium">
                        {new Date(selectedLink.last_scraped_at).toLocaleDateString()}
                      </p>
                    </div>
                  )}
                </div>

                {/* Preview */}
                {selectedLink.cached_content_preview && (
                  <div className="space-y-2">
                    <Label className="text-muted-foreground">Content Preview</Label>
                    <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-lg">
                      {selectedLink.cached_content_preview}...
                    </p>
                  </div>
                )}

                {/* Scrape History */}
                <div className="space-y-2">
                  <Label className="text-muted-foreground">Scrape History</Label>
                  {scrapeHistory.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No scrape history yet</p>
                  ) : (
                    <div className="space-y-2">
                      {scrapeHistory.slice(0, 10).map((history) => (
                        <div
                          key={history.id}
                          className="flex items-center justify-between p-2 rounded-lg bg-muted/50 text-sm cursor-pointer hover:bg-muted"
                          onClick={() => handleViewContent(history)}
                        >
                          <div className="flex items-center gap-2">
                            {history.status === "success" ? (
                              <Check className="h-3 w-3 text-green-600" />
                            ) : (
                              <X className="h-3 w-3 text-red-600" />
                            )}
                            <span>
                              {new Date(history.scraped_at).toLocaleDateString()}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-muted-foreground">
                            {history.word_count > 0 && (
                              <span>{history.word_count} words</span>
                            )}
                            <Eye className="h-3 w-3" />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </ScrollArea>
          </div>
        )}
      </div>

      {/* Edit Group Dialog */}
      <Dialog open={editGroupOpen} onOpenChange={setEditGroupOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Group</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-name">Name</Label>
              <Input
                id="edit-name"
                value={newGroup.name}
                onChange={(e) => setNewGroup({ ...newGroup, name: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-desc">Description</Label>
              <Textarea
                id="edit-desc"
                value={newGroup.description}
                onChange={(e) => setNewGroup({ ...newGroup, description: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label>Color</Label>
              <div className="flex flex-wrap gap-2">
                {GROUP_COLORS.map((color) => (
                  <button
                    key={color.value}
                    className={`w-8 h-8 rounded-full border-2 transition-all ${
                      newGroup.color === color.value
                        ? "border-foreground scale-110"
                        : "border-transparent"
                    }`}
                    style={{ backgroundColor: color.value }}
                    onClick={() => setNewGroup({ ...newGroup, color: color.value })}
                  />
                ))}
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="edit-shared"
                checked={newGroup.is_shared}
                onCheckedChange={(checked) => setNewGroup({ ...newGroup, is_shared: checked })}
              />
              <Label htmlFor="edit-shared">Share with organization</Label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditGroupOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleUpdateGroup}>Save Changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Content Viewer Sheet */}
      <Sheet open={contentViewerOpen} onOpenChange={setContentViewerOpen}>
        <SheetContent side="right" className="w-[600px] sm:max-w-[600px]">
          <SheetHeader>
            <SheetTitle>{viewingContent?.title || "Scraped Content"}</SheetTitle>
            <SheetDescription>
              Scraped on {viewingContent && new Date(viewingContent.scraped_at).toLocaleString()}
            </SheetDescription>
          </SheetHeader>
          <div className="mt-4 space-y-4">
            {/* Stats */}
            <div className="flex flex-wrap gap-4 text-sm">
              <div className="flex items-center gap-1">
                <FileText className="h-4 w-4 text-muted-foreground" />
                {viewingContent?.word_count.toLocaleString()} words
              </div>
              <div className="flex items-center gap-1">
                <LinkIcon className="h-4 w-4 text-muted-foreground" />
                {viewingContent?.links_found} links
              </div>
              {viewingContent?.indexed_to_rag && (
                <div className="flex items-center gap-1 text-green-600">
                  <Database className="h-4 w-4" />
                  Indexed
                </div>
              )}
            </div>

            {/* Content */}
            <ScrollArea className="h-[calc(100vh-200px)] rounded-lg border">
              <div className="p-4 whitespace-pre-wrap text-sm">
                {viewingContent?.content || "No content available"}
              </div>
            </ScrollArea>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}

// Link Card Component
interface LinkCardProps {
  link: SavedLink;
  scraping: boolean;
  onScrape: () => void;
  onScrapeAndIndex: () => void;
  onDelete: () => void;
  onSelect: () => void;
  isSelected: boolean;
}

function LinkCard({
  link,
  scraping,
  onScrape,
  onScrapeAndIndex,
  onDelete,
  onSelect,
  isSelected,
}: LinkCardProps) {
  return (
    <div
      className={`group flex items-start gap-3 p-3 rounded-lg border transition-colors cursor-pointer ${
        isSelected ? "border-primary bg-primary/5" : "hover:bg-muted/50"
      }`}
      onClick={onSelect}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h4 className="font-medium truncate text-sm">
            {link.title || new URL(link.url).hostname}
          </h4>
          {link.last_scrape_status === "success" && (
            <Badge variant="secondary" className="text-xs">
              <Check className="h-3 w-3 mr-1" />
              Scraped
            </Badge>
          )}
          {link.auto_scrape && (
            <Badge variant="outline" className="text-xs">
              Auto
            </Badge>
          )}
        </div>
        <a
          href={link.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-muted-foreground hover:text-primary truncate block"
          onClick={(e) => e.stopPropagation()}
        >
          {link.url}
        </a>
        {link.cached_content_preview && (
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
            {link.cached_content_preview}
          </p>
        )}
      </div>
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={(e) => e.stopPropagation()}
            >
              {scraping ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <MoreVertical className="h-4 w-4" />
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onScrape(); }} disabled={scraping}>
              <Play className="h-4 w-4 mr-2" />
              Scrape Now
            </DropdownMenuItem>
            <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onScrapeAndIndex(); }} disabled={scraping}>
              <Database className="h-4 w-4 mr-2" />
              Scrape & Index
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={(e) => {
                e.stopPropagation();
                window.open(link.url, "_blank");
              }}
            >
              <ExternalLink className="h-4 w-4 mr-2" />
              Open URL
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-destructive"
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}
