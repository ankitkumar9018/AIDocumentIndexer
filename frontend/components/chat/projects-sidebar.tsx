"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import {
  FolderOpen,
  MessageSquare,
  Plus,
  ChevronRight,
  ChevronDown,
  MoreVertical,
  Edit,
  Trash2,
  FolderInput,
  FolderMinus,
  Search,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Project {
  id: string;
  name: string;
  sessionIds: string[];
  createdAt: string;
  collapsed: boolean;
}

interface ProjectsSidebarProps {
  sessions: Array<{
    id: string;
    title: string;
    created_at: string;
    message_count?: number;
  }>;
  activeSessionId?: string | null;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession?: (sessionId: string) => void;
}

// ---------------------------------------------------------------------------
// LocalStorage helpers
// ---------------------------------------------------------------------------

const PROJECTS_STORAGE_KEY = "chat-projects";

function loadProjects(): Project[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(PROJECTS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as Project[];
  } catch {
    return [];
  }
}

function saveProjects(projects: Project[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(projects));
  } catch {
    // Storage full or unavailable -- silently ignore
  }
}

function generateId(): string {
  return `proj_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ProjectsSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
}: ProjectsSidebarProps) {
  // ---- State ----
  const [projects, setProjects] = useState<Project[]>([]);
  const [hydrated, setHydrated] = useState(false);

  // Dialog state
  const [showNewDialog, setShowNewDialog] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [renameProjectId, setRenameProjectId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [confirmDeleteProjectId, setConfirmDeleteProjectId] = useState<string | null>(null);

  // Search
  const [searchQuery, setSearchQuery] = useState("");

  // ---- Hydrate from localStorage on mount ----
  useEffect(() => {
    setProjects(loadProjects());
    setHydrated(true);
  }, []);

  // ---- Persist to localStorage on change ----
  useEffect(() => {
    if (hydrated) {
      saveProjects(projects);
    }
  }, [projects, hydrated]);

  // ---- Derived data ----

  // Build a set of session IDs that belong to any project
  const assignedSessionIds = useMemo(() => {
    const ids = new Set<string>();
    for (const project of projects) {
      for (const sid of project.sessionIds) {
        ids.add(sid);
      }
    }
    return ids;
  }, [projects]);

  // Sessions that are not assigned to any project
  const ungroupedSessions = useMemo(() => {
    return sessions.filter((s) => !assignedSessionIds.has(s.id));
  }, [sessions, assignedSessionIds]);

  // Build a lookup map for sessions by id
  const sessionMap = useMemo(() => {
    const map = new Map<string, (typeof sessions)[number]>();
    for (const s of sessions) {
      map.set(s.id, s);
    }
    return map;
  }, [sessions]);

  // Filtered results for search
  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return null; // null means "no search active"
    const q = searchQuery.toLowerCase();
    return sessions.filter((s) => s.title?.toLowerCase().includes(q));
  }, [sessions, searchQuery]);

  // ---- Project CRUD ----

  const handleCreateProject = useCallback(() => {
    const name = newProjectName.trim();
    if (!name) return;
    const newProject: Project = {
      id: generateId(),
      name,
      sessionIds: [],
      createdAt: new Date().toISOString(),
      collapsed: false,
    };
    setProjects((prev) => [...prev, newProject]);
    setNewProjectName("");
    setShowNewDialog(false);
  }, [newProjectName]);

  const handleRenameProject = useCallback(() => {
    const name = renameValue.trim();
    if (!name || !renameProjectId) return;
    setProjects((prev) =>
      prev.map((p) => (p.id === renameProjectId ? { ...p, name } : p))
    );
    setRenameProjectId(null);
    setRenameValue("");
  }, [renameProjectId, renameValue]);

  const handleDeleteProject = useCallback(
    (projectId: string) => {
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
      setConfirmDeleteProjectId(null);
    },
    []
  );

  const toggleProjectCollapsed = useCallback((projectId: string) => {
    setProjects((prev) =>
      prev.map((p) =>
        p.id === projectId ? { ...p, collapsed: !p.collapsed } : p
      )
    );
  }, []);

  // ---- Session <-> Project assignment ----

  const moveSessionToProject = useCallback(
    (sessionId: string, projectId: string) => {
      setProjects((prev) =>
        prev.map((p) => {
          // Remove from any project that currently has it
          const filtered = p.sessionIds.filter((sid) => sid !== sessionId);
          // Add to target project
          if (p.id === projectId) {
            return { ...p, sessionIds: [...filtered, sessionId] };
          }
          return { ...p, sessionIds: filtered };
        })
      );
    },
    []
  );

  const removeSessionFromProject = useCallback((sessionId: string) => {
    setProjects((prev) =>
      prev.map((p) => ({
        ...p,
        sessionIds: p.sessionIds.filter((sid) => sid !== sessionId),
      }))
    );
  }, []);

  // ---- Rendering helpers ----

  const renderSessionItem = (
    session: { id: string; title: string; created_at: string; message_count?: number },
    options?: { inProject?: string }
  ) => {
    const isActive = session.id === activeSessionId;

    return (
      <div
        key={session.id}
        className={cn(
          "group flex items-center gap-2 rounded-md px-2 py-1.5 text-sm cursor-pointer transition-colors",
          isActive
            ? "bg-primary/10 text-primary font-medium"
            : "hover:bg-muted text-foreground"
        )}
        onClick={() => onSelectSession(session.id)}
      >
        <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        <span className="truncate flex-1 min-w-0">
          {session.title || "Untitled chat"}
        </span>

        {/* Session actions dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={(e) => e.stopPropagation()}
            >
              <MoreVertical className="h-3.5 w-3.5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            {/* Move to project submenu */}
            {projects.length > 0 && (
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>
                  <FolderInput className="h-4 w-4 mr-2" />
                  Move to project
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-48">
                  {projects.map((project) => (
                    <DropdownMenuItem
                      key={project.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        moveSessionToProject(session.id, project.id);
                      }}
                    >
                      <FolderOpen className="h-4 w-4 mr-2" />
                      {project.name}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            )}

            {/* Remove from project (only if in a project) */}
            {options?.inProject && (
              <DropdownMenuItem
                onClick={(e) => {
                  e.stopPropagation();
                  removeSessionFromProject(session.id);
                }}
              >
                <FolderMinus className="h-4 w-4 mr-2" />
                Remove from project
              </DropdownMenuItem>
            )}

            {/* Delete session */}
            {onDeleteSession && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive focus:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete chat
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    );
  };

  const renderProjectFolder = (project: Project) => {
    // Resolve sessions that still exist
    const projectSessions = project.sessionIds
      .map((sid) => sessionMap.get(sid))
      .filter(Boolean) as (typeof sessions)[number][];

    const sessionCount = projectSessions.length;

    return (
      <div key={project.id} className="mb-1">
        {/* Project header */}
        <div
          className="group flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm cursor-pointer hover:bg-muted transition-colors"
          onClick={() => toggleProjectCollapsed(project.id)}
        >
          {project.collapsed ? (
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          )}
          <FolderOpen className="h-4 w-4 shrink-0 text-amber-500" />
          <span className="truncate flex-1 min-w-0 font-medium">
            {project.name}
          </span>
          <span className="text-xs text-muted-foreground shrink-0">
            ({sessionCount})
          </span>

          {/* Project actions dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => e.stopPropagation()}
              >
                <MoreVertical className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44">
              <DropdownMenuItem
                onClick={(e) => {
                  e.stopPropagation();
                  setRenameProjectId(project.id);
                  setRenameValue(project.name);
                }}
              >
                <Edit className="h-4 w-4 mr-2" />
                Rename
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  setConfirmDeleteProjectId(project.id);
                }}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete project
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Expanded session list */}
        {!project.collapsed && (
          <div className="ml-4 border-l border-border pl-2 mt-0.5">
            {projectSessions.length === 0 ? (
              <p className="text-xs text-muted-foreground px-2 py-1 italic">
                No chats in this project
              </p>
            ) : (
              projectSessions.map((session) =>
                renderSessionItem(session, { inProject: project.id })
              )
            )}
          </div>
        )}
      </div>
    );
  };

  // ---- Search results mode ----

  if (filteredSessions) {
    return (
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b">
          <h3 className="text-sm font-semibold">Projects</h3>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setShowNewDialog(true)}
            title="New project"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        {/* Search */}
        <div className="px-3 py-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search chats..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 pl-8 text-sm"
            />
          </div>
        </div>

        {/* Search results */}
        <ScrollArea className="flex-1">
          <div className="px-1 py-1">
            {filteredSessions.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-4">
                No chats found
              </p>
            ) : (
              filteredSessions.map((session) => renderSessionItem(session))
            )}
          </div>
        </ScrollArea>

        {/* Dialogs */}
        {renderNewProjectDialog()}
      </div>
    );
  }

  // ---- Dialogs ----

  function renderNewProjectDialog() {
    return (
      <Dialog open={showNewDialog} onOpenChange={setShowNewDialog}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle>New Project</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <Input
              placeholder="Project name"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleCreateProject();
                }
              }}
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateProject}
              disabled={!newProjectName.trim()}
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  function renderRenameDialog() {
    return (
      <Dialog
        open={!!renameProjectId}
        onOpenChange={(open) => {
          if (!open) {
            setRenameProjectId(null);
            setRenameValue("");
          }
        }}
      >
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle>Rename Project</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <Input
              placeholder="Project name"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleRenameProject();
                }
              }}
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setRenameProjectId(null);
                setRenameValue("");
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleRenameProject}
              disabled={!renameValue.trim()}
            >
              Rename
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  function renderDeleteConfirmDialog() {
    const project = projects.find((p) => p.id === confirmDeleteProjectId);
    if (!project) return null;

    return (
      <Dialog
        open={!!confirmDeleteProjectId}
        onOpenChange={(open) => {
          if (!open) setConfirmDeleteProjectId(null);
        }}
      >
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle>Delete Project</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground py-2">
            Delete project &quot;{project.name}&quot;? The chats inside will not
            be deleted -- they will move back to the ungrouped section.
          </p>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmDeleteProjectId(null)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => handleDeleteProject(project.id)}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  // ---- Main render ----

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b">
        <h3 className="text-sm font-semibold">Projects</h3>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setShowNewDialog(true)}
          title="New project"
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>

      {/* Search */}
      <div className="px-3 py-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-8 pl-8 text-sm"
          />
        </div>
      </div>

      {/* Scrollable content */}
      <ScrollArea className="flex-1">
        <div className="px-1 py-1">
          {/* Project folders */}
          {projects.map((project) => renderProjectFolder(project))}

          {/* Ungrouped separator */}
          {projects.length > 0 && ungroupedSessions.length > 0 && (
            <div className="flex items-center gap-2 px-2 pt-3 pb-1">
              <div className="flex-1 h-px bg-border" />
              <span className="text-xs text-muted-foreground shrink-0">
                Ungrouped ({ungroupedSessions.length})
              </span>
              <div className="flex-1 h-px bg-border" />
            </div>
          )}

          {/* Ungrouped sessions */}
          {ungroupedSessions.length > 0 ? (
            ungroupedSessions.map((session) => renderSessionItem(session))
          ) : (
            projects.length === 0 && sessions.length === 0 && (
              <p className="text-xs text-muted-foreground text-center py-4">
                No chat sessions yet
              </p>
            )
          )}

          {/* Empty state when all sessions are in projects */}
          {projects.length > 0 &&
            ungroupedSessions.length === 0 &&
            sessions.length > 0 && (
              <div className="flex items-center gap-2 px-2 pt-3 pb-1">
                <div className="flex-1 h-px bg-border" />
                <span className="text-xs text-muted-foreground shrink-0">
                  Ungrouped (0)
                </span>
                <div className="flex-1 h-px bg-border" />
              </div>
            )}
        </div>
      </ScrollArea>

      {/* Dialogs */}
      {renderNewProjectDialog()}
      {renderRenameDialog()}
      {renderDeleteConfirmDialog()}
    </div>
  );
}
