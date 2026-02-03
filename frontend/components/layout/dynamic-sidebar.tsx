"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  ChevronDown,
  ChevronRight,
  Pin,
  PinOff,
  Star,
  StarOff,
  Search,
  Sparkles,
  LayoutDashboard,
} from "lucide-react";
import { MenuSection, getIcon, ROLE_LEVELS, getRoleName } from "@/lib/menu-config";
import { useMenuContext } from "@/hooks/use-menu";

interface DynamicSidebarProps {
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
}

export function DynamicSidebar({ collapsed = false, onCollapsedChange }: DynamicSidebarProps) {
  const pathname = usePathname();
  const {
    sections,
    mode,
    preferences,
    isLoading,
    toggleMode,
    pinSection,
    unpinSection,
    collapseSection,
    expandSection,
    addFavorite,
    removeFavorite,
    collapsedKeys,
    isSimpleMode,
  } = useMenuContext();

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<MenuSection[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  // Filter sections by search
  const filteredSections = searchQuery
    ? sections.filter(
        (s) =>
          s.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.key.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : sections;

  // Handle search
  const handleSearch = useCallback(
    async (query: string) => {
      setSearchQuery(query);
      if (query.length > 1) {
        setIsSearching(true);
        // Local filter is sufficient for now
        setIsSearching(false);
      }
    },
    []
  );

  // Check if a section is active
  const isActive = (path: string) => pathname === path || pathname.startsWith(path + "/");

  // Render a single menu item
  const renderMenuItem = (section: MenuSection, depth: number = 0) => {
    const Icon = getIcon(section.icon);
    const active = isActive(section.path);
    const hasChildren = section.children && section.children.length > 0;
    const isCollapsed = collapsedKeys.has(section.key);
    const isPinned = preferences?.pinnedSections.includes(section.key);
    const isFavorite = preferences?.favorites.includes(section.key);

    if (hasChildren) {
      return (
        <Collapsible
          key={section.key}
          open={!isCollapsed}
          onOpenChange={(open) => (open ? expandSection(section.key) : collapseSection(section.key))}
        >
          <div className="group relative">
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                className={cn(
                  "w-full justify-start gap-2 px-3 py-2",
                  depth > 0 && "pl-8",
                  active && "bg-accent"
                )}
              >
                <Icon className="h-4 w-4 shrink-0" />
                {!collapsed && (
                  <>
                    <span className="flex-1 text-left">{section.label}</span>
                    {section.badge && (
                      <Badge variant="secondary" className="text-xs">
                        {section.badge}
                      </Badge>
                    )}
                    {isCollapsed ? (
                      <ChevronRight className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </>
                )}
              </Button>
            </CollapsibleTrigger>

            {/* Hover actions */}
            {!collapsed && (
              <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 flex gap-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={(e) => {
                          e.stopPropagation();
                          isPinned ? unpinSection(section.key) : pinSection(section.key);
                        }}
                      >
                        {isPinned ? <PinOff className="h-3 w-3" /> : <Pin className="h-3 w-3" />}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>{isPinned ? "Unpin" : "Pin to top"}</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            )}
          </div>

          <CollapsibleContent>
            <div className="ml-4 border-l pl-2">
              {section.children?.map((child) => renderMenuItem(child, depth + 1))}
            </div>
          </CollapsibleContent>
        </Collapsible>
      );
    }

    return (
      <div key={section.key} className="group relative">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={active ? "secondary" : "ghost"}
                className={cn(
                  "w-full justify-start gap-2 px-3 py-2",
                  depth > 0 && "pl-8",
                  section.pinned && "border-l-2 border-primary"
                )}
                asChild
              >
                <Link href={section.path}>
                  <Icon className="h-4 w-4 shrink-0" />
                  {!collapsed && (
                    <>
                      <span className="flex-1 text-left">{section.label}</span>
                      {section.badge && (
                        <Badge variant="secondary" className="text-xs">
                          {section.badge}
                        </Badge>
                      )}
                    </>
                  )}
                </Link>
              </Button>
            </TooltipTrigger>
            {collapsed && <TooltipContent side="right">{section.label}</TooltipContent>}
          </Tooltip>
        </TooltipProvider>

        {/* Hover actions */}
        {!collapsed && (
          <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 flex gap-1">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation();
                      isPinned ? unpinSection(section.key) : pinSection(section.key);
                    }}
                  >
                    {isPinned ? <PinOff className="h-3 w-3" /> : <Pin className="h-3 w-3" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{isPinned ? "Unpin" : "Pin to top"}</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation();
                      isFavorite ? removeFavorite(section.key) : addFavorite(section.key);
                    }}
                  >
                    {isFavorite ? (
                      <StarOff className="h-3 w-3" />
                    ) : (
                      <Star className="h-3 w-3" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{isFavorite ? "Remove from favorites" : "Add to favorites"}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="flex flex-col h-full p-4 space-y-4">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-8 bg-muted rounded animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-4">
          <Link href="/dashboard" className="flex items-center gap-2">
            <LayoutDashboard className="h-5 w-5" />
            {!collapsed && <span className="font-semibold">Dashboard</span>}
          </Link>
        </div>

        {/* Search */}
        {!collapsed && (
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search menu..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              className="pl-8"
            />
          </div>
        )}
      </div>

      {/* Mode Toggle */}
      {!collapsed && (
        <div className="p-4 border-b">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              <Label htmlFor="menu-mode" className="text-sm">
                {isSimpleMode ? "Simple Mode" : "Complete Mode"}
              </Label>
            </div>
            <Switch id="menu-mode" checked={!isSimpleMode} onCheckedChange={toggleMode} />
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {isSimpleMode ? "Essential features only" : "All features enabled"}
          </p>
        </div>
      )}

      {/* Menu Sections */}
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {/* Pinned sections first */}
          {preferences?.pinnedSections && preferences.pinnedSections.length > 0 && (
            <>
              {!collapsed && (
                <div className="px-3 py-2 text-xs font-medium text-muted-foreground flex items-center gap-1">
                  <Pin className="h-3 w-3" />
                  Pinned
                </div>
              )}
              {filteredSections
                .filter((s) => preferences.pinnedSections.includes(s.key))
                .map((section) => renderMenuItem(section))}
              <Separator className="my-2" />
            </>
          )}

          {/* Main sections */}
          {filteredSections
            .filter((s) => !preferences?.pinnedSections.includes(s.key))
            .map((section) => renderMenuItem(section))}
        </div>
      </ScrollArea>

      {/* Footer with role info */}
      {!collapsed && preferences && (
        <div className="p-4 border-t text-xs text-muted-foreground">
          <div className="flex items-center justify-between">
            <span>Menu Mode:</span>
            <Badge variant="outline">{mode}</Badge>
          </div>
        </div>
      )}
    </div>
  );
}
