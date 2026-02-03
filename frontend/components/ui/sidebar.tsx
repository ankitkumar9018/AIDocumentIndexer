"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  Upload,
  FileText,
  Globe,
  Settings,
  Users,
  BarChart3,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Sparkles,
  Link2,
  Wrench,
  FileType,
  Mic,
  FileSearch,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface SidebarProps {
  className?: string;
}

interface NavItem {
  title: string;
  href: string;
  icon: React.ElementType;
  badge?: number;
  adminOnly?: boolean;
}

const mainNavItems: NavItem[] = [
  {
    title: "Chat",
    href: "/dashboard/chat",
    icon: MessageSquare,
  },
  {
    title: "Documents",
    href: "/dashboard/documents",
    icon: FileText,
  },
  {
    title: "Upload",
    href: "/dashboard/upload",
    icon: Upload,
  },
  {
    title: "Generate",
    href: "/dashboard/generate",
    icon: Sparkles,
  },
  {
    title: "Web Scraper",
    href: "/dashboard/scraper",
    icon: Globe,
  },
  {
    title: "Link Groups",
    href: "/dashboard/link-groups",
    icon: Link2,
  },
  {
    title: "Collections",
    href: "/dashboard/collections",
    icon: FolderOpen,
  },
];

const toolsNavItems: NavItem[] = [
  {
    title: "PDF Tools",
    href: "/dashboard/tools/pdf",
    icon: FileType,
  },
  {
    title: "Audio Tools",
    href: "/dashboard/audio",
    icon: Mic,
  },
  {
    title: "Deep Research",
    href: "/dashboard/reports",
    icon: FileSearch,
  },
];

const adminNavItems: NavItem[] = [
  {
    title: "Users",
    href: "/dashboard/admin/users",
    icon: Users,
    adminOnly: true,
  },
  {
    title: "Analytics",
    href: "/dashboard/admin/analytics",
    icon: BarChart3,
    adminOnly: true,
  },
  {
    title: "Settings",
    href: "/dashboard/admin/settings",
    icon: Settings,
    adminOnly: true,
  },
];

export function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = React.useState(false);

  // Default user for demo mode; in production, use useSession() from next-auth
  // Example with auth: const { data: session } = useSession();
  // const user = session?.user ?? defaultUser;
  const user = {
    name: "John Doe",
    email: "john@example.com",
    role: "admin",
    tier: 100,
    avatar: null,
  };

  const isAdmin = user.role === "admin" || user.tier >= 90;

  return (
    <div
      className={cn(
        "relative flex flex-col border-r bg-card transition-all duration-300",
        collapsed ? "w-16" : "w-64",
        className
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center border-b px-4">
        <Link href="/dashboard" className="flex items-center gap-2">
          <FileText className="h-6 w-6 text-primary" />
          {!collapsed && (
            <span className="text-lg font-bold">AIDocIndexer</span>
          )}
        </Link>
      </div>

      {/* Collapse Toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute -right-3 top-20 z-10 h-6 w-6 rounded-full border bg-background shadow-md"
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? (
          <ChevronRight className="h-4 w-4" />
        ) : (
          <ChevronLeft className="h-4 w-4" />
        )}
      </Button>

      {/* Navigation */}
      <ScrollArea className="flex-1 py-4">
        <nav className="space-y-1 px-2">
          {/* Main Navigation */}
          {mainNavItems.map((item) => (
            <NavLink
              key={item.href}
              item={item}
              isActive={pathname === item.href || pathname.startsWith(item.href + "/")}
              collapsed={collapsed}
            />
          ))}

          {/* Tools Section */}
          <div className="my-4 border-t" />
          {!collapsed && (
            <p className="mb-2 px-3 text-xs font-semibold uppercase text-muted-foreground">
              Tools
            </p>
          )}
          {toolsNavItems.map((item) => (
            <NavLink
              key={item.href}
              item={item}
              isActive={pathname === item.href || pathname.startsWith(item.href + "/")}
              collapsed={collapsed}
            />
          ))}

          {/* Admin Section */}
          {isAdmin && (
            <>
              <div className="my-4 border-t" />
              {!collapsed && (
                <p className="mb-2 px-3 text-xs font-semibold uppercase text-muted-foreground">
                  Admin
                </p>
              )}
              {adminNavItems.map((item) => (
                <NavLink
                  key={item.href}
                  item={item}
                  isActive={pathname === item.href || pathname.startsWith(item.href + "/")}
                  collapsed={collapsed}
                />
              ))}
            </>
          )}
        </nav>
      </ScrollArea>

      {/* User Section */}
      <div className="border-t p-4">
        <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
          <Avatar className="h-8 w-8">
            <AvatarImage src={user.avatar || undefined} />
            <AvatarFallback>
              {user.name
                .split(" ")
                .map((n) => n[0])
                .join("")}
            </AvatarFallback>
          </Avatar>
          {!collapsed && (
            <div className="flex-1 overflow-hidden">
              <p className="truncate text-sm font-medium">{user.name}</p>
              <p className="truncate text-xs text-muted-foreground">
                Tier {user.tier}
              </p>
            </div>
          )}
          {!collapsed && (
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <LogOut className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

interface NavLinkProps {
  item: NavItem;
  isActive: boolean;
  collapsed: boolean;
}

function NavLink({ item, isActive, collapsed }: NavLinkProps) {
  const Icon = item.icon;

  return (
    <Link
      href={item.href}
      className={cn(
        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
        isActive
          ? "bg-primary text-primary-foreground"
          : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
        collapsed && "justify-center px-2"
      )}
      title={collapsed ? item.title : undefined}
    >
      <Icon className="h-5 w-5 shrink-0" />
      {!collapsed && (
        <>
          <span className="flex-1">{item.title}</span>
          {item.badge !== undefined && item.badge > 0 && (
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary-foreground text-xs font-medium text-primary">
              {item.badge > 99 ? "99+" : item.badge}
            </span>
          )}
        </>
      )}
    </Link>
  );
}
