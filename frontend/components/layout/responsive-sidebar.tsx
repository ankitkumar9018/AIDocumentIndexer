"use client";

import * as React from "react";
import { Menu, X, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

/**
 * Breakpoint for mobile/desktop switch (in pixels).
 */
const MOBILE_BREAKPOINT = 768;

/**
 * Hook to detect if we're on mobile.
 */
export function useIsMobile() {
  const [isMobile, setIsMobile] = React.useState(false);

  React.useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < MOBILE_BREAKPOINT);
    };

    // Check on mount
    checkMobile();

    // Check on resize
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return isMobile;
}

/**
 * Hook to manage sidebar state with localStorage persistence.
 */
export function useSidebarState(defaultCollapsed = false) {
  const [isCollapsed, setIsCollapsed] = React.useState(defaultCollapsed);
  const [isOpen, setIsOpen] = React.useState(false); // For mobile sheet

  // Load from localStorage on mount
  React.useEffect(() => {
    const stored = localStorage.getItem("sidebar-collapsed");
    if (stored !== null) {
      setIsCollapsed(stored === "true");
    }
  }, []);

  // Save to localStorage when changed
  const toggleCollapsed = React.useCallback(() => {
    setIsCollapsed((prev) => {
      const newValue = !prev;
      localStorage.setItem("sidebar-collapsed", String(newValue));
      return newValue;
    });
  }, []);

  const toggleOpen = React.useCallback(() => {
    setIsOpen((prev) => !prev);
  }, []);

  return {
    isCollapsed,
    setIsCollapsed,
    toggleCollapsed,
    isOpen,
    setIsOpen,
    toggleOpen,
  };
}

/**
 * Props for ResponsiveSidebar component.
 */
interface ResponsiveSidebarProps {
  /** Sidebar content to render */
  children: React.ReactNode;
  /** Header content (shown at top of sidebar) */
  header?: React.ReactNode;
  /** Footer content (shown at bottom of sidebar) */
  footer?: React.ReactNode;
  /** Width when expanded (desktop) */
  expandedWidth?: string;
  /** Width when collapsed (desktop) */
  collapsedWidth?: string;
  /** Additional class name */
  className?: string;
  /** Whether sidebar is collapsible on desktop */
  collapsible?: boolean;
  /** Default collapsed state */
  defaultCollapsed?: boolean;
  /** Controlled collapsed state */
  collapsed?: boolean;
  /** Callback when collapsed state changes */
  onCollapsedChange?: (collapsed: boolean) => void;
  /** Side to show sidebar on */
  side?: "left" | "right";
  /** Accessibility label for mobile trigger */
  mobileLabel?: string;
}

/**
 * Responsive Sidebar Component
 *
 * Renders a sidebar that:
 * - On desktop: Shows as fixed sidebar with optional collapse
 * - On mobile: Shows as a Sheet (slide-in panel)
 *
 * Features:
 * - Automatic mobile detection
 * - Collapsible on desktop
 * - Persistent collapsed state (localStorage)
 * - Smooth animations
 * - Accessible
 *
 * Usage:
 * ```tsx
 * <ResponsiveSidebar
 *   header={<Logo />}
 *   footer={<UserProfile />}
 * >
 *   <NavItems />
 * </ResponsiveSidebar>
 * ```
 */
export function ResponsiveSidebar({
  children,
  header,
  footer,
  expandedWidth = "16rem",
  collapsedWidth = "4rem",
  className,
  collapsible = true,
  defaultCollapsed = false,
  collapsed: controlledCollapsed,
  onCollapsedChange,
  side = "left",
  mobileLabel = "Open menu",
}: ResponsiveSidebarProps) {
  const isMobile = useIsMobile();
  const {
    isCollapsed: internalCollapsed,
    toggleCollapsed: internalToggle,
    isOpen,
    setIsOpen,
  } = useSidebarState(defaultCollapsed);

  // Support both controlled and uncontrolled modes
  const isCollapsed = controlledCollapsed ?? internalCollapsed;
  const toggleCollapsed = onCollapsedChange
    ? () => onCollapsedChange(!isCollapsed)
    : internalToggle;

  // Mobile: Render as Sheet
  if (isMobile) {
    return (
      <>
        {/* Mobile trigger button */}
        <Button
          variant="ghost"
          size="icon"
          className={cn(
            "fixed z-50 md:hidden",
            side === "left" ? "top-4 left-4" : "top-4 right-4"
          )}
          onClick={() => setIsOpen(true)}
          aria-label={mobileLabel}
        >
          <Menu className="h-6 w-6" />
        </Button>

        {/* Mobile sheet */}
        <Sheet open={isOpen} onOpenChange={setIsOpen}>
          <SheetContent
            side={side}
            className={cn("w-72 p-0 flex flex-col", className)}
          >
            {/* Visually hidden title for accessibility */}
            <SheetTitle className="sr-only">{mobileLabel}</SheetTitle>

            {/* Header */}
            {header && (
              <div className="flex-shrink-0 border-b px-4 py-4">{header}</div>
            )}

            {/* Content */}
            <div className="flex-1 overflow-y-auto px-2 py-4">{children}</div>

            {/* Footer */}
            {footer && (
              <div className="flex-shrink-0 border-t px-4 py-4">{footer}</div>
            )}
          </SheetContent>
        </Sheet>
      </>
    );
  }

  // Desktop: Render as fixed sidebar
  return (
    <aside
      className={cn(
        "hidden md:flex flex-col fixed inset-y-0 z-40 border-r bg-background transition-all duration-300 ease-in-out",
        side === "left" ? "left-0" : "right-0",
        className
      )}
      style={{
        width: isCollapsed ? collapsedWidth : expandedWidth,
      }}
    >
      {/* Header */}
      {header && (
        <div
          className={cn(
            "flex-shrink-0 border-b transition-all duration-300",
            isCollapsed ? "px-2 py-4" : "px-4 py-4"
          )}
        >
          {header}
        </div>
      )}

      {/* Content */}
      <div
        className={cn(
          "flex-1 overflow-y-auto transition-all duration-300",
          isCollapsed ? "px-1 py-4" : "px-2 py-4"
        )}
      >
        {children}
      </div>

      {/* Footer */}
      {footer && (
        <div
          className={cn(
            "flex-shrink-0 border-t transition-all duration-300",
            isCollapsed ? "px-2 py-4" : "px-4 py-4"
          )}
        >
          {footer}
        </div>
      )}

      {/* Collapse toggle button */}
      {collapsible && (
        <Button
          variant="ghost"
          size="icon"
          className={cn(
            "absolute top-4 -right-3 h-6 w-6 rounded-full border bg-background shadow-md hover:bg-accent",
            side === "right" && "-left-3 -right-auto"
          )}
          onClick={toggleCollapsed}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {side === "left" ? (
            isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )
          ) : isCollapsed ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </Button>
      )}
    </aside>
  );
}

/**
 * Sidebar context for child components.
 */
interface SidebarContextValue {
  isCollapsed: boolean;
  isMobile: boolean;
}

const SidebarContext = React.createContext<SidebarContextValue | undefined>(
  undefined
);

export function useSidebarContext() {
  const context = React.useContext(SidebarContext);
  if (!context) {
    throw new Error("useSidebarContext must be used within ResponsiveSidebar");
  }
  return context;
}

/**
 * Sidebar Nav Item Component
 *
 * A navigation item that adapts to sidebar collapsed state.
 */
interface SidebarNavItemProps {
  /** Icon to display */
  icon: React.ReactNode;
  /** Label text */
  label: string;
  /** Whether item is active */
  active?: boolean;
  /** Click handler */
  onClick?: () => void;
  /** Href for link behavior */
  href?: string;
  /** Additional class name */
  className?: string;
  /** Whether collapsed (if not using context) */
  collapsed?: boolean;
}

export function SidebarNavItem({
  icon,
  label,
  active = false,
  onClick,
  href,
  className,
  collapsed = false,
}: SidebarNavItemProps) {
  const baseClasses = cn(
    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
    "hover:bg-accent hover:text-accent-foreground",
    active && "bg-accent text-accent-foreground",
    collapsed && "justify-center px-2",
    className
  );

  const content = (
    <>
      <span className="flex-shrink-0">{icon}</span>
      {!collapsed && <span className="truncate">{label}</span>}
    </>
  );

  if (href) {
    return (
      <a href={href} className={baseClasses} title={collapsed ? label : undefined}>
        {content}
      </a>
    );
  }

  return (
    <button
      onClick={onClick}
      className={cn(baseClasses, "w-full text-left")}
      title={collapsed ? label : undefined}
    >
      {content}
    </button>
  );
}

/**
 * Sidebar Section Component
 *
 * Groups nav items with an optional title.
 */
interface SidebarSectionProps {
  /** Section title */
  title?: string;
  /** Section content */
  children: React.ReactNode;
  /** Additional class name */
  className?: string;
  /** Whether collapsed (if not using context) */
  collapsed?: boolean;
}

export function SidebarSection({
  title,
  children,
  className,
  collapsed = false,
}: SidebarSectionProps) {
  return (
    <div className={cn("mb-4", className)}>
      {title && !collapsed && (
        <h3 className="mb-2 px-3 text-xs font-semibold uppercase text-muted-foreground tracking-wider">
          {title}
        </h3>
      )}
      {collapsed && title && <div className="border-t my-2" />}
      <div className="space-y-1">{children}</div>
    </div>
  );
}
