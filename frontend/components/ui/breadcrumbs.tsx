"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight, Home } from "lucide-react";
import { cn } from "@/lib/utils";

export interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: React.ReactNode;
}

export interface BreadcrumbsProps extends React.HTMLAttributes<HTMLElement> {
  items?: BreadcrumbItem[];
  showHome?: boolean;
  homeHref?: string;
  separator?: React.ReactNode;
  maxItems?: number;
}

// Mapping of path segments to display labels
const pathLabels: Record<string, string> = {
  dashboard: "Dashboard",
  chat: "Chat",
  upload: "Upload",
  documents: "Documents",
  create: "Create",
  workflows: "Workflows",
  audio: "Audio Overviews",
  connectors: "Connectors",
  collaboration: "Collaboration",
  knowledge: "Knowledge Graph",
  scraper: "Web Scraper",
  costs: "Cost Tracking",
  profile: "Profile",
  admin: "Admin",
  users: "Users",
  agents: "Agents",
  audit: "Audit Logs",
  settings: "Settings",
  integrations: "Integrations",
};

// Generate breadcrumb items from a pathname
function generateBreadcrumbs(pathname: string): BreadcrumbItem[] {
  const segments = pathname.split("/").filter(Boolean);
  const items: BreadcrumbItem[] = [];

  let currentPath = "";

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    currentPath += `/${segment}`;

    // Check if this is a dynamic segment (e.g., [id])
    const isDynamic = segment.match(/^[a-f0-9-]{36}$/i) || segment.match(/^\d+$/);

    if (isDynamic) {
      // For dynamic segments, use "Details" as label
      items.push({
        label: "Details",
        href: i < segments.length - 1 ? currentPath : undefined,
      });
    } else {
      const label = pathLabels[segment] || segment.charAt(0).toUpperCase() + segment.slice(1);
      items.push({
        label,
        href: i < segments.length - 1 ? currentPath : undefined,
      });
    }
  }

  return items;
}

export function Breadcrumbs({
  items,
  showHome = true,
  homeHref = "/dashboard",
  separator,
  maxItems = 4,
  className,
  ...props
}: BreadcrumbsProps) {
  const pathname = usePathname();

  // Auto-generate breadcrumbs from pathname if items not provided
  const breadcrumbItems = items || generateBreadcrumbs(pathname);

  // Handle max items with ellipsis
  let displayItems = breadcrumbItems;
  let hasEllipsis = false;

  if (breadcrumbItems.length > maxItems) {
    hasEllipsis = true;
    displayItems = [
      breadcrumbItems[0],
      ...breadcrumbItems.slice(-(maxItems - 1)),
    ];
  }

  const separatorElement = separator || (
    <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" aria-hidden="true" />
  );

  return (
    <nav
      aria-label="Breadcrumb"
      className={cn("flex items-center text-sm", className)}
      {...props}
    >
      <ol className="flex items-center flex-wrap gap-1.5" role="list">
        {/* Home link */}
        {showHome && (
          <>
            <li className="flex items-center">
              <Link
                href={homeHref}
                className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Home"
              >
                <Home className="h-4 w-4" />
                <span className="sr-only">Home</span>
              </Link>
            </li>
            {(displayItems.length > 0 || hasEllipsis) && (
              <li className="flex items-center" role="presentation" aria-hidden="true">
                {separatorElement}
              </li>
            )}
          </>
        )}

        {/* Ellipsis for collapsed items */}
        {hasEllipsis && (
          <>
            <li className="flex items-center">
              <span
                className="text-muted-foreground px-1"
                aria-label="More items"
              >
                ...
              </span>
            </li>
            <li className="flex items-center" role="presentation" aria-hidden="true">
              {separatorElement}
            </li>
          </>
        )}

        {/* Breadcrumb items */}
        {displayItems.map((item, index) => {
          const isLast = index === displayItems.length - 1;
          const skipFirst = hasEllipsis && index === 0;

          // Skip first item if we have ellipsis (already shown)
          if (skipFirst && hasEllipsis) {
            return (
              <React.Fragment key={index}>
                <li className="flex items-center">
                  {item.href ? (
                    <Link
                      href={item.href}
                      className="text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5"
                    >
                      {item.icon}
                      {item.label}
                    </Link>
                  ) : (
                    <span
                      className={cn(
                        "flex items-center gap-1.5",
                        isLast ? "text-foreground font-medium" : "text-muted-foreground"
                      )}
                      aria-current={isLast ? "page" : undefined}
                    >
                      {item.icon}
                      {item.label}
                    </span>
                  )}
                </li>
                {!isLast && (
                  <li className="flex items-center" role="presentation" aria-hidden="true">
                    {separatorElement}
                  </li>
                )}
              </React.Fragment>
            );
          }

          return (
            <React.Fragment key={index}>
              <li className="flex items-center">
                {item.href ? (
                  <Link
                    href={item.href}
                    className="text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5"
                  >
                    {item.icon}
                    {item.label}
                  </Link>
                ) : (
                  <span
                    className={cn(
                      "flex items-center gap-1.5",
                      isLast ? "text-foreground font-medium" : "text-muted-foreground"
                    )}
                    aria-current={isLast ? "page" : undefined}
                  >
                    {item.icon}
                    {item.label}
                  </span>
                )}
              </li>
              {!isLast && (
                <li className="flex items-center" role="presentation" aria-hidden="true">
                  {separatorElement}
                </li>
              )}
            </React.Fragment>
          );
        })}
      </ol>
    </nav>
  );
}

// Compact version for use in page headers
export function PageBreadcrumbs({ className, ...props }: BreadcrumbsProps) {
  return (
    <Breadcrumbs
      className={cn("mb-4", className)}
      {...props}
    />
  );
}
