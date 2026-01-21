"use client";

import * as React from "react";
import { Loader2, CheckCircle, AlertCircle, Pause } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

/**
 * Loading state status types.
 */
export type LoadingStatus =
  | "idle"
  | "loading"
  | "success"
  | "error"
  | "paused";

/**
 * Props for LoadingState component.
 */
interface LoadingStateProps {
  /** Whether the component is in a loading state */
  isLoading: boolean;
  /** Optional loading message to display */
  message?: string;
  /** Optional progress value (0-100) */
  progress?: number;
  /** Child content to render (becomes semi-transparent when loading) */
  children: React.ReactNode;
  /** Loading status for icon/color variations */
  status?: LoadingStatus;
  /** Additional class name for the container */
  className?: string;
  /** Whether to show the overlay (default: true) */
  showOverlay?: boolean;
  /** Whether to disable pointer events on children when loading (default: true) */
  disableInteraction?: boolean;
  /** Optional sub-message for additional context */
  subMessage?: string;
  /** Whether to show spinner (default: true when loading) */
  showSpinner?: boolean;
  /** Spinner size in pixels */
  spinnerSize?: number;
}

/**
 * Unified Loading State Component
 *
 * Provides consistent loading states across the application.
 * Wraps content and shows loading overlay with progress when active.
 *
 * Features:
 * - Progress bar with optional percentage
 * - Loading spinner
 * - Status-based icons and colors
 * - Semi-transparent overlay
 * - Disabled interaction during loading
 *
 * Usage:
 * ```tsx
 * <LoadingState isLoading={isProcessing} message="Processing..." progress={50}>
 *   <YourContent />
 * </LoadingState>
 * ```
 */
export function LoadingState({
  isLoading,
  message,
  progress,
  children,
  status = "loading",
  className,
  showOverlay = true,
  disableInteraction = true,
  subMessage,
  showSpinner = true,
  spinnerSize = 24,
}: LoadingStateProps) {
  // Don't show overlay if not loading
  if (!isLoading) {
    return <>{children}</>;
  }

  // Determine icon based on status
  const StatusIcon = {
    idle: null,
    loading: Loader2,
    success: CheckCircle,
    error: AlertCircle,
    paused: Pause,
  }[status];

  // Determine color based on status
  const statusColor = {
    idle: "text-muted-foreground",
    loading: "text-primary",
    success: "text-green-500",
    error: "text-destructive",
    paused: "text-yellow-500",
  }[status];

  return (
    <div className={cn("relative", className)}>
      {/* Loading overlay */}
      {showOverlay && (
        <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg">
          <div className="flex flex-col items-center gap-3 max-w-xs text-center px-4">
            {/* Spinner/Icon */}
            {showSpinner && StatusIcon && (
              <StatusIcon
                className={cn(
                  statusColor,
                  status === "loading" && "animate-spin"
                )}
                style={{ width: spinnerSize, height: spinnerSize }}
              />
            )}

            {/* Message */}
            {message && (
              <p className="text-sm font-medium text-foreground">{message}</p>
            )}

            {/* Sub-message */}
            {subMessage && (
              <p className="text-xs text-muted-foreground">{subMessage}</p>
            )}

            {/* Progress bar */}
            {progress !== undefined && (
              <div className="w-full space-y-1">
                <Progress value={progress} className="h-2" />
                <p className="text-xs text-muted-foreground text-center">
                  {Math.round(progress)}%
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Children with reduced opacity and disabled interaction */}
      <div
        className={cn(
          isLoading && "opacity-50 transition-opacity duration-200",
          isLoading && disableInteraction && "pointer-events-none select-none"
        )}
        aria-hidden={isLoading}
      >
        {children}
      </div>
    </div>
  );
}

/**
 * Inline Loading Indicator
 *
 * A simpler inline loading indicator for buttons, table cells, etc.
 */
interface InlineLoadingProps {
  /** Whether loading */
  isLoading: boolean;
  /** Optional message */
  message?: string;
  /** Size in pixels */
  size?: number;
  /** Additional class name */
  className?: string;
}

export function InlineLoading({
  isLoading,
  message,
  size = 16,
  className,
}: InlineLoadingProps) {
  if (!isLoading) return null;

  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <Loader2
        className="animate-spin text-primary"
        style={{ width: size, height: size }}
      />
      {message && <span className="text-sm text-muted-foreground">{message}</span>}
    </span>
  );
}

/**
 * Skeleton Placeholder
 *
 * Animated skeleton for content that's loading.
 */
interface SkeletonPlaceholderProps {
  /** Width of skeleton */
  width?: string | number;
  /** Height of skeleton */
  height?: string | number;
  /** Additional class name */
  className?: string;
  /** Whether to animate */
  animate?: boolean;
  /** Shape variant */
  variant?: "text" | "circle" | "rectangle";
}

export function SkeletonPlaceholder({
  width = "100%",
  height = 20,
  className,
  animate = true,
  variant = "text",
}: SkeletonPlaceholderProps) {
  const variantClasses = {
    text: "rounded",
    circle: "rounded-full",
    rectangle: "rounded-lg",
  };

  return (
    <div
      className={cn(
        "bg-muted",
        animate && "animate-pulse",
        variantClasses[variant],
        className
      )}
      style={{
        width: typeof width === "number" ? `${width}px` : width,
        height: typeof height === "number" ? `${height}px` : height,
      }}
      role="status"
      aria-label="Loading..."
    />
  );
}

/**
 * Full Page Loading
 *
 * Loading state that covers the entire page.
 */
interface FullPageLoadingProps {
  /** Loading message */
  message?: string;
  /** Whether visible */
  isLoading: boolean;
  /** Progress value */
  progress?: number;
}

export function FullPageLoading({
  message = "Loading...",
  isLoading,
  progress,
}: FullPageLoadingProps) {
  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="flex flex-col items-center gap-4 p-8 rounded-lg bg-card shadow-lg max-w-sm w-full mx-4">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
        <p className="text-lg font-medium text-foreground">{message}</p>
        {progress !== undefined && (
          <div className="w-full space-y-2">
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-muted-foreground text-center">
              {Math.round(progress)}% complete
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Button Loading State Hook
 *
 * Helper hook for managing button loading states.
 */
export function useLoadingState(initialState = false) {
  const [isLoading, setIsLoading] = React.useState(initialState);
  const [message, setMessage] = React.useState<string | undefined>();
  const [progress, setProgress] = React.useState<number | undefined>();

  const startLoading = React.useCallback((msg?: string) => {
    setIsLoading(true);
    setMessage(msg);
    setProgress(undefined);
  }, []);

  const stopLoading = React.useCallback(() => {
    setIsLoading(false);
    setMessage(undefined);
    setProgress(undefined);
  }, []);

  const updateProgress = React.useCallback((value: number, msg?: string) => {
    setProgress(value);
    if (msg) setMessage(msg);
  }, []);

  return {
    isLoading,
    message,
    progress,
    startLoading,
    stopLoading,
    updateProgress,
    setMessage,
    setProgress,
  };
}
