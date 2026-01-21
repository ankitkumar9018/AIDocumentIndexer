"use client";

import * as React from "react";
import {
  AlertCircle,
  RefreshCw,
  ArrowRight,
  Clock,
  Upload,
  Settings,
  Search,
  HelpCircle,
  Zap,
  WifiOff,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/**
 * Error types with associated recovery actions.
 */
export type ErrorCode =
  | "RATE_LIMIT"
  | "CONTEXT_TOO_LONG"
  | "NO_SOURCES_FOUND"
  | "NETWORK_ERROR"
  | "AUTHENTICATION_ERROR"
  | "VALIDATION_ERROR"
  | "SERVER_ERROR"
  | "TIMEOUT_ERROR"
  | "QUOTA_EXCEEDED"
  | "NOT_FOUND"
  | "UNKNOWN";

/**
 * Recovery action definition.
 */
interface RecoveryAction {
  label: string;
  description?: string;
  action: () => void | Promise<void>;
  primary?: boolean;
  icon?: React.ReactNode;
}

/**
 * Error with recovery information.
 */
export interface RecoverableError {
  message: string;
  code: ErrorCode;
  details?: string;
  retryable?: boolean;
  retryAfter?: number; // Seconds until retry is recommended
}

/**
 * Props for ErrorRecovery component.
 */
interface ErrorRecoveryProps {
  /** The error to display */
  error: RecoverableError;
  /** Callback when retry is requested */
  onRetry?: () => void | Promise<void>;
  /** Callback when the error is dismissed */
  onDismiss?: () => void;
  /** Custom recovery actions (merged with defaults) */
  customActions?: RecoveryAction[];
  /** Additional class name */
  className?: string;
  /** Whether to show default recovery actions */
  showDefaultActions?: boolean;
  /** Compact mode (less padding, smaller text) */
  compact?: boolean;
}

/**
 * Maps error codes to default recovery actions.
 */
const DEFAULT_RECOVERY_ACTIONS: Record<ErrorCode, RecoveryAction[]> = {
  RATE_LIMIT: [
    {
      label: "Wait and retry",
      description: "Automatically retry after the rate limit resets",
      action: () => {},
      primary: true,
      icon: <Clock className="w-4 h-4" />,
    },
  ],
  CONTEXT_TOO_LONG: [
    {
      label: "Summarize context",
      description: "Reduce the context size and try again",
      action: () => {},
      primary: true,
      icon: <Zap className="w-4 h-4" />,
    },
    {
      label: "Split query",
      description: "Break your question into smaller parts",
      action: () => {},
      icon: <ArrowRight className="w-4 h-4" />,
    },
  ],
  NO_SOURCES_FOUND: [
    {
      label: "Broaden search",
      description: "Try with fewer or different keywords",
      action: () => {},
      primary: true,
      icon: <Search className="w-4 h-4" />,
    },
    {
      label: "Upload documents",
      description: "Add more source documents to search",
      action: () => {},
      icon: <Upload className="w-4 h-4" />,
    },
  ],
  NETWORK_ERROR: [
    {
      label: "Check connection",
      description: "Verify your internet connection",
      action: () => {},
      primary: true,
      icon: <WifiOff className="w-4 h-4" />,
    },
    {
      label: "Retry",
      description: "Try the request again",
      action: () => {},
      icon: <RefreshCw className="w-4 h-4" />,
    },
  ],
  AUTHENTICATION_ERROR: [
    {
      label: "Sign in again",
      description: "Your session may have expired",
      action: () => {},
      primary: true,
      icon: <ArrowRight className="w-4 h-4" />,
    },
  ],
  VALIDATION_ERROR: [
    {
      label: "Review input",
      description: "Check your input for errors",
      action: () => {},
      primary: true,
      icon: <Search className="w-4 h-4" />,
    },
  ],
  SERVER_ERROR: [
    {
      label: "Retry",
      description: "The server encountered an error, try again",
      action: () => {},
      primary: true,
      icon: <RefreshCw className="w-4 h-4" />,
    },
    {
      label: "Report issue",
      description: "Let us know if this persists",
      action: () => {},
      icon: <HelpCircle className="w-4 h-4" />,
    },
  ],
  TIMEOUT_ERROR: [
    {
      label: "Retry",
      description: "The request took too long, try again",
      action: () => {},
      primary: true,
      icon: <RefreshCw className="w-4 h-4" />,
    },
    {
      label: "Simplify request",
      description: "Try a simpler query",
      action: () => {},
      icon: <Zap className="w-4 h-4" />,
    },
  ],
  QUOTA_EXCEEDED: [
    {
      label: "Upgrade plan",
      description: "Increase your usage limits",
      action: () => {},
      primary: true,
      icon: <ArrowRight className="w-4 h-4" />,
    },
    {
      label: "Wait for reset",
      description: "Quota resets periodically",
      action: () => {},
      icon: <Clock className="w-4 h-4" />,
    },
  ],
  NOT_FOUND: [
    {
      label: "Go back",
      description: "Return to the previous page",
      action: () => {},
      primary: true,
      icon: <ArrowRight className="w-4 h-4" />,
    },
  ],
  UNKNOWN: [
    {
      label: "Retry",
      description: "Try the action again",
      action: () => {},
      primary: true,
      icon: <RefreshCw className="w-4 h-4" />,
    },
    {
      label: "Get help",
      description: "Contact support for assistance",
      action: () => {},
      icon: <HelpCircle className="w-4 h-4" />,
    },
  ],
};

/**
 * Error Recovery Component
 *
 * Displays errors with actionable recovery options.
 * Shows contextual recovery actions based on error type.
 *
 * Features:
 * - Error-type specific recovery actions
 * - Automatic retry countdown for rate limits
 * - Custom action support
 * - Compact and full modes
 *
 * Usage:
 * ```tsx
 * <ErrorRecovery
 *   error={{ message: "Too many requests", code: "RATE_LIMIT", retryAfter: 60 }}
 *   onRetry={() => fetchData()}
 *   onDismiss={() => setError(null)}
 * />
 * ```
 */
export function ErrorRecovery({
  error,
  onRetry,
  onDismiss,
  customActions = [],
  className,
  showDefaultActions = true,
  compact = false,
}: ErrorRecoveryProps) {
  const [countdown, setCountdown] = React.useState(error.retryAfter || 0);
  const [isRetrying, setIsRetrying] = React.useState(false);

  // Countdown timer for rate limits
  React.useEffect(() => {
    if (countdown <= 0) return;

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [countdown]);

  // Auto-retry after countdown
  React.useEffect(() => {
    if (
      countdown === 0 &&
      error.retryAfter &&
      error.retryAfter > 0 &&
      onRetry
    ) {
      handleRetry();
    }
  }, [countdown, error.retryAfter]);

  // Get recovery actions
  const defaultActions = showDefaultActions
    ? DEFAULT_RECOVERY_ACTIONS[error.code] || DEFAULT_RECOVERY_ACTIONS.UNKNOWN
    : [];

  // Merge default actions with onRetry handler
  const actions = [
    ...defaultActions.map((action) => ({
      ...action,
      action:
        action.label.toLowerCase().includes("retry") && onRetry
          ? onRetry
          : action.action,
    })),
    ...customActions,
  ];

  const handleRetry = async () => {
    if (!onRetry || isRetrying) return;

    setIsRetrying(true);
    try {
      await onRetry();
    } finally {
      setIsRetrying(false);
    }
  };

  return (
    <Alert
      variant="destructive"
      className={cn(
        "relative",
        compact ? "p-3" : "p-4",
        className
      )}
    >
      <AlertCircle className={cn("h-4 w-4", compact && "h-3.5 w-3.5")} />
      <AlertTitle className={cn(compact && "text-sm")}>
        {getErrorTitle(error.code)}
      </AlertTitle>
      <AlertDescription className={cn("mt-2", compact && "text-xs mt-1")}>
        <p>{error.message}</p>

        {error.details && (
          <p className="mt-1 text-xs text-muted-foreground">{error.details}</p>
        )}

        {/* Countdown for rate limits */}
        {countdown > 0 && (
          <p className="mt-2 text-sm font-medium">
            Retrying in {formatCountdown(countdown)}...
          </p>
        )}

        {/* Recovery actions */}
        {actions.length > 0 && (
          <div
            className={cn(
              "flex flex-wrap gap-2",
              compact ? "mt-2" : "mt-4"
            )}
          >
            {actions.map((action, index) => (
              <TooltipProvider key={index}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={action.primary ? "default" : "outline"}
                      size={compact ? "sm" : "default"}
                      onClick={() => action.action()}
                      disabled={isRetrying || countdown > 0}
                      className={cn(compact && "h-7 text-xs")}
                    >
                      {action.icon}
                      <span className={cn(action.icon && "ml-2")}>
                        {action.label}
                      </span>
                    </Button>
                  </TooltipTrigger>
                  {action.description && (
                    <TooltipContent>
                      <p>{action.description}</p>
                    </TooltipContent>
                  )}
                </Tooltip>
              </TooltipProvider>
            ))}
          </div>
        )}
      </AlertDescription>

      {/* Dismiss button */}
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="absolute top-2 right-2 p-1 rounded-full hover:bg-destructive/20 transition-colors"
          aria-label="Dismiss error"
        >
          <span className="sr-only">Dismiss</span>
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      )}
    </Alert>
  );
}

/**
 * Get human-readable title for error code.
 */
function getErrorTitle(code: ErrorCode): string {
  const titles: Record<ErrorCode, string> = {
    RATE_LIMIT: "Too Many Requests",
    CONTEXT_TOO_LONG: "Context Too Long",
    NO_SOURCES_FOUND: "No Sources Found",
    NETWORK_ERROR: "Connection Error",
    AUTHENTICATION_ERROR: "Authentication Required",
    VALIDATION_ERROR: "Invalid Input",
    SERVER_ERROR: "Server Error",
    TIMEOUT_ERROR: "Request Timeout",
    QUOTA_EXCEEDED: "Quota Exceeded",
    NOT_FOUND: "Not Found",
    UNKNOWN: "Error",
  };
  return titles[code] || "Error";
}

/**
 * Format countdown seconds as human-readable string.
 */
function formatCountdown(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
}

/**
 * Hook for managing error state with recovery.
 */
export function useErrorRecovery() {
  const [error, setError] = React.useState<RecoverableError | null>(null);

  const setErrorFromResponse = React.useCallback(
    (response: { status?: number; message?: string; code?: string }) => {
      const code = mapStatusToErrorCode(response.status, response.code);
      setError({
        message: response.message || "An error occurred",
        code,
        retryable: isRetryable(code),
        retryAfter: code === "RATE_LIMIT" ? 60 : undefined,
      });
    },
    []
  );

  const clearError = React.useCallback(() => {
    setError(null);
  }, []);

  return {
    error,
    setError,
    setErrorFromResponse,
    clearError,
  };
}

/**
 * Map HTTP status code to error code.
 */
function mapStatusToErrorCode(
  status?: number,
  codeHint?: string
): ErrorCode {
  if (codeHint) {
    const normalized = codeHint.toUpperCase().replace(/-/g, "_");
    if (normalized in DEFAULT_RECOVERY_ACTIONS) {
      return normalized as ErrorCode;
    }
  }

  if (!status) return "UNKNOWN";

  if (status === 429) return "RATE_LIMIT";
  if (status === 401 || status === 403) return "AUTHENTICATION_ERROR";
  if (status === 404) return "NOT_FOUND";
  if (status === 408) return "TIMEOUT_ERROR";
  if (status === 413) return "CONTEXT_TOO_LONG";
  if (status === 422) return "VALIDATION_ERROR";
  if (status >= 500) return "SERVER_ERROR";

  return "UNKNOWN";
}

/**
 * Check if error code is retryable.
 */
function isRetryable(code: ErrorCode): boolean {
  return [
    "RATE_LIMIT",
    "NETWORK_ERROR",
    "SERVER_ERROR",
    "TIMEOUT_ERROR",
  ].includes(code);
}

/**
 * Error Boundary with Recovery
 *
 * Class component for catching render errors.
 */
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <ErrorRecovery
          error={{
            message: this.state.error?.message || "Something went wrong",
            code: "UNKNOWN",
            details: "The application encountered an unexpected error.",
          }}
          onRetry={() => this.setState({ hasError: false, error: null })}
        />
      );
    }

    return this.props.children;
  }
}
