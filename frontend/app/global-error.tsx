"use client";

import { useEffect } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface GlobalErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: GlobalErrorProps) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Global application error:", error);
  }, [error]);

  return (
    <html>
      <body>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#0a0a0a",
            color: "#fafafa",
            fontFamily:
              'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          }}
        >
          <div style={{ textAlign: "center", padding: "1rem" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                marginBottom: "1.5rem",
              }}
            >
              <div
                style={{
                  padding: "1rem",
                  borderRadius: "50%",
                  backgroundColor: "rgba(239, 68, 68, 0.1)",
                }}
              >
                <AlertTriangle
                  style={{ width: "4rem", height: "4rem", color: "#ef4444" }}
                />
              </div>
            </div>

            <h1
              style={{
                fontSize: "2rem",
                fontWeight: "bold",
                marginBottom: "0.5rem",
              }}
            >
              Critical Error
            </h1>
            <p
              style={{
                color: "#a1a1aa",
                maxWidth: "28rem",
                margin: "0 auto 1rem",
              }}
            >
              A critical error occurred that prevented the application from
              loading. Please try refreshing the page.
            </p>

            {error.digest && (
              <p style={{ fontSize: "0.875rem", color: "#71717a", marginBottom: "1.5rem" }}>
                Error ID:{" "}
                <code
                  style={{
                    backgroundColor: "#27272a",
                    padding: "0.25rem 0.5rem",
                    borderRadius: "0.25rem",
                    fontSize: "0.75rem",
                  }}
                >
                  {error.digest}
                </code>
              </p>
            )}

            <button
              onClick={reset}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                backgroundColor: "#fafafa",
                color: "#0a0a0a",
                padding: "0.75rem 1.5rem",
                borderRadius: "0.5rem",
                border: "none",
                cursor: "pointer",
                fontWeight: "500",
              }}
            >
              <RefreshCw style={{ width: "1rem", height: "1rem" }} />
              Reload Page
            </button>

            <div
              style={{
                marginTop: "2rem",
                fontSize: "0.875rem",
                color: "#71717a",
              }}
            >
              <p>
                If the problem persists, please contact{" "}
                <a
                  href="mailto:support@example.com"
                  style={{ color: "#3b82f6", textDecoration: "none" }}
                >
                  support
                </a>
              </p>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
