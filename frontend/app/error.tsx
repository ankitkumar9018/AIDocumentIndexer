"use client";

import { useEffect } from "react";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function Error({ error, reset }: ErrorProps) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Application error:", error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center px-4">
        <div className="flex justify-center mb-6">
          <div className="p-4 rounded-full bg-destructive/10">
            <AlertTriangle className="h-16 w-16 text-destructive" />
          </div>
        </div>

        <h1 className="text-4xl font-bold text-foreground mb-2">
          Something Went Wrong
        </h1>
        <p className="text-muted-foreground max-w-md mx-auto mb-4">
          An unexpected error occurred. Our team has been notified and is
          working to fix the issue.
        </p>

        {error.digest && (
          <p className="text-sm text-muted-foreground mb-6">
            Error ID:{" "}
            <code className="bg-muted px-2 py-1 rounded text-xs">
              {error.digest}
            </code>
          </p>
        )}

        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
          <Button onClick={reset} variant="default">
            <RefreshCw className="h-4 w-4 mr-2" />
            Try Again
          </Button>
          <Button asChild variant="outline">
            <Link href="/">
              <Home className="h-4 w-4 mr-2" />
              Go to Dashboard
            </Link>
          </Button>
        </div>

        <div className="bg-muted rounded-lg p-4 max-w-lg mx-auto text-left">
          <h3 className="font-semibold text-sm mb-2">What you can try:</h3>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Refresh the page and try again</li>
            <li>• Clear your browser cache</li>
            <li>• Check your internet connection</li>
            <li>• Try again in a few minutes</li>
          </ul>
        </div>

        <div className="mt-8 text-sm text-muted-foreground">
          <p>
            Need help?{" "}
            <Link href="/support" className="text-primary hover:underline">
              Contact support
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
