"use client";

import Link from "next/link";
import { FileQuestion, Home, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center px-4">
        <div className="flex justify-center mb-6">
          <div className="p-4 rounded-full bg-muted">
            <FileQuestion className="h-16 w-16 text-muted-foreground" />
          </div>
        </div>

        <h1 className="text-6xl font-bold text-foreground mb-2">404</h1>
        <h2 className="text-2xl font-semibold text-foreground mb-4">
          Page Not Found
        </h2>
        <p className="text-muted-foreground max-w-md mx-auto mb-8">
          The page you're looking for doesn't exist or has been moved.
          Please check the URL or navigate back to the dashboard.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button asChild variant="default">
            <Link href="/">
              <Home className="h-4 w-4 mr-2" />
              Go to Dashboard
            </Link>
          </Button>
          <Button asChild variant="outline" onClick={() => window.history.back()}>
            <span className="cursor-pointer">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Go Back
            </span>
          </Button>
        </div>

        <div className="mt-12 text-sm text-muted-foreground">
          <p>
            If you believe this is an error,{" "}
            <Link href="/support" className="text-primary hover:underline">
              contact support
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
