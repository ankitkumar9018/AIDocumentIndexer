"use client";

import { useState, useEffect, use } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Play, GitBranch, Lock, AlertCircle, CheckCircle2, Clock, Copy, ExternalLink } from "lucide-react";
import { toast } from "sonner";

interface SharedWorkflowInfo {
  workflow_id: string;
  workflow_name: string;
  workflow_description: string | null;
  permission_level: string;
  input_schema: Array<{
    name: string;
    type: string;
    label: string;
    description?: string;
    required?: boolean;
    default?: unknown;
    options?: string[];
  }>;
  requires_password: boolean;
  is_valid: boolean;
  error?: string;
}

interface ExecutionResult {
  execution_id: string;
  status: string;
  output_data?: Record<string, unknown>;
  error_message?: string;
}

export default function SharedWorkflowPage({ params }: { params: Promise<{ token: string }> }) {
  const resolvedParams = use(params);
  const { token } = resolvedParams;

  const [isLoading, setIsLoading] = useState(true);
  const [workflowInfo, setWorkflowInfo] = useState<SharedWorkflowInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const [isPasswordVerified, setIsPasswordVerified] = useState(false);
  const [inputs, setInputs] = useState<Record<string, unknown>>({});
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  useEffect(() => {
    fetchWorkflowInfo();
  }, [token]);

  const fetchWorkflowInfo = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/v1/public/shared/workflow/${token}`);

      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData.detail || "Failed to load shared workflow");
        return;
      }

      const data = await response.json();
      setWorkflowInfo(data);

      // Initialize inputs with defaults
      const defaultInputs: Record<string, unknown> = {};
      for (const field of data.input_schema || []) {
        if (field.default !== undefined) {
          defaultInputs[field.name] = field.default;
        }
      }
      setInputs(defaultInputs);

      // If no password required, mark as verified
      if (!data.requires_password) {
        setIsPasswordVerified(true);
      }
    } catch {
      setError("Failed to connect to server");
    } finally {
      setIsLoading(false);
    }
  };

  const verifyPassword = async () => {
    try {
      const response = await fetch(`/api/v1/public/shared/workflow/${token}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });

      if (response.ok) {
        setIsPasswordVerified(true);
        toast.success("Access granted");
      } else {
        toast.error("Invalid password");
      }
    } catch {
      toast.error("Failed to verify password");
    }
  };

  const handleInputChange = (name: string, value: unknown) => {
    setInputs((prev) => ({ ...prev, [name]: value }));
  };

  const executeWorkflow = async () => {
    if (!workflowInfo) return;

    setIsExecuting(true);
    setExecutionResult(null);

    try {
      const response = await fetch(`/api/v1/public/shared/workflow/${token}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          inputs,
          password: workflowInfo.requires_password ? password : undefined,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        toast.error(errorData.detail || "Execution failed");
        return;
      }

      const result = await response.json();
      setExecutionResult(result);

      // Start polling for status if pending/running
      if (result.status === "pending" || result.status === "running") {
        pollExecutionStatus(result.execution_id);
      } else if (result.status === "completed") {
        toast.success("Workflow completed successfully");
      }
    } catch {
      toast.error("Failed to execute workflow");
    } finally {
      setIsExecuting(false);
    }
  };

  const pollExecutionStatus = async (executionId: string) => {
    setIsPolling(true);

    const poll = async () => {
      try {
        const response = await fetch(
          `/api/v1/public/shared/workflow/${token}/status/${executionId}`
        );

        if (response.ok) {
          const status = await response.json();
          setExecutionResult(status);

          if (status.status === "completed") {
            toast.success("Workflow completed successfully");
            setIsPolling(false);
          } else if (status.status === "failed") {
            toast.error("Workflow execution failed");
            setIsPolling(false);
          } else {
            // Continue polling
            setTimeout(poll, 2000);
          }
        }
      } catch {
        setIsPolling(false);
      }
    };

    poll();
  };

  const renderInputField = (field: SharedWorkflowInfo["input_schema"][0]) => {
    const value = inputs[field.name] ?? "";

    switch (field.type) {
      case "textarea":
        return (
          <Textarea
            id={field.name}
            value={value as string}
            onChange={(e) => handleInputChange(field.name, e.target.value)}
            placeholder={field.description}
            rows={4}
          />
        );
      case "number":
        return (
          <Input
            id={field.name}
            type="number"
            value={value as number}
            onChange={(e) => handleInputChange(field.name, parseFloat(e.target.value) || 0)}
            placeholder={field.description}
          />
        );
      case "select":
        return (
          <select
            id={field.name}
            value={value as string}
            onChange={(e) => handleInputChange(field.name, e.target.value)}
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            <option value="">Select {field.label}</option>
            {field.options?.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        );
      case "checkbox":
        return (
          <div className="flex items-center space-x-2">
            <input
              id={field.name}
              type="checkbox"
              checked={value as boolean}
              onChange={(e) => handleInputChange(field.name, e.target.checked)}
              className="h-4 w-4 rounded border-gray-300"
            />
            <Label htmlFor={field.name} className="text-sm text-muted-foreground">
              {field.description}
            </Label>
          </div>
        );
      default:
        return (
          <Input
            id={field.name}
            type="text"
            value={value as string}
            onChange={(e) => handleInputChange(field.name, e.target.value)}
            placeholder={field.description}
          />
        );
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !workflowInfo) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <Card className="w-full max-w-md">
          <CardContent className="pt-6">
            <div className="flex flex-col items-center text-center">
              <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
              <h2 className="text-xl font-semibold mb-2">Access Denied</h2>
              <p className="text-muted-foreground">
                {error || "This shared workflow link is invalid or has expired."}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Password verification screen
  if (workflowInfo.requires_password && !isPasswordVerified) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 bg-muted/30">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mb-4">
              <Lock className="h-6 w-6 text-primary" />
            </div>
            <CardTitle>Password Protected</CardTitle>
            <CardDescription>
              Enter the password to access this workflow
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && verifyPassword()}
                placeholder="Enter password"
              />
            </div>
            <Button onClick={verifyPassword} className="w-full">
              Access Workflow
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-muted/30 p-4 md:p-8">
      <div className="max-w-2xl mx-auto">
        <Card>
          <CardHeader>
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                  <GitBranch className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle>{workflowInfo.workflow_name}</CardTitle>
                  {workflowInfo.workflow_description && (
                    <CardDescription>{workflowInfo.workflow_description}</CardDescription>
                  )}
                </div>
              </div>
              <Badge variant="secondary" className="capitalize">
                {workflowInfo.permission_level}
              </Badge>
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Input Fields */}
            {workflowInfo.input_schema && workflowInfo.input_schema.length > 0 && (
              <div className="space-y-4">
                <h3 className="font-medium">Inputs</h3>
                {workflowInfo.input_schema.map((field) => (
                  <div key={field.name} className="space-y-2">
                    <Label htmlFor={field.name}>
                      {field.label}
                      {field.required && <span className="text-red-500 ml-1">*</span>}
                    </Label>
                    {renderInputField(field)}
                    {field.description && field.type !== "checkbox" && (
                      <p className="text-xs text-muted-foreground">{field.description}</p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Execute Button */}
            {workflowInfo.permission_level !== "viewer" && (
              <Button
                onClick={executeWorkflow}
                disabled={isExecuting || isPolling}
                className="w-full"
                size="lg"
              >
                {isExecuting || isPolling ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    {isPolling ? "Running..." : "Starting..."}
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Run Workflow
                  </>
                )}
              </Button>
            )}

            {/* Execution Result */}
            {executionResult && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  {executionResult.status === "completed" && (
                    <CheckCircle2 className="h-5 w-5 text-green-500" />
                  )}
                  {executionResult.status === "failed" && (
                    <AlertCircle className="h-5 w-5 text-red-500" />
                  )}
                  {(executionResult.status === "running" || executionResult.status === "pending") && (
                    <Clock className="h-5 w-5 text-yellow-500" />
                  )}
                  <span className="font-medium capitalize">{executionResult.status}</span>
                </div>

                {executionResult.error_message && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{executionResult.error_message}</AlertDescription>
                  </Alert>
                )}

                {executionResult.output_data && executionResult.status === "completed" && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Output</Label>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          navigator.clipboard.writeText(
                            JSON.stringify(executionResult.output_data, null, 2)
                          );
                          toast.success("Copied to clipboard");
                        }}
                      >
                        <Copy className="h-4 w-4 mr-1" />
                        Copy
                      </Button>
                    </div>
                    <pre className="bg-muted p-4 rounded-lg overflow-auto text-sm max-h-64">
                      {JSON.stringify(executionResult.output_data, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {/* Copy Link */}
            {workflowInfo.permission_level === "editor" && (
              <div className="pt-4 border-t">
                <Button
                  variant="outline"
                  onClick={() => {
                    toast.info("Copy functionality - redirect to workflow editor to make a copy");
                  }}
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Make a Copy
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <p className="text-center text-xs text-muted-foreground mt-4">
          Powered by AIDocumentIndexer
        </p>
      </div>
    </div>
  );
}
