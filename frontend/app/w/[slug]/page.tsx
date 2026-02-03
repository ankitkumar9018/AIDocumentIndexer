"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import {
  Loader2,
  Play,
  AlertCircle,
  CheckCircle,
  Clock,
  Workflow,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface InputField {
  name: string;
  type: string;
  label: string;
  description?: string;
  required: boolean;
  default?: any;
  options?: string[];
}

interface PublicWorkflow {
  id: string;
  name: string;
  description?: string;
  input_schema: InputField[];
  branding?: {
    logo?: string;
    primaryColor?: string;
    companyName?: string;
  };
}

interface ExecutionStatus {
  execution_id: string;
  status: string;
  started_at?: string;
  completed_at?: string;
  output_data?: any;
  error_message?: string;
}

export default function PublicWorkflowPage() {
  const params = useParams();
  const slug = params.slug as string;

  const [workflow, setWorkflow] = useState<PublicWorkflow | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [inputs, setInputs] = useState<Record<string, any>>({});
  const [executing, setExecuting] = useState(false);
  const [executionId, setExecutionId] = useState<string | null>(null);
  const [executionStatus, setExecutionStatus] = useState<ExecutionStatus | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);

  // Fetch workflow info
  useEffect(() => {
    const fetchWorkflow = async () => {
      try {
        const response = await fetch(`/api/v1/public/workflows/${slug}`);
        if (!response.ok) {
          if (response.status === 404) {
            setError("Workflow not found or not published");
          } else {
            setError("Failed to load workflow");
          }
          return;
        }
        const data = await response.json();
        setWorkflow(data);

        // Initialize inputs with defaults
        const defaultInputs: Record<string, any> = {};
        data.input_schema.forEach((field: InputField) => {
          if (field.default !== undefined) {
            defaultInputs[field.name] = field.default;
          }
        });
        setInputs(defaultInputs);
      } catch (err) {
        setError("Failed to load workflow");
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflow();
  }, [slug]);

  // Poll for execution status
  useEffect(() => {
    if (!executionId || !polling) return;

    const pollStatus = async () => {
      try {
        const response = await fetch(
          `/api/v1/public/workflows/${slug}/status/${executionId}`
        );
        if (response.ok) {
          const data = await response.json();
          setExecutionStatus(data);

          // Stop polling if completed or failed
          if (data.status === "completed" || data.status === "failed" || data.status === "cancelled") {
            setPolling(false);
          }
        }
      } catch (err) {
        console.error("Failed to poll status:", err);
      }
    };

    const interval = setInterval(pollStatus, 2000);
    pollStatus(); // Initial call

    return () => clearInterval(interval);
  }, [executionId, polling, slug]);

  const handleExecute = async () => {
    if (!workflow) return;

    // Validate required inputs
    for (const field of workflow.input_schema) {
      if (field.required && !inputs[field.name]) {
        setExecutionError(`Missing required field: ${field.label}`);
        return;
      }
    }

    setExecuting(true);
    setExecutionError(null);
    setExecutionStatus(null);
    setExecutionId(null);

    try {
      const response = await fetch(`/api/v1/public/workflows/${slug}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs }),
      });

      const data = await response.json();

      if (!response.ok) {
        setExecutionError(data.detail || "Execution failed");
        return;
      }

      setExecutionId(data.execution_id);
      setExecutionStatus({
        execution_id: data.execution_id,
        status: data.status,
      });
      setPolling(true);
    } catch (err) {
      setExecutionError("Failed to execute workflow");
    } finally {
      setExecuting(false);
    }
  };

  const renderInputField = (field: InputField) => {
    const value = inputs[field.name] ?? "";

    switch (field.type) {
      case "textarea":
        return (
          <Textarea
            id={field.name}
            value={value}
            onChange={(e) => setInputs({ ...inputs, [field.name]: e.target.value })}
            placeholder={field.description || `Enter ${field.label}...`}
            className="min-h-[100px]"
          />
        );

      case "select":
        return (
          <Select
            value={value}
            onValueChange={(v) => setInputs({ ...inputs, [field.name]: v })}
          >
            <SelectTrigger>
              <SelectValue placeholder={`Select ${field.label}`} />
            </SelectTrigger>
            <SelectContent>
              {(field.options || []).map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );

      case "number":
        return (
          <Input
            id={field.name}
            type="number"
            value={value}
            onChange={(e) => setInputs({ ...inputs, [field.name]: parseFloat(e.target.value) })}
            placeholder={field.description || `Enter ${field.label}...`}
          />
        );

      case "checkbox":
        return (
          <div className="flex items-center space-x-2">
            <Checkbox
              id={field.name}
              checked={value === true}
              onCheckedChange={(checked) => setInputs({ ...inputs, [field.name]: checked })}
            />
            <Label htmlFor={field.name} className="text-sm">
              {field.description || field.label}
            </Label>
          </div>
        );

      case "date":
        return (
          <Input
            id={field.name}
            type="date"
            value={value}
            onChange={(e) => setInputs({ ...inputs, [field.name]: e.target.value })}
          />
        );

      default:
        return (
          <Input
            id={field.name}
            value={value}
            onChange={(e) => setInputs({ ...inputs, [field.name]: e.target.value })}
            placeholder={field.description || `Enter ${field.label}...`}
          />
        );
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "text-green-500";
      case "failed":
        return "text-red-500";
      case "running":
        return "text-blue-500";
      case "pending":
        return "text-yellow-500";
      default:
        return "text-muted-foreground";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "failed":
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case "running":
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      case "pending":
        return <Clock className="h-5 w-5 text-yellow-500" />;
      default:
        return <Clock className="h-5 w-5" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading workflow...</p>
        </div>
      </div>
    );
  }

  if (error || !workflow) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted p-4">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-5 w-5" />
              Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>{error || "Workflow not found"}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const brandColor = workflow.branding?.primaryColor || "#8b5cf6";

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted p-4 md:p-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          {workflow.branding?.logo ? (
            <img
              src={workflow.branding.logo}
              alt={workflow.branding.companyName || "Logo"}
              className="h-12 mx-auto mb-4"
            />
          ) : (
            <div
              className="h-16 w-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
              style={{ backgroundColor: `${brandColor}20` }}
            >
              <Workflow className="h-8 w-8" style={{ color: brandColor }} />
            </div>
          )}
          <h1 className="text-2xl font-bold mb-2">{workflow.name}</h1>
          {workflow.description && (
            <p className="text-muted-foreground">{workflow.description}</p>
          )}
        </div>

        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Input</CardTitle>
            <CardDescription>
              {workflow.input_schema.length > 0
                ? "Fill in the required fields to run this workflow"
                : "This workflow has no input fields. Click Run to execute."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {workflow.input_schema.map((field) => (
              <div key={field.name} className="space-y-2">
                {field.type !== "checkbox" && (
                  <>
                    <Label htmlFor={field.name} className="flex items-center gap-2">
                      {field.label}
                      {field.required && <span className="text-destructive">*</span>}
                    </Label>
                    {field.description && field.type !== "checkbox" && (
                      <p className="text-xs text-muted-foreground">{field.description}</p>
                    )}
                  </>
                )}
                {renderInputField(field)}
              </div>
            ))}
          </CardContent>
          <CardFooter>
            <Button
              onClick={handleExecute}
              disabled={executing || polling}
              className="w-full"
              style={{ backgroundColor: brandColor }}
            >
              {executing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : polling ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Workflow
                </>
              )}
            </Button>
          </CardFooter>
        </Card>

        {/* Error */}
        {executionError && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{executionError}</AlertDescription>
          </Alert>
        )}

        {/* Execution Status */}
        {executionStatus && (
          <Card className="mt-4">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg flex items-center gap-2">
                  {getStatusIcon(executionStatus.status)}
                  Execution Status
                </CardTitle>
                <CardDescription>
                  ID: {executionStatus.execution_id.slice(0, 8)}...
                </CardDescription>
              </div>
              <Badge
                variant="outline"
                className={cn("capitalize", getStatusColor(executionStatus.status))}
              >
                {executionStatus.status}
              </Badge>
            </CardHeader>
            <CardContent>
              {executionStatus.status === "running" && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Workflow is running...</span>
                </div>
              )}

              {executionStatus.status === "completed" && executionStatus.output_data && (
                <div className="space-y-2">
                  <Label>Output</Label>
                  <div className="bg-muted rounded-lg p-4 overflow-auto max-h-[300px]">
                    <pre className="text-sm whitespace-pre-wrap">
                      {JSON.stringify(executionStatus.output_data, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {executionStatus.status === "failed" && executionStatus.error_message && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Execution Failed</AlertTitle>
                  <AlertDescription>{executionStatus.error_message}</AlertDescription>
                </Alert>
              )}

              {executionStatus.started_at && (
                <div className="mt-4 text-sm text-muted-foreground">
                  Started: {new Date(executionStatus.started_at).toLocaleString()}
                  {executionStatus.completed_at && (
                    <>
                      <br />
                      Completed: {new Date(executionStatus.completed_at).toLocaleString()}
                    </>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-muted-foreground">
          Powered by{" "}
          <a
            href="/"
            className="hover:underline"
            style={{ color: brandColor }}
          >
            AIDocumentIndexer
          </a>
        </div>
      </div>
    </div>
  );
}
