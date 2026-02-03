"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import {
  Loader2,
  Play,
  Copy,
  Check,
  AlertCircle,
  Zap,
  FileText,
  Globe,
  Heart,
  CheckCircle,
  User,
  GitCompare,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
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

interface SkillInput {
  name: string;
  type: string;
  description?: string;
  required: boolean;
  default?: any;
  options?: string[];
}

interface PublicSkill {
  id: string;
  name: string;
  description?: string;
  category: string;
  icon: string;
  inputs: SkillInput[];
  outputs: any[];
  branding?: {
    logo?: string;
    primaryColor?: string;
    companyName?: string;
  };
}

const skillIcons: Record<string, React.ReactNode> = {
  "file-text": <FileText className="h-6 w-6" />,
  "check-circle": <CheckCircle className="h-6 w-6" />,
  globe: <Globe className="h-6 w-6" />,
  heart: <Heart className="h-6 w-6" />,
  user: <User className="h-6 w-6" />,
  "git-compare": <GitCompare className="h-6 w-6" />,
  sparkles: <Sparkles className="h-6 w-6" />,
  zap: <Zap className="h-6 w-6" />,
};

export default function PublicSkillPage() {
  const params = useParams();
  const slug = params.slug as string;

  const [skill, setSkill] = useState<PublicSkill | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [inputs, setInputs] = useState<Record<string, any>>({});
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Fetch skill info
  useEffect(() => {
    const fetchSkill = async () => {
      try {
        const response = await fetch(`/api/v1/public/skills/${slug}`);
        if (!response.ok) {
          if (response.status === 404) {
            setError("Skill not found or not published");
          } else {
            setError("Failed to load skill");
          }
          return;
        }
        const data = await response.json();
        setSkill(data);

        // Initialize inputs with defaults
        const defaultInputs: Record<string, any> = {};
        data.inputs.forEach((input: SkillInput) => {
          if (input.default !== undefined) {
            defaultInputs[input.name] = input.default;
          }
        });
        setInputs(defaultInputs);
      } catch (err) {
        setError("Failed to load skill");
      } finally {
        setLoading(false);
      }
    };

    fetchSkill();
  }, [slug]);

  const handleExecute = async () => {
    if (!skill) return;

    // Validate required inputs
    for (const input of skill.inputs) {
      if (input.required && !inputs[input.name]) {
        setExecutionError(`Missing required input: ${input.name}`);
        return;
      }
    }

    setExecuting(true);
    setExecutionError(null);
    setResult(null);

    try {
      const response = await fetch(`/api/v1/public/skills/${slug}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs }),
      });

      const data = await response.json();

      if (!response.ok) {
        setExecutionError(data.detail || "Execution failed");
        return;
      }

      setResult(data);
    } catch (err) {
      setExecutionError("Failed to execute skill");
    } finally {
      setExecuting(false);
    }
  };

  const handleCopy = () => {
    if (result?.output) {
      const text = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output, null, 2);
      navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const renderInput = (input: SkillInput) => {
    const value = inputs[input.name] ?? "";

    switch (input.type) {
      case "textarea":
      case "text":
        if (input.name === "content" || input.name === "document" || input.name.includes("text")) {
          return (
            <Textarea
              id={input.name}
              value={value}
              onChange={(e) => setInputs({ ...inputs, [input.name]: e.target.value })}
              placeholder={input.description || `Enter ${input.name}...`}
              className="min-h-[150px]"
            />
          );
        }
        return (
          <Input
            id={input.name}
            value={value}
            onChange={(e) => setInputs({ ...inputs, [input.name]: e.target.value })}
            placeholder={input.description || `Enter ${input.name}...`}
          />
        );

      case "select":
        return (
          <Select
            value={value}
            onValueChange={(v) => setInputs({ ...inputs, [input.name]: v })}
          >
            <SelectTrigger>
              <SelectValue placeholder={`Select ${input.name}`} />
            </SelectTrigger>
            <SelectContent>
              {(input.options || ["short", "medium", "long"]).map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt.charAt(0).toUpperCase() + opt.slice(1)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );

      case "number":
        return (
          <Input
            id={input.name}
            type="number"
            value={value}
            onChange={(e) => setInputs({ ...inputs, [input.name]: parseFloat(e.target.value) })}
            placeholder={input.description || `Enter ${input.name}...`}
          />
        );

      default:
        return (
          <Input
            id={input.name}
            value={value}
            onChange={(e) => setInputs({ ...inputs, [input.name]: e.target.value })}
            placeholder={input.description || `Enter ${input.name}...`}
          />
        );
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading skill...</p>
        </div>
      </div>
    );
  }

  if (error || !skill) {
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
            <p>{error || "Skill not found"}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const brandColor = skill.branding?.primaryColor || "#3b82f6";

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted p-4 md:p-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          {skill.branding?.logo ? (
            <img
              src={skill.branding.logo}
              alt={skill.branding.companyName || "Logo"}
              className="h-12 mx-auto mb-4"
            />
          ) : (
            <div
              className="h-16 w-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
              style={{ backgroundColor: `${brandColor}20` }}
            >
              <div style={{ color: brandColor }}>
                {skillIcons[skill.icon] || <Zap className="h-8 w-8" />}
              </div>
            </div>
          )}
          <h1 className="text-2xl font-bold mb-2">{skill.name}</h1>
          {skill.description && (
            <p className="text-muted-foreground">{skill.description}</p>
          )}
          <Badge variant="secondary" className="mt-2">
            {skill.category}
          </Badge>
        </div>

        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Input</CardTitle>
            <CardDescription>
              Fill in the required fields to run this skill
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {skill.inputs.map((input) => (
              <div key={input.name} className="space-y-2">
                <Label htmlFor={input.name} className="flex items-center gap-2">
                  {input.name.charAt(0).toUpperCase() + input.name.slice(1).replace(/_/g, " ")}
                  {input.required && <span className="text-destructive">*</span>}
                </Label>
                {input.description && (
                  <p className="text-xs text-muted-foreground">{input.description}</p>
                )}
                {renderInput(input)}
              </div>
            ))}
          </CardContent>
          <CardFooter>
            <Button
              onClick={handleExecute}
              disabled={executing}
              className="w-full"
              style={{ backgroundColor: brandColor }}
            >
              {executing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Skill
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

        {/* Result */}
        {result && (
          <Card className="mt-4">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  Result
                </CardTitle>
                <CardDescription>
                  Completed in {result.execution_time_ms}ms
                </CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={handleCopy}>
                {copied ? (
                  <Check className="h-4 w-4" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </CardHeader>
            <CardContent>
              <div className="bg-muted rounded-lg p-4 overflow-auto max-h-[400px]">
                <pre className="text-sm whitespace-pre-wrap">
                  {typeof result.output === "string"
                    ? result.output
                    : JSON.stringify(result.output, null, 2)}
                </pre>
              </div>
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
