"use client";

/**
 * AIDocumentIndexer - External Services Documentation (Phase 53)
 * ==============================================================
 *
 * Shows users which features require external services/APIs and their pricing.
 *
 * Features:
 * - Clear documentation of required API keys per feature
 * - Pricing information
 * - Service status indicators (configured/not configured)
 * - Fallback chain visualization
 */

import * as React from "react";
import {
  AlertCircle,
  Check,
  ChevronRight,
  Cloud,
  Cpu,
  Database,
  ExternalLink,
  HelpCircle,
  Key,
  Mic,
  Search,
  Settings,
  Sparkles,
  X,
  Zap,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";

// =============================================================================
// TYPES
// =============================================================================

interface ExternalService {
  name: string;
  envKey: string;
  required: boolean;
  features: string[];
  pricing: string;
  pricingUrl?: string;
  docUrl?: string;
  status?: "configured" | "not_configured" | "optional";
}

interface ServiceCategory {
  name: string;
  description: string;
  icon: React.ElementType;
  services: ExternalService[];
  fallbackChain?: string[];
}

interface ExternalServicesProps {
  className?: string;
  configuredKeys?: string[];
}

// =============================================================================
// SERVICE DEFINITIONS
// =============================================================================

const SERVICE_CATEGORIES: ServiceCategory[] = [
  {
    name: "LLM Providers",
    description: "Language model providers for chat and generation",
    icon: Sparkles,
    services: [
      {
        name: "OpenAI",
        envKey: "OPENAI_API_KEY",
        required: false,
        features: ["Chat (GPT-4o)", "Embeddings", "TTS"],
        pricing: "Pay-per-token (~$2.50/1M tokens for GPT-4o)",
        pricingUrl: "https://openai.com/pricing",
        docUrl: "https://platform.openai.com/docs",
      },
      {
        name: "Anthropic",
        envKey: "ANTHROPIC_API_KEY",
        required: false,
        features: ["Chat (Claude)", "Vision Analysis", "RLM (10M+ context)"],
        pricing: "Pay-per-token (~$3/1M tokens for Claude 3.5 Sonnet)",
        pricingUrl: "https://anthropic.com/pricing",
        docUrl: "https://docs.anthropic.com",
      },
      {
        name: "Google AI",
        envKey: "GOOGLE_API_KEY",
        required: false,
        features: ["Chat (Gemini)"],
        pricing: "Pay-per-token (free tier available)",
        pricingUrl: "https://ai.google.dev/pricing",
        docUrl: "https://ai.google.dev/docs",
      },
      {
        name: "Groq",
        envKey: "GROQ_API_KEY",
        required: false,
        features: ["Fast Inference (Llama 3.1)"],
        pricing: "Free tier available, then pay-per-token",
        pricingUrl: "https://groq.com",
        docUrl: "https://console.groq.com/docs",
      },
      {
        name: "Ollama",
        envKey: null as any,
        required: false,
        features: ["Local Chat", "Local Embeddings", "Privacy"],
        pricing: "Free (runs locally, requires GPU)",
        docUrl: "https://ollama.ai",
      },
    ],
    fallbackChain: ["OpenAI", "Anthropic", "Google AI", "Groq", "Ollama"],
  },
  {
    name: "Embedding Providers",
    description: "Vector embedding models for document search",
    icon: Database,
    services: [
      {
        name: "OpenAI Embeddings",
        envKey: "OPENAI_API_KEY",
        required: false,
        features: ["text-embedding-3-large", "text-embedding-3-small"],
        pricing: "$0.13/1M tokens (large), $0.02/1M (small)",
        pricingUrl: "https://openai.com/pricing",
      },
      {
        name: "Voyage AI",
        envKey: "VOYAGE_API_KEY",
        required: false,
        features: ["voyage-3-large (Best quality)", "voyage-3-lite"],
        pricing: "$0.06/1M tokens",
        pricingUrl: "https://www.voyageai.com/pricing",
        docUrl: "https://docs.voyageai.com",
      },
      {
        name: "Cohere",
        envKey: "COHERE_API_KEY",
        required: false,
        features: ["embed-v3", "Reranking"],
        pricing: "Free tier (100K tokens/month), then $0.10/1M",
        pricingUrl: "https://cohere.com/pricing",
        docUrl: "https://docs.cohere.com",
      },
      {
        name: "Jina AI",
        envKey: "JINA_API_KEY",
        required: false,
        features: ["jina-embeddings-v3", "Multimodal"],
        pricing: "Free tier available",
        pricingUrl: "https://jina.ai/embeddings",
        docUrl: "https://jina.ai/embeddings",
      },
    ],
    fallbackChain: ["OpenAI", "Voyage AI", "Cohere", "Jina AI", "Local Model"],
  },
  {
    name: "Text-to-Speech",
    description: "Audio generation for document overviews",
    icon: Mic,
    services: [
      {
        name: "Cartesia",
        envKey: "CARTESIA_API_KEY",
        required: false,
        features: ["Sonic 2.0 (Ultra-fast, 40ms TTFA)"],
        pricing: "Pay-per-character",
        pricingUrl: "https://cartesia.ai/pricing",
        docUrl: "https://docs.cartesia.ai",
      },
      {
        name: "ElevenLabs",
        envKey: "ELEVENLABS_API_KEY",
        required: false,
        features: ["Flash v2.5 (75ms TTFB)", "Voice Cloning", "70+ Languages"],
        pricing: "Pay-per-character ($0.30/1000 chars)",
        pricingUrl: "https://elevenlabs.io/pricing",
        docUrl: "https://elevenlabs.io/docs",
      },
      {
        name: "OpenAI TTS",
        envKey: "OPENAI_API_KEY",
        required: false,
        features: ["TTS-1", "TTS-1-HD"],
        pricing: "$15/1M characters",
        pricingUrl: "https://openai.com/pricing",
      },
    ],
    fallbackChain: ["Cartesia", "ElevenLabs", "OpenAI TTS"],
  },
  {
    name: "Vision & OCR",
    description: "Document scanning and visual understanding",
    icon: Search,
    services: [
      {
        name: "Claude Vision",
        envKey: "ANTHROPIC_API_KEY",
        required: false,
        features: ["Chart Analysis", "Table Extraction", "VLM Processing"],
        pricing: "Included with Claude API usage",
      },
      {
        name: "OpenAI Vision",
        envKey: "OPENAI_API_KEY",
        required: false,
        features: ["GPT-4o Vision", "Image Analysis"],
        pricing: "Included with GPT-4o API usage",
      },
      {
        name: "Surya OCR",
        envKey: null as any,
        required: false,
        features: ["97.7% Accuracy", "90+ Languages", "Local Processing"],
        pricing: "Free (open source, runs locally)",
        docUrl: "https://github.com/VikParuchuri/surya",
      },
    ],
    fallbackChain: ["Surya OCR", "Claude Vision", "Tesseract"],
  },
  {
    name: "Web & Integrations",
    description: "External content and workspace integrations",
    icon: Cloud,
    services: [
      {
        name: "Firecrawl",
        envKey: "FIRECRAWL_API_KEY",
        required: false,
        features: ["Web Scraping", "URL Import"],
        pricing: "Free tier (500 pages/month)",
        pricingUrl: "https://firecrawl.dev/pricing",
        docUrl: "https://docs.firecrawl.dev",
      },
      {
        name: "Google OAuth",
        envKey: "GOOGLE_CLIENT_ID",
        required: false,
        features: ["SSO Login", "Google Drive Import"],
        pricing: "Free",
        docUrl: "https://developers.google.com/identity",
      },
      {
        name: "Notion",
        envKey: "NOTION_INTEGRATION_TOKEN",
        required: false,
        features: ["Workspace Import"],
        pricing: "Free (Notion API)",
        docUrl: "https://developers.notion.com",
      },
      {
        name: "Slack",
        envKey: "SLACK_BOT_TOKEN",
        required: false,
        features: ["Notifications", "Bot Integration"],
        pricing: "Free (Slack API)",
        docUrl: "https://api.slack.com",
      },
    ],
  },
  {
    name: "Infrastructure",
    description: "Backend services and distributed computing",
    icon: Cpu,
    services: [
      {
        name: "Redis",
        envKey: "REDIS_URL",
        required: true,
        features: ["Caching", "Task Queue", "Session Storage"],
        pricing: "Self-hosted free, or cloud ($5-50/month)",
        docUrl: "https://redis.io/docs",
      },
      {
        name: "PostgreSQL",
        envKey: "DATABASE_URL",
        required: true,
        features: ["Document Storage", "User Data", "Audit Logs"],
        pricing: "Self-hosted free, or cloud ($5-50/month)",
        docUrl: "https://postgresql.org/docs",
      },
      {
        name: "Ray Cluster",
        envKey: "RAY_ADDRESS",
        required: false,
        features: ["Distributed ML", "Parallel Processing", "10x Throughput"],
        pricing: "Self-hosted free",
        docUrl: "https://docs.ray.io",
      },
      {
        name: "Sentry",
        envKey: "SENTRY_DSN",
        required: false,
        features: ["Error Tracking", "Performance Monitoring"],
        pricing: "Free tier (5K errors/month)",
        pricingUrl: "https://sentry.io/pricing",
        docUrl: "https://docs.sentry.io",
      },
    ],
  },
];

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExternalServicesPanel({ className, configuredKeys = [] }: ExternalServicesProps) {
  // Simulate checking which keys are configured
  const [configuredServices, setConfiguredServices] = React.useState<Set<string>>(
    new Set(configuredKeys)
  );

  // In a real app, this would fetch from the backend
  React.useEffect(() => {
    // Mock: Check which services are configured
    const mockConfigured = new Set([
      "OPENAI_API_KEY",
      "ANTHROPIC_API_KEY",
      "REDIS_URL",
      "DATABASE_URL",
    ]);
    setConfiguredServices(mockConfigured);
  }, []);

  const getServiceStatus = (service: ExternalService): "configured" | "not_configured" | "optional" => {
    if (!service.envKey) return "optional"; // No key needed (e.g., local services)
    if (configuredServices.has(service.envKey)) return "configured";
    return service.required ? "not_configured" : "optional";
  };

  const requiredMissing = SERVICE_CATEGORIES.flatMap((cat) =>
    cat.services.filter(
      (s) => s.required && s.envKey && !configuredServices.has(s.envKey)
    )
  );

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">External Services</h2>
        <p className="text-muted-foreground mt-1">
          Configure API keys and understand which features require external services
        </p>
      </div>

      {/* Required Services Alert */}
      {requiredMissing.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Required Services Missing</AlertTitle>
          <AlertDescription>
            The following required services are not configured:{" "}
            {requiredMissing.map((s) => s.name).join(", ")}
          </AlertDescription>
        </Alert>
      )}

      {/* Quick Status Overview */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Configuration Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {SERVICE_CATEGORIES.map((category) => {
              const configured = category.services.filter(
                (s) => getServiceStatus(s) === "configured"
              ).length;
              const total = category.services.length;
              const hasRequired = category.services.some((s) => s.required);
              const missingRequired = category.services.some(
                (s) => s.required && getServiceStatus(s) === "not_configured"
              );

              return (
                <div
                  key={category.name}
                  className={cn(
                    "p-3 rounded-lg border",
                    missingRequired && "border-red-300 bg-red-50",
                    !missingRequired && configured > 0 && "border-green-300 bg-green-50",
                    !missingRequired && configured === 0 && "border-gray-200 bg-gray-50"
                  )}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <category.icon className="h-4 w-4" />
                    <span className="text-sm font-medium">{category.name}</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {configured}/{total} configured
                    {hasRequired && (
                      <Badge
                        variant={missingRequired ? "destructive" : "outline"}
                        className="ml-2 text-xs"
                      >
                        {missingRequired ? "Required!" : "OK"}
                      </Badge>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Service Categories */}
      <Accordion type="multiple" className="space-y-4" defaultValue={["LLM Providers"]}>
        {SERVICE_CATEGORIES.map((category) => (
          <AccordionItem
            key={category.name}
            value={category.name}
            className="border rounded-lg px-4"
          >
            <AccordionTrigger className="hover:no-underline">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <category.icon className="h-5 w-5 text-primary" />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">{category.name}</h3>
                  <p className="text-sm text-muted-foreground">{category.description}</p>
                </div>
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 pt-4">
                {/* Fallback Chain */}
                {category.fallbackChain && (
                  <div className="p-3 rounded-lg bg-muted/50 mb-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="h-4 w-4 text-amber-500" />
                      <span className="text-sm font-medium">Automatic Fallback Chain</span>
                    </div>
                    <div className="flex items-center gap-1 text-sm text-muted-foreground flex-wrap">
                      {category.fallbackChain.map((provider, i) => (
                        <React.Fragment key={provider}>
                          <span
                            className={cn(
                              "px-2 py-0.5 rounded",
                              configuredServices.has(
                                category.services.find((s) => s.name === provider)?.envKey || ""
                              )
                                ? "bg-green-100 text-green-800"
                                : "bg-gray-100"
                            )}
                          >
                            {provider}
                          </span>
                          {i < category.fallbackChain!.length - 1 && (
                            <ChevronRight className="h-3 w-3" />
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      If the primary provider fails, the system automatically tries the next one.
                      Fallback events are logged and can be monitored in the{" "}
                      <a href="/admin" className="text-primary hover:underline">Admin Log Viewer</a>.
                    </p>
                  </div>
                )}

                {/* Services Grid */}
                <div className="grid gap-4 md:grid-cols-2">
                  {category.services.map((service) => {
                    const status = getServiceStatus(service);

                    return (
                      <Card
                        key={service.name}
                        className={cn(
                          "relative overflow-hidden",
                          status === "configured" && "ring-1 ring-green-500",
                          status === "not_configured" && service.required && "ring-1 ring-red-500"
                        )}
                      >
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-base flex items-center gap-2">
                              {service.name}
                              {service.required && (
                                <Badge variant="destructive" className="text-xs">
                                  Required
                                </Badge>
                              )}
                            </CardTitle>
                            <StatusBadge status={status} />
                          </div>
                          {service.envKey && (
                            <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                              {service.envKey}
                            </code>
                          )}
                        </CardHeader>
                        <CardContent className="space-y-3">
                          {/* Features */}
                          <div>
                            <p className="text-xs font-medium text-muted-foreground mb-1">
                              Features
                            </p>
                            <div className="flex flex-wrap gap-1">
                              {service.features.map((feature) => (
                                <Badge key={feature} variant="secondary" className="text-xs">
                                  {feature}
                                </Badge>
                              ))}
                            </div>
                          </div>

                          {/* Pricing */}
                          <div>
                            <p className="text-xs font-medium text-muted-foreground mb-1">
                              Pricing
                            </p>
                            <p className="text-sm">{service.pricing}</p>
                          </div>

                          {/* Links */}
                          <div className="flex gap-2 pt-2">
                            {service.docUrl && (
                              <Button
                                variant="outline"
                                size="sm"
                                className="gap-1 text-xs"
                                asChild
                              >
                                <a
                                  href={service.docUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                >
                                  Docs
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              </Button>
                            )}
                            {service.pricingUrl && (
                              <Button
                                variant="outline"
                                size="sm"
                                className="gap-1 text-xs"
                                asChild
                              >
                                <a
                                  href={service.pricingUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                >
                                  Pricing
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              </Button>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>

      {/* Help Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <HelpCircle className="h-5 w-5" />
            Need Help?
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Most features work with just <strong>OpenAI</strong> or <strong>Anthropic</strong> API
            keys. The system will automatically use fallback providers if one is unavailable.
          </p>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="p-4 rounded-lg border">
              <h4 className="font-medium mb-2">Minimal Setup</h4>
              <p className="text-sm text-muted-foreground mb-2">
                For basic functionality:
              </p>
              <ul className="text-sm space-y-1">
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  OPENAI_API_KEY
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  REDIS_URL
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  DATABASE_URL
                </li>
              </ul>
            </div>
            <div className="p-4 rounded-lg border">
              <h4 className="font-medium mb-2">Recommended Setup</h4>
              <p className="text-sm text-muted-foreground mb-2">
                For best quality:
              </p>
              <ul className="text-sm space-y-1">
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  OPENAI_API_KEY
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  ANTHROPIC_API_KEY
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  VOYAGE_API_KEY
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  CARTESIA_API_KEY
                </li>
              </ul>
            </div>
            <div className="p-4 rounded-lg border">
              <h4 className="font-medium mb-2">Enterprise Setup</h4>
              <p className="text-sm text-muted-foreground mb-2">
                For maximum features:
              </p>
              <ul className="text-sm space-y-1">
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  All recommended keys
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  RAY_ADDRESS (scaling)
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  COHERE_API_KEY
                </li>
                <li className="flex items-center gap-2">
                  <Check className="h-3 w-3 text-green-500" />
                  SENTRY_DSN
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// =============================================================================
// STATUS BADGE
// =============================================================================

function StatusBadge({ status }: { status: "configured" | "not_configured" | "optional" }) {
  if (status === "configured") {
    return (
      <Badge className="bg-green-100 text-green-800 border-green-300 gap-1">
        <Check className="h-3 w-3" />
        Configured
      </Badge>
    );
  }

  if (status === "not_configured") {
    return (
      <Badge variant="destructive" className="gap-1">
        <X className="h-3 w-3" />
        Not Configured
      </Badge>
    );
  }

  return (
    <Badge variant="outline" className="gap-1">
      <HelpCircle className="h-3 w-3" />
      Optional
    </Badge>
  );
}

export default ExternalServicesPanel;
