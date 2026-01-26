"use client";

/**
 * AIDocumentIndexer - Agent Builder (Phase 55)
 * =============================================
 *
 * Visual builder for creating Voice and Chat agents.
 *
 * Features:
 * - Voice Agent Builder with TTS configuration
 * - Chat Agent Builder with knowledge base linking
 * - Agent workflow configuration
 * - Testing interface
 */

import * as React from "react";
import {
  Bot,
  Check,
  ChevronRight,
  Code,
  Copy,
  Database,
  ExternalLink,
  FileText,
  Globe,
  Layers,
  Loader2,
  MessageSquare,
  Mic,
  Play,
  Plus,
  Save,
  Settings,
  Share2,
  Sparkles,
  Volume2,
  Wand2,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import {
  Alert,
  AlertDescription,
} from "@/components/ui/alert";

// =============================================================================
// TYPES
// =============================================================================

type AgentType = "voice" | "chat";

interface AgentConfig {
  id?: string;
  name: string;
  description: string;
  type: AgentType;
  systemPrompt: string;
  knowledgeBases: string[];
  // Voice-specific
  voiceProvider?: string;
  voiceId?: string;
  speechSpeed?: number;
  // Chat-specific
  enableStreaming?: boolean;
  maxContextLength?: number;
  // Common
  llmProvider: string;
  llmModel: string;
  temperature: number;
  maxTokens: number;
  enableMemory: boolean;
  // Publishing
  isPublished?: boolean;
  embedToken?: string;
  publishConfig?: {
    allowedDomains: string[];
    rateLimit: number;
    welcomeMessage?: string;
    branding?: {
      primaryColor?: string;
      logoUrl?: string;
      agentDisplayName?: string;
    };
  };
}

interface KnowledgeBase {
  id: string;
  name: string;
  documentCount: number;
  type: string;
}

interface AgentTemplateItem {
  id: string;
  name: string;
  description: string;
  type: string;
  version: string;
  category?: string;
  complexity?: string;
}

interface AgentBuilderProps {
  className?: string;
  initialAgent?: AgentConfig;
  onSave?: (agent: AgentConfig) => void;
  onCancel?: () => void;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

const VOICE_PROVIDERS = [
  { id: "openai", name: "OpenAI TTS", voices: ["alloy", "echo", "fable", "onyx", "nova", "shimmer"] },
  { id: "elevenlabs", name: "ElevenLabs", voices: ["Rachel", "Domi", "Bella", "Antoni", "Josh"] },
  { id: "edge", name: "Edge TTS (Free)", voices: ["en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural"] },
  { id: "cartesia", name: "Cartesia (Fast)", voices: ["sonic-2", "sonic-multilingual"] },
];

const LLM_PROVIDERS = [
  { id: "openai", name: "OpenAI", models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"] },
  { id: "anthropic", name: "Anthropic", models: ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"] },
  { id: "ollama", name: "Ollama (Local)", models: ["llama3.2", "mistral", "mixtral"] },
];

const DEFAULT_VOICE_AGENT: AgentConfig = {
  name: "",
  description: "",
  type: "voice",
  systemPrompt: "You are a helpful voice assistant. Keep responses concise and natural for spoken delivery.",
  knowledgeBases: [],
  voiceProvider: "edge",
  voiceId: "en-US-JennyNeural",
  speechSpeed: 1.0,
  llmProvider: "openai",
  llmModel: "gpt-4o-mini",
  temperature: 0.7,
  maxTokens: 500,
  enableMemory: true,
};

const DEFAULT_CHAT_AGENT: AgentConfig = {
  name: "",
  description: "",
  type: "chat",
  systemPrompt: "You are a helpful AI assistant with access to a knowledge base. Answer questions based on the provided context.",
  knowledgeBases: [],
  enableStreaming: true,
  maxContextLength: 4000,
  llmProvider: "openai",
  llmModel: "gpt-4o",
  temperature: 0.7,
  maxTokens: 2000,
  enableMemory: true,
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function AgentBuilder({
  className,
  initialAgent,
  onSave,
  onCancel,
}: AgentBuilderProps) {
  const [agent, setAgent] = React.useState<AgentConfig>(
    initialAgent || DEFAULT_CHAT_AGENT
  );
  const [knowledgeBases, setKnowledgeBases] = React.useState<KnowledgeBase[]>([]);
  const [templates, setTemplates] = React.useState<AgentTemplateItem[]>([]);
  const [loadingTemplates, setLoadingTemplates] = React.useState(false);
  const [selectedTemplate, setSelectedTemplate] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [testing, setTesting] = React.useState(false);
  const [testInput, setTestInput] = React.useState("");
  const [testOutput, setTestOutput] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);

  // Load knowledge bases and templates
  React.useEffect(() => {
    async function loadKnowledgeBases() {
      try {
        const response = await fetch(`${API_BASE}/collections`, {
          credentials: "include",
        });
        if (response.ok) {
          const data = await response.json();
          setKnowledgeBases(
            data.collections?.map((c: any) => ({
              id: c.name,
              name: c.name,
              documentCount: c.count || 0,
              type: "documents",
            })) || []
          );
        }
      } catch (err) {
        console.error("Failed to load knowledge bases:", err);
      }
    }

    async function loadTemplates() {
      setLoadingTemplates(true);
      try {
        const response = await fetch(`${API_BASE}/v1/agent-templates/agents`, {
          credentials: "include",
        });
        if (response.ok) {
          const data = await response.json();
          setTemplates(data || []);
        }
      } catch (err) {
        console.error("Failed to load templates:", err);
      } finally {
        setLoadingTemplates(false);
      }
    }

    loadKnowledgeBases();
    loadTemplates();
  }, []);

  // Load template configuration
  const loadFromTemplate = async (templateId: string) => {
    setLoadingTemplates(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/v1/agent-templates/agents/${templateId}`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error("Failed to load template");

      const templateData = await response.json();

      // Map template to agent config
      const newAgent: AgentConfig = {
        name: agent.name || templateData.name,
        description: templateData.description,
        type: templateData.type === "voice" ? "voice" : "chat",
        systemPrompt: templateData.system_prompt || "",
        knowledgeBases: [],
        llmProvider: templateData.llm_config?.provider || "openai",
        llmModel: templateData.llm_config?.model || "gpt-4o",
        temperature: templateData.llm_config?.temperature || 0.7,
        maxTokens: templateData.llm_config?.max_tokens || 2000,
        enableMemory: templateData.capabilities?.memory ?? true,
        // Voice-specific
        voiceProvider: templateData.voice_config?.provider || "edge",
        voiceId: templateData.voice_config?.voice_id || "en-US-JennyNeural",
        speechSpeed: templateData.voice_config?.speed || 1.0,
        // Chat-specific
        enableStreaming: templateData.capabilities?.streaming ?? true,
        maxContextLength: templateData.llm_config?.max_context || 4000,
      };

      setAgent(newAgent);
      setSelectedTemplate(templateId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load template");
    } finally {
      setLoadingTemplates(false);
    }
  };

  // Handle agent type change
  const handleTypeChange = (type: AgentType) => {
    setAgent(type === "voice" ? { ...DEFAULT_VOICE_AGENT, name: agent.name } : { ...DEFAULT_CHAT_AGENT, name: agent.name });
  };

  // Update agent field
  const updateAgent = <K extends keyof AgentConfig>(key: K, value: AgentConfig[K]) => {
    setAgent((prev) => ({ ...prev, [key]: value }));
  };

  // Handle save
  const handleSave = async () => {
    if (!agent.name.trim()) {
      setError("Please enter an agent name");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      // Build the request payload with proper field mapping
      const payload = {
        name: agent.name,
        description: agent.description,
        agent_type: agent.type === "voice" ? "voice_agent" : "chat_agent",
        agent_mode: agent.type === "voice" ? "voice" : "chat",
        default_temperature: agent.temperature,
        max_tokens: agent.maxTokens,
        default_model: agent.llmModel,
        system_prompt: agent.systemPrompt,
        knowledge_bases: agent.knowledgeBases,
        settings: {
          enable_memory: agent.enableMemory,
          enable_streaming: agent.enableStreaming,
          max_context_length: agent.maxContextLength,
        },
        // Voice agent specific
        ...(agent.type === "voice" && {
          tts_config: {
            provider: agent.voiceProvider || "openai",
            voice_id: agent.voiceId || "alloy",
            speed: agent.speechSpeed || 1.0,
          },
        }),
      };

      const url = agent.id
        ? `${API_BASE}/v1/agent/agents/${agent.id}`
        : `${API_BASE}/v1/agent/agents`;

      const response = await fetch(url, {
        method: agent.id ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to save agent");
      }

      const savedAgent = await response.json();

      // Map response back to AgentConfig format
      const mappedAgent: AgentConfig = {
        ...agent,
        id: savedAgent.id,
        isPublished: savedAgent.is_published,
        embedToken: savedAgent.embed_token,
      };

      onSave?.(mappedAgent);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save agent");
    } finally {
      setSaving(false);
    }
  };

  // Handle test
  const handleTest = async () => {
    if (!testInput.trim()) return;

    setTesting(true);
    setTestOutput("");

    try {
      // Use the agent execute endpoint for testing
      const response = await fetch(`${API_BASE}/v1/agent/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          message: testInput,
          mode: "agent",
          context: {
            agent_config: {
              type: agent.type,
              system_prompt: agent.systemPrompt,
              temperature: agent.temperature,
              max_tokens: agent.maxTokens,
              knowledge_bases: agent.knowledgeBases,
            },
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Test failed");
      }

      const result = await response.json();
      setTestOutput(result.output || result.response || "No response");
    } catch (err) {
      setTestOutput(`Error: ${err instanceof Error ? err.message : "Test failed"}`);
    } finally {
      setTesting(false);
    }
  };

  // Get current provider's voices/models
  const currentVoiceProvider = VOICE_PROVIDERS.find((p) => p.id === agent.voiceProvider);
  const currentLLMProvider = LLM_PROVIDERS.find((p) => p.id === agent.llmProvider);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Bot className="h-6 w-6" />
            {agent.id ? "Edit Agent" : "Create Agent"}
          </h2>
          <p className="text-muted-foreground mt-1">
            Build a custom {agent.type === "voice" ? "voice" : "chat"} agent with knowledge base access
          </p>
        </div>
        <div className="flex gap-2">
          {onCancel && (
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <Button onClick={handleSave} disabled={saving} className="gap-2">
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
            Save Agent
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Template Selection */}
      {templates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wand2 className="h-5 w-5" />
              Start from Template
            </CardTitle>
            <CardDescription>
              Choose a pre-configured template or start from scratch
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              <button
                type="button"
                onClick={() => {
                  setSelectedTemplate(null);
                  setAgent(DEFAULT_CHAT_AGENT);
                }}
                className={cn(
                  "p-4 rounded-lg border-2 text-left transition-all",
                  !selectedTemplate
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                )}
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-gray-100">
                    <Plus className="h-5 w-5 text-gray-600" />
                  </div>
                  <div>
                    <div className="font-medium">Start Fresh</div>
                    <div className="text-xs text-muted-foreground">
                      Build from scratch
                    </div>
                  </div>
                </div>
              </button>

              {templates.map((template) => (
                <button
                  key={template.id}
                  type="button"
                  onClick={() => loadFromTemplate(template.id)}
                  disabled={loadingTemplates}
                  className={cn(
                    "p-4 rounded-lg border-2 text-left transition-all",
                    selectedTemplate === template.id
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  )}
                >
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "p-2 rounded-lg",
                      template.type === "voice" ? "bg-purple-100" : "bg-blue-100"
                    )}>
                      {template.type === "voice" ? (
                        <Mic className="h-5 w-5 text-purple-600" />
                      ) : (
                        <MessageSquare className="h-5 w-5 text-blue-600" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{template.name}</div>
                      <div className="text-xs text-muted-foreground truncate">
                        {template.description}
                      </div>
                    </div>
                    {selectedTemplate === template.id && (
                      <Check className="h-5 w-5 text-primary shrink-0" />
                    )}
                  </div>
                  <div className="flex gap-1 mt-2">
                    <Badge variant="outline" className="text-xs">
                      {template.type}
                    </Badge>
                    {template.complexity && (
                      <Badge variant="secondary" className="text-xs">
                        {template.complexity}
                      </Badge>
                    )}
                  </div>
                </button>
              ))}
            </div>

            {loadingTemplates && (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                <span className="ml-2 text-sm text-muted-foreground">Loading template...</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Agent Type Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Agent Type</CardTitle>
          <CardDescription>Choose the type of agent you want to create</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <button
              type="button"
              onClick={() => handleTypeChange("chat")}
              className={cn(
                "p-4 rounded-lg border-2 text-left transition-all",
                agent.type === "chat"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              )}
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-100">
                  <MessageSquare className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <div className="font-medium">Chat Agent</div>
                  <div className="text-sm text-muted-foreground">
                    Text-based conversations with knowledge base
                  </div>
                </div>
              </div>
            </button>

            <button
              type="button"
              onClick={() => handleTypeChange("voice")}
              className={cn(
                "p-4 rounded-lg border-2 text-left transition-all",
                agent.type === "voice"
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              )}
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-100">
                  <Mic className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <div className="font-medium">Voice Agent</div>
                  <div className="text-sm text-muted-foreground">
                    Voice-enabled with text-to-speech
                  </div>
                </div>
              </div>
            </button>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Basic Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Basic Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Agent Name</Label>
              <Input
                id="name"
                value={agent.name}
                onChange={(e) => updateAgent("name", e.target.value)}
                placeholder="My Knowledge Assistant"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                value={agent.description}
                onChange={(e) => updateAgent("description", e.target.value)}
                placeholder="A helpful assistant for..."
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="systemPrompt">System Prompt</Label>
              <Textarea
                id="systemPrompt"
                value={agent.systemPrompt}
                onChange={(e) => updateAgent("systemPrompt", e.target.value)}
                rows={4}
                placeholder="You are a helpful assistant..."
              />
            </div>
          </CardContent>
        </Card>

        {/* Knowledge Base Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Knowledge Bases
            </CardTitle>
            <CardDescription>
              Select collections for the agent to search
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {knowledgeBases.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No knowledge bases available. Upload documents first.
                </p>
              ) : (
                knowledgeBases.map((kb) => (
                  <div
                    key={kb.id}
                    className={cn(
                      "flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors",
                      agent.knowledgeBases.includes(kb.id)
                        ? "border-primary bg-primary/5"
                        : "hover:bg-muted/50"
                    )}
                    onClick={() => {
                      const isSelected = agent.knowledgeBases.includes(kb.id);
                      updateAgent(
                        "knowledgeBases",
                        isSelected
                          ? agent.knowledgeBases.filter((id) => id !== kb.id)
                          : [...agent.knowledgeBases, kb.id]
                      );
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="font-medium">{kb.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {kb.documentCount} documents
                        </div>
                      </div>
                    </div>
                    {agent.knowledgeBases.includes(kb.id) && (
                      <Check className="h-5 w-5 text-primary" />
                    )}
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* LLM Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5" />
              LLM Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Provider</Label>
                <Select
                  value={agent.llmProvider}
                  onValueChange={(v) => {
                    const provider = LLM_PROVIDERS.find((p) => p.id === v);
                    updateAgent("llmProvider", v);
                    if (provider) {
                      updateAgent("llmModel", provider.models[0]);
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {LLM_PROVIDERS.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Model</Label>
                <Select
                  value={agent.llmModel}
                  onValueChange={(v) => updateAgent("llmModel", v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {currentLLMProvider?.models.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Temperature</Label>
                <span className="text-sm text-muted-foreground">{agent.temperature}</span>
              </div>
              <Slider
                value={[agent.temperature]}
                onValueChange={([v]) => updateAgent("temperature", v)}
                min={0}
                max={2}
                step={0.1}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Max Tokens</Label>
                <span className="text-sm text-muted-foreground">{agent.maxTokens}</span>
              </div>
              <Slider
                value={[agent.maxTokens]}
                onValueChange={([v]) => updateAgent("maxTokens", v)}
                min={100}
                max={4000}
                step={100}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label>Enable Memory</Label>
                <p className="text-xs text-muted-foreground">
                  Remember conversation context
                </p>
              </div>
              <Switch
                checked={agent.enableMemory}
                onCheckedChange={(v) => updateAgent("enableMemory", v)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Voice Configuration (Voice Agent Only) */}
        {agent.type === "voice" && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Volume2 className="h-5 w-5" />
                Voice Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Voice Provider</Label>
                  <Select
                    value={agent.voiceProvider}
                    onValueChange={(v) => {
                      const provider = VOICE_PROVIDERS.find((p) => p.id === v);
                      updateAgent("voiceProvider", v);
                      if (provider) {
                        updateAgent("voiceId", provider.voices[0]);
                      }
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {VOICE_PROVIDERS.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Voice</Label>
                  <Select
                    value={agent.voiceId}
                    onValueChange={(v) => updateAgent("voiceId", v)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {currentVoiceProvider?.voices.map((v) => (
                        <SelectItem key={v} value={v}>
                          {v}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Speech Speed</Label>
                  <span className="text-sm text-muted-foreground">{agent.speechSpeed}x</span>
                </div>
                <Slider
                  value={[agent.speechSpeed || 1.0]}
                  onValueChange={([v]) => updateAgent("speechSpeed", v)}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Chat Configuration (Chat Agent Only) */}
        {agent.type === "chat" && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                Chat Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Enable Streaming</Label>
                  <p className="text-xs text-muted-foreground">
                    Stream responses in real-time
                  </p>
                </div>
                <Switch
                  checked={agent.enableStreaming}
                  onCheckedChange={(v) => updateAgent("enableStreaming", v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Max Context Length</Label>
                  <span className="text-sm text-muted-foreground">{agent.maxContextLength} tokens</span>
                </div>
                <Slider
                  value={[agent.maxContextLength || 4000]}
                  onValueChange={([v]) => updateAgent("maxContextLength", v)}
                  min={1000}
                  max={16000}
                  step={500}
                />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Publish Section - Only show for saved agents */}
        {agent.id && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Share2 className="h-5 w-5" />
                Publish & Embed
              </CardTitle>
              <CardDescription>
                Deploy your agent as an embeddable widget for external use
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {agent.isPublished ? (
                <>
                  <div className="flex items-center gap-2 text-green-600">
                    <Check className="h-4 w-4" />
                    <span className="font-medium">Agent is published</span>
                  </div>

                  {/* Embed Code */}
                  <div className="space-y-2">
                    <Label>Embed Code</Label>
                    <div className="relative">
                      <pre className="p-3 bg-muted rounded-lg text-xs overflow-x-auto">
{`<script src="${typeof window !== 'undefined' ? window.location.origin : ''}/static/embed/agent-widget.js"></script>
<script>
  AIAgent.init({
    token: "${agent.embedToken}",
    position: "bottom-right",
    theme: "light"
  });
</script>`}
                      </pre>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="absolute top-2 right-2"
                        onClick={() => {
                          navigator.clipboard.writeText(`<script src="${window.location.origin}/static/embed/agent-widget.js"></script>
<script>
  AIAgent.init({
    token: "${agent.embedToken}",
    position: "bottom-right",
    theme: "light"
  });
</script>`);
                        }}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  {/* Widget URL */}
                  <div className="space-y-2">
                    <Label>Direct Widget URL</Label>
                    <div className="flex gap-2">
                      <Input
                        value={`${typeof window !== 'undefined' ? window.location.origin : ''}/embed/chat/${agent.embedToken}`}
                        readOnly
                        className="text-sm"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => window.open(`/embed/chat/${agent.embedToken}`, '_blank')}
                      >
                        <ExternalLink className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  {/* Publish Settings */}
                  <Separator />
                  <div className="space-y-3">
                    <Label>Settings</Label>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label className="text-xs text-muted-foreground">Allowed Domains</Label>
                        <p className="text-sm">{agent.publishConfig?.allowedDomains?.join(', ') || '*'}</p>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Rate Limit</Label>
                        <p className="text-sm">{agent.publishConfig?.rateLimit || 100} req/hour</p>
                      </div>
                    </div>
                  </div>

                  <Button
                    variant="destructive"
                    className="w-full"
                    onClick={async () => {
                      try {
                        const response = await fetch(`/api/v1/agent/agents/${agent.id}/publish`, {
                          method: 'DELETE',
                        });
                        if (response.ok) {
                          updateAgent('isPublished', false);
                          updateAgent('embedToken', undefined);
                        }
                      } catch (err) {
                        console.error('Failed to unpublish:', err);
                      }
                    }}
                  >
                    Unpublish Agent
                  </Button>
                </>
              ) : (
                <PublishAgentDialog
                  agentId={agent.id}
                  agentName={agent.name}
                  agentType={agent.type}
                  onPublished={(result) => {
                    updateAgent('isPublished', true);
                    updateAgent('embedToken', result.embedToken);
                    updateAgent('publishConfig', {
                      allowedDomains: ['*'],
                      rateLimit: 100,
                    });
                  }}
                />
              )}
            </CardContent>
          </Card>
        )}
      </div>

      {/* Test Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-5 w-5" />
            Test Agent
          </CardTitle>
          <CardDescription>
            Try out your agent configuration before saving
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              value={testInput}
              onChange={(e) => setTestInput(e.target.value)}
              placeholder="Enter a test message..."
              onKeyDown={(e) => e.key === "Enter" && handleTest()}
            />
            <Button onClick={handleTest} disabled={testing || !testInput.trim()}>
              {testing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            </Button>
          </div>

          {testOutput && (
            <div className="p-4 rounded-lg bg-muted">
              <p className="text-sm whitespace-pre-wrap">{testOutput}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// =============================================================================
// PUBLISH AGENT DIALOG
// =============================================================================

interface PublishAgentDialogProps {
  agentId: string;
  agentName: string;
  agentType: AgentType;
  onPublished: (result: { embedToken: string; embedCode: string; widgetUrl: string }) => void;
}

function PublishAgentDialog({ agentId, agentName, agentType, onPublished }: PublishAgentDialogProps) {
  const [open, setOpen] = React.useState(false);
  const [publishing, setPublishing] = React.useState(false);
  const [config, setConfig] = React.useState({
    allowedDomains: "*",
    rateLimit: 100,
    welcomeMessage: `Hi! I'm ${agentName}. How can I help you today?`,
    primaryColor: "#6366f1",
  });

  const handlePublish = async () => {
    setPublishing(true);
    try {
      const response = await fetch(`/api/v1/agent/agents/${agentId}/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          allowed_domains: config.allowedDomains.split(',').map(d => d.trim()),
          rate_limit: config.rateLimit,
          welcome_message: config.welcomeMessage,
          branding: {
            primary_color: config.primaryColor,
            agent_display_name: agentName,
          },
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to publish agent');
      }

      const result = await response.json();
      onPublished({
        embedToken: result.embed_token,
        embedCode: result.embed_code,
        widgetUrl: result.widget_url,
      });
      setOpen(false);
    } catch (err) {
      console.error('Publish error:', err);
    } finally {
      setPublishing(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="w-full gap-2">
          <Globe className="h-4 w-4" />
          Publish Agent
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Share2 className="h-5 w-5" />
            Publish {agentName}
          </DialogTitle>
          <DialogDescription>
            Make your {agentType} agent available as an embeddable widget
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Allowed Domains */}
          <div className="space-y-2">
            <Label>Allowed Domains</Label>
            <Input
              value={config.allowedDomains}
              onChange={(e) => setConfig({ ...config, allowedDomains: e.target.value })}
              placeholder="* (all domains) or example.com, app.example.com"
            />
            <p className="text-xs text-muted-foreground">
              Comma-separated list of domains that can embed this agent. Use * for all domains.
            </p>
          </div>

          {/* Rate Limit */}
          <div className="space-y-2">
            <Label>Rate Limit (requests/hour)</Label>
            <Input
              type="number"
              value={config.rateLimit}
              onChange={(e) => setConfig({ ...config, rateLimit: parseInt(e.target.value) || 100 })}
              min={1}
              max={10000}
            />
          </div>

          {/* Welcome Message */}
          <div className="space-y-2">
            <Label>Welcome Message</Label>
            <Textarea
              value={config.welcomeMessage}
              onChange={(e) => setConfig({ ...config, welcomeMessage: e.target.value })}
              placeholder="Hi! How can I help you?"
              rows={2}
            />
          </div>

          {/* Primary Color */}
          <div className="space-y-2">
            <Label>Widget Color</Label>
            <div className="flex gap-2">
              <Input
                type="color"
                value={config.primaryColor}
                onChange={(e) => setConfig({ ...config, primaryColor: e.target.value })}
                className="w-16 h-10 p-1"
              />
              <Input
                value={config.primaryColor}
                onChange={(e) => setConfig({ ...config, primaryColor: e.target.value })}
                placeholder="#6366f1"
              />
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handlePublish} disabled={publishing} className="gap-2">
            {publishing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Globe className="h-4 w-4" />
            )}
            Publish Agent
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// AGENT BUILDER MODAL
// =============================================================================

interface AgentBuilderModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialAgent?: AgentConfig;
  onSave?: (agent: AgentConfig) => void;
}

export function AgentBuilderModal({
  open,
  onOpenChange,
  initialAgent,
  onSave,
}: AgentBuilderModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <AgentBuilder
          initialAgent={initialAgent}
          onSave={(agent) => {
            onSave?.(agent);
            onOpenChange(false);
          }}
          onCancel={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// CREATE AGENT BUTTON
// =============================================================================

interface CreateAgentButtonProps {
  onAgentCreated?: (agent: AgentConfig) => void;
}

export function CreateAgentButton({ onAgentCreated }: CreateAgentButtonProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <>
      <Button onClick={() => setOpen(true)} className="gap-2">
        <Plus className="h-4 w-4" />
        Create Agent
      </Button>

      <AgentBuilderModal
        open={open}
        onOpenChange={setOpen}
        onSave={onAgentCreated}
      />
    </>
  );
}

export default AgentBuilder;
