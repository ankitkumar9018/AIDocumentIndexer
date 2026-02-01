"use client";

/**
 * AIDocumentIndexer - BYOK API Keys Settings Panel
 * =================================================
 *
 * Allows users to configure their own API keys for LLM providers.
 * When a personal key is set, it is used instead of the shared system key.
 *
 * Features:
 * - Per-provider API key management (save, test, remove)
 * - Password-style input with show/hide toggle
 * - Mocked test button to validate keys (simulates API call)
 * - Status badges: Active, Not configured, Invalid
 * - Master "Enable BYOK" toggle
 * - Encrypted localStorage persistence (Base64 for demo)
 * - Security information footer
 *
 * This is a CLIENT-SIDE ONLY implementation. Keys are stored in
 * localStorage with Base64 encoding (not production-grade encryption).
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  Key,
  Check,
  X,
  AlertTriangle,
  Loader2,
  Trash2,
  Eye,
  EyeOff,
  Plus,
  Shield,
  Cpu,
  Brain,
  Sparkles,
  Zap,
  Cloud,
  Users,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatRelativeTime } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { toast } from "sonner";

// =============================================================================
// TYPES
// =============================================================================

interface ProviderConfig {
  id: string;
  name: string;
  placeholder: string;
  icon: React.ElementType;
  description: string;
  docsUrl: string;
  keyPrefix?: string;
}

interface StoredKeyData {
  key: string;
  lastTested: string | null;
  isValid: boolean | null;
}

interface BYOKStorage {
  enabled: boolean;
  keys: Record<string, StoredKeyData | null>;
}

type KeyStatus = "active" | "invalid" | "not_configured" | "untested";

interface ApiKeysPanelProps {
  className?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = "byok_keys";

const PROVIDERS: ProviderConfig[] = [
  {
    id: "openai",
    name: "OpenAI",
    placeholder: "sk-...",
    icon: Cpu,
    description: "GPT-4o, GPT-4o Mini, Embeddings, TTS",
    docsUrl: "https://platform.openai.com/api-keys",
    keyPrefix: "sk-",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    placeholder: "sk-ant-...",
    icon: Brain,
    description: "Claude 3.5 Sonnet, Claude 3.5 Haiku, Vision",
    docsUrl: "https://console.anthropic.com/settings/keys",
    keyPrefix: "sk-ant-",
  },
  {
    id: "google",
    name: "Google AI",
    placeholder: "AIza...",
    icon: Sparkles,
    description: "Gemini Pro, Gemini Flash",
    docsUrl: "https://aistudio.google.com/app/apikey",
    keyPrefix: "AIza",
  },
  {
    id: "mistral",
    name: "Mistral",
    placeholder: "...",
    icon: Cloud,
    description: "Mistral Large, Mistral Medium, Codestral",
    docsUrl: "https://console.mistral.ai/api-keys",
  },
  {
    id: "groq",
    name: "Groq",
    placeholder: "gsk_...",
    icon: Zap,
    description: "Ultra-fast inference for Llama, Mixtral",
    docsUrl: "https://console.groq.com/keys",
    keyPrefix: "gsk_",
  },
  {
    id: "together",
    name: "Together AI",
    placeholder: "...",
    icon: Users,
    description: "Open-source model hosting and fine-tuning",
    docsUrl: "https://api.together.ai/settings/api-keys",
  },
];

// =============================================================================
// STORAGE HELPERS
// =============================================================================

function encodeKey(key: string): string {
  try {
    return btoa(key);
  } catch {
    return key;
  }
}

function decodeKey(encoded: string): string {
  try {
    return atob(encoded);
  } catch {
    return encoded;
  }
}

function loadBYOKStorage(): BYOKStorage {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { enabled: false, keys: {} };
    }
    const parsed = JSON.parse(raw) as BYOKStorage;
    // Decode keys on load
    const decodedKeys: Record<string, StoredKeyData | null> = {};
    for (const [providerId, data] of Object.entries(parsed.keys)) {
      if (data && data.key) {
        decodedKeys[providerId] = {
          ...data,
          key: decodeKey(data.key),
        };
      } else {
        decodedKeys[providerId] = null;
      }
    }
    return { enabled: parsed.enabled, keys: decodedKeys };
  } catch {
    return { enabled: false, keys: {} };
  }
}

function saveBYOKStorage(storage: BYOKStorage): void {
  try {
    // Encode keys before saving
    const encodedKeys: Record<string, StoredKeyData | null> = {};
    for (const [providerId, data] of Object.entries(storage.keys)) {
      if (data && data.key) {
        encodedKeys[providerId] = {
          ...data,
          key: encodeKey(data.key),
        };
      } else {
        encodedKeys[providerId] = null;
      }
    }
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ enabled: storage.enabled, keys: encodedKeys })
    );
  } catch {
    console.error("Failed to save BYOK keys to localStorage");
  }
}

function maskKey(key: string): string {
  if (key.length <= 8) {
    return "*".repeat(key.length);
  }
  const prefix = key.slice(0, 4);
  const suffix = key.slice(-4);
  const masked = "*".repeat(Math.min(key.length - 8, 20));
  return `${prefix}${masked}${suffix}`;
}

function getKeyStatus(data: StoredKeyData | null | undefined): KeyStatus {
  if (!data || !data.key) return "not_configured";
  if (data.isValid === true) return "active";
  if (data.isValid === false) return "invalid";
  return "untested";
}

// =============================================================================
// STATUS BADGE COMPONENT
// =============================================================================

function StatusBadge({ status }: { status: KeyStatus }) {
  switch (status) {
    case "active":
      return (
        <Badge className="bg-green-100 text-green-800 border-green-300 gap-1">
          <Check className="h-3 w-3" />
          Active
        </Badge>
      );
    case "invalid":
      return (
        <Badge variant="destructive" className="gap-1">
          <X className="h-3 w-3" />
          Invalid
        </Badge>
      );
    case "untested":
      return (
        <Badge variant="outline" className="gap-1 border-amber-300 text-amber-700 bg-amber-50">
          <AlertTriangle className="h-3 w-3" />
          Untested
        </Badge>
      );
    case "not_configured":
    default:
      return (
        <Badge variant="outline" className="gap-1 text-muted-foreground">
          <Key className="h-3 w-3" />
          Not configured
        </Badge>
      );
  }
}

// =============================================================================
// PROVIDER CARD COMPONENT
// =============================================================================

interface ProviderCardProps {
  provider: ProviderConfig;
  keyData: StoredKeyData | null | undefined;
  byokEnabled: boolean;
  onSave: (providerId: string, key: string) => void;
  onRemove: (providerId: string) => void;
  onTest: (providerId: string) => void;
  isTesting: boolean;
}

function ProviderCard({
  provider,
  keyData,
  byokEnabled,
  onSave,
  onRemove,
  onTest,
  isTesting,
}: ProviderCardProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [showKey, setShowKey] = useState(false);

  const status = getKeyStatus(keyData);
  const hasKey = status !== "not_configured";
  const IconComponent = provider.icon;

  const handleSave = () => {
    if (!inputValue.trim()) {
      toast.error("Please enter an API key");
      return;
    }
    onSave(provider.id, inputValue.trim());
    setInputValue("");
    setIsEditing(false);
    setShowKey(false);
  };

  const handleCancel = () => {
    setInputValue("");
    setIsEditing(false);
    setShowKey(false);
  };

  const handleRemove = () => {
    onRemove(provider.id);
    setInputValue("");
    setIsEditing(false);
    setShowKey(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSave();
    } else if (e.key === "Escape") {
      handleCancel();
    }
  };

  return (
    <Card
      className={cn(
        "transition-all duration-200",
        !byokEnabled && "opacity-60 pointer-events-none",
        status === "active" && "ring-1 ring-green-300",
        status === "invalid" && "ring-1 ring-red-300"
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          {/* Provider Info */}
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <div className="p-2 rounded-lg bg-primary/10 shrink-0">
              <IconComponent className="h-5 w-5 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-semibold text-sm">{provider.name}</h3>
                <StatusBadge status={status} />
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                {provider.description}
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 shrink-0">
            {hasKey && !isEditing && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onTest(provider.id)}
                  disabled={isTesting || !byokEnabled}
                  className="gap-1 text-xs"
                >
                  {isTesting ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <Zap className="h-3 w-3" />
                  )}
                  Test
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRemove}
                  disabled={!byokEnabled}
                  className="gap-1 text-xs text-destructive hover:text-destructive"
                >
                  <Trash2 className="h-3 w-3" />
                  Remove
                </Button>
              </>
            )}
            {!hasKey && !isEditing && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsEditing(true)}
                disabled={!byokEnabled}
                className="gap-1 text-xs"
              >
                <Plus className="h-3 w-3" />
                Add Key
              </Button>
            )}
          </div>
        </div>

        {/* Key Display (when saved and not editing) */}
        {hasKey && !isEditing && keyData && (
          <div className="mt-3 space-y-1.5">
            <div className="flex items-center gap-2">
              <code className="text-xs bg-muted px-2 py-1 rounded font-mono flex-1 truncate">
                {showKey ? keyData.key : maskKey(keyData.key)}
              </code>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowKey(!showKey)}
                className="h-7 w-7 p-0 shrink-0"
              >
                {showKey ? (
                  <EyeOff className="h-3.5 w-3.5" />
                ) : (
                  <Eye className="h-3.5 w-3.5" />
                )}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsEditing(true)}
                disabled={!byokEnabled}
                className="h-7 text-xs px-2 shrink-0"
              >
                Change
              </Button>
            </div>
            {keyData.lastTested && (
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                {status === "active" && <Check className="h-3 w-3 text-green-600" />}
                {status === "invalid" && <X className="h-3 w-3 text-red-600" />}
                Last tested: {formatRelativeTime(keyData.lastTested)}
              </p>
            )}
          </div>
        )}

        {/* Key Input (when editing) */}
        {isEditing && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Input
                  type={showKey ? "text" : "password"}
                  placeholder={provider.placeholder}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="pr-9 font-mono text-sm"
                  autoFocus
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowKey(!showKey)}
                  className="absolute right-0 top-0 h-full w-9 p-0"
                  tabIndex={-1}
                >
                  {showKey ? (
                    <EyeOff className="h-3.5 w-3.5" />
                  ) : (
                    <Eye className="h-3.5 w-3.5" />
                  )}
                </Button>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={handleSave} className="gap-1 text-xs">
                <Check className="h-3 w-3" />
                Save Key
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleCancel}
                className="gap-1 text-xs"
              >
                <X className="h-3 w-3" />
                Cancel
              </Button>
              <a
                href={provider.docsUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-primary hover:underline ml-auto"
              >
                Get API key
              </a>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ApiKeysPanel({ className }: ApiKeysPanelProps) {
  const [storage, setStorage] = useState<BYOKStorage>({
    enabled: false,
    keys: {},
  });
  const [testingProvider, setTestingProvider] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = loadBYOKStorage();
    setStorage(saved);
    setLoaded(true);
  }, []);

  // Persist to localStorage on change (skip initial load)
  useEffect(() => {
    if (loaded) {
      saveBYOKStorage(storage);
    }
  }, [storage, loaded]);

  const handleToggleBYOK = useCallback((enabled: boolean) => {
    setStorage((prev) => ({ ...prev, enabled }));
    if (enabled) {
      toast.success("BYOK enabled -- your personal keys will be used when available");
    } else {
      toast("BYOK disabled -- shared system keys will be used");
    }
  }, []);

  const handleSaveKey = useCallback((providerId: string, key: string) => {
    setStorage((prev) => ({
      ...prev,
      keys: {
        ...prev.keys,
        [providerId]: {
          key,
          lastTested: null,
          isValid: null,
        },
      },
    }));
    const provider = PROVIDERS.find((p) => p.id === providerId);
    toast.success(`${provider?.name || providerId} API key saved`);
  }, []);

  const handleRemoveKey = useCallback((providerId: string) => {
    setStorage((prev) => {
      const newKeys = { ...prev.keys };
      newKeys[providerId] = null;
      return { ...prev, keys: newKeys };
    });
    const provider = PROVIDERS.find((p) => p.id === providerId);
    toast.success(`${provider?.name || providerId} API key removed`);
  }, []);

  const handleTestKey = useCallback(
    async (providerId: string) => {
      const keyData = storage.keys[providerId];
      if (!keyData || !keyData.key) {
        toast.error("No key to test");
        return;
      }

      setTestingProvider(providerId);
      const provider = PROVIDERS.find((p) => p.id === providerId);

      try {
        // Simulated API validation (mocked with 1.5s delay)
        // In a production implementation, this would call a backend endpoint
        // like POST /api/v1/byok/test with the encrypted key + provider
        await new Promise((resolve) => setTimeout(resolve, 1500));

        // Mock: Check if the key looks plausible based on prefix
        const key = keyData.key;
        let isValid = true;

        // Basic format validation
        if (provider?.keyPrefix && !key.startsWith(provider.keyPrefix)) {
          isValid = false;
        }
        if (key.length < 10) {
          isValid = false;
        }

        const now = new Date().toISOString();

        setStorage((prev) => ({
          ...prev,
          keys: {
            ...prev.keys,
            [providerId]: {
              ...keyData,
              lastTested: now,
              isValid,
            },
          },
        }));

        if (isValid) {
          toast.success(
            `${provider?.name || providerId} key validated successfully`
          );
        } else {
          toast.error(
            `${provider?.name || providerId} key appears to be invalid. Please check the format.`
          );
        }
      } catch {
        toast.error("Failed to test key. Please try again.");
      } finally {
        setTestingProvider(null);
      }
    },
    [storage.keys]
  );

  // Calculate summary stats
  const configuredCount = PROVIDERS.filter((p) => {
    const data = storage.keys[p.id];
    return data && data.key;
  }).length;

  const activeCount = PROVIDERS.filter((p) => {
    const data = storage.keys[p.id];
    return data && data.key && data.isValid === true;
  }).length;

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <CardTitle className="flex items-center gap-2 text-xl">
                <Key className="h-5 w-5" />
                API Keys (Bring Your Own Key)
              </CardTitle>
              <CardDescription>
                Use your own API keys for LLM providers. When set, your personal
                key is used instead of the shared system key.
              </CardDescription>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <Label htmlFor="byok-toggle" className="text-sm font-medium">
                Enable BYOK
              </Label>
              <Switch
                id="byok-toggle"
                checked={storage.enabled}
                onCheckedChange={handleToggleBYOK}
              />
            </div>
          </div>
        </CardHeader>

        {/* Summary Stats */}
        {storage.enabled && (
          <CardContent className="pt-0">
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Key className="h-3.5 w-3.5" />
                {configuredCount} of {PROVIDERS.length} configured
              </span>
              <Separator orientation="vertical" className="h-4" />
              <span className="flex items-center gap-1">
                <Check className="h-3.5 w-3.5 text-green-600" />
                {activeCount} active
              </span>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Provider Cards */}
      <div className="grid gap-3">
        {PROVIDERS.map((provider) => (
          <ProviderCard
            key={provider.id}
            provider={provider}
            keyData={storage.keys[provider.id]}
            byokEnabled={storage.enabled}
            onSave={handleSaveKey}
            onRemove={handleRemoveKey}
            onTest={handleTestKey}
            isTesting={testingProvider === provider.id}
          />
        ))}
      </div>

      {/* Security Info */}
      <Alert>
        <Shield className="h-4 w-4" />
        <AlertTitle>Security Notice</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>
            Keys are encoded and stored in your browser&apos;s local storage.
            They are never sent to our servers, shared with other users, or
            logged in any way.
          </p>
          <p className="text-xs">
            For production deployments, keys should be stored server-side with
            proper encryption at rest. This client-side implementation is
            intended for development and demonstration purposes.
          </p>
        </AlertDescription>
      </Alert>
    </div>
  );
}

export default ApiKeysPanel;
