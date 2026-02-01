"use client";

import { MessageSquare, Globe, Eye, Info } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TabsContent } from "@/components/ui/tabs";

interface InstructionsTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
  isLoading: boolean;
  hasChanges: boolean;
}

export function InstructionsTab({
  localSettings,
  handleSettingChange,
  isLoading,
  hasChanges,
}: InstructionsTabProps) {
  const isEnabled = localSettings["system.custom_instructions_enabled"] as boolean ?? false;
  const orgPrompt = localSettings["system.org_system_prompt"] as string ?? "";
  const responseLanguage = localSettings["system.default_response_language"] as string || "auto";
  const appendMode = localSettings["system.custom_instructions_append_mode"] as string || "prepend";

  const buildPreview = (): string => {
    if (!isEnabled) {
      return "(Custom instructions are disabled. Enable the switch above to activate.)";
    }

    if (!orgPrompt.trim()) {
      return "(No organization system prompt configured. Enter a prompt above.)";
    }

    const languageLine =
      responseLanguage !== "auto"
        ? `\n\nDefault response language: ${languageLabel(responseLanguage)}`
        : "";

    const modeLabel =
      appendMode === "prepend"
        ? "[Prepended before RAG system prompt]"
        : appendMode === "append"
          ? "[Appended after RAG system prompt]"
          : "[Replaces RAG system prompt]";

    return `${orgPrompt}${languageLine}\n\n--- ${modeLabel} ---\n[Variables such as {{user_name}}, {{user_role}}, {{date}} will be resolved at runtime]`;
  };

  const languageLabel = (value: string): string => {
    const labels: Record<string, string> = {
      auto: "Auto-detect",
      en: "English",
      es: "Spanish",
      fr: "French",
      de: "German",
      ja: "Japanese",
      zh: "Chinese",
      ko: "Korean",
      pt: "Portuguese",
      ar: "Arabic",
      hi: "Hindi",
    };
    return labels[value] ?? value;
  };

  return (
    <TabsContent value="instructions" className="space-y-6">
      {/* Organization System Prompt */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Custom Instructions
          </CardTitle>
          <CardDescription>
            Set organization-wide default system prompts and custom instructions injected into all LLM calls
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Enable / Disable Switch */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Custom Instructions</p>
              <p className="text-sm text-muted-foreground">
                When enabled, the organization prompt and settings below are injected into every LLM interaction
              </p>
            </div>
            <Switch
              checked={isEnabled}
              onCheckedChange={(checked) =>
                handleSettingChange("system.custom_instructions_enabled", checked)
              }
            />
          </div>

          {/* Organization System Prompt Textarea */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Organization System Prompt</Label>
            <p className="text-sm text-muted-foreground">
              Applied to ALL users&apos; LLM interactions. Define your organization&apos;s tone, rules, and context.
            </p>
            <Textarea
              placeholder="You are a helpful assistant for Acme Corp. Always respond professionally. Our products are: Widget Pro, Widget Lite, and Widget Enterprise. Never discuss competitor products negatively."
              className="min-h-[160px] font-mono text-sm"
              value={orgPrompt}
              onChange={(e) =>
                handleSettingChange("system.org_system_prompt", e.target.value)
              }
              disabled={!isEnabled}
            />
            <div className="flex flex-wrap gap-1.5 pt-1">
              <span className="text-xs text-muted-foreground">Variables:</span>
              <Badge variant="secondary" className="text-xs font-mono">
                {"{{user_name}}"}
              </Badge>
              <Badge variant="secondary" className="text-xs font-mono">
                {"{{user_role}}"}
              </Badge>
              <Badge variant="secondary" className="text-xs font-mono">
                {"{{date}}"}
              </Badge>
            </div>
          </div>

          {/* Default Response Language */}
          <div className="space-y-2 pt-4 border-t">
            <Label className="text-sm font-medium flex items-center gap-2">
              <Globe className="h-4 w-4" />
              Default Response Language
            </Label>
            <Select
              value={responseLanguage}
              onValueChange={(value) =>
                handleSettingChange("system.default_response_language", value)
              }
              disabled={!isEnabled}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto-detect</SelectItem>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="es">Spanish</SelectItem>
                <SelectItem value="fr">French</SelectItem>
                <SelectItem value="de">German</SelectItem>
                <SelectItem value="ja">Japanese</SelectItem>
                <SelectItem value="zh">Chinese</SelectItem>
                <SelectItem value="ko">Korean</SelectItem>
                <SelectItem value="pt">Portuguese</SelectItem>
                <SelectItem value="ar">Arabic</SelectItem>
                <SelectItem value="hi">Hindi</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Set the default language for LLM responses. &quot;Auto-detect&quot; matches the language of the user&apos;s query.
            </p>
          </div>

          {/* Append Mode */}
          <div className="space-y-2 pt-4 border-t">
            <Label className="text-sm font-medium">Instruction Append Mode</Label>
            <Select
              value={appendMode}
              onValueChange={(value) =>
                handleSettingChange("system.custom_instructions_append_mode", value)
              }
              disabled={!isEnabled}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="prepend">Prepend (before RAG system prompt)</SelectItem>
                <SelectItem value="append">Append (after RAG system prompt)</SelectItem>
                <SelectItem value="replace">Replace (override RAG system prompt)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Controls how the organization prompt is combined with the RAG system prompt
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Preview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Preview
          </CardTitle>
          <CardDescription>
            Preview of the final system prompt that will be sent to the LLM
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border bg-muted/30 p-4">
            <p className="text-xs font-medium text-muted-foreground mb-2">
              Final system prompt that will be sent:
            </p>
            <pre className="text-sm whitespace-pre-wrap font-mono leading-relaxed">
              {buildPreview()}
            </pre>
          </div>
        </CardContent>
      </Card>

      {/* Info Box */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-3 p-3 rounded-lg border bg-blue-50/50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800">
            <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 shrink-0" />
            <div className="space-y-1 text-sm">
              <p className="font-medium text-blue-900 dark:text-blue-200">
                How Custom Instructions Work
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 dark:text-blue-300">
                <li>
                  Organization prompts are combined with the RAG system prompt for all users based on the append mode selected above.
                </li>
                <li>
                  Individual users can add personal instructions from their profile settings, which are applied after the organization prompt.
                </li>
                <li>
                  Variables like <code className="font-mono text-xs bg-blue-100 dark:bg-blue-900 px-1 rounded">{"{{user_name}}"}</code> and <code className="font-mono text-xs bg-blue-100 dark:bg-blue-900 px-1 rounded">{"{{user_role}}"}</code> are resolved at runtime for each user session.
                </li>
                <li>
                  Disabling custom instructions does not delete your saved prompt -- it simply stops injecting it into LLM calls.
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
