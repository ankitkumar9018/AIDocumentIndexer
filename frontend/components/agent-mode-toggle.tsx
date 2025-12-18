"use client";

import { useState } from "react";
import { Bot, MessageSquare, Zap, Info, Settings2, Brain, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  useAgentMode,
  useToggleAgentMode,
  useAgentPreferences,
  useUpdateAgentPreferences,
} from "@/lib/api/hooks";
import type { ExecutionMode } from "@/lib/api/client";

interface AgentModeToggleProps {
  sessionId?: string;
  className?: string;
  showLabel?: boolean;
  onModeChange?: (mode: ExecutionMode) => void;
}

export function AgentModeToggle({
  sessionId,
  className,
  showLabel = true,
  onModeChange,
}: AgentModeToggleProps) {
  const [isOpen, setIsOpen] = useState(false);

  const { data: modeData, isLoading: modeLoading } = useAgentMode(sessionId);
  const { data: preferences } = useAgentPreferences();
  const toggleMode = useToggleAgentMode();
  const updatePreferences = useUpdateAgentPreferences();

  const isAgentMode = modeData?.agent_mode_enabled ?? true;
  const autoDetect = modeData?.auto_detect_complexity ?? true;
  const generalChatEnabled = modeData?.general_chat_enabled ?? true;
  const fallbackToGeneral = modeData?.fallback_to_general ?? true;

  const handleToggle = async () => {
    try {
      await toggleMode.mutateAsync();
      onModeChange?.(isAgentMode ? "chat" : "agent");
    } catch (error) {
      console.error("Failed to toggle agent mode:", error);
    }
  };

  const handleAutoDetectChange = async (enabled: boolean) => {
    try {
      await updatePreferences.mutateAsync({
        auto_detect_complexity: enabled,
      });
    } catch (error) {
      console.error("Failed to update auto-detect setting:", error);
    }
  };

  const handleGeneralChatChange = async (enabled: boolean) => {
    try {
      await updatePreferences.mutateAsync({
        general_chat_enabled: enabled,
      });
    } catch (error) {
      console.error("Failed to update general chat setting:", error);
    }
  };

  const handleFallbackChange = async (enabled: boolean) => {
    try {
      await updatePreferences.mutateAsync({
        fallback_to_general: enabled,
      });
    } catch (error) {
      console.error("Failed to update fallback setting:", error);
    }
  };

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={isAgentMode ? "default" : "outline"}
              size="sm"
              onClick={handleToggle}
              disabled={modeLoading || toggleMode.isPending}
              className={cn(
                "gap-2 transition-all",
                isAgentMode && "bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600"
              )}
            >
              {isAgentMode ? (
                <>
                  <Bot className="h-4 w-4" />
                  {showLabel && <span>Agent Mode</span>}
                  <Badge variant="secondary" className="text-[10px] px-1 py-0 bg-white/20">
                    ON
                  </Badge>
                </>
              ) : (
                <>
                  <MessageSquare className="h-4 w-4" />
                  {showLabel && <span>Chat Mode</span>}
                </>
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p className="font-medium">
              {isAgentMode ? "Agent Mode Active" : "Chat Mode Active"}
            </p>
            <p className="text-xs text-muted-foreground">
              {isAgentMode
                ? "Complex tasks use multi-agent orchestration"
                : "Direct chat with RAG-powered responses"}
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <Settings2 className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-80" align="end">
          <div className="space-y-4">
            <div className="space-y-2">
              <h4 className="font-medium flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Agent Settings
              </h4>
              <p className="text-xs text-muted-foreground">
                Configure how the AI processes your requests
              </p>
            </div>

            <div className="space-y-3 pt-2">
              {/* Agent Mode Toggle */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="agent-mode" className="text-sm font-medium">
                    Agent Mode
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Enable multi-agent orchestration
                  </p>
                </div>
                <Switch
                  id="agent-mode"
                  checked={isAgentMode}
                  onCheckedChange={handleToggle}
                  disabled={toggleMode.isPending}
                />
              </div>

              {/* Auto-Detect Complexity */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="auto-detect" className="text-sm font-medium">
                    Auto-Detect Complexity
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Automatically route complex queries
                  </p>
                </div>
                <Switch
                  id="auto-detect"
                  checked={autoDetect}
                  onCheckedChange={handleAutoDetectChange}
                  disabled={updatePreferences.isPending}
                />
              </div>

              {/* General Chat Section */}
              <div className="pt-2 border-t space-y-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Brain className="h-3.5 w-3.5" />
                  <span className="font-medium">General Chat Options</span>
                </div>

                {/* General Chat Mode */}
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="general-chat" className="text-sm font-medium">
                      General Chat
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Answer questions without documents
                    </p>
                  </div>
                  <Switch
                    id="general-chat"
                    checked={generalChatEnabled}
                    onCheckedChange={handleGeneralChatChange}
                    disabled={updatePreferences.isPending}
                  />
                </div>

                {/* Smart Fallback to General */}
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="fallback-general" className="text-sm font-medium">
                      Smart Fallback
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Use general chat when no documents match
                    </p>
                  </div>
                  <Switch
                    id="fallback-general"
                    checked={fallbackToGeneral}
                    onCheckedChange={handleFallbackChange}
                    disabled={updatePreferences.isPending || !generalChatEnabled}
                  />
                </div>
              </div>

              {/* Cost Threshold */}
              {preferences && (
                <div className="pt-2 border-t">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">
                      Approval threshold
                    </span>
                    <span className="font-medium">
                      ${preferences.require_approval_above_usd.toFixed(2)}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Ask for approval above this cost
                  </p>
                </div>
              )}
            </div>

            {/* Mode Explanation */}
            <div className="pt-3 border-t">
              <div className="flex items-start gap-2 text-xs">
                <Info className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
                <div className="text-muted-foreground">
                  <p className="font-medium text-foreground">
                    {isAgentMode ? "Agent Mode" : "Chat Mode"}
                  </p>
                  <p className="mt-1">
                    {isAgentMode
                      ? "Uses specialized AI agents (Generator, Critic, Research) to handle complex multi-step tasks with quality assurance."
                      : "Direct conversation with RAG-powered search. Best for simple questions and document lookups."}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
