"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import {
  Sparkles,
  Play,
  Users,
  MessageSquare,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  RefreshCw,
  ChevronRight,
  Bot,
  ThumbsUp,
  ThumbsDown,
  AlertCircle,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  useCollaborationSessions,
  useCollaborationSession,
  useCollaborationCritiques,
  useCollaborationModes,
  useCreateCollaborationSession,
  useRunCollaboration,
  useEstimateCollaborationCost,
} from "@/lib/api";

type CollaborationMode = "single" | "review" | "full" | "debate";

const modeDescriptions: Record<CollaborationMode, { title: string; description: string; agents: string[] }> = {
  single: {
    title: "Single Agent",
    description: "One LLM generates the response directly",
    agents: ["Generator"],
  },
  review: {
    title: "Generator + Critic",
    description: "Generator creates, critic reviews and suggests improvements",
    agents: ["Generator", "Critic"],
  },
  full: {
    title: "Full Collaboration",
    description: "Generator, critic, and synthesizer work together",
    agents: ["Generator", "Critic", "Synthesizer"],
  },
  debate: {
    title: "Multi-Agent Debate",
    description: "Multiple agents debate to reach the best answer",
    agents: ["Agent 1", "Agent 2", "Moderator"],
  },
};

export default function CollaborationPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [prompt, setPrompt] = useState("");
  const [context, setContext] = useState("");
  const [mode, setMode] = useState<CollaborationMode>("full");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  // Queries - only fetch when authenticated
  const { data: sessions, isLoading: sessionsLoading, refetch: refetchSessions } = useCollaborationSessions(undefined);
  const { data: selectedSession } = useCollaborationSession(selectedSessionId || "");
  const { data: critiques } = useCollaborationCritiques(selectedSessionId || "");
  const { data: modes } = useCollaborationModes();

  // Mutations
  const createSession = useCreateCollaborationSession();
  const runCollaboration = useRunCollaboration();
  const estimateCost = useEstimateCollaborationCost();

  const handleCreate = async () => {
    if (!prompt) return;

    try {
      const session = await createSession.mutateAsync({
        prompt,
        context: context || undefined,
        mode,
      });
      setSelectedSessionId(session.id);
      await runCollaboration.mutateAsync(session.id);
      refetchSessions();
    } catch (error) {
      console.error("Collaboration failed:", error);
    }
  };

  const handleEstimate = async () => {
    if (!prompt) return;
    try {
      await estimateCost.mutateAsync({
        prompt,
        context: context || undefined,
        mode,
      });
    } catch (error) {
      console.error("Estimate failed:", error);
    }
  };

  const isLoading = createSession.isPending || runCollaboration.isPending;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "generating":
      case "critiquing":
      case "synthesizing":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "generating":
        return "Generating initial response...";
      case "critiquing":
        return "Critic reviewing...";
      case "synthesizing":
        return "Synthesizing final answer...";
      default:
        return status.charAt(0).toUpperCase() + status.slice(1);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Multi-LLM Collaboration</h1>
          <p className="text-muted-foreground">
            Use multiple AI agents to generate higher-quality responses
          </p>
        </div>
        <Button onClick={() => refetchSessions()} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Mode Selection */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {(Object.keys(modeDescriptions) as CollaborationMode[]).map((m) => (
          <Card
            key={m}
            className={`cursor-pointer transition-all ${
              mode === m ? "border-primary ring-2 ring-primary/20" : "hover:border-primary/50"
            }`}
            onClick={() => setMode(m)}
          >
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                {m === "single" && <Bot className="h-4 w-4" />}
                {m === "review" && <MessageSquare className="h-4 w-4" />}
                {m === "full" && <Sparkles className="h-4 w-4" />}
                {m === "debate" && <Users className="h-4 w-4" />}
                {modeDescriptions[m].title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground mb-2">
                {modeDescriptions[m].description}
              </p>
              <div className="flex flex-wrap gap-1">
                {modeDescriptions[m].agents.map((agent) => (
                  <span
                    key={agent}
                    className="text-xs bg-muted px-2 py-0.5 rounded"
                  >
                    {agent}
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Create Session Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            New Collaboration
          </CardTitle>
          <CardDescription>
            Enter your prompt and optional context to start a multi-agent collaboration
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Prompt</label>
            <textarea
              placeholder="What would you like the AI agents to work on?"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full min-h-[100px] p-3 rounded-md border bg-background resize-none"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Context (optional)</label>
            <textarea
              placeholder="Add any additional context or background information..."
              value={context}
              onChange={(e) => setContext(e.target.value)}
              className="w-full min-h-[80px] p-3 rounded-md border bg-background resize-none"
            />
          </div>

          <div className="flex gap-2">
            <Button
              onClick={handleCreate}
              disabled={!prompt || isLoading}
              className="flex-1"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Start Collaboration
            </Button>
            <Button
              onClick={handleEstimate}
              variant="outline"
              disabled={!prompt || estimateCost.isPending}
            >
              {estimateCost.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Zap className="h-4 w-4" />
              )}
            </Button>
          </div>

          {estimateCost.data && (
            <div className="p-3 rounded-lg bg-muted text-sm">
              <span className="text-muted-foreground">Estimated cost: </span>
              <span className="font-medium">
                ${estimateCost.data.estimated_cost?.toFixed(4) || "0.0000"}
              </span>
              <span className="text-muted-foreground ml-2">
                (cost includes all steps)
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Content Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Session History */}
        <Card>
          <CardHeader>
            <CardTitle>Session History</CardTitle>
            <CardDescription>Previous collaboration sessions</CardDescription>
          </CardHeader>
          <CardContent>
            {sessionsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : sessions?.sessions && sessions.sessions.length > 0 ? (
              <div className="space-y-2">
                {sessions.sessions.slice(0, 10).map((session: { id: string; prompt: string; mode: string; status: string; created_at: string }) => (
                  <div
                    key={session.id}
                    className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedSessionId === session.id
                        ? "border-primary bg-primary/5"
                        : "hover:bg-muted/50"
                    }`}
                    onClick={() => setSelectedSessionId(session.id)}
                  >
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      {getStatusIcon(session.status)}
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">
                          {session.prompt.slice(0, 50)}
                          {session.prompt.length > 50 ? "..." : ""}
                        </p>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span className="capitalize">{session.mode}</span>
                          <span>-</span>
                          <span>{new Date(session.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Sparkles className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No collaboration sessions yet</p>
                <p className="text-sm">Start a new collaboration above</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Session Details */}
        <Card>
          <CardHeader>
            <CardTitle>Session Details</CardTitle>
            <CardDescription>
              {selectedSession
                ? `${modeDescriptions[selectedSession.mode as CollaborationMode]?.title || selectedSession.mode} collaboration`
                : "Select a session to view details"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedSession ? (
              <div className="space-y-4">
                {/* Status */}
                <div className="flex items-center gap-2 p-2 rounded-lg bg-muted">
                  {getStatusIcon(selectedSession.status)}
                  <span className="text-sm">{getStatusLabel(selectedSession.status)}</span>
                </div>

                {/* Prompt */}
                <div>
                  <h4 className="text-sm font-medium mb-1">Prompt</h4>
                  <p className="text-sm text-muted-foreground">
                    {selectedSession.prompt}
                  </p>
                </div>

                {/* Generated Response */}
                {selectedSession.initial_generation?.content && (
                  <div>
                    <h4 className="text-sm font-medium mb-1 flex items-center gap-2">
                      <Bot className="h-4 w-4" />
                      Generated Response
                    </h4>
                    <div className="p-3 rounded-lg bg-muted/50 text-sm">
                      {selectedSession.initial_generation.content.slice(0, 500)}
                      {selectedSession.initial_generation.content.length > 500 ? "..." : ""}
                    </div>
                  </div>
                )}

                {/* Critiques */}
                {critiques?.critiques && critiques.critiques.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                      <MessageSquare className="h-4 w-4" />
                      Critiques
                    </h4>
                    <div className="space-y-2">
                      {critiques.critiques.map((critique, i: number) => (
                        <div key={i} className="p-3 rounded-lg border">
                          <div className="flex items-center gap-2 mb-2">
                            {critique.overall_score >= 7 ? (
                              <ThumbsUp className="h-4 w-4 text-green-500" />
                            ) : critique.overall_score <= 4 ? (
                              <ThumbsDown className="h-4 w-4 text-red-500" />
                            ) : (
                              <AlertCircle className="h-4 w-4 text-yellow-500" />
                            )}
                            <span className="text-sm">Score: {critique.overall_score}/10</span>
                          </div>
                          {critique.weaknesses && critique.weaknesses.length > 0 && (
                            <p className="text-sm text-muted-foreground">
                              {critique.weaknesses.join("; ")}
                            </p>
                          )}
                          {critique.suggestions && critique.suggestions.length > 0 && (
                            <div className="mt-2">
                              <p className="text-xs font-medium">Suggestions:</p>
                              <ul className="text-xs text-muted-foreground list-disc list-inside">
                                {critique.suggestions.map((s: string, j: number) => (
                                  <li key={j}>{s}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Final Response */}
                {selectedSession.final_output && (
                  <div>
                    <h4 className="text-sm font-medium mb-1 flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-primary" />
                      Final Response
                    </h4>
                    <div className="p-3 rounded-lg bg-primary/5 border border-primary/20 text-sm">
                      {selectedSession.final_output}
                    </div>
                  </div>
                )}

                {/* Error */}
                {selectedSession.error_message && (
                  <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-600">
                    {selectedSession.error_message}
                  </div>
                )}

                {/* Metadata */}
                <div className="flex flex-wrap gap-2 pt-2 border-t">
                  <span className="text-xs bg-muted px-2 py-1 rounded">
                    Mode: {selectedSession.mode}
                  </span>
                  {selectedSession.total_tokens && (
                    <span className="text-xs bg-muted px-2 py-1 rounded">
                      {selectedSession.total_tokens.toLocaleString()} tokens
                    </span>
                  )}
                  {selectedSession.total_cost && (
                    <span className="text-xs bg-muted px-2 py-1 rounded">
                      ${selectedSession.total_cost.toFixed(4)}
                    </span>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Users className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No session selected</p>
                <p className="text-sm">Select a session from the list to view details</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
