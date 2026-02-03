"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Lightbulb,
  TrendingUp,
  AlertCircle,
  Link2,
  Clock,
  Target,
  Sparkles,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  ExternalLink
} from "lucide-react";

interface Insight {
  id: string;
  type: string;
  title: string;
  description: string;
  priority: number;
  confidence: number;
  related_documents: string[];
  action_items: string[];
  created_at: string;
  dismissed: boolean;
}

interface InsightFeedState {
  insights: Insight[];
  loading: boolean;
  activeTab: string;
}

const insightIcons: Record<string, React.ReactNode> = {
  trending_topic: <TrendingUp className="h-5 w-5 text-blue-500" />,
  knowledge_gap: <AlertCircle className="h-5 w-5 text-amber-500" />,
  document_connection: <Link2 className="h-5 w-5 text-green-500" />,
  stale_content: <Clock className="h-5 w-5 text-red-500" />,
  recommendation: <Target className="h-5 w-5 text-purple-500" />,
  emerging_pattern: <Sparkles className="h-5 w-5 text-pink-500" />,
};

const insightLabels: Record<string, string> = {
  trending_topic: "Trending Topic",
  knowledge_gap: "Knowledge Gap",
  document_connection: "Connection Found",
  stale_content: "Stale Content",
  recommendation: "Recommendation",
  emerging_pattern: "Emerging Pattern",
};

const priorityColors: Record<number, string> = {
  1: "bg-red-100 text-red-800",
  2: "bg-amber-100 text-amber-800",
  3: "bg-blue-100 text-blue-800",
  4: "bg-gray-100 text-gray-800",
};

export function InsightFeed() {
  const [state, setState] = useState<InsightFeedState>({
    insights: [],
    loading: false,
    activeTab: "all",
  });

  const fetchInsights = async () => {
    setState((prev) => ({ ...prev, loading: true }));
    try {
      const response = await fetch("/api/v1/intelligence/insights/feed");
      if (response.ok) {
        const data = await response.json();
        setState((prev) => ({
          ...prev,
          insights: data.insights || [],
          loading: false,
        }));
      }
    } catch (error) {
      console.error("Failed to fetch insights:", error);
      setState((prev) => ({ ...prev, loading: false }));
    }
  };

  const dismissInsight = async (id: string) => {
    try {
      await fetch(`/api/v1/intelligence/insights/${id}/dismiss`, {
        method: "POST",
      });
      setState((prev) => ({
        ...prev,
        insights: prev.insights.filter((i) => i.id !== id),
      }));
    } catch (error) {
      console.error("Failed to dismiss insight:", error);
    }
  };

  const provideFeedback = async (id: string, helpful: boolean) => {
    try {
      await fetch("/api/v1/intelligence/insights/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ insight_id: id, helpful }),
      });
    } catch (error) {
      console.error("Failed to track feedback:", error);
    }
  };

  useEffect(() => {
    fetchInsights();
  }, []);

  const filteredInsights =
    state.activeTab === "all"
      ? state.insights
      : state.insights.filter((i) => i.type === state.activeTab);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Lightbulb className="h-6 w-6" />
            Insight Feed
          </h2>
          <p className="text-muted-foreground">
            Proactive intelligence about your knowledge base
          </p>
        </div>

        <Button onClick={fetchInsights} disabled={state.loading} variant="outline">
          <RefreshCw className={`h-4 w-4 mr-2 ${state.loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Filter Tabs */}
      <Tabs value={state.activeTab} onValueChange={(v) => setState((prev) => ({ ...prev, activeTab: v }))}>
        <TabsList>
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="trending_topic">
            <TrendingUp className="h-4 w-4 mr-1" />
            Trending
          </TabsTrigger>
          <TabsTrigger value="knowledge_gap">
            <AlertCircle className="h-4 w-4 mr-1" />
            Gaps
          </TabsTrigger>
          <TabsTrigger value="document_connection">
            <Link2 className="h-4 w-4 mr-1" />
            Connections
          </TabsTrigger>
          <TabsTrigger value="recommendation">
            <Target className="h-4 w-4 mr-1" />
            Recommendations
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {/* Insight List */}
      <ScrollArea className="h-[600px]">
        <div className="space-y-4">
          {filteredInsights.map((insight) => (
            <Card key={insight.id} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <Avatar className="h-10 w-10">
                      <AvatarFallback className="bg-muted">
                        {insightIcons[insight.type] || <Lightbulb className="h-5 w-5" />}
                      </AvatarFallback>
                    </Avatar>
                    <div>
                      <CardTitle className="text-base">{insight.title}</CardTitle>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline">
                          {insightLabels[insight.type] || insight.type}
                        </Badge>
                        <Badge className={priorityColors[insight.priority] || priorityColors[4]}>
                          Priority {insight.priority}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {(insight.confidence * 100).toFixed(0)}% confidence
                        </span>
                      </div>
                    </div>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {new Date(insight.created_at).toLocaleDateString()}
                  </span>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">{insight.description}</p>

                {insight.action_items.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2">Suggested Actions:</h4>
                    <ul className="text-sm space-y-1">
                      {insight.action_items.map((action, idx) => (
                        <li key={idx} className="flex items-center gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                          {action}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {insight.related_documents.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2">Related Documents:</h4>
                    <div className="flex flex-wrap gap-2">
                      {insight.related_documents.slice(0, 3).map((docId) => (
                        <Button key={docId} variant="outline" size="sm" asChild>
                          <a href={`/dashboard/documents/${docId}`}>
                            <ExternalLink className="h-3 w-3 mr-1" />
                            {docId.slice(0, 8)}...
                          </a>
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Feedback Actions */}
                <div className="flex items-center gap-2 pt-2 border-t">
                  <span className="text-xs text-muted-foreground mr-2">Was this helpful?</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => provideFeedback(insight.id, true)}
                  >
                    <ThumbsUp className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => provideFeedback(insight.id, false)}
                  >
                    <ThumbsDown className="h-4 w-4" />
                  </Button>
                  <div className="flex-1" />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => dismissInsight(insight.id)}
                  >
                    Dismiss
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>

      {filteredInsights.length === 0 && !state.loading && (
        <Card>
          <CardContent className="py-12 text-center">
            <Lightbulb className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Insights Yet</h3>
            <p className="text-muted-foreground">
              Insights will appear as you add more documents and interact with the system
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
