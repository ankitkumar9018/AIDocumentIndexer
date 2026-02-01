"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Microscope,
  Search,
  FileText,
  Sparkles,
  History,
  BookOpen,
  Settings2,
  Loader2,
  Star,
  Trash2,
  AlertCircle,
  RefreshCw,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { api } from "@/lib/api";
import { useSession } from "next-auth/react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { DeepResearchPanel } from "@/components/research/deep-research-panel";

interface ResearchHistoryItem {
  id: string;
  query: string;
  status: string;
  summary?: string;
  confidence_score?: number;
  models_used?: string[];
  is_starred?: boolean;
  created_at: string;
}

export default function ResearchPage() {
  const { data: session } = useSession();
  const [query, setQuery] = useState("");
  const [isResearching, setIsResearching] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [researchHistory, setResearchHistory] = useState<ResearchHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchResearchHistory = useCallback(async () => {
    if (!session?.user) return;

    try {
      setLoading(true);
      setError(null);
      const response = await api.get("/api/v1/research/history");

      if (response.data?.research_history) {
        setResearchHistory(response.data.research_history);
      }
    } catch (err: any) {
      console.error("Failed to fetch research history:", err);
      setResearchHistory([]);
      setError(err?.response?.data?.detail || "Failed to load research history. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [session?.user]);

  useEffect(() => {
    fetchResearchHistory();
  }, [fetchResearchHistory]);

  const toggleStar = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const response = await api.patch(`/api/v1/research/${id}/star`);
      if (response.data) {
        setResearchHistory((prev) =>
          prev.map((r) => (r.id === id ? { ...r, is_starred: response.data.is_starred } : r))
        );
      }
    } catch (err) {
      console.error("Failed to toggle star:", err);
    }
  };

  const deleteResearch = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await api.delete(`/api/v1/research/${id}`);
      setResearchHistory((prev) => prev.filter((r) => r.id !== id));
    } catch (err) {
      console.error("Failed to delete research:", err);
    }
  };

  const handleStartResearch = () => {
    if (!query.trim()) return;
    setIsResearching(true);
    setShowResults(true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Microscope className="h-8 w-8 text-primary" />
            Deep Research
          </h1>
          <p className="text-muted-foreground mt-1">
            Multi-round verification with cross-model fact-checking
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline">
            <Settings2 className="h-4 w-4 mr-2" />
            Settings
          </Button>
          <Button variant="outline">
            <History className="h-4 w-4 mr-2" />
            History
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription className="flex items-center justify-between">
            <span>{error}</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setError(null);
                fetchResearchHistory();
              }}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Search Input */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Enter your research question..."
                className="pl-10 h-12 text-lg"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleStartResearch()}
              />
            </div>
            <Button
              size="lg"
              onClick={handleStartResearch}
              disabled={!query.trim() || isResearching}
              className="h-12 px-8"
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Research
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Deep Research uses multiple AI models to verify facts and cross-check sources
          </p>
        </CardContent>
      </Card>

      {/* Main Content */}
      <Tabs defaultValue={showResults ? "results" : "history"}>
        <TabsList>
          <TabsTrigger value="results" disabled={!showResults}>
            <Sparkles className="h-4 w-4 mr-2" />
            Research Results
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="results" className="mt-4">
          {showResults && (
            <DeepResearchPanel
              query={query}
              isOpen={true}
              onClose={() => {
                setShowResults(false);
                setIsResearching(false);
              }}
            />
          )}
        </TabsContent>

        <TabsContent value="history" className="mt-4">
          {loading ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground">Loading research history...</p>
              </CardContent>
            </Card>
          ) : researchHistory.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Microscope className="h-12 w-12 text-muted-foreground/50 mb-4" />
                <h3 className="text-lg font-medium">No research history</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Start your first research to build your history
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {researchHistory.map((item) => (
                <Card
                  key={item.id}
                  className="cursor-pointer hover:shadow-md transition-shadow group"
                  onClick={() => {
                    setQuery(item.query);
                    setShowResults(true);
                  }}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <CardTitle className="text-base line-clamp-2">
                        {item.query}
                      </CardTitle>
                      <div className="flex items-center gap-1 shrink-0 ml-2">
                        <Badge variant="secondary">
                          {item.status}
                        </Badge>
                        <button
                          onClick={(e) => toggleStar(item.id, e)}
                          className="p-1 hover:bg-muted rounded opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Star
                            className={`h-4 w-4 ${
                              item.is_starred ? "fill-yellow-500 text-yellow-500" : "text-muted-foreground"
                            }`}
                          />
                        </button>
                        <button
                          onClick={(e) => deleteResearch(item.id, e)}
                          className="p-1 hover:bg-destructive/10 hover:text-destructive rounded opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                    <CardDescription className="text-xs">
                      {new Date(item.created_at).toLocaleDateString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      {item.confidence_score !== undefined && (
                        <span className="flex items-center gap-1">
                          <Sparkles className="h-3 w-3" />
                          {Math.round(item.confidence_score * 100)}% confidence
                        </span>
                      )}
                      <span className="flex items-center gap-1">
                        <BookOpen className="h-3 w-3" />
                        View Report
                      </span>
                    </div>
                    {item.summary && (
                      <p className="text-xs text-muted-foreground mt-2 line-clamp-2">
                        {item.summary}
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
