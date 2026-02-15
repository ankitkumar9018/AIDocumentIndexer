"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  FolderKanban,
  Sparkles,
  RefreshCw,
  FileText,
  Clock,
  Network,
  Target,
  Layers
} from "lucide-react";
import { api } from "@/lib/api";

interface SmartCollection {
  id: string;
  name: string;
  description: string;
  strategy: string;
  document_count: number;
  coherence_score: number;
  created_at: string;
  document_ids: string[];
}

interface OrganizationResult {
  success: boolean;
  collections: SmartCollection[];
  stats: {
    total_documents: number;
    organized_documents: number;
    avg_coherence: number;
  };
}

const strategyIcons: Record<string, React.ReactNode> = {
  topic_cluster: <Target className="h-4 w-4" />,
  entity_based: <Network className="h-4 w-4" />,
  time_based: <Clock className="h-4 w-4" />,
  similarity: <Layers className="h-4 w-4" />,
  hybrid: <Sparkles className="h-4 w-4" />,
};

const strategyLabels: Record<string, string> = {
  topic_cluster: "Topic Clustering",
  entity_based: "Entity-Based",
  time_based: "Time-Based",
  similarity: "Similarity",
  hybrid: "Hybrid (All)",
};

export function SmartCollections() {
  const [collections, setCollections] = useState<SmartCollection[]>([]);
  const [isOrganizing, setIsOrganizing] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState("hybrid");
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState<OrganizationResult["stats"] | null>(null);

  const organizeDocuments = async () => {
    setIsOrganizing(true);
    setProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90));
      }, 500);

      const { data: result } = await api.post<OrganizationResult>("/intelligence/collections/organize", {
        strategy: selectedStrategy,
      });

      clearInterval(progressInterval);
      setProgress(100);

      setCollections(result.collections);
      setStats(result.stats);
    } catch (error) {
      console.error("Organization failed:", error);
    } finally {
      setIsOrganizing(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const fetchCollections = async () => {
    try {
      const { data } = await api.get<{ collections: SmartCollection[] }>("/intelligence/collections/smart");
      setCollections(data.collections || []);
    } catch (error) {
      console.error("Failed to fetch collections:", error);
    }
  };

  useEffect(() => {
    fetchCollections();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <FolderKanban className="h-6 w-6" />
            Smart Collections
          </h2>
          <p className="text-muted-foreground">
            AI-powered document organization using clustering algorithms
          </p>
        </div>

        <div className="flex items-center gap-4">
          <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Strategy" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(strategyLabels).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  <div className="flex items-center gap-2">
                    {strategyIcons[value]}
                    {label}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button onClick={organizeDocuments} disabled={isOrganizing}>
                  {isOrganizing ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Sparkles className="h-4 w-4 mr-2" />
                  )}
                  {isOrganizing ? "Organizing..." : "Organize Documents"}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Automatically organize documents into smart collections</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>

      {/* Progress */}
      {isOrganizing && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Analyzing documents...</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-3 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{stats.total_documents}</div>
              <p className="text-sm text-muted-foreground">Total Documents</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{stats.organized_documents}</div>
              <p className="text-sm text-muted-foreground">Organized</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{(stats.avg_coherence * 100).toFixed(0)}%</div>
              <p className="text-sm text-muted-foreground">Avg Coherence</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Collections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {collections.map((collection) => (
          <Card key={collection.id} className="hover:shadow-lg transition-shadow cursor-pointer">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{collection.name}</CardTitle>
                <Badge variant="secondary">
                  {strategyIcons[collection.strategy]}
                  <span className="ml-1">{collection.document_count}</span>
                </Badge>
              </div>
              <CardDescription>{collection.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Coherence Score</span>
                    <span>{(collection.coherence_score * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={collection.coherence_score * 100} />
                </div>

                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <FileText className="h-4 w-4" />
                  <span>{collection.document_count} documents</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {collections.length === 0 && !isOrganizing && (
        <Card>
          <CardContent className="py-12 text-center">
            <FolderKanban className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Smart Collections Yet</h3>
            <p className="text-muted-foreground mb-4">
              Click "Organize Documents" to automatically create smart collections
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
