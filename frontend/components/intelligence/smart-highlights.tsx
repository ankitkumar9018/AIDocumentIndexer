"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Highlighter,
  Sparkles,
  BookOpen,
  Target,
  Quote,
  BarChart3,
  CheckSquare,
  User,
  Building,
  MapPin,
  Zap,
  Clock,
  Brain
} from "lucide-react";
import { api } from "@/lib/api";

interface Highlight {
  id: string;
  type: string;
  text: string;
  start_position: number;
  end_position: number;
  confidence: number;
  entity_type?: string;
  importance_score: number;
}

interface DocumentAnalysis {
  document_id: string;
  title: string;
  highlights: Highlight[];
  reading_time_minutes: number;
  difficulty_level: string;
  key_topics: string[];
  summary: string;
  entity_counts: Record<string, number>;
}

const highlightColors: Record<string, string> = {
  key_point: "bg-yellow-200 border-yellow-400",
  entity: "bg-blue-200 border-blue-400",
  definition: "bg-green-200 border-green-400",
  statistic: "bg-purple-200 border-purple-400",
  quote: "bg-pink-200 border-pink-400",
  action_item: "bg-orange-200 border-orange-400",
};

const highlightIcons: Record<string, React.ReactNode> = {
  key_point: <Target className="h-4 w-4" />,
  entity: <User className="h-4 w-4" />,
  definition: <BookOpen className="h-4 w-4" />,
  statistic: <BarChart3 className="h-4 w-4" />,
  quote: <Quote className="h-4 w-4" />,
  action_item: <CheckSquare className="h-4 w-4" />,
};

const entityIcons: Record<string, React.ReactNode> = {
  PERSON: <User className="h-4 w-4" />,
  ORGANIZATION: <Building className="h-4 w-4" />,
  LOCATION: <MapPin className="h-4 w-4" />,
};

const difficultyColors: Record<string, string> = {
  easy: "bg-green-100 text-green-800",
  medium: "bg-amber-100 text-amber-800",
  hard: "bg-red-100 text-red-800",
};

interface SmartHighlightsProps {
  documentId?: string;
}

export function SmartHighlights({ documentId }: SmartHighlightsProps) {
  const [analysis, setAnalysis] = useState<DocumentAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState("all");
  const [settings, setSettings] = useState({
    showKeyPoints: true,
    showEntities: true,
    showDefinitions: true,
    showStatistics: true,
    showQuotes: true,
    showActionItems: true,
  });

  const analyzeDocument = async () => {
    if (!documentId) return;

    setIsAnalyzing(true);
    try {
      const { data } = await api.post<DocumentAnalysis>("/intelligence/highlights/analyze", { document_id: documentId });
      setAnalysis(data);
    } catch (error) {
      console.error("Analysis failed:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const filteredHighlights =
    analysis?.highlights.filter((h) => {
      if (activeTab !== "all" && h.type !== activeTab) return false;
      if (h.type === "key_point" && !settings.showKeyPoints) return false;
      if (h.type === "entity" && !settings.showEntities) return false;
      if (h.type === "definition" && !settings.showDefinitions) return false;
      if (h.type === "statistic" && !settings.showStatistics) return false;
      if (h.type === "quote" && !settings.showQuotes) return false;
      if (h.type === "action_item" && !settings.showActionItems) return false;
      return true;
    }) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Highlighter className="h-6 w-6" />
            Smart Highlights
          </h2>
          <p className="text-muted-foreground">
            AI-powered reading mode with automatic highlighting
          </p>
        </div>

        <Button onClick={analyzeDocument} disabled={isAnalyzing || !documentId}>
          {isAnalyzing ? (
            <Zap className="h-4 w-4 mr-2 animate-pulse" />
          ) : (
            <Sparkles className="h-4 w-4 mr-2" />
          )}
          {isAnalyzing ? "Analyzing..." : "Analyze Document"}
        </Button>
      </div>

      {/* Document Summary */}
      {analysis && (
        <Card>
          <CardHeader>
            <CardTitle>{analysis.title}</CardTitle>
            <CardDescription>Document Analysis Results</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="flex items-center gap-3">
                <Clock className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{analysis.reading_time_minutes} min</p>
                  <p className="text-xs text-muted-foreground">Reading Time</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Brain className="h-5 w-5 text-muted-foreground" />
                <div>
                  <Badge className={difficultyColors[analysis.difficulty_level]}>
                    {analysis.difficulty_level}
                  </Badge>
                  <p className="text-xs text-muted-foreground">Difficulty</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Target className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{analysis.highlights.length}</p>
                  <p className="text-xs text-muted-foreground">Highlights</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <BookOpen className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{analysis.key_topics.length}</p>
                  <p className="text-xs text-muted-foreground">Topics</p>
                </div>
              </div>
            </div>

            {/* Key Topics */}
            <div className="mb-4">
              <h4 className="text-sm font-medium mb-2">Key Topics</h4>
              <div className="flex flex-wrap gap-2">
                {analysis.key_topics.map((topic, idx) => (
                  <Badge key={idx} variant="secondary">
                    {topic}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Entity Counts */}
            <div className="mb-4">
              <h4 className="text-sm font-medium mb-2">Entities Found</h4>
              <div className="flex gap-4">
                {Object.entries(analysis.entity_counts).map(([type, count]) => (
                  <div key={type} className="flex items-center gap-2">
                    {entityIcons[type] || <User className="h-4 w-4" />}
                    <span className="text-sm">{type}: {count}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Summary */}
            <div>
              <h4 className="text-sm font-medium mb-2">Summary</h4>
              <p className="text-sm text-muted-foreground">{analysis.summary}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Settings and Filters */}
      {analysis && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Highlight Settings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4">
              <div className="flex items-center space-x-2">
                <Switch
                  id="key-points"
                  checked={settings.showKeyPoints}
                  onCheckedChange={(v) => setSettings({ ...settings, showKeyPoints: v })}
                />
                <Label htmlFor="key-points" className="flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Key Points
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="entities"
                  checked={settings.showEntities}
                  onCheckedChange={(v) => setSettings({ ...settings, showEntities: v })}
                />
                <Label htmlFor="entities" className="flex items-center gap-2">
                  <User className="h-4 w-4" />
                  Entities
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="definitions"
                  checked={settings.showDefinitions}
                  onCheckedChange={(v) => setSettings({ ...settings, showDefinitions: v })}
                />
                <Label htmlFor="definitions" className="flex items-center gap-2">
                  <BookOpen className="h-4 w-4" />
                  Definitions
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="statistics"
                  checked={settings.showStatistics}
                  onCheckedChange={(v) => setSettings({ ...settings, showStatistics: v })}
                />
                <Label htmlFor="statistics" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Statistics
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="quotes"
                  checked={settings.showQuotes}
                  onCheckedChange={(v) => setSettings({ ...settings, showQuotes: v })}
                />
                <Label htmlFor="quotes" className="flex items-center gap-2">
                  <Quote className="h-4 w-4" />
                  Quotes
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="action-items"
                  checked={settings.showActionItems}
                  onCheckedChange={(v) => setSettings({ ...settings, showActionItems: v })}
                />
                <Label htmlFor="action-items" className="flex items-center gap-2">
                  <CheckSquare className="h-4 w-4" />
                  Action Items
                </Label>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Highlights List */}
      {analysis && (
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="all">All ({analysis.highlights.length})</TabsTrigger>
            <TabsTrigger value="key_point">
              <Target className="h-4 w-4 mr-1" />
              Key Points
            </TabsTrigger>
            <TabsTrigger value="entity">
              <User className="h-4 w-4 mr-1" />
              Entities
            </TabsTrigger>
            <TabsTrigger value="action_item">
              <CheckSquare className="h-4 w-4 mr-1" />
              Actions
            </TabsTrigger>
          </TabsList>

          <TabsContent value={activeTab}>
            <ScrollArea className="h-[400px]">
              <div className="space-y-3 pr-4">
                {filteredHighlights.map((highlight) => (
                  <Card
                    key={highlight.id}
                    className={`border-l-4 ${highlightColors[highlight.type]}`}
                  >
                    <CardContent className="py-3">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div className="mt-1">
                            {highlightIcons[highlight.type]}
                          </div>
                          <div>
                            <p className="text-sm">{highlight.text}</p>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge variant="outline" className="text-xs">
                                {highlight.type.replace("_", " ")}
                              </Badge>
                              {highlight.entity_type && (
                                <Badge variant="secondary" className="text-xs">
                                  {highlight.entity_type}
                                </Badge>
                              )}
                              <span className="text-xs text-muted-foreground">
                                {(highlight.confidence * 100).toFixed(0)}% confident
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-xs text-muted-foreground">Importance:</span>
                          <Progress
                            value={highlight.importance_score * 100}
                            className="w-16 h-2"
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      )}

      {!analysis && !isAnalyzing && (
        <Card>
          <CardContent className="py-12 text-center">
            <Highlighter className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Document Selected</h3>
            <p className="text-muted-foreground">
              Select a document to analyze and highlight key information
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
