"use client";

/**
 * Natural Language Database Query Interface (Phase 65)
 *
 * Provides UI for querying the database using natural language:
 * - Text-to-SQL conversion
 * - Interactive clarification
 * - Auto-visualization
 * - Query history
 */

import { useState, useCallback } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Loader2,
  Database,
  Search,
  Code,
  BarChart3,
  History,
  Play,
  Copy,
  AlertCircle,
  CheckCircle,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";

interface QueryResult {
  answer?: string;
  sql?: string;
  data?: Record<string, unknown>[];
  columns?: string[];
  visualization?: {
    chart_type: string;
    chart_code?: string;
    insights?: string[];
  };
  error?: string;
  execution_time_ms?: number;
  row_count?: number;
}

interface ClarificationQuestion {
  question: string;
  options: string[];
}

interface QueryHistoryItem {
  id: string;
  question: string;
  sql: string;
  timestamp: Date;
  success: boolean;
}

export function NaturalLanguageQueryInterface() {
  // Query state
  const [question, setQuestion] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  // Clarification state
  const [clarifications, setClarifications] = useState<ClarificationQuestion[]>([]);
  const [clarificationAnswers, setClarificationAnswers] = useState<Record<number, string>>({});

  // History state
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);

  // Visualization state
  const [chartType, setChartType] = useState("auto");

  // Execute natural language query
  const executeQuery = useCallback(async () => {
    if (!question.trim()) {
      toast.error("Please enter a question");
      return;
    }

    setIsQuerying(true);
    setQueryResult(null);
    setClarifications([]);

    try {
      const { data: result } = await api.post<QueryResult & { status?: string; questions?: ClarificationQuestion[] }>("/database/nl-query", {
        question,
        visualization_type: chartType !== "auto" ? chartType : undefined,
        context: Object.keys(clarificationAnswers).length > 0 ? clarificationAnswers : undefined,
      });

      // Check if clarification needed
      if (result.status === "clarification_needed") {
        setClarifications(result.questions || []);
        toast.info("Please answer the clarification questions");
        return;
      }

      setQueryResult(result);

      // Add to history
      if (result.sql) {
        setQueryHistory((prev) => [
          {
            id: Date.now().toString(),
            question,
            sql: result.sql!,
            timestamp: new Date(),
            success: !result.error,
          },
          ...prev.slice(0, 19), // Keep last 20
        ]);
      }

      toast.success("Query executed successfully");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Query failed");
    } finally {
      setIsQuerying(false);
    }
  }, [question, chartType, clarificationAnswers]);

  // Execute raw SQL
  const executeSQL = useCallback(async (sql: string) => {
    setIsQuerying(true);

    try {
      const { data: result } = await api.post<QueryResult>("/database/execute-sql", { sql });

      setQueryResult(result);
      toast.success("SQL executed successfully");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "SQL execution failed");
    } finally {
      setIsQuerying(false);
    }
  }, []);

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const answerClarification = (index: number, answer: string) => {
    setClarificationAnswers((prev) => ({ ...prev, [index]: answer }));
  };

  const submitClarifications = () => {
    setClarifications([]);
    executeQuery();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Database className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-2xl font-bold">Natural Language Database Query</h1>
          <p className="text-muted-foreground">
            Ask questions about your data in plain English
          </p>
        </div>
      </div>

      <Tabs defaultValue="query" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="query" className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Query
          </TabsTrigger>
          <TabsTrigger value="results" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Results
          </TabsTrigger>
          <TabsTrigger value="history" className="flex items-center gap-2">
            <History className="h-4 w-4" />
            History
          </TabsTrigger>
        </TabsList>

        {/* Query Tab */}
        <TabsContent value="query" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
                Ask a Question
              </CardTitle>
              <CardDescription>
                Query your database using natural language. The system will convert your question to SQL.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Your Question</Label>
                <Textarea
                  placeholder="How many documents were uploaded this month? Show me the top 10 by size."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={3}
                  className="resize-none"
                />
              </div>

              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <Label>Visualization Type</Label>
                  <Select value={chartType} onValueChange={setChartType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Auto-detect" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto-detect</SelectItem>
                      <SelectItem value="bar">Bar Chart</SelectItem>
                      <SelectItem value="line">Line Chart</SelectItem>
                      <SelectItem value="pie">Pie Chart</SelectItem>
                      <SelectItem value="table">Table Only</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="pt-6">
                  <Button onClick={executeQuery} disabled={isQuerying} size="lg">
                    {isQuerying ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Running...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Run Query
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Clarification Questions */}
              {clarifications.length > 0 && (
                <Card className="border-yellow-500/50 bg-yellow-500/10">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-yellow-500" />
                      Clarification Needed
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {clarifications.map((q, i) => (
                      <div key={i} className="space-y-2">
                        <Label>{q.question}</Label>
                        <Select
                          value={clarificationAnswers[i] || ""}
                          onValueChange={(v) => answerClarification(i, v)}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select an option" />
                          </SelectTrigger>
                          <SelectContent>
                            {q.options.map((opt, j) => (
                              <SelectItem key={j} value={opt}>
                                {opt}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    ))}
                    <Button onClick={submitClarifications}>
                      Continue with Query
                    </Button>
                  </CardContent>
                </Card>
              )}

              {/* Generated SQL Preview */}
              {queryResult?.sql && (
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Code className="h-5 w-5" />
                        Generated SQL
                      </CardTitle>
                      <div className="flex gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyToClipboard(queryResult.sql || "")}
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => executeSQL(queryResult.sql || "")}
                        >
                          Re-run
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <pre className="bg-muted p-3 rounded-md text-sm overflow-x-auto">
                      <code>{queryResult.sql}</code>
                    </pre>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results" className="space-y-4">
          {queryResult ? (
            <>
              {/* Summary */}
              {queryResult.answer && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="whitespace-pre-wrap">{queryResult.answer}</p>
                    <div className="flex gap-4 mt-3 text-sm text-muted-foreground">
                      {queryResult.row_count !== undefined && (
                        <Badge variant="secondary">{queryResult.row_count} rows</Badge>
                      )}
                      {queryResult.execution_time_ms !== undefined && (
                        <Badge variant="outline">{queryResult.execution_time_ms.toFixed(0)}ms</Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Data Table */}
              {queryResult.data && queryResult.data.length > 0 && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Data</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-96">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            {queryResult.columns?.map((col, i) => (
                              <TableHead key={i}>{col}</TableHead>
                            )) ||
                              Object.keys(queryResult.data[0]).map((col, i) => (
                                <TableHead key={i}>{col}</TableHead>
                              ))}
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {queryResult.data.map((row, i) => (
                            <TableRow key={i}>
                              {Object.values(row).map((val, j) => (
                                <TableCell key={j}>
                                  {typeof val === "object" ? JSON.stringify(val) : String(val ?? "")}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </ScrollArea>
                  </CardContent>
                </Card>
              )}

              {/* Visualization */}
              {queryResult.visualization && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Visualization
                      <Badge variant="secondary">{queryResult.visualization.chart_type}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {queryResult.visualization.insights && (
                      <div className="mb-4 space-y-1">
                        <Label>Key Insights</Label>
                        <ul className="list-disc list-inside text-sm text-muted-foreground">
                          {queryResult.visualization.insights.map((insight, i) => (
                            <li key={i}>{insight}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <div className="h-64 bg-muted rounded-md flex items-center justify-center text-muted-foreground">
                      {/* Chart would be rendered here using a charting library */}
                      <p>Chart visualization placeholder</p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Error */}
              {queryResult.error && (
                <Card className="border-red-500/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-red-500 flex items-center gap-2">
                      <AlertCircle className="h-5 w-5" />
                      Error
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="text-sm text-red-500 whitespace-pre-wrap">
                      {queryResult.error}
                    </pre>
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Run a query to see results here</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Query History</CardTitle>
              <CardDescription>Your recent database queries</CardDescription>
            </CardHeader>
            <CardContent>
              {queryHistory.length > 0 ? (
                <ScrollArea className="h-96">
                  <div className="space-y-3">
                    {queryHistory.map((item) => (
                      <Card key={item.id} className="cursor-pointer hover:bg-muted/50">
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                {item.success ? (
                                  <CheckCircle className="h-4 w-4 text-green-500" />
                                ) : (
                                  <AlertCircle className="h-4 w-4 text-red-500" />
                                )}
                                <p className="font-medium truncate">{item.question}</p>
                              </div>
                              <pre className="text-xs text-muted-foreground bg-muted p-2 rounded mt-2 overflow-x-auto">
                                {item.sql}
                              </pre>
                              <p className="text-xs text-muted-foreground mt-2">
                                {item.timestamp.toLocaleString()}
                              </p>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setQuestion(item.question);
                                toast.success("Query loaded");
                              }}
                            >
                              Reuse
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="py-12 text-center text-muted-foreground">
                  <History className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No query history yet</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default NaturalLanguageQueryInterface;
