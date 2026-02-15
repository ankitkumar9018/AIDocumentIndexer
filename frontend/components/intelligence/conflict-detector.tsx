"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import {
  AlertTriangle,
  FileWarning,
  RefreshCw,
  Check,
  X,
  ChevronRight,
  FileText,
  Scale,
  Calendar,
  Hash,
  HelpCircle
} from "lucide-react";
import { api } from "@/lib/api";

interface ConflictSource {
  document_id: string;
  text: string;
  location: string;
}

interface Conflict {
  id: string;
  type: string;
  severity: string;
  description: string;
  source_a: ConflictSource;
  source_b: ConflictSource;
  confidence: number;
  detected_at: string;
  resolved: boolean;
  resolution?: string;
}

interface ConflictReport {
  conflicts: Conflict[];
  total_documents: number;
  documents_with_conflicts: number;
  severity_counts: Record<string, number>;
}

const conflictIcons: Record<string, React.ReactNode> = {
  factual: <FileWarning className="h-5 w-5" />,
  numerical: <Hash className="h-5 w-5" />,
  temporal: <Calendar className="h-5 w-5" />,
  definitional: <HelpCircle className="h-5 w-5" />,
  procedural: <Scale className="h-5 w-5" />,
};

const severityColors: Record<string, string> = {
  critical: "bg-red-100 text-red-800 border-red-200",
  high: "bg-orange-100 text-orange-800 border-orange-200",
  medium: "bg-amber-100 text-amber-800 border-amber-200",
  low: "bg-blue-100 text-blue-800 border-blue-200",
};

export function ConflictDetector() {
  const [report, setReport] = useState<ConflictReport | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedConflict, setSelectedConflict] = useState<Conflict | null>(null);
  const [resolution, setResolution] = useState("");
  const [resolutionChoice, setResolutionChoice] = useState<"a" | "b" | "both" | "neither">("a");

  const analyzeConflicts = async () => {
    setIsAnalyzing(true);
    setProgress(0);

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 5, 90));
      }, 300);

      const { data } = await api.post<ConflictReport>("/intelligence/conflicts/analyze");

      clearInterval(progressInterval);
      setProgress(100);

      setReport(data);
    } catch (error) {
      console.error("Analysis failed:", error);
    } finally {
      setIsAnalyzing(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const resolveConflict = async () => {
    if (!selectedConflict) return;

    try {
      await api.post(`/intelligence/conflicts/${selectedConflict.id}/resolve`, {
        resolution_type: resolutionChoice,
        notes: resolution,
      });

      // Update local state
      if (report) {
        setReport({
          ...report,
          conflicts: report.conflicts.map((c) =>
            c.id === selectedConflict.id ? { ...c, resolved: true, resolution } : c
          ),
        });
      }

      setSelectedConflict(null);
      setResolution("");
    } catch (error) {
      console.error("Resolution failed:", error);
    }
  };

  const unresolvedCount = report?.conflicts.filter((c) => !c.resolved).length || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <AlertTriangle className="h-6 w-6" />
            Conflict Detector
          </h2>
          <p className="text-muted-foreground">
            Identify contradictions and inconsistencies across documents
          </p>
        </div>

        <Button onClick={analyzeConflicts} disabled={isAnalyzing}>
          {isAnalyzing ? (
            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <FileWarning className="h-4 w-4 mr-2" />
          )}
          {isAnalyzing ? "Analyzing..." : "Analyze Conflicts"}
        </Button>
      </div>

      {/* Progress */}
      {isAnalyzing && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Comparing documents...</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary Stats */}
      {report && (
        <div className="grid grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{report.total_documents}</div>
              <p className="text-sm text-muted-foreground">Documents Analyzed</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-amber-600">{report.conflicts.length}</div>
              <p className="text-sm text-muted-foreground">Total Conflicts</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-red-600">{unresolvedCount}</div>
              <p className="text-sm text-muted-foreground">Unresolved</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-green-600">
                {report.conflicts.length - unresolvedCount}
              </div>
              <p className="text-sm text-muted-foreground">Resolved</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Severity Breakdown */}
      {report && report.severity_counts && (
        <div className="flex gap-2">
          {Object.entries(report.severity_counts).map(([severity, count]) => (
            <Badge key={severity} className={severityColors[severity]}>
              {severity}: {count}
            </Badge>
          ))}
        </div>
      )}

      {/* Conflict List */}
      {report && (
        <ScrollArea className="h-[500px]">
          <div className="space-y-4">
            {report.conflicts.map((conflict) => (
              <Card
                key={conflict.id}
                className={`cursor-pointer hover:shadow-md transition-shadow ${
                  conflict.resolved ? "opacity-60" : ""
                }`}
                onClick={() => !conflict.resolved && setSelectedConflict(conflict)}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-full ${severityColors[conflict.severity]}`}>
                        {conflictIcons[conflict.type] || <AlertTriangle className="h-5 w-5" />}
                      </div>
                      <div>
                        <CardTitle className="text-base flex items-center gap-2">
                          {conflict.type.charAt(0).toUpperCase() + conflict.type.slice(1)} Conflict
                          {conflict.resolved && (
                            <Check className="h-4 w-4 text-green-600" />
                          )}
                        </CardTitle>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline" className={severityColors[conflict.severity]}>
                            {conflict.severity}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {(conflict.confidence * 100).toFixed(0)}% confidence
                          </span>
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">{conflict.description}</p>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-muted rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <FileText className="h-4 w-4" />
                        <span className="text-xs font-medium">Source A</span>
                      </div>
                      <p className="text-sm line-clamp-2">{conflict.source_a.text}</p>
                    </div>
                    <div className="p-3 bg-muted rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <FileText className="h-4 w-4" />
                        <span className="text-xs font-medium">Source B</span>
                      </div>
                      <p className="text-sm line-clamp-2">{conflict.source_b.text}</p>
                    </div>
                  </div>

                  {conflict.resolved && conflict.resolution && (
                    <Alert className="mt-4">
                      <Check className="h-4 w-4" />
                      <AlertTitle>Resolved</AlertTitle>
                      <AlertDescription>{conflict.resolution}</AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </ScrollArea>
      )}

      {/* Resolution Dialog */}
      <Dialog open={!!selectedConflict} onOpenChange={() => setSelectedConflict(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Resolve Conflict</DialogTitle>
          </DialogHeader>

          {selectedConflict && (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">{selectedConflict.description}</p>

              <div className="grid grid-cols-2 gap-4">
                <Card className="border-2 border-blue-200">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Source A</CardTitle>
                    <CardDescription className="text-xs">
                      {selectedConflict.source_a.document_id}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm">{selectedConflict.source_a.text}</p>
                  </CardContent>
                </Card>

                <Card className="border-2 border-purple-200">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Source B</CardTitle>
                    <CardDescription className="text-xs">
                      {selectedConflict.source_b.document_id}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm">{selectedConflict.source_b.text}</p>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-4">
                <Label>Which source is correct?</Label>
                <RadioGroup value={resolutionChoice} onValueChange={(v) => setResolutionChoice(v as any)}>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="a" id="a" />
                    <Label htmlFor="a">Source A is correct</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="b" id="b" />
                    <Label htmlFor="b">Source B is correct</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="both" id="both" />
                    <Label htmlFor="both">Both are valid (different contexts)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="neither" id="neither" />
                    <Label htmlFor="neither">Neither is correct</Label>
                  </div>
                </RadioGroup>

                <div className="space-y-2">
                  <Label>Resolution Notes</Label>
                  <Textarea
                    value={resolution}
                    onChange={(e) => setResolution(e.target.value)}
                    placeholder="Explain your resolution decision..."
                  />
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedConflict(null)}>
              Cancel
            </Button>
            <Button onClick={resolveConflict}>
              <Check className="h-4 w-4 mr-2" />
              Resolve Conflict
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {!report && !isAnalyzing && (
        <Card>
          <CardContent className="py-12 text-center">
            <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Analysis Yet</h3>
            <p className="text-muted-foreground mb-4">
              Click "Analyze Conflicts" to detect contradictions in your documents
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
