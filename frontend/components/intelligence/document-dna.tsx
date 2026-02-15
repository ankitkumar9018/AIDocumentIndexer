"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Dna,
  Search,
  Copy,
  FileCheck,
  AlertTriangle,
  RefreshCw,
  FileText,
  Fingerprint,
  GitCompare,
  Percent
} from "lucide-react";
import { api } from "@/lib/api";

interface DocumentDNA {
  document_id: string;
  title: string;
  minhash_signature: string;
  simhash: string;
  ngram_fingerprint: string[];
  content_length: number;
  created_at: string;
}

interface DNAMatch {
  document_id: string;
  title: string;
  similarity: number;
  match_type: string;
  matching_sections: string[];
}

interface DuplicateCheckResult {
  duplicates: DNAMatch[];
  near_duplicates: DNAMatch[];
  total_checked: number;
}

interface PlagiarismCheckResult {
  document_id: string;
  matches: DNAMatch[];
  overall_similarity: number;
  is_suspicious: boolean;
  flagged_sections: Array<{
    text: string;
    similarity: number;
    source_document: string;
  }>;
}

export function DocumentDNA() {
  const [activeTab, setActiveTab] = useState("fingerprints");
  const [dnaRecords, setDnaRecords] = useState<DocumentDNA[]>([]);
  const [duplicateResult, setDuplicateResult] = useState<DuplicateCheckResult | null>(null);
  const [plagiarismResult, setPlagiarismResult] = useState<PlagiarismCheckResult | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  const fetchDNARecords = async () => {
    try {
      const { data } = await api.get<{ records: DocumentDNA[] }>("/intelligence/dna/list");
      setDnaRecords(data.records || []);
    } catch (error) {
      console.error("Failed to fetch DNA records:", error);
    }
  };

  const checkDuplicates = async () => {
    setIsChecking(true);
    try {
      const { data } = await api.post<DuplicateCheckResult>("/intelligence/dna/check-duplicates");
      setDuplicateResult(data);
    } catch (error) {
      console.error("Duplicate check failed:", error);
    } finally {
      setIsChecking(false);
    }
  };

  const checkPlagiarism = async (documentId: string) => {
    setIsChecking(true);
    setSelectedDocId(documentId);
    try {
      const { data } = await api.post<PlagiarismCheckResult>("/intelligence/dna/plagiarism-check", { document_id: documentId });
      setPlagiarismResult(data);
    } catch (error) {
      console.error("Plagiarism check failed:", error);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    fetchDNARecords();
  }, []);

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.9) return "text-red-600";
    if (similarity >= 0.7) return "text-orange-600";
    if (similarity >= 0.5) return "text-amber-600";
    return "text-green-600";
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Dna className="h-6 w-6" />
            Document DNA
          </h2>
          <p className="text-muted-foreground">
            Document fingerprinting for deduplication and plagiarism detection
          </p>
        </div>

        <div className="flex gap-2">
          <Button onClick={checkDuplicates} disabled={isChecking} variant="outline">
            <Copy className="h-4 w-4 mr-2" />
            Check Duplicates
          </Button>
          <Button onClick={fetchDNARecords} variant="ghost">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="fingerprints">
            <Fingerprint className="h-4 w-4 mr-2" />
            Fingerprints
          </TabsTrigger>
          <TabsTrigger value="duplicates">
            <Copy className="h-4 w-4 mr-2" />
            Duplicates
          </TabsTrigger>
          <TabsTrigger value="plagiarism">
            <AlertTriangle className="h-4 w-4 mr-2" />
            Plagiarism
          </TabsTrigger>
        </TabsList>

        {/* Fingerprints Tab */}
        <TabsContent value="fingerprints">
          <ScrollArea className="h-[500px]">
            <div className="space-y-4">
              {dnaRecords.map((record) => (
                <Card key={record.document_id}>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base flex items-center gap-2">
                        <FileText className="h-4 w-4" />
                        {record.title || record.document_id}
                      </CardTitle>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => checkPlagiarism(record.document_id)}
                      >
                        <Search className="h-3 w-3 mr-1" />
                        Check Plagiarism
                      </Button>
                    </div>
                    <CardDescription>
                      Created: {new Date(record.created_at).toLocaleDateString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">MinHash:</span>
                        <code className="ml-2 text-xs bg-muted px-2 py-1 rounded">
                          {record.minhash_signature.slice(0, 20)}...
                        </code>
                      </div>
                      <div>
                        <span className="text-muted-foreground">SimHash:</span>
                        <code className="ml-2 text-xs bg-muted px-2 py-1 rounded">
                          {record.simhash.slice(0, 20)}...
                        </code>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Content Length:</span>
                        <span className="ml-2">{record.content_length.toLocaleString()} chars</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">N-gram Count:</span>
                        <span className="ml-2">{record.ngram_fingerprint.length}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </ScrollArea>

          {dnaRecords.length === 0 && (
            <Card>
              <CardContent className="py-12 text-center">
                <Dna className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No DNA Records</h3>
                <p className="text-muted-foreground">
                  Document DNA is generated automatically when documents are processed
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Duplicates Tab */}
        <TabsContent value="duplicates">
          {duplicateResult ? (
            <div className="space-y-6">
              {/* Stats */}
              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">{duplicateResult.total_checked}</div>
                    <p className="text-sm text-muted-foreground">Documents Checked</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-red-600">
                      {duplicateResult.duplicates.length}
                    </div>
                    <p className="text-sm text-muted-foreground">Exact Duplicates</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-amber-600">
                      {duplicateResult.near_duplicates.length}
                    </div>
                    <p className="text-sm text-muted-foreground">Near Duplicates</p>
                  </CardContent>
                </Card>
              </div>

              {/* Exact Duplicates */}
              {duplicateResult.duplicates.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base flex items-center gap-2">
                      <Copy className="h-4 w-4 text-red-600" />
                      Exact Duplicates
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {duplicateResult.duplicates.map((match) => (
                        <div
                          key={match.document_id}
                          className="flex items-center justify-between p-3 bg-red-50 rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            <FileText className="h-4 w-4 text-red-600" />
                            <span>{match.title || match.document_id}</span>
                          </div>
                          <Badge variant="destructive">
                            {(match.similarity * 100).toFixed(0)}% match
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Near Duplicates */}
              {duplicateResult.near_duplicates.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base flex items-center gap-2">
                      <GitCompare className="h-4 w-4 text-amber-600" />
                      Near Duplicates
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {duplicateResult.near_duplicates.map((match) => (
                        <div
                          key={match.document_id}
                          className="flex items-center justify-between p-3 bg-amber-50 rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            <FileText className="h-4 w-4 text-amber-600" />
                            <span>{match.title || match.document_id}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Progress
                              value={match.similarity * 100}
                              className="w-24 h-2"
                            />
                            <span className={getSimilarityColor(match.similarity)}>
                              {(match.similarity * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {duplicateResult.duplicates.length === 0 &&
                duplicateResult.near_duplicates.length === 0 && (
                  <Alert>
                    <FileCheck className="h-4 w-4" />
                    <AlertTitle>All Clear</AlertTitle>
                    <AlertDescription>
                      No duplicate documents detected in your knowledge base.
                    </AlertDescription>
                  </Alert>
                )}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <Copy className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Duplicate Check Yet</h3>
                <p className="text-muted-foreground mb-4">
                  Click "Check Duplicates" to scan for duplicate documents
                </p>
                <Button onClick={checkDuplicates} disabled={isChecking}>
                  {isChecking ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Search className="h-4 w-4 mr-2" />
                  )}
                  Check Now
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Plagiarism Tab */}
        <TabsContent value="plagiarism">
          {plagiarismResult ? (
            <div className="space-y-6">
              {/* Summary */}
              <Card className={plagiarismResult.is_suspicious ? "border-red-200" : "border-green-200"}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">
                      Plagiarism Check Results
                    </CardTitle>
                    <Badge
                      variant={plagiarismResult.is_suspicious ? "destructive" : "default"}
                    >
                      {plagiarismResult.is_suspicious ? "Suspicious" : "Clear"}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="flex justify-between mb-2">
                        <span className="text-sm">Overall Similarity</span>
                        <span className={getSimilarityColor(plagiarismResult.overall_similarity)}>
                          {(plagiarismResult.overall_similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={plagiarismResult.overall_similarity * 100}
                        className="h-3"
                      />
                    </div>
                    <Percent className="h-8 w-8 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>

              {/* Flagged Sections */}
              {plagiarismResult.flagged_sections.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-amber-600" />
                      Flagged Sections
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {plagiarismResult.flagged_sections.map((section, idx) => (
                        <div
                          key={idx}
                          className="p-4 bg-amber-50 rounded-lg border border-amber-200"
                        >
                          <p className="text-sm mb-2">"{section.text}"</p>
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">
                              Source: {section.source_document}
                            </span>
                            <Badge variant="outline" className="text-amber-600">
                              {(section.similarity * 100).toFixed(0)}% similar
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Similar Documents */}
              {plagiarismResult.matches.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Similar Documents</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {plagiarismResult.matches.map((match) => (
                        <div
                          key={match.document_id}
                          className="flex items-center justify-between p-3 bg-muted rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            <FileText className="h-4 w-4" />
                            <span>{match.title || match.document_id}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Progress
                              value={match.similarity * 100}
                              className="w-24 h-2"
                            />
                            <span className={getSimilarityColor(match.similarity)}>
                              {(match.similarity * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">Select a Document</h3>
                <p className="text-muted-foreground">
                  Go to the Fingerprints tab and click "Check Plagiarism" on a document
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
