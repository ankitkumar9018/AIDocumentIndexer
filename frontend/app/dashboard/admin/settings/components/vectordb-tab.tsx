"use client";

import { useState, useCallback } from "react";
import { useSession } from "next-auth/react";
import {
  Search,
  Database,
  RefreshCw,
  Loader2,
  CheckCircle,
  AlertCircle,
  Trash2,
  ChevronDown,
  ChevronRight,
  FileText,
  Wrench,
  Eye,
  AlertTriangle,
  HardDrive,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { TabsContent } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface VectorDBStatus {
  status: string;
  backend: string;
  collection_name: string;
  total_chunks: number;
  total_documents: number;
  hnsw_config: Record<string, any>;
  persist_directory: string;
  metadata_coverage: {
    total_chunks: number;
    missing_filename: number;
    missing_content_type: number;
    content_type_breakdown: Record<string, number>;
  };
  documents: Array<{ document_id: string; chunk_count: number }>;
}

interface QueryResult {
  chunk_id: string;
  document_id: string;
  document_filename: string;
  content: string;
  content_length: number;
  score: number;
  similarity_score: number;
  page_number: number;
  section_title: string;
  metadata: Record<string, any>;
}

interface QueryResponse {
  query: string;
  search_type: string;
  top_k: number;
  result_count: number;
  results: QueryResult[];
}

interface ChunkDetail {
  chunk_id: string;
  content: string;
  content_length: number;
  page_number: number;
  chunk_index: number;
  section_title: string;
  content_type: string;
  token_count: number;
  document_filename: string;
  has_embedding: boolean;
  embedding_dimensions: number;
}

interface RepairReport {
  filename_backfilled: number;
  garbage_chunks_found: number;
  back_matter_chunks_found: number;
  missing_embeddings: number;
  errors: string[];
  garbage_chunk_ids: string[];
  back_matter_chunk_ids: string[];
}

export function VectorDBTab() {
  const { data: session } = useSession();
  const accessToken = (session as any)?.accessToken as string | undefined;

  // Status
  const [status, setStatus] = useState<VectorDBStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);

  // Query
  const [queryText, setQueryText] = useState("");
  const [searchType, setSearchType] = useState("hybrid");
  const [topK, setTopK] = useState(10);
  const [queryDocId, setQueryDocId] = useState("");
  const [queryResults, setQueryResults] = useState<QueryResponse | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set());

  // Document chunks
  const [inspectDocId, setInspectDocId] = useState("");
  const [docChunks, setDocChunks] = useState<ChunkDetail[] | null>(null);
  const [docChunksLoading, setDocChunksLoading] = useState(false);
  const [docChunksTotal, setDocChunksTotal] = useState(0);

  // Repair
  const [repairReport, setRepairReport] = useState<RepairReport | null>(null);
  const [repairLoading, setRepairLoading] = useState(false);

  const getHeaders = useCallback(() => ({
    "Authorization": `Bearer ${accessToken || ""}`,
    "Content-Type": "application/json",
  }), [accessToken]);

  // ---- Status ----
  const fetchStatus = useCallback(async () => {
    setStatusLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/admin/vectordb/status`, { headers: getHeaders() });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setStatus(data);
      toast.success("Vector DB status loaded");
    } catch (e: any) {
      toast.error(`Failed to load status: ${e.message}`);
    } finally {
      setStatusLoading(false);
    }
  }, [getHeaders]);

  // ---- Query ----
  const runQuery = useCallback(async () => {
    if (!queryText.trim()) {
      toast.error("Enter a search query");
      return;
    }
    setQueryLoading(true);
    setExpandedResults(new Set());
    try {
      const res = await fetch(`${API_BASE_URL}/admin/vectordb/query`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify({
          query: queryText,
          search_type: searchType,
          top_k: topK,
          document_id: queryDocId || undefined,
        }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setQueryResults(data);
      toast.success(`Found ${data.result_count} results`);
    } catch (e: any) {
      toast.error(`Query failed: ${e.message}`);
    } finally {
      setQueryLoading(false);
    }
  }, [queryText, searchType, topK, queryDocId, getHeaders]);

  // ---- Document Chunks ----
  const fetchDocChunks = useCallback(async () => {
    if (!inspectDocId.trim()) {
      toast.error("Enter a document ID");
      return;
    }
    setDocChunksLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/admin/vectordb/documents/${inspectDocId}/chunks`, {
        headers: getHeaders(),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setDocChunks(data.chunks);
      setDocChunksTotal(data.total_chunks);
      toast.success(`Loaded ${data.total_chunks} chunks`);
    } catch (e: any) {
      toast.error(`Failed to load chunks: ${e.message}`);
    } finally {
      setDocChunksLoading(false);
    }
  }, [inspectDocId, getHeaders]);

  // ---- Repair ----
  const runRepair = useCallback(async () => {
    setRepairLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/admin/vectordb/repair`, {
        method: "POST",
        headers: getHeaders(),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = await res.json();
      setRepairReport(data);
      toast.success("Repair scan completed");
    } catch (e: any) {
      toast.error(`Repair failed: ${e.message}`);
    } finally {
      setRepairLoading(false);
    }
  }, [getHeaders]);

  // ---- Delete Chunk ----
  const deleteChunk = useCallback(async (chunkId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/admin/vectordb/chunks/${chunkId}`, {
        method: "DELETE",
        headers: getHeaders(),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      toast.success(`Chunk ${chunkId.slice(0, 8)}... deleted`);

      // Remove from local state (functional updaters to avoid stale closures on rapid deletes)
      setDocChunks(prev => prev ? prev.filter(c => c.chunk_id !== chunkId) : prev);
      setDocChunksTotal(prev => prev - 1);
      setQueryResults(prev => prev ? {
        ...prev,
        results: prev.results.filter(r => r.chunk_id !== chunkId),
        result_count: prev.result_count - 1,
      } : prev);
    } catch (e: any) {
      toast.error(`Delete failed: ${e.message}`);
    }
  }, [getHeaders]);

  const toggleResultExpanded = (index: number) => {
    setExpandedResults(prev => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  return (
    <TabsContent value="vectordb" className="space-y-6">
      {/* ====== STATUS SECTION ====== */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Vector Database Status
              </CardTitle>
              <CardDescription>ChromaDB health, collection info, and metadata coverage</CardDescription>
            </div>
            <Button onClick={fetchStatus} disabled={statusLoading} variant="outline" size="sm">
              {statusLoading ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <RefreshCw className="h-4 w-4 mr-1" />}
              {statusLoading ? "Loading..." : "Check Status"}
            </Button>
          </div>
        </CardHeader>
        {status && (
          <CardContent className="space-y-4">
            {/* Health badge */}
            <div className="flex items-center gap-4 flex-wrap">
              <Badge variant={status.status === "healthy" ? "default" : "destructive"} className="text-sm">
                {status.status === "healthy" ? <CheckCircle className="h-3 w-3 mr-1" /> : <AlertCircle className="h-3 w-3 mr-1" />}
                {status.status}
              </Badge>
              <span className="text-sm text-muted-foreground">Backend: {status.backend}</span>
              <span className="text-sm text-muted-foreground">Collection: {status.collection_name}</span>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Total Chunks</p>
                <p className="text-2xl font-bold">{status.total_chunks.toLocaleString()}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Documents</p>
                <p className="text-2xl font-bold">{status.total_documents}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Missing Filenames</p>
                <p className="text-2xl font-bold">{status.metadata_coverage.missing_filename}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Missing Content Type</p>
                <p className="text-2xl font-bold">{status.metadata_coverage.missing_content_type}</p>
              </div>
            </div>

            {/* HNSW Config */}
            <div>
              <h4 className="text-sm font-medium mb-2">HNSW Configuration</h4>
              <div className="flex gap-3 flex-wrap">
                {Object.entries(status.hnsw_config).map(([key, value]) => (
                  <Badge key={key} variant="outline" className="text-xs">
                    {key}: {String(value)}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Content type breakdown */}
            {Object.keys(status.metadata_coverage.content_type_breakdown).length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2">Content Type Breakdown</h4>
                <div className="flex gap-2 flex-wrap">
                  {Object.entries(status.metadata_coverage.content_type_breakdown).map(([type, count]) => (
                    <Badge key={type} variant="secondary" className="text-xs">
                      {type}: {count}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Document list */}
            {status.documents.length > 0 && (
              <Collapsible>
                <CollapsibleTrigger className="flex items-center gap-1 text-sm font-medium hover:underline">
                  <ChevronRight className="h-4 w-4" />
                  Documents ({status.documents.length})
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="mt-2 max-h-48 overflow-y-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Document ID</TableHead>
                          <TableHead className="text-right">Chunks</TableHead>
                          <TableHead></TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {status.documents.map((doc) => (
                          <TableRow key={doc.document_id}>
                            <TableCell className="font-mono text-xs">{doc.document_id}</TableCell>
                            <TableCell className="text-right">{doc.chunk_count}</TableCell>
                            <TableCell>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => {
                                  setInspectDocId(doc.document_id);
                                  // Auto-fetch chunks
                                  setTimeout(() => {
                                    const el = document.getElementById("inspect-section");
                                    el?.scrollIntoView({ behavior: "smooth" });
                                  }, 100);
                                }}
                              >
                                <Eye className="h-3 w-3 mr-1" />
                                Inspect
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}

            <p className="text-xs text-muted-foreground">Path: {status.persist_directory}</p>
          </CardContent>
        )}
      </Card>

      {/* ====== QUERY SECTION ====== */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Query Vector Database
          </CardTitle>
          <CardDescription>Run direct searches against the vector store to debug retrieval quality</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-2">
              <Label>Search Query</Label>
              <Input
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="e.g. list all planetary boundaries"
                onKeyDown={(e) => e.key === "Enter" && runQuery()}
              />
            </div>
            <div>
              <Label>Search Type</Label>
              <Select value={searchType} onValueChange={setSearchType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hybrid">Hybrid</SelectItem>
                  <SelectItem value="vector">Vector (Semantic)</SelectItem>
                  <SelectItem value="keyword">Keyword</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Top K</Label>
              <Input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
                min={1}
                max={50}
              />
            </div>
          </div>

          <div className="flex items-end gap-4">
            <div className="flex-1">
              <Label>Filter by Document ID (optional)</Label>
              <Input
                value={queryDocId}
                onChange={(e) => setQueryDocId(e.target.value)}
                placeholder="UUID of specific document"
              />
            </div>
            <Button onClick={runQuery} disabled={queryLoading}>
              {queryLoading ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Search className="h-4 w-4 mr-1" />}
              {queryLoading ? "Searching..." : "Search"}
            </Button>
          </div>

          {/* Query Results */}
          {queryResults && (
            <div className="space-y-3 mt-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">
                  Results: {queryResults.result_count} ({queryResults.search_type} search)
                </h4>
                <span className="text-xs text-muted-foreground">Query: &quot;{queryResults.query}&quot;</span>
              </div>

              <ScrollArea className="max-h-[600px]">
                <div className="space-y-2">
                  {queryResults.results.map((r, idx) => (
                    <div key={r.chunk_id} className="border rounded-lg p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge variant="outline" className="text-xs font-mono">#{idx + 1}</Badge>
                          <Badge variant={r.score > 0.5 ? "default" : r.score > 0.2 ? "secondary" : "destructive"} className="text-xs">
                            Score: {r.score}
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Sim: {r.similarity_score}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            <FileText className="h-3 w-3 inline mr-1" />
                            {r.document_filename}
                          </span>
                          {r.page_number > 0 && (
                            <span className="text-xs text-muted-foreground">p.{r.page_number}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleResultExpanded(idx)}
                          >
                            {expandedResults.has(idx) ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-destructive hover:text-destructive"
                            onClick={() => deleteChunk(r.chunk_id)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>

                      {/* Content preview */}
                      <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                        {expandedResults.has(idx) ? r.content : r.content.slice(0, 150) + (r.content.length > 150 ? "..." : "")}
                      </p>

                      {/* Expanded metadata */}
                      {expandedResults.has(idx) && (
                        <div className="bg-muted/30 rounded p-2 text-xs space-y-1">
                          <p><span className="font-medium">Chunk ID:</span> {r.chunk_id}</p>
                          <p><span className="font-medium">Document ID:</span> {r.document_id}</p>
                          <p><span className="font-medium">Content Length:</span> {r.content_length} chars</p>
                          {r.section_title && <p><span className="font-medium">Section:</span> {r.section_title}</p>}
                          {Object.keys(r.metadata).length > 0 && (
                            <div>
                              <span className="font-medium">Metadata:</span>
                              <pre className="mt-1 text-xs overflow-x-auto">{JSON.stringify(r.metadata, null, 2)}</pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ====== INSPECT DOCUMENT SECTION ====== */}
      <Card id="inspect-section">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Inspect Document Chunks
          </CardTitle>
          <CardDescription>View all chunks for a specific document directly from ChromaDB</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <Label>Document ID</Label>
              <Input
                value={inspectDocId}
                onChange={(e) => setInspectDocId(e.target.value)}
                placeholder="Enter document UUID"
                onKeyDown={(e) => e.key === "Enter" && fetchDocChunks()}
              />
            </div>
            <Button onClick={fetchDocChunks} disabled={docChunksLoading}>
              {docChunksLoading ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Eye className="h-4 w-4 mr-1" />}
              {docChunksLoading ? "Loading..." : "Inspect"}
            </Button>
          </div>

          {docChunks && (
            <div className="space-y-3">
              <h4 className="text-sm font-medium">
                Total Chunks: {docChunksTotal}
              </h4>
              <ScrollArea className="max-h-[500px]">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">#</TableHead>
                      <TableHead>Content Preview</TableHead>
                      <TableHead className="w-16">Page</TableHead>
                      <TableHead className="w-24">Type</TableHead>
                      <TableHead className="w-20">Tokens</TableHead>
                      <TableHead className="w-20">Embed</TableHead>
                      <TableHead className="w-12"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {docChunks.map((chunk) => (
                      <TableRow key={chunk.chunk_id}>
                        <TableCell className="font-mono text-xs">{chunk.chunk_index}</TableCell>
                        <TableCell>
                          <p className="text-xs max-w-md truncate" title={chunk.content}>
                            {chunk.content.slice(0, 120)}...
                          </p>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            {chunk.content_length} chars
                            {chunk.section_title && ` | ${chunk.section_title}`}
                          </p>
                        </TableCell>
                        <TableCell className="text-xs">{chunk.page_number || "-"}</TableCell>
                        <TableCell>
                          {chunk.content_type ? (
                            <Badge variant="secondary" className="text-xs">{chunk.content_type}</Badge>
                          ) : (
                            <span className="text-xs text-muted-foreground">-</span>
                          )}
                        </TableCell>
                        <TableCell className="text-xs">{chunk.token_count}</TableCell>
                        <TableCell>
                          {chunk.has_embedding ? (
                            <Badge variant="outline" className="text-xs">
                              <CheckCircle className="h-3 w-3 mr-1 text-green-500" />
                              {chunk.embedding_dimensions}d
                            </Badge>
                          ) : (
                            <Badge variant="destructive" className="text-xs">None</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-destructive hover:text-destructive h-6 w-6 p-0"
                            onClick={() => deleteChunk(chunk.chunk_id)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ====== REPAIR SECTION ====== */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Wrench className="h-5 w-5" />
                Repair & Diagnostics
              </CardTitle>
              <CardDescription>
                Scan for issues and auto-fix: missing filenames, garbage chunks, back-matter detection
              </CardDescription>
            </div>
            <Button onClick={runRepair} disabled={repairLoading} variant="outline">
              {repairLoading ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : <Wrench className="h-4 w-4 mr-1" />}
              {repairLoading ? "Scanning..." : "Run Repair Scan"}
            </Button>
          </div>
        </CardHeader>
        {repairReport && (
          <CardContent className="space-y-4">
            {/* Repair stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-green-500/10 rounded-lg p-3">
                <p className="text-xs text-muted-foreground">Filenames Backfilled</p>
                <p className="text-2xl font-bold text-green-600">{repairReport.filename_backfilled}</p>
              </div>
              <div className={`rounded-lg p-3 ${repairReport.garbage_chunks_found > 0 ? "bg-red-500/10" : "bg-muted/50"}`}>
                <p className="text-xs text-muted-foreground">Garbage Chunks</p>
                <p className={`text-2xl font-bold ${repairReport.garbage_chunks_found > 0 ? "text-red-600" : ""}`}>
                  {repairReport.garbage_chunks_found}
                </p>
              </div>
              <div className={`rounded-lg p-3 ${repairReport.back_matter_chunks_found > 0 ? "bg-yellow-500/10" : "bg-muted/50"}`}>
                <p className="text-xs text-muted-foreground">Back-matter Chunks</p>
                <p className={`text-2xl font-bold ${repairReport.back_matter_chunks_found > 0 ? "text-yellow-600" : ""}`}>
                  {repairReport.back_matter_chunks_found}
                </p>
              </div>
              <div className={`rounded-lg p-3 ${repairReport.missing_embeddings > 0 ? "bg-red-500/10" : "bg-muted/50"}`}>
                <p className="text-xs text-muted-foreground">Missing Embeddings</p>
                <p className={`text-2xl font-bold ${repairReport.missing_embeddings > 0 ? "text-red-600" : ""}`}>
                  {repairReport.missing_embeddings}
                </p>
              </div>
            </div>

            {/* Errors */}
            {repairReport.errors.length > 0 && (
              <div className="bg-red-500/10 rounded-lg p-3">
                <h4 className="text-sm font-medium text-red-600 mb-1 flex items-center gap-1">
                  <AlertTriangle className="h-4 w-4" />
                  Errors ({repairReport.errors.length})
                </h4>
                {repairReport.errors.map((err, i) => (
                  <p key={i} className="text-xs text-red-600">{err}</p>
                ))}
              </div>
            )}

            {/* Garbage chunk IDs */}
            {repairReport.garbage_chunk_ids.length > 0 && (
              <Collapsible>
                <CollapsibleTrigger className="flex items-center gap-1 text-sm font-medium hover:underline text-red-600">
                  <ChevronRight className="h-4 w-4" />
                  Garbage Chunk IDs ({repairReport.garbage_chunk_ids.length})
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
                    {repairReport.garbage_chunk_ids.map((id) => (
                      <div key={id} className="flex items-center justify-between text-xs font-mono bg-muted/50 rounded px-2 py-1">
                        <span>{id}</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive h-5 px-2"
                          onClick={() => deleteChunk(id)}
                        >
                          <Trash2 className="h-3 w-3 mr-1" />
                          Delete
                        </Button>
                      </div>
                    ))}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}

            {/* Back-matter chunk IDs */}
            {repairReport.back_matter_chunk_ids.length > 0 && (
              <Collapsible>
                <CollapsibleTrigger className="flex items-center gap-1 text-sm font-medium hover:underline text-yellow-600">
                  <ChevronRight className="h-4 w-4" />
                  Back-matter Chunk IDs ({repairReport.back_matter_chunk_ids.length})
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
                    {repairReport.back_matter_chunk_ids.map((id) => (
                      <div key={id} className="flex items-center justify-between text-xs font-mono bg-muted/50 rounded px-2 py-1">
                        <span>{id}</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive h-5 px-2"
                          onClick={() => deleteChunk(id)}
                        >
                          <Trash2 className="h-3 w-3 mr-1" />
                          Delete
                        </Button>
                      </div>
                    ))}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}

            {repairReport.garbage_chunks_found === 0 && repairReport.back_matter_chunks_found === 0 && repairReport.missing_embeddings === 0 && repairReport.errors.length === 0 && (
              <div className="flex items-center gap-2 text-green-600 text-sm">
                <CheckCircle className="h-4 w-4" />
                Vector database looks healthy. No issues found.
              </div>
            )}
          </CardContent>
        )}
      </Card>
    </TabsContent>
  );
}
