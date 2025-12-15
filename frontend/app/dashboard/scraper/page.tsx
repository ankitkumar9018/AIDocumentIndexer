"use client";

import { useState } from "react";
import {
  Globe,
  Plus,
  Search,
  ExternalLink,
  FileText,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Link as LinkIcon,
  Download,
  Trash2,
  Play,
  RefreshCw,
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
  useScrapeJobs,
  useScrapeJob,
  useCreateScrapeJob,
  useRunScrapeJob,
  useScrapeUrlImmediate,
  useScrapeAndQuery,
  useExtractLinks,
} from "@/lib/api";

type ScrapeMode = "immediate" | "job" | "query" | "links";

export default function ScraperPage() {
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<ScrapeMode>("immediate");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  // Queries
  const { data: jobs, isLoading: jobsLoading, refetch: refetchJobs } = useScrapeJobs();
  const { data: selectedJob } = useScrapeJob(selectedJobId || "");

  // Mutations
  const createJob = useCreateScrapeJob();
  const runJob = useRunScrapeJob();
  const scrapeImmediate = useScrapeUrlImmediate();
  const scrapeAndQuery = useScrapeAndQuery();
  const extractLinks = useExtractLinks();

  const handleScrape = async () => {
    if (!url) return;

    try {
      switch (mode) {
        case "immediate":
          await scrapeImmediate.mutateAsync({ url });
          break;
        case "job":
          const job = await createJob.mutateAsync({
            urls: [url],
            storage_mode: "permanent",
          });
          setSelectedJobId(job.id);
          await runJob.mutateAsync(job.id);
          break;
        case "query":
          if (!query) return;
          await scrapeAndQuery.mutateAsync({ url, query });
          break;
        case "links":
          await extractLinks.mutateAsync({ url, maxDepth: 2, sameDomainOnly: true });
          break;
      }
      refetchJobs();
    } catch (error) {
      console.error("Scrape failed:", error);
    }
  };

  const isLoading =
    scrapeImmediate.isPending ||
    createJob.isPending ||
    runJob.isPending ||
    scrapeAndQuery.isPending ||
    extractLinks.isPending;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "scraping":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Web Scraper</h1>
          <p className="text-muted-foreground">
            Scrape web pages and add them to your knowledge base
          </p>
        </div>
        <Button onClick={() => refetchJobs()} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Scrape Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Globe className="h-5 w-5" />
            New Scrape
          </CardTitle>
          <CardDescription>
            Enter a URL to scrape and add to your document index
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Mode Selection */}
          <div className="flex flex-wrap gap-2">
            <Button
              variant={mode === "immediate" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("immediate")}
            >
              <FileText className="h-4 w-4 mr-2" />
              Quick Scrape
            </Button>
            <Button
              variant={mode === "job" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("job")}
            >
              <Download className="h-4 w-4 mr-2" />
              Save to Index
            </Button>
            <Button
              variant={mode === "query" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("query")}
            >
              <Search className="h-4 w-4 mr-2" />
              Scrape & Query
            </Button>
            <Button
              variant={mode === "links" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("links")}
            >
              <LinkIcon className="h-4 w-4 mr-2" />
              Extract Links
            </Button>
          </div>

          {/* URL Input */}
          <div className="flex gap-2">
            <Input
              placeholder="https://example.com/page"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="flex-1"
            />
            <Button onClick={handleScrape} disabled={!url || isLoading}>
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              <span className="ml-2">Scrape</span>
            </Button>
          </div>

          {/* Query Input (for query mode) */}
          {mode === "query" && (
            <Input
              placeholder="Ask a question about the page..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          )}

          {/* Mode Descriptions */}
          <p className="text-sm text-muted-foreground">
            {mode === "immediate" &&
              "Quick scrape extracts content without saving to the index."}
            {mode === "job" &&
              "Save to index permanently stores the scraped content for RAG queries."}
            {mode === "query" &&
              "Scrape & query extracts content and immediately answers your question."}
            {mode === "links" &&
              "Extract links discovers all URLs linked from the page."}
          </p>
        </CardContent>
      </Card>

      {/* Results Section */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Recent Jobs */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Jobs</CardTitle>
            <CardDescription>Your scraping history</CardDescription>
          </CardHeader>
          <CardContent>
            {jobsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : jobs?.jobs && jobs.jobs.length > 0 ? (
              <div className="space-y-2">
                {jobs.jobs.slice(0, 10).map((job) => (
                  <div
                    key={job.id}
                    className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-colors ${
                      selectedJobId === job.id
                        ? "border-primary bg-primary/5"
                        : "hover:bg-muted/50"
                    }`}
                    onClick={() => setSelectedJobId(job.id)}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      {getStatusIcon(job.status)}
                      <div className="min-w-0">
                        <p className="text-sm font-medium truncate">
                          {job.pages?.[0]?.url || "Unknown URL"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(job.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <ExternalLink className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Globe className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No scrape jobs yet</p>
                <p className="text-sm">Enter a URL above to get started</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Job Details / Results */}
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
            <CardDescription>
              {selectedJob
                ? `Details for job ${selectedJob.id.slice(0, 8)}...`
                : "Select a job to view details"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {scrapeImmediate.data ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">
                    {scrapeImmediate.data.title || "Untitled Page"}
                  </h4>
                  <p className="text-sm text-muted-foreground line-clamp-3">
                    {scrapeImmediate.data.content?.slice(0, 300)}...
                  </p>
                </div>
                <div className="flex gap-2">
                  <span className="text-xs bg-muted px-2 py-1 rounded">
                    {scrapeImmediate.data.word_count || 0} words
                  </span>
                </div>
              </div>
            ) : scrapeAndQuery.data ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Content</h4>
                  <p className="text-sm">{scrapeAndQuery.data.content?.slice(0, 500)}...</p>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Source</h4>
                  <p className="text-sm text-muted-foreground">
                    {scrapeAndQuery.data.title || "Unknown"}
                  </p>
                </div>
              </div>
            ) : extractLinks.data ? (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground mb-3">
                  Found {extractLinks.data.count} links
                </p>
                <div className="max-h-64 overflow-y-auto space-y-1">
                  {extractLinks.data.links?.slice(0, 20).map((link: string, i: number) => (
                    <a
                      key={i}
                      href={link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-sm text-primary hover:underline truncate"
                    >
                      <ExternalLink className="h-3 w-3 flex-shrink-0" />
                      {link}
                    </a>
                  ))}
                </div>
              </div>
            ) : selectedJob ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  {getStatusIcon(selectedJob.status)}
                  <span className="capitalize">{selectedJob.status}</span>
                </div>
                <div>
                  <h4 className="text-sm font-medium mb-1">Pages</h4>
                  {selectedJob.pages?.slice(0, 5).map((page, i: number) => (
                    <p key={i} className="text-sm text-muted-foreground truncate">
                      {page.url}
                    </p>
                  ))}
                </div>
                {selectedJob.pages_scraped !== undefined && (
                  <p className="text-sm">
                    Pages scraped: {selectedJob.pages_scraped} / {selectedJob.total_pages}
                  </p>
                )}
                {selectedJob.error_message && (
                  <p className="text-sm text-red-500">{selectedJob.error_message}</p>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Search className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No results yet</p>
                <p className="text-sm">Scrape a URL to see results here</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
