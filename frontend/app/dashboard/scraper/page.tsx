"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useSession } from "next-auth/react";
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
  Layers,
  Database,
  AlertCircle,
  Map,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useValidation } from "@/lib/validations/use-validation";
import { urlSchema } from "@/lib/validations";
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
  useIndexScrapeJobContent,
  useIndexScrapedPages,
} from "@/lib/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

type ScrapeMode = "immediate" | "job" | "query" | "links" | "sitemap" | "search";

interface SSEProgress {
  status: string;
  current_url: string;
  pages_found: number;
  pages_crawled: number;
  progress_pct: number;
}

export default function ScraperPage() {
  const { data: session, status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<ScrapeMode>("immediate");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  // Subpage crawling options
  const [crawlSubpages, setCrawlSubpages] = useState(false);
  const [maxDepth, setMaxDepth] = useState(2);
  const [sameDomainOnly, setSameDomainOnly] = useState(true);

  // Sitemap crawl options
  const [sitemapMaxPages, setSitemapMaxPages] = useState(100);
  const [sitemapStorageMode, setSitemapStorageMode] = useState<
    "temporary" | "permanent"
  >("permanent");
  const [sitemapFollowIndex, setSitemapFollowIndex] = useState(true);

  // Search crawl options
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMaxResults, setSearchMaxResults] = useState(5);
  const [searchStorageMode, setSearchStorageMode] = useState<
    "temporary" | "permanent"
  >("permanent");

  // Sitemap/search results
  const [sitemapResult, setSitemapResult] = useState<any>(null);
  const [searchCrawlResult, setSearchCrawlResult] = useState<any>(null);
  const [sitemapLoading, setSitemapLoading] = useState(false);
  const [searchCrawlLoading, setSearchCrawlLoading] = useState(false);
  const [sitemapError, setSitemapError] = useState<string | null>(null);
  const [searchCrawlError, setSearchCrawlError] = useState<string | null>(null);

  // SSE progress state
  const [sseProgress, setSseProgress] = useState<SSEProgress | null>(null);
  const [sseJobId, setSseJobId] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // URL validation
  const {
    validate: validateUrl,
    error: urlError,
    clearError: clearUrlError,
  } = useValidation(urlSchema);

  // Queries - only fetch when authenticated
  const {
    data: jobs,
    isLoading: jobsLoading,
    refetch: refetchJobs,
  } = useScrapeJobs(undefined, 50, { enabled: isAuthenticated });
  const { data: selectedJob } = useScrapeJob(selectedJobId || "");

  // Mutations
  const createJob = useCreateScrapeJob();
  const runJob = useRunScrapeJob();
  const scrapeImmediate = useScrapeUrlImmediate();
  const scrapeAndQuery = useScrapeAndQuery();
  const extractLinks = useExtractLinks();
  const indexJobContent = useIndexScrapeJobContent();
  const indexScrapedPages = useIndexScrapedPages();

  // Track if immediate scrape result has been indexed
  const [immediateIndexResult, setImmediateIndexResult] = useState<{
    status: string;
    documents_indexed: number;
    entities_extracted: number;
  } | null>(null);

  // SSE connection helper
  const connectSSE = useCallback(
    (jobId: string) => {
      // Close any existing connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }

      setSseJobId(jobId);
      setSseProgress({
        status: "connecting",
        current_url: "",
        pages_found: 0,
        pages_crawled: 0,
        progress_pct: 0,
      });

      const sseUrl = `${API_BASE_URL}/scraper/jobs/${jobId}/stream`;
      const es = new EventSource(sseUrl);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setSseProgress({
            status: data.status || "scraping",
            current_url: data.current_url || "",
            pages_found: data.pages_found || 0,
            pages_crawled: data.pages_crawled || 0,
            progress_pct: data.progress_pct || 0,
          });

          // Close connection when terminal state reached
          if (data.status === "completed" || data.status === "failed") {
            es.close();
            eventSourceRef.current = null;
            refetchJobs();
          }
        } catch {
          // Ignore parse errors for non-JSON messages
        }
      };

      es.onerror = () => {
        setSseProgress((prev) =>
          prev
            ? { ...prev, status: "connection_error" }
            : {
                status: "connection_error",
                current_url: "",
                pages_found: 0,
                pages_crawled: 0,
                progress_pct: 0,
              }
        );
        es.close();
        eventSourceRef.current = null;
      };
    },
    [refetchJobs]
  );

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  // Handle indexing scraped pages to RAG
  const handleIndexToRag = async () => {
    if (!scrapeImmediate.data) return;

    // Get pages from the scrape result
    const pages =
      "pages" in scrapeImmediate.data
        ? scrapeImmediate.data.pages
        : [scrapeImmediate.data];

    try {
      const result = await indexScrapedPages.mutateAsync({
        pages: pages as any[],
        sourceId: `scrape_${Date.now()}`,
      });
      setImmediateIndexResult(result);
    } catch (error) {
      console.error("Failed to index pages:", error);
    }
  };

  // Handle sitemap crawl submission
  const handleSitemapCrawl = async () => {
    const urlResult = validateUrl(url);
    if (!urlResult.success) return;

    setSitemapLoading(true);
    setSitemapError(null);
    setSitemapResult(null);

    try {
      const token = (session as any)?.accessToken;
      const response = await fetch(`${API_BASE_URL}/scraper/sitemap-crawl`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          url,
          max_pages: sitemapMaxPages,
          storage_mode: sitemapStorageMode,
          follow_index: sitemapFollowIndex,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({
          detail: "Sitemap crawl failed",
        }));
        throw new Error(errData.detail || "Sitemap crawl failed");
      }

      const data = await response.json();
      setSitemapResult(data);

      // If a job_id is returned, connect SSE for progress
      if (data.job_id) {
        setSelectedJobId(data.job_id);
        connectSSE(data.job_id);
      }

      refetchJobs();
    } catch (error: any) {
      console.error("Sitemap crawl failed:", error);
      setSitemapError(error.message || "Sitemap crawl failed");
    } finally {
      setSitemapLoading(false);
    }
  };

  // Handle search crawl submission
  const handleSearchCrawl = async () => {
    if (!searchQuery.trim()) return;

    setSearchCrawlLoading(true);
    setSearchCrawlError(null);
    setSearchCrawlResult(null);

    try {
      const token = (session as any)?.accessToken;
      const response = await fetch(`${API_BASE_URL}/scraper/search-crawl`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          query: searchQuery,
          max_results: searchMaxResults,
          storage_mode: searchStorageMode,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({
          detail: "Search crawl failed",
        }));
        throw new Error(errData.detail || "Search crawl failed");
      }

      const data = await response.json();
      setSearchCrawlResult(data);

      // If a job_id is returned, connect SSE for progress
      if (data.job_id) {
        setSelectedJobId(data.job_id);
        connectSSE(data.job_id);
      }

      refetchJobs();
    } catch (error: any) {
      console.error("Search crawl failed:", error);
      setSearchCrawlError(error.message || "Search crawl failed");
    } finally {
      setSearchCrawlLoading(false);
    }
  };

  const handleScrape = async () => {
    // Validate URL before proceeding
    const urlResult = validateUrl(url);
    if (!urlResult.success) return;

    // Reset index result when starting a new scrape
    setImmediateIndexResult(null);

    const scrapeConfig = crawlSubpages
      ? {
          crawl_subpages: true,
          max_depth: maxDepth,
          same_domain_only: sameDomainOnly,
        }
      : undefined;

    try {
      switch (mode) {
        case "immediate":
          await scrapeImmediate.mutateAsync({
            url,
            config: scrapeConfig,
          });
          break;
        case "job": {
          const job = await createJob.mutateAsync({
            urls: [url],
            storage_mode: "permanent",
            crawl_subpages: crawlSubpages,
            max_depth: maxDepth,
            same_domain_only: sameDomainOnly,
          });
          setSelectedJobId(job.id);
          connectSSE(job.id);
          await runJob.mutateAsync({
            jobId: job.id,
            crawlSubpages,
            maxDepth,
            sameDomainOnly,
          });
          break;
        }
        case "query":
          if (!query) return;
          await scrapeAndQuery.mutateAsync({
            url,
            query,
            config: scrapeConfig,
          });
          break;
        case "links":
          await extractLinks.mutateAsync({
            url,
            maxDepth: maxDepth,
            sameDomainOnly: sameDomainOnly,
          });
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
    extractLinks.isPending ||
    indexScrapedPages.isPending ||
    sitemapLoading ||
    searchCrawlLoading;

  const getStatusIcon = (jobStatus: string) => {
    switch (jobStatus) {
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
            <Button
              variant={mode === "sitemap" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("sitemap")}
            >
              <Map className="h-4 w-4 mr-2" />
              Sitemap Crawl
            </Button>
            <Button
              variant={mode === "search" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("search")}
            >
              <Search className="h-4 w-4 mr-2" />
              Search & Crawl
            </Button>
          </div>

          {/* Sitemap Crawl Mode */}
          {mode === "sitemap" && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="sitemap-url">Website URL</Label>
                <div className="flex gap-2">
                  <Input
                    id="sitemap-url"
                    placeholder="https://example.com"
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value);
                      clearUrlError();
                    }}
                    className={`flex-1 ${urlError ? "border-red-500 focus-visible:ring-red-500" : ""}`}
                  />
                  <Button
                    onClick={handleSitemapCrawl}
                    disabled={!url || isLoading}
                  >
                    {sitemapLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Map className="h-4 w-4" />
                    )}
                    <span className="ml-2">Start Sitemap Crawl</span>
                  </Button>
                </div>
                {urlError && (
                  <p className="text-sm text-red-500 flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" />
                    {urlError}
                  </p>
                )}
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="sitemap-max-pages">Max Pages</Label>
                  <Input
                    id="sitemap-max-pages"
                    type="number"
                    min={1}
                    max={10000}
                    value={sitemapMaxPages}
                    onChange={(e) =>
                      setSitemapMaxPages(parseInt(e.target.value) || 100)
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sitemap-storage">Storage Mode</Label>
                  <Select
                    value={sitemapStorageMode}
                    onValueChange={(value: "temporary" | "permanent") =>
                      setSitemapStorageMode(value)
                    }
                  >
                    <SelectTrigger id="sitemap-storage">
                      <SelectValue placeholder="Select storage mode" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="temporary">Temporary</SelectItem>
                      <SelectItem value="permanent">Permanent</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="sitemap-follow-index"
                  checked={sitemapFollowIndex}
                  onCheckedChange={(checked) =>
                    setSitemapFollowIndex(checked === true)
                  }
                />
                <Label
                  htmlFor="sitemap-follow-index"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  Follow Sitemap Index
                </Label>
              </div>

              {sitemapError && (
                <p className="text-sm text-red-500 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  {sitemapError}
                </p>
              )}
            </div>
          )}

          {/* Search & Crawl Mode */}
          {mode === "search" && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="search-query">Search Query</Label>
                <div className="flex gap-2">
                  <Input
                    id="search-query"
                    placeholder="Enter a search query..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex-1"
                  />
                  <Button
                    onClick={handleSearchCrawl}
                    disabled={!searchQuery.trim() || isLoading}
                  >
                    {searchCrawlLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Search className="h-4 w-4" />
                    )}
                    <span className="ml-2">Search & Crawl</span>
                  </Button>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="search-max-results">Max Results</Label>
                  <Input
                    id="search-max-results"
                    type="number"
                    min={1}
                    max={100}
                    value={searchMaxResults}
                    onChange={(e) =>
                      setSearchMaxResults(parseInt(e.target.value) || 5)
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="search-storage">Storage Mode</Label>
                  <Select
                    value={searchStorageMode}
                    onValueChange={(value: "temporary" | "permanent") =>
                      setSearchStorageMode(value)
                    }
                  >
                    <SelectTrigger id="search-storage">
                      <SelectValue placeholder="Select storage mode" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="temporary">Temporary</SelectItem>
                      <SelectItem value="permanent">Permanent</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {searchCrawlError && (
                <p className="text-sm text-red-500 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  {searchCrawlError}
                </p>
              )}
            </div>
          )}

          {/* Original modes: URL Input, Query, Subpage options */}
          {mode !== "sitemap" && mode !== "search" && (
            <>
              {/* URL Input */}
              <div className="space-y-2">
                <div className="flex gap-2">
                  <Input
                    placeholder="https://example.com/page"
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value);
                      clearUrlError();
                    }}
                    className={`flex-1 ${urlError ? "border-red-500 focus-visible:ring-red-500" : ""}`}
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
                {urlError && (
                  <p className="text-sm text-red-500 flex items-center gap-1">
                    <AlertCircle className="h-3 w-3" />
                    {urlError}
                  </p>
                )}
              </div>

              {/* Query Input (for query mode) */}
              {mode === "query" && (
                <Input
                  placeholder="Ask a question about the page..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              )}

              {/* Subpage Crawling Options (for all modes except links) */}
              <div className="space-y-4 p-4 rounded-lg bg-muted/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="h-4 w-4 text-muted-foreground" />
                    <Label
                      htmlFor="crawl-subpages"
                      className="text-sm font-medium"
                    >
                      Crawl Subpages
                    </Label>
                  </div>
                  <Switch
                    id="crawl-subpages"
                    checked={crawlSubpages}
                    onCheckedChange={setCrawlSubpages}
                  />
                </div>

                {crawlSubpages && (
                  <div className="space-y-4 pl-6 border-l-2 border-primary/20">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-sm">
                          Max Depth: {maxDepth}
                        </Label>
                        <span className="text-xs text-muted-foreground">
                          {maxDepth === 1
                            ? "Only starting page"
                            : `Up to ${maxDepth} levels deep`}
                        </span>
                      </div>
                      <Slider
                        min={1}
                        max={10}
                        step={1}
                        value={[maxDepth]}
                        onValueChange={(value) => setMaxDepth(value[0])}
                        className="w-full"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <Label htmlFor="same-domain" className="text-sm">
                        Same domain only
                      </Label>
                      <Switch
                        id="same-domain"
                        checked={sameDomainOnly}
                        onCheckedChange={setSameDomainOnly}
                      />
                    </div>

                    <p className="text-xs text-muted-foreground">
                      {sameDomainOnly
                        ? "Only pages from the same domain will be crawled"
                        : "Warning: Pages from any domain will be crawled"}
                    </p>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Mode Descriptions */}
          <p className="text-sm text-muted-foreground">
            {mode === "immediate" &&
              !crawlSubpages &&
              "Quick scrape extracts content without saving to the index."}
            {mode === "immediate" &&
              crawlSubpages &&
              `Crawl up to ${maxDepth} levels and display content (not saved).`}
            {mode === "job" &&
              !crawlSubpages &&
              "Save to index permanently stores the scraped content for RAG queries."}
            {mode === "job" &&
              crawlSubpages &&
              `Crawl and save up to ${maxDepth} levels of linked pages to the index.`}
            {mode === "query" &&
              !crawlSubpages &&
              "Scrape & query extracts content and immediately answers your question."}
            {mode === "query" &&
              crawlSubpages &&
              `Crawl up to ${maxDepth} levels and answer your question using all pages.`}
            {mode === "links" &&
              `Extract links discovers all URLs linked from the page (up to ${maxDepth} levels deep).`}
            {mode === "sitemap" &&
              "Sitemap crawl discovers and scrapes pages from the site's sitemap.xml for comprehensive coverage."}
            {mode === "search" &&
              "Search the web for a query and crawl the top results to add to your knowledge base."}
          </p>
        </CardContent>
      </Card>

      {/* SSE Progress Display */}
      {sseProgress && sseJobId && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {sseProgress.status === "completed" ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : sseProgress.status === "failed" ||
                sseProgress.status === "connection_error" ? (
                <XCircle className="h-5 w-5 text-red-500" />
              ) : (
                <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
              )}
              Crawl Progress
            </CardTitle>
            <CardDescription>
              Job {sseJobId.slice(0, 8)}... &mdash;{" "}
              <span className="capitalize">{sseProgress.status}</span>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={sseProgress.progress_pct} className="w-full" />
            <div className="grid gap-3 sm:grid-cols-3 text-sm">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Pages Found:</span>
                <span className="font-medium">{sseProgress.pages_found}</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Pages Crawled:</span>
                <span className="font-medium">
                  {sseProgress.pages_crawled}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Layers className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Progress:</span>
                <span className="font-medium">
                  {Math.round(sseProgress.progress_pct)}%
                </span>
              </div>
            </div>
            {sseProgress.current_url && (
              <div className="text-sm">
                <span className="text-muted-foreground">Currently crawling: </span>
                <span className="font-mono text-xs break-all">
                  {sseProgress.current_url}
                </span>
              </div>
            )}
            {(sseProgress.status === "completed" ||
              sseProgress.status === "failed" ||
              sseProgress.status === "connection_error") && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setSseProgress(null);
                  setSseJobId(null);
                }}
              >
                Dismiss
              </Button>
            )}
          </CardContent>
        </Card>
      )}

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
            {/* Sitemap result display */}
            {mode === "sitemap" && sitemapResult ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Map className="h-4 w-4" />
                  Sitemap crawl{" "}
                  {sitemapResult.status === "completed"
                    ? "completed"
                    : "started"}
                </div>
                {sitemapResult.job_id && (
                  <p className="text-xs text-muted-foreground">
                    Job ID: {sitemapResult.job_id}
                  </p>
                )}
                {sitemapResult.pages_found !== undefined && (
                  <p className="text-sm">
                    Pages found: {sitemapResult.pages_found}
                  </p>
                )}
                {sitemapResult.pages_crawled !== undefined && (
                  <p className="text-sm">
                    Pages crawled: {sitemapResult.pages_crawled}
                  </p>
                )}
                {sitemapResult.pages &&
                  Array.isArray(sitemapResult.pages) &&
                  sitemapResult.pages.length > 0 && (
                    <div className="max-h-64 overflow-y-auto space-y-2">
                      {sitemapResult.pages
                        .slice(0, 10)
                        .map((page: any, i: number) => (
                          <div
                            key={i}
                            className="p-3 rounded-lg bg-muted/30 border"
                          >
                            <a
                              href={page.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sm font-medium text-primary hover:underline flex items-center gap-1"
                            >
                              {page.title || page.url}
                              <ExternalLink className="h-3 w-3" />
                            </a>
                            {page.content && (
                              <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                {page.content.slice(0, 150)}...
                              </p>
                            )}
                            {page.word_count !== undefined && (
                              <span className="text-xs text-muted-foreground">
                                {page.word_count} words
                              </span>
                            )}
                          </div>
                        ))}
                      {sitemapResult.pages.length > 10 && (
                        <p className="text-sm text-muted-foreground text-center">
                          +{sitemapResult.pages.length - 10} more pages
                        </p>
                      )}
                    </div>
                  )}
                {sitemapResult.error && (
                  <p className="text-sm text-red-500">
                    {sitemapResult.error}
                  </p>
                )}
              </div>
            ) : /* Search crawl result display */
            mode === "search" && searchCrawlResult ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Search className="h-4 w-4" />
                  Search crawl{" "}
                  {searchCrawlResult.status === "completed"
                    ? "completed"
                    : "started"}
                </div>
                {searchCrawlResult.job_id && (
                  <p className="text-xs text-muted-foreground">
                    Job ID: {searchCrawlResult.job_id}
                  </p>
                )}
                {searchCrawlResult.results_count !== undefined && (
                  <p className="text-sm">
                    Results found: {searchCrawlResult.results_count}
                  </p>
                )}
                {searchCrawlResult.pages_crawled !== undefined && (
                  <p className="text-sm">
                    Pages crawled: {searchCrawlResult.pages_crawled}
                  </p>
                )}
                {searchCrawlResult.pages &&
                  Array.isArray(searchCrawlResult.pages) &&
                  searchCrawlResult.pages.length > 0 && (
                    <div className="max-h-64 overflow-y-auto space-y-2">
                      {searchCrawlResult.pages
                        .slice(0, 10)
                        .map((page: any, i: number) => (
                          <div
                            key={i}
                            className="p-3 rounded-lg bg-muted/30 border"
                          >
                            <a
                              href={page.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sm font-medium text-primary hover:underline flex items-center gap-1"
                            >
                              {page.title || page.url}
                              <ExternalLink className="h-3 w-3" />
                            </a>
                            {page.content && (
                              <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                {page.content.slice(0, 150)}...
                              </p>
                            )}
                            {page.word_count !== undefined && (
                              <span className="text-xs text-muted-foreground">
                                {page.word_count} words
                              </span>
                            )}
                          </div>
                        ))}
                      {searchCrawlResult.pages.length > 10 && (
                        <p className="text-sm text-muted-foreground text-center">
                          +{searchCrawlResult.pages.length - 10} more pages
                        </p>
                      )}
                    </div>
                  )}
                {searchCrawlResult.error && (
                  <p className="text-sm text-red-500">
                    {searchCrawlResult.error}
                  </p>
                )}
              </div>
            ) : scrapeImmediate.data ? (
              <div className="space-y-4">
                {/* Check if it's a multi-page response (has 'pages' property) */}
                {"pages" in scrapeImmediate.data ? (
                  <>
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Layers className="h-4 w-4" />
                      {scrapeImmediate.data.total_pages} pages scraped
                      <span className="text-muted-foreground">
                        ({scrapeImmediate.data.total_word_count} words total)
                      </span>
                    </div>
                    <div className="max-h-64 overflow-y-auto space-y-3">
                      {scrapeImmediate.data.pages
                        .slice(0, 10)
                        .map((page, i: number) => (
                          <div
                            key={i}
                            className="p-3 rounded-lg bg-muted/30 border"
                          >
                            <a
                              href={page.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sm font-medium text-primary hover:underline flex items-center gap-1"
                            >
                              {page.title || page.url}
                              <ExternalLink className="h-3 w-3" />
                            </a>
                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                              {page.content?.slice(0, 150)}...
                            </p>
                            <span className="text-xs text-muted-foreground">
                              {page.word_count} words
                            </span>
                          </div>
                        ))}
                      {scrapeImmediate.data.total_pages > 10 && (
                        <p className="text-sm text-muted-foreground text-center">
                          +{scrapeImmediate.data.total_pages - 10} more pages
                        </p>
                      )}
                    </div>
                  </>
                ) : (
                  <>
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
                  </>
                )}

                {/* Save to Index button */}
                <div className="pt-4 border-t">
                  {immediateIndexResult ? (
                    <div className="flex items-center gap-2 text-sm text-green-600">
                      <CheckCircle className="h-4 w-4" />
                      <span>
                        Indexed {immediateIndexResult.documents_indexed} chunks,{" "}
                        {immediateIndexResult.entities_extracted} entities
                        extracted
                      </span>
                    </div>
                  ) : (
                    <Button
                      onClick={handleIndexToRag}
                      disabled={indexScrapedPages.isPending}
                      variant="outline"
                      className="w-full"
                    >
                      {indexScrapedPages.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Database className="h-4 w-4 mr-2" />
                      )}
                      Save to Knowledge Base
                    </Button>
                  )}
                </div>
              </div>
            ) : scrapeAndQuery.data ? (
              <div className="space-y-4">
                {/* Answer - shown first and prominently */}
                {scrapeAndQuery.data.answer && (
                  <div>
                    <h4 className="font-medium mb-2">Answer</h4>
                    <div className="text-sm bg-muted/50 p-3 rounded-lg whitespace-pre-wrap">
                      {scrapeAndQuery.data.answer}
                    </div>
                  </div>
                )}
                {/* Source info */}
                <div>
                  <h4 className="font-medium mb-2">Source</h4>
                  <a
                    href={scrapeAndQuery.data.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-primary hover:underline flex items-center gap-1"
                  >
                    {scrapeAndQuery.data.title || scrapeAndQuery.data.url}
                    <ExternalLink className="h-3 w-3" />
                  </a>
                  <p className="text-xs text-muted-foreground mt-1">
                    {scrapeAndQuery.data.word_count} words | Model:{" "}
                    {scrapeAndQuery.data.model || "default"}
                  </p>
                </div>
                {/* Collapsible raw content */}
                <details className="text-sm">
                  <summary className="cursor-pointer font-medium text-muted-foreground hover:text-foreground">
                    View scraped content
                  </summary>
                  <p className="mt-2 text-muted-foreground line-clamp-6">
                    {scrapeAndQuery.data.content?.slice(0, 1000)}...
                  </p>
                </details>
              </div>
            ) : extractLinks.data ? (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground mb-3">
                  Found {extractLinks.data.count} links
                </p>
                <div className="max-h-64 overflow-y-auto space-y-1">
                  {extractLinks.data.links
                    ?.slice(0, 20)
                    .map((link: string, i: number) => (
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
                    <p
                      key={i}
                      className="text-sm text-muted-foreground truncate"
                    >
                      {page.url}
                    </p>
                  ))}
                </div>
                {selectedJob.pages_scraped !== undefined && (
                  <p className="text-sm">
                    Pages scraped: {selectedJob.pages_scraped} /{" "}
                    {selectedJob.total_pages}
                  </p>
                )}
                {selectedJob.error_message && (
                  <p className="text-sm text-red-500">
                    {selectedJob.error_message}
                  </p>
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
