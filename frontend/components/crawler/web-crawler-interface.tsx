"use client";

/**
 * Web Crawler Interface Component (Phase 65)
 *
 * Provides UI for web crawling capabilities:
 * - Single URL crawling
 * - Site crawling (multi-page)
 * - LLM extraction with custom schemas
 * - Query any website
 */

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Loader2,
  Globe,
  Link2,
  FileText,
  Search,
  Layers,
  ExternalLink,
  Copy,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { toast } from "sonner";

interface CrawlResult {
  url: string;
  success: boolean;
  status_code: number;
  title: string;
  word_count: number;
  crawl_time_ms: number;
  error?: string;
  content?: string;
  markdown?: string;
  extracted?: Record<string, unknown>;
  link_count: number;
  internal_link_count: number;
  external_link_count: number;
}

interface QueryResult {
  answer?: string;
  source: string;
  title?: string;
  word_count?: number;
  crawl_time_ms?: number;
  error?: string;
}

interface SiteCrawlResult {
  start_url: string;
  pages_crawled: number;
  pages_successful: number;
  total_words: number;
  crawl_time_ms: number;
  pages: CrawlResult[];
}

export function WebCrawlerInterface() {
  // Single crawl state
  const [crawlUrl, setCrawlUrl] = useState("");
  const [bypassCache, setBypassCache] = useState(false);
  const [crawlResult, setCrawlResult] = useState<CrawlResult | null>(null);
  const [isCrawling, setIsCrawling] = useState(false);

  // Extraction state
  const [extractUrl, setExtractUrl] = useState("");
  const [extractSchema, setExtractSchema] = useState(
    '{\n  "title": "str",\n  "author": "str",\n  "date": "str",\n  "content": "str"\n}'
  );
  const [extractPrompt, setExtractPrompt] = useState("");
  const [extractResult, setExtractResult] = useState<CrawlResult | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);

  // Query state
  const [queryUrl, setQueryUrl] = useState("");
  const [queryQuestion, setQueryQuestion] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  // Site crawl state
  const [siteUrl, setSiteUrl] = useState("");
  const [maxPages, setMaxPages] = useState(10);
  const [sameDomainOnly, setSameDomainOnly] = useState(true);
  const [urlPattern, setUrlPattern] = useState("");
  const [siteCrawlResult, setSiteCrawlResult] = useState<SiteCrawlResult | null>(null);
  const [isSiteCrawling, setIsSiteCrawling] = useState(false);

  // API calls
  const crawlSingleUrl = useCallback(async () => {
    if (!crawlUrl) {
      toast.error("Please enter a URL");
      return;
    }

    setIsCrawling(true);
    setCrawlResult(null);

    try {
      const response = await fetch("/api/v1/crawler/crawl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: crawlUrl, bypass_cache: bypassCache }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Crawl failed");
      }

      setCrawlResult(result);
      toast.success(`Crawled ${result.title || crawlUrl}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Crawl failed");
    } finally {
      setIsCrawling(false);
    }
  }, [crawlUrl, bypassCache]);

  const crawlWithExtraction = useCallback(async () => {
    if (!extractUrl) {
      toast.error("Please enter a URL");
      return;
    }

    setIsExtracting(true);
    setExtractResult(null);

    try {
      let schema = undefined;
      if (extractSchema.trim()) {
        try {
          schema = JSON.parse(extractSchema);
        } catch {
          toast.error("Invalid JSON schema");
          setIsExtracting(false);
          return;
        }
      }

      const response = await fetch("/api/v1/crawler/crawl/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: extractUrl,
          schema,
          extraction_prompt: extractPrompt || undefined,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Extraction failed");
      }

      setExtractResult(result);
      toast.success("Extracted structured data");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Extraction failed");
    } finally {
      setIsExtracting(false);
    }
  }, [extractUrl, extractSchema, extractPrompt]);

  const queryWebsite = useCallback(async () => {
    if (!queryUrl || !queryQuestion) {
      toast.error("Please enter both URL and question");
      return;
    }

    setIsQuerying(true);
    setQueryResult(null);

    try {
      const response = await fetch("/api/v1/crawler/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: queryUrl, question: queryQuestion }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Query failed");
      }

      setQueryResult(result);
      toast.success("Got answer");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Query failed");
    } finally {
      setIsQuerying(false);
    }
  }, [queryUrl, queryQuestion]);

  const crawlSite = useCallback(async () => {
    if (!siteUrl) {
      toast.error("Please enter a starting URL");
      return;
    }

    setIsSiteCrawling(true);
    setSiteCrawlResult(null);

    try {
      const response = await fetch("/api/v1/crawler/crawl/site", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_url: siteUrl,
          max_pages: maxPages,
          same_domain_only: sameDomainOnly,
          url_pattern: urlPattern || undefined,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Site crawl failed");
      }

      setSiteCrawlResult(result);
      toast.success(`Crawled ${result.pages_successful} pages`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Site crawl failed");
    } finally {
      setIsSiteCrawling(false);
    }
  }, [siteUrl, maxPages, sameDomainOnly, urlPattern]);

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Globe className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-2xl font-bold">Web Crawler</h1>
          <p className="text-muted-foreground">
            Crawl websites, extract structured data, and query web content
          </p>
        </div>
      </div>

      <Tabs defaultValue="crawl" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="crawl" className="flex items-center gap-2">
            <Link2 className="h-4 w-4" />
            Single URL
          </TabsTrigger>
          <TabsTrigger value="extract" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Extract Data
          </TabsTrigger>
          <TabsTrigger value="query" className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Query Website
          </TabsTrigger>
          <TabsTrigger value="site" className="flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Crawl Site
          </TabsTrigger>
        </TabsList>

        {/* Single URL Crawl */}
        <TabsContent value="crawl" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Crawl Single URL</CardTitle>
              <CardDescription>
                Crawl a webpage and extract its content, links, and metadata
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <Input
                  placeholder="https://example.com/page"
                  value={crawlUrl}
                  onChange={(e) => setCrawlUrl(e.target.value)}
                  className="flex-1"
                />
                <Button onClick={crawlSingleUrl} disabled={isCrawling}>
                  {isCrawling ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Crawling...
                    </>
                  ) : (
                    <>
                      <Globe className="mr-2 h-4 w-4" />
                      Crawl
                    </>
                  )}
                </Button>
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  id="bypass-cache"
                  checked={bypassCache}
                  onCheckedChange={setBypassCache}
                />
                <Label htmlFor="bypass-cache">Bypass cache</Label>
              </div>

              {crawlResult && (
                <Card className="mt-4">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{crawlResult.title || "Result"}</CardTitle>
                      <Badge variant={crawlResult.success ? "default" : "destructive"}>
                        {crawlResult.success ? (
                          <CheckCircle className="mr-1 h-3 w-3" />
                        ) : (
                          <AlertCircle className="mr-1 h-3 w-3" />
                        )}
                        {crawlResult.status_code}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Words:</span>
                        <span className="ml-2 font-medium">{crawlResult.word_count}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Time:</span>
                        <span className="ml-2 font-medium">{crawlResult.crawl_time_ms.toFixed(0)}ms</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Internal Links:</span>
                        <span className="ml-2 font-medium">{crawlResult.internal_link_count}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">External Links:</span>
                        <span className="ml-2 font-medium">{crawlResult.external_link_count}</span>
                      </div>
                    </div>

                    {crawlResult.markdown && (
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <Label>Content Preview</Label>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => copyToClipboard(crawlResult.markdown || "")}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                        </div>
                        <ScrollArea className="h-48 rounded-md border p-3">
                          <pre className="text-xs whitespace-pre-wrap">
                            {crawlResult.markdown.slice(0, 2000)}
                            {crawlResult.markdown.length > 2000 && "..."}
                          </pre>
                        </ScrollArea>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Extract Data */}
        <TabsContent value="extract" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>LLM Data Extraction</CardTitle>
              <CardDescription>
                Crawl a webpage and use LLM to extract structured data based on a schema
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>URL</Label>
                <Input
                  placeholder="https://example.com/article"
                  value={extractUrl}
                  onChange={(e) => setExtractUrl(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label>Extraction Schema (JSON)</Label>
                <Textarea
                  placeholder='{"title": "str", "content": "str"}'
                  value={extractSchema}
                  onChange={(e) => setExtractSchema(e.target.value)}
                  rows={4}
                  className="font-mono text-sm"
                />
              </div>

              <div className="space-y-2">
                <Label>Custom Extraction Prompt (optional)</Label>
                <Textarea
                  placeholder="Extract the main article content, ignoring navigation and ads..."
                  value={extractPrompt}
                  onChange={(e) => setExtractPrompt(e.target.value)}
                  rows={2}
                />
              </div>

              <Button onClick={crawlWithExtraction} disabled={isExtracting}>
                {isExtracting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Extracting...
                  </>
                ) : (
                  <>
                    <FileText className="mr-2 h-4 w-4" />
                    Extract Data
                  </>
                )}
              </Button>

              {extractResult?.extracted && (
                <Card className="mt-4">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Extracted Data</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-64 rounded-md border p-3">
                      <pre className="text-sm whitespace-pre-wrap">
                        {JSON.stringify(extractResult.extracted, null, 2)}
                      </pre>
                    </ScrollArea>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Query Website */}
        <TabsContent value="query" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Query Any Website</CardTitle>
              <CardDescription>
                Ask a question about any webpage and get an AI-powered answer
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Website URL</Label>
                <Input
                  placeholder="https://example.com"
                  value={queryUrl}
                  onChange={(e) => setQueryUrl(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label>Your Question</Label>
                <Textarea
                  placeholder="What are the main features mentioned on this page?"
                  value={queryQuestion}
                  onChange={(e) => setQueryQuestion(e.target.value)}
                  rows={3}
                />
              </div>

              <Button onClick={queryWebsite} disabled={isQuerying}>
                {isQuerying ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Querying...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    Query Website
                  </>
                )}
              </Button>

              {queryResult && (
                <Card className="mt-4">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Answer</CardTitle>
                      {queryResult.title && (
                        <Badge variant="secondary">{queryResult.title}</Badge>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {queryResult.error ? (
                      <div className="text-red-500">{queryResult.error}</div>
                    ) : (
                      <>
                        <p className="whitespace-pre-wrap">{queryResult.answer}</p>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <a
                            href={queryResult.source}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center hover:text-primary"
                          >
                            <ExternalLink className="mr-1 h-3 w-3" />
                            Source
                          </a>
                          {queryResult.word_count && <span>{queryResult.word_count} words</span>}
                          {queryResult.crawl_time_ms && (
                            <span>{queryResult.crawl_time_ms.toFixed(0)}ms</span>
                          )}
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Site Crawl */}
        <TabsContent value="site" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Crawl Multiple Pages</CardTitle>
              <CardDescription>
                Crawl multiple pages from a website following internal links
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Starting URL</Label>
                <Input
                  placeholder="https://example.com"
                  value={siteUrl}
                  onChange={(e) => setSiteUrl(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label>Max Pages: {maxPages}</Label>
                <Slider
                  value={[maxPages]}
                  onValueChange={(v) => setMaxPages(v[0])}
                  min={1}
                  max={100}
                  step={1}
                />
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  id="same-domain"
                  checked={sameDomainOnly}
                  onCheckedChange={setSameDomainOnly}
                />
                <Label htmlFor="same-domain">Same domain only</Label>
              </div>

              <div className="space-y-2">
                <Label>URL Pattern (regex, optional)</Label>
                <Input
                  placeholder="/blog/.*"
                  value={urlPattern}
                  onChange={(e) => setUrlPattern(e.target.value)}
                />
              </div>

              <Button onClick={crawlSite} disabled={isSiteCrawling}>
                {isSiteCrawling ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Crawling Site...
                  </>
                ) : (
                  <>
                    <Layers className="mr-2 h-4 w-4" />
                    Start Crawl
                  </>
                )}
              </Button>

              {siteCrawlResult && (
                <Card className="mt-4">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Site Crawl Results</CardTitle>
                    <div className="flex gap-4 text-sm">
                      <Badge variant="default">
                        {siteCrawlResult.pages_successful}/{siteCrawlResult.pages_crawled} pages
                      </Badge>
                      <Badge variant="secondary">{siteCrawlResult.total_words} total words</Badge>
                      <Badge variant="outline">
                        {(siteCrawlResult.crawl_time_ms / 1000).toFixed(1)}s
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-64">
                      <div className="space-y-2">
                        {siteCrawlResult.pages.map((page, i) => (
                          <div
                            key={i}
                            className="flex items-center justify-between p-2 rounded-md border"
                          >
                            <div className="flex-1 min-w-0">
                              <p className="font-medium truncate">{page.title || page.url}</p>
                              <p className="text-sm text-muted-foreground truncate">{page.url}</p>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant={page.success ? "default" : "destructive"} className="ml-2">
                                {page.status_code}
                              </Badge>
                              <span className="text-sm text-muted-foreground">{page.word_count} words</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default WebCrawlerInterface;
