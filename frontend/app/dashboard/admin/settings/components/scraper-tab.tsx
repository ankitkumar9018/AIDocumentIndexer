"use client";

import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Globe, Shield, Zap } from "lucide-react";

interface ScraperTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function ScraperTab({ localSettings, handleSettingChange }: ScraperTabProps) {
  return (
    <TabsContent value="scraper" className="space-y-6">
      {/* Core Scraping Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Globe className="h-5 w-5" />
            Web Scraping Engine
          </CardTitle>
          <CardDescription>
            Configure the web crawler for scraping and indexing external content
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Use Crawl4AI</p>
              <p className="text-sm text-muted-foreground">
                Advanced scraper with JS rendering, anti-bot bypass, LLM-optimized output
              </p>
            </div>
            <Switch
              checked={localSettings["scraping.use_crawl4ai"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("scraping.use_crawl4ai", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Headless Browser</p>
              <p className="text-sm text-muted-foreground">
                Run browser without visible window (disable for debugging)
              </p>
            </div>
            <Switch
              checked={localSettings["scraping.headless_browser"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("scraping.headless_browser", checked)}
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="text-sm font-medium">Page Load Timeout (seconds)</label>
              <Input
                type="number"
                min="10"
                max="120"
                value={localSettings["scraping.timeout_seconds"] as number ?? 30}
                onChange={(e) => handleSettingChange("scraping.timeout_seconds", parseInt(e.target.value))}
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Max Crawl Depth</label>
              <Input
                type="number"
                min="0"
                max="5"
                value={localSettings["scraping.max_depth"] as number ?? 2}
                onChange={(e) => handleSettingChange("scraping.max_depth", parseInt(e.target.value))}
                className="mt-1"
              />
              <p className="text-xs text-muted-foreground mt-1">0 = single page only</p>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="text-sm font-medium">Rate Limit (ms between requests)</label>
              <Input
                type="number"
                min="500"
                max="5000"
                step="100"
                value={localSettings["scraping.rate_limit_ms"] as number ?? 1000}
                onChange={(e) => handleSettingChange("scraping.rate_limit_ms", parseInt(e.target.value))}
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Max Pages per Site</label>
              <Input
                type="number"
                min="10"
                max="1000"
                value={localSettings["crawler.max_pages_per_site"] as number ?? 100}
                onChange={(e) => handleSettingChange("crawler.max_pages_per_site", parseInt(e.target.value))}
                className="mt-1"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Content Extraction */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Content Extraction
          </CardTitle>
          <CardDescription>
            Control what gets extracted from crawled pages
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Extract Links</p>
              <p className="text-sm text-muted-foreground">Index links found on scraped pages</p>
            </div>
            <Switch
              checked={localSettings["scraping.extract_links"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("scraping.extract_links", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Extract Images</p>
              <p className="text-sm text-muted-foreground">Extract image URLs from pages</p>
            </div>
            <Switch
              checked={localSettings["scraping.extract_images"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("scraping.extract_images", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">LLM Content Extraction</p>
              <p className="text-sm text-muted-foreground">Use LLM to extract structured data from pages</p>
            </div>
            <Switch
              checked={localSettings["crawler.llm_extraction_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("crawler.llm_extraction_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Smart Extraction</p>
              <p className="text-sm text-muted-foreground">Auto-detect entities, facts, dates from pages</p>
            </div>
            <Switch
              checked={localSettings["crawler.smart_extraction_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("crawler.smart_extraction_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Respect robots.txt</p>
              <p className="text-sm text-muted-foreground">Follow website crawling rules</p>
            </div>
            <Switch
              checked={localSettings["scraping.respect_robots_txt"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("scraping.respect_robots_txt", checked)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Anti-Detection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Anti-Detection
          </CardTitle>
          <CardDescription>
            Stealth features to avoid bot detection on protected sites
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Stealth Mode</p>
              <p className="text-sm text-muted-foreground">
                Fingerprint spoofing and user simulation for anti-bot bypass
              </p>
            </div>
            <Switch
              checked={localSettings["crawler.stealth_mode_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("crawler.stealth_mode_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Magic Mode</p>
              <p className="text-sm text-muted-foreground">
                Crawl4AI advanced anti-detection (enhanced stealth)
              </p>
            </div>
            <Switch
              checked={localSettings["crawler.magic_mode_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("crawler.magic_mode_enabled", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">User Agent Rotation</p>
              <p className="text-sm text-muted-foreground">
                Rotate browser user agents to avoid detection
              </p>
            </div>
            <Switch
              checked={localSettings["crawler.user_agent_rotation_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("crawler.user_agent_rotation_enabled", checked)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Proxy Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Proxy Configuration
          </CardTitle>
          <CardDescription>
            Configure proxy servers to distribute requests and avoid IP blocking
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Proxy Rotation</p>
              <p className="text-sm text-muted-foreground">
                Use proxy servers to distribute requests and avoid IP blocking
              </p>
            </div>
            <Switch
              checked={localSettings["scraping.proxy_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("scraping.proxy_enabled", checked)}
            />
          </div>

          {(localSettings["scraping.proxy_enabled"] as boolean) && (
            <>
              <div className="space-y-1">
                <label className="text-sm font-medium">Proxy List</label>
                <p className="text-xs text-muted-foreground">
                  Comma-separated proxy URLs (e.g., http://proxy1:8080,socks5://proxy2:1080)
                </p>
                <Input
                  value={localSettings["scraping.proxy_list"] as string ?? ""}
                  onChange={(e) => handleSettingChange("scraping.proxy_list", e.target.value)}
                  placeholder="http://proxy1:8080, http://proxy2:8080"
                />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium">Rotation Strategy</label>
                <Select
                  value={localSettings["scraping.proxy_rotation_strategy"] as string ?? "round_robin"}
                  onValueChange={(v) => handleSettingChange("scraping.proxy_rotation_strategy", v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="round_robin">Round Robin</SelectItem>
                    <SelectItem value="random">Random</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Advanced Crawling */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Advanced Crawling
          </CardTitle>
          <CardDescription>
            Advanced features for more reliable and intelligent crawling
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Jina Reader Fallback</p>
              <p className="text-sm text-muted-foreground">
                Use Jina Reader API as fallback when Crawl4AI fails (free: 1000 req/day)
              </p>
            </div>
            <Switch
              checked={localSettings["scraping.jina_reader_fallback"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("scraping.jina_reader_fallback", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Adaptive Crawling</p>
              <p className="text-sm text-muted-foreground">
                Automatically stop crawling when site is sufficiently covered
              </p>
            </div>
            <Switch
              checked={localSettings["scraping.adaptive_crawling"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("scraping.adaptive_crawling", checked)}
            />
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Crash Recovery</p>
              <p className="text-sm text-muted-foreground">
                Persist crawl state to resume after failures
              </p>
            </div>
            <Switch
              checked={localSettings["scraping.crash_recovery_enabled"] as boolean ?? false}
              onCheckedChange={(checked) => handleSettingChange("scraping.crash_recovery_enabled", checked)}
            />
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
