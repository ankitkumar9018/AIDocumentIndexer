"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SmartCollections } from "@/components/intelligence/smart-collections";
import { InsightFeed } from "@/components/intelligence/insight-feed";
import { ConflictDetector } from "@/components/intelligence/conflict-detector";
import { DocumentDNA } from "@/components/intelligence/document-dna";
import { SmartHighlights } from "@/components/intelligence/smart-highlights";
import {
  Brain,
  FolderKanban,
  Lightbulb,
  AlertTriangle,
  Dna,
  Highlighter
} from "lucide-react";

export default function IntelligencePage() {
  return (
    <div className="container py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          <Brain className="h-8 w-8" />
          Intelligence Hub
        </h1>
        <p className="text-muted-foreground">
          AI-powered insights, analysis, and organization for your knowledge base
        </p>
      </div>

      <Tabs defaultValue="collections" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="collections" className="flex items-center gap-2">
            <FolderKanban className="h-4 w-4" />
            <span className="hidden sm:inline">Collections</span>
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex items-center gap-2">
            <Lightbulb className="h-4 w-4" />
            <span className="hidden sm:inline">Insights</span>
          </TabsTrigger>
          <TabsTrigger value="conflicts" className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            <span className="hidden sm:inline">Conflicts</span>
          </TabsTrigger>
          <TabsTrigger value="dna" className="flex items-center gap-2">
            <Dna className="h-4 w-4" />
            <span className="hidden sm:inline">DNA</span>
          </TabsTrigger>
          <TabsTrigger value="highlights" className="flex items-center gap-2">
            <Highlighter className="h-4 w-4" />
            <span className="hidden sm:inline">Highlights</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="collections">
          <SmartCollections />
        </TabsContent>

        <TabsContent value="insights">
          <InsightFeed />
        </TabsContent>

        <TabsContent value="conflicts">
          <ConflictDetector />
        </TabsContent>

        <TabsContent value="dna">
          <DocumentDNA />
        </TabsContent>

        <TabsContent value="highlights">
          <SmartHighlights />
        </TabsContent>
      </Tabs>
    </div>
  );
}
