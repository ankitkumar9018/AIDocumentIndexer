"use client";

import { useState, useCallback } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { UploadedDocumentPreview } from "@/components/preview/UploadedDocumentPreview";
import {
  Maximize2,
  Download,
  Edit,
  Database,
  AlertCircle,
  Sparkles,
  Network,
  Globe,
  FileText,
  ArrowRight,
  Pencil,
  Plus,
  Trash2,
  Save,
  X,
  Loader2,
  Wand2,
  MoreVertical,
  Star,
  Images,
  RefreshCw,
} from "lucide-react";
import { Document, useDocumentEntities } from "@/lib/api";
import { api } from "@/lib/api/client";
import { toast } from "sonner";

interface DocumentDetailPanelProps {
  document: Document | null;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onOpenFullView: (docId: string) => void;
  onDownload: (docId: string, name: string) => void;
  onEditTags: (doc: Document) => void;
  onGenerateEmbeddings: (docId: string) => void;
  onRefetch: () => void;
  // Additional actions matching context menu
  onToggleFavorite?: (docId: string) => void;
  isFavorite?: boolean;
  onAutoTag?: (docId: string, name: string) => void;
  isAutoTagging?: boolean;
  onReanalyzeImages?: (doc: Document) => void;
  onExtractKG?: (docId: string, name: string) => void;
  isExtractingKG?: boolean;
  onReprocess?: (docId: string, name: string) => void;
  isReprocessing?: boolean;
  onDelete?: (docId: string, name: string) => void;
}

export function DocumentDetailPanel({
  document: doc,
  isOpen,
  onOpenChange,
  onOpenFullView,
  onDownload,
  onEditTags,
  onGenerateEmbeddings,
  onRefetch,
  onToggleFavorite,
  isFavorite,
  onAutoTag,
  isAutoTagging,
  onReanalyzeImages,
  onExtractKG,
  isExtractingKG,
  onReprocess,
  isReprocessing,
  onDelete,
}: DocumentDetailPanelProps) {
  const [activeTab, setActiveTab] = useState("preview");
  const [isEditingQuestions, setIsEditingQuestions] = useState(false);
  const [editedQuestions, setEditedQuestions] = useState<string[]>([]);
  const [isSavingQuestions, setIsSavingQuestions] = useState(false);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);

  // Lazy-load KG entities only when that tab is active
  const {
    data: kgData,
    isLoading: kgLoading,
  } = useDocumentEntities(doc?.id || "", {
    enabled: activeTab === "knowledge-graph" && !!doc?.id,
  });

  const startEditingQuestions = useCallback(() => {
    if (doc?.enhanced_metadata?.hypothetical_questions) {
      setEditedQuestions([...doc.enhanced_metadata.hypothetical_questions]);
    } else {
      setEditedQuestions([""]);
    }
    setIsEditingQuestions(true);
  }, [doc?.enhanced_metadata?.hypothetical_questions]);

  const handleSaveQuestions = useCallback(async () => {
    if (!doc) return;
    const filtered = editedQuestions.filter(q => q.trim().length > 0);
    if (filtered.length === 0) {
      toast.error("At least one question is required");
      return;
    }
    setIsSavingQuestions(true);
    try {
      await api.updateHypotheticalQuestions(doc.id, filtered);
      toast.success("Hypothetical questions updated");
      setIsEditingQuestions(false);
      onRefetch();
    } catch (e: any) {
      toast.error("Failed to save questions", { description: e.message });
    } finally {
      setIsSavingQuestions(false);
    }
  }, [doc, editedQuestions, onRefetch]);

  const handleGenerateMore = useCallback(async () => {
    if (!doc) return;
    setIsGeneratingQuestions(true);
    try {
      const result = await api.generateMoreQuestions(doc.id, 5);
      toast.success(`Generated ${result.count} new question${result.count !== 1 ? "s" : ""}`);
      onRefetch();
    } catch (e: any) {
      toast.error("Failed to generate questions", { description: e.message });
    } finally {
      setIsGeneratingQuestions(false);
    }
  }, [doc, onRefetch]);

  if (!doc) return null;

  const enhanced = doc.enhanced_metadata;
  const hasEnhanced = !!enhanced && !!enhanced.summary_short;
  const hasKg = (doc.kg_entity_count || 0) > 0;

  return (
    <Sheet open={isOpen} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-[480px] sm:w-[540px] sm:max-w-[50vw] p-0">
        <div className="flex flex-col h-full">
          <SheetHeader className="px-6 py-4 border-b">
            <div className="flex items-center justify-between">
              <SheetTitle className="text-base truncate pr-4">
                {doc.name}
              </SheetTitle>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Button
                size="sm"
                variant="default"
                onClick={() => onOpenFullView(doc.id)}
              >
                <Maximize2 className="h-3 w-3 mr-1" />
                Full Screen
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onDownload(doc.id, doc.name)}
              >
                <Download className="h-3 w-3 mr-1" />
                Download
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onEditTags(doc)}
              >
                <Edit className="h-3 w-3 mr-1" />
                Tags
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button size="sm" variant="outline" className="px-2">
                    <MoreVertical className="h-3.5 w-3.5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {onToggleFavorite && (
                    <DropdownMenuItem onClick={() => onToggleFavorite(doc.id)}>
                      <Star className={`h-4 w-4 mr-2 ${isFavorite ? "fill-yellow-400 text-yellow-400" : ""}`} />
                      {isFavorite ? "Remove from Favorites" : "Add to Favorites"}
                    </DropdownMenuItem>
                  )}
                  {onAutoTag && (
                    <DropdownMenuItem
                      onClick={() => onAutoTag(doc.id, doc.name)}
                      disabled={isAutoTagging}
                    >
                      <Sparkles className="h-4 w-4 mr-2" />
                      Auto-generate Tags
                    </DropdownMenuItem>
                  )}
                  {onReanalyzeImages && doc.images_extracted_count > 0 && (
                    <DropdownMenuItem onClick={() => onReanalyzeImages(doc)}>
                      <Images className="h-4 w-4 mr-2" />
                      {doc.image_analysis_status === "completed" ? "Re-analyze Images" : "Analyze Images"}
                    </DropdownMenuItem>
                  )}
                  {onExtractKG && (
                    <DropdownMenuItem
                      onClick={() => onExtractKG(doc.id, doc.name)}
                      disabled={isExtractingKG || doc.kg_extraction_status === "processing"}
                    >
                      <Network className="h-4 w-4 mr-2" />
                      {doc.kg_extraction_status === "completed" ? "Re-extract KG" : "Extract KG"}
                    </DropdownMenuItem>
                  )}
                  {onReprocess && (
                    <DropdownMenuItem
                      onClick={() => onReprocess(doc.id, doc.name)}
                      disabled={isReprocessing}
                    >
                      <RefreshCw className={`h-4 w-4 mr-2 ${isReprocessing ? "animate-spin" : ""}`} />
                      Reprocess Document
                    </DropdownMenuItem>
                  )}
                  {onDelete && (
                    <>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive focus:text-destructive"
                        onClick={() => onDelete(doc.id, doc.name)}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </SheetHeader>

          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="flex-1 flex flex-col min-h-0"
          >
            <TabsList className="mx-6 mt-3 w-fit">
              <TabsTrigger value="preview">Preview</TabsTrigger>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="insights">
                Insights
                {hasEnhanced && (
                  <Sparkles className="h-3 w-3 ml-1 text-amber-500" />
                )}
              </TabsTrigger>
              <TabsTrigger value="knowledge-graph">
                KG
                {hasKg && (
                  <Network className="h-3 w-3 ml-1 text-blue-500" />
                )}
              </TabsTrigger>
            </TabsList>

            {/* Preview Tab */}
            <TabsContent value="preview" className="flex-1 min-h-0 px-6 py-3">
              <ScrollArea className="h-full">
                <UploadedDocumentPreview
                  documentId={doc.id}
                  fileName={doc.name}
                  fileType={doc.file_type || doc.name?.split(".").pop() || "txt"}
                  className="min-h-[300px]"
                />
              </ScrollArea>
            </TabsContent>

            {/* Overview Tab */}
            <TabsContent value="overview" className="flex-1 min-h-0 px-6 py-3">
              <ScrollArea className="h-full">
                <div className="space-y-4 text-sm">
                  {/* Enhanced Summary */}
                  {enhanced?.summary_short && (
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <p className="font-medium text-muted-foreground mb-1">Summary</p>
                      <p>{enhanced.summary_short}</p>
                      {enhanced.summary_detailed && enhanced.summary_detailed !== enhanced.summary_short && (
                        <details className="mt-2">
                          <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                            Show detailed summary
                          </summary>
                          <p className="mt-2 text-muted-foreground">{enhanced.summary_detailed}</p>
                        </details>
                      )}
                    </div>
                  )}

                  {/* File Info */}
                  <div>
                    <p className="font-medium text-muted-foreground">Filename</p>
                    <p>{doc.name}</p>
                  </div>
                  <div className="flex gap-6">
                    <div>
                      <p className="font-medium text-muted-foreground">Type</p>
                      <p>{doc.file_type || "Unknown"}</p>
                    </div>
                    {doc.file_size > 0 && (
                      <div>
                        <p className="font-medium text-muted-foreground">Size</p>
                        <p>
                          {doc.file_size > 1048576
                            ? `${(doc.file_size / 1048576).toFixed(1)} MB`
                            : `${(doc.file_size / 1024).toFixed(1)} KB`}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Language & Document Type badges */}
                  {(enhanced?.language || enhanced?.document_type) && (
                    <div className="flex items-center gap-2">
                      {enhanced.language && (
                        <Badge variant="outline" className="gap-1">
                          <Globe className="h-3 w-3" />
                          {enhanced.language}
                        </Badge>
                      )}
                      {enhanced.document_type && (
                        <Badge variant="outline" className="gap-1">
                          <FileText className="h-3 w-3" />
                          {enhanced.document_type}
                        </Badge>
                      )}
                    </div>
                  )}

                  {doc.collection && (
                    <div>
                      <p className="font-medium text-muted-foreground">Collection</p>
                      <p>{doc.collection}</p>
                    </div>
                  )}

                  {doc.tags && doc.tags.length > 0 && (
                    <div>
                      <p className="font-medium text-muted-foreground">Tags</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {doc.tags.map((tag: string) => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {doc.created_at && (
                    <div>
                      <p className="font-medium text-muted-foreground">Created</p>
                      <p>{new Date(doc.created_at).toLocaleString()}</p>
                    </div>
                  )}

                  {/* Status Row */}
                  <div className="space-y-2 pt-2 border-t">
                    <p className="font-medium text-muted-foreground">Status</p>

                    {/* Embedding Coverage */}
                    {doc.chunk_count > 0 && (
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className={`flex items-center gap-1 text-xs ${
                              doc.has_all_embeddings
                                ? "text-green-600 dark:text-green-400"
                                : "text-amber-600 dark:text-amber-400"
                            }`}
                          >
                            {doc.has_all_embeddings ? (
                              <Database className="h-3.5 w-3.5" />
                            ) : (
                              <AlertCircle className="h-3.5 w-3.5" />
                            )}
                            Embeddings: {doc.embedding_coverage?.toFixed(0) || 0}% ({doc.embedding_count}/{doc.chunk_count})
                          </span>
                        </div>
                        {!doc.has_all_embeddings && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onGenerateEmbeddings(doc.id)}
                            className="h-6 text-xs"
                          >
                            Generate
                          </Button>
                        )}
                      </div>
                    )}

                    {/* KG Status */}
                    <div className="flex items-center gap-2 text-xs">
                      <Network className="h-3.5 w-3.5 text-muted-foreground" />
                      <span>
                        Knowledge Graph:{" "}
                        {doc.kg_extraction_status === "completed" ? (
                          <span className="text-green-600 dark:text-green-400">
                            {doc.kg_entity_count} entities, {doc.kg_relation_count} relations
                          </span>
                        ) : doc.kg_extraction_status === "processing" ? (
                          <span className="text-blue-600">Processing...</span>
                        ) : (
                          <span className="text-muted-foreground">Not extracted</span>
                        )}
                      </span>
                    </div>

                    {/* Enhancement Status */}
                    <div className="flex items-center gap-2 text-xs">
                      <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
                      <span>
                        Enhancement:{" "}
                        {hasEnhanced ? (
                          <span className="text-green-600 dark:text-green-400">
                            Enhanced
                            {enhanced?.enhanced_at && (
                              <span className="text-muted-foreground ml-1">
                                ({new Date(enhanced.enhanced_at).toLocaleDateString()})
                              </span>
                            )}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">Not enhanced</span>
                        )}
                      </span>
                    </div>

                    {/* Image Analysis */}
                    {doc.images_extracted_count > 0 && (
                      <div className="flex items-center gap-2 text-xs">
                        <Database className="h-3.5 w-3.5 text-muted-foreground" />
                        <span>
                          Images:{" "}
                          {doc.image_analysis_status === "completed" ? (
                            <span className="text-green-600 dark:text-green-400">
                              {doc.images_analyzed_count}/{doc.images_extracted_count} analyzed
                            </span>
                          ) : doc.image_analysis_status === "processing" ? (
                            <span className="text-blue-600">Analyzing...</span>
                          ) : (
                            <span className="text-muted-foreground">
                              {doc.images_extracted_count} found, not analyzed
                            </span>
                          )}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </ScrollArea>
            </TabsContent>

            {/* Insights Tab */}
            <TabsContent value="insights" className="flex-1 min-h-0 px-6 py-3">
              <ScrollArea className="h-full">
                {hasEnhanced ? (
                  <div className="space-y-5 text-sm">
                    {/* Keywords */}
                    {enhanced?.keywords && enhanced.keywords.length > 0 && (
                      <div>
                        <p className="font-medium text-muted-foreground mb-2">Keywords</p>
                        <div className="flex flex-wrap gap-1.5">
                          {enhanced.keywords.map((kw: string) => (
                            <Badge key={kw} variant="secondary" className="text-xs">
                              {kw}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Topics */}
                    {enhanced?.topics && enhanced.topics.length > 0 && (
                      <div>
                        <p className="font-medium text-muted-foreground mb-2">Topics</p>
                        <div className="flex flex-wrap gap-1.5">
                          {enhanced.topics.map((topic: string) => (
                            <Badge key={topic} variant="outline" className="text-xs bg-blue-50 dark:bg-blue-950/30">
                              {topic}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Entities */}
                    {enhanced?.entities && Object.keys(enhanced.entities).length > 0 && (
                      <div>
                        <p className="font-medium text-muted-foreground mb-2">Entities</p>
                        <div className="space-y-2">
                          {Object.entries(enhanced.entities).map(([type, items]) => (
                            <div key={type}>
                              <p className="text-xs font-medium text-muted-foreground capitalize mb-1">
                                {type}
                              </p>
                              <div className="flex flex-wrap gap-1">
                                {(items as string[]).map((item: string) => (
                                  <Badge
                                    key={item}
                                    variant="outline"
                                    className="text-xs"
                                  >
                                    {item}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Hypothetical Questions */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <p className="font-medium text-muted-foreground">Hypothetical Questions</p>
                        {!isEditingQuestions ? (
                          <div className="flex items-center gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 text-xs"
                              onClick={handleGenerateMore}
                              disabled={isGeneratingQuestions}
                            >
                              {isGeneratingQuestions ? (
                                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                              ) : (
                                <Wand2 className="h-3 w-3 mr-1" />
                              )}
                              {isGeneratingQuestions ? "Generating..." : "Generate More"}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 text-xs"
                              onClick={startEditingQuestions}
                            >
                              <Pencil className="h-3 w-3 mr-1" />
                              Edit
                            </Button>
                          </div>
                        ) : (
                          <div className="flex items-center gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 text-xs"
                              onClick={() => setIsEditingQuestions(false)}
                              disabled={isSavingQuestions}
                            >
                              <X className="h-3 w-3 mr-1" />
                              Cancel
                            </Button>
                            <Button
                              size="sm"
                              variant="default"
                              className="h-7 text-xs"
                              onClick={handleSaveQuestions}
                              disabled={isSavingQuestions}
                            >
                              {isSavingQuestions ? (
                                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                              ) : (
                                <Save className="h-3 w-3 mr-1" />
                              )}
                              Save
                            </Button>
                          </div>
                        )}
                      </div>

                      {isEditingQuestions ? (
                        <div className="space-y-2">
                          {editedQuestions.map((q, i) => (
                            <div key={i} className="flex items-start gap-2">
                              <span className="text-xs text-muted-foreground mt-2 min-w-[20px]">{i + 1}.</span>
                              <Textarea
                                value={q}
                                onChange={(e) => {
                                  const updated = [...editedQuestions];
                                  updated[i] = e.target.value;
                                  setEditedQuestions(updated);
                                }}
                                className="min-h-[60px] text-sm"
                                placeholder="Enter a question..."
                              />
                              {editedQuestions.length > 1 && (
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  className="h-8 w-8 mt-0.5 text-destructive hover:text-destructive"
                                  onClick={() => {
                                    setEditedQuestions(editedQuestions.filter((_, idx) => idx !== i));
                                  }}
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              )}
                            </div>
                          ))}
                          <Button
                            size="sm"
                            variant="outline"
                            className="w-full h-8 text-xs"
                            onClick={() => setEditedQuestions([...editedQuestions, ""])}
                          >
                            <Plus className="h-3 w-3 mr-1" />
                            Add Question
                          </Button>
                          <p className="text-xs text-muted-foreground italic">
                            Changes will re-embed questions in the vector store for retrieval.
                          </p>
                        </div>
                      ) : enhanced?.hypothetical_questions && enhanced.hypothetical_questions.length > 0 ? (
                        <>
                          <ol className="list-decimal list-inside space-y-1.5 text-sm text-muted-foreground">
                            {enhanced.hypothetical_questions.map((q: string, i: number) => (
                              <li key={i}>{q}</li>
                            ))}
                          </ol>
                          <p className="text-xs text-muted-foreground mt-2 italic">
                            These questions are embedded in the vector store for improved retrieval.
                          </p>
                        </>
                      ) : (
                        <div className="flex flex-col items-center gap-2 py-4">
                          <p className="text-sm text-muted-foreground">No hypothetical questions generated yet.</p>
                          <Button
                            size="sm"
                            variant="outline"
                            className="text-xs"
                            onClick={handleGenerateMore}
                            disabled={isGeneratingQuestions}
                          >
                            {isGeneratingQuestions ? (
                              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                            ) : (
                              <Wand2 className="h-3 w-3 mr-1" />
                            )}
                            {isGeneratingQuestions ? "Generating..." : "Generate with LLM"}
                          </Button>
                        </div>
                      )}
                    </div>

                    {/* Model info */}
                    {enhanced?.model_used && (
                      <div className="pt-2 border-t text-xs text-muted-foreground">
                        Enhanced with {enhanced.model_used}
                        {enhanced.enhanced_at && (
                          <> on {new Date(enhanced.enhanced_at).toLocaleString()}</>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
                    <Sparkles className="h-10 w-10 mb-3 opacity-30" />
                    <p className="font-medium">Not yet enhanced</p>
                    <p className="text-sm mt-1">
                      Enhance this document to extract keywords, topics, entities, and hypothetical questions.
                    </p>
                  </div>
                )}
              </ScrollArea>
            </TabsContent>

            {/* Knowledge Graph Tab */}
            <TabsContent value="knowledge-graph" className="flex-1 min-h-0 px-6 py-3">
              <ScrollArea className="h-full">
                {kgLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full" />
                  </div>
                ) : kgData?.entities && kgData.entities.length > 0 ? (
                  <div className="space-y-5 text-sm">
                    {/* Entities */}
                    <div>
                      <p className="font-medium text-muted-foreground mb-2">
                        Entities ({kgData.entities.length})
                      </p>
                      <div className="space-y-2">
                        {kgData.entities.map((entity: any) => (
                          <div
                            key={entity.id}
                            className="p-2 rounded border bg-muted/30"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium">{entity.name}</span>
                              <Badge variant="outline" className="text-xs capitalize">
                                {entity.entity_type}
                              </Badge>
                            </div>
                            {entity.description && (
                              <p className="text-xs text-muted-foreground mt-1">
                                {entity.description}
                              </p>
                            )}
                            <p className="text-xs text-muted-foreground mt-1">
                              {entity.mention_count} mention{entity.mention_count !== 1 ? "s" : ""}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Relations */}
                    {kgData.relations && kgData.relations.length > 0 && (
                      <div>
                        <p className="font-medium text-muted-foreground mb-2">
                          Relations ({kgData.relations.length})
                        </p>
                        <div className="space-y-1.5">
                          {kgData.relations.map((rel: any) => (
                            <div
                              key={rel.id}
                              className="flex items-center gap-2 p-2 rounded border bg-muted/30 text-xs"
                            >
                              <span className="font-medium">{rel.source_entity_name}</span>
                              <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                              <Badge variant="outline" className="text-xs flex-shrink-0">
                                {rel.relation_type}
                              </Badge>
                              <ArrowRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                              <span className="font-medium">{rel.target_entity_name}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
                    <Network className="h-10 w-10 mb-3 opacity-30" />
                    <p className="font-medium">No knowledge graph data</p>
                    <p className="text-sm mt-1">
                      Extract knowledge graph to see entities and relationships from this document.
                    </p>
                  </div>
                )}
              </ScrollArea>
            </TabsContent>
          </Tabs>
        </div>
      </SheetContent>
    </Sheet>
  );
}
