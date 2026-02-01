"use client";

import { useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Save,
  Download,
  Share2,
  Plus,
  Trash2,
  GripVertical,
  ChevronDown,
  ChevronRight,
  Sparkles,
  RefreshCw,
  FileText,
  ExternalLink,
  MoreVertical,
  Edit,
  Copy,
  Eye,
  EyeOff,
  MessageSquare,
  Loader2,
  Check,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface Citation {
  id: string;
  documentId: string;
  documentTitle: string;
  snippet: string;
  pageNumber?: number;
  relevanceScore: number;
}

interface ReportSection {
  id: string;
  title: string;
  content: string;
  citations: Citation[];
  collapsed: boolean;
  isGenerating: boolean;
}

interface Report {
  id: string;
  title: string;
  description: string;
  sections: ReportSection[];
  status: "draft" | "published";
}

// Mock report data
const mockReport: Report = {
  id: "1",
  title: "Q4 2024 Market Analysis",
  description: "Comprehensive analysis of market trends and competitor landscape",
  status: "draft",
  sections: [
    {
      id: "s1",
      title: "Executive Summary",
      content:
        "The Q4 2024 market analysis reveals significant trends in the AI and machine learning sector. Key findings indicate a 35% year-over-year growth in enterprise AI adoption, with particular strength in document processing and knowledge management solutions.",
      citations: [
        {
          id: "c1",
          documentId: "doc1",
          documentTitle: "Gartner AI Trends Report 2024",
          snippet:
            "Enterprise AI adoption grew by 35% YoY, with document processing leading at 42% growth.",
          pageNumber: 12,
          relevanceScore: 0.95,
        },
        {
          id: "c2",
          documentId: "doc2",
          documentTitle: "McKinsey Digital Transformation Study",
          snippet:
            "Knowledge management solutions showed the highest ROI among enterprise software categories.",
          pageNumber: 28,
          relevanceScore: 0.88,
        },
      ],
      collapsed: false,
      isGenerating: false,
    },
    {
      id: "s2",
      title: "Market Landscape",
      content:
        "The competitive landscape has evolved significantly, with new entrants disrupting traditional players. Cloud-native solutions now account for 67% of new deployments, up from 45% in 2023.",
      citations: [
        {
          id: "c3",
          documentId: "doc3",
          documentTitle: "IDC Cloud Market Report",
          snippet:
            "Cloud-native solutions reached 67% market share in enterprise deployments.",
          pageNumber: 5,
          relevanceScore: 0.92,
        },
      ],
      collapsed: false,
      isGenerating: false,
    },
    {
      id: "s3",
      title: "Key Recommendations",
      content: "",
      citations: [],
      collapsed: false,
      isGenerating: false,
    },
  ],
};

export default function ReportEditorPage() {
  const params = useParams();
  const router = useRouter();
  const isNew = params.id === "new";

  const [report, setReport] = useState<Report>(
    isNew
      ? {
          id: "new",
          title: "",
          description: "",
          sections: [],
          status: "draft",
        }
      : mockReport
  );
  const [isSaving, setIsSaving] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);

  const handleTitleChange = (title: string) => {
    setReport((prev) => ({ ...prev, title }));
  };

  const handleDescriptionChange = (description: string) => {
    setReport((prev) => ({ ...prev, description }));
  };

  const addSection = () => {
    const newSection: ReportSection = {
      id: `s${Date.now()}`,
      title: "New Section",
      content: "",
      citations: [],
      collapsed: false,
      isGenerating: false,
    };
    setReport((prev) => ({
      ...prev,
      sections: [...prev.sections, newSection],
    }));
    setActiveSection(newSection.id);
  };

  const updateSection = (sectionId: string, updates: Partial<ReportSection>) => {
    setReport((prev) => ({
      ...prev,
      sections: prev.sections.map((s) =>
        s.id === sectionId ? { ...s, ...updates } : s
      ),
    }));
  };

  const deleteSection = (sectionId: string) => {
    setReport((prev) => ({
      ...prev,
      sections: prev.sections.filter((s) => s.id !== sectionId),
    }));
    if (activeSection === sectionId) {
      setActiveSection(null);
    }
  };

  const toggleSectionCollapse = (sectionId: string) => {
    updateSection(sectionId, {
      collapsed: !report.sections.find((s) => s.id === sectionId)?.collapsed,
    });
  };

  const generateSectionContent = async (sectionId: string) => {
    updateSection(sectionId, { isGenerating: true });

    // Simulate AI generation
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const section = report.sections.find((s) => s.id === sectionId);
    if (section) {
      updateSection(sectionId, {
        content:
          "Based on our analysis of the available documents, we recommend the following strategic priorities for Q1 2025:\n\n1. Accelerate cloud-native infrastructure adoption\n2. Invest in AI-powered automation capabilities\n3. Focus on knowledge management to improve operational efficiency\n4. Establish partnerships with emerging technology providers",
        citations: [
          {
            id: `c${Date.now()}`,
            documentId: "doc4",
            documentTitle: "Internal Strategy Document",
            snippet:
              "Strategic priorities should align with market trends and competitive positioning.",
            relevanceScore: 0.85,
          },
        ],
        isGenerating: false,
      });
    }

    toast.success("Content generated successfully");
  };

  const handleSave = async () => {
    setIsSaving(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsSaving(false);
    toast.success("Report saved");
  };

  const handleExport = async (format: "pdf" | "docx") => {
    toast.success(`Exporting as ${format.toUpperCase()}...`);
  };

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b">
        <div className="flex items-center gap-4">
          <Link href="/dashboard/reports">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </Link>
          <div>
            <Input
              value={report.title}
              onChange={(e) => handleTitleChange(e.target.value)}
              placeholder="Report Title"
              className="text-xl font-bold border-none shadow-none px-0 h-auto focus-visible:ring-0"
            />
            <Input
              value={report.description}
              onChange={(e) => handleDescriptionChange(e.target.value)}
              placeholder="Add a description..."
              className="text-sm text-muted-foreground border-none shadow-none px-0 h-auto mt-1 focus-visible:ring-0"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline">
            {report.status === "draft" ? "Draft" : "Published"}
          </Badge>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowPreview(!showPreview)}
                >
                  {showPreview ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {showPreview ? "Hide Preview" : "Show Preview"}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => handleExport("pdf")}>
                <FileText className="h-4 w-4 mr-2" />
                Export as PDF
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleExport("docx")}>
                <FileText className="h-4 w-4 mr-2" />
                Export as Word
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Button variant="outline">
            <Share2 className="h-4 w-4 mr-2" />
            Share
          </Button>

          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex gap-4 pt-4 overflow-hidden">
        {/* Sections List */}
        <div className="w-64 flex-shrink-0">
          <Card className="h-full flex flex-col">
            <CardHeader className="py-3 px-4 border-b">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Sections</CardTitle>
                <Button variant="ghost" size="sm" onClick={addSection}>
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <ScrollArea className="flex-1">
              <div className="p-2 space-y-1">
                {report.sections.map((section, index) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={cn(
                      "w-full flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors text-left",
                      activeSection === section.id
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-muted"
                    )}
                  >
                    <GripVertical className="h-4 w-4 opacity-50 cursor-grab" />
                    <span className="flex-1 truncate">
                      {index + 1}. {section.title || "Untitled"}
                    </span>
                    {section.citations.length > 0 && (
                      <Badge variant="secondary" className="h-5 px-1.5 text-xs">
                        {section.citations.length}
                      </Badge>
                    )}
                  </button>
                ))}
                {report.sections.length === 0 && (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    No sections yet
                  </p>
                )}
              </div>
            </ScrollArea>
          </Card>
        </div>

        {/* Editor */}
        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            {activeSection ? (
              <SectionEditor
                section={report.sections.find((s) => s.id === activeSection)!}
                onUpdate={(updates) => updateSection(activeSection, updates)}
                onDelete={() => deleteSection(activeSection)}
                onGenerate={() => generateSectionContent(activeSection)}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <FileText className="h-12 w-12 text-muted-foreground/50 mb-4" />
                <h3 className="text-lg font-medium">Select a section to edit</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Or create a new section to get started
                </p>
                <Button variant="outline" className="mt-4" onClick={addSection}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add Section
                </Button>
              </div>
            )}
          </ScrollArea>
        </div>

        {/* Preview Panel */}
        {showPreview && (
          <div className="w-96 flex-shrink-0">
            <Card className="h-full">
              <CardHeader className="py-3 px-4 border-b">
                <CardTitle className="text-sm">Preview</CardTitle>
              </CardHeader>
              <ScrollArea className="h-[calc(100%-3rem)]">
                <div className="p-4 prose prose-sm dark:prose-invert max-w-none">
                  <h1>{report.title || "Untitled Report"}</h1>
                  <p className="text-muted-foreground">{report.description}</p>
                  {report.sections.map((section, index) => (
                    <div key={section.id}>
                      <h2>
                        {index + 1}. {section.title}
                      </h2>
                      <p>{section.content || "No content yet..."}</p>
                      {section.citations.length > 0 && (
                        <div className="not-prose my-4">
                          <p className="text-xs text-muted-foreground mb-2">
                            Sources:
                          </p>
                          {section.citations.map((citation, i) => (
                            <div
                              key={citation.id}
                              className="text-xs text-muted-foreground"
                            >
                              [{i + 1}] {citation.documentTitle}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

interface SectionEditorProps {
  section: ReportSection;
  onUpdate: (updates: Partial<ReportSection>) => void;
  onDelete: () => void;
  onGenerate: () => void;
}

function SectionEditor({
  section,
  onUpdate,
  onDelete,
  onGenerate,
}: SectionEditorProps) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <Input
              value={section.title}
              onChange={(e) => onUpdate({ title: e.target.value })}
              placeholder="Section Title"
              className="text-lg font-semibold border-none shadow-none px-0 h-auto focus-visible:ring-0"
            />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon">
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>
                  <Copy className="h-4 w-4 mr-2" />
                  Duplicate
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive"
                  onClick={onDelete}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete Section
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Generate Button */}
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onGenerate}
              disabled={section.isGenerating}
            >
              {section.isGenerating ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4 mr-2" />
              )}
              {section.isGenerating ? "Generating..." : "Generate with AI"}
            </Button>
            <Button variant="outline" size="sm" disabled={section.isGenerating}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Regenerate
            </Button>
          </div>

          {/* Content Editor */}
          <Textarea
            value={section.content}
            onChange={(e) => onUpdate({ content: e.target.value })}
            placeholder="Write your section content here, or use AI to generate it based on your documents..."
            className="min-h-[200px] resize-none"
          />

          {/* Citations */}
          {section.citations.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">Citations</h4>
                <Badge variant="secondary">
                  {section.citations.length} sources
                </Badge>
              </div>
              <div className="space-y-2">
                {section.citations.map((citation) => (
                  <CitationCard key={citation.id} citation={citation} />
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  return (
    <Card className="bg-muted/50">
      <CardContent className="p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              <span className="text-sm font-medium truncate">
                {citation.documentTitle}
              </span>
              {citation.pageNumber && (
                <Badge variant="outline" className="text-xs">
                  p. {citation.pageNumber}
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              &ldquo;{citation.snippet}&rdquo;
            </p>
          </div>
          <div className="flex items-center gap-1">
            <Badge
              variant="secondary"
              className={cn(
                "text-xs",
                citation.relevanceScore >= 0.9
                  ? "bg-green-500/10 text-green-600"
                  : citation.relevanceScore >= 0.7
                  ? "bg-yellow-500/10 text-yellow-600"
                  : "bg-gray-500/10 text-gray-600"
              )}
            >
              {Math.round(citation.relevanceScore * 100)}%
            </Badge>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ExternalLink className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
