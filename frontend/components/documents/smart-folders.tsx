"use client";

import { useState, useEffect } from "react";
import {
  Folder,
  FolderPlus,
  Sparkles,
  Tag,
  FileText,
  AlertCircle,
  RefreshCw,
  ChevronRight,
  MoreVertical,
  Edit,
  Trash2,
  Eye,
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
import { Badge } from "@/components/ui/badge";
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
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface SmartFolder {
  id: string;
  name: string;
  description: string;
  rules: CategoryRule[];
  documentCount: number;
  color: string;
  icon: string;
  isAuto: boolean;
  lastUpdated: Date;
}

interface CategoryRule {
  type: "keyword" | "topic" | "entity" | "date_range" | "file_type";
  value: string;
  operator: "contains" | "equals" | "starts_with" | "regex";
}

interface Document {
  id: string;
  title: string;
  category: string;
  confidence: number;
  tags: string[];
}

// Mock smart folders
const mockFolders: SmartFolder[] = [
  {
    id: "1",
    name: "Financial Reports",
    description: "Quarterly reports, budgets, and financial statements",
    rules: [
      { type: "topic", value: "finance", operator: "contains" },
      { type: "keyword", value: "report", operator: "contains" },
    ],
    documentCount: 45,
    color: "#10B981",
    icon: "chart",
    isAuto: true,
    lastUpdated: new Date(),
  },
  {
    id: "2",
    name: "Technical Documentation",
    description: "API docs, architecture, and technical specs",
    rules: [
      { type: "topic", value: "technology", operator: "contains" },
      { type: "file_type", value: "md", operator: "equals" },
    ],
    documentCount: 128,
    color: "#6366F1",
    icon: "code",
    isAuto: true,
    lastUpdated: new Date(),
  },
  {
    id: "3",
    name: "Legal Documents",
    description: "Contracts, agreements, and legal correspondence",
    rules: [
      { type: "entity", value: "contract", operator: "contains" },
      { type: "keyword", value: "legal", operator: "contains" },
    ],
    documentCount: 32,
    color: "#F59E0B",
    icon: "gavel",
    isAuto: true,
    lastUpdated: new Date(),
  },
  {
    id: "4",
    name: "Meeting Notes",
    description: "Notes from team meetings and discussions",
    rules: [
      { type: "keyword", value: "meeting", operator: "contains" },
      { type: "keyword", value: "notes", operator: "contains" },
    ],
    documentCount: 89,
    color: "#EC4899",
    icon: "users",
    isAuto: true,
    lastUpdated: new Date(),
  },
];

export function SmartFolders() {
  const [folders, setFolders] = useState<SmartFolder[]>(mockFolders);
  const [isCreating, setIsCreating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [selectedFolder, setSelectedFolder] = useState<SmartFolder | null>(null);

  const handleAutoOrganize = async () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);

    // Simulate analysis progress
    for (let i = 0; i <= 100; i += 10) {
      await new Promise((resolve) => setTimeout(resolve, 200));
      setAnalysisProgress(i);
    }

    setIsAnalyzing(false);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Smart Folders
          </h3>
          <p className="text-sm text-muted-foreground">
            AI-powered document organization
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleAutoOrganize}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4 mr-2" />
            )}
            Auto-Organize
          </Button>
          <Dialog open={isCreating} onOpenChange={setIsCreating}>
            <DialogTrigger asChild>
              <Button size="sm">
                <FolderPlus className="h-4 w-4 mr-2" />
                New Folder
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Smart Folder</DialogTitle>
                <DialogDescription>
                  Define rules for automatic document categorization.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Folder Name</label>
                  <Input placeholder="e.g., Project Documents" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Description</label>
                  <Input placeholder="What documents should go here?" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Auto-categorize by</label>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline" className="cursor-pointer hover:bg-primary hover:text-primary-foreground">
                      Keywords
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer hover:bg-primary hover:text-primary-foreground">
                      Topics
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer hover:bg-primary hover:text-primary-foreground">
                      Entities
                    </Badge>
                    <Badge variant="outline" className="cursor-pointer hover:bg-primary hover:text-primary-foreground">
                      File Type
                    </Badge>
                  </div>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsCreating(false)}>
                  Cancel
                </Button>
                <Button onClick={() => setIsCreating(false)}>Create</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Analysis Progress */}
      {isAnalyzing && (
        <Card>
          <CardContent className="pt-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Analyzing documents...</span>
                <span className="text-muted-foreground">{analysisProgress}%</span>
              </div>
              <Progress value={analysisProgress} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Folders Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {folders.map((folder) => (
          <Card
            key={folder.id}
            className="group cursor-pointer hover:shadow-md transition-shadow"
            onClick={() => setSelectedFolder(folder)}
          >
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className="p-2 rounded-lg"
                    style={{ backgroundColor: `${folder.color}20` }}
                  >
                    <Folder
                      className="h-5 w-5"
                      style={{ color: folder.color }}
                    />
                  </div>
                  <div>
                    <CardTitle className="text-base">{folder.name}</CardTitle>
                    <CardDescription className="text-xs line-clamp-1">
                      {folder.description}
                    </CardDescription>
                  </div>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 opacity-0 group-hover:opacity-100"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem>
                      <Eye className="h-4 w-4 mr-2" />
                      View Documents
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                      <Edit className="h-4 w-4 mr-2" />
                      Edit Rules
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem className="text-destructive">
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">
                    {folder.documentCount} documents
                  </span>
                </div>
                {folder.isAuto && (
                  <Badge variant="secondary" className="text-xs">
                    <Sparkles className="h-3 w-3 mr-1" />
                    Auto
                  </Badge>
                )}
              </div>
              <div className="flex flex-wrap gap-1 mt-2">
                {folder.rules.slice(0, 2).map((rule, i) => (
                  <Badge key={i} variant="outline" className="text-xs">
                    {rule.type}: {rule.value}
                  </Badge>
                ))}
                {folder.rules.length > 2 && (
                  <Badge variant="outline" className="text-xs">
                    +{folder.rules.length - 2} more
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function DocumentInsights() {
  const [duplicates, setDuplicates] = useState<Document[][]>([
    [
      { id: "1", title: "Q4 Report Final.pdf", category: "Reports", confidence: 0.95, tags: ["finance"] },
      { id: "2", title: "Q4 Report Final (1).pdf", category: "Reports", confidence: 0.95, tags: ["finance"] },
    ],
    [
      { id: "3", title: "Contract Draft.docx", category: "Legal", confidence: 0.88, tags: ["contracts"] },
      { id: "4", title: "Contract Draft v2.docx", category: "Legal", confidence: 0.88, tags: ["contracts"] },
    ],
  ]);

  const [suggestions, setSuggestions] = useState([
    { documentId: "5", title: "Meeting Notes Jan.md", suggestedFolder: "Meeting Notes", confidence: 0.92 },
    { documentId: "6", title: "Budget 2024.xlsx", suggestedFolder: "Financial Reports", confidence: 0.89 },
    { documentId: "7", title: "API Design Doc.md", suggestedFolder: "Technical Documentation", confidence: 0.95 },
  ]);

  return (
    <div className="space-y-6">
      {/* Duplicate Detection */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-yellow-500" />
                Duplicate Detection
              </CardTitle>
              <CardDescription>
                {duplicates.length} potential duplicate groups found
              </CardDescription>
            </div>
            <Button variant="outline" size="sm">
              Review All
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[200px]">
            <div className="space-y-3">
              {duplicates.map((group, groupIndex) => (
                <div key={groupIndex} className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant="secondary" className="text-xs">
                      {group.length} similar files
                    </Badge>
                    <Button variant="ghost" size="sm" className="h-7 text-xs">
                      Merge
                    </Button>
                  </div>
                  {group.map((doc) => (
                    <div key={doc.id} className="flex items-center gap-2 text-sm py-1">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="truncate">{doc.title}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Category Suggestions */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base flex items-center gap-2">
                <Tag className="h-4 w-4 text-primary" />
                Suggested Categories
              </CardTitle>
              <CardDescription>
                AI-suggested folder assignments
              </CardDescription>
            </div>
            <Button variant="outline" size="sm">
              Apply All
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {suggestions.map((suggestion) => (
              <div
                key={suggestion.documentId}
                className="flex items-center justify-between p-2 border rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">{suggestion.title}</p>
                    <p className="text-xs text-muted-foreground">
                      Move to: <span className="text-primary">{suggestion.suggestedFolder}</span>
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    {Math.round(suggestion.confidence * 100)}%
                  </Badge>
                  <Button variant="ghost" size="sm" className="h-7">
                    Apply
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
