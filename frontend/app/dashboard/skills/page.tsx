"use client";

import { useState, useCallback, useMemo, useEffect } from "react";
import { useSession } from "next-auth/react";
import {
  Search,
  Filter,
  Grid,
  List,
  Play,
  Settings,
  Info,
  ExternalLink,
  Download,
  Star,
  Clock,
  Zap,
  CheckCircle,
  FileText,
  Globe,
  Heart,
  User,
  GitCompare,
  Sparkles,
  Code,
  MoreVertical,
  Upload,
  Loader2,
  Copy,
  Check,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { skillRegistry, builtInSkills } from "@/lib/skills/registry";
import { Skill, SkillCategory, SkillInput } from "@/lib/skills/types";
import { api, useLLMProviders } from "@/lib/api";

const categoryIcons: Record<SkillCategory, React.ReactNode> = {
  analysis: <FileText className="h-4 w-4" />,
  generation: <Sparkles className="h-4 w-4" />,
  extraction: <User className="h-4 w-4" />,
  transformation: <Globe className="h-4 w-4" />,
  validation: <CheckCircle className="h-4 w-4" />,
  integration: <Code className="h-4 w-4" />,
};

const skillIcons: Record<string, React.ReactNode> = {
  "file-text": <FileText className="h-5 w-5" />,
  "check-circle": <CheckCircle className="h-5 w-5" />,
  globe: <Globe className="h-5 w-5" />,
  heart: <Heart className="h-5 w-5" />,
  user: <User className="h-5 w-5" />,
  "git-compare": <GitCompare className="h-5 w-5" />,
};

export default function SkillsPage() {
  const { status } = useSession();
  const isAuthenticated = status === "authenticated";

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<SkillCategory | "all">("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [selectedSkill, setSelectedSkill] = useState<Skill | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showPublishDialog, setShowPublishDialog] = useState(false);
  const [skillToPublish, setSkillToPublish] = useState<Skill | null>(null);

  // Skills loaded from backend API
  const [skills, setSkills] = useState<Skill[]>([]);
  const [isLoadingSkills, setIsLoadingSkills] = useState(true);
  const [skillsError, setSkillsError] = useState<string | null>(null);

  // Fetch skills from backend API
  const fetchSkills = useCallback(async () => {
    if (!isAuthenticated) return;

    setIsLoadingSkills(true);
    setSkillsError(null);

    try {
      const response = await api.get<{ skills: any[] }>("/skills/list");
      const backendSkills = response.data.skills || [];

      // Transform backend skills to frontend Skill format
      const transformedSkills: Skill[] = backendSkills.map((s: any) => ({
        id: s.skill_key || s.id,
        name: s.name,
        description: s.description || "",
        category: s.category as SkillCategory,
        icon: s.icon || "zap",
        version: s.version || "1.0.0",
        author: s.is_builtin ? "System" : "Custom",
        tags: s.tags || [],
        inputs: (s.inputs || []).map((i: any) => ({
          name: i.name,
          type: i.type || "text",
          description: i.description || "",
          required: i.required !== false,
          default: i.default,
        })),
        outputs: (s.outputs || []).map((o: any) => ({
          name: o.name,
          type: o.type || "text",
          description: o.description || "",
        })),
        config: s.config || {},
        // Required Skill interface fields
        status: s.is_active !== false ? "active" : "disabled",
        usageCount: s.use_count || 0,
        averageLatency: s.avg_execution_time_ms || undefined,
        requiresAuth: false,
        apiKeyRequired: false,
        // Extra metadata
        stats: {
          uses: s.use_count || 0,
          avgLatency: s.avg_execution_time_ms || 0,
          successRate: 0.95,
        },
        isBuiltin: s.is_builtin,
        isPublic: s.is_public,
        isOwner: s.is_owner,
      }));

      setSkills(transformedSkills);
    } catch (error: any) {
      console.error("Failed to fetch skills:", error);
      // Fallback to local registry if API fails
      const localSkills = skillRegistry.list();
      setSkills(localSkills);
      if (error.response?.status !== 404) {
        setSkillsError("Using local skills - backend unavailable");
      }
    } finally {
      setIsLoadingSkills(false);
    }
  }, [isAuthenticated]);

  // Load skills on mount and when auth changes
  useEffect(() => {
    fetchSkills();
  }, [fetchSkills]);

  // Fetch configured LLM providers
  const { data: providersData } = useLLMProviders({ enabled: isAuthenticated });

  // Get active providers from Settings
  const availableProviders = useMemo((): ConfiguredProvider[] => {
    if (!providersData?.providers?.length) {
      return [];
    }

    // Return only active providers
    return providersData.providers.filter((p: any) => p.is_active) as ConfiguredProvider[];
  }, [providersData]);

  // Get default provider for initial selection
  const defaultProvider = useMemo(() => {
    return availableProviders.find(p => p.is_default) || availableProviders[0];
  }, [availableProviders]);

  const filteredSkills = skills.filter((skill) => {
    const matchesSearch =
      skill.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      skill.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      skill.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesCategory = selectedCategory === "all" || skill.category === selectedCategory;

    return matchesSearch && matchesCategory;
  });

  const categories: SkillCategory[] = ["analysis", "generation", "extraction", "transformation", "validation", "integration"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Zap className="h-8 w-8 text-primary" />
            Skills Marketplace
          </h1>
          <p className="text-muted-foreground mt-1">
            Extend your document processing with AI-powered skills
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => setShowImportDialog(true)}>
            <Download className="h-4 w-4 mr-2" />
            Import Skill
          </Button>
          <Button onClick={() => setShowCreateDialog(true)}>
            <Code className="h-4 w-4 mr-2" />
            Create Skill
          </Button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search skills..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === "grid" ? "secondary" : "ghost"}
            size="icon"
            onClick={() => setViewMode("grid")}
          >
            <Grid className="h-4 w-4" />
          </Button>
          <Button
            variant={viewMode === "list" ? "secondary" : "ghost"}
            size="icon"
            onClick={() => setViewMode("list")}
          >
            <List className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Category Tabs */}
      <Tabs value={selectedCategory} onValueChange={(v) => setSelectedCategory(v as SkillCategory | "all")}>
        <TabsList className="flex-wrap h-auto gap-1 p-1">
          <TabsTrigger value="all">All Skills</TabsTrigger>
          {categories.map((category) => (
            <TabsTrigger key={category} value={category} className="capitalize">
              {categoryIcons[category]}
              <span className="ml-1">{category}</span>
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value={selectedCategory} className="mt-6">
          {filteredSkills.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Zap className="h-12 w-12 text-muted-foreground/50 mb-4" />
                <h3 className="text-lg font-medium">No skills found</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Try adjusting your search or category filter
                </p>
              </CardContent>
            </Card>
          ) : viewMode === "grid" ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredSkills.map((skill) => (
                <SkillCard
                  key={skill.id}
                  skill={skill}
                  onSelect={() => setSelectedSkill(skill)}
                  onPublish={() => {
                    setSkillToPublish(skill);
                    setShowPublishDialog(true);
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredSkills.map((skill) => (
                <SkillListItem
                  key={skill.id}
                  skill={skill}
                  onSelect={() => setSelectedSkill(skill)}
                />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Skill Detail Dialog */}
      <SkillDetailDialog
        skill={selectedSkill}
        open={!!selectedSkill}
        onClose={() => setSelectedSkill(null)}
        availableProviders={availableProviders}
        defaultProvider={defaultProvider}
      />

      {/* Create Skill Dialog */}
      <CreateSkillDialog
        open={showCreateDialog}
        onClose={() => setShowCreateDialog(false)}
        onSuccess={fetchSkills}
      />

      {/* Import Skill Dialog */}
      <ImportSkillDialog
        open={showImportDialog}
        onClose={() => setShowImportDialog(false)}
        onSuccess={fetchSkills}
      />

      {/* Publish Skill Dialog */}
      <PublishSkillDialog
        skill={skillToPublish}
        open={showPublishDialog}
        onClose={() => {
          setShowPublishDialog(false);
          setSkillToPublish(null);
        }}
        onSuccess={fetchSkills}
      />
    </div>
  );
}

function SkillCard({ skill, onSelect, onPublish }: { skill: Skill; onSelect: () => void; onPublish?: () => void }) {
  return (
    <Card className="group hover:shadow-md transition-shadow cursor-pointer" onClick={onSelect}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              {skillIcons[skill.icon] || <Zap className="h-5 w-5" />}
            </div>
            <div>
              <CardTitle className="text-base">{skill.name}</CardTitle>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="secondary" className="text-xs capitalize">
                  {skill.category}
                </Badge>
                <span className="text-xs text-muted-foreground">v{skill.version}</span>
              </div>
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
                <Play className="h-4 w-4 mr-2" />
                Run Skill
              </DropdownMenuItem>
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onPublish?.(); }}>
                <ExternalLink className="h-4 w-4 mr-2" />
                Publish
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="h-4 w-4 mr-2" />
                Configure
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Info className="h-4 w-4 mr-2" />
                View Details
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground line-clamp-2">{skill.description}</p>
        <div className="flex flex-wrap gap-1 mt-3">
          {skill.tags.slice(0, 3).map((tag) => (
            <Badge key={tag} variant="outline" className="text-xs">
              {tag}
            </Badge>
          ))}
        </div>
      </CardContent>
      <CardFooter className="pt-0">
        <div className="flex items-center justify-between w-full text-xs text-muted-foreground">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <Play className="h-3 w-3" />
              {skill.usageCount.toLocaleString()}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {skill.averageLatency ? `${(skill.averageLatency / 1000).toFixed(1)}s` : "N/A"}
            </span>
          </div>
          <Badge
            variant="outline"
            className={cn(
              "text-xs",
              skill.status === "active"
                ? "bg-green-500/10 text-green-600 border-green-500/20"
                : "bg-gray-500/10 text-gray-600"
            )}
          >
            {skill.status}
          </Badge>
        </div>
      </CardFooter>
    </Card>
  );
}

function SkillListItem({ skill, onSelect }: { skill: Skill; onSelect: () => void }) {
  return (
    <Card className="cursor-pointer hover:bg-muted/50 transition-colors" onClick={onSelect}>
      <CardContent className="flex items-center gap-4 p-4">
        <div className="p-2 rounded-lg bg-primary/10 text-primary">
          {skillIcons[skill.icon] || <Zap className="h-5 w-5" />}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-medium truncate">{skill.name}</h3>
            <Badge variant="secondary" className="text-xs capitalize">
              {skill.category}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground truncate">{skill.description}</p>
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Play className="h-3 w-3" />
            {skill.usageCount.toLocaleString()}
          </span>
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {skill.averageLatency ? `${(skill.averageLatency / 1000).toFixed(1)}s` : "N/A"}
          </span>
          <Button size="sm" variant="outline" onClick={(e) => e.stopPropagation()}>
            <Play className="h-3 w-3 mr-1" />
            Run
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// Provider type for configured LLM providers
interface ConfiguredProvider {
  id: string;
  name: string;
  provider_type: string;
  default_chat_model?: string;
  is_active: boolean;
  is_default: boolean;
}

function SkillDetailDialog({
  skill,
  open,
  onClose,
  availableProviders,
  defaultProvider,
}: {
  skill: Skill | null;
  open: boolean;
  onClose: () => void;
  availableProviders: ConfiguredProvider[];
  defaultProvider?: ConfiguredProvider;
}) {
  const [activeTab, setActiveTab] = useState<"info" | "run">("run");
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<{ output: any; error?: string; executionTime?: number; model?: string } | null>(null);
  const [copied, setCopied] = useState(false);
  // Default to default provider from settings
  const [selectedProviderId, setSelectedProviderId] = useState(defaultProvider?.id || "");

  // Reset state when skill changes
  const resetState = useCallback(() => {
    setInputValues({});
    setResult(null);
    setActiveTab("run");
    setSelectedProviderId(defaultProvider?.id || "");
  }, [defaultProvider]);

  // Initialize default values when skill changes
  useState(() => {
    if (skill) {
      const defaults: Record<string, any> = {};
      skill.inputs.forEach((input) => {
        if (input.default !== undefined) {
          defaults[input.name] = input.default;
        }
      });
      setInputValues(defaults);
    }
  });

  if (!skill) return null;

  const handleInputChange = (name: string, value: any) => {
    setInputValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleFileUpload = async (name: string, file: File) => {
    // Read file content
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setInputValues((prev) => ({
        ...prev,
        [name]: {
          filename: file.name,
          content: content,
          type: file.type,
        },
      }));
    };
    reader.readAsText(file);
  };

  const handleRunSkill = async () => {
    // Validate required inputs
    const missingRequired = skill.inputs.filter(
      (input) => input.required && !inputValues[input.name]
    );
    if (missingRequired.length > 0) {
      setResult({
        output: null,
        error: `Missing required inputs: ${missingRequired.map((i) => i.name).join(", ")}`,
      });
      return;
    }

    setIsRunning(true);
    setResult(null);

    // Get selected provider details
    const selectedProvider = availableProviders.find(p => p.id === selectedProviderId);

    try {
      // Call the skills execution API
      const response = await api.post<{ output: any; execution_time_ms: number; model_used: string }>("/skills/execute", {
        skill_id: skill.id,
        inputs: inputValues,
        provider_id: selectedProviderId,
        model: selectedProvider?.default_chat_model,
      });

      setResult({
        output: response.data.output,
        executionTime: response.data.execution_time_ms,
        model: response.data.model_used,
      });
    } catch (error: any) {
      setResult({
        output: null,
        error: error.response?.data?.detail || error.message || "Skill execution failed",
      });
    } finally {
      setIsRunning(false);
    }
  };

  const copyToClipboard = async () => {
    if (result?.output) {
      const text = typeof result.output === "string"
        ? result.output
        : JSON.stringify(result.output, null, 2);
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(open) => { if (!open) { resetState(); onClose(); } }}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              {skillIcons[skill.icon] || <Zap className="h-5 w-5" />}
            </div>
            <div>
              <DialogTitle>{skill.name}</DialogTitle>
              <DialogDescription asChild>
                <div className="flex items-center gap-2 mt-1 text-sm text-muted-foreground">
                  <Badge variant="secondary" className="capitalize">
                    {skill.category}
                  </Badge>
                  <span>v{skill.version}</span>
                  <span>by {skill.author}</span>
                </div>
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "info" | "run")} className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="run">
              <Play className="h-4 w-4 mr-2" />
              Run Skill
            </TabsTrigger>
            <TabsTrigger value="info">
              <Info className="h-4 w-4 mr-2" />
              Details
            </TabsTrigger>
          </TabsList>

          <TabsContent value="run" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
                {/* LLM Provider Selection */}
                <div className="space-y-2">
              <Label className="text-sm font-medium">AI Provider</Label>
              {availableProviders.length > 0 ? (
                <>
                  <Select value={selectedProviderId} onValueChange={setSelectedProviderId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a provider" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableProviders.map((provider) => (
                        <SelectItem key={provider.id} value={provider.id}>
                          <div className="flex items-center gap-2">
                            <span>{provider.name}</span>
                            <span className="text-xs text-muted-foreground">
                              ({provider.provider_type} • {provider.default_chat_model})
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Select the AI provider configured in Settings
                  </p>
                </>
              ) : (
                <div className="p-3 bg-muted/50 rounded-lg text-sm text-muted-foreground">
                  No LLM providers configured.{" "}
                  <a href="/dashboard/admin/settings" className="text-primary underline">
                    Add providers in Settings
                  </a>
                </div>
              )}
            </div>

            <Separator />

            {/* Input Fields */}
            <div className="space-y-4">
              <h4 className="text-sm font-medium">Inputs</h4>
              {skill.inputs.map((input) => (
                <SkillInputField
                  key={input.name}
                  input={input}
                  value={inputValues[input.name]}
                  onChange={(value) => handleInputChange(input.name, value)}
                  onFileUpload={(file) => handleFileUpload(input.name, file)}
                />
              ))}
            </div>

            <Separator />

            {/* Run Button */}
            <Button
              onClick={handleRunSkill}
              disabled={isRunning}
              className="w-full"
              size="lg"
            >
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Skill
                </>
              )}
            </Button>

            {/* Result Display */}
            {result && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium">Output</h4>
                  {result.output && (
                    <Button variant="ghost" size="sm" onClick={copyToClipboard}>
                      {copied ? (
                        <Check className="h-4 w-4 mr-1" />
                      ) : (
                        <Copy className="h-4 w-4 mr-1" />
                      )}
                      {copied ? "Copied!" : "Copy"}
                    </Button>
                  )}
                </div>
                {result.error ? (
                  <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                      <p className="text-sm text-destructive">{result.error}</p>
                    </div>
                  </div>
                ) : (
                  <ScrollArea className="h-64 border rounded-lg">
                    <pre className="p-4 text-sm whitespace-pre-wrap">
                      {typeof result.output === "string"
                        ? result.output
                        : JSON.stringify(result.output, null, 2)}
                    </pre>
                  </ScrollArea>
                )}
              </div>
            )}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="info" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
            <p className="text-sm">{skill.description}</p>

            <Separator />

            {/* Inputs Schema */}
            <div>
              <h4 className="text-sm font-medium mb-2">Input Schema</h4>
              <div className="space-y-2">
                {skill.inputs.map((input) => (
                  <div key={input.name} className="flex items-start gap-2 text-sm">
                    <Badge variant="outline" className="text-xs">
                      {input.type}
                    </Badge>
                    <div>
                      <span className="font-medium">{input.name}</span>
                      {input.required && <span className="text-red-500 ml-1">*</span>}
                      <p className="text-muted-foreground text-xs">{input.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Outputs Schema */}
            <div>
              <h4 className="text-sm font-medium mb-2">Output Schema</h4>
              <div className="space-y-2">
                {skill.outputs.map((output) => (
                  <div key={output.name} className="flex items-start gap-2 text-sm">
                    <Badge variant="outline" className="text-xs">
                      {output.type}
                    </Badge>
                    <div>
                      <span className="font-medium">{output.name}</span>
                      <p className="text-muted-foreground text-xs">{output.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-1">
              {skill.tags.map((tag) => (
                <Badge key={tag} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>

            {/* Stats */}
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Play className="h-4 w-4" />
                {skill.usageCount.toLocaleString()} executions
              </span>
              <span className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                Avg: {skill.averageLatency ? `${(skill.averageLatency / 1000).toFixed(1)}s` : "N/A"}
              </span>
            </div>
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}

// Dynamic input field component based on input type
function SkillInputField({
  input,
  value,
  onChange,
  onFileUpload,
}: {
  input: SkillInput;
  value: any;
  onChange: (value: any) => void;
  onFileUpload: (file: File) => void;
}) {
  switch (input.type) {
    case "text":
      return (
        <div className="space-y-2">
          <Label>
            {input.name}
            {input.required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <Textarea
            placeholder={input.description}
            value={value || ""}
            onChange={(e) => onChange(e.target.value)}
            rows={4}
          />
          <p className="text-xs text-muted-foreground">{input.description}</p>
        </div>
      );

    case "document":
    case "file":
      return (
        <div className="space-y-2">
          <Label>
            {input.name}
            {input.required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <div className="flex flex-col gap-2">
            {/* File upload option */}
            <div className="border-2 border-dashed rounded-lg p-4 text-center hover:bg-muted/50 transition-colors">
              <input
                type="file"
                id={`file-${input.name}`}
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onFileUpload(file);
                }}
                accept={input.type === "document" ? ".txt,.md,.pdf,.docx" : "*"}
              />
              <label
                htmlFor={`file-${input.name}`}
                className="cursor-pointer flex flex-col items-center gap-2"
              >
                <Upload className="h-8 w-8 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">
                  {value?.filename || "Click to upload or drag and drop"}
                </span>
              </label>
            </div>
            {/* Or paste text directly */}
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">or paste text</span>
              </div>
            </div>
            <Textarea
              placeholder="Paste document content here..."
              value={typeof value === "string" ? value : value?.content || ""}
              onChange={(e) => onChange(e.target.value)}
              rows={4}
            />
          </div>
          <p className="text-xs text-muted-foreground">{input.description}</p>
        </div>
      );

    case "json":
      return (
        <div className="space-y-2">
          <Label>
            {input.name}
            {input.required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <Textarea
            placeholder='{"key": "value"}'
            value={typeof value === "string" ? value : JSON.stringify(value || {}, null, 2)}
            onChange={(e) => {
              try {
                onChange(JSON.parse(e.target.value));
              } catch {
                onChange(e.target.value);
              }
            }}
            rows={4}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">{input.description}</p>
        </div>
      );

    case "image":
      return (
        <div className="space-y-2">
          <Label>
            {input.name}
            {input.required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <div className="border-2 border-dashed rounded-lg p-4 text-center hover:bg-muted/50 transition-colors">
            <input
              type="file"
              id={`image-${input.name}`}
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFileUpload(file);
              }}
              accept="image/*"
            />
            <label
              htmlFor={`image-${input.name}`}
              className="cursor-pointer flex flex-col items-center gap-2"
            >
              {value?.content ? (
                <img
                  src={value.content}
                  alt="Preview"
                  className="max-h-32 rounded"
                />
              ) : (
                <Upload className="h-8 w-8 text-muted-foreground" />
              )}
              <span className="text-sm text-muted-foreground">
                {value?.filename || "Click to upload image"}
              </span>
            </label>
          </div>
          <p className="text-xs text-muted-foreground">{input.description}</p>
        </div>
      );

    default:
      return (
        <div className="space-y-2">
          <Label>
            {input.name}
            {input.required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <Input
            placeholder={input.description}
            value={value || ""}
            onChange={(e) => onChange(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">{input.description}</p>
        </div>
      );
  }
}

// Simulate skill execution locally when API is not available
async function simulateSkillExecution(
  skill: Skill,
  inputs: Record<string, any>
): Promise<{ output: any; error?: string }> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const textInput = inputs.document || inputs.text || inputs.content || "";
  const content = typeof textInput === "object" ? textInput.content : textInput;

  switch (skill.id) {
    case "summarizer":
      if (!content) {
        return { output: null, error: "Please provide document content to summarize" };
      }
      const sentences = content.split(/[.!?]+/).filter((s: string) => s.trim().length > 0);
      const summaryLength = inputs.length || "medium";
      const numSentences = summaryLength === "short" ? 2 : summaryLength === "long" ? 6 : 4;
      return {
        output: `**Summary (${summaryLength}):**\n\n${sentences.slice(0, Math.min(numSentences, sentences.length)).join(". ").trim()}.`,
      };

    case "fact-checker":
      if (!content) {
        return { output: null, error: "Please provide text to fact-check" };
      }
      return {
        output: {
          claims_found: 3,
          verified: 2,
          unverified: 1,
          analysis: "This is a simulated fact-check result. In production, this would use AI to verify claims against trusted sources.",
          claims: [
            { text: "Sample claim 1", status: "verified", confidence: 0.95 },
            { text: "Sample claim 2", status: "verified", confidence: 0.87 },
            { text: "Sample claim 3", status: "unverified", confidence: 0.45 },
          ],
        },
      };

    case "translator":
      if (!content) {
        return { output: null, error: "Please provide text to translate" };
      }
      const targetLang = inputs.target_language || "Spanish";
      return {
        output: `**Translated to ${targetLang}:**\n\n[This is a simulated translation. In production, this would use AI translation.]\n\nOriginal text length: ${content.length} characters`,
      };

    case "sentiment-analyzer":
      if (!content) {
        return { output: null, error: "Please provide text to analyze" };
      }
      return {
        output: {
          overall_sentiment: "positive",
          confidence: 0.82,
          breakdown: {
            positive: 0.65,
            neutral: 0.25,
            negative: 0.10,
          },
          key_phrases: ["sample phrase", "another phrase"],
          analysis: "This is a simulated sentiment analysis. In production, this would use AI to analyze the emotional tone of the text.",
        },
      };

    case "entity-extractor":
      if (!content) {
        return { output: null, error: "Please provide text to extract entities from" };
      }
      return {
        output: {
          entities: [
            { type: "PERSON", text: "Sample Person", count: 1 },
            { type: "ORGANIZATION", text: "Sample Org", count: 2 },
            { type: "LOCATION", text: "Sample Place", count: 1 },
            { type: "DATE", text: "2024", count: 3 },
          ],
          total_entities: 7,
          analysis: "This is a simulated entity extraction. In production, this would use NLP to identify named entities.",
        },
      };

    default:
      return {
        output: `Skill "${skill.name}" executed successfully with inputs:\n${JSON.stringify(inputs, null, 2)}\n\n(This is a simulated result - the actual skill backend is not yet implemented)`,
      };
  }
}

// Create Skill Dialog
function CreateSkillDialog({
  open,
  onClose,
  onSuccess,
}: {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState<SkillCategory>("analysis");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [inputs, setInputs] = useState<Array<{ name: string; type: string; description: string; required: boolean }>>([
    { name: "content", type: "text", description: "Input text to process", required: true },
  ]);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAddInput = () => {
    setInputs([...inputs, { name: "", type: "text", description: "", required: false }]);
  };

  const handleRemoveInput = (index: number) => {
    setInputs(inputs.filter((_, i) => i !== index));
  };

  const handleInputChange = (index: number, field: string, value: any) => {
    const newInputs = [...inputs];
    newInputs[index] = { ...newInputs[index], [field]: value };
    setInputs(newInputs);
  };

  const handleCreate = async () => {
    if (!name.trim()) {
      setError("Skill name is required");
      return;
    }
    if (!systemPrompt.trim()) {
      setError("System prompt is required");
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      // Create skill via API
      await api.post("/skills", {
        name,
        description,
        category,
        system_prompt: systemPrompt,
        inputs: inputs.filter((i) => i.name.trim()),
        outputs: [{ name: "result", type: "text", description: "Skill output" }],
      });
      onSuccess?.();
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to create skill");
    } finally {
      setIsCreating(false);
    }
  };

  const resetForm = () => {
    setName("");
    setDescription("");
    setCategory("analysis");
    setSystemPrompt("");
    setInputs([{ name: "text", type: "text", description: "Input text to process", required: true }]);
    setError(null);
  };

  return (
    <Dialog open={open} onOpenChange={(open) => { if (!open) { resetForm(); onClose(); } }}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            Create Custom Skill
          </DialogTitle>
          <DialogDescription>
            Define a new AI skill with custom inputs and processing logic
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-4 py-4">
            {error && (
              <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            {/* Basic Info */}
            <div className="space-y-2">
              <Label htmlFor="skill-name">Skill Name *</Label>
              <Input
                id="skill-name"
                placeholder="e.g., Contract Analyzer"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="skill-description">Description</Label>
              <Textarea
                id="skill-description"
                placeholder="Describe what this skill does..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
              />
            </div>

            <div className="space-y-2">
              <Label>Category</Label>
              <Select value={category} onValueChange={(v) => setCategory(v as SkillCategory)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="analysis">Analysis</SelectItem>
                  <SelectItem value="generation">Generation</SelectItem>
                  <SelectItem value="extraction">Extraction</SelectItem>
                  <SelectItem value="transformation">Transformation</SelectItem>
                  <SelectItem value="validation">Validation</SelectItem>
                  <SelectItem value="integration">Integration</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Separator />

            {/* System Prompt */}
            <div className="space-y-2">
              <Label htmlFor="system-prompt">System Prompt *</Label>
              <Textarea
                id="system-prompt"
                placeholder="You are an AI assistant that specializes in..."
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                rows={4}
                className="font-mono text-sm"
              />
              <p className="text-xs text-muted-foreground">
                Define the AI's behavior and capabilities. Use {"{{input_name}}"} to reference inputs.
              </p>
            </div>

            <Separator />

            {/* Inputs */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Inputs</Label>
                <Button variant="outline" size="sm" onClick={handleAddInput}>
                  <Zap className="h-3 w-3 mr-1" />
                  Add Input
                </Button>
              </div>

              {inputs.map((input, index) => (
                <div key={index} className="grid grid-cols-12 gap-2 items-start p-3 border rounded-lg">
                  <div className="col-span-3">
                    <Input
                      placeholder="name"
                      value={input.name}
                      onChange={(e) => handleInputChange(index, "name", e.target.value)}
                      className="text-sm"
                    />
                  </div>
                  <div className="col-span-2">
                    <Select
                      value={input.type}
                      onValueChange={(v) => handleInputChange(index, "type", v)}
                    >
                      <SelectTrigger className="text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="text">text</SelectItem>
                        <SelectItem value="document">document</SelectItem>
                        <SelectItem value="file">file</SelectItem>
                        <SelectItem value="json">json</SelectItem>
                        <SelectItem value="image">image</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="col-span-5">
                    <Input
                      placeholder="description"
                      value={input.description}
                      onChange={(e) => handleInputChange(index, "description", e.target.value)}
                      className="text-sm"
                    />
                  </div>
                  <div className="col-span-1 flex items-center justify-center">
                    <label className="flex items-center gap-1 text-xs">
                      <input
                        type="checkbox"
                        checked={input.required}
                        onChange={(e) => handleInputChange(index, "required", e.target.checked)}
                        className="h-3 w-3"
                      />
                      Req
                    </label>
                  </div>
                  <div className="col-span-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleRemoveInput(index)}
                      disabled={inputs.length === 1}
                      className="h-8 w-8"
                    >
                      <AlertCircle className="h-4 w-4 text-muted-foreground" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </ScrollArea>

        <DialogFooter>
          <Button variant="outline" onClick={() => { resetForm(); onClose(); }}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={isCreating}>
            {isCreating ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <CheckCircle className="h-4 w-4 mr-2" />
                Create Skill
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Import Skill Dialog
function ImportSkillDialog({
  open,
  onClose,
  onSuccess,
}: {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}) {
  const [importMethod, setImportMethod] = useState<"json" | "url">("json");
  const [jsonContent, setJsonContent] = useState("");
  const [url, setUrl] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setJsonContent(event.target?.result as string);
      };
      reader.readAsText(file);
    }
  };

  const handleImport = async () => {
    setIsImporting(true);
    setError(null);

    try {
      let skillData;

      if (importMethod === "json") {
        if (!jsonContent.trim()) {
          throw new Error("Please provide JSON content");
        }
        try {
          skillData = JSON.parse(jsonContent);
        } catch {
          throw new Error("Invalid JSON format");
        }
      } else {
        if (!url.trim()) {
          throw new Error("Please provide a URL");
        }
        // Fetch from URL
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error("Failed to fetch skill from URL");
        }
        skillData = await response.json();
      }

      // Validate skill structure
      if (!skillData.name) {
        throw new Error("Invalid skill format: missing name field");
      }

      // Import via API (backend expects { skill_data: ... })
      await api.post("/skills/import", { skill_data: skillData });
      onSuccess?.();

      onClose();
    } catch (err: any) {
      setError(err.message || "Import failed");
    } finally {
      setIsImporting(false);
    }
  };

  const resetForm = () => {
    setImportMethod("json");
    setJsonContent("");
    setUrl("");
    setError(null);
  };

  const sampleSkillJson = `{
  "name": "Custom Analyzer",
  "description": "Analyzes text for specific patterns",
  "category": "analysis",
  "version": "1.0.0",
  "author": "Your Name",
  "inputs": [
    {
      "name": "text",
      "type": "text",
      "description": "Text to analyze",
      "required": true
    }
  ],
  "outputs": [
    {
      "name": "analysis",
      "type": "json",
      "description": "Analysis results"
    }
  ],
  "system_prompt": "You are an expert analyzer..."
}`;

  return (
    <Dialog open={open} onOpenChange={(open) => { if (!open) { resetForm(); onClose(); } }}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Import Skill
          </DialogTitle>
          <DialogDescription>
            Import a skill from JSON file or URL
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {error && (
            <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          {/* Import Method */}
          <Tabs value={importMethod} onValueChange={(v) => setImportMethod(v as "json" | "url")}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="json">JSON / File</TabsTrigger>
              <TabsTrigger value="url">From URL</TabsTrigger>
            </TabsList>

            <TabsContent value="json" className="space-y-4 mt-4">
              {/* File upload */}
              <div className="border-2 border-dashed rounded-lg p-4 text-center hover:bg-muted/50 transition-colors">
                <input
                  type="file"
                  id="skill-file"
                  className="hidden"
                  accept=".json"
                  onChange={handleFileUpload}
                />
                <label htmlFor="skill-file" className="cursor-pointer flex flex-col items-center gap-2">
                  <Upload className="h-8 w-8 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">
                    Click to upload JSON file
                  </span>
                </label>
              </div>

              {/* Or paste JSON */}
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">or paste JSON</span>
                </div>
              </div>

              <Textarea
                placeholder={sampleSkillJson}
                value={jsonContent}
                onChange={(e) => setJsonContent(e.target.value)}
                rows={12}
                className="font-mono text-xs"
              />
            </TabsContent>

            <TabsContent value="url" className="space-y-4 mt-4">
              <div className="space-y-2">
                <Label htmlFor="skill-url">Skill URL</Label>
                <Input
                  id="skill-url"
                  placeholder="https://example.com/skills/my-skill.json"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Enter the URL of a JSON file containing the skill definition
                </p>
              </div>
            </TabsContent>
          </Tabs>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => { resetForm(); onClose(); }}>
            Cancel
          </Button>
          <Button onClick={handleImport} disabled={isImporting}>
            {isImporting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Importing...
              </>
            ) : (
              <>
                <Download className="h-4 w-4 mr-2" />
                Import Skill
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Publish Skill Dialog
function PublishSkillDialog({
  skill,
  open,
  onClose,
  onSuccess,
}: {
  skill: Skill | null;
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}) {
  const [customSlug, setCustomSlug] = useState("");
  const [rateLimit, setRateLimit] = useState(100);
  const [isPublishing, setIsPublishing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [publishResult, setPublishResult] = useState<{
    public_url: string;
    embed_code: string;
    public_slug: string;
  } | null>(null);
  const [copied, setCopied] = useState<"url" | "embed" | null>(null);

  if (!skill) return null;

  const handlePublish = async () => {
    setIsPublishing(true);
    setError(null);

    try {
      const response = await api.post<{ public_url: string; embed_code: string; public_slug: string }>(`/skills/${skill.id}/publish`, {
        custom_slug: customSlug || undefined,
        rate_limit: rateLimit,
        allowed_domains: ["*"],
        require_api_key: false,
      });

      setPublishResult({
        public_url: response.data.public_url,
        embed_code: response.data.embed_code,
        public_slug: response.data.public_slug,
      });
      onSuccess?.();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to publish skill");
    } finally {
      setIsPublishing(false);
    }
  };

  const handleUnpublish = async () => {
    setIsPublishing(true);
    setError(null);

    try {
      await api.post(`/skills/${skill.id}/unpublish`);
      setPublishResult(null);
      onSuccess?.();
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to unpublish skill");
    } finally {
      setIsPublishing(false);
    }
  };

  const copyToClipboard = async (type: "url" | "embed") => {
    if (!publishResult) return;
    const text = type === "url" ? publishResult.public_url : publishResult.embed_code;
    await navigator.clipboard.writeText(text);
    setCopied(type);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <Dialog open={open} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ExternalLink className="h-5 w-5" />
            Publish Skill
          </DialogTitle>
          <DialogDescription>
            Make "{skill.name}" accessible via a public URL without authentication
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {error && (
            <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          {publishResult ? (
            // Show publish result
            <div className="space-y-4">
              <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <div className="flex items-center gap-2 text-green-600 mb-2">
                  <CheckCircle className="h-5 w-5" />
                  <span className="font-medium">Skill Published!</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Your skill is now accessible at the public URL below.
                </p>
              </div>

              {/* Public URL */}
              <div className="space-y-2">
                <Label>Public URL</Label>
                <div className="flex gap-2">
                  <Input value={publishResult.public_url} readOnly className="font-mono text-sm" />
                  <Button variant="outline" size="icon" onClick={() => copyToClipboard("url")}>
                    {copied === "url" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>

              {/* Embed Code */}
              <div className="space-y-2">
                <Label>Embed Code</Label>
                <div className="flex gap-2">
                  <Textarea
                    value={publishResult.embed_code}
                    readOnly
                    className="font-mono text-xs"
                    rows={4}
                  />
                </div>
                <Button variant="outline" size="sm" onClick={() => copyToClipboard("embed")}>
                  {copied === "embed" ? <Check className="h-4 w-4 mr-2" /> : <Copy className="h-4 w-4 mr-2" />}
                  Copy Embed Code
                </Button>
              </div>

              <Separator />

              <Button variant="destructive" onClick={handleUnpublish} disabled={isPublishing}>
                {isPublishing ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
                Unpublish Skill
              </Button>
            </div>
          ) : (
            // Show publish form
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="custom-slug">Custom URL Slug (optional)</Label>
                <Input
                  id="custom-slug"
                  placeholder="e.g., my-summarizer"
                  value={customSlug}
                  onChange={(e) => setCustomSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ""))}
                />
                <p className="text-xs text-muted-foreground">
                  Leave empty for auto-generated slug. Use lowercase letters, numbers, and hyphens.
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="rate-limit">Rate Limit (requests/minute)</Label>
                <Input
                  id="rate-limit"
                  type="number"
                  value={rateLimit}
                  onChange={(e) => setRateLimit(parseInt(e.target.value) || 100)}
                  min={1}
                  max={1000}
                />
              </div>

              <div className="p-3 bg-muted rounded-lg text-sm">
                <p className="font-medium mb-1">What happens when you publish:</p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1">
                  <li>A public URL will be generated for this skill</li>
                  <li>Anyone with the URL can use the skill</li>
                  <li>Usage will be rate-limited to prevent abuse</li>
                  <li>You can unpublish at any time</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            {publishResult ? "Close" : "Cancel"}
          </Button>
          {!publishResult && (
            <Button onClick={handlePublish} disabled={isPublishing}>
              {isPublishing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Publishing...
                </>
              ) : (
                <>
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Publish Skill
                </>
              )}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
