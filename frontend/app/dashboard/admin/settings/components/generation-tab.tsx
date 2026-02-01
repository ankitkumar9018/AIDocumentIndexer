"use client";

import {
  PenTool,
  FileText,
  CheckCircle,
  Eye,
  Sparkles,
  BarChart3,
  Zap,
  Brain,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";

interface OllamaModel {
  name: string;
  parameter_size?: string;
}

interface OllamaLocalModels {
  chat_models?: OllamaModel[];
}

interface Provider {
  provider_type: string;
  is_active: boolean;
}

interface ProvidersData {
  providers?: Provider[];
}

interface GenerationTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
  providersData: ProvidersData | undefined;
  ollamaLocalModels: OllamaLocalModels | undefined;
}

export function GenerationTab({
  localSettings,
  handleSettingChange,
  providersData,
  ollamaLocalModels,
}: GenerationTabProps) {
  return (
    <TabsContent value="generation" className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <PenTool className="h-5 w-5" />
            Document Generation Settings
          </CardTitle>
          <CardDescription>
            Configure AI-powered document and presentation generation features
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Output Settings */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Output Settings
            </h4>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Include Sources</p>
                  <p className="text-sm text-muted-foreground">
                    Add source citations to documents
                  </p>
                </div>
                <Switch
                  checked={localSettings["generation.include_sources"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("generation.include_sources", checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Include Images</p>
                  <p className="text-sm text-muted-foreground">
                    Auto-generate relevant images
                  </p>
                </div>
                <Switch
                  checked={localSettings["generation.include_images"] as boolean ?? true}
                  onCheckedChange={(checked) => handleSettingChange("generation.include_images", checked)}
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Image Backend</label>
              <Select
                value={localSettings["generation.image_backend"] as string || "picsum"}
                onValueChange={(value) => handleSettingChange("generation.image_backend", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="picsum">Picsum (Free - No API Key)</SelectItem>
                  <SelectItem value="unsplash">Unsplash (Requires API Key)</SelectItem>
                  <SelectItem value="pexels">Pexels (Requires API Key)</SelectItem>
                  <SelectItem value="openai">OpenAI DALL-E (Requires API Key)</SelectItem>
                  <SelectItem value="stability">Stability AI (Requires API Key)</SelectItem>
                  <SelectItem value="automatic1111">Automatic1111 (Local SD)</SelectItem>
                  <SelectItem value="disabled">Disabled</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">Source for auto-generated images</p>
            </div>
          </div>

          {/* Quality Review Settings */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              Quality Review
            </h4>
            <p className="text-sm text-muted-foreground">
              LLM-based review of generated content for quality assurance
            </p>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Enable Quality Review</p>
                <p className="text-sm text-muted-foreground">
                  Use LLM to review generated content quality
                </p>
              </div>
              <Switch
                checked={localSettings["generation.enable_quality_review"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("generation.enable_quality_review", checked)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Minimum Quality Score</label>
              <Input
                type="number"
                step="0.1"
                min="0"
                max="1"
                placeholder="0.7"
                value={localSettings["generation.min_quality_score"] as number ?? 0.7}
                onChange={(e) => handleSettingChange("generation.min_quality_score", parseFloat(e.target.value))}
              />
              <p className="text-xs text-muted-foreground">Score threshold (0-1) for content to pass review</p>
            </div>
          </div>

          {/* Vision Analysis Settings */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Eye className="h-4 w-4" />
              Vision Analysis (PPTX)
            </h4>
            <p className="text-sm text-muted-foreground">
              Use vision models to analyze templates and review generated slides for layout issues
            </p>

            {/* Template Vision Analysis */}
            <div className="space-y-3 p-3 rounded-lg border bg-muted/30">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Template Vision Analysis</p>
                  <p className="text-sm text-muted-foreground">
                    Analyze template slides to learn visual styling and layout
                  </p>
                </div>
                <Switch
                  checked={localSettings["generation.enable_template_vision_analysis"] as boolean ?? false}
                  onCheckedChange={(checked) => handleSettingChange("generation.enable_template_vision_analysis", checked)}
                />
              </div>
              <div className="space-y-2 pt-2">
                <label className="text-sm font-medium">Template Vision Model</label>
                <Select
                  value={localSettings["generation.template_vision_model"] as string || "auto"}
                  onValueChange={(value) => handleSettingChange("generation.template_vision_model", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Use default vision model)</SelectItem>
                    <SelectGroup>
                      <SelectLabel>OpenAI (Vision-capable)</SelectLabel>
                      {providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                        <>
                          <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                          <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                          <SelectItem value="gpt-4-vision-preview">GPT-4 Vision</SelectItem>
                        </>
                      )}
                      {!providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                        <SelectItem value="_no_openai" disabled>No OpenAI provider configured</SelectItem>
                      )}
                    </SelectGroup>
                    <SelectGroup>
                      <SelectLabel>Anthropic (Vision-capable)</SelectLabel>
                      {providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                        <>
                          <SelectItem value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</SelectItem>
                          <SelectItem value="claude-3-opus-20240229">Claude 3 Opus</SelectItem>
                          <SelectItem value="claude-3-sonnet-20240229">Claude 3 Sonnet</SelectItem>
                          <SelectItem value="claude-3-haiku-20240307">Claude 3 Haiku</SelectItem>
                        </>
                      )}
                      {!providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                        <SelectItem value="_no_anthropic" disabled>No Anthropic provider configured</SelectItem>
                      )}
                    </SelectGroup>
                    <SelectGroup>
                      <SelectLabel>Ollama (Local - Downloaded)</SelectLabel>
                      {ollamaLocalModels?.chat_models?.filter(m =>
                        m.name.includes("llava") || m.name.includes("bakllava") ||
                        m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                        m.name.includes("minicpm-v")
                      ).map(model => (
                        <SelectItem key={model.name} value={model.name}>
                          {model.name} {model.parameter_size && `(${model.parameter_size})`}
                        </SelectItem>
                      ))}
                      {(!ollamaLocalModels?.chat_models || ollamaLocalModels.chat_models.filter(m =>
                        m.name.includes("llava") || m.name.includes("bakllava") ||
                        m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                        m.name.includes("minicpm-v")
                      ).length === 0) && (
                        <SelectItem value="_no_ollama_vision" disabled>No vision models downloaded (try: ollama pull llava)</SelectItem>
                      )}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">Vision model for analyzing template styling</p>
              </div>
            </div>

            {/* Slide Review */}
            <div className="space-y-3 p-3 rounded-lg border bg-muted/30">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Vision-Based Slide Review</p>
                  <p className="text-sm text-muted-foreground">
                    Review generated slides for visual issues (overlaps, truncation)
                  </p>
                </div>
                <Switch
                  checked={localSettings["generation.enable_vision_review"] as boolean ?? false}
                  onCheckedChange={(checked) => handleSettingChange("generation.enable_vision_review", checked)}
                />
              </div>
              <div className="space-y-3 pt-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Vision Review Model</label>
                  <Select
                    value={localSettings["generation.vision_review_model"] as string || "auto"}
                    onValueChange={(value) => handleSettingChange("generation.vision_review_model", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto (Use default vision model)</SelectItem>
                      <SelectGroup>
                        <SelectLabel>OpenAI (Vision-capable)</SelectLabel>
                        {providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                          <>
                            <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                            <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                            <SelectItem value="gpt-4-vision-preview">GPT-4 Vision</SelectItem>
                          </>
                        )}
                        {!providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                          <SelectItem value="_no_openai_review" disabled>No OpenAI provider configured</SelectItem>
                        )}
                      </SelectGroup>
                      <SelectGroup>
                        <SelectLabel>Anthropic (Vision-capable)</SelectLabel>
                        {providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                          <>
                            <SelectItem value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</SelectItem>
                            <SelectItem value="claude-3-opus-20240229">Claude 3 Opus</SelectItem>
                            <SelectItem value="claude-3-sonnet-20240229">Claude 3 Sonnet</SelectItem>
                            <SelectItem value="claude-3-haiku-20240307">Claude 3 Haiku</SelectItem>
                          </>
                        )}
                        {!providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                          <SelectItem value="_no_anthropic_review" disabled>No Anthropic provider configured</SelectItem>
                        )}
                      </SelectGroup>
                      <SelectGroup>
                        <SelectLabel>Ollama (Local - Downloaded)</SelectLabel>
                        {ollamaLocalModels?.chat_models?.filter(m =>
                          m.name.includes("llava") || m.name.includes("bakllava") ||
                          m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                          m.name.includes("minicpm-v")
                        ).map(model => (
                          <SelectItem key={model.name} value={model.name}>
                            {model.name} {model.parameter_size && `(${model.parameter_size})`}
                          </SelectItem>
                        ))}
                        {(!ollamaLocalModels?.chat_models || ollamaLocalModels.chat_models.filter(m =>
                          m.name.includes("llava") || m.name.includes("bakllava") ||
                          m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                          m.name.includes("minicpm-v")
                        ).length === 0) && (
                          <SelectItem value="_no_ollama_vision_review" disabled>No vision models downloaded (try: ollama pull llava)</SelectItem>
                        )}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Review All Slides</p>
                    <p className="text-xs text-muted-foreground">
                      If off, only reviews slides at risk of issues
                    </p>
                  </div>
                  <Switch
                    checked={localSettings["generation.vision_review_all_slides"] as boolean ?? false}
                    onCheckedChange={(checked) => handleSettingChange("generation.vision_review_all_slides", checked)}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Per-Slide Constraints */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Content Fitting
            </h4>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Per-Slide Constraints</p>
                <p className="text-sm text-muted-foreground">
                  Learn layout constraints from template for each slide type
                </p>
              </div>
              <Switch
                checked={localSettings["generation.enable_per_slide_constraints"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("generation.enable_per_slide_constraints", checked)}
              />
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">LLM Content Rewriting</p>
                <p className="text-sm text-muted-foreground">
                  Use LLM to condense overflowing content intelligently
                </p>
              </div>
              <Switch
                checked={localSettings["generation.enable_llm_rewrite"] as boolean ?? true}
                onCheckedChange={(checked) => handleSettingChange("generation.enable_llm_rewrite", checked)}
              />
            </div>
          </div>

          {/* Chart Generation */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Chart Generation
            </h4>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">Auto-Generate Charts</p>
                <p className="text-sm text-muted-foreground">
                  Automatically create charts from data in content
                </p>
              </div>
              <Switch
                checked={localSettings["generation.auto_charts"] as boolean ?? false}
                onCheckedChange={(checked) => handleSettingChange("generation.auto_charts", checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Chart Style</label>
                <Select
                  value={localSettings["generation.chart_style"] as string || "business"}
                  onValueChange={(value) => handleSettingChange("generation.chart_style", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="business">Business (Professional)</SelectItem>
                    <SelectItem value="modern">Modern (Clean lines)</SelectItem>
                    <SelectItem value="minimal">Minimal (Simple)</SelectItem>
                    <SelectItem value="colorful">Colorful (Vibrant)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Chart DPI</label>
                <Input
                  type="number"
                  min="72"
                  max="300"
                  placeholder="150"
                  value={localSettings["generation.chart_dpi"] as number ?? 150}
                  onChange={(e) => handleSettingChange("generation.chart_dpi", parseInt(e.target.value) || 150)}
                />
                <p className="text-xs text-muted-foreground">Resolution for chart images</p>
              </div>
            </div>
          </div>

          {/* Feature LLM Defaults */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Feature LLM Defaults
            </h4>
            <p className="text-sm text-muted-foreground">
              Configure default LLMs for Skills and Deep Research features
            </p>

            {/* Skills Default LLM */}
            <div className="space-y-3 p-3 rounded-lg border bg-muted/30">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                <p className="font-medium">Skills Default Model</p>
              </div>
              <div className="space-y-2">
                <Select
                  value={localSettings["features.skills_default_model"] as string || "gpt-4o"}
                  onValueChange={(value) => handleSettingChange("features.skills_default_model", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectLabel>OpenAI</SelectLabel>
                      {providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                        <>
                          <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                          <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                          <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                        </>
                      )}
                      {!providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                        <SelectItem value="_no_openai_skills" disabled>No OpenAI provider configured</SelectItem>
                      )}
                    </SelectGroup>
                    <SelectGroup>
                      <SelectLabel>Anthropic</SelectLabel>
                      {providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                        <>
                          <SelectItem value="claude-3-5-sonnet">Claude 3.5 Sonnet</SelectItem>
                          <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                          <SelectItem value="claude-3-haiku">Claude 3 Haiku</SelectItem>
                        </>
                      )}
                      {!providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                        <SelectItem value="_no_anthropic_skills" disabled>No Anthropic provider configured</SelectItem>
                      )}
                    </SelectGroup>
                    <SelectGroup>
                      <SelectLabel>Google</SelectLabel>
                      {providersData?.providers?.some(p => p.provider_type === "google" && p.is_active) && (
                        <>
                          <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                          <SelectItem value="gemini-1.5-pro">Gemini 1.5 Pro</SelectItem>
                        </>
                      )}
                      {!providersData?.providers?.some(p => p.provider_type === "google" && p.is_active) && (
                        <SelectItem value="_no_google_skills" disabled>No Google provider configured</SelectItem>
                      )}
                    </SelectGroup>
                    <SelectGroup>
                      <SelectLabel>Ollama (Local)</SelectLabel>
                      {ollamaLocalModels?.chat_models?.slice(0, 10).map(model => (
                        <SelectItem key={model.name} value={model.name}>
                          {model.name} {model.parameter_size && `(${model.parameter_size})`}
                        </SelectItem>
                      ))}
                      {(!ollamaLocalModels?.chat_models || ollamaLocalModels.chat_models.length === 0) && (
                        <SelectItem value="_no_ollama_skills" disabled>No Ollama models downloaded</SelectItem>
                      )}
                    </SelectGroup>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">Default model used when running Skills</p>
              </div>
            </div>

            {/* Research Default LLMs */}
            <div className="space-y-3 p-3 rounded-lg border bg-muted/30">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-primary" />
                <p className="font-medium">Deep Research Default Models</p>
              </div>
              <p className="text-xs text-muted-foreground mb-2">
                Select which models are enabled by default for multi-LLM verification
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                <div className="flex items-center justify-between p-2 rounded border">
                  <span className="text-sm">GPT-4o</span>
                  <Switch
                    checked={localSettings["features.research_enable_gpt4o"] as boolean ?? true}
                    onCheckedChange={(checked) => handleSettingChange("features.research_enable_gpt4o", checked)}
                  />
                </div>
                <div className="flex items-center justify-between p-2 rounded border">
                  <span className="text-sm">GPT-4o Mini</span>
                  <Switch
                    checked={localSettings["features.research_enable_gpt4o_mini"] as boolean ?? false}
                    onCheckedChange={(checked) => handleSettingChange("features.research_enable_gpt4o_mini", checked)}
                  />
                </div>
                <div className="flex items-center justify-between p-2 rounded border">
                  <span className="text-sm">Claude 3.5 Sonnet</span>
                  <Switch
                    checked={localSettings["features.research_enable_claude_sonnet"] as boolean ?? true}
                    onCheckedChange={(checked) => handleSettingChange("features.research_enable_claude_sonnet", checked)}
                  />
                </div>
                <div className="flex items-center justify-between p-2 rounded border">
                  <span className="text-sm">Claude 3 Haiku</span>
                  <Switch
                    checked={localSettings["features.research_enable_claude_haiku"] as boolean ?? false}
                    onCheckedChange={(checked) => handleSettingChange("features.research_enable_claude_haiku", checked)}
                  />
                </div>
                <div className="flex items-center justify-between p-2 rounded border">
                  <span className="text-sm">Gemini Pro</span>
                  <Switch
                    checked={localSettings["features.research_enable_gemini"] as boolean ?? true}
                    onCheckedChange={(checked) => handleSettingChange("features.research_enable_gemini", checked)}
                  />
                </div>
                <div className="flex items-center justify-between p-2 rounded border">
                  <span className="text-sm">Llama 3.1 70B</span>
                  <Switch
                    checked={localSettings["features.research_enable_llama"] as boolean ?? false}
                    onCheckedChange={(checked) => handleSettingChange("features.research_enable_llama", checked)}
                  />
                </div>
              </div>
              <div className="space-y-2 pt-2">
                <label className="text-sm font-medium">Default Verification Rounds</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  value={localSettings["features.research_default_rounds"] as number ?? 3}
                  onChange={(e) => handleSettingChange("features.research_default_rounds", parseInt(e.target.value) || 3)}
                />
                <p className="text-xs text-muted-foreground">Number of verification rounds for Deep Research</p>
              </div>
            </div>
          </div>

          {/* Style Defaults */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <PenTool className="h-4 w-4" />
              Style Defaults
            </h4>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Default Tone</label>
                <Select
                  value={localSettings["generation.default_tone"] as string || "professional"}
                  onValueChange={(value) => handleSettingChange("generation.default_tone", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="professional">Professional</SelectItem>
                    <SelectItem value="casual">Casual</SelectItem>
                    <SelectItem value="technical">Technical</SelectItem>
                    <SelectItem value="friendly">Friendly</SelectItem>
                    <SelectItem value="academic">Academic</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Default Style</label>
                <Select
                  value={localSettings["generation.default_style"] as string || "business"}
                  onValueChange={(value) => handleSettingChange("generation.default_style", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="business">Business</SelectItem>
                    <SelectItem value="creative">Creative</SelectItem>
                    <SelectItem value="minimal">Minimal</SelectItem>
                    <SelectItem value="bold">Bold</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
