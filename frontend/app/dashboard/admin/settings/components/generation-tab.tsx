"use client";

import {
  PenTool,
  FileText,
  CheckCircle,
  Eye,
  Sparkles,
  BarChart3,
} from "lucide-react";
import { Input } from "@/components/ui/input";
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
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["generation.include_sources"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("generation.include_sources", e.target.checked)}
                />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg border">
                <div>
                  <p className="font-medium">Include Images</p>
                  <p className="text-sm text-muted-foreground">
                    Auto-generate relevant images
                  </p>
                </div>
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["generation.include_images"] as boolean ?? true}
                  onChange={(e) => handleSettingChange("generation.include_images", e.target.checked)}
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Image Backend</label>
              <select
                className="w-full h-10 px-3 rounded-md border bg-background"
                value={localSettings["generation.image_backend"] as string || "picsum"}
                onChange={(e) => handleSettingChange("generation.image_backend", e.target.value)}
              >
                <option value="picsum">Picsum (Free - No API Key)</option>
                <option value="unsplash">Unsplash (Requires API Key)</option>
                <option value="pexels">Pexels (Requires API Key)</option>
                <option value="openai">OpenAI DALL-E (Requires API Key)</option>
                <option value="stability">Stability AI (Requires API Key)</option>
                <option value="automatic1111">Automatic1111 (Local SD)</option>
                <option value="disabled">Disabled</option>
              </select>
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
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["generation.enable_quality_review"] as boolean ?? true}
                onChange={(e) => handleSettingChange("generation.enable_quality_review", e.target.checked)}
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
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["generation.enable_template_vision_analysis"] as boolean ?? false}
                  onChange={(e) => handleSettingChange("generation.enable_template_vision_analysis", e.target.checked)}
                />
              </div>
              <div className="space-y-2 pt-2">
                <label className="text-sm font-medium">Template Vision Model</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["generation.template_vision_model"] as string || "auto"}
                  onChange={(e) => handleSettingChange("generation.template_vision_model", e.target.value)}
                >
                  <option value="auto">Auto (Use default vision model)</option>
                  <optgroup label="OpenAI (Vision-capable)">
                    {providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                      <>
                        <option value="gpt-4o">GPT-4o</option>
                        <option value="gpt-4o-mini">GPT-4o Mini</option>
                        <option value="gpt-4-vision-preview">GPT-4 Vision</option>
                      </>
                    )}
                    {!providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                      <option disabled>No OpenAI provider configured</option>
                    )}
                  </optgroup>
                  <optgroup label="Anthropic (Vision-capable)">
                    {providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                      <>
                        <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                        <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                        <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
                        <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                      </>
                    )}
                    {!providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                      <option disabled>No Anthropic provider configured</option>
                    )}
                  </optgroup>
                  <optgroup label="Ollama (Local - Downloaded)">
                    {ollamaLocalModels?.chat_models?.filter(m =>
                      m.name.includes("llava") || m.name.includes("bakllava") ||
                      m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                      m.name.includes("minicpm-v")
                    ).map(model => (
                      <option key={model.name} value={model.name}>
                        {model.name} {model.parameter_size && `(${model.parameter_size})`}
                      </option>
                    ))}
                    {(!ollamaLocalModels?.chat_models || ollamaLocalModels.chat_models.filter(m =>
                      m.name.includes("llava") || m.name.includes("bakllava") ||
                      m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                      m.name.includes("minicpm-v")
                    ).length === 0) && (
                      <option disabled>No vision models downloaded (try: ollama pull llava)</option>
                    )}
                  </optgroup>
                </select>
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
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={localSettings["generation.enable_vision_review"] as boolean ?? false}
                  onChange={(e) => handleSettingChange("generation.enable_vision_review", e.target.checked)}
                />
              </div>
              <div className="space-y-3 pt-2">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Vision Review Model</label>
                  <select
                    className="w-full h-10 px-3 rounded-md border bg-background"
                    value={localSettings["generation.vision_review_model"] as string || "auto"}
                    onChange={(e) => handleSettingChange("generation.vision_review_model", e.target.value)}
                  >
                    <option value="auto">Auto (Use default vision model)</option>
                    <optgroup label="OpenAI (Vision-capable)">
                      {providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                        <>
                          <option value="gpt-4o">GPT-4o</option>
                          <option value="gpt-4o-mini">GPT-4o Mini</option>
                          <option value="gpt-4-vision-preview">GPT-4 Vision</option>
                        </>
                      )}
                      {!providersData?.providers?.some(p => p.provider_type === "openai" && p.is_active) && (
                        <option disabled>No OpenAI provider configured</option>
                      )}
                    </optgroup>
                    <optgroup label="Anthropic (Vision-capable)">
                      {providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                        <>
                          <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                          <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                          <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
                          <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                        </>
                      )}
                      {!providersData?.providers?.some(p => p.provider_type === "anthropic" && p.is_active) && (
                        <option disabled>No Anthropic provider configured</option>
                      )}
                    </optgroup>
                    <optgroup label="Ollama (Local - Downloaded)">
                      {ollamaLocalModels?.chat_models?.filter(m =>
                        m.name.includes("llava") || m.name.includes("bakllava") ||
                        m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                        m.name.includes("minicpm-v")
                      ).map(model => (
                        <option key={model.name} value={model.name}>
                          {model.name} {model.parameter_size && `(${model.parameter_size})`}
                        </option>
                      ))}
                      {(!ollamaLocalModels?.chat_models || ollamaLocalModels.chat_models.filter(m =>
                        m.name.includes("llava") || m.name.includes("bakllava") ||
                        m.name.includes("vision") || m.name.includes("qwen2.5vl") ||
                        m.name.includes("minicpm-v")
                      ).length === 0) && (
                        <option disabled>No vision models downloaded (try: ollama pull llava)</option>
                      )}
                    </optgroup>
                  </select>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">Review All Slides</p>
                    <p className="text-xs text-muted-foreground">
                      If off, only reviews slides at risk of issues
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    className="h-4 w-4"
                    checked={localSettings["generation.vision_review_all_slides"] as boolean ?? false}
                    onChange={(e) => handleSettingChange("generation.vision_review_all_slides", e.target.checked)}
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
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["generation.enable_per_slide_constraints"] as boolean ?? true}
                onChange={(e) => handleSettingChange("generation.enable_per_slide_constraints", e.target.checked)}
              />
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg border">
              <div>
                <p className="font-medium">LLM Content Rewriting</p>
                <p className="text-sm text-muted-foreground">
                  Use LLM to condense overflowing content intelligently
                </p>
              </div>
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["generation.enable_llm_rewrite"] as boolean ?? true}
                onChange={(e) => handleSettingChange("generation.enable_llm_rewrite", e.target.checked)}
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
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={localSettings["generation.auto_charts"] as boolean ?? false}
                onChange={(e) => handleSettingChange("generation.auto_charts", e.target.checked)}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Chart Style</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["generation.chart_style"] as string || "business"}
                  onChange={(e) => handleSettingChange("generation.chart_style", e.target.value)}
                >
                  <option value="business">Business (Professional)</option>
                  <option value="modern">Modern (Clean lines)</option>
                  <option value="minimal">Minimal (Simple)</option>
                  <option value="colorful">Colorful (Vibrant)</option>
                </select>
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

          {/* Style Defaults */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <PenTool className="h-4 w-4" />
              Style Defaults
            </h4>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Default Tone</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["generation.default_tone"] as string || "professional"}
                  onChange={(e) => handleSettingChange("generation.default_tone", e.target.value)}
                >
                  <option value="professional">Professional</option>
                  <option value="casual">Casual</option>
                  <option value="technical">Technical</option>
                  <option value="friendly">Friendly</option>
                  <option value="academic">Academic</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Default Style</label>
                <select
                  className="w-full h-10 px-3 rounded-md border bg-background"
                  value={localSettings["generation.default_style"] as string || "business"}
                  onChange={(e) => handleSettingChange("generation.default_style", e.target.value)}
                >
                  <option value="business">Business</option>
                  <option value="creative">Creative</option>
                  <option value="minimal">Minimal</option>
                  <option value="bold">Bold</option>
                </select>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
