"use client";

import { getErrorMessage } from "@/lib/errors";
import {
  Scan,
  Sparkles,
  FileText,
  Zap,
  HardDrive,
  Database,
  CheckCircle,
  AlertCircle,
  Loader2,
  Download,
  Eye,
  Image,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TabsContent } from "@/components/ui/tabs";

interface OcrModel {
  name: string;
  size: string;
}

interface OcrModelsData {
  model_dir?: string;
  status?: string;
  total_size?: string;
  downloaded?: OcrModel[];
}

interface OcrSettingsData {
  'ocr.provider'?: string;
  'ocr.paddle.variant'?: string;
  'ocr.paddle.languages'?: string[];
  'ocr.paddle.auto_download'?: boolean;
  'ocr.tesseract.fallback_enabled'?: boolean;
}

interface OcrData {
  settings?: OcrSettingsData;
  models?: OcrModelsData;
}

interface VlmConfig {
  enabled: boolean;
  provider: string;
  model: string;
  max_images_per_request: number;
  auto_process_visual_docs: boolean;
  extract_tables: boolean;
  extract_charts: boolean;
  ocr_fallback: boolean;
}

interface VisionModel {
  name: string;
  size?: number;
  family?: string;
  families?: string[];
  parameter_size?: string;
}

interface DownloadResult {
  success: boolean;
  message: string;
}

interface OcrTabProps {
  ocrLoading: boolean;
  ocrData: OcrData | undefined;
  refetchOCR: () => void;
  updateOCRSettings: {
    mutate: (settings: Record<string, unknown>) => void;
  };
  downloadModels: {
    mutateAsync: (params: { languages: string[]; variant: string }) => Promise<{
      status: string;
      downloaded: string[];
    }>;
  };
  downloadingModels: boolean;
  setDownloadingModels: (value: boolean) => void;
  downloadResult: DownloadResult | null;
  setDownloadResult: (value: DownloadResult | null) => void;
  // VLM Configuration props
  vlmConfig?: VlmConfig;
  vlmLoading?: boolean;
  updateVlmConfig?: {
    mutate: (config: Partial<VlmConfig>) => void;
    isPending?: boolean;
  };
  testVlm?: {
    mutateAsync: (params: { imageUrl?: string; testText?: string }) => Promise<{
      success: boolean;
      provider?: string;
      model?: string;
      capabilities?: string[];
      error?: string;
      message: string;
    }>;
    isPending?: boolean;
  };
  visionModels?: VisionModel[];
}

export function OcrTab({
  ocrLoading,
  ocrData,
  refetchOCR,
  updateOCRSettings,
  downloadModels,
  downloadingModels,
  setDownloadingModels,
  downloadResult,
  setDownloadResult,
  vlmConfig,
  vlmLoading,
  updateVlmConfig,
  testVlm,
  visionModels,
}: OcrTabProps) {
  return (
    <TabsContent value="ocr" className="space-y-6">
      {ocrLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          {/* VLM (Vision Language Model) Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Vision Language Model (VLM)
              </CardTitle>
              <CardDescription>
                Configure AI vision models for processing images, charts, and infographics during document upload
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {vlmLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <>
                  {/* VLM Enable Toggle */}
                  <div className="flex items-center justify-between p-3 rounded-lg border">
                    <div>
                      <p className="font-medium">Enable VLM Processing</p>
                      <p className="text-sm text-muted-foreground">
                        Use vision models to analyze images and generate descriptions during document processing
                      </p>
                    </div>
                    <Switch
                      checked={vlmConfig?.enabled ?? false}
                      onCheckedChange={(checked) => {
                        updateVlmConfig?.mutate({ enabled: checked });
                      }}
                    />
                  </div>

                  {vlmConfig?.enabled && (
                    <>
                      {/* Provider Selection */}
                      <div className="space-y-2">
                        <Label htmlFor="vlm-provider">VLM Provider</Label>
                        <Select
                          value={vlmConfig?.provider || 'ollama'}
                          onValueChange={(value) => {
                            updateVlmConfig?.mutate({ provider: value });
                          }}
                        >
                          <SelectTrigger id="vlm-provider">
                            <SelectValue placeholder="Select VLM provider" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="ollama">
                              <div className="flex items-center gap-2">
                                <Sparkles className="h-4 w-4" />
                                <div>
                                  <p className="font-medium">Ollama (Local)</p>
                                  <p className="text-xs text-muted-foreground">Free, runs locally with llava, etc.</p>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="openai">
                              <div className="flex items-center gap-2">
                                <Zap className="h-4 w-4" />
                                <div>
                                  <p className="font-medium">OpenAI</p>
                                  <p className="text-xs text-muted-foreground">GPT-4o vision (requires API key)</p>
                                </div>
                              </div>
                            </SelectItem>
                            <SelectItem value="anthropic">
                              <div className="flex items-center gap-2">
                                <FileText className="h-4 w-4" />
                                <div>
                                  <p className="font-medium">Anthropic</p>
                                  <p className="text-xs text-muted-foreground">Claude 3 vision (requires API key)</p>
                                </div>
                              </div>
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Model Selection */}
                      <div className="space-y-2">
                        <Label htmlFor="vlm-model">Vision Model</Label>
                        {vlmConfig?.provider === 'ollama' && visionModels && visionModels.length > 0 ? (
                          <Select
                            value={vlmConfig?.model || 'llava'}
                            onValueChange={(value) => {
                              updateVlmConfig?.mutate({ model: value });
                            }}
                          >
                            <SelectTrigger id="vlm-model">
                              <SelectValue placeholder="Select vision model" />
                            </SelectTrigger>
                            <SelectContent>
                              {visionModels.map((model) => (
                                <SelectItem key={model.name} value={model.name}>
                                  <div className="flex items-center gap-2">
                                    <Image className="h-4 w-4" />
                                    <span>{model.name}</span>
                                    {model.parameter_size && (
                                      <Badge variant="outline" className="text-xs">
                                        {model.parameter_size}
                                      </Badge>
                                    )}
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        ) : (
                          <Input
                            id="vlm-model"
                            value={vlmConfig?.model || ''}
                            onChange={(e) => {
                              updateVlmConfig?.mutate({ model: e.target.value });
                            }}
                            placeholder={
                              vlmConfig?.provider === 'ollama' ? 'llava' :
                              vlmConfig?.provider === 'openai' ? 'gpt-4o' :
                              'claude-3-5-sonnet-20241022'
                            }
                          />
                        )}
                        {vlmConfig?.provider === 'ollama' && (!visionModels || visionModels.length === 0) && (
                          <p className="text-xs text-amber-600">
                            No vision models detected in Ollama. Pull a vision model like llava: ollama pull llava
                          </p>
                        )}
                      </div>

                      {/* Max Images */}
                      <div className="space-y-2">
                        <Label htmlFor="vlm-max-images">Max Images per Request</Label>
                        <Input
                          id="vlm-max-images"
                          type="number"
                          min={1}
                          max={20}
                          value={vlmConfig?.max_images_per_request || 10}
                          onChange={(e) => {
                            updateVlmConfig?.mutate({ max_images_per_request: parseInt(e.target.value) || 10 });
                          }}
                          className="w-24"
                        />
                        <p className="text-xs text-muted-foreground">
                          Maximum number of images to process per document (1-20)
                        </p>
                      </div>

                      {/* Processing Options */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label htmlFor="vlm-auto-process">Auto-Process Visual Documents</Label>
                            <p className="text-sm text-muted-foreground">
                              Automatically analyze images in uploaded documents
                            </p>
                          </div>
                          <Switch
                            id="vlm-auto-process"
                            checked={vlmConfig?.auto_process_visual_docs ?? true}
                            onCheckedChange={(checked) => {
                              updateVlmConfig?.mutate({ auto_process_visual_docs: checked });
                            }}
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label htmlFor="vlm-extract-tables">Extract Tables</Label>
                            <p className="text-sm text-muted-foreground">
                              Use vision model to extract table contents
                            </p>
                          </div>
                          <Switch
                            id="vlm-extract-tables"
                            checked={vlmConfig?.extract_tables ?? true}
                            onCheckedChange={(checked) => {
                              updateVlmConfig?.mutate({ extract_tables: checked });
                            }}
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label htmlFor="vlm-extract-charts">Extract Charts</Label>
                            <p className="text-sm text-muted-foreground">
                              Use vision model to describe charts and graphs
                            </p>
                          </div>
                          <Switch
                            id="vlm-extract-charts"
                            checked={vlmConfig?.extract_charts ?? true}
                            onCheckedChange={(checked) => {
                              updateVlmConfig?.mutate({ extract_charts: checked });
                            }}
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label htmlFor="vlm-ocr-fallback">OCR Fallback</Label>
                            <p className="text-sm text-muted-foreground">
                              Fall back to OCR if VLM fails to process image
                            </p>
                          </div>
                          <Switch
                            id="vlm-ocr-fallback"
                            checked={vlmConfig?.ocr_fallback ?? true}
                            onCheckedChange={(checked) => {
                              updateVlmConfig?.mutate({ ocr_fallback: checked });
                            }}
                          />
                        </div>
                      </div>

                      {/* Test VLM */}
                      {testVlm && (
                        <div className="pt-4 border-t">
                          <Button
                            onClick={async () => {
                              try {
                                const result = await testVlm.mutateAsync({});
                                if (result.success) {
                                  alert(`VLM test successful!\nProvider: ${result.provider}\nModel: ${result.model}`);
                                } else {
                                  alert(`VLM test failed: ${result.error || result.message}`);
                                }
                              } catch (error) {
                                alert(`VLM test error: ${getErrorMessage(error)}`);
                              }
                            }}
                            disabled={testVlm.isPending}
                            variant="outline"
                            className="w-full"
                          >
                            {testVlm.isPending ? (
                              <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Testing VLM...
                              </>
                            ) : (
                              <>
                                <Play className="h-4 w-4 mr-2" />
                                Test VLM Connection
                              </>
                            )}
                          </Button>
                        </div>
                      )}
                    </>
                  )}

                  {!vlmConfig?.enabled && (
                    <div className="p-3 bg-muted/50 rounded-lg text-sm text-muted-foreground">
                      Enable VLM to use AI vision models for analyzing images, charts, and infographics
                      in your documents. This creates searchable text descriptions from visual content.
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* OCR Provider Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Scan className="h-5 w-5" />
                OCR Provider Configuration
              </CardTitle>
              <CardDescription>
                Configure OCR engine, models, and languages for text extraction from images
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Provider Selection */}
              <div className="space-y-2">
                <Label htmlFor="ocr-provider">OCR Provider</Label>
                <Select
                  value={ocrData?.settings?.['ocr.provider'] as string || 'paddleocr'}
                  onValueChange={(value) => {
                    updateOCRSettings.mutate({ 'ocr.provider': value });
                  }}
                >
                  <SelectTrigger id="ocr-provider">
                    <SelectValue placeholder="Select OCR provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="paddleocr">
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4" />
                        <div>
                          <p className="font-medium">PaddleOCR</p>
                          <p className="text-xs text-muted-foreground">Deep learning-based, high accuracy</p>
                        </div>
                      </div>
                    </SelectItem>
                    <SelectItem value="tesseract">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4" />
                        <div>
                          <p className="font-medium">Tesseract</p>
                          <p className="text-xs text-muted-foreground">Traditional OCR, fast and lightweight</p>
                        </div>
                      </div>
                    </SelectItem>
                    <SelectItem value="auto">
                      <div className="flex items-center gap-2">
                        <Zap className="h-4 w-4" />
                        <div>
                          <p className="font-medium">Auto (Try Both)</p>
                          <p className="text-xs text-muted-foreground">PaddleOCR with Tesseract fallback</p>
                        </div>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* PaddleOCR Settings (conditional) */}
              {ocrData?.settings?.['ocr.provider'] !== 'tesseract' && (
                <>
                  {/* Model Variant */}
                  <div className="space-y-2">
                    <Label htmlFor="ocr-variant">Model Variant</Label>
                    <Select
                      value={ocrData?.settings?.['ocr.paddle.variant'] as string || 'server'}
                      onValueChange={(value) => {
                        updateOCRSettings.mutate({ 'ocr.paddle.variant': value });
                      }}
                    >
                      <SelectTrigger id="ocr-variant">
                        <SelectValue placeholder="Select model variant" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="server">
                          <div>
                            <p className="font-medium">Server (Accurate)</p>
                            <p className="text-xs text-muted-foreground">Higher accuracy, slower processing</p>
                          </div>
                        </SelectItem>
                        <SelectItem value="mobile">
                          <div>
                            <p className="font-medium">Mobile (Fast)</p>
                            <p className="text-xs text-muted-foreground">Faster processing, lower accuracy</p>
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Languages */}
                  <div className="space-y-2">
                    <Label>Languages</Label>
                    <div className="flex flex-wrap gap-2">
                      {['en', 'de', 'fr', 'es', 'zh', 'ja', 'ko', 'ar'].map((lang) => {
                        const labels: Record<string, string> = {
                          en: 'English',
                          de: 'German',
                          fr: 'French',
                          es: 'Spanish',
                          zh: 'Chinese',
                          ja: 'Japanese',
                          ko: 'Korean',
                          ar: 'Arabic',
                        };
                        const currentLanguages = (ocrData?.settings?.['ocr.paddle.languages'] as string[]) || ['en', 'de'];
                        const isSelected = currentLanguages.includes(lang);

                        return (
                          <Button
                            key={lang}
                            variant={isSelected ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => {
                              const newLanguages = isSelected
                                ? currentLanguages.filter(l => l !== lang)
                                : [...currentLanguages, lang];
                              updateOCRSettings.mutate({ 'ocr.paddle.languages': newLanguages });
                            }}
                          >
                            {labels[lang]}
                          </Button>
                        );
                      })}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Selected: {((ocrData?.settings?.['ocr.paddle.languages'] as string[]) || []).join(', ')}
                    </p>
                  </div>

                  {/* Auto Download Toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="auto-download">Auto-Download Models</Label>
                      <p className="text-sm text-muted-foreground">
                        Automatically download missing models on startup
                      </p>
                    </div>
                    <Switch
                      id="auto-download"
                      checked={ocrData?.settings?.['ocr.paddle.auto_download'] as boolean || true}
                      onCheckedChange={(checked) => {
                        updateOCRSettings.mutate({ 'ocr.paddle.auto_download': checked });
                      }}
                    />
                  </div>
                </>
              )}

              {/* Tesseract Fallback */}
              {ocrData?.settings?.['ocr.provider'] === 'paddleocr' && (
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="tesseract-fallback">Tesseract Fallback</Label>
                    <p className="text-sm text-muted-foreground">
                      Fall back to Tesseract if PaddleOCR fails
                    </p>
                  </div>
                  <Switch
                    id="tesseract-fallback"
                    checked={ocrData?.settings?.['ocr.tesseract.fallback_enabled'] as boolean || true}
                    onCheckedChange={(checked) => {
                      updateOCRSettings.mutate({ 'ocr.tesseract.fallback_enabled': checked });
                    }}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Model Management */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <HardDrive className="h-5 w-5" />
                Downloaded Models
              </CardTitle>
              <CardDescription>
                Manage PaddleOCR model downloads and storage
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Model Status */}
              <div className="grid gap-4">
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex items-center gap-3">
                    <Database className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="font-medium">Model Directory</p>
                      <p className="text-sm text-muted-foreground font-mono">
                        {ocrData?.models?.model_dir || './data/paddle_models'}
                      </p>
                    </div>
                  </div>
                  <Badge variant={ocrData?.models?.status === 'installed' ? 'default' : 'secondary'}>
                    {ocrData?.models?.status || 'unknown'}
                  </Badge>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex items-center gap-3">
                    <HardDrive className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="font-medium">Total Size</p>
                      <p className="text-sm text-muted-foreground">
                        {ocrData?.models?.total_size || '0 MB'}
                      </p>
                    </div>
                  </div>
                  <Badge variant="outline">
                    {ocrData?.models?.downloaded?.length || 0} models
                  </Badge>
                </div>
              </div>

              {/* Downloaded Models List */}
              {ocrData?.models?.downloaded && ocrData.models.downloaded.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Model Files:</p>
                  <div className="space-y-1 max-h-60 overflow-y-auto">
                    {ocrData.models.downloaded.map((model, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2 rounded border text-sm"
                      >
                        <div className="flex items-center gap-2">
                          <CheckCircle className="h-4 w-4 text-green-500" />
                          <span className="font-mono text-xs">{model.name}</span>
                        </div>
                        <span className="text-muted-foreground text-xs">{model.size}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Download Button */}
              <div className="pt-4 border-t">
                <Button
                  onClick={async () => {
                    setDownloadingModels(true);
                    setDownloadResult(null);
                    try {
                      const result = await downloadModels.mutateAsync({
                        languages: (ocrData?.settings?.['ocr.paddle.languages'] as string[]) || ['en', 'de'],
                        variant: (ocrData?.settings?.['ocr.paddle.variant'] as string) || 'server',
                      });
                      setDownloadResult({
                        success: result.status === 'success',
                        message: `Downloaded ${result.downloaded.length} languages successfully`,
                      });
                      refetchOCR();
                    } catch (error) {
                      setDownloadResult({
                        success: false,
                        message: getErrorMessage(error),
                      });
                    } finally {
                      setDownloadingModels(false);
                    }
                  }}
                  disabled={downloadingModels}
                  className="w-full"
                >
                  {downloadingModels ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Downloading Models...
                    </>
                  ) : (
                    <>
                      <Download className="h-4 w-4 mr-2" />
                      Download Selected Models
                    </>
                  )}
                </Button>

                {downloadResult && (
                  <Alert className="mt-3">
                    {downloadResult.success ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red-500" />
                    )}
                    <AlertDescription>{downloadResult.message}</AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </TabsContent>
  );
}
