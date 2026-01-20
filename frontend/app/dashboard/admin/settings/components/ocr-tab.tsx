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
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Switch } from "@/components/ui/switch";
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
}: OcrTabProps) {
  return (
    <TabsContent value="ocr" className="space-y-6">
      {ocrLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          {/* OCR Provider Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Scan className="h-5 w-5" />
                OCR Provider Configuration
              </CardTitle>
              <CardDescription>
                Configure OCR engine, models, and languages for document processing
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
