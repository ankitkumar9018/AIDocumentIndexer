"use client";

import { TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Volume2, Mic } from "lucide-react";

interface AudioTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function AudioTab({ localSettings, handleSettingChange }: AudioTabProps) {
  return (
    <TabsContent value="audio" className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Volume2 className="h-5 w-5" />
            Text-to-Speech Configuration
          </CardTitle>
          <CardDescription>
            Configure TTS providers, fallback chains, and voice generation settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Default TTS Provider */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Default TTS Provider</label>
            <Select
              value={localSettings["tts.default_provider"] as string ?? "openai"}
              onValueChange={(value) => handleSettingChange("tts.default_provider", value)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="openai">OpenAI TTS</SelectItem>
                <SelectItem value="elevenlabs">ElevenLabs</SelectItem>
                <SelectItem value="chatterbox">Chatterbox (Resemble AI)</SelectItem>
                <SelectItem value="cosyvoice">CosyVoice2 (Alibaba)</SelectItem>
                <SelectItem value="fish_speech">Fish Speech</SelectItem>
                <SelectItem value="edge">Edge TTS (Free)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Primary provider for audio overview generation and TTS
            </p>
          </div>

          {/* Ultra-Fast Provider */}
          <div className="space-y-3">
            <label className="text-sm font-medium">Ultra-Fast Streaming Provider</label>
            <Select
              value={localSettings["tts.ultra_fast_provider"] as string ?? "cosyvoice"}
              onValueChange={(value) => handleSettingChange("tts.ultra_fast_provider", value)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="cosyvoice">CosyVoice2 (150ms latency)</SelectItem>
                <SelectItem value="chatterbox">Chatterbox</SelectItem>
                <SelectItem value="fish_speech">Fish Speech</SelectItem>
                <SelectItem value="openai">OpenAI TTS</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Provider used for real-time streaming with minimal latency
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Provider Toggles */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Mic className="h-5 w-5" />
            TTS Providers
          </CardTitle>
          <CardDescription>
            Enable or disable individual TTS providers
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Chatterbox */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Chatterbox TTS</p>
              <p className="text-sm text-muted-foreground">
                Resemble AI open-source model with emotional expressiveness
              </p>
            </div>
            <Switch
              checked={localSettings["tts.chatterbox_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("tts.chatterbox_enabled", checked)}
            />
          </div>
          {(localSettings["tts.chatterbox_enabled"] as boolean ?? true) && (
            <div className="ml-6 space-y-3 p-3 bg-muted/30 rounded-lg">
              <div>
                <label className="text-sm font-medium">Emotional Exaggeration</label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localSettings["tts.chatterbox_exaggeration"] as number ?? 0.5}
                  onChange={(e) => handleSettingChange("tts.chatterbox_exaggeration", parseFloat(e.target.value))}
                  className="mt-1 w-32"
                />
                <p className="text-xs text-muted-foreground mt-1">Emotion intensity (0.0 = neutral, 1.0 = dramatic)</p>
              </div>
              <div>
                <label className="text-sm font-medium">CFG Weight</label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={localSettings["tts.chatterbox_cfg_weight"] as number ?? 0.5}
                  onChange={(e) => handleSettingChange("tts.chatterbox_cfg_weight", parseFloat(e.target.value))}
                  className="mt-1 w-32"
                />
                <p className="text-xs text-muted-foreground mt-1">Classifier-free guidance weight (0.0-1.0)</p>
              </div>
            </div>
          )}

          {/* CosyVoice */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">CosyVoice2</p>
              <p className="text-sm text-muted-foreground">
                Alibaba open-source — 150ms latency, multilingual, voice cloning
              </p>
            </div>
            <Switch
              checked={localSettings["tts.cosyvoice_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("tts.cosyvoice_enabled", checked)}
            />
          </div>

          {/* Fish Speech */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Fish Speech</p>
              <p className="text-sm text-muted-foreground">
                Multilingual TTS — ELO 1339, 13 languages, zero-shot voice cloning
              </p>
            </div>
            <Switch
              checked={localSettings["tts.fish_speech_enabled"] as boolean ?? true}
              onCheckedChange={(checked) => handleSettingChange("tts.fish_speech_enabled", checked)}
            />
          </div>

          <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg text-sm text-muted-foreground">
            When the default provider is unavailable, the system automatically falls back through the configured provider chain.
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
