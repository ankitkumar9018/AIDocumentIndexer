"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Download,
  Share2,
  RotateCcw,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

interface AudioPlayerProps {
  src: string;
  title?: string;
  subtitle?: string;
  transcript?: TranscriptSegment[];
  speakers?: { id: string; name: string; voice?: string }[];
  onEnded?: () => void;
  className?: string;
}

interface TranscriptSegment {
  speaker: string;
  text: string;
  startTime?: number;
  endTime?: number;
  emotion?: string;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function AudioPlayer({
  src,
  title,
  subtitle,
  transcript,
  speakers,
  onEnded,
  className,
}: AudioPlayerProps) {
  // Build speaker name map from speakers array or use defaults
  const speakerNameMap: Record<string, string> = {};
  if (speakers && speakers.length > 0) {
    speakers.forEach(s => {
      speakerNameMap[s.id] = s.name;
    });
  }
  // Get speaker display name
  const getSpeakerName = (speakerId: string): string => {
    if (speakerNameMap[speakerId]) {
      return speakerNameMap[speakerId];
    }
    // Fallback defaults
    if (speakerId === "host1") return "Host 1";
    if (speakerId === "host2") return "Host 2";
    return speakerId;
  };
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState(-1);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Fetch audio with auth and create blob URL
  useEffect(() => {
    if (!src) return;

    let cancelled = false;

    const fetchAudio = async () => {
      try {
        setIsLoading(true);
        setLoadError(null);

        // For relative API URLs, fetch with auth
        if (src.startsWith('/api/')) {
          const response = await api.fetchWithAuth(src);
          if (!response.ok) {
            throw new Error(`Failed to load audio: ${response.status}`);
          }
          const blob = await response.blob();
          if (!cancelled) {
            const url = URL.createObjectURL(blob);
            setBlobUrl(url);
            // Note: isLoading will be set to false by handleCanPlay/handleLoadedMetadata
          }
        } else {
          // External URL, use directly
          setBlobUrl(src);
          // For external URLs, let the audio element handle loading state
        }
      } catch (error) {
        console.error('Failed to load audio:', error);
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : 'Failed to load audio');
          setIsLoading(false);
        }
      }
    };

    fetchAudio();

    return () => {
      cancelled = true;
      if (blobUrl && blobUrl.startsWith('blob:')) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [src]);

  // Update current time and active segment
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);

      // Find active transcript segment (only if timing info is available)
      if (transcript) {
        const index = transcript.findIndex(
          (seg) =>
            seg.startTime !== undefined &&
            seg.endTime !== undefined &&
            audio.currentTime >= seg.startTime &&
            audio.currentTime < seg.endTime
        );
        setActiveSegmentIndex(index);
      }
    };

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsLoading(false);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      onEnded?.();
    };

    const handleCanPlay = () => {
      setIsLoading(false);
    };

    const handleWaiting = () => {
      setIsLoading(true);
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("canplay", handleCanPlay);
    audio.addEventListener("waiting", handleWaiting);

    return () => {
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("canplay", handleCanPlay);
      audio.removeEventListener("waiting", handleWaiting);
    };
  }, [transcript, onEnded, blobUrl]); // Added blobUrl to re-attach listeners when audio element is created

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const handleSeek = useCallback((value: number[]) => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = value[0];
    setCurrentTime(value[0]);
  }, []);

  const handleVolumeChange = useCallback((value: number[]) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newVolume = value[0];
    audio.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  }, []);

  const toggleMute = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isMuted) {
      audio.volume = volume || 1;
      setIsMuted(false);
    } else {
      audio.volume = 0;
      setIsMuted(true);
    }
  }, [isMuted, volume]);

  const skip = useCallback((seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = Math.max(
      0,
      Math.min(audio.duration, audio.currentTime + seconds)
    );
  }, []);

  const changePlaybackRate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const rates = [0.5, 0.75, 1, 1.25, 1.5, 2];
    const currentIndex = rates.indexOf(playbackRate);
    const nextIndex = (currentIndex + 1) % rates.length;
    const newRate = rates[nextIndex];

    audio.playbackRate = newRate;
    setPlaybackRate(newRate);
  }, [playbackRate]);

  const restart = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = 0;
    audio.play();
    setIsPlaying(true);
  }, []);

  const jumpToSegment = useCallback((segment: TranscriptSegment) => {
    const audio = audioRef.current;
    if (!audio || segment.startTime === undefined) return;

    audio.currentTime = segment.startTime;
    if (!isPlaying) {
      audio.play();
      setIsPlaying(true);
    }
  }, [isPlaying]);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Show error state
  if (loadError) {
    return (
      <div className={cn("space-y-4", className)}>
        <div className="bg-card border border-destructive/50 rounded-lg p-4 text-center">
          <p className="text-sm text-destructive">{loadError}</p>
          <Button
            variant="outline"
            size="sm"
            className="mt-2"
            onClick={() => {
              setLoadError(null);
              setBlobUrl(null);
            }}
          >
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Hidden audio element */}
      {blobUrl && <audio ref={audioRef} src={blobUrl} preload="metadata" />}

      {/* Main player card */}
      <div className="bg-card border rounded-lg p-4 space-y-4">
        {/* Title and info */}
        {(title || subtitle) && (
          <div className="space-y-1">
            {title && <h3 className="font-semibold text-lg">{title}</h3>}
            {subtitle && (
              <p className="text-sm text-muted-foreground">{subtitle}</p>
            )}
          </div>
        )}

        {/* Progress bar */}
        <div className="space-y-2">
          <Slider
            value={[currentTime]}
            max={duration || 100}
            step={0.1}
            onValueChange={handleSeek}
            className="cursor-pointer"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* Restart */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" onClick={restart}>
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Restart</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Skip back */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" onClick={() => skip(-10)}>
                    <SkipBack className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Back 10s</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Play/Pause */}
            <Button
              size="icon"
              className="h-12 w-12 rounded-full"
              onClick={togglePlay}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : isPlaying ? (
                <Pause className="h-6 w-6" />
              ) : (
                <Play className="h-6 w-6 ml-0.5" />
              )}
            </Button>

            {/* Skip forward */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" onClick={() => skip(10)}>
                    <SkipForward className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Forward 10s</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Playback rate */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs font-mono"
                    onClick={changePlaybackRate}
                  >
                    {playbackRate}x
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Change speed</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          <div className="flex items-center gap-2">
            {/* Volume */}
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={toggleMute}>
                {isMuted ? (
                  <VolumeX className="h-4 w-4" />
                ) : (
                  <Volume2 className="h-4 w-4" />
                )}
              </Button>
              <Slider
                value={[isMuted ? 0 : volume]}
                max={1}
                step={0.01}
                onValueChange={handleVolumeChange}
                className="w-20"
              />
            </div>

            {/* Download */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" asChild disabled={!blobUrl}>
                    <a href={blobUrl || '#'} download="audio.mp3">
                      <Download className="h-4 w-4" />
                    </a>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Download</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Share */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      navigator.clipboard.writeText(window.location.href);
                    }}
                  >
                    <Share2 className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Copy link</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </div>

      {/* Transcript with sync */}
      {transcript && transcript.length > 0 && (
        <div className="bg-card border rounded-lg p-4">
          <h4 className="font-medium mb-3">Transcript</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {transcript.map((segment, index) => (
              <div
                key={index}
                className={cn(
                  "p-2 rounded transition-colors",
                  segment.startTime !== undefined && "cursor-pointer",
                  index === activeSegmentIndex
                    ? "bg-primary/10 border-l-2 border-primary"
                    : segment.startTime !== undefined ? "hover:bg-muted" : ""
                )}
                onClick={() => segment.startTime !== undefined && jumpToSegment(segment)}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-primary">
                    {getSpeakerName(segment.speaker)}
                  </span>
                  {segment.startTime !== undefined && (
                    <span className="text-xs text-muted-foreground">
                      {formatTime(segment.startTime)}
                    </span>
                  )}
                  {segment.emotion && (
                    <span className="text-xs text-muted-foreground italic">
                      ({segment.emotion})
                    </span>
                  )}
                </div>
                <p className="text-sm">{segment.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
