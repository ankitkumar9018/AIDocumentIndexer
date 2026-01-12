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
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  Bookmark,
  BookmarkCheck,
  Clock,
  ListMusic,
  X,
  Send,
  Mic,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

interface InteractiveAudioPlayerProps {
  src: string;
  title?: string;
  subtitle?: string;
  transcript?: TranscriptSegment[];
  sections?: Section[];
  onEnded?: () => void;
  onAskQuestion?: (question: string, timestamp: number) => Promise<string>;
  className?: string;
}

interface TranscriptSegment {
  speaker: string;
  text: string;
  startTime: number;
  endTime: number;
}

interface Section {
  id: string;
  title: string;
  startTime: number;
  endTime: number;
  summary?: string;
}

interface Bookmark {
  id: string;
  timestamp: number;
  note: string;
  createdAt: Date;
}

interface QAItem {
  question: string;
  answer: string;
  timestamp: number;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function InteractiveAudioPlayer({
  src,
  title,
  subtitle,
  transcript,
  sections,
  onEnded,
  onAskQuestion,
  className,
}: InteractiveAudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState(-1);
  const [activeSectionIndex, setActiveSectionIndex] = useState(-1);

  // Interactive features state
  const [isSmartPaused, setIsSmartPaused] = useState(false);
  const [smartPauseQuestion, setSmartPauseQuestion] = useState("");
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [qaHistory, setQaHistory] = useState<QAItem[]>([]);
  const [isAskingQuestion, setIsAskingQuestion] = useState(false);
  const [showSummaryMode, setShowSummaryMode] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState("");

  // Section navigation
  const [showSections, setShowSections] = useState(false);

  // Auto-scroll transcript
  const transcriptRef = useRef<HTMLDivElement>(null);

  // Update current time and active segments
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);

      // Find active transcript segment
      if (transcript) {
        const index = transcript.findIndex(
          (seg) =>
            audio.currentTime >= seg.startTime &&
            audio.currentTime < seg.endTime
        );
        setActiveSegmentIndex(index);
      }

      // Find active section
      if (sections) {
        const sectionIndex = sections.findIndex(
          (sec) =>
            audio.currentTime >= sec.startTime &&
            audio.currentTime < sec.endTime
        );
        setActiveSectionIndex(sectionIndex);
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
  }, [transcript, sections, onEnded]);

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeSegmentIndex >= 0 && transcriptRef.current) {
      const activeElement = transcriptRef.current.querySelector(
        `[data-segment="${activeSegmentIndex}"]`
      );
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [activeSegmentIndex]);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isSmartPaused) {
      setIsSmartPaused(false);
      setSmartPauseQuestion("");
    }

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying, isSmartPaused]);

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

  // Smart Pause - pause and ask a question
  const handleSmartPause = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.pause();
    setIsPlaying(false);
    setIsSmartPaused(true);
  }, []);

  // Instant Replay - replay last 10 seconds
  const instantReplay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    audio.currentTime = Math.max(0, audio.currentTime - 10);
    if (!isPlaying) {
      audio.play();
      setIsPlaying(true);
    }
  }, [isPlaying]);

  // Jump to section
  const jumpToSection = useCallback(
    (section: Section) => {
      const audio = audioRef.current;
      if (!audio) return;

      audio.currentTime = section.startTime;
      if (!isPlaying) {
        audio.play();
        setIsPlaying(true);
      }
    },
    [isPlaying]
  );

  // Navigate between sections
  const navigateSection = useCallback(
    (direction: "prev" | "next") => {
      if (!sections) return;

      let targetIndex: number;
      if (direction === "prev") {
        targetIndex = Math.max(0, activeSectionIndex - 1);
      } else {
        targetIndex = Math.min(sections.length - 1, activeSectionIndex + 1);
      }

      if (targetIndex !== activeSectionIndex) {
        jumpToSection(sections[targetIndex]);
      }
    },
    [sections, activeSectionIndex, jumpToSection]
  );

  // Jump to transcript segment
  const jumpToSegment = useCallback(
    (segment: TranscriptSegment) => {
      const audio = audioRef.current;
      if (!audio) return;

      audio.currentTime = segment.startTime;
      if (!isPlaying) {
        audio.play();
        setIsPlaying(true);
      }
    },
    [isPlaying]
  );

  // Add bookmark
  const addBookmark = useCallback(() => {
    const newBookmark: Bookmark = {
      id: Date.now().toString(),
      timestamp: currentTime,
      note: `Bookmark at ${formatTime(currentTime)}`,
      createdAt: new Date(),
    };
    setBookmarks((prev) => [...prev, newBookmark]);
  }, [currentTime]);

  // Remove bookmark
  const removeBookmark = useCallback((id: string) => {
    setBookmarks((prev) => prev.filter((b) => b.id !== id));
  }, []);

  // Jump to bookmark
  const jumpToBookmark = useCallback(
    (bookmark: Bookmark) => {
      const audio = audioRef.current;
      if (!audio) return;

      audio.currentTime = bookmark.timestamp;
      if (!isPlaying) {
        audio.play();
        setIsPlaying(true);
      }
    },
    [isPlaying]
  );

  // Handle asking a question
  const handleAskQuestion = useCallback(async () => {
    if (!currentQuestion.trim() || !onAskQuestion) return;

    setIsAskingQuestion(true);
    try {
      const answer = await onAskQuestion(currentQuestion, currentTime);
      setQaHistory((prev) => [
        ...prev,
        {
          question: currentQuestion,
          answer,
          timestamp: currentTime,
        },
      ]);
      setCurrentQuestion("");

      // Resume playback if was smart paused
      if (isSmartPaused) {
        setIsSmartPaused(false);
        const audio = audioRef.current;
        if (audio) {
          audio.play();
          setIsPlaying(true);
        }
      }
    } catch (error) {
      console.error("Failed to get answer:", error);
    } finally {
      setIsAskingQuestion(false);
    }
  }, [currentQuestion, currentTime, onAskQuestion, isSmartPaused]);

  // Get summary of current position
  const getCurrentSummary = useCallback(() => {
    if (!sections || activeSectionIndex < 0) return null;
    return sections[activeSectionIndex]?.summary;
  }, [sections, activeSectionIndex]);

  const hasBookmarkAtCurrentTime = bookmarks.some(
    (b) => Math.abs(b.timestamp - currentTime) < 1
  );

  return (
    <div className={cn("space-y-4", className)}>
      {/* Hidden audio element */}
      <audio ref={audioRef} src={src} preload="metadata" />

      {/* Main player card */}
      <div className="bg-card border rounded-lg p-4 space-y-4">
        {/* Title and section info */}
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            {title && <h3 className="font-semibold text-lg">{title}</h3>}
            {subtitle && (
              <p className="text-sm text-muted-foreground">{subtitle}</p>
            )}
            {sections && activeSectionIndex >= 0 && (
              <Badge variant="secondary" className="mt-1">
                {sections[activeSectionIndex]?.title}
              </Badge>
            )}
          </div>

          {/* Summary mode toggle */}
          {sections && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={showSummaryMode ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowSummaryMode(!showSummaryMode)}
                  >
                    <FileText className="h-4 w-4 mr-1" />
                    Summary
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Toggle summary mode</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>

        {/* Summary display */}
        {showSummaryMode && getCurrentSummary() && (
          <div className="bg-muted/50 rounded-lg p-3 text-sm">
            <p className="text-muted-foreground">{getCurrentSummary()}</p>
          </div>
        )}

        {/* Section navigation */}
        {sections && sections.length > 0 && (
          <div className="flex items-center justify-center gap-2 py-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigateSection("prev")}
              disabled={activeSectionIndex <= 0}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm text-muted-foreground min-w-[120px] text-center">
              Section {activeSectionIndex + 1} of {sections.length}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigateSection("next")}
              disabled={activeSectionIndex >= sections.length - 1}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Progress bar */}
        <div className="space-y-2">
          {/* Section markers on progress bar */}
          <div className="relative">
            <Slider
              value={[currentTime]}
              max={duration || 100}
              step={0.1}
              onValueChange={handleSeek}
              className="cursor-pointer"
            />
            {/* Section markers */}
            {sections && duration > 0 && (
              <div className="absolute top-0 left-0 right-0 h-full pointer-events-none">
                {sections.map((section, i) => (
                  <div
                    key={section.id}
                    className="absolute h-2 w-0.5 bg-primary/50 -top-1"
                    style={{ left: `${(section.startTime / duration) * 100}%` }}
                    title={section.title}
                  />
                ))}
              </div>
            )}
            {/* Bookmark markers */}
            {bookmarks.length > 0 && duration > 0 && (
              <div className="absolute top-0 left-0 right-0 h-full pointer-events-none">
                {bookmarks.map((bookmark) => (
                  <div
                    key={bookmark.id}
                    className="absolute h-2 w-2 bg-yellow-500 rounded-full -top-1"
                    style={{
                      left: `${(bookmark.timestamp / duration) * 100}%`,
                    }}
                  />
                ))}
              </div>
            )}
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Main Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
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

            {/* Instant Replay */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" onClick={instantReplay}>
                    <SkipBack className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Replay last 10s</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Play/Pause */}
            <Button
              size="icon"
              className="h-12 w-12 rounded-full mx-2"
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

          <div className="flex items-center gap-1">
            {/* Smart Pause - Ask a question */}
            {onAskQuestion && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={isSmartPaused ? "default" : "ghost"}
                      size="icon"
                      onClick={handleSmartPause}
                    >
                      <MessageSquare className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Smart Pause - Ask a question</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}

            {/* Bookmark */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={addBookmark}
                    disabled={hasBookmarkAtCurrentTime}
                  >
                    {hasBookmarkAtCurrentTime ? (
                      <BookmarkCheck className="h-4 w-4 text-yellow-500" />
                    ) : (
                      <Bookmark className="h-4 w-4" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Add bookmark</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {/* Section list */}
            {sections && (
              <Sheet open={showSections} onOpenChange={setShowSections}>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <ListMusic className="h-4 w-4" />
                  </Button>
                </SheetTrigger>
                <SheetContent>
                  <SheetHeader>
                    <SheetTitle>Sections</SheetTitle>
                  </SheetHeader>
                  <ScrollArea className="h-[calc(100vh-8rem)] mt-4">
                    <div className="space-y-2 pr-4">
                      {sections.map((section, index) => (
                        <div
                          key={section.id}
                          className={cn(
                            "p-3 rounded-lg cursor-pointer transition-colors",
                            index === activeSectionIndex
                              ? "bg-primary/10 border-l-2 border-primary"
                              : "hover:bg-muted"
                          )}
                          onClick={() => {
                            jumpToSection(section);
                            setShowSections(false);
                          }}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium">{section.title}</span>
                            <span className="text-xs text-muted-foreground">
                              {formatTime(section.startTime)}
                            </span>
                          </div>
                          {section.summary && (
                            <p className="text-sm text-muted-foreground line-clamp-2">
                              {section.summary}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </SheetContent>
              </Sheet>
            )}

            {/* Volume */}
            <div className="flex items-center gap-1">
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
                className="w-16"
              />
            </div>

            {/* Download */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" asChild>
                    <a href={src} download>
                      <Download className="h-4 w-4" />
                    </a>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Download</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </div>

      {/* Smart Pause Question Input */}
      {isSmartPaused && onAskQuestion && (
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4 text-primary" />
            <span className="font-medium">Ask a question about the content</span>
            <Button
              variant="ghost"
              size="icon"
              className="ml-auto h-6 w-6"
              onClick={() => {
                setIsSmartPaused(false);
                setCurrentQuestion("");
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex gap-2">
            <Input
              placeholder="Type your question..."
              value={currentQuestion}
              onChange={(e) => setCurrentQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleAskQuestion();
              }}
              disabled={isAskingQuestion}
            />
            <Button
              onClick={handleAskQuestion}
              disabled={!currentQuestion.trim() || isAskingQuestion}
            >
              {isAskingQuestion ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Paused at {formatTime(currentTime)}. Playback will resume after you
            ask your question.
          </p>
        </div>
      )}

      {/* Transcript, Bookmarks, Q&A Tabs */}
      {(transcript || bookmarks.length > 0 || qaHistory.length > 0) && (
        <div className="bg-card border rounded-lg p-4">
          <Tabs defaultValue="transcript">
            <TabsList className="mb-3">
              {transcript && (
                <TabsTrigger value="transcript">Transcript</TabsTrigger>
              )}
              {bookmarks.length > 0 && (
                <TabsTrigger value="bookmarks">
                  Bookmarks ({bookmarks.length})
                </TabsTrigger>
              )}
              {qaHistory.length > 0 && (
                <TabsTrigger value="qa">Q&A ({qaHistory.length})</TabsTrigger>
              )}
            </TabsList>

            {/* Transcript Tab */}
            {transcript && (
              <TabsContent value="transcript">
                <ScrollArea className="h-64" ref={transcriptRef}>
                  <div className="space-y-2 pr-4">
                    {transcript.map((segment, index) => (
                      <div
                        key={index}
                        data-segment={index}
                        className={cn(
                          "p-2 rounded cursor-pointer transition-colors",
                          index === activeSegmentIndex
                            ? "bg-primary/10 border-l-2 border-primary"
                            : "hover:bg-muted"
                        )}
                        onClick={() => jumpToSegment(segment)}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-medium text-primary">
                            {segment.speaker}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {formatTime(segment.startTime)}
                          </span>
                        </div>
                        <p className="text-sm">{segment.text}</p>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </TabsContent>
            )}

            {/* Bookmarks Tab */}
            {bookmarks.length > 0 && (
              <TabsContent value="bookmarks">
                <ScrollArea className="h-64">
                  <div className="space-y-2 pr-4">
                    {bookmarks
                      .sort((a, b) => a.timestamp - b.timestamp)
                      .map((bookmark) => (
                        <div
                          key={bookmark.id}
                          className="flex items-center justify-between p-2 rounded hover:bg-muted"
                        >
                          <button
                            className="flex items-center gap-2 text-left"
                            onClick={() => jumpToBookmark(bookmark)}
                          >
                            <Clock className="h-4 w-4 text-yellow-500" />
                            <span className="text-sm font-medium">
                              {formatTime(bookmark.timestamp)}
                            </span>
                            <span className="text-sm text-muted-foreground">
                              {bookmark.note}
                            </span>
                          </button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => removeBookmark(bookmark.id)}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      ))}
                  </div>
                </ScrollArea>
              </TabsContent>
            )}

            {/* Q&A History Tab */}
            {qaHistory.length > 0 && (
              <TabsContent value="qa">
                <ScrollArea className="h-64">
                  <div className="space-y-4 pr-4">
                    {qaHistory.map((qa, index) => (
                      <div key={index} className="space-y-2">
                        <div className="flex items-start gap-2">
                          <MessageSquare className="h-4 w-4 mt-0.5 text-primary" />
                          <div>
                            <p className="text-sm font-medium">{qa.question}</p>
                            <span className="text-xs text-muted-foreground">
                              at {formatTime(qa.timestamp)}
                            </span>
                          </div>
                        </div>
                        <div className="pl-6 text-sm text-muted-foreground">
                          {qa.answer}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </TabsContent>
            )}
          </Tabs>
        </div>
      )}
    </div>
  );
}
