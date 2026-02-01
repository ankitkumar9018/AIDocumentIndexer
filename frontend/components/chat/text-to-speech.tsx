"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Volume2, VolumeX, Pause, Play, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface TextToSpeechProps {
  text: string;
  disabled?: boolean;
  className?: string;
  size?: "default" | "sm" | "lg" | "icon";
  autoPlay?: boolean;  // Automatically start speaking when text changes
  onComplete?: () => void;  // Callback when speech finishes
  onStart?: () => void;  // Callback when speech starts
}

// Check if Web Speech Synthesis is available
const isSpeechSynthesisSupported = () => {
  if (typeof window === "undefined") return false;
  return "speechSynthesis" in window;
};

export function TextToSpeech({
  text,
  disabled = false,
  className,
  size = "icon",
  autoPlay = false,
  onComplete,
  onStart,
}: TextToSpeechProps) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Track mounted state to prevent hydration mismatch
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isSpeaking) {
        window.speechSynthesis.cancel();
      }
    };
  }, [isSpeaking]);

  // Handle speech synthesis events
  const speak = useCallback(() => {
    if (!isSpeechSynthesisSupported()) {
      toast.error("Text-to-speech not supported", {
        description: "Your browser doesn't support speech synthesis.",
      });
      return;
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    if (!text.trim()) {
      toast.info("No text to speak", {
        description: "The response is empty.",
      });
      return;
    }

    setIsLoading(true);

    // Create utterance
    const utterance = new SpeechSynthesisUtterance(text);

    // Get available voices and select a good one
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(
      (voice) =>
        voice.lang.startsWith("en") &&
        (voice.name.includes("Google") ||
          voice.name.includes("Natural") ||
          voice.name.includes("Samantha") ||
          voice.name.includes("Daniel"))
    ) || voices.find((voice) => voice.lang.startsWith("en")) || voices[0];

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    utterance.onstart = () => {
      setIsLoading(false);
      setIsSpeaking(true);
      setIsPaused(false);
      onStart?.();
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      setIsPaused(false);
      onComplete?.();
    };

    utterance.onerror = (event) => {
      console.error("Speech synthesis error:", event);
      setIsLoading(false);
      setIsSpeaking(false);
      setIsPaused(false);
      if (event.error !== "canceled") {
        toast.error("Speech synthesis error", {
          description: `Error: ${event.error}`,
        });
      }
    };

    utteranceRef.current = utterance;

    // Voices may not be loaded immediately, wait if needed
    if (voices.length === 0) {
      window.speechSynthesis.onvoiceschanged = () => {
        const newVoices = window.speechSynthesis.getVoices();
        const voice = newVoices.find((v) => v.lang.startsWith("en"));
        if (voice) {
          utterance.voice = voice;
        }
        window.speechSynthesis.speak(utterance);
      };
    } else {
      window.speechSynthesis.speak(utterance);
    }
  }, [text, onStart, onComplete]);

  // Auto-play effect: start speaking when text changes and autoPlay is enabled
  const lastTextRef = useRef<string>("");
  useEffect(() => {
    if (autoPlay && text && text !== lastTextRef.current && !isSpeaking && !disabled) {
      lastTextRef.current = text;
      speak();
    }
  }, [autoPlay, text, isSpeaking, disabled, speak]);

  const stop = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    setIsPaused(false);
  }, []);

  const togglePause = useCallback(() => {
    if (isPaused) {
      window.speechSynthesis.resume();
      setIsPaused(false);
    } else {
      window.speechSynthesis.pause();
      setIsPaused(true);
    }
  }, [isPaused]);

  const handleClick = useCallback(() => {
    if (isSpeaking) {
      if (isPaused) {
        togglePause();
      } else {
        stop();
      }
    } else {
      speak();
    }
  }, [isSpeaking, isPaused, speak, stop, togglePause]);

  // Prevent hydration mismatch by not rendering until mounted
  // (server returns false for isSpeechSynthesisSupported, client returns true)
  if (!isMounted || !isSpeechSynthesisSupported()) {
    return null;
  }

  const getIcon = () => {
    if (isLoading) {
      return <Loader2 className="h-4 w-4 animate-spin" />;
    }
    if (isSpeaking && !isPaused) {
      return <VolumeX className="h-4 w-4" />;
    }
    if (isSpeaking && isPaused) {
      return <Play className="h-4 w-4" />;
    }
    return <Volume2 className="h-4 w-4" />;
  };

  const getTitle = () => {
    if (isSpeaking && !isPaused) return "Stop speaking";
    if (isSpeaking && isPaused) return "Resume speaking";
    return "Read aloud";
  };

  return (
    <Button
      type="button"
      variant="ghost"
      size={size}
      onClick={handleClick}
      disabled={disabled || isLoading || !text.trim()}
      className={cn(
        "shrink-0 transition-all",
        isSpeaking && !isPaused && "text-primary",
        className
      )}
      title={getTitle()}
      aria-label={getTitle()}
    >
      {getIcon()}
    </Button>
  );
}
