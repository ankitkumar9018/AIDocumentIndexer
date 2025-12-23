"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Mic, MicOff, Loader2, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface VoiceInputProps {
  onTranscript: (text: string) => void;
  disabled?: boolean;
  className?: string;
}

// Check if Web Speech API is available
const isSpeechRecognitionSupported = () => {
  if (typeof window === "undefined") return false;
  return "webkitSpeechRecognition" in window || "SpeechRecognition" in window;
};

export function VoiceInput({
  onTranscript,
  disabled = false,
  className,
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    if (!isSpeechRecognitionSupported()) {
      toast.error("Voice input not supported", {
        description: "Your browser doesn't support speech recognition. Try Chrome or Edge.",
      });
      return;
    }

    try {
      // Request microphone permission
      await navigator.mediaDevices.getUserMedia({ audio: true });

      // Create speech recognition instance
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
      const recognition = new SpeechRecognition();

      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      let finalTranscript = "";
      let interimTranscript = "";

      recognition.onstart = () => {
        setIsRecording(true);
        setRecordingTime(0);
        // Start timer
        timerRef.current = setInterval(() => {
          setRecordingTime((prev) => prev + 1);
        }, 1000);
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        interimTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + " ";
          } else {
            interimTranscript += transcript;
          }
        }
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error("Speech recognition error:", event.error);
        if (event.error === "not-allowed") {
          toast.error("Microphone access denied", {
            description: "Please allow microphone access to use voice input.",
          });
        } else if (event.error !== "aborted") {
          toast.error("Voice recognition error", {
            description: `Error: ${event.error}`,
          });
        }
        stopRecording();
      };

      recognition.onend = () => {
        setIsRecording(false);
        setIsProcessing(false);
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        setRecordingTime(0);

        const transcript = finalTranscript.trim();
        if (transcript) {
          onTranscript(transcript);
        }
      };

      recognition.start();
      recognitionRef.current = recognition;
    } catch (error) {
      console.error("Failed to start recording:", error);
      toast.error("Failed to start recording", {
        description: "Please check your microphone permissions.",
      });
    }
  }, [onTranscript]);

  const stopRecording = useCallback(() => {
    if (recognitionRef.current) {
      setIsProcessing(true);
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // Format recording time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  if (!isSpeechRecognitionSupported()) {
    return null; // Don't show button if not supported
  }

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {isRecording && (
        <div className="flex items-center gap-2 text-sm text-destructive animate-pulse">
          <div className="h-2 w-2 rounded-full bg-destructive" />
          <span>{formatTime(recordingTime)}</span>
        </div>
      )}
      <Button
        type="button"
        variant={isRecording ? "destructive" : "outline"}
        size="icon"
        onClick={toggleRecording}
        disabled={disabled || isProcessing}
        className={cn(
          "shrink-0 transition-all",
          isRecording && "animate-pulse"
        )}
        title={isRecording ? "Stop recording" : "Start voice input"}
        aria-label={isRecording ? "Stop voice recording" : "Start voice recording"}
      >
        {isProcessing ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : isRecording ? (
          <Square className="h-4 w-4" />
        ) : (
          <Mic className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}

// TypeScript declarations for Web Speech API
declare global {
  interface Window {
    webkitSpeechRecognition: typeof SpeechRecognition;
    SpeechRecognition: typeof SpeechRecognition;
  }
}
