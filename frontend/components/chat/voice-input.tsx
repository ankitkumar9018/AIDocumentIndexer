"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Mic, MicOff, Loader2, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

// Web Speech API types (not available in all TypeScript libs)
interface SpeechRecognitionResult {
  readonly isFinal: boolean;
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  readonly transcript: string;
  readonly confidence: number;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  readonly error: string;
  readonly message: string;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: ((this: SpeechRecognitionInstance, ev: Event) => void) | null;
  onend: ((this: SpeechRecognitionInstance, ev: Event) => void) | null;
  onerror: ((this: SpeechRecognitionInstance, ev: SpeechRecognitionErrorEvent) => void) | null;
  onresult: ((this: SpeechRecognitionInstance, ev: SpeechRecognitionEvent) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

interface SpeechRecognitionConstructor {
  new (): SpeechRecognitionInstance;
}

interface VoiceInputProps {
  onTranscript: (text: string) => void;
  disabled?: boolean;
  className?: string;
  continuousMode?: boolean;  // Keep listening after each transcript
  autoSend?: boolean;  // Automatically send after silence (when continuousMode is true)
  onAutoSend?: () => void;  // Callback when auto-send is triggered
  silenceTimeout?: number;  // Milliseconds of silence before auto-send (default: 1500)
  onListeningStart?: () => void;  // Callback when listening starts
  onListeningEnd?: () => void;  // Callback when listening ends
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
  continuousMode = false,
  autoSend = false,
  onAutoSend,
  silenceTimeout = 1500,
  onListeningStart,
  onListeningEnd,
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  // Track support status in state to avoid hydration mismatch
  // Initialize to false - will be updated after mount on client
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const pendingTranscriptRef = useRef<string>("");

  // Check for speech recognition support after mount (avoids hydration mismatch)
  useEffect(() => {
    setIsSupported(isSpeechRecognitionSupported());
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
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
      const SpeechRecognitionAPI = (window as Window & { webkitSpeechRecognition?: SpeechRecognitionConstructor; SpeechRecognition?: SpeechRecognitionConstructor }).webkitSpeechRecognition || (window as Window & { SpeechRecognition?: SpeechRecognitionConstructor }).SpeechRecognition;
      if (!SpeechRecognitionAPI) return;
      const recognition = new SpeechRecognitionAPI();

      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      let finalTranscript = "";
      let interimTranscript = "";

      recognition.onstart = () => {
        setIsRecording(true);
        setRecordingTime(0);
        onListeningStart?.();
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

        // Store current transcript for auto-send
        pendingTranscriptRef.current = finalTranscript.trim();

        // Reset silence timer on new speech
        if (autoSend && continuousMode && finalTranscript.trim()) {
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
          }
          silenceTimerRef.current = setTimeout(() => {
            const transcript = pendingTranscriptRef.current;
            if (transcript) {
              onTranscript(transcript);
              pendingTranscriptRef.current = "";
              finalTranscript = "";
              onAutoSend?.();
            }
          }, silenceTimeout);
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
        onListeningEnd?.();
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
        setRecordingTime(0);

        // Only send transcript if not in continuous mode (or if stopping manually)
        // In continuous mode with autoSend, the silence timer handles sending
        const transcript = finalTranscript.trim();
        if (transcript && !continuousMode) {
          onTranscript(transcript);
        } else if (transcript && continuousMode && !autoSend) {
          // In continuous mode without autoSend, send on manual stop
          onTranscript(transcript);
        }
        pendingTranscriptRef.current = "";
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

  // Don't render until we've checked support (after hydration)
  // This prevents hydration mismatch between server and client
  if (!isSupported) {
    return null;
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

