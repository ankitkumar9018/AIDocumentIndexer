"use client";

/**
 * AIDocumentIndexer - Onboarding Flow
 * ====================================
 *
 * Phase 19: Zero-friction user experience with progressive disclosure.
 *
 * Key Principles:
 * - Landing: ONE clear action (drop files or connect Drive)
 * - Processing starts IMMEDIATELY on drop (no submit button)
 * - Show extracted text as it's parsed
 * - Enable queries before embedding completes
 *
 * Features:
 * - Welcome wizard for first-time users
 * - Quick-start templates
 * - Progressive feature discovery
 * - Instant gratification (show results while processing)
 * - One-click audio overview
 */

import * as React from "react";
import { useState, useCallback, useEffect } from "react";
import {
  ArrowRight,
  CheckCircle2,
  Cloud,
  FileText,
  FolderOpen,
  Headphones,
  MessageSquare,
  Mic,
  Search,
  Sparkles,
  Upload,
  Wand2,
  X,
  Zap,
  BookOpen,
  BarChart3,
  Globe,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

// =============================================================================
// Types
// =============================================================================

export type OnboardingStep =
  | "welcome"
  | "choose-source"
  | "upload"
  | "processing"
  | "ready"
  | "complete";

interface OnboardingState {
  step: OnboardingStep;
  hasCompletedBefore: boolean;
  documentsUploaded: number;
  documentsProcessed: number;
  selectedSource: "upload" | "google-drive" | "url" | null;
}

interface OnboardingFlowProps {
  onComplete: () => void;
  onSkip: () => void;
  className?: string;
}

// =============================================================================
// Animation Variants
// =============================================================================

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

const scaleIn = {
  initial: { scale: 0.9, opacity: 0 },
  animate: { scale: 1, opacity: 1 },
  exit: { scale: 0.9, opacity: 0 },
};

// =============================================================================
// Welcome Step
// =============================================================================

function WelcomeStep({ onNext }: { onNext: () => void }) {
  return (
    <motion.div {...fadeIn} className="text-center max-w-2xl mx-auto">
      <div className="flex justify-center mb-6">
        <div className="relative">
          <div className="h-20 w-20 rounded-2xl bg-primary/10 flex items-center justify-center">
            <Sparkles className="h-10 w-10 text-primary" />
          </div>
          <motion.div
            className="absolute -top-1 -right-1 h-6 w-6 rounded-full bg-green-500 flex items-center justify-center"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Zap className="h-3 w-3 text-white" />
          </motion.div>
        </div>
      </div>

      <h1 className="text-3xl font-bold tracking-tight mb-3">
        Welcome to AIDocumentIndexer
      </h1>
      <p className="text-lg text-muted-foreground mb-8">
        Transform your documents into intelligent, searchable knowledge. Ask questions, get answers, and generate insights in seconds.
      </p>

      <div className="grid grid-cols-3 gap-4 mb-8">
        <FeatureHighlight
          icon={<Search className="h-5 w-5" />}
          title="Smart Search"
          description="Find answers instantly"
        />
        <FeatureHighlight
          icon={<MessageSquare className="h-5 w-5" />}
          title="Chat with Docs"
          description="Natural conversations"
        />
        <FeatureHighlight
          icon={<Headphones className="h-5 w-5" />}
          title="Audio Overview"
          description="Listen to summaries"
        />
      </div>

      <Button size="lg" onClick={onNext} className="gap-2">
        Get Started
        <ArrowRight className="h-4 w-4" />
      </Button>
    </motion.div>
  );
}

function FeatureHighlight({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="text-center p-4 rounded-lg bg-muted/50">
      <div className="flex justify-center mb-2 text-primary">{icon}</div>
      <p className="font-medium text-sm">{title}</p>
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  );
}

// =============================================================================
// Choose Source Step
// =============================================================================

function ChooseSourceStep({
  onSelect,
}: {
  onSelect: (source: "upload" | "google-drive" | "url") => void;
}) {
  return (
    <motion.div {...fadeIn} className="max-w-3xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold mb-2">Add Your Documents</h2>
        <p className="text-muted-foreground">
          Choose how you want to import your documents
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <SourceCard
          icon={<Upload className="h-8 w-8" />}
          title="Upload Files"
          description="Drag & drop or browse your computer"
          badge="Instant"
          onClick={() => onSelect("upload")}
          recommended
        />
        <SourceCard
          icon={<Cloud className="h-8 w-8" />}
          title="Google Drive"
          description="Connect your Drive account"
          badge="Sync"
          onClick={() => onSelect("google-drive")}
        />
        <SourceCard
          icon={<Globe className="h-8 w-8" />}
          title="Web URL"
          description="Import from any webpage"
          badge="Quick"
          onClick={() => onSelect("url")}
        />
      </div>
    </motion.div>
  );
}

function SourceCard({
  icon,
  title,
  description,
  badge,
  onClick,
  recommended,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  badge: string;
  onClick: () => void;
  recommended?: boolean;
}) {
  return (
    <Card
      className={cn(
        "relative cursor-pointer transition-all hover:border-primary hover:shadow-md",
        recommended && "border-primary"
      )}
      onClick={onClick}
    >
      {recommended && (
        <Badge className="absolute -top-2 -right-2" variant="default">
          Recommended
        </Badge>
      )}
      <CardContent className="pt-6 text-center">
        <div className="flex justify-center mb-4 text-primary">{icon}</div>
        <h3 className="font-semibold mb-1">{title}</h3>
        <p className="text-sm text-muted-foreground mb-3">{description}</p>
        <Badge variant="secondary">{badge}</Badge>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Upload Step (Immediate Processing)
// =============================================================================

function UploadStep({
  onFilesAdded,
  isProcessing,
}: {
  onFilesAdded: (files: File[]) => void;
  isProcessing: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        onFilesAdded(files);
      }
    },
    [onFilesAdded]
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      onFilesAdded(files);
    }
  };

  return (
    <motion.div {...fadeIn} className="max-w-2xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold mb-2">Drop Your Files</h2>
        <p className="text-muted-foreground">
          Processing starts immediately - no waiting!
        </p>
      </div>

      <div
        className={cn(
          "relative border-2 border-dashed rounded-xl p-12 text-center transition-all",
          isDragging
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-muted-foreground/25 hover:border-primary/50",
          isProcessing && "pointer-events-none opacity-60"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          accept=".pdf,.doc,.docx,.txt,.md,.ppt,.pptx,.xls,.xlsx,.csv"
        />

        <motion.div
          animate={isDragging ? { scale: 1.1 } : { scale: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center">
            <Upload className="h-8 w-8 text-primary" />
          </div>
          <div>
            <p className="text-lg font-medium">
              {isDragging ? "Drop files here!" : "Drag & drop your files"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              or click to browse
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-2 mt-2">
            <Badge variant="outline">PDF</Badge>
            <Badge variant="outline">Word</Badge>
            <Badge variant="outline">PowerPoint</Badge>
            <Badge variant="outline">Excel</Badge>
            <Badge variant="outline">Text</Badge>
          </div>
        </motion.div>
      </div>

      <p className="text-center text-xs text-muted-foreground mt-4">
        Supports PDF, Word, PowerPoint, Excel, and plain text files up to 100MB each
      </p>
    </motion.div>
  );
}

// =============================================================================
// Processing Step (Instant Gratification)
// =============================================================================

function ProcessingStep({
  documentsUploaded,
  documentsProcessed,
  onTryQuery,
}: {
  documentsUploaded: number;
  documentsProcessed: number;
  onTryQuery: () => void;
}) {
  const progress = documentsUploaded > 0
    ? Math.round((documentsProcessed / documentsUploaded) * 100)
    : 0;

  const canQuery = documentsProcessed > 0;

  return (
    <motion.div {...fadeIn} className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="inline-flex h-16 w-16 rounded-2xl bg-primary/10 items-center justify-center mb-4"
        >
          <Wand2 className="h-8 w-8 text-primary" />
        </motion.div>
        <h2 className="text-2xl font-bold mb-2">Processing Your Documents</h2>
        <p className="text-muted-foreground">
          {canQuery
            ? "You can start asking questions while we finish processing!"
            : "Please wait while we analyze your documents..."}
        </p>
      </div>

      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span>Processing documents...</span>
              <span className="text-muted-foreground">
                {documentsProcessed} / {documentsUploaded}
              </span>
            </div>
            <Progress value={progress} className="h-2" />

            <div className="grid grid-cols-3 gap-4 pt-4">
              <ProcessingStage
                icon={<FileText className="h-4 w-4" />}
                label="Extracting Text"
                done={documentsProcessed > 0}
              />
              <ProcessingStage
                icon={<Search className="h-4 w-4" />}
                label="Building Index"
                done={documentsProcessed >= documentsUploaded / 2}
              />
              <ProcessingStage
                icon={<Sparkles className="h-4 w-4" />}
                label="Generating Insights"
                done={documentsProcessed >= documentsUploaded}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Instant Gratification: Allow queries even while processing */}
      {canQuery && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 text-green-600 mb-4">
            <CheckCircle2 className="h-4 w-4" />
            <span className="text-sm font-medium">
              {documentsProcessed} document{documentsProcessed !== 1 ? "s" : ""} ready!
            </span>
          </div>

          <Button size="lg" onClick={onTryQuery} className="gap-2">
            <MessageSquare className="h-4 w-4" />
            Try asking a question
          </Button>
        </motion.div>
      )}
    </motion.div>
  );
}

function ProcessingStage({
  icon,
  label,
  done,
}: {
  icon: React.ReactNode;
  label: string;
  done: boolean;
}) {
  return (
    <div className="text-center">
      <div
        className={cn(
          "inline-flex h-10 w-10 rounded-full items-center justify-center mb-2 transition-colors",
          done ? "bg-green-500/10 text-green-500" : "bg-muted text-muted-foreground"
        )}
      >
        {done ? <CheckCircle2 className="h-5 w-5" /> : icon}
      </div>
      <p className={cn("text-xs", done ? "text-foreground" : "text-muted-foreground")}>
        {label}
      </p>
    </div>
  );
}

// =============================================================================
// Ready Step (Feature Discovery)
// =============================================================================

function ReadyStep({ onComplete }: { onComplete: () => void }) {
  return (
    <motion.div {...fadeIn} className="max-w-3xl mx-auto">
      <div className="text-center mb-8">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", duration: 0.5 }}
          className="inline-flex h-16 w-16 rounded-2xl bg-green-500/10 items-center justify-center mb-4"
        >
          <CheckCircle2 className="h-8 w-8 text-green-500" />
        </motion.div>
        <h2 className="text-2xl font-bold mb-2">You're All Set!</h2>
        <p className="text-muted-foreground">
          Here's what you can do with your documents
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-4 mb-8">
        <FeatureCard
          icon={<MessageSquare className="h-6 w-6" />}
          title="Chat with Documents"
          description="Ask questions in natural language and get instant answers with sources"
          action="Start Chatting"
        />
        <FeatureCard
          icon={<Headphones className="h-6 w-6" />}
          title="Audio Overview"
          description="Generate a podcast-style summary of your documents"
          action="Generate Audio"
          badge="Popular"
        />
        <FeatureCard
          icon={<BookOpen className="h-6 w-6" />}
          title="Generate Reports"
          description="Create comprehensive reports from your document knowledge"
          action="Create Report"
        />
        <FeatureCard
          icon={<BarChart3 className="h-6 w-6" />}
          title="Knowledge Graph"
          description="Visualize connections between concepts in your documents"
          action="Explore Graph"
        />
      </div>

      <div className="text-center">
        <Button size="lg" onClick={onComplete} className="gap-2">
          <Sparkles className="h-4 w-4" />
          Start Exploring
        </Button>
      </div>
    </motion.div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
  action,
  badge,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  action: string;
  badge?: string;
}) {
  return (
    <Card className="relative overflow-hidden hover:shadow-md transition-shadow">
      {badge && (
        <Badge className="absolute top-3 right-3" variant="secondary">
          {badge}
        </Badge>
      )}
      <CardContent className="pt-6">
        <div className="flex items-start gap-4">
          <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center shrink-0 text-primary">
            {icon}
          </div>
          <div className="flex-1">
            <h3 className="font-semibold mb-1">{title}</h3>
            <p className="text-sm text-muted-foreground mb-3">{description}</p>
            <Button variant="outline" size="sm" className="gap-1">
              {action}
              <ArrowRight className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Main Onboarding Flow
// =============================================================================

export function OnboardingFlow({
  onComplete,
  onSkip,
  className,
}: OnboardingFlowProps) {
  const [state, setState] = useState<OnboardingState>({
    step: "welcome",
    hasCompletedBefore: false,
    documentsUploaded: 0,
    documentsProcessed: 0,
    selectedSource: null,
  });

  const goToStep = (step: OnboardingStep) => {
    setState((prev) => ({ ...prev, step }));
  };

  const handleSourceSelect = (source: "upload" | "google-drive" | "url") => {
    setState((prev) => ({ ...prev, selectedSource: source, step: "upload" }));
  };

  const handleFilesAdded = async (files: File[]) => {
    setState((prev) => ({
      ...prev,
      documentsUploaded: prev.documentsUploaded + files.length,
      step: "processing",
    }));

    // Simulate processing (in real app, this would be actual API calls)
    for (let i = 0; i < files.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setState((prev) => ({
        ...prev,
        documentsProcessed: prev.documentsProcessed + 1,
      }));
    }

    // Automatically go to ready step when done
    setTimeout(() => {
      goToStep("ready");
    }, 500);
  };

  const handleTryQuery = () => {
    goToStep("ready");
  };

  return (
    <div className={cn("min-h-[500px] flex flex-col", className)}>
      {/* Skip Button */}
      <div className="flex justify-end mb-4">
        <Button variant="ghost" size="sm" onClick={onSkip} className="text-muted-foreground">
          Skip for now
          <X className="h-4 w-4 ml-1" />
        </Button>
      </div>

      {/* Progress Indicator */}
      <div className="flex justify-center mb-8">
        <StepIndicator
          steps={["Welcome", "Source", "Upload", "Processing", "Ready"]}
          currentStep={
            state.step === "welcome"
              ? 0
              : state.step === "choose-source"
              ? 1
              : state.step === "upload"
              ? 2
              : state.step === "processing"
              ? 3
              : 4
          }
        />
      </div>

      {/* Step Content */}
      <div className="flex-1 flex items-center justify-center p-4">
        <AnimatePresence mode="wait">
          {state.step === "welcome" && (
            <WelcomeStep key="welcome" onNext={() => goToStep("choose-source")} />
          )}
          {state.step === "choose-source" && (
            <ChooseSourceStep key="source" onSelect={handleSourceSelect} />
          )}
          {state.step === "upload" && (
            <UploadStep
              key="upload"
              onFilesAdded={handleFilesAdded}
              isProcessing={state.documentsUploaded > 0}
            />
          )}
          {state.step === "processing" && (
            <ProcessingStep
              key="processing"
              documentsUploaded={state.documentsUploaded}
              documentsProcessed={state.documentsProcessed}
              onTryQuery={handleTryQuery}
            />
          )}
          {state.step === "ready" && (
            <ReadyStep key="ready" onComplete={onComplete} />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// =============================================================================
// Step Indicator
// =============================================================================

function StepIndicator({
  steps,
  currentStep,
}: {
  steps: string[];
  currentStep: number;
}) {
  return (
    <div className="flex items-center gap-2">
      {steps.map((step, index) => (
        <React.Fragment key={step}>
          <div
            className={cn(
              "h-2 w-2 rounded-full transition-all",
              index === currentStep
                ? "w-6 bg-primary"
                : index < currentStep
                ? "bg-primary/50"
                : "bg-muted"
            )}
          />
        </React.Fragment>
      ))}
    </div>
  );
}

// =============================================================================
// Quick Start Modal (for returning users)
// =============================================================================

export function QuickStartModal({
  open,
  onClose,
  onAction,
}: {
  open: boolean;
  onClose: () => void;
  onAction: (action: string) => void;
}) {
  if (!open) return null;

  const quickActions = [
    { id: "upload", icon: <Upload className="h-5 w-5" />, label: "Upload files" },
    { id: "chat", icon: <MessageSquare className="h-5 w-5" />, label: "Start chatting" },
    { id: "audio", icon: <Headphones className="h-5 w-5" />, label: "Audio overview" },
    { id: "search", icon: <Search className="h-5 w-5" />, label: "Search documents" },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-background rounded-xl p-6 max-w-md w-full mx-4 shadow-xl"
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Quick Start</h3>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {quickActions.map((action) => (
            <Button
              key={action.id}
              variant="outline"
              className="h-auto py-4 flex-col gap-2"
              onClick={() => {
                onAction(action.id);
                onClose();
              }}
            >
              {action.icon}
              <span className="text-sm">{action.label}</span>
            </Button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

export default OnboardingFlow;
