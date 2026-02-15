"use client";

import * as React from "react";
import {
  Settings,
  Users,
  Shield,
  FileText,
  ChevronRight,
  Save,
  RotateCcw,
  Download,
  Search,
  Plus,
  Trash2,
  Edit3,
  Check,
  X,
  AlertTriangle,
  Info,
  Eye,
  EyeOff,
  Filter,
  Calendar,
  Activity,
  BarChart3,
  Lock,
  Unlock,
  Key,
  Database,
  Cpu,
  Bell,
  Zap,
  Cloud,
  Server,
  RefreshCw,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";

// =============================================================================
// TYPES
// =============================================================================

type SettingCategory = "general" | "security" | "processing" | "integrations" | "notifications" | "advanced";
type SettingType = "string" | "number" | "boolean" | "select" | "multi_select" | "json" | "secret";

interface SettingDefinition {
  key: string;
  name: string;
  description: string;
  category: SettingCategory;
  type: SettingType;
  default_value: any;
  options?: Array<{ value: string; label: string }>;
  validation?: { min?: number; max?: number; max_length?: number };
  requires_restart: boolean;
  sensitive: boolean;
}

interface SettingValue {
  value: any;
  definition: SettingDefinition;
  is_default: boolean;
}

interface User {
  user_id: string;
  email?: string;
  role: string;
  permissions: string[];
  created_at?: string;
}

interface AuditLog {
  id: string;
  timestamp: string;
  action: string;
  user_id: string;
  org_id: string;
  resource_type: string;
  resource_id: string;
  ip_address?: string;
  details: Record<string, any>;
}

interface SystemStats {
  organization: {
    id: string;
    name: string;
    tier: string;
  };
  users: {
    total: number;
    by_role: Record<string, number>;
  };
  activity: {
    recent_actions: number;
    by_action: Record<string, number>;
  };
  settings: {
    custom_overrides: number;
    total_available: number;
  };
}

// =============================================================================
// API HELPER
// =============================================================================

import { api } from "@/lib/api";

async function fetchSettings(category?: string): Promise<Record<string, SettingValue>> {
  try {
    const endpoint = category
      ? `/admin/settings?category=${category}`
      : `/admin/settings`;
    const { data } = await api.get<Record<string, SettingValue>>(endpoint);
    return data;
  } catch (error) {
    console.error("Error fetching settings:", error);
    return {};
  }
}

async function updateSetting(key: string, value: any): Promise<boolean> {
  try {
    await api.put(`/admin/settings/${key}`, { value });
    return true;
  } catch (error) {
    console.error("Error updating setting:", error);
    return false;
  }
}

async function resetSetting(key: string): Promise<boolean> {
  try {
    await api.delete(`/admin/settings/${key}`);
    return true;
  } catch (error) {
    console.error("Error resetting setting:", error);
    return false;
  }
}

// =============================================================================
// COMPREHENSIVE SETTINGS DEFINITIONS
// =============================================================================

const ALL_SETTINGS: SettingDefinition[] = [
  // General Settings
  {
    key: "org_name",
    name: "Organization Name",
    description: "Display name for your organization",
    category: "general",
    type: "string",
    default_value: "",
    validation: { max_length: 100 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "environment",
    name: "Environment",
    description: "Application environment",
    category: "general",
    type: "select",
    default_value: "development",
    options: [
      { value: "development", label: "Development" },
      { value: "staging", label: "Staging" },
      { value: "production", label: "Production" },
    ],
    requires_restart: true,
    sensitive: false,
  },
  {
    key: "default_language",
    name: "Default Language",
    description: "Default language for the application",
    category: "general",
    type: "select",
    default_value: "en",
    options: [
      { value: "en", label: "English" },
      { value: "es", label: "Spanish" },
      { value: "fr", label: "French" },
      { value: "de", label: "German" },
      { value: "ja", label: "Japanese" },
    ],
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "timezone",
    name: "Timezone",
    description: "Default timezone for dates and times",
    category: "general",
    type: "select",
    default_value: "UTC",
    options: [
      { value: "UTC", label: "UTC" },
      { value: "America/New_York", label: "Eastern Time" },
      { value: "America/Los_Angeles", label: "Pacific Time" },
      { value: "Europe/London", label: "London" },
      { value: "Asia/Tokyo", label: "Tokyo" },
    ],
    requires_restart: false,
    sensitive: false,
  },

  // Security Settings
  {
    key: "session_timeout_minutes",
    name: "Session Timeout",
    description: "Minutes of inactivity before session expires",
    category: "security",
    type: "number",
    default_value: 60,
    validation: { min: 5, max: 1440 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "mfa_required",
    name: "Require MFA",
    description: "Require multi-factor authentication for all users",
    category: "security",
    type: "boolean",
    default_value: false,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "password_min_length",
    name: "Minimum Password Length",
    description: "Minimum characters required for passwords",
    category: "security",
    type: "number",
    default_value: 8,
    validation: { min: 6, max: 32 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "api_rate_limit",
    name: "API Rate Limit",
    description: "Maximum API requests per minute per user",
    category: "security",
    type: "number",
    default_value: 60,
    validation: { min: 10, max: 1000 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "secret_key",
    name: "Secret Key",
    description: "JWT signing key for authentication (requires restart)",
    category: "security",
    type: "secret",
    default_value: "",
    requires_restart: true,
    sensitive: true,
  },

  // Processing Settings - LLM
  {
    key: "default_llm_provider",
    name: "Default LLM Provider",
    description: "Primary LLM provider for chat and generation",
    category: "processing",
    type: "select",
    default_value: "openai",
    options: [
      { value: "openai", label: "OpenAI (GPT-4o)" },
      { value: "anthropic", label: "Anthropic (Claude)" },
      { value: "google", label: "Google (Gemini)" },
      { value: "groq", label: "Groq (Fast inference)" },
      { value: "together", label: "Together AI" },
      { value: "mistral", label: "Mistral AI" },
      { value: "deepseek", label: "DeepSeek" },
    ],
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "default_chat_model",
    name: "Default Chat Model",
    description: "Model to use for chat responses",
    category: "processing",
    type: "select",
    default_value: "gpt-4o",
    options: [
      { value: "gpt-4o", label: "GPT-4o (Best overall)" },
      { value: "gpt-4o-mini", label: "GPT-4o Mini (Faster)" },
      { value: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet" },
      { value: "claude-3-5-haiku-20241022", label: "Claude 3.5 Haiku (Fast)" },
      { value: "gemini-1.5-pro", label: "Gemini 1.5 Pro" },
      { value: "llama-3.1-70b", label: "Llama 3.1 70B" },
    ],
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "embedding_provider",
    name: "Embedding Provider",
    description: "Provider for document embeddings",
    category: "processing",
    type: "select",
    default_value: "openai",
    options: [
      { value: "openai", label: "OpenAI" },
      { value: "voyage", label: "Voyage AI (Best quality)" },
      { value: "cohere", label: "Cohere" },
      { value: "jina", label: "Jina AI" },
    ],
    requires_restart: true,
    sensitive: false,
  },
  {
    key: "embedding_model",
    name: "Embedding Model",
    description: "Model used for document embeddings",
    category: "processing",
    type: "select",
    default_value: "text-embedding-3-large",
    options: [
      { value: "text-embedding-3-large", label: "OpenAI Large (Best)" },
      { value: "text-embedding-3-small", label: "OpenAI Small (Faster)" },
      { value: "colbert-v2", label: "ColBERT v2 (Retrieval optimized)" },
    ],
    requires_restart: true,
    sensitive: false,
  },

  // Processing Settings - RAG
  {
    key: "rag_top_k",
    name: "RAG Top-K Results",
    description: "Number of documents to retrieve for each query",
    category: "processing",
    type: "number",
    default_value: 10,
    validation: { min: 1, max: 50 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "rag_similarity_threshold",
    name: "Similarity Threshold",
    description: "Minimum similarity score for retrieved documents (0-1)",
    category: "processing",
    type: "number",
    default_value: 0.55,
    validation: { min: 0, max: 1 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "enable_reranking",
    name: "Enable Reranking",
    description: "Use cross-encoder to rerank retrieved documents",
    category: "processing",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "enable_hybrid_search",
    name: "Enable Hybrid Search",
    description: "Combine dense and sparse (BM25) retrieval",
    category: "processing",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "enable_colbert",
    name: "Enable ColBERT Retrieval",
    description: "Use ColBERT PLAID for late interaction search",
    category: "processing",
    type: "boolean",
    default_value: false,
    requires_restart: false,
    sensitive: false,
  },

  // Processing Settings - TTS/Audio
  {
    key: "tts_provider",
    name: "TTS Provider",
    description: "Text-to-speech provider for audio generation",
    category: "processing",
    type: "select",
    default_value: "openai",
    options: [
      { value: "openai", label: "OpenAI TTS" },
      { value: "elevenlabs", label: "ElevenLabs (High quality)" },
      { value: "cartesia", label: "Cartesia (Ultra-fast)" },
    ],
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "default_voice",
    name: "Default Voice",
    description: "Default voice for audio generation",
    category: "processing",
    type: "select",
    default_value: "alloy",
    options: [
      { value: "alloy", label: "Alloy (Neutral)" },
      { value: "echo", label: "Echo (Male)" },
      { value: "fable", label: "Fable (British)" },
      { value: "onyx", label: "Onyx (Deep)" },
      { value: "nova", label: "Nova (Female)" },
      { value: "shimmer", label: "Shimmer (Warm)" },
    ],
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "enable_audio_overviews",
    name: "Enable Audio Overviews",
    description: "Allow generating audio summaries of documents",
    category: "processing",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },

  // Processing Settings - Document
  {
    key: "max_file_size_mb",
    name: "Max File Size (MB)",
    description: "Maximum file size allowed for uploads",
    category: "processing",
    type: "number",
    default_value: 100,
    validation: { min: 1, max: 500 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "concurrent_processing_limit",
    name: "Concurrent Processing Limit",
    description: "Maximum documents processed simultaneously",
    category: "processing",
    type: "number",
    default_value: 10,
    validation: { min: 1, max: 50 },
    requires_restart: true,
    sensitive: false,
  },
  {
    key: "ocr_engine",
    name: "OCR Engine",
    description: "Primary OCR engine for document processing",
    category: "processing",
    type: "select",
    default_value: "surya",
    options: [
      { value: "surya", label: "Surya (97.7% accuracy)" },
      { value: "tesseract", label: "Tesseract (Open Source)" },
      { value: "claude_vision", label: "Claude Vision (Best quality)" },
    ],
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "chunking_strategy",
    name: "Chunking Strategy",
    description: "How documents are split for indexing",
    category: "processing",
    type: "select",
    default_value: "semantic",
    options: [
      { value: "semantic", label: "Semantic (AI-powered)" },
      { value: "fixed", label: "Fixed Size" },
      { value: "sentence", label: "Sentence-based" },
      { value: "paragraph", label: "Paragraph-based" },
    ],
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "bulk_upload_max_concurrent",
    name: "Bulk Upload Concurrency",
    description: "Max concurrent documents in bulk upload",
    category: "processing",
    type: "number",
    default_value: 4,
    validation: { min: 1, max: 20 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "embedding_batch_size",
    name: "Embedding Batch Size",
    description: "Number of chunks to embed in a single batch",
    category: "processing",
    type: "number",
    default_value: 100,
    validation: { min: 10, max: 500 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "upload_dir",
    name: "Upload Directory",
    description: "Directory for uploaded files",
    category: "processing",
    type: "string",
    default_value: "./uploads",
    validation: { max_length: 255 },
    requires_restart: true,
    sensitive: false,
  },
  {
    key: "audio_output_dir",
    name: "Audio Output Directory",
    description: "Directory for generated audio files",
    category: "processing",
    type: "string",
    default_value: "./audio_output",
    validation: { max_length: 255 },
    requires_restart: true,
    sensitive: false,
  },

  // Integrations - API Keys (LLM)
  {
    key: "openai_api_key",
    name: "OpenAI API Key",
    description: "API key for OpenAI services (GPT, embeddings, TTS)",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "anthropic_api_key",
    name: "Anthropic API Key",
    description: "API key for Anthropic Claude models",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "google_api_key",
    name: "Google AI API Key",
    description: "API key for Google AI (Gemini) models",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "groq_api_key",
    name: "Groq API Key",
    description: "API key for Groq fast inference",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "mistral_api_key",
    name: "Mistral API Key",
    description: "API key for Mistral AI models",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "together_api_key",
    name: "Together AI API Key",
    description: "API key for Together AI inference",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "deepseek_api_key",
    name: "DeepSeek API Key",
    description: "API key for DeepSeek models",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },

  // Integrations - API Keys (Embeddings)
  {
    key: "voyage_api_key",
    name: "Voyage AI API Key",
    description: "API key for Voyage AI embeddings",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "cohere_api_key",
    name: "Cohere API Key",
    description: "API key for Cohere reranking and embeddings",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "jina_api_key",
    name: "Jina AI API Key",
    description: "API key for Jina AI embeddings",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },

  // Integrations - API Keys (TTS)
  {
    key: "elevenlabs_api_key",
    name: "ElevenLabs API Key",
    description: "API key for ElevenLabs TTS",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "cartesia_api_key",
    name: "Cartesia API Key",
    description: "API key for Cartesia ultra-fast TTS",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },

  // Integrations - API Keys (Other)
  {
    key: "firecrawl_api_key",
    name: "Firecrawl API Key",
    description: "API key for Firecrawl web scraping",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },

  // Integrations - OAuth
  {
    key: "google_drive_enabled",
    name: "Google Drive Integration",
    description: "Enable Google Drive document import",
    category: "integrations",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "google_oauth_client_id",
    name: "Google OAuth Client ID",
    description: "Client ID for Google OAuth (Drive integration)",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "google_oauth_client_secret",
    name: "Google OAuth Client Secret",
    description: "Client secret for Google OAuth",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "notion_integration_token",
    name: "Notion Integration Token",
    description: "Token for Notion workspace integration",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "slack_bot_token",
    name: "Slack Bot Token",
    description: "OAuth token for Slack bot",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "slack_signing_secret",
    name: "Slack Signing Secret",
    description: "Signing secret for Slack request verification",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },
  {
    key: "slack_webhook_url",
    name: "Slack Webhook URL",
    description: "Webhook for Slack notifications",
    category: "integrations",
    type: "secret",
    default_value: "",
    requires_restart: false,
    sensitive: true,
  },

  // Notifications
  {
    key: "email_notifications",
    name: "Email Notifications",
    description: "Enable email notifications",
    category: "notifications",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "notification_events",
    name: "Notification Events",
    description: "Events that trigger notifications",
    category: "notifications",
    type: "multi_select",
    default_value: ["processing_complete", "processing_error"],
    options: [
      { value: "processing_complete", label: "Processing Complete" },
      { value: "processing_error", label: "Processing Error" },
      { value: "user_invited", label: "User Invited" },
      { value: "storage_warning", label: "Storage Warning" },
      { value: "api_limit_warning", label: "API Limit Warning" },
    ],
    requires_restart: false,
    sensitive: false,
  },

  // Advanced - Feature Flags
  {
    key: "enable_knowledge_graph",
    name: "Enable Knowledge Graph",
    description: "Extract and use knowledge graphs from documents",
    category: "advanced",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "enable_query_expansion",
    name: "Enable Query Expansion",
    description: "Expand queries with synonyms and related terms",
    category: "advanced",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "enable_hyde",
    name: "Enable HyDE",
    description: "Use Hypothetical Document Embeddings for retrieval",
    category: "advanced",
    type: "boolean",
    default_value: true,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "enable_sufficiency_check",
    name: "Enable Sufficiency Detection",
    description: "Check if context is sufficient before answering",
    category: "advanced",
    type: "boolean",
    default_value: false,
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "sufficiency_threshold",
    name: "Sufficiency Threshold",
    description: "Minimum confidence level required (0-1)",
    category: "advanced",
    type: "number",
    default_value: 0.7,
    validation: { min: 0, max: 1 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "experimental_features",
    name: "Experimental Features",
    description: "Enable experimental features",
    category: "advanced",
    type: "multi_select",
    default_value: [],
    options: [
      { value: "recursive_lm", label: "Recursive Language Model (10M+ context)" },
      { value: "tree_of_thoughts", label: "Tree of Thoughts reasoning" },
      { value: "streaming_ingestion", label: "Streaming document ingestion" },
      { value: "voice_agents", label: "Voice-enabled AI agents" },
      { value: "warp_retrieval", label: "WARP Engine (3x faster search)" },
      { value: "colpali_visual", label: "ColPali Visual Document Retrieval" },
      { value: "sufficiency_detection", label: "RAG Sufficiency Detection" },
      // Phase 62/63 features
      { value: "answer_refiner", label: "Answer Refiner (+20% quality)" },
      { value: "ttt_compression", label: "TTT Compression (35x faster long context)" },
      { value: "fast_chunking", label: "Fast Chunking (33x faster via Chonkie)" },
      { value: "docling_parser", label: "Docling Parser (97.9% table accuracy)" },
      { value: "agent_evaluation", label: "Agent Evaluation Metrics" },
    ],
    requires_restart: false,
    sensitive: false,
  },

  // Advanced - Cache
  {
    key: "cache_ttl_seconds",
    name: "Cache TTL (seconds)",
    description: "How long to cache query results",
    category: "advanced",
    type: "number",
    default_value: 3600,
    validation: { min: 60, max: 86400 },
    requires_restart: false,
    sensitive: false,
  },
  {
    key: "embedding_cache_ttl",
    name: "Embedding Cache TTL (seconds)",
    description: "How long to cache embeddings (default: 7 days)",
    category: "advanced",
    type: "number",
    default_value: 604800,
    validation: { min: 3600, max: 2592000 },
    requires_restart: false,
    sensitive: false,
  },

  // Advanced - Debug
  {
    key: "debug_mode",
    name: "Debug Mode",
    description: "Enable detailed logging (affects performance)",
    category: "advanced",
    type: "boolean",
    default_value: false,
    requires_restart: false,
    sensitive: false,
  },

  // Advanced - Infrastructure
  {
    key: "database_url",
    name: "Database URL",
    description: "PostgreSQL connection string (requires restart)",
    category: "advanced",
    type: "secret",
    default_value: "",
    requires_restart: true,
    sensitive: true,
  },
  {
    key: "redis_url",
    name: "Redis URL",
    description: "Redis connection string for caching and task queue (requires restart)",
    category: "advanced",
    type: "secret",
    default_value: "",
    requires_restart: true,
    sensitive: true,
  },
  {
    key: "sentry_dsn",
    name: "Sentry DSN",
    description: "Sentry error tracking DSN (leave empty to disable)",
    category: "advanced",
    type: "secret",
    default_value: "",
    requires_restart: true,
    sensitive: true,
  },
  {
    key: "celery_worker_concurrency",
    name: "Celery Worker Concurrency",
    description: "Number of concurrent Celery workers",
    category: "advanced",
    type: "number",
    default_value: 4,
    validation: { min: 1, max: 32 },
    requires_restart: true,
    sensitive: false,
  },
];

// Create lookup by key
const SETTINGS_BY_KEY = Object.fromEntries(ALL_SETTINGS.map(s => [s.key, s]));

// =============================================================================
// MAIN ADMIN PANEL
// =============================================================================

interface AdminPanelProps {
  className?: string;
}

export function AdminPanel({ className }: AdminPanelProps) {
  const [activeTab, setActiveTab] = React.useState<string>("overview");

  return (
    <div className={cn("h-full flex flex-col", className)}>
      <div className="flex items-center justify-between p-6 border-b">
        <div>
          <h1 className="text-2xl font-bold">Admin Settings</h1>
          <p className="text-muted-foreground">
            Manage organization settings, users, and view audit logs
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <div className="border-b px-6">
          <TabsList className="h-12">
            <TabsTrigger value="overview" className="gap-2">
              <BarChart3 className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="settings" className="gap-2">
              <Settings className="h-4 w-4" />
              Settings
            </TabsTrigger>
            <TabsTrigger value="users" className="gap-2">
              <Users className="h-4 w-4" />
              Users
            </TabsTrigger>
            <TabsTrigger value="security" className="gap-2">
              <Shield className="h-4 w-4" />
              Security
            </TabsTrigger>
            <TabsTrigger value="audit" className="gap-2">
              <FileText className="h-4 w-4" />
              Audit Logs
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-hidden">
          <TabsContent value="overview" className="h-full m-0 p-6">
            <OverviewTab />
          </TabsContent>
          <TabsContent value="settings" className="h-full m-0 p-6">
            <SettingsTab />
          </TabsContent>
          <TabsContent value="users" className="h-full m-0 p-6">
            <UsersTab />
          </TabsContent>
          <TabsContent value="security" className="h-full m-0 p-6">
            <SecurityTab />
          </TabsContent>
          <TabsContent value="audit" className="h-full m-0 p-6">
            <AuditLogsTab />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

// =============================================================================
// OVERVIEW TAB
// =============================================================================

function OverviewTab() {
  const [stats, setStats] = React.useState<SystemStats | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    // Simulated stats - replace with actual API call
    setTimeout(() => {
      setStats({
        organization: {
          id: "org_123",
          name: "Acme Corp",
          tier: "enterprise",
        },
        users: {
          total: 24,
          by_role: {
            org_admin: 2,
            manager: 5,
            editor: 8,
            viewer: 9,
          },
        },
        activity: {
          recent_actions: 156,
          by_action: {
            document_uploaded: 45,
            query_executed: 78,
            setting_updated: 12,
            user_invited: 8,
            user_role_updated: 5,
            document_deleted: 8,
          },
        },
        settings: {
          custom_overrides: 7,
          total_available: ALL_SETTINGS.length,
        },
      });
      setLoading(false);
    }, 500);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Users"
          value={stats.users.total}
          icon={Users}
          trend="+3 this week"
        />
        <StatCard
          title="Recent Activity"
          value={stats.activity.recent_actions}
          icon={Activity}
          subtitle="Last 7 days"
        />
        <StatCard
          title="Custom Settings"
          value={`${stats.settings.custom_overrides}/${stats.settings.total_available}`}
          icon={Settings}
          subtitle="Modified"
        />
        <StatCard
          title="Plan"
          value={stats.organization.tier}
          icon={Shield}
          className="capitalize"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Users by Role */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Users by Role</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(stats.users.by_role).map(([role, count]) => (
                <div key={role} className="flex items-center gap-4">
                  <div className="w-24 text-sm capitalize">
                    {role.replace("_", " ")}
                  </div>
                  <div className="flex-1">
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all"
                        style={{
                          width: `${(count / stats.users.total) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                  <div className="w-8 text-sm text-muted-foreground">{count}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Activity Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(stats.activity.by_action)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 6)
                .map(([action, count]) => (
                  <div key={action} className="flex items-center justify-between">
                    <span className="text-sm capitalize">
                      {action.replace(/_/g, " ")}
                    </span>
                    <Badge variant="secondary">{count}</Badge>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" className="gap-2">
              <Plus className="h-4 w-4" />
              Invite User
            </Button>
            <Button variant="outline" className="gap-2">
              <Download className="h-4 w-4" />
              Export Audit Logs
            </Button>
            <Button variant="outline" className="gap-2">
              <Shield className="h-4 w-4" />
              Security Review
            </Button>
            <Button variant="outline" className="gap-2">
              <Key className="h-4 w-4" />
              Manage API Keys
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ElementType;
  trend?: string;
  subtitle?: string;
  className?: string;
}

function StatCard({ title, value, icon: Icon, trend, subtitle, className }: StatCardProps) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className={cn("text-2xl font-bold mt-1", className)}>{value}</p>
            {trend && (
              <p className="text-xs text-green-600 mt-1">{trend}</p>
            )}
            {subtitle && (
              <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
            )}
          </div>
          <div className="p-2 bg-primary/10 rounded-lg">
            <Icon className="h-5 w-5 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// SETTINGS TAB - COMPLETELY REBUILT
// =============================================================================

function SettingsTab() {
  const [category, setCategory] = React.useState<SettingCategory>("general");
  const [settings, setSettings] = React.useState<Record<string, any>>({});
  const [modified, setModified] = React.useState<Set<string>>(new Set());
  const [saving, setSaving] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [searchQuery, setSearchQuery] = React.useState("");

  const categories: Array<{ value: SettingCategory; label: string; icon: React.ElementType; description: string }> = [
    { value: "general", label: "General", icon: Settings, description: "Basic application settings" },
    { value: "security", label: "Security", icon: Shield, description: "Authentication and access control" },
    { value: "processing", label: "AI & Processing", icon: Cpu, description: "LLM, RAG, and document processing" },
    { value: "integrations", label: "Integrations", icon: Key, description: "API keys and external services" },
    { value: "notifications", label: "Notifications", icon: Bell, description: "Email and alert settings" },
    { value: "advanced", label: "Advanced", icon: Server, description: "Infrastructure and experimental" },
  ];

  // Load settings
  React.useEffect(() => {
    const loadSettings = async () => {
      setLoading(true);
      try {
        const data = await fetchSettings();
        if (Object.keys(data).length > 0) {
          // Use API data
          const values: Record<string, any> = {};
          for (const [key, setting] of Object.entries(data)) {
            values[key] = setting.value;
          }
          setSettings(values);
        } else {
          // Use defaults from definitions
          const defaults: Record<string, any> = {};
          for (const def of ALL_SETTINGS) {
            defaults[def.key] = def.default_value;
          }
          setSettings(defaults);
        }
      } catch (error) {
        // Use defaults on error
        const defaults: Record<string, any> = {};
        for (const def of ALL_SETTINGS) {
          defaults[def.key] = def.default_value;
        }
        setSettings(defaults);
      }
      setLoading(false);
    };
    loadSettings();
  }, []);

  // Filter settings by category and search
  const filteredSettings = ALL_SETTINGS.filter((def) => {
    const matchesCategory = def.category === category;
    const matchesSearch = !searchQuery ||
      def.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      def.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      def.key.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  // Group settings by subcategory for better organization
  const groupedSettings = React.useMemo(() => {
    const groups: Record<string, SettingDefinition[]> = {};

    for (const setting of filteredSettings) {
      let group = "General";

      if (category === "processing") {
        if (setting.key.includes("llm") || setting.key.includes("chat") || setting.key.includes("model")) {
          group = "LLM & Chat";
        } else if (setting.key.includes("embedding")) {
          group = "Embeddings";
        } else if (setting.key.includes("rag") || setting.key.includes("hybrid") || setting.key.includes("colbert") || setting.key.includes("reranking")) {
          group = "Retrieval & RAG";
        } else if (setting.key.includes("tts") || setting.key.includes("voice") || setting.key.includes("audio")) {
          group = "Audio & TTS";
        } else if (setting.key.includes("ocr") || setting.key.includes("chunk") || setting.key.includes("file") || setting.key.includes("upload") || setting.key.includes("dir")) {
          group = "Document Processing";
        }
      } else if (category === "integrations") {
        if (setting.key.includes("openai") || setting.key.includes("anthropic") || setting.key.includes("google_api") || setting.key.includes("groq") || setting.key.includes("mistral") || setting.key.includes("together") || setting.key.includes("deepseek")) {
          group = "LLM Providers";
        } else if (setting.key.includes("voyage") || setting.key.includes("cohere") || setting.key.includes("jina")) {
          group = "Embedding Providers";
        } else if (setting.key.includes("elevenlabs") || setting.key.includes("cartesia")) {
          group = "TTS Providers";
        } else if (setting.key.includes("google") || setting.key.includes("notion") || setting.key.includes("slack")) {
          group = "Workspace Integrations";
        } else {
          group = "Other Services";
        }
      } else if (category === "advanced") {
        if (setting.key.includes("database") || setting.key.includes("redis") || setting.key.includes("sentry") || setting.key.includes("celery")) {
          group = "Infrastructure";
        } else if (setting.key.includes("cache")) {
          group = "Caching";
        } else if (setting.key.includes("enable") || setting.key.includes("experimental") || setting.key.includes("sufficiency")) {
          group = "Feature Flags";
        } else {
          group = "Other";
        }
      }

      if (!groups[group]) groups[group] = [];
      groups[group].push(setting);
    }

    return groups;
  }, [filteredSettings, category]);

  const handleValueChange = (key: string, value: any) => {
    setSettings((prev) => ({
      ...prev,
      [key]: value,
    }));
    setModified((prev) => new Set(prev).add(key));
  };

  const handleSave = async () => {
    setSaving(true);
    const modifiedKeys = Array.from(modified);
    let successCount = 0;

    for (const key of modifiedKeys) {
      const success = await updateSetting(key, settings[key]);
      if (success) successCount++;
    }

    if (successCount === modifiedKeys.length) {
      setModified(new Set());
    }
    setSaving(false);
  };

  const handleReset = async (key: string) => {
    const definition = SETTINGS_BY_KEY[key];
    if (definition) {
      handleValueChange(key, definition.default_value);
      await resetSetting(key);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="flex gap-6 h-full">
      {/* Category Sidebar */}
      <div className="w-56 shrink-0">
        <div className="mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search settings..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>
        <nav className="space-y-1">
          {categories.map(({ value, label, icon: Icon, description }) => (
            <button
              key={value}
              onClick={() => setCategory(value)}
              className={cn(
                "w-full flex flex-col items-start gap-1 px-3 py-3 text-sm rounded-lg transition-colors text-left",
                category === value
                  ? "bg-primary text-primary-foreground"
                  : "hover:bg-muted"
              )}
            >
              <div className="flex items-center gap-2">
                <Icon className="h-4 w-4" />
                <span className="font-medium">{label}</span>
              </div>
              <span className={cn(
                "text-xs",
                category === value ? "text-primary-foreground/80" : "text-muted-foreground"
              )}>
                {description}
              </span>
            </button>
          ))}
        </nav>
      </div>

      {/* Settings Content */}
      <div className="flex-1 min-w-0">
        <Card className="h-full flex flex-col">
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-4">
            <div>
              <CardTitle className="capitalize flex items-center gap-2">
                {categories.find(c => c.value === category)?.icon &&
                  React.createElement(categories.find(c => c.value === category)!.icon, { className: "h-5 w-5" })
                }
                {categories.find(c => c.value === category)?.label} Settings
              </CardTitle>
              <CardDescription>
                {categories.find(c => c.value === category)?.description}
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => window.location.reload()} className="gap-2">
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
              {modified.size > 0 && (
                <Button onClick={handleSave} disabled={saving} className="gap-2">
                  <Save className="h-4 w-4" />
                  {saving ? "Saving..." : `Save ${modified.size} changes`}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-8">
                {Object.entries(groupedSettings).map(([groupName, groupSettings]) => (
                  <div key={groupName}>
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
                      {groupName === "Infrastructure" && <Database className="h-4 w-4" />}
                      {groupName === "LLM Providers" && <Zap className="h-4 w-4" />}
                      {groupName === "Feature Flags" && <Zap className="h-4 w-4" />}
                      {groupName}
                    </h3>
                    <div className="space-y-4">
                      {groupSettings.map((definition) => (
                        <SettingField
                          key={definition.key}
                          definition={definition}
                          value={settings[definition.key]}
                          isModified={modified.has(definition.key)}
                          onChange={(value) => handleValueChange(definition.key, value)}
                          onReset={() => handleReset(definition.key)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

interface SettingFieldProps {
  definition: SettingDefinition;
  value: any;
  isModified: boolean;
  onChange: (value: any) => void;
  onReset: () => void;
}

function SettingField({ definition, value, isModified, onChange, onReset }: SettingFieldProps) {
  const [showSecret, setShowSecret] = React.useState(false);
  const isDefault = JSON.stringify(value) === JSON.stringify(definition.default_value);

  return (
    <div className="flex items-start gap-4 p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          <Label className="text-base font-medium">{definition.name}</Label>
          {isModified && (
            <Badge variant="default" className="text-xs">Unsaved</Badge>
          )}
          {!isDefault && !isModified && (
            <Badge variant="secondary" className="text-xs">Custom</Badge>
          )}
          {definition.sensitive && (
            <Badge variant="outline" className="text-xs gap-1">
              <Lock className="h-3 w-3" />
              Sensitive
            </Badge>
          )}
          {definition.requires_restart && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="text-xs gap-1 text-amber-600 border-amber-600">
                    <AlertTriangle className="h-3 w-3" />
                    Restart
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>Requires restart to take effect</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <p className="text-sm text-muted-foreground">{definition.description}</p>

        <div className="pt-2">
          {definition.type === "boolean" && (
            <div className="flex items-center gap-3">
              <Switch
                checked={value ?? false}
                onCheckedChange={onChange}
              />
              <span className="text-sm text-muted-foreground">
                {value ? "Enabled" : "Disabled"}
              </span>
            </div>
          )}

          {definition.type === "string" && (
            <Input
              value={value ?? ""}
              onChange={(e) => onChange(e.target.value)}
              maxLength={definition.validation?.max_length}
              className="max-w-md"
              placeholder={`Enter ${definition.name.toLowerCase()}`}
            />
          )}

          {definition.type === "number" && (
            <div className="flex items-center gap-4 max-w-md">
              <Input
                type="number"
                value={value ?? 0}
                onChange={(e) => onChange(Number(e.target.value))}
                min={definition.validation?.min}
                max={definition.validation?.max}
                className="w-32"
              />
              {definition.validation?.min !== undefined && definition.validation?.max !== undefined && (
                <span className="text-xs text-muted-foreground">
                  Range: {definition.validation.min} - {definition.validation.max}
                </span>
              )}
            </div>
          )}

          {definition.type === "select" && (
            <Select value={value ?? ""} onValueChange={onChange}>
              <SelectTrigger className="max-w-md">
                <SelectValue placeholder={`Select ${definition.name.toLowerCase()}`} />
              </SelectTrigger>
              <SelectContent>
                {definition.options?.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          {definition.type === "multi_select" && (
            <div className="flex flex-wrap gap-2 max-w-lg">
              {definition.options?.map((opt) => {
                const selected = (value || []).includes(opt.value);
                return (
                  <Badge
                    key={opt.value}
                    variant={selected ? "default" : "outline"}
                    className="cursor-pointer transition-colors"
                    onClick={() => {
                      const current = value || [];
                      const newValue = selected
                        ? current.filter((v: string) => v !== opt.value)
                        : [...current, opt.value];
                      onChange(newValue);
                    }}
                  >
                    {selected && <Check className="h-3 w-3 mr-1" />}
                    {opt.label}
                  </Badge>
                );
              })}
            </div>
          )}

          {definition.type === "secret" && (
            <div className="flex items-center gap-2 max-w-md">
              <div className="relative flex-1">
                <Input
                  type={showSecret ? "text" : "password"}
                  value={value ?? ""}
                  onChange={(e) => onChange(e.target.value)}
                  className="pr-10 font-mono"
                  placeholder={value ? "••••••••••••" : `Enter ${definition.name.toLowerCase()}`}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-0 top-0 h-full px-3"
                  onClick={() => setShowSecret(!showSecret)}
                >
                  {showSecret ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
              {value && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="outline" className="text-green-600 border-green-600">
                        <Check className="h-3 w-3 mr-1" />
                        Set
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>API key is configured</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          )}

          {definition.type === "json" && (
            <Textarea
              value={typeof value === "object" ? JSON.stringify(value, null, 2) : value ?? ""}
              onChange={(e) => {
                try {
                  onChange(JSON.parse(e.target.value));
                } catch {
                  // Keep as string if not valid JSON
                }
              }}
              className="max-w-lg font-mono text-sm"
              rows={4}
              placeholder="Enter JSON value"
            />
          )}
        </div>
      </div>

      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={onReset}
              disabled={isDefault && !isModified}
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Reset to default</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
}

// =============================================================================
// USERS TAB
// =============================================================================

function UsersTab() {
  const [users, setUsers] = React.useState<User[]>([]);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [roleFilter, setRoleFilter] = React.useState<string>("all");
  const [inviteOpen, setInviteOpen] = React.useState(false);

  React.useEffect(() => {
    // Simulated users
    setUsers([
      { user_id: "user_1", email: "admin@acme.com", role: "org_admin", permissions: ["*"], created_at: "2024-01-15" },
      { user_id: "user_2", email: "john@acme.com", role: "manager", permissions: ["document:*", "user:read"], created_at: "2024-02-20" },
      { user_id: "user_3", email: "jane@acme.com", role: "editor", permissions: ["document:read", "document:create"], created_at: "2024-03-10" },
      { user_id: "user_4", email: "bob@acme.com", role: "viewer", permissions: ["document:read"], created_at: "2024-04-05" },
    ]);
  }, []);

  const filteredUsers = users.filter((user) => {
    const matchesSearch = !searchQuery ||
      user.email?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.user_id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesRole = roleFilter === "all" || user.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  const roles = ["org_admin", "manager", "editor", "viewer", "api_user"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search users..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 w-64"
            />
          </div>
          <Select value={roleFilter} onValueChange={setRoleFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Filter by role" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Roles</SelectItem>
              {roles.map((role) => (
                <SelectItem key={role} value={role} className="capitalize">
                  {role.replace("_", " ")}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
          <DialogTrigger asChild>
            <Button className="gap-2">
              <Plus className="h-4 w-4" />
              Invite User
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Invite User</DialogTitle>
              <DialogDescription>
                Send an invitation to join your organization
              </DialogDescription>
            </DialogHeader>
            <InviteUserForm onClose={() => setInviteOpen(false)} />
          </DialogContent>
        </Dialog>
      </div>

      {/* Users Table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Permissions</TableHead>
              <TableHead>Joined</TableHead>
              <TableHead className="w-16"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredUsers.map((user) => (
              <TableRow key={user.user_id}>
                <TableCell>
                  <div className="flex items-center gap-3">
                    <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                      <span className="text-sm font-medium text-primary">
                        {user.email?.[0].toUpperCase() || "U"}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium">{user.email || user.user_id}</p>
                      <p className="text-xs text-muted-foreground">{user.user_id}</p>
                    </div>
                  </div>
                </TableCell>
                <TableCell>
                  <RoleBadge role={user.role} />
                </TableCell>
                <TableCell>
                  <div className="flex flex-wrap gap-1">
                    {user.permissions.slice(0, 3).map((perm) => (
                      <Badge key={perm} variant="outline" className="text-xs">
                        {perm}
                      </Badge>
                    ))}
                    {user.permissions.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{user.permissions.length - 3}
                      </Badge>
                    )}
                  </div>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {user.created_at}
                </TableCell>
                <TableCell>
                  <UserActions user={user} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
}

function RoleBadge({ role }: { role: string }) {
  const colors: Record<string, string> = {
    super_admin: "bg-red-100 text-red-800 border-red-200",
    org_admin: "bg-purple-100 text-purple-800 border-purple-200",
    manager: "bg-blue-100 text-blue-800 border-blue-200",
    editor: "bg-green-100 text-green-800 border-green-200",
    viewer: "bg-gray-100 text-gray-800 border-gray-200",
    api_user: "bg-amber-100 text-amber-800 border-amber-200",
  };

  return (
    <Badge className={cn("capitalize", colors[role] || colors.viewer)}>
      {role.replace("_", " ")}
    </Badge>
  );
}

function UserActions({ user }: { user: User }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon">
          <ChevronRight className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem className="gap-2">
          <Edit3 className="h-4 w-4" />
          Edit Role
        </DropdownMenuItem>
        <DropdownMenuItem className="gap-2">
          <Shield className="h-4 w-4" />
          View Permissions
        </DropdownMenuItem>
        <DropdownMenuItem className="gap-2 text-destructive">
          <Trash2 className="h-4 w-4" />
          Remove User
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function InviteUserForm({ onClose }: { onClose: () => void }) {
  const [email, setEmail] = React.useState("");
  const [role, setRole] = React.useState("viewer");
  const [sending, setSending] = React.useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSending(true);
    await new Promise((r) => setTimeout(r, 1000));
    setSending(false);
    onClose();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label>Email Address</Label>
        <Input
          type="email"
          placeholder="user@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>
      <div className="space-y-2">
        <Label>Role</Label>
        <Select value={role} onValueChange={setRole}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="viewer">Viewer</SelectItem>
            <SelectItem value="editor">Editor</SelectItem>
            <SelectItem value="manager">Manager</SelectItem>
            <SelectItem value="org_admin">Admin</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onClose}>
          Cancel
        </Button>
        <Button type="submit" disabled={sending}>
          {sending ? "Sending..." : "Send Invitation"}
        </Button>
      </DialogFooter>
    </form>
  );
}

// =============================================================================
// SECURITY TAB
// =============================================================================

function SecurityTab() {
  const [mfaEnabled, setMfaEnabled] = React.useState(false);
  const [ipRestriction, setIpRestriction] = React.useState(false);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* MFA Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock className="h-5 w-5" />
              Multi-Factor Authentication
            </CardTitle>
            <CardDescription>
              Require additional verification for all users
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Require MFA for all users</p>
                <p className="text-sm text-muted-foreground">
                  Users must set up MFA on next login
                </p>
              </div>
              <Switch checked={mfaEnabled} onCheckedChange={setMfaEnabled} />
            </div>
            <Separator />
            <div className="space-y-2">
              <p className="text-sm font-medium">MFA Status</p>
              <div className="flex items-center gap-2">
                <Badge variant="secondary">18/24 users enabled</Badge>
                <Badge variant="outline">75% adoption</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* IP Restriction */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              IP Restrictions
            </CardTitle>
            <CardDescription>
              Limit access to specific IP addresses
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Enable IP whitelist</p>
                <p className="text-sm text-muted-foreground">
                  Only allow access from approved IPs
                </p>
              </div>
              <Switch checked={ipRestriction} onCheckedChange={setIpRestriction} />
            </div>
            {ipRestriction && (
              <>
                <Separator />
                <div className="space-y-2">
                  <Label>Allowed IP Ranges (CIDR)</Label>
                  <Input placeholder="192.168.1.0/24" />
                  <Button variant="outline" size="sm" className="gap-2">
                    <Plus className="h-4 w-4" />
                    Add Range
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Session Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Session Management
            </CardTitle>
            <CardDescription>
              Configure session timeout and security
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Session Timeout (minutes)</Label>
              <Input type="number" defaultValue={60} min={5} max={1440} />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Force logout on IP change</p>
                <p className="text-sm text-muted-foreground">
                  Invalidate session if IP address changes
                </p>
              </div>
              <Switch />
            </div>
          </CardContent>
        </Card>

        {/* API Security */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Key className="h-5 w-5" />
              API Security
            </CardTitle>
            <CardDescription>
              Configure API rate limits and restrictions
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Rate Limit (requests/minute)</Label>
              <Input type="number" defaultValue={60} min={10} max={1000} />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Require API key for all requests</p>
                <p className="text-sm text-muted-foreground">
                  Disable anonymous API access
                </p>
              </div>
              <Switch defaultChecked />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// =============================================================================
// AUDIT LOGS TAB
// =============================================================================

function AuditLogsTab() {
  const [logs, setLogs] = React.useState<AuditLog[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [filters, setFilters] = React.useState({
    action: "",
    dateRange: "7d",
  });

  React.useEffect(() => {
    // Simulated audit logs
    setTimeout(() => {
      setLogs([
        {
          id: "log_1",
          timestamp: "2024-01-20T14:30:00Z",
          action: "user_role_updated",
          user_id: "admin@acme.com",
          org_id: "org_123",
          resource_type: "user",
          resource_id: "john@acme.com",
          ip_address: "192.168.1.100",
          details: { old_role: "viewer", new_role: "editor" },
        },
        {
          id: "log_2",
          timestamp: "2024-01-20T14:25:00Z",
          action: "setting_updated",
          user_id: "admin@acme.com",
          org_id: "org_123",
          resource_type: "setting",
          resource_id: "max_file_size_mb",
          ip_address: "192.168.1.100",
          details: { previous_value: 50, new_value: 100 },
        },
        {
          id: "log_3",
          timestamp: "2024-01-20T14:20:00Z",
          action: "document_uploaded",
          user_id: "john@acme.com",
          org_id: "org_123",
          resource_type: "document",
          resource_id: "doc_456",
          ip_address: "192.168.1.101",
          details: { filename: "report.pdf", size: 2500000 },
        },
        {
          id: "log_4",
          timestamp: "2024-01-20T14:15:00Z",
          action: "user_invited",
          user_id: "admin@acme.com",
          org_id: "org_123",
          resource_type: "invite",
          resource_id: "invite_789",
          ip_address: "192.168.1.100",
          details: { email: "newuser@acme.com", role: "viewer" },
        },
      ]);
      setLoading(false);
    }, 500);
  }, []);

  const handleExport = async (format: "csv" | "json") => {
    // Trigger export
    console.log("Exporting as", format);
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Select
            value={filters.action}
            onValueChange={(v) => setFilters((f) => ({ ...f, action: v }))}
          >
            <SelectTrigger className="w-48">
              <SelectValue placeholder="All Actions" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">All Actions</SelectItem>
              <SelectItem value="user_role_updated">Role Updated</SelectItem>
              <SelectItem value="setting_updated">Setting Updated</SelectItem>
              <SelectItem value="document_uploaded">Document Uploaded</SelectItem>
              <SelectItem value="user_invited">User Invited</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={filters.dateRange}
            onValueChange={(v) => setFilters((f) => ({ ...f, dateRange: v }))}
          >
            <SelectTrigger className="w-36">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="24h">Last 24 hours</SelectItem>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="gap-2">
              <Download className="h-4 w-4" />
              Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => handleExport("csv")}>
              Export as CSV
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => handleExport("json")}>
              Export as JSON
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Logs Table */}
      <Card>
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>User</TableHead>
                <TableHead>Resource</TableHead>
                <TableHead>IP Address</TableHead>
                <TableHead>Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {logs.map((log) => (
                <TableRow key={log.id}>
                  <TableCell className="text-muted-foreground">
                    {new Date(log.timestamp).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <ActionBadge action={log.action} />
                  </TableCell>
                  <TableCell>{log.user_id}</TableCell>
                  <TableCell>
                    <span className="text-muted-foreground">{log.resource_type}/</span>
                    {log.resource_id}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {log.ip_address}
                  </TableCell>
                  <TableCell>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger>
                          <Badge variant="outline" className="cursor-help">
                            {Object.keys(log.details).length} fields
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent>
                          <pre className="text-xs">
                            {JSON.stringify(log.details, null, 2)}
                          </pre>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Card>
    </div>
  );
}

function ActionBadge({ action }: { action: string }) {
  const colors: Record<string, string> = {
    user_role_updated: "bg-purple-100 text-purple-800",
    setting_updated: "bg-blue-100 text-blue-800",
    document_uploaded: "bg-green-100 text-green-800",
    document_deleted: "bg-red-100 text-red-800",
    user_invited: "bg-amber-100 text-amber-800",
    user_removed: "bg-red-100 text-red-800",
    audit_logs_exported: "bg-gray-100 text-gray-800",
  };

  return (
    <Badge className={cn("capitalize", colors[action] || "bg-gray-100 text-gray-800")}>
      {action.replace(/_/g, " ")}
    </Badge>
  );
}

export default AdminPanel;
