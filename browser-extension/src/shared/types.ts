/**
 * Shared Types for Browser Extension
 * ====================================
 */

// Message types for extension communication
export enum MessageType {
  // Search & Chat
  SEARCH = 'search',
  SEARCH_QUERY = 'search-query',
  CHAT = 'chat',
  CHAT_STREAM = 'chat-stream',

  // Page capture
  CAPTURE_PAGE = 'capture-page',
  CAPTURE_SELECTION = 'capture-selection',

  // Settings
  GET_SETTINGS = 'get-settings',
  SET_SETTINGS = 'set-settings',

  // Collections
  GET_COLLECTIONS = 'get-collections',

  // History
  GET_CAPTURE_HISTORY = 'get-capture-history',

  // Connection
  CHECK_CONNECTION = 'check-connection',

  // License
  GET_LICENSE = 'get-license',
  VALIDATE_LICENSE = 'validate-license',
  CHECK_FEATURE = 'check-feature',
}

export interface Message {
  type: MessageType;
  payload?: any;
}

export interface MessageResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

// Search result
export interface SearchResult {
  id: string;
  documentId?: string;
  content: string;
  score?: number;
  title?: string;
  metadata?: {
    title?: string;
    source?: string;
    page?: number;
  };
}

// Chat message
export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  sources?: SearchResult[];
  citations?: Array<{
    title?: string;
    score?: number;
    metadata?: Record<string, any>;
  }>;
}

// Chat response
export interface ChatResponse {
  message: string;
  sources: SearchResult[];
  conversationId?: string;
}

// Captured page
export interface CapturedPage {
  url: string;
  title: string;
  content: string;
  description: string;
  author: string;
  capturedAt: string;
  collectionId?: string;
}

// Collection
export interface Collection {
  id: string;
  name: string;
  description?: string;
  documentCount?: number;
  document_count?: number;
}

// Extension settings
export interface ExtensionSettings {
  serverUrl: string;
  apiKey: string;
  autoCapture: boolean;
  captureOnVisit?: boolean;
  defaultCollectionId: string;
  excludedDomains: string[];
  resultsCount: number;
  showCitations: boolean;
  theme?: 'light' | 'dark' | 'system';
}

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface HealthCheckResponse {
  status: string;
  version: string;
}

export interface SearchRequest {
  query: string;
  collectionId?: string;
  topK?: number;
  filters?: Record<string, any>;
}

export interface ChatRequest {
  message: string;
  history?: ChatMessage[];
  collectionId?: string;
  conversationId?: string;
}
