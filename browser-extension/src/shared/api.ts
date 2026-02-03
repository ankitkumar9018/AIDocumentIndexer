/**
 * API Client for Browser Extension
 * ==================================
 *
 * Communicates with AIDocumentIndexer backend.
 */

import {
  SearchResult,
  ChatResponse,
  Collection,
  CapturedPage,
  ChatMessage,
  HealthCheckResponse,
  ExtensionSettings,
} from './types';

export class ApiClient {
  private baseUrl: string = 'http://localhost:8000';
  private apiKey: string = '';

  constructor() {
    this.loadSettings();
  }

  private async loadSettings(): Promise<void> {
    try {
      const settings = await chrome.storage.sync.get(['serverUrl', 'apiKey']);
      if (settings.serverUrl) {
        this.baseUrl = settings.serverUrl.replace(/\/$/, '');
      }
      if (settings.apiKey) {
        this.apiKey = settings.apiKey;
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }

  async updateSettings(settings: Partial<ExtensionSettings>): Promise<void> {
    if (settings.serverUrl) {
      this.baseUrl = settings.serverUrl.replace(/\/$/, '');
    }
    if (settings.apiKey !== undefined) {
      this.apiKey = settings.apiKey;
    }
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}/api/v1${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        ...this.getHeaders(),
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text().catch(() => 'Unknown error');
      throw new Error(`API error (${response.status}): ${error}`);
    }

    return response.json();
  }

  // Health check
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  // Search
  async search(
    query: string,
    collectionId?: string,
    topK: number = 10
  ): Promise<SearchResult[]> {
    const response = await this.request<{ results: SearchResult[] }>('/search', {
      method: 'POST',
      body: JSON.stringify({
        query,
        collection_id: collectionId,
        top_k: topK,
      }),
    });
    return response.results || [];
  }

  // Chat
  async chat(
    message: string,
    history: ChatMessage[] = [],
    collectionId?: string
  ): Promise<ChatResponse> {
    const response = await this.request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        history: history.map((m) => ({
          role: m.role,
          content: m.content,
        })),
        collection_id: collectionId,
      }),
    });
    return response;
  }

  // Chat streaming
  async *chatStream(
    message: string,
    history: ChatMessage[] = [],
    collectionId?: string
  ): AsyncGenerator<string, void, unknown> {
    const url = `${this.baseUrl}/api/v1/chat/stream`;

    const response = await fetch(url, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        message,
        history: history.map((m) => ({
          role: m.role,
          content: m.content,
        })),
        collection_id: collectionId,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Chat stream failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter((line) => line.startsWith('data: '));

        for (const line of lines) {
          const data = line.slice(6); // Remove 'data: ' prefix
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              yield parsed.content;
            }
          } catch {
            // Ignore parse errors for partial JSON
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // Collections
  async getCollections(): Promise<Collection[]> {
    const response = await this.request<{ collections: Collection[] }>('/collections');
    return response.collections || [];
  }

  async createCollection(name: string, description?: string): Promise<Collection> {
    return this.request<Collection>('/collections', {
      method: 'POST',
      body: JSON.stringify({ name, description }),
    });
  }

  // Page capture
  async capturePage(page: CapturedPage): Promise<{ documentId: string }> {
    return this.request<{ documentId: string }>('/documents/capture', {
      method: 'POST',
      body: JSON.stringify({
        url: page.url,
        title: page.title,
        content: page.content,
        metadata: {
          description: page.description,
          author: page.author,
          captured_at: page.capturedAt,
        },
        collection_id: page.collectionId,
      }),
    });
  }

  // Quick actions
  async summarize(text: string): Promise<string> {
    const response = await this.request<{ summary: string }>('/skills/summarize', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
    return response.summary;
  }

  async translate(text: string, targetLanguage: string): Promise<string> {
    const response = await this.request<{ translation: string }>('/skills/translate', {
      method: 'POST',
      body: JSON.stringify({ text, target_language: targetLanguage }),
    });
    return response.translation;
  }

  async explain(text: string): Promise<string> {
    const response = await this.request<{ explanation: string }>('/skills/explain', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
    return response.explanation;
  }
}

// Singleton instance
let apiClient: ApiClient | null = null;

export function getApiClient(): ApiClient {
  if (!apiClient) {
    apiClient = new ApiClient();
  }
  return apiClient;
}
