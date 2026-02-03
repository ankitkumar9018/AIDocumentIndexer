/**
 * AIDocumentIndexer Side Panel Script
 * ====================================
 *
 * Full chat interface for interacting with indexed documents.
 * Supports streaming responses, citations, and collection filtering.
 */

import { MessageType, ChatMessage, SearchResult, Collection } from '../shared/types';

// DOM Elements
const messagesContainer = document.getElementById('messages') as HTMLDivElement;
const messageInput = document.getElementById('message-input') as HTMLTextAreaElement;
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;
const newChatBtn = document.getElementById('new-chat-btn') as HTMLButtonElement;
const settingsBtn = document.getElementById('settings-btn') as HTMLButtonElement;
const collectionSelect = document.getElementById('collection-select') as HTMLSelectElement;

// State
let messages: ChatMessage[] = [];
let isStreaming = false;
let currentStreamMessageId: string | null = null;
let selectedCollectionId: string | null = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadCollections();
  setupEventListeners();
  adjustTextareaHeight();
});

// Event Listeners
function setupEventListeners(): void {
  // Send message
  sendBtn.addEventListener('click', sendMessage);
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Input changes
  messageInput.addEventListener('input', () => {
    adjustTextareaHeight();
    sendBtn.disabled = !messageInput.value.trim() || isStreaming;
  });

  // New chat
  newChatBtn.addEventListener('click', startNewChat);

  // Settings
  settingsBtn.addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });

  // Collection selection
  collectionSelect.addEventListener('change', (e) => {
    selectedCollectionId = (e.target as HTMLSelectElement).value || null;
  });

  // Suggestion buttons
  document.querySelectorAll('.suggestion-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const query = (btn as HTMLElement).dataset.query;
      if (query) {
        messageInput.value = query;
        adjustTextareaHeight();
        sendBtn.disabled = false;
        messageInput.focus();
      }
    });
  });
}

// Load collections
async function loadCollections(): Promise<void> {
  try {
    const response = await chrome.runtime.sendMessage({
      type: MessageType.GET_COLLECTIONS,
    });

    if (response.success && response.data) {
      renderCollections(response.data);
    }
  } catch (error) {
    console.error('Failed to load collections:', error);
  }
}

function renderCollections(collections: Collection[]): void {
  // Keep the "All Documents" option
  collectionSelect.innerHTML = '<option value="">All Documents</option>';

  collections.forEach((collection) => {
    const option = document.createElement('option');
    option.value = collection.id;
    option.textContent = `${collection.name} (${collection.document_count || 0})`;
    collectionSelect.appendChild(option);
  });
}

// Adjust textarea height
function adjustTextareaHeight(): void {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// Send message
async function sendMessage(): Promise<void> {
  const content = messageInput.value.trim();
  if (!content || isStreaming) return;

  // Hide welcome message
  const welcomeMessage = messagesContainer.querySelector('.welcome-message');
  if (welcomeMessage) {
    welcomeMessage.remove();
  }

  // Add user message
  const userMessage: ChatMessage = {
    id: Date.now().toString(),
    role: 'user',
    content,
    timestamp: new Date().toISOString(),
  };
  messages.push(userMessage);
  renderMessage(userMessage);

  // Clear input
  messageInput.value = '';
  adjustTextareaHeight();
  sendBtn.disabled = true;

  // Show typing indicator
  showTypingIndicator();

  // Start streaming
  isStreaming = true;

  try {
    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    currentStreamMessageId = assistantMessageId;
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    };
    messages.push(assistantMessage);

    // Send chat request
    const response = await chrome.runtime.sendMessage({
      type: MessageType.CHAT,
      payload: {
        message: content,
        collectionId: selectedCollectionId,
        stream: true,
      },
    });

    // Remove typing indicator
    hideTypingIndicator();

    if (response.success && response.data) {
      // Update the assistant message with full response
      const lastMessage = messages[messages.length - 1];
      lastMessage.content = response.data.content || response.data.answer || '';
      lastMessage.citations = response.data.citations || response.data.sources;
      renderMessage(lastMessage, true);
    } else {
      // Show error
      showError(response.error || 'Failed to get response');
      // Remove the empty assistant message
      messages.pop();
    }
  } catch (error) {
    hideTypingIndicator();
    showError('Connection error. Please check your settings.');
    // Remove the empty assistant message
    messages.pop();
  } finally {
    isStreaming = false;
    currentStreamMessageId = null;
    sendBtn.disabled = !messageInput.value.trim();
  }
}

// Render a message
function renderMessage(message: ChatMessage, replace = false): void {
  const existingEl = document.querySelector(`[data-message-id="${message.id}"]`);

  if (existingEl && replace) {
    existingEl.outerHTML = createMessageHTML(message);
  } else if (!existingEl) {
    messagesContainer.insertAdjacentHTML('beforeend', createMessageHTML(message));
  }

  // Scroll to bottom
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createMessageHTML(message: ChatMessage): string {
  const isUser = message.role === 'user';
  const avatarContent = isUser
    ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
    : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 10 4a2 2 0 0 1 2-2"></path></svg>';

  let citationsHTML = '';
  if (message.citations && message.citations.length > 0) {
    citationsHTML = `
      <div class="citations">
        <div class="citations-header">Sources</div>
        ${message.citations
          .slice(0, 5)
          .map(
            (citation) => `
          <div class="citation">
            <svg class="citation-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
            <span class="citation-title">${escapeHtml(citation.metadata?.title || citation.title || 'Document')}</span>
            ${citation.score ? `<span class="citation-score">${Math.round(citation.score * 100)}%</span>` : ''}
          </div>
        `
          )
          .join('')}
      </div>
    `;
  }

  return `
    <div class="message ${message.role}" data-message-id="${message.id}">
      <div class="message-avatar">${avatarContent}</div>
      <div class="message-content">
        ${formatMessageContent(message.content)}
        ${citationsHTML}
      </div>
    </div>
  `;
}

function formatMessageContent(content: string): string {
  // Simple markdown-like formatting
  return escapeHtml(content)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// Typing indicator
function showTypingIndicator(): void {
  const indicator = document.createElement('div');
  indicator.id = 'typing-indicator';
  indicator.className = 'message assistant';
  indicator.innerHTML = `
    <div class="message-avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 10 4a2 2 0 0 1 2-2"></path>
      </svg>
    </div>
    <div class="message-content">
      <div class="typing-indicator">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  `;
  messagesContainer.appendChild(indicator);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideTypingIndicator(): void {
  const indicator = document.getElementById('typing-indicator');
  if (indicator) {
    indicator.remove();
  }
}

// Error display
function showError(message: string): void {
  const errorEl = document.createElement('div');
  errorEl.className = 'error-message';
  errorEl.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"></circle>
      <line x1="12" y1="8" x2="12" y2="12"></line>
      <line x1="12" y1="16" x2="12.01" y2="16"></line>
    </svg>
    ${escapeHtml(message)}
  `;
  messagesContainer.appendChild(errorEl);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // Auto-remove after 5 seconds
  setTimeout(() => {
    errorEl.remove();
  }, 5000);
}

// Start new chat
function startNewChat(): void {
  messages = [];
  messagesContainer.innerHTML = `
    <div class="welcome-message">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="welcome-icon">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
      <h2>Welcome to AIDocIndexer</h2>
      <p>Ask questions about your indexed documents or search for specific information.</p>
      <div class="suggestions">
        <button class="suggestion-btn" data-query="What documents have I uploaded?">
          What documents have I uploaded?
        </button>
        <button class="suggestion-btn" data-query="Summarize my recent documents">
          Summarize my recent documents
        </button>
        <button class="suggestion-btn" data-query="Search for...">
          Search for specific topics
        </button>
      </div>
    </div>
  `;

  // Re-attach suggestion button listeners
  document.querySelectorAll('.suggestion-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const query = (btn as HTMLElement).dataset.query;
      if (query) {
        messageInput.value = query;
        adjustTextareaHeight();
        sendBtn.disabled = false;
        messageInput.focus();
      }
    });
  });

  messageInput.focus();
}

// Utility
function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
