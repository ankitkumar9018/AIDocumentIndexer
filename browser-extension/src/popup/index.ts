/**
 * AIDocumentIndexer Popup Script
 * ================================
 */

import { MessageType, SearchResult } from '../shared/types';

// DOM Elements
const searchInput = document.getElementById('search-input') as HTMLInputElement;
const searchBtn = document.getElementById('search-btn') as HTMLButtonElement;
const searchResults = document.getElementById('search-results') as HTMLDivElement;
const captureBtn = document.getElementById('capture-btn') as HTMLButtonElement;
const chatBtn = document.getElementById('chat-btn') as HTMLButtonElement;
const connectionStatus = document.getElementById('connection-status') as HTMLSpanElement;
const connectionText = document.getElementById('connection-text') as HTMLSpanElement;
const settingsLink = document.getElementById('settings-link') as HTMLAnchorElement;
const historyLink = document.getElementById('history-link') as HTMLAnchorElement;

// State
let debounceTimer: number | null = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await checkConnection();
  setupEventListeners();
});

function setupEventListeners(): void {
  // Search
  searchInput.addEventListener('input', handleSearchInput);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      performSearch();
    }
  });
  searchBtn.addEventListener('click', performSearch);

  // Actions
  captureBtn.addEventListener('click', capturePage);
  chatBtn.addEventListener('click', openChat);

  // Navigation
  settingsLink.addEventListener('click', (e) => {
    e.preventDefault();
    chrome.runtime.openOptionsPage();
  });

  historyLink.addEventListener('click', (e) => {
    e.preventDefault();
    showHistory();
  });
}

// Connection check
async function checkConnection(): Promise<void> {
  connectionStatus.className = 'status-dot connecting';
  connectionText.textContent = 'Connecting...';

  try {
    const response = await chrome.runtime.sendMessage({
      type: MessageType.CHECK_CONNECTION,
    });

    if (response.success && response.data.connected) {
      connectionStatus.className = 'status-dot connected';
      connectionText.textContent = 'Connected';
    } else {
      connectionStatus.className = 'status-dot disconnected';
      connectionText.textContent = 'Disconnected';
    }
  } catch {
    connectionStatus.className = 'status-dot disconnected';
    connectionText.textContent = 'Disconnected';
  }
}

// Search with debounce
function handleSearchInput(): void {
  if (debounceTimer) {
    clearTimeout(debounceTimer);
  }

  const query = searchInput.value.trim();
  if (query.length < 2) {
    searchResults.classList.add('hidden');
    return;
  }

  debounceTimer = window.setTimeout(() => {
    performSearch();
  }, 300);
}

async function performSearch(): Promise<void> {
  const query = searchInput.value.trim();
  if (!query) return;

  searchResults.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  searchResults.classList.remove('hidden');

  try {
    const response = await chrome.runtime.sendMessage({
      type: MessageType.SEARCH,
      payload: { query },
    });

    if (response.success && response.data) {
      renderSearchResults(response.data);
    } else {
      searchResults.innerHTML = '<div class="search-result">No results found</div>';
    }
  } catch (error) {
    searchResults.innerHTML = '<div class="search-result">Search failed</div>';
  }
}

function renderSearchResults(results: SearchResult[]): void {
  if (results.length === 0) {
    searchResults.innerHTML = '<div class="search-result">No results found</div>';
    return;
  }

  searchResults.innerHTML = results
    .slice(0, 5)
    .map(
      (result) => `
      <div class="search-result" data-id="${result.id}">
        <div class="search-result-title">${escapeHtml(result.metadata.title || 'Untitled')}</div>
        <div class="search-result-snippet">${escapeHtml(result.content.slice(0, 100))}...</div>
      </div>
    `
    )
    .join('');

  // Add click handlers
  searchResults.querySelectorAll('.search-result').forEach((el) => {
    el.addEventListener('click', () => {
      const id = (el as HTMLElement).dataset.id;
      if (id) {
        openDocument(id);
      }
    });
  });
}

// Page capture
async function capturePage(): Promise<void> {
  captureBtn.disabled = true;
  captureBtn.textContent = 'Saving...';

  try {
    const response = await chrome.runtime.sendMessage({
      type: MessageType.CAPTURE_PAGE,
    });

    if (response.success) {
      captureBtn.textContent = 'Saved!';
      setTimeout(() => {
        captureBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
            <polyline points="17 21 17 13 7 13 7 21"></polyline>
            <polyline points="7 3 7 8 15 8"></polyline>
          </svg>
          Save This Page
        `;
        captureBtn.disabled = false;
      }, 2000);
    } else {
      captureBtn.textContent = 'Failed';
      setTimeout(() => {
        captureBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
            <polyline points="17 21 17 13 7 13 7 21"></polyline>
            <polyline points="7 3 7 8 15 8"></polyline>
          </svg>
          Save This Page
        `;
        captureBtn.disabled = false;
      }, 2000);
    }
  } catch (error) {
    captureBtn.textContent = 'Failed';
    captureBtn.disabled = false;
  }
}

// Open chat in side panel
async function openChat(): Promise<void> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.windowId) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
    window.close();
  }
}

// Open document (future: could open in side panel or new tab)
function openDocument(documentId: string): void {
  // For now, open chat with a query about this document
  openChat();
}

// Show capture history
async function showHistory(): Promise<void> {
  try {
    const response = await chrome.runtime.sendMessage({
      type: MessageType.GET_CAPTURE_HISTORY,
    });

    if (response.success && response.data) {
      renderHistory(response.data);
    }
  } catch (error) {
    console.error('Failed to load history:', error);
  }
}

function renderHistory(history: Array<{ title: string; url: string; capturedAt: string }>): void {
  if (history.length === 0) {
    searchResults.innerHTML = '<div class="search-result">No capture history</div>';
    searchResults.classList.remove('hidden');
    return;
  }

  searchResults.innerHTML = history
    .slice(0, 10)
    .map(
      (item) => `
      <div class="search-result">
        <div class="search-result-title">${escapeHtml(item.title)}</div>
        <div class="search-result-snippet">${escapeHtml(new Date(item.capturedAt).toLocaleString())}</div>
      </div>
    `
    )
    .join('');

  searchResults.classList.remove('hidden');
}

// Utility
function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
