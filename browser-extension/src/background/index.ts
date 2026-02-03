/**
 * AIDocumentIndexer Browser Extension - Background Service Worker
 * =================================================================
 *
 * Handles:
 * - Context menu actions
 * - Page capture
 * - API communication
 * - Side panel management
 */

import { ApiClient } from '../shared/api';
import { MessageType, Message, SearchResult, CapturedPage } from '../shared/types';

// Initialize API client
const api = new ApiClient();

// Context menu IDs
const MENU_CAPTURE_PAGE = 'capture-page';
const MENU_CAPTURE_SELECTION = 'capture-selection';
const MENU_SEARCH_SELECTION = 'search-selection';

// ============================================================================
// Initialization
// ============================================================================

chrome.runtime.onInstalled.addListener(async () => {
  console.log('AIDocumentIndexer extension installed');

  // Create context menus
  chrome.contextMenus.create({
    id: MENU_CAPTURE_PAGE,
    title: 'Save page to AIDocumentIndexer',
    contexts: ['page'],
  });

  chrome.contextMenus.create({
    id: MENU_CAPTURE_SELECTION,
    title: 'Save selection to AIDocumentIndexer',
    contexts: ['selection'],
  });

  chrome.contextMenus.create({
    id: MENU_SEARCH_SELECTION,
    title: 'Search "%s" in AIDocumentIndexer',
    contexts: ['selection'],
  });

  // Initialize storage with defaults
  const settings = await chrome.storage.sync.get([
    'serverUrl',
    'apiKey',
    'autoCapture',
    'captureHistory',
  ]);

  if (!settings.serverUrl) {
    await chrome.storage.sync.set({
      serverUrl: 'http://localhost:8000',
      apiKey: '',
      autoCapture: false,
      captureHistory: [],
    });
  }
});

// ============================================================================
// Context Menu Handlers
// ============================================================================

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;

  switch (info.menuItemId) {
    case MENU_CAPTURE_PAGE:
      await capturePage(tab);
      break;

    case MENU_CAPTURE_SELECTION:
      await captureSelection(tab, info.selectionText || '');
      break;

    case MENU_SEARCH_SELECTION:
      await searchSelection(info.selectionText || '');
      break;
  }
});

// ============================================================================
// Command Handlers
// ============================================================================

chrome.commands.onCommand.addListener(async (command) => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  switch (command) {
    case 'capture_page':
      await capturePage(tab);
      break;

    case 'open_side_panel':
      await chrome.sidePanel.open({ windowId: tab.windowId });
      break;
  }
});

// ============================================================================
// Message Handlers
// ============================================================================

chrome.runtime.onMessage.addListener((message: Message, sender, sendResponse) => {
  handleMessage(message, sender, sendResponse);
  return true; // Indicates async response
});

async function handleMessage(
  message: Message,
  sender: chrome.runtime.MessageSender,
  sendResponse: (response: any) => void
): Promise<void> {
  try {
    switch (message.type) {
      case MessageType.SEARCH:
        const results = await api.search(message.payload.query);
        sendResponse({ success: true, data: results });
        break;

      case MessageType.CHAT:
        const response = await api.chat(
          message.payload.message,
          message.payload.history,
          message.payload.collectionId
        );
        sendResponse({ success: true, data: response });
        break;

      case MessageType.CAPTURE_PAGE:
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab) {
          await capturePage(tab);
          sendResponse({ success: true });
        } else {
          sendResponse({ success: false, error: 'No active tab' });
        }
        break;

      case MessageType.GET_SETTINGS:
        const settings = await chrome.storage.sync.get([
          'serverUrl',
          'apiKey',
          'autoCapture',
        ]);
        sendResponse({ success: true, data: settings });
        break;

      case MessageType.SET_SETTINGS:
        await chrome.storage.sync.set(message.payload);
        await api.updateSettings(message.payload);
        sendResponse({ success: true });
        break;

      case MessageType.GET_COLLECTIONS:
        const collections = await api.getCollections();
        sendResponse({ success: true, data: collections });
        break;

      case MessageType.GET_CAPTURE_HISTORY:
        const history = await getCaptureHistory();
        sendResponse({ success: true, data: history });
        break;

      case MessageType.CHECK_CONNECTION:
        const connected = await api.checkHealth();
        sendResponse({ success: true, data: { connected } });
        break;

      default:
        sendResponse({ success: false, error: 'Unknown message type' });
    }
  } catch (error) {
    console.error('Message handler error:', error);
    sendResponse({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

// ============================================================================
// Page Capture
// ============================================================================

async function capturePage(tab: chrome.tabs.Tab): Promise<void> {
  if (!tab.id || !tab.url) return;

  // Execute content script to extract page content
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      // Extract main content
      const article = document.querySelector('article');
      const main = document.querySelector('main');
      const body = document.body;

      const contentElement = article || main || body;
      const content = contentElement?.innerText || '';

      // Extract metadata
      const title = document.title;
      const description =
        document.querySelector('meta[name="description"]')?.getAttribute('content') || '';
      const author =
        document.querySelector('meta[name="author"]')?.getAttribute('content') || '';

      return {
        title,
        content,
        description,
        author,
        html: document.documentElement.outerHTML,
      };
    },
  });

  if (!result) {
    showNotification('Capture Failed', 'Could not extract page content');
    return;
  }

  const capturedPage: CapturedPage = {
    url: tab.url,
    title: result.title,
    content: result.content,
    description: result.description,
    author: result.author,
    capturedAt: new Date().toISOString(),
  };

  try {
    // Send to server
    await api.capturePage(capturedPage);

    // Save to history
    await addToCaptureHistory(capturedPage);

    // Show success notification
    showNotification('Page Captured', `"${result.title}" saved to your knowledge base`);
  } catch (error) {
    console.error('Capture error:', error);
    showNotification(
      'Capture Failed',
      error instanceof Error ? error.message : 'Unknown error'
    );
  }
}

async function captureSelection(tab: chrome.tabs.Tab, selection: string): Promise<void> {
  if (!selection.trim()) return;

  const capturedPage: CapturedPage = {
    url: tab.url || '',
    title: `Selection from ${tab.title}`,
    content: selection,
    description: '',
    author: '',
    capturedAt: new Date().toISOString(),
  };

  try {
    await api.capturePage(capturedPage);
    await addToCaptureHistory(capturedPage);
    showNotification('Selection Captured', `${selection.slice(0, 50)}... saved`);
  } catch (error) {
    showNotification('Capture Failed', 'Could not save selection');
  }
}

// ============================================================================
// Search
// ============================================================================

async function searchSelection(selection: string): Promise<void> {
  if (!selection.trim()) return;

  // Open side panel with search query
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.windowId) {
    await chrome.sidePanel.open({ windowId: tab.windowId });

    // Send search query to side panel
    setTimeout(() => {
      chrome.runtime.sendMessage({
        type: MessageType.SEARCH_QUERY,
        payload: { query: selection },
      });
    }, 500);
  }
}

// ============================================================================
// Capture History
// ============================================================================

async function getCaptureHistory(): Promise<CapturedPage[]> {
  const { captureHistory = [] } = await chrome.storage.local.get('captureHistory');
  return captureHistory;
}

async function addToCaptureHistory(page: CapturedPage): Promise<void> {
  const history = await getCaptureHistory();

  // Add to front, keep last 100
  history.unshift(page);
  if (history.length > 100) {
    history.pop();
  }

  await chrome.storage.local.set({ captureHistory: history });
}

// ============================================================================
// Notifications
// ============================================================================

function showNotification(title: string, message: string): void {
  chrome.notifications.create({
    type: 'basic',
    iconUrl: chrome.runtime.getURL('assets/icons/icon-128.png'),
    title,
    message,
  });
}

// ============================================================================
// Side Panel
// ============================================================================

chrome.action.onClicked.addListener(async (tab) => {
  // Toggle side panel on action click
  if (tab.windowId) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});

// Export for testing
export { api, capturePage, handleMessage };
