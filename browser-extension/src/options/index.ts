/**
 * AIDocumentIndexer Options Page Script
 * ======================================
 *
 * Manages extension settings and configuration.
 */

import { MessageType, ExtensionSettings } from '../shared/types';

// Default settings
const DEFAULT_SETTINGS: ExtensionSettings = {
  serverUrl: 'http://localhost:8000',
  apiKey: '',
  autoCapture: false,
  defaultCollectionId: '',
  excludedDomains: ['google.com', 'facebook.com', 'twitter.com', 'localhost'],
  resultsCount: 5,
  showCitations: true,
};

// DOM Elements
const serverUrlInput = document.getElementById('server-url') as HTMLInputElement;
const apiKeyInput = document.getElementById('api-key') as HTMLInputElement;
const testConnectionBtn = document.getElementById('test-connection') as HTMLButtonElement;
const connectionStatus = document.getElementById('connection-status') as HTMLSpanElement;
const autoCaptureCheckbox = document.getElementById('auto-capture') as HTMLInputElement;
const defaultCollectionSelect = document.getElementById('default-collection') as HTMLSelectElement;
const excludeDomainsTextarea = document.getElementById('exclude-domains') as HTMLTextAreaElement;
const resultsCountSelect = document.getElementById('results-count') as HTMLSelectElement;
const showCitationsCheckbox = document.getElementById('show-citations') as HTMLInputElement;
const clearHistoryBtn = document.getElementById('clear-history') as HTMLButtonElement;
const exportDataBtn = document.getElementById('export-data') as HTMLButtonElement;
const saveSettingsBtn = document.getElementById('save-settings') as HTMLButtonElement;
const resetSettingsBtn = document.getElementById('reset-settings') as HTMLButtonElement;
const shortcutsLink = document.getElementById('shortcuts-link') as HTMLAnchorElement;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  await loadCollections();
  setupEventListeners();
});

// Load settings from storage
async function loadSettings(): Promise<void> {
  const result = await chrome.storage.sync.get('settings');
  const settings: ExtensionSettings = { ...DEFAULT_SETTINGS, ...result.settings };

  serverUrlInput.value = settings.serverUrl;
  apiKeyInput.value = settings.apiKey;
  autoCaptureCheckbox.checked = settings.autoCapture;
  defaultCollectionSelect.value = settings.defaultCollectionId;
  excludeDomainsTextarea.value = settings.excludedDomains.join('\n');
  resultsCountSelect.value = settings.resultsCount.toString();
  showCitationsCheckbox.checked = settings.showCitations;
}

// Save settings to storage
async function saveSettings(): Promise<void> {
  const settings: ExtensionSettings = {
    serverUrl: serverUrlInput.value.trim().replace(/\/$/, ''),
    apiKey: apiKeyInput.value.trim(),
    autoCapture: autoCaptureCheckbox.checked,
    defaultCollectionId: defaultCollectionSelect.value,
    excludedDomains: excludeDomainsTextarea.value
      .split('\n')
      .map((d) => d.trim().toLowerCase())
      .filter((d) => d.length > 0),
    resultsCount: parseInt(resultsCountSelect.value, 10),
    showCitations: showCitationsCheckbox.checked,
  };

  await chrome.storage.sync.set({ settings });
  showToast('Settings saved!', 'success');
}

// Load collections from server
async function loadCollections(): Promise<void> {
  try {
    const response = await chrome.runtime.sendMessage({
      type: MessageType.GET_COLLECTIONS,
    });

    if (response.success && response.data) {
      // Clear existing options (except the first "No default")
      while (defaultCollectionSelect.options.length > 1) {
        defaultCollectionSelect.remove(1);
      }

      // Add collection options
      response.data.forEach((collection: { id: string; name: string }) => {
        const option = document.createElement('option');
        option.value = collection.id;
        option.textContent = collection.name;
        defaultCollectionSelect.appendChild(option);
      });

      // Restore selected value
      const result = await chrome.storage.sync.get('settings');
      if (result.settings?.defaultCollectionId) {
        defaultCollectionSelect.value = result.settings.defaultCollectionId;
      }
    }
  } catch (error) {
    console.error('Failed to load collections:', error);
  }
}

// Test connection to server
async function testConnection(): Promise<void> {
  connectionStatus.className = 'status loading';
  connectionStatus.textContent = 'Testing...';

  try {
    const response = await chrome.runtime.sendMessage({
      type: MessageType.CHECK_CONNECTION,
      payload: { serverUrl: serverUrlInput.value.trim() },
    });

    if (response.success && response.data.connected) {
      connectionStatus.className = 'status success';
      connectionStatus.textContent = 'Connected!';

      // Reload collections after successful connection
      await loadCollections();
    } else {
      connectionStatus.className = 'status error';
      connectionStatus.textContent = response.error || 'Connection failed';
    }
  } catch (error) {
    connectionStatus.className = 'status error';
    connectionStatus.textContent = 'Connection failed';
  }
}

// Clear capture history
async function clearHistory(): Promise<void> {
  if (!confirm('Are you sure you want to clear your capture history? This cannot be undone.')) {
    return;
  }

  await chrome.storage.local.remove('captureHistory');
  showToast('Capture history cleared', 'success');
}

// Export settings
async function exportSettings(): Promise<void> {
  const result = await chrome.storage.sync.get('settings');
  const settings = result.settings || DEFAULT_SETTINGS;

  // Remove sensitive data
  const exportData = {
    ...settings,
    apiKey: settings.apiKey ? '***' : '',
    exportedAt: new Date().toISOString(),
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = 'aidocindexer-settings.json';
  a.click();

  URL.revokeObjectURL(url);
  showToast('Settings exported', 'success');
}

// Reset to default settings
async function resetSettings(): Promise<void> {
  if (!confirm('Are you sure you want to reset all settings to defaults?')) {
    return;
  }

  await chrome.storage.sync.set({ settings: DEFAULT_SETTINGS });
  await loadSettings();
  showToast('Settings reset to defaults', 'success');
}

// Setup event listeners
function setupEventListeners(): void {
  testConnectionBtn.addEventListener('click', testConnection);
  clearHistoryBtn.addEventListener('click', clearHistory);
  exportDataBtn.addEventListener('click', exportSettings);
  saveSettingsBtn.addEventListener('click', saveSettings);
  resetSettingsBtn.addEventListener('click', resetSettings);

  // Handle shortcuts link click (since chrome:// URLs can't be linked directly)
  shortcutsLink.addEventListener('click', (e) => {
    e.preventDefault();
    chrome.tabs.create({ url: 'chrome://extensions/shortcuts' });
  });

  // Auto-save on input changes (debounced)
  let saveTimeout: ReturnType<typeof setTimeout>;
  const autoSave = () => {
    clearTimeout(saveTimeout);
    saveTimeout = setTimeout(() => {
      saveSettings();
    }, 1000);
  };

  serverUrlInput.addEventListener('input', autoSave);
  apiKeyInput.addEventListener('input', autoSave);
  autoCaptureCheckbox.addEventListener('change', autoSave);
  defaultCollectionSelect.addEventListener('change', autoSave);
  excludeDomainsTextarea.addEventListener('input', autoSave);
  resultsCountSelect.addEventListener('change', autoSave);
  showCitationsCheckbox.addEventListener('change', autoSave);
}

// Show toast notification
function showToast(message: string, type: 'success' | 'error' = 'success'): void {
  // Remove existing toast
  const existingToast = document.querySelector('.toast');
  if (existingToast) {
    existingToast.remove();
  }

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3000);
}
