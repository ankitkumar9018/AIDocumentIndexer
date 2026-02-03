/**
 * Browser API Polyfill
 * ====================
 *
 * Provides a unified API for Chrome (chrome.*) and Firefox (browser.*)
 * Extension APIs. Automatically detects the environment and uses the
 * appropriate namespace.
 */

// Detect browser environment
const isFirefox = typeof browser !== 'undefined';
const isChrome = typeof chrome !== 'undefined' && !isFirefox;

// Type definitions for cross-browser compatibility
export interface BrowserAPI {
  storage: typeof chrome.storage;
  runtime: typeof chrome.runtime;
  tabs: typeof chrome.tabs;
  contextMenus: typeof chrome.contextMenus;
  notifications: typeof chrome.notifications;
  commands: typeof chrome.commands;
  scripting?: typeof chrome.scripting;
  sidePanel?: typeof chrome.sidePanel;
}

// Get the appropriate browser API
function getBrowserAPI(): BrowserAPI {
  if (isFirefox) {
    // Firefox uses browser.* namespace with Promises
    return (globalThis as any).browser as BrowserAPI;
  } else if (isChrome) {
    // Chrome uses chrome.* namespace
    return chrome as BrowserAPI;
  }
  throw new Error('Unsupported browser environment');
}

// Export unified browser API
export const browserAPI = getBrowserAPI();

/**
 * Promisified storage operations
 */
export const storage = {
  /**
   * Get items from storage
   */
  async get<T extends Record<string, any>>(keys: string | string[] | null): Promise<T> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.storage.local.get(keys) as Promise<T>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.storage.local.get(keys, (result) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(result as T);
          }
        });
      }
    });
  },

  /**
   * Set items in storage
   */
  async set(items: Record<string, any>): Promise<void> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.storage.local.set(items) as Promise<void>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.storage.local.set(items, () => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve();
          }
        });
      }
    });
  },

  /**
   * Remove items from storage
   */
  async remove(keys: string | string[]): Promise<void> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.storage.local.remove(keys) as Promise<void>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.storage.local.remove(keys, () => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve();
          }
        });
      }
    });
  },

  /**
   * Clear all storage
   */
  async clear(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.storage.local.clear() as Promise<void>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.storage.local.clear(() => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve();
          }
        });
      }
    });
  },
};

/**
 * Promisified tabs operations
 */
export const tabs = {
  /**
   * Get current active tab
   */
  async getCurrent(): Promise<chrome.tabs.Tab | null> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.tabs.query({ active: true, currentWindow: true }) as Promise<chrome.tabs.Tab[]>)
          .then((tabs) => resolve(tabs[0] || null))
          .catch(reject);
      } else {
        browserAPI.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(tabs[0] || null);
          }
        });
      }
    });
  },

  /**
   * Query tabs
   */
  async query(queryInfo: chrome.tabs.QueryInfo): Promise<chrome.tabs.Tab[]> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.tabs.query(queryInfo) as Promise<chrome.tabs.Tab[]>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.tabs.query(queryInfo, (tabs) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(tabs);
          }
        });
      }
    });
  },

  /**
   * Send message to tab
   */
  async sendMessage<T>(tabId: number, message: any): Promise<T> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.tabs.sendMessage(tabId, message) as Promise<T>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.tabs.sendMessage(tabId, message, (response) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(response as T);
          }
        });
      }
    });
  },

  /**
   * Create a new tab
   */
  async create(createProperties: chrome.tabs.CreateProperties): Promise<chrome.tabs.Tab> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.tabs.create(createProperties) as Promise<chrome.tabs.Tab>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.tabs.create(createProperties, (tab) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(tab);
          }
        });
      }
    });
  },
};

/**
 * Promisified runtime operations
 */
export const runtime = {
  /**
   * Send message to background script
   */
  async sendMessage<T>(message: any): Promise<T> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.runtime.sendMessage(message) as Promise<T>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.runtime.sendMessage(message, (response) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(response as T);
          }
        });
      }
    });
  },

  /**
   * Add message listener
   */
  onMessage: browserAPI.runtime.onMessage,

  /**
   * Get extension URL
   */
  getURL(path: string): string {
    return browserAPI.runtime.getURL(path);
  },

  /**
   * Open options page
   */
  openOptionsPage(): void {
    browserAPI.runtime.openOptionsPage();
  },
};

/**
 * Promisified context menu operations
 */
export const contextMenus = {
  /**
   * Create context menu item
   */
  create(createProperties: chrome.contextMenus.CreateProperties): void {
    browserAPI.contextMenus.create(createProperties);
  },

  /**
   * Remove all context menu items
   */
  async removeAll(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.contextMenus.removeAll() as Promise<void>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.contextMenus.removeAll(() => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve();
          }
        });
      }
    });
  },

  /**
   * Add click listener
   */
  onClicked: browserAPI.contextMenus.onClicked,
};

/**
 * Notifications API
 */
export const notifications = {
  /**
   * Create notification
   */
  async create(
    notificationId: string,
    options: chrome.notifications.NotificationOptions<true>
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      if (isFirefox) {
        (browserAPI.notifications.create(notificationId, options) as Promise<string>)
          .then(resolve)
          .catch(reject);
      } else {
        browserAPI.notifications.create(notificationId, options, (id) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(id);
          }
        });
      }
    });
  },
};

/**
 * Commands API
 */
export const commands = {
  /**
   * Add command listener
   */
  onCommand: browserAPI.commands.onCommand,
};

/**
 * Scripting API (Chrome MV3 only, uses tabs.executeScript in Firefox)
 */
export const scripting = {
  /**
   * Execute script in tab
   */
  async executeScript(
    tabId: number,
    details: { func?: () => any; files?: string[] }
  ): Promise<any[]> {
    if (isChrome && browserAPI.scripting) {
      return browserAPI.scripting.executeScript({
        target: { tabId },
        func: details.func,
        files: details.files,
      });
    } else if (isFirefox) {
      // Firefox MV2 uses tabs.executeScript
      return new Promise((resolve, reject) => {
        const executeOptions: any = {};
        if (details.files) {
          executeOptions.file = details.files[0];
        }
        if (details.func) {
          executeOptions.code = `(${details.func.toString()})()`;
        }
        (browserAPI.tabs as any).executeScript(tabId, executeOptions, (results: any[]) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(results);
          }
        });
      });
    }
    throw new Error('Scripting API not available');
  },
};

/**
 * Side panel API (Chrome MV3 only)
 */
export const sidePanel = {
  /**
   * Open side panel
   */
  async open(options?: { tabId?: number; windowId?: number }): Promise<void> {
    if (isChrome && browserAPI.sidePanel) {
      return (browserAPI.sidePanel as any).open(options);
    }
    // Firefox uses sidebar_action, opened via browser action or keyboard shortcut
    console.warn('Side panel API not available in Firefox, use sidebar instead');
  },

  /**
   * Set panel behavior
   */
  async setPanelBehavior(options: { openPanelOnActionClick?: boolean }): Promise<void> {
    if (isChrome && browserAPI.sidePanel) {
      return (browserAPI.sidePanel as any).setPanelBehavior(options);
    }
    // No equivalent in Firefox
  },
};

// Export browser detection
export { isFirefox, isChrome };
