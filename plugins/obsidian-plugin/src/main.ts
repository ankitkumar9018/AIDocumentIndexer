/**
 * AIDocumentIndexer Obsidian Plugin
 * ==================================
 *
 * Integrates Obsidian with AIDocumentIndexer for:
 * - Semantic search across your vault
 * - RAG-powered AI chat about your notes
 * - Automatic sync of vault to knowledge base
 * - Knowledge graph visualization
 */

import {
  App,
  Editor,
  MarkdownView,
  Modal,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  TFile,
  TFolder,
  debounce,
  requestUrl,
  RequestUrlParam,
} from "obsidian";

// =============================================================================
// Types & Interfaces
// =============================================================================

interface AIDocIndexerSettings {
  serverUrl: string;
  apiKey: string;
  autoSync: boolean;
  syncOnSave: boolean;
  syncInterval: number; // minutes
  excludeFolders: string[];
  excludePatterns: string[];
  defaultCollection: string;
  showChatInSidebar: boolean;
}

interface SearchResult {
  id: string;
  documentId: string;
  content: string;
  score: number;
  title: string;
  metadata: {
    path?: string;
    source?: string;
  };
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

const DEFAULT_SETTINGS: AIDocIndexerSettings = {
  serverUrl: "http://localhost:8000",
  apiKey: "",
  autoSync: false,
  syncOnSave: true,
  syncInterval: 60,
  excludeFolders: [".obsidian", ".trash"],
  excludePatterns: [],
  defaultCollection: "obsidian-vault",
  showChatInSidebar: true,
};

// =============================================================================
// Main Plugin Class
// =============================================================================

export default class AIDocIndexerPlugin extends Plugin {
  settings: AIDocIndexerSettings;
  private syncDebounced: ReturnType<typeof debounce>;
  private chatHistory: ChatMessage[] = [];

  async onload() {
    await this.loadSettings();

    // Create debounced sync function
    this.syncDebounced = debounce(
      (file: TFile) => this.syncFile(file),
      5000,
      true
    );

    // Add ribbon icon for quick search
    this.addRibbonIcon("search", "AIDocIndexer Search", () => {
      new SearchModal(this.app, this).open();
    });

    // Add ribbon icon for chat
    this.addRibbonIcon("message-circle", "AIDocIndexer Chat", () => {
      new ChatModal(this.app, this).open();
    });

    // Add command: Search
    this.addCommand({
      id: "aidocindexer-search",
      name: "Search with AI",
      callback: () => {
        new SearchModal(this.app, this).open();
      },
    });

    // Add command: Chat
    this.addCommand({
      id: "aidocindexer-chat",
      name: "Chat with AI",
      callback: () => {
        new ChatModal(this.app, this).open();
      },
    });

    // Add command: Sync current file
    this.addCommand({
      id: "aidocindexer-sync-file",
      name: "Sync current file to knowledge base",
      editorCallback: async (editor: Editor, view: MarkdownView) => {
        const file = view.file;
        if (file) {
          await this.syncFile(file);
          new Notice(`Synced: ${file.name}`);
        }
      },
    });

    // Add command: Sync entire vault
    this.addCommand({
      id: "aidocindexer-sync-vault",
      name: "Sync entire vault to knowledge base",
      callback: async () => {
        await this.syncVault();
      },
    });

    // Add command: Ask about selection
    this.addCommand({
      id: "aidocindexer-ask-selection",
      name: "Ask AI about selection",
      editorCallback: async (editor: Editor) => {
        const selection = editor.getSelection();
        if (selection) {
          const modal = new ChatModal(this.app, this);
          modal.initialQuery = `Explain this: "${selection}"`;
          modal.open();
        } else {
          new Notice("Please select some text first");
        }
      },
    });

    // Register file change events for auto-sync
    if (this.settings.syncOnSave) {
      this.registerEvent(
        this.app.vault.on("modify", (file) => {
          if (file instanceof TFile && this.shouldSync(file)) {
            this.syncDebounced(file);
          }
        })
      );
    }

    // Add settings tab
    this.addSettingTab(new AIDocIndexerSettingTab(this.app, this));

    // Register sidebar view if enabled
    if (this.settings.showChatInSidebar) {
      this.registerView("aidocindexer-chat-view", (leaf) => {
        return new ChatView(leaf, this);
      });
    }
  }

  onunload() {
    // Cleanup
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  // ===========================================================================
  // API Methods
  // ===========================================================================

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.settings.apiKey) {
      headers["Authorization"] = `Bearer ${this.settings.apiKey}`;
    }
    return headers;
  }

  async testConnection(): Promise<boolean> {
    try {
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/health`,
        method: "GET",
      });
      return response.status === 200;
    } catch (e) {
      console.error("Connection test failed:", e);
      return false;
    }
  }

  async search(query: string, limit: number = 10): Promise<SearchResult[]> {
    try {
      const response = await requestUrl({
        url: `${this.settings.serverUrl}/api/v1/search`,
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({
          query,
          limit,
          collection_id: this.settings.defaultCollection,
        }),
      });

      if (response.status === 200) {
        return response.json.results || [];
      }
      return [];
    } catch (e) {
      console.error("Search failed:", e);
      new Notice("Search failed. Check your connection settings.");
      return [];
    }
  }

  async chat(message: string): Promise<string> {
    try {
      // Add user message to history
      this.chatHistory.push({ role: "user", content: message });

      const response = await requestUrl({
        url: `${this.settings.serverUrl}/api/v1/chat/message`,
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({
          message,
          collection_id: this.settings.defaultCollection,
          history: this.chatHistory.slice(-10), // Send last 10 messages
        }),
      });

      if (response.status === 200) {
        const answer = response.json.answer || response.json.response || "";
        const sources = response.json.sources || [];

        // Add assistant response to history
        this.chatHistory.push({
          role: "assistant",
          content: answer,
          sources,
        });

        return answer;
      }

      throw new Error(`HTTP ${response.status}`);
    } catch (e) {
      console.error("Chat failed:", e);
      new Notice("Chat failed. Check your connection settings.");
      return "I'm sorry, I couldn't process your request. Please check your connection.";
    }
  }

  clearChatHistory() {
    this.chatHistory = [];
  }

  // ===========================================================================
  // Sync Methods
  // ===========================================================================

  shouldSync(file: TFile): boolean {
    // Check excluded folders
    for (const folder of this.settings.excludeFolders) {
      if (file.path.startsWith(folder + "/")) {
        return false;
      }
    }

    // Check excluded patterns
    for (const pattern of this.settings.excludePatterns) {
      if (new RegExp(pattern).test(file.path)) {
        return false;
      }
    }

    // Only sync markdown files
    return file.extension === "md";
  }

  async syncFile(file: TFile): Promise<boolean> {
    try {
      const content = await this.app.vault.read(file);
      const metadata = this.app.metadataCache.getFileCache(file);

      // Prepare document data
      const documentData = {
        title: file.basename,
        content: content,
        metadata: {
          path: file.path,
          folder: file.parent?.path,
          created: file.stat.ctime,
          modified: file.stat.mtime,
          tags: metadata?.tags?.map((t) => t.tag) || [],
          frontmatter: metadata?.frontmatter || {},
          source: "obsidian",
        },
      };

      const response = await requestUrl({
        url: `${this.settings.serverUrl}/api/v1/upload/web`,
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({
          url: `obsidian://${file.path}`,
          title: file.basename,
          content: content,
          collection_id: this.settings.defaultCollection,
          metadata: documentData.metadata,
        }),
      });

      return response.status === 200 || response.status === 201;
    } catch (e) {
      console.error(`Failed to sync ${file.path}:`, e);
      return false;
    }
  }

  async syncVault(): Promise<void> {
    new Notice("Starting vault sync...");

    const files = this.app.vault.getMarkdownFiles();
    const filesToSync = files.filter((f) => this.shouldSync(f));

    let synced = 0;
    let failed = 0;

    for (const file of filesToSync) {
      const success = await this.syncFile(file);
      if (success) {
        synced++;
      } else {
        failed++;
      }

      // Progress update every 10 files
      if ((synced + failed) % 10 === 0) {
        new Notice(`Syncing: ${synced + failed}/${filesToSync.length}`);
      }
    }

    new Notice(`Sync complete: ${synced} synced, ${failed} failed`);
  }
}

// =============================================================================
// Search Modal
// =============================================================================

class SearchModal extends Modal {
  plugin: AIDocIndexerPlugin;
  private searchInput: HTMLInputElement;
  private resultsContainer: HTMLElement;

  constructor(app: App, plugin: AIDocIndexerPlugin) {
    super(app);
    this.plugin = plugin;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("aidocindexer-modal");

    // Title
    contentEl.createEl("h2", { text: "AIDocIndexer Search" });

    // Search input
    const inputContainer = contentEl.createDiv("search-input-container");
    this.searchInput = inputContainer.createEl("input", {
      type: "text",
      placeholder: "Search your knowledge base...",
    });
    this.searchInput.addClass("aidocindexer-search-input");

    // Search button
    const searchButton = inputContainer.createEl("button", { text: "Search" });
    searchButton.addClass("mod-cta");

    // Results container
    this.resultsContainer = contentEl.createDiv("search-results");

    // Event handlers
    searchButton.onclick = () => this.performSearch();
    this.searchInput.onkeydown = (e) => {
      if (e.key === "Enter") {
        this.performSearch();
      }
    };

    // Focus input
    this.searchInput.focus();
  }

  async performSearch() {
    const query = this.searchInput.value.trim();
    if (!query) return;

    this.resultsContainer.empty();
    this.resultsContainer.createEl("p", { text: "Searching..." });

    const results = await this.plugin.search(query);

    this.resultsContainer.empty();

    if (results.length === 0) {
      this.resultsContainer.createEl("p", { text: "No results found." });
      return;
    }

    for (const result of results) {
      const resultEl = this.resultsContainer.createDiv("search-result");

      // Title
      const titleEl = resultEl.createEl("h4", { text: result.title });
      titleEl.addClass("search-result-title");

      // Score
      const scoreEl = resultEl.createEl("span", {
        text: `Score: ${(result.score * 100).toFixed(1)}%`,
      });
      scoreEl.addClass("search-result-score");

      // Content preview
      const contentEl = resultEl.createEl("p", {
        text: result.content.slice(0, 200) + "...",
      });
      contentEl.addClass("search-result-content");

      // Click to open
      if (result.metadata?.path) {
        resultEl.addClass("clickable");
        resultEl.onclick = async () => {
          const file = this.app.vault.getAbstractFileByPath(
            result.metadata.path!
          );
          if (file instanceof TFile) {
            await this.app.workspace.getLeaf().openFile(file);
            this.close();
          }
        };
      }
    }
  }

  onClose() {
    const { contentEl } = this;
    contentEl.empty();
  }
}

// =============================================================================
// Chat Modal
// =============================================================================

class ChatModal extends Modal {
  plugin: AIDocIndexerPlugin;
  initialQuery?: string;
  private chatInput: HTMLTextAreaElement;
  private chatContainer: HTMLElement;

  constructor(app: App, plugin: AIDocIndexerPlugin) {
    super(app);
    this.plugin = plugin;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("aidocindexer-modal", "aidocindexer-chat-modal");

    // Header
    const header = contentEl.createDiv("chat-header");
    header.createEl("h2", { text: "AIDocIndexer Chat" });

    const clearButton = header.createEl("button", { text: "Clear History" });
    clearButton.onclick = () => {
      this.plugin.clearChatHistory();
      this.chatContainer.empty();
      new Notice("Chat history cleared");
    };

    // Chat container
    this.chatContainer = contentEl.createDiv("chat-container");

    // Input area
    const inputArea = contentEl.createDiv("chat-input-area");

    this.chatInput = inputArea.createEl("textarea", {
      placeholder: "Ask a question about your notes...",
    });
    this.chatInput.addClass("aidocindexer-chat-input");

    const sendButton = inputArea.createEl("button", { text: "Send" });
    sendButton.addClass("mod-cta");

    // Event handlers
    sendButton.onclick = () => this.sendMessage();
    this.chatInput.onkeydown = (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    };

    // If there's an initial query, send it
    if (this.initialQuery) {
      this.chatInput.value = this.initialQuery;
      this.sendMessage();
    }

    this.chatInput.focus();
  }

  async sendMessage() {
    const message = this.chatInput.value.trim();
    if (!message) return;

    // Add user message to UI
    this.addMessage("user", message);
    this.chatInput.value = "";

    // Show typing indicator
    const typingEl = this.chatContainer.createDiv("chat-message assistant");
    typingEl.createEl("p", { text: "Thinking..." });

    // Get response
    const response = await this.plugin.chat(message);

    // Remove typing indicator and add response
    typingEl.remove();
    this.addMessage("assistant", response);

    // Scroll to bottom
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  addMessage(role: "user" | "assistant", content: string) {
    const messageEl = this.chatContainer.createDiv(`chat-message ${role}`);

    const roleLabel = messageEl.createEl("strong", {
      text: role === "user" ? "You" : "AI",
    });

    // Render markdown content
    const contentEl = messageEl.createDiv("message-content");
    contentEl.innerHTML = this.renderMarkdown(content);
  }

  renderMarkdown(content: string): string {
    // Basic markdown rendering
    return content
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br>");
  }

  onClose() {
    const { contentEl } = this;
    contentEl.empty();
  }
}

// =============================================================================
// Chat View (Sidebar)
// =============================================================================

import { ItemView, WorkspaceLeaf } from "obsidian";

class ChatView extends ItemView {
  plugin: AIDocIndexerPlugin;

  constructor(leaf: WorkspaceLeaf, plugin: AIDocIndexerPlugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType(): string {
    return "aidocindexer-chat-view";
  }

  getDisplayText(): string {
    return "AIDocIndexer Chat";
  }

  getIcon(): string {
    return "message-circle";
  }

  async onOpen() {
    const container = this.containerEl.children[1];
    container.empty();
    container.createEl("h4", { text: "AIDocIndexer Chat" });
    container.createEl("p", { text: "Use the chat command to start a conversation." });
  }

  async onClose() {
    // Cleanup
  }
}

// =============================================================================
// Settings Tab
// =============================================================================

class AIDocIndexerSettingTab extends PluginSettingTab {
  plugin: AIDocIndexerPlugin;

  constructor(app: App, plugin: AIDocIndexerPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    containerEl.createEl("h2", { text: "AIDocumentIndexer Settings" });

    // Connection settings
    containerEl.createEl("h3", { text: "Connection" });

    new Setting(containerEl)
      .setName("Server URL")
      .setDesc("URL of your AIDocumentIndexer server")
      .addText((text) =>
        text
          .setPlaceholder("http://localhost:8000")
          .setValue(this.plugin.settings.serverUrl)
          .onChange(async (value) => {
            this.plugin.settings.serverUrl = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("API Key")
      .setDesc("API key for authentication (optional)")
      .addText((text) => {
        text
          .setPlaceholder("Enter API key")
          .setValue(this.plugin.settings.apiKey)
          .onChange(async (value) => {
            this.plugin.settings.apiKey = value;
            await this.plugin.saveSettings();
          });
        text.inputEl.type = "password";
      });

    new Setting(containerEl)
      .setName("Test Connection")
      .setDesc("Verify connection to server")
      .addButton((button) =>
        button.setButtonText("Test").onClick(async () => {
          const connected = await this.plugin.testConnection();
          new Notice(connected ? "Connected!" : "Connection failed");
        })
      );

    // Sync settings
    containerEl.createEl("h3", { text: "Sync" });

    new Setting(containerEl)
      .setName("Default Collection")
      .setDesc("Collection name for synced notes")
      .addText((text) =>
        text
          .setPlaceholder("obsidian-vault")
          .setValue(this.plugin.settings.defaultCollection)
          .onChange(async (value) => {
            this.plugin.settings.defaultCollection = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Sync on Save")
      .setDesc("Automatically sync files when saved")
      .addToggle((toggle) =>
        toggle
          .setValue(this.plugin.settings.syncOnSave)
          .onChange(async (value) => {
            this.plugin.settings.syncOnSave = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Excluded Folders")
      .setDesc("Folders to exclude from sync (comma-separated)")
      .addText((text) =>
        text
          .setPlaceholder(".obsidian, .trash")
          .setValue(this.plugin.settings.excludeFolders.join(", "))
          .onChange(async (value) => {
            this.plugin.settings.excludeFolders = value
              .split(",")
              .map((s) => s.trim())
              .filter((s) => s);
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Sync Vault Now")
      .setDesc("Sync all files to knowledge base")
      .addButton((button) =>
        button.setButtonText("Sync Vault").onClick(async () => {
          await this.plugin.syncVault();
        })
      );
  }
}
