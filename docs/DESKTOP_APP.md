# AIDocumentIndexer Desktop App

The AIDocumentIndexer Desktop App is a cross-platform application built with Tauri 2.0 that provides full RAG functionality with optional offline mode.

## Features

### Operating Modes

#### LOCAL MODE (Offline)
- **No internet required** - All processing happens on your device
- **Local LLM** - Powered by Ollama (llama3.2, mistral, etc.)
- **Local Embeddings** - nomic-embed-text for semantic search
- **SQLite Storage** - Documents stored locally
- **Privacy First** - Your data never leaves your machine

#### SERVER MODE (Connected)
- **Connect to backend** - Use your existing AIDocumentIndexer server
- **Full features** - Access all web app capabilities
- **Multi-user** - Share knowledge base with team
- **Cloud sync** - Documents synchronized across devices

### Key Capabilities

| Feature | LOCAL MODE | SERVER MODE |
|---------|------------|-------------|
| Document Upload | ✅ | ✅ |
| Semantic Search | ✅ | ✅ |
| RAG Chat | ✅ | ✅ |
| Knowledge Graph | ❌ | ✅ |
| Multi-user | ❌ | ✅ |
| Offline Support | ✅ | ❌ |
| System Tray | ✅ | ✅ |
| File Watching | ✅ | ✅ |

---

## Installation

### Prerequisites

1. **Rust** (for building from source)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Node.js 18+**
   ```bash
   # macOS
   brew install node

   # Ubuntu
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

3. **Ollama** (for LOCAL MODE)
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Windows
   # Download from https://ollama.com/download
   ```

### Build from Source

```bash
# Clone repository
git clone https://github.com/your-repo/AIDocumentIndexer.git
cd AIDocumentIndexer/desktop-tauri

# Install dependencies
npm install

# Development mode
npm run tauri:dev

# Production build
npm run tauri:build
```

### Build Output

| Platform | Location |
|----------|----------|
| macOS | `src-tauri/target/release/bundle/dmg/*.dmg` |
| Windows | `src-tauri/target/release/bundle/msi/*.msi` |
| Linux | `src-tauri/target/release/bundle/deb/*.deb` |
| Linux (AppImage) | `src-tauri/target/release/bundle/appimage/*.AppImage` |

---

## First-Time Setup

### 1. Choose Operating Mode

On first launch, you'll be prompted to choose between:

- **LOCAL MODE** - For offline, privacy-focused usage
- **SERVER MODE** - For team collaboration with a backend

### 2. LOCAL MODE Setup

1. **Start Ollama**
   ```bash
   ollama serve
   ```

2. **Pull Required Models**
   ```bash
   # Chat model
   ollama pull llama3.2

   # Embedding model
   ollama pull nomic-embed-text
   ```

3. **Select Models in Settings**
   - Chat Model: `llama3.2` (recommended) or `mistral`, `phi3`
   - Embedding Model: `nomic-embed-text`

### 3. SERVER MODE Setup

1. Enter your server URL: `http://localhost:8000` or your deployed instance
2. (Optional) Enter API key if authentication is enabled
3. Test connection to verify

---

## User Interface

### Main Navigation

```
┌─────────────────────────────────────────────────┐
│  [Logo]  AIDocumentIndexer      [Settings] [─] │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────┐                                    │
│  │  Chat   │  ← Main chat interface             │
│  ├─────────┤                                    │
│  │ Search  │  ← Document search                 │
│  ├─────────┤                                    │
│  │  Docs   │  ← Document management             │
│  ├─────────┤                                    │
│  │Settings │  ← Configuration                   │
│  └─────────┘                                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Chat Page

- **Message Input** - Type your questions
- **Streaming Responses** - See AI responses as they generate
- **Source Citations** - View which documents were used
- **Code Highlighting** - Syntax-highlighted code blocks
- **Copy/Regenerate** - Message action buttons

### Search Page

- **Semantic Search** - Natural language queries
- **Filters** - File type, date range, folder
- **Results Preview** - Expandable snippets
- **Quick Actions** - Open, view in chat, delete

### Documents Page

- **Upload Area** - Drag and drop or click to browse
- **Document List** - All indexed documents
- **Status Indicators** - Processing state
- **Bulk Actions** - Select multiple for batch operations

### Settings Page

#### Mode Settings
- Switch between LOCAL and SERVER modes
- Configure server URL and API key

#### LOCAL MODE Settings
- **Ollama URL**: Default `http://localhost:11434`
- **Chat Model**: Select from available Ollama models
- **Embedding Model**: Select embedding model
- **Refresh Models**: Fetch latest from Ollama

#### Processing Settings
- **Chunk Size**: Default 1000 tokens
- **Chunk Overlap**: Default 200 tokens
- **Max File Size**: Maximum upload size

#### Appearance
- **Theme**: System, Light, or Dark
- **Compact Mode**: Reduce UI spacing

---

## File Watching

The desktop app can automatically index files from watched folders.

### Setup Watched Folders

1. Go to **Settings** → **File Watching**
2. Click **Add Folder**
3. Select a folder to watch
4. Configure options:
   - **Recursive**: Include subfolders
   - **File Types**: Filter by extension
   - **Ignore Patterns**: Skip certain files

### How It Works

```
Watched Folder
    │
    ├── New file detected
    │   └── Automatically queued for indexing
    │
    ├── File modified
    │   └── Re-indexed with new content
    │
    └── File deleted
        └── Removed from index
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd/Ctrl + N` | New chat |
| `Cmd/Ctrl + K` | Quick search |
| `Cmd/Ctrl + U` | Upload files |
| `Cmd/Ctrl + ,` | Open settings |
| `Cmd/Ctrl + Enter` | Send message |
| `Escape` | Close modal/dialog |

---

## System Tray

When minimized, the app runs in the system tray:

- **Click** - Open main window
- **Right-click** - Show menu
  - Quick Search
  - New Chat
  - Settings
  - Quit

---

## API Reference (LOCAL MODE)

The desktop app exposes Tauri commands for interacting with the local backend:

### Chat

```typescript
import { invoke } from '@tauri-apps/api/core';

// Send a chat message
const response = await invoke('chat_local', {
  message: 'What is the architecture?',
  collectionId: 'default'
});
```

### Search

```typescript
// Search documents
const results = await invoke('search_local', {
  query: 'authentication',
  limit: 10
});
```

### Documents

```typescript
// Index a file
await invoke('index_file', {
  path: '/path/to/document.pdf'
});

// List documents
const docs = await invoke('list_documents', {
  collectionId: 'default'
});
```

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Check available models
ollama list
```

### Build Errors

```bash
# Clean and rebuild
cd desktop-tauri
cargo clean
npm run tauri:build
```

### Performance Issues

- **Slow embeddings**: Use smaller model (`nomic-embed-text`)
- **Slow chat**: Use faster model (`phi3`, `mistral`)
- **High memory**: Reduce chunk size in settings

### Debug Mode

```bash
# Run with debug logging
RUST_LOG=debug npm run tauri:dev
```

---

## Data Storage

### LOCAL MODE Data Locations

| Data | Location |
|------|----------|
| Documents | `~/.aidocindexer/documents/` |
| Database | `~/.aidocindexer/data.db` |
| Embeddings | `~/.aidocindexer/embeddings/` |
| Config | `~/.aidocindexer/config.json` |
| Logs | `~/.aidocindexer/logs/` |

### Backup

```bash
# Backup all data
cp -r ~/.aidocindexer ~/aidocindexer-backup

# Restore
cp -r ~/aidocindexer-backup ~/.aidocindexer
```

---

## Performance Benchmarks

| Operation | LOCAL MODE | SERVER MODE |
|-----------|------------|-------------|
| Document Upload (10MB) | 2-5s | 1-3s |
| Embedding Generation | 100-500ms/chunk | 50-200ms/chunk |
| Search Query | 50-200ms | 20-100ms |
| Chat Response | 2-10s | 1-5s |

*Benchmarks on M1 MacBook Pro with llama3.2*

---

## Security

### LOCAL MODE Security

- **No network calls** (except Ollama if local)
- **Data encrypted at rest** (SQLCipher option)
- **No telemetry** - Zero data collection
- **Sandboxed** - Tauri security model

### SERVER MODE Security

- **TLS/HTTPS** - Encrypted communication
- **JWT Authentication** - Secure tokens
- **API Key** - Optional additional auth

---

## Updating

### Auto-Update (Coming Soon)

The app will check for updates on launch and notify you when available.

### Manual Update

```bash
cd desktop-tauri
git pull origin main
npm install
npm run tauri:build
```

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

### Development Setup

```bash
# Install Tauri CLI
cargo install tauri-cli

# Run in development
npm run tauri:dev

# Hot reload frontend
npm run dev
```

---

## FAQ

**Q: Can I use both LOCAL and SERVER modes?**
A: Yes, you can switch between modes in Settings. However, documents are not automatically synced between modes.

**Q: What Ollama models work best?**
A: For chat: `llama3.2` (best quality) or `phi3` (fastest). For embeddings: `nomic-embed-text`.

**Q: How much disk space do I need?**
A: The app itself is 2-3MB. Ollama models require 2-8GB each. Document storage varies by content.

**Q: Can I run without Ollama?**
A: Yes, use SERVER MODE to connect to a backend with cloud LLM providers.
