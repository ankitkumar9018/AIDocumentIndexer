# AIDocumentIndexer Browser Extension

The AIDocumentIndexer Browser Extension allows you to capture web pages, search your knowledge base, and chat with your documents from any browser tab.

## Features

| Feature | Description |
|---------|-------------|
| **Page Capture** | Save any webpage to your knowledge base |
| **Quick Search** | Search popup with keyboard shortcut |
| **Side Panel Chat** | Full RAG chat in browser sidebar |
| **Selection Search** | Highlight text to search instantly |
| **Context Menu** | Right-click actions on any page |
| **Notifications** | Status updates for captures and processing |

---

## Supported Browsers

| Browser | Manifest Version | Status |
|---------|-----------------|--------|
| Chrome | v3 | ✅ Supported |
| Edge | v3 | ✅ Supported |
| Brave | v3 | ✅ Supported |
| Firefox | v2 | ✅ Supported |
| Safari | - | 🚧 Coming Soon |

---

## Installation

### Chrome / Edge / Brave (Development)

```bash
# Build the extension
cd browser-extension
npm install
npm run build

# Load in browser:
# 1. Open chrome://extensions/ (or edge://extensions/)
# 2. Enable "Developer mode" (top right)
# 3. Click "Load unpacked"
# 4. Select the browser-extension/dist/ folder
```

### Firefox (Development)

```bash
# Build Firefox version
cd browser-extension
npm install
npm run build:firefox

# Load in Firefox:
# 1. Open about:debugging
# 2. Click "This Firefox"
# 3. Click "Load Temporary Add-on"
# 4. Select manifest.json from dist-firefox/
```

### Chrome Web Store (Production)

```bash
# Create ZIP for submission
cd browser-extension
npm run build
cd dist
zip -r ../aidocindexer-extension.zip .
```

### Firefox Add-ons (Production)

```bash
# Create ZIP for submission
npm run build:firefox
cd dist-firefox
zip -r ../aidocindexer-firefox.zip .
```

---

## Configuration

### First-Time Setup

1. Click the extension icon in the toolbar
2. Click the **Settings** (gear) icon
3. Enter your server URL: `http://localhost:8000`
4. (Optional) Enter API key if authentication is enabled
5. Click **Test Connection** to verify
6. Click **Save**

### Settings Options

| Setting | Description | Default |
|---------|-------------|---------|
| Server URL | Backend API endpoint | `http://localhost:8000` |
| API Key | Authentication token (optional) | - |
| Auto-Capture | Capture pages automatically | Off |
| Notifications | Show status notifications | On |
| Default Collection | Target collection for captures | Default |

---

## Usage Guide

### 1. Popup (Quick Search)

Click the extension icon or use `Ctrl+Shift+K` (Mac: `Cmd+Shift+K`):

```
┌─────────────────────────────────────┐
│  [Search icon] Search documents...  │
├─────────────────────────────────────┤
│  Recent Searches                    │
│  • machine learning                 │
│  • api documentation                │
│  • authentication flow              │
├─────────────────────────────────────┤
│  [Capture Page]    [Open Chat]      │
└─────────────────────────────────────┘
```

**Features:**
- Type to search across all documents
- Click result to open in new tab
- View recent searches
- Quick capture current page
- Open side panel chat

### 2. Side Panel Chat

Click "Open Chat" or use `Ctrl+Shift+L` (Mac: `Cmd+Shift+L`):

```
┌─────────────────────────────────────┐
│  AIDocumentIndexer Chat    [⚙️] [×] │
├─────────────────────────────────────┤
│                                     │
│  [User]: What is our auth flow?     │
│                                     │
│  [AI]: Based on your documents,     │
│  the authentication flow uses...    │
│                                     │
│  📄 Sources: auth.md, api-spec.pdf  │
│                                     │
├─────────────────────────────────────┤
│  [Type a message...]         [Send] │
└─────────────────────────────────────┘
```

**Features:**
- Full conversation interface
- Streaming responses
- Source citations with links
- Code syntax highlighting
- Conversation history

### 3. Context Menu (Right-Click)

Right-click on any page or selected text:

```
┌─────────────────────────────────────┐
│  AIDocumentIndexer                  │
│  ├── 📄 Save Page to Knowledge Base │
│  ├── 🔍 Search "selected text"      │
│  └── 💬 Ask about "selected text"   │
└─────────────────────────────────────┘
```

**Actions:**
- **Save Page** - Capture entire page content
- **Search Selection** - Search with highlighted text
- **Ask about Selection** - Send selection to chat

### 4. Page Capture

Capture any webpage using:
- Context menu: Right-click → "Save Page to Knowledge Base"
- Popup: Click "Capture Page" button
- Keyboard: `Ctrl+Shift+S` (Mac: `Cmd+Shift+S`)

**Capture Process:**
1. Content script extracts page content
2. HTML converted to clean markdown
3. Sent to backend for processing
4. Notification shown on completion

**What's Captured:**
- Page title and URL
- Main content (article, body)
- Headings and structure
- Code blocks (preserved)
- Tables and lists
- Images (optional)

**What's Excluded:**
- Navigation menus
- Ads and trackers
- Comments sections
- Footers and sidebars (configurable)

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+K` / `Cmd+Shift+K` | Open quick search popup |
| `Ctrl+Shift+S` / `Cmd+Shift+S` | Capture current page |
| `Ctrl+Shift+L` / `Cmd+Shift+L` | Open side panel chat |

### Customize Shortcuts

**Chrome:**
1. Go to `chrome://extensions/shortcuts`
2. Find "AIDocumentIndexer"
3. Click the edit box for any command
4. Press your desired key combination

**Firefox:**
1. Go to `about:addons`
2. Click ⚙️ → "Manage Extension Shortcuts"
3. Edit shortcuts as needed

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKGROUND SERVICE WORKER                 │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │   API Client   │  │    Storage     │  │ Message Router│  │
│  │   (fetch)      │  │   (chrome)     │  │  (events)     │  │
│  └────────────────┘  └────────────────┘  └───────────────┘  │
└───────────────────────────────┬─────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│     POPUP       │   │   SIDE PANEL    │   │    CONTENT      │
│  (quick search) │   │    (chat)       │   │   (capture)     │
│                 │   │                 │   │                 │
│  - Search box   │   │  - Messages     │   │  - DOM access   │
│  - Results list │   │  - Input        │   │  - Selection    │
│  - Quick actions│   │  - Sources      │   │  - Extraction   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Background | `src/background/index.ts` | API calls, message routing |
| Popup | `src/popup/` | Quick search interface |
| Side Panel | `src/sidepanel/` | Full chat interface |
| Content Script | `src/content/capture.ts` | Page content extraction |
| Options | `src/options/` | Settings page |
| Shared | `src/shared/` | Types, utilities, browser polyfill |

---

## Message Protocol

### Content → Background

```typescript
// Capture page content
{
  type: 'capture-page',
  payload: {
    url: string,
    title: string,
    content: string,
    html?: string
  }
}

// Search request
{
  type: 'search',
  payload: {
    query: string,
    limit?: number
  }
}
```

### Background → Popup/Panel

```typescript
// Search results
{
  type: 'search-result',
  payload: {
    results: Document[],
    query: string
  }
}

// Capture status
{
  type: 'capture-status',
  payload: {
    status: 'pending' | 'processing' | 'complete' | 'error',
    documentId?: string,
    error?: string
  }
}
```

---

## API Integration

### Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/search` | POST | Semantic search |
| `/api/v1/chat/message` | POST | Send chat message |
| `/api/v1/upload/web` | POST | Upload web capture |
| `/api/v1/documents` | GET | List documents |

### Example API Calls

```typescript
// Search documents
const response = await fetch(`${serverUrl}/api/v1/search`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  },
  body: JSON.stringify({
    query: 'authentication',
    limit: 10
  })
});

// Upload captured page
const response = await fetch(`${serverUrl}/api/v1/upload/web`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  },
  body: JSON.stringify({
    url: pageUrl,
    title: pageTitle,
    content: pageContent,
    metadata: {
      captured_at: new Date().toISOString()
    }
  })
});
```

---

## Troubleshooting

### Extension Not Loading

```bash
# Rebuild the extension
cd browser-extension
npm run clean
npm run build
```

Then reload in `chrome://extensions/`.

### Connection Errors

1. Check server is running: `curl http://localhost:8000/health`
2. Verify URL in extension settings
3. Check CORS settings on backend
4. Try with API key if auth is enabled

### Capture Not Working

1. Check content script permissions
2. Some pages block content scripts (e.g., Chrome Web Store)
3. Check console for errors: Right-click extension icon → "Inspect popup"

### Firefox-Specific Issues

```bash
# Ensure Firefox build is used
npm run build:firefox

# Check manifest version
cat dist-firefox/manifest.json | grep manifest_version
# Should show: "manifest_version": 2
```

### Debug Mode

1. Open popup/sidepanel
2. Right-click → "Inspect"
3. Check Console for errors
4. Check Network tab for failed requests

---

## Privacy & Permissions

### Permissions Requested

| Permission | Purpose |
|------------|---------|
| `storage` | Save settings locally |
| `activeTab` | Access current tab for capture |
| `contextMenus` | Right-click menu integration |
| `notifications` | Status notifications |
| `scripting` | Execute content scripts |
| `sidePanel` | Side panel UI (Chrome) |

### Data Handling

- **Captured Content** - Sent only to configured server URL
- **Settings** - Stored locally in browser
- **No Tracking** - Zero analytics or telemetry
- **No Third-Party** - No external services

---

## Development

### Project Structure

```
browser-extension/
├── manifest.json           # Chrome manifest (MV3)
├── manifest.firefox.json   # Firefox manifest (MV2)
├── vite.config.ts          # Chrome build config
├── vite.config.firefox.ts  # Firefox build config
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
├── src/
│   ├── background/         # Service worker
│   │   └── index.ts        # Main background script
│   ├── content/            # Content scripts
│   │   └── capture.ts      # Page capture logic
│   ├── popup/              # Popup UI
│   │   ├── index.html
│   │   └── App.tsx
│   ├── sidepanel/          # Side panel UI
│   │   ├── index.html
│   │   └── Panel.tsx
│   ├── options/            # Settings page
│   │   ├── index.html
│   │   └── Options.tsx
│   └── shared/             # Shared code
│       ├── types.ts        # Type definitions
│       ├── api.ts          # API client
│       └── browser-polyfill.ts  # Cross-browser compat
└── assets/
    ├── icon.svg            # Source icon
    └── icons/              # Generated PNGs
```

### Build Commands

```bash
# Development (watch mode)
npm run dev

# Production build (Chrome)
npm run build

# Production build (Firefox)
npm run build:firefox

# Build both
npm run build:all

# Type checking
npm run typecheck

# Linting
npm run lint
```

### Testing

```bash
# Manual testing
# 1. Load extension in browser
# 2. Open any webpage
# 3. Test capture, search, and chat features

# Check service worker
# chrome://serviceworker-internals/
```

---

## FAQ

**Q: Can I use the extension without a backend?**
A: No, the extension requires a running AIDocumentIndexer backend.

**Q: Does it work on all websites?**
A: Most websites. Some (like chrome://, about:, and extension pages) block content scripts.

**Q: How much data is stored locally?**
A: Only settings and recent searches. All documents are on the backend.

**Q: Can I use with multiple servers?**
A: Currently one server at a time. Switch in Settings to change.

**Q: Is Firefox fully supported?**
A: Yes, with Manifest V2. Some features (like side panel) use Firefox's sidebar instead.
