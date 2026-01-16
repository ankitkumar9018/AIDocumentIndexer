# Desktop Client Applications Architecture

## Overview

Three client applications for watching local folders on users' computers and uploading documents to the server:

1. **Electron Desktop App** - Full-featured desktop application
2. **CLI Tool** - Command-line utility for automation
3. **Browser Extension** - Monitor download folders

---

## 1. Electron Desktop App

### Features
- System tray application (runs in background)
- Full web app experience in desktop window
- Local folder watching with file upload
- Offline queue (uploads when connection restored)
- Native OS notifications
- Auto-start on login (optional)
- Cross-platform: Windows, macOS, Linux

### Tech Stack
- Electron + React (same frontend as web app)
- electron-builder for packaging
- electron-store for local settings
- chokidar for file watching
- axios for API uploads

### Project Structure
```
desktop-clients/electron/
├── package.json
├── electron-builder.json
├── src/
│   ├── main/                  # Electron main process
│   │   ├── index.ts           # Entry point
│   │   ├── tray.ts            # System tray
│   │   ├── watcher.ts         # File watching service
│   │   ├── uploader.ts        # Upload queue & API
│   │   ├── store.ts           # Local storage
│   │   └── auto-launch.ts     # Auto-start
│   ├── preload/               # Preload scripts
│   │   └── index.ts
│   └── renderer/              # React app (shared with web)
│       └── ... (uses frontend/ as base)
├── assets/
│   └── icons/
└── build/
```

### Key Components

#### Main Process (watcher.ts)
```typescript
class LocalFileWatcher {
  private watchers: Map<string, FSWatcher> = new Map();
  private uploadQueue: FileUploadItem[] = [];

  addDirectory(config: WatchConfig): void;
  removeDirectory(path: string): void;
  processQueue(): Promise<void>;
  getStatus(): WatcherStatus;
}
```

#### Upload Queue (uploader.ts)
```typescript
class UploadService {
  private queue: UploadItem[] = [];
  private apiClient: APIClient;

  async uploadFile(item: UploadItem): Promise<void>;
  retryFailed(): void;
  pauseUploads(): void;
  resumeUploads(): void;
}
```

### Settings Stored Locally
- Server URL
- API token (encrypted)
- Watched directories list
- Default collection/folder/access tier per directory
- Auto-start preference
- Upload queue (persisted for offline)

---

## 2. CLI Tool

### Features
- Watch directories from command line
- Run as daemon/service
- Configuration via file or flags
- Status commands
- Cross-platform

### Tech Stack
- Python (can reuse existing watcher code)
- Click for CLI
- watchdog for file watching
- requests for API uploads

### Project Structure
```
desktop-clients/cli/
├── pyproject.toml
├── mandala_sync/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py           # CLI commands
│   ├── watcher.py       # File watching
│   ├── uploader.py      # API uploads
│   ├── config.py        # Configuration
│   └── daemon.py        # Background service
└── README.md
```

### Commands
```bash
# Login/configure
mandala-sync login --server https://your-server.com
mandala-sync config set default_collection "My Docs"

# Watch directories
mandala-sync watch /path/to/folder --recursive --collection "Reports"
mandala-sync unwatch /path/to/folder
mandala-sync list                    # List watched directories

# Status & control
mandala-sync status                  # Show status
mandala-sync start                   # Start daemon
mandala-sync stop                    # Stop daemon
mandala-sync queue                   # Show upload queue
mandala-sync retry                   # Retry failed uploads
```

### Config File (~/.mandala-sync/config.yaml)
```yaml
server: https://your-server.com
token: encrypted-token-here
directories:
  - path: /home/user/Documents
    recursive: true
    collection: Documents
    access_tier: 1
    folder_id: null
  - path: /home/user/Reports
    recursive: false
    collection: Reports
    access_tier: 2
```

---

## 3. Browser Extension

### Features
- Monitor browser downloads folder
- Auto-upload downloaded files matching patterns
- Quick upload via right-click context menu
- Popup showing recent uploads
- Support Chrome, Firefox, Edge

### Tech Stack
- WebExtension API (cross-browser)
- React for popup UI
- Native messaging (optional for folder access)

### Project Structure
```
desktop-clients/extension/
├── manifest.json
├── src/
│   ├── background.ts      # Service worker
│   ├── popup/             # Popup React app
│   ├── options/           # Options page
│   └── content/           # Content scripts
├── assets/
└── build/
```

### Manifest Permissions
```json
{
  "permissions": [
    "downloads",
    "storage",
    "notifications",
    "contextMenus"
  ],
  "host_permissions": [
    "https://your-server.com/*"
  ]
}
```

---

## API Endpoints for Desktop Clients

The server already has these endpoints that clients will use:

### Upload
- `POST /api/v1/upload/file` - Upload single file
- `POST /api/v1/upload/batch` - Upload multiple files

### Authentication
- `POST /api/v1/auth/login` - Get token
- `POST /api/v1/auth/refresh` - Refresh token
- `GET /api/v1/auth/me` - Verify token

### Collections/Folders
- `GET /api/v1/documents/collections/list` - List collections
- `GET /api/v1/folders` - List folders
- `GET /api/v1/admin/access-tiers` - List access tiers

### Status (optional sync)
- `POST /api/v1/watcher/sync` - Sync local watcher config (new endpoint)
- `GET /api/v1/watcher/client-config` - Get client config (new endpoint)

---

## Implementation Priority

1. **CLI Tool** - Quickest to implement, uses existing Python code
2. **Electron App** - Most feature-complete, best UX
3. **Browser Extension** - Limited by browser APIs

---

## Security Considerations

1. **Token Storage**: Use OS keychain (Electron) or encrypted file (CLI)
2. **HTTPS Only**: Enforce TLS for all API communication
3. **File Validation**: Check file types before upload
4. **Rate Limiting**: Respect server rate limits
5. **Local Encryption**: Encrypt sensitive config data

---

## Getting Started

### CLI Tool (Fastest to implement)
```bash
cd desktop-clients/cli
pip install -e .
mandala-sync login --server https://localhost:8000
mandala-sync watch ~/Documents
```

### Electron App
```bash
cd desktop-clients/electron
npm install
npm run dev
```

### Browser Extension
```bash
cd desktop-clients/extension
npm install
npm run build
# Load unpacked extension in browser
```
