# Mandala Sync CLI

Command-line tool for syncing local folders with Mandala Document Indexer.

## Installation

```bash
# From the cli directory
pip install -e .

# Or install directly
pip install mandala-sync
```

## Quick Start

```bash
# Login to your Mandala server
mandala-sync login --server https://your-server.com

# Add a directory to watch
mandala-sync watch ~/Documents --collection "My Docs"

# Start watching and uploading
mandala-sync start

# Or scan existing files first
mandala-sync start --scan
```

## Commands

### Authentication

```bash
# Login
mandala-sync login --server https://your-server.com

# Check login status
mandala-sync whoami

# Logout
mandala-sync logout
```

### Directory Management

```bash
# Add directory to watch
mandala-sync watch /path/to/folder [OPTIONS]

Options:
  --recursive / --no-recursive  Watch subdirectories (default: yes)
  -c, --collection TEXT         Collection name for uploaded files
  -t, --tier INTEGER            Access tier 1-5 (default: 1)
  -f, --folder TEXT             Target folder ID

# Remove directory from watch list
mandala-sync unwatch /path/to/folder

# List watched directories
mandala-sync list
```

### Watching & Uploading

```bash
# Start watching directories
mandala-sync start

# Start with initial scan of existing files
mandala-sync start --scan

# Check status
mandala-sync status

# Scan and upload existing files in a directory
mandala-sync scan /path/to/folder
```

### Configuration

```bash
# View all settings
mandala-sync config get

# View specific setting
mandala-sync config get default_collection

# Set a value
mandala-sync config set default_collection "Reports"
mandala-sync config set default_access_tier 2
```

## Configuration File

Settings are stored in `~/.mandala-sync/config.yaml`:

```yaml
server: https://your-server.com
directories:
  - path: /home/user/Documents
    recursive: true
    collection: Documents
    access_tier: 1
    folder_id: null
    enabled: true
settings:
  auto_start: false
  default_collection: null
  default_access_tier: 1
```

## Supported File Types

- Documents: PDF, DOCX, DOC, ODT, RTF
- Presentations: PPTX, PPT, ODP, KEY
- Spreadsheets: XLSX, XLS, CSV, ODS
- Images: PNG, JPG, JPEG, TIFF, BMP, WEBP
- Text: TXT, MD, RST, HTML, XML, JSON
- Email: EML, MSG
- Archives: ZIP

## Running as a Service

### Linux (systemd)

Create `/etc/systemd/system/mandala-sync.service`:

```ini
[Unit]
Description=Mandala Sync Service
After=network.target

[Service]
Type=simple
User=your-username
ExecStart=/usr/local/bin/mandala-sync start
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable mandala-sync
sudo systemctl start mandala-sync
```

### macOS (launchd)

Create `~/Library/LaunchAgents/com.mandala.sync.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mandala.sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/mandala-sync</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.mandala.sync.plist
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```
