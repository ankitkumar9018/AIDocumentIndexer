# Connector System

## Overview

Connectors allow syncing documents from external sources (Google Drive, Notion, Confluence, custom servers, etc.) into the document index.

```mermaid
flowchart TD
    subgraph "External Sources"
        GD[Google Drive]
        NO[Notion]
        CF[Confluence]
        CS[Custom Server]
    end

    subgraph "Connector Layer"
        SCHED[Sync Scheduler<br/>periodic / manual]
        DISC[Resource Discovery<br/>list files & changes]
        DL[Content Download]
    end

    subgraph "Storage Decision"
        MODE{Storage Mode?}
        DOWNLOAD_MODE[Download & Store<br/>is_stored_locally=true]
        PROCESS_MODE[Process Only<br/>is_stored_locally=false<br/>keep link]
    end

    subgraph "Pipeline"
        PIPE[Document Pipeline<br/>parse → chunk → embed]
        DB[(PostgreSQL<br/>Document + SyncedResource)]
    end

    GD --> SCHED
    NO --> SCHED
    CF --> SCHED
    CS --> SCHED
    SCHED --> DISC
    DISC --> DL
    DL --> MODE

    MODE -->|download| DOWNLOAD_MODE
    MODE -->|process_only| PROCESS_MODE

    DOWNLOAD_MODE --> PIPE
    PROCESS_MODE --> PIPE
    PIPE --> DB
```

## Storage Modes

```mermaid
flowchart LR
    subgraph "Download & Store (default)"
        D1[File saved to disk]
        D2[Full preview available]
        D3[Works offline]
        D4[Uses storage space]
    end

    subgraph "Process Only"
        P1[File deleted after processing]
        P2[External preview only]
        P3[Requires network]
        P4[Minimal storage]
    end
```

### Storage Mode Resolution

```mermaid
flowchart TD
    Q{Per-connector<br/>override set?}
    Q -->|Yes, not 'global_default'| USE_CONNECTOR[Use connector's<br/>storage_mode]
    Q -->|No or 'global_default'| GLOBAL{Global setting<br/>connector.storage_mode?}
    GLOBAL -->|Set| USE_GLOBAL[Use global setting]
    GLOBAL -->|Not set| DEFAULT[Default: 'download']
```

## Sync Flow

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant C as Connector (e.g., GDrive)
    participant DB as PostgreSQL
    participant P as Pipeline

    S->>C: list_resources()
    C-->>S: Resource list (name, id, modified_at)

    S->>DB: Check SyncedResource table
    DB-->>S: Last sync timestamps

    loop For each new/modified resource
        S->>C: download_resource(id)
        C-->>S: File content (bytes)

        S->>DB: Check storage mode (per-connector → global)

        alt Download Mode
            S->>S: Save file to storage/documents/uuid/
            S->>DB: Create Document (is_stored_locally=true)
        else Process Only Mode
            S->>S: Save temp file
            S->>DB: Create Document (is_stored_locally=false)
        end

        S->>P: Trigger processing pipeline
        P->>P: Parse → Chunk → Embed → Index

        alt Process Only
            S->>S: Delete temp file from disk
        end

        S->>DB: Create/Update SyncedResource record
    end
```

## Source Metadata

Every document tracks its provenance:

```json
{
  "source_url": "https://drive.google.com/file/d/abc123",
  "source_type": "google_drive",
  "is_stored_locally": false,
  "upload_source_info": {
    "upload_method": "connector_sync",
    "connector_type": "google_drive",
    "connector_name": "Team Drive",
    "external_id": "abc123",
    "external_path": "/Shared/Reports/Q4.pdf",
    "external_url": "https://drive.google.com/file/d/abc123",
    "synced_at": "2024-01-15T10:30:00Z",
    "storage_mode": "process_only"
  }
}
```

## External Document Preview

When a document is stored externally (`is_stored_locally=false`):

```mermaid
flowchart TD
    DOC[External Document] --> CHECK{Source Type?}

    CHECK -->|Google Drive| GDOCS[Google Docs Viewer<br/>iframe embed]
    CHECK -->|Direct PDF URL| IFRAME[Direct iframe<br/>embed attempt]
    CHECK -->|Other| PLACEHOLDER[Placeholder with<br/>summary + Open in Source]

    GDOCS --> WORKS{iframe<br/>loads?}
    IFRAME --> WORKS
    WORKS -->|Yes| PREVIEW[Show embedded preview]
    WORKS -->|No| FALLBACK[Show placeholder<br/>+ error message]

    PLACEHOLDER --> ACTIONS
    FALLBACK --> ACTIONS
    PREVIEW --> ACTIONS

    ACTIONS[User Actions]
    ACTIONS --> OPEN[🔗 Open in Source<br/>opens external URL]
    ACTIONS --> IMPORT[💾 Import Copy<br/>downloads & stores locally]
```

### Import Copy Endpoint

`POST /documents/{doc_id}/import-local`

Downloads the file from `source_url`, saves to local storage, and sets `is_stored_locally=true`. After import, the standard local preview takes over.

## Admin Settings

### Global Storage Settings (Admin > Settings > Ingestion)

| Setting | Options | Description |
|---------|---------|-------------|
| Default Storage Mode | Download / Process Only | How new connector syncs store files |
| Store Source Metadata | On / Off | Track upload provenance |

### Per-Connector Override (Connector Settings)

| Option | Description |
|--------|-------------|
| Use Global Default | Follow the global setting |
| Download & Store | Always store files locally |
| Process Only | Never store files, keep links |

### Storage Stats (Admin > Settings > Ingestion)

Shows breakdown of:
- Local vs external document counts
- Storage size by category
- Per-source-type breakdown (e.g., google_drive: 22 docs, 450 MB, 12 external)
