# AIDocumentIndexer Connectors

Connectors allow AIDocumentIndexer to sync data from external services into your knowledge base.

## Available Connectors

| Connector | Data Types | Auth Method | Incremental Sync |
|-----------|------------|-------------|------------------|
| [Notion](#notion) | Pages, Databases | OAuth 2.0 / API Token | ✅ |
| [GitHub](#github) | Repos, Issues, PRs | OAuth 2.0 / PAT | ✅ |
| [Slack](#slack) | Messages, Threads, Files | OAuth 2.0 / Bot Token | ✅ |
| [Google Drive](#google-drive) | Files, Google Docs | OAuth 2.0 | ✅ |
| [Gmail / IMAP](#email) | Emails, Attachments | OAuth 2.0 / IMAP | ✅ |
| [Confluence](#confluence) | Pages, Spaces | API Token | ✅ |
| [Jira](#jira) | Issues, Comments | API Token | ✅ |

---

## Notion

Sync pages and databases from Notion workspaces.

### Setup

1. **Create Integration**
   - Go to [Notion Integrations](https://www.notion.so/my-integrations)
   - Click "New integration"
   - Give it a name and select workspace
   - Copy the Internal Integration Token

2. **Share Pages with Integration**
   - Open pages/databases you want to sync
   - Click "..." → "Add connections" → Select your integration

3. **Configure in AIDocumentIndexer**
   ```python
   from backend.connectors.notion import create_notion_connector

   connector = create_notion_connector(
       api_token="secret_xxx",
       database_ids=["db_id_1", "db_id_2"],  # Optional
   )
   ```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_token` | str | Required | Notion API token |
| `database_ids` | list | [] | Specific databases to sync |
| `include_pages` | bool | True | Sync standalone pages |
| `include_databases` | bool | True | Sync database items |
| `max_depth` | int | 3 | Nested page depth limit |
| `sync_interval_minutes` | int | 60 | Sync frequency |

### What's Synced

- Page titles and content
- Database properties and values
- Block content (text, code, lists, etc.)
- Child pages (up to max_depth)
- Last modified timestamps

### Usage

```python
# Fetch all pages
async for page in connector.fetch_pages():
    print(f"Page: {page.title}")
    print(f"Content: {page.content[:200]}...")

# Fetch database items
async for item in connector.fetch_database_items("database_id"):
    print(f"Item: {item.properties}")

# Incremental sync (only changed pages)
async for page in connector.fetch_pages(modified_since=last_sync):
    # Process changed pages
    pass
```

---

## GitHub

Sync repositories, issues, and pull requests from GitHub.

### Setup

1. **Create Personal Access Token**
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token (classic) with scopes:
     - `repo` - Full repository access
     - `read:org` - Read org membership (optional)

2. **Configure Connector**
   ```python
   from backend.connectors.github import create_github_connector

   connector = create_github_connector(
       access_token="ghp_xxx",
       repositories=["owner/repo1", "owner/repo2"],
   )
   ```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `access_token` | str | Required | GitHub PAT |
| `repositories` | list | [] | Repos to sync (owner/repo) |
| `include_issues` | bool | True | Sync issues |
| `include_prs` | bool | True | Sync pull requests |
| `include_readme` | bool | True | Sync README files |
| `include_code` | bool | False | Sync source files |
| `file_extensions` | list | [".md", ".rst"] | File types to sync |
| `max_file_size_kb` | int | 500 | Max file size |
| `sync_interval_minutes` | int | 30 | Sync frequency |

### What's Synced

- Repository metadata
- README and documentation
- Issues with comments
- Pull requests with reviews
- Selected source files
- Labels and milestones

### Usage

```python
# List repositories
async for repo in connector.list_repositories():
    print(f"Repo: {repo.full_name}")

# Fetch issues
async for issue in connector.fetch_issues("owner/repo"):
    print(f"Issue #{issue.number}: {issue.title}")

# Fetch repository content
async for file in connector.fetch_repository_files("owner/repo"):
    print(f"File: {file.path}")
```

---

## Slack

Sync channel messages, threads, and files from Slack workspaces.

### Setup

1. **Create Slack App**
   - Go to [Slack API Apps](https://api.slack.com/apps)
   - Create new app "From scratch"
   - Add Bot Token Scopes:
     - `channels:history` - View messages in public channels
     - `channels:read` - View channel info
     - `files:read` - View files
     - `users:read` - View user info
   - Install to workspace and copy Bot Token

2. **Invite Bot to Channels**
   - In each channel: `/invite @YourBotName`

3. **Configure Connector**
   ```python
   from backend.connectors.slack import create_slack_connector

   connector = create_slack_connector(
       bot_token="xoxb-xxx",
       channel_ids=["C123", "C456"],  # Optional
   )
   ```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bot_token` | str | Required | Slack Bot Token |
| `channel_ids` | list | [] | Channels to sync (empty = all) |
| `include_threads` | bool | True | Sync thread replies |
| `include_files` | bool | True | Sync shared files |
| `include_private` | bool | False | Sync private channels |
| `max_messages_per_channel` | int | 10000 | Message limit |
| `days_to_sync` | int | 90 | Historical messages |
| `sync_interval_minutes` | int | 15 | Sync frequency |

### What's Synced

- Channel messages with metadata
- Thread replies
- User mentions (resolved to names)
- Reactions and timestamps
- Shared files (text content)
- Channel descriptions

### Usage

```python
# List channels
async for channel in connector.list_channels():
    print(f"Channel: #{channel.name}")

# Fetch messages
async for message in connector.fetch_channel_messages("C123"):
    print(f"{message.user_name}: {message.text}")

# Fetch thread replies
async for reply in connector.fetch_thread_replies("C123", "1234567890.123456"):
    print(f"Reply: {reply.text}")

# Incremental sync
async for message in connector.fetch_channel_messages("C123", oldest=last_sync):
    # Process new messages
    pass
```

---

## Google Drive

Sync files and Google Docs from Google Drive.

### Setup

1. **Create Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create new project
   - Enable Google Drive API
   - Create OAuth 2.0 credentials
   - Set redirect URI (e.g., `http://localhost:8000/callback`)

2. **Get OAuth Token**
   - Complete OAuth flow to get access_token and refresh_token

3. **Configure Connector**
   ```python
   from backend.connectors.google_drive import create_google_drive_connector

   connector = create_google_drive_connector(
       access_token="ya29.xxx",
       refresh_token="1//xxx",
       client_id="xxx.apps.googleusercontent.com",
       client_secret="xxx",
   )
   ```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `access_token` | str | Required | OAuth access token |
| `refresh_token` | str | Recommended | For token refresh |
| `client_id` | str | Required | OAuth client ID |
| `client_secret` | str | Required | OAuth client secret |
| `folder_ids` | list | [] | Specific folders (empty = all) |
| `include_shared` | bool | True | Include shared files |
| `include_trashed` | bool | False | Include trash |
| `file_types` | list | [] | Filter by MIME type |
| `max_file_size_mb` | int | 100 | Max file size |
| `sync_interval_minutes` | int | 60 | Sync frequency |

### What's Synced

- Files (PDF, DOCX, TXT, etc.)
- Google Docs (exported as text)
- Google Sheets (exported as CSV)
- Google Slides (exported as text)
- File metadata and permissions
- Folder structure

### Usage

```python
# List files
async for file in connector.list_files():
    print(f"File: {file.name} ({file.mime_type})")

# Get file content
content = await connector.get_file_content(file)
print(f"Content: {content[:500]}...")

# Watch for changes
async for change in connector.get_changes(start_page_token):
    print(f"Changed: {change.name}")
```

---

## Email

Sync emails via Gmail API or IMAP.

### Gmail API Setup

1. **Enable Gmail API**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Add scope: `https://www.googleapis.com/auth/gmail.readonly`

2. **Configure Connector**
   ```python
   from backend.connectors.email import create_gmail_connector

   connector = create_gmail_connector(
       access_token="ya29.xxx",
       labels=["INBOX", "IMPORTANT"],
       days_to_sync=30,
   )
   ```

### IMAP Setup

```python
from backend.connectors.email import create_imap_connector

connector = create_imap_connector(
    host="imap.example.com",
    username="user@example.com",
    password="app-specific-password",
    port=993,
    use_ssl=True,
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `labels` / `folders` | list | ["INBOX"] | Labels/folders to sync |
| `include_sent` | bool | False | Include sent mail |
| `include_attachments` | bool | True | Download attachments |
| `max_attachment_size_mb` | int | 25 | Max attachment size |
| `max_emails` | int | 1000 | Email limit |
| `days_to_sync` | int | 30 | Historical emails |
| `sync_interval_minutes` | int | 60 | Sync frequency |

### What's Synced

- Email subjects and bodies
- Sender and recipient info
- Dates and labels/folders
- Attachments (text content)
- Thread information

### Usage

```python
# Fetch emails (Gmail)
async for email in gmail_connector.fetch_emails():
    print(f"From: {email.from_address}")
    print(f"Subject: {email.subject}")
    print(f"Body: {email.content[:200]}...")

# Fetch emails (IMAP)
emails = imap_connector.fetch_emails(folder="INBOX")
for email in emails:
    print(f"Subject: {email.subject}")

# Download attachment
content = await gmail_connector.get_attachment(
    message_id=email.id,
    attachment_id=email.attachments[0]["id"]
)
```

---

## Confluence

Sync pages and spaces from Atlassian Confluence.

### Setup

1. **Create API Token**
   - Go to [Atlassian Account](https://id.atlassian.com/manage/api-tokens)
   - Create API token

2. **Configure Connector**
   ```python
   from backend.connectors.confluence import create_confluence_connector

   connector = create_confluence_connector(
       base_url="https://yourcompany.atlassian.net/wiki",
       username="email@example.com",
       api_token="xxx",
       space_keys=["DOCS", "ENG"],  # Optional
   )
   ```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base_url` | str | Required | Confluence URL |
| `username` | str | Required | Email (Cloud) |
| `api_token` | str | Required | API token |
| `is_cloud` | bool | True | Cloud vs Server/DC |
| `space_keys` | list | [] | Spaces to sync |
| `include_archived` | bool | False | Archived pages |
| `include_attachments` | bool | True | Page attachments |
| `max_attachment_size_mb` | int | 50 | Max attachment |
| `max_pages` | int | 10000 | Page limit |
| `cql_filter` | str | None | Additional CQL |
| `sync_interval_minutes` | int | 60 | Sync frequency |

### What's Synced

- Page titles and content
- Page hierarchy (parents/children)
- Labels and metadata
- Attachments
- Space information
- Version history

### Usage

```python
# List spaces
async for space in connector.list_spaces():
    print(f"Space: {space.key} - {space.name}")

# Fetch pages
async for page in connector.fetch_pages(space_key="DOCS"):
    print(f"Page: {page.title}")
    print(f"Content: {page.content[:200]}...")

# Search
async for page in connector.search("authentication guide"):
    print(f"Found: {page.title}")

# Incremental sync
async for page in connector.fetch_pages(modified_since=last_sync):
    # Process changed pages
    pass
```

---

## Jira

Sync issues and comments from Atlassian Jira.

### Setup

1. **Create API Token**
   - Go to [Atlassian Account](https://id.atlassian.com/manage/api-tokens)
   - Create API token

2. **Configure Connector**
   ```python
   from backend.connectors.jira import create_jira_connector

   connector = create_jira_connector(
       base_url="https://yourcompany.atlassian.net",
       username="email@example.com",
       api_token="xxx",
       project_keys=["PROJ", "DEV"],  # Optional
   )
   ```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base_url` | str | Required | Jira URL |
| `username` | str | Required | Email (Cloud) |
| `api_token` | str | Required | API token |
| `is_cloud` | bool | True | Cloud vs Server/DC |
| `project_keys` | list | [] | Projects to sync |
| `issue_types` | list | [] | Issue types filter |
| `statuses` | list | [] | Status filter |
| `include_comments` | bool | True | Sync comments |
| `include_attachments` | bool | True | Sync attachments |
| `max_issues` | int | 10000 | Issue limit |
| `jql_filter` | str | None | Custom JQL |
| `sync_interval_minutes` | int | 30 | Sync frequency |

### What's Synced

- Issue summary and description
- Status, priority, type
- Assignee and reporter
- Comments with authors
- Attachments (text content)
- Labels and components
- Linked issues
- Fix versions

### Usage

```python
# List projects
async for project in connector.list_projects():
    print(f"Project: {project.key} - {project.name}")

# Fetch issues
async for issue in connector.fetch_issues(project_key="PROJ"):
    print(f"{issue.key}: {issue.summary}")
    print(f"Status: {issue.status}")
    print(f"Description: {issue.description[:200]}...")

# Custom JQL query
async for issue in connector.fetch_issues(
    jql='project = PROJ AND status = "In Progress"'
):
    print(f"In Progress: {issue.key}")

# Search
async for issue in connector.search_issues("authentication bug"):
    print(f"Found: {issue.key} - {issue.summary}")
```

---

## Common Patterns

### Incremental Sync

All connectors support incremental sync to fetch only changed items:

```python
# Store last sync timestamp
last_sync = datetime.now(timezone.utc)

# Later, sync only changes
async for item in connector.fetch_items(modified_since=last_sync):
    process(item)

# Update timestamp
last_sync = datetime.now(timezone.utc)
```

### Error Handling

```python
from backend.connectors.base import ConnectorError

try:
    async for item in connector.fetch_items():
        process(item)
except ConnectorError as e:
    logger.error(f"Connector error: {e}")
    # Handle rate limits, auth errors, etc.
```

### Pagination

Connectors automatically handle pagination. Just iterate:

```python
# This handles all pages automatically
async for item in connector.fetch_all_items():
    process(item)
```

### Rate Limiting

All connectors implement rate limiting to avoid API throttling:

```python
# Built-in rate limiting
connector = create_connector(
    rate_limit_per_second=10,  # If supported
)
```

---

## Creating Custom Connectors

To create a new connector:

```python
from backend.connectors.base import BaseConnector, ConnectorConfig

class MyConnectorConfig(ConnectorConfig):
    api_key: str
    workspace_id: str
    max_items: int = 1000

class MyConnector(BaseConnector):
    def __init__(self, config: MyConnectorConfig):
        self.config = config
        self._client = None

    async def test_connection(self) -> bool:
        # Test API connection
        pass

    async def fetch_items(self) -> AsyncGenerator[MyItem, None]:
        # Fetch and yield items
        pass

def create_my_connector(api_key: str, workspace_id: str) -> MyConnector:
    config = MyConnectorConfig(
        api_key=api_key,
        workspace_id=workspace_id
    )
    return MyConnector(config)
```

---

## Troubleshooting

### Authentication Errors

- Verify API token/credentials are correct
- Check token hasn't expired
- Ensure required scopes/permissions are granted

### Rate Limiting

- Reduce `sync_interval_minutes`
- Implement exponential backoff
- Use incremental sync instead of full sync

### Missing Data

- Check permissions on source items
- Verify connector has access to resources
- Check filtering options aren't too restrictive

### Timeout Errors

- Increase timeout settings
- Reduce batch sizes
- Use pagination

---

## Security Best Practices

1. **Store credentials securely** - Use environment variables or secrets manager
2. **Use least privilege** - Request only necessary permissions
3. **Rotate tokens regularly** - Update API tokens periodically
4. **Audit access** - Monitor connector activity
5. **Encrypt in transit** - Always use HTTPS
