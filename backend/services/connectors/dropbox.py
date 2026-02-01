"""
AIDocumentIndexer - Dropbox Connector
======================================

Connector for syncing content from Dropbox.

Supports:
- Files and folders
- Shared folders
- Paper documents
- Change detection via cursor
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio

import structlog

from backend.services.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorType,
    Resource,
    ResourceType,
    Change,
    ChangeType,
)
from backend.services.connectors.registry import ConnectorRegistry

logger = structlog.get_logger(__name__)

# Check for Dropbox SDK
try:
    import dropbox
    from dropbox.files import FileMetadata, FolderMetadata, DeletedMetadata
    from dropbox.exceptions import ApiError, AuthError
    HAS_DROPBOX = True
except ImportError:
    HAS_DROPBOX = False
    logger.info("Dropbox SDK not available - install with: pip install dropbox")


@ConnectorRegistry.register(ConnectorType.DROPBOX)
class DropboxConnector(BaseConnector):
    """
    Connector for Dropbox cloud storage.

    Uses the Dropbox API to sync files and folders.
    Supports OAuth 2.0 authentication.
    """

    connector_type = ConnectorType.DROPBOX
    display_name = "Dropbox"
    description = "Sync files and folders from Dropbox"
    icon = "dropbox"

    # File extensions to sync by default
    DEFAULT_EXTENSIONS = [
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".txt", ".md", ".rtf", ".csv", ".json", ".xml",
        ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ]

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[Any] = None
        self._cursor: Optional[str] = None

    async def _get_client(self):
        """Get or create Dropbox client."""
        if not HAS_DROPBOX:
            raise RuntimeError("Dropbox SDK not installed. Run: pip install dropbox")

        if self._client is None:
            access_token = self.config.credentials.get("access_token", "")
            if not access_token:
                raise ValueError("Dropbox access_token is required")

            self._client = dropbox.Dropbox(access_token)

        return self._client

    async def authenticate(self) -> bool:
        """Authenticate with Dropbox API."""
        try:
            client = await self._get_client()

            # Run in executor since dropbox SDK is sync
            loop = asyncio.get_running_loop()
            account = await loop.run_in_executor(
                None, client.users_get_current_account
            )

            self._authenticated = True
            self.log_info(
                "Authenticated with Dropbox",
                user=account.name.display_name,
                email=account.email,
            )
            return True

        except AuthError as e:
            self.log_error("Dropbox authentication failed", error=str(e))
            return False
        except Exception as e:
            self.log_error("Dropbox authentication error", error=str(e))
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh OAuth tokens."""
        # Dropbox uses long-lived tokens or refresh tokens
        # Implement refresh logic if using short-lived tokens
        refresh_token = self.config.credentials.get("refresh_token")
        if refresh_token:
            # Would call Dropbox OAuth to refresh
            pass
        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List files and folders from Dropbox."""
        client = await self._get_client()
        resources = []

        try:
            loop = asyncio.get_running_loop()
            path = folder_id or ""

            if page_token:
                # Continue from cursor
                result = await loop.run_in_executor(
                    None,
                    lambda: client.files_list_folder_continue(page_token)
                )
            else:
                # Start fresh listing
                result = await loop.run_in_executor(
                    None,
                    lambda: client.files_list_folder(
                        path,
                        recursive=False,
                        limit=min(page_size, 2000),
                    )
                )

            for entry in result.entries:
                resource = self._parse_entry(entry)
                if resource:
                    resources.append(resource)

            # Return cursor for pagination
            next_token = result.cursor if result.has_more else None

            return resources, next_token

        except ApiError as e:
            self.log_error("Dropbox list failed", error=str(e))
            return [], None
        except Exception as e:
            self.log_error("Failed to list Dropbox resources", error=str(e))
            return [], None

    def _parse_entry(self, entry) -> Optional[Resource]:
        """Parse a Dropbox entry into a Resource."""
        try:
            if isinstance(entry, FolderMetadata):
                return Resource(
                    id=entry.id,
                    name=entry.name,
                    resource_type=ResourceType.FOLDER,
                    path=entry.path_display,
                    mime_type="application/vnd.dropbox.folder",
                    size=0,
                    created_at=None,
                    modified_at=None,
                    metadata={
                        "path_lower": entry.path_lower,
                        "shared_folder_id": getattr(entry, "shared_folder_id", None),
                    },
                )

            elif isinstance(entry, FileMetadata):
                return Resource(
                    id=entry.id,
                    name=entry.name,
                    resource_type=ResourceType.FILE,
                    path=entry.path_display,
                    mime_type=self._get_mime_type(entry.name),
                    size=entry.size,
                    created_at=None,  # Dropbox doesn't track creation time
                    modified_at=entry.server_modified,
                    metadata={
                        "path_lower": entry.path_lower,
                        "rev": entry.rev,
                        "content_hash": entry.content_hash,
                        "is_downloadable": entry.is_downloadable,
                    },
                )

            return None

        except Exception as e:
            self.log_error("Failed to parse Dropbox entry", error=str(e))
            return None

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type from filename."""
        extension = filename.lower().split(".")[-1] if "." in filename else ""
        mime_types = {
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "txt": "text/plain",
            "md": "text/markdown",
            "csv": "text/csv",
            "json": "application/json",
            "xml": "application/xml",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "svg": "image/svg+xml",
        }
        return mime_types.get(extension, "application/octet-stream")

    async def download_resource(self, resource: Resource) -> bytes:
        """Download file content from Dropbox."""
        client = await self._get_client()

        if resource.resource_type == ResourceType.FOLDER:
            # Return folder metadata as content
            return f"Folder: {resource.name}\nPath: {resource.path}".encode("utf-8")

        try:
            loop = asyncio.get_running_loop()
            path = resource.metadata.get("path_lower", resource.path)

            metadata, response = await loop.run_in_executor(
                None,
                lambda: client.files_download(path)
            )

            return response.content

        except ApiError as e:
            self.log_error("Dropbox download failed", error=str(e))
            raise ValueError(f"Failed to download: {resource.path}")

    async def get_changes(
        self,
        since: Optional[datetime] = None,
        page_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """Get changes since last sync using Dropbox cursor."""
        client = await self._get_client()
        changes = []

        try:
            loop = asyncio.get_running_loop()

            # Use stored cursor or get a new one
            cursor = page_token or self._cursor

            if cursor:
                result = await loop.run_in_executor(
                    None,
                    lambda: client.files_list_folder_continue(cursor)
                )
            else:
                # Get initial cursor
                result = await loop.run_in_executor(
                    None,
                    lambda: client.files_list_folder_get_latest_cursor("")
                )
                self._cursor = result.cursor
                return [], None

            for entry in result.entries:
                change = self._parse_change(entry)
                if change:
                    changes.append(change)

            # Update cursor
            self._cursor = result.cursor
            next_token = result.cursor if result.has_more else None

            return changes, next_token

        except Exception as e:
            self.log_error("Failed to get Dropbox changes", error=str(e))
            return [], None

    def _parse_change(self, entry) -> Optional[Change]:
        """Parse a Dropbox entry into a Change."""
        try:
            if isinstance(entry, DeletedMetadata):
                return Change(
                    resource_id=entry.path_lower,
                    change_type=ChangeType.DELETED,
                    timestamp=datetime.utcnow(),
                    metadata={"path": entry.path_display},
                )

            resource = self._parse_entry(entry)
            if resource:
                # Determine if created or modified based on metadata
                return Change(
                    resource_id=resource.id,
                    change_type=ChangeType.MODIFIED,
                    timestamp=resource.modified_at or datetime.utcnow(),
                    resource=resource,
                )

            return None

        except Exception as e:
            self.log_error("Failed to parse Dropbox change", error=str(e))
            return None

    async def close(self):
        """Close the connector."""
        self._client = None
        self._cursor = None

    @classmethod
    def get_oauth_url(
        cls,
        client_id: str,
        redirect_uri: str,
        state: str,
    ) -> str:
        """Generate Dropbox OAuth authorization URL."""
        return (
            f"https://www.dropbox.com/oauth2/authorize"
            f"?client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
            f"&state={state}"
            f"&token_access_type=offline"
        )

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "sync_path": {
                    "type": "string",
                    "title": "Sync Path",
                    "description": "Dropbox path to sync (empty = root)",
                    "default": "",
                },
                "file_extensions": {
                    "type": "array",
                    "title": "File Extensions",
                    "description": "Extensions to sync (empty = defaults)",
                    "items": {"type": "string"},
                    "default": [],
                },
                "include_shared": {
                    "type": "boolean",
                    "title": "Include Shared",
                    "description": "Include shared folders",
                    "default": True,
                },
                "recursive": {
                    "type": "boolean",
                    "title": "Recursive",
                    "description": "Sync subfolders recursively",
                    "default": True,
                },
            },
        }

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get credentials schema."""
        return {
            "type": "object",
            "required": ["access_token"],
            "properties": {
                "access_token": {
                    "type": "string",
                    "title": "Access Token",
                    "description": "Dropbox OAuth access token",
                    "format": "password",
                },
                "refresh_token": {
                    "type": "string",
                    "title": "Refresh Token",
                    "description": "OAuth refresh token (for long-lived access)",
                    "format": "password",
                },
            },
        }
