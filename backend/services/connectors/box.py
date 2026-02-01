"""
AIDocumentIndexer - Box Connector
==================================

Connector for syncing content from Box.

Supports:
- Files and folders
- Shared links
- Box Notes
- Metadata and classifications
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

# Check for Box SDK
try:
    from boxsdk import OAuth2, Client
    from boxsdk.exception import BoxAPIException
    HAS_BOX = True
except ImportError:
    HAS_BOX = False
    logger.info("Box SDK not available - install with: pip install boxsdk")


@ConnectorRegistry.register(ConnectorType.BOX)
class BoxConnector(BaseConnector):
    """
    Connector for Box cloud storage.

    Uses the Box API to sync files and folders.
    Supports OAuth 2.0 and JWT authentication.
    """

    connector_type = ConnectorType.BOX
    display_name = "Box"
    description = "Sync files and folders from Box"
    icon = "box"

    # File extensions to sync by default
    DEFAULT_EXTENSIONS = [
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".txt", ".md", ".rtf", ".csv", ".json", ".xml",
        ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".boxnote",  # Box Notes
    ]

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[Any] = None
        self._stream_position: Optional[str] = None

    async def _get_client(self):
        """Get or create Box client."""
        if not HAS_BOX:
            raise RuntimeError("Box SDK not installed. Run: pip install boxsdk")

        if self._client is None:
            access_token = self.config.credentials.get("access_token", "")
            client_id = self.config.credentials.get("client_id", "")
            client_secret = self.config.credentials.get("client_secret", "")

            if not access_token:
                raise ValueError("Box access_token is required")

            oauth = OAuth2(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
            )
            self._client = Client(oauth)

        return self._client

    async def authenticate(self) -> bool:
        """Authenticate with Box API."""
        try:
            client = await self._get_client()

            # Run in executor since Box SDK is sync
            loop = asyncio.get_running_loop()
            user = await loop.run_in_executor(
                None, lambda: client.user().get()
            )

            self._authenticated = True
            self.log_info(
                "Authenticated with Box",
                user=user.name,
                login=user.login,
            )
            return True

        except BoxAPIException as e:
            self.log_error("Box authentication failed", error=str(e))
            return False
        except Exception as e:
            self.log_error("Box authentication error", error=str(e))
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh OAuth tokens."""
        # Box SDK handles token refresh automatically if refresh_token provided
        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List files and folders from Box."""
        client = await self._get_client()
        resources = []

        try:
            loop = asyncio.get_running_loop()
            box_folder_id = folder_id or "0"  # "0" is root folder in Box

            # Get folder
            folder = await loop.run_in_executor(
                None, lambda: client.folder(folder_id=box_folder_id).get()
            )

            # Get items with pagination
            offset = int(page_token) if page_token else 0
            items = await loop.run_in_executor(
                None,
                lambda: folder.get_items(
                    limit=min(page_size, 1000),
                    offset=offset,
                )
            )

            item_list = list(items)
            for item in item_list:
                resource = self._parse_item(item)
                if resource:
                    resources.append(resource)

            # Calculate next page token
            next_offset = offset + len(item_list)
            # Box doesn't easily tell us if there are more items
            # Use presence of items as indicator
            next_token = str(next_offset) if len(item_list) == page_size else None

            return resources, next_token

        except BoxAPIException as e:
            self.log_error("Box list failed", error=str(e))
            return [], None
        except Exception as e:
            self.log_error("Failed to list Box resources", error=str(e))
            return [], None

    def _parse_item(self, item) -> Optional[Resource]:
        """Parse a Box item into a Resource."""
        try:
            item_type = item.type

            if item_type == "folder":
                return Resource(
                    id=item.id,
                    name=item.name,
                    resource_type=ResourceType.FOLDER,
                    path=f"/{item.name}",
                    mime_type="application/vnd.box.folder",
                    size=0,
                    created_at=datetime.fromisoformat(item.created_at.replace("Z", "+00:00")) if hasattr(item, "created_at") and item.created_at else None,
                    modified_at=datetime.fromisoformat(item.modified_at.replace("Z", "+00:00")) if hasattr(item, "modified_at") and item.modified_at else None,
                    metadata={
                        "item_status": getattr(item, "item_status", None),
                    },
                )

            elif item_type == "file":
                return Resource(
                    id=item.id,
                    name=item.name,
                    resource_type=ResourceType.FILE,
                    path=f"/{item.name}",
                    mime_type=self._get_mime_type(item.name),
                    size=getattr(item, "size", 0) or 0,
                    created_at=datetime.fromisoformat(item.created_at.replace("Z", "+00:00")) if hasattr(item, "created_at") and item.created_at else None,
                    modified_at=datetime.fromisoformat(item.modified_at.replace("Z", "+00:00")) if hasattr(item, "modified_at") and item.modified_at else None,
                    metadata={
                        "sha1": getattr(item, "sha1", None),
                        "version_number": getattr(item, "version_number", None),
                    },
                )

            elif item_type == "web_link":
                return Resource(
                    id=item.id,
                    name=item.name,
                    resource_type=ResourceType.LINK,
                    path=f"/{item.name}",
                    mime_type="application/vnd.box.weblink",
                    size=0,
                    metadata={
                        "url": getattr(item, "url", None),
                    },
                )

            return None

        except Exception as e:
            self.log_error("Failed to parse Box item", error=str(e))
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
            "boxnote": "application/vnd.box.note",
        }
        return mime_types.get(extension, "application/octet-stream")

    async def download_resource(self, resource: Resource) -> bytes:
        """Download file content from Box."""
        client = await self._get_client()

        if resource.resource_type == ResourceType.FOLDER:
            return f"Folder: {resource.name}\nPath: {resource.path}".encode("utf-8")

        if resource.resource_type == ResourceType.LINK:
            url = resource.metadata.get("url", "")
            return f"Web Link: {resource.name}\nURL: {url}".encode("utf-8")

        try:
            loop = asyncio.get_running_loop()

            # Download file content
            content = await loop.run_in_executor(
                None,
                lambda: client.file(file_id=resource.id).content()
            )

            return content

        except BoxAPIException as e:
            self.log_error("Box download failed", error=str(e))
            raise ValueError(f"Failed to download: {resource.path}")

    async def get_changes(
        self,
        since: Optional[datetime] = None,
        page_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """Get changes using Box events stream."""
        client = await self._get_client()
        changes = []

        try:
            loop = asyncio.get_running_loop()

            stream_position = page_token or self._stream_position or "0"

            events = await loop.run_in_executor(
                None,
                lambda: client.events().get_events(
                    stream_position=stream_position,
                    stream_type="changes",
                )
            )

            for event in events.get("entries", []):
                change = self._parse_event(event)
                if change:
                    changes.append(change)

            # Update stream position
            next_position = events.get("next_stream_position")
            self._stream_position = next_position

            return changes, next_position

        except Exception as e:
            self.log_error("Failed to get Box changes", error=str(e))
            return [], None

    def _parse_event(self, event: Dict) -> Optional[Change]:
        """Parse a Box event into a Change."""
        try:
            event_type = event.get("event_type", "")
            source = event.get("source", {})
            created_at = event.get("created_at")

            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else datetime.utcnow()

            # Map event types to change types
            if event_type in ["ITEM_CREATE", "ITEM_UPLOAD"]:
                change_type = ChangeType.CREATED
            elif event_type in ["ITEM_MODIFY", "ITEM_RENAME", "ITEM_MOVE"]:
                change_type = ChangeType.MODIFIED
            elif event_type in ["ITEM_TRASH", "ITEM_DELETE"]:
                change_type = ChangeType.DELETED
            else:
                return None

            return Change(
                resource_id=source.get("id", ""),
                change_type=change_type,
                timestamp=timestamp,
                metadata={
                    "event_type": event_type,
                    "source_type": source.get("type"),
                    "source_name": source.get("name"),
                },
            )

        except Exception as e:
            self.log_error("Failed to parse Box event", error=str(e))
            return None

    async def close(self):
        """Close the connector."""
        self._client = None
        self._stream_position = None

    @classmethod
    def get_oauth_url(
        cls,
        client_id: str,
        redirect_uri: str,
        state: str,
    ) -> str:
        """Generate Box OAuth authorization URL."""
        return (
            f"https://account.box.com/api/oauth2/authorize"
            f"?client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
            f"&state={state}"
        )

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "folder_id": {
                    "type": "string",
                    "title": "Folder ID",
                    "description": "Box folder ID to sync (empty = root)",
                    "default": "0",
                },
                "file_extensions": {
                    "type": "array",
                    "title": "File Extensions",
                    "description": "Extensions to sync (empty = defaults)",
                    "items": {"type": "string"},
                    "default": [],
                },
                "include_trashed": {
                    "type": "boolean",
                    "title": "Include Trashed",
                    "description": "Include trashed items",
                    "default": False,
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
                "client_id": {
                    "type": "string",
                    "title": "Client ID",
                    "description": "Box OAuth Client ID",
                },
                "client_secret": {
                    "type": "string",
                    "title": "Client Secret",
                    "description": "Box OAuth Client Secret",
                    "format": "password",
                },
                "access_token": {
                    "type": "string",
                    "title": "Access Token",
                    "description": "Box OAuth access token",
                    "format": "password",
                },
                "refresh_token": {
                    "type": "string",
                    "title": "Refresh Token",
                    "description": "OAuth refresh token",
                    "format": "password",
                },
            },
        }
