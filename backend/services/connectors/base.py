"""
AIDocumentIndexer - Base Connector
===================================

Abstract base class for all data source connectors.

Provides a unified interface for:
- Authentication
- Resource listing
- Document sync
- Change detection
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, AsyncGenerator

import structlog
from pydantic import BaseModel, Field

from backend.services.base import BaseService

logger = structlog.get_logger(__name__)


class ConnectorType(str, Enum):
    """Supported connector types."""
    GOOGLE_DRIVE = "google_drive"
    NOTION = "notion"
    CONFLUENCE = "confluence"
    ONEDRIVE = "onedrive"
    SHAREPOINT = "sharepoint"
    SLACK = "slack"
    YOUTUBE = "youtube"
    DROPBOX = "dropbox"
    GITHUB = "github"


class ResourceType(str, Enum):
    """Types of resources that can be synced."""
    FILE = "file"
    FOLDER = "folder"
    PAGE = "page"
    DATABASE = "database"
    MESSAGE = "message"
    VIDEO = "video"


class SyncStatus(str, Enum):
    """Status of a sync operation."""
    PENDING = "pending"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ConnectorConfig(BaseModel):
    """Configuration for a connector instance."""
    connector_type: ConnectorType
    credentials: Dict[str, Any] = Field(default_factory=dict)
    sync_config: Dict[str, Any] = Field(default_factory=dict)

    # Sync settings
    folders: List[str] = Field(default_factory=list, description="Folder IDs to sync")
    file_types: List[str] = Field(default_factory=list, description="File extensions to include")
    exclude_patterns: List[str] = Field(default_factory=list, description="Patterns to exclude")
    sync_interval_minutes: int = Field(default=60, description="Sync interval")
    max_file_size_mb: int = Field(default=50, description="Max file size to sync")


class Resource(BaseModel):
    """A resource from an external data source."""
    id: str
    name: str
    resource_type: ResourceType
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    parent_id: Optional[str] = None
    path: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    web_url: Optional[str] = None
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Change(BaseModel):
    """A change detected in the data source."""
    resource_id: str
    change_type: str  # created, modified, deleted, moved
    timestamp: datetime
    resource: Optional[Resource] = None
    previous_parent_id: Optional[str] = None


class SyncResult(BaseModel):
    """Result of a sync operation."""
    status: SyncStatus
    connector_instance_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    resources_synced: int = 0
    resources_failed: int = 0
    bytes_synced: int = 0
    documents_created: int = 0
    documents_updated: int = 0
    documents_deleted: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    change_token: Optional[str] = None


class BaseConnector(BaseService):
    """
    Abstract base class for data source connectors.

    All connectors must implement:
    - authenticate(): Validate credentials
    - list_resources(): List available resources
    - get_resource(): Get a specific resource
    - download_resource(): Download resource content
    - get_changes(): Get changes since last sync
    """

    connector_type: ConnectorType = None
    display_name: str = "Base Connector"
    description: str = "Abstract base connector"
    icon: str = "link"

    def __init__(
        self,
        config: ConnectorConfig,
        session=None,
        organization_id=None,
        user_id=None,
    ):
        super().__init__(session, organization_id, user_id)
        self.config = config
        self._authenticated = False

    @property
    def is_authenticated(self) -> bool:
        """Check if connector is authenticated."""
        return self._authenticated

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the data source.

        Returns:
            True if authentication successful

        Raises:
            AuthenticationError if authentication fails
        """
        pass

    @abstractmethod
    async def refresh_credentials(self) -> Dict[str, Any]:
        """
        Refresh authentication credentials (e.g., OAuth tokens).

        Returns:
            Updated credentials dict
        """
        pass

    @abstractmethod
    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """
        List resources in the data source.

        Args:
            folder_id: Parent folder ID (None for root)
            page_token: Pagination token
            page_size: Number of resources per page

        Returns:
            Tuple of (resources list, next page token or None)
        """
        pass

    @abstractmethod
    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """
        Get a specific resource by ID.

        Args:
            resource_id: Resource identifier

        Returns:
            Resource or None if not found
        """
        pass

    @abstractmethod
    async def download_resource(self, resource_id: str) -> Optional[bytes]:
        """
        Download resource content.

        Args:
            resource_id: Resource identifier

        Returns:
            Resource content as bytes, or None
        """
        pass

    @abstractmethod
    async def get_changes(
        self,
        since_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """
        Get changes since the last sync.

        Args:
            since_token: Change token from previous sync

        Returns:
            Tuple of (changes list, new change token)
        """
        pass

    async def sync_all(
        self,
        connector_instance_id: str,
    ) -> SyncResult:
        """
        Perform a full sync of all configured resources.

        Args:
            connector_instance_id: Database ID of the connector instance

        Returns:
            SyncResult with details of the operation
        """
        result = SyncResult(
            status=SyncStatus.SYNCING,
            connector_instance_id=connector_instance_id,
            started_at=datetime.utcnow(),
        )

        try:
            # Ensure authenticated
            if not self._authenticated:
                await self.authenticate()

            # Get folders to sync
            folders_to_sync = self.config.folders or [None]  # None = root

            for folder_id in folders_to_sync:
                await self._sync_folder(folder_id, result)

            result.status = SyncStatus.COMPLETED
            result.completed_at = datetime.utcnow()

        except Exception as e:
            self.log_error("Sync failed", error=e)
            result.status = SyncStatus.FAILED
            result.errors.append({
                "type": "sync_error",
                "message": str(e),
            })
            result.completed_at = datetime.utcnow()

        return result

    async def sync_incremental(
        self,
        connector_instance_id: str,
        change_token: Optional[str] = None,
    ) -> SyncResult:
        """
        Perform an incremental sync based on changes.

        Args:
            connector_instance_id: Database ID of the connector instance
            change_token: Token from previous sync

        Returns:
            SyncResult with details of the operation
        """
        result = SyncResult(
            status=SyncStatus.SYNCING,
            connector_instance_id=connector_instance_id,
            started_at=datetime.utcnow(),
        )

        try:
            # Ensure authenticated
            if not self._authenticated:
                await self.authenticate()

            # Get changes
            changes, new_token = await self.get_changes(change_token)
            result.change_token = new_token

            for change in changes:
                try:
                    await self._process_change(change, result)
                except Exception as e:
                    self.log_warning("Failed to process change", error=str(e), change=change.resource_id)
                    result.errors.append({
                        "type": "change_error",
                        "resource_id": change.resource_id,
                        "message": str(e),
                    })
                    result.resources_failed += 1

            result.status = SyncStatus.COMPLETED if not result.errors else SyncStatus.PARTIAL
            result.completed_at = datetime.utcnow()

        except Exception as e:
            self.log_error("Incremental sync failed", error=e)
            result.status = SyncStatus.FAILED
            result.errors.append({
                "type": "sync_error",
                "message": str(e),
            })
            result.completed_at = datetime.utcnow()

        return result

    async def _sync_folder(
        self,
        folder_id: Optional[str],
        result: SyncResult,
    ):
        """Sync all resources in a folder recursively."""
        page_token = None

        while True:
            resources, page_token = await self.list_resources(
                folder_id=folder_id,
                page_token=page_token,
            )

            for resource in resources:
                try:
                    if resource.resource_type == ResourceType.FOLDER:
                        # Recurse into subfolders
                        await self._sync_folder(resource.id, result)
                    else:
                        # Sync file
                        await self._sync_resource(resource, result)
                except Exception as e:
                    self.log_warning("Failed to sync resource", error=str(e), resource_id=resource.id)
                    result.errors.append({
                        "type": "resource_error",
                        "resource_id": resource.id,
                        "message": str(e),
                    })
                    result.resources_failed += 1

            if not page_token:
                break

    async def _sync_resource(
        self,
        resource: Resource,
        result: SyncResult,
    ):
        """Sync a single resource."""
        # Check file type filter
        if self.config.file_types:
            ext = resource.name.split(".")[-1].lower() if "." in resource.name else ""
            if ext not in self.config.file_types:
                return

        # Check file size
        if resource.size_bytes and resource.size_bytes > self.config.max_file_size_mb * 1024 * 1024:
            self.log_debug("Skipping large file", resource_id=resource.id, size=resource.size_bytes)
            return

        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if pattern in resource.name or (resource.path and pattern in resource.path):
                return

        # Download content
        content = await self.download_resource(resource.id)
        if not content:
            result.resources_failed += 1
            return

        # Create or update document
        # This would integrate with the document service
        # For now, just count
        result.resources_synced += 1
        result.bytes_synced += len(content)
        result.documents_created += 1

    async def _process_change(
        self,
        change: Change,
        result: SyncResult,
    ):
        """Process a single change."""
        if change.change_type == "deleted":
            # Mark document as deleted
            result.documents_deleted += 1
        elif change.change_type in ["created", "modified"]:
            if change.resource:
                await self._sync_resource(change.resource, result)

    def get_oauth_url(self, state: str) -> Optional[str]:
        """
        Get OAuth authorization URL for connectors that use OAuth.

        Args:
            state: State parameter for OAuth callback

        Returns:
            Authorization URL or None if not OAuth-based
        """
        return None

    async def handle_oauth_callback(
        self,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        """
        Handle OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code
            state: State parameter

        Returns:
            Credentials dict with tokens
        """
        raise NotImplementedError("OAuth not supported for this connector")

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Get JSON schema for connector configuration.

        Returns:
            JSON schema dict
        """
        return {
            "type": "object",
            "properties": {
                "folders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Folder IDs to sync",
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to include (e.g., pdf, docx)",
                },
                "sync_interval_minutes": {
                    "type": "integer",
                    "minimum": 5,
                    "default": 60,
                    "description": "Sync interval in minutes",
                },
            },
        }

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """
        Get JSON schema for connector credentials.

        Returns:
            JSON schema dict
        """
        return {
            "type": "object",
            "properties": {},
        }
