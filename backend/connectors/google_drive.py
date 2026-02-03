"""
Google Drive Connector for AIDocumentIndexer
=============================================

Fetches files and documents from Google Drive.
Supports:
- Google Docs, Sheets, Slides (exported as text)
- PDFs and other documents
- Folder traversal
- Shared drives
- Incremental sync via change tokens
"""

import asyncio
import hashlib
import io
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DriveFile(BaseModel):
    """Represents a Google Drive file."""
    id: str
    name: str
    mime_type: str
    content: str = ""
    size: int = 0
    created_time: datetime
    modified_time: datetime
    web_view_link: str = ""
    parents: list[str] = Field(default_factory=list)
    owners: list[str] = Field(default_factory=list)
    shared: bool = False
    trashed: bool = False

    @property
    def is_google_doc(self) -> bool:
        return self.mime_type.startswith("application/vnd.google-apps")

    @property
    def extension(self) -> str:
        if "." in self.name:
            return self.name.rsplit(".", 1)[-1].lower()
        return ""

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.content}:{self.modified_time}".encode()).hexdigest()[:16]


class DriveFolder(BaseModel):
    """Represents a Google Drive folder."""
    id: str
    name: str
    parents: list[str] = Field(default_factory=list)
    path: str = ""


class GoogleDriveConnectorConfig(BaseModel):
    """Configuration for Google Drive connector."""
    credentials_json: str  # Service account or OAuth credentials JSON
    access_token: Optional[str] = None  # For OAuth flow
    refresh_token: Optional[str] = None
    folder_ids: list[str] = Field(default_factory=list)  # Root folders to sync
    include_shared: bool = True
    include_trashed: bool = False
    supported_mime_types: list[str] = Field(
        default_factory=lambda: [
            "application/vnd.google-apps.document",
            "application/vnd.google-apps.spreadsheet",
            "application/vnd.google-apps.presentation",
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/html",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]
    )
    max_file_size_mb: int = 50
    sync_interval_minutes: int = 60


class GoogleDriveConnector:
    """
    Google Drive API connector for fetching files.

    Usage:
        connector = GoogleDriveConnector(access_token="ya29.xxx")
        async for file in connector.fetch_all_files():
            print(f"Fetched: {file.name}")
    """

    BASE_URL = "https://www.googleapis.com/drive/v3"
    UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3"

    # Export MIME types for Google Workspace files
    EXPORT_MIME_TYPES = {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
        "application/vnd.google-apps.presentation": "text/plain",
        "application/vnd.google-apps.drawing": "image/png",
    }

    def __init__(self, config: GoogleDriveConnectorConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token = config.access_token
        self._start_page_token: Optional[str] = None

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=60.0,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[dict]:
        client = await self._get_client()
        try:
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Google Drive: Authentication failed - token may be expired")
            elif e.response.status_code == 403:
                logger.error("Google Drive: Access forbidden")
            else:
                logger.error(f"Google Drive API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Google Drive request failed: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test the API connection."""
        data = await self._request("GET", "/about", params={"fields": "user"})
        return data is not None

    async def get_user_info(self) -> Optional[dict]:
        """Get current user information."""
        return await self._request(
            "GET", "/about", params={"fields": "user,storageQuota"}
        )

    # =========================================================================
    # File Listing
    # =========================================================================

    async def list_files(
        self,
        folder_id: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 100,
    ) -> AsyncGenerator[DriveFile, None]:
        """
        List files in Drive or a specific folder.

        Args:
            folder_id: Parent folder ID (None for all files)
            query: Additional query string
            page_size: Number of files per page

        Yields:
            DriveFile objects (without content)
        """
        page_token = None
        fields = "nextPageToken,files(id,name,mimeType,size,createdTime,modifiedTime,webViewLink,parents,owners,shared,trashed)"

        # Build query
        q_parts = []
        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")
        if not self.config.include_trashed:
            q_parts.append("trashed = false")
        if query:
            q_parts.append(query)

        q = " and ".join(q_parts) if q_parts else None

        while True:
            params: dict[str, Any] = {
                "pageSize": page_size,
                "fields": fields,
            }
            if q:
                params["q"] = q
            if page_token:
                params["pageToken"] = page_token
            if self.config.include_shared:
                params["includeItemsFromAllDrives"] = True
                params["supportsAllDrives"] = True

            data = await self._request("GET", "/files", params=params)
            if not data:
                break

            for file in data.get("files", []):
                # Skip unsupported types
                if file["mimeType"] not in self.config.supported_mime_types:
                    if not file["mimeType"].startswith("application/vnd.google-apps"):
                        continue

                # Skip folders
                if file["mimeType"] == "application/vnd.google-apps.folder":
                    continue

                yield DriveFile(
                    id=file["id"],
                    name=file["name"],
                    mime_type=file["mimeType"],
                    size=int(file.get("size", 0)),
                    created_time=datetime.fromisoformat(
                        file["createdTime"].replace("Z", "+00:00")
                    ),
                    modified_time=datetime.fromisoformat(
                        file["modifiedTime"].replace("Z", "+00:00")
                    ),
                    web_view_link=file.get("webViewLink", ""),
                    parents=file.get("parents", []),
                    owners=[o.get("emailAddress", "") for o in file.get("owners", [])],
                    shared=file.get("shared", False),
                    trashed=file.get("trashed", False),
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            await asyncio.sleep(0.1)

    async def list_folders(
        self,
        parent_id: Optional[str] = None,
    ) -> AsyncGenerator[DriveFolder, None]:
        """List folders."""
        page_token = None
        q = "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            q += f" and '{parent_id}' in parents"

        while True:
            params: dict[str, Any] = {
                "q": q,
                "pageSize": 100,
                "fields": "nextPageToken,files(id,name,parents)",
            }
            if page_token:
                params["pageToken"] = page_token

            data = await self._request("GET", "/files", params=params)
            if not data:
                break

            for folder in data.get("files", []):
                yield DriveFolder(
                    id=folder["id"],
                    name=folder["name"],
                    parents=folder.get("parents", []),
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    # =========================================================================
    # File Content
    # =========================================================================

    async def get_file_content(self, file: DriveFile) -> Optional[str]:
        """
        Get the text content of a file.

        For Google Docs/Sheets/Slides, exports to text format.
        For other files, downloads and extracts text.
        """
        # Check size limit
        if file.size > self.config.max_file_size_mb * 1024 * 1024:
            logger.warning(f"File {file.name} exceeds size limit, skipping")
            return None

        if file.is_google_doc:
            return await self._export_google_file(file)
        else:
            return await self._download_file(file)

    async def _export_google_file(self, file: DriveFile) -> Optional[str]:
        """Export a Google Workspace file to text."""
        export_mime = self.EXPORT_MIME_TYPES.get(file.mime_type)
        if not export_mime:
            return None

        client = await self._get_client()
        try:
            response = await client.get(
                f"/files/{file.id}/export",
                params={"mimeType": export_mime},
            )
            response.raise_for_status()

            if export_mime.startswith("text/"):
                return response.text
            else:
                return None  # Binary formats not supported for text extraction
        except Exception as e:
            logger.error(f"Failed to export {file.name}: {e}")
            return None

    async def _download_file(self, file: DriveFile) -> Optional[str]:
        """Download and extract text from a file."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"/files/{file.id}",
                params={"alt": "media"},
            )
            response.raise_for_status()

            # Handle text files directly
            if file.mime_type.startswith("text/"):
                return response.text

            # For PDFs and Office docs, return indication that processing is needed
            # In production, you'd use a PDF parser or document converter here
            if file.mime_type == "application/pdf":
                return f"[PDF Document: {file.name} - {file.size} bytes]"

            return None
        except Exception as e:
            logger.error(f"Failed to download {file.name}: {e}")
            return None

    # =========================================================================
    # Full Sync
    # =========================================================================

    async def fetch_all_files(
        self,
        folder_ids: Optional[list[str]] = None,
        with_content: bool = True,
    ) -> AsyncGenerator[DriveFile, None]:
        """
        Fetch all files with content.

        Args:
            folder_ids: Specific folders to sync (or all if None)
            with_content: Whether to fetch file content

        Yields:
            DriveFile objects with content
        """
        if folder_ids:
            # Fetch from specific folders
            for folder_id in folder_ids:
                async for file in self.list_files(folder_id=folder_id):
                    if with_content:
                        content = await self.get_file_content(file)
                        if content:
                            file.content = content
                    yield file
        elif self.config.folder_ids:
            # Fetch from configured folders
            for folder_id in self.config.folder_ids:
                async for file in self.list_files(folder_id=folder_id):
                    if with_content:
                        content = await self.get_file_content(file)
                        if content:
                            file.content = content
                    yield file
        else:
            # Fetch all accessible files
            async for file in self.list_files():
                if with_content:
                    content = await self.get_file_content(file)
                    if content:
                        file.content = content
                yield file

    # =========================================================================
    # Change Detection (Incremental Sync)
    # =========================================================================

    async def get_start_page_token(self) -> Optional[str]:
        """Get the starting page token for change tracking."""
        data = await self._request("GET", "/changes/startPageToken")
        if data:
            return data.get("startPageToken")
        return None

    async def get_changes(
        self,
        page_token: str,
    ) -> AsyncGenerator[DriveFile, None]:
        """
        Get files that changed since the page token.

        Args:
            page_token: Page token from previous sync

        Yields:
            Changed DriveFile objects
        """
        current_token = page_token

        while current_token:
            params = {
                "pageToken": current_token,
                "fields": "nextPageToken,newStartPageToken,changes(fileId,file(id,name,mimeType,size,createdTime,modifiedTime,trashed))",
            }

            data = await self._request("GET", "/changes", params=params)
            if not data:
                break

            for change in data.get("changes", []):
                file_data = change.get("file")
                if not file_data:
                    continue

                # Skip trashed files
                if file_data.get("trashed") and not self.config.include_trashed:
                    continue

                yield DriveFile(
                    id=file_data["id"],
                    name=file_data.get("name", ""),
                    mime_type=file_data.get("mimeType", ""),
                    size=int(file_data.get("size", 0)),
                    created_time=datetime.fromisoformat(
                        file_data.get("createdTime", "1970-01-01T00:00:00Z").replace("Z", "+00:00")
                    ),
                    modified_time=datetime.fromisoformat(
                        file_data.get("modifiedTime", "1970-01-01T00:00:00Z").replace("Z", "+00:00")
                    ),
                    trashed=file_data.get("trashed", False),
                )

            current_token = data.get("nextPageToken")
            if not current_token:
                # Save new start token for next sync
                self._start_page_token = data.get("newStartPageToken")
                break

            await asyncio.sleep(0.1)


def create_google_drive_connector(
    access_token: str,
    folder_ids: list[str] = None,
    include_shared: bool = True,
) -> GoogleDriveConnector:
    """
    Create a Google Drive connector instance.

    Args:
        access_token: OAuth access token
        folder_ids: List of folder IDs to sync
        include_shared: Include shared files

    Returns:
        Configured GoogleDriveConnector instance
    """
    config = GoogleDriveConnectorConfig(
        credentials_json="",
        access_token=access_token,
        folder_ids=folder_ids or [],
        include_shared=include_shared,
    )
    return GoogleDriveConnector(config)
