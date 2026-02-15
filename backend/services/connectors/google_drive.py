"""
AIDocumentIndexer - Google Drive Connector
============================================

Connects to Google Drive for document synchronization.

Features:
- OAuth 2.0 authentication
- Folder and file listing
- Incremental sync with change detection
- Support for Google Docs, Sheets, Slides export
"""

import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import structlog

from backend.services.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorType,
    Resource,
    ResourceType,
    Change,
)
from backend.services.connectors.registry import ConnectorRegistry
from backend.core.config import settings

logger = structlog.get_logger(__name__)

# MIME type mappings for Google Workspace files
GOOGLE_EXPORT_MIMETYPES = {
    "application/vnd.google-apps.document": {
        "export_mime": "application/pdf",
        "extension": "pdf",
    },
    "application/vnd.google-apps.spreadsheet": {
        "export_mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "extension": "xlsx",
    },
    "application/vnd.google-apps.presentation": {
        "export_mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "extension": "pptx",
    },
    "application/vnd.google-apps.drawing": {
        "export_mime": "application/pdf",
        "extension": "pdf",
    },
}

# Supported file MIME types
SUPPORTED_MIMETYPES = [
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/markdown",
    "text/csv",
    # Google Workspace types
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
]


@ConnectorRegistry.register(ConnectorType.GOOGLE_DRIVE)
class GoogleDriveConnector(BaseConnector):
    """
    Google Drive connector for document synchronization.
    """

    connector_type = ConnectorType.GOOGLE_DRIVE
    display_name = "Google Drive"
    description = "Sync documents from Google Drive including Google Docs, Sheets, and Slides"
    icon = "google-drive"

    def __init__(
        self,
        config: ConnectorConfig,
        session=None,
        organization_id=None,
        user_id=None,
    ):
        super().__init__(config, session, organization_id, user_id)
        self._service = None
        self._credentials = None

    async def authenticate(self) -> bool:
        """Authenticate with Google Drive using OAuth credentials."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds_data = self.config.credentials

            if not creds_data.get("access_token"):
                self.log_error("No access token in credentials")
                return False

            self._credentials = Credentials(
                token=creds_data.get("access_token"),
                refresh_token=creds_data.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=getattr(settings, "GOOGLE_CLIENT_ID", None),
                client_secret=getattr(settings, "GOOGLE_CLIENT_SECRET", None),
            )

            # Build the Drive service
            self._service = build("drive", "v3", credentials=self._credentials)

            # Verify authentication by making a simple request
            self._service.about().get(fields="user").execute()

            self._authenticated = True
            self.log_info("Google Drive authentication successful")
            return True

        except Exception as e:
            self.log_error("Google Drive authentication failed", error=e)
            self._authenticated = False
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh OAuth tokens."""
        try:
            from google.auth.transport.requests import Request

            if self._credentials and self._credentials.expired:
                self._credentials.refresh(Request())

                return {
                    "access_token": self._credentials.token,
                    "refresh_token": self._credentials.refresh_token,
                    "expiry": self._credentials.expiry.isoformat() if self._credentials.expiry else None,
                }

            return self.config.credentials

        except Exception as e:
            self.log_error("Failed to refresh credentials", error=e)
            raise

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> Tuple[List[Resource], Optional[str]]:
        """List files and folders in Google Drive."""
        if not self._service:
            await self.authenticate()

        try:
            # Build query
            query_parts = ["trashed = false"]

            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            else:
                query_parts.append("'root' in parents")

            # Filter to supported MIME types
            mime_filter = " or ".join([f"mimeType = '{m}'" for m in SUPPORTED_MIMETYPES])
            mime_filter += " or mimeType = 'application/vnd.google-apps.folder'"
            query_parts.append(f"({mime_filter})")

            query = " and ".join(query_parts)

            # Execute request
            response = self._service.files().list(
                q=query,
                pageSize=page_size,
                pageToken=page_token,
                fields="nextPageToken, files(id, name, mimeType, size, parents, createdTime, modifiedTime, webViewLink)",
                orderBy="modifiedTime desc",
            ).execute()

            files = response.get("files", [])
            next_page_token = response.get("nextPageToken")

            # Convert to Resource objects
            resources = []
            for file in files:
                is_folder = file.get("mimeType") == "application/vnd.google-apps.folder"

                resources.append(Resource(
                    id=file["id"],
                    name=file["name"],
                    resource_type=ResourceType.FOLDER if is_folder else ResourceType.FILE,
                    mime_type=file.get("mimeType"),
                    size_bytes=int(file.get("size", 0)) if file.get("size") else None,
                    parent_id=(file.get("parents") or [None])[0],
                    created_at=datetime.fromisoformat(file["createdTime"].replace("Z", "+00:00")) if file.get("createdTime") else None,
                    modified_at=datetime.fromisoformat(file["modifiedTime"].replace("Z", "+00:00")) if file.get("modifiedTime") else None,
                    web_url=file.get("webViewLink"),
                    metadata={
                        "drive_id": file["id"],
                        "mime_type": file.get("mimeType"),
                    },
                ))

            return resources, next_page_token

        except Exception as e:
            self.log_error("Failed to list resources", error=e, folder_id=folder_id)
            raise

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a specific file or folder."""
        if not self._service:
            await self.authenticate()

        try:
            file = self._service.files().get(
                fileId=resource_id,
                fields="id, name, mimeType, size, parents, createdTime, modifiedTime, webViewLink",
            ).execute()

            is_folder = file.get("mimeType") == "application/vnd.google-apps.folder"

            return Resource(
                id=file["id"],
                name=file["name"],
                resource_type=ResourceType.FOLDER if is_folder else ResourceType.FILE,
                mime_type=file.get("mimeType"),
                size_bytes=int(file.get("size", 0)) if file.get("size") else None,
                parent_id=(file.get("parents") or [None])[0],
                created_at=datetime.fromisoformat(file["createdTime"].replace("Z", "+00:00")) if file.get("createdTime") else None,
                modified_at=datetime.fromisoformat(file["modifiedTime"].replace("Z", "+00:00")) if file.get("modifiedTime") else None,
                web_url=file.get("webViewLink"),
            )

        except Exception as e:
            self.log_error("Failed to get resource", error=e, resource_id=resource_id)
            return None

    async def download_resource(self, resource_id: str) -> Optional[bytes]:
        """Download file content."""
        if not self._service:
            await self.authenticate()

        try:
            # Get file metadata first
            file = self._service.files().get(
                fileId=resource_id,
                fields="id, mimeType, size",
            ).execute()

            mime_type = file.get("mimeType")

            # Handle Google Workspace files (export)
            if mime_type in GOOGLE_EXPORT_MIMETYPES:
                export_config = GOOGLE_EXPORT_MIMETYPES[mime_type]
                response = self._service.files().export(
                    fileId=resource_id,
                    mimeType=export_config["export_mime"],
                ).execute()
                return response

            # Regular file download
            from googleapiclient.http import MediaIoBaseDownload

            request = self._service.files().get_media(fileId=resource_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            return buffer.getvalue()

        except Exception as e:
            self.log_error("Failed to download resource", error=e, resource_id=resource_id)
            return None

    async def get_changes(
        self,
        since_token: Optional[str] = None,
    ) -> Tuple[List[Change], Optional[str]]:
        """Get changes since the last sync."""
        if not self._service:
            await self.authenticate()

        try:
            # Get start page token if not provided
            if not since_token:
                response = self._service.changes().getStartPageToken().execute()
                since_token = response.get("startPageToken")
                return [], since_token

            changes = []
            page_token = since_token

            while page_token:
                response = self._service.changes().list(
                    pageToken=page_token,
                    fields="nextPageToken, newStartPageToken, changes(fileId, removed, time, file(id, name, mimeType, size, parents, modifiedTime))",
                    includeRemoved=True,
                    pageSize=100,
                ).execute()

                for change_data in response.get("changes", []):
                    file_id = change_data.get("fileId")
                    removed = change_data.get("removed", False)
                    file_data = change_data.get("file")

                    change_type = "deleted" if removed else ("modified" if file_data else "created")

                    resource = None
                    if file_data and not removed:
                        is_folder = file_data.get("mimeType") == "application/vnd.google-apps.folder"
                        resource = Resource(
                            id=file_data["id"],
                            name=file_data.get("name", "Unknown"),
                            resource_type=ResourceType.FOLDER if is_folder else ResourceType.FILE,
                            mime_type=file_data.get("mimeType"),
                            size_bytes=int(file_data.get("size", 0)) if file_data.get("size") else None,
                            parent_id=file_data.get("parents", [None])[0] if file_data.get("parents") else None,
                            modified_at=datetime.fromisoformat(file_data["modifiedTime"].replace("Z", "+00:00")) if file_data.get("modifiedTime") else None,
                        )

                    changes.append(Change(
                        resource_id=file_id,
                        change_type=change_type,
                        timestamp=datetime.fromisoformat(change_data["time"].replace("Z", "+00:00")) if change_data.get("time") else datetime.utcnow(),
                        resource=resource,
                    ))

                page_token = response.get("nextPageToken")

                if response.get("newStartPageToken"):
                    return changes, response.get("newStartPageToken")

            return changes, since_token

        except Exception as e:
            self.log_error("Failed to get changes", error=e)
            raise

    def get_oauth_url(self, state: str) -> Optional[str]:
        """Get Google OAuth authorization URL."""
        client_id = getattr(settings, "GOOGLE_CLIENT_ID", None)
        redirect_uri = getattr(settings, "GOOGLE_REDIRECT_URI", None)

        if not client_id or not redirect_uri:
            return None

        scopes = [
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.metadata.readonly",
        ]

        from urllib.parse import urlencode

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }

        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    async def handle_oauth_callback(
        self,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        import httpx

        client_id = getattr(settings, "GOOGLE_CLIENT_ID", None)
        client_secret = getattr(settings, "GOOGLE_CLIENT_SECRET", None)
        redirect_uri = getattr(settings, "GOOGLE_REDIRECT_URI", None)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )

            if response.status_code != 200:
                logger.error("OAuth token exchange failed", status_code=response.status_code)
                raise ValueError(f"OAuth token exchange failed (HTTP {response.status_code})")

            tokens = response.json()

            return {
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "expiry": tokens.get("expires_in"),
                "token_type": tokens.get("token_type"),
            }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "folders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Google Drive folder IDs to sync",
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to include",
                    "default": ["pdf", "docx", "xlsx", "pptx", "txt", "md"],
                },
                "include_google_docs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include Google Docs/Sheets/Slides (exported as PDF/Office)",
                },
                "sync_interval_minutes": {
                    "type": "integer",
                    "minimum": 5,
                    "default": 60,
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
                    "description": "OAuth access token",
                },
                "refresh_token": {
                    "type": "string",
                    "description": "OAuth refresh token",
                },
            },
        }
