"""
AIDocumentIndexer - OneDrive/SharePoint Connector
==================================================

Connector for syncing content from Microsoft OneDrive and SharePoint.

Supports:
- Personal OneDrive
- OneDrive for Business
- SharePoint document libraries
- File and folder sync with delta queries
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode, quote

import structlog
import httpx

from backend.services.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorType,
    Resource,
    ResourceType,
    Change,
)
from backend.services.connectors.registry import ConnectorRegistry

logger = structlog.get_logger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


@ConnectorRegistry.register(ConnectorType.ONEDRIVE)
class OneDriveConnector(BaseConnector):
    """
    Connector for Microsoft OneDrive and SharePoint.

    Uses Microsoft Graph API for file operations.
    Supports OAuth 2.0 authentication with MSAL.
    """

    connector_type = ConnectorType.ONEDRIVE
    display_name = "OneDrive / SharePoint"
    description = "Sync files from Microsoft OneDrive and SharePoint"
    icon = "cloud"

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.config.credentials.get('access_token', '')}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=GRAPH_BASE_URL,
                headers=self._headers,
                timeout=30.0,
            )
        return self._client

    async def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API."""
        try:
            client = await self._get_client()
            response = await client.get("/me")

            if response.status_code == 200:
                self._authenticated = True
                user_data = response.json()
                self.log_info(
                    "Authenticated with Microsoft",
                    user=user_data.get("displayName"),
                    email=user_data.get("mail") or user_data.get("userPrincipalName"),
                )
                return True
            elif response.status_code == 401:
                # Try to refresh token
                new_creds = await self.refresh_credentials()
                if new_creds.get("access_token"):
                    self.config.credentials = new_creds
                    self._client = None  # Reset client with new token
                    return await self.authenticate()

            self.log_error("Microsoft authentication failed", status=response.status_code)
            return False

        except Exception as e:
            self.log_error("Microsoft authentication error", error=e)
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh OAuth access token."""
        refresh_token = self.config.credentials.get("refresh_token")
        client_id = self.config.credentials.get("client_id")
        client_secret = self.config.credentials.get("client_secret")
        tenant_id = self.config.credentials.get("tenant_id", "common")

        if not refresh_token:
            return self.config.credentials

        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                    "scope": "Files.Read.All Sites.Read.All offline_access",
                },
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    **self.config.credentials,
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token", refresh_token),
                    "expires_in": data.get("expires_in"),
                }

        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List files and folders from OneDrive/SharePoint."""
        client = await self._get_client()
        resources = []

        try:
            # Determine the drive to use
            drive_id = self.config.sync_config.get("drive_id")
            site_id = self.config.sync_config.get("site_id")

            if site_id:
                # SharePoint site
                base_path = f"/sites/{site_id}/drive"
            elif drive_id:
                base_path = f"/drives/{drive_id}"
            else:
                # Default to user's OneDrive
                base_path = "/me/drive"

            # Build URL
            if folder_id:
                url = f"{base_path}/items/{folder_id}/children"
            else:
                url = f"{base_path}/root/children"

            # Add pagination
            params = {"$top": min(page_size, 200)}
            if page_token:
                # Use skiptoken for pagination
                params["$skiptoken"] = page_token

            # Add select to reduce payload
            params["$select"] = "id,name,size,file,folder,parentReference,createdDateTime,lastModifiedDateTime,webUrl"

            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                for item in data.get("value", []):
                    resource = self._parse_drive_item(item)
                    if resource:
                        resources.append(resource)

                # Get next page token
                next_link = data.get("@odata.nextLink")
                next_token = None
                if next_link and "$skiptoken=" in next_link:
                    next_token = next_link.split("$skiptoken=")[1].split("&")[0]

                return resources, next_token
            else:
                self.log_error("Failed to list OneDrive items", status=response.status_code)

        except Exception as e:
            self.log_error("Failed to list OneDrive resources", error=e)

        return resources, None

    def _parse_drive_item(self, item: Dict[str, Any]) -> Optional[Resource]:
        """Parse a OneDrive/SharePoint item into a Resource."""
        try:
            item_id = item.get("id")
            name = item.get("name", "Untitled")

            # Determine type
            if "folder" in item:
                resource_type = ResourceType.FOLDER
                mime_type = None
            else:
                resource_type = ResourceType.FILE
                mime_type = item.get("file", {}).get("mimeType")

            # Get parent info
            parent_ref = item.get("parentReference", {})
            parent_id = parent_ref.get("id")

            # Parse dates
            created_at = None
            modified_at = None

            if item.get("createdDateTime"):
                created_at = datetime.fromisoformat(item["createdDateTime"].replace("Z", "+00:00"))
            if item.get("lastModifiedDateTime"):
                modified_at = datetime.fromisoformat(item["lastModifiedDateTime"].replace("Z", "+00:00"))

            return Resource(
                id=item_id,
                name=name,
                resource_type=resource_type,
                mime_type=mime_type,
                size_bytes=item.get("size"),
                parent_id=parent_id,
                path=parent_ref.get("path"),
                created_at=created_at,
                modified_at=modified_at,
                web_url=item.get("webUrl"),
                download_url=item.get("@microsoft.graph.downloadUrl"),
                metadata={
                    "drive_id": parent_ref.get("driveId"),
                    "site_id": parent_ref.get("siteId"),
                    "shared": item.get("shared") is not None,
                },
            )

        except Exception as e:
            self.log_warning("Failed to parse OneDrive item", error=str(e))

        return None

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a specific file or folder."""
        client = await self._get_client()

        try:
            # Get drive context
            drive_id = self.config.sync_config.get("drive_id")
            site_id = self.config.sync_config.get("site_id")

            if site_id:
                url = f"/sites/{site_id}/drive/items/{resource_id}"
            elif drive_id:
                url = f"/drives/{drive_id}/items/{resource_id}"
            else:
                url = f"/me/drive/items/{resource_id}"

            response = await client.get(url)

            if response.status_code == 200:
                return self._parse_drive_item(response.json())

        except Exception as e:
            self.log_error("Failed to get OneDrive item", error=e, resource_id=resource_id)

        return None

    async def download_resource(self, resource_id: str) -> Optional[bytes]:
        """Download file content."""
        client = await self._get_client()

        try:
            # Get drive context
            drive_id = self.config.sync_config.get("drive_id")
            site_id = self.config.sync_config.get("site_id")

            if site_id:
                url = f"/sites/{site_id}/drive/items/{resource_id}/content"
            elif drive_id:
                url = f"/drives/{drive_id}/items/{resource_id}/content"
            else:
                url = f"/me/drive/items/{resource_id}/content"

            response = await client.get(url, follow_redirects=True)

            if response.status_code == 200:
                return response.content
            elif response.status_code == 302:
                # Follow redirect to download URL
                download_url = response.headers.get("Location")
                if download_url:
                    async with httpx.AsyncClient() as download_client:
                        dl_response = await download_client.get(download_url)
                        if dl_response.status_code == 200:
                            return dl_response.content

        except Exception as e:
            self.log_error("Failed to download OneDrive file", error=e, resource_id=resource_id)

        return None

    async def get_changes(
        self,
        since_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """Get changes using delta query."""
        client = await self._get_client()
        changes = []

        try:
            # Build delta URL
            drive_id = self.config.sync_config.get("drive_id")
            site_id = self.config.sync_config.get("site_id")

            if site_id:
                base_url = f"/sites/{site_id}/drive/root/delta"
            elif drive_id:
                base_url = f"/drives/{drive_id}/root/delta"
            else:
                base_url = "/me/drive/root/delta"

            # Use delta link if available
            if since_token and since_token.startswith("http"):
                url = since_token
            else:
                url = base_url
                if since_token:
                    url += f"?token={since_token}"

            response = await client.get(url)

            if response.status_code == 200:
                data = response.json()

                for item in data.get("value", []):
                    # Check if item was deleted
                    if item.get("deleted"):
                        changes.append(Change(
                            resource_id=item["id"],
                            change_type="deleted",
                            timestamp=datetime.utcnow(),
                        ))
                    else:
                        resource = self._parse_drive_item(item)
                        if resource:
                            # Determine if created or modified
                            change_type = "modified"
                            if resource.created_at and resource.modified_at:
                                # If created and modified are within 1 second, it's new
                                diff = abs((resource.modified_at - resource.created_at).total_seconds())
                                if diff < 1:
                                    change_type = "created"

                            changes.append(Change(
                                resource_id=resource.id,
                                change_type=change_type,
                                timestamp=resource.modified_at or datetime.utcnow(),
                                resource=resource,
                            ))

                # Get delta link for next sync
                new_token = data.get("@odata.deltaLink") or data.get("@odata.nextLink")

                return changes, new_token

        except Exception as e:
            self.log_error("Failed to get OneDrive changes", error=e)

        return changes, None

    def get_oauth_url(self, state: str) -> Optional[str]:
        """Get Microsoft OAuth URL."""
        client_id = self.config.credentials.get("client_id")
        redirect_uri = self.config.credentials.get("redirect_uri")
        tenant_id = self.config.credentials.get("tenant_id", "common")

        if client_id and redirect_uri:
            params = {
                "client_id": client_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "response_mode": "query",
                "scope": "Files.Read.All Sites.Read.All offline_access User.Read",
                "state": state,
            }
            return f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize?{urlencode(params)}"
        return None

    async def handle_oauth_callback(
        self,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        """Exchange OAuth code for tokens."""
        client_id = self.config.credentials.get("client_id")
        client_secret = self.config.credentials.get("client_secret")
        redirect_uri = self.config.credentials.get("redirect_uri")
        tenant_id = self.config.credentials.get("tenant_id", "common")

        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                    "scope": "Files.Read.All Sites.Read.All offline_access User.Read",
                },
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token"),
                    "expires_in": data.get("expires_in"),
                    "token_type": data.get("token_type"),
                }
            else:
                raise Exception(f"OAuth token exchange failed: {response.text}")

    async def list_sites(self) -> List[Dict[str, Any]]:
        """List accessible SharePoint sites."""
        client = await self._get_client()
        sites = []

        try:
            response = await client.get("/sites?search=*")

            if response.status_code == 200:
                data = response.json()
                for site in data.get("value", []):
                    sites.append({
                        "id": site["id"],
                        "name": site.get("displayName", site.get("name")),
                        "web_url": site.get("webUrl"),
                    })

        except Exception as e:
            self.log_error("Failed to list SharePoint sites", error=e)

        return sites

    async def list_drives(self, site_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List drives (document libraries) in a site or user's OneDrive."""
        client = await self._get_client()
        drives = []

        try:
            if site_id:
                url = f"/sites/{site_id}/drives"
            else:
                url = "/me/drives"

            response = await client.get(url)

            if response.status_code == 200:
                data = response.json()
                for drive in data.get("value", []):
                    drives.append({
                        "id": drive["id"],
                        "name": drive.get("name"),
                        "drive_type": drive.get("driveType"),
                        "quota": drive.get("quota"),
                    })

        except Exception as e:
            self.log_error("Failed to list drives", error=e)

        return drives

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get schema for OneDrive credentials."""
        return {
            "type": "object",
            "properties": {
                "client_id": {
                    "type": "string",
                    "description": "Azure AD application client ID",
                },
                "client_secret": {
                    "type": "string",
                    "description": "Azure AD application client secret",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Azure AD tenant ID (or 'common' for multi-tenant)",
                    "default": "common",
                },
                "redirect_uri": {
                    "type": "string",
                    "description": "OAuth redirect URI",
                },
                "access_token": {
                    "type": "string",
                    "description": "OAuth access token",
                },
                "refresh_token": {
                    "type": "string",
                    "description": "OAuth refresh token",
                },
            },
            "required": ["client_id"],
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get schema for OneDrive connector configuration."""
        schema = super().get_config_schema()
        schema["properties"].update({
            "drive_id": {
                "type": "string",
                "description": "Specific drive ID to sync",
            },
            "site_id": {
                "type": "string",
                "description": "SharePoint site ID",
            },
            "include_shared": {
                "type": "boolean",
                "default": False,
                "description": "Include shared files",
            },
        })
        return schema

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Also register as SharePoint
@ConnectorRegistry.register(ConnectorType.SHAREPOINT)
class SharePointConnector(OneDriveConnector):
    """SharePoint connector - alias for OneDrive with SharePoint focus."""

    connector_type = ConnectorType.SHAREPOINT
    display_name = "SharePoint"
    description = "Sync documents from SharePoint sites and libraries"
    icon = "share-2"
