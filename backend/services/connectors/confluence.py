"""
AIDocumentIndexer - Confluence Connector
=========================================

Connector for syncing content from Atlassian Confluence.

Supports:
- Spaces and pages
- Attachments
- Page hierarchy
- Labels and metadata
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import base64

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


@ConnectorRegistry.register(ConnectorType.CONFLUENCE)
class ConfluenceConnector(BaseConnector):
    """
    Connector for Atlassian Confluence.

    Supports both Confluence Cloud and Data Center/Server.
    Uses REST API v2 for Cloud, v1 for Server.
    """

    connector_type = ConnectorType.CONFLUENCE
    display_name = "Confluence"
    description = "Sync pages and spaces from Atlassian Confluence"
    icon = "book-open"

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None
        self._is_cloud = True

    @property
    def _base_url(self) -> str:
        """Get API base URL."""
        base = self.config.credentials.get("base_url", "").rstrip("/")
        if self._is_cloud:
            return f"{base}/wiki/api/v2"
        return f"{base}/rest/api"

    @property
    def _headers(self) -> Dict[str, str]:
        """Get API headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Support both API token (Cloud) and Personal Access Token (Server)
        email = self.config.credentials.get("email")
        api_token = self.config.credentials.get("api_token")
        pat = self.config.credentials.get("personal_access_token")

        if email and api_token:
            # Cloud authentication
            auth_string = f"{email}:{api_token}"
            encoded = base64.b64encode(auth_string.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        elif pat:
            # Server/Data Center PAT
            headers["Authorization"] = f"Bearer {pat}"

        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self._headers,
                timeout=30.0,
            )
        return self._client

    async def authenticate(self) -> bool:
        """Authenticate with Confluence API."""
        try:
            client = await self._get_client()

            # Test authentication by getting current user
            response = await client.get(f"{self._base_url}/user/current")

            if response.status_code == 200:
                self._authenticated = True
                user_data = response.json()
                self.log_info(
                    "Authenticated with Confluence",
                    user=user_data.get("displayName") or user_data.get("publicName"),
                )
                return True
            elif response.status_code == 404:
                # Try v1 API for Server/Data Center
                self._is_cloud = False
                response = await client.get(f"{self._base_url}/user/current")
                if response.status_code == 200:
                    self._authenticated = True
                    return True

            self.log_error("Confluence authentication failed", status=response.status_code)
            return False

        except Exception as e:
            self.log_error("Confluence authentication error", error=e)
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh credentials - not needed for basic auth."""
        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List pages from Confluence."""
        client = await self._get_client()
        resources = []

        try:
            if folder_id:
                # List pages in a specific space or under a parent page
                if folder_id.startswith("space:"):
                    space_key = folder_id.replace("space:", "")
                    resources, next_token = await self._list_space_pages(
                        space_key, page_token, page_size, client
                    )
                else:
                    # List child pages
                    resources = await self._list_child_pages(folder_id, client)
                    next_token = None
                return resources, next_token
            else:
                # List all spaces
                return await self._list_spaces(page_token, page_size, client)

        except Exception as e:
            self.log_error("Failed to list Confluence resources", error=e)

        return resources, None

    async def _list_spaces(
        self,
        page_token: Optional[str],
        page_size: int,
        client: httpx.AsyncClient,
    ) -> tuple[List[Resource], Optional[str]]:
        """List all accessible spaces."""
        resources = []

        params = {"limit": min(page_size, 250)}
        if page_token:
            params["cursor"] = page_token

        url = f"{self._base_url}/spaces"
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()

            for space in data.get("results", []):
                resource = Resource(
                    id=f"space:{space['key']}",
                    name=space.get("name", space["key"]),
                    resource_type=ResourceType.FOLDER,
                    web_url=space.get("_links", {}).get("webui"),
                    metadata={
                        "type": "space",
                        "key": space["key"],
                        "description": space.get("description", {}).get("plain", {}).get("value"),
                        "status": space.get("status"),
                    },
                )
                resources.append(resource)

            # Get next page cursor
            next_cursor = None
            links = data.get("_links", {})
            if links.get("next"):
                # Extract cursor from next link
                next_link = links["next"]
                if "cursor=" in next_link:
                    next_cursor = next_link.split("cursor=")[1].split("&")[0]

            return resources, next_cursor

        return resources, None

    async def _list_space_pages(
        self,
        space_key: str,
        page_token: Optional[str],
        page_size: int,
        client: httpx.AsyncClient,
    ) -> tuple[List[Resource], Optional[str]]:
        """List pages in a space."""
        resources = []

        if self._is_cloud:
            # Cloud API v2
            params = {
                "space-id": space_key,
                "limit": min(page_size, 250),
                "body-format": "storage",
            }
            if page_token:
                params["cursor"] = page_token

            url = f"{self._base_url}/pages"
        else:
            # Server API v1
            params = {
                "spaceKey": space_key,
                "limit": min(page_size, 100),
                "expand": "version,ancestors",
            }
            if page_token:
                params["start"] = int(page_token)

            url = f"{self._base_url}/content"

        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()

            results = data.get("results", [])
            for page in results:
                resource = self._parse_page(page, space_key)
                if resource:
                    resources.append(resource)

            # Get next page
            next_token = None
            if self._is_cloud:
                links = data.get("_links", {})
                if links.get("next"):
                    next_link = links["next"]
                    if "cursor=" in next_link:
                        next_token = next_link.split("cursor=")[1].split("&")[0]
            else:
                # Server pagination
                size = data.get("size", 0)
                start = data.get("start", 0)
                total = data.get("totalSize", 0)
                if start + size < total:
                    next_token = str(start + size)

            return resources, next_token

        return resources, None

    async def _list_child_pages(
        self,
        page_id: str,
        client: httpx.AsyncClient,
    ) -> List[Resource]:
        """List child pages of a page."""
        resources = []

        if self._is_cloud:
            url = f"{self._base_url}/pages/{page_id}/children"
        else:
            url = f"{self._base_url}/content/{page_id}/child/page"

        response = await client.get(url)

        if response.status_code == 200:
            data = response.json()

            for page in data.get("results", []):
                resource = self._parse_page(page, parent_id=page_id)
                if resource:
                    resources.append(resource)

        return resources

    def _parse_page(
        self,
        page: Dict[str, Any],
        space_key: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Optional[Resource]:
        """Parse a Confluence page into a Resource."""
        try:
            page_id = page.get("id")
            title = page.get("title", "Untitled")

            # Get dates
            created_at = None
            modified_at = None

            if "createdAt" in page:
                created_at = datetime.fromisoformat(page["createdAt"].replace("Z", "+00:00"))
            elif "history" in page:
                created_at = datetime.fromisoformat(
                    page["history"]["createdDate"].replace("Z", "+00:00")
                )

            version = page.get("version", {})
            if "createdAt" in version:
                modified_at = datetime.fromisoformat(version["createdAt"].replace("Z", "+00:00"))
            elif "when" in version:
                modified_at = datetime.fromisoformat(version["when"].replace("Z", "+00:00"))

            # Get parent
            if not parent_id:
                ancestors = page.get("ancestors", [])
                if ancestors:
                    parent_id = ancestors[-1].get("id")

            # Get web URL
            web_url = None
            links = page.get("_links", {})
            if "webui" in links:
                base = self.config.credentials.get("base_url", "").rstrip("/")
                web_url = f"{base}/wiki{links['webui']}"

            return Resource(
                id=page_id,
                name=title,
                resource_type=ResourceType.PAGE,
                parent_id=parent_id or (f"space:{space_key}" if space_key else None),
                web_url=web_url,
                created_at=created_at,
                modified_at=modified_at,
                metadata={
                    "type": "page",
                    "space_key": space_key or page.get("spaceId"),
                    "version": version.get("number", 1),
                    "status": page.get("status"),
                },
            )

        except Exception as e:
            self.log_warning("Failed to parse Confluence page", error=str(e))

        return None

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a specific page."""
        client = await self._get_client()

        if resource_id.startswith("space:"):
            # Return space as folder
            space_key = resource_id.replace("space:", "")
            url = f"{self._base_url}/spaces/{space_key}"
            response = await client.get(url)

            if response.status_code == 200:
                space = response.json()
                return Resource(
                    id=resource_id,
                    name=space.get("name", space_key),
                    resource_type=ResourceType.FOLDER,
                    metadata={"type": "space", "key": space_key},
                )
        else:
            if self._is_cloud:
                url = f"{self._base_url}/pages/{resource_id}"
            else:
                url = f"{self._base_url}/content/{resource_id}"

            response = await client.get(url)

            if response.status_code == 200:
                return self._parse_page(response.json())

        return None

    async def download_resource(self, resource_id: str) -> Optional[bytes]:
        """Download page content as HTML/text."""
        client = await self._get_client()

        try:
            if self._is_cloud:
                # Get page with body
                url = f"{self._base_url}/pages/{resource_id}"
                params = {"body-format": "storage"}
            else:
                url = f"{self._base_url}/content/{resource_id}"
                params = {"expand": "body.storage"}

            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                # Extract body content
                if self._is_cloud:
                    body = data.get("body", {}).get("storage", {}).get("value", "")
                else:
                    body = data.get("body", {}).get("storage", {}).get("value", "")

                # Convert HTML to plain text (basic)
                content = self._html_to_text(body)
                return content.encode("utf-8")

        except Exception as e:
            self.log_error("Failed to download Confluence page", error=e, resource_id=resource_id)

        return None

    def _html_to_text(self, html: str) -> str:
        """Basic HTML to text conversion."""
        import re

        # Remove script and style elements
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Replace common elements with newlines
        text = re.sub(r"<(br|hr|p|div|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)

        # Replace list items
        text = re.sub(r"<li[^>]*>", "\n• ", text, flags=re.IGNORECASE)

        # Remove all remaining tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&amp;", "&")
        text = text.replace("&quot;", '"')

        # Clean up whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()

        return text

    async def get_changes(
        self,
        since_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """Get changes since last sync using CQL search."""
        client = await self._get_client()
        changes = []

        try:
            # Use CQL to find recently modified content
            if since_token:
                # Parse token as timestamp
                since_date = since_token
                cql = f'lastModified >= "{since_date}"'
            else:
                cql = 'type = page'

            if self._is_cloud:
                # Cloud doesn't support CQL in v2, use search
                url = f"{self._base_url}/search"
                params = {"cql": cql, "limit": 100}
            else:
                url = f"{self._base_url}/content/search"
                params = {"cql": cql, "limit": 100}

            response = await client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                for item in data.get("results", []):
                    resource = self._parse_page(item.get("content", item))
                    if resource:
                        changes.append(Change(
                            resource_id=resource.id,
                            change_type="modified",
                            timestamp=resource.modified_at or datetime.utcnow(),
                            resource=resource,
                        ))

            # New token is current timestamp
            new_token = datetime.utcnow().strftime("%Y-%m-%d")

            return changes, new_token

        except Exception as e:
            self.log_error("Failed to get Confluence changes", error=e)

        return changes, None

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get schema for Confluence credentials."""
        return {
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": "Confluence instance URL (e.g., https://yoursite.atlassian.net)",
                },
                "email": {
                    "type": "string",
                    "description": "Email address (for Cloud)",
                },
                "api_token": {
                    "type": "string",
                    "description": "API token (for Cloud)",
                },
                "personal_access_token": {
                    "type": "string",
                    "description": "Personal Access Token (for Server/Data Center)",
                },
            },
            "required": ["base_url"],
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get schema for Confluence connector configuration."""
        schema = super().get_config_schema()
        schema["properties"].update({
            "spaces": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Space keys to sync (empty for all)",
            },
            "include_archived": {
                "type": "boolean",
                "default": False,
                "description": "Include archived pages",
            },
            "include_attachments": {
                "type": "boolean",
                "default": True,
                "description": "Sync page attachments",
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
