"""
AIDocumentIndexer - Notion Connector
====================================

Connector for syncing content from Notion workspaces.

Supports:
- Pages and databases
- Nested content
- Rich text extraction
- Database queries
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio

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

NOTION_API_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"


@ConnectorRegistry.register(ConnectorType.NOTION)
class NotionConnector(BaseConnector):
    """
    Connector for Notion workspaces.

    Uses the Notion API to sync pages and databases.
    Supports both internal integration tokens and OAuth.
    """

    connector_type = ConnectorType.NOTION
    display_name = "Notion"
    description = "Sync pages and databases from Notion workspaces"
    icon = "file-text"

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.config.credentials.get('access_token', '')}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=NOTION_BASE_URL,
                headers=self._headers,
                timeout=30.0,
            )
        return self._client

    async def authenticate(self) -> bool:
        """Authenticate with Notion API."""
        try:
            client = await self._get_client()
            response = await client.get("/users/me")

            if response.status_code == 200:
                self._authenticated = True
                user_data = response.json()
                self.log_info("Authenticated with Notion", user=user_data.get("name"))
                return True
            else:
                self.log_error("Notion authentication failed", status=response.status_code)
                return False

        except Exception as e:
            self.log_error("Notion authentication error", error=e)
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh OAuth tokens if applicable."""
        # Notion OAuth tokens don't expire for internal integrations
        # For public integrations, implement refresh logic here
        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List pages and databases from Notion."""
        client = await self._get_client()
        resources = []

        try:
            if folder_id:
                # List children of a specific page/database
                resources.extend(await self._list_page_children(folder_id, client))
            else:
                # Search all accessible content
                payload = {
                    "page_size": min(page_size, 100),
                }
                if page_token:
                    payload["start_cursor"] = page_token

                response = await client.post("/search", json=payload)

                if response.status_code == 200:
                    data = response.json()

                    for item in data.get("results", []):
                        resource = self._parse_notion_object(item)
                        if resource:
                            resources.append(resource)

                    next_cursor = data.get("next_cursor") if data.get("has_more") else None
                    return resources, next_cursor
                else:
                    self.log_error("Notion search failed", status=response.status_code)

        except Exception as e:
            self.log_error("Failed to list Notion resources", error=e)

        return resources, None

    async def _list_page_children(
        self,
        page_id: str,
        client: httpx.AsyncClient,
    ) -> List[Resource]:
        """List children of a page."""
        resources = []

        try:
            response = await client.get(f"/blocks/{page_id}/children")

            if response.status_code == 200:
                data = response.json()

                for block in data.get("results", []):
                    if block.get("type") == "child_page":
                        resource = Resource(
                            id=block["id"],
                            name=block.get("child_page", {}).get("title", "Untitled"),
                            resource_type=ResourceType.PAGE,
                            parent_id=page_id,
                            created_at=datetime.fromisoformat(block["created_time"].replace("Z", "+00:00")),
                            modified_at=datetime.fromisoformat(block["last_edited_time"].replace("Z", "+00:00")),
                            metadata={"type": "child_page"},
                        )
                        resources.append(resource)
                    elif block.get("type") == "child_database":
                        resource = Resource(
                            id=block["id"],
                            name=block.get("child_database", {}).get("title", "Untitled Database"),
                            resource_type=ResourceType.DATABASE,
                            parent_id=page_id,
                            created_at=datetime.fromisoformat(block["created_time"].replace("Z", "+00:00")),
                            modified_at=datetime.fromisoformat(block["last_edited_time"].replace("Z", "+00:00")),
                            metadata={"type": "child_database"},
                        )
                        resources.append(resource)

        except Exception as e:
            self.log_error("Failed to list page children", error=e, page_id=page_id)

        return resources

    def _parse_notion_object(self, obj: Dict[str, Any]) -> Optional[Resource]:
        """Parse a Notion API object into a Resource."""
        try:
            obj_type = obj.get("object")
            obj_id = obj.get("id")

            if obj_type == "page":
                # Extract title from properties
                title = "Untitled"
                props = obj.get("properties", {})

                for prop in props.values():
                    if prop.get("type") == "title":
                        title_parts = prop.get("title", [])
                        if title_parts:
                            title = "".join(t.get("plain_text", "") for t in title_parts)
                        break

                # Get parent info
                parent = obj.get("parent", {})
                parent_id = None
                if parent.get("type") == "page_id":
                    parent_id = parent.get("page_id")
                elif parent.get("type") == "database_id":
                    parent_id = parent.get("database_id")

                return Resource(
                    id=obj_id,
                    name=title or "Untitled",
                    resource_type=ResourceType.PAGE,
                    parent_id=parent_id,
                    web_url=obj.get("url"),
                    created_at=datetime.fromisoformat(obj["created_time"].replace("Z", "+00:00")),
                    modified_at=datetime.fromisoformat(obj["last_edited_time"].replace("Z", "+00:00")),
                    metadata={
                        "type": "page",
                        "icon": obj.get("icon"),
                        "cover": obj.get("cover"),
                        "archived": obj.get("archived", False),
                    },
                )

            elif obj_type == "database":
                # Extract database title
                title_parts = obj.get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_parts) if title_parts else "Untitled Database"

                parent = obj.get("parent", {})
                parent_id = parent.get("page_id") if parent.get("type") == "page_id" else None

                return Resource(
                    id=obj_id,
                    name=title,
                    resource_type=ResourceType.DATABASE,
                    parent_id=parent_id,
                    web_url=obj.get("url"),
                    created_at=datetime.fromisoformat(obj["created_time"].replace("Z", "+00:00")),
                    modified_at=datetime.fromisoformat(obj["last_edited_time"].replace("Z", "+00:00")),
                    metadata={
                        "type": "database",
                        "icon": obj.get("icon"),
                        "archived": obj.get("archived", False),
                        "is_inline": obj.get("is_inline", False),
                    },
                )

        except Exception as e:
            self.log_warning("Failed to parse Notion object", error=str(e), obj_id=obj.get("id"))

        return None

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a specific page or database."""
        client = await self._get_client()

        # Try as page first
        try:
            response = await client.get(f"/pages/{resource_id}")
            if response.status_code == 200:
                return self._parse_notion_object(response.json())
        except Exception:
            pass

        # Try as database
        try:
            response = await client.get(f"/databases/{resource_id}")
            if response.status_code == 200:
                return self._parse_notion_object(response.json())
        except Exception:
            pass

        return None

    async def download_resource(self, resource_id: str) -> Optional[bytes]:
        """Download page content as text."""
        client = await self._get_client()

        try:
            # Get all blocks in the page
            content_parts = []
            cursor = None

            while True:
                url = f"/blocks/{resource_id}/children"
                if cursor:
                    url += f"?start_cursor={cursor}"

                response = await client.get(url)

                if response.status_code != 200:
                    break

                data = response.json()

                for block in data.get("results", []):
                    text = self._extract_block_text(block)
                    if text:
                        content_parts.append(text)

                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")

            if content_parts:
                return "\n\n".join(content_parts).encode("utf-8")

        except Exception as e:
            self.log_error("Failed to download Notion page", error=e, resource_id=resource_id)

        return None

    def _extract_block_text(self, block: Dict[str, Any]) -> str:
        """Extract plain text from a Notion block."""
        block_type = block.get("type")
        block_data = block.get(block_type, {})

        # Text-based blocks
        text_blocks = [
            "paragraph", "heading_1", "heading_2", "heading_3",
            "bulleted_list_item", "numbered_list_item", "to_do",
            "toggle", "quote", "callout",
        ]

        if block_type in text_blocks:
            rich_text = block_data.get("rich_text", [])
            return "".join(t.get("plain_text", "") for t in rich_text)

        elif block_type == "code":
            rich_text = block_data.get("rich_text", [])
            language = block_data.get("language", "")
            code = "".join(t.get("plain_text", "") for t in rich_text)
            return f"```{language}\n{code}\n```"

        elif block_type == "equation":
            return block_data.get("expression", "")

        elif block_type == "divider":
            return "---"

        return ""

    async def get_changes(
        self,
        since_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """
        Get changes since last sync.

        Notion doesn't have a native changes API, so we do a full search
        and compare modification times.
        """
        # For Notion, we typically need to do a full resync
        # or use webhooks for real-time updates

        # Return empty for now - full sync is recommended
        return [], None

    def get_oauth_url(self, state: str) -> Optional[str]:
        """Get Notion OAuth URL."""
        client_id = self.config.credentials.get("client_id")
        redirect_uri = self.config.credentials.get("redirect_uri")

        if client_id and redirect_uri:
            return (
                f"https://api.notion.com/v1/oauth/authorize"
                f"?client_id={client_id}"
                f"&response_type=code"
                f"&owner=user"
                f"&redirect_uri={redirect_uri}"
                f"&state={state}"
            )
        return None

    async def handle_oauth_callback(
        self,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        """Exchange OAuth code for access token."""
        client_id = self.config.credentials.get("client_id")
        client_secret = self.config.credentials.get("client_secret")
        redirect_uri = self.config.credentials.get("redirect_uri")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.notion.com/v1/oauth/token",
                auth=(client_id, client_secret),
                json={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "access_token": data["access_token"],
                    "workspace_id": data.get("workspace_id"),
                    "workspace_name": data.get("workspace_name"),
                    "workspace_icon": data.get("workspace_icon"),
                    "bot_id": data.get("bot_id"),
                }
            else:
                raise Exception(f"OAuth token exchange failed: {response.text}")

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get schema for Notion credentials."""
        return {
            "type": "object",
            "properties": {
                "access_token": {
                    "type": "string",
                    "description": "Notion integration token or OAuth access token",
                },
                "client_id": {
                    "type": "string",
                    "description": "OAuth client ID (for public integrations)",
                },
                "client_secret": {
                    "type": "string",
                    "description": "OAuth client secret (for public integrations)",
                },
                "redirect_uri": {
                    "type": "string",
                    "description": "OAuth redirect URI",
                },
            },
            "required": ["access_token"],
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get schema for Notion connector configuration."""
        schema = super().get_config_schema()
        schema["properties"].update({
            "include_databases": {
                "type": "boolean",
                "default": True,
                "description": "Include database contents in sync",
            },
            "max_depth": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum depth for nested pages",
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
