"""
Confluence Connector for AIDocumentIndexer
==========================================

Fetches pages and spaces from Atlassian Confluence.
Supports:
- Cloud and Data Center/Server instances
- Space filtering
- Page hierarchy traversal
- Attachments
- Incremental sync via CQL
"""

import asyncio
import base64
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConfluencePage(BaseModel):
    """Represents a Confluence page."""
    id: str
    title: str
    space_key: str
    space_name: str = ""
    body_html: str = ""
    body_text: str = ""
    url: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: str = ""
    version: int = 1
    parent_id: Optional[str] = None
    labels: list[str] = Field(default_factory=list)
    ancestors: list[str] = Field(default_factory=list)
    attachments: list[dict] = Field(default_factory=list)

    @property
    def content(self) -> str:
        """Get the best text content."""
        return self.body_text or self._html_to_text(self.body_html)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.title}:{self.body_text}:{self.version}".encode()).hexdigest()[:16]

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Simple HTML to text conversion."""
        if not html:
            return ""
        # Remove scripts and styles
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Decode entities
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        return text


class ConfluenceSpace(BaseModel):
    """Represents a Confluence space."""
    key: str
    name: str
    type: str  # global, personal
    description: str = ""
    homepage_id: Optional[str] = None
    url: str = ""


class ConfluenceAttachment(BaseModel):
    """Represents a Confluence attachment."""
    id: str
    page_id: str
    title: str
    filename: str
    media_type: str
    file_size: int
    download_url: str
    created_at: Optional[datetime] = None


class ConfluenceConnectorConfig(BaseModel):
    """Configuration for Confluence connector."""
    # Authentication
    base_url: str  # e.g., https://yourcompany.atlassian.net/wiki
    username: str  # email for Cloud
    api_token: str  # API token for Cloud, password for Server/DC

    # Instance type
    is_cloud: bool = True  # Cloud vs Data Center/Server

    # Sync settings
    space_keys: list[str] = Field(default_factory=list)  # Empty = all spaces
    include_archived: bool = False
    include_attachments: bool = True
    max_attachment_size_mb: int = 50
    max_pages: int = 10000
    sync_interval_minutes: int = 60

    # CQL filters
    cql_filter: Optional[str] = None  # Additional CQL conditions


class ConfluenceConnector:
    """
    Confluence API connector for fetching pages and spaces.

    Usage:
        connector = ConfluenceConnector(config)
        async for page in connector.fetch_pages():
            print(f"Page: {page.title}")
    """

    def __init__(self, config: ConfluenceConnectorConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

        # API paths differ between Cloud and Server/DC
        if config.is_cloud:
            self.api_base = f"{config.base_url.rstrip('/')}/rest/api"
        else:
            self.api_base = f"{config.base_url.rstrip('/')}/rest/api"

    @property
    def _auth_header(self) -> str:
        """Generate Basic Auth header."""
        credentials = f"{self.config.username}:{self.config.api_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    @property
    def headers(self) -> dict:
        return {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
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
        url = f"{self.api_base}{endpoint}"
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Confluence API error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Confluence request error: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test the API connection."""
        data = await self._request("GET", "/space", params={"limit": 1})
        return data is not None

    async def get_current_user(self) -> Optional[dict]:
        """Get current user info."""
        if self.config.is_cloud:
            return await self._request("GET", "/user/current")
        else:
            return await self._request("GET", "/user/current")

    async def list_spaces(self) -> AsyncGenerator[ConfluenceSpace, None]:
        """List all accessible spaces."""
        start = 0
        limit = 100

        while True:
            params = {
                "start": start,
                "limit": limit,
                "expand": "description.plain,homepage",
            }

            data = await self._request("GET", "/space", params=params)
            if not data:
                break

            for space_data in data.get("results", []):
                # Filter by configured space keys
                if self.config.space_keys and space_data["key"] not in self.config.space_keys:
                    continue

                # Skip archived if not included
                if not self.config.include_archived and space_data.get("status") == "archived":
                    continue

                yield ConfluenceSpace(
                    key=space_data["key"],
                    name=space_data["name"],
                    type=space_data.get("type", "global"),
                    description=space_data.get("description", {}).get("plain", {}).get("value", ""),
                    homepage_id=space_data.get("homepage", {}).get("id"),
                    url=f"{self.config.base_url}/spaces/{space_data['key']}",
                )

            # Check for more results
            if data.get("size", 0) < limit:
                break
            start += limit
            await asyncio.sleep(0.1)

    async def fetch_pages(
        self,
        space_key: Optional[str] = None,
        modified_since: Optional[datetime] = None,
        max_pages: Optional[int] = None,
    ) -> AsyncGenerator[ConfluencePage, None]:
        """
        Fetch pages from Confluence.

        Args:
            space_key: Specific space to fetch from
            modified_since: Only fetch pages modified after this date
            max_pages: Maximum pages to fetch

        Yields:
            ConfluencePage objects
        """
        max_pages = max_pages or self.config.max_pages
        count = 0

        # Build CQL query
        cql_parts = ["type=page"]

        if space_key:
            cql_parts.append(f"space={space_key}")
        elif self.config.space_keys:
            spaces = " OR ".join(f"space={s}" for s in self.config.space_keys)
            cql_parts.append(f"({spaces})")

        if modified_since:
            date_str = modified_since.strftime("%Y-%m-%d %H:%M")
            cql_parts.append(f"lastModified >= \"{date_str}\"")

        if self.config.cql_filter:
            cql_parts.append(self.config.cql_filter)

        cql = " AND ".join(cql_parts)
        cql += " ORDER BY lastModified DESC"

        start = 0
        limit = 50

        while count < max_pages:
            params = {
                "cql": cql,
                "start": start,
                "limit": min(limit, max_pages - count),
                "expand": "body.storage,version,ancestors,space,history.createdBy,metadata.labels",
            }

            data = await self._request("GET", "/content/search", params=params)
            if not data:
                break

            for page_data in data.get("results", []):
                page = await self._parse_page(page_data)
                if page:
                    count += 1
                    yield page

                    if count >= max_pages:
                        break

            # Check for more results
            if data.get("size", 0) < limit:
                break
            start += limit
            await asyncio.sleep(0.1)

    async def _parse_page(self, data: dict) -> Optional[ConfluencePage]:
        """Parse page data from API response."""
        try:
            # Extract body content
            body_html = data.get("body", {}).get("storage", {}).get("value", "")

            # Extract dates
            created_at = None
            updated_at = None
            created_by = ""

            history = data.get("history", {})
            if history.get("createdDate"):
                created_at = datetime.fromisoformat(history["createdDate"].replace("Z", "+00:00"))
            if history.get("createdBy", {}).get("displayName"):
                created_by = history["createdBy"]["displayName"]

            version_data = data.get("version", {})
            if version_data.get("when"):
                updated_at = datetime.fromisoformat(version_data["when"].replace("Z", "+00:00"))

            # Extract labels
            labels = []
            label_data = data.get("metadata", {}).get("labels", {}).get("results", [])
            for label in label_data:
                labels.append(label.get("name", ""))

            # Extract ancestors (parent pages)
            ancestors = []
            for ancestor in data.get("ancestors", []):
                ancestors.append(ancestor.get("id", ""))

            # Get parent ID (immediate parent)
            parent_id = ancestors[-1] if ancestors else None

            # Get space info
            space = data.get("space", {})

            # Build page URL
            page_url = f"{self.config.base_url}{data.get('_links', {}).get('webui', '')}"

            page = ConfluencePage(
                id=data["id"],
                title=data.get("title", "Untitled"),
                space_key=space.get("key", ""),
                space_name=space.get("name", ""),
                body_html=body_html,
                body_text=ConfluencePage._html_to_text(body_html),
                url=page_url,
                created_at=created_at,
                updated_at=updated_at,
                created_by=created_by,
                version=version_data.get("number", 1),
                parent_id=parent_id,
                labels=labels,
                ancestors=ancestors,
            )

            # Fetch attachments if enabled
            if self.config.include_attachments:
                attachments = await self._get_page_attachments(data["id"])
                page.attachments = [
                    {
                        "id": att.id,
                        "filename": att.filename,
                        "media_type": att.media_type,
                        "size": att.file_size,
                    }
                    for att in attachments
                ]

            return page

        except Exception as e:
            logger.error(f"Error parsing page {data.get('id')}: {e}")
            return None

    async def _get_page_attachments(self, page_id: str) -> list[ConfluenceAttachment]:
        """Get attachments for a page."""
        attachments = []

        data = await self._request(
            "GET",
            f"/content/{page_id}/child/attachment",
            params={"expand": "version"},
        )

        if not data:
            return attachments

        for att_data in data.get("results", []):
            try:
                # Check file size
                file_size = att_data.get("extensions", {}).get("fileSize", 0)
                max_size = self.config.max_attachment_size_mb * 1024 * 1024

                if file_size > max_size:
                    logger.debug(f"Skipping large attachment: {att_data.get('title')}")
                    continue

                created_at = None
                version = att_data.get("version", {})
                if version.get("when"):
                    created_at = datetime.fromisoformat(version["when"].replace("Z", "+00:00"))

                download_url = f"{self.config.base_url}{att_data.get('_links', {}).get('download', '')}"

                attachments.append(ConfluenceAttachment(
                    id=att_data["id"],
                    page_id=page_id,
                    title=att_data.get("title", ""),
                    filename=att_data.get("title", ""),
                    media_type=att_data.get("extensions", {}).get("mediaType", "application/octet-stream"),
                    file_size=file_size,
                    download_url=download_url,
                    created_at=created_at,
                ))
            except Exception as e:
                logger.error(f"Error parsing attachment: {e}")

        return attachments

    async def get_page(self, page_id: str) -> Optional[ConfluencePage]:
        """Get a specific page by ID."""
        data = await self._request(
            "GET",
            f"/content/{page_id}",
            params={"expand": "body.storage,version,ancestors,space,history.createdBy,metadata.labels"},
        )

        if not data:
            return None

        return await self._parse_page(data)

    async def download_attachment(self, attachment: ConfluenceAttachment) -> Optional[bytes]:
        """Download attachment content."""
        client = await self._get_client()
        try:
            response = await client.get(attachment.download_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading attachment {attachment.filename}: {e}")
            return None

    async def search(
        self,
        query: str,
        space_key: Optional[str] = None,
        limit: int = 20,
    ) -> AsyncGenerator[ConfluencePage, None]:
        """
        Search pages using CQL.

        Args:
            query: Search text
            space_key: Limit to specific space
            limit: Maximum results

        Yields:
            Matching ConfluencePage objects
        """
        cql = f'type=page AND (title~"{query}" OR text~"{query}")'

        if space_key:
            cql += f" AND space={space_key}"
        elif self.config.space_keys:
            spaces = " OR ".join(f"space={s}" for s in self.config.space_keys)
            cql += f" AND ({spaces})"

        params = {
            "cql": cql,
            "limit": limit,
            "expand": "body.storage,version,space",
        }

        data = await self._request("GET", "/content/search", params=params)
        if not data:
            return

        for page_data in data.get("results", []):
            page = await self._parse_page(page_data)
            if page:
                yield page


def create_confluence_connector(
    base_url: str,
    username: str,
    api_token: str,
    space_keys: list[str] = None,
    is_cloud: bool = True,
) -> ConfluenceConnector:
    """
    Create a Confluence connector instance.

    Args:
        base_url: Confluence base URL (e.g., https://company.atlassian.net/wiki)
        username: Username or email
        api_token: API token (Cloud) or password (Server/DC)
        space_keys: List of space keys to sync (empty = all)
        is_cloud: True for Atlassian Cloud, False for Server/Data Center

    Returns:
        Configured ConfluenceConnector instance
    """
    config = ConfluenceConnectorConfig(
        base_url=base_url,
        username=username,
        api_token=api_token,
        space_keys=space_keys or [],
        is_cloud=is_cloud,
    )
    return ConfluenceConnector(config)
