"""
Notion Connector for AIDocumentIndexer
=======================================

Fetches pages and databases from Notion and indexes them.
Supports:
- Fetching all pages from a workspace
- Fetching specific databases
- Incremental sync (only changed pages)
- Rich text extraction
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NotionPage(BaseModel):
    """Represents a Notion page."""
    id: str
    title: str
    content: str
    url: str
    created_time: datetime
    last_edited_time: datetime
    parent_type: str  # workspace, page_id, database_id
    parent_id: Optional[str] = None
    properties: dict = Field(default_factory=dict)
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Compute content hash for change detection."""
        content_str = f"{self.title}:{self.content}:{self.last_edited_time.isoformat()}"
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


class NotionDatabase(BaseModel):
    """Represents a Notion database."""
    id: str
    title: str
    url: str
    properties: dict
    page_count: int = 0


class NotionConnectorConfig(BaseModel):
    """Configuration for Notion connector."""
    api_key: str
    workspace_id: Optional[str] = None
    database_ids: list[str] = Field(default_factory=list)
    page_ids: list[str] = Field(default_factory=list)
    sync_interval_minutes: int = 60
    include_subpages: bool = True
    max_pages: int = 1000


class NotionConnector:
    """
    Notion API connector for fetching and syncing content.

    Usage:
        connector = NotionConnector(api_key="secret_xxx")
        async for page in connector.fetch_all_pages():
            print(f"Fetched: {page.title}")
    """

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(self, config: NotionConnectorConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._last_sync: Optional[datetime] = None
        self._page_cache: dict[str, str] = {}  # page_id -> content_hash

    @property
    def headers(self) -> dict:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            client = await self._get_client()
            response = await client.get("/users/me")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Notion connection test failed: {e}")
            return False

    async def get_user_info(self) -> Optional[dict]:
        """Get current user/bot info."""
        try:
            client = await self._get_client()
            response = await client.get("/users/me")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None

    # =========================================================================
    # Page Fetching
    # =========================================================================

    async def fetch_all_pages(
        self,
        since: Optional[datetime] = None,
    ) -> AsyncGenerator[NotionPage, None]:
        """
        Fetch all accessible pages.

        Args:
            since: Only fetch pages modified after this time

        Yields:
            NotionPage objects
        """
        # Fetch from search API (gets all accessible pages)
        async for page in self._search_pages(since=since):
            yield page

        # Fetch from specific databases if configured
        for db_id in self.config.database_ids:
            async for page in self._fetch_database_pages(db_id, since=since):
                yield page

    async def _search_pages(
        self,
        query: str = "",
        since: Optional[datetime] = None,
    ) -> AsyncGenerator[NotionPage, None]:
        """Search for pages using Notion search API."""
        client = await self._get_client()
        has_more = True
        start_cursor = None
        page_count = 0

        while has_more and page_count < self.config.max_pages:
            body: dict[str, Any] = {
                "filter": {"property": "object", "value": "page"},
                "page_size": 100,
            }

            if query:
                body["query"] = query

            if start_cursor:
                body["start_cursor"] = start_cursor

            try:
                response = await client.post("/search", json=body)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Search failed: {e}")
                break

            for result in data.get("results", []):
                if result.get("object") != "page":
                    continue

                # Check modification time filter
                last_edited = datetime.fromisoformat(
                    result["last_edited_time"].replace("Z", "+00:00")
                )
                if since and last_edited <= since:
                    continue

                # Fetch full page content
                page = await self._fetch_page_content(result)
                if page:
                    page_count += 1
                    yield page

            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

            # Rate limiting
            await asyncio.sleep(0.1)

    async def _fetch_database_pages(
        self,
        database_id: str,
        since: Optional[datetime] = None,
    ) -> AsyncGenerator[NotionPage, None]:
        """Fetch all pages from a database."""
        client = await self._get_client()
        has_more = True
        start_cursor = None

        while has_more:
            body: dict[str, Any] = {"page_size": 100}

            if start_cursor:
                body["start_cursor"] = start_cursor

            # Filter by last edited time if syncing incrementally
            if since:
                body["filter"] = {
                    "timestamp": "last_edited_time",
                    "last_edited_time": {
                        "after": since.isoformat(),
                    },
                }

            try:
                response = await client.post(
                    f"/databases/{database_id}/query",
                    json=body,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Database query failed: {e}")
                break

            for result in data.get("results", []):
                page = await self._fetch_page_content(result)
                if page:
                    yield page

            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

            await asyncio.sleep(0.1)

    async def _fetch_page_content(
        self,
        page_data: dict,
    ) -> Optional[NotionPage]:
        """Fetch full content of a page."""
        page_id = page_data["id"]

        try:
            # Get page blocks (content)
            blocks = await self._fetch_blocks(page_id)
            content = self._blocks_to_text(blocks)

            # Extract title
            title = self._extract_title(page_data)

            # Get parent info
            parent = page_data.get("parent", {})
            parent_type = parent.get("type", "workspace")
            parent_id = parent.get(parent_type)

            # Build page object
            page = NotionPage(
                id=page_id,
                title=title,
                content=content,
                url=page_data.get("url", f"https://notion.so/{page_id.replace('-', '')}"),
                created_time=datetime.fromisoformat(
                    page_data["created_time"].replace("Z", "+00:00")
                ),
                last_edited_time=datetime.fromisoformat(
                    page_data["last_edited_time"].replace("Z", "+00:00")
                ),
                parent_type=parent_type,
                parent_id=str(parent_id) if parent_id else None,
                properties=page_data.get("properties", {}),
            )
            page.content_hash = page.compute_hash()

            return page

        except Exception as e:
            logger.error(f"Failed to fetch page {page_id}: {e}")
            return None

    async def _fetch_blocks(
        self,
        block_id: str,
        depth: int = 0,
        max_depth: int = 3,
    ) -> list[dict]:
        """Recursively fetch all blocks of a page."""
        if depth > max_depth:
            return []

        client = await self._get_client()
        blocks: list[dict] = []
        has_more = True
        start_cursor = None

        while has_more:
            params = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor

            try:
                response = await client.get(
                    f"/blocks/{block_id}/children",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to fetch blocks: {e}")
                break

            for block in data.get("results", []):
                blocks.append(block)

                # Recursively fetch children if present
                if block.get("has_children"):
                    children = await self._fetch_blocks(
                        block["id"],
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    blocks.extend(children)

            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

            await asyncio.sleep(0.05)

        return blocks

    # =========================================================================
    # Content Extraction
    # =========================================================================

    def _extract_title(self, page_data: dict) -> str:
        """Extract page title from properties."""
        properties = page_data.get("properties", {})

        # Try common title property names
        for prop_name in ["title", "Name", "name", "Title"]:
            prop = properties.get(prop_name, {})
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                return self._rich_text_to_plain(title_parts)

        # Fallback: first title-type property
        for prop in properties.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                return self._rich_text_to_plain(title_parts)

        return "Untitled"

    def _blocks_to_text(self, blocks: list[dict]) -> str:
        """Convert Notion blocks to plain text."""
        text_parts: list[str] = []

        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            if block_type in ("paragraph", "heading_1", "heading_2", "heading_3"):
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                if text:
                    if block_type.startswith("heading"):
                        level = block_type[-1]
                        text = f"{'#' * int(level)} {text}"
                    text_parts.append(text)

            elif block_type == "bulleted_list_item":
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                if text:
                    text_parts.append(f"• {text}")

            elif block_type == "numbered_list_item":
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                if text:
                    text_parts.append(f"1. {text}")

            elif block_type == "to_do":
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                checked = "☑" if block_data.get("checked") else "☐"
                if text:
                    text_parts.append(f"{checked} {text}")

            elif block_type == "toggle":
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                if text:
                    text_parts.append(f"▸ {text}")

            elif block_type == "code":
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                language = block_data.get("language", "")
                if text:
                    text_parts.append(f"```{language}\n{text}\n```")

            elif block_type == "quote":
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                if text:
                    text_parts.append(f"> {text}")

            elif block_type == "callout":
                rich_text = block_data.get("rich_text", [])
                text = self._rich_text_to_plain(rich_text)
                emoji = block_data.get("icon", {}).get("emoji", "💡")
                if text:
                    text_parts.append(f"{emoji} {text}")

            elif block_type == "divider":
                text_parts.append("---")

            elif block_type == "table_row":
                cells = block_data.get("cells", [])
                row_text = " | ".join(
                    self._rich_text_to_plain(cell) for cell in cells
                )
                if row_text:
                    text_parts.append(f"| {row_text} |")

            elif block_type == "child_page":
                title = block_data.get("title", "Untitled")
                text_parts.append(f"📄 [{title}]")

            elif block_type == "child_database":
                title = block_data.get("title", "Untitled")
                text_parts.append(f"📊 [{title}]")

            elif block_type == "link_to_page":
                text_parts.append("🔗 [Linked Page]")

            elif block_type == "embed":
                url = block_data.get("url", "")
                if url:
                    text_parts.append(f"[Embedded: {url}]")

            elif block_type == "bookmark":
                url = block_data.get("url", "")
                if url:
                    text_parts.append(f"[Bookmark: {url}]")

            elif block_type == "image":
                caption = self._rich_text_to_plain(block_data.get("caption", []))
                text_parts.append(f"[Image: {caption or 'No caption'}]")

            elif block_type == "video":
                text_parts.append("[Video]")

            elif block_type == "file":
                name = block_data.get("name", "file")
                text_parts.append(f"[File: {name}]")

            elif block_type == "pdf":
                text_parts.append("[PDF]")

            elif block_type == "equation":
                expression = block_data.get("expression", "")
                if expression:
                    text_parts.append(f"$${expression}$$")

        return "\n\n".join(text_parts)

    def _rich_text_to_plain(self, rich_text: list[dict]) -> str:
        """Convert Notion rich text to plain text."""
        return "".join(
            item.get("plain_text", "")
            for item in rich_text
        )

    # =========================================================================
    # Database Operations
    # =========================================================================

    async def list_databases(self) -> list[NotionDatabase]:
        """List all accessible databases."""
        client = await self._get_client()
        databases: list[NotionDatabase] = []
        has_more = True
        start_cursor = None

        while has_more:
            body: dict[str, Any] = {
                "filter": {"property": "object", "value": "database"},
                "page_size": 100,
            }

            if start_cursor:
                body["start_cursor"] = start_cursor

            try:
                response = await client.post("/search", json=body)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Database search failed: {e}")
                break

            for result in data.get("results", []):
                if result.get("object") != "database":
                    continue

                # Extract title
                title_items = result.get("title", [])
                title = self._rich_text_to_plain(title_items) or "Untitled"

                db = NotionDatabase(
                    id=result["id"],
                    title=title,
                    url=result.get("url", ""),
                    properties=result.get("properties", {}),
                )
                databases.append(db)

            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

            await asyncio.sleep(0.1)

        return databases

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def sync_incremental(self) -> AsyncGenerator[NotionPage, None]:
        """
        Perform incremental sync - only fetch changed pages.

        Yields:
            NotionPage objects that have changed since last sync
        """
        since = self._last_sync

        async for page in self.fetch_all_pages(since=since):
            # Check if content actually changed
            old_hash = self._page_cache.get(page.id)
            if old_hash and old_hash == page.content_hash:
                continue

            # Update cache
            self._page_cache[page.id] = page.content_hash
            yield page

        self._last_sync = datetime.now(timezone.utc)

    def get_sync_status(self) -> dict:
        """Get current sync status."""
        return {
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "cached_pages": len(self._page_cache),
            "config": {
                "databases": self.config.database_ids,
                "pages": self.config.page_ids,
                "max_pages": self.config.max_pages,
            },
        }


# Factory function for easy instantiation
def create_notion_connector(
    api_key: str,
    database_ids: list[str] = None,
    page_ids: list[str] = None,
    max_pages: int = 1000,
) -> NotionConnector:
    """
    Create a Notion connector instance.

    Args:
        api_key: Notion integration API key
        database_ids: List of database IDs to sync
        page_ids: List of specific page IDs to sync
        max_pages: Maximum pages to fetch

    Returns:
        Configured NotionConnector instance
    """
    config = NotionConnectorConfig(
        api_key=api_key,
        database_ids=database_ids or [],
        page_ids=page_ids or [],
        max_pages=max_pages,
    )
    return NotionConnector(config)
