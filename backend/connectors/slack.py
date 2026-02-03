"""
Slack Connector for AIDocumentIndexer
=====================================

Fetches messages, files, and threads from Slack workspaces.
Supports:
- Channel messages and threads
- Direct messages
- File attachments
- Canvas documents
- Incremental sync
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SlackMessage(BaseModel):
    """Represents a Slack message."""
    ts: str  # Slack timestamp (unique ID)
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    text: str
    thread_ts: Optional[str] = None
    reply_count: int = 0
    reactions: list[dict] = Field(default_factory=list)
    files: list[dict] = Field(default_factory=list)
    timestamp: datetime
    permalink: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.channel_id}:{self.ts}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.text}:{self.ts}".encode()).hexdigest()[:16]


class SlackChannel(BaseModel):
    """Represents a Slack channel."""
    id: str
    name: str
    is_private: bool = False
    is_archived: bool = False
    topic: str = ""
    purpose: str = ""
    member_count: int = 0
    created: datetime


class SlackFile(BaseModel):
    """Represents a Slack file."""
    id: str
    name: str
    title: str
    mimetype: str
    filetype: str
    size: int
    url_private: str
    content: Optional[str] = None
    channel_id: Optional[str] = None
    user_id: str
    created: datetime


class SlackConnectorConfig(BaseModel):
    """Configuration for Slack connector."""
    bot_token: str  # xoxb-...
    user_token: Optional[str] = None  # xoxp-... (for user-level access)
    channel_ids: list[str] = Field(default_factory=list)
    include_private: bool = False
    include_dms: bool = False
    include_threads: bool = True
    include_files: bool = True
    max_messages_per_channel: int = 1000
    sync_interval_minutes: int = 30
    oldest_days: int = 90  # Only fetch messages from last N days


class SlackConnector:
    """
    Slack API connector for fetching workspace content.

    Usage:
        connector = SlackConnector(bot_token="xoxb-xxx")
        async for message in connector.fetch_channel_messages("C123456"):
            print(f"{message.user_name}: {message.text}")
    """

    BASE_URL = "https://slack.com/api"

    def __init__(self, config: SlackConnectorConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._user_cache: dict[str, str] = {}  # user_id -> display_name
        self._channel_cache: dict[str, str] = {}  # channel_id -> channel_name

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.bot_token}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=30.0,
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
            data = response.json()

            if not data.get("ok"):
                error = data.get("error", "Unknown error")
                logger.error(f"Slack API error: {error}")
                return None

            return data
        except Exception as e:
            logger.error(f"Slack request failed: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test the API connection."""
        data = await self._request("GET", "/auth.test")
        return data is not None and data.get("ok")

    async def get_workspace_info(self) -> Optional[dict]:
        """Get workspace information."""
        return await self._request("GET", "/team.info")

    # =========================================================================
    # User Management
    # =========================================================================

    async def _get_user_name(self, user_id: str) -> str:
        """Get user display name, with caching."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        data = await self._request("GET", "/users.info", params={"user": user_id})
        if data and "user" in data:
            name = data["user"].get("real_name") or data["user"].get("name", user_id)
            self._user_cache[user_id] = name
            return name

        return user_id

    async def _get_channel_name(self, channel_id: str) -> str:
        """Get channel name, with caching."""
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id]

        data = await self._request(
            "GET", "/conversations.info", params={"channel": channel_id}
        )
        if data and "channel" in data:
            name = data["channel"].get("name", channel_id)
            self._channel_cache[channel_id] = name
            return name

        return channel_id

    # =========================================================================
    # Channel Operations
    # =========================================================================

    async def list_channels(
        self,
        include_private: bool = False,
    ) -> AsyncGenerator[SlackChannel, None]:
        """List all accessible channels."""
        cursor = None
        types = "public_channel"
        if include_private:
            types += ",private_channel"

        while True:
            params: dict[str, Any] = {
                "types": types,
                "limit": 200,
                "exclude_archived": True,
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._request("GET", "/conversations.list", params=params)
            if not data:
                break

            for channel in data.get("channels", []):
                yield SlackChannel(
                    id=channel["id"],
                    name=channel["name"],
                    is_private=channel.get("is_private", False),
                    is_archived=channel.get("is_archived", False),
                    topic=channel.get("topic", {}).get("value", ""),
                    purpose=channel.get("purpose", {}).get("value", ""),
                    member_count=channel.get("num_members", 0),
                    created=datetime.fromtimestamp(
                        channel["created"], tz=timezone.utc
                    ),
                )

            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

            await asyncio.sleep(0.5)  # Rate limiting

    # =========================================================================
    # Message Operations
    # =========================================================================

    async def fetch_channel_messages(
        self,
        channel_id: str,
        oldest: Optional[datetime] = None,
        latest: Optional[datetime] = None,
    ) -> AsyncGenerator[SlackMessage, None]:
        """
        Fetch messages from a channel.

        Args:
            channel_id: Slack channel ID
            oldest: Only fetch messages after this time
            latest: Only fetch messages before this time

        Yields:
            SlackMessage objects
        """
        channel_name = await self._get_channel_name(channel_id)
        cursor = None
        message_count = 0

        # Calculate oldest timestamp
        if oldest is None:
            oldest = datetime.now(timezone.utc) - timedelta(days=self.config.oldest_days)

        while message_count < self.config.max_messages_per_channel:
            params: dict[str, Any] = {
                "channel": channel_id,
                "limit": 200,
                "oldest": str(oldest.timestamp()),
            }
            if latest:
                params["latest"] = str(latest.timestamp())
            if cursor:
                params["cursor"] = cursor

            data = await self._request("GET", "/conversations.history", params=params)
            if not data:
                break

            for msg in data.get("messages", []):
                if msg.get("type") != "message":
                    continue
                if msg.get("subtype") in ("channel_join", "channel_leave", "bot_message"):
                    continue

                user_id = msg.get("user", "")
                user_name = await self._get_user_name(user_id) if user_id else "Unknown"

                message = SlackMessage(
                    ts=msg["ts"],
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=user_id,
                    user_name=user_name,
                    text=msg.get("text", ""),
                    thread_ts=msg.get("thread_ts"),
                    reply_count=msg.get("reply_count", 0),
                    reactions=msg.get("reactions", []),
                    files=msg.get("files", []),
                    timestamp=datetime.fromtimestamp(float(msg["ts"]), tz=timezone.utc),
                    permalink=msg.get("permalink"),
                )

                message_count += 1
                yield message

                # Fetch thread replies if enabled
                if self.config.include_threads and msg.get("reply_count", 0) > 0:
                    async for reply in self._fetch_thread_replies(
                        channel_id, msg["ts"], channel_name
                    ):
                        message_count += 1
                        yield reply

            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

            await asyncio.sleep(0.5)

    async def _fetch_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
        channel_name: str,
    ) -> AsyncGenerator[SlackMessage, None]:
        """Fetch replies in a thread."""
        cursor = None

        while True:
            params: dict[str, Any] = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._request("GET", "/conversations.replies", params=params)
            if not data:
                break

            for msg in data.get("messages", [])[1:]:  # Skip parent message
                user_id = msg.get("user", "")
                user_name = await self._get_user_name(user_id) if user_id else "Unknown"

                yield SlackMessage(
                    ts=msg["ts"],
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=user_id,
                    user_name=user_name,
                    text=msg.get("text", ""),
                    thread_ts=thread_ts,
                    reply_count=0,
                    reactions=msg.get("reactions", []),
                    files=msg.get("files", []),
                    timestamp=datetime.fromtimestamp(float(msg["ts"]), tz=timezone.utc),
                )

            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

            await asyncio.sleep(0.3)

    # =========================================================================
    # File Operations
    # =========================================================================

    async def fetch_files(
        self,
        channel_id: Optional[str] = None,
    ) -> AsyncGenerator[SlackFile, None]:
        """Fetch files from workspace or specific channel."""
        if not self.config.include_files:
            return

        page = 1
        while True:
            params: dict[str, Any] = {
                "count": 100,
                "page": page,
            }
            if channel_id:
                params["channel"] = channel_id

            data = await self._request("GET", "/files.list", params=params)
            if not data:
                break

            for file in data.get("files", []):
                yield SlackFile(
                    id=file["id"],
                    name=file.get("name", ""),
                    title=file.get("title", ""),
                    mimetype=file.get("mimetype", ""),
                    filetype=file.get("filetype", ""),
                    size=file.get("size", 0),
                    url_private=file.get("url_private", ""),
                    channel_id=file.get("channels", [None])[0],
                    user_id=file.get("user", ""),
                    created=datetime.fromtimestamp(
                        file.get("created", 0), tz=timezone.utc
                    ),
                )

            paging = data.get("paging", {})
            if page >= paging.get("pages", 1):
                break

            page += 1
            await asyncio.sleep(0.5)

    async def download_file(self, file: SlackFile) -> Optional[bytes]:
        """Download file content."""
        if not file.url_private:
            return None

        client = await self._get_client()
        try:
            response = await client.get(
                file.url_private,
                headers={"Authorization": f"Bearer {self.config.bot_token}"},
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download file {file.id}: {e}")
            return None

    # =========================================================================
    # Full Sync
    # =========================================================================

    async def sync_all(
        self,
        channel_ids: Optional[list[str]] = None,
    ) -> AsyncGenerator[SlackMessage | SlackFile, None]:
        """
        Sync all messages and files.

        Args:
            channel_ids: Specific channels to sync (or all if None)

        Yields:
            SlackMessage and SlackFile objects
        """
        # Get channels to sync
        if channel_ids:
            channels = channel_ids
        elif self.config.channel_ids:
            channels = self.config.channel_ids
        else:
            channels = [
                ch.id async for ch in self.list_channels(self.config.include_private)
            ]

        # Fetch messages from each channel
        for channel_id in channels:
            logger.info(f"Syncing Slack channel: {channel_id}")
            async for message in self.fetch_channel_messages(channel_id):
                yield message

        # Fetch files
        if self.config.include_files:
            async for file in self.fetch_files():
                yield file


# Import timedelta for the connector
from datetime import timedelta


def create_slack_connector(
    bot_token: str,
    channel_ids: list[str] = None,
    include_private: bool = False,
    include_threads: bool = True,
    include_files: bool = True,
) -> SlackConnector:
    """
    Create a Slack connector instance.

    Args:
        bot_token: Slack bot token (xoxb-...)
        channel_ids: List of channel IDs to sync
        include_private: Include private channels
        include_threads: Include thread replies
        include_files: Include file attachments

    Returns:
        Configured SlackConnector instance
    """
    config = SlackConnectorConfig(
        bot_token=bot_token,
        channel_ids=channel_ids or [],
        include_private=include_private,
        include_threads=include_threads,
        include_files=include_files,
    )
    return SlackConnector(config)
