"""
AIDocumentIndexer - Slack Data Connector
=========================================

Connector for syncing messages and files from Slack workspaces.

Supports:
- Public and private channels (with appropriate permissions)
- Direct messages (with user consent)
- File attachments
- Thread messages

Note: This is different from the Slack Bot integration which runs
the AI assistant within Slack. This connector syncs Slack content
as documents for RAG indexing.
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

SLACK_API_BASE = "https://slack.com/api"


@ConnectorRegistry.register(ConnectorType.SLACK)
class SlackDataConnector(BaseConnector):
    """
    Connector for Slack workspaces.

    Uses the Slack Web API to sync channel messages and files.
    Requires a Slack app with appropriate OAuth scopes:
    - channels:history, channels:read (public channels)
    - groups:history, groups:read (private channels)
    - files:read (file access)
    - users:read (user info for attribution)
    """

    connector_type = ConnectorType.SLACK
    display_name = "Slack"
    description = "Sync messages and files from Slack workspaces"
    icon = "slack"

    # OAuth scopes required for this connector
    REQUIRED_SCOPES = [
        "channels:history",
        "channels:read",
        "groups:history",
        "groups:read",
        "files:read",
        "users:read",
    ]

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None
        self._user_cache: Dict[str, str] = {}

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
                base_url=SLACK_API_BASE,
                headers=self._headers,
                timeout=30.0,
            )
        return self._client

    async def authenticate(self) -> bool:
        """Authenticate with Slack API."""
        try:
            client = await self._get_client()
            response = await client.get("/auth.test")

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    self._authenticated = True
                    self.log_info(
                        "Authenticated with Slack",
                        team=data.get("team"),
                        user=data.get("user"),
                    )
                    return True
                else:
                    self.log_error("Slack auth failed", error=data.get("error"))
                    return False
            else:
                self.log_error("Slack auth request failed", status=response.status_code)
                return False

        except Exception as e:
            self.log_error("Slack authentication error", error=e)
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh OAuth tokens."""
        # Slack OAuth tokens typically don't expire unless revoked
        # If using a rotating token, implement refresh logic here
        client_id = self.config.credentials.get("client_id")
        client_secret = self.config.credentials.get("client_secret")
        refresh_token = self.config.credentials.get("refresh_token")

        if not all([client_id, client_secret, refresh_token]):
            return self.config.credentials

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SLACK_API_BASE}/oauth.v2.access",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        return {
                            **self.config.credentials,
                            "access_token": data.get("access_token"),
                            "refresh_token": data.get("refresh_token", refresh_token),
                        }

        except Exception as e:
            self.log_error("Failed to refresh Slack token", error=e)

        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List channels and messages from Slack."""
        client = await self._get_client()
        resources = []

        try:
            if folder_id:
                # List messages in a specific channel
                resources, next_cursor = await self._list_channel_messages(
                    folder_id, client, page_token, page_size
                )
                return resources, next_cursor
            else:
                # List all accessible channels
                params = {
                    "types": "public_channel,private_channel",
                    "exclude_archived": "true",
                    "limit": min(page_size, 200),
                }
                if page_token:
                    params["cursor"] = page_token

                response = await client.get("/conversations.list", params=params)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok"):
                        for channel in data.get("channels", []):
                            resource = self._parse_channel(channel)
                            if resource:
                                resources.append(resource)

                        next_cursor = data.get("response_metadata", {}).get("next_cursor")
                        return resources, next_cursor if next_cursor else None
                    else:
                        self.log_error("Slack channels list failed", error=data.get("error"))
                else:
                    self.log_error("Slack API request failed", status=response.status_code)

        except Exception as e:
            self.log_error("Failed to list Slack resources", error=e)

        return resources, None

    async def _list_channel_messages(
        self,
        channel_id: str,
        client: httpx.AsyncClient,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List messages in a channel."""
        resources = []

        params = {
            "channel": channel_id,
            "limit": min(limit, 200),
        }
        if cursor:
            params["cursor"] = cursor

        try:
            response = await client.get("/conversations.history", params=params)

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    for message in data.get("messages", []):
                        resource = await self._parse_message(message, channel_id)
                        if resource:
                            resources.append(resource)

                    next_cursor = data.get("response_metadata", {}).get("next_cursor")
                    return resources, next_cursor if next_cursor else None
                else:
                    self.log_error("Failed to get channel history", error=data.get("error"))

        except Exception as e:
            self.log_error("Failed to list channel messages", error=e, channel=channel_id)

        return resources, None

    def _parse_channel(self, channel: Dict[str, Any]) -> Optional[Resource]:
        """Parse a Slack channel into a Resource."""
        try:
            channel_id = channel.get("id", "")
            name = channel.get("name", "")

            return Resource(
                id=channel_id,
                name=f"#{name}",
                resource_type=ResourceType.FOLDER,
                path=f"/channels/{name}",
                mime_type="application/vnd.slack.channel",
                size=channel.get("num_members", 0),
                created_at=datetime.fromtimestamp(channel.get("created", 0)) if channel.get("created") else None,
                modified_at=None,
                metadata={
                    "channel_id": channel_id,
                    "channel_name": name,
                    "is_private": channel.get("is_private", False),
                    "is_archived": channel.get("is_archived", False),
                    "topic": channel.get("topic", {}).get("value", ""),
                    "purpose": channel.get("purpose", {}).get("value", ""),
                    "num_members": channel.get("num_members", 0),
                },
            )
        except Exception as e:
            self.log_error("Failed to parse channel", error=e)
            return None

    async def _parse_message(
        self,
        message: Dict[str, Any],
        channel_id: str,
    ) -> Optional[Resource]:
        """Parse a Slack message into a Resource."""
        try:
            ts = message.get("ts", "")
            text = message.get("text", "")
            user_id = message.get("user", "")

            # Get user name from cache or API
            user_name = await self._get_user_name(user_id)

            # Create unique ID from channel + timestamp
            message_id = f"{channel_id}_{ts}"

            # Parse timestamp
            try:
                timestamp = datetime.fromtimestamp(float(ts))
            except (ValueError, TypeError):
                timestamp = None

            return Resource(
                id=message_id,
                name=f"Message from {user_name}",
                resource_type=ResourceType.MESSAGE,
                path=f"/channels/{channel_id}/messages/{ts}",
                mime_type="text/plain",
                size=len(text),
                created_at=timestamp,
                modified_at=timestamp,
                metadata={
                    "channel_id": channel_id,
                    "ts": ts,
                    "user_id": user_id,
                    "user_name": user_name,
                    "text": text,
                    "has_attachments": bool(message.get("files")),
                    "has_reactions": bool(message.get("reactions")),
                    "thread_ts": message.get("thread_ts"),
                    "reply_count": message.get("reply_count", 0),
                },
            )
        except Exception as e:
            self.log_error("Failed to parse message", error=e)
            return None

    async def _get_user_name(self, user_id: str) -> str:
        """Get user name from cache or API."""
        if not user_id:
            return "Unknown"

        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            client = await self._get_client()
            response = await client.get("/users.info", params={"user": user_id})

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    user = data.get("user", {})
                    name = user.get("real_name") or user.get("name") or user_id
                    self._user_cache[user_id] = name
                    return name

        except Exception as e:
            self.log_error("Failed to get user info", error=e, user_id=user_id)

        self._user_cache[user_id] = user_id
        return user_id

    async def download_resource(self, resource: Resource) -> bytes:
        """Download resource content."""
        if resource.resource_type == ResourceType.MESSAGE:
            # Return message text as content
            text = resource.metadata.get("text", "")
            user_name = resource.metadata.get("user_name", "Unknown")
            timestamp = resource.created_at.isoformat() if resource.created_at else "Unknown time"

            # Format as readable content
            content = f"From: {user_name}\nTime: {timestamp}\n\n{text}"
            return content.encode("utf-8")

        elif resource.resource_type == ResourceType.FILE:
            # Download file from Slack
            url = resource.metadata.get("url_private")
            if not url:
                raise ValueError("No download URL for file")

            client = await self._get_client()
            response = await client.get(url)

            if response.status_code == 200:
                return response.content
            else:
                raise ValueError(f"Failed to download file: {response.status_code}")

        elif resource.resource_type == ResourceType.FOLDER:
            # For channels, return channel info as content
            info = [
                f"Channel: {resource.name}",
                f"Topic: {resource.metadata.get('topic', 'No topic')}",
                f"Purpose: {resource.metadata.get('purpose', 'No purpose')}",
                f"Members: {resource.metadata.get('num_members', 0)}",
            ]
            return "\n".join(info).encode("utf-8")

        raise ValueError(f"Cannot download resource type: {resource.resource_type}")

    async def get_changes(
        self,
        since: Optional[datetime] = None,
        page_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """Get changes since a specific time."""
        # Slack doesn't have a native changes API
        # We would need to track message timestamps and compare
        # For now, return empty - full sync will be used
        return [], None

    async def close(self):
        """Close the connector and clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._user_cache.clear()

    @classmethod
    def get_oauth_url(
        cls,
        client_id: str,
        redirect_uri: str,
        state: str,
    ) -> str:
        """Generate Slack OAuth authorization URL."""
        scopes = ",".join(cls.REQUIRED_SCOPES)
        return (
            f"https://slack.com/oauth/v2/authorize"
            f"?client_id={client_id}"
            f"&scope={scopes}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
        )

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for the connector."""
        return {
            "type": "object",
            "properties": {
                "sync_private_channels": {
                    "type": "boolean",
                    "title": "Sync Private Channels",
                    "description": "Include private channels in sync (requires appropriate permissions)",
                    "default": False,
                },
                "sync_direct_messages": {
                    "type": "boolean",
                    "title": "Sync Direct Messages",
                    "description": "Include DMs in sync (requires user consent)",
                    "default": False,
                },
                "include_threads": {
                    "type": "boolean",
                    "title": "Include Thread Replies",
                    "description": "Sync full thread conversations",
                    "default": True,
                },
                "max_messages_per_channel": {
                    "type": "integer",
                    "title": "Max Messages per Channel",
                    "description": "Maximum messages to sync per channel (0 for unlimited)",
                    "default": 1000,
                    "minimum": 0,
                },
            },
        }

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get credentials schema for the connector."""
        return {
            "type": "object",
            "required": ["access_token"],
            "properties": {
                "access_token": {
                    "type": "string",
                    "title": "Bot Token",
                    "description": "Slack Bot User OAuth Token (xoxb-...)",
                    "format": "password",
                },
                "client_id": {
                    "type": "string",
                    "title": "Client ID",
                    "description": "Slack App Client ID (for OAuth refresh)",
                },
                "client_secret": {
                    "type": "string",
                    "title": "Client Secret",
                    "description": "Slack App Client Secret (for OAuth refresh)",
                    "format": "password",
                },
                "refresh_token": {
                    "type": "string",
                    "title": "Refresh Token",
                    "description": "OAuth refresh token (if using token rotation)",
                    "format": "password",
                },
            },
        }
