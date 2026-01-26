"""
AIDocumentIndexer - YouTube Connector
======================================

Connector for syncing video transcripts from YouTube.

Supports:
- Individual videos
- Playlists
- Channels
- Auto-generated and manual captions
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs
import re

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

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


@ConnectorRegistry.register(ConnectorType.YOUTUBE)
class YouTubeConnector(BaseConnector):
    """
    Connector for YouTube video transcripts.

    Uses YouTube Data API for metadata and
    youtube-transcript-api for captions.
    """

    connector_type = ConnectorType.YOUTUBE
    display_name = "YouTube"
    description = "Sync video transcripts from YouTube"
    icon = "youtube"

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _api_key(self) -> str:
        """Get YouTube API key."""
        return self.config.credentials.get("api_key", "")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=YOUTUBE_API_BASE,
                timeout=30.0,
            )
        return self._client

    async def authenticate(self) -> bool:
        """Verify API key is valid."""
        try:
            client = await self._get_client()

            # Test API key with a simple request
            response = await client.get(
                "/videos",
                params={
                    "key": self._api_key,
                    "part": "snippet",
                    "id": "dQw4w9WgXcQ",  # Test video
                },
            )

            if response.status_code == 200:
                self._authenticated = True
                self.log_info("Authenticated with YouTube API")
                return True
            elif response.status_code == 403:
                self.log_error("YouTube API key invalid or quota exceeded")
                return False
            else:
                self.log_error("YouTube authentication failed", status=response.status_code)
                return False

        except Exception as e:
            self.log_error("YouTube authentication error", error=e)
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """API key doesn't need refresh."""
        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 50,
    ) -> tuple[List[Resource], Optional[str]]:
        """List videos from a playlist or channel."""
        client = await self._get_client()
        resources = []

        try:
            if folder_id:
                if folder_id.startswith("playlist:"):
                    # List videos in playlist
                    playlist_id = folder_id.replace("playlist:", "")
                    return await self._list_playlist_videos(
                        playlist_id, page_token, page_size, client
                    )
                elif folder_id.startswith("channel:"):
                    # List channel's uploads
                    channel_id = folder_id.replace("channel:", "")
                    return await self._list_channel_videos(
                        channel_id, page_token, page_size, client
                    )
            else:
                # List configured playlists/channels as folders
                playlists = self.config.sync_config.get("playlists", [])
                channels = self.config.sync_config.get("channels", [])

                for playlist_id in playlists:
                    resources.append(Resource(
                        id=f"playlist:{playlist_id}",
                        name=await self._get_playlist_name(playlist_id, client),
                        resource_type=ResourceType.FOLDER,
                        metadata={"type": "playlist"},
                    ))

                for channel_id in channels:
                    resources.append(Resource(
                        id=f"channel:{channel_id}",
                        name=await self._get_channel_name(channel_id, client),
                        resource_type=ResourceType.FOLDER,
                        metadata={"type": "channel"},
                    ))

                return resources, None

        except Exception as e:
            self.log_error("Failed to list YouTube resources", error=e)

        return resources, None

    async def _list_playlist_videos(
        self,
        playlist_id: str,
        page_token: Optional[str],
        page_size: int,
        client: httpx.AsyncClient,
    ) -> tuple[List[Resource], Optional[str]]:
        """List videos in a playlist."""
        resources = []

        params = {
            "key": self._api_key,
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": min(page_size, 50),
        }
        if page_token:
            params["pageToken"] = page_token

        response = await client.get("/playlistItems", params=params)

        if response.status_code == 200:
            data = response.json()

            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                content = item.get("contentDetails", {})
                video_id = content.get("videoId")

                if video_id:
                    # Parse published date
                    published_at = None
                    if snippet.get("publishedAt"):
                        published_at = datetime.fromisoformat(
                            snippet["publishedAt"].replace("Z", "+00:00")
                        )

                    resource = Resource(
                        id=video_id,
                        name=snippet.get("title", "Untitled Video"),
                        resource_type=ResourceType.VIDEO,
                        parent_id=f"playlist:{playlist_id}",
                        web_url=f"https://www.youtube.com/watch?v={video_id}",
                        created_at=published_at,
                        metadata={
                            "type": "video",
                            "channel_id": snippet.get("channelId"),
                            "channel_title": snippet.get("channelTitle"),
                            "description": snippet.get("description", "")[:500],
                            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                        },
                    )
                    resources.append(resource)

            next_token = data.get("nextPageToken")
            return resources, next_token

        return resources, None

    async def _list_channel_videos(
        self,
        channel_id: str,
        page_token: Optional[str],
        page_size: int,
        client: httpx.AsyncClient,
    ) -> tuple[List[Resource], Optional[str]]:
        """List videos from a channel's uploads."""
        # First, get the uploads playlist ID
        params = {
            "key": self._api_key,
            "part": "contentDetails",
            "id": channel_id,
        }

        response = await client.get("/channels", params=params)

        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])

            if items:
                uploads_playlist = items[0].get("contentDetails", {}).get(
                    "relatedPlaylists", {}
                ).get("uploads")

                if uploads_playlist:
                    return await self._list_playlist_videos(
                        uploads_playlist, page_token, page_size, client
                    )

        return [], None

    async def _get_playlist_name(
        self,
        playlist_id: str,
        client: httpx.AsyncClient,
    ) -> str:
        """Get playlist name."""
        try:
            response = await client.get(
                "/playlists",
                params={
                    "key": self._api_key,
                    "part": "snippet",
                    "id": playlist_id,
                },
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                if items:
                    return items[0].get("snippet", {}).get("title", playlist_id)

        except Exception as e:
            self.log_debug("Failed to get playlist name, using ID", error=str(e), playlist_id=playlist_id)

        return playlist_id

    async def _get_channel_name(
        self,
        channel_id: str,
        client: httpx.AsyncClient,
    ) -> str:
        """Get channel name."""
        try:
            response = await client.get(
                "/channels",
                params={
                    "key": self._api_key,
                    "part": "snippet",
                    "id": channel_id,
                },
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                if items:
                    return items[0].get("snippet", {}).get("title", channel_id)

        except Exception as e:
            self.log_debug("Failed to get channel name, using ID", error=str(e), channel_id=channel_id)

        return channel_id

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get video metadata."""
        client = await self._get_client()

        # Handle folder types
        if resource_id.startswith("playlist:") or resource_id.startswith("channel:"):
            return await self._get_folder_resource(resource_id, client)

        try:
            response = await client.get(
                "/videos",
                params={
                    "key": self._api_key,
                    "part": "snippet,contentDetails,statistics",
                    "id": resource_id,
                },
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])

                if items:
                    item = items[0]
                    snippet = item.get("snippet", {})
                    content = item.get("contentDetails", {})
                    stats = item.get("statistics", {})

                    # Parse duration
                    duration = content.get("duration", "")
                    duration_seconds = self._parse_duration(duration)

                    published_at = None
                    if snippet.get("publishedAt"):
                        published_at = datetime.fromisoformat(
                            snippet["publishedAt"].replace("Z", "+00:00")
                        )

                    return Resource(
                        id=resource_id,
                        name=snippet.get("title", "Untitled Video"),
                        resource_type=ResourceType.VIDEO,
                        web_url=f"https://www.youtube.com/watch?v={resource_id}",
                        created_at=published_at,
                        metadata={
                            "type": "video",
                            "channel_id": snippet.get("channelId"),
                            "channel_title": snippet.get("channelTitle"),
                            "description": snippet.get("description", ""),
                            "duration_seconds": duration_seconds,
                            "view_count": int(stats.get("viewCount", 0)),
                            "like_count": int(stats.get("likeCount", 0)),
                            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                            "tags": snippet.get("tags", []),
                        },
                    )

        except Exception as e:
            self.log_error("Failed to get video", error=e, video_id=resource_id)

        return None

    async def _get_folder_resource(
        self,
        resource_id: str,
        client: httpx.AsyncClient,
    ) -> Optional[Resource]:
        """Get playlist or channel as folder."""
        if resource_id.startswith("playlist:"):
            playlist_id = resource_id.replace("playlist:", "")
            name = await self._get_playlist_name(playlist_id, client)
            return Resource(
                id=resource_id,
                name=name,
                resource_type=ResourceType.FOLDER,
                web_url=f"https://www.youtube.com/playlist?list={playlist_id}",
                metadata={"type": "playlist"},
            )
        elif resource_id.startswith("channel:"):
            channel_id = resource_id.replace("channel:", "")
            name = await self._get_channel_name(channel_id, client)
            return Resource(
                id=resource_id,
                name=name,
                resource_type=ResourceType.FOLDER,
                web_url=f"https://www.youtube.com/channel/{channel_id}",
                metadata={"type": "channel"},
            )
        return None

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds."""
        # Format: PT#H#M#S
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        return 0

    async def download_resource(self, resource_id: str) -> Optional[bytes]:
        """Download video transcript."""
        try:
            # Try to use youtube-transcript-api
            transcript = await self._get_transcript(resource_id)

            if transcript:
                # Format transcript as text
                lines = []
                for entry in transcript:
                    text = entry.get("text", "").strip()
                    if text:
                        lines.append(text)

                return "\n".join(lines).encode("utf-8")

        except Exception as e:
            self.log_error("Failed to get transcript", error=e, video_id=resource_id)

        return None

    async def _get_transcript(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get transcript for a video."""
        try:
            # Try youtube-transcript-api if available
            from youtube_transcript_api import YouTubeTranscriptApi

            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get manual transcript first, then auto-generated
            transcript = None

            try:
                transcript = transcript_list.find_manually_created_transcript(
                    self.config.sync_config.get("languages", ["en"])
                )
            except Exception as e:
                self.log_debug("No manual transcript, trying auto-generated", error=str(e), video_id=video_id)
                try:
                    transcript = transcript_list.find_generated_transcript(
                        self.config.sync_config.get("languages", ["en"])
                    )
                except Exception as e2:
                    self.log_debug("No auto-generated transcript available", error=str(e2), video_id=video_id)

            if transcript:
                return transcript.fetch()

        except ImportError:
            self.log_warning("youtube-transcript-api not installed")
        except Exception as e:
            self.log_debug("Transcript not available", error=str(e), video_id=video_id)

        return None

    async def get_changes(
        self,
        since_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """Get new videos since last sync."""
        # YouTube doesn't have a native changes API
        # We can check for new uploads in configured playlists/channels
        changes = []

        # For now, return empty - full sync is recommended
        return changes, None

    @staticmethod
    def parse_video_url(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
            r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def parse_playlist_url(url: str) -> Optional[str]:
        """Extract playlist ID from YouTube URL."""
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc:
            params = parse_qs(parsed.query)
            if "list" in params:
                return params["list"][0]
        return None

    @staticmethod
    def parse_channel_url(url: str) -> Optional[str]:
        """Extract channel ID from YouTube URL."""
        patterns = [
            r"youtube\.com/channel/([a-zA-Z0-9_-]+)",
            r"youtube\.com/c/([a-zA-Z0-9_-]+)",
            r"youtube\.com/@([a-zA-Z0-9_-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get schema for YouTube credentials."""
        return {
            "type": "object",
            "properties": {
                "api_key": {
                    "type": "string",
                    "description": "YouTube Data API key",
                },
            },
            "required": ["api_key"],
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get schema for YouTube connector configuration."""
        schema = super().get_config_schema()
        schema["properties"].update({
            "playlists": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Playlist IDs to sync",
            },
            "channels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Channel IDs to sync",
            },
            "videos": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Individual video IDs to sync",
            },
            "languages": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["en"],
                "description": "Preferred languages for transcripts",
            },
            "include_auto_generated": {
                "type": "boolean",
                "default": True,
                "description": "Include auto-generated captions",
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
