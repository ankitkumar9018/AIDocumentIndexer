"""
AIDocumentIndexer - Multi-Channel Gateway
==========================================

Central gateway for routing messages across multiple communication platforms.
Unified interface for Slack, Discord, Teams, WhatsApp, Telegram, and more.

Based on OpenClaw architecture patterns for multi-channel AI assistants.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable, Union
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio
import json

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


class ChannelType(str, Enum):
    """Supported communication channels."""
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    WEB = "web"
    API = "api"
    EMAIL = "email"


class MessageType(str, Enum):
    """Types of messages."""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    INTERACTIVE = "interactive"
    SYSTEM = "system"


@dataclass
class ChannelUser:
    """User across channels."""
    channel_id: str  # User ID in the specific channel
    channel_type: ChannelType
    internal_user_id: Optional[str] = None  # Mapped internal user
    display_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncomingMessage:
    """Message received from any channel."""
    message_id: str
    channel_type: ChannelType
    channel_id: str  # Channel/workspace/group ID
    user: ChannelUser
    content: str
    message_type: MessageType = MessageType.TEXT
    timestamp: datetime = field(default_factory=datetime.utcnow)
    thread_id: Optional[str] = None
    reply_to: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "channel_type": self.channel_type.value,
            "channel_id": self.channel_id,
            "user": {
                "channel_id": self.user.channel_id,
                "display_name": self.user.display_name,
            },
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "thread_id": self.thread_id,
            "attachments": self.attachments,
        }


@dataclass
class OutgoingMessage:
    """Message to send to a channel."""
    channel_type: ChannelType
    channel_id: str
    content: str
    message_type: MessageType = MessageType.TEXT
    thread_id: Optional[str] = None
    reply_to: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    interactive_elements: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelConfig:
    """Configuration for a channel."""
    channel_type: ChannelType
    enabled: bool = True
    credentials: Dict[str, str] = field(default_factory=dict)
    webhook_url: Optional[str] = None
    rate_limit_per_minute: int = 60
    max_message_length: int = 4000
    supports_threads: bool = True
    supports_reactions: bool = True
    supports_files: bool = True


@dataclass
class DeliveryResult:
    """Result of message delivery."""
    success: bool
    channel_type: ChannelType
    message_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Channel Interface
# =============================================================================

class ChannelHandler(ABC):
    """Abstract base class for channel handlers."""

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the channel type this handler manages."""
        pass

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> DeliveryResult:
        """Send a message to the channel."""
        pass

    @abstractmethod
    async def validate_config(self, config: ChannelConfig) -> bool:
        """Validate channel configuration."""
        pass

    async def format_message(self, content: str, **kwargs) -> str:
        """Format message for the specific channel."""
        return content

    async def handle_webhook(self, payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Handle incoming webhook from the channel."""
        return None


# =============================================================================
# Channel Implementations
# =============================================================================

class SlackChannelHandler(ChannelHandler):
    """Handler for Slack messages."""

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.SLACK

    async def send_message(self, message: OutgoingMessage) -> DeliveryResult:
        """Send message to Slack."""
        try:
            from slack_sdk.web.async_client import AsyncWebClient

            config = self._get_config()
            if not config or not config.credentials.get("bot_token"):
                return DeliveryResult(
                    success=False,
                    channel_type=self.channel_type,
                    error="Slack bot token not configured",
                )

            client = AsyncWebClient(token=config.credentials["bot_token"])

            # Format for Slack
            blocks = self._format_slack_blocks(message)

            response = await client.chat_postMessage(
                channel=message.channel_id,
                text=message.content,
                blocks=blocks if blocks else None,
                thread_ts=message.thread_id,
            )

            return DeliveryResult(
                success=True,
                channel_type=self.channel_type,
                message_id=response.get("ts"),
                metadata={"channel": response.get("channel")},
            )

        except Exception as e:
            logger.error("Slack send failed", error=str(e))
            return DeliveryResult(
                success=False,
                channel_type=self.channel_type,
                error=str(e),
            )

    async def validate_config(self, config: ChannelConfig) -> bool:
        """Validate Slack configuration."""
        return bool(config.credentials.get("bot_token"))

    def _get_config(self) -> Optional[ChannelConfig]:
        """Get Slack config from settings."""
        # Would fetch from database/settings in production
        return None

    def _format_slack_blocks(self, message: OutgoingMessage) -> Optional[List[Dict]]:
        """Format message as Slack blocks."""
        if message.interactive_elements:
            return message.interactive_elements.get("blocks")
        return None


class DiscordChannelHandler(ChannelHandler):
    """Handler for Discord messages."""

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.DISCORD

    async def send_message(self, message: OutgoingMessage) -> DeliveryResult:
        """Send message to Discord."""
        try:
            import aiohttp

            config = self._get_config()
            if not config or not config.webhook_url:
                return DeliveryResult(
                    success=False,
                    channel_type=self.channel_type,
                    error="Discord webhook not configured",
                )

            # Format for Discord
            payload = {
                "content": message.content[:2000],  # Discord limit
            }

            if message.interactive_elements:
                payload["embeds"] = message.interactive_elements.get("embeds", [])

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.webhook_url,
                    json=payload,
                ) as response:
                    if response.status in [200, 204]:
                        return DeliveryResult(
                            success=True,
                            channel_type=self.channel_type,
                        )
                    else:
                        text = await response.text()
                        return DeliveryResult(
                            success=False,
                            channel_type=self.channel_type,
                            error=f"Discord error: {response.status} - {text}",
                        )

        except Exception as e:
            logger.error("Discord send failed", error=str(e))
            return DeliveryResult(
                success=False,
                channel_type=self.channel_type,
                error=str(e),
            )

    async def validate_config(self, config: ChannelConfig) -> bool:
        """Validate Discord configuration."""
        return bool(config.webhook_url)

    def _get_config(self) -> Optional[ChannelConfig]:
        """Get Discord config from settings."""
        return None


class TeamsChannelHandler(ChannelHandler):
    """Handler for Microsoft Teams messages."""

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.TEAMS

    async def send_message(self, message: OutgoingMessage) -> DeliveryResult:
        """Send message to Teams."""
        try:
            import aiohttp

            config = self._get_config()
            if not config or not config.webhook_url:
                return DeliveryResult(
                    success=False,
                    channel_type=self.channel_type,
                    error="Teams webhook not configured",
                )

            # Format as Adaptive Card
            payload = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": [
                                {
                                    "type": "TextBlock",
                                    "text": message.content[:28000],  # Teams limit
                                    "wrap": True,
                                }
                            ],
                        },
                    }
                ],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.webhook_url,
                    json=payload,
                ) as response:
                    if response.status == 200:
                        return DeliveryResult(
                            success=True,
                            channel_type=self.channel_type,
                        )
                    else:
                        text = await response.text()
                        return DeliveryResult(
                            success=False,
                            channel_type=self.channel_type,
                            error=f"Teams error: {response.status} - {text}",
                        )

        except Exception as e:
            logger.error("Teams send failed", error=str(e))
            return DeliveryResult(
                success=False,
                channel_type=self.channel_type,
                error=str(e),
            )

    async def validate_config(self, config: ChannelConfig) -> bool:
        """Validate Teams configuration."""
        return bool(config.webhook_url)

    def _get_config(self) -> Optional[ChannelConfig]:
        """Get Teams config from settings."""
        return None


class TelegramChannelHandler(ChannelHandler):
    """Handler for Telegram messages."""

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.TELEGRAM

    async def send_message(self, message: OutgoingMessage) -> DeliveryResult:
        """Send message to Telegram."""
        try:
            import aiohttp

            config = self._get_config()
            if not config or not config.credentials.get("bot_token"):
                return DeliveryResult(
                    success=False,
                    channel_type=self.channel_type,
                    error="Telegram bot token not configured",
                )

            bot_token = config.credentials["bot_token"]
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

            payload = {
                "chat_id": message.channel_id,
                "text": message.content[:4096],  # Telegram limit
                "parse_mode": "Markdown",
            }

            if message.reply_to:
                payload["reply_to_message_id"] = message.reply_to

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    data = await response.json()
                    if data.get("ok"):
                        return DeliveryResult(
                            success=True,
                            channel_type=self.channel_type,
                            message_id=str(data["result"]["message_id"]),
                        )
                    else:
                        return DeliveryResult(
                            success=False,
                            channel_type=self.channel_type,
                            error=data.get("description", "Unknown error"),
                        )

        except Exception as e:
            logger.error("Telegram send failed", error=str(e))
            return DeliveryResult(
                success=False,
                channel_type=self.channel_type,
                error=str(e),
            )

    async def validate_config(self, config: ChannelConfig) -> bool:
        """Validate Telegram configuration."""
        return bool(config.credentials.get("bot_token"))

    def _get_config(self) -> Optional[ChannelConfig]:
        """Get Telegram config from settings."""
        return None


class WebChannelHandler(ChannelHandler):
    """Handler for web-based messages (WebSocket/SSE)."""

    def __init__(self):
        self._connections: Dict[str, asyncio.Queue] = {}

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WEB

    async def send_message(self, message: OutgoingMessage) -> DeliveryResult:
        """Send message to web client via WebSocket queue."""
        if message.channel_id not in self._connections:
            return DeliveryResult(
                success=False,
                channel_type=self.channel_type,
                error="Client not connected",
            )

        try:
            queue = self._connections[message.channel_id]
            await queue.put({
                "type": "message",
                "content": message.content,
                "timestamp": datetime.utcnow().isoformat(),
                "attachments": message.attachments,
            })

            return DeliveryResult(
                success=True,
                channel_type=self.channel_type,
            )

        except Exception as e:
            return DeliveryResult(
                success=False,
                channel_type=self.channel_type,
                error=str(e),
            )

    async def validate_config(self, config: ChannelConfig) -> bool:
        """Web channel doesn't need validation."""
        return True

    def register_connection(self, client_id: str) -> asyncio.Queue:
        """Register a new web client connection."""
        queue = asyncio.Queue(maxsize=100)
        self._connections[client_id] = queue
        return queue

    def unregister_connection(self, client_id: str):
        """Unregister a web client connection."""
        self._connections.pop(client_id, None)


# =============================================================================
# Multi-Channel Gateway
# =============================================================================

class ChannelGateway:
    """
    Central gateway for multi-channel communication.

    Routes messages to/from various platforms with:
    - Unified message format
    - Channel-specific formatting
    - Rate limiting
    - Delivery tracking
    """

    def __init__(self):
        """Initialize the gateway with channel handlers."""
        self._handlers: Dict[ChannelType, ChannelHandler] = {}
        self._configs: Dict[ChannelType, ChannelConfig] = {}
        self._message_handlers: List[Callable[[IncomingMessage], Awaitable[Optional[str]]]] = []

        # Register default handlers
        self._register_default_handlers()

        logger.info("ChannelGateway initialized")

    def _register_default_handlers(self):
        """Register built-in channel handlers."""
        self._handlers[ChannelType.SLACK] = SlackChannelHandler()
        self._handlers[ChannelType.DISCORD] = DiscordChannelHandler()
        self._handlers[ChannelType.TEAMS] = TeamsChannelHandler()
        self._handlers[ChannelType.TELEGRAM] = TelegramChannelHandler()
        self._handlers[ChannelType.WEB] = WebChannelHandler()

    def register_handler(self, handler: ChannelHandler):
        """Register a custom channel handler."""
        self._handlers[handler.channel_type] = handler
        logger.info(f"Registered handler for {handler.channel_type.value}")

    def configure_channel(self, config: ChannelConfig):
        """Configure a channel."""
        self._configs[config.channel_type] = config
        logger.info(
            f"Configured channel {config.channel_type.value}",
            enabled=config.enabled,
        )

    def register_message_handler(
        self,
        handler: Callable[[IncomingMessage], Awaitable[Optional[str]]],
    ):
        """
        Register a handler for incoming messages.

        The handler should process the message and return an optional response.
        """
        self._message_handlers.append(handler)

    async def send(
        self,
        channel_type: ChannelType,
        channel_id: str,
        content: str,
        **kwargs,
    ) -> DeliveryResult:
        """
        Send a message to a channel.

        Args:
            channel_type: Target channel
            channel_id: Target channel/user ID
            content: Message content
            **kwargs: Additional message options

        Returns:
            DeliveryResult with success status
        """
        handler = self._handlers.get(channel_type)
        if not handler:
            return DeliveryResult(
                success=False,
                channel_type=channel_type,
                error=f"No handler for channel: {channel_type.value}",
            )

        config = self._configs.get(channel_type)
        if config and not config.enabled:
            return DeliveryResult(
                success=False,
                channel_type=channel_type,
                error=f"Channel {channel_type.value} is disabled",
            )

        # Truncate if needed
        max_length = config.max_message_length if config else 4000
        if len(content) > max_length:
            content = content[:max_length - 3] + "..."

        message = OutgoingMessage(
            channel_type=channel_type,
            channel_id=channel_id,
            content=content,
            thread_id=kwargs.get("thread_id"),
            reply_to=kwargs.get("reply_to"),
            attachments=kwargs.get("attachments", []),
            interactive_elements=kwargs.get("interactive_elements"),
        )

        return await handler.send_message(message)

    async def broadcast(
        self,
        content: str,
        channels: Optional[List[ChannelType]] = None,
        **kwargs,
    ) -> Dict[ChannelType, DeliveryResult]:
        """
        Broadcast a message to multiple channels.

        Args:
            content: Message content
            channels: List of channels (all enabled if None)
            **kwargs: Channel IDs and options

        Returns:
            Dictionary of results per channel
        """
        results = {}

        if channels is None:
            channels = [
                ct for ct, config in self._configs.items()
                if config.enabled
            ]

        for channel_type in channels:
            channel_id = kwargs.get(f"{channel_type.value}_channel_id")
            if channel_id:
                results[channel_type] = await self.send(
                    channel_type=channel_type,
                    channel_id=channel_id,
                    content=content,
                    **kwargs,
                )

        return results

    async def receive(self, message: IncomingMessage) -> Optional[str]:
        """
        Process an incoming message.

        Routes through all registered message handlers.

        Args:
            message: Incoming message

        Returns:
            Optional response to send back
        """
        logger.info(
            "Received message",
            channel=message.channel_type.value,
            user=message.user.display_name,
        )

        for handler in self._message_handlers:
            try:
                response = await handler(message)
                if response:
                    # Send response back to the same channel
                    await self.send(
                        channel_type=message.channel_type,
                        channel_id=message.channel_id,
                        content=response,
                        thread_id=message.thread_id or message.message_id,
                    )
                    return response
            except Exception as e:
                logger.error(
                    "Message handler failed",
                    handler=handler.__name__,
                    error=str(e),
                )

        return None

    def get_enabled_channels(self) -> List[ChannelType]:
        """Get list of enabled channels."""
        return [
            ct for ct, config in self._configs.items()
            if config.enabled
        ]

    def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all channels."""
        status = {}
        for channel_type in ChannelType:
            config = self._configs.get(channel_type)
            handler = self._handlers.get(channel_type)
            status[channel_type.value] = {
                "configured": config is not None,
                "enabled": config.enabled if config else False,
                "handler_registered": handler is not None,
            }
        return status


# =============================================================================
# RAG Integration Handler
# =============================================================================

class RAGMessageHandler:
    """
    Message handler that integrates with the RAG system.

    Processes incoming messages through the RAG pipeline and returns answers.
    """

    def __init__(self):
        self._rag_service = None

    async def _get_rag_service(self):
        """Lazy load RAG service."""
        if self._rag_service is None:
            from backend.services.rag import get_rag_service
            self._rag_service = get_rag_service()
        return self._rag_service

    async def handle_message(self, message: IncomingMessage) -> Optional[str]:
        """
        Process message through RAG.

        Args:
            message: Incoming message

        Returns:
            RAG-generated response
        """
        # Skip non-text messages
        if message.message_type != MessageType.TEXT:
            return None

        # Skip empty or command messages
        content = message.content.strip()
        if not content or content.startswith("/"):
            return None

        try:
            rag_service = await self._get_rag_service()

            # Query RAG
            response = await rag_service.query(
                question=content,
                user_id=message.user.internal_user_id or message.user.channel_id,
            )

            return response.content

        except Exception as e:
            logger.error("RAG query failed", error=str(e))
            return "I'm sorry, I encountered an error processing your request."


# =============================================================================
# Singleton
# =============================================================================

_channel_gateway: Optional[ChannelGateway] = None


def get_channel_gateway() -> ChannelGateway:
    """Get or create the channel gateway singleton."""
    global _channel_gateway
    if _channel_gateway is None:
        _channel_gateway = ChannelGateway()

        # Register default RAG handler
        rag_handler = RAGMessageHandler()
        _channel_gateway.register_message_handler(rag_handler.handle_message)

    return _channel_gateway
