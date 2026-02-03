"""
AIDocumentIndexer - Messaging Bot Services
===========================================

Integration with messaging platforms for RAG-powered chatbots.

Supported Platforms:
- Telegram Bot API
- WhatsApp Business API (via Twilio or direct)
- Discord (via existing bot integration)

Features:
- Receive messages from messaging apps
- Query RAG system with user messages
- Send responses back to users
- Support for media attachments
- Conversation context management
"""

import asyncio
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4

import aiohttp
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums & Types
# =============================================================================

class MessagingPlatform(str, Enum):
    """Supported messaging platforms."""
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    DISCORD = "discord"
    SLACK = "slack"


class MessageType(str, Enum):
    """Types of messages."""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    COMMAND = "command"


# =============================================================================
# Data Models
# =============================================================================

class IncomingMessage(BaseModel):
    """Incoming message from a messaging platform."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: MessagingPlatform
    chat_id: str
    user_id: str
    username: Optional[str] = None
    message_type: MessageType = MessageType.TEXT
    text: Optional[str] = None
    media_url: Optional[str] = None
    media_type: Optional[str] = None
    reply_to_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class OutgoingMessage(BaseModel):
    """Outgoing message to a messaging platform."""
    chat_id: str
    text: str
    parse_mode: Optional[str] = "Markdown"  # Markdown, HTML, None
    reply_to_id: Optional[str] = None
    media_url: Optional[str] = None
    media_type: Optional[str] = None
    buttons: Optional[List[Dict[str, str]]] = None


class BotConfig(BaseModel):
    """Bot configuration."""
    platform: MessagingPlatform
    bot_token: str
    webhook_secret: Optional[str] = None
    api_base_url: Optional[str] = None
    enabled: bool = True
    # RAG settings
    default_collection: Optional[str] = None
    system_prompt: Optional[str] = None
    max_response_length: int = 4000


class ConversationContext(BaseModel):
    """Conversation context for multi-turn chat."""
    chat_id: str
    platform: MessagingPlatform
    messages: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Telegram Bot Service
# =============================================================================

class TelegramBot:
    """
    Telegram Bot integration.

    Handles incoming messages from Telegram and sends responses
    using the Telegram Bot API.
    """

    def __init__(self, config: BotConfig):
        """Initialize Telegram bot."""
        self.config = config
        self.api_base = "https://api.telegram.org"
        self._http_client: Optional[aiohttp.ClientSession] = None

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.closed:
            self._http_client = aiohttp.ClientSession()
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.close()

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature (if using webhook secret)."""
        if not self.config.webhook_secret:
            return True

        expected = hmac.new(
            self.config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def parse_update(self, update: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse Telegram update into IncomingMessage."""
        message = update.get("message") or update.get("edited_message")
        if not message:
            return None

        chat = message.get("chat", {})
        user = message.get("from", {})

        # Determine message type
        msg_type = MessageType.TEXT
        text = message.get("text")
        media_url = None
        media_type = None

        if message.get("photo"):
            msg_type = MessageType.IMAGE
            # Get largest photo
            photos = message["photo"]
            if photos:
                photo = max(photos, key=lambda p: p.get("file_size", 0))
                media_url = photo.get("file_id")  # Need to fetch actual URL
                media_type = "image/jpeg"
        elif message.get("document"):
            msg_type = MessageType.DOCUMENT
            doc = message["document"]
            media_url = doc.get("file_id")
            media_type = doc.get("mime_type")
        elif message.get("voice"):
            msg_type = MessageType.AUDIO
            voice = message["voice"]
            media_url = voice.get("file_id")
            media_type = voice.get("mime_type", "audio/ogg")
        elif text and text.startswith("/"):
            msg_type = MessageType.COMMAND

        return IncomingMessage(
            id=str(message.get("message_id")),
            platform=MessagingPlatform.TELEGRAM,
            chat_id=str(chat.get("id")),
            user_id=str(user.get("id")),
            username=user.get("username"),
            message_type=msg_type,
            text=text or message.get("caption"),
            media_url=media_url,
            media_type=media_type,
            reply_to_id=str(message.get("reply_to_message", {}).get("message_id"))
                if message.get("reply_to_message") else None,
            timestamp=datetime.fromtimestamp(message.get("date", 0)),
            raw_data=update,
        )

    async def send_message(self, message: OutgoingMessage) -> Dict[str, Any]:
        """Send message to Telegram."""
        client = await self._get_client()

        url = f"{self.api_base}/bot{self.config.bot_token}/sendMessage"

        payload = {
            "chat_id": message.chat_id,
            "text": message.text[:4096],  # Telegram limit
        }

        if message.parse_mode:
            payload["parse_mode"] = message.parse_mode

        if message.reply_to_id:
            payload["reply_to_message_id"] = message.reply_to_id

        if message.buttons:
            # Build inline keyboard
            keyboard = {
                "inline_keyboard": [
                    [{"text": btn["text"], "callback_data": btn.get("data", btn["text"])}]
                    for btn in message.buttons
                ]
            }
            payload["reply_markup"] = json.dumps(keyboard)

        async with client.post(url, json=payload) as response:
            result = await response.json()
            if not result.get("ok"):
                logger.error("Telegram send failed", error=result.get("description"))
            return result

    async def send_typing_action(self, chat_id: str) -> None:
        """Send typing indicator."""
        client = await self._get_client()
        url = f"{self.api_base}/bot{self.config.bot_token}/sendChatAction"

        async with client.post(url, json={"chat_id": chat_id, "action": "typing"}):
            pass

    async def get_file_url(self, file_id: str) -> Optional[str]:
        """Get download URL for a file."""
        client = await self._get_client()
        url = f"{self.api_base}/bot{self.config.bot_token}/getFile"

        async with client.get(url, params={"file_id": file_id}) as response:
            result = await response.json()
            if result.get("ok"):
                file_path = result["result"]["file_path"]
                return f"{self.api_base}/file/bot{self.config.bot_token}/{file_path}"
            return None

    async def set_webhook(self, webhook_url: str) -> bool:
        """Set webhook URL for receiving updates."""
        client = await self._get_client()
        url = f"{self.api_base}/bot{self.config.bot_token}/setWebhook"

        payload = {"url": webhook_url}
        if self.config.webhook_secret:
            payload["secret_token"] = self.config.webhook_secret

        async with client.post(url, json=payload) as response:
            result = await response.json()
            return result.get("ok", False)


# =============================================================================
# WhatsApp Bot Service (Twilio)
# =============================================================================

class WhatsAppBot:
    """
    WhatsApp Bot integration via Twilio.

    For direct WhatsApp Business API, use WhatsAppBusinessBot instead.
    """

    def __init__(self, config: BotConfig):
        """Initialize WhatsApp bot."""
        self.config = config
        # Twilio credentials from config or environment
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "")
        self.api_base = "https://api.twilio.com/2010-04-01"
        self._http_client: Optional[aiohttp.ClientSession] = None

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get or create HTTP client with auth."""
        if self._http_client is None or self._http_client.closed:
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            self._http_client = aiohttp.ClientSession(auth=auth)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.close()

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        """Verify Twilio webhook signature."""
        # Twilio uses X-Twilio-Signature header
        # Simplified verification - implement full validation in production
        return True

    def parse_webhook(self, form_data: Dict[str, str]) -> Optional[IncomingMessage]:
        """Parse Twilio webhook data into IncomingMessage."""
        # Twilio sends form-urlencoded data
        if not form_data.get("From"):
            return None

        from_number = form_data.get("From", "").replace("whatsapp:", "")
        to_number = form_data.get("To", "").replace("whatsapp:", "")

        msg_type = MessageType.TEXT
        media_url = None
        media_type = None

        # Check for media
        num_media = int(form_data.get("NumMedia", 0))
        if num_media > 0:
            media_url = form_data.get("MediaUrl0")
            media_type = form_data.get("MediaContentType0")
            if media_type and media_type.startswith("image"):
                msg_type = MessageType.IMAGE
            elif media_type and media_type.startswith("audio"):
                msg_type = MessageType.AUDIO
            else:
                msg_type = MessageType.DOCUMENT

        return IncomingMessage(
            id=form_data.get("MessageSid", str(uuid4())),
            platform=MessagingPlatform.WHATSAPP,
            chat_id=from_number,  # Use sender as chat ID for 1:1
            user_id=from_number,
            username=form_data.get("ProfileName"),
            message_type=msg_type,
            text=form_data.get("Body"),
            media_url=media_url,
            media_type=media_type,
            raw_data=form_data,
        )

    async def send_message(self, message: OutgoingMessage) -> Dict[str, Any]:
        """Send WhatsApp message via Twilio."""
        client = await self._get_client()

        url = f"{self.api_base}/Accounts/{self.account_sid}/Messages.json"

        # Format phone number for WhatsApp
        to_number = message.chat_id
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"

        from_number = self.from_number
        if not from_number.startswith("whatsapp:"):
            from_number = f"whatsapp:{from_number}"

        payload = {
            "From": from_number,
            "To": to_number,
            "Body": message.text[:1600],  # WhatsApp limit
        }

        if message.media_url:
            payload["MediaUrl"] = message.media_url

        async with client.post(url, data=payload) as response:
            result = await response.json()
            if response.status >= 400:
                logger.error("WhatsApp send failed", error=result)
            return result


# =============================================================================
# Unified Bot Handler
# =============================================================================

class MessagingBotHandler:
    """
    Unified handler for all messaging bot platforms.

    Processes incoming messages, queries RAG, and sends responses.
    """

    def __init__(
        self,
        rag_query_fn: Callable[[str, Optional[str]], Awaitable[str]],
    ):
        """
        Initialize bot handler.

        Args:
            rag_query_fn: Async function that takes (query, collection) and returns response
        """
        self.rag_query = rag_query_fn
        self._bots: Dict[MessagingPlatform, Any] = {}
        self._contexts: Dict[str, ConversationContext] = {}

    def register_bot(self, platform: MessagingPlatform, bot: Any) -> None:
        """Register a bot for a platform."""
        self._bots[platform] = bot
        logger.info("Bot registered", platform=platform)

    def get_bot(self, platform: MessagingPlatform) -> Optional[Any]:
        """Get bot for platform."""
        return self._bots.get(platform)

    def _get_context_key(self, platform: MessagingPlatform, chat_id: str) -> str:
        """Get context storage key."""
        return f"{platform.value}:{chat_id}"

    def get_context(
        self,
        platform: MessagingPlatform,
        chat_id: str,
    ) -> ConversationContext:
        """Get or create conversation context."""
        key = self._get_context_key(platform, chat_id)
        if key not in self._contexts:
            self._contexts[key] = ConversationContext(
                chat_id=chat_id,
                platform=platform,
            )
        return self._contexts[key]

    async def handle_message(
        self,
        message: IncomingMessage,
        collection: Optional[str] = None,
    ) -> Optional[str]:
        """
        Handle incoming message and generate response.

        Args:
            message: Incoming message from platform
            collection: Optional collection to query

        Returns:
            Response text or None if no response needed
        """
        # Skip non-text messages for now
        if message.message_type not in (MessageType.TEXT, MessageType.COMMAND):
            return "I can only process text messages for now. Please send text."

        text = message.text
        if not text:
            return None

        # Handle commands
        if message.message_type == MessageType.COMMAND:
            return await self._handle_command(message)

        # Get conversation context
        context = self.get_context(message.platform, message.chat_id)

        # Add user message to context
        context.messages.append({
            "role": "user",
            "content": text,
        })
        context.updated_at = datetime.utcnow()

        # Limit context length
        if len(context.messages) > 10:
            context.messages = context.messages[-10:]

        try:
            # Query RAG system
            response = await self.rag_query(text, collection)

            # Add response to context
            context.messages.append({
                "role": "assistant",
                "content": response,
            })

            return response

        except Exception as e:
            logger.error("RAG query failed", error=str(e))
            return "I'm sorry, I encountered an error processing your request. Please try again."

    async def _handle_command(self, message: IncomingMessage) -> str:
        """Handle bot commands."""
        command = message.text.split()[0].lower() if message.text else ""

        if command in ("/start", "/help"):
            return (
                "Welcome to AIDocumentIndexer Bot!\n\n"
                "I can answer questions based on your organization's documents.\n\n"
                "Commands:\n"
                "/search <query> - Search documents\n"
                "/clear - Clear conversation history\n"
                "/help - Show this message"
            )
        elif command == "/clear":
            context = self.get_context(message.platform, message.chat_id)
            context.messages.clear()
            return "Conversation history cleared."
        elif command == "/search":
            query = message.text[8:].strip() if len(message.text) > 8 else ""
            if not query:
                return "Please provide a search query. Example: /search quarterly report"
            # Treat as regular query
            message.text = query
            message.message_type = MessageType.TEXT
            return await self.handle_message(message)
        else:
            return f"Unknown command: {command}. Use /help for available commands."

    async def process_and_reply(
        self,
        message: IncomingMessage,
        collection: Optional[str] = None,
    ) -> bool:
        """
        Process message and send reply via the appropriate platform.

        Args:
            message: Incoming message
            collection: Optional collection to query

        Returns:
            True if reply was sent successfully
        """
        bot = self.get_bot(message.platform)
        if not bot:
            logger.warning("No bot registered", platform=message.platform)
            return False

        # Send typing indicator
        if hasattr(bot, "send_typing_action"):
            await bot.send_typing_action(message.chat_id)

        # Generate response
        response_text = await self.handle_message(message, collection)
        if not response_text:
            return False

        # Send response
        response = OutgoingMessage(
            chat_id=message.chat_id,
            text=response_text,
            reply_to_id=message.id,
        )

        result = await bot.send_message(response)
        return bool(result)


# =============================================================================
# Factory & Singleton
# =============================================================================

_bot_handler: Optional[MessagingBotHandler] = None


async def create_bot_handler(
    rag_query_fn: Callable[[str, Optional[str]], Awaitable[str]],
) -> MessagingBotHandler:
    """Create and configure bot handler."""
    global _bot_handler

    handler = MessagingBotHandler(rag_query_fn)

    # Register Telegram bot if configured
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if telegram_token:
        config = BotConfig(
            platform=MessagingPlatform.TELEGRAM,
            bot_token=telegram_token,
            webhook_secret=os.getenv("TELEGRAM_WEBHOOK_SECRET"),
        )
        bot = TelegramBot(config)
        handler.register_bot(MessagingPlatform.TELEGRAM, bot)

    # Register WhatsApp bot if configured
    if os.getenv("TWILIO_ACCOUNT_SID"):
        config = BotConfig(
            platform=MessagingPlatform.WHATSAPP,
            bot_token="",  # Uses Twilio credentials
        )
        bot = WhatsAppBot(config)
        handler.register_bot(MessagingPlatform.WHATSAPP, bot)

    _bot_handler = handler
    return handler


def get_bot_handler() -> Optional[MessagingBotHandler]:
    """Get existing bot handler."""
    return _bot_handler
