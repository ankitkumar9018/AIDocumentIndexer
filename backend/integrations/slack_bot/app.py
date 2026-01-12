"""
AIDocumentIndexer - Slack Bot Application
==========================================

Main Slack Bot application using Bolt framework.
Handles OAuth, event subscriptions, and command routing.
"""

import os
import uuid
from typing import Optional, Dict, Any

import structlog
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_sdk.web.async_client import AsyncWebClient

from backend.core.config import settings
from backend.integrations.slack_bot.handlers import SlackEventHandler
from backend.integrations.slack_bot.commands import SlackCommandHandler

logger = structlog.get_logger(__name__)


class SlackBotApp:
    """
    Main Slack Bot application.

    Manages the Bolt app lifecycle and provides access to handlers.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        signing_secret: Optional[str] = None,
        app_token: Optional[str] = None,
    ):
        """
        Initialize the Slack Bot.

        Args:
            bot_token: Slack bot token (xoxb-...)
            signing_secret: Slack signing secret for verification
            app_token: Slack app-level token for socket mode (xapp-...)
        """
        self.bot_token = bot_token or getattr(settings, "SLACK_BOT_TOKEN", None)
        self.signing_secret = signing_secret or getattr(settings, "SLACK_SIGNING_SECRET", None)
        self.app_token = app_token or getattr(settings, "SLACK_APP_TOKEN", None)

        if not self.bot_token or not self.signing_secret:
            logger.warning("Slack credentials not configured, bot will not be available")
            self._app = None
            return

        # Initialize Bolt app
        self._app = AsyncApp(
            token=self.bot_token,
            signing_secret=self.signing_secret,
        )

        # Initialize handlers
        self.event_handler = SlackEventHandler(self._app)
        self.command_handler = SlackCommandHandler(self._app)

        # Register all handlers
        self._register_handlers()

        logger.info("Slack Bot initialized")

    @property
    def app(self) -> Optional[AsyncApp]:
        """Get the Bolt app instance."""
        return self._app

    @property
    def handler(self) -> Optional[AsyncSlackRequestHandler]:
        """Get FastAPI request handler for HTTP mode."""
        if self._app:
            return AsyncSlackRequestHandler(self._app)
        return None

    def _register_handlers(self):
        """Register all event and command handlers."""
        if not self._app:
            return

        # Register slash commands
        self._app.command("/ask")(self.command_handler.handle_ask_command)
        self._app.command("/search")(self.command_handler.handle_search_command)
        self._app.command("/summarize")(self.command_handler.handle_summarize_command)
        self._app.command("/agent")(self.command_handler.handle_agent_command)

        # Register message events
        self._app.event("message")(self.event_handler.handle_message)
        self._app.event("app_mention")(self.event_handler.handle_app_mention)

        # Register file events
        self._app.event("file_shared")(self.event_handler.handle_file_shared)

        # Register app home
        self._app.event("app_home_opened")(self.event_handler.handle_app_home_opened)

        # Register shortcuts
        self._app.shortcut("summarize_thread")(self.command_handler.handle_summarize_shortcut)

        logger.info("Slack handlers registered")

    async def get_bot_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the bot user."""
        if not self._app:
            return None

        try:
            client = AsyncWebClient(token=self.bot_token)
            response = await client.auth_test()
            return {
                "bot_id": response.get("bot_id"),
                "user_id": response.get("user_id"),
                "team_id": response.get("team_id"),
                "team": response.get("team"),
                "url": response.get("url"),
            }
        except Exception as e:
            logger.error("Failed to get bot info", error=str(e))
            return None

    async def send_message(
        self,
        channel: str,
        text: str,
        blocks: Optional[list] = None,
        thread_ts: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message to a Slack channel.

        Args:
            channel: Channel ID or name
            text: Message text (fallback for notifications)
            blocks: Block Kit blocks for rich formatting
            thread_ts: Thread timestamp for replies

        Returns:
            Message response or None on error
        """
        if not self._app:
            return None

        try:
            client = AsyncWebClient(token=self.bot_token)
            response = await client.chat_postMessage(
                channel=channel,
                text=text,
                blocks=blocks,
                thread_ts=thread_ts,
            )
            return response.data
        except Exception as e:
            logger.error("Failed to send message", error=str(e), channel=channel)
            return None

    async def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing message."""
        if not self._app:
            return None

        try:
            client = AsyncWebClient(token=self.bot_token)
            response = await client.chat_update(
                channel=channel,
                ts=ts,
                text=text,
                blocks=blocks,
            )
            return response.data
        except Exception as e:
            logger.error("Failed to update message", error=str(e))
            return None

    async def add_reaction(
        self,
        channel: str,
        timestamp: str,
        name: str,
    ) -> bool:
        """Add a reaction to a message."""
        if not self._app:
            return False

        try:
            client = AsyncWebClient(token=self.bot_token)
            await client.reactions_add(
                channel=channel,
                timestamp=timestamp,
                name=name,
            )
            return True
        except Exception as e:
            logger.warning("Failed to add reaction", error=str(e))
            return False


# Global singleton instance
_slack_bot: Optional[SlackBotApp] = None


def get_slack_bot() -> Optional[SlackBotApp]:
    """Get or create the global Slack bot instance."""
    global _slack_bot
    if _slack_bot is None:
        _slack_bot = SlackBotApp()
    return _slack_bot
