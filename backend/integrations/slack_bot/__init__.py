"""
AIDocumentIndexer - Slack Bot Integration
==========================================

Provides Slack bot functionality for accessing the AI assistant
directly from Slack workspaces.

Features:
- /ask command for document queries
- /search command for document search
- /summarize command for thread/channel summaries
- Direct messages for conversational AI
- File upload and processing
"""

from backend.integrations.slack_bot.app import SlackBotApp
from backend.integrations.slack_bot.handlers import SlackEventHandler
from backend.integrations.slack_bot.commands import SlackCommandHandler

__all__ = [
    "SlackBotApp",
    "SlackEventHandler",
    "SlackCommandHandler",
]
