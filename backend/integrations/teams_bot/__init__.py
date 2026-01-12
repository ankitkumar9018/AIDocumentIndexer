"""
AIDocumentIndexer - Microsoft Teams Bot Integration
====================================================

Bot integration for Microsoft Teams using Bot Framework SDK.

Features:
- Direct messaging with AI assistant
- Channel mentions and replies
- Adaptive cards for rich responses
- Document search and query
"""

from backend.integrations.teams_bot.bot import TeamsBot
from backend.integrations.teams_bot.handlers import TeamsActivityHandler

__all__ = ["TeamsBot", "TeamsActivityHandler"]
