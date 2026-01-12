"""
AIDocumentIndexer - Discord Bot Integration
============================================

Discord bot for accessing AI document assistant.
"""

from backend.integrations.discord_bot.bot import DiscordBot
from backend.integrations.discord_bot.handlers import DiscordCommandHandler

__all__ = ["DiscordBot", "DiscordCommandHandler"]
