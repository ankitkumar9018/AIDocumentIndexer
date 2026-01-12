"""
AIDocumentIndexer - Telegram Bot Integration
=============================================

Telegram bot for document Q&A, search, and AI assistant
functionality through Telegram messaging.
"""

from backend.integrations.telegram_bot.bot import TelegramBot
from backend.integrations.telegram_bot.handlers import TelegramCommandHandler

__all__ = ["TelegramBot", "TelegramCommandHandler"]
