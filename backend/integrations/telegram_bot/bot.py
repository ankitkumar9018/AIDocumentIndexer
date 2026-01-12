"""
AIDocumentIndexer - Telegram Bot
================================

Core Telegram bot implementation using python-telegram-bot.
Handles connection, events, and command routing.
"""

import asyncio
from typing import Optional, Dict, Any, List, Set

import structlog

logger = structlog.get_logger(__name__)


class TelegramBot:
    """
    Telegram bot for AIDocumentIndexer.

    Provides document Q&A, search, and AI assistant
    functionality through Telegram.

    Usage:
        bot = TelegramBot(
            token="your-bot-token",
            organization_id="org-uuid",
        )
        await bot.start()
    """

    def __init__(
        self,
        token: str,
        organization_id: Optional[str] = None,
        allowed_users: Optional[List[int]] = None,
        allowed_groups: Optional[List[int]] = None,
        admin_users: Optional[List[int]] = None,
    ):
        """
        Initialize Telegram bot.

        Args:
            token: Telegram bot token from BotFather
            organization_id: Organization context for RAG queries
            allowed_users: List of user IDs allowed to use the bot
            allowed_groups: List of group/chat IDs where bot can respond
            admin_users: List of admin user IDs with elevated permissions
        """
        self.token = token
        self.organization_id = organization_id
        self.allowed_users: Set[int] = set(allowed_users or [])
        self.allowed_groups: Set[int] = set(allowed_groups or [])
        self.admin_users: Set[int] = set(admin_users or [])
        self._application = None
        self._handler = None
        self._running = False

    def _is_authorized(self, user_id: int, chat_id: int) -> bool:
        """Check if user is authorized to use the bot."""
        # If no restrictions, allow everyone
        if not self.allowed_users and not self.allowed_groups:
            return True

        # Check user whitelist
        if self.allowed_users and user_id in self.allowed_users:
            return True

        # Check group whitelist (for group chats)
        if self.allowed_groups and chat_id in self.allowed_groups:
            return True

        # Admins always have access
        if user_id in self.admin_users:
            return True

        return False

    async def start(self):
        """Start the Telegram bot."""
        try:
            from telegram import Update, BotCommand
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
                ContextTypes,
            )

            # Build application
            self._application = Application.builder().token(self.token).build()

            # Import handler
            from backend.integrations.telegram_bot.handlers import TelegramCommandHandler

            self._handler = TelegramCommandHandler(
                organization_id=self.organization_id,
            )

            # Command handlers
            async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle /start command."""
                user = update.effective_user
                if not self._is_authorized(user.id, update.effective_chat.id):
                    await update.message.reply_text(
                        "Sorry, you're not authorized to use this bot."
                    )
                    return

                welcome_text = (
                    f"👋 Hello {user.first_name}!\n\n"
                    "I'm your AI Document Assistant. I can help you:\n\n"
                    "📚 Answer questions about your documents\n"
                    "🔍 Search through your document library\n"
                    "📄 Summarize documents\n"
                    "📋 List your recent documents\n\n"
                    "Use /help to see all available commands."
                )
                await update.message.reply_text(welcome_text)

            async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle /help command."""
                if not self._is_authorized(update.effective_user.id, update.effective_chat.id):
                    return
                help_text = self._handler.get_help_text()
                await update.message.reply_text(help_text, parse_mode="Markdown")

            async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle /ask command."""
                user = update.effective_user
                if not self._is_authorized(user.id, update.effective_chat.id):
                    return

                question = " ".join(context.args) if context.args else ""

                # Send typing indicator
                await update.message.chat.send_action("typing")

                response = await self._handler.handle_question(
                    question=question,
                    user_id=str(user.id),
                    chat_id=str(update.effective_chat.id),
                )
                await update.message.reply_text(
                    response["text"],
                    parse_mode="Markdown",
                )

            async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle /search command."""
                user = update.effective_user
                if not self._is_authorized(user.id, update.effective_chat.id):
                    return

                query = " ".join(context.args) if context.args else ""

                await update.message.chat.send_action("typing")

                response = await self._handler.handle_search(
                    query=query,
                    user_id=str(user.id),
                )
                await update.message.reply_text(
                    response["text"],
                    parse_mode="Markdown",
                )

            async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle /summarize command."""
                user = update.effective_user
                if not self._is_authorized(user.id, update.effective_chat.id):
                    return

                doc_identifier = " ".join(context.args) if context.args else ""

                await update.message.chat.send_action("typing")

                response = await self._handler.handle_summarize(
                    doc_identifier=doc_identifier,
                    user_id=str(user.id),
                )
                await update.message.reply_text(
                    response["text"],
                    parse_mode="Markdown",
                )

            async def docs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle /docs command."""
                user = update.effective_user
                if not self._is_authorized(user.id, update.effective_chat.id):
                    return

                await update.message.chat.send_action("typing")

                response = await self._handler.handle_list_documents(
                    user_id=str(user.id),
                )
                await update.message.reply_text(
                    response["text"],
                    parse_mode="Markdown",
                )

            async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle regular messages as questions."""
                user = update.effective_user
                if not self._is_authorized(user.id, update.effective_chat.id):
                    return

                # In private chats, treat any message as a question
                # In groups, only respond when mentioned or replied to
                chat_type = update.effective_chat.type
                message = update.message

                should_respond = False

                if chat_type == "private":
                    should_respond = True
                elif message.reply_to_message and message.reply_to_message.from_user:
                    # Respond to replies to bot's messages
                    bot_user = await context.bot.get_me()
                    if message.reply_to_message.from_user.id == bot_user.id:
                        should_respond = True
                elif message.text:
                    # Check for @mention in groups
                    bot_user = await context.bot.get_me()
                    if f"@{bot_user.username}" in message.text:
                        should_respond = True

                if not should_respond:
                    return

                question = message.text or ""
                # Remove bot mention if present
                bot_user = await context.bot.get_me()
                if bot_user.username:
                    question = question.replace(f"@{bot_user.username}", "").strip()

                if not question:
                    return

                await message.chat.send_action("typing")

                response = await self._handler.handle_question(
                    question=question,
                    user_id=str(user.id),
                    chat_id=str(update.effective_chat.id),
                )
                await message.reply_text(
                    response["text"],
                    parse_mode="Markdown",
                )

            # Register handlers
            self._application.add_handler(CommandHandler("start", start_command))
            self._application.add_handler(CommandHandler("help", help_command))
            self._application.add_handler(CommandHandler("ask", ask_command))
            self._application.add_handler(CommandHandler("search", search_command))
            self._application.add_handler(CommandHandler("summarize", summarize_command))
            self._application.add_handler(CommandHandler("docs", docs_command))
            self._application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
            )

            # Set bot commands menu
            commands = [
                BotCommand("ask", "Ask a question about your documents"),
                BotCommand("search", "Search for documents"),
                BotCommand("summarize", "Summarize a document"),
                BotCommand("docs", "List your recent documents"),
                BotCommand("help", "Show help information"),
            ]

            async def post_init(application: Application):
                await application.bot.set_my_commands(commands)
                bot_info = await application.bot.get_me()
                logger.info(
                    "Telegram bot started",
                    username=bot_info.username,
                    bot_id=bot_info.id,
                )

            self._application.post_init = post_init
            self._running = True

            # Run the bot
            await self._application.initialize()
            await self._application.start()
            await self._application.updater.start_polling(drop_pending_updates=True)

            logger.info("Telegram bot polling started")

            # Keep running until stopped
            while self._running:
                await asyncio.sleep(1)

        except ImportError:
            logger.error(
                "python-telegram-bot not installed. Run: pip install python-telegram-bot"
            )
            raise
        except Exception as e:
            logger.error("Failed to start Telegram bot", error=str(e))
            raise

    async def stop(self):
        """Stop the Telegram bot."""
        self._running = False
        if self._application:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
            logger.info("Telegram bot stopped")

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running


async def run_telegram_bot(
    token: str,
    organization_id: Optional[str] = None,
    **kwargs,
):
    """
    Convenience function to run the Telegram bot.

    Args:
        token: Telegram bot token
        organization_id: Organization context
        **kwargs: Additional bot configuration
    """
    bot = TelegramBot(
        token=token,
        organization_id=organization_id,
        **kwargs,
    )
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
