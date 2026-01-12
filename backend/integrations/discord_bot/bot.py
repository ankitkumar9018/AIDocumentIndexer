"""
AIDocumentIndexer - Discord Bot
================================

Core Discord bot implementation using discord.py.
Handles connection, events, and command routing.
"""

import asyncio
from typing import Optional, Dict, Any, List

import structlog

logger = structlog.get_logger(__name__)


class DiscordBot:
    """
    Discord bot for AIDocumentIndexer.

    Provides document Q&A, search, and AI assistant
    functionality through Discord.

    Usage:
        bot = DiscordBot(
            token="your-bot-token",
            organization_id="org-uuid",
        )
        await bot.start()
    """

    def __init__(
        self,
        token: str,
        organization_id: Optional[str] = None,
        command_prefix: str = "!",
        allowed_channels: Optional[List[str]] = None,
        allowed_roles: Optional[List[str]] = None,
    ):
        """
        Initialize Discord bot.

        Args:
            token: Discord bot token
            organization_id: Organization context for RAG queries
            command_prefix: Prefix for text commands (default: !)
            allowed_channels: List of channel IDs where bot can respond
            allowed_roles: List of role IDs that can use the bot
        """
        self.token = token
        self.organization_id = organization_id
        self.command_prefix = command_prefix
        self.allowed_channels = set(allowed_channels or [])
        self.allowed_roles = set(allowed_roles or [])
        self._client = None
        self._handler = None

    async def start(self):
        """Start the Discord bot."""
        try:
            import discord
            from discord import app_commands

            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True

            self._client = discord.Client(intents=intents)
            self._tree = app_commands.CommandTree(self._client)

            # Import handler
            from backend.integrations.discord_bot.handlers import DiscordCommandHandler

            self._handler = DiscordCommandHandler(
                organization_id=self.organization_id,
            )

            # Register events
            @self._client.event
            async def on_ready():
                logger.info(
                    "Discord bot connected",
                    user=str(self._client.user),
                    guilds=len(self._client.guilds),
                )
                # Sync slash commands
                await self._tree.sync()
                logger.info("Slash commands synced")

            @self._client.event
            async def on_message(message: discord.Message):
                # Ignore bot's own messages
                if message.author == self._client.user:
                    return

                # Check channel restrictions
                if self.allowed_channels and str(message.channel.id) not in self.allowed_channels:
                    return

                # Check role restrictions
                if self.allowed_roles:
                    user_roles = {str(role.id) for role in getattr(message.author, "roles", [])}
                    if not user_roles.intersection(self.allowed_roles):
                        return

                # Handle mentions
                if self._client.user in message.mentions:
                    content = message.content.replace(f"<@{self._client.user.id}>", "").strip()
                    if content:
                        async with message.channel.typing():
                            response = await self._handler.handle_question(
                                question=content,
                                user_id=str(message.author.id),
                                channel_id=str(message.channel.id),
                            )
                            await message.reply(response["text"])
                    return

                # Handle prefix commands
                if message.content.startswith(self.command_prefix):
                    await self._handle_prefix_command(message)

            # Register slash commands
            @self._tree.command(name="ask", description="Ask a question about your documents")
            async def ask_command(interaction: discord.Interaction, question: str):
                await interaction.response.defer()
                response = await self._handler.handle_question(
                    question=question,
                    user_id=str(interaction.user.id),
                    channel_id=str(interaction.channel_id),
                )
                await interaction.followup.send(response["text"])

            @self._tree.command(name="search", description="Search for documents")
            async def search_command(interaction: discord.Interaction, query: str):
                await interaction.response.defer()
                response = await self._handler.handle_search(
                    query=query,
                    user_id=str(interaction.user.id),
                )
                await interaction.followup.send(response["text"])

            @self._tree.command(name="summarize", description="Summarize a document")
            async def summarize_command(interaction: discord.Interaction, document: str):
                await interaction.response.defer()
                response = await self._handler.handle_summarize(
                    doc_identifier=document,
                    user_id=str(interaction.user.id),
                )
                await interaction.followup.send(response["text"])

            @self._tree.command(name="docs", description="List recent documents")
            async def docs_command(interaction: discord.Interaction):
                await interaction.response.defer()
                response = await self._handler.handle_list_documents(
                    user_id=str(interaction.user.id),
                )
                await interaction.followup.send(response["text"])

            @self._tree.command(name="help", description="Show help information")
            async def help_command(interaction: discord.Interaction):
                help_text = self._handler.get_help_text()
                await interaction.response.send_message(help_text)

            # Start the bot
            await self._client.start(self.token)

        except ImportError:
            logger.error("discord.py not installed. Run: pip install discord.py")
            raise
        except Exception as e:
            logger.error("Failed to start Discord bot", error=str(e))
            raise

    async def _handle_prefix_command(self, message):
        """Handle prefix-based commands."""
        import discord

        content = message.content[len(self.command_prefix):].strip()
        parts = content.split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        async with message.channel.typing():
            if command == "help":
                response = {"text": self._handler.get_help_text()}
            elif command == "ask":
                response = await self._handler.handle_question(
                    question=args,
                    user_id=str(message.author.id),
                    channel_id=str(message.channel.id),
                )
            elif command == "search":
                response = await self._handler.handle_search(
                    query=args,
                    user_id=str(message.author.id),
                )
            elif command == "summarize":
                response = await self._handler.handle_summarize(
                    doc_identifier=args,
                    user_id=str(message.author.id),
                )
            elif command == "docs":
                response = await self._handler.handle_list_documents(
                    user_id=str(message.author.id),
                )
            else:
                response = {
                    "text": f"Unknown command: `{command}`. Use `{self.command_prefix}help` for available commands."
                }

            await message.reply(response["text"])

    async def stop(self):
        """Stop the Discord bot."""
        if self._client:
            await self._client.close()
            logger.info("Discord bot disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if bot is connected."""
        return self._client is not None and self._client.is_ready()


async def run_discord_bot(
    token: str,
    organization_id: Optional[str] = None,
    **kwargs,
):
    """
    Convenience function to run the Discord bot.

    Args:
        token: Discord bot token
        organization_id: Organization context
        **kwargs: Additional bot configuration
    """
    bot = DiscordBot(
        token=token,
        organization_id=organization_id,
        **kwargs,
    )
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
