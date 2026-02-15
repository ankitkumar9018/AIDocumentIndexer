"""
AIDocumentIndexer - Slack Event Handlers
=========================================

Handles Slack events like messages, mentions, and file uploads.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List

import structlog
from slack_bolt.async_app import AsyncApp, AsyncAck, AsyncSay, AsyncRespond
from slack_sdk.web.async_client import AsyncWebClient

from backend.db.database import async_session_context
from backend.db.models import BotConnection, BotPlatform

logger = structlog.get_logger(__name__)


class SlackEventHandler:
    """
    Handles Slack events like messages, mentions, and file uploads.
    """

    def __init__(self, app: AsyncApp):
        self.app = app

    async def handle_message(
        self,
        message: Dict[str, Any],
        say: AsyncSay,
        client: AsyncWebClient,
    ):
        """
        Handle incoming messages.

        Only responds to DMs or when specifically configured channels.
        """
        # Ignore bot messages and message_changed events
        if message.get("bot_id") or message.get("subtype"):
            return

        channel_type = message.get("channel_type")
        text = message.get("text", "")
        user = message.get("user")
        channel = message.get("channel")
        thread_ts = message.get("thread_ts") or message.get("ts")

        # Only respond in DMs for now
        if channel_type != "im":
            return

        logger.info(
            "Received DM",
            user=user,
            channel=channel,
            text_preview=text[:50] if text else "",
        )

        # Process the message
        try:
            response = await self._process_query(text, user, channel)

            await say(
                text=response["text"],
                blocks=response.get("blocks"),
                thread_ts=thread_ts,
            )

        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            await say(
                text="Sorry, I encountered an error processing your message. Please try again.",
                thread_ts=thread_ts,
            )

    async def handle_app_mention(
        self,
        event: Dict[str, Any],
        say: AsyncSay,
        client: AsyncWebClient,
    ):
        """
        Handle @mentions of the bot.

        Extracts the query after the mention and responds.
        """
        text = event.get("text", "")
        user = event.get("user")
        channel = event.get("channel")
        thread_ts = event.get("thread_ts") or event.get("ts")

        # Remove the mention from the text
        # Text format: "<@BOTID> query"
        import re
        query = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        if not query:
            await say(
                text="Hi! How can I help you? Try asking me a question about your documents.",
                thread_ts=thread_ts,
            )
            return

        logger.info(
            "Received mention",
            user=user,
            channel=channel,
            query_preview=query[:50],
        )

        # Add thinking reaction
        try:
            await client.reactions_add(
                channel=channel,
                timestamp=event.get("ts"),
                name="thinking_face",
            )
        except Exception:
            pass  # Reaction failures are non-critical

        try:
            response = await self._process_query(query, user, channel)

            await say(
                text=response["text"],
                blocks=response.get("blocks"),
                thread_ts=thread_ts,
            )

            # Remove thinking reaction and add checkmark
            try:
                await client.reactions_remove(
                    channel=channel,
                    timestamp=event.get("ts"),
                    name="thinking_face",
                )
                await client.reactions_add(
                    channel=channel,
                    timestamp=event.get("ts"),
                    name="white_check_mark",
                )
            except Exception:
                pass  # Reaction failures are non-critical

        except Exception as e:
            logger.error("Failed to process mention", error=str(e))
            await say(
                text="Sorry, I encountered an error. Please try again.",
                thread_ts=thread_ts,
            )

    async def handle_file_shared(
        self,
        event: Dict[str, Any],
        client: AsyncWebClient,
        say: AsyncSay,
    ):
        """
        Handle file uploads.

        Downloads and processes uploaded files.
        """
        file_id = event.get("file_id")
        channel = event.get("channel_id")
        user = event.get("user_id")

        logger.info("File shared", file_id=file_id, channel=channel, user=user)

        try:
            # Get file info
            file_info = await client.files_info(file=file_id)
            file_data = file_info.get("file", {})

            filename = file_data.get("name", "unknown")
            filetype = file_data.get("filetype", "")
            url_private = file_data.get("url_private")

            # Check if it's a supported file type
            supported_types = ["pdf", "docx", "doc", "txt", "md", "pptx", "xlsx"]
            if filetype not in supported_types:
                await say(
                    text=f"I can't process {filetype} files yet. Supported types: {', '.join(supported_types)}",
                    channel=channel,
                )
                return

            await say(
                text=f"Processing file: {filename}... I'll let you know when it's ready.",
                channel=channel,
            )

            # Download and process the file
            try:
                import aiohttp
                import tempfile
                import os

                # Download file using Slack's private URL with auth token
                headers = {"Authorization": f"Bearer {client.token}"}

                async with aiohttp.ClientSession() as session:
                    async with session.get(url_private, headers=headers) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to download: HTTP {response.status}")

                        file_content = await response.read()

                # Save to temp file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, filename)

                try:
                    with open(temp_path, "wb") as f:
                        f.write(file_content)

                    # Process through document pipeline
                    from backend.services.pipeline import DocumentPipeline

                    pipeline = DocumentPipeline()

                    # Create a unique collection for Slack uploads if user context available
                    collection_name = f"slack_uploads_{user}" if user else "slack_uploads"

                    # Process the document
                    result = await pipeline.process_document(
                        file_path=temp_path,
                        metadata={
                            "original_filename": filename,
                            "source": "slack",
                            "channel_id": channel,
                            "user_id": user,
                            "file_id": file_id,
                        },
                        collection=collection_name,
                    )
                finally:
                    # Always cleanup temp file
                    try:
                        os.remove(temp_path)
                        os.rmdir(temp_dir)
                    except Exception:
                        pass

                chunks_count = result.get("chunks_created", 0)
                await say(
                    text=f"✅ File '{filename}' has been processed and added to your documents!\n"
                         f"• Created {chunks_count} searchable chunks\n"
                         f"You can now ask questions about it.",
                    channel=channel,
                )

            except ImportError:
                # Document processor not available
                logger.warning("Document processor not available for Slack file upload")
                await say(
                    text=f"File '{filename}' received but document processing is not configured. "
                         f"Please contact your administrator.",
                    channel=channel,
                )

            except Exception as download_error:
                logger.error("Failed to download/process file", error=str(download_error))
                await say(
                    text=f"I had trouble processing '{filename}'. Error: {str(download_error)[:100]}",
                    channel=channel,
                )

        except Exception as e:
            logger.error("Failed to process file", error=str(e), file_id=file_id)
            await say(
                text="Sorry, I couldn't process that file. Please try again.",
                channel=channel,
            )

    async def handle_app_home_opened(
        self,
        event: Dict[str, Any],
        client: AsyncWebClient,
    ):
        """
        Handle App Home tab opened.

        Displays a welcome message and quick actions.
        """
        user = event.get("user")

        try:
            await client.views_publish(
                user_id=user,
                view={
                    "type": "home",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": "Welcome to AIDocumentIndexer",
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "I'm your AI assistant for document intelligence. Here's what I can do:",
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Commands*\n"
                                        "- `/ask <question>` - Ask questions about your documents\n"
                                        "- `/search <query>` - Search across all documents\n"
                                        "- `/summarize` - Summarize a thread or conversation\n"
                                        "- `/agent <task>` - Run an AI agent task",
                            },
                        },
                        {
                            "type": "divider",
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Quick Actions*",
                            },
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Upload Document",
                                    },
                                    "action_id": "upload_document",
                                },
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "View Documents",
                                    },
                                    "action_id": "view_documents",
                                    "url": "https://your-app.com/dashboard/documents",
                                },
                            ],
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": "You can also DM me directly or @mention me in any channel!",
                                },
                            ],
                        },
                    ],
                },
            )
        except Exception as e:
            logger.error("Failed to publish home view", error=str(e), user=user)

    async def _process_query(
        self,
        query: str,
        user_id: str,
        channel_id: str,
    ) -> Dict[str, Any]:
        """
        Process a user query using the RAG system.

        Returns a response dict with 'text' and optional 'blocks'.
        """
        # Get organization context from bot connection
        org_id = await self._get_organization_for_channel(channel_id)

        if not org_id:
            return {
                "text": "This workspace hasn't been connected to an organization yet. Please set up the integration in your dashboard.",
            }

        try:
            # Use the chat service to get a response
            from backend.services.chat import ChatService

            async with async_session_context() as session:
                chat_service = ChatService(
                    session=session,
                    organization_id=org_id,
                )

                response = await chat_service.get_response(
                    query=query,
                    user_id=user_id,
                    source="slack",
                )

                # Format response with sources
                text = response.get("answer", "I couldn't find a relevant answer.")
                sources = response.get("sources", [])

                blocks = [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": text,
                        },
                    },
                ]

                if sources:
                    source_text = "*Sources:*\n" + "\n".join([
                        f"- {s.get('document_name', 'Unknown')} (relevance: {s.get('score', 0):.0%})"
                        for s in sources[:3]
                    ])

                    blocks.append({
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": source_text,
                            },
                        ],
                    })

                return {
                    "text": text,
                    "blocks": blocks,
                }

        except Exception as e:
            logger.error("Query processing failed", error=str(e))
            return {
                "text": "Sorry, I encountered an error processing your question. Please try again.",
            }

    async def _get_organization_for_channel(self, channel_id: str) -> Optional[uuid.UUID]:
        """Get the organization ID associated with a Slack workspace."""
        try:
            from sqlalchemy import select

            async with async_session_context() as session:
                # Look up bot connection by workspace
                query = select(BotConnection).where(
                    BotConnection.platform == BotPlatform.SLACK,
                    BotConnection.is_active == True,
                )

                result = await session.execute(query)
                connection = result.scalar_one_or_none()

                if connection:
                    return connection.organization_id

        except Exception as e:
            logger.error("Failed to get organization", error=str(e))

        return None
