"""
AIDocumentIndexer - Slack Slash Commands
=========================================

Handles Slack slash commands for document operations.

Commands:
- /ask <question> - Ask questions about documents
- /search <query> - Search across documents
- /summarize - Summarize thread or channel
- /agent <task> - Run an agent task
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


class SlackCommandHandler:
    """
    Handles Slack slash commands.
    """

    def __init__(self, app: AsyncApp):
        self.app = app

    async def handle_ask_command(
        self,
        ack: AsyncAck,
        command: Dict[str, Any],
        respond: AsyncRespond,
        client: AsyncWebClient,
    ):
        """
        Handle /ask command.

        Usage: /ask <question about your documents>
        """
        await ack()

        query = command.get("text", "").strip()
        user_id = command.get("user_id")
        channel_id = command.get("channel_id")

        if not query:
            await respond(
                text="Please provide a question. Usage: `/ask <your question>`",
                response_type="ephemeral",
            )
            return

        logger.info("Ask command", user=user_id, query_preview=query[:50])

        # Show thinking message
        await respond(
            text=f"Searching for an answer to: _{query}_...",
            response_type="ephemeral",
        )

        try:
            response = await self._query_documents(query, user_id, channel_id)

            # Post the response publicly
            await respond(
                text=response["text"],
                blocks=response.get("blocks"),
                response_type="in_channel",
            )

        except Exception as e:
            logger.error("Ask command failed", error=str(e))
            await respond(
                text="Sorry, I encountered an error. Please try again.",
                response_type="ephemeral",
            )

    async def handle_search_command(
        self,
        ack: AsyncAck,
        command: Dict[str, Any],
        respond: AsyncRespond,
        client: AsyncWebClient,
    ):
        """
        Handle /search command.

        Usage: /search <search query>
        """
        await ack()

        query = command.get("text", "").strip()
        user_id = command.get("user_id")
        channel_id = command.get("channel_id")

        if not query:
            await respond(
                text="Please provide a search query. Usage: `/search <query>`",
                response_type="ephemeral",
            )
            return

        logger.info("Search command", user=user_id, query=query)

        try:
            results = await self._search_documents(query, user_id, channel_id)

            if not results:
                await respond(
                    text=f"No documents found matching: _{query}_",
                    response_type="ephemeral",
                )
                return

            # Format results as blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Search Results for: {query}",
                    },
                },
            ]

            for i, doc in enumerate(results[:5], 1):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{i}. {doc['name']}*\n{doc['preview'][:200]}...",
                    },
                    "accessory": {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View",
                        },
                        "url": doc.get("url", "#"),
                        "action_id": f"view_doc_{doc['id']}",
                    },
                })

            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Found {len(results)} matching documents",
                    },
                ],
            })

            await respond(
                text=f"Found {len(results)} documents matching: {query}",
                blocks=blocks,
                response_type="in_channel",
            )

        except Exception as e:
            logger.error("Search command failed", error=str(e))
            await respond(
                text="Sorry, search failed. Please try again.",
                response_type="ephemeral",
            )

    async def handle_summarize_command(
        self,
        ack: AsyncAck,
        command: Dict[str, Any],
        respond: AsyncRespond,
        client: AsyncWebClient,
    ):
        """
        Handle /summarize command.

        Usage: /summarize [thread|channel|<url>]
        """
        await ack()

        text = command.get("text", "").strip()
        user_id = command.get("user_id")
        channel_id = command.get("channel_id")

        logger.info("Summarize command", user=user_id, text=text)

        await respond(
            text="Generating summary...",
            response_type="ephemeral",
        )

        try:
            # Get channel history
            messages = await self._get_channel_messages(client, channel_id, limit=50)

            if not messages:
                await respond(
                    text="No messages found to summarize.",
                    response_type="ephemeral",
                )
                return

            # Generate summary using LLM
            summary = await self._summarize_messages(messages, user_id, channel_id)

            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Channel Summary",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": summary,
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Based on the last {len(messages)} messages",
                        },
                    ],
                },
            ]

            await respond(
                text=summary,
                blocks=blocks,
                response_type="in_channel",
            )

        except Exception as e:
            logger.error("Summarize command failed", error=str(e))
            await respond(
                text="Sorry, I couldn't generate a summary. Please try again.",
                response_type="ephemeral",
            )

    async def handle_agent_command(
        self,
        ack: AsyncAck,
        command: Dict[str, Any],
        respond: AsyncRespond,
        client: AsyncWebClient,
    ):
        """
        Handle /agent command.

        Usage: /agent <task description>
        """
        await ack()

        task = command.get("text", "").strip()
        user_id = command.get("user_id")
        channel_id = command.get("channel_id")

        if not task:
            await respond(
                text="Please provide a task. Usage: `/agent <task description>`\n"
                     "Example: `/agent Create a summary report of Q4 sales data`",
                response_type="ephemeral",
            )
            return

        logger.info("Agent command", user=user_id, task_preview=task[:50])

        await respond(
            text=f"Starting agent task: _{task}_\nThis may take a few moments...",
            response_type="in_channel",
        )

        try:
            result = await self._run_agent_task(task, user_id, channel_id)

            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Agent Task Completed",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Task:* {task}",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Result:*\n{result['output']}",
                    },
                },
            ]

            if result.get("artifacts"):
                artifacts_text = "\n".join([
                    f"- {a['name']}: {a['url']}"
                    for a in result["artifacts"]
                ])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Generated Files:*\n{artifacts_text}",
                    },
                })

            await respond(
                text=f"Task completed: {result['output'][:200]}...",
                blocks=blocks,
                response_type="in_channel",
            )

        except Exception as e:
            logger.error("Agent command failed", error=str(e))
            await respond(
                text=f"Agent task failed: {str(e)}",
                response_type="in_channel",
            )

    async def handle_summarize_shortcut(
        self,
        ack: AsyncAck,
        shortcut: Dict[str, Any],
        client: AsyncWebClient,
    ):
        """
        Handle the summarize thread shortcut.

        Triggered from message menu.
        """
        await ack()

        message = shortcut.get("message", {})
        channel_id = shortcut.get("channel", {}).get("id")
        thread_ts = message.get("thread_ts") or message.get("ts")
        user_id = shortcut.get("user", {}).get("id")

        logger.info("Summarize shortcut", channel=channel_id, thread=thread_ts)

        try:
            # Get thread messages
            response = await client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=100,
            )

            messages = response.get("messages", [])

            if len(messages) < 2:
                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="This thread doesn't have enough messages to summarize.",
                )
                return

            # Generate summary
            summary = await self._summarize_messages(messages, user_id, channel_id)

            # Post summary in thread
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f"*Thread Summary*\n\n{summary}",
            )

        except Exception as e:
            logger.error("Summarize shortcut failed", error=str(e))
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text="Sorry, I couldn't summarize this thread.",
            )

    async def _query_documents(
        self,
        query: str,
        user_id: str,
        channel_id: str,
    ) -> Dict[str, Any]:
        """Query documents using RAG."""
        org_id = await self._get_organization_for_workspace(channel_id)

        if not org_id:
            return {
                "text": "This workspace hasn't been connected yet. Please set up the integration.",
            }

        try:
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
                    blocks.append({
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": "*Sources:* " + ", ".join([
                                    s.get("document_name", "Unknown")
                                    for s in sources[:3]
                                ]),
                            },
                        ],
                    })

                return {"text": text, "blocks": blocks}

        except Exception as e:
            logger.error("Query failed", error=str(e))
            return {"text": "Sorry, I encountered an error."}

    async def _search_documents(
        self,
        query: str,
        user_id: str,
        channel_id: str,
    ) -> List[Dict[str, Any]]:
        """Search documents."""
        org_id = await self._get_organization_for_workspace(channel_id)

        if not org_id:
            return []

        try:
            from backend.services.document_service import DocumentService

            async with async_session_context() as session:
                doc_service = DocumentService(
                    session=session,
                    organization_id=org_id,
                )

                results = await doc_service.search(query=query, limit=10)

                return [
                    {
                        "id": str(doc.id),
                        "name": doc.filename or doc.title or "Untitled",
                        "preview": doc.enhanced_metadata.get("summary_short", "")
                        if doc.enhanced_metadata else "",
                        "url": f"/dashboard/documents/{doc.id}",
                    }
                    for doc in results
                ]

        except Exception as e:
            logger.error("Search failed", error=str(e))
            return []

    async def _summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        channel_id: str,
    ) -> str:
        """Summarize a list of messages using LLM."""
        org_id = await self._get_organization_for_workspace(channel_id)

        # Format messages for summarization
        message_text = "\n".join([
            f"- {m.get('text', '')}"
            for m in messages
            if m.get("text") and not m.get("bot_id")
        ])

        try:
            from backend.services.llm import LLMService

            async with async_session_context() as session:
                llm_service = LLMService(
                    session=session,
                    organization_id=org_id,
                )

                response = await llm_service.generate(
                    prompt=f"Please summarize the following conversation in a clear, concise format:\n\n{message_text}",
                    system_prompt="You are a helpful assistant that summarizes conversations. Provide key points and action items if any.",
                    max_tokens=500,
                )

                return response.get("content", "Unable to generate summary.")

        except Exception as e:
            logger.error("Summarization failed", error=str(e))
            return "Sorry, I couldn't generate a summary."

    async def _run_agent_task(
        self,
        task: str,
        user_id: str,
        channel_id: str,
    ) -> Dict[str, Any]:
        """Run an agent task."""
        org_id = await self._get_organization_for_workspace(channel_id)

        if not org_id:
            return {"output": "Workspace not connected.", "artifacts": []}

        try:
            from backend.services.agent_orchestrator import AgentOrchestrator

            async with async_session_context() as session:
                orchestrator = AgentOrchestrator(
                    session=session,
                    organization_id=org_id,
                )

                result = await orchestrator.execute_task(
                    task=task,
                    source="slack",
                    user_id=user_id,
                )

                return {
                    "output": result.get("output", "Task completed."),
                    "artifacts": result.get("artifacts", []),
                }

        except Exception as e:
            logger.error("Agent task failed", error=str(e))
            raise

    async def _get_channel_messages(
        self,
        client: AsyncWebClient,
        channel_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent channel messages."""
        try:
            response = await client.conversations_history(
                channel=channel_id,
                limit=limit,
            )
            return response.get("messages", [])
        except Exception as e:
            logger.error("Failed to get messages", error=str(e))
            return []

    async def _get_organization_for_workspace(
        self,
        channel_id: str,
    ) -> Optional[uuid.UUID]:
        """Get organization ID for workspace."""
        try:
            from sqlalchemy import select

            async with async_session_context() as session:
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
