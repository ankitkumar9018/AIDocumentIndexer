"""
AIDocumentIndexer - Teams Activity Handlers
============================================

Handlers for different Teams activity types.
"""

import re
from typing import Dict, Any, Optional, List

import structlog

logger = structlog.get_logger(__name__)


class TeamsActivityHandler:
    """
    Handler for Teams activities.

    Processes messages, commands, and card actions.
    """

    def __init__(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.organization_id = organization_id
        self.user_id = user_id

    async def on_message(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming message.

        Args:
            activity: Teams activity object

        Returns:
            Response message
        """
        text = activity.get("text", "").strip()
        from_user = activity.get("from", {})
        conversation = activity.get("conversation", {})

        # Remove bot mention from text
        text = self._remove_mentions(text, activity)

        logger.info(
            "Received message",
            text=text[:100] if text else "",
            user=from_user.get("name"),
            conversation_type=conversation.get("conversationType"),
        )

        # Check for commands
        if text.startswith("/"):
            return await self._handle_command(text, activity)

        # Regular message - process as question
        return await self.handle_question(text, activity)

    def _remove_mentions(self, text: str, activity: Dict[str, Any]) -> str:
        """Remove @mentions from text."""
        mentions = activity.get("entities", [])

        for entity in mentions:
            if entity.get("type") == "mention":
                mentioned_text = entity.get("text", "")
                text = text.replace(mentioned_text, "").strip()

        return text

    async def _handle_command(
        self,
        text: str,
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle slash commands."""
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            return self._create_help_response()
        elif command == "/search":
            return await self.handle_search(args, activity)
        elif command == "/ask":
            return await self.handle_question(args, activity)
        elif command == "/summarize":
            return await self.handle_summarize(args, activity)
        elif command == "/documents":
            return await self.handle_list_documents(activity)
        else:
            return {
                "type": "message",
                "text": f"Unknown command: {command}\n\nUse /help to see available commands.",
            }

    def _create_help_response(self) -> Dict[str, Any]:
        """Create help message with available commands."""
        from backend.integrations.teams_bot.cards import create_help_card

        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": create_help_card(),
                }
            ],
        }

    async def handle_question(
        self,
        question: str,
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle a question using RAG.

        Args:
            question: User's question
            activity: Teams activity for context

        Returns:
            Response with answer
        """
        if not question:
            return {
                "type": "message",
                "text": "Please provide a question. For example:\n\n`What are our company policies on remote work?`",
            }

        try:
            # Get answer from RAG service
            answer, sources = await self._query_rag(question)

            # Create response card
            from backend.integrations.teams_bot.cards import create_answer_card

            card = create_answer_card(
                question=question,
                answer=answer,
                sources=sources,
            )

            return {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            }

        except Exception as e:
            logger.error("Error processing question", error=e)
            return {
                "type": "message",
                "text": f"Sorry, I encountered an error processing your question. Please try again later.",
            }

    async def handle_search(
        self,
        query: str,
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle document search.

        Args:
            query: Search query
            activity: Teams activity

        Returns:
            Response with search results
        """
        if not query:
            return {
                "type": "message",
                "text": "Please provide a search query. For example:\n\n`/search quarterly report 2024`",
            }

        try:
            # Search documents
            results = await self._search_documents(query)

            # Create results card
            from backend.integrations.teams_bot.cards import create_search_results_card

            card = create_search_results_card(query=query, results=results)

            return {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            }

        except Exception as e:
            logger.error("Error searching documents", error=e)
            return {
                "type": "message",
                "text": f"Sorry, I encountered an error searching documents. Please try again later.",
            }

    async def handle_summarize(
        self,
        args: str,
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle document summarization.

        Args:
            args: Document name or ID
            activity: Teams activity

        Returns:
            Response with summary
        """
        if not args:
            return {
                "type": "message",
                "text": "Please specify a document to summarize. For example:\n\n`/summarize Q4 Report.pdf`",
            }

        try:
            summary = await self._summarize_document(args)

            return {
                "type": "message",
                "text": f"**Summary of {args}:**\n\n{summary}",
            }

        except Exception as e:
            logger.error("Error summarizing document", error=e)
            return {
                "type": "message",
                "text": f"Sorry, I couldn't find or summarize that document.",
            }

    async def handle_list_documents(
        self,
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        List recent documents.

        Args:
            activity: Teams activity

        Returns:
            Response with document list
        """
        try:
            documents = await self._get_recent_documents()

            from backend.integrations.teams_bot.cards import create_documents_list_card

            card = create_documents_list_card(documents)

            return {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            }

        except Exception as e:
            logger.error("Error listing documents", error=e)
            return {
                "type": "message",
                "text": "Sorry, I couldn't retrieve the document list.",
            }

    async def _query_rag(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        """Query RAG service for an answer."""
        try:
            from backend.services.rag import RAGService
            from backend.db.database import async_session_context

            async with async_session_context() as session:
                rag = RAGService(
                    session=session,
                    organization_id=self.organization_id,
                    user_id=self.user_id,
                )

                result = await rag.query(
                    query=question,
                    max_results=5,
                )

                answer = result.get("response", "I couldn't find an answer to your question.")
                sources = result.get("sources", [])

                return answer, sources

        except Exception as e:
            logger.error("RAG query failed", error=e)
            return "I'm having trouble accessing the knowledge base. Please try again later.", []

    async def _search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search documents."""
        try:
            from backend.services.search_cache import SearchService
            from backend.db.database import async_session_context

            async with async_session_context() as session:
                search = SearchService(
                    session=session,
                    organization_id=self.organization_id,
                )

                results = await search.search(
                    query=query,
                    limit=10,
                )

                return results

        except Exception as e:
            logger.error("Search failed", error=e)
            return []

    async def _summarize_document(self, doc_identifier: str) -> str:
        """Get or generate document summary."""
        try:
            from backend.db.database import async_session_context
            from backend.db.models import Document
            from sqlalchemy import select

            async with async_session_context() as session:
                # Try to find document by name
                # PHASE 12 FIX: Include docs from user's org AND docs without org (legacy/shared)
                from sqlalchemy import or_
                query = select(Document).where(
                    or_(
                        Document.organization_id == self.organization_id,
                        Document.organization_id.is_(None),
                    ),
                    Document.filename.ilike(f"%{doc_identifier}%"),
                )
                result = await session.execute(query)
                doc = result.scalar_one_or_none()

                if doc and doc.enhanced_metadata:
                    return doc.enhanced_metadata.get("summary_short", "No summary available.")

                return "Document not found or no summary available."

        except Exception as e:
            logger.error("Summarization failed", error=e)
            return "Could not generate summary."

    async def _get_recent_documents(self) -> List[Dict[str, Any]]:
        """Get recent documents."""
        try:
            from backend.db.database import async_session_context
            from backend.db.models import Document
            from sqlalchemy import select

            async with async_session_context() as session:
                # PHASE 12 FIX: Include docs from user's org AND docs without org (legacy/shared)
                from sqlalchemy import or_
                query = (
                    select(Document)
                    .where(
                        or_(
                            Document.organization_id == self.organization_id,
                            Document.organization_id.is_(None),
                        )
                    )
                    .order_by(Document.created_at.desc())
                    .limit(10)
                )
                result = await session.execute(query)
                documents = result.scalars().all()

                return [
                    {
                        "id": str(doc.id),
                        "name": doc.filename or doc.title,
                        "type": doc.file_type,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    }
                    for doc in documents
                ]

        except Exception as e:
            logger.error("Failed to get documents", error=e)
            return []
