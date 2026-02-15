"""
AIDocumentIndexer - Telegram Bot Command Handlers
==================================================

Handles all Telegram bot commands including document Q&A,
search, summarization, and document listing.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


class TelegramCommandHandler:
    """
    Handles Telegram bot commands for document interaction.

    Provides RAG-based Q&A, document search, summarization,
    and document listing functionality.
    """

    def __init__(
        self,
        organization_id: Optional[str] = None,
        max_response_length: int = 4000,  # Telegram limit is 4096
    ):
        """
        Initialize command handler.

        Args:
            organization_id: Organization context for queries
            max_response_length: Maximum response length (Telegram limit)
        """
        self.organization_id = organization_id
        self.max_response_length = max_response_length
        self._rag_service = None
        self._search_service = None

    async def _get_rag_service(self):
        """Lazy load RAG service."""
        if self._rag_service is None:
            try:
                from backend.services.rag import RAGService
                self._rag_service = RAGService()
            except Exception as e:
                logger.warning("RAG service not available", error=str(e))
        return self._rag_service

    async def _get_search_service(self):
        """Lazy load search service."""
        if self._search_service is None:
            try:
                from backend.services.search_cache import SearchService
                self._search_service = SearchService()
            except ImportError:
                logger.warning("Search service not available")
        return self._search_service

    async def handle_question(
        self,
        question: str,
        user_id: str,
        chat_id: str,
    ) -> Dict[str, Any]:
        """
        Handle a document Q&A question.

        Args:
            question: User's question
            user_id: Telegram user ID
            chat_id: Telegram chat ID

        Returns:
            Response dict with 'text' and optional 'sources'
        """
        if not question.strip():
            return {
                "text": "Please provide a question.\n\nExample: `/ask What are the key findings in the Q4 report?`"
            }

        try:
            rag_service = await self._get_rag_service()
            if rag_service is None:
                return {
                    "text": "⚠️ Sorry, the document Q&A service is temporarily unavailable."
                }

            # Query RAG service
            result = await rag_service.query(
                query=question,
                organization_id=self.organization_id,
                user_id=user_id,
                metadata={
                    "source": "telegram",
                    "chat_id": chat_id,
                },
            )

            # Format response
            answer = result.get("answer", "I couldn't find a relevant answer.")
            sources = result.get("sources", [])

            response_text = answer

            # Add source citations if available
            if sources:
                source_text = "\n\n📚 *Sources:*\n"
                for i, source in enumerate(sources[:5], 1):
                    doc_name = source.get("document_name", "Unknown")
                    # Escape markdown special characters
                    doc_name = self._escape_markdown(doc_name)
                    source_text += f"{i}. {doc_name}\n"
                response_text += source_text

            # Truncate if too long
            if len(response_text) > self.max_response_length:
                response_text = response_text[: self.max_response_length - 30] + "\n\n... _(truncated)_"

            logger.info(
                "Telegram question answered",
                user_id=user_id,
                question_length=len(question),
                answer_length=len(response_text),
            )

            return {"text": response_text, "sources": sources}

        except Exception as e:
            logger.error("Error handling Telegram question", error=str(e))
            return {
                "text": "⚠️ Sorry, I encountered an error processing your question. Please try again."
            }

    async def handle_search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Handle document search.

        Args:
            query: Search query
            user_id: Telegram user ID
            limit: Maximum results to return

        Returns:
            Response dict with search results
        """
        if not query.strip():
            return {
                "text": "Please provide a search query.\n\nExample: `/search quarterly revenue`"
            }

        try:
            search_service = await self._get_search_service()
            if search_service is None:
                return {
                    "text": "⚠️ Sorry, the search service is temporarily unavailable."
                }

            # Perform search
            results = await search_service.search(
                query=query,
                organization_id=self.organization_id,
                limit=limit,
            )

            if not results:
                return {
                    "text": f"🔍 No documents found matching: *{self._escape_markdown(query)}*\n\nTry different keywords or upload more documents."
                }

            # Format results
            response_text = f"🔍 *Search Results for:* {self._escape_markdown(query)}\n\n"

            for i, doc in enumerate(results, 1):
                name = self._escape_markdown(doc.get("name", "Untitled"))
                doc_type = doc.get("type", "document").upper()
                updated = doc.get("updated_at", "")
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        updated = dt.strftime("%b %d, %Y")
                    except (ValueError, AttributeError):
                        pass

                response_text += f"*{i}. {name}*\n"
                response_text += f"   📁 {doc_type}"
                if updated:
                    response_text += f" | 📅 {updated}"
                response_text += "\n\n"

            # Truncate if needed
            if len(response_text) > self.max_response_length:
                response_text = response_text[: self.max_response_length - 30] + "\n... _(truncated)_"

            logger.info(
                "Telegram search completed",
                user_id=user_id,
                query=query,
                results_count=len(results),
            )

            return {"text": response_text, "results": results}

        except Exception as e:
            logger.error("Error handling Telegram search", error=str(e))
            return {
                "text": "⚠️ Sorry, I encountered an error performing the search. Please try again."
            }

    async def handle_summarize(
        self,
        doc_identifier: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Handle document summarization.

        Args:
            doc_identifier: Document name or ID
            user_id: Telegram user ID

        Returns:
            Response dict with summary
        """
        if not doc_identifier.strip():
            return {
                "text": "Please specify a document to summarize.\n\nExample: `/summarize Q4-Report.pdf`"
            }

        try:
            rag_service = await self._get_rag_service()
            if rag_service is None:
                return {
                    "text": "⚠️ Sorry, the summarization service is temporarily unavailable."
                }

            # Generate summary using RAG
            result = await rag_service.query(
                query=f"Provide a comprehensive summary of the document: {doc_identifier}",
                organization_id=self.organization_id,
                user_id=user_id,
                metadata={
                    "source": "telegram",
                    "task": "summarize",
                    "document": doc_identifier,
                },
            )

            summary = result.get("answer", "")

            if not summary or "not found" in summary.lower():
                return {
                    "text": f"❌ Could not find or summarize document: *{self._escape_markdown(doc_identifier)}*\n\nUse /docs to see available documents."
                }

            response_text = f"📄 *Summary of {self._escape_markdown(doc_identifier)}:*\n\n{summary}"

            # Truncate if needed
            if len(response_text) > self.max_response_length:
                response_text = response_text[: self.max_response_length - 30] + "\n\n... _(truncated)_"

            logger.info(
                "Telegram summarization completed",
                user_id=user_id,
                document=doc_identifier,
            )

            return {"text": response_text}

        except Exception as e:
            logger.error("Error handling Telegram summarize", error=str(e))
            return {
                "text": "⚠️ Sorry, I encountered an error generating the summary. Please try again."
            }

    async def handle_list_documents(
        self,
        user_id: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Handle listing recent documents.

        Args:
            user_id: Telegram user ID
            limit: Maximum documents to list

        Returns:
            Response dict with document list
        """
        try:
            search_service = await self._get_search_service()
            if search_service is None:
                return {
                    "text": "⚠️ Sorry, the document service is temporarily unavailable."
                }

            # Get recent documents
            documents = await search_service.list_documents(
                organization_id=self.organization_id,
                limit=limit,
                sort_by="updated_at",
                sort_order="desc",
            )

            if not documents:
                return {
                    "text": "📭 No documents found.\n\nUpload documents to get started!"
                }

            response_text = "📚 *Recent Documents:*\n\n"

            for i, doc in enumerate(documents, 1):
                name = self._escape_markdown(doc.get("name", "Untitled"))
                doc_type = doc.get("type", "doc").upper()
                updated = doc.get("updated_at", "")
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        updated = dt.strftime("%b %d")
                    except (ValueError, AttributeError):
                        updated = ""

                response_text += f"{i}. *{name}* ({doc_type})"
                if updated:
                    response_text += f" - {updated}"
                response_text += "\n"

            response_text += "\n💡 Use `/search <query>` to find specific documents."

            logger.info(
                "Telegram document list returned",
                user_id=user_id,
                documents_count=len(documents),
            )

            return {"text": response_text, "documents": documents}

        except Exception as e:
            logger.error("Error handling Telegram docs list", error=str(e))
            return {
                "text": "⚠️ Sorry, I encountered an error listing documents. Please try again."
            }

    def _escape_markdown(self, text: str) -> str:
        """Escape special Markdown characters for Telegram."""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    def get_help_text(self) -> str:
        """
        Get help text for Telegram bot commands.

        Returns:
            Formatted help message (Markdown)
        """
        return """
🤖 *AIDocumentIndexer Bot Help*

*Commands:*
• `/ask <question>` \\- Ask a question about your documents
• `/search <query>` \\- Search for documents
• `/summarize <document>` \\- Get a summary of a document
• `/docs` \\- List your recent documents
• `/help` \\- Show this help message

*Quick Tips:*
• In private chats, just type your question directly
• In groups, mention @bot or reply to the bot's messages
• Use specific keywords for better search results

*Examples:*
• `/ask What were the key findings in the quarterly report?`
• `/search marketing strategy 2024`
• `/summarize Q4\\-Financial\\-Report\\.pdf`

📖 For more information, visit our documentation\\.
        """.strip()
