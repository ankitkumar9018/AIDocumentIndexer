"""
AIDocumentIndexer - Auto-Tagging Service
=========================================

LLM-based automatic document tagging service.
Analyzes document content and generates relevant tags/collections.

LLM provider and model are configured via Admin UI (Operation-Level Config).
Configure the "auto_tagging" operation in Admin > Settings > LLM Configuration.
"""

import asyncio
from typing import List, Optional
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

logger = structlog.get_logger(__name__)


class AutoTaggerService:
    """
    Generate tags for documents using LLM analysis.

    Features:
    - Analyzes document name and content sample
    - Suggests from existing collections when relevant
    - Returns concise, relevant tags

    LLM Configuration:
    - Configure via Admin UI: Admin > Settings > LLM Configuration
    - Operation type: "auto_tagging"
    """

    SYSTEM_PROMPT = """You are a document classification assistant. Your job is to analyze documents and generate relevant tags/categories.

Rules:
1. First tag should be the primary category/collection (most important)
2. Tags should be concise (1-3 words each)
3. If existing collections match the document, prefer those names
4. Focus on: topic, industry, document type, project/client name
5. Return ONLY a comma-separated list of tags, nothing else
6. Never return more than the requested number of tags

Example output: Marketing, FedEx Campaign, Presentation, Q4 2023"""

    def __init__(self):
        """
        Initialize the auto-tagger service.

        LLM provider and model are configured via Admin UI (Operation-Level Config).
        Configure the "auto_tagging" operation in Admin > Settings > LLM Configuration.
        """
        self.model = "unknown"  # Will be set when LLM is retrieved
        self.provider = "unknown"  # Will be set when LLM is retrieved

    async def _get_llm(self):
        """Get LLM instance using database-driven configuration.

        The LLM provider/model is configured via Admin UI:
        Admin > Settings > LLM Configuration > Operation-Level Config > Auto Tagging
        """
        from backend.services.llm import EnhancedLLMFactory

        llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="auto_tagging",
            user_id=None,  # System-level operation
        )

        # Track which model is being used for logging
        self.model = config.model if config else "unknown"
        self.provider = config.provider_type if config else "unknown"

        logger.debug(
            "Auto-tagger using LLM",
            provider=self.provider,
            model=self.model,
        )

        return llm

    async def generate_tags(
        self,
        document_name: str,
        content_sample: str,
        existing_collections: Optional[List[str]] = None,
        max_tags: int = 5
    ) -> List[str]:
        """
        Generate tags for a document using LLM.

        Args:
            document_name: Name of the document file
            content_sample: First ~2000 characters of document content
            existing_collections: List of existing collection names to suggest from
            max_tags: Maximum number of tags to generate (default 5)

        Returns:
            List of generated tags, with primary collection first
        """
        try:
            # Build the prompt
            collections_hint = ""
            if existing_collections:
                # Limit to 20 collections to avoid prompt bloat
                limited_collections = existing_collections[:20]
                collections_hint = f"\n\nExisting collections in the system (prefer these if relevant):\n{', '.join(limited_collections)}"

            # Truncate content sample
            truncated_content = content_sample[:2000] if content_sample else "No content available"

            prompt = f"""Analyze this document and suggest 1-{max_tags} relevant tags/categories.

Document Name: {document_name}

Content Sample:
{truncated_content}
{collections_hint}

Return ONLY a comma-separated list of tags (1-{max_tags} tags):"""

            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            # Get LLM using database-driven configuration
            llm = await self._get_llm()

            # Retry logic for empty responses
            max_retries = 3
            response_text = ""

            for attempt in range(max_retries):
                response = await llm.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)

                if response_text and response_text.strip():
                    logger.debug(
                        "LLM response received for tagging",
                        document_name=document_name,
                        attempt=attempt + 1,
                        response_length=len(response_text),
                    )
                    break
                else:
                    logger.warning(
                        "Empty LLM response for tagging, retrying",
                        document_name=document_name,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        raw_response=repr(response_text) if response_text else "None",
                    )
                    await asyncio.sleep(1.0)
            else:
                # All retries failed
                logger.error(
                    "LLM returned empty response after all retries for tagging",
                    document_name=document_name,
                    retries=max_retries,
                )
                return []

            # Parse response into tag list
            tags = self._parse_tags(response_text, max_tags)

            if not tags:
                logger.warning(
                    "No tags parsed from LLM response",
                    document_name=document_name,
                    response_preview=response_text[:200] if response_text else "",
                )

            logger.info(
                "Generated tags for document",
                document_name=document_name,
                tags=tags,
                tag_count=len(tags)
            )

            return tags

        except Exception as e:
            logger.error(
                "Failed to generate tags",
                document_name=document_name,
                error=str(e),
                exc_info=True
            )
            # Return empty list on failure - caller can handle gracefully
            return []

    def _parse_tags(self, response: str, max_tags: int) -> List[str]:
        """
        Parse LLM response into a clean list of tags.

        Args:
            response: Raw LLM response
            max_tags: Maximum tags to return

        Returns:
            List of cleaned tag strings
        """
        # Clean up response
        response = response.strip()

        # Remove any markdown formatting or quotes
        response = response.replace("*", "").replace('"', "").replace("'", "")

        # Split by comma and clean each tag
        raw_tags = response.split(",")

        tags = []
        for tag in raw_tags:
            # Clean whitespace and newlines
            cleaned = tag.strip().strip("\n").strip()

            # Skip empty tags
            if not cleaned:
                continue

            # Skip if tag is too long (likely an error)
            if len(cleaned) > 50:
                continue

            tags.append(cleaned)

            # Stop at max_tags
            if len(tags) >= max_tags:
                break

        return tags

    async def suggest_collection(
        self,
        document_name: str,
        content_sample: str,
        existing_collections: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Suggest a single collection/category for a document.

        This is a convenience method that returns only the primary tag.

        Args:
            document_name: Name of the document file
            content_sample: Sample of document content
            existing_collections: Existing collections to prefer

        Returns:
            Single collection name, or None if generation fails
        """
        tags = await self.generate_tags(
            document_name=document_name,
            content_sample=content_sample,
            existing_collections=existing_collections,
            max_tags=1
        )

        return tags[0] if tags else None


# Module-level convenience function
async def auto_tag_document(
    document_name: str,
    content_sample: str,
    existing_collections: Optional[List[str]] = None,
    max_tags: int = 5,
) -> List[str]:
    """
    Convenience function to generate tags for a document.

    LLM provider and model are configured via Admin UI (Operation-Level Config).
    Configure the "auto_tagging" operation in Admin > Settings > LLM Configuration.

    Args:
        document_name: Name of the document file
        content_sample: Sample of document content
        existing_collections: Existing collections to suggest from
        max_tags: Maximum tags to generate

    Returns:
        List of generated tags
    """
    service = AutoTaggerService()
    return await service.generate_tags(
        document_name=document_name,
        content_sample=content_sample,
        existing_collections=existing_collections,
        max_tags=max_tags
    )
