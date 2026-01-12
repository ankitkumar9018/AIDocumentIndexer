"""
AIDocumentIndexer - Mention Parser Service
===========================================

Parses @-mention syntax in chat messages to extract document/folder filters.

Supported @mention patterns:
- @folder:FolderName       → Filter to documents in folder
- @document:filename.pdf   → Filter to specific document
- @doc:filename.pdf        → Shorthand for @document
- @tag:tagname             → Filter by collection/tag
- @recent:7d               → Filter to last 7 days (d=days, w=weeks, m=months)
- @all                     → Search all documents (removes default filters)

Example:
    "What are the Q4 results? @folder:Finance @tag:quarterly"
    → Query: "What are the Q4 results?"
    → Filters: folder_name="Finance", collection="quarterly"
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MentionFilter:
    """Represents a parsed @mention filter."""
    mention_type: str  # folder, document, tag, recent, all
    value: str  # The value after the colon (e.g., "Marketing" for @folder:Marketing)
    original: str  # Original mention text (e.g., "@folder:Marketing")
    resolved_id: Optional[str] = None  # Resolved UUID if applicable
    resolved_name: Optional[str] = None  # Resolved display name


@dataclass
class ParsedQuery:
    """Result of parsing a chat query for @mentions."""
    clean_query: str  # Query with @mentions removed
    original_query: str  # Original query
    mentions: List[MentionFilter] = field(default_factory=list)

    # Resolved filter values
    folder_ids: List[str] = field(default_factory=list)
    folder_names: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    document_names: List[str] = field(default_factory=list)
    collection_filters: List[str] = field(default_factory=list)
    date_filter: Optional[datetime] = None  # Filter docs modified after this date
    search_all: bool = False  # True if @all was specified

    # For autocomplete suggestions
    partial_mention: Optional[str] = None  # Incomplete mention being typed
    partial_type: Optional[str] = None  # Type of partial mention (folder, doc, tag)

    def has_filters(self) -> bool:
        """Check if any filters are active."""
        return bool(
            self.folder_ids or
            self.folder_names or
            self.document_ids or
            self.document_names or
            self.collection_filters or
            self.date_filter or
            self.search_all
        )


class MentionParser:
    """
    Parses @-mention syntax from chat queries.

    Usage:
        parser = MentionParser()
        result = parser.parse("What are sales? @folder:Marketing @recent:7d")
        print(result.clean_query)  # "What are sales?"
        print(result.folder_names)  # ["Marketing"]
        print(result.date_filter)  # datetime 7 days ago
    """

    # Regex patterns for @mentions
    MENTION_PATTERNS = {
        "folder": re.compile(r'@folder:([^\s]+)', re.IGNORECASE),
        "document": re.compile(r'@(?:document|doc):([^\s]+)', re.IGNORECASE),
        "tag": re.compile(r'@tag:([^\s]+)', re.IGNORECASE),
        "recent": re.compile(r'@recent:(\d+)([dwm])', re.IGNORECASE),
        "all": re.compile(r'@all\b', re.IGNORECASE),
    }

    # Pattern for detecting partial mentions (for autocomplete)
    PARTIAL_MENTION = re.compile(r'@(\w*)(?::([^\s]*))?$')

    def __init__(self):
        """Initialize the mention parser."""
        logger.debug("MentionParser initialized")

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a query string for @mentions.

        Args:
            query: The user's chat query

        Returns:
            ParsedQuery with extracted filters and clean query
        """
        result = ParsedQuery(
            original_query=query,
            clean_query=query,
        )

        # Find all mentions
        for mention_type, pattern in self.MENTION_PATTERNS.items():
            matches = pattern.finditer(query)
            for match in matches:
                mention = self._process_match(mention_type, match)
                if mention:
                    result.mentions.append(mention)

                    # Apply to appropriate filter list
                    self._apply_mention_to_result(result, mention)

                    # Remove mention from clean query
                    result.clean_query = result.clean_query.replace(match.group(0), "").strip()

        # Clean up extra whitespace in clean query
        result.clean_query = " ".join(result.clean_query.split())

        # Check for partial mention at end (for autocomplete)
        partial_match = self.PARTIAL_MENTION.search(query)
        if partial_match:
            partial_type = partial_match.group(1).lower() if partial_match.group(1) else None
            partial_value = partial_match.group(2) if partial_match.group(2) else ""

            # Only set partial if it's incomplete (no value after colon yet or cursor at end)
            if partial_type and not partial_value and query.endswith(f"@{partial_type}:"):
                result.partial_type = partial_type
                result.partial_mention = f"@{partial_type}:"
            elif partial_type and query.endswith(f"@{partial_type}"):
                result.partial_type = partial_type
                result.partial_mention = f"@{partial_type}"

        logger.debug(
            "Query parsed",
            original=query[:50],
            clean=result.clean_query[:50] if result.clean_query else "",
            mention_count=len(result.mentions),
            has_filters=result.has_filters(),
        )

        return result

    def _process_match(self, mention_type: str, match: re.Match) -> Optional[MentionFilter]:
        """Process a regex match and create a MentionFilter."""
        original = match.group(0)

        if mention_type == "all":
            return MentionFilter(
                mention_type="all",
                value="",
                original=original,
            )
        elif mention_type == "recent":
            number = int(match.group(1))
            unit = match.group(2).lower()
            return MentionFilter(
                mention_type="recent",
                value=f"{number}{unit}",
                original=original,
            )
        else:
            value = match.group(1)
            # Handle quoted values (e.g., @folder:"My Folder Name")
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            return MentionFilter(
                mention_type=mention_type,
                value=value,
                original=original,
            )

    def _apply_mention_to_result(self, result: ParsedQuery, mention: MentionFilter):
        """Apply a mention filter to the parsed result."""
        if mention.mention_type == "folder":
            # Check if it looks like a UUID
            try:
                UUID(mention.value)
                result.folder_ids.append(mention.value)
            except ValueError:
                result.folder_names.append(mention.value)

        elif mention.mention_type == "document":
            # Check if it looks like a UUID
            try:
                UUID(mention.value)
                result.document_ids.append(mention.value)
            except ValueError:
                result.document_names.append(mention.value)

        elif mention.mention_type == "tag":
            result.collection_filters.append(mention.value)

        elif mention.mention_type == "recent":
            # Parse time period
            number = int(mention.value[:-1])
            unit = mention.value[-1].lower()

            if unit == "d":
                delta = timedelta(days=number)
            elif unit == "w":
                delta = timedelta(weeks=number)
            elif unit == "m":
                delta = timedelta(days=number * 30)  # Approximate month
            else:
                delta = timedelta(days=7)  # Default to 7 days

            # Set date filter to current time minus delta
            result.date_filter = datetime.utcnow() - delta

        elif mention.mention_type == "all":
            result.search_all = True

    async def resolve_mentions(
        self,
        parsed: ParsedQuery,
        session,  # AsyncSession
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ParsedQuery:
        """
        Resolve mention names to IDs using database lookups.

        This converts folder/document names to their actual UUIDs
        for filtering in the vector store.

        Args:
            parsed: The parsed query result
            session: Database session
            organization_id: Organization scope
            user_id: User scope for permissions

        Returns:
            Updated ParsedQuery with resolved IDs
        """
        from sqlalchemy import select
        from backend.db.models import Document, Folder

        # Resolve folder names to IDs
        if parsed.folder_names:
            for folder_name in parsed.folder_names:
                query = select(Folder).where(
                    Folder.name.ilike(f"%{folder_name}%")
                )
                if organization_id:
                    try:
                        org_uuid = UUID(organization_id)
                        query = query.where(Folder.organization_id == org_uuid)
                    except ValueError:
                        pass

                result = await session.execute(query)
                folders = result.scalars().all()

                for folder in folders:
                    if str(folder.id) not in parsed.folder_ids:
                        parsed.folder_ids.append(str(folder.id))
                        # Update the mention with resolved info
                        for mention in parsed.mentions:
                            if mention.mention_type == "folder" and mention.value == folder_name:
                                mention.resolved_id = str(folder.id)
                                mention.resolved_name = folder.name

        # Resolve document names to IDs
        if parsed.document_names:
            for doc_name in parsed.document_names:
                query = select(Document).where(
                    Document.file_name.ilike(f"%{doc_name}%")
                )
                if organization_id:
                    try:
                        org_uuid = UUID(organization_id)
                        query = query.where(Document.organization_id == org_uuid)
                    except ValueError:
                        pass

                result = await session.execute(query)
                documents = result.scalars().all()

                for doc in documents:
                    if str(doc.id) not in parsed.document_ids:
                        parsed.document_ids.append(str(doc.id))
                        # Update the mention with resolved info
                        for mention in parsed.mentions:
                            if mention.mention_type == "document" and mention.value == doc_name:
                                mention.resolved_id = str(doc.id)
                                mention.resolved_name = doc.file_name

        logger.debug(
            "Mentions resolved",
            folder_ids=len(parsed.folder_ids),
            document_ids=len(parsed.document_ids),
            collection_filters=parsed.collection_filters,
        )

        return parsed

    def to_rag_filters(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """
        Convert parsed mentions to RAG service filter parameters.

        Args:
            parsed: The parsed query result

        Returns:
            Dict of filter parameters for RAGService.query()
        """
        filters = {}

        # Collection/tag filter - combine multiple with OR logic
        if parsed.collection_filters:
            if len(parsed.collection_filters) == 1:
                filters["collection_filter"] = parsed.collection_filters[0]
            else:
                # For multiple collections, use the first one
                # (RAG service currently supports single collection filter)
                filters["collection_filter"] = parsed.collection_filters[0]

        # Document ID filter
        if parsed.document_ids:
            filters["document_ids"] = parsed.document_ids

        # Folder ID filter - get documents in folders
        if parsed.folder_ids:
            filters["folder_ids"] = parsed.folder_ids

        # Date filter
        if parsed.date_filter:
            filters["date_after"] = parsed.date_filter

        # Search all - might need to override default org scope
        if parsed.search_all:
            filters["search_all"] = True

        return filters


# =============================================================================
# Autocomplete Support
# =============================================================================

@dataclass
class AutocompleteSuggestion:
    """A single autocomplete suggestion."""
    type: str  # folder, document, tag
    value: str  # The value to insert
    display: str  # Display text
    description: Optional[str] = None  # Optional description


class MentionAutocomplete:
    """
    Provides autocomplete suggestions for @mentions.

    Usage:
        autocomplete = MentionAutocomplete()
        suggestions = await autocomplete.get_suggestions(
            partial="@folder:Mar",
            session=db_session,
            organization_id=org_id,
        )
    """

    # Mention type suggestions
    TYPE_SUGGESTIONS = [
        AutocompleteSuggestion(
            type="type",
            value="@folder:",
            display="@folder:",
            description="Filter by folder name"
        ),
        AutocompleteSuggestion(
            type="type",
            value="@document:",
            display="@document:",
            description="Filter by document name"
        ),
        AutocompleteSuggestion(
            type="type",
            value="@doc:",
            display="@doc:",
            description="Shorthand for @document"
        ),
        AutocompleteSuggestion(
            type="type",
            value="@tag:",
            display="@tag:",
            description="Filter by collection/tag"
        ),
        AutocompleteSuggestion(
            type="type",
            value="@recent:7d",
            display="@recent:7d",
            description="Last 7 days"
        ),
        AutocompleteSuggestion(
            type="type",
            value="@recent:30d",
            display="@recent:30d",
            description="Last 30 days"
        ),
        AutocompleteSuggestion(
            type="type",
            value="@all",
            display="@all",
            description="Search all documents"
        ),
    ]

    async def get_suggestions(
        self,
        partial: str,
        session,  # AsyncSession
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[AutocompleteSuggestion]:
        """
        Get autocomplete suggestions for a partial @mention.

        Args:
            partial: The partial mention text (e.g., "@fold", "@folder:Mark")
            session: Database session
            organization_id: Organization scope
            user_id: User scope
            limit: Maximum suggestions to return

        Returns:
            List of AutocompleteSuggestion objects
        """
        from sqlalchemy import select
        from backend.db.models import Document, Folder

        suggestions = []

        # Check what type of completion is needed
        if partial == "@" or partial.startswith("@") and ":" not in partial:
            # Suggest mention types
            type_prefix = partial[1:].lower() if len(partial) > 1 else ""
            for suggestion in self.TYPE_SUGGESTIONS:
                if suggestion.value.lower().startswith(f"@{type_prefix}"):
                    suggestions.append(suggestion)
            return suggestions[:limit]

        # Parse the partial mention
        if partial.startswith("@folder:"):
            value_prefix = partial[8:].lower()
            suggestions = await self._get_folder_suggestions(
                session, value_prefix, organization_id, limit
            )
        elif partial.startswith("@document:") or partial.startswith("@doc:"):
            prefix_len = 10 if partial.startswith("@document:") else 5
            value_prefix = partial[prefix_len:].lower()
            suggestions = await self._get_document_suggestions(
                session, value_prefix, organization_id, limit
            )
        elif partial.startswith("@tag:"):
            value_prefix = partial[5:].lower()
            suggestions = await self._get_tag_suggestions(
                session, value_prefix, organization_id, limit
            )
        elif partial.startswith("@recent:"):
            suggestions = [
                AutocompleteSuggestion(type="recent", value="@recent:1d", display="@recent:1d", description="Last 24 hours"),
                AutocompleteSuggestion(type="recent", value="@recent:7d", display="@recent:7d", description="Last 7 days"),
                AutocompleteSuggestion(type="recent", value="@recent:14d", display="@recent:14d", description="Last 2 weeks"),
                AutocompleteSuggestion(type="recent", value="@recent:30d", display="@recent:30d", description="Last 30 days"),
                AutocompleteSuggestion(type="recent", value="@recent:3m", display="@recent:3m", description="Last 3 months"),
            ]

        return suggestions[:limit]

    async def _get_folder_suggestions(
        self,
        session,
        prefix: str,
        organization_id: Optional[str],
        limit: int,
    ) -> List[AutocompleteSuggestion]:
        """Get folder name suggestions."""
        from sqlalchemy import select
        from backend.db.models import Folder

        query = select(Folder).where(
            Folder.name.ilike(f"%{prefix}%")
        ).limit(limit)

        if organization_id:
            try:
                org_uuid = UUID(organization_id)
                query = query.where(Folder.organization_id == org_uuid)
            except ValueError:
                pass

        result = await session.execute(query)
        folders = result.scalars().all()

        return [
            AutocompleteSuggestion(
                type="folder",
                value=f"@folder:{folder.name}",
                display=folder.name,
                description=f"Folder • {folder.document_count or 0} documents"
            )
            for folder in folders
        ]

    async def _get_document_suggestions(
        self,
        session,
        prefix: str,
        organization_id: Optional[str],
        limit: int,
    ) -> List[AutocompleteSuggestion]:
        """Get document name suggestions."""
        from sqlalchemy import select
        from backend.db.models import Document

        query = select(Document).where(
            Document.file_name.ilike(f"%{prefix}%")
        ).limit(limit)

        if organization_id:
            try:
                org_uuid = UUID(organization_id)
                query = query.where(Document.organization_id == org_uuid)
            except ValueError:
                pass

        result = await session.execute(query)
        documents = result.scalars().all()

        return [
            AutocompleteSuggestion(
                type="document",
                value=f"@document:{doc.file_name}",
                display=doc.file_name,
                description=f"{doc.collection or 'No tag'}"
            )
            for doc in documents
        ]

    async def _get_tag_suggestions(
        self,
        session,
        prefix: str,
        organization_id: Optional[str],
        limit: int,
    ) -> List[AutocompleteSuggestion]:
        """Get tag/collection suggestions."""
        from sqlalchemy import select, func
        from backend.db.models import Document

        # Get distinct collections
        query = select(Document.collection, func.count(Document.id)).where(
            Document.collection.isnot(None),
            Document.collection.ilike(f"%{prefix}%")
        ).group_by(Document.collection).limit(limit)

        if organization_id:
            try:
                org_uuid = UUID(organization_id)
                query = query.where(Document.organization_id == org_uuid)
            except ValueError:
                pass

        result = await session.execute(query)
        tags = result.all()

        return [
            AutocompleteSuggestion(
                type="tag",
                value=f"@tag:{tag[0]}",
                display=tag[0],
                description=f"{tag[1]} documents"
            )
            for tag in tags if tag[0]
        ]


# =============================================================================
# Singleton Instances
# =============================================================================

_mention_parser: Optional[MentionParser] = None
_mention_autocomplete: Optional[MentionAutocomplete] = None


def get_mention_parser() -> MentionParser:
    """Get or create mention parser singleton."""
    global _mention_parser
    if _mention_parser is None:
        _mention_parser = MentionParser()
    return _mention_parser


def get_mention_autocomplete() -> MentionAutocomplete:
    """Get or create mention autocomplete singleton."""
    global _mention_autocomplete
    if _mention_autocomplete is None:
        _mention_autocomplete = MentionAutocomplete()
    return _mention_autocomplete
