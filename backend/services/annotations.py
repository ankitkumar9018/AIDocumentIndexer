"""
AIDocumentIndexer - Collaborative Annotations Service
======================================================

Real-time collaborative annotation system for team document markup:
1. Highlights - Mark important passages
2. Comments - Add discussions
3. Questions - Flag unclear sections
4. Corrections - Suggest fixes
5. Links - Connect related content

Supports real-time updates via WebSocket broadcast.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class AnnotationType(str, Enum):
    """Types of annotations supported."""
    HIGHLIGHT = "highlight"
    COMMENT = "comment"
    QUESTION = "question"
    CORRECTION = "correction"
    LINK = "link"


class AnnotationStatus(str, Enum):
    """Status of an annotation."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass
class AnnotationReply:
    """A reply to an annotation."""
    id: str
    annotation_id: str
    user_id: str
    user_name: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "annotation_id": self.annotation_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationReply":
        return cls(
            id=data["id"],
            annotation_id=data["annotation_id"],
            user_id=data["user_id"],
            user_name=data.get("user_name", "Unknown"),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
        )


@dataclass
class Annotation:
    """A document annotation with position and content."""
    id: str
    document_id: str
    user_id: str
    user_name: str
    annotation_type: AnnotationType
    content: str
    selected_text: str
    start_offset: int
    end_offset: int
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    status: AnnotationStatus = AnnotationStatus.ACTIVE
    replies: List[AnnotationReply] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    color: Optional[str] = None  # For highlights
    linked_annotation_id: Optional[str] = None  # For link type
    linked_document_id: Optional[str] = None  # For link type
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "annotation_type": self.annotation_type.value,
            "content": self.content,
            "selected_text": self.selected_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "status": self.status.value,
            "replies": [r.to_dict() for r in self.replies],
            "tags": self.tags,
            "color": self.color,
            "linked_annotation_id": self.linked_annotation_id,
            "linked_document_id": self.linked_document_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        return cls(
            id=data["id"],
            document_id=data["document_id"],
            user_id=data["user_id"],
            user_name=data.get("user_name", "Unknown"),
            annotation_type=AnnotationType(data["annotation_type"]),
            content=data["content"],
            selected_text=data.get("selected_text", ""),
            start_offset=data.get("start_offset", 0),
            end_offset=data.get("end_offset", 0),
            chunk_id=data.get("chunk_id"),
            page_number=data.get("page_number"),
            status=AnnotationStatus(data.get("status", "active")),
            replies=[AnnotationReply.from_dict(r) for r in data.get("replies", [])],
            tags=data.get("tags", []),
            color=data.get("color"),
            linked_annotation_id=data.get("linked_annotation_id"),
            linked_document_id=data.get("linked_document_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else datetime.utcnow(),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolved_by=data.get("resolved_by"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AnnotationSummary:
    """Summary of annotations for a document."""
    document_id: str
    total_count: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_user: Dict[str, int]
    recent_activity: List[Dict[str, Any]]


class AnnotationService:
    """
    Manage collaborative document annotations.

    Provides:
    - CRUD operations for annotations
    - Real-time WebSocket notifications
    - Thread-based discussions via replies
    - Search and filtering
    """

    def __init__(
        self,
        cache=None,
        websocket_manager=None,
        annotation_ttl_days: int = 365,
    ):
        """
        Initialize annotation service.

        Args:
            cache: RedisCache or similar for persistence
            websocket_manager: WebSocket manager for real-time updates
            annotation_ttl_days: Days to keep annotations
        """
        self.cache = cache
        self.ws_manager = websocket_manager
        self.annotation_ttl = annotation_ttl_days * 86400

        # In-memory storage (fallback)
        self._annotations: Dict[str, Annotation] = {}
        self._document_index: Dict[str, List[str]] = {}  # document_id -> annotation_ids

    async def create_annotation(
        self,
        document_id: str,
        user_id: str,
        user_name: str,
        annotation_type: AnnotationType,
        content: str,
        selected_text: str,
        start_offset: int,
        end_offset: int,
        chunk_id: Optional[str] = None,
        page_number: Optional[int] = None,
        tags: Optional[List[str]] = None,
        color: Optional[str] = None,
        linked_annotation_id: Optional[str] = None,
        linked_document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Annotation:
        """
        Create a new annotation.

        Args:
            document_id: Document being annotated
            user_id: User creating annotation
            user_name: Display name of user
            annotation_type: Type of annotation
            content: Annotation content/comment
            selected_text: Text that was selected
            start_offset: Start position in document
            end_offset: End position in document
            chunk_id: Optional chunk reference
            page_number: Optional page number
            tags: Optional tags for categorization
            color: Optional highlight color
            linked_annotation_id: For link type - linked annotation
            linked_document_id: For link type - linked document
            metadata: Additional metadata

        Returns:
            Created Annotation
        """
        annotation = Annotation(
            id=str(uuid4()),
            document_id=document_id,
            user_id=user_id,
            user_name=user_name,
            annotation_type=annotation_type,
            content=content,
            selected_text=selected_text,
            start_offset=start_offset,
            end_offset=end_offset,
            chunk_id=chunk_id,
            page_number=page_number,
            tags=tags or [],
            color=color,
            linked_annotation_id=linked_annotation_id,
            linked_document_id=linked_document_id,
            metadata=metadata or {},
        )

        # Store annotation
        await self._save_annotation(annotation)

        # Update document index
        if document_id not in self._document_index:
            self._document_index[document_id] = []
        self._document_index[document_id].append(annotation.id)

        # Broadcast to connected users
        await self._broadcast_event(
            document_id,
            "annotation_created",
            annotation.to_dict(),
        )

        logger.info(
            "Annotation created",
            annotation_id=annotation.id,
            document_id=document_id,
            user_id=user_id[:8],
            type=annotation_type.value,
        )

        return annotation

    async def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """Get annotation by ID."""
        # Check memory
        if annotation_id in self._annotations:
            return self._annotations[annotation_id]

        # Check persistent storage
        if self.cache:
            try:
                data = await self.cache.get(f"annotation:{annotation_id}")
                if data:
                    parsed = json.loads(data) if isinstance(data, str) else data
                    annotation = Annotation.from_dict(parsed)
                    self._annotations[annotation_id] = annotation
                    return annotation
            except Exception as e:
                logger.warning("Failed to load annotation", error=str(e))

        return None

    async def update_annotation(
        self,
        annotation_id: str,
        user_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        color: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Annotation]:
        """
        Update an annotation.

        Args:
            annotation_id: Annotation to update
            user_id: User making update
            content: New content (if updating)
            tags: New tags (if updating)
            color: New color (if updating)
            metadata: New metadata (if updating)

        Returns:
            Updated Annotation or None if not found
        """
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return None

        # Update fields
        if content is not None:
            annotation.content = content
        if tags is not None:
            annotation.tags = tags
        if color is not None:
            annotation.color = color
        if metadata is not None:
            annotation.metadata = {**annotation.metadata, **metadata}

        annotation.updated_at = datetime.utcnow()

        # Save
        await self._save_annotation(annotation)

        # Broadcast update
        await self._broadcast_event(
            annotation.document_id,
            "annotation_updated",
            annotation.to_dict(),
        )

        logger.info(
            "Annotation updated",
            annotation_id=annotation_id,
            user_id=user_id[:8],
        )

        return annotation

    async def delete_annotation(
        self,
        annotation_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete an annotation.

        Args:
            annotation_id: Annotation to delete
            user_id: User deleting (for audit)

        Returns:
            True if deleted, False if not found
        """
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return False

        document_id = annotation.document_id

        # Remove from storage
        self._annotations.pop(annotation_id, None)

        if document_id in self._document_index:
            self._document_index[document_id] = [
                aid for aid in self._document_index[document_id]
                if aid != annotation_id
            ]

        if self.cache:
            await self.cache.delete(f"annotation:{annotation_id}")

        # Broadcast deletion
        await self._broadcast_event(
            document_id,
            "annotation_deleted",
            {"annotation_id": annotation_id},
        )

        logger.info(
            "Annotation deleted",
            annotation_id=annotation_id,
            user_id=user_id[:8],
        )

        return True

    async def add_reply(
        self,
        annotation_id: str,
        user_id: str,
        user_name: str,
        content: str,
    ) -> Optional[AnnotationReply]:
        """
        Add a reply to an annotation.

        Args:
            annotation_id: Annotation to reply to
            user_id: User replying
            user_name: Display name
            content: Reply content

        Returns:
            Created AnnotationReply or None if annotation not found
        """
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return None

        reply = AnnotationReply(
            id=str(uuid4()),
            annotation_id=annotation_id,
            user_id=user_id,
            user_name=user_name,
            content=content,
        )

        annotation.replies.append(reply)
        annotation.updated_at = datetime.utcnow()

        await self._save_annotation(annotation)

        # Broadcast reply
        await self._broadcast_event(
            annotation.document_id,
            "reply_added",
            {
                "annotation_id": annotation_id,
                "reply": reply.to_dict(),
            },
        )

        logger.info(
            "Reply added",
            annotation_id=annotation_id,
            reply_id=reply.id,
            user_id=user_id[:8],
        )

        return reply

    async def resolve_annotation(
        self,
        annotation_id: str,
        user_id: str,
    ) -> Optional[Annotation]:
        """
        Mark an annotation as resolved.

        Args:
            annotation_id: Annotation to resolve
            user_id: User resolving

        Returns:
            Updated Annotation or None if not found
        """
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return None

        annotation.status = AnnotationStatus.RESOLVED
        annotation.resolved_at = datetime.utcnow()
        annotation.resolved_by = user_id
        annotation.updated_at = datetime.utcnow()

        await self._save_annotation(annotation)

        # Broadcast resolution
        await self._broadcast_event(
            annotation.document_id,
            "annotation_resolved",
            {
                "annotation_id": annotation_id,
                "resolved_by": user_id,
            },
        )

        logger.info(
            "Annotation resolved",
            annotation_id=annotation_id,
            user_id=user_id[:8],
        )

        return annotation

    async def reopen_annotation(
        self,
        annotation_id: str,
        user_id: str,
    ) -> Optional[Annotation]:
        """Reopen a resolved annotation."""
        annotation = await self.get_annotation(annotation_id)
        if not annotation:
            return None

        annotation.status = AnnotationStatus.ACTIVE
        annotation.resolved_at = None
        annotation.resolved_by = None
        annotation.updated_at = datetime.utcnow()

        await self._save_annotation(annotation)

        await self._broadcast_event(
            annotation.document_id,
            "annotation_reopened",
            {"annotation_id": annotation_id},
        )

        return annotation

    async def get_document_annotations(
        self,
        document_id: str,
        include_resolved: bool = False,
        annotation_type: Optional[AnnotationType] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Annotation]:
        """
        Get all annotations for a document.

        Args:
            document_id: Document to get annotations for
            include_resolved: Include resolved annotations
            annotation_type: Filter by type
            user_id: Filter by user
            tags: Filter by tags (any match)

        Returns:
            List of Annotations
        """
        annotation_ids = self._document_index.get(document_id, [])

        # Load from cache if needed
        if self.cache and not annotation_ids:
            try:
                data = await self.cache.get(f"doc_annotations:{document_id}")
                if data:
                    annotation_ids = json.loads(data) if isinstance(data, str) else data
                    self._document_index[document_id] = annotation_ids
            except Exception as e:
                logger.warning("Failed to load document annotations", error=str(e))

        annotations = []
        for aid in annotation_ids:
            annotation = await self.get_annotation(aid)
            if annotation:
                annotations.append(annotation)

        # Apply filters
        if not include_resolved:
            annotations = [a for a in annotations if a.status == AnnotationStatus.ACTIVE]

        if annotation_type:
            annotations = [a for a in annotations if a.annotation_type == annotation_type]

        if user_id:
            annotations = [a for a in annotations if a.user_id == user_id]

        if tags:
            annotations = [a for a in annotations if any(t in a.tags for t in tags)]

        # Sort by position
        annotations.sort(key=lambda a: a.start_offset)

        return annotations

    async def get_annotation_summary(self, document_id: str) -> AnnotationSummary:
        """Get summary statistics for document annotations."""
        annotations = await self.get_document_annotations(document_id, include_resolved=True)

        by_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_user: Dict[str, int] = {}

        for a in annotations:
            by_type[a.annotation_type.value] = by_type.get(a.annotation_type.value, 0) + 1
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
            by_user[a.user_name] = by_user.get(a.user_name, 0) + 1

        # Recent activity (last 10 updates)
        sorted_annotations = sorted(annotations, key=lambda a: a.updated_at, reverse=True)
        recent_activity = [
            {
                "annotation_id": a.id,
                "type": a.annotation_type.value,
                "user": a.user_name,
                "updated_at": a.updated_at.isoformat(),
                "preview": a.content[:50] + "..." if len(a.content) > 50 else a.content,
            }
            for a in sorted_annotations[:10]
        ]

        return AnnotationSummary(
            document_id=document_id,
            total_count=len(annotations),
            by_type=by_type,
            by_status=by_status,
            by_user=by_user,
            recent_activity=recent_activity,
        )

    async def search_annotations(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        annotation_types: Optional[List[AnnotationType]] = None,
        limit: int = 50,
    ) -> List[Annotation]:
        """
        Search across annotations.

        Args:
            query: Search query (searches content and selected_text)
            document_ids: Limit to specific documents
            user_id: Limit to specific user
            annotation_types: Limit to specific types
            limit: Max results

        Returns:
            Matching Annotations
        """
        query_lower = query.lower()
        results = []

        # Get all annotations to search
        all_annotation_ids = set()
        if document_ids:
            for doc_id in document_ids:
                all_annotation_ids.update(self._document_index.get(doc_id, []))
        else:
            for ids in self._document_index.values():
                all_annotation_ids.update(ids)

        for aid in all_annotation_ids:
            annotation = await self.get_annotation(aid)
            if not annotation:
                continue

            # Apply filters
            if user_id and annotation.user_id != user_id:
                continue

            if annotation_types and annotation.annotation_type not in annotation_types:
                continue

            # Search in content and selected text
            if query_lower in annotation.content.lower() or query_lower in annotation.selected_text.lower():
                results.append(annotation)

            # Also search in replies
            elif any(query_lower in r.content.lower() for r in annotation.replies):
                results.append(annotation)

            if len(results) >= limit:
                break

        return results

    async def get_user_annotations(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Annotation]:
        """Get all annotations by a user."""
        results = []

        for annotation_ids in self._document_index.values():
            for aid in annotation_ids:
                annotation = await self.get_annotation(aid)
                if annotation and annotation.user_id == user_id:
                    results.append(annotation)
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break

        # Sort by creation date descending
        results.sort(key=lambda a: a.created_at, reverse=True)
        return results

    async def _save_annotation(self, annotation: Annotation) -> None:
        """Save annotation to storage."""
        self._annotations[annotation.id] = annotation

        if self.cache:
            try:
                await self.cache.set(
                    f"annotation:{annotation.id}",
                    json.dumps(annotation.to_dict()),
                    ttl=self.annotation_ttl,
                )

                # Update document index in cache
                doc_annotations = self._document_index.get(annotation.document_id, [])
                if annotation.id not in doc_annotations:
                    doc_annotations.append(annotation.id)
                    self._document_index[annotation.document_id] = doc_annotations

                await self.cache.set(
                    f"doc_annotations:{annotation.document_id}",
                    json.dumps(doc_annotations),
                    ttl=self.annotation_ttl,
                )
            except Exception as e:
                logger.warning("Failed to save annotation to cache", error=str(e))

    async def _broadcast_event(
        self,
        document_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Broadcast event to connected users."""
        if self.ws_manager:
            try:
                await self.ws_manager.broadcast_to_document(
                    document_id,
                    {
                        "type": event_type,
                        "document_id": document_id,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception as e:
                logger.warning("WebSocket broadcast failed", error=str(e))


# =============================================================================
# Convenience Functions
# =============================================================================

_service_instance: Optional[AnnotationService] = None


def get_annotation_service(
    cache=None,
    websocket_manager=None,
) -> AnnotationService:
    """
    Get or create the annotation service singleton.

    Args:
        cache: Optional cache for persistence
        websocket_manager: Optional WebSocket manager

    Returns:
        AnnotationService instance
    """
    global _service_instance

    if _service_instance is None:
        _service_instance = AnnotationService(
            cache=cache,
            websocket_manager=websocket_manager,
        )

    return _service_instance


async def create_highlight(
    document_id: str,
    user_id: str,
    user_name: str,
    selected_text: str,
    start_offset: int,
    end_offset: int,
    color: str = "#FFFF00",
    note: Optional[str] = None,
) -> Annotation:
    """
    Convenience function to create a highlight annotation.

    Args:
        document_id: Document ID
        user_id: User ID
        user_name: User display name
        selected_text: Text being highlighted
        start_offset: Start position
        end_offset: End position
        color: Highlight color (default yellow)
        note: Optional note

    Returns:
        Created Annotation
    """
    service = get_annotation_service()
    return await service.create_annotation(
        document_id=document_id,
        user_id=user_id,
        user_name=user_name,
        annotation_type=AnnotationType.HIGHLIGHT,
        content=note or "",
        selected_text=selected_text,
        start_offset=start_offset,
        end_offset=end_offset,
        color=color,
    )


async def create_comment(
    document_id: str,
    user_id: str,
    user_name: str,
    selected_text: str,
    start_offset: int,
    end_offset: int,
    comment: str,
) -> Annotation:
    """
    Convenience function to create a comment annotation.

    Args:
        document_id: Document ID
        user_id: User ID
        user_name: User display name
        selected_text: Text being commented on
        start_offset: Start position
        end_offset: End position
        comment: Comment text

    Returns:
        Created Annotation
    """
    service = get_annotation_service()
    return await service.create_annotation(
        document_id=document_id,
        user_id=user_id,
        user_name=user_name,
        annotation_type=AnnotationType.COMMENT,
        content=comment,
        selected_text=selected_text,
        start_offset=start_offset,
        end_offset=end_offset,
    )
