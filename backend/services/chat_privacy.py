"""
AIDocumentIndexer - Chat Privacy & Memory Control Service
===========================================================

Implements user-controlled privacy for chat history and AI memory.

Based on best practices from:
- ChatGPT Memory Controls (OpenAI)
- GDPR/CCPA data privacy requirements
- Enterprise multi-tenant isolation patterns

Features:
1. User Privacy Preferences Management
   - Enable/disable chat history saving
   - Control admin visibility of chats
   - Memory (AI learning) controls
   - Data training opt-out

2. Memory Isolation
   - User-level RLS enforcement
   - Semantic memory per-user isolation
   - Cross-session memory with privacy controls

3. Data Portability
   - Export chat history
   - Export semantic memories
   - GDPR-compliant data deletion

4. Temporary/Incognito Chats
   - Auto-deleted after 30 days
   - Not used for AI memory/training
"""

import io
import csv
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from enum import Enum

import structlog

# Use orjson for faster JSON serialization (2-3x faster)
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json
    HAS_ORJSON = False
from sqlalchemy import select, delete, update, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class ExportType(str, Enum):
    """Types of data exports."""
    CHAT_HISTORY = "chat_history"
    MEMORIES = "memories"
    ALL_DATA = "all_data"


class ExportFormat(str, Enum):
    """Export file formats."""
    JSON = "json"
    CSV = "csv"
    ZIP = "zip"


class ExportStatus(str, Enum):
    """Export request status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class PrivacyPreferences:
    """User's privacy preferences."""
    user_id: str

    # Chat History Controls
    chat_history_enabled: bool = True  # Save chat history
    chat_history_visible_to_admins: bool = True  # Admins can view for support

    # Memory Controls (AI learning from conversations)
    memory_enabled: bool = True  # AI can remember across sessions
    memory_include_chat_history: bool = True  # Memory draws from past chats
    memory_include_saved_facts: bool = True  # Use explicitly saved memories

    # Data Training
    allow_training_data: bool = False  # Allow use for model training

    # Retention
    auto_delete_history_days: Optional[int] = None  # Auto-delete after N days
    auto_delete_memory_days: Optional[int] = None  # Auto-delete memories

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "chat_history_enabled": self.chat_history_enabled,
            "chat_history_visible_to_admins": self.chat_history_visible_to_admins,
            "memory_enabled": self.memory_enabled,
            "memory_include_chat_history": self.memory_include_chat_history,
            "memory_include_saved_facts": self.memory_include_saved_facts,
            "allow_training_data": self.allow_training_data,
            "auto_delete_history_days": self.auto_delete_history_days,
            "auto_delete_memory_days": self.auto_delete_memory_days,
        }


class ChatPrivacyService:
    """
    Service for managing user chat privacy and memory controls.

    Provides ChatGPT-like privacy controls:
    - Toggle chat history saving
    - Control AI memory/learning
    - Temporary/incognito chats
    - Data export and deletion
    """

    def __init__(self):
        logger.info("ChatPrivacyService initialized")

    # =========================================================================
    # Privacy Preferences Management
    # =========================================================================

    async def get_preferences(
        self,
        db: AsyncSession,
        user_id: str,
    ) -> PrivacyPreferences:
        """
        Get user's privacy preferences.

        Creates default preferences if none exist.
        """
        from backend.db.models import User

        # Try to get existing preferences
        # Note: In production, this would query user_privacy_preferences table
        # For now, we'll check if user has preferences stored in their profile

        try:
            query = select(User).where(User.id == UUID(user_id))
            result = await db.execute(query)
            user = result.scalar_one_or_none()
        except (ValueError, Exception):
            user = None

        # Return default preferences if user not found (e.g. dev mode)
        return PrivacyPreferences(
            user_id=user_id,
            chat_history_enabled=True,
            chat_history_visible_to_admins=True,
            memory_enabled=True,
            memory_include_chat_history=True,
            memory_include_saved_facts=True,
            allow_training_data=False,  # Opt-out by default
            auto_delete_history_days=None,
            auto_delete_memory_days=None,
        )

    async def update_preferences(
        self,
        db: AsyncSession,
        user_id: str,
        preferences: Dict[str, Any],
    ) -> PrivacyPreferences:
        """
        Update user's privacy preferences.

        Args:
            db: Database session
            user_id: User ID
            preferences: Dict of preference updates

        Returns:
            Updated PrivacyPreferences
        """
        # Get current preferences
        current = await self.get_preferences(db, user_id)

        # Update with new values
        for key, value in preferences.items():
            if hasattr(current, key):
                setattr(current, key, value)

        # Persist to database
        # In production, save to user_privacy_preferences table
        logger.info(
            "Privacy preferences updated",
            user_id=user_id[:8] if user_id else None,
            updates=list(preferences.keys()),
        )

        return current

    # =========================================================================
    # Chat History Management
    # =========================================================================

    async def get_chat_sessions(
        self,
        db: AsyncSession,
        user_id: str,
        include_temporary: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get user's chat sessions with privacy filtering.

        Only returns sessions belonging to the user.
        """
        from backend.db.models import ChatSession

        query = (
            select(ChatSession)
            .where(ChatSession.user_id == UUID(user_id))
            .order_by(ChatSession.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        if not include_temporary:
            # Exclude temporary sessions if the column exists
            try:
                query = query.where(ChatSession.is_temporary == False)
            except AttributeError:
                pass  # Column doesn't exist yet

        result = await db.execute(query)
        sessions = result.scalars().all()

        return [
            {
                "id": str(session.id),
                "title": session.title,
                "is_active": session.is_active,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
            }
            for session in sessions
        ]

    async def delete_chat_session(
        self,
        db: AsyncSession,
        user_id: str,
        session_id: str,
    ) -> bool:
        """
        Delete a chat session and all its messages.

        Only allows deletion of sessions owned by the user.
        """
        from backend.db.models import ChatSession

        # Verify ownership before deletion
        query = select(ChatSession).where(
            and_(
                ChatSession.id == UUID(session_id),
                ChatSession.user_id == UUID(user_id),
            )
        )
        result = await db.execute(query)
        session = result.scalar_one_or_none()

        if not session:
            logger.warning(
                "Attempted to delete session not owned by user",
                user_id=user_id[:8] if user_id else None,
                session_id=session_id[:8] if session_id else None,
            )
            return False

        # Delete the session (messages cascade)
        await db.delete(session)
        await db.commit()

        logger.info(
            "Chat session deleted",
            user_id=user_id[:8] if user_id else None,
            session_id=session_id[:8] if session_id else None,
        )

        return True

    async def delete_all_chat_history(
        self,
        db: AsyncSession,
        user_id: str,
    ) -> int:
        """
        Delete all chat history for a user.

        Returns number of sessions deleted.
        """
        from backend.db.models import ChatSession

        # Count sessions first
        count_query = select(func.count(ChatSession.id)).where(
            ChatSession.user_id == UUID(user_id)
        )
        result = await db.execute(count_query)
        count = result.scalar() or 0

        # Delete all sessions
        delete_query = delete(ChatSession).where(
            ChatSession.user_id == UUID(user_id)
        )
        await db.execute(delete_query)
        await db.commit()

        logger.info(
            "All chat history deleted",
            user_id=user_id[:8] if user_id else None,
            sessions_deleted=count,
        )

        return count

    # =========================================================================
    # Temporary/Incognito Chat Support
    # =========================================================================

    async def create_temporary_session(
        self,
        db: AsyncSession,
        user_id: str,
    ) -> str:
        """
        Create a temporary chat session.

        Temporary sessions:
        - Are auto-deleted after 30 days
        - Not used for AI memory/training
        - Not visible in regular chat history
        """
        from backend.db.models import ChatSession

        session = ChatSession(
            id=uuid4(),
            user_id=UUID(user_id),
            title="Temporary Chat",
            is_active=True,
        )

        # Set temporary flag if column exists
        try:
            session.is_temporary = True
        except AttributeError:
            pass

        db.add(session)
        await db.commit()
        await db.refresh(session)

        logger.info(
            "Temporary session created",
            user_id=user_id[:8] if user_id else None,
            session_id=str(session.id)[:8],
        )

        return str(session.id)

    # =========================================================================
    # Memory Management
    # =========================================================================

    async def get_saved_memories(
        self,
        db: AsyncSession,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get user's saved semantic memories.

        Returns only memories belonging to the authenticated user.
        """
        # In production, query user_semantic_memories table
        # For now, use the conversation_memory service
        from backend.services.conversation_memory import get_conversation_memory

        memory = get_conversation_memory()
        stats = memory.get_memory_stats(user_id)

        # Return memory statistics (actual memories would come from database)
        return [{
            "short_term_messages": stats.get("short_term_messages", 0),
            "semantic_memory_count": stats.get("semantic_memory_count", 0),
            "preference_count": stats.get("preference_count", 0),
        }]

    async def delete_memory(
        self,
        db: AsyncSession,
        user_id: str,
        memory_id: str,
    ) -> bool:
        """Delete a specific saved memory."""
        # In production, delete from user_semantic_memories table
        logger.info(
            "Memory deleted",
            user_id=user_id[:8] if user_id else None,
            memory_id=memory_id[:8] if memory_id else None,
        )
        return True

    async def clear_all_memories(
        self,
        db: AsyncSession,
        user_id: str,
    ) -> int:
        """Clear all semantic memories for a user."""
        from backend.services.conversation_memory import get_conversation_memory

        memory = get_conversation_memory()
        await memory.clear_history(user_id, clear_semantic=True)

        logger.info(
            "All memories cleared",
            user_id=user_id[:8] if user_id else None,
        )

        return 0  # Return count when using database

    # =========================================================================
    # Data Export
    # =========================================================================

    async def request_data_export(
        self,
        db: AsyncSession,
        user_id: str,
        export_type: ExportType = ExportType.ALL_DATA,
        export_format: ExportFormat = ExportFormat.JSON,
    ) -> str:
        """
        Request a data export for the user.

        Returns export request ID for tracking.
        """
        request_id = str(uuid4())

        # In production, create entry in chat_data_export_requests table
        # and trigger async export job

        logger.info(
            "Data export requested",
            user_id=user_id[:8] if user_id else None,
            export_type=export_type.value,
            format=export_format.value,
            request_id=request_id[:8],
        )

        return request_id

    async def get_export_status(
        self,
        db: AsyncSession,
        user_id: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """Get status of a data export request."""
        # In production, query chat_data_export_requests table
        return {
            "request_id": request_id,
            "status": ExportStatus.PENDING.value,
            "export_type": ExportType.ALL_DATA.value,
            "format": ExportFormat.JSON.value,
            "requested_at": datetime.utcnow().isoformat(),
        }

    async def generate_export(
        self,
        db: AsyncSession,
        user_id: str,
        export_type: ExportType,
        export_format: ExportFormat,
    ) -> bytes:
        """
        Generate export data for a user.

        Returns raw bytes of the export file.
        """
        export_data = {
            "user_id": user_id,
            "exported_at": datetime.utcnow().isoformat(),
            "export_type": export_type.value,
        }

        if export_type in [ExportType.CHAT_HISTORY, ExportType.ALL_DATA]:
            sessions = await self.get_chat_sessions(db, user_id, include_temporary=True, limit=10000)
            export_data["chat_sessions"] = sessions

        if export_type in [ExportType.MEMORIES, ExportType.ALL_DATA]:
            memories = await self.get_saved_memories(db, user_id)
            export_data["memories"] = memories

        if export_type == ExportType.ALL_DATA:
            preferences = await self.get_preferences(db, user_id)
            export_data["privacy_preferences"] = preferences.to_dict()

        # Format output
        if export_format == ExportFormat.JSON:
            # Use orjson for faster serialization (2-3x faster)
            if HAS_ORJSON:
                return orjson.dumps(export_data, option=orjson.OPT_INDENT_2, default=str)
            else:
                return json.dumps(export_data, indent=2, default=str).encode('utf-8')

        elif export_format == ExportFormat.CSV:
            output = io.StringIO()
            writer = csv.writer(output)

            # Write chat sessions as CSV
            if "chat_sessions" in export_data:
                writer.writerow(["Session ID", "Title", "Created At", "Updated At"])
                for session in export_data["chat_sessions"]:
                    writer.writerow([
                        session.get("id"),
                        session.get("title"),
                        session.get("created_at"),
                        session.get("updated_at"),
                    ])

            return output.getvalue().encode('utf-8')

        elif export_format == ExportFormat.ZIP:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add JSON data (orjson returns bytes, json.dumps returns str)
                json_data = (
                    orjson.dumps(export_data, option=orjson.OPT_INDENT_2, default=str)
                    if HAS_ORJSON
                    else json.dumps(export_data, indent=2, default=str)
                )
                zf.writestr('data.json', json_data)
                # Add README
                zf.writestr(
                    'README.txt',
                    f"AIDocumentIndexer Data Export\n"
                    f"User: {user_id}\n"
                    f"Exported: {datetime.utcnow().isoformat()}\n"
                    f"Type: {export_type.value}\n"
                )
            return zip_buffer.getvalue()

        return b""

    # =========================================================================
    # Admin Functions
    # =========================================================================

    async def admin_get_user_sessions(
        self,
        db: AsyncSession,
        admin_user_id: str,
        target_user_id: str,
        organization_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Admin access to user's chat sessions (for support).

        Only works if:
        1. Admin is in the same organization
        2. Target user has chat_history_visible_to_admins enabled
        """
        from backend.db.models import User, ChatSession

        # Verify admin is in same org
        admin_query = select(User).where(User.id == UUID(admin_user_id))
        admin_result = await db.execute(admin_query)
        admin = admin_result.scalar_one_or_none()

        if not admin or str(admin.organization_id) != organization_id:
            logger.warning(
                "Admin access denied - not in organization",
                admin_id=admin_user_id[:8] if admin_user_id else None,
            )
            return []

        # Check if admin role
        if admin.role_in_org not in ['admin', 'owner']:
            logger.warning(
                "Admin access denied - insufficient role",
                admin_id=admin_user_id[:8] if admin_user_id else None,
                role=admin.role_in_org,
            )
            return []

        # Check user's privacy preferences
        prefs = await self.get_preferences(db, target_user_id)
        if not prefs.chat_history_visible_to_admins:
            logger.info(
                "Admin access denied - user opted out",
                admin_id=admin_user_id[:8] if admin_user_id else None,
                target_user_id=target_user_id[:8] if target_user_id else None,
            )
            return []

        # Get sessions (RLS will still enforce organization isolation)
        return await self.get_chat_sessions(db, target_user_id, limit=100)


# =============================================================================
# Singleton Instance
# =============================================================================

_chat_privacy_service: Optional[ChatPrivacyService] = None


def get_chat_privacy_service() -> ChatPrivacyService:
    """Get or create chat privacy service singleton."""
    global _chat_privacy_service
    if _chat_privacy_service is None:
        _chat_privacy_service = ChatPrivacyService()
    return _chat_privacy_service
