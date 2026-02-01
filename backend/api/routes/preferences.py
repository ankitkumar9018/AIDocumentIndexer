"""
AIDocumentIndexer - User Preferences API Routes
================================================

Endpoints for managing user UI preferences and settings.
"""

from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.api.middleware.auth import get_current_user
from backend.db.database import get_async_session, async_session_context
from backend.db.models import UserPreferences

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class UserPreferencesResponse(BaseModel):
    """User preferences response."""
    # UI Theme
    theme: str = "system"

    # Document List View
    documents_view_mode: str = "grid"
    documents_sort_by: str = "created_at"
    documents_sort_order: str = "desc"
    documents_page_size: int = 20

    # Default Filters
    default_collection: Optional[str] = None
    default_folder_id: Optional[str] = None

    # Search Preferences
    search_include_content: bool = True
    search_results_per_page: int = 10

    # Chat/RAG Preferences
    chat_show_sources: bool = True
    chat_expand_sources: bool = False

    # Sidebar State
    sidebar_collapsed: bool = False

    # Recent Items
    recent_documents: Optional[List[str]] = None
    recent_searches: Optional[List[str]] = None


class UpdatePreferencesRequest(BaseModel):
    """Request to update user preferences."""
    # All fields optional - only update provided values
    theme: Optional[str] = Field(None, pattern="^(light|dark|system)$")
    documents_view_mode: Optional[str] = Field(None, pattern="^(grid|list|table)$")
    documents_sort_by: Optional[str] = Field(None, pattern="^(created_at|name|file_size|updated_at)$")
    documents_sort_order: Optional[str] = Field(None, pattern="^(asc|desc)$")
    documents_page_size: Optional[int] = Field(None, ge=5, le=100)
    default_collection: Optional[str] = Field(None, max_length=200)
    default_folder_id: Optional[str] = None
    search_include_content: Optional[bool] = None
    search_results_per_page: Optional[int] = Field(None, ge=5, le=50)
    chat_show_sources: Optional[bool] = None
    chat_expand_sources: Optional[bool] = None
    sidebar_collapsed: Optional[bool] = None


class AddRecentItemRequest(BaseModel):
    """Request to add a recent item."""
    item_type: str = Field(..., pattern="^(document|search)$")
    item: str = Field(..., min_length=1, max_length=500)


class SavedSearch(BaseModel):
    """A saved search configuration."""
    name: str = Field(..., min_length=1, max_length=100, description="Name for this saved search")
    query: str = Field(..., min_length=1, max_length=1000, description="The search query")
    # Optional filters
    collection: Optional[str] = Field(None, max_length=200)
    folder_id: Optional[str] = None
    include_subfolders: bool = True
    date_from: Optional[str] = None  # ISO date string
    date_to: Optional[str] = None
    file_types: Optional[List[str]] = None  # e.g., ["pdf", "docx"]
    search_mode: str = Field("hybrid", pattern="^(hybrid|vector|keyword)$")


class SavedSearchResponse(BaseModel):
    """Response for a saved search."""
    name: str
    query: str
    collection: Optional[str] = None
    folder_id: Optional[str] = None
    include_subfolders: bool = True
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    file_types: Optional[List[str]] = None
    search_mode: str = "hybrid"
    created_at: str


class SavedSearchesResponse(BaseModel):
    """List of saved searches."""
    searches: List[SavedSearchResponse]
    count: int


# =============================================================================
# Helper Functions
# =============================================================================

def prefs_to_response(prefs: UserPreferences) -> UserPreferencesResponse:
    """Convert UserPreferences model to response."""
    return UserPreferencesResponse(
        theme=prefs.theme,
        documents_view_mode=prefs.documents_view_mode,
        documents_sort_by=prefs.documents_sort_by,
        documents_sort_order=prefs.documents_sort_order,
        documents_page_size=prefs.documents_page_size,
        default_collection=prefs.default_collection,
        default_folder_id=str(prefs.default_folder_id) if prefs.default_folder_id else None,
        search_include_content=prefs.search_include_content,
        search_results_per_page=prefs.search_results_per_page,
        chat_show_sources=prefs.chat_show_sources,
        chat_expand_sources=prefs.chat_expand_sources,
        sidebar_collapsed=prefs.sidebar_collapsed,
        recent_documents=prefs.recent_documents,
        recent_searches=prefs.recent_searches,
    )


async def get_or_create_preferences(
    user_id: str,
    session: AsyncSession,
) -> UserPreferences:
    """Get user preferences, creating default if not exists."""
    result = await session.execute(
        select(UserPreferences).where(UserPreferences.user_id == user_id)
    )
    prefs = result.scalar_one_or_none()

    if not prefs:
        # Create default preferences
        import uuid
        prefs = UserPreferences(
            id=uuid.uuid4(),
            user_id=uuid.UUID(user_id),
        )
        session.add(prefs)
        await session.commit()
        await session.refresh(prefs)
        logger.info("Created default preferences for user", user_id=user_id)

    return prefs


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("", response_model=UserPreferencesResponse)
async def get_preferences(
    user: dict = Depends(get_current_user),
):
    """
    Get current user's preferences.

    Returns default values if no preferences have been saved yet.
    """
    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)
        return prefs_to_response(prefs)


@router.patch("", response_model=UserPreferencesResponse)
async def update_preferences(
    request: UpdatePreferencesRequest,
    user: dict = Depends(get_current_user),
):
    """
    Update user preferences.

    Only updates fields that are provided in the request.
    """
    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)

        # Update only provided fields
        update_data = request.model_dump(exclude_unset=True)

        if "default_folder_id" in update_data:
            # Handle UUID conversion
            folder_id = update_data.pop("default_folder_id")
            if folder_id:
                import uuid
                prefs.default_folder_id = uuid.UUID(folder_id)
            else:
                prefs.default_folder_id = None

        for field, value in update_data.items():
            if hasattr(prefs, field):
                setattr(prefs, field, value)

        await session.commit()
        await session.refresh(prefs)

        logger.info(
            "Updated user preferences",
            user_id=user["sub"],
            updated_fields=list(update_data.keys()),
        )

        return prefs_to_response(prefs)


@router.post("/recent", response_model=UserPreferencesResponse)
async def add_recent_item(
    request: AddRecentItemRequest,
    user: dict = Depends(get_current_user),
):
    """
    Add an item to recent documents or searches.

    Maintains a list of the last 10 items, removing oldest when full.
    """
    MAX_RECENT_ITEMS = 10

    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)

        if request.item_type == "document":
            recent_list = prefs.recent_documents or []
            # Remove if already exists (will re-add at front)
            if request.item in recent_list:
                recent_list.remove(request.item)
            # Add to front
            recent_list.insert(0, request.item)
            # Trim to max size
            prefs.recent_documents = recent_list[:MAX_RECENT_ITEMS]
        else:  # search
            recent_list = prefs.recent_searches or []
            if request.item in recent_list:
                recent_list.remove(request.item)
            recent_list.insert(0, request.item)
            prefs.recent_searches = recent_list[:MAX_RECENT_ITEMS]

        await session.commit()
        await session.refresh(prefs)

        return prefs_to_response(prefs)


@router.delete("/recent/{item_type}")
async def clear_recent_items(
    item_type: str,
    user: dict = Depends(get_current_user),
):
    """
    Clear recent documents or searches.
    """
    if item_type not in ["documents", "searches"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="item_type must be 'documents' or 'searches'",
        )

    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)

        if item_type == "documents":
            prefs.recent_documents = []
        else:
            prefs.recent_searches = []

        await session.commit()

        return {"message": f"Cleared recent {item_type}"}


@router.post("/reset", response_model=UserPreferencesResponse)
async def reset_preferences(
    user: dict = Depends(get_current_user),
):
    """
    Reset all preferences to defaults.
    """
    async with async_session_context() as session:
        result = await session.execute(
            select(UserPreferences).where(UserPreferences.user_id == user["sub"])
        )
        prefs = result.scalar_one_or_none()

        if prefs:
            # Reset to defaults
            prefs.theme = "system"
            prefs.documents_view_mode = "grid"
            prefs.documents_sort_by = "created_at"
            prefs.documents_sort_order = "desc"
            prefs.documents_page_size = 20
            prefs.default_collection = None
            prefs.default_folder_id = None
            prefs.search_include_content = True
            prefs.search_results_per_page = 10
            prefs.chat_show_sources = True
            prefs.chat_expand_sources = False
            prefs.sidebar_collapsed = False
            prefs.recent_documents = []
            prefs.recent_searches = []

            await session.commit()
            await session.refresh(prefs)

            logger.info("Reset user preferences to defaults", user_id=user["sub"])

            return prefs_to_response(prefs)
        else:
            # Create new default preferences
            prefs = await get_or_create_preferences(user["sub"], session)
            return prefs_to_response(prefs)


# =============================================================================
# Saved Searches Endpoints
# =============================================================================

MAX_SAVED_SEARCHES = 20


@router.get("/searches", response_model=SavedSearchesResponse)
async def list_saved_searches(
    user: dict = Depends(get_current_user),
):
    """
    List all saved searches for the current user.
    """
    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)
        saved = prefs.saved_searches or []

        searches = [
            SavedSearchResponse(
                name=s.get("name", ""),
                query=s.get("query", ""),
                collection=s.get("collection"),
                folder_id=s.get("folder_id"),
                include_subfolders=s.get("include_subfolders", True),
                date_from=s.get("date_from"),
                date_to=s.get("date_to"),
                file_types=s.get("file_types"),
                search_mode=s.get("search_mode", "hybrid"),
                created_at=s.get("created_at", ""),
            )
            for s in saved
        ]

        return SavedSearchesResponse(searches=searches, count=len(searches))


@router.post("/searches", response_model=SavedSearchResponse)
async def save_search(
    request: SavedSearch,
    user: dict = Depends(get_current_user),
):
    """
    Save a search configuration.

    If a search with the same name exists, it will be updated.
    Maximum 20 saved searches per user.
    """
    from datetime import datetime

    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)
        saved = prefs.saved_searches or []

        # Check if search with same name exists
        existing_idx = None
        for i, s in enumerate(saved):
            if s.get("name") == request.name:
                existing_idx = i
                break

        # Create search entry
        search_entry = {
            "name": request.name,
            "query": request.query,
            "collection": request.collection,
            "folder_id": request.folder_id,
            "include_subfolders": request.include_subfolders,
            "date_from": request.date_from,
            "date_to": request.date_to,
            "file_types": request.file_types,
            "search_mode": request.search_mode,
            "created_at": datetime.utcnow().isoformat(),
        }

        if existing_idx is not None:
            # Update existing
            saved[existing_idx] = search_entry
            logger.info("Updated saved search", user_id=user["sub"], name=request.name)
        else:
            # Add new
            if len(saved) >= MAX_SAVED_SEARCHES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Maximum {MAX_SAVED_SEARCHES} saved searches allowed. Delete some to add more.",
                )
            saved.append(search_entry)
            logger.info("Created saved search", user_id=user["sub"], name=request.name)

        prefs.saved_searches = saved
        await session.commit()

        return SavedSearchResponse(
            name=search_entry["name"],
            query=search_entry["query"],
            collection=search_entry["collection"],
            folder_id=search_entry["folder_id"],
            include_subfolders=search_entry["include_subfolders"],
            date_from=search_entry["date_from"],
            date_to=search_entry["date_to"],
            file_types=search_entry["file_types"],
            search_mode=search_entry["search_mode"],
            created_at=search_entry["created_at"],
        )


@router.get("/searches/{name}", response_model=SavedSearchResponse)
async def get_saved_search(
    name: str,
    user: dict = Depends(get_current_user),
):
    """
    Get a specific saved search by name.
    """
    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)
        saved = prefs.saved_searches or []

        for s in saved:
            if s.get("name") == name:
                return SavedSearchResponse(
                    name=s.get("name", ""),
                    query=s.get("query", ""),
                    collection=s.get("collection"),
                    folder_id=s.get("folder_id"),
                    include_subfolders=s.get("include_subfolders", True),
                    date_from=s.get("date_from"),
                    date_to=s.get("date_to"),
                    file_types=s.get("file_types"),
                    search_mode=s.get("search_mode", "hybrid"),
                    created_at=s.get("created_at", ""),
                )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved search '{name}' not found",
        )


@router.delete("/searches/{name}")
async def delete_saved_search(
    name: str,
    user: dict = Depends(get_current_user),
):
    """
    Delete a saved search by name.
    """
    async with async_session_context() as session:
        prefs = await get_or_create_preferences(user["sub"], session)
        saved = prefs.saved_searches or []

        # Find and remove
        new_saved = [s for s in saved if s.get("name") != name]

        if len(new_saved) == len(saved):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Saved search '{name}' not found",
            )

        prefs.saved_searches = new_saved
        await session.commit()

        logger.info("Deleted saved search", user_id=user["sub"], name=name)

        return {"message": f"Deleted saved search '{name}'"}
