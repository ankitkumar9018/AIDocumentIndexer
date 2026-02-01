"""
AIDocumentIndexer - Folder API Routes
======================================

API endpoints for folder management:
- CRUD operations for folders
- Folder tree navigation
- Move operations for folders and documents
- Folder-scoped document listing
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import get_current_user, CurrentUser
from backend.services.folder_service import FolderService, get_folder_service
from backend.db.database import get_async_session, async_session_context
from backend.db.models import User
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateFolderRequest(BaseModel):
    """Request to create a new folder."""
    name: str = Field(..., min_length=1, max_length=255, description="Folder name")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID (null for root)")
    access_tier_id: Optional[str] = Field(None, description="Access tier ID (inherits from parent if not set)")
    inherit_permissions: bool = Field(True, description="Whether to inherit parent's permissions")
    description: Optional[str] = Field(None, max_length=1000, description="Folder description")
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Hex color code")
    tags: Optional[List[str]] = Field(None, description="Tags for folder categorization")


class UpdateFolderRequest(BaseModel):
    """Request to update a folder."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    access_tier_id: Optional[str] = None
    inherit_permissions: Optional[bool] = None
    tags: Optional[List[str]] = Field(None, description="Tags for folder categorization")


class MoveFolderRequest(BaseModel):
    """Request to move a folder."""
    new_parent_id: Optional[str] = Field(None, description="New parent folder ID (null for root)")


class FolderResponse(BaseModel):
    """Folder response model."""
    id: str
    name: str
    path: str
    depth: int
    parent_folder_id: Optional[str]
    access_tier_id: str
    inherit_permissions: bool
    description: Optional[str]
    color: Optional[str]
    tags: Optional[List[str]] = None
    created_at: Optional[str]
    created_by_id: Optional[str]
    document_count: Optional[int] = None


class FolderTreeNode(BaseModel):
    """Folder tree node with children."""
    id: str
    name: str
    path: str
    depth: int
    parent_folder_id: Optional[str]
    color: Optional[str]
    tags: Optional[List[str]] = None
    children: List["FolderTreeNode"] = []


class BreadcrumbItem(BaseModel):
    """Breadcrumb item for folder path."""
    id: str
    name: str


# Enable forward references for recursive model
FolderTreeNode.model_rebuild()


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/", response_model=FolderResponse, status_code=status.HTTP_201_CREATED)
async def create_folder(
    request: CreateFolderRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Create a new folder.

    - Root folders (no parent) require explicit access_tier_id
    - Child folders inherit parent's tier if not specified
    """
    try:
        folder_service = get_folder_service()

        # Get user's access tier level
        user_tier_level = current_user.get("access_tier_level", 1)

        # If creating in a parent folder, check access
        if request.parent_folder_id:
            parent = await folder_service.get_folder_by_id(request.parent_folder_id)
            if not parent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Parent folder not found"
                )
            effective_tier = await folder_service.get_effective_tier_level(parent)
            if user_tier_level < effective_tier:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="No access to parent folder"
                )

        folder = await folder_service.create_folder(
            name=request.name,
            parent_folder_id=request.parent_folder_id,
            access_tier_id=request.access_tier_id,
            created_by_id=current_user.get("sub"),
            inherit_permissions=request.inherit_permissions,
            description=request.description,
            color=request.color,
            tags=request.tags,
        )

        return FolderResponse(
            id=str(folder.id),
            name=folder.name,
            path=folder.path,
            depth=folder.depth,
            parent_folder_id=str(folder.parent_folder_id) if folder.parent_folder_id else None,
            access_tier_id=str(folder.access_tier_id),
            inherit_permissions=folder.inherit_permissions,
            description=folder.description,
            color=folder.color,
            tags=folder.tags or [],
            created_at=folder.created_at.isoformat() if folder.created_at else None,
            created_by_id=str(folder.created_by_id) if folder.created_by_id else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Error creating folder", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create folder"
        )


@router.get("/", response_model=List[FolderResponse])
async def list_folders(
    parent_id: Optional[str] = Query(None, description="Parent folder ID (null for root level)"),
    include_document_count: bool = Query(False, description="Include document count"),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    List folders at a specific level.

    - If parent_id is null, returns root-level folders
    - Only returns folders the user can access
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)

        folders = await folder_service.list_folders(
            parent_folder_id=parent_id,
            user_tier_level=user_tier_level,
            include_document_count=include_document_count,
        )

        return [FolderResponse(**f) for f in folders]

    except Exception as e:
        logger.error("Error listing folders", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list folders"
        )


@router.get("/tree", response_model=List[FolderTreeNode])
async def get_folder_tree(
    root_folder_id: Optional[str] = Query(None, description="Root folder ID (null for full tree)"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get folder tree structure for navigation.

    Returns nested folder structure with children.
    Respects user's use_folder_permissions_only flag for permission-based filtering.
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)
        user_id = current_user.get("sub")  # JWT uses 'sub' for user ID

        # Fetch use_folder_permissions_only from database for real-time permission check
        use_folder_permissions_only = False
        if user_id:
            result = await db.execute(
                select(User.use_folder_permissions_only).where(User.id == user_id)
            )
            row = result.scalar_one_or_none()
            if row is not None:
                use_folder_permissions_only = row

        tree = await folder_service.get_folder_tree(
            user_tier_level=user_tier_level,
            root_folder_id=root_folder_id,
            user_id=user_id,
            use_folder_permissions_only=use_folder_permissions_only,
        )

        return tree

    except Exception as e:
        logger.error("Error getting folder tree", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get folder tree"
        )


@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get folder details by ID."""
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)

        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Check access
        effective_tier = await folder_service.get_effective_tier_level(folder)
        if user_tier_level < effective_tier:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this folder"
            )

        return FolderResponse(
            id=str(folder.id),
            name=folder.name,
            path=folder.path,
            depth=folder.depth,
            parent_folder_id=str(folder.parent_folder_id) if folder.parent_folder_id else None,
            access_tier_id=str(folder.access_tier_id),
            inherit_permissions=folder.inherit_permissions,
            description=folder.description,
            color=folder.color,
            tags=folder.tags or [],
            created_at=folder.created_at.isoformat() if folder.created_at else None,
            created_by_id=str(folder.created_by_id) if folder.created_by_id else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting folder", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get folder"
        )


@router.get("/{folder_id}/breadcrumbs", response_model=List[BreadcrumbItem])
async def get_folder_breadcrumbs(
    folder_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get breadcrumb path from root to folder."""
    try:
        folder_service = get_folder_service()

        breadcrumbs = await folder_service.get_folder_path_breadcrumbs(folder_id)
        return [BreadcrumbItem(**b) for b in breadcrumbs]

    except Exception as e:
        logger.error("Error getting breadcrumbs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get breadcrumbs"
        )


@router.patch("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: str,
    request: UpdateFolderRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Update folder metadata.

    Renaming a folder updates the paths of all descendants.
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)
        user_id = current_user.get("sub")

        # Check access
        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        effective_tier = await folder_service.get_effective_tier_level(folder)
        is_admin = user_tier_level >= 100
        is_creator = str(folder.created_by_id) == user_id if folder.created_by_id else False

        if not is_admin and not is_creator:
            if user_tier_level < effective_tier:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="No access to modify this folder"
                )

        updated = await folder_service.update_folder(
            folder_id=folder_id,
            name=request.name,
            description=request.description,
            color=request.color,
            access_tier_id=request.access_tier_id,
            inherit_permissions=request.inherit_permissions,
            tags=request.tags,
        )

        return FolderResponse(
            id=str(updated.id),
            name=updated.name,
            path=updated.path,
            depth=updated.depth,
            parent_folder_id=str(updated.parent_folder_id) if updated.parent_folder_id else None,
            access_tier_id=str(updated.access_tier_id),
            inherit_permissions=updated.inherit_permissions,
            description=updated.description,
            color=updated.color,
            tags=updated.tags or [],
            created_at=updated.created_at.isoformat() if updated.created_at else None,
            created_by_id=str(updated.created_by_id) if updated.created_by_id else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating folder", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update folder"
        )


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: str,
    recursive: bool = Query(False, description="Delete all contents recursively"),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Delete a folder.

    - If recursive=False and folder has contents, returns error
    - If recursive=True, deletes all subfolders and unlinks documents
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)
        user_id = current_user.get("sub")

        # Check access
        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        is_admin = user_tier_level >= 100
        is_creator = str(folder.created_by_id) == user_id if folder.created_by_id else False

        if not is_admin and not is_creator:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin or folder creator can delete folders"
            )

        deleted = await folder_service.delete_folder(
            folder_id=folder_id,
            recursive=recursive,
            user_tier_level=user_tier_level,
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        return {"message": "Folder deleted successfully", "folder_id": folder_id}

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting folder", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete folder"
        )


@router.post("/{folder_id}/move", response_model=FolderResponse)
async def move_folder(
    folder_id: str,
    request: MoveFolderRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Move folder to a new parent.

    - Cannot move folder into its own descendant
    - Updates paths of all descendants
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)

        # Check source folder access
        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        effective_tier = await folder_service.get_effective_tier_level(folder)
        if user_tier_level < effective_tier:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to move this folder"
            )

        moved = await folder_service.move_folder(
            folder_id=folder_id,
            new_parent_id=request.new_parent_id,
            user_tier_level=user_tier_level,
        )

        if not moved:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        return FolderResponse(
            id=str(moved.id),
            name=moved.name,
            path=moved.path,
            depth=moved.depth,
            parent_folder_id=str(moved.parent_folder_id) if moved.parent_folder_id else None,
            access_tier_id=str(moved.access_tier_id),
            inherit_permissions=moved.inherit_permissions,
            description=moved.description,
            color=moved.color,
            tags=moved.tags or [],
            created_at=moved.created_at.isoformat() if moved.created_at else None,
            created_by_id=str(moved.created_by_id) if moved.created_by_id else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error moving folder", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to move folder"
        )


@router.get("/{folder_id}/documents")
async def list_folder_documents(
    folder_id: str,
    include_subfolders: bool = Query(True, description="Include documents from subfolders"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    List documents in a folder.

    Returns paginated list of documents the user can access.
    """
    from sqlalchemy import select, func
    from backend.db.models import Document, AccessTier
    from backend.db.database import get_async_session
    import uuid

    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)

        # Check folder access
        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        effective_tier = await folder_service.get_effective_tier_level(folder)
        if user_tier_level < effective_tier:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to this folder"
            )

        # Get document IDs
        doc_ids = await folder_service.get_folder_document_ids(
            folder_id=folder_id,
            include_subfolders=include_subfolders,
            user_tier_level=user_tier_level,
        )

        # Get paginated documents
        async with async_session_context() as session:
            if not doc_ids:
                return {
                    "items": [],
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                }

            doc_uuids = [uuid.UUID(did) for did in doc_ids]

            # Count total
            count_result = await session.execute(
                select(func.count(Document.id)).where(Document.id.in_(doc_uuids))
            )
            total = count_result.scalar() or 0

            # Get page
            offset = (page - 1) * page_size
            result = await session.execute(
                select(Document)
                .where(Document.id.in_(doc_uuids))
                .order_by(Document.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            documents = result.scalars().all()

            items = [
                {
                    "id": str(doc.id),
                    "filename": doc.filename,
                    "original_filename": doc.original_filename,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "title": doc.title,
                    "processing_status": doc.processing_status.value if doc.processing_status else None,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "folder_id": str(doc.folder_id) if doc.folder_id else None,
                    "tags": doc.tags or [],
                }
                for doc in documents
            ]

            total_pages = (total + page_size - 1) // page_size

            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing folder documents", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


# =============================================================================
# Folder Permission Endpoints
# =============================================================================

class GrantPermissionRequest(BaseModel):
    """Request to grant folder permission."""
    user_id: str = Field(..., description="User ID to grant permission to")
    permission_level: str = Field(
        "view",
        description="Permission level: view, edit, or manage"
    )
    inherit_to_children: bool = Field(
        True,
        description="Whether permission cascades to subfolders"
    )


class FolderPermissionResponse(BaseModel):
    """Folder permission response model."""
    id: str
    folder_id: str
    user_id: str
    user_email: str
    user_name: Optional[str]
    permission_level: str
    inherit_to_children: bool
    granted_by_id: Optional[str]
    created_at: Optional[str]


@router.get("/{folder_id}/permissions", response_model=List[FolderPermissionResponse])
async def get_folder_permissions(
    folder_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get all permissions for a folder.

    Only accessible by admins or users with MANAGE permission on the folder.
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)
        user_id = current_user.get("sub")

        # Check folder exists
        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Check access (admin or MANAGE permission)
        is_admin = user_tier_level >= 100
        has_manage = await folder_service.check_user_has_folder_permission(
            folder_id, user_id, "manage"
        )

        if not is_admin and not has_manage:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to view folder permissions"
            )

        permissions = await folder_service.get_folder_permissions(folder_id)
        return [FolderPermissionResponse(**p) for p in permissions]

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting folder permissions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get permissions"
        )


@router.post("/{folder_id}/permissions", response_model=FolderPermissionResponse)
async def grant_folder_permission(
    folder_id: str,
    request: GrantPermissionRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Grant a user permission to access a folder.

    Only accessible by admins or users with MANAGE permission on the folder.
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)
        granting_user_id = current_user.get("sub")

        # Check folder exists
        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Check access (admin or MANAGE permission)
        is_admin = user_tier_level >= 100
        has_manage = await folder_service.check_user_has_folder_permission(
            folder_id, granting_user_id, "manage"
        )

        if not is_admin and not has_manage:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to grant folder permissions"
            )

        # Validate permission level
        valid_levels = ["view", "edit", "manage"]
        if request.permission_level not in valid_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid permission level. Must be one of: {valid_levels}"
            )

        # Grant permission
        permission = await folder_service.grant_folder_permission(
            folder_id=folder_id,
            user_id=request.user_id,
            permission_level=request.permission_level,
            granted_by_id=granting_user_id,
            inherit_to_children=request.inherit_to_children,
        )

        # Fetch full permission data with user info
        permissions = await folder_service.get_folder_permissions(folder_id)
        for p in permissions:
            if p["user_id"] == request.user_id:
                return FolderPermissionResponse(**p)

        # Fallback if not found (shouldn't happen)
        return FolderPermissionResponse(
            id=str(permission.id),
            folder_id=str(permission.folder_id),
            user_id=str(permission.user_id),
            user_email="",
            user_name=None,
            permission_level=permission.permission_level,
            inherit_to_children=permission.inherit_to_children,
            granted_by_id=str(permission.granted_by_id) if permission.granted_by_id else None,
            created_at=permission.created_at.isoformat() if permission.created_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error granting folder permission", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to grant permission"
        )


@router.delete("/{folder_id}/permissions/{user_id}")
async def revoke_folder_permission(
    folder_id: str,
    user_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Revoke a user's permission to a folder.

    Only accessible by admins or users with MANAGE permission on the folder.
    """
    try:
        folder_service = get_folder_service()
        user_tier_level = current_user.get("access_tier_level", 1)
        revoking_user_id = current_user.get("sub")

        # Check folder exists
        folder = await folder_service.get_folder_by_id(folder_id)
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Check access (admin or MANAGE permission)
        is_admin = user_tier_level >= 100
        has_manage = await folder_service.check_user_has_folder_permission(
            folder_id, revoking_user_id, "manage"
        )

        if not is_admin and not has_manage:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No access to revoke folder permissions"
            )

        # Revoke permission
        deleted = await folder_service.revoke_folder_permission(folder_id, user_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Permission not found"
            )

        return {"message": "Permission revoked successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error revoking folder permission", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke permission"
        )
