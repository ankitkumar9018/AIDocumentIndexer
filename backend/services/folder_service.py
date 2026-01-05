"""
AIDocumentIndexer - Folder Service
===================================

Service for managing hierarchical folder structure.
Uses materialized path pattern for efficient subtree queries.

Features:
- CRUD operations for folders
- Hierarchy management (move, get tree)
- Permission-aware folder listing
- Document filtering by folder (for RAG scoping)
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

import structlog
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.db.models import Folder, Document, AccessTier
from backend.db.database import get_async_session

logger = structlog.get_logger(__name__)


class FolderService:
    """Service for folder CRUD and hierarchy management."""

    def __init__(self, session: Optional[AsyncSession] = None):
        """Initialize folder service with optional session."""
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get database session."""
        if self._session:
            return self._session
        async for session in get_async_session():
            return session
        raise RuntimeError("Could not get database session")

    async def create_folder(
        self,
        name: str,
        parent_folder_id: Optional[str] = None,
        access_tier_id: Optional[str] = None,
        created_by_id: Optional[str] = None,
        inherit_permissions: bool = True,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Folder:
        """
        Create a new folder.

        Args:
            name: Folder display name
            parent_folder_id: Parent folder ID (None for root)
            access_tier_id: Access tier for permissions
            created_by_id: User who created the folder
            inherit_permissions: Whether to inherit parent's permissions
            description: Optional folder description
            color: Optional hex color for UI

        Returns:
            Created Folder object
        """
        session = await self._get_session()

        # Compute path and depth based on parent
        if parent_folder_id:
            parent = await self.get_folder_by_id(parent_folder_id)
            if not parent:
                raise ValueError(f"Parent folder {parent_folder_id} not found")
            path = f"{parent.path}{name}/"
            depth = parent.depth + 1
            # Use parent's access tier if not specified
            if not access_tier_id:
                access_tier_id = str(parent.access_tier_id)
        else:
            # Root folder
            path = f"/{name}/"
            depth = 0
            # Must have access_tier_id for root folders
            if not access_tier_id:
                # Get default tier (lowest level)
                result = await session.execute(
                    select(AccessTier).order_by(AccessTier.level.asc()).limit(1)
                )
                default_tier = result.scalar_one_or_none()
                if default_tier:
                    access_tier_id = str(default_tier.id)
                else:
                    raise ValueError("No access tiers found, cannot create folder")

        # Check for duplicate folder name in same parent
        existing = await self._get_folder_by_parent_and_name(parent_folder_id, name)
        if existing:
            raise ValueError(f"Folder '{name}' already exists in this location")

        # Create folder
        folder = Folder(
            id=uuid.uuid4(),
            name=name,
            path=path,
            parent_folder_id=uuid.UUID(parent_folder_id) if parent_folder_id else None,
            depth=depth,
            access_tier_id=uuid.UUID(access_tier_id),
            inherit_permissions=inherit_permissions,
            created_by_id=uuid.UUID(created_by_id) if created_by_id else None,
            description=description,
            color=color,
        )

        session.add(folder)
        await session.commit()
        await session.refresh(folder)

        logger.info(
            "Created folder",
            folder_id=str(folder.id),
            name=name,
            path=path,
            depth=depth,
        )

        return folder

    async def get_folder_by_id(
        self,
        folder_id: str,
        include_children: bool = False,
    ) -> Optional[Folder]:
        """
        Get folder by ID.

        Args:
            folder_id: Folder UUID
            include_children: Whether to eager load subfolders

        Returns:
            Folder object or None
        """
        session = await self._get_session()

        query = select(Folder).where(Folder.id == uuid.UUID(folder_id))

        if include_children:
            query = query.options(selectinload(Folder.subfolders))

        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def _get_folder_by_parent_and_name(
        self,
        parent_folder_id: Optional[str],
        name: str,
    ) -> Optional[Folder]:
        """Check if folder with name exists in parent."""
        session = await self._get_session()

        if parent_folder_id:
            query = select(Folder).where(
                and_(
                    Folder.parent_folder_id == uuid.UUID(parent_folder_id),
                    Folder.name == name,
                )
            )
        else:
            query = select(Folder).where(
                and_(
                    Folder.parent_folder_id.is_(None),
                    Folder.name == name,
                )
            )

        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def list_folders(
        self,
        parent_folder_id: Optional[str] = None,
        user_tier_level: int = 100,
        include_document_count: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List folders at a specific level that user can access.

        Args:
            parent_folder_id: Parent folder ID (None for root level)
            user_tier_level: User's access tier level for filtering
            include_document_count: Include count of documents in each folder

        Returns:
            List of folder dicts with metadata
        """
        session = await self._get_session()

        # Build base query
        if parent_folder_id:
            query = select(Folder).where(
                Folder.parent_folder_id == uuid.UUID(parent_folder_id)
            )
        else:
            query = select(Folder).where(Folder.parent_folder_id.is_(None))

        # Order by name
        query = query.order_by(Folder.name.asc())

        result = await session.execute(query)
        folders = result.scalars().all()

        # Filter by access tier and build response
        folder_list = []
        for folder in folders:
            effective_tier = await self.get_effective_tier_level(folder)
            if user_tier_level >= effective_tier:
                folder_dict = {
                    "id": str(folder.id),
                    "name": folder.name,
                    "path": folder.path,
                    "depth": folder.depth,
                    "parent_folder_id": str(folder.parent_folder_id) if folder.parent_folder_id else None,
                    "access_tier_id": str(folder.access_tier_id),
                    "inherit_permissions": folder.inherit_permissions,
                    "description": folder.description,
                    "color": folder.color,
                    "created_at": folder.created_at.isoformat() if folder.created_at else None,
                    "created_by_id": str(folder.created_by_id) if folder.created_by_id else None,
                }

                if include_document_count:
                    count = await self._count_folder_documents(str(folder.id))
                    folder_dict["document_count"] = count

                folder_list.append(folder_dict)

        return folder_list

    async def get_folder_tree(
        self,
        user_tier_level: int = 100,
        root_folder_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get full folder tree structure for navigation.

        Args:
            user_tier_level: User's access tier level for filtering
            root_folder_id: Start from specific folder (None for all roots)

        Returns:
            Nested tree structure of folders
        """
        session = await self._get_session()

        # Get all folders, ordered by path for hierarchical ordering
        query = select(Folder).order_by(Folder.path.asc())

        if root_folder_id:
            # Get subtree starting from specified folder
            root = await self.get_folder_by_id(root_folder_id)
            if root:
                query = query.where(Folder.path.like(f"{root.path}%"))

        result = await session.execute(query)
        all_folders = result.scalars().all()

        # Build tree structure
        folder_map: Dict[str, Dict[str, Any]] = {}
        roots: List[Dict[str, Any]] = []

        for folder in all_folders:
            effective_tier = await self.get_effective_tier_level(folder)
            if user_tier_level < effective_tier:
                continue  # Skip folders user can't access

            folder_dict = {
                "id": str(folder.id),
                "name": folder.name,
                "path": folder.path,
                "depth": folder.depth,
                "parent_folder_id": str(folder.parent_folder_id) if folder.parent_folder_id else None,
                "color": folder.color,
                "children": [],
            }
            folder_map[str(folder.id)] = folder_dict

            if folder.parent_folder_id:
                parent_id = str(folder.parent_folder_id)
                if parent_id in folder_map:
                    folder_map[parent_id]["children"].append(folder_dict)
            else:
                roots.append(folder_dict)

        return roots

    async def update_folder(
        self,
        folder_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
        access_tier_id: Optional[str] = None,
        inherit_permissions: Optional[bool] = None,
    ) -> Optional[Folder]:
        """
        Update folder metadata.

        Note: Renaming a folder updates paths of all descendants.
        """
        session = await self._get_session()

        folder = await self.get_folder_by_id(folder_id)
        if not folder:
            return None

        old_path = folder.path

        # Update simple fields
        if description is not None:
            folder.description = description
        if color is not None:
            folder.color = color
        if access_tier_id is not None:
            folder.access_tier_id = uuid.UUID(access_tier_id)
        if inherit_permissions is not None:
            folder.inherit_permissions = inherit_permissions

        # Handle rename (requires path update for descendants)
        if name and name != folder.name:
            # Check for duplicate
            existing = await self._get_folder_by_parent_and_name(
                str(folder.parent_folder_id) if folder.parent_folder_id else None,
                name,
            )
            if existing and str(existing.id) != folder_id:
                raise ValueError(f"Folder '{name}' already exists in this location")

            # Compute new path
            if folder.parent_folder_id:
                parent = await self.get_folder_by_id(str(folder.parent_folder_id))
                new_path = f"{parent.path}{name}/"
            else:
                new_path = f"/{name}/"

            folder.name = name
            folder.path = new_path

            # Update all descendant paths
            await self._update_descendant_paths(old_path, new_path)

        await session.commit()
        await session.refresh(folder)

        logger.info("Updated folder", folder_id=folder_id, name=folder.name)
        return folder

    async def _update_descendant_paths(self, old_prefix: str, new_prefix: str) -> int:
        """Update paths of all folders under old_prefix to use new_prefix."""
        session = await self._get_session()

        # Find all descendants (path starts with old_prefix but is not old_prefix itself)
        result = await session.execute(
            select(Folder).where(
                and_(
                    Folder.path.like(f"{old_prefix}%"),
                    Folder.path != old_prefix,
                )
            )
        )
        descendants = result.scalars().all()

        count = 0
        for folder in descendants:
            # Replace old prefix with new prefix
            folder.path = new_prefix + folder.path[len(old_prefix):]
            count += 1

        await session.commit()
        return count

    async def move_folder(
        self,
        folder_id: str,
        new_parent_id: Optional[str],
        user_tier_level: int = 100,
    ) -> Optional[Folder]:
        """
        Move folder to a new parent.

        Args:
            folder_id: Folder to move
            new_parent_id: New parent folder ID (None for root)
            user_tier_level: User's tier level for permission check

        Returns:
            Updated folder or None if not found/not allowed
        """
        session = await self._get_session()

        folder = await self.get_folder_by_id(folder_id)
        if not folder:
            return None

        # Check if moving to a descendant (invalid)
        if new_parent_id:
            new_parent = await self.get_folder_by_id(new_parent_id)
            if not new_parent:
                raise ValueError(f"Target folder {new_parent_id} not found")
            if new_parent.path.startswith(folder.path):
                raise ValueError("Cannot move folder into its own descendant")

            # Check user has access to target
            target_tier = await self.get_effective_tier_level(new_parent)
            if user_tier_level < target_tier:
                raise PermissionError("No access to target folder")

        old_path = folder.path

        # Compute new path and depth
        if new_parent_id:
            new_parent = await self.get_folder_by_id(new_parent_id)
            new_path = f"{new_parent.path}{folder.name}/"
            new_depth = new_parent.depth + 1
        else:
            new_path = f"/{folder.name}/"
            new_depth = 0

        # Check for name conflict in new location
        existing = await self._get_folder_by_parent_and_name(new_parent_id, folder.name)
        if existing:
            raise ValueError(f"Folder '{folder.name}' already exists in target location")

        # Update folder
        folder.parent_folder_id = uuid.UUID(new_parent_id) if new_parent_id else None
        folder.path = new_path
        folder.depth = new_depth

        # Update descendant paths and depths
        await self._update_descendants_after_move(old_path, new_path, new_depth)

        await session.commit()
        await session.refresh(folder)

        logger.info(
            "Moved folder",
            folder_id=folder_id,
            old_path=old_path,
            new_path=new_path,
        )

        return folder

    async def _update_descendants_after_move(
        self,
        old_prefix: str,
        new_prefix: str,
        parent_depth: int,
    ) -> None:
        """Update paths and depths of all descendants after move."""
        session = await self._get_session()

        result = await session.execute(
            select(Folder).where(
                and_(
                    Folder.path.like(f"{old_prefix}%"),
                    Folder.path != old_prefix,
                )
            )
        )
        descendants = result.scalars().all()

        old_prefix_len = len(old_prefix)
        for folder in descendants:
            # Calculate relative depth from moved folder
            relative_depth = folder.depth - (old_prefix.count("/") - 1)
            folder.path = new_prefix + folder.path[old_prefix_len:]
            folder.depth = parent_depth + relative_depth

        await session.commit()

    async def delete_folder(
        self,
        folder_id: str,
        recursive: bool = False,
        user_tier_level: int = 100,
    ) -> bool:
        """
        Delete a folder.

        Args:
            folder_id: Folder to delete
            recursive: If True, delete all contents (subfolders and documents)
            user_tier_level: User's tier for permission check

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If folder has contents and recursive=False
        """
        session = await self._get_session()

        folder = await self.get_folder_by_id(folder_id)
        if not folder:
            return False

        # Check permissions
        effective_tier = await self.get_effective_tier_level(folder)
        if user_tier_level < effective_tier:
            raise PermissionError("No access to delete this folder")

        # Check for contents
        has_subfolders = await self._has_subfolders(folder_id)
        has_documents = await self._count_folder_documents(folder_id) > 0

        if (has_subfolders or has_documents) and not recursive:
            raise ValueError(
                "Folder has contents. Use recursive=True to delete all contents."
            )

        if recursive:
            # Documents will have folder_id set to NULL due to ON DELETE SET NULL
            # Subfolders will be deleted via CASCADE
            pass

        # Delete folder
        await session.delete(folder)
        await session.commit()

        logger.info(
            "Deleted folder",
            folder_id=folder_id,
            path=folder.path,
            recursive=recursive,
        )

        return True

    async def _has_subfolders(self, folder_id: str) -> bool:
        """Check if folder has any subfolders."""
        session = await self._get_session()
        result = await session.execute(
            select(func.count(Folder.id)).where(
                Folder.parent_folder_id == uuid.UUID(folder_id)
            )
        )
        return result.scalar() > 0

    async def _count_folder_documents(
        self,
        folder_id: str,
        include_subfolders: bool = False,
    ) -> int:
        """Count documents in folder."""
        session = await self._get_session()

        if include_subfolders:
            folder = await self.get_folder_by_id(folder_id)
            if not folder:
                return 0
            # Get all folder IDs in subtree
            folder_ids = await self._get_subtree_folder_ids(folder.path)
            result = await session.execute(
                select(func.count(Document.id)).where(
                    Document.folder_id.in_([uuid.UUID(fid) for fid in folder_ids])
                )
            )
        else:
            result = await session.execute(
                select(func.count(Document.id)).where(
                    Document.folder_id == uuid.UUID(folder_id)
                )
            )

        return result.scalar() or 0

    async def _get_subtree_folder_ids(self, path_prefix: str) -> List[str]:
        """Get all folder IDs under a path prefix."""
        session = await self._get_session()
        result = await session.execute(
            select(Folder.id).where(Folder.path.like(f"{path_prefix}%"))
        )
        return [str(row[0]) for row in result.fetchall()]

    async def get_folder_document_ids(
        self,
        folder_id: str,
        include_subfolders: bool = True,
        user_tier_level: int = 100,
    ) -> List[str]:
        """
        Get all document IDs in a folder for RAG query filtering.

        Args:
            folder_id: Folder to get documents from
            include_subfolders: Include documents from all subfolders
            user_tier_level: User's tier for access filtering

        Returns:
            List of document IDs that user can access
        """
        session = await self._get_session()

        folder = await self.get_folder_by_id(folder_id)
        if not folder:
            return []

        # Check folder access
        effective_tier = await self.get_effective_tier_level(folder)
        if user_tier_level < effective_tier:
            return []  # No access to this folder

        if include_subfolders:
            # Get all folders in subtree
            folder_ids = await self._get_subtree_folder_ids(folder.path)
            folder_uuids = [uuid.UUID(fid) for fid in folder_ids]

            # Get documents from all folders that user can access
            query = select(Document.id, Document.access_tier_id).where(
                Document.folder_id.in_(folder_uuids)
            )
        else:
            query = select(Document.id, Document.access_tier_id).where(
                Document.folder_id == uuid.UUID(folder_id)
            )

        result = await session.execute(query)
        rows = result.fetchall()

        # Filter by document access tier
        accessible_ids = []
        for doc_id, tier_id in rows:
            # Get tier level for this document
            tier_result = await session.execute(
                select(AccessTier.level).where(AccessTier.id == tier_id)
            )
            tier_level = tier_result.scalar()
            if tier_level and user_tier_level >= tier_level:
                accessible_ids.append(str(doc_id))

        return accessible_ids

    async def get_effective_tier_level(self, folder: Folder) -> int:
        """
        Get the effective access tier level for a folder.

        If inherit_permissions is True, walks up the tree to find
        the first folder with inherit_permissions=False, or uses root's tier.

        Args:
            folder: Folder object

        Returns:
            Effective tier level (integer)
        """
        session = await self._get_session()

        current = folder
        while current.inherit_permissions and current.parent_folder_id:
            # Get parent
            result = await session.execute(
                select(Folder).where(Folder.id == current.parent_folder_id)
            )
            parent = result.scalar_one_or_none()
            if parent:
                current = parent
            else:
                break

        # Get tier level
        result = await session.execute(
            select(AccessTier.level).where(AccessTier.id == current.access_tier_id)
        )
        level = result.scalar()
        return level if level else 1  # Default to lowest tier if not found

    async def get_folder_path_breadcrumbs(
        self,
        folder_id: str,
    ) -> List[Dict[str, str]]:
        """
        Get breadcrumb path from root to folder.

        Returns list of {id, name} dicts representing the path.
        """
        session = await self._get_session()

        folder = await self.get_folder_by_id(folder_id)
        if not folder:
            return []

        # Parse path to get folder names
        path_parts = [p for p in folder.path.split("/") if p]
        breadcrumbs = []

        current_path = "/"
        for part in path_parts:
            current_path += f"{part}/"
            # Find folder with this path
            result = await session.execute(
                select(Folder.id, Folder.name).where(Folder.path == current_path)
            )
            row = result.first()
            if row:
                breadcrumbs.append({"id": str(row[0]), "name": row[1]})

        return breadcrumbs


# Singleton instance
_folder_service: Optional[FolderService] = None


def get_folder_service() -> FolderService:
    """Get the folder service singleton."""
    global _folder_service
    if _folder_service is None:
        _folder_service = FolderService()
    return _folder_service
