"""
Generation Templates Service
============================

Service for managing document generation templates.
Handles CRUD operations, system templates seeding, and template usage tracking.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import structlog
from sqlalchemy import select, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import GenerationTemplate, User, TemplateCategory

logger = structlog.get_logger(__name__)


# =============================================================================
# System Templates - Pre-built templates available to all users
# =============================================================================

SYSTEM_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "Business Report",
        "description": "Professional business report with formal structure, executive summary, and data-driven insights.",
        "category": TemplateCategory.REPORT.value,
        "settings": {
            "output_format": "docx",
            "theme": "business",
            "font_family": "professional",
            "layout_template": "standard",
            "include_toc": True,
            "include_sources": True,
            "use_existing_docs": True,
        },
        "tags": ["business", "report", "professional", "formal"],
    },
    {
        "name": "Project Proposal",
        "description": "Compelling project proposal with clear objectives, methodology, timeline, and budget sections.",
        "category": TemplateCategory.PROPOSAL.value,
        "settings": {
            "output_format": "docx",
            "theme": "modern",
            "font_family": "professional",
            "layout_template": "standard",
            "include_toc": True,
            "include_sources": True,
            "use_existing_docs": True,
        },
        "tags": ["proposal", "project", "modern", "structured"],
    },
    {
        "name": "Executive Presentation",
        "description": "High-impact executive presentation with clean visuals, key metrics, and strategic insights.",
        "category": TemplateCategory.PRESENTATION.value,
        "settings": {
            "output_format": "pptx",
            "theme": "elegant",
            "font_family": "professional",
            "layout_template": "standard",
            "include_toc": False,
            "include_sources": True,
            "use_existing_docs": True,
            "enable_animations": True,
        },
        "tags": ["presentation", "executive", "elegant", "slides"],
    },
    {
        "name": "Meeting Notes",
        "description": "Structured meeting notes with action items, decisions, and follow-up tasks.",
        "category": TemplateCategory.MEETING_NOTES.value,
        "settings": {
            "output_format": "md",
            "theme": "minimal",
            "font_family": "professional",
            "layout_template": "minimal",
            "include_toc": False,
            "include_sources": False,
            "use_existing_docs": False,
        },
        "tags": ["meeting", "notes", "action-items", "minimal"],
    },
    {
        "name": "Technical Documentation",
        "description": "Comprehensive technical documentation with code samples, diagrams, and detailed explanations.",
        "category": TemplateCategory.DOCUMENTATION.value,
        "settings": {
            "output_format": "md",
            "theme": "business",
            "font_family": "technical",
            "layout_template": "standard",
            "include_toc": True,
            "include_sources": True,
            "use_existing_docs": True,
        },
        "tags": ["technical", "documentation", "developer", "reference"],
    },
    {
        "name": "Creative Pitch Deck",
        "description": "Visually striking pitch deck for creative presentations, startups, and marketing.",
        "category": TemplateCategory.PRESENTATION.value,
        "settings": {
            "output_format": "pptx",
            "theme": "creative",
            "font_family": "modern",
            "layout_template": "wide",
            "include_toc": False,
            "include_sources": False,
            "use_existing_docs": False,
            "enable_animations": True,
            "enable_images": True,
        },
        "tags": ["creative", "pitch", "startup", "marketing", "visual"],
    },
]


# =============================================================================
# Template Service Class
# =============================================================================

class GenerationTemplateService:
    """Service for managing generation templates."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def seed_system_templates(self) -> int:
        """
        Seed system templates if they don't exist.

        Returns:
            Number of templates created.
        """
        created_count = 0

        for template_data in SYSTEM_TEMPLATES:
            # Check if template already exists
            existing = await self.db.execute(
                select(GenerationTemplate).where(
                    and_(
                        GenerationTemplate.name == template_data["name"],
                        GenerationTemplate.is_system == True,
                    )
                )
            )
            if existing.scalar_one_or_none():
                continue

            # Create system template
            template = GenerationTemplate(
                id=uuid.uuid4(),
                user_id=None,  # System templates have no owner
                name=template_data["name"],
                description=template_data.get("description"),
                category=template_data.get("category", TemplateCategory.CUSTOM.value),
                settings=template_data["settings"],
                tags=template_data.get("tags"),
                is_public=True,
                is_system=True,
                use_count=0,
            )
            self.db.add(template)
            created_count += 1

        if created_count > 0:
            await self.db.commit()
            logger.info("Seeded system templates", count=created_count)

        return created_count

    async def list_templates(
        self,
        user_id: Optional[uuid.UUID] = None,
        category: Optional[str] = None,
        include_system: bool = True,
        include_public: bool = True,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[GenerationTemplate], int]:
        """
        List available templates for a user.

        Args:
            user_id: Current user's ID (for private templates)
            category: Filter by category
            include_system: Include system templates
            include_public: Include public templates from other users
            search: Search in name/description
            limit: Max results
            offset: Offset for pagination

        Returns:
            Tuple of (templates, total_count)
        """
        # Build visibility filter
        visibility_conditions = []

        if include_system:
            visibility_conditions.append(GenerationTemplate.is_system == True)

        if user_id:
            visibility_conditions.append(GenerationTemplate.user_id == user_id)

        if include_public:
            visibility_conditions.append(
                and_(
                    GenerationTemplate.is_public == True,
                    GenerationTemplate.is_system == False,
                )
            )

        if not visibility_conditions:
            return [], 0

        # Base query
        query = select(GenerationTemplate).where(or_(*visibility_conditions))

        # Category filter
        if category:
            query = query.where(GenerationTemplate.category == category)

        # Search filter
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                or_(
                    GenerationTemplate.name.ilike(search_pattern),
                    GenerationTemplate.description.ilike(search_pattern),
                )
            )

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total_count = total_result.scalar() or 0

        # Apply ordering and pagination
        query = query.order_by(
            GenerationTemplate.is_system.desc(),  # System templates first
            GenerationTemplate.use_count.desc(),  # Most used next
            GenerationTemplate.created_at.desc(),
        )
        query = query.limit(limit).offset(offset)

        result = await self.db.execute(query)
        templates = list(result.scalars().all())

        return templates, total_count

    async def get_template(
        self,
        template_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[GenerationTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: Template ID
            user_id: Current user's ID (for access control)

        Returns:
            Template if found and accessible, None otherwise
        """
        query = select(GenerationTemplate).where(GenerationTemplate.id == template_id)
        result = await self.db.execute(query)
        template = result.scalar_one_or_none()

        if not template:
            return None

        # Check access
        if template.is_system or template.is_public:
            return template

        if user_id and template.user_id == user_id:
            return template

        return None

    async def create_template(
        self,
        user_id: uuid.UUID,
        name: str,
        settings: Dict[str, Any],
        description: Optional[str] = None,
        category: str = TemplateCategory.CUSTOM.value,
        default_collections: Optional[List[str]] = None,
        is_public: bool = False,
        tags: Optional[List[str]] = None,
        thumbnail: Optional[str] = None,
    ) -> GenerationTemplate:
        """
        Create a new user template.

        Args:
            user_id: Owner's ID
            name: Template name
            settings: Generation settings
            description: Template description
            category: Template category
            default_collections: Default collections for style learning
            is_public: Whether template is visible to other users
            tags: Tags for filtering
            thumbnail: Base64 encoded thumbnail

        Returns:
            Created template
        """
        template = GenerationTemplate(
            id=uuid.uuid4(),
            user_id=user_id,
            name=name,
            description=description,
            category=category,
            settings=settings,
            default_collections=default_collections,
            is_public=is_public,
            is_system=False,
            tags=tags,
            thumbnail=thumbnail,
            use_count=0,
        )

        self.db.add(template)
        await self.db.commit()
        await self.db.refresh(template)

        logger.info(
            "Created generation template",
            template_id=str(template.id),
            name=name,
            user_id=str(user_id),
        )

        return template

    async def update_template(
        self,
        template_id: uuid.UUID,
        user_id: uuid.UUID,
        **updates,
    ) -> Optional[GenerationTemplate]:
        """
        Update a user's template.

        Args:
            template_id: Template ID
            user_id: Current user's ID (must own template)
            **updates: Fields to update

        Returns:
            Updated template if successful, None otherwise
        """
        query = select(GenerationTemplate).where(
            and_(
                GenerationTemplate.id == template_id,
                GenerationTemplate.user_id == user_id,
                GenerationTemplate.is_system == False,
            )
        )
        result = await self.db.execute(query)
        template = result.scalar_one_or_none()

        if not template:
            return None

        # Apply updates
        allowed_fields = {
            "name", "description", "category", "settings",
            "default_collections", "is_public", "tags", "thumbnail",
        }

        for field, value in updates.items():
            if field in allowed_fields and value is not None:
                setattr(template, field, value)

        await self.db.commit()
        await self.db.refresh(template)

        logger.info(
            "Updated generation template",
            template_id=str(template_id),
        )

        return template

    async def delete_template(
        self,
        template_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """
        Delete a user's template.

        Args:
            template_id: Template ID
            user_id: Current user's ID (must own template)

        Returns:
            True if deleted, False otherwise
        """
        query = select(GenerationTemplate).where(
            and_(
                GenerationTemplate.id == template_id,
                GenerationTemplate.user_id == user_id,
                GenerationTemplate.is_system == False,
            )
        )
        result = await self.db.execute(query)
        template = result.scalar_one_or_none()

        if not template:
            return False

        await self.db.delete(template)
        await self.db.commit()

        logger.info(
            "Deleted generation template",
            template_id=str(template_id),
        )

        return True

    async def duplicate_template(
        self,
        template_id: uuid.UUID,
        user_id: uuid.UUID,
        new_name: Optional[str] = None,
    ) -> Optional[GenerationTemplate]:
        """
        Duplicate a template for a user.

        Args:
            template_id: Template ID to duplicate
            user_id: New owner's ID
            new_name: Optional new name (defaults to "Copy of [original]")

        Returns:
            New template if successful, None otherwise
        """
        original = await self.get_template(template_id, user_id)
        if not original:
            return None

        new_template = GenerationTemplate(
            id=uuid.uuid4(),
            user_id=user_id,
            name=new_name or f"Copy of {original.name}",
            description=original.description,
            category=original.category,
            settings=original.settings.copy() if original.settings else {},
            default_collections=original.default_collections.copy() if original.default_collections else None,
            is_public=False,  # Duplicates start as private
            is_system=False,
            tags=original.tags.copy() if original.tags else None,
            thumbnail=original.thumbnail,
            use_count=0,
        )

        self.db.add(new_template)
        await self.db.commit()
        await self.db.refresh(new_template)

        logger.info(
            "Duplicated generation template",
            original_id=str(template_id),
            new_id=str(new_template.id),
            user_id=str(user_id),
        )

        return new_template

    async def record_template_use(self, template_id: uuid.UUID) -> None:
        """
        Record that a template was used.

        Args:
            template_id: Template ID
        """
        query = select(GenerationTemplate).where(GenerationTemplate.id == template_id)
        result = await self.db.execute(query)
        template = result.scalar_one_or_none()

        if template:
            template.use_count += 1
            template.last_used_at = datetime.utcnow()
            await self.db.commit()

    async def get_popular_templates(
        self,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[GenerationTemplate]:
        """
        Get most popular templates.

        Args:
            category: Filter by category
            limit: Max results

        Returns:
            List of popular templates
        """
        query = select(GenerationTemplate).where(
            or_(
                GenerationTemplate.is_system == True,
                GenerationTemplate.is_public == True,
            )
        )

        if category:
            query = query.where(GenerationTemplate.category == category)

        query = query.order_by(GenerationTemplate.use_count.desc()).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_user_templates(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> List[GenerationTemplate]:
        """
        Get all templates owned by a user.

        Args:
            user_id: User's ID
            limit: Max results

        Returns:
            List of user's templates
        """
        query = (
            select(GenerationTemplate)
            .where(GenerationTemplate.user_id == user_id)
            .order_by(GenerationTemplate.updated_at.desc())
            .limit(limit)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())


# =============================================================================
# Utility Functions
# =============================================================================

async def ensure_system_templates(db: AsyncSession) -> int:
    """
    Ensure system templates exist in the database.

    Args:
        db: Database session

    Returns:
        Number of templates created
    """
    service = GenerationTemplateService(db)
    return await service.seed_system_templates()
