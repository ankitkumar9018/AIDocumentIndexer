"""
Generation Templates Service
============================

Service for managing document generation templates.
Handles CRUD operations, system templates seeding, and template usage tracking.
"""

import uuid
from dataclasses import dataclass, field
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
            safe = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            search_pattern = f"%{safe}%"
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
# Template Learning from Successful Generations
# =============================================================================

@dataclass
class LearnedTemplateSettings:
    """Settings extracted from a successful document generation."""
    output_format: str
    theme: str
    font_family: str
    layout_template: str
    include_toc: bool
    include_sources: bool
    use_existing_docs: bool
    enable_images: bool = False
    enable_quality_review: bool = True
    style_guide: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateSuggestion:
    """A suggested template based on generation analysis."""
    name: str
    description: str
    category: str
    settings: Dict[str, Any]
    confidence_score: float  # 0-1, how good this template would be
    based_on_generation_id: str
    tags: List[str]
    is_duplicate_of: Optional[uuid.UUID] = None  # If similar to existing template


class TemplateLearningService:
    """Service for learning templates from successful document generations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self._template_service = GenerationTemplateService(db)

    async def analyze_generation_for_template(
        self,
        generation_id: str,
        user_rating: int,  # 1-5 rating from user
        generation_settings: Dict[str, Any],
        generation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TemplateSuggestion]:
        """
        Analyze a completed generation and suggest creating a template.

        Only suggests templates for highly-rated generations (4+ rating).

        Args:
            generation_id: ID of the completed generation
            user_rating: User's rating of the generation (1-5)
            generation_settings: Settings used for the generation
            generation_metadata: Additional metadata from generation

        Returns:
            TemplateSuggestion if appropriate, None otherwise
        """
        # Only learn from highly-rated generations
        if user_rating < 4:
            logger.debug(
                "Generation rating too low for template learning",
                rating=user_rating,
                generation_id=generation_id,
            )
            return None

        # Extract key settings
        extracted_settings = self._extract_template_settings(
            generation_settings,
            generation_metadata or {},
        )

        # Check for duplicate templates
        similar_template = await self._find_similar_template(extracted_settings)
        if similar_template:
            # Existing similar template found - increment usage instead
            await self._template_service.record_template_use(similar_template.id)
            logger.debug(
                "Similar template exists, incremented usage",
                template_id=str(similar_template.id),
                template_name=similar_template.name,
            )
            return TemplateSuggestion(
                name=similar_template.name,
                description=similar_template.description or "",
                category=similar_template.category,
                settings=extracted_settings,
                confidence_score=0.3,  # Low score since duplicate
                based_on_generation_id=generation_id,
                tags=similar_template.tags or [],
                is_duplicate_of=similar_template.id,
            )

        # Generate template suggestion
        suggestion = await self._generate_template_suggestion(
            generation_id=generation_id,
            settings=extracted_settings,
            metadata=generation_metadata or {},
            rating=user_rating,
        )

        return suggestion

    def _extract_template_settings(
        self,
        generation_settings: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract reusable template settings from generation config."""
        settings = {
            "output_format": generation_settings.get("output_format", "docx"),
            "theme": metadata.get("theme", "business"),
            "font_family": metadata.get("font_family", "professional"),
            "layout_template": metadata.get("layout_template", "standard"),
            "include_toc": metadata.get("include_toc", False),
            "include_sources": metadata.get("include_sources", True),
            "use_existing_docs": metadata.get("use_existing_docs", True),
            "enable_images": metadata.get("enable_images", False),
            "enable_quality_review": metadata.get("enable_quality_review", True),
        }

        # Include style guide if present
        if "style_guide" in metadata:
            settings["style_guide"] = {
                "tone": metadata["style_guide"].get("tone"),
                "vocabulary_level": metadata["style_guide"].get("vocabulary_level"),
                "structure_pattern": metadata["style_guide"].get("structure_pattern"),
            }

        # Include quality threshold if present
        if "quality_threshold" in metadata:
            settings["quality_threshold"] = metadata["quality_threshold"]

        return settings

    async def _find_similar_template(
        self,
        settings: Dict[str, Any],
    ) -> Optional[GenerationTemplate]:
        """Find an existing template with similar settings."""
        # Get all templates
        query = select(GenerationTemplate).where(
            or_(
                GenerationTemplate.is_system == True,
                GenerationTemplate.is_public == True,
            )
        )
        result = await self.db.execute(query)
        templates = result.scalars().all()

        # Compare settings similarity
        for template in templates:
            if not template.settings:
                continue

            similarity = self._calculate_settings_similarity(
                settings,
                template.settings,
            )

            if similarity > 0.85:  # 85% similarity threshold
                return template

        return None

    def _calculate_settings_similarity(
        self,
        settings1: Dict[str, Any],
        settings2: Dict[str, Any],
    ) -> float:
        """Calculate similarity score between two settings dicts."""
        if not settings1 or not settings2:
            return 0.0

        # Key settings to compare
        key_fields = [
            "output_format",
            "theme",
            "font_family",
            "layout_template",
            "include_toc",
            "include_sources",
            "use_existing_docs",
            "enable_images",
        ]

        matches = 0
        total = 0

        for field in key_fields:
            if field in settings1 or field in settings2:
                total += 1
                if settings1.get(field) == settings2.get(field):
                    matches += 1

        return matches / total if total > 0 else 0.0

    async def _generate_template_suggestion(
        self,
        generation_id: str,
        settings: Dict[str, Any],
        metadata: Dict[str, Any],
        rating: int,
    ) -> TemplateSuggestion:
        """Generate a template suggestion from generation data."""
        # Determine category based on output format
        output_format = settings.get("output_format", "docx")
        if output_format == "pptx":
            category = TemplateCategory.PRESENTATION.value
        elif output_format == "md":
            category = TemplateCategory.DOCUMENTATION.value
        else:
            category = TemplateCategory.CUSTOM.value

        # Generate name based on settings
        theme = settings.get("theme", "business")
        name_parts = [theme.title()]
        if output_format == "pptx":
            name_parts.append("Presentation")
        elif output_format == "docx":
            name_parts.append("Document")
        elif output_format == "md":
            name_parts.append("Markdown")
        else:
            name_parts.append(output_format.upper())

        name = " ".join(name_parts)

        # Generate description
        desc_parts = []
        if settings.get("include_sources"):
            desc_parts.append("with source citations")
        if settings.get("enable_images"):
            desc_parts.append("including images")
        if settings.get("include_toc"):
            desc_parts.append("with table of contents")

        description = f"Auto-generated {name.lower()} template"
        if desc_parts:
            description += " " + ", ".join(desc_parts)
        description += f". Rated {rating}/5 by user."

        # Generate tags
        tags = [theme, output_format]
        if settings.get("include_sources"):
            tags.append("sourced")
        if settings.get("enable_images"):
            tags.append("visual")
        if settings.get("enable_quality_review"):
            tags.append("quality-checked")

        # Calculate confidence based on rating
        confidence = (rating - 3) * 0.25 + 0.5  # 4->0.75, 5->1.0

        return TemplateSuggestion(
            name=name,
            description=description,
            category=category,
            settings=settings,
            confidence_score=confidence,
            based_on_generation_id=generation_id,
            tags=tags,
        )

    async def create_template_from_suggestion(
        self,
        suggestion: TemplateSuggestion,
        user_id: uuid.UUID,
        custom_name: Optional[str] = None,
        is_public: bool = False,
    ) -> Optional[GenerationTemplate]:
        """
        Create a new template from a suggestion.

        Args:
            suggestion: The template suggestion
            user_id: User ID to own the template
            custom_name: Optional custom name override
            is_public: Whether to make the template public

        Returns:
            Created template or None if it was a duplicate
        """
        if suggestion.is_duplicate_of:
            logger.info(
                "Skipping template creation - duplicate exists",
                duplicate_id=str(suggestion.is_duplicate_of),
            )
            return None

        template = await self._template_service.create_template(
            user_id=user_id,
            name=custom_name or suggestion.name,
            settings=suggestion.settings,
            description=suggestion.description,
            category=suggestion.category,
            is_public=is_public,
            tags=suggestion.tags,
        )

        logger.info(
            "Created template from learning",
            template_id=str(template.id),
            name=template.name,
            confidence_score=suggestion.confidence_score,
        )

        return template

    async def get_recommended_templates(
        self,
        user_id: uuid.UUID,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[GenerationTemplate]:
        """
        Get recommended templates for a user based on their history.

        Args:
            user_id: User's ID
            context: Optional context (title, description, output_format)
            limit: Maximum recommendations

        Returns:
            List of recommended templates
        """
        # Get user's templates ordered by usage
        user_templates = await self._template_service.get_user_templates(
            user_id=user_id,
            limit=limit,
        )

        # Get popular system templates
        popular = await self._template_service.get_popular_templates(
            limit=limit,
        )

        # If context provided, filter by relevance
        if context:
            output_format = context.get("output_format")
            if output_format:
                # Prioritize matching format
                user_templates = [
                    t for t in user_templates
                    if t.settings and t.settings.get("output_format") == output_format
                ] + [
                    t for t in user_templates
                    if not t.settings or t.settings.get("output_format") != output_format
                ]

                popular = [
                    t for t in popular
                    if t.settings and t.settings.get("output_format") == output_format
                ] + [
                    t for t in popular
                    if not t.settings or t.settings.get("output_format") != output_format
                ]

        # Combine and deduplicate
        seen_ids = set()
        recommendations = []

        for template in user_templates + popular:
            if template.id not in seen_ids and len(recommendations) < limit:
                seen_ids.add(template.id)
                recommendations.append(template)

        return recommendations


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


async def suggest_template_from_generation(
    db: AsyncSession,
    generation_id: str,
    user_rating: int,
    generation_settings: Dict[str, Any],
    generation_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[TemplateSuggestion]:
    """
    Suggest creating a template from a successful generation.

    Convenience function wrapping TemplateLearningService.

    Args:
        db: Database session
        generation_id: ID of the generation
        user_rating: User's rating (1-5)
        generation_settings: Settings used for generation
        generation_metadata: Additional metadata

    Returns:
        TemplateSuggestion if appropriate, None otherwise
    """
    service = TemplateLearningService(db)
    return await service.analyze_generation_for_template(
        generation_id=generation_id,
        user_rating=user_rating,
        generation_settings=generation_settings,
        generation_metadata=generation_metadata,
    )
