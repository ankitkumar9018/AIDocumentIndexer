"""
AIDocumentIndexer - Prompt Templates Service
==============================================

Service for managing reusable prompt templates.
Users can save prompts with predefined LLM settings for quick reuse.

Features:
- User-owned and system-wide templates
- Public/private visibility
- Template variables with placeholders
- Category-based organization
- Usage tracking
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import PromptTemplate

logger = structlog.get_logger(__name__)


@dataclass
class TemplateVariable:
    """Template variable definition."""
    name: str
    description: str
    default: str = ""
    required: bool = False


@dataclass
class TemplateInfo:
    """Template information for listing."""
    id: str
    name: str
    description: Optional[str]
    category: str
    is_public: bool
    is_system: bool
    is_owner: bool
    use_count: int
    created_at: datetime
    model_id: Optional[str] = None
    temperature: Optional[float] = None


class PromptTemplateService:
    """
    Service for managing prompt templates.

    Usage:
        service = PromptTemplateService()

        # Create a template
        template_id = await service.create_template(
            db, user_id, "Summarize", "Summarize the following text...",
            category="summarization", is_public=True
        )

        # List user's templates
        templates = await service.list_templates(db, user_id)

        # Apply a template
        rendered = await service.apply_template(
            db, template_id, variables={"topic": "AI"}
        )
    """

    # Default categories
    DEFAULT_CATEGORIES = [
        "general",
        "summarization",
        "analysis",
        "coding",
        "writing",
        "translation",
        "qa",
        "creative",
    ]

    # Variable pattern for template rendering
    VARIABLE_PATTERN = re.compile(r'\{\{(\w+)\}\}')

    async def create_template(
        self,
        db: AsyncSession,
        user_id: Optional[str],
        name: str,
        prompt_text: str,
        description: Optional[str] = None,
        category: str = "general",
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_public: bool = False,
        is_system: bool = False,
        tags: Optional[List[str]] = None,
        variables: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Create a new prompt template.

        Args:
            db: Database session
            user_id: Owner user ID (None for system templates)
            name: Template name
            prompt_text: The prompt text (can include {{variables}})
            description: Optional description
            category: Template category
            system_prompt: Optional system prompt
            model_id: Optional preferred model
            temperature: Optional temperature setting
            max_tokens: Optional max tokens setting
            is_public: Whether template is visible to all users
            is_system: Whether this is a system template
            tags: Optional list of tags
            variables: Optional list of variable definitions

        Returns:
            Template ID if successful, None otherwise
        """
        try:
            # Auto-detect variables from prompt text if not provided
            if variables is None:
                variables = self._extract_variables(prompt_text)

            template = PromptTemplate(
                user_id=uuid.UUID(user_id) if user_id else None,
                name=name,
                description=description,
                category=category,
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                is_public=is_public,
                is_system=is_system,
                tags=tags,
                variables=variables,
            )

            db.add(template)
            await db.commit()
            await db.refresh(template)

            logger.info(
                "Created prompt template",
                template_id=str(template.id),
                name=name,
                category=category,
                is_public=is_public,
            )

            return str(template.id)

        except Exception as e:
            logger.error("Failed to create template", error=str(e))
            await db.rollback()
            return None

    async def get_template(
        self,
        db: AsyncSession,
        template_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a template by ID.

        Args:
            db: Database session
            template_id: Template UUID
            user_id: Optional user ID for ownership check

        Returns:
            Template data dictionary or None
        """
        try:
            result = await db.execute(
                select(PromptTemplate).where(PromptTemplate.id == uuid.UUID(template_id))
            )
            template = result.scalar_one_or_none()

            if not template:
                return None

            # Check access permission
            if not self._can_access_template(template, user_id):
                logger.warning(
                    "Access denied to template",
                    template_id=template_id,
                    user_id=user_id,
                )
                return None

            return self._template_to_dict(template, user_id)

        except Exception as e:
            logger.error("Failed to get template", error=str(e))
            return None

    async def update_template(
        self,
        db: AsyncSession,
        template_id: str,
        user_id: str,
        name: Optional[str] = None,
        prompt_text: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        variables: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Update an existing template.

        Args:
            db: Database session
            template_id: Template UUID
            user_id: User ID for ownership verification
            ... (other params same as create)

        Returns:
            True if successful
        """
        try:
            result = await db.execute(
                select(PromptTemplate).where(PromptTemplate.id == uuid.UUID(template_id))
            )
            template = result.scalar_one_or_none()

            if not template:
                return False

            # Check ownership (only owner can update, unless system admin)
            if template.user_id and str(template.user_id) != user_id:
                logger.warning(
                    "User cannot update template they don't own",
                    template_id=template_id,
                    user_id=user_id,
                )
                return False

            # Update fields
            if name is not None:
                template.name = name
            if prompt_text is not None:
                template.prompt_text = prompt_text
                # Re-extract variables if prompt changed
                if variables is None:
                    template.variables = self._extract_variables(prompt_text)
            if description is not None:
                template.description = description
            if category is not None:
                template.category = category
            if system_prompt is not None:
                template.system_prompt = system_prompt
            if model_id is not None:
                template.model_id = model_id
            if temperature is not None:
                template.temperature = temperature
            if max_tokens is not None:
                template.max_tokens = max_tokens
            if is_public is not None:
                template.is_public = is_public
            if tags is not None:
                template.tags = tags
            if variables is not None:
                template.variables = variables

            await db.commit()

            logger.info(
                "Updated prompt template",
                template_id=template_id,
            )

            return True

        except Exception as e:
            logger.error("Failed to update template", error=str(e))
            await db.rollback()
            return False

    async def delete_template(
        self,
        db: AsyncSession,
        template_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a template.

        Args:
            db: Database session
            template_id: Template UUID
            user_id: User ID for ownership verification

        Returns:
            True if successful
        """
        try:
            result = await db.execute(
                select(PromptTemplate).where(PromptTemplate.id == uuid.UUID(template_id))
            )
            template = result.scalar_one_or_none()

            if not template:
                return False

            # Check ownership
            if template.user_id and str(template.user_id) != user_id:
                logger.warning(
                    "User cannot delete template they don't own",
                    template_id=template_id,
                    user_id=user_id,
                )
                return False

            # Cannot delete system templates via API
            if template.is_system:
                logger.warning(
                    "Cannot delete system template",
                    template_id=template_id,
                )
                return False

            await db.execute(
                delete(PromptTemplate).where(PromptTemplate.id == uuid.UUID(template_id))
            )
            await db.commit()

            logger.info("Deleted prompt template", template_id=template_id)
            return True

        except Exception as e:
            logger.error("Failed to delete template", error=str(e))
            await db.rollback()
            return False

    async def list_templates(
        self,
        db: AsyncSession,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        include_public: bool = True,
        include_system: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TemplateInfo]:
        """
        List templates accessible to a user.

        Args:
            db: Database session
            user_id: User ID (shows their templates + public)
            category: Filter by category
            search: Search in name and description
            include_public: Include public templates
            include_system: Include system templates
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of TemplateInfo objects
        """
        try:
            # Build query
            query = select(PromptTemplate)

            # Access filter
            conditions = []
            if user_id:
                conditions.append(PromptTemplate.user_id == uuid.UUID(user_id))
            if include_public:
                conditions.append(PromptTemplate.is_public == True)
            if include_system:
                conditions.append(PromptTemplate.is_system == True)

            if conditions:
                query = query.where(or_(*conditions))

            # Category filter
            if category:
                query = query.where(PromptTemplate.category == category)

            # Search filter
            if search:
                safe = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                search_pattern = f"%{safe}%"
                query = query.where(
                    or_(
                        PromptTemplate.name.ilike(search_pattern),
                        PromptTemplate.description.ilike(search_pattern),
                    )
                )

            # Order by use count (most popular first), then created_at
            query = query.order_by(
                PromptTemplate.use_count.desc(),
                PromptTemplate.created_at.desc(),
            )

            # Pagination
            query = query.limit(limit).offset(offset)

            result = await db.execute(query)
            templates = result.scalars().all()

            return [
                TemplateInfo(
                    id=str(t.id),
                    name=t.name,
                    description=t.description,
                    category=t.category,
                    is_public=t.is_public,
                    is_system=t.is_system,
                    is_owner=user_id and str(t.user_id) == user_id if t.user_id else False,
                    use_count=t.use_count,
                    created_at=t.created_at,
                    model_id=t.model_id,
                    temperature=t.temperature,
                )
                for t in templates
            ]

        except Exception as e:
            logger.error("Failed to list templates", error=str(e))
            return []

    async def get_categories(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Get all template categories with counts.

        Args:
            db: Database session

        Returns:
            List of category info with counts
        """
        try:
            result = await db.execute(
                select(
                    PromptTemplate.category,
                    func.count(PromptTemplate.id).label("count"),
                )
                .group_by(PromptTemplate.category)
                .order_by(func.count(PromptTemplate.id).desc())
            )

            categories = []
            for row in result:
                categories.append({
                    "name": row.category,
                    "count": row.count,
                })

            # Add any default categories not in DB
            existing = {c["name"] for c in categories}
            for default_cat in self.DEFAULT_CATEGORIES:
                if default_cat not in existing:
                    categories.append({"name": default_cat, "count": 0})

            return categories

        except Exception as e:
            logger.error("Failed to get categories", error=str(e))
            return [{"name": cat, "count": 0} for cat in self.DEFAULT_CATEGORIES]

    async def apply_template(
        self,
        db: AsyncSession,
        template_id: str,
        user_id: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply a template and return rendered prompt with settings.

        Args:
            db: Database session
            template_id: Template UUID
            user_id: User ID for access check
            variables: Variable values to substitute

        Returns:
            Dictionary with rendered prompt and settings
        """
        try:
            template = await self.get_template(db, template_id, user_id)
            if not template:
                return None

            # Render prompt with variables
            rendered_prompt = self._render_template(
                template["prompt_text"],
                variables or {},
                template.get("variables", []),
            )

            rendered_system = None
            if template.get("system_prompt"):
                rendered_system = self._render_template(
                    template["system_prompt"],
                    variables or {},
                    template.get("variables", []),
                )

            # Update use count
            await db.execute(
                update(PromptTemplate)
                .where(PromptTemplate.id == uuid.UUID(template_id))
                .values(
                    use_count=PromptTemplate.use_count + 1,
                    last_used_at=datetime.utcnow(),
                )
            )
            await db.commit()

            return {
                "prompt_text": rendered_prompt,
                "system_prompt": rendered_system,
                "model_id": template.get("model_id"),
                "temperature": template.get("temperature"),
                "max_tokens": template.get("max_tokens"),
                "template_id": template_id,
                "template_name": template["name"],
            }

        except Exception as e:
            logger.error("Failed to apply template", error=str(e))
            return None

    async def duplicate_template(
        self,
        db: AsyncSession,
        template_id: str,
        user_id: str,
        new_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Duplicate a template for a user.

        Args:
            db: Database session
            template_id: Source template UUID
            user_id: User who will own the copy
            new_name: Optional new name (defaults to "Copy of ...")

        Returns:
            New template ID if successful
        """
        try:
            template = await self.get_template(db, template_id, user_id)
            if not template:
                return None

            name = new_name or f"Copy of {template['name']}"

            return await self.create_template(
                db,
                user_id=user_id,
                name=name,
                prompt_text=template["prompt_text"],
                description=template.get("description"),
                category=template.get("category", "general"),
                system_prompt=template.get("system_prompt"),
                model_id=template.get("model_id"),
                temperature=template.get("temperature"),
                max_tokens=template.get("max_tokens"),
                is_public=False,  # Copies are private by default
                is_system=False,
                tags=template.get("tags"),
                variables=template.get("variables"),
            )

        except Exception as e:
            logger.error("Failed to duplicate template", error=str(e))
            return None

    def _extract_variables(self, prompt_text: str) -> List[Dict[str, Any]]:
        """Extract variable definitions from prompt text."""
        matches = self.VARIABLE_PATTERN.findall(prompt_text)
        seen = set()
        variables = []

        for var_name in matches:
            if var_name not in seen:
                seen.add(var_name)
                variables.append({
                    "name": var_name,
                    "description": f"Value for {var_name}",
                    "default": "",
                    "required": True,
                })

        return variables

    def _render_template(
        self,
        template_text: str,
        values: Dict[str, str],
        variable_defs: List[Dict[str, Any]],
    ) -> str:
        """Render template with variable substitution."""
        result = template_text

        # Apply provided values
        for var_name, var_value in values.items():
            result = result.replace(f"{{{{{var_name}}}}}", var_value)

        # Apply defaults for missing variables
        for var_def in variable_defs:
            var_name = var_def["name"]
            if var_name not in values and var_def.get("default"):
                result = result.replace(f"{{{{{var_name}}}}}", var_def["default"])

        return result

    def _can_access_template(
        self,
        template: PromptTemplate,
        user_id: Optional[str],
    ) -> bool:
        """Check if user can access a template."""
        # Public and system templates are accessible to everyone
        if template.is_public or template.is_system:
            return True

        # Private templates only accessible to owner
        if template.user_id and user_id:
            return str(template.user_id) == user_id

        return False

    def _template_to_dict(
        self,
        template: PromptTemplate,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert template model to dictionary."""
        return {
            "id": str(template.id),
            "user_id": str(template.user_id) if template.user_id else None,
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "tags": template.tags,
            "prompt_text": template.prompt_text,
            "system_prompt": template.system_prompt,
            "model_id": template.model_id,
            "temperature": template.temperature,
            "max_tokens": template.max_tokens,
            "is_public": template.is_public,
            "is_system": template.is_system,
            "is_owner": user_id and str(template.user_id) == user_id if template.user_id else False,
            "use_count": template.use_count,
            "last_used_at": template.last_used_at.isoformat() if template.last_used_at else None,
            "variables": template.variables or [],
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat(),
        }


# Singleton instance
_template_service: Optional[PromptTemplateService] = None


def get_prompt_template_service() -> PromptTemplateService:
    """Get the prompt template service singleton."""
    global _template_service
    if _template_service is None:
        _template_service = PromptTemplateService()
    return _template_service
