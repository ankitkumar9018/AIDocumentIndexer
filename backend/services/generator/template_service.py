"""
Template Service

Manages the templates library for document generation.
Provides listing, selection, and metadata for built-in templates.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Default templates directory
DEFAULT_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "data", "templates"
)


@dataclass
class TemplateMetadata:
    """Metadata for a document template."""

    id: str
    name: str
    description: str = ""
    category: str = "general"  # corporate, creative, academic, pitch, etc.
    file_type: str = "pptx"  # pptx, docx, pdf, xlsx
    preview_image: str = ""  # Path to thumbnail
    tags: List[str] = field(default_factory=list)

    # Design characteristics
    primary_color: str = "#2563EB"
    style: str = "professional"  # minimal, bold, professional, etc.
    tone: str = "formal"  # formal, casual, modern, classic

    # Constraints
    recommended_slides: int = 10
    max_bullet_chars: int = 70
    supports_images: bool = True

    # File info
    file_path: str = ""
    file_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "file_type": self.file_type,
            "preview_image": self.preview_image,
            "tags": self.tags,
            "primary_color": self.primary_color,
            "style": self.style,
            "tone": self.tone,
            "recommended_slides": self.recommended_slides,
            "max_bullet_chars": self.max_bullet_chars,
            "supports_images": self.supports_images,
            "file_path": self.file_path,
            "file_size": self.file_size,
        }


class TemplateService:
    """Service for managing document templates.

    Provides:
    - Listing available templates by type and category
    - Getting template metadata
    - Recommending templates based on content
    - Importing custom templates
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize the template service.

        Args:
            templates_dir: Directory containing templates (default: data/templates)
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self._cache: Dict[str, TemplateMetadata] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure the template cache is initialized."""
        if not self._initialized:
            self._scan_templates()
            self._initialized = True

    def _scan_templates(self):
        """Scan the templates directory and build cache."""
        self._cache.clear()

        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        # File type directories
        file_types = ["pptx", "docx", "xlsx", "pdf"]

        for file_type in file_types:
            type_dir = os.path.join(self.templates_dir, file_type)
            if not os.path.exists(type_dir):
                continue

            # Scan category subdirectories
            for category in os.listdir(type_dir):
                category_dir = os.path.join(type_dir, category)
                if not os.path.isdir(category_dir):
                    continue

                # Scan templates in category
                for item in os.listdir(category_dir):
                    item_path = os.path.join(category_dir, item)

                    # Check if it's a template file
                    if item.endswith(f".{file_type}"):
                        metadata = self._load_template_metadata(
                            item_path, file_type, category
                        )
                        if metadata:
                            self._cache[metadata.id] = metadata

                    # Check for template.json manifest
                    manifest_path = os.path.join(item_path, "template.json")
                    if os.path.isdir(item_path) and os.path.exists(manifest_path):
                        metadata = self._load_manifest(manifest_path, file_type, category)
                        if metadata:
                            self._cache[metadata.id] = metadata

        logger.info(f"Scanned {len(self._cache)} templates")

    def _load_template_metadata(
        self,
        file_path: str,
        file_type: str,
        category: str,
    ) -> Optional[TemplateMetadata]:
        """Load metadata for a template file.

        First checks for accompanying template.json, then derives from filename.
        """
        # Check for manifest file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        manifest_path = os.path.join(
            os.path.dirname(file_path),
            f"{base_name}.json"
        )

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)

                return TemplateMetadata(
                    id=data.get("id", base_name),
                    name=data.get("name", base_name.replace("_", " ").title()),
                    description=data.get("description", ""),
                    category=data.get("category", category),
                    file_type=file_type,
                    preview_image=data.get("preview_image", ""),
                    tags=data.get("tags", []),
                    primary_color=data.get("primary_color", "#2563EB"),
                    style=data.get("style", "professional"),
                    tone=data.get("tone", "formal"),
                    recommended_slides=data.get("recommended_slides", 10),
                    max_bullet_chars=data.get("max_bullet_chars", 70),
                    supports_images=data.get("supports_images", True),
                    file_path=file_path,
                    file_size=os.path.getsize(file_path),
                )
            except Exception as e:
                logger.warning(f"Could not load manifest {manifest_path}: {e}")

        # Derive metadata from filename
        return TemplateMetadata(
            id=f"{file_type}-{category}-{base_name}",
            name=base_name.replace("_", " ").replace("-", " ").title(),
            description=f"{category.title()} {file_type.upper()} template",
            category=category,
            file_type=file_type,
            tags=[category, file_type],
            file_path=file_path,
            file_size=os.path.getsize(file_path),
        )

    def _load_manifest(
        self,
        manifest_path: str,
        file_type: str,
        category: str,
    ) -> Optional[TemplateMetadata]:
        """Load template metadata from a manifest file."""
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)

            # Find the template file
            template_dir = os.path.dirname(manifest_path)
            template_file = None
            for f in os.listdir(template_dir):
                if f.endswith(f".{file_type}"):
                    template_file = os.path.join(template_dir, f)
                    break

            if not template_file:
                return None

            return TemplateMetadata(
                id=data.get("id", os.path.basename(template_dir)),
                name=data.get("name", "Unnamed Template"),
                description=data.get("description", ""),
                category=data.get("category", category),
                file_type=file_type,
                preview_image=data.get("preview_image", ""),
                tags=data.get("tags", []),
                primary_color=data.get("primary_color", "#2563EB"),
                style=data.get("style", "professional"),
                tone=data.get("tone", "formal"),
                recommended_slides=data.get("recommended_slides", 10),
                max_bullet_chars=data.get("max_bullet_chars", 70),
                supports_images=data.get("supports_images", True),
                file_path=template_file,
                file_size=os.path.getsize(template_file),
            )
        except Exception as e:
            logger.warning(f"Could not load manifest {manifest_path}: {e}")
            return None

    def list_templates(
        self,
        file_type: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[TemplateMetadata]:
        """List available templates with optional filtering.

        Args:
            file_type: Filter by document type (pptx, docx, etc.)
            category: Filter by category (corporate, creative, etc.)
            tags: Filter by tags

        Returns:
            List of matching template metadata
        """
        self._ensure_initialized()

        results = []
        for template in self._cache.values():
            # Apply filters
            if file_type and template.file_type != file_type:
                continue
            if category and template.category != category:
                continue
            if tags:
                if not any(tag in template.tags for tag in tags):
                    continue

            results.append(template)

        # Sort by name
        results.sort(key=lambda t: t.name)
        return results

    def get_template(self, template_id: str) -> Optional[TemplateMetadata]:
        """Get a template by ID.

        Args:
            template_id: Unique template identifier

        Returns:
            TemplateMetadata or None if not found
        """
        self._ensure_initialized()
        return self._cache.get(template_id)

    def get_template_path(self, template_id: str) -> Optional[str]:
        """Get the file path for a template.

        Args:
            template_id: Unique template identifier

        Returns:
            File path or None if not found
        """
        template = self.get_template(template_id)
        return template.file_path if template else None

    def get_categories(self, file_type: Optional[str] = None) -> List[str]:
        """Get available categories.

        Args:
            file_type: Optional filter by document type

        Returns:
            List of category names
        """
        self._ensure_initialized()

        categories = set()
        for template in self._cache.values():
            if file_type and template.file_type != file_type:
                continue
            categories.add(template.category)

        return sorted(list(categories))

    async def get_recommended_template(
        self,
        topic: str,
        file_type: str = "pptx",
        tone: Optional[str] = None,
    ) -> Optional[TemplateMetadata]:
        """Get a recommended template based on topic.

        Uses simple keyword matching. For more sophisticated recommendations,
        this could use an LLM to analyze the topic.

        Args:
            topic: Document topic
            file_type: Target document type
            tone: Optional desired tone

        Returns:
            Recommended template metadata
        """
        self._ensure_initialized()

        topic_lower = topic.lower()

        # Keyword to category mapping
        category_keywords = {
            "corporate": ["business", "company", "corporate", "enterprise", "professional"],
            "creative": ["design", "creative", "artistic", "visual", "modern"],
            "academic": ["research", "study", "university", "academic", "thesis", "paper"],
            "pitch": ["startup", "investor", "pitch", "funding", "venture"],
            "financial": ["budget", "financial", "revenue", "profit", "cost"],
            "project": ["project", "timeline", "milestone", "task", "planning"],
        }

        # Find best matching category
        best_category = None
        best_score = 0

        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in topic_lower)
            if score > best_score:
                best_score = score
                best_category = category

        # Get templates in that category
        templates = self.list_templates(file_type=file_type, category=best_category)

        # Filter by tone if specified
        if tone and templates:
            tone_matches = [t for t in templates if t.tone == tone]
            if tone_matches:
                templates = tone_matches

        # Return first match or None
        return templates[0] if templates else None

    def import_template(
        self,
        file_path: str,
        metadata: TemplateMetadata,
    ) -> bool:
        """Import a custom template.

        Args:
            file_path: Path to the template file
            metadata: Template metadata

        Returns:
            True if imported successfully
        """
        if not os.path.exists(file_path):
            logger.error(f"Template file not found: {file_path}")
            return False

        # Determine destination
        dest_dir = os.path.join(
            self.templates_dir,
            metadata.file_type,
            metadata.category
        )
        os.makedirs(dest_dir, exist_ok=True)

        # Copy template file
        import shutil
        dest_file = os.path.join(dest_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest_file)

        # Save manifest
        manifest_path = os.path.join(
            dest_dir,
            os.path.splitext(os.path.basename(file_path))[0] + ".json"
        )

        try:
            with open(manifest_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save manifest: {e}")
            return False

        # Update metadata with new path
        metadata.file_path = dest_file
        metadata.file_size = os.path.getsize(dest_file)

        # Add to cache
        self._cache[metadata.id] = metadata

        logger.info(f"Imported template: {metadata.id}")
        return True

    def refresh(self):
        """Refresh the template cache."""
        self._initialized = False
        self._scan_templates()


# Singleton instance
_template_service: Optional[TemplateService] = None


def get_template_service() -> TemplateService:
    """Get the singleton TemplateService instance."""
    global _template_service
    if _template_service is None:
        _template_service = TemplateService()
    return _template_service
