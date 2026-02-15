"""
AIDocumentIndexer - Document Templates API Routes
==================================================

API endpoints for browsing built-in document templates (PPTX, DOCX, XLSX).
These are file templates for document generation, not prompt templates.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/document-templates", tags=["Document Templates"])

# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "templates"


def _validate_template_path_segment(segment: str, name: str) -> str:
    """Validate a template path segment contains only safe characters (prevent path traversal)."""
    if not re.match(r'^[a-zA-Z0-9_.-]+$', segment):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {name}",
        )
    return segment


def _validate_resolved_path(path: Path) -> None:
    """Ensure a resolved path stays within the templates directory."""
    try:
        path.resolve().relative_to(TEMPLATES_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid template path",
        )


# =============================================================================
# Response Models
# =============================================================================

class TemplateMetadata(BaseModel):
    """Metadata for a document template."""
    id: str
    name: str
    description: str
    category: str
    file_type: str  # pptx, docx, xlsx
    tags: List[str] = []
    primary_color: str = "#000000"
    style: str = "professional"
    tone: str = "formal"
    recommended_slides: Optional[int] = None  # For PPTX
    max_bullet_chars: Optional[int] = None
    supports_images: bool = False
    preview_url: Optional[str] = None


class TemplateCategory(BaseModel):
    """Category with templates."""
    name: str
    display_name: str
    templates: List[TemplateMetadata]


class TemplatesByFileType(BaseModel):
    """Templates grouped by file type."""
    file_type: str
    display_name: str
    categories: List[TemplateCategory]
    total_count: int


class ExternalTemplateSource(BaseModel):
    """External template source reference."""
    name: str
    url: str
    description: str
    file_types: List[str]


class TemplateListResponse(BaseModel):
    """Response for listing all templates."""
    templates: List[TemplateMetadata]
    total: int
    file_types: List[str]
    categories: List[str]


class ExternalSourcesResponse(BaseModel):
    """Response for external template sources."""
    sources: List[ExternalTemplateSource]


# =============================================================================
# Helper Functions
# =============================================================================

def get_category_display_name(category: str) -> str:
    """Get display name for a category."""
    display_names = {
        "corporate": "Corporate",
        "creative": "Creative",
        "academic": "Academic",
        "pitch": "Pitch Decks",
        "reports": "Reports",
        "proposals": "Proposals",
        "letters": "Letters",
        "financial": "Financial",
        "project": "Project Management",
        "data": "Data & Analytics",
    }
    return display_names.get(category, category.title())


def get_file_type_display_name(file_type: str) -> str:
    """Get display name for a file type."""
    display_names = {
        "pptx": "PowerPoint Presentations",
        "docx": "Word Documents",
        "xlsx": "Excel Spreadsheets",
        "pdf": "PDF Documents",
    }
    return display_names.get(file_type, file_type.upper())


def load_template_metadata(json_path: Path) -> Optional[TemplateMetadata]:
    """Load template metadata from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Add preview URL based on template location
            template_id = data.get("id", "")
            file_type = data.get("file_type", "")
            category = data.get("category", "")
            data["preview_url"] = f"/api/v1/document-templates/{file_type}/{category}/{template_id}/preview"
            return TemplateMetadata(**data)
    except Exception as e:
        logger.warning("Failed to load template metadata", path=str(json_path), error=str(e))
        return None


def scan_templates() -> List[TemplateMetadata]:
    """Scan templates directory and return all template metadata."""
    templates = []

    if not TEMPLATES_DIR.exists():
        return templates

    # Scan each file type directory
    for file_type_dir in TEMPLATES_DIR.iterdir():
        if not file_type_dir.is_dir():
            continue

        file_type = file_type_dir.name

        # Scan each category directory
        for category_dir in file_type_dir.iterdir():
            if not category_dir.is_dir():
                continue

            # Find all JSON metadata files
            for json_file in category_dir.glob("*.json"):
                metadata = load_template_metadata(json_file)
                if metadata:
                    templates.append(metadata)

    return templates


# =============================================================================
# External Template Sources (for "View More" links)
# =============================================================================

EXTERNAL_SOURCES = [
    ExternalTemplateSource(
        name="Slidesgo",
        url="https://slidesgo.com/",
        description="3000+ free presentation themes and Google Slides templates",
        file_types=["pptx"]
    ),
    ExternalTemplateSource(
        name="SlidesCarnival",
        url="https://www.slidescarnival.com/",
        description="Free professional PowerPoint and Google Slides templates",
        file_types=["pptx"]
    ),
    ExternalTemplateSource(
        name="PresentationGO",
        url="https://www.presentationgo.com/",
        description="3300+ free PowerPoint templates and diagrams",
        file_types=["pptx"]
    ),
    ExternalTemplateSource(
        name="24Slides",
        url="https://24slides.com/templates/",
        description="Weekly updated professional presentation templates",
        file_types=["pptx"]
    ),
    ExternalTemplateSource(
        name="Microsoft Word Templates",
        url="https://create.microsoft.com/en-us/templates/word",
        description="Official Microsoft Word templates for documents",
        file_types=["docx"]
    ),
    ExternalTemplateSource(
        name="Template.net",
        url="https://www.template.net/editable/word",
        description="80,000+ Word document templates",
        file_types=["docx"]
    ),
    ExternalTemplateSource(
        name="Microsoft Excel Templates",
        url="https://create.microsoft.com/en-us/templates/excel",
        description="Official Microsoft Excel spreadsheet templates",
        file_types=["xlsx"]
    ),
    ExternalTemplateSource(
        name="Vertex42",
        url="https://www.vertex42.com/ExcelTemplates/",
        description="Free Excel templates for business and personal use",
        file_types=["xlsx"]
    ),
    ExternalTemplateSource(
        name="Smartsheet",
        url="https://www.smartsheet.com/32-free-excel-spreadsheet-templates",
        description="Free Excel spreadsheet templates for project management",
        file_types=["xlsx"]
    ),
]


# =============================================================================
# Routes
# =============================================================================

@router.get("", response_model=TemplateListResponse)
async def list_templates(
    file_type: Optional[str] = Query(None, description="Filter by file type (pptx, docx, xlsx)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in name, description, tags"),
):
    """
    List all available built-in document templates.

    Optionally filter by file type, category, or search term.
    """
    templates = scan_templates()

    # Apply filters
    if file_type:
        templates = [t for t in templates if t.file_type == file_type]

    if category:
        templates = [t for t in templates if t.category == category]

    if search:
        search_lower = search.lower()
        templates = [
            t for t in templates
            if search_lower in t.name.lower()
            or search_lower in t.description.lower()
            or any(search_lower in tag.lower() for tag in t.tags)
        ]

    # Get unique file types and categories
    all_templates = scan_templates()
    file_types = sorted(set(t.file_type for t in all_templates))
    categories = sorted(set(t.category for t in all_templates))

    return TemplateListResponse(
        templates=templates,
        total=len(templates),
        file_types=file_types,
        categories=categories,
    )


@router.get("/by-type", response_model=List[TemplatesByFileType])
async def list_templates_by_type():
    """
    List all templates grouped by file type and category.

    Returns a hierarchical structure for easy display.
    """
    templates = scan_templates()

    # Group by file type
    by_type: dict = {}
    for template in templates:
        ft = template.file_type
        if ft not in by_type:
            by_type[ft] = {}

        cat = template.category
        if cat not in by_type[ft]:
            by_type[ft][cat] = []

        by_type[ft][cat].append(template)

    # Build response
    result = []
    for file_type in sorted(by_type.keys()):
        categories = []
        for category in sorted(by_type[file_type].keys()):
            categories.append(TemplateCategory(
                name=category,
                display_name=get_category_display_name(category),
                templates=by_type[file_type][category],
            ))

        result.append(TemplatesByFileType(
            file_type=file_type,
            display_name=get_file_type_display_name(file_type),
            categories=categories,
            total_count=sum(len(c.templates) for c in categories),
        ))

    return result


@router.get("/external-sources", response_model=ExternalSourcesResponse)
async def get_external_sources(
    file_type: Optional[str] = Query(None, description="Filter by file type"),
):
    """
    Get external template sources for "View More" links.

    These are third-party websites with additional templates.
    """
    sources = EXTERNAL_SOURCES

    if file_type:
        sources = [s for s in sources if file_type in s.file_types]

    return ExternalSourcesResponse(sources=sources)


@router.get("/{file_type}/{category}/{template_id}")
async def get_template_metadata(
    file_type: str,
    category: str,
    template_id: str,
):
    """
    Get metadata for a specific template.
    """
    file_type = _validate_template_path_segment(file_type, "file type")
    category = _validate_template_path_segment(category, "category")
    template_id = _validate_template_path_segment(template_id, "template ID")

    json_path = TEMPLATES_DIR / file_type / category / f"{template_id.replace(f'{category}-', '')}.json"
    _validate_resolved_path(json_path)

    # Try alternative path patterns
    if not json_path.exists():
        # Try with full template_id as filename
        for json_file in (TEMPLATES_DIR / file_type / category).glob("*.json"):
            metadata = load_template_metadata(json_file)
            if metadata and metadata.id == template_id:
                return metadata

    if json_path.exists():
        metadata = load_template_metadata(json_path)
        if metadata:
            return metadata

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")


@router.get("/{file_type}/{category}/{template_id}/download")
async def download_template(
    file_type: str,
    category: str,
    template_id: str,
):
    """
    Download the actual template file.
    """
    file_type = _validate_template_path_segment(file_type, "file type")
    category = _validate_template_path_segment(category, "category")
    template_id = _validate_template_path_segment(template_id, "template ID")

    # Find the template file
    template_dir = TEMPLATES_DIR / file_type / category

    if not template_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template category not found")

    # Look for the template file
    template_name = template_id.replace(f"{category}-", "")
    template_file = template_dir / f"{template_name}.{file_type}"

    if not template_file.exists():
        # Try finding by scanning JSON files
        for json_file in template_dir.glob("*.json"):
            metadata = load_template_metadata(json_file)
            if metadata and metadata.id == template_id:
                template_file = json_file.with_suffix(f".{file_type}")
                break

    if not template_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template file not found")

    _validate_resolved_path(template_file)

    return FileResponse(
        path=str(template_file),
        filename=template_file.name,
        media_type="application/octet-stream",
    )


@router.get("/{file_type}/{category}/{template_id}/preview")
async def get_template_preview(
    file_type: str,
    category: str,
    template_id: str,
):
    """
    Get a preview image for the template.

    Returns a PNG preview if available, otherwise a placeholder response.
    """
    file_type = _validate_template_path_segment(file_type, "file type")
    category = _validate_template_path_segment(category, "category")
    template_id = _validate_template_path_segment(template_id, "template ID")

    template_dir = TEMPLATES_DIR / file_type / category

    if not template_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template category not found")

    # Look for preview image
    template_name = template_id.replace(f"{category}-", "")
    preview_file = template_dir / f"{template_name}.png"

    if not preview_file.exists():
        preview_file = template_dir / f"{template_name}_preview.png"

    if preview_file.exists():
        _validate_resolved_path(preview_file)
        return FileResponse(
            path=str(preview_file),
            media_type="image/png",
        )

    # No preview available - return 204 No Content
    return {"message": "No preview available", "template_id": template_id}
