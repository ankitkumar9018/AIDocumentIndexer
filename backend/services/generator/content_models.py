"""
Content models with Pydantic validation for structured LLM output.

These models define the structure of generated content and enforce
template constraints through validation.

The content goes through a review stage where users can view and edit
each slide/page/section before final document rendering.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime


class ContentStatus(str, Enum):
    """Status of content in the review pipeline."""
    DRAFT = "draft"           # Initial generation
    PENDING_REVIEW = "pending_review"  # Awaiting user review
    APPROVED = "approved"     # User approved
    EDITED = "edited"         # User made edits
    REGENERATING = "regenerating"  # Being regenerated based on feedback
    FINAL = "final"           # Ready for document rendering


class EditAction(str, Enum):
    """Actions user can take on content."""
    APPROVE = "approve"           # Accept as is
    EDIT = "edit"                 # Direct edit
    REGENERATE = "regenerate"     # Ask LLM to regenerate
    ENHANCE = "enhance"           # Ask LLM to enhance
    SHORTEN = "shorten"           # Ask LLM to make more concise
    EXPAND = "expand"             # Ask LLM to add more detail
    CHANGE_TONE = "change_tone"   # Make more formal/casual
    DELETE = "delete"             # Remove this section


# =============================================================================
# PPTX Content Models
# =============================================================================

class BulletPoint(BaseModel):
    """A single bullet point with optional sub-bullets."""
    # NOTE: max_length is NOT set on Field because it would fail validation
    # BEFORE the validator can truncate. We handle length in the validator.
    text: str = Field(..., description="Bullet text (auto-truncated to 100 chars)")
    sub_bullets: List[str] = Field(default=[], description="Sub-bullet points (max 5)")

    @field_validator('text', mode='before')
    @classmethod
    def validate_text(cls, v):
        if not isinstance(v, str):
            v = str(v)
        if len(v) > 100:
            # Truncate with ellipsis instead of failing
            return v[:97] + "..."
        return v

    @field_validator('sub_bullets', mode='before')
    @classmethod
    def validate_sub_bullets(cls, v):
        if not isinstance(v, list):
            return []
        return [b[:77] + "..." if len(str(b)) > 80 else str(b) for b in v[:5]]


class SlideContent(BaseModel):
    """Content for a single slide with validation."""

    # Identification
    slide_number: int = Field(default=0, description="Slide position in presentation")
    slide_id: str = Field(default="", description="Unique identifier for this slide")

    # Content
    # NOTE: max_length removed from fields with truncation validators
    # to avoid validation errors before truncation can occur.
    layout: str = Field(
        default="title_content",
        description="Layout type: title_slide, title_only, title_content, two_column, image_text, blank"
    )
    title: str = Field(default="", description="Slide title (auto-truncated to 80 chars)")
    subtitle: Optional[str] = Field(default=None, description="Subtitle for title slides (auto-truncated to 120 chars)")
    bullets: List[BulletPoint] = Field(default=[], description="Bullet points (max 8)")
    body_text: Optional[str] = Field(default=None, description="Body text (auto-truncated to 600 chars)")
    speaker_notes: str = Field(default="", description="Speaker notes (auto-truncated to 500 chars)")

    # Image handling
    image_description: Optional[str] = Field(default=None, description="Image description (auto-truncated to 150 chars)")
    image_path: Optional[str] = Field(default=None, description="Path to existing image")

    # Two-column content
    left_column: Optional[List[BulletPoint]] = Field(default=None, description="Left column bullets")
    right_column: Optional[List[BulletPoint]] = Field(default=None, description="Right column bullets")

    # Review status
    status: ContentStatus = Field(default=ContentStatus.DRAFT)
    user_feedback: Optional[str] = Field(default=None, description="User feedback for regeneration")
    edit_history: List[Dict[str, Any]] = Field(default=[], description="History of edits")

    @field_validator('title', mode='before')
    @classmethod
    def validate_title(cls, v):
        if not v:
            return ""
        v = str(v)
        if len(v) > 80:
            return v[:77] + "..."
        return v

    @field_validator('subtitle', mode='before')
    @classmethod
    def validate_subtitle(cls, v):
        if not v:
            return None
        v = str(v)
        if len(v) > 120:
            return v[:117] + "..."
        return v

    @field_validator('body_text', mode='before')
    @classmethod
    def validate_body_text(cls, v):
        if not v:
            return None
        v = str(v)
        if len(v) > 600:
            return v[:597] + "..."
        return v

    @field_validator('speaker_notes', mode='before')
    @classmethod
    def validate_speaker_notes(cls, v):
        if not v:
            return ""
        v = str(v)
        if len(v) > 500:
            return v[:497] + "..."
        return v

    @field_validator('image_description', mode='before')
    @classmethod
    def validate_image_description(cls, v):
        if not v:
            return None
        v = str(v)
        if len(v) > 150:
            return v[:147] + "..."
        return v

    @field_validator('bullets', mode='before')
    @classmethod
    def validate_bullets(cls, v):
        if not v:
            return []
        # Limit to 8 bullets maximum
        return v[:8]

    def to_preview_dict(self) -> dict:
        """Convert to preview format for UI display.

        Returns data in format expected by frontend ContentReviewItem:
        - item_id (mapped from slide_id)
        - item_number (mapped from slide_number)
        - preview_text (generated from bullets)
        """
        # Generate preview text from bullets
        preview_parts = []
        for b in self.bullets[:3]:  # First 3 bullets for preview
            preview_parts.append(b.text[:50] + "..." if len(b.text) > 50 else b.text)
        preview_text = " | ".join(preview_parts) if preview_parts else (self.body_text or "")[:100]

        return {
            # Frontend ContentReviewItem expects these fields
            "item_id": self.slide_id,
            "item_number": self.slide_number,
            "title": self.title,
            "status": self.status.value,
            "preview_text": preview_text,
            # Also include original slide fields for backward compatibility
            "slide_number": self.slide_number,
            "slide_id": self.slide_id,
            "layout": self.layout,
            "subtitle": self.subtitle,
            "bullets": [{"text": b.text, "sub_bullets": b.sub_bullets} for b in self.bullets],
            "body_text": self.body_text,
            "speaker_notes": self.speaker_notes,
            "image_description": self.image_description,
            "char_counts": {
                "title": len(self.title),
                "bullets_total": sum(len(b.text) for b in self.bullets),
                "speaker_notes": len(self.speaker_notes),
            }
        }


class PresentationContent(BaseModel):
    """Complete presentation content with all slides."""

    # Metadata
    title: str = Field(..., description="Presentation title (auto-truncated to 100 chars)")
    subtitle: Optional[str] = Field(default=None, description="Presentation subtitle (auto-truncated to 150 chars)")
    author: str = Field(default="", description="Author name")
    created_at: datetime = Field(default_factory=datetime.now)

    # Content
    slides: List[SlideContent] = Field(..., min_length=1, description="Slides (max 50)")

    @field_validator('title', mode='before')
    @classmethod
    def validate_title(cls, v):
        if not v:
            return ""
        v = str(v)
        if len(v) > 100:
            return v[:97] + "..."
        return v

    @field_validator('subtitle', mode='before')
    @classmethod
    def validate_subtitle(cls, v):
        if not v:
            return None
        v = str(v)
        if len(v) > 150:
            return v[:147] + "..."
        return v

    @field_validator('slides', mode='before')
    @classmethod
    def validate_slides(cls, v):
        if not v:
            return []
        return v[:50]  # Limit to 50 slides

    # Review status
    overall_status: ContentStatus = Field(default=ContentStatus.DRAFT)

    # Template info
    template_id: Optional[str] = Field(default=None)
    theme_name: Optional[str] = Field(default=None)

    def get_slides_for_review(self) -> List[dict]:
        """Get all slides in preview format for review UI."""
        return [slide.to_preview_dict() for slide in self.slides]

    def get_pending_slides(self) -> List[SlideContent]:
        """Get slides that still need review."""
        return [s for s in self.slides if s.status in [ContentStatus.DRAFT, ContentStatus.PENDING_REVIEW]]

    def approve_slide(self, slide_number: int) -> None:
        """Mark a slide as approved."""
        for slide in self.slides:
            if slide.slide_number == slide_number:
                slide.status = ContentStatus.APPROVED
                break
        self._update_overall_status()

    def approve_all(self) -> None:
        """Approve all slides."""
        for slide in self.slides:
            slide.status = ContentStatus.APPROVED
        self.overall_status = ContentStatus.APPROVED

    def _update_overall_status(self) -> None:
        """Update overall status based on individual slide statuses."""
        statuses = [s.status for s in self.slides]
        if all(s in [ContentStatus.APPROVED, ContentStatus.FINAL] for s in statuses):
            self.overall_status = ContentStatus.APPROVED
        elif any(s == ContentStatus.REGENERATING for s in statuses):
            self.overall_status = ContentStatus.REGENERATING
        elif any(s == ContentStatus.EDITED for s in statuses):
            self.overall_status = ContentStatus.EDITED


# =============================================================================
# DOCX Content Models
# =============================================================================

class ParagraphContent(BaseModel):
    """Content for a paragraph in a document."""
    text: str = Field(..., description="Paragraph text (auto-truncated to 2000 chars)")
    style: str = Field(default="Normal", description="Word style to apply")

    @field_validator('text', mode='before')
    @classmethod
    def validate_text(cls, v):
        if not v:
            return ""
        v = str(v)
        if len(v) > 2000:
            return v[:1997] + "..."
        return v


class DocumentSection(BaseModel):
    """A section of a document (chapter, heading, etc.)."""

    section_number: int = Field(default=0)
    section_id: str = Field(default="")

    # Content
    heading: str = Field(..., description="Section heading (auto-truncated to 100 chars)")
    heading_level: int = Field(default=1, ge=1, le=6, description="Heading level (1-6)")
    paragraphs: List[ParagraphContent] = Field(default=[])
    bullet_points: List[str] = Field(default=[])

    @field_validator('heading', mode='before')
    @classmethod
    def validate_heading(cls, v):
        if not v:
            return ""
        v = str(v)
        if len(v) > 100:
            return v[:97] + "..."
        return v

    # Review status
    status: ContentStatus = Field(default=ContentStatus.DRAFT)
    user_feedback: Optional[str] = Field(default=None)

    def to_preview_dict(self) -> dict:
        """Convert to preview format for UI display.

        Returns data in format expected by frontend ContentReviewItem:
        - item_id (mapped from section_id)
        - item_number (mapped from section_number)
        - title (mapped from heading)
        - preview_text (generated from paragraphs)
        """
        # Generate preview text from first paragraph
        preview_text = ""
        if self.paragraphs:
            first_para = self.paragraphs[0].text
            preview_text = first_para[:100] + "..." if len(first_para) > 100 else first_para
        elif self.bullet_points:
            preview_text = " | ".join(self.bullet_points[:3])[:100]

        return {
            # Frontend ContentReviewItem expects these fields
            "item_id": self.section_id,
            "item_number": self.section_number,
            "title": self.heading,
            "status": self.status.value,
            "preview_text": preview_text,
            # Also include original section fields for backward compatibility
            "section_number": self.section_number,
            "section_id": self.section_id,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "paragraphs": [{"text": p.text[:200] + "..." if len(p.text) > 200 else p.text, "style": p.style} for p in self.paragraphs],
            "bullet_points": self.bullet_points[:10],
            "word_count": sum(len(p.text.split()) for p in self.paragraphs),
        }


class DocumentContent(BaseModel):
    """Complete document content."""

    title: str = Field(..., description="Document title (auto-truncated to 150 chars)")
    subtitle: Optional[str] = Field(default=None, description="Document subtitle (auto-truncated to 200 chars)")
    author: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)

    # Content
    sections: List[DocumentSection] = Field(..., min_length=1)

    # Review status
    overall_status: ContentStatus = Field(default=ContentStatus.DRAFT)

    @field_validator('title', mode='before')
    @classmethod
    def validate_title(cls, v):
        if not v:
            return ""
        v = str(v)
        if len(v) > 150:
            return v[:147] + "..."
        return v

    @field_validator('subtitle', mode='before')
    @classmethod
    def validate_subtitle(cls, v):
        if not v:
            return None
        v = str(v)
        if len(v) > 200:
            return v[:197] + "..."
        return v

    def get_sections_for_review(self) -> List[dict]:
        """Get all sections in preview format for review UI."""
        return [section.to_preview_dict() for section in self.sections]


# =============================================================================
# XLSX Content Models
# =============================================================================

class CellContent(BaseModel):
    """Content for a spreadsheet cell."""
    value: Any = Field(...)
    formula: Optional[str] = Field(default=None)
    style: Optional[str] = Field(default=None, description="Named style to apply")
    number_format: Optional[str] = Field(default=None)


class RowContent(BaseModel):
    """A row in a spreadsheet."""
    cells: List[CellContent] = Field(default=[])
    is_header: bool = Field(default=False)


class SheetContent(BaseModel):
    """Content for a worksheet."""

    sheet_number: int = Field(default=0)
    sheet_id: str = Field(default="")
    name: str = Field(..., description="Sheet name (auto-truncated to 31 chars)")

    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, v):
        if not v:
            return "Sheet1"
        v = str(v)
        if len(v) > 31:
            return v[:28] + "..."
        return v

    # Content
    headers: List[str] = Field(default=[])
    rows: List[RowContent] = Field(default=[])

    # Column widths
    column_widths: Dict[int, int] = Field(default={})

    # Review status
    status: ContentStatus = Field(default=ContentStatus.DRAFT)
    user_feedback: Optional[str] = Field(default=None)

    def to_preview_dict(self) -> dict:
        """Convert to preview format for UI display.

        Returns data in format expected by frontend ContentReviewItem:
        - item_id (mapped from sheet_id)
        - item_number (mapped from sheet_number)
        - title (mapped from name)
        - preview_text (generated from headers and row count)
        """
        # Generate preview text
        preview_text = f"{len(self.headers)} columns, {len(self.rows)} rows"
        if self.headers:
            header_preview = ", ".join(self.headers[:4])
            if len(self.headers) > 4:
                header_preview += "..."
            preview_text = f"Columns: {header_preview}"

        return {
            # Frontend ContentReviewItem expects these fields
            "item_id": self.sheet_id,
            "item_number": self.sheet_number,
            "title": self.name,
            "status": self.status.value,
            "preview_text": preview_text,
            # Also include original sheet fields for backward compatibility
            "sheet_number": self.sheet_number,
            "sheet_id": self.sheet_id,
            "sheet_name": self.name,  # For SheetContentReview compatibility
            "name": self.name,
            "headers": self.headers,
            "row_count": len(self.rows),
            "column_count": len(self.headers),
            "preview_rows": [
                [c.value for c in row.cells[:5]]
                for row in self.rows[:5]
            ],
        }


class SpreadsheetContent(BaseModel):
    """Complete spreadsheet content."""

    title: str = Field(..., description="Spreadsheet title (auto-truncated to 100 chars)")
    created_at: datetime = Field(default_factory=datetime.now)

    sheets: List[SheetContent] = Field(..., min_length=1)

    overall_status: ContentStatus = Field(default=ContentStatus.DRAFT)

    @field_validator('title', mode='before')
    @classmethod
    def validate_title(cls, v):
        if not v:
            return ""
        v = str(v)
        if len(v) > 100:
            return v[:97] + "..."
        return v


# =============================================================================
# Content Constraints (used by template analyzer)
# =============================================================================

class ContentConstraints(BaseModel):
    """Constraints derived from template analysis."""

    # PPTX constraints
    title_max_chars: int = Field(default=60)
    subtitle_max_chars: int = Field(default=100)
    bullet_max_chars: int = Field(default=70)
    bullets_per_slide: int = Field(default=7)
    body_max_chars: int = Field(default=500)
    speaker_notes_max_chars: int = Field(default=300)

    # DOCX constraints
    heading_max_chars: int = Field(default=80)
    paragraph_max_chars: int = Field(default=1000)

    # XLSX constraints
    cell_max_chars: int = Field(default=256)
    sheet_name_max_chars: int = Field(default=31)

    def to_llm_context(self) -> str:
        """Generate constraints section for LLM prompt."""
        return f"""CONTENT CONSTRAINTS (you MUST follow these):
- Slide/Section titles: max {self.title_max_chars} characters
- Subtitles: max {self.subtitle_max_chars} characters
- Bullet points: max {self.bullet_max_chars} characters each
- Bullets per slide: max {self.bullets_per_slide}
- Body text: max {self.body_max_chars} characters
- Speaker notes/descriptions: max {self.speaker_notes_max_chars} characters
- Document headings: max {self.heading_max_chars} characters
- Paragraphs: max {self.paragraph_max_chars} characters"""


# =============================================================================
# Content Edit Request Models
# =============================================================================

class ContentEditRequest(BaseModel):
    """Request to edit content at a specific location."""

    content_type: str = Field(..., description="pptx, docx, xlsx")
    item_id: str = Field(..., description="Slide/section/sheet ID")
    action: EditAction

    # For direct edits
    field_name: Optional[str] = Field(default=None, description="Field to edit (title, bullets, etc.)")
    new_value: Optional[Any] = Field(default=None, description="New value for the field")

    # For LLM regeneration
    feedback: Optional[str] = Field(default=None, description="User feedback for regeneration")
    regenerate_prompt: Optional[str] = Field(default=None, description="Custom prompt for regeneration")


class ContentReviewSession(BaseModel):
    """Tracks a content review session."""

    session_id: str
    content_type: str
    created_at: datetime = Field(default_factory=datetime.now)

    # Content being reviewed
    presentation: Optional[PresentationContent] = None
    document: Optional[DocumentContent] = None
    spreadsheet: Optional[SpreadsheetContent] = None

    # Review progress
    total_items: int = Field(default=0)
    reviewed_items: int = Field(default=0)
    approved_items: int = Field(default=0)

    # Edit history
    edits: List[ContentEditRequest] = Field(default=[])

    def get_progress(self) -> dict:
        """Get review progress summary."""
        return {
            "total": self.total_items,
            "reviewed": self.reviewed_items,
            "approved": self.approved_items,
            "remaining": self.total_items - self.reviewed_items,
            "progress_percent": (self.reviewed_items / self.total_items * 100) if self.total_items > 0 else 0,
        }

    def is_complete(self) -> bool:
        """Check if all items have been reviewed and approved."""
        return self.approved_items == self.total_items
