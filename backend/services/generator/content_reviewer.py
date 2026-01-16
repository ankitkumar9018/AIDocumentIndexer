"""
Content Review Service

Provides a proper content viewing and editing interface between
LLM content generation and final document rendering.

Users can:
- View generated content for each slide/page/sheet
- Edit content directly
- Request LLM enhancement/regeneration
- Approve content before final rendering
"""

import uuid
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import json
import logging

from .content_models import (
    ContentStatus,
    EditAction,
    SlideContent,
    BulletPoint,
    PresentationContent,
    DocumentSection,
    DocumentContent,
    SheetContent,
    SpreadsheetContent,
    ContentEditRequest,
    ContentReviewSession,
    ContentConstraints,
)

logger = logging.getLogger(__name__)


class ContentReviewService:
    """
    Service for reviewing and editing generated content before rendering.

    This is the "man in the middle" that allows users to view, edit, and
    approve content between LLM generation and final document creation.
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the review service.

        Args:
            llm_client: Optional LLM client for regeneration requests
        """
        self.llm_client = llm_client
        self.sessions: Dict[str, ContentReviewSession] = {}
        self.constraints: ContentConstraints = ContentConstraints()

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(
        self,
        content: PresentationContent | DocumentContent | SpreadsheetContent,
        content_type: str,
    ) -> ContentReviewSession:
        """
        Create a new review session for generated content.

        Args:
            content: The generated content to review
            content_type: Type of content (pptx, docx, xlsx)

        Returns:
            New ContentReviewSession
        """
        session_id = str(uuid.uuid4())

        session = ContentReviewSession(
            session_id=session_id,
            content_type=content_type,
            created_at=datetime.now(),
        )

        # Attach content and count items
        if content_type == "pptx" and isinstance(content, PresentationContent):
            session.presentation = content
            session.total_items = len(content.slides)
            # Assign IDs if missing
            for i, slide in enumerate(content.slides):
                if not slide.slide_id:
                    slide.slide_id = f"slide_{i+1}"
                slide.slide_number = i + 1
                slide.status = ContentStatus.PENDING_REVIEW

        elif content_type == "docx" and isinstance(content, DocumentContent):
            session.document = content
            session.total_items = len(content.sections)
            for i, section in enumerate(content.sections):
                if not section.section_id:
                    section.section_id = f"section_{i+1}"
                section.section_number = i + 1
                section.status = ContentStatus.PENDING_REVIEW

        elif content_type == "xlsx" and isinstance(content, SpreadsheetContent):
            session.spreadsheet = content
            session.total_items = len(content.sheets)
            for i, sheet in enumerate(content.sheets):
                if not sheet.sheet_id:
                    sheet.sheet_id = f"sheet_{i+1}"
                sheet.sheet_number = i + 1
                sheet.status = ContentStatus.PENDING_REVIEW

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ContentReviewSession]:
        """Get an existing review session."""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a review session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    # =========================================================================
    # Content Viewing
    # =========================================================================

    def get_all_items_preview(self, session_id: str) -> List[dict]:
        """
        Get preview of all content items for the review UI.

        Returns a list of simplified dictionaries for display.
        """
        session = self.get_session(session_id)
        if not session:
            return []

        if session.content_type == "pptx" and session.presentation:
            return session.presentation.get_slides_for_review()
        elif session.content_type == "docx" and session.document:
            return session.document.get_sections_for_review()
        elif session.content_type == "xlsx" and session.spreadsheet:
            return [sheet.to_preview_dict() for sheet in session.spreadsheet.sheets]

        return []

    def get_item_detail(self, session_id: str, item_id: str) -> Optional[dict]:
        """
        Get detailed content for a specific item (slide/section/sheet).

        This includes all fields and validation info for editing.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        if session.content_type == "pptx" and session.presentation:
            for slide in session.presentation.slides:
                if slide.slide_id == item_id:
                    return {
                        "type": "slide",
                        "data": slide.model_dump(),
                        "constraints": {
                            "title_max_chars": self.constraints.title_max_chars,
                            "bullet_max_chars": self.constraints.bullet_max_chars,
                            "bullets_per_slide": self.constraints.bullets_per_slide,
                            "speaker_notes_max_chars": self.constraints.speaker_notes_max_chars,
                        },
                        "validation": self._validate_slide(slide),
                    }

        elif session.content_type == "docx" and session.document:
            for section in session.document.sections:
                if section.section_id == item_id:
                    return {
                        "type": "section",
                        "data": section.model_dump(),
                        "constraints": {
                            "heading_max_chars": self.constraints.heading_max_chars,
                            "paragraph_max_chars": self.constraints.paragraph_max_chars,
                        },
                        "validation": self._validate_section(section),
                    }

        elif session.content_type == "xlsx" and session.spreadsheet:
            for sheet in session.spreadsheet.sheets:
                if sheet.sheet_id == item_id:
                    return {
                        "type": "sheet",
                        "data": sheet.model_dump(),
                        "constraints": {
                            "cell_max_chars": self.constraints.cell_max_chars,
                            "sheet_name_max_chars": self.constraints.sheet_name_max_chars,
                        },
                    }

        return None

    def get_review_status(self, session_id: str) -> dict:
        """Get overall review status and progress."""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        # Count statuses
        items = self._get_all_items(session)
        status_counts = {}
        for item in items:
            status = item.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "session_id": session_id,
            "content_type": session.content_type,
            "progress": session.get_progress(),
            "status_breakdown": status_counts,
            "is_complete": session.is_complete(),
            "can_render": all(
                item.status in [ContentStatus.APPROVED, ContentStatus.EDITED, ContentStatus.FINAL]
                for item in items
            ),
        }

    # =========================================================================
    # Content Editing
    # =========================================================================

    def edit_item(self, session_id: str, edit_request: ContentEditRequest) -> dict:
        """
        Apply an edit to a content item.

        Supports:
        - Direct field edits
        - Approval
        - Deletion
        - LLM regeneration requests

        Returns result with updated content.
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        # Find the item
        item = self._find_item(session, edit_request.item_id)
        if not item:
            return {"success": False, "error": f"Item {edit_request.item_id} not found"}

        # Track the edit
        session.edits.append(edit_request)

        # Handle different actions
        if edit_request.action == EditAction.APPROVE:
            return self._handle_approve(session, item)

        elif edit_request.action == EditAction.EDIT:
            return self._handle_direct_edit(session, item, edit_request)

        elif edit_request.action == EditAction.DELETE:
            return self._handle_delete(session, edit_request.item_id)

        elif edit_request.action in [EditAction.REGENERATE, EditAction.ENHANCE,
                                      EditAction.SHORTEN, EditAction.EXPAND,
                                      EditAction.CHANGE_TONE]:
            return self._handle_llm_action(session, item, edit_request)

        return {"success": False, "error": f"Unknown action: {edit_request.action}"}

    def batch_approve(self, session_id: str, item_ids: List[str]) -> dict:
        """Approve multiple items at once."""
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        approved = []
        for item_id in item_ids:
            item = self._find_item(session, item_id)
            if item:
                item.status = ContentStatus.APPROVED
                approved.append(item_id)
                session.approved_items += 1

        session.reviewed_items = session.approved_items

        return {
            "success": True,
            "approved": approved,
            "progress": session.get_progress(),
        }

    def approve_all(self, session_id: str) -> dict:
        """Approve all pending items."""
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        items = self._get_all_items(session)
        for item in items:
            item.status = ContentStatus.APPROVED

        session.approved_items = session.total_items
        session.reviewed_items = session.total_items

        return {
            "success": True,
            "approved_count": len(items),
            "progress": session.get_progress(),
        }

    # =========================================================================
    # Direct Edit Handlers
    # =========================================================================

    def _handle_approve(self, session: ContentReviewSession, item) -> dict:
        """Handle approval of an item."""
        item.status = ContentStatus.APPROVED
        session.approved_items += 1
        session.reviewed_items += 1

        return {
            "success": True,
            "item_id": item.slide_id if hasattr(item, 'slide_id') else item.section_id if hasattr(item, 'section_id') else item.sheet_id,
            "status": item.status.value,
            "progress": session.get_progress(),
        }

    def _handle_direct_edit(self, session: ContentReviewSession, item, edit_request: ContentEditRequest) -> dict:
        """Handle direct field edit."""
        if not edit_request.field_name or edit_request.new_value is None:
            return {"success": False, "error": "field_name and new_value required for edit"}

        try:
            # Handle nested fields (e.g., "bullets[0].text")
            if '[' in edit_request.field_name:
                self._set_nested_value(item, edit_request.field_name, edit_request.new_value)
            else:
                setattr(item, edit_request.field_name, edit_request.new_value)

            # Record edit history
            if hasattr(item, 'edit_history'):
                item.edit_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "field": edit_request.field_name,
                    "action": "direct_edit",
                })

            item.status = ContentStatus.EDITED
            session.reviewed_items += 1

            return {
                "success": True,
                "item_id": edit_request.item_id,
                "field": edit_request.field_name,
                "status": item.status.value,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_delete(self, session: ContentReviewSession, item_id: str) -> dict:
        """Handle deletion of an item."""
        try:
            if session.content_type == "pptx" and session.presentation:
                session.presentation.slides = [
                    s for s in session.presentation.slides if s.slide_id != item_id
                ]
            elif session.content_type == "docx" and session.document:
                session.document.sections = [
                    s for s in session.document.sections if s.section_id != item_id
                ]
            elif session.content_type == "xlsx" and session.spreadsheet:
                session.spreadsheet.sheets = [
                    s for s in session.spreadsheet.sheets if s.sheet_id != item_id
                ]

            session.total_items -= 1

            return {"success": True, "deleted": item_id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_llm_action(self, session: ContentReviewSession, item, edit_request: ContentEditRequest) -> dict:
        """Handle LLM-based actions (regenerate, enhance, etc.)."""
        if not self.llm_client:
            return {"success": False, "error": "LLM client not configured"}

        item.status = ContentStatus.REGENERATING
        item.user_feedback = edit_request.feedback

        # Build regeneration prompt based on action
        prompt = self._build_regeneration_prompt(item, edit_request)

        try:
            # Call LLM for regeneration
            regenerated = self._regenerate_content(item, prompt, edit_request.action)

            # Update item with regenerated content
            self._apply_regenerated_content(item, regenerated)

            item.status = ContentStatus.PENDING_REVIEW

            return {
                "success": True,
                "item_id": edit_request.item_id,
                "action": edit_request.action.value,
                "regenerated": True,
                "new_content": item.to_preview_dict() if hasattr(item, 'to_preview_dict') else None,
            }

        except Exception as e:
            item.status = ContentStatus.DRAFT
            return {"success": False, "error": str(e)}

    # =========================================================================
    # LLM Regeneration
    # =========================================================================

    def _build_regeneration_prompt(self, item, edit_request: ContentEditRequest) -> str:
        """Build prompt for LLM regeneration based on action type."""
        action = edit_request.action
        feedback = edit_request.feedback or ""

        # Get current content summary
        if hasattr(item, 'title'):
            current = f"Current title: {item.title}"
            if hasattr(item, 'bullets') and item.bullets:
                current += f"\nCurrent bullets: {[b.text for b in item.bullets]}"
        else:
            current = str(item)

        prompts = {
            EditAction.REGENERATE: f"""Regenerate this content completely.
{current}

User feedback: {feedback}

Generate new content that addresses the feedback while maintaining the same topic.""",

            EditAction.ENHANCE: f"""Enhance and improve this content.
{current}

User feedback: {feedback}

Make the content more engaging, professional, and impactful.""",

            EditAction.SHORTEN: f"""Make this content more concise.
{current}

User feedback: {feedback}

Reduce length while keeping the key points. Maximum 70 chars per bullet.""",

            EditAction.EXPAND: f"""Expand this content with more detail.
{current}

User feedback: {feedback}

Add more depth and detail while staying within constraints.""",

            EditAction.CHANGE_TONE: f"""Change the tone of this content.
{current}

User feedback: {feedback}

Adjust the tone as requested (more formal, casual, etc.).""",
        }

        return prompts.get(action, f"Improve this content: {current}")

    def _regenerate_content(self, item, prompt: str, action: EditAction) -> dict:
        """Call LLM to regenerate content."""
        # This would call the actual LLM client
        # For now, return a placeholder
        if self.llm_client:
            # Actual LLM call would go here
            pass

        # Placeholder response
        return {
            "title": item.title if hasattr(item, 'title') else "",
            "bullets": [],
        }

    def _apply_regenerated_content(self, item, regenerated: dict) -> None:
        """Apply regenerated content to item."""
        if hasattr(item, 'title') and 'title' in regenerated:
            item.title = regenerated['title']

        if hasattr(item, 'bullets') and 'bullets' in regenerated:
            item.bullets = [
                BulletPoint(text=b['text'], sub_bullets=b.get('sub_bullets', []))
                if isinstance(b, dict) else BulletPoint(text=b)
                for b in regenerated['bullets']
            ]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _find_item(self, session: ContentReviewSession, item_id: str):
        """Find an item by ID in the session content."""
        if session.content_type == "pptx" and session.presentation:
            for slide in session.presentation.slides:
                if slide.slide_id == item_id:
                    return slide

        elif session.content_type == "docx" and session.document:
            for section in session.document.sections:
                if section.section_id == item_id:
                    return section

        elif session.content_type == "xlsx" and session.spreadsheet:
            for sheet in session.spreadsheet.sheets:
                if sheet.sheet_id == item_id:
                    return sheet

        return None

    def _get_all_items(self, session: ContentReviewSession) -> list:
        """Get all content items from session."""
        if session.content_type == "pptx" and session.presentation:
            return session.presentation.slides
        elif session.content_type == "docx" and session.document:
            return session.document.sections
        elif session.content_type == "xlsx" and session.spreadsheet:
            return session.spreadsheet.sheets
        return []

    def _validate_slide(self, slide: SlideContent) -> dict:
        """Validate slide content against constraints."""
        warnings = []
        errors = []

        if len(slide.title) > self.constraints.title_max_chars:
            warnings.append(f"Title exceeds {self.constraints.title_max_chars} chars")

        if len(slide.bullets) > self.constraints.bullets_per_slide:
            warnings.append(f"Too many bullets ({len(slide.bullets)} > {self.constraints.bullets_per_slide})")

        for i, bullet in enumerate(slide.bullets):
            if len(bullet.text) > self.constraints.bullet_max_chars:
                warnings.append(f"Bullet {i+1} exceeds {self.constraints.bullet_max_chars} chars")

        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
        }

    def _validate_section(self, section: DocumentSection) -> dict:
        """Validate document section against constraints."""
        warnings = []
        errors = []

        if len(section.heading) > self.constraints.heading_max_chars:
            warnings.append(f"Heading exceeds {self.constraints.heading_max_chars} chars")

        for i, para in enumerate(section.paragraphs):
            if len(para.text) > self.constraints.paragraph_max_chars:
                warnings.append(f"Paragraph {i+1} exceeds {self.constraints.paragraph_max_chars} chars")

        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
        }

    def _set_nested_value(self, obj, path: str, value) -> None:
        """Set a value at a nested path like 'bullets[0].text'."""
        import re
        parts = re.split(r'\.|\[|\]', path)
        parts = [p for p in parts if p]  # Remove empty strings

        current = obj
        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        last_part = parts[-1]
        if last_part.isdigit():
            current[int(last_part)] = value
        else:
            setattr(current, last_part, value)

    # =========================================================================
    # Export for Rendering
    # =========================================================================

    def get_approved_content(self, session_id: str) -> Optional[PresentationContent | DocumentContent | SpreadsheetContent]:
        """
        Get the approved content ready for document rendering.

        Only returns content if all items are approved.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        # Check if all items are approved
        items = self._get_all_items(session)
        if not all(item.status in [ContentStatus.APPROVED, ContentStatus.EDITED, ContentStatus.FINAL] for item in items):
            logger.warning(f"Session {session_id} has unapproved items")
            return None

        # Return the appropriate content
        if session.content_type == "pptx":
            return session.presentation
        elif session.content_type == "docx":
            return session.document
        elif session.content_type == "xlsx":
            return session.spreadsheet

        return None

    def force_get_content(self, session_id: str) -> Optional[PresentationContent | DocumentContent | SpreadsheetContent]:
        """
        Get content regardless of approval status.
        Use with caution - bypasses review process.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        if session.content_type == "pptx":
            return session.presentation
        elif session.content_type == "docx":
            return session.document
        elif session.content_type == "xlsx":
            return session.spreadsheet

        return None
