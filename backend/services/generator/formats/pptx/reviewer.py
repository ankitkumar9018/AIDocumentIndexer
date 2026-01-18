"""
PPTX Slide Reviewer
====================

Provides LLM-based review of generated slides to catch
issues like text overflow, content quality, and layout problems.

Two review modes:
1. Content-based review: LLM reviews slide content (title, bullets, etc.) directly
2. Visual review (future): LLM reviews rendered slide images
"""

import os
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from pathlib import Path

import structlog

_logger = structlog.get_logger(__name__)


# Type alias for LLM generate function
LLMGenerateFunc = Callable[[str], Awaitable[str]]


@dataclass
class ReviewIssue:
    """An issue found during slide review."""
    issue_type: str  # text_overflow, image_placement, color_contrast, visual_balance, empty_placeholder
    description: str
    severity: str  # low, medium, high
    suggestion: str
    auto_fixable: bool = False
    fix_action: Optional[str] = None  # Action to take: truncate_title, condense_bullets, split_slide, etc.
    fix_params: Optional[Dict[str, Any]] = None  # Parameters for the fix


@dataclass
class SlideFixResult:
    """Result of fixing a slide."""
    slide_index: int
    fixes_applied: int
    original_content: Dict[str, Any]
    fixed_content: Dict[str, Any]
    changes_made: List[str]


@dataclass
class ReviewResult:
    """Result of reviewing a single slide."""
    slide_index: int
    has_issues: bool
    issues: List[ReviewIssue] = field(default_factory=list)
    overall_score: float = 1.0  # 0-1, 1 being perfect

    @classmethod
    def parse(cls, response: str, slide_index: int) -> "ReviewResult":
        """Parse LLM response into ReviewResult."""
        import json
        try:
            data = json.loads(response)
            issues = []
            for issue_data in data.get("issues", []):
                issues.append(ReviewIssue(
                    issue_type=issue_data.get("type", "unknown"),
                    description=issue_data.get("description", ""),
                    severity=issue_data.get("severity", "low"),
                    suggestion=issue_data.get("suggestion", ""),
                    auto_fixable=issue_data.get("auto_fixable", False),
                    fix_params=issue_data.get("fix_params"),
                ))
            return cls(
                slide_index=slide_index,
                has_issues=len(issues) > 0,
                issues=issues,
                overall_score=data.get("overall_score", 1.0 if not issues else 0.5),
            )
        except (json.JSONDecodeError, KeyError) as e:
            _logger.warning(f"Failed to parse review response: {e}")
            return cls(slide_index=slide_index, has_issues=False)


class SlideReviewer:
    """Review generated slides using LLM content analysis."""

    def __init__(self, llm_generate_func: Optional[LLMGenerateFunc] = None):
        """
        Initialize the slide reviewer.

        Args:
            llm_generate_func: Async function that takes a prompt string and returns LLM response.
                              If not provided, will use basic heuristic checks only.
        """
        self.llm_generate = llm_generate_func

    async def review_slide_content(
        self,
        slide_content: Dict[str, Any],
        slide_index: int,
        template_constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """
        Review slide content using LLM (no image needed).

        Args:
            slide_content: Dict with title, bullets, layout, has_image, etc.
            slide_index: Index of the slide being reviewed
            template_constraints: Optional constraints from template (max chars, etc.)

        Returns:
            ReviewResult with any issues found
        """
        # Always do basic heuristic checks first
        basic_result = self._basic_review(slide_content, slide_index, template_constraints)

        # If no LLM, return basic result
        if not self.llm_generate:
            return basic_result

        # Use LLM for deeper content review
        try:
            llm_result = await self._llm_content_review(slide_content, slide_index, template_constraints)

            # Merge issues from both reviews
            all_issues = basic_result.issues + llm_result.issues

            # Deduplicate by issue type + description
            seen = set()
            unique_issues = []
            for issue in all_issues:
                key = (issue.issue_type, issue.description[:50])
                if key not in seen:
                    seen.add(key)
                    unique_issues.append(issue)

            return ReviewResult(
                slide_index=slide_index,
                has_issues=len(unique_issues) > 0,
                issues=unique_issues,
                overall_score=min(basic_result.overall_score, llm_result.overall_score),
            )
        except Exception as e:
            _logger.warning(f"LLM content review failed: {e}")
            return basic_result

    async def _llm_content_review(
        self,
        slide_content: Dict[str, Any],
        slide_index: int,
        template_constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Use LLM to review the complete slide - content, layout, positioning, styling."""

        title = slide_content.get("title", "")
        bullets = slide_content.get("bullets", [])
        layout = slide_content.get("layout", "standard")
        has_image = slide_content.get("has_image", False)
        speaker_notes = slide_content.get("speaker_notes", "")

        # Slide dimensions and positioning info
        slide_width = slide_content.get("slide_width", 13.33)  # inches (16:9)
        slide_height = slide_content.get("slide_height", 7.5)

        # Get placeholder specs from template analysis (if available)
        title_ph = slide_content.get("title_placeholder", {})
        body_ph = slide_content.get("body_placeholder", {})

        title_position = {
            "x": title_ph.get("x", 0.6),
            "y": title_ph.get("y", 0.4),
            "w": title_ph.get("width", 12.0),
            "h": title_ph.get("height", 0.8),
        }
        content_position = {
            "x": body_ph.get("x", 0.6),
            "y": body_ph.get("y", 1.5),
            "w": body_ph.get("width", 6.0),
            "h": body_ph.get("height", 5.0),
        }
        image_position = slide_content.get("image_position", {"x": 7.0, "y": 2.0, "w": 5.5, "h": 4.5})

        # Styling info from placeholders or defaults
        title_font_size = title_ph.get("font_size", slide_content.get("title_font_size", 32))
        body_font_size = body_ph.get("font_size", slide_content.get("body_font_size", 18))
        title_max_chars = title_ph.get("max_chars", 50)
        body_max_chars = body_ph.get("max_chars", 500)
        body_max_bullets = body_ph.get("max_bullets", 7)

        accent_color = slide_content.get("accent_color", "#4F81BD")
        text_color = slide_content.get("text_color", "#000000")
        template_name = slide_content.get("template_name", "Default")

        # Theme information
        primary_color = slide_content.get("primary_color", "#333333")
        font_heading = slide_content.get("font_heading", "Calibri")
        font_body = slide_content.get("font_body", "Calibri")
        theme_style = slide_content.get("theme_style", "")
        layout_recommended_use = slide_content.get("layout_recommended_use", "general content")

        # Format bullets for prompt
        bullets_text = ""
        total_bullet_chars = 0
        for i, bullet in enumerate(bullets):
            text = bullet.get("text", bullet) if isinstance(bullet, dict) else str(bullet)
            sub_bullets = bullet.get("sub_bullets", []) if isinstance(bullet, dict) else []
            bullets_text += f"  {i+1}. [{len(text)} chars] {text}\n"
            total_bullet_chars += len(text)
            for j, sub in enumerate(sub_bullets):
                sub_text = sub.get("text", sub) if isinstance(sub, dict) else str(sub)
                bullets_text += f"     {j+1}. [{len(sub_text)} chars] {sub_text}\n"
                total_bullet_chars += len(sub_text)

        # Build constraints text from template analysis or defaults
        constraints_text = ""
        if template_constraints:
            constraints_text = f"""
PLACEHOLDER CONSTRAINTS (from template analysis):
- Title placeholder: max {template_constraints.get('title_max_chars', title_max_chars)} chars
- Body placeholder: max {template_constraints.get('body_max_chars', body_max_chars)} chars total
- Bullets per slide: max {template_constraints.get('bullets_per_slide', body_max_bullets)}
- Characters per bullet: max {template_constraints.get('bullet_max_chars', 80)} chars
"""
        else:
            constraints_text = f"""
PLACEHOLDER CONSTRAINTS (estimated from dimensions):
- Title placeholder: {title_position['w']}" wide → max ~{title_max_chars} chars at {title_font_size}pt
- Body placeholder: {content_position['w']}" x {content_position['h']}" → max ~{body_max_bullets} bullets
- Estimated chars/bullet: ~{int(content_position['w'] * 8)} chars at {body_font_size}pt
"""

        # Build theme description for prompt
        theme_info = f"""
TEMPLATE THEME:
- Template: {template_name}
- Primary Color: {primary_color}
- Accent Color: {accent_color}
- Text Color: {text_color}
- Heading Font: {font_heading}
- Body Font: {font_body}
- Layout Purpose: {layout_recommended_use}"""
        if theme_style:
            theme_info += f"\n- Style: {theme_style}"

        # Calculate if content will overflow
        title_overflow = len(title) > title_max_chars
        bullets_overflow = len(bullets) > body_max_bullets
        content_chars_overflow = total_bullet_chars > body_max_chars

        overflow_warnings = []
        if title_overflow:
            overflow_warnings.append(f"⚠️ TITLE OVERFLOW: {len(title)} chars > {title_max_chars} max")
        if bullets_overflow:
            overflow_warnings.append(f"⚠️ TOO MANY BULLETS: {len(bullets)} > {body_max_bullets} max")
        if content_chars_overflow:
            overflow_warnings.append(f"⚠️ CONTENT OVERFLOW: {total_bullet_chars} chars > {body_max_chars} max")

        overflow_section = ""
        if overflow_warnings:
            overflow_section = "\n⚠️ DETECTED ISSUES:\n" + "\n".join(overflow_warnings) + "\n"

        prompt = f"""You are reviewing a PowerPoint slide as if you were looking at the actual rendered slide.
The template has specific placeholder sizes - content MUST fit within these constraints.
{overflow_section}
═══════════════════════════════════════════════════════════════════════
SLIDE {slide_index + 1} - COMPLETE SPECIFICATION
═══════════════════════════════════════════════════════════════════════

SLIDE CANVAS:
- Dimensions: {slide_width}" x {slide_height}" (16:9 widescreen)
- Layout Type: {layout}
{theme_info}

TITLE PLACEHOLDER:
- Position: x={title_position['x']}", y={title_position['y']}"
- Size: {title_position['w']}" x {title_position['h']}"
- Max capacity: ~{title_max_chars} characters at {title_font_size}pt
- Current text: "{title}" [{len(title)} chars] {"⚠️ OVERFLOW!" if title_overflow else "✓ OK"}
- Font: {font_heading}, {title_font_size}pt, Bold

BODY PLACEHOLDER:
- Position: x={content_position['x']}", y={content_position['y']}"
- Size: {content_position['w']}" x {content_position['h']}"
- Max capacity: ~{body_max_bullets} bullets, ~{body_max_chars} total chars
- Current: {len(bullets)} bullets, {total_bullet_chars} chars {"⚠️ OVERFLOW!" if bullets_overflow or content_chars_overflow else "✓ OK"}
- Font: {font_body}, {body_font_size}pt

BULLET CONTENT:
{bullets_text if bullets_text else "  (no bullet points)"}

IMAGE AREA:
- Has Image: {has_image}
- Position: x={image_position['x']}", y={image_position['y']}"
- Size: {image_position['w']}" x {image_position['h']}"
{constraints_text}
═══════════════════════════════════════════════════════════════════════

REVIEW THIS SLIDE FOR:
1. TEXT OVERFLOW - Does content fit the placeholder sizes? (CRITICAL - check chars vs max)
2. VISUAL BALANCE - Is content well-distributed? Too crowded or too sparse?
3. LAYOUT APPROPRIATENESS - Does the layout match the content type?
4. IMAGE PLACEMENT - If has image, is positioning/size appropriate?
5. READABILITY - Font sizes adequate? Too much text per bullet?
6. PROFESSIONAL APPEARANCE - Would this look good in a business presentation?
7. CONTENT QUALITY - Is the text clear, concise, and well-structured?
8. THEME CONSISTENCY - Does the content tone match the template style?

Return ONLY valid JSON (no markdown code blocks):
{{
    "issues": [
        {{
            "type": "text_overflow|visual_balance|layout_mismatch|image_placement|readability|professionalism|content_quality|theme_mismatch",
            "description": "Specific description of the issue",
            "severity": "low|medium|high",
            "suggestion": "Specific actionable fix"
        }}
    ],
    "overall_score": 0.0 to 1.0 (1.0 = presentation-ready),
    "summary": "One sentence summary of slide quality"
}}

If the slide looks good: {{"issues": [], "overall_score": 1.0, "summary": "Slide fits template constraints and is presentation-ready"}}
"""

        response = await self.llm_generate(prompt)
        return ReviewResult.parse(response, slide_index)

    def _basic_review(
        self,
        slide_content: Dict[str, Any],
        slide_index: int,
        template_constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """
        Perform basic heuristic review without LLM.

        Checks:
        - Title length
        - Bullet count
        - Bullet text length
        - Layout consistency
        """
        issues = []

        # Get constraints (use defaults if not provided)
        max_title = 60
        max_bullet = 80
        max_bullets = 7
        if template_constraints:
            max_title = template_constraints.get("title_max_chars", 60)
            max_bullet = template_constraints.get("bullet_max_chars", 80)
            max_bullets = template_constraints.get("bullets_per_slide", 7)

        title = slide_content.get("title", "")
        if len(title) > max_title:
            issues.append(ReviewIssue(
                issue_type="text_overflow",
                description=f"Title is {len(title)} chars (max {max_title})",
                severity="medium",
                suggestion=f"Shorten title to under {max_title} characters",
                auto_fixable=False,
            ))

        bullets = slide_content.get("bullets", [])
        if len(bullets) > max_bullets:
            issues.append(ReviewIssue(
                issue_type="visual_balance",
                description=f"Slide has {len(bullets)} bullets (max {max_bullets})",
                severity="medium",
                suggestion=f"Reduce to {max_bullets} bullets or split into multiple slides",
                auto_fixable=False,
            ))

        for i, bullet in enumerate(bullets):
            text = bullet.get("text", bullet) if isinstance(bullet, dict) else str(bullet)
            if len(text) > max_bullet:
                issues.append(ReviewIssue(
                    issue_type="text_overflow",
                    description=f"Bullet {i + 1} is {len(text)} chars (max {max_bullet})",
                    severity="low",
                    suggestion="Shorten bullet or break into sub-bullets",
                    auto_fixable=False,
                ))

            # Check sub-bullets
            sub_bullets = bullet.get("sub_bullets", []) if isinstance(bullet, dict) else []
            for j, sub in enumerate(sub_bullets):
                sub_text = sub.get("text", sub) if isinstance(sub, dict) else str(sub)
                if len(sub_text) > max_bullet:
                    issues.append(ReviewIssue(
                        issue_type="text_overflow",
                        description=f"Sub-bullet {i+1}.{j+1} is {len(sub_text)} chars",
                        severity="low",
                        suggestion="Shorten sub-bullet text",
                        auto_fixable=False,
                    ))

        # Check layout consistency
        layout = slide_content.get("layout", "standard")
        has_image = slide_content.get("has_image", False)
        if layout in ["image_focused", "two_column"] and not has_image:
            issues.append(ReviewIssue(
                issue_type="layout_mismatch",
                description=f"Layout '{layout}' expects image but none provided",
                severity="low",
                suggestion="Add image or change layout to 'standard'",
                auto_fixable=False,
            ))

        return ReviewResult(
            slide_index=slide_index,
            has_issues=len(issues) > 0,
            issues=issues,
            overall_score=max(0.0, 1.0 - (len(issues) * 0.15)),
        )

    async def review_all_slides(
        self,
        sections: List[Any],
        template_constraints: Optional[Dict[str, Any]] = None,
    ) -> List[ReviewResult]:
        """
        Review all slide content before rendering.

        Args:
            sections: List of Section objects with title, content, etc.
            template_constraints: Optional constraints from template

        Returns:
            List of ReviewResults, one per section/slide
        """
        results = []

        for idx, section in enumerate(sections):
            # Extract content from Section object
            slide_content = {
                "title": getattr(section, "title", "") or "",
                "bullets": self._extract_bullets(getattr(section, "content", "") or ""),
                "layout": getattr(section, "layout", "standard") or "standard",
                "has_image": bool(getattr(section, "image_path", None)),
                "speaker_notes": getattr(section, "speaker_notes", "") or "",
            }

            result = await self.review_slide_content(
                slide_content, idx, template_constraints
            )
            results.append(result)

            if result.has_issues:
                for issue in result.issues:
                    _logger.warning(
                        "Slide review issue",
                        slide=idx + 1,
                        type=issue.issue_type,
                        severity=issue.severity,
                        description=issue.description,
                    )

        return results

    def _extract_bullets(self, content: str) -> List[Dict[str, Any]]:
        """Extract bullet points from content string."""
        if not content:
            return []

        bullets = []
        lines = content.split("\n")
        current_bullet = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for main bullet markers
            if line.startswith(("- ", "* ", "• ", "▸ ")):
                if current_bullet:
                    bullets.append(current_bullet)
                current_bullet = {"text": line[2:], "sub_bullets": []}
            elif line.startswith(("  - ", "  * ", "  • ", "    - ")):
                # Sub-bullet
                if current_bullet:
                    sub_text = line.lstrip(" -•*▸▹")
                    current_bullet["sub_bullets"].append({"text": sub_text})
            elif line[0].isdigit() and "." in line[:3]:
                # Numbered list
                if current_bullet:
                    bullets.append(current_bullet)
                parts = line.split(".", 1)
                if len(parts) > 1:
                    current_bullet = {"text": parts[1].strip(), "sub_bullets": []}

        if current_bullet:
            bullets.append(current_bullet)

        return bullets

    async def fix_slide_content(
        self,
        slide_content: Dict[str, Any],
        review_result: ReviewResult,
        template_constraints: Optional[Dict[str, Any]] = None,
    ) -> SlideFixResult:
        """
        Fix identified issues in slide content.

        Args:
            slide_content: Original slide content dict
            review_result: Review result with identified issues
            template_constraints: Template constraints for max lengths

        Returns:
            SlideFixResult with fixed content
        """
        if not review_result.has_issues:
            return SlideFixResult(
                slide_index=review_result.slide_index,
                fixes_applied=0,
                original_content=slide_content.copy(),
                fixed_content=slide_content.copy(),
                changes_made=[],
            )

        # Get constraints
        max_title = 60
        max_bullet = 80
        max_bullets = 7
        if template_constraints:
            max_title = template_constraints.get("title_max_chars", 60)
            max_bullet = template_constraints.get("bullet_max_chars", 80)
            max_bullets = template_constraints.get("bullets_per_slide", 7)

        fixed_content = slide_content.copy()
        changes_made = []
        fixes_applied = 0

        # Process each issue
        for issue in review_result.issues:
            try:
                if issue.issue_type == "text_overflow":
                    if "Title" in issue.description or "title" in issue.description:
                        # Fix title overflow
                        original_title = fixed_content.get("title", "")
                        if len(original_title) > max_title:
                            fixed_title = await self._condense_text(
                                original_title, max_title, "title"
                            )
                            fixed_content["title"] = fixed_title
                            changes_made.append(f"Condensed title from {len(original_title)} to {len(fixed_title)} chars")
                            fixes_applied += 1

                    elif "Bullet" in issue.description or "bullet" in issue.description:
                        # Fix bullet overflow
                        bullets = fixed_content.get("bullets", [])
                        fixed_bullets = await self._fix_bullet_overflow(
                            bullets, max_bullet
                        )
                        if fixed_bullets != bullets:
                            fixed_content["bullets"] = fixed_bullets
                            changes_made.append("Condensed overflowing bullets")
                            fixes_applied += 1

                elif issue.issue_type == "visual_balance":
                    if "bullets" in issue.description.lower():
                        # Too many bullets - consolidate
                        bullets = fixed_content.get("bullets", [])
                        if len(bullets) > max_bullets:
                            fixed_bullets = await self._consolidate_bullets(
                                bullets, max_bullets
                            )
                            fixed_content["bullets"] = fixed_bullets
                            changes_made.append(f"Consolidated {len(bullets)} bullets to {len(fixed_bullets)}")
                            fixes_applied += 1

                elif issue.issue_type == "layout_mismatch":
                    # Change layout to standard if no image
                    if not fixed_content.get("has_image", False):
                        fixed_content["layout"] = "standard"
                        changes_made.append(f"Changed layout to 'standard' (no image)")
                        fixes_applied += 1

            except Exception as e:
                _logger.warning(f"Failed to fix issue {issue.issue_type}: {e}")

        return SlideFixResult(
            slide_index=review_result.slide_index,
            fixes_applied=fixes_applied,
            original_content=slide_content,
            fixed_content=fixed_content,
            changes_made=changes_made,
        )

    async def _condense_text(
        self, text: str, max_chars: int, text_type: str = "text"
    ) -> str:
        """
        Condense text to fit within max_chars using LLM or heuristics.

        Args:
            text: Original text to condense
            max_chars: Maximum allowed characters
            text_type: Type of text (title, bullet, etc.) for context

        Returns:
            Condensed text
        """
        if len(text) <= max_chars:
            return text

        # Try LLM-based condensing first
        if self.llm_generate:
            try:
                prompt = f"""Condense this {text_type} to fit within {max_chars} characters while preserving the key meaning.

Original ({len(text)} chars): "{text}"

Requirements:
- Must be {max_chars} characters or less
- Preserve the essential meaning
- Use abbreviations if helpful
- Remove unnecessary words
- Keep it professional and clear

Return ONLY the condensed text, nothing else."""

                response = await self.llm_generate(prompt)
                condensed = response.strip().strip('"')

                # Verify it fits
                if len(condensed) <= max_chars:
                    return condensed
                else:
                    # LLM failed to meet constraint, fall back to heuristic
                    _logger.warning(f"LLM condensed text still too long: {len(condensed)} > {max_chars}")
            except Exception as e:
                _logger.warning(f"LLM text condensing failed: {e}")

        # Heuristic fallback: smart truncation
        return self._smart_truncate(text, max_chars)

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Truncate text intelligently at word boundaries.

        Args:
            text: Text to truncate
            max_chars: Maximum characters

        Returns:
            Truncated text with ellipsis
        """
        if len(text) <= max_chars:
            return text

        # Reserve space for ellipsis
        target_len = max_chars - 3

        # Try to break at word boundary
        truncated = text[:target_len]
        last_space = truncated.rfind(" ")

        if last_space > target_len * 0.7:  # Keep at least 70% of content
            truncated = truncated[:last_space]

        return truncated.rstrip() + "..."

    async def _fix_bullet_overflow(
        self, bullets: List[Dict[str, Any]], max_chars: int
    ) -> List[Dict[str, Any]]:
        """
        Fix bullets that exceed max character limit.

        Args:
            bullets: List of bullet dicts with 'text' and 'sub_bullets'
            max_chars: Maximum characters per bullet

        Returns:
            Fixed bullets list
        """
        fixed_bullets = []

        for bullet in bullets:
            text = bullet.get("text", bullet) if isinstance(bullet, dict) else str(bullet)
            sub_bullets = bullet.get("sub_bullets", []) if isinstance(bullet, dict) else []

            # Fix main bullet
            if len(text) > max_chars:
                text = await self._condense_text(text, max_chars, "bullet point")

            # Fix sub-bullets
            fixed_subs = []
            for sub in sub_bullets:
                sub_text = sub.get("text", sub) if isinstance(sub, dict) else str(sub)
                if len(sub_text) > max_chars:
                    sub_text = await self._condense_text(sub_text, max_chars, "sub-bullet")
                fixed_subs.append({"text": sub_text})

            fixed_bullets.append({
                "text": text,
                "sub_bullets": fixed_subs,
            })

        return fixed_bullets

    async def _consolidate_bullets(
        self, bullets: List[Dict[str, Any]], max_bullets: int
    ) -> List[Dict[str, Any]]:
        """
        Consolidate bullets to fit within max_bullets limit.

        Args:
            bullets: Original bullet list
            max_bullets: Maximum allowed bullets

        Returns:
            Consolidated bullet list
        """
        if len(bullets) <= max_bullets:
            return bullets

        # Try LLM-based consolidation first
        if self.llm_generate:
            try:
                bullet_texts = []
                for b in bullets:
                    text = b.get("text", b) if isinstance(b, dict) else str(b)
                    bullet_texts.append(f"- {text}")

                bullets_str = "\n".join(bullet_texts)

                prompt = f"""Consolidate these {len(bullets)} bullet points into {max_bullets} bullets while preserving all key information.

Original bullets:
{bullets_str}

Requirements:
- Exactly {max_bullets} bullets
- Preserve all essential information
- Combine related points
- Keep each bullet concise (under 80 chars)
- Format: Return ONLY the bullet points, one per line, starting with "- "

Return ONLY the consolidated bullets, nothing else."""

                response = await self.llm_generate(prompt)

                # Parse response into bullets
                consolidated = []
                for line in response.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        consolidated.append({"text": line[2:], "sub_bullets": []})
                    elif line.startswith("• "):
                        consolidated.append({"text": line[2:], "sub_bullets": []})
                    elif line and not line.startswith("#"):  # Skip markdown headers
                        consolidated.append({"text": line, "sub_bullets": []})

                if len(consolidated) <= max_bullets and len(consolidated) > 0:
                    return consolidated[:max_bullets]
                else:
                    _logger.warning(f"LLM consolidation returned {len(consolidated)} bullets, expected {max_bullets}")

            except Exception as e:
                _logger.warning(f"LLM bullet consolidation failed: {e}")

        # Heuristic fallback: keep first N bullets, merge last ones into one
        if len(bullets) > max_bullets:
            kept = bullets[:max_bullets - 1]
            remaining = bullets[max_bullets - 1:]

            # Combine remaining bullets
            combined_text = "; ".join(
                b.get("text", b) if isinstance(b, dict) else str(b)
                for b in remaining
            )

            # Truncate if too long
            if len(combined_text) > 80:
                combined_text = self._smart_truncate(combined_text, 80)

            kept.append({"text": combined_text, "sub_bullets": []})
            return kept

        return bullets

    async def review_and_fix_slide(
        self,
        slide_content: Dict[str, Any],
        slide_index: int,
        template_constraints: Optional[Dict[str, Any]] = None,
    ) -> tuple[ReviewResult, SlideFixResult]:
        """
        Review a slide and automatically fix any issues found.

        Args:
            slide_content: Slide content to review and fix
            slide_index: Index of the slide
            template_constraints: Template constraints

        Returns:
            Tuple of (ReviewResult, SlideFixResult)
        """
        # First, review the slide
        review_result = await self.review_slide_content(
            slide_content, slide_index, template_constraints
        )

        # If issues found, fix them
        if review_result.has_issues:
            fix_result = await self.fix_slide_content(
                slide_content, review_result, template_constraints
            )

            # Log fixes
            if fix_result.fixes_applied > 0:
                _logger.info(
                    "Applied slide fixes",
                    slide=slide_index + 1,
                    fixes=fix_result.fixes_applied,
                    changes=fix_result.changes_made,
                )

            return review_result, fix_result
        else:
            # No issues, return unchanged
            return review_result, SlideFixResult(
                slide_index=slide_index,
                fixes_applied=0,
                original_content=slide_content,
                fixed_content=slide_content,
                changes_made=[],
            )


class VisionSlideReviewer:
    """Review rendered slide images using vision-capable LLM (PPTEval style).

    This reviewer renders slides to images and uses a vision model (like GPT-4V,
    Claude 3, or Ollama vision models) to evaluate visual quality aspects that
    cannot be detected from content alone.

    Based on research from PPTEval framework which evaluates slides across:
    - Content: Text quality, relevance, completeness
    - Design: Layout, colors, fonts, visual appeal
    - Coherence: Consistency across slides, flow

    Reference: https://github.com/anthonywu/PPTEval
    """

    def __init__(
        self,
        vision_llm_func: Optional[Callable[[str, str], Awaitable[str]]] = None,
        render_backend: str = "libreoffice",  # "libreoffice" or "unoconv"
    ):
        """
        Initialize the vision slide reviewer.

        Args:
            vision_llm_func: Async function that takes (prompt, image_path) and returns response.
                            If None, uses Ollama with llava or similar vision model.
            render_backend: Backend for rendering slides to images.
        """
        self.vision_llm = vision_llm_func
        self.render_backend = render_backend
        self._temp_dir = None

    async def review_slide_image(
        self,
        image_path: str,
        slide_content: Dict[str, Any],
        slide_index: int,
    ) -> ReviewResult:
        """
        Review a rendered slide image using vision LLM.

        Args:
            image_path: Path to the PNG image of the slide
            slide_content: Expected content dict for validation
            slide_index: Index of the slide

        Returns:
            ReviewResult with visual issues found
        """
        if not self.vision_llm:
            _logger.warning("No vision LLM available, skipping visual review")
            return ReviewResult(slide_index=slide_index, has_issues=False)

        title = slide_content.get("title", "")
        bullets = slide_content.get("bullets", [])
        has_image = slide_content.get("has_image", False)

        prompt = f"""Analyze this PowerPoint slide image for visual quality issues.

EXPECTED CONTENT:
- Title: "{title}"
- Number of bullet points: {len(bullets)}
- Has image: {has_image}

EVALUATE THE FOLLOWING (based on PPTEval framework):

1. DESIGN QUALITY (40% weight):
   - Text visibility and contrast
   - Font sizes appropriate for presentation
   - Color scheme and visual appeal
   - Layout balance and spacing
   - Image placement and sizing (if applicable)

2. CONTENT RENDERING (30% weight):
   - All expected text is visible (no truncation)
   - Bullet points are readable
   - No overlapping elements
   - No empty placeholders visible

3. COHERENCE (30% weight):
   - Title is clearly distinguishable
   - Visual hierarchy is clear
   - Elements are aligned properly
   - Professional appearance overall

4. CRITICAL POSITIONING CHECKS (PHASE 10 - Report as HIGH severity issues):
   - FOOTER POSITION: The date (left) and page number (right) MUST appear at the VERY BOTTOM
     of the slide, within the bottom 10% of the slide height. If footer elements appear in the
     middle or upper portion of the slide, this is a CRITICAL positioning error.
   - TITLE-BODY GAP: There must be visible vertical spacing between where the title text ends
     and where the body/bullet content begins. If text appears crowded against the title with
     minimal gap (less than approximately 1cm visual gap), flag as HIGH severity spacing issue.
   - CONTENT BOUNDS: Body text and bullets must NOT extend into the bottom 15% of the slide
     (the footer zone). If content appears to overlap or crowd the footer area, flag as HIGH severity.

Return your analysis as JSON (no markdown blocks):
{{
    "issues": [
        {{
            "type": "design|content_rendering|coherence",
            "description": "Specific description of the visual issue",
            "severity": "low|medium|high",
            "suggestion": "How to fix this issue",
            "auto_fixable": false
        }}
    ],
    "scores": {{
        "design": 0.0 to 1.0,
        "content_rendering": 0.0 to 1.0,
        "coherence": 0.0 to 1.0
    }},
    "overall_score": 0.0 to 1.0,
    "summary": "One-sentence summary of slide visual quality"
}}

If the slide looks visually correct: {{"issues": [], "scores": {{"design": 1.0, "content_rendering": 1.0, "coherence": 1.0}}, "overall_score": 1.0, "summary": "Slide renders correctly with good visual quality"}}
"""

        try:
            response = await self.vision_llm(prompt, image_path)
            return self._parse_vision_response(response, slide_index)
        except Exception as e:
            _logger.error(f"Vision review failed for slide {slide_index}: {e}")
            return ReviewResult(slide_index=slide_index, has_issues=False)

    def _parse_vision_response(self, response: str, slide_index: int) -> ReviewResult:
        """Parse vision LLM response into ReviewResult."""
        import json
        try:
            # Try to extract JSON from response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)
            issues = []
            for issue_data in data.get("issues", []):
                issues.append(ReviewIssue(
                    issue_type=issue_data.get("type", "design"),
                    description=issue_data.get("description", ""),
                    severity=issue_data.get("severity", "low"),
                    suggestion=issue_data.get("suggestion", ""),
                    auto_fixable=issue_data.get("auto_fixable", False),
                ))

            return ReviewResult(
                slide_index=slide_index,
                has_issues=len(issues) > 0,
                issues=issues,
                overall_score=data.get("overall_score", 1.0 if not issues else 0.7),
            )
        except (json.JSONDecodeError, KeyError) as e:
            _logger.warning(f"Failed to parse vision review response: {e}")
            return ReviewResult(slide_index=slide_index, has_issues=False)

    def render_slide_to_image(
        self,
        pptx_path: str,
        slide_index: int,
        output_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Render a single slide from a PPTX file to a PNG image.

        Args:
            pptx_path: Path to the PPTX file
            slide_index: Index of the slide to render (0-based)
            output_dir: Directory for output image (uses temp if None)

        Returns:
            Path to the rendered PNG image, or None if rendering failed
        """
        if output_dir is None:
            if self._temp_dir is None:
                self._temp_dir = tempfile.mkdtemp(prefix="slide_review_")
            output_dir = self._temp_dir

        output_path = os.path.join(output_dir, f"slide_{slide_index}.png")

        try:
            if self.render_backend == "libreoffice":
                return self._render_with_libreoffice(pptx_path, slide_index, output_path)
            else:
                _logger.warning(f"Unknown render backend: {self.render_backend}")
                return None
        except Exception as e:
            _logger.error(f"Failed to render slide {slide_index}: {e}")
            return None

    def _render_with_libreoffice(
        self,
        pptx_path: str,
        slide_index: int,
        output_path: str,
    ) -> Optional[str]:
        """Render slide using LibreOffice headless mode via PDF intermediate.

        LibreOffice --convert-to png only exports the first slide.
        We use a two-step approach: PPTX → PDF → PNG per page using pdftoppm.
        """
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(pptx_path))[0]

        # Use a cached PDF if we've already converted this PPTX
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

        try:
            # Step 1: Convert PPTX to PDF (if not already done)
            if not os.path.exists(pdf_path):
                _logger.info(f"Converting PPTX to PDF: {pptx_path}")
                result = subprocess.run(
                    [
                        "soffice",
                        "--headless",
                        "--convert-to", "pdf",
                        "--outdir", output_dir,
                        pptx_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,  # PDF conversion can take longer
                )

                if result.returncode != 0:
                    _logger.warning(f"LibreOffice PDF conversion failed: {result.stderr}")
                    return None

                if not os.path.exists(pdf_path):
                    _logger.warning(f"PDF file not created: {pdf_path}")
                    return None

            # Step 2: Extract specific page as PNG using pdftoppm
            # pdftoppm creates files like: prefix-01.png, prefix-02.png, etc.
            slide_prefix = os.path.join(output_dir, f"slide_{base_name}")

            # Check if we already extracted this slide
            # pdftoppm uses 1-based page numbers with zero-padded format
            expected_png = f"{slide_prefix}-{slide_index + 1:02d}.png"

            if not os.path.exists(expected_png):
                # Use pdftoppm to convert just the specific page
                # -f = first page, -l = last page (both same for single page)
                page_num = slide_index + 1  # pdftoppm uses 1-based
                result = subprocess.run(
                    [
                        "pdftoppm",
                        "-png",
                        "-f", str(page_num),
                        "-l", str(page_num),
                        "-r", "150",  # 150 DPI for good quality
                        pdf_path,
                        slide_prefix,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    _logger.warning(f"pdftoppm conversion failed: {result.stderr}")
                    # Fall back to ImageMagick if available
                    return self._render_with_imagemagick(pdf_path, slide_index, output_path)

            # Find the output file (pdftoppm may use different padding)
            possible_files = [
                f"{slide_prefix}-{slide_index + 1:02d}.png",  # 01, 02, etc.
                f"{slide_prefix}-{slide_index + 1}.png",      # 1, 2, etc.
                f"{slide_prefix}-{slide_index + 1:03d}.png",  # 001, 002, etc.
            ]

            for candidate in possible_files:
                if os.path.exists(candidate):
                    if candidate != output_path:
                        os.rename(candidate, output_path)
                    return output_path

            _logger.warning(f"Slide PNG not found. Tried: {possible_files}")
            return None

        except subprocess.TimeoutExpired:
            _logger.error("Conversion timed out")
            return None
        except FileNotFoundError as e:
            _logger.warning(f"Required tool not found: {e}")
            return None

    def _render_with_imagemagick(
        self,
        pdf_path: str,
        slide_index: int,
        output_path: str,
    ) -> Optional[str]:
        """Fallback: Use ImageMagick to extract a page from PDF."""
        try:
            # ImageMagick uses 0-based page indexing
            result = subprocess.run(
                [
                    "magick",
                    "-density", "150",
                    f"{pdf_path}[{slide_index}]",  # Extract specific page
                    output_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and os.path.exists(output_path):
                return output_path

            _logger.warning(f"ImageMagick conversion failed: {result.stderr}")
            return None

        except FileNotFoundError:
            _logger.warning("ImageMagick not found")
            return None
        except Exception as e:
            _logger.warning(f"ImageMagick failed: {e}")
            return None

    async def review_presentation(
        self,
        pptx_path: str,
        sections: List[Any],
    ) -> List[ReviewResult]:
        """
        Review all slides in a presentation visually.

        Args:
            pptx_path: Path to the generated PPTX file
            sections: List of Section objects with expected content

        Returns:
            List of ReviewResults for each slide
        """
        results = []

        for idx, section in enumerate(sections):
            # Render slide to image
            image_path = self.render_slide_to_image(pptx_path, idx)

            if not image_path:
                _logger.warning(f"Could not render slide {idx} for visual review")
                results.append(ReviewResult(slide_index=idx, has_issues=False))
                continue

            # Extract expected content from section
            slide_content = {
                "title": getattr(section, "title", "") or "",
                "bullets": self._extract_bullets(getattr(section, "content", "") or ""),
                "has_image": bool(getattr(section, "image_path", None)),
            }

            # Review the rendered image
            result = await self.review_slide_image(image_path, slide_content, idx)
            results.append(result)

            if result.has_issues:
                for issue in result.issues:
                    _logger.warning(
                        "Visual review issue",
                        slide=idx + 1,
                        type=issue.issue_type,
                        severity=issue.severity,
                        description=issue.description[:100],
                    )

        return results

    def _extract_bullets(self, content: str) -> List[str]:
        """Extract bullet text from content string."""
        if not content:
            return []

        bullets = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "• ", "▸ ")):
                bullets.append(line[2:])
            elif line and line[0].isdigit() and "." in line[:3]:
                parts = line.split(".", 1)
                if len(parts) > 1:
                    bullets.append(parts[1].strip())

        return bullets

    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


async def create_ollama_vision_func(model: str = "llava") -> Optional[Callable]:
    """
    Create a vision LLM function using Ollama with a vision-capable model.

    Args:
        model: Ollama model name (llava, llava:13b, bakllava, etc.)

    Returns:
        Async function that takes (prompt, image_path) and returns response text.
    """
    try:
        import httpx
        import base64

        async def ollama_vision(prompt: str, image_path: str) -> str:
            """Call Ollama vision model with image."""
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": [image_data],
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    _logger.error(f"Ollama API error: {response.status_code}")
                    return ""

        # Test connection
        async with httpx.AsyncClient(timeout=5.0) as client:
            test = await client.get("http://localhost:11434/api/tags")
            if test.status_code != 200:
                _logger.warning("Ollama server not responding")
                return None

        return ollama_vision

    except ImportError:
        _logger.warning("httpx not installed, cannot use Ollama vision")
        return None
    except Exception as e:
        _logger.warning(f"Could not create Ollama vision function: {e}")
        return None


def apply_review_fixes(slide, issues: List[ReviewIssue]) -> int:
    """
    Apply auto-fixable review issues to a slide.

    Args:
        slide: python-pptx slide object
        issues: List of review issues

    Returns:
        Number of fixes applied
    """
    fixes_applied = 0

    for issue in issues:
        if not issue.auto_fixable:
            continue

        try:
            if issue.issue_type == "empty_placeholder":
                # Remove empty placeholders
                for shape in list(slide.shapes):
                    if hasattr(shape, 'is_placeholder') and shape.is_placeholder:
                        if not hasattr(shape, 'text') or not shape.text.strip():
                            sp = shape._element
                            sp.getparent().remove(sp)
                            fixes_applied += 1

        except Exception as e:
            _logger.warning(f"Failed to apply fix: {e}")

    return fixes_applied
