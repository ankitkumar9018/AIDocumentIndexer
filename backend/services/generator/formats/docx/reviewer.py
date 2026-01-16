"""
DOCX Section Reviewer
=====================

Provides LLM-based review of document sections to catch
issues like text overflow, structure problems, and content quality.
"""

from typing import List, Optional, Dict, Any

import structlog

from ..base_reviewer import (
    BaseContentReviewer,
    ReviewResult,
    ReviewIssue,
    FixResult,
    LLMGenerateFunc,
)

_logger = structlog.get_logger(__name__)


class DOCXSectionReviewer(BaseContentReviewer):
    """Review document sections for content quality and constraints."""

    @property
    def format_name(self) -> str:
        return "docx"

    @property
    def item_name(self) -> str:
        return "section"

    async def review_item(
        self,
        item_content: Dict[str, Any],
        item_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Review a document section."""
        # Always do basic checks first
        basic_result = self._basic_review(item_content, item_index, constraints)

        # If no LLM, return basic result
        if not self.llm_generate:
            return basic_result

        # Use LLM for deeper review
        try:
            llm_result = await self._llm_review(item_content, item_index, constraints)

            # Merge issues
            all_issues = basic_result.issues + llm_result.issues

            # Deduplicate
            seen = set()
            unique_issues = []
            for issue in all_issues:
                key = (issue.issue_type, issue.description[:50])
                if key not in seen:
                    seen.add(key)
                    unique_issues.append(issue)

            return ReviewResult(
                item_index=item_index,
                item_type="section",
                has_issues=len(unique_issues) > 0,
                issues=unique_issues,
                overall_score=min(basic_result.overall_score, llm_result.overall_score),
            )
        except Exception as e:
            _logger.warning(f"LLM section review failed: {e}")
            return basic_result

    def _basic_review(
        self,
        section_content: Dict[str, Any],
        section_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Perform basic heuristic review."""
        issues = []

        # Get constraints
        max_heading = constraints.get("heading_max_chars", 80) if constraints else 80
        max_paragraph = constraints.get("paragraph_max_chars", 2000) if constraints else 2000
        max_bullet = constraints.get("bullet_max_chars", 150) if constraints else 150

        # Check heading
        heading = section_content.get("heading", "")
        if len(heading) > max_heading:
            issues.append(ReviewIssue(
                issue_type="text_overflow",
                description=f"Heading is {len(heading)} chars (max {max_heading})",
                severity="medium",
                suggestion=f"Shorten heading to under {max_heading} characters",
                location=f"Section {section_index + 1} heading",
                auto_fixable=True,
            ))

        # Check paragraphs
        paragraphs = section_content.get("paragraphs", [])
        for i, para in enumerate(paragraphs):
            text = para.get("text", para) if isinstance(para, dict) else str(para)
            if len(text) > max_paragraph:
                issues.append(ReviewIssue(
                    issue_type="text_overflow",
                    description=f"Paragraph {i + 1} is {len(text)} chars (max {max_paragraph})",
                    severity="low",
                    suggestion="Consider breaking into smaller paragraphs",
                    location=f"Section {section_index + 1}, paragraph {i + 1}",
                    auto_fixable=False,
                ))

        # Check bullet points
        bullets = section_content.get("bullet_points", [])
        for i, bullet in enumerate(bullets):
            text = bullet.get("text", bullet) if isinstance(bullet, dict) else str(bullet)
            if len(text) > max_bullet:
                issues.append(ReviewIssue(
                    issue_type="text_overflow",
                    description=f"Bullet {i + 1} is {len(text)} chars (max {max_bullet})",
                    severity="low",
                    suggestion="Shorten bullet point",
                    location=f"Section {section_index + 1}, bullet {i + 1}",
                    auto_fixable=True,
                ))

        return ReviewResult(
            item_index=section_index,
            item_type="section",
            has_issues=len(issues) > 0,
            issues=issues,
            overall_score=max(0.0, 1.0 - (len(issues) * 0.15)),
        )

    async def _llm_review(
        self,
        section_content: Dict[str, Any],
        section_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Use LLM to review section content."""
        heading = section_content.get("heading", "")
        heading_level = section_content.get("heading_level", 1)
        paragraphs = section_content.get("paragraphs", [])
        bullets = section_content.get("bullet_points", [])

        # Format content for prompt
        para_text = ""
        total_words = 0
        for i, para in enumerate(paragraphs):
            text = para.get("text", para) if isinstance(para, dict) else str(para)
            words = len(text.split())
            total_words += words
            preview = text[:200] + "..." if len(text) > 200 else text
            para_text += f"  {i + 1}. [{words} words] {preview}\n"

        bullets_text = ""
        for i, bullet in enumerate(bullets):
            text = bullet.get("text", bullet) if isinstance(bullet, dict) else str(bullet)
            bullets_text += f"  • [{len(text)} chars] {text}\n"

        constraints_text = ""
        if constraints:
            constraints_text = f"""
CONSTRAINTS:
- Max heading: {constraints.get('heading_max_chars', 80)} chars
- Max paragraph: {constraints.get('paragraph_max_chars', 2000)} chars
- Max bullet: {constraints.get('bullet_max_chars', 150)} chars
"""

        # Theme info
        font_heading = section_content.get("font_heading", "Calibri")
        font_body = section_content.get("font_body", "Calibri")

        prompt = f"""Review this document section for quality and formatting issues.

═══════════════════════════════════════════════════════════════════════
SECTION {section_index + 1}
═══════════════════════════════════════════════════════════════════════

HEADING:
- Level: H{heading_level}
- Text: "{heading}" [{len(heading)} chars]
- Font: {font_heading}

PARAGRAPHS ({len(paragraphs)} total, {total_words} words):
{para_text if para_text else "  (no paragraphs)"}

BULLET POINTS ({len(bullets)} total):
{bullets_text if bullets_text else "  (no bullets)"}
{constraints_text}
═══════════════════════════════════════════════════════════════════════

REVIEW FOR:
1. TEXT OVERFLOW - Do heading/paragraphs/bullets fit constraints?
2. STRUCTURE - Is the section well-organized?
3. READABILITY - Is the text clear and easy to read?
4. COMPLETENESS - Does the section cover its topic adequately?
5. CONTENT QUALITY - Is the information accurate and professional?

Return ONLY valid JSON (no markdown code blocks):
{{
    "issues": [
        {{
            "type": "text_overflow|structure|readability|completeness|content_quality",
            "description": "Specific description",
            "severity": "low|medium|high",
            "suggestion": "Actionable fix"
        }}
    ],
    "overall_score": 0.0 to 1.0,
    "summary": "One sentence summary"
}}

If the section looks good: {{"issues": [], "overall_score": 1.0, "summary": "Section is well-written"}}
"""

        response = await self.llm_generate(prompt)
        return ReviewResult.parse_json(response, section_index, "section")

    async def fix_item(
        self,
        item_content: Dict[str, Any],
        review_result: ReviewResult,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FixResult:
        """Fix identified issues in section content."""
        if not review_result.has_issues:
            return FixResult(
                item_index=review_result.item_index,
                fixes_applied=0,
                original_content=item_content.copy(),
                fixed_content=item_content.copy(),
                changes_made=[],
            )

        max_heading = constraints.get("heading_max_chars", 80) if constraints else 80
        max_bullet = constraints.get("bullet_max_chars", 150) if constraints else 150

        fixed_content = item_content.copy()
        changes_made = []
        fixes_applied = 0

        for issue in review_result.issues:
            try:
                if issue.issue_type == "text_overflow":
                    if "heading" in issue.description.lower():
                        heading = fixed_content.get("heading", "")
                        if len(heading) > max_heading:
                            fixed_heading = await self._condense_text(heading, max_heading, "heading")
                            fixed_content["heading"] = fixed_heading
                            changes_made.append(f"Condensed heading from {len(heading)} to {len(fixed_heading)} chars")
                            fixes_applied += 1

                    elif "bullet" in issue.description.lower():
                        bullets = fixed_content.get("bullet_points", [])
                        fixed_bullets = []
                        for bullet in bullets:
                            text = bullet.get("text", bullet) if isinstance(bullet, dict) else str(bullet)
                            if len(text) > max_bullet:
                                text = await self._condense_text(text, max_bullet, "bullet")
                            fixed_bullets.append(text)
                        if fixed_bullets != bullets:
                            fixed_content["bullet_points"] = fixed_bullets
                            changes_made.append("Condensed overflowing bullets")
                            fixes_applied += 1

            except Exception as e:
                _logger.warning(f"Failed to fix issue: {e}")

        return FixResult(
            item_index=review_result.item_index,
            fixes_applied=fixes_applied,
            original_content=item_content,
            fixed_content=fixed_content,
            changes_made=changes_made,
        )
