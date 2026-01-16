"""
Base Content Reviewer
=====================

Abstract base class for format-specific content reviewers.
Provides common functionality for LLM-based content review and auto-fix.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable

import structlog

_logger = structlog.get_logger(__name__)


# Type alias for LLM generate function
LLMGenerateFunc = Callable[[str], Awaitable[str]]


@dataclass
class ReviewIssue:
    """An issue found during content review."""
    issue_type: str  # text_overflow, content_quality, formatting, structure, etc.
    description: str
    severity: str  # low, medium, high
    suggestion: str
    location: Optional[str] = None  # e.g., "section 2", "slide 3", "cell B5"
    auto_fixable: bool = False
    fix_action: Optional[str] = None
    fix_params: Optional[Dict[str, Any]] = None


@dataclass
class ReviewResult:
    """Result of reviewing content."""
    item_index: int  # Index of the item reviewed (slide, section, sheet)
    item_type: str  # "slide", "section", "sheet", etc.
    has_issues: bool
    issues: List[ReviewIssue] = field(default_factory=list)
    overall_score: float = 1.0  # 0-1, 1 being perfect

    @classmethod
    def parse_json(cls, response: str, item_index: int, item_type: str) -> "ReviewResult":
        """Parse LLM JSON response into ReviewResult."""
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
                    location=issue_data.get("location"),
                    auto_fixable=issue_data.get("auto_fixable", False),
                    fix_params=issue_data.get("fix_params"),
                ))
            return cls(
                item_index=item_index,
                item_type=item_type,
                has_issues=len(issues) > 0,
                issues=issues,
                overall_score=data.get("overall_score", 1.0 if not issues else 0.5),
            )
        except (json.JSONDecodeError, KeyError) as e:
            _logger.warning(f"Failed to parse review response: {e}")
            return cls(item_index=item_index, item_type=item_type, has_issues=False)


@dataclass
class FixResult:
    """Result of fixing content."""
    item_index: int
    fixes_applied: int
    original_content: Dict[str, Any]
    fixed_content: Dict[str, Any]
    changes_made: List[str]


class BaseContentReviewer(ABC):
    """
    Abstract base class for content reviewers.

    Subclasses implement format-specific review and fix logic.
    """

    def __init__(self, llm_generate_func: Optional[LLMGenerateFunc] = None):
        """
        Initialize the content reviewer.

        Args:
            llm_generate_func: Async function that takes a prompt string and returns LLM response.
                              If not provided, will use basic heuristic checks only.
        """
        self.llm_generate = llm_generate_func

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name (e.g., 'pptx', 'docx', 'xlsx')."""
        ...

    @property
    @abstractmethod
    def item_name(self) -> str:
        """Return the name of items being reviewed (e.g., 'slide', 'section', 'sheet')."""
        ...

    @abstractmethod
    async def review_item(
        self,
        item_content: Dict[str, Any],
        item_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """
        Review a single item (slide, section, sheet).

        Args:
            item_content: Content of the item to review
            item_index: Index of the item
            constraints: Optional constraints from template

        Returns:
            ReviewResult with any issues found
        """
        ...

    @abstractmethod
    async def fix_item(
        self,
        item_content: Dict[str, Any],
        review_result: ReviewResult,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FixResult:
        """
        Fix identified issues in item content.

        Args:
            item_content: Original content
            review_result: Review result with issues
            constraints: Constraints for fixes

        Returns:
            FixResult with fixed content
        """
        ...

    async def review_and_fix(
        self,
        item_content: Dict[str, Any],
        item_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> tuple[ReviewResult, FixResult]:
        """
        Review an item and automatically fix any issues.

        Args:
            item_content: Content to review and fix
            item_index: Index of the item
            constraints: Optional constraints

        Returns:
            Tuple of (ReviewResult, FixResult)
        """
        review_result = await self.review_item(item_content, item_index, constraints)

        if review_result.has_issues:
            fix_result = await self.fix_item(item_content, review_result, constraints)

            if fix_result.fixes_applied > 0:
                _logger.info(
                    f"Applied {self.item_name} fixes",
                    item=item_index + 1,
                    fixes=fix_result.fixes_applied,
                    changes=fix_result.changes_made,
                )

            return review_result, fix_result
        else:
            return review_result, FixResult(
                item_index=item_index,
                fixes_applied=0,
                original_content=item_content,
                fixed_content=item_content,
                changes_made=[],
            )

    async def review_all(
        self,
        items: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[ReviewResult]:
        """
        Review all items.

        Args:
            items: List of item content dicts
            constraints: Optional constraints

        Returns:
            List of ReviewResults
        """
        results = []
        for idx, item in enumerate(items):
            result = await self.review_item(item, idx, constraints)
            results.append(result)

            if result.has_issues:
                for issue in result.issues:
                    _logger.warning(
                        f"{self.format_name} review issue",
                        item=idx + 1,
                        type=issue.issue_type,
                        severity=issue.severity,
                        description=issue.description,
                    )

        return results

    # =========================================================================
    # Common helper methods
    # =========================================================================

    async def _condense_text(
        self, text: str, max_chars: int, text_type: str = "text"
    ) -> str:
        """Condense text to fit within max_chars using LLM or heuristics."""
        if len(text) <= max_chars:
            return text

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

                if len(condensed) <= max_chars:
                    return condensed
                else:
                    _logger.warning(f"LLM condensed text still too long: {len(condensed)} > {max_chars}")
            except Exception as e:
                _logger.warning(f"LLM text condensing failed: {e}")

        return self._smart_truncate(text, max_chars)

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """Truncate text intelligently at word boundaries."""
        if len(text) <= max_chars:
            return text

        target_len = max_chars - 3
        truncated = text[:target_len]
        last_space = truncated.rfind(" ")

        if last_space > target_len * 0.7:
            truncated = truncated[:last_space]

        return truncated.rstrip() + "..."

    def _basic_text_checks(
        self,
        text: str,
        max_chars: int,
        text_type: str,
        location: str,
    ) -> List[ReviewIssue]:
        """Perform basic text validation checks."""
        issues = []

        if len(text) > max_chars:
            issues.append(ReviewIssue(
                issue_type="text_overflow",
                description=f"{text_type} is {len(text)} chars (max {max_chars})",
                severity="medium",
                suggestion=f"Shorten {text_type.lower()} to under {max_chars} characters",
                location=location,
                auto_fixable=True,
                fix_action="condense",
                fix_params={"max_chars": max_chars, "text_type": text_type.lower()},
            ))

        return issues
