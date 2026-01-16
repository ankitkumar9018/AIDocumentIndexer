"""
XLSX Sheet Reviewer
===================

Provides LLM-based review of spreadsheet sheets to catch
issues like data quality, structure problems, and formatting.
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


class XLSXSheetReviewer(BaseContentReviewer):
    """Review spreadsheet sheets for data quality and constraints."""

    @property
    def format_name(self) -> str:
        return "xlsx"

    @property
    def item_name(self) -> str:
        return "sheet"

    async def review_item(
        self,
        item_content: Dict[str, Any],
        item_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Review a spreadsheet sheet."""
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
                item_type="sheet",
                has_issues=len(unique_issues) > 0,
                issues=unique_issues,
                overall_score=min(basic_result.overall_score, llm_result.overall_score),
            )
        except Exception as e:
            _logger.warning(f"LLM sheet review failed: {e}")
            return basic_result

    def _basic_review(
        self,
        sheet_content: Dict[str, Any],
        sheet_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Perform basic heuristic review."""
        issues = []

        # Get constraints
        max_sheet_name = constraints.get("sheet_name_max_chars", 31) if constraints else 31
        max_cell = constraints.get("cell_max_chars", 256) if constraints else 256
        max_header = constraints.get("header_max_chars", 50) if constraints else 50

        # Check sheet name
        sheet_name = sheet_content.get("name", "")
        if len(sheet_name) > max_sheet_name:
            issues.append(ReviewIssue(
                issue_type="text_overflow",
                description=f"Sheet name is {len(sheet_name)} chars (max {max_sheet_name})",
                severity="high",  # Excel enforces this limit
                suggestion=f"Shorten sheet name to {max_sheet_name} characters",
                location=f"Sheet {sheet_index + 1}",
                auto_fixable=True,
            ))

        # Check headers
        headers = sheet_content.get("headers", [])
        for i, header in enumerate(headers):
            if len(header) > max_header:
                issues.append(ReviewIssue(
                    issue_type="text_overflow",
                    description=f"Header '{header[:20]}...' is {len(header)} chars (max {max_header})",
                    severity="low",
                    suggestion="Shorten header text or use abbreviations",
                    location=f"Sheet {sheet_index + 1}, column {i + 1}",
                    auto_fixable=True,
                ))

        # Check rows for cell overflow
        rows = sheet_content.get("rows", [])
        for row_idx, row in enumerate(rows[:20]):  # Check first 20 rows
            cells = row.get("cells", []) if isinstance(row, dict) else row
            for col_idx, cell in enumerate(cells):
                value = cell.get("value", cell) if isinstance(cell, dict) else str(cell)
                if isinstance(value, str) and len(value) > max_cell:
                    issues.append(ReviewIssue(
                        issue_type="text_overflow",
                        description=f"Cell value is {len(value)} chars (max {max_cell})",
                        severity="medium",
                        suggestion="Truncate cell content or use notes",
                        location=f"Sheet {sheet_index + 1}, row {row_idx + 1}, col {col_idx + 1}",
                        auto_fixable=True,
                    ))

        # Check for empty headers
        if headers and any(not h.strip() for h in headers):
            issues.append(ReviewIssue(
                issue_type="structure",
                description="Some column headers are empty",
                severity="low",
                suggestion="Add descriptive headers for all columns",
                location=f"Sheet {sheet_index + 1}",
                auto_fixable=False,
            ))

        # Check for data consistency
        if rows and headers:
            expected_cols = len(headers)
            inconsistent_rows = []
            for row_idx, row in enumerate(rows[:50]):
                cells = row.get("cells", []) if isinstance(row, dict) else row
                if len(cells) != expected_cols:
                    inconsistent_rows.append(row_idx + 1)

            if inconsistent_rows:
                issues.append(ReviewIssue(
                    issue_type="structure",
                    description=f"Rows {inconsistent_rows[:5]} have inconsistent column counts",
                    severity="medium",
                    suggestion="Ensure all rows have the same number of columns",
                    location=f"Sheet {sheet_index + 1}",
                    auto_fixable=False,
                ))

        return ReviewResult(
            item_index=sheet_index,
            item_type="sheet",
            has_issues=len(issues) > 0,
            issues=issues,
            overall_score=max(0.0, 1.0 - (len(issues) * 0.15)),
        )

    async def _llm_review(
        self,
        sheet_content: Dict[str, Any],
        sheet_index: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Use LLM to review sheet content."""
        sheet_name = sheet_content.get("name", f"Sheet{sheet_index + 1}")
        headers = sheet_content.get("headers", [])
        rows = sheet_content.get("rows", [])

        # Format preview of data
        header_preview = ", ".join(headers[:8])
        if len(headers) > 8:
            header_preview += f" (+{len(headers) - 8} more)"

        data_preview = ""
        for i, row in enumerate(rows[:5]):
            cells = row.get("cells", []) if isinstance(row, dict) else row
            values = []
            for cell in cells[:5]:
                val = cell.get("value", cell) if isinstance(cell, dict) else str(cell)
                val_str = str(val)[:30] + "..." if len(str(val)) > 30 else str(val)
                values.append(val_str)
            data_preview += f"  Row {i + 1}: {', '.join(values)}\n"

        constraints_text = ""
        if constraints:
            constraints_text = f"""
CONSTRAINTS:
- Sheet name: max {constraints.get('sheet_name_max_chars', 31)} chars
- Cell content: max {constraints.get('cell_max_chars', 256)} chars
- Header: max {constraints.get('header_max_chars', 50)} chars
"""

        prompt = f"""Review this spreadsheet sheet for data quality and structure issues.

═══════════════════════════════════════════════════════════════════════
SHEET: "{sheet_name}" (Index: {sheet_index + 1})
═══════════════════════════════════════════════════════════════════════

STRUCTURE:
- Sheet name: "{sheet_name}" [{len(sheet_name)} chars]
- Columns: {len(headers)}
- Rows: {len(rows)}

HEADERS:
{header_preview}

DATA PREVIEW (first 5 rows):
{data_preview if data_preview else "  (no data)"}
{constraints_text}
═══════════════════════════════════════════════════════════════════════

REVIEW FOR:
1. TEXT OVERFLOW - Do sheet name/headers/cells fit constraints?
2. STRUCTURE - Are columns and rows well-organized?
3. DATA QUALITY - Is the data consistent and properly formatted?
4. HEADERS - Are column headers descriptive and appropriate?
5. COMPLETENESS - Does the data appear complete?

Return ONLY valid JSON (no markdown code blocks):
{{
    "issues": [
        {{
            "type": "text_overflow|structure|data_quality|headers|completeness",
            "description": "Specific description",
            "severity": "low|medium|high",
            "suggestion": "Actionable fix"
        }}
    ],
    "overall_score": 0.0 to 1.0,
    "summary": "One sentence summary"
}}

If the sheet looks good: {{"issues": [], "overall_score": 1.0, "summary": "Sheet is well-structured"}}
"""

        response = await self.llm_generate(prompt)
        return ReviewResult.parse_json(response, sheet_index, "sheet")

    async def fix_item(
        self,
        item_content: Dict[str, Any],
        review_result: ReviewResult,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FixResult:
        """Fix identified issues in sheet content."""
        if not review_result.has_issues:
            return FixResult(
                item_index=review_result.item_index,
                fixes_applied=0,
                original_content=item_content.copy(),
                fixed_content=item_content.copy(),
                changes_made=[],
            )

        max_sheet_name = constraints.get("sheet_name_max_chars", 31) if constraints else 31
        max_header = constraints.get("header_max_chars", 50) if constraints else 50
        max_cell = constraints.get("cell_max_chars", 256) if constraints else 256

        fixed_content = item_content.copy()
        changes_made = []
        fixes_applied = 0

        for issue in review_result.issues:
            try:
                if issue.issue_type == "text_overflow":
                    if "sheet name" in issue.description.lower():
                        name = fixed_content.get("name", "")
                        if len(name) > max_sheet_name:
                            # Sheet names must be truncated (Excel limit)
                            fixed_name = name[:max_sheet_name - 3] + "..."
                            fixed_content["name"] = fixed_name
                            changes_made.append(f"Truncated sheet name to {max_sheet_name} chars")
                            fixes_applied += 1

                    elif "header" in issue.description.lower():
                        headers = fixed_content.get("headers", [])
                        fixed_headers = []
                        for h in headers:
                            if len(h) > max_header:
                                h = await self._condense_text(h, max_header, "header")
                            fixed_headers.append(h)
                        if fixed_headers != headers:
                            fixed_content["headers"] = fixed_headers
                            changes_made.append("Condensed overflowing headers")
                            fixes_applied += 1

                    elif "cell" in issue.description.lower():
                        # Note: Cell fixing would require more complex row/cell traversal
                        # For now, just log that it needs attention
                        _logger.info("Cell overflow detected - manual review recommended")

            except Exception as e:
                _logger.warning(f"Failed to fix issue: {e}")

        return FixResult(
            item_index=review_result.item_index,
            fixes_applied=fixes_applied,
            original_content=item_content,
            fixed_content=fixed_content,
            changes_made=changes_made,
        )
