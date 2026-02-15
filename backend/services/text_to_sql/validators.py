"""
AIDocumentIndexer - SQL Validators
===================================

SQL validation and safety checks for Text-to-SQL queries.
"""

import re
from typing import List, Optional, Set, Tuple

import sqlparse
import structlog

from backend.services.connectors.database.base import (
    QueryValidationResult,
    DANGEROUS_KEYWORDS,
    DANGEROUS_PATTERNS,
)

logger = structlog.get_logger(__name__)


class SQLValidator:
    """
    SQL query validator for ensuring read-only, safe queries.

    Uses multiple validation strategies:
    1. Keyword blocklist
    2. Pattern matching for dangerous constructs
    3. Statement type verification
    4. Table reference extraction
    """

    def __init__(
        self,
        allowed_tables: Set[str] = None,
        max_query_length: int = 10000,
    ):
        self.allowed_tables = allowed_tables or set()
        self.max_query_length = max_query_length

    def validate(self, query: str) -> QueryValidationResult:
        """
        Validate a SQL query for safety.

        Args:
            query: SQL query to validate

        Returns:
            QueryValidationResult with validation details
        """
        result = QueryValidationResult(
            is_valid=True,
            is_read_only=True,
            errors=[],
            warnings=[],
            tables_referenced=[],
        )

        # Check query length
        if len(query) > self.max_query_length:
            result.is_valid = False
            result.errors.append(f"Query exceeds maximum length of {self.max_query_length}")
            return result

        # Normalize for analysis
        normalized = query.upper().strip()

        # Remove string literals to avoid false positives
        normalized_clean = self._remove_string_literals(normalized)

        # Check for multiple statements
        if self._has_multiple_statements(normalized_clean):
            result.is_valid = False
            result.is_read_only = False
            result.errors.append("Multiple statements not allowed")

        # Determine query type
        result.query_type = self._get_query_type(normalized_clean)

        # Validate query type
        if result.query_type not in ("SELECT", "WITH", "EXPLAIN"):
            result.is_valid = False
            result.is_read_only = False
            result.errors.append(f"Only SELECT queries allowed, got: {result.query_type}")

        # Check WITH clause ends with SELECT
        if result.query_type == "WITH":
            if not self._with_ends_with_select(normalized_clean):
                result.is_valid = False
                result.errors.append("WITH clause must end with SELECT")

        # Check for dangerous keywords
        dangerous_found = self._check_dangerous_keywords(normalized_clean)
        if dangerous_found:
            result.is_valid = False
            result.is_read_only = False
            for keyword in dangerous_found:
                result.errors.append(f"Dangerous keyword found: {keyword}")

        # Check for dangerous patterns
        if self._check_dangerous_patterns(normalized):
            result.is_valid = False
            result.warnings.append("Suspicious pattern detected")

        # Extract referenced tables
        result.tables_referenced = self._extract_tables(normalized_clean)

        # Check table allowlist if configured
        if self.allowed_tables:
            for table in result.tables_referenced:
                if table.lower() not in {t.lower() for t in self.allowed_tables}:
                    result.is_valid = False
                    result.errors.append(f"Table not allowed: {table}")

        return result

    def _remove_string_literals(self, query: str) -> str:
        """Remove string literals to avoid false positives."""
        # Replace single-quoted strings
        query = re.sub(r"'[^']*'", "''", query)
        # Replace double-quoted identifiers
        query = re.sub(r'"[^"]*"', '""', query)
        return query

    def _has_multiple_statements(self, query: str) -> bool:
        """Check for multiple SQL statements."""
        # Split by semicolon and check for multiple non-empty statements
        statements = [s.strip() for s in query.split(';') if s.strip()]
        return len(statements) > 1

    def _get_query_type(self, query: str) -> str:
        """Get the type of SQL query."""
        query = query.strip()
        if query.startswith("SELECT"):
            return "SELECT"
        elif query.startswith("WITH"):
            return "WITH"
        elif query.startswith("EXPLAIN"):
            return "EXPLAIN"
        elif query:
            return query.split()[0]
        return "UNKNOWN"

    def _with_ends_with_select(self, query: str) -> bool:
        """Check that a WITH clause ends with SELECT."""
        # Find the last main clause after all CTEs
        # WITH clauses can have multiple CTEs separated by commas
        # The final statement should be SELECT
        parts = query.split("SELECT")
        if len(parts) < 2:
            return False

        # Check that there's a SELECT after all WITH/AS definitions
        last_part = parts[-1].strip()
        # If the last part starts with keywords that follow SELECT, it's valid
        return bool(last_part)

    def _check_dangerous_keywords(self, query: str) -> List[str]:
        """Check for dangerous keywords in query."""
        found = []
        for keyword in DANGEROUS_KEYWORDS:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query):
                found.append(keyword)
        return found

    def _check_dangerous_patterns(self, query: str) -> bool:
        """Check for dangerous patterns in query."""
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names referenced in the query."""
        tables = set()

        # Match FROM clause tables
        from_matches = re.findall(
            r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)',
            query
        )
        tables.update(from_matches)

        # Match JOIN clause tables
        join_matches = re.findall(
            r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)',
            query
        )
        tables.update(join_matches)

        # Extract just the table name if schema.table format
        clean_tables = []
        for table in tables:
            if '.' in table:
                clean_tables.append(table.split('.')[-1])
            else:
                clean_tables.append(table)

        return list(set(clean_tables))

    def validate_syntax(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Pre-validate SQL syntax using sqlparse before hitting the database.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Empty query"

        # Check unbalanced parentheses
        open_count = query.count("(")
        close_count = query.count(")")
        if open_count != close_count:
            return False, f"Unbalanced parentheses: {open_count} opening vs {close_count} closing"

        # Check unterminated string literals
        single_quotes = query.count("'")
        if single_quotes % 2 != 0:
            # Account for escaped quotes
            escaped = query.count("\\'")
            if (single_quotes - escaped) % 2 != 0:
                return False, "Unterminated string literal (unmatched single quote)"

        # Parse with sqlparse
        try:
            parsed = sqlparse.parse(query)
            if not parsed:
                return False, "Failed to parse SQL statement"

            stmt = parsed[0]

            # Check statement type is SELECT/WITH
            stmt_type = stmt.get_type()
            if stmt_type and stmt_type not in ("SELECT", "UNKNOWN"):
                return False, f"Only SELECT queries allowed, got: {stmt_type}"

            # Check for empty parsed statement
            tokens = [t for t in stmt.tokens if not t.is_whitespace]
            if not tokens:
                return False, "Empty SQL statement after parsing"

        except Exception as e:
            return False, f"SQL parse error: {str(e)}"

        return True, None

    def sanitize_identifier(self, identifier: str) -> str:
        """
        Sanitize a SQL identifier (table/column name).

        Args:
            identifier: The identifier to sanitize

        Returns:
            Sanitized identifier safe for use in queries
        """
        # Only allow alphanumeric and underscore
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            # Quote the identifier to make it safe
            escaped = identifier.replace('"', '""')
            return f'"{escaped}"'
        return identifier
