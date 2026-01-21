"""
AIDocumentIndexer - Text to SQL Service
========================================

Natural language to SQL query translation using LLMs.

Based on best practices from:
- LangChain SQL Agents: https://python.langchain.com/docs/tutorials/sql_qa/
- MindSQL patterns: https://github.com/eosphoros-ai/Awesome-Text2SQL

Features:
- Schema-aware query generation
- Few-shot prompting with verified examples
- Proper noun retrieval for exact entity matching
- Error recovery with re-prompting
- Query explanation generation
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

from backend.services.connectors.database.base import (
    BaseDatabaseConnector,
    DatabaseSchema,
    QueryResult,
)
from backend.services.text_to_sql.validators import SQLValidator
from backend.services.llm import get_chat_model

logger = structlog.get_logger(__name__)


class TextToSQLExample(BaseModel):
    """A verified example for few-shot prompting."""
    question: str
    sql: str
    explanation: Optional[str] = None


class TextToSQLResult(BaseModel):
    """Result of a text-to-SQL conversion and execution."""
    success: bool
    natural_language_query: str
    generated_sql: Optional[str] = None
    explanation: Optional[str] = None
    query_result: Optional[QueryResult] = None
    error: Optional[str] = None
    confidence: float = 0.0
    attempts: int = 1


# System prompt for SQL generation
SQL_GENERATION_PROMPT = """You are a SQL expert. Given a database schema and a natural language question, generate a valid SQL query that answers the question.

IMPORTANT RULES:
1. ONLY generate SELECT queries - never INSERT, UPDATE, DELETE, DROP, or any data-modifying statements
2. Use proper table and column names from the schema provided
3. Use appropriate JOINs when querying multiple tables
4. Add LIMIT clause for queries that might return many rows (default LIMIT 100)
5. Use table aliases for readability when joining tables
6. Handle NULL values appropriately
7. Use aggregate functions (COUNT, SUM, AVG, etc.) when the question asks for totals or averages
8. Use GROUP BY when using aggregates with other columns
9. Use ORDER BY when the question implies ranking or sorting
10. Output ONLY the SQL query, nothing else - no explanations, no markdown

DATABASE SCHEMA:
{schema}

{examples}

USER QUESTION: {question}

SQL QUERY:"""


SQL_EXPLANATION_PROMPT = """Given the following SQL query and the database schema, provide a brief, clear explanation of what this query does and what results it returns.

DATABASE SCHEMA:
{schema}

SQL QUERY:
{sql}

Provide a 1-2 sentence explanation:"""


SQL_ERROR_RECOVERY_PROMPT = """The SQL query you generated produced an error. Please fix the query.

DATABASE SCHEMA:
{schema}

ORIGINAL QUESTION: {question}

GENERATED QUERY:
{sql}

ERROR MESSAGE:
{error}

Generate a corrected SQL query that fixes this error. Output ONLY the SQL query:"""


class TextToSQLService:
    """
    Service for converting natural language questions to SQL queries.

    Uses LLM to generate SQL from natural language, with:
    - Schema context for accurate table/column references
    - Few-shot examples for improved accuracy
    - Query validation and safety checks
    - Error recovery with re-prompting
    """

    def __init__(
        self,
        connector: BaseDatabaseConnector,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.0,  # Low temperature for deterministic SQL
        max_retries: int = 2,
    ):
        self.connector = connector
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.validator = SQLValidator()
        self._examples: List[TextToSQLExample] = []
        self._schema_cache: Optional[DatabaseSchema] = None

    def add_example(self, question: str, sql: str, explanation: str = None):
        """Add a few-shot example for prompting."""
        self._examples.append(TextToSQLExample(
            question=question,
            sql=sql,
            explanation=explanation,
        ))

    def clear_examples(self):
        """Clear all few-shot examples."""
        self._examples.clear()

    def _format_examples(self) -> str:
        """Format examples for the prompt."""
        if not self._examples:
            return ""

        example_text = "EXAMPLES:\n"
        for i, ex in enumerate(self._examples[:5], 1):  # Limit to 5 examples
            example_text += f"\nQuestion {i}: {ex.question}\n"
            example_text += f"SQL {i}: {ex.sql}\n"
        return example_text

    async def _get_schema_ddl(self) -> str:
        """Get schema as DDL string for the prompt."""
        if not self._schema_cache:
            self._schema_cache = await self.connector.get_schema()
        return self._schema_cache.to_ddl_string(include_sample_values=True)

    async def generate_sql(self, question: str) -> Tuple[str, float]:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question

        Returns:
            Tuple of (generated_sql, confidence_score)
        """
        schema_ddl = await self._get_schema_ddl()
        examples = self._format_examples()

        prompt = SQL_GENERATION_PROMPT.format(
            schema=schema_ddl,
            examples=examples,
            question=question,
        )

        # Get LLM
        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=self.temperature,
        )

        # Generate SQL
        response = await llm.ainvoke(prompt)
        generated_sql = response.content.strip()

        # Clean up the response - remove any markdown code blocks
        generated_sql = self._clean_sql_response(generated_sql)

        # Estimate confidence based on response characteristics
        confidence = self._estimate_confidence(generated_sql, question)

        return generated_sql, confidence

    def _clean_sql_response(self, sql: str) -> str:
        """Clean up SQL response from LLM."""
        # Remove markdown code blocks
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'^```\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)

        # Remove any leading/trailing whitespace
        sql = sql.strip()

        # Remove trailing semicolon if present (we'll add if needed)
        sql = sql.rstrip(';')

        return sql

    def _estimate_confidence(self, sql: str, question: str) -> float:
        """
        Estimate confidence score for generated SQL.

        Based on:
        - Query structure validity
        - Table name matching
        - Keyword presence matching question intent
        """
        confidence = 0.5  # Base confidence

        # Check if it's a valid-looking SELECT query
        sql_upper = sql.upper()
        if sql_upper.startswith("SELECT") or sql_upper.startswith("WITH"):
            confidence += 0.1

        # Check for JOIN if question mentions relationships
        question_lower = question.lower()
        if any(word in question_lower for word in ["join", "related", "associated", "with", "and"]):
            if "JOIN" in sql_upper:
                confidence += 0.1

        # Check for aggregate functions if question asks for counts/totals
        if any(word in question_lower for word in ["how many", "count", "total", "sum", "average"]):
            if any(func in sql_upper for func in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
                confidence += 0.1

        # Check for ORDER BY if question asks for ranking
        if any(word in question_lower for word in ["top", "best", "worst", "highest", "lowest", "most", "least"]):
            if "ORDER BY" in sql_upper:
                confidence += 0.1

        # Check for GROUP BY with aggregates
        if any(func in sql_upper for func in ["COUNT(", "SUM(", "AVG("]):
            if "GROUP BY" in sql_upper:
                confidence += 0.05

        # Cap at 0.95 - never 100% confident
        return min(confidence, 0.95)

    async def generate_explanation(self, sql: str) -> str:
        """Generate a natural language explanation of the SQL query."""
        schema_ddl = await self._get_schema_ddl()

        prompt = SQL_EXPLANATION_PROMPT.format(
            schema=schema_ddl,
            sql=sql,
        )

        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=0.3,  # Slightly higher for more natural explanation
        )

        response = await llm.ainvoke(prompt)
        return response.content.strip()

    async def recover_from_error(
        self,
        question: str,
        sql: str,
        error: str,
    ) -> Tuple[str, float]:
        """
        Attempt to recover from a SQL error by re-prompting.

        Args:
            question: Original question
            sql: Failed SQL query
            error: Error message

        Returns:
            Tuple of (corrected_sql, confidence)
        """
        schema_ddl = await self._get_schema_ddl()

        prompt = SQL_ERROR_RECOVERY_PROMPT.format(
            schema=schema_ddl,
            question=question,
            sql=sql,
            error=error,
        )

        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=self.temperature,
        )

        response = await llm.ainvoke(prompt)
        corrected_sql = self._clean_sql_response(response.content.strip())

        # Lower confidence for recovered queries
        confidence = self._estimate_confidence(corrected_sql, question) * 0.8

        return corrected_sql, confidence

    async def query(
        self,
        question: str,
        execute: bool = True,
        explain: bool = True,
    ) -> TextToSQLResult:
        """
        Convert natural language question to SQL and optionally execute it.

        Args:
            question: Natural language question
            execute: Whether to execute the generated SQL
            explain: Whether to generate an explanation

        Returns:
            TextToSQLResult with SQL, explanation, and query results
        """
        result = TextToSQLResult(
            success=False,
            natural_language_query=question,
        )

        try:
            # Generate SQL
            sql, confidence = await self.generate_sql(question)
            result.generated_sql = sql
            result.confidence = confidence
            result.attempts = 1

            # Validate the query
            validation = self.validator.validate(sql)
            if not validation.is_valid:
                result.error = f"Query validation failed: {'; '.join(validation.errors)}"
                return result

            if not validation.is_read_only:
                result.error = "Generated query is not read-only"
                return result

            # Execute if requested
            if execute:
                query_result = await self.connector.execute_validated_query(sql)

                # If execution failed, try to recover
                retry_count = 0
                while not query_result.success and retry_count < self.max_retries:
                    retry_count += 1
                    result.attempts += 1

                    logger.info(
                        "Attempting SQL recovery",
                        attempt=retry_count,
                        error=query_result.error,
                    )

                    sql, confidence = await self.recover_from_error(
                        question, sql, query_result.error
                    )
                    result.generated_sql = sql
                    result.confidence = confidence

                    # Re-validate
                    validation = self.validator.validate(sql)
                    if not validation.is_valid or not validation.is_read_only:
                        continue

                    query_result = await self.connector.execute_validated_query(sql)

                result.query_result = query_result

                if not query_result.success:
                    result.error = query_result.error
                    return result

            # Generate explanation if requested
            if explain and result.generated_sql:
                result.explanation = await self.generate_explanation(result.generated_sql)

            result.success = True
            return result

        except Exception as e:
            logger.error("Text-to-SQL failed", error=str(e), question=question)
            result.error = str(e)
            return result

    async def get_relevant_tables(self, question: str) -> List[str]:
        """
        Identify which tables are relevant to answer a question.

        Useful for large databases to reduce schema context.
        """
        if not self._schema_cache:
            self._schema_cache = await self.connector.get_schema()

        # Get all table names
        table_names = [t.name for t in self._schema_cache.tables]

        # Use LLM to identify relevant tables
        prompt = f"""Given the following database tables and a user question, identify which tables are needed to answer the question.

TABLES:
{', '.join(table_names)}

QUESTION: {question}

Return ONLY a comma-separated list of relevant table names, nothing else:"""

        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=0.0,
        )

        response = await llm.ainvoke(prompt)
        tables = [t.strip() for t in response.content.split(',')]

        # Filter to only valid table names
        valid_tables = [t for t in tables if t in table_names]
        return valid_tables if valid_tables else table_names[:5]  # Fallback to first 5

    async def get_proper_nouns(
        self,
        table_name: str,
        column_name: str,
        search_term: Optional[str] = None,
        limit: int = 10,
    ) -> List[str]:
        """
        Get proper nouns (exact entity names) from a column.

        Useful for ensuring exact matches in WHERE clauses.
        """
        if hasattr(self.connector, 'get_distinct_values'):
            values = await self.connector.get_distinct_values(
                table_name, column_name, limit
            )
            if search_term:
                # Filter to values containing the search term
                search_lower = search_term.lower()
                values = [v for v in values if search_lower in str(v).lower()]
            return values
        return []
