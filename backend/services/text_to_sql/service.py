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

    # =========================================================================
    # Phase 65: Interactive Query Building
    # =========================================================================

    async def classify_query(self, question: str) -> Dict[str, Any]:
        """
        Classify a query to determine if clarification is needed.

        Returns:
            Dict with 'type' (clear/ambiguous/unanswerable) and 'clarifications' if needed
        """
        prompt = f"""Analyze this database question and classify it:

Question: {question}

Determine if the question is:
1. CLEAR - Can be directly translated to SQL with available schema
2. AMBIGUOUS - Needs clarification (e.g., unclear time range, multiple interpretations)
3. UNANSWERABLE - Cannot be answered with the available data

If AMBIGUOUS, provide 2-3 clarifying questions.

Respond in this JSON format only:
{{
    "type": "clear|ambiguous|unanswerable",
    "confidence": 0.0-1.0,
    "clarifications": ["question1", "question2"],
    "interpretation": "How you understood the question"
}}"""

        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=0.1,
        )

        response = await llm.ainvoke(prompt)

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Failed to parse classification", error=str(e))

        return {"type": "clear", "confidence": 0.5, "clarifications": []}

    async def interactive_query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Interactive query building with clarification flow.

        Args:
            question: User's question
            context: Previous context from clarification flow

        Returns:
            Dict with status, clarifications needed, or query result
        """
        # Classify the query first
        classification = await self.classify_query(question)

        if classification["type"] == "ambiguous":
            return {
                "status": "clarification_needed",
                "questions": classification.get("clarifications", []),
                "interpretation": classification.get("interpretation", ""),
            }

        if classification["type"] == "unanswerable":
            return {
                "status": "unanswerable",
                "reason": "This question cannot be answered with the available data.",
            }

        # Generate SQL with preview
        sql, confidence = await self.generate_sql(question)

        # Validate
        validation = self.validator.validate(sql)
        if not validation.is_valid:
            return {
                "status": "error",
                "errors": validation.errors,
            }

        # Estimate cost
        cost = await self.estimate_query_cost(sql)

        # Get preview (limited rows)
        preview = None
        if validation.is_read_only:
            preview_sql = sql
            if "LIMIT" not in sql.upper():
                preview_sql = f"{sql} LIMIT 5"
            try:
                preview_result = await self.connector.execute_validated_query(preview_sql)
                if preview_result.success:
                    preview = preview_result.rows[:5]
            except Exception:
                pass

        return {
            "status": "ready",
            "sql": sql,
            "explanation": await self.generate_explanation(sql),
            "confidence": confidence,
            "estimated_cost": cost,
            "preview": preview,
        }

    async def estimate_query_cost(self, sql: str) -> Dict[str, Any]:
        """
        Estimate the cost/complexity of a SQL query before execution.

        Returns estimated row count, execution complexity, and warnings.
        """
        cost = {
            "estimated_rows": None,
            "complexity": "low",
            "warnings": [],
            "estimated_time_ms": None,
        }

        sql_upper = sql.upper()

        # Check for expensive operations
        if "SELECT *" in sql_upper:
            cost["warnings"].append("Using SELECT * - consider specifying columns")

        if "JOIN" in sql_upper:
            join_count = sql_upper.count("JOIN")
            if join_count > 2:
                cost["complexity"] = "high"
                cost["warnings"].append(f"Query has {join_count} JOINs - may be slow")
            elif join_count > 0:
                cost["complexity"] = "medium"

        if "LIKE '%%" in sql_upper:
            cost["complexity"] = "high"
            cost["warnings"].append("Leading wildcard in LIKE - cannot use index")

        if "ORDER BY" in sql_upper and "LIMIT" not in sql_upper:
            cost["warnings"].append("ORDER BY without LIMIT - may sort large result set")

        if "GROUP BY" in sql_upper:
            cost["complexity"] = "medium"

        # Try to get EXPLAIN output if supported
        try:
            if hasattr(self.connector, 'explain_query'):
                explain_result = await self.connector.explain_query(sql)
                if explain_result:
                    cost["explain"] = explain_result
                    # Parse estimated rows from EXPLAIN if available
                    if "rows=" in str(explain_result):
                        row_match = re.search(r'rows=(\d+)', str(explain_result))
                        if row_match:
                            cost["estimated_rows"] = int(row_match.group(1))
        except Exception:
            pass

        return cost

    # =========================================================================
    # Phase 65: Auto-Visualization (LIDA-style)
    # =========================================================================

    async def suggest_visualization(
        self,
        sql: str,
        query_result: QueryResult,
    ) -> Dict[str, Any]:
        """
        Suggest the best visualization for query results.

        Based on Microsoft LIDA patterns for automatic chart generation.
        """
        if not query_result.success or not query_result.rows:
            return {"type": None, "reason": "No data to visualize"}

        # Analyze result structure
        columns = query_result.columns or []
        rows = query_result.rows
        sample = rows[:10] if len(rows) > 10 else rows

        # Determine data types
        numeric_cols = []
        categorical_cols = []
        date_cols = []

        for col in columns:
            sample_values = [row.get(col) for row in sample if row.get(col) is not None]
            if not sample_values:
                continue

            # Check if numeric
            try:
                [float(v) for v in sample_values[:5]]
                numeric_cols.append(col)
            except (ValueError, TypeError):
                # Check if date-like
                if any(isinstance(v, str) and re.match(r'\d{4}-\d{2}', str(v)) for v in sample_values[:3]):
                    date_cols.append(col)
                else:
                    categorical_cols.append(col)

        # Suggest visualization based on data structure
        suggestion = {
            "type": None,
            "x_axis": None,
            "y_axis": None,
            "group_by": None,
            "insights": [],
        }

        row_count = len(rows)

        # Time series: date column + numeric
        if date_cols and numeric_cols:
            suggestion["type"] = "line"
            suggestion["x_axis"] = date_cols[0]
            suggestion["y_axis"] = numeric_cols[0]
            suggestion["insights"].append("Time series data - line chart recommended")

        # Single numeric with grouping: bar chart
        elif categorical_cols and numeric_cols:
            if len(rows) <= 20:
                suggestion["type"] = "bar"
                suggestion["x_axis"] = categorical_cols[0]
                suggestion["y_axis"] = numeric_cols[0]
                suggestion["insights"].append("Categorical comparison - bar chart recommended")
            else:
                suggestion["type"] = "scatter"
                suggestion["insights"].append("Many categories - scatter plot recommended")

        # Two numeric columns: scatter
        elif len(numeric_cols) >= 2:
            suggestion["type"] = "scatter"
            suggestion["x_axis"] = numeric_cols[0]
            suggestion["y_axis"] = numeric_cols[1]
            suggestion["insights"].append("Two numeric columns - scatter plot recommended")

        # Single column with counts: pie chart
        elif len(columns) == 2 and numeric_cols and row_count <= 10:
            suggestion["type"] = "pie"
            suggestion["group_by"] = categorical_cols[0] if categorical_cols else columns[0]
            suggestion["insights"].append("Distribution data - pie chart recommended")

        # Fallback to table
        else:
            suggestion["type"] = "table"
            suggestion["insights"].append("Complex data - table view recommended")

        return suggestion

    async def generate_chart_code(
        self,
        query_result: QueryResult,
        chart_type: str,
        library: str = "plotly",
    ) -> str:
        """
        Generate visualization code for the query results.

        Args:
            query_result: Query results to visualize
            chart_type: Type of chart (bar, line, scatter, pie)
            library: Visualization library (plotly, matplotlib)

        Returns:
            Python code for generating the chart
        """
        suggestion = await self.suggest_visualization("", query_result)

        prompt = f"""Generate {library} code to create a {chart_type} chart.

Data columns: {query_result.columns}
Sample data (first 3 rows): {query_result.rows[:3]}
X-axis: {suggestion.get('x_axis', 'auto')}
Y-axis: {suggestion.get('y_axis', 'auto')}

Generate ONLY the Python code, no explanations:"""

        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=0.2,
        )

        response = await llm.ainvoke(prompt)

        # Clean up code response
        code = response.content.strip()
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'\s*```$', '', code)

        return code

    async def query_with_visualization(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Full pipeline: question -> SQL -> execute -> visualize.

        Returns query results with visualization suggestion.
        """
        # Run the query
        result = await self.query(question, execute=True, explain=True)

        if not result.success or not result.query_result:
            return {
                "success": False,
                "error": result.error,
            }

        # Get visualization suggestion
        viz = await self.suggest_visualization(
            result.generated_sql,
            result.query_result,
        )

        return {
            "success": True,
            "sql": result.generated_sql,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "rows": result.query_result.rows,
            "columns": result.query_result.columns,
            "row_count": result.query_result.row_count,
            "visualization": viz,
        }
