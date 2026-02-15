"""
AIDocumentIndexer - Text to SQL Service
========================================

Natural language to SQL query translation using LLMs.

Based on best practices from:
- LangChain SQL Agents: https://python.langchain.com/docs/tutorials/sql_qa/
- MindSQL patterns: https://github.com/eosphoros-ai/Awesome-Text2SQL

Features:
- Schema-aware query generation
- Schema linking for table pruning
- Few-shot prompting with verified examples (static + dynamic)
- Proper noun retrieval for exact entity matching
- Error recovery with classification and targeted hints
- Query complexity routing with adaptive prompts
- Self-consistency multi-candidate generation
- Sample row injection for value-aware SQL
- SQL syntax pre-validation
- Query explanation and answer generation
"""

import asyncio
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
    answer: Optional[str] = None
    query_result: Optional[QueryResult] = None
    error: Optional[str] = None
    confidence: float = 0.0
    attempts: int = 1
    # Enhanced metadata
    complexity: Optional[str] = None
    schema_tables_used: List[str] = Field(default_factory=list)
    candidates_generated: int = 1
    features_used: List[str] = Field(default_factory=list)


# =============================================================================
# Prompt Templates
# =============================================================================

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
10. Output ONLY the raw SQL query - no explanations, no markdown code fences, no comments

DATABASE SCHEMA:
{schema}

{sample_data}

{examples}

USER QUESTION: {question}

Respond with ONLY the SQL query, nothing else:"""


SQL_GENERATION_PROMPT_EASY = """You are a SQL expert. Generate a SQL SELECT query for this question.

Rules: Only SELECT. Use exact column/table names from schema. Add LIMIT 100 if needed. Output ONLY the SQL, no explanations.

SCHEMA:
{schema}

{sample_data}

{examples}

QUESTION: {question}

SQL:"""


SQL_GENERATION_PROMPT_HARD = """You are a SQL expert. Given a database schema and a complex question, think step-by-step to generate the correct SQL query.

IMPORTANT RULES:
1. ONLY generate SELECT queries
2. Use proper table and column names from the schema
3. Use JOINs, subqueries, window functions as needed
4. Add LIMIT clause for large result sets (default LIMIT 100)
5. Handle NULL values and edge cases

Think through the problem:
1. What tables and columns are needed?
2. What JOINs or subqueries are required?
3. What aggregations, groupings, or window functions apply?
4. What ordering and filtering is needed?

DATABASE SCHEMA:
{schema}

{sample_data}

{examples}

USER QUESTION: {question}

Think step-by-step, then output the final SQL query after "SQL:":"""


SQL_EXPLANATION_PROMPT = """Given the following SQL query and the database schema, provide a brief, clear explanation of what this query does and what results it returns.

DATABASE SCHEMA:
{schema}

SQL QUERY:
{sql}

Provide a 1-2 sentence explanation:"""


SQL_ERROR_RECOVERY_PROMPT = """The SQL query you generated produced an error. Please fix the query.

ERROR TYPE: {error_type}
HINT: {error_hint}

DATABASE SCHEMA:
{schema}

ORIGINAL QUESTION: {question}

GENERATED QUERY:
{sql}

ERROR MESSAGE:
{error}

Generate a corrected SQL query that fixes this error. Output ONLY the SQL query:"""


SQL_ANSWER_GENERATION_PROMPT = """You are a data analyst. A user asked a question about their database, and the SQL query has been executed. Based on the question and the query results, provide a clear, concise natural language answer.

USER QUESTION: {question}

SQL QUERY:
{sql}

QUERY RESULTS ({row_count} rows):
{results_summary}

Provide a helpful, human-readable answer that directly addresses the user's question. Include specific numbers, names, and key facts from the results. If the results are empty, explain that no matching data was found. Keep the answer to 2-4 sentences unless the question requires more detail."""


# =============================================================================
# Error Classification
# =============================================================================

ERROR_CLASSIFICATIONS = {
    "column_not_found": {
        "patterns": [
            r"column .+ does not exist",
            r"Unknown column",
            r"no such column",
            r"invalid column name",
        ],
        "hint": "Check column names against the schema. The column may be in a different table or have a different name.",
    },
    "table_not_found": {
        "patterns": [
            r"relation .+ does not exist",
            r"Table .+ doesn't exist",
            r"no such table",
            r"invalid object name",
        ],
        "hint": "Check table names against the schema. Use exact table names as shown in the schema.",
    },
    "syntax_error": {
        "patterns": [
            r"syntax error",
            r"parse error",
            r"unexpected token",
        ],
        "hint": "Fix the SQL syntax. Check for missing commas, parentheses, or keywords.",
    },
    "ambiguous_column": {
        "patterns": [
            r"ambiguous column",
            r"column reference .+ is ambiguous",
        ],
        "hint": "Use table aliases to qualify ambiguous column references (e.g., t.column_name).",
    },
    "type_mismatch": {
        "patterns": [
            r"type mismatch",
            r"invalid input syntax for type",
            r"cannot cast",
            r"operator does not exist",
        ],
        "hint": "Check data types. You may need CAST() or a different comparison operator.",
    },
    "aggregation_error": {
        "patterns": [
            r"must appear in the GROUP BY",
            r"not a GROUP BY expression",
            r"is not in the GROUP BY",
        ],
        "hint": "Every non-aggregated column in SELECT must appear in GROUP BY.",
    },
}


def _classify_error(error: str) -> Dict[str, str]:
    """Classify a SQL error and return type + targeted hint."""
    error_lower = error.lower()
    for error_type, config in ERROR_CLASSIFICATIONS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, error_lower, re.IGNORECASE):
                return {
                    "error_type": error_type,
                    "error_hint": config["hint"],
                }
    return {
        "error_type": "unknown",
        "error_hint": "Review the query carefully against the schema and fix the issue.",
    }


# =============================================================================
# Main Service
# =============================================================================

class TextToSQLService:
    """
    Service for converting natural language questions to SQL queries.

    Uses LLM to generate SQL from natural language, with:
    - Schema context with optional table pruning (schema linking)
    - Few-shot examples (static + dynamic embedding-based)
    - Query validation and safety checks
    - Error recovery with classification
    - Complexity-adaptive prompts
    - Self-consistency multi-candidate generation
    - Sample row injection
    """

    def __init__(
        self,
        connector: BaseDatabaseConnector,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 2,
        schema_annotations: Optional[Dict[str, Any]] = None,
        # Feature flags (can be overridden per-query)
        schema_linking: Optional[bool] = None,
        sample_rows: Optional[bool] = None,
        self_consistency: Optional[bool] = None,
        num_candidates: int = 3,
        dynamic_examples: Optional[bool] = None,
        complexity_routing: Optional[bool] = None,
    ):
        self.connector = connector
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.validator = SQLValidator()
        self._examples: List[TextToSQLExample] = []
        self._schema_cache: Optional[DatabaseSchema] = None
        self._schema_annotations = schema_annotations

        # Feature flags
        self._schema_linking = schema_linking
        self._sample_rows = sample_rows
        self._self_consistency = self_consistency
        self._num_candidates = num_candidates
        self._dynamic_examples = dynamic_examples
        self._complexity_routing = complexity_routing

        # Read settings from settings service (synchronous defaults for init)
        try:
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            stored_timeout = settings_svc.get_default_value("database.query_timeout_seconds")
            if stored_timeout is not None:
                self.query_timeout = int(stored_timeout)
            else:
                self.query_timeout = 30
            stored_retries = settings_svc.get_default_value("database.max_retries")
            if stored_retries is not None:
                self.max_retries = int(stored_retries)
        except Exception:
            self.query_timeout = 30

    async def _resolve_feature_flag(self, flag_value: Optional[bool], setting_key: str, default: bool) -> bool:
        """Resolve a feature flag: per-query override > admin setting > default."""
        if flag_value is not None:
            return flag_value
        try:
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            val = await settings_svc.get_setting(setting_key)
            if val is not None:
                return bool(val)
        except Exception:
            pass
        return default

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

    def _format_examples(self, examples: Optional[List[TextToSQLExample]] = None) -> str:
        """Format examples for the prompt."""
        exs = examples or self._examples
        if not exs:
            return ""

        example_text = "EXAMPLES:\n"
        for i, ex in enumerate(exs[:5], 1):
            example_text += f"\nQuestion {i}: {ex.question}\n"
            example_text += f"SQL {i}: {ex.sql}\n"
        return example_text

    async def _get_schema(self) -> DatabaseSchema:
        """Get the database schema (cached)."""
        if not self._schema_cache:
            self._schema_cache = await self.connector.get_schema()
        return self._schema_cache

    async def _get_schema_ddl(self, compact: bool = False) -> str:
        """Get full schema as DDL string for the prompt."""
        schema = await self._get_schema()
        if compact:
            return schema.to_compact_ddl(annotations=self._schema_annotations)
        return schema.to_ddl_string(
            include_sample_values=True,
            annotations=self._schema_annotations,
        )

    async def _get_linked_schema_ddl(
        self,
        question: str,
        compact: bool = False,
        use_embeddings: bool = False,
    ) -> Tuple[str, List[str]]:
        """
        Get schema DDL pruned to relevant tables via schema linking.

        Returns:
            Tuple of (ddl_string, list_of_table_names_used)
        """
        schema = await self._get_schema()

        # Only use linking if there are enough tables to benefit
        if len(schema.tables) <= 3:
            ddl = await self._get_schema_ddl(compact=compact)
            return ddl, [t.name for t in schema.tables]

        from backend.services.text_to_sql.schema_linker import SchemaLinker
        linker = SchemaLinker(schema, annotations=self._schema_annotations)
        linked_tables = await linker.link(
            question, use_embeddings=use_embeddings, max_tables=8
        )

        # Build a pruned schema with only selected tables
        pruned = DatabaseSchema(
            database_name=schema.database_name,
            connector_type=schema.connector_type,
            tables=[t for t in schema.tables if t.name in linked_tables],
            views=schema.views,
            last_updated=schema.last_updated,
        )

        if compact:
            ddl = pruned.to_compact_ddl(annotations=self._schema_annotations)
        else:
            ddl = pruned.to_ddl_string(
                include_sample_values=True,
                annotations=self._schema_annotations,
            )

        return ddl, linked_tables

    async def _get_sample_rows(self, table_names: List[str], limit: int = 3) -> str:
        """Get sample data rows for the given tables, formatted as markdown."""
        parts = []
        for table_name in table_names[:5]:  # Max 5 tables
            try:
                sample_result = await self.connector.get_sample_data(table_name, limit=limit)
                if sample_result.success and sample_result.rows and sample_result.columns:
                    # Format as markdown table
                    header = " | ".join(str(c) for c in sample_result.columns)
                    separator = " | ".join("---" for _ in sample_result.columns)
                    rows = []
                    for row in sample_result.rows[:limit]:
                        cells = []
                        for v in row:
                            s = str(v) if v is not None else "NULL"
                            cells.append(s[:50])  # Truncate long values
                        rows.append(" | ".join(cells))

                    parts.append(f"-- Sample rows from {table_name}:\n-- {header}\n-- {separator}")
                    for r in rows:
                        parts.append(f"-- {r}")
            except Exception as e:
                logger.debug("Failed to get sample data", table=table_name, error=str(e))

        if parts:
            return "SAMPLE DATA:\n" + "\n".join(parts)
        return ""

    async def _get_entity_context(self, question: str) -> str:
        """Get entity names from knowledge graph for better WHERE clause matching."""
        try:
            from backend.services.knowledge_graph import get_kg_service
            from backend.services.settings import get_settings_service

            settings_svc = get_settings_service()
            kg_enabled = await settings_svc.get_setting("rag.knowledge_graph_enabled")
            if not kg_enabled:
                return ""

            kg_service = await get_kg_service()
            entities = await kg_service.find_entities_by_query(question, limit=10)
            if not entities:
                return ""

            entity_lines = []
            for e in entities[:8]:
                entity_lines.append(f"- {e.name} ({e.entity_type.value})")
                if e.aliases:
                    entity_lines.append(f"  Aliases: {', '.join(e.aliases[:3])}")

            return "\nKNOWN ENTITIES (use exact names for WHERE clauses):\n" + "\n".join(entity_lines) + "\n"
        except Exception:
            return ""

    async def generate_sql(
        self,
        question: str,
        schema_ddl: Optional[str] = None,
        examples: Optional[List[TextToSQLExample]] = None,
        sample_data: str = "",
        complexity: str = "medium",
        temperature: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question
            schema_ddl: Pre-built schema DDL (if None, fetches full schema)
            examples: Override examples list
            sample_data: Pre-formatted sample data string
            complexity: Complexity level for prompt selection
            temperature: Override temperature

        Returns:
            Tuple of (generated_sql, confidence_score)
        """
        if schema_ddl is None:
            schema_ddl = await self._get_schema_ddl()

        formatted_examples = self._format_examples(examples)

        # Get entity context from knowledge graph
        entity_context = await self._get_entity_context(question)
        if entity_context:
            formatted_examples = formatted_examples + entity_context

        # Select prompt template based on complexity
        if complexity == "easy":
            prompt_template = SQL_GENERATION_PROMPT_EASY
        elif complexity == "hard":
            prompt_template = SQL_GENERATION_PROMPT_HARD
        else:
            prompt_template = SQL_GENERATION_PROMPT

        prompt = prompt_template.format(
            schema=schema_ddl,
            examples=formatted_examples,
            sample_data=sample_data,
            question=question,
        )

        # Get LLM
        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=temperature if temperature is not None else self.temperature,
        )

        # Generate SQL
        response = await llm.ainvoke(prompt)
        generated_sql = response.content.strip()

        # For hard queries, extract SQL after "SQL:" marker
        if complexity == "hard" and "SQL:" in generated_sql:
            sql_part = generated_sql.split("SQL:")[-1].strip()
            if sql_part:
                generated_sql = sql_part

        # Clean up the response
        generated_sql = self._clean_sql_response(generated_sql)

        # Estimate confidence
        confidence = self._estimate_confidence(generated_sql, question)

        return generated_sql, confidence

    async def generate_sql_candidates(
        self,
        question: str,
        num_candidates: int = 3,
        schema_ddl: Optional[str] = None,
        examples: Optional[List[TextToSQLExample]] = None,
        sample_data: str = "",
        complexity: str = "medium",
    ) -> List[Tuple[str, float]]:
        """
        Generate multiple SQL candidates at different temperatures for self-consistency.

        Returns:
            List of (sql, confidence) tuples
        """
        temperatures = [0.0, 0.3, 0.5][:num_candidates]

        tasks = [
            self.generate_sql(
                question=question,
                schema_ddl=schema_ddl,
                examples=examples,
                sample_data=sample_data,
                complexity=complexity,
                temperature=temp,
            )
            for temp in temperatures
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Candidate generation failed", error=str(r))
                continue
            candidates.append(r)

        return candidates

    async def _pick_best_candidate(
        self, candidates: List[Tuple[str, float]]
    ) -> Tuple[str, float]:
        """
        Pick the best SQL candidate: first one that passes validation and executes.

        Falls back to the first candidate if none execute successfully.
        """
        for sql, confidence in candidates:
            # Syntax check
            syntax_ok, _ = self.validator.validate_syntax(sql)
            if not syntax_ok:
                continue

            # Safety check
            validation = self.validator.validate(sql)
            if not validation.is_valid or not validation.is_read_only:
                continue

            # Try execution
            try:
                result = await self.connector.execute_validated_query(sql)
                if result.success:
                    return sql, confidence
            except Exception:
                continue

        # Fallback: return first candidate
        if candidates:
            return candidates[0]
        return "", 0.0

    def _clean_sql_response(self, raw: str) -> str:
        """Extract SQL from LLM response, stripping explanation text and markdown."""
        # 1. Try to extract from ```sql ... ``` code block (most common)
        code_block = re.search(r'```sql\s*\n?(.*?)```', raw, re.DOTALL | re.IGNORECASE)
        if code_block:
            sql = code_block.group(1).strip()
            return sql.rstrip(';')

        # 2. Try generic ``` ... ``` code block
        code_block = re.search(r'```\s*\n?(.*?)```', raw, re.DOTALL)
        if code_block:
            candidate = code_block.group(1).strip()
            # Only use if it looks like SQL
            if candidate.upper().startswith(("SELECT", "WITH")):
                return candidate.rstrip(';')

        # 3. Try to find a standalone SELECT/WITH statement in the text
        # Match from SELECT/WITH to end, stopping at explanation text
        sql_match = re.search(
            r'((?:SELECT|WITH)\b.+?)(?:\n\s*\n\*\*|\n\s*\nExplanation|\n\s*\nNote:|\n\s*\n---|\Z)',
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if sql_match:
            sql = sql_match.group(1).strip()
            return sql.rstrip(';')

        # 4. Fallback: strip markdown fences from edges (original behavior)
        sql = re.sub(r'^```sql\s*', '', raw, flags=re.IGNORECASE)
        sql = re.sub(r'^```\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)
        sql = sql.strip().rstrip(';')
        return sql

    def _estimate_confidence(self, sql: str, question: str) -> float:
        """Estimate confidence score for generated SQL."""
        confidence = 0.5

        sql_upper = sql.upper()
        if sql_upper.startswith("SELECT") or sql_upper.startswith("WITH"):
            confidence += 0.1

        question_lower = question.lower()
        if any(word in question_lower for word in ["join", "related", "associated", "with", "and"]):
            if "JOIN" in sql_upper:
                confidence += 0.1

        if any(word in question_lower for word in ["how many", "count", "total", "sum", "average"]):
            if any(func in sql_upper for func in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
                confidence += 0.1

        if any(word in question_lower for word in ["top", "best", "worst", "highest", "lowest", "most", "least"]):
            if "ORDER BY" in sql_upper:
                confidence += 0.1

        if any(func in sql_upper for func in ["COUNT(", "SUM(", "AVG("]):
            if "GROUP BY" in sql_upper:
                confidence += 0.05

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
            temperature=0.3,
        )

        response = await llm.ainvoke(prompt)
        return response.content.strip()

    async def generate_answer(
        self,
        question: str,
        sql: str,
        query_result: QueryResult,
        max_rows_for_llm: int = 30,
    ) -> str:
        """Generate a human-readable answer from SQL query results."""
        rows_to_show = query_result.rows[:max_rows_for_llm]

        if query_result.columns and rows_to_show:
            header = " | ".join(str(c) for c in query_result.columns)
            separator = "-" * min(len(header), 120)
            row_lines = []
            for row in rows_to_show:
                row_lines.append(" | ".join(str(v) for v in row))
            results_summary = f"{header}\n{separator}\n" + "\n".join(row_lines)
            if len(query_result.rows) > max_rows_for_llm:
                results_summary += f"\n... ({len(query_result.rows) - max_rows_for_llm} more rows truncated)"
        else:
            results_summary = "(No results returned)"

        prompt = SQL_ANSWER_GENERATION_PROMPT.format(
            question=question,
            sql=sql,
            row_count=query_result.row_count,
            results_summary=results_summary,
        )

        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=0.3,
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
        Attempt to recover from a SQL error by re-prompting with classified error hints.
        """
        schema_ddl = await self._get_schema_ddl()
        error_info = _classify_error(error)

        prompt = SQL_ERROR_RECOVERY_PROMPT.format(
            schema=schema_ddl,
            question=question,
            sql=sql,
            error=error,
            error_type=error_info["error_type"],
            error_hint=error_info["error_hint"],
        )

        llm = get_chat_model(
            provider=self.llm_provider,
            model=self.llm_model,
            temperature=0.2,  # Slightly above 0 to avoid repeating same mistake
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
        # Per-query feature overrides
        schema_linking: Optional[bool] = None,
        sample_rows: Optional[bool] = None,
        self_consistency: Optional[bool] = None,
        num_candidates: Optional[int] = None,
        dynamic_examples: Optional[bool] = None,
        complexity_routing: Optional[bool] = None,
        num_examples: int = 3,
    ) -> TextToSQLResult:
        """
        Convert natural language question to SQL and optionally execute it.

        Args:
            question: Natural language question
            execute: Whether to execute the generated SQL
            explain: Whether to generate an explanation
            schema_linking: Override schema linking flag
            sample_rows: Override sample rows flag
            self_consistency: Override self-consistency flag
            num_candidates: Number of candidates for self-consistency
            dynamic_examples: Override dynamic examples flag
            complexity_routing: Override complexity routing flag
            num_examples: Max number of few-shot examples

        Returns:
            TextToSQLResult with SQL, explanation, and query results
        """
        result = TextToSQLResult(
            success=False,
            natural_language_query=question,
        )

        # Check if text-to-SQL is enabled
        try:
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()
            text_to_sql_enabled = await settings_svc.get_setting("database.text_to_sql_enabled")
            if text_to_sql_enabled is False:
                result.error = "Text-to-SQL is disabled in settings"
                return result
        except Exception:
            pass

        try:
            features_used = []

            # Resolve feature flags
            use_schema_linking = await self._resolve_feature_flag(
                schema_linking if schema_linking is not None else self._schema_linking,
                "database.schema_linking_enabled", True
            )
            use_sample_rows = await self._resolve_feature_flag(
                sample_rows if sample_rows is not None else self._sample_rows,
                "database.sample_rows_enabled", True
            )
            use_self_consistency = await self._resolve_feature_flag(
                self_consistency if self_consistency is not None else self._self_consistency,
                "database.self_consistency_enabled", False
            )
            use_dynamic_examples = await self._resolve_feature_flag(
                dynamic_examples if dynamic_examples is not None else self._dynamic_examples,
                "database.dynamic_examples_enabled", True
            )
            use_complexity_routing = await self._resolve_feature_flag(
                complexity_routing if complexity_routing is not None else self._complexity_routing,
                "database.complexity_routing_enabled", True
            )

            effective_num_candidates = num_candidates or self._num_candidates

            # --- Complexity routing ---
            complexity = "medium"
            schema = await self._get_schema()
            if use_complexity_routing:
                from backend.services.text_to_sql.complexity_router import ComplexityRouter
                router = ComplexityRouter()
                complexity, complexity_meta = router.classify(
                    question, num_tables=len(schema.tables)
                )
                result.complexity = complexity
                features_used.append("complexity_routing")

            # --- Schema linking ---
            schema_tables_used = [t.name for t in schema.tables]
            use_compact_ddl = complexity == "easy"

            if use_schema_linking and len(schema.tables) > 3:
                schema_ddl, schema_tables_used = await self._get_linked_schema_ddl(
                    question, compact=use_compact_ddl
                )
                result.schema_tables_used = schema_tables_used
                features_used.append("schema_linking")
            elif use_compact_ddl:
                schema_ddl = await self._get_schema_ddl(compact=True)
            else:
                schema_ddl = await self._get_schema_ddl(compact=False)

            # --- Sample rows ---
            sample_data_str = ""
            if use_sample_rows:
                sample_data_str = await self._get_sample_rows(schema_tables_used)
                if sample_data_str:
                    features_used.append("sample_rows")

            # --- Examples ---
            examples_to_use = self._examples[:num_examples] if self._examples else []

            # --- Generate SQL ---
            if use_self_consistency:
                features_used.append("self_consistency")
                candidates = await self.generate_sql_candidates(
                    question=question,
                    num_candidates=effective_num_candidates,
                    schema_ddl=schema_ddl,
                    examples=examples_to_use if examples_to_use else None,
                    sample_data=sample_data_str,
                    complexity=complexity,
                )
                result.candidates_generated = len(candidates)

                if candidates:
                    sql, confidence = await self._pick_best_candidate(candidates)
                else:
                    sql, confidence = await self.generate_sql(
                        question=question,
                        schema_ddl=schema_ddl,
                        examples=examples_to_use if examples_to_use else None,
                        sample_data=sample_data_str,
                        complexity=complexity,
                    )
            else:
                sql, confidence = await self.generate_sql(
                    question=question,
                    schema_ddl=schema_ddl,
                    examples=examples_to_use if examples_to_use else None,
                    sample_data=sample_data_str,
                    complexity=complexity,
                )

            result.generated_sql = sql
            result.confidence = confidence
            result.attempts = 1
            result.features_used = features_used

            # --- Syntax pre-validation ---
            syntax_ok, syntax_error = self.validator.validate_syntax(sql)
            if not syntax_ok:
                logger.info("Syntax pre-validation failed, attempting recovery", error=syntax_error)
                # Go straight to recovery without hitting DB
                sql, confidence = await self.recover_from_error(
                    question, sql, f"Syntax error: {syntax_error}"
                )
                result.generated_sql = sql
                result.confidence = confidence
                result.attempts += 1

            # --- Safety validation ---
            validation = self.validator.validate(sql)
            if not validation.is_valid:
                result.error = f"Query validation failed: {'; '.join(validation.errors)}"
                return result

            if not validation.is_read_only:
                result.error = "Generated query is not read-only"
                return result

            # --- Execute ---
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

            # Generate explanation
            if explain and result.generated_sql:
                result.explanation = await self.generate_explanation(result.generated_sql)

            # Generate human-readable answer
            if execute and result.query_result and result.query_result.success:
                try:
                    result.answer = await self.generate_answer(
                        question=question,
                        sql=result.generated_sql,
                        query_result=result.query_result,
                    )
                except Exception as e:
                    logger.warning("Answer generation failed", error=str(e))

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
        """
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

        sql, confidence = await self.generate_sql(question)

        validation = self.validator.validate(sql)
        if not validation.is_valid:
            return {
                "status": "error",
                "errors": validation.errors,
            }

        cost = await self.estimate_query_cost(sql)

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
        """Estimate the cost/complexity of a SQL query before execution."""
        cost = {
            "estimated_rows": None,
            "complexity": "low",
            "warnings": [],
            "estimated_time_ms": None,
        }

        sql_upper = sql.upper()

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

        try:
            if hasattr(self.connector, 'explain_query'):
                explain_result = await self.connector.explain_query(sql)
                if explain_result:
                    cost["explain"] = explain_result
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
        """Suggest the best visualization for query results."""
        if not query_result.success or not query_result.rows:
            return {"type": None, "reason": "No data to visualize"}

        columns = query_result.columns or []
        rows = query_result.rows
        sample = rows[:10] if len(rows) > 10 else rows

        numeric_cols = []
        categorical_cols = []
        date_cols = []

        for col in columns:
            sample_values = [row.get(col) for row in sample if row.get(col) is not None]
            if not sample_values:
                continue

            try:
                [float(v) for v in sample_values[:5]]
                numeric_cols.append(col)
            except (ValueError, TypeError):
                if any(isinstance(v, str) and re.match(r'\d{4}-\d{2}', str(v)) for v in sample_values[:3]):
                    date_cols.append(col)
                else:
                    categorical_cols.append(col)

        suggestion = {
            "type": None,
            "x_axis": None,
            "y_axis": None,
            "group_by": None,
            "insights": [],
        }

        row_count = len(rows)

        if date_cols and numeric_cols:
            suggestion["type"] = "line"
            suggestion["x_axis"] = date_cols[0]
            suggestion["y_axis"] = numeric_cols[0]
            suggestion["insights"].append("Time series data - line chart recommended")
        elif categorical_cols and numeric_cols:
            if len(rows) <= 20:
                suggestion["type"] = "bar"
                suggestion["x_axis"] = categorical_cols[0]
                suggestion["y_axis"] = numeric_cols[0]
                suggestion["insights"].append("Categorical comparison - bar chart recommended")
            else:
                suggestion["type"] = "scatter"
                suggestion["insights"].append("Many categories - scatter plot recommended")
        elif len(numeric_cols) >= 2:
            suggestion["type"] = "scatter"
            suggestion["x_axis"] = numeric_cols[0]
            suggestion["y_axis"] = numeric_cols[1]
            suggestion["insights"].append("Two numeric columns - scatter plot recommended")
        elif len(columns) == 2 and numeric_cols and row_count <= 10:
            suggestion["type"] = "pie"
            suggestion["group_by"] = categorical_cols[0] if categorical_cols else columns[0]
            suggestion["insights"].append("Distribution data - pie chart recommended")
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
        """Generate visualization code for the query results."""
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

        code = response.content.strip()
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'\s*```$', '', code)

        return code

    async def query_with_visualization(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """Full pipeline: question -> SQL -> execute -> visualize."""
        result = await self.query(question, execute=True, explain=True)

        if not result.success or not result.query_result:
            return {
                "success": False,
                "error": result.error,
            }

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
