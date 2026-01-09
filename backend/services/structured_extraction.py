"""
AIDocumentIndexer - Structured Data Extraction Service
=======================================================

Implements StructRAG pattern for extracting and aggregating numerical data
from documents. Enables accurate calculation queries like "What is the total
spending by Company A in last 4 months?"

Key Features:
- Pydantic-based structured output extraction
- Multi-document value aggregation
- Calculation breakdown and verification
- Confidence scoring based on data completeness

Based on:
- AI21 StructRAG patterns
- LangChain structured output with Pydantic
- Text-to-SQL aggregation approaches
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# =============================================================================
# Pydantic Models for Structured Extraction
# =============================================================================


class ExtractedValue(BaseModel):
    """A single extracted numerical value with context."""

    value: float = Field(description="The numerical value extracted")
    unit: str = Field(
        default="",
        description="Unit of measurement (USD, EUR, %, units, etc.)",
    )
    category: str = Field(
        default="",
        description="Category/type of this value (spending, revenue, cost, etc.)",
    )
    entity: str = Field(
        default="",
        description="Entity this value belongs to (company name, person, etc.)",
    )
    time_period: Optional[str] = Field(
        default=None,
        description="Time period if mentioned (Q1 2024, January, etc.)",
    )
    source_document: str = Field(
        default="",
        description="Document this was extracted from",
    )
    source_chunk_id: Optional[str] = Field(
        default=None,
        description="Chunk ID for precise source tracking",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score 0-1 for this extraction",
    )
    context_snippet: str = Field(
        default="",
        description="Surrounding text for verification",
    )

    class Config:
        extra = "allow"


class AggregationType(str, Enum):
    """Types of aggregation operations."""

    SUM = "sum"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"
    COUNT = "count"
    MEDIAN = "median"


class AggregationResult(BaseModel):
    """Result of aggregating numerical values."""

    total: float = Field(description="Aggregated result value")
    count: int = Field(description="Number of values aggregated")
    values: List[ExtractedValue] = Field(
        default_factory=list,
        description="Individual values that were aggregated",
    )
    breakdown_by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of totals by category",
    )
    breakdown_by_period: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of totals by time period",
    )
    breakdown_by_entity: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of totals by entity",
    )
    calculation_method: str = Field(
        default="sum",
        description="How aggregation was performed",
    )
    unit: str = Field(
        default="",
        description="Common unit of aggregated values",
    )
    confidence_score: float = Field(
        default=0.0,
        description="Overall confidence in the aggregation (0-1)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about data quality or completeness",
    )

    class Config:
        extra = "allow"


class ExtractionQuery(BaseModel):
    """Parsed query for structured extraction."""

    original_query: str = Field(description="Original user query")
    target_entity: Optional[str] = Field(
        default=None,
        description="Entity to filter by (e.g., 'Company A')",
    )
    target_category: Optional[str] = Field(
        default=None,
        description="Category to filter by (e.g., 'spending', 'revenue')",
    )
    time_filter: Optional[str] = Field(
        default=None,
        description="Time period filter (e.g., 'last 4 months')",
    )
    aggregation_type: AggregationType = Field(
        default=AggregationType.SUM,
        description="Type of aggregation requested",
    )


# =============================================================================
# Aggregation Keywords and Patterns
# =============================================================================


AGGREGATION_KEYWORDS = [
    "total",
    "sum",
    "average",
    "avg",
    "mean",
    "how much",
    "how many",
    "count",
    "number of",
    "spending",
    "revenue",
    "cost",
    "expense",
    "income",
    "profit",
    "loss",
    "budget",
    "combined",
    "aggregate",
    "overall",
    "cumulative",
    "maximum",
    "minimum",
    "highest",
    "lowest",
]

AGGREGATION_PATTERNS = [
    r"total\s+\w+",
    r"sum\s+of",
    r"how\s+much\s+\w+\s+(spent|earned|cost|paid)",
    r"average\s+\w+",
    r"combined\s+\w+",
    r"aggregate\s+\w+",
    r"\d+\s+(months?|years?|quarters?|weeks?|days?)",
    r"(last|past|previous)\s+\d+\s+(months?|years?|quarters?)",
    r"(q[1-4]|quarter)\s*\d*",
    r"(fy|fiscal\s+year)\s*\d+",
]


# =============================================================================
# Structured Extractor
# =============================================================================


class StructuredExtractor:
    """
    Extract and aggregate numerical data from documents.

    Uses LLM with structured output to reliably extract numerical values
    and perform calculations across multiple documents.
    """

    def __init__(self, llm=None, embedding_service=None):
        """
        Initialize the extractor.

        Args:
            llm: Language model for extraction (optional, will be lazy loaded)
            embedding_service: For semantic similarity (optional)
        """
        self._llm = llm
        self._embedding_service = embedding_service

    async def _get_llm(self):
        """Lazy load LLM if not provided."""
        if self._llm is None:
            from backend.services.rag import get_llm

            self._llm = await get_llm()
        return self._llm

    def is_aggregation_query(self, query: str) -> bool:
        """
        Check if a query requires numerical aggregation.

        Args:
            query: User's question

        Returns:
            True if the query appears to need aggregation
        """
        query_lower = query.lower()

        # Check keywords
        if any(kw in query_lower for kw in AGGREGATION_KEYWORDS):
            return True

        # Check patterns
        for pattern in AGGREGATION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True

        return False

    async def parse_query(self, query: str) -> ExtractionQuery:
        """
        Parse a user query to extract structured parameters.

        Args:
            query: User's natural language query

        Returns:
            ExtractionQuery with parsed parameters
        """
        llm = await self._get_llm()

        prompt = f"""Analyze this query and extract structured parameters for data aggregation.

Query: "{query}"

Extract:
1. Target entity (company, person, product name if mentioned)
2. Target category (spending, revenue, cost, etc.)
3. Time filter (time period mentioned like "last 4 months", "Q1 2024")
4. Aggregation type (sum, average, max, min, count)

Respond in this exact JSON format:
{{
    "target_entity": "<entity name or null>",
    "target_category": "<category or null>",
    "time_filter": "<time period or null>",
    "aggregation_type": "<sum|average|max|min|count>"
}}

JSON response:"""

        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            # Extract JSON from response
            import json

            # Find JSON in response
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return ExtractionQuery(
                    original_query=query,
                    target_entity=parsed.get("target_entity"),
                    target_category=parsed.get("target_category"),
                    time_filter=parsed.get("time_filter"),
                    aggregation_type=AggregationType(
                        parsed.get("aggregation_type", "sum")
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to parse query: {e}")

        # Fallback: basic parsing
        return self._basic_query_parse(query)

    def _basic_query_parse(self, query: str) -> ExtractionQuery:
        """Basic query parsing without LLM."""
        query_lower = query.lower()

        # Detect aggregation type
        agg_type = AggregationType.SUM
        if "average" in query_lower or "avg" in query_lower or "mean" in query_lower:
            agg_type = AggregationType.AVERAGE
        elif "maximum" in query_lower or "highest" in query_lower or "max" in query_lower:
            agg_type = AggregationType.MAX
        elif "minimum" in query_lower or "lowest" in query_lower or "min" in query_lower:
            agg_type = AggregationType.MIN
        elif "count" in query_lower or "how many" in query_lower or "number of" in query_lower:
            agg_type = AggregationType.COUNT

        # Detect time filter
        time_filter = None
        time_patterns = [
            r"(last|past|previous)\s+(\d+)\s+(months?|years?|quarters?|weeks?|days?)",
            r"(q[1-4])\s*(\d{4})?",
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?",
            r"(\d{4})",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                time_filter = match.group(0)
                break

        # Detect category
        category = None
        category_keywords = {
            "spending": ["spend", "spent", "spending", "expenditure"],
            "revenue": ["revenue", "sales", "income", "earned"],
            "cost": ["cost", "costs", "expense", "expenses"],
            "profit": ["profit", "profits", "margin"],
            "budget": ["budget", "budgeted", "allocated"],
        }
        for cat, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                category = cat
                break

        return ExtractionQuery(
            original_query=query,
            target_entity=None,  # Hard to extract without LLM
            target_category=category,
            time_filter=time_filter,
            aggregation_type=agg_type,
        )

    async def extract_numerical_values(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        entity_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[ExtractedValue]:
        """
        Extract all numerical values relevant to a query from document chunks.

        Args:
            query: User's query
            chunks: List of document chunks with content and metadata
            entity_filter: Optional entity to filter by
            category_filter: Optional category to filter by

        Returns:
            List of extracted numerical values
        """
        llm = await self._get_llm()
        all_values = []

        # Process chunks in batches to avoid token limits
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_text = self._format_chunks_for_extraction(batch)

            filter_instructions = ""
            if entity_filter:
                filter_instructions += f"\nFocus on values related to: {entity_filter}"
            if category_filter:
                filter_instructions += f"\nLook for {category_filter} values"

            prompt = f"""Extract ALL numerical values from these documents that could be relevant to: "{query}"
{filter_instructions}

Documents:
{batch_text}

For each numerical value found, extract:
1. The exact numerical value (as a number)
2. The unit (USD, EUR, %, units, etc.)
3. The category (spending, revenue, cost, quantity, etc.)
4. The entity it belongs to (company name, person, product)
5. Time period if mentioned
6. A short context snippet (max 100 chars)

Respond with a JSON array of objects. If no values found, return [].
Example format:
[
    {{"value": 50000, "unit": "USD", "category": "spending", "entity": "Company A", "time_period": "Q1 2024", "context": "Company A spent $50,000 in Q1 2024"}}
]

JSON array:"""

            try:
                response = await llm.ainvoke(prompt)
                content = response.content.strip()

                # Extract JSON array from response
                import json

                # Find JSON array in response
                array_match = re.search(r"\[[\s\S]*\]", content)
                if array_match:
                    try:
                        parsed = json.loads(array_match.group())
                        for item in parsed:
                            if isinstance(item, dict) and "value" in item:
                                # Get source info from chunk
                                source_doc = ""
                                chunk_id = None
                                if batch:
                                    source_doc = batch[0].get("metadata", {}).get(
                                        "document_name", ""
                                    )
                                    chunk_id = batch[0].get("chunk_id")

                                value = ExtractedValue(
                                    value=float(item.get("value", 0)),
                                    unit=str(item.get("unit", "")),
                                    category=str(item.get("category", "")),
                                    entity=str(item.get("entity", "")),
                                    time_period=item.get("time_period"),
                                    source_document=source_doc,
                                    source_chunk_id=chunk_id,
                                    confidence=0.8,  # Base confidence
                                    context_snippet=str(item.get("context", ""))[:150],
                                )
                                all_values.append(value)
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse JSON from extraction response")
            except Exception as e:
                logger.warning(f"Extraction failed for batch: {e}")

        # Apply filters
        if entity_filter:
            entity_lower = entity_filter.lower()
            all_values = [
                v
                for v in all_values
                if entity_lower in v.entity.lower()
                or entity_lower in v.context_snippet.lower()
            ]

        if category_filter:
            category_lower = category_filter.lower()
            all_values = [
                v
                for v in all_values
                if category_lower in v.category.lower()
                or category_lower in v.context_snippet.lower()
            ]

        logger.info(
            f"Extracted {len(all_values)} values from {len(chunks)} chunks",
            entity_filter=entity_filter,
            category_filter=category_filter,
        )

        return all_values

    def _format_chunks_for_extraction(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for the extraction prompt."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", chunk.get("text", ""))
            doc_name = chunk.get("metadata", {}).get("document_name", f"Document {i}")
            parts.append(f"[{doc_name}]\n{content}\n")
        return "\n---\n".join(parts)

    def aggregate_values(
        self,
        values: List[ExtractedValue],
        aggregation_type: AggregationType = AggregationType.SUM,
    ) -> AggregationResult:
        """
        Aggregate extracted values with detailed breakdown.

        Args:
            values: List of extracted values to aggregate
            aggregation_type: Type of aggregation to perform

        Returns:
            AggregationResult with totals and breakdowns
        """
        if not values:
            return AggregationResult(
                total=0,
                count=0,
                values=[],
                calculation_method=aggregation_type.value,
                confidence_score=0,
                warnings=["No numerical values found for aggregation"],
            )

        # Extract raw values
        raw_values = [v.value for v in values]

        # Calculate based on aggregation type
        if aggregation_type == AggregationType.SUM:
            total = sum(raw_values)
        elif aggregation_type == AggregationType.AVERAGE:
            total = sum(raw_values) / len(raw_values)
        elif aggregation_type == AggregationType.MAX:
            total = max(raw_values)
        elif aggregation_type == AggregationType.MIN:
            total = min(raw_values)
        elif aggregation_type == AggregationType.COUNT:
            total = len(raw_values)
        elif aggregation_type == AggregationType.MEDIAN:
            sorted_vals = sorted(raw_values)
            n = len(sorted_vals)
            if n % 2 == 0:
                total = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            else:
                total = sorted_vals[n // 2]
        else:
            total = sum(raw_values)

        # Build breakdowns
        by_category: Dict[str, List[float]] = {}
        by_period: Dict[str, List[float]] = {}
        by_entity: Dict[str, List[float]] = {}

        for v in values:
            if v.category:
                by_category.setdefault(v.category, []).append(v.value)
            if v.time_period:
                by_period.setdefault(v.time_period, []).append(v.value)
            if v.entity:
                by_entity.setdefault(v.entity, []).append(v.value)

        # Aggregate breakdowns
        breakdown_by_category = {k: sum(v) for k, v in by_category.items()}
        breakdown_by_period = {k: sum(v) for k, v in by_period.items()}
        breakdown_by_entity = {k: sum(v) for k, v in by_entity.items()}

        # Determine common unit
        units = [v.unit for v in values if v.unit]
        common_unit = max(set(units), key=units.count) if units else ""

        # Calculate confidence
        # More data points = higher confidence (up to a limit)
        data_confidence = min(1.0, len(values) / 5)

        # Consistent units = higher confidence
        unique_units = len(set(units)) if units else 1
        unit_confidence = 1.0 if unique_units == 1 else 0.7

        # Overall confidence
        confidence_score = (data_confidence * 0.6 + unit_confidence * 0.4)

        # Generate warnings
        warnings = []
        if len(values) < 3:
            warnings.append(
                f"Only {len(values)} data point(s) found - result may be incomplete"
            )
        if unique_units > 1:
            warnings.append(
                f"Mixed units detected ({', '.join(set(units))}) - verify unit consistency"
            )

        return AggregationResult(
            total=round(total, 2),
            count=len(values),
            values=values,
            breakdown_by_category=breakdown_by_category,
            breakdown_by_period=breakdown_by_period,
            breakdown_by_entity=breakdown_by_entity,
            calculation_method=aggregation_type.value,
            unit=common_unit,
            confidence_score=round(confidence_score, 2),
            warnings=warnings,
        )

    async def extract_and_aggregate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> AggregationResult:
        """
        Complete extraction and aggregation pipeline.

        Args:
            query: User's natural language query
            chunks: Document chunks to extract from

        Returns:
            AggregationResult with calculated values
        """
        # Parse the query
        parsed_query = await self.parse_query(query)

        logger.info(
            "Parsed aggregation query",
            target_entity=parsed_query.target_entity,
            target_category=parsed_query.target_category,
            time_filter=parsed_query.time_filter,
            aggregation_type=parsed_query.aggregation_type.value,
        )

        # Extract values
        values = await self.extract_numerical_values(
            query=query,
            chunks=chunks,
            entity_filter=parsed_query.target_entity,
            category_filter=parsed_query.target_category,
        )

        # Filter by time if specified
        if parsed_query.time_filter and values:
            values = self._filter_by_time(values, parsed_query.time_filter)

        # Aggregate
        result = self.aggregate_values(values, parsed_query.aggregation_type)

        return result

    def _filter_by_time(
        self,
        values: List[ExtractedValue],
        time_filter: str,
    ) -> List[ExtractedValue]:
        """Filter values by time period (basic implementation)."""
        # This is a simplified filter - could be enhanced with date parsing
        time_lower = time_filter.lower()
        filtered = []

        for v in values:
            # If value has a time period, check if it matches
            if v.time_period:
                if time_lower in v.time_period.lower():
                    filtered.append(v)
            # If no time period on value, include it (may be relevant)
            else:
                filtered.append(v)

        # If filter reduced too much, return original
        if len(filtered) < len(values) * 0.3 and len(values) > 3:
            logger.debug(
                f"Time filter too restrictive, keeping {len(filtered)} of {len(values)} values"
            )

        return filtered if filtered else values

    async def generate_aggregation_response(
        self,
        query: str,
        result: AggregationResult,
    ) -> str:
        """
        Generate a natural language response for an aggregation result.

        Args:
            query: Original user query
            result: Aggregation result to describe

        Returns:
            Natural language response string
        """
        llm = await self._get_llm()

        # Format the result data
        values_summary = "\n".join(
            f"  - {v.entity or 'Unknown'}: {v.value} {v.unit} ({v.category}, {v.time_period or 'no date'})"
            for v in result.values[:10]  # Limit to first 10
        )

        prompt = f"""Generate a clear, professional answer to this question based on the extracted data.

Question: "{query}"

Extracted Data:
- Total ({result.calculation_method}): {result.total} {result.unit}
- Number of data points: {result.count}

Individual values found:
{values_summary}

{"More values not shown..." if len(result.values) > 10 else ""}

Breakdown by category: {result.breakdown_by_category or "N/A"}
Breakdown by time period: {result.breakdown_by_period or "N/A"}
Breakdown by entity: {result.breakdown_by_entity or "N/A"}

Confidence: {result.confidence_score * 100:.0f}%
Warnings: {', '.join(result.warnings) if result.warnings else 'None'}

Provide:
1. A direct answer to the question
2. Brief breakdown if relevant
3. Note any limitations or caveats

Response:"""

        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback to basic response
            return (
                f"Based on {result.count} data point(s), the {result.calculation_method} "
                f"is {result.total} {result.unit}. "
                f"{'Note: ' + result.warnings[0] if result.warnings else ''}"
            )


# =============================================================================
# Singleton instance
# =============================================================================

_extractor_instance: Optional[StructuredExtractor] = None


def get_structured_extractor() -> StructuredExtractor:
    """Get or create the structured extractor singleton."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = StructuredExtractor()
    return _extractor_instance


async def is_aggregation_query(query: str) -> bool:
    """Convenience function to check if a query needs aggregation."""
    extractor = get_structured_extractor()
    return extractor.is_aggregation_query(query)
