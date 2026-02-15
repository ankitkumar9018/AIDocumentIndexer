"""
AIDocumentIndexer - Query Complexity Router
=============================================

Heuristic classifier for SQL query complexity.
Routes questions to appropriate prompt templates and strategies.
"""

import re
from typing import Dict, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Complexity signal keywords and their weights
COMPLEXITY_SIGNALS = {
    # Join indicators (+1)
    "join": 1, "combined": 1, "along with": 1, "together with": 1,
    "related to": 1, "associated": 1, "linked": 1, "connected to": 1,

    # Aggregation (+1)
    "count": 1, "total": 1, "sum": 1, "average": 1, "avg": 1,
    "maximum": 1, "minimum": 1, "aggregate": 1,

    # Time analysis (+2)
    "trend": 2, "over time": 2, "month over month": 2, "year over year": 2,
    "growth": 2, "change over": 2, "time series": 2, "by month": 2,
    "by year": 2, "by week": 2, "by day": 2, "quarterly": 2,
    "monthly": 2, "yearly": 2, "daily": 2, "seasonal": 2,

    # Window functions (+3)
    "running total": 3, "cumulative": 3, "moving average": 3,
    "rank": 3, "ranking": 3, "percentile": 3, "ntile": 3,
    "row number": 3, "lag": 3, "lead": 3, "first value": 3,
    "partition": 3,

    # Subquery indicators (+2)
    "compared to": 2, "relative to": 2, "percentage of total": 2,
    "as a fraction": 2, "versus": 2, "vs": 2, "more than average": 2,
    "above average": 2, "below average": 2,

    # Multi-entity (+2)
    "and also": 2, "as well as": 2, "additionally": 2,
    "furthermore": 2, "along with their": 2,

    # Pivoting / complex transforms (+3)
    "pivot": 3, "crosstab": 3, "transpose": 3, "matrix": 3,
    "correlation": 3,
}


class ComplexityRouter:
    """
    Classifies natural language questions by SQL complexity.

    Uses heuristic keyword scoring (no LLM call) for fast routing:
    - easy: score <= 1 (simple lookups, single-table aggregates)
    - medium: score <= 4 (joins, GROUP BY, basic time analysis)
    - hard: score > 4 (window functions, subqueries, multi-step)
    """

    def classify(self, question: str, num_tables: int = 0) -> Tuple[str, Dict]:
        """
        Classify question complexity.

        Args:
            question: Natural language question
            num_tables: Number of tables in the schema (more tables = harder)

        Returns:
            Tuple of (level, metadata) where level is "easy", "medium", or "hard"
        """
        question_lower = question.lower()
        score = 0
        signals_found = []

        # Check each signal
        for signal, weight in COMPLEXITY_SIGNALS.items():
            if signal in question_lower:
                score += weight
                signals_found.append(signal)

        # Bonus for question length (longer questions tend to be more complex)
        word_count = len(question.split())
        if word_count > 20:
            score += 1
        if word_count > 35:
            score += 1

        # Bonus if schema is large (more potential for confusion)
        if num_tables > 15:
            score += 1

        # Classify
        if score <= 1:
            level = "easy"
        elif score <= 4:
            level = "medium"
        else:
            level = "hard"

        metadata = {
            "score": score,
            "signals": signals_found,
            "word_count": word_count,
            "num_tables": num_tables,
        }

        logger.debug(
            "Query complexity classified",
            level=level,
            score=score,
            signals=signals_found,
        )

        return level, metadata
