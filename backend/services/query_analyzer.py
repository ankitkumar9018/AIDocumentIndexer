"""
AIDocumentIndexer - Intelligent Query Analyzer
===============================================

Analyzes queries to automatically determine:
1. Query complexity and appropriate intelligence level
2. Which tools are needed (calculator, fact-checker, etc.)
3. Whether extended thinking is required
4. Whether multi-model verification would help

The LLM decides what features to use based on query analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum
import re
import asyncio

import structlog

from backend.services.llm import LLMFactory
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class QueryComplexity(str, Enum):
    """Detected query complexity level."""
    SIMPLE = "simple"           # Direct lookup, single fact
    MODERATE = "moderate"       # Some reasoning required
    COMPLEX = "complex"         # Multi-step reasoning
    HIGHLY_COMPLEX = "highly_complex"  # Deep analysis needed


class QueryType(str, Enum):
    """Types of queries."""
    FACTUAL = "factual"         # Looking up facts
    ANALYTICAL = "analytical"   # Requires analysis
    MATHEMATICAL = "mathematical"  # Involves calculations
    COMPARATIVE = "comparative"    # Comparing things
    TEMPORAL = "temporal"          # Time/date related
    PROCEDURAL = "procedural"      # How-to questions
    CREATIVE = "creative"          # Open-ended
    VERIFICATION = "verification"  # Fact-checking claims


@dataclass
class QueryAnalysis:
    """Result of analyzing a query."""
    original_query: str
    complexity: QueryComplexity
    query_types: List[QueryType]
    recommended_intelligence_level: str
    recommended_tools: List[str]
    enable_extended_thinking: bool
    enable_ensemble_voting: bool
    enable_parallel_knowledge: bool
    confidence: float
    reasoning: str
    detected_patterns: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "complexity": self.complexity.value,
            "query_types": [t.value for t in self.query_types],
            "recommended_intelligence_level": self.recommended_intelligence_level,
            "recommended_tools": self.recommended_tools,
            "enable_extended_thinking": self.enable_extended_thinking,
            "enable_ensemble_voting": self.enable_ensemble_voting,
            "enable_parallel_knowledge": self.enable_parallel_knowledge,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "detected_patterns": self.detected_patterns,
        }


class QueryAnalyzerService:
    """
    Analyzes queries to intelligently determine which features to use.

    Uses both:
    1. Pattern matching for quick detection
    2. LLM analysis for complex cases
    """

    # Patterns for quick detection
    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\%\^]\s*\d+',  # Basic math
        r'calculate|compute|sum|total|average|mean|percentage|percent',
        r'how much|how many|what is \d',
        r'square root|sqrt|log|sin|cos|tan',
        r'\$\d+|\d+\s*dollars|€\d+|\d+\s*euros',
    ]

    DATE_PATTERNS = [
        r'how many days|weeks|months|years',
        r'when was|when is|when will',
        r'date of|birthday|anniversary',
        r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}',
        r'today|tomorrow|yesterday|last week|next month',
    ]

    FACT_CHECK_PATTERNS = [
        r'is it true that|is it correct that',
        r'verify|confirm|check if|fact check',
        r'did .+ really|was it actually',
        r'prove|evidence|source',
    ]

    COMPLEX_PATTERNS = [
        r'explain|analyze|compare|contrast|evaluate',
        r'what are the implications|consequences',
        r'pros and cons|advantages and disadvantages',
        r'why does|how does .+ work|mechanism',
        r'relationship between|connection between',
    ]

    MULTI_STEP_PATTERNS = [
        r'step by step|one by one|in order',
        r'first .+ then|after that|finally',
        r'multiple|several|various|all the',
        r'list .+ and explain|describe each',
    ]

    ANALYSIS_PROMPT = """Analyze this query and determine the best approach to answer it.

Query: {query}

Consider:
1. Is this a simple lookup or does it require reasoning?
2. Does it involve calculations or mathematical concepts?
3. Does it require comparing multiple things?
4. Does it involve dates or time-based reasoning?
5. Is it asking to verify facts or claims?
6. Would multiple AI models help verify the answer?
7. Does it need deep, extended thinking?

Respond in JSON:
{{
    "complexity": "simple|moderate|complex|highly_complex",
    "query_types": ["factual", "analytical", "mathematical", "comparative", "temporal", "procedural", "creative", "verification"],
    "needs_calculator": true/false,
    "needs_date_calculator": true/false,
    "needs_fact_checker": true/false,
    "needs_code_executor": true/false,
    "needs_extended_thinking": true/false,
    "needs_multi_model_verification": true/false,
    "needs_parallel_knowledge": true/false,
    "reasoning": "brief explanation"
}}"""

    def __init__(self):
        """Initialize the query analyzer."""
        self.provider = settings.DEFAULT_LLM_PROVIDER
        self.model = settings.DEFAULT_CHAT_MODEL

    def quick_analyze(self, query: str) -> QueryAnalysis:
        """
        Quick pattern-based analysis without LLM call.
        Used as first pass or when LLM is not available.
        """
        query_lower = query.lower()
        detected_tools: Set[str] = set()
        query_types: Set[QueryType] = set()
        patterns_found: Dict[str, List[str]] = {}

        # Check for math patterns
        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, query_lower):
                detected_tools.add("calculator")
                query_types.add(QueryType.MATHEMATICAL)
                patterns_found.setdefault("math", []).append(pattern)
                break

        # Check for date patterns
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, query_lower):
                detected_tools.add("date_calculator")
                query_types.add(QueryType.TEMPORAL)
                patterns_found.setdefault("date", []).append(pattern)
                break

        # Check for fact-check patterns
        for pattern in self.FACT_CHECK_PATTERNS:
            if re.search(pattern, query_lower):
                detected_tools.add("fact_checker")
                query_types.add(QueryType.VERIFICATION)
                patterns_found.setdefault("fact_check", []).append(pattern)
                break

        # Check for complexity indicators
        is_complex = False
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                is_complex = True
                query_types.add(QueryType.ANALYTICAL)
                patterns_found.setdefault("complex", []).append(pattern)
                break

        # Check for multi-step indicators
        is_multi_step = False
        for pattern in self.MULTI_STEP_PATTERNS:
            if re.search(pattern, query_lower):
                is_multi_step = True
                patterns_found.setdefault("multi_step", []).append(pattern)
                break

        # Determine complexity level
        complexity = QueryComplexity.SIMPLE
        if is_multi_step and is_complex:
            complexity = QueryComplexity.HIGHLY_COMPLEX
        elif is_complex or is_multi_step:
            complexity = QueryComplexity.COMPLEX
        elif len(detected_tools) > 0 or len(query_types) > 1:
            complexity = QueryComplexity.MODERATE

        # Default query type if none detected
        if not query_types:
            if "?" in query:
                query_types.add(QueryType.FACTUAL)
            else:
                query_types.add(QueryType.FACTUAL)

        # Determine intelligence level
        if complexity == QueryComplexity.HIGHLY_COMPLEX:
            intelligence_level = "maximum"
            enable_extended_thinking = True
            enable_ensemble = True
        elif complexity == QueryComplexity.COMPLEX:
            intelligence_level = "enhanced"
            enable_extended_thinking = False
            enable_ensemble = False
        elif complexity == QueryComplexity.MODERATE:
            intelligence_level = "standard"
            enable_extended_thinking = False
            enable_ensemble = False
        else:
            intelligence_level = "basic"
            enable_extended_thinking = False
            enable_ensemble = False

        # Enable parallel knowledge for complex analytical queries
        enable_parallel = (
            complexity in [QueryComplexity.COMPLEX, QueryComplexity.HIGHLY_COMPLEX]
            and QueryType.ANALYTICAL in query_types
        )

        return QueryAnalysis(
            original_query=query,
            complexity=complexity,
            query_types=list(query_types),
            recommended_intelligence_level=intelligence_level,
            recommended_tools=list(detected_tools),
            enable_extended_thinking=enable_extended_thinking,
            enable_ensemble_voting=enable_ensemble,
            enable_parallel_knowledge=enable_parallel,
            confidence=0.7,  # Pattern matching confidence
            reasoning=f"Pattern-based analysis: {len(patterns_found)} pattern types matched",
            detected_patterns=patterns_found,
        )

    async def analyze(
        self,
        query: str,
        use_llm: bool = True,
        context: str = "",
    ) -> QueryAnalysis:
        """
        Analyze a query to determine optimal settings.

        Args:
            query: The user's query
            use_llm: Whether to use LLM for analysis
            context: Optional context from previous messages

        Returns:
            QueryAnalysis with recommendations
        """
        # Quick pattern-based analysis first
        quick_result = self.quick_analyze(query)

        # If patterns already indicate high complexity, or LLM not requested, return quick result
        if not use_llm or quick_result.complexity == QueryComplexity.HIGHLY_COMPLEX:
            logger.info(
                "Using pattern-based analysis",
                complexity=quick_result.complexity.value,
                tools=quick_result.recommended_tools,
            )
            return quick_result

        # Use LLM for more nuanced analysis
        try:
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.1,
                max_tokens=512,
            )

            prompt = self.ANALYSIS_PROMPT.format(query=query)
            response = await llm.ainvoke(prompt)
            content = response.content

            # Parse JSON response
            import json
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())

                # Map complexity
                complexity_map = {
                    "simple": QueryComplexity.SIMPLE,
                    "moderate": QueryComplexity.MODERATE,
                    "complex": QueryComplexity.COMPLEX,
                    "highly_complex": QueryComplexity.HIGHLY_COMPLEX,
                }
                complexity = complexity_map.get(
                    analysis.get("complexity", "simple"),
                    QueryComplexity.MODERATE
                )

                # Map query types
                type_map = {
                    "factual": QueryType.FACTUAL,
                    "analytical": QueryType.ANALYTICAL,
                    "mathematical": QueryType.MATHEMATICAL,
                    "comparative": QueryType.COMPARATIVE,
                    "temporal": QueryType.TEMPORAL,
                    "procedural": QueryType.PROCEDURAL,
                    "creative": QueryType.CREATIVE,
                    "verification": QueryType.VERIFICATION,
                }
                query_types = [
                    type_map[t] for t in analysis.get("query_types", ["factual"])
                    if t in type_map
                ]

                # Collect recommended tools
                tools = []
                if analysis.get("needs_calculator"):
                    tools.append("calculator")
                if analysis.get("needs_date_calculator"):
                    tools.append("date_calculator")
                if analysis.get("needs_fact_checker"):
                    tools.append("fact_checker")
                if analysis.get("needs_code_executor"):
                    tools.append("code_executor")

                # Determine intelligence level
                if complexity == QueryComplexity.HIGHLY_COMPLEX:
                    intelligence_level = "maximum"
                elif complexity == QueryComplexity.COMPLEX:
                    intelligence_level = "enhanced"
                elif complexity == QueryComplexity.MODERATE:
                    intelligence_level = "standard"
                else:
                    intelligence_level = "basic"

                result = QueryAnalysis(
                    original_query=query,
                    complexity=complexity,
                    query_types=query_types or [QueryType.FACTUAL],
                    recommended_intelligence_level=intelligence_level,
                    recommended_tools=tools,
                    enable_extended_thinking=analysis.get("needs_extended_thinking", False),
                    enable_ensemble_voting=analysis.get("needs_multi_model_verification", False),
                    enable_parallel_knowledge=analysis.get("needs_parallel_knowledge", False),
                    confidence=0.9,
                    reasoning=analysis.get("reasoning", "LLM analysis"),
                    detected_patterns=quick_result.detected_patterns,
                )

                logger.info(
                    "LLM query analysis complete",
                    complexity=result.complexity.value,
                    intelligence_level=result.recommended_intelligence_level,
                    tools=result.recommended_tools,
                )

                return result

        except Exception as e:
            logger.warning("LLM analysis failed, using pattern matching", error=str(e))

        # Fall back to quick analysis
        return quick_result

    async def should_use_tool(
        self,
        tool_name: str,
        query: str,
        context: str = "",
    ) -> bool:
        """
        Determine if a specific tool should be used for a query.

        Args:
            tool_name: Name of the tool (calculator, fact_checker, etc.)
            query: The user's query
            context: Optional context

        Returns:
            True if the tool should be used
        """
        analysis = await self.analyze(query, use_llm=False)
        return tool_name in analysis.recommended_tools


# Singleton instance
_query_analyzer: Optional[QueryAnalyzerService] = None


def get_query_analyzer() -> QueryAnalyzerService:
    """Get or create the query analyzer singleton."""
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzerService()
    return _query_analyzer
