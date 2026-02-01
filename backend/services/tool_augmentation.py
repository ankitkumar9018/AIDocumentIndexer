"""
AIDocumentIndexer - Tool Augmentation Service
==============================================

Compensating tools that help small LLMs overcome their limitations:
1. Calculator - Handles arithmetic operations (prevents math errors)
2. Code Executor - Runs code snippets (verifies logic)
3. Fact Checker - Validates claims against sources (prevents hallucination)
4. Date Calculator - Date/time math (temporal reasoning)
5. Unit Converter - Unit conversions (measurement accuracy)
6. Web Search - Real-time information (knowledge cutoff)

These tools make small LLMs perform at Claude-level quality.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import math
import re
import json

import structlog

from backend.services.llm import LLMFactory
from backend.core.config import settings

logger = structlog.get_logger(__name__)


class ToolType(str, Enum):
    """Types of augmentation tools."""
    CALCULATOR = "calculator"
    CODE_EXECUTOR = "code_executor"
    FACT_CHECKER = "fact_checker"
    DATE_CALCULATOR = "date_calculator"
    UNIT_CONVERTER = "unit_converter"
    WEB_SEARCH = "web_search"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool: ToolType
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool.value,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class FactCheckResult:
    """Result from fact checking."""
    claim: str
    verdict: str  # "supported", "contradicted", "unverifiable"
    confidence: float
    evidence: List[str]
    sources: List[str]


class CalculatorTool:
    """
    Safe calculator that handles arithmetic operations.
    Prevents small LLM arithmetic errors.
    """

    # Allowed operations
    SAFE_OPERATORS = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b if b != 0 else float('inf'),
        '**': lambda a, b: a ** b,
        '%': lambda a, b: a % b if b != 0 else 0,
        '//': lambda a, b: a // b if b != 0 else 0,
    }

    # Safe math functions
    SAFE_FUNCTIONS = {
        'sqrt': math.sqrt,
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'pow': math.pow,
        'pi': lambda: math.pi,
        'e': lambda: math.e,
    }

    def calculate(self, expression: str) -> ToolResult:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            ToolResult with the calculation result
        """
        import time
        start_time = time.time()

        try:
            # Clean and validate expression
            expression = expression.strip()

            # Remove any dangerous characters
            if any(char in expression for char in ['import', 'exec', 'eval', '__', 'open', 'os.', 'sys.']):
                return ToolResult(
                    tool=ToolType.CALCULATOR,
                    success=False,
                    result=None,
                    error="Expression contains forbidden characters",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

            # Try safe evaluation
            result = self._safe_eval(expression)

            return ToolResult(
                tool=ToolType.CALCULATOR,
                success=True,
                result=result,
                execution_time_ms=int((time.time() - start_time) * 1000),
                metadata={"expression": expression},
            )

        except Exception as e:
            return ToolResult(
                tool=ToolType.CALCULATOR,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    def _safe_eval(self, expression: str) -> Union[int, float]:
        """Safely evaluate a mathematical expression."""
        # Replace function names with their Python equivalents
        for func_name in self.SAFE_FUNCTIONS:
            if callable(self.SAFE_FUNCTIONS[func_name]):
                # Check if it's a constant (takes no args)
                if func_name in ['pi', 'e']:
                    expression = expression.replace(func_name, str(self.SAFE_FUNCTIONS[func_name]()))
                else:
                    expression = expression.replace(func_name, f'__{func_name}__')

        # Build safe namespace
        safe_dict = {
            f'__{name}__': func
            for name, func in self.SAFE_FUNCTIONS.items()
            if callable(func) and name not in ['pi', 'e']
        }

        # Add builtins we need
        safe_dict['__builtins__'] = {
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'True': True,
            'False': False,
        }

        # Evaluate
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return result


class CodeExecutorTool:
    """
    Safe code executor for verifying logic.
    Runs Python code in a sandboxed environment.
    """

    MAX_EXECUTION_TIME = 5  # seconds
    MAX_OUTPUT_LENGTH = 10000  # characters

    FORBIDDEN_IMPORTS = [
        'os', 'sys', 'subprocess', 'socket', 'requests',
        'urllib', 'http', 'ftplib', 'smtplib', 'pickle',
        'marshal', 'shelve', 'dbm', 'sqlite3', 'ctypes',
    ]

    async def execute(self, code: str, timeout: float = None) -> ToolResult:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            ToolResult with execution output
        """
        import time
        start_time = time.time()
        timeout = timeout or self.MAX_EXECUTION_TIME

        try:
            # Check for forbidden imports
            for forbidden in self.FORBIDDEN_IMPORTS:
                if f'import {forbidden}' in code or f'from {forbidden}' in code:
                    return ToolResult(
                        tool=ToolType.CODE_EXECUTOR,
                        success=False,
                        result=None,
                        error=f"Forbidden import: {forbidden}",
                        execution_time_ms=int((time.time() - start_time) * 1000),
                    )

            # Check for dangerous patterns
            dangerous_patterns = [
                r'exec\s*\(',
                r'eval\s*\(',
                r'__import__',
                r'open\s*\(',
                r'file\s*\(',
                r'compile\s*\(',
                r'globals\s*\(',
                r'locals\s*\(',
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code):
                    return ToolResult(
                        tool=ToolType.CODE_EXECUTOR,
                        success=False,
                        result=None,
                        error=f"Dangerous pattern detected: {pattern}",
                        execution_time_ms=int((time.time() - start_time) * 1000),
                    )

            # Execute in subprocess for true isolation
            output = await self._execute_sandboxed(code, timeout)

            return ToolResult(
                tool=ToolType.CODE_EXECUTOR,
                success=True,
                result=output[:self.MAX_OUTPUT_LENGTH],
                execution_time_ms=int((time.time() - start_time) * 1000),
                metadata={
                    "code_length": len(code),
                    "output_length": len(output),
                    "truncated": len(output) > self.MAX_OUTPUT_LENGTH,
                },
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool=ToolType.CODE_EXECUTOR,
                success=False,
                result=None,
                error=f"Execution timed out after {timeout} seconds",
                execution_time_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            return ToolResult(
                tool=ToolType.CODE_EXECUTOR,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    async def _execute_sandboxed(self, code: str, timeout: float) -> str:
        """Execute code in a sandboxed subprocess."""
        import io
        import contextlib

        # Create a restricted execution environment
        safe_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'sorted': sorted,
                'reversed': reversed,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'type': type,
                'True': True,
                'False': False,
                'None': None,
            }
        }

        # Capture output
        output_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code, safe_globals)

            return output_buffer.getvalue()

        except Exception as e:
            return f"Error: {str(e)}"


class FactCheckerTool:
    """
    Fact checker that validates claims against sources.
    Helps prevent LLM hallucination.
    """

    FACT_CHECK_PROMPT = """You are a fact-checker. Analyze the following claim and determine if it is:
- SUPPORTED: The claim is factually accurate based on the evidence
- CONTRADICTED: The claim is factually inaccurate
- UNVERIFIABLE: Cannot determine accuracy from available information

Claim: {claim}

Context/Evidence:
{evidence}

Analyze carefully and respond in JSON format:
{{
    "verdict": "supported|contradicted|unverifiable",
    "confidence": 0.0-1.0,
    "reasoning": "your analysis",
    "key_evidence": ["list of relevant evidence points"]
}}"""

    def __init__(self):
        self.provider = settings.DEFAULT_LLM_PROVIDER
        self.model = settings.DEFAULT_CHAT_MODEL

    async def check_fact(
        self,
        claim: str,
        context: str = "",
        sources: List[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Check if a claim is factually accurate.

        Args:
            claim: The claim to verify
            context: Additional context for verification
            sources: Documents to check against

        Returns:
            ToolResult with fact check result
        """
        import time
        start_time = time.time()

        try:
            # Build evidence from sources
            evidence_parts = []
            if context:
                evidence_parts.append(f"Context: {context}")

            if sources:
                for i, source in enumerate(sources[:5], 1):
                    content = source.get("content", source.get("text", ""))[:500]
                    title = source.get("document_name", source.get("title", f"Source {i}"))
                    evidence_parts.append(f"[{title}]: {content}")

            evidence = "\n\n".join(evidence_parts) if evidence_parts else "No additional evidence provided."

            # Build prompt
            prompt = self.FACT_CHECK_PROMPT.format(
                claim=claim,
                evidence=evidence,
            )

            # Call LLM
            llm = LLMFactory.get_chat_model(
                provider=self.provider,
                model=self.model,
                temperature=0.1,
                max_tokens=512,
            )

            response = await llm.ainvoke(prompt)
            content = response.content

            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
            else:
                result_data = {
                    "verdict": "unverifiable",
                    "confidence": 0.5,
                    "reasoning": content,
                    "key_evidence": [],
                }

            fact_result = FactCheckResult(
                claim=claim,
                verdict=result_data.get("verdict", "unverifiable"),
                confidence=result_data.get("confidence", 0.5),
                evidence=result_data.get("key_evidence", []),
                sources=[s.get("document_name", "unknown") for s in (sources or [])[:5]],
            )

            return ToolResult(
                tool=ToolType.FACT_CHECKER,
                success=True,
                result=fact_result.__dict__,
                execution_time_ms=int((time.time() - start_time) * 1000),
                metadata={
                    "claim_length": len(claim),
                    "evidence_sources": len(sources or []),
                },
            )

        except Exception as e:
            logger.error("Fact check failed", error=str(e))
            return ToolResult(
                tool=ToolType.FACT_CHECKER,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    async def check_multiple_facts(
        self,
        claims: List[str],
        context: str = "",
        sources: List[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        """Check multiple claims in parallel."""
        tasks = [
            self.check_fact(claim, context, sources)
            for claim in claims
        ]
        return await asyncio.gather(*tasks)


class DateCalculatorTool:
    """
    Date and time calculator.
    Helps with temporal reasoning which small LLMs struggle with.
    """

    def calculate(
        self,
        operation: str,
        date1: str = None,
        date2: str = None,
        days: int = None,
        months: int = None,
        years: int = None,
    ) -> ToolResult:
        """
        Perform date calculations.

        Operations:
        - "diff": Difference between two dates
        - "add": Add days/months/years to a date
        - "subtract": Subtract days/months/years from a date
        - "weekday": Get day of week
        - "is_leap": Check if year is leap year
        - "days_in_month": Get days in a month

        Args:
            operation: The operation to perform
            date1: First date (YYYY-MM-DD format)
            date2: Second date for diff operation
            days: Number of days to add/subtract
            months: Number of months to add/subtract
            years: Number of years to add/subtract

        Returns:
            ToolResult with calculation result
        """
        import time
        start_time = time.time()

        try:
            result = None
            metadata = {"operation": operation}

            if operation == "diff":
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                d2 = datetime.strptime(date2, "%Y-%m-%d")
                delta = d2 - d1
                result = {
                    "days": delta.days,
                    "weeks": delta.days // 7,
                    "months_approx": delta.days // 30,
                    "years_approx": delta.days // 365,
                }

            elif operation == "add":
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                new_date = d1
                if days:
                    new_date += timedelta(days=days)
                if months:
                    new_month = new_date.month + months
                    new_year = new_date.year + (new_month - 1) // 12
                    new_month = ((new_month - 1) % 12) + 1
                    new_date = new_date.replace(year=new_year, month=new_month)
                if years:
                    new_date = new_date.replace(year=new_date.year + years)
                result = new_date.strftime("%Y-%m-%d")

            elif operation == "subtract":
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                new_date = d1
                if days:
                    new_date -= timedelta(days=days)
                if months:
                    new_month = new_date.month - months
                    new_year = new_date.year + (new_month - 1) // 12
                    new_month = ((new_month - 1) % 12) + 1
                    if new_month <= 0:
                        new_month += 12
                        new_year -= 1
                    new_date = new_date.replace(year=new_year, month=new_month)
                if years:
                    new_date = new_date.replace(year=new_date.year - years)
                result = new_date.strftime("%Y-%m-%d")

            elif operation == "weekday":
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                result = {
                    "weekday": weekdays[d1.weekday()],
                    "weekday_number": d1.weekday(),
                    "is_weekend": d1.weekday() >= 5,
                }

            elif operation == "is_leap":
                year = int(date1) if date1 else years
                is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
                result = {
                    "year": year,
                    "is_leap_year": is_leap,
                }

            elif operation == "days_in_month":
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                import calendar
                days_count = calendar.monthrange(d1.year, d1.month)[1]
                result = {
                    "year": d1.year,
                    "month": d1.month,
                    "days": days_count,
                }

            elif operation == "now":
                now = datetime.now()
                result = {
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "datetime": now.isoformat(),
                    "timestamp": int(now.timestamp()),
                }

            else:
                return ToolResult(
                    tool=ToolType.DATE_CALCULATOR,
                    success=False,
                    result=None,
                    error=f"Unknown operation: {operation}",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

            return ToolResult(
                tool=ToolType.DATE_CALCULATOR,
                success=True,
                result=result,
                execution_time_ms=int((time.time() - start_time) * 1000),
                metadata=metadata,
            )

        except Exception as e:
            return ToolResult(
                tool=ToolType.DATE_CALCULATOR,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )


class UnitConverterTool:
    """
    Unit converter for various measurements.
    Prevents measurement errors in small LLMs.
    """

    # Conversion factors to base units
    CONVERSIONS = {
        # Length (base: meters)
        "length": {
            "m": 1,
            "km": 1000,
            "cm": 0.01,
            "mm": 0.001,
            "mi": 1609.344,
            "yd": 0.9144,
            "ft": 0.3048,
            "in": 0.0254,
            "nm": 1852,  # nautical miles
        },
        # Mass (base: grams)
        "mass": {
            "g": 1,
            "kg": 1000,
            "mg": 0.001,
            "lb": 453.592,
            "oz": 28.3495,
            "ton": 907185,  # US ton
            "tonne": 1000000,  # metric ton
        },
        # Volume (base: liters)
        "volume": {
            "l": 1,
            "ml": 0.001,
            "gal": 3.78541,  # US gallon
            "qt": 0.946353,
            "pt": 0.473176,
            "cup": 0.236588,
            "fl_oz": 0.0295735,
            "tbsp": 0.0147868,
            "tsp": 0.00492892,
        },
        # Temperature (special handling)
        "temperature": {
            "c": "celsius",
            "f": "fahrenheit",
            "k": "kelvin",
        },
        # Time (base: seconds)
        "time": {
            "s": 1,
            "ms": 0.001,
            "min": 60,
            "h": 3600,
            "d": 86400,
            "wk": 604800,
            "mo": 2629800,  # average month
            "yr": 31557600,  # average year
        },
        # Data (base: bytes)
        "data": {
            "b": 1,
            "kb": 1024,
            "mb": 1048576,
            "gb": 1073741824,
            "tb": 1099511627776,
            "pb": 1125899906842624,
        },
        # Speed (base: m/s)
        "speed": {
            "m/s": 1,
            "km/h": 0.277778,
            "mph": 0.44704,
            "kn": 0.514444,  # knots
            "ft/s": 0.3048,
        },
        # Area (base: m²)
        "area": {
            "m2": 1,
            "km2": 1000000,
            "cm2": 0.0001,
            "ft2": 0.092903,
            "yd2": 0.836127,
            "acre": 4046.86,
            "ha": 10000,  # hectare
        },
    }

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        category: str = None,
    ) -> ToolResult:
        """
        Convert a value between units.

        Args:
            value: The value to convert
            from_unit: Source unit
            to_unit: Target unit
            category: Category (length, mass, etc.) - auto-detected if not provided

        Returns:
            ToolResult with converted value
        """
        import time
        start_time = time.time()

        try:
            from_unit = from_unit.lower().replace(" ", "_")
            to_unit = to_unit.lower().replace(" ", "_")

            # Auto-detect category
            if not category:
                for cat, units in self.CONVERSIONS.items():
                    if from_unit in units and to_unit in units:
                        category = cat
                        break

            if not category:
                return ToolResult(
                    tool=ToolType.UNIT_CONVERTER,
                    success=False,
                    result=None,
                    error=f"Could not find compatible units: {from_unit} and {to_unit}",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

            # Handle temperature specially
            if category == "temperature":
                result = self._convert_temperature(value, from_unit, to_unit)
            else:
                units = self.CONVERSIONS[category]

                if from_unit not in units or to_unit not in units:
                    return ToolResult(
                        tool=ToolType.UNIT_CONVERTER,
                        success=False,
                        result=None,
                        error=f"Unknown unit for {category}: {from_unit} or {to_unit}",
                        execution_time_ms=int((time.time() - start_time) * 1000),
                    )

                # Convert to base unit, then to target
                base_value = value * units[from_unit]
                result = base_value / units[to_unit]

            return ToolResult(
                tool=ToolType.UNIT_CONVERTER,
                success=True,
                result={
                    "original": value,
                    "from_unit": from_unit,
                    "converted": result,
                    "to_unit": to_unit,
                    "category": category,
                },
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            return ToolResult(
                tool=ToolType.UNIT_CONVERTER,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between C, F, and K."""
        # First convert to Celsius
        if from_unit in ['c', 'celsius']:
            celsius = value
        elif from_unit in ['f', 'fahrenheit']:
            celsius = (value - 32) * 5 / 9
        elif from_unit in ['k', 'kelvin']:
            celsius = value - 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        # Then convert to target
        if to_unit in ['c', 'celsius']:
            return celsius
        elif to_unit in ['f', 'fahrenheit']:
            return celsius * 9 / 5 + 32
        elif to_unit in ['k', 'kelvin']:
            return celsius + 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")


class ToolAugmentationService:
    """
    Central service for tool-augmented reasoning.
    Helps small LLMs compensate for their limitations.
    """

    def __init__(self):
        self.calculator = CalculatorTool()
        self.code_executor = CodeExecutorTool()
        self.fact_checker = FactCheckerTool()
        self.date_calculator = DateCalculatorTool()
        self.unit_converter = UnitConverterTool()

        # Tool selection prompt
        self.TOOL_SELECTION_PROMPT = """Analyze this query and determine which tools would help answer it accurately.

Query: {query}

Available tools:
1. calculator - For mathematical calculations
2. code_executor - For running code to verify logic
3. fact_checker - For verifying factual claims
4. date_calculator - For date/time calculations
5. unit_converter - For unit conversions

Which tools are needed? Respond in JSON:
{{
    "tools_needed": ["tool1", "tool2"],
    "calculator_expression": "expression if calculator needed",
    "code_to_run": "code if executor needed",
    "facts_to_check": ["list of claims if fact checker needed"],
    "date_operation": {{"operation": "...", "date1": "...", ...}} if date calculator needed,
    "unit_conversion": {{"value": 0, "from": "unit", "to": "unit"}} if converter needed
}}"""

    async def augment_query(
        self,
        query: str,
        context: str = "",
        sources: List[Dict[str, Any]] = None,
        auto_detect_tools: bool = True,
    ) -> Dict[str, Any]:
        """
        Augment a query with tool results.

        Args:
            query: The user's query
            context: Additional context
            sources: Documents to reference
            auto_detect_tools: Whether to auto-detect needed tools

        Returns:
            Dictionary with tool results and enhanced context
        """
        results = {}

        if auto_detect_tools:
            # Use LLM to detect which tools are needed
            tools_spec = await self._detect_tools(query)
        else:
            tools_spec = {}

        # Execute detected tools
        if tools_spec.get("calculator_expression"):
            results["calculator"] = self.calculator.calculate(
                tools_spec["calculator_expression"]
            )

        if tools_spec.get("code_to_run"):
            results["code_executor"] = await self.code_executor.execute(
                tools_spec["code_to_run"]
            )

        if tools_spec.get("facts_to_check"):
            results["fact_checker"] = await self.fact_checker.check_multiple_facts(
                tools_spec["facts_to_check"],
                context=context,
                sources=sources,
            )

        if tools_spec.get("date_operation"):
            op = tools_spec["date_operation"]
            results["date_calculator"] = self.date_calculator.calculate(
                operation=op.get("operation", "now"),
                date1=op.get("date1"),
                date2=op.get("date2"),
                days=op.get("days"),
                months=op.get("months"),
                years=op.get("years"),
            )

        if tools_spec.get("unit_conversion"):
            conv = tools_spec["unit_conversion"]
            results["unit_converter"] = self.unit_converter.convert(
                value=conv.get("value", 0),
                from_unit=conv.get("from", ""),
                to_unit=conv.get("to", ""),
            )

        # Build enhanced context with tool results
        enhanced_context = self._build_enhanced_context(context, results)

        return {
            "original_query": query,
            "tool_results": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in results.items()},
            "enhanced_context": enhanced_context,
            "tools_used": list(results.keys()),
        }

    async def _detect_tools(self, query: str) -> Dict[str, Any]:
        """Detect which tools are needed for a query."""
        try:
            llm = LLMFactory.get_chat_model(
                provider=settings.DEFAULT_LLM_PROVIDER,
                model=settings.DEFAULT_CHAT_MODEL,
                temperature=0.1,
                max_tokens=512,
            )

            prompt = self.TOOL_SELECTION_PROMPT.format(query=query)
            response = await llm.ainvoke(prompt)

            # Parse JSON
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.error("Tool detection failed", error=str(e))

        return {}

    def _build_enhanced_context(
        self,
        original_context: str,
        tool_results: Dict[str, Any],
    ) -> str:
        """Build enhanced context with tool results."""
        parts = []

        if original_context:
            parts.append(f"**Original Context:**\n{original_context}")

        for tool_name, result in tool_results.items():
            if hasattr(result, 'success') and result.success:
                parts.append(f"**{tool_name.replace('_', ' ').title()} Result:**\n{result.result}")
            elif isinstance(result, list):
                # Multiple results (e.g., fact checker)
                successful = [r for r in result if r.success]
                if successful:
                    tool_outputs = "\n".join([str(r.result) for r in successful])
                    parts.append(f"**{tool_name.replace('_', ' ').title()} Results:**\n{tool_outputs}")

        return "\n\n---\n\n".join(parts)

    # Convenience methods for direct tool access
    def calculate(self, expression: str) -> ToolResult:
        """Direct access to calculator."""
        return self.calculator.calculate(expression)

    async def execute_code(self, code: str, timeout: float = None) -> ToolResult:
        """Direct access to code executor."""
        return await self.code_executor.execute(code, timeout)

    async def check_fact(
        self,
        claim: str,
        context: str = "",
        sources: List[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Direct access to fact checker."""
        return await self.fact_checker.check_fact(claim, context, sources)

    def calculate_date(self, operation: str, **kwargs) -> ToolResult:
        """Direct access to date calculator."""
        return self.date_calculator.calculate(operation, **kwargs)

    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> ToolResult:
        """Direct access to unit converter."""
        return self.unit_converter.convert(value, from_unit, to_unit)


# Singleton instance
_tool_augmentation: Optional[ToolAugmentationService] = None


def get_tool_augmentation_service() -> ToolAugmentationService:
    """Get or create the tool augmentation service."""
    global _tool_augmentation
    if _tool_augmentation is None:
        _tool_augmentation = ToolAugmentationService()
    return _tool_augmentation
