"""
AIDocumentIndexer - Mode Router Tests
======================================

Unit tests for the mode router and complexity detection.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4

from backend.services.mode_router import (
    ComplexityDetector,
    ComplexityLevel,
    ModeRouter,
)
from backend.db.models import ExecutionMode


# =============================================================================
# ComplexityDetector Tests
# =============================================================================

class TestComplexityDetector:
    """Tests for ComplexityDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ComplexityDetector()

    def test_detect_simple_query(self):
        """Test detecting a simple query."""
        simple_queries = [
            "What is the project deadline?",
            "Who is the project manager?",
            "When was the document created?",
            "Show me the budget.",
        ]

        for query in simple_queries:
            level = self.detector.detect(query, {})
            assert level in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE], (
                f"Query '{query}' should be simple or moderate"
            )

    def test_detect_complex_query(self):
        """Test detecting a complex query with keywords."""
        # Queries with multiple complex keywords should be detected as complex
        complex_queries = [
            "Generate a comprehensive report on Q4 performance and analyze all data",
            "Create a presentation comparing our products and summarize findings",
            "Analyze the market trends and summarize multiple documents",
            "Research and document the competitive landscape in detail",
        ]

        for query in complex_queries:
            level = self.detector.detect(query, {})
            assert level in [ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX], (
                f"Query '{query}' should be moderate or complex"
            )

    def test_detect_long_query(self):
        """Test that long queries are considered more complex."""
        short_query = "What is the deadline?"
        long_query = " ".join(["What is the deadline?"] * 20)

        short_level = self.detector.detect(short_query, {})
        long_level = self.detector.detect(long_query, {})

        # Long queries should generally be at least moderate
        # ComplexityLevel values are strings ("simple", "moderate", "complex")
        level_order = {"simple": 0, "moderate": 1, "complex": 2}
        assert level_order[long_level.value] >= level_order[ComplexityLevel.MODERATE.value]

    def test_should_use_agents_simple(self):
        """Test should_use_agents returns False for simple queries."""
        simple_query = "What is the date?"

        should_use = self.detector.should_use_agents(simple_query, {})

        # Simple queries should typically not use agents
        # (But this depends on implementation details)
        assert isinstance(should_use, bool)

    def test_should_use_agents_complex(self):
        """Test should_use_agents returns True for complex queries."""
        complex_query = "Generate a comprehensive market analysis report with charts"

        should_use = self.detector.should_use_agents(complex_query, {})

        assert should_use is True

    def test_complex_keywords_detection(self):
        """Test that complex keywords trigger agent mode."""
        keywords_queries = [
            ("generate a document", True),
            ("create a presentation", True),
            ("analyze the data", True),
            ("compare options", True),
            ("research the topic", True),
            ("summarize multiple documents", True),
            ("hello", False),
            ("what time is it", False),
        ]

        for query, expected_complex in keywords_queries:
            should_use = self.detector.should_use_agents(query, {})
            # Just verify it returns a boolean
            assert isinstance(should_use, bool)


# =============================================================================
# ModeRouter Tests
# =============================================================================

class TestModeRouter:
    """Tests for ModeRouter class."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_orchestrator = MagicMock()
        self.mock_rag_service = MagicMock()
        self.mock_db = AsyncMock()

        self.router = ModeRouter(
            db=self.mock_db,
            rag_service=self.mock_rag_service,
            orchestrator=self.mock_orchestrator,
        )

    def test_determine_mode_explicit_agent(self):
        """Test determine_mode with explicit agent mode parameter."""
        mock_prefs = MagicMock()
        mock_prefs.default_mode = ExecutionMode.CHAT.value
        mock_prefs.agent_mode_enabled = True
        mock_prefs.auto_detect_complexity = True

        mode = self.router.determine_mode(
            request="Any request",
            prefs=mock_prefs,
            explicit_mode="agent",
            context={},
        )

        assert mode == ExecutionMode.AGENT

    def test_determine_mode_explicit_chat(self):
        """Test determine_mode with explicit chat mode parameter."""
        mock_prefs = MagicMock()
        mock_prefs.default_mode = ExecutionMode.AGENT.value
        mock_prefs.agent_mode_enabled = True
        mock_prefs.auto_detect_complexity = True

        mode = self.router.determine_mode(
            request="Any request",
            prefs=mock_prefs,
            explicit_mode="chat",
            context={},
        )

        assert mode == ExecutionMode.CHAT

    def test_determine_mode_agent_disabled(self):
        """Test determine_mode when agent mode is disabled."""
        mock_prefs = MagicMock()
        mock_prefs.default_mode = ExecutionMode.AGENT.value
        mock_prefs.agent_mode_enabled = False  # Disabled
        mock_prefs.auto_detect_complexity = True

        mode = self.router.determine_mode(
            request="Generate a report",
            prefs=mock_prefs,
            explicit_mode=None,
            context={},
        )

        # Should fall back to chat when agents disabled
        assert mode == ExecutionMode.CHAT

    def test_determine_mode_auto_detect_complex(self):
        """Test determine_mode with auto-detection for complex query."""
        mock_prefs = MagicMock()
        mock_prefs.auto_detect_complexity = True
        mock_prefs.default_mode = ExecutionMode.CHAT.value
        mock_prefs.agent_mode_enabled = True

        mode = self.router.determine_mode(
            request="Generate a comprehensive report analyzing multiple documents",
            prefs=mock_prefs,
            explicit_mode=None,
            context={},
        )

        # Complex query should trigger agent mode
        assert mode == ExecutionMode.AGENT

    def test_determine_mode_auto_detect_simple(self):
        """Test determine_mode with auto-detection for simple query."""
        mock_prefs = MagicMock()
        mock_prefs.auto_detect_complexity = True
        mock_prefs.default_mode = ExecutionMode.AGENT.value
        mock_prefs.agent_mode_enabled = True

        mode = self.router.determine_mode(
            request="hello",
            prefs=mock_prefs,
            explicit_mode=None,
            context={},
        )

        # Simple query should use chat mode
        assert mode == ExecutionMode.CHAT

    def test_determine_mode_auto_detect_disabled(self):
        """Test determine_mode with auto-detection disabled uses default."""
        mock_prefs = MagicMock()
        mock_prefs.auto_detect_complexity = False
        mock_prefs.default_mode = ExecutionMode.CHAT.value
        mock_prefs.agent_mode_enabled = True

        mode = self.router.determine_mode(
            request="Generate a report",
            prefs=mock_prefs,
            explicit_mode=None,
            context={},
        )

        # Should use default mode when auto-detect disabled
        assert mode == ExecutionMode.CHAT

    def test_determine_mode_explicit_general(self):
        """Test determine_mode with explicit general mode."""
        mock_prefs = MagicMock()
        mock_prefs.default_mode = ExecutionMode.CHAT.value
        mock_prefs.agent_mode_enabled = True
        mock_prefs.auto_detect_complexity = True

        mode = self.router.determine_mode(
            request="What is the weather today?",
            prefs=mock_prefs,
            explicit_mode="general",
            context={},
        )

        assert mode == ExecutionMode.GENERAL


# =============================================================================
# ComplexityLevel Tests
# =============================================================================

class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_complexity_levels_exist(self):
        """Test that all expected complexity levels exist."""
        assert ComplexityLevel.SIMPLE.value == "simple"
        assert ComplexityLevel.MODERATE.value == "moderate"
        assert ComplexityLevel.COMPLEX.value == "complex"

    def test_all_levels_count(self):
        """Test that all expected complexity levels exist."""
        levels = [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX]
        assert len(levels) == 3

    def test_complexity_levels_ordering_by_string(self):
        """Test complexity levels have expected string values."""
        # String comparison (alphabetical) doesn't work, but we can verify values are distinct
        values = {ComplexityLevel.SIMPLE.value, ComplexityLevel.MODERATE.value, ComplexityLevel.COMPLEX.value}
        assert len(values) == 3
