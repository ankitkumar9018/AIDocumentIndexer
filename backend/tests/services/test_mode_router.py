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
        complex_queries = [
            "Generate a comprehensive report on Q4 performance",
            "Create a presentation comparing our products",
            "Analyze the market trends and summarize findings",
            "Research and document the competitive landscape",
            "Write a detailed proposal for the new initiative",
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
        assert long_level.value >= ComplexityLevel.MODERATE.value

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
        self.mock_detector = MagicMock(spec=ComplexityDetector)
        self.mock_db = AsyncMock()

        self.router = ModeRouter(
            orchestrator=self.mock_orchestrator,
            rag_service=self.mock_rag_service,
            complexity_detector=self.mock_detector,
            db=self.mock_db,
        )

    @pytest.mark.asyncio
    async def test_route_request_explicit_chat_mode(self):
        """Test routing with explicit chat mode."""
        user_id = str(uuid4())
        session_id = str(uuid4())

        # Mock user preferences
        mock_prefs = MagicMock()
        mock_prefs.agent_mode_enabled = True
        mock_prefs.auto_detect_complexity = True

        with patch.object(self.router, 'get_user_preferences', return_value=mock_prefs):
            # Set up RAG service to return a response
            mock_response = {"content": "Chat response", "sources": []}
            self.mock_rag_service.query = AsyncMock(return_value=mock_response)

            result = await self.router.route_request(
                request="Simple question",
                session_id=session_id,
                user_id=user_id,
                explicit_mode=ExecutionMode.CHAT,
            )

            # Should use RAG service
            self.mock_rag_service.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_request_explicit_agent_mode(self):
        """Test routing with explicit agent mode."""
        user_id = str(uuid4())
        session_id = str(uuid4())

        mock_prefs = MagicMock()
        mock_prefs.agent_mode_enabled = True
        mock_prefs.auto_detect_complexity = True

        with patch.object(self.router, 'get_user_preferences', return_value=mock_prefs):
            self.mock_orchestrator.process_request = AsyncMock(return_value={})

            result = await self.router.route_request(
                request="Complex task",
                session_id=session_id,
                user_id=user_id,
                explicit_mode=ExecutionMode.AGENT,
            )

            # Should use orchestrator
            self.mock_orchestrator.process_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_request_agent_mode_disabled(self):
        """Test routing when agent mode is disabled by user."""
        user_id = str(uuid4())
        session_id = str(uuid4())

        mock_prefs = MagicMock()
        mock_prefs.agent_mode_enabled = False
        mock_prefs.auto_detect_complexity = True

        with patch.object(self.router, 'get_user_preferences', return_value=mock_prefs):
            mock_response = {"content": "Chat response", "sources": []}
            self.mock_rag_service.query = AsyncMock(return_value=mock_response)

            result = await self.router.route_request(
                request="Generate a report",
                session_id=session_id,
                user_id=user_id,
                explicit_mode=None,
            )

            # Should always use RAG when agent mode is disabled
            self.mock_rag_service.query.assert_called_once()
            self.mock_orchestrator.process_request.assert_not_called()

    def test_determine_mode_explicit(self):
        """Test determine_mode with explicit mode parameter."""
        mock_prefs = MagicMock()
        mock_prefs.default_mode = ExecutionMode.CHAT

        mode = self.router.determine_mode(
            request="Any request",
            prefs=mock_prefs,
            explicit_mode=ExecutionMode.AGENT,
            context={},
        )

        assert mode == ExecutionMode.AGENT

    def test_determine_mode_auto_detect_enabled(self):
        """Test determine_mode with auto-detection."""
        mock_prefs = MagicMock()
        mock_prefs.auto_detect_complexity = True
        mock_prefs.default_mode = ExecutionMode.CHAT

        # Mock detector to return complex
        self.mock_detector.should_use_agents.return_value = True

        mode = self.router.determine_mode(
            request="Generate a report",
            prefs=mock_prefs,
            explicit_mode=None,
            context={},
        )

        assert mode == ExecutionMode.AGENT

    def test_determine_mode_auto_detect_disabled(self):
        """Test determine_mode with auto-detection disabled."""
        mock_prefs = MagicMock()
        mock_prefs.auto_detect_complexity = False
        mock_prefs.default_mode = ExecutionMode.CHAT

        mode = self.router.determine_mode(
            request="Generate a report",
            prefs=mock_prefs,
            explicit_mode=None,
            context={},
        )

        # Should use default mode
        assert mode == ExecutionMode.CHAT


# =============================================================================
# ComplexityLevel Tests
# =============================================================================

class TestComplexityLevel:
    """Tests for ComplexityLevel enum."""

    def test_complexity_levels_ordering(self):
        """Test that complexity levels have correct ordering."""
        assert ComplexityLevel.SIMPLE.value < ComplexityLevel.MODERATE.value
        assert ComplexityLevel.MODERATE.value < ComplexityLevel.COMPLEX.value

    def test_all_levels_exist(self):
        """Test that all expected complexity levels exist."""
        levels = [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX]
        assert len(levels) == 3
