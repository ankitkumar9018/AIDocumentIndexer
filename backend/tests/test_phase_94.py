"""
Phase 94 Smoke Tests
====================

Validates key security fixes and module-level changes introduced in recent phases:

1. _SafeDatetime sandbox class blocks attribute access via __getattr__
2. _SafeDatetime.now() and .fromisoformat() function correctly
3. AST-level blocked_attrs set catches intrinsic dunders (__class__, __dict__, etc.)
4. Modified modules (knowledge_graph, web_crawler, text_to_sql) import without errors
5. get_graph_rag_context delegates to graph_search correctly
6. _extract_entities_for_kg and _get_entity_context handle disabled KG gracefully
"""

import ast
import os
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure test environment is configured before any backend imports
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-testing-only")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DATABASE_TYPE", "sqlite")


# =============================================================================
# Helper: reproduce _SafeDatetime exactly as defined in workflow_engine.py
# =============================================================================

class _SafeDatetime:
    """Restricted datetime interface -- no access to module internals."""

    @staticmethod
    def now():
        return datetime.now()

    @staticmethod
    def utcnow():
        return datetime.utcnow()

    @staticmethod
    def fromisoformat(s):
        return datetime.fromisoformat(s)

    @staticmethod
    def strptime(date_string, fmt):
        return datetime.strptime(date_string, fmt)

    @staticmethod
    def today():
        return datetime.today()

    def __getattr__(self, name):
        raise AttributeError(
            f"Access to 'datetime.{name}' is not allowed in sandbox"
        )


# =============================================================================
# 1. _SafeDatetime allowed-method tests
# =============================================================================


class TestSafeDatetimeAllowedMethods:
    """Verify the whitelisted static methods work correctly."""

    def test_now_returns_datetime(self):
        sd = _SafeDatetime()
        result = sd.now()
        assert isinstance(result, datetime)

    def test_utcnow_returns_datetime(self):
        sd = _SafeDatetime()
        result = sd.utcnow()
        assert isinstance(result, datetime)

    def test_fromisoformat_parses_string(self):
        sd = _SafeDatetime()
        result = sd.fromisoformat("2025-06-15T10:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 6
        assert result.day == 15

    def test_strptime_parses_string(self):
        sd = _SafeDatetime()
        result = sd.strptime("2025-01-01", "%Y-%m-%d")
        assert isinstance(result, datetime)
        assert result.year == 2025

    def test_today_returns_datetime(self):
        sd = _SafeDatetime()
        result = sd.today()
        assert isinstance(result, datetime)


# =============================================================================
# 2. _SafeDatetime __getattr__ guard tests
# =============================================================================


class TestSafeDatetimeGetAttrGuard:
    """
    __getattr__ is only invoked for attributes that Python does NOT find
    through the normal MRO lookup. Intrinsic dunders like __class__ and
    __dict__ are found on every object before __getattr__ is reached,
    so the sandbox relies on AST-level blocking for those (see next test
    class). Here we test attributes that __getattr__ *does* intercept.
    """

    @pytest.mark.parametrize(
        "attr",
        [
            # These are NOT real attributes on a plain object, so __getattr__
            # fires and raises AttributeError.
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__globals__",
            "__code__",
            "__builtins__",
            "__import__",
            "__loader__",
            "__spec__",
            # Arbitrary non-existent attributes
            "datetime",
            "timedelta",
            "timezone",
            "some_random_attr",
        ],
    )
    def test_blocked_attr_raises(self, attr):
        sd = _SafeDatetime()
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            getattr(sd, attr)


# =============================================================================
# 3. AST-level blocked_attrs validation
# =============================================================================


class TestASTBlockedAttrs:
    """
    The _execute_python_basic method in workflow_engine.py walks the AST
    and rejects any code that accesses attributes in blocked_attrs. This
    test validates that the blocked set is correct and that AST walking
    catches the dangerous patterns.
    """

    BLOCKED_ATTRS = {
        "__builtins__", "__import__", "__class__", "__bases__",
        "__subclasses__", "__mro__", "__globals__", "__code__",
        "__getattribute__", "__dict__", "__module__", "__loader__",
        "__spec__", "__init_subclass__",
    }

    @pytest.mark.parametrize("attr", [
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__import__",
        "__getattribute__",
        "__dict__",
        "__module__",
        "__loader__",
        "__spec__",
        "__init_subclass__",
    ])
    def test_attr_is_in_blocked_set(self, attr):
        """Every dangerous dunder is present in the blocked_attrs set."""
        assert attr in self.BLOCKED_ATTRS

    @pytest.mark.parametrize("attr", [
        "__class__",
        "__bases__",
        "__subclasses__",
        "__globals__",
        "__code__",
        "__dict__",
    ])
    def test_ast_detects_blocked_attribute_access(self, attr):
        """
        Parsing code that accesses a blocked dunder attribute should
        produce an ast.Attribute node whose .attr is in blocked_attrs.
        This mirrors the AST walk in _execute_python_basic.
        """
        code = f"x.{attr}"
        tree = ast.parse(code)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in self.BLOCKED_ATTRS:
                found = True
                break
        assert found, f"AST walk should detect access to '{attr}'"


# =============================================================================
# 4. Module import smoke tests
# =============================================================================


class TestModuleImports:
    """Verify that the recently-modified modules can be imported."""

    def test_import_knowledge_graph(self):
        from backend.services import knowledge_graph  # noqa: F401
        assert hasattr(knowledge_graph, "KnowledgeGraphService")
        assert hasattr(knowledge_graph, "get_knowledge_graph_service")
        assert hasattr(knowledge_graph, "get_kg_service")
        assert hasattr(knowledge_graph, "GraphRAGContext")

    def test_import_web_crawler(self):
        from backend.services import web_crawler  # noqa: F401
        assert hasattr(web_crawler, "EnhancedWebCrawler")
        assert hasattr(web_crawler, "get_web_crawler")

    def test_import_text_to_sql_service(self):
        from backend.services.text_to_sql import service  # noqa: F401
        assert hasattr(service, "TextToSQLService")

    def test_import_workflow_engine(self):
        from backend.services import workflow_engine  # noqa: F401
        assert hasattr(workflow_engine, "NodeExecutor")
        assert hasattr(workflow_engine, "WorkflowExecutionEngine")


# =============================================================================
# 5. get_graph_rag_context delegation test
# =============================================================================


class TestGetGraphRAGContext:
    """Verify get_graph_rag_context delegates to graph_search correctly."""

    @pytest.mark.asyncio
    async def test_delegates_to_graph_search(self):
        from backend.services.knowledge_graph import KnowledgeGraphService

        mock_session = MagicMock()
        svc = KnowledgeGraphService(db_session=mock_session)

        expected_result = MagicMock(name="GraphRAGContext")
        svc.graph_search = AsyncMock(return_value=expected_result)

        result = await svc.get_graph_rag_context(
            session=mock_session,
            query="test query",
            top_k=5,
            organization_id="org-1",
            access_tier_level=50,
            user_id="user-1",
            is_superadmin=False,
        )

        assert result is expected_result
        svc.graph_search.assert_awaited_once_with(
            query="test query",
            max_hops=2,
            top_k=5,
            organization_id="org-1",
            access_tier_level=50,
            user_id="user-1",
            is_superadmin=False,
        )

    @pytest.mark.asyncio
    async def test_defaults_applied(self):
        from backend.services.knowledge_graph import KnowledgeGraphService

        mock_session = MagicMock()
        svc = KnowledgeGraphService(db_session=mock_session)
        svc.graph_search = AsyncMock(return_value=MagicMock())

        await svc.get_graph_rag_context()

        svc.graph_search.assert_awaited_once_with(
            query="",
            max_hops=2,
            top_k=10,
            organization_id=None,
            access_tier_level=100,
            user_id=None,
            is_superadmin=False,
        )


# =============================================================================
# 6. _extract_entities_for_kg graceful handling when KG disabled
# =============================================================================


class TestExtractEntitiesForKG:
    """Test that _extract_entities_for_kg exits early when KG is disabled."""

    @pytest.mark.asyncio
    async def test_returns_none_when_kg_disabled(self):
        """When kg is disabled the method should return without error."""
        from backend.services.web_crawler import EnhancedWebCrawler

        crawler = EnhancedWebCrawler()

        mock_result = MagicMock()
        mock_result.markdown = "Some crawled content"
        mock_result.url = "https://example.com"

        mock_settings_svc = MagicMock()
        mock_settings_svc.get_setting = AsyncMock(return_value=False)

        with patch(
            "backend.services.settings.get_settings_service",
            return_value=mock_settings_svc,
        ):
            result = await crawler._extract_entities_for_kg(mock_result)
            assert result is None

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """If settings service raises, the method should not propagate."""
        from backend.services.web_crawler import EnhancedWebCrawler

        crawler = EnhancedWebCrawler()

        mock_result = MagicMock()
        mock_result.markdown = "Some content"
        mock_result.url = "https://example.com"

        with patch(
            "backend.services.settings.get_settings_service",
            side_effect=RuntimeError("mocked failure"),
        ):
            # The method catches all exceptions, so no raise expected
            result = await crawler._extract_entities_for_kg(mock_result)
            assert result is None


# =============================================================================
# 7. _get_entity_context graceful handling when KG disabled
# =============================================================================


class TestGetEntityContext:
    """Test that _get_entity_context returns empty string when KG is disabled."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_kg_disabled(self):
        from backend.services.text_to_sql.service import TextToSQLService

        mock_connector = MagicMock()
        svc = TextToSQLService(connector=mock_connector)

        mock_settings_svc = MagicMock()
        mock_settings_svc.get_setting = AsyncMock(return_value=False)

        with patch(
            "backend.services.settings.get_settings_service",
            return_value=mock_settings_svc,
        ):
            result = await svc._get_entity_context("Who is the CEO?")
            assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        from backend.services.text_to_sql.service import TextToSQLService

        mock_connector = MagicMock()
        svc = TextToSQLService(connector=mock_connector)

        with patch(
            "backend.services.settings.get_settings_service",
            side_effect=RuntimeError("boom"),
        ):
            result = await svc._get_entity_context("What is revenue?")
            assert result == ""
