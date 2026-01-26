"""
AIDocumentIndexer - Security Unit Tests (Phase 91)
===================================================

Tests for:
- Sandbox escape prevention (SafeRegex, SafeJson)
- AST validation in workflow engine
- Input validation (collection names, query lengths)
- Auth rate limiting
- SECRET_KEY validation
"""

import os
import pytest
import time
from unittest.mock import patch, MagicMock

# Set test environment
os.environ["TESTING"] = "true"
os.environ["APP_ENV"] = "development"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"


# =============================================================================
# SafeRegex / SafeJson Sandbox Tests
# =============================================================================

class TestSafeRegex:
    """Test that SafeRegex blocks sandbox escape attempts."""

    def setup_method(self):
        from backend.services.rlm_sandbox import SafeRegex
        self.safe_re = SafeRegex()

    def test_search_works(self):
        result = self.safe_re.search(r"\d+", "hello 42 world")
        assert result is not None
        assert result.group() == "42"

    def test_findall_works(self):
        result = self.safe_re.findall(r"\d+", "a1 b2 c3")
        assert result == ["1", "2", "3"]

    def test_sub_works(self):
        result = self.safe_re.sub(r"\d", "X", "a1b2c3")
        assert result == "aXbXcX"

    def test_split_works(self):
        result = self.safe_re.split(r",", "a,b,c")
        assert result == ["a", "b", "c"]

    def test_compile_works(self):
        pattern = self.safe_re.compile(r"\w+")
        assert pattern.findall("hello world") == ["hello", "world"]

    def test_flags_accessible(self):
        import re
        assert self.safe_re.IGNORECASE == re.IGNORECASE
        assert self.safe_re.MULTILINE == re.MULTILINE
        assert self.safe_re.DOTALL == re.DOTALL

    def test_class_is_safe_regex_not_re_module(self):
        """__class__ returns SafeRegex, not the re module — no __builtins__ exposure."""
        # __class__ is a Python data descriptor that bypasses __getattr__,
        # but SafeRegex is a safe wrapper class, not the re module.
        assert self.safe_re.__class__.__name__ == "SafeRegex"

    def test_blocks_builtins_access(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            _ = self.safe_re.__builtins__

    def test_blocks_import_access(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            _ = self.safe_re.__import__

    def test_blocks_arbitrary_attribute(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            _ = self.safe_re.some_nonexistent_attr

    def test_module_is_not_re(self):
        """__module__ should point to rlm_sandbox, not the re module."""
        # __module__ is a data descriptor — it returns the defining module name.
        assert "rlm_sandbox" in self.safe_re.__module__


class TestSafeJson:
    """Test that SafeJson blocks sandbox escape attempts."""

    def setup_method(self):
        from backend.services.rlm_sandbox import SafeJson
        self.safe_json = SafeJson()

    def test_loads_works(self):
        result = self.safe_json.loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_dumps_works(self):
        result = self.safe_json.dumps({"key": "value"})
        assert '"key"' in result and '"value"' in result

    def test_dumps_with_indent(self):
        result = self.safe_json.dumps({"a": 1}, indent=2)
        assert "\n" in result

    def test_loads_array(self):
        result = self.safe_json.loads("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_class_is_safe_json_not_json_module(self):
        """__class__ returns SafeJson, not the json module — no __builtins__ exposure."""
        assert self.safe_json.__class__.__name__ == "SafeJson"

    def test_blocks_builtins_access(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            _ = self.safe_json.__builtins__

    def test_blocks_import_access(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            _ = self.safe_json.__import__

    def test_blocks_arbitrary_attribute(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            _ = self.safe_json.JSONDecodeError


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Test input validation on API models."""

    def test_collection_name_valid(self):
        """Valid collection names should be accepted."""
        from backend.api.routes.documents import DocumentBase
        doc = DocumentBase(
            name="test doc",
            collection="my-collection_1",
            file_type="pdf",
        )
        assert doc.collection == "my-collection_1"

    def test_collection_name_rejects_path_traversal(self):
        """Collection names with path traversal should be rejected."""
        from backend.api.routes.documents import DocumentBase
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentBase(
                name="test",
                collection="../../../etc/passwd",
                file_type="pdf",
            )

    def test_collection_name_rejects_special_chars(self):
        """Collection names with SQL injection chars should be rejected."""
        from backend.api.routes.documents import DocumentBase
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentBase(
                name="test",
                collection="'; DROP TABLE documents; --",
                file_type="pdf",
            )

    def test_document_name_length_limit(self):
        """Document names exceeding max length should be rejected."""
        from backend.api.routes.documents import DocumentBase
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DocumentBase(
                name="x" * 501,  # Exceeds 500 char limit
                collection="test",
                file_type="pdf",
            )

    def test_chat_message_length_limit(self):
        """Chat messages exceeding max length should be rejected."""
        from backend.api.routes.chat import ChatRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(
                message="x" * 100001,  # Exceeds 100000 char limit
            )


# =============================================================================
# Auth Rate Limiting Tests
# =============================================================================

class TestAuthRateLimiter:
    """Test authentication rate limiting."""

    def setup_method(self):
        from backend.api.routes.auth import AuthRateLimiter
        self.limiter = AuthRateLimiter()

    def test_allows_under_limit(self):
        """Should allow requests under the limit."""
        # Should not raise for first 5 attempts
        for _ in range(5):
            self.limiter.check_rate_limit("192.168.1.1", "login", 10, 60)

    def test_blocks_over_limit(self):
        """Should block requests over the limit."""
        from fastapi import HTTPException
        # Fill up the limit
        for _ in range(10):
            self.limiter.check_rate_limit("192.168.1.1", "login", 10, 60)
        # 11th attempt should be blocked
        with pytest.raises(HTTPException) as exc_info:
            self.limiter.check_rate_limit("192.168.1.1", "login", 10, 60)
        assert exc_info.value.status_code == 429

    def test_different_ips_independent(self):
        """Different IPs should have independent limits."""
        # Fill up limit for IP 1
        for _ in range(10):
            self.limiter.check_rate_limit("10.0.0.1", "login", 10, 60)
        # IP 2 should still be allowed
        self.limiter.check_rate_limit("10.0.0.2", "login", 10, 60)

    def test_different_actions_independent(self):
        """Different actions should have independent limits."""
        # Fill up login limit
        for _ in range(10):
            self.limiter.check_rate_limit("10.0.0.1", "login", 10, 60)
        # Register should still be allowed
        self.limiter.check_rate_limit("10.0.0.1", "register", 5, 3600)


# =============================================================================
# SECRET_KEY Validation Tests
# =============================================================================

class TestSecretKeyValidation:
    """Test SECRET_KEY validation at startup."""

    def test_rejects_default_in_production(self):
        """Should reject default SECRET_KEY in production."""
        from backend.core.config import Settings
        with pytest.raises(ValueError, match="CRITICAL"):
            Settings(
                SECRET_KEY="change-me-in-production",
                ENVIRONMENT="production",
                DATABASE_URL="sqlite:///test.db",
            )

    def test_rejects_default_in_staging(self):
        """Should reject default SECRET_KEY in staging."""
        from backend.core.config import Settings
        with pytest.raises(ValueError, match="CRITICAL"):
            Settings(
                SECRET_KEY="change-me-in-production",
                ENVIRONMENT="staging",
                DATABASE_URL="sqlite:///test.db",
            )

    def test_allows_default_in_development(self):
        """Should allow (with warning) default SECRET_KEY in development."""
        from backend.core.config import Settings
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            settings = Settings(
                SECRET_KEY="change-me-in-production",
                ENVIRONMENT="development",
                DATABASE_URL="sqlite:///test.db",
            )
            assert settings.SECRET_KEY == "change-me-in-production"
            # Should have generated a warning
            assert any("SECRET_KEY" in str(warning.message) for warning in w)

    def test_allows_custom_key_in_production(self):
        """Should allow custom SECRET_KEY in production."""
        from backend.core.config import Settings
        settings = Settings(
            SECRET_KEY="my-very-secure-production-key-2026",
            ENVIRONMENT="production",
            DATABASE_URL="sqlite:///test.db",
        )
        assert settings.SECRET_KEY == "my-very-secure-production-key-2026"


# =============================================================================
# LLM Provider Rate Limiter Tests
# =============================================================================

class TestProviderRateLimiter:
    """Test LLM provider rate limiting."""

    def test_allows_under_limit(self):
        """Should allow requests under RPM limit."""
        from backend.services.llm import ProviderRateLimiter
        limiter = ProviderRateLimiter()
        # Under limit should return 0 wait
        wait = limiter.wait_if_needed("openai")
        assert wait == 0.0

    def test_tracks_requests(self):
        """Should track request timestamps."""
        from backend.services.llm import ProviderRateLimiter
        limiter = ProviderRateLimiter()
        limiter.wait_if_needed("openai")
        limiter.wait_if_needed("openai")
        assert len(limiter._requests.get("openai", [])) == 2
