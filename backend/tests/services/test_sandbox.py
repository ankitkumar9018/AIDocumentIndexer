"""
AIDocumentIndexer - Sandbox Escape Prevention Tests (Phase 91)
==============================================================

Tests for sandbox security across all three execution environments:
- rlm_sandbox.py (LocalSandbox with SafeRegex/SafeJson)
- recursive_lm.py (RLM code execution with SafeRegex/SafeJson)
- workflow_engine.py (Python code step execution with AST validation)

These tests verify that LLM-generated code cannot escape the sandbox
via module internals (__builtins__, __import__, __class__.__bases__).
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment
os.environ["TESTING"] = "true"
os.environ["APP_ENV"] = "development"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"


# =============================================================================
# LocalSandbox Execution Tests (rlm_sandbox.py)
# =============================================================================

class TestLocalSandbox:
    """Test that LocalSandbox properly restricts code execution."""

    def setup_method(self):
        from backend.services.rlm_sandbox import LocalSandbox, SandboxConfig
        self.sandbox = LocalSandbox(SandboxConfig())

    @pytest.mark.asyncio
    async def test_basic_code_execution(self):
        """Simple code should execute successfully."""
        code = "print('hello world')"
        result = await self.sandbox.execute(
            code=code,
            context="test context",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"] * len(ps),
        )
        assert result.success
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_context_accessible(self):
        """Code should have access to the context variable."""
        code = "print(len(context))"
        result = await self.sandbox.execute(
            code=code,
            context="test data 12345",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.success
        assert "15" in result.output

    @pytest.mark.asyncio
    async def test_final_function_works(self):
        """FINAL() should set the answer and stop execution."""
        code = "FINAL('the answer is 42')"
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.success
        assert result.variables["answer"]["content"] == "the answer is 42"
        assert result.variables["answer"]["ready"] is True

    @pytest.mark.asyncio
    async def test_regex_works_in_sandbox(self):
        """Safe regex operations should work inside sandbox."""
        code = """
result = re.findall(r'\\d+', 'abc 123 def 456')
print(result)
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.success
        assert "123" in result.output

    @pytest.mark.asyncio
    async def test_json_works_in_sandbox(self):
        """Safe JSON operations should work inside sandbox."""
        code = """
data = json.loads('{"key": "value"}')
print(json.dumps(data))
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.success
        assert "key" in result.output

    @pytest.mark.asyncio
    async def test_import_blocked(self):
        """Import statements should be blocked."""
        code = "import os"
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        # exec() with restricted __builtins__ should raise NameError or similar
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_builtins_import_blocked(self):
        """__import__ should not be accessible."""
        code = "os = __import__('os')"
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_re_class_bases_subclasses_escape_blocked(self):
        """Full sandbox escape chain via re.__class__.__bases__[0].__subclasses__() should be blocked.

        Even though __class__ is accessible (Python data descriptor), the sandbox
        environment restricts __builtins__ so __import__ is not available. The
        real security is that SafeRegex is NOT the re module and doesn't expose
        __builtins__ or __import__."""
        code = """
# Try to use re to access os module via class hierarchy
try:
    subs = re.__class__.__bases__[0].__subclasses__()
    # Even if we get subclasses, we can't import anything
    # because __builtins__ is restricted
    os_module = __import__('os')
    print(os_module.listdir('/'))
except Exception as e:
    print(f'blocked: {e}')
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        # Either it errors or the output shows it was blocked
        if result.output:
            assert "blocked" in result.output.lower()
        else:
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_re_builtins_escape_blocked(self):
        """Access to re.__builtins__ should be blocked by __getattr__."""
        code = """
b = re.__builtins__
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_json_builtins_escape_blocked(self):
        """Access to json.__builtins__ should be blocked by __getattr__."""
        code = """
b = json.__builtins__
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_import_via_builtins_blocked(self):
        """Cannot use __import__ from restricted builtins."""
        code = """
os = __import__('os')
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_open_file_blocked(self):
        """open() should not be available in sandbox."""
        code = "f = open('/etc/passwd')"
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """Output exceeding max_output_chars should be truncated."""
        from backend.services.rlm_sandbox import SandboxConfig, LocalSandbox
        config = SandboxConfig(max_output_chars=50)
        sandbox = LocalSandbox(config)
        code = "print('A' * 1000)"
        result = await sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["response"],
        )
        assert result.truncated
        assert len(result.output) <= 50

    @pytest.mark.asyncio
    async def test_llm_query_accessible(self):
        """LLM query function should be callable from sandbox."""
        code = """
response = llm('What is 2+2?')
print(response)
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "4",
            llm_batch=lambda ps: ["4"],
        )
        assert result.success
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_llm_batch_accessible(self):
        """LLM batch function should be callable from sandbox."""
        code = """
responses = llm_batch(['q1', 'q2'])
print(len(responses))
"""
        result = await self.sandbox.execute(
            code=code,
            context="",
            llm_query=lambda p: "r",
            llm_batch=lambda ps: [f"r{i}" for i in range(len(ps))],
        )
        assert result.success
        assert "2" in result.output


# =============================================================================
# Workflow Engine Python Execution Tests
# =============================================================================

class TestWorkflowEngineSandbox:
    """Test workflow engine's Python code execution sandbox."""

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Simple variable assignment should work."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "result = 42",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "success"
        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_import_blocked(self):
        """Import statements should be detected and blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "import os",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"
        assert "not allowed" in result["error"].lower() or "import" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_from_import_blocked(self):
        """from X import Y should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "from os import path",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_eval_blocked(self):
        """eval() should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "result = eval('1+1')",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"
        assert "not allowed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_exec_blocked(self):
        """exec() should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "exec('print(1)')",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_open_blocked(self):
        """open() should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "f = open('/etc/passwd')",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_dunder_class_blocked(self):
        """__class__ attribute access should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "x = ''.__class__",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"
        assert "__class__" in result["error"]

    @pytest.mark.asyncio
    async def test_dunder_bases_blocked(self):
        """__bases__ attribute access should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "x = str.__bases__",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"
        assert "__bases__" in result["error"]

    @pytest.mark.asyncio
    async def test_dunder_subclasses_blocked(self):
        """__subclasses__() should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "x = str.__subclasses__",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_dunder_globals_blocked(self):
        """__globals__ attribute access should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "x = (lambda: 0).__globals__",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_dunder_builtins_blocked(self):
        """__builtins__ access should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "x = None.__builtins__",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_getattr_blocked(self):
        """getattr() should be blocked."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "x = getattr('', '__class__')",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_variables_accessible(self):
        """Context variables should be accessible in code."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "result = variables['x'] + variables['y']",
            {"variables": {"x": 10, "y": 20}, "input": {}},
        )
        assert result["status"] == "success"
        assert result["result"] == 30

    @pytest.mark.asyncio
    async def test_output_dict_works(self):
        """The output dict should be writable from code."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "output['message'] = 'done'",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "success"
        assert result["output"]["message"] == "done"

    @pytest.mark.asyncio
    async def test_syntax_error_handled(self):
        """Syntax errors should return error status, not crash."""
        from backend.services.workflow_engine import NodeExecutor
        engine = NodeExecutor.__new__(NodeExecutor)

        result = await engine._execute_python_basic(
            "def incomplete(",
            {"variables": {}, "input": {}},
        )
        assert result["status"] == "error"
        assert "syntax" in result["error"].lower()


# =============================================================================
# Sandbox Factory Tests
# =============================================================================

class TestSandboxFactory:
    """Test sandbox factory and configuration."""

    def test_create_local_sandbox(self):
        from backend.services.rlm_sandbox import create_sandbox, SandboxConfig, SandboxType, LocalSandbox
        config = SandboxConfig(sandbox_type=SandboxType.LOCAL)
        sandbox = create_sandbox(config)
        assert isinstance(sandbox, LocalSandbox)

    def test_create_docker_sandbox(self):
        from backend.services.rlm_sandbox import create_sandbox, SandboxConfig, SandboxType, DockerSandbox
        config = SandboxConfig(sandbox_type=SandboxType.DOCKER)
        sandbox = create_sandbox(config)
        assert isinstance(sandbox, DockerSandbox)

    def test_create_modal_sandbox(self):
        from backend.services.rlm_sandbox import create_sandbox, SandboxConfig, SandboxType, ModalSandbox
        config = SandboxConfig(sandbox_type=SandboxType.MODAL)
        sandbox = create_sandbox(config)
        assert isinstance(sandbox, ModalSandbox)

    def test_create_prime_sandbox(self):
        from backend.services.rlm_sandbox import create_sandbox, SandboxConfig, SandboxType, PrimeSandbox
        config = SandboxConfig(sandbox_type=SandboxType.PRIME)
        sandbox = create_sandbox(config)
        assert isinstance(sandbox, PrimeSandbox)

    def test_default_sandbox_is_local(self):
        from backend.services.rlm_sandbox import create_sandbox, LocalSandbox
        sandbox = create_sandbox()
        assert isinstance(sandbox, LocalSandbox)

    def test_sandbox_config_defaults(self):
        from backend.services.rlm_sandbox import SandboxConfig, SandboxType
        config = SandboxConfig()
        assert config.sandbox_type == SandboxType.LOCAL
        assert config.timeout_seconds == 120.0
        assert config.max_output_chars == 8192
        assert config.max_memory_mb == 512


# =============================================================================
# SafeRegex Edge Cases
# =============================================================================

class TestSafeRegexEdgeCases:
    """Additional edge case tests for SafeRegex."""

    def setup_method(self):
        from backend.services.rlm_sandbox import SafeRegex
        self.safe_re = SafeRegex()

    def test_match_works(self):
        result = self.safe_re.match(r"hello", "hello world")
        assert result is not None
        assert result.group() == "hello"

    def test_match_no_match(self):
        result = self.safe_re.match(r"world", "hello world")
        assert result is None

    def test_finditer_works(self):
        results = list(self.safe_re.finditer(r"\d+", "a1 b2 c3"))
        assert len(results) == 3
        assert results[0].group() == "1"

    def test_sub_with_count(self):
        result = self.safe_re.sub(r"\d", "X", "a1b2c3", count=2)
        assert result == "aXbXc3"

    def test_split_with_maxsplit(self):
        result = self.safe_re.split(r",", "a,b,c,d", maxsplit=2)
        assert result == ["a", "b", "c,d"]

    def test_compile_with_flags(self):
        pattern = self.safe_re.compile(r"hello", self.safe_re.IGNORECASE)
        assert pattern.match("HELLO") is not None

    def test_search_with_flags(self):
        result = self.safe_re.search(r"hello", "HELLO", self.safe_re.IGNORECASE)
        assert result is not None

    def test_no_access_to_internal_functions(self):
        """Should not be able to access non-whitelisted re functions."""
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            self.safe_re.purge()

    def test_no_access_to_error_class(self):
        """Should not be able to access re.error."""
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            self.safe_re.error


# =============================================================================
# SafeJson Edge Cases
# =============================================================================

class TestSafeJsonEdgeCases:
    """Additional edge case tests for SafeJson."""

    def setup_method(self):
        from backend.services.rlm_sandbox import SafeJson
        self.safe_json = SafeJson()

    def test_dumps_sort_keys(self):
        result = self.safe_json.dumps({"b": 2, "a": 1}, sort_keys=True)
        assert result.index('"a"') < result.index('"b"')

    def test_dumps_ensure_ascii_false(self):
        result = self.safe_json.dumps({"key": "café"}, ensure_ascii=False)
        assert "café" in result

    def test_loads_invalid_json_raises(self):
        import json
        with pytest.raises(json.JSONDecodeError):
            self.safe_json.loads("not valid json")

    def test_loads_nested_objects(self):
        data = self.safe_json.loads('{"a": {"b": {"c": 1}}}')
        assert data["a"]["b"]["c"] == 1

    def test_no_access_to_decoder(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            self.safe_json.JSONDecoder

    def test_no_access_to_encoder(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            self.safe_json.JSONEncoder

    def test_no_access_to_tool(self):
        with pytest.raises(AttributeError, match="not allowed in sandbox"):
            self.safe_json.tool
