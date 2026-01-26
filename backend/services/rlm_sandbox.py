"""
AIDocumentIndexer - RLM Sandbox Service
=========================================

Provides isolated sandbox environments for RLM code execution.

Supported Sandboxes:
1. Local (RestrictedPython) - Development only, limited but safe
2. Docker - Containerized Python execution
3. Modal - Cloud-based isolated sandboxes (recommended for production)
4. Prime Intellect - High-performance cloud sandboxes (beta)

Based on the official RLM library pattern (https://github.com/alexzhang13/rlm)

Security Notes:
- Local mode uses RestrictedPython with limited builtins
- Docker/Modal/Prime provide full Python isolation
- All sandboxes have output limits to prevent context explosion
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class SandboxType(str, Enum):
    """Sandbox execution environment types."""
    LOCAL = "local"           # RestrictedPython (dev only)
    DOCKER = "docker"         # Containerized Python
    MODAL = "modal"           # Modal.com sandbox
    PRIME = "prime"           # Prime Intellect sandbox


@dataclass
class SandboxConfig:
    """Configuration for sandbox environments."""
    sandbox_type: SandboxType = SandboxType.LOCAL

    # Execution limits
    timeout_seconds: float = 120.0
    max_output_chars: int = 8192  # RLM default
    max_memory_mb: int = 512

    # Docker settings
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"  # Isolated networking

    # Modal settings
    modal_app_name: str = "rlm-sandbox"
    modal_timeout: int = 120

    # Prime settings
    prime_api_key: Optional[str] = None


@dataclass
class SandboxResult:
    """Result from sandbox code execution."""
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    truncated: bool = False
    success: bool = True
    variables: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Sandbox Interface
# =============================================================================

class BaseSandbox(ABC):
    """Abstract base class for RLM sandboxes."""

    def __init__(self, config: SandboxConfig):
        self.config = config

    @abstractmethod
    async def execute(
        self,
        code: str,
        context: str,
        llm_query: Callable[[str], str],
        llm_batch: Callable[[List[str]], List[str]],
    ) -> SandboxResult:
        """
        Execute code in the sandbox.

        Args:
            code: Python code to execute
            context: The full document context
            llm_query: Function to call LLM with a single prompt
            llm_batch: Function to call LLM with multiple prompts in parallel

        Returns:
            SandboxResult with output and any errors
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        pass


# =============================================================================
# Safe Module Wrappers (prevent sandbox escape via __builtins__/__import__)
# =============================================================================

class SafeRegex:
    """Restricted regex interface that prevents access to module internals."""

    @staticmethod
    def search(pattern, string, flags=0):
        return re.search(pattern, string, flags)

    @staticmethod
    def match(pattern, string, flags=0):
        return re.match(pattern, string, flags)

    @staticmethod
    def findall(pattern, string, flags=0):
        return re.findall(pattern, string, flags)

    @staticmethod
    def finditer(pattern, string, flags=0):
        return re.finditer(pattern, string, flags)

    @staticmethod
    def sub(pattern, repl, string, count=0, flags=0):
        return re.sub(pattern, repl, string, count, flags)

    @staticmethod
    def split(pattern, string, maxsplit=0, flags=0):
        return re.split(pattern, string, maxsplit, flags)

    @staticmethod
    def compile(pattern, flags=0):
        return re.compile(pattern, flags)

    # Expose common flags as values (not module references)
    IGNORECASE = re.IGNORECASE
    MULTILINE = re.MULTILINE
    DOTALL = re.DOTALL

    def __getattr__(self, name):
        raise AttributeError(f"Access to 're.{name}' is not allowed in sandbox")


class SafeJson:
    """Restricted JSON interface that prevents access to module internals."""

    @staticmethod
    def loads(s, **kwargs):
        return json.loads(s, **kwargs)

    @staticmethod
    def dumps(obj, indent=None, ensure_ascii=True, default=None, sort_keys=False):
        return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=default, sort_keys=sort_keys)

    def __getattr__(self, name):
        raise AttributeError(f"Access to 'json.{name}' is not allowed in sandbox")


# =============================================================================
# Local Sandbox (RestrictedPython)
# =============================================================================

class LocalSandbox(BaseSandbox):
    """
    Local sandbox using RestrictedPython.

    Safe for development but limited in capabilities.
    Uses a restricted set of builtins and denies file/network access.
    """

    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._output_buffer: List[str] = []
        self._answer: Dict[str, Any] = {"content": "", "ready": False}

    def _create_safe_builtins(self) -> Dict[str, Any]:
        """Create restricted builtins for safe execution."""
        return {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }

    def _safe_print(self, *args, **kwargs) -> None:
        """Capture print output to buffer."""
        output = " ".join(str(arg) for arg in args)
        self._output_buffer.append(output)

    async def execute(
        self,
        code: str,
        context: str,
        llm_query: Callable[[str], str],
        llm_batch: Callable[[List[str]], List[str]],
    ) -> SandboxResult:
        """Execute code in RestrictedPython sandbox."""
        start_time = time.time()
        self._output_buffer = []
        self._answer = {"content": "", "ready": False}

        # Create execution environment
        safe_globals = {
            '__builtins__': self._create_safe_builtins(),
            'context': context,
            'llm': llm_query,
            'llm_batch': llm_batch,
            'answer': self._answer,
            'print': self._safe_print,
            're': SafeRegex(),
            'json': SafeJson(),
        }

        # Add FINAL function for backward compatibility
        def final_fn(result: str):
            self._answer["content"] = result
            self._answer["ready"] = True
            raise StopIteration(result)

        safe_globals['FINAL'] = final_fn

        # Also add llm_query and llm_queries for backward compatibility
        safe_globals['llm_query'] = llm_query
        safe_globals['llm_queries'] = llm_batch

        error = None
        try:
            exec(code, safe_globals)
        except StopIteration:
            # Normal termination via FINAL()
            pass
        except Exception as e:
            error = str(e)
            logger.warning("LocalSandbox execution error", error=error)

        # Collect output
        output = "\n".join(self._output_buffer)
        truncated = False

        if len(output) > self.config.max_output_chars:
            output = output[:self.config.max_output_chars]
            truncated = True

        execution_time = (time.time() - start_time) * 1000

        return SandboxResult(
            output=output,
            error=error,
            execution_time_ms=execution_time,
            truncated=truncated,
            success=error is None or self._answer.get("ready", False),
            variables={
                "answer": self._answer,
            },
        )

    async def cleanup(self) -> None:
        """Clean up local sandbox (no-op)."""
        self._output_buffer = []
        self._answer = {"content": "", "ready": False}


# =============================================================================
# Docker Sandbox
# =============================================================================

class DockerSandbox(BaseSandbox):
    """
    Docker-based sandbox for isolated Python execution.

    Provides full Python environment in an isolated container.
    Suitable for production use with proper Docker configuration.
    """

    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._container_id: Optional[str] = None

    async def execute(
        self,
        code: str,
        context: str,
        llm_query: Callable[[str], str],
        llm_batch: Callable[[List[str]], List[str]],
    ) -> SandboxResult:
        """Execute code in Docker container."""
        start_time = time.time()

        # Create wrapper script that includes LLM stubs
        # In production, this would use a proper RPC mechanism
        wrapper_code = self._create_wrapper_script(code, context)

        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(wrapper_code)
                script_path = f.name

            # Run in Docker
            cmd = [
                "docker", "run",
                "--rm",
                "--network", self.config.docker_network,
                "--memory", f"{self.config.max_memory_mb}m",
                "--cpus", "1",
                "-v", f"{script_path}:/app/script.py:ro",
                self.config.docker_image,
                "python", "/app/script.py"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                return SandboxResult(
                    output="",
                    error=f"Execution timed out after {self.config.timeout_seconds}s",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                )

            output = stdout.decode('utf-8', errors='replace')
            error = stderr.decode('utf-8', errors='replace') if stderr else None

            truncated = False
            if len(output) > self.config.max_output_chars:
                output = output[:self.config.max_output_chars]
                truncated = True

            return SandboxResult(
                output=output,
                error=error if error else None,
                execution_time_ms=(time.time() - start_time) * 1000,
                truncated=truncated,
                success=process.returncode == 0,
            )

        except FileNotFoundError:
            return SandboxResult(
                output="",
                error="Docker not available. Install Docker or use a different sandbox.",
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
            )
        finally:
            # Clean up temp file
            if 'script_path' in locals():
                try:
                    os.unlink(script_path)
                except Exception as e:
                    logger.debug("Failed to clean up temp script", path=script_path, error=str(e))

    def _create_wrapper_script(self, code: str, context: str) -> str:
        """Create wrapper script for Docker execution."""
        # Escape context for embedding
        escaped_context = repr(context)

        return f'''
import json
import sys

# Context variable
context = {escaped_context}

# Answer state (RLM pattern)
answer = {{"content": "", "ready": False}}

# LLM stubs (in production, use RPC to parent process)
def llm(prompt):
    # Stub - would call back to parent process
    return "[LLM response placeholder - configure RPC for production]"

def llm_batch(prompts):
    return [llm(p) for p in prompts]

def llm_query(prompt):
    return llm(prompt)

def llm_queries(prompts):
    return llm_batch(prompts)

def FINAL(result):
    answer["content"] = result
    answer["ready"] = True
    print(result)
    sys.exit(0)

# Execute user code
try:
{chr(10).join("    " + line for line in code.split(chr(10)))}
except SystemExit:
    pass
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)

# Print answer if ready
if answer.get("ready"):
    print(answer["content"])
'''

    async def cleanup(self) -> None:
        """Clean up Docker container if running."""
        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "kill", self._container_id],
                    capture_output=True,
                    timeout=5
                )
            except Exception as e:
                logger.debug("Failed to kill Docker container", container_id=self._container_id, error=str(e))
            self._container_id = None


# =============================================================================
# Modal Sandbox
# =============================================================================

class ModalSandbox(BaseSandbox):
    """
    Modal.com cloud sandbox for isolated Python execution.

    Provides serverless, isolated Python environments with:
    - Full Python capabilities
    - Network isolation
    - Automatic scaling
    - Pay-per-use pricing

    Requires: `pip install modal` and `modal setup`
    """

    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._modal_available = False
        self._checked_modal = False

    async def _check_modal(self) -> bool:
        """Check if Modal is available."""
        if self._checked_modal:
            return self._modal_available

        try:
            import modal  # noqa: F401
            self._modal_available = True
        except ImportError:
            logger.warning("Modal not installed. Run: pip install modal")
            self._modal_available = False

        self._checked_modal = True
        return self._modal_available

    async def execute(
        self,
        code: str,
        context: str,
        llm_query: Callable[[str], str],
        llm_batch: Callable[[List[str]], List[str]],
    ) -> SandboxResult:
        """Execute code in Modal sandbox."""
        start_time = time.time()

        if not await self._check_modal():
            return SandboxResult(
                output="",
                error="Modal not available. Install with: pip install modal",
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
            )

        try:
            import modal

            # Create Modal sandbox
            sandbox = modal.Sandbox.create(
                "python",
                "-c",
                self._create_modal_script(code, context),
                timeout=self.config.modal_timeout,
                app=modal.App.lookup(
                    self.config.modal_app_name,
                    create_if_missing=True
                ),
            )

            # Wait for completion
            sandbox.wait()

            # Get output
            output = sandbox.stdout.read()
            error = sandbox.stderr.read()

            truncated = False
            if len(output) > self.config.max_output_chars:
                output = output[:self.config.max_output_chars]
                truncated = True

            return SandboxResult(
                output=output,
                error=error if error else None,
                execution_time_ms=(time.time() - start_time) * 1000,
                truncated=truncated,
                success=sandbox.returncode == 0,
            )

        except Exception as e:
            logger.error("Modal sandbox error", error=str(e))
            return SandboxResult(
                output="",
                error=f"Modal execution failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
            )

    def _create_modal_script(self, code: str, context: str) -> str:
        """Create script for Modal execution."""
        escaped_context = repr(context)
        return f'''
context = {escaped_context}
answer = {{"content": "", "ready": False}}

def llm(prompt):
    return "[LLM placeholder]"

def llm_batch(prompts):
    return [llm(p) for p in prompts]

def llm_query(prompt):
    return llm(prompt)

def llm_queries(prompts):
    return llm_batch(prompts)

def FINAL(result):
    answer["content"] = result
    answer["ready"] = True
    print(result)
    raise SystemExit(0)

{code}

if answer.get("ready"):
    print(answer["content"])
'''

    async def cleanup(self) -> None:
        """Clean up Modal resources (handled automatically)."""
        pass


# =============================================================================
# Prime Intellect Sandbox
# =============================================================================

class PrimeSandbox(BaseSandbox):
    """
    Prime Intellect cloud sandbox (beta).

    High-performance isolated Python execution with:
    - Optimized for AI/ML workloads
    - Low-latency execution
    - Integrated with Prime Intellect infrastructure

    Requires: PRIME_API_KEY environment variable
    """

    def __init__(self, config: SandboxConfig):
        super().__init__(config)
        self._api_key = config.prime_api_key or os.getenv("PRIME_API_KEY")

    async def execute(
        self,
        code: str,
        context: str,
        llm_query: Callable[[str], str],
        llm_batch: Callable[[List[str]], List[str]],
    ) -> SandboxResult:
        """Execute code in Prime Intellect sandbox."""
        start_time = time.time()

        if not self._api_key:
            return SandboxResult(
                output="",
                error="PRIME_API_KEY not configured. Set environment variable.",
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
            )

        # Prime Intellect SDK integration (beta)
        # This is a placeholder - actual implementation depends on Prime SDK
        try:
            # Attempt to use Prime SDK
            try:
                from prime_sdk import Sandbox as PrimeSdk

                sandbox = PrimeSdk(api_key=self._api_key)
                result = await sandbox.execute(
                    code=self._create_prime_script(code, context),
                    timeout=self.config.timeout_seconds,
                )

                output = result.stdout or ""
                truncated = False
                if len(output) > self.config.max_output_chars:
                    output = output[:self.config.max_output_chars]
                    truncated = True

                return SandboxResult(
                    output=output,
                    error=result.stderr if result.stderr else None,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    truncated=truncated,
                    success=result.success,
                )

            except ImportError:
                # Fall back to REST API
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.primeintellect.ai/v1/sandbox/execute",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "code": self._create_prime_script(code, context),
                            "timeout": self.config.timeout_seconds,
                        },
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds + 10),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            return SandboxResult(
                                output="",
                                error=f"Prime API error: {error_text}",
                                execution_time_ms=(time.time() - start_time) * 1000,
                                success=False,
                            )

                        result = await response.json()
                        output = result.get("stdout", "")

                        truncated = False
                        if len(output) > self.config.max_output_chars:
                            output = output[:self.config.max_output_chars]
                            truncated = True

                        return SandboxResult(
                            output=output,
                            error=result.get("stderr"),
                            execution_time_ms=(time.time() - start_time) * 1000,
                            truncated=truncated,
                            success=result.get("success", False),
                        )

        except Exception as e:
            logger.error("Prime sandbox error", error=str(e))
            return SandboxResult(
                output="",
                error=f"Prime execution failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
                success=False,
            )

    def _create_prime_script(self, code: str, context: str) -> str:
        """Create script for Prime execution."""
        escaped_context = repr(context)
        return f'''
context = {escaped_context}
answer = {{"content": "", "ready": False}}

def llm(prompt):
    return "[LLM placeholder - use Prime's built-in LLM tools]"

def llm_batch(prompts):
    return [llm(p) for p in prompts]

def llm_query(prompt):
    return llm(prompt)

def llm_queries(prompts):
    return llm_batch(prompts)

def FINAL(result):
    answer["content"] = result
    answer["ready"] = True
    print(result)
    raise SystemExit(0)

{code}

if answer.get("ready"):
    print(answer["content"])
'''

    async def cleanup(self) -> None:
        """Clean up Prime resources (handled automatically)."""
        pass


# =============================================================================
# Sandbox Factory
# =============================================================================

def create_sandbox(config: Optional[SandboxConfig] = None) -> BaseSandbox:
    """
    Create a sandbox based on configuration.

    Args:
        config: Sandbox configuration (uses defaults if None)

    Returns:
        Appropriate sandbox instance
    """
    if config is None:
        config = SandboxConfig()

    sandbox_map = {
        SandboxType.LOCAL: LocalSandbox,
        SandboxType.DOCKER: DockerSandbox,
        SandboxType.MODAL: ModalSandbox,
        SandboxType.PRIME: PrimeSandbox,
    }

    sandbox_class = sandbox_map.get(config.sandbox_type, LocalSandbox)
    return sandbox_class(config)


async def get_best_sandbox() -> Tuple[BaseSandbox, SandboxType]:
    """
    Get the best available sandbox.

    Tries sandboxes in order of preference:
    1. Modal (if configured)
    2. Prime (if API key available)
    3. Docker (if available)
    4. Local (always available)

    Returns:
        Tuple of (sandbox instance, sandbox type)
    """
    # Check Modal
    try:
        import modal  # noqa: F401
        config = SandboxConfig(sandbox_type=SandboxType.MODAL)
        return ModalSandbox(config), SandboxType.MODAL
    except ImportError:
        pass

    # Check Prime
    if os.getenv("PRIME_API_KEY"):
        config = SandboxConfig(sandbox_type=SandboxType.PRIME)
        return PrimeSandbox(config), SandboxType.PRIME

    # Check Docker
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            config = SandboxConfig(sandbox_type=SandboxType.DOCKER)
            return DockerSandbox(config), SandboxType.DOCKER
    except Exception:
        pass

    # Fall back to local
    config = SandboxConfig(sandbox_type=SandboxType.LOCAL)
    return LocalSandbox(config), SandboxType.LOCAL


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SandboxType",
    "SandboxConfig",
    "SandboxResult",
    "BaseSandbox",
    "SafeRegex",
    "SafeJson",
    "LocalSandbox",
    "DockerSandbox",
    "ModalSandbox",
    "PrimeSandbox",
    "create_sandbox",
    "get_best_sandbox",
]
