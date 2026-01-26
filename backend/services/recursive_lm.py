"""
AIDocumentIndexer - Recursive Language Model Service (Enhanced)
================================================================

Implements Recursive Language Models (RLM) for processing 10M+ token contexts.

Based on official RLM library patterns (https://github.com/alexzhang13/rlm)
and MIT/Prime Intellect research (arXiv:2512.24601).

RLM Architecture:
- Context stored as Python variable, not in prompt
- Model accesses context programmatically (slice, search, partition)
- Recursive calls process portions with sub-LLM
- O(log N) complexity for sparse retrieval operations
- Flat scaling to 10M+ tokens (vs quadratic for standard attention)

Key Features (Phase 36 Enhancement):
- Multiple sandbox backends (Local, Docker, Modal, Prime)
- Official RLM library integration when available
- Answer state pattern ({content, ready}) for diffusion-style output
- Sub-LLM parallelization with llm_batch
- Trajectory logging for debugging
- 2.5x accuracy improvement over baseline

Performance (from RLM paper):
- 62% accuracy on CodeQA vs 24% GPT-5 baseline
- 91.33% accuracy on BrowseComp-Plus (6-11M tokens)
- Efficient token usage through programmatic context access

Security:
- RestrictedPython for local execution
- Docker containers for isolation
- Modal/Prime cloud sandboxes for production

Research:
- "Recursive Language Models" (MIT CSAIL, 2025)
- Prime Intellect RLM implementation
- arXiv:2512.24601
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import structlog

from backend.core.config import settings
from backend.core.performance import LRUCache
from backend.services.rlm_sandbox import (
    SandboxType,
    SandboxConfig,
    SandboxResult,
    BaseSandbox,
    SafeRegex,
    SafeJson,
    create_sandbox,
    get_best_sandbox,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ExecutionMode(str, Enum):
    """Execution modes for RLM code."""
    RESTRICTED = "restricted"  # RestrictedPython (safe, limited)
    SANDBOX = "sandbox"        # E2B sandbox (full Python, isolated)
    LOCAL = "local"           # Local execution (development only)
    DOCKER = "docker"         # Docker container
    MODAL = "modal"           # Modal.com sandbox
    PRIME = "prime"           # Prime Intellect sandbox
    AUTO = "auto"             # Auto-detect best available


@dataclass
class RLMConfig:
    """Configuration for Recursive Language Model."""
    # Models
    root_model: str = "gpt-4o"              # Main reasoning model
    recursive_model: str = "gpt-4o-mini"    # For recursive calls (cheaper)
    root_provider: str = "openai"
    recursive_provider: str = "openai"

    # Execution limits
    max_depth: int = 5                      # Max recursion depth
    max_iterations: int = 20                # Max REPL iterations
    max_code_length: int = 5000             # Max generated code length
    timeout_seconds: float = 120.0          # Overall timeout

    # Context handling
    max_preview_chars: int = 2000           # Initial context preview
    chunk_size: int = 5000                  # Default chunk size for splitting
    chunk_overlap: int = 500                # Overlap between chunks
    max_output_chars: int = 8192            # Max REPL output (RLM default)

    # Execution mode
    execution_mode: ExecutionMode = ExecutionMode.AUTO

    # Sandbox settings
    sandbox_config: Optional[SandboxConfig] = None

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Logging/debugging
    verbose: bool = False
    log_trajectory: bool = False
    trajectory_log_dir: str = "./logs/rlm"


@dataclass(slots=True)
class RLMResult:
    """Result from RLM processing."""
    answer: str
    reasoning_steps: List[str] = field(default_factory=list)
    code_executed: List[str] = field(default_factory=list)
    intermediate_results: List[str] = field(default_factory=list)
    iterations: int = 0
    depth: int = 0
    tokens_processed: int = 0
    execution_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    sandbox_type: Optional[str] = None
    trajectory_id: Optional[str] = None


@dataclass
class ExecutionContext:
    """Context for code execution in RLM."""
    context: str                            # The full document context
    query: str                              # Original query
    results: List[Any] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    iteration: int = 0


@dataclass
class TrajectoryStep:
    """Single step in RLM execution trajectory."""
    iteration: int
    code: str
    output: str
    error: Optional[str] = None
    answer_state: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# RLM Prompts (Updated for Official RLM Pattern)
# =============================================================================

SYSTEM_PROMPT = """You are an AI assistant that processes very long documents by writing Python code.

You have access to:
- `context`: A variable containing the full document text
- `answer`: A dictionary with "content" (your response) and "ready" (set True when done)
- `llm(prompt)`: Call a sub-LLM with a single prompt
- `llm_batch(prompts)`: Call sub-LLMs in parallel with multiple prompts
- `print()`: Output debugging information

Your task is to answer the user's query by programmatically exploring the context.

## Code Patterns

### 1. Preview Context
```python
# See the beginning of the document
print(context[:2000])
```

### 2. Search for Keywords
```python
import re
# Find relevant sections
matches = list(re.finditer(r'revenue|earnings|profit', context, re.I))
for m in matches[:5]:
    start = max(0, m.start() - 200)
    end = min(len(context), m.end() + 200)
    print(f"Found at {m.start()}: {context[start:end]}")
```

### 3. Split and Process Chunks
```python
# Process in chunks
chunk_size = 5000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size-500)]

# Analyze each chunk (parallel)
prompts = [f"Extract key information from: {c[:1000]}..." for c in chunks[:10]]
results = llm_batch(prompts)

# Synthesize
synthesis = llm(f"Combine these findings: {results}")
answer["content"] = synthesis
answer["ready"] = True
```

### 4. Targeted Extraction
```python
# Find specific section
idx = context.lower().find("financial summary")
if idx >= 0:
    section = context[idx:idx+3000]
    result = llm(f"Extract the financial figures from: {section}")
    answer["content"] = result
    answer["ready"] = True
else:
    print("Section not found, trying alternative...")
```

## Rules
1. Set `answer["ready"] = True` when you have the complete answer
2. Use `llm()` for complex reasoning on context portions
3. Use `llm_batch()` for parallel processing of multiple chunks
4. Keep intermediate outputs concise (max 8192 chars displayed)
5. Handle missing/not found cases gracefully

## Alternative: FINAL() Function
For backward compatibility, you can also use:
```python
FINAL("Your answer here")  # Sets answer and stops execution
```
"""

USER_PROMPT_TEMPLATE = """Query: {query}

The document context is stored in the `context` variable ({context_length:,} characters, ~{token_estimate:,} tokens).

Preview of the document:
```
{context_preview}
```

Write Python code to answer the query. Set `answer["ready"] = True` when done."""


# =============================================================================
# Trajectory Logger
# =============================================================================

class TrajectoryLogger:
    """Logs RLM execution trajectories for debugging and visualization."""

    def __init__(self, log_dir: str = "./logs/rlm"):
        self.log_dir = log_dir
        self.steps: List[TrajectoryStep] = []
        self.trajectory_id = hashlib.md5(
            f"{time.time()}".encode()
        ).hexdigest()[:8]

    def log_step(
        self,
        iteration: int,
        code: str,
        output: str,
        error: Optional[str] = None,
        answer_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single execution step."""
        step = TrajectoryStep(
            iteration=iteration,
            code=code,
            output=output,
            error=error,
            answer_state=answer_state,
        )
        self.steps.append(step)

    def save(self, query: str, result: RLMResult) -> str:
        """Save trajectory to file."""
        import os
        os.makedirs(self.log_dir, exist_ok=True)

        trajectory = {
            "id": self.trajectory_id,
            "query": query,
            "result": {
                "answer": result.answer,
                "success": result.success,
                "error": result.error,
                "iterations": result.iterations,
                "execution_time_ms": result.execution_time_ms,
            },
            "steps": [
                {
                    "iteration": s.iteration,
                    "code": s.code,
                    "output": s.output,
                    "error": s.error,
                    "answer_state": s.answer_state,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
        }

        filepath = f"{self.log_dir}/trajectory_{self.trajectory_id}.json"
        with open(filepath, 'w') as f:
            json.dump(trajectory, f, indent=2)

        logger.debug("Trajectory saved", path=filepath)
        return filepath


# =============================================================================
# Code Executor (Legacy, for backward compatibility)
# =============================================================================

class CodeExecutor:
    """Safe code execution for RLM (legacy interface)."""

    def __init__(self, config: RLMConfig):
        self.config = config
        self._final_answer: Optional[str] = None
        self._execution_log: List[str] = []

    def execute(
        self,
        code: str,
        context: str,
        llm_query_fn: Callable[[str], str],
        llm_queries_fn: Callable[[List[str]], List[str]],
    ) -> Tuple[Optional[str], List[str], Optional[str]]:
        """
        Execute code safely (legacy interface).

        Returns:
            Tuple of (final_answer, execution_log, error)
        """
        self._final_answer = None
        self._execution_log = []
        answer_state = {"content": "", "ready": False}

        def final_handler(answer: str):
            self._final_answer = answer
            answer_state["content"] = answer
            answer_state["ready"] = True
            raise StopIteration(answer)

        def capture_print(*args, **kwargs):
            output = " ".join(str(arg) for arg in args)
            self._execution_log.append(output)

        # Create execution environment
        safe_globals = self._create_safe_globals(
            context=context,
            llm=llm_query_fn,
            llm_batch=llm_queries_fn,
            llm_query=llm_query_fn,
            llm_queries=llm_queries_fn,
            final_fn=final_handler,
            answer_state=answer_state,
            print_fn=capture_print,
        )

        try:
            if len(code) > self.config.max_code_length:
                return None, [], f"Code too long ({len(code)} > {self.config.max_code_length})"

            # Execute with timeout to prevent infinite loops blocking the process
            import concurrent.futures
            timeout = getattr(self.config, 'timeout_seconds', 120)

            def _run_code():
                exec(code, safe_globals)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_code)
                try:
                    future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.warning("Code execution timed out", timeout=timeout)
                    return None, self._execution_log, f"Execution timed out after {timeout}s"

            # Check answer state
            if answer_state.get("ready"):
                self._final_answer = answer_state.get("content", "")

            return self._final_answer, self._execution_log, None

        except StopIteration:
            return self._final_answer, self._execution_log, None

        except Exception as e:
            logger.warning("Code execution error", error=str(e))
            return None, self._execution_log, str(e)

    def _create_safe_globals(
        self,
        context: str,
        llm: Callable,
        llm_batch: Callable,
        llm_query: Callable,
        llm_queries: Callable,
        final_fn: Callable,
        answer_state: Dict,
        print_fn: Callable,
    ) -> Dict[str, Any]:
        """Create restricted globals for safe execution."""
        safe_builtins = {
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

        return {
            '__builtins__': safe_builtins,
            'context': context,
            'answer': answer_state,
            'llm': llm,
            'llm_batch': llm_batch,
            'llm_query': llm_query,
            'llm_queries': llm_queries,
            'FINAL': final_fn,
            'print': print_fn,
            're': SafeRegex(),
            'json': SafeJson(),
        }


# =============================================================================
# Recursive Language Model Service
# =============================================================================

class RecursiveLMService:
    """
    Recursive Language Model for processing unlimited context.

    Implements the official RLM pattern:
    1. Store context as external variable
    2. Let model write code to access context
    3. Execute code in sandboxed environment
    4. Support recursive LLM calls for complex reasoning
    5. Use answer state pattern for diffusion-style output

    Features:
    - Multiple sandbox backends (Local, Docker, Modal, Prime)
    - Parallel sub-LLM calls via llm_batch
    - Trajectory logging for debugging
    - Automatic sandbox detection

    Usage:
        service = RecursiveLMService()
        result = await service.process(
            query="What is Apple's revenue?",
            context=very_long_document,  # Can be 10M+ characters
        )
    """

    def __init__(self, config: Optional[RLMConfig] = None):
        self.config = config or RLMConfig()
        self._root_llm = None
        self._recursive_llm = None
        self._sandbox: Optional[BaseSandbox] = None
        self._sandbox_type: Optional[SandboxType] = None
        self._executor = CodeExecutor(self.config)  # Legacy fallback
        self._cache = LRUCache[RLMResult](capacity=100)
        self._initialized = False
        # Shared thread pool for sync-to-async bridging (avoid per-call overhead)
        import concurrent.futures
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Check for official RLM library
        self._official_rlm_available = False
        try:
            import rlm
            self._official_rlm_available = True
            logger.info("Official RLM library detected")
        except ImportError:
            pass

    async def initialize(self) -> bool:
        """Initialize LLM clients and sandbox."""
        if self._initialized:
            return True

        try:
            from backend.services.llm import LLMFactory

            # Initialize root model (for main reasoning)
            self._root_llm = LLMFactory.get_chat_model(
                provider=self.config.root_provider,
                model=self.config.root_model,
                temperature=0.0,
                max_tokens=4096,
            )

            # Initialize recursive model (for chunk processing)
            self._recursive_llm = LLMFactory.get_chat_model(
                provider=self.config.recursive_provider,
                model=self.config.recursive_model,
                temperature=0.0,
                max_tokens=2048,
            )

            # Initialize sandbox
            await self._initialize_sandbox()

            logger.info(
                "RLM service initialized",
                root_model=self.config.root_model,
                recursive_model=self.config.recursive_model,
                sandbox_type=self._sandbox_type.value if self._sandbox_type else "none",
                official_rlm=self._official_rlm_available,
            )

            self._initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize RLM service", error=str(e))
            return False

    async def _initialize_sandbox(self) -> None:
        """Initialize the execution sandbox based on configuration."""
        mode = self.config.execution_mode

        if mode == ExecutionMode.AUTO:
            # Auto-detect best available sandbox
            self._sandbox, self._sandbox_type = await get_best_sandbox()
        else:
            # Map execution mode to sandbox type
            mode_to_sandbox = {
                ExecutionMode.LOCAL: SandboxType.LOCAL,
                ExecutionMode.RESTRICTED: SandboxType.LOCAL,
                ExecutionMode.DOCKER: SandboxType.DOCKER,
                ExecutionMode.MODAL: SandboxType.MODAL,
                ExecutionMode.PRIME: SandboxType.PRIME,
            }
            sandbox_type = mode_to_sandbox.get(mode, SandboxType.LOCAL)
            config = self.config.sandbox_config or SandboxConfig(
                sandbox_type=sandbox_type,
                timeout_seconds=self.config.timeout_seconds,
                max_output_chars=self.config.max_output_chars,
            )
            self._sandbox = create_sandbox(config)
            self._sandbox_type = sandbox_type

    async def close(self) -> None:
        """Clean up resources including thread pool."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None
        if self._sandbox:
            try:
                await self._sandbox.close()
            except Exception:
                pass
            self._sandbox = None
        self._initialized = False
        logger.debug("RecursiveLMService closed")

    async def process(
        self,
        query: str,
        context: str,
        max_iterations: Optional[int] = None,
    ) -> RLMResult:
        """
        Process a query against a long context using RLM.

        Args:
            query: The question to answer
            context: The full document context (can be 10M+ chars)
            max_iterations: Override max iterations

        Returns:
            RLMResult with answer and execution details
        """
        if not await self.initialize():
            return RLMResult(
                answer="",
                success=False,
                error="RLM service not initialized",
            )

        start_time = time.time()
        max_iter = max_iterations or self.config.max_iterations

        # Check cache
        if self.config.enable_cache:
            cache_key = self._cache_key(query, context)
            cached = await self._cache.get(cache_key)
            if cached:
                logger.debug("RLM cache hit")
                return cached

        logger.info(
            "Starting RLM processing",
            query_length=len(query),
            context_length=len(context),
            estimated_tokens=len(context) // 4,
            sandbox=self._sandbox_type.value if self._sandbox_type else "none",
        )

        # Create trajectory logger if enabled
        trajectory_logger = None
        if self.config.log_trajectory:
            trajectory_logger = TrajectoryLogger(self.config.trajectory_log_dir)

        # Try official RLM library first
        if self._official_rlm_available:
            result = await self._process_with_official_rlm(
                query, context, max_iter, trajectory_logger
            )
            if result.success:
                return result
            # Fall through to custom implementation

        # Use custom implementation
        result = await self._process_custom(
            query, context, max_iter, trajectory_logger
        )

        # Save trajectory
        if trajectory_logger:
            trajectory_logger.save(query, result)
            result.trajectory_id = trajectory_logger.trajectory_id

        # Cache result
        if self.config.enable_cache and result.success:
            cache_key = self._cache_key(query, context)
            await self._cache.set(cache_key, result)

        return result

    async def _process_with_official_rlm(
        self,
        query: str,
        context: str,
        max_iterations: int,
        trajectory_logger: Optional[TrajectoryLogger],
    ) -> RLMResult:
        """Process using the official RLM library."""
        try:
            import rlm as rlm_lib

            start_time = time.time()

            # Map our config to RLM library config
            backend_kwargs = {"model_name": self.config.root_model}

            # Determine environment
            env_map = {
                SandboxType.LOCAL: "local",
                SandboxType.DOCKER: "docker",
                SandboxType.MODAL: "modal",
                SandboxType.PRIME: "prime",
            }
            environment = env_map.get(self._sandbox_type, "local")

            # Create RLM instance
            rlm_instance = rlm_lib.RLM(
                backend=self.config.root_provider,
                backend_kwargs=backend_kwargs,
                environment=environment,
                verbose=self.config.verbose,
            )

            # Construct prompt with context
            full_prompt = f"""Context (stored in variable 'context'):
{context[:self.config.max_preview_chars]}{"..." if len(context) > self.config.max_preview_chars else ""}

Query: {query}

Note: The full context ({len(context):,} chars) is available in the 'context' variable."""

            # Execute
            result = rlm_instance.completion(full_prompt)

            execution_time = (time.time() - start_time) * 1000

            return RLMResult(
                answer=result.response if hasattr(result, 'response') else str(result),
                iterations=getattr(result, 'iterations', 1),
                tokens_processed=len(context) // 4,
                execution_time_ms=execution_time,
                success=True,
                sandbox_type=environment,
            )

        except Exception as e:
            logger.warning(
                "Official RLM failed, falling back to custom",
                error=str(e)
            )
            return RLMResult(
                answer="",
                success=False,
                error=f"Official RLM error: {str(e)}",
            )

    async def _process_custom(
        self,
        query: str,
        context: str,
        max_iterations: int,
        trajectory_logger: Optional[TrajectoryLogger],
    ) -> RLMResult:
        """Process using custom RLM implementation."""
        start_time = time.time()

        # Create execution context
        exec_context = ExecutionContext(
            context=context,
            query=query,
        )

        # Prepare initial prompt
        context_preview = context[:self.config.max_preview_chars]
        if len(context) > self.config.max_preview_chars:
            context_preview += "\n... (truncated)"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=query,
            context_length=len(context),
            token_estimate=len(context) // 4,
            context_preview=context_preview,
        )

        reasoning_steps = []
        code_executed = []
        intermediate_results = []
        final_answer = None
        error = None

        try:
            for iteration in range(max_iterations):
                exec_context.iteration = iteration

                # Generate code from LLM
                code = await self._generate_code(user_prompt, reasoning_steps)

                if not code:
                    error = "No code generated"
                    break

                code_executed.append(code)
                reasoning_steps.append(f"Iteration {iteration + 1}: Generated code")

                # Execute code
                sandbox_result = await self._execute_code_sandbox(
                    code, exec_context
                )

                # Log trajectory
                if trajectory_logger:
                    trajectory_logger.log_step(
                        iteration=iteration,
                        code=code,
                        output=sandbox_result.output,
                        error=sandbox_result.error,
                        answer_state=sandbox_result.variables.get("answer"),
                    )

                intermediate_results.append(sandbox_result.output)

                if sandbox_result.error:
                    reasoning_steps.append(f"Execution error: {sandbox_result.error}")
                    user_prompt = f"Previous code had an error: {sandbox_result.error}\n\nPlease fix and try again.\n\n{user_prompt}"
                    continue

                # Check answer state
                answer_state = sandbox_result.variables.get("answer", {})
                if answer_state.get("ready"):
                    final_answer = answer_state.get("content", "")
                    reasoning_steps.append(f"Answer ready in iteration {iteration + 1}")
                    break

                # Check for FINAL output in stdout
                if sandbox_result.output and sandbox_result.success:
                    # If output looks like a complete answer, use it
                    if len(sandbox_result.output) > 50:
                        final_answer = sandbox_result.output
                        reasoning_steps.append(f"Answer from output in iteration {iteration + 1}")
                        break

                # Continue with refined prompt
                user_prompt = f"Previous iteration output: {sandbox_result.output[:1000]}\n\nContinue processing.\n\n{user_prompt}"

        except asyncio.TimeoutError:
            error = f"Processing timed out after {self.config.timeout_seconds}s"
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            logger.exception("RLM processing error")

        execution_time = (time.time() - start_time) * 1000

        result = RLMResult(
            answer=final_answer or "",
            reasoning_steps=reasoning_steps,
            code_executed=code_executed,
            intermediate_results=intermediate_results,
            iterations=exec_context.iteration + 1,
            depth=exec_context.depth,
            tokens_processed=len(context) // 4,
            execution_time_ms=execution_time,
            success=final_answer is not None,
            error=error,
            sandbox_type=self._sandbox_type.value if self._sandbox_type else None,
        )

        logger.info(
            "RLM processing complete",
            success=result.success,
            iterations=result.iterations,
            execution_time_ms=round(execution_time, 2),
        )

        return result

    async def _generate_code(
        self,
        user_prompt: str,
        previous_steps: List[str],
    ) -> Optional[str]:
        """Generate Python code to process the query."""
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await self._root_llm.ainvoke(messages)
            content = response.content

            # Extract code from response
            code = self._extract_code(content)
            return code

        except Exception as e:
            logger.error("Code generation failed", error=str(e))
            return None

    def _extract_code(self, content: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find code blocks
        code_pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(code_pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, look for code-like content
        lines = content.strip().split('\n')
        code_lines = []
        in_code = False

        code_starters = (
            'import ', 'from ', 'def ', 'class ', '#',
            'context', 'preview', 'chunks', 'result',
            'FINAL', 'llm', 'answer', 'if ', 'for ', 'while ',
            'try:', 'with '
        )

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(code_starters):
                in_code = True
            if in_code:
                code_lines.append(line)

        return '\n'.join(code_lines) if code_lines else content

    async def _execute_code_sandbox(
        self,
        code: str,
        exec_context: ExecutionContext,
    ) -> SandboxResult:
        """Execute code in sandbox and return result."""
        # Create LLM functions for sandbox
        async def async_llm_query(prompt: str) -> str:
            from langchain_core.messages import HumanMessage
            response = await self._recursive_llm.ainvoke([HumanMessage(content=prompt)])
            return response.content

        async def async_llm_batch(prompts: List[str]) -> List[str]:
            from langchain_core.messages import HumanMessage
            tasks = [
                self._recursive_llm.ainvoke([HumanMessage(content=p)])
                for p in prompts
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                r.content if hasattr(r, 'content') else str(r)
                for r in responses
            ]

        # Wrap for sync execution in sandbox using shared thread pool
        thread_pool = self._thread_pool

        def sync_llm_query(prompt: str) -> str:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, run directly
                return asyncio.run(async_llm_query(prompt))
            # Use shared thread pool to run async in separate thread
            future = thread_pool.submit(asyncio.run, async_llm_query(prompt))
            return future.result(timeout=60)

        def sync_llm_batch(prompts: List[str]) -> List[str]:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(async_llm_batch(prompts))
            future = thread_pool.submit(asyncio.run, async_llm_batch(prompts))
            return future.result(timeout=120)

        # Execute in sandbox
        if self._sandbox:
            try:
                result = await asyncio.wait_for(
                    self._sandbox.execute(
                        code=code,
                        context=exec_context.context,
                        llm_query=sync_llm_query,
                        llm_batch=sync_llm_batch,
                    ),
                    timeout=self.config.timeout_seconds,
                )
                return result
            except asyncio.TimeoutError:
                return SandboxResult(
                    output="",
                    error="Sandbox execution timed out",
                    success=False,
                )
            except Exception as e:
                logger.warning("Sandbox execution failed", error=str(e))
                # Fall through to legacy executor

        # Legacy fallback
        loop = asyncio.get_running_loop()
        try:
            answer, log, error = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._executor.execute,
                    code,
                    exec_context.context,
                    sync_llm_query,
                    sync_llm_batch,
                ),
                timeout=self.config.timeout_seconds,
            )

            return SandboxResult(
                output="\n".join(log),
                error=error,
                success=error is None or answer is not None,
                variables={"answer": {"content": answer or "", "ready": answer is not None}},
            )
        except asyncio.TimeoutError:
            return SandboxResult(
                output="",
                error="Execution timed out",
                success=False,
            )

    def _cache_key(self, query: str, context: str) -> str:
        """Generate cache key for query + context."""
        context_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"rlm:{query_hash}:{context_hash}"

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._sandbox:
            await self._sandbox.cleanup()


# =============================================================================
# Convenience Functions
# =============================================================================

_rlm_service: Optional[RecursiveLMService] = None
_service_lock = asyncio.Lock()


async def get_rlm_service(
    config: Optional[RLMConfig] = None,
) -> RecursiveLMService:
    """Get or create RLM service singleton."""
    global _rlm_service

    if _rlm_service is not None:
        return _rlm_service

    async with _service_lock:
        if _rlm_service is not None:
            return _rlm_service

        _rlm_service = RecursiveLMService(config)
        return _rlm_service


async def process_long_context(
    query: str,
    context: str,
    config: Optional[RLMConfig] = None,
) -> RLMResult:
    """
    Process a query against a long context.

    Convenience function for one-off RLM processing.

    Args:
        query: The question to answer
        context: The document context (can be very long)
        config: Optional RLM configuration

    Returns:
        RLMResult with answer and execution details
    """
    service = await get_rlm_service(config)
    return await service.process(query, context)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RLMConfig",
    "RLMResult",
    "ExecutionMode",
    "ExecutionContext",
    "TrajectoryStep",
    "TrajectoryLogger",
    "RecursiveLMService",
    "CodeExecutor",
    "get_rlm_service",
    "process_long_context",
]
