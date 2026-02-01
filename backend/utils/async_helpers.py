"""
Async helper utilities for safe background task execution.

This module provides utilities to handle async tasks safely,
ensuring errors are logged and not silently lost.

Inspired by OpenClaw/Moltbot patterns.
"""

import asyncio
from typing import Any, Callable, Coroutine, Optional, TypeVar
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def create_safe_task(
    coro: Coroutine[Any, Any, T],
    name: Optional[str] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    on_success: Optional[Callable[[T], None]] = None,
) -> asyncio.Task[T]:
    """
    Create an asyncio task with automatic error logging.

    Unlike plain asyncio.create_task(), this wrapper ensures:
    - Errors are logged with full context
    - Optional callbacks for error/success handling
    - Task names for easier debugging

    Args:
        coro: The coroutine to run
        name: Optional task name for logging
        on_error: Optional callback for error handling
        on_success: Optional callback for success handling

    Returns:
        The created task

    Example:
        >>> create_safe_task(
        ...     process_document(doc_id),
        ...     name=f"process_doc_{doc_id}",
        ...     on_error=lambda e: mark_doc_failed(doc_id, str(e))
        ... )
    """
    task = asyncio.create_task(coro, name=name)

    def handle_result(t: asyncio.Task) -> None:
        try:
            exc = t.exception()
            if exc:
                logger.error(
                    "Background task failed",
                    task_name=name or "unnamed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                if on_error:
                    try:
                        on_error(exc)
                    except Exception as callback_error:
                        logger.error(
                            "Error callback failed",
                            task_name=name,
                            callback_error=str(callback_error),
                        )
            else:
                # Success case
                if on_success:
                    try:
                        result = t.result()
                        on_success(result)
                    except Exception as callback_error:
                        logger.error(
                            "Success callback failed",
                            task_name=name,
                            callback_error=str(callback_error),
                        )
        except asyncio.CancelledError:
            logger.debug("Background task cancelled", task_name=name)
        except asyncio.InvalidStateError:
            pass  # Task not done yet

    task.add_done_callback(handle_result)
    return task


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float,
    default: Optional[T] = None,
    task_name: Optional[str] = None,
) -> Optional[T]:
    """
    Run a coroutine with a timeout, returning default on timeout.

    This is cross-platform compatible (works on Windows, Mac, Linux)
    unlike signal-based timeouts.

    Args:
        coro: The coroutine to run
        timeout_seconds: Maximum time to wait
        default: Value to return on timeout
        task_name: Optional name for logging

    Returns:
        The coroutine result or default on timeout

    Example:
        >>> result = await run_with_timeout(
        ...     fetch_embeddings(texts),
        ...     timeout_seconds=30,
        ...     default=[],
        ...     task_name="embedding_fetch"
        ... )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "Coroutine timed out",
            task_name=task_name,
            timeout_seconds=timeout_seconds,
        )
        return default


async def run_with_concurrency(
    tasks: list[Coroutine[Any, Any, T]],
    max_concurrent: int = 4,
    task_name: Optional[str] = None,
) -> list[T]:
    """
    Run multiple coroutines with concurrency control.

    Uses a semaphore to limit parallel execution, preventing
    resource exhaustion.

    Inspired by OpenClaw's runWithConcurrency pattern.

    Args:
        tasks: List of coroutines to run
        max_concurrent: Maximum concurrent executions
        task_name: Optional name prefix for logging

    Returns:
        List of results in the same order as input tasks

    Example:
        >>> results = await run_with_concurrency(
        ...     [embed_chunk(c) for c in chunks],
        ...     max_concurrent=4,
        ...     task_name="embedding"
        ... )
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(index: int, task: Coroutine[Any, Any, T]) -> tuple[int, T]:
        async with semaphore:
            try:
                result = await task
                return (index, result)
            except Exception as e:
                logger.error(
                    "Concurrent task failed",
                    task_name=f"{task_name}_{index}" if task_name else f"task_{index}",
                    error=str(e),
                )
                raise

    # Preserve order by tracking indices
    indexed_results = await asyncio.gather(
        *[limited(i, t) for i, t in enumerate(tasks)],
        return_exceptions=True,
    )

    # Sort by index and extract results
    results = []
    for item in sorted(indexed_results, key=lambda x: x[0] if isinstance(x, tuple) else -1):
        if isinstance(item, tuple):
            results.append(item[1])
        elif isinstance(item, Exception):
            raise item
        else:
            results.append(item)

    return results


class DebouncedTask:
    """
    A debounced async task that delays execution until a quiet period.

    Inspired by OpenClaw's dirty tracking with 1.5s debounce.

    Example:
        >>> reindex_task = DebouncedTask(
        ...     reindex_documents,
        ...     delay_seconds=1.5,
        ...     name="reindex"
        ... )
        >>> await reindex_task.trigger()  # Starts timer
        >>> await reindex_task.trigger()  # Resets timer
        >>> # After 1.5s of no triggers, reindex_documents() runs
    """

    def __init__(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        delay_seconds: float = 1.5,
        name: Optional[str] = None,
    ):
        self.coro_factory = coro_factory
        self.delay_seconds = delay_seconds
        self.name = name
        self._timer_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def trigger(self) -> None:
        """Trigger the debounced task, resetting the timer."""
        async with self._lock:
            # Cancel existing timer
            if self._timer_task and not self._timer_task.done():
                self._timer_task.cancel()
                try:
                    await self._timer_task
                except asyncio.CancelledError:
                    pass

            # Start new timer
            self._timer_task = create_safe_task(
                self._delayed_execute(),
                name=f"debounce_{self.name}" if self.name else "debounce",
            )

    async def _delayed_execute(self) -> None:
        """Wait for delay then execute."""
        await asyncio.sleep(self.delay_seconds)
        try:
            await self.coro_factory()
            logger.debug("Debounced task executed", name=self.name)
        except Exception as e:
            logger.error("Debounced task failed", name=self.name, error=str(e))
            raise

    async def cancel(self) -> None:
        """Cancel any pending execution."""
        async with self._lock:
            if self._timer_task and not self._timer_task.done():
                self._timer_task.cancel()
                try:
                    await self._timer_task
                except asyncio.CancelledError:
                    pass


async def retry_with_backoff(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    task_name: Optional[str] = None,
) -> T:
    """
    Retry a coroutine with exponential backoff.

    Useful for handling transient failures like rate limits
    or network issues.

    Args:
        coro_factory: Factory function that creates the coroutine
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay cap
        exponential: Use exponential backoff if True
        task_name: Optional name for logging

    Returns:
        The coroutine result

    Raises:
        The last exception if all retries fail

    Example:
        >>> result = await retry_with_backoff(
        ...     lambda: embed_with_api(texts),
        ...     max_retries=3,
        ...     task_name="embedding"
        ... )
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(
                    "All retry attempts failed",
                    task_name=task_name,
                    attempts=max_retries + 1,
                    error=str(e),
                )
                raise

            delay = base_delay * (2**attempt if exponential else 1)
            delay = min(delay, max_delay)

            logger.warning(
                "Retry attempt failed, backing off",
                task_name=task_name,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay_seconds=delay,
                error=str(e),
            )
            await asyncio.sleep(delay)

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry_with_backoff")
