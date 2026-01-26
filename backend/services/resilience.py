"""
AIDocumentIndexer - Resilience Patterns
=======================================

Phase 69: Circuit breaker and retry patterns for service resilience.

Implements:
- Circuit Breaker: Prevents cascading failures by stopping requests to failing services
- Retry with Exponential Backoff: Retries transient failures with increasing delays
- Timeout handling: Prevents hanging on slow services

Usage:
    from backend.services.resilience import circuit_breaker, retry_with_backoff

    # Circuit breaker for API calls
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def call_external_api():
        ...

    # Retry with exponential backoff
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def call_with_retry():
        ...
"""

import asyncio
import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation, requests pass through
    OPEN = "open"           # Failing, requests are blocked
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Failures before opening circuit
    recovery_timeout: float = 30.0   # Seconds before trying again
    success_threshold: int = 2       # Successes in half-open to close
    timeout: float = 10.0            # Request timeout in seconds

    # Exceptions that count as failures (empty = all exceptions)
    failure_exceptions: Set[type] = field(default_factory=lambda: {
        Exception,  # All exceptions by default
    })

    # Exceptions to ignore (don't count as failures)
    ignore_exceptions: Set[type] = field(default_factory=lambda: {
        KeyboardInterrupt,
        SystemExit,
    })


class CircuitBreaker:
    """
    Circuit breaker implementation for service resilience.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, all requests fail immediately
    - HALF_OPEN: Testing recovery, limited requests allowed

    Transitions:
    - CLOSED -> OPEN: When failure_threshold is reached
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: On any failure
    """

    # Global registry of circuit breakers by name
    _instances: Dict[str, "CircuitBreaker"] = {}

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

        # Register instance
        CircuitBreaker._instances[name] = self

        logger.info(
            "Circuit breaker initialized",
            name=name,
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout,
        )

    @classmethod
    def get(cls, name: str) -> Optional["CircuitBreaker"]:
        """Get circuit breaker by name."""
        return cls._instances.get(name)

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {
            name: cb.get_stats()
            for name, cb in cls._instances.items()
        }

    @property
    def state(self) -> CircuitState:
        """Get current state, handling automatic transitions."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        "Circuit breaker entering half-open state",
                        name=self.name,
                        elapsed_seconds=elapsed,
                    )
        return self._state

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
            },
        }

    async def _record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        "Circuit breaker closed (service recovered)",
                        name=self.name,
                    )

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed request."""
        async with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker reopened (failure in half-open)",
                    name=self.name,
                    error=str(error),
                )
            else:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit breaker opened (threshold reached)",
                        name=self.name,
                        failure_count=self._failure_count,
                        error=str(error),
                    )

    def _should_count_failure(self, error: Exception) -> bool:
        """Check if exception should count as a failure."""
        # Ignore certain exceptions
        for exc_type in self.config.ignore_exceptions:
            if isinstance(error, exc_type):
                return False

        # Count if in failure_exceptions
        for exc_type in self.config.failure_exceptions:
            if isinstance(error, exc_type):
                return True

        return False

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of function

        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If function fails
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit breaker '{self.name}' is open. "
                f"Service unavailable, will retry after {self.config.recovery_timeout}s"
            )

        try:
            # Apply timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout,
            )
            await self._record_success()
            return result

        except asyncio.TimeoutError as e:
            await self._record_failure(e)
            raise
        except Exception as e:
            if self._should_count_failure(e):
                await self._record_failure(e)
            raise

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info("Circuit breaker manually reset", name=self.name)


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    success_threshold: int = 2,
    timeout: float = 10.0,
) -> Callable:
    """
    Decorator to apply circuit breaker pattern to async functions.

    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before testing recovery
        success_threshold: Successes in half-open to close
        timeout: Request timeout in seconds

    Example:
        @circuit_breaker(name="openai_api", failure_threshold=3)
        async def call_openai(prompt: str):
            return await client.chat(prompt)
    """
    def decorator(func: Callable) -> Callable:
        cb_name = name or func.__name__
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
        )
        cb = CircuitBreaker(cb_name, config)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)

        # Attach circuit breaker for inspection
        wrapper.circuit_breaker = cb
        return wrapper

    return decorator


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry with backoff."""
    max_retries: int = 3
    base_delay: float = 1.0      # Initial delay in seconds
    max_delay: float = 60.0      # Maximum delay
    exponential_base: float = 2.0
    jitter: bool = True          # Add randomness to prevent thundering herd

    # Exceptions that trigger retry (empty = all exceptions)
    retry_exceptions: Set[type] = field(default_factory=lambda: {
        Exception,
    })

    # Exceptions that should NOT be retried
    no_retry_exceptions: Set[type] = field(default_factory=lambda: {
        KeyboardInterrupt,
        SystemExit,
        ValueError,  # Usually indicates bad input
        TypeError,   # Usually indicates programming error
    })


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """
    Execute async function with retry and exponential backoff.

    Args:
        func: Async function to execute
        *args, **kwargs: Arguments to pass to function
        config: Retry configuration

    Returns:
        Result of function

    Raises:
        Last exception if all retries fail
    """
    config = config or RetryConfig()
    import random

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # Check if we should retry this exception
            should_retry = False
            for exc_type in config.no_retry_exceptions:
                if isinstance(e, exc_type):
                    raise  # Don't retry

            for exc_type in config.retry_exceptions:
                if isinstance(e, exc_type):
                    should_retry = True
                    break

            if not should_retry:
                raise

            # Check if we have retries left
            if attempt >= config.max_retries:
                logger.warning(
                    "All retries exhausted",
                    function=func.__name__,
                    attempts=attempt + 1,
                    error=str(e),
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay,
            )

            # Add jitter
            if config.jitter:
                delay = delay * (0.5 + random.random())

            logger.info(
                "Retrying after failure",
                function=func.__name__,
                attempt=attempt + 1,
                max_retries=config.max_retries,
                delay_seconds=round(delay, 2),
                error=str(e),
            )

            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """
    Decorator to apply retry with exponential backoff to async functions.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add randomness to delays

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        async def call_api():
            return await client.request()
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(func, *args, config=config, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Combined Resilience Pattern
# =============================================================================

def resilient(
    name: Optional[str] = None,
    # Circuit breaker settings
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    # Retry settings
    max_retries: int = 3,
    base_delay: float = 1.0,
    # Timeout
    timeout: float = 30.0,
) -> Callable:
    """
    Combined decorator applying both circuit breaker and retry patterns.

    Order of operations:
    1. Circuit breaker checks if requests should be allowed
    2. If allowed, retry logic handles transient failures
    3. Persistent failures trigger circuit breaker

    Example:
        @resilient(name="openai", failure_threshold=3, max_retries=2)
        async def call_openai(prompt: str):
            return await client.chat(prompt)
    """
    def decorator(func: Callable) -> Callable:
        # Apply retry first (inner), then circuit breaker (outer)
        retried = with_retry(
            max_retries=max_retries,
            base_delay=base_delay,
        )(func)

        protected = circuit_breaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            timeout=timeout,
        )(retried)

        return protected

    return decorator


# =============================================================================
# Health Check Utilities
# =============================================================================

async def check_circuit_breakers() -> Dict[str, Any]:
    """
    Check health of all circuit breakers.

    Returns:
        Health status for all circuit breakers
    """
    stats = CircuitBreaker.get_all_stats()

    open_circuits = [
        name for name, s in stats.items()
        if s["state"] == CircuitState.OPEN.value
    ]

    return {
        "healthy": len(open_circuits) == 0,
        "total_circuits": len(stats),
        "open_circuits": open_circuits,
        "circuits": stats,
    }
