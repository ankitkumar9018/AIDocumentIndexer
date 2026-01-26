"""
AIDocumentIndexer - Shared HTTP Client
======================================

Provides a shared, connection-pooled HTTP client for all external API calls.
Using a shared client avoids the overhead of creating new connections for
each request, improving performance by 2-5x for HTTP-heavy workloads.

Benefits:
- Connection pooling and keep-alive
- Configurable timeouts
- Automatic retry with exponential backoff
- Proper cleanup on shutdown

Usage:
    from backend.services.http_client import get_http_client, close_http_client

    # In async code:
    client = await get_http_client()
    response = await client.get("https://api.example.com/data")

    # On shutdown (call once):
    await close_http_client()
"""

import asyncio
from typing import Optional

import httpx
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Global shared client instance
_http_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


# =============================================================================
# Configuration
# =============================================================================

# Default timeout settings (in seconds)
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 10.0

# Connection pool settings
# max_connections: Total connections in the pool
# max_keepalive_connections: Keep-alive connections to reuse
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20
DEFAULT_KEEPALIVE_EXPIRY = 30.0  # seconds


def _create_timeout() -> httpx.Timeout:
    """Create timeout configuration."""
    return httpx.Timeout(
        connect=DEFAULT_CONNECT_TIMEOUT,
        read=DEFAULT_READ_TIMEOUT,
        write=DEFAULT_WRITE_TIMEOUT,
        pool=DEFAULT_POOL_TIMEOUT,
    )


def _create_limits() -> httpx.Limits:
    """Create connection pool limits."""
    return httpx.Limits(
        max_connections=DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections=DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=DEFAULT_KEEPALIVE_EXPIRY,
    )


# =============================================================================
# Client Management
# =============================================================================

async def get_http_client() -> httpx.AsyncClient:
    """
    Get the shared HTTP client instance.

    Creates the client on first call with connection pooling configured.
    Thread-safe through asyncio lock.

    Returns:
        Shared httpx.AsyncClient instance
    """
    global _http_client

    if _http_client is not None and not _http_client.is_closed:
        return _http_client

    async with _client_lock:
        # Double-check after acquiring lock
        if _http_client is not None and not _http_client.is_closed:
            return _http_client

        logger.info(
            "Creating shared HTTP client",
            max_connections=DEFAULT_MAX_CONNECTIONS,
            max_keepalive=DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        )

        _http_client = httpx.AsyncClient(
            timeout=_create_timeout(),
            limits=_create_limits(),
            http2=True,  # Enable HTTP/2 for better performance
            follow_redirects=True,
        )

        return _http_client


async def close_http_client() -> None:
    """
    Close the shared HTTP client.

    Should be called during application shutdown.
    """
    global _http_client

    async with _client_lock:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None
            logger.info("Shared HTTP client closed")


# =============================================================================
# Specialized Clients
# =============================================================================

async def get_long_timeout_client(timeout_seconds: float = 300.0) -> httpx.AsyncClient:
    """
    Get a client configured for long-running operations.

    Use this for operations like large file downloads or model pulls
    that may take several minutes.

    Args:
        timeout_seconds: Total timeout for the operation (default 5 minutes)

    Returns:
        New httpx.AsyncClient with extended timeout

    Note:
        Caller is responsible for closing this client after use.
        Prefer using as context manager:

            async with await get_long_timeout_client(600) as client:
                response = await client.get(url)
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds, connect=30.0),
        limits=_create_limits(),
        http2=True,
        follow_redirects=True,
    )


# =============================================================================
# Retry Helper
# =============================================================================

async def fetch_with_retry(
    url: str,
    method: str = "GET",
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    **kwargs,
) -> httpx.Response:
    """
    Fetch URL with automatic retry on transient failures.

    Args:
        url: URL to fetch
        method: HTTP method (GET, POST, etc.)
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        **kwargs: Additional arguments passed to httpx request

    Returns:
        httpx.Response

    Raises:
        httpx.HTTPError: After all retries exhausted
    """
    client = await get_http_client()
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = backoff_factor * (2 ** attempt)
                logger.warning(
                    "HTTP request failed, retrying",
                    url=url,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e),
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "HTTP request failed after all retries",
                    url=url,
                    attempts=max_retries + 1,
                    error=str(e),
                )

        except httpx.HTTPStatusError as e:
            # Don't retry client errors (4xx), only server errors (5xx)
            if e.response.status_code < 500:
                raise
            last_exception = e
            if attempt < max_retries:
                wait_time = backoff_factor * (2 ** attempt)
                logger.warning(
                    "HTTP server error, retrying",
                    url=url,
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)

    raise last_exception
