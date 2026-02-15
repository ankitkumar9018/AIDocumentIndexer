"""
AIDocumentIndexer - Enhanced Web Crawler (Phase 65)
====================================================

World-class web crawler with LLM-powered extraction capabilities.

Features:
- Anti-bot bypass (stealth mode, magic mode)
- LLM-powered structured data extraction
- Schema-based extraction for known patterns
- Website querying (answer questions about any URL)
- Rate limiting and politeness
- Link extraction and site mapping

Usage:
    from backend.services.web_crawler import get_web_crawler

    crawler = get_web_crawler()
    result = await crawler.crawl("https://example.com")

    # With LLM extraction
    extracted = await crawler.crawl_with_extraction(
        url="https://news.site.com/article",
        schema={"title": "str", "author": "str", "content": "str"}
    )

    # Query any website
    answer = await crawler.query_website(
        url="https://docs.python.org",
        question="How do I create a virtual environment?"
    )
"""

import asyncio
import gzip
import json
import random
import re
import urllib.robotparser
import xml.etree.ElementTree as ET
try:
    import defusedxml.ElementTree as SafeET
except ImportError:
    SafeET = None
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


def _validate_crawl_url(url: str) -> str:
    """Validate crawl URL to prevent SSRF attacks.

    Blocks private IPs, cloud metadata endpoints, and non-HTTP protocols.
    Returns the URL if valid, raises ValueError otherwise.
    """
    import ipaddress
    import socket

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP/HTTPS protocols are allowed")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname")

    blocked_hosts = {"metadata.google.internal", "metadata.gcp.internal", "169.254.169.254"}
    if hostname in blocked_hosts:
        raise ValueError("URL blocked: targets metadata endpoint")

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("URL blocked: targets private/internal network")
    except ValueError as ve:
        if "blocked" in str(ve):
            raise
        # DNS hostname — resolve and check
        try:
            resolved = ipaddress.ip_address(socket.gethostbyname(hostname))
            if resolved.is_private or resolved.is_loopback or resolved.is_link_local or resolved.is_reserved:
                raise ValueError("URL blocked: resolves to private/internal network")
        except socket.gaierror:
            pass  # DNS failure will be caught by httpx

    return url

# Try to import crawl4ai
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    HAS_CRAWL4AI = True
    try:
        from crawl4ai.deep_crawl import BFSDeepCrawlStrategy
        HAS_DEEP_CRAWL = True
    except ImportError:
        HAS_DEEP_CRAWL = False
except ImportError:
    HAS_CRAWL4AI = False
    HAS_DEEP_CRAWL = False
    AsyncWebCrawler = None
    logger.warning("crawl4ai not installed. Install with: pip install crawl4ai")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CrawlResult:
    """Result from crawling a URL."""
    url: str
    success: bool
    status_code: int = 200
    content: str = ""
    markdown: str = ""
    title: str = ""
    error: Optional[str] = None

    # Extracted data (if schema provided)
    extracted: Optional[Dict[str, Any]] = None

    # Links found on the page
    links: List[str] = field(default_factory=list)
    internal_links: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)

    # Metadata
    crawled_at: datetime = field(default_factory=datetime.utcnow)
    crawl_time_ms: float = 0.0

    # Source info
    content_type: str = ""
    word_count: int = 0

    # Recovery state
    state: Optional[Dict[str, Any]] = None


@dataclass
class WebCrawlerConfig:
    """Configuration for the web crawler."""
    # Anti-bot settings
    stealth_mode: bool = True
    magic_mode: bool = True  # Crawl4AI advanced bypass
    simulate_user: bool = True

    # Rate limiting
    min_delay: float = 1.0  # Minimum delay between requests (seconds)
    max_delay: float = 3.0  # Maximum delay (randomized)
    requests_per_minute: int = 30

    # User agent rotation
    rotate_user_agent: bool = True

    # Timeout settings
    page_timeout: int = 30000  # milliseconds
    navigation_timeout: int = 60000

    # Content settings
    extract_links: bool = True
    convert_to_markdown: bool = True
    max_content_length: int = 500000  # 500KB

    # LLM extraction
    llm_extraction_enabled: bool = True
    llm_model: str = "ollama/llama3.2"  # Default to local model

    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour


# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


# =============================================================================
# Proxy Manager
# =============================================================================

class ProxyManager:
    """Manage proxy rotation for web crawling."""

    def __init__(self, proxies: List[str], strategy: str = "round_robin"):
        """
        Initialize proxy manager.

        Args:
            proxies: List of proxy URLs (e.g., ["http://proxy1:8080", "socks5://proxy2:1080"])
            strategy: Rotation strategy - "round_robin" or "random"
        """
        self.proxies = proxies
        self.strategy = strategy  # round_robin, random
        self._index = 0

    def get_proxy(self) -> Optional[str]:
        """
        Get the next proxy based on the rotation strategy.

        Returns:
            Proxy URL string, or None if no proxies configured
        """
        if not self.proxies:
            return None
        if self.strategy == "random":
            return random.choice(self.proxies)
        proxy = self.proxies[self._index % len(self.proxies)]
        self._index += 1
        return proxy


# =============================================================================
# Enhanced Web Crawler
# =============================================================================

class EnhancedWebCrawler:
    """
    World-class web crawler with anti-bot bypass and LLM extraction.

    This crawler provides:
    1. Anti-bot bypass using stealth mode and user agent rotation
    2. LLM-powered content extraction for structured data
    3. Schema-based extraction for known patterns
    4. Website querying capability
    5. Rate limiting and polite crawling
    """

    def __init__(
        self,
        config: Optional[WebCrawlerConfig] = None,
        proxy_manager: Optional[ProxyManager] = None,
    ):
        """
        Initialize the web crawler.

        Args:
            config: Optional configuration, uses defaults if not provided
            proxy_manager: Optional proxy manager for proxy rotation
        """
        self.config = config or WebCrawlerConfig()
        self.proxy_manager = proxy_manager
        self._crawler: Optional[Any] = None
        self._llm_client = None
        self._cache: Dict[str, Tuple[CrawlResult, datetime]] = {}
        self._request_times: List[datetime] = []
        self._robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._domain_crawl_delays: Dict[str, float] = {}

        # Feature flags (loaded from settings on first use)
        self._settings_loaded = False
        self.use_crawl4ai = HAS_CRAWL4AI
        self.jina_fallback_enabled = True
        self.adaptive_crawling_enabled = False
        self.crash_recovery_enabled = True
        self.stealth_mode_enabled = True
        self.magic_mode_enabled = True
        self.headless_browser = True
        self.llm_extraction_enabled = True
        self.smart_extraction_enabled = True
        self.user_agent_rotation_enabled = True
        self.respect_robots_txt = True
        self.extract_images = True
        self.max_crawl_depth = 2
        self.max_pages_per_site = 100

        if not HAS_CRAWL4AI:
            logger.warning("crawl4ai not available, using fallback HTTP client")

    async def _load_settings(self) -> None:
        """Load configuration from settings service (called lazily on first crawl)."""
        if self._settings_loaded:
            return

        try:
            from backend.services.settings import get_settings_service
            settings_svc = get_settings_service()

            # Core scraping settings
            self.use_crawl4ai = HAS_CRAWL4AI and (await settings_svc.get_setting("scraping.use_crawl4ai") != False)
            self.headless_browser = await settings_svc.get_setting("scraping.headless_browser") != False
            timeout_seconds = await settings_svc.get_setting("scraping.timeout_seconds")
            if timeout_seconds is not None:
                self.config.page_timeout = int(timeout_seconds) * 1000
            _max_depth = await settings_svc.get_setting("scraping.max_depth")
            self.max_crawl_depth = int(_max_depth) if _max_depth is not None else 2
            rate_limit_ms = await settings_svc.get_setting("scraping.rate_limit_ms")
            if rate_limit_ms is not None:
                delay_seconds = int(rate_limit_ms) / 1000.0
                self.config.min_delay = delay_seconds
                self.config.max_delay = delay_seconds * 2
            self.respect_robots_txt = await settings_svc.get_setting("scraping.respect_robots_txt") != False
            self.config.extract_links = await settings_svc.get_setting("scraping.extract_links") != False
            self.extract_images = await settings_svc.get_setting("scraping.extract_images") != False

            # Phase 96 scraping settings
            proxy_enabled = await settings_svc.get_setting("scraping.proxy_enabled")
            if proxy_enabled:
                proxy_list_str = await settings_svc.get_setting("scraping.proxy_list") or ""
                proxy_strategy = await settings_svc.get_setting("scraping.proxy_rotation_strategy") or "round_robin"
                proxies = [p.strip() for p in proxy_list_str.split(",") if p.strip()]
                if proxies:
                    self.proxy_manager = ProxyManager(proxies=proxies, strategy=proxy_strategy)
                    logger.info("Proxy rotation enabled", count=len(proxies), strategy=proxy_strategy)

            self.jina_fallback_enabled = await settings_svc.get_setting("scraping.jina_reader_fallback") != False
            self.adaptive_crawling_enabled = bool(await settings_svc.get_setting("scraping.adaptive_crawling"))
            self.crash_recovery_enabled = await settings_svc.get_setting("scraping.crash_recovery_enabled") != False

            # Crawler feature settings
            _max_pages = await settings_svc.get_setting("crawler.max_pages_per_site")
            self.max_pages_per_site = int(_max_pages) if _max_pages is not None else 100
            self.stealth_mode_enabled = await settings_svc.get_setting("crawler.stealth_mode_enabled") != False
            self.magic_mode_enabled = await settings_svc.get_setting("crawler.magic_mode_enabled") != False
            self.llm_extraction_enabled = await settings_svc.get_setting("crawler.llm_extraction_enabled") != False
            self.config.llm_extraction_enabled = self.llm_extraction_enabled
            self.smart_extraction_enabled = await settings_svc.get_setting("crawler.smart_extraction_enabled") != False
            self.user_agent_rotation_enabled = await settings_svc.get_setting("crawler.user_agent_rotation_enabled") != False
            self.config.rotate_user_agent = self.user_agent_rotation_enabled

            self._settings_loaded = True
            logger.info("Web crawler settings loaded from admin configuration")

        except Exception as e:
            logger.warning("Failed to load crawler settings, using defaults", error=str(e))
            self._settings_loaded = True  # Don't retry on every request

    async def _get_crawler(self) -> Optional[Any]:
        """Get or create the crawler instance."""
        if not HAS_CRAWL4AI or not self.use_crawl4ai:
            return None

        if self._crawler is None:
            browser_config = BrowserConfig(
                headless=self.headless_browser,
                verbose=False,
            )
            self._crawler = AsyncWebCrawler(config=browser_config)
            await self._crawler.start()

        return self._crawler

    async def close(self) -> None:
        """Close the crawler and release resources."""
        if self._crawler is not None:
            await self._crawler.close()
            self._crawler = None

    def _get_user_agent(self) -> str:
        """Get a random user agent."""
        if self.config.rotate_user_agent:
            return random.choice(USER_AGENTS)
        return USER_AGENTS[0]

    async def _apply_rate_limit(self, url: Optional[str] = None) -> None:
        """Apply rate limiting between requests, respecting robots.txt crawl_delay."""
        # Clean old request times (older than 1 minute)
        now = datetime.utcnow()
        self._request_times = [
            t for t in self._request_times
            if (now - t).total_seconds() < 60
        ]

        # Check if we've exceeded the rate limit
        if len(self._request_times) >= self.config.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0]).total_seconds()
            if wait_time > 0:
                logger.debug("Rate limit reached, waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)

        # Determine delay: use max of configured delay and robots.txt crawl_delay
        configured_delay = random.uniform(self.config.min_delay, self.config.max_delay)
        robots_delay = 0.0
        if url:
            domain = urlparse(url).netloc
            robots_delay = self._domain_crawl_delays.get(domain, 0.0)

        delay = max(configured_delay, robots_delay)
        if robots_delay > configured_delay and robots_delay > 0:
            logger.debug("Using robots.txt crawl_delay", domain=urlparse(url).netloc, delay=robots_delay)
        await asyncio.sleep(delay)

        self._request_times.append(datetime.utcnow())

    def _check_cache(self, url: str) -> Optional[CrawlResult]:
        """Check if URL is in cache and still valid."""
        if not self.config.cache_enabled:
            return None

        if url in self._cache:
            result, cached_at = self._cache[url]
            age = (datetime.utcnow() - cached_at).seconds
            if age < self.config.cache_ttl_seconds:
                logger.debug("Cache hit", url=url, age=age)
                return result
            else:
                del self._cache[url]

        return None

    _MAX_CACHE_SIZE = 1000
    _MAX_ROBOTS_CACHE_SIZE = 500

    def _cache_result(self, url: str, result: CrawlResult) -> None:
        """Cache a crawl result."""
        if self.config.cache_enabled:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self._MAX_CACHE_SIZE:
                oldest_keys = sorted(self._cache, key=lambda k: self._cache[k][1])[:100]
                for k in oldest_keys:
                    del self._cache[k]
            self._cache[url] = (result, datetime.utcnow())

    def _extract_links(
        self,
        html: str,
        base_url: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract links from HTML content.

        Returns:
            Tuple of (all_links, internal_links, external_links)
        """
        # Simple regex-based link extraction
        link_pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(link_pattern, html, re.IGNORECASE)

        base_domain = urlparse(base_url).netloc

        all_links = []
        internal_links = []
        external_links = []

        for href in matches:
            # Skip non-HTTP links
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            # Make absolute URL
            if href.startswith('/'):
                full_url = urljoin(base_url, href)
            elif not href.startswith('http'):
                full_url = urljoin(base_url, href)
            else:
                full_url = href

            # Clean URL (remove fragments)
            full_url = full_url.split('#')[0]

            if full_url and full_url not in all_links:
                all_links.append(full_url)

                link_domain = urlparse(full_url).netloc
                if link_domain == base_domain:
                    internal_links.append(full_url)
                else:
                    external_links.append(full_url)

        return all_links, internal_links, external_links

    async def crawl(
        self,
        url: str,
        bypass_cache: bool = False,
    ) -> CrawlResult:
        """
        Crawl a URL and return the content.

        Args:
            url: URL to crawl
            bypass_cache: If True, skip cache lookup

        Returns:
            CrawlResult with content and metadata
        """
        # SSRF prevention: validate URL before crawling
        _validate_crawl_url(url)

        # Load settings from admin configuration on first use
        await self._load_settings()

        # Check cache first
        if not bypass_cache:
            cached = self._check_cache(url)
            if cached:
                return cached

        start_time = datetime.utcnow()

        # Apply rate limiting (respects robots.txt crawl_delay)
        await self._apply_rate_limit(url)

        try:
            if self.use_crawl4ai:
                result = await self._crawl_with_crawl4ai(url)
            else:
                result = await self._crawl_with_httpx(url)

            # Calculate timing
            result.crawl_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Cache successful result
            if result.success:
                self._cache_result(url, result)

                # Extract entities for knowledge graph enrichment
                if self.smart_extraction_enabled and result.markdown and len(result.markdown) > 100:
                    asyncio.create_task(self._extract_entities_for_kg(result))

            return result

        except Exception as e:
            logger.warning("Primary crawl failed, trying fallbacks", url=url, error=str(e))

            # Fallback 1: try httpx if crawl4ai was primary
            if self.use_crawl4ai:
                try:
                    result = await self._crawl_with_httpx(url)
                    result.crawl_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    if result.success:
                        self._cache_result(url, result)
                        return result
                except Exception as httpx_err:
                    logger.warning("httpx fallback also failed", url=url, error=str(httpx_err))

            # Fallback 2: try Jina Reader API (gated on setting)
            if self.jina_fallback_enabled:
                try:
                    result = await self._crawl_with_jina(url)
                    result.crawl_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    if result.success:
                        self._cache_result(url, result)
                        return result
                except Exception as jina_err:
                    logger.warning("Jina fallback also failed", url=url, error=str(jina_err))

            logger.error("All crawl methods failed", url=url, error=str(e))
            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
                crawl_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _crawl_with_crawl4ai(self, url: str) -> CrawlResult:
        """Crawl using crawl4ai (preferred method)."""
        crawler = await self._get_crawler()

        # Configure crawl run with admin settings
        run_kwargs: Dict[str, Any] = {
            "word_count_threshold": 10,
            "bypass_cache": True,
            "page_timeout": self.config.page_timeout,
        }

        # Pass magic mode if enabled (Crawl4AI advanced anti-detection)
        if self.magic_mode_enabled:
            run_kwargs["magic"] = True

        # Pass proxy if available
        if self.proxy_manager:
            proxy = self.proxy_manager.get_proxy()
            if proxy:
                run_kwargs["proxy"] = proxy

        run_config = CrawlerRunConfig(**run_kwargs)

        result = await crawler.arun(
            url=url,
            config=run_config,
        )

        if not result.success:
            return CrawlResult(
                url=url,
                success=False,
                error=result.error_message or "Unknown error",
            )

        # Extract links
        all_links, internal, external = self._extract_links(
            result.html or "",
            url,
        )

        return CrawlResult(
            url=url,
            success=True,
            status_code=result.status_code or 200,
            content=result.html or "",
            markdown=result.markdown or "",
            title=result.title or "",
            links=all_links,
            internal_links=internal,
            external_links=external,
            word_count=len((result.markdown or "").split()),
        )

    async def _crawl_with_httpx(self, url: str) -> CrawlResult:
        """Fallback crawl using httpx."""
        import httpx

        headers = {
            "User-Agent": self._get_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)

            if response.status_code != 200:
                return CrawlResult(
                    url=url,
                    success=False,
                    status_code=response.status_code,
                    error=f"HTTP {response.status_code}",
                )

            content = response.text

            # Simple title extraction
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else ""

            # Extract links
            all_links, internal, external = self._extract_links(content, url)

            # Basic HTML to text conversion
            # Remove scripts and styles
            text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return CrawlResult(
                url=url,
                success=True,
                status_code=response.status_code,
                content=content,
                markdown=text,  # Simple text extraction
                title=title,
                links=all_links,
                internal_links=internal,
                external_links=external,
                content_type=response.headers.get("content-type", ""),
                word_count=len(text.split()),
            )

    async def _crawl_with_jina(self, url: str) -> CrawlResult:
        """
        Fallback crawl using Jina Reader API.

        Jina Reader converts any URL to clean markdown content via their
        r.jina.ai endpoint. Used as a last resort when both crawl4ai and
        httpx fail (e.g., for heavily JS-rendered pages).

        Args:
            url: URL to crawl

        Returns:
            CrawlResult with markdown content from Jina Reader
        """
        import httpx

        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "application/json",
            "User-Agent": self._get_user_agent(),
        }

        logger.info("Attempting Jina Reader fallback", url=url)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(jina_url, headers=headers, follow_redirects=True)

            if response.status_code != 200:
                return CrawlResult(
                    url=url,
                    success=False,
                    status_code=response.status_code,
                    error=f"Jina Reader HTTP {response.status_code}",
                )

            try:
                data = response.json()
                content = data.get("data", {}).get("content", "") or data.get("content", "")
                title = data.get("data", {}).get("title", "") or data.get("title", "")
            except (json.JSONDecodeError, ValueError):
                # If not JSON, use raw text
                content = response.text
                title = ""

            if not content:
                return CrawlResult(
                    url=url,
                    success=False,
                    error="Jina Reader returned empty content",
                )

            logger.info("Jina Reader fallback succeeded", url=url, content_length=len(content))

            return CrawlResult(
                url=url,
                success=True,
                status_code=200,
                content=content,
                markdown=content,
                title=title,
                content_type="text/markdown",
                word_count=len(content.split()),
            )

    async def crawl_with_extraction(
        self,
        url: str,
        schema: Optional[Dict[str, str]] = None,
        extraction_prompt: Optional[str] = None,
    ) -> CrawlResult:
        """
        Crawl a URL and extract structured data using LLM.

        Args:
            url: URL to crawl
            schema: Optional dict of field_name -> type for extraction
            extraction_prompt: Optional custom extraction prompt

        Returns:
            CrawlResult with extracted data in 'extracted' field
        """
        # First, crawl the page
        result = await self.crawl(url)

        if not result.success:
            return result

        if not self.config.llm_extraction_enabled:
            return result

        # Extract data using LLM
        try:
            extracted = await self._llm_extract(
                content=result.markdown or result.content,
                schema=schema,
                extraction_prompt=extraction_prompt,
            )
            result.extracted = extracted
        except Exception as e:
            logger.warning("LLM extraction failed", url=url, error=str(e))
            result.extracted = {"error": str(e)}

        return result

    async def _llm_extract(
        self,
        content: str,
        schema: Optional[Dict[str, str]] = None,
        extraction_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured data from content using LLM.

        Args:
            content: Text content to extract from
            schema: Dict of field_name -> type for extraction
            extraction_prompt: Custom extraction prompt

        Returns:
            Dict of extracted data
        """
        # Truncate content if too long
        max_content = 8000
        if len(content) > max_content:
            content = content[:max_content] + "..."

        # Build extraction prompt
        if extraction_prompt:
            prompt = extraction_prompt
        elif schema:
            schema_str = json.dumps(schema, indent=2)
            prompt = f"""Extract the following data from the content below.
Return ONLY valid JSON matching this schema:
{schema_str}

If a field cannot be found, use null.

Content:
{content}

JSON Output:"""
        else:
            # Default: smart extraction
            prompt = f"""Analyze this content and extract the key information.
Return a JSON object with these fields:
- title: Main title or headline
- summary: Brief 2-3 sentence summary
- main_points: List of key points (up to 5)
- entities: Dict of entity_type -> list of entities (people, organizations, places)
- dates: List of dates mentioned
- links_mentioned: List of URLs mentioned in text

Content:
{content}

JSON Output:"""

        # Call LLM for extraction
        try:
            from backend.services.llm_router import get_llm_router

            llm = get_llm_router()
            response = await llm.generate(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1,  # Low temperature for structured output
            )

            # Parse JSON from response
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)

            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {"raw_response": response_text}

        except Exception as e:
            logger.error("LLM extraction error", error=str(e))
            return {"error": str(e)}

    async def query_website(
        self,
        url: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Answer a question about a website's content.

        This crawls the URL and uses RAG to answer the question.

        Args:
            url: URL to query
            question: Question to answer about the content

        Returns:
            Dict with answer, source, and confidence
        """
        # Crawl the page
        result = await self.crawl(url)

        if not result.success:
            return {
                "answer": None,
                "error": result.error,
                "source": url,
            }

        # Use LLM to answer question
        try:
            from backend.services.llm_router import get_llm_router

            llm = get_llm_router()

            content = result.markdown or result.content
            # Truncate if too long
            if len(content) > 12000:
                content = content[:12000] + "..."

            prompt = f"""Based on the following content from {url}, answer this question:

Question: {question}

Content:
{content}

Instructions:
1. Answer based ONLY on the content provided
2. If the answer is not in the content, say "I couldn't find this information in the page"
3. Be concise but complete
4. Cite specific parts of the content when relevant

Answer:"""

            response = await llm.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
            )

            answer = response.get("content", "") if isinstance(response, dict) else str(response)

            return {
                "answer": answer,
                "source": url,
                "title": result.title,
                "word_count": result.word_count,
                "crawl_time_ms": result.crawl_time_ms,
            }

        except Exception as e:
            logger.error("Query website failed", url=url, error=str(e))
            return {
                "answer": None,
                "error": str(e),
                "source": url,
            }

    async def crawl_site(
        self,
        start_url: str,
        max_pages: int = 10,
        same_domain_only: bool = True,
        url_pattern: Optional[str] = None,
    ) -> List[CrawlResult]:
        """
        Crawl multiple pages from a site.

        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            same_domain_only: Only follow internal links
            url_pattern: Optional regex pattern to filter URLs

        Returns:
            List of CrawlResult objects
        """
        # SSRF prevention: validate start URL
        _validate_crawl_url(start_url)

        results = []
        visited = set()
        to_visit = [start_url]

        while to_visit and len(results) < max_pages:
            url = to_visit.pop(0)

            if url in visited:
                continue

            visited.add(url)

            # Check URL pattern if specified
            if url_pattern and not re.search(url_pattern, url):
                continue

            # Crawl the page
            result = await self.crawl(url)
            results.append(result)

            if result.success:
                # Add new links to queue
                links = result.internal_links if same_domain_only else result.links
                for link in links:
                    if link not in visited and link not in to_visit:
                        to_visit.append(link)

        logger.info(
            "Site crawl completed",
            start_url=start_url,
            pages_crawled=len(results),
            pages_successful=sum(1 for r in results if r.success),
        )

        return results

    # =========================================================================
    # Sitemap Crawling (v0.8.0)
    # =========================================================================

    async def crawl_sitemap(self, url: str, max_pages: int = 50) -> List[CrawlResult]:
        """
        Crawl a website using its sitemap.xml for URL discovery.

        Fetches the sitemap from the domain, parses it to extract URLs,
        prioritizes recently updated content based on <lastmod> dates,
        and crawls each URL up to max_pages.

        Handles:
        - Standard <urlset> sitemaps
        - Sitemap indexes (<sitemapindex>) with nested sitemaps
        - Compressed .xml.gz sitemaps
        - Sitemap URLs discovered from robots.txt

        Args:
            url: Base URL of the site (e.g., "https://example.com")
            max_pages: Maximum number of pages to crawl

        Returns:
            List of CrawlResult objects from crawled sitemap URLs
        """
        import httpx

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        sitemap_urls_with_dates: List[Tuple[str, Optional[str]]] = []

        logger.info("Starting sitemap crawl", base_url=base_url, max_pages=max_pages)

        # Try to discover sitemap URLs from robots.txt first
        robots_info = await self.parse_robots_txt(url)
        sitemap_locations = robots_info.get("sitemaps", [])

        # Add default sitemap location if none found in robots.txt
        default_sitemap = f"{base_url}/sitemap.xml"
        if default_sitemap not in sitemap_locations:
            sitemap_locations.insert(0, default_sitemap)

        async with httpx.AsyncClient(timeout=30.0) as client:
            for sitemap_url in sitemap_locations:
                try:
                    urls = await self._fetch_sitemap(client, sitemap_url)
                    sitemap_urls_with_dates.extend(urls)
                    logger.info(
                        "Parsed sitemap",
                        sitemap_url=sitemap_url,
                        urls_found=len(urls),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to fetch sitemap",
                        sitemap_url=sitemap_url,
                        error=str(e),
                    )

        if not sitemap_urls_with_dates:
            logger.warning("No URLs found in sitemaps", base_url=base_url)
            return []

        # Sort by lastmod date (most recent first), URLs without dates go last
        sitemap_urls_with_dates.sort(
            key=lambda x: x[1] if x[1] else "",
            reverse=True,
        )

        # Crawl URLs up to max_pages
        results: List[CrawlResult] = []
        for page_url, lastmod in sitemap_urls_with_dates[:max_pages]:
            logger.info(
                "Crawling sitemap URL",
                url=page_url,
                lastmod=lastmod,
                progress=f"{len(results) + 1}/{min(len(sitemap_urls_with_dates), max_pages)}",
            )
            result = await self.crawl(page_url)
            results.append(result)

        logger.info(
            "Sitemap crawl completed",
            base_url=base_url,
            total_urls_found=len(sitemap_urls_with_dates),
            pages_crawled=len(results),
            pages_successful=sum(1 for r in results if r.success),
        )

        return results

    async def _fetch_sitemap(
        self,
        client: Any,
        sitemap_url: str,
        depth: int = 0,
        max_depth: int = 3,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Fetch and parse a single sitemap URL.

        Handles both regular XML and gzip-compressed sitemaps,
        as well as sitemap index files that reference other sitemaps.

        Args:
            client: httpx.AsyncClient instance
            sitemap_url: URL of the sitemap to fetch
            depth: Current recursion depth
            max_depth: Maximum recursion depth for nested sitemaps

        Returns:
            List of (url, lastmod) tuples extracted from the sitemap
        """
        if depth >= max_depth:
            logger.warning("Sitemap recursion depth limit reached", sitemap_url=sitemap_url, depth=depth)
            return []
        headers = {"User-Agent": self._get_user_agent()}
        response = await client.get(sitemap_url, headers=headers, follow_redirects=True)

        if response.status_code != 200:
            return []

        # Handle gzip-compressed sitemaps
        MAX_SITEMAP_SIZE = 50 * 1024 * 1024  # 50MB decompressed limit
        if sitemap_url.endswith(".gz"):
            try:
                decompressed = gzip.decompress(response.content)
                if len(decompressed) > MAX_SITEMAP_SIZE:
                    logger.warning("Sitemap too large after decompression", sitemap_url=sitemap_url, size=len(decompressed))
                    return []
                xml_content = decompressed.decode("utf-8")
            except Exception:
                return []
        else:
            xml_content = response.text

        urls_with_dates: List[Tuple[str, Optional[str]]] = []

        try:
            if SafeET is None:
                logger.warning("defusedxml not installed — refusing to parse external XML for safety", sitemap_url=sitemap_url)
                return []
            root = SafeET.fromstring(xml_content)
        except (ET.ParseError, Exception) as parse_err:
            logger.warning("Failed to parse sitemap XML", sitemap_url=sitemap_url, error=str(parse_err))
            return []

        # Handle namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Check if this is a sitemap index
        if root.tag == f"{ns}sitemapindex":
            for sitemap_elem in root.findall(f"{ns}sitemap"):
                loc_elem = sitemap_elem.find(f"{ns}loc")
                if loc_elem is not None and loc_elem.text:
                    # Recursively fetch nested sitemaps
                    nested_urls = await self._fetch_sitemap(client, loc_elem.text.strip(), depth=depth + 1, max_depth=max_depth)
                    urls_with_dates.extend(nested_urls)
        else:
            # Standard urlset
            for url_elem in root.findall(f"{ns}url"):
                loc_elem = url_elem.find(f"{ns}loc")
                lastmod_elem = url_elem.find(f"{ns}lastmod")

                if loc_elem is not None and loc_elem.text:
                    loc = loc_elem.text.strip()
                    lastmod = lastmod_elem.text.strip() if lastmod_elem is not None and lastmod_elem.text else None
                    urls_with_dates.append((loc, lastmod))

        return urls_with_dates

    # =========================================================================
    # robots.txt Parsing (v0.8.0)
    # =========================================================================

    async def parse_robots_txt(self, url: str) -> Dict[str, Any]:
        """
        Parse the robots.txt file for a given domain.

        Uses urllib.robotparser to parse the robots.txt and extracts
        allowed/disallowed paths, crawl delay, and sitemap URLs.

        Args:
            url: Any URL from the domain (e.g., "https://example.com/page")

        Returns:
            Dict with keys:
                - allowed_paths: List of explicitly allowed path patterns
                - disallowed_paths: List of disallowed path patterns
                - crawl_delay: Crawl delay in seconds (or None)
                - sitemaps: List of sitemap URLs found in robots.txt
        """
        import httpx

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        result: Dict[str, Any] = {
            "allowed_paths": [],
            "disallowed_paths": [],
            "crawl_delay": None,
            "sitemaps": [],
        }

        try:
            # Fetch robots.txt content manually for more control
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    robots_url,
                    headers={"User-Agent": self._get_user_agent()},
                    follow_redirects=True,
                )

                if response.status_code != 200:
                    logger.debug("No robots.txt found", url=robots_url, status=response.status_code)
                    return result

                robots_content = response.text

            # Parse with RobotFileParser
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            rp.parse(robots_content.splitlines())

            # Extract crawl delay and store for rate limiting
            crawl_delay = rp.crawl_delay("*")
            if crawl_delay is not None:
                result["crawl_delay"] = float(crawl_delay)
                # Store per-domain crawl_delay so _apply_rate_limit() can enforce it
                self._domain_crawl_delays[parsed.netloc] = float(crawl_delay)

            # Extract sitemap URLs from robots.txt content
            for line in robots_content.splitlines():
                line = line.strip()
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    if sitemap_url:
                        result["sitemaps"].append(sitemap_url)
                elif line.lower().startswith("allow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        result["allowed_paths"].append(path)
                elif line.lower().startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        result["disallowed_paths"].append(path)

            # Cache the parser for is_url_allowed checks (bounded)
            domain = parsed.netloc
            if len(self._robots_cache) >= self._MAX_ROBOTS_CACHE_SIZE:
                # Evict arbitrary old entries
                keys_to_remove = list(self._robots_cache.keys())[:50]
                for k in keys_to_remove:
                    del self._robots_cache[k]
                    self._domain_crawl_delays.pop(k, None)
            self._robots_cache[domain] = rp

            logger.info(
                "Parsed robots.txt",
                url=robots_url,
                allowed=len(result["allowed_paths"]),
                disallowed=len(result["disallowed_paths"]),
                sitemaps=len(result["sitemaps"]),
                crawl_delay=result["crawl_delay"],
            )

        except Exception as e:
            logger.warning("Failed to parse robots.txt", url=robots_url, error=str(e))

        return result

    async def is_url_allowed(self, url: str) -> bool:
        """
        Check if a URL is allowed to be crawled per the site's robots.txt.

        Caches the robots.txt parser per domain to avoid repeated fetches.

        Args:
            url: Full URL to check

        Returns:
            True if the URL is allowed (or if robots.txt cannot be checked),
            False if the URL is disallowed
        """
        parsed = urlparse(url)
        domain = parsed.netloc

        # Load robots.txt if not cached for this domain
        if domain not in self._robots_cache:
            await self.parse_robots_txt(url)

        rp = self._robots_cache.get(domain)
        if rp is None:
            # If we couldn't load robots.txt, allow by default
            return True

        try:
            return rp.can_fetch("*", url)
        except Exception:
            return True

    # =========================================================================
    # Search-based URL Discovery (v0.8.0)
    # =========================================================================

    async def search_and_crawl(self, query: str, max_results: int = 5) -> List[CrawlResult]:
        """
        Discover URLs via DuckDuckGo search and crawl them.

        Uses DuckDuckGo HTML search to find relevant URLs for a query,
        then crawls each discovered URL.

        Args:
            query: Search query string
            max_results: Maximum number of URLs to discover and crawl

        Returns:
            List of CrawlResult objects from crawled search result URLs
        """
        import httpx

        logger.info("Starting search-and-crawl", query=query, max_results=max_results)

        # Search DuckDuckGo
        discovered_urls: List[str] = []
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers={
                        "User-Agent": self._get_user_agent(),
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    follow_redirects=True,
                )

                if response.status_code == 200:
                    html = response.text

                    # Extract result URLs from DuckDuckGo HTML results
                    # DuckDuckGo result links are in <a class="result__a" href="...">
                    url_pattern = r'href="(https?://[^"]+)"'
                    matches = re.findall(url_pattern, html)

                    for match in matches:
                        # Filter out DuckDuckGo internal URLs
                        if "duckduckgo.com" in match:
                            continue
                        # Decode DuckDuckGo redirect URLs
                        if "uddg=" in match:
                            uddg_match = re.search(r'uddg=([^&]+)', match)
                            if uddg_match:
                                from urllib.parse import unquote
                                decoded_url = unquote(uddg_match.group(1))
                                if decoded_url not in discovered_urls:
                                    discovered_urls.append(decoded_url)
                        elif match not in discovered_urls:
                            discovered_urls.append(match)

                        if len(discovered_urls) >= max_results:
                            break

                    logger.info(
                        "DuckDuckGo search completed",
                        query=query,
                        urls_found=len(discovered_urls),
                    )
                else:
                    logger.warning(
                        "DuckDuckGo search failed",
                        status_code=response.status_code,
                    )

        except Exception as e:
            logger.error("Search failed", query=query, error=str(e))
            return []

        if not discovered_urls:
            logger.warning("No URLs discovered from search", query=query)
            return []

        # Crawl discovered URLs
        results: List[CrawlResult] = []
        for i, url in enumerate(discovered_urls[:max_results]):
            logger.info(
                "Crawling search result",
                url=url,
                progress=f"{i + 1}/{len(discovered_urls[:max_results])}",
            )
            result = await self.crawl(url)
            results.append(result)

        logger.info(
            "Search-and-crawl completed",
            query=query,
            pages_crawled=len(results),
            pages_successful=sum(1 for r in results if r.success),
        )

        return results

    # =========================================================================
    # Crash Recovery State (v0.8.0)
    # =========================================================================

    async def crawl_with_recovery(
        self,
        url: str,
        state_file: Optional[str] = None,
    ) -> CrawlResult:
        """
        Crawl a URL with crash recovery support.

        If a state_file is provided and exists, resumes from the saved state.
        After crawling, saves the state to the file for future recovery.
        This is useful for long-running crawl operations that may be interrupted.

        Args:
            url: URL to crawl
            state_file: Optional path to a JSON file for saving/loading crawl state.
                        If None, no state persistence is performed.

        Returns:
            CrawlResult with state dict included for external state tracking
        """
        state: Dict[str, Any] = {}

        # Load existing state if available
        if state_file:
            state_path = Path(state_file)
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    logger.info(
                        "Loaded crawl recovery state",
                        state_file=state_file,
                        url=state.get("url"),
                        last_status=state.get("status"),
                    )

                    # If the previous crawl succeeded for this URL, return cached state
                    if state.get("url") == url and state.get("status") == "completed":
                        logger.info("Returning cached result from recovery state", url=url)
                        return CrawlResult(
                            url=url,
                            success=True,
                            content=state.get("content", ""),
                            markdown=state.get("markdown", ""),
                            title=state.get("title", ""),
                            word_count=state.get("word_count", 0),
                            state=state,
                        )
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to load recovery state", error=str(e))

        # Update state to in-progress
        state.update({
            "url": url,
            "status": "in_progress",
            "started_at": datetime.utcnow().isoformat(),
        })

        if state_file:
            self._save_state(state_file, state)

        # Perform the crawl
        try:
            result = await self.crawl(url, bypass_cache=True)

            # Update state with result
            state.update({
                "status": "completed" if result.success else "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "success": result.success,
                "title": result.title,
                "content": result.content[:10000] if result.content else "",  # Truncate for state file
                "markdown": result.markdown[:10000] if result.markdown else "",
                "word_count": result.word_count,
                "error": result.error,
                "crawl_time_ms": result.crawl_time_ms,
            })

            # Save state
            if state_file:
                self._save_state(state_file, state)

            # Attach state to result
            result.state = state

            logger.info(
                "Crawl with recovery completed",
                url=url,
                success=result.success,
                state_file=state_file,
            )

            return result

        except Exception as e:
            state.update({
                "status": "error",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat(),
            })
            if state_file:
                self._save_state(state_file, state)

            logger.error("Crawl with recovery failed", url=url, error=str(e))
            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
                state=state,
            )

    def _save_state(self, state_file: str, state: Dict[str, Any]) -> None:
        """
        Save crawl state to a JSON file.

        Args:
            state_file: Path to the state file
            state: State dictionary to save
        """
        try:
            state_path = Path(state_file)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(state, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to save crawl state", state_file=state_file, error=str(e))

    async def _extract_entities_for_kg(self, result: CrawlResult) -> None:
        """
        Extract entities from crawled content and store in knowledge graph.
        Runs as a background task to not block crawl response.
        """
        try:
            from backend.services.knowledge_graph import get_kg_service
            from backend.services.settings import get_settings_service

            settings_svc = get_settings_service()
            kg_enabled = await settings_svc.get_setting("rag.knowledge_graph_enabled")
            if not kg_enabled:
                return

            kg_service = await get_kg_service()

            # Use first 5000 chars of markdown for entity extraction
            text = result.markdown[:5000] if result.markdown else ""
            if not text:
                return

            entities = await kg_service.extract_entities(text)
            if entities:
                logger.info(
                    "Extracted entities from crawled page",
                    url=result.url,
                    entity_count=len(entities),
                )
        except Exception as e:
            logger.debug("KG entity extraction from crawl skipped", error=str(e))


# =============================================================================
# Factory Function
# =============================================================================

_web_crawler: Optional[EnhancedWebCrawler] = None


def get_web_crawler(
    config: Optional[WebCrawlerConfig] = None,
) -> EnhancedWebCrawler:
    """
    Get or create the web crawler instance.

    Args:
        config: Optional configuration

    Returns:
        EnhancedWebCrawler instance
    """
    global _web_crawler

    if _web_crawler is None or config is not None:
        _web_crawler = EnhancedWebCrawler(config=config)

    return _web_crawler


async def close_web_crawler() -> None:
    """Close the web crawler and release resources."""
    global _web_crawler

    if _web_crawler is not None:
        await _web_crawler.close()
        _web_crawler = None
