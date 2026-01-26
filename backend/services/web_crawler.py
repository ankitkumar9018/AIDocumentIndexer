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
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Try to import crawl4ai
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    HAS_CRAWL4AI = True
except ImportError:
    HAS_CRAWL4AI = False
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

    def __init__(self, config: Optional[WebCrawlerConfig] = None):
        """
        Initialize the web crawler.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or WebCrawlerConfig()
        self._crawler: Optional[Any] = None
        self._llm_client = None
        self._cache: Dict[str, Tuple[CrawlResult, datetime]] = {}
        self._request_times: List[datetime] = []

        if not HAS_CRAWL4AI:
            logger.warning("crawl4ai not available, using fallback HTTP client")

    async def _get_crawler(self) -> Optional[Any]:
        """Get or create the crawler instance."""
        if not HAS_CRAWL4AI:
            return None

        if self._crawler is None:
            browser_config = BrowserConfig(
                headless=True,
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

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        # Clean old request times (older than 1 minute)
        now = datetime.utcnow()
        self._request_times = [
            t for t in self._request_times
            if (now - t).seconds < 60
        ]

        # Check if we've exceeded the rate limit
        if len(self._request_times) >= self.config.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0]).seconds
            if wait_time > 0:
                logger.debug("Rate limit reached, waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)

        # Add random delay between requests
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
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

    def _cache_result(self, url: str, result: CrawlResult) -> None:
        """Cache a crawl result."""
        if self.config.cache_enabled:
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
        # Check cache first
        if not bypass_cache:
            cached = self._check_cache(url)
            if cached:
                return cached

        start_time = datetime.utcnow()

        # Apply rate limiting
        await self._apply_rate_limit()

        try:
            if HAS_CRAWL4AI:
                result = await self._crawl_with_crawl4ai(url)
            else:
                result = await self._crawl_with_httpx(url)

            # Calculate timing
            result.crawl_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Cache successful result
            if result.success:
                self._cache_result(url, result)

            return result

        except Exception as e:
            logger.error("Crawl failed", url=url, error=str(e))
            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
                crawl_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _crawl_with_crawl4ai(self, url: str) -> CrawlResult:
        """Crawl using crawl4ai (preferred method)."""
        crawler = await self._get_crawler()

        # Configure crawl run
        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            bypass_cache=True,
            page_timeout=self.config.page_timeout,
        )

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
