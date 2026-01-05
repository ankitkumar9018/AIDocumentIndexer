"""
AIDocumentIndexer - Web Scraping Service
=========================================

Web scraping service using Crawl4AI for converting web pages
to LLM-ready content and integrating into the RAG pipeline.
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

# Try to import crawl4ai, provide fallback if not available
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("crawl4ai not installed. Web scraping functionality will be limited.")


# =============================================================================
# Enums and Types
# =============================================================================

class ScrapeStatus(str, Enum):
    """Status of a scrape job."""
    PENDING = "pending"
    SCRAPING = "scraping"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentType(str, Enum):
    """Type of scraped content."""
    ARTICLE = "article"
    DOCUMENTATION = "documentation"
    BLOG = "blog"
    NEWS = "news"
    GENERAL = "general"


class StorageMode(str, Enum):
    """How to handle scraped content."""
    IMMEDIATE = "immediate"  # Use for current query, don't store
    PERMANENT = "permanent"  # Store in RAG for future queries


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScrapeConfig:
    """Configuration for scraping."""
    extract_images: bool = False
    extract_links: bool = True
    extract_metadata: bool = True
    max_depth: int = 3  # How many levels to follow links
    max_pages: int = 50  # Max pages to scrape
    timeout: int = 30  # Seconds
    user_agent: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    wait_for_js: bool = True  # Wait for JavaScript to load
    js_wait_time: float = 2.0  # Seconds to wait for JS


@dataclass
class ScrapedPage:
    """A single scraped page."""
    url: str
    title: str
    content: str  # Markdown content
    html: Optional[str] = None
    text: Optional[str] = None  # Plain text
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = None
    word_count: int = 0


@dataclass
class ScrapeJob:
    """A scraping job with one or more URLs."""
    id: str
    user_id: str
    urls: List[str]
    config: ScrapeConfig
    storage_mode: StorageMode
    status: ScrapeStatus
    pages: List[ScrapedPage] = field(default_factory=list)
    total_pages: int = 0
    pages_scraped: int = 0
    pages_failed: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    access_tier: int = 1
    collection: Optional[str] = None


@dataclass
class ScrapeResult:
    """Result of a scrape operation."""
    success: bool
    job_id: str
    pages_scraped: int
    pages_failed: int
    content: List[ScrapedPage]
    total_word_count: int
    error: Optional[str] = None


# =============================================================================
# Fallback Scraper (when crawl4ai not available)
# =============================================================================

class FallbackScraper:
    """Simple scraper using requests + BeautifulSoup as fallback."""

    async def scrape_url(self, url: str, config: ScrapeConfig) -> ScrapedPage:
        """Scrape a single URL using basic HTTP requests."""
        import aiohttp
        from bs4 import BeautifulSoup

        headers = config.headers.copy()
        if config.user_agent:
            headers["User-Agent"] = config.user_agent
        else:
            headers["User-Agent"] = "Mozilla/5.0 (compatible; AIDocumentIndexer/1.0)"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as response:
                html = await response.text()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get title
        title = soup.title.string if soup.title else urlparse(url).netloc

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Convert to basic markdown
        content = self._html_to_markdown(soup)

        # Extract links - handle both absolute and relative URLs
        links = []
        if config.extract_links:
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # Skip empty, anchor-only, javascript, and mailto links
                if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                    continue
                # Normalize relative URLs to absolute
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                # Validate it's a proper URL
                parsed = urlparse(href)
                if parsed.scheme in ['http', 'https'] and parsed.netloc:
                    links.append(href)

        # Extract images - handle both absolute and relative URLs
        images = []
        if config.extract_images:
            for img in soup.find_all("img", src=True):
                src = img["src"]
                if not src or src.startswith('data:'):
                    continue
                # Normalize relative URLs to absolute
                if not src.startswith(('http://', 'https://')):
                    src = urljoin(url, src)
                images.append(src)

        # Extract metadata
        metadata = {}
        if config.extract_metadata:
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property")
                content_attr = meta.get("content")
                if name and content_attr:
                    metadata[name] = content_attr

        # Compute hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return ScrapedPage(
            url=url,
            title=title,
            content=content,
            html=html,
            text=text,
            metadata=metadata,
            links=links[:50],  # Limit links
            images=images[:20],  # Limit images
            content_hash=content_hash,
            word_count=len(text.split()),
        )

    def _html_to_markdown(self, soup) -> str:
        """Convert HTML to basic markdown."""
        lines = []

        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "li", "blockquote", "pre", "code"]):
            tag = element.name

            if tag.startswith("h"):
                level = int(tag[1])
                lines.append(f"{'#' * level} {element.get_text(strip=True)}\n")
            elif tag == "p":
                text = element.get_text(strip=True)
                if text:
                    lines.append(f"{text}\n")
            elif tag == "li":
                lines.append(f"- {element.get_text(strip=True)}")
            elif tag == "blockquote":
                lines.append(f"> {element.get_text(strip=True)}\n")
            elif tag in ["pre", "code"]:
                lines.append(f"```\n{element.get_text()}\n```\n")

        return "\n".join(lines)


# =============================================================================
# Web Scraper Service
# =============================================================================

class WebScraperService:
    """
    Service for scraping web pages and converting to RAG-ready content.

    Uses Crawl4AI for robust scraping with JavaScript support,
    or falls back to basic scraping if not available.
    """

    def __init__(self):
        self._jobs: Dict[str, ScrapeJob] = {}
        self._scraped_urls: Dict[str, ScrapedPage] = {}  # URL -> page cache
        self._fallback_scraper = FallbackScraper()

    async def _scrape_with_crawl4ai(
        self,
        url: str,
        config: ScrapeConfig,
    ) -> ScrapedPage:
        """Scrape using Crawl4AI."""
        if not CRAWL4AI_AVAILABLE:
            raise RuntimeError("crawl4ai not installed")

        # Configure browser settings
        browser_config = BrowserConfig(
            headless=True,
            user_agent=config.user_agent or "Mozilla/5.0 (compatible; AIDocumentIndexer/1.0)",
        )

        # Configure crawl run settings - use new API style
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=10,
            # Link extraction settings - key for subpage crawling
            exclude_external_links=False,  # Keep external links for analysis
            exclude_social_media_links=True,  # Skip social media links
            # Wait for JavaScript if configured
            wait_until="domcontentloaded",
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=url,
                config=run_config,
            )

            if not result.success:
                raise ValueError(f"Crawl failed: {result.error_message}")

            # Extract content
            content = result.markdown or ""
            text = result.cleaned_html or result.markdown or ""

            # Compute hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Extract links - crawl4ai returns dicts with 'href' key
            links = []
            if result.links:
                internal = result.links.get("internal", [])
                external = result.links.get("external", [])
                logger.debug(
                    "Extracted links from page",
                    url=url,
                    internal_count=len(internal),
                    external_count=len(external),
                )
                for link_item in internal + external:
                    if isinstance(link_item, dict):
                        href = link_item.get("href", "")
                    else:
                        href = str(link_item)
                    if href:
                        # Normalize relative URLs
                        if not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        links.append(href)
            else:
                logger.warning("No links extracted from page", url=url)

            # Extract images
            images = []
            if result.media:
                for img_item in result.media.get("images", []):
                    if isinstance(img_item, dict):
                        src = img_item.get("src", "")
                    else:
                        src = str(img_item)
                    if src:
                        images.append(src)

            return ScrapedPage(
                url=url,
                title=result.metadata.get("title", urlparse(url).netloc),
                content=content,
                html=result.html,
                text=text,
                metadata=result.metadata or {},
                links=links,
                images=images,
                content_hash=content_hash,
                word_count=len(text.split()) if text else 0,
            )

    async def _scrape_url(
        self,
        url: str,
        config: ScrapeConfig,
    ) -> ScrapedPage:
        """Scrape a single URL."""
        # Check cache
        if url in self._scraped_urls:
            cached = self._scraped_urls[url]
            # Return cached if less than 1 hour old
            age = (datetime.utcnow() - cached.scraped_at).total_seconds()
            if age < 3600:
                logger.debug("Returning cached page", url=url)
                return cached

        # Try crawl4ai first
        if CRAWL4AI_AVAILABLE:
            try:
                page = await self._scrape_with_crawl4ai(url, config)
            except Exception as e:
                logger.warning("Crawl4AI failed, falling back", url=url, error=str(e))
                page = await self._fallback_scraper.scrape_url(url, config)
        else:
            page = await self._fallback_scraper.scrape_url(url, config)

        # Cache result
        self._scraped_urls[url] = page

        return page

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def create_job(
        self,
        user_id: str,
        urls: List[str],
        storage_mode: StorageMode = StorageMode.IMMEDIATE,
        config: Optional[ScrapeConfig] = None,
        access_tier: int = 1,
        collection: Optional[str] = None,
    ) -> ScrapeJob:
        """Create a new scrape job."""
        if config is None:
            config = ScrapeConfig()

        # Validate URLs
        validated_urls = []
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme not in ["http", "https"]:
                logger.warning("Invalid URL scheme", url=url)
                continue
            validated_urls.append(url)

        if not validated_urls:
            raise ValueError("No valid URLs provided")

        job = ScrapeJob(
            id=str(uuid4()),
            user_id=user_id,
            urls=validated_urls,
            config=config,
            storage_mode=storage_mode,
            status=ScrapeStatus.PENDING,
            total_pages=len(validated_urls),
            access_tier=access_tier,
            collection=collection,
        )

        self._jobs[job.id] = job

        logger.info(
            "Created scrape job",
            job_id=job.id,
            urls=len(validated_urls),
            storage_mode=storage_mode.value,
        )

        return job

    async def run_job(self, job_id: str) -> ScrapeResult:
        """Run a scrape job."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status not in [ScrapeStatus.PENDING, ScrapeStatus.FAILED]:
            raise ValueError(f"Job is not runnable: {job.status.value}")

        logger.info("Starting scrape job", job_id=job_id, urls=len(job.urls))

        job.status = ScrapeStatus.SCRAPING
        pages = []
        total_words = 0

        for url in job.urls:
            try:
                page = await self._scrape_url(url, job.config)
                pages.append(page)
                job.pages_scraped += 1
                total_words += page.word_count

                logger.debug(
                    "Scraped page",
                    url=url,
                    title=page.title,
                    words=page.word_count,
                )

            except Exception as e:
                logger.error("Failed to scrape URL", url=url, error=str(e))
                job.pages_failed += 1

        job.pages = pages
        job.status = ScrapeStatus.COMPLETED
        job.completed_at = datetime.utcnow()

        logger.info(
            "Scrape job completed",
            job_id=job_id,
            pages_scraped=job.pages_scraped,
            pages_failed=job.pages_failed,
            total_words=total_words,
        )

        return ScrapeResult(
            success=job.pages_scraped > 0,
            job_id=job_id,
            pages_scraped=job.pages_scraped,
            pages_failed=job.pages_failed,
            content=pages,
            total_word_count=total_words,
        )

    async def scrape_url_immediate(
        self,
        url: str,
        config: Optional[ScrapeConfig] = None,
    ) -> ScrapedPage:
        """Scrape a single URL immediately without creating a job."""
        if config is None:
            config = ScrapeConfig()

        return await self._scrape_url(url, config)

    async def scrape_and_query(
        self,
        url: str,
        query: str,
        config: Optional[ScrapeConfig] = None,
    ) -> Dict[str, Any]:
        """
        Scrape a URL and use LLM to answer query with scraped content as context.

        Returns the scraped content along with the LLM-generated answer.
        """
        import time
        from langchain_core.messages import HumanMessage, SystemMessage
        from backend.services.llm import EnhancedLLMFactory

        start_time = time.time()
        page = await self.scrape_url_immediate(url, config)

        # Build prompt with scraped content as context
        system_prompt = """You are a helpful assistant that answers questions based on the provided web page content.
Answer the user's question using ONLY the information from the provided content.
If the answer cannot be found in the content, say so clearly."""

        user_prompt = f"""Web page content from {url}:

{page.content}

---

Question: {query}

Please answer the question based on the web page content above."""

        # Get LLM and generate answer
        llm, llm_config = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="rag",
            track_usage=True,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)

        processing_time = (time.time() - start_time) * 1000

        return {
            "url": url,
            "title": page.title,
            "content": page.content,
            "word_count": page.word_count,
            "scraped_at": page.scraped_at.isoformat(),
            "query": query,
            "answer": answer,
            "model": llm_config.model if llm_config else "default",
            "processing_time_ms": processing_time,
            "context_ready": True,
        }

    def get_job(self, job_id: str) -> Optional[ScrapeJob]:
        """Get a scrape job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        user_id: str,
        status: Optional[ScrapeStatus] = None,
        limit: int = 50,
    ) -> List[ScrapeJob]:
        """List jobs for a user."""
        jobs = [j for j in self._jobs.values() if j.user_id == user_id]

        if status:
            jobs = [j for j in jobs if j.status == status]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def get_cached_page(self, url: str) -> Optional[ScrapedPage]:
        """Get a cached page if available."""
        return self._scraped_urls.get(url)

    def clear_cache(self, max_age_hours: int = 24):
        """Clear old cached pages."""
        now = datetime.utcnow()
        to_remove = []

        for url, page in self._scraped_urls.items():
            age = (now - page.scraped_at).total_seconds() / 3600
            if age > max_age_hours:
                to_remove.append(url)

        for url in to_remove:
            del self._scraped_urls[url]

        logger.info("Cleared cache", removed=len(to_remove))

    async def extract_links(
        self,
        url: str,
        max_depth: int = 1,
        same_domain_only: bool = True,
    ) -> List[str]:
        """Extract links from a page for crawling."""
        page = await self.scrape_url_immediate(url)

        links = []
        base_domain = urlparse(url).netloc

        for link in page.links:
            if same_domain_only:
                link_domain = urlparse(link).netloc
                if link_domain != base_domain:
                    continue
            links.append(link)

        return links[:100]  # Limit to prevent runaway crawling

    async def crawl_with_subpages(
        self,
        start_url: str,
        config: Optional[ScrapeConfig] = None,
        max_depth: int = 2,
        same_domain_only: bool = True,
        rate_limit_seconds: float = 0.5,
    ) -> List[ScrapedPage]:
        """
        Recursively crawl pages following links up to max_depth.

        Args:
            start_url: The starting URL to crawl
            config: Scraping configuration
            max_depth: Maximum depth to crawl (1 = only start page, 2 = start + 1 level, etc.)
            same_domain_only: Only follow links to the same domain
            rate_limit_seconds: Delay between requests to avoid overwhelming servers

        Returns:
            List of scraped pages
        """
        if config is None:
            config = ScrapeConfig()

        visited: Set[str] = set()
        pages: List[ScrapedPage] = []
        base_domain = urlparse(start_url).netloc

        def normalize_url(url: str, base_url: str) -> Optional[str]:
            """Normalize a URL, handling relative paths."""
            if not url:
                return None

            # Skip non-navigable links
            if url.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'data:')):
                return None

            # Handle relative URLs
            if not url.startswith(('http://', 'https://')):
                url = urljoin(base_url, url)

            # Parse and validate
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return None

            if not parsed.netloc:
                return None

            # Remove fragments and normalize (trailing slashes matter for some sites)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                normalized += f"?{parsed.query}"

            return normalized

        async def crawl_recursive(url: str, depth: int):
            """Recursively crawl a URL and its links."""
            # Stop conditions
            if depth > max_depth:
                return
            if url in visited:
                return
            if len(visited) >= config.max_pages:
                logger.info("Reached max pages limit", max_pages=config.max_pages)
                return

            # Check domain restriction
            if same_domain_only:
                url_domain = urlparse(url).netloc
                if url_domain != base_domain:
                    return

            visited.add(url)

            try:
                logger.debug("Crawling page", url=url, depth=depth, visited=len(visited))
                page = await self._scrape_url(url, config)
                pages.append(page)

                # If we haven't reached max depth, crawl child links
                if depth < max_depth and len(visited) < config.max_pages:
                    child_links = []
                    for link in page.links:
                        normalized = normalize_url(link, url)
                        if normalized and normalized not in visited:
                            child_links.append(normalized)

                    logger.debug(
                        "Found child links",
                        url=url,
                        depth=depth,
                        total_links=len(page.links),
                        valid_links=len(child_links),
                    )

                    # Crawl child links with rate limiting
                    for child_url in child_links[:config.max_pages - len(visited)]:
                        if rate_limit_seconds > 0:
                            await asyncio.sleep(rate_limit_seconds)
                        await crawl_recursive(child_url, depth + 1)

            except Exception as e:
                logger.warning("Failed to crawl page", url=url, error=str(e))

        # Start crawling from depth 1
        await crawl_recursive(start_url, 1)

        logger.info(
            "Subpage crawl completed",
            start_url=start_url,
            pages_crawled=len(pages),
            max_depth=max_depth,
        )

        return pages

    async def run_job_with_subpages(
        self,
        job_id: str,
        crawl_subpages: bool = False,
        max_depth: int = 2,
        same_domain_only: bool = True,
    ) -> ScrapeResult:
        """
        Run a scrape job, optionally crawling subpages.

        Args:
            job_id: The job ID to run
            crawl_subpages: Whether to crawl linked pages
            max_depth: Maximum crawl depth if crawl_subpages is True
            same_domain_only: Only crawl same-domain links

        Returns:
            ScrapeResult with all scraped pages
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status not in [ScrapeStatus.PENDING, ScrapeStatus.FAILED]:
            raise ValueError(f"Job is not runnable: {job.status.value}")

        logger.info(
            "Starting scrape job",
            job_id=job_id,
            urls=len(job.urls),
            crawl_subpages=crawl_subpages,
            max_depth=max_depth if crawl_subpages else 1,
        )

        job.status = ScrapeStatus.SCRAPING
        all_pages: List[ScrapedPage] = []
        total_words = 0

        for url in job.urls:
            try:
                if crawl_subpages:
                    # Crawl with subpages
                    pages = await self.crawl_with_subpages(
                        start_url=url,
                        config=job.config,
                        max_depth=max_depth,
                        same_domain_only=same_domain_only,
                    )
                    all_pages.extend(pages)
                    job.pages_scraped += len(pages)
                    for page in pages:
                        total_words += page.word_count
                else:
                    # Single page scrape
                    page = await self._scrape_url(url, job.config)
                    all_pages.append(page)
                    job.pages_scraped += 1
                    total_words += page.word_count

                logger.debug(
                    "Scraped URL",
                    url=url,
                    pages=len(all_pages),
                )

            except Exception as e:
                logger.error("Failed to scrape URL", url=url, error=str(e))
                job.pages_failed += 1

        job.pages = all_pages
        job.status = ScrapeStatus.COMPLETED
        job.completed_at = datetime.utcnow()

        logger.info(
            "Scrape job completed",
            job_id=job_id,
            pages_scraped=job.pages_scraped,
            pages_failed=job.pages_failed,
            total_words=total_words,
        )

        return ScrapeResult(
            success=job.pages_scraped > 0,
            job_id=job_id,
            pages_scraped=job.pages_scraped,
            pages_failed=job.pages_failed,
            content=all_pages,
            total_word_count=total_words,
        )

    def to_rag_documents(
        self,
        pages: List[ScrapedPage],
        chunk_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Convert scraped pages to RAG-ready documents.

        Returns documents in a format suitable for the embedding pipeline.
        """
        documents = []

        for page in pages:
            # Split content into chunks if needed
            content = page.content
            chunks = []

            if len(content) > chunk_size:
                # Simple chunking by paragraphs
                paragraphs = content.split("\n\n")
                current_chunk = ""

                for para in paragraphs:
                    if len(current_chunk) + len(para) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para

                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks = [content]

            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": "web_scrape",
                        "url": page.url,
                        "title": page.title,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "scraped_at": page.scraped_at.isoformat(),
                        "content_hash": page.content_hash,
                        **page.metadata,
                    },
                })

        return documents


# =============================================================================
# Module-level singleton and helpers
# =============================================================================

_scraper_service: Optional[WebScraperService] = None


def get_scraper_service() -> WebScraperService:
    """Get the web scraper service singleton."""
    global _scraper_service
    if _scraper_service is None:
        _scraper_service = WebScraperService()
    return _scraper_service
