"""
AIDocumentIndexer - Web Crawler API Routes (Phase 65)
======================================================

API endpoints for web crawling and content extraction.

Endpoints:
- POST /crawl - Crawl a URL and get content
- POST /crawl/extract - Crawl with LLM extraction
- POST /query - Query a website with a question
- POST /crawl/site - Crawl multiple pages from a site
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field, HttpUrl

import structlog

from backend.services.web_crawler import (
    get_web_crawler,
    CrawlResult,
    WebCrawlerConfig,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/crawler", tags=["crawler"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CrawlRequest(BaseModel):
    """Request to crawl a URL."""
    url: str = Field(..., description="URL to crawl")
    bypass_cache: bool = Field(default=False, description="Skip cache lookup")


class CrawlWithExtractionRequest(BaseModel):
    """Request to crawl with LLM extraction."""
    url: str = Field(..., description="URL to crawl")
    schema: Optional[Dict[str, str]] = Field(
        default=None,
        description="Schema for extraction (field_name: type)",
        example={"title": "str", "author": "str", "date": "str", "content": "str"}
    )
    extraction_prompt: Optional[str] = Field(
        default=None,
        description="Custom extraction prompt"
    )


class WebQueryRequest(BaseModel):
    """Request to query a website."""
    url: str = Field(..., description="URL to query")
    question: str = Field(..., description="Question to answer about the page")


class SiteCrawlRequest(BaseModel):
    """Request to crawl a site."""
    start_url: str = Field(..., description="Starting URL")
    max_pages: int = Field(default=10, ge=1, le=100, description="Max pages to crawl")
    same_domain_only: bool = Field(default=True, description="Only follow internal links")
    url_pattern: Optional[str] = Field(default=None, description="Regex pattern to filter URLs")


class CrawlResponse(BaseModel):
    """Response from crawl endpoint."""
    url: str
    success: bool
    status_code: int = 200
    title: str = ""
    word_count: int = 0
    crawl_time_ms: float = 0.0
    error: Optional[str] = None

    # Content (optional, can be large)
    content: Optional[str] = None
    markdown: Optional[str] = None

    # Extracted data
    extracted: Optional[Dict[str, Any]] = None

    # Links
    link_count: int = 0
    internal_link_count: int = 0
    external_link_count: int = 0

    class Config:
        from_attributes = True


class WebQueryResponse(BaseModel):
    """Response from query endpoint."""
    answer: Optional[str]
    source: str
    title: Optional[str] = None
    word_count: Optional[int] = None
    crawl_time_ms: Optional[float] = None
    error: Optional[str] = None


class SiteCrawlResponse(BaseModel):
    """Response from site crawl."""
    start_url: str
    pages_crawled: int
    pages_successful: int
    total_words: int
    crawl_time_ms: float
    pages: List[CrawlResponse]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/crawl", response_model=CrawlResponse)
async def crawl_url(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl a URL and return the content.

    This endpoint crawls a single URL and returns:
    - Page content (HTML and/or markdown)
    - Title and word count
    - Extracted links (internal and external)

    The crawler uses anti-bot bypass techniques and caches results
    for faster subsequent requests.
    """
    try:
        crawler = get_web_crawler()
        result = await crawler.crawl(
            url=request.url,
            bypass_cache=request.bypass_cache,
        )

        return CrawlResponse(
            url=result.url,
            success=result.success,
            status_code=result.status_code,
            title=result.title,
            word_count=result.word_count,
            crawl_time_ms=result.crawl_time_ms,
            error=result.error,
            content=result.content[:50000] if result.content else None,  # Limit size
            markdown=result.markdown[:50000] if result.markdown else None,
            link_count=len(result.links),
            internal_link_count=len(result.internal_links),
            external_link_count=len(result.external_links),
        )

    except Exception as e:
        logger.error("Crawl endpoint error", url=request.url, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/crawl/extract", response_model=CrawlResponse)
async def crawl_with_extraction(request: CrawlWithExtractionRequest) -> CrawlResponse:
    """
    Crawl a URL and extract structured data using LLM.

    This endpoint crawls a URL and uses an LLM to extract
    structured data based on a provided schema or default extraction.

    Example schema:
    ```json
    {
        "title": "str",
        "author": "str",
        "date": "str",
        "content": "str",
        "tags": "list"
    }
    ```

    The LLM will attempt to extract these fields from the page content.
    """
    try:
        crawler = get_web_crawler()
        result = await crawler.crawl_with_extraction(
            url=request.url,
            schema=request.schema,
            extraction_prompt=request.extraction_prompt,
        )

        return CrawlResponse(
            url=result.url,
            success=result.success,
            status_code=result.status_code,
            title=result.title,
            word_count=result.word_count,
            crawl_time_ms=result.crawl_time_ms,
            error=result.error,
            extracted=result.extracted,
            markdown=result.markdown[:20000] if result.markdown else None,
            link_count=len(result.links),
            internal_link_count=len(result.internal_links),
            external_link_count=len(result.external_links),
        )

    except Exception as e:
        logger.error("Crawl extraction error", url=request.url, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/query", response_model=WebQueryResponse)
async def query_website(request: WebQueryRequest) -> WebQueryResponse:
    """
    Answer a question about a website's content.

    This endpoint crawls the specified URL and uses an LLM to
    answer the provided question based on the page content.

    The answer is grounded in the actual page content - if the
    information isn't found, the response will indicate that.
    """
    try:
        crawler = get_web_crawler()
        result = await crawler.query_website(
            url=request.url,
            question=request.question,
        )

        return WebQueryResponse(
            answer=result.get("answer"),
            source=result.get("source", request.url),
            title=result.get("title"),
            word_count=result.get("word_count"),
            crawl_time_ms=result.get("crawl_time_ms"),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error("Query website error", url=request.url, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/crawl/site", response_model=SiteCrawlResponse)
async def crawl_site(request: SiteCrawlRequest) -> SiteCrawlResponse:
    """
    Crawl multiple pages from a website.

    This endpoint starts from the provided URL and follows links
    to crawl up to max_pages. It respects rate limits and
    can be configured to only follow internal links.

    Use url_pattern to filter which URLs to crawl (regex).
    """
    try:
        import time
        start_time = time.time()

        crawler = get_web_crawler()
        results = await crawler.crawl_site(
            start_url=request.start_url,
            max_pages=request.max_pages,
            same_domain_only=request.same_domain_only,
            url_pattern=request.url_pattern,
        )

        total_time = (time.time() - start_time) * 1000

        # Convert results to response format
        pages = [
            CrawlResponse(
                url=r.url,
                success=r.success,
                status_code=r.status_code,
                title=r.title,
                word_count=r.word_count,
                crawl_time_ms=r.crawl_time_ms,
                error=r.error,
                link_count=len(r.links),
                internal_link_count=len(r.internal_links),
                external_link_count=len(r.external_links),
            )
            for r in results
        ]

        return SiteCrawlResponse(
            start_url=request.start_url,
            pages_crawled=len(results),
            pages_successful=sum(1 for r in results if r.success),
            total_words=sum(r.word_count for r in results),
            crawl_time_ms=total_time,
            pages=pages,
        )

    except Exception as e:
        logger.error("Site crawl error", url=request.start_url, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/status")
async def crawler_status() -> Dict[str, Any]:
    """
    Get the status of the web crawler.

    Returns information about:
    - Crawler availability
    - Cache status
    - Rate limit status
    """
    try:
        from backend.services.web_crawler import HAS_CRAWL4AI

        crawler = get_web_crawler()

        return {
            "status": "available",
            "crawl4ai_available": HAS_CRAWL4AI,
            "cache_enabled": crawler.config.cache_enabled,
            "cache_size": len(crawler._cache),
            "stealth_mode": crawler.config.stealth_mode,
            "llm_extraction_enabled": crawler.config.llm_extraction_enabled,
            "requests_per_minute": crawler.config.requests_per_minute,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
