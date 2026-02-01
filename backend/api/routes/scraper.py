"""
AIDocumentIndexer - Web Scraping API Routes
============================================

Endpoints for web scraping and content extraction.
"""

from datetime import datetime
from typing import Optional, List
from urllib.parse import urlparse
import ipaddress
import socket

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl, field_validator
import json as json_module
import asyncio as asyncio_module
import structlog

from backend.api.middleware.auth import AuthenticatedUser
from backend.services.scraper import (
    WebScraperService,
    ScrapeJob,
    ScrapedPage,
    ScrapeConfig,
    ScrapeStatus,
    StorageMode,
    get_scraper_service,
)
from backend.services.crawl_scheduler import get_scheduler_service

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# SSRF Protection
# =============================================================================

def is_internal_ip(ip_str: str) -> bool:
    """Check if an IP address is internal/private."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private or
            ip.is_loopback or
            ip.is_link_local or
            ip.is_multicast or
            ip.is_reserved or
            ip.is_unspecified
        )
    except ValueError:
        return False


def validate_url_for_ssrf(url: str) -> str:
    """
    Validate URL to prevent SSRF attacks.

    Blocks:
    - Internal IP addresses (10.x.x.x, 172.16.x.x, 192.168.x.x, 127.x.x.x)
    - Localhost variants
    - File:// and other dangerous protocols
    - Cloud metadata endpoints

    Returns the URL if valid, raises ValueError otherwise.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise ValueError(f"Invalid URL format: {url}")

    # Only allow http and https
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only HTTP/HTTPS URLs are allowed, got: {parsed.scheme}")

    # Get hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname")

    # Block localhost variants
    localhost_patterns = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "[::1]",
        "0177.0.0.1",  # Octal
        "2130706433",  # Decimal
        "0x7f000001",  # Hex
    ]
    hostname_lower = hostname.lower()
    if hostname_lower in localhost_patterns or hostname_lower.startswith("127."):
        raise ValueError("Localhost URLs are not allowed")

    # Block cloud metadata endpoints (AWS, GCP, Azure)
    metadata_hostnames = [
        "169.254.169.254",  # AWS/GCP metadata
        "metadata.google.internal",
        "metadata.google.com",
        "100.100.100.200",  # Alibaba Cloud
    ]
    if hostname_lower in metadata_hostnames:
        raise ValueError("Cloud metadata endpoints are not allowed")

    # Try to resolve hostname and check if it resolves to internal IP
    try:
        # Get all IP addresses for the hostname
        resolved_ips = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for result in resolved_ips:
            ip_str = result[4][0]
            if is_internal_ip(ip_str):
                raise ValueError(f"URL resolves to internal IP address: {ip_str}")
    except socket.gaierror:
        # Could not resolve hostname - that's okay, might be valid external domain
        pass
    except ValueError:
        # Re-raise our validation errors
        raise

    return url


# =============================================================================
# Pydantic Models
# =============================================================================

class ScrapeConfigRequest(BaseModel):
    """Configuration for scraping."""
    extract_images: bool = Field(default=False, description="Extract image URLs")
    extract_links: bool = Field(default=True, description="Extract page links")
    extract_metadata: bool = Field(default=True, description="Extract meta tags")
    max_depth: int = Field(default=3, ge=1, le=10, description="Link follow depth")
    max_pages: int = Field(default=50, ge=1, le=200, description="Max pages to scrape")
    timeout: int = Field(default=30, ge=5, le=120, description="Timeout in seconds")
    wait_for_js: bool = Field(default=True, description="Wait for JavaScript")
    crawl_subpages: bool = Field(default=False, description="Whether to crawl linked subpages")
    same_domain_only: bool = Field(default=True, description="Only crawl links from the same domain")


class CreateScrapeJobRequest(BaseModel):
    """Request to create a scrape job."""
    urls: List[str] = Field(..., min_length=1, max_length=20, description="URLs to scrape")
    storage_mode: str = Field(default="immediate", description="immediate or permanent")
    config: Optional[ScrapeConfigRequest] = None
    access_tier: int = Field(default=1, ge=1, le=100, description="Access tier for stored content")
    collection: Optional[str] = Field(None, description="Collection to store in")
    crawl_subpages: bool = Field(default=False, description="Whether to crawl linked subpages")
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum depth for subpage crawling")
    same_domain_only: bool = Field(default=True, description="Only crawl links from the same domain")

    @field_validator('urls')
    @classmethod
    def validate_urls(cls, v: List[str]) -> List[str]:
        """Validate all URLs for SSRF protection."""
        validated = []
        for url in v:
            try:
                validated.append(validate_url_for_ssrf(url))
            except ValueError as e:
                raise ValueError(f"Invalid URL '{url}': {e}")
        return validated


class ScrapeUrlRequest(BaseModel):
    """Request to scrape a single URL immediately."""
    url: str = Field(..., description="URL to scrape")
    config: Optional[ScrapeConfigRequest] = None

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL for SSRF protection."""
        return validate_url_for_ssrf(v)


class ScrapeAndQueryRequest(BaseModel):
    """Request to scrape and query."""
    url: str = Field(..., description="URL to scrape")
    query: str = Field(..., min_length=3, description="Query to answer using scraped content")
    config: Optional[ScrapeConfigRequest] = None

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL for SSRF protection."""
        return validate_url_for_ssrf(v)


class ScrapedPageResponse(BaseModel):
    """Scraped page response."""
    url: str
    title: str
    content: str
    word_count: int
    links_count: int
    images_count: int
    scraped_at: datetime
    metadata: dict


class ScrapeJobResponse(BaseModel):
    """Scrape job response."""
    id: str
    user_id: str
    status: str
    storage_mode: str
    total_pages: int
    pages_scraped: int
    pages_failed: int
    pages: List[ScrapedPageResponse]
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    crawl_subpages: bool = False
    max_depth: int = 1
    same_domain_only: bool = True


class ScrapeJobListResponse(BaseModel):
    """List of scrape jobs."""
    jobs: List[ScrapeJobResponse]
    total: int


class IndexJobResponse(BaseModel):
    """Response for indexing a job's content."""
    status: str
    documents_indexed: int = 0
    entities_extracted: int = 0
    chunks_processed: int = 0
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def page_to_response(page: ScrapedPage) -> ScrapedPageResponse:
    """Convert ScrapedPage to response model."""
    return ScrapedPageResponse(
        url=page.url,
        title=page.title,
        content=page.content,
        word_count=page.word_count,
        links_count=len(page.links),
        images_count=len(page.images),
        scraped_at=page.scraped_at,
        metadata=page.metadata,
    )


def job_to_response(
    job: ScrapeJob,
    crawl_subpages: bool = False,
    max_depth: int = 1,
    same_domain_only: bool = True,
) -> ScrapeJobResponse:
    """Convert ScrapeJob to response model."""
    return ScrapeJobResponse(
        id=job.id,
        user_id=job.user_id,
        status=job.status.value,
        storage_mode=job.storage_mode.value,
        total_pages=job.total_pages,
        pages_scraped=job.pages_scraped,
        pages_failed=job.pages_failed,
        pages=[page_to_response(p) for p in job.pages],
        created_at=job.created_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        crawl_subpages=crawl_subpages,
        max_depth=max_depth,
        same_domain_only=same_domain_only,
    )


def request_to_config(request: Optional[ScrapeConfigRequest]) -> ScrapeConfig:
    """Convert request config to ScrapeConfig."""
    if request is None:
        return ScrapeConfig()

    return ScrapeConfig(
        extract_images=request.extract_images,
        extract_links=request.extract_links,
        extract_metadata=request.extract_metadata,
        max_depth=request.max_depth,
        max_pages=request.max_pages,
        timeout=request.timeout,
        wait_for_js=request.wait_for_js,
    )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/jobs", response_model=ScrapeJobResponse, status_code=status.HTTP_201_CREATED)
async def create_scrape_job(
    request: CreateScrapeJobRequest,
    user: AuthenticatedUser,
):
    """
    Create a new web scraping job.

    Storage modes:
    - immediate: Scraped content is returned but not stored permanently
    - permanent: Content is stored in the RAG knowledge base for future queries
    """
    logger.info(
        "Creating scrape job",
        user_id=user.user_id,
        urls=len(request.urls),
        storage_mode=request.storage_mode,
    )

    # Parse storage mode
    try:
        storage_mode = StorageMode(request.storage_mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid storage mode: {request.storage_mode}. Use 'immediate' or 'permanent'.",
        )

    service = get_scraper_service()

    try:
        job = await service.create_job(
            user_id=user.user_id,
            urls=request.urls,
            storage_mode=storage_mode,
            config=request_to_config(request.config),
            access_tier=request.access_tier,
            collection=request.collection,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return job_to_response(job)


@router.get("/jobs", response_model=ScrapeJobListResponse)
async def list_scrape_jobs(
    user: AuthenticatedUser,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(default=50, ge=1, le=100),
):
    """
    List scrape jobs for the current user.
    """
    service = get_scraper_service()

    # Parse status filter
    scrape_status = None
    if status_filter:
        try:
            scrape_status = ScrapeStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    jobs = service.list_jobs(
        user_id=user.user_id,
        status=scrape_status,
        limit=limit,
    )

    return ScrapeJobListResponse(
        jobs=[job_to_response(j) for j in jobs],
        total=len(jobs),
    )


@router.get("/jobs/{job_id}", response_model=ScrapeJobResponse)
async def get_scrape_job(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Get a specific scrape job.
    """
    service = get_scraper_service()

    job = service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    # Verify ownership
    if job.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    return job_to_response(job)


class RunScrapeJobRequest(BaseModel):
    """Request to run a scrape job with options."""
    crawl_subpages: bool = Field(default=False, description="Whether to crawl linked subpages")
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum depth for subpage crawling")
    same_domain_only: bool = Field(default=True, description="Only crawl links from the same domain")


@router.post("/jobs/{job_id}/run", response_model=ScrapeJobResponse)
async def run_scrape_job(
    job_id: str,
    user: AuthenticatedUser,
    request: Optional[RunScrapeJobRequest] = None,
):
    """
    Run a pending scrape job.

    If crawl_subpages is True, the scraper will follow links from each URL
    up to max_depth levels deep, only crawling pages from the same domain
    if same_domain_only is True.
    """
    crawl_subpages = request.crawl_subpages if request else False
    max_depth = request.max_depth if request else 2
    same_domain_only = request.same_domain_only if request else True

    logger.info(
        "Running scrape job",
        job_id=job_id,
        crawl_subpages=crawl_subpages,
        max_depth=max_depth,
    )

    service = get_scraper_service()

    job = service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    # Verify ownership
    if job.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        result = await service.run_job_with_subpages(
            job_id=job_id,
            crawl_subpages=crawl_subpages,
            max_depth=max_depth,
            same_domain_only=same_domain_only,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Scrape job failed", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scraping failed: {str(e)}",
        )

    # Get updated job
    job = service.get_job(job_id)
    return job_to_response(job, crawl_subpages, max_depth, same_domain_only)


class ScrapedPagesResponse(BaseModel):
    """Response for multiple scraped pages."""
    pages: List[ScrapedPageResponse]
    total_pages: int
    total_word_count: int


@router.post("/scrape")
async def scrape_url_immediate(
    request: ScrapeUrlRequest,
    user: AuthenticatedUser,
):
    """
    Scrape a single URL immediately.

    Returns the scraped content without creating a job.
    If crawl_subpages is enabled in config, returns multiple pages.
    """
    config = request.config
    crawl_subpages = config.crawl_subpages if config else False
    max_depth = config.max_depth if config else 3
    same_domain_only = config.same_domain_only if config else True

    logger.info(
        "Scraping URL",
        url=request.url,
        user_id=user.user_id,
        crawl_subpages=crawl_subpages,
        max_depth=max_depth,
    )

    service = get_scraper_service()

    try:
        if crawl_subpages:
            # Crawl with subpages
            pages = await service.crawl_with_subpages(
                start_url=request.url,
                config=request_to_config(config),
                max_depth=max_depth,
                same_domain_only=same_domain_only,
            )
            total_words = sum(p.word_count for p in pages)
            return ScrapedPagesResponse(
                pages=[page_to_response(p) for p in pages],
                total_pages=len(pages),
                total_word_count=total_words,
            )
        else:
            # Single page scrape
            page = await service.scrape_url_immediate(
                url=request.url,
                config=request_to_config(config),
            )
            return page_to_response(page)
    except Exception as e:
        logger.error("Scrape failed", url=request.url, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scrape URL: {str(e)}",
        )


class IndexPagesRequest(BaseModel):
    """Request to index scraped pages directly."""
    pages: List[dict] = Field(..., description="List of scraped page objects to index")
    source_id: Optional[str] = Field(None, description="Optional source identifier")


@router.post("/index-pages", response_model=IndexJobResponse)
async def index_pages_directly(
    request: IndexPagesRequest,
    user: AuthenticatedUser,
):
    """
    Index scraped pages directly into the RAG pipeline.

    This allows content from a quick scrape (immediate mode) to be
    permanently indexed without creating a full scrape job first.
    """
    from datetime import datetime as dt

    logger.info(
        "Indexing pages directly",
        user_id=user.user_id,
        pages_count=len(request.pages),
    )

    if not request.pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pages provided",
        )

    service = get_scraper_service()

    # Convert dict pages to ScrapedPage objects
    scraped_pages = []
    for page_data in request.pages:
        scraped_pages.append(ScrapedPage(
            url=page_data.get("url", ""),
            title=page_data.get("title", ""),
            content=page_data.get("content", ""),
            metadata=page_data.get("metadata", {}),
            word_count=page_data.get("word_count", 0),
            scraped_at=dt.fromisoformat(page_data["scraped_at"]) if page_data.get("scraped_at") else dt.utcnow(),
        ))

    try:
        result = await service.index_pages_content(
            pages=scraped_pages,
            source_id=request.source_id,
        )

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to index content"),
            )

        return IndexJobResponse(
            status=result.get("status", "success"),
            documents_indexed=result.get("documents_indexed", 0),
            entities_extracted=result.get("entities_extracted", 0),
            chunks_processed=result.get("chunks_processed", 0),
            error=result.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to index pages", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index content: {str(e)}",
        )


@router.post("/scrape-and-query")
async def scrape_and_query(
    request: ScrapeAndQueryRequest,
    user: AuthenticatedUser,
):
    """
    Scrape a URL and use it as context for a query.

    This endpoint scrapes the URL and returns the content
    ready to be used as context for a RAG query.
    If crawl_subpages is enabled, multiple pages are scraped and combined.
    """
    config = request.config
    crawl_subpages = config.crawl_subpages if config else False
    max_depth = config.max_depth if config else 3
    same_domain_only = config.same_domain_only if config else True

    logger.info(
        "Scrape and query",
        url=request.url,
        query=request.query[:50],
        user_id=user.user_id,
        crawl_subpages=crawl_subpages,
        max_depth=max_depth,
    )

    service = get_scraper_service()

    try:
        if crawl_subpages:
            # Crawl multiple pages and combine content for query
            pages = await service.crawl_with_subpages(
                start_url=request.url,
                config=request_to_config(config),
                max_depth=max_depth,
                same_domain_only=same_domain_only,
            )

            # Combine content from all pages
            combined_content = "\n\n---\n\n".join([
                f"## {p.title}\nURL: {p.url}\n\n{p.content}"
                for p in pages
            ])
            total_words = sum(p.word_count for p in pages)

            # Use combined content for query
            import time
            from langchain_core.messages import HumanMessage, SystemMessage
            from backend.services.llm import EnhancedLLMFactory

            start_time = time.time()

            system_prompt = """You are a helpful assistant that answers questions based on the provided web page content.
Answer the user's question using ONLY the information from the provided content.
If the answer cannot be found in the content, say so clearly.
When citing information, mention which page it came from."""

            user_prompt = f"""Web content from {len(pages)} pages starting at {request.url}:

{combined_content[:50000]}

---

Question: {request.query}

Please answer the question based on the web content above."""

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
                "url": request.url,
                "title": pages[0].title if pages else "Unknown",
                "content": combined_content[:5000] + "..." if len(combined_content) > 5000 else combined_content,
                "word_count": total_words,
                "scraped_at": pages[0].scraped_at.isoformat() if pages else None,
                "query": request.query,
                "answer": answer,
                "model": llm_config.model if llm_config else "default",
                "processing_time_ms": processing_time,
                "context_ready": True,
                "pages_scraped": len(pages),
            }
        else:
            # Single page scrape and query
            result = await service.scrape_and_query(
                url=request.url,
                query=request.query,
                config=request_to_config(config),
            )
            return result
    except Exception as e:
        logger.error("Scrape and query failed", url=request.url, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scrape and query: {str(e)}",
        )


@router.get("/cache")
async def get_cached_page(
    user: AuthenticatedUser,
    url: str = Query(..., description="URL to check in cache"),
):
    """
    Get a cached scraped page if available.
    """
    service = get_scraper_service()

    page = service.get_cached_page(url)
    if not page:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="URL not in cache",
        )

    return page_to_response(page)


@router.delete("/cache")
async def clear_cache(
    user: AuthenticatedUser,
    max_age_hours: int = Query(default=24, ge=1, le=168),
):
    """
    Clear old cached pages.

    Admin only.
    """
    if not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    service = get_scraper_service()
    service.clear_cache(max_age_hours=max_age_hours)

    return {"message": f"Cleared cache entries older than {max_age_hours} hours"}


@router.post("/extract-links")
async def extract_links(
    user: AuthenticatedUser,
    url: str = Query(..., description="URL to extract links from"),
    max_depth: int = Query(default=3, ge=1, le=10),
    same_domain_only: bool = Query(default=True),
):
    """
    Extract links from a page.

    Useful for discovering pages to scrape.
    """
    service = get_scraper_service()

    try:
        links = await service.extract_links(
            url=url,
            max_depth=max_depth,
            same_domain_only=same_domain_only,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract links: {str(e)}",
        )

    return {
        "url": url,
        "links": links,
        "count": len(links),
    }


@router.post("/jobs/{job_id}/index", response_model=IndexJobResponse)
async def index_job_content(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Index an existing completed job's content into the RAG pipeline.

    This allows content that was scraped with 'immediate' mode to be
    permanently indexed later for future RAG queries. The content will be:
    - Chunked and embedded
    - Indexed in the vector store
    - Processed for Knowledge Graph entity extraction

    Use this endpoint when you want to save previously scraped content
    for future searches.
    """
    logger.info(
        "Indexing job content",
        job_id=job_id,
        user_id=user.user_id,
    )

    service = get_scraper_service()

    job = service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    # Verify ownership
    if job.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    try:
        result = await service.index_job_content(job_id)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to index content"),
            )

        return IndexJobResponse(
            status=result.get("status", "success"),
            documents_indexed=result.get("documents_indexed", 0),
            entities_extracted=result.get("entities_extracted", 0),
            chunks_processed=result.get("chunks_processed", 0),
            error=result.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to index job content", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index content: {str(e)}",
        )


@router.get("/jobs/{job_id}/documents")
async def get_job_documents(
    job_id: str,
    user: AuthenticatedUser,
    chunk_size: int = Query(default=1000, ge=200, le=4000),
):
    """
    Get scraped content as RAG-ready documents.

    Returns chunked documents suitable for embedding and storage.
    """
    service = get_scraper_service()

    job = service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    # Verify ownership
    if job.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    if job.status != ScrapeStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed: {job.status.value}",
        )

    documents = service.to_rag_documents(job.pages, chunk_size=chunk_size)

    return {
        "job_id": job_id,
        "documents": documents,
        "total": len(documents),
    }


# =============================================================================
# Sitemap Crawling
# =============================================================================

class SitemapCrawlRequest(BaseModel):
    """Request to crawl a site via its sitemap."""
    url: str = Field(..., description="Base URL of the site (e.g., https://example.com)")
    max_pages: int = Field(default=50, ge=1, le=500, description="Maximum pages to crawl from sitemap")
    storage_mode: str = Field(default="permanent", description="immediate or permanent")
    access_tier: int = Field(default=1, ge=1, le=100, description="Access tier for stored content")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        return validate_url_for_ssrf(v)


@router.post("/sitemap-crawl")
async def sitemap_crawl(
    request: SitemapCrawlRequest,
    user: AuthenticatedUser,
):
    """
    Crawl a website using its sitemap.xml for URL discovery.

    Fetches the site's sitemap.xml, extracts URLs, and crawls them.
    URLs are prioritized by lastmod date (newest first).
    Results can be stored permanently in the RAG knowledge base.
    """
    from backend.services.web_crawler import get_web_crawler

    logger.info(
        "Sitemap crawl requested",
        url=request.url,
        max_pages=request.max_pages,
        user_id=user.user_id,
    )

    try:
        crawler = get_web_crawler()
        results = await crawler.crawl_sitemap(
            url=request.url,
            max_pages=request.max_pages,
        )

        # If permanent storage requested, index the content
        if request.storage_mode == "permanent" and results:
            service = get_scraper_service()
            from backend.services.scraper import ScrapedPage, ScrapeConfig, StorageMode, ScrapeJob, ScrapeStatus
            from uuid import uuid4

            pages = []
            for r in results:
                if r.success:
                    pages.append(ScrapedPage(
                        url=r.url,
                        title=r.title,
                        content=r.markdown or r.content,
                        word_count=r.word_count,
                    ))

            if pages:
                result = await service.index_pages_content(
                    pages=pages,
                    source_id=f"sitemap_{urlparse(request.url).netloc}",
                )

        return {
            "url": request.url,
            "pages_found": len(results),
            "pages_successful": sum(1 for r in results if r.success),
            "total_words": sum(r.word_count for r in results if r.success),
            "storage_mode": request.storage_mode,
            "pages": [
                {
                    "url": r.url,
                    "title": r.title,
                    "success": r.success,
                    "word_count": r.word_count,
                    "error": r.error,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.error("Sitemap crawl failed", url=request.url, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sitemap crawl failed: {str(e)}",
        )


# =============================================================================
# SSE Progress Streaming
# =============================================================================

@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(
    job_id: str,
    user: AuthenticatedUser,
):
    """
    Stream real-time progress of a scrape job via Server-Sent Events (SSE).

    Events include: status updates, page completions, errors, and final summary.
    Connect using EventSource in the browser.
    """
    service = get_scraper_service()
    job = service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    if job.user_id != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this job",
        )

    async def event_generator():
        """Generate SSE events for job progress."""
        last_pages_scraped = 0
        last_status = None

        while True:
            current_job = service.get_job(job_id)
            if not current_job:
                yield f"data: {json_module.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                break

            # Send status update if changed
            if current_job.status != last_status:
                last_status = current_job.status
                yield f"data: {json_module.dumps({'type': 'status', 'status': current_job.status.value, 'job_id': job_id})}\n\n"

            # Send page progress if new pages scraped
            if current_job.pages_scraped > last_pages_scraped:
                for page in current_job.pages[last_pages_scraped:]:
                    yield f"data: {json_module.dumps({'type': 'page_complete', 'url': page.url, 'title': page.title, 'word_count': page.word_count, 'pages_scraped': current_job.pages_scraped, 'total_pages': current_job.total_pages})}\n\n"
                last_pages_scraped = current_job.pages_scraped

            # Check if job is done
            if current_job.status in [ScrapeStatus.COMPLETED, ScrapeStatus.FAILED]:
                summary = {
                    'type': 'complete',
                    'status': current_job.status.value,
                    'pages_scraped': current_job.pages_scraped,
                    'pages_failed': current_job.pages_failed,
                    'total_words': sum(p.word_count for p in current_job.pages),
                }
                yield f"data: {json_module.dumps(summary)}\n\n"
                break

            await asyncio_module.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Search-based Crawling
# =============================================================================

class SearchCrawlRequest(BaseModel):
    """Request to search for and crawl pages."""
    query: str = Field(..., min_length=3, description="Search query for finding relevant pages")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum search results to crawl")
    storage_mode: str = Field(default="immediate", description="immediate or permanent")
    access_tier: int = Field(default=1, ge=1, le=100, description="Access tier for stored content")


@router.post("/search-crawl")
async def search_and_crawl(
    request: SearchCrawlRequest,
    user: AuthenticatedUser,
):
    """
    Search the web for relevant pages and crawl them.

    Uses DuckDuckGo to find pages matching the query, then crawls
    each result to extract content. Optionally stores results in
    the RAG knowledge base for future queries.
    """
    from backend.services.web_crawler import get_web_crawler

    logger.info(
        "Search and crawl requested",
        query=request.query,
        max_results=request.max_results,
        user_id=user.user_id,
    )

    try:
        crawler = get_web_crawler()
        results = await crawler.search_and_crawl(
            query=request.query,
            max_results=request.max_results,
        )

        # If permanent storage requested, index the content
        if request.storage_mode == "permanent" and results:
            service = get_scraper_service()
            from backend.services.scraper import ScrapedPage

            pages = []
            for r in results:
                if r.success:
                    pages.append(ScrapedPage(
                        url=r.url,
                        title=r.title,
                        content=r.markdown or r.content,
                        word_count=r.word_count,
                    ))

            if pages:
                await service.index_pages_content(
                    pages=pages,
                    source_id=f"search_{request.query[:50].replace(' ', '_')}",
                )

        return {
            "query": request.query,
            "results_found": len(results),
            "results_successful": sum(1 for r in results if r.success),
            "total_words": sum(r.word_count for r in results if r.success),
            "storage_mode": request.storage_mode,
            "results": [
                {
                    "url": r.url,
                    "title": r.title,
                    "success": r.success,
                    "word_count": r.word_count,
                    "error": r.error,
                    "snippet": (r.markdown or "")[:200] if r.success else None,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.error("Search and crawl failed", query=request.query, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search and crawl failed: {str(e)}",
        )


# =============================================================================
# Robots.txt Checking
# =============================================================================

@router.get("/robots-txt")
async def check_robots_txt(
    user: AuthenticatedUser,
    url: str = Query(..., description="URL to check robots.txt for"),
):
    """
    Parse and return robots.txt rules for a domain.

    Returns allowed/disallowed paths, crawl delay, and sitemap URLs.
    """
    from backend.services.web_crawler import get_web_crawler

    try:
        crawler = get_web_crawler()
        rules = await crawler.parse_robots_txt(url)
        return rules
    except Exception as e:
        logger.error("robots.txt check failed", url=url, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse robots.txt: {str(e)}",
        )


# =============================================================================
# Scheduled Crawl Pydantic Models
# =============================================================================

class ScheduledCrawlCreate(BaseModel):
    """Request to create a scheduled crawl."""
    url: str = Field(..., description="URL to crawl on schedule")
    schedule: str = Field(
        ...,
        description="Cron expression (e.g. '0 */6 * * *' for every 6 hours)",
    )
    crawl_config: dict = Field(
        default_factory=lambda: {"max_pages": 50, "max_depth": 3, "storage_mode": "permanent"},
        description="Crawl configuration (max_pages, max_depth, storage_mode, etc.)",
    )
    enabled: bool = Field(default=True, description="Whether the schedule is active")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL for SSRF protection."""
        return validate_url_for_ssrf(v)


class ScheduledCrawlUpdate(BaseModel):
    """Request to update a scheduled crawl (partial update)."""
    url: Optional[str] = Field(None, description="Updated URL")
    schedule: Optional[str] = Field(None, description="Updated cron expression")
    crawl_config: Optional[dict] = Field(None, description="Updated crawl configuration")
    enabled: Optional[bool] = Field(None, description="Enable or disable the schedule")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate URL for SSRF protection."""
        if v is not None:
            return validate_url_for_ssrf(v)
        return v


class ScheduledCrawlResponse(BaseModel):
    """Response model for a scheduled crawl."""
    id: str
    url: str
    schedule: str
    crawl_config: dict
    enabled: bool
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_content_hash: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None


class ScheduledCrawlListResponse(BaseModel):
    """Response model for listing scheduled crawls."""
    schedules: List[ScheduledCrawlResponse]
    total: int


# =============================================================================
# Scheduled Crawl API Endpoints
# =============================================================================

@router.post("/scheduled", response_model=ScheduledCrawlResponse, status_code=status.HTTP_201_CREATED)
async def create_scheduled_crawl(
    request: ScheduledCrawlCreate,
    user: AuthenticatedUser,
):
    """
    Create a new scheduled/recurring crawl.

    Registers a periodic crawl task that runs on the specified cron schedule.
    Uses Celery Beat for dispatch. Content is hashed between runs to detect
    changes and avoid redundant re-indexing.
    """
    logger.info(
        "Creating scheduled crawl",
        url=request.url,
        schedule=request.schedule,
        user_id=user.user_id,
    )

    scheduler = get_scheduler_service()

    try:
        scheduled_crawl = scheduler.create_schedule(
            url=request.url,
            schedule=request.schedule,
            config=request.crawl_config,
            user_id=user.user_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return ScheduledCrawlResponse(**scheduled_crawl.to_dict())


@router.get("/scheduled", response_model=ScheduledCrawlListResponse)
async def list_scheduled_crawls(
    user: AuthenticatedUser,
):
    """
    List all scheduled crawls.

    Returns all scheduled crawl configurations for the current user.
    Admins see all schedules.
    """
    scheduler = get_scheduler_service()

    if user.is_admin():
        schedules = scheduler.list_schedules()
    else:
        schedules = scheduler.list_schedules(user_id=user.user_id)

    return ScheduledCrawlListResponse(
        schedules=[ScheduledCrawlResponse(**s.to_dict()) for s in schedules],
        total=len(schedules),
    )


@router.get("/scheduled/{schedule_id}", response_model=ScheduledCrawlResponse)
async def get_scheduled_crawl(
    schedule_id: str,
    user: AuthenticatedUser,
):
    """
    Get a specific scheduled crawl by ID.
    """
    scheduler = get_scheduler_service()

    scheduled_crawl = scheduler.get_schedule(schedule_id)
    if not scheduled_crawl:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scheduled crawl not found: {schedule_id}",
        )

    # Verify ownership (admins can access any schedule)
    if scheduled_crawl.created_by != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this scheduled crawl",
        )

    return ScheduledCrawlResponse(**scheduled_crawl.to_dict())


@router.put("/scheduled/{schedule_id}", response_model=ScheduledCrawlResponse)
async def update_scheduled_crawl(
    schedule_id: str,
    request: ScheduledCrawlUpdate,
    user: AuthenticatedUser,
):
    """
    Update an existing scheduled crawl.

    Supports partial updates. If the cron schedule or enabled state changes,
    the Celery Beat registration is updated accordingly.
    """
    scheduler = get_scheduler_service()

    # Check existence and ownership
    existing = scheduler.get_schedule(schedule_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scheduled crawl not found: {schedule_id}",
        )

    if existing.created_by != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this scheduled crawl",
        )

    # Build updates dict from non-None fields
    updates = {k: v for k, v in request.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    try:
        updated_crawl = scheduler.update_schedule(schedule_id, updates)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    logger.info(
        "Updated scheduled crawl",
        schedule_id=schedule_id,
        updates=list(updates.keys()),
        user_id=user.user_id,
    )

    return ScheduledCrawlResponse(**updated_crawl.to_dict())


@router.delete("/scheduled/{schedule_id}")
async def delete_scheduled_crawl(
    schedule_id: str,
    user: AuthenticatedUser,
):
    """
    Delete a scheduled crawl.

    Removes the schedule and unregisters the corresponding Celery Beat task.
    """
    scheduler = get_scheduler_service()

    # Check existence and ownership
    existing = scheduler.get_schedule(schedule_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scheduled crawl not found: {schedule_id}",
        )

    if existing.created_by != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this scheduled crawl",
        )

    deleted = scheduler.delete_schedule(schedule_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete scheduled crawl",
        )

    logger.info(
        "Deleted scheduled crawl",
        schedule_id=schedule_id,
        user_id=user.user_id,
    )

    return {"message": f"Scheduled crawl {schedule_id} deleted successfully"}


@router.post("/scheduled/{schedule_id}/run")
async def run_scheduled_crawl(
    schedule_id: str,
    user: AuthenticatedUser,
):
    """
    Manually trigger a scheduled crawl execution.

    Runs the crawl immediately regardless of the cron schedule. Computes
    a content hash and re-indexes only if content has changed since the
    last execution.
    """
    scheduler = get_scheduler_service()

    # Check existence and ownership
    existing = scheduler.get_schedule(schedule_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scheduled crawl not found: {schedule_id}",
        )

    if existing.created_by != user.user_id and not user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this scheduled crawl",
        )

    logger.info(
        "Manually triggering scheduled crawl",
        schedule_id=schedule_id,
        user_id=user.user_id,
    )

    try:
        result = await scheduler.execute_scheduled_crawl(schedule_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Scheduled crawl execution failed",
            schedule_id=schedule_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scheduled crawl execution failed: {str(e)}",
        )

    return result
