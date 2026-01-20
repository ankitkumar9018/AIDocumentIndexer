"""
AIDocumentIndexer - Web Scraping API Routes
============================================

Endpoints for web scraping and content extraction.
"""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, HttpUrl
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

logger = structlog.get_logger(__name__)

router = APIRouter()


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


class ScrapeUrlRequest(BaseModel):
    """Request to scrape a single URL immediately."""
    url: str = Field(..., description="URL to scrape")
    config: Optional[ScrapeConfigRequest] = None


class ScrapeAndQueryRequest(BaseModel):
    """Request to scrape and query."""
    url: str = Field(..., description="URL to scrape")
    query: str = Field(..., min_length=3, description="Query to answer using scraped content")
    config: Optional[ScrapeConfigRequest] = None


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


class IndexJobResponse(BaseModel):
    """Response for indexing a job's content."""
    status: str
    documents_indexed: int = 0
    entities_extracted: int = 0
    chunks_processed: int = 0
    error: Optional[str] = None


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
