"""
AIDocumentIndexer - Reports (Sparkpages) API Routes
====================================================

Endpoints for AI-generated reports with citations.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.db.database import get_async_session
from backend.db.models import Report, ReportStatus
from backend.api.middleware.auth import AuthenticatedUser

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class ReportSection(BaseModel):
    """A section within a report."""
    id: str
    title: str
    content: str
    order: int = 0
    citations: List[str] = Field(default_factory=list)


class ReportCitation(BaseModel):
    """A citation reference."""
    id: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    title: str
    content: str
    source: Optional[str] = None
    page: Optional[int] = None


class ReportCreateRequest(BaseModel):
    """Request to create a new report."""
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    sections: List[ReportSection] = Field(default_factory=list)
    citations: List[ReportCitation] = Field(default_factory=list)


class ReportUpdateRequest(BaseModel):
    """Request to update a report."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    status: Optional[str] = None
    sections: Optional[List[ReportSection]] = None
    citations: Optional[List[ReportCitation]] = None
    is_starred: Optional[bool] = None
    is_public: Optional[bool] = None


class ReportResponse(BaseModel):
    """Report response model."""
    id: str
    title: str
    description: Optional[str]
    status: str
    sections: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    section_count: int
    citation_count: int
    is_starred: bool
    is_public: bool
    view_count: int
    thumbnail_url: Optional[str]
    created_at: datetime
    updated_at: datetime


class ReportListResponse(BaseModel):
    """Report list item response."""
    id: str
    title: str
    description: Optional[str]
    status: str
    section_count: int
    citation_count: int
    is_starred: bool
    thumbnail_url: Optional[str]
    created_at: datetime
    updated_at: datetime


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("", response_model=ReportResponse)
async def create_report(
    request: ReportCreateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Create a new report."""
    logger.info("Creating report", title=request.title, user_id=user.user_id)

    report = Report(
        user_id=user.user_id,
        title=request.title,
        description=request.description,
        status=ReportStatus.DRAFT.value,
        sections=[s.model_dump() for s in request.sections],
        citations=[c.model_dump() for c in request.citations],
        section_count=len(request.sections),
        citation_count=len(request.citations),
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)

    return ReportResponse(
        id=str(report.id),
        title=report.title,
        description=report.description,
        status=report.status,
        sections=report.sections or [],
        citations=report.citations or [],
        section_count=report.section_count,
        citation_count=report.citation_count,
        is_starred=report.is_starred,
        is_public=report.is_public,
        view_count=report.view_count,
        thumbnail_url=report.thumbnail_url,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


@router.get("/list")
async def list_reports(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    status: Optional[str] = Query(None, description="Filter by status"),
    starred: Optional[bool] = Query(None, description="Filter by starred"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List user's reports."""
    conditions = [Report.user_id == user.user_id]

    if status:
        conditions.append(Report.status == status)

    if starred is not None:
        conditions.append(Report.is_starred == starred)

    if search:
        search_pattern = f"%{search}%"
        conditions.append(
            or_(
                Report.title.ilike(search_pattern),
                Report.description.ilike(search_pattern),
            )
        )

    # Get total count
    count_query = select(func.count(Report.id)).where(and_(*conditions))
    total = (await db.execute(count_query)).scalar() or 0

    # Get reports
    query = (
        select(Report)
        .where(and_(*conditions))
        .order_by(Report.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    reports = result.scalars().all()

    return {
        "reports": [
            {
                "id": str(r.id),
                "title": r.title,
                "description": r.description,
                "status": r.status,
                "section_count": r.section_count,
                "citation_count": r.citation_count,
                "is_starred": r.is_starred,
                "thumbnail_url": r.thumbnail_url,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
            }
            for r in reports
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Get a specific report."""
    try:
        report_uuid = UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report ID")

    result = await db.execute(
        select(Report).where(
            Report.id == report_uuid,
            or_(Report.user_id == user.user_id, Report.is_public == True),
        )
    )
    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Increment view count if not owner
    if report.user_id != user.user_id:
        report.view_count += 1
        await db.commit()

    return ReportResponse(
        id=str(report.id),
        title=report.title,
        description=report.description,
        status=report.status,
        sections=report.sections or [],
        citations=report.citations or [],
        section_count=report.section_count,
        citation_count=report.citation_count,
        is_starred=report.is_starred,
        is_public=report.is_public,
        view_count=report.view_count,
        thumbnail_url=report.thumbnail_url,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


@router.put("/{report_id}", response_model=ReportResponse)
async def update_report(
    report_id: str,
    request: ReportUpdateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Update a report."""
    try:
        report_uuid = UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report ID")

    result = await db.execute(
        select(Report).where(Report.id == report_uuid, Report.user_id == user.user_id)
    )
    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Update fields
    if request.title is not None:
        report.title = request.title
    if request.description is not None:
        report.description = request.description
    if request.status is not None:
        if request.status not in [s.value for s in ReportStatus]:
            raise HTTPException(status_code=400, detail=f"Invalid status: {request.status}")
        report.status = request.status
    if request.sections is not None:
        report.sections = [s.model_dump() for s in request.sections]
        report.section_count = len(request.sections)
    if request.citations is not None:
        report.citations = [c.model_dump() for c in request.citations]
        report.citation_count = len(request.citations)
    if request.is_starred is not None:
        report.is_starred = request.is_starred
    if request.is_public is not None:
        report.is_public = request.is_public

    await db.commit()
    await db.refresh(report)

    return ReportResponse(
        id=str(report.id),
        title=report.title,
        description=report.description,
        status=report.status,
        sections=report.sections or [],
        citations=report.citations or [],
        section_count=report.section_count,
        citation_count=report.citation_count,
        is_starred=report.is_starred,
        is_public=report.is_public,
        view_count=report.view_count,
        thumbnail_url=report.thumbnail_url,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


@router.patch("/{report_id}/star")
async def toggle_star(
    report_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Toggle report starred status."""
    try:
        report_uuid = UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report ID")

    result = await db.execute(
        select(Report).where(Report.id == report_uuid, Report.user_id == user.user_id)
    )
    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    report.is_starred = not report.is_starred
    await db.commit()

    return {"id": str(report.id), "is_starred": report.is_starred}


@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Delete a report."""
    try:
        report_uuid = UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report ID")

    result = await db.execute(
        select(Report).where(Report.id == report_uuid, Report.user_id == user.user_id)
    )
    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    await db.delete(report)
    await db.commit()

    return {"message": "Report deleted successfully"}
