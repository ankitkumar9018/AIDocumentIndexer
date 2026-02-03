"""
AIDocumentIndexer - PDF Tools API Routes
=========================================

API endpoints for PDF manipulation:
- Merge PDFs
- Split PDF
- Extract pages
- Rotate pages
- Compress PDF
- Convert to/from images
- Edit metadata
- Add watermark
- Rearrange pages
"""

import io
import os
import tempfile
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.pdf_tools import (
    PDFToolsService,
    PDFToolResult,
    PDFInfo,
    PDFMetadata,
    get_pdf_tools,
)
from backend.api.middleware.auth import get_current_user

router = APIRouter(prefix="/pdf-tools", tags=["PDF Tools"])


# =============================================================================
# Request/Response Models
# =============================================================================

class MergeRequest(BaseModel):
    """Request to merge multiple PDFs."""
    # Files will be uploaded separately


class SplitRequest(BaseModel):
    """Request to split a PDF."""
    ranges: List[str] = Field(
        ...,
        description="Page ranges to split (e.g., ['1-5', '6-10', '11-'])",
        example=["1-5", "6-10"],
    )


class ExtractRequest(BaseModel):
    """Request to extract pages from a PDF."""
    pages: List[int] = Field(
        ...,
        description="Page numbers to extract (1-indexed)",
        example=[1, 3, 5],
    )


class RotateRequest(BaseModel):
    """Request to rotate PDF pages."""
    rotation: int = Field(
        ...,
        description="Rotation angle (90, 180, 270, or -90)",
        example=90,
    )
    pages: Optional[List[int]] = Field(
        None,
        description="Pages to rotate (1-indexed). If null, rotate all.",
    )


class CompressRequest(BaseModel):
    """Request to compress a PDF."""
    image_quality: int = Field(
        75,
        ge=1,
        le=100,
        description="Quality for image compression (1-100)",
    )


class ToImagesRequest(BaseModel):
    """Request to convert PDF to images."""
    dpi: int = Field(150, ge=72, le=600, description="Image resolution")
    format: str = Field("png", description="Output format (png, jpeg)")
    pages: Optional[List[int]] = Field(None, description="Pages to convert")


class MetadataRequest(BaseModel):
    """Request to update PDF metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None


class WatermarkRequest(BaseModel):
    """Request to add watermark to PDF."""
    text: str = Field(..., description="Watermark text")
    opacity: float = Field(0.3, ge=0, le=1, description="Opacity (0-1)")
    angle: int = Field(45, description="Rotation angle")
    pages: Optional[List[int]] = Field(None, description="Pages to watermark")


class RearrangeRequest(BaseModel):
    """Request to rearrange PDF pages."""
    new_order: List[int] = Field(
        ...,
        description="New page order (1-indexed, can include duplicates)",
        example=[3, 1, 2, 4],
    )


class PDFInfoResponse(BaseModel):
    """PDF information response."""
    page_count: int
    file_size: int
    is_encrypted: bool
    has_text: bool
    has_images: bool
    metadata: dict
    page_sizes: List[List[float]]


class ToolResultResponse(BaseModel):
    """Generic tool result response."""
    success: bool
    message: str
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

async def read_upload_file(file: UploadFile) -> bytes:
    """Read uploaded file contents."""
    contents = await file.read()
    await file.seek(0)
    return contents


def create_pdf_response(
    pdf_bytes: bytes,
    filename: str,
) -> StreamingResponse:
    """Create a streaming response for PDF download."""
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


# =============================================================================
# Routes
# =============================================================================

@router.post("/info", response_model=PDFInfoResponse)
async def get_pdf_info(file: UploadFile = File(...)):
    """Get information about a PDF file."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    try:
        info = tools.get_info(contents)
        return PDFInfoResponse(
            page_count=info.page_count,
            file_size=info.file_size,
            is_encrypted=info.is_encrypted,
            has_text=info.has_text,
            has_images=info.has_images,
            metadata={
                "title": info.metadata.title,
                "author": info.metadata.author,
                "subject": info.metadata.subject,
                "keywords": info.metadata.keywords,
                "creator": info.metadata.creator,
            },
            page_sizes=[[w, h] for w, h in info.page_sizes],
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to get PDF info: {e}")


@router.post("/merge")
async def merge_pdfs(files: List[UploadFile] = File(...)):
    """
    Merge multiple PDFs into one.

    Upload multiple PDF files to merge them in order.
    """
    if len(files) < 2:
        raise HTTPException(400, "At least 2 PDF files required")

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"File {f.filename} is not a PDF")

    tools = get_pdf_tools()

    # Read all files
    sources = []
    for f in files:
        sources.append(await read_upload_file(f))

    result = tools.merge(sources)

    if not result.success:
        raise HTTPException(500, result.error or "Merge failed")

    return create_pdf_response(result.output_bytes, "merged.pdf")


@router.post("/split")
async def split_pdf(
    file: UploadFile = File(...),
    ranges: str = Form(..., description="Comma-separated ranges (e.g., '1-5,6-10')"),
):
    """
    Split a PDF into multiple files based on page ranges.

    Returns a ZIP file containing the split PDFs.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    # Parse ranges
    range_list = [r.strip() for r in ranges.split(",")]

    with tempfile.TemporaryDirectory() as temp_dir:
        results = tools.split(contents, range_list, temp_dir)

        # Check for errors
        errors = [r for r in results if not r.success]
        if errors:
            raise HTTPException(500, f"Split failed: {errors[0].error}")

        # Create ZIP with results
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, result in enumerate(results):
                if result.output_path:
                    zf.write(result.output_path, f"part_{i+1}.pdf")

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="split_pdfs.zip"',
            },
        )


@router.post("/extract")
async def extract_pages(
    file: UploadFile = File(...),
    pages: str = Form(..., description="Comma-separated page numbers (e.g., '1,3,5')"),
):
    """Extract specific pages from a PDF."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    # Parse pages
    page_list = [int(p.strip()) for p in pages.split(",")]

    result = tools.extract_pages(contents, page_list)

    if not result.success:
        raise HTTPException(500, result.error or "Extract failed")

    return create_pdf_response(result.output_bytes, "extracted.pdf")


@router.post("/rotate")
async def rotate_pages(
    file: UploadFile = File(...),
    rotation: int = Form(..., description="Rotation angle (90, 180, 270, -90)"),
    pages: Optional[str] = Form(None, description="Comma-separated pages (e.g., '1,3,5')"),
):
    """Rotate pages in a PDF."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    page_list = None
    if pages:
        page_list = [int(p.strip()) for p in pages.split(",")]

    result = tools.rotate_pages(contents, rotation, page_list)

    if not result.success:
        raise HTTPException(500, result.error or "Rotate failed")

    return create_pdf_response(result.output_bytes, "rotated.pdf")


@router.post("/compress")
async def compress_pdf(
    file: UploadFile = File(...),
    quality: int = Form(75, ge=1, le=100, description="Image quality (1-100)"),
):
    """Compress a PDF to reduce file size."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    result = tools.compress(contents, image_quality=quality)

    if not result.success:
        raise HTTPException(500, result.error or "Compress failed")

    return create_pdf_response(result.output_bytes, "compressed.pdf")


@router.post("/to-images")
async def pdf_to_images(
    file: UploadFile = File(...),
    dpi: int = Form(150, ge=72, le=600),
    format: str = Form("png", description="Output format (png, jpeg)"),
    pages: Optional[str] = Form(None, description="Comma-separated pages"),
):
    """
    Convert PDF pages to images.

    Returns a ZIP file containing the images.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    page_list = None
    if pages:
        page_list = [int(p.strip()) for p in pages.split(",")]

    with tempfile.TemporaryDirectory() as temp_dir:
        results = tools.to_images(
            contents,
            output_dir=temp_dir,
            dpi=dpi,
            image_format=format,
            pages=page_list,
        )

        # Check for errors
        errors = [r for r in results if not r.success]
        if errors:
            raise HTTPException(500, f"Conversion failed: {errors[0].error}")

        # Create ZIP with images
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for result in results:
                if result.output_path:
                    filename = os.path.basename(result.output_path)
                    zf.write(result.output_path, filename)

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="pdf_images.zip"',
            },
        )


@router.post("/from-images")
async def images_to_pdf(
    files: List[UploadFile] = File(...),
    page_size: str = Form("a4", description="Page size (a4, letter, or WxH)"),
):
    """
    Create a PDF from images.

    Upload images in the order you want them in the PDF.
    """
    tools = get_pdf_tools()

    # Read images
    images = []
    for f in files:
        if not f.content_type.startswith("image/"):
            raise HTTPException(400, f"File {f.filename} is not an image")
        images.append(await read_upload_file(f))

    result = tools.from_images(images, page_size=page_size)

    if not result.success:
        raise HTTPException(500, result.error or "Conversion failed")

    return create_pdf_response(result.output_bytes, "images.pdf")


@router.post("/metadata")
async def update_metadata(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
):
    """Update PDF metadata."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    metadata = PDFMetadata(
        title=title,
        author=author,
        subject=subject,
        keywords=keywords,
    )

    result = tools.set_metadata(contents, metadata)

    if not result.success:
        raise HTTPException(500, result.error or "Metadata update failed")

    return create_pdf_response(result.output_bytes, "updated.pdf")


@router.post("/watermark")
async def add_watermark(
    file: UploadFile = File(...),
    text: str = Form(..., description="Watermark text"),
    opacity: float = Form(0.3, ge=0, le=1),
    angle: int = Form(45),
    pages: Optional[str] = Form(None, description="Comma-separated pages"),
):
    """Add a text watermark to PDF pages."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    page_list = None
    if pages:
        page_list = [int(p.strip()) for p in pages.split(",")]

    result = tools.add_watermark(
        contents,
        watermark_text=text,
        opacity=opacity,
        angle=angle,
        pages=page_list,
    )

    if not result.success:
        raise HTTPException(500, result.error or "Watermark failed")

    return create_pdf_response(result.output_bytes, "watermarked.pdf")


@router.post("/rearrange")
async def rearrange_pages(
    file: UploadFile = File(...),
    order: str = Form(
        ...,
        description="New page order (e.g., '3,1,2,4')",
    ),
):
    """
    Rearrange pages in a PDF.

    Specify the new order as comma-separated page numbers.
    Pages can be duplicated or omitted.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    contents = await read_upload_file(file)
    tools = get_pdf_tools()

    # Parse order
    new_order = [int(p.strip()) for p in order.split(",")]

    result = tools.rearrange_pages(contents, new_order)

    if not result.success:
        raise HTTPException(500, result.error or "Rearrange failed")

    return create_pdf_response(result.output_bytes, "rearranged.pdf")
