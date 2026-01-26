"""
AIDocumentIndexer - Vision Document Processing API Routes
==========================================================

API endpoints for OCR and vision-based document processing.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import base64

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from backend.services.vision_document_processor import (
    VisionDocumentProcessor,
    VisionConfig,
    OCREngine,
    DocumentType,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])

# Global processor instance
_processor: Optional[VisionDocumentProcessor] = None


def get_processor() -> VisionDocumentProcessor:
    """Get or create the vision processor."""
    global _processor
    if _processor is None:
        _processor = VisionDocumentProcessor()
    return _processor


# =============================================================================
# Request/Response Models
# =============================================================================

class OCREngineEnum(str, Enum):
    """Available OCR engines."""
    SURYA = "surya"
    TESSERACT = "tesseract"
    CLAUDE_VISION = "claude"
    MISTRAL = "mistral"  # Phase 68: Mistral OCR 3 - 74% win rate


class DocumentTypeEnum(str, Enum):
    """Document types for specialized processing."""
    GENERAL = "general"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    FORM = "form"
    TABLE = "table"
    HANDWRITTEN = "handwritten"


class OCRResultResponse(BaseModel):
    """OCR result."""
    text: str
    confidence: float
    engine_used: str
    language_detected: Optional[str] = None
    processing_time_ms: float


class TableResponse(BaseModel):
    """Extracted table."""
    rows: List[List[str]]
    headers: Optional[List[str]] = None
    confidence: float


class StructuredDataResponse(BaseModel):
    """Structured data from document."""
    document_type: str
    fields: Dict[str, Any]
    confidence: float


class VisionResultResponse(BaseModel):
    """Complete vision processing result."""
    text: str
    ocr_confidence: float
    engine_used: str
    tables: List[TableResponse]
    structured_data: Optional[StructuredDataResponse] = None
    page_count: int
    processing_time_ms: float


class ProcessBase64Request(BaseModel):
    """Request to process base64-encoded image."""
    image_data: str = Field(..., description="Base64-encoded image data")
    document_type: Optional[DocumentTypeEnum] = Field(None, description="Document type hint")
    extract_tables: bool = Field(True, description="Extract tables from document")
    extract_structured: bool = Field(True, description="Extract structured data")


class InvoiceExtractResponse(BaseModel):
    """Extracted invoice data."""
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    payment_method: Optional[str] = None
    notes: Optional[str] = None
    confidence: float


class ConfigResponse(BaseModel):
    """Vision processor configuration."""
    primary_engine: str
    fallback_engine: str
    use_vision_llm: bool
    vision_model: str
    detect_tables: bool
    detect_handwriting: bool
    extract_structured: bool
    language_hints: List[str]
    min_confidence: float


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/process", response_model=VisionResultResponse)
async def process_image(
    file: UploadFile = File(..., description="Image file to process"),
    document_type: Optional[str] = Form(None, description="Document type hint"),
    extract_tables: bool = Form(True, description="Extract tables"),
    extract_structured: bool = Form(True, description="Extract structured data"),
):
    """
    Process an image file with OCR and optional structured extraction.

    Supports: PNG, JPG, JPEG, GIF, WEBP, PDF

    OCR Engines (automatic fallback):
    1. Surya (97.7% accuracy)
    2. Tesseract (92% accuracy)
    3. Claude Vision (95%+ for complex layouts)
    """
    logger.info(
        "Processing image",
        filename=file.filename,
        content_type=file.content_type,
        document_type=document_type,
    )

    # Validate file type
    allowed_types = [
        "image/png", "image/jpeg", "image/jpg", "image/gif",
        "image/webp", "application/pdf",
    ]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}",
        )

    try:
        processor = get_processor()
        image_data = await file.read()

        # Parse document type
        doc_type = None
        if document_type:
            try:
                doc_type = DocumentType(document_type)
            except ValueError:
                pass

        # Check if PDF
        if file.content_type == "application/pdf" or (
            file.filename and file.filename.lower().endswith(".pdf")
        ):
            result = await processor.process_pdf(image_data, doc_type)
        else:
            result = await processor.process_image(
                image_data,
                document_type=doc_type,
                extract_tables=extract_tables,
                extract_structured=extract_structured,
            )

        # Build response
        tables_response = [
            TableResponse(
                rows=t.rows,
                headers=t.headers,
                confidence=t.confidence,
            )
            for t in result.tables
        ]

        structured_response = None
        if result.structured_data:
            structured_response = StructuredDataResponse(
                document_type=result.structured_data.document_type.value,
                fields=result.structured_data.fields,
                confidence=result.structured_data.confidence,
            )

        logger.info(
            "Image processed",
            text_length=len(result.text),
            tables_found=len(result.tables),
            has_structured=result.structured_data is not None,
        )

        return VisionResultResponse(
            text=result.text,
            ocr_confidence=result.ocr_result.confidence,
            engine_used=result.ocr_result.engine_used.value,
            tables=tables_response,
            structured_data=structured_response,
            page_count=result.page_count,
            processing_time_ms=result.ocr_result.processing_time_ms,
        )

    except Exception as e:
        logger.error("Image processing failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed: {str(e)}")


@router.post("/process-base64", response_model=VisionResultResponse)
async def process_base64_image(request: ProcessBase64Request):
    """
    Process a base64-encoded image.

    Useful for API integrations where file upload is not convenient.
    """
    logger.info(
        "Processing base64 image",
        document_type=request.document_type,
        data_length=len(request.image_data),
    )

    try:
        # Decode base64
        try:
            image_data = base64.b64decode(request.image_data)
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 image data")

        processor = get_processor()

        # Parse document type
        doc_type = None
        if request.document_type:
            doc_type = DocumentType(request.document_type.value)

        result = await processor.process_image(
            image_data,
            document_type=doc_type,
            extract_tables=request.extract_tables,
            extract_structured=request.extract_structured,
        )

        # Build response
        tables_response = [
            TableResponse(
                rows=t.rows,
                headers=t.headers,
                confidence=t.confidence,
            )
            for t in result.tables
        ]

        structured_response = None
        if result.structured_data:
            structured_response = StructuredDataResponse(
                document_type=result.structured_data.document_type.value,
                fields=result.structured_data.fields,
                confidence=result.structured_data.confidence,
            )

        return VisionResultResponse(
            text=result.text,
            ocr_confidence=result.ocr_result.confidence,
            engine_used=result.ocr_result.engine_used.value,
            tables=tables_response,
            structured_data=structured_response,
            page_count=result.page_count,
            processing_time_ms=result.ocr_result.processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Base64 image processing failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed: {str(e)}")


@router.post("/extract-invoice", response_model=InvoiceExtractResponse)
async def extract_invoice(
    file: UploadFile = File(..., description="Invoice/receipt image"),
):
    """
    Extract structured data from an invoice or receipt.

    Returns vendor info, amounts, line items, dates, etc.
    Achieves 95-98% accuracy on structured field extraction.
    """
    logger.info("Extracting invoice data", filename=file.filename)

    try:
        from backend.services.vision_document_processor import InvoiceExtractor

        image_data = await file.read()
        extractor = InvoiceExtractor()
        result = await extractor.extract(image_data)

        fields = result.fields

        return InvoiceExtractResponse(
            vendor_name=fields.get("vendor_name"),
            vendor_address=fields.get("vendor_address"),
            invoice_number=fields.get("invoice_number"),
            invoice_date=fields.get("invoice_date"),
            due_date=fields.get("due_date"),
            subtotal=fields.get("subtotal"),
            tax=fields.get("tax"),
            total=fields.get("total"),
            currency=fields.get("currency"),
            line_items=fields.get("line_items", []),
            payment_method=fields.get("payment_method"),
            notes=fields.get("notes"),
            confidence=result.confidence,
        )

    except Exception as e:
        logger.error("Invoice extraction failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Extraction failed: {str(e)}")


@router.post("/extract-tables")
async def extract_tables(
    file: UploadFile = File(..., description="Image with tables"),
) -> Dict[str, Any]:
    """
    Extract tables from an image.

    Returns structured table data with headers and rows.
    """
    logger.info("Extracting tables", filename=file.filename)

    try:
        from backend.services.vision_document_processor import TableExtractor

        image_data = await file.read()
        extractor = TableExtractor()
        tables = await extractor.extract_tables(image_data)

        return {
            "tables_found": len(tables),
            "tables": [
                {
                    "headers": t.headers,
                    "rows": t.rows,
                    "confidence": t.confidence,
                }
                for t in tables
            ],
        }

    except Exception as e:
        logger.error("Table extraction failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Extraction failed: {str(e)}")


@router.get("/config", response_model=ConfigResponse)
async def get_vision_config():
    """Get current vision processor configuration."""
    processor = get_processor()
    config = processor.config

    return ConfigResponse(
        primary_engine=config.primary_engine.value,
        fallback_engine=config.fallback_engine.value,
        use_vision_llm=config.use_vision_llm,
        vision_model=config.vision_model,
        detect_tables=config.detect_tables,
        detect_handwriting=config.detect_handwriting,
        extract_structured=config.extract_structured,
        language_hints=config.language_hints,
        min_confidence=config.min_confidence,
    )


@router.get("/engines")
async def list_ocr_engines() -> Dict[str, Any]:
    """List available OCR engines with benchmarks."""
    return {
        "engines": [
            {
                "engine": "mistral",
                "name": "Mistral OCR 3",
                "accuracy": "98%+",
                "speed": "Medium",
                "languages": "100+",
                "description": "Phase 68: Highest accuracy (74% win rate). Best for complex documents, tables, handwriting.",
                "is_primary": True,
                "requires_api_key": True,
            },
            {
                "engine": "surya",
                "name": "Surya OCR",
                "accuracy": "97.7%",
                "speed": "Fast",
                "languages": "90+",
                "description": "Highest accuracy open-source OCR. Best for most documents.",
                "is_primary": False,
            },
            {
                "engine": "tesseract",
                "name": "Tesseract OCR",
                "accuracy": "92%",
                "speed": "Medium",
                "languages": "100+",
                "description": "Reliable fallback. Good for clean documents.",
                "is_primary": False,
            },
            {
                "engine": "claude",
                "name": "Claude Vision",
                "accuracy": "95%+",
                "speed": "Slow",
                "languages": "Multi",
                "description": "Best for complex layouts. Uses API calls.",
                "is_primary": False,
            },
        ],
        "document_types": [
            {"type": "general", "description": "General documents"},
            {"type": "invoice", "description": "Invoices with structured extraction"},
            {"type": "receipt", "description": "Receipts with structured extraction"},
            {"type": "form", "description": "Forms with field detection"},
            {"type": "table", "description": "Documents with tables"},
            {"type": "handwritten", "description": "Handwritten documents"},
        ],
    }


# =============================================================================
# Phase 68: Mistral OCR 3 Specific Endpoints
# =============================================================================

class MistralOCRRequest(BaseModel):
    """Request for Mistral OCR processing."""
    extract_tables: bool = Field(True, description="Extract structured tables")
    extract_images: bool = Field(False, description="Extract images with positions")
    languages: Optional[List[str]] = Field(None, description="Language hints (e.g., ['en', 'de'])")


class MistralOCRResponse(BaseModel):
    """Response from Mistral OCR."""
    text: str
    page_count: int
    tables_count: int
    confidence: float
    model: str
    processing_time_ms: float


@router.post("/mistral-ocr", response_model=MistralOCRResponse)
async def process_with_mistral_ocr(
    file: UploadFile = File(..., description="Document to process"),
    extract_tables: bool = Form(True, description="Extract tables"),
    extract_images: bool = Form(False, description="Extract images"),
):
    """
    Process document using Mistral OCR 3.

    Phase 68: Mistral OCR 3 features:
    - 74% win rate over previous OCR versions
    - Superior handling of forms, scanned docs, complex tables
    - Handwriting recognition
    - Structured output with reading order

    Supports: PDF, PNG, JPG, JPEG, GIF, WEBP, TIFF
    """
    logger.info(
        "Processing with Mistral OCR 3",
        filename=file.filename,
        extract_tables=extract_tables,
    )

    try:
        from backend.services.mistral_ocr import get_mistral_ocr

        service = await get_mistral_ocr()

        # Save file temporarily
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await service.process_document(
                tmp_path,
                extract_tables=extract_tables,
                extract_images=extract_images,
            )
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)

        return MistralOCRResponse(
            text=result.text,
            page_count=result.page_count,
            tables_count=len(result.tables),
            confidence=result.confidence,
            model=result.model,
            processing_time_ms=result.processing_time_ms,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Mistral OCR processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@router.get("/mistral-ocr/health")
async def mistral_ocr_health() -> Dict[str, Any]:
    """Check Mistral OCR 3 service health."""
    try:
        from backend.services.mistral_ocr import get_mistral_ocr

        service = await get_mistral_ocr()
        return await service.health_check()

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


@router.get("/health")
async def vision_health() -> Dict[str, Any]:
    """Check vision processing service health."""
    try:
        processor = get_processor()

        # Check if engines are available
        engines_status = {}
        for engine_type, engine in processor._engines.items():
            try:
                await engine.initialize()
                engines_status[engine_type.value] = "available"
            except Exception:
                engines_status[engine_type.value] = "unavailable"

        return {
            "status": "healthy",
            "engines": engines_status,
            "primary_engine": processor.config.primary_engine.value,
            "fallback_engine": processor.config.fallback_engine.value,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
