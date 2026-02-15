"""
AIDocumentIndexer - Mistral OCR 3 Integration
==============================================

Mistral OCR 3 provides state-of-the-art document understanding with:
- 74% win rate over previous OCR models
- Forms, scanned documents, complex tables, handwriting support
- Multi-page document processing
- Structured output with text ordering
- Image extraction and referencing

Phase 68: Integrated as primary OCR option for high-accuracy document processing.
"""

import asyncio
import base64
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class MistralOCRModel(str, Enum):
    """Available Mistral OCR models."""
    OCR_3 = "mistral-ocr-3"  # Latest model
    OCR_2 = "mistral-ocr-2"  # Previous version


@dataclass
class BoundingBox:
    """Bounding box for text or image location."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    page: int = 0


@dataclass
class OCRTextBlock:
    """A block of text extracted by OCR."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    block_type: str = "text"  # text, heading, list_item, table_cell
    page: int = 0
    order: int = 0


@dataclass
class OCRTable:
    """A table extracted from the document."""
    cells: List[List[str]]
    rows: int
    cols: int
    bounding_box: Optional[BoundingBox] = None
    page: int = 0


@dataclass
class OCRImage:
    """An image reference extracted from the document."""
    image_id: str
    alt_text: str
    bounding_box: Optional[BoundingBox] = None
    page: int = 0
    base64_data: Optional[str] = None


@dataclass
class MistralOCRResult:
    """Result from Mistral OCR processing."""
    text: str
    text_blocks: List[OCRTextBlock] = field(default_factory=list)
    tables: List[OCRTable] = field(default_factory=list)
    images: List[OCRImage] = field(default_factory=list)
    page_count: int = 1
    confidence: float = 0.0
    model: str = "mistral-ocr-3"
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MistralOCRService:
    """
    Mistral OCR 3 service for high-accuracy document OCR.

    Features:
    - 74% win rate over OCR 2 in benchmarks
    - Complex table understanding
    - Handwriting recognition
    - Form field extraction
    - Multi-page document support
    - Structured output with reading order

    Usage:
        service = MistralOCRService()
        result = await service.process_document("document.pdf")
        print(result.text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: MistralOCRModel = MistralOCRModel.OCR_3,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize Mistral OCR service.

        Args:
            api_key: Mistral API key (or from MISTRAL_API_KEY env var)
            model: OCR model to use
            base_url: Custom API base URL (optional)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.base_url = base_url or os.getenv(
            "MISTRAL_API_BASE_URL", "https://api.mistral.ai/v1"
        )
        self.timeout = timeout
        self._client = None
        self._initialized = False

        if not self.api_key:
            logger.warning(
                "Mistral API key not provided. Set MISTRAL_API_KEY environment variable."
            )

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                from backend.services.http_client import get_http_client
                self._client = await get_http_client()
            except ImportError:
                import httpx
                self._client = httpx.AsyncClient(timeout=self.timeout)

        return self._client

    def _encode_file(self, file_path: Union[str, Path]) -> tuple[str, str]:
        """
        Encode a file to base64 for API submission.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (base64_data, mime_type)
        """
        file_path = Path(file_path)

        # Determine MIME type
        suffix = file_path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(suffix, "application/octet-stream")

        # Read and encode file
        with open(file_path, "rb") as f:
            data = f.read()
            base64_data = base64.b64encode(data).decode("utf-8")

        return base64_data, mime_type

    async def process_document(
        self,
        file_path: Union[str, Path],
        extract_tables: bool = True,
        extract_images: bool = False,
        languages: Optional[List[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> MistralOCRResult:
        """
        Process a document using Mistral OCR 3.

        Args:
            file_path: Path to PDF or image file
            extract_tables: Extract structured tables
            extract_images: Extract images with positions
            languages: Hint for document languages (e.g., ["en", "de"])
            page_range: Optional (start, end) page range for PDFs

        Returns:
            MistralOCRResult with extracted content
        """
        import time
        start_time = time.time()

        if not self.api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable."
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(
            "Processing document with Mistral OCR 3",
            file=str(file_path),
            model=self.model.value,
        )

        # Encode file
        base64_data, mime_type = self._encode_file(file_path)

        # Build request payload
        payload = {
            "model": self.model.value,
            "document": {
                "type": "base64",
                "source_type": mime_type,
                "data": base64_data,
            },
            "include_tables": extract_tables,
            "include_images": extract_images,
        }

        if languages:
            payload["languages"] = languages

        if page_range:
            payload["page_range"] = {
                "start": page_range[0],
                "end": page_range[1],
            }

        # Make API request
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await client.post(
                f"{self.base_url}/ocr",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.error(
                    "Mistral OCR API error",
                    status_code=response.status_code,
                    error=response.text,
                )
                raise RuntimeError(
                    f"Mistral OCR API error ({response.status_code})"
                )

            result_data = response.json()

        except Exception as e:
            logger.error("Mistral OCR request failed", error=str(e))
            raise

        # Parse response
        processing_time = (time.time() - start_time) * 1000
        result = self._parse_response(result_data, processing_time)

        logger.info(
            "Mistral OCR processing complete",
            pages=result.page_count,
            text_blocks=len(result.text_blocks),
            tables=len(result.tables),
            processing_time_ms=round(processing_time, 2),
        )

        return result

    async def process_image(
        self,
        image_data: bytes,
        mime_type: str = "image/png",
        extract_tables: bool = True,
    ) -> MistralOCRResult:
        """
        Process an image from bytes.

        Args:
            image_data: Raw image bytes
            mime_type: Image MIME type
            extract_tables: Extract structured tables

        Returns:
            MistralOCRResult with extracted content
        """
        import time
        start_time = time.time()

        if not self.api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable."
            )

        base64_data = base64.b64encode(image_data).decode("utf-8")

        payload = {
            "model": self.model.value,
            "document": {
                "type": "base64",
                "source_type": mime_type,
                "data": base64_data,
            },
            "include_tables": extract_tables,
        }

        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await client.post(
            f"{self.base_url}/ocr",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Mistral OCR API error ({response.status_code}): {response.text}"
            )

        processing_time = (time.time() - start_time) * 1000
        return self._parse_response(response.json(), processing_time)

    async def process_url(
        self,
        url: str,
        extract_tables: bool = True,
    ) -> MistralOCRResult:
        """
        Process a document from URL.

        Args:
            url: URL to PDF or image
            extract_tables: Extract structured tables

        Returns:
            MistralOCRResult with extracted content
        """
        import time
        start_time = time.time()

        if not self.api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable."
            )

        payload = {
            "model": self.model.value,
            "document": {
                "type": "url",
                "url": url,
            },
            "include_tables": extract_tables,
        }

        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await client.post(
            f"{self.base_url}/ocr",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Mistral OCR API error ({response.status_code}): {response.text}"
            )

        processing_time = (time.time() - start_time) * 1000
        return self._parse_response(response.json(), processing_time)

    def _parse_response(
        self,
        data: Dict[str, Any],
        processing_time: float,
    ) -> MistralOCRResult:
        """
        Parse Mistral OCR API response into structured result.

        Args:
            data: API response data
            processing_time: Processing time in milliseconds

        Returns:
            MistralOCRResult
        """
        text_blocks = []
        tables = []
        images = []
        full_text_parts = []

        # Parse pages
        pages = data.get("pages", [])
        page_count = len(pages) if pages else 1

        for page_idx, page in enumerate(pages):
            # Parse text blocks
            for block_idx, block in enumerate(page.get("blocks", [])):
                block_type = block.get("type", "text")

                if block_type in ("text", "heading", "list_item"):
                    text = block.get("text", "")
                    confidence = block.get("confidence", 0.0)

                    bbox = None
                    if "bounding_box" in block:
                        bb = block["bounding_box"]
                        bbox = BoundingBox(
                            x_min=bb.get("x_min", 0),
                            y_min=bb.get("y_min", 0),
                            x_max=bb.get("x_max", 0),
                            y_max=bb.get("y_max", 0),
                            page=page_idx,
                        )

                    text_blocks.append(OCRTextBlock(
                        text=text,
                        confidence=confidence,
                        bounding_box=bbox,
                        block_type=block_type,
                        page=page_idx,
                        order=block_idx,
                    ))

                    full_text_parts.append(text)

                elif block_type == "table":
                    cells = block.get("cells", [])
                    rows = block.get("rows", 0)
                    cols = block.get("cols", 0)

                    bbox = None
                    if "bounding_box" in block:
                        bb = block["bounding_box"]
                        bbox = BoundingBox(
                            x_min=bb.get("x_min", 0),
                            y_min=bb.get("y_min", 0),
                            x_max=bb.get("x_max", 0),
                            y_max=bb.get("y_max", 0),
                            page=page_idx,
                        )

                    tables.append(OCRTable(
                        cells=cells,
                        rows=rows,
                        cols=cols,
                        bounding_box=bbox,
                        page=page_idx,
                    ))

                    # Add table as markdown to full text
                    if cells:
                        table_md = self._table_to_markdown(cells)
                        full_text_parts.append(table_md)

                elif block_type == "image":
                    image_id = block.get("id", f"img_{page_idx}_{block_idx}")
                    alt_text = block.get("alt_text", "")

                    bbox = None
                    if "bounding_box" in block:
                        bb = block["bounding_box"]
                        bbox = BoundingBox(
                            x_min=bb.get("x_min", 0),
                            y_min=bb.get("y_min", 0),
                            x_max=bb.get("x_max", 0),
                            y_max=bb.get("y_max", 0),
                            page=page_idx,
                        )

                    images.append(OCRImage(
                        image_id=image_id,
                        alt_text=alt_text,
                        bounding_box=bbox,
                        page=page_idx,
                        base64_data=block.get("data"),
                    ))

                    full_text_parts.append(f"[Image: {alt_text}]")

        # Calculate average confidence
        if text_blocks:
            avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks)
        else:
            avg_confidence = data.get("confidence", 0.0)

        # Combine all text
        full_text = "\n\n".join(full_text_parts) if full_text_parts else data.get("text", "")

        return MistralOCRResult(
            text=full_text,
            text_blocks=text_blocks,
            tables=tables,
            images=images,
            page_count=page_count,
            confidence=avg_confidence,
            model=self.model.value,
            processing_time_ms=processing_time,
            metadata=data.get("metadata", {}),
        )

    def _table_to_markdown(self, cells: List[List[str]]) -> str:
        """Convert table cells to markdown format."""
        if not cells:
            return ""

        lines = []

        # Header row
        if len(cells) > 0:
            header = "| " + " | ".join(str(c) for c in cells[0]) + " |"
            lines.append(header)
            separator = "| " + " | ".join("---" for _ in cells[0]) + " |"
            lines.append(separator)

        # Data rows
        for row in cells[1:]:
            row_str = "| " + " | ".join(str(c) for c in row) + " |"
            lines.append(row_str)

        return "\n".join(lines)

    async def batch_process(
        self,
        file_paths: List[Union[str, Path]],
        extract_tables: bool = True,
        max_concurrent: int = 5,
    ) -> List[MistralOCRResult]:
        """
        Process multiple documents concurrently.

        Args:
            file_paths: List of file paths
            extract_tables: Extract structured tables
            max_concurrent: Maximum concurrent requests

        Returns:
            List of MistralOCRResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(file_path):
            async with semaphore:
                try:
                    return await self.process_document(
                        file_path, extract_tables=extract_tables
                    )
                except Exception as e:
                    logger.error(
                        "Batch OCR failed for file",
                        file=str(file_path),
                        error=str(e),
                    )
                    # Return empty result on error
                    return MistralOCRResult(
                        text="",
                        metadata={"error": str(e), "file": str(file_path)},
                    )

        tasks = [process_with_limit(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)

        return results

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Mistral OCR API health.

        Returns:
            Health status including API availability
        """
        try:
            if not self.api_key:
                return {
                    "status": "error",
                    "message": "API key not configured",
                }

            client = await self._get_client()
            headers = {"Authorization": f"Bearer {self.api_key}"}

            # Check models endpoint
            response = await client.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10.0,
            )

            if response.status_code == 200:
                models = response.json().get("data", [])
                ocr_models = [m for m in models if "ocr" in m.get("id", "").lower()]

                return {
                    "status": "healthy",
                    "api_available": True,
                    "ocr_models": [m.get("id") for m in ocr_models],
                    "current_model": self.model.value,
                }
            else:
                return {
                    "status": "degraded",
                    "api_available": False,
                    "error": f"API returned {response.status_code}",
                }

        except Exception as e:
            return {
                "status": "error",
                "api_available": False,
                "error": str(e),
            }


# Singleton instance
_mistral_ocr_instance: Optional[MistralOCRService] = None


async def _load_settings_from_admin() -> Dict[str, Any]:
    """
    Load Mistral OCR settings from admin settings with env var fallback.

    Settings hierarchy:
    1. Admin settings (database) - takes priority
    2. Environment variables - fallback

    Returns:
        Dict with api_key, model, enabled, extract_tables
    """
    settings = {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "model": os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-3"),
        "enabled": os.getenv("MISTRAL_OCR_ENABLED", "false").lower() == "true",
        "extract_tables": os.getenv("MISTRAL_OCR_EXTRACT_TABLES", "true").lower() == "true",
    }

    try:
        from backend.services.settings import get_settings_service
        settings_service = get_settings_service()

        # Load from admin settings (these override env vars if set)
        admin_settings = await settings_service.get_all_settings()

        # API key from admin settings (if set and not empty)
        if admin_settings.get("ocr.mistral_api_key"):
            settings["api_key"] = admin_settings["ocr.mistral_api_key"]

        # Model from admin settings
        if admin_settings.get("ocr.mistral_model"):
            settings["model"] = admin_settings["ocr.mistral_model"]

        # Enabled flag
        if "ocr.mistral_enabled" in admin_settings:
            settings["enabled"] = admin_settings["ocr.mistral_enabled"]

        # Extract tables default
        if "ocr.mistral_extract_tables" in admin_settings:
            settings["extract_tables"] = admin_settings["ocr.mistral_extract_tables"]

        logger.debug(
            "Loaded Mistral OCR settings",
            enabled=settings["enabled"],
            model=settings["model"],
            has_api_key=bool(settings["api_key"]),
        )

    except Exception as e:
        logger.warning(
            "Failed to load admin settings for Mistral OCR, using env vars",
            error=str(e),
        )

    return settings


async def get_mistral_ocr() -> MistralOCRService:
    """
    Get or create Mistral OCR service singleton.

    Configuration is loaded from admin settings with env var fallback.
    """
    global _mistral_ocr_instance

    if _mistral_ocr_instance is None:
        # Load settings from admin settings or env vars
        settings = await _load_settings_from_admin()

        # Determine model enum
        model_str = settings["model"]
        if model_str == "mistral-ocr-2":
            model = MistralOCRModel.OCR_2
        else:
            model = MistralOCRModel.OCR_3

        _mistral_ocr_instance = MistralOCRService(
            api_key=settings["api_key"],
            model=model,
        )

    return _mistral_ocr_instance


def reset_mistral_ocr() -> None:
    """Reset the Mistral OCR singleton to reload settings."""
    global _mistral_ocr_instance
    _mistral_ocr_instance = None
    logger.info("Mistral OCR service reset, will reload settings on next use")
