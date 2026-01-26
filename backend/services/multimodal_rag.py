"""
AIDocumentIndexer - Multimodal RAG Service
==========================================

Implements multimodal RAG for handling images, tables, and charts in documents.

Features:
- Image captioning and description using vision models
- Table extraction and structured data parsing
- Chart/diagram interpretation
- OCR enhancement for scanned documents
- Visual element indexing for retrieval
"""

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class VisualElementType(str, Enum):
    """Types of visual elements in documents."""
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    EQUATION = "equation"
    LOGO = "logo"
    UNKNOWN = "unknown"


@dataclass
class ExtractedTable:
    """Structured data from a table."""
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    page_number: Optional[int] = None
    table_index: int = 0
    raw_text: str = ""

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers and not self.rows:
            return ""

        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**\n")

        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        for row in self.rows:
            # Ensure row has same columns as headers
            padded_row = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(self.headers)]) + " |")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Convert table to CSV format."""
        lines = []
        if self.headers:
            lines.append(",".join(f'"{h}"' for h in self.headers))
        for row in self.rows:
            lines.append(",".join(f'"{cell}"' for cell in row))
        return "\n".join(lines)


@dataclass
class VisualElement:
    """A visual element extracted from a document."""
    element_type: VisualElementType
    description: str
    content: Optional[str] = None  # Text content if applicable
    image_data: Optional[bytes] = None  # Raw image data
    page_number: Optional[int] = None
    position: Optional[Dict[str, float]] = None  # Bounding box
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For tables
    structured_data: Optional[ExtractedTable] = None


@dataclass
class MultimodalChunk:
    """A chunk that may contain text and visual elements."""
    text_content: str
    visual_elements: List[VisualElement] = field(default_factory=list)
    page_number: Optional[int] = None
    chunk_index: int = 0

    def get_enhanced_content(self) -> str:
        """Get content with visual element descriptions."""
        parts = [self.text_content]

        for element in self.visual_elements:
            if element.element_type == VisualElementType.TABLE:
                if element.structured_data:
                    parts.append(f"\n[TABLE: {element.description}]\n{element.structured_data.to_markdown()}")
            elif element.element_type == VisualElementType.CHART:
                parts.append(f"\n[CHART: {element.description}]")
            elif element.element_type == VisualElementType.IMAGE:
                parts.append(f"\n[IMAGE: {element.description}]")
            elif element.element_type == VisualElementType.DIAGRAM:
                parts.append(f"\n[DIAGRAM: {element.description}]")

        return "\n".join(parts)


# =============================================================================
# Prompts
# =============================================================================

IMAGE_CAPTION_PROMPT = """Describe this image in detail for document retrieval purposes.

Consider:
1. What is shown in the image? (people, objects, scenes, data)
2. What type of image is it? (photo, diagram, chart, screenshot, etc.)
3. What key information can be extracted?
4. What context might this image provide in a document?

Provide a clear, detailed description that would help someone find this image when searching for related information."""


TABLE_EXTRACTION_PROMPT = """Extract the table data from this image into structured format.

Return JSON in this exact format:
{{
  "headers": ["column1", "column2", ...],
  "rows": [
    ["cell1", "cell2", ...],
    ["cell1", "cell2", ...]
  ],
  "caption": "table title or caption if visible",
  "notes": "any additional notes about the table"
}}

Be precise with the cell values. If a cell is empty, use an empty string."""


CHART_ANALYSIS_PROMPT = """Analyze this chart or graph and extract the key information.

Describe:
1. Chart type (bar, line, pie, scatter, etc.)
2. What data is being visualized
3. Axis labels and scales
4. Key trends, patterns, or insights
5. Any notable data points or outliers
6. The main takeaway or conclusion

Provide a comprehensive textual description that captures the information in the chart."""


DIAGRAM_DESCRIPTION_PROMPT = """Describe this diagram or flowchart in detail.

Consider:
1. What type of diagram is this? (flowchart, architecture, process, etc.)
2. What are the main components or elements?
3. How are elements connected or related?
4. What process or concept is being illustrated?
5. What are the key steps or stages shown?

Provide a clear description that conveys the diagram's meaning to someone who cannot see it."""


# =============================================================================
# Multimodal RAG Service
# =============================================================================

class MultimodalRAGService:
    """
    Service for processing and indexing multimodal content.

    Handles images, tables, charts, and diagrams in documents
    to enable rich multimodal retrieval.
    """

    def __init__(
        self,
        vision_model=None,
        embedding_service=None,
        ocr_service=None,
    ):
        self.vision_model = vision_model
        self.embeddings = embedding_service
        self.ocr = ocr_service

        # Check for vision model availability
        self._has_vision = self._check_vision_availability()

    def _check_vision_availability(self) -> bool:
        """Check if vision model is available."""
        # Check for common vision model environment variables
        # Supports both free (Ollama, local) and paid (OpenAI, Anthropic) options
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        has_ollama = bool(os.getenv("OLLAMA_BASE_URL") or os.getenv("USE_OLLAMA"))
        has_local = bool(os.getenv("LOCAL_VISION_MODEL"))
        return has_openai or has_anthropic or has_ollama or has_local

    async def _get_vision_model(self):
        """
        Get or initialize vision model.

        Supports multiple providers with fallback:
        1. FREE: Ollama with LLaVA or similar vision model
        2. FREE: Local vision model (if configured)
        3. PAID: OpenAI GPT-4o
        4. PAID: Anthropic Claude

        Configure via environment variables:
        - USE_OLLAMA=true + OLLAMA_BASE_URL for free local vision
        - OPENAI_API_KEY for paid OpenAI
        - ANTHROPIC_API_KEY for paid Anthropic
        """
        if self.vision_model:
            return self.vision_model

        # Option 1: FREE - Ollama with vision model (e.g., LLaVA, Bakllava)
        if os.getenv("USE_OLLAMA") or os.getenv("OLLAMA_BASE_URL"):
            try:
                from langchain_ollama import ChatOllama
                ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                vision_model = os.getenv("OLLAMA_VISION_MODEL", "llava")
                logger.info(f"Using FREE Ollama vision model: {vision_model}")
                return ChatOllama(
                    model=vision_model,
                    base_url=ollama_base,
                )
            except ImportError:
                logger.debug("langchain_ollama not installed")
            except Exception as e:
                logger.warning(f"Ollama vision init failed: {e}")

        # Option 2: PAID - OpenAI (if configured)
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                logger.info("Using PAID OpenAI GPT-4o vision model")
                return ChatOpenAI(model="gpt-4o", max_tokens=1000)
            except ImportError:
                pass

        # Option 3: PAID - Anthropic (if configured)
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic
                logger.info("Using PAID Anthropic Claude vision model")
                return ChatAnthropic(model="claude-3-5-sonnet-20241022", max_tokens=1000)
            except ImportError:
                pass

        logger.warning(
            "No vision model available. Configure one of: "
            "USE_OLLAMA=true (free), OPENAI_API_KEY (paid), or ANTHROPIC_API_KEY (paid)"
        )
        return None

    # -------------------------------------------------------------------------
    # Image Processing
    # -------------------------------------------------------------------------

    async def caption_image(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> str:
        """
        Generate a caption/description for an image.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpg, etc.)

        Returns:
            Text description of the image
        """
        if not self._has_vision:
            return "[Image - vision model not available for captioning]"

        model = await self._get_vision_model()
        if not model:
            return "[Image - unable to load vision model]"

        try:
            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mime_type = f"image/{image_format.lower()}"

            # Create message with image
            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": IMAGE_CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            )

            response = await model.ainvoke([message])
            caption = response.content if hasattr(response, 'content') else str(response)

            logger.debug("Generated image caption", caption_length=len(caption))
            return caption

        except Exception as e:
            logger.error("Image captioning failed", error=str(e))
            return f"[Image - captioning error: {str(e)[:100]}]"

    async def extract_table_from_image(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> Optional[ExtractedTable]:
        """
        Extract table data from an image of a table.

        Args:
            image_data: Raw image bytes
            image_format: Image format

        Returns:
            ExtractedTable or None if extraction fails
        """
        if not self._has_vision:
            return None

        model = await self._get_vision_model()
        if not model:
            return None

        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mime_type = f"image/{image_format.lower()}"

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": TABLE_EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            )

            response = await model.ainvoke([message])
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            return ExtractedTable(
                headers=data.get("headers", []),
                rows=data.get("rows", []),
                caption=data.get("caption"),
                raw_text=content,
            )

        except Exception as e:
            logger.error("Table extraction failed", error=str(e))
            return None

    async def analyze_chart(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> str:
        """
        Analyze a chart/graph image and extract information.

        Args:
            image_data: Raw image bytes
            image_format: Image format

        Returns:
            Text description of chart content
        """
        if not self._has_vision:
            return "[Chart - vision model not available]"

        model = await self._get_vision_model()
        if not model:
            return "[Chart - unable to load vision model]"

        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mime_type = f"image/{image_format.lower()}"

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": CHART_ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            )

            response = await model.ainvoke([message])
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error("Chart analysis failed", error=str(e))
            return f"[Chart - analysis error: {str(e)[:100]}]"

    async def describe_diagram(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> str:
        """
        Describe a diagram or flowchart.

        Args:
            image_data: Raw image bytes
            image_format: Image format

        Returns:
            Text description of diagram
        """
        if not self._has_vision:
            return "[Diagram - vision model not available]"

        model = await self._get_vision_model()
        if not model:
            return "[Diagram - unable to load vision model]"

        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mime_type = f"image/{image_format.lower()}"

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": DIAGRAM_DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            )

            response = await model.ainvoke([message])
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error("Diagram description failed", error=str(e))
            return f"[Diagram - description error: {str(e)[:100]}]"

    # -------------------------------------------------------------------------
    # Visual Element Classification
    # -------------------------------------------------------------------------

    async def classify_visual_element(
        self,
        image_data: bytes,
        image_format: str = "png",
    ) -> VisualElementType:
        """
        Classify the type of visual element in an image.

        Args:
            image_data: Raw image bytes
            image_format: Image format

        Returns:
            VisualElementType classification
        """
        if not self._has_vision:
            return VisualElementType.UNKNOWN

        model = await self._get_vision_model()
        if not model:
            return VisualElementType.UNKNOWN

        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mime_type = f"image/{image_format.lower()}"

            classification_prompt = """Classify this image into ONE of these categories:
- IMAGE: A photograph or general image
- TABLE: A data table
- CHART: A chart, graph, or plot
- DIAGRAM: A diagram, flowchart, or schematic
- SCREENSHOT: A screenshot of a screen or interface
- EQUATION: A mathematical equation or formula
- LOGO: A logo or brand image

Respond with just the category name in uppercase."""

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": classification_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            )

            response = await model.ainvoke([message])
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            content_upper = content.strip().upper()
            for vtype in VisualElementType:
                if vtype.value.upper() in content_upper:
                    return vtype

            return VisualElementType.UNKNOWN

        except Exception as e:
            logger.error("Visual classification failed", error=str(e))
            return VisualElementType.UNKNOWN

    # -------------------------------------------------------------------------
    # Document Processing
    # -------------------------------------------------------------------------

    async def process_visual_element(
        self,
        image_data: bytes,
        image_format: str = "png",
        page_number: Optional[int] = None,
    ) -> VisualElement:
        """
        Process a visual element and generate description.

        Args:
            image_data: Raw image bytes
            image_format: Image format
            page_number: Page number in document

        Returns:
            VisualElement with description
        """
        # Classify the element
        element_type = await self.classify_visual_element(image_data, image_format)

        # Process based on type
        if element_type == VisualElementType.TABLE:
            table = await self.extract_table_from_image(image_data, image_format)
            description = table.to_markdown() if table else "Table data"
            return VisualElement(
                element_type=element_type,
                description=description,
                structured_data=table,
                page_number=page_number,
            )

        elif element_type == VisualElementType.CHART:
            description = await self.analyze_chart(image_data, image_format)
            return VisualElement(
                element_type=element_type,
                description=description,
                page_number=page_number,
            )

        elif element_type == VisualElementType.DIAGRAM:
            description = await self.describe_diagram(image_data, image_format)
            return VisualElement(
                element_type=element_type,
                description=description,
                page_number=page_number,
            )

        else:
            # Generic image captioning
            description = await self.caption_image(image_data, image_format)
            return VisualElement(
                element_type=element_type,
                description=description,
                page_number=page_number,
            )

    async def enhance_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        images: List[Tuple[int, bytes, str]],  # (page, data, format)
    ) -> List[MultimodalChunk]:
        """
        Enhance document chunks with visual element descriptions.

        Args:
            chunks: List of text chunks with page info
            images: List of (page_number, image_data, format) tuples

        Returns:
            List of MultimodalChunks with visual context
        """
        # Process all images
        visual_elements: Dict[int, List[VisualElement]] = {}
        for page_num, image_data, img_format in images:
            element = await self.process_visual_element(
                image_data,
                img_format,
                page_number=page_num,
            )
            if page_num not in visual_elements:
                visual_elements[page_num] = []
            visual_elements[page_num].append(element)

        # Create multimodal chunks
        multimodal_chunks = []
        for i, chunk in enumerate(chunks):
            page_num = chunk.get("page_number", 0)
            text = chunk.get("content", "")

            mc = MultimodalChunk(
                text_content=text,
                visual_elements=visual_elements.get(page_num, []),
                page_number=page_num,
                chunk_index=i,
            )
            multimodal_chunks.append(mc)

        return multimodal_chunks

    # -------------------------------------------------------------------------
    # Table Utilities
    # -------------------------------------------------------------------------

    def parse_html_table(self, html: str) -> Optional[ExtractedTable]:
        """
        Parse an HTML table into structured format.

        Args:
            html: HTML string containing a table

        Returns:
            ExtractedTable or None
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table')

            if not table:
                return None

            headers = []
            rows = []

            # Extract headers
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text(strip=True))

            # Extract rows
            for tr in table.find_all('tr')[1:]:  # Skip header row
                row = []
                for td in tr.find_all('td'):
                    row.append(td.get_text(strip=True))
                if row:
                    rows.append(row)

            return ExtractedTable(
                headers=headers,
                rows=rows,
            )

        except Exception as e:
            logger.error("HTML table parsing failed", error=str(e))
            return None

    def detect_table_in_text(self, text: str) -> List[ExtractedTable]:
        """
        Detect and extract tables from plain text.

        Uses heuristics to identify tabular data in text.

        Args:
            text: Plain text that may contain tables

        Returns:
            List of detected tables
        """
        tables = []

        # Pattern 1: Pipe-separated (markdown)
        markdown_pattern = r'(\|.+\|\n)(\|[-:\s|]+\|\n)((?:\|.+\|\n)+)'
        for match in re.finditer(markdown_pattern, text):
            try:
                header_line = match.group(1).strip()
                data_lines = match.group(3).strip().split('\n')

                headers = [h.strip() for h in header_line.split('|') if h.strip()]
                rows = []
                for line in data_lines:
                    row = [c.strip() for c in line.split('|') if c.strip()]
                    if row:
                        rows.append(row)

                tables.append(ExtractedTable(headers=headers, rows=rows))
            except (ValueError, IndexError) as e:
                logger.debug("Table extraction failed for block", error=str(e))
                continue

        # Pattern 2: Tab-separated
        lines = text.split('\n')
        tab_table_start = None
        tab_headers = []
        tab_rows = []

        for i, line in enumerate(lines):
            if '\t' in line:
                cells = line.split('\t')
                if len(cells) >= 2:
                    if tab_table_start is None:
                        tab_table_start = i
                        tab_headers = cells
                    else:
                        tab_rows.append(cells)
            else:
                if tab_table_start is not None and tab_rows:
                    tables.append(ExtractedTable(headers=tab_headers, rows=tab_rows))
                tab_table_start = None
                tab_headers = []
                tab_rows = []

        # Add final table if exists
        if tab_table_start is not None and tab_rows:
            tables.append(ExtractedTable(headers=tab_headers, rows=tab_rows))

        return tables

    # -------------------------------------------------------------------------
    # OCR Enhancement
    # -------------------------------------------------------------------------

    async def enhance_ocr_text(
        self,
        image_data: bytes,
        ocr_text: str,
        image_format: str = "png",
    ) -> str:
        """
        Use vision model to enhance/correct OCR text.

        Args:
            image_data: Original image
            ocr_text: Text from OCR
            image_format: Image format

        Returns:
            Enhanced text
        """
        if not self._has_vision:
            return ocr_text

        model = await self._get_vision_model()
        if not model:
            return ocr_text

        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mime_type = f"image/{image_format.lower()}"

            enhance_prompt = f"""The following text was extracted from an image using OCR.
Please review the image and correct any OCR errors in the text.
Only make corrections where you can clearly see the correct text in the image.

OCR Text:
{ocr_text}

Corrected Text:"""

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": enhance_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            )

            response = await model.ainvoke([message])
            return response.content if hasattr(response, 'content') else ocr_text

        except Exception as e:
            logger.error("OCR enhancement failed", error=str(e))
            return ocr_text


# =============================================================================
# Factory Function
# =============================================================================

def get_multimodal_rag_service(
    vision_model=None,
    embedding_service=None,
) -> MultimodalRAGService:
    """Create configured multimodal RAG service."""
    return MultimodalRAGService(
        vision_model=vision_model,
        embedding_service=embedding_service,
    )
