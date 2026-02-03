"""
AIDocumentIndexer - Workflow Knowledge Service
===============================================

Provides comprehensive knowledge source integration for workflow Agent nodes.
Supports multiple knowledge source types for Voice/Chat agents:

1. Collection/Knowledge Base IDs - Existing indexed collections
2. Folder Sources - Documents in specific folders
3. URL Sources - Web pages to scrape and use as context
4. File Sources - Files uploaded during workflow execution
5. Text Sources - Raw text blocks as knowledge
6. Database Sources - External database queries
7. API Sources - External API data fetching

This service processes all knowledge sources and provides unified context
for RAG-based agent execution in workflows.
"""

import uuid
import asyncio
import hashlib
import aiohttp
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings

logger = structlog.get_logger(__name__)


class KnowledgeSourceType(str, Enum):
    """Types of knowledge sources for workflow agents."""
    COLLECTION = "collection"  # Existing indexed collections
    FOLDER = "folder"  # Documents in folders
    URL = "url"  # Web pages to scrape
    FILE = "file"  # Uploaded files
    TEXT = "text"  # Raw text blocks
    DATABASE = "database"  # External database queries
    API = "api"  # External API calls


class KnowledgeSourceConfig:
    """Configuration for a single knowledge source."""

    def __init__(
        self,
        source_type: KnowledgeSourceType,
        value: Any,
        options: Optional[Dict] = None,
    ):
        self.source_type = source_type
        self.value = value
        self.options = options or {}

    def to_dict(self) -> Dict:
        return {
            "type": self.source_type.value,
            "value": self.value,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeSourceConfig":
        return cls(
            source_type=KnowledgeSourceType(data.get("type", "text")),
            value=data.get("value"),
            options=data.get("options", {}),
        )


class ProcessedKnowledge:
    """Result of processing knowledge sources."""

    def __init__(
        self,
        content: str,
        source_type: KnowledgeSourceType,
        metadata: Optional[Dict] = None,
        chunks: Optional[List[Dict]] = None,
    ):
        self.content = content
        self.source_type = source_type
        self.metadata = metadata or {}
        self.chunks = chunks or []

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)


class WorkflowKnowledgeService:
    """
    Service for processing and integrating knowledge sources in workflows.

    Provides unified knowledge retrieval from multiple source types
    for Voice and Chat Agent nodes.
    """

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._processed_cache: Dict[str, ProcessedKnowledge] = {}
        self.logger = logger.bind(service="workflow_knowledge")

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for URL fetching."""
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def close(self):
        """Close HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    def _cache_key(self, source: KnowledgeSourceConfig) -> str:
        """Generate cache key for a knowledge source."""
        data = f"{source.source_type.value}:{source.value}:{str(source.options)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def process_knowledge_sources(
        self,
        sources: List[KnowledgeSourceConfig],
        organization_id: Optional[str] = None,
        max_context_length: int = 8000,
    ) -> Tuple[str, List[Dict], Dict]:
        """
        Process multiple knowledge sources and combine into unified context.

        Args:
            sources: List of knowledge source configurations
            organization_id: Organization ID for access control
            max_context_length: Maximum combined context length (in words)

        Returns:
            Tuple of (combined_context, all_sources, metadata)
        """
        processed_items: List[ProcessedKnowledge] = []
        all_sources: List[Dict] = []
        errors: List[str] = []

        # Process sources in parallel where possible
        tasks = []
        for source in sources:
            cache_key = self._cache_key(source)

            # Check cache first
            if cache_key in self._processed_cache:
                processed_items.append(self._processed_cache[cache_key])
                continue

            # Create processing task
            tasks.append(self._process_single_source(source, organization_id))

        # Execute all processing tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(str(result))
                    self.logger.warning(
                        "Knowledge source processing failed",
                        error=str(result),
                    )
                elif result:
                    processed_items.append(result)
                    # Cache the result
                    cache_key = self._cache_key(sources[len(processed_items) - 1])
                    self._processed_cache[cache_key] = result

        # Combine all processed content
        combined_parts = []
        current_words = 0

        for item in processed_items:
            if current_words + item.word_count <= max_context_length:
                combined_parts.append(f"[Source: {item.source_type.value}]\n{item.content}")
                current_words += item.word_count

                # Add to sources list
                all_sources.append({
                    "type": item.source_type.value,
                    "metadata": item.metadata,
                    "word_count": item.word_count,
                })
            else:
                # Truncate to fit
                remaining_words = max_context_length - current_words
                if remaining_words > 100:  # Only include if meaningful
                    truncated = " ".join(item.content.split()[:remaining_words])
                    combined_parts.append(f"[Source: {item.source_type.value}]\n{truncated}...")
                    all_sources.append({
                        "type": item.source_type.value,
                        "metadata": item.metadata,
                        "word_count": remaining_words,
                        "truncated": True,
                    })
                break

        combined_context = "\n\n---\n\n".join(combined_parts)

        metadata = {
            "total_sources": len(processed_items),
            "total_words": current_words,
            "errors": errors if errors else None,
            "processed_at": datetime.utcnow().isoformat(),
        }

        return combined_context, all_sources, metadata

    async def _process_single_source(
        self,
        source: KnowledgeSourceConfig,
        organization_id: Optional[str] = None,
    ) -> Optional[ProcessedKnowledge]:
        """Process a single knowledge source."""
        try:
            if source.source_type == KnowledgeSourceType.TEXT:
                return await self._process_text_source(source)
            elif source.source_type == KnowledgeSourceType.URL:
                return await self._process_url_source(source)
            elif source.source_type == KnowledgeSourceType.FILE:
                return await self._process_file_source(source, organization_id)
            elif source.source_type == KnowledgeSourceType.COLLECTION:
                return await self._process_collection_source(source, organization_id)
            elif source.source_type == KnowledgeSourceType.FOLDER:
                return await self._process_folder_source(source, organization_id)
            elif source.source_type == KnowledgeSourceType.DATABASE:
                return await self._process_database_source(source)
            elif source.source_type == KnowledgeSourceType.API:
                return await self._process_api_source(source)
            else:
                self.logger.warning(f"Unknown source type: {source.source_type}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to process {source.source_type}: {e}")
            raise

    async def _process_text_source(
        self,
        source: KnowledgeSourceConfig,
    ) -> ProcessedKnowledge:
        """Process raw text as knowledge source."""
        text = source.value
        if not isinstance(text, str):
            text = str(text)

        # Apply any text preprocessing from options
        if source.options.get("strip_whitespace", True):
            text = " ".join(text.split())

        title = source.options.get("title", "Inline Text")

        return ProcessedKnowledge(
            content=text,
            source_type=KnowledgeSourceType.TEXT,
            metadata={
                "title": title,
                "char_count": len(text),
            },
        )

    async def _process_url_source(
        self,
        source: KnowledgeSourceConfig,
    ) -> ProcessedKnowledge:
        """Process URL by scraping and extracting content."""
        url = source.value

        try:
            session = await self._get_http_session()

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; AIDocIndexer/1.0; +https://aidocindexer.com)",
                **source.options.get("headers", {}),
            }

            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise ValueError(f"URL returned status {response.status}")

                content_type = response.headers.get("Content-Type", "")

                if "text/html" in content_type:
                    html = await response.text()
                    text = await self._extract_text_from_html(html)
                elif "application/json" in content_type:
                    json_data = await response.json()
                    text = self._json_to_text(json_data)
                elif "text/plain" in content_type:
                    text = await response.text()
                else:
                    # Try to read as text
                    text = await response.text()

            # Apply length limit if specified
            max_length = source.options.get("max_length", 10000)
            if len(text) > max_length:
                text = text[:max_length] + "..."

            return ProcessedKnowledge(
                content=text,
                source_type=KnowledgeSourceType.URL,
                metadata={
                    "url": url,
                    "content_type": content_type,
                    "fetched_at": datetime.utcnow().isoformat(),
                },
            )

        except Exception as e:
            self.logger.error(f"URL fetch failed: {url}", error=str(e))
            raise

    async def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML content."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, nav, footer, header elements
            for element in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Get text from main content areas
            main_content = soup.find(["main", "article"]) or soup.find("body")
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)

        except ImportError:
            # Fallback: Simple regex-based extraction
            import re
            # Remove script and style tags
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", text)
            # Clean up whitespace
            text = " ".join(text.split())
            return text

    def _json_to_text(self, data: Any, prefix: str = "") -> str:
        """Convert JSON data to readable text format."""
        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, prefix + "  "))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._json_to_text(item, prefix + "  "))
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)

    async def _process_file_source(
        self,
        source: KnowledgeSourceConfig,
        organization_id: Optional[str] = None,
    ) -> ProcessedKnowledge:
        """Process uploaded file as knowledge source."""
        file_path = source.value

        # Handle both path strings and file info dicts
        if isinstance(file_path, dict):
            file_path = file_path.get("path") or file_path.get("file_path")

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and process accordingly
        suffix = path.suffix.lower()

        if suffix in [".txt", ".md", ".csv", ".json"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = await self._extract_pdf_text(path)
        elif suffix in [".docx", ".doc"]:
            text = await self._extract_docx_text(path)
        elif suffix in [".xlsx", ".xls"]:
            text = await self._extract_excel_text(path)
        else:
            # Try to read as text
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                raise ValueError(f"Unsupported file type: {suffix}")

        return ProcessedKnowledge(
            content=text,
            source_type=KnowledgeSourceType.FILE,
            metadata={
                "filename": path.name,
                "file_type": suffix,
                "file_size": path.stat().st_size,
            },
        )

    async def _extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF file."""
        try:
            import fitz  # PyMuPDF

            text_parts = []
            with fitz.open(str(path)) as doc:
                for page in doc:
                    text_parts.append(page.get_text())

            return "\n\n".join(text_parts)
        except ImportError:
            # Fallback to pdfplumber
            try:
                import pdfplumber

                text_parts = []
                with pdfplumber.open(str(path)) as pdf:
                    for page in pdf.pages:
                        text_parts.append(page.extract_text() or "")

                return "\n\n".join(text_parts)
            except ImportError:
                raise ImportError("PDF processing requires PyMuPDF or pdfplumber")

    async def _extract_docx_text(self, path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document

            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise ImportError("DOCX processing requires python-docx")

    async def _extract_excel_text(self, path: Path) -> str:
        """Extract text from Excel file."""
        try:
            import pandas as pd

            # Read all sheets
            excel_file = pd.ExcelFile(str(path))
            text_parts = []

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                text_parts.append(f"Sheet: {sheet_name}\n{df.to_string()}")

            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError("Excel processing requires pandas and openpyxl")

    async def _process_collection_source(
        self,
        source: KnowledgeSourceConfig,
        organization_id: Optional[str] = None,
    ) -> ProcessedKnowledge:
        """Query existing collection/knowledge base for relevant content."""
        collection_id = source.value
        query = source.options.get("query", "")
        top_k = source.options.get("top_k", 10)

        try:
            from backend.services.rag import RAGService

            rag = RAGService()

            # Query the collection
            result = await rag.query(
                query=query or "Summarize the key information",
                organization_id=organization_id,
                collection_filter=[collection_id] if isinstance(collection_id, str) else collection_id,
                top_k=top_k,
            )

            # Extract context from results
            context = result.get("context", "")
            if not context:
                # Build context from sources
                sources = result.get("sources", [])
                context = "\n\n".join([s.get("content", "") for s in sources])

            return ProcessedKnowledge(
                content=context,
                source_type=KnowledgeSourceType.COLLECTION,
                metadata={
                    "collection_id": collection_id,
                    "query": query,
                    "sources_count": len(result.get("sources", [])),
                },
                chunks=[s for s in result.get("sources", [])],
            )

        except Exception as e:
            self.logger.error(f"Collection query failed: {collection_id}", error=str(e))
            raise

    async def _process_folder_source(
        self,
        source: KnowledgeSourceConfig,
        organization_id: Optional[str] = None,
    ) -> ProcessedKnowledge:
        """Query documents in a specific folder."""
        folder_id = source.value
        query = source.options.get("query", "")
        top_k = source.options.get("top_k", 10)

        try:
            from backend.services.rag import RAGService

            rag = RAGService()

            result = await rag.query(
                query=query or "Summarize the documents in this folder",
                organization_id=organization_id,
                folder_filter=folder_id,
                top_k=top_k,
            )

            context = result.get("context", "")
            if not context:
                sources = result.get("sources", [])
                context = "\n\n".join([s.get("content", "") for s in sources])

            return ProcessedKnowledge(
                content=context,
                source_type=KnowledgeSourceType.FOLDER,
                metadata={
                    "folder_id": folder_id,
                    "query": query,
                    "sources_count": len(result.get("sources", [])),
                },
                chunks=[s for s in result.get("sources", [])],
            )

        except Exception as e:
            self.logger.error(f"Folder query failed: {folder_id}", error=str(e))
            raise

    async def _process_database_source(
        self,
        source: KnowledgeSourceConfig,
    ) -> ProcessedKnowledge:
        """Query external database for context."""
        db_config = source.value

        if isinstance(db_config, str):
            # Assume it's a connection string with query
            raise ValueError("Database source requires structured config with connection and query")

        connection_string = db_config.get("connection_string")
        query = db_config.get("query")

        if not connection_string or not query:
            raise ValueError("Database source requires connection_string and query")

        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.engine import Engine

            # Create engine and execute query
            engine = create_engine(connection_string)

            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()

            # Convert to text format
            if rows:
                text_lines = [", ".join(str(col) for col in columns)]
                for row in rows[:100]:  # Limit rows
                    text_lines.append(", ".join(str(val) for val in row))
                content = "\n".join(text_lines)
            else:
                content = "No results found"

            return ProcessedKnowledge(
                content=content,
                source_type=KnowledgeSourceType.DATABASE,
                metadata={
                    "query": query,
                    "row_count": len(rows),
                    "columns": list(columns),
                },
            )

        except Exception as e:
            self.logger.error("Database query failed", error=str(e))
            raise

    async def _process_api_source(
        self,
        source: KnowledgeSourceConfig,
    ) -> ProcessedKnowledge:
        """Fetch data from external API."""
        api_config = source.value

        if isinstance(api_config, str):
            # Assume it's a URL
            api_config = {"url": api_config, "method": "GET"}

        url = api_config.get("url")
        method = api_config.get("method", "GET").upper()
        headers = api_config.get("headers", {})
        body = api_config.get("body")

        if not url:
            raise ValueError("API source requires url")

        try:
            session = await self._get_http_session()

            kwargs = {"headers": headers}
            if body and method in ["POST", "PUT", "PATCH"]:
                kwargs["json"] = body

            async with session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    raise ValueError(f"API returned status {response.status}")

                content_type = response.headers.get("Content-Type", "")

                if "application/json" in content_type:
                    data = await response.json()
                    text = self._json_to_text(data)
                else:
                    text = await response.text()

            return ProcessedKnowledge(
                content=text,
                source_type=KnowledgeSourceType.API,
                metadata={
                    "url": url,
                    "method": method,
                    "status": response.status,
                },
            )

        except Exception as e:
            self.logger.error(f"API fetch failed: {url}", error=str(e))
            raise


# Factory function for creating service instances
def get_workflow_knowledge_service(
    session: Optional[AsyncSession] = None,
) -> WorkflowKnowledgeService:
    """Get a workflow knowledge service instance."""
    return WorkflowKnowledgeService(session=session)
