"""
AIDocumentIndexer - Image Analysis Service
==========================================

Smart image analysis with deduplication for document processing.

Features:
- SHA-256 hash for exact duplicate detection
- Cache captions in AnalyzedImage table
- Skip small images (icons, decorations)
- Respect max_images_per_document limit
- Track analysis statistics
"""

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from backend.db.database import get_async_session, async_session_context
from backend.db.models import Document, AnalyzedImage, Chunk
from backend.services.settings import get_settings_service
from backend.services.multimodal_rag import get_multimodal_rag_service

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageAnalysisStats:
    """Statistics from image analysis operation."""
    images_found: int = 0
    images_analyzed: int = 0
    images_skipped_small: int = 0
    images_skipped_duplicate: int = 0
    images_failed: int = 0
    cached_used: int = 0
    newly_analyzed: int = 0
    total_time_ms: int = 0


@dataclass
class ImageAnalysisResult:
    """Result from analyzing document images."""
    success: bool
    stats: ImageAnalysisStats
    captions: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class VisionStatus:
    """Status of vision model availability."""
    available: bool
    provider: Optional[str]
    model: Optional[str]
    issues: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None


# =============================================================================
# Image Analysis Service
# =============================================================================

class ImageAnalysisService:
    """
    Smart image analysis with deduplication.

    Features:
    - SHA-256 hash for exact duplicate detection
    - Cache captions in AnalyzedImage table
    - Skip small images (icons, decorations)
    - Respect max_images_per_document limit
    - Track analysis statistics
    """

    def __init__(self):
        self._settings = get_settings_service()
        self._multimodal = get_multimodal_rag_service()

    def compute_image_hash(self, image_data: bytes) -> str:
        """Compute SHA-256 hash of image data."""
        return hashlib.sha256(image_data).hexdigest()

    async def check_vision_available(self) -> VisionStatus:
        """
        Check if vision model is configured and responding.

        Returns status with provider info and any issues found.
        """
        import os

        issues = []
        provider = None
        model = None

        # Check Ollama (free option) - PRIORITIZE FREE LOCAL MODEL
        has_ollama = bool(os.getenv("USE_OLLAMA") or os.getenv("OLLAMA_BASE_URL"))
        if has_ollama:
            provider = "ollama"
            model = os.getenv("OLLAMA_VISION_MODEL", "llava")
            # TODO: Could add ping check to verify ollama is running

        # Check OpenAI - only if key is valid (not a placeholder)
        openai_key = os.getenv("OPENAI_API_KEY", "")
        has_openai = bool(openai_key and not openai_key.startswith("sk-your-") and len(openai_key) > 20)
        if has_openai and not provider:
            provider = "openai"
            model = "gpt-4o"

        # Check Anthropic - only if key is valid (not a placeholder)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        has_anthropic = bool(anthropic_key and not anthropic_key.startswith("sk-ant-") and len(anthropic_key) > 10)
        if has_anthropic and not provider:
            provider = "anthropic"
            model = "claude-3-5-sonnet"

        available = provider is not None

        if not available:
            issues.append("No vision model configured")

        return VisionStatus(
            available=available,
            provider=provider,
            model=model,
            issues=issues,
            recommendation="Run 'ollama pull llava' for free local vision" if not available else None,
        )

    async def find_cached_caption(
        self,
        db: AsyncSession,
        image_hash: str,
    ) -> Optional[str]:
        """
        Check if we've already analyzed this image.

        Returns the cached caption if found, None otherwise.
        Also increments usage_count for cached images.
        """
        result = await db.execute(
            select(AnalyzedImage).where(AnalyzedImage.image_hash == image_hash)
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update usage count
            existing.usage_count += 1
            await db.commit()
            return existing.caption

        return None

    async def cache_caption(
        self,
        db: AsyncSession,
        image_hash: str,
        caption: str,
        element_type: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        document_id: Optional[UUID] = None,
    ) -> AnalyzedImage:
        """
        Cache an image analysis result for future reuse.
        """
        analyzed_image = AnalyzedImage(
            image_hash=image_hash,
            caption=caption,
            element_type=element_type,
            analysis_provider=provider,
            analysis_model=model,
            first_document_id=document_id,
            usage_count=1,
        )
        db.add(analyzed_image)
        await db.commit()
        await db.refresh(analyzed_image)
        return analyzed_image

    async def analyze_document_images(
        self,
        document_id: str,
        force: bool = False,
        skip_duplicates: bool = True,
        db: Optional[AsyncSession] = None,
    ) -> ImageAnalysisResult:
        """
        Analyze or re-analyze images for a document.

        Args:
            document_id: Document ID to analyze
            force: If True, re-analyze even if already done
            skip_duplicates: If True, use cached results for identical images
            db: Optional database session

        Returns:
            ImageAnalysisResult with stats and captions
        """
        import time
        from backend.processors.universal import UniversalProcessor

        start_time = time.time()
        stats = ImageAnalysisStats()

        async def _analyze(session: AsyncSession) -> ImageAnalysisResult:
            nonlocal stats

            # Get document
            doc_uuid = UUID(document_id)
            result = await session.execute(
                select(Document).where(Document.id == doc_uuid)
            )
            document = result.scalar_one_or_none()

            if not document:
                return ImageAnalysisResult(
                    success=False,
                    stats=stats,
                    error=f"Document not found: {document_id}",
                )

            # Check if already completed and not forcing
            if document.image_analysis_status == "completed" and not force:
                return ImageAnalysisResult(
                    success=True,
                    stats=stats,
                    error="Image analysis already completed. Use force=True to re-analyze.",
                )

            # Update status to processing
            document.image_analysis_status = "processing"
            await session.commit()

            try:
                # Get settings
                max_images = await self._settings.get_setting("rag.max_images_per_document") or 50
                min_size_kb = await self._settings.get_setting("rag.min_image_size_kb") or 5
                enable_dedup = await self._settings.get_setting("rag.image_duplicate_detection") or True

                # Extract images from document
                processor = UniversalProcessor()
                if not document.file_path:
                    return ImageAnalysisResult(
                        success=False,
                        stats=stats,
                        error="Document has no file path",
                    )

                extracted = processor.process(document.file_path)
                images = extracted.extracted_images or []
                stats.images_found = len(images)

                if not images:
                    document.image_analysis_status = "not_applicable"
                    document.images_extracted_count = 0
                    document.images_analyzed_count = 0
                    await session.commit()
                    return ImageAnalysisResult(
                        success=True,
                        stats=stats,
                    )

                document.images_extracted_count = len(images)

                # Check vision availability
                vision_status = await self.check_vision_available()
                if not vision_status.available:
                    document.image_analysis_status = "skipped"
                    document.image_analysis_error = "No vision model configured"
                    await session.commit()
                    return ImageAnalysisResult(
                        success=False,
                        stats=stats,
                        error="No vision model configured",
                    )

                # Process images
                captions = []
                images_to_process = images[:max_images] if max_images > 0 else images

                for i, img in enumerate(images_to_process):
                    # Skip small images
                    if hasattr(img, 'data') and len(img.data) < min_size_kb * 1024:
                        stats.images_skipped_small += 1
                        continue

                    # Compute hash
                    image_hash = self.compute_image_hash(img.data if hasattr(img, 'data') else b'')

                    # Check for duplicate
                    if skip_duplicates and enable_dedup:
                        cached = await self.find_cached_caption(session, image_hash)
                        if cached:
                            captions.append(cached)
                            stats.cached_used += 1
                            stats.images_skipped_duplicate += 1
                            continue

                    # Analyze with vision model
                    try:
                        caption = await self._multimodal.caption_image(img)
                        captions.append(caption)
                        stats.newly_analyzed += 1
                        stats.images_analyzed += 1

                        # Cache the result
                        await self.cache_caption(
                            session,
                            image_hash,
                            caption,
                            element_type=getattr(img, 'element_type', 'image'),
                            provider=vision_status.provider,
                            model=vision_status.model,
                            document_id=doc_uuid,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to analyze image",
                            document_id=document_id,
                            image_index=i,
                            error=str(e),
                        )
                        stats.images_failed += 1

                # Update document with results
                document.images_analyzed_count = stats.images_analyzed + stats.cached_used
                document.image_analysis_status = "completed"
                document.image_analysis_completed_at = datetime.utcnow()
                document.image_analysis_error = None
                await session.commit()

                stats.total_time_ms = int((time.time() - start_time) * 1000)

                logger.info(
                    "Image analysis completed",
                    document_id=document_id,
                    stats=stats.__dict__,
                )

                return ImageAnalysisResult(
                    success=True,
                    stats=stats,
                    captions=captions,
                )

            except Exception as e:
                document.image_analysis_status = "failed"
                document.image_analysis_error = str(e)
                await session.commit()

                logger.error(
                    "Image analysis failed",
                    document_id=document_id,
                    error=str(e),
                )

                return ImageAnalysisResult(
                    success=False,
                    stats=stats,
                    error=str(e),
                )

        if db:
            return await _analyze(db)
        else:
            async with async_session_context() as session:
                return await _analyze(session)

    async def get_unanalyzed_document_count(
        self,
        db: AsyncSession,
        user_id: Optional[str] = None,
        access_tier_level: int = 100,
    ) -> int:
        """Get count of documents with unanalyzed images."""
        from backend.db.models import AccessTier

        query = (
            select(func.count(Document.id))
            .join(AccessTier, Document.access_tier_id == AccessTier.id)
            .where(
                AccessTier.level <= access_tier_level,
                Document.images_extracted_count > 0,
                Document.image_analysis_status.in_(["pending", "skipped", "failed"]),
            )
        )

        result = await db.scalar(query)
        return result or 0


# =============================================================================
# Singleton Instance
# =============================================================================

_service_instance: Optional[ImageAnalysisService] = None
_service_lock = threading.Lock()


def get_image_analysis_service() -> ImageAnalysisService:
    """Get or create the image analysis service singleton."""
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ImageAnalysisService()

    return _service_instance
