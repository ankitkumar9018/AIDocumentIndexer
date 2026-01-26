"""
AIDocumentIndexer - ColPali Visual Document Retriever
=====================================================

Phase 28: ColPali for Multimodal Document Retrieval (ICLR 2025)

Implements visual document retrieval using ColPali - a Vision Language Model
that uses ColBERT-style late interaction for visual features.

Key Benefits:
- Retrieves documents directly from images without OCR
- ColBERT-style late interaction for visual features
- Outperforms all baselines on ViDoRe benchmark
- +40% accuracy on visual document retrieval

Based on: https://arxiv.org/abs/2407.01449

Usage:
    retriever = await get_colpali_retriever()

    # Index document images
    await retriever.index_images(images, document_ids)

    # Search with text query
    results = await retriever.search("What is the revenue trend?", top_k=5)
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# Check for ColPali availability
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    from colpali_engine.utils.torch_utils import get_torch_device
    import torch
    HAS_COLPALI = True
except ImportError:
    HAS_COLPALI = False
    logger.info("ColPali not available - install with: pip install colpali-engine")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ColPaliConfig:
    """Configuration for ColPali retriever."""

    # Model settings
    model_name: str = "vidore/colpali-v1.2"  # Latest ColPali model
    device: str = "auto"  # auto, cpu, cuda, mps

    # Index settings
    index_path: str = "./data/colpali_index"
    index_name: str = "visual_documents"

    # Search settings
    top_k: int = 10
    batch_size: int = 4  # Images per batch (memory-intensive)

    # Image processing
    max_image_size: int = 1024  # Max dimension for images

    # Cache settings
    use_cache: bool = True
    cache_embeddings: bool = True


@dataclass
class ColPaliSearchResult:
    """Search result from ColPali retrieval."""

    document_id: str
    image_id: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ColPali Retriever
# =============================================================================

class ColPaliRetriever:
    """
    ColPali visual document retriever.

    Phase 28: Implements visual document retrieval using ColPali's
    late interaction scoring for vision-language matching.

    Performance (from ViDoRe benchmark):
    - Outperforms all text-based baselines on visual documents
    - No OCR required - works directly on images
    - ColBERT-style scoring for fine-grained matching
    """

    def __init__(self, config: Optional[ColPaliConfig] = None):
        """
        Initialize ColPali retriever.

        Args:
            config: ColPali configuration
        """
        self.config = config or ColPaliConfig()
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._device: Optional[str] = None
        self._index_built = False
        self._lock = asyncio.Lock()

        # In-memory index storage
        self._image_embeddings: Dict[str, torch.Tensor] = {}
        self._image_metadata: Dict[str, Dict[str, Any]] = {}

        # Ensure index directory exists
        Path(self.config.index_path).mkdir(parents=True, exist_ok=True)

        if not HAS_COLPALI:
            logger.warning(
                "ColPali not installed - visual document retrieval disabled. "
                "Install with: pip install colpali-engine"
            )

    @property
    def is_available(self) -> bool:
        """Check if ColPali retrieval is available."""
        return HAS_COLPALI

    async def initialize(self) -> bool:
        """
        Initialize the ColPali model (lazy loading).

        Returns:
            True if initialization successful
        """
        if not HAS_COLPALI:
            logger.error("ColPali not available - cannot initialize")
            return False

        if self._model is not None:
            return True

        async with self._lock:
            if self._model is not None:
                return True

            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._load_model)

                logger.info(
                    "ColPali model initialized",
                    model=self.config.model_name,
                    device=self._device,
                )
                return True

            except Exception as e:
                logger.error("Failed to initialize ColPali", error=str(e))
                return False

    def _load_model(self):
        """Load ColPali model and processor (sync, run in executor)."""
        # Determine device
        if self.config.device == "auto":
            self._device = get_torch_device()
        else:
            self._device = self.config.device

        logger.info(f"Loading ColPali model on {self._device}...")

        # Load model
        self._model = ColPali.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self._device != "cpu" else torch.float32,
            device_map=self._device,
        ).eval()

        # Load processor
        self._processor = ColPaliProcessor.from_pretrained(self.config.model_name)

        logger.info("ColPali model loaded successfully")

    async def index_images(
        self,
        images: List[Union[Image.Image, str, bytes]],
        document_ids: List[str],
        image_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        force_rebuild: bool = False,
    ) -> bool:
        """
        Index document images for retrieval.

        Args:
            images: List of PIL Images, file paths, or bytes
            document_ids: Document ID for each image
            image_ids: Optional unique ID for each image (auto-generated if not provided)
            metadata: Optional metadata for each image
            force_rebuild: If True, reindex even if already indexed

        Returns:
            True if indexing successful
        """
        if not await self.initialize():
            return False

        if len(images) != len(document_ids):
            raise ValueError("images and document_ids must have same length")

        # Generate image IDs if not provided
        if image_ids is None:
            image_ids = [f"{doc_id}_img_{i}" for i, doc_id in enumerate(document_ids)]

        # Default metadata
        if metadata is None:
            metadata = [{} for _ in images]

        async with self._lock:
            try:
                loop = asyncio.get_running_loop()

                # Process in batches
                for batch_start in range(0, len(images), self.config.batch_size):
                    batch_end = min(batch_start + self.config.batch_size, len(images))

                    batch_images = images[batch_start:batch_end]
                    batch_doc_ids = document_ids[batch_start:batch_end]
                    batch_img_ids = image_ids[batch_start:batch_end]
                    batch_metadata = metadata[batch_start:batch_end]

                    # Skip already indexed (unless force_rebuild)
                    if not force_rebuild:
                        new_indices = [
                            i for i, img_id in enumerate(batch_img_ids)
                            if img_id not in self._image_embeddings
                        ]
                        if not new_indices:
                            continue

                        batch_images = [batch_images[i] for i in new_indices]
                        batch_doc_ids = [batch_doc_ids[i] for i in new_indices]
                        batch_img_ids = [batch_img_ids[i] for i in new_indices]
                        batch_metadata = [batch_metadata[i] for i in new_indices]

                    # Load and preprocess images
                    pil_images = await loop.run_in_executor(
                        None,
                        lambda: [self._load_image(img) for img in batch_images]
                    )

                    # Generate embeddings
                    embeddings = await loop.run_in_executor(
                        None,
                        self._embed_images,
                        pil_images
                    )

                    # Store embeddings and metadata
                    for i, (img_id, doc_id, emb, meta) in enumerate(
                        zip(batch_img_ids, batch_doc_ids, embeddings, batch_metadata)
                    ):
                        self._image_embeddings[img_id] = emb
                        self._image_metadata[img_id] = {
                            "document_id": doc_id,
                            "image_id": img_id,
                            **meta
                        }

                    logger.debug(
                        f"Indexed batch {batch_start//self.config.batch_size + 1}",
                        images=len(batch_images)
                    )

                self._index_built = True

                logger.info(
                    "ColPali indexing complete",
                    total_images=len(self._image_embeddings),
                )
                return True

            except Exception as e:
                logger.error("ColPali indexing failed", error=str(e))
                return False

    def _load_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """Load and preprocess an image."""
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Resize if too large
        max_size = self.config.max_image_size
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        return pil_image

    def _embed_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Generate embeddings for images (sync, run in executor)."""
        with torch.no_grad():
            # Process images
            batch_images = self._processor.process_images(images).to(self._device)

            # Generate embeddings
            embeddings = self._model(**batch_images)

            # Return as list of tensors (one per image)
            return [emb.cpu() for emb in embeddings]

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[ColPaliSearchResult]:
        """
        Search for relevant document images using a text query.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            document_ids: Optional filter to specific documents

        Returns:
            List of search results sorted by relevance
        """
        if not await self.initialize():
            return []

        if not self._image_embeddings:
            logger.warning("No images indexed - cannot search")
            return []

        top_k = top_k or self.config.top_k

        try:
            loop = asyncio.get_running_loop()

            # Embed query
            query_embedding = await loop.run_in_executor(
                None,
                self._embed_query,
                query
            )

            # Score against all indexed images
            scores = await loop.run_in_executor(
                None,
                self._compute_scores,
                query_embedding,
                document_ids
            )

            # Sort by score and return top_k
            sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            for rank, (img_id, score) in enumerate(sorted_results):
                meta = self._image_metadata.get(img_id, {})
                results.append(ColPaliSearchResult(
                    document_id=meta.get("document_id", ""),
                    image_id=img_id,
                    score=float(score),
                    rank=rank,
                    metadata=meta,
                ))

            logger.debug(
                "ColPali search complete",
                query=query[:50],
                results=len(results),
            )

            return results

        except Exception as e:
            logger.error("ColPali search failed", error=str(e))
            return []

    def _embed_query(self, query: str) -> torch.Tensor:
        """Embed a text query (sync, run in executor)."""
        with torch.no_grad():
            # Process query
            batch_query = self._processor.process_queries([query]).to(self._device)

            # Generate embedding
            query_embedding = self._model(**batch_query)

            return query_embedding[0].cpu()

    def _compute_scores(
        self,
        query_embedding: torch.Tensor,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Compute similarity scores (sync, run in executor)."""
        scores = []

        for img_id, img_embedding in self._image_embeddings.items():
            # Filter by document IDs if provided
            if document_ids:
                meta = self._image_metadata.get(img_id, {})
                if meta.get("document_id") not in document_ids:
                    continue

            # ColBERT-style late interaction scoring
            # Max-sim: for each query token, find max similarity with image tokens
            # Then sum over all query tokens
            similarity = torch.einsum(
                "qd,id->qi",
                query_embedding.float(),
                img_embedding.float()
            )
            score = similarity.max(dim=1).values.sum().item()

            scores.append((img_id, score))

        return scores

    async def index_pdf_pages(
        self,
        pdf_path: str,
        document_id: str,
        dpi: int = 150,
    ) -> bool:
        """
        Index all pages of a PDF document.

        Args:
            pdf_path: Path to PDF file
            document_id: Document ID for the PDF
            dpi: Resolution for rendering PDF pages

        Returns:
            True if indexing successful
        """
        try:
            # Try pdf2image if available
            try:
                from pdf2image import convert_from_path

                loop = asyncio.get_running_loop()
                images = await loop.run_in_executor(
                    None,
                    lambda: convert_from_path(pdf_path, dpi=dpi)
                )

            except ImportError:
                logger.error("pdf2image not installed - cannot index PDF pages")
                return False

            # Generate image IDs
            image_ids = [f"{document_id}_page_{i}" for i in range(len(images))]
            document_ids = [document_id] * len(images)
            metadata = [{"page_number": i, "source": "pdf"} for i in range(len(images))]

            return await self.index_images(
                images=images,
                document_ids=document_ids,
                image_ids=image_ids,
                metadata=metadata,
            )

        except Exception as e:
            logger.error("Failed to index PDF pages", error=str(e))
            return False

    async def remove_document(self, document_id: str) -> bool:
        """
        Remove all images for a document from the index.

        Args:
            document_id: Document ID to remove

        Returns:
            True if removal successful
        """
        async with self._lock:
            to_remove = [
                img_id for img_id, meta in self._image_metadata.items()
                if meta.get("document_id") == document_id
            ]

            for img_id in to_remove:
                self._image_embeddings.pop(img_id, None)
                self._image_metadata.pop(img_id, None)

            logger.info(
                "Removed document from ColPali index",
                document_id=document_id,
                images_removed=len(to_remove),
            )

            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "available": self.is_available,
            "model_loaded": self._model is not None,
            "index_built": self._index_built,
            "indexed_images": len(self._image_embeddings),
            "unique_documents": len(set(
                m.get("document_id") for m in self._image_metadata.values()
            )),
            "model_name": self.config.model_name,
            "device": self._device,
        }

    async def save_index(self, path: Optional[str] = None) -> bool:
        """
        Save the index to disk.

        Args:
            path: Save path (uses config.index_path if not provided)

        Returns:
            True if save successful
        """
        if not HAS_COLPALI:
            return False

        save_path = Path(path or self.config.index_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save embeddings
            embeddings_path = save_path / "embeddings.pt"
            torch.save(self._image_embeddings, embeddings_path)

            # Save metadata
            import json
            metadata_path = save_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self._image_metadata, f)

            logger.info("ColPali index saved", path=str(save_path))
            return True

        except Exception as e:
            logger.error("Failed to save ColPali index", error=str(e))
            return False

    async def load_index(self, path: Optional[str] = None) -> bool:
        """
        Load the index from disk.

        Args:
            path: Load path (uses config.index_path if not provided)

        Returns:
            True if load successful
        """
        if not HAS_COLPALI:
            return False

        load_path = Path(path or self.config.index_path)

        try:
            embeddings_path = load_path / "embeddings.pt"
            metadata_path = load_path / "metadata.json"

            if not embeddings_path.exists() or not metadata_path.exists():
                logger.warning("No saved index found", path=str(load_path))
                return False

            # Load embeddings
            self._image_embeddings = torch.load(embeddings_path)

            # Load metadata
            import json
            with open(metadata_path, "r") as f:
                self._image_metadata = json.load(f)

            self._index_built = True

            logger.info(
                "ColPali index loaded",
                path=str(load_path),
                images=len(self._image_embeddings),
            )
            return True

        except Exception as e:
            logger.error("Failed to load ColPali index", error=str(e))
            return False


# =============================================================================
# Singleton Management
# =============================================================================

_colpali_retriever: Optional[ColPaliRetriever] = None
_retriever_lock = asyncio.Lock()


async def get_colpali_retriever(
    config: Optional[ColPaliConfig] = None,
) -> ColPaliRetriever:
    """
    Get or create the global ColPali retriever instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        ColPali retriever instance
    """
    global _colpali_retriever

    if _colpali_retriever is None:
        async with _retriever_lock:
            if _colpali_retriever is None:
                _colpali_retriever = ColPaliRetriever(config)

    return _colpali_retriever


# =============================================================================
# Hybrid Visual + Text Search
# =============================================================================

async def hybrid_visual_text_search(
    query: str,
    text_results: List[Dict[str, Any]],
    colpali_retriever: Optional[ColPaliRetriever] = None,
    visual_weight: float = 0.4,
    text_weight: float = 0.6,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combine visual (ColPali) and text-based search results.

    Args:
        query: Search query
        text_results: Results from text-based retrieval
        colpali_retriever: ColPali retriever instance
        visual_weight: Weight for visual results (0-1)
        text_weight: Weight for text results (0-1)
        top_k: Number of results to return

    Returns:
        Combined and reranked results
    """
    if colpali_retriever is None:
        colpali_retriever = await get_colpali_retriever()

    # Get visual results
    visual_results = await colpali_retriever.search(query, top_k=top_k * 2)

    # Normalize text scores
    text_scores = {}
    if text_results:
        max_text_score = max(r.get("score", 0) for r in text_results) or 1.0
        for r in text_results:
            doc_id = r.get("document_id") or r.get("doc_id")
            if doc_id:
                text_scores[doc_id] = r.get("score", 0) / max_text_score

    # Normalize visual scores
    visual_scores = {}
    if visual_results:
        max_visual_score = max(r.score for r in visual_results) or 1.0
        for r in visual_results:
            if r.document_id:
                # Take max score if document has multiple images
                current = visual_scores.get(r.document_id, 0)
                visual_scores[r.document_id] = max(current, r.score / max_visual_score)

    # Combine scores
    all_doc_ids = set(text_scores.keys()) | set(visual_scores.keys())
    combined = []

    for doc_id in all_doc_ids:
        text_score = text_scores.get(doc_id, 0)
        visual_score = visual_scores.get(doc_id, 0)

        combined_score = (text_weight * text_score) + (visual_weight * visual_score)

        combined.append({
            "document_id": doc_id,
            "score": combined_score,
            "text_score": text_score,
            "visual_score": visual_score,
            "source": "hybrid_visual_text",
        })

    # Sort by combined score
    combined.sort(key=lambda x: x["score"], reverse=True)

    return combined[:top_k]
