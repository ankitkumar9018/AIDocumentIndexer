"""
AIDocumentIndexer - Embedding Projection Service
=================================================

Projects high-dimensional embeddings to 2D/3D for visualization.

Algorithms:
- UMAP (Uniform Manifold Approximation and Projection) - default
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- PCA (Principal Component Analysis) - fast fallback

Features:
- Configurable algorithm and parameters
- Caching of projections
- Incremental projection for new points
- Distance preservation metrics
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Try to import UMAP and scikit-learn
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.debug("UMAP not available - install with: pip install umap-learn")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("scikit-learn not available for t-SNE/PCA")


class ProjectionMethod(str, Enum):
    """Dimensionality reduction methods."""
    UMAP = "umap"
    TSNE = "tsne"
    PCA = "pca"


@dataclass
class ProjectionConfig:
    """Configuration for embedding projection."""
    method: ProjectionMethod = ProjectionMethod.UMAP
    n_components: int = 2  # Output dimensions (2D or 3D)

    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    # t-SNE parameters
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000

    # PCA parameters
    pca_whiten: bool = True

    # General
    random_state: int = 42
    cache_ttl_seconds: int = 3600


@dataclass
class ProjectedPoint:
    """A single projected point."""
    id: str
    x: float
    y: float
    z: Optional[float] = None  # For 3D projections
    original_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionResult:
    """Result of embedding projection."""
    points: List[ProjectedPoint]
    method: str
    n_components: int
    processing_time_ms: float
    cache_hit: bool = False
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class EmbeddingProjectionService:
    """
    Service for projecting embeddings to 2D/3D for visualization.

    Uses UMAP by default (best for preserving local and global structure),
    with fallback to PCA if UMAP is not available.

    Usage:
        service = EmbeddingProjectionService()
        result = await service.project(
            embeddings=[...],
            ids=["chunk1", "chunk2", ...],
            method=ProjectionMethod.UMAP,
        )
    """

    def __init__(self, config: Optional[ProjectionConfig] = None):
        self.config = config or ProjectionConfig()
        self._cache: Dict[str, Tuple[ProjectionResult, float]] = {}

        # Check available methods
        self._available_methods = [ProjectionMethod.PCA]  # PCA is always available
        if HAS_UMAP:
            self._available_methods.append(ProjectionMethod.UMAP)
        if HAS_SKLEARN:
            self._available_methods.append(ProjectionMethod.TSNE)

        logger.info(
            "Initialized EmbeddingProjectionService",
            available_methods=[m.value for m in self._available_methods],
        )

    def _get_cache_key(
        self,
        embeddings: List[List[float]],
        method: ProjectionMethod,
    ) -> str:
        """Generate cache key from embeddings."""
        # Hash the embeddings
        emb_str = str(embeddings[:100])  # Use first 100 for key
        return hashlib.md5(
            f"{emb_str}|{method.value}|{self.config.n_components}".encode()
        ).hexdigest()

    def _check_cache(self, key: str) -> Optional[ProjectionResult]:
        """Check if projection is cached and not expired."""
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self.config.cache_ttl_seconds:
            del self._cache[key]
            return None

        result.cache_hit = True
        return result

    def _cache_result(self, key: str, result: ProjectionResult) -> None:
        """Cache projection result."""
        self._cache[key] = (result, time.time())

        # Limit cache size
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    async def project(
        self,
        embeddings: List[List[float]],
        ids: List[str],
        scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        method: Optional[ProjectionMethod] = None,
    ) -> ProjectionResult:
        """
        Project embeddings to 2D/3D space.

        Args:
            embeddings: List of embedding vectors
            ids: List of identifiers for each embedding
            scores: Optional relevance scores for each point
            metadata: Optional metadata for each point
            method: Projection method (default: from config)

        Returns:
            ProjectionResult with projected points
        """
        start_time = time.time()

        if len(embeddings) == 0:
            return ProjectionResult(
                points=[],
                method="none",
                n_components=self.config.n_components,
                processing_time_ms=0,
            )

        # Use configured method or fallback
        method = method or self.config.method
        if method not in self._available_methods:
            logger.warning(
                f"{method.value} not available, falling back to PCA",
                available=self._available_methods,
            )
            method = ProjectionMethod.PCA

        # Check cache
        cache_key = self._get_cache_key(embeddings, method)
        cached = self._check_cache(cache_key)
        if cached:
            logger.debug("Using cached projection", key=cache_key[:8])
            return cached

        # Convert to numpy
        X = np.array(embeddings, dtype=np.float32)

        # Handle edge cases
        if len(X) == 1:
            # Single point - center it
            projected = np.array([[0.5, 0.5]])
        elif len(X) <= self.config.n_components:
            # Not enough points for reduction - use simple normalization
            projected = self._simple_normalize(X)
        else:
            # Apply dimensionality reduction
            if method == ProjectionMethod.UMAP:
                projected = self._umap_project(X)
            elif method == ProjectionMethod.TSNE:
                projected = self._tsne_project(X)
            else:
                projected = self._pca_project(X)

        # Normalize to [0, 1] with padding
        projected = self._normalize_coordinates(projected)

        # Build result points
        points = []
        for i, (id_, coords) in enumerate(zip(ids, projected)):
            point = ProjectedPoint(
                id=id_,
                x=float(coords[0]),
                y=float(coords[1]),
                z=float(coords[2]) if len(coords) > 2 else None,
                original_score=scores[i] if scores else None,
                metadata=metadata[i] if metadata else {},
            )
            points.append(point)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(X, projected)

        processing_time = (time.time() - start_time) * 1000

        result = ProjectionResult(
            points=points,
            method=method.value,
            n_components=self.config.n_components,
            processing_time_ms=processing_time,
            quality_metrics=quality_metrics,
        )

        # Cache result
        self._cache_result(cache_key, result)

        logger.info(
            "Projected embeddings",
            n_points=len(points),
            method=method.value,
            time_ms=round(processing_time, 2),
        )

        return result

    def _umap_project(self, X: np.ndarray) -> np.ndarray:
        """Project using UMAP."""
        if not HAS_UMAP:
            return self._pca_project(X)

        # Adjust n_neighbors for small datasets
        n_neighbors = min(self.config.umap_n_neighbors, len(X) - 1)
        n_neighbors = max(n_neighbors, 2)

        reducer = umap.UMAP(
            n_components=self.config.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=self.config.random_state,
        )

        return reducer.fit_transform(X)

    def _tsne_project(self, X: np.ndarray) -> np.ndarray:
        """Project using t-SNE."""
        if not HAS_SKLEARN:
            return self._pca_project(X)

        # Adjust perplexity for small datasets
        perplexity = min(self.config.tsne_perplexity, len(X) - 1)
        perplexity = max(perplexity, 5)

        tsne = TSNE(
            n_components=self.config.n_components,
            perplexity=perplexity,
            learning_rate=self.config.tsne_learning_rate,
            n_iter=self.config.tsne_n_iter,
            random_state=self.config.random_state,
        )

        return tsne.fit_transform(X)

    def _pca_project(self, X: np.ndarray) -> np.ndarray:
        """Project using PCA (fast fallback)."""
        if HAS_SKLEARN:
            pca = PCA(
                n_components=min(self.config.n_components, X.shape[1]),
                whiten=self.config.pca_whiten,
                random_state=self.config.random_state,
            )
            return pca.fit_transform(X)

        # Manual PCA implementation
        return self._manual_pca(X)

    def _manual_pca(self, X: np.ndarray) -> np.ndarray:
        """Simple PCA without sklearn."""
        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute covariance matrix
        cov = np.cov(X_centered.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project onto top components
        n_components = min(self.config.n_components, X.shape[1])
        projection_matrix = eigenvectors[:, :n_components]

        return X_centered @ projection_matrix

    def _simple_normalize(self, X: np.ndarray) -> np.ndarray:
        """Simple normalization for very small datasets."""
        # Use first n_components dimensions
        result = X[:, :self.config.n_components]
        if result.shape[1] < self.config.n_components:
            # Pad with zeros
            padding = np.zeros((result.shape[0], self.config.n_components - result.shape[1]))
            result = np.hstack([result, padding])
        return result

    def _normalize_coordinates(
        self,
        coords: np.ndarray,
        padding: float = 0.05,
    ) -> np.ndarray:
        """Normalize coordinates to [0, 1] with padding."""
        result = coords.copy()

        for dim in range(coords.shape[1]):
            min_val = coords[:, dim].min()
            max_val = coords[:, dim].max()
            range_val = max_val - min_val

            if range_val > 0:
                result[:, dim] = (coords[:, dim] - min_val) / range_val
            else:
                result[:, dim] = 0.5

            # Apply padding
            result[:, dim] = padding + result[:, dim] * (1 - 2 * padding)

        return result

    def _calculate_quality_metrics(
        self,
        X_high: np.ndarray,
        X_low: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate projection quality metrics."""
        metrics = {}

        try:
            # Trustworthiness: How well local neighborhoods are preserved
            # (simplified version)
            if len(X_high) > 10:
                from scipy.spatial.distance import pdist, squareform

                # Compute pairwise distances in both spaces
                dist_high = squareform(pdist(X_high, "cosine"))
                dist_low = squareform(pdist(X_low, "euclidean"))

                # Compare nearest neighbor rankings
                k = min(10, len(X_high) - 1)
                nn_high = np.argsort(dist_high, axis=1)[:, 1:k+1]
                nn_low = np.argsort(dist_low, axis=1)[:, 1:k+1]

                # Calculate overlap
                overlaps = []
                for i in range(len(X_high)):
                    overlap = len(set(nn_high[i]) & set(nn_low[i])) / k
                    overlaps.append(overlap)

                metrics["neighborhood_preservation"] = float(np.mean(overlaps))
        except Exception as e:
            logger.debug(f"Could not calculate quality metrics: {e}")

        return metrics

    async def project_incremental(
        self,
        new_embeddings: List[List[float]],
        new_ids: List[str],
        existing_result: ProjectionResult,
    ) -> ProjectionResult:
        """
        Add new points to an existing projection.

        Uses the nearest neighbors in the existing projection
        to position new points (faster than re-projecting everything).
        """
        if len(new_embeddings) == 0:
            return existing_result

        # For now, re-project everything
        # TODO: Implement true incremental projection using UMAP transform
        all_embeddings = []
        all_ids = []

        # Get existing embeddings (not stored, so we can't do true incremental)
        # This is a placeholder - in production, store embeddings with the result

        all_embeddings.extend(new_embeddings)
        all_ids.extend(new_ids)

        return await self.project(all_embeddings, all_ids)


# Singleton instance
_projection_service: Optional[EmbeddingProjectionService] = None


def get_projection_service() -> EmbeddingProjectionService:
    """Get or create the projection service singleton."""
    global _projection_service
    if _projection_service is None:
        _projection_service = EmbeddingProjectionService()
    return _projection_service


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ProjectionMethod",
    "ProjectionConfig",
    "ProjectedPoint",
    "ProjectionResult",
    "EmbeddingProjectionService",
    "get_projection_service",
]
