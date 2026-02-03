"""
MinHash/LSH Deduplication Service
=================================

Provides O(n) approximate deduplication using MinHash signatures
and Locality-Sensitive Hashing (LSH).

Benefits:
- O(n) instead of O(n^2) for duplicate detection
- Configurable accuracy vs speed trade-off
- Memory efficient for large document sets
- Graceful fallback to exact Jaccard if datasketch unavailable

Usage:
    dedup = MinHashDeduplicator()

    # Add documents
    for doc_id, text in documents:
        dedup.add_document(doc_id, text)

    # Find duplicates
    duplicates = dedup.find_duplicates(threshold=0.8)

    # Or check single document
    is_dup, similar_id = dedup.is_duplicate(new_text, threshold=0.8)
"""

import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)

# Try to import datasketch for MinHash/LSH
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    MinHash = None
    MinHashLSH = None
    logger.info("datasketch not installed, using exact Jaccard fallback")


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""
    is_duplicate: bool
    similar_doc_id: Optional[str] = None
    similarity: float = 0.0
    method: str = "minhash"  # "minhash" or "exact_jaccard"


@dataclass
class DuplicateCluster:
    """A cluster of duplicate/near-duplicate documents."""
    canonical_id: str
    member_ids: List[str]
    similarity_scores: Dict[str, float] = field(default_factory=dict)


class MinHashDeduplicator:
    """
    MinHash-based document deduplicator with LSH indexing.

    Provides O(n) duplicate detection instead of O(n^2) pairwise comparison.

    Features:
    - MinHash signatures for fast approximate Jaccard similarity
    - LSH index for O(1) candidate retrieval
    - Configurable number of permutations (accuracy vs memory)
    - Automatic fallback to exact Jaccard if datasketch unavailable
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        weights: Tuple[float, float] = (0.5, 0.5),
        shingle_size: int = 3,
    ):
        """
        Initialize the deduplicator.

        Args:
            num_perm: Number of permutations for MinHash (more = accurate, slower)
            threshold: Default similarity threshold for duplicates
            weights: LSH weights for (false positive, false negative) trade-off
            shingle_size: Size of character shingles (3-5 recommended)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size

        # MinHash signatures storage
        self._signatures: Dict[str, Any] = {}  # doc_id -> MinHash or word set
        self._texts: Dict[str, str] = {}  # doc_id -> original text (for fallback)

        # LSH index (if datasketch available)
        self._lsh: Optional[Any] = None
        self._using_minhash = HAS_DATASKETCH

        if HAS_DATASKETCH:
            self._lsh = MinHashLSH(
                threshold=threshold,
                num_perm=num_perm,
                weights=weights,
            )
            logger.debug(
                "MinHash LSH initialized",
                num_perm=num_perm,
                threshold=threshold,
            )
        else:
            logger.debug("Using exact Jaccard fallback (datasketch not available)")

    def _get_shingles(self, text: str) -> Set[str]:
        """Extract character shingles from text."""
        text = text.lower().strip()
        if len(text) < self.shingle_size:
            return {text}

        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingles.add(text[i:i + self.shingle_size])
        return shingles

    def _get_word_set(self, text: str) -> Set[str]:
        """Extract word set from text (for Jaccard)."""
        return set(text.lower().split())

    def _create_minhash(self, shingles: Set[str]) -> Any:
        """Create MinHash signature from shingles."""
        if not HAS_DATASKETCH:
            return None

        m = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        return m

    def _exact_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute exact Jaccard similarity (fallback)."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def add_document(
        self,
        doc_id: str,
        text: str,
        use_shingles: bool = True,
    ) -> None:
        """
        Add a document to the deduplicator index.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            use_shingles: Use character shingles (True) or word tokens (False)
        """
        if not text or not text.strip():
            return

        # Store original text for fallback
        self._texts[doc_id] = text

        if self._using_minhash:
            # Create MinHash signature
            tokens = self._get_shingles(text) if use_shingles else self._get_word_set(text)
            minhash = self._create_minhash(tokens)
            self._signatures[doc_id] = minhash

            # Add to LSH index
            try:
                self._lsh.insert(doc_id, minhash)
            except ValueError:
                # Document already exists, update it
                pass
        else:
            # Store word set for exact Jaccard
            self._signatures[doc_id] = self._get_word_set(text)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id in self._signatures:
            if self._using_minhash and self._lsh:
                try:
                    self._lsh.remove(doc_id)
                except KeyError:
                    pass
            del self._signatures[doc_id]

        if doc_id in self._texts:
            del self._texts[doc_id]

    def is_duplicate(
        self,
        text: str,
        threshold: Optional[float] = None,
        use_shingles: bool = True,
    ) -> DeduplicationResult:
        """
        Check if text is a duplicate of any indexed document.

        Args:
            text: Text to check
            threshold: Similarity threshold (uses default if None)
            use_shingles: Use character shingles (True) or word tokens (False)

        Returns:
            DeduplicationResult with duplicate status and similar doc info
        """
        if not text or not text.strip():
            return DeduplicationResult(is_duplicate=False)

        threshold = threshold or self.threshold

        if self._using_minhash:
            return self._check_duplicate_minhash(text, threshold, use_shingles)
        else:
            return self._check_duplicate_exact(text, threshold)

    def _check_duplicate_minhash(
        self,
        text: str,
        threshold: float,
        use_shingles: bool,
    ) -> DeduplicationResult:
        """Check for duplicates using MinHash LSH."""
        tokens = self._get_shingles(text) if use_shingles else self._get_word_set(text)
        query_minhash = self._create_minhash(tokens)

        # Query LSH for candidates
        candidates = self._lsh.query(query_minhash)

        if not candidates:
            return DeduplicationResult(
                is_duplicate=False,
                method="minhash",
            )

        # Find best match among candidates
        best_match = None
        best_similarity = 0.0

        for doc_id in candidates:
            if doc_id in self._signatures:
                sim = query_minhash.jaccard(self._signatures[doc_id])
                if sim >= threshold and sim > best_similarity:
                    best_similarity = sim
                    best_match = doc_id

        if best_match:
            return DeduplicationResult(
                is_duplicate=True,
                similar_doc_id=best_match,
                similarity=best_similarity,
                method="minhash",
            )

        return DeduplicationResult(
            is_duplicate=False,
            method="minhash",
        )

    def _check_duplicate_exact(
        self,
        text: str,
        threshold: float,
    ) -> DeduplicationResult:
        """Check for duplicates using exact Jaccard (fallback)."""
        query_words = self._get_word_set(text)

        best_match = None
        best_similarity = 0.0

        for doc_id, doc_words in self._signatures.items():
            sim = self._exact_jaccard(query_words, doc_words)
            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = doc_id

        if best_match:
            return DeduplicationResult(
                is_duplicate=True,
                similar_doc_id=best_match,
                similarity=best_similarity,
                method="exact_jaccard",
            )

        return DeduplicationResult(
            is_duplicate=False,
            method="exact_jaccard",
        )

    def find_all_duplicates(
        self,
        threshold: Optional[float] = None,
    ) -> List[DuplicateCluster]:
        """
        Find all duplicate clusters in the indexed documents.

        Uses Union-Find for efficient clustering.

        Args:
            threshold: Similarity threshold (uses default if None)

        Returns:
            List of duplicate clusters
        """
        threshold = threshold or self.threshold

        if self._using_minhash:
            return self._find_duplicates_minhash(threshold)
        else:
            return self._find_duplicates_exact(threshold)

    def _find_duplicates_minhash(self, threshold: float) -> List[DuplicateCluster]:
        """Find duplicates using MinHash."""
        # Union-Find data structure
        parent: Dict[str, str] = {doc_id: doc_id for doc_id in self._signatures}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Find similar pairs
        for doc_id, minhash in self._signatures.items():
            candidates = self._lsh.query(minhash)
            for candidate_id in candidates:
                if candidate_id != doc_id and candidate_id in self._signatures:
                    sim = minhash.jaccard(self._signatures[candidate_id])
                    if sim >= threshold:
                        union(doc_id, candidate_id)

        # Build clusters
        clusters: Dict[str, List[str]] = {}
        for doc_id in self._signatures:
            root = find(doc_id)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(doc_id)

        # Filter to only clusters with duplicates
        result = []
        for canonical_id, members in clusters.items():
            if len(members) > 1:
                result.append(DuplicateCluster(
                    canonical_id=canonical_id,
                    member_ids=members,
                ))

        return result

    def _find_duplicates_exact(self, threshold: float) -> List[DuplicateCluster]:
        """Find duplicates using exact Jaccard (O(n^2) fallback)."""
        doc_ids = list(self._signatures.keys())
        n = len(doc_ids)

        # Union-Find
        parent = {doc_id: doc_id for doc_id in doc_ids}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # O(n^2) pairwise comparison
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._exact_jaccard(
                    self._signatures[doc_ids[i]],
                    self._signatures[doc_ids[j]],
                )
                if sim >= threshold:
                    union(doc_ids[i], doc_ids[j])

        # Build clusters
        clusters: Dict[str, List[str]] = {}
        for doc_id in doc_ids:
            root = find(doc_id)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(doc_id)

        result = []
        for canonical_id, members in clusters.items():
            if len(members) > 1:
                result.append(DuplicateCluster(
                    canonical_id=canonical_id,
                    member_ids=members,
                ))

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplicator statistics."""
        return {
            "num_documents": len(self._signatures),
            "using_minhash": self._using_minhash,
            "num_permutations": self.num_perm,
            "threshold": self.threshold,
            "method": "minhash_lsh" if self._using_minhash else "exact_jaccard",
            "complexity": "O(n)" if self._using_minhash else "O(n^2)",
        }

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._signatures.clear()
        self._texts.clear()
        if self._using_minhash:
            self._lsh = MinHashLSH(
                threshold=self.threshold,
                num_perm=self.num_perm,
            )


# Singleton instance
_deduplicator: Optional[MinHashDeduplicator] = None


def get_minhash_deduplicator(
    num_perm: int = 128,
    threshold: float = 0.8,
) -> MinHashDeduplicator:
    """
    Get or create the MinHash deduplicator singleton.

    Args:
        num_perm: Number of permutations (only used on first call)
        threshold: Similarity threshold (only used on first call)

    Returns:
        MinHashDeduplicator instance
    """
    global _deduplicator

    if _deduplicator is None:
        _deduplicator = MinHashDeduplicator(
            num_perm=num_perm,
            threshold=threshold,
        )

    return _deduplicator


def is_minhash_available() -> bool:
    """Check if MinHash/LSH is available."""
    return HAS_DATASKETCH
