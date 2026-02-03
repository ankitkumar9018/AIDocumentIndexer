"""
AIDocumentIndexer - Document DNA Service
==========================================

Instant document fingerprinting system that:
- Creates unique fingerprints for documents (content hash, structure hash, semantic hash)
- Enables fast O(1) deduplication
- Tracks document lineage and versions
- Detects plagiarism and content copying
- Identifies document families (related documents)
- Supports partial matching for modified documents
"""

import asyncio
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import struct

import structlog
import numpy as np

logger = structlog.get_logger(__name__)


class FingerprintType(str, Enum):
    """Types of fingerprints for documents."""
    CONTENT_HASH = "content_hash"  # Exact content hash (MD5/SHA256)
    STRUCTURE_HASH = "structure_hash"  # Document structure fingerprint
    SEMANTIC_HASH = "semantic_hash"  # Locality-sensitive hash for similarity
    MINHASH = "minhash"  # MinHash for Jaccard similarity
    SIMHASH = "simhash"  # SimHash for Hamming distance
    NGRAM_HASH = "ngram_hash"  # N-gram based fingerprint


class MatchType(str, Enum):
    """Types of document matches."""
    EXACT = "exact"  # Identical documents
    NEAR_DUPLICATE = "near_duplicate"  # Very similar (>95%)
    SIMILAR = "similar"  # Similar content (>80%)
    PARTIAL = "partial"  # Partial overlap (>50%)
    RELATED = "related"  # Some relation (>30%)
    DIFFERENT = "different"  # No significant relation


@dataclass
class DocumentDNA:
    """Complete DNA profile for a document."""
    document_id: str
    content_hash: str  # SHA256 of normalized content
    structure_hash: str  # Hash of document structure
    minhash_signature: List[int]  # MinHash signature (128 values)
    simhash_signature: int  # 64-bit SimHash
    ngram_fingerprints: Set[int]  # Set of n-gram hashes
    word_count: int
    unique_word_count: int
    avg_word_length: float
    paragraph_count: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "structure_hash": self.structure_hash,
            "minhash_signature": self.minhash_signature,
            "simhash_signature": self.simhash_signature,
            "ngram_fingerprint_count": len(self.ngram_fingerprints),
            "word_count": self.word_count,
            "unique_word_count": self.unique_word_count,
            "avg_word_length": self.avg_word_length,
            "paragraph_count": self.paragraph_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DNAMatch:
    """Result of comparing two document DNAs."""
    document_id_1: str
    document_id_2: str
    match_type: MatchType
    overall_similarity: float
    content_similarity: float
    structure_similarity: float
    semantic_similarity: float
    matching_segments: List[Tuple[str, str]]  # Pairs of matching text segments
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id_1": self.document_id_1,
            "document_id_2": self.document_id_2,
            "match_type": self.match_type.value,
            "overall_similarity": self.overall_similarity,
            "content_similarity": self.content_similarity,
            "structure_similarity": self.structure_similarity,
            "semantic_similarity": self.semantic_similarity,
            "matching_segment_count": len(self.matching_segments),
            "metadata": self.metadata,
        }


@dataclass
class DocumentLineage:
    """Tracks document versions and derivations."""
    document_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    changes_from_parent: Dict[str, Any] = field(default_factory=dict)


class DocumentDNAService:
    """
    Service for creating and comparing document fingerprints.

    Features:
    - Multiple fingerprinting algorithms
    - O(1) exact duplicate detection
    - Near-duplicate detection via LSH
    - Plagiarism detection
    - Version tracking
    """

    def __init__(
        self,
        num_minhash_permutations: int = 128,
        ngram_size: int = 5,
        simhash_bits: int = 64,
        near_duplicate_threshold: float = 0.95,
        similar_threshold: float = 0.80,
    ):
        self.num_permutations = num_minhash_permutations
        self.ngram_size = ngram_size
        self.simhash_bits = simhash_bits
        self.near_duplicate_threshold = near_duplicate_threshold
        self.similar_threshold = similar_threshold

        # Pre-generate random hash functions for MinHash
        np.random.seed(42)
        self._hash_funcs = [
            (np.random.randint(1, 2**31), np.random.randint(0, 2**31))
            for _ in range(num_minhash_permutations)
        ]

        # DNA database (in production, use Redis or dedicated store)
        self._dna_store: Dict[str, DocumentDNA] = {}
        self._content_hash_index: Dict[str, List[str]] = {}  # hash -> doc_ids
        self._minhash_lsh: Dict[int, Set[str]] = {}  # LSH bands

    async def create_dna(
        self,
        document_id: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentDNA:
        """
        Create a complete DNA profile for a document.

        Args:
            document_id: Unique document identifier
            content: Document text content
            title: Optional document title
            metadata: Optional metadata

        Returns:
            DocumentDNA with all fingerprints
        """
        # Normalize content
        normalized = self._normalize_text(content)
        words = normalized.split()

        # Create various fingerprints
        content_hash = self._create_content_hash(normalized)
        structure_hash = self._create_structure_hash(content)
        minhash_sig = self._create_minhash(normalized)
        simhash_sig = self._create_simhash(normalized)
        ngram_fps = self._create_ngram_fingerprints(normalized)

        # Calculate statistics
        word_count = len(words)
        unique_words = set(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])

        dna = DocumentDNA(
            document_id=document_id,
            content_hash=content_hash,
            structure_hash=structure_hash,
            minhash_signature=minhash_sig,
            simhash_signature=simhash_sig,
            ngram_fingerprints=ngram_fps,
            word_count=word_count,
            unique_word_count=len(unique_words),
            avg_word_length=float(avg_word_len),
            paragraph_count=paragraph_count,
            metadata=metadata or {},
        )

        # Store DNA
        self._store_dna(dna)

        logger.info(
            "Created document DNA",
            document_id=document_id,
            content_hash=content_hash[:16],
            word_count=word_count,
        )

        return dna

    async def find_duplicates(
        self,
        dna: DocumentDNA,
        threshold: float = 0.95,
    ) -> List[DNAMatch]:
        """
        Find duplicate or similar documents.

        Args:
            dna: DNA of document to check
            threshold: Minimum similarity threshold

        Returns:
            List of matches sorted by similarity
        """
        matches: List[DNAMatch] = []

        # First, check exact duplicates via content hash
        exact_matches = self._content_hash_index.get(dna.content_hash, [])
        for doc_id in exact_matches:
            if doc_id != dna.document_id:
                matches.append(DNAMatch(
                    document_id_1=dna.document_id,
                    document_id_2=doc_id,
                    match_type=MatchType.EXACT,
                    overall_similarity=1.0,
                    content_similarity=1.0,
                    structure_similarity=1.0,
                    semantic_similarity=1.0,
                    matching_segments=[],
                ))

        # Check near-duplicates via LSH
        candidates = self._find_lsh_candidates(dna)

        for candidate_id in candidates:
            if candidate_id == dna.document_id:
                continue
            if candidate_id in [m.document_id_2 for m in matches]:
                continue

            candidate_dna = self._dna_store.get(candidate_id)
            if not candidate_dna:
                continue

            match = await self.compare_dna(dna, candidate_dna)
            if match.overall_similarity >= threshold:
                matches.append(match)

        # Sort by similarity
        matches.sort(key=lambda m: m.overall_similarity, reverse=True)
        return matches

    async def compare_dna(
        self,
        dna1: DocumentDNA,
        dna2: DocumentDNA,
    ) -> DNAMatch:
        """
        Compare two document DNAs.

        Args:
            dna1: First document DNA
            dna2: Second document DNA

        Returns:
            DNAMatch with detailed comparison
        """
        # Content similarity (exact match check)
        content_sim = 1.0 if dna1.content_hash == dna2.content_hash else 0.0

        # Structure similarity
        structure_sim = 1.0 if dna1.structure_hash == dna2.structure_hash else 0.0

        # Semantic similarity via MinHash (Jaccard estimate)
        minhash_sim = self._minhash_similarity(
            dna1.minhash_signature,
            dna2.minhash_signature
        )

        # SimHash similarity (Hamming distance based)
        simhash_sim = self._simhash_similarity(
            dna1.simhash_signature,
            dna2.simhash_signature
        )

        # N-gram overlap
        ngram_sim = self._ngram_similarity(
            dna1.ngram_fingerprints,
            dna2.ngram_fingerprints
        )

        # Overall similarity (weighted combination)
        overall = (
            0.1 * content_sim +
            0.1 * structure_sim +
            0.3 * minhash_sim +
            0.2 * simhash_sim +
            0.3 * ngram_sim
        )

        # If content hash matches, it's exact
        if content_sim == 1.0:
            overall = 1.0

        # Determine match type
        if overall >= 1.0:
            match_type = MatchType.EXACT
        elif overall >= self.near_duplicate_threshold:
            match_type = MatchType.NEAR_DUPLICATE
        elif overall >= self.similar_threshold:
            match_type = MatchType.SIMILAR
        elif overall >= 0.5:
            match_type = MatchType.PARTIAL
        elif overall >= 0.3:
            match_type = MatchType.RELATED
        else:
            match_type = MatchType.DIFFERENT

        return DNAMatch(
            document_id_1=dna1.document_id,
            document_id_2=dna2.document_id,
            match_type=match_type,
            overall_similarity=overall,
            content_similarity=content_sim,
            structure_similarity=structure_sim,
            semantic_similarity=minhash_sim,
            matching_segments=[],  # Would require full text comparison
            metadata={
                "minhash_similarity": minhash_sim,
                "simhash_similarity": simhash_sim,
                "ngram_similarity": ngram_sim,
            },
        )

    async def detect_plagiarism(
        self,
        content: str,
        min_match_length: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Detect potential plagiarism in content.

        Args:
            content: Content to check
            min_match_length: Minimum matching segment length

        Returns:
            List of potential plagiarism matches
        """
        # Create temporary DNA
        temp_dna = await self.create_dna("temp_check", content)

        # Find similar documents
        matches = await self.find_duplicates(temp_dna, threshold=0.3)

        plagiarism_results = []
        for match in matches:
            if match.match_type in [MatchType.EXACT, MatchType.NEAR_DUPLICATE, MatchType.SIMILAR]:
                plagiarism_results.append({
                    "matched_document_id": match.document_id_2,
                    "similarity": match.overall_similarity,
                    "match_type": match.match_type.value,
                    "confidence": "high" if match.overall_similarity > 0.9 else "medium",
                })

        # Clean up temp DNA
        del self._dna_store["temp_check"]

        return plagiarism_results

    async def track_version(
        self,
        document_id: str,
        new_content: str,
        parent_document_id: Optional[str] = None,
    ) -> DocumentLineage:
        """
        Track document version and create lineage.

        Args:
            document_id: New document ID
            new_content: New content
            parent_document_id: Parent document if known

        Returns:
            DocumentLineage with version info
        """
        new_dna = await self.create_dna(document_id, new_content)

        if parent_document_id:
            parent_dna = self._dna_store.get(parent_document_id)
            if parent_dna:
                match = await self.compare_dna(parent_dna, new_dna)
                return DocumentLineage(
                    document_id=document_id,
                    parent_id=parent_document_id,
                    version=2,  # Would increment from parent
                    changes_from_parent={
                        "similarity_to_parent": match.overall_similarity,
                        "change_percentage": 1 - match.overall_similarity,
                    },
                )

        # Try to find parent automatically
        matches = await self.find_duplicates(new_dna, threshold=0.7)
        if matches:
            best_match = matches[0]
            return DocumentLineage(
                document_id=document_id,
                parent_id=best_match.document_id_2,
                version=2,
                changes_from_parent={
                    "similarity_to_parent": best_match.overall_similarity,
                    "auto_detected": True,
                },
            )

        return DocumentLineage(document_id=document_id, version=1)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent fingerprinting."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def _create_content_hash(self, content: str) -> str:
        """Create SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _create_structure_hash(self, content: str) -> str:
        """Create hash based on document structure."""
        # Extract structure features
        lines = content.split('\n')
        structure = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                structure.append('E')  # Empty
            elif stripped.startswith('#'):
                structure.append('H')  # Header
            elif stripped.startswith('- ') or stripped.startswith('* '):
                structure.append('L')  # List
            elif stripped.startswith('```'):
                structure.append('C')  # Code
            elif len(stripped) > 100:
                structure.append('P')  # Paragraph
            else:
                structure.append('S')  # Short

        structure_str = ''.join(structure)
        return hashlib.md5(structure_str.encode()).hexdigest()

    def _create_minhash(self, content: str) -> List[int]:
        """Create MinHash signature for Jaccard similarity estimation."""
        # Generate shingles (word n-grams)
        words = content.split()
        shingles = set()
        for i in range(len(words) - self.ngram_size + 1):
            shingle = ' '.join(words[i:i + self.ngram_size])
            shingles.add(hash(shingle) & 0xFFFFFFFF)

        if not shingles:
            return [0] * self.num_permutations

        # Compute MinHash signature
        signature = []
        for a, b in self._hash_funcs:
            min_hash = float('inf')
            for shingle in shingles:
                h = (a * shingle + b) % (2**31 - 1)
                min_hash = min(min_hash, h)
            signature.append(int(min_hash))

        return signature

    def _create_simhash(self, content: str) -> int:
        """Create SimHash for Hamming distance based similarity."""
        words = content.split()
        word_counts = Counter(words)

        # Initialize bit vector
        v = [0] * self.simhash_bits

        for word, count in word_counts.items():
            # Get hash of word
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)

            for i in range(self.simhash_bits):
                bit = (word_hash >> i) & 1
                if bit:
                    v[i] += count
                else:
                    v[i] -= count

        # Create final hash
        simhash = 0
        for i in range(self.simhash_bits):
            if v[i] > 0:
                simhash |= (1 << i)

        return simhash

    def _create_ngram_fingerprints(self, content: str) -> Set[int]:
        """Create set of n-gram fingerprints."""
        fingerprints = set()

        # Character n-grams
        for i in range(len(content) - self.ngram_size + 1):
            ngram = content[i:i + self.ngram_size]
            fingerprints.add(hash(ngram) & 0xFFFFFFFF)

        return fingerprints

    def _minhash_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Calculate Jaccard similarity estimate from MinHash signatures."""
        if not sig1 or not sig2:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def _simhash_similarity(self, hash1: int, hash2: int) -> float:
        """Calculate similarity from SimHash (based on Hamming distance)."""
        xor = hash1 ^ hash2
        hamming_distance = bin(xor).count('1')
        return 1 - (hamming_distance / self.simhash_bits)

    def _ngram_similarity(self, fps1: Set[int], fps2: Set[int]) -> float:
        """Calculate Jaccard similarity of n-gram fingerprints."""
        if not fps1 or not fps2:
            return 0.0
        intersection = len(fps1 & fps2)
        union = len(fps1 | fps2)
        return intersection / union if union > 0 else 0.0

    def _store_dna(self, dna: DocumentDNA) -> None:
        """Store DNA and update indices."""
        self._dna_store[dna.document_id] = dna

        # Update content hash index
        if dna.content_hash not in self._content_hash_index:
            self._content_hash_index[dna.content_hash] = []
        self._content_hash_index[dna.content_hash].append(dna.document_id)

        # Update LSH index (band the MinHash signature)
        band_size = 4
        num_bands = len(dna.minhash_signature) // band_size

        for band_idx in range(num_bands):
            start = band_idx * band_size
            band = tuple(dna.minhash_signature[start:start + band_size])
            band_hash = hash(band)

            if band_hash not in self._minhash_lsh:
                self._minhash_lsh[band_hash] = set()
            self._minhash_lsh[band_hash].add(dna.document_id)

    def _find_lsh_candidates(self, dna: DocumentDNA) -> Set[str]:
        """Find candidate similar documents using LSH."""
        candidates: Set[str] = set()

        band_size = 4
        num_bands = len(dna.minhash_signature) // band_size

        for band_idx in range(num_bands):
            start = band_idx * band_size
            band = tuple(dna.minhash_signature[start:start + band_size])
            band_hash = hash(band)

            if band_hash in self._minhash_lsh:
                candidates.update(self._minhash_lsh[band_hash])

        return candidates

    def get_dna(self, document_id: str) -> Optional[DocumentDNA]:
        """Retrieve stored DNA for a document."""
        return self._dna_store.get(document_id)

    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored document DNAs."""
        if not self._dna_store:
            return {"total_documents": 0}

        dnas = list(self._dna_store.values())
        return {
            "total_documents": len(dnas),
            "unique_content_hashes": len(self._content_hash_index),
            "potential_duplicates": sum(
                len(ids) - 1 for ids in self._content_hash_index.values() if len(ids) > 1
            ),
            "avg_word_count": np.mean([d.word_count for d in dnas]),
            "avg_unique_words": np.mean([d.unique_word_count for d in dnas]),
            "lsh_buckets": len(self._minhash_lsh),
        }

    async def list_all_dna(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List all document DNA records with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of DNA records as dictionaries
        """
        dnas = list(self._dna_store.values())

        # Sort by creation date (newest first)
        dnas.sort(key=lambda d: d.created_at, reverse=True)

        # Apply pagination
        paginated = dnas[offset:offset + limit]

        return [dna.to_dict() for dna in paginated]

    async def get_total_count(self) -> int:
        """Get total count of DNA records."""
        return len(self._dna_store)


# Singleton instance
_document_dna_service: Optional[DocumentDNAService] = None


def get_document_dna_service() -> DocumentDNAService:
    """Get or create the document DNA service singleton."""
    global _document_dna_service
    if _document_dna_service is None:
        _document_dna_service = DocumentDNAService()
    return _document_dna_service
