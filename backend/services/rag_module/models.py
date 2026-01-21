"""
AIDocumentIndexer - RAG Data Models
====================================

Data models for RAG service: Source citations, responses, and streaming.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.services.rag_verifier import VerificationResult
    from backend.services.corrective_rag import CRAGResult
    from backend.services.context_sufficiency import ContextSufficiencyResult


@dataclass
class Source:
    """Source citation for RAG response."""
    document_id: str
    document_name: str
    chunk_id: str
    collection: Optional[str] = None  # Collection/tag for document grouping
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    relevance_score: float = 0.0  # RRF score for ranking (may be tiny ~0.01-0.03)
    similarity_score: float = 0.0  # Original vector cosine similarity (0-1) for display
    snippet: str = ""
    full_content: str = ""  # Full chunk content for source viewer
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Context expansion (surrounding chunks for navigation)
    prev_chunk_snippet: Optional[str] = None
    next_chunk_snippet: Optional[str] = None
    chunk_index: Optional[int] = None


@dataclass
class RAGResponse:
    """Response from RAG query."""
    content: str
    sources: List[Source]
    query: str
    model: str
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Verification/confidence fields
    confidence_score: Optional[float] = None  # 0-1 confidence
    confidence_level: Optional[str] = None  # "high", "medium", "low"
    verification_result: Optional["VerificationResult"] = None
    # Suggested follow-up questions
    suggested_questions: List[str] = field(default_factory=list)
    # Confidence warning for UI display (empty string = no warning)
    confidence_warning: str = ""
    # CRAG result if query was refined
    crag_result: Optional["CRAGResult"] = None
    # Context sufficiency check result (Phase 2 enhancement)
    context_sufficiency: Optional["ContextSufficiencyResult"] = None


@dataclass
class StreamChunk:
    """A chunk in streaming response."""
    type: str  # "content", "source", "metadata", "done", "error"
    data: Any
