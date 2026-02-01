"""
AIDocumentIndexer - Speculative RAG
====================================

Implements Speculative RAG (Google Research, ICLR 2025): generates draft
responses from document subsets in parallel using a smaller drafter model,
then selects the best one via the main verifier model.

Expected impact: 15-50% latency reduction, 5-13% accuracy improvement.

Reference: https://arxiv.org/abs/2407.08223
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DraftResponse:
    """A draft answer generated from a document subset."""
    answer: str
    source_indices: List[int]
    confidence: float = 0.0


@dataclass
class SpeculativeResult:
    """Result of speculative RAG generation."""
    answer: str
    selected_draft: int
    total_drafts: int
    confidence: float
    latency_saved: bool


async def _generate_draft(
    llm: Any,
    query: str,
    context_subset: str,
    draft_index: int,
) -> DraftResponse:
    """Generate a single draft response from a context subset."""
    from langchain.schema import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "You are a precise RAG assistant. Answer the question using ONLY the provided context. "
            "Be concise and factual. If the context doesn't contain the answer, say so."
        )),
        HumanMessage(content=f"Context:\n{context_subset}\n\nQuestion: {query}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        return DraftResponse(
            answer=content,
            source_indices=[draft_index],
            confidence=0.5,  # Placeholder, verifier will score
        )
    except Exception as e:
        logger.warning("Draft generation failed", draft_index=draft_index, error=str(e))
        return DraftResponse(answer="", source_indices=[draft_index], confidence=0.0)


async def _verify_and_select(
    verifier_llm: Any,
    query: str,
    drafts: List[DraftResponse],
    full_context: str,
) -> int:
    """Use the verifier model to select the best draft."""
    from langchain.schema import HumanMessage, SystemMessage

    if not drafts or all(not d.answer for d in drafts):
        return 0

    # Build comparison prompt
    draft_text = "\n\n".join(
        f"[Draft {i+1}]:\n{d.answer}" for i, d in enumerate(drafts) if d.answer
    )

    messages = [
        SystemMessage(content=(
            "You are a response quality evaluator. Given multiple draft answers to a question, "
            "select the BEST one based on accuracy, completeness, and relevance to the context. "
            "Reply with ONLY the draft number (e.g., '1' or '2')."
        )),
        HumanMessage(content=(
            f"Question: {query}\n\n"
            f"Reference Context:\n{full_context[:4000]}\n\n"
            f"Draft Answers:\n{draft_text}\n\n"
            f"Which draft number is the best answer? Reply with just the number."
        )),
    ]

    try:
        response = await verifier_llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        # Parse the number from response
        for char in content.strip():
            if char.isdigit():
                idx = int(char) - 1
                if 0 <= idx < len(drafts):
                    return idx
        return 0
    except Exception as e:
        logger.warning("Verifier selection failed, using first draft", error=str(e))
        return 0


def _split_context_into_subsets(
    documents: List[Any],
    num_subsets: int = 3,
) -> List[str]:
    """Split retrieved documents into subsets for parallel drafting."""
    if not documents:
        return [""]

    # Convert to text list
    doc_texts = []
    for doc in documents:
        if hasattr(doc, 'page_content'):
            doc_texts.append(doc.page_content)
        elif isinstance(doc, dict):
            doc_texts.append(doc.get('content', doc.get('text', str(doc))))
        else:
            doc_texts.append(str(doc))

    if len(doc_texts) <= num_subsets:
        return ["\n\n".join(doc_texts)]

    # Distribute documents across subsets (round-robin)
    subsets = [[] for _ in range(num_subsets)]
    for i, text in enumerate(doc_texts):
        subsets[i % num_subsets].append(text)

    return ["\n\n".join(subset) for subset in subsets if subset]


async def speculative_rag_generate(
    query: str,
    documents: List[Any],
    drafter_llm: Any,
    verifier_llm: Any,
    full_context: str,
    num_drafts: int = 3,
) -> Optional[SpeculativeResult]:
    """
    Generate response using Speculative RAG pattern.

    1. Split documents into subsets
    2. Generate draft answers in parallel using drafter model
    3. Verify and select best draft using verifier model

    Args:
        query: User question
        documents: Retrieved documents
        drafter_llm: Smaller/faster model for draft generation
        verifier_llm: Main model for verification
        full_context: Full formatted context string
        num_drafts: Number of parallel drafts to generate

    Returns:
        SpeculativeResult or None if speculative RAG fails
    """
    if not documents or len(documents) < 2:
        return None

    # Split context into subsets
    subsets = _split_context_into_subsets(documents, num_subsets=num_drafts)

    if len(subsets) < 2:
        return None

    # Generate drafts in parallel
    draft_tasks = [
        _generate_draft(drafter_llm, query, subset, i)
        for i, subset in enumerate(subsets)
    ]

    drafts = await asyncio.gather(*draft_tasks, return_exceptions=True)

    # Filter out exceptions
    valid_drafts = [d for d in drafts if isinstance(d, DraftResponse) and d.answer]

    if not valid_drafts:
        return None

    if len(valid_drafts) == 1:
        return SpeculativeResult(
            answer=valid_drafts[0].answer,
            selected_draft=0,
            total_drafts=1,
            confidence=valid_drafts[0].confidence,
            latency_saved=True,
        )

    # Verify and select best draft
    selected_idx = await _verify_and_select(verifier_llm, query, valid_drafts, full_context)

    selected = valid_drafts[selected_idx]

    logger.info(
        "Speculative RAG completed",
        total_drafts=len(valid_drafts),
        selected_draft=selected_idx,
        answer_length=len(selected.answer),
    )

    return SpeculativeResult(
        answer=selected.answer,
        selected_draft=selected_idx,
        total_drafts=len(valid_drafts),
        confidence=0.8,  # Verified drafts get higher confidence
        latency_saved=True,
    )
