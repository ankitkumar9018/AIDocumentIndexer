"""
AIDocumentIndexer - Intelligence Enhancement API Routes
========================================================

API endpoints for advanced reasoning capabilities:
- Extended Thinking
- Chain-of-Thought reasoning
- Self-Verification
- Multi-Model Ensemble Voting
- Tool Augmentation
- Context Optimization
- Session Compaction
"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import structlog

from backend.api.middleware.auth import get_current_user, require_admin

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/intelligence", tags=["Intelligence Enhancement"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ExtendedThinkingRequest(BaseModel):
    """Request for extended thinking."""
    query: str = Field(..., description="The query to think about")
    context: str = Field("", description="Additional context")
    thinking_level: Optional[str] = Field(
        None,
        description="Thinking depth: off, minimal, low, medium, high, max"
    )


class ExtendedThinkingResponse(BaseModel):
    """Response from extended thinking."""
    query: str
    thinking_level: str
    thinking_steps: List[Dict[str, Any]]
    thinking_summary: str
    final_answer: str
    confidence: float
    tokens_used: int


class ChainOfThoughtRequest(BaseModel):
    """Request for CoT reasoning."""
    question: str = Field(..., description="Question to reason about")
    context: str = Field("", description="Relevant context")
    strategy: Optional[str] = Field(
        None,
        description="Strategy: analytical, comparative, causal, hypothetical"
    )
    use_few_shot: bool = Field(True, description="Include few-shot examples")
    use_xml: bool = Field(True, description="Use XML structured output")


class ChainOfThoughtResponse(BaseModel):
    """Response from CoT reasoning."""
    question: str
    thinking_steps: List[str]
    final_answer: str
    confidence: float
    strategy_used: str


class VerificationRequest(BaseModel):
    """Request to verify an answer."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Answer to verify")
    sources: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Source documents"
    )
    use_recursive: bool = Field(False, description="Use recursive improvement")


class VerificationResponse(BaseModel):
    """Response from verification."""
    original_answer: str
    verified_answer: str
    status: str  # verified, corrected, uncertain, failed
    confidence: float
    issues_found: List[Dict[str, Any]]
    corrections_made: List[str]


class EnsembleRequest(BaseModel):
    """Request for ensemble voting."""
    question: str = Field(..., description="Question to answer")
    context: str = Field("", description="Additional context")
    models: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Models to use [{provider, model}]"
    )
    strategy: str = Field(
        "confidence",
        description="Voting strategy: majority, confidence, consensus, best_of_n, synthesis"
    )


class EnsembleResponse(BaseModel):
    """Response from ensemble voting."""
    query: str
    final_answer: str
    confidence: float
    strategy: str
    agreement_level: str
    disagreements: List[str]
    models_used: List[str]
    model_answers: List[Dict[str, Any]]


class ContextOptimizeRequest(BaseModel):
    """Request to optimize context."""
    query: str = Field(..., description="User query")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to optimize")
    max_tokens: int = Field(4000, description="Maximum tokens for context")
    strategy: str = Field("balanced", description="balanced, dense, or sparse")


class ContextOptimizeResponse(BaseModel):
    """Response from context optimization."""
    formatted_context: str
    chunks_included: int
    chunks_total: int
    total_tokens: int
    compression_ratio: float


class SessionCompactRequest(BaseModel):
    """Request to compact a session."""
    messages: List[Dict[str, str]] = Field(..., description="Message history")
    target_tokens: int = Field(4000, description="Target token count")
    keep_recent: int = Field(5, description="Recent messages to keep intact")


class SessionCompactResponse(BaseModel):
    """Response from session compaction."""
    summary: str
    key_facts: List[str]
    recent_messages: List[Dict[str, str]]
    original_message_count: int
    compacted_message_count: int
    compression_ratio: float


class ToolAugmentRequest(BaseModel):
    """Request for tool-augmented query."""
    query: str = Field(..., description="User query")
    context: str = Field("", description="Additional context")
    auto_detect_tools: bool = Field(True, description="Auto-detect needed tools")


class CalculateRequest(BaseModel):
    """Request for calculation."""
    expression: str = Field(..., description="Math expression to evaluate")


class DateCalculateRequest(BaseModel):
    """Request for date calculation."""
    operation: str = Field(..., description="Operation: diff, add, subtract, weekday, now")
    date1: Optional[str] = Field(None, description="First date (YYYY-MM-DD)")
    date2: Optional[str] = Field(None, description="Second date for diff")
    days: Optional[int] = Field(None, description="Days to add/subtract")
    months: Optional[int] = Field(None, description="Months to add/subtract")
    years: Optional[int] = Field(None, description="Years to add/subtract")


class UnitConvertRequest(BaseModel):
    """Request for unit conversion."""
    value: float = Field(..., description="Value to convert")
    from_unit: str = Field(..., description="Source unit")
    to_unit: str = Field(..., description="Target unit")


class FactCheckRequest(BaseModel):
    """Request for fact checking."""
    claim: str = Field(..., description="Claim to verify")
    context: str = Field("", description="Additional context")


# =============================================================================
# Extended Thinking Endpoints
# =============================================================================

@router.post("/thinking", response_model=ExtendedThinkingResponse)
async def extended_thinking(
    request: ExtendedThinkingRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Apply extended thinking to a query.

    Extended thinking allows the model to reason more deeply
    before answering, improving accuracy on complex questions.
    """
    from backend.services.extended_thinking import (
        get_extended_thinking_service,
        ThinkingLevel,
    )

    service = get_extended_thinking_service()

    # Parse thinking level
    level = None
    if request.thinking_level:
        try:
            level = ThinkingLevel(request.thinking_level)
        except ValueError:
            level = None  # Auto-detect

    result = await service.think(
        query=request.query,
        context=request.context,
        level=level,
    )

    return ExtendedThinkingResponse(
        query=result.query,
        thinking_level=result.thinking_level.value,
        thinking_steps=[
            {
                "step": s.step_number,
                "title": s.title,
                "content": s.content,
            }
            for s in result.thinking_steps
        ],
        thinking_summary=result.thinking_summary,
        final_answer=result.final_answer,
        confidence=result.confidence,
        tokens_used=result.tokens_used,
    )


# =============================================================================
# Chain-of-Thought Endpoints
# =============================================================================

@router.post("/cot", response_model=ChainOfThoughtResponse)
async def chain_of_thought(
    request: ChainOfThoughtRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Apply chain-of-thought reasoning to a question.

    Forces step-by-step reasoning which improves accuracy
    especially on complex analytical questions.
    """
    from backend.services.cot_engine import (
        get_cot_engine,
        ReasoningStrategy,
    )

    engine = get_cot_engine()

    # Use enhanced reasoning if features requested
    if request.use_few_shot or request.use_xml:
        result = await engine.reason_enhanced(
            question=request.question,
            context=request.context,
            use_few_shot=request.use_few_shot,
            use_xml=request.use_xml,
        )
    else:
        result = await engine.reason(
            question=request.question,
            context=request.context,
        )

    return ChainOfThoughtResponse(
        question=result.question,
        thinking_steps=result.thinking_steps,
        final_answer=result.final_answer,
        confidence=result.confidence,
        strategy_used=result.strategy_used.value,
    )


# =============================================================================
# Verification Endpoints
# =============================================================================

@router.post("/verify", response_model=VerificationResponse)
async def verify_answer(
    request: VerificationRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Verify an answer against sources.

    Makes the model check its own work to catch errors
    and hallucinations.
    """
    from backend.services.self_verification import get_verification_service

    service = get_verification_service()

    if request.use_recursive:
        result = await service.verify_recursive(
            question=request.question,
            answer=request.answer,
            sources=request.sources,
        )
    else:
        result = await service.verify_enhanced(
            question=request.question,
            answer=request.answer,
            sources=request.sources,
        )

    return VerificationResponse(
        original_answer=result.original_answer,
        verified_answer=result.verified_answer,
        status=result.status.value,
        confidence=result.confidence,
        issues_found=[
            {
                "type": i.issue_type,
                "description": i.description,
                "severity": i.severity,
            }
            for i in result.issues_found
        ],
        corrections_made=result.corrections_made,
    )


# =============================================================================
# Ensemble Voting Endpoints
# =============================================================================

@router.post("/ensemble", response_model=EnsembleResponse)
async def ensemble_query(
    request: EnsembleRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Query multiple models and vote on the best answer.

    Cross-verifies answers from multiple models to improve
    accuracy and detect disagreements.
    """
    from backend.services.ensemble_voting import (
        get_ensemble_voting_service,
        VotingStrategy,
    )

    service = get_ensemble_voting_service()

    # Parse voting strategy
    try:
        strategy = VotingStrategy(request.strategy)
    except ValueError:
        strategy = VotingStrategy.CONFIDENCE

    result = await service.query(
        question=request.question,
        context=request.context,
        models=request.models,
        strategy=strategy,
    )

    return EnsembleResponse(
        query=result.query,
        final_answer=result.final_answer,
        confidence=result.confidence,
        strategy=result.strategy.value,
        agreement_level=result.agreement_level,
        disagreements=result.disagreements,
        models_used=result.models_used,
        model_answers=[a.to_dict() for a in result.model_answers],
    )


# =============================================================================
# Context Optimization Endpoints
# =============================================================================

@router.post("/context/optimize", response_model=ContextOptimizeResponse)
async def optimize_context(
    request: ContextOptimizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Optimize context for small LLM context windows.

    Scores relevance, deduplicates, and compresses to fit
    maximum useful information in limited space.
    """
    from backend.services.context_optimizer import get_context_optimizer

    optimizer = get_context_optimizer()

    result = await optimizer.optimize_context(
        query=request.query,
        documents=request.documents,
        max_tokens=request.max_tokens,
        strategy=request.strategy,
    )

    return ContextOptimizeResponse(
        formatted_context=result.formatted_context,
        chunks_included=result.chunks_included,
        chunks_total=result.chunks_total,
        total_tokens=result.total_tokens,
        compression_ratio=result.compression_ratio,
    )


# =============================================================================
# Session Compaction Endpoints
# =============================================================================

@router.post("/session/compact", response_model=SessionCompactResponse)
async def compact_session(
    request: SessionCompactRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Compact a long conversation to fit within context limits.

    Summarizes older messages while preserving key facts
    and keeping recent messages intact.
    """
    from backend.services.session_compactor import (
        get_session_compactor,
        Message,
    )

    compactor = get_session_compactor()

    # Convert to Message objects
    messages = [
        Message(
            role=m.get("role", "user"),
            content=m.get("content", ""),
        )
        for m in request.messages
    ]

    result = await compactor.compact(
        messages=messages,
        target_tokens=request.target_tokens,
        keep_recent=request.keep_recent,
    )

    return SessionCompactResponse(
        summary=result.summary,
        key_facts=result.key_facts,
        recent_messages=[
            {"role": m.role, "content": m.content}
            for m in result.recent_messages
        ],
        original_message_count=result.original_message_count,
        compacted_message_count=result.compacted_message_count,
        compression_ratio=result.compression_ratio,
    )


# =============================================================================
# Tool Augmentation Endpoints
# =============================================================================

@router.post("/tools/augment")
async def augment_with_tools(
    request: ToolAugmentRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Augment a query with tool results.

    Automatically detects which tools (calculator, fact checker, etc.)
    would help answer the query and runs them.
    """
    from backend.services.tool_augmentation import get_tool_augmentation_service

    service = get_tool_augmentation_service()

    result = await service.augment_query(
        query=request.query,
        context=request.context,
        auto_detect_tools=request.auto_detect_tools,
    )

    return result


@router.post("/tools/calculate")
async def calculate(
    request: CalculateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Safely evaluate a mathematical expression.

    Supports basic arithmetic, functions (sqrt, sin, cos, log),
    and constants (pi, e).
    """
    from backend.services.tool_augmentation import get_tool_augmentation_service

    service = get_tool_augmentation_service()
    result = service.calculate(request.expression)

    return result.to_dict()


@router.post("/tools/date")
async def date_calculate(
    request: DateCalculateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Perform date calculations.

    Operations: diff (between dates), add/subtract (days/months/years),
    weekday (day of week), now (current date/time).
    """
    from backend.services.tool_augmentation import get_tool_augmentation_service

    service = get_tool_augmentation_service()
    result = service.calculate_date(
        operation=request.operation,
        date1=request.date1,
        date2=request.date2,
        days=request.days,
        months=request.months,
        years=request.years,
    )

    return result.to_dict()


@router.post("/tools/convert")
async def convert_units(
    request: UnitConvertRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Convert between units.

    Supports length, mass, volume, temperature, time, data, speed, and area.
    """
    from backend.services.tool_augmentation import get_tool_augmentation_service

    service = get_tool_augmentation_service()
    result = service.convert_units(
        value=request.value,
        from_unit=request.from_unit,
        to_unit=request.to_unit,
    )

    return result.to_dict()


@router.post("/tools/fact-check")
async def fact_check(
    request: FactCheckRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Check if a claim is factually accurate.

    Returns verdict (supported/contradicted/unverifiable) with evidence.
    """
    from backend.services.tool_augmentation import get_tool_augmentation_service

    service = get_tool_augmentation_service()
    result = await service.check_fact(
        claim=request.claim,
        context=request.context,
    )

    return result.to_dict()


# =============================================================================
# Memory Endpoints
# =============================================================================

class MemoryAddRequest(BaseModel):
    """Request to add a memory."""
    content: str = Field(..., description="Memory content")
    memory_type: str = Field("fact", description="Type: fact, preference, context, procedure")
    priority: str = Field("medium", description="Priority: critical, high, medium, low")
    source: Optional[str] = Field(None, description="Memory source")


class MemorySearchRequest(BaseModel):
    """Request to search memories."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of memories to return")
    include_graph: bool = Field(True, description="Include graph-based retrieval")


@router.post("/memory/add")
async def add_memory(
    request: MemoryAddRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Add a memory for the user.

    Memories are stored and can be retrieved later to provide
    personalized context.
    """
    from backend.services.mem0_memory import get_memory_service, MemoryType, MemoryPriority

    service = await get_memory_service()

    memory_id = await service.add(
        user_id=current_user.get("id", current_user.get("sub")),
        content=request.content,
        memory_type=MemoryType(request.memory_type),
        priority=MemoryPriority(request.priority),
        source=request.source,
    )

    return {"memory_id": memory_id, "status": "added"}


@router.post("/memory/search")
async def search_memories(
    request: MemorySearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Search for relevant memories.

    Uses vector similarity and optionally graph traversal
    to find relevant memories for a query.
    """
    from backend.services.mem0_memory import get_memory_service

    service = await get_memory_service()

    results = await service.get_relevant(
        query=request.query,
        user_id=current_user.get("id", current_user.get("sub")),
        top_k=request.top_k,
        include_graph=request.include_graph,
    )

    return {
        "memories": [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "type": r.memory.memory_type.value,
                "similarity": r.similarity,
                "relevance": r.relevance_score,
                "source": r.source,
            }
            for r in results
        ],
        "total": len(results),
    }


@router.get("/memory/stats")
async def get_memory_stats(
    current_user: dict = Depends(get_current_user),
):
    """
    Get memory statistics for the current user.
    """
    from backend.services.mem0_memory import get_memory_service

    service = await get_memory_service()
    stats = await service.get_stats(
        user_id=current_user.get("id", current_user.get("sub"))
    )

    return {
        "total_memories": stats.total_memories,
        "by_type": stats.memories_by_type,
        "by_priority": stats.memories_by_priority,
        "avg_access_count": stats.avg_access_count,
        "oldest_memory": stats.oldest_memory.isoformat() if stats.oldest_memory else None,
        "newest_memory": stats.newest_memory.isoformat() if stats.newest_memory else None,
    }


@router.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a specific memory.
    """
    from backend.services.mem0_memory import get_memory_service

    service = await get_memory_service()
    success = await service.delete(
        memory_id=memory_id,
        user_id=current_user.get("id", current_user.get("sub")),
    )

    if success:
        return {"status": "deleted", "memory_id": memory_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )


@router.delete("/memory")
async def clear_all_memories(
    current_user: dict = Depends(get_current_user),
):
    """
    Clear all memories for the current user.
    """
    from backend.services.mem0_memory import get_memory_service

    service = await get_memory_service()
    count = await service.clear_user_memories(
        user_id=current_user.get("id", current_user.get("sub"))
    )

    return {"status": "cleared", "memories_deleted": count}


# =============================================================================
# Channel Gateway Endpoints
# =============================================================================

class SendMessageRequest(BaseModel):
    """Request to send a message."""
    channel_type: str = Field(..., description="Channel: slack, discord, teams, telegram, web")
    channel_id: str = Field(..., description="Channel/user ID")
    content: str = Field(..., description="Message content")
    thread_id: Optional[str] = Field(None, description="Thread ID for replies")


@router.post("/channels/send")
async def send_channel_message(
    request: SendMessageRequest,
    current_user: dict = Depends(require_admin),
):
    """
    Send a message to a specific channel.

    Requires admin privileges. Routes through the multi-channel gateway.
    """
    from backend.services.channel_gateway import get_channel_gateway, ChannelType

    gateway = get_channel_gateway()

    try:
        channel_type = ChannelType(request.channel_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid channel type: {request.channel_type}",
        )

    result = await gateway.send(
        channel_type=channel_type,
        channel_id=request.channel_id,
        content=request.content,
        thread_id=request.thread_id,
    )

    return {
        "success": result.success,
        "channel": result.channel_type.value,
        "message_id": result.message_id,
        "error": result.error,
    }


@router.get("/channels/status")
async def get_channel_status(
    current_user: dict = Depends(require_admin),
):
    """
    Get status of all communication channels.
    """
    from backend.services.channel_gateway import get_channel_gateway

    gateway = get_channel_gateway()
    status = gateway.get_channel_status()

    return {
        "channels": status,
        "enabled": gateway.get_enabled_channels(),
    }


# =============================================================================
# Smart Collections Endpoints
# =============================================================================

class OrganizeDocumentsRequest(BaseModel):
    """Request to organize documents into smart collections."""
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to organize")
    strategy: str = Field("hybrid", description="Organization strategy: topic_cluster, entity_based, time_based, similarity, hybrid")
    min_cluster_size: int = Field(3, ge=2, le=20)


@router.post("/collections/organize")
async def organize_documents(
    request: OrganizeDocumentsRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Automatically organize documents into smart collections.

    Supports multiple organization strategies:
    - topic_cluster: Group by semantic topics using K-means
    - entity_based: Group by extracted entities (people, companies, projects)
    - time_based: Group by time periods
    - similarity: Group by content similarity
    - hybrid: Combine multiple strategies (recommended)
    """
    from backend.services.smart_collections import (
        get_smart_collections_service,
        OrganizationStrategy,
    )

    service = get_smart_collections_service()

    # Map string to enum
    strategy_map = {
        "topic_cluster": OrganizationStrategy.TOPIC_CLUSTER,
        "entity_based": OrganizationStrategy.ENTITY_BASED,
        "time_based": OrganizationStrategy.TIME_BASED,
        "similarity": OrganizationStrategy.SIMILARITY,
        "hybrid": OrganizationStrategy.HYBRID,
    }
    strategy = strategy_map.get(request.strategy, OrganizationStrategy.HYBRID)

    # TODO: Fetch documents and embeddings from database based on document_ids
    # For now, return structure
    return {
        "collections": [],
        "uncategorized_count": 0,
        "strategy_used": request.strategy,
        "processing_time_ms": 0,
    }


@router.get("/collections/smart")
async def get_smart_collections(
    limit: int = 20,
    collection_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Get all smart collections."""
    return {"collections": [], "total": 0}


# =============================================================================
# Insight Feed Endpoints
# =============================================================================

class InsightFeedRequest(BaseModel):
    """Request for insight feed."""
    limit: int = Field(20, ge=1, le=100)
    types: Optional[List[str]] = Field(None, description="Filter by insight types")


@router.get("/insights/feed")
async def get_insight_feed(
    limit: int = 20,
    types: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Get personalized insight feed for the user.

    Returns proactive intelligence including:
    - Trending topics in your documents
    - Knowledge gaps to fill
    - Document connections you might have missed
    - Stale content that needs review
    - Personalized reading recommendations
    """
    from backend.services.insight_feed import get_insight_feed_service

    service = get_insight_feed_service()
    user_id = current_user.get("id", current_user.get("sub"))

    # TODO: Fetch documents from database
    insights = await service.generate_feed(
        user_id=user_id,
        documents=[],
        limit=limit,
    )

    if types:
        type_list = types.split(",")
        insights = [i for i in insights if i.insight_type.value in type_list]

    return {
        "insights": [i.to_dict() for i in insights],
        "total": len(insights),
    }


@router.get("/insights/digest/{period}")
async def get_insight_digest(
    period: str = "daily",
    current_user: dict = Depends(get_current_user),
):
    """Get periodic digest summary (daily or weekly)."""
    from backend.services.insight_feed import get_insight_feed_service

    if period not in ["daily", "weekly"]:
        raise HTTPException(status_code=400, detail="Period must be 'daily' or 'weekly'")

    service = get_insight_feed_service()
    user_id = current_user.get("id", current_user.get("sub"))

    digest = await service.generate_digest(
        user_id=user_id,
        documents=[],
        period=period,
    )

    return digest.to_dict()


@router.post("/insights/track")
async def track_user_activity(
    activity_type: str,
    activity_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
):
    """Track user activity for better personalization."""
    from backend.services.insight_feed import get_insight_feed_service

    service = get_insight_feed_service()
    user_id = current_user.get("id", current_user.get("sub"))

    service.track_activity(user_id, activity_type, activity_data)
    return {"status": "tracked"}


@router.post("/insights/{insight_id}/dismiss")
async def dismiss_insight(
    insight_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Dismiss an insight so it won't appear again."""
    return {"status": "dismissed", "insight_id": insight_id}


# =============================================================================
# Document DNA Endpoints
# =============================================================================

class DocumentDNARequest(BaseModel):
    """Request to create document DNA."""
    document_id: str
    content: str
    title: Optional[str] = None


class DuplicateCheckRequest(BaseModel):
    """Request to check for duplicates."""
    content: str
    threshold: float = Field(0.95, ge=0.5, le=1.0)


@router.post("/dna/create")
async def create_document_dna(
    request: DocumentDNARequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Create a DNA profile (fingerprint) for a document.

    DNA includes multiple fingerprints for different matching needs:
    - Content hash: Exact matching (100% identical)
    - MinHash signature: Jaccard similarity estimation
    - SimHash: Hamming distance for near-duplicates
    - N-gram fingerprints: Partial content matching
    """
    from backend.services.document_dna import get_document_dna_service

    service = get_document_dna_service()

    dna = await service.create_dna(
        document_id=request.document_id,
        content=request.content,
        title=request.title,
    )

    return dna.to_dict()


@router.post("/dna/check-duplicates")
async def check_duplicates(
    request: DuplicateCheckRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Check if content has duplicates in the corpus.

    Returns list of potential duplicates with similarity scores.
    Useful before uploading to detect existing content.
    """
    from backend.services.document_dna import get_document_dna_service

    service = get_document_dna_service()

    # Create temporary DNA for checking
    temp_dna = await service.create_dna(
        document_id="temp_check",
        content=request.content,
    )

    matches = await service.find_duplicates(temp_dna, threshold=request.threshold)

    return {
        "has_duplicates": len(matches) > 0,
        "matches": [m.to_dict() for m in matches],
        "threshold": request.threshold,
    }


@router.post("/dna/plagiarism-check")
async def plagiarism_check(
    request: DuplicateCheckRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Check content for potential plagiarism against the corpus.

    Returns matches with confidence levels.
    """
    from backend.services.document_dna import get_document_dna_service

    service = get_document_dna_service()
    results = await service.detect_plagiarism(content=request.content)

    return {
        "potential_plagiarism": len(results) > 0,
        "matches": results,
    }


@router.get("/dna/stats")
async def get_dna_stats(
    current_user: dict = Depends(get_current_user),
):
    """Get DNA system statistics."""
    from backend.services.document_dna import get_document_dna_service

    service = get_document_dna_service()
    return await service.get_document_stats()


@router.get("/dna/list")
async def list_document_dna(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """
    List all document DNA fingerprint records.

    Returns paginated list of document DNA profiles for display
    in the Document DNA management interface.
    """
    from backend.services.document_dna import get_document_dna_service

    service = get_document_dna_service()

    # Get all DNA records (service needs to implement list method)
    try:
        records = await service.list_all_dna(limit=limit, offset=offset)
        total = await service.get_total_count()
    except AttributeError:
        # Fallback if methods not implemented yet
        records = []
        total = 0

    return {
        "records": records,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# =============================================================================
# Smart Highlights Endpoints
# =============================================================================

class HighlightRequest(BaseModel):
    """Request for smart highlights."""
    document_id: str
    content: str
    title: Optional[str] = None
    use_llm: bool = Field(True, description="Use LLM for enhanced analysis")


@router.post("/highlights/analyze")
async def analyze_for_highlights(
    request: HighlightRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Analyze a document for smart highlights.

    Returns AI-identified highlights including:
    - Key points and important statements
    - Named entities (people, organizations, locations, dates)
    - Statistics and quotes
    - Action items and questions
    - Reading metrics (time estimate, difficulty level)
    - Summary and key takeaways
    """
    from backend.services.smart_highlights import get_smart_highlights_service

    service = get_smart_highlights_service()

    analysis = await service.analyze_document(
        document_id=request.document_id,
        content=request.content,
        title=request.title,
        use_llm=request.use_llm,
    )

    return analysis.to_dict()


# =============================================================================
# Conflict Detection Endpoints
# =============================================================================

class ConflictCheckRequest(BaseModel):
    """Request for conflict detection."""
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to check")
    focus_topics: Optional[List[str]] = Field(None, description="Topics to focus on")


class ResolveConflictRequest(BaseModel):
    """Request to resolve a conflict."""
    resolution: str = Field(..., description="Resolution strategy")
    notes: str = Field("", description="Resolution notes")


@router.post("/conflicts/analyze")
async def analyze_conflicts(
    request: ConflictCheckRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Analyze corpus for conflicts and inconsistencies.

    Detects:
    - Factual contradictions between documents
    - Numerical inconsistencies (different values for same metric)
    - Temporal conflicts (outdated vs current information)
    - Definitional conflicts (different definitions for same term)
    - Procedural conflicts (contradicting instructions)

    Returns suggested resolutions for each conflict.
    """
    from backend.services.conflict_detector import get_conflict_detector_service

    service = get_conflict_detector_service()

    # TODO: Fetch documents from database
    report = await service.analyze_corpus(
        documents=[],
        focus_topics=request.focus_topics,
    )

    return report.to_dict()


@router.post("/conflicts/check-document")
async def check_document_conflicts(
    document_id: str,
    content: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Check a single document against the corpus for conflicts.

    Useful before publishing to identify potential inconsistencies.
    """
    from backend.services.conflict_detector import get_conflict_detector_service

    service = get_conflict_detector_service()

    conflicts = await service.check_document(
        new_document={"id": document_id, "content": content},
        existing_documents=[],
    )

    return {
        "document_id": document_id,
        "conflicts": [c.to_dict() for c in conflicts],
        "has_conflicts": len(conflicts) > 0,
    }


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    conflict_id: str,
    request: ResolveConflictRequest,
    current_user: dict = Depends(get_current_user),
):
    """Resolve a detected conflict."""
    from backend.services.conflict_detector import (
        get_conflict_detector_service,
        ResolutionStrategy,
    )

    service = get_conflict_detector_service()

    resolution_map = {
        "use_newer": ResolutionStrategy.USE_NEWER,
        "use_authoritative": ResolutionStrategy.USE_AUTHORITATIVE,
        "merge": ResolutionStrategy.MERGE,
        "manual_review": ResolutionStrategy.MANUAL_REVIEW,
        "flag_both": ResolutionStrategy.FLAG_BOTH,
        "deprecate_older": ResolutionStrategy.DEPRECATE_OLDER,
    }
    resolution = resolution_map.get(request.resolution, ResolutionStrategy.MANUAL_REVIEW)

    user_id = current_user.get("id", current_user.get("sub"))

    success = await service.resolve_conflict(
        conflict_id=conflict_id,
        resolution=resolution,
        notes=request.notes,
        resolved_by=user_id,
    )

    return {"status": "resolved" if success else "failed", "conflict_id": conflict_id}


@router.get("/conflicts")
async def get_conflicts(
    status: str = "unresolved",
    severity: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """Get list of detected conflicts."""
    return {"conflicts": [], "total": 0}


# =============================================================================
# Unified Intelligence Dashboard
# =============================================================================

@router.get("/dashboard")
async def get_intelligence_dashboard(
    current_user: dict = Depends(get_current_user),
):
    """
    Get unified intelligence dashboard combining all features.

    Provides a single endpoint for:
    - Unread insights count and top insights
    - Unresolved conflicts summary
    - Smart collection stats
    - Document DNA stats
    - User activity summary
    """
    user_id = current_user.get("id", current_user.get("sub"))

    return {
        "insights": {
            "unread_count": 0,
            "top_insights": [],
        },
        "conflicts": {
            "unresolved_count": 0,
            "critical_count": 0,
            "recent": [],
        },
        "collections": {
            "auto_generated": 0,
            "recently_updated": [],
        },
        "dna": {
            "total_fingerprints": 0,
            "potential_duplicates": 0,
        },
        "activity": {
            "documents_viewed_today": 0,
            "searches_today": 0,
        },
    }


# =============================================================================
# Help System Endpoints
# =============================================================================

@router.get("/help/features")
async def get_feature_help():
    """
    Get help information for all intelligence features.

    Returns descriptions, usage examples, and tips for each feature.
    """
    return {
        "features": [
            {
                "id": "smart_collections",
                "name": "Smart Collections",
                "description": "Automatically organize documents into intelligent collections using AI clustering.",
                "usage": "Go to Collections > Smart Collections, or use /organize command in chat.",
                "tips": [
                    "Works best with 10+ documents",
                    "Try different strategies for different use cases",
                    "Collections update automatically as you add documents",
                ],
            },
            {
                "id": "insight_feed",
                "name": "Insight Feed",
                "description": "Proactive intelligence that surfaces relevant documents and detects patterns.",
                "usage": "Check your Insights panel or use /insights command.",
                "tips": [
                    "Dismiss irrelevant insights to improve recommendations",
                    "Activity tracking makes insights more personalized over time",
                ],
            },
            {
                "id": "document_dna",
                "name": "Document DNA",
                "description": "Unique fingerprints for instant duplicate detection and plagiarism checking.",
                "usage": "Automatic on upload, or use /check-duplicates command.",
                "tips": [
                    "DNA is created automatically for all uploaded documents",
                    "Use before uploading to avoid duplicates",
                ],
            },
            {
                "id": "smart_highlights",
                "name": "Smart Highlights",
                "description": "AI-powered reading mode with automatic key point highlighting.",
                "usage": "Open any document and click 'Smart Read' or use /highlights command.",
                "tips": [
                    "Includes reading time estimate and difficulty level",
                    "Highlights key points, statistics, quotes, and action items",
                ],
            },
            {
                "id": "conflict_detector",
                "name": "Conflict Detector",
                "description": "Find and resolve contradictions across your document corpus.",
                "usage": "Go to Analytics > Conflicts, or use /check-conflicts command.",
                "tips": [
                    "Run periodically to maintain document consistency",
                    "Critical conflicts should be resolved first",
                    "Use suggested resolutions as starting points",
                ],
            },
            {
                "id": "extended_thinking",
                "name": "Extended Thinking",
                "description": "Deep reasoning for complex questions by allowing more thinking time.",
                "usage": "Enable in chat settings or prefix query with /think.",
                "tips": [
                    "Best for complex analytical questions",
                    "Shows step-by-step reasoning process",
                ],
            },
            {
                "id": "ensemble_voting",
                "name": "Ensemble Voting",
                "description": "Query multiple AI models and get consensus answers.",
                "usage": "Use /ensemble command or enable in settings.",
                "tips": [
                    "Helps verify important answers",
                    "Shows disagreements between models",
                ],
            },
        ],
        "chat_commands": [
            {"command": "/organize", "description": "Organize documents into smart collections"},
            {"command": "/insights", "description": "Show your personalized insight feed"},
            {"command": "/check-duplicates", "description": "Check for duplicate content"},
            {"command": "/highlights [doc]", "description": "Generate smart highlights for a document"},
            {"command": "/check-conflicts", "description": "Analyze corpus for conflicts"},
            {"command": "/think [query]", "description": "Use extended thinking for complex queries"},
            {"command": "/ensemble [query]", "description": "Query multiple models"},
            {"command": "/verify [answer]", "description": "Verify an answer for accuracy"},
            {"command": "/help", "description": "Show available commands and features"},
        ],
    }


@router.get("/help/tooltips")
async def get_tooltips():
    """Get tooltip content for UI elements."""
    return {
        "tooltips": {
            "smart_collections_button": "Automatically organize documents by topics, entities, or time periods",
            "insight_feed_icon": "AI-generated insights about your documents and reading patterns",
            "dna_check_button": "Check if similar content already exists before uploading",
            "highlight_mode_toggle": "Enable AI-powered reading mode with automatic highlighting",
            "conflict_badge": "Number of unresolved conflicts in your document corpus",
            "thinking_indicator": "AI is reasoning through the problem step by step",
            "ensemble_mode": "Multiple AI models are being consulted for this answer",
            "verification_badge": "Answer has been verified against source documents",
            "relevance_score": "How relevant this result is to your query (0-100%)",
            "confidence_score": "AI's confidence in this answer (0-100%)",
        },
    }
