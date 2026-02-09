"""
Memory Management API — View, edit, delete, and export global user memories.
"""
import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.middleware.auth import AuthenticatedUser
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()


# ── Request/Response Models ──────────────────────────────────────────────

class MemoryResponse(BaseModel):
    """Single memory entry."""
    id: str
    content: str
    memory_type: str
    priority: str
    created_at: str
    last_accessed: str
    access_count: int
    source: Optional[str] = None
    entities: List[str] = []
    decay_score: float = 1.0
    metadata: dict = {}


class MemoryListResponse(BaseModel):
    """Paginated list of memories."""
    memories: List[MemoryResponse]
    total: int
    page: int
    page_size: int


class MemoryStatsResponse(BaseModel):
    """Memory statistics."""
    total_memories: int
    memories_by_type: dict
    memories_by_priority: dict
    avg_access_count: float
    oldest_memory: Optional[str] = None
    newest_memory: Optional[str] = None


class MemoryUpdateRequest(BaseModel):
    """Update a memory entry."""
    content: Optional[str] = None
    priority: Optional[str] = Field(None, pattern="^(critical|high|medium|low)$")


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("", response_model=MemoryListResponse)
async def list_memories(
    user: AuthenticatedUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    memory_type: Optional[str] = Query(None, pattern="^(fact|preference|context|procedure|entity|relationship)$"),
    priority: Optional[str] = Query(None, pattern="^(critical|high|medium|low)$"),
    search: Optional[str] = Query(None, max_length=200),
):
    """List all global memories for the current user."""
    from backend.services.mem0_memory import get_memory_service, MemoryType

    try:
        memory_service = await get_memory_service()
        mem_type = MemoryType(memory_type) if memory_type else None
        all_memories = await memory_service._store.get_user_memories(
            user_id=user.user_id,
            memory_type=mem_type,
            limit=1000,
        )

        # Apply priority filter
        if priority:
            all_memories = [m for m in all_memories if m.priority.value == priority]

        # Apply text search
        if search:
            search_lower = search.lower()
            all_memories = [m for m in all_memories if search_lower in m.content.lower()]

        # Sort by last_accessed descending
        all_memories.sort(key=lambda m: m.last_accessed, reverse=True)

        total = len(all_memories)
        start = (page - 1) * page_size
        page_memories = all_memories[start:start + page_size]

        return MemoryListResponse(
            memories=[
                MemoryResponse(
                    id=m.id,
                    content=m.content,
                    memory_type=m.memory_type.value,
                    priority=m.priority.value,
                    created_at=m.created_at.isoformat(),
                    last_accessed=m.last_accessed.isoformat(),
                    access_count=m.access_count,
                    source=m.source,
                    entities=m.entities,
                    decay_score=m.decay_score,
                    metadata=m.metadata,
                )
                for m in page_memories
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        logger.error("Failed to list memories", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    user: AuthenticatedUser,
):
    """Get memory usage statistics."""
    from backend.services.mem0_memory import get_memory_service

    try:
        memory_service = await get_memory_service()
        stats = await memory_service.get_stats(user.user_id)
        return MemoryStatsResponse(
            total_memories=stats.total_memories,
            memories_by_type=stats.memories_by_type,
            memories_by_priority=stats.memories_by_priority,
            avg_access_count=stats.avg_access_count,
            oldest_memory=stats.oldest_memory.isoformat() if stats.oldest_memory else None,
            newest_memory=stats.newest_memory.isoformat() if stats.newest_memory else None,
        )
    except Exception as e:
        logger.error("Failed to get memory stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@router.get("/export")
async def export_memories(
    user: AuthenticatedUser,
):
    """Export all memories as JSON download."""
    from backend.services.mem0_memory import get_memory_service

    try:
        memory_service = await get_memory_service()
        all_memories = await memory_service._store.get_user_memories(
            user_id=user.user_id, limit=10000
        )

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "user_id": user.user_id,
            "total_memories": len(all_memories),
            "memories": [m.to_dict() for m in all_memories],
        }

        json_bytes = json.dumps(export_data, indent=2, default=str).encode("utf-8")

        return StreamingResponse(
            iter([json_bytes]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=memories_{user.user_id[:8]}_{datetime.now().strftime('%Y%m%d')}.json"},
        )
    except Exception as e:
        logger.error("Failed to export memories", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to export memories: {str(e)}")


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    user: AuthenticatedUser,
):
    """Get a single memory by ID."""
    from backend.services.mem0_memory import get_memory_service

    try:
        memory_service = await get_memory_service()
        memory = await memory_service._store.get(memory_id)

        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        if memory.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to access this memory")

        return MemoryResponse(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type.value,
            priority=memory.priority.value,
            created_at=memory.created_at.isoformat(),
            last_accessed=memory.last_accessed.isoformat(),
            access_count=memory.access_count,
            source=memory.source,
            entities=memory.entities,
            decay_score=memory.decay_score,
            metadata=memory.metadata,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memory", error=str(e), memory_id=memory_id)
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest,
    user: AuthenticatedUser,
):
    """Update a memory's content or priority."""
    from backend.services.mem0_memory import get_memory_service, MemoryPriority

    try:
        memory_service = await get_memory_service()
        memory = await memory_service._store.get(memory_id)

        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        if memory.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to update this memory")

        if request.content is not None:
            memory.content = request.content
        if request.priority is not None:
            memory.priority = MemoryPriority(request.priority)

        await memory_service._store.update(memory)

        logger.info("Memory updated", memory_id=memory_id, user_id=user.user_id)

        return MemoryResponse(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type.value,
            priority=memory.priority.value,
            created_at=memory.created_at.isoformat(),
            last_accessed=memory.last_accessed.isoformat(),
            access_count=memory.access_count,
            source=memory.source,
            entities=memory.entities,
            decay_score=memory.decay_score,
            metadata=memory.metadata,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update memory", error=str(e), memory_id=memory_id)
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    user: AuthenticatedUser,
):
    """Delete a single memory."""
    from backend.services.mem0_memory import get_memory_service

    try:
        memory_service = await get_memory_service()
        deleted = await memory_service.delete(memory_id, user_id=user.user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found or already deleted")

        logger.info("Memory deleted", memory_id=memory_id, user_id=user.user_id)
        return {"deleted": True, "memory_id": memory_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete memory", error=str(e), memory_id=memory_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")


@router.delete("")
async def clear_all_memories(
    user: AuthenticatedUser,
    confirm: bool = Query(False, description="Must be true to actually clear"),
):
    """Clear all memories for the current user. Requires confirm=true."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to actually clear all memories. This is irreversible."
        )

    from backend.services.mem0_memory import get_memory_service

    try:
        memory_service = await get_memory_service()
        count = await memory_service.clear_user_memories(user.user_id)
        logger.info("All memories cleared", user_id=user.user_id, count=count)
        return {"cleared": True, "count": count}
    except Exception as e:
        logger.error("Failed to clear memories", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")
