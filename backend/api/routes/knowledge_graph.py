"""
AIDocumentIndexer - Knowledge Graph API Routes
===============================================

API endpoints for knowledge graph visualization and exploration.
Provides access to entities, relationships, and graph statistics.
"""

import uuid
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.db.database import get_async_session
from backend.db.models import (
    Entity, EntityMention, EntityRelation,
    EntityType, RelationType, Document,
)
from backend.api.middleware.auth import get_current_user
from backend.services.knowledge_graph import get_knowledge_graph_service

logger = structlog.get_logger(__name__)

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================

class EntityResponse(BaseModel):
    """Entity response model."""
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    aliases: List[str] = []
    mention_count: int = 0
    created_at: str

    class Config:
        from_attributes = True


class RelationResponse(BaseModel):
    """Relationship response model."""
    id: str
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    relation_type: str
    relation_label: Optional[str] = None
    weight: float = 1.0

    class Config:
        from_attributes = True


class GraphStatsResponse(BaseModel):
    """Knowledge graph statistics."""
    total_entities: int = 0
    total_relations: int = 0
    total_mentions: int = 0
    entity_type_distribution: dict = Field(default_factory=dict)
    relation_type_distribution: dict = Field(default_factory=dict)
    top_entities: List[EntityResponse] = []
    documents_with_entities: int = 0


class GraphDataResponse(BaseModel):
    """Graph data for visualization."""
    nodes: List[dict] = []
    edges: List[dict] = []
    stats: GraphStatsResponse


class EntityNeighborhoodResponse(BaseModel):
    """Entity neighborhood with connected entities and relations."""
    entity: EntityResponse
    neighbors: List[EntityResponse] = []
    relations: List[RelationResponse] = []


class EntitySearchResponse(BaseModel):
    """Entity search results."""
    entities: List[EntityResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class DocumentEntitiesResponse(BaseModel):
    """Entities found in a document."""
    document_id: str
    document_name: str
    entities: List[EntityResponse]
    relations: List[RelationResponse]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """
    Get knowledge graph statistics.

    Returns summary statistics about the knowledge graph including
    entity counts, relationship counts, and distributions.
    """
    try:
        # Get basic counts
        entity_count = await db.scalar(select(func.count(Entity.id))) or 0
        relation_count = await db.scalar(select(func.count(EntityRelation.id))) or 0
        mention_count = await db.scalar(select(func.count(EntityMention.id))) or 0

        # Entity type distribution
        result = await db.execute(
            select(Entity.entity_type, func.count(Entity.id))
            .group_by(Entity.entity_type)
        )
        entity_type_dist = {str(row[0].value): row[1] for row in result.all()}

        # Relation type distribution
        result = await db.execute(
            select(EntityRelation.relation_type, func.count(EntityRelation.id))
            .group_by(EntityRelation.relation_type)
        )
        relation_type_dist = {str(row[0].value): row[1] for row in result.all()}

        # Top entities by mention count (subquery approach)
        mention_counts = (
            select(
                EntityMention.entity_id,
                func.count(EntityMention.id).label("mention_count")
            )
            .group_by(EntityMention.entity_id)
            .subquery()
        )

        result = await db.execute(
            select(Entity, mention_counts.c.mention_count)
            .outerjoin(mention_counts, Entity.id == mention_counts.c.entity_id)
            .order_by(desc(mention_counts.c.mention_count))
            .limit(10)
        )

        top_entities = []
        for row in result.all():
            entity = row[0]
            count = row[1] or 0
            top_entities.append(EntityResponse(
                id=str(entity.id),
                name=entity.name,
                entity_type=entity.entity_type.value,
                description=entity.description,
                aliases=entity.aliases or [],
                mention_count=count,
                created_at=entity.created_at.isoformat() if entity.created_at else "",
            ))

        # Documents with entities
        docs_with_entities = await db.scalar(
            select(func.count(func.distinct(EntityMention.document_id)))
        ) or 0

        return GraphStatsResponse(
            total_entities=entity_count,
            total_relations=relation_count,
            total_mentions=mention_count,
            entity_type_distribution=entity_type_dist,
            relation_type_distribution=relation_type_dist,
            top_entities=top_entities,
            documents_with_entities=docs_with_entities,
        )

    except Exception as e:
        logger.error("Failed to get graph stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph statistics: {str(e)}"
        )


@router.get("/data", response_model=GraphDataResponse)
async def get_graph_data(
    limit: int = Query(100, ge=1, le=500, description="Maximum nodes to return"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """
    Get graph data for visualization.

    Returns nodes (entities) and edges (relationships) formatted
    for use with visualization libraries like vis.js or d3.js.
    """
    try:
        # Build entity query
        entity_query = select(Entity)
        if entity_type:
            try:
                et = EntityType(entity_type)
                entity_query = entity_query.where(Entity.entity_type == et)
            except ValueError:
                pass

        entity_query = entity_query.limit(limit)

        result = await db.execute(entity_query)
        entities = list(result.scalars().all())
        entity_ids = [e.id for e in entities]

        # Get relations between these entities
        relations = []
        if entity_ids:
            result = await db.execute(
                select(EntityRelation)
                .options(
                    selectinload(EntityRelation.source_entity),
                    selectinload(EntityRelation.target_entity),
                )
                .where(
                    and_(
                        EntityRelation.source_entity_id.in_(entity_ids),
                        EntityRelation.target_entity_id.in_(entity_ids),
                    )
                )
            )
            relations = list(result.scalars().all())

        # Format nodes for visualization
        nodes = []
        for entity in entities:
            nodes.append({
                "id": str(entity.id),
                "label": entity.name,
                "type": entity.entity_type.value,
                "description": entity.description,
                "group": entity.entity_type.value,
            })

        # Format edges for visualization
        edges = []
        for rel in relations:
            edges.append({
                "id": str(rel.id),
                "from": str(rel.source_entity_id),
                "to": str(rel.target_entity_id),
                "label": rel.relation_label or rel.relation_type.value,
                "type": rel.relation_type.value,
                "weight": rel.weight,
            })

        # Calculate inline stats
        entity_count = len(entities)
        relation_count = len(relations)

        # Entity type distribution from fetched entities
        entity_type_dist = {}
        for entity in entities:
            et_value = entity.entity_type.value
            entity_type_dist[et_value] = entity_type_dist.get(et_value, 0) + 1

        # Relation type distribution from fetched relations
        relation_type_dist = {}
        for rel in relations:
            rt_value = rel.relation_type.value
            relation_type_dist[rt_value] = relation_type_dist.get(rt_value, 0) + 1

        stats = GraphStatsResponse(
            total_entities=entity_count,
            total_relations=relation_count,
            total_mentions=0,  # Not needed for visualization
            entity_type_distribution=entity_type_dist,
            relation_type_distribution=relation_type_dist,
            top_entities=[],  # Not needed for visualization
            documents_with_entities=0,  # Not needed for visualization
        )

        return GraphDataResponse(
            nodes=nodes,
            edges=edges,
            stats=stats,
        )

    except Exception as e:
        logger.error("Failed to get graph data", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get graph data: {str(e)}"
        )


@router.get("/entities", response_model=EntitySearchResponse)
async def search_entities(
    query: Optional[str] = Query(None, description="Search query"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """
    Search entities in the knowledge graph.

    Supports filtering by name/alias search and entity type.
    """
    try:
        # Build base query
        base_query = select(Entity)
        count_query = select(func.count(Entity.id))

        conditions = []

        # Name search
        if query:
            search_pattern = f"%{query}%"
            conditions.append(
                or_(
                    Entity.name.ilike(search_pattern),
                    Entity.description.ilike(search_pattern),
                )
            )

        # Entity type filter
        if entity_type:
            try:
                et = EntityType(entity_type)
                conditions.append(Entity.entity_type == et)
            except ValueError:
                pass

        if conditions:
            base_query = base_query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total = await db.scalar(count_query) or 0

        # Apply pagination
        offset = (page - 1) * page_size
        base_query = base_query.offset(offset).limit(page_size)

        result = await db.execute(base_query)
        entities = list(result.scalars().all())

        # Get mention counts
        entity_ids = [e.id for e in entities]
        mention_counts = {}

        if entity_ids:
            result = await db.execute(
                select(
                    EntityMention.entity_id,
                    func.count(EntityMention.id)
                )
                .where(EntityMention.entity_id.in_(entity_ids))
                .group_by(EntityMention.entity_id)
            )
            mention_counts = {row[0]: row[1] for row in result.all()}

        entity_responses = [
            EntityResponse(
                id=str(e.id),
                name=e.name,
                entity_type=e.entity_type.value,
                description=e.description,
                aliases=e.aliases or [],
                mention_count=mention_counts.get(e.id, 0),
                created_at=e.created_at.isoformat() if e.created_at else "",
            )
            for e in entities
        ]

        return EntitySearchResponse(
            entities=entity_responses,
            total=total,
            page=page,
            page_size=page_size,
            has_more=(offset + len(entities)) < total,
        )

    except Exception as e:
        logger.error("Failed to search entities", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search entities: {str(e)}"
        )


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """Get a specific entity by ID."""
    try:
        entity_uuid = uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid entity ID format"
        )

    result = await db.execute(
        select(Entity).where(Entity.id == entity_uuid)
    )
    entity = result.scalar_one_or_none()

    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entity not found"
        )

    # Get mention count
    mention_count = await db.scalar(
        select(func.count(EntityMention.id))
        .where(EntityMention.entity_id == entity_uuid)
    ) or 0

    return EntityResponse(
        id=str(entity.id),
        name=entity.name,
        entity_type=entity.entity_type.value,
        description=entity.description,
        aliases=entity.aliases or [],
        mention_count=mention_count,
        created_at=entity.created_at.isoformat() if entity.created_at else "",
    )


@router.get("/entities/{entity_id}/neighborhood", response_model=EntityNeighborhoodResponse)
async def get_entity_neighborhood(
    entity_id: str,
    max_hops: int = Query(2, ge=1, le=5, description="Maximum graph hops"),
    max_neighbors: int = Query(20, ge=1, le=50, description="Maximum neighbors"),
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """
    Get an entity's neighborhood in the graph.

    Returns the entity and all entities connected to it within
    the specified number of hops.
    """
    try:
        entity_uuid = uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid entity ID format"
        )

    # Get the entity
    result = await db.execute(
        select(Entity).where(Entity.id == entity_uuid)
    )
    entity = result.scalar_one_or_none()

    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entity not found"
        )

    # Use knowledge graph service for neighborhood traversal
    kg_service = await get_knowledge_graph_service(db)
    neighbors, relations = await kg_service.get_entity_neighborhood(
        entity_id=entity_uuid,
        max_hops=max_hops,
        max_neighbors=max_neighbors,
    )

    # Get mention counts
    entity_ids = [n.id for n in neighbors]
    mention_counts = {}

    if entity_ids:
        result = await db.execute(
            select(
                EntityMention.entity_id,
                func.count(EntityMention.id)
            )
            .where(EntityMention.entity_id.in_(entity_ids))
            .group_by(EntityMention.entity_id)
        )
        mention_counts = {row[0]: row[1] for row in result.all()}

    # Format response
    entity_response = EntityResponse(
        id=str(entity.id),
        name=entity.name,
        entity_type=entity.entity_type.value,
        description=entity.description,
        aliases=entity.aliases or [],
        mention_count=mention_counts.get(entity.id, 0),
        created_at=entity.created_at.isoformat() if entity.created_at else "",
    )

    neighbor_responses = [
        EntityResponse(
            id=str(n.id),
            name=n.name,
            entity_type=n.entity_type.value,
            description=n.description,
            aliases=n.aliases or [],
            mention_count=mention_counts.get(n.id, 0),
            created_at=n.created_at.isoformat() if n.created_at else "",
        )
        for n in neighbors if n.id != entity_uuid
    ]

    relation_responses = [
        RelationResponse(
            id=str(r.id),
            source_entity_id=str(r.source_entity_id),
            source_entity_name=r.source_entity.name if r.source_entity else "",
            target_entity_id=str(r.target_entity_id),
            target_entity_name=r.target_entity.name if r.target_entity else "",
            relation_type=r.relation_type.value,
            relation_label=r.relation_label,
            weight=r.weight,
        )
        for r in relations
    ]

    return EntityNeighborhoodResponse(
        entity=entity_response,
        neighbors=neighbor_responses,
        relations=relation_responses,
    )


@router.get("/documents/{document_id}/entities", response_model=DocumentEntitiesResponse)
async def get_document_entities(
    document_id: str,
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """
    Get all entities mentioned in a specific document.

    Returns entities and their relationships found in the document.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )

    # Get document
    result = await db.execute(
        select(Document).where(Document.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Get entities mentioned in this document
    result = await db.execute(
        select(Entity)
        .join(EntityMention, Entity.id == EntityMention.entity_id)
        .where(EntityMention.document_id == doc_uuid)
        .distinct()
    )
    entities = list(result.scalars().all())
    entity_ids = [e.id for e in entities]

    # Get relations between these entities
    relations = []
    if len(entity_ids) > 1:
        result = await db.execute(
            select(EntityRelation)
            .options(
                selectinload(EntityRelation.source_entity),
                selectinload(EntityRelation.target_entity),
            )
            .where(
                and_(
                    EntityRelation.source_entity_id.in_(entity_ids),
                    EntityRelation.target_entity_id.in_(entity_ids),
                )
            )
        )
        relations = list(result.scalars().all())

    # Get mention counts
    mention_counts = {}
    if entity_ids:
        result = await db.execute(
            select(
                EntityMention.entity_id,
                func.count(EntityMention.id)
            )
            .where(EntityMention.entity_id.in_(entity_ids))
            .group_by(EntityMention.entity_id)
        )
        mention_counts = {row[0]: row[1] for row in result.all()}

    entity_responses = [
        EntityResponse(
            id=str(e.id),
            name=e.name,
            entity_type=e.entity_type.value,
            description=e.description,
            aliases=e.aliases or [],
            mention_count=mention_counts.get(e.id, 0),
            created_at=e.created_at.isoformat() if e.created_at else "",
        )
        for e in entities
    ]

    relation_responses = [
        RelationResponse(
            id=str(r.id),
            source_entity_id=str(r.source_entity_id),
            source_entity_name=r.source_entity.name if r.source_entity else "",
            target_entity_id=str(r.target_entity_id),
            target_entity_name=r.target_entity.name if r.target_entity else "",
            relation_type=r.relation_type.value,
            relation_label=r.relation_label,
            weight=r.weight,
        )
        for r in relations
    ]

    return DocumentEntitiesResponse(
        document_id=document_id,
        document_name=document.name,
        entities=entity_responses,
        relations=relation_responses,
    )


@router.get("/types")
async def get_entity_types(
    _user = Depends(get_current_user),
):
    """Get all available entity types."""
    return {
        "entity_types": [e.value for e in EntityType],
        "relation_types": [r.value for r in RelationType],
    }


# =============================================================================
# Entity Extraction Endpoints
# =============================================================================

class ExtractionResponse(BaseModel):
    """Response for entity extraction."""
    status: str
    document_id: Optional[str] = None
    entities_extracted: int = 0
    relations_extracted: int = 0
    message: Optional[str] = None


class BulkExtractionResponse(BaseModel):
    """Response for bulk entity extraction."""
    status: str
    message: str
    total_documents: int = 0


@router.post("/extract/{document_id}", response_model=ExtractionResponse)
async def extract_entities_from_document(
    document_id: str,
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """
    Trigger entity extraction for a specific document.

    This endpoint extracts named entities and relationships from the
    document's chunks and stores them in the knowledge graph.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )

    # Verify document exists
    result = await db.execute(
        select(Document).where(Document.id == doc_uuid)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        kg_service = await get_knowledge_graph_service(db)
        entity_count, relation_count = await kg_service.process_document_for_graph(doc_uuid)

        return ExtractionResponse(
            status="success",
            document_id=str(doc_uuid),
            entities_extracted=entity_count,
            relations_extracted=relation_count,
            message=f"Successfully extracted {entity_count} entities and {relation_count} relations",
        )

    except Exception as e:
        logger.error("Failed to extract entities from document", document_id=document_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract entities: {str(e)}"
        )


@router.post("/extract-all", response_model=BulkExtractionResponse)
async def extract_entities_from_all_documents(
    db: AsyncSession = Depends(get_async_session),
    _user = Depends(get_current_user),
):
    """
    Trigger entity extraction for all documents.

    This endpoint queues entity extraction for all documents that don't
    already have entities extracted. This is a synchronous operation
    that processes documents one by one.

    For large document collections, this may take a long time.
    """
    try:
        # Get all documents
        result = await db.execute(
            select(Document.id, Document.filename)
            .where(Document.processing_status == "completed")  # Only process completed documents
        )
        documents = result.all()

        if not documents:
            return BulkExtractionResponse(
                status="success",
                message="No documents found to process",
                total_documents=0,
            )

        total_entities = 0
        total_relations = 0
        processed_count = 0
        error_count = 0

        kg_service = await get_knowledge_graph_service(db)

        for doc_id, doc_filename in documents:
            try:
                entity_count, relation_count = await kg_service.process_document_for_graph(doc_id)
                total_entities += entity_count
                total_relations += relation_count
                processed_count += 1
                logger.info(
                    "Extracted entities from document",
                    document_id=str(doc_id),
                    document_name=doc_filename,
                    entities=entity_count,
                    relations=relation_count,
                )
            except Exception as e:
                error_count += 1
                logger.error(
                    "Failed to extract entities from document",
                    document_id=str(doc_id),
                    document_name=doc_filename,
                    error=str(e),
                )
                continue

        return BulkExtractionResponse(
            status="success" if error_count == 0 else "partial",
            message=f"Processed {processed_count}/{len(documents)} documents. "
                    f"Extracted {total_entities} entities and {total_relations} relations. "
                    f"{error_count} errors." if error_count > 0 else
                    f"Processed {processed_count} documents. Extracted {total_entities} entities and {total_relations} relations.",
            total_documents=processed_count,
        )

    except Exception as e:
        logger.error("Failed to run bulk entity extraction", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract entities: {str(e)}"
        )


# =============================================================================
# Graph Maintenance Endpoints
# =============================================================================

class CleanupResponse(BaseModel):
    """Response model for cleanup operations."""
    status: str
    message: str
    orphan_entities_removed: int = 0
    orphan_relations_removed: int = 0


@router.post(
    "/cleanup",
    response_model=CleanupResponse,
    summary="Clean up orphan entities",
    description="Remove orphan entities (no document mentions) and orphan relationships.",
)
async def cleanup_graph(
    dry_run: bool = Query(False, description="Preview changes without applying them"),
    db: AsyncSession = Depends(get_async_session),
    current_user: dict = Depends(get_current_user),
):
    """
    Clean up orphan entities and relationships from the knowledge graph.

    Orphan entities are those that no longer have any document mentions
    (e.g., after documents are deleted). This endpoint removes them
    to keep the graph clean and performant.
    """
    from sqlalchemy import delete, not_

    try:
        # Find orphan entities (no mentions)
        mentioned_entity_ids = select(EntityMention.entity_id).distinct()
        orphan_entities_result = await db.execute(
            select(Entity.id)
            .where(not_(Entity.id.in_(mentioned_entity_ids)))
        )
        orphan_entity_ids = [row[0] for row in orphan_entities_result.all()]

        orphan_entities_count = len(orphan_entity_ids)
        orphan_relations_count = 0

        if orphan_entity_ids:
            # Count relations to be deleted
            orphan_relations_result = await db.scalar(
                select(func.count(EntityRelation.id))
                .where(
                    EntityRelation.source_entity_id.in_(orphan_entity_ids) |
                    EntityRelation.target_entity_id.in_(orphan_entity_ids)
                )
            )
            orphan_relations_count = orphan_relations_result or 0

            if not dry_run:
                # Delete relationships involving orphan entities
                await db.execute(
                    delete(EntityRelation)
                    .where(
                        EntityRelation.source_entity_id.in_(orphan_entity_ids) |
                        EntityRelation.target_entity_id.in_(orphan_entity_ids)
                    )
                )

                # Delete orphan entities
                await db.execute(
                    delete(Entity).where(Entity.id.in_(orphan_entity_ids))
                )

                await db.commit()

        # Also clean up any remaining orphan relationships
        valid_entity_ids = select(Entity.id)
        additional_orphan_relations = await db.scalar(
            select(func.count(EntityRelation.id))
            .where(
                not_(
                    and_(
                        EntityRelation.source_entity_id.in_(valid_entity_ids),
                        EntityRelation.target_entity_id.in_(valid_entity_ids),
                    )
                )
            )
        )

        if additional_orphan_relations and additional_orphan_relations > 0:
            orphan_relations_count += additional_orphan_relations
            if not dry_run:
                await db.execute(
                    delete(EntityRelation)
                    .where(
                        not_(
                            and_(
                                EntityRelation.source_entity_id.in_(valid_entity_ids),
                                EntityRelation.target_entity_id.in_(valid_entity_ids),
                            )
                        )
                    )
                )
                await db.commit()

        if dry_run:
            return CleanupResponse(
                status="preview",
                message=f"Would remove {orphan_entities_count} orphan entities and {orphan_relations_count} orphan relationships.",
                orphan_entities_removed=orphan_entities_count,
                orphan_relations_removed=orphan_relations_count,
            )
        else:
            logger.info(
                "Knowledge graph cleanup complete",
                orphan_entities=orphan_entities_count,
                orphan_relations=orphan_relations_count,
            )
            return CleanupResponse(
                status="success",
                message=f"Removed {orphan_entities_count} orphan entities and {orphan_relations_count} orphan relationships.",
                orphan_entities_removed=orphan_entities_count,
                orphan_relations_removed=orphan_relations_count,
            )

    except Exception as e:
        logger.error("Failed to cleanup knowledge graph", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup graph: {str(e)}"
        )
