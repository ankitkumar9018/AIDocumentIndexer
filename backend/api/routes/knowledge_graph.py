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
from backend.api.middleware.auth import get_org_filter, AuthenticatedUser
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
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get knowledge graph statistics.

    Returns summary statistics about the knowledge graph including
    entity counts, relationship counts, and distributions.
    """
    try:
        # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
        org_id = get_org_filter(user)

        # Build base queries with org filtering
        entity_base = select(func.count(Entity.id))
        relation_base = select(func.count(EntityRelation.id))
        mention_base = select(func.count(EntityMention.id))

        if org_id and not user.is_superadmin:
            entity_base = entity_base.where(
                or_(
                    Entity.organization_id == org_id,
                    Entity.organization_id.is_(None),
                )
            )
            relation_base = relation_base.where(
                or_(
                    EntityRelation.organization_id == org_id,
                    EntityRelation.organization_id.is_(None),
                )
            )
            # EntityMention doesn't have org_id, filter via Document join
            mention_base = mention_base.select_from(
                EntityMention.__table__.join(
                    Document.__table__,
                    EntityMention.document_id == Document.id
                )
            ).where(
                or_(
                    Document.organization_id == org_id,
                    Document.organization_id.is_(None),
                )
            )

        # Get basic counts
        entity_count = await db.scalar(entity_base) or 0
        relation_count = await db.scalar(relation_base) or 0
        mention_count = await db.scalar(mention_base) or 0

        # Entity type distribution with org filtering
        entity_type_query = select(Entity.entity_type, func.count(Entity.id)).group_by(Entity.entity_type)
        if org_id and not user.is_superadmin:
            entity_type_query = entity_type_query.where(
                or_(
                    Entity.organization_id == org_id,
                    Entity.organization_id.is_(None),
                )
            )
        result = await db.execute(entity_type_query)
        entity_type_dist = {str(row[0].value): row[1] for row in result.all()}

        # Relation type distribution with org filtering
        relation_type_query = select(EntityRelation.relation_type, func.count(EntityRelation.id)).group_by(EntityRelation.relation_type)
        if org_id and not user.is_superadmin:
            relation_type_query = relation_type_query.where(
                or_(
                    EntityRelation.organization_id == org_id,
                    EntityRelation.organization_id.is_(None),
                )
            )
        result = await db.execute(relation_type_query)
        relation_type_dist = {str(row[0].value): row[1] for row in result.all()}

        # Top entities by mention count (subquery approach) with org filtering
        mention_counts = (
            select(
                EntityMention.entity_id,
                func.count(EntityMention.id).label("mention_count")
            )
            .group_by(EntityMention.entity_id)
            .subquery()
        )

        top_entities_query = (
            select(Entity, mention_counts.c.mention_count)
            .outerjoin(mention_counts, Entity.id == mention_counts.c.entity_id)
        )
        if org_id and not user.is_superadmin:
            top_entities_query = top_entities_query.where(
                or_(
                    Entity.organization_id == org_id,
                    Entity.organization_id.is_(None),
                )
            )
        top_entities_query = top_entities_query.order_by(desc(mention_counts.c.mention_count)).limit(10)

        result = await db.execute(top_entities_query)

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

        # Documents with entities (filtered by org)
        docs_with_entities_query = select(func.count(func.distinct(EntityMention.document_id)))
        if org_id and not user.is_superadmin:
            docs_with_entities_query = docs_with_entities_query.select_from(
                EntityMention.__table__.join(
                    Document.__table__,
                    EntityMention.document_id == Document.id
                )
            ).where(
                or_(
                    Document.organization_id == org_id,
                    Document.organization_id.is_(None),
                )
            )
        docs_with_entities = await db.scalar(docs_with_entities_query) or 0

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
    user: AuthenticatedUser,
    limit: int = Query(100, ge=1, le=500, description="Maximum nodes to return"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get graph data for visualization.

    Returns nodes (entities) and edges (relationships) formatted
    for use with visualization libraries like vis.js or d3.js.
    """
    try:
        # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
        org_id = get_org_filter(user)

        # Build entity query with org filtering
        entity_query = select(Entity)
        if entity_type:
            try:
                et = EntityType(entity_type)
                entity_query = entity_query.where(Entity.entity_type == et)
            except ValueError:
                pass

        # Apply organization filter
        if org_id and not user.is_superadmin:
            entity_query = entity_query.where(
                or_(
                    Entity.organization_id == org_id,
                    Entity.organization_id.is_(None),
                )
            )

        entity_query = entity_query.limit(limit)

        result = await db.execute(entity_query)
        entities = list(result.scalars().all())
        entity_ids = [e.id for e in entities]

        # Get relations between these entities (with org filtering)
        relations = []
        if entity_ids:
            relations_query = (
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
            if org_id and not user.is_superadmin:
                relations_query = relations_query.where(
                    or_(
                        EntityRelation.organization_id == org_id,
                        EntityRelation.organization_id.is_(None),
                    )
                )
            result = await db.execute(relations_query)
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
    user: AuthenticatedUser,
    query: Optional[str] = Query(None, description="Search query"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Search entities in the knowledge graph.

    Supports filtering by name/alias search and entity type.
    """
    try:
        # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
        org_id = get_org_filter(user)

        # Build base query
        base_query = select(Entity)
        count_query = select(func.count(Entity.id))

        conditions = []

        # Organization filter
        if org_id and not user.is_superadmin:
            conditions.append(
                or_(
                    Entity.organization_id == org_id,
                    Entity.organization_id.is_(None),
                )
            )

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
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Get a specific entity by ID."""
    try:
        entity_uuid = uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid entity ID format"
        )

    # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
    org_id = get_org_filter(user)
    entity_query = select(Entity).where(Entity.id == entity_uuid)
    if org_id and not user.is_superadmin:
        entity_query = entity_query.where(
            or_(
                Entity.organization_id == org_id,
                Entity.organization_id.is_(None),
            )
        )

    result = await db.execute(entity_query)
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
    user: AuthenticatedUser,
    max_hops: int = Query(2, ge=1, le=5, description="Maximum graph hops"),
    max_neighbors: int = Query(20, ge=1, le=50, description="Maximum neighbors"),
    db: AsyncSession = Depends(get_async_session),
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

    # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
    org_id = get_org_filter(user)

    # Get the entity with org filtering
    entity_query = select(Entity).where(Entity.id == entity_uuid)
    if org_id and not user.is_superadmin:
        entity_query = entity_query.where(
            or_(
                Entity.organization_id == org_id,
                Entity.organization_id.is_(None),
            )
        )

    result = await db.execute(entity_query)
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
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
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

    # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
    org_id = get_org_filter(user)

    # Get document with org filtering
    doc_query = select(Document).where(Document.id == doc_uuid)
    if org_id and not user.is_superadmin:
        doc_query = doc_query.where(
            or_(
                Document.organization_id == org_id,
                Document.organization_id.is_(None),
            )
        )

    result = await db.execute(doc_query)
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
    _user: AuthenticatedUser,
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
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
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

    # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
    org_id = get_org_filter(user)

    # Verify document exists with org filtering
    doc_query = select(Document).where(Document.id == doc_uuid)
    if org_id and not user.is_superadmin:
        doc_query = doc_query.where(
            or_(
                Document.organization_id == org_id,
                Document.organization_id.is_(None),
            )
        )

    result = await db.execute(doc_query)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        kg_service = await get_knowledge_graph_service(db)
        # process_document_for_graph returns Dict[str, int] with keys: entities, relations, mentions
        stats = await kg_service.process_document_for_graph(doc_uuid)
        entity_count = stats.get("entities", 0)
        relation_count = stats.get("relations", 0)

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
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Trigger entity extraction for all documents.

    This endpoint queues entity extraction for all documents that don't
    already have entities extracted. This is a synchronous operation
    that processes documents one by one.

    For large document collections, this may take a long time.
    """
    try:
        # PHASE 12 FIX: Apply organization filtering for multi-tenant isolation
        org_id = get_org_filter(user)

        # Get all documents with org filtering
        docs_query = (
            select(Document.id, Document.filename)
            .where(Document.processing_status == "completed")  # Only process completed documents
        )
        if org_id and not user.is_superadmin:
            docs_query = docs_query.where(
                or_(
                    Document.organization_id == org_id,
                    Document.organization_id.is_(None),
                )
            )

        result = await db.execute(docs_query)
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
                # process_document_for_graph returns Dict[str, int] with keys: entities, relations, mentions
                stats = await kg_service.process_document_for_graph(doc_id)
                entity_count = stats.get("entities", 0)
                relation_count = stats.get("relations", 0)
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
    _user: AuthenticatedUser,
    dry_run: bool = Query(False, description="Preview changes without applying them"),
    db: AsyncSession = Depends(get_async_session),
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


# =============================================================================
# Extraction Job Endpoints
# =============================================================================

class StartExtractionJobRequest(BaseModel):
    """Request to start an extraction job."""
    only_new_documents: bool = True
    document_ids: Optional[List[str]] = None
    provider_id: Optional[str] = None  # Optional LLM provider override for extraction


class ExtractionJobResponse(BaseModel):
    """Response for extraction job operations."""
    job_id: str
    status: str
    message: str


class ExtractionJobProgressResponse(BaseModel):
    """Progress information for an extraction job."""
    job_id: str
    status: str
    progress_percent: float
    processed_documents: int
    total_documents: int
    failed_documents: int
    total_entities: int
    total_relations: int
    current_document: Optional[str] = None
    current_document_id: Optional[str] = None
    estimated_remaining_seconds: Optional[float] = None
    avg_doc_processing_time: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    can_cancel: bool = False
    can_pause: bool = False
    can_resume: bool = False
    error_count: int = 0
    only_new_documents: bool = True


class ExtractionJobSummaryResponse(BaseModel):
    """Summary for listing extraction jobs."""
    job_id: str
    status: str
    progress_percent: float
    processed_documents: int
    total_documents: int
    created_at: str
    completed_at: Optional[str] = None


class PendingExtractionResponse(BaseModel):
    """Response for pending extraction count."""
    pending_count: int
    has_running_job: bool
    running_job_id: Optional[str] = None


@router.post("/extraction-jobs", response_model=ExtractionJobResponse)
async def start_extraction_job(
    request: StartExtractionJobRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Start a background extraction job for knowledge graph entities.

    The job runs in the background, allowing the user to navigate away
    and return to check progress later.
    """
    from fastapi import BackgroundTasks
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)

        # Check for existing running job
        org_id = uuid.UUID(user.organization_id) if user.organization_id else None
        existing = await service.get_running_job(org_id)
        if existing:
            return ExtractionJobResponse(
                job_id=str(existing.id),
                status="already_running",
                message="An extraction job is already running. Check its progress or cancel it first.",
            )

        # Create new job
        job = await service.create_job(
            user_id=uuid.UUID(user.user_id),
            organization_id=org_id,
            only_new_documents=request.only_new_documents,
            document_ids=request.document_ids,
            provider_id=request.provider_id,
        )

        # Start the job in background
        await service.start_job(job.id)

        logger.info(
            "Started KG extraction job",
            job_id=str(job.id),
            user_id=user.user_id,
            only_new=request.only_new_documents,
        )

        return ExtractionJobResponse(
            job_id=str(job.id),
            status="started",
            message="Extraction job started successfully. You can navigate away and check progress later.",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to start extraction job", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start extraction job: {str(e)}",
        )


@router.get("/extraction-jobs/current", response_model=Optional[ExtractionJobProgressResponse])
async def get_current_extraction_job(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Get the currently running extraction job for the user's organization."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        org_id = uuid.UUID(user.organization_id) if user.organization_id else None
        job = await service.get_running_job(org_id)

        if not job:
            return None

        progress = await service.get_progress(job.id)
        return ExtractionJobProgressResponse(**progress)

    except Exception as e:
        logger.error("Failed to get current extraction job", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get extraction job: {str(e)}",
        )


@router.get("/extraction-jobs/pending", response_model=PendingExtractionResponse)
async def get_pending_extraction_count(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Get count of documents pending extraction and running job status."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        org_id = uuid.UUID(user.organization_id) if user.organization_id else None

        pending_count = await service.get_documents_pending_extraction(org_id)
        running_job = await service.get_running_job(org_id)

        return PendingExtractionResponse(
            pending_count=pending_count,
            has_running_job=running_job is not None,
            running_job_id=str(running_job.id) if running_job else None,
        )

    except Exception as e:
        logger.error("Failed to get pending extraction count", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pending count: {str(e)}",
        )


@router.get("/extraction-jobs/{job_id}", response_model=ExtractionJobProgressResponse)
async def get_extraction_job_progress(
    job_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Get progress of a specific extraction job."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        progress = await service.get_progress(uuid.UUID(job_id))

        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Extraction job not found",
            )

        return ExtractionJobProgressResponse(**progress)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get extraction job progress", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job progress: {str(e)}",
        )


@router.post("/extraction-jobs/{job_id}/cancel", response_model=ExtractionJobResponse)
async def cancel_extraction_job(
    job_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Cancel a running extraction job."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        cancelled = await service.cancel_job(uuid.UUID(job_id))

        if cancelled:
            return ExtractionJobResponse(
                job_id=job_id,
                status="cancelled",
                message="Extraction job cancelled successfully.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be cancelled (not running or already completed)",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel extraction job", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}",
        )


@router.post("/extraction-jobs/{job_id}/pause", response_model=ExtractionJobResponse)
async def pause_extraction_job(
    job_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Pause a running extraction job."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        paused = await service.pause_job(uuid.UUID(job_id))

        if paused:
            return ExtractionJobResponse(
                job_id=job_id,
                status="paused",
                message="Extraction job paused. You can resume it later.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be paused (not running)",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pause extraction job", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause job: {str(e)}",
        )


@router.post("/extraction-jobs/{job_id}/resume", response_model=ExtractionJobResponse)
async def resume_extraction_job(
    job_id: str,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
):
    """Resume a paused extraction job."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        resumed = await service.resume_job(uuid.UUID(job_id))

        if resumed:
            return ExtractionJobResponse(
                job_id=job_id,
                status="resumed",
                message="Extraction job resumed.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be resumed (not paused)",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume extraction job", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume job: {str(e)}",
        )


@router.get("/extraction-jobs", response_model=List[ExtractionJobSummaryResponse])
async def list_extraction_jobs(
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_async_session),
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(20, le=100),
):
    """List extraction jobs for the user's organization."""
    from backend.services.kg_extraction_job import KGExtractionJobService

    try:
        service = KGExtractionJobService(db)
        org_id = uuid.UUID(user.organization_id) if user.organization_id else None

        jobs = await service.list_jobs(
            organization_id=org_id,
            status=status_filter,
            limit=limit,
        )

        return [
            ExtractionJobSummaryResponse(
                job_id=str(job.id),
                status=job.status,
                progress_percent=job.get_progress_percent(),
                processed_documents=job.processed_documents,
                total_documents=job.total_documents,
                created_at=job.created_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
            )
            for job in jobs
        ]

    except Exception as e:
        logger.error("Failed to list extraction jobs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}",
        )
